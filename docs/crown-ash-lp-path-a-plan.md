# Crown & Ash LP Bridge — Path A Implementation Plan

**Status:** Plan, ready to execute. Gated only on v10.11.15 (the apply-pipeline
fix) landing first. Once that's verified, this is ~1 week of focused work.

**Premise:** Path A is the 80/20 bridge between Crown & Ash and the existing
DEX/LP machinery. Every C&A action pays a small QUG tax → the tax accumulates
in a per-season reserve → at season close the reserve deepens the pool's QUG
side → LPs holding the season's index token (LP-CA-SEASON-N) earn the spread
when they redeem.

No new crates, no new consensus rules. Pure composition of existing DEX +
new tax handler in `q-api-server`.

---

## Files touched (concrete)

| File | Change | LOC |
|---|---|---|
| `crates/q-types/src/lib.rs` | Add `MIN_CROWN_ASH_ACTION_FEE_RAW` constant + `CrownAshActionTaxed` event variant | ~20 |
| `crates/q-api-server/src/handlers.rs` | New handler `crown_ash_submit_action_with_tax` wrapping the existing C&A action handler. Reads tax schedule, deducts from signer wallet, increments season-reserve key | ~120 |
| `crates/q-storage/src/lib.rs` | New CF_MANIFEST keys: `cna:tax:reserve:<season>` (u128 LE) and `cna:lp:season:<season>` (pool-id) | ~30 |
| `crates/q-api-server/src/main.rs` | Wire the new handler into the existing C&A route (replace direct sim-action call with tax-wrapped call) | ~10 |
| `tools/quillon-wallet-mcp/src/index.ts` | 2 new tools: `crown_ash_lp_deposit`, `crown_ash_lp_status` | ~80 |
| `docs/crown-ash-lp-revenue-share-v1.md` | Mark as implemented | minor |
| `CLAUDE.md` | Note the action-tax flow in the C&A section | minor |

Total: ~260 LOC + tests.

---

## Day-by-day breakdown

### Day 1 — Tax constants + schedule

- Add `MIN_CROWN_ASH_ACTION_FEE_RAW = 21_000_u128` to `q-types` (matches existing tx fee floor)
- Add a const `ACTION_TAX_MULTIPLIER[action_class]` table — see schedule in `crown-ash-lp-revenue-share-v1.md`:
  - `RaiseArmy` 1×, `BuildImprovement` 2×, `DeclareWar` 5×, `LaunchPlot` 4×, etc.
- Unit tests for the multiplier table

### Day 2 — Storage keys + season management

- Add CF_MANIFEST keys:
  - `cna:current_season` → u32 LE (incremented at season-open)
  - `cna:tax:reserve:<season>` → u128 LE (accumulates across the season)
  - `cna:tax:total_burned` → u128 LE (running total of the 50% burn slice)
- Helper functions `read_current_season`, `bump_season`, `add_to_season_reserve`
- Unit tests with a temporary RocksDB

### Day 3 — Action-tax handler

- Modify `crown_ash_submit_action` to:
  1. Parse the action, compute `tax = MIN_FEE × multiplier(action_class)`
  2. Atomically subtract `tax` from the signer's QUG balance (use the v10.11.5 atomic-debit primitive)
  3. Split: 50% burn (subtract from minted-supply counter), 30% to `cna:tax:reserve:<current_season>`, 20% to operator dev-fee wallet
  4. Submit the underlying C&A action (existing path)
- If wallet balance < tax: return `402 PAYMENT REQUIRED` with a clear error
- Emit `CrownAshActionTaxed` event for explorer + LP UI

### Day 4 — LP pool initialization

- At first season-open, create a new DEX pool `QUG/LP-CA-SEASON-1`
  - Operator seeds with a small balanced amount (e.g., 100 QUG + initial LP-CA-SEASON-1 tokens)
- Add the season-close hook: read `cna:tax:reserve:<season>`, transfer 100% of that reserve into the pool's QUG side via the existing `add_liquidity_qug_only` path (single-sided deposit)
- This rebalances the pool: LP-CA-SEASON-1 holders now redeem against a richer pool
- New CF key `cna:lp:season:<season>` → pool ID (for stable lookup at close time)

### Day 5 — MCP tools

- `crown_ash_lp_deposit { amount_qug, season?: number }` — wraps `add_liquidity` for the current season's pool
- `crown_ash_lp_status { season?: number, wallet?: string }` — reports current pool TVL, the season's accumulated tax reserve, the caller's LP token holdings, projected APY (extrapolated from 7-day inflow rate)
- Both tools are signed (X-Wallet-Auth)

### Day 6 — Frontend stub

- Add a `<CrownAshLPCard />` component in `gui/quantum-wallet/src/components/`
- Renders: current season number, pool TVL, your stake, accumulated rewards, "Deposit" button
- Wire into the wallet UI's main grid (right of the DEX Screen card)
- Don't ship the dashboard at this stage — just the card. Dashboard is for v10.12.

### Day 7 — End-to-end test + roll-out

- Spin up a Docker test container via `quillon-docker-test`
- Mock 3 agent wallets, each submits 100 mixed-class actions over 24 simulated hours
- Verify:
  - Each agent's QUG balance debited correctly (sum of taxes paid)
  - Burn counter incremented by sum of 50% slices
  - Operator wallet credited by sum of 20% slices
  - Season reserve key holds sum of 30% slices
- Force-close the season; verify pool QUG side jumps by the reserve amount
- LP a 4th wallet that didn't play; redeem after close; verify the proceeds match the formula `share × (final_qug_reserve / total_LP_supply)`
- Then ship via the standard ha-deploy.sh full pipeline (`./scripts/ha-deploy.sh full -y`)

---

## Anti-spam / edge cases

- **Insufficient balance**: tx rejected with `402 PAYMENT REQUIRED`, no game-state mutation
- **Whale capture of LP**: max 25% of LP-CA-SEASON-N issuance per wallet (enforced at `add_liquidity` time)
- **Operator runaway**: 20% dev-fee slice configurable; can be set to 0 for a community-only season
- **Action-spam farming**: per-faction rate limit (max 50 actions/turn) — already in sim; verify on-chain enforcement
- **Season rollover during in-flight tx**: tax-pays-into-old-season at submission time, not at apply time — keeps accounting clean

---

## Why this is "Path A" not "the real one"

Path A reuses the DEX add_liquidity primitive verbatim. The single-sided
QUG deposit at season close is mathematically equivalent to a swap that
favors LP holders (constant-product math: more QUG, same LP token supply
→ each LP token redeems for more underlying value when burned).

That works for v1 but has limitations:

- **No per-faction risk slicing.** Everyone in the pool benefits from
  any season's activity equally. You can't bet "Salt League will play
  more than Black Abbey this season."
- **No time-weighted shares.** Someone who LPs on day 6 gets the same
  pro-rata as someone who LPed on day 1.
- **No prize escrow.** Operator's dev-fee slice doesn't get earmarked
  as tournament prize — that requires a separate escrow account.

Path B (the proper `q-crown-ash-lp` crate) addresses all three but is
~5× the work. Ship Path A first to validate, then Path B as the
operator measures actual usage.

---

## Verification: how to know it's working

After deploy, an LP holder running:

```
mcp__quillon-wallet__crown_ash_lp_status
```

should see something like:

```
Season 1 (open 2026-06-01 → 2026-06-08)
  Pool: pool-qug-lp-ca-season-1
  Pool TVL:        1,247.30 QUG
  Tax reserve:     12.45 QUG (will merge at close)
  Your LP balance: 50 LP-CA-SEASON-1 (4.01% of issuance)
  Your projected redeem if season closed now: 50.50 QUG
  Recent 7d action-tax inflow rate: 1.78 QUG/day
  Projected APY: 13.0%
```

A non-LP holder running the same tool sees the pool stats but `Your LP
balance: 0` and `Your projected redeem: 0`.

---

## Out of scope for Path A

These are all valid features but deferred to Path B (`v10.12.x+`):

- Per-faction LP pools (FACTION-SALT, FACTION-CROWN, etc.)
- War prediction pools
- Plot insurance pools
- Inter-agent escrow
- Faction governance DAOs
- Aesthetic sigil NFTs
- AI-vs-AI market-making competitions
- Multi-AI mining cooperatives

Stubs for all of these live in `crates/q-crown-ash-lp/src/stubs/` (created
in this commit) and in the MCP as tools that return "stub — coming after
Path A validation completes."

---

## Failure modes to monitor after deploy

1. **Action volume crashes**: if tax > marginal value of an action, agents
   stop playing. Watch the per-day action-count metric; if it drops >50%
   from pre-tax baseline, lower the multiplier.
2. **LP TVL stagnates**: if the projected APY isn't compelling, nobody
   stakes. Raise the operator's 20% slice cut (donate some back to LPs)
   or accept that early seasons have low TVL.
3. **Whale captures LP**: cap per-wallet at 25% (enforced); if a wallet
   tries to LP more, the excess gets refunded with a clear error.
4. **Season boundary race**: tax submitted at 23:59:59 lands in old
   season's reserve; verified via tx timestamp at handler time, not at
   apply time.

---

## Operator decision before kicking off

The implementation here assumes:
- Operator's dev-fee slice = 20%
- Burn slice = 50%
- LP slice = 30%
- Min tax = MIN_TRANSACTION_FEE (21,000 raw)
- Multipliers per the schedule in `crown-ash-lp-revenue-share-v1.md`

If you want a community-only run (no operator slice), set dev-fee = 0
and LP = 50%. The handler reads these from a `q-types::Constants` table
so they're tunable at compile time per season if desired.

Ready to implement when v10.11.15 is confirmed and signed off.
