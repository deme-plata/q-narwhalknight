# Crown & Ash Action Tax → LP Revenue Share (Design v1)

**Status:** Design proposal, not implemented. Targets v10.11.16+, after the
chain-wide apply-pipeline bug is fixed (v10.11.15 / .14 in flight).

**Premise:** Operator asked "is there an LP so we can earn from others
gaming?" Currently no — Crown & Ash's `treasury` field is FixedPoint
internal state with zero bridge to on-chain QUG. This doc proposes the
minimum bridge that turns "AI agents play C&A" into "everyone in the LP
earns from agents playing C&A," and also into a primitive that scales
gracefully as agent intelligence increases (Claude 5, GPT-6, Grok 5+, …).

---

## The mechanism

Every Crown & Ash `GameAction` submitted as an on-chain tx pays a small
**action tax** in QUG (currently sims are free of QUG cost). The tax
splits three ways:

```
  Action tax = MIN_ACTION_FEE × multiplier(action_class)

  ┌──────────────────────────────────────────────────────────────┐
  │  50% → BURN                          chain monetary policy   │
  │  30% → crown-ash-lp-rev-share pool   LPs earn from activity  │
  │  20% → operator dev-fee wallet       infra maintenance       │
  └──────────────────────────────────────────────────────────────┘
```

The LP pool is a **passive QUG accumulator**. LPs stake QUG → they earn
their share of inflows proportional to stake-weight + time-in-pool.

LPs hold an LP-CA-REV token that's redeemable for `(stake + accumulated
rewards)` at any time. The token itself trades on the DEX — speculators
can buy LP-CA-REV expecting Crown & Ash activity to spike (e.g., right
before a tournament season opens).

---

## Action tax schedule (proposal)

| Action | Multiplier | Rationale |
|---|---|---|
| `RaiseArmy` | 1× | base — cheap, common |
| `MoveArmy*` | 1× | base |
| `BuildImprovement` | 2× | persistent state change |
| `ProposeTreaty` | 1× | spam-resistant if treaties cost something |
| `AcceptTreaty` | 0.5× | cheaper to encourage acceptance |
| `DeclareWar` | 5× | high stakes, high cost — anti-spam |
| `ConvertProvince` | 3× | religion shifts have long tail effects |
| `LaunchPlot` | 4× | intrigue is expensive |
| `ArrangeMarriage` | 2× | dynastic actions |
| `EstablishTradeRoute` | 2× | economic infrastructure |

`MIN_ACTION_FEE` = `MIN_TRANSACTION_FEE` (21,000 raw u128 units in the
current chain — see q-types::MIN_TRANSACTION_FEE constant). At ~$3K per
QUG pool-implied, 1× fee ≈ $0.063. 5× ≈ $0.31. Cheap enough that an AI
playing actively doesn't go broke; expensive enough that spam wars cost
real money.

---

## Why this matches the user's "earn from others gaming" intent

- **Pure passive income.** You stake QUG into the pool; you never need
  to play Crown & Ash. As long as other agents are active, you earn.
- **Pro-rata, anti-whale.** Stake weight is sqrt(stake) not linear, so a
  whale provider doesn't crowd out small LPs. (See `q-dex::sqrt_weight`
  pattern already used for QSHARE staking.)
- **Aligns incentives toward game depth.** LPs WANT agents to play more.
  If the game is shallow, agents play less, LPs earn less. Operator +
  LPs both want the game's depth to grow. This is why the design
  principle in `quillon://opportunities` ("make depth scale with agent
  intelligence") matters — it's not just nice design, it's revenue.
- **Composable with other AI economic primitives.** Faction-share tokens
  (FACTION-SALT etc.) and war prediction pools layer on top. LP-CA-REV
  is the foundation primitive.

---

## Implementation surface

### Rust side (q-narwhalknight crates)

1. `crates/q-types/src/lib.rs` — add `MIN_CROWN_ASH_ACTION_FEE` constant.
2. `crates/q-api-server/src/handlers.rs` — the Crown & Ash action handler
   (find via `grep crown_ash_submit_action`) deducts the tax from the
   submitting wallet BEFORE queueing the action. If the wallet can't pay,
   return 402 PAYMENT REQUIRED with a clear error.
3. New crate `q-crown-ash-lp` (or extend `q-dex`):
   - Pool struct: `{ total_stake, total_rewards_accumulated, lp_shares: HashMap<wallet, share_count>, last_activity_block }`
   - On every action-tax inflow, increment `total_rewards_accumulated`.
   - Stake: lock QUG, receive shares. Share count = sqrt(stake_amount).
   - Redeem: burn shares, receive `(stake + earned)` proportional to
     share-of-total.
4. `crates/q-crown-ash-types` — emit a `CrownAshActionTaxed` event on
   every action so explorer + LP UI can render.

### MCP side

- New tool `crown_ash_lp_stake` — stake QUG into the LP pool.
- New tool `crown_ash_lp_redeem` — burn shares, claim rewards.
- New tool `crown_ash_lp_stats` — pool TVL, your share %, your earned QUG,
  recent action-tax inflows, projected yield.

### Frontend side

- A `CrownAshLPPanel` component in the wallet UI showing:
  - Pool TVL
  - Your stake + share %
  - Your accumulated rewards (claim button)
  - Recent action-tax inflows graph (last 24h)
  - Projected APY (extrapolate from 7-day activity)

---

## Why this is gated on v10.11.15 (apply-pipeline fix)

Right now, Crown & Ash relational actions (DeclareWar, ProposeTreaty,
RaiseArmy) queue but never resolve — same chain-wide tx-drop bug we're
chasing in v10.11.13/.14 instrumentation. If we ship the action-tax
collection BEFORE the apply path works, agents pay the tax but actions
still don't fire — pure money-burn, no game effect, terrible UX.

Once v10.11.15 lands the apply fix and we verify with the docker-test
skill, the action-tax flow becomes safe to ship.

---

## Sketch of agent yield math

Assume:
- 10 active AI agents (Claude, Codex, Cursor, Grok, 6 more)
- Each submits ~30 actions per game day (mix of cheap + expensive)
- Avg action tax = ~2× MIN = 42,000 raw u128 per action
- Total daily action-tax inflow = 10 × 30 × 42,000 raw = 12,600,000 raw / day
- 30% to LP pool = 3,780,000 raw / day

For a pool with 1,000 QUG TVL:
- 3,780,000 raw = 0.00378 QUG / day shared across LPs
- Annualized: ~1.38 QUG / year per 1,000 QUG staked = **0.138% APY**

Looks tiny — because action volumes are small early. The yield scales
linearly with player count + activity. Ten agents → 1.38 QUG/yr. A
hundred agents → 13.8 QUG/yr. A thousand → 138 QUG/yr per kQUG staked.

**This is why "design for room to grow" matters in the resource doc.**
If the game is rich enough that smarter agents play MORE and submit
MORE actions, yield compounds. If the game is shallow and agents quit
quickly, yield dries up. The LP design is the operator's economic
incentive to keep the game deep.

---

## Anti-spam considerations

- Min `RaiseArmy` cap per turn per faction (already in sim — verify it's enforced).
- Geometric cost increase for repeated `DeclareWar` against the same target within 100 turns (anti-grinding).
- Tx-mempool fee floor remains (no zero-fee crown-ash actions).
- LP pool capped (no infinite-stake whale capture): max 10% of supply staked at any time.

---

## Test plan

1. Once v10.11.15 lands, spin up a Docker test container with 3 mock AI agents (scripted action submitters).
2. Each agent submits 100 actions of varied class. Verify:
   - Wallet balance debited by the correct tax amount per action.
   - LP pool accumulator grows by 30% × tax.
   - Burn counter grows by 50% × tax.
   - Operator wallet gains 20% × tax.
3. Stake 100 QUG from a 4th wallet that doesn't play. Wait 1000 turns.
4. Redeem; check earned QUG matches the formula `(your_shares / total_shares) × pool_accumulator`.

---

## What this design doesn't do (deliberately)

- **No oracles.** Resolution is on-chain via Crown & Ash game state.
- **No off-chain bots.** Agents submit signed actions; the tax is
  collected atomically as part of the tx-apply.
- **No upgradeable contracts.** This is a Rust-native handler change,
  not a smart-contract deployment. Changes ship via q-api-server binary
  bump (v10.11.16+).
- **No human-only entry tier.** Humans + AIs use the same pool. No
  segregated markets.

---

## Open questions for the operator

1. Should the operator's 20% slice be optional / configurable? Some
   operators may want 0% (donate to LPs) for maximum LP yield.
2. Should the LP pool be one global pool, or per-faction? Per-faction
   means LPs bet on a SPECIFIC faction's activity — more interesting
   strategy, more complex book-keeping.
3. Should staked QUG be locked for a minimum period (anti-flash-LP)?
   E.g., 1000 turns minimum lockup before redemption.
4. Should the operator publish a "season cap" on total tax inflow per
   season? Caps prevent runaway action-spam farming.

---

## Why now (vs later)

The user pivoted today from "should we reset mainnet" to "fix in place
and stand by what we shipped." That's the moment to design forward, not
just bug-fix. The LP revenue-share is the natural next-generation
mechanic that turns the apply-pipeline fix into something more than
just "things work again" — it adds NEW economic surface that didn't
exist before, anchored on the agent-versus-agent gameplay we already
have working.

Ship order:
1. v10.11.15 — fix the apply pipeline (in flight)
2. v10.11.16 — bridge Crown & Ash treasury to on-chain QUG (separate doc)
3. v10.11.17 — action tax + LP revenue share (this doc)
4. v10.12.x — faction-share tokens + war prediction pools + plot insurance
   (deferred until the foundation is proven)

Each shipped version validates the next; no big-bang deploy.
