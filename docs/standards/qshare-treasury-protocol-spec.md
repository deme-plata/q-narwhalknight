# QSHARE-1: Quillon Treasury Share Token Protocol

| Field | Value |
|---|---|
| **Standard number** | QSHARE-1 |
| **Title** | Autonomous premium-arbitrage treasury share token, layered above QCREDIT yield vaults |
| **Status** | Draft |
| **Type** | Standards Track — Application Protocol |
| **Created** | 2026-05-19 |
| **License** | Apache-2.0 |
| **Authors** | Claude Opus 4.7 (Anthropic), Viktor Sandstrøm Kristensen (Quillon Foundation) |
| **Version** | 1.0-draft |
| **Companion** | AFL-1 (`docs/standards/afl-1-protocol-spec.md`), QCREDIT vault (`crates/q-vm/src/contracts/qcredit_vault.rs`) |

> **Note on hosting**: as with AFL-1, this document is intended to live in a vendor-neutral repository once initial drafting stabilises. The current `docs/standards/` location is provisional.

---

## Abstract

This document defines QSHARE-1, an autonomous on-chain implementation of the equity-swap accumulation strategy popularised by MicroStrategy / Strategy (Michael Saylor) but executed as a smart contract rather than corporate action. QSHARE is the L3 layer of Quillon Graph's three-layer capital stack — L1 digital capital (QUG) → L2 digital credit (QCREDIT yield vaults) → L3 treasury share (QSHARE) — completing a structural mirror of Strategy's stock-bond-bitcoin design but with three key differences: (a) autonomous execution by smart contract, (b) on-chain observable NAV and market price, (c) AI-agent-callable via the AFL-1 protocol and accompanying MCP tools.

QSHARE-1 is licensed under Apache-2.0. Reference implementation lives in Q-NarwhalKnight; any chain with comparable credit-vault primitives may implement it.

---

## 1. Motivation

### 1.1 The Saylor premium-arbitrage loop

Strategy's accumulation of Bitcoin since 2020 has not been driven by market-timing. It has been driven by a structural reflexivity: when BTC rises, MSTR's stock rises *more* than BTC (because the market prices in expected future accumulation), creating a premium of market-price over NAV. During premium windows, Strategy issues new equity (ATM offerings, convertibles) and exchanges the proceeds for additional BTC. The result: BTC-per-share rises over time without market-timing.

The two essential ingredients are:

1. A tradeable instrument whose market price reflexively rises faster than its underlying
2. A mechanism for issuing more of that instrument when its price is above NAV

Both ingredients are missing from chain-native treasury designs today. Tokens that *hold* an underlying asset (DAI, sETH, MSTR-style structures on other chains) typically peg to or under-track the NAV; they do not autonomously exploit overvaluation.

### 1.2 QCREDIT is L2, not L3

QCREDIT, defined in `crates/q-vm/src/contracts/qcredit_vault.rs`, is explicitly the L2 layer of a 3-layer stack inspired by Strategy. From the source comment:

```
//! Digital credit layer inspired by Strategy's 3-layer stack:
//!   L1 digital capital (QUG) → L2 digital credit (QCREDIT) → L3 products
```

QCREDIT is a yield-bearing wrapper (lock QUG, mint QCREDIT, redeem after tier lock with yield). It is structurally analogous to Strategy's convertible bonds and preferred shares — credit instruments that pay yield but are NOT the equity that trades at premium.

The L3 product layer is currently unbuilt. QSHARE-1 specifies it.

### 1.3 Why this matters for the agent-native chain

Once QSHARE exists, AI agents can run the accumulation loop autonomously. There is no human CEO discretion — the strategy executes when on-chain NAV/market-price ratio crosses the threshold, period. This is a substantial improvement over Strategy's design (where Saylor himself is the human-in-the-loop allocator), because:

- The strategy is reproducible across chains, jurisdictions, and management successions
- The decisions are auditable on-chain after the fact
- The strategy is AI-agent-callable via AFL-1 + MCP — agents can both trigger mint events and observe premium ratios
- The combination of L2 yield (5-25% APY) AND L3 premium accumulation gives token holders dual exposure

QSHARE is what Strategy would build if Strategy were a smart contract.

---

## 2. Architecture

### 2.1 Layer stack

```
L3 │  QSHARE      ← market-priced treasury share, mints on premium
   │              ←   reflexive accumulation: BTC-per-share equivalent
   │              ←   for Quillon Graph
   │
L2 │  QCREDIT     ← yield-bearing vault (5%/10%/15%/25% APY tiers)
   │              ← Bronze 7d, Silver 30d, Gold 90d, Platinum 180d
   │
L1 │  QUG         ← native digital capital, proof-of-work mined
```

### 2.2 QSHARE state

A single global smart contract maintains:

| Field | Type | Description |
|---|---|---|
| `treasury_qcredit_basket` | `HashMap<CreditTier, u128>` | QCREDIT holdings split by tier (in raw u128 units, decimals=24) |
| `treasury_pending_qug` | `u128` | QUG awaiting deposit into a new QCREDIT position |
| `circulating_qshare` | `u128` | Total QSHARE supply (raw units, decimals=24) |
| `last_mint_height` | `u64` | Block height of most recent autonomous mint |
| `mint_cooldown_blocks` | `u64` | Min blocks between mint events (default 360 ≈ 6 min) |
| `nav_oracle_state` | `NavOracleState` | Cached NAV calculation + refresh metadata |

### 2.3 NAV computation

NAV per QSHARE is computed deterministically on-chain:

```
nav_total_qug_equivalent =
    treasury_pending_qug
    + sum over (tier, amount) in treasury_qcredit_basket:
        amount × (1 + accrued_yield_at(tier, time_in_position))

nav_per_qshare = nav_total_qug_equivalent / circulating_qshare
```

`accrued_yield_at(tier, time)` is computed using the QCREDIT vault's APY formula (`crates/q-vm/src/contracts/qcredit_vault.rs::CreditTier::apy_bps`). Yield accrues continuously between block heights using linear interpolation per the vault's compounding rules.

Each tier's accrued yield is tracked separately; positions cannot be partially withdrawn, only redeemed in full at tier expiry.

### 2.4 Market price observation

QSHARE's market price is read from the on-chain DEX pool QSHARE/QUG. The contract uses the time-weighted average price (TWAP) over the last 30 blocks to resist single-block manipulation. If no QSHARE/QUG pool exists yet, NAV is used as a floor and minting is paused.

### 2.5 Premium ratio

```
premium_ratio = market_price_twap_30_blocks / nav_per_qshare
```

Mint conditions and buyback conditions branch off this ratio.

---

## 3. Mint mechanism

### 3.1 Autonomous mint trigger

QSHARE mints new shares when ALL of the following hold:

1. `premium_ratio > MINT_THRESHOLD` (default 1.5; configurable on-chain via governance)
2. `current_height - last_mint_height >= mint_cooldown_blocks` (default 360 blocks)
3. `dex_pool_depth(QSHARE/QUG) >= MIN_POOL_DEPTH` (default 1000 QUG; prevents thin-pool manipulation)
4. Caller has paid the mint trigger fee (default 0.01 QUG, returned + bounty if mint succeeds)

The mint trigger is *permissionless* — any address can call `try_autonomous_mint` and receive a bounty for successfully triggering a profitable mint. This includes AI agents acting via AFL-1.

### 3.2 Mint sizing

The amount of QSHARE minted per trigger is capped by both pool depth and treasury liquidity:

```
max_mint_amount =
    min(
        dex_pool_qug_reserves * MAX_POOL_FRACTION,    // default 0.5% of pool depth
        circulating_qshare * MAX_INFLATION_PER_MINT   // default 2% of circulating
    )
```

The 0.5% pool-fraction cap is the key safety property: the strategy cannot self-bid its own QSHARE price above realistic market levels by minting too much at once.

### 3.3 Mint execution

When triggered, the contract:

1. Mints `max_mint_amount` of QSHARE to itself
2. Atomically swaps the new QSHARE → QUG via the DEX pool (typically at premium price)
3. Deposits the resulting QUG into the highest-APY available QCREDIT tier (Platinum / 180-day / 25% APY by default)
4. Updates `treasury_qcredit_basket` and `circulating_qshare` accordingly
5. Pays the bounty to the trigger caller (default 0.5% of the QUG accumulation, capped at 1 QUG per trigger)
6. Emits `QShareMint` event with mint amount, premium ratio at trigger, and QUG accumulated

After the mint:

```
new_nav_per_qshare = (old_treasury + qug_accumulated) / (circulating + minted)
```

If the premium was greater than 1.0 (which it must have been to trigger), `new_nav_per_qshare > old_nav_per_qshare`. The token holder is strictly better off — their share now backs more QUG.

### 3.4 Buyback mechanism (the symmetric path)

When `premium_ratio < DISCOUNT_THRESHOLD` (default 0.95), the contract may exercise a buyback:

1. Withdraw a small amount of yield from the QCREDIT basket (limited to accrued yield; never principal)
2. Use the QUG to buy back QSHARE on the DEX
3. Burn the bought-back QSHARE

This produces a NAV-positive feedback: discount situations are partially absorbed by buybacks. Buyback caps are tighter than mint caps (max 0.1% of pool depth per trigger) to prevent gaming.

---

## 4. Smart contract interface

### 4.1 Public functions

```rust
// Anyone can call. Returns the bounty if mint succeeds.
pub fn try_autonomous_mint(
    caller: Address,
    trigger_fee_paid: u128,
) -> Result<MintResult, MintError>;

// Anyone can call. Returns the bounty if buyback succeeds.
pub fn try_buyback(
    caller: Address,
    trigger_fee_paid: u128,
) -> Result<BuybackResult, BuybackError>;

// Read-only.
pub fn nav_per_qshare() -> u128;
pub fn market_price_twap() -> u128;
pub fn premium_ratio() -> f64;
pub fn next_mint_eligible_at_height() -> u64;
pub fn treasury_composition() -> TreasuryComposition;
```

### 4.2 Events

```rust
QShareMint {
    minted: u128,
    qug_accumulated: u128,
    new_nav_per_qshare: u128,
    premium_ratio_at_trigger: u64, // ×1000 for integer encoding
    trigger_caller: Address,
    bounty_paid: u128,
    block_height: u64,
}

QShareBuyback {
    burned: u128,
    qug_spent: u128,
    new_nav_per_qshare: u128,
    discount_ratio_at_trigger: u64,
    trigger_caller: Address,
    bounty_paid: u128,
    block_height: u64,
}
```

---

## 5. Agentic interface — MCP tools and AFL-1 integration

The strategic case for QSHARE-1 depends on AI agents being able to operate the loop autonomously. Two MCP tools complete this surface:

### 5.1 `mcp__quillon-wallet__qshare_status`

```typescript
server.tool(
  "qshare_status",
  "Get current QSHARE treasury status — NAV per share, market price, premium ratio, next mint eligibility, and treasury composition.",
  {},
  async () => { /* read from contract via /api/v1/qshare/status */ }
);
```

Read-only, no auth needed. Returns human-readable JSON with all the values an agent needs to decide whether to call `try_autonomous_mint`.

### 5.2 `mcp__quillon-wallet__qshare_try_mint`

```typescript
server.tool(
  "qshare_try_mint",
  "Trigger the autonomous QSHARE mint if conditions are met. Pays a small trigger fee; returns bounty + mint details if successful.",
  {
    wallet: z.string().describe("qnk... address — must hold at least 0.01 QUG for trigger fee"),
  },
  async ({ wallet }) => { /* submit tx via AFL-1 /api/v1/agent/submit */ }
);
```

The tool submits via AFL-1's `/api/v1/agent/submit` endpoint, signing the call with the agent's X-Wallet-Auth header. No browser flow required.

This means: an AI agent observing the chain can autonomously trigger QSHARE accumulation events whenever the premium-ratio threshold is crossed, earning the trigger bounty as compensation, and earning long-term exposure (via held QSHARE) to the appreciating NAV.

### 5.3 Multi-vendor agent compatibility

Because the trigger is permissionless and the MCP tools are vendor-neutral, the QSHARE loop is operatable by:

- Claude Code agents (Anthropic)
- Codex agents (OpenAI)
- Grok Build agents (xAI)
- DeepSeek agents
- Any MCP-supporting client (including future agentic frameworks)

This is the multi-vendor agentic-money story made observable: agents from different model vendors competing/cooperating to trigger profitable mints, each earning bounties, all contributing to QSHARE NAV growth.

---

## 6. Security considerations

### 6.1 NAV oracle manipulation

The NAV calculation is fully deterministic on-chain — no external oracles. The risk surface is the DEX market-price oracle (§2.4), which uses a 30-block TWAP. Attackers attempting to spike QSHARE market price to trigger an unprofitable mint would need to sustain manipulation across ~3 minutes of blocks, against a pool of significant depth (the 0.5% pool-fraction cap prevents catastrophic damage even if manipulation succeeds momentarily).

### 6.2 Premium-of-the-premium MEV

A mint trigger captures bounty + benefits anyone holding QSHARE before the mint. MEV searchers might front-run the mint trigger. Mitigation: the trigger fee is paid in full to the caller upon success, and the bounty is small enough that the strategic gain from front-running is bounded. The 360-block cooldown also limits the frequency of MEV opportunities.

### 6.3 Treasury concentration in QCREDIT Platinum

Default policy deposits accumulated QUG into Platinum (180-day lock, 25% APY). If 100% of treasury is in Platinum, redemption flexibility is limited. The contract SHOULD maintain a tier-diversification policy: e.g., 40% Platinum, 30% Gold, 20% Silver, 10% Bronze. Configurable via governance.

### 6.4 Sybil-resistance of permissionless triggers

The 0.01 QUG trigger fee + cooldown + pool-depth requirements naturally Sybil-resist the trigger function. An attacker spawning many wallets to spam triggers gains nothing (the cooldown applies globally, not per-wallet), and each spam attempt costs 0.01 QUG.

### 6.5 Risk to QSHARE holders if QUG falls

NAV is denominated in QUG-equivalent. If QUG itself falls in USD terms, QSHARE NAV falls with it (in USD). This is the inverse of Strategy's risk: MSTR holders are exposed to BTC drawdowns. QSHARE holders are exposed to QUG drawdowns. This is *features, not bug* — long-QUG conviction is the prerequisite for holding QSHARE.

### 6.6 Activation height gating

Per CLAUDE.md mainnet-safety rules, QSHARE activation MUST be height-gated. The new token type, contract code, and mint logic activate at `QSHARE_ACTIVATION_HEIGHT`, set ~4 weeks (40,000 blocks) after this spec lands.

---

## 7. Reference implementation

Quillon Graph reference implementation:

- Smart contract: `crates/q-vm/src/contracts/qshare_treasury.rs` (to be written)
- HTTP API: `crates/q-api-server/src/qshare_api.rs` (status + try_mint endpoints)
- MCP tools: `tools/quillon-wallet-mcp/src/index.ts` (additions to existing MCP server)
- Trading-bot strategy: `crates/q-trading-bot/src/strategies/qcredit_dca.rs` (this commit — Phase 1 DCA against QCREDIT Platinum)

---

## 8. Phasing — three deployments

QSHARE-1 lands in phases to allow market data to inform parameter calibration:

### Phase 1 — QCREDIT-DCA (this commit)

Trading-bot strategy that simply DCAs QUG into QCREDIT Platinum tier. Earns 25% APY base yield without any premium-arbitrage. Confirms the L2 economics work and produces a measured QCREDIT-supply trajectory.

**Deliverable**: `qcredit_dca.rs` strategy module. Live test on agent wallet `qnk7154929a...` (see CLAUDE.md memory `agent-wallet-endowment`).

### Phase 2 — QSHARE bootstrap (~6 weeks after Phase 1)

QSHARE token contract deployed with `circulating_qshare = 0`. Initial QSHARE issuance is via a one-shot bootstrap: deposit accumulated QCREDIT from Phase 1, receive QSHARE 1:1 against NAV. DEX pool QSHARE/QUG created.

No premium-arbitrage yet; just the share token.

### Phase 3 — Autonomous arbitrage activation (~10 weeks after Phase 1)

`try_autonomous_mint` activates at the height-gated boundary. From this point forward, any agent can trigger mints when premium > 1.5. The full Saylor loop is live, executing autonomously.

---

## 9. Acceptance criteria

For QSHARE-1 to be considered fully deployed:

1. QCREDIT-DCA trading-bot strategy operational on at least one wallet (Phase 1)
2. QSHARE token contract deployed and active, NAV calculation correct (Phase 2)
3. At least 10 successful autonomous-mint trigger events recorded on-chain (Phase 3)
4. MCP tools `qshare_status` and `qshare_try_mint` callable from all four major agentic CLIs
5. Treasury composition diversified across tiers per §6.3
6. No NAV-oracle-manipulation events in first 10,000 blocks of Phase 3
7. Bounty-paid-to-trigger-callers proves Sybil-resistance and incentive alignment

---

## 10. What this is NOT

- **Not a security or investment instrument under traditional regulation**. QSHARE is a permissionless on-chain token; treatment under local jurisdiction is the holder's responsibility.
- **Not a guaranteed-yield product**. NAV per share is expected to rise over time given premium-arbitrage activity, but no guarantee exists. QUG drawdowns flow through to QSHARE.
- **Not a competitor to QCREDIT**. QCREDIT is the credit layer (L2); QSHARE is the share layer (L3). They are complementary parts of the same stack.
- **Not Saylor's strategy verbatim**. Differences: autonomous (no CEO), on-chain (no SEC filings), AI-callable (no broker), with 0.5% pool-depth caps that don't exist in equity markets.

---

## 11. Companion: AFL-1

QSHARE-1 leverages AFL-1's transaction-submission infrastructure for agent-driven mint triggers:

- Agents authenticate via X-Wallet-Auth (AFL-1 §3)
- Mint-trigger calls submit through `/api/v1/agent/submit` (AFL-1 §4.1)
- Replay protection via body_hash (AFL-1 §3.5)
- No browser flow

QSHARE without AFL-1 still works (any wallet can call), but the agent-native experience requires AFL-1's signed-submit path.

---

## 12. Test vectors

> Full test vectors deferred to v1-final after reference implementation telemetry.

Required test vector classes:

- NAV calculation: given a treasury_qcredit_basket + time, compute nav_per_qshare exactly
- Mint sizing: given premium_ratio, pool_depth, circulating_qshare, compute max_mint_amount
- Bounty calculation: given mint_amount and qug_accumulated, compute bounty
- TWAP construction: given 30 price samples, compute the 30-block TWAP

---

## 13. Acknowledgements

Inspired by:

- **MicroStrategy / Strategy** (Michael Saylor) — the original premium-arbitrage accumulation playbook. QSHARE is what that strategy looks like when expressed as a smart contract.
- **AFL-1 protocol** (Quillon Graph) — the agentic-AI submission path that makes permissionless triggers practical for AI agents.
- **QCREDIT vault** (Quillon Graph) — the L2 credit layer that QSHARE composes on top of. Code comment in `qcredit_vault.rs` already references "Strategy's 3-layer stack" — QSHARE is the L3 that was promised.
- **The cryptography mailing list discussion of Montana** (Alejandro Montana, May 2026) — for the parallel exploration of time-based scarcity vs. fee-based access. Different solutions to different audiences; same underlying recognition that post-quantum design space is wide.

---

## 14. Copyright

Apache-2.0. Reference implementation in Q-NarwhalKnight licensed identically. Other chains with compatible credit-vault primitives may implement QSHARE-1 without further restriction.

---

*Drafted by Claude Opus 4.7 and the Quillon Foundation. The companion code in `crates/q-trading-bot/src/strategies/qcredit_dca.rs` (Phase 1) lands in the same commit; Phase 2 and Phase 3 follow in subsequent PRs as the L3 contract and on-chain DEX pool are wired up.*
