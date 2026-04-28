# State Root, DEX Swaps, and Token Balances — Third Technical Review
**Date:** 2026-04-28  
**Author:** Claude Sonnet 4.6 synthesis  
**Context:** The second review established that a `balance_root` in block headers is the permanent fix for native coin balance divergence. This third review addresses the follow-on question: *How does `balance_root` interact with DEX swaps, token balances, and complex on-chain state? What is the complete state root, and how do we protect it?*

---

## 1. The Problem: `balance_root` Only Covers Native Coin

The proposed `balance_root` commits to `wallet_balance_*` keys — native QUG coin holdings. But Q-NarwhalKnight has much broader state:

| State type | RocksDB key prefix | Protected by current balance_root? |
|---|---|---|
| Native coin wallets | `wallet_balance_*` | ✅ (planned) |
| Token balances (DEX tokens, native USD) | `token_balance_*` | ❌ Not yet |
| DEX pool reserves | `liquidity_pool:*` | ❌ Not yet |
| Swap history | `swap_history_*` | ❌ Not yet |
| Smart contract state | `contract_*` | ❌ Not yet |
| Stake positions | `stake_position_*` | ❌ Not yet |
| Price history | `price_history_*` | ❌ Not yet |

If we only commit to native coin balances, all other state remains unprotected. Two nodes could agree on native coin balances but disagree on DEX pool reserves, causing completely different swap outcomes for the same transaction.

**Example:**
```text
Pool: QUG/USD
Node A reserves: 1,000,000 QUG / 29,486,811.50 USD
Node B reserves: 1,000,000 QUG / 29,000,000 USD (diverged)

User submits swap: sell 1,000 QUG
Node A gives: ~29,486 USD
Node B gives: ~29,000 USD
```

Two nodes apply the same transaction and produce different outputs. This is the same problem as native coin divergence, just at the DEX layer.

---

## 2. The User's Actual Position and Why It's Correctly Isolated

**Current position on Epsilon:**  
- 29,486,811.50 native coin USD (hedged via DEX swap)
- This is stored as `token_balance_*` keys, NOT `wallet_balance_*` keys

**Why the balance checkpoint (v10.4.14/v10.4.15) does NOT touch this:**
```rust
// The checkpoint only purges native coin keys:
let deleted = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
// token_balance_* keys are completely untouched
```

The user's 29,486,811.50 native coin USD is **safe on Epsilon** and will remain exactly as-is. The checkpoint only imports/overwrites native QUG coin wallet balances.

**Why testing only on Delta's container is the right call:**  
Any accidental corruption of Epsilon's DEX token balances would affect this position. The container in Delta is isolated — it has no real funds. All development and testing happens there until the mechanism is proven.

---

## 3. The Complete Picture: What Does "Full State" Mean?

A full state commitment for Q-NarwhalKnight must eventually cover:

### 3.1 Native Coin State (Layer 1 — already in balance_root plan)

```rust
// The simplest: sorted list of (address: [u8;32], balance: u128)
// All 1,332 wallets (and growing), committed per block
// Blake3 hash of sorted canonical serialization
```

### 3.2 Token State (Layer 2)

```text
For each ERC-20-like token contract:
  For each holder address:
    token_balance_{token_id}_{holder_address} = amount
```

These must be included in the state root for DEX operations to be deterministic.

### 3.3 DEX Pool State (Layer 3)

```text
For each pool:
  liquidity_pool:{token_a}:{token_b} = {
    reserve_a: u128,
    reserve_b: u128,
    total_lp_tokens: u128,
    fee_rate: u64,
    ...
  }
```

Pool reserves change with every swap. If they diverge between nodes, AMM pricing diverges, and swap outputs differ for the same input.

### 3.4 The Full State Root

The complete state root (Ethereum-style) is:

```rust
pub struct FullStateRoot {
    /// Native coin balances: Blake3(sorted wallet:balance pairs)
    pub native_coin_root: [u8; 32],
    
    /// Token balances: Blake3(sorted token_id:holder:amount triples)
    pub token_root: [u8; 32],
    
    /// DEX pool reserves: Blake3(sorted pool_id:reserve_a:reserve_b:lp_supply)
    pub dex_root: [u8; 32],
    
    /// Contract state hashes: Blake3(sorted contract_id:state_hash pairs)
    pub contract_root: [u8; 32],
    
    /// Combined: Blake3(native_coin_root || token_root || dex_root || contract_root)
    pub state_root: [u8; 32],
}
```

In practice, `block.balance_root` should become `block.state_root: [u8; 32]` covering ALL of this.

---

## 4. How DEX Swaps Interact with Deterministic State

### 4.1 The Determinism Problem in DEX Swaps

A DEX swap is NOT a simple balance transfer. It involves:

1. **Constant-product formula**: `(x + Δx)(y − Δy) = k`, where `k = x * y` (current reserves)
2. **Fee deduction**: typically 0.3% of input, retained in pool
3. **Slippage bounds check**: reject if output < minimum
4. **LP token minting/burning**: when adding/removing liquidity

If pool reserves `x` and `y` diverge between nodes, the output `Δy` for the same `Δx` differs. This means two nodes apply the same swap transaction and produce different state.

### 4.2 Current Vulnerability

Right now, `liquidity_pool:*` keys in RocksDB can diverge between nodes via:
- Different startup sequences
- DEX state re-sync via P2P (pools synced every 5 minutes, not per-block)
- The v10.4.14/v10.4.15 checkpoint explicitly skips `liquidity_pool:*` keys

This means two nodes can agree on the block containing a swap, compute different outputs (due to diverged pool state), and silently produce different token balances.

### 4.3 The Fix: Pool State Must Be Block-Transition State

Pool reserves must be derived purely from the sequence of swap transactions in the chain — just like native coin balances must be derived from coinbase and transfer transactions. The fix:

```text
Pool state at block N = apply_all_swaps(initial_pool_state, blocks 0..N)
```

Not:
```text
Pool state at block N = P2P_gossip_of_pool_state + startup_adjustments
```

### 4.4 Transaction Ordering Within a Block

For deterministic swap outcomes, transaction ordering within a block must be canonical. Ethereum uses the order in which transactions appear in the block (miners control this). In Q-NarwhalKnight, blocks produced by DAG-Knight need a deterministic transaction order:

```text
Option A: Sort by transaction hash (deterministic but miner can manipulate by choosing txs)
Option B: Order by submission timestamp (non-deterministic — nodes see txs at different times)
Option C: Order by fee (highest fee first, deterministic if fees are explicit)
Option D: Include transactions in the order the block producer chose (committed to block hash)
```

**Recommendation**: Option D (order committed to block hash). The block producer chooses an order, commits it to the block, and all nodes replay exactly that order. This is what Ethereum does.

---

## 5. What Changes for the swap/token flow in the balance_root architecture

### 5.1 Each Swap Transaction Must Be On-Chain

Currently, some DEX fee distributions happen off-chain (startup adjustments). Under `balance_root`, every state change must flow through a transaction in a block:

```rust
// Each DEX swap is a transaction in the block:
Transaction::Swap {
    pool_id: [u8; 32],
    sender: [u8; 32],
    amount_in: u128,
    token_in: [u8; 32],  // or native coin indicator
    min_amount_out: u128,
    deadline: u64,
}

// The block producer executes this deterministically:
// 1. Load pool reserves from pre-state
// 2. Compute output: Δy = (y * Δx * (1 - fee)) / (x + Δx * (1 - fee))
// 3. Update pool reserves in state
// 4. Update sender token balance
// 5. Update recipient token balance
// All of this flows into the post-state balance_root
```

### 5.2 LP Token Operations

Adding/removing liquidity changes the pool state and the user's LP token balance — both must be on-chain:

```rust
Transaction::AddLiquidity {
    pool_id: [u8; 32],
    provider: [u8; 32],
    amount_a: u128,
    amount_b: u128,
    min_lp_tokens: u128,
}

Transaction::RemoveLiquidity {
    pool_id: [u8; 32],
    provider: [u8; 32],
    lp_tokens_in: u128,
    min_amount_a: u128,
    min_amount_b: u128,
}
```

### 5.3 The `native coin USD` Position

The user's 29,486,811.50 native coin USD is a token balance (`token_balance_{USD_contract}_{user_address}`). Under the full `state_root`:

```text
state_root includes token_root
token_root includes token_balance_{USD_contract}_{user_address} = 29486811_500000...
```

Once `state_root` is in blocks, this balance is **just as protected as native coin** — any block that changes it incorrectly will produce a state root that doesn't match what other nodes compute, and will be rejected.

---

## 6. Phased Rollout for Full State Commitment

### Phase 0 (current) — Native coin checkpoint only
- ✅ Import 1,332 native coin balances from Epsilon snapshot
- ✅ Integrity verification (count, total supply, SHA-256)
- ⬜ Replay blocks H+1..tip to catch up post-checkpoint native coin changes
- Token balances, pool state: not touched

### Phase 2a — Native coin `balance_root` in blocks
- `balance_root` covers only `wallet_balance_*` (native QUG)
- Activates at height A₁ (after native coin transition is verified stable)
- DEX/token state still unprotected at this stage

### Phase 2b — Extend to token balances
- `balance_root` expands to include `token_balance_*`
- Requires: all token balance changes go through on-chain transactions
- The current P2P token sync must be removed
- Checkpoint updated to include token balance snapshot (including the 29,486,811.50 USD position)

### Phase 2c — Full state root including DEX pools
- `state_root` replaces `balance_root`
- Covers: native coin + tokens + DEX pool reserves
- Requires: all pool state changes go through on-chain transactions
- The 5-minute P2P pool sync must be removed
- Pool state initialized from genesis or a verified checkpoint snapshot

### Phase 3 — Remove ALL off-chain state mutations
- No startup DEX adjustments
- No authority peer sync for any state type
- No backward HashMap sync
- No P2P pool gossip (only block transactions)
- All state changes: block transactions only

---

## 7. DEX Pool State Checkpoint (The Missing Piece)

The current balance checkpoint (v10.4.14) captures only native coin balances. To protect the user's DEX position, we need a DEX pool state checkpoint as well. This should be captured from Epsilon at the same height (16,538,868) and include:

```rust
// What to snapshot per pool:
pub struct PoolCheckpoint {
    pub pool_id: [u8; 32],       // (token_a, token_b) hash
    pub token_a: [u8; 32],       // token contract address
    pub token_b: [u8; 32],       // token contract address (or native coin marker)
    pub reserve_a: u128,         // reserves at checkpoint height
    pub reserve_b: u128,         // reserves at checkpoint height
    pub total_lp_supply: u128,   // total LP tokens at checkpoint height
    pub fee_rate_bps: u32,       // fee in basis points (e.g., 30 = 0.3%)
}
```

And for each token holder position (including the 29,486,811.50 USD):
```rust
pub struct TokenBalanceCheckpoint {
    pub token_id: [u8; 32],      // token contract address
    pub holder: [u8; 32],        // wallet address
    pub balance: u128,           // balance at checkpoint height (raw, in token decimals)
}
```

This would be a **v10.5.x** feature — the native coin checkpoint (v10.4.14/v10.4.15) must be proven stable on Delta's container first.

---

## 8. The Economic Hedge and Why the Architecture Protects It

The user's rationale for holding native coin USD: **hedge against native coin price decline while maintaining economic force to buy back**.

Under the full `state_root` architecture:
1. The 29,486,811.50 USD token balance is committed to every block's state root
2. Any swap, transfer, or other change to this balance must appear as a transaction in a block
3. All nodes independently compute the same post-state root
4. Disagreement about the balance → disagreement about the state root → block rejected
5. The balance becomes as immutable as Bitcoin's UTXO set, without manual intervention

The checkpoint approach today is an emergency fix. The `state_root` approach is the permanent guarantee that makes the hedge cryptographically protected.

---

## 9. Summary

| Question | Answer |
|---|---|
| Does the balance checkpoint touch token balances (native USD)? | No. Only `wallet_balance_*` (native QUG) is touched. Token balances are safe. |
| Can DEX swaps cause state divergence even after balance_root? | Yes, if pool reserves diverge and are not covered by the state root. |
| What is the complete state that needs consensus protection? | Native coin + token balances + DEX pool reserves + contract state. |
| What is the phased path to full protection? | Phase 2a: native coin root → Phase 2b: + token root → Phase 2c: full state root. |
| What must change for DEX operations to be deterministic? | All pool state changes must flow through on-chain transactions; P2P pool sync removed. |
| When should the user's USD position be snapshotted? | In v10.5.x, after native coin checkpoint is proven. Capture token + pool state from Epsilon at same height. |
| How long until the user's position is cryptographically guaranteed? | Phase 2b estimated 8-12 weeks from now, after Phase 2a (native coin root) is stable on mainnet. |

---

## 10. External Consultation: Incorporating DEX Concerns

The third wave of external advisors (Advisor A + Advisor B + DeepSeek) all note the same unresolved issue:

**The `balance_root` is necessary but not sufficient once DEX operations exist.**

Key excerpt from Advisor A (2026-04-28):
> "Any mutation outside the block processing pipeline will immediately cause a root mismatch. The team should ensure all DEX adjustments are on-chain before the activation height. The existing DEX startup adjustment must be a no-op or removed before the fork block."

This confirms: if DEX startup adjustments run after `balance_root` is active, they will change token balances and pool reserves WITHOUT going through block transactions. The node will then compute a different state root from its peers → it gets kicked off the network.

The sequence for DEX correctness must be:
1. Remove `apply_dex_qug_adjustments()` (or make it a no-op) — Phase 3
2. Remove the 5-minute P2P pool sync — Phase 3  
3. Ensure all DEX operations flow through block transactions — Phase 2c
4. Add token + pool state to `state_root` — Phase 2c
5. Snapshot Epsilon's token/pool state as the canonical starting point — v10.5.x

Only after all of these steps are complete does the user's native coin USD position have the same consensus-level protection as native QUG coin.

---

*This document is the third in a series:*
- *`docs/technical-review-balance-divergence-root-cause-2026-04-28.md` — root cause analysis*
- *`docs/technical-review-why-decentralization-failed-2026-04-28.md` — architectural diagnosis*
- *`docs/technical-review-state-root-dex-tokens-2026-04-28.md` — this document: DEX/token state and the full state root*
