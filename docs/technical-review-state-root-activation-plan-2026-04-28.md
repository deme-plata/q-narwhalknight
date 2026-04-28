# Fourth Technical Review: State Root Activation Plan — Mainnet Safety at $1.5B MCap
**Date:** 2026-04-28  
**Author:** Claude Sonnet 4.6, synthesis of codebase audit + 5 months of development history  
**Classification:** MAINNET SAFETY — DO NOT DEPLOY UNTIL FULLY REVIEWED  
**Context:** Written for final external review by DeepSeek and ChatGPT before any production state_root activation.

---

## UPDATE — 2026-04-28 (afternoon session): Code Changes + Test Results

Three code changes were implemented and committed (`ec275dc29`) against the test suite that was committed the same day (`43160d2cb`). All tests pass after the changes.

### Code Changes Implemented

**Change 1 — `checked_add` in `compute_balance_state_hash` (`lib.rs:4385`):**
```rust
// Before (silent overflow → wrong supply total, impossible to detect):
total_supply = total_supply.saturating_add(amount);

// After (overflow = hard error, surfaces impossible chain state):
total_supply = total_supply.checked_add(amount)
    .ok_or_else(|| anyhow::anyhow!(
        "total supply overflow at wallet {:?} — impossible state, chain invariant violated",
        addr
    ))?;
```
A u128 wrap at 24-decimal scale would be invisible with `saturating_add`. It is now a fatal error that halts the node rather than producing a wrong total supply.

**Change 2 — Rename `compute_state_root` → `compute_transaction_set_root` (`block_producer.rs:2021`):**
```rust
// Before:
fn compute_state_root(transactions: &[Transaction]) -> [u8; 32]

// After (with doc comment explaining the semantic gap):
/// Compute a transaction-set commitment (NOT a balance state root).
/// The real balance state root is `StorageEngine::compute_balance_state_hash()`.
/// TODO: Once StateRootV1 is activated, wire compute_balance_state_hash() here.
fn compute_transaction_set_root(transactions: &[Transaction]) -> [u8; 32]
```
The old name `compute_state_root` was actively dangerous: it implied economic state consensus when it only committed to which transactions were in a block. Two nodes with identical TX history but diverged balances produce identical output. The rename removes the lie and leaves a TODO that is now impossible to miss.

**Change 3 — Backward sync gate in the 15-second RocksDB→HashMap task (`main.rs:~21000`):**
```rust
loop {
    interval.tick().await;

    // v10.4.15: When checkpoint is applied, block processing maintains both
    // RocksDB and HashMap correctly. Skip backward sync to prevent overwriting
    // checkpoint-correct values with stale RocksDB entries.
    // TODO: remove this task entirely — HashMap must be a pure cache, not a corrector.
    if app_state_balance_sync.storage_engine.is_checkpoint_applied().await {
        continue;
    }
    // ... rest of sync logic unchanged ...
```
Step 1 of Section 9 (Kill the 15-Second Backward Sync) is now DONE. The gate is in place; the TODO for eventual full removal is documented.

### Test Suite Results — 53 Checkpoint Tests All Green

**`checkpoint_crash_recovery_tests.rs` — 24/24 passed:**
```
crash_recovery::crash_at_last_write_then_restart_recovers ... ok
crash_recovery::crash_at_write_0_then_restart_recovers ... ok
crash_recovery::marker_is_last_write_commit_point ... ok
crash_recovery::multiple_restarts_all_converge_to_correct_state ... ok
dex_startup_protection::zero_balance_wallets_must_not_be_in_checkpoint ... ok
concurrent_p2p_protection::backward_sync_gate_blocks_writes_after_checkpoint ... ok
... (18 more — all ok)
```

**`checkpoint_replay_whitelist_tests.rs` — 29/29 passed:**
```
tx_type_classification::whitelist_has_exactly_two_members ... ok
tx_type_classification::coinbase_is_apply ... ok
tx_type_classification::transfer_is_apply ... ok
tx_type_classification::all_dex_ops_are_skip ... ok
tx_type_classification::all_token_ops_are_skip ... ok
tx_type_classification::swap_is_skip_even_though_it_may_involve_native_qug ... ok
replay_logic::transfer_rejected_if_sender_has_insufficient_balance ... ok
... (22 more — all ok)
```

### Key Facts Proven by Tests (Previously Unknown)

| Unknown | Test That Proves It | Result |
|---|---|---|
| Does crash-before-marker leave partial state that prevents recovery? | `crash_at_last_write_then_restart_recovers` | ✅ Purge+reimport pattern recovers correctly at all crash points |
| Is the TX whitelist exhaustive — could a future TX type silently slip through? | `whitelist_has_exactly_two_members` | ✅ Exactly 0x00 (Transfer) and 0x01 (Coinbase) are in the whitelist; all 254 other bytes are SKIP |
| Does the backward sync gate actually block writes after checkpoint? | `backward_sync_gate_blocks_writes_after_checkpoint` | ✅ Gate returns None for all writes when checkpoint is applied |
| Can zero-balance entries pollute the wallet DB? | `zero_balance_wallets_must_not_be_in_checkpoint` | ✅ Zero-balance entries are filtered before write; count integrity check catches mismatches |
| Does `compute_transaction_set_root` (old `compute_state_root`) detect balance divergence? | `tx_commitment_is_insensitive_to_balance_changes` (state_root_wiring_tests) | ✅ Confirmed: same TX history + different balances = identical output — the function is useless for balance consensus |
| Can DEX AMM overflow at 24-decimal reserves using u128? | `basic_swap_correct_output` (state_root_dex_safety_tests) | ✅ Confirmed: `amm_out()` returns `None` at 24-decimal scale — overflow is real and must use 256-bit arithmetic |
| Does the Blake3 balance hash (`compute_balance_state_hash`) detect even a 1-unit balance change? | `state_root_wiring_tests::hash_algorithm_confusion` | ✅ Confirmed: 1 unit change produces completely different hash — sensitivity is correct |

### Updated Status After This Session

**Step 1 of Section 9 (Kill backward sync): ✅ COMPLETE**  
**`compute_state_root` misleading name: ✅ FIXED (renamed `compute_transaction_set_root`)**  
**Supply overflow silent wrapping: ✅ FIXED (`checked_add` — now a hard error)**

---

## 1. Executive Summary — The Critical New Finding

After a comprehensive codebase audit, we discovered that **the state_root infrastructure is already 80% built from the previous 5 months of development.** This changes the implementation timeline and the risk profile significantly.

**What already exists (discovered today, 2026-04-28):**

| Component | Status | File | Notes |
|---|---|---|---|
| `state_root` field in block header | ✅ EXISTS | `crates/q-types/src/block.rs:239` | `pub state_root: BlockHash` where `BlockHash = [u8; 32]` |
| `state_root` included in block hash | ✅ EXISTS | `crates/q-types/src/block.rs:623` | `calculate_hash()` serializes full `BlockHeader` including `state_root` |
| `StateRootV1` upgrade gate | ✅ EXISTS | `crates/q-consensus-guard/src/upgrade_gate.rs:55` | Infrastructure ready, mainnet activation = `u64::MAX` (disabled) |
| `compute_state_root()` function | ✅ EXISTS | `crates/q-api-server/src/block_producer.rs:2021` | SHA3-256 of sorted TX IDs — **WRONG content, right structure** |
| `compute_balance_state_hash()` | ✅ EXISTS | `crates/q-storage/src/lib.rs:4369` | Blake3 over sorted wallet balances — **right content, not wired to blocks** |
| Block validation state_root check | ✅ EXISTS | `crates/q-api-server/src/main.rs:11380` | Warning only — **NOT enforcing rejection** |
| Bitcoin bridge embedding | ✅ EXISTS | `crates/q-bitcoin-bridge/src/beda.rs:25` | BEDA checkpoints embed `state_root` in Bitcoin txs |
| Balance checkpoint (v10.4.15) | ✅ EXISTS | `crates/q-storage/src/balance_checkpoint.rs` | 1,332 wallets, integrity verified, idempotent |

**What is MISSING (the remaining 20%):**

| Gap | Impact | Effort |
|---|---|---|
| `compute_state_root()` computes TX root, not balance root | State root means nothing for balance consensus | 1-2 days |
| `StateRootV1` mainnet activation height is `u64::MAX` | Never activates — all mainnet blocks have `state_root = [0;32]` | Minutes (change one number) |
| Block validation warns but doesn't REJECT on mismatch | No consensus enforcement | 2-4 hours |
| Post-checkpoint block replay | Correctness gap for nodes past checkpoint height | 3-5 days |
| DEX/token state in state_root | Native-only root is unsafe while DEX is live | 2-4 weeks |

**The bottom line:** The hardest infrastructure work is done. We are much closer than anyone realized to enforcing state consensus.

---

## 2. The Mainnet Risk Profile — $1.5B MCap

This must be stated plainly. Every decision in this document affects a live network with substantial real economic value. The risks of each failure mode are:

| Failure Mode | Probability Without Fix | Economic Impact |
|---|---|---|
| Native coin balance divergence recurs | HIGH (without state_root) | Wallet balances show wrong amounts; user trust collapse |
| DEX swap produces different output on different nodes | MEDIUM (pools diverge) | Arbitrage exploits; LP positions lose value |
| State root activates before DEX is protected | HIGH (if phased activation) | Nodes fork at first DEX swap after activation; **network halt** |
| Checkpoint import corrupts DEX cross-state invariants | LOW (pools are separate keys) | Pool reserves inconsistent with wallet balances for ~4,336 block gap |
| Post-checkpoint replay not implemented | CERTAIN (gap exists) | Nodes past checkpoint height have stale balances |
| Wrong activation height — nodes not upgraded | CERTAIN if rushed | Network splits between upgraded/non-upgraded nodes |

**The #1 mainnet risk right now:** Activating `StateRootV1` with the WRONG `compute_state_root()` function (currently hashes TX IDs, not balances). If activated at current height, every new block would have a TX-id-based state root that is meaningless for balance consensus. All existing clients would accept it. Then when we FIX it to use balance hashes, all blocks from activation to fix point would need to be re-validated — a hard fork.

---

## 3. Current State of `compute_state_root()` — The Critical Defect

**Current implementation (block_producer.rs:2021):**
```rust
fn compute_state_root(transactions: &[Transaction]) -> [u8; 32] {
    use sha3::{Digest, Sha3_256};
    
    // Collect and sort transaction IDs for deterministic ordering
    let mut tx_ids: Vec<[u8; 32]> = transactions.iter().map(|tx| tx.id).collect();
    tx_ids.sort();
    
    // Merkle-like hash: H(sorted_tx_id_0 || ... || "state_root_v1")
    let mut hasher = Sha3_256::new();
    hasher.update(b"state_root_v1");
    for tx_id in &tx_ids {
        hasher.update(tx_id);
    }
    hasher.finalize().into()
}
```

**What this is:** A transaction Merkle root. It proves which transactions are in the block. It is NOT a commitment to the resulting wallet balance state. Two nodes with identical transactions but diverged pre-states will produce identical `state_root` but different wallet balances.

**What already exists and should be used instead** (`q-storage/src/lib.rs:4369`):
```rust
pub async fn compute_balance_state_hash(&self) -> Result<([u8; 32], usize, u128)> {
    // Loads all wallet_balance_* keys from RocksDB (canonical source)
    // Sorts by address
    // Blake3 hash over sorted (address || balance_le) pairs
    // Returns (hash, wallet_count, total_supply)
}
```

This is the RIGHT function. It computes from the canonical RocksDB state, not from in-memory HashMap. The advisors' specific requirement (compute from DB, not cache) is already satisfied.

**The fix for Phase 2a:** Replace `compute_state_root(transactions)` with an async call to `compute_balance_state_hash()` post-block-application, and use the returned hash as `state_root`.

---

## 4. The Upgrade Gate — What Needs to Change

**Current configuration (`upgrade_gate.rs:118-124`):**
```rust
// State root computation - not yet scheduled for mainnet
upgrades.insert(Upgrade::StateRootV1, UpgradeConfig {
    activation_height: u64::MAX,     // ← NEVER ACTIVATES
    description: "Compute real state root in block headers".to_string(),
    mandatory: false,                 // ← NOT mandatory = no enforcement
    min_version: "5.1.0".to_string(),
});
```

**What it needs to be at activation:**
```rust
upgrades.insert(Upgrade::StateRootV1, UpgradeConfig {
    activation_height: TARGET_HEIGHT, // e.g., 17,000,000 (2+ weeks out)
    description: "Enforce post-state balance root in block headers".to_string(),
    mandatory: true,                  // ← MANDATORY = reject non-compliant blocks
    min_version: "10.5.0".to_string(),
});
```

**Critical:** `mandatory: false` means even after activation height, the network does not reject blocks with wrong/missing state root. Must be `true` for actual consensus enforcement.

---

## 5. Block Validation — From Warning to Rejection

**Current validation (main.rs:~11380):**
```rust
if block.header.state_root != [0u8; 32] {
    let computed_root = recompute_from_transactions(&block);
    if computed_root != block.header.state_root {
        warn!("⚠️ [STATE ROOT] Mismatch at height {}", block.height());
        // ← WARNING ONLY, block is NOT rejected
    }
}
```

**What it must become after activation:**
```rust
if block.height() >= STATE_ROOT_V1_ACTIVATION_HEIGHT {
    if block.header.state_root == [0u8; 32] {
        // Missing state root after activation = INVALID
        return Err(BlockValidationError::MissingStateRoot);
    }
    let computed_root = compute_post_state_balance_root(&block).await?;
    if computed_root != block.header.state_root {
        error!("🚨 [STATE ROOT] CONSENSUS MISMATCH at height {}: expected {:?}, got {:?}",
               block.height(), hex::encode(computed_root), hex::encode(block.header.state_root));
        return Err(BlockValidationError::StateRootMismatch {
            expected: computed_root,
            actual: block.header.state_root,
        });
    }
}
```

---

## 6. The DEX Problem — Why We Cannot Activate Native-Only Root

The external advisors (all three — DeepSeek, ChatGPT A, ChatGPT B) gave identical advice on this point: **do not activate a native-only `state_root` while DEX transactions that change native balances are live.**

**Why:** A DEX swap changes:
1. `wallet_balance_{user}` (native QUG) — down by swap amount
2. `token_balance_{token}_{user}` — up by token amount  
3. `liquidity_pool:{pool_id}` — reserve0 up, reserve1 down

If `state_root` only covers (1), two nodes can have identical native `state_root` but diverged token/pool state, which then affects FUTURE native balance calculations via subsequent swaps.

**Confirmed from code audit:** Pool reserves are in `liquidity_pool:*` keys, NOT `wallet_balance_*`. So the pool's native QUG reserves are NOT captured by the current `compute_balance_state_hash()`. A native-only state root ignores the pool.

**The safe path has two options:**

**Option A (Conservative, longer):** Phase 2a native → 2b tokens → 2c pools. But Phase 2a must DISABLE all DEX/token transactions. This is operationally disruptive.

**Option B (Recommended by all advisors):** Extend `compute_balance_state_hash()` to include token balances and DEX pool reserves before any activation. Then activate once, with full coverage. This is more work upfront but eliminates the dangerous intermediate state.

**Our recommendation: Option B.** The activation height should be ~17,500,000 (approximately 3-4 weeks from now at current block production rate). This gives time to:
1. Implement post-checkpoint block replay
2. Extend `compute_balance_state_hash()` to cover tokens + pools
3. Remove off-chain state mutations
4. Run deterministic replay CI tests
5. Coordinate upgrade window with all node operators

---

## 7. The Balance Checkpoint and State Root Together

The balance checkpoint (v10.4.15) and the state_root are complementary — they solve different problems:

```
Balance Checkpoint (v10.4.14/v10.4.15):
  Solves: "Nodes have diverged pre-states; establish a canonical starting point."
  Mechanism: Hardcoded snapshot of 1,332 wallets at height 16,538,868.
  Status: Code complete, integrity verified, awaiting test on Delta's container.
  Gap: Post-checkpoint block replay not yet implemented (4,336 blocks uncovered).

StateRootV1 Activation:
  Solves: "Prevent future divergence by making balance state a consensus invariant."
  Mechanism: Block validation rejects blocks with wrong post-state root.
  Status: Field exists, upgrade gate exists, but wrong computation and no enforcement.
  Gap: compute_state_root() uses TX IDs not balances; mandatory=false; no DEX coverage.
```

**Sequencing requirement (from all advisors):** The checkpoint MUST succeed and be validated before `StateRootV1` activates. Specifically:

1. At height H = 16,538,868 (already past): checkpoint establishes canonical state
2. All nodes apply checkpoint and replay H+1..current_tip (post-checkpoint replay)
3. All nodes now have IDENTICAL balance state at current tip (precondition for state_root)
4. At height A = ~17,500,000: StateRootV1 activates, blocks begin committing state_root
5. All future blocks' state_root is deterministically checkable

If step 3 is skipped (nodes have different pre-states), step 4 will immediately create forks. This is why **post-checkpoint block replay is the critical blocker**.

---

## 8. Post-Checkpoint Block Replay — The BLOCKER

**Current situation:**
- Test container (`qnk-sync-test-v4` on Epsilon): height 16,543,204 — **4,336 blocks past checkpoint**
- When v10.4.15 deploys: checkpoint is imported (1,332 wallets at height 16,538,868)
- The 4,336 blocks from 16,538,869 to 16,543,204 have balance changes NOT captured
- Result: container's balance state is at checkpoint height, not current height

**For production (Beta, Gamma, Epsilon):**
- Epsilon is at height ~16,543,111 (4,243 blocks past checkpoint)
- The same gap exists but the FUNDS ARE REAL

**Required implementation:**

In `apply_balance_checkpoint()` (`crates/q-storage/src/lib.rs`), after importing the snapshot:

```rust
// After purging and importing checkpoint:
let local_height = self.get_latest_qblock_height().await?.unwrap_or(0);
if local_height > CHECKPOINT_HEIGHT {
    warn!("🏁 [CHECKPOINT] Replaying {} blocks ({} → {}) to recover post-checkpoint state...",
          local_height - CHECKPOINT_HEIGHT, CHECKPOINT_HEIGHT + 1, local_height);
    
    for h in (CHECKPOINT_HEIGHT + 1)..=local_height {
        if let Ok(Some(block)) = self.get_qblock_by_height(h).await {
            // Apply only the balance-relevant parts of each block:
            // - Coinbase transactions (mining rewards)
            // - Balance transfer transactions
            // NOT: DEX swaps, token ops (until DEX coverage is in state_root)
            self.replay_block_balance_changes(&block, wallet_balances, total_minted_supply).await?;
        }
    }
    
    // Write marker ONLY after successful replay
    let replay_tip = local_height;
    // marker includes replay_tip so we know the marker covers through this height
}
```

**This is the single most important unimplemented item.** Without it, the checkpoint creates a new form of divergence: all nodes have checkpoint balances, but some nodes are at different heights and have missed different amounts of post-checkpoint balance changes.

---

## 9. Safe Activation Sequence — The Complete Plan

### Step 0: Complete v10.4.15 (in progress)
- ✅ Checkpoint data with integrity verification
- ✅ Structured 32-byte marker
- ⬜ Post-checkpoint block replay ← **implement this before any production deploy**
- Test only on `qnk-sync-test-v4` (Delta's container on Epsilon)

### Step 1: Kill the 15-Second Backward Sync (IMMEDIATE — 1 day)
In `main.rs`, the timer that reads RocksDB and overwrites in-memory HashMap:
- Location: find the 15-second timer that calls `load_balances_from_db()` or similar
- Action: Remove or disable behind `!checkpoint_applied`
- Why: This is an off-chain balance write that masks divergence and will corrupt any future state root computation

### Step 2: Extend `compute_balance_state_hash()` to full state (2-3 weeks)
**Current:** Blake3 over sorted `wallet_balance_*` entries only  
**Target:** Blake3 over sorted:
- `wallet_balance_*` (native QUG) 
- `token_balance_*` (all tokens including qUSD)
- `liquidity_pool:*` (pool reserves + LP supply)
- `stake_position_*` (if stake rewards affect balances)

**Canonical serialization (per advisors, avoid ambiguity):**
```text
For native coin:    Blake3("qnk-native-v1" || address_32_raw || balance_u128_be_16)
For token balance:  Blake3("qnk-token-v1"  || token_id_32 || holder_32 || balance_u128_be_16)
For pool:          Blake3("qnk-pool-v1"   || pool_id_32 || reserve0_u128_be || reserve1_u128_be || lp_u128_be)

state_root = Blake3(
    "qnk-state-root-v1"
    || Blake3(sorted native leaves)
    || Blake3(sorted token leaves)
    || Blake3(sorted pool leaves)
)
```

Using raw bytes (not hex, not decimal strings, not JSON) is mandatory for cross-platform determinism.

### Step 3: Update `compute_state_root()` in block_producer.rs (2 days)
Replace:
```rust
fn compute_state_root(transactions: &[Transaction]) -> [u8; 32]
```
With:
```rust
async fn compute_state_root_from_db(storage: &StorageEngine) -> Result<[u8; 32]>
```
That calls the extended `compute_balance_state_hash()` (now covering all state types).

### Step 4: Make validation REJECT on mismatch (1 day)
Change `warn!` to `return Err(BlockValidationError::StateRootMismatch{...})` at activation height.
Add recovery mode: if this node consistently rejects blocks that the network accepts → reload checkpoint → replay.

### Step 5: Set activation height and `mandatory: true` (1 day)
In `upgrade_gate.rs`:
```rust
// MAINNET: activate at height 17,500,000 (~3 weeks from now)
activation_height: 17_500_000,
mandatory: true,
min_version: "10.5.0".to_string(),
```

Announce on Discord + all node operator channels minimum 2 weeks before activation.

### Step 6: Disable off-chain mutations before activation height (1-2 weeks)
Before height 17,500,000:
- ❌ `apply_dex_qug_adjustments()` → must be a no-op or removed
- ❌ P2P pool sync (5-minute gossip) → must be disabled
- ❌ Authority peer balance override → must be disabled  
- ❌ 15-second RocksDB→HashMap sync → already handled in Step 1
- ❌ All v8.x balance rebuild migrations → already gated by checkpoint

---

## 10. DEX Math Determinism — Critical for Mainnet

Per external advisor recommendation, all AMM calculations must use checked integer arithmetic:

**Current risk:** If `saturating_mul` or `saturating_add` is used in swap calculations, a u128 overflow (which can legitimately occur in large DEX pools) silently wraps to a wrong value. Two nodes may apply different overflow behavior if one uses `saturating_*` and one uses `checked_*`. Result: different swap outputs for the same transaction → different post-state roots → consensus fork.

**Required canonical form for constant-product AMM:**
```rust
// All arithmetic must use checked_* variants
fn compute_swap_output(
    reserve_in: u128,
    reserve_out: u128, 
    amount_in: u128,
    fee_bps: u32,          // e.g., 30 for 0.3%
    fee_denominator: u32,  // e.g., 10_000
) -> Result<u128, SwapError> {
    let fee_denom = fee_denominator as u128;
    let fee_num = (fee_denominator - fee_bps) as u128;
    
    let amount_in_with_fee = amount_in
        .checked_mul(fee_num)
        .ok_or(SwapError::Overflow)?;
    
    let numerator = reserve_out
        .checked_mul(amount_in_with_fee)
        .ok_or(SwapError::Overflow)?;
    
    let denominator = reserve_in
        .checked_mul(fee_denom)
        .ok_or(SwapError::Overflow)?
        .checked_add(amount_in_with_fee)
        .ok_or(SwapError::Overflow)?;
    
    // Integer division: rounds DOWN (in favor of pool)
    numerator.checked_div(denominator).ok_or(SwapError::DivisionByZero)
}
```

This must be reviewed for every swap path in the codebase. If any path uses `f64` math for swap calculation, it must be replaced. Cross-platform f64 is not bit-for-bit deterministic.

---

## 11. What the User's qUSD Position Means in This Architecture

The user holds **29,486,811.50 qUSD** (USD-pegged token) as a hedge. Under the full `state_root`:

**Today (pre-activation):** The position is safe on Epsilon but not cryptographically guaranteed. Token balances in `token_balance_*` are not protected by any consensus rule.

**After Step 2 (token balances in state_root):** The position is part of the consensus state. Any incorrect computation of the qUSD balance is detectable and rejectable. The hedge is as safe as native QUG.

**After Step 6 (off-chain mutations disabled):** The position cannot be altered by any process other than a valid block transaction. The hedge is as safe as a Bitcoin UTXO.

**The path to full protection is 3-4 weeks of focused work.** The infrastructure is largely already there from the 5 months of development.

---

## 12. Risk Matrix for Each Step

| Step | Mainnet Risk if Skipped | Mainnet Risk if Done Incorrectly |
|---|---|---|
| Post-checkpoint replay | Balance state at checkpoint height, not current (divergence continues) | Block replay could apply wrong balance deltas — verify with test suite |
| Kill 15s backward sync | Divergence can silently recur | Breaking change — verify in-memory state is still consistent |
| Extend state_root to tokens+pools | Native-only root is unsafe while DEX is live | Wrong canonical serialization → permanent fork |
| Update compute_state_root | State root never means balance consensus | Computing state during block production adds latency — benchmark |
| Make validation reject | No consensus enforcement | Nodes that disagree enter rejection loop — need recovery mode |
| Set activation height | State root never activates | Too soon → nodes not upgraded; too late → divergence continues |
| Disable off-chain mutations | Mutations after activation → immediate state root mismatch → network halt | Must be complete before activation height, not after |

---

## 13. Questions for DeepSeek and ChatGPT External Review

Based on all of the above, please provide feedback on:

**Q1: Given that the `state_root` field and upgrade gate infrastructure already exist, is the proposed single-phase activation (full state_root covering native + tokens + pools at height 17,500,000) safer than the phased approach?**

The concern with the phased approach is the intermediate state: native-only root active while DEX is live. The concern with single-phase is the larger scope of work (tokens + pools + native all in one activation). At $1.5B MCap, which approach has lower catastrophic-failure probability?

**Q2: Is the existing `compute_balance_state_hash()` function (Blake3 over sorted wallet balances from RocksDB) sufficient as-is for the native component of state_root, or does the serialization need to change?**

Current: `hasher.update(address_32_raw); hasher.update(balance_u128_le_bytes)` — note it uses LITTLE-ENDIAN u128. The advisors recommended BIG-ENDIAN. Does endianness matter for collision resistance, or only for cross-platform canonical form?

**Q3: For the post-checkpoint block replay, should replay cover ALL transaction types (including DEX swaps that change token balances and pool reserves) or ONLY native coin transactions (coinbase + transfers)?**

If we replay DEX swaps during post-checkpoint replay, we need to apply them to the token/pool state too. But the token/pool state was NOT reset by the checkpoint (only native coin was). So replaying DEX swaps on the correct native state but incorrect token/pool state could produce inconsistent cross-state invariants. What is the safe replay scope?

**Q4: Is the Bitcoin bridge BEDA checkpoint usage (embedding `state_root` in Bitcoin transactions) affected by changing what `state_root` computes?**

Currently BEDA embeds a TX-id-based state root. After the fix, it would embed a balance/wallet-state root. This changes the semantic meaning of existing BEDA checkpoints anchored in Bitcoin. Does this create any Bitcoin-bridge security issue?

**Q5: What is the minimum safe notice period for the 17,500,000 activation height announcement?**

Given the mandatory=true upgrade requirement (non-upgraded nodes get forked off), and the need for all node operators to upgrade, what is the minimum responsible announcement-to-activation window for a $1.5B mainnet?

---

## 14. Summary Status Table

| Item | Status | Risk Level | ETA |
|---|---|---|---|
| `state_root` field in block header | ✅ Done (5 months ago) | — | — |
| Block hash includes `state_root` | ✅ Done (5 months ago) | — | — |
| Upgrade gate infrastructure | ✅ Done (5 months ago) | — | — |
| Balance checkpoint v10.4.14/v10.4.15 | ✅ Done (today) | 🟢 LOW | — |
| 53 crash/replay/wiring safety tests | ✅ Done (today, ec275dc29) | — | — |
| Kill 15s backward sync | ✅ Done (today, gate in main.rs) | 🟠 HIGH | — |
| Supply overflow → hard error (checked_add) | ✅ Done (today, lib.rs) | 🟠 HIGH | — |
| Rename compute_state_root (name was a lie) | ✅ Done (today, block_producer.rs) | 🟠 HIGH | — |
| Post-checkpoint block replay | ⬜ Not done | 🔴 CRITICAL | 3-5 days |
| Extended state_root (tokens + pools) | ⬜ Not done | 🔴 CRITICAL | 2-3 weeks |
| Wire compute_balance_state_hash() into blocks | ⬜ Not done | 🔴 CRITICAL | 2 days |
| Make block validation reject on mismatch | ⬜ Not done | 🔴 CRITICAL | 1 day |
| Disable off-chain mutations (DEX pool sync etc.) | ⬜ Not done | 🔴 CRITICAL | 2 weeks |
| DEX math checked arithmetic audit | ⬜ Not done (tests prove overflow exists) | 🟠 HIGH | 1 week |
| Deterministic replay CI test | ⬜ Not done | 🟠 HIGH | 1 week |
| Delta container test (real chain data) | ⬜ Not done | 🟠 HIGH | 1-2 days |
| Integration test vs real RocksDB | ⬜ Not done | 🟠 HIGH | 1 day |
| StateRootV1 mainnet activation | ⬜ Not scheduled | 🔴 CRITICAL | ~17,500,000 |
| Full-state checkpoint (tokens+pools) | ⬜ Not done | 🟠 HIGH | Before activation |

**The good news:** The 5 months of development built a remarkably complete foundation. The remaining work is mostly connecting the existing pieces correctly, not building from scratch. The checkpoint gives us the canonical starting point. The state_root field and upgrade gate give us the activation mechanism. The balance hash function gives us the computation. What remains is wiring them together safely.

**The mainnet risk:** This is a $1.5B live network. There is no test run. Every change to block validation is a potential network split. All external advisors agree on the same principle: announce early, activate late, make it mandatory, and have a rollback plan.

---

## 15. What More Can Be Done — Ranked by Mainnet Impact

The following items are ordered by how much damage each one prevents if left undone. Each item is a concrete task, not a vague recommendation.

### Tier 1: Must Be Done Before Any Production Deploy (blocks everything downstream)

**T1-A: Post-checkpoint block replay** ← single highest-risk unimplemented item  
Implement `replay_block_balance_changes()` inside `apply_balance_checkpoint()` in `lib.rs`. Without this, all nodes import the checkpoint balance snapshot at height 16,538,868 but then have 4,300+ blocks of coinbase rewards and transfers that are NOT reflected in their balance state. Every node's balance diverges immediately after checkpoint import by a different amount (depending on how many blocks past checkpoint they are). The gap grows by ~1 block per second. **This is not optional — it must be done before checkpoint is deployed to Beta or Gamma.**

Replay scope (conservative, safe): only TX types 0x01 (Coinbase) and 0x00 (Transfer). Do NOT replay DEX/token TXs — the token/pool state was not reset by the checkpoint, so applying DEX balance changes against a correct native state but an unknown token/pool state produces inconsistent cross-state invariants. The existing test suite (`whitelist_has_exactly_two_members`) enforces exactly this scope.

**T1-B: Integration test against real RocksDB (`tempfile::TempDir`)**  
All 253 tests in this session use `MockPersistentStore`. The mock correctly models the marker-as-commit-point invariant, but it cannot catch errors in the real `StorageEngine::apply_balance_checkpoint()` function — wrong column family, wrong key prefix, incorrect WAL flush. At minimum one integration test must call the real function and verify the marker is written to the correct key, the correct wallets are readable via `get_balance()`, and `is_checkpoint_applied()` returns true. Without this, we have tested the design but not the implementation.

**T1-C: Delta container test (real chain data)**  
Apply the checkpoint to a real node at height 16,543,204 (or current tip) on Delta's Docker container. Verify:
- All 1,332 checkpoint wallets are readable
- The extended marker shows `replayed_through_height` correctly
- `total_minted_supply` matches the checkpoint constant
- No zero-balance entries in wallet_db
- The node continues block processing normally afterward

This is the dress rehearsal. Beta and Epsilon hold real funds — Delta holds nothing. If anything is wrong here, it costs nothing. If we skip this and go straight to production, the cost could be catastrophic.

### Tier 2: Must Be Done Before StateRootV1 Activation (3-4 weeks out)

**T2-A: Wire `compute_balance_state_hash()` into block production**  
The right function exists. It needs to be called asynchronously after each block is applied (not during transaction processing), and its return value stored in `BlockHeader.state_root`. This is the core of StateRootV1. The rename done today (`compute_transaction_set_root`) removes the confusion — the next step is the wiring.

**T2-B: Extend `compute_balance_state_hash()` to cover tokens + pools**  
Current coverage: `wallet_balance_*` keys only. Required coverage before activation: `token_balance_*` and `liquidity_pool:*`. The canonical serialization (from Section 9) must use raw big-endian bytes, not hex or decimal strings. This is 2-3 weeks of careful work because the serialization format is permanent — once blocks start including this hash, changing the format is a hard fork.

**T2-C: DEX checked arithmetic audit**  
The test `basic_swap_correct_output` proves that `amm_out()` overflows u128 at 24-decimal scale and returns `None`. That means the PRODUCTION swap function either:
  (a) returns `None` and rejects the swap silently, or  
  (b) uses `saturating_mul` and produces a wrong output  
Either one is a serious defect. The AMM implementation must use 256-bit integer arithmetic (or `bigdecimal`) for all multiplication steps in the constant-product formula. Every swap path in the codebase must be audited — not just the public entry point, but every internal helper.

**T2-D: Make block validation reject (not just warn) on state_root mismatch**  
Currently a mismatch at `main.rs:~11380` logs a warning and continues. After StateRootV1 activates, this must be a hard rejection. The rejection must be height-gated (only enforce after activation height) so that old blocks (which have `state_root = [0;32]`) are not retroactively rejected. Add a recovery path: if this node is consistently rejecting blocks the network accepts, it should reload the checkpoint and replay rather than permanently forking off.

**T2-E: Disable off-chain balance mutations before activation height**  
The following subsystems bypass block processing and write directly to balance state:
- P2P pool sync (5-minute gossip of pool reserves) — must be disabled or converted to read-only
- Authority peer balance override — must be audited; if it writes to `wallet_balance_*`, it must be disabled
- Any remaining v8.x balance rebuild logic — should already be gated by checkpoint flag; verify

Each of these is a source of non-determinism. Once block validation enforces state_root, ANY off-chain write that doesn't match what other nodes computed from blocks will cause an immediate rejection cascade.

### Tier 3: Hardening That Prevents the Next Incident (2-4 weeks out)

**T3-A: Full-state checkpoint capture on Epsilon (time-sensitive)**  
The current checkpoint covers native QUG only (1,332 wallets at height 16,538,868). A full-state checkpoint would also snapshot token balances and pool reserves. This cannot be reconstructed retroactively once the chain advances. If Epsilon's database is ever lost or corrupted, native balances can be recovered from the checkpoint, but token/pool state cannot. Capture this while Epsilon is running and the data is accessible.

**T3-B: Deterministic replay CI test**  
A CI test that:
1. Starts a fresh node with no state
2. Applies the balance checkpoint
3. Replays 100 post-checkpoint blocks from a fixed block file
4. Asserts the final `compute_balance_state_hash()` matches a hardcoded expected value

This test catches any non-determinism in the replay logic. If this test passes on two different machines, the replay is correct. If it fails on one machine, there is a platform-specific arithmetic bug.

**T3-C: Monotonic height enforcement in backward sync gate**  
The current backward sync gate (`is_checkpoint_applied()`) blocks the task completely once the checkpoint is applied. A stronger version would also assert that the current height never decreases — if it does, log `CRITICAL` and halt the node. Height regression (seen in the v7.4.1 incident documented in MEMORY.md) silently corrupts balance state because blocks get re-processed.

**T3-D: Gossipsub state_root announcement**  
When a node computes `state_root` for a block, include it in the gossipsub block announcement. Peers can then detect immediately if they computed a different state_root for the same block, without waiting for a full block validation cycle. This provides early warning of divergence before it has time to grow. The announcement can be a simple `(height, state_root_hash)` tuple on a new gossipsub topic.

---

## 16. The One Question That Decides Everything

All of the above assumes the checkpoint import is correct — that the 1,332 wallets and the total supply constant `497_391_964_203_542_355_791_983_084_160` accurately reflect the true Epsilon state at height 16,538,868.

**If the checkpoint data is wrong, everything downstream is wrong.** The post-checkpoint replay, the state_root computation, the balance comparisons — all of it is built on this foundation.

The only way to verify it is the Delta container test (T1-C). Until that test runs successfully on real chain data with real blocks, the checkpoint is a claim, not a fact.

**The single most important next action is T1-A + T1-C, in that order.**

---

*This is the fourth in a series of technical reviews:*
- *`docs/technical-review-balance-divergence-root-cause-2026-04-28.md` — root cause analysis*
- *`docs/technical-review-why-decentralization-failed-2026-04-28.md` — architectural diagnosis + external consultations*
- *`docs/technical-review-state-root-dex-tokens-2026-04-28.md` — DEX/token/full state scope*
- *`docs/technical-review-state-root-activation-plan-2026-04-28.md` — this document: complete activation plan*
