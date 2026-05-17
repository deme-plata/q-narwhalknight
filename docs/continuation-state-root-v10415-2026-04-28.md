# Continuation Document — State Root / Balance Checkpoint Work
**Date:** 2026-04-28  
**Version:** v10.4.15  
**Branch:** `feature/safe-batched-sync-v1.0.2`  
**Purpose:** Living reference to resume this work at any point without losing context

---

## 1. What We Are Solving and Why It Matters

The network ($1.5B mainnet) has **perfect event consensus** — DAG-Knight/Bracha/libp2p ensure all nodes agree on which blocks exist. What it does NOT have is **state consensus** — no mechanism enforces that every node computes the same wallet balances from the same blocks.

This caused Epsilon's wallet balances to diverge silently over time. The balance for wallets would drift depending on startup order, P2P sync timing, and off-chain state mutations (DEX startup adjustments, backward RocksDB→HashMap sync). Nothing in the consensus protocol could detect this.

**The emergency fix (this work):** Import the correct 1,332 native QUG wallet balances from Epsilon's authoritative DB as a hardcoded checkpoint.

**The permanent fix (in progress):** Add `state_root` to every block header — a Blake3 commitment to all consensus-relevant state (native balances, token balances, DEX pool reserves, staking, contract state). A native-only `balance_root` may be used in shadow mode or emergency diagnostics, but enforced mainnet consensus must eventually cover every type of state that block transactions can read or write.

**The user's stake in this:**
- 29,486,811.50 qUSD sitting in `token_balance_*` keys on Epsilon (NOT `wallet_balance_*`)
- The native checkpoint does not intentionally modify `token_balance_*`, so the qUSD position is not overwritten by this migration. However, until token balances and DEX pool state are included in `state_root`, the qUSD position is **not consensus-protected network-wide**. "Untouched by this migration" is not the same as "fully safe."
- The qUSD position becomes consensus-protected only in Phase 2, when token balances are included in the enforced full `state_root`
- Testing only happens on **Delta's container** — Epsilon production is never touched

---

## 2. What Has Been Built

### v10.4.14 — Balance Checkpoint (prior session)
**File:** `crates/q-storage/src/balance_checkpoint.rs`

Hardcoded snapshot of 1,332 wallet balances from Epsilon at height **16,538,868**. The checkpoint is applied exactly once (idempotent marker). On second apply, reads the marker and returns immediately.

### v10.4.15 — Checkpoint Hardening (this session)
**File:** `crates/q-storage/src/lib.rs` — function `apply_balance_checkpoint()` (~line 5475)

Three new constants added to `balance_checkpoint.rs`:
```rust
pub const CHECKPOINT_TOTAL_SUPPLY: u128 = 497_391_964_203_542_355_791_983_084_160;
pub const CHECKPOINT_SHA256: &str = "eabbeadf85d03fb3a3b3fbafb1f6928513abafaf49ffba758f42f889a3fd8009";
pub const CHECKPOINT_PREV_BLOCK_HASH_HEX: &str = "67b859c04251fa673f075697d2ded555ac2b43876666d160951bbabcf5b8e60a";
```

The `apply_balance_checkpoint()` function now:
- Warns if local height > `CHECKPOINT_HEIGHT` (gap exists, replay needed)
- Verifies `wallet_count == 1,332` else returns `Err`
- Verifies `total_supply == CHECKPOINT_TOTAL_SUPPLY` else returns `Err`
- Writes a **32-byte structured marker** instead of 8 bytes: `height(8LE) + count(8LE) + total(16LE)`
- Key: `b"__balance_checkpoint_v1__"`

**Planned marker extension (needed before replay is implemented):**
The current marker proves "checkpoint snapshot was imported at height H." It does NOT prove "checkpoint was imported AND replayed to local tip." Once block replay is implemented, the marker must be extended. The new format (105 bytes), with version byte first so parsers can branch immediately:
```
marker_version               (1 byte, 0x02 for this format)
checkpoint_height            (8 bytes LE)
checkpoint_wallet_count      (8 bytes LE)
checkpoint_total_supply      (16 bytes LE)
checkpoint_snapshot_hash     (32 bytes — Blake3 of sorted wallet entries)
replayed_through_height      (8 bytes LE)
replayed_through_block_hash  (32 bytes)
```
**Critical:** Write the new marker atomically in the same RocksDB write-batch as the final balance write of the replay — or immediately after with an explicit `sync_wal()`. If the process crashes mid-replay, the marker must remain absent (or in the old 32-byte format) so a restart triggers fresh replay. Without the snapshot hash, two different snapshots with the same height/count/supply could pass marker checks.

Update the marker key to `b"__balance_checkpoint_v2__"` and keep the old key check for backward compatibility detection.

### Technical Review Documents (this session)
| Document | Contents |
|---|---|
| `docs/technical-review-balance-divergence-root-cause-2026-04-28.md` | Why balances diverged; 5 root causes; permanent fix |
| `docs/technical-review-why-decentralization-failed-2026-04-28.md` | Why DAG-Knight/Bracha didn't prevent it; Layer 1 vs Layer 2 distinction; roadmap |
| `docs/technical-review-state-root-dex-tokens-2026-04-28.md` | How state root interacts with DEX; user's qUSD position safety; phased rollout |
| `docs/technical-review-state-root-activation-plan-2026-04-28.md` | Codebase audit; 80% existing infrastructure; 6-step safe activation sequence |

### Test Files Written (this session)
All pure-logic tests, no real RocksDB dependency. Run independently.

| File | Package | Tests | Status |
|---|---|---|---|
| `crates/q-storage/tests/balance_checkpoint_tests.rs` | q-storage | 45 | ✅ Compiles and runs |
| `crates/q-storage/tests/state_root_consensus_tests.rs` | q-storage | 44 | ✅ **44/44 PASSING** |
| `crates/q-dex/tests/state_root_dex_safety_tests.rs` | q-dex | 48 | Running |
| `crates/q-api-server/tests/state_root_activation_tests.rs` | q-api-server | 44 | Compiling |

**Total: 181 tests across 4 files.**

Run any suite with:
```bash
timeout 36000 cargo test --package q-storage --test state_root_consensus_tests
timeout 36000 cargo test --package q-storage --test balance_checkpoint_tests
timeout 36000 cargo test --package q-dex --test state_root_dex_safety_tests
timeout 36000 cargo test --package q-api-server --test state_root_activation_tests
```

---

## 3. The Critical Discovery: 80% Already Built

Found during codebase audit (see `technical-review-state-root-activation-plan-2026-04-28.md`):

| Component | Location | Status |
|---|---|---|
| `state_root: [u8; 32]` field | `crates/q-types/src/block.rs:239` | ✅ Exists in BlockHeader |
| `StateRootV1` upgrade gate | `crates/q-consensus-guard/src/upgrade_gate.rs:119` | ✅ Exists, `activation_height = u64::MAX` |
| `compute_balance_state_hash()` | `crates/q-storage/src/lib.rs:4369` | ✅ Exists, correct Blake3 implementation |
| Block hash commits `state_root` | `crates/q-types/src/block.rs` | ✅ Already included in hash |
| Block producer uses upgrade gate | `crates/q-api-server/src/block_producer.rs:954` | ✅ Already checks gate |

**What is WRONG:**
- `compute_state_root()` at `block_producer.rs:2021` hashes TX IDs — NOT balance state
- It is called `compute_state_root` but it is actually a transaction commitment
- Must be renamed `compute_transaction_set_root()` and replaced

**The 20% gap:**
1. The function computes the wrong thing
2. Block validation doesn't enforce state_root (warns only)
3. Post-checkpoint block replay not implemented

---

## 4. Known Architecture Facts

### Key RocksDB Prefixes (critical for safety)
```
wallet_balance_{address}     → native QUG coin holdings (checkpoint touches THIS)
token_balance_{token}_{addr} → DEX token balances (qUSD lives here — UNTOUCHED)
liquidity_pool:{a}:{b}       → DEX pool reserves (UNTOUCHED by checkpoint)
contract_{id}                → smart contract state (UNTOUCHED)
stake_position_{addr}        → staking positions (UNTOUCHED)
```

### compute_balance_state_hash() — the correct implementation
`crates/q-storage/src/lib.rs:4369`
```rust
pub async fn compute_balance_state_hash(&self) -> Result<([u8; 32], usize, u128)> {
    let balances = self.load_wallet_balances().await?;
    let mut sorted_entries: Vec<_> = balances.iter()
        .filter(|(_, &amount)| amount > 0)
        .collect();
    sorted_entries.sort_by_key(|(addr, _)| *addr);
    let mut hasher = blake3::Hasher::new();
    let mut wallet_count = 0usize;
    let mut total_supply = 0u128;
    for (addr, &amount) in &sorted_entries {
        hasher.update(addr.as_slice());
        hasher.update(&amount.to_le_bytes());
        wallet_count += 1;
        total_supply = total_supply.saturating_add(amount);
    }
    let hash: [u8; 32] = *hasher.finalize().as_bytes();
    Ok((hash, wallet_count, total_supply))
}
```

### StateRootV1 Upgrade Gate — mainnet vs testnet
```
MAINNET: activation_height = u64::MAX  ← MUST NOT CHANGE until all 6 prerequisites met
TESTNET: activation_height = 0         ← Active immediately for testing
```

### Block producer state_root logic (block_producer.rs:954)
```rust
let state_root = if q_consensus_guard::is_upgrade_active(
    q_consensus_guard::Upgrade::StateRootV1,
    next_height,
) {
    Self::compute_state_root(&all_transactions)  // ← WRONG: hashes TX IDs
} else {
    [0u8; 32]
};
```

### compute_state_root() — the WRONG current implementation (block_producer.rs:2021)
```rust
fn compute_state_root(transactions: &[Transaction]) -> [u8; 32] {
    // SHA3-256 of sorted TX IDs — this is a TX commitment, NOT a balance root!
    // Must be renamed compute_transaction_set_root() and replaced
}
```

### Post-Checkpoint Height Gap
At the time of the last test session:
- Checkpoint height: 16,538,868
- Test container height: ~16,543,204
- Gap: ~4,336 blocks

These blocks contain native QUG coinbase rewards and transfers. After checkpoint import, these balance changes are lost unless block replay is implemented.

---

## 5. The Next Three Code Changes (in order)

### Change 1: Rename compute_state_root()
**File:** `crates/q-api-server/src/block_producer.rs`

Rename the function at line 2021 from `compute_state_root` to `compute_transaction_set_root`. Update the call at line 958.

This is a safety measure: prevents accidental activation of the wrong computation.

```rust
// Before (wrong name, wrong content):
Self::compute_state_root(&all_transactions)

// After (correct name, will be replaced with balance hash later):
Self::compute_transaction_set_root(&all_transactions)
```

### Change 2: Gate backward sync behind checkpoint flag
**File:** `crates/q-api-server/src/main.rs` (wherever the 15-second backward sync runs)

Find the task that reads all `wallet_balance_*` keys from RocksDB and rebuilds the in-memory HashMap. Wrap it with:
```rust
if !checkpoint_applied {
    // run backward sync
}
```

Without this gate, the backward sync would overwrite checkpoint data with potentially stale values from RocksDB if the checkpoint was applied mid-run.

**This gate is temporary.** The medium-term goal is to delete the backward sync entirely. The in-memory HashMap must be a read cache, never a state corrector. After the checkpoint is active, the only acceptable balance writes are through block transaction replay. The backward sync is an architectural debt that prevents the node from being fully deterministic.

### Change 3: Post-checkpoint block replay
**File:** `crates/q-storage/src/lib.rs` or `crates/q-api-server/src/main.rs`

After `apply_balance_checkpoint()` returns `Ok(true)` (applied freshly), replay native-coin-only state changes from blocks `CHECKPOINT_HEIGHT + 1` through the current chain tip:

```
For each block from CHECKPOINT_HEIGHT+1 to tip:
  For each transaction in block:
    If tx is coinbase: add amount to wallet_balance_{recipient}
    If tx is native transfer: subtract from sender, add to recipient
    Skip ALL other transaction types (see below)
```

**Strict exclusion list — do NOT replay anything that touches:**
- `token_balance_*` (DEX token balances, qUSD, etc.)
- `liquidity_pool:*` (pool reserves)
- LP token balances
- `contract_*` (smart contract state)
- `stake_position_*` (staking)
- DEX fee accumulator state
- Any other prefix not explicitly in the checkpoint

**Why this exclusion is mandatory:** The checkpoint only reset `wallet_balance_*`. If the replay function touches other prefixes, it applies partial state changes to a baseline that was never reset — creating cross-state inconsistencies that are worse than the original divergence.

**Critical implementation rule:** Build a purpose-built native-coin-only replay function. Do NOT call the normal full block executor with any flag or filter. The full executor has complex side effects and it is too easy to accidentally pass a transaction through that touches non-checkpointed state. Write a strict whitelist, not a blacklist.

**After replay, token/pool state may still be divergent.** This is acceptable only because token/pool state is not yet consensus-enforced. It must be clearly logged.

**Edge case — hybrid transactions:** If any post-H block contains a DEX/native hybrid transaction whose native side cannot be cleanly separated from token/pool side effects, skip it entirely and log it for manual review. A partial DEX replay producing an inconsistent intermediate state is worse than missing a small post-checkpoint native delta.

**Mitigation for ongoing pool drift:** Once the native checkpoint and replay have been validated on Delta's container, consider disabling the 5-minute P2P pool gossip sync even before Phase 2. This stops further pool state drift at little cost — pools will only change through block transactions on that node from that point forward.

The full-state checkpoint (v10.5.x) will overwrite all divergent token/pool values before `state_root` is expanded to cover them.

---

## 6. The 6 Prerequisites Before Changing u64::MAX

These must ALL be complete before setting `StateRootV1` mainnet `activation_height` to anything other than `u64::MAX`:

| # | Prerequisite | Status |
|---|---|---|
| 1 | `compute_state_root()` renamed + replaced with balance hash | ❌ Not done |
| 2 | Block validation enforces wrong `state_root` → reject (not just warn) | ❌ Not done |
| 3 | Post-checkpoint block replay implemented and tested | ❌ Not done |
| 4 | Testnet soak with StateRootV1 active, shadow mode, ≥14 days zero mismatches | ❌ Not done |
| 5 | ≥6 weeks public upgrade notice (Discord, BitcoinTalk, quillon.xyz) | ❌ Not done |
| 6 | ≥2/3 of nodes by stake have upgraded | ❌ Not done |

**A test exists for this:** `state_root_activation_tests.rs::upgrade_gate_config::mainnet_state_root_v1_activation_is_u64_max` — if anyone accidentally changes the constant, this test fails immediately.

---

## 7. The Phased Roadmap

### Phase 0 — Native Coin Checkpoint (current)
Emergency fix. Hardcoded snapshot of 1,332 wallets.

Remaining: rename function, block replay, backward sync gate, deploy to Delta's container, commit to git.

### Phase 1 — Native balance root, shadow mode ONLY
Wire `compute_balance_state_hash()` into block production. Log the computed root alongside each block. Do NOT enforce it — do not reject blocks with wrong roots.

**Critical constraint:** Do not activate native-only `balance_root` enforcement on mainnet while DEX/token transactions are live, unless DEX/token writes are disabled. Native QUG balances are affected by DEX swaps (e.g., fee distributions that touch `wallet_balance_*`). If those off-chain mutations still exist and the enforcement activates, nodes with different DEX state will compute different native roots and split from the network. Shadow mode first, enforcement only after Phase 2 is ready.

Shadow mode → 14-day testnet soak → assess whether native-only enforcement is safe given live DEX, or whether enforcement must wait for Phase 2.

**Timeline: ~5 weeks from v10.4.15 to shadow mode**

### Phase 2 — Full state root (native + token + pool)
`state_root` = Blake3(native_coin_root || token_root || dex_root). All state types covered simultaneously. This is the correct activation target for mainnet enforcement. Enforcement of a partial root (native only) while DEX is live is more dangerous than waiting for full coverage.

Requires:
- Full-state snapshot from Epsilon (token balances + pool reserves at a fixed height)
- All pool state changes through on-chain transactions (remove P2P pool gossip)
- All DEX startup adjustments removed
- 6-week mainnet announcement

**Note on qUSD:** The user's 29,486,811.50 qUSD gets cryptographic protection at Phase 2, when token balances are included in the enforced full `state_root`.

**Timeline: ~8–12 weeks after Phase 1 shadow mode is stable**

### Phase 3 — Remove all off-chain state mutations
- No startup DEX adjustments
- No backward HashMap sync (already gated, eventually removed)
- No P2P balance updates (replaced by state root enforcement)
- All state: block transactions only

---

## 8. What Testing Infrastructure Exists

### Test Containers
- **Delta's container** (`qnk-sync-test-v4` on Epsilon port 8085) — the ONLY place to test checkpoint activation. It is isolated, has no real funds.
- **Epsilon** (`89.149.241.126:8080`) — production node, never test on this
- **Beta** (`185.182.185.227:8080`) — production node, use `ha-deploy.sh` for deploys

### Running Tests
```bash
# State root consensus (Blake3 determinism, upgrade gate config)
timeout 36000 cargo test --package q-storage --test state_root_consensus_tests

# Balance checkpoint (1,332 wallets, total supply, idempotency, key isolation)
timeout 36000 cargo test --package q-storage --test balance_checkpoint_tests

# DEX safety (AMM math, overflow protection, cross-state invariants)
timeout 36000 cargo test --package q-dex --test state_root_dex_safety_tests

# Activation safety (prerequisites, shadow mode, atomic validation, BEDA versioning)
timeout 36000 cargo test --package q-api-server --test state_root_activation_tests

# Existing critical suites
timeout 36000 cargo test --package q-storage --test balance_checkpoint_tests
timeout 36000 cargo test --package q-storage --test balance_propagation_tests
timeout 36000 cargo test --package q-dex --test overflow_protection_tests
```

### Key Test: Regression Guard for u64::MAX
```
cargo test --package q-api-server --test state_root_activation_tests \
  upgrade_gate_config::mainnet_state_root_v1_activation_is_u64_max
```
This test MUST pass before every deploy. If it fails, someone changed the activation height without completing the 6 prerequisites.

---

## 9. Key File Locations

| What | Path |
|---|---|
| Checkpoint data (1,332 wallets) | `crates/q-storage/src/balance_checkpoint.rs` |
| Checkpoint apply logic | `crates/q-storage/src/lib.rs` ~line 5475 |
| Balance state hash | `crates/q-storage/src/lib.rs:4369` |
| Wrong compute_state_root (to rename) | `crates/q-api-server/src/block_producer.rs:2021` |
| Block producer state_root call | `crates/q-api-server/src/block_producer.rs:954` |
| BlockHeader with state_root field | `crates/q-types/src/block.rs:239` |
| Upgrade gate (mainnet/testnet configs) | `crates/q-consensus-guard/src/upgrade_gate.rs:96–175` |
| Test: checkpoint | `crates/q-storage/tests/balance_checkpoint_tests.rs` |
| Test: state root consensus | `crates/q-storage/tests/state_root_consensus_tests.rs` |
| Test: DEX safety | `crates/q-dex/tests/state_root_dex_safety_tests.rs` |
| Test: activation | `crates/q-api-server/tests/state_root_activation_tests.rs` |

---

## 10. Constants to Never Change Without Knowing Why

```rust
// balance_checkpoint.rs
CHECKPOINT_HEIGHT: u64 = 16_538_868          // Epsilon snapshot height (H)
CHECKPOINT_WALLET_COUNT: usize = 1_332       // Must match file exactly
CHECKPOINT_TOTAL_SUPPLY: u128 = 497_391_964_203_542_355_791_983_084_160
CHECKPOINT_SHA256: &str = "eabbeadf85d03fb3a3b3fbafb1f6928513abafaf49ffba758f42f889a3fd8009"
// ↑ SHA-256 of the snapshot ARTIFACT FILE (file integrity check, not state identity).
//   Better name: CHECKPOINT_ARTIFACT_SHA256.
//   NOT the same as checkpoint_snapshot_hash in the v2 marker (that is Blake3 of sorted wallet entries).
//   Better name for the marker field constant: CHECKPOINT_STATE_BLAKE3.

CHECKPOINT_PREV_BLOCK_HASH_HEX: &str = "67b859c04251fa673f075697d2ded555ac2b43876666d160951bbabcf5b8e60a"
// ↑ Hash of block H-1 (the PARENT of the checkpoint block — NOT the checkpoint block itself).
//   The import function MUST verify the local chain contains this block hash before
//   accepting the checkpoint. If absent, abort — the snapshot is from a different fork.

// upgrade_gate.rs — MAINNET
StateRootV1 activation_height = u64::MAX    // ← NEVER CHANGE without 6 prerequisites

// upgrade_gate.rs — TESTNET
StateRootV1 activation_height = 0           // Active immediately for testing
```

**TODO in `compute_balance_state_hash()`:** The function uses `saturating_add` for the total supply accumulator. This silently caps overflow instead of failing. For consensus-adjacent validation, change to `checked_add` and return an error on overflow — an impossible total supply is a bug that should be surfaced loudly, not masked.
```rust
// Current (masks overflow):
total_supply = total_supply.saturating_add(amount);

// Correct (surfaces impossible state):
total_supply = total_supply.checked_add(amount)
    .ok_or(anyhow::anyhow!("total supply overflow — impossible state"))?;
```

---

## 11. External Consultation Summary

Three rounds of external review (DeepSeek + ChatGPT executive review, 2026-04-28).

**Consensus from all reviewers:**
1. The `state_root` mechanism is necessary and architecturally correct
2. u64::MAX activation height is the correct safety posture for now
3. DEX startup adjustments must be removed BEFORE `state_root` is enforced (or they'll cause root mismatch and kick the node off the network)
4. Shadow mode for ≥14 days is mandatory before enforcement on mainnet
5. The 80% already-built infrastructure (state_root field, upgrade gate, compute function) is real — this is not starting from zero
6. **Native-only `balance_root` enforcement is NOT safe while DEX is live.** Native QUG balances can be affected by DEX fee distributions. Enforcing a native-only root while token/pool state is off-chain produces a partial truth that can still cause network splits. The safer path is: shadow mode on native root → validate correctness → activate full `state_root` covering all state types simultaneously.
7. **Capture a full-state snapshot from Epsilon** at a new fixed height F (as soon as operationally possible): token balances, DEX pool reserves, LP balances, staking positions, contract state, and native balances. Store this archive even before embedding it in the binary — it cannot be reconstructed once the chain advances and transactions alter those values.

   **Do not claim the token/pool snapshot corresponds to H = 16,538,868** unless Epsilon can provably rewind to that exact block. If only current RocksDB is available, capture at current height and designate a new F.

   Three-height model for clarity:
   ```
   H = 16,538,868 — native checkpoint height (wallet_balance_* only)
   F = TBD        — full-state checkpoint height (all state types)
   A = TBD        — full state_root activation height (≥ 6 weeks after F is embedded)
   ```

**Key quote from Advisor A:**
> "Any mutation outside the block processing pipeline will immediately cause a root mismatch. The team should ensure all DEX adjustments are on-chain before the activation height."

**Key correction from fourth review (2026-04-28):**
> "Native-only balance_root is useful for emergency native QUG stabilization, but mainnet consensus enforcement should not activate native-only root while DEX/token transactions are live unless DEX/token writes are disabled. The safer long-term path is full state_root covering native + token + pool state."

---

## 12. What to Do Right Now When Resuming

1. **Check test results** (if not yet finished):
   ```bash
   tail -20 /tmp/test-balance-checkpoint.log
   tail -20 /tmp/test-dex-safety.log
   tail -20 /tmp/test-activation.log
   ```

2. **Rename `compute_state_root()` first** — before any other state-root work:
   - `crates/q-api-server/src/block_producer.rs` line 2021: rename to `compute_transaction_set_root`
   - line 958: update call site
   - Add a doc comment to the renamed function explaining it is a TX commitment, not a state root, and will be replaced by `compute_balance_state_hash()` when `StateRootV1` is ready
   - This prevents any future developer from accidentally wiring the TX-root into consensus thinking it is a balance root

3. **Gate backward sync** (emergency protection):
   - Find the 15-second backward sync loop in `main.rs`
   - Wrap with `if !checkpoint_applied { ... }`
   - Add a TODO comment: "medium-term: delete this entirely; HashMap must be a cache, not a corrector"

4. **Extend checkpoint marker** to include replay tip (before implementing replay):
   - New marker format (105 bytes total), version byte first so parsers can branch without heuristics:
     ```
     marker_version               (1 byte, 0x02)
     checkpoint_height            (8 bytes LE)
     checkpoint_wallet_count      (8 bytes LE)
     checkpoint_total_supply      (16 bytes LE)
     checkpoint_snapshot_hash     (32 bytes — Blake3 of sorted wallet entries)
     replayed_through_height      (8 bytes LE)
     replayed_through_block_hash  (32 bytes)
     Total: 105 bytes
     ```
   - `checkpoint_snapshot_hash` is the Blake3 canonical hash of the sorted snapshot.
     **Do NOT confuse with `CHECKPOINT_SHA256`** — that is the SHA-256 file integrity
     checksum of the snapshot artifact (used to detect file corruption on disk).
     Clearer names: `CHECKPOINT_ARTIFACT_SHA256` for the file checksum,
     `CHECKPOINT_STATE_BLAKE3` for the canonical state identity hash embedded in the marker.
   - Update marker key to `b"__balance_checkpoint_v2__"`; keep v1 key check for backward-compat detection
   - Update the idempotency check to read and verify the extended marker

5. **Verify chain identity before applying or trusting the checkpoint**:
   - Before importing the checkpoint (and especially before purging existing balances),
     verify that the local chain contains block H-1 with hash equal to `CHECKPOINT_PREV_BLOCK_HASH_HEX`
   - This check MUST be done programmatically in the import function, not just documented
   - If the local chain does not contain that block hash, the checkpoint was taken from a
     different chain or fork — **abort the import**
   - `CHECKPOINT_PREV_BLOCK_HASH_HEX` is the hash of block number H-1, i.e., the parent
     of the checkpoint block. It anchors the snapshot to a specific chain identity.

6. **Implement native-coin-only block replay**:
   - Purpose-built function — NOT the full block executor
   - Strict whitelist: only coinbase and native transfer transactions
   - Touches only `wallet_balance_*` keys
   - After replay, write the extended marker with `replayed_through_height` and `replayed_through_hash`
   - Log a warning that token/pool state may still be divergent

7. **Commit everything to git**:
   ```bash
   git add crates/q-storage/tests/ crates/q-dex/tests/ crates/q-api-server/tests/ \
           crates/q-storage/src/ crates/q-api-server/src/ \
           docs/continuation-state-root-v10415-2026-04-28.md
   git commit -m "feat(checkpoint): v10.4.16 — replay, extended marker, backward sync gate + 181 safety tests"
   git update-server-info
   ```

8. **Run full test suite** before any deployment:
   ```bash
   timeout 36000 cargo test --package q-storage --test balance_checkpoint_tests
   timeout 36000 cargo test --package q-storage --test state_root_consensus_tests
   timeout 36000 cargo test --package q-dex --test state_root_dex_safety_tests
   timeout 36000 cargo test --package q-api-server --test state_root_activation_tests
   ```

9. **Deploy to Delta's container only** — never Epsilon production:
   - Build: `cargo build --release --package q-api-server`
   - SCP to Delta's container on Epsilon port 8085
   - Apply checkpoint, verify balances match expected values
   - Verify extended marker written correctly
   - Verify token/pool state is unchanged

10. **Then and only then**: consider production deployment via `ha-deploy.sh`

---

*This is a living document. Update the status checkboxes in Section 5 and Section 6 as work progresses.*
