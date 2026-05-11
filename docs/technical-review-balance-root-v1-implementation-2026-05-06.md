# Technical Review: BalanceRootV1 Implementation Plan
**Date:** 2026-05-06
**Author:** Server Beta (Claude Code)
**Branch:** `feature/safe-batched-sync-v1.0.2`
**Status:** Draft — For DeepSeek Review
**Classification:** MAINNET SAFETY — Activation at height 18,600,000 (~14 days from tip at ~17.3M)
**Predecessor documents:**
- `docs/technical-review-state-root-activation-plan-2026-04-28.md` — full activation plan
- `docs/continuation-state-root-v10415-2026-04-28.md` — living reference
- `docs/technical-review-balance-finality-bracha-rb-2026-05-01.md` — Bracha RB balance finality
- `docs/technical-review-genesis-sync-balance-integrity-2026-05-06.md` — genesis sync fixes

---

## 1. Executive Summary

The upgrade gate `BalanceRootV1` (Upgrade id = 9) is already registered in the mainnet schedule at activation height **18,600,000** with `mandatory: true` and `min_version: "10.6.0"`. The computation function `compute_balance_root_for_block()` exists and is correct. What is not yet done: wiring the function into block production and block validation, writing a test suite to verify cross-node determinism, and establishing the soak-test baseline before enforcement activates.

This document covers the complete implementation plan for Phase 1 of BalanceRootV1: shadow mode (log the root, do not reject), running from now through height 18,600,000. Enforcement (Phase 2) activates automatically at the gate height. The 14-day testnet soak uses Delta's Docker environment, keeping Beta, Gamma, and Epsilon untouched and fully serving production traffic throughout.

---

## 2. What Already Exists

### 2.1 Upgrade Gate Configuration

**File:** `crates/q-consensus-guard/src/upgrade_gate.rs`

```rust
// MAINNET (line 139):
upgrades.insert(Upgrade::BalanceRootV1, UpgradeConfig {
    activation_height: 18_600_000,  // ~2 weeks from current tip
    description: "Enforce balance state root in block headers".to_string(),
    mandatory: true,
    min_version: "10.6.0".to_string(),
});

// TESTNET (line 187):
upgrades.insert(Upgrade::BalanceRootV1, UpgradeConfig {
    activation_height: 0,           // Immediate for testnet
    description: "Enforce balance state root in block headers".to_string(),
    mandatory: true,
    min_version: "10.6.0".to_string(),
});
```

The gate is registered. No code change needed to the gate itself.

### 2.2 Computation Function

**File:** `crates/q-storage/src/lib.rs` (~line 4426)

```rust
pub async fn compute_balance_root_for_block(&self) -> Result<[u8; 32]> {
    let balances = self.load_wallet_balances().await?;
    let mut sorted: Vec<([u8; 32], u128)> = balances
        .into_iter()
        .filter(|(_, amount)| *amount > 0)
        .collect();
    if sorted.is_empty() {
        return Ok([0u8; 32]);
    }
    sorted.sort_by_key(|(addr, _)| *addr);

    let mut root_hasher = blake3::Hasher::new();
    root_hasher.update(b"balance_root_v1"); // domain separator

    for (addr, amount) in &sorted {
        let mut leaf_hasher = blake3::Hasher::new();
        leaf_hasher.update(addr.as_slice());
        leaf_hasher.update(&amount.to_be_bytes()); // big-endian per spec
        let leaf = leaf_hasher.finalize();
        root_hasher.update(leaf.as_bytes());
    }
    Ok(*root_hasher.finalize().as_bytes())
}
```

This is the canonical implementation: domain separator, big-endian balance encoding, Blake3 leaf-then-root structure. It is intentionally different from `compute_balance_state_hash()` (which uses little-endian and no domain separator — that function is legacy and used only for internal diagnostics).

### 2.3 Distinct from StateRootV1

`StateRootV1` (Upgrade id = 7) remains at `u64::MAX` on mainnet — it requires full state coverage including token balances and DEX pool reserves, which is months away. `BalanceRootV1` (Upgrade id = 9) covers native QUG balances only and is ready to activate at 18,600,000. The two are independent upgrade gates with independent activation heights.

### 2.4 Existing Bracha Finality Engine

The `BalanceFinalityEngine` (committed 2026-05-01, Bracha-RB over DAG-Knight) runs in `f=0` shadow mode. It gossips balance proposals over `/qnk/mainnet-genesis/consensus/balance-rb` and logs deliveries. It does NOT write to the `balance_finality` CF yet (delivery is instant in f=0 mode). This infrastructure is complementary to BalanceRootV1: Bracha finality ensures cross-node balance agreement for out-of-block credits; BalanceRootV1 enforces that agreement is reflected in block headers.

---

## 3. What Is Not Yet Done

| Gap | Impact | Effort |
|-----|--------|--------|
| `compute_balance_root_for_block()` not called in block producer | Root is never set in headers; enforcement at 18,600,000 would immediately reject all blocks | 1 day |
| Block validation does not check `balance_root` field | Even if producers set it, validators ignore it until enforcement logic is wired | 1 day |
| No `balance_determinism_tests.rs` test suite | Cannot verify cross-node hash agreement before activating enforcement | 2 days |
| Shadow-mode logging not in place | No way to detect mismatches before they become consensus failures | 4 hours |
| `balance_root` not in `/api/v1/health` response | No monitoring visibility into root state | 2 hours |

---

## 4. Implementation Plan

### 4.1 Phase 1: Shadow Mode (v10.6.0) — Ship Before Height ~18,000,000

Shadow mode: the block producer computes and sets `balance_root` in the header. Validators log a warning on mismatch. No blocks are rejected. This gives a minimum 600,000-block observation window (~7 days) before enforcement at 18,600,000.

#### 4.1.1 Block Producer Wiring

**File:** `crates/q-api-server/src/block_producer.rs` (~line 954)

**Current:**
```rust
let state_root = if q_consensus_guard::is_upgrade_active(
    q_consensus_guard::Upgrade::StateRootV1,
    next_height,
) {
    Self::compute_transaction_set_root(&all_transactions)
} else {
    [0u8; 32]
};
```

**After (add balance_root alongside state_root):**
```rust
// Existing StateRootV1 (unchanged — still at u64::MAX on mainnet)
let state_root = if q_consensus_guard::is_upgrade_active(
    q_consensus_guard::Upgrade::StateRootV1,
    next_height,
) {
    Self::compute_transaction_set_root(&all_transactions)
} else {
    [0u8; 32]
};

// BalanceRootV1: shadow mode starts at first block where gate is active.
// At height < 18,600,000: gate inactive, root = [0;32].
// At height >= 18,600,000: gate active, compute real root.
let balance_root = if q_consensus_guard::is_upgrade_active(
    q_consensus_guard::Upgrade::BalanceRootV1,
    next_height,
) {
    match storage_engine.compute_balance_root_for_block().await {
        Ok(root) => {
            info!(
                "🌿 [BALANCE ROOT] Height {}: root={}, wallets={}",
                next_height,
                hex::encode(&root[..8]),
                storage_engine.wallet_count_from_db().await.unwrap_or(0)
            );
            root
        }
        Err(e) => {
            error!("🚨 [BALANCE ROOT] compute failed at height {}: {}", next_height, e);
            [0u8; 32]
        }
    }
} else {
    [0u8; 32]
};
```

The `BlockHeader` struct already has a `state_root: [u8; 32]` field. In v10.6.0, `balance_root` is written to that field when `BalanceRootV1` is active (repurposing the existing field), OR a new `balance_root: [u8; 32]` field is added to `BlockHeader`. The choice depends on whether changing `state_root` to mean "balance root" during the window before `StateRootV1` activates creates confusion.

**Recommended:** Add a new `balance_root: [u8; 32]` field to `BlockHeader`. This avoids semantic ambiguity: `state_root` remains reserved for the eventual full-state root; `balance_root` is the native-coin-only commitment active at 18,600,000. The field already needs to participate in `calculate_hash()` for consensus — add it there.

#### 4.1.2 Block Validation (Shadow Mode — Log, Don't Reject)

**File:** `crates/q-api-server/src/main.rs` (block validation section, ~line 11380)

```rust
// Shadow mode: check balance_root but only log, never reject.
// Enforcement begins when BalanceRootV1 activates at height 18,600,000.
if q_consensus_guard::is_upgrade_active(
    q_consensus_guard::Upgrade::BalanceRootV1,
    block.height(),
) {
    match storage_engine.compute_balance_root_for_block().await {
        Ok(computed) => {
            if computed != block.header.balance_root {
                if block.header.balance_root == [0u8; 32] {
                    // Peer did not set balance_root — old binary.
                    warn!(
                        "⚠️  [BALANCE ROOT] Block {} from peer {} has balance_root=[0;32] \
                         — peer may be running pre-v10.6.0 binary. Advisory only.",
                        block.height(),
                        peer_id
                    );
                } else {
                    error!(
                        "💥 [BALANCE ROOT MISMATCH] Height {}: our_root={}, block_root={} \
                         — CONSENSUS DIVERGENCE DETECTED. Reporting only (enforcement pending).",
                        block.height(),
                        hex::encode(&computed[..8]),
                        hex::encode(&block.header.balance_root[..8])
                    );
                    // TODO (v10.7.0 / enforcement):
                    // return Err(BlockValidationError::BalanceRootMismatch { ... });
                }
            } else {
                debug!(
                    "✅ [BALANCE ROOT] Block {} root verified: {}",
                    block.height(),
                    hex::encode(&computed[..8])
                );
            }
        }
        Err(e) => {
            warn!("[BALANCE ROOT] Could not compute root for validation at height {}: {}", block.height(), e);
        }
    }
}
```

#### 4.1.3 Health Endpoint Exposure

**File:** `crates/q-api-server/src/handlers.rs` (health handler)

Add to the health response struct:
```rust
pub struct HealthResponse {
    // ... existing fields ...
    pub balance_state_hash:    String,  // existing (legacy, LE, no domain sep)
    pub balance_root_v1:       Option<String>,  // new: hex of compute_balance_root_for_block()
    pub balance_root_active:   bool,    // true when height >= 18,600,000
    pub wallet_count:          usize,
    pub total_supply_qug:      f64,
}
```

This is what the `q-balance-root-v1-test` container's health check compares against Epsilon to verify hash agreement before activation.

---

### 4.2 Phase 2: Enforcement (v10.7.0) — Activates Automatically at Height 18,600,000

At height 18,600,000 the upgrade gate fires. The change from warning to rejection is a 3-line change:

```rust
// Change the shadow-mode warning path to a hard rejection:
if computed != block.header.balance_root {
    error!("🚨 [BALANCE ROOT MISMATCH] Height {}: consensus failure", block.height());
    return Err(BlockValidationError::BalanceRootMismatch {
        height:    block.height(),
        expected:  computed,
        actual:    block.header.balance_root,
    });
}
```

This change goes into v10.7.0. The rule: v10.6.0 must be running on all nodes BEFORE 18,600,000 or those nodes get forked off when mandatory enforcement begins.

---

## 5. Test Suite: `balance_determinism_tests.rs`

**File:** `crates/q-storage/tests/balance_determinism_tests.rs` (new)

The purpose of this suite is to prove that `compute_balance_root_for_block()` is:
1. Deterministic across restarts on the same node
2. Independent of insertion order
3. Sensitive to single-unit balance changes
4. Consistent with the domain separator and big-endian encoding spec

```rust
// Test 1: Empty state returns [0;32]
#[test]
fn empty_state_returns_zero_root() {
    // Given: no non-zero balances
    // When: compute_balance_root_for_block()
    // Then: [0u8; 32]
}

// Test 2: Single wallet, deterministic
#[test]
fn single_wallet_root_is_deterministic() {
    // Same address + amount → same root, always
}

// Test 3: Order independence
#[test]
fn root_is_insertion_order_independent() {
    // Insert A then B → same root as insert B then A
}

// Test 4: 1-unit sensitivity
#[test]
fn root_changes_on_one_unit_balance_delta() {
    // Root(wallet=1000 units) ≠ Root(wallet=1001 units)
}

// Test 5: Domain separator correctness
#[test]
fn root_uses_balance_root_v1_domain_separator() {
    // Manually compute Blake3("balance_root_v1" || leaf_hash) and verify match
}

// Test 6: Big-endian encoding
#[test]
fn root_uses_big_endian_balance_encoding() {
    // Build root manually with to_be_bytes() and verify it matches function output
}

// Test 7: Determinism after restart (simulated via fresh storage)
#[test]
fn root_is_same_after_simulated_restart() {
    // Apply same balances to two independent storage instances → same root
}

// Test 8: Cross-node agreement simulation
#[test]
fn two_nodes_with_same_chain_produce_same_root() {
    // Node A and Node B both process blocks [genesis, B1, B2]
    // Apply identical coinbase rewards and transfers
    // compute_balance_root_for_block() must return identical [u8; 32]
}

// Test 9: No zero-balance wallets in root
#[test]
fn zero_balance_wallets_excluded_from_root() {
    // Wallet with balance=0 must not affect the root
    // Root({A: 1000, B: 0}) == Root({A: 1000})
}

// Test 10: 1332-wallet scale (checkpoint wallet count)
#[test]
fn root_stable_at_checkpoint_scale() {
    // 1332 wallets at checkpoint amounts → root matches hardcoded expected
    // (This test pins the root so any future regression is immediately visible)
}
```

Run with:
```bash
timeout 36000 cargo test --package q-storage --test balance_determinism_tests
```

---

## 6. Hash Verification Protocol: Delta vs. Epsilon

Before activation at 18,600,000, the `balance_root` produced by Delta's test container must match Epsilon's. The verification steps are:

### 6.1 Current Baseline (from `q-sync-bracha-rb` on Delta)

| Field | Value |
|-------|-------|
| Container | `q-sync-bracha-rb` on Delta (5.79.79.158) |
| Height | 17,321,063 |
| Wallet count | 1,340 |
| `balance_state_hash` | `51447d38626352b8...` (legacy LE, no domain sep) |
| `balance_root_v1` | NOT YET COMPUTED (v10.6.0 not deployed to container) |

The legacy `balance_state_hash=51447d38...` must be verified against Epsilon's value before the BalanceRootV1 activation. If these diverge, the balance state itself is divergent — activating enforcement on divergent state would immediately fork the network.

**Verification command:**
```bash
# On Beta (queries both):
ssh root@5.79.79.158 "docker exec q-sync-bracha-rb curl -s http://localhost:8080/api/v1/health" \
  | python3 -c "import json,sys; d=json.load(sys.stdin)['data']; print('Delta hash:', d.get('balance_state_hash','?'), 'wallets:', d.get('wallet_count','?'))"

ssh root@89.149.241.126 "curl -s http://localhost:8080/api/v1/health" \
  | python3 -c "import json,sys; d=json.load(sys.stdin)['data']; print('Epsilon hash:', d.get('balance_state_hash','?'), 'wallets:', d.get('wallet_count','?'))"
```

**Expected outcome before activation:** Both hashes match. If they do not, the divergence must be diagnosed and corrected before setting 18,600,000 as the enforcement height.

---

## 7. What This Does NOT Change

- **StateRootV1** (`u64::MAX` on mainnet) — unaffected. Remains disabled until full-state coverage (native + token + pool) is ready.
- **DEX/token balances** — not included in `BalanceRootV1`. Only `wallet_balance_*` keys (native QUG) are hashed. This is a deliberate scoping decision consistent with the checkpoint approach.
- **Mining rewards (Path A)** — block coinbase processing in `process_block_mining_rewards_tx()` is unchanged. It is the primary correct path.
- **Bracha Finality Engine** — remains in f=0 shadow mode. BalanceRootV1 and Bracha-RB are orthogonal: Bracha ensures cross-node agreement for out-of-block credits; BalanceRootV1 enforces that the resulting native balance state is committed into block headers.
- **P2P balance gossip** (`Q_ENABLE_BALANCE_GOSSIP`) — still disabled by default. The balance root is computed from RocksDB, not from in-memory gossip state.

---

## 8. Deployment Procedure

### Why Delta Docker is the Right Test Environment

- **Isolated from production:** Separate container, separate data directory under `/home/orobit/docker-balance-root-test/` — completely disposable
- **Epsilon untouched:** It continues serving frontend users and miners. quillon.xyz stays live throughout the 14-day soak
- **Beta/Gamma untouched:** They continue serving miners via the load balancer. No user disruption during the soak period
- **Delta Docker can be wiped and restarted without consequence:** No real funds, no production dependencies
- **Real network conditions:** The container connects to actual mainnet peers (Epsilon, Beta, Gamma) and syncs real blocks. This is not a simulation
- **Tests the exact binary that will go to production:** The `q-api-server` binary built on Beta is the same binary SCP'd to Delta and later deployed to Beta/Gamma/Epsilon via `ha-deploy.sh`

```bash
# Step 1: Run Phase 1 tests (on Beta locally — no network impact)
cargo test --package q-storage --test balance_determinism_tests

# Step 2: Build v10.6.0 binary (on Beta)
# First bump Cargo.toml version to 10.6.0
cargo build --release --package q-api-server

# Step 3: SCP binary to Delta and start a NEW test container
scp target/release/q-api-server root@5.79.79.158:/home/q-api-server-v10.6.0

ssh root@5.79.79.158 "docker run -d \
  --name q-balance-root-v1-test \
  --memory=6g \
  -p 8086:8080 -p 9006:9001 \
  -v /home/q-api-server-v10.6.0:/opt/q-api-server:ro \
  -v /home/orobit/docker-balance-root-test:/data \
  -e Q_NETWORK_ID=mainnet-genesis \
  -e Q_DB_PATH=/data/db \
  -e Q_P2P_PORT=9001 \
  -e RUST_LOG=info \
  -e ROCKSDB_BLOCK_CACHE_MB=2048 \
  -e Q_TOR_BOOTSTRAP_TIMEOUT=5 \
  debian:12 \
  bash -c '
    apt-get update -qq && apt-get install -y -qq libssl3 ca-certificates curl >/dev/null 2>&1 && \
    cp /opt/q-api-server /usr/local/bin/q-api-server && \
    chmod +x /usr/local/bin/q-api-server && \
    /usr/local/bin/q-api-server --port 8080 2>&1
  '"

# Step 4: Monitor for BalanceRootV1 log messages
# At height < 18,600,000 (current): advisory only, no enforcement
ssh root@5.79.79.158 "docker logs -f q-balance-root-v1-test 2>&1 | grep -E 'BALANCE ROOT|balance_root|BalanceRootV1'"

# Step 5: Check balance_state_hash matches Epsilon
# On Delta container:
ssh root@5.79.79.158 "docker exec q-balance-root-v1-test curl -s http://localhost:8080/api/v1/health" | python3 -c "import json,sys; d=json.load(sys.stdin)['data']; print('hash:', d['balance_state_hash'], 'wallets:', d['wallet_count'])"
# On Epsilon (via Beta):
ssh root@89.149.241.126 "curl -s http://localhost:8080/api/v1/health" | python3 -c "import json,sys; d=json.load(sys.stdin)['data']; print('hash:', d['balance_state_hash'], 'wallets:', d['wallet_count'])"
# These MUST match before activation

# Step 6: 14-day soak — watch for any BALANCE ROOT MISMATCH logs
ssh root@5.79.79.158 "docker logs q-balance-root-v1-test 2>&1 | grep '💥 \[BALANCE ROOT'"
# Must return 0 results for the full soak period

# Step 7: Only after soak — deploy to Beta/Gamma via ha-deploy.sh
./scripts/ha-deploy.sh full -y

# EPSILON: LAST and only after Beta+Gamma verified clean for 14 days
```

### Baseline Hash to Verify Before Activation

The existing `q-sync-bracha-rb` container on Delta has:
- `balance_state_hash=51447d38626352b8...`
- `wallets=1340`

This hash must be verified against Epsilon's `balance_state_hash` before the `BalanceRootV1` enforcement height. Any divergence indicates pre-existing balance state disagreement that must be resolved first — activating enforcement on a divergent state would cause an immediate network split.

---

## 9. Rollback Plan

### If Shadow Mode Reveals Persistent Mismatches

If the Delta test container logs `💥 [BALANCE ROOT MISMATCH]` consistently during the soak:

1. Stop the container and inspect: `docker logs q-balance-root-v1-test 2>&1 | grep 'BALANCE ROOT MISMATCH' | head -20`
2. Compare wallet counts and specific balance differences between Delta and Epsilon
3. The mismatch is diagnostic data — it reveals which wallets diverge and by how much
4. Fix the divergence root cause (likely a replay gap or an off-chain mutation that wasn't cleared)
5. Wipe the container data: `rm -rf /home/orobit/docker-balance-root-test/` on Delta and restart
6. Do NOT proceed to production deployment until the soak completes clean

### If Production Deployment Needs Rollback

The `ha-deploy.sh rollback` command restores the previous binary on Beta and Gamma within 60 seconds. Epsilon requires manual rollback (SCP previous binary, restart service). The fallback binary is always kept at `q-api-server.backup-v{PREV_VERSION}` on each server.

---

## 10. Timeline

| Milestone | Target | Block Height | Status |
|-----------|--------|-------------|--------|
| Write `balance_determinism_tests.rs` | 2026-05-07 | any | Not started |
| Wire `compute_balance_root_for_block()` into block producer | 2026-05-07 | any | Not started |
| Shadow-mode validation logging | 2026-05-07 | any | Not started |
| Health endpoint exposes `balance_root_v1` | 2026-05-07 | any | Not started |
| Build v10.6.0 binary | 2026-05-08 | any | Not started |
| Start `q-balance-root-v1-test` on Delta | 2026-05-08 | any | Not started |
| Verify `balance_state_hash` match: Delta vs. Epsilon | 2026-05-08 | ~17,400,000 | Not started |
| 14-day soak begins | 2026-05-08 | ~17,400,000 | Not started |
| Discord/BitcoinTalk upgrade announcement | 2026-05-08 | ~17,400,000 | Not started |
| Soak completes (zero mismatches) | 2026-05-22 | ~18,600,000 | Not started |
| Deploy v10.6.0 to Beta/Gamma | 2026-05-22 | ~18,580,000 | Not started |
| Deploy v10.6.0 to Epsilon | 2026-05-22 | ~18,590,000 | Not started |
| **BalanceRootV1 enforcement activates** | 2026-05-22 | **18,600,000** | Scheduled |

---

## 11. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| `compute_balance_root_for_block()` produces different hashes on two nodes with same chain | LOW — function is deterministic and tested | CRITICAL — network split at activation | 14-day soak on Delta; hash comparison with Epsilon before go-live |
| Off-chain balance mutation (backward sync, P2P gossip) produces root mismatch | MEDIUM — backward sync is gated but gossip may still write | HIGH — shadow-mode mismatches appear | Verify backward sync gate is still in place; run `journalctl` grep during soak |
| Delta container falls behind network tip before soak completes | LOW — Epsilon at 10Gbit keeps it synced | MEDIUM — soak data invalid if diverged | Monitor height gap: `docker exec q-balance-root-v1-test curl -s http://localhost:8080/api/v1/status | jq .height` |
| Activation height 18,600,000 reached before Beta/Gamma upgraded | LOW — ~14 days notice | CRITICAL — Beta/Gamma forked off network | `ha-deploy.sh` deploy at 18,590,000 (10,000 blocks = ~2.8 hours of margin) |
| `balance_root` field addition breaks serialization for old nodes | LOW — new field defaults to [0;32] pre-activation | HIGH — old nodes reject new blocks | The existing `state_root` field can be repurposed; or add `balance_root` field with backward-compat defaults |
| Delta container OOM during soak | LOW — 6GB RAM, ROCKSDB_BLOCK_CACHE_MB=2048 | LOW — restart, soak continues | Monitor: `docker stats q-balance-root-v1-test --no-stream` |

---

## 12. Files to Create or Modify

### New Files
| File | Purpose |
|------|---------|
| `crates/q-storage/tests/balance_determinism_tests.rs` | 10+ tests proving cross-node hash determinism |

### Modified Files
| File | Change |
|------|--------|
| `crates/q-api-server/src/block_producer.rs` (~line 954) | Compute `balance_root` when `BalanceRootV1` gate active |
| `crates/q-api-server/src/main.rs` (~line 11380) | Shadow-mode mismatch logging in block validation |
| `crates/q-api-server/src/handlers.rs` | Add `balance_root_v1` and `balance_root_active` to health response |
| `crates/q-types/src/block.rs` (~line 239) | Add `balance_root: [u8; 32]` field to `BlockHeader` (or repurpose `state_root`) |

### Non-Changes (Intentional)
| Component | Reason |
|-----------|--------|
| `crates/q-consensus-guard/src/upgrade_gate.rs` | Gate already configured correctly |
| `crates/q-storage/src/lib.rs:4426` | `compute_balance_root_for_block()` already correct |
| DEX / token balance paths | Not in scope for BalanceRootV1 (native QUG only) |
| `StateRootV1` activation height | Remains at `u64::MAX` — independent gate |
| Bracha Finality Engine | Continues in shadow mode — orthogonal system |

---

## 13. Open Questions for DeepSeek Review

**Q1: New `balance_root` field vs. repurposing `state_root`**

The `BlockHeader` has `state_root: [u8; 32]`, currently `[0;32]` on mainnet (StateRootV1 gate at u64::MAX). Can BalanceRootV1 write into `state_root`? Arguments for: no struct change needed, simpler. Arguments against: `state_root` is semantically reserved for the eventual full-state root covering tokens+pools; using it for native-only balance root creates a future rename problem and potential confusion with the StateRootV1 upgrade gate. Which is safer?

**Q2: Timing of root computation relative to block production**

The block producer applies transactions, then calls `compute_balance_root_for_block()`. This function reads `wallet_balance_*` from RocksDB. Is there a race between the RocksDB write of newly applied transactions and the read inside `compute_balance_root_for_block()`? On a busy node, could the root be computed on a slightly stale state?

**Q3: Root computation latency impact**

`compute_balance_root_for_block()` calls `load_wallet_balances()` which iterates all `wallet_balance_*` keys. At 1,340 wallets (current) this is fast. At 100,000 wallets (expected in 6 months), how much does this add to block production time? Should there be a background pre-computation that caches the root and invalidates it on any balance write?

**Q4: Should validators compute the root before or after applying the block?**

Current proposal: validators compute the root AFTER applying the received block, then compare against the block's `balance_root`. This matches what the producer did. But if there is any ordering dependency between "apply the block" and "compute the root for comparison", a bug here could cause false positive mismatches. Is there a cleaner approach?

**Q5: Soak go/no-go criteria**

The proposed criterion is "zero `💥 [BALANCE ROOT MISMATCH]` log lines over 14 days on the Delta container." Is this sufficient? Should there also be a positive criterion — e.g., "at least N consecutive blocks where `balance_root` matches across container and Epsilon"? What is the minimum acceptable soak duration for a mandatory upgrade on a $1.5B mainnet?

---

## 14. Summary Status Table

| Item | Status | Risk | ETA |
|------|--------|------|-----|
| Upgrade gate (18,600,000, mandatory) | ✅ Done | — | — |
| `compute_balance_root_for_block()` function | ✅ Done | — | — |
| Legacy `compute_balance_state_hash()` | ✅ Done (kept for diagnostics) | — | — |
| Backward sync gate | ✅ Done (2026-04-28 session) | — | — |
| `balance_determinism_tests.rs` | Not done | HIGH | 2026-05-07 |
| Block producer wiring | Not done | CRITICAL | 2026-05-07 |
| Validator shadow-mode logging | Not done | CRITICAL | 2026-05-07 |
| Health endpoint `balance_root_v1` field | Not done | MEDIUM | 2026-05-07 |
| v10.6.0 build + Delta container start | Not done | CRITICAL | 2026-05-08 |
| Hash match: Delta vs. Epsilon | Not done | CRITICAL | 2026-05-08 |
| 14-day Delta soak (zero mismatches) | Not done | CRITICAL | 2026-05-22 |
| Upgrade announcement (Discord + BitcoinTalk) | Not done | HIGH | 2026-05-08 |
| Deploy v10.6.0 to Beta/Gamma/Epsilon | Not done | CRITICAL | 2026-05-22 |
| **BalanceRootV1 enforcement at 18,600,000** | Scheduled | — | ~2026-05-22 |

---

## 15. References

| Component | Location |
|-----------|----------|
| BalanceRootV1 upgrade gate (mainnet) | `crates/q-consensus-guard/src/upgrade_gate.rs:139` |
| BalanceRootV1 upgrade gate (testnet) | `crates/q-consensus-guard/src/upgrade_gate.rs:187` |
| `compute_balance_root_for_block()` | `crates/q-storage/src/lib.rs:4426` |
| Legacy `compute_balance_state_hash()` | `crates/q-storage/src/lib.rs:4388` |
| Block producer state_root wiring | `crates/q-api-server/src/block_producer.rs:954` |
| Block validation (warning → reject path) | `crates/q-api-server/src/main.rs:~11380` |
| `BlockHeader.state_root` field | `crates/q-types/src/block.rs:239` |
| Bracha Finality Engine | `crates/q-storage/src/balance_finality_engine.rs` |
| State root activation plan | `docs/technical-review-state-root-activation-plan-2026-04-28.md` |
| Continuation document | `docs/continuation-state-root-v10415-2026-04-28.md` |
| Bracha RB design | `docs/technical-review-balance-finality-bracha-rb-2026-05-01.md` |
| Genesis sync fixes | `docs/technical-review-genesis-sync-balance-integrity-2026-05-06.md` |

---

*Document version: v1.0 — 2026-05-06*
*Prepared for DeepSeek R1 external review*
