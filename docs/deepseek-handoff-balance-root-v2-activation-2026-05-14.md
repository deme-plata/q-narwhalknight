# DeepSeek Handoff — balance_root_v2 Activation Wiring + Tests

**Date:** 2026-05-14
**Branch base:** `feature/safe-batched-sync-v1.0.2` at commit `0d51458d` or later
**Companion docs:**
- `docs/blueprints-ivc-snark-2026-05-13.md` (Blueprint 1: SMT spec — already implemented in `crates/q-storage/src/balance_smt.rs`)
- `docs/consensus-design-v2-2026-05-10.md` (overall consensus state map)
- `docs/session-handoff-v10.9.20-2026-05-14.md` (what landed in v10.9.20)

**Audience:** External implementer (DeepSeek). The codebase is the Q-NarwhalKnight quantum-DAG-BFT chain, ~$2B market cap, live mainnet at Beta/Gamma/Delta/Epsilon. **Treat every change as consensus-critical.**

**Why this is being outsourced:** the wiring touches `save_wallet_balances` (the single chokepoint for balance writes; the Rule-1 max-wins guard lives here per CLAUDE.md). That function has been a vector for catastrophic bugs in the past (May 2026 balance divergence incident). Outsourcing lets the in-tree author keep their hands off it while a careful reviewer / second pair of eyes works it through. Submit work as PR-ready commits on a feature branch.

---

## 1. Goal

`balance_root_v2` is a Sparse Merkle Tree (SMT) replacement for `balance_root_v1` (the BLAKE3 flat hash). The SMT module is **already committed** at `crates/q-storage/src/balance_smt.rs` (743 LOC, 12 passing unit tests). What's missing is:

1. **Constructor wiring** — open the SMT alongside the main RocksDB so it shares the same physical DB instance and benefits from RocksDB's WAL/checkpoint semantics.
2. **Incremental update wiring** — every wallet balance change must update the SMT in the *same* `WriteBatch` so SMT state cannot diverge from wallet state under crash/restart.
3. **Shadow-mode comparison** — gated by env var `Q_BALANCE_ROOT_V2_SHADOW=1`, log a MISMATCH ERROR if the v2 root disagrees with the v1 root. v1 stays canonical until the upgrade gate fires.
4. **Activation enforcement** — once `Upgrade::BalanceRootV2` activates, block validation requires `BlockHeader::balance_state_root` to equal the SMT root, and producers fill it in.
5. **Activation rebuild** — at the activation height, every node performs a one-time `BalanceSmt::rebuild_from_balances` against the current wallet table so the post-activation SMT root matches deterministically across all nodes.

The `Upgrade::BalanceRootV2` variant is already added to `crates/q-consensus-guard/src/upgrade_gate.rs` (mainnet: `u64::MAX` — dormant; testnet: `0` — immediate).

## 2. Hard constraints (read first, twice)

The codebase has four NON-NEGOTIABLE balance-integrity rules at the top of `CLAUDE.md`. Internalize them before writing any line of code:

1. **`save_wallet_balances` MUST be max-wins** — i.e., if `existing >= new`, skip the write. Do NOT remove or weaken this guard. Your SMT wiring goes *inside* this function but *after* the max-wins check, only on the rows that actually get written.
2. **Replay code MUST gate on `is_checkpoint_applied()`** — never run `replay_post_checkpoint_balances` on a node that bootstrapped from genesis (such as Epsilon). The SMT rebuild at activation is NOT the same as balance replay, but study the guard so you don't accidentally trigger it.
3. **Epsilon's wallet balances are authoritative** — no code path may write a lower balance to any wallet on Epsilon. The activation rebuild reads from wallet table; it does not write to it.
4. **All balance-touching code is tested on Alpha Docker BEFORE Beta/Gamma/Delta/Epsilon.**

Additional rules from this codebase:
- **No `unwrap()` outside tests.** Use `?` or `.context()` or proper match arms.
- **No `todo!()` / `unimplemented!()` / placeholders that "look right but aren't".**
- **Build for Debian 12 via the Epsilon Docker pattern** — do NOT compile on Beta. Spec is in `CLAUDE.md` under "COMPILATION & BUILD REQUIREMENTS". Working setup: `ssh root@89.149.241.126 "cd /home/orobit/q-narwhalknight-src && docker run --rm -v $(pwd):/src -v /home/orobit/target-debian12:/src/target -w /src rust:bookworm bash -c 'apt-get update -qq && apt-get install -y -qq libssl-dev pkg-config cmake clang libudev-dev libclang-dev >/dev/null 2>&1 && cargo check --package q-api-server'"`. Reuse the `target-debian12/` cache (incremental builds ~5 min).
- **All consensus-rule changes are height-gated via `q-consensus-guard::Upgrade::*`.** The shadow-mode comparison is NOT a consensus rule (no header field check), so it doesn't need a gate. Activation enforcement IS a consensus rule and gates on `Upgrade::BalanceRootV2`.

## 3. Jobs (numbered, atomic, in dependency order)

Implement these as separate commits. Each commit must compile (`cargo check --package <crate>` on Epsilon Docker) and pass its own tests before moving on. The numbered order is the dependency order.

### Job D1 — Constructor wiring (StorageEngine opens BalanceSmt)

**Files:**
- `crates/q-storage/src/lib.rs` — modify `StorageEngine::open()` around line 706.

**What to do:**
1. Add `CF_BALANCE_SMT` (already defined at `crates/q-storage/src/balance_smt.rs:28` as `pub const CF_BALANCE_SMT: &str = "cf_balance_smt"`) to the `ColumnFamilyDescriptor` list passed to `DB::open_cf_descriptors`. Use default `Options`.
2. After the DB is opened, call `BalanceSmt::open(db.clone())?` to construct the SMT instance.
3. Store the SMT as a field on `StorageEngine`: `pub balance_smt: Arc<BalanceSmt>`.
4. Expose a `pub fn balance_smt(&self) -> Arc<BalanceSmt>` accessor — callers in q-api-server will need this.
5. The constructor MUST be backwards-compatible: opening an EXISTING DB that does not yet have `cf_balance_smt` must succeed. RocksDB auto-creates missing CFs when `create_missing_column_families(true)` is set on `Options` — verify this flag is set; if not, set it.

**Acceptance test (write this as part of the commit):**

```rust
// crates/q-storage/tests/storage_engine_smt_open_test.rs
#[tokio::test]
async fn storage_engine_opens_with_smt() {
    let tmp = TempDir::new().unwrap();
    let node_id = [1u8; 32];
    let engine = StorageEngine::open(tmp.path(), node_id).await.unwrap();
    let smt = engine.balance_smt();
    assert_eq!(smt.root(), smt.genesis_root());
}

#[tokio::test]
async fn storage_engine_opens_existing_db_without_smt_cf() {
    // Create a DB without cf_balance_smt (simulates a pre-v10.9.22 DB).
    let tmp = TempDir::new().unwrap();
    {
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let cfs = vec![
            rocksdb::ColumnFamilyDescriptor::new("default", rocksdb::Options::default()),
            // No CF_BALANCE_SMT here — simulating an upgrade path.
        ];
        let _db = rocksdb::DB::open_cf_descriptors(&opts, tmp.path(), cfs).unwrap();
    }
    // Now open StorageEngine — should auto-create cf_balance_smt.
    let engine = StorageEngine::open(tmp.path(), [1u8; 32]).await.unwrap();
    let smt = engine.balance_smt();
    assert_eq!(smt.root(), smt.genesis_root());
}
```

**Risk:** opening with a new CF list against an existing DB is the most likely place for a startup failure on Beta/Gamma/Delta/Epsilon. Test thoroughly on a copy of Epsilon's DB (or a fresh Docker run) before submitting.

---

### Job D2 — Atomic SMT update in `save_wallet_balances`

**Files:**
- `crates/q-storage/src/lib.rs` — modify `pub async fn save_wallet_balances` around line 4435.

**Current behavior** (must preserve exactly):
1. Iterate `balances: &HashMap<[u8; 32], u128>`.
2. For each `(addr, new)`: load existing; if `existing >= new`, skip (max-wins guard). Otherwise queue write.
3. Issue a single `WriteBatch` with all `wallet_balance_<hex>` puts.
4. Sync to disk.

**New behavior (additive — do not change step 2's max-wins logic):**
1-2. Unchanged.
3. **Before issuing the WriteBatch, call `self.balance_smt.apply_to_batch(&mut batch, &accepted_updates)?`** where `accepted_updates: Vec<([u8; 32], u128)>` contains only the entries that PASSED the max-wins check (skipped entries do not touch the SMT, preserving the invariant that the SMT reflects the canonical wallet state).
4. Issue the WriteBatch. The SMT internal nodes + root pointer + wallet rows all commit atomically.
5. **After successful `db.write(batch)`**, call `self.balance_smt.commit_root(new_root)` to advance the in-memory cached root.

Pseudocode:

```rust
pub async fn save_wallet_balances(&self, balances: &HashMap<[u8; 32], u128>) -> Result<()> {
    // ... existing max-wins logic produces `accepted: Vec<([u8; 32], u128)>` ...

    if accepted.is_empty() {
        return Ok(());
    }

    let mut batch = WriteBatch::default();

    // ... existing put_cf for wallet_balance_<hex> on `accepted` ...

    // NEW: feed the SAME accepted updates into the SMT inside the SAME batch.
    let new_smt_root = self.balance_smt.apply_to_batch(&mut batch, &accepted)
        .context("balance_smt.apply_to_batch")?;

    // Single atomic commit. Wallet rows + SMT nodes + SMT root pointer.
    self.hot_db.write_batch_sync(batch).await
        .context("save_wallet_balances atomic commit")?;

    // After successful commit, advance the in-memory cached SMT root.
    self.balance_smt.commit_root(new_smt_root);

    Ok(())
}
```

**Critical invariant:** the SMT only ever sees writes that PASSED the max-wins check. If max-wins rejects a (lower) balance for wallet A, the SMT must NOT see A's lower value. This keeps the SMT root deterministic across nodes — every node's max-wins decision is identical (compares against its own current state), so every node's SMT update set is identical for a given set of incoming balances.

**Why a single function rather than two:** the wallet CF row and the SMT path commit together. If a power-cut happens between writing the wallet row and updating the SMT, the SMT root will not match the new wallet table on reboot and the next compute_balance_root_for_block call will surface the divergence loudly (good — we want loud failures). If they're in a single `WriteBatch` and `write_batch_sync`, RocksDB guarantees both-or-neither.

**Acceptance test:**

```rust
// crates/q-storage/tests/save_wallet_balances_smt_atomic_test.rs

#[tokio::test]
async fn smt_reflects_only_max_wins_accepted_updates() {
    let tmp = TempDir::new().unwrap();
    let engine = StorageEngine::open(tmp.path(), [1u8; 32]).await.unwrap();

    // Seed wallet A with 1000.
    let a = [0xAA; 32];
    let mut seed = HashMap::new();
    seed.insert(a, 1000u128);
    engine.save_wallet_balances(&seed).await.unwrap();
    let root_after_seed = engine.balance_smt().root();
    assert_ne!(root_after_seed, engine.balance_smt().genesis_root());

    // Attempt to overwrite A with a LOWER value 500 — max-wins must reject it.
    // The SMT root must NOT change as a result.
    let mut lower = HashMap::new();
    lower.insert(a, 500u128);
    engine.save_wallet_balances(&lower).await.unwrap();
    assert_eq!(engine.balance_smt().root(), root_after_seed,
               "SMT advanced on a max-wins-rejected lower value (BUG)");

    // Now write a HIGHER value 1500 — max-wins accepts; SMT must advance.
    let mut higher = HashMap::new();
    higher.insert(a, 1500u128);
    engine.save_wallet_balances(&higher).await.unwrap();
    assert_ne!(engine.balance_smt().root(), root_after_seed,
               "SMT did NOT advance on a max-wins-accepted update (BUG)");
}

#[tokio::test]
async fn smt_root_survives_crash_and_reopen() {
    let tmp = TempDir::new().unwrap();
    let final_root = {
        let engine = StorageEngine::open(tmp.path(), [1u8; 32]).await.unwrap();
        let mut updates = HashMap::new();
        for i in 0..50u8 {
            let mut a = [0u8; 32];
            a[0] = i;
            updates.insert(a, 1_000_000_000_000u128 + (i as u128));
        }
        engine.save_wallet_balances(&updates).await.unwrap();
        engine.balance_smt().root()
    };
    // Drop engine, simulate cold restart.
    let engine = StorageEngine::open(tmp.path(), [1u8; 32]).await.unwrap();
    assert_eq!(engine.balance_smt().root(), final_root,
               "persisted SMT root did not survive restart");
}
```

**Risk:** changing `save_wallet_balances` is the single highest-risk change in this entire handoff. ANY bug here corrupts balances. Submit the diff as a separate commit, sized as small as physically possible. The reviewer should be able to read the entire diff in 5 minutes.

---

### Job D3 — Shadow mode comparison in `compute_balance_root_for_block`

**Files:**
- `crates/q-storage/src/lib.rs` — modify `pub async fn compute_balance_root_for_block` around line 4553.

**Current behavior:** computes `balance_root_v1` via BLAKE3 flat hash over sorted `(addr, balance)` pairs. Returns 32 bytes. Used by the integrity_api and (in some paths) header construction.

**New behavior:**
1. Compute `balance_root_v1` exactly as today (no change to the v1 path).
2. If `std::env::var("Q_BALANCE_ROOT_V2_SHADOW").map(|v| v == "1").unwrap_or(false)`:
   - Read `let v2_root = self.balance_smt().root();`.
   - Compare `v1` and `v2`. They will NOT be the same value (different domain separators, different structure). The interesting question is: does v2 change at the same blocks as v1 changes? Specifically:
     - Hash both roots together: `let combo = blake3::hash(&[v1[..], v2[..]].concat());`
     - Log INFO once per height: `"📊 [SHADOW] height=H v1_root=<hex8> v2_root=<hex8> combo=<hex8>"`
     - Persist `(height, v1, v2, combo)` to CF_MANIFEST under key `balance_root_shadow:<height_be_hex>`.
   - This produces a per-height audit trail that operators across nodes can compare:  if two nodes both have shadow mode enabled and produce different `combo` values at the same height, the SMT update path is non-deterministic and activation must be blocked.
3. v1 remains the return value (canonical).
4. If `is_upgrade_active(Upgrade::BalanceRootV2, height)` is true (post-activation), return `v2_root` instead of `v1_root`. **Do not implement this branch yet** — leave a `TODO(balance-root-v2-activation)` comment. Activation comes after weeks of green shadow soak.

**Acceptance test:**

```rust
// crates/q-storage/tests/balance_root_shadow_mode_test.rs

#[tokio::test]
async fn shadow_mode_persists_per_height_record() {
    std::env::set_var("Q_BALANCE_ROOT_V2_SHADOW", "1");
    let tmp = TempDir::new().unwrap();
    let engine = StorageEngine::open(tmp.path(), [1u8; 32]).await.unwrap();

    let mut updates = HashMap::new();
    updates.insert([0x11; 32], 100u128);
    engine.save_wallet_balances(&updates).await.unwrap();

    let _root = engine.compute_balance_root_for_block().await.unwrap();
    // ... query CF_MANIFEST for balance_root_shadow:<height> and assert it exists ...
}
```

**Risk:** medium. Adds work on every call to compute_balance_root_for_block, but only when env var is set. Safe default = no behavior change.

---

### Job D4 — Activation rebuild path

**Files:**
- `crates/q-storage/src/lib.rs` — add a new public method.

**What to do:** when the chain crosses the activation height (`Upgrade::BalanceRootV2`), every node must perform a one-time rebuild of the SMT from its current wallet table so all nodes converge on the same v2 root at activation height.

```rust
/// One-time rebuild of the SMT from the live wallet table.
///
/// Call EXACTLY ONCE, at the activation height for `Upgrade::BalanceRootV2`,
/// before any post-activation block applies new balance changes. Idempotent
/// but expensive (O(N log N) RocksDB writes), so guarded by an internal
/// `cf_balance_smt_activation_done` sentinel key.
///
/// Returns the new SMT root.
pub async fn rebuild_balance_smt_at_activation(&self, activation_height: u64) -> Result<[u8; 32]> {
    // 1. Check sentinel — return Ok(cached_root) if already done.
    // 2. Read full wallet table into HashMap<[u8; 32], u128>.
    // 3. Call self.balance_smt().rebuild_from_balances(&map).
    // 4. Write sentinel.
    // 5. Return the new root.
}
```

**Wiring caller** (separate small commit):
- In `crates/q-api-server/src/main.rs` block-application path, after a block at `height == activation_height` is applied, call `storage_engine.rebuild_balance_smt_at_activation(height).await?` and log the result loudly. The header check that the new SMT root matches `block.header.balance_state_root` happens in the block-validation path (Job D5).

**Acceptance test:**

```rust
#[tokio::test]
async fn rebuild_is_idempotent() {
    let tmp = TempDir::new().unwrap();
    let engine = StorageEngine::open(tmp.path(), [1u8; 32]).await.unwrap();
    let mut balances = HashMap::new();
    for i in 0..100u8 {
        let mut a = [0u8; 32]; a[0] = i;
        balances.insert(a, (i as u128) * 1_000);
    }
    engine.save_wallet_balances(&balances).await.unwrap();

    let r1 = engine.rebuild_balance_smt_at_activation(100).await.unwrap();
    let r2 = engine.rebuild_balance_smt_at_activation(100).await.unwrap();
    assert_eq!(r1, r2, "rebuild not idempotent");
}
```

---

### Job D5 — Header enforcement at activation height

**Files:**
- `crates/q-types/src/block.rs` — `BlockHeader` already has a `balance_state_root: [u8; 32]` field? Verify; if absent, ADD it (this is a serialization change and itself needs an upgrade gate for the serialization version — see existing `Upgrade::StateRootV1`).
- `crates/q-api-server/src/main.rs` — block validation path: at `height >= activation_height`, reject blocks where `header.balance_state_root != local_smt_root`.
- Producer side: when building a block at `height >= activation_height`, set `header.balance_state_root = self.storage_engine.balance_smt().root()` *after* the block's balance updates have been applied to the SMT.

**Risk:** consensus-rule change. Get this height-gating right or the chain forks.

---

### Job D6 — Cross-node determinism integration test

**Files:**
- `crates/q-storage/tests/balance_root_v2_determinism_test.rs` (new)

**What to do:** spin up two `StorageEngine` instances (different tmp dirs, same logical wallet sequence). Apply the same 1000 random balance updates to each. After each application, both engines' `balance_smt().root()` must be byte-identical.

Pseudocode:

```rust
#[tokio::test]
async fn smt_root_is_deterministic_across_engines() {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let tmp_a = TempDir::new().unwrap();
    let tmp_b = TempDir::new().unwrap();
    let engine_a = StorageEngine::open(tmp_a.path(), [1u8; 32]).await.unwrap();
    let engine_b = StorageEngine::open(tmp_b.path(), [2u8; 32]).await.unwrap();

    for block in 0..1000u32 {
        let mut updates_a = HashMap::new();
        let mut updates_b = HashMap::new();
        let n = rng.gen_range(1..=20);
        for _ in 0..n {
            let mut addr = [0u8; 32];
            rng.fill(&mut addr);
            let amount: u128 = rng.gen::<u64>() as u128 * 1_000_000;
            updates_a.insert(addr, amount);
            updates_b.insert(addr, amount);
        }
        engine_a.save_wallet_balances(&updates_a).await.unwrap();
        engine_b.save_wallet_balances(&updates_b).await.unwrap();
        assert_eq!(
            engine_a.balance_smt().root(),
            engine_b.balance_smt().root(),
            "SMT root divergence at block {} (engines applied identical updates)", block
        );
    }
}
```

This test alone is what unblocks activation: 1000 blocks of identical updates producing identical roots is the strongest unit-test signal that the SMT update path is deterministic.

---

### Job D7 — Reorg correctness

**Files:**
- `crates/q-storage/tests/balance_root_v2_reorg_test.rs` (new)

The handoff blueprint doc covers the spec; the test must:
1. Apply blocks A → B1 → C1, capture root R_C1.
2. Reorg: rewind to A, apply B2 → C2, capture root R_C2.
3. Verify R_C1 ≠ R_C2.
4. Verify that re-applying the original path (rewind to A, then B1 → C1) reproduces R_C1 exactly.

Reorgs trigger balance reversion. Verify that the SMT *also* reverts — if `save_wallet_balances` is called with the pre-fork balance values during reorg, the SMT root should go back to the pre-fork value.

---

### Job D8 — Performance benchmark at scale

**Files:**
- `crates/q-storage/benches/balance_smt_bench.rs` (new, use criterion)

Benchmark scenarios:
1. **Cold SMT** — 100K initial wallets, 1K random updates per batch, time per batch.
2. **Hot SMT** — 1M wallets already, 1K random updates per batch.
3. **Tight cluster** — 1K updates all targeting addresses with the same first byte (worst-case prefix sharing).

**Activation target:** a single `save_wallet_balances` call with 1K updates must complete in **< 100ms** on Epsilon-class hardware. If it doesn't, activation is blocked until the SMT is optimized (e.g., parent-write deduplication for shared prefixes, RocksDB write-buffer tuning).

Report results in `docs/balance-smt-bench-results-<date>.md`.

---

### Job D9 — Real-chain rebuild verification

**Files:**
- A standalone CLI: `crates/q-storage/src/bin/verify_smt_rebuild.rs` (new)

A binary that opens an existing RocksDB at a given path, reads the wallet table, and computes the SMT root via `BalanceSmt::rebuild_from_balances`. Prints the root and the number of wallets.

Run against:
1. A fresh Docker node's DB after 10 blocks (small N).
2. Epsilon's mainnet DB snapshot (real N, ~10⁵+ wallets). Operator runs this manually on a read-only snapshot — do NOT modify production DB.
3. Beta's DB snapshot.
4. Gamma's DB snapshot.
5. Delta's DB snapshot.

All four production nodes must produce byte-identical SMT roots at the same height. If they don't, the wallet tables themselves are divergent — a balance-integrity bug we need to fix BEFORE balance_root_v2 activation.

---

## 4. Wiring summary diagram

```
┌────────────────────────────────────────────────────────────────────┐
│  SHADOW MODE — Q_BALANCE_ROOT_V2_SHADOW=1, mainnet HEIGHT < ∞      │
│                                                                     │
│  save_wallet_balances():                                            │
│    accepted = max_wins_filter(balances)                             │
│    smt.apply_to_batch(&mut batch, &accepted)   ◄── Job D2          │
│    db.write_batch_sync(batch)                                       │
│    smt.commit_root(new_root)                                        │
│                                                                     │
│  compute_balance_root_for_block():                                  │
│    v1 = blake3_flat_hash(wallet_table)                             │
│    if shadow_enabled:                                               │
│       v2 = smt.root()                                               │
│       persist (height, v1, v2, combo) to CF_MANIFEST  ◄── Job D3   │
│    return v1                                  ◄── v1 still canonical│
└────────────────────────────────────────────────────────────────────┘

  ┌─ if (operator-driven, manual) — verify_smt_rebuild bin ◄── Job D9
  │   reads wallet table from each production node's DB snapshot,
  │   prints SMT root. Cross-node comparison done by operator.
  │
  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  ACTIVATION HEIGHT REACHED (Upgrade::BalanceRootV2 fires)       │
  │                                                                  │
  │  At height == activation_height, before applying the block:     │
  │    storage.rebuild_balance_smt_at_activation()  ◄── Job D4      │
  │    [one-time O(N) rebuild from wallet table]                    │
  │                                                                  │
  │  POST-ACTIVATION block validation (height ≥ activation_height): │
  │    if header.balance_state_root != smt.root():                  │
  │       REJECT BLOCK                              ◄── Job D5       │
  └─────────────────────────────────────────────────────────────────┘
```

## 5. Order of work + estimated complexity

| Job | LOC est. | Difficulty | Risk | Test files |
|---|---|---|---|---|
| D1 — Constructor wiring | ~30 | Low | Medium (DB schema) | 1 |
| D2 — Atomic update in save_wallet_balances | ~25 | Medium | **HIGH** | 1 |
| D3 — Shadow mode | ~50 | Low | Low | 1 |
| D4 — Rebuild path | ~60 | Medium | Medium | 1 |
| D5 — Header enforcement | ~80 | Medium | **HIGH** | 1 |
| D6 — Determinism test | ~80 | Low | Low | 1 (this *is* the test) |
| D7 — Reorg correctness test | ~120 | Medium | Low | 1 |
| D8 — Benchmark | ~150 | Low | Low | 1 |
| D9 — Real-chain rebuild bin | ~100 | Low | Low | n/a |

Submit each job as its own commit on a feature branch like `feat/balance-root-v2-activation-wiring`. Don't squash. Don't bundle.

## 6. What NOT to do

- **Don't touch `BalanceSmt` itself.** It's stable at 743 LOC with 12 passing tests. Bug fixes only if Job D6 / D7 surface real determinism issues.
- **Don't enable activation on mainnet in this branch.** The activation_height for mainnet stays `u64::MAX`. Activation flip is a separate, deliberate operator decision after weeks of green shadow-mode soak.
- **Don't add new column families beyond CF_BALANCE_SMT.** The shadow-mode audit trail uses CF_MANIFEST with a `balance_root_shadow:` key prefix — that CF already exists.
- **Don't change the SMT hash domain separators or tree structure.** `LEAF_TAG = "smt_leaf_v2"`, `NODE_TAG = "smt_node_v2"`, depth 256, BLAKE3 — all locked. Changing any of these silently breaks cross-node root agreement.
- **Don't add fallback paths.** No "if SMT fails, just skip" — that means we ship without the protection. SMT failures must be loud (ERROR log + propagated `Result::Err`).

## 7. Submission

PR target: `feature/safe-batched-sync-v1.0.2` (the canonical session branch, not main). Don't push to GitHub — push to `code.quillon.xyz` per `CLAUDE.md`. Run `git update-server-info` after every push.

Verification I will run on receiving your branch:
1. `cargo check --package q-storage` on Epsilon Docker (must be green).
2. `cargo test --package q-storage` on Epsilon Docker (all your new tests must pass).
3. Inspect the `save_wallet_balances` diff line-by-line for Rule-1 max-wins compliance.
4. Manual diff review against this spec — every section above mapped to a commit.

If anything is unclear, leave a comment in the PR description and I'll clarify before you ship.

---

## Appendix A: Files you'll touch

- `crates/q-storage/src/lib.rs` — Jobs D1, D2, D3, D4
- `crates/q-storage/src/balance_smt.rs` — **DO NOT MODIFY** (stable, 12 tests pass)
- `crates/q-storage/tests/storage_engine_smt_open_test.rs` — new (Job D1)
- `crates/q-storage/tests/save_wallet_balances_smt_atomic_test.rs` — new (Job D2)
- `crates/q-storage/tests/balance_root_shadow_mode_test.rs` — new (Job D3)
- `crates/q-storage/tests/balance_root_v2_determinism_test.rs` — new (Job D6)
- `crates/q-storage/tests/balance_root_v2_reorg_test.rs` — new (Job D7)
- `crates/q-storage/benches/balance_smt_bench.rs` — new (Job D8)
- `crates/q-storage/src/bin/verify_smt_rebuild.rs` — new (Job D9)
- `crates/q-types/src/block.rs` — Job D5 (verify `BlockHeader::balance_state_root` exists)
- `crates/q-api-server/src/main.rs` — Job D5 (validation + producer wiring at activation height) + Job D4 caller

## Appendix B: Existing related code you should read

- `crates/q-storage/src/balance_smt.rs` — full SMT implementation. Read the module docstring + every public method comment. The mental model:
  - Depth 256, addressed by 32-byte wallet addr (256 bits).
  - Empty subtree hashes precomputed at every depth.
  - `apply_to_batch` composes with an external WriteBatch — designed exactly for what Job D2 needs.
- `crates/q-storage/src/lib.rs:4220` (`save_wallet_balance` — singular) and `lib.rs:4435` (`save_wallet_balances` — plural). The plural is the max-wins entry point. The singular is the raw setter.
- `crates/q-storage/src/lib.rs:4553` (`compute_balance_root_for_block`). The v1 implementation. Read for context on the canonical computation.
- `crates/q-consensus-guard/src/upgrade_gate.rs:71` for the existing Upgrade variants, `:130` for `Upgrade::BalanceRootV2` mainnet config, and `:225` for testnet config (added in this session).
- `crates/q-api-server/src/integrity_api.rs:67` (`get_balance_root`) and `:166` (`get_emission`). These are the cross-node comparison endpoints — once you have shadow mode emitting `balance_root_shadow:<height>` entries to CF_MANIFEST, you can add a new endpoint `/api/v1/integrity/balance_root_v2/:height` that reads from those entries.

## Appendix C: How to verify your work without breaking mainnet

1. Build for Debian 12 via the Epsilon Docker pattern (see Section 2 hard constraints).
2. Run all new + existing tests:
   ```
   cargo test --package q-storage
   cargo test --package q-consensus-guard
   ```
3. Spin up a fresh Docker test node on Epsilon (NOT Beta/Gamma/Delta/Epsilon production) with `Q_BALANCE_ROOT_V2_SHADOW=1` set. Let it sync for 24 hours. Verify zero MISMATCH log lines.
4. Repeat on a second fresh Docker test node. Compare their `combo` values at each height — must be byte-identical.

Only after step 4 is green for 48+ hours does the operator (= person flipping the activation height) consider mainnet activation. That decision is NOT in scope for your PR.

---

Done. Ship one commit per job, smallest-possible diffs, tests included. Ping the maintainer with the branch name when ready.
