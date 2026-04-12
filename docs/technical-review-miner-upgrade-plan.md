# Q-Miner Upgrade Plan: Mining Fairness Phases A→C

**Date:** 2026-04-12  
**Key finding:** Phase A and B require ZERO miner changes. Phase C requires new VDF kernel but is backward-compatible.

---

## Upgrade Matrix

| Phase | Server Change | Miner Change | Backward Compatible |
|-------|--------------|-------------|-------------------|
| **A: Difficulty-weighted rewards** | `block_producer.rs` reward calc | **None** | Yes |
| **B: LWMA difficulty adjustment** | Challenge endpoint returns dynamic difficulty | **None** | Yes |
| **C: Genus-2 VDF fair lane** | Dual-path verification | New VDF kernel | Yes (old miners keep BLAKE3) |

---

## Why Phase A Needs No Miner Changes

Reward calculation is server-side only (`block_producer.rs:1286`). The server receives:
- `nonce`, `hash`, `difficulty_target`, `miner_address`

It already has the solution hash. It can compute leading zero bits and weight rewards without the miner knowing. Old miners see the same `accepted: true` response but get rewards proportional to difficulty achieved.

## Why Phase B Needs No Miner Changes

Miners fetch challenges every 50s or on SSE new-block signal. The challenge response already contains `difficulty_target`. When LWMA changes the target, the next challenge fetch automatically includes it. No version requirement.

## Phase C: What Miners Need

The fields already exist in MiningSolutionRequest (all Optional, all `None` today):
- `vdf_output: Option<String>` — Mumford point
- `vdf_proof: Option<String>` — Wesolowski proof
- `vdf_checkpoints: Option<Vec<String>>`
- `vdf_iterations_count: Option<u64>`

**Deployment sequence:**
1. Height H-100: Challenge endpoint starts returning optional Genus-2 params
2. Auto-updater pushes new miner binary with Genus-2 support
3. Height H: VDF lane activates, CPU miners can submit Genus-2 solutions
4. Old miners continue on BLAKE3 lane (lower reward but still valid)

## Auto-Updater Coverage

| Component | Auto-Update | Check Interval | Mechanism |
|-----------|------------|---------------|-----------|
| q-miner (standalone) | Yes | 5 min | `/api/v1/version` → SHA-256 verified download → self_replace |
| slint-wallet (GUI) | Yes | 60s after login, then 4h | Same mechanism |
| GPU kernel | Bundled in binary | With binary update | Kernel compiled from inline source |

## Files to Modify Per Phase

**Phase A (server only):**
- `crates/q-api-server/src/block_producer.rs:1286` — replace equal split with difficulty-weighted

**Phase B (server only):**
- `crates/q-mining/src/difficulty.rs` — replace stub with LWMA
- `crates/q-api-server/src/handlers.rs:9474` — dynamic difficulty from state
- `crates/q-api-server/src/main.rs` — periodic difficulty recalculation task

**Phase C (server + miner):**
- `crates/q-api-server/src/main.rs:15095` — already has dual-path verification
- `crates/q-mining/src/gpu.rs` — new Genus-2 OpenCL kernel
- `gui/slint-wallet/src/miner.rs:649` — populate VDF fields when activated
- `crates/q-miner/src/main.rs` — detect activation height, switch algorithm
