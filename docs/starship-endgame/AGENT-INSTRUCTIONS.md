# Starship Endgame — Multi-Terminal Agent Instructions

> Coordinated parallel development across 4 Claude Code terminals.
> Created: 2026-03-10
> Branch: `feature/safe-batched-sync-v1.0.2`

---

## Terminal Assignments

| Terminal | Agent Role | Server | Issues | Priority |
|----------|-----------|--------|--------|----------|
| **T1** (this terminal) | Coordinator / PR-005 Hardening | Beta | #012, #013, #014 | HIGH |
| **T2** | Bridge Safety + Compute Verification | Beta | #016 | HIGH |
| **T3** | Tunnel Completion (streams + handshake) | Beta | #002 (remaining) | CRITICAL |
| **T4** | Frontend + Visualization | Beta | #008, #019, #020 | MEDIUM |

---

## T1: Coordinator — Compute Hardening (PR-005)

**You are the coordinator terminal. Work on PR-005: Compute Hardening.**

### Context
- Branch: `feature/safe-batched-sync-v1.0.2`
- Working dir: `/opt/orobit/shared/q-narwhalknight`
- PR-005 is currently DRAFT, closes issues #012, #013, #014

### Issue #012: Async GPU Monitoring
**File**: `crates/q-compute/src/resource_monitor.rs`

1. Replace `std::process::Command` with `tokio::process::Command` in `try_nvidia_smi()` and `try_rocm_smi()`
2. Add a `GpuCache` struct with `last_result: Option<GpuStats>`, `last_query: Instant`, `ttl: Duration` (2s)
3. Detect GPU backend once at startup: `GpuBackend::detect()` → `Nvidia | Rocm | None`
4. If async GPU query takes >200ms, return cached value + log warning
5. Test: `#[tokio::test]` that verifies GPU monitoring doesn't block (time the future)

### Issue #013: Core Enforcement
**File**: `crates/q-compute/src/orchestrator.rs`

1. `core_affinity` crate is already in Cargo.toml — use `core_affinity::set_for_current(CoreId { id: N })`
2. When scheduler assigns cores to a layer, call `enforce_core_affinity(layer, core_range)`
3. Mining layer gets cores 0..N (best cache locality for PoW)
4. AI Inference layer gets cores N..N+M
5. Graceful fallback: if `set_for_current` fails, warn and continue (don't crash)
6. Test: check `/proc/self/status` Cpus_allowed after pinning

### Issue #014: Inference Revenue Wiring
**File**: `crates/q-compute/src/inference_pool.rs`

1. Wire `InferenceWorkerPool` revenue callback to orchestrator layer stats
2. When inference completes, call `orchestrator.record_inference_revenue(tokens, model, price_per_token)`
3. Sync `max_concurrent` workers with core budget from orchestrator
4. Add per-token pricing config: `INFERENCE_PRICE_PER_TOKEN` env var (default: 0.0001 QUG)
5. Revenue should appear in `GET /api/v1/compute/status` under `layers[5].revenue`

### Verification
```bash
cargo test --package q-compute
cargo test --package q-compute --features metrics
cargo check --package q-api-server
```

---

## T2: Bridge Safety + Compute Verification (#016)

**You are working on Issue #016: Bridge Compute Verification.**

### Context
- Branch: `feature/safe-batched-sync-v1.0.2`
- Working dir: `/opt/orobit/shared/q-narwhalknight`
- `bridge_safety.rs` already exists at `crates/q-api-server/src/bridge_safety.rs` (925 lines)
- `bridge_attestations_topic()` already exists on NetworkId in `crates/q-types/src/lib.rs`

### Task: Add Multi-Node Verification Quorum

1. **Read first**:
   - `crates/q-api-server/src/bridge_safety.rs` — existing safety controller
   - `crates/q-api-server/src/bitcoin_bridge_api.rs` — existing bridge API
   - `crates/q-compute/src/orchestrator.rs` — Layer 3 assignment
   - `docs/starship-endgame/issues/016-bridge-safety-compute-verification.md`

2. **Add BridgeAttestation struct** to `bridge_safety.rs`:
   ```rust
   pub struct BridgeAttestation {
       pub swap_id: String,
       pub verifier_peer_id: String,
       pub chain: String,  // "BTC", "ETH", "ZEC", "IRON"
       pub tx_hash: String,
       pub amount: u64,
       pub confirmed: bool,
       pub signature: Vec<u8>,  // Ed25519 signature over (swap_id, confirmed, amount)
       pub timestamp: u64,
   }
   ```

3. **Add AttestationCollector**:
   - `pending: HashMap<String, Vec<BridgeAttestation>>` — keyed by swap_id
   - `check_quorum(swap_id) -> Option<bool>` — returns true if 2-of-3 confirmed
   - `timeout_stale(max_age: Duration)` — remove attestations older than 5 minutes
   - `submit_attestation(att: BridgeAttestation)` — add to pending, check quorum

4. **Wire into gossipsub**:
   - Already subscribed to bridge-attestations topic
   - Publish local attestation after verification
   - Handle incoming attestations in gossipsub handler

5. **Wire into orchestrator** Layer 3:
   - When bridge deposit detected, create verification task
   - Assign to Layer 3 workers
   - Workers query external RPC and produce attestation

### Verification
```bash
cargo check --package q-api-server
cargo test --package q-api-server -- bridge
```

---

## T3: Tunnel Completion — Streams + Handshake (#002)

**You are completing Issue #002: P2P Compute Tunnels.**

### Context
- Branch: `feature/safe-batched-sync-v1.0.2`
- Working dir: `/opt/orobit/shared/q-narwhalknight`
- Gossipsub peer discovery is DONE (PR-004, 3/6 criteria closed)
- Remaining: tunnel handshake, multiplexed streams, auto-open, redundant verification

### What's done (DON'T redo):
- `compute_tunnel_topic()` on NetworkId — DONE
- Topic subscription at startup — DONE
- 30s peer announcement task — DONE
- Gossipsub handler for "/compute-tunnel" — DONE
- TunnelManager integrated into Orchestrator — DONE
- PeerRegistry with TTL eviction + score-based selection — DONE

### Remaining work (3/6 criteria):

1. **Tunnel Handshake Protocol** (`crates/q-compute/src/tunnel.rs`):
   - Add NOISE XX handshake for tunnel establishment
   - `TunnelManager::open_tunnel(peer_id)` should perform handshake before data flow
   - Use libp2p's noise protocol or standalone `snow` crate
   - Add `TunnelState::Handshaking` state

2. **Multiplexed Streams** (`crates/q-compute/src/tunnel.rs`):
   - Each tunnel should support multiple concurrent streams (yamux-style)
   - Mining stream, AI inference stream, bridge verify stream
   - `tunnel.open_stream(StreamType::Mining)` → `TunnelStream`
   - Backpressure: if stream queue > 1000, drop oldest

3. **Auto-Open Tunnels** (`crates/q-compute/src/orchestrator.rs`):
   - When PeerRegistry discovers a high-score peer, auto-open tunnel
   - Threshold: score > 0.7 AND peer has needed capabilities
   - Rate limit: max 2 tunnel opens per second
   - Close tunnels to low-score peers on cleanup tick

4. **Redundant Compute Verification**:
   - Send same task to 2+ peers, verify results match
   - `TunnelManager::submit_redundant(task, min_confirmations: 2)`
   - Used by bridge verification (Layer 3) and AI inference (Layer 5)

### Key files:
- `crates/q-compute/src/tunnel.rs` (840 lines) — TunnelManager, PeerRegistry
- `crates/q-compute/src/orchestrator.rs` (623 lines) — Scheduler, Layer assignments
- `crates/q-compute/src/lib.rs` (267 lines) — Types: TunnelPayload, ComputePeerInfo

### Verification
```bash
cargo test --package q-compute
cargo check --package q-api-server
```

---

## T4: Frontend + Visualization (#008, #019, #020)

**You are working on frontend compute visualization and payment UI.**

### Context
- Branch: `feature/safe-batched-sync-v1.0.2`
- Working dir: `/opt/orobit/shared/q-narwhalknight`
- Frontend: `gui/quantum-wallet/` (React + TypeScript + Vite)
- Build: `cd gui/quantum-wallet && npm run build`
- Deploy: Nginx serves from `dist-final/`

### Issue #008: Tunnel Mesh Visualization
- Compute panel is done (shows layers, CPU/GPU stats)
- Need: D3.js force-directed graph showing P2P compute tunnels
- Data source: `GET /api/v1/compute/status` returns `cluster_peers` array
- Each peer has: peer_id, available_cores, gpu_tflops, ram_available_gb, compute_mode
- Tunnels have: tunnel_id, peer_id, state, bytes_sent, bytes_received
- Show nodes as circles sized by GPU TFLOPS, edges as tunnels with bandwidth labels
- Use existing D3 dependency (already in package.json)

### Issue #019: Payment Request API
- Read `docs/starship-endgame/issues/019-payment-request-api.md` first
- Add QR code generation for payment requests
- Frontend: PaymentRequest component with amount input, QR display, status polling

### Issue #020: Merchant POS Mode
- Read `docs/starship-endgame/issues/020-merchant-pos-mode.md` first
- Kiosk-style payment screen for merchants
- Large QR code, amount display, payment confirmation animation

### Verification
```bash
cd gui/quantum-wallet && npm run build
# Check for TypeScript errors
cd gui/quantum-wallet && npx tsc --noEmit
```

---

## Coordination Rules

1. **Don't touch each other's files**: Each terminal has assigned files (see above)
2. **Shared files**: If you need to edit `main.rs`, `orchestrator.rs`, or `lib.rs`, coordinate:
   - T1 owns `orchestrator.rs` and `resource_monitor.rs`
   - T2 owns `bridge_safety.rs` and `bitcoin_bridge_api.rs`
   - T3 owns `tunnel.rs`
   - T4 owns `gui/quantum-wallet/`
3. **Compilation conflicts**: Only ONE terminal runs `cargo build` at a time (shared target dir)
4. **Commit convention**: `feat(v9.5.x): <description> [Starship #NNN]`
5. **Test before committing**: `cargo test --package <your-package>`
