# STARSHIP ENDGAME REVOLUTION — Issues & Tasks

## Issue #001: Compute Orchestrator Core
**Priority**: CRITICAL
**Assigned**: Beta
**Status**: Open

Create `crates/q-compute/` with the adaptive resource governor that monitors
CPU/GPU/RAM/NET every 100ms and assigns work across 8 priority layers.

**Acceptance**:
- [ ] Resource sampling at 100ms resolution
- [ ] Priority preemption (mining always wins)
- [ ] Core pinning with work-stealing
- [ ] CLI flag `--compute-mode=full|mining-only|eco`
- [ ] Prometheus metrics export

---

## Issue #002: P2P Compute Tunnel (Miner/Node Mesh)
**Priority**: CRITICAL
**Assigned**: Beta + Epsilon
**Status**: Open

Build encrypted tunnels between miners and nodes so compute tasks flow
directly peer-to-peer without going through the API server.

**Architecture**:
```
Miner A ←──tunnel──→ Node Beta ←──tunnel──→ Node Epsilon
   │                    │                       │
   ├── Mining hash ────→│                       │
   │                    ├── AI inference task ──→│
   │                    │←── AI result ─────────┤
   │←── Proof task ─────┤                       │
   ├── Proof result ───→│                       │
   │                    ├── Bridge verify ──────→│
```

**Tunnel Protocol**:
- Gossipsub topic: `/qnk/{network}/compute-tunnel`
- Encrypted with node's Ed25519 session key
- Multiplexed: mining + inference + proofs over single connection
- Backpressure: sender respects receiver's capacity announcement
- Heartbeat every 10s, reconnect on failure

**Tunnel Types**:
1. **Miner→Node tunnel**: mining solutions + hash rate telemetry
2. **Node→Node tunnel**: task distribution + result verification
3. **Node→Miner tunnel**: push compute tasks to idle miner GPUs
4. **Miner→Miner tunnel**: collaborative proof generation

**Tasks**:
- [ ] Tunnel handshake protocol (Ed25519 + X25519 key exchange)
- [ ] Multiplexed stream (yamux over libp2p)
- [ ] Capacity announcement (cores, GPU TFLOPS, RAM, bandwidth)
- [ ] Task routing (assign to cheapest/closest available)
- [ ] Result verification (2-of-3 redundant compute)
- [ ] Tunnel dashboard in frontend

---

## Issue #003: GPU Mining Acceleration
**Priority**: HIGH
**Assigned**: Epsilon
**Status**: Open

Add GPU hash computation for 10-100x mining speedup.
Auto-detect GPU vendor, compile appropriate shader/kernel.

- [ ] OpenCL backend (AMD + Intel + NVIDIA)
- [ ] CUDA backend (optional feature flag)
- [ ] Vulkan compute fallback
- [ ] GPU memory pool (avoid alloc/dealloc per hash)
- [ ] Benchmark: CPU vs GPU hash rate comparison

---

## Issue #004: Game Trainer Mode — Performance Cheat Engine
**Priority**: HIGH
**Assigned**: Beta
**Status**: Open

Think of running a QNK node like playing a game on EXTREME difficulty.
The "Trainer" is a built-in performance optimizer that auto-applies every
known trick to maximize output — like a cheat engine for compute.

**Trainer Cheats (Auto-Applied)**:

```
┌─────────────────── QNK TRAINER v1.0 ─────────────────────┐
│                                                            │
│  [F1]  INFINITE CORES     — Pin all cores, no idle        │
│  [F2]  GOD MODE MEMORY    — Huge pages, mlock, zero swap  │
│  [F3]  SPEED HACK x100    — SIMD + GPU + io_uring         │
│  [F4]  WALL HACK          — See all peer compute capacity │
│  [F5]  AIM BOT            — Auto-assign optimal tasks     │
│  [F6]  NO CLIP            — Bypass OS scheduler limits    │
│  [F7]  INFINITE AMMO      — Never run out of work queue   │
│  [F8]  RAPID FIRE         — Batch submit mining solutions │
│  [F9]  TELEPORT           — Zero-copy data paths          │
│  [F10] PRESTIGE MODE      — Overclock everything safely   │
│  [F11] NUKE               — Max all settings (YOLO)       │
│  [F12] TRAINER MENU       — Toggle individual cheats      │
│                                                            │
│  STATUS: [ALL CHEATS ACTIVE]  Performance: 847% boost     │
│                                                            │
│  CPU: 100% ████████████████  Mining: 4,200 H/s            │
│  GPU: 100% ████████████████  Inference: 120 tok/s         │
│  RAM:  95% ████████████████  ZK Proofs: 8/s               │
│  NET:  90% ███████████████░  Tunnels: 12 active           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**What Each "Cheat" Actually Does**:

| Cheat | Real Implementation |
|-------|-------------------|
| INFINITE CORES | `core_affinity` pin + `SCHED_FIFO` real-time priority |
| GOD MODE MEMORY | `madvise(MADV_HUGEPAGE)` + `mlockall(MCL_CURRENT\|MCL_FUTURE)` |
| SPEED HACK x100 | Enable AVX-512/AVX2 SIMD + GPU compute + io_uring |
| WALL HACK | Subscribe to `/qnk/*/compute-tunnel` gossipsub, see all peers |
| AIM BOT | Adaptive task scheduler assigns work to optimal hardware |
| NO CLIP | `SCHED_FIFO` for mining, `nice -20` for inference, IRQ steering |
| INFINITE AMMO | Prefetch work queue, speculative task generation |
| RAPID FIRE | Batch 8 mining solutions per gossipsub message |
| TELEPORT | `splice()` zero-copy, `mmap()` zero-copy DB reads |
| PRESTIGE MODE | CPU governor `performance`, disable C-states, max turbo |
| NUKE | All of the above simultaneously |

**Implementation**:
- [ ] `crates/q-compute/src/trainer.rs` — Trainer engine
- [ ] Auto-detect hardware capabilities on startup
- [ ] Apply safe defaults, allow user to enable aggressive mode
- [ ] TUI overlay showing trainer status (ratatui)
- [ ] Log performance gains vs baseline
- [ ] `--trainer=full|safe|off` CLI flag

---

## Issue #005: Distributed AI Inference Pool
**Priority**: MEDIUM
**Assigned**: Epsilon
**Status**: Open

Split large LLMs across multiple nodes using tensor parallelism.
Users pay QUG for inference. Nodes earn inference fees.

- [ ] Wire q-ai-inference into compute orchestrator
- [ ] Tensor parallelism: split model layers across nodes
- [ ] KV-cache sharing via gossipsub
- [ ] Inference pricing oracle (QUG per token)
- [ ] API: `POST /api/v1/compute/inference`

---

## Issue #006: ZK Proof Farm
**Priority**: MEDIUM
**Assigned**: Gamma
**Status**: Open

Background ZK proof generation using idle compute.
Other users/apps can request proofs and pay QUG.

- [ ] zk-STARK proof generation as background task
- [ ] GPU-accelerated NTT (Number Theoretic Transform)
- [ ] Proof marketplace API
- [ ] Recursive proof batching (amortize cost)
- [ ] Verification: any node can verify in O(log n)

---

## Issue #007: OS-Level Auto-Tuning
**Priority**: HIGH
**Assigned**: Beta
**Status**: Open

Auto-detect OS and apply maximum performance settings on startup.

**Linux**:
- [ ] `sysctl -w net.core.somaxconn=65535`
- [ ] `sysctl -w vm.nr_hugepages=1024`
- [ ] `sysctl -w kernel.sched_min_granularity_ns=100000`
- [ ] CPU frequency governor → `performance`
- [ ] Disable transparent huge pages compaction
- [ ] Set IRQ affinity away from mining cores

**Windows**:
- [ ] Set process priority to HIGH
- [ ] Set thread affinity masks
- [ ] Disable power throttling
- [ ] Large pages privilege
- [ ] Disable Nagle algorithm on sockets

---

## Issue #008: Tunnel Mesh Visualization
**Priority**: LOW
**Assigned**: Beta
**Status**: Open

Frontend visualization showing compute tunnels between all nodes.
Real-time data flow, capacity heatmap, task routing.

```
     Beta ──────── Epsilon
    / │  \        / │
   /  │   \      /  │
 Gamma  Delta  Alpha │
   \    │    /      │
    \   │   /      /
     Windows Node ─┘

 ═══ = compute tunnel (thick = high bandwidth)
 ─── = P2P gossipsub
```

- [ ] D3.js force-directed graph
- [ ] Real-time bandwidth per tunnel
- [ ] Click node → see compute breakdown
- [ ] Animate task flow between nodes
