# Q-NarwhalKnight Developer Setup — Multi-Agent Git Collaboration

## Quick Start for New Claude Code Terminals

### 1. Clone the Repository

```bash
# Clone from Beta's git daemon (fastest, read+write)
git clone git://185.182.185.227/q-narwhalknight /opt/orobit/shared/q-narwhalknight-dev
cd /opt/orobit/shared/q-narwhalknight-dev

# Or shallow clone (faster for large repos):
git clone --depth 1 -b feature/safe-batched-sync-v1.0.2 git://185.182.185.227/q-narwhalknight /opt/orobit/shared/q-narwhalknight-dev
```

### 2. Set Up Git Identity

```bash
git config user.name "Claude Worker N"
git config user.email "claude-worker-N@q-narwhalknight.dev"
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Build & Verify

```bash
# Quick syntax check
timeout 600 cargo check --package <your-crate>

# Full workspace check (takes longer)
timeout 3600 cargo check --workspace
```

### 5. Push Changes Back to Beta

```bash
# Push your branch to Beta's git daemon
git push origin feature/your-feature-name
```

---

## Server Info

| Server | IP | Role | Git Port |
|--------|-----|------|----------|
| **Beta** (origin) | `185.182.185.227` | Primary dev server, git daemon | `9418` (git://) |
| **Epsilon** | `89.149.241.126` | 48-core build server, production | SSH only |
| **Gamma** | `109.205.176.60` | Backup node | SSH only |

## Repository Structure

```
q-narwhalknight/
├── Cargo.toml              # Workspace root
├── crates/
│   ├── q-flux/             # High-performance reverse proxy (NEW)
│   ├── q-api-server/       # Main API server + blockchain node
│   ├── q-miner/            # Mining client
│   ├── q-storage/          # RocksDB storage engine
│   ├── q-types/            # Shared types
│   ├── q-network/          # P2P networking (libp2p)
│   ├── q-dex/              # Decentralized exchange
│   └── ... (40+ crates)
├── gui/
│   ├── quantum-wallet/     # React frontend
│   └── slint-wallet/       # Native desktop wallet
├── q-flux.toml             # q-flux config
└── CLAUDE.md               # Full development guide
```

## Build Notes

- **Rust version**: 1.86+ required
- **First build**: ~30 min (PQ crypto crates are slow)
- **Use timeouts**: `timeout 36000 cargo build` — never run cargo without timeout
- **Release build**: `cargo build --release --package <crate>`
- **After changing constants**: `cargo clean --package <crate>` before rebuild

## Branching Convention

- `main` — stable release
- `feature/safe-batched-sync-v1.0.2` — current active development branch
- `feature/<name>` — new feature branches

## How to Submit Work

1. Create your feature branch
2. Make changes, commit with descriptive messages
3. Push to Beta: `git push origin feature/your-branch`
4. The lead developer will review and merge

## Current Open Tasks

**IMPORTANT: Issues are NOT on GitHub!** Issues are tracked in `ISSUES.md` in the repo root.
Do NOT use `gh` CLI or GitHub API. Read `ISSUES.md` for open tasks.

```bash
cat ISSUES.md   # See all open issues
```

### Quick Issue Summary:
- **Issue #1** (`feature/q-queue`): Build `q-queue` — lock-free universal queue system with io_uring, SIMD, <500ns IPC latency
- **Issue #2** (`feature/q-flux-phase2`): q-flux Phase 2 — io_uring + SIMD HTTP parsing
- **Issue #3** (`feature/q-flux-phase3`): q-flux Phase 3 — HTTP/2, QUIC, kTLS
- **Issue #4** (`feature/q-flux-phase4`): q-flux Phase 4 — libp2p compatibility

### Workflow:
1. Read `ISSUES.md` for full specs
2. Pick an unassigned issue
3. Create the feature branch listed in the issue
4. Implement, test with `cargo check`
5. Push: `git push origin feature/your-branch`
