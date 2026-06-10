# Instructions for Claude Code Agents

You are working on q-flux, a high-performance TLS reverse proxy.

## Getting Started

1. Read the task list:
```
cat /opt/orobit/shared/q-narwhalknight/crates/q-flux/CONTRIBUTING.md
```

2. Pick a task (TASK 1-5). Create a branch for your work:
```
cd /opt/orobit/shared/q-narwhalknight
git checkout -b flux/task-N-short-description
```

3. Read the relevant source files before making changes:
- `crates/q-flux/src/proxy.rs` — HTTP/WebSocket handler (main integration point)
- `crates/q-flux/src/worker.rs` — Worker thread setup, creates PeerTracker
- `crates/q-flux/src/upstream.rs` — Connection pool, super-cluster failover
- `crates/q-flux/src/libp2p_aware.rs` — BandwidthLimiter, CircuitBreaker, GossipsubDedup, PeerTracker
- `crates/q-flux/src/admin.rs` — Admin HTTP server (port 9090)
- `crates/q-flux/src/config.rs` — TOML config structs

4. After making changes, verify:
```
cargo check --package q-flux
```

5. Commit your work:
```
git add -A crates/q-flux/
git commit -m "feat(q-flux): short description of what you did"
```

## Rules

- Do NOT touch files outside `crates/q-flux/`
- Do NOT deploy or restart any services
- Do NOT run `cargo build --release`
- Do NOT push to remote or run ha-deploy.sh
- Only `cargo check` — no full builds, the maintainer handles builds and deploys
