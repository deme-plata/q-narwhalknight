---
name: quillon-docker-test
description: Compile the current branch's q-api-server in Epsilon Docker (rust:bookworm, target-debian12 cache), then deploy fresh sync containers on Epsilon and Delta and watch them join mainnet-genesis until they reach tip or wedge. Invoke when the user says "docker test on epsilon and delta", "test the latest binary in docker", "run a sync smoke test", or any close paraphrase. Pass an optional version string as the argument; if absent, read it from Cargo.toml workspace.package.version.
---

# Quillon Docker Test — the first Quillon skill

## What this skill does

Three phases, all of them on Epsilon (never on Beta — Beta is the live dev
endpoint, see CLAUDE.md balance-integrity rules):

1. **Build** a Debian-12 release binary inside the `rust:bookworm` Docker
   image using the `/home/orobit/target-debian12/` incremental cache.
2. **Deploy** a fresh-DB sync container on Epsilon (port 8085) AND on Delta
   (SSH-jumped via Beta because Beta can SSH to Delta but Epsilon cannot).
3. **Monitor** both containers' sync progress for up to 1 hour: docker
   stats, docker logs (rate-limited grep), API status, tshark frame counts.
   Report on tip-reach, wedge, OOM, divergence vs production.

## When to invoke

Invoke when the user says things like:
- "docker test on epsilon and delta"
- "test the latest binary in docker"
- "run a sync smoke test"
- "go" after a build that needs validation
- "compile and docker test"

Do NOT invoke for:
- Production deploys (use `./scripts/ha-deploy.sh full -y` instead)
- Beta-side testing (Beta is live mainnet; never run a test container there)
- Slint/wallet/frontend work (this skill is node-only)

## Procedure

### Phase 0 — Preflight

1. Resolve the version:
   - If argument given, use it as the version tag (`$VERSION`).
   - Else read `Cargo.toml` workspace.package.version.
2. Confirm the branch is committed and pushed to the local git daemon:
   ```bash
   git status --short
   # if dirty: ask user whether to proceed without committing
   git log --oneline -1
   ```
3. Update server info so Epsilon can pull:
   ```bash
   git update-server-info
   ```

### Phase 1 — Build (Epsilon Docker)

Use `helpers/build.sh $VERSION`. The script:
- SSH to Epsilon, pulls the current branch on `/home/orobit/q-narwhalknight-src/`
- Launches `rust:bookworm` Docker with `/home/orobit/target-debian12` mounted as `/src/target`
- Runs `cargo build --release --package q-api-server`
- Streams output to `/home/orobit/tmp/build-v${VERSION}.log`
- Returns 0 if `Finished` line appears, non-zero on any `error[E`

**IMPORTANT** — do NOT pipe through `tail`. `tail` buffers until EOF and
hides 30+ minutes of compile progress. Stream raw and grep with
`--line-buffered`.

### Phase 2 — Deploy fresh containers

Use `helpers/deploy.sh $VERSION`. The script:
- On Epsilon: docker run `debian:12`, mount the just-built binary read-only,
  open ports 8085→8080 and 9005→9001, fresh DB at
  `/home/orobit/docker-sync-test-v${VERSION}/`, env:
  - `Q_NETWORK_ID=mainnet-genesis`
  - `Q_DB_PATH=/data/db`
  - `Q_P2P_PORT=9001`
  - `RUST_LOG=info`
  - `ROCKSDB_BLOCK_CACHE_MB=2048`
  - `Q_TOR_BOOTSTRAP_TIMEOUT=5` (else Tor blocks startup 120s)
- On Delta: SCP binary Beta→Delta (Epsilon cannot SSH Delta), then
  docker run with the same env but ports 8086→8080 and 9006→9001.
- Container name pattern: `q-sync-test-v${VERSION}-{epsilon,delta}`.
- Records `sync_start_epoch.txt` inside the container so monitor.sh can
  compute BPS.

### Phase 3 — Monitor

Use `helpers/monitor.sh $VERSION`. The script tails:
- Container `docker logs` filtered for height-progress lines + errors
- Container `docker stats` every 60s (CPU, RAM, network I/O)
- API status: `curl -s localhost:8085/api/v1/status` (height, peers)
- Optionally tshark on `docker0` for P2P frame counts (host-side, IP-filtered)

Emit events on:
- height progress every 5,000 blocks
- container exits unexpectedly (OOM, panic)
- height stalls for > 5 min at the same value (wedge)
- height diverges from Epsilon production by > 10 blocks (fork suspicion)
- successful tip-reach (height ≥ production height - 50)

Run for up to 1 hour; user can stop earlier with TaskStop on the Monitor.

## Final report shape

```
=== Quillon Docker Sync Test v${VERSION} ===
Epsilon container: <verdict>  final height <H_e>  ~<BPS_e> BPS  (uptime <T>)
Delta   container: <verdict>  final height <H_d>  ~<BPS_d> BPS  (uptime <T>)
Production tip   : <H_prod>  (reference)

Verdicts: TIP_REACHED | SYNCING | WEDGED | DIVERGED | CRASHED

Next step: <auto-recommended action>
  - If both TIP_REACHED:  ready for ha-deploy.sh full -y
  - If WEDGED:            kill container, check logs, file bug
  - If DIVERGED:          investigate fork; do NOT deploy
  - If CRASHED:           check stderr, file bug
```

## Hard-won lessons encoded in this skill

- **Never pipe cargo through `tail -N`**. `tail` buffers until EOF; you get
  no progress visibility. Stream raw or use `grep --line-buffered`.
- **Build for Debian 12** on Epsilon Docker, not on Epsilon's host. Epsilon
  host is Ubuntu 24.04 (glibc 2.39); Beta/Gamma/Delta are Debian 12
  (glibc 2.36). Native binaries are not portable.
- **Use `Q_P2P_PORT=9001` env var, NOT `--p2p-port` CLI flag.** Per
  CLAUDE.md memory, the CLI flag doesn't exist on q-api-server.
- **Always set `Q_TOR_BOOTSTRAP_TIMEOUT=5`** in test containers. Default
  is 120s which makes startup look hung.
- **Mount target-debian12 cache**. Without it, cold build is 30+ min.
  With cache, incremental is 5-25 min.
- **Container's binary mount is read-only**: `-v /path/to/binary:/opt/q-api-server:ro`.
  The container's bash copies it to `/usr/local/bin/q-api-server` to get
  exec perms.

## See also

- `CLAUDE.md` — "DOCKER SYNC TESTING ON EPSILON (Debian 12)" section
- `docs/starship-endgame/STATUS-2026-03-10.md` — history of canary testing
- Memory `cluster_topology_corrected.md` — Beta=dev, Epsilon=prod
