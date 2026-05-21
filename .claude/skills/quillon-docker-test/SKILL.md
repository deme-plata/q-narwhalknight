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

### Phase 4 — Post-deploy verification (MANDATORY on production restart)

If the user promotes the new binary to production (i.e. the cowboy path:
SCP binary into the deploy dir, update the `q-api-server-stable` symlink,
`systemctl restart q-api-server`), the skill MUST run
`helpers/verify_post_deploy.sh $VERSION` afterward. It checks four things:

1. The running PID's `/proc/$pid/exe` actually resolves to the new
   versioned binary (catches "symlink swap forgotten" / "systemd cached
   old path" mistakes).
2. `GET /api/v1/status` returns a height (catches "service active but
   binary panicked silently" — systemd will report active even when the
   inner process exited on a bad path).
3. The height advances over a 15-second sample (catches "service up but
   wedged on a bad migration" — important for any release that changes
   storage schema or balance accounting).
4. **The operator's wallet balance is intact.** Signed via X-Wallet-Auth
   against `/api/v1/wallets/$WALLET/balance`. This is the single most
   important post-deploy check: it confirms that (a) the new binary's
   auth path still works, (b) the agent's QUG balance was not zeroed or
   regressed by the new release. The skill bit this gap on its first
   real invocation 2026-05-21; the operator caught it.

Skip this phase ONLY if you didn't touch production — pure docker-test
runs don't need it.

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

## Sparse-chain reality (CRITICAL — read before interpreting any sync result)

Q-NarwhalKnight's mainnet-genesis chain is **sparse by design AND by historical
damage**. See `docs/technical-review-sparse-chain-truth-v1.md`. The short version:

| Height range | % present | Cause |
|---|---|---|
| 0 – 7M     | ~3%      | Compaction loss (kill -9, Mar 2026) + v10.2.8 cleanup |
| 7M – 14M   | ~50%     | v10.2.8 `qblock:height:` cleanup damage |
| 14M – 15M  | 78%      | Cleanup tail + protocol sparsity |
| 15M – tip  | 93–96%   | **Pure DAG-Knight design** — not every round produces an anchor |

Implications for sync testing:

- **Do NOT use `data.upgrades.current_height` from `/api/v1/status` as a height signal.**
  That field tracks the *contiguously-applied* tip for the upgrade gate. On a fresh
  node syncing this chain it stays at 0 forever — even after applying millions of
  discrete blocks — because no contiguous range exists from genesis. Old monitor.sh
  v1 made this mistake and would have reported any fresh sync as WEDGED.
- **Do read from logs.** monitor.sh v2 extracts height from three sources, in
  priority order:
  1. **Progress-bar line** (preferred): `┃ ⏳ Waiting for blocks... | <H>/<TIP> | <BPS> b/s`
  2. **HTTP BOOTSTRAP** (fallback): `[HTTP BOOTSTRAP] ... our_height=<H>`
  3. **synced_through** (v10.9.55+, sparse-chain-aware): `synced_through=<H>`
- **Don't panic at 30 b/s in the 0–7M range.** That's the floor on this chain —
  the sync code is asking for `qblock:height:N+1` and most of those don't exist.
  Truly stuck means same height for >5 min. Slow ≠ stuck.
- **Expected BPS by range** (Epsilon 10Gbit, fresh DB):
  - 0–7M:     ~30 b/s    (heavy sparsity → many requests for absent keys)
  - 7M–14M:   ~100–300 b/s  (mixed sparsity)
  - 14M–tip:  ~280–1100 b/s  (dense)
- **A fresh full sync of all 18M+ blocks takes ~6 days on Epsilon.** Don't expect
  TIP_REACHED in a 1h smoke test from genesis. The 1h test should validate boot +
  peer discovery + steady progress in whatever range the node is in.

## Hard-won lessons encoded in this skill

- **VERSION arg auto-strips a leading `v`.** Pass `10.11.3-latest` or `v10.11.3-latest`
  — the helpers normalize to no-prefix and add `v` themselves for filenames /
  container names. Passing `v...` to a v1 helper produced `q-sync-test-vv10.11.3-...`.
- **Concurrent build collision.** Two `qnk-build-*` containers race for the cargo
  target dir lock and the second one stalls indefinitely on git-submodule fetch
  for cutlass/candle. `build.sh` now aborts if any `qnk-build-*` is already up.
- **Port-collision from stale test containers.** A previous version's
  `q-sync-test-vX-epsilon` holding port 8085 blocks the new deploy with "port
  already allocated". `deploy.sh` now pre-checks and aborts with a `docker rm -f`
  hint.
- **Source sync route — origin can be wrong remote.** Local
  `agent/cross-shard-simd-validation` and the same-named branch on GitHub
  `origin` can have **disjoint histories** (verified 2026-05-21: `git merge-base`
  returned empty). The Beta git daemon is the authoritative source:
  ```bash
  ssh root@89.149.241.126 \
    "cd /home/orobit/q-narwhalknight-src && \
     git fetch git://185.182.185.227:9418/q-narwhalknight <branch> && \
     git checkout <branch> && \
     git reset --hard FETCH_HEAD"
  ```
  Don't rely on `git fetch origin` to land local changes on Epsilon.
- **Never pipe cargo through `tail -N`**. `tail -N` (limit count) buffers until
  EOF; you get no progress visibility. `tail -F` (follow) is fine and is what
  the build Monitor uses.
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
- **Setup wizard auto-runs on first boot.** Without an admin wallet, the binary
  blocks on an interactive OAuth2 login URL. `deploy.sh` passes
  `--admin-wallet <addr>` to bypass it for test containers. Without that flag,
  `docker logs` shows `NODE SETUP — LOGIN` and the container hangs forever.
- **mawk vs gawk.** Beta's `/usr/bin/awk` is `mawk`, which doesn't support
  gawk's 3-arg `match($0, regex, capture_array)`. Use `grep -oE` pipelines
  for capture-group extraction in helpers that run on Beta.
- **Direct-IP for PROD_API.** `https://quillon.xyz/api/v1/status` was observed
  routing to a wrong backend (a fresh-syncing node at height 1.21M instead of
  Epsilon's true 18.17M tip, peer ID `12D3KooWEZKN...`, IPv6 multiaddr). Always
  hit Epsilon directly via `http://89.149.241.126:8080/api/v1/status` for the
  authoritative tip.

## Run history

| Date       | Version            | Verdict       | Lesson learned                                                                           |
|------------|--------------------|---------------|------------------------------------------------------------------------------------------|
| 2026-05-21 | v10.10.13          | TIP_REACHED → shipped | Phase 4 (post-deploy wallet-balance verification) added                          |
| 2026-05-21 | v10.11.3-latest    | **CRASHED** (exit 139)  | Caught axum-overlapping-route panic — `/api/v1/integrity/balance-root` registered twice in `main.rs:25572` (v10.11.0 handler) + `main.rs:25752` (v10.7.0 integrity_api). Skill v2 lessons baked in: sparse-chain log-based height (no API field), double-v fix, concurrent-build + port-collision guards, mawk-compatible grep pipelines, direct-IP PROD_API (q-flux LB on quillon.xyz routes to wrong backend), `--admin-wallet` flag in container start to skip setup wizard. |

## See also

- `docs/technical-review-sparse-chain-truth-v1.md` — **the authoritative reference**
  for why fresh nodes look stuck (they aren't — the chain is sparse).
- `CLAUDE.md` — "DOCKER SYNC TESTING ON EPSILON (Debian 12)" section
- `docs/starship-endgame/STATUS-2026-03-10.md` — history of canary testing
- Memory `cluster_topology_corrected.md` — Beta=dev, Epsilon=prod
- Memory `sparse_chain_awareness.md` — short version of the sparse-chain reality
