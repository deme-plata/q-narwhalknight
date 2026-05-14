# Quillon Graph v10.9.16 — Release Notes

**2026-05-13**

A focused release with two things miners will notice immediately and one thing fresh-node operators have been asking for. No protocol changes, no upgrade fork, drop-in replacement of the previous binary.

---

## Download

```
wget https://quillon.xyz/downloads/q-api-server-v10.9.16
chmod +x q-api-server-v10.9.16
./q-api-server-v10.9.16 --port 8080
```

For the auto-update path on existing installations, the file at `q-api-server-linux-x86_64` has been refreshed — your `slint-wallet` and update-aware tooling will pick it up on the next version check.

**SHA-256:** `fc39c63b9cd204df58a1e660d7f2ea53dcf8a4854486494d50750e8508c600f2`
**Size:** 89 MB
**Target:** Linux x86-64, glibc 2.36+ (Debian 12, Ubuntu 22.04+, modern derivatives)

---

## What's new for miners

### Submissions no longer get silently rejected under load

We tuned the reverse proxy in front of the genesis API node. Before today, when many miners submitted shares at once, the proxy was running out of upstream slots and rejecting requests with HTTP 502 after 100 milliseconds — a 100-millisecond timeout you didn't see, but your miner did. We watched it happen ~48 times per second sustained, including from miners doing nothing wrong.

The buffer is now 4× larger (2048 vs 512 slots) and the wait window is 7.5× longer (750 ms vs 100 ms). After the change, the same workload produces fewer than one error per second instead of forty-eight.

**You should see**: a drop in your miner's "submit failed / retrying" rate, and a noticeable bump in accepted shares per hour if you were near the rejection threshold before.

**You don't need to upgrade your miner to see this benefit** — it's a server-side fix that helps every existing miner version. The v10.9.16 client binary on its own doesn't change the mining protocol.

---

## What's new for node operators

### Fresh nodes now find peers much faster

When a freshly-started node first joins the network, libp2p has to negotiate with the four bootstrap peers (Epsilon, Delta, Gamma, Beta-prod). On a quiet day this works fine. On a busy day — or any day where one of the four peers is having a bad minute — the previous code would settle for **one** connection and stop trying for the others. The result was a brand-new node with one peer, syncing at maybe 9 blocks per second instead of 150.

v10.9.16 reworks this in three concrete ways:

1. **Epsilon is dialed first, every time.** Previously the dial order was randomized by Rust's `HashMap` (intentional, to prevent collision attacks — but it meant Epsilon, the 10 Gbit supernode, sometimes came last). Now the dial order follows the declaration order in the bootstrap list. Epsilon TCP 9001 is always the first dial.

2. **The warmup keeps trying until at least 3 peers are connected** (or it gives up cleanly after ~100 seconds with a useful error). Previously it stopped at the first peer regardless. New default target is 3; tunable via `Q_BOOTSTRAP_MIN_PEERS` if your environment is unusual.

3. **Early-bail when the network is unreachable.** If a node behind a heavy corporate firewall hits zero connections and zero pending dials after 21 seconds, the warmup aborts with a clear "network unreachable" log line and a hint to check firewall / set `Q_BOOTSTRAP_PEER`. Previously the node would hang the full 100 seconds before giving up silently.

### New `--from-genesis` flag

If you want a node that syncs every single block from height 0 (instead of bootstrapping from the BAL-001 balance checkpoint at height ~16.5M), just add the flag:

```
./q-api-server-v10.9.16 --port 8080 --from-genesis
```

This is what trustless-audit setups want: a node that builds the entire chain locally, never trusting any bootstrap snapshot. It also makes the test/development experience smoother — no more remembering to set `Q_SKIP_CHECKPOINT=1` *and* `Q_GENESIS_SYNC_ONLY=1` in the right order (env vars must come before the binary on the command line; if you got the order wrong, the flags silently failed).

Most users **should not use this flag**. A fresh sync from genesis takes 5-12 hours depending on your bandwidth. The default checkpoint bootstrap is the right path for normal mining and wallet use.

### Memory pressure during pre-checkpoint backfill is bounded

Continuing the v10.9.14 line: nodes that bootstrap from the BAL-001 checkpoint and then backfill the historical blocks (so the explorer can serve full chain history) used to occasionally hit OOM at ~8 GB or ~16 GB depending on Docker memory limits. v10.9.14 introduced a memory budget gate that watches cgroup-reported RSS and pauses dispatch when over a soft threshold. v10.9.16 ships this as the default. A test container has been running on Epsilon since this morning — RSS stays steady around 4 GB throughout Phase 2 backfill, no OOM events.

---

## What's new visually

Anyone launching with `--tui` will see the boot sequence has been completely redesigned:

```
    ────────────────────────────────────────────────────────────────────

         ██████╗     ██╗   ██╗   ██████╗
        ██╔═══██╗    ██║   ██║  ██╔════╝     Quillon Graph
        ██║   ██║    ██║   ██║  ██║ ███╗     quantum-enhanced
        ██║▄▄ ██║    ██║   ██║  ██║  ██║     DAG-BFT consensus
        ╚██████╔╝    ╚██████╔╝  ╚█████╔╝     codename · NarwhalKnight
         ╚══▀▀═╝      ╚═════╝    ╚════╝

         v10.9.16  ·  18 cores  ·  host vmi2628966

         "When silicon dreams of qubits, the chain endures."

    ────────────────────────────────────────────────────────────────────

    ▶ Initializing node  ·  press q once TUI loads to quit cleanly

    [01] ✓ Memory protection      THP disabled · jemalloc engaged
    [02] ✓ Async runtime          18 workers / 18 cores · 100%
    [03] ✓ Crypto provider        rustls + ring loaded
    [04] ⚠ Environment            no .env found · using process env / defaults
    [05] ✓ Config validation      checked
    [06] ✓ Arguments parsed       port=8080 tui=on
    [07] ✓ Network                mainnet-genesis
    [08] ✓ Database               ./data-mainnet-genesis
    [09] ✓ Admin wallet           efca1e8c1f…e50723
    [10] ▶ Starting subsystems    RocksDB · libp2p · gossipsub · API · TUI
```

The tagline rotates from a small pool of quantum/blockchain lines, picked deterministically from the version string — every release has its own quote, stable per binary so you can predict it.

Functionally identical to before. Just a nicer thing to look at while the node spins up.

---

## What's coming next

Several pieces are queued and partly written but not in this release:

- **Trustless light-client mode (instant-bootstrap recursive zk-SNARK)** — the work that lets a fresh node start mining and serving transactions in 10 milliseconds instead of 5-12 hours, with cryptographic certainty instead of trusting a checkpoint. We've written the technical plan, the engineering blueprints, and shipped the sparse Merkle tree storage layer in shadow mode (it runs but doesn't yet feed consensus). Real activation is a 2027 milestone — months of soak per CLAUDE.md mainnet-safety rules. We don't shortcut this.

- **Progressive archival in the explorer** — when a fresh node hasn't backfilled an old block yet, the API will return HTTP 202 with an ETA instead of HTTP 404. The wallet UI will render "block #X not yet indexed on this node, ETA 18h, [query peer]" instead of failing silently. Independent of the SNARK work; this lands in a near-term v10.9.x.

- **TUI fast-readiness panel** — a persistent readiness banner that shows `FAST-READY`, `CHECKPOINT-TRUST`, `GENESIS-SYNC`, or `ARCHIVE-COMPLETE` at all times, plus a capability matrix (mine ✓, transact ✓, query state ✓, query history ◐ backfilling). Built to show the moment a node transitions from "trust me" to "cryptographically verified." Ships in v10.9.x.

- **WSS fallback when TCP is blocked**, **per-peer dial-error reason logging**, and a few other peer-discovery enhancements. Most of the underlying error-logging is already there; only WSS fallback is missing in v10.9.16. Coming in v10.9.17.

---

## Upgrade procedure

For the genesis nodes (Beta, Gamma, Delta) the rolling-deploy pipeline (`scripts/ha-deploy.sh`) will be used in the next maintenance window. End users do not need to do anything — the binary at `q-api-server-linux-x86_64` updates with each release.

For your own node:

```
# stop the running node cleanly (Ctrl-C if foreground, or systemctl stop q-api-server)
wget https://quillon.xyz/downloads/q-api-server-v10.9.16
chmod +x q-api-server-v10.9.16
# replace your existing binary or symlink, then start as usual
./q-api-server-v10.9.16 --port 8080
```

Your data directory, wallet, and identity keys are unchanged. No reindex, no resync, no migration. The binary boots up exactly where the previous one left off.

---

## Reporting issues

Discord: `#node-operators` or `#mining` channels.
Bitcointalk: the Quillon Graph thread.

Particular things to look for and let us know:
- Whether your fresh-node sync speed visibly improves (compare blocks/sec in the TUI before and after).
- Whether you see the `🔄 [CONNECTION WARMUP v10.9.16]` log line in `journalctl -u q-api-server` (this is a signal the new code is active).
- Whether the rotating tagline catches your eye — we picked 8 lines from a pool of physicist-blockchain crossovers; if you have ideas for a v9 line, send it.

---

— Quillon Graph maintainers, 2026-05-13
