# Sparse-Chain Reality — What Node Operators Need to Know

**Status**: This document explains why fresh `q-api-server` installs appear
to wedge during sync, what's actually happening, and how to recover.
Required reading for anyone running a new node from genesis.

**TL;DR**: The Quillon Graph mainnet chain is **sparse below height ~7M**
(only ~3% of heights have blocks). Fresh nodes trying to genesis-walk to
tip will appear stuck for hours unless they know this. Set
`Q_FAST_SYNC=1` to skip the sparse historical region via checkpoint,
or set `Q_KNOWN_PERMANENT_GAPS` to whitelist sparse ranges.

---

## The reality

Most blockchains have a contiguous chain from genesis to tip:

```
heights:   0  1  2  3  4  5  6  7  8  9  10 ...
blocks:    ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ...
density:   100%
```

Quillon Graph is **NOT** contiguous. Two distinct regions exist:

```
height 0 ---------------- 7M ------------------ 15M ------- 18.2M (tip)
density:  ████░░░░░░░░░░░░░ █████████████████████ ████████████████
                ~3%                   ~40%             93-96%
       (historical damage)    (protocol-natural)    (current operation)
```

- **Pre-7M heights** are ~3% populated. ~97% of heights between 0 and 7M
  have no block at all. This is **historical damage**, not protocol design:
  - March 2026 RocksDB compaction loss
  - April 2026 v10.2.8 cleanup bug
  Both are documented in `docs/technical-review-sparse-chain-truth-v1.md`.

- **7M-15M** is moderately dense (~40%) — partial recovery of older data.

- **Post-15M** is 93-96% dense. This is **protocol-natural** DAG-Knight
  sparsity — heights advance non-contiguously when validator anchors
  fall in non-consecutive rounds. This is normal and expected forever.

---

## What this means when you run `./q-api-server`

A fresh node starts at height 0 and tries to sync forward. With the
default sync algorithm, here's what happens:

1. Request `qblock:height:1` from a peer
2. Peer returns "not found" (height 1 is in the 97%-empty zone)
3. Fresh node treats this as "peer doesn't have height 1 yet" and waits
4. After timeout, retries with different peer → same answer
5. **Sync wedges. Operator sees no progress for 15-30 min and gives up.**

This is why **87 days went by with zero organic node onboardings**. New
users hit this trap, conclude "the software is broken", and walk away.

---

## How to actually sync a fresh node

You have three options, ordered by recommended-first:

### Option 1: Checkpoint sync (RECOMMENDED for new operators)

```bash
Q_NETWORK_ID=mainnet-genesis \
Q_FAST_SYNC=1 \                          # ← jumps past sparse region
Q_TOR_BOOTSTRAP_TIMEOUT=5 \
./q-api-server-v10.10.2 --port 8080 --admin-wallet qnk<YOUR>
```

`Q_FAST_SYNC=1` makes the node download a checkpoint snapshot of recent
state (post-15M, in the dense region) and start from there. You don't
get the historical pre-7M heights, but you ARE a fully-participating
node on the current chain within ~5-30 minutes (vs hours-to-never
for genesis walk).

**Tradeoff**: you trust the checkpoint signers (Beta + Gamma + Epsilon).
For a non-validator user, this is the same trust model as any other
node-operator's "trust the genesis hash" assumption.

### Option 2: Genesis sync with sparse-aware config

```bash
Q_NETWORK_ID=mainnet-genesis \
Q_KNOWN_PERMANENT_GAPS=25988:100440,250000:500000,2500000:3000000 \
                                          # ← whitelist known sparse ranges
Q_GAP_TRUST_SINGLE_PEER=1 \              # ← trust gap reports from single peer
Q_GENESIS_SYNC_ONLY=1 \                  # ← stay in lookahead window
Q_SKIP_BALANCE_REPLAY=1 \                # ← per CLAUDE.md rule 2
Q_TOR_BOOTSTRAP_TIMEOUT=5 \
./q-api-server-v10.10.2 --port 8080 --admin-wallet qnk<YOUR>
```

Sync walks from genesis, but knows to skip the listed gap ranges.
This gives you a fuller historical view but takes 5-12 hours and is
fragile — operators not on this exact set of env vars get the wedged
behavior above.

`Q_KNOWN_PERMANENT_GAPS` is comma-separated `start:end` ranges. The
example above covers the worst pre-100K gap, the 250K-500K dense void,
and the 2.5M-3M sparse area. More gaps exist (the chain is genuinely
sparse, not a few discrete gaps), but these three are the load-bearing
ones for getting unstuck.

### Option 3: Snapshot import (fastest for ops)

If you have RocksDB-level access to an existing healthy node:

```bash
# On healthy node:
systemctl stop q-api-server
tar czf chain-snapshot.tar.gz data-mainnet-genesis/
systemctl start q-api-server

# On new node:
scp <healthy>:chain-snapshot.tar.gz .
tar xzf chain-snapshot.tar.gz
./q-api-server-v10.10.2 --port 8080 --admin-wallet qnk<YOUR>
```

New node starts at the snapshot's height, no sync needed. ~30 minutes
to first-block-produced. Used for HA-rolling deployments per
CLAUDE.md `ha-deploy.sh`.

---

## What v10.10.2 changes

v10.10.2 bakes the most-important fresh-install defaults into the binary
so fresh `./q-api-server` runs are workable out-of-box:

- `Q_NETWORK_ID=mainnet-genesis` (was `testnet` — caused "no peers" trap)
- `Q_TOR_BOOTSTRAP_TIMEOUT=5` (was 120 — blocked startup for 2 min)
- `Q_ROCKSDB_WRITE_RATE_MB=400` (was 200 — throttled turbo-sync)
- Refreshed all 3 stale hardcoded peer-IDs (Beta/Gamma/Delta)

v10.10.2 does **NOT** yet bake `Q_FAST_SYNC=1` as default. That's a
behavioral change with security implications (trust model shifts from
"genesis is the only trusted thing" to "checkpoint signers are also
trusted"). Operators must opt in via env var explicitly.

Future v10.11.x will offer a `--mode quick-start` flag that:
1. Auto-uses checkpoint sync
2. Auto-sets gap whitelist for known sparse zones
3. Skips problematic balance replay
4. Documents the trust assumption it makes

---

## Symptoms checklist — which problem are you hitting?

If you see this in TUI or logs:

| Symptom | Likely cause | Fix |
|---|---|---|
| "WrongPeerId" warnings, then connections succeed | Stale hardcoded peer-IDs | Fixed in v10.10.2 — upgrade |
| Stuck at height 0 for 30+ min, has peers | Sparse-chain genesis-walk wedge | Add `Q_FAST_SYNC=1` |
| `current_height` matches `safe_floor` and stops | Hit a sparse gap | Add `Q_KNOWN_PERMANENT_GAPS` |
| No peers found, network reports empty | `Q_NETWORK_ID=testnet` default | Fixed in v10.10.2 — upgrade |
| Sits on "Bootstrapping Tor" for 2 min | `Q_TOR_BOOTSTRAP_TIMEOUT=120` default | Fixed in v10.10.2 — upgrade |
| Setup wizard opens browser link | Fresh install with no `--admin-wallet` | Pass `--admin-wallet qnk<YOUR>` |
| Process in state `T (stopped)` | Accidental Ctrl-Z in TUI | `kill -CONT <pid>` to resume |
| Sync runs but `qnk_peers_connected = 0` | Cosmetic metric bug | Ignore — check `qnk_libp2p_rx_bytes_total` for real traffic |

---

## Future work

This document describes the current state honestly. The chain WILL get
denser over time as forward-progress fills in. The sparse pre-7M region
is permanent — it cannot be recovered, only papered over with snapshot/
checkpoint trust.

Tracked work that improves the operator experience:

- **v10.11.x**: `--mode quick-start` flag (one-command checkpoint sync)
- **bitmap-diff sync** (`docs/v10.9.56-bitmap-diff-sync-design.md`): peers
  exchange compact bitmaps of which heights they have, removing the
  guesswork from sync
- **AEGIS-QL trust bootstrap**: lower bar for fresh nodes to acquire
  first-trusted-peer relationship (currently requires manual operator
  attention to register first peer)

---

## References

- `docs/technical-review-sparse-chain-truth-v1.md` — empirical density
  analysis (~3% pre-7M, 93-96% post-15M)
- `docs/technical-review-sparse-dag-sync-optimization-v1.md` —
  algorithm design for sparse-aware sync
- `docs/technical-review-turbo-sync-sparse-fix.md` — v10.9.41+ fix
  rationale
- `docs/v10.9.55-balance-replay-removal-2026-05-18.md` — why chain-
  replay balance migration is unsafe on sparse chains
- `docs/v10.9.56-bitmap-diff-sync-design.md` — future bitmap-based sync

If you hit issues not covered here, file an issue with:
1. Your `q-api-server` version (`--version`)
2. Output of `/metrics` endpoint (filter `qnk_*` lines)
3. Last 50 lines of node logs

We track Node Operator issues in `docs/qnk-node-operator-guide.pdf`.
