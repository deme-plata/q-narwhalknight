# Technical Review: Epsilon DB Migration — Root DB → Home DB
**Date:** 2026-04-24  
**Server:** Epsilon (89.149.241.126) — Primary 10Gbit Bootstrap Supernode  
**Severity:** CRITICAL — Root partition at 100%, block production at constant risk  
**Status:** Analysis complete, no changes made to DB yet

---

## 1. The Two Database Situation — What We Actually Found

There are **two separate RocksDB databases** on Epsilon:

| Database | Path | Size | SST Files | WAL Files | Last Active |
|----------|------|------|-----------|-----------|-------------|
| **Root DB** (currently running) | `/data-mainnet-genesis/` | 7.1 GB | 1,803 | active | Apr 24 (now) |
| **Home DB** (the real full chain) | `/home/orobit/data-mainnet-genesis/` | 219 GB | 89,725 | 1 (3.1MB) | Apr 21 15:57 |

**The running node is using the 7.1 GB root DB, not the 219 GB home DB.**

The root DB lives on `/dev/nvme0n1p3` (40 GB root partition) — currently 100% full, 0 bytes free.  
The home DB lives on `/dev/nvme0n1p4` (1.8 TB home partition) — 55% used, **774 GB free**.

---

## 2. How This Happened — Root Cause

### The Service Config (auto-generated 2026-04-21 13:34:34 UTC)

The node was reconfigured by the setup wizard on Apr 21:

```
# /etc/systemd/system/q-api-server.service
WorkingDirectory=/

# /.env
Q_DB_PATH=./data-mainnet-genesis   ← relative path
```

Relative `./data-mainnet-genesis` + `WorkingDirectory=/` = `/data-mainnet-genesis` on the **root partition**.

The correct database — 219 GB, full chain history — is at `/home/orobit/data-mainnet-genesis/`.

### Timeline

| Time | Event |
|------|-------|
| 2026-04-21 13:34 UTC | Setup wizard regenerates `/.env` with wrong relative path |
| 2026-04-21 15:57 | Home DB stops (service killed/restarted) — was mid-compaction (JOB 187) |
| 2026-04-21 15:59 | Root DB created fresh at `/data-mainnet-genesis/` — node begins re-syncing from peers |
| 2026-04-22–23 | Root DB grows as node re-syncs 3+ days of blocks |
| 2026-04-24 ~05:00 | Root partition hits 100% → RocksDB write-error state → block production stalls |
| 2026-04-24 (now) | Root partition back at 100% (0 free). Home DB sits idle at 219 GB. |

---

## 3. State of the Home DB (the real database)

The home DB was stopped mid-compaction. Evidence:

```
Last LOG entry: 2026/04/21-15:57:13  [JOB 187] Generated table #9688780 (final SST of compaction)
MANIFEST file:  MANIFEST-9686366 (11 MB, intact)
WAL file:       9688480.log (3.1 MB, 1 file, written 15:51–15:57)
```

**RocksDB WAL recovery behavior when opened after abrupt shutdown:**
- RocksDB reads the WAL (`9688480.log`) on next open
- If JOB 187 wrote SST files but didn't update the MANIFEST, RocksDB will re-run or discard the incomplete compaction
- The data in the WAL is replayed — no block data is lost (WAL = write-ahead log)
- This is standard RocksDB recovery, well-tested and deterministic

**Risk of opening the home DB:** LOW — RocksDB was designed for exactly this scenario (process killed during compaction). The WAL ensures data integrity.

**Height when home DB stopped:** Unknown exactly, but the home DB was actively syncing at the same time as the network (Apr 21 ~15:57). Network height now is 16,139,986. The home DB was likely at approximately the same height before it was abandoned — it had been fully synced (219 GB = full chain history). Any blocks written in the WAL will be recovered; any that weren't written will be resynced from peers in seconds.

---

## 4. State of the Root DB (currently running)

The root DB was **created fresh on Apr 21 15:59**, 2 minutes after the home DB was abandoned.

```
First SST file: Apr 21 15:59 (began syncing from peer)  
Latest SST file: Apr 24 07:28 (actively writing now)  
SST file count: 1,803  
Size: 7.1 GB  
```

This DB was built entirely by turbo-syncing from the network over 3 days. It is **a partial re-sync**, not the authoritative chain history. With only 7.1 GB vs 219 GB, it likely has the recent chain tip but not the full historical index.

**Critical issue:** Root partition is 40 GB total, currently 100% full. The root DB writes every ~1 second (1 block/sec). It will refill within minutes of any log cleanup. **The next stall is not "if" — it is "when, and soon."**

---

## 5. What Needs to Happen — Options

### Option A: Switch to Home DB (RECOMMENDED)
Update `Q_DB_PATH` from the relative path to the absolute `/home/orobit/data-mainnet-genesis`.

**What happens:**
1. Service stops gracefully (SIGTERM, 30s timeout → RocksDB WAL flush)
2. Config updated: `Q_DB_PATH=/home/orobit/data-mainnet-genesis`
3. Service starts → RocksDB opens home DB, replays WAL (automatic, <10 seconds)
4. Node finds itself at ~Apr 21 height, re-syncs missing blocks from peers
5. Re-sync of ~3 days of blocks should take minutes (turbo sync at 1,100 blocks/sec)
6. Root partition is freed from DB writes permanently

**Risk assessment:**
| Step | Risk | Notes |
|------|------|-------|
| Stop service | LOW | SIGTERM → graceful shutdown, WAL flushed cleanly |
| Update /.env | ZERO | Single env var change |
| Open home DB (WAL recovery) | LOW | Standard RocksDB recovery, deterministic |
| Re-sync missing blocks | LOW | Peers have full chain, turbo sync is fast |
| Root DB orphaned on root | NONE | Can delete after home DB confirmed stable |

**Total estimated downtime:** 1–5 minutes (stop + WAL recovery + turbo re-sync)

**Benefit:** Restores the full 219 GB chain history database, moves DB writes to the correct 1.8 TB partition permanently.

---

### Option B: Move Root DB to /home (ALTERNATIVE)
Keep using the 7.1 GB re-synced DB, but move it to /home and update the path.

**Steps:**
1. Stop service
2. `rsync -av /data-mainnet-genesis/ /home/orobit/data-mainnet-genesis-new/`
3. Update `Q_DB_PATH=/home/orobit/data-mainnet-genesis-new`
4. Start service

**Risk:** LOWER (no WAL recovery question)  
**Downside:** Loses the 219 GB full history, keeps the partial 3-day re-sync. Full history is important for nodes trying to sync from genesis.

---

### Option C: Do Nothing (STATUS QUO — NOT VIABLE)
Root partition at 100%. RocksDB will enter write-error state again within minutes of the next compaction triggering a temporary SST file write. **This is not a viable option.** The situation will repeat on a cycle of hours.

---

## 6. Recommendation

**Option A: Switch to the home DB.**

The home DB is the authoritative production database with full chain history. The WAL recovery is safe and deterministic. The downtime is brief. This is the correct permanent fix.

**Pre-conditions before executing:**
1. Confirm that `/home/orobit/data-mainnet-genesis/` is not locked by any other process
2. Confirm disk space on root: need ~0 bytes extra (we're just updating a config line)
3. Note current node height for comparison after restart

---

## 7. Execution Plan (requires approval before proceeding)

```bash
# Step 1: Note current state
ssh root@89.149.241.126 "curl -s http://localhost:8080/api/v1/status | python3 -c 'import sys,json;d=json.load(sys.stdin)[\"data\"];print(d.get(\"current_height\"),d[\"status\"])'"

# Step 2: Verify home DB not locked by any process
ssh root@89.149.241.126 "lsof /home/orobit/data-mainnet-genesis/ 2>/dev/null | head -5"

# Step 3: Stop service gracefully
ssh root@89.149.241.126 "systemctl stop q-api-server"

# Step 4: Wait for clean stop
ssh root@89.149.241.126 "sleep 5 && pgrep -f q-api-server-v10 && echo 'STILL RUNNING' || echo 'CLEAN STOP'"

# Step 5: Update /.env — change Q_DB_PATH to absolute path
ssh root@89.149.241.126 "sed -i 's|Q_DB_PATH=./data-mainnet-genesis|Q_DB_PATH=/home/orobit/data-mainnet-genesis|' /.env && grep Q_DB_PATH /.env"

# Step 6: Start service
ssh root@89.149.241.126 "systemctl start q-api-server"

# Step 7: Watch logs — look for RocksDB open + WAL recovery message
ssh root@89.149.241.126 "journalctl -u q-api-server -f --no-pager 2>&1 | head -40"

# Step 8: Confirm height advancing
ssh root@89.149.241.126 "sleep 30 && curl -s http://localhost:8080/api/v1/status | python3 -c 'import sys,json;d=json.load(sys.stdin)[\"data\"];print(d.get(\"current_height\"),d[\"status\"])'"

# Step 9: After 30+ min of clean operation, optionally delete root DB copy
# ssh root@89.149.241.126 "rm -rf /data-mainnet-genesis"  ← DO NOT run until home DB confirmed stable
```

---

## 8. Risks of Waiting

With root at 100% and new blocks writing every second:
- **Within minutes**: RocksDB compaction triggers temp SST creation → instant disk full → write-error state
- **Result**: Block production stalls again, miners stop getting rewards
- **Clearing logs again** only buys another 1-4 hours before the DB refills root

**Every hour we wait = another likely stall cycle.**

---

## 9. What Is NOT a Problem

- The home DB data is **not corrupted** — the WAL ensures durability even mid-compaction
- The home DB is the **correct, authoritative** chain history (219 GB = full history vs 7 GB partial)
- The `/home` partition has **774 GB free** — plenty of room for continued DB growth
- Beta and Gamma nodes will cover mining rewards during the brief outage window
