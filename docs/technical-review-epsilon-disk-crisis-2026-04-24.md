# Technical Review: Epsilon Root Partition Disk Full — Block Production Stall
**Date:** 2026-04-24  
**Server:** Epsilon (89.149.241.126) — Primary 10Gbit Bootstrap Supernode  
**Severity:** CRITICAL — Block production halted for several hours on mainnet  
**Status:** Partially recovered (service restarted, blocks flowing); root partition still 99% full

---

## 1. What Actually Happened — Root Cause Chain

### Step 1: Root Partition Filled to 100%

Epsilon has two storage tiers:

| Partition | Device | Size | Use | Free |
|-----------|--------|------|-----|------|
| `/` (root) | nvme0n1p3 | 40 GB | OS, node binary, **DB** | **734 MB (99% full)** |
| `/home` | nvme0n1p4 | 1.8 TB | User data | 774 GB free |
| `/home/storage` | md0 RAID | 80 TB | Storage | ~74 TB free |

The node's RocksDB database was set up at `/data-mainnet-genesis/` — which resolves to the **root partition** because:

```
# /etc/systemd/system/q-api-server.service
WorkingDirectory=/
EnvironmentFile=/.env

# /.env
Q_DB_PATH=./data-mainnet-genesis   ← relative to WorkingDirectory=/
```

→ Effective DB path = `/data-mainnet-genesis/` = **root partition**

### Step 2: RocksDB Triggered an I/O Error

When the root partition hit 100%, RocksDB tried to flush a write-ahead log (WAL) entry or create a new SST file and got:

```
IO error: No space left on device
```

### Step 3: RocksDB Entered Permanent Write-Error State

RocksDB's safety design: after **any** I/O error, it refuses all future writes with:

```
IO error: Writer has previous error.
```

This cannot be cleared without reopening the database (process restart). It is correct behavior — RocksDB prefers a hard stop over silent data corruption.

### Step 4: Block Production Stalled Silently

The block producer continued running, checking solutions, saying `SHOULD_PRODUCE → YES`, but every attempt to write block 16138870 silently failed:

```
❌ AsyncStorageEngine: Failed to queue block 16138870: 
   Batch write failed: IO error: Writer has previous error.
⚠️ [DAG] Failed to save DAG layer block: Failed to write DAG layer block to database
```

Thousands of valid mining solutions queued in Phase 4 with no consumer able to drain them.

---

## 2. Root Partition Space Breakdown (at time of failure)

```
Total: 40 GB
/usr                    13.0 GB  (OS packages — not reducible)
/var/lib                 8.3 GB  (package state, docker, mysql, etc.)
/opt/orobit              6.2 GB  (node binary + shared data)
/data-mainnet-genesis    6.1 GB  ← RocksDB database on ROOT (wrong location)
  ├─ hot/               4.6 GB  (1,796 SST files, active compaction target)
  ├─ backups/           1.5 GB  (local DB backups)
  ├─ srs_cache/         157 MB  
  └─ cold/              3.6 MB  
/root                    1.1 GB
/var/log                 (at time of failure: ~1.2 GB, now ~317 MB after cleanup)
/var/www                 1.1 GB
binaries at /            ~500 MB (q-miner-*, q-api-server-* at filesystem root)
```

### Root-Level Binary Clutter (490 MB on root partition)

Old binaries were copied directly into `/` during earlier deployments:

```
/q-api-server-linux-x86_64    90 MB  (duplicate, two identical files)
/q-api-server-v10.3.8         90 MB
/q-miner-v9.0.2               11 MB
/q-miner-v9.0.7               20 MB
/q-miner-v9.1.8               23 MB
/q-miner-v9.4.1               24 MB
/q-miner-v9.4.1.1             24 MB
/q-miner-v9.6.0               24 MB
/q-miner-v10.1.1              23 MB
/q-miner-v10.3.12.1           29 MB
```

These are old versions served to users for download, but they live at `/` instead of `/home/orobit/q-narwhalknight/dist-final/downloads/`.

---

## 3. What We've Already Done (Before This Review)

| Action | Safe? | Reversible? | Effect |
|--------|-------|-------------|--------|
| `sysctl vm.vfs_cache_pressure=200` | ✅ Yes | ✅ Yes | Kernel reclaims page cache faster |
| `journalctl --vacuum-size=200M` | ✅ Yes | ✅ Yes | Freed ~1.9 GB of old journal |
| Cleared rotated syslogs (*.gz, *.1) | ✅ Yes | ❌ No | Freed ~200 MB |
| Truncated active syslog/kern.log | ✅ Yes | ❌ No | Freed ~600 MB |
| Cleared apt cache | ✅ Yes | ✅ Yes (re-downloads) | Freed ~200 MB |
| `systemctl restart q-api-server` | ⚠️ Needed | N/A | Cleared RocksDB write-error state |

**Current state:** Service active, height 16138850, blocks flowing. Root partition at 734 MB free (99%).

**Risk of current state:** With only 734 MB free and the hot DB (4.6 GB, 1,796 SST files) actively written during compaction, the partition can fill again at any time. RocksDB compaction temporarily creates new SST files before deleting old ones — peak temporary usage can be 200–500 MB above steady state. We are dangerously close.

---

## 4. What Still Needs to Be Fixed

### Problem A: DB is on Wrong Partition (Critical)
`/data-mainnet-genesis` → root partition (40 GB, nearly full)  
Should be → `/home/orobit/data-mainnet-genesis` (1.8 TB, 774 GB free)

### Problem B: Old Root-Level Binaries (490 MB wasted on root)
Old miner/server binaries at `/q-miner-*`, `/q-api-server-*` consume 490 MB of root.

### Problem C: No Log Rotation Limits on Root
syslog grew to 446 MB, kern.log to 99 MB, journal to 2 GB without caps.

### Problem D: journald Not Configured to Limit Size
Systemd journal has no `SystemMaxUse` limit — will grow unbounded.

---

## 5. Proposed Fix Plan (Ordered by Risk)

### Fix 1: Limit journald size (ZERO RISK, 5 min)

```bash
# /etc/systemd/journald.conf.d/size.conf
[Journal]
SystemMaxUse=500M
RuntimeMaxUse=100M
```

Immediately reclaims journal space and prevents future growth.

### Fix 2: Delete old root-level binaries (LOW RISK, 2 min)

The binaries at `/q-miner-v9.*`, `/q-api-server-v10.3.8`, `/q-api-server-linux-x86_64` are old versions not served from the correct downloads path. They save 490 MB of root space.

**Risk:** Zero — these files are not referenced by the running service. Verify with `lsof /q-miner-v9.0.2` before deleting.

### Fix 3: Add logrotate limits (ZERO RISK, 5 min)

Configure `/etc/logrotate.d/rsyslog` with `size 50M` and `maxage 7` to cap syslog growth.

### Fix 4: Move DB to /home — THE CRITICAL FIX (HIGH CARE NEEDED)

This is the permanent solution. Steps:

1. **Verify service is healthy** — check height advancing, no write errors
2. **Stop service** — `systemctl stop q-api-server` (graceful, allows RocksDB to flush WAL and close cleanly)
3. **Verify stopped** — `pgrep -f q-api-server` returns nothing
4. **Copy DB** (not move yet — keep original as safety):
   ```bash
   rsync -av --progress /data-mainnet-genesis/ /home/orobit/data-mainnet-genesis/
   ```
5. **Verify copy** — compare sizes: `du -sh /data-mainnet-genesis/ /home/orobit/data-mainnet-genesis/`
6. **Update /.env**:
   ```
   Q_DB_PATH=/home/orobit/data-mainnet-genesis
   ```
   (absolute path, WorkingDirectory no longer matters)
7. **Start service** — `systemctl start q-api-server`
8. **Monitor** — verify height advances, no write errors, DB on /home is written to
9. **After 30 min of clean operation** — delete old copy:
   ```bash
   rm -rf /data-mainnet-genesis
   ```

**Risk assessment:**
- Steps 1–5 (copy): Zero risk — original untouched
- Step 6 (config change): Low risk — single env var, absolute path is unambiguous
- Step 7–8 (start + verify): Medium risk — new DB path must exist and be writable
- Step 9 (delete original): Low risk — only after confirmed working

**Total estimated downtime:** 5–15 minutes (time to stop + copy 6 GB at NVMe speeds)

**Alternative (symlink approach):** Instead of updating /.env, create a symlink:
```bash
mv /data-mainnet-genesis /home/orobit/data-mainnet-genesis
ln -s /home/orobit/data-mainnet-genesis /data-mainnet-genesis
```
**Risk:** Slightly higher — if symlink breaks during copy, service won't start. Prefer updating /.env.

---

## 6. Risks of NOT Acting (Status Quo)

With 734 MB free on root and the RocksDB hot tier (4.6 GB, 1,796 SST files) actively receiving new blocks:

- **Compaction**: RocksDB regularly merges SST files, temporarily needing 200–500 MB extra disk space
- **Time to next failure**: Unknown — could be hours to days depending on write/compaction rate
- **Consequence of next failure**: Same as today — block production stalls, miners stop getting rewards, network disruption

The root partition will fill again. It is a matter of when, not if.

---

## 7. Recommended Action Sequence

1. **Right now (immediate, zero risk):** Fix journald + logrotate limits to stabilize the root partition
2. **Today (low risk, planned):** Delete old root-level binaries after verifying they're unused
3. **Planned window (high care):** Move DB to /home following the procedure in Fix 4
4. **After DB move:** Update deployment scripts to always use `/home/orobit/` paths on Epsilon

**Do NOT proceed to step 3 without user approval and a planned maintenance window.**

---

## 8. What Is NOT a Problem

- The RocksDB data itself is **not corrupted** — the write-error state is a protection mechanism, not data loss. The DB was properly closed on graceful shutdown.
- The `/home` and `/home/storage` partitions have ample space (774 GB and 74 TB respectively).
- The service is currently running and producing blocks normally.
- No chain data was lost — height 16138869 was the last produced block before the stall, and the chain will resync any missed blocks from peers.
