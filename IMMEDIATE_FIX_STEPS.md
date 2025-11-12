# IMMEDIATE FIX - Stop Data Loss NOW

## Problem
Node loses all blocks on every restart due to SIGKILL before RocksDB flush completes.

## Fix in 3 Steps (5 minutes)

### Step 1: Fix Systemd Service (1 min)
```bash
# Edit service file
nano /etc/systemd/system/q-api-server.service

# Find [Service] section and add/update:
[Service]
TimeoutStopSec=300
KillMode=mixed
SendSIGKILL=yes

# Save and exit (Ctrl+X, Y, Enter)

# Reload systemd
systemctl daemon-reload
```

### Step 2: Repair Database (1 min)
```bash
# Make sure service is stopped
systemctl stop q-api-server

# Run repair tool (from q-narwhalknight directory)
cd /opt/orobit/shared/q-narwhalknight
./target/release/repair-database ./data-mine6/hot

# When prompted:
# Choose option 1: Fix qblock:latest pointer to 0
# (This resets pointer to match reality - 0 blocks)
```

### Step 3: Start with v0.9.76 P2P Gap Fill (3 min)
```bash
# Start service
systemctl start q-api-server

# Monitor P2P gap fill (blocks will sync from peers)
journalctl -u q-api-server -f | grep --line-buffered -E "GAP FILL|Height:|SYNC"
```

Expected output:
```
🚨 CRITICAL GAP DETECTED IN BLOCKCHAIN!
   Missing block at height: 1
📡 [GAP FILL] Found 5 peers with height >= 1
✅ [GAP FILL] Gap fill request sent
⏳ [GAP FILL] Waiting 15s for responses...
✅ [GAP FILL] SUCCESS! Gap filled completely!
```

## What This Fixes

✅ **Systemd timeout increased**: 90s → 300s (5 minutes)
✅ **Graceful shutdown**: Node has time to flush RocksDB
✅ **Database repaired**: Pointer now matches reality
✅ **P2P recovery**: Automatically syncs blocks from peers

## Verify Success

After 30 minutes, check:
```bash
# Should show increasing height (not stuck at 0)
journalctl -u q-api-server --since "30 minutes ago" | grep "Height:" | tail -10

# Should show blocks in database
./target/release/repair-database ./data-mine6/hot
# Should show: "Total blocks found: 1000+" (not 0)
```

## If It Still Fails

Try manual database reset and full resync:
```bash
systemctl stop q-api-server
rm -rf ./data-mine6/hot/*
systemctl start q-api-server
# Will sync from genesis via P2P gap fill
```

---
**DO THESE 3 STEPS NOW** - Each restart without this fix = more data loss!
