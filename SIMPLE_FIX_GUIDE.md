# Simple Guide: Fix Height Reset Issue

**Problem**: Your node keeps resetting to lower block heights (3000 → 1400 → 558 → 0)

**Root Cause**: Database corruption or wrong database pointer

---

## 🎯 Quick Solution (Choose One)

### Option 1: Delete and Resync (EASIEST)

**This is the safest option - you'll get a fresh copy from Server Beta:**

```bash
# Stop your node
pkill q-api-server
# or if using systemd:
# systemctl stop q-api-server

# Delete your corrupted database
rm -rf ./data

# Download latest version
wget http://185.182.185.227/downloads/q-api-server-linux-x86_64
chmod +x q-api-server-linux-x86_64

# Start fresh
./q-api-server-linux-x86_64 --port 8080
```

**Result**: Your node will sync from Server Beta (currently at ~600+ blocks) and stay synced.

---

### Option 2: Diagnose First (If You Want to Understand)

**This checks if your blocks are still there:**

```bash
# Stop your node
pkill q-api-server

# Download diagnostic tool
wget http://185.182.185.227/downloads/repair-database
chmod +x repair-database

# Run diagnostic (replace './data/hot' with your actual path)
./repair-database ./data/hot
```

**What it will show:**

**Scenario A - Blocks are still there:**
```
Total blocks found: 3000
Highest block: 3000
Current pointer: 558
⚠️  Pointer is WRONG! Should be 3000
```
**Action**: Type `1` to fix the pointer, then restart your node.

**Scenario B - Blocks were deleted:**
```
Total blocks found: 558
Highest block: 558
Current pointer: 558
✅ Pointer is correct
```
**Action**: Your blocks were actually deleted. Use Option 1 to resync.

---

## 🤔 Why Is This Happening?

The issue is **database-level corruption**. Here's what's going on:

1. **During Runtime**: v0.9.0-beta-emergency prevents height from going backwards
2. **On Restart**: Node loads whatever is in the database files
3. **The Problem**: Something is corrupting your database OR the database pointer

**Possible causes:**
- Unclean shutdown (CTRL+C, crash, power loss)
- RocksDB compaction issues
- Disk errors
- Running multiple nodes with same database

---

## 🛡️ Prevent Future Issues

### 1. Always stop cleanly
```bash
# Good (graceful shutdown):
pkill -SIGTERM q-api-server
systemctl stop q-api-server

# Bad (data corruption risk):
pkill -9 q-api-server
CTRL+C repeatedly
```

### 2. Don't reuse databases
Each node needs its own database directory:
```bash
# Node 1
Q_DB_PATH=./data-node1 ./q-api-server --port 8080

# Node 2
Q_DB_PATH=./data-node2 ./q-api-server --port 8081
```

### 3. Make backups
```bash
# Before upgrading or testing
tar -czf backup-$(date +%s).tar.gz ./data
```

---

## 📊 What Server Beta Shows

Current Server Beta status:
- **Height**: ~600+ blocks and growing
- **Database**: 2.1GB, healthy
- **Version**: v0.9.0-beta-emergency
- **Uptime**: Stable

When you sync from Server Beta, you'll get this same state.

---

## ❓ Still Having Issues?

**Tell me:**
1. Which option did you choose?
2. What was the output?
3. Is your node still resetting?

**Check these:**
```bash
# See current height
curl http://localhost:8080/api/node/info | jq .height

# Watch logs
tail -f /var/log/syslog | grep q-api-server
# or
journalctl -u q-api-server -f

# Check database size
du -sh ./data
```

---

## 🚀 Expected Result

After fixing:
- ✅ Height stays stable (doesn't reset)
- ✅ Height increases as blocks are mined
- ✅ No more "sync starts over" messages
- ✅ Stays connected to network peers

---

**Bottom line**: Delete your database and resync. It's the fastest and safest solution.

Your node will sync in a few minutes and stay stable.
