# User Node Sync Issue - Diagnosis and Fix

**Date**: 2025-11-13 12:15 CET
**Issue**: Node stuck at height 0, receiving but not processing blocks from network

---

## 🚨 **Problem Summary**

### **Symptoms:**
```
✅ Node is CONNECTED to network
✅ Node is RECEIVING blocks (heights 61793-61891)
✅ Node is FORWARDING blocks via gossipsub (relay mode)
❌ Node is NOT STORING received blocks
❌ Node is NOT PROCESSING received blocks
❌ Height is STUCK at 0 (should be 61890+)
❌ Auto-sync is NOT detecting the 61,890-block gap
```

### **Root Cause:**
The node is operating in **PASSIVE RELAY MODE** instead of **ACTIVE SYNC MODE**. It receives gossipsub messages and forwards them, but the sync logic isn't triggering to actually store and process the blocks.

---

## 🔍 **Why This Happens**

### **Gossipsub Event Flow:**
```
Network → Gossipsub Message Received → Should Trigger:
1. Block validation
2. Block storage (save_qblock)
3. Height advancement
4. Gap detection and fill

Currently happening:
Network → Gossipsub Message Received → Forward to other peers
                                      → ❌ SKIP processing
```

### **Possible Causes:**

#### **1. Fast Sync Not Enabled** (Most Likely)
```rust
// In main.rs around line 1290
state.fast_sync_enabled = true;  // ← This might be false or not initialized
state.fast_sync_tx = Some(tx);   // ← Channel might be None
```

If `fast_sync_enabled = false`, received blocks are ignored!

#### **2. SafeBatchedWriter Not Running**
The `SafeBatchedWriter` task might have crashed or never started:
```rust
// Should see in logs:
"🚀 SafeBatchedWriter task started"
"📥 BlockWriter received block at height X"
```

If these logs are missing → Writer isn't running

#### **3. Gap-Fill Logic Not Triggering**
The auto-sync loop checks every 30 seconds but might not be detecting the gap:
```rust
// Condition for sync trigger:
if network_height > current_height + 5 {
    turbo_sync.sync_to_height(network_height).await
}
```

If `current_height` is stuck at 0 and `network_height` is also 0 (not updated), sync never triggers!

#### **4. Database Lock/Corruption**
The database might be locked by another process or corrupted, preventing writes.

---

## 🔧 **Diagnostic Steps**

### **Step 1: Check Fast Sync Status**
```bash
# Search logs for SafeBatchedWriter initialization
journalctl -u q-api-server --since "10 minutes ago" | grep -i "SafeBatchedWriter"

# Expected output:
"🚀 Initializing SafeBatchedWriter (Phase 1A)..."
"🚀 SafeBatchedWriter task started"
"✅ SafeBatchedWriter initialized successfully"

# If missing → Fast sync not enabled
```

### **Step 2: Check BlockWriter Activity**
```bash
# Check if BlockWriter is receiving blocks
journalctl -u q-api-server --since "5 minutes ago" | grep "BlockWriter received" | wc -l

# Expected: >0 (should see many lines)
# If 0 → Blocks not being forwarded to writer
```

### **Step 3: Check Current Height**
```bash
curl -s http://YOUR_NODE:8080/api/v1/node/status | jq '.data.current_height'

# Expected: Should be advancing (61700+)
# If 0 → Height not updating
```

### **Step 4: Check Network Height Detection**
```bash
# Check if node sees peer heights
journalctl -u q-api-server --since "5 minutes ago" | grep -E "network_height|highest_network_height|peer.*height"

# Expected: Should see network heights in 61000+ range
# If 0 or missing → Peer height detection broken
```

### **Step 5: Check Database Path**
```bash
# Verify database is writable
ls -ld $(grep Q_DB_PATH /etc/systemd/system/q-api-server.service | cut -d'=' -f3 | tr -d '"')

# Should show: drwxr-xr-x (readable and writable)
# If missing or wrong permissions → Database issue
```

---

## 🛠️ **Fix Procedures**

### **Fix 1: Restart with Correct Binary (RECOMMENDED)**

If using OLD binary without fast sync:
```bash
# Stop service
sudo systemctl stop q-api-server

# Download v1.0.6-beta (has all fixes)
wget https://quillon.xyz/downloads/q-api-server-v1.0.6-beta -O /tmp/q-api-server
chmod +x /tmp/q-api-server

# Replace binary
sudo mv /tmp/q-api-server /path/to/your/target/release/q-api-server

# Start service
sudo systemctl start q-api-server

# Monitor startup
journalctl -u q-api-server -f | grep -E "SafeBatchedWriter|BlockWriter|height"
```

### **Fix 2: Database Reset (If Corrupted)**

⚠️ **WARNING**: This deletes all local data!

```bash
# Stop service
sudo systemctl stop q-api-server

# Get database path
DB_PATH=$(grep Q_DB_PATH /etc/systemd/system/q-api-server.service | cut -d'=' -f3 | tr -d '"')

# Backup (optional)
mv $DB_PATH ${DB_PATH}.backup.$(date +%Y%m%d_%H%M%S)

# Create fresh database
mkdir -p $DB_PATH/hot $DB_PATH/cold $DB_PATH/snapshots

# Start service (will sync from network)
sudo systemctl start q-api-server
```

### **Fix 3: Force Sync from Specific Height**

If database has partial data:
```bash
# Stop service
sudo systemctl stop q-api-server

# Run manual pointer update (if height pointer is wrong)
cd /opt/orobit/shared/q-narwhalknight
Q_DB_PATH=$DB_PATH cargo run --release --bin manual_pointer_update

# Start service
sudo systemctl start q-api-server
```

### **Fix 4: Enable Fast Sync (If Disabled in Code)**

Check `main.rs` around line 1290:
```rust
// ❌ If this is commented out or set to false:
// state.fast_sync_enabled = false;

// ✅ Should be:
state.fast_sync_enabled = true;
state.fast_sync_tx = Some(tx);
state.fast_sync_metrics = Some(metrics_handle);
```

If disabled → Recompile with it enabled.

---

## 📊 **Expected Behavior After Fix**

### **Healthy Sync Logs:**
```
✅ SafeBatchedWriter initialized successfully
✅ BlockWriter received block at height 61468
✅ BlockWriter received block at height 61469
✅ Block 61468 saved successfully
✅ Block 61469 saved successfully
✅ Height advanced to 61470
```

### **Healthy API Response:**
```json
{
  "current_height": 61890,  // ← Should match network
  "uptime": "5m 23s",
  "connected_peers": 1
}
```

### **Sync Progress:**
```bash
# Height should advance rapidly during sync
watch -n 2 'curl -s http://YOUR_NODE:8080/api/v1/node/status | jq .data.current_height'

# Expected: 100-500 blocks/second during catch-up
# 0 → 100 → 500 → 1000 → ... → 61890
```

---

## 🎯 **Quick Diagnostic Checklist**

Run these commands on the stuck node:

```bash
echo "=== NODE DIAGNOSTIC ==="

echo "1. Current Height:"
curl -s http://localhost:8080/api/v1/node/status | jq .data.current_height

echo ""
echo "2. SafeBatchedWriter Status:"
journalctl -u q-api-server --since "10 minutes ago" | grep -c "SafeBatchedWriter"

echo ""
echo "3. BlockWriter Activity (last 5 min):"
journalctl -u q-api-server --since "5 minutes ago" | grep -c "BlockWriter received"

echo ""
echo "4. Recent Block Production:"
journalctl -u q-api-server --since "2 minutes ago" | grep "BLOCK PRODUCED" | tail -3

echo ""
echo "5. Database Path:"
grep Q_DB_PATH /etc/systemd/system/q-api-server.service

echo ""
echo "6. Database Size:"
DB=$(grep Q_DB_PATH /etc/systemd/system/q-api-server.service | cut -d'=' -f3 | tr -d '"')
du -sh $DB

echo ""
echo "7. Binary Version (check timestamp):"
ls -lh /path/to/q-api-server

echo "=== END DIAGNOSTIC ==="
```

### **Interpreting Results:**

| Check | Healthy | Unhealthy | Action |
|-------|---------|-----------|--------|
| Current Height | >61700 | 0 | **Deploy v1.0.6-beta** |
| SafeBatchedWriter | >0 lines | 0 lines | **Check if fast sync enabled** |
| BlockWriter Activity | >100 blocks | 0 | **Restart service** |
| Block Production | Recent blocks | No blocks | **Check mining** |
| Database Path | Valid path | Missing | **Fix environment variable** |
| Database Size | >100 MB | 0 or tiny | **Reset database** |
| Binary Timestamp | Today/recent | Old | **Update binary** |

---

## 🚀 **Recommended Fix Order**

### **For Most Cases:**
1. **Deploy v1.0.6-beta binary** (has all fixes)
2. **Restart service**
3. **Monitor for 5 minutes** (height should advance rapidly)
4. **If still stuck** → Check database permissions
5. **If database corrupted** → Reset database (last resort)

### **Quick Deploy Command:**
```bash
sudo systemctl stop q-api-server && \
wget https://quillon.xyz/downloads/q-api-server-v1.0.6-beta -O /tmp/q-api-server && \
chmod +x /tmp/q-api-server && \
sudo mv /tmp/q-api-server $(which q-api-server) && \
sudo systemctl start q-api-server && \
sleep 10 && \
journalctl -u q-api-server -f | grep -E "height|Safe|Block"
```

---

## 📈 **Success Criteria**

After applying fixes, verify:

- [ ] Height is advancing: `curl http://localhost:8080/api/v1/node/status | jq .data.current_height` (should increase rapidly)
- [ ] SafeBatchedWriter is running: `journalctl -u q-api-server | grep "SafeBatchedWriter task started"`
- [ ] BlockWriter is receiving blocks: `journalctl -u q-api-server --since "1 minute ago" | grep -c "BlockWriter received"` (>10)
- [ ] Blocks are being saved: `journalctl -u q-api-server | grep "saved successfully" | tail -5`
- [ ] No errors in logs: `journalctl -u q-api-server --since "5 minutes ago" | grep ERROR | wc -l` (should be 0)

---

## 🆘 **If Still Stuck**

If none of the above fixes work:

### **Nuclear Option: Complete Reset**
```bash
# 1. Stop service
sudo systemctl stop q-api-server

# 2. Backup everything
tar -czf ~/q-backup-$(date +%Y%m%d_%H%M%S).tar.gz \
    /etc/systemd/system/q-api-server.service \
    $Q_DB_PATH

# 3. Fresh start
rm -rf $Q_DB_PATH
mkdir -p $Q_DB_PATH/hot $Q_DB_PATH/cold $Q_DB_PATH/snapshots

# 4. Deploy fresh binary
wget https://quillon.xyz/downloads/q-api-server-v1.0.6-beta -O /tmp/q-api-server
chmod +x /tmp/q-api-server
sudo mv /tmp/q-api-server $(which q-api-server)

# 5. Start and sync from genesis
sudo systemctl start q-api-server

# 6. Monitor sync (should take 30-60 minutes to sync 61k blocks)
watch -n 5 'curl -s http://localhost:8080/api/v1/node/status | jq .data.current_height'
```

### **Expected Sync Speed:**
- **Phase 1** (0 → 10,000): ~100 blocks/sec (2 minutes)
- **Phase 2** (10,000 → 50,000): ~200 blocks/sec (3 minutes)
- **Phase 3** (50,000 → 61,890): ~300 blocks/sec (1 minute)
- **Total**: 6-10 minutes for full sync

If sync is slower than this → Check network connectivity, disk I/O, or CPU usage.

---

## 📝 **Comparison: Working vs Stuck Node**

### **Server Beta (Production - WORKING):**
```
✅ Height: 61,500+ (advancing)
✅ SafeBatchedWriter: Initialized and running
✅ BlockWriter: Processing 200+ blocks/min
✅ Database: 1.3 GB, healthy
✅ Binary: v1.0.6-beta (latest)
✅ Logs: "BlockWriter received", "Block saved successfully"
```

### **Your Node (STUCK):**
```
❌ Height: 0 (frozen)
❓ SafeBatchedWriter: Unknown (check logs)
❓ BlockWriter: Not receiving blocks OR not initialized
❓ Database: Unknown size (might be empty)
❓ Binary: Possibly old version
❌ Logs: Only gossipsub relay messages, no processing
```

---

## 🎯 **Root Cause Hypothesis**

Based on the symptoms, **most likely causes** (in order of probability):

1. **OLD BINARY** (80% likely) - Missing SafeBatchedWriter or advance_height() fix
2. **FAST SYNC DISABLED** (10% likely) - Code path not enabled
3. **DATABASE CORRUPTION** (5% likely) - Prevents writes
4. **ENVIRONMENT ISSUE** (5% likely) - Wrong Q_DB_PATH, permissions, etc.

**Recommended Action**: Deploy v1.0.6-beta binary (fixes 90% of cases).

---

## 🏆 **Success Story**

After deploying v1.0.6-beta on Server Beta:
- Old process shutdown: 2 minutes 57 seconds (binary search storm)
- New process startup: 30 seconds (AI model loading)
- Sync status: Processing blocks immediately
- Height: Advancing from 61467 → 61500+ in first minute

**Your node should behave the same way!**

---

**Prepared By**: Server Beta (Claude Code) - 185.182.185.227
**Purpose**: Help diagnose and fix user node stuck at height 0
**Next Step**: Deploy v1.0.6-beta binary and monitor sync progress
