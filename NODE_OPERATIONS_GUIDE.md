# Q-NarwhalKnight Node Operations Guide
## How to Continue After Emergency Recovery

**Date:** 2025-11-15 19:50 UTC
**Node Status:** ✅ STABLE - Normal Operations Resumed
**Current Height:** 88,667 blocks and growing

---

## 🎯 IMMEDIATE RECOMMENDATION: Keep Running As-Is

**DO NOTHING** - The node is healthy and stable. Let it continue running.

### Current Status:
- ✅ Service: Active (1h 30m uptime)
- ✅ Height: 88,667 blocks (+172 since recovery)
- ✅ Mining: Active (~2.4 blocks/min)
- ✅ API: Working perfectly
- ✅ Data: Zero loss, all blocks intact

---

## 📋 DAILY OPERATIONS CHECKLIST

### Morning Check (Once Daily):
```bash
# 1. Service health
systemctl status q-api-server

# 2. Current height
curl -s http://localhost:8080/metrics | grep qnk_node_height

# 3. Check logs for errors
journalctl -u q-api-server --since "24 hours ago" | grep -i error | tail -20

# 4. Disk space
df -h /opt/orobit/shared/q-narwhalknight/data-mine11

# 5. Memory usage
ps aux | grep q-api-server | grep -v grep
```

**Expected:** No errors, height increasing, disk/memory stable.

---

## 🔒 DEPLOYMENT FREEZE RULES

**⚠️  ACTIVE UNTIL FURTHER NOTICE ⚠️**

### What You CAN Do (Safe):
✅ **Restart service** (uses same binary)
```bash
systemctl restart q-api-server
```

✅ **Monitor metrics**
```bash
curl http://localhost:8080/metrics
curl http://localhost:8080/health
curl http://localhost:8080/api/v1/node/status
```

✅ **Backup database**
```bash
# Stop service first
systemctl stop q-api-server

# Backup
cp -r data-mine11 /backups/data-mine11-$(date +%Y%m%d)

# Restart
systemctl start q-api-server
```

✅ **Read logs**
```bash
journalctl -u q-api-server -f
```

### What You CANNOT Do (Dangerous):
🚫 **Rebuild from source**
```bash
# DON'T DO THIS:
cargo build --release --package q-api-server  # ❌ May break compatibility
```

🚫 **Replace binary**
```bash
# DON'T DO THIS:
cp new-binary target/release/q-api-server  # ❌ Risk losing data access
```

🚫 **Deploy code changes**
- Wait for schema versioning
- Wait for migration strategy
- Test on backup database first

---

## 🆘 WHEN TO TAKE ACTION

### Scenario 1: Service Crashes
**Symptoms:** Process dies, systemd shows "inactive"

**Action:**
```bash
# Check why it crashed
journalctl -u q-api-server --since "10 minutes ago" | tail -50

# Restart (safe - uses same binary)
systemctl restart q-api-server

# Verify recovery
curl http://localhost:8080/metrics | grep qnk_node_height
```

### Scenario 2: Height Stops Growing
**Symptoms:** Height stays same for >10 minutes

**Action:**
```bash
# Check if mining is stuck
curl http://localhost:8080/api/v1/node/status | jq '.data.consensus_status'

# Check logs
journalctl -u q-api-server --since "10 minutes ago" | grep -i "mining\|block"

# If stuck, restart
systemctl restart q-api-server
```

### Scenario 3: Disk Space Low
**Symptoms:** df shows >90% usage

**Action:**
```bash
# Check database size
du -sh data-mine11/

# Clean old backups if needed
rm -rf /backups/old-backups-*

# Restart if disk was full
systemctl restart q-api-server
```

### Scenario 4: Database Pointer Corruption (Again)
**Symptoms:** Service won't start, logs show "block does NOT exist"

**Action:**
```bash
# Stop service
systemctl stop q-api-server

# Run fixed repair tool
./target/release/repair-database ./data-mine11/hot

# Choose option 1 to fix pointer

# Restart
systemctl start q-api-server
```

---

## 📊 MONITORING COMMANDS

### Quick Health Check:
```bash
#!/bin/bash
echo "=== Q-NarwhalKnight Health Check ==="
echo ""
echo "Service:" $(systemctl is-active q-api-server)
echo "Height:" $(curl -s http://localhost:8080/metrics | grep qnk_node_height | awk '{print $2}')
echo "Mining:" $(curl -s http://localhost:8080/api/v1/node/status | jq -r '.data.consensus_status')
echo "Peers:" $(curl -s http://localhost:8080/api/v1/node/status | jq -r '.data.connected_peers')
echo "Uptime:" $(systemctl status q-api-server | grep Active | awk '{print $3,$4,$5}')
```

### Watch Live Height:
```bash
watch -n 5 'curl -s http://localhost:8080/metrics | grep qnk_node_height'
```

### Monitor Logs:
```bash
journalctl -u q-api-server -f --since "5 minutes ago"
```

---

## 🔧 LONG-TERM IMPROVEMENTS (When Ready)

### Phase 1: Schema Versioning (1-2 weeks)
**Goal:** Prevent future serialization issues

**Tasks:**
1. Add version byte to QBlock serialization
2. Implement version-aware deserialization
3. Test on backup database
4. Deploy with backward compatibility

**Status:** Planning phase

### Phase 2: Fix Silent Failures (1-2 weeks)
**Goal:** Make errors visible instead of silent

**Location:** `crates/q-storage/src/lib.rs:562-563`

**Change:**
```rust
// FROM:
Err(e) => {
    warn!("Failed: {}", e);
    Ok(None)  // ❌ Silent
}

// TO:
Err(e) => {
    error!("CRITICAL: {}", e);
    Err(e.into())  // ✅ Loud
}
```

**Status:** Code ready, deployment blocked by freeze

### Phase 3: Create Genesis Block (Optional)
**Goal:** Cosmetic - make chain start from block 0

**Priority:** P4 (nice to have, not required)

**Status:** Not started

---

## 📈 PERFORMANCE EXPECTATIONS

### Normal Operation:
- **Block Production:** ~2-3 blocks/minute
- **Memory Usage:** 4-6 GB RSS
- **Disk Growth:** ~50-100 MB/day
- **CPU Usage:** Moderate (40-60%)
- **API Response:** <500ms for most endpoints

### Warning Signs:
- ⚠️  Height stopped for >10 minutes
- ⚠️  Memory >8 GB
- ⚠️  Disk >90% full
- ⚠️  CPU at 100% sustained
- ⚠️  API timeouts

---

## 🎯 ACTUAL API ENDPOINTS (Use These!)

### Working Endpoints:
```bash
# Node status (includes height)
curl http://localhost:8080/api/v1/node/status

# Network supply (includes height)
curl http://localhost:8080/api/v1/network/supply

# Health check
curl http://localhost:8080/health

# Prometheus metrics
curl http://localhost:8080/metrics

# Bootstrap peers
curl http://localhost:8080/api/v1/status

# Mining challenge
curl http://localhost:8080/api/v1/mining/challenge

# Wallet operations
curl http://localhost:8080/api/v1/wallets
```

### Non-Existent Endpoints (Don't Use):
```bash
# These were NEVER implemented:
❌ /api/blockchain/height
❌ /api/blocks/latest
❌ /api/explorer/block/{id}
❌ /blocks/{id}
```

---

## 🔐 SECURITY REMINDERS

### Binary Protection:
- ✅ Current binary archived: `/backups/emergency-binaries/q-api-server-working-1763229134`
- ✅ MD5 checksum: `cd99234a3d7bf2c44bda1d49bd928237`
- ✅ Read-only permissions: `chmod 444`

### If Binary Gets Replaced:
```bash
# Emergency rollback (5 minutes)
systemctl stop q-api-server
cp /backups/emergency-binaries/q-api-server-working-1763229134 \
   /opt/orobit/shared/q-narwhalknight/target/release/q-api-server
chmod +x /opt/orobit/shared/q-narwhalknight/target/release/q-api-server
systemctl start q-api-server
```

---

## 📞 ESCALATION CRITERIA

### Call for Help If:
1. **Service crashes repeatedly** (>3 times in 1 hour)
2. **Height stops growing** for >30 minutes
3. **Database errors** in logs
4. **Binary gets replaced** accidentally
5. **Disk fills up** (>95%)
6. **Memory leak** (>10 GB and growing)

### Emergency Contacts:
- **DevOps:** Binary rollback
- **Database:** Emergency restore
- **Development:** Code fixes (when freeze lifts)

---

## ✅ CURRENT STATUS SUMMARY

**As of 2025-11-15 19:50 UTC:**

| Metric | Value | Status |
|--------|-------|--------|
| Service | Active | ✅ Healthy |
| Height | 88,667 | ✅ Growing |
| Uptime | 1h 30m | ✅ Stable |
| Data Loss | 0 blocks | ✅ Perfect |
| API | Working | ✅ Operational |
| Mining | Active | ✅ Producing |

**Recommendation:** Continue normal operations. Monitor daily. Wait for freeze to lift before making code changes.

---

## 🎉 KEY TAKEAWAYS

1. **Node is healthy** - All systems operational
2. **Emergency was false alarm** - Diagnostic tool bugs misled us
3. **Zero data loss** - All 88,667 blocks intact
4. **API works fine** - We tested wrong endpoints
5. **Keep it simple** - Don't change what's working

**Next review:** 2025-11-16 12:00 UTC (24h check-in)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-15 19:50 UTC
**Status:** Active Operations Guide

---
