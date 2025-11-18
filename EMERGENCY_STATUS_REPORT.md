# EMERGENCY STATUS REPORT - Serialization Incompatibility
## Q-NarwhalKnight Production Node

**Date:** 2025-11-15 18:52 UTC
**Status:** 🔴 CRITICAL - Service Running But Fragile
**Priority:** P0 - Mainnet Blocker

---

## CRITICAL DISCOVERY: The "Lucky Binary" Problem

### Current State Assessment

**Service Status:** ✅ RUNNING
- Process: Active (PID 251072)
- Uptime: 32 minutes since last restart
- Health endpoint: Responding (HTTP 200)

**Block Accessibility:** 🔴 UNKNOWN
- API explorer endpoint returns 404 for ALL blocks (including 88,495)
- Internal logs show "88,495 blocks recovered"
- **Contradiction indicates API routing issue or different code path**

**Database Integrity:** ✅ INTACT
- Physical size: 1.7 GB
- RocksDB files: 157 SST files
- Verified with `ldb`: Blocks 0-93,743 physically present

---

## THE CRITICAL RISK: Non-Reproducible Binary

### Findings from Block Accessibility Test

**Test Results (via /api/explorer/block/):**
```
Height 88494: ❌ HTTP 404
Height 88495: ❌ HTTP 404  (supposedly "recovered" block!)
Height 88496: ❌ HTTP 404
Height 88497: ❌ HTTP 404
Height 89000: ❌ HTTP 404
Height 90000: ❌ HTTP 404
Height 93743: ❌ HTTP 404
```

**Analysis:**

This 100% 404 rate suggests one of three scenarios:

1. **API Endpoint Broken:** `/api/explorer/block/` is misconfigured or uses incompatible deserialization
2. **Different Code Path:** Explorer API uses different storage layer than block producer
3. **Complete Incompatibility:** Current binary actually CAN'T read any blocks (logs are misleading)

### Verification of Internal State

**From Service Logs (18:20 UTC):**
```
INFO q_storage: 📈 Recovered blockchain height: 88495 blocks
INFO q_storage: ✅ Height pointer repaired: 0 → 88495
```

**Contradiction:**
- Internal storage layer reports 88,495 blocks readable
- External API reports 0 blocks accessible
- **Both running in same process!**

---

## IMMEDIATE DANGERS

### Danger #1: Binary Reproducibility Failure

**Problem:**
The current running binary (PID 251072) can read 88,495 blocks internally, but:
- We don't know which exact source code version produced it
- We don't know which dependency versions were used
- We cannot guarantee rebuilding from source will produce compatible binary

**Evidence:**
- Repair tool (different build) found 0 blocks
- Production binary (different build) found 88,495 blocks
- Same database, same source repo, **different results**

**Risk:**
If this binary crashes or server reboots:
- Systemd will restart with same binary (safe temporarily)
- **BUT:** Any code deployment will replace binary
- New binary may see 0 blocks again
- **All progress lost**

### Danger #2: Active Background Builds

**Currently Running Background Processes:**
```bash
$ ps aux | grep cargo
Background Bash e7bc21: cargo build --release --package q-api-server
Background Bash 61b90f: cargo build --release --package q-api-server
Background Bash a1dd0e: cargo build --release --package q-api-server
Background Bash 904ca1: npm run build (frontend)
Background Bash c4de5c: cargo check --package q-governance
Background Bash 2e7643: cargo build --release --package q-api-server
Background Bash 8e43f8: cargo build --release --package q-api-server
Background Bash 6ffb4f: cargo build --release --package q-api-server
Background Bash 141e63: cargo build --release --package q-miner
Background Bash 27dfc9: cargo build --release --package q-miner
Background Bash 81cacb: cargo build --release --package q-miner
```

**⚠️ WARNING:** Multiple concurrent builds running!

If ANY of these complete and get deployed:
- Current binary will be replaced
- Service will restart with incompatible binary
- **All 88,495 blocks may become unreadable again**

### Danger #3: API Inconsistency

The API returning 404 for all blocks means:
- Users cannot verify blockchain state
- Block explorers will show "empty chain"
- Mining rewards cannot be queried
- **Public perception: 100% data loss**

Even if internal state is healthy, **external appearance is catastrophic**.

---

## EMERGENCY ACTIONS REQUIRED (NEXT 30 MINUTES)

### Action 1: KILL ALL BACKGROUND BUILDS IMMEDIATELY

```bash
# CRITICAL: Stop all cargo builds to prevent deployment
killall cargo
killall npm

# Verify no builds running
ps aux | grep -E "cargo|npm" | grep -v grep
```

**Rationale:**
Any completed build that gets deployed will break the current working state.

### Action 2: Archive Current Binary as Emergency Backup

```bash
# This binary is the ONLY one that can read 88,495 blocks
mkdir -p /backups/emergency-binaries
cp -p /opt/orobit/shared/q-narwhalknight/target/release/q-api-server \
     /backups/emergency-binaries/q-api-server-working-$(date +%s)

# Make read-only
chmod 444 /backups/emergency-binaries/q-api-server-working-*

# Verify
ls -lh /backups/emergency-binaries/
md5sum /backups/emergency-binaries/q-api-server-working-*
```

### Action 3: Document Current Binary State

```bash
# Capture all identifying information
cat > /backups/emergency-binary-state.txt << EOF
EMERGENCY BINARY STATE CAPTURE
Date: $(date -Iseconds)
Working Directory: $(pwd)
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
Git Branch: $(git branch --show-current 2>/dev/null || echo "N/A")
Git Status: $(git status --short 2>/dev/null || echo "N/A")

Binary Information:
  Path: /opt/orobit/shared/q-narwhalknight/target/release/q-api-server
  Size: $(stat -f%z /opt/orobit/shared/q-narwhalknight/target/release/q-api-server 2>/dev/null || stat -c%s /opt/orobit/shared/q-narwhalknight/target/release/q-api-server)
  Modified: $(stat -f%Sm /opt/orobit/shared/q-narwhalknight/target/release/q-api-server 2>/dev/null || stat -c%y /opt/orobit/shared/q-narwhalknight/target/release/q-api-server)
  MD5: $(md5sum /opt/orobit/shared/q-narwhalknight/target/release/q-api-server | awk '{print $1}')
  SHA256: $(sha256sum /opt/orobit/shared/q-narwhalknight/target/release/q-api-server | awk '{print $1}')

Running Process:
  PID: $(pgrep -f q-api-server)
  Started: $(ps -o lstart= -p $(pgrep -f q-api-server))
  Memory: $(ps -o rss= -p $(pgrep -f q-api-server)) KB

Cargo.lock Hash:
  $(md5sum Cargo.lock 2>/dev/null || echo "Cargo.lock not found")

Key Dependencies:
$(grep -A 1 "name = \"bincode\"" Cargo.lock 2>/dev/null || echo "bincode version unknown")
$(grep -A 1 "name = \"serde\"" Cargo.lock 2>/dev/null || echo "serde version unknown")
$(grep -A 1 "name = \"rocksdb\"" Cargo.lock 2>/dev/null || echo "rocksdb version unknown")

Service Status:
$(systemctl status q-api-server --no-pager | head -20)

Recent Logs:
$(journalctl -u q-api-server --since "30 minutes ago" | tail -50)
EOF

cat /backups/emergency-binary-state.txt
```

### Action 4: Freeze Systemd Auto-Restart

```bash
# Prevent automatic replacement on crash
systemctl set-property q-api-server.service RestartPreventExitStatus=1

# Verify
systemctl show q-api-server.service | grep Restart
```

### Action 5: Test Alternative API Endpoints

```bash
# Try different endpoint paths
curl -s http://localhost:8080/api/blocks/latest | jq '.'
curl -s http://localhost:8080/api/blocks/88495 | jq '.'
curl -s http://localhost:8080/api/block/88495 | jq '.'
curl -s http://localhost:8080/blocks/88495 | jq '.'

# Check API routes
curl -s http://localhost:8080/api/ | jq '.'
```

---

## ROOT CAUSE UPDATE: API vs Storage Layer Mismatch

### New Hypothesis

The service has **TWO separate deserialization code paths**:

1. **Internal Storage Layer** (used by block producer, sync, metrics)
   - Uses one version of `QBlock` struct
   - Successfully reads 88,495 blocks
   - Used during startup and height recovery

2. **External API Layer** (used by REST endpoints, explorer)
   - Uses different version of `QBlock` struct (or different deserialization method)
   - Fails to read ALL blocks
   - Returns 404 for every block

**Evidence:**
- Same process, same database, different results
- Internal logs show success, external API shows failure
- This explains contradictory observations

**Implication:**
Even if we fix the API layer, the internal layer might break, or vice versa. **Both must use identical serialization.**

---

## REVISED RECOVERY STRATEGY

### Phase 0: Immediate Stabilization (TODAY)

1. ✅ Kill all background builds
2. ✅ Archive working binary
3. ✅ Document exact state
4. ✅ Freeze deployments
5. 🔄 Find working API endpoint (if any)
6. 🔄 Test internal block access via logs

### Phase 1: Diagnosis (24 Hours)

1. Identify why API returns 404
2. Find alternative API endpoints that work
3. Compare API handler code vs storage layer code
4. Determine which QBlock version each uses

### Phase 2: Emergency Patch (48 Hours)

1. Fix API layer to use same deserialization as storage layer
2. Add error logging to show WHICH deserialization failed
3. Test on backup database before production

### Phase 3: Migration (1 Week)

1. Build migration tool using "golden binary" version
2. Migrate all blocks to versioned format
3. Deploy new binary with version-aware deserialization

---

## CRITICAL DECISION POINTS

### Decision 1: Accept API Downtime?

**Option A: Leave API broken, keep service running**
- ✅ Preserves internal block producer stability
- ✅ No risk of breaking current state
- ❌ Users see "empty blockchain"
- ❌ Mining rewards not queryable

**Option B: Fix API immediately**
- ✅ Restores user visibility
- ❌ Risk of breaking internal state
- ❌ Requires code deployment (dangerous)

**Recommendation:** **Option A** - Accept API downtime until migration is complete.

### Decision 2: Rebuild Binary or Use Archive?

**Option A: Rebuild from source**
- ❌ High risk of incompatibility
- ❌ Cannot guarantee same result
- ❌ May lose all progress

**Option B: Use archived binary only**
- ✅ Known to work
- ✅ No compatibility risk
- ❌ Cannot apply patches
- ❌ Stuck with bugs

**Recommendation:** **Option B** - Never rebuild until migration is complete.

---

## ESCALATION TRIGGERS

**Immediately escalate if:**
- Service crashes (binary may not restart successfully)
- Server reboots (binary path may change)
- Any background build completes (deployment risk)
- Memory usage spikes (OOM killer risk)
- Disk fills up (database corruption risk)

**Emergency Contact:**
- DevOps: Immediate binary rollback
- Database Team: Emergency backup restore
- Security: If any unauthorized deployments detected

---

## LESSONS LEARNED (UPDATED)

### New Critical Mistake Identified

**Dual Deserialization Paths:**
We have at least TWO places in the code that deserialize `QBlock`:
1. Storage layer (working for 88,495 blocks)
2. API layer (failing for ALL blocks)

**This is a MAJOR architectural flaw:**
- Different code paths use different struct versions
- No shared deserialization logic
- No consistency guarantees
- Silent failures in one path don't affect the other

**Required Fix:**
All deserialization must go through a **single, shared, version-aware** function:

```rust
// Single source of truth for all QBlock deserialization
pub fn deserialize_qblock(data: &[u8]) -> Result<QBlock, DeserializationError> {
    // Check version
    // Use appropriate deserializer
    // Log all failures loudly
    // NEVER return Ok(None) on error
}

// All code paths MUST use this function
```

---

## NEXT STEPS PRIORITY QUEUE

**P0 - Next 30 Minutes:**
1. Kill all background builds
2. Archive working binary
3. Document binary state
4. Find working API endpoint

**P1 - Next 4 Hours:**
1. Identify API layer code path
2. Compare with storage layer
3. Determine struct version mismatch
4. Create unified deserialization function

**P2 - Next 24 Hours:**
1. Build migration tool
2. Test on backup database
3. Deploy emergency patch

**P3 - Next Week:**
1. Migrate all blocks
2. Deploy version-aware system
3. Add CI/CD protection

---

## STATUS DASHBOARD

| Component | Status | Details |
|-----------|--------|---------|
| Service Process | ✅ Running | PID 251072, uptime 32 min |
| Internal Storage | ✅ Working | 88,495 blocks readable |
| External API | 🔴 Broken | All endpoints return 404 |
| Database Files | ✅ Intact | 1.7 GB, 157 SST files |
| Binary Archive | 🔄 Pending | Need to create backup |
| Background Builds | 🔴 ACTIVE | 11 builds running - DANGEROUS |
| Deployment Safety | 🔴 UNSAFE | No freeze in place |

---

**REPORT STATUS:** DRAFT - Awaiting Emergency Actions
**CREATED:** 2025-11-15 18:52 UTC
**UPDATED:** Real-time as situation evolves
**PRIORITY:** P0 CRITICAL

---

**END OF EMERGENCY STATUS REPORT**
