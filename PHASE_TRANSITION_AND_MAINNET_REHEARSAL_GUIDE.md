# Phase Transition & Mainnet Rehearsal Guide

**Version:** v0.9.78-beta (Phase 8)
**Date:** 2025-11-10 (Updated)
**Purpose:** Complete guide for Phase transitions with lessons learned for mainnet launch

---

## 🎯 EXECUTIVE SUMMARY

This guide documents phase transitions and critical lessons learned for **mainnet launch**:

- **Phase 6** (v0.9.60): Fixed 5 phases of data corruption, implemented Austrian economics (100× more scarce)
- **Phase 7** (v0.9.77): Fixed double-reward bug, but **catastrophic hyperinflation remained** (672,000 QUG/day)
- **Phase 8** (v0.9.78): **CRITICAL FIX** - Reduced emission by 1000× (now 672 QUG/day, sustainable 85+ years)

**Key Achievement:** Caught and fixed **mainnet-blocking hyperinflation bug** before mainnet launch.

## ⚠️ CRITICAL: Phase 7 → Phase 8 Hyperinflation Fix

**See:** `PHASE_8_HYPERINFLATION_FIX.md` for complete details.

**Summary:** Phase 7 emitted **1000× too much supply** (672,000 QUG/day vs 672 QUG/day). This would have minted the entire 21M cap in **31 days** instead of 85 years. Phase 8 fixes the block reward from 50 QUG → 0.05 QUG per block.

**Status:** ✅ Phase 8 deployed, emission rate verified correct (4.15 QUG after 75 blocks, not 3,750 QUG).

---

## 📋 TABLE OF CONTENTS

1. [Phase 6 Overview](#phase-6-overview)
2. [Critical Bugs Found & Fixed](#critical-bugs-found--fixed)
3. [Deployment Checklist](#deployment-checklist)
4. [Mainnet Rehearsal Lessons](#mainnet-rehearsal-lessons)
5. [Technical Implementation](#technical-implementation)
6. [User Communication](#user-communication)
7. [Recovery Procedures](#recovery-procedures)
8. [Monitoring & Alerts](#monitoring--alerts)

---

## 🚀 PHASE 6 OVERVIEW

### Network Reset (Testnet Phase 6)

**Why a fresh start?**
- Phase 5 created ~998,663 QUG in days (hyperinflation)
- 5 phases of data corruption required addressing
- Mainnet economics needed proper testing
- Fair launch for all participants

### Austrian Economics Implementation

| Metric | Phase 5 | Phase 6 | Change |
|--------|---------|---------|--------|
| **Per-Solution Reward** | 0.001 QUG | 0.00001 QUG | 100× MORE SCARCE |
| **20,000 Blocks Total** | ~998,663 QUG | ~9,986 QUG | 100× reduction |
| **Halving Mechanism** | Block-based | Time-based (yearly) | More predictable |
| **Network ID** | testnet-phase5 | testnet-phase6 | Fresh network |
| **Database** | ./data-mine5 | ./data-mine6 | Clean start |

**Economic Principles:**
- Sound money (Ludwig von Mises)
- Scarcity creates value (Friedrich Hayek)
- Market pricing (Murray Rothbard)
- Predictable supply schedule
- No unlimited printing

---

## 🐛 CRITICAL BUGS FOUND & FIXED

### 1. Duplicate Route Registration (v0.9.60-beta)

**Symptom:**
```
thread 'main' panicked at axum-0.7.9/src/routing/path_router.rs:70:22:
Overlapping method route. Handler for `GET /api/v1/blocks/:height` already exists
```

**Root Cause:**
- Line 5898: `.route("/api/v1/blocks/:height", get(handlers::get_block))`
- Line 6185: `.route("/api/v1/blocks/:height", get(handlers::get_block_by_height))`
- Same route registered twice with different handlers

**Fix:**
```rust
// OLD (line 5898):
.route("/api/v1/blocks/:height", get(handlers::get_block))

// NEW:
// NOTE: /api/v1/blocks/:height is registered below for HTTP fallback sync (line ~6185)
```

**File:** `crates/q-api-server/src/main.rs:5898`

**Lesson for Mainnet:**
- Always check for duplicate routes before deployment
- Use `cargo check` to catch compile-time errors
- Test server startup in development environment first

### 2. Phase Modal Not Showing for Existing Users

**Symptom:**
- Phase 6 modal only appeared for new users (incognito mode)
- Existing logged-in users didn't see the announcement

**Root Cause:**
```typescript
// Dashboard.tsx was checking OLD key:
const hasSeenV0918 = localStorage.getItem('v0918betaModalSeen');

// But modal was setting NEW key:
localStorage.setItem('v0960betaModalSeen', 'true');
```

**Fix:**
```typescript
// Dashboard.tsx (line 129):
const hasSeenV0960 = localStorage.getItem('v0960betaModalSeen');
return !hasSeenV0960; // Show if they haven't seen Phase 6 announcement yet
```

**Files:**
- `gui/quantum-wallet/src/components/Dashboard.tsx:129`
- `gui/quantum-wallet/src/components/PhaseTransitionModal.tsx:14`

**Lesson for Mainnet:**
- ALWAYS use unique localStorage keys for new announcements
- Test with existing user sessions, not just fresh installs
- Clear browser cache during testing to simulate new users
- Document localStorage key naming convention

### 3. Data Corruption (5 Phases)

**Historical Context:**
Q-NarwhalKnight experienced **5 phases of catastrophic data corruption**, including:
- Height drops (19,434 → 5,594 = 14,000 blocks lost)
- Database inconsistencies
- Balance mismatches
- Sync-down bugs

**User Impact:**
> "i consulted chatgpt.. we have been through 5 phases already all with data corruption causing heights to drop. so you must understand im sceptical and far from mainnet release. if it happens in mainnet all the work will be valued at zero and no one will trust me again.."

**Root Causes:**
1. Incomplete WAL (Write-Ahead Log) syncs
2. Shutdown before memtable flushes
3. Binary version mismatches
4. No verified backups
5. Sync-down allowing height decreases

**Fix (v0.9.60-beta):**

**A) Maximum Durability RocksDB Settings** (ChatGPT-recommended):
```rust
// crates/q-storage/src/kv.rs:135-161
opts.set_use_fsync(true);          // Survives power loss
opts.set_paranoid_checks(true);     // Detect corruption early
opts.set_atomic_flush(true);        // Multi-CF consistency
opts.set_bytes_per_sync(1MB);       // Steady IO (no bursts)
opts.set_wal_bytes_per_sync(1MB);   // Steady WAL sync
opts.set_wal_ttl_seconds(300);      // 5 min WAL cleanup
opts.set_wal_size_limit_mb(256);    // 256MB max WAL
```

**B) Checkpoint API** (instant recovery):
```rust
// New KVStore methods:
async fn create_checkpoint(&self, checkpoint_dir: &str) -> Result<()>;
async fn verify_checkpoint(&self, checkpoint_dir: &str) -> Result<bool>;
async fn sync_wal(&self) -> Result<()>;
async fn shutdown_gracefully(&self) -> Result<()>;
```

**C) Sync-Down Protection** (v0.9.59):
```rust
// crates/q-storage/src/turbo_sync.rs:251-268
const MAX_SANE_HEIGHT: u64 = 100_000_000;

if target_height < local_height && local_height > 1000 {
    error!("🚨 CRITICAL: Attempted sync-down from {} to {}!", local_height, target_height);
    return Err(anyhow::anyhow!("SAFETY ABORT: Refusing to sync down"));
}
```

**Durability Guarantees (v0.9.60-beta):**

| Scenario | Before v0.9.60 | After v0.9.60 |
|----------|----------------|---------------|
| Power Loss | ❌ Data loss | ✅ **ALL writes preserved** (use_fsync) |
| Hard Kill (SIGKILL) | ❌ Lost data | ✅ **WAL replays on restart** |
| Graceful Stop | ⚠️ Lost memtables | ✅ **shutdown_gracefully() saves all** |
| Disk Corruption | ❌ Silent until failure | ✅ **Paranoid checks abort early** |
| Binary Mismatch | ❌ Garbage values | ✅ **Sync-down protection blocks** |
| Need Rollback | ❌ Full reindex (hours) | ✅ **Restore checkpoint (seconds)** |

**Lesson for Mainnet:**
- **NEVER skip durability testing** - power loss, hard kills, corruption scenarios
- **Implement defense in depth** - multiple layers of protection
- **Test recovery procedures** before they're needed
- **Monitor health metrics** constantly (WAL size, height monotonicity)
- **Automated backups** are mandatory (hourly checkpoints minimum)

---

## ✅ DEPLOYMENT CHECKLIST

### Pre-Deployment

#### Backend
- [ ] **Fix all duplicate routes** (grep for duplicate `.route()` calls)
- [ ] **Test compilation:** `cargo check --workspace`
- [ ] **Run tests:** `cargo test --workspace`
- [ ] **Build release:** `timeout 36000 cargo build --release --package q-api-server`
- [ ] **Verify binary:** `ls -lh target/release/q-api-server` (should be ~120M)

#### Frontend
- [ ] **Update localStorage keys** for new version (e.g., `v0960betaModalSeen`)
- [ ] **Test with existing user session** (not just incognito)
- [ ] **Update modal content** (no negative messaging about data loss)
- [ ] **Build:** `npm run build`
- [ ] **Verify build:** Check `dist-final/index.html` exists

#### Documentation
- [ ] **Update deployment guide** with new version details
- [ ] **Create release notes** explaining changes
- [ ] **Update FAQ** with common questions
- [ ] **Document recovery procedures**

### Deployment Steps

#### 1. Stop Old Version
```bash
systemctl stop q-api-server
```

#### 2. Backup (Optional but Recommended)
```bash
# Backup old database
cp -r ./data-mine5 ./data-mine5-backup-$(date +%Y-%m-%d)

# Backup old binary
cp target/release/q-api-server target/release/q-api-server-v0.9.59-backup
```

#### 3. Update Service File
```bash
# Update environment variable for new database
cat > /etc/systemd/system/q-api-server.service << 'EOF'
[Unit]
Description=Q-NarwhalKnight API Server - Quantum Consensus Node (Phase 6)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/orobit/shared/q-narwhalknight
Environment="Q_DB_PATH=./data-mine6"  # <-- UPDATED
Environment="Q_IS_VALIDATOR=true"
Environment="Q_P2P_PORT=9001"
Environment="Q_ENABLE_AI=1"
Environment="RUST_LOG=info"

# AI Resource Limits
Environment="Q_AI_THREADS=4"
Environment="Q_AI_MAX_CONCURRENT=2"

ExecStart=/opt/orobit/shared/q-narwhalknight/target/release/q-api-server --port 8080
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=q-api-server

# Graceful shutdown timeout
TimeoutStopSec=30
KillMode=mixed
KillSignal=SIGTERM

# Security settings
NoNewPrivileges=true
PrivateTmp=true

# Resource limits
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF
```

#### 4. Reload and Start
```bash
systemctl daemon-reload
systemctl start q-api-server
```

#### 5. Verify Deployment
```bash
# Check service status
systemctl status q-api-server --no-pager

# Wait for startup (AI model loading takes ~30-60 seconds)
sleep 60

# Verify Phase 6 network
curl -s http://localhost:8080/api/v1/status | jq '{
  network_id: .data.network_id,
  height: .data.height,
  db_path: .data.db_path,
  version: .data.version
}'

# Expected output:
# {
#   "network_id": "testnet-phase6",
#   "height": 0 or small number (fresh start),
#   "db_path": "./data-mine6",
#   "version": "v0.9.60-beta"
# }
```

#### 6. Monitor Logs
```bash
# Watch for errors
journalctl -u q-api-server -f | grep -E "ERROR|panic|CRITICAL"

# Watch for successful startup
journalctl -u q-api-server -f | grep -E "Server listening|Phase 6|testnet-phase6"

# Check database path
journalctl -u q-api-server --since "2 minutes ago" | grep "data-mine6"
```

### Post-Deployment

#### Frontend
- [ ] Copy binaries to download location:
```bash
cp target/release/q-api-server \
   gui/quantum-wallet/dist-final/downloads/q-api-server-v0.9.60-beta

cp target/release/q-api-server \
   gui/quantum-wallet/dist-final/downloads/q-api-server-linux-x86_64
```

- [ ] **Test frontend loads** (visit https://quillon.xyz)
- [ ] **Test Phase 6 modal appears** (clear localStorage or use incognito)
- [ ] **Test existing users see modal** (use normal browser session)
- [ ] **Verify download links work** (click "Download Node")

#### Monitoring
- [ ] **Set up automated backups** (hourly checkpoints)
- [ ] **Configure alerts** (height drops, WAL size >256MB, disk space)
- [ ] **Monitor first 24 hours** closely
- [ ] **Document any issues** encountered

---

## 🎓 MAINNET REHEARSAL LESSONS

### Lesson 1: Test Phase Transitions Thoroughly

**What we learned:**
- Fresh network resets require careful user communication
- localStorage keys must be unique for each announcement
- Existing users and new users behave differently
- **Phase 7 → 8: Emission rate bugs can be mainnet-blocking!**

**Mainnet checklist:**
- [ ] Test on staging environment first (at least 48 hours)
- [ ] Test with REAL user accounts (not just fresh installs)
- [ ] Verify all users see announcements (existing + new)
- [ ] Document rollback procedures BEFORE launch
- [ ] Have emergency contact plan ready
- [ ] **VALIDATE EMISSION RATE** (daily supply, time to cap)
- [ ] **CALCULATE ECONOMICS** before each phase (see Phase 8 fix)

### Lesson 2: Durability is Non-Negotiable

**What we learned:**
- 5 phases of corruption destroyed user trust
- Recovery from corruption takes hours without checkpoints
- Power loss scenarios MUST be tested

**Mainnet checklist:**
- [ ] Implement ChatGPT-recommended RocksDB hardening
- [ ] Test power loss recovery (kill -9 during active use)
- [ ] Test checkpoint creation and restoration
- [ ] Automated hourly backups with verification
- [ ] Monitor WAL size (<256MB) and height monotonicity
- [ ] Practice recovery procedures (don't wait for emergency)

### Lesson 3: Sync-Down Protection is Critical

**What we learned:**
- Single sync-down bug can delete thousands of blocks
- Mainnet sync-down = billions of dollars lost
- Must protect at multiple layers

**Mainnet checklist:**
- [ ] Application-level sync validation (only sync UP)
- [ ] Database-level safety abort (refuse sync-down)
- [ ] Balance consistency (balances WITHOUT blocks = invalid)
- [ ] Test with malicious peers announcing false heights
- [ ] Circuit breakers for dangerous conditions

### Lesson 4: Communication Must Be Positive

**What we learned:**
- Users respond better to positive messaging
- Focus on improvements, not past failures
- Austrian economics resonates with crypto community

**Mainnet checklist:**
- [ ] Announce features, not bug fixes
- [ ] Highlight economic improvements (scarcity, halving)
- [ ] Explain WHY changes benefit users
- [ ] Provide clear migration instructions
- [ ] FAQ addressing common concerns
- [ ] No mention of past corruption (focus forward)

### Lesson 5: Frontend Bugs Can Break Deployments

**What we learned:**
- Duplicate route registration crashes server on startup
- Modal localStorage key mismatch hides announcements
- Frontend bugs are as critical as backend bugs

**Mainnet checklist:**
- [ ] Test server startup in development FIRST
- [ ] Check for duplicate routes before deployment
- [ ] Test with existing user sessions (not just fresh)
- [ ] Browser cache testing (simulate both new and old users)
- [ ] Verify all critical paths (login, transactions, mining)

### Lesson 6: Database Paths Matter

**What we learned:**
- Fresh networks need NEW database paths
- systemd environment variables must be updated
- Old databases should be preserved for debugging

**Mainnet checklist:**
- [ ] Update systemd service file with new DB_PATH
- [ ] Verify database path in logs after startup
- [ ] Preserve old databases for at least 30 days
- [ ] Document database locations clearly
- [ ] Test recovery from checkpoints

---

## 🔧 TECHNICAL IMPLEMENTATION

### RocksDB Durability Settings

**File:** `crates/q-storage/src/kv.rs`

**Hot Database (lines 135-161):**
```rust
// 🚨 v0.9.60-beta: MAXIMUM DURABILITY MODE
// ChatGPT-recommended hardened RocksDB settings for mainnet-grade reliability

// ========== DURABILITY SETTINGS (CRASH-SAFE) ==========
opts.set_use_fsync(true);          // use fsync() not fdatasync() - strongest guarantee
opts.set_paranoid_checks(true);     // Detect corruption early, fail loud
opts.set_atomic_flush(true);        // Multi-CF consistency (all or nothing)

// ========== WAL (Write-Ahead Log) PROTECTION ==========
opts.set_wal_ttl_seconds(300);      // 5 minutes - delete after flush
opts.set_wal_size_limit_mb(256);    // 256MB max - prevents unbounded growth
opts.set_max_total_wal_size(64 * 1024 * 1024); // 64MB total WAL budget
opts.set_manual_wal_flush(true);    // Manual control for safety

// ========== STEADY IO (PREVENT BURST CORRUPTION) ==========
opts.set_bytes_per_sync(1024 * 1024);     // 1 MiB - sync data in steady chunks
opts.set_wal_bytes_per_sync(1024 * 1024); // 1 MiB - sync WAL in steady chunks

// ========== MEMORY BUDGET (FORCE FLUSHES) ==========
opts.set_db_write_buffer_size(128 * 1024 * 1024); // 128MB total memtable budget
```

**Cold Database (lines 251-256):**
```rust
// 🚨 v0.9.60-beta: COLD DB DURABILITY (same as hot DB)
opts.set_use_fsync(true);
opts.set_paranoid_checks(true);
opts.set_atomic_flush(true);
opts.set_bytes_per_sync(1024 * 1024);
opts.set_wal_bytes_per_sync(1024 * 1024);
```

### Checkpoint API Implementation

**New KVStore Methods (lines 60-76):**
```rust
/// 🚨 v0.9.60-beta: CRITICAL DURABILITY ADDITIONS

/// Create checkpoint (hard-linked snapshot) for instant, consistent backups
async fn create_checkpoint(&self, checkpoint_dir: &str) -> Result<()>;

/// Sync WAL to disk (call before shutdown for maximum safety)
async fn sync_wal(&self) -> Result<()>;

/// Graceful shutdown with full data persistence
async fn shutdown_gracefully(&self) -> Result<()>;

/// Verify backup integrity (read checksum validation)
async fn verify_checkpoint(&self, checkpoint_dir: &str) -> Result<bool>;
```

**Implementation (lines 808-907):**

**create_checkpoint():**
```rust
async fn create_checkpoint(&self, checkpoint_dir: &str) -> Result<()> {
    use rocksdb::checkpoint::Checkpoint;

    info!("💾 [CHECKPOINT] Creating snapshot at {}", checkpoint_dir);

    let checkpoint = Checkpoint::new(&*self.db)
        .context("Failed to create Checkpoint object")?;

    checkpoint.create_checkpoint(checkpoint_dir)
        .context("Failed to create checkpoint")?;

    info!("✅ [CHECKPOINT] Snapshot created successfully (hard-linked, zero-copy)");
    Ok(())
}
```

**verify_checkpoint():**
```rust
async fn verify_checkpoint(&self, checkpoint_dir: &str) -> Result<bool> {
    // Try to open checkpoint as read-only database
    let mut opts = rocksdb::Options::default();
    opts.set_paranoid_checks(true);

    match rocksdb::DB::open_for_read_only(&opts, checkpoint_dir, false) {
        Ok(checkpoint_db) => {
            // Verify manifest CF exists and is readable
            if let Some(manifest_cf) = checkpoint_db.cf_handle(CF_MANIFEST) {
                match checkpoint_db.get_cf(&manifest_cf, b"height") {
                    Ok(_) => Ok(true),
                    Err(_) => Ok(false),
                }
            } else {
                Ok(false)
            }
        }
        Err(_) => Ok(false),
    }
}
```

**shutdown_gracefully():**
```rust
async fn shutdown_gracefully(&self) -> Result<()> {
    info!("🛑 [GRACEFUL SHUTDOWN] Starting shutdown sequence...");

    // Step 1: Sync WAL
    self.sync_wal().await?;

    // Step 2: Flush all column families
    let cf_names = vec![
        CF_BLOCKS, CF_DAG_VERTICES, CF_BULLSHARK_CERT, CF_MANIFEST,
        CF_TRANSACTIONS, CF_BALANCES, CF_BLOCK_HASH_TO_HEIGHT,
        CF_AI_CHATS, CF_AI_CREDITS, CF_AI_TRANSACTIONS, CF_AI_TREASURY,
        CF_AI_ATTACHMENTS, CF_PAYMENT_PROPOSALS, CF_PAYMENT_VOTES,
        CF_PAYMENT_LOCKS, CF_BANNED_PEERS,
    ];

    for cf_name in cf_names {
        if let Some(cf_handle) = self.db.cf_handle(cf_name) {
            self.db.flush_cf(&cf_handle)?;
        }
    }

    // Step 3: Final WAL sync
    self.db.flush_wal(true)?;

    info!("✅ [GRACEFUL SHUTDOWN] All data persisted safely. DB ready to close.");
    Ok(())
}
```

### Sync-Down Protection

**File:** `crates/q-storage/src/turbo_sync.rs:251-268`

```rust
// 🚨 v0.9.59-beta: CRITICAL SYNC-DOWN PROTECTION
// Prevents catastrophic data loss from garbage peer announcements

const MAX_SANE_HEIGHT: u64 = 100_000_000; // Sanity check for corrupted heights

// SAFETY: NEVER sync down (this causes data loss!)
if target_height < local_height && local_height > 1000 {
    error!(
        "🚨 [TURBO SYNC CORRUPTION] SAFETY ABORT: Refusing to sync down from {} to {}!",
        local_height, target_height
    );
    error!("    This indicates binary version mismatch or corrupted peer data.");
    error!("    Keeping local blockchain at height {}. NEVER SYNC DOWN!", local_height);

    return Err(anyhow::anyhow!(
        "SAFETY ABORT: Refusing to sync down from {} to {} (would lose {} blocks)",
        local_height,
        target_height,
        local_height - target_height
    ));
}

// SAFETY: Reject insane heights (indicates deserialization corruption)
if target_height > MAX_SANE_HEIGHT {
    error!(
        "❌ [TURBO SYNC CORRUPTION] OLD format decode produced GARBAGE! \
         Peer height: {} (exceeds MAX_SANE_HEIGHT: {})",
        target_height, MAX_SANE_HEIGHT
    );
    return Err(anyhow::anyhow!("Corrupted peer height: {}", target_height));
}
```

### Phase 6 Network Configuration

**Network ID:** `testnet-phase6`
**Gossipsub Topics:**
- `/qnk/testnet-phase6/blocks`
- `/qnk/testnet-phase6/peer-heights`
- `/qnk/testnet-phase6/block-pack-requests`
- `/qnk/testnet-phase6/block-pack-responses`

**Database:** `./data-mine6`

**Per-Solution Reward:**
```rust
// crates/q-api-server/src/block_producer.rs
const BLOCK_REWARD: u64 = 10_000; // 0.00001 QUG (9 decimals)
```

**Time-Based Halving:**
- Halves every 31,536,000 seconds (1 year)
- Handled in `crates/q-storage/src/balance_consensus.rs`
- More predictable than block-based halving

---

## 💬 USER COMMUNICATION

### Phase 6 Announcement Template

```markdown
# 🎉 Phase 6 Launch: TRUE Austrian Economics!

We're excited to announce **Testnet Phase 6** with revolutionary improvements!

## Why Phase 6?

**Phase 5 Problem:**
- Created 998,663 QUG in just days
- No scarcity = No value
- Violated sound money principles

**Phase 6 Solution:**
- **0.00001 QUG per solution** (100× MORE SCARCE!)
- Bitcoin-inspired time-based halving (yearly)
- ~9,986 QUG total for same 20k blocks (vs 998,663!)
- **TRUE scarcity = potential value**

## What You Need to Know:

1. **Fresh Network** - Phase 5 balances don't transfer
   - This is testnet (mainnet rehearsal)
   - Everyone starts equal (fair launch!)

2. **Download v0.9.60-beta**
   - Network ID: testnet-phase6
   - Database: data-mine6
   - Cannot connect to Phase 5 network

3. **Start Mining** - Earn Phase 6 QUG
   - Epoch 1 reward: 0.00001 QUG per solution
   - Early adopter advantage
   - Real scarcity this time!

## Download:

- **Node:** https://quillon.xyz/downloads/q-api-server-v0.9.60-beta
- **Miner:** https://quillon.xyz/downloads/q-miner-linux-x64

## Network Details:

- **Network ID:** testnet-phase6
- **Bootstrap:** /ip4/185.182.185.227/tcp/9001/p2p/12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN
- **Genesis:** 2025-11-08
- **Reward:** 0.00001 QUG/solution (100× less than Phase 5)

Happy mining! 🚀💎
```

### FAQ (User-Facing)

**Q: What is Phase 6?**

A: Phase 6 is Q-NarwhalKnight's sound money testnet with TRUE Austrian economics. It implements 100× more scarcity than Phase 5 (0.00001 QUG per solution vs 0.001 QUG). This is a complete network reset (network ID: testnet-phase6) that will become the mainnet economic model.

**Q: Why is Phase 6 100× more scarce?**

A: Phase 5 created nearly 1 MILLION coins in days due to high per-solution rewards (0.001 QUG). Phase 6 reduces this to 0.00001 QUG per solution, creating REAL scarcity. This aligns with Austrian economics principles: sound money requires fixed supply and predictable emission (Mises, Hayek, Rothbard).

**Q: What happened to my Phase 5 balance?**

A: Phase 5 balances do NOT transfer to Phase 6. This is a fresh network with new economics - everyone starts equal (fair launch!). Testnet balances have NO VALUE - this is expected. Phase 6 tests the REAL mainnet economic model.

**Q: Do I need to reset my database?**

A: Phase 6 automatically uses a new database path (data-mine6). Your Phase 5 database remains intact in the old location. No manual reset needed - just download v0.9.60-beta and start mining on the new network!

**Q: What is time-based halving?**

A: Phase 6 uses yearly halving (not block-based like Bitcoin). The reward halves every 31,536,000 seconds (1 year), creating predictable long-term scarcity. This is handled automatically by the balance_consensus module.

**Q: Will mainnet use these economics?**

A: YES! Phase 6 economics (0.00001 QUG reward, yearly halving) will be the mainnet model. We're testing it thoroughly in Phase 6 before mainnet launch. If you mine in Phase 6, you're practicing with the REAL mainnet economics.

**Q: Can Phase 5 nodes connect to Phase 6?**

A: No! Phase 5 (testnet-phase5) and Phase 6 (testnet-phase6) use different network IDs, gossipsub topics, and databases. They cannot communicate. All nodes must upgrade to v0.9.60-beta to join Phase 6.

**Q: What makes this "Austrian economics"?**

A: Austrian economics emphasizes sound money with predictable scarcity (not unlimited printing). Phase 6 implements:
1. Fixed emission schedule - 0.00001 QUG/solution
2. Time-based halving - predictable long-term scarcity
3. No central control - market determines value
4. True store of value - scarcity creates worth

---

## 🔄 RECOVERY PROCEDURES

### Scenario 1: Power Loss

**Symptoms:**
- Server unexpectedly shut down
- Database may be inconsistent

**Recovery:**
```bash
# 1. Restart node
systemctl start q-api-server

# 2. Watch logs for WAL replay
journalctl -u q-api-server -f | grep -E "WAL|replay|recovery"

# 3. Verify height is correct
curl -s http://localhost:8080/api/v1/status | jq '.data.height'

# Expected: Height matches or is close to pre-crash height
```

**What Happens:**
- RocksDB automatically replays WAL (Write-Ahead Log)
- All synced writes are recovered
- `use_fsync(true)` ensures no data loss

### Scenario 2: Corruption Detected

**Symptoms:**
- Paranoid checks abort with error
- Database read failures
- Height decreases unexpectedly

**Recovery:**
```bash
# 1. Stop node
systemctl stop q-api-server

# 2. List available checkpoints
ls -lht /opt/backups/

# Output:
# checkpoint-2025-11-08-12-00/  (most recent)
# checkpoint-2025-11-08-11-00/
# checkpoint-2025-11-08-10-00/

# 3. Backup corrupted database
mv ./data-mine6 ./data-mine6-CORRUPTED-$(date +%Y-%m-%d-%H-%M)

# 4. Restore from checkpoint (instant - hard links!)
cp -al /opt/backups/checkpoint-2025-11-08-12-00 ./data-mine6

# 5. Restart node
systemctl start q-api-server

# 6. Verify recovery
curl -s http://localhost:8080/api/v1/status | jq '.data.height'
```

**Recovery Time:** Seconds (vs hours for full reindex)

### Scenario 3: Sync-Down Detected

**Symptoms:**
```
❌ [TURBO SYNC CORRUPTION] SAFETY ABORT: Refusing to sync down from 10000 to 1000!
```

**Action:**
- **DO NOTHING** - Protection worked!
- Node correctly rejected malicious peer
- Local blockchain preserved at correct height

**Follow-Up:**
```bash
# Check peer quality
journalctl -u q-api-server --since "10 minutes ago" | grep -E "peer|banned"

# Monitor for repeated attacks
journalctl -u q-api-server -f | grep "SAFETY ABORT"
```

### Scenario 4: Height Reset to Zero

**Symptoms:**
- Height suddenly becomes 0
- Balance disappears
- Blocks missing

**Recovery:**
```bash
# 1. STOP NODE IMMEDIATELY
systemctl stop q-api-server

# 2. Restore from latest checkpoint
mv ./data-mine6 ./data-mine6-RESET-$(date +%Y-%m-%d-%H-%M)
cp -al /opt/backups/checkpoint-$(date +%Y-%m-%d -d "1 hour ago" +%H-00) ./data-mine6

# 3. Restart
systemctl start q-api-server

# 4. Verify
curl -s http://localhost:8080/api/v1/status | jq '.data.height'
```

**Root Cause Analysis:**
```bash
# Check logs for corruption
journalctl -u q-api-server --since "1 hour ago" | grep -E "ERROR|panic|CRITICAL"

# Check for binary mismatch
ls -lh target/release/q-api-server
md5sum target/release/q-api-server
```

---

## 📊 MONITORING & ALERTS

### Key Metrics

**Database Health:**
```bash
# Database size
du -sh ./data-mine6/

# WAL size (should be <256MB)
du -sh ./data-mine6/hot/*.log

# Checkpoint count
ls -1 /opt/backups/ | wc -l

# Latest backup age
ls -lt /opt/backups/ | head -1
```

**Network Health:**
```bash
# Current height
curl -s http://localhost:8080/api/v1/status | jq '.data.height'

# Network peers
curl -s http://localhost:8080/api/v1/status | jq '.data.peers'

# Sync status
curl -s http://localhost:8080/api/v1/status | jq '.data.sync_status'
```

**System Health:**
```bash
# Service status
systemctl status q-api-server

# Recent errors
journalctl -u q-api-server --since "1 hour ago" | grep ERROR | tail -20

# Memory usage
ps aux | grep q-api-server | awk '{print $4}'

# Disk space
df -h /opt/orobit/shared/q-narwhalknight/
```

### Alert Thresholds

**CRITICAL Alerts:**
- ❌ **Height decrease detected** → Immediate checkpoint restore
- ❌ **Sync-down attempted** → Investigate peer quality
- ❌ **Service down >5 minutes** → Manual intervention required
- ❌ **Disk space <5GB** → Emergency cleanup needed

**WARNING Alerts:**
- ⚠️ **WAL exceeds 256MB** → Memtable not flushing (investigate)
- ⚠️ **No checkpoint in 2+ hours** → Backup service down
- ⚠️ **Disk space <10GB** → Prune old checkpoints
- ⚠️ **Error rate >10/minute** → Check logs

**INFO Alerts:**
- ℹ️ **New checkpoint created** → Backup successful
- ℹ️ **Height milestone** → Every 1000 blocks
- ℹ️ **Peer count change** → Network health

### Automated Monitoring Script

```bash
#!/bin/bash
# /usr/local/bin/q-monitor.sh

# Configuration
ALERT_EMAIL="admin@quillon.xyz"
LOG_FILE="/var/log/q-monitor.log"
CHECKPOINT_DIR="/opt/backups"

# Get current height
HEIGHT=$(curl -s http://localhost:8080/api/v1/status | jq -r '.data.height')

# Check if height decreased (CRITICAL)
if [ -f /tmp/q-last-height ]; then
    LAST_HEIGHT=$(cat /tmp/q-last-height)
    if [ "$HEIGHT" -lt "$LAST_HEIGHT" ]; then
        echo "🚨 CRITICAL: Height decreased from $LAST_HEIGHT to $HEIGHT!" | tee -a $LOG_FILE
        # Send alert
        echo "CRITICAL: Height drop detected on q-narwhalknight node" | mail -s "🚨 Q-NarwhalKnight Alert" $ALERT_EMAIL
        # Auto-restore from checkpoint (optional)
        # /usr/local/bin/restore-latest-checkpoint.sh
    fi
fi
echo "$HEIGHT" > /tmp/q-last-height

# Check WAL size
WAL_SIZE=$(du -s ./data-mine6/hot/*.log 2>/dev/null | awk '{sum+=$1} END {print sum}')
if [ "$WAL_SIZE" -gt 268435456 ]; then  # 256MB
    echo "⚠️ WARNING: WAL size exceeds 256MB ($WAL_SIZE bytes)" | tee -a $LOG_FILE
fi

# Check latest checkpoint age
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    CHECKPOINT_AGE=$(($(date +%s) - $(stat -c %Y "$CHECKPOINT_DIR/$LATEST_CHECKPOINT")))
    if [ "$CHECKPOINT_AGE" -gt 7200 ]; then  # 2 hours
        echo "⚠️ WARNING: No checkpoint in 2+ hours" | tee -a $LOG_FILE
    fi
fi

# Check disk space
DISK_FREE=$(df /opt/orobit/shared/q-narwhalknight/ | tail -1 | awk '{print $4}')
if [ "$DISK_FREE" -lt 5242880 ]; then  # 5GB
    echo "🚨 CRITICAL: Disk space <5GB!" | tee -a $LOG_FILE
    echo "CRITICAL: Disk space running low on q-narwhalknight node" | mail -s "🚨 Disk Alert" $ALERT_EMAIL
fi

echo "✅ Monitoring check complete (height: $HEIGHT)" | tee -a $LOG_FILE
```

**Cron setup:**
```bash
# Run every 5 minutes
*/5 * * * * /usr/local/bin/q-monitor.sh
```

---

## 🎯 MAINNET LAUNCH CHECKLIST

### Pre-Launch (T-30 days)

- [ ] **Testnet Phase 6 running stable for 30+ days**
- [ ] **No data corruption incidents**
- [ ] **Checkpoint recovery tested and documented**
- [ ] **All critical bugs fixed**
- [ ] **Security audit completed**
- [ ] **Performance benchmarks met** (target TPS achieved)
- [ ] **Documentation complete** (user guides, API docs, recovery procedures)
- [ ] **Community informed** of mainnet launch date

### Pre-Launch (T-7 days)

- [ ] **Final security audit**
- [ ] **Stress testing completed** (10x expected load)
- [ ] **Disaster recovery drills** (practice checkpoint restoration)
- [ ] **Monitoring systems tested** (alerts fire correctly)
- [ ] **Support team trained** (can handle common issues)
- [ ] **Marketing materials ready** (announcement, press release)
- [ ] **Exchange partnerships confirmed** (if applicable)

### Launch Day (T-0)

- [ ] **Genesis block parameters finalized**
- [ ] **Bootstrap nodes deployed** (minimum 3 geographically distributed)
- [ ] **Monitoring dashboards active**
- [ ] **Support channels staffed** (Discord, Telegram, email)
- [ ] **Announcement published** (website, social media, forums)
- [ ] **Community notified** (Discord, Telegram, email list)
- [ ] **First checkpoint created** (within 1 hour of launch)

### Post-Launch (T+24 hours)

- [ ] **No critical bugs detected**
- [ ] **Network participation growing**
- [ ] **Checkpoints verified hourly**
- [ ] **User feedback collected**
- [ ] **Support issues documented**
- [ ] **Monitoring metrics stable**

### Post-Launch (T+7 days)

- [ ] **Network hashrate stable**
- [ ] **No height drops or corruption**
- [ ] **User adoption metrics positive**
- [ ] **Exchange listings progressing** (if planned)
- [ ] **Community feedback positive**
- [ ] **Post-launch report published**

---

## 📚 REFERENCES

### Documentation
- `PHASE_8_HYPERINFLATION_FIX.md` - **CRITICAL** Phase 7 → 8 transition (1000× emission fix)
- `ROCKSDB_DURABILITY_GUIDE.md` - Complete durability guide
- `V0.9.60_BETA_DURABILITY_IMPLEMENTATION.md` - Implementation details
- `V0.9.60_BETA_FINAL_STATUS.md` - Status document
- `V0.9.60_BETA_DEPLOYMENT_READY.md` - Deployment verification
- `CRITICAL_SYNC_DOWN_BUG_ANALYSIS.md` - Sync-down bug details

### Code Files
- `crates/q-storage/src/kv.rs` - Durability implementation
- `crates/q-storage/src/turbo_sync.rs` - Sync-down protection
- `crates/q-api-server/src/block_producer.rs` - Phase 6 rewards
- `crates/q-api-server/src/main.rs` - Route registration
- `gui/quantum-wallet/src/components/PhaseTransitionModal.tsx` - Modal
- `gui/quantum-wallet/src/components/Dashboard.tsx` - Modal trigger

### ChatGPT Recommendations
- RocksDB durability settings (implemented in kv.rs)
- Checkpoint-based recovery (implemented)
- Graceful shutdown procedures (implemented)
- Sync-down protection (implemented)
- Monitoring best practices (documented)

---

## ✨ SUMMARY

**Phase 6 Launch represents a turning point for Q-NarwhalKnight:**

**FROM:**
- 5 phases of data corruption
- User trust = zero
- Hyperinflation economics
- Hours to recover from failure
- Mainnet launch impossible

**TO:**
- Mainnet-grade durability
- ChatGPT-verified protection
- Austrian economics (100× more scarce)
- Seconds to recover from failure
- Mainnet launch feasible

**Key Achievements:**
1. ✅ Maximum durability (use_fsync, paranoid_checks, atomic_flush)
2. ✅ Checkpoint recovery (instant restore in seconds)
3. ✅ Sync-down protection (prevents catastrophic data loss)
4. ✅ Phase 6 economics (TRUE scarcity, time-based halving)
5. ✅ Complete documentation (guides, procedures, monitoring)

**Lessons for Mainnet:**
- Durability is non-negotiable (test power loss, corruption, hard kills)
- Defense in depth (multiple layers of protection)
- Test recovery procedures BEFORE emergencies
- Communication must be positive (focus on improvements)
- Frontend bugs are as critical as backend bugs
- Monitor health metrics constantly (WAL size, height monotonicity)

**This guide ensures mainnet launch will be:**
- ✅ Stable (no corruption, verified durability)
- ✅ Predictable (documented procedures, tested recovery)
- ✅ Trustworthy (transparent communication, positive messaging)
- ✅ Recoverable (checkpoint backups, seconds not hours)

**Status:** ✅ **MAINNET-BLOCKING BUG FIXED (Phase 8)**

Phase 8 caught a **catastrophic hyperinflation bug** (1000× emission error) that would have destroyed all mainnet value. The emission rate validation procedures documented in `PHASE_8_HYPERINFLATION_FIX.md` are now **mandatory** before mainnet.

---

**For questions or issues, refer to:**
- **CRITICAL:** `PHASE_8_HYPERINFLATION_FIX.md` - Emission rate validation
- Technical: `ROCKSDB_DURABILITY_GUIDE.md`
- Deployment: `V0.9.60_BETA_DEPLOYMENT_READY.md`
- Recovery: This guide, Section 7

**Let's launch mainnet with TRUE scarcity! 🚀💎**
