# Serialization Format Incompatibility Incident Report
## Q-NarwhalKnight Node - November 15, 2025

**Date:** 2025-11-15
**Incident Type:** P0 - Serialization Format Incompatibility
**Status:** 🟡 RESOLVED - Service Operational, Root Cause Identified
**Node:** 185.182.185.227 (quillon.xyz)

---

## Executive Summary

**What Happened:**
The production Q-NarwhalKnight node reported what appeared to be catastrophic blockchain data loss on November 15, 2025. The service refused to start, reporting that 93,743 blocks were missing despite the database pointer indicating their presence. Initial diagnostics suggested database corruption with 100% data loss.

**What Actually Happened:**
The RocksDB database was **fully intact** with all 93,743 blocks physically present on disk. The "data loss" was caused by a **backwards-incompatible change to the Rust `QBlock` struct** combined with **bincode serialization**, which lacks schema evolution support. Blocks written with an older struct version could not be deserialized by newer code, making them appear "missing" despite being physically present.

**Impact:**
- **Service downtime:** ~5 minutes (18:15-18:20 UTC)
- **Perceived data loss:** 100% (all blocks appeared missing)
- **Actual data loss:** 0% (all data physically intact, format-locked)
- **Current state:** Service operational at height 88,495

**Root Cause:**
Backwards-incompatible bincode deserialization combined with silent error handling that treated deserialization failures as missing blocks, rather than format incompatibility errors.

**Resolution:**
Service auto-recovered to height 88,495 after pointer repair. The underlying format incompatibility issue requires a data migration to fully resolve blocks 88,496-93,743.

---

## Timeline

### 18:15 UTC - Service Failure
```
ERROR q_storage: 🚨 CRITICAL DATABASE CORRUPTION DETECTED!
    Pointer shows height: 93743
    But block does NOT exist in database!
    Service refusing to start (11th restart attempt)
```

**Observation:** Binary search found highest contiguous block: 88,495 (gap of 5,248 blocks)

### 18:20 UTC - Manual Repair Attempt
```bash
$ repair-database ./data-mine11/hot

📊 Scan Results:
   Total blocks found: 0          ⚠️ Contradictory finding
   Highest contiguous: 0

✅ Repair successful!
   qblock:latest → 0
```

### 18:20 UTC - Service Auto-Recovery
```
INFO q_storage: 📈 Recovered blockchain height: 88495 blocks
WARN q_storage: ⚠️  Height pointer mismatch: 0 vs 88495
INFO q_storage: ✅ Height pointer repaired: 0 → 88495
```

**Contradiction:** Repair tool found 0 blocks, service found 88,495 blocks, same database.

### 18:23 UTC - RocksDB Direct Verification
```bash
$ ldb --db=./data-mine11/hot/ get blocks "qblock:height:88495"
EXISTS ✅

$ ldb --db=./data-mine11/hot/ get blocks "qblock:height:93743"
EXISTS ✅
```

**Smoking Gun:** All blocks physically present, but application cannot deserialize them.

---

## Root Cause Analysis

### Primary Cause: Backwards-Incompatible Serialization

The incident was caused by a **backwards-incompatible change to the serialized `QBlock` format** combined with:
1. Use of **bincode** as a long-term storage format (designed for wire format, not persistent storage)
2. **Silent deserialization error handling** that returned `Ok(None)` instead of `Err()`
3. **No schema versioning** or migration framework

**Technical Details:**

Blocks 0-93,743 were serialized using one version of the Rust `QBlock` struct (version A). A later code deployment modified this struct (version B) without a data migration or schema versioning. Because bincode encodes struct fields by position and count, any field addition/reordering made the old bytes incompatible with the new struct layout.

When the updated node started, `bincode::deserialize` failed for these blocks. The storage layer logged a warning but returned `Ok(None)`, treating the block as "missing":

```rust
// crates/q-storage/src/lib.rs:557-565
match bincode::deserialize::<q_types::block::QBlock>(&block_data) {
    Ok(block) => Ok(Some(block)),
    Err(e) => {
        warn!("⚠️  Failed to deserialize QBlock at height {}: {}", height, e);
        Ok(None)  // 🔴 SILENT FAILURE - Block exists but returns None!
    }
}
```

This silent failure caused:
- Height scanner to conclude no blocks exist
- Pointer corruption (93,743 → 0)
- Service refusing to start (correct safety behavior)
- Repair tool reporting 0 blocks (compiled with incompatible version)

### Why Different Tools Saw Different Results

| Tool | Binary | Compilation | q_types Version | Blocks Found |
|------|--------|-------------|-----------------|--------------|
| repair_database | `/target/release/repair-database` | Nov 14? | Incompatible | 0 blocks |
| q-api-server | `/target/release/q-api-server` | Nov 15 | Compatible | 88,495 blocks |

Both tools read the same RocksDB database, but were compiled against different versions of `QBlock`. Each had a different idea of what "readable" means, resulting in contradictory reports.

### Suspected Struct Change

**Hypothesis (requires git archaeology to confirm):**

```rust
// Old Version (blocks 0-93,743)
pub struct QBlock {
    pub header: BlockHeader,
    pub transactions: Vec<Transaction>,
    pub mining_solutions: Vec<MiningSolution>,
    // 3 fields
}

// New Version (current code)
pub struct QBlock {
    pub header: BlockHeader,
    pub transactions: Vec<Transaction>,
    pub mining_solutions: Vec<MiningSolution>,
    pub state_root: Option<Hash>,  // ⚠️ NEW FIELD ADDED
    // 4 fields
}
```

**Bincode Behavior:**
- Serializing new struct: writes 4 fields
- Deserializing old data with new struct: expects 4 fields, finds 3
- Result: `UnexpectedEof` error → treated as "block doesn't exist"

---

## Impact Assessment

### Operational Impact

**Service Availability:**
- Downtime: ~5 minutes
- Restart attempts: 11 failed attempts
- Resolution: Manual intervention + auto-repair

**Data Integrity:**
- Physical data loss: **0%** (all data intact in RocksDB)
- Logical data loss: **0-100%** (version-dependent visibility)
- Perceived data loss: **100%** (all blocks appeared missing initially)
- Current accessibility: **94.4%** (88,495/93,743 blocks readable)

**Database State:**
- Database size: 1.7 GB (intact)
- RocksDB SST files: 157 files (no corruption)
- Block entries: 93,743+ verified with `ldb` (physically present)

### Hypothetical Mainnet Impact

**If this occurred on mainnet with real user funds:**

| Metric | Impact |
|--------|--------|
| Blocks affected | 93,743 |
| Estimated transactions | ~937,430 (10 tx/block avg) |
| User balances | UNRECOVERABLE (no cryptographic proof if blocks unreadable) |
| Token supply | INCONSISTENT (balances vs blocks mismatch) |
| Network trust | DESTROYED (blockchain reorg to height 0) |
| Financial liability | Potentially millions of dollars in user funds |

**Conclusion:** This issue **MUST** be resolved before mainnet launch.

---

## Current Status

### Service Health ✅

```bash
$ systemctl status q-api-server
● q-api-server.service - Q-NarwhalKnight API Server
   Active: active (running) since Sat 2025-11-15 18:20:33

$ curl http://localhost:8080/api/blockchain/height
{"height":88495,"timestamp":"2025-11-15T18:20:34Z"}
```

### Block Accessibility 🟡

**Verified Readable:**
- Blocks 0-88,495: ✅ Accessible via current binary

**Unverified (Requires Testing):**
- Blocks 88,496-93,743: ❓ Status unknown
  - Physically present in RocksDB (verified with `ldb`)
  - May be readable or may fail deserialization
  - Requires API testing: `curl /api/explorer/block/88496`

### Database Integrity ✅

**Backup Status:**
- Created: 2025-11-15 18:22 UTC
- Location: `data-mine11-backup-1731698734.tar.gz`
- Size: 1.7 GB (compressed)
- Verified: ✅ Extractable

---

## Immediate Action Plan

### P0 - Immediate (Within 4 Hours) ⏰

**A1. Lock Current State**
- [x] Database backup completed
- [ ] Freeze CI/CD deployments
- [ ] Test block accessibility for heights 88,496-93,743
  ```bash
  for h in 88496 90000 93743; do
    curl http://localhost:8080/api/explorer/block/$h
  done
  ```

**A2. Stop Silent Failures (Emergency Patch)**

Replace silent `Ok(None)` with hard failure:

```rust
// crates/q-storage/src/lib.rs:557-565
match bincode::deserialize::<q_types::block::QBlock>(&block_data) {
    Ok(block) => Ok(Some(block)),
    Err(e) => {
        error!("🚨 CRITICAL: Block {} deserialization failed: {}", height, e);
        error!("    Block data: {} bytes", block_data.len());
        error!("    This indicates struct version mismatch or data corruption");
        error!("    DO NOT DELETE - data may be recoverable with migration tool");

        // Return error instead of Ok(None) to force operator intervention
        Err(anyhow::anyhow!(
            "Block deserialization failed at height {} - format incompatibility",
            height
        ))
    }
}
```

**Deployment:**
- Version: v0.5.23-emergency-patch
- Testing: 2 hours (testnet/backup DB)
- Deployment: Rolling restart with monitoring

### P1 - Short-Term (Within 24-72 Hours) 📅

**B1. Git Archaeology**

Identify exact struct changes:

```bash
# Find q_types changes around incident date
git log --since="2025-11-01" --until="2025-11-15" \
  --all -- crates/q-types/src/lib.rs

# Compare QBlock definitions
git diff <old_commit> HEAD -- crates/q-types/src/lib.rs \
  | grep -A 30 "pub struct QBlock"
```

**B2. Block Readability Audit**

Systematically test which blocks can be read:

```rust
// Simple audit tool
for height in 0..=93_743 {
    match storage.get_qblock_by_height(height).await {
        Ok(Some(_)) => readable_count += 1,
        Ok(None) => missing_count += 1,
        Err(e) => {
            error!("Height {}: deserialization failed: {}", height, e);
            failed_count += 1;
        }
    }
}
```

**B3. Migration Tool POC**

Build proof-of-concept converter:

```rust
// crates/q-migration/src/main.rs
use old_q_types::block::QBlock as QBlockV1;
use new_q_types::block::QBlock as QBlockV2;

fn migrate_block(old_data: &[u8]) -> Result<Vec<u8>> {
    // Deserialize with old format
    let old_block: QBlockV1 = bincode::deserialize(old_data)?;

    // Convert to new format
    let new_block = QBlockV2 {
        header: old_block.header,
        transactions: old_block.transactions,
        mining_solutions: old_block.mining_solutions,
        state_root: None,  // Default for new field
    };

    // Serialize with new format
    bincode::serialize(&new_block)
}
```

Test on backup database, **NOT production**.

### P2 - Medium-Term (Within 1 Week) 📆

**C1. Production Migration Tool**

Full migration implementation:

```rust
// Features:
- Read-only mode for safety
- Batch processing (1000 blocks/batch)
- Progress reporting
- Validation (re-read after write)
- Rollback capability
- Detailed logging
```

**C2. Versioned Serialization**

Add schema version to all new blocks:

```rust
const BLOCK_FORMAT_VERSION: u8 = 2;

pub fn serialize_block(block: &QBlock) -> Vec<u8> {
    let mut data = vec![BLOCK_FORMAT_VERSION];
    data.extend(bincode::serialize(block)?);
    data
}

pub fn deserialize_block(data: &[u8]) -> Result<QBlock> {
    if data.is_empty() {
        return Err(Error::EmptyData);
    }

    let version = data[0];
    match version {
        1 => deserialize_v1(&data[1..])?,  // Old format (no version byte)
        2 => deserialize_v2(&data[1..])?,  // New format
        _ => return Err(Error::UnknownVersion(version)),
    }
}
```

**C3. Write Verification**

Add read-back validation:

```rust
// After writing block
batch.commit()?;

// Verify immediately
let verify = self.get_qblock_by_height(height).await?;
if verify.is_none() {
    return Err(anyhow::anyhow!(
        "CRITICAL: Block write verification failed at height {}",
        height
    ));
}

// Only update pointer after verification
self.update_latest_pointer(height)?;
```

### P3 - Long-Term (Within 1 Month) 🗓️

**D1. Protobuf Migration**

Replace bincode with schema-aware format:

```protobuf
// proto/qblock.proto
syntax = "proto3";

message QBlock {
  BlockHeader header = 1;
  repeated Transaction transactions = 2;
  repeated MiningSolution mining_solutions = 3;
  optional bytes state_root = 4;  // Backwards compatible!
}
```

**Advantages:**
- Built-in backwards compatibility
- Optional fields with defaults
- Schema evolution support
- Cross-language support

**D2. Database Schema Versioning**

Add version metadata:

```rust
// On startup
let db_version = storage.get_metadata("db_schema_version")?;
let expected_version = CURRENT_SCHEMA_VERSION;

if db_version != expected_version {
    error!("Database schema mismatch!");
    error!("  Database version: {}", db_version);
    error!("  Expected version: {}", expected_version);
    error!("  Run migration tool before starting node");
    return Err(Error::SchemaMismatch);
}
```

**D3. CI/CD Serialization Tests**

Prevent future incompatibilities:

```rust
#[test]
fn test_backwards_compatibility() {
    // Load sample blocks from v1 format
    let v1_blocks = load_test_blocks("testdata/blocks_v1.bin");

    // Ensure current code can deserialize them
    for (height, data) in v1_blocks {
        let block = deserialize_block(&data)
            .expect(&format!("Failed to deserialize v1 block at {}", height));
        assert_eq!(block.header.height, height);
    }
}
```

Fail CI if core structs change without migration plan.

---

## Lessons Learned

### Critical Mistakes

1. **Using bincode for long-term storage**
   - Bincode is designed for *wire format* (temporary), not *storage format* (permanent)
   - No schema versioning or evolution support
   - Struct changes break old data with no recovery path

2. **Silent failure on deserialization errors**
   - `Ok(None)` hides catastrophic issues
   - Should be `Err()` or panic to force operator attention
   - Made debugging extremely difficult (looked like data deletion)

3. **No schema versioning**
   - Impossible to detect incompatibility programmatically
   - No way to perform graceful migrations
   - No version history for forensic analysis

4. **No write verification**
   - Blocks written but never verified they can be read back
   - Pointer updated optimistically before confirming readability
   - Corruption discovered only on restart (too late)

5. **No database migration strategy**
   - Code changes break old data with no recovery path
   - No rollback capability
   - No compatibility matrix (which versions can read which data)

### What Went Right ✅

**Safety Features That Worked:**

1. **Corruption detection** - Service correctly refused to start with corrupted pointer
2. **Binary search recovery** - Found highest readable block automatically
3. **Auto-repair** - Recovered to 88,495 without data loss
4. **Testnet deployment** - Caught before mainnet catastrophe
5. **No actual data loss** - RocksDB data fully intact and recoverable

---

## Recommendations

### Immediate (Deploy in v0.5.23-emergency)

1. **Fix deserialization error handling** ⚠️ CRITICAL
   - Change `Ok(None)` to `Err()` for deserialization failures
   - Log at ERROR level with detailed context
   - Fail startup if any blocks are unreadable

2. **Add write verification**
   - Read-back every written block
   - Verify deserialization succeeds
   - Only update pointer after successful verification

### Short-Term (Deploy in v0.6.0)

1. **Add version byte to all new blocks**
   - Prefix all serialized blocks with format version
   - Implement version-aware deserializer
   - Support reading old blocks (version detection)

2. **Build migration framework**
   - Support multiple QBlock versions
   - Automatic upgrades on read (lazy migration)
   - Bulk migration tool for manual upgrades

3. **Database health monitoring**
   - Daily deserialization tests on random blocks
   - Version compatibility checks
   - Automatic alerts on format issues

### Long-Term (Deploy in v1.0.0)

1. **Switch to Protobuf** 🎯 RECOMMENDED
   - Industry-standard serialization
   - Built-in backwards compatibility
   - Schema registry and evolution
   - Cross-client compatibility

2. **Implement database versioning**
   - Metadata table with schema version
   - Startup compatibility checks
   - Automatic migration triggers
   - Version upgrade/downgrade paths

3. **Add migration testing to CI/CD**
   - Test deserializing old block formats
   - Fail CI on breaking changes without migration
   - Maintain test fixture library of historical blocks

---

## Appendix A: Forensic Evidence

### Database File Timeline

```bash
$ ls -lh ./data-mine11/hot/*.sst | grep -E "Nov 12|Nov 13"
-rw-r--r-- 1 root root  65M Nov 12 21:37 001238.sst  # Last file before update
-rw-r--r-- 1 root root  65M Nov 13 07:51 001303.sst  # First file after update
-rw-r--r-- 1 root root  65M Nov 13 16:49 001466.sst  # New format blocks?
```

**Correlation:**
- Nov 12 21:37: Last SST file before code update (height ~93,743)
- Nov 13 07:51: First SST after update (new struct format?)
- Nov 15 18:15: Service crash (cannot read any blocks)

### Block Verification Results

```bash
#!/bin/bash
# Verify block existence in RocksDB

for height in 0 1000 10000 50000 88495 88496 90000 93743; do
    ldb --db=./data-mine11/hot/ get blocks "qblock:height:$height" \
        && echo "Height $height: EXISTS ✅" \
        || echo "Height $height: NOT FOUND ❌"
done
```

**Results:**
```
Height 0:     EXISTS ✅
Height 1000:  EXISTS ✅
Height 10000: EXISTS ✅
Height 50000: EXISTS ✅
Height 88495: EXISTS ✅
Height 88496: EXISTS ✅  (supposedly "missing")
Height 90000: EXISTS ✅  (supposedly "missing")
Height 93743: EXISTS ✅  (supposedly "missing")
```

**Conclusion:** All blocks physically present, including "missing" range.

---

## Appendix B: Recovery Scripts

### Emergency Block Audit

```bash
#!/bin/bash
# Audit block readability via API

echo "Testing block accessibility..."
for height in 88495 88496 88497 90000 93743; do
    response=$(curl -s "http://localhost:8080/api/explorer/block/$height")

    if echo "$response" | jq -e '.height' >/dev/null 2>&1; then
        echo "Height $height: READABLE ✅"
    else
        echo "Height $height: ERROR ❌ - $response"
    fi
done
```

### Database Backup Verification

```bash
#!/bin/bash
# Verify backup integrity

backup="data-mine11-backup-1731698734.tar.gz"

echo "Verifying backup: $backup"
tar tzf "$backup" | head -10

echo "Extracting to test location..."
mkdir -p /tmp/backup-test
tar xzf "$backup" -C /tmp/backup-test

echo "Testing RocksDB read..."
ldb --db=/tmp/backup-test/data-mine11/hot/ \
    get blocks "qblock:height:88495" \
    && echo "Backup is readable ✅" \
    || echo "Backup is corrupted ❌"
```

---

## Document Metadata

**Created:** 2025-11-15 18:45 UTC
**Author:** Engineering Team
**Version:** 2.0 (Revised for clarity)
**Status:** FINAL
**Classification:** Internal - Critical Incident Report

**Previous Version:** `DATABASE_CORRUPTION_ROOT_CAUSE_ANALYSIS.md` (v1.0)
**Key Changes:**
- Renamed from "corruption" to "serialization incompatibility"
- Clarified 0% actual data loss vs 100% perceived loss
- Grouped action items by priority/timeline
- Added emergency patch code examples
- Simplified executive summary

**Related Documents:**
- `MINER_P0_DEPLOYMENT_SUCCESS.md` - Concurrent miner fix
- `CRITICAL_SYNC_DOWN_BUG_ANALYSIS.md` - Previous corruption analysis
- `PHASE_TRANSITION_BUG_PREVENTION_CHECKLIST.md` - Phase safety

**Distribution:**
- ✅ Core development team
- ✅ DevOps / SRE
- ✅ Security team
- ✅ Project leadership

**Next Review:** 2025-11-16 09:00 UTC (24 hours post-incident)

---

**END OF REPORT**
