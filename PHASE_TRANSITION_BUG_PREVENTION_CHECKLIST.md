# Phase Transition Bug Prevention Checklist

**Created**: 2025-11-10 after Phase 8 network isolation bug (FOUR BUGS FOUND!)
**Updated**: 2025-11-10 after discovering Phase 9 hardcoded Phase 7 fallbacks (BUG #5!)
**Updated**: 2025-12-07 after Phase 14 encryption keys issue (confirmed encryption keys step)
**Updated**: 2025-12-08 after Phase 14 chain fork discovery (BUG #6 - genesis checkpoint requirement!)
**Updated**: 2025-12-10 after Phase 15 P2P balance propagation failure (BUG #7 - handlers.rs network ID defaults!)
**Updated**: 2025-12-12 after Phase 16 P2P sync failure (BUG #8 - main.rs hardcoded NetworkId ignores Q_NETWORK_ID!)
**Updated**: 2025-12-12 after Phase 16 additional Docker sync failure (BUG #9 - main.rs .unwrap_or(TestnetPhase15) fallbacks!)
**Updated**: 2026-02-14 added browser js-libp2p config (Checklist #21 - frontend P2P gossipsub topics!)
**Updated**: 2026-02-14 Phase 20 transition revealed FOUR MORE BUGS (#10-13): GENESIS_TIMESTAMP locations, LEGACY_FIXED_REWARD decimal mismatch, emission tracking never wired, bootstrap peer IDs
**Updated**: 2026-02-14 Phase 21 transition revealed THREE MORE BUGS (#15-17): encryption key format mismatch, frontend not rebuilt after transition, frontend rebuild must happen BEFORE users access site
**Updated**: 2026-02-14 Phase 22 transition revealed THREE MORE BUGS (#18-20): systemd service file not updated (stale data dir), drop-in overrides conflict, running process environment not verified after restart, frontend localStorage balance cache survives phase transition
**Updated**: 2026-02-15 Phase 24 transition added 3-server cleanup steps (#33-36): Alpha Docker cleanup, Alpha env file creation, Gamma/Beta service file updates, encryption key removal on ALL servers
**Updated**: 2026-02-18 Mainnet 2026.1.1 rehearsal revealed SIX MORE BUGS (#22-27): LZ4 compress/decompress mismatch, lockfree_producer error propagation, save_qblock timestamp filter, systemd deployment race, staggered start chain forks, binary version mismatch
**Updated**: 2026-02-22 Mainnet 2026.1.3 user connectivity failure revealed BUG #28: ALL hardcoded bootstrap peer IDs were stale after identity regeneration. External nodes could not establish stable P2P connections.
**Purpose**: Prevent recurrence of NetworkId parsing bugs AND chain forks during phase transitions
**References**:
- `PHASE_8_NETWORK_ISOLATION_BUG.md` - Bugs #1 & #2 analysis
- `PHASE_8_THREE_BUGS_IDENTIFIED.md` - Analysis of Bugs #1, #2, #3
- `PHASE_8_FOUR_BUGS_FINAL.md` - Complete analysis of all 4 bugs
- `PHASE_9_HARDCODED_PHASE7_BUG_FIX.md` - Bug #5 analysis
- `crates/q-storage/src/genesis_checkpoint.rs` - Bug #6 fix: Genesis checkpoint validation module

## 🔥 CRITICAL LESSONS FROM PHASE 8, PHASE 9 & PHASE 14

Phase 8 revealed **FOUR SEPARATE, CASCADING BUGS** that ALL had to be fixed.
Phase 9 revealed **BUG #5**: Hardcoded fallback values throughout main.rs.
Phase 14 revealed **BUG #6**: Nodes that fail sync create independent chains that CANNOT merge!

### Bug #1: Missing from_str() Parser Case
- Symptom: Q_NETWORK_ID environment variable couldn't parse "testnet-phase8"
- Impact: Fell back to wrong default phase
- **Lesson**: ALWAYS update from_str() when adding enum variants
- **Location**: `crates/q-types/src/lib.rs:795-804`

### Bug #2: Environment Variable Ignored
- Symptom: Code prioritized CLI `--network` arg over Q_NETWORK_ID env var
- Impact: Systemd services with Q_NETWORK_ID were COMPLETELY IGNORED
- **Lesson**: Environment variables MUST be checked BEFORE command line arguments
- **Location**: `crates/q-api-server/src/main.rs:486`

### Bug #3: NetworkConfig::testnet() Hard-Coded to Phase 6
- Symptom: Even with Bugs #1 & #2 fixed, still showed Phase 6
- Impact: NetworkConfig returned wrong phase despite correct parsing
- **Lesson**: Update NetworkConfig::testnet() network_id field when transitioning
- **Location**: `crates/q-types/src/lib.rs:846`

### Bug #4: Block Producer Hard-Coded to Phase 7
- Symptom: Node subscribed to Phase 8 topics but published Phase 7 blocks
- Impact: Even with Bugs #1-3 fixed, created blocks with wrong network_id
- **Lesson**: Update block creation code when transitioning phases
- **Location**: `crates/q-api-server/src/block_producer.rs:306-307`

### Bug #5: Hardcoded Fallback Values in main.rs (Phase 9 Discovery!)
- Symptom: Despite fixing Bugs #1-4, Phase 9 nodes still published to Phase 7 topics
- Impact: 12+ fallback `.unwrap_or(TestnetPhase7)` values never updated
- **Lesson**: Global search & replace ALL fallback values during phase transition
- **Location**: `crates/q-api-server/src/main.rs` (15 occurrences fixed)
- **Additional Issue**: Wrong env var name (`Q_NETWORK` vs `Q_NETWORK_ID`)

**ALL FIVE bugs had to be fixed for Phase 9 to work!**

## 🚨 Mandatory Checklist for Adding New NetworkId Phases

When adding a new phase (e.g., TestnetPhase9, TestnetPhase10), you MUST complete ALL items:

### 1. Enum Declaration (`crates/q-types/src/lib.rs`)
```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkId {
    // ... existing phases

    /// Phase X: [Description]
    #[serde(rename = "testnet-phaseX")]
    TestnetPhaseX,  // ✅ ADD THIS
}
```

### 2. `as_str()` Method
```rust
pub fn as_str(&self) -> &'static str {
    match self {
        // ... existing phases
        NetworkId::TestnetPhaseX => "testnet-phaseX",  // ✅ ADD THIS
        NetworkId::Mainnet => "mainnet",
    }
}
```

### 3. `display_name()` Method
```rust
pub fn display_name(&self) -> &'static str {
    match self {
        // ... existing phases
        NetworkId::TestnetPhaseX => "Q-NarwhalKnight Testnet Phase X ([Description])",  // ✅ ADD THIS
        NetworkId::Mainnet => "Q-NarwhalKnight Mainnet",
    }
}
```

### 4. ⚠️ **CRITICAL** `from_str()` Parser - **THIS IS THE BUG THAT WAS MISSED!**
```rust
impl std::str::FromStr for NetworkId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            // ... existing phases
            "testnet-phaseX" => Ok(NetworkId::TestnetPhaseX),  // ✅ ADD THIS
            "mainnet" => Ok(NetworkId::Mainnet),
            _ => Err(format!("Invalid network ID: {}", s)),
        }
    }
}
```

**WHY THIS IS CRITICAL**: Without this, Q_NETWORK_ID="testnet-phaseX" environment variable will fail to parse and fall back to default, causing WRONG NETWORK ID and complete network isolation!

### 5. Update `Default` Implementation (if transitioning)
```rust
impl Default for NetworkId {
    fn default() -> Self {
        NetworkId::TestnetPhaseX  // ✅ UPDATE THIS to latest phase
    }
}
```

### 6. ⚠️ **CRITICAL** `NetworkConfig::testnet()` network_id Field - **THIS WAS BUG #3!**
```rust
pub fn testnet() -> Self {
    Self {
        // ✅ UPDATE THIS to latest phase!
        network_id: NetworkId::TestnetPhaseX,  // NOT Phase 5, 6, 7, etc!
        genesis_hash: [...],
        // ... rest of config
    }
}
```

**WHY THIS IS CRITICAL**: Even if from_str() and environment variables work correctly, if testnet() returns a config with the wrong network_id, the node will display the wrong phase! This was Bug #3 - the hardest to find because the code flow LOOKED correct but had a hard-coded value buried in the config constructor.

### 7. `default_api_port()` Method
```rust
pub fn default_api_port(&self) -> u16 {
    match self {
        // ... existing phases
        NetworkId::TestnetPhaseX => 8080,  // ✅ ADD THIS
        NetworkId::Mainnet => 8080,
    }
}
```

### 7. `default_p2p_port()` Method
```rust
pub fn default_p2p_port(&self) -> u16 {
    match self {
        // ... existing phases
        NetworkId::TestnetPhaseX => 9001,  // ✅ ADD THIS
        NetworkId::Mainnet => 9001,
    }
}
```

### 8. `NetworkConfig::from_network_id()` Method
```rust
pub fn from_network_id(network_id: NetworkId) -> Self {
    match network_id {
        // ... existing phases
        NetworkId::TestnetPhaseX => Self::testnet(),  // ✅ ADD THIS
        NetworkId::Mainnet => Self::mainnet(),
    }
}
```

### 9. ⚠️ **CRITICAL** Block Producer Phase and Network ID - **THIS WAS BUG #4!**

**Location**: `crates/q-api-server/src/block_producer.rs` (search for "Create block")

```rust
// In the produce_block() method:
let block = QBlock {
    header: BlockHeader {
        height: self.current_height + 1,
        phase: X,  // ✅ UPDATE THIS to new phase number!
        network_id: "testnet-phaseX".to_string(),  // ✅ UPDATE THIS to new phase!
        prev_block_hash: self.latest_block_hash,
        // ... rest of header
    },
    // ... rest of block
};
```

**WHY THIS IS CRITICAL**: This is where blocks are CREATED. Even if all config, environment variables, and NetworkConfig are correct, if the block producer creates blocks with the wrong phase/network_id, you'll get the dreaded "subscribe to phase8, publish to phase7" mismatch! Blocks carry their own network_id field, and gossipsub publications use the block's embedded network_id, NOT the config's network_id.

**Bug #4 Symptom**:
- ✅ Node displays correct phase
- ✅ Subscribes to correct topics
- ❌ Publishes to WRONG topics (old phase)
- 🔴 Result: Complete network isolation

### 10. ⚠️ **CRITICAL** Environment Variable Priority (`crates/q-api-server/src/main.rs`)

**THIS WAS THE PHASE 8 BUG #2!** Environment variables MUST be checked BEFORE CLI arguments!

```rust
// ❌ WRONG (Phase 8 bug):
let network_str = matches.get_one::<String>("network")
    .map(|s| s.as_str())
    .unwrap_or("testnet");  // Ignores Q_NETWORK_ID completely!

// ✅ CORRECT:
let network_str = std::env::var("Q_NETWORK_ID")  // Check env var FIRST
    .ok()
    .or_else(|| matches.get_one::<String>("network").map(|s| s.to_string()))
    .unwrap_or_else(|| "testnet-phaseX".to_string());  // Default to NEW phase

let network_id = network_str.parse::<q_types::NetworkId>()
    .unwrap_or_else(|e| {
        warn!("Invalid network '{}': {}. Defaulting to Phase X.", network_str, e);
        q_types::NetworkId::TestnetPhaseX  // ✅ Fallback to NEW phase
    });
```

**WHY THIS IS CRITICAL**:
- Systemd services use environment variables (Q_NETWORK_ID), not CLI args
- If CLI args are checked first, Q_NETWORK_ID is COMPLETELY IGNORED
- Result: Service runs wrong phase even though Q_NETWORK_ID is set correctly

### 11. ⚠️ **CRITICAL** Global Search & Replace ALL Fallback Values (`crates/q-api-server/src/main.rs`)

**THIS WAS THE PHASE 9 BUG #5!** ALL `.unwrap_or()` fallback values must be updated!

Search for ALL instances of:
```bash
grep -n "unwrap_or(q_types::NetworkId::TestnetPhase" crates/q-api-server/src/main.rs
```

You'll find 10+ occurrences like:
```rust
// ❌ WRONG - Hardcoded old phase fallback
let network_id = std::env::var("Q_NETWORK_ID")
    .ok()
    .and_then(|s| s.parse::<q_types::NetworkId>().ok())
    .unwrap_or(q_types::NetworkId::TestnetPhase7);  // ← OLD PHASE!

// ✅ CORRECT - Updated to new phase
let network_id = std::env::var("Q_NETWORK_ID")
    .ok()
    .and_then(|s| s.parse::<q_types::NetworkId>().ok())
    .unwrap_or(q_types::NetworkId::TestnetPhaseX);  // ← NEW PHASE!
```

**ALSO CHECK**: Environment variable name consistency
```rust
// ❌ WRONG - Using Q_NETWORK instead of Q_NETWORK_ID
std::env::var("Q_NETWORK")

// ✅ CORRECT - Canonical name
std::env::var("Q_NETWORK_ID")
```

**WHY THIS IS CRITICAL**:
- These fallbacks trigger when environment variables aren't available
- Even with correct systemd config, some code paths may not have Q_NETWORK_ID
- Result: Node publishes to WRONG phase topics, causing network isolation
- Symptoms: "InsufficientPeers" errors despite being on correct phase

**Locations to Check** (Phase 9 had 15 fixes):
- Block broadcasting (lines ~4244, 4757, 4876)
- Turbo sync requests (lines ~3296)
- Block pack responses (lines ~3083)
- Peer height announcements (lines ~3666)
- Gap fill requests (lines ~2854)
- Batch block responses (lines ~2538)
- Block validation (lines ~2651)
- And more...

## 🧪 Testing Requirements

### Unit Tests (Add to `crates/q-types/src/lib.rs`)
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_id_parsing() {
        // Test ALL phases can be parsed
        assert_eq!("testnet-phase5".parse::<NetworkId>().unwrap(), NetworkId::TestnetPhase5);
        assert_eq!("testnet-phase6".parse::<NetworkId>().unwrap(), NetworkId::TestnetPhase6);
        assert_eq!("testnet-phase7".parse::<NetworkId>().unwrap(), NetworkId::TestnetPhase7);
        assert_eq!("testnet-phase8".parse::<NetworkId>().unwrap(), NetworkId::TestnetPhase8);
        assert_eq!("testnet-phaseX".parse::<NetworkId>().unwrap(), NetworkId::TestnetPhaseX);  // ✅ ADD THIS
        assert_eq!("mainnet".parse::<NetworkId>().unwrap(), NetworkId::Mainnet);
    }

    #[test]
    fn test_phase_x_gossipsub_topics() {
        let phase = NetworkId::TestnetPhaseX;
        assert_eq!(phase.as_str(), "testnet-phaseX");
        assert!(phase.gossipsub_topic_prefix().contains("phaseX"));
        assert!(phase.blocks_topic().contains("phaseX"));
        assert!(phase.transactions_topic().contains("phaseX"));
    }
}
```

### Integration Test
```bash
# Test environment variable parsing
export Q_NETWORK_ID="testnet-phaseX"
./target/release/q-api-server --version
# Should output: "Network: Q-NarwhalKnight Testnet Phase X"
```

## 🚀 Deployment Verification Checklist

After deploying new phase, ALWAYS verify:

### 1. Environment Variable
```bash
sudo systemctl show q-api-server | grep Q_NETWORK_ID
# Expected: Environment=Q_NETWORK_ID=testnet-phaseX
```

### 2. Startup Logs - Network Name
```bash
journalctl -u q-api-server --since "30 seconds ago" | grep "Network:"
# Expected: "Network: Q-NarwhalKnight Testnet Phase X"
# NOT: Phase 5, 6, 7, or 8 (unless that's what you want)
```

### 3. Gossipsub Subscribe Topics
```bash
journalctl -u q-api-server --since "30 seconds ago" | grep "Subscribed to testnet"
# Expected: /qnk/testnet-phaseX/blocks
# NOT: Different phase number!
```

### 4. Gossipsub Publish Topics
```bash
journalctl -u q-api-server --since "1 minute ago" | grep "Publishing.*gossipsub"
# Expected: /qnk/testnet-phaseX/blocks
# NOT: Different phase number!
```

### 5. Verify No Publication Failures
```bash
journalctl -u q-api-server --since "1 minute ago" | grep "Failed to publish"
# If you see "InsufficientPeers", topics are WRONG!
```

### 6. Verify Network Sync
```bash
curl http://localhost:8080/stats
# Check: "height" should increase from network blocks
# Check: "sync_status" should NOT be "Solo mining"
```

## ⚠️ Common Mistakes to Avoid

1. **🔥 Block producer creates wrong phase blocks** ← **THIS WAS BUG #4 - HARDEST TO FIND!**
   - Symptom: Subscribe to phaseX, publish to phaseY (topic mismatch)
   - Impact: 100% network isolation even with correct config/env vars
   - Fix: Update phase and network_id in block_producer.rs BlockHeader creation
   - Location: `crates/q-api-server/src/block_producer.rs` line ~306
   - **This bug is SILENT - config looks perfect but blocks are wrong!**

2. **🔥 CLI args checked before environment variables** ← **THIS WAS BUG #2!**
   - Symptom: Q_NETWORK_ID completely ignored, systemd services use wrong phase
   - Impact: 100% network isolation despite correct environment variables
   - Fix: Check std::env::var("Q_NETWORK_ID") BEFORE CLI arguments
   - Location: `crates/q-api-server/src/main.rs` line ~486

3. **Adding enum without updating from_str()** ← **THIS WAS BUG #1!**
   - Symptom: Q_NETWORK_ID can't parse new phase string
   - Impact: Falls back to default phase
   - Fix: Always update from_str() when adding enum variant
   - Location: `crates/q-types/src/lib.rs` line ~795

4. **NetworkConfig::testnet() hard-coded to old phase** ← **THIS WAS BUG #3!**
   - Symptom: Even with Bugs #1 & #2 fixed, still shows old phase
   - Impact: Config returns wrong network_id
   - Fix: Update NetworkConfig::testnet() network_id field
   - Location: `crates/q-types/src/lib.rs` line ~846

5. **Forgetting to update default()**
   - Symptom: Nodes without Q_NETWORK_ID use old phase
   - Fix: Update default() to latest phase during transition
   - Location: `crates/q-types/src/lib.rs` line ~807

6. **Testing with wrong environment variable**
   - Mistake: Testing with "testnet-phase7" when code expects "testnet-phase8"
   - Fix: Always verify environment variable matches new phase string

5. **Not verifying gossipsub topics in logs**
   - Mistake: Assuming topics are correct without checking
   - Fix: ALWAYS grep logs for topic subscriptions/publications

5. **Binary not rebuilt after code changes**
   - Mistake: Editing code but forgetting to rebuild
   - Fix: Always `cargo build --release` after NetworkId changes

## 📝 Git Commit Template

When adding new phase:

```
feat(vX.Y.Z-beta): Add NetworkId::TestnetPhaseX

Complete NetworkId implementation for Phase X:
- [X] Added TestnetPhaseX enum variant
- [X] Updated as_str() method
- [X] Updated display_name() method
- [X] Updated from_str() parser ← CRITICAL!
- [X] Updated default() to PhaseX
- [X] Updated default_api_port()
- [X] Updated default_p2p_port()
- [X] Updated NetworkConfig::from_network_id()
- [X] Added unit tests for parsing
- [X] Added integration test for gossipsub topics
- [X] Verified deployment with logs

Phase X Changes:
- [Description of economic/consensus changes]
- Block reward: X QUG
- Database: data-mineX
- Gossipsub topics: /qnk/testnet-phaseX/*

Testing:
✅ Unit tests pass
✅ Parsing "testnet-phaseX" succeeds
✅ Gossipsub topics contain "phaseX"
✅ Deployment verified with logs

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
```

## 🆘 Troubleshooting

### Node shows wrong phase in logs?
1. Check systemd service file: `cat /etc/systemd/system/q-api-server.service | grep Q_NETWORK_ID`
2. Verify from_str() has case for your phase: `grep "testnet-phaseX" crates/q-types/src/lib.rs`
3. Rebuild: `cargo build --release --package q-api-server`
4. Restart: `systemctl restart q-api-server`

### Topics mismatch (subscribe to one phase, publish to another)?
- **Root Cause**: from_str() not updated, causing fallback
- **Fix**: Add case to from_str() and rebuild

### Publications failing with "InsufficientPeers"?
- **Root Cause**: Publishing to wrong topic (no peers on that topic)
- **Fix**: Verify gossipsub topics in logs match Q_NETWORK_ID

### 12. ⚠️ **IMPORTANT** Systemd Service File (`/etc/systemd/system/q-api-server.service`)

**Update the service description and environment variables for the new phase:**

```ini
[Unit]
Description=Q-NarwhalKnight API Server - Phase X ([Phase Description])
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/orobit/shared/q-narwhalknight

# ✅ v0.9.XX-beta Phase X Configuration
Environment="Q_DB_PATH=./data-mineX"  # ✅ UPDATE THIS
Environment="Q_NETWORK_ID=testnet-phaseX"  # ✅ UPDATE THIS
Environment="Q_IS_VALIDATOR=true"
Environment="Q_P2P_PORT=9001"
Environment="Q_ENABLE_AI=1"
Environment="RUST_LOG=info"
```

**After updating, reload systemd:**
```bash
systemctl daemon-reload
systemctl restart q-api-server
```

### 13. ⚠️ **IMPORTANT** Phase Transition Modal (Frontend)

**Update the Phase Transition Modal to show for the new phase:**

**Location 1**: `gui/quantum-wallet/src/components/PhaseTransitionModal.tsx`

Update the localStorage key to a NEW unique name:
```typescript
useEffect(() => {
  localStorage.setItem('phaseXDescriptionModalSeen', 'true');  // ✅ UPDATE THIS to new unique key
}, []);
```

**Location 2**: `gui/quantum-wallet/src/components/Dashboard.tsx`

Update the modal visibility check:
```typescript
const [showPhaseModal, setShowPhaseModal] = useState(() => {
  const hasSeenPhaseX = localStorage.getItem('phaseXDescriptionModalSeen');  // ✅ UPDATE THIS
  return !hasSeenPhaseX;
});
```

**WHY THIS IS IMPORTANT**: Using a new localStorage key ensures ALL users see the phase transition announcement, even if they've dismissed previous phase modals. This is critical for communicating breaking changes and new features.

### Bug #18: Systemd Service File Not Updated (Phase 22 Discovery!)
- Symptom: After Phase 22 code deployed, server still used `data-mine21` and `testnet-phase21` topics
- Impact: Old smart contracts, balances, and 4.6M blocks from previous phases came back
- Root cause: `ha-deploy.sh` only copies the binary, does NOT update the service file
- HA deploy script aborted before reaching Beta upgrade step (Gamma soak failed)
- Manual deploy of Beta still used old Phase 21 service file
- **Lesson**: Service file update is a SEPARATE step from binary deployment
- **Location**: `/etc/systemd/system/q-api-server.service`
- **Required fields to update**: `Description`, `Q_DB_PATH`, `Q_NETWORK_ID`, `Q_ENCRYPTION_KEYS_FILE`, `Q_ENCRYPTION_PASSPHRASE`

### Bug #19: Systemd Drop-in Overrides Conflict (Phase 22 Discovery!)
- Symptom: Unexpected `Q_DISABLE_DEX=1` from previous phase persisted
- Impact: DEX disabled on new phase even though not intended
- Root cause: Drop-in files at `/etc/systemd/system/q-api-server.service.d/` survive service file edits
- **Lesson**: ALWAYS check and clean drop-in overrides during phase transition
- **Location**: `/etc/systemd/system/q-api-server.service.d/*.conf`

### Bug #20: Running Process Environment Not Verified (Phase 22 Discovery!)
- Symptom: `systemctl show` shows correct Phase 22 config, but `cat /proc/PID/environ` shows Phase 21
- Impact: Server appears to be on Phase 22 (service file is correct) but actually runs Phase 21
- Root cause: Process was started before `systemctl daemon-reload`, or kill/restart didn't fully cycle
- **Lesson**: ALWAYS verify `/proc/PID/environ` after restart, not just `systemctl show`
- **Verification command**: `tr '\0' '\n' < /proc/$(pgrep -f "q-api-server --port" | head -1)/environ | grep Q_NETWORK_ID`

### Bug #21: Frontend localStorage Balance Cache Survives Phase Transition (Phase 22 Discovery!)
- Symptom: QUGUSD balance from Phase 21 (79,702 QUGUSD) still shows after Phase 22 transition
- Impact: Users see stale balances from old phase, confusing and misleading
- Root cause: Dashboard.tsx caches `cachedBalance`, `cachedQugusdBalance`, `highestKnownBalances` in localStorage. These survive page refreshes and phase transitions.
- **Fix**: Added phase-aware cache clearing in Dashboard.tsx that checks `lastNetworkPhase` key
- **Lesson**: Any localStorage balance cache MUST be invalidated on phase change
- **Location**: `gui/quantum-wallet/src/components/Dashboard.tsx`

## 📚 Additional Resources

- **Full Bug Analysis**: `PHASE_8_NETWORK_ISOLATION_BUG.md`
- **Phase Transition Guide**: `PHASE_TRANSITION_AND_MAINNET_REHEARSAL_GUIDE.md`
- **Claude Development Guide**: `CLAUDE.md`

---

**Remember**: The from_str() parser is the most commonly forgotten step!
Always update it when adding new NetworkId variants.

---

### 14. ⚠️ **CRITICAL** Encryption Keys Reset for Fresh Database

**Discovered during Phase 13 transition (December 2025)**
**Confirmed during Phase 14 transition (December 2025)** - Same issue occurred, proving this step is mandatory!

When creating a fresh database directory (e.g., `data-mine13`), you MUST also:

1. **Remove old encryption keys file**:
```bash
rm -f /opt/encryption.keys
```

2. **Clean the database directory completely**:
```bash
rm -rf ./data-mineX/*
```

**WHY THIS IS CRITICAL**:
- The encryption system stores AES-GCM keys in `/opt/encryption.keys`
- If you change the passphrase in systemd (e.g., `Q_ENCRYPTION_PASSPHRASE=Qnk-Phase13-...`)
- But the old keys file still exists with keys derived from the OLD passphrase
- Result: "AES-GCM decryption failed (wrong passphrase?): aead::Error"
- The service will crash repeatedly with exit code 1

**Symptoms**:
```
Error: Failed to open hot database
Caused by:
    AES-GCM decryption failed (wrong passphrase?): aead::Error
```

**Fix**:
```bash
# Stop service
systemctl stop q-api-server

# Remove old encryption keys (CRITICAL!)
rm -f /opt/encryption.keys

# Clean fresh database directory
rm -rf ./data-mineX/*

# Restart service (will generate new keys)
systemctl start q-api-server
```

**Phase 13 Lesson**: Always treat phase transitions as requiring BOTH:
1. Fresh database directory
2. Fresh encryption keys

This ensures cryptographic isolation between phases and prevents passphrase mismatch errors.

---

## 🔥 BUG #6: Chain Fork from Sync Connectivity Failures (Phase 14 Discovery!)

**Discovered**: 2025-12-08 during Phase 14 testing
**Impact**: CATASTROPHIC - Network splits into irreconcilable chain forks
**Root Cause**: Nodes that cannot sync from bootstrap mine independent chains with different genesis blocks

### How Bug #6 Manifests:

1. **Bootstrap node starts first** - mines blocks at height 1, 2, 3...
2. **New node joins network** - discovers bootstrap peer via DHT
3. **Sync attempt fails** (various reasons: peer selection bug, timeout, etc.)
4. **Node starts solo mining** - creates NEW genesis block at height 1
5. **Different genesis hash** - now on completely separate chain
6. **Blocks cannot merge** - DAG-Knight requires common ancestry back to genesis

**Result**: Two (or more) nodes with:
- ✅ Same network_id (testnet-phase14)
- ✅ Same gossipsub topics
- ✅ Can discover each other
- ❌ **DIFFERENT genesis blocks** → permanent chain fork

### Why This is Different from Bugs #1-5:

Bugs #1-5 were **configuration errors** - wrong phase strings caused topic isolation.
Bug #6 is a **chain ancestry problem** - even with perfect configuration, nodes on different genesis chains CANNOT sync because they have incompatible block histories.

### Kaspa's Solution: Hardcoded Genesis Checkpoint

Kaspa prevents this by:
1. **Hardcoding the genesis block hash into the binary**
2. **Rejecting blocks that don't chain back to this hash**
3. **Peer handshake validation** - disconnect peers with wrong genesis

---

### 15. ⚠️ **CRITICAL** Genesis Block Checkpoint Hardcoding

**THIS PREVENTS BUG #6!** The genesis hash MUST be hardcoded before phase transition.

**Location**: `crates/q-types/src/lib.rs` - `NetworkConfig::testnet()`

```rust
pub fn testnet() -> Self {
    Self {
        network_id: NetworkId::TestnetPhaseX,  // ✅ Update phase
        genesis_hash: [
            // ✅ HARDCODE ACTUAL GENESIS HASH FOR PHASE X
            // Run compute_genesis_hash.rs to get this value from bootstrap
            0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11,
            0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
            0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11,
            0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
        ],
        chain_id: X,  // ✅ Update chain ID
        // ... rest of config
    }
}
```

**How to Compute Genesis Hash Before Transition**:

```bash
# 1. Start bootstrap node, mine at least 1 block
./target/release/q-api-server --port 8080

# 2. Get genesis block hash
curl -s http://localhost:8080/api/v1/blocks/1 | jq -r '.hash'

# 3. Or use the genesis_checkpoint helper
cargo run --package q-storage --bin compute_genesis_hash
```

**WHY THIS IS CRITICAL**:
- Without hardcoded genesis, any node can create its own genesis
- Nodes with different genesis blocks will NEVER merge
- The network becomes permanently fragmented
- **This is the #1 cause of irreconcilable chain forks**

### 16. ⚠️ **CRITICAL** Block Chain Validation During Sync

**Integrate genesis checkpoint validation into block sync handlers.**

**Location**: `crates/q-network/src/unified_network_manager.rs` - `handle_block_pack_response()`

```rust
// In handle_block_pack_response():
async fn handle_block_pack_response(&mut self, blocks: Vec<QBlock>, from_peer: PeerId) {
    // ✅ ADD THIS: Validate blocks chain back to our genesis
    let validator = GenesisCheckpointValidator::new(self.network_id);

    match validator.validate_block_chain(&blocks) {
        ChainValidationResult::Valid => {
            // Blocks are on our chain - proceed with storage
            self.store_blocks(blocks).await;
        }
        ChainValidationResult::GenesisHashMismatch { expected, found } => {
            // ❌ REJECT - Different chain!
            warn!("🚨 CHAIN FORK DETECTED! Peer {} is on a different chain", from_peer);
            warn!("   Expected genesis: {}", hex::encode(expected));
            warn!("   Peer's genesis:   {}", hex::encode(found));

            // Disconnect and ban peer
            self.ban_peer(from_peer, "genesis_mismatch");
            return;
        }
        ChainValidationResult::NeedMoreBlocks { lowest_height_needed } => {
            // Request earlier blocks to verify chain ancestry
            self.request_blocks_from_height(lowest_height_needed, from_peer).await;
        }
        _ => {
            // Other validation errors - log and skip
            warn!("Block validation failed: {:?}", result);
        }
    }
}
```

**WHY THIS IS CRITICAL**:
- Prevents accepting blocks from forked chains
- Detects chain forks BEFORE storing invalid blocks
- Automatically bans peers on wrong chains
- Keeps the network unified on the canonical chain

### 17. ⚠️ **CRITICAL** Peer Genesis Hash Verification

**Add genesis hash to peer handshake and validate on connection.**

**Location**: `crates/q-network/src/unified_network_manager.rs` - `handle_peer_connected()`

```rust
// In peer handshake:
pub struct HandshakePayload {
    pub network_id: NetworkId,
    pub genesis_hash: [u8; 32],  // ✅ ADD THIS
    pub peer_id: PeerId,
    pub height: u64,
    // ... other fields
}

// During connection:
async fn handle_peer_connected(&mut self, peer: PeerId, handshake: HandshakePayload) {
    // ✅ ADD THIS: Verify genesis hash BEFORE accepting peer
    let our_genesis = self.config.genesis_hash;

    if handshake.genesis_hash != our_genesis {
        warn!("🚨 PEER ON WRONG CHAIN! Disconnecting {}", peer);
        warn!("   Our genesis:    {}", hex::encode(our_genesis));
        warn!("   Their genesis:  {}", hex::encode(handshake.genesis_hash));

        self.disconnect_peer(peer).await;
        return;
    }

    // Genesis matches - accept peer
    info!("✅ Peer {} verified on same genesis chain", peer);
    self.accept_peer(peer, handshake).await;
}
```

**WHY THIS IS CRITICAL**:
- Rejects peers on different chains IMMEDIATELY during handshake
- Prevents wasting bandwidth on incompatible block exchanges
- Provides clear diagnostic logging for chain fork detection
- Essential for network partition recovery scenarios

---

## 🆕 Phase X Fork Prevention Deployment Checklist

Before transitioning to ANY new phase, complete these additional items:

### Fork Prevention Pre-Flight Checklist:

- [ ] **Bootstrap node has produced blocks** - Mine at least 10 blocks on bootstrap
- [ ] **Genesis hash computed** - Extract hash from block at height 1
- [ ] **Genesis hash hardcoded** - Update `NetworkConfig::testnet().genesis_hash`
- [ ] **Genesis checkpoint module enabled** - Verify `genesis_checkpoint.rs` is compiled
- [ ] **Block sync validates genesis** - Check `handle_block_pack_response()` has validation
- [ ] **Peer handshake validates genesis** - Check handshake includes genesis_hash field
- [ ] **Binary rebuilt** - `cargo build --release --package q-api-server`
- [ ] **Binary deployed to all nodes** - Copy to `/downloads/q-api-server-vX.Y.Z-beta`
- [ ] **Test fresh node sync** - Start new node and verify it syncs to bootstrap chain
- [ ] **Test malicious genesis rejection** - Manually test that wrong genesis is rejected

### Computing Genesis Hash for Phase X:

```bash
# Method 1: From running bootstrap node
GENESIS_HASH=$(curl -s http://localhost:8080/api/v1/blocks/1 | jq -r '.hash')
echo "Genesis hash for Phase X: $GENESIS_HASH"

# Method 2: Using genesis_checkpoint helper
cargo run --package q-storage --example compute_genesis_hash -- --db-path ./data-mineX

# Method 3: Manual computation from block bytes
# See crates/q-storage/src/genesis_checkpoint.rs for implementation
```

### Formatting Genesis Hash for Code:

```bash
# Convert hex string to Rust array format
HASH="aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"
echo "[$( echo $HASH | sed 's/\(..\)/0x\1, /g' | sed 's/, $//' )]"
# Output: [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99]
```

---

## 🔗 Genesis Checkpoint Module Reference

**Module Location**: `crates/q-storage/src/genesis_checkpoint.rs`

**Key Components**:
- `GenesisCheckpointValidator` - Validates blocks chain back to genesis
- `ChainValidationResult` - Enum for validation outcomes
- `compute_genesis_hash_from_block()` - Helper to compute hash from block
- `format_hash_as_rust_array()` - Format hash for code embedding

**Integration Points**:
1. `unified_network_manager.rs` - Block sync validation
2. `turbo_sync.rs` - Batch sync validation
3. `kv.rs` - Database block storage validation
4. Peer handshake protocol - Connection-time validation

**Usage Example**:
```rust
use q_storage::genesis_checkpoint::{GenesisCheckpointValidator, ChainValidationResult};

let validator = GenesisCheckpointValidator::new(NetworkId::TestnetPhase15);

// Validate incoming blocks
match validator.validate_block_chain(&received_blocks) {
    ChainValidationResult::Valid => { /* Accept blocks */ }
    ChainValidationResult::GenesisHashMismatch { .. } => { /* Ban peer, reject blocks */ }
    _ => { /* Handle other cases */ }
}
```

---

## 🔥 BUG #8: main.rs Hardcoded NetworkId Values Ignore Q_NETWORK_ID (Phase 16 Discovery!)

**Discovered**: 2025-12-12 during Phase 16 Docker sync testing
**Impact**: CRITICAL - P2P sync completely fails between nodes on same network
**Root Cause**: 7+ locations in main.rs use `NetworkId::TestnetPhase15` hardcoded instead of reading Q_NETWORK_ID

### How Bug #8 Manifests:

1. **Bootstrap node starts** on Phase 16 (Q_NETWORK_ID=testnet-phase16)
2. **Docker node connects** to bootstrap via libp2p DHT
3. **Sync requests sent** but FAIL with "outbound failure" and "Timeout"
4. **Log analysis reveals**: Both nodes publishing to `/qnk/testnet-phase15/` topics despite Q_NETWORK_ID=testnet-phase16!
5. **Root cause**: main.rs has hardcoded `NetworkId::TestnetPhase15` values that ignore environment variable

**Result**: Two nodes configured for Phase 16 but communicating on Phase 15 topics!

### The 7 Bug Locations Found in main.rs:

```bash
# Search command to find all hardcoded network IDs:
grep -n "NetworkId::TestnetPhase" crates/q-api-server/src/main.rs | grep -v "Q_NETWORK_ID"
```

**Location 1** - Line ~2182 (peer heights announcement):
```rust
// ❌ WRONG - Hardcoded Phase15 ignores Q_NETWORK_ID
let network_id = q_types::NetworkId::TestnetPhase15;

// ✅ CORRECT - Read from environment variable
let network_id = std::env::var("Q_NETWORK_ID")
    .ok()
    .and_then(|s| s.parse::<q_types::NetworkId>().ok())
    .unwrap_or(q_types::NetworkId::TestnetPhase16); // Default to current phase
```

**Location 2** - Line ~4701 (gossip handling):
```rust
// ❌ WRONG
let network_id = q_types::NetworkId::TestnetPhase15;

// ✅ CORRECT
let network_id = std::env::var("Q_NETWORK_ID")
    .ok()
    .and_then(|s| s.parse::<q_types::NetworkId>().ok())
    .unwrap_or(q_types::NetworkId::TestnetPhase16);
```

**Location 3** - Line ~4833 (block validation):
```rust
// ❌ WRONG
let expected_network_id = q_types::NetworkId::TestnetPhase15.as_str();

// ✅ CORRECT
let expected_network_id = std::env::var("Q_NETWORK_ID")
    .ok()
    .and_then(|s| s.parse::<q_types::NetworkId>().ok())
    .unwrap_or(q_types::NetworkId::TestnetPhase16)
    .as_str();
```

**Location 4** - Line ~5098 (gap request):
```rust
// ❌ WRONG
let network_id = q_types::NetworkId::TestnetPhase15;

// ✅ CORRECT
let network_id = std::env::var("Q_NETWORK_ID")
    .ok()
    .and_then(|s| s.parse::<q_types::NetworkId>().ok())
    .unwrap_or(q_types::NetworkId::TestnetPhase16);
```

**Location 5** - Line ~5186 (block validation 2):
```rust
// Same pattern as Location 3
```

**Location 6** - Line ~6464 (peer announcement):
```rust
// Same pattern as Locations 1, 2, 4
```

**Location 7** - Line ~9215 (block request):
```rust
// Same pattern as Locations 1, 2, 4, 6
```

### Why Bug #8 is DIFFERENT from Bug #5 and #7:

- **Bug #5**: `.unwrap_or(TestnetPhase7)` fallback values (Phase 9) - affects when env var IS set but fails to parse
- **Bug #7**: Inconsistent defaults in handlers.rs (Phase 15) - different files have different hardcoded phases
- **Bug #8**: **COMPLETELY IGNORES Q_NETWORK_ID** (Phase 16) - uses direct `NetworkId::TestnetPhase15` without ANY env var check

**This bug bypasses ALL environment variable handling!**

### Symptoms Checklist for Bug #8:

- [ ] Docker container connects to bootstrap via DHT
- [ ] Gossipsub shows connected peers
- [ ] Block sync requests consistently fail with "Timeout"
- [ ] Log shows topic `/qnk/testnet-phase15/` despite `Q_NETWORK_ID=testnet-phase16`
- [ ] Both nodes on "same" network but can't communicate
- [ ] No obvious error messages - everything looks "normal"

### 19. ⚠️ **CRITICAL** main.rs Network ID Must Read From Environment Variable

**THIS PREVENTS BUG #8!** ALL network ID usages in main.rs MUST read Q_NETWORK_ID!

**Before EVERY phase transition, run this check:**

```bash
# Find ALL direct NetworkId usages that DON'T read from Q_NETWORK_ID
grep -n "NetworkId::TestnetPhase" crates/q-api-server/src/main.rs | grep -v "Q_NETWORK_ID"

# If ANY results appear, they MUST be changed to read from env var!
```

**Correct Pattern (Use EVERYWHERE):**

```rust
// ✅ CORRECT - Always read from Q_NETWORK_ID with sensible default
let network_id = std::env::var("Q_NETWORK_ID")
    .ok()
    .and_then(|s| s.parse::<q_types::NetworkId>().ok())
    .unwrap_or(q_types::NetworkId::TestnetPhase16); // Update default for current phase
```

**NEVER USE:**

```rust
// ❌ WRONG - Hardcoded value ignores environment
let network_id = q_types::NetworkId::TestnetPhase15;
```

### Phase Transition main.rs Checklist:

- [ ] **Line ~2182** - Peer heights announcement reads Q_NETWORK_ID
- [ ] **Line ~4701** - Gossip handling reads Q_NETWORK_ID
- [ ] **Line ~4833** - Block validation reads Q_NETWORK_ID
- [ ] **Line ~5098** - Gap request reads Q_NETWORK_ID
- [ ] **Line ~5186** - Block validation 2 reads Q_NETWORK_ID
- [ ] **Line ~6464** - Peer announcement reads Q_NETWORK_ID
- [ ] **Line ~9215** - Block request reads Q_NETWORK_ID
- [ ] **Global grep** - No direct `NetworkId::TestnetPhase*` without Q_NETWORK_ID check
- [ ] **Binary rebuilt** - `cargo build --release --package q-api-server`
- [ ] **Test P2P sync** - Docker node successfully syncs with bootstrap

### Diagnostic Commands for Bug #8:

```bash
# 1. Check which gossipsub topics are being used:
journalctl -u q-api-server --since "1 minute ago" | grep "gossipsub topic\|Publishing.*to"

# 2. Look for sync request failures:
journalctl -u q-api-server --since "5 minutes ago" | grep "Timeout\|outbound failure"

# 3. Verify network ID in peer heights:
journalctl -u q-api-server --since "1 minute ago" | grep "peer-heights"

# 4. Check Docker container logs:
docker logs <container> 2>&1 | tail -50 | grep -E "topic|phase|sync"
```

---

## 🔥 BUG #7: P2P Balance Propagation Failure - Inconsistent Network ID Defaults (Phase 15 Discovery!)

**Discovered**: 2025-12-10 during Phase 15 mining reward testing
**Impact**: CRITICAL - Mining rewards credited locally but NEVER propagate to network
**Root Cause**: `handlers.rs` has 4+ different hardcoded network ID defaults scattered throughout the file

### How Bug #7 Manifests:

1. **Server Alpha (Docker)** mines blocks successfully - shows 300 blocks found
2. **Mining stats show** 85.97 KH/s hash rate, 32.29% network share
3. **BUT balance shows 0.0000 QUG** on Server Beta despite mining on connected node
4. **Root cause**: Balance updates broadcast to WRONG gossipsub topic

**Why this happens**:
- Server Alpha (Docker) runs WITHOUT `Q_NETWORK_ID` environment variable set
- `handlers.rs` has hardcoded defaults like `"testnet-phase14"`, `"testnet-phase12"`, etc.
- Balance updates are broadcast to `/qnk/testnet-phase14/balance-updates`
- Server Beta listens on `/qnk/testnet-phase15/balance-updates`
- **Messages never reach Server Beta!**

### The 4 Bug Locations Found in handlers.rs:

```bash
# Search command to find all inconsistent defaults:
grep -n "unwrap_or_else.*testnet-phase" crates/q-api-server/src/handlers.rs
```

**Location 1** - Line ~85 (Network stats):
```rust
// ❌ WRONG - Phase 5 default!
network_id: std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "testnet-phase5".to_string()),

// ✅ CORRECT - Must match current phase
network_id: std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "testnet-phase15".to_string()),
```

**Location 2** - Line ~465 (API response):
```rust
// ❌ WRONG - Phase 12 default!
"network_id": std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "testnet-phase12".to_string()),

// ✅ CORRECT
"network_id": std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "testnet-phase15".to_string()),
```

**Location 3** - Line ~2221 (Balance update broadcast):
```rust
// ❌ WRONG - Phase 10 default!
std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "testnet-phase10".to_string());

// ✅ CORRECT
std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "testnet-phase15".to_string());
```

**Location 4** - Lines ~4986-4990 (Mining reward handler):
```rust
// ❌ WRONG - Phase 14 default!
let network_id_str = std::env::var("Q_NETWORK_ID")
    .unwrap_or_else(|_| "testnet-phase14".to_string());
let network_id = network_id_str.parse::<q_types::NetworkId>()
    .unwrap_or(q_types::NetworkId::TestnetPhase14);

// ✅ CORRECT - Use current phase
let network_id_str = std::env::var("Q_NETWORK_ID")
    .unwrap_or_else(|_| "testnet-phase15".to_string());
let network_id = network_id_str.parse::<q_types::NetworkId>()
    .unwrap_or(q_types::NetworkId::TestnetPhase15);
```

### Why Bug #7 is SILENT and DECEPTIVE:

1. **Mining WORKS** - You see hash rate, blocks found, everything looks fine
2. **Local balance updates** - The mining node's local database IS updated
3. **P2P connection works** - libp2p shows connected peers
4. **Sync works** - Blocks sync correctly between nodes
5. **ONLY balance propagation fails** - Balance updates go to wrong topic

**This bug is nearly impossible to diagnose without knowing to look for it!**

### Symptoms Checklist:

- [ ] Mining shows blocks found but balance stays 0.0000
- [ ] `journalctl -u q-api-server | grep "balance_updates"` shows broadcasts
- [ ] BUT receiving node shows `Loaded 0 transactions` for miner wallet
- [ ] No errors in logs - everything appears normal

### 18. ⚠️ **CRITICAL** handlers.rs Network ID Consistency Check

**THIS PREVENTS BUG #7!** ALL network ID defaults MUST match current phase.

**Before EVERY phase transition, run this check:**

```bash
# Find ALL inconsistent phase defaults in handlers.rs
grep -n "testnet-phase" crates/q-api-server/src/handlers.rs | grep -v "phase15"
# If ANY results appear, they MUST be updated!

# Also check main.rs
grep -n "testnet-phase" crates/q-api-server/src/main.rs | grep -v "phase15"

# And block_producer.rs
grep -n "testnet-phase" crates/q-api-server/src/block_producer.rs | grep -v "phase15"
```

**Global Search & Replace for Phase Transition:**

```bash
# When transitioning from phase15 to phase16:
cd crates/q-api-server/src/

# Find all occurrences
grep -rn "testnet-phase15" .
grep -rn "TestnetPhase15" .

# Replace (carefully review each change!)
sed -i 's/testnet-phase15/testnet-phase16/g' handlers.rs main.rs block_producer.rs
sed -i 's/TestnetPhase15/TestnetPhase16/g' handlers.rs main.rs block_producer.rs
```

### Phase Transition Handlers.rs Checklist:

- [ ] **Line ~85** - Network stats `network_id` default updated
- [ ] **Line ~465** - API response `network_id` default updated
- [ ] **Line ~2221** - Balance update broadcast network ID updated
- [ ] **Lines ~4986-4990** - Mining reward handler network ID updated
- [ ] **Global grep** - No old phase references remain in handlers.rs
- [ ] **Binary rebuilt** - `cargo build --release --package q-api-server`
- [ ] **Test P2P balance propagation** - Mine on Docker, verify balance on Server Beta

### Diagnostic Commands for Bug #7:

```bash
# 1. Check which topic balance updates are being broadcast to:
journalctl -u q-api-server --since "5 minutes ago" | grep -i "balance.*topic\|gossipsub.*balance"

# 2. Check if balance updates are being received:
journalctl -u q-api-server --since "5 minutes ago" | grep "P2P BALANCE\|balance_updates"

# 3. Check wallet transaction count:
journalctl -u q-api-server --since "5 minutes ago" | grep "Loaded.*transactions"

# 4. Verify network ID in use:
journalctl -u q-api-server --since "30 seconds ago" | grep "Network:\|network_id\|testnet-phase"
```

---

## 🔥 BUG #9: main.rs .unwrap_or(TestnetPhase15) Fallback Values (Phase 16 Discovery - Second Wave!)

**Discovered**: 2025-12-12 during Phase 16 Docker sync testing (same day as Bug #8)
**Impact**: CRITICAL - P2P sync fails even after fixing Bug #8 due to additional fallback defaults
**Root Cause**: 7 additional `.unwrap_or(q_types::NetworkId::TestnetPhase15)` fallback values in main.rs

### How Bug #9 Differs from Bug #8:

- **Bug #8**: Direct hardcoded `let network_id = q_types::NetworkId::TestnetPhase15;` (ignores Q_NETWORK_ID entirely)
- **Bug #9**: Has Q_NETWORK_ID check, but `.unwrap_or(TestnetPhase15)` fallback defaults to wrong phase when parsing fails or env var is missing

### The 7 Bug #9 Locations Found in main.rs:

```bash
# Search command to find all fallback defaults:
grep -n "unwrap_or(q_types::NetworkId::TestnetPhase" crates/q-api-server/src/main.rs
```

**Location 1** - Line ~3479:
```rust
// ❌ WRONG - Fallback to Phase 15
.unwrap_or(q_types::NetworkId::TestnetPhase15); // ✅ v1.1.0-beta: Phase 13 default (Bug #5 fix)

// ✅ CORRECT - Updated to Phase 16
.unwrap_or(q_types::NetworkId::TestnetPhase16); // ✅ v1.3.3-beta: Phase 16 default (Bug #9 fix)
```

**Location 2** - Lines ~5417-5419 (multi-line):
```rust
// ❌ WRONG
.unwrap_or(
    q_types::NetworkId::TestnetPhase15,
); // ✅ v1.1.0-beta: Phase 13 default (Bug #5 fix)

// ✅ CORRECT
.unwrap_or(
    q_types::NetworkId::TestnetPhase16,
); // ✅ v1.3.3-beta: Phase 16 default (Bug #9 fix)
```

**Location 3** - Line ~5757:
```rust
// ❌ WRONG
.unwrap_or(q_types::NetworkId::TestnetPhase15);

// ✅ CORRECT
.unwrap_or(q_types::NetworkId::TestnetPhase16); // ✅ v1.3.3-beta: Phase 16 default (Bug #9 fix)
```

**Location 4** - Line ~6823:
```rust
// ❌ WRONG
.unwrap_or(q_types::NetworkId::TestnetPhase15);

// ✅ CORRECT
.unwrap_or(q_types::NetworkId::TestnetPhase16); // ✅ v1.3.3-beta: Phase 16 default (Bug #9 fix)
```

**Location 5** - Line ~7637:
```rust
// ❌ WRONG
.unwrap_or(q_types::NetworkId::TestnetPhase15);

// ✅ CORRECT
.unwrap_or(q_types::NetworkId::TestnetPhase16); // ✅ v1.3.3-beta: Phase 16 default (Bug #9 fix)
```

**Location 6** - Line ~8554:
```rust
// ❌ WRONG
.unwrap_or(q_types::NetworkId::TestnetPhase15);

// ✅ CORRECT
.unwrap_or(q_types::NetworkId::TestnetPhase16); // ✅ v1.3.3-beta: Phase 16 default (Bug #9 fix)
```

**Location 7** - Line ~8685:
```rust
// ❌ WRONG
.unwrap_or(q_types::NetworkId::TestnetPhase15);

// ✅ CORRECT
.unwrap_or(q_types::NetworkId::TestnetPhase16); // ✅ v1.3.3-beta: Phase 16 default (Bug #9 fix)
```

### Why Bug #9 is DIFFERENT from Bug #5:

- **Bug #5** (Phase 9): Found 12-15 `.unwrap_or(TestnetPhase7)` fallbacks - old phase from initial development
- **Bug #9** (Phase 16): These are NEWER fallbacks from Bug #5 fix that were set to Phase 15 but never updated for Phase 16!

**The irony**: Bug #5 was fixed by updating fallbacks to Phase 15, but those same fallbacks now need updating for each new phase!

### 20. ⚠️ **CRITICAL** main.rs Fallback Default Update Check

**THIS PREVENTS BUG #9!** ALL `.unwrap_or()` fallback NetworkIds MUST match current phase.

**Before EVERY phase transition, run this check:**

```bash
# Find ALL .unwrap_or() fallback defaults in main.rs
grep -n "unwrap_or(q_types::NetworkId::TestnetPhase" crates/q-api-server/src/main.rs

# If ANY results show old phase (e.g., Phase15 when transitioning to Phase16), UPDATE THEM!
```

**Global Search & Replace for Phase Transition:**

```bash
# When transitioning from phase15 to phase16:
cd crates/q-api-server/src/

# Find all .unwrap_or fallbacks with old phase
grep -n "unwrap_or(q_types::NetworkId::TestnetPhase15)" main.rs

# Replace (carefully review changes!)
sed -i 's/unwrap_or(q_types::NetworkId::TestnetPhase15)/unwrap_or(q_types::NetworkId::TestnetPhase16)/g' main.rs
```

### Phase Transition main.rs Fallback Checklist (Bug #9 Prevention):

- [ ] **Line ~3479** - First fallback updated to new phase
- [ ] **Lines ~5417-5419** - Multi-line fallback updated
- [ ] **Line ~5757** - Third fallback updated
- [ ] **Line ~6823** - Fourth fallback updated
- [ ] **Line ~7637** - Fifth fallback updated
- [ ] **Line ~8554** - Sixth fallback updated
- [ ] **Line ~8685** - Seventh fallback updated
- [ ] **Global grep** - Zero results for `unwrap_or(q_types::NetworkId::TestnetPhase{OLD})`
- [ ] **Binary rebuilt** - `cargo build --release --package q-api-server`
- [ ] **Test Docker sync** - Fresh container syncs successfully from bootstrap

### Diagnostic Commands for Bug #9:

```bash
# 1. Check current fallback defaults in binary:
strings target/release/q-api-server | grep -i "testnet-phase" | sort -u

# 2. Verify gossipsub topic in use:
journalctl -u q-api-server --since "1 minute ago" | grep "peer-heights\|turbo-sync"

# 3. Check Docker container network ID:
docker logs <container> 2>&1 | grep "Network:\|network_id"

# 4. Monitor sync attempts:
docker logs <container> 2>&1 | grep -E "sync|blocks|height"
```

### Key Lesson from Bug #9:

**Every phase transition requires updating TWO types of NetworkId references:**

1. **Bug #8 type**: Direct hardcoded `let network_id = NetworkId::TestnetPhaseX;`
2. **Bug #9 type**: Fallback defaults `.unwrap_or(NetworkId::TestnetPhaseX)`

**Both must be updated or P2P sync will fail!**

---

### 21. ⚠️ **CRITICAL** Browser js-libp2p Config (Frontend P2P)

**Discovered**: 2026-02-14 during Phase 20 transition planning
**Impact**: CRITICAL - Browser nodes subscribe to WRONG gossipsub topics, cannot receive blocks/transactions
**Root Cause**: Browser P2P config has its own `NETWORK_ID` constant and hardcoded fallback values

**The browser has its OWN libp2p node** (js-libp2p) that connects to the network via WebSocket.
If the browser's NETWORK_ID doesn't match the server's phase, the browser will:
- Subscribe to wrong gossipsub topics (e.g., `/qnk/testnet-phase19/blocks` instead of `/qnk/testnet-phase20/blocks`)
- Never receive blocks from the network
- Show stale data in the explorer and dashboard

**6 locations must be updated in the frontend:**

**Location 1** - `gui/quantum-wallet/src/libp2p/config.ts` (line 10) - **PRIMARY**:
```typescript
// ❌ WRONG - Old phase
export const NETWORK_ID = 'testnet-phase19'

// ✅ CORRECT - Updated to new phase
export const NETWORK_ID = 'testnet-phase20'
```
This is the MOST IMPORTANT location. All TOPICS constants derive from this value automatically:
- `BLOCKS: /qnk/${NETWORK_ID}/blocks`
- `TRANSACTIONS: /qnk/${NETWORK_ID}/transactions`
- `PEER_HEIGHTS: /qnk/${NETWORK_ID}/peer-heights`
- etc.

**Location 2** - `gui/quantum-wallet/src/libp2p/decoder.ts` (line 172) - **FALLBACK**:
```typescript
// ❌ WRONG
networkId = headerArray[2] || 'testnet-phase19'

// ✅ CORRECT
networkId = headerArray[2] || 'testnet-phase20'
```

**Location 3** - `gui/quantum-wallet/src/libp2p/decoder.ts` (line 208) - **FALLBACK**:
```typescript
// ❌ WRONG
networkId = headerObj.network_id || 'testnet-phase19'

// ✅ CORRECT
networkId = headerObj.network_id || 'testnet-phase20'
```

**Location 4** - `gui/quantum-wallet/src/hooks/useRealtimeBlocks.ts` (line 550) - **FALLBACK**:
```typescript
// ❌ WRONG
networkId: apiBlock.network_id || 'testnet-phase19',

// ✅ CORRECT
networkId: apiBlock.network_id || 'testnet-phase20',
```

**Location 5** - `gui/quantum-wallet/src/contexts/LibP2PContext.tsx` (line 149) - **COMMENT**:
```typescript
// Update documentation comment to reference new phase
// * - Network: testnet-phase20
```

**Location 6** - `gui/quantum-wallet/src/libp2p/config.ts` (line 9) - **COMMENT**:
```typescript
// 🔥 v6.3.0-browser: Updated to match Server Beta (testnet-phase20)
```

**Phase Transition Frontend Checklist:**

```bash
# Find ALL browser phase references:
grep -rn "testnet-phase19" gui/quantum-wallet/src/

# Replace (carefully review each change!):
cd gui/quantum-wallet/src/
sed -i 's/testnet-phase19/testnet-phase20/g' libp2p/config.ts libp2p/decoder.ts hooks/useRealtimeBlocks.ts contexts/LibP2PContext.tsx

# Rebuild frontend:
cd gui/quantum-wallet && npx vite build

# Deploy frontend:
cp -r dist/* dist-final/
```

**WHY THIS IS CRITICAL**:
- Browser nodes are the USER-FACING interface - if they're on wrong topics, users see stale data
- The `config.ts` NETWORK_ID propagates to ALL topic subscriptions automatically
- But the fallback values in `decoder.ts` and `useRealtimeBlocks.ts` are SEPARATE and must be updated individually
- If ONLY `config.ts` is updated but fallbacks are missed, blocks received via HTTP API will show wrong networkId

---

## 🔥 PHASE 20 TRANSITION BUGS (2026-02-14) - FOUR MORE BUGS FOUND!

Phase 20 (testnet-phase20, genesis Feb 14 2026) revealed **FOUR ADDITIONAL BUGS** (#10-13).
These are **NOT NetworkId parsing bugs** but instead critical data bugs that break emission,
mining rewards, and P2P connectivity.

### Bug #10: Multiple GENESIS_TIMESTAMP Locations Not Updated
- **Symptom**: Emission calculations use wrong genesis date (Oct 2025 instead of Feb 2026)
- **Impact**: Era calculations wrong, time-based rewards miscalculated, emission schedule shifted
- **Root cause**: GENESIS_TIMESTAMP is defined in FOUR separate locations, only 2 were updated by sed:
  1. `crates/q-storage/src/emission_controller.rs:29` - Updated by transition (pub const)
  2. `crates/q-api-server/src/integrated_mining.rs:275` - Updated by transition
  3. `crates/q-api-server/src/handlers.rs:810` - **MISSED!** Used for miner display rewards
  4. `crates/q-storage/src/balance_consensus.rs:41` - **MISSED!** Used for adaptive reward calculation
- **Fix**: Must grep for ALL GENESIS_TIMESTAMP definitions and the OLD timestamp value:
  ```bash
  # Find all GENESIS_TIMESTAMP definitions:
  grep -rn "GENESIS_TIMESTAMP.*=" crates/ --include="*.rs" | grep -v "test"
  # Find all instances of the old timestamp VALUE (catches non-const usage):
  grep -rn "1761436800\|1733980800" crates/ --include="*.rs"
  ```
- **Lesson**: GENESIS_TIMESTAMP is NOT a single constant - it's duplicated across 4 crates.
  Search by VALUE not by NAME to catch all instances.

### Bug #11: LEGACY_FIXED_REWARD Wrong Decimal Format (8-dec vs 24-dec)
- **Symptom**: Actual coinbase reward is 5e-18 QUG but log message says "0.05 QUG"
- **Impact**: Miners receive essentially zero rewards during bootstrap phase (first 200K blocks)
- **Root cause**: `block_producer.rs` line 1075:
  ```rust
  const LEGACY_FIXED_REWARD: u128 = 5_000_000; // "0.05 QUG (Phase 1-10 legacy)"
  ```
  5,000,000 = 0.05 QUG with **8 decimals**, but the system uses **24 decimals** since v3.0.4.
  With 24 decimals, 5,000,000 = 5 × 10⁻¹⁸ QUG (effectively zero).
- **Additional**: Log message "Using FIXED reward (0.05 QUG)" is a **hardcoded string** that
  doesn't compute from the actual value, masking the bug.
- **Fix**: Replace LEGACY_FIXED_REWARD with `calculate_block_reward_time_based()` call:
  ```rust
  let reward = crate::handlers::calculate_block_reward_time_based(
      q_storage::emission_controller::GENESIS_TIMESTAMP,
      block_timestamp,
  );
  ```
- **Lesson**: After the v3.0.4 migration to 24 decimals, ANY constant with fewer than ~18 zeros
  is likely wrong. Grep for all reward/amount constants and verify they use 24 decimals:
  ```bash
  grep -rn "REWARD.*=.*[0-9]" crates/ --include="*.rs" | grep -v "10.*000.*000.*000.*000"
  ```

### Bug #12: Emission Stats Never Recorded (record_daily_emission never called)
- **Symptom**: Explorer page shows "0.000000 / 7,191.8 QUG mined today" despite blocks being produced
- **Impact**: Emission analytics endpoint returns near-zero data, explorer emission card is useless
- **Root cause**: `EmissionController::record_daily_emission()` exists (v6.2.4) but is NEVER
  called from block_producer.rs or anywhere in q-api-server. The function was added but never wired.
- **Additional**: `EmissionController::record_emission()` (which updates `total_emitted_this_era`)
  is also never called, so era emission tracking is always zero.
- **Fix**: Add call in block_producer.rs after `total_reward` is determined:
  ```rust
  // v6.3.0: Record emission for daily stats tracking
  if let Some(ref bc) = self.balance_consensus {
      if let Err(e) = bc.record_daily_emission(block_timestamp, total_reward).await {
          warn!("Failed to record daily emission: {}", e);
      }
  }
  ```
- **Lesson**: When adding tracking/analytics functions, ALWAYS wire them to the actual code path.
  Search for `pub.*fn.*record` and verify each has at least one caller.

### Bug #13: Bootstrap Peer IDs Not Updated for New Phase Data Directory
- **Symptom**: New nodes can't connect to bootstrap peers via hardcoded peer IDs
- **Impact**: P2P connectivity fails for nodes compiled with old peer IDs
- **Root cause**: Phase transitions create new data directories (data-mine20/) which generate
  new libp2p identity keys, producing new PeerIDs. Hardcoded peer IDs become stale.
- **Locations requiring update** (ALL of these):
  1. `crates/q-network/src/unified_network_manager.rs:82-90` - HARDCODED_BOOTSTRAP_PEERS array
  2. `crates/q-network/src/unified_network_manager.rs:102` - HARDCODED_BOOTSTRAP_PEER constant
  3. `crates/q-api-server/src/main.rs:193-197` - is_allowed_balance_update_origin()
  4. `gui/quantum-wallet/src/libp2p/config.ts:26` - Browser bootstrap peer
  5. `gui/quantum-wallet/src/contexts/LibP2PContext.tsx` - Comment/backup peer ID
  6. `gui/quantum-wallet/src/libp2p/torConfig.ts` - Tor bridge peer ID
  7. `CLAUDE.md` - Documentation peer IDs
- **Fix**: After starting the new phase, get peer IDs from API:
  ```bash
  # Get Beta PeerID:
  curl -s http://localhost:8080/api/v1/peer-id | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['peer_id'])"
  # Get Gamma PeerID:
  ssh root@109.205.176.60 'curl -s http://localhost:8080/api/v1/peer-id' | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['peer_id'])"
  # Then replace ALL old peer IDs:
  OLD_PEER="<old phase peer id>"
  NEW_PEER="<new peer id from curl output>"
  grep -rn "$OLD_PEER" crates/ gui/ CLAUDE.md --include="*.rs" --include="*.ts" --include="*.tsx" --include="*.md"
  ```
- **Lesson**: Peer IDs change with data directories. Add a post-transition step to query
  peer IDs from the running servers and update all hardcoded references.

---

### 22. ⚠️ **CRITICAL** GENESIS_TIMESTAMP Constants (Multiple Files)

When starting a new phase with a fresh genesis date, you MUST update ALL of these:

**Location 1** - `crates/q-storage/src/emission_controller.rs` (line 29) - **PRIMARY**:
```rust
pub const GENESIS_TIMESTAMP: u64 = 1771027200; // UPDATE to new genesis unix timestamp
```

**Location 2** - `crates/q-api-server/src/handlers.rs` (line 810) - **MINER DISPLAY**:
```rust
pub const GENESIS_TIMESTAMP: u64 = 1771027200; // UPDATE - used by calculate_block_reward_time_based()
```

**Location 3** - `crates/q-storage/src/balance_consensus.rs` (line 41) - **ADAPTIVE REWARD**:
```rust
pub const GENESIS_TIMESTAMP: u64 = 1771027200; // UPDATE - used for adaptive reward era calculation
```

**Location 4** - `crates/q-api-server/src/integrated_mining.rs` (line 275) - **MINING**:
```rust
const GENESIS_TIMESTAMP: u64 = 1771027200; // UPDATE - used for mining reward time-based calc
```

**Verification**:
```bash
# Search by OLD timestamp value to catch ALL instances:
grep -rn "OLD_TIMESTAMP_VALUE" crates/ --include="*.rs"
# Should return ZERO results after update
```

### 23. ⚠️ **CRITICAL** Bootstrap Peer IDs (7 Locations)

After starting the new phase and servers generate new identities:

```bash
# Step 1: Get new PeerIDs from running servers
BETA_PEER=$(curl -s http://localhost:8080/api/v1/peer-id | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['peer_id'])")
GAMMA_PEER=$(ssh root@109.205.176.60 'curl -s http://localhost:8080/api/v1/peer-id' | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['peer_id'])")

# Step 2: Replace in all 7 locations
grep -rn "OLD_BETA_PEER_ID" crates/ gui/ CLAUDE.md --include="*.rs" --include="*.ts" --include="*.tsx" --include="*.md"
# Replace each occurrence with $BETA_PEER
```

**Locations**:
1. `crates/q-network/src/unified_network_manager.rs:82-90` - HARDCODED_BOOTSTRAP_PEERS
2. `crates/q-network/src/unified_network_manager.rs:102` - HARDCODED_BOOTSTRAP_PEER
3. `crates/q-api-server/src/main.rs:193-197` - is_allowed_balance_update_origin()
4. `gui/quantum-wallet/src/libp2p/config.ts:26` - Browser bootstrap
5. `gui/quantum-wallet/src/contexts/LibP2PContext.tsx` - Fallback peer
6. `gui/quantum-wallet/src/libp2p/torConfig.ts` - Tor bridge peer
7. `CLAUDE.md` - Documentation

### 24. ⚠️ **CRITICAL** Encryption Keys: NEVER Pre-Generate (Bug #15)

**THIS PREVENTS BUG #15!** The server's `EncryptionManager` creates keys in a specific binary
format with ZK-STARK proof. Manual key generation (e.g., `openssl rand`) produces incompatible
files that cause database open failures and crash loops.

**CORRECT — Let server auto-create:**
```bash
# Step 1: Delete any existing keys file (old phase or manually created)
rm -f /opt/encryption-phaseX.keys

# Step 2: Start server — it auto-generates proper format keys
systemctl start q-api-server

# Step 3: Verify keys were created
ls -la /opt/encryption-phaseX.keys
# Should be ~120 bytes (binary format), NOT 65 bytes (hex text)
```

**WRONG — Manual key generation (causes crash loop!):**
```bash
# ❌ NEVER DO THIS:
openssl rand -hex 32 > /opt/encryption-phaseX.keys       # 65-byte ASCII text
openssl rand -base64 32 > /opt/encryption-phaseX.keys     # 45-byte ASCII text
dd if=/dev/urandom bs=32 count=1 > /opt/encryption-phaseX.keys  # 32-byte binary (wrong format)
```

**Symptoms of bad encryption keys:**
```
Error: Failed to open hot database
Caused by:
    AES-GCM decryption failed (wrong passphrase?): aead::Error
```

**Emergency fix (service crash-looping):**
```bash
systemctl stop q-api-server
rm -f /opt/encryption-phaseX.keys   # Delete the bad keys
rm -rf ./data-mineX/hot/*           # Optional: clean corrupted DB if keys were used
systemctl start q-api-server        # Server will auto-generate proper keys
```

### 25. ⚠️ **CRITICAL** Frontend Rebuild + Verification (Bug #16)

**THIS PREVENTS BUG #16!** The frontend JS bundle contains hardcoded gossipsub topic strings.
After ANY phase transition, the frontend MUST be rebuilt and the deployed bundle verified.

**Phase Transition Frontend Deploy Checklist:**

```bash
# 1. Rebuild frontend (Vite outputs to dist-final/ directly)
cd gui/quantum-wallet && npm run build

# 2. Verify deployed bundle uses NEW phase (MANDATORY!)
grep -o "testnet-phase[0-9]*" dist-final/assets/index-*.js | sort -u
# ✅ EXPECTED: testnet-phaseNEW (e.g., testnet-phase21)
# ❌ FAILURE:  testnet-phaseOLD (e.g., testnet-phase20) — config.ts not saved!

# 3. Check index.html references new assets
grep "index-" dist-final/index.html
# Should reference the latest timestamp-suffixed JS file

# 4. Tell users to hard refresh (Ctrl+Shift+R) to bypass browser cache
```

**Why this is NOT caught by normal testing:**
- Backend deploys + binary restarts → backend works on new phase
- Browser opens quillon.xyz → nginx serves STALE JS from dist-final/
- Stale JS subscribes to OLD phase topics → no P2P blocks → HTTP fallback
- HTTP fallback works → blocks display normally → looks "fine"
- But P2P verification shows 0/0 → user notices something is wrong
- Without user report, this could go unnoticed for days

**The fix order matters:**
1. Deploy backend binary first (so server is on new phase)
2. Rebuild frontend immediately after (so browser matches server)
3. Verify the deployed JS bundle before announcing the transition

---

## 🔥 BUG #14: DEX/Contracts/Tokens/RWA Not Reset on Phase Transition (Phase 20 Discovery!)

**Discovered**: 2026-02-14 during Phase 20 deployment
**Impact**: CRITICAL - Stale Phase 19 tokens, DEX pools, contracts, and RWA data pollutes Phase 20
**Root Cause**: Phase transitions only reset blocks and balances, but CF_MANIFEST contract/pool/token data survives

### How Bug #14 Manifests:

1. **Phase 20 starts** with fresh `data-mine20/` directory
2. **P2P state sync** connects to Phase 19 nodes (e.g., Alpha Docker still running old phase)
3. **State sync imports** Phase 19 contracts, tokens, pools, token balances via gossipsub
4. **Dashboard shows** 32 stale tokens (LLAMA×4, CHAD×2, VAULT×2, SMOL, FOMO, TRUMP etc.)
5. **12 stale DEX pools** with incorrect reserves from Phase 19 trading history
6. **RWA tokens** (VAULT, FORGE) duplicated — both stale P2P-synced copies AND fresh startup copies

### What Data Types Survive Phase Transitions (All in CF_MANIFEST):

| Prefix | Data Type | Reset by DB wipe? | Re-imported via P2P? |
|--------|-----------|-------------------|---------------------|
| `contract_` | Smart contracts & tokens | Yes | **Yes** (state_sync) |
| `liquidity_pool:` | DEX pools | Yes | **Yes** (gossipsub) |
| `token_balance_` | Custom token balances | Yes | **Yes** (state_sync) |
| `stake_position_` | Staking positions | Yes | **Yes** (state_sync) |
| `swap_history_` | DEX swap history | Yes | No |
| `collateral_vault` | Collateral vault data | Yes | No |

### The Hidden Problem: P2P Cross-Phase Contamination

Even with a fresh database, if ANY node on the old phase is still running:
- State sync discovers it via DHT and imports its contracts/pools/tokens
- Gossipsub pool updates flow from old-phase nodes to new-phase nodes
- **The gossipsub topic includes network_id** (`/qnk/testnet-phase20/pools`) but state_sync
  uses HTTP which does NOT filter by network_id

### Fix (v6.3.0+):

1. **Storage method**: `StorageEngine::purge_dex_and_contracts()` — scans and deletes all
   contract_, liquidity_pool:, token_balance_, stake_position_, swap_history_ prefixes
2. **Admin endpoint**: `POST /api/v1/admin/purge-phase-data` (founder wallet auth required)
3. **In-memory cleanup**: Clears liquidity_pools, token_balances, and deployed_contracts DashMaps

### Prevention Checklist:

- [ ] **Stop ALL old-phase nodes** before starting new phase (Alpha Docker containers!)
- [ ] **Verify no old-phase peers** in gossipsub after new phase starts
- [ ] **Call purge endpoint** after first startup if stale data detected
- [ ] **Check token count**: `curl /api/v1/dex/supported-tokens` should show only built-in tokens
- [ ] **Check pool count**: `curl /api/v1/dex/pools` should show 0 pools initially

### Diagnostic Commands:

```bash
# Check for stale tokens (should only show QUG, QUGUSD, VAULT, FORGE on fresh phase):
curl -s localhost:8080/api/v1/dex/supported-tokens | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Tokens: {len(d.get(\"data\",[]))}')"

# Check for stale pools (should be 0 on fresh phase):
curl -s localhost:8080/api/v1/dex/pools | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Pools: {len(d.get(\"data\",[]))}')"

# Purge stale data (requires founder wallet):
curl -X POST localhost:8080/api/v1/admin/purge-phase-data -H "x-wallet-auth: FOUNDER_WALLET_HEX"
```

---

## 🔥 PHASE 21 TRANSITION BUGS (2026-02-14) - THREE MORE BUGS FOUND!

Phase 21 (testnet-phase21, genesis Feb 15 2026 00:00:00 UTC / 1771113600) revealed
**THREE ADDITIONAL BUGS** (#15-17). These are infrastructure/deployment bugs that cause
silent P2P failures and database crashes.

### Bug #15: Encryption Keys Format Mismatch (Server Crash Loop)

- **Symptom**: Service crash loops with `Error: Failed to open hot database` after phase transition
- **Impact**: CRITICAL — server restarts every 10 seconds, never starts successfully
- **Root cause**: Transition script generated encryption keys with `openssl rand -hex 32` (65-byte
  ASCII text file). But the server's `EncryptionManager` expects a specific binary format with
  ZK-STARK proof (~120 bytes). The format mismatch causes AES-GCM decryption to fail on every
  database open attempt.
- **How it manifests**:
  1. Transition script creates `/opt/encryption-phase21.keys` with `openssl rand -hex 32`
  2. Server starts, reads the bad keys file
  3. Tries to derive AES-GCM key from the file contents
  4. Database open fails: `"AES-GCM decryption failed (wrong passphrase?): aead::Error"`
  5. Service exits with code 1, systemd restarts after 10s, same crash repeats
  6. Beta crashed 9 times in 5 minutes before root cause identified
- **Fix**: NEVER pre-generate encryption keys files. Let the server auto-create them on first boot.
  ```bash
  # If bad keys file exists, just delete it:
  rm -f /opt/encryption-phaseX.keys
  # The server will auto-generate proper format keys on next start
  systemctl restart q-api-server
  ```
- **Transition script fix**: Removed the `openssl rand` key generation step entirely.
  The updated script just removes any existing keys file and lets the server self-provision.
- **Lesson**: The encryption key file has a specific binary format (not plain hex).
  NEVER create it manually — always let the server auto-generate it on first boot.
  If in doubt, delete the keys file and restart.
- **Key files**: `scripts/transition-phase21.sh`, `/opt/encryption-phaseX.keys`

### Bug #16: Frontend JS Bundle Not Rebuilt After Phase Transition (P2P Block Verification Failure)

- **Symptom**: P2P Network Overview shows "1 connections • 6 topics" but "Verified: 0/0"
- **Impact**: CRITICAL — blocks arrive via HTTP fallback only, bypassing light client verification
- **Root cause**: After updating `config.ts` with `NETWORK_ID = 'testnet-phase21'` and all
  backend code, the frontend was NOT rebuilt. The deployed JS bundle still referenced
  `testnet-phase20` gossipsub topics.
- **How it manifests**:
  1. Server broadcasts blocks on `/qnk/testnet-phase21/blocks` (correct)
  2. Browser's JS bundle subscribes to `/qnk/testnet-phase20/blocks` (old, stale)
  3. Topics don't match → gossipsub messages never delivered to browser
  4. After 10 seconds with no P2P blocks, HTTP fallback activates
  5. HTTP-fetched blocks bypass `verifyBlock()` → verification counter stays 0/0
  6. User sees blocks updating (via HTTP) but "Verified: 0/0" in Network Map modal
  7. The browser IS connected (1 peer, 6 topics shown) — but on the WRONG topic names
- **Why it's deceptive**: The P2P connection status shows "connected" because the WebSocket
  connection to the bootstrap peer is established. The 6 topics reflect the browser's
  subscriptions — the browser THINKS it's subscribed to the right topics. But the actual
  topic strings contain the old phase name, so the server never routes messages to this peer.
- **Fix**: Always rebuild the frontend immediately after any phase transition:
  ```bash
  cd gui/quantum-wallet && npm run build
  # Vite outputs directly to dist-final/ (build.outDir configured)
  # Then verify the new bundle uses the correct phase:
  grep -o "testnet-phase[0-9]*" dist-final/assets/index-*.js | head -5
  # Should show: testnet-phase21 (NOT testnet-phase20)
  ```
- **Verification**: After deploying, tell users to **hard refresh** (Ctrl+Shift+R) to bypass
  browser cache. The new JS bundle will subscribe to correct topics and P2P blocks will flow.
- **Lesson**: Frontend rebuild is as CRITICAL as the backend binary rebuild during phase
  transitions. The browser's P2P node has its own gossipsub topic subscriptions derived from
  the `NETWORK_ID` constant in `config.ts` — if the JS bundle is stale, P2P is broken.
- **Key files**: `gui/quantum-wallet/src/libp2p/config.ts`, `gui/quantum-wallet/dist-final/`

### Bug #17: HTTP Fallback Blocks Bypass Verification (Design Gap)

- **Symptom**: "Verified: 0/0" even when blocks are flowing (via HTTP), user thinks network is broken
- **Impact**: MEDIUM — verification stats misleading, users lose confidence in network integrity
- **Root cause**: The `useRealtimeBlocks.ts` hook has two block reception paths:
  1. **P2P path** (`handleBlockMessage`) → calls `verifyBlock()` → updates `verificationStats`
  2. **HTTP path** (`fetchBlocksViaHttp`) → directly adds blocks to state → NO verification

  When P2P is down (wrong topics, mesh not formed, etc.), ALL blocks come via HTTP and
  verification never runs. The "Verified: 0/0" display in NetworkMapModal makes users think
  no blocks are being verified at all.
- **Current behavior**: HTTP fallback blocks are NOT verified because:
  - They come from a trusted server (our own API), so verification is less critical
  - The 8-point verification checks (structure, timestamp, network, phase, etc.) would all pass
    for HTTP blocks since the server already validated them
  - But NOT running verification means the counter stays at 0/0, which is confusing
- **Recommended fix**: Add verification to HTTP fallback blocks in `fetchBlocksViaHttp()`:
  ```typescript
  // In fetchBlocksViaHttp(), after converting blocks to QBlock format:
  newBlocks.forEach(async (block) => {
    try {
      const verificationResult = await verifyBlock(block)
      setVerificationStats(prev => ({
        ...prev,
        blocksVerified: prev.blocksVerified + 1,
        blocksValid: prev.blocksValid + (verificationResult?.valid ? 1 : 0),
        blocksInvalid: prev.blocksInvalid + (verificationResult?.valid ? 0 : 1),
      }))
      // Create verified block with result attached
      const verifiedBlock: VerifiedBlock = { ...block, verification: verificationResult }
      setBlockHistory(prev => [verifiedBlock, ...prev].slice(0, MAX_BLOCK_HISTORY))
    } catch (e) { /* ... */ }
  })
  ```
- **Status**: NOT YET FIXED — mitigation is to ensure P2P works (fix Bug #16 first).
  When P2P works, blocks flow via gossipsub and verification runs normally.
- **Lesson**: Any fallback data path should produce the same UI state as the primary path.
  If the primary path produces "X/Y verified" stats, the fallback should too, even if the
  actual verification is less meaningful for trusted-source data.
- **Key file**: `gui/quantum-wallet/src/hooks/useRealtimeBlocks.ts`

---

## 📋 COMPLETE PHASE TRANSITION CHECKLIST (Updated Phase 21, Bug #15-17)

When transitioning to a NEW phase (e.g., Phase 20 → Phase 21), complete ALL items:

### Pre-Transition (Code Changes)
- [ ] 1. Add enum variant to `NetworkId` (lib.rs)
- [ ] 2. Update `as_str()` method (lib.rs)
- [ ] 3. Update `display_name()` method (lib.rs)
- [ ] 4. Update `from_str()` parser (lib.rs)
- [ ] 5. Update `Default` impl (lib.rs)
- [ ] 6. Update `default_api_port()` and `default_p2p_port()` (lib.rs)
- [ ] 7. Update `NetworkConfig::testnet()` (lib.rs) - genesis_hash, launch_time, version, chain_id
- [ ] 8. Update `from_network_id()` (lib.rs)
- [ ] 9. Update block_producer.rs `phase:` field
- [ ] 10. Update tests (lib.rs - chain_id, launch_time, from_network_id)
- [ ] 11. **Sed replace** ALL `testnet-phaseN` → `testnet-phaseN+1` in:
  - main.rs (~23 locations)
  - handlers.rs (~11 locations)
  - state_sync_api.rs, upgrade_verifier.rs
  - auto_cluster.rs, network_bridge.rs, zk_peer_height_proof.rs
- [ ] 12. **GENESIS_TIMESTAMP** - Update ALL 4 locations (Bug #10):
  - emission_controller.rs, handlers.rs, balance_consensus.rs, integrated_mining.rs
- [ ] 13. **LEGACY_FIXED_REWARD** - Verify 24-decimal format (Bug #11)
- [ ] 14. **record_daily_emission()** - Verify it's wired (Bug #12)

### Pre-Transition (Frontend)
- [ ] 15. Update `gui/quantum-wallet/src/libp2p/config.ts` NETWORK_ID + PROTOCOL_VERSION
- [ ] 16. Update decoder.ts, useRealtimeBlocks.ts, LibP2PContext.tsx fallbacks
- [ ] 17. Bump version in Cargo.toml

### Pre-Transition (Infrastructure — NEW from Bug #14!)
- [ ] 18. **STOP ALL old-phase nodes** on ALL servers (Alpha Docker, Gamma, Beta)
  ```bash
  # Alpha: Stop ALL Docker containers running old phase
  ssh root@161.35.219.10 "docker ps --format '{{.Names}}' | xargs -I{} docker stop {}"
  # Gamma: Stop service
  ssh root@109.205.176.60 "systemctl stop q-api-server"
  # Beta: Stop service
  systemctl stop q-api-server
  ```
- [ ] 19. **Remove old encryption keys** on ALL servers (Bug #15 — NEVER pre-generate!)
  ```bash
  # IMPORTANT: Only DELETE old keys. Do NOT generate new ones manually!
  # The server will auto-create proper-format keys on first boot.
  rm -f /opt/encryption-phase{OLD}.keys
  ssh root@109.205.176.60 "rm -f /opt/encryption-phase{OLD}.keys"
  # ❌ NEVER DO: openssl rand -hex 32 > /opt/encryption-phaseX.keys
  # ❌ This creates incompatible format → crash loop!
  ```

### Transition (Infrastructure) — Bugs #18-20 prevention!
- [ ] 20. **Update systemd service file on Beta** (Bug #18 — CRITICAL!):
  ```bash
  # Edit the service file — update ALL of these fields:
  nano /etc/systemd/system/q-api-server.service
  #   Description=Q-NarwhalKnight API Server - Phase X [Description]
  #   Q_DB_PATH=./data-mineX          # ✅ MUST be new data dir
  #   Q_NETWORK_ID=testnet-phaseX     # ✅ MUST match new phase
  #   Q_ENCRYPTION_KEYS_FILE=/opt/encryption-phaseX.keys  # ✅ NEW key file
  #   Q_ENCRYPTION_PASSPHRASE=Qnk-PhaseX-...              # ✅ NEW passphrase
  ```
- [ ] 20b. **Check and clean drop-in overrides** (Bug #19):
  ```bash
  ls /etc/systemd/system/q-api-server.service.d/ 2>/dev/null
  # Review each .conf file — remove any that shouldn't carry over
  # Common problematic overrides: Q_DISABLE_DEX=1, RUST_LOG overrides
  # Delete or update overrides as needed:
  # rm /etc/systemd/system/q-api-server.service.d/disable-dex.conf
  ```
- [ ] 20c. **Reload systemd daemon** (MUST be done before starting!):
  ```bash
  systemctl daemon-reload
  ```
- [ ] 21. Start Beta on new phase, verify blocks producing
  - If crash loop with `"Failed to open hot database"`: `rm -f /opt/encryption-phaseX.keys && systemctl restart` (Bug #15)
  - **CRITICAL: Verify RUNNING process environment** (Bug #20):
    ```bash
    systemctl start q-api-server
    sleep 5
    # Verify the PROCESS (not just systemd) has correct config:
    PID=$(pgrep -f "q-api-server --port" | head -1)
    tr '\0' '\n' < /proc/$PID/environ | grep -E "Q_DB_PATH|Q_NETWORK_ID"
    # ✅ Expected: Q_DB_PATH=./data-mineX  Q_NETWORK_ID=testnet-phaseX
    # ❌ If it shows old phase: kill -9 $PID && systemctl start q-api-server
    ```
  - Also check startup logs:
    ```bash
    journalctl -u q-api-server --since "1 minute ago" | grep -E "Network:|data-mine|testnet-phase"
    ```
- [ ] 22. **Update systemd service file on Gamma** (same as step 20):
  ```bash
  ssh root@109.205.176.60 "nano /etc/systemd/system/q-api-server.service"
  # Update: Description, Q_DB_PATH, Q_NETWORK_ID, encryption keys/passphrase
  ssh root@109.205.176.60 "ls /etc/systemd/system/q-api-server.service.d/ 2>/dev/null"
  # Clean drop-in overrides if needed
  ssh root@109.205.176.60 "systemctl daemon-reload"
  ```
- [ ] 23. Start Gamma on new phase, verify P2P sync from Beta
- [ ] 23b. **HA deploy does NOT work for phase transitions!**
  - `ha-deploy.sh` only deploys the binary — it does NOT update service files
  - Height decrease checks will fail (new data dir has fewer blocks)
  - Gamma can't sync from Beta if they're on different phases
  - **For phase transitions**: Update service files manually on each server, then restart
  - **After transition**: HA deploy works normally for subsequent binary updates
- [ ] 24. **Verify NO stale tokens/pools** (Bug #14 prevention):
  ```bash
  curl -s localhost:8080/api/v1/dex/supported-tokens | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))"
  # Expected: 2-4 (QUG, QUGUSD, VAULT, FORGE only)
  ```

### Post-Transition (Peer IDs + Deploy + Frontend)
- [ ] 25. **Bootstrap Peer IDs** - Get new PeerIDs from API, update 7 locations (Bug #13)
- [ ] 26. Build and deploy updated binary (with new peer IDs)
- [ ] 27. **Rebuild frontend** (Bug #16 — CRITICAL before users access site!):
  ```bash
  cd gui/quantum-wallet && npm run build
  # Verify the new JS bundle uses the correct phase:
  grep -o "testnet-phase[0-9]*" dist-final/assets/index-*.js | sort -u
  # Expected: testnet-phase21 (or whatever the new phase is)
  # ❌ If you see the OLD phase name, the config.ts update was not saved!
  ```
- [ ] 28. **Copy binary to downloads**:
  ```bash
  cp target/release/q-api-server gui/quantum-wallet/dist-final/downloads/q-api-server-v{VERSION}-beta
  ```
- [ ] 29. Verify emission stats showing on explorer
- [ ] 29b. **Frontend localStorage phase-aware cache clearing** (Bug #21):
  - Update `Dashboard.tsx` `currentPhase` constant to match new phase:
    ```typescript
    const currentPhase = 'testnet-phaseX';  // ✅ UPDATE THIS
    ```
  - This ensures stale balance caches from previous phase are auto-cleared
  - Rebuild frontend after this change!
- [ ] 30. Update CLAUDE.md with new PeerIDs, network ID, gossipsub topics
- [ ] 31. **If stale data found**: Call `POST /api/v1/admin/purge-phase-data` and restart
- [ ] 32. **Verify P2P block verification** (Bug #16 & #17 prevention):
  ```bash
  # Open browser, navigate to quillon.xyz, hard refresh (Ctrl+Shift+R)
  # Open Network Map modal (P2P button in bottom bar)
  # Expected: "Verified: X/X" (X > 0) within 30 seconds
  # ❌ If "Verified: 0/0" after 30 seconds:
  #   1. Check deployed JS uses correct phase: grep "testnet-phase" dist-final/assets/index-*.js
  #   2. Clear browser cache or hard refresh
  #   3. Check browser console for P2P debug logs
  ```

### Pre-Transition (3-Server Infrastructure Cleanup — Phase 24 Discovery!)

**All 3 servers (Alpha, Beta, Gamma) must be cleaned and configured BEFORE starting the new phase.**

- [ ] 33. **Server Alpha: Stop containers, delete old DBs and binaries**:
  ```bash
  ssh root@161.35.219.10 'bash -s' << 'EOF'
  # Stop and remove ALL containers
  docker stop $(docker ps -q) 2>/dev/null
  docker rm $(docker ps -aq) 2>/dev/null
  # Delete ALL old data directories (data-mine*)
  rm -rf /opt/orobit/shared/q-narwhalknight/data-mine*
  rm -rf /data/q-*
  # Delete old binaries
  rm -f /opt/orobit/shared/q-narwhalknight/q-api-server*
  # Remove old encryption keys
  rm -f /opt/encryption-phase*.keys /opt/encryption.keys
  EOF
  ```

- [ ] 34. **Server Alpha: Create Docker env file for new phase**:
  ```bash
  ssh root@161.35.219.10 'cat > /opt/orobit/shared/q-narwhalknight/phaseX.env << ENV
  Q_DB_PATH=./data-mineX
  Q_NETWORK_ID=testnet-phaseX
  Q_IS_VALIDATOR=true
  Q_P2P_PORT=9001
  Q_ENABLE_AI=0
  RUST_LOG=info
  Q_ALLOW_SOLO_MINING=true
  Q_TURBO_SYNC=1
  Q_BATCHED_WRITES=1
  Q_STATE_SYNC=1
  Q_PREFLIGHT_CHECK=1
  Q_ENCRYPTION_KEYS_FILE=/opt/encryption-phaseX.keys
  Q_ENCRYPTION_PASSPHRASE=Qnk-PhaseX-Description-2026-ServerAlpha-Canary-Key
  Q_EXTERNAL_ADDRESS=/ip4/161.35.219.10/tcp/9001
  Q_FILTER_DOCKER_ADDRESSES=true
  Q_GOSSIPSUB_HEARTBEAT_MS=50
  Q_GOSSIPSUB_FLOOD_PUBLISH=true
  ENV'
  ```

- [ ] 35. **Server Gamma: Update service file for new phase**:
  ```bash
  ssh root@109.205.176.60 'sed -i \
    -e "s/data-mine[0-9]*/data-mineX/g" \
    -e "s/testnet-phase[0-9]*/testnet-phaseX/g" \
    -e "s/encryption-phase[0-9]*.keys/encryption-phaseX.keys/g" \
    /etc/systemd/system/q-api-server.service && systemctl daemon-reload'
  # Also remove old encryption keys:
  ssh root@109.205.176.60 'rm -f /opt/encryption-phase*.keys'
  ```

- [ ] 36. **Server Beta: Update service file for new phase** (already in step 20, but reinforced):
  ```bash
  # Verify ALL 4 env vars are updated:
  grep -E "Q_DB_PATH|Q_NETWORK_ID|ENCRYPTION_KEYS|ENCRYPTION_PASS" /etc/systemd/system/q-api-server.service
  # Expected: data-mineX, testnet-phaseX, encryption-phaseX.keys, PhaseX passphrase
  systemctl daemon-reload
  rm -f /opt/encryption-phase*.keys
  ```

**WHY THIS IS CRITICAL**: Phase transitions require ALL 3 servers to be on the same phase simultaneously.
If Alpha still has old containers or Gamma has stale service config, P2P will fail or old-phase blocks
will be gossiped to the new network, causing confusion. Clean ALL servers BEFORE starting any of them.

---

## MAINNET 2026.1 FAILURE ANALYSIS (Feb 15-18, 2026)

**Result**: 76% success rate (25/33 checklist items passed). Four critical gaps found.

### Score Card

| # | Checklist Item | Status | Impact if Failed |
|---|---------------|--------|------------------|
| 1 | Enum variant added | PASS | Network ID unrecognized |
| 2 | as_str() updated | PASS | Wrong gossipsub topics |
| 3 | display_name() updated | PASS | Cosmetic only |
| 4 | from_str() parser | PASS | Env var parsing fails |
| 5 | Default impl | PASS | Wrong default network |
| 6 | Port methods | PASS | Wrong ports |
| 7 | NetworkConfig | PASS | Wrong genesis config |
| 8 | from_network_id() | PASS | Config mismatch |
| 9 | Block producer phase | PASS | Wrong blocks |
| 10 | Env var priority | PASS | Q_NETWORK_ID ignored |
| 11 | Fallback values | PASS | Wrong topic fallbacks |
| 12 | Service file | PASS | Stale data dir |
| 13 | Phase modal | PASS | No user notification |
| 14 | Encryption keys | PASS | Crash loop |
| 15 | Genesis checkpoint | PASS | Chain forks |
| 16 | Block chain validation | PASS | Accept wrong chain |
| 17 | Peer genesis validation | PASS | Wrong peers |
| 18 | Service file update | PASS | Stale config |
| 19 | Drop-in overrides | PASS | Hidden env vars |
| 20 | Process env verification | PASS | Stale process |
| 21 | Frontend config.ts | PASS | Wrong P2P topics |
| 22 | GENESIS_TIMESTAMP (4 locs) | PASS | Wrong emission |
| 23 | Bootstrap peer IDs | PASS | P2P connection fails |
| 24 | Encryption auto-create | PASS | Crash loop |
| 25 | Frontend rebuild | PASS | Stale JS bundle |
| 26-28 | Deploy + downloads | PASS | - |
| 29 | Wallet purge | **FAIL** | Stale balances persist |
| 30 | Balance_update contamination | **FAIL** | Cross-network balance sync |
| 31 | Emission controller rebuild | **FAIL** | Cumulative emission wrong |
| 32 | CollateralVault reset | **FAIL** | minted_qugusd not cleared |
| 33-36 | 3-server cleanup | PASS | Cross-phase pollution |

### 4 Critical Gaps NOT in Previous Checklist

#### Gap 1: Wallet Balance Purge Too Narrow
- **What happened**: wallet_balances purged by prefix scan, but scan only caught
  wallets with the standard prefix format. Some balance entries stored with
  alternate key formats survived the purge.
- **Impact**: Users saw phantom balances from mainnet2026 on mainnet2026.1
- **Fix needed**: Add checklist item to verify balance count is exactly 0 after purge
- **Checklist addition**: Item 37 - "Verify zero balances after purge"

#### Gap 2: Balance Update P2P Contamination
- **What happened**: HTTP bootstrap fallback fetched `/api/v1/node/status` from
  peers on different networks without checking network_id in the response.
  Delta (mainnet2026.2) synced 320K blocks from Beta (mainnet2026.1).
- **Impact**: Wrong-network data imported via HTTP, bypassing gossipsub topic isolation
- **Fix needed**: Add network_id to status endpoint + validate in HTTP fallback
- **Checklist addition**: Item 38 - "HTTP bootstrap validates network_id"
- **Already fixed**: handlers.rs (network_id in response) + main.rs (validation)

#### Gap 3: Emission Controller State Not Reset
- **What happened**: EmissionController preserves cumulative emission tracking
  across phase transitions. Starting a new phase with existing emission state
  causes the budget-based error correction to think it's over/under-emitted.
- **Impact**: Reward calculations use stale era/emission data from old phase
- **Fix needed**: Verify EmissionController state is fresh on new phase genesis
- **Checklist addition**: Item 39 - "EmissionController state fresh on new genesis"

#### Gap 4: CollateralVault.minted_qugusd Not Cleared
- **What happened**: CollateralVault tracks `minted_qugusd` in-memory. On phase
  transition, the vault is reconstructed but `minted_qugusd` may be restored
  from persisted state, creating phantom QUGUSD supply tracking.
- **Impact**: Collateral ratio calculations use stale data
- **Fix needed**: Verify vault state is fresh on new phase data directory
- **Checklist addition**: Item 40 - "CollateralVault state fresh on new data dir"

### New Checklist Items (37-40)

Add these to the pre-transition checklist for all future phase transitions:

- [ ] **37. Verify zero balances after purge**: After wallet purge, run `balance_count()` and assert == 0
- [ ] **38. HTTP bootstrap validates network_id**: Verify `/api/v1/node/status` returns `network_id` and bootstrap code checks it
- [ ] **39. EmissionController state fresh on new genesis**: Verify `total_supply == 0` and `days_tracked == 0` after transition
- [ ] **40. CollateralVault state fresh on new data dir**: Verify `minted_qugusd == 0` and `qug_price_usd == default` after transition

---

## 🔥 MAINNET 2026.1.1 REHEARSAL BUGS (Feb 18, 2026) - SIX MORE BUGS FOUND!

**Updated**: 2026-02-18 after mainnet2026.1.1 rehearsal revealed block storage, compression,
and deployment bugs that caused a 4-day block production stall.

The mainnet2026.1.1 rehearsal (4-day dry run before Feb 22 mainnet2026.2 launch) exposed
**SIX ADDITIONAL BUGS** (#22-27) related to block storage format, error recovery, genesis
timestamp filtering, deployment procedures, and chain convergence.

### Bug #22: LZ4 Compress/Decompress Parameter Mismatch (Block Corruption)

- **Symptom**: ALL blocks fail to decompress after being saved to RocksDB → block production stalls
- **Impact**: CATASTROPHIC — every block saved becomes unreadable, stalling the entire chain
- **Root cause**: `precompressed_storage.rs` line 94:
  ```rust
  lz4::block::compress(raw, None, true)  // true = prepend 4-byte size prefix
  ```
  But line 143:
  ```rust
  lz4::block::decompress(&self.data, Some(self.original_size as i32))
  // Some(n) means "data does NOT have a size prefix"
  ```
  So decompress tries to read the 4-byte size prefix as compressed LZ4 data → corruption.
- **The subtle trap**: Both compress and decompress "succeed" (no error returned), but the
  decompressed data is garbage because the size prefix bytes get interpreted as LZ4 stream data.
- **Fix**: Changed block storage from LZ4 to QRAW (no app-level compression):
  ```rust
  // block_writer.rs: Use QRAW format (no app-level compression)
  let compressed = PrecompressedBlock::compress(&slim_bytes, CompressionAlgorithm::None)
      .context("Failed to wrap QBlock in QRAW format")?;
  ```
  RocksDB CF_BLOCKS has its own transparent LZ4 compression, so no data size increase.
- **Pattern**: When using the `lz4` crate, `prepend_size` in compress MUST match `uncompressed_size` in decompress:
  - `compress(data, None, true)` → `decompress(data, None)` (reads size from prefix)
  - `compress(data, None, false)` → `decompress(data, Some(size))` (uses explicit size)
- **Lesson**: NEVER double-compress (app-level LZ4 + RocksDB LZ4). Use one or the other.
  RocksDB compression is transparent and reliable. App-level compression adds footgun risk.
- **Key files**: `crates/q-storage/src/precompressed_storage.rs`, `crates/q-storage/src/block_writer.rs`

### Bug #23: lockfree_producer sync_from_storage Error Propagation

- **Symptom**: Block production stalls silently; duplicate nonce floods in logs
- **Impact**: CRITICAL — blocks can't advance; miners waste hash power on stale heights
- **Root cause**: `lockfree_producer.rs` `sync_from_storage()` uses `?` operator on block reads:
  ```rust
  let block = storage.get_qblock_by_height(height)?;  // If decompress fails → abort entire sync
  ```
  When Bug #22 causes decompression failures, `sync_from_storage()` aborts completely.
  This leaves `pool_last_produced_height` at 0 (never updated), so the pool's duplicate
  check rejects all new blocks as "already produced at this height".
- **How it cascades**:
  1. Block produced at height N → saved (corrupted by Bug #22)
  2. Next `sync_from_storage()` call → tries to read block N → LZ4 error → returns Err
  3. `pool_last_produced_height` stays at 0 (or stale value)
  4. Pool rejects next block as duplicate → "Block N already produced by pool"
  5. Mining submissions flood with "Duplicate nonce" warnings → no new blocks ever produced
- **Fix**: Catch decompression errors gracefully and fall back to height-only mode:
  ```rust
  match storage.get_qblock_by_height(height) {
      Ok(block) => { /* normal sync */ },
      Err(e) => {
          warn!("⚠️ Failed to read block at height {}: {}", height, e);
          // Fall through to height-only mode instead of aborting
      }
  }
  ```
- **Lesson**: Storage read errors in non-critical paths must NEVER abort with `?`.
  Use graceful degradation (height-only mode) when full block reads fail.
- **Key file**: `crates/q-api-server/src/lockfree_producer.rs` (sync_from_storage)

### Bug #24: save_qblock() Genesis Timestamp Filter Rejects Rehearsal Blocks

- **Symptom**: All blocks rejected silently during save; chain appears to advance but blocks don't persist
- **Impact**: HIGH — blocks mined but not saved to DB, lost on restart
- **Root cause**: `save_qblock()` has a genesis timestamp filter:
  ```rust
  if block.header.timestamp < GENESIS_TIMESTAMP {
      warn!("Rejected block: timestamp before genesis");
      return Ok(());  // Silent rejection!
  }
  ```
  The `GENESIS_TIMESTAMP` was set to the mainnet2026.2 genesis (Feb 22 2026), but the
  rehearsal network (mainnet2026.1.1) had timestamps from Feb 18 2026 — 4 days before
  the hardcoded genesis timestamp.
- **Impact**: Every rehearsal block was silently rejected because its timestamp was
  "before genesis". The block appeared to be produced (in-memory) but never persisted to RocksDB.
- **Fix**: Either:
  1. Set GENESIS_TIMESTAMP appropriately for the rehearsal network, OR
  2. Make the filter network-aware (read genesis timestamp from config, not hardcoded constant)
- **Lesson**: Genesis timestamp filters MUST match the actual network being run.
  Hardcoded future timestamps will reject all blocks on rehearsal/canary networks.
- **Key files**: `crates/q-storage/src/block_writer.rs` (save_qblock timestamp filter)

### Bug #25: systemd Restart=always Prevents Raw Kill Deployments

- **Symptom**: After `kill -9 PID`, process immediately restarts with old binary
- **Impact**: MEDIUM — deployment procedure fails, old binary keeps running
- **Root cause**: systemd service file has `Restart=always` + `RestartSec=10`. When the
  process is killed (even with SIGKILL), systemd treats it as a crash and restarts it
  within 10 seconds — with the OLD binary still in place.
- **How it manifests during deployment**:
  1. Build new binary → overwrites `target/release/q-api-server`
  2. `kill -9 $(pgrep q-api-server)` → process dies
  3. 10 seconds later → systemd restarts the service → uses the NEW binary (if in-place)
  4. BUT if you haven't finished copying/replacing the binary, it restarts with the OLD one
  5. Multiple kill attempts create a frantic race condition
- **Correct procedure**:
  ```bash
  # ✅ CORRECT: Use systemctl to stop (waits for clean shutdown, prevents restart)
  systemctl stop q-api-server

  # ✅ Then replace binary, then start
  cp target/release/q-api-server /path/to/binary
  systemctl start q-api-server

  # ❌ WRONG: Raw kill triggers immediate systemd restart
  kill -9 $(pgrep q-api-server)  # systemd will restart it!
  ```
- **Emergency stop** (if systemctl stop hangs):
  ```bash
  systemctl mask q-api-server    # Prevent restart
  kill -9 $(pgrep q-api-server)  # Now safe to kill
  # ... do your deployment ...
  systemctl unmask q-api-server  # Re-enable
  systemctl start q-api-server   # Start with new binary
  ```
- **Lesson**: ALWAYS use `systemctl stop` for managed services, NEVER raw kill.
- **Key file**: `/etc/systemd/system/q-api-server.service` (Restart=always)

### Bug #26: Independent Chain Forks from Staggered Server Starts

- **Symptom**: Two servers show `peers=1` but `effective_solo=true`; heights diverge
- **Impact**: HIGH — servers produce independent chains that can never merge
- **Root cause**: When multiple servers start at different times with fresh data directories:
  1. Server A starts first → mines genesis block → height 1, 2, 3...
  2. Server B starts 30 min later → mines ITS OWN genesis block → height 1, 2, 3...
  3. Both discover each other via DHT → `peers=1` ✅
  4. But they have DIFFERENT genesis blocks → can't merge chains
  5. Both continue mining independently → heights diverge (A=85, B=167)
  6. `effective_solo=true` on both — they know about each other but can't use each other's blocks
- **This is different from Bug #6** (which is about sync failure): Here, sync was never
  attempted because both nodes were mining solo before discovering each other.
- **Fix**: Correct server start sequence:
  ```bash
  # 1. Start bootstrap server FIRST
  systemctl start q-api-server  # on Beta

  # 2. Wait for bootstrap to mine at least 1 block
  sleep 30
  journalctl -u q-api-server --since "30 seconds ago" | grep "height"

  # 3. ONLY THEN start secondary servers
  ssh root@109.205.176.60 "systemctl start q-api-server"  # Gamma

  # 4. Verify Gamma syncs FROM Beta (not mining its own genesis)
  ssh root@109.205.176.60 "journalctl -u q-api-server --since '1 min ago' | grep 'turbo.*sync\|Gossipsub BLOCK'"
  ```
- **Detection**: If you see `effective_solo=true` on a node that reports `peers > 0`,
  the nodes are on different genesis chains and CANNOT converge.
- **Recovery**: Wipe data on the secondary node and restart (it will sync from bootstrap):
  ```bash
  ssh root@SECONDARY "systemctl stop q-api-server && rm -rf ./data-NETWORK/* && systemctl start q-api-server"
  ```
- **Lesson**: NEVER start multiple servers simultaneously with fresh data. Always start
  bootstrap first, verify it's producing blocks, THEN start secondary nodes.

### Bug #27: Binary Version Mismatch After Build

- **Symptom**: `q-api-server --version` reports old version (e.g., 7.3.3) despite Cargo.toml saying 7.3.5
- **Impact**: LOW but confusing — makes deployment verification unreliable
- **Root cause**: Multiple scenarios:
  1. Cargo.toml version bumped AFTER the build started → binary has old version
  2. Concurrent `cargo build` from another session holds the build lock → your build waits
     and may use stale artifacts
  3. Build completes but a DIFFERENT binary (from earlier build) is in target/release/
- **Detection**:
  ```bash
  # ALWAYS verify after build:
  target/release/q-api-server --version
  # Compare with:
  grep '^version' Cargo.toml | head -1
  # They MUST match!
  ```
- **Fix**: Always bump version BEFORE building, verify after:
  ```bash
  # 1. Bump version in Cargo.toml
  # 2. Build
  cargo build --release --package q-api-server
  # 3. Verify version matches
  target/release/q-api-server --version | grep "$(grep '^version' Cargo.toml | head -1 | grep -o '[0-9.]*')"
  ```
- **Concurrent build fix**: Check for other cargo processes first:
  ```bash
  pgrep -f "cargo" && echo "⚠️ Another cargo process running - wait for it to finish!"
  ```
- **Lesson**: NEVER trust the binary version without verifying. Build lock contention from
  parallel Claude Code sessions is common and causes stale artifacts.

---

### New Checklist Items (41-48) — Mainnet Rehearsal Additions

Add these to the phase transition checklist:

- [ ] **41. Block storage format compatible**: Verify `PrecompressedBlock::compress()` and `decompress()` roundtrip correctly. Test: `compress(data) → decompress(result) == data`. If using LZ4, verify `prepend_size` parameter matches decompress expectations.
- [ ] **42. lockfree_producer graceful degradation**: Verify `sync_from_storage()` catches block read errors and falls back to height-only mode instead of aborting with `?`.
- [ ] **43. save_qblock() timestamp filter matches network**: Verify `GENESIS_TIMESTAMP` in block_writer.rs matches the actual genesis timestamp of the network being launched (not a future mainnet date on a rehearsal).
- [ ] **44. Use systemctl for service management**: NEVER use `kill -9` for deployment. ALWAYS use `systemctl stop/start`. Document in deployment runbook.
- [ ] **45. Sequential server start order**: Start bootstrap (Beta) first → wait for blocks → then start Gamma → verify Gamma syncs FROM Beta (not mining independently).
- [ ] **46. Verify binary version after build**: After every `cargo build`, run `target/release/q-api-server --version` and confirm it matches `Cargo.toml`. If mismatch, rebuild.
- [ ] **47. No concurrent cargo builds**: Before building, check `pgrep -f cargo`. Wait for any existing build to finish before starting a new one.
- [ ] **48. Data wipe preserves identity key**: When wiping data directories, preserve `libp2p_identity.key` if you want to keep the same PeerID:
  ```bash
  # ✅ CORRECT: Preserve identity
  cp data-NETWORK/libp2p_identity.key /tmp/libp2p_identity.key.backup
  rm -rf data-NETWORK/*
  mv /tmp/libp2p_identity.key.backup data-NETWORK/libp2p_identity.key

  # ✅ ALSO CORRECT: Fresh identity (if you want new PeerID)
  rm -rf data-NETWORK/*
  # Server will generate new identity on start
  # ⚠️ But then you MUST update all hardcoded PeerID references!
  ```

---

### Updated Complete Checklist Summary (Items 1-48)

| # | Category | Item | Risk if Missed |
|---|----------|------|----------------|
| 1-8 | NetworkId Code | Enum, parsers, config, ports | Network isolation |
| 9 | Block Producer | Phase/network_id in blocks | Topic mismatch |
| 10-11 | Env Vars | Priority + fallback values | Wrong network |
| 12-13 | Service + UI | Systemd file, phase modal | Stale config |
| 14 | Encryption | Keys reset (auto-create only!) | Crash loop |
| 15-17 | Genesis | Checkpoint, chain validation, peer validation | Chain forks |
| 18-20 | Deployment | Service file, drop-ins, process env | Hidden stale config |
| 21 | Frontend P2P | Browser gossipsub topics | No P2P blocks |
| 22-23 | Genesis Timestamps | 4 locations + bootstrap peer IDs | Wrong emission/P2P fail |
| 24-25 | Encryption + Frontend | Auto-create keys, rebuild JS | Crash loop, stale UI |
| 26-28 | Deploy | Binary + downloads copy | Missing downloads |
| 29 | Phase Cache | Frontend localStorage clearing | Stale balance display |
| 30-32 | Peer IDs + Verification | CLAUDE.md, purge, P2P verify | Stale docs, stale data |
| 33-36 | 3-Server Cleanup | Alpha/Beta/Gamma reset | Cross-phase pollution |
| 37-40 | State Reset | Balances, HTTP, emission, vault | Phantom data |
| **41** | **Storage Format** | **LZ4/QRAW roundtrip test** | **ALL blocks corrupted** |
| **42** | **Error Recovery** | **sync_from_storage graceful** | **Block production stall** |
| **43** | **Timestamp Filter** | **save_qblock() genesis check** | **Silent block rejection** |
| **44** | **Service Mgmt** | **systemctl not kill** | **Race condition on deploy** |
| **45** | **Start Order** | **Bootstrap first, then secondaries** | **Independent chain forks** |
| **46** | **Version Check** | **Verify binary version post-build** | **Deploy wrong version** |
| **47** | **Build Safety** | **No concurrent cargo builds** | **Stale artifacts** |
| **48** | **Data Wipe** | **Preserve or regenerate identity** | **PeerID mismatch** |
| **49** | **Batch Turbo Genesis** | **save_qblocks_batch_turbo genesis filter** | **Pre-genesis contamination** |
| **50** | **Rogue Peers** | **Pre-genesis nodes on future network IDs** | **Canary contamination** |
| **51** | **Running Deleted Binary** | **Check /proc/PID/exe after cargo build** | **Old binary runs silently** |
| **52** | **ALL servers updated** | **SCP new binary to ALL servers, not just Beta** | **Mixed-version network** |
| **53** | **Genesis timestamp = now** | **Canary genesis must match actual network ts** | **All blocks rejected silently** |

---

## 🔥 MAINNET 2026.2 PRE-LAUNCH BUGS (Feb 18, 2026) — THREE MORE BUGS!

**Updated**: 2026-02-18 during mainnet2026.2 canary soak (Delta) before Feb 22 launch.

These bugs were discovered when running the Delta canary node on `mainnet2026.2` while
Beta/Gamma ran `mainnet2026.1.1`. They could corrupt the Feb 22 launch if not fixed.

### Bug #28: save_qblocks_batch_turbo() Missing Genesis Timestamp Filter

- **Symptom**: Canary node (Delta) syncs thousands of pre-genesis blocks from rogue peer;
  chain height shows 133 blocks with timestamps from the WRONG network/time period
- **Impact**: HIGH — canary database contaminated with invalid blocks; requires full wipe
- **Root cause**: `save_qblocks_batch_turbo()` in `crates/q-storage/src/lib.rs` does NOT
  check the genesis timestamp before writing blocks. It bypasses the filter that exists
  in `save_qblock()`:
  ```rust
  // save_qblock() ✅ HAS this filter:
  if block.header.timestamp > 0 && block.header.timestamp < genesis_ts {
      warn!("Rejecting pre-genesis block");
      return Ok(());
  }

  // save_qblocks_batch_turbo() ❌ MISSING this filter entirely
  // Pre-genesis blocks written directly to RocksDB without check
  ```
- **How it was triggered**: Rogue peer published on `/qnk/mainnet2026.2/peer-heights`
  with height 311,755. Delta registered this peer and initiated turbo sync, calling
  `save_qblocks_batch_turbo()` to save the downloaded blocks. All 133 blocks stored
  despite having timestamps before the Feb 22 genesis.
- **Fix**: Add genesis filter at start of `save_qblocks_batch_turbo()`:
  ```rust
  let genesis_ts = crate::balance_consensus::active_genesis_timestamp();
  let filtered: Vec<_> = blocks.iter()
      .filter(|b| b.header.timestamp == 0 || b.header.timestamp >= genesis_ts)
      .cloned()
      .collect();
  if filtered.is_empty() {
      warn!("🧹 [GENESIS FILTER] Batch of {} blocks all pre-genesis, skipping", blocks.len());
      return Ok(());
  }
  ```
- **Lesson**: ANY code path that writes blocks to storage MUST apply the genesis filter.
  This includes batch writes, turbo sync, warp sync, and checkpoint imports.
- **Key file**: `crates/q-storage/src/lib.rs` (`save_qblocks_batch_turbo`)

### Bug #29: Pre-Genesis Rogue Nodes Contaminating Canary Nodes

- **Symptom**: Unknown peer `12D3KooWDspLtKpQwSxqTNZXKAFZcRdSXAVxPEyrCLjLDyMVZday`
  publishing on `/qnk/mainnet2026.2/peer-heights` with height 311,755 days before launch.
  Delta registered this peer and attempted to sync 311,755 blocks — would take 481 hours.
  Block production DISABLED: "311,622 behind (local=133, net=311,755)"
- **Impact**: HIGH — canary node completely locked out of block production; can't test
  pre-launch configuration; risks contaminating the official launch database
- **Root cause**: Someone "guessed" the future network configuration from documentation
  (CLAUDE.md, deployment scripts, or the Delta service file which is publicly visible
  via leaked/compromised access). They ran a node with:
  - `Q_NETWORK_ID=mainnet2026.2`
  - `Q_DB_PATH=./data-mainnet2026.2`
  - Modified or old binary without genesis timestamp guard
  This let them start producing blocks in "mainnet2026.2" 4 days before the real launch.
  Their blocks have Feb 18 timestamps (before Feb 22 genesis), so they're technically invalid.
- **Why it's not catastrophic for the real launch**:
  1. `save_qblock()` genesis filter rejects their blocks (timestamp < 1771761600)
  2. The real genesis block (height 0, timestamp 1771761600) will be different from their
     genesis block → genesis checkpoint validation rejects their chain
  3. If Bug #28 is fixed, `save_qblocks_batch_turbo()` also rejects their blocks
- **Impact on the canary**: The rogue peer's height announcement caused:
  1. Delta's `block_production_v2.rs` gap check to fire: "311,622 behind → disabled"
  2. Endless "Waiting for blocks... 0% | 133/311755 | 0 b/s | ETA: 481h" log spam
  3. Turbo sync repeatedly failing: "no peers have blocks we need" (peer refuses actual sync)
  4. 133 corrupt LZ4 blocks from the PREVIOUS binary already in the database
- **Fix for immediate impact**: Stop Delta, wipe `data-mainnet2026.2/`, update binary,
  add pre-genesis peer registration guard (don't add peer to registry if peer.height is
  announced but we're before genesis timestamp), restart Delta
- **Long-term fix (for official launch)**: The genesis timestamp filter (Bug #28) prevents
  contamination. Additionally, consider rate-limiting peer height registrations and adding
  sanity check: "if now() < genesis_ts AND peer announces height > 0, log warning but
  don't initiate sync"
- **Lesson**: Never publicly document future network IDs or database paths until launch day.
  Consider using non-obvious config names for canary testing.
- **Also discovered**: Block production disabled when gap > 1000. Rogue peer at 311,755
  vs local 133 = gap of 311,622 → production disabled for DAYS until the peer disappears.
  The gap check should also be genesis-aware: if now() < genesis_ts, don't disable production
  based on rogue peer heights.

### Bug #30: Running Deleted Binary After cargo build (Canary Missed Fix)

- **Symptom**: Gamma received the new binary (21:03) but the LZ4 fix was applied at 21:05.
  Delta had an OLD binary (`q-api-server-v7.3.1`) that was never updated when Beta and Gamma
  got the LZ4 fix. Delta ran with the broken binary for hours, creating 133 corrupt blocks.
- **Impact**: MEDIUM — affects only nodes not included in the rolling deploy pipeline
- **Root cause**: The ha-deploy.sh pipeline covers Beta and Gamma but NOT Delta (canary).
  Delta's binary (`/opt/orobit/shared/q-narwhalknight/q-api-server-v7.3.1`) was a manually
  placed pre-release binary and not updated when fixes were applied to the production build.
  Additionally, on Beta itself: `cargo build` ran at 21:05, replacing the binary on disk,
  but the process started at 21:00:44 still used the OLD inode (deleted from directory).
  `/proc/PID/exe` showed `(deleted)` — the running process was the pre-fix binary.
- **Detection**:
  ```bash
  readlink /proc/$(pgrep -f q-api-server | head -1)/exe
  # If output ends with "(deleted)" → running pre-fix binary, restart NOW
  ```
- **Fix**:
  ```bash
  # After EVERY cargo build:
  readlink /proc/$(pgrep -f q-api-server | head -1)/exe
  # Must NOT show "(deleted)"
  # If deleted: systemctl restart q-api-server
  ```
- **For canary/non-pipeline nodes (Delta)**: Add to deployment runbook:
  ```bash
  # After building new binary, ALWAYS update ALL servers:
  scp target/release/q-api-server root@5.79.79.158:/opt/orobit/shared/q-narwhalknight/q-api-server.new
  ssh root@5.79.79.158 "systemctl stop q-api-server && mv q-api-server.new q-api-server-v7.3.6 && systemctl start q-api-server"
  ```
- **Lesson**: EVERY server in the network must receive the same binary build. Having one
  server on an old binary (even a canary) creates mixed-version behavior that is hard to debug.

### Bug #31: Pool Topics Using display_name() Instead of as_str() (Found 2026-02-19)

- **Severity**: MEDIUM (pool P2P broken, mining pool coordination fails)
- **Location**: `crates/q-api-server/src/main.rs:2037`
- **Problem**: `PoolTopics::new(&turbo_network_id.display_name())` used `display_name()` (e.g., "Q-NarwhalKnight Mainnet 2026.1.1 (Rehearsal)") instead of `as_str()` (e.g., "mainnet2026.1.1") for gossipsub pool topic construction
- **Symptom**: Pool messages rejected: `REJECTED message from wrong network: topic='/qnk/Q-NarwhalKnight Mainnet 2026.1.1 (Rehearsal)/pool/pplns-state'`
- **Fix**: Changed to `PoolTopics::new(turbo_network_id.as_str())`
- **Prevention**: Gossipsub topics ALWAYS use `as_str()` (machine format), NEVER `display_name()` (human-readable format). Add to checklist.

### Bug #32: REHEARSAL_GENESIS_TIMESTAMP Off By 1 Year (Found 2026-02-19)

- **Severity**: HIGH (emission controller calculates wrong targets)
- **Location**: `crates/q-storage/src/emission_controller.rs:74`
- **Problem**: `REHEARSAL_GENESIS_TIMESTAMP` was 1739836800 (Feb 18, 2025) but comment said Feb 18, 2026. Value was exactly 1 year (31,536,000 seconds) too early.
- **Symptom**: Emission correction showed -99.97% error, target=2,633,414 QUG when only 977 QUG had been mined in a few days on the rehearsal chain
- **Fix**: Changed to 1771372800 (Feb 18, 2026 00:00 UTC)
- **Prevention**: ALWAYS verify Unix timestamps with a converter tool. Never trust comments alone.

### Bug #33: Incremental Build Cache Serving Stale Code (Found 2026-02-19)

- **Severity**: HIGH (binary doesn't match source code)
- **Location**: `crates/q-api-server/src/lib.rs` — lib.rs had `1700000000` hardcoded, was fixed to `active_genesis_timestamp()` in source, but cargo incremental build cached the old compiled version
- **Symptom**: Two emission controllers running simultaneously with different genesis timestamps. Source code was correct but binary contained old code.
- **Fix**: `cargo clean --package q-api-server` + full rebuild
- **Prevention**: After changing constants, run `cargo clean --package <crate>` before building. For critical deployments, always clean build.

### Bug #34: Encryption Keys Are Auto-Generated (Checklist Correction)

- **Severity**: LOW (documentation only)
- **Problem**: CLAUDE.md launch procedure included manual key generation steps (`dd if=/dev/urandom ...`) but the node auto-generates encryption keys on startup in `data-{network}/encryption.keys`
- **Fix**: Remove manual key generation steps from launch procedure. Encryption keys are managed automatically by the node.

### Bug #35: Persisted EmissionController State Contains Stale Genesis Timestamp (Found 2026-02-19)

- **Severity**: HIGH (persisted state overwrites runtime fix on every restart)
- **Location**: `crates/q-storage/src/balance_consensus.rs` (`restore_emission_state()`)
- **Problem**: EmissionController is serialized to RocksDB via serde_json including the
  `genesis_timestamp` field. When restored on restart, the old wrong timestamp (`1739836800`)
  overwrote the corrected constant. Fixing Bug #29 had NO EFFECT because the persisted
  state kept restoring the old value.
- **Fix**: Added `set_genesis_timestamp()` method to EmissionController. Called after
  `restore_from_bytes()` in `balance_consensus.rs` to override with current correct value.
- **Files**:
  - `crates/q-storage/src/emission_controller.rs` (added `set_genesis_timestamp()`)
  - `crates/q-storage/src/balance_consensus.rs` (call after restore)
- **Prevention**: Serialized state may contain hardcoded constants that were wrong when persisted.
  After deserializing any state, re-apply current runtime values for constants that may have
  been fixed since the state was saved. Never trust persisted constants — always override.

### Bug #36: Hardcoded Genesis Timestamp in lib.rs (Two Instances) (Found 2026-02-19)

- **Severity**: HIGH (block producer used wrong genesis, emission engine A always wrong)
- **Location**: `crates/q-api-server/src/lib.rs` (lines ~2223 and ~3451)
- **Problem**: TWO hardcoded `genesis_timestamp = 1700000000` values from testnet Phase 8
  (Nov 14, 2023). These were never updated when `active_genesis_timestamp()` was introduced.
  Result: TWO emission engines running simultaneously — Engine A (lib.rs, wrong genesis)
  and Engine B (main.rs, correct genesis). The block producer pool captured Engine A's
  reference before main.rs could replace it, so actual block rewards used the wrong engine.
- **Symptom**: Two emission lines in logs: one with `target=5,947,334 QUG` (Engine A,
  genesis=Nov 2023) and one with `target=2,633,660 QUG` (Engine B, wrong rehearsal genesis).
- **Fix**: Replaced both hardcoded values with `q_storage::balance_consensus::active_genesis_timestamp()`
- **Prevention**: Search for ALL instances of hardcoded constants (`grep -rn "1700000000"`)
  before considering a fix complete. Dual-engine architectures are especially dangerous.

### Bug #37: Stale Bootstrap Peer IDs After Identity Regeneration (Found 2026-02-22)

- **Severity**: CRITICAL (external nodes CANNOT connect to network)
- **Locations** (ALL must be updated simultaneously):
  - `crates/q-network/src/unified_network_manager.rs:82-91` — `HARDCODED_BOOTSTRAP_PEERS`
  - `crates/q-network/src/unified_network_manager.rs:104` — `HARDCODED_BOOTSTRAP_PEER`
  - `crates/q-api-server/src/main.rs:197-202` — `is_allowed_balance_update_origin()`
  - `crates/q-storage/src/turbo_sync.rs:2348` — `BOOTSTRAP_PEER` constant
  - `crates/q-types/src/lib.rs:4190-4198` — `NetworkConfig::testnet()` bootstrap_peers
  - `crates/q-types/src/lib.rs:4911-4913` — test assertions for peer IDs
  - `gui/quantum-wallet/src/libp2p/config.ts:30` — frontend WebSocket bootstrap
  - `gui/quantum-wallet/src/libp2p/torConfig.ts` — Tor bridge peer IDs
  - `CLAUDE.md` — documentation bootstrap peer IDs section
- **Problem**: When servers regenerate their libp2p identity (e.g., new data directory,
  network ID change), the peer ID changes. But 6+ files still contain the OLD peer IDs.
  libp2p connections fail because the peer ID in the multiaddress doesn't match the actual
  identity. Nodes can sometimes connect briefly via dynamic discovery (Q_BOOTSTRAP_URL)
  but connections drop because the DHT caches the wrong peer ID.
- **Symptom**: User node shows "Peer 12D3KooW... is NOT connected and has 0 cached addresses"
  and "Cannot dial peer without knowing its address" even though turbo sync requests are
  briefly sent. Tried from HK, Singapore, and US — all fail identically.
- **Fix**: Updated ALL 6 source files with current peer IDs fetched via:
  ```bash
  curl -s http://185.182.185.227:8080/api/v1/status | jq -r '.data.peer_id'  # Beta
  curl -s http://109.205.176.60:8080/api/v1/status | jq -r '.data.peer_id'   # Gamma
  curl -s http://5.79.79.158:8080/api/v1/status | jq -r '.data.peer_id'      # Delta
  ```
- **Prevention Checklist Item**: After ANY network/data directory change:
  1. Fetch current peer IDs from ALL servers using the curl commands above
  2. `grep -rn "12D3KooW" crates/ gui/quantum-wallet/src/libp2p/` to find ALL occurrences
  3. Update every occurrence to match the LIVE peer IDs
  4. Rebuild, deploy, and verify new nodes can connect

---

## 🗓️ MAINNET 2026.2 LAUNCH CHECKLIST (Feb 22, 2026)

This is the complete runbook for the Feb 22 12:00 UTC mainnet2026.2 launch.
**Tick every item.** Do NOT skip.

### Pre-Launch (Feb 19-21): Preparation

- [ ] **P1. Update NetworkId enum** in `crates/q-types/src/lib.rs`:
  - Verify `Mainnet2026_2` variant exists with `as_str() → "mainnet2026.2"`
  - Verify `from_str("mainnet2026.2")` parses correctly
  - Verify `gossipsub_topic_prefix()` returns `/qnk/mainnet2026.2`

- [ ] **P2. Verify genesis timestamp** = `1771761600` in ALL locations:
  - `crates/q-storage/src/balance_consensus.rs` (GENESIS_TIMESTAMP constant)
  - `crates/q-storage/src/emission_controller.rs` (GENESIS_TIMESTAMP)
  - `crates/q-api-server/src/lib.rs` — NO hardcoded genesis timestamps (Bug #33 fix: must use `active_genesis_timestamp()`)
  - Frontend `gui/quantum-wallet/src/` (any hardcoded genesis references)
  - Confirm `active_genesis_timestamp()` returns `1771761600` when `Q_NETWORK_ID=mainnet2026.2`
  - Run: `grep -rn "1700000000\|1739836800\|1771372800" crates/` to verify NO stale timestamps remain
  - Verify `set_genesis_timestamp()` is called after `restore_emission_state()` (Bug #32 fix)

- [ ] **P3. Fix save_qblocks_batch_turbo() genesis filter** (Bug #28):
  Add genesis timestamp check to batch write path before write to RocksDB.

- [ ] **P4. Add pre-genesis turbo sync guard** (Bug #29):
  In `main.rs` where `sync_to_height()` is called: if `now() < genesis_ts`, skip sync.
  Also: if `now() < genesis_ts`, don't disable block production based on peer heights.

- [ ] **P5. Verify pool gossipsub topics use as_str()** (Bug #28):
  - Check `PoolTopics::new()` call uses `turbo_network_id.as_str()`, NOT `.display_name()`
  - Also check email/calendar topic subscriptions use correct format

- [ ] **P6. Verify genesis timestamps with Unix converter** (Bug #29):
  - Run: `python3 -c "from datetime import datetime; print(datetime.utcfromtimestamp(1771761600))"` → must show `2026-02-22 12:00:00`
  - Verify REHEARSAL_GENESIS_TIMESTAMP is correct if a rehearsal chain is used

- [ ] **P7. Clean build for deployment** (Bug #30):
  - Run `cargo clean --package q-api-server` before final release build
  - NEVER rely on incremental builds for production deploys with changed constants

- [ ] **P8. Build release binary** with version bumped past current:
  ```bash
  # In Cargo.toml: version = "7.3.7" (or next version)
  cargo build --release --package q-api-server
  target/release/q-api-server --version  # Verify matches Cargo.toml
  ```

- [ ] **P6. Check for running deleted binary** after build:
  ```bash
  readlink /proc/$(pgrep -f q-api-server | head -1)/exe
  # Must NOT show "(deleted)" — if it does, restart the service
  ```

- [ ] **P7. Verify LZ4/QRAW roundtrip** (Bug #22 prevention):
  ```bash
  cargo test --package q-storage -- precompressed  # Must pass
  ```

- [ ] **P8. Stop ALL canary nodes (Delta) early** to prevent data contamination:
  ```bash
  ssh root@5.79.79.158 "systemctl stop q-api-server"
  # Wipe canary data ONLY after final binary is ready:
  ssh root@5.79.79.158 "rm -rf /opt/orobit/shared/q-narwhalknight/data-mainnet2026.2/"
  ```

- [ ] **P9. SCP new binary to ALL servers** (not just through ha-deploy.sh):
  ```bash
  scp target/release/q-api-server root@109.205.176.60:/opt/orobit/shared/q-narwhalknight/q-api-server-v7.x.x
  scp target/release/q-api-server root@5.79.79.158:/opt/orobit/shared/q-narwhalknight/q-api-server-v7.x.x
  ```

### T-30min (Feb 22 11:30 UTC): Shutdown Old Network

- [ ] **L1. Stop BOTH Beta and Gamma simultaneously** (prevents HTTP state sync contamination):
  ```bash
  systemctl stop q-api-server                               # Beta
  ssh root@109.205.176.60 "systemctl stop q-api-server"     # Gamma
  ```

- [ ] **L2. Verify both processes are dead** (no (deleted) running binary):
  ```bash
  pgrep -f q-api-server  # Beta: should return nothing
  ssh root@109.205.176.60 "pgrep -f q-api-server"  # Gamma: nothing
  ```

- [ ] **L3. Update service files on Beta AND Gamma** — change these env vars:
  ```ini
  Environment="Q_DB_PATH=./data-mainnet2026.2"
  Environment="Q_NETWORK_ID=mainnet2026.2"
  ```
  Remove or change: `Q_ENCRYPTION_KEYS_FILE`, `Q_ENCRYPTION_PASSPHRASE` (let auto-create)
  DO NOT manually generate encryption keys (Bug #14 / #15).

- [ ] **L4. Update ExecStart** to point to the new binary version:
  ```ini
  ExecStart=/opt/orobit/shared/q-narwhalknight/target/release/q-api-server --port 8080
  ```

- [ ] **L5. Run systemctl daemon-reload** on both Beta and Gamma:
  ```bash
  systemctl daemon-reload
  ssh root@109.205.176.60 "systemctl daemon-reload"
  ```

### T-10min (Feb 22 11:50 UTC): Start Bootstrap

- [ ] **L6. Start Beta FIRST** (it is the bootstrap node):
  ```bash
  systemctl start q-api-server
  ```

- [ ] **L7. Wait 60s, then capture Beta's new peer ID**:
  ```bash
  sleep 60
  journalctl -u q-api-server --since "1 min ago" | grep "Local peer id"
  # Save this peer ID — needed for hardcoding later
  ```

- [ ] **L8. Start Gamma** (only after Beta is confirmed running):
  ```bash
  ssh root@109.205.176.60 "systemctl start q-api-server"
  # Capture Gamma peer ID from logs
  ```

- [ ] **L9. Start Delta** (third bootstrap):
  ```bash
  ssh root@5.79.79.158 "systemctl stop q-api-server ; rm -rf data-mainnet2026.2/ ; systemctl start q-api-server"
  ```
  Note: Wipe Delta's data-mainnet2026.2 to remove any canary contamination.

### T-0 (Feb 22 12:00 UTC): Genesis

- [ ] **L10. Verify genesis block produced** (check Beta logs at 12:00 UTC):
  ```bash
  journalctl -u q-api-server --since "12:00:00" | grep -E "Block.*produced|NEW BLOCK|genesis"
  ```

- [ ] **L11. Verify network_id** in produced blocks:
  ```bash
  journalctl -u q-api-server --since "12:00" | grep "mainnet2026.2"
  ```

- [ ] **L12. Verify emission rate** (~0.083 QUG/block at 1 bps → 2,625,000/year):
  ```bash
  journalctl -u q-api-server --since "12:00" | grep -E "emission|reward"
  ```

- [ ] **L13. Verify Gamma syncs FROM Beta** (not mining its own genesis — Bug #26):
  ```bash
  ssh root@109.205.176.60 "journalctl -u q-api-server --since '12:00' | grep -E 'turbo.*sync|Gossipsub BLOCK from'"
  # Should see blocks received from Beta, NOT "Block produced"
  ```

### T+5min: Post-Launch Verification

- [ ] **L14. Verify P2P peers connected** on both Beta and Gamma:
  ```bash
  journalctl -u q-api-server --since "12:00" | grep -E "peers.*connected|Gossipsub"
  ```

- [ ] **L15. Frontend shows correct network**:
  - Visit quillon.xyz → explorer should show `mainnet2026.2`
  - Block height should be > 0 and increasing
  - Network should NOT show `mainnet2026.1.1`

- [ ] **L16. Clear frontend localStorage** on all devices (Bug #21):
  Users should hard-refresh or clear cache to avoid stale balance display.
  Add a banner to the frontend: "Mainnet 2026.2 launched! Clear cache and reconnect."

### T+30min: Publish Binary and Update Docs

- [ ] **L17. Copy binary to downloads** (only AFTER launch confirmed):
  ```bash
  cp target/release/q-api-server gui/quantum-wallet/dist-final/downloads/q-api-server-v7.x.x
  cp target/release/q-api-server gui/quantum-wallet/dist-final/downloads/q-api-server-linux-x86_64
  chmod +x gui/quantum-wallet/dist-final/downloads/q-api-server-v7.x.x
  ```

- [ ] **L18. Update hardcoded bootstrap peer IDs** in code with new mainnet2026.2 peer IDs:
  - `gui/quantum-wallet/src/libp2p/config.ts`
  - `gui/quantum-wallet/src/libp2p/torConfig.ts`
  - `CLAUDE.md` (Bootstrap Peer IDs section)
  - `docs/MAINNET_2026_2_TRANSITION_GUIDE.md` (Bootstrap Peers table)

- [ ] **L19. Rebuild frontend** with new peer IDs:
  ```bash
  cd gui/quantum-wallet && npm run build
  ```

- [ ] **L20. Deploy frontend** (use ha-deploy.sh):
  ```bash
  cargo build --release --package q-api-server  # Version-bump first!
  echo "y" | ./scripts/ha-deploy.sh full
  ```

- [ ] **L21. Announce on Discord + BitcoinTalk**:
  - Post the wget download link for v7.x.x
  - Include the new bootstrap peer IDs
  - Include migration instructions (stop old node, fresh data dir, new binary)

### Updated Complete Checklist Summary (Items 49-53)

| # | Category | Item | Risk if Missed |
|---|----------|------|----------------|
| **49** | **Batch Turbo Genesis** | **Add genesis filter to save_qblocks_batch_turbo()** | **Rogue pre-genesis blocks stored** |
| **50** | **Rogue Pre-Genesis Peers** | **Don't sync/disable production before genesis_ts** | **Canary locked out 100s of hours** |
| **51** | **Deleted Binary Check** | **readlink /proc/PID/exe after build** | **Old binary runs silently** |
| **52** | **ALL Servers Updated** | **SCP binary to Delta too, not just ha-deploy.sh** | **Mixed-version, delta runs old code** |
| **53** | **Canary Genesis Timestamp** | **Canary service file must have correct genesis_ts** | **All blocks silently rejected** |
