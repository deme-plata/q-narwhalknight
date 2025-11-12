# Q-NarwhalKnight Phase Transition Guide

## 🎯 Purpose

This guide provides a comprehensive checklist for transitioning between network phases (testnet phases or testnet → mainnet). It ensures all necessary components are updated and serves as a rehearsal for the eventual mainnet launch.

---

## 📋 Phase Transition Checklist

### Phase Information

**Example Transition:**
- **From:** `testnet-phase5` (v0.9.59-beta)
- **To:** `testnet-phase6` (v0.9.60-beta)
- **Date:** 2025-11-08
- **Reason:** Austrian economics mining rewards + sync-down bug fix + mainnet rehearsal

---

## 🔧 Code Changes Required

### 1. Network ID Update

**File:** `crates/q-types/src/lib.rs`

**Location:** NetworkId enum definition

**Changes:**
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkId {
    Testnet,           // Deprecated - Phase 0-2
    TestnetPhase3,     // Deprecated
    TestnetPhase4,     // Deprecated
    TestnetPhase5,     // Previous phase
    TestnetPhase6,     // ✅ NEW - Current phase
    Mainnet,           // Future
}

impl Default for NetworkId {
    fn default() -> Self {
        // ✅ UPDATE THIS for new phase
        Self::TestnetPhase6  // Changed from TestnetPhase5
    }
}
```

**Also update:**
- `from_str()` implementation - add new phase parsing
- `to_string()` implementation - add new phase display name
- Any match statements that handle all NetworkId variants

---

### 2. Gossipsub Topics Update

**File:** `crates/q-types/src/lib.rs`

**Location:** NetworkId implementation methods

**Update all topic methods:**

```rust
impl NetworkId {
    pub fn blocks_topic(&self) -> String {
        match self {
            NetworkId::TestnetPhase6 => "/qnk/testnet-phase6/blocks".to_string(),
            NetworkId::Mainnet => "/qnk/mainnet/blocks".to_string(),
            _ => "/qnk/testnet-phase5/blocks".to_string(), // Fallback for old phases
        }
    }

    pub fn peer_heights_topic(&self) -> String {
        match self {
            NetworkId::TestnetPhase6 => "/qnk/testnet-phase6/peer-heights".to_string(),
            NetworkId::Mainnet => "/qnk/mainnet/peer-heights".to_string(),
            _ => "/qnk/testnet-phase5/peer-heights".to_string(),
        }
    }

    pub fn block_pack_requests_topic(&self) -> String {
        match self {
            NetworkId::TestnetPhase6 => "/qnk/testnet-phase6/block-pack-requests".to_string(),
            NetworkId::Mainnet => "/qnk/mainnet/block-pack-requests".to_string(),
            _ => "/qnk/testnet-phase5/block-pack-requests".to_string(),
        }
    }

    pub fn block_pack_responses_topic(&self) -> String {
        match self {
            NetworkId::TestnetPhase6 => "/qnk/testnet-phase6/block-pack-responses".to_string(),
            NetworkId::Mainnet => "/qnk/mainnet/block-pack-responses".to_string(),
            _ => "/qnk/testnet-phase5/block-pack-responses".to_string(),
        }
    }

    // ✅ ADD any new topics introduced in this phase
    pub fn ai_capability_topic(&self) -> String {
        match self {
            NetworkId::TestnetPhase6 => "qnk/ai/node-capability/v1".to_string(),
            NetworkId::Mainnet => "qnk/mainnet/ai/node-capability/v1".to_string(),
            _ => "qnk/ai/node-capability/v1".to_string(),
        }
    }
}
```

**IMPORTANT:** Verify ALL topic methods are updated!

---

### 3. Database Path Update

**File:** `crates/q-api-server/src/main.rs`

**Location:** Database initialization

**Before:**
```rust
let db_path = std::env::var("Q_DB_PATH")
    .unwrap_or_else(|_| "./data".to_string());
```

**After (Phase 6):**
```rust
let db_path = std::env::var("Q_DB_PATH")
    .unwrap_or_else(|_| "./data-mine6".to_string());
```

**Also update:**
- GUI default path in `gui/quantum-wallet/src/constants/paths.ts` or equivalent
- Docker compose files
- Deployment scripts
- Backup scripts (adjust backup paths)

---

### 4. Genesis Block Update

**File:** `crates/q-types/src/block.rs` or genesis configuration

**Updates needed:**
```rust
// Update genesis timestamp for new phase
pub const GENESIS_TIMESTAMP_PHASE6: u64 = 1731081600; // 2025-11-08 12:00:00 UTC

// Update genesis hash if needed
pub const GENESIS_HASH_PHASE6: &str = "phase6_genesis_hash_here";
```

**Note:** For testnet transitions, you may want a fresh genesis. For mainnet, genesis is permanent!

---

### 5. Mining Rewards Economics Update

**File:** `crates/q-mining/src/lib.rs` or `crates/q-mining/src/rewards.rs`

**Phase 6 - Austrian Economics Improvements:**

```rust
/// Phase 6: Improved Austrian Economics Mining Rewards
///
/// Key principles:
/// - Sound money: Predictable, algorithmic issuance
/// - Halving schedule: Similar to Bitcoin but faster convergence
/// - Early adopter rewards: Higher initial rewards encourage network bootstrapping
/// - Long-term sustainability: Tail emission after halving epochs
///
/// Emission Schedule:
/// - Epoch 1 (blocks 1-210,000): 50 QNK per block
/// - Epoch 2 (blocks 210,001-420,000): 25 QNK per block
/// - Epoch 3 (blocks 420,001-630,000): 12.5 QNK per block
/// - Epoch 4 (blocks 630,001-840,000): 6.25 QNK per block
/// - Epoch 5+ (blocks 840,001+): 3.125 QNK per block (tail emission)
///
/// Total supply: ~21 million QNK (asymptotic)
///
pub fn calculate_block_reward_phase6(height: u64) -> u64 {
    const INITIAL_REWARD: u64 = 50_000_000; // 50 QNK (8 decimals)
    const HALVING_INTERVAL: u64 = 210_000; // Blocks per epoch
    const MIN_REWARD: u64 = 3_125_000; // 3.125 QNK tail emission

    let epoch = height / HALVING_INTERVAL;

    if epoch >= 4 {
        // Tail emission: constant 3.125 QNK forever
        MIN_REWARD
    } else {
        // Halving schedule: 50 → 25 → 12.5 → 6.25
        INITIAL_REWARD >> epoch // Right shift = divide by 2^epoch
    }
}

/// Development fee: 2% of block reward
/// Used for ongoing development, security audits, and infrastructure
pub fn calculate_dev_fee_phase6(block_reward: u64) -> u64 {
    block_reward / 50 // 2% = 1/50
}
```

**Rationale for Austrian Economics:**
- **Sound Money:** Predictable, algorithmic issuance (no central bank)
- **Scarcity:** Fixed maximum supply creates digital scarcity
- **Time Preference:** Higher early rewards reflect higher time preference of early adopters
- **Long-term Sustainability:** Tail emission ensures perpetual security budget
- **Free Market:** Rewards determined by protocol, not central authority

---

### 6. Frontend Modal Update

**File:** `gui/quantum-wallet/src/components/PhaseTransitionModal.tsx`

**Update phase information:**

```tsx
const PhaseTransitionModal: React.FC = () => {
    const [isOpen, setIsOpen] = useState(true);

    return (
        <Modal isOpen={isOpen} onClose={() => setIsOpen(false)}>
            <div className="phase-transition-modal">
                <h2>🎉 Welcome to Testnet Phase 6!</h2>

                <div className="phase-info">
                    <h3>What's New in Phase 6:</h3>
                    <ul>
                        <li>✅ <strong>Austrian Economics Mining:</strong> Sound money principles with halving schedule</li>
                        <li>✅ <strong>Sync-Down Protection:</strong> Critical bug fix preventing data loss</li>
                        <li>✅ <strong>Fresh Network:</strong> Clean start with new gossipsub topics</li>
                        <li>✅ <strong>Mainnet Rehearsal:</strong> Testing transition procedures</li>
                    </ul>
                </div>

                <div className="economics-info">
                    <h3>💰 New Emission Schedule:</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Epoch</th>
                                <th>Block Range</th>
                                <th>Reward</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>1</td>
                                <td>1 - 210,000</td>
                                <td>50 QNK</td>
                            </tr>
                            <tr>
                                <td>2</td>
                                <td>210,001 - 420,000</td>
                                <td>25 QNK</td>
                            </tr>
                            <tr>
                                <td>3</td>
                                <td>420,001 - 630,000</td>
                                <td>12.5 QNK</td>
                            </tr>
                            <tr>
                                <td>4</td>
                                <td>630,001 - 840,000</td>
                                <td>6.25 QNK</td>
                            </tr>
                            <tr>
                                <td>5+</td>
                                <td>840,001+</td>
                                <td>3.125 QNK (tail)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div className="action-required">
                    <h3>⚠️ Action Required:</h3>
                    <p>
                        This is a <strong>fresh network</strong> with a new database.
                        Your Phase 5 wallet and balance are preserved but not transferred.
                    </p>
                    <p>
                        To participate in Phase 6:
                    </p>
                    <ol>
                        <li>Download the latest node software (v0.9.60-beta)</li>
                        <li>Start mining to earn Phase 6 QNK</li>
                        <li>Test transaction features on the new network</li>
                    </ol>
                </div>

                <div className="network-details">
                    <h4>Network Details:</h4>
                    <ul>
                        <li><strong>Network ID:</strong> testnet-phase6</li>
                        <li><strong>Database Path:</strong> ./data-mine6</li>
                        <li><strong>Genesis Time:</strong> 2025-11-08 12:00:00 UTC</li>
                        <li><strong>Bootstrap Node:</strong> 185.182.185.227:9001</li>
                    </ul>
                </div>

                <button
                    className="btn-primary"
                    onClick={() => setIsOpen(false)}
                >
                    Start Mining on Phase 6!
                </button>
            </div>
        </Modal>
    );
};
```

---

### 7. Version Number Update

**File:** `Cargo.toml` (workspace root)

**Update version:**
```toml
[workspace.package]
version = "0.9.60-beta"  # Phase 6 launch version
```

**Tag convention:**
- Testnet phase transitions: `v0.X.X-beta`
- Mainnet launch: `v1.0.0`
- Post-mainnet: Standard semver `vX.Y.Z`

---

### 8. Bootstrap Peer Configuration

**File:** `crates/q-network/src/unified_network_manager.rs` or config

**Update bootstrap peers if needed:**

```rust
pub fn default_bootstrap_peers_phase6() -> Vec<Multiaddr> {
    vec![
        // Server Beta - Primary bootstrap
        "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN"
            .parse()
            .expect("Valid multiaddr"),

        // Add additional bootstrap nodes as network grows
    ]
}
```

**Note:** For mainnet, bootstrap peers should be geographically distributed and run by independent entities.

---

## 🚀 Deployment Procedure

### Pre-Deployment Checklist

- [ ] All code changes committed to git
- [ ] Version number updated in `Cargo.toml`
- [ ] Git tag created (e.g., `v0.9.60-beta`)
- [ ] Compilation successful: `cargo build --release --workspace`
- [ ] Tests passing: `cargo test --workspace`
- [ ] Frontend built: `cd gui/quantum-wallet && npm run build`

### Server Deployment Steps

#### Step 1: Stop Old Phase Services

```bash
# On all servers
systemctl stop q-api-server
systemctl stop q-miner  # If running
```

#### Step 2: Backup Old Phase Data (Optional)

```bash
# Backup Phase 5 database (optional - for reference)
tar -czf phase5-backup-$(date +%Y%m%d-%H%M%S).tar.gz ./data

# Move to backup location
mv phase5-backup-*.tar.gz /mnt/backup/phase5/
```

#### Step 3: Deploy New Binary

```bash
# Build on development server
timeout 36000 cargo build --release --package q-api-server

# Copy to production servers
scp target/release/q-api-server root@185.182.185.227:/usr/local/bin/q-api-server-v0.9.60-beta
scp target/release/q-api-server root@161.35.219.10:/usr/local/bin/q-api-server-v0.9.60-beta

# On each server:
cp /usr/local/bin/q-api-server-v0.9.60-beta /usr/local/bin/q-api-server
chmod +x /usr/local/bin/q-api-server
```

#### Step 4: Configure Environment

```bash
# Update systemd service file if needed
cat > /etc/systemd/system/q-api-server.service <<EOF
[Unit]
Description=Q-NarwhalKnight API Server (Phase 6)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/orobit/shared/q-narwhalknight
Environment="Q_DB_PATH=./data-mine6"
Environment="Q_NETWORK_ID=testnet-phase6"
Environment="RUST_LOG=info"
ExecStart=/usr/local/bin/q-api-server --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
```

#### Step 5: Start New Phase Services

```bash
# Start API server
systemctl start q-api-server

# Verify it started
systemctl status q-api-server

# Check logs
journalctl -u q-api-server -f
```

#### Step 6: Verify Network Operation

```bash
# Check API responds
curl http://localhost:8080/api/v1/status

# Expected output should show:
# - network_id: "testnet-phase6"
# - current_height: 1 (or low number for new phase)
# - Database path: ./data-mine6

# Check peer connections
curl http://localhost:8080/api/v1/peers

# Monitor block production
journalctl -u q-api-server -f | grep "Produced block"
```

---

## 🧪 Testing Checklist

### Functional Testing

- [ ] Genesis block created successfully
- [ ] Mining rewards calculated correctly (50 QNK for epoch 1)
- [ ] Dev fee calculated correctly (2% = 1 QNK)
- [ ] Transactions work (send/receive)
- [ ] Balance updates correctly
- [ ] Gossipsub topics subscribed correctly
- [ ] Peer discovery working
- [ ] Block propagation working
- [ ] Turbo sync working (test with fresh node)

### Economics Testing

- [ ] Block reward = 50 QNK at height 1
- [ ] Block reward = 25 QNK at height 210,001
- [ ] Block reward = 12.5 QNK at height 420,001
- [ ] Dev fee = 2% at all heights
- [ ] Total supply tracking correctly

### Regression Testing

- [ ] No sync-down bugs (monitor for 24 hours)
- [ ] No database corruption
- [ ] No memory leaks
- [ ] No crash bugs
- [ ] Performance acceptable (TPS, latency)

---

## 📝 Communication Plan

### User Announcement Template

```markdown
# 🎉 Testnet Phase 6 Launch Announcement

We're excited to announce the launch of **Q-NarwhalKnight Testnet Phase 6**!

## What's New:

### 💰 Austrian Economics Mining Rewards
Phase 6 introduces a Bitcoin-inspired halving schedule with sound money principles:
- **Initial Reward:** 50 QNK per block
- **Halving Schedule:** Every 210,000 blocks
- **Maximum Supply:** ~21 million QNK
- **Tail Emission:** 3.125 QNK perpetual security budget

### 🛡️ Critical Bug Fixes
- Fixed catastrophic sync-down bug that could cause data loss
- Added corruption detection for network protocol mismatches
- Enhanced database safety checks

### 🚀 Mainnet Rehearsal
This phase transition serves as a dress rehearsal for the eventual mainnet launch,
testing all migration procedures and ensuring smooth transitions.

## Migration Guide:

### For Node Operators:
1. Download v0.9.60-beta from [releases page]
2. Stop your Phase 5 node
3. Start the new Phase 6 node (uses ./data-mine6 database)
4. Begin mining on the fresh network

### For Miners:
1. Update to latest miner binary (v0.9.60-beta)
2. Point to Phase 6 node (port 8080)
3. Start mining to earn Phase 6 QNK

## Important Notes:

⚠️ **This is a FRESH NETWORK** - Phase 5 balances do not transfer.
This is intentional for testing phase transitions before mainnet.

✅ **Your Phase 5 wallet is safe** - It's preserved in ./data and can be
accessed by running a Phase 5 node separately if needed.

## Network Details:

- **Network ID:** testnet-phase6
- **Genesis Time:** 2025-11-08 12:00:00 UTC
- **Bootstrap Node:** 185.182.185.227:9001
- **Database Path:** ./data-mine6

Happy mining! 🎉
```

---

## 🔄 Mainnet Transition Differences

When transitioning to mainnet (THE BIG ONE), additional considerations:

### 1. Permanence
- **Testnet:** Can reset, fix bugs, iterate
- **Mainnet:** PERMANENT - no resets, no do-overs

### 2. Genesis Block
- **Testnet:** Fresh genesis each phase
- **Mainnet:** ONE genesis block, forever

### 3. Pre-mine / Initial Distribution
- **Testnet:** Not applicable
- **Mainnet:** May include:
  - Development team allocation
  - Early contributor rewards
  - Treasury allocation
  - Public sale allocations

### 4. Economics Finalization
- **Testnet:** Can adjust rewards, emission
- **Mainnet:** MUST be final - immutable monetary policy

### 5. Security Audit
- **Testnet:** Internal testing
- **Mainnet:** MANDATORY third-party security audit

### 6. Legal/Regulatory
- **Testnet:** Not applicable
- **Mainnet:** May require:
  - Legal review
  - Regulatory compliance
  - Entity formation
  - Terms of service

### 7. Communication
- **Testnet:** Community announcement
- **Mainnet:** Major marketing campaign, press releases, exchange listings

### 8. Bootstrap Infrastructure
- **Testnet:** Single/few bootstrap nodes
- **Mainnet:** Geographically distributed, independently operated bootstrap nodes

---

## 📊 Monitoring & Metrics

### Key Metrics to Track

1. **Network Health:**
   - Peer count
   - Block production rate
   - Network hashrate
   - Average block time

2. **Economics:**
   - Total supply issued
   - Current epoch
   - Block reward amount
   - Dev fee collected

3. **Performance:**
   - TPS (transactions per second)
   - Block propagation latency
   - Sync time for new nodes
   - Database size growth

4. **Security:**
   - No sync-down events
   - No database corruption
   - No consensus failures
   - No double-spend attempts

### Alerting Thresholds

- Block production stopped for >60 seconds → CRITICAL
- Peer count < 2 → WARNING
- Sync-down detected → CRITICAL SHUTDOWN
- Database errors → CRITICAL

---

## ✅ Post-Deployment Verification

### 24-Hour Checklist

- [ ] Network producing blocks consistently
- [ ] Multiple peers connected and syncing
- [ ] Mining rewards distributing correctly
- [ ] Transactions processing successfully
- [ ] No error logs or crashes
- [ ] Database growing as expected
- [ ] Frontend showing correct phase info

### 1-Week Checklist

- [ ] Economics working as designed (halving logic testable at height 210,000 if reached)
- [ ] Network stable with no major issues
- [ ] Community feedback positive
- [ ] Performance metrics acceptable
- [ ] Ready for next phase (or mainnet!)

---

## 📚 Additional Resources

- **Code Repository:** https://code.quillon.xyz/q-narwhalknight
- **Documentation:** ./docs/
- **Community:** Discord/Telegram/Forum
- **Support:** GitHub Issues

---

## 🎯 Success Criteria

A phase transition is considered successful when:

✅ All checklist items completed
✅ Network producing blocks for 24+ hours
✅ No critical bugs discovered
✅ Mining rewards working correctly
✅ Community able to participate
✅ Documentation updated
✅ Monitoring in place

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Next Review:** Before mainnet launch
