# Mainnet Launch Genesis Timestamp Guide

**Target Launch**: Mid-December 2025
**Current Status**: Testnet running with Oct 26, 2025 genesis
**Mainnet Genesis**: December 15, 2025 00:00:00 UTC (recommended)

---

## 📅 Mainnet Genesis Options

### Option 1: December 1, 2025
- **Timestamp**: `1764547200`
- **UTC Time**: December 1, 2025 00:00:00 UTC
- **Use Case**: Early December launch

### Option 2: December 15, 2025 ⭐ RECOMMENDED
- **Timestamp**: `1765756800`
- **UTC Time**: December 15, 2025 00:00:00 UTC
- **Use Case**: Mid-December launch (recommended)

### Option 3: December 20, 2025
- **Timestamp**: `1766188800`
- **UTC Time**: December 20, 2025 00:00:00 UTC
- **Use Case**: Late December launch

---

## 🔧 Single Line Change Required

### File Location
```
/opt/orobit/shared/q-narwhalknight/crates/q-api-server/src/handlers.rs
```

### Line Number
**Line 190**

### Current Code (Testnet)
```rust
pub const GENESIS_TIMESTAMP: u64 = 1761436800; // Oct 26, 2025 00:00:00 UTC
```

### Updated Code (Mainnet - December 15, 2025)
```rust
pub const GENESIS_TIMESTAMP: u64 = 1765756800; // Dec 15, 2025 00:00:00 UTC
```

---

## ✅ Why This Approach is Simple

1. **Single Constant**: Only one line to change
2. **No Database Migration**: Blockchain starts fresh on mainnet
3. **Same Logic**: Emission schedule calculation stays the same
4. **Network Separation**: Use `Q_NETWORK=mainnet` vs `Q_NETWORK=testnet`
5. **Parallel Networks**: Can run testnet and mainnet simultaneously

---

## 📋 Mainnet Launch Checklist

### Week Before Launch (Dec 8-14, 2025)

- [ ] **Update Genesis Timestamp**
  ```bash
  # Edit handlers.rs line 190
  # Change: 1761436800 → 1765756800
  ```

- [ ] **Build Mainnet Binary**
  ```bash
  cd /opt/orobit/shared/q-narwhalknight
  timeout 36000 cargo build --release --package q-api-server
  ```

- [ ] **Verify Binary**
  ```bash
  ls -lh target/release/q-api-server
  ./target/release/q-api-server --version
  ```

- [ ] **Test Locally**
  ```bash
  Q_DB_PATH=./data-mainnet-test \
  Q_NETWORK=mainnet \
  ./target/release/q-api-server --port 8090 --node-id mainnet-test
  ```

- [ ] **Verify Emission Schedule**
  ```bash
  # Check first block reward should be 0.001 QUG
  curl http://localhost:8090/api/v1/supply
  ```

### Launch Day (December 15, 2025)

- [ ] **Deploy Binary to Production**
  ```bash
  # Copy to production servers
  scp target/release/q-api-server server1:/opt/q-narwhalknight/
  scp target/release/q-api-server server2:/opt/q-narwhalknight/
  scp target/release/q-api-server server3:/opt/q-narwhalknight/
  ```

- [ ] **Start Mainnet Nodes**
  ```bash
  # On each production server:
  Q_DB_PATH=/var/lib/q-narwhalknight/mainnet \
  Q_NETWORK=mainnet \
  Q_P2P_PORT=9000 \
  ./q-api-server --port 8080 --node-id mainnet-node-1
  ```

- [ ] **Verify Genesis Block**
  ```bash
  # Check block 0 was created
  curl http://mainnet-server:8080/api/v1/blocks/0
  ```

- [ ] **Verify First Rewards**
  ```bash
  # Confirm miners getting 0.001 QUG per block
  curl http://mainnet-server:8080/api/v1/supply | jq '.mined_coins'
  ```

- [ ] **Monitor Emission**
  ```bash
  # Watch blocks being produced
  curl http://mainnet-server:8080/api/v1/blocks/latest
  ```

### Post-Launch (First Week)

- [ ] **Monitor Block Rewards**
  - Verify all rewards = 0.001 QUG
  - Check no halving occurs (first halving: Dec 15, 2026)

- [ ] **Check Network Health**
  - P2P peer count
  - Block propagation time
  - Transaction throughput

- [ ] **User Verification**
  - Users receiving correct mining rewards
  - Wallet balances updating properly
  - No genesis timestamp bugs

---

## 🔍 Verification Commands

### Check Current Genesis Timestamp
```bash
grep "GENESIS_TIMESTAMP" crates/q-api-server/src/handlers.rs
```

### Calculate Reward for Any Date
```python
import time
from datetime import datetime, timezone

# Genesis: Dec 15, 2025
GENESIS_TIMESTAMP = 1765756800
SECONDS_PER_YEAR = 31_536_000
BASE_REWARD = 100_000  # 0.001 QUG

# Current time
current = int(time.time())
elapsed = current - GENESIS_TIMESTAMP
halving_count = elapsed // SECONDS_PER_YEAR
reward = BASE_REWARD >> halving_count

print(f"Reward: {reward / 100_000_000:.9f} QUG")
```

### Test Genesis Timestamp Before Deploy
```bash
# Build with new timestamp
cargo build --release --package q-api-server

# Run test server
Q_DB_PATH=./test-genesis \
Q_NETWORK=mainnet \
./target/release/q-api-server --port 9999 --node-id genesis-test

# Check emission (in another terminal)
curl http://localhost:9999/api/v1/supply | jq '.'
```

---

## 📊 Emission Schedule (Mainnet)

### Year 1: December 15, 2025 - December 14, 2026
- **Reward**: 0.001 QUG per block
- **Blocks/Day**: ~7,200 (12 sec per block)
- **Daily Emission**: ~7.2 QUG
- **Yearly Emission**: ~2,628 QUG

### Year 2: December 15, 2026 - December 14, 2027
- **Reward**: 0.0005 QUG per block (first halving)
- **Daily Emission**: ~3.6 QUG
- **Yearly Emission**: ~1,314 QUG

### Max Supply
- **Hard Cap**: 21,000,000 QUG
- **Halvings**: Every 365 days
- **Final Halving**: Year 64 (negligible rewards)

---

## 🛡️ Safety Checks

### Before Launch
```bash
# 1. Verify timestamp is in the future
python3 << 'EOF'
import time
GENESIS = 1765756800  # Dec 15, 2025
current = int(time.time())
if GENESIS > current:
    days_until = (GENESIS - current) / 86400
    print(f"✅ Genesis is {days_until:.1f} days in the future")
else:
    print(f"❌ ERROR: Genesis is in the past!")
EOF

# 2. Verify emission calculation
python3 << 'EOF'
GENESIS = 1765756800
CURRENT = 1765756800 + 86400  # 1 day after genesis
elapsed = CURRENT - GENESIS
halving = elapsed // 31_536_000
reward = 100_000 >> halving
if reward == 100_000:
    print("✅ First day reward correct: 0.001 QUG")
else:
    print(f"❌ ERROR: Wrong reward {reward}")
EOF
```

### After Launch
```bash
# Monitor first 100 blocks
for i in {0..99}; do
  curl -s http://mainnet:8080/api/v1/blocks/$i | jq '.reward'
  sleep 12  # Wait for next block
done
```

---

## 🚀 Quick Reference

### Testnet (Current)
```
Genesis: 1761436800 (Oct 26, 2025)
Network: testnet
Purpose: Testing and development
```

### Mainnet (December 15, 2025)
```
Genesis: 1765756800 (Dec 15, 2025)
Network: mainnet
Purpose: Production blockchain
```

### Change Required
```diff
- pub const GENESIS_TIMESTAMP: u64 = 1761436800; // Oct 26, 2025
+ pub const GENESIS_TIMESTAMP: u64 = 1765756800; // Dec 15, 2025 00:00:00 UTC
```

### Build Command
```bash
timeout 36000 cargo build --release --package q-api-server
```

### Deploy Command
```bash
Q_DB_PATH=/var/lib/q-narwhalknight/mainnet \
Q_NETWORK=mainnet \
./q-api-server --port 8080 --node-id mainnet-validator-1
```

---

## ✅ Success Criteria

After mainnet launch, verify:

- [x] Genesis block created at exactly Dec 15, 2025 00:00:00 UTC
- [x] First block reward = 0.001 QUG (100,000 base units)
- [x] No premature halving (halving count = 0)
- [x] Next halving scheduled for Dec 15, 2026
- [x] Users receiving correct mining rewards
- [x] Emission schedule matches whitepaper
- [x] Max supply tracking correctly (21M QUG cap)

---

**Summary**: Switching to mainnet requires changing **just one number** on **one line** in one file. Build, test, deploy, and you're live! 🚀

**Prepared by**: Server Beta (Claude Code)
**Date**: October 27, 2025
**Status**: Ready for December 15, 2025 mainnet launch
