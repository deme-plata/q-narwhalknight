# 🔧 Q-NarwhalKnight v0.1.1-beta Mining Troubleshooting Guide

## 🎯 Quick Diagnosis

**Is your miner finding solutions but not earning rewards?** Follow this guide step-by-step.

---

## ⚠️ Common Issues & Solutions

### 📍 Issue #1: Mining to Localhost Instead of Public IP

**Symptoms:**
- ✅ Miner shows: `Found solution for Block #0, nonce: 123456, hash: abc123...`
- ❌ No rewards appearing in wallet
- ❌ Node logs don't show incoming mining submissions

**Root Cause:** Your miner is submitting to `http://127.0.0.1:8080` instead of your actual node's public IP.

**Fix:**
```bash
# ❌ WRONG - Mining to localhost
./q-miner --mode solo --wallet qnk1your_address --server http://127.0.0.1:8080

# ✅ CORRECT - Mining to your actual node IP
./q-miner --mode solo --wallet qnk1your_address --server http://45.45.218.154:8080
```

**How to find your node's IP:**
```bash
# On your node server, run:
curl ifconfig.me

# Use this IP in your miner command
```

---

### 🔌 Issue #2: Port 8081 Already in Use

**Error Message:**
```
ERROR q_api_server: P2P listener failed: Address already in use (os error 98)
```

**Root Cause:** You're running multiple nodes on the same server, or a previous node didn't shut down cleanly.

**Fix:**
```bash
# 1. Check what's using port 8081
sudo ss -tulpn | grep 8081

# 2. Kill the conflicting process
sudo killall q-api-server

# 3. Wait 5 seconds for port to release
sleep 5

# 4. Restart your node
Q_DB_PATH=./data-node1 ./q-api-server --port 8080 --node-id node1
```

**Running Multiple Nodes? Use Different P2P Ports:**
```bash
# Node 1
Q_DB_PATH=./data-node1 Q_P2P_PORT=8081 ./q-api-server --port 8080 --node-id node1

# Node 2 (use different P2P port)
Q_DB_PATH=./data-node2 Q_P2P_PORT=8082 ./q-api-server --port 8090 --node-id node2
```

---

### 🚫 Issue #3: Node Not Accepting Mining Submissions

**Symptoms:**
- ✅ Node is running
- ✅ Miner is running
- ❌ Miner shows connection errors

**Verification Steps:**

**Step 1: Test node health from miner machine**
```bash
# Replace with your node's actual IP
curl http://45.45.218.154:8080/api/v1/health

# Should return: {"status":"healthy",...}
```

**Step 2: Test mining endpoint**
```bash
curl http://45.45.218.154:8080/api/v1/mining/status

# Should return mining difficulty and current block
```

**Step 3: Check firewall**
```bash
# On your node server, allow port 8080
sudo ufw allow 8080/tcp
sudo ufw status

# Should show: 8080/tcp ALLOW Anywhere
```

---

### 📊 Issue #4: No Blocks Being Produced

**Symptoms:**
- ✅ Node starts successfully
- ✅ All subsystems initialized
- ❌ No new blocks appearing

**Diagnostic Commands:**

```bash
# Check current block height
curl http://your-node-ip:8080/api/v1/statistics/network | jq '.data.current_height'

# Check mining queue
curl http://your-node-ip:8080/api/v1/mining/status | jq '.data.queue_size'

# Check peer count
curl http://your-node-ip:8080/api/v1/network/peers | jq '.data | length'
```

**Expected Values:**
- Current height: Should be increasing over time
- Queue size: Should show pending transactions or mining solutions
- Peer count: Should be > 0 (at least bootstrap peer)

**Fix if height is stuck:**
```bash
# 1. Check node logs for errors
tail -f node.log | grep ERROR

# 2. Verify bootstrap peer connection
curl http://your-node-ip:8080/api/v1/network/peers

# Should show: {"peer_id": "...", "address": "/ip4/91.216.4.55/..."}
```

---

### 💰 Issue #5: Mining Rewards Not Appearing

**Symptoms:**
- ✅ Miner finding solutions
- ✅ Node receiving submissions
- ❌ Wallet balance not increasing

**Verification:**

**Step 1: Check your wallet address is correct**
```bash
# In your miner command, verify address format:
./q-miner --node-url http://your-ip:8080 --wallet-address qnk1...your_address
```

**Step 2: Query your wallet balance**
```bash
curl http://your-node-ip:8080/api/v1/wallet/balance/qnk1...your_address
```

**Step 3: Check recent transactions**
```bash
# On the web interface, go to Explorer → Recent Network Activity
# Look for mining rewards to your address
```

**Step 4: Verify mining difficulty**
```bash
curl http://your-node-ip:8080/api/v1/mining/status | jq '.data.difficulty'

# If difficulty is very high, solutions might be rejected
# Expected: difficulty should be reasonable for CPU mining
```

---

## 🔍 Complete Diagnostic Checklist

Run through this checklist when troubleshooting:

### ✅ Node Health
```bash
# 1. Node is running
ps aux | grep q-api-server

# 2. Node is listening on correct port
sudo ss -tulpn | grep 8080

# 3. Node health endpoint responds
curl http://localhost:8080/api/v1/health

# 4. P2P listener is working (no port conflicts)
sudo ss -tulpn | grep 8081

# 5. Bootstrap peer is connected
curl http://localhost:8080/api/v1/network/peers | jq '.'
```

### ✅ Miner Configuration
```bash
# 1. Miner is using PUBLIC IP (not 127.0.0.1)
echo "Check your miner command for --node-url"

# 2. Wallet address is correct format
echo "Address should start with 'qnk1'"

# 3. Miner can reach node
curl http://your-public-ip:8080/api/v1/health

# 4. Mining endpoint is accessible
curl http://your-public-ip:8080/api/v1/mining/status
```

### ✅ Network Connectivity
```bash
# 1. Firewall allows port 8080
sudo ufw status | grep 8080

# 2. Node is reachable from external network
# (Run this from a DIFFERENT machine)
curl http://your-public-ip:8080/api/v1/health

# 3. Tor is working (if enabled)
curl http://localhost:8080/api/v1/network/peers | grep -i tor
```

---

## 🚀 Optimal Mining Setup

**For Best Results:**

### Single Node + Single Miner
```bash
# On your server (45.45.218.154):
Q_DB_PATH=./qnk-data ./q-api-server --port 8080 --node-id my-node

# On your miner machine (or same server):
./q-miner \
  --mode solo \
  --wallet qnk1your_wallet_address_here \
  --server http://45.45.218.154:8080 \
  --threads 4 \
  --intensity 7
```

### Multiple Miners → One Node
```bash
# Node (once):
Q_DB_PATH=./qnk-data ./q-api-server --port 8080 --node-id central-node

# Miner 1:
./q-miner --mode solo --wallet qnk1address1 --server http://node-ip:8080 --threads 2

# Miner 2:
./q-miner --mode solo --wallet qnk1address2 --server http://node-ip:8080 --threads 2

# Miner 3:
./q-miner --mode solo --wallet qnk1address3 --server http://node-ip:8080 --threads 2
```

### GPU Mining (if available)
```bash
./q-miner \
  --mode solo \
  --wallet qnk1your_wallet \
  --server http://node-ip:8080 \
  --gpu \
  --intensity 9
```

---

## 📞 Getting Help

**Still stuck? Provide this info in Discord:**

```
🔧 My Setup:
- Node IP: [your public IP]
- Node Port: [usually 8080]
- Miner Command: [paste your full command]
- Node Version: v0.1.1-beta

📊 Diagnostics:
[Paste output of these commands]
curl http://your-ip:8080/api/v1/health
curl http://your-ip:8080/api/v1/mining/status
curl http://your-ip:8080/api/v1/network/peers

📋 Recent Logs:
[Paste last 20 lines of node logs]
tail -20 node.log

🔍 Miner Output:
[Paste last 20 lines showing solutions]
```

---

## 🎓 Understanding Mining Flow

```
┌─────────────┐
│   Q-Miner   │  1. Fetches current block template
│             │  2. Computes Poseidon2 hashes
│             │  3. Finds valid nonce (hash < difficulty)
└──────┬──────┘
       │ 4. Submits solution via HTTP POST
       ▼
┌─────────────┐
│  API Server │  5. Validates solution
│  (Node)     │  6. Adds to mining queue
│             │  7. Includes in next QBlock
└──────┬──────┘
       │ 8. Block production (DAG-Knight consensus)
       ▼
┌─────────────┐
│   QBlock    │  9. Mining reward transaction created
│ (Confirmed) │  10. Reward sent to miner's wallet address
└─────────────┘
```

**Key Points:**
- ⏱️ Solutions must arrive before next block is produced
- 🎯 Difficulty adjusts based on network hashrate
- 💎 Rewards are paid per valid solution, not per block
- 🔒 ZK-STARK proofs ensure privacy of mining rewards

---

## ⚡ Performance Tips

### Optimizing Mining Speed
```bash
# Use all CPU cores at maximum intensity
./q-miner --mode solo --wallet qnk1your_address --server http://node-ip:8080 --threads 0 --intensity 10

# Reduce logging overhead
export RUST_LOG=warn

# Run miner with high priority
nice -n -10 ./q-miner --mode solo --wallet qnk1your_address --server http://node-ip:8080

# Benchmark your hardware first
./q-miner --benchmark --duration 60
```

### Optimizing Node Performance
```bash
# Increase file descriptors
ulimit -n 65536

# Run with release build (faster)
cargo build --release --package q-api-server

# Use optimized database path (SSD recommended)
Q_DB_PATH=/path/to/fast/ssd/qnk-data ./q-api-server
```

---

## 🔐 Security Reminders

- 🚫 **Never share your wallet private key**
- ✅ **Public wallet address (qnk1...) is safe to share**
- 🔒 **Keep node API port (8080) behind firewall if not mining remotely**
- 🧅 **Tor mode recommended for maximum privacy**
- 🛡️ **ZK-STARK ensures transaction privacy automatically**

---

**Good luck mining! ⚛️ May the quantum consensus be with you! 🚀**

*Q-NarwhalKnight: The world's first quantum-resistant DAG-BFT consensus system*
