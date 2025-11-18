# libp2p Network Propagation Testing Guide
## How to Verify Gossipsub, DHT, and Decentralization Actually Work
### Q-NarwhalKnight v1.0.16-beta

**Test Date**: 2025-11-18
**Purpose**: Prove that transactions actually propagate to all nodes via libp2p
**Target**: Verify gossipsub reaches 100+ nodes in <1.5 seconds

---

## Table of Contents

1. [Quick Test (5 minutes)](#1-quick-test-5-minutes)
2. [Comprehensive Test (30 minutes)](#2-comprehensive-test-30-minutes)
3. [Production Network Test (Live)](#3-production-network-test-live)
4. [Automated Test Suite](#4-automated-test-suite)
5. [Metrics and Monitoring](#5-metrics-and-monitoring)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Quick Test (5 minutes)

### Test Setup: Single Node + Bootstrap Node

**What We're Testing**: Can your local node discover and connect to the bootstrap node?

#### Step 1: Check Bootstrap Node is Running

```bash
# SSH to bootstrap node (185.182.185.227)
ssh root@185.182.185.227

# Check q-api-server is running
systemctl status q-api-server

# Expected output:
# ● q-api-server.service - Q-NarwhalKnight API Server
#    Active: active (running) since ...
#    Main PID: 927549

# Check P2P port is listening
ss -tuln | grep 9001

# Expected output:
# tcp   LISTEN  0  128  0.0.0.0:9001  0.0.0.0:*
```

#### Step 2: Start Your Local Node

```bash
# On your local machine
cd /opt/orobit/shared/q-narwhalknight

# Set network ID (CRITICAL - must match bootstrap)
export Q_NETWORK_ID="testnet-phase8"

# Start node with libp2p enabled
Q_DB_PATH=./data-test timeout 36000 ./target/release/q-api-server \
    --port 8090 \
    --node-id test-node-1 \
    2>&1 | tee test-node.log
```

#### Step 3: Verify Connection in Logs

**Look for these log messages** (within 30 seconds):

```
✅ SUCCESS INDICATORS:

[2025-11-18T10:00:01Z INFO  q_api_server] 🌐 Starting libp2p network with Peer ID: 12D3KooW...
[2025-11-18T10:00:02Z INFO  q_network] 🔗 Connecting to bootstrap peer: /ip4/185.182.185.227/tcp/9001/p2p/12D3KooWRX...
[2025-11-18T10:00:03Z INFO  q_network] ✅ Connected to bootstrap node
[2025-11-18T10:00:05Z INFO  q_network] 📡 Subscribed to topic: /qnk/testnet-phase8/transactions
[2025-11-18T10:00:10Z INFO  q_network] 🔍 Discovered peer: 12D3KooW... (total: 5 peers)
[2025-11-18T10:00:15Z INFO  q_network] 🔍 Discovered peer: 12D3KooW... (total: 10 peers)

❌ FAILURE INDICATORS:

[ERROR] Failed to connect to bootstrap node: Connection refused
[WARN] No peers discovered after 30 seconds
[ERROR] Failed to subscribe to gossipsub topic
```

#### Step 4: Send Test Transaction

```bash
# In another terminal, send a test transaction
curl -X POST http://localhost:8090/api/v1/send_transaction \
  -H "Content-Type: application/json" \
  -H "X-Wallet-Auth: test-signature" \
  -d '{
    "from": "qnk1234567890abcdef",
    "to": "qnk0987654321fedcba",
    "amount": 0.001,
    "token_type": "QUG",
    "mnemonic": "your test mnemonic here"
  }'
```

#### Step 5: Verify Broadcast in Logs

**Search logs for broadcast confirmation**:

```bash
grep "broadcast to.*P2P network" test-node.log

# Expected output:
# [INFO] 📤 Transaction abc123... broadcast to testnet-phase8 P2P network via gossipsub
```

**If you see this**: ✅ **Your node successfully broadcast to the network!**

---

## 2. Comprehensive Test (30 minutes)

### Test Setup: 3 Local Nodes + Bootstrap Node

**What We're Testing**: Do transactions propagate between multiple local nodes?

#### Step 1: Start Bootstrap Node (Skip if using production)

```bash
# Terminal 1: Bootstrap Node
Q_DB_PATH=./data-bootstrap \
Q_P2P_PORT=9001 \
./target/release/q-api-server \
    --port 8080 \
    --node-id bootstrap \
    2>&1 | tee bootstrap.log
```

**Get Bootstrap Peer ID**:
```bash
grep "Starting libp2p network with Peer ID" bootstrap.log | tail -1

# Output: Starting libp2p network with Peer ID: 12D3KooWABC123...
# Copy this Peer ID for next steps
```

#### Step 2: Start Node A

```bash
# Terminal 2: Node A
Q_DB_PATH=./data-node-a \
Q_P2P_PORT=9002 \
./target/release/q-api-server \
    --port 8081 \
    --node-id node-a \
    2>&1 | tee node-a.log &

# Wait 10 seconds for startup
sleep 10

# Check peer count
curl -s http://localhost:8081/api/v1/status | jq '.peer_count'
# Expected: 1-5 (should see bootstrap + maybe others)
```

#### Step 3: Start Node B

```bash
# Terminal 3: Node B
Q_DB_PATH=./data-node-b \
Q_P2P_PORT=9003 \
./target/release/q-api-server \
    --port 8082 \
    --node-id node-b \
    2>&1 | tee node-b.log &

sleep 10

curl -s http://localhost:8082/api/v1/status | jq '.peer_count'
# Expected: 2-6 (bootstrap + node-a + maybe others)
```

#### Step 4: Start Node C

```bash
# Terminal 4: Node C
Q_DB_PATH=./data-node-c \
Q_P2P_PORT=9004 \
./target/release/q-api-server \
    --port 8083 \
    --node-id node-c \
    2>&1 | tee node-c.log &

sleep 10

curl -s http://localhost:8083/api/v1/status | jq '.peer_count'
# Expected: 3-7 (bootstrap + node-a + node-b + maybe others)
```

#### Step 5: Send Transaction from Node A

```bash
# Create wallet on Node A
WALLET_A=$(curl -s -X POST http://localhost:8081/api/v1/create_wallet | jq -r '.data.address')
echo "Node A Wallet: $WALLET_A"

# Send transaction FROM Node A
TX_HASH=$(curl -s -X POST http://localhost:8081/api/v1/send_transaction \
  -H "Content-Type: application/json" \
  -d "{
    \"from\": \"$WALLET_A\",
    \"to\": \"qnk0987654321fedcba\",
    \"amount\": 0.001,
    \"token_type\": \"QUG\",
    \"mnemonic\": \"abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about\"
  }" | jq -r '.data.transaction_hash')

echo "Transaction Hash: $TX_HASH"
```

#### Step 6: Verify Propagation to Other Nodes

**Check Node B received the transaction** (should happen within 1-2 seconds):

```bash
# Query Node B's mempool (NOT Node A where we sent it)
curl -s "http://localhost:8082/api/v1/transaction/$TX_HASH" | jq '.'

# Expected output:
# {
#   "success": true,
#   "data": {
#     "transaction_hash": "abc123...",
#     "status": "InMempool",  ← Transaction is in Node B's mempool!
#     "from": "...",
#     "to": "...",
#     "amount": 100000
#   }
# }
```

**Check Node C received the transaction**:

```bash
curl -s "http://localhost:8083/api/v1/transaction/$TX_HASH" | jq '.data.status'

# Expected: "InMempool"
```

**If both Node B and Node C show "InMempool"**: ✅ **GOSSIPSUB PROPAGATION WORKS!**

#### Step 7: Verify in Logs

**Search all logs for the transaction hash**:

```bash
# Node A should show "broadcast"
grep -i "$TX_HASH" node-a.log | grep broadcast

# Expected:
# [INFO] 📤 Transaction abc123... broadcast to testnet-phase8 P2P network

# Node B should show "received via gossipsub"
grep -i "$TX_HASH" node-b.log | grep -i "gossip\|received"

# Expected:
# [DEBUG] Received transaction abc123... via gossipsub from peer 12D3KooW...

# Node C should also show received
grep -i "$TX_HASH" node-c.log | grep -i "gossip\|received"
```

---

## 3. Production Network Test (Live)

### Test Against Real Bootstrap Node (185.182.185.227)

**What We're Testing**: Can we discover real production peers?

#### Step 1: Start Node with Verbose Logging

```bash
export RUST_LOG=debug  # Enable debug logging
export Q_NETWORK_ID="testnet-phase8"

Q_DB_PATH=./data-production-test \
./target/release/q-api-server \
    --port 8090 \
    --node-id production-test \
    2>&1 | tee production-test.log
```

#### Step 2: Monitor Peer Discovery

**Watch for peer discovery in real-time**:

```bash
# In another terminal
tail -f production-test.log | grep -i "discovered peer\|connected to\|peer count"

# Expected output (within 2 minutes):
# [INFO] 🔍 Discovered peer: 12D3KooW... (total: 1 peers)
# [INFO] 🔍 Discovered peer: 12D3KooW... (total: 5 peers)
# [INFO] 🔍 Discovered peer: 12D3KooW... (total: 10 peers)
# ... (should reach 20-50+ peers after 5 minutes)
```

#### Step 3: Query Peer Count via API

```bash
# Every 10 seconds, check peer count
watch -n 10 'curl -s http://localhost:8090/api/v1/status | jq ".peer_count"'

# Expected progression:
# t=0s:   0 peers
# t=10s:  1-3 peers (bootstrap)
# t=30s:  5-10 peers (DHT discovery)
# t=60s:  15-25 peers
# t=120s: 30-50 peers
# t=300s: 50-100+ peers (network saturated)
```

#### Step 4: List All Discovered Peers

```bash
curl -s http://localhost:8090/api/v1/peers | jq '.data.peers[] | {peer_id: .peer_id, address: .address}'

# Expected output:
# {
#   "peer_id": "12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN",
#   "address": "/ip4/185.182.185.227/tcp/9001"
# }
# {
#   "peer_id": "12D3KooWABC...",
#   "address": "/ip4/192.168.1.50/tcp/9001"
# }
# ... (50-100+ peers)
```

#### Step 5: Send Transaction to Production Network

```bash
# Send real transaction
curl -X POST http://localhost:8090/api/v1/send_transaction \
  -H "Content-Type: application/json" \
  -d '{
    "from": "qnk'$(openssl rand -hex 32)'",
    "to": "qnk0987654321fedcba",
    "amount": 0.001,
    "token_type": "QUG",
    "mnemonic": "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
  }' | tee /tmp/tx_response.json

TX_HASH=$(jq -r '.data.transaction_hash' /tmp/tx_response.json)
echo "Sent transaction: $TX_HASH"
```

#### Step 6: Verify Production Nodes Received It

**Query bootstrap node** (should have received via gossipsub):

```bash
# SSH to bootstrap node
ssh root@185.182.185.227

# Check if transaction is in mempool
curl -s "http://localhost:8080/api/v1/transaction/$TX_HASH" | jq '.data.status'

# Expected: "InMempool" (if it arrived)
```

**If bootstrap node shows "InMempool"**: ✅ **PRODUCTION NETWORK PROPAGATION VERIFIED!**

---

## 4. Automated Test Suite

### Create Automated Test Script

**Save as**: `test_gossipsub_propagation.sh`

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Q-NarwhalKnight Gossipsub Propagation Test"
echo "=========================================="

# Configuration
NUM_NODES=5
BASE_PORT=8080
BASE_P2P_PORT=9000
NETWORK_ID="testnet-phase8"
TEST_DURATION=120  # 2 minutes

# Cleanup previous test
echo "🧹 Cleaning up previous test data..."
rm -rf ./data-test-* ./test-*.log

# Start nodes
echo "🚀 Starting $NUM_NODES test nodes..."
PIDS=()
for i in $(seq 0 $((NUM_NODES - 1))); do
    HTTP_PORT=$((BASE_PORT + i))
    P2P_PORT=$((BASE_P2P_PORT + i))
    NODE_ID="test-node-$i"

    echo "  Starting $NODE_ID on port $HTTP_PORT (P2P: $P2P_PORT)"

    Q_DB_PATH="./data-test-$i" \
    Q_P2P_PORT=$P2P_PORT \
    Q_NETWORK_ID=$NETWORK_ID \
    ./target/release/q-api-server \
        --port $HTTP_PORT \
        --node-id $NODE_ID \
        2>&1 > "test-$i.log" &

    PIDS+=($!)

    # Stagger startup
    sleep 2
done

echo "✅ All nodes started. PIDs: ${PIDS[@]}"

# Wait for peer discovery
echo "⏳ Waiting 30 seconds for peer discovery..."
sleep 30

# Check peer counts
echo "📊 Checking peer counts..."
for i in $(seq 0 $((NUM_NODES - 1))); do
    HTTP_PORT=$((BASE_PORT + i))
    PEER_COUNT=$(curl -s "http://localhost:$HTTP_PORT/api/v1/status" | jq -r '.peer_count // 0')
    echo "  Node $i: $PEER_COUNT peers"

    if [ "$PEER_COUNT" -lt 2 ]; then
        echo "  ⚠️  WARNING: Node $i has fewer than 2 peers!"
    fi
done

# Send test transaction from node 0
echo "📤 Sending test transaction from node 0..."
TX_HASH=$(curl -s -X POST "http://localhost:$BASE_PORT/api/v1/send_transaction" \
    -H "Content-Type: application/json" \
    -d '{
        "from": "qnk1234567890abcdef",
        "to": "qnk0987654321fedcba",
        "amount": 0.001,
        "token_type": "QUG",
        "mnemonic": "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
    }' | jq -r '.data.transaction_hash // "FAILED"')

if [ "$TX_HASH" == "FAILED" ] || [ -z "$TX_HASH" ]; then
    echo "❌ Failed to send transaction"
    exit 1
fi

echo "✅ Transaction sent: $TX_HASH"

# Wait for propagation
echo "⏳ Waiting 5 seconds for gossipsub propagation..."
sleep 5

# Check if transaction reached other nodes
echo "🔍 Checking transaction propagation..."
PROPAGATED_COUNT=0
for i in $(seq 1 $((NUM_NODES - 1))); do  # Skip node 0 (sender)
    HTTP_PORT=$((BASE_PORT + i))
    STATUS=$(curl -s "http://localhost:$HTTP_PORT/api/v1/transaction/$TX_HASH" | jq -r '.data.status // "NOT_FOUND"')

    if [ "$STATUS" == "InMempool" ] || [ "$STATUS" == "Confirmed" ]; then
        echo "  ✅ Node $i: Received ($STATUS)"
        ((PROPAGATED_COUNT++))
    else
        echo "  ❌ Node $i: NOT received ($STATUS)"
    fi
done

# Calculate success rate
SUCCESS_RATE=$((PROPAGATED_COUNT * 100 / (NUM_NODES - 1)))
echo ""
echo "=========================================="
echo "📊 TEST RESULTS"
echo "=========================================="
echo "Nodes tested: $NUM_NODES"
echo "Transaction propagated to: $PROPAGATED_COUNT / $((NUM_NODES - 1)) nodes"
echo "Success rate: $SUCCESS_RATE%"

if [ "$SUCCESS_RATE" -ge 80 ]; then
    echo "✅ TEST PASSED (≥80% propagation)"
    EXIT_CODE=0
else
    echo "❌ TEST FAILED (<80% propagation)"
    EXIT_CODE=1
fi

# Cleanup
echo ""
echo "🧹 Cleaning up test nodes..."
for PID in "${PIDS[@]}"; do
    kill $PID 2>/dev/null || true
done

exit $EXIT_CODE
```

**Run the test**:

```bash
chmod +x test_gossipsub_propagation.sh
./test_gossipsub_propagation.sh

# Expected output:
# ✅ TEST PASSED (≥80% propagation)
```

---

## 5. Metrics and Monitoring

### Real-Time Gossipsub Metrics

**Add to your node startup** (if Prometheus enabled):

```bash
# Enable Prometheus metrics
export Q_ENABLE_METRICS=true
export Q_METRICS_PORT=9090

# Start node
./target/release/q-api-server --port 8080 --node-id metrics-test
```

**Query Prometheus metrics**:

```bash
# Gossipsub message count
curl -s http://localhost:9090/metrics | grep gossipsub_messages_sent

# Output:
# gossipsub_messages_sent{topic="/qnk/testnet-phase8/transactions"} 42

# Peer count
curl -s http://localhost:9090/metrics | grep libp2p_peers_connected

# Output:
# libp2p_peers_connected 23
```

### Custom Metrics via API

```bash
# Get detailed network statistics
curl -s http://localhost:8080/api/v1/network/stats | jq '.'

# Expected output:
# {
#   "peer_count": 23,
#   "topics_subscribed": [
#     "/qnk/testnet-phase8/transactions",
#     "/qnk/testnet-phase8/blocks",
#     "/qnk/testnet-phase8/peer-heights"
#   ],
#   "messages_sent": 145,
#   "messages_received": 892,
#   "bandwidth_in_bytes": 1234567,
#   "bandwidth_out_bytes": 567890
# }
```

---

## 6. Troubleshooting

### Problem: "No peers discovered"

**Symptoms**:
```
[WARN] No peers discovered after 30 seconds
peer_count: 0
```

**Diagnosis**:

```bash
# 1. Check bootstrap node is reachable
telnet 185.182.185.227 9001

# Expected: "Connected to 185.182.185.227"
# If "Connection refused": Bootstrap node is down

# 2. Check firewall
sudo iptables -L | grep 9001

# If firewall is blocking, open port:
sudo iptables -A INPUT -p tcp --dport 9001 -j ACCEPT

# 3. Check network ID matches
grep "Q_NETWORK_ID" .env  # Should be "testnet-phase8"
```

---

### Problem: "Transaction not propagating"

**Symptoms**:
```
Node A sent transaction
Node B status: "NOT_FOUND"
```

**Diagnosis**:

```bash
# 1. Check gossipsub subscription
curl http://localhost:8081/api/v1/network/stats | jq '.topics_subscribed'

# Expected: Should include "/qnk/testnet-phase8/transactions"

# 2. Check transaction was actually broadcast
grep "broadcast to.*P2P network" node-a.log

# If missing: Transaction not broadcast (check libp2p_discovery is Some)

# 3. Check Node B is listening to correct topic
grep "Subscribed to topic" node-b.log

# Expected: "Subscribed to topic: /qnk/testnet-phase8/transactions"
```

**Fix**:

```rust
// In crates/q-api-server/src/handlers.rs line 1570
// Ensure libp2p_discovery is initialized:
if let Some(ref libp2p) = state.libp2p_discovery {  // ← Check this exists
    // Broadcast code here
}
```

---

### Problem: "Peers discovered but transaction not received"

**Symptoms**:
```
peer_count: 15
Transaction status on other nodes: "NOT_FOUND"
```

**Diagnosis**:

```bash
# 1. Check message ID deduplication
grep "duplicate.*message" node-b.log

# If found: Message was received but marked as duplicate

# 2. Check signature verification
grep "Invalid.*signature" node-b.log

# If found: Transaction signature is invalid (not broadcast to other nodes)

# 3. Check topic name matches
grep "publish_topic" node-a.log
grep "GossipsubEvent::Message.*topic" node-b.log

# Should be same topic: "/qnk/testnet-phase8/transactions"
```

---

### Problem: "Gossipsub bandwidth too high"

**Symptoms**:
```
bandwidth_out_bytes: 500000000  (500 MB)
Network congestion
```

**Solutions**:

```rust
// 1. Enable message compression (if not already)
// In crates/q-network/src/unified_network_manager.rs

use libp2p::gossipsub::MessageAuthenticity;

let gossipsub_config = GossipsubConfigBuilder::default()
    .validation_mode(ValidationMode::Strict)
    .message_id_fn(|msg| {
        // Use content hash to deduplicate
        Sha256::digest(&msg.data).to_vec()
    })
    .duplicate_cache_time(Duration::from_secs(120))  // Increase cache time
    .max_transmit_size(1024 * 1024)  // Reduce max size to 1 MB
    .build()?;

// 2. Implement message batching
// Send 100 transactions in one gossipsub message instead of 100 separate messages
```

---

## 7. Success Criteria Checklist

Use this checklist to verify libp2p network propagation:

### ✅ Basic Connectivity

- [ ] Bootstrap node is reachable (telnet test)
- [ ] Local node discovers bootstrap peer within 10 seconds
- [ ] Peer count increases over time (1 → 5 → 10+ peers)
- [ ] Subscribed to gossipsub topics (check logs)

### ✅ Transaction Propagation

- [ ] Transaction broadcast shows in sender logs ("📤 broadcast to P2P network")
- [ ] Transaction reaches 80%+ of test nodes within 2 seconds
- [ ] Other nodes show "InMempool" status when queried
- [ ] Receiver logs show gossipsub message reception

### ✅ DHT Functionality

- [ ] Kademlia peer discovery working (new peers appear over time)
- [ ] DHT routing table populated (check `/api/v1/peers`)
- [ ] Peers from multiple IP addresses (not just localhost)

### ✅ Production Network

- [ ] Discovers 20+ production peers within 2 minutes
- [ ] Transaction sent to production network appears on bootstrap node
- [ ] Gossipsub topics match production network ID

---

## 8. Performance Benchmarks

### Expected Performance Targets:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Peer Discovery** | 10 peers in 30s | `curl /api/v1/status` |
| **Transaction Propagation** | 80% of nodes in 1.5s | Automated test script |
| **Gossipsub Latency** | <500ms per hop | Check timestamps in logs |
| **DHT Query Time** | <2s to find random peer | `time curl /api/v1/peers` |
| **Bandwidth Usage** | <10 Mbps per node | Prometheus metrics |

### Run Performance Test:

```bash
#!/bin/bash
# performance_test.sh

echo "🏎️  Running libp2p Performance Test..."

# Start 10 nodes
for i in {0..9}; do
    Q_DB_PATH="./perf-test-$i" \
    ./target/release/q-api-server --port $((8080 + i)) --node-id "perf-$i" \
        2>&1 > "perf-$i.log" &
done

# Wait for network formation
sleep 60

# Send 100 transactions from node 0
for i in {1..100}; do
    curl -s -X POST http://localhost:8080/api/v1/send_transaction \
        -H "Content-Type: application/json" \
        -d "{
            \"from\": \"qnk$(openssl rand -hex 16)\",
            \"to\": \"qnk$(openssl rand -hex 16)\",
            \"amount\": 0.001,
            \"token_type\": \"QUG\",
            \"mnemonic\": \"abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about\"
        }" &

    sleep 0.1  # 10 TPS
done

wait

# Measure propagation
echo "📊 Measuring propagation rate..."
# (Check logs for propagation timestamps)
```

---

## Conclusion

**If you can verify ALL of the following**, libp2p network propagation is working correctly:

1. ✅ **Peer Discovery**: Nodes discover 10+ peers within 1 minute
2. ✅ **Gossipsub Subscription**: Logs show "Subscribed to topic: /qnk/.../transactions"
3. ✅ **Transaction Broadcast**: Sender logs show "📤 broadcast to P2P network"
4. ✅ **Transaction Reception**: Receiver logs show "Received transaction via gossipsub"
5. ✅ **Multi-Node Propagation**: 80%+ of test nodes receive transaction within 2 seconds
6. ✅ **Production Network**: Live production peers discovered and transactions propagate

**Next Steps**:
- Run automated test suite daily
- Monitor Prometheus metrics in production
- Set up alerts for peer count < 5
- Verify transaction propagation rate > 90%

---

**Questions? Issues?**
- Check logs: `tail -f node.log | grep -i "gossip\|peer\|broadcast"`
- Enable debug: `export RUST_LOG=debug`
- Test connectivity: `telnet 185.182.185.227 9001`

**END OF TEST PLAN**
