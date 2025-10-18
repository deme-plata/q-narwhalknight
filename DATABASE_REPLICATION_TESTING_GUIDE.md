# Database Replication Testing Guide

## Overview

This guide walks you through testing the database replication system with **real nodes**, **real transactions**, and **real faucet requests** to verify that database state synchronizes automatically across all nodes.

## Test Scenarios

We'll test three critical scenarios:
1. **Faucet Request Replication** - Request coins on Node A, verify balance appears on Node B
2. **Transaction Replication** - Send transaction on Node A, verify it appears on Node B
3. **Wallet Balance Synchronization** - Verify balances stay in sync across nodes

---

## Setup: Running Two Nodes Locally

### Terminal 1: Node A (Primary Node)

```bash
cd /opt/orobit/shared/q-narwhalknight

# Set unique database path and port
export Q_DB_PATH=./data-node-a
export Q_NODE_ID=node-a

# Clean start
rm -rf $Q_DB_PATH
mkdir -p $Q_DB_PATH

# Build and run Node A on port 8080
timeout 36000 cargo build --release --package q-api-server
./target/release/q-api-server --port 8080 2>&1 | tee /tmp/node-a.log
```

**Expected Output:**
```
🚀 Starting Q-NarwhalKnight Zero-Knowledge Discovery
🆔 Local Peer ID: 12D3KooW... (Node A)
📢 Subscribed to gossipsub topic: /qnk/database-updates/1.0.0
🌉 Database Replication Bridge started
📤 Outgoing: Replication Manager → Gossipsub
📥 Incoming: Gossipsub → Replication Manager
✅ Q-NarwhalKnight API Server running on 0.0.0.0:8080
```

### Terminal 2: Node B (Secondary Node)

```bash
cd /opt/orobit/shared/q-narwhalknight

# Set unique database path and port
export Q_DB_PATH=./data-node-b
export Q_NODE_ID=node-b

# Clean start
rm -rf $Q_DB_PATH
mkdir -p $Q_DB_PATH

# Run Node B on port 8081
./target/release/q-api-server --port 8081 2>&1 | tee /tmp/node-b.log
```

**Expected Output:**
```
🚀 Starting Q-NarwhalKnight Zero-Knowledge Discovery
🆔 Local Peer ID: 12D3KooW... (Node B, different from A)
✨ mDNS discovered: 12D3KooW... (Node A) at /ip4/127.0.0.1/tcp/...
🔗 Connected to peer: 12D3KooW... (Node A)
📢 Subscribed to gossipsub topic: /qnk/database-updates/1.0.0
🌉 Database Replication Bridge started
✅ Q-NarwhalKnight API Server running on 0.0.0.0:8081
```

**Key Indicators of Success:**
- ✅ Node B discovers Node A via mDNS within 1-2 seconds
- ✅ Both nodes subscribe to `/qnk/database-updates/1.0.0` topic
- ✅ Connection established between peers

---

## Test 1: Faucet Request Replication

### Step 1: Generate a Test Wallet

```bash
# Terminal 3
curl -X POST http://localhost:8080/api/wallet/create
```

**Response:**
```json
{
  "address": "qnk1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0",
  "public_key": "...",
  "mnemonic": "word1 word2 word3 ... word24",
  "phase": "Phase1"
}
```

**Save the address** - we'll use it for testing!

### Step 2: Request Faucet on Node A

```bash
# Request 10 QNK coins from faucet
curl -X POST http://localhost:8080/api/faucet \
  -H "Content-Type: application/json" \
  -d '{
    "address": "qnk1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "amount": 10000000000,
  "transaction_id": "0x123abc...",
  "message": "Faucet request successful! 10 QNK sent to your address"
}
```

### Step 3: Verify Balance on Node A

```bash
curl http://localhost:8080/api/wallet/qnk1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0/balance
```

**Expected Response:**
```json
{
  "address": "qnk1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0",
  "balance": 10000000000,
  "balance_qnk": "10.000000000"
}
```

### Step 4: Wait for Replication (5 minutes max)

**Watch Node A logs:**
```
📸 Creating database snapshot for broadcast
✅ Broadcast snapshot with manifest CID: QmXyz123...
📤 Forwarding database update to gossipsub: type=Snapshot, size=... bytes
```

**Watch Node B logs:**
```
📨 Gossipsub message received from ...: topic=/qnk/database-updates/1.0.0
📥 Received database update from gossipsub: type=Snapshot
📦 Restoring snapshot to temporary location: ./data-sync-...
✅ Successfully restored snapshot from peer
```

**OR** trigger immediate snapshot manually:
```bash
# Force snapshot on Node A
curl -X POST http://localhost:8080/api/storage/backup \
  -H "Content-Type: application/json" \
  -d '{
    "db_path": "./data-node-a",
    "compress": true,
    "replication": 3
  }'
```

### Step 5: Verify Balance on Node B

```bash
# Check same wallet on Node B (different port!)
curl http://localhost:8081/api/wallet/qnk1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0/balance
```

**Expected Response (SYNCHRONIZED!):**
```json
{
  "address": "qnk1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0",
  "balance": 10000000000,
  "balance_qnk": "10.000000000"
}
```

**✅ SUCCESS CRITERIA:**
- Balance on Node B matches Node A
- Balance appeared without any direct request to Node B
- Replication happened automatically via gossipsub + IPFS

---

## Test 2: Transaction Replication

### Step 1: Create Two Wallets

```bash
# Wallet 1 (Sender)
curl -X POST http://localhost:8080/api/wallet/create
# Save address as WALLET_A

# Wallet 2 (Receiver)
curl -X POST http://localhost:8080/api/wallet/create
# Save address as WALLET_B
```

### Step 2: Fund Wallet A via Faucet

```bash
curl -X POST http://localhost:8080/api/faucet \
  -H "Content-Type: application/json" \
  -d '{
    "address": "WALLET_A"
  }'
```

### Step 3: Send Transaction on Node A

```bash
curl -X POST http://localhost:8080/api/transaction/send \
  -H "Content-Type: application/json" \
  -d '{
    "from": "WALLET_A",
    "to": "WALLET_B",
    "amount": 5000000000,
    "signature": "mock_signature_for_testing"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "transaction_id": "0xabc123...",
  "status": "pending"
}
```

### Step 4: Check Transaction Status on Node A

```bash
curl http://localhost:8080/api/transaction/0xabc123.../status
```

**Expected Response:**
```json
{
  "transaction_id": "0xabc123...",
  "status": "confirmed",
  "confirmations": 1
}
```

### Step 5: Wait for Replication (or Force Snapshot)

```bash
# Option 1: Wait 5 minutes for automatic snapshot

# Option 2: Force immediate backup
curl -X POST http://localhost:8080/api/storage/backup \
  -H "Content-Type: application/json" \
  -d '{"db_path": "./data-node-a", "compress": true, "replication": 3}'
```

### Step 6: Verify Transaction on Node B

```bash
# Check transaction exists on Node B
curl http://localhost:8081/api/transaction/0xabc123.../status
```

**Expected Response:**
```json
{
  "transaction_id": "0xabc123...",
  "status": "confirmed",
  "confirmations": 1
}
```

### Step 7: Verify Balances Synchronized

```bash
# Check Wallet A balance on Node B (should be 5 QNK after sending 5)
curl http://localhost:8081/api/wallet/WALLET_A/balance

# Check Wallet B balance on Node B (should be 5 QNK after receiving)
curl http://localhost:8081/api/wallet/WALLET_B/balance
```

**Expected Responses:**
```json
// Wallet A (sent 5 QNK from 10 QNK)
{"balance": 5000000000, "balance_qnk": "5.000000000"}

// Wallet B (received 5 QNK)
{"balance": 5000000000, "balance_qnk": "5.000000000"}
```

**✅ SUCCESS CRITERIA:**
- Transaction appears on Node B without direct submission
- Balances match on both nodes
- Transaction status synchronized

---

## Test 3: Multi-Node Stress Test

### Setup: Run 3+ Nodes

```bash
# Terminal 1: Node A (port 8080, data-node-a)
# Terminal 2: Node B (port 8081, data-node-b)
# Terminal 3: Node C (port 8082, data-node-c)
# Terminal 4: Node D (port 8083, data-node-d)
```

### Test Script: Rapid Faucet Requests

```bash
#!/bin/bash
# save as test_replication_stress.sh

# Create 10 wallets
WALLETS=()
for i in {1..10}; do
  RESPONSE=$(curl -s -X POST http://localhost:8080/api/wallet/create)
  ADDRESS=$(echo $RESPONSE | jq -r '.address')
  WALLETS+=($ADDRESS)
  echo "Created wallet $i: $ADDRESS"
done

# Request faucet for all wallets on Node A
for WALLET in "${WALLETS[@]}"; do
  echo "Requesting faucet for $WALLET on Node A..."
  curl -s -X POST http://localhost:8080/api/faucet \
    -H "Content-Type: application/json" \
    -d "{\"address\": \"$WALLET\"}"
  sleep 1
done

# Force snapshot
echo "Forcing snapshot broadcast..."
curl -s -X POST http://localhost:8080/api/storage/backup \
  -H "Content-Type: application/json" \
  -d '{"db_path": "./data-node-a", "compress": true, "replication": 3}'

# Wait for replication
echo "Waiting 30 seconds for replication..."
sleep 30

# Verify all wallets on Node B, C, D
for NODE_PORT in 8081 8082 8083; do
  echo ""
  echo "========================================="
  echo "Verifying on Node (port $NODE_PORT)"
  echo "========================================="

  for WALLET in "${WALLETS[@]}"; do
    BALANCE=$(curl -s http://localhost:$NODE_PORT/api/wallet/$WALLET/balance | jq -r '.balance_qnk')
    echo "Wallet $WALLET: $BALANCE QNK"

    if [ "$BALANCE" != "10.000000000" ]; then
      echo "❌ MISMATCH! Expected 10 QNK, got $BALANCE"
    else
      echo "✅ Synchronized correctly"
    fi
  done
done
```

### Run the Stress Test

```bash
chmod +x test_replication_stress.sh
./test_replication_stress.sh
```

**Expected Output:**
```
Created wallet 1: qnk1abc...
Created wallet 2: qnk1def...
...
Requesting faucet for qnk1abc... on Node A
...
Forcing snapshot broadcast...
Waiting 30 seconds for replication...

=========================================
Verifying on Node (port 8081)
=========================================
Wallet qnk1abc...: 10.000000000 QNK
✅ Synchronized correctly
Wallet qnk1def...: 10.000000000 QNK
✅ Synchronized correctly
...

=========================================
Verifying on Node (port 8082)
=========================================
Wallet qnk1abc...: 10.000000000 QNK
✅ Synchronized correctly
...
```

**✅ SUCCESS CRITERIA:**
- All 10 wallets replicated to all nodes
- All balances match across nodes
- No data loss or corruption

---

## Test 4: Network Partition Recovery

### Simulate Network Partition

```bash
# Terminal 1: Start Node A
export Q_DB_PATH=./data-node-a
./target/release/q-api-server --port 8080

# Terminal 2: Start Node B
export Q_DB_PATH=./data-node-b
./target/release/q-api-server --port 8081

# Wait for connection, then STOP Node B
# Ctrl+C in Terminal 2
```

### Create Data on Node A During Partition

```bash
# Node B is offline - Node A operates independently
for i in {1..5}; do
  curl -X POST http://localhost:8080/api/wallet/create
  curl -X POST http://localhost:8080/api/faucet -H "Content-Type: application/json" -d "{\"address\": \"wallet$i\"}"
done
```

### Restart Node B (Partition Healed)

```bash
# Terminal 2: Restart Node B
export Q_DB_PATH=./data-node-b
./target/release/q-api-server --port 8081
```

**Expected Behavior:**
```
# Node B logs:
🔗 Connected to peer: 12D3KooW... (Node A)
📥 Received database update from gossipsub: type=Snapshot
📦 Downloading snapshot from IPFS...
✅ Successfully restored snapshot from peer
🔄 Requesting full sync from network (catching up)
```

### Verify Synchronization After Recovery

```bash
# Check that Node B now has all wallets created during partition
curl http://localhost:8081/api/wallet/wallet1/balance
curl http://localhost:8081/api/wallet/wallet2/balance
# ... should all return 10 QNK
```

**✅ SUCCESS CRITERIA:**
- Node B automatically catches up after reconnection
- All data created during partition appears on Node B
- No manual intervention required

---

## Monitoring and Debugging

### Check Replication Statistics

```bash
# API endpoint for replication stats (TODO: implement this)
curl http://localhost:8080/api/storage/replication/stats
```

**Expected Response:**
```json
{
  "updates_sent": 15,
  "updates_received": 10,
  "bytes_synced": 2048576,
  "peers_synced": 3,
  "last_sync_time": "2025-01-15T10:30:00Z",
  "current_sequence": 25
}
```

### Monitor Gossipsub Topics

```bash
# Check gossipsub connectivity
curl http://localhost:8080/api/network/gossipsub/topics
```

**Expected Response:**
```json
{
  "topics": [
    "/qnk/blocks/1.0.0",
    "/qnk/votes/1.0.0",
    "/qnk/ack/1.0.0",
    "/qnk/database-updates/1.0.0"
  ],
  "peers": 3
}
```

### Watch Logs in Real-Time

```bash
# Terminal 1: Node A logs
tail -f /tmp/node-a.log | grep -E "(snapshot|replication|gossipsub)"

# Terminal 2: Node B logs
tail -f /tmp/node-b.log | grep -E "(snapshot|replication|gossipsub)"
```

**Key Log Messages to Watch:**

**Node A (Broadcaster):**
```
📸 Creating database snapshot for broadcast
✅ Broadcast snapshot with manifest CID: QmXyz...
📤 Forwarding database update to gossipsub
📨 Published message to gossipsub topic: /qnk/database-updates/1.0.0
```

**Node B (Receiver):**
```
📨 Gossipsub message received from ...: topic=/qnk/database-updates/1.0.0
📥 Received database update from gossipsub: type=Snapshot
📦 Restoring snapshot to temporary location
✅ Successfully restored snapshot from peer
```

---

## Performance Benchmarks

### Measure Replication Latency

```bash
#!/bin/bash
# measure_replication_latency.sh

echo "Creating wallet and requesting faucet on Node A..."
START_TIME=$(date +%s)

WALLET=$(curl -s -X POST http://localhost:8080/api/wallet/create | jq -r '.address')
curl -s -X POST http://localhost:8080/api/faucet -H "Content-Type: application/json" -d "{\"address\": \"$WALLET\"}"

# Force snapshot
curl -s -X POST http://localhost:8080/api/storage/backup \
  -H "Content-Type: application/json" \
  -d '{"db_path": "./data-node-a", "compress": true, "replication": 3}'

# Poll Node B until balance appears
while true; do
  BALANCE=$(curl -s http://localhost:8081/api/wallet/$WALLET/balance | jq -r '.balance')

  if [ "$BALANCE" != "null" ] && [ "$BALANCE" != "0" ]; then
    END_TIME=$(date +%s)
    LATENCY=$((END_TIME - START_TIME))
    echo "✅ Replication complete in $LATENCY seconds"
    break
  fi

  echo "⏳ Waiting for replication..."
  sleep 2
done
```

**Expected Latency:**
- Automatic replication: 5-300 seconds (depends on snapshot interval)
- Forced snapshot: 10-30 seconds (depends on database size and network)

### Measure Throughput

```bash
# Create 100 wallets and measure total sync time
time for i in {1..100}; do
  curl -s -X POST http://localhost:8080/api/wallet/create
  curl -s -X POST http://localhost:8080/api/faucet -d "{\"address\": \"wallet$i\"}"
done

# Force snapshot
curl -X POST http://localhost:8080/api/storage/backup -d '{"db_path": "./data-node-a"}'

# Verify all 100 wallets on Node B
for i in {1..100}; do
  curl -s http://localhost:8081/api/wallet/wallet$i/balance
done
```

---

## Troubleshooting

### Issue: Nodes Not Discovering Each Other

**Symptoms:**
- Node B logs don't show "mDNS discovered" message
- No connection established

**Solution:**
```bash
# Check if mDNS is working (Linux only)
avahi-browse -a

# Check firewall rules
sudo iptables -L

# Ensure both nodes are on same network interface
ip addr show
```

### Issue: Gossipsub Messages Not Propagating

**Symptoms:**
- Node A broadcasts snapshot
- Node B doesn't receive message

**Diagnostics:**
```bash
# Check gossipsub subscriptions
curl http://localhost:8080/api/network/gossipsub/topics
curl http://localhost:8081/api/network/gossipsub/topics

# Verify both show /qnk/database-updates/1.0.0
```

**Solution:**
```bash
# Restart both nodes with verbose logging
RUST_LOG=debug ./target/release/q-api-server --port 8080
RUST_LOG=debug ./target/release/q-api-server --port 8081
```

### Issue: Snapshot Download Fails

**Symptoms:**
- Node B logs show "Failed to download snapshot"
- IPFS errors

**Solution:**
```bash
# Check IPFS daemon is running
ipfs id

# Verify IPFS connectivity
ipfs swarm peers

# Check manifest CID is valid
ipfs cat <manifest_cid>
```

### Issue: Database Corruption After Restore

**Symptoms:**
- Node B crashes after restore
- "Invalid RocksDB format" errors

**Solution:**
```bash
# Stop Node B
# Delete corrupted database
rm -rf ./data-node-b

# Restart Node B (will sync from scratch)
export Q_DB_PATH=./data-node-b
./target/release/q-api-server --port 8081
```

---

## Production Deployment Checklist

Before deploying to production:

- [ ] **Test with 3+ nodes** on same network
- [ ] **Test with nodes on different networks** (internet)
- [ ] **Test network partition recovery**
- [ ] **Test with large database** (1GB+)
- [ ] **Measure replication latency** under load
- [ ] **Verify signature verification** is enabled
- [ ] **Test rate limiting** for snapshot broadcasts
- [ ] **Monitor IPFS storage usage** and implement cleanup
- [ ] **Test Byzantine fault tolerance** (malicious nodes)
- [ ] **Implement backup/restore procedures**
- [ ] **Set up alerting** for replication failures
- [ ] **Document operational procedures**

---

## Success Metrics

Your database replication is working correctly if:

✅ **Wallets created on Node A appear on Node B automatically**
✅ **Faucet requests on Node A update balances on Node B**
✅ **Transactions sent on Node A appear on Node B**
✅ **Replication completes within 30 seconds** (with forced snapshot)
✅ **Replication completes within 5 minutes** (automatic snapshot)
✅ **Nodes recover automatically after network partitions**
✅ **No data loss or corruption** during replication
✅ **Gossipsub messages propagate to all peers**
✅ **IPFS downloads complete successfully**
✅ **Sequence numbers prevent duplicate processing**

---

## Next Steps

Once basic replication is working:

1. **Implement Incremental Updates** - Send only changed chunks instead of full snapshots
2. **Add Signature Verification** - Verify DatabaseUpdate messages are from trusted nodes
3. **Implement Consensus Checks** - Ensure replicated data matches consensus
4. **Add Merkle Proofs** - Verify partial state without downloading entire database
5. **Optimize Snapshot Frequency** - Dynamic intervals based on activity
6. **Add Monitoring Dashboard** - Real-time replication health visualization
7. **Implement Pruning** - Clean up old IPFS snapshots to save storage

---

**Ready to test!** 🚀

Start with Test 1 (Faucet Replication) to verify basic functionality, then move on to more complex scenarios.
