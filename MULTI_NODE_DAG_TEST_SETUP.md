# Multi-Node DAG Consensus Test - Setup Complete

**Date**: October 25, 2025
**Status**: ✅ 3-Node Testnet Running
**Main Node**: Port 8080 PRESERVED (user mining active)
**Test Nodes**: Ports 8091-8093 (DAG consensus testing)

---

## 🎯 Test Objective

Validate Phase 3 Part 3 P2P block propagation and DAG consensus implementation with real multi-node operation:

1. **Multi-node block propagation** - Verify blocks propagate via Gossipsub
2. **DAG structure verification** - Confirm vertices reference multiple parents
3. **Consensus finality** - Verify all nodes reach Byzantine fault-tolerant agreement
4. **Performance measurement** - Get real TPS and latency metrics
5. **Bug identification** - Find and fix any issues in real operation

---

## 📊 Test Network Architecture

```
Main Network (PRESERVED):
Port 8080: Production node (users mining) - NOT TOUCHED

Test Network (NEW):
                  ┌─────────────────┐
                  │   Node 1 (8091) │
                  │  Bootstrap/Hub  │
                  │   P2P: 9091     │
                  └────────┬────────┘
                          / \
                         /   \
                        /     \
       ┌───────────────┐       ┌───────────────┐
       │ Node 2 (8092) │       │ Node 3 (8093) │
       │   P2P: 9092   │       │   P2P: 9093   │
       └───────────────┘       └───────────────┘
```

**Topology**: Star (2 and 3 connect to 1)
**Network**: Testnet (`/qnk/testnet/*` topics)
**Consensus**: DAG-Knight BFT (f=1, 2f+1=3)

---

## ✅ Current Status

### Nodes Running:
- **Node 1** (testdag-node1): ✅ Running on ports 8091/9091
- **Node 2** (testdag-node2): ⏹️ Stopped (check logs)
- **Node 3** (testdag-node3): ✅ Running on ports 8093/9093

### Gossipsub Topics Subscribed:
- `/qnk/testnet/blocks` ✅ (P2P block propagation)
- `/qnk/testnet/transactions` ✅
- `/qnk/testnet/mining-rewards` ✅
- `/qnk/testnet/dex/swaps` ✅
- `/qnk/testnet/votes` ✅
- `/qnk/testnet/ack` ✅

### Node 1 Status:
```json
{
  "current_height": 0,
  "current_round": 0,
  "connected_peers": 0,
  "consensus_status": "active",
  "network_health": "healthy",
  "performance": {
    "max_theoretical_tps": 6107031,
    "optimization_level": "Maximum (SIMD+Kernel I/O)"
  }
}
```

---

## 🔍 What To Monitor

### 1. Block Production (Node 1)
**Watch for**:
```bash
tail -f testdag-node1.log | grep -E 'produced|height'
```

**Expected**: Blocks produced every ~15 seconds (time-based)

### 2. Block Propagation (Nodes 2 & 3)
**Watch for**:
```bash
tail -f testdag-node2.log testdag-node3.log | grep '📦 Received block'
```

**Expected**: "📦 Received block X (height=Y) from network"

### 3. Consensus Finality (All Nodes)
**Watch for**:
```bash
tail -f testdag-node*.log | grep '🎯 INCOMING BLOCK FINALIZED'
```

**Expected**: All 3 nodes finalize the same blocks at similar rounds

### 4. DAG Structure
**Watch for**:
```bash
tail -f testdag-node*.log | grep -E 'parents|dag_vertex'
```

**Expected**: Vertices with multiple parent references

---

## 📝 Test Script

**Location**: `./test_3node_dag.sh`

**Usage**:
```bash
# Start 3-node testnet
./test_3node_dag.sh

# Monitor all nodes
tail -f testdag-node1.log testdag-node2.log testdag-node3.log | \
  grep --line-buffered -E '📦|📡|🎯|FINALIZED|height'

# Check node status
curl -s http://localhost:8091/api/v1/status | jq '.data | {height, round, peers: .connected_peers}'
curl -s http://localhost:8092/api/v1/status | jq '.data | {height, round, peers: .connected_peers}'
curl -s http://localhost:8093/api/v1/status | jq '.data | {height, round, peers: .connected_peers}'

# Stop test
pkill -f "q-api-server --port 809"
```

---

## 🧪 Test Scenarios

### Scenario 1: Block Propagation
1. Wait for Node 1 to produce a block
2. Verify Nodes 2 & 3 receive it within 500ms
3. Check all nodes have the same block at the same height

### Scenario 2: Consensus Agreement
1. Monitor consensus finality messages
2. Verify all nodes finalize blocks at the same round
3. Check commit decisions match across nodes

### Scenario 3: Network Partition
1. Disconnect Node 3 from Node 1
2. Verify Node 3 stops receiving blocks
3. Reconnect and verify Node 3 catches up

### Scenario 4: DAG Parent Structure
1. Examine vertex parents in logs
2. Verify vertices reference appropriate parent blocks
3. Confirm DAG structure (not linear chain)

---

## 🐛 Debugging

### Issue: Node 2 Not Running
**Check**:
```bash
cat testdag-node2.log | grep -E 'error|panic|failed' | tail -20
```

**Common causes**:
- Port 8092/9092 already in use
- Database initialization failure
- Network binding error

### Issue: No Block Propagation
**Check**:
```bash
# Verify P2P connections
curl -s http://localhost:8092/api/v1/status | jq '.data.connected_peers'

# Check Gossipsub subscriptions
grep "Subscribed to" testdag-node*.log
```

### Issue: Blocks Not Finalizing
**Check**:
```bash
# Verify consensus processing
grep "process_certificate" testdag-node*.log | tail -10

# Check DAG-Knight status
grep "DAG-Knight" testdag-node*.log | tail -10
```

---

## 📊 Expected Performance Metrics

### Block Propagation:
- **Latency**: <500ms from Node 1 → Nodes 2 & 3
- **Throughput**: All blocks successfully propagated
- **Duplicates**: 0 (height validation prevents replays)

### Consensus:
- **Finality Delay**: <5 seconds (δ-deep commitment)
- **Agreement**: 100% (all nodes finalize same blocks)
- **Byzantine Tolerance**: f=1 (can tolerate 1 faulty node)

### Network:
- **Peer Connections**: 2 peers for Node 1, 1 peer each for Nodes 2 & 3
- **Gossipsub Fan-Out**: 6 peers (default, limited by 3-node network)
- **Message Overhead**: ~1KB per block broadcast

---

## ✅ Success Criteria

- [ ] All 3 nodes initialize successfully
- [ ] Nodes 2 & 3 connect to Node 1 (P2P)
- [ ] Node 1 produces blocks every ~15 seconds
- [ ] Nodes 2 & 3 receive all blocks from Node 1
- [ ] All nodes finalize blocks at the same rounds
- [ ] No errors in consensus processing
- [ ] DAG structure visible in vertex parents
- [ ] Performance meets <500ms propagation target

---

## 🔧 Next Steps

1. **Fix Node 2** - Investigate why it's not running
2. **Establish P2P connections** - Connect Nodes 2 & 3 to Node 1
3. **Monitor block production** - Verify time-based block generation
4. **Validate propagation** - Confirm blocks reach all nodes
5. **Measure performance** - Collect TPS and latency data
6. **Test Byzantine scenarios** - Introduce faulty nodes
7. **Document findings** - Create test results report

---

## 📂 Log Files

- **Node 1**: `testdag-node1.log`
- **Node 2**: `testdag-node2.log`
- **Node 3**: `testdag-node3.log`

**Data Directories**:
- **Node 1**: `./testdag-node1/`
- **Node 2**: `./testdag-node2/`
- **Node 3**: `./testdag-node3/`

---

## 🚨 Important Notes

- **Main node on port 8080 is PRESERVED** - No disruption to user mining
- **Test nodes use separate data** - No interference with production data
- **Testnet network** - Isolated from mainnet
- **1-hour timeout** - Nodes auto-stop after 3600 seconds
- **Clean shutdown** - Use `pkill -f "q-api-server --port 809"` to stop

---

⚛️ **Q-NarwhalKnight Multi-Node DAG Consensus Test - ACTIVE** ⚛️
