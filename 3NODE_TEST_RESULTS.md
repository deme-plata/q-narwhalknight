# 3-Node Network Propagation Test Results

**Date**: 2025-10-25
**Test**: Transaction propagation across libp2p network
**Status**: ⚠️  Nodes starting but NOT CONNECTED

---

## Summary

✅ **GOOD NEWS**:
- All 3 test nodes start successfully
- APIs respond on ports 8091, 8092, 8093
- Wallet creation works
- P2P ports (9101, 9102, 9103) avoid conflicts with nova-chat

❌ **CRITICAL ISSUE**:
- **Nodes have ZERO P2P peers** (they're not connecting to each other!)
- Without P2P connectivity, transaction propagation cannot be tested

---

## Test Configuration

```
Node 1: API 8091, P2P 9101, DB ./testnet-data/node1
Node 2: API 8092, P2P 9102, DB ./testnet-data/node2
Node 3: API 8093, P2P 9103, DB ./testnet-data/node3

Production: API 8080, P2P 9000 (still running - no conflict)
```

---

## Test Results

### ✅ What Works

1. **Node Startup**
   ```
   ✅ API ready on port 8091
   ✅ API ready on port 8092
   ✅ API ready on port 8093
   ```

2. **Wallet Creation**
   ```
   Node 1: qnk8d006700e28d4727df468f64893856b8e414ef49056660e8d1af1386404fb40b
   Node 2: qnkfd2a9401e1b07aadafe37f1571064e336596ba474d1999de06199739d709bae6
   Node 3: qnk1801f89a7787c24b060bfa846bdaf0a6a5783454c9ba2225b507515edf7d11cc
   ```

### ❌ What Doesn't Work

1. **P2P Connectivity** (CRITICAL)
   ```json
   Node 8091: {"data": []} // 0 peers
   Node 8092: {"data": []} // 0 peers
   Node 8093: {"data": []} // 0 peers
   ```

2. **Transaction Propagation**
   - Can't test because nodes aren't connected
   - Need P2P peers for gossip protocol

---

## Root Cause: No Peer Discovery

The nodes are starting but **not discovering each other**. This means:

1. **Bootstrap/seed nodes** not configured
2. **mDNS peer discovery** might be disabled
3. **Manual peer connection** endpoints not used
4. **Listening addresses** might not be correct

---

## How to Fix

### Option 1: Add Bootstrap Peers (Recommended)

When starting each node, provide bootstrap peers:

```bash
# Node 2 connects to Node 1
Q_BOOTSTRAP_PEERS="/ip4/127.0.0.1/tcp/9101" \
  ./target/release/q-api-server --port 8092 --node-id node2

# Node 3 connects to Node 1 and 2
Q_BOOTSTRAP_PEERS="/ip4/127.0.0.1/tcp/9101,/ip4/127.0.0.1/tcp/9102" \
  ./target/release/q-api-server --port 8093 --node-id node3
```

### Option 2: Use Manual Peer Connection API

After starting all nodes, connect them manually:

```bash
# Get Node 1's peer ID from logs or API
node1_peer_id="12D3KooW..."  # From startup logs

# Connect Node 2 to Node 1
curl -X POST http://localhost:8092/api/v1/network/peers/connect \
  -H "Content-Type: application/json" \
  -d "{
    \"multiaddr\": \"/ip4/127.0.0.1/tcp/9101/p2p/$node1_peer_id\"
  }"
```

### Option 3: Enable mDNS Discovery

Check if mDNS is enabled in the network configuration:

```rust
// In crates/q-network/src/lib.rs or similar
let mdns_config = mdns::Config {
    enable: true,  // Should be true for local discovery
    ..Default::default()
};
```

---

## Manual Testing Commands

I created `test_3node_manual.sh` with all the curl commands you need.

### Quick Test Steps:

```bash
# 1. Create wallets
addr1=$(curl -s -X POST http://localhost:8091/api/v1/wallets/create \
  -H "Content-Type: application/json" \
  -d '{"label": "Node1"}' | jq -r '.data.address_formatted')

addr2=$(curl -s -X POST http://localhost:8092/api/v1/wallets/create \
  -H "Content-Type: application/json" \
  -d '{"label": "Node2"}' | jq -r '.data.address_formatted')

# 2. Fund wallets
curl -X POST http://localhost:8091/api/v1/faucet \
  -H "Content-Type: application/json" \
  -d "{\"address\": \"$addr1\"}"

curl -X POST http://localhost:8092/api/v1/faucet \
  -H "Content-Type: application/json" \
  -d "{\"address\": \"$addr2\"}"

# 3. Check balances
curl "http://localhost:8091/api/v1/wallets/$addr1/balance" | jq '.data.balance_qug'

# 4. Send transaction (will fail if nodes not connected!)
curl -X POST http://localhost:8091/api/v1/transactions/send \
  -H "Content-Type: application/json" \
  -d "{
    \"from\": \"$addr1\",
    \"to\": \"$addr2\",
    \"amount_qug\": 10,
    \"fee_qug\": 0.1
  }" | jq '.'

# 5. Check if transaction propagated to Node 2
curl "http://localhost:8092/api/v1/wallets/$addr2/balance" | jq '.data.balance_qug'
```

---

## Next Steps

1. **PRIORITY**: Fix P2P peer discovery
   - Add bootstrap peer configuration
   - OR enable mDNS for local discovery
   - OR implement manual peer connection API

2. **Test**: Verify nodes can see each other
   ```bash
   curl http://localhost:8091/api/v1/network/active-peers | jq '.data | length'
   # Should show 2 peers (nodes 2 and 3)
   ```

3. **Test**: Send transaction and verify propagation
   - Send from Node 1
   - Check balance on Node 2
   - Check balance on Node 3
   - All should show the same balance

---

## Code Areas to Investigate

### 1. Network Initialization (main.rs or lib.rs)

Look for where libp2p network is initialized:

```rust
// Find this code
let network_manager = UnifiedNetworkManager::new(...)?;

// Check if bootstrap peers are configured
// Check if mDNS is enabled
// Check listen addresses
```

### 2. Peer Discovery Configuration

```bash
# Search for peer discovery config
grep -r "mdns\|bootstrap\|peer.*discover" crates/q-network/
```

### 3. Manual Peer Connection Endpoint

```bash
# Check if this endpoint exists and works
grep -r "peers/connect\|connect_peer" crates/q-api-server/src/
```

---

## Files Created

- `test_3node_network.sh` - Automated test (has issues with address capture)
- `test_3node_manual.sh` - Manual curl command guide
- `3NODE_TEST_RESULTS.md` - This document

---

## Logs

Test node logs are at:
- `testnet-data/node1.log`
- `testnet-data/node2.log`
- `testnet-data/node3.log`

Check for peer connection attempts:
```bash
grep -i "peer\|connect\|mdns" testnet-data/node*.log
```

---

**BOTTOM LINE**: The test infrastructure works, but **P2P peer discovery/connection is broken**. This is the blocker for testing transaction propagation.

Fix peer connectivity first, then the rest of the test will work.
