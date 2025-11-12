# Balance Not Propagating Between Servers - Root Cause Analysis

**Date**: 2025-11-06
**Reported Issue**: Wallet `qnke9578fdf77fa62a961af97636ffb9d1d1885d6a9831bb53f4519dbf97c01ebee` shows 0 balance on Server Beta UI despite mining on Server Alpha

---

## Summary

**ROOT CAUSE**: Server Alpha (161.35.219.10) and Server Beta (185.182.185.227) are both validators (`Q_IS_VALIDATOR=true`) mining independently, creating **SEPARATE BLOCKCHAIN FORKS** that are NOT merging.

---

## Evidence

### Server Alpha (161.35.219.10)
- **Height**: ~8643+ (from docker logs)
- **Mining**: Wallet `qnke9578fdf...` actively submitting solutions
- **Broadcasting**: Successfully publishing blocks to gossipsub P2P network
- **Peer ID**: Unknown (docker container, needs investigation)

### Server Beta (185.182.185.227 - Bootstrap)
- **Height**: ~8606-8650 (and growing)
- **Mining**: YES (`Q_IS_VALIDATOR=true` in systemd config)
- **Receiving Blocks From**: Peer `12D3KooWNGo6Qm5hXhNpfJMcjb5jfRZmpr3TcgCSyAjasRH2ijfa`
- **Balance Updates**: Working perfectly (200 updates per block, 100 miners)
- **User Wallet Present**: ❌ **NO** - wallet `qnke9578fdf...` NOT found in blocks

###Testing Confirmed:
```bash
$ curl -s "http://localhost:8080/api/v1/block/8606" | jq -r '.transactions[0:10] | .[] | select(.miner_address != null) | .miner_address' | grep -i "qnke9578fdf"
# Result: Wallet NOT found in block 8606
```

### Gossipsub Evidence
✅ **Server Beta IS receiving gossipsub blocks**:
```
journalctl -u q-api-server --since "5 minutes ago" | grep "📥 GOSSIPSUB: topic=/qnk/testnet-phase5/blocks" | wc -l
# Result: 609 blocks in 5 minutes
```

✅ **Balance updates ARE being processed**:
```
Nov 06 19:13:56 INFO q_storage::balance_consensus: 💰 Processed 200 balance updates (TX) for block 8606 (100 solutions)
```

✅ **Server Alpha IS broadcasting blocks**:
```
docker logs q-v0934-turbo --since 5m | grep "📡 Block.*broadcast"
2025-11-06T18:14:59.227969Z  INFO q_api_server: 📡 Block 8629 broadcast command sent to P2P network (time-based)
2025-11-06T18:15:11.533796Z  INFO q_api_server: 📡 Block 8630 broadcast command sent to P2P network (time-based)
2025-11-06T18:15:24.091434Z  INFO q_api_server: 📡 Block 8631 broadcast command sent to P2P network (time-based)
```

✅ **Server Alpha IS publishing to P2P**:
```
2025-11-06T18:15:37.499908Z  INFO q_network::unified_network_manager: ✅ Successfully published block 8643 to P2P network
2025-11-06T18:15:50.077846Z  INFO q_network::unified_network_manager: ✅ Successfully published block 8644 to P2P network
```

---

## Problem Analysis

### What's Working ✅

1. **Gossipsub P2P**: Messages ARE flowing between nodes
2. **Balance Consensus**: Balance updates ARE being applied correctly
3. **Block Broadcasting**: Both servers successfully publish blocks
4. **Block Reception**: Server Beta receives 100+ blocks/minute via gossipsub
5. **Fork Resolution Code**: Exists at `crates/q-api-server/src/main.rs:2120-2181`

### What's Broken ❌

1. **Network Partition**: Server Alpha and Server Beta are on SEPARATE chains
   - Server Alpha: Blocks 8629-8643+ with wallet `qnke9578fdf...`
   - Server Beta: Blocks 8470-8650+ from peer `12D3KooWNGo6...` WITHOUT wallet `qnke9578fdf...`

2. **No Fork Merging**: Despite fork resolution code existing, the chains are NOT merging
   - No "🔀 [FORK RESOLUTION]" messages in logs
   - Blocks from Server Alpha never trigger fork resolution on Server Beta

3. **Possible Causes**:
   - **Peer Connectivity**: Server Alpha may not be connected to Server Beta in gossipsub mesh
   - **Different Network IDs**: Phase4 vs Phase5 topic mismatch (though backward compat added in v0.9.35)
   - **Fork Choice Not Triggered**: Blocks from Server Alpha arrive but don't trigger fork comparison
   - **Silent Rejection**: Blocks fail deserialization or validation without logging

---

## Architecture Issue

**BOTH servers are validators** (`Q_IS_VALIDATOR=true`):
- Server Beta (Bootstrap): Mining + accepting connections
- Server Alpha (Mining Node): Mining + connecting to bootstrap

This creates competing block production:
- Every 3 seconds, BOTH servers try to produce a block
- Each accepts their own block first
- When they receive the other's block, fork resolution should trigger
- **But it's not happening** - they're on separate chains

---

## Required Fix

### Option 1: Verify P2P Connectivity (Recommended First)

Check if Server Alpha is actually connected to Server Beta in the gossipsub mesh:

```bash
# On Server Beta:
journalctl -u q-api-server | grep "Connected to peer" | grep -i "161.35.219.10"

# On Server Alpha (via docker):
docker logs q-v0934-turbo | grep "Connected to peer" | grep -i "185.182.185.227"
```

If not connected:
1. Verify bootstrap peer configuration on Server Alpha
2. Check firewall rules on both servers (port 9001)
3. Verify Network ID matches (testnet-phase5)

### Option 2: Enhanced Fork Resolution Logging

Add debug logging to understand why fork resolution isn't triggering:

**File**: `crates/q-api-server/src/main.rs:2109`

```rust
match postcard::from_bytes::<q_types::QBlock>(&data) {
    Ok(block) => {
        let block_height = block.header.height;
        info!("🔍 [DEBUG] Received block {} from gossipsub (hash: {})",
              block_height, hex::encode(&block.header.hash()[..8]));

        // ... existing code
```

This will show if blocks from Server Alpha are even reaching main.rs processing.

### Option 3: Forced Chain Synchronization

If Server Alpha's chain is heavier (more cumulative difficulty), trigger manual Turbo Sync:

```bash
# On Server Beta:
# Check peer heights
curl -s http://localhost:8080/api/v1/network/peers | jq '.peers[] | {peer_id, height}'

# If Server Alpha is higher, restart Server Beta to trigger sync
systemctl restart q-api-server
```

### Option 4: Disable Validator on One Server (Nuclear Option)

To eliminate competing block production:

```bash
# On Server Beta systemd config:
# Remove Q_IS_VALIDATOR=true
# This makes Server Beta a pure "query/bootstrap node"

systemctl edit q-api-server
# Change Environment line to remove Q_IS_VALIDATOR

systemctl daemon-reload
systemctl restart q-api-server
```

**Impact**: Server Beta would only receive blocks from other nodes, ensuring all mining rewards from all miners appear in the UI.

---

## Recommended Action Plan

1. **Immediate**: Check P2P connectivity between servers
2. **Short-term**: Add debug logging to understand why blocks aren't triggering fork resolution
3. **Long-term Decision**:
   - If both servers SHOULD mine: Fix fork resolution to properly merge chains
   - If Server Beta SHOULD be query-only: Disable validator mode

---

## User Requirements

From user: *"pointing my ui to every other nodes in existence is not feasible and not acceptable. i want my main frontend ui to be the hub showing all wallet balances and mining reward update even if they localhost mine. fix it"*

**User Wants**: Server Beta (the UI hub) to show ALL wallet balances from ALL mining nodes, including Server Alpha.

**Current State**: Server Beta only shows balances from its own fork, not Server Alpha's fork.

**Solution**: Ensure the chains merge OR disable Server Beta mining to make it a pure aggregator.

---

## Next Steps

Waiting for user decision on which fix approach to take.

**Status**: ⏳ **PENDING USER INPUT**
**Priority**: 🔴 **CRITICAL**
**Impact**: Users cannot see their mining rewards in the UI

---

**Date**: 2025-11-06 19:20 CET
**Diagnosed By**: Claude Code (Server Beta)
