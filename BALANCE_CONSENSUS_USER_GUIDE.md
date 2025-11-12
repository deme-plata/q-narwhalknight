# Balance Consensus - User Guide

**Date**: 2025-11-03 21:15 CET
**Issue**: Mining to localhost doesn't show rewards on Server Beta
**Status**: IDENTIFIED & DOCUMENTED (Implementation planned for v0.9.0-beta)

---

## 🎯 TL;DR - What You Need to Know

**Problem**: You mine to localhost (port 8330), but your balance doesn't show on the frontend (which connects to Server Beta on port 8080).

**Why**: Each node has its own independent balance database. Mining rewards are stored locally and not automatically synced to other nodes.

**Solution Right Now**: **Mine directly to Server Beta**
```bash
./q-miner-v0.8.11-beta --node http://185.182.185.227:8080 --wallet qnkYOUR_WALLET
```

**Future Solution**: v0.9.0-beta will implement balance consensus (balances synced via blocks)

---

## 📊 Understanding the Architecture

### Current System (v0.8.11-beta)

```
┌─────────────────────────────────────┐
│   Your Localhost Node (Port 8330)  │
│   Database: ./data-local/           │
│   Your Balance: 1000 QNK ✅         │
└─────────────────────────────────────┘
         ↕ P2P (blocks only)
         ❌ Balances NOT synced
┌─────────────────────────────────────┐
│   Server Beta (Port 8080)           │
│   Database: /opt/.../data/          │
│   Your Balance: 0 QNK ❌            │
│                                     │
│   ← Frontend queries THIS node      │
└─────────────────────────────────────┘
```

**Result**: You have rewards (on localhost), but frontend shows 0 (queries Server Beta)

---

## ✅ Workarounds (Available Now)

### Option 1: Mine to Server Beta (RECOMMENDED)

**Advantages**:
- ✅ Works immediately
- ✅ Balance shows on frontend
- ✅ No configuration needed
- ✅ Most reliable

**How To**:
```bash
# Stop your current miner
pkill q-miner

# Download latest miner
wget https://quillon.xyz/downloads/q-miner-v0.8.11-beta
chmod +x q-miner-v0.8.11-beta

# Start mining to Server Beta
./q-miner-v0.8.11-beta \
  --node http://185.182.185.227:8080 \
  --wallet qnkYOUR_WALLET_ADDRESS \
  --threads 4
```

**Disadvantages**:
- Requires internet connection
- Depends on Server Beta availability
- Slightly higher latency

### Option 2: Query Localhost Node Directly

**Advantages**:
- See your actual balance
- Verify mining is working

**How To**:
```bash
# Check balance on localhost node
curl http://localhost:8330/api/wallet/qnkYOUR_WALLET_ADDRESS

# Or check node status
curl http://localhost:8330/api/node/info
```

**Disadvantages**:
- Manual queries required
- Frontend still shows 0
- Not user-friendly

### Option 3: Run Frontend Against Localhost

**Advantages**:
- See local balance in UI
- Full local development

**How To**:
1. Clone frontend repository
2. Edit API endpoint to `http://localhost:8330`
3. Run frontend locally
4. Access at `http://localhost:3000`

**Disadvantages**:
- Requires frontend setup
- Can't see other miners' activity
- Isolated from network

---

## 🔮 Future Solution: v0.9.0-Beta (In Development)

### What Will Change

**Balance Consensus Implementation**:
- Blocks will include `balance_updates` field
- When nodes sync blocks, they also sync balances
- All nodes will have identical balance state
- Mining to localhost will work (balances propagate via P2P)

### Timeline

**Development**: 1-2 days (implementation)
**Testing**: 2-3 days (critical to test thoroughly)
**Deployment**: Staged rollout (test → production)

**Estimated Release**: 1 week

### What You'll Need to Do

**When v0.9.0-beta releases**:
1. Download new miner binary
2. Download new node binary (if running your own node)
3. Restart miner/node
4. Balances will sync automatically

**No data loss**: Existing balances will be preserved

---

## 🧪 Technical Details (For Advanced Users)

### Current Balance Flow

**When you mine to localhost**:
1. Miner submits solution to localhost:8330
2. Localhost node validates solution
3. Localhost node updates its local database: `balance[YOUR_WALLET] += REWARD`
4. Localhost node produces block
5. Localhost node broadcasts block to network (P2P)
6. Server Beta receives block and stores it
7. ❌ **Server Beta does NOT update balances** (balance state not in blocks)

### v0.9.0 Balance Flow

**After v0.9.0-beta upgrade**:
1-5. Same as above
6. Server Beta receives block with `balance_updates` field
7. ✅ **Server Beta applies balance updates**: `balance[YOUR_WALLET] += REWARD`
8. ✅ **All nodes have identical balance state**

### Why Blocks Don't Include Balances Now

**Historical Reason**: Q-NarwhalKnight started as a hybrid architecture:
- Blockchain for blocks (DAG-Knight consensus)
- Local database for balances (RocksDB)

**Design Assumption**: Most miners would mine to public nodes (like Server Beta)

**Reality**: Users want to mine to localhost for:
- Privacy
- Control
- Local testing
- Offline operation

**Solution**: Add balance consensus to support local mining

---

## ❓ FAQ

### Q: Is my localhost mining wasted?
**A**: No! Your rewards exist in your localhost database. They're just not visible on Server Beta. You can query localhost directly to see them.

### Q: Can I transfer my localhost balance to Server Beta?
**A**: Not currently. Balances are node-local. Wait for v0.9.0-beta balance consensus.

### Q: What if I lose my localhost database?
**A**: Your rewards are lost (local storage only). This is why mining to Server Beta is recommended until v0.9.0-beta.

### Q: Can I run multiple miners to the same localhost node?
**A**: Yes! All miners can point to your localhost:8330 node.

### Q: Does Server Beta see my localhost blocks?
**A**: Yes, blocks are synced. Just not the balance state.

### Q: Will v0.9.0 break existing nodes?
**A**: No, it's backwards compatible. Old nodes ignore the new balance_updates field.

---

## 📞 Support

**Discord**: https://discord.gg/quillon-xyz
**GitHub Issues**: https://github.com/quantum-dag-labs/Q-NarwhalKnight/issues
**Documentation**: https://docs.quillon.xyz

---

## 📝 Summary

**Current State**:
- ✅ Localhost mining WORKS (rewards are added)
- ❌ Balances NOT synced between nodes
- ✅ Workaround: Mine to Server Beta

**Future State** (v0.9.0-beta):
- ✅ Localhost mining WORKS
- ✅ Balances SYNCED via blocks
- ✅ All nodes have identical balance state

**What To Do Now**:
Mine directly to Server Beta: `./q-miner-v0.8.11-beta --node http://185.182.185.227:8080 --wallet qnkYOUR_WALLET`

---

**Related Documentation**:
- `LOCALHOST_MINING_DESIGN_ANALYSIS.md` - Technical analysis
- `V0.9.0_BETA_BALANCE_CONSENSUS_IMPLEMENTATION_PLAN.md` - Implementation plan
- `V0.8.11_BETA_FINAL_STATUS.md` - Current system status
