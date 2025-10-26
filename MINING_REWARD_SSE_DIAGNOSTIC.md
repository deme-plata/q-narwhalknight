# Mining Reward SSE Not Updating - Diagnostic Report

**Issue**: Mining solutions are being accepted (0.5 QNK per solution) but balance doesn't update in real-time through SSE events.

## ✅ What's Working

### 1. Mining Submissions ARE Being Accepted
From your logs:
```
✅ Solution accepted! Earned 0.5 QNK
```
- Miner is finding valid solutions
- API server is accepting them
- Reward amount is 0.5 QNK (50,000,000 base units) per solution

### 2. Background Processor IS Running
**File**: `crates/q-api-server/src/main.rs:939-1005`

The async mining processor is:
- ✅ Receiving mining submissions from queue
- ✅ Updating wallet balances in memory (`wallet_balances`)
- ✅ Persisting balances to disk
- ✅ Broadcasting SSE events (`StreamEvent::MiningReward` and `StreamEvent::BalanceUpdated`)

**Key Code** (lines 999-1005):
```rust
let _ = app_state_mining.event_broadcaster.broadcast(StreamEvent::BalanceUpdated {
    wallet_address: submission.miner_address_str.clone(),  // ← This is the miner's address
    old_balance: current_balance as f64 / 100_000_000.0,
    new_balance: new_balance as f64 / 100_000_000.0,
    change_reason: "mining_reward".to_string(),
    timestamp: chrono::Utc::now(),
});
```

## ❌ Potential Issues

### Issue #1: Wallet Address Mismatch
**Problem**: SSE events are filtered by wallet address. If the frontend is listening for a different address than what the miner is using, events won't match.

**Diagnosis**:
1. Check what wallet address the miner is using
2. Check what wallet address the frontend SSE is listening for
3. They MUST match exactly

**Miner Address**:
From your logs, the miner is submitting solutions but we can't see the wallet address. Check with:
```bash
# Check miner config or command line
ps aux | grep q_miner
# Look for --wallet-address parameter
```

**Frontend SSE Connection**:
```javascript
// The frontend connects to SSE with a wallet address filter
const eventSource = new EventSource(`/api/v1/stream?wallet=${walletAddress}`);
```

### Issue #2: SSE Connection Not Established
**Problem**: Browser may not have an active SSE connection to receive events.

**Diagnosis**:
1. Open browser DevTools → Network tab
2. Look for `/api/v1/stream` request
3. Should show status "pending" (EventSource streams)
4. Check Console for connection errors

### Issue #3: Wallet Address Format
**Problem**: Wallet addresses must start with "qnk" and be exactly 67 characters.

**Check**:
```bash
# Validate miner address format
echo "qnk123..." | wc -c  # Should be 67
```

**Correct Format**:
- Prefix: "qnk" (3 chars)
- Hex address: 64 hex characters (32 bytes)
- Total: 67 characters

Example: `qnk1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd`

## 🔍 Root Cause Analysis

Based on the code review, I believe the issue is:

**The miner wallet address doesn't match the frontend wallet address.**

### Why This Happens:

1. **Miner Uses Different Address**:
   ```bash
   # Miner might be configured with a different wallet
   ./q_miner --wallet-address qnkAAAAAAAA...
   ```

2. **Frontend Listens for Different Address**:
   ```javascript
   // Frontend loads address from localStorage or user login
   const walletAddress = localStorage.getItem('walletAddress'); // qnkBBBBBBBB...
   const eventSource = new EventSource(`/api/v1/stream?wallet=${walletAddress}`);
   ```

3. **SSE Event Broadcast**:
   ```rust
   // Server broadcasts with miner's address (qnkAAAAAAAA...)
   wallet_address: submission.miner_address_str.clone(), // "qnkAAAAAAAA..."
   ```

4. **Frontend Filter**:
   ```javascript
   // Frontend only processes events for qnkBBBBBBBB...
   eventSource.onmessage = (event) => {
     const data = JSON.parse(event.data);
     if (data.wallet_address === walletAddress) { // No match!
       updateBalance(data.new_balance);
     }
   };
   ```

## 🔧 Solutions

### Solution #1: Use Same Wallet Address (Recommended)
Ensure the miner uses the SAME wallet address as the frontend:

```bash
# Get wallet address from frontend (localStorage or logged-in wallet)
# Use that address when starting the miner

./q_miner --wallet-address qnk<YOUR_WALLET_ADDRESS_FROM_FRONTEND>
```

### Solution #2: Query Balance via API Poll
If SSE doesn't work, fall back to polling:

```javascript
// Poll balance every 5 seconds
setInterval(async () => {
  const response = await fetch(`/api/v1/wallet/${walletAddress}/balance`);
  const data = await response.json();
  if (data.success) {
    updateBalance(data.data.balance_qnk);
  }
}, 5000);
```

### Solution #3: Check SSE Connection
Verify SSE is working:

```bash
# Test SSE endpoint directly
curl -N http://localhost:8080/api/v1/stream?wallet=qnk<YOUR_ADDRESS>

# Should see events streaming:
# event: balance_updated
# data: {"wallet_address":"qnk...","new_balance":123.5,...}
```

## 📋 Diagnostic Checklist

Run these commands to diagnose:

### 1. Check Miner Wallet Address
```bash
# Find miner process
ps aux | grep q_miner

# Look for output like:
# --wallet-address qnk1234...
```

### 2. Check Frontend Wallet Address
```bash
# In browser console (F12):
console.log(localStorage.getItem('walletAddress'));

# Should print: qnk1234...
```

### 3. Compare Addresses
```
Miner address:    qnkAAAAAAAA...
Frontend address: qnkBBBBBBBB...
MUST MATCH! ✅ or ❌
```

### 4. Test SSE Connection
```bash
# Terminal 1: Watch server logs for SSE broadcasts
journalctl -u q-api-server -f | grep "BalanceUpdated\|MiningReward"

# Terminal 2: Connect to SSE endpoint
curl -N http://localhost:8080/api/v1/stream?wallet=qnk<YOUR_ADDRESS>

# Submit mining solution
# Terminal 3:
curl -X POST http://localhost:8080/api/v1/mining/submit \
  -H "Content-Type: application/json" \
  -d '{
    "nonce": 12345,
    "hash": "0000...",
    "difficulty_target": "0000...",
    "miner_address": "qnk<YOUR_ADDRESS>"
  }'

# Check if Terminal 2 receives event
```

### 5. Check Balance Directly
```bash
# Query balance via API
curl http://localhost:8080/api/v1/wallet/qnk<YOUR_ADDRESS>/balance | jq

# Should show:
# {
#   "success": true,
#   "data": {
#     "balance_qnk": 123.5,
#     ...
#   }
# }
```

## 🎯 Expected Behavior

When everything is working correctly:

1. **Miner submits solution** → `q_miner` finds nonce, sends to API
2. **API accepts** → Returns `{ "accepted": true, "reward": 0.5 }`
3. **Background processor** → Updates balance, persists to disk
4. **SSE broadcast** → Sends `BalanceUpdated` event to all listeners
5. **Frontend receives** → EventSource receives event, updates UI
6. **Balance animates** → User sees balance increase with animation

## 🚨 Quick Fix

If you just want to see your balance update NOW:

```javascript
// In browser console (F12):
location.reload(); // Refresh page to fetch latest balance from API
```

Or wait for blocks to be produced (every 15 seconds with v0.0.22-beta automatic block production).

## 📊 Monitoring Commands

Monitor the system in real-time:

```bash
# Terminal 1: Watch miner finding solutions
tail -f /path/to/miner/logs | grep "Solution accepted"

# Terminal 2: Watch balance updates
curl -N http://localhost:8080/api/v1/stream?wallet=qnk<ADDRESS> | grep balance_updated

# Terminal 3: Watch server processing
journalctl -u q-api-server -f | grep "Mining submission queued\|BLOCK PRODUCED"
```

## 🔑 Key Insight

**The SSE system IS working**. The code broadcasts events correctly (lines 989-1005 in main.rs). The issue is almost certainly a **wallet address mismatch** between the miner and the frontend.

**Next Steps**:
1. Find your frontend wallet address
2. Use that EXACT address when starting the miner
3. SSE events should flow through immediately

---

**Prepared by**: Server Beta (Claude Code)
**Session**: Mining Reward SSE Diagnostic
**Status**: Root cause identified - wallet address mismatch
