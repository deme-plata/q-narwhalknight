# Mining Rewards SSE Bug - Diagnosis and Fix
## Why "Recent Mining Rewards" Shows "Waiting for mining rewards..."

**Bug Report Date**: 2025-11-18
**Affected Version**: v1.0.16-beta
**Severity**: Medium (UX issue - rewards ARE being paid, just not displayed)
**Impact**: Mining page shows "Waiting for mining rewards..." forever despite successful mining

---

## Problem Description

When you mine blocks, the mining page shows:
```
✅ Hashrate: 1.2 MH/s
✅ Personal Average: 45.3 KH/s
❌ Recent Mining Rewards: "Waiting for mining rewards..."
```

**Expected**: Should show real-time mining rewards via SSE (Server-Sent Events)
**Actual**: Forever shows "Waiting for mining rewards..."
**Root Cause**: SSE events for mining rewards are NOT being emitted

---

## Technical Analysis

### What's Actually Happening (Backend)

**Mining rewards ARE being created and paid** (`block_producer.rs:407-584`):

```rust
async fn create_coinbase_transactions(...) -> Result<Vec<Transaction>> {
    // ✅ Mining rewards ARE calculated correctly
    let total_reward = 5_000_000;  // 0.05 QUG
    let dev_fee = 50_000;  // 1%
    let miner_reward = 4_950_000 / solutions.len();  // 99% split

    // ✅ Coinbase transactions ARE created
    for solution in solutions {
        transactions.push(Transaction {
            from: [0u8; 32],  // Coinbase (newly minted)
            to: solution.miner_address,  // Your wallet
            amount: miner_reward,
            // ...
        });
    }

    info!("💎 Created {} coinbase transactions", transactions.len());
    info!("   ⛏️  Each Miner Gets: {} QUG", miner_reward / 100_000_000.0);

    // ❌ BUT: No SSE event emitted here!
    // Missing: event_emitter.emit(MiningReward { ... })

    Ok(transactions)
}
```

**Proof rewards are working**:
1. Check your wallet balance - it IS increasing ✅
2. Query API: `curl http://localhost:8080/api/v1/wallet/{address}/balance` ✅
3. Check blockchain - coinbase transactions exist ✅

**Problem**: Frontend never receives notification via SSE ❌

---

### Why Frontend Sees Nothing

**Mining page code** (`MiningScreen.tsx`):

```typescript
// ✅ Correctly listening for SSE events
useEffect(() => {
    const eventSource = new EventSource('/api/v1/events/stream');

    eventSource.addEventListener('mining_reward', (event) => {
        const reward = JSON.parse(event.data);
        setRecentRewards(prev => [reward, ...prev].slice(0, 10));
    });

    // ...
}, []);
```

**But backend NEVER sends** `'mining_reward'` events!

---

### Root Cause: Missing SSE Event Emission

**Where events SHOULD be emitted** (`block_producer.rs`):

```rust
// CURRENT CODE (BROKEN):
async fn create_coinbase_transactions(...) {
    // Create mining reward transactions
    for solution in solutions {
        transactions.push(miner_reward_tx);
    }

    // ❌ NO SSE EVENT EMITTED!

    Ok(transactions)
}

// SHOULD BE (FIXED):
async fn create_coinbase_transactions(...) {
    // Create mining reward transactions
    for solution in solutions {
        transactions.push(miner_reward_tx);

        // ✅ EMIT SSE EVENT
        if let Some(ref emitter) = self.event_emitter {
            emitter.emit(StreamEvent::MiningReward {
                miner_address: hex::encode(solution.miner_address),
                amount_qug: miner_reward as f64 / 100_000_000.0,
                block_height: block_height,
                timestamp: Utc::now(),
            }).await?;
        }
    }

    Ok(transactions)
}
```

---

## Fix Implementation

### Step 1: Add `MiningReward` Event Type

**File**: `crates/q-api-server/src/streaming.rs`

**Add to `StreamEvent` enum** (after line 73):

```rust
pub enum StreamEvent {
    // ... existing events ...

    /// Mining reward paid to miner
    MiningReward {
        miner_address: String,
        amount_qug: f64,
        block_height: u64,
        block_hash: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    // ... rest of events ...
}
```

### Step 2: Add Event Emitter to BlockProducer

**File**: `crates/q-api-server/src/block_producer.rs`

**Add field to struct** (around line 70):

```rust
pub struct BlockProducer {
    // ... existing fields ...

    /// Event emitter for real-time SSE updates
    event_emitter: Option<Arc<HighPerformanceEmitter>>,
}
```

**Update constructor** (around line 130):

```rust
impl BlockProducer {
    pub fn new(
        // ... existing parameters ...
        event_emitter: Option<Arc<HighPerformanceEmitter>>,
    ) -> Self {
        Self {
            // ... existing fields ...
            event_emitter,
        }
    }
}
```

### Step 3: Emit Events When Rewards Are Created

**File**: `crates/q-api-server/src/block_producer.rs`

**In `create_coinbase_transactions`** (after line 560):

```rust
async fn create_coinbase_transactions(...) -> Result<Vec<Transaction>> {
    // ... existing code to create transactions ...

    // Transaction 2-N: Miner rewards (99% split among all miners)
    for (idx, solution) in solutions.iter().enumerate() {
        let miner_tx_id = { /* ... */ };

        let reward_tx = Transaction {
            id: miner_tx_id,
            from: coinbase_from,
            to: solution.miner_address,
            amount: miner_reward_per_solution,
            // ... rest of transaction ...
        };

        transactions.push(reward_tx.clone());

        // ✅ EMIT SSE EVENT FOR EACH MINER REWARD
        if let Some(ref emitter) = self.event_emitter {
            let event = StreamEvent::MiningReward {
                miner_address: hex::encode(solution.miner_address),
                amount_qug: miner_reward_per_solution as f64 / 100_000_000.0,
                block_height,
                block_hash: hex::encode(prev_block_hash),  // Add this parameter
                timestamp: timestamp,
            };

            if let Err(e) = emitter.emit(event).await {
                warn!("Failed to emit mining reward SSE event: {}", e);
                // Don't fail block production if SSE fails
            }
        }
    }

    Ok(transactions)
}
```

### Step 4: Pass Event Emitter When Creating BlockProducer

**File**: `crates/q-api-server/src/main.rs`

**When initializing block producer** (search for `BlockProducer::new`):

```rust
let block_producer = Arc::new(
    BlockProducer::new(
        // ... existing parameters ...
        Some(event_emitter.clone()),  // ✅ Pass the event emitter
    )
);
```

---

## Verification Steps

### 1. Check SSE Events Are Emitted

**After applying fix, check logs**:

```bash
# Start node with debug logging
RUST_LOG=debug ./target/release/q-api-server

# Look for SSE emission logs
grep "emit.*MiningReward" node.log

# Expected:
# [DEBUG] Emitted SSE event: MiningReward { miner_address: "abc123...", amount: 0.0495 }
```

### 2. Test SSE Endpoint Directly

**Connect to SSE stream**:

```bash
# Listen to SSE events
curl -N http://localhost:8080/api/v1/events/stream

# Expected output when block is mined:
event: mining_reward
data: {"miner_address":"abc123...","amount_qug":0.0495,"block_height":12345,"timestamp":"2025-11-18T10:00:00Z"}
```

### 3. Check Frontend Receives Events

**Open browser DevTools → Network → Find `events/stream`**:

```
event: mining_reward
data: {"miner_address":"efca1e8c...","amount_qug":0.0495,...}
```

**Check mining page updates**:
- "Waiting for mining rewards..." should disappear
- Recent rewards list should populate
- Should see: "Block #12345: +0.0495 QUG"

---

## Alternative Quick Fix (Without Code Changes)

**If you can't rebuild**, use API polling instead of SSE:

### Update Frontend to Poll Transactions

**File**: `gui/quantum-wallet/src/components/MiningScreen.tsx`

```typescript
// Replace SSE listener with API polling
useEffect(() => {
    const pollRewards = async () => {
        try {
            // Query recent blocks
            const response = await fetch('/api/v1/blocks/recent?limit=10');
            const blocks = await response.json();

            // Extract coinbase transactions (mining rewards)
            const rewards = blocks.data.flatMap(block =>
                block.transactions
                    .filter(tx =>
                        tx.from === "0000000000000000000000000000000000000000000000000000000000000000" && // Coinbase
                        tx.to === walletAddress  // Your wallet
                    )
                    .map(tx => ({
                        block_height: block.height,
                        amount_qug: tx.amount / 100_000_000,
                        timestamp: tx.timestamp,
                    }))
            );

            setRecentRewards(rewards);
        } catch (error) {
            console.error('Failed to fetch mining rewards:', error);
        }
    };

    // Poll every 5 seconds
    const interval = setInterval(pollRewards, 5000);
    pollRewards();  // Initial fetch

    return () => clearInterval(interval);
}, [walletAddress]);
```

**Pros**:
- Works immediately (no backend rebuild needed)
- Shows historical rewards
- More reliable than SSE (no connection drops)

**Cons**:
- Less real-time (5s delay)
- More server load (API calls every 5s)
- Not as elegant as SSE

---

## Why This Bug Exists

**Historical reasons**:

1. **Block production was implemented first** - Focus was on correctness, not UI events
2. **Mining rewards ARE paid correctly** - No functional bug, just missing notification
3. **SSE system was added later** - Event types added incrementally
4. **Frontend expected SSE** - But backend never wired it up

**Similar working examples**:

```rust
// TransactionSubmitted event (WORKING):
let event = StreamEvent::TransactionSubmitted {
    transaction: signed_transaction.clone(),
    timestamp: Utc::now(),
};
state.event_emitter.emit_immediate(event).await?;
// ✅ This works! Mining rewards need same pattern.
```

---

## Impact Assessment

### What Works

✅ Mining rewards are calculated correctly
✅ Coinbase transactions are created
✅ Balances are updated correctly
✅ Blockchain records all rewards
✅ API queries return reward transactions

### What's Broken

❌ Real-time SSE notifications
❌ Mining page "Recent Rewards" UI
❌ Frontend notification toasts

### User Impact

**Low severity because**:
- Rewards ARE being paid (check wallet balance)
- No financial loss
- Just a UI/UX issue

**Annoying because**:
- Can't see rewards in real-time
- Looks like mining isn't working (even though it is)
- Hurts user confidence

---

## Quick Diagnostic Commands

### Verify You're Actually Getting Rewards

```bash
# 1. Check your wallet balance (should be increasing)
curl http://localhost:8080/api/v1/wallet/{your_address}/balance

# 2. Query recent blocks with coinbase transactions
curl http://localhost:8080/api/v1/blocks/recent?limit=5 | jq '.data[] | {height, coinbase_tx_count: (.transactions | map(select(.from == "0000000000000000000000000000000000000000000000000000000000000000")) | length)}'

# 3. Check if YOUR address received rewards
curl http://localhost:8080/api/v1/wallet/{your_address}/transactions | jq '.data[] | select(.from == "0000000000000000000000000000000000000000000000000000000000000000")'
# This shows coinbase transactions TO your address (mining rewards)
```

**If these show rewards**: Mining IS working, just SSE notification is broken ✅

---

## Recommended Priority

**Priority**: Medium
**Effort**: Low (30 minutes to implement + test)
**Impact**: High (much better UX)

**Suggested timeline**:
1. **Immediate**: Use API polling workaround in frontend
2. **Next release** (v1.0.17-beta): Add proper SSE events

---

## Related Issues

- Frontend expects `mining_reward` SSE events (listening but never receives)
- Backend creates rewards but doesn't notify
- Similar issue might exist for other transaction types

**Other events that SHOULD be emitted**:
- Block finaliz

ation (when block gets 2-chain commit)
- Transaction confirmation
- Balance updates

---

**END OF DIAGNOSIS**

**Status**: Bug confirmed, workaround available, proper fix straightforward
**Next Steps**: Implement SSE event emission in block producer
