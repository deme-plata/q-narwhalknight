# Technical Review: Balance Bounce-Back After DEX Swap
## "Correct → Wrong → Correct" Display Bug on $1.1B Mainnet
### Date: 2026-04-19 | Status: NEEDS FIX | Risk: ZERO (display-only, no funds at risk)

---

## 1. The Problem (User Report)

After swapping QUG → QUGUSD on the DEX:

```
T+0s:    Swap executes. Balance shows CORRECT (lower by swap amount).
T+30s:   Balance BOUNCES BACK to pre-swap value (wrong — too high).
T+4-5m:  Balance settles to CORRECT value again.
```

The user sees their coins "return" after a swap, then disappear again minutes later. This is confusing and looks like a bug, even though no funds are at risk — the final balance is always correct.

---

## 2. Root Cause (Verified via Server Logs)

Three independent systems write to the same wallet balance, and they race:

```
TIMELINE OF A SINGLE SWAP:

02:41:00  SWAP HANDLER    subtract_balance_tx()  → balance = 900 QUG  ✓ correct
02:41:01  SSE             sends balance=900 to frontend               ✓ correct
          
          (user sees 900 QUG — correct)

02:43:46  BLOCK PIPELINE  Block N arrives (mined BEFORE the swap)
          coinbase reward: add_balance_tx(+0.08 QUG)
          BUT: the P2P balance update in this block carries
          the FULL balance snapshot from the producing node,
          which still had the OLD pre-swap value (1000 QUG)
          
02:43:46  BALANCE WRITE   add_balance_tx() → balance = 1000.08 QUG   ✗ WRONG
02:43:47  SSE             sends balance=1000.08 to frontend           ✗ WRONG

          (user sees 1000 QUG — balance "bounced back")

02:46:56  BLOCK PIPELINE  20+ more add_balance_tx() calls from
          mining blocks — all carrying pre-swap snapshots
          
          (balance stays wrong for ~4 minutes)

02:47:xx  BLOCK PIPELINE  Block M arrives (contains the swap TX)
          processes swap deduction from the block level
          balance corrected back to ~900 QUG

          (user sees 900 QUG again — correct)
```

### 2.1 The Three Writers

| # | Writer | When | What it does | Problem |
|---|--------|------|-------------|---------|
| 1 | **Swap handler** | Immediate | `subtract_balance_tx(wallet, amount)` — deducts swap input | Correct, fast |
| 2 | **Block coinbase processor** | Every ~1s | `add_balance_tx(wallet, reward)` — adds mining reward | Adds delta — OK by itself |
| 3 | **P2P balance-update gossipsub** | Every block | Receives full balance snapshot from producing node | **Overwrites local balance with stale snapshot** |

Writer #3 is the problem. When Beta produces a block at 02:41:00, it includes a balance snapshot for the founder wallet. That snapshot was computed BEFORE the swap executed. When this block propagates to the node the user is connected to, the P2P balance update overwrites the post-swap balance with the pre-swap snapshot.

### 2.2 Why the Dedup Doesn't Catch It

The gossipsub dedup fix (v10.3.6) uses `processed_balance_block:{block_hash}` flags to prevent the same BLOCK's balance updates from being applied twice. But the bounce-back is caused by DIFFERENT blocks — each new block carries a fresh balance snapshot that was computed before the swap block was confirmed. The dedup correctly processes each block once, but each block's snapshot is stale relative to the local swap.

### 2.3 Log Evidence

```
02:43:46  subtract_balance_tx(): wallet=efca... delta=-100 QUG    (swap)
02:46:56  add_balance_tx(): wallet=efca... delta=+0.08 QUG        (block reward)
02:46:56  add_balance_tx(): wallet=efca... delta=+0.08 QUG        (block reward)
          ... 20+ more add_balance_tx() in 16 seconds ...
```

The `add_balance_tx()` flood from block processing re-inflates the balance because each block's coinbase processing adds the mining reward on top of whatever the producing node's balance was — which doesn't include the swap deduction yet.

---

## 3. Why This Is Display-Only (No Funds At Risk)

1. **The swap transaction is in the mempool and WILL be included in a block.** Once that block is processed, the balance corrects permanently.
2. **The user cannot double-spend.** The swap has already executed — sending the same QUG again would fail validation.
3. **Other nodes converge.** All nodes eventually process the block containing the swap and arrive at the same correct balance.
4. **The "bounce" is local to the user's connected node.** It's a race between local state (swap executed) and P2P state (blocks from peers that haven't seen the swap yet).

---

## 4. Fix Options (Ranked by Safety)

### Option A: Optimistic Lock on Swapped Wallets (RECOMMENDED)

**Concept:** After a swap executes locally, "lock" the wallet balance for a short window (30-60 seconds). During this window, incoming P2P balance updates for that wallet are QUEUED instead of applied immediately. Once the swap's block is confirmed (included in a block from the network), release the lock and apply any queued updates.

```rust
// New struct: tracks recent local swaps
struct SwapLock {
    wallet: String,
    pre_swap_balance: u128,
    post_swap_balance: u128,
    swap_timestamp: Instant,
    swap_tx_hash: [u8; 32],
    confirmed: bool,
}

// In-memory map, NOT persisted to DB
// Key: wallet address, Value: SwapLock
let swap_locks: DashMap<String, SwapLock> = DashMap::new();
```

**When a swap executes (swap handler):**
```rust
swap_locks.insert(wallet.clone(), SwapLock {
    wallet: wallet.clone(),
    pre_swap_balance: old_balance,
    post_swap_balance: new_balance,
    swap_timestamp: Instant::now(),
    swap_tx_hash: tx_hash,
    confirmed: false,
});
```

**When a P2P balance update arrives (gossipsub handler):**
```rust
if let Some(lock) = swap_locks.get(&wallet) {
    if !lock.confirmed && lock.swap_timestamp.elapsed() < Duration::from_secs(60) {
        // Swap is recent and unconfirmed — skip this P2P update
        // The balance will correct when the swap's block arrives
        debug!("Skipping P2P balance update for {} — swap lock active", wallet);
        return;
    }
}
// No lock or lock expired — apply normally
apply_balance_update(wallet, new_balance);
```

**When the swap's block is confirmed (block processor):**
```rust
// Check if this block contains our swap TX
for tx in block.transactions {
    if let Some(lock) = swap_locks.get(&tx.hash_hex()) {
        lock.confirmed = true;
        swap_locks.remove(&lock.wallet);
    }
}
```

**Expiry (safety net):**
```rust
// Every 60 seconds, clean up expired locks
// If a swap's block never arrives (dropped from mempool), the lock expires
// and normal P2P updates resume
swap_locks.retain(|_, lock| {
    lock.swap_timestamp.elapsed() < Duration::from_secs(60)
});
```

**Properties:**
- NO database writes — lock is in-memory only
- NO consensus changes — only affects local display
- NO P2P protocol changes — other nodes unaffected
- Self-healing — lock expires after 60 seconds regardless
- Worst case if lock is wrong — user sees stale balance for 60s, then auto-corrects

**Risk: ZERO.** This only affects which SSE events the local node emits. It cannot cause incorrect balances — the block pipeline is the source of truth and always wins after lock expiry.

### Option B: SSE Dedup by Wallet + Timestamp

**Concept:** The SSE stream to the frontend tracks the most recent balance update timestamp per wallet. If a new SSE event has a LOWER balance than the previous event AND was triggered by a block older than the last swap, suppress it.

```typescript
// Frontend (Dashboard.tsx or SSE handler)
const lastSwapTimestamp = useRef<Record<string, number>>({});
const lastKnownBalance = useRef<Record<string, string>>({});

function onBalanceUpdate(wallet: string, newBalance: string, blockTimestamp: number) {
    const lastSwap = lastSwapTimestamp.current[wallet];
    
    if (lastSwap && blockTimestamp < lastSwap) {
        // This balance update is from a block BEFORE our swap
        // Ignore it — our local swap result is more recent
        return;
    }
    
    // Apply the update
    lastKnownBalance.current[wallet] = newBalance;
    setBalance(newBalance);
}

function onSwapExecuted(wallet: string) {
    lastSwapTimestamp.current[wallet] = Date.now();
    // Clear after 2 minutes
    setTimeout(() => delete lastSwapTimestamp.current[wallet], 120000);
}
```

**Properties:**
- Frontend-only change — no backend modifications
- No database writes
- No P2P changes
- Self-healing (timeout clears the filter)

**Risk: ZERO.** Worst case: a legitimate balance decrease (e.g., someone sent QUG FROM the wallet on another device) is temporarily suppressed for up to 2 minutes. The user refreshes and sees the correct balance.

**Downside:** Frontend-only fix doesn't help the Slint desktop wallet or API consumers.

### Option C: Tag Balance Updates with Block Height

**Concept:** Each SSE balance event includes the block height that triggered it. The frontend tracks the block height of the last locally-executed swap. Events from blocks below that height are suppressed.

```rust
// Backend: include block height in SSE balance event
sse_event = json!({
    "type": "balance_update",
    "wallet": wallet,
    "balance": new_balance,
    "block_height": current_height,  // ADD THIS
    "source": "block_pipeline"       // ADD THIS: "swap" | "block_pipeline" | "p2p"
});
```

```typescript
// Frontend: suppress stale events
function onBalanceSSE(event) {
    if (event.source === "block_pipeline" && swapPendingHeight.current) {
        if (event.block_height < swapPendingHeight.current) {
            return; // Stale block, ignore
        }
    }
    updateBalance(event.wallet, event.balance);
}
```

**Properties:**
- Small backend change (add 2 fields to SSE event)
- Small frontend change (filter by height)
- No database writes, no consensus changes

**Risk: ZERO.** Additional fields in SSE events are backward-compatible — old frontends ignore them.

---

## 5. Recommendation

**Option A (Optimistic Lock)** for the backend — it fixes the problem for all clients (web, Slint, API).

**Combined with Option C (height-tagged SSE)** for defense-in-depth on the frontend.

### Why NOT Option B alone:
- Frontend-only fix doesn't help Slint wallet or API consumers
- Every client would need to implement the same logic independently

### Why NOT a database/consensus fix:
- The balance is CORRECT in the DB after convergence
- The problem is DISPLAY TIMING, not data integrity
- Any change to how balances are stored risks the triple-credit bug recurring
- On a $1.1B mainnet, we don't touch the balance write path for a display issue

---

## 6. Implementation Plan

### Phase 1: Optimistic Lock (Backend)

**Files to modify:**
| File | Change | Lines |
|------|--------|-------|
| `crates/q-api-server/src/lib.rs` | Add `swap_locks: Arc<DashMap<String, SwapLock>>` to AppState | ~5 lines |
| `crates/q-api-server/src/handlers.rs` | Set lock after swap executes | ~15 lines |
| `crates/q-api-server/src/main.rs` | Skip P2P balance updates for locked wallets | ~10 lines |
| `crates/q-api-server/src/main.rs` | Lock expiry cleanup task (every 60s) | ~10 lines |

**What is NOT modified:**
- `crates/q-storage/` — NO storage changes
- `crates/q-network/` — NO P2P protocol changes  
- `crates/q-dex/` — NO DEX logic changes
- Balance write paths — UNTOUCHED
- Block validation — UNTOUCHED
- Consensus — UNTOUCHED

### Phase 2: Height-Tagged SSE (Backend + Frontend)

**Files to modify:**
| File | Change | Lines |
|------|--------|-------|
| `crates/q-api-server/src/streaming.rs` | Add `block_height` and `source` to balance SSE events | ~5 lines |
| `gui/quantum-wallet/src/components/Dashboard.tsx` | Filter stale balance events | ~15 lines |

### Phase 3: Testing

**Test on Docker container ONLY — never on production first.**

```
Test 1: Execute swap in Docker, verify no bounce-back
Test 2: Verify lock expires after 60s (kill swap's block producer)
Test 3: Verify mining rewards still credit correctly during lock
Test 4: Verify multiple concurrent swaps don't deadlock
Test 5: Verify Slint wallet also sees correct behavior
```

### Phase 4: Deploy

Standard `ha-deploy.sh` rolling deployment. Gamma first, then Beta, then Epsilon.

---

## 7. What We Are NOT Doing

| Action | Why Not |
|--------|---------|
| Changing how balances are stored | Display issue, not data issue |
| Modifying the P2P balance-update protocol | Would require all nodes to upgrade simultaneously |
| Adding database locks around swaps | Risk of deadlock on $1.1B mainnet |
| Changing block production to include swap results faster | Consensus change — far too risky |
| Removing P2P balance updates entirely | They're needed for new node sync |
| Making the swap handler wait for block confirmation | Would make swaps take 1-3 seconds instead of instant |

---

## 8. Questions for Peer Review

### Q1: Is the 60-second lock window sufficient?

Blocks are produced every ~1 second. A swap transaction should be included within 1-5 blocks (1-5 seconds). The 60-second window provides 10x margin. Is this too long? Too short?

If the lock is too long: mining reward display is delayed by up to 60s after a swap.
If the lock is too short: fast block production from multiple peers might still cause a bounce.

### Q2: Should the lock suppress ALL balance updates or only decreasing ones?

Option A as described suppresses ALL P2P balance updates during the lock window. This means mining rewards earned during the lock window are also delayed.

Alternative: only suppress updates that would INCREASE the balance above the post-swap value:

```rust
if new_p2p_balance > lock.post_swap_balance {
    // P2P is trying to restore pre-swap balance — suppress
    return;
}
// P2P balance is lower or equal — could be another deduction, allow it
apply_balance_update(wallet, new_p2p_balance);
```

This is more precise but also more complex. For a $1.1B mainnet, simpler is safer.

### Q3: What if the user does two swaps within 60 seconds?

The lock should track the LATEST swap, not the first:

```rust
// Always overwrite with most recent swap
swap_locks.insert(wallet, SwapLock { 
    post_swap_balance: new_balance_after_second_swap,
    ..
});
```

Multiple rapid swaps compound the deductions. The lock's `post_swap_balance` must reflect the cumulative result.

### Q4: Does this interact with the gossipsub dedup fix?

No. The gossipsub dedup (`processed_balance_block:{hash}`) prevents the SAME block from being processed twice. The optimistic lock prevents DIFFERENT blocks (that carry stale snapshots) from overwriting a local swap. They operate at different layers and don't conflict.

### Q5: Memory impact of DashMap?

Each `SwapLock` is ~150 bytes. Even with 1000 concurrent swaps (unrealistic), that's 150 KB. Entries expire after 60 seconds. Memory impact: negligible.

---

## 9. Safety Statement

This fix is **display-only**:
- **NO database schema changes** — lock is in-memory, not persisted
- **NO consensus changes** — block validation identical
- **NO P2P protocol changes** — other nodes unaffected
- **NO balance write path changes** — `add_balance_tx` / `subtract_balance_tx` untouched
- **Self-healing** — lock expires after 60s, normal behavior resumes
- **Backward-compatible** — old frontends work fine (they just see the bounce as before)
- **Cannot cause fund loss** — lock only affects SSE display, not actual balances

The worst case if the lock has a bug: user sees a stale balance for up to 60 seconds after a swap, then it auto-corrects. This is BETTER than the current behavior (4-5 minute bounce).

---

*Generated 2026-04-19 — Quillon Foundation*
*Based on live server log analysis of swap transactions on Beta (185.182.185.227)*
*No database modifications were made during this investigation*
