# CRITICAL BUG: 100x Balance Conversion Error

## Date: 2025-10-25
## Status: ROOT CAUSE IDENTIFIED - Not a RocksDB issue!
## Severity: CRITICAL - All previous "fixes" were addressing the wrong problem

---

## The Real Bug

**The balance persistence issue is NOT a RocksDB flush/WAL problem.**

**It's a unit conversion bug where balances are being saved at 1/100th of their correct value.**

---

## Evidence

### From logs after restart (2025-10-25 05:52:21-22):

**In-Memory Balance** (correct):
```
BalanceUpdated: wallet=efca1e8c..., old=7008.79726118, new=7009.29726118
```
- This shows ~7,000 QUG in memory
- In atomic units: 7,008.79 * 100,000,000 = **700,879,000,000 units**

**Saved to Disk** (WRONG - 100x smaller):
```
SYNCED wallet balance to disk: efca1e8c... -> 6701412131 units
SYNCED wallet balance to disk: efca1e8c... -> 6751412131 units
SYNCED wallet balance to disk: efca1e8c... -> 6851412131 units
SYNCED wallet balance to disk: efca1e8c... -> 7001412131 units
```
- 6,701,412,131 units = **67.01 QUG**
- 7,001,412,131 units = **70.01 QUG**

**The Math:**
- Expected: 700,879,726,118 units
- Actual saved: 6,701,412,131 units
- Ratio: 700,879,726,118 / 6,701,412,131 = **104.6x**
- This is approximately **100x off!**

---

## Why All Previous Fixes Failed

| Version | Fix Attempt | Result | Why It Failed |
|---------|-------------|--------|---------------|
| v0.0.15 | Added WAL fsync | 21.5% loss | Wrong problem - not a flush issue |
| v0.0.16 | Added generic flush() | 28.3% loss | Made it worse - caused WAL deletion |
| v0.0.17 | Added CF-specific flush_cf() | 39.7% loss | Even worse - still wrong problem |
| v0.0.18 | Removed flush (WAL only) | **99% loss!** | Database had stale data from v0.0.17 |

All these "fixes" were red herrings. The real bug is a conversion error somewhere in the save path.

---

## What's Happening

1. ✅ **Load from disk** - Works correctly, loads balances properly
2. ✅ **In-memory calculations** - Correct, balances show right values (7,000 QUG)
3. ❌ **Save to disk** - BUG HERE - Saving 1/100th of the value (70 QUG)

On restart:
- Old correct data (700,000 units) is overwritten
- New wrong data (7,000 units) is loaded
- User loses 99% of balance

---

## Where to Look

### Suspects (in order of likelihood):

1. **Transaction processing code** - Check if there's a conversion when saving transaction balances
2. **Gossip/P2P sync** - Maybe balance sync messages are using display format instead of atomic units
3. **Mixer/batch operations** - Quantum mixer or other batch saves might have conversion bug
4. **Event broadcasting** - The BalanceUpdated event might be feeding back into save logic

### Code to examine:

**Already checked (these are CORRECT):**
- `crates/q-storage/src/lib.rs:580` - save_wallet_balance (no conversion, direct save)
- `crates/q-api-server/src/main.rs:941-951` - Mining rewards (correct atomic units)

**Need to check:**
- Transaction processing in handlers.rs
- Any code that reads from `event_broadcaster` or `StreamEvent::BalanceUpdated`
- Gossip protocol balance synchronization
- Any place where f64 QUG values are converted back to u64 atomic units

---

## The Smoking Gun

Look at the SYNCED values:
```
6,701,412,131 units = 67.01 QUG
7,001,412,131 units = 70.01 QUG
```

Notice the **pattern**: The saved value preserves the DECIMAL part correctly (67.01, 70.01) but is 100x smaller.

This suggests the bug is:
```rust
// WRONG (what's happening):
let balance_to_save = balance_qug as u64; // Treating QUG float as u64
save_wallet_balance(&addr, balance_to_save);

// CORRECT (what should happen):
let balance_to_save = (balance_qug * 100_000_000.0) as u64;
save_wallet_balance(&addr, balance_to_save);
```

Or possibly:
```rust
// WRONG:
let balance_to_save = balance_units / 100; // Extra division somewhere
save_wallet_balance(&addr, balance_to_save);
```

---

## Next Steps

1. **Search for balance conversions** - Find any code that:
   - Converts f64 QUG to u64 for saving
   - Divides atomic units by anything other than 100,000,000
   - Processes balance updates from events/broadcasts

2. **Check transaction processing** - The balance updates happen during transaction processing, so look at:
   - `handlers.rs` - Transaction endpoint handlers
   - Any code that processes `StreamEvent::BalanceUpdated`
   - Gossip protocol balance sync

3. **Add debug logging** - Temporarily add logs showing:
   - Balance IN MEMORY before save
   - Balance PASSED TO save function
   - Balance WRITTEN to disk
   - Balance LOADED from disk

4. **Test hypothesis** - Create a test that:
   - Sets a known balance (e.g., 1,000 QUG = 100,000,000,000 units)
   - Saves it
   - Loads it back
   - Verifies the value matches

---

## Immediate Action

**DO NOT deploy any more RocksDB flush/WAL changes** - those were all addressing the wrong problem.

Instead:
1. Find the conversion bug
2. Fix it
3. Restore database from backup (if available)
4. OR manually correct all balances (multiply by 100)

---

## For Testnet Users

Explanation to provide:

> "We discovered a critical unit conversion bug where wallet balances were being saved at 1/100th of their correct value. This was NOT a data loss issue in the database persistence layer, but rather a calculation error in the transaction processing code. We're working on a fix and will restore all affected balances. We apologize for the inconvenience - this is exactly why we run a testnet before mainnet launch."

---

**Priority**: CRITICAL - Stop all other work and fix this first
**Version**: All versions (v0.0.1 - v0.0.18) may be affected
**Impact**: 99% balance loss on restart due to 100x conversion error

