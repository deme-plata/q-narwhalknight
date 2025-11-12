# DEX Balance 10x Display Issue - Diagnostic

## Error Message
```
❌ Swap failed: Insufficient QUG balance
Required: 96,480,538,004 base units
Available: 96,470,811,004 base units
```

## Unit Conversions
```
Available Balance:
- Base units: 96,470,811,004
- ÷ 1e8 (correct): 964.70811004 QUG ✅
- ÷ 1e9 (wrong):   96.47081100 QUG ❌

Required for Swap:
- Base units: 96,480,538,004
- ÷ 1e8 (correct): 964.80538004 QUG ✅
- ÷ 1e9 (wrong):   96.48053800 QUG ❌

Shortfall:
- Base units: 9,727,000
- QUG: 0.09727 QUG
```

## The Real Issue

You're trying to swap 964.80 QUG but only have 964.70 QUG - short by 0.09727 QUG!

## But Why "10x Wrong"?

If the frontend balance display is showing `96.47 QUG` instead of `964.70 QUG`, then the display is dividing by **1e9 instead of 1e8** - showing 10x less than you actually have!

## Check Frontend Display

1. **What the frontend shows**: Open DEX screen, check QUG balance
2. **What you actually have**: 964.70811004 QUG (from blockchain)
3. **If frontend shows**: 96.47 QUG → **BUG! Dividing by 1e9 instead of 1e8**

## Diagnostic Steps

### Step 1: Check API Response
Open browser console and run:
```javascript
qnkAPI.getWalletBalance(localStorage.getItem('walletAddress'))
  .then(r => console.log('balance_qnk:', r.data.balance_qnk));
```

Expected: `964.70811004` (correct)
If you see: `96.47081100` → API is wrong
If you see: `964.70811004` → API is correct, frontend display is wrong

### Step 2: Check localStorage Cache
```javascript
console.log('Cached balance:', localStorage.getItem('cachedBalance'));
```

Expected: `964.70811004` or similar
If you see: `96.47...` → Cache has wrong value

### Step 3: Check DexScreen Balance Display
Look at the "QUG" balance shown in the DEX token selector.

Should show: `964.70 QUG`
If shows: `96.47 QUG` → Display bug!

## Possible Bugs

### Bug 1: API Response Wrong
If `balance_qnk` from API is `96.47` instead of `964.70`, then backend is dividing by 1e9:

**Location**: `handlers.rs:2834`
```rust
"balance_qnk": balance as f64 / 100_000_000.0,  // Should be 1e8
```

We already fixed this in v0.9.48-beta! Check if old binary is running.

### Bug 2: Frontend Dividing Again
If API returns correct `964.70` but frontend divides by 10 again:

**Location**: `DexScreen.tsx` or `TokenBar.tsx`
```tsx
nativeQugBalance = response.data.balance_qnk / 10;  // ❌ Don't do this!
```

### Bug 3: Cached Balance Wrong
If localStorage has old wrong value cached.

## The Swap Failure is Correct!

The swap IS failing correctly because you don't have enough balance:
- You have: 964.70811004 QUG
- You need: 964.80538004 QUG
- Short by: 0.09727 QUG

**This is not a bug** - you literally don't have enough!

## But the Display Might Be Wrong

If the DEX screen shows `96.47 QUG` balance, that's the bug we need to fix.

## Next Steps

1. Check what balance is displayed in the DEX screen
2. If it shows `96.47 QUG` → We have a display bug (dividing by 1e9)
3. If it shows `964.70 QUG` → No display bug, you just need more QUG for the swap

## To Fix: Reduce Swap Amount

If your balance is `964.70 QUG`, try swapping a smaller amount like:
- `960 QUG` - leaves buffer
- `950 QUG` - safe margin
- Never swap 100% of balance (leave some for fees)
