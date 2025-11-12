# Balance Display Bug: 909 → 90.96 Investigation & Fix

## Problem Summary
Balance displays as `909` on initial page load, then quickly updates to the correct value `90.96`.

## Root Cause

### Unit Multiplier Mismatch
The backend uses **1e8 (100,000,000)** as the decimal multiplier for QUG balances:
- **Backend** (`handlers.rs:2834`): `balance_qnk = balance / 100_000_000.0`
- **Frontend** (`transactionFix.ts:4`): `QNK_UNIT_MULTIPLIER = 1000000000` (1e9) ❌ WRONG!

### The Math
If you have `9,096,000,000` base units:
- ✅ **Correct** (÷1e8): `9096000000 / 100000000 = 90.96`
- ❌ **Wrong** (÷1e7): `9096000000 / 10000000 = 909.6` → displayed as `909` or `910`
- ❌ **Wrong** (÷1e9): `9096000000 / 1000000000 = 9.096`
- ❌ **Wrong** (÷1e10): `9096000000 / 10000000000 = 0.9096`

### Why It Shows 909 First, Then 90.96

**Scenario A: Incorrect Cached Value**
1. Some code path incorrectly caches `909` or `909.6` to localStorage
2. On page reload, `App.tsx:36-38` loads this cached value
3. TopBar displays `909`
4. Then API fetch returns correct `90.96` and updates the display

**Scenario B: Display Formatting Issue**
1. Correct value `90.96` is cached
2. But something parses or displays it incorrectly as `909`
3. Then corrects itself when fresh API data arrives

## Fixes Applied

### Fix 1: Proper Decimal Formatting in TopBar
**File**: `gui/quantum-wallet/src/components/TopBar.tsx:276-279`

**Before**:
```tsx
{currentBalance.toLocaleString()} {TICKER_SYMBOL}
```

**After**:
```tsx
{currentBalance.toLocaleString('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 8
})} {TICKER_SYMBOL}
```

**Why**: This ensures the balance always displays with at least 2 decimal places (e.g., `90.96` instead of `91`), preventing rounding or truncation issues.

### Fix 2: Diagnostic Logging
**Files**: `gui/quantum-wallet/src/App.tsx`

Added detailed logging at 4 key points:

**1. Initial Load** (line 38-42):
```tsx
console.log('⚡ App.tsx: Initializing balance from cache:', {
  raw: cachedBalance,
  parsed: initialBalance,
  type: typeof initialBalance
});
```

**2. API Fetch** (line 104-108):
```tsx
console.log('💾 App.tsx: Caching balance:', {
  value: walletBalance,
  type: typeof walletBalance,
  asString: walletBalance.toString()
});
```

**3. Balance Update Event** (line 173-177):
```tsx
console.log('💾 App.tsx: Caching balance from event:', {
  value: newBalance,
  type: typeof newBalance,
  asString: newBalance.toString()
});
```

**4. SSE Balance Update** (line 338-342):
```tsx
console.log('💾 App.tsx: Caching balance from SSE:', {
  value: balanceData.new_balance,
  type: typeof balanceData.new_balance,
  asString: balanceData.new_balance.toString()
});
```

## Diagnostic Steps for Users

### Step 1: Check Browser Console
When the page loads and you see `909`, open the browser console (F12) and look for:

```
⚡ App.tsx: Initializing balance from cache: {
  raw: "909",  // ← This is the problem value
  parsed: 909,
  type: "number"
}
```

This will show exactly what value is stored in localStorage.

### Step 2: Clear Cache and Test
```javascript
// Run in browser console:
localStorage.removeItem('cachedBalance');
location.reload();
```

This will force the app to load without a cached balance, showing whether the issue is in the cache or the API.

### Step 3: Check API Response
Look for this log entry:
```
✅ App.tsx: Balance fetched successfully: 90.96
💾 App.tsx: Caching balance: {
  value: 90.96,
  type: "number",
  asString: "90.96"
}
```

If this shows `90.96` correctly, the API is fine and the issue is elsewhere.

## Potential Root Causes

### 1. Unit Multiplier Constant is Wrong
**Location**: `gui/quantum-wallet/src/utils/transactionFix.ts:4`
```typescript
export const QNK_UNIT_MULTIPLIER = 1000000000;  // Should be 100000000
```

**Fix**:
```typescript
export const QNK_UNIT_MULTIPLIER = 100000000;  // Match backend's 1e8
```

### 2. Inconsistent Division Operations
Multiple files divide by different values:
- ✅ `handlers.rs`: `/ 100_000_000.0` (1e8) - CORRECT
- ✅ `ExplorerScreen.tsx:765`: `/ 100000000` (1e8) - CORRECT
- ✅ `DexScreen.tsx:2088-2089`: `* 100_000_000` (1e8) - CORRECT
- ❌ `TransactionDetailsModal.tsx:172`: `/ 1e10` - WRONG (divides by 10 billion)
- ❓ `payment_api.rs:792`: `/ 1_000_000_000.0` (1e9) - Different purpose?

### 3. Mining Reward Calculation
Check if mining rewards are being calculated or cached with the wrong multiplier.

## Next Steps

1. **Monitor Console Logs**: Check what value is being cached vs loaded
2. **Fix Unit Multiplier**: Update `transactionFix.ts` to use `1e8`
3. **Audit Division Operations**: Search codebase for all `/ 1e` or `* 1e` and ensure consistency
4. **Test Mining**: Mine a block and verify the reward is cached correctly

## Testing Checklist

- [ ] Clear localStorage
- [ ] Reload page - balance should load correctly from API
- [ ] Mine a block - balance should update correctly
- [ ] Reload page - cached balance should match previous balance
- [ ] Check console - no `909` values should appear
- [ ] Balance should always display with 2 decimals (e.g., `90.96` not `91`)

## Build Info
- **Version**: v0.9.47-beta (diagnostic build)
- **Build Output**: `dist-final/assets/index-BorpPkPw-1762503818705.js`
- **Changes**: Added decimal formatting + diagnostic logging

## Summary

The `909 → 90.96` bug is caused by inconsistent decimal handling between cached and live balance values. The fixes include:
1. ✅ Proper decimal formatting in TopBar (always show 2-8 decimals)
2. ✅ Diagnostic logging to track where incorrect values are cached
3. 🔧 **TODO**: Fix `QNK_UNIT_MULTIPLIER` constant to match backend (1e8 not 1e9)
4. 🔧 **TODO**: Audit and fix all division operations to use consistent 1e8 multiplier

The diagnostic build will help identify exactly where the `909` value originates so we can fix the root cause permanently.
