# Balance Display Issue: 909 → 90.96 Investigation

## Problem
Balance shows as `909` on initial load, then updates to `90.96` shortly after.

## Root Cause Hypothesis

### Backend Uses 1e8 Decimals
From `handlers.rs:2834`:
```rust
"balance_qnk": balance as f64 / 100_000_000.0,
```

The backend stores balances in **base units** and divides by **100,000,000 (1e8)** to convert to QNK.

So if you have `9,096,000,000` base units:
- **Correct**: `9096000000 / 100000000 = 90.96` ✅
- **Wrong** (÷1e7): `9096000000 / 10000000 = 909.6` ❌
- **Wrong** (÷1e10): `9096000000 / 10000000000 = 0.9096` ❌

### The Bug: Unit Multiplier Mismatch

**Frontend says 1e9:**
`gui/quantum-wallet/src/utils/transactionFix.ts:4`:
```typescript
export const QNK_UNIT_MULTIPLIER = 1000000000;  // 1e9 - WRONG!
```

**Backend uses 1e8:**
```rust
balance as f64 / 100_000_000.0  // 1e8 - CORRECT
```

### Where the Issue Occurs

**Scenario 1: Initial Cache Value**
1. Some code path caches `909` or `909.6` to localStorage
2. On page reload, `App.tsx:36-38` loads this:
   ```tsx
   const cachedBalance = localStorage.getItem('cachedBalance');
   const initialBalance = cachedBalance ? parseFloat(cachedBalance) : 0;
   ```
3. `909` is displayed in TopBar
4. Then API fetch returns correct `90.96` and updates the display

**Scenario 2: Incorrect Division**
Somewhere in the codebase, a balance value is being divided by the wrong unit multiplier:
- If stored value is `90960` (already in wrong units)
- And divided by 100: `90960 / 100 = 909.6` → displayed as `909` or `910`

## Evidence from Codebase

### Inconsistent Unit Handling

**TransactionDetailsModal.tsx:172** divides by **1e10**:
```tsx
{((transaction.amount || 0) / 1e10).toLocaleString(...)}
```

**ExplorerScreen.tsx:765** divides by **1e8** (correct):
```tsx
amount: foundTx.amount ? (foundTx.amount / 100000000) : 0,
```

**DexScreen.tsx:2088** multiplies by **1e8** (correct):
```tsx
amount_in: Math.floor(parseFloat(swapAmount) * 100_000_000), // 8 decimals (1e8)
```

## Diagnostic Steps

### Step 1: Check Cached Balance
```javascript
// Open browser console on the wallet page
console.log('Cached balance:', localStorage.getItem('cachedBalance'));
console.log('Type:', typeof localStorage.getItem('cachedBalance'));
console.log('Parsed:', parseFloat(localStorage.getItem('cachedBalance')));
```

### Step 2: Check API Response
```javascript
// Check what the API actually returns
fetch('/api/v1/wallet/balance?wallet_address=qnk...your address...', {
  headers: {
    'X-Wallet-Auth': '...'  // Your auth header
  }
})
.then(r => r.json())
.then(data => {
  console.log('API Response:', data);
  console.log('balance:', data.data.balance);
  console.log('balance_qnk:', data.data.balance_qnk);
});
```

### Step 3: Track Balance Updates
Add logging to `App.tsx:100`, `App.tsx:165`, `App.tsx:324`:
```tsx
console.log('💾 Caching balance:', walletBalance, 'Type:', typeof walletBalance);
localStorage.setItem('cachedBalance', walletBalance.toString());
```

## Likely Fix

### Fix 1: Update Unit Multiplier Constant
```typescript
// gui/quantum-wallet/src/utils/transactionFix.ts
export const QNK_UNIT_MULTIPLIER = 100000000;  // Change from 1e9 to 1e8
```

### Fix 2: Ensure Consistent toLocaleString Usage
```tsx
// TopBar.tsx:276 - Add decimal formatting
{currentBalance.toLocaleString('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 8
})} {TICKER_SYMBOL}
```

###Fix 3: Validate Cached Balance Format
```tsx
// App.tsx:36-38
const cachedBalance = localStorage.getItem('cachedBalance');
let initialBalance = cachedBalance ? parseFloat(cachedBalance) : 0;

// Validate range - if balance is unreasonably small/large, it's likely in wrong units
if (initialBalance > 0 && initialBalance < 1) {
  // Might be stored in base units, convert
  initialBalance = initialBalance * 100000000;
} else if (initialBalance > 1000000) {
  // Might be in base units, convert to QNK
  initialBalance = initialBalance / 100000000;
}
console.log('⚡ App.tsx: Initializing balance from cache:', initialBalance);
```

## Testing

1. Clear localStorage: `localStorage.clear()`
2. Refresh page
3. Check initial balance display
4. Check balance after API fetch
5. Mine a block and check balance update
6. Refresh page again and verify cached balance is correct

## Next Steps

1. Add console logging to track where `909` value is being set
2. Check if mining rewards or transaction processing is caching incorrect values
3. Fix the `QNK_UNIT_MULTIPLIER` constant
4. Add validation when reading cached balance
5. Update all division operations to use correct 1e8 multiplier
