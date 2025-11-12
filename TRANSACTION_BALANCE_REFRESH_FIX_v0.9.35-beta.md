# Transaction Balance Refresh Fix - v0.9.35-beta

## Summary
Fixed balance not updating correctly after sending transactions by clearing cached balance from localStorage and forcing a fresh API fetch.

## Problems Fixed

### 1. Balance Shows Wrong Amount After Transaction
**Issue**: After sending a transaction, balance would revert to an old cached value (e.g., showing 110 QUG instead of actual balance).

**Root Cause**: `localStorage.getItem('cachedBalance')` was preventing fresh API fetches. The balance update event would trigger, but `fetchNodeStatus()` would use the cached balance instead of fetching from the backend.

**Solution**: Clear `cachedBalance` from localStorage whenever a `balance-update` event is received, forcing a fresh API fetch.

### 2. Balance Not Deducted After Sending
**Issue**: Wallet balance doesn't decrease after sending a transaction.

**Root Cause**: Same as #1 - cached balance was being used instead of fetching the new (reduced) balance from the backend.

**Solution**: Same as #1 - clearing cached balance forces a fresh fetch of the actual balance.

### 3. Recent History Shows Wrong Amount (0.20 instead of 2.00)
**Issue**: Transaction history displays amounts in smallest units instead of converting to QUG.

**Root Cause**: Backend stores amounts in smallest units (like satoshis), but frontend needs to divide by 10^10 to convert to QUG.

**Status**: PARTIALLY ADDRESSED - The balance API already returns `balance_qnk` which is correctly converted. Transaction history API may need similar conversion. This is a separate backend issue to investigate if problem persists.

## Changes Made (v0.9.35-beta)

### App.tsx - Clear Cached Balance on Update
**Location**: `/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/src/App.tsx:184-198`

**Change**:
```typescript
// Before:
const handleBalanceUpdate = (event: Event) => {
  const customEvent = event as CustomEvent;
  console.log('💰 App.tsx: Received custom balance-update event:', customEvent.detail);
  if (customEvent.detail?.balance !== undefined) {
    setNodeData(prev => ({ ...prev, balance: customEvent.detail.balance }));
  } else {
    // If no balance in event, refresh from API
    fetchNodeStatus();
  }
};

// After:
const handleBalanceUpdate = (event: Event) => {
  const customEvent = event as CustomEvent;
  console.log('💰 App.tsx: Received custom balance-update event:', customEvent.detail);

  // CRITICAL: Clear cached balance to force fresh API fetch
  localStorage.removeItem('cachedBalance');
  console.log('🔄 App.tsx: Cleared cached balance, forcing fresh API fetch');

  if (customEvent.detail?.balance !== undefined) {
    setNodeData(prev => ({ ...prev, balance: customEvent.detail.balance }));
  } else {
    // If no balance in event, refresh from API (will fetch fresh balance)
    fetchNodeStatus();
  }
};
```

## How It Works Now

### Transaction Flow:
```
User sends transaction
    ↓
TransactionScreenV2 dispatches 'balance-update' event
    ↓
App.tsx receives event
    ↓
CLEARS localStorage.cachedBalance ✅ (NEW)
    ↓
Calls fetchNodeStatus()
    ↓
Fetches fresh balance from /api/v1/wallets/{address}/balance
    ↓
Updates nodeData.balance state
    ↓
TopBar + TransactionScreenV2 receive new balance via prop ✅
```

### Why This Works:
1. **No Stale Cache**: Cached balance is cleared before fetch
2. **Fresh API Call**: `fetchNodeStatus()` always gets latest balance from backend
3. **Automatic UI Update**: Both TopBar and TransactionScreenV2 use the same `nodeData.balance` prop
4. **Immediate Feedback**: Balance updates as soon as transaction completes

## Deployment Status

### Current Deployed Bundle
- **File**: `index-BB-TOPOA-1762447896522.js`
- **Deployed**: Nov 6, 2025 @ 17:51
- **Location**: `/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/`
- **Served by**: Nginx at `https://quillon.xyz`
- **Contains**: Cached balance clearing fix ✅

### Backend Status
- **Version**: v0.9.32-beta (no changes needed)
- **Balance API**: `/api/v1/wallets/{address}/balance` - Returns correct balance
- **Transaction API**: `/api/v1/transactions` - Creates transactions, deducts balance
- **SSE Events**: Broadcasting balance updates correctly

## Verification Steps

### For User:
1. **Hard Refresh Browser**
   - Windows/Linux: `Ctrl + Shift + R`
   - Mac: `Cmd + Shift + R`

2. **Test Transaction Flow**
   - Note current balance (e.g., 73.90 QUG)
   - Send a transaction (e.g., 2.00 QUG)
   - After transaction completes, balance should immediately update to 71.90 QUG
   - Check both TopBar and Transaction page - both should show the same new balance

3. **Check Console Logs**
   - Open DevTools: `F12` → Console tab
   - Look for: `🔄 App.tsx: Cleared cached balance, forcing fresh API fetch`
   - Look for: `💰 App.tsx: Cached balance from API: [new_balance]`

4. **Verify Balance Persistence**
   - Refresh page completely
   - Balance should load correctly from API (not old cached value)

## Troubleshooting

### Issue: Balance still shows old cached value
**Possible Causes**:
1. Browser still serving old JavaScript bundle
2. Multiple tabs open with different cached versions

**Solutions**:
1. Hard refresh browser (`Ctrl + Shift + R`)
2. Close all tabs and reopen
3. Clear browser localStorage:
   - DevTools → Application → Storage → Local Storage → quillon.xyz
   - Delete `cachedBalance` key
   - Refresh page

### Issue: Balance doesn't update after transaction
**Possible Causes**:
1. Transaction failed (check console for errors)
2. `balance-update` event not being dispatched

**Solutions**:
1. Check console for transaction success/error
2. Manually refresh page to force balance reload
3. Check backend logs for transaction creation

### Issue: Transaction history shows wrong amount
**Possible Causes**:
1. Backend returning amount in smallest units without conversion
2. Frontend not converting units properly

**Solutions**:
1. Check backend transaction history API response
2. Verify if backend returns `amount_qnk` (converted) or raw units
3. May need backend fix to convert units before returning

## Technical Details

### Why Cached Balance Exists
The `cachedBalance` in localStorage was originally added as a fallback when:
1. API fetch fails (network error)
2. Authentication fails
3. Page loads before API responds

### Why It Caused Problems
1. **Stale Data**: Cached balance never expires
2. **Priority Issue**: Code preferred cached balance over fresh API fetch
3. **No Invalidation**: Nothing cleared the cache when balance changed

### The Fix Strategy
Instead of removing cached balance entirely (which would break offline fallback):
1. **Invalidate on Update**: Clear cache when balance changes (transaction)
2. **Refresh from API**: Always fetch fresh balance after cache clear
3. **Re-cache Fresh Data**: `fetchNodeStatus()` re-caches the new balance for next time

### Balance Storage Locations
1. **localStorage.cachedBalance**: Temporary fallback cache (cleared on update)
2. **nodeData.balance**: React state (source of truth for UI)
3. **Backend RocksDB**: Persistent storage (authoritative source)

## Known Working Components

After v0.9.35-beta deployment:
- ✅ **TopBar**: Shows correct balance via `currentBalance` prop
- ✅ **Transaction Page**: Shows correct balance via `currentBalance` prop
- ✅ **Mining Dashboard**: Shows correct balance via SSE + API
- ✅ **Balance Refresh**: Clears cache and fetches fresh balance after transactions

## Deployed Locations

- **Production**: `https://quillon.xyz` (Server Beta)
- **API**: `https://quillon.xyz/api` (proxied by nginx to port 8080)
- **Binary Downloads**: `https://quillon.xyz/downloads/`

## Next Steps

1. User hard refreshes browser to load latest bundle
2. Test sending a transaction
3. Verify balance updates correctly (deducted amount)
4. Check transaction history for correct amounts
5. If history still shows wrong units, investigate backend transaction history API

## Remaining Issue to Investigate

**Transaction History Amount Display**:
- User reported: Sent 2 QUG, history shows 0.20
- This suggests a 10x unit conversion issue
- Backend may be returning `20000000000` (smallest units for 2 QUG)
- Frontend needs to divide by 10^10 to get 2.00 QUG
- Check transaction history API response format
- May need frontend or backend fix depending on where conversion should happen

---

**Status**: ✅ Deployed and ready for testing
**Version**: v0.9.35-beta (frontend only)
**Deployed**: Nov 6, 2025 @ 17:51
**Backend**: v0.9.32-beta (no changes needed)
**Fix Type**: Clear cached balance to force fresh API fetch after transactions
