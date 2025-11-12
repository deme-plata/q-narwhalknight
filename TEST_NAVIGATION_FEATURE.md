# Testing Guide: Dashboard to Transaction Navigation

## Quick Test Checklist

### Test 1: QUG Coin Navigation
1. Open Dashboard
2. Locate the QUG (Quillon Graph) wallet card
3. Click anywhere on the QUG card OR click the "Send" button
4. **Expected**: Navigate to TransactionV2 screen
5. **Verify**: 
   - Wallet card at top shows "Quillon Graph" and QUG balance
   - Amount input label shows "Amount (QUG)"
   - Available balance shows QUG balance
   - Dropdown has QUG selected

### Test 2: QUGUSD Coin Navigation
1. Open Dashboard
2. Locate the QUGUSD (Quillon USD) wallet card
3. Click the "Send" button on QUGUSD card
4. **Expected**: Navigate to TransactionV2 screen
5. **Verify**:
   - Wallet card at top shows "Quillon USD" and QUGUSD balance
   - Amount input label shows "Amount (QUGUSD)"
   - Available balance shows QUGUSD balance
   - Dropdown has QUGUSD selected

### Test 3: USD Coin Navigation
1. Open Dashboard
2. Locate the USD (US Dollar) wallet card
3. Click the "Send" button on USD card
4. **Expected**: Navigate to TransactionV2 screen
5. **Verify**:
   - Wallet card at top shows "US Dollar" and USD balance
   - Amount input label shows "Amount (USD)"
   - Available balance shows USD balance
   - Dropdown has USD selected

### Test 4: Coin Switching via Dropdown
1. Navigate to TransactionV2 from any coin
2. Open the "Select Coin to Send" dropdown
3. Select a different coin
4. **Expected**: UI updates immediately
5. **Verify**:
   - Wallet card updates with new coin info
   - Amount input label updates
   - Available balance updates
   - Fee display updates

### Test 5: Validation with Insufficient Balance
1. Navigate to TransactionV2 with any coin
2. Enter recipient address
3. Enter amount greater than available balance
4. **Expected**: Error message appears
5. **Verify**: Error mentions correct coin symbol and shows correct available balance

### Test 6: Direct Navigation (No Pre-selection)
1. Navigate to TransactionV2 directly via menu (not from Dashboard)
2. **Expected**: Defaults to QUG
3. **Verify**:
   - QUG is selected in dropdown
   - QUG balance shown
   - Can still switch to other coins

## Developer Testing

### Verify localStorage Mechanism
```javascript
// In browser console, test pre-selection:
localStorage.setItem('selectedCoinForSend', 'QUGUSD');
// Then navigate to Transactions screen
// Should show QUGUSD selected
```

### Verify State Clearing
```javascript
// After opening TransactionV2:
localStorage.getItem('selectedCoinForSend'); // Should be null
// The key is removed after reading to prevent stale state
```

### Check Console Logs
Look for these log messages:
- `Coin send clicked: [SYMBOL]` when clicking card/button
- `💰 Selected wallet balance: [NUMBER]` in validation
- Fresh balance fetches for QUG, QUGUSD, USD

## Edge Cases to Test

1. **Zero Balance**: Click on coin with 0 balance → should still navigate and show 0
2. **Network Error**: Simulate API failure → should gracefully handle missing balance data
3. **Multiple Quick Clicks**: Click different coin cards rapidly → should only use last click
4. **Back Navigation**: After sending, go back to Dashboard → pre-selection should be cleared
5. **Refresh Page**: Refresh on TransactionV2 → should default to QUG (localStorage cleared)

## Success Criteria

✅ All coins (QUG, QUGUSD, USD) can be selected from Dashboard  
✅ Wallet card displays correct coin information  
✅ Balance validation uses correct coin balance  
✅ Dropdown allows switching between coins  
✅ Error messages reference correct coin  
✅ No console errors or warnings  
✅ Smooth UX with no lag or flicker  

## Known Limitations

- Only supports QUG, QUGUSD, and USD (other coins coming soon)
- Requires active wallet session to fetch balances
- USD requires payment API to be available
- QUGUSD requires multi-token balance endpoint

## Troubleshooting

**Issue**: "Coin not pre-selected after clicking"
- Check console for localStorage writes
- Verify App.tsx handleCoinSendClick is called
- Check TransactionV2 useEffect runs

**Issue**: "Balance shows 0 or incorrect"
- Check API endpoints are responding
- Verify wallet authentication
- Check cached balance in localStorage

**Issue**: "Dropdown doesn't appear"
- Verify multiple coins were fetched
- Check walletBalances array length
- Ensure API calls succeeded

---

**Test Status**: Ready for QA  
**Last Updated**: 2025-11-05
