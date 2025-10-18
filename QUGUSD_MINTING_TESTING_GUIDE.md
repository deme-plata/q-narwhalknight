# QUGUSD Minting - Testing Guide

## Prerequisites

✅ **API Server Running**
```bash
# Check if running
curl -s http://localhost:8090/api/v1/node/status | jq .success

# If not running, start it
timeout 36000 cargo run --bin q-api-server --release
```

✅ **Frontend Built**
```bash
# Check if dist-final has new files
ls -lh gui/quantum-wallet/dist-final/assets/
# Should show: index-BmtfLu2-.js (newly built)
```

✅ **QUG Balance Available**
```bash
# Check your wallet balance
curl http://localhost:8090/api/v1/wallet/<your-address>/tokens | jq

# If zero, use faucet or mine some QUG
curl -X POST http://localhost:8090/api/v1/faucet \
  -H "Content-Type: application/json" \
  -d '{"wallet_address": "qnk<your-address>"}'
```

## Testing QUGUSD Minting

### Option 1: GUI Testing (Recommended)

1. **Open the Wallet GUI**:
   ```
   http://localhost:5173
   ```

2. **Navigate to DEX Screen**:
   - Click on "DEX" in the navigation menu

3. **Find QUGUSD Token**:
   - Scroll to the QUGUSD token card in the token list

4. **Click "Mint USD" Button**:
   - Green button with 💵 emoji

5. **Enter Minting Parameters**:
   - **Collateral Amount**: Amount of QUG to lock (e.g., `1`)
   - **Collateral Ratio**: Percentage (e.g., `160%`)
     - Minimum: 150%
     - Safe: 200%
     - Maximum: 300%
   - **QUGUSD to Mint**: Auto-calculated based on collateral and ratio

6. **Review Details**:
   ```
   QUG Price: $42.50
   Collateral Value: $42.50 (1 QUG × $42.50)
   Actual Ratio: 160.0%
   QUGUSD to Mint: 26.56
   ```

7. **Click "Mint QUGUSD"**:
   - Transaction will be submitted to the blockchain
   - Should see success notification

8. **Verify Balances Updated**:
   - **QUG**: Should decrease by collateral amount
   - **QUGUSD**: Should increase by minted amount
   - Check "Tokens" section or refresh the page

### Option 2: API Testing (Advanced)

```bash
# Get your wallet address from localStorage (or use a known address)
WALLET_ADDRESS="qnk<your-32-byte-address>"

# Mint 26.56 QUGUSD with 1 QUG collateral
curl -X POST http://localhost:8090/api/v1/quillon-bank/stablecoin/mint \
  -H "Content-Type: application/json" \
  -d "{
    \"amount\": 2656000000,
    \"collateral_type\": \"QUG\",
    \"collateral_amount\": 1.0,
    \"wallet_address\": \"$WALLET_ADDRESS\",
    \"reason\": \"Manual testing via API\"
  }" | jq

# Expected response:
# {
#   "success": true,
#   "data": {
#     "transaction_id": "0x...",
#     "amount_minted": 2656000000,
#     "collateral_locked": 1.0,
#     "collateral_ratio": 160.0,
#     "finalized_in_seconds": 0.05
#   },
#   "error": null,
#   "timestamp": "2025-10-17T..."
# }
```

### Verification

#### 1. Check Balances via API
```bash
curl http://localhost:8090/api/v1/wallet/$WALLET_ADDRESS/tokens | jq

# Should show:
# {
#   "success": true,
#   "data": {
#     "address": "...",
#     "tokens": {
#       "QUG": {
#         "balance": "3.00000000",  # Decreased by 1
#         "balance_base_units": 300000000,
#         "usd_value": 127.5
#       },
#       "QUGUSD": {
#         "balance": "26.56000000",  # Increased by minted amount
#         "balance_base_units": 2656000000,
#         "usd_value": 26.56
#       }
#     },
#     "total_usd_value": 154.06
#   },
#   "error": null
# }
```

#### 2. Check API Server Logs
```bash
tail -f api-server.log | grep -A5 "Minting QUGUSD"

# Expected log entries:
# [INFO] 💰 Minting 26.56 QUGUSD with 1 QUG collateral
# [INFO] 👤 Minting for wallet: qnk1234abcd...
# [INFO] 💰 Updated QUGUSD balance for 1234abcd: 0.00 → 26.56 (minted: 26.56)
# [INFO] 🔒 Locked 1 QUG as collateral: 4.00 → 3.00
# [INFO] ✅ Minted 2656000000 QUGUSD in 0.05s
```

#### 3. Check Storage Persistence
```bash
# Check that balances are persisted to RocksDB
# (This ensures balances survive server restarts)

# Restart API server
killall q-api-server
timeout 36000 cargo run --bin q-api-server --release &

# Wait for startup (~5 seconds)
sleep 5

# Check balances again - should be the same
curl http://localhost:8090/api/v1/wallet/$WALLET_ADDRESS/tokens | jq
```

## Common Issues

### Issue 1: "HTTP error! status: 500"

**Cause**: Backend error, likely missing wallet address

**Solution**:
- Ensure frontend has been rebuilt with latest changes
- Clear browser cache and reload
- Check API server logs for detailed error

```bash
# Rebuild frontend
cd gui/quantum-wallet
npm run build

# Clear browser cache:
# - Open DevTools (F12)
# - Right-click Reload button → "Empty Cache and Hard Reload"
```

### Issue 2: "No wallet address found"

**Cause**: Wallet not logged in or localStorage cleared

**Solution**:
- Log in with your mnemonic phrase
- Check localStorage has `walletAddress` key:
  ```javascript
  console.log(localStorage.getItem('walletAddress'));
  // Should show: qnk<64-hex-chars>
  ```

### Issue 3: "Insufficient QUG balance"

**Cause**: Not enough QUG to lock as collateral

**Solution**:
- Request from faucet: `/api/v1/faucet`
- Mine QUG: Use mining feature in GUI
- Reduce collateral amount or increase ratio

### Issue 4: Balances not updating in GUI

**Cause**: Frontend caching or SSE not connected

**Solution**:
- Refresh the page (F5 or Ctrl+R)
- Check browser console for SSE errors
- Navigate away and back to trigger re-fetch

```javascript
// In browser console:
window.location.reload();
```

## Expected Behavior Summary

| Action | QUG Balance | QUGUSD Balance | Collateral Ratio |
|--------|-------------|----------------|------------------|
| Initial | 4.00 | 0.00 | N/A |
| Mint 26.56 with 1 QUG | 3.00 | 26.56 | 160% |
| Mint 13.28 with 0.5 QUG | 2.50 | 39.84 | 160% |

**Collateral Calculation**:
```
QUG Price = $42.50
Collateral Value = QUG Amount × $42.50

QUGUSD to Mint = Collateral Value / (Ratio / 100)
Example: $42.50 / 1.60 = 26.56 QUGUSD
```

## Next Steps After Successful Minting

1. **Create Liquidity Pools**:
   ```bash
   ./setup_qug_qugusd_pool.sh qnk<your-address>
   ```

2. **Test Swapping**:
   - Navigate to DEX screen
   - Use Swap feature to trade QUG ↔ QUGUSD

3. **Monitor Position Health**:
   - Check collateral ratio doesn't drop below 150%
   - Add more collateral if QUG price drops

4. **Test Redemption** (when implemented):
   - Burn QUGUSD to unlock QUG collateral

## Troubleshooting Commands

```bash
# Check API server is responding
curl http://localhost:8090/api/v1/health

# Check wallet exists
curl http://localhost:8090/api/v1/wallets

# Check recent transactions
curl http://localhost:8090/api/v1/transactions/recent?limit=10 | jq

# View full API server logs
tail -100 api-server.log

# Check frontend console logs
# Open browser DevTools (F12) → Console tab

# Rebuild everything
cargo build --release --package q-api-server
cd gui/quantum-wallet && npm run build
```

## Success Criteria

✅ Minting completes without errors
✅ QUG balance decreases by collateral amount
✅ QUGUSD balance increases by minted amount
✅ Balances persist after page refresh
✅ Transaction appears in Recent Activity
✅ API logs show successful minting

## Support

If issues persist:
1. Check `QUGUSD_MINTING_FIX_COMPLETE.md` for technical details
2. Review API server logs for specific errors
3. Verify wallet address format is correct (qnk + 64 hex chars)
4. Ensure frontend was rebuilt after code changes

---

**Happy Minting! 💵**

The Q-NarwhalKnight quantum blockchain now supports collateralized stablecoin minting with full balance tracking and storage persistence.
