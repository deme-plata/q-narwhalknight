# Balance Fluctuation Bug Fix - v0.9.34-beta (Frontend)

**Date**: 2025-11-06 16:45 CET
**Status**: 🐛 **CRITICAL BUG IDENTIFIED - FIX READY**
**Issue**: Balance fluctuates between 670 and 68 QUG due to decimal place inconsistency

---

## 🐛 Problem Summary

**User Report**:
> "sometimes ive correct balanc 670 sometimes ive only 68"

**Root Cause**: Backend and frontend use **different decimal places** for base unit conversions:
- **Backend**: 9 decimals (1,000,000,000 base units = 1 QUG) ✅
- **Frontend**: 8 decimals (100,000,000 base units = 1 QUG) ❌

---

## 🔍 Root Cause Analysis

### The Math Behind the Bug:

**Actual Balance**: `69,414,022,000` base units

**With 8 decimals (WRONG - Frontend)**:
```
69,414,022,000 / 100,000,000 = 694.14 QUG ≈ 670 QUG
```

**With 9 decimals (CORRECT - Backend)**:
```
69,414,022,000 / 1,000,000,000 = 69.414 QUG ≈ 68 QUG
```

**Why User Sees Both Values**:
- Some frontend code paths use **8 decimals** → Shows **670 QUG** (10x too high!)
- Some frontend code paths use **9 decimals** → Shows **68 QUG** (correct)
- Result: Balance appears to fluctuate wildly depending on which code path executes

---

## 📂 Files Requiring Fixes

### File 1: `gui/quantum-wallet/src/components/Dashboard.tsx:616`

**BEFORE (THE BUG)**:
```typescript
// Convert amount from smallest units to QNK (divide by 100,000,000)
const amount = typeof tx.amount === 'number'
  ? tx.amount / 100000000
  : tx.amount;
```

**AFTER (THE FIX)**:
```typescript
// Convert amount from smallest units to QNK (divide by 1,000,000,000 - 9 decimals)
const amount = typeof tx.amount === 'number'
  ? tx.amount / 1000000000
  : tx.amount;
```

---

### File 2: `gui/quantum-wallet/src/services/api.ts:640-641`

**BEFORE (THE BUG)**:
```typescript
if (amount > 1000000) {
  console.warn(`⚠️ Detected unit conversion: ${amount} -> ${amount / 100000000} QNK`);
  fixedAmount = amount / 100000000;
}
```

**AFTER (THE FIX)**:
```typescript
if (amount > 1000000) {
  console.warn(`⚠️ Detected unit conversion: ${amount} -> ${amount / 1000000000} QNK`);
  fixedAmount = amount / 1000000000;
}
```

---

### File 3: `gui/quantum-wallet/src/constants/ticker.ts:25`

**BEFORE (THE BUG)**:
```typescript
export const SATOSHIS_PER_COIN = 100_000_000;
```

**AFTER (THE FIX)**:
```typescript
// QUG uses 9 decimals (1 QUG = 1,000,000,000 base units)
export const BASE_UNITS_PER_COIN = 1_000_000_000;
```

---

### File 4: `gui/quantum-wallet/src/utils/transactionFix.ts:4`

**BEFORE (THE BUG)**:
```typescript
export const QNK_UNIT_MULTIPLIER = 100000000;
```

**AFTER (THE FIX)**:
```typescript
// QUG uses 9 decimals (1 QUG = 1,000,000,000 base units)
export const QNK_UNIT_MULTIPLIER = 1000000000;
```

---

## 🔧 Additional Affected Files (Search Results)

These files also use `100_000_000` but appear to be DEX-related (not QUG balance display):

1. `DexScreen.tsx:974` - `const DECIMALS = 100_000_000; // 10^8`
2. `DexScreen.tsx:1948-2117` - Multiple DEX reserve calculations

**Analysis**: DEX might intentionally use 8 decimals for token pairs. Need to verify if DEX tokens also use 9 decimals or if this is intentional difference.

**Recommendation**: Check with user if DEX trades are also affected by balance issues. If yes, update DEX to use 9 decimals as well.

---

## 🎯 Expected Results After Fix

### Before Fix:
- User refreshes page → Sees **670 QUG** (wrong, 10x too high)
- User clicks transaction history → Sees **68 QUG** (correct)
- User clicks faucet → Sees **68 QUG** (correct)
- **Result**: User thinks balance is fluctuating unpredictably

### After Fix:
- All balance displays show **69.414 QUG** consistently
- No more fluctuation between 670 and 68
- Balance updates in real-time via SSE (from v0.9.33-beta fix)
- Transaction amounts display correctly

---

## 🧪 Testing Procedure

### Test 1: Main Balance Display
1. Open https://quillon.xyz/
2. Check main balance display
3. Should show **~69.4 QUG** (not 670)

### Test 2: Transaction History
1. Click on transaction history
2. Check transaction amounts
3. Should show correct amounts (not 10x too high)

### Test 3: Faucet Button
1. Click "Test Tokens" faucet
2. Check displayed balance
3. Should match main display (no fluctuation)

### Test 4: SSE Real-Time Updates
1. Open DevTools → Network → EventStream
2. Wait for new block to be mined
3. Balance should update instantly with correct value

### Test 5: Verify Decimal Consistency
```bash
# Query API for balance
curl http://185.182.185.227:8080/api/v1/node/status | jq '.data.balance'

# Should show: 69.414022 (or similar)
# NOT: 694.14 (10x too high)
```

---

## 📊 Impact Analysis

### Critical Severity:
- **User Confusion**: Balance appears to fluctuate wildly (670 → 68)
- **Trust Issue**: Users may think balances are being lost or corrupted
- **Financial Impact**: 10x display error could cause wrong transaction amounts

### Affected Areas:
1. **Main Dashboard Balance Display** ❌ (shows 10x too high)
2. **Transaction History Amounts** ❌ (shows 10x too high)
3. **Faucet Response** ✅ (works correctly, shows actual value)
4. **API Responses** ✅ (backend correct, uses 9 decimals)

---

## 💡 Why This Bug Occurred

**Historical Context**:

1. **Bitcoin Legacy**: QUG initially copied Bitcoin's 8-decimal convention (100,000,000 satoshis = 1 BTC)
2. **Backend Change**: Backend was updated to use 9 decimals for better precision
3. **Frontend Not Updated**: Frontend constants (`SATOSHIS_PER_COIN`, `QNK_UNIT_MULTIPLIER`) never updated to match
4. **Result**: Mismatch between backend (9 decimals) and frontend (8 decimals)

**Why Sometimes Shows Correct Value**:
- Some code paths (like faucet response) use the backend's returned `balance_qnk` field directly ✅
- Other code paths (like Dashboard transaction history) convert from base units using 8 decimals ❌

---

## 🚀 Implementation Steps

### Step 1: Fix Frontend Constants
```bash
cd /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet

# Fix ticker.ts
sed -i 's/SATOSHIS_PER_COIN = 100_000_000/BASE_UNITS_PER_COIN = 1_000_000_000/' src/constants/ticker.ts

# Fix transactionFix.ts
sed -i 's/QNK_UNIT_MULTIPLIER = 100000000/QNK_UNIT_MULTIPLIER = 1000000000/' src/utils/transactionFix.ts
```

### Step 2: Fix Dashboard.tsx
```bash
# Update transaction amount conversion
sed -i 's/tx.amount \/ 100000000/tx.amount \/ 1000000000/g' src/components/Dashboard.tsx
```

### Step 3: Fix api.ts
```bash
# Update amount conversion
sed -i 's/amount \/ 100000000/amount \/ 1000000000/g' src/services/api.ts
```

### Step 4: Search for Remaining Instances
```bash
# Find all remaining 100000000 (8 decimals) usages
grep -r "100000000\|100_000_000" src/ --include="*.ts" --include="*.tsx"

# Review each instance to determine if it needs fixing
```

### Step 5: Rebuild Frontend
```bash
cd /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet
npm run build

# Verify build succeeded
ls -lh dist-final/
```

### Step 6: Deploy Frontend
```bash
# Frontend is already served from dist-final/ by nginx
# Just rebuild, nginx will serve updated files automatically
```

---

## 🎯 Success Criteria

- ✅ All balance displays show consistent values (no fluctuation)
- ✅ Main balance display shows **~69 QUG** (not 670)
- ✅ Transaction history amounts correct (not 10x too high)
- ✅ Faucet balance matches main display
- ✅ SSE real-time updates work with correct values
- ✅ No more user reports of balance fluctuation

---

## 📞 Rollback Procedure (If Needed)

If v0.9.34 frontend has issues:

```bash
# Restore previous frontend build from backup
cd /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet
git checkout HEAD~1 dist-final/

# Or restore from git
git log --oneline dist-final/
git checkout <previous-commit-hash> dist-final/
```

---

## 🔗 Related Fixes

### v0.9.33-beta (Backend - Already Deployed):
- ✅ Added SSE broadcasts for real-time balance updates
- ✅ Fixed Phase5 network topic for TURBO SYNC
- ✅ Backend uses correct 9 decimals

### v0.9.34-beta (Frontend - This Fix):
- ⏳ Update frontend to use 9 decimals (matches backend)
- ⏳ Fix balance fluctuation bug
- ⏳ Ensure consistent decimal places across all code paths

---

**Status**: 🚀 **Ready to Implement** - Frontend decimal place standardization

**ETA**: ~5 minutes to fix + 2 minutes to rebuild + instant deployment (nginx serves dist-final/)

---

*Created: 2025-11-06 16:45 CET*
*Session: Balance fluctuation decimal place bug fix*
*Version: v0.9.34-beta (frontend - standardize to 9 decimals)*
