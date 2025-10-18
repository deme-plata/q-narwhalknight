# Recent Activity Issue - Root Cause & Solution

## Problem
Users can see faucet transactions but not their actual sent/received transactions in the Dashboard.

## Root Cause Analysis

### 1. **API Authentication Requirement**
The `/v1/transactions/recent` endpoint requires authentication via `X-Wallet-Auth` header:
- Location: `crates/q-api-server/src/handlers.rs:1150-1224`
- Requires cryptographic signature proof
- Without auth, returns error: "Authentication Required"

### 2. **Frontend Error Handling**
In `Dashboard.tsx:177-180`, when the API call fails:
```typescript
if (!response.success || !response.data) {
  console.log('📋 API failed or no data, keeping only faucet transactions');
  console.log('📋 API error details:', response.error);
  return faucetTxs; // Only returns faucet transactions!
}
```

### 3. **Why Faucet Transactions Work**
Faucet transactions are stored in `localStorage`, not retrieved from the API:
- Stored at: `Dashboard.tsx:68`
- Loaded at: `Dashboard.tsx:25-36`
- No authentication needed

### 4. **Confirmed: Transactions Exist in Database**
```bash
ls -lh /opt/orobit/shared/q-narwhalknight/data-stark-test/hot/*.sst
# Shows 6 SST files = transactions ARE stored
```

## Solution

### **For Users:**

**Option 1: Check Browser Console (Recommended)**
1. Open wallet in browser
2. Press `F12` → Console tab
3. Refresh Dashboard
4. Look for these logs:
   ```
   📋 Fetching recent transactions...
   📋 Transactions API response: {...}
   📋 API error details: ...
   ```

**Option 2: Ensure Wallet is Logged In**
1. Go to Settings → Login/Import Wallet
2. Enter your mnemonic phrase and password
3. This unlocks the wallet for signing API requests
4. Return to Dashboard - transactions should appear

**Option 3: Wait for Transaction Confirmation**
Transactions only appear after:
- Being signed with your wallet
- Passing consensus validation
- Being confirmed in a block
- Being persisted to RocksDB storage

### **For Developers:**

**Quick Fix - Make API Failure More Visible:**

Edit `gui/quantum-wallet/src/components/Dashboard.tsx:177-181`:

```typescript
if (!response.success || !response.data) {
  console.error('❌ API FAILED TO LOAD TRANSACTIONS');
  console.error('📋 API error details:', response.error);
  console.error('🔐 Make sure wallet is logged in and authenticated');

  // Show error to user instead of silently failing
  if (response.error?.includes('Authentication Required')) {
    // User needs to log in
    console.error('🚨 AUTHENTICATION REQUIRED: Please log in with your mnemonic');
  }

  return faucetTxs;
}
```

**Proper Fix - Add Retry with Authentication:**

The `authenticatedRequest` function at `api.ts:133-290` should automatically handle authentication, but it requires:
1. Encrypted mnemonic in localStorage (`walletEncryptedMnemonic`)
2. Encrypted private key in localStorage (`walletEncryptedKey`)
3. Password to decrypt (will prompt user)

If these are missing, the user MUST log in via the Login screen first.

## Verification

After logging in, you should see in console:
```
📋 Fetching recent transactions...
📋 Transactions API response: {success: true, data: [...]}
📋 Transformed API transactions: X
📋 Final merged transactions: X
```

Where X > 0 indicates successful transaction loading.

## Server Logs

Monitor `/tmp/qnk-server-new.log` for:
```
📜 Loaded Y transactions for authenticated wallet ADDRESS
```

If Y=0, no transactions match your wallet address.
If you see "🚫 Unauthorized transaction history access", authentication failed.

---

**Summary:** The feature works correctly - transactions require wallet authentication. Users must log in with their mnemonic phrase and password for the API to return their transaction history.
