# Frontend Balance Display Fix - v0.9.30-beta

**Date**: 2025-11-06 14:45 CET
**Status**: 🔍 **ROOT CAUSE IDENTIFIED**
**Issue**: Balance shows zero in main display but shows 336 QUG correctly when clicking faucet button

---

## 🐛 Problem Summary

**Symptoms**:
- Main balance display shows `0 QUG`
- Clicking "Test Tokens" faucet button shows correct balance (`336 QUG`)
- Backend logs confirm balance exists and is growing

**User Report**:
> "balance is still zero but pressing on test tokens faucet showed me balance 336"

---

## 🔍 Root Cause Analysis

### Backend is Working Correctly ✅

**Evidence from previous investigation** (`DEV_FEE_WORKING_CONFIRMATION_v0.9.30.md`):
1. ✅ Dev fee configuration correct (1% to master account)
2. ✅ Coinbase transactions created properly
3. ✅ In-memory balances updated correctly
4. ✅ Periodic persistence to RocksDB working (every 15-30 seconds)
5. ✅ Master account exists in RocksDB
6. ✅ P2P blocks processed correctly
7. ✅ Balance confirmed: **336 QUG** (represents ~672 blocks of dev fees)

### Frontend Has Two Different Balance Fetch Mechanisms ❌

#### Mechanism 1: Main Balance Display (AUTHENTICATED)

**File**: `gui/quantum-wallet/src/components/Dashboard.tsx:340-346`

```typescript
// Fetch fresh QUG balance from API (includes mining rewards)
let qugBalance = 0;
try {
  const balanceResponse = await qnkAPI.getWalletBalance(currentWalletAddress);
  if (balanceResponse.success && balanceResponse.data) {
    qugBalance = balanceResponse.data.balance_qnk || 0;
    console.log('💰 Fresh QUG balance fetched:', qugBalance);
```

**API Method**: `gui/quantum-wallet/src/services/api.ts:508-513`

```typescript
// Get wallet balance by address (AUTHENTICATED - requires signature)
async getWalletBalance(walletAddress?: string): Promise<ApiResponse<any>> {
  // Use stored wallet address if none provided
  const address = walletAddress || localStorage.getItem('walletAddress') || '';
  console.log('🔍 Fetching balance for wallet address:', address);
  return this.authenticatedRequest<any>(`/v1/wallets/${address}/balance`);
}
```

**Backend Endpoint**: `crates/q-api-server/src/main.rs:5376-5378`

```rust
.route(
    "/api/v1/wallets/:address/balance",
    get(handlers::get_wallet_balance),
) // Get wallet balance by address (requires authentication)
```

**Problem**:
- This endpoint **requires Ed25519 signature authentication**
- User doesn't have the master account's private key
- Authentication fails silently, returns 0 or null

#### Mechanism 2: Faucet Button (PUBLIC, WORKS CORRECTLY ✅)

**File**: `gui/quantum-wallet/src/components/Dashboard.tsx:1486-1492`

```typescript
const result = await qnkAPI.requestFaucet(currentWalletAddress);

if (result.success) {
  const receivedAmount = result.data?.amount_qnk || result.data?.new_balance_qnk || 10;

  if (result.data?.new_balance_qnk) {
    setNodeStatus(prev => prev ? {...prev, balance: result.data.new_balance_qnk} : prev);
  }
```

**API Method**: `gui/quantum-wallet/src/services/api.ts:498-505`

```typescript
// Request test tokens from faucet
async requestFaucet(walletAddress?: string): Promise<ApiResponse<any>> {
  const body = walletAddress ? { wallet_address: walletAddress } : {};
  console.log('🚰 Faucet request body:', body);
  return this.request<any>('/api/v1/faucet', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}
```

**Backend Endpoint**: `crates/q-api-server/src/main.rs:5380`

```rust
.route("/api/v1/faucet", post(handlers::faucet)) // Test token faucet
```

**Why It Works**:
- Faucet endpoint is **PUBLIC** (no authentication required)
- Returns `new_balance_qnk` field in response
- Frontend updates display with this value

---

## 🎯 Solution Options

### Option 1: Make `/v1/wallets/:address/balance` PUBLIC ✅ (RECOMMENDED)

**Why**:
- Balance information is already public on blockchain
- No security risk - anyone can query any address's balance
- Simplest fix - just remove authentication requirement

**Changes Required**:

**File**: `crates/q-api-server/src/main.rs:5376-5378`

```rust
// BEFORE:
.route(
    "/api/v1/wallets/:address/balance",
    get(handlers::get_wallet_balance),
) // Get wallet balance by address (requires authentication)

// AFTER:
.route(
    "/api/v1/wallets/:address/balance",
    get(handlers::get_wallet_balance_public),
) // Get wallet balance by address (PUBLIC - no auth required)
```

**Handler Function** (add new function or modify existing):

```rust
/// Get wallet balance by address (PUBLIC endpoint - no authentication required)
///
/// Balance information is already public on the blockchain, so no authentication needed.
/// This allows users to query any wallet's balance, including the master account.
pub async fn get_wallet_balance_public(
    Path(address): Path<String>,
    State(app_state): State<Arc<AppState>>,
) -> Json<ApiResponse<BalanceResponse>> {
    // Strip "qnk" prefix if present
    let hex_address = if address.starts_with("qnk") {
        &address[3..]
    } else {
        &address
    };

    // Decode hex address
    let address_bytes = match hex::decode(hex_address) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => {
            return Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid wallet address format".to_string()),
                timestamp: chrono::Utc::now().to_rfc3339(),
            });
        }
    };

    // Query balance from in-memory map (fast, O(1) lookup)
    let balances = app_state.wallet_balances.read().await;
    let balance_base_units = balances.get(&address_bytes).copied().unwrap_or(0);
    drop(balances);

    // Convert from base units (9 decimals) to QUG
    let balance_qnk = balance_base_units as f64 / 1_000_000_000.0;

    Json(ApiResponse {
        success: true,
        data: Some(BalanceResponse {
            balance_qnk,
            balance_base_units,
            address: format!("qnk{}", hex_address),
        }),
        error: None,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}
```

### Option 2: Use `/v1/node/status` Endpoint for Master Account Balance

**Why**:
- `/v1/node/status` already returns balance field
- PUBLIC endpoint (no authentication)
- Already being called by Dashboard

**Changes Required**:

**File**: `gui/quantum-wallet/src/components/Dashboard.tsx:340-370`

Add special case for master account:

```typescript
const MASTER_ACCOUNT = 'qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723';

const fetchWalletBalances = async () => {
  console.log('💰 Fetching wallet balances...');
  const currentWalletAddress = localStorage.getItem('walletAddress');

  if (!currentWalletAddress) {
    console.warn('⚠️ No wallet address found');
    return;
  }

  // Special case: If viewing master account, get balance from node status
  if (currentWalletAddress === MASTER_ACCOUNT) {
    const nodeStatusResponse = await qnkAPI.getNodeStatus();
    if (nodeStatusResponse.success && nodeStatusResponse.data) {
      const qugBalance = nodeStatusResponse.data.balance || 0;
      console.log('💰 Master account balance from node status:', qugBalance);
      setWalletBalances(prev => ({
        ...prev,
        QUG: {
          ...prev.QUG,
          balance: qugBalance,
          history: [...(prev.QUG.history || []), {
            timestamp: Date.now(),
            balance: qugBalance
          }].slice(-50) // Keep last 50 data points
        }
      }));
      return;
    }
  }

  // Normal authenticated flow for user wallets
  let qugBalance = 0;
  try {
    const balanceResponse = await qnkAPI.getWalletBalance(currentWalletAddress);
    // ... rest of existing code
  }
}
```

**Pros**:
- No backend changes required
- Works immediately
- Only frontend change

**Cons**:
- Hacky special-case logic
- Only fixes master account display
- Doesn't solve the general authentication problem

### Option 3: Make Frontend Use Faucet Endpoint for Balance Query

**Why**:
- Faucet endpoint already works and returns balance
- No backend changes needed

**Changes Required**:

**File**: `gui/quantum-wallet/src/services/api.ts` - Add new method:

```typescript
// Get wallet balance via faucet endpoint (PUBLIC, no auth required)
// This is a workaround using the faucet endpoint which returns balance in response
async getWalletBalancePublic(walletAddress?: string): Promise<ApiResponse<any>> {
  const address = walletAddress || localStorage.getItem('walletAddress') || '';
  console.log('🔍 Fetching balance (via faucet endpoint):', address);

  // Faucet endpoint returns { new_balance_qnk: number, ... }
  const result = await this.requestFaucet(address);

  if (result.success && result.data?.new_balance_qnk !== undefined) {
    return {
      success: true,
      data: {
        balance_qnk: result.data.new_balance_qnk,
        balance_base_units: Math.floor(result.data.new_balance_qnk * 1_000_000_000),
        address: address
      },
      error: null,
      timestamp: result.timestamp
    };
  }

  return result;
}
```

**Pros**:
- No backend changes
- Works for all wallets (not just master account)

**Cons**:
- Abusing faucet endpoint for balance queries
- Faucet might have rate limiting or cooldown
- Not semantically correct

---

## 📊 Recommendation

**Implement Option 1** - Make `/v1/wallets/:address/balance` endpoint PUBLIC

**Rationale**:
1. **Security**: Balance information is already public on blockchain
2. **Simplicity**: Clean architectural solution
3. **Future-proof**: Other features may need public balance queries
4. **Semantics**: Balance endpoint should be public by nature

**Implementation Steps**:
1. Modify `crates/q-api-server/src/main.rs:5376-5378` to remove authentication
2. Add or modify `handlers::get_wallet_balance` to not require `X-Wallet-Auth` header
3. Test with master account address
4. Deploy to production

---

## 🧪 Testing Procedure

### Test 1: Query Master Account Balance (PUBLIC)

```bash
# Should return 336+ QUG without authentication
curl http://quillon.xyz/api/v1/wallets/qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723/balance

# Expected response:
{
  "success": true,
  "data": {
    "balance_qnk": 336.0,
    "balance_base_units": 336000000000,
    "address": "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723"
  },
  "error": null,
  "timestamp": "2025-11-06T14:45:00Z"
}
```

### Test 2: Frontend Balance Display

1. Open https://quillon.xyz/
2. Navigate to Dashboard
3. Balance should show **336+ QUG** without clicking faucet button
4. Balance should update in real-time as blocks are mined

---

## 📈 Expected Outcome

**After Fix**:
- Main balance display shows correct balance (336+ QUG)
- No need to click faucet button
- Balance updates automatically via SSE
- All wallets can query their balance without authentication
- Master account balance visible to everyone (as intended)

---

**Status**: 🚀 **Ready to Implement** - Backend change required to make balance endpoint public

**ETA**: ~10 minutes to implement + test

---

*Created: 2025-11-06 14:45 CET*
*Session: Frontend balance display fix*
*Version: v0.9.30-beta (backend works correctly, frontend needs endpoint change)*
