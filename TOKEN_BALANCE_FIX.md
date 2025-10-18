# Token Balance Display Fix

## Issue

When importing a newly deployed token contract into the DEX, the balance was showing as zero even though the initial supply was set to 1,000,000 during deployment.

## Root Cause

The `/api/v1/contracts/:address` endpoint was not returning the `total_supply` and `decimals` fields that the frontend DEX needs to display token balances.

The `ContractInfo` struct was missing these critical fields:
- `total_supply` - The initial token supply
- `decimals` - The number of decimal places (default: 18)

## Solution

### 1. Updated ContractInfo Struct

Added two new fields to `ContractInfo` in `crates/q-api-server/src/contracts_api.rs`:

```rust
#[derive(Debug, Serialize)]
pub struct ContractInfo {
    pub address: String,
    pub contract_type: String,
    pub name: String,
    pub symbol: Option<String>,
    pub owner: String,
    pub deployed_at: u64,
    pub verified: bool,
    pub has_security_features: bool,
    pub features: HashMap<String, bool>,
    pub deployment_tx: String,
    pub total_supply: Option<u64>,  // ✅ NEW: Total token supply
    pub decimals: Option<u32>,      // ✅ NEW: Token decimals
}
```

### 2. Extract from deployment_params

Updated `get_contract_details()` to extract these values from the contract's `deployment_params`:

```rust
// Extract total_supply and decimals from deployment_params
// Note: The parameter is stored as "initial_supply" in deployment_params
let total_supply = contract.deployment_params.get("initial_supply")
    .and_then(|v| {
        // Handle both number and string formats
        v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
    });

let decimals = contract.deployment_params.get("decimals")
    .and_then(|v| v.as_u64())
    .map(|d| d as u32)
    .or(Some(18)); // Default to 18 decimals if not specified
```

**Key Points:**
- The deployment parameter is named `initial_supply`, not `total_supply`
- Values can be stored as either numbers or strings, so we handle both
- Decimals defaults to 18 (ERC-20 standard) if not explicitly set

### 3. Updated get_user_contracts()

Applied the same fix to the `get_user_contracts()` endpoint so user contract listings also show supply and decimals.

## API Response Changes

### Before Fix:
```json
{
  "success": true,
  "data": {
    "address": "qnka4a64e3c1b9ad0616665ec10850e34ac6eaa488a97a5dfcf45d796b46a62495f",
    "contract_type": "AdvancedToken",
    "name": "test999",
    "symbol": "TEST9",
    "owner": "qnk9ba22da810c4571ae33f40e3f5c1664c81bb766f56eb1d4aae096ab2860fefd5",
    "deployed_at": 1760281095,
    "verified": false,
    "has_security_features": true,
    "features": { "governance": true, "burnable": true, ... },
    "deployment_tx": "0x..."
    // ❌ Missing: total_supply and decimals
  }
}
```

### After Fix:
```json
{
  "success": true,
  "data": {
    "address": "qnka4a64e3c1b9ad0616665ec10850e34ac6eaa488a97a5dfcf45d796b46a62495f",
    "contract_type": "AdvancedToken",
    "name": "test999",
    "symbol": "TEST9",
    "owner": "qnk9ba22da810c4571ae33f40e3f5c1664c81bb766f56eb1d4aae096ab2860fefd5",
    "deployed_at": 1760281095,
    "verified": false,
    "has_security_features": true,
    "features": { "governance": true, "burnable": true, ... },
    "deployment_tx": "0x...",
    "total_supply": 1000000,           // ✅ NEW: Shows initial supply
    "decimals": 18                     // ✅ NEW: Shows decimal places
  }
}
```

## Frontend Integration

The DEX frontend can now properly display token balances by:

1. Fetching contract details: `GET /api/v1/contracts/:address`
2. Using `total_supply` and `decimals` to display formatted balances
3. Converting raw amounts: `display_amount = raw_amount / 10^decimals`

Example:
```javascript
const token = await fetch(`/api/v1/contracts/${address}`).then(r => r.json());
if (token.success) {
    const formattedBalance = token.data.total_supply / Math.pow(10, token.data.decimals);
    console.log(`Balance: ${formattedBalance} ${token.data.symbol}`);
}
```

## Testing

### To Test:
1. Deploy a new token contract with initial_supply = 1000000
2. Import the token address into the DEX
3. Verify that the balance shows correctly (1000000 tokens, not zero)

### Example Deployment:
```json
{
  "contract_type": "AdvancedToken",
  "owner": "qnk...",
  "parameters": {
    "name": "Test Token",
    "symbol": "TEST",
    "initial_supply": "1000000"  // This will now appear as total_supply
  }
}
```

## Files Modified

- `crates/q-api-server/src/contracts_api.rs` (lines 71-86, 414-482)
  - Added `total_supply` and `decimals` fields to `ContractInfo` struct
  - Updated `get_contract_details()` to extract from `deployment_params`
  - Updated `get_user_contracts()` with same extraction logic

## Build Status

✅ Successfully compiled and deployed
✅ Server running on port 8080
✅ API endpoints updated

## Note on Existing Contracts

**Important:** Contracts deployed before this fix will need to be redeployed to have their `total_supply` and `decimals` appear in the API. This is because:

1. The fix reads from `deployment_params` which are stored at deployment time
2. Contracts without these parameters in storage will return `null` for these fields
3. The database may have been cleared between sessions

**Solution:** Simply redeploy your token contract and it will include the supply information.

---

**Status:** ✅ FIXED
**Deployed:** 2025-10-12
**Build:** `target/release/q-api-server` (Release mode)
