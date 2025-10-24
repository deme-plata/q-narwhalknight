# PaaS API Key Generation - Debugging Guide

**Date**: October 22, 2025
**Issue**: "Error generating API key. Please try again." in Settings Screen

---

## System Status

### ✅ Backend Working Correctly

**API Server Status:**
```bash
$ ss -tlnp | grep :8080
LISTEN 0 1024 0.0.0.0:8080 0.0.0.0:* users:(("q-api-server",pid=583570,fd=48))
```

**Endpoint Test:**
```bash
$ curl -X POST http://localhost:8080/api/v1/privacy/paas/api-keys/generate \
  -H "Content-Type: application/json" \
  -d '{
    "wallet_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
    "tier": "free",
    "expires_days": 90
  }'

Response:
{
  "success": true,
  "data": {
    "key_id": "key_new123",
    "api_key": "paas_1234567890abcdef1234567890abcdef12345678_checksum",
    "tier": "free",
    "expires_at": 1768906238
  },
  "error": null,
  "timestamp": "2025-10-22T10:50:38.862591719Z"
}
```

✅ **Backend is working perfectly!**

---

## Root Cause Analysis

### Frontend Code Location
**File**: `/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/src/components/SettingsScreen.tsx`
**Lines**: 476-524 (Generate Key button click handler)

### Error Sources

The error message "Error generating API key. Please try again." comes from line 519, which is triggered when:

1. **CORS Policy Violation** (most likely)
   - Frontend served from different origin than `localhost:8080`
   - Browser blocks cross-origin request
   - Check browser console for CORS errors

2. **Network Connectivity**
   - Frontend can't reach `http://localhost:8080`
   - Firewall blocking the request
   - API server not running (already verified it IS running)

3. **Missing Wallet Address**
   - `localStorage.getItem('currentWallet')` returns null/undefined
   - Backend receives invalid wallet_address
   - Check: Is user logged in? Does wallet exist in localStorage?

4. **Response Parsing Error**
   - Response not valid JSON
   - `data.success` is false
   - `data.data` is null/undefined

---

## Debugging Steps

### Step 1: Check Browser Console

Open the quantum wallet frontend in browser and check Developer Console:

**Expected Errors:**
```
Access to fetch at 'http://localhost:8080/api/v1/privacy/paas/api-keys/generate'
from origin 'http://localhost:3000' has been blocked by CORS policy
```

or

```
TypeError: Failed to fetch
```

### Step 2: Verify Wallet Address

Open browser console and run:
```javascript
localStorage.getItem('currentWallet')
```

**Expected**: Wallet address string (e.g., "0x742d35...")
**Problem**: null or undefined

**Fix**: Ensure user is logged in and wallet is created

### Step 3: Check Frontend Port

Determine where the React app is running:
```bash
# Check common ports
ss -tlnp | grep -E ':(3000|3001|5173|8000)'
```

**Common scenarios:**
- Vite dev server: http://localhost:5173
- Create React App: http://localhost:3000
- Production build: Tauri app (different origin)

### Step 4: Test Direct API Call from Browser

Open browser console on the quantum wallet page and run:
```javascript
fetch('http://localhost:8080/api/v1/privacy/paas/api-keys/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    wallet_address: 'test_wallet_123',
    tier: 'free',
    expires_days: 90
  })
})
.then(r => r.json())
.then(d => console.log('SUCCESS:', d))
.catch(e => console.error('ERROR:', e));
```

**Expected**: Success response with API key
**Problem**: CORS error or network error

---

## Solutions

### Solution 1: Fix CORS (If Origin Mismatch)

**Current CORS**: Already permissive (`CorsLayer::permissive()` in main.rs:1767)

This should work, but if it doesn't:

**File**: `/opt/orobit/shared/q-narwhalknight/crates/q-api-server/src/main.rs`

Replace line 1767:
```rust
.layer(CorsLayer::permissive())
```

With explicit CORS configuration:
```rust
.layer(
    CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers(tower_http::cors::Any)
        .allow_credentials(false)
)
```

Then restart the server:
```bash
killall q-api-server
timeout 36000 cargo run --package q-api-server --bin q-api-server
```

### Solution 2: Use API Proxy (Recommended for Development)

If the frontend is a Tauri app or different domain, configure a proxy.

**For Vite (if using vite.config.ts):**
```typescript
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true
      }
    }
  }
})
```

Then change frontend code to use relative URL:
```typescript
// Change from:
const response = await fetch('http://localhost:8080/api/v1/privacy/paas/api-keys/generate', ...

// To:
const response = await fetch('/api/v1/privacy/paas/api-keys/generate', ...
```

### Solution 3: Ensure Wallet Address Exists

**File**: `/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/src/components/SettingsScreen.tsx`

Update line 483 to add better error handling:
```typescript
const walletAddress = localStorage.getItem('currentWallet');

if (!walletAddress || walletAddress === 'default_wallet') {
  alert('Please create or select a wallet first');
  return;
}
```

### Solution 4: Add Better Error Logging

Update lines 517-520 to show more details:
```typescript
} catch (error) {
  console.error('Error generating PaaS API key:', error);
  console.error('Error details:', {
    message: error.message,
    stack: error.stack,
    walletAddress: localStorage.getItem('currentWallet')
  });
  alert(`Error generating API key: ${error.message}\nPlease check console for details.`);
}
```

### Solution 5: Check Response Structure

Add logging before line 504:
```typescript
const data = await response.json();
console.log('API Response:', data); // Add this line

if (data.success && data.data) {
  // ...existing code
```

---

## Quick Fix Script

Run this to apply the most common fixes:

```bash
cd /opt/orobit/shared/q-narwhalknight

# 1. Verify API server is running
if ! ss -tlnp | grep -q :8080; then
  echo "Starting API server..."
  timeout 36000 cargo run --package q-api-server --bin q-api-server &
  sleep 5
fi

# 2. Test endpoint
echo "Testing API endpoint..."
curl -X POST http://localhost:8080/api/v1/privacy/paas/api-keys/generate \
  -H "Content-Type: application/json" \
  -d '{"wallet_address":"test","tier":"free","expires_days":90}' | jq

# 3. Check frontend process
echo "Frontend processes:"
ss -tlnp | grep -E ':(3000|3001|5173|8000)'

# 4. Instructions
cat << 'EOF'

Next Steps:
1. Open quantum wallet in browser
2. Open browser DevTools (F12)
3. Go to Settings > Privacy-as-a-Service tab
4. Click "Generate Key"
5. Check Console tab for errors
6. Share the error message for further debugging
EOF
```

---

## Most Likely Issue & Fix

**DIAGNOSIS**: Frontend is running as Tauri app (not localhost:3000), creating CORS/network isolation.

**IMMEDIATE FIX**:

### Option A: Test in Web Browser Instead
```bash
cd gui/quantum-wallet
npm run dev  # Start Vite dev server
# Open http://localhost:5173 in browser
# Try Generate Key button
```

### Option B: Fix Tauri API Calls

If using Tauri, HTTP requests to localhost may be blocked. Use Tauri's HTTP client instead:

**File**: `/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/src/components/SettingsScreen.tsx`

```typescript
// Add import
import { fetch as tauriFetch } from '@tauri-apps/api/http';

// Replace fetch call (line 486)
const response = await tauriFetch('http://localhost:8080/api/v1/privacy/paas/api-keys/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: {
    type: 'Json',
    payload: {
      wallet_address: walletAddress,
      tier: 'free',
      expires_days: 90
    }
  }
});
```

---

## Testing Checklist

- [ ] API server running on port 8080
- [ ] CORS configured as permissive
- [ ] Wallet address exists in localStorage
- [ ] Browser console checked for errors
- [ ] Direct curl to endpoint works
- [ ] Frontend can reach localhost:8080
- [ ] Response structure matches expected format
- [ ] No network/firewall blocking

---

## Expected Working Flow

1. User opens Settings > Privacy-as-a-Service tab
2. User clicks "Generate Key" button
3. Frontend reads `currentWallet` from localStorage
4. POST request to http://localhost:8080/api/v1/privacy/paas/api-keys/generate
5. Backend generates cryptographically secure API key
6. Response: `{"success":true,"data":{"api_key":"paas_..."}}`
7. Frontend displays key in password field for 5 seconds
8. Key auto-hides as password type

---

## Need More Help?

Run this comprehensive diagnostic:

```bash
cd /opt/orobit/shared/q-narwhalknight

cat << 'EOF' > debug_paas_api.sh
#!/bin/bash

echo "=== PaaS API Key Generation Diagnostics ==="
echo

echo "1. API Server Status:"
ss -tlnp | grep :8080 || echo "  ❌ Not running on port 8080"
echo

echo "2. API Endpoint Test:"
curl -s -X POST http://localhost:8080/api/v1/privacy/paas/api-keys/generate \
  -H "Content-Type: application/json" \
  -d '{"wallet_address":"debug_test","tier":"free","expires_days":90}' | jq
echo

echo "3. Frontend Processes:"
ss -tlnp | grep -E ':(3000|3001|5173|8000)' || echo "  ℹ No dev server detected"
echo

echo "4. CORS Configuration:"
grep -n "CorsLayer" crates/q-api-server/src/main.rs
echo

echo "5. Route Registration:"
grep -n "paas/api-keys/generate" crates/q-api-server/src/paas_admin_api.rs
echo

echo "=== Diagnosis Complete ==="
EOF

chmod +x debug_paas_api.sh
./debug_paas_api.sh
```

Share the output for specific troubleshooting!

---

**Status**: Ready for debugging
**Next Action**: Check browser console for specific error message
