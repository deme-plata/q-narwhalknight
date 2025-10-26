# PaaS API Key Generation - Testing Guide

**Date**: October 22, 2025
**Status**: ✅ Fixed with comprehensive error logging
**File**: `gui/quantum-wallet/src/components/SettingsScreen.tsx:480-555`

---

## What Was Fixed

### Problem:
"Error generating API key. Please try again." with no debugging information.

### Solution Applied:

✅ **Added wallet address validation** - Checks if wallet exists before API call
✅ **Comprehensive console logging** - All steps logged with `[PaaS]` prefix
✅ **Better error messages** - Specific error details shown to user
✅ **Success feedback** - Green checkmark message when key generated
✅ **Error diagnostics** - Stack traces and error types logged

---

## Testing Steps

### Step 1: Start the Frontend (if not already running)

```bash
cd /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet

# Check if already running
ss -tlnp | grep :5173

# If not running, start it
npm run dev
```

**Expected Output:**
```
VITE v4.x.x  ready in XXX ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### Step 2: Open Browser and Developer Tools

1. Open browser to: **http://localhost:5173**
2. Press **F12** to open Developer Tools
3. Go to **Console** tab
4. Clear the console (click 🚫 or press Ctrl+L)

### Step 3: Navigate to Settings

1. In the quantum wallet UI, go to **Settings** (⚙️ icon)
2. Click on **Privacy-as-a-Service** tab
3. Scroll to "PaaS API Configuration" section

### Step 4: Test API Key Generation

Click the **"Generate Key"** button

### Expected Scenarios:

#### Scenario A: No Wallet Created (Expected)

**Console Output:**
```
[PaaS] Generating API key... {hasWallet: false, wallet: null}
```

**Alert Message:**
```
Please create or select a wallet first before generating an API key.
```

**Action**: Go to Dashboard and create a wallet first

---

#### Scenario B: Wallet Exists, API Server Running (Success)

**Console Output:**
```
[PaaS] Generating API key... {hasWallet: true, wallet: "0x742d35..."}
[PaaS] Calling API endpoint...
[PaaS] API response status: 200 OK
[PaaS] API response data: {success: true, data: {key_id: "...", api_key: "paas_..."}}
[PaaS] API key generated successfully: key_new123
```

**UI Behavior:**
1. Password field changes to text type
2. Shows generated key: `paas_1234567890abcdef1234567890abcdef12345678_checksum`
3. Green success message appears: "✓ API key generated successfully! (Visible for 5 seconds)"
4. After 5 seconds:
   - Field changes back to password type
   - Success message disappears
   - Key is hidden as `••••••••••`

**Result**: ✅ **SUCCESS**

---

#### Scenario C: API Server Not Running (Error)

**Console Output:**
```
[PaaS] Generating API key... {hasWallet: true, wallet: "0x742d35..."}
[PaaS] Calling API endpoint...
[PaaS] Error generating PaaS API key: TypeError: Failed to fetch
[PaaS] Error details: {message: "Failed to fetch", type: "TypeError"}
```

**Alert Message:**
```
Error generating API key: Failed to fetch

Check browser console (F12) for details.
```

**Fix**: Start the API server:
```bash
cd /opt/orobit/shared/q-narwhalknight
timeout 36000 cargo run --package q-api-server --bin q-api-server
```

---

#### Scenario D: CORS Error (Unlikely but possible)

**Console Output:**
```
[PaaS] Generating API key... {hasWallet: true, wallet: "0x742d35..."}
[PaaS] Calling API endpoint...

Access to fetch at 'http://localhost:8080/api/v1/privacy/paas/api-keys/generate'
from origin 'http://localhost:5173' has been blocked by CORS policy
```

**Fix**: CORS is already permissive, but if you see this, restart API server

---

#### Scenario E: Backend Returns Error

**Console Output:**
```
[PaaS] Generating API key... {hasWallet: true, wallet: "invalid"}
[PaaS] Calling API endpoint...
[PaaS] API response status: 400 Bad Request
[PaaS] API error response: {"success":false,"error":"Invalid wallet address"}
```

**Alert Message:**
```
Error generating API key: API request failed: 400 Bad Request

Check browser console (F12) for details.
```

**Fix**: Check wallet address format

---

## Manual Testing Commands

### Test 1: Check Wallet in Browser Console

Open browser console on the quantum wallet page and run:
```javascript
localStorage.getItem('currentWallet')
```

**Expected**:
- If wallet exists: `"0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"` (or similar)
- If no wallet: `null`

**Action**: If null, create a wallet first

### Test 2: Direct API Call from Browser

Run this in the browser console to bypass the UI and test the endpoint directly:

```javascript
fetch('http://localhost:8080/api/v1/privacy/paas/api-keys/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    wallet_address: localStorage.getItem('currentWallet') || 'test_wallet_123',
    tier: 'free',
    expires_days: 90
  })
})
.then(r => r.json())
.then(d => {
  console.log('✅ SUCCESS:', d);
  if (d.success && d.data) {
    console.log('🔑 API Key:', d.data.api_key);
  }
})
.catch(e => {
  console.error('❌ ERROR:', e);
});
```

**Expected Success Output:**
```
✅ SUCCESS: {
  success: true,
  data: {
    key_id: "key_new123",
    api_key: "paas_1234567890abcdef1234567890abcdef12345678_checksum",
    tier: "free",
    expires_at: 1768906238
  },
  error: null,
  timestamp: "2025-10-22T10:50:38Z"
}
🔑 API Key: paas_1234567890abcdef1234567890abcdef12345678_checksum
```

### Test 3: Check Backend Logs

In a separate terminal, watch the API server logs:

```bash
# Find the API server process
ps aux | grep q-api-server

# Follow logs (if using systemd)
journalctl -u q-api-server -f

# Or check stdout if running in terminal
# (Just run the server in a terminal and watch output)
```

**Expected Log When Key Generated:**
```
POST /api/v1/privacy/paas/api-keys/generate
  Generating API key for wallet: 0x742d35...
  Key generated: key_new123
  Response: 200 OK
```

---

## Troubleshooting

### Issue: "Please create or select a wallet first"

**Diagnosis**: No wallet in localStorage

**Solution**:
1. Go to Dashboard
2. Click "Create Wallet" or "Import Wallet"
3. Complete wallet creation
4. Return to Settings > Privacy-as-a-Service
5. Try "Generate Key" again

---

### Issue: "Failed to fetch"

**Diagnosis**: API server not running or not reachable

**Check**:
```bash
ss -tlnp | grep :8080
```

**If empty**:
```bash
cd /opt/orobit/shared/q-narwhalknight
timeout 36000 cargo run --package q-api-server --bin q-api-server
```

**Wait for**:
```
Server listening on 0.0.0.0:8080
```

---

### Issue: "API returned unsuccessful response"

**Diagnosis**: Backend responded but with error

**Check Console** for the full response object logged by `[PaaS] API response data:`

**Common Causes**:
- Invalid wallet address format
- Database connection error
- Rate limit exceeded
- Internal server error

**Solution**: Check backend logs for specific error

---

### Issue: Key generated but not displayed

**Diagnosis**: DOM selector couldn't find the password input

**Console Shows**:
```
[PaaS] Could not find password input element
```

**Solution**: Check that the password input field exists on the page with placeholder containing "paas_"

---

## Success Criteria

✅ **All criteria must pass:**

1. [ ] Console shows `[PaaS] Generating API key...`
2. [ ] Console shows `hasWallet: true` and wallet address
3. [ ] Console shows `[PaaS] API response status: 200 OK`
4. [ ] Console shows `[PaaS] API key generated successfully`
5. [ ] UI displays the API key in text format
6. [ ] Green success message appears
7. [ ] After 5 seconds, key becomes hidden (password type)
8. [ ] Success message disappears

---

## Quick Diagnostic Script

Run this to verify the entire chain is working:

```bash
cd /opt/orobit/shared/q-narwhalknight

cat << 'EOF' > test_paas_key_generation.sh
#!/bin/bash

echo "=== PaaS API Key Generation - Quick Test ==="
echo

# 1. Check API server
echo "1. Checking API server..."
if ss -tlnp | grep -q :8080; then
  echo "   ✅ API server running on port 8080"
else
  echo "   ❌ API server NOT running"
  echo "   Start with: timeout 36000 cargo run --package q-api-server --bin q-api-server"
  exit 1
fi
echo

# 2. Check frontend
echo "2. Checking frontend..."
if ss -tlnp | grep -q :5173; then
  echo "   ✅ Frontend running on port 5173"
else
  echo "   ⚠️  Frontend not detected on port 5173"
  echo "   Start with: cd gui/quantum-wallet && npm run dev"
fi
echo

# 3. Test API endpoint
echo "3. Testing API endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:8080/api/v1/privacy/paas/api-keys/generate \
  -H "Content-Type: application/json" \
  -d '{"wallet_address":"test_wallet_debug","tier":"free","expires_days":90}')

if echo "$RESPONSE" | grep -q '"success":true'; then
  echo "   ✅ API endpoint working"
  echo "   Generated key ID: $(echo "$RESPONSE" | grep -o '"key_id":"[^"]*"' | cut -d'"' -f4)"
else
  echo "   ❌ API endpoint error"
  echo "   Response: $RESPONSE"
  exit 1
fi
echo

# 4. Instructions
cat << 'INSTRUCTIONS'
4. Next steps:
   ✅ Backend is working!

   Now test the frontend:
   1. Open http://localhost:5173 in browser
   2. Press F12 to open Developer Tools
   3. Go to Console tab
   4. Navigate to Settings > Privacy-as-a-Service
   5. Click "Generate Key" button
   6. Watch console for [PaaS] logs

   Expected console output:
   [PaaS] Generating API key...
   [PaaS] API response status: 200 OK
   [PaaS] API key generated successfully

   If you see "hasWallet: false", create a wallet first from the Dashboard.

=== Test Complete ===
INSTRUCTIONS
EOF

chmod +x test_paas_key_generation.sh
./test_paas_key_generation.sh
```

---

## Video Walkthrough (Expected Behavior)

**Step-by-step visual guide:**

1. **Browser shows quantum wallet** at http://localhost:5173
2. **Click Settings** gear icon in navigation
3. **Click "Privacy-as-a-Service"** tab
4. **See "PaaS API Configuration"** card with password field
5. **Open DevTools** (F12) and go to Console tab
6. **Click "Generate Key"** button
7. **Console logs appear** with green `[PaaS]` prefix
8. **Password field becomes text** showing the full API key
9. **Green success message** appears below the field
10. **Wait 5 seconds**
11. **Field returns to password** dots (`••••••••••`)
12. **Success message disappears**

**Total duration**: ~7-8 seconds

---

## Support

### If still seeing errors after following this guide:

1. **Copy all console output** (everything with `[PaaS]` prefix)
2. **Screenshot the error** alert message
3. **Run the diagnostic script** and copy output
4. **Check backend logs** if API server is running

**Provide**:
- Console logs
- Error screenshot
- Diagnostic script output
- Backend logs (if available)

This will allow for precise debugging of the specific issue.

---

**Status**: ✅ Ready for testing
**Estimated Time**: 2-3 minutes to verify working
**Success Rate**: Should be 100% if API server is running and wallet exists
