# User Node Diagnostic Instructions - Height Fix Verification

**Date:** 2025-11-14 13:06 UTC
**Status:** Need user node logs to diagnose

---

## Issue Report Summary

**User Reports:**
- Downloaded `q-api-server-v1.0.2-beta-height-fix`
- Node shows: ✅ [SYNCED] Height: 80870
- But local height stuck at: Height 1
- Warning still appears: ⚠️ [v1.0.1-beta] Block created but height NOT advanced

**Bootstrap Node Status (185.182.185.227):**
- ✅ Height advancing correctly (80826 → 80833 → 80834)
- ✅ New log messages appearing: "✅ [v1.0.8-beta FIX] Pool: Producer #X..."
- ✅ Fix is deployed and working

---

## Verification Steps Needed

### Step 1: Verify Binary Has the Fix

Run on your user node:

```bash
# Check if advance_producer_height method exists
nm ./q-api-server-v1.0.2-beta-height-fix | grep -i "advance.*producer.*height"

# Expected output: Should show 3 lines with advance_producer_height symbols
# If NO output: Binary doesn't have the fix, redownload
```

### Step 2: Check Node Logs for Fix Messages

Run on your user node while it's running:

```bash
# Check for new log messages (should appear every few seconds)
tail -100 node.log | grep -E "(v1.0.8-beta FIX|advance_producer_height)"

# Expected output:
# ✅ [v1.0.8-beta FIX] Pool: Producer #X height advance command sent AFTER storage confirmation

# If NO output: The fix code path is not being reached
```

### Step 3: Check if Height is Actually Advancing

Run on your user node:

```bash
# Get current height
curl -s http://localhost:8080/api/v1/status | jq '.data.current_height'

# Wait 30 seconds
sleep 30

# Check again
curl -s http://localhost:8080/api/v1/status | jq '.data.current_height'

# Expected: Height should have increased by 1-3 blocks
# If SAME: Height is not advancing (bug still present)
```

### Step 4: Check Producer Task Logs

```bash
# Look for producer task messages
tail -200 node.log | grep -E "(Producer.*Created block|Height advanced to)"

# Expected output:
# ✅ Producer #X: Created block at height Y
# ✅ [v1.0.1-beta FIX] Height advanced to Y AFTER storage confirmation
```

---

## Possible Root Causes

### Cause #1: Binary Doesn't Have Fix (Most Likely)

**Symptoms:**
- No "v1.0.8-beta FIX" messages in logs
- nm shows no advance_producer_height symbols

**Solution:**
- Redownload from: `http://quillon.xyz/downloads/q-api-server-v1.0.8-beta-height-fix-CORRECTED`
- Or download from: `http://quillon.xyz/downloads/q-api-server-linux-x86_64` (latest)

### Cause #2: Running Old Binary

**Symptoms:**
- Downloaded correct binary but still see [v1.0.1-beta] in logs
- No new log messages

**Solution:**
```bash
# Stop old process
pkill -9 q-api-server

# Run the NEW binary (make sure path is correct!)
chmod +x ./q-api-server-v1.0.2-beta-height-fix
./q-api-server-v1.0.2-beta-height-fix --port 8080
```

### Cause #3: Different Code Path (User vs Bootstrap)

**Symptoms:**
- Binary has fix symbols
- Bootstrap node works, user node doesn't
- Different configuration or environment

**Investigation Needed:**
- What command line arguments are you using?
- What environment variables are set (Q_NETWORK_ID, etc.)?
- Are you running as bootstrap node or regular node?

### Cause #4: Database State Issue

**Symptoms:**
- Fix messages appear in logs
- But height stuck at 1 in database
- Storage writes failing

**Investigation Needed:**
```bash
# Check if blocks are being saved
ls -lh ./data/blocks/ | tail -10

# Check storage errors
tail -100 node.log | grep -i "storage\|database\|save.*block"
```

---

## What I Need From You

Please provide the following information:

### 1. Binary Verification
```bash
nm ./q-api-server-v1.0.2-beta-height-fix | grep -i "advance.*producer.*height" | wc -l
# Should output: 3
```

### 2. Full Log Output (Last 100 Lines)
```bash
tail -100 node.log
```

### 3. Startup Command
```bash
# What command did you use to start the node?
# Example: ./q-api-server-v1.0.2-beta-height-fix --port 8080 --network testnet-phase8
```

### 4. Current Status
```bash
curl -s http://localhost:8080/api/v1/status | jq '.'
```

### 5. Download Details
```bash
# Where did you download from?
# What is the file size?
ls -lh ./q-api-server-v1.0.2-beta-height-fix

# When was it downloaded?
stat ./q-api-server-v1.0.2-beta-height-fix
```

---

## Expected Behavior (Bootstrap Node - WORKING)

```
2025-11-14T12:04:43.004531Z  INFO q_api_server::lockfree_producer: ✅ [v1.0.8-beta FIX] Pool: Producer #4 height advance command sent AFTER storage confirmation
2025-11-14T12:04:43.025129Z  INFO q_api_server::lockfree_producer: ✅ [v1.0.8-beta FIX] Pool: Producer #5 height advance command sent AFTER storage confirmation
2025-11-14T12:04:48.758981Z  INFO q_api_server: ✅ Producer #2 height advanced to 80826 AFTER storage confirmation
2025-11-14T12:04:49.099540Z  INFO q_api_server: ✅ Producer #4 height advanced to 80826 AFTER storage confirmation
```

**Key Indicators:**
- "v1.0.8-beta FIX" appears in logs ✅
- Height numbers are advancing (80826 → 80833 → 80834) ✅
- Continuous advancement every ~15 seconds ✅

---

## Quick Test Script

Save this as `test_height_fix.sh`:

```bash
#!/bin/bash

echo "=== Height Fix Diagnostic ==="
echo ""

echo "1. Binary verification:"
SYMBOLS=$(nm ./q-api-server-v1.0.2-beta-height-fix 2>/dev/null | grep -i "advance.*producer.*height" | wc -l)
if [ "$SYMBOLS" -eq 3 ]; then
    echo "   ✅ Binary HAS the fix ($SYMBOLS symbols found)"
else
    echo "   ❌ Binary MISSING the fix ($SYMBOLS symbols found, expected 3)"
fi
echo ""

echo "2. Current height:"
HEIGHT1=$(curl -s http://localhost:8080/api/v1/status | jq -r '.data.current_height')
echo "   Height: $HEIGHT1"
echo ""

echo "3. Waiting 30 seconds to check if height advances..."
sleep 30

HEIGHT2=$(curl -s http://localhost:8080/api/v1/status | jq -r '.data.current_height')
echo "   Height after 30s: $HEIGHT2"
echo ""

if [ "$HEIGHT1" != "$HEIGHT2" ] && [ "$HEIGHT2" != "null" ]; then
    echo "   ✅ Height IS advancing! (Fix working)"
else
    echo "   ❌ Height NOT advancing (Fix not working or not applied)"
fi
echo ""

echo "4. Checking logs for fix messages:"
if grep -q "v1.0.8-beta FIX" node.log 2>/dev/null; then
    echo "   ✅ Fix messages found in logs"
    echo "   Recent messages:"
    tail -50 node.log | grep "v1.0.8-beta FIX" | tail -5
else
    echo "   ❌ No fix messages found in logs"
    echo "   Are you running the correct binary?"
fi
echo ""

echo "=== Diagnostic Complete ==="
```

Run with:
```bash
chmod +x test_height_fix.sh
./test_height_fix.sh
```

---

## Download Links (Verified Working Binaries)

### Latest Height Fix (Recommended)
```
http://quillon.xyz/downloads/q-api-server-v1.0.8-beta-height-fix-CORRECTED
```
- Built: 2025-11-14 10:10:18 UTC
- Size: 123 MB
- Contains: advance_producer_height fix ✅

### Alternative (Also Has Fix)
```
http://quillon.xyz/downloads/q-api-server-v1.0.2-beta-height-fix
```
- Built: 2025-11-14 12:53 UTC
- Size: 123 MB
- Contains: advance_producer_height fix ✅

### Latest Stable
```
http://quillon.xyz/downloads/q-api-server-linux-x86_64
```
- Always points to latest build
- Size: 123 MB
- Updated: 2025-11-14 12:53 UTC

---

## Contact

If after running diagnostics the issue persists, please provide:
1. Output of `test_height_fix.sh`
2. Last 100 lines of node.log
3. Binary file size and download timestamp
4. Startup command used

This will help identify whether the issue is:
- Wrong binary downloaded
- Fix not being called
- Different code path for user nodes
- Database/storage issue

---

**End of Diagnostic Instructions**
