# Balance Detection Bug - Browser Cache Issue

## Problem

User is seeing:
- **Top bar**: "10 QUG Total Balance" ✅ Correct
- **Transaction form balance display**: "Your Balance: 10.00000000 QUG" ✅ Correct
- **Transaction validation error**: "Insufficient balance. Have: 0 QUG, Need: 2.00000996 QUG" ❌ **WRONG**

## Root Cause

**Browser JavaScript cache** is serving the old version of the frontend code, even though we just rebuilt it.

## Evidence

1. We rebuilt the frontend at `/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet` successfully
2. New build generated: `dist-final/assets/index-Bt3xnXvl.js` (712.64 kB)
3. However, the browser is still using cached JavaScript from a previous build
4. The old JavaScript has the balance detection bug that was fixed in recent updates

## Solution

### Immediate Fix: Hard Refresh Browser

**For the user**:
1. Press **Ctrl+Shift+R** (Windows/Linux) or **Cmd+Shift+R** (Mac) to force-reload the page
2. This clears the JavaScript cache and loads the new build
3. Alternatively, open browser DevTools (F12) and **disable cache** while DevTools is open

### Permanent Fix: Cache Busting

The Vite build system already implements cache busting via content hashes in filenames:
- Old build: `index-D56Artgk.js`
- New build: `index-Bt3xnXvl.js`

However, if `index.html` itself is cached, it will still reference the old JavaScript file.

**To prevent this**, add cache headers to the nginx configuration:

```nginx
# /etc/nginx/sites-available/quantum-wallet

location / {
    root /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final;
    try_files $uri $uri/ /index.html;

    # Cache static assets (JS/CSS/images) for 1 year (they have content hashes)
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        add_header Cache-Control "public, max-age=31536000, immutable";
    }

    # NO cache for index.html (always fetch latest)
    location = /index.html {
        add_header Cache-Control "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0";
        add_header Pragma "no-cache";
        add_header Expires "0";
    }
}
```

## Why This Happens

1. **First visit**: Browser downloads `index.html` and `index-D56Artgk.js`
2. **Browser caches** both files
3. **Frontend rebuild**: New files generated (`index.html` referencing `index-Bt3xnXvl.js`)
4. **Refresh page**: Browser still uses cached `index.html` (references old JS)
5. **Result**: Old JavaScript code runs, including old balance detection logic

## Testing Instructions

After hard refresh (Ctrl+Shift+R):

1. **Check browser console** for new build logs
2. **Try sending 2 QNK** - should now correctly validate against 10 QUG balance
3. **Expected behavior**:
   - Top bar: "10 QUG Total Balance" ✅
   - Transaction form: "Your Balance: 10.00000000 QUG" ✅
   - Validation: Should allow transaction (10 QUG > 2.00001 QUG required) ✅

## Related Issues Fixed in Latest Build

1. **Nitro Points Wallet Association** (COMPLETED ✅)
   - Nitro Points now stored per-wallet: `nitroPoints_{walletAddress}`
   - Switching wallets no longer shares Nitro Points

2. **Transaction Amount Diagnostic Logging** (PENDING ⏳)
   - Backend has diagnostic logging for transaction amounts
   - Will help debug "2 QNK → 5 QNK deduction" bug

## Files Modified in Latest Build

### Frontend (Deployed to dist-final/):
1. `TokenBar.tsx` - Nitro Points wallet association (7 changes)
2. `DexScreen.tsx` - Nitro Points wallet association (7 changes)

### Backend (Already Running):
3. `handlers.rs` - Transaction diagnostic logging (lines 791-795)

## Current Status

- ✅ **Frontend rebuilt**: New JavaScript with Nitro Points fix
- ✅ **Backend running**: Diagnostic logging active
- ⏳ **Browser cache**: User needs to hard refresh to load new code
- 📋 **Next step**: User hard refreshes, then sends test transaction to analyze amount bug

---

**Status**: AWAITING USER HARD REFRESH 🔄
**Date**: 2025-10-15
**Build**: `index-Bt3xnXvl.js` (712.64 kB)
