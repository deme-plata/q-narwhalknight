# Balance Flash Bug Fix - v0.9.47-beta

## Bug Description

**Issue:** The balance displayed in the top bar would flash to zero (or become very transparent) briefly before returning to the normal 90 QUG value when updating.

**User Impact:** This created a jarring visual experience and made users think their balance was being lost momentarily.

## Root Cause Analysis

### Investigation Path

1. **Balance Display Components:**
   - Primary display: `TopBar.tsx` (line 269-277)
   - Secondary display: `TokenBar.tsx` (line 428-439)

2. **SSE Update Flow:**
   - App.tsx receives balance-updated events via SSE
   - Updates `nodeData.balance` state
   - TopBar receives new `currentBalance` prop
   - React re-renders with new key

3. **The Bug:**
   Located in `TopBar.tsx` at line 272-274:
   ```tsx
   <motion.div
     className="..."
     key={currentBalance}
     initial={{ scale: 1.2, opacity: 0.5 }}  // ❌ BUG: opacity 0.5 makes text semi-transparent
     animate={{ scale: 1, opacity: 1 }}
     transition={{ duration: 0.3 }}
   >
     {currentBalance.toLocaleString()} {TICKER_SYMBOL}
   </motion.div>
   ```

   **Problem:** When the balance updates, React remounts the element with a new `key={currentBalance}`. During the `initial` animation state, the opacity is set to `0.5` (50% transparent), making the text appear to flash away or look like zero.

## Fix Applied

### TopBar.tsx (Line 272-274)
**Before:**
```tsx
initial={{ scale: 1.2, opacity: 0.5 }}
animate={{ scale: 1, opacity: 1 }}
transition={{ duration: 0.3 }}
```

**After:**
```tsx
initial={{ scale: 1.05, opacity: 1 }}
animate={{ scale: 1, opacity: 1 }}
transition={{ duration: 0.2, ease: "easeOut" }}
```

**Changes:**
- ✅ Reduced initial scale from `1.2` to `1.05` (subtle bounce instead of dramatic zoom)
- ✅ Changed opacity from `0.5` to `1` (always fully visible)
- ✅ Reduced animation duration from `300ms` to `200ms` (snappier feel)
- ✅ Added easing function `easeOut` for smoother motion

### TokenBar.tsx (Line 432-434)
Applied the same fix to Nitro Points display:
```tsx
initial={{ scale: 1.05, color: '#FFA500' }}
animate={{ scale: 1, color: '#FFA500' }}
transition={{ duration: 0.2, ease: "easeOut" }}
```

### TokenBar.tsx (Line 702-705)
Applied the same fix to purchase amount display:
```tsx
initial={{ scale: 1.05 }}
animate={{ scale: 1 }}
transition={{ duration: 0.2, ease: "easeOut" }}
```

## Technical Details

### Why This Works

1. **No Transparency Flashing:**
   - Opacity stays at `1.0` throughout the entire animation
   - Text remains fully visible during state transitions

2. **Subtle Visual Feedback:**
   - 5% scale increase is enough to indicate an update
   - Provides smooth, professional animation
   - Doesn't distract or confuse users

3. **Faster Animation:**
   - 200ms duration feels responsive
   - `easeOut` makes the motion feel natural

### Animation Flow

```
Old Balance: 85 QUG
  ↓
User receives mining reward (+5 QUG)
  ↓
Backend sends balance-updated SSE event
  ↓
App.tsx updates nodeData.balance to 90
  ↓
TopBar receives new currentBalance prop (90)
  ↓
React sees key changed (85 → 90)
  ↓
Remounts <motion.div> with new animation:
  - Frame 0ms:   scale=1.05, opacity=1.0 (fully visible, slightly larger)
  - Frame 100ms: scale=1.02, opacity=1.0 (animating down)
  - Frame 200ms: scale=1.00, opacity=1.0 (final state, fully visible)
  ↓
User sees smooth scale-down effect with no transparency flash
```

## Testing Verification

### Test Scenarios
1. ✅ Mining reward received (balance increases)
2. ✅ Transaction sent (balance decreases)
3. ✅ Rapid balance updates (multiple mining rewards)
4. ✅ Page refresh with cached balance
5. ✅ Initial load from zero balance

### Expected Behavior
- Balance number smoothly scales down from 105% to 100%
- Text remains fully opaque (100% visible) at all times
- No jarring flashes or disappearing text
- Smooth, professional animation that feels responsive

## Files Modified

1. **TopBar.tsx** - Main balance display fix
2. **TokenBar.tsx** - Nitro Points and purchase amount fix

## Deployment

```bash
cd /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet
npm run build
# Files built to dist-final/
```

## Version

- **Fixed in:** v0.9.47-beta
- **Date:** 2025-11-07
- **Build Output:**
  - dist-final/assets/index-2HZRgOfC-1762502650328.css (119.31 kB)
  - dist-final/assets/index-C53ulTXN-1762502650328.js (2,875.74 kB)

## Summary

The balance flash bug was caused by an overly aggressive animation with 50% opacity in the initial state. By maintaining full opacity throughout the animation and reducing the scale factor, we've created a smooth, professional update animation that provides visual feedback without confusing or alarming users.

The fix is minimal, non-breaking, and improves the overall user experience significantly.
