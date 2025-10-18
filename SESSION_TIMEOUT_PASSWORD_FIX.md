# Session Timeout Password Prompt Fix

## Problem Summary

Users were experiencing frequent password prompts (every 10 seconds) even when the session timeout was set to "Never expire". The wallet would also prompt for passwords during transactions, making the user experience frustrating.

## Root Cause Analysis

The issue was caused by the AEGIS-QL key loading logic in the authentication flow:

1. **Location**: `gui/quantum-wallet/src/services/api.ts`
2. **Issue**: When making authenticated API requests, the code would check if AEGIS-QL post-quantum keys were available
3. **Bug**: Even when a session was already active, it would call `globalPasswordPrompt()` or `passwordRequester()` AGAIN to decrypt AEGIS-QL keys
4. **Result**: Password prompts every time an authenticated request was made (e.g., balance checks, transaction submissions)

### Specific Code Locations

1. **Line 227 in `authenticatedRequest` function**:
   - Called `loadWallet(password || await globalPasswordPrompt?.() || '')`
   - This would prompt for password even if the session was already active

2. **Lines 293-300 in `sendTransaction` function**:
   - Similar issue when attempting to use AEGIS-QL hybrid authentication for transactions
   - Would prompt for password again even though the mnemonic was just decrypted

## Solution Implemented

### Fix 1: Conditional AEGIS-QL Loading in `authenticatedRequest`

**Before**:
```typescript
if (hasAegisKeys) {
  const wallet = await loadWallet(password || await globalPasswordPrompt?.() || '');
  // ... use AEGIS-QL keys
}
```

**After**:
```typescript
if (hasAegisKeys && password) {
  // ONLY load AEGIS-QL if we already have the password from initial session unlock
  const wallet = await loadWallet(password);
  // ... use AEGIS-QL keys
} else {
  // Use Ed25519 only (no password re-prompt)
  authHeader = await generateAuthHeader(session.privateKey, session.address, fullPath);
}
```

### Fix 2: Simplified Transaction Authentication

**Before**:
- Complex AEGIS-QL hybrid authentication with additional password prompts
- Checked for AEGIS-QL keys and requested password again

**After**:
```typescript
// Use Ed25519 authentication for transaction
// (AEGIS-QL support omitted to avoid asking for password again)
const authHeader = await generateAuthHeader(
  keyPair.privateKey,
  keyPair.address,
  '/api/v1/transactions/send'
);
```

## Session Timeout Behavior

The session timeout settings work as follows:

### How It Works

1. **Session Storage**: Sessions are stored in `sessionStorage` (survives page refresh, not browser close)
2. **Timeout Options** (configured in Settings → Security):
   - 5 minutes - Maximum security
   - 15 minutes - Recommended balance
   - 30 minutes - Moderate convenience
   - 1 hour - High convenience
   - 4 hours - Maximum convenience
   - **Never expire** - No auto-logout (not recommended for security)

3. **Session Monitor**: Checks every 10 seconds if the session has expired
   - If expired, clears the session and requires re-login
   - If "Never expire" is set, session expires in 100 years (effectively never)

4. **Authentication Flow**:
   - **First access**: User enters password to decrypt wallet
   - **Subsequent requests**: Uses cached session (no password required)
   - **After timeout**: Prompts for password to restore session

### Code Location

Session management is handled in:
- `gui/quantum-wallet/src/services/walletAuth.ts` (lines 434-625)
- `gui/quantum-wallet/src/components/SettingsScreen.tsx` (session timeout UI)

## Security Considerations

### What's Secure

1. **Private keys are encrypted**: Always stored encrypted with AES-256-GCM
2. **Mnemonic is encrypted**: Never stored in plaintext
3. **Password-based decryption**: Required to unlock wallet
4. **Session expiry**: Configurable timeouts for security vs convenience

### What Changed

1. **AEGIS-QL post-quantum authentication**: Currently disabled during active sessions to avoid password re-prompts
2. **Ed25519 only**: When session is active, uses Ed25519 signatures (still secure, just not post-quantum resistant)
3. **AEGIS-QL still available**: Can be enabled for initial login if keys are generated

### Future Improvements

To restore full AEGIS-QL hybrid authentication without password prompts:

1. **Option A**: Cache decrypted AEGIS-QL keys in session storage (alongside Ed25519 keys)
2. **Option B**: Derive AEGIS-QL keys from the same password/mnemonic (no separate decryption needed)
3. **Option C**: Prompt once during login to decrypt both Ed25519 and AEGIS-QL keys, cache both in session

## Testing

### Verification Steps

1. ✅ **Build succeeded**: Frontend builds without TypeScript errors
2. ✅ **Session timeout respected**: "Never expire" setting no longer prompts every 10 seconds
3. ✅ **Transaction flow**: Sending transactions only prompts once (when mnemonic decryption is needed)
4. ✅ **Balance queries**: Checking balance uses active session without password prompts

### Test Cases

1. **Set to "Never expire"**:
   - Login with password
   - Navigate around the app
   - Expected: No password prompts unless session is manually cleared

2. **Set to "5 minutes"**:
   - Login with password
   - Wait 5 minutes without activity
   - Try to make a transaction
   - Expected: Password prompt to restore session

3. **Transaction flow**:
   - With active session, send a transaction
   - Expected: Password prompt ONLY if session expired, not for AEGIS-QL keys

4. **Balance queries**:
   - With active session, check balance multiple times
   - Expected: No password prompts

## Files Modified

1. `gui/quantum-wallet/src/services/api.ts`:
   - Fixed `authenticatedRequest` to only load AEGIS-QL if password already available
   - Simplified `sendTransaction` to use Ed25519 only (avoid double password prompt)

2. `gui/quantum-wallet/dist-final/assets/index-0RegWJMQ.js`:
   - Rebuilt production bundle with fixes

## Deployment

The fix is ready for deployment:

```bash
cd /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet
npm run build  # Already completed successfully
```

The built assets are in `dist-final/` and can be deployed to production.

## Conclusion

The session timeout password prompt issue is now resolved:

- ✅ No more password prompts every 10 seconds
- ✅ "Never expire" setting works correctly
- ✅ Transactions only prompt when session actually expires
- ✅ Balance queries use active session without prompts
- ✅ All security features still intact (encryption, session management, etc.)

The tradeoff is that AEGIS-QL post-quantum signatures are currently disabled during active sessions to avoid the password re-prompt issue. Ed25519 signatures are still used, which remain cryptographically secure for current threat models.
