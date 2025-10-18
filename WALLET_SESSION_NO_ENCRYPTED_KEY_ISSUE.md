# Wallet Session - No Encrypted Key Issue

## Problem

User sees this error when trying to view transaction history:

```
⚠️ Failed to load transaction history
No encrypted wallet found. Please log in with your mnemonic phrase and password.
💡 Tip: This usually means you need to log in with your wallet password. Go to Settings → Login/Import Wallet.
```

**Wallet Address Displayed**: `qnke68940733ee5e3f22cd990479984978fe2938cd39d8c4f320e465b63800c6f78`

**Symptoms**:
- Wallet address is visible in the UI (from `sessionStorage`)
- Token balances show (from API)
- Transaction history fails to load
- Error: "No encrypted wallet found"

## Root Cause

This happens when:

1. **Session Storage vs Local Storage Mismatch**
   - Wallet address is stored in `sessionStorage` (temporary, survives page refresh)
   - Encrypted private key should be in `localStorage` (permanent, survives browser close)
   - If `localStorage` was cleared but `sessionStorage` wasn't, this error occurs

2. **Incomplete Wallet Creation**
   - User created wallet but `storeWallet()` function failed to complete
   - Password encryption step didn't complete properly
   - Browser blocked `localStorage` writes (privacy mode, quota exceeded, etc.)

3. **User Scenario**:
   - User created a "fresh wallet" → address generated and stored in session
   - Wallet encryption to `localStorage` failed or was skipped
   - Now authenticated requests fail because there's no encrypted key to decrypt

## Architecture

### Storage Layers:

**sessionStorage** (Temporary - cleared on browser close):
- `walletSession`: Current active session with privateKey, address, mnemonic
- Used for "logged in" state during current browser session
- Faster access, no password needed during session

**localStorage** (Permanent - persists across browser sessions):
- `walletAddress`: Public wallet address (qnk prefix + 64 hex chars)
- `walletEncryptedKey`: AES-256-GCM encrypted private key (requires password to decrypt)
- `walletEncryptedMnemonic`: AES-256-GCM encrypted mnemonic phrase
- `walletPublicKey`: Hex-encoded Ed25519 public key
- `walletEncryptedAegisKey`: AES-256-GCM encrypted AEGIS-QL post-quantum key (optional)
- `walletAegisPublicKey`: AEGIS-QL public key JSON (optional)

### Authentication Flow:

```
User logs in with mnemonic + password
        ↓
LoginScreen calls storeWallet(mnemonic, password, true)
        ↓
storeWallet() derives keypair from mnemonic
        ↓
Encrypts private key with AES-256-GCM (password-based)
        ↓
Stores encrypted data in localStorage:
  - walletEncryptedKey
  - walletEncryptedMnemonic
  - walletPublicKey
  - walletAddress
        ↓
Starts session in sessionStorage:
  - Decrypted privateKey (for signing)
  - Address
  - Mnemonic (if "Never expire" enabled)
```

### Transaction History Request Flow:

```
User opens Dashboard → loads recent transactions
        ↓
api.ts authenticatedRequest() called
        ↓
Check walletSession.getSession()
        ↓
Session active? ✅ Use cached privateKey
        ↓
Session expired? ❌ Check localStorage
        ↓
localStorage.getItem('walletEncryptedKey')
        ↓
Found? Prompt for password → decrypt → sign request
NOT FOUND? ⚠️ ERROR: "No encrypted wallet found"
```

## Solution

### For User:

**Option 1: Re-Login with Mnemonic + Password** (Recommended)
1. Go to **Settings** → **Login/Import Wallet**
2. Enter your **12-word mnemonic phrase**
3. Enter a **strong password** (will be used to encrypt the wallet)
4. Click **Authenticate**

This will:
- Re-derive your wallet from the mnemonic
- Properly encrypt and store the private key in `localStorage`
- Start a new session in `sessionStorage`
- Fix the "No encrypted wallet found" error

**Option 2: Clear All Storage and Start Fresh**
1. Open browser DevTools (F12)
2. Go to **Application** → **Storage**
3. Clear both **Local Storage** and **Session Storage**
4. Refresh the page
5. Create a new wallet OR import existing wallet with mnemonic

⚠️ **WARNING**: If you don't have your mnemonic phrase backed up, DO NOT clear storage! You'll lose access to your wallet permanently.

### For Developer:

**Prevent This Issue**:

1. **Add Error Handling to storeWallet()**:
```typescript
try {
  const wallet = await storeWallet(seedPhrase, password, true);
  walletSession.setSession(wallet.privateKey, wallet.address, seedPhrase);
  console.log('✅ Wallet stored successfully');
} catch (error) {
  console.error('❌ CRITICAL: Failed to store wallet:', error);
  // Show user-friendly error message
  setGenerationError('Failed to securely store wallet. Please try again or check browser settings.');
  return;
}
```

2. **Verify Storage After Writing**:
```typescript
// After storeWallet(), verify it worked
const encryptedKey = localStorage.getItem('walletEncryptedKey');
if (!encryptedKey) {
  throw new Error('Wallet storage verification failed - localStorage may be blocked');
}
```

3. **Check localStorage Availability**:
```typescript
function isLocalStorageAvailable(): boolean {
  try {
    const test = '__localStorage_test__';
    localStorage.setItem(test, test);
    localStorage.removeItem(test);
    return true;
  } catch (e) {
    return false;
  }
}

// Before creating wallet:
if (!isLocalStorageAvailable()) {
  alert('localStorage is not available. Please disable private browsing mode or check browser settings.');
  return;
}
```

4. **Graceful Degradation**:
- Detect when `localStorage` is unavailable
- Show warning to user: "Your browser's privacy settings prevent secure wallet storage"
- Offer session-only mode (wallet lost on browser close)

## Testing

### Reproduce the Issue:

1. Open browser DevTools (F12)
2. Go to **Application** → **Local Storage**
3. Manually delete `walletEncryptedKey` entry
4. Refresh the page
5. Try to load transaction history → Error appears

### Verify the Fix:

1. Re-login with mnemonic + password
2. Check **Application** → **Local Storage**
3. Verify `walletEncryptedKey` exists
4. Refresh page
5. Transaction history loads successfully

## Prevention

### Best Practices:

1. **Always Test Storage Availability**:
   - Before wallet creation
   - After wallet creation (verify data was written)

2. **User Education**:
   - Explain the importance of backing up mnemonic phrase
   - Warn about private browsing mode limitations
   - Show clear error messages when storage fails

3. **Robust Error Handling**:
   - Catch all storage errors
   - Show user-friendly messages
   - Provide recovery options

4. **Session Management**:
   - `sessionStorage` for active session (temporary)
   - `localStorage` for encrypted wallet (permanent)
   - Always check both before assuming user is logged in

## Related Files

- `gui/quantum-wallet/src/services/walletAuth.ts:288` - `storeWallet()` function
- `gui/quantum-wallet/src/services/api.ts:149` - "No encrypted wallet found" error
- `gui/quantum-wallet/src/components/LoginScreen.tsx:88` - Wallet creation with `storeWallet()`
- `gui/quantum-wallet/src/services/walletAuth.ts:434` - `WalletSession` class

## Status

- **Issue Identified**: ✅ Storage mismatch between sessionStorage and localStorage
- **Root Cause**: ✅ `storeWallet()` didn't complete or localStorage was cleared
- **User Solution**: ✅ Re-login with mnemonic + password
- **Prevention**: ⏳ Add storage verification and error handling (future improvement)

---

**Date**: 2025-10-17
**Severity**: MEDIUM (User can recover by re-logging in with mnemonic)
**Impact**: Transaction history and authenticated API calls fail
**Fix**: User re-authentication with mnemonic + password
