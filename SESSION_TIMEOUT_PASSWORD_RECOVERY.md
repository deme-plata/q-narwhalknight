# Session Timeout with Password Recovery - Implementation Complete

## Overview

Implemented automatic session timeout with password-based mnemonic recovery. After the configured timeout period (5min, 15min, 30min, 1hr, 4hr, or never), the system will prompt for the password to decrypt and restore the mnemonic, instead of requiring the full 12-word phrase again.

## Problem

User reported: "setting max security to 5 minutes dont work it never apears the password modal"

The session timeout setting was working, but after timeout expired, users had to re-enter their full 12-word mnemonic phrase instead of just their password.

## Root Cause

1. **Session timeout was clearing the plaintext mnemonic** (`localStorage['walletSeed']`)
2. **BUT the encrypted mnemonic was not being stored** when the wallet was created
3. **No password recovery mechanism** to decrypt and restore the mnemonic
4. **User had to log in again with full mnemonic** (bad UX)

## Solution Implemented

### 1. Store Encrypted Mnemonic During Login

**File**: `gui/quantum-wallet/src/services/walletAuth.ts:241-267`

```typescript
export async function storeWallet(
  mnemonic: string,
  password: string
): Promise<WalletKeyPair> {
  const keyPair = await keypairFromMnemonic(mnemonic);
  const encryptedPrivateKey = await encryptPrivateKey(keyPair.privateKey, password);

  // NEW: Also encrypt the mnemonic using the same password
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const encryptedMnemonic = await encryptPrivateKey(mnemonicBytes, password);

  // Store encrypted private key, mnemonic, and public address
  localStorage.setItem('walletAddress', keyPair.address);
  localStorage.setItem('walletEncryptedKey', encryptedPrivateKey);
  localStorage.setItem('walletEncryptedMnemonic', encryptedMnemonic); // NEW
  localStorage.setItem('walletPublicKey', bytesToHex(keyPair.publicKey));

  // Remove plaintext mnemonic when password is provided (security)
  localStorage.removeItem('walletSeed');

  return keyPair;
}
```

**Encryption Details**:
- Uses AES-256-GCM (Galois/Counter Mode) for authenticated encryption
- Password-based key derivation with PBKDF2 (100,000 iterations, SHA-256)
- Random 128-bit salt and 96-bit IV for each encryption
- Same security level as the private key encryption

### 2. Add Mnemonic Recovery Function

**File**: `gui/quantum-wallet/src/services/walletAuth.ts:291-309`

```typescript
/**
 * Decrypt and recover mnemonic from encrypted storage
 * Returns the plaintext mnemonic after successful password verification
 */
export async function recoverMnemonic(password: string): Promise<string> {
  const encryptedMnemonic = localStorage.getItem('walletEncryptedMnemonic');

  if (!encryptedMnemonic) {
    throw new Error('No encrypted mnemonic found.');
  }

  try {
    const mnemonicBytes = await decryptPrivateKey(encryptedMnemonic, password);
    const mnemonic = new TextDecoder().decode(mnemonicBytes);
    return mnemonic;
  } catch (error) {
    throw new Error('Failed to decrypt mnemonic. Incorrect password.');
  }
}
```

### 3. Session Monitor (Active Timeout Checking)

**File**: `gui/quantum-wallet/src/services/walletAuth.ts:470-500`

```typescript
/**
 * Start monitoring session expiry
 * Checks every 10 seconds if the session has expired and clears it automatically
 */
private startSessionMonitor() {
  // Clear any existing interval
  if (this.sessionCheckInterval !== null) {
    clearInterval(this.sessionCheckInterval);
  }

  // Check session expiry every 10 seconds
  this.sessionCheckInterval = window.setInterval(() => {
    if (this.privateKey && this.address) {
      // Check if session has expired
      if (Date.now() > this.expiresAt) {
        console.log('🔒 Session expired - clearing session');
        this.clearSession();
      }
    }
  }, 10000); // Check every 10 seconds
}
```

**What happens when session expires**:
1. Session monitor detects expiry every 10 seconds
2. Calls `clearSession()` which removes:
   - `sessionStorage['walletSession']` (encrypted private key + address)
   - `localStorage['walletSeed']` (plaintext mnemonic)
3. User continues browsing the wallet (read-only operations work)
4. When user tries to send a transaction, password prompt appears

### 4. Automatic Password Prompt on Transaction

**File**: `gui/quantum-wallet/src/services/api.ts:281-339`

```typescript
async sendTransaction(from: string, to: string, amount: number, memo?: string) {
  const fromAddress = from || localStorage.getItem('walletAddress') || '';

  // Get mnemonic from localStorage
  let mnemonic = localStorage.getItem('walletSeed') || '';

  // If mnemonic not found, try to recover from encrypted storage with password
  if (!mnemonic) {
    const encryptedMnemonic = localStorage.getItem('walletEncryptedMnemonic');

    if (encryptedMnemonic) {
      // Prompt for password to decrypt mnemonic
      const password = window.prompt('🔒 Session expired. Enter your password to continue:');

      if (!password) {
        return {
          success: false,
          error: 'Password required to decrypt wallet. Transaction cancelled.',
        };
      }

      try {
        // Import recovery function
        const { recoverMnemonic, walletSession, keypairFromMnemonic } = await import('./walletAuth');

        // Decrypt mnemonic
        mnemonic = await recoverMnemonic(password);

        // Restore session
        const keyPair = await keypairFromMnemonic(mnemonic);
        walletSession.setSession(keyPair.privateKey, keyPair.address);

        // Store mnemonic for current session
        localStorage.setItem('walletSeed', mnemonic);

        console.log('✅ Mnemonic recovered from encrypted storage');
      } catch (error) {
        return {
          success: false,
          error: 'Failed to decrypt wallet. Incorrect password.',
        };
      }
    } else {
      // No encrypted mnemonic found - user must log in again
      return {
        success: false,
        error: 'Wallet seed not found. Please log in again with your mnemonic phrase.',
      };
    }
  }

  // Continue with transaction...
}
```

### 5. Updated Login Flow

**File**: `gui/quantum-wallet/src/components/LoginScreen.tsx:27-51`

```typescript
if (response.success && response.data) {
  // Store wallet address and ID
  localStorage.setItem('walletAddress', response.data.address_formatted || '');
  localStorage.setItem('walletId', response.data.id);

  // If user provided a password, encrypt and store the wallet + mnemonic
  if (password) {
    try {
      const wallet = await storeWallet(seedPhrase, password);
      // Automatically start session
      walletSession.setSession(wallet.privateKey, wallet.address);
      // Store plaintext mnemonic for current session
      localStorage.setItem('walletSeed', seedPhrase);
      console.log('✅ Wallet encrypted with password-protected mnemonic');
      console.log('✅ Session started - mnemonic available for transaction signing');
    } catch (error) {
      console.error('Failed to encrypt wallet:', error);
      // Fallback: store plaintext mnemonic if encryption fails
      localStorage.setItem('walletSeed', seedPhrase);
    }
  } else {
    // No password provided - store plaintext mnemonic (legacy mode, less secure)
    localStorage.setItem('walletSeed', seedPhrase);
    console.log('⚠️ Mnemonic stored in plaintext (no password protection)');
  }

  onAuthenticate();
}
```

## User Experience Flow

### Scenario 1: User with 5-minute timeout

1. **Login**: User enters mnemonic + password
   - Mnemonic encrypted and stored in `localStorage['walletEncryptedMnemonic']`
   - Plaintext mnemonic stored in `localStorage['walletSeed']` for current session
   - Session timeout set to 5 minutes

2. **Normal Usage** (within 5 minutes):
   - User sends transactions normally
   - Mnemonic read from `localStorage['walletSeed']`
   - No password prompt needed

3. **After 5 minutes**:
   - Session monitor detects expiry
   - Clears `localStorage['walletSeed']` (plaintext mnemonic)
   - Encrypted mnemonic remains in `localStorage['walletEncryptedMnemonic']`

4. **User tries to send transaction**:
   - Browser prompt appears: "🔒 Session expired. Enter your password to continue:"
   - User enters password
   - Mnemonic decrypted from encrypted storage
   - Session restored for another 5 minutes
   - Transaction proceeds

5. **Wrong password**:
   - Error: "Failed to decrypt wallet. Incorrect password."
   - User can try again or refresh page to log in with full mnemonic

### Scenario 2: User without password (legacy mode)

1. **Login**: User enters only mnemonic (no password)
   - Plaintext mnemonic stored in `localStorage['walletSeed']`
   - NO encrypted backup created
   - Session timeout ignored (always "never")

2. **Usage**:
   - Mnemonic remains in localStorage permanently
   - No password prompts
   - Less secure but convenient

## Security Analysis

### What's Stored in localStorage

**With Password Protection**:
```
localStorage['walletAddress']          = "qnk05d5e43b5db607..."  (public)
localStorage['walletId']               = "uuid-v4"               (public)
localStorage['walletPublicKey']        = "hex(public_key)"       (public)
localStorage['walletEncryptedKey']     = "{salt,iv,data}"        (AES-256-GCM encrypted)
localStorage['walletEncryptedMnemonic']= "{salt,iv,data}"        (AES-256-GCM encrypted)
localStorage['walletSeed']             = "word1 word2..."        (CLEARED after timeout)
localStorage['walletSessionTimeout']   = "5" | "15" | "30" | "60" | "240" | "never"
```

**Without Password Protection (Legacy)**:
```
localStorage['walletAddress']          = "qnk05d5e43b5db607..."  (public)
localStorage['walletId']               = "uuid-v4"               (public)
localStorage['walletSeed']             = "word1 word2..."        (PLAINTEXT - INSECURE)
```

### Encryption Security

- **Algorithm**: AES-256-GCM (Galois/Counter Mode)
- **Key Derivation**: PBKDF2 with SHA-256
- **Iterations**: 100,000 (NIST recommended minimum)
- **Salt**: 128-bit random (unique per encryption)
- **IV**: 96-bit random (recommended for GCM)
- **Authentication**: GCM provides built-in authentication tag

**Attack Resistance**:
- ✅ **Brute Force**: 100K iterations slow down password guessing
- ✅ **Rainbow Tables**: Random salt prevents pre-computed attacks
- ✅ **Replay**: Random IV ensures different ciphertext each time
- ✅ **Tampering**: GCM authentication tag detects modifications
- ⚠️ **Weak Passwords**: User must choose strong password

### Session Timeout Options

| Timeout | Security | Convenience | Recommended For |
|---------|----------|-------------|----------------|
| 5 min   | Maximum  | Low         | Shared/public devices |
| 15 min  | High     | Balanced    | **Recommended** |
| 30 min  | Moderate | Good        | Personal devices |
| 1 hour  | Low      | High        | Trusted environments |
| 4 hours | Very Low | Very High   | Testing/development |
| Never   | None     | Maximum     | Not recommended |

## Browser Storage Comparison

| Storage Type | Lifetime | Scope | Security |
|--------------|----------|-------|----------|
| `localStorage` | Permanent (until cleared) | Domain-wide | ⚠️ Accessible to all tabs, vulnerable to XSS |
| `sessionStorage` | Tab session only | Single tab | ✅ Isolated per tab, cleared on tab close |

**Current Implementation**:
- Encrypted wallet data: `localStorage` (survives tab close, requires password)
- Active session data: `sessionStorage` (cleared on tab close, temporary)
- Plaintext mnemonic: `localStorage` (cleared after timeout, temporary)

## Testing Instructions

### Test 1: Password Recovery (5-minute timeout)

1. Clear browser storage: `localStorage.clear(); sessionStorage.clear()`
2. Navigate to https://quillon.xyz/
3. Click "Generate Quantum Entropy" to create a mnemonic
4. Enter mnemonic + password: "TestPassword123"
5. Go to Settings → Security → Set "5 minutes" timeout
6. Verify console logs:
   ```
   ✅ Wallet encrypted with password-protected mnemonic
   ✅ Session started - mnemonic available for transaction signing
   ```

7. Check localStorage:
   ```javascript
   console.log(localStorage.getItem('walletEncryptedMnemonic')); // Should show encrypted JSON
   console.log(localStorage.getItem('walletSeed')); // Should show plaintext mnemonic
   ```

8. **Wait 5 minutes** (or manually clear for testing):
   ```javascript
   // Manual test: clear plaintext mnemonic
   localStorage.removeItem('walletSeed');
   sessionStorage.removeItem('walletSession');
   ```

9. Try to send a transaction
10. Browser prompt should appear: "🔒 Session expired. Enter your password to continue:"
11. Enter password: "TestPassword123"
12. Transaction should proceed
13. Check console:
    ```
    ✅ Mnemonic recovered from encrypted storage
    ```

### Test 2: Wrong Password

1. After timeout, try to send transaction
2. Enter wrong password in prompt
3. Should see error: "Failed to decrypt wallet. Incorrect password."
4. Try again with correct password
5. Should succeed

### Test 3: No Password (Legacy Mode)

1. Clear storage and reload
2. Generate new mnemonic
3. **Do NOT enter a password** (leave password field empty)
4. Login
5. Check console:
   ```
   ⚠️ Mnemonic stored in plaintext (no password protection)
   ```

6. Check localStorage:
   ```javascript
   console.log(localStorage.getItem('walletEncryptedMnemonic')); // null
   console.log(localStorage.getItem('walletSeed')); // Should show plaintext mnemonic
   ```

7. Mnemonic should remain in localStorage permanently
8. No password prompts should appear

### Test 4: Session Monitor

1. Login with password and 5-minute timeout
2. Open browser console
3. Watch for automatic session expiry:
   ```
   🔒 Session expired - cleared wallet session and mnemonic
   ⚠️ Please log in again to continue using the wallet
   ```

4. Verify mnemonic cleared:
   ```javascript
   console.log(localStorage.getItem('walletSeed')); // null
   ```

5. Encrypted mnemonic should still exist:
   ```javascript
   console.log(localStorage.getItem('walletEncryptedMnemonic')); // Still there
   ```

## Files Modified

### Backend (None - frontend-only feature)
- No backend changes required

### Frontend

1. **`gui/quantum-wallet/src/services/walletAuth.ts`**
   - Line 245-267: Store encrypted mnemonic alongside encrypted private key
   - Line 291-309: Add `recoverMnemonic()` function
   - Line 318-322: Add `sessionCheckInterval` property
   - Line 327-329: Start session monitor in constructor
   - Line 421-437: Update `clearSession()` to remove plaintext mnemonic
   - Line 470-500: Add `startSessionMonitor()` and `stopSessionMonitor()`

2. **`gui/quantum-wallet/src/services/api.ts`**
   - Line 281-339: Add automatic password prompt and mnemonic recovery in `sendTransaction()`

3. **`gui/quantum-wallet/src/components/LoginScreen.tsx`**
   - Line 27-51: Update login flow to handle password-protected vs plaintext mnemonic storage

4. **`gui/quantum-wallet/src/components/SettingsScreen.tsx`**
   - Line 19-28: Load and save session timeout setting
   - Line 188-254: UI for session timeout configuration
   - Line 285-296: Display current timeout setting

5. **`gui/quantum-wallet/src/hooks/usePasswordPrompt.ts`** (NEW FILE)
   - Password prompt hook for future use (not currently used, `window.prompt` used instead)

## Build Status

✅ **Frontend Built Successfully**: `gui/quantum-wallet/dist-final/`
✅ **Backend Running**: Port 8080, database: `./data`
✅ **Dev Server**: Port 5175 (development)
✅ **Production**: https://quillon.xyz/ (served via nginx)

## Known Limitations

1. **Browser Prompt**: Currently uses `window.prompt()` for password entry
   - **Future Enhancement**: Use proper modal component from `PasswordModalProvider`
   - Requires updating `PasswordModalContext` to work with mnemonic recovery

2. **No Multi-Device Sync**: Encrypted wallet stored locally only
   - If user switches devices, must re-enter full mnemonic
   - **Future Enhancement**: Optional cloud backup with end-to-end encryption

3. **No Password Strength Requirements**: User can choose weak passwords
   - **Future Enhancement**: Password strength meter + requirements

4. **XSS Vulnerability**: localStorage accessible to JavaScript
   - **Mitigation**: Password-based encryption reduces impact
   - **Future Enhancement**: Consider using IndexedDB with additional security layers

5. **No Hardware Wallet Support**: All keys stored in browser
   - **Future Enhancement**: Ledger/Trezor integration for cold storage

## Next Steps

### Immediate (Ready to Test)
- ✅ Session timeout with password recovery implemented
- ✅ Encrypted mnemonic storage
- ✅ Automatic session monitoring
- ⏳ User testing with real transactions

### Short-term Enhancements
1. Replace `window.prompt()` with proper password modal
2. Add password strength requirements (min 8 chars, uppercase, numbers, symbols)
3. Add "Remember for X minutes" checkbox after password entry
4. Show countdown timer for session expiry in UI

### Long-term Enhancements
1. Hardware wallet support (Ledger, Trezor)
2. Biometric authentication (WebAuthn/FIDO2)
3. Multi-signature wallet support
4. Cloud backup with end-to-end encryption
5. Password recovery via security questions or email

## Status

✅ **COMPLETE - READY FOR TESTING**

The session timeout with password recovery is now fully implemented and ready for user testing. When the user sets a security timeout (e.g., 5 minutes), they will be prompted for their password to continue using the wallet after the timeout expires, instead of having to re-enter the full 12-word mnemonic phrase.

---

**Implemented by**: Claude Code (Server Beta)
**Date**: 2025-10-12
**Feature**: Session Timeout with Password-Based Mnemonic Recovery
**Security Level**: AES-256-GCM + PBKDF2 (100K iterations)
