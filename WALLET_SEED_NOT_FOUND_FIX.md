# "Wallet Seed Not Found" Error - How to Fix

## Problem
You're seeing the error: **"Wallet seed not found. Please log in again with your mnemonic phrase."**

This means your wallet's encrypted data is not present in the browser's localStorage, which is required to authenticate API requests like fetching transactions.

## Why This Happens

When you create a new wallet through the LoginScreen, the system needs to:
1. Generate or accept a mnemonic phrase (12 words)
2. **Encrypt it with a password** you provide
3. Save the encrypted data to localStorage as `walletEncryptedKey`

If you skipped the password step or the data wasn't saved properly, authentication will fail.

## How to Fix - Step by Step

### Step 1: Go to Settings → Login/Import Wallet

1. Click on the **Settings** icon in the navigation
2. Look for **"Login/Import Wallet"** section

### Step 2: Enter Your Mnemonic Phrase

If you still have your 12-word mnemonic phrase:
1. Paste it into the seed phrase field
2. **IMPORTANT**: Enter a strong password (required!)
3. Click "Authenticate"

### Step 3: Or Generate a New Wallet

If you lost your mnemonic phrase:
1. Click **"Generate Quantum Entropy"** to create a new wallet
2. **SAVE THE MNEMONIC PHRASE SOMEWHERE SAFE!**
3. Enter a password (required!)
4. Click "Authenticate"

⚠️ **WARNING**: Generating a new wallet will create a completely different address. Any funds in your old wallet will be inaccessible without the old mnemonic.

## What the System Does When You Log In

When you click "Authenticate" with a mnemonic + password:

1. **Encrypts your mnemonic** with AES-256-GCM using your password
2. **Saves to localStorage**:
   - `walletEncryptedMnemonic` - Your encrypted seed phrase
   - `walletEncryptedKey` - Your encrypted private key
   - `walletAddress` - Your public wallet address
   - `walletAegisPublicKey` - Post-quantum public key (optional)
   - `walletEncryptedAegisKey` - Post-quantum private key (optional)

3. **Creates a session** so you don't need to re-enter your password for every transaction

## Why Password is Required

The password is used to:
- Encrypt your mnemonic phrase before storing it locally
- Decrypt it when you need to sign transactions
- Protect your wallet even if someone gains access to your browser storage

**Never share your password or mnemonic phrase with anyone!**

## Checking If Wallet is Properly Stored

Open browser console (F12) and run:

```javascript
console.log({
  hasEncryptedKey: !!localStorage.getItem('walletEncryptedKey'),
  hasEncryptedMnemonic: !!localStorage.getItem('walletEncryptedMnemonic'),
  hasWalletAddress: !!localStorage.getItem('walletAddress'),
  walletAddress: localStorage.getItem('walletAddress')
});
```

**You should see**:
```
{
  hasEncryptedKey: true,
  hasEncryptedMnemonic: true,
  hasWalletAddress: true,
  walletAddress: "qnk..."
}
```

If any of these are `false`, you need to log in again with your mnemonic + password.

## After Logging In Correctly

Once you've logged in with your mnemonic + password:

1. ✅ Your wallet will be encrypted and saved
2. ✅ A session will be created
3. ✅ You'll be able to fetch transactions
4. ✅ You'll be able to send transactions
5. ✅ Recent Activity will load properly

## Still Not Working?

If you've logged in correctly but still see the error:

1. **Hard refresh** the page: `Ctrl + Shift + R` (Windows/Linux) or `Cmd + Shift + R` (Mac)
2. **Check the console** for any other error messages
3. **Verify** your encrypted wallet data is present (see "Checking If Wallet is Properly Stored" above)

## Technical Details

The authentication flow for fetching transactions:

1. Dashboard calls `fetchRecentTransactions()`
2. This calls `qnkAPI.getRecentTransactions()`
3. Which calls `authenticatedRequest()`
4. If no active session exists, it looks for `walletEncryptedKey` in localStorage
5. If found, prompts for password to decrypt it
6. Uses decrypted key to generate `X-Wallet-Auth` header
7. Makes authenticated API request to `/v1/transactions/recent`

**Missing `walletEncryptedKey` = No authentication = Error**

## Security Best Practices

1. ✅ **Always use a strong password** for wallet encryption
2. ✅ **Never store your mnemonic phrase in plain text**
3. ✅ **Write down your mnemonic phrase on paper** (not digitally)
4. ✅ **Keep your mnemonic phrase in a safe place**
5. ✅ **Never share your password or mnemonic with anyone**

---

**Bottom Line**: Go to Settings → Login/Import Wallet, enter your mnemonic + password, and click Authenticate. This will properly encrypt and store your wallet data so you can fetch transactions.
