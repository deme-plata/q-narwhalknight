# How to Create Your Wallet - Quick Guide

## Problem
You're seeing: **"No encrypted wallet data found"**

This means you haven't created or imported a wallet yet!

## Solution: Create a New Wallet

### Option 1: Generate a New Wallet (Recommended for new users)

1. **Go to Settings** (gear icon in navigation)

2. **Scroll to "Login/Import Wallet" section**

3. **Click "Generate Quantum Entropy"** button
   - Wait for the system to generate a secure 12-word phrase
   - The phrase will appear in the text box automatically

4. **CRITICAL: SAVE YOUR MNEMONIC PHRASE!**
   - Write down the 12 words on paper
   - Store them in a safe place
   - You'll need this to recover your wallet if you lose access!

5. **Enter a Password**
   - Type a strong password in the "Password" field
   - This password encrypts your wallet locally
   - You'll need it to sign transactions

6. **Click "Authenticate"**
   - Your wallet will be encrypted and saved
   - You'll be logged in automatically
   - Recent Activity will now load!

### Option 2: Import Existing Wallet

If you already have a 12-word mnemonic phrase:

1. **Go to Settings → Login/Import Wallet**

2. **Paste your 12-word phrase** into the "BIP39 Quantum Seed Phrase" field

3. **Enter a password** (can be different from your old password)

4. **Click "Authenticate"**

## After Creating Your Wallet

Once you click "Authenticate", the system will:

1. ✅ Generate your wallet address (starts with "qnk...")
2. ✅ Encrypt your mnemonic with your password
3. ✅ Save encrypted data to localStorage:
   - `walletEncryptedMnemonic`
   - `walletEncryptedKey`
   - `walletAddress`
   - `walletAegisPublicKey` (post-quantum encryption)
   - `walletEncryptedAegisKey` (post-quantum encryption)
4. ✅ Create an active session
5. ✅ Redirect you to the Dashboard

## What You'll See

After wallet creation:

- **Dashboard** will load with your wallet address displayed
- **Balance** will show (initially 0 QNK)
- **Recent Activity** will load (initially empty)
- **Faucet button** will appear if your balance is 0

## Get Some Test Tokens

Once your wallet is created:

1. Look for the **green coin icon** (appears when balance is 0)
2. Click it to request **10 QNK** from the faucet
3. Wait a few seconds
4. Your balance will update
5. Transaction will appear in Recent Activity!

## Common Mistakes

❌ **Forgetting to save your mnemonic phrase**
- You'll lose access to your wallet forever if you forget it!

❌ **Not entering a password**
- Password is REQUIRED for wallet encryption
- Without it, your wallet won't be saved

❌ **Using a weak password**
- Use a strong password to protect your funds
- Don't use "password" or "123456"!

## Security Reminders

1. ✅ **Write down your 12-word mnemonic on paper** (not digitally!)
2. ✅ **Never share your mnemonic with anyone**
3. ✅ **Keep your password safe**
4. ✅ **Back up your mnemonic in multiple secure locations**
5. ✅ **Never store your mnemonic in email, cloud storage, or screenshots**

## Technical Details

### What is a Mnemonic Phrase?

A mnemonic phrase (also called seed phrase) is a list of 12 words that represents your wallet's private key. It's like a master password that:

- Generates your wallet address
- Creates your private key for signing transactions
- Allows you to recover your wallet on any device

### What is the Password For?

The password you enter:

- **Encrypts your mnemonic** before storing it in browser localStorage
- **Protects your wallet** even if someone gains access to your browser
- **Unlocks your wallet** when you need to sign transactions

Your password is **NEVER sent to the server** - it's only used locally to encrypt/decrypt your wallet data.

### Post-Quantum Encryption

This wallet uses **AEGIS-QL** post-quantum encryption in addition to Ed25519:

- **Ed25519**: Classical elliptic curve cryptography (current standard)
- **AEGIS-QL**: Quantum-resistant authenticated encryption
- **Hybrid Mode**: Both are used together for maximum security

This means your wallet is secure against both current attacks AND future quantum computers!

## Troubleshooting

### "No encrypted wallet data found" persists

1. Make sure you clicked "Authenticate" after entering mnemonic + password
2. Check browser console (F12) for error messages
3. Try hard refresh: `Ctrl + Shift + R` (Windows/Linux) or `Cmd + Shift + R` (Mac)

### "Authentication failed"

- Check that you entered the mnemonic correctly (12 words, separated by spaces)
- Make sure you entered a password
- Try generating a new wallet instead of importing

### Still having issues?

Check the browser console (F12 → Console tab) for error messages and share them with support.

---

**TL;DR**: Go to Settings → Login/Import Wallet → Click "Generate Quantum Entropy" → Save the 12 words → Enter a password → Click "Authenticate". Done!
