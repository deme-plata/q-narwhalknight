/**
 * Simplified Wallet Service for Bounty dApp
 * Handles Ed25519 signing for authentication
 */

import * as ed25519 from '@noble/ed25519';

export interface WalletKeyPair {
  publicKey: Uint8Array;
  privateKey: Uint8Array;
  address: string;
}

/**
 * Generate authentication message for signing
 * Message = "Q-NarwhalKnight Authentication\n\nAddress: {address}\nTimestamp: {timestamp}"
 */
export function generateAuthMessage(address: string): string {
  const timestamp = Date.now();
  return `Q-NarwhalKnight Bounty Authentication\n\nAddress: ${address}\nTimestamp: ${timestamp}\n\nBy signing this message, you authenticate with the Q-NarwhalKnight Bounty Campaign.`;
}

/**
 * Sign authentication message with Ed25519
 */
export async function signMessage(
  message: string,
  privateKey: Uint8Array
): Promise<string> {
  const messageBytes = new TextEncoder().encode(message);
  // Use SHA-256 for hashing (compatible with browser crypto API)
  const hashBuffer = await crypto.subtle.digest('SHA-256', messageBytes);
  const messageHash = new Uint8Array(hashBuffer);
  const signature = await ed25519.sign(messageHash, privateKey);
  return bytesToHex(signature);
}

/**
 * Derive address from public key
 */
export function deriveAddress(publicKey: Uint8Array): string {
  return 'qnk' + bytesToHex(publicKey);
}

/**
 * Generate a new Ed25519 keypair
 */
export async function generateKeyPair(): Promise<WalletKeyPair> {
  const privateKey = crypto.getRandomValues(new Uint8Array(32));
  const publicKey = await ed25519.getPublicKey(privateKey);
  const address = deriveAddress(publicKey);

  return {
    publicKey,
    privateKey,
    address,
  };
}

/**
 * Derive keypair from mnemonic seed phrase
 */
export async function keypairFromMnemonic(mnemonic: string): Promise<WalletKeyPair> {
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  // Use SHA-256 for key derivation
  const hashBuffer = await crypto.subtle.digest('SHA-256', mnemonicBytes);
  const privateKey = new Uint8Array(hashBuffer);
  const publicKey = await ed25519.getPublicKey(privateKey);
  const address = deriveAddress(publicKey);

  return {
    publicKey,
    privateKey,
    address,
  };
}

/**
 * Store wallet temporarily in sessionStorage (for active session only)
 */
export function storeWalletSession(keyPair: WalletKeyPair): void {
  const walletData = {
    privateKey: Array.from(keyPair.privateKey),
    publicKey: Array.from(keyPair.publicKey),
    address: keyPair.address,
    expiresAt: Date.now() + 24 * 60 * 60 * 1000, // 24 hours
  };
  sessionStorage.setItem('bounty_wallet', JSON.stringify(walletData));
}

/**
 * Load wallet from sessionStorage
 */
export function loadWalletSession(): WalletKeyPair | null {
  const stored = sessionStorage.getItem('bounty_wallet');
  if (!stored) return null;

  try {
    const data = JSON.parse(stored);

    // Check if expired
    if (Date.now() > data.expiresAt) {
      clearWalletSession();
      return null;
    }

    return {
      privateKey: new Uint8Array(data.privateKey),
      publicKey: new Uint8Array(data.publicKey),
      address: data.address,
    };
  } catch (error) {
    console.error('Failed to load wallet session:', error);
    return null;
  }
}

/**
 * Clear wallet session
 */
export function clearWalletSession(): void {
  sessionStorage.removeItem('bounty_wallet');
}

/**
 * Check if wallet session exists and is valid
 */
export function hasWalletSession(): boolean {
  return loadWalletSession() !== null;
}

// Helper functions

function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0) {
    throw new Error('Invalid hex string');
  }
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
  }
  return bytes;
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}
