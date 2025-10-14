/**
 * Wallet Authentication Service
 *
 * Provides Ed25519 signature-based authentication for Q-NarwhalKnight wallet APIs.
 * Implements the authentication protocol defined in WALLET_AUTHENTICATION.md
 */

import * as ed25519 from '@noble/ed25519';
import { sha3_256 } from '@noble/hashes/sha3';

export interface AuthHeader {
  address: string;
  timestamp: number;
  scheme: 'Ed25519' | 'Dilithium5' | 'Hybrid' | 'UltraSecure';
  signature?: string;
  dilithium5_signature?: string;
  dilithium5_public_key?: string;
  sphincs_signature?: string;
  sphincs_public_key?: string;
}

export interface WalletKeyPair {
  publicKey: Uint8Array;
  privateKey: Uint8Array;
  address: string; // qnk-prefixed hex address
}

/**
 * Generate authentication challenge
 * Challenge = SHA3-256(address || timestamp || request_path)
 */
export function generateChallenge(
  address: string,
  timestamp: number,
  requestPath: string
): Uint8Array {
  // Remove 'qnk' prefix if present
  const addressHex = address.startsWith('qnk') ? address.substring(3) : address;
  const addressBytes = hexToBytes(addressHex);

  // Convert timestamp to 8-byte little-endian
  const timestampBytes = new Uint8Array(8);
  const view = new DataView(timestampBytes.buffer);
  view.setBigInt64(0, BigInt(timestamp), true); // true = little-endian

  // Convert request path to UTF-8 bytes
  const pathBytes = new TextEncoder().encode(requestPath);

  // Concatenate: address || timestamp || path
  const combined = new Uint8Array(
    addressBytes.length + timestampBytes.length + pathBytes.length
  );
  combined.set(addressBytes, 0);
  combined.set(timestampBytes, addressBytes.length);
  combined.set(pathBytes, addressBytes.length + timestampBytes.length);

  // Hash with SHA3-256
  return sha3_256(combined);
}

/**
 * Sign authentication challenge with Ed25519
 */
export async function signChallenge(
  challenge: Uint8Array,
  privateKey: Uint8Array
): Promise<Uint8Array> {
  return await ed25519.sign(challenge, privateKey);
}

/**
 * Generate complete authentication header for API request
 */
export async function generateAuthHeader(
  privateKey: Uint8Array,
  address: string,
  requestPath: string
): Promise<string> {
  const timestamp = Math.floor(Date.now() / 1000);

  // Generate and sign challenge
  const challenge = generateChallenge(address, timestamp, requestPath);
  const signature = await signChallenge(challenge, privateKey);

  const authHeader: AuthHeader = {
    address,
    timestamp,
    scheme: 'Ed25519',
    signature: bytesToHex(signature),
  };

  return JSON.stringify(authHeader);
}

/**
 * Verify that a public key derives to the expected address
 */
export function deriveAddress(publicKey: Uint8Array): string {
  // Q-NarwhalKnight address = "qnk" + hex(publicKey)
  // For Ed25519, the public key IS the address (32 bytes)
  return 'qnk' + bytesToHex(publicKey);
}

/**
 * Generate a new Ed25519 keypair
 */
export async function generateKeyPair(): Promise<WalletKeyPair> {
  const privateKey = ed25519.utils.randomPrivateKey();
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
  // Hash mnemonic to get private key (same as backend implementation)
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const privateKey = sha3_256(mnemonicBytes);
  const publicKey = await ed25519.getPublicKey(privateKey);
  const address = deriveAddress(publicKey);

  return {
    publicKey,
    privateKey,
    address,
  };
}

/**
 * Encrypt private key with password using WebCrypto API
 */
export async function encryptPrivateKey(
  privateKey: Uint8Array,
  password: string
): Promise<string> {
  // Generate random salt for key derivation
  const salt = crypto.getRandomValues(new Uint8Array(16));

  // Derive encryption key from password using PBKDF2
  const passwordKey = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    'PBKDF2',
    false,
    ['deriveBits']
  );

  const derivedBits = await crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      salt,
      iterations: 100000, // 100K iterations for security
      hash: 'SHA-256',
    },
    passwordKey,
    256 // 256 bits for AES-256
  );

  // Import derived key for AES-GCM encryption
  const encryptionKey = await crypto.subtle.importKey(
    'raw',
    derivedBits,
    'AES-GCM',
    false,
    ['encrypt']
  );

  // Generate random IV for AES-GCM
  const iv = crypto.getRandomValues(new Uint8Array(12));

  // Encrypt private key
  const encryptedData = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    encryptionKey,
    privateKey
  );

  // Return as JSON with salt, iv, and encrypted data
  return JSON.stringify({
    salt: Array.from(salt),
    iv: Array.from(iv),
    data: Array.from(new Uint8Array(encryptedData)),
  });
}

/**
 * Decrypt private key with password
 */
export async function decryptPrivateKey(
  encryptedJson: string,
  password: string
): Promise<Uint8Array> {
  const { salt, iv, data } = JSON.parse(encryptedJson);

  // Derive decryption key from password
  const passwordKey = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    'PBKDF2',
    false,
    ['deriveBits']
  );

  const derivedBits = await crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      salt: new Uint8Array(salt),
      iterations: 100000,
      hash: 'SHA-256',
    },
    passwordKey,
    256
  );

  // Import derived key for decryption
  const decryptionKey = await crypto.subtle.importKey(
    'raw',
    derivedBits,
    'AES-GCM',
    false,
    ['decrypt']
  );

  // Decrypt private key
  const decryptedData = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: new Uint8Array(iv) },
    decryptionKey,
    new Uint8Array(data)
  );

  return new Uint8Array(decryptedData);
}

/**
 * Store encrypted wallet in localStorage
 * Now also encrypts and stores the mnemonic for password-based recovery
 */
export async function storeWallet(
  mnemonic: string,
  password: string
): Promise<WalletKeyPair> {
  const keyPair = await keypairFromMnemonic(mnemonic);
  const encryptedPrivateKey = await encryptPrivateKey(keyPair.privateKey, password);

  // Also encrypt the mnemonic using the same password
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const encryptedMnemonic = await encryptPrivateKey(mnemonicBytes, password);

  // Store encrypted private key, mnemonic, and public address
  localStorage.setItem('walletAddress', keyPair.address);
  localStorage.setItem('walletEncryptedKey', encryptedPrivateKey);
  localStorage.setItem('walletEncryptedMnemonic', encryptedMnemonic);
  localStorage.setItem('walletPublicKey', bytesToHex(keyPair.publicKey));

  // DO NOT store plaintext mnemonic or private key when password is provided
  // Remove any existing plaintext keys (security cleanup)
  localStorage.removeItem('walletSeed'); // Remove old plaintext mnemonic

  return keyPair;
}

/**
 * Load and decrypt wallet from localStorage
 */
export async function loadWallet(password: string): Promise<WalletKeyPair> {
  const address = localStorage.getItem('walletAddress');
  const encryptedKey = localStorage.getItem('walletEncryptedKey');
  const publicKeyHex = localStorage.getItem('walletPublicKey');

  if (!address || !encryptedKey || !publicKeyHex) {
    throw new Error('No wallet found in storage');
  }

  const privateKey = await decryptPrivateKey(encryptedKey, password);
  const publicKey = hexToBytes(publicKeyHex);

  return {
    publicKey,
    privateKey,
    address,
  };
}

/**
 * Decrypt and recover mnemonic from encrypted storage
 * Returns the plaintext mnemonic after successful password verification
 */
export async function recoverMnemonic(password: string): Promise<string> {
  const encryptedMnemonic = localStorage.getItem('walletEncryptedMnemonic');

  if (!encryptedMnemonic) {
    throw new Error('No encrypted mnemonic found. Wallet may not have been created with password protection.');
  }

  try {
    const mnemonicBytes = await decryptPrivateKey(encryptedMnemonic, password);
    const mnemonic = new TextDecoder().decode(mnemonicBytes);
    return mnemonic;
  } catch (error) {
    throw new Error('Failed to decrypt mnemonic. Incorrect password or corrupted data.');
  }
}

/**
 * Check if a wallet exists in storage
 */
export function hasStoredWallet(): boolean {
  return !!(
    localStorage.getItem('walletAddress') &&
    localStorage.getItem('walletEncryptedKey')
  );
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

/**
 * Session-based wallet cache (persisted to sessionStorage)
 * Avoids asking for password on every request within same browser session
 */
class WalletSession {
  private privateKey: Uint8Array | null = null;
  private address: string | null = null;
  private expiresAt: number = 0;
  private sessionCheckInterval: number | null = null;

  constructor() {
    // Try to restore session from sessionStorage on initialization
    this.restoreSession();
    // Start monitoring session expiry
    this.startSessionMonitor();
  }

  /**
   * Restore session from sessionStorage (survives page refresh, not browser close)
   */
  private restoreSession() {
    try {
      const stored = sessionStorage.getItem('walletSession');
      if (stored) {
        const data = JSON.parse(stored);
        // Convert arrays back to Uint8Array
        this.privateKey = new Uint8Array(data.privateKey);
        this.address = data.address;
        this.expiresAt = data.expiresAt;

        // Check if expired
        if (Date.now() > this.expiresAt) {
          this.clearSession();
        }
      }
    } catch (error) {
      console.error('Failed to restore session:', error);
      this.clearSession();
    }
  }

  /**
   * Persist session to sessionStorage
   */
  private persistSession() {
    try {
      if (this.privateKey && this.address) {
        const data = {
          privateKey: Array.from(this.privateKey),
          address: this.address,
          expiresAt: this.expiresAt,
        };
        sessionStorage.setItem('walletSession', JSON.stringify(data));
      }
    } catch (error) {
      console.error('Failed to persist session:', error);
    }
  }

  /**
   * Get session timeout from settings (in minutes)
   * Supports: '5', '15', '30', '60', '240', 'never'
   * Default: never (for user convenience)
   */
  private getTimeoutMinutes(): number | null {
    const setting = localStorage.getItem('walletSessionTimeout') || 'never';
    if (setting === 'never') {
      return null; // Never expire
    }
    return parseInt(setting, 10) || null;
  }

  /**
   * Set wallet session with configurable timeout
   * Timeout is read from localStorage (walletSessionTimeout setting)
   */
  setSession(privateKey: Uint8Array, address: string) {
    this.privateKey = privateKey;
    this.address = address;

    const timeoutMinutes = this.getTimeoutMinutes();
    if (timeoutMinutes === null) {
      // Never expire - set to far future (100 years)
      this.expiresAt = Date.now() + 100 * 365 * 24 * 60 * 60 * 1000;
    } else {
      // Set expiry based on user's preference
      this.expiresAt = Date.now() + timeoutMinutes * 60 * 1000;
    }

    // Persist to sessionStorage
    this.persistSession();
  }

  /**
   * Get wallet session if valid
   */
  getSession(): { privateKey: Uint8Array; address: string } | null {
    if (!this.privateKey || !this.address || Date.now() > this.expiresAt) {
      this.clearSession();
      return null;
    }
    return { privateKey: this.privateKey, address: this.address };
  }

  /**
   * Clear wallet session
   */
  clearSession() {
    this.privateKey = null;
    this.address = null;
    this.expiresAt = 0;

    // Clear from sessionStorage
    try {
      sessionStorage.removeItem('walletSession');
      // Also clear plaintext mnemonic from localStorage for security
      // This forces user to re-enter mnemonic after session timeout
      localStorage.removeItem('walletSeed');
      console.log('🔒 Session expired - cleared wallet session and mnemonic');
      console.log('⚠️ Please log in again to continue using the wallet');
    } catch (error) {
      console.error('Failed to clear session from storage:', error);
    }
  }

  /**
   * Check if session is active
   */
  isActive(): boolean {
    return !!this.getSession();
  }

  /**
   * Get remaining session time in seconds
   */
  getRemainingTime(): number {
    if (!this.privateKey || !this.address) {
      return 0;
    }
    const remaining = Math.max(0, this.expiresAt - Date.now());
    return Math.floor(remaining / 1000);
  }

  /**
   * Refresh session timeout (reset timer)
   */
  refreshSession() {
    if (this.privateKey && this.address) {
      const timeoutMinutes = this.getTimeoutMinutes();
      if (timeoutMinutes === null) {
        // Never expire
        this.expiresAt = Date.now() + 100 * 365 * 24 * 60 * 60 * 1000;
      } else {
        this.expiresAt = Date.now() + timeoutMinutes * 60 * 1000;
      }

      // Persist updated expiry
      this.persistSession();
    }
  }

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

  /**
   * Stop monitoring session expiry
   */
  stopSessionMonitor() {
    if (this.sessionCheckInterval !== null) {
      clearInterval(this.sessionCheckInterval);
      this.sessionCheckInterval = null;
    }
  }
}

export const walletSession = new WalletSession();
