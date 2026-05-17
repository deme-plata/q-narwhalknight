/**
 * Wallet service for key generation, signing, and address derivation.
 *
 * Key derivation: BIP39 mnemonic -> SHA3-256 seed -> Ed25519 keypair
 * Address format: "qnk" + hex(pubkey) (67 chars total)
 *
 * v2.0.0: Added hybrid post-quantum signing (Ed25519 + Dilithium5)
 * Matches: gui/quantum-wallet/src/services/walletAuth.ts
 */

import * as ed from '@noble/ed25519';
import { sha3_256 } from '@noble/hashes/sha3.js';
import { bytesToHex } from '@noble/hashes/utils.js';
import { generateMnemonic as bip39Generate, validateMnemonic as bip39Validate } from '@scure/bip39';
import { wordlist } from '@scure/bip39/wordlists/english.js';
import {
  loadPQCrypto,
  isPQCryptoAvailable,
  dilithium5KeyGenFromMnemonic,
  dilithium5Sign,
  kyber1024KeyGenFromMnemonic,
  getPQBackendType,
  type Dilithium5Keypair,
  type Kyber1024Keypair,
  type SignatureMode,
} from './pqCrypto';

const QNK_PREFIX = 'qnk';

// ============================================================================
// Types
// ============================================================================

export interface WalletKeys {
  /** Ed25519 public key (32 bytes) */
  publicKey: Uint8Array;
  /** QNK address ("qnk" + hex(pubkey)) */
  address: string;
  /** Dilithium5 public key (2,592 bytes) — undefined if PQ unavailable */
  dilithium5PublicKey?: Uint8Array;
  /** Kyber1024 public key (1,568 bytes) — undefined if PQ unavailable */
  kyber1024PublicKey?: Uint8Array;
  /** Which PQ backend is active */
  pqBackendType: string;
}

export interface TransactionSignature {
  /** Hex-encoded Ed25519 signature */
  signature: string;
  /** Hex-encoded Ed25519 public key */
  publicKey: string;
  /** Hex-encoded Dilithium5 signature (4,627 bytes) — included in hybrid mode */
  dilithium5Signature?: string;
  /** Hex-encoded Dilithium5 public key (2,592 bytes) — included in hybrid mode */
  dilithium5PublicKey?: string;
  /** Signature mode used */
  signatureMode: SignatureMode;
}

// ============================================================================
// Classical Key Operations (Ed25519)
// ============================================================================

/**
 * Generate a new BIP39 mnemonic phrase (24 words / 256 bits).
 */
export function generateMnemonic(): string {
  return bip39Generate(wordlist, 128); // 12 words (128 bits)
}

/**
 * Validate a BIP39 mnemonic phrase.
 */
export function validateMnemonicPhrase(mnemonic: string): boolean {
  return bip39Validate(mnemonic, wordlist);
}

/**
 * Derive a QNK address from a mnemonic.
 *
 * CRITICAL: Must match server and desktop wallet derivation:
 *   1. SHA3-256(mnemonic_utf8_bytes) -> 32-byte Ed25519 private key
 *   2. Ed25519 public key from private key
 *   3. Address = "qnk" + hex(pubkey)
 */
export async function deriveAddress(mnemonic: string): Promise<string> {
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const privateKey = sha3_256(mnemonicBytes);
  const publicKey = await ed.getPublicKeyAsync(privateKey);
  privateKey.fill(0);
  return QNK_PREFIX + bytesToHex(publicKey);
}

/**
 * Derive the public key hex from a mnemonic.
 */
export async function derivePublicKey(mnemonic: string): Promise<string> {
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const privateKey = sha3_256(mnemonicBytes);
  const publicKey = await ed.getPublicKeyAsync(privateKey);
  privateKey.fill(0);
  return bytesToHex(publicKey);
}

// ============================================================================
// Post-Quantum Key Operations
// ============================================================================

/**
 * Generate all wallet keys from a mnemonic (Ed25519 + PQ if available).
 *
 * This is the main entry point for wallet creation/import.
 * Generates Ed25519 (always) + Dilithium5 + Kyber1024 (if PQ available).
 */
export async function deriveAllKeys(mnemonic: string): Promise<WalletKeys> {
  // Ed25519 (always available)
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const privateKey = sha3_256(mnemonicBytes);
  const publicKey = await ed.getPublicKeyAsync(privateKey);
  const address = QNK_PREFIX + bytesToHex(publicKey);
  privateKey.fill(0);

  // Attempt PQ key generation
  let dilithium5PublicKey: Uint8Array | undefined;
  let kyber1024PublicKey: Uint8Array | undefined;

  try {
    await loadPQCrypto();

    if (isPQCryptoAvailable()) {
      console.log('[WALLET] Generating Dilithium5 keypair...');
      const dilithiumKeypair = await dilithium5KeyGenFromMnemonic(mnemonic);
      dilithium5PublicKey = dilithiumKeypair.publicKey;
      console.log(`[WALLET] Dilithium5 pk: ${dilithiumKeypair.publicKey.length} bytes`);

      console.log('[WALLET] Generating Kyber1024 keypair...');
      const kyberKeypair = await kyber1024KeyGenFromMnemonic(mnemonic);
      kyber1024PublicKey = kyberKeypair.publicKey;
      console.log(`[WALLET] Kyber1024 pk: ${kyberKeypair.publicKey.length} bytes`);
    }
  } catch (err) {
    console.warn('[WALLET] PQ key generation failed:', err);
  }

  return {
    publicKey,
    address,
    dilithium5PublicKey,
    kyber1024PublicKey,
    pqBackendType: getPQBackendType(),
  };
}

/**
 * Get Dilithium5 keypair from mnemonic for signing.
 * Returns undefined if PQ crypto is unavailable.
 */
export async function getDilithium5Keypair(
  mnemonic: string
): Promise<Dilithium5Keypair | undefined> {
  try {
    await loadPQCrypto();
    if (!isPQCryptoAvailable()) return undefined;
    return await dilithium5KeyGenFromMnemonic(mnemonic);
  } catch {
    return undefined;
  }
}

/**
 * Get Kyber1024 keypair from mnemonic.
 * Returns undefined if PQ crypto is unavailable.
 */
export async function getKyber1024Keypair(
  mnemonic: string
): Promise<Kyber1024Keypair | undefined> {
  try {
    await loadPQCrypto();
    if (!isPQCryptoAvailable()) return undefined;
    return await kyber1024KeyGenFromMnemonic(mnemonic);
  } catch {
    return undefined;
  }
}

// ============================================================================
// Transaction Signing
// ============================================================================

/**
 * Sign a transaction payload with Ed25519 (classical) only.
 * Used when PQ crypto is unavailable or for legacy API endpoints.
 */
export async function signTransaction(
  mnemonic: string,
  payload: Record<string, unknown>
): Promise<TransactionSignature> {
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const privateKey = sha3_256(mnemonicBytes);

  try {
    const message = new TextEncoder().encode(JSON.stringify(payload));
    const signature = await ed.signAsync(message, privateKey);
    const publicKey = await ed.getPublicKeyAsync(privateKey);

    return {
      signature: bytesToHex(signature),
      publicKey: bytesToHex(publicKey),
      signatureMode: 'ed25519',
    };
  } finally {
    privateKey.fill(0);
  }
}

/**
 * Sign a transaction payload with hybrid Ed25519 + Dilithium5.
 *
 * This is the preferred signing method. Falls back to Ed25519-only
 * if PQ crypto is unavailable.
 *
 * Matches: gui/quantum-wallet/src/services/walletAuth.ts:signTransactionForP2P()
 */
export async function signTransactionHybrid(
  mnemonic: string,
  payload: Record<string, unknown>
): Promise<TransactionSignature> {
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const privateKey = sha3_256(mnemonicBytes);

  try {
    const message = new TextEncoder().encode(JSON.stringify(payload));
    const publicKey = await ed.getPublicKeyAsync(privateKey);

    // Ed25519 signature (always)
    const ed25519Sig = await ed.signAsync(message, privateKey);

    const result: TransactionSignature = {
      signature: bytesToHex(ed25519Sig),
      publicKey: bytesToHex(publicKey),
      signatureMode: 'ed25519',
    };

    // Attempt Dilithium5 signature for hybrid mode
    try {
      await loadPQCrypto();
      if (isPQCryptoAvailable()) {
        const dilithiumKeypair = await dilithium5KeyGenFromMnemonic(mnemonic);

        // Sign the SHA3-256 hash of the message (matches web wallet P2P protocol)
        const txHash = sha3_256(message);
        const dilSig = await dilithium5Sign(txHash, dilithiumKeypair.secretKey);

        result.dilithium5Signature = bytesToHex(dilSig);
        result.dilithium5PublicKey = bytesToHex(dilithiumKeypair.publicKey);
        result.signatureMode = 'hybrid';

        console.log(`[WALLET] Hybrid signature: Ed25519(64B) + Dilithium5(${dilSig.length}B)`);

        // Zero PQ secret key
        dilithiumKeypair.secretKey.fill(0);
      }
    } catch (pqErr) {
      console.warn('[WALLET] PQ signing failed, using Ed25519 only:', pqErr);
    }

    return result;
  } finally {
    privateKey.fill(0);
  }
}

/**
 * Sign a P2P transaction using the binary format matching the web wallet.
 *
 * Hash = SHA3-256(from_bytes || to_bytes || amount_le8 || nonce_le4 || timestamp_le8 || memo_bytes)
 *
 * Matches: gui/quantum-wallet/src/services/walletAuth.ts:signTransactionForP2P()
 */
export async function signP2PTransaction(
  mnemonic: string,
  params: {
    from: string;
    to: string;
    amount: number;
    nonce: number;
    memo?: string;
  }
): Promise<TransactionSignature & { nonce: number; timestamp: number; amountAtomic: bigint }> {
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const privateKey = sha3_256(mnemonicBytes);

  try {
    const publicKey = await ed.getPublicKeyAsync(privateKey);
    const timestamp = Math.floor(Date.now() / 1000);

    // Amount in 9-decimal atomic units (matches P2P protocol i64 range)
    const QUG_DECIMALS = 1_000_000_000n;
    const amountAtomic = BigInt(Math.floor(params.amount * Number(QUG_DECIMALS)));

    // Build binary transaction hash (matches web wallet)
    const fromHex = params.from.startsWith('qnk') ? params.from.substring(3) : params.from;
    const toHex = params.to.startsWith('qnk') ? params.to.substring(3) : params.to;
    const fromBytes = hexToBytes(fromHex);
    const toBytes = hexToBytes(toHex);

    const amountBytes = new Uint8Array(8);
    new DataView(amountBytes.buffer).setBigInt64(0, amountAtomic, true);

    const nonceBytes = new Uint8Array(4);
    new DataView(nonceBytes.buffer).setUint32(0, params.nonce, true);

    const timestampBytes = new Uint8Array(8);
    new DataView(timestampBytes.buffer).setBigInt64(0, BigInt(timestamp), true);

    const memoBytes = params.memo ? new TextEncoder().encode(params.memo) : new Uint8Array(0);

    // Concatenate all fields
    const totalLen = fromBytes.length + toBytes.length + amountBytes.length
      + nonceBytes.length + timestampBytes.length + memoBytes.length;
    const combined = new Uint8Array(totalLen);
    let offset = 0;
    combined.set(fromBytes, offset); offset += fromBytes.length;
    combined.set(toBytes, offset); offset += toBytes.length;
    combined.set(amountBytes, offset); offset += amountBytes.length;
    combined.set(nonceBytes, offset); offset += nonceBytes.length;
    combined.set(timestampBytes, offset); offset += timestampBytes.length;
    combined.set(memoBytes, offset);

    const txHash = sha3_256(combined);

    // Ed25519 signature
    const ed25519Sig = await ed.signAsync(txHash, privateKey);

    const result: TransactionSignature & { nonce: number; timestamp: number; amountAtomic: bigint } = {
      signature: bytesToHex(ed25519Sig),
      publicKey: bytesToHex(publicKey),
      signatureMode: 'ed25519',
      nonce: params.nonce,
      timestamp,
      amountAtomic,
    };

    // Attempt Dilithium5 hybrid signature
    try {
      await loadPQCrypto();
      if (isPQCryptoAvailable()) {
        const dilKeypair = await dilithium5KeyGenFromMnemonic(mnemonic);
        const dilSig = await dilithium5Sign(txHash, dilKeypair.secretKey);

        result.dilithium5Signature = bytesToHex(dilSig);
        result.dilithium5PublicKey = bytesToHex(dilKeypair.publicKey);
        result.signatureMode = 'hybrid';

        dilKeypair.secretKey.fill(0);
      }
    } catch {
      // Ed25519-only fallback
    }

    return result;
  } finally {
    privateKey.fill(0);
  }
}

// ============================================================================
// Auth Challenge
// ============================================================================

/**
 * Create a signed auth challenge for server login.
 * Uses hybrid signing when PQ crypto is available.
 */
export async function createAuthChallenge(mnemonic: string): Promise<{
  wallet_address: string;
  signature: string;
  public_key: string;
  timestamp: number;
  dilithium5_signature?: string;
  dilithium5_public_key?: string;
  signature_mode: SignatureMode;
}> {
  const timestamp = Math.floor(Date.now() / 1000);
  const address = await deriveAddress(mnemonic);

  const payload = { action: 'auth', address, timestamp };
  const { signature, publicKey, dilithium5Signature, dilithium5PublicKey, signatureMode } =
    await signTransactionHybrid(mnemonic, payload);

  return {
    wallet_address: address,
    signature,
    public_key: publicKey,
    timestamp,
    dilithium5_signature: dilithium5Signature,
    dilithium5_public_key: dilithium5PublicKey,
    signature_mode: signatureMode,
  };
}

// ============================================================================
// Signature Verification
// ============================================================================

/**
 * Verify an Ed25519 signature.
 */
export async function verifySignature(
  message: Uint8Array,
  signature: Uint8Array,
  publicKey: Uint8Array
): Promise<boolean> {
  return ed.verifyAsync(signature, message, publicKey);
}

// ============================================================================
// Internal Helpers
// ============================================================================

function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return bytes;
}
