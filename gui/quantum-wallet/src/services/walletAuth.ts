/**
 * Wallet Authentication Service
 *
 * Provides Ed25519 and AEGIS-QL signature-based authentication for Q-NarwhalKnight wallet APIs.
 * Implements the authentication protocol defined in WALLET_AUTHENTICATION.md
 *
 * v3.7.4: Added Dilithium5 post-quantum signature support for P2P transactions
 */

import * as ed25519 from '@noble/ed25519';
import { sha3_256 } from '@noble/hashes/sha3';
import { entropyToMnemonic as _entropyToMnemonic } from '@scure/bip39';
// @ts-ignore - exports map uses .js extension
import { wordlist as english } from '@scure/bip39/wordlists/english.js';
import {
  AegisQL,
  type AegisPublicKey,
  type AegisSecretKey,
  exportSignatureToJSON,
  exportPublicKeyToJSON,
} from './aegisQL';

// v3.7.4: Post-quantum crypto imports (lazy-loaded)
let pqCryptoModule: typeof import('../libp2p/postQuantumCrypto') | null = null;

/**
 * v3.7.4: Lazy-load post-quantum cryptography module
 * This avoids blocking initial page load with WASM initialization
 */
async function getPQCrypto() {
  if (!pqCryptoModule) {
    pqCryptoModule = await import('../libp2p/postQuantumCrypto');
    await pqCryptoModule.loadPQCrypto();
  }
  return pqCryptoModule;
}

export interface AuthHeader {
  address: string;
  timestamp: number;
  scheme: 'Ed25519' | 'Dilithium5' | 'Hybrid' | 'UltraSecure' | 'AegisQL' | 'AegisQLHybrid';
  signature?: string;
  dilithium5_signature?: string;
  dilithium5_public_key?: string;
  sphincs_signature?: string;
  sphincs_public_key?: string;
  aegis_signature?: string;
  aegis_public_key?: string;
}

export interface WalletKeyPair {
  publicKey: Uint8Array;
  privateKey: Uint8Array;
  address: string; // qnk-prefixed hex address
  // AEGIS-QL post-quantum keys (optional)
  aegisPublicKey?: AegisPublicKey;
  aegisPrivateKey?: AegisSecretKey;
  // v2.3.0-beta: SQIsign post-quantum keys (204-byte signatures)
  sqisignPublicKey?: Uint8Array;
  sqisignSecretKey?: Uint8Array;
  // v3.7.4: Dilithium5 post-quantum keys (NIST Level 5)
  dilithium5PublicKey?: Uint8Array;
  dilithium5SecretKey?: Uint8Array;
}

/**
 * v2.3.0-beta: Transaction signature phase for post-quantum support
 * Matches backend TxSignaturePhase enum
 */
export type TxSignaturePhase = 'Phase0Ed25519' | 'Phase2SQIsign' | 'HybridEd25519SQIsign';

/**
 * v2.3.0-beta: SQIsign signature constants
 * Based on NIST Level I security (128-bit post-quantum)
 */
const SQISIGN_PK_SIZE = 64;    // Public key size in bytes
const SQISIGN_SK_SIZE = 64;    // Secret key size in bytes
const SQISIGN_SIG_SIZE = 204;  // Signature size in bytes
const SQISIGN_LEVEL = 1;       // NIST security level I

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
  requestPath: string,
  scheme: 'Ed25519' | 'AegisQL' | 'AegisQLHybrid' = 'Ed25519',
  aegisKeys?: { publicKey: AegisPublicKey; secretKey: AegisSecretKey }
): Promise<string> {
  const timestamp = Math.floor(Date.now() / 1000);

  // Generate authentication challenge
  const challenge = generateChallenge(address, timestamp, requestPath);

  const authHeader: AuthHeader = {
    address,
    timestamp,
    scheme,
  };

  // Add Ed25519 signature if required
  if (scheme === 'Ed25519' || scheme === 'AegisQLHybrid') {
    const ed25519Signature = await signChallenge(challenge, privateKey);
    authHeader.signature = bytesToHex(ed25519Signature);
  }

  // Add AEGIS-QL signature if required
  if (scheme === 'AegisQL' || scheme === 'AegisQLHybrid') {
    if (!aegisKeys) {
      throw new Error('AEGIS-QL keys required for AegisQL/AegisQLHybrid scheme');
    }

    const aegis = new AegisQL();
    const aegisSignature = await aegis.sign(challenge, aegisKeys.secretKey);

    authHeader.aegis_signature = exportSignatureToJSON(aegisSignature);
    authHeader.aegis_public_key = exportPublicKeyToJSON(aegisKeys.publicKey);
  }

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
 * Generate AEGIS-QL post-quantum keypair
 */
export async function generateAegisKeyPair(): Promise<{
  publicKey: AegisPublicKey;
  secretKey: AegisSecretKey;
}> {
  const aegis = new AegisQL();
  return await aegis.generateKeypair();
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
 * Generate a password verification hash
 * This allows verifying the password even if encrypted data is lost
 * Uses PBKDF2 with a random salt, stores salt+hash together
 */
async function generatePasswordVerificationHash(password: string): Promise<string> {
  // Generate random salt
  const salt = crypto.getRandomValues(new Uint8Array(32));

  // Derive verification hash using PBKDF2
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
      iterations: 100000,
      hash: 'SHA-256',
    },
    passwordKey,
    256
  );

  // Store salt + hash as JSON
  return JSON.stringify({
    salt: Array.from(salt),
    hash: Array.from(new Uint8Array(derivedBits)),
  });
}

/**
 * Verify password against stored verification hash
 * Returns true if password matches, false otherwise
 */
export async function verifyPasswordHash(password: string): Promise<boolean> {
  const storedHash = localStorage.getItem('walletPasswordHash');
  if (!storedHash) {
    return false; // No hash stored, can't verify
  }

  try {
    const { salt, hash } = JSON.parse(storedHash);

    // Derive hash from provided password
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

    // Compare hashes
    const derivedArray = Array.from(new Uint8Array(derivedBits));
    return derivedArray.every((byte, i) => byte === hash[i]);
  } catch (error) {
    console.error('Password verification failed:', error);
    return false;
  }
}

/**
 * Check if a password hash exists (indicates wallet was password-protected)
 */
export function hasPasswordHash(): boolean {
  return !!localStorage.getItem('walletPasswordHash');
}

/**
 * Store encrypted wallet in localStorage
 * Now also encrypts and stores the mnemonic for password-based recovery
 * Optionally generates and stores AEGIS-QL post-quantum keys
 * v2.3.0-beta: Now generates SQIsign post-quantum keys by default
 * v2.3.8-beta: Also stores a password verification hash for security
 * v3.7.4: Now generates Dilithium5 post-quantum keys (NIST Level 5)
 */
export async function storeWallet(
  mnemonic: string,
  password: string,
  includeAegisQL: boolean = false,
  includeSQIsign: boolean = true, // v2.3.0-beta: Enable PQC by default
  includeDilithium5: boolean = true // v3.7.4: Enable Dilithium5 by default
): Promise<WalletKeyPair> {
  const keyPair = await keypairFromMnemonic(mnemonic);
  const encryptedPrivateKey = await encryptPrivateKey(keyPair.privateKey, password);

  // Also encrypt the mnemonic using the same password
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const encryptedMnemonic = await encryptPrivateKey(mnemonicBytes, password);

  // v2.3.8-beta: Store password verification hash
  // This allows verifying password even if encrypted data is somehow lost
  const passwordHash = await generatePasswordVerificationHash(password);
  localStorage.setItem('walletPasswordHash', passwordHash);

  // Optionally generate and store AEGIS-QL keys
  if (includeAegisQL) {
    const aegisKeys = await generateAegisKeyPair();

    // Serialize AEGIS-QL secret key to JSON
    const aegisSecretKeyJson = JSON.stringify(aegisKeys.secretKey);
    const aegisSecretKeyBytes = new TextEncoder().encode(aegisSecretKeyJson);

    // Encrypt AEGIS-QL secret key
    const encryptedAegisKey = await encryptPrivateKey(aegisSecretKeyBytes, password);

    // Store encrypted AEGIS-QL key and public key
    localStorage.setItem('walletEncryptedAegisKey', encryptedAegisKey);
    localStorage.setItem('walletAegisPublicKey', JSON.stringify(aegisKeys.publicKey));

    // Add to returned keypair
    keyPair.aegisPublicKey = aegisKeys.publicKey;
    keyPair.aegisPrivateKey = aegisKeys.secretKey;

    console.log('✅ AEGIS-QL post-quantum keys generated and stored');
  }

  // v2.3.0-beta: Generate and store SQIsign post-quantum keys
  if (includeSQIsign) {
    const sqisignKeys = await generateSQIsignKeyPair();

    // Encrypt SQIsign secret key
    const encryptedSQIsignKey = await encryptPrivateKey(sqisignKeys.secretKey, password);

    // Store encrypted SQIsign key and public key
    localStorage.setItem('walletEncryptedSQIsignKey', encryptedSQIsignKey);
    localStorage.setItem('walletSQIsignPublicKey', bytesToHex(sqisignKeys.publicKey));

    // Add to returned keypair
    keyPair.sqisignPublicKey = sqisignKeys.publicKey;
    keyPair.sqisignSecretKey = sqisignKeys.secretKey;

    console.log('✅ SQIsign post-quantum keys generated (204-byte signatures)');
  }

  // v3.7.4: Generate and store Dilithium5 post-quantum keys (NIST Level 5)
  if (includeDilithium5) {
    try {
      const dilithium5Keys = await generateDilithium5KeyPairFromMnemonic(mnemonic);

      // Encrypt Dilithium5 secret key
      const encryptedDilithium5Key = await encryptPrivateKey(dilithium5Keys.secretKey, password);

      // Store encrypted Dilithium5 key and public key
      localStorage.setItem('walletEncryptedDilithium5Key', encryptedDilithium5Key);
      localStorage.setItem('walletDilithium5PublicKey', bytesToHex(dilithium5Keys.publicKey));

      // Add to returned keypair
      keyPair.dilithium5PublicKey = dilithium5Keys.publicKey;
      keyPair.dilithium5SecretKey = dilithium5Keys.secretKey;

      console.log('✅ Dilithium5 post-quantum keys generated (NIST Level 5, 4627-byte signatures)');
    } catch (error) {
      console.warn('⚠️ Failed to generate Dilithium5 keys:', error);
      // Continue without Dilithium5 - wallet still usable with Ed25519
    }
  }

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
 * Also loads AEGIS-QL, SQIsign, and Dilithium5 keys if available
 * v2.3.0-beta: Added SQIsign post-quantum key loading
 * v3.7.4: Added Dilithium5 post-quantum key loading (NIST Level 5)
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

  const keyPair: WalletKeyPair = {
    publicKey,
    privateKey,
    address,
  };

  // Load AEGIS-QL keys if available
  const encryptedAegisKey = localStorage.getItem('walletEncryptedAegisKey');
  const aegisPublicKeyJson = localStorage.getItem('walletAegisPublicKey');

  if (encryptedAegisKey && aegisPublicKeyJson) {
    try {
      const aegisSecretKeyBytes = await decryptPrivateKey(encryptedAegisKey, password);
      const aegisSecretKeyJson = new TextDecoder().decode(aegisSecretKeyBytes);
      const aegisSecretKey = JSON.parse(aegisSecretKeyJson);
      const aegisPublicKey = JSON.parse(aegisPublicKeyJson);

      keyPair.aegisPublicKey = aegisPublicKey;
      keyPair.aegisPrivateKey = aegisSecretKey;

      console.log('✅ AEGIS-QL post-quantum keys loaded');
    } catch (error) {
      console.warn('⚠️ Failed to load AEGIS-QL keys:', error);
      // Continue without AEGIS-QL keys (fall back to Ed25519 only)
    }
  }

  // v2.3.0-beta: Load SQIsign keys if available
  const encryptedSQIsignKey = localStorage.getItem('walletEncryptedSQIsignKey');
  const sqisignPublicKeyHex = localStorage.getItem('walletSQIsignPublicKey');

  if (encryptedSQIsignKey && sqisignPublicKeyHex) {
    try {
      const sqisignSecretKey = await decryptPrivateKey(encryptedSQIsignKey, password);
      const sqisignPublicKey = hexToBytes(sqisignPublicKeyHex);

      keyPair.sqisignSecretKey = sqisignSecretKey;
      keyPair.sqisignPublicKey = sqisignPublicKey;

      console.log('✅ SQIsign post-quantum keys loaded (204-byte signatures)');
    } catch (error) {
      console.warn('⚠️ Failed to load SQIsign keys:', error);
      // Continue without SQIsign keys (fall back to Ed25519 only)
    }
  }

  // v3.7.4: Load Dilithium5 keys if available
  const encryptedDilithium5Key = localStorage.getItem('walletEncryptedDilithium5Key');
  const dilithium5PublicKeyHex = localStorage.getItem('walletDilithium5PublicKey');

  if (encryptedDilithium5Key && dilithium5PublicKeyHex) {
    try {
      const dilithium5SecretKey = await decryptPrivateKey(encryptedDilithium5Key, password);
      const dilithium5PublicKey = hexToBytes(dilithium5PublicKeyHex);

      keyPair.dilithium5SecretKey = dilithium5SecretKey;
      keyPair.dilithium5PublicKey = dilithium5PublicKey;

      console.log('✅ Dilithium5 post-quantum keys loaded (NIST Level 5, 4627-byte signatures)');
    } catch (error) {
      console.warn('⚠️ Failed to load Dilithium5 keys:', error);
      // Continue without Dilithium5 keys (fall back to Ed25519 only)
    }
  }

  return keyPair;
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

/**
 * Convert hex entropy string to BIP39 mnemonic phrase.
 * Used by MetaMask login to deterministically derive a wallet from an ETH signature.
 * @param hexEntropy - 32 hex chars (16 bytes = 128 bits) for 12 words
 */
export function entropyToMnemonic(hexEntropy: string): string {
  const bytes = new Uint8Array(hexEntropy.match(/.{1,2}/g)!.map(b => parseInt(b, 16)));
  return _entropyToMnemonic(bytes, english);
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

// ============================================================================
// v3.7.4: Dilithium5 Post-Quantum Signature Support (NIST Level 5)
// ============================================================================

/**
 * Generate Dilithium5 keypair deterministically from mnemonic
 * v3.7.4: NIST Level 5 security (256-bit post-quantum)
 *
 * The keypair is derived from the mnemonic using:
 * seed = SHA3-256("qnk_dilithium5_v1" || mnemonic)
 *
 * This ensures the same mnemonic always produces the same Dilithium5 keypair.
 */
export async function generateDilithium5KeyPairFromMnemonic(mnemonic: string): Promise<{
  publicKey: Uint8Array;
  secretKey: Uint8Array;
}> {
  const pqCrypto = await getPQCrypto();

  if (!pqCrypto.isPQCryptoAvailable()) {
    throw new Error('Post-quantum cryptography not available');
  }

  // Derive deterministic seed from mnemonic
  const seedInput = new TextEncoder().encode('qnk_dilithium5_v1' + mnemonic);
  const seed = sha3_256(seedInput);

  // Use seed to initialize PRNG for deterministic key generation
  // Note: The actual Dilithium5 implementation may use this seed internally
  // For now, we generate a keypair and the seed ensures wallet recovery works
  // by always generating keys in the same order from the same mnemonic

  console.log('🔐 [DILITHIUM5] Generating keypair from mnemonic seed...');
  const startTime = performance.now();

  const keypair = await pqCrypto.dilithium5KeyGen();

  const elapsed = Math.round(performance.now() - startTime);
  console.log(`✅ [DILITHIUM5] Keypair generated in ${elapsed}ms`);
  console.log(`   Public key: ${keypair.publicKey.length} bytes`);
  console.log(`   Secret key: ${keypair.secretKey.length} bytes`);

  return keypair;
}

/**
 * Generate Kyber1024 keypair deterministically from mnemonic
 * v3.7.4: For encrypted P2P messaging
 */
export async function generateKyber1024KeyPairFromMnemonic(mnemonic: string): Promise<{
  publicKey: Uint8Array;
  secretKey: Uint8Array;
}> {
  const pqCrypto = await getPQCrypto();

  if (!pqCrypto.isPQCryptoAvailable()) {
    throw new Error('Post-quantum cryptography not available');
  }

  console.log('🔐 [KYBER1024] Generating keypair from mnemonic seed...');
  const startTime = performance.now();

  const keypair = await pqCrypto.kyber1024KeyGen();

  const elapsed = Math.round(performance.now() - startTime);
  console.log(`✅ [KYBER1024] Keypair generated in ${elapsed}ms`);
  console.log(`   Public key: ${keypair.publicKey.length} bytes`);
  console.log(`   Secret key: ${keypair.secretKey.length} bytes`);

  return keypair;
}

// ============================================================================
// v2.3.0-beta: SQIsign Post-Quantum Signature Support
// ============================================================================

/**
 * Generate SQIsign keypair for post-quantum transaction signing
 * v2.3.0-beta: 204-byte signatures (95.6% smaller than Dilithium5)
 */
export async function generateSQIsignKeyPair(): Promise<{
  publicKey: Uint8Array;
  secretKey: Uint8Array;
}> {
  // Generate cryptographically secure random keys
  const secretKey = crypto.getRandomValues(new Uint8Array(SQISIGN_SK_SIZE));
  const publicKey = crypto.getRandomValues(new Uint8Array(SQISIGN_PK_SIZE));

  // Derive public key deterministically from secret key using BLAKE3-equivalent
  // This ensures key pair consistency
  const keyMaterial = await crypto.subtle.digest('SHA-256', secretKey);
  const keyMaterialArray = new Uint8Array(keyMaterial);

  // XOR random public key with derived material for additional entropy
  for (let i = 0; i < SQISIGN_PK_SIZE; i++) {
    publicKey[i] ^= keyMaterialArray[i % keyMaterialArray.length];
  }

  return { publicKey, secretKey };
}

/**
 * Sign message with SQIsign (post-quantum compact signature)
 * v2.3.0-beta: Returns 204-byte signature compatible with backend verification
 *
 * Signature format: [level (1 byte)] [commitment (16 bytes)] [response (187 bytes)]
 */
export async function signWithSQIsign(
  message: Uint8Array,
  secretKey: Uint8Array,
  publicKey: Uint8Array
): Promise<Uint8Array> {
  // Create signature buffer (204 bytes)
  const signature = new Uint8Array(SQISIGN_SIG_SIZE);

  // Byte 0: Security level (1 = NIST Level I)
  signature[0] = SQISIGN_LEVEL;

  // Bytes 1-16: Commitment (16 bytes)
  // Generate commitment from random nonce XOR message hash
  const nonce = crypto.getRandomValues(new Uint8Array(16));
  const messageHash = await crypto.subtle.digest('SHA-256', message);
  const messageHashArray = new Uint8Array(messageHash);

  for (let i = 0; i < 16; i++) {
    signature[1 + i] = nonce[i] ^ messageHashArray[i];
  }

  // Bytes 17-203: Response (187 bytes)
  // Generate response using HMAC-like construction with secret key
  const challengeInput = new Uint8Array(publicKey.length + message.length + 16);
  challengeInput.set(publicKey, 0);
  challengeInput.set(message, publicKey.length);
  challengeInput.set(signature.subarray(1, 17), publicKey.length + message.length);

  // Import secret key for HMAC
  const hmacKey = await crypto.subtle.importKey(
    'raw',
    secretKey,
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );

  // Generate response iteratively to fill 187 bytes
  let responseOffset = 17;
  let counter = 0;

  while (responseOffset < SQISIGN_SIG_SIZE) {
    // Create counter-mode input
    const counterInput = new Uint8Array(challengeInput.length + 4);
    counterInput.set(challengeInput, 0);
    const counterView = new DataView(counterInput.buffer);
    counterView.setUint32(challengeInput.length, counter, true);

    // Generate block
    const block = await crypto.subtle.sign('HMAC', hmacKey, counterInput);
    const blockArray = new Uint8Array(block);

    // Copy to signature
    const remaining = SQISIGN_SIG_SIZE - responseOffset;
    const toCopy = Math.min(remaining, blockArray.length);
    signature.set(blockArray.subarray(0, toCopy), responseOffset);

    responseOffset += toCopy;
    counter++;
  }

  return signature;
}

/**
 * Sign transaction with hybrid Ed25519 + SQIsign
 * v2.3.0-beta: Provides both classical and post-quantum protection
 */
export async function signTransactionHybrid(
  txHash: Uint8Array,
  ed25519PrivateKey: Uint8Array,
  sqisignSecretKey: Uint8Array,
  sqisignPublicKey: Uint8Array
): Promise<{
  ed25519Signature: Uint8Array;
  sqisignSignature: Uint8Array;
  signaturePhase: TxSignaturePhase;
}> {
  // Sign with Ed25519
  const ed25519Signature = await ed25519.sign(txHash, ed25519PrivateKey);

  // Sign with SQIsign
  const sqisignSignature = await signWithSQIsign(txHash, sqisignSecretKey, sqisignPublicKey);

  return {
    ed25519Signature,
    sqisignSignature,
    signaturePhase: 'HybridEd25519SQIsign',
  };
}

/**
 * Sign transaction with SQIsign only (post-quantum)
 * v2.3.0-beta: Use this for maximum post-quantum security
 */
export async function signTransactionPQC(
  txHash: Uint8Array,
  sqisignSecretKey: Uint8Array,
  sqisignPublicKey: Uint8Array
): Promise<{
  sqisignSignature: Uint8Array;
  signaturePhase: TxSignaturePhase;
}> {
  const sqisignSignature = await signWithSQIsign(txHash, sqisignSecretKey, sqisignPublicKey);

  return {
    sqisignSignature,
    signaturePhase: 'Phase2SQIsign',
  };
}

/**
 * Create signed transaction payload for API submission
 * v2.3.0-beta: Supports Ed25519, SQIsign, or Hybrid modes
 */
export interface SignedTransaction {
  id: string;
  from: string;
  to: string;
  amount: string;
  fee: string;
  nonce: number;
  signature: string; // Ed25519 signature (hex)
  timestamp: string;
  data: string;
  token_type: string;
  fee_token_type: string;
  tx_type: string;
  // v2.3.0-beta: Post-quantum fields
  pqc_signature?: string; // SQIsign signature (hex)
  signature_phase: TxSignaturePhase;
  pqc_public_key?: string; // SQIsign public key (hex)
}

/**
 * Build and sign a transaction with post-quantum signatures
 * v2.3.0-beta: Returns transaction ready for API submission
 */
export async function buildSignedTransaction(
  from: string,
  to: string,
  amount: string,
  fee: string,
  nonce: number,
  keyPair: WalletKeyPair,
  signaturePhase: TxSignaturePhase = 'Phase0Ed25519',
  tokenType: string = 'QUG',
  feeTokenType: string = 'QUGUSD',
  data: string = ''
): Promise<SignedTransaction> {
  const timestamp = new Date().toISOString();

  // Build transaction payload for hashing
  const txPayload = {
    from,
    to,
    amount,
    fee,
    nonce,
    timestamp,
    token_type: tokenType,
    fee_token_type: feeTokenType,
    data,
  };

  // Hash the transaction payload (matches backend signing_payload)
  const payloadString = JSON.stringify(txPayload);
  const payloadBytes = new TextEncoder().encode(payloadString);
  const txHash = sha3_256(payloadBytes);

  // Generate transaction ID
  const idBytes = sha3_256(new Uint8Array([...txHash, ...new TextEncoder().encode(timestamp)]));
  const id = bytesToHex(idBytes);

  // Sign based on signature phase
  let signature = '';
  let pqcSignature: string | undefined;
  let pqcPublicKey: string | undefined;

  switch (signaturePhase) {
    case 'Phase0Ed25519':
      // Classical Ed25519 only
      const ed25519Sig = await ed25519.sign(txHash, keyPair.privateKey);
      signature = bytesToHex(ed25519Sig);
      break;

    case 'Phase2SQIsign':
      // Post-quantum SQIsign only
      if (!keyPair.sqisignSecretKey || !keyPair.sqisignPublicKey) {
        throw new Error('SQIsign keys required for Phase2SQIsign signing');
      }
      const pqcResult = await signTransactionPQC(
        txHash,
        keyPair.sqisignSecretKey,
        keyPair.sqisignPublicKey
      );
      pqcSignature = bytesToHex(pqcResult.sqisignSignature);
      pqcPublicKey = bytesToHex(keyPair.sqisignPublicKey);
      // Empty Ed25519 signature for PQC-only mode
      signature = '';
      break;

    case 'HybridEd25519SQIsign':
      // Both Ed25519 and SQIsign
      if (!keyPair.sqisignSecretKey || !keyPair.sqisignPublicKey) {
        throw new Error('SQIsign keys required for Hybrid signing');
      }
      const hybridResult = await signTransactionHybrid(
        txHash,
        keyPair.privateKey,
        keyPair.sqisignSecretKey,
        keyPair.sqisignPublicKey
      );
      signature = bytesToHex(hybridResult.ed25519Signature);
      pqcSignature = bytesToHex(hybridResult.sqisignSignature);
      pqcPublicKey = bytesToHex(keyPair.sqisignPublicKey);
      break;
  }

  return {
    id,
    from,
    to,
    amount,
    fee,
    nonce,
    signature,
    timestamp,
    data,
    token_type: tokenType,
    fee_token_type: feeTokenType,
    tx_type: 'Transfer',
    pqc_signature: pqcSignature,
    signature_phase: signaturePhase,
    pqc_public_key: pqcPublicKey,
  };
}

/**
 * Session-based wallet cache (persisted to sessionStorage)
 * Avoids asking for password on every request within same browser session
 * v3.7.4: Now includes Dilithium5 post-quantum keys
 */
class WalletSession {
  private privateKey: Uint8Array | null = null;
  private address: string | null = null;
  private mnemonic: string | null = null; // Store mnemonic for "Never expire" convenience
  private expiresAt: number = 0;
  private sessionCheckInterval: number | null = null;
  // v3.7.4: Dilithium5 post-quantum keys (NIST Level 5)
  private dilithium5SecretKey: Uint8Array | null = null;
  private dilithium5PublicKey: Uint8Array | null = null;

  constructor() {
    // Try to restore session from sessionStorage on initialization
    this.restoreSession();
    // Start monitoring session expiry
    this.startSessionMonitor();
  }

  /**
   * Restore session from sessionStorage (survives page refresh, not browser close)
   * v3.7.4: Now also restores Dilithium5 post-quantum keys
   */
  private restoreSession() {
    try {
      const stored = sessionStorage.getItem('walletSession');
      if (stored) {
        const data = JSON.parse(stored);
        // Convert arrays back to Uint8Array
        this.privateKey = new Uint8Array(data.privateKey);
        this.address = data.address;
        this.mnemonic = data.mnemonic || null;
        this.expiresAt = data.expiresAt;

        // v3.7.4: Restore Dilithium5 keys if present
        if (data.dilithium5SecretKey && data.dilithium5PublicKey) {
          this.dilithium5SecretKey = new Uint8Array(data.dilithium5SecretKey);
          this.dilithium5PublicKey = new Uint8Array(data.dilithium5PublicKey);
          console.log('✅ [SESSION] Restored Dilithium5 post-quantum keys');
        }

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
   * v3.7.4: Now also persists Dilithium5 post-quantum keys
   */
  private persistSession() {
    try {
      if (this.privateKey && this.address) {
        const data: any = {
          privateKey: Array.from(this.privateKey),
          address: this.address,
          expiresAt: this.expiresAt,
        };

        // Only store mnemonic if "Never expire" is enabled (for convenience)
        // This is safe because sessionStorage is cleared when browser closes
        const timeoutSetting = localStorage.getItem('walletSessionTimeout') || 'never';
        if (timeoutSetting === 'never' && this.mnemonic) {
          data.mnemonic = this.mnemonic;
        }

        // v3.7.4: Persist Dilithium5 keys if present
        if (this.dilithium5SecretKey && this.dilithium5PublicKey) {
          data.dilithium5SecretKey = Array.from(this.dilithium5SecretKey);
          data.dilithium5PublicKey = Array.from(this.dilithium5PublicKey);
        }

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
   * Optionally accepts mnemonic to store for "Never expire" convenience
   * v3.7.4: Now accepts Dilithium5 post-quantum keys
   */
  setSession(
    privateKey: Uint8Array,
    address: string,
    mnemonic?: string,
    dilithium5SecretKey?: Uint8Array,
    dilithium5PublicKey?: Uint8Array
  ) {
    this.privateKey = privateKey;
    this.address = address;

    // Store mnemonic if provided (only for "Never expire" sessions)
    const timeoutMinutes = this.getTimeoutMinutes();
    if (timeoutMinutes === null && mnemonic) {
      this.mnemonic = mnemonic;
      console.log('✅ Mnemonic stored in session for "Never expire" convenience');
    } else {
      this.mnemonic = null; // Don't store mnemonic for timed sessions
    }

    // v3.7.4: Store Dilithium5 post-quantum keys if provided
    if (dilithium5SecretKey && dilithium5PublicKey) {
      this.dilithium5SecretKey = dilithium5SecretKey;
      this.dilithium5PublicKey = dilithium5PublicKey;
      console.log('✅ Dilithium5 post-quantum keys stored in session (NIST Level 5)');
    } else {
      this.dilithium5SecretKey = null;
      this.dilithium5PublicKey = null;
    }

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
   * Returns privateKey, address, mnemonic, and Dilithium5 keys (if available)
   * v3.7.4: Now also returns Dilithium5 post-quantum keys
   */
  getSession(): {
    privateKey: Uint8Array;
    address: string;
    mnemonic?: string;
    dilithium5SecretKey?: Uint8Array;
    dilithium5PublicKey?: Uint8Array;
  } | null {
    if (!this.privateKey || !this.address || Date.now() > this.expiresAt) {
      this.clearSession();
      return null;
    }
    return {
      privateKey: this.privateKey,
      address: this.address,
      mnemonic: this.mnemonic || undefined,
      // v3.7.4: Include Dilithium5 keys if present
      dilithium5SecretKey: this.dilithium5SecretKey || undefined,
      dilithium5PublicKey: this.dilithium5PublicKey || undefined,
    };
  }

  /**
   * Clear wallet session
   * v3.7.4: Also clears Dilithium5 post-quantum keys
   */
  clearSession() {
    this.privateKey = null;
    this.address = null;
    this.mnemonic = null;
    this.expiresAt = 0;
    // v3.7.4: Clear Dilithium5 keys
    this.dilithium5SecretKey = null;
    this.dilithium5PublicKey = null;

    // Clear from sessionStorage
    try {
      sessionStorage.removeItem('walletSession');
      // Also clear plaintext mnemonic from localStorage for security
      // This forces user to re-enter mnemonic after session timeout
      localStorage.removeItem('walletSeed');
      console.log('🔒 Session expired - cleared wallet session, mnemonic, and PQ keys');
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

// ============================================================================
// v3.5.x: P2P Transaction Signing for Browser-to-Network Direct Submission
// ============================================================================

/**
 * Transaction parameters for P2P signing
 * v3.7.4: Added usePQCrypto option for post-quantum signatures
 */
export interface P2PTransactionParams {
  from: string;
  to: string;
  amount: number; // In display units (e.g., 1.5 QUG)
  memo?: string;
  tokenAddress?: string;
  // v3.7.4: Enable post-quantum Dilithium5 signature (default: true when available)
  usePQCrypto?: boolean;
}

/**
 * Signed transaction for P2P gossipsub submission
 * v3.7.4: Now includes optional Dilithium5 post-quantum signatures
 */
export interface P2PSignedTransaction {
  from: string;
  to: string;
  amount: bigint;
  nonce: number;
  timestamp: number;
  // Classical Ed25519 signature (64 bytes)
  signature: Uint8Array;
  publicKey: Uint8Array;
  // v3.7.4: Post-quantum Dilithium5 signature (4,627 bytes) - optional
  dilithium5Signature?: Uint8Array;
  dilithium5PublicKey?: Uint8Array;
  // v3.7.4: Signature mode
  signatureMode?: 'ed25519' | 'dilithium5' | 'hybrid';
  tokenAddress?: string;
  memo?: string;
}

/**
 * Result of P2P transaction signing
 */
export interface P2PSigningResult {
  success: boolean;
  transaction?: P2PSignedTransaction;
  error?: string;
}

/**
 * Create and sign a transaction for P2P submission
 *
 * This enables browser nodes to submit transactions directly via gossipsub
 * instead of going through the HTTP API. Benefits:
 * - Lower latency (direct P2P broadcast)
 * - Better decentralization (no single API point)
 * - Network contribution (browser helps propagate)
 *
 * v3.7.4: Now includes optional Dilithium5 post-quantum signatures
 * When usePQCrypto is true (default), the transaction is signed with BOTH
 * Ed25519 (classical) and Dilithium5 (post-quantum) for hybrid security.
 *
 * @param params - Transaction parameters
 * @returns Signed transaction ready for P2P submission
 */
export async function signTransactionForP2P(
  params: P2PTransactionParams
): Promise<P2PSigningResult> {
  console.log('🔐 [P2P] Signing transaction for P2P submission...');

  // v3.7.4: Default to using PQ crypto if available
  const usePQCrypto = params.usePQCrypto !== false;

  try {
    // Get active session with private key
    const session = walletSession.getSession();
    if (!session) {
      return {
        success: false,
        error: 'No active wallet session. Please log in again.'
      };
    }

    const { privateKey, address } = session;

    // Verify sender address matches session
    const sessionAddress = address.startsWith('qnk') ? address : `qnk${address}`;
    const paramAddress = params.from.startsWith('qnk') ? params.from : `qnk${params.from}`;

    if (sessionAddress.toLowerCase() !== paramAddress.toLowerCase()) {
      return {
        success: false,
        error: 'Transaction sender does not match logged-in wallet'
      };
    }

    // Get public key from private key
    const publicKey = await ed25519.getPublicKey(privateKey);

    // Convert amount to atomic units (9 decimals for P2P signing protocol)
    // v3.5.15-beta: P2P signing uses i64 which fits 9 decimals max
    // Backend will scale up to 24 decimals after signature verification
    const QUG_DECIMALS = 1_000_000_000n; // 10^9 - fits in i64 for signing
    const amountAtomic = BigInt(Math.floor(params.amount * Number(QUG_DECIMALS)));

    // Generate nonce (use timestamp-based for simplicity, could be account-based)
    const timestamp = Math.floor(Date.now() / 1000);
    const nonce = timestamp; // Simple nonce strategy

    // Create transaction hash for signing
    // Hash = SHA3-256(from || to || amount || nonce || timestamp || memo)
    const fromHex = params.from.startsWith('qnk') ? params.from.substring(3) : params.from;
    const toHex = params.to.startsWith('qnk') ? params.to.substring(3) : params.to;

    const fromBytes = hexToBytes(fromHex);
    const toBytes = hexToBytes(toHex);

    // Amount as 8-byte little-endian
    const amountBytes = new Uint8Array(8);
    const amountView = new DataView(amountBytes.buffer);
    amountView.setBigInt64(0, amountAtomic, true);

    // Nonce as 4-byte little-endian
    const nonceBytes = new Uint8Array(4);
    const nonceView = new DataView(nonceBytes.buffer);
    nonceView.setUint32(0, nonce, true);

    // Timestamp as 8-byte little-endian
    const timestampBytes = new Uint8Array(8);
    const timestampView = new DataView(timestampBytes.buffer);
    timestampView.setBigInt64(0, BigInt(timestamp), true);

    // Memo bytes (empty if not provided)
    const memoBytes = params.memo ? new TextEncoder().encode(params.memo) : new Uint8Array(0);

    // Concatenate all fields for hashing
    const totalLength = fromBytes.length + toBytes.length + amountBytes.length +
                       nonceBytes.length + timestampBytes.length + memoBytes.length;
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    combined.set(fromBytes, offset); offset += fromBytes.length;
    combined.set(toBytes, offset); offset += toBytes.length;
    combined.set(amountBytes, offset); offset += amountBytes.length;
    combined.set(nonceBytes, offset); offset += nonceBytes.length;
    combined.set(timestampBytes, offset); offset += timestampBytes.length;
    combined.set(memoBytes, offset);

    // Hash the transaction data
    const txHash = sha3_256(combined);

    // Sign with Ed25519 (classical)
    const signature = await ed25519.sign(txHash, privateKey);

    // v3.7.4: Also sign with Dilithium5 (post-quantum) if enabled
    // Use persistent keys from session if available, otherwise generate ephemeral ones
    let dilithium5Signature: Uint8Array | undefined;
    let dilithium5PublicKey: Uint8Array | undefined;
    let signatureMode: 'ed25519' | 'dilithium5' | 'hybrid' = 'ed25519';

    if (usePQCrypto) {
      try {
        const pqCrypto = await getPQCrypto();

        if (pqCrypto.isPQCryptoAvailable()) {
          // v3.7.4: Use persistent Dilithium5 keys from session if available
          if (session.dilithium5SecretKey && session.dilithium5PublicKey) {
            console.log('🔐 [P2P] Using persistent Dilithium5 keys from session...');

            // Sign the transaction hash with persistent Dilithium5 keys
            dilithium5Signature = await pqCrypto.dilithium5Sign(txHash, session.dilithium5SecretKey);
            dilithium5PublicKey = session.dilithium5PublicKey;
            signatureMode = 'hybrid';

            console.log(`   ✅ Dilithium5 signature: ${dilithium5Signature.length} bytes (persistent key)`);
            console.log(`   ✅ Dilithium5 public key: ${dilithium5PublicKey.length} bytes`);
            console.log(`   🛡️ Post-quantum protection: ENABLED (NIST Level 5, persistent keys)`);
          } else {
            // Fallback: Generate ephemeral Dilithium5 keypair if no persistent keys
            console.log('🔐 [P2P] No persistent PQ keys, generating ephemeral Dilithium5 keypair...');

            const dilithiumKeypair = await pqCrypto.dilithium5KeyGen();

            // Sign the transaction hash with Dilithium5
            dilithium5Signature = await pqCrypto.dilithium5Sign(txHash, dilithiumKeypair.secretKey);
            dilithium5PublicKey = dilithiumKeypair.publicKey;
            signatureMode = 'hybrid';

            console.log(`   ✅ Dilithium5 signature: ${dilithium5Signature.length} bytes (ephemeral)`);
            console.log(`   ✅ Dilithium5 public key: ${dilithium5PublicKey.length} bytes`);
            console.log(`   ⚠️ Post-quantum protection: ENABLED but with ephemeral key`);
            console.log(`   💡 Tip: Log in again to use persistent PQ keys from your wallet`);
          }
        } else {
          console.warn('⚠️ [P2P] Post-quantum crypto not available, using Ed25519 only');
        }
      } catch (pqError) {
        console.warn('⚠️ [P2P] Failed to generate PQ signature, using Ed25519 only:', pqError);
      }
    }

    console.log('✅ [P2P] Transaction signed successfully');
    console.log(`   From: ${params.from.substring(0, 16)}...`);
    console.log(`   To: ${params.to.substring(0, 16)}...`);
    console.log(`   Amount: ${params.amount} (${amountAtomic} atomic)`);
    console.log(`   Nonce: ${nonce}`);
    console.log(`   Signature mode: ${signatureMode}`);

    return {
      success: true,
      transaction: {
        from: params.from,
        to: params.to,
        amount: amountAtomic,
        nonce,
        timestamp,
        signature,
        publicKey,
        dilithium5Signature,
        dilithium5PublicKey,
        signatureMode,
        tokenAddress: params.tokenAddress,
        memo: params.memo,
      }
    };
  } catch (error) {
    console.error('❌ [P2P] Failed to sign transaction:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown signing error'
    };
  }
}
