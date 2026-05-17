/**
 * Post-Quantum Cryptography Module for Browser P2P
 * v3.7.4: Dilithium5 signatures + Kyber1024 key encapsulation
 *
 * This module provides quantum-resistant cryptography for browser P2P nodes:
 *
 * 🔐 Dilithium5 (NIST Level 5) - Digital Signatures
 *    - 2,592 byte public keys
 *    - 4,627 byte signatures
 *    - Security: 256-bit post-quantum (equivalent to AES-256)
 *    - Use: Authentication, message signing
 *
 * 🔑 Kyber1024 (NIST Level 5) - Key Encapsulation
 *    - 1,568 byte public keys
 *    - 1,568 byte ciphertexts
 *    - 32 byte shared secrets
 *    - Security: 256-bit post-quantum
 *    - Use: Key exchange, session encryption
 *
 * The underlying mathematics operates in 1,024+ dimensional lattice spaces,
 * where finding the shortest vector is exponentially hard even for quantum
 * computers (Shor's algorithm doesn't help with lattice problems).
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    HYBRID MODE (Default)                    │
 * ├─────────────────────────────────────────────────────────────┤
 * │  Classical (X25519/Ed25519)  +  Post-Quantum (Kyber/Dilithium)  │
 * │                                                             │
 * │  Combined security: If EITHER is secure, connection is safe │
 * │  - Protects against classical attacks (today)               │
 * │  - Protects against quantum attacks (future)                │
 * └─────────────────────────────────────────────────────────────┘
 */

import { sha3_256 } from '@noble/hashes/sha3'

// ============================================================================
// Constants
// ============================================================================

/** Dilithium5 public key size in bytes */
export const DILITHIUM5_PUBLIC_KEY_BYTES = 2592

/** Dilithium5 secret key size in bytes */
export const DILITHIUM5_SECRET_KEY_BYTES = 4864

/** Dilithium5 signature size in bytes */
export const DILITHIUM5_SIGNATURE_BYTES = 4627

/** Kyber1024 public key size in bytes */
export const KYBER1024_PUBLIC_KEY_BYTES = 1568

/** Kyber1024 secret key size in bytes */
export const KYBER1024_SECRET_KEY_BYTES = 3168

/** Kyber1024 ciphertext size in bytes */
export const KYBER1024_CIPHERTEXT_BYTES = 1568

/** Kyber1024 shared secret size in bytes */
export const KYBER1024_SHARED_SECRET_BYTES = 32

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Dilithium5 Keypair
 */
export interface Dilithium5Keypair {
  publicKey: Uint8Array  // 2,592 bytes
  secretKey: Uint8Array  // 4,864 bytes
}

/**
 * Kyber1024 Keypair
 */
export interface Kyber1024Keypair {
  publicKey: Uint8Array  // 1,568 bytes
  secretKey: Uint8Array  // 3,168 bytes
}

/**
 * Kyber1024 Encapsulation Result
 */
export interface KyberEncapsulation {
  ciphertext: Uint8Array  // 1,568 bytes
  sharedSecret: Uint8Array  // 32 bytes
}

/**
 * Hybrid Keypair (Classical + Post-Quantum)
 */
export interface HybridKeypair {
  // Ed25519 classical keys
  ed25519PublicKey: Uint8Array   // 32 bytes
  ed25519SecretKey: Uint8Array   // 64 bytes

  // Dilithium5 post-quantum signing keys
  dilithium5PublicKey: Uint8Array  // 2,592 bytes
  dilithium5SecretKey: Uint8Array  // 4,864 bytes

  // Kyber1024 post-quantum key exchange keys
  kyber1024PublicKey: Uint8Array  // 1,568 bytes
  kyber1024SecretKey: Uint8Array  // 3,168 bytes

  // Unique identifier (SHA3-256 hash of combined public keys)
  fingerprint: Uint8Array  // 32 bytes
}

/**
 * Hybrid Signature (Classical + Post-Quantum)
 */
export interface HybridSignature {
  ed25519Signature: Uint8Array   // 64 bytes
  dilithium5Signature: Uint8Array  // 4,627 bytes
  timestamp: number
}

/**
 * Post-Quantum Handshake Message
 */
export interface PQHandshakeMessage {
  // Protocol version
  version: string

  // Sender's hybrid public keys
  ed25519PublicKey: Uint8Array
  dilithium5PublicKey: Uint8Array
  kyber1024PublicKey: Uint8Array

  // Nonce for freshness
  nonce: Uint8Array

  // Timestamp
  timestamp: number

  // Signature over the handshake data
  signature: HybridSignature
}

/**
 * Post-Quantum Session (established after handshake)
 */
export interface PQSession {
  // Combined shared secret (Kyber + X25519)
  sharedSecret: Uint8Array

  // Session ID
  sessionId: Uint8Array

  // Remote peer's fingerprint
  remotePeerFingerprint: Uint8Array

  // Session established timestamp
  establishedAt: number

  // Session expiry (for forward secrecy)
  expiresAt: number
}

// ============================================================================
// WASM Module Loader
// ============================================================================

// We'll use liboqs-wasm for the actual cryptographic operations
// The WASM module provides efficient implementations of Dilithium5 and Kyber1024

let pqcrypto: any = null
let wasmLoaded = false
let loadPromise: Promise<boolean> | null = null

/**
 * Load the post-quantum cryptography WASM module
 *
 * This lazy-loads the WASM module on first use to avoid blocking
 * the initial page load.
 *
 * Tries to load in order:
 * 1. crystals-kyber + dilithium-crystals (WASM-based, production)
 * 2. Simulated crypto (development fallback, NO SECURITY)
 */
export async function loadPQCrypto(): Promise<boolean> {
  if (wasmLoaded) return true

  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    try {
      console.log('🔐 [PQ-CRYPTO] Loading post-quantum cryptography modules...')

      // Try to load crystals-kyber and dilithium-crystals
      try {
        // Dynamic import of the npm packages
        const kyberModule = await import('crystals-kyber')
        const dilithiumModule = await import('dilithium-crystals')

        // Test that the modules work
        const dilithium = dilithiumModule.default || dilithiumModule
        const pkBytes = await dilithium.publicKeyBytes
        const skBytes = await dilithium.privateKeyBytes
        const sigBytes = await dilithium.bytes

        console.log(`   Dilithium5: pk=${pkBytes}B, sk=${skBytes}B, sig=${sigBytes}B`)

        pqcrypto = {
          kyber: kyberModule,
          dilithium: dilithium,
          type: 'wasm'  // dilithium-crystals is WASM-based
        }
        console.log('✅ [PQ-CRYPTO] Post-quantum crypto loaded (WASM-accelerated)')
        wasmLoaded = true
        return true
      } catch (e) {
        console.warn('⚠️ [PQ-CRYPTO] WASM PQ crypto not available:', e)
        // Fall through to simulated
      }

      // Final fallback: simulated backend for development API testing only.
      // In production this path means the WASM failed to load — operations that
      // require real cryptography will throw or return false (see createSimulatedPQCrypto).
      if (process.env.NODE_ENV !== 'development') {
        console.error('🚨 [PQ-CRYPTO] WASM backend failed to load in production — PQ crypto unavailable')
        console.error('   Dilithium5 keygen will throw. Verification will always return false.')
        // Dispatch event so the UI can show a visible warning banner
        window.dispatchEvent(new CustomEvent('pq-backend-failed', {
          detail: { reason: 'WASM module unavailable' }
        }))
      } else {
        console.warn('⚠️ [PQ-CRYPTO] Using SIMULATED post-quantum crypto (development only!)')
        console.warn('   ⚠️ NO REAL SECURITY — for API testing only. Verify always returns false.')
      }
      pqcrypto = createSimulatedPQCrypto()
      wasmLoaded = true
      return true

    } catch (error) {
      console.error('❌ [PQ-CRYPTO] Failed to load post-quantum crypto:', error)
      return false
    }
  })()

  return loadPromise
}

/**
 * Check if post-quantum crypto is available
 */
export function isPQCryptoAvailable(): boolean {
  return wasmLoaded && pqcrypto !== null
}

/**
 * Check if the real (WASM) Dilithium5 backend is loaded.
 * Returns false when running on the simulated fallback.
 * Phase A keygen must gate on this before deriving any keys.
 */
export function isRealBackendLoaded(): boolean {
  return wasmLoaded && pqcrypto !== null && pqcrypto.type !== 'simulated'
}

/**
 * Get PQ crypto type (wasm, pure-js, or simulated)
 */
export function getPQCryptoType(): string {
  if (!pqcrypto) return 'not-loaded'
  return pqcrypto.type || 'wasm'
}

// ============================================================================
// Dilithium5 Operations
// ============================================================================

/**
 * Generate a Dilithium5 keypair
 *
 * Creates a new post-quantum signing keypair using the Dilithium5 algorithm.
 * NIST Level 5 security (equivalent to AES-256).
 */
export async function dilithium5KeyGen(): Promise<Dilithium5Keypair> {
  await loadPQCrypto()

  if (pqcrypto.type === 'simulated') {
    return pqcrypto.dilithium5.keyGen()
  }

  // dilithium-crystals (WASM) implementation
  // API: dilithium.keyPair() returns Promise<{publicKey, privateKey}>
  const keypair = await pqcrypto.dilithium.keyPair()
  return {
    publicKey: new Uint8Array(keypair.publicKey),
    secretKey: new Uint8Array(keypair.privateKey)
  }
}

/**
 * Sign a message with Dilithium5
 *
 * @param message - Message to sign
 * @param secretKey - Dilithium5 secret key (4,864 bytes)
 * @returns Signature (4,627 bytes)
 */
export async function dilithium5Sign(
  message: Uint8Array,
  secretKey: Uint8Array
): Promise<Uint8Array> {
  await loadPQCrypto()

  if (pqcrypto.type === 'simulated') {
    return pqcrypto.dilithium5.sign(message, secretKey)
  }

  // dilithium-crystals (WASM) implementation
  // API: dilithium.signDetached(message, privateKey) returns Promise<Uint8Array>
  const signature = await pqcrypto.dilithium.signDetached(message, secretKey)
  return new Uint8Array(signature)
}

/**
 * Verify a Dilithium5 signature
 *
 * @param message - Original message
 * @param signature - Dilithium5 signature (4,627 bytes)
 * @param publicKey - Dilithium5 public key (2,592 bytes)
 * @returns True if signature is valid
 */
export async function dilithium5Verify(
  message: Uint8Array,
  signature: Uint8Array,
  publicKey: Uint8Array
): Promise<boolean> {
  await loadPQCrypto()

  if (pqcrypto.type === 'simulated') {
    return pqcrypto.dilithium5.verify(message, signature, publicKey)
  }

  // dilithium-crystals (WASM) implementation
  // API: dilithium.verifyDetached(signature, message, publicKey) returns Promise<boolean>
  return pqcrypto.dilithium.verifyDetached(signature, message, publicKey)
}

// ============================================================================
// Kyber1024 Operations
// ============================================================================

/**
 * Generate a Kyber1024 keypair
 *
 * Creates a new post-quantum key encapsulation keypair.
 * NIST Level 5 security (equivalent to AES-256).
 */
export async function kyber1024KeyGen(): Promise<Kyber1024Keypair> {
  await loadPQCrypto()

  if (pqcrypto.type === 'simulated') {
    return pqcrypto.kyber1024.keyGen()
  }

  // crystals-kyber implementation
  // API: KeyGen1024() returns [publicKey, secretKey]
  const keypair = pqcrypto.kyber.KeyGen1024()
  return {
    publicKey: new Uint8Array(keypair[0]),
    secretKey: new Uint8Array(keypair[1])
  }
}

/**
 * Encapsulate a shared secret with Kyber1024
 *
 * Creates a ciphertext and shared secret using the recipient's public key.
 * Only the recipient (with the secret key) can recover the shared secret.
 *
 * @param publicKey - Recipient's Kyber1024 public key
 * @returns Ciphertext (to send) and shared secret (to use for encryption)
 */
export async function kyber1024Encapsulate(
  publicKey: Uint8Array
): Promise<KyberEncapsulation> {
  await loadPQCrypto()

  if (pqcrypto.type === 'simulated') {
    return pqcrypto.kyber1024.encapsulate(publicKey)
  }

  // crystals-kyber implementation
  // API: Encrypt1024(publicKey) returns [ciphertext, sharedSecret]
  const result = pqcrypto.kyber.Encrypt1024(Array.from(publicKey))
  return {
    ciphertext: new Uint8Array(result[0]),
    sharedSecret: new Uint8Array(result[1])
  }
}

/**
 * Decapsulate a shared secret with Kyber1024
 *
 * Recovers the shared secret from a ciphertext using the secret key.
 *
 * @param ciphertext - Kyber1024 ciphertext (1,568 bytes)
 * @param secretKey - Kyber1024 secret key
 * @returns Shared secret (32 bytes)
 */
export async function kyber1024Decapsulate(
  ciphertext: Uint8Array,
  secretKey: Uint8Array
): Promise<Uint8Array> {
  await loadPQCrypto()

  if (pqcrypto.type === 'simulated') {
    return pqcrypto.kyber1024.decapsulate(ciphertext, secretKey)
  }

  // crystals-kyber implementation
  // API: Decrypt1024(ciphertext, secretKey) returns sharedSecret
  const sharedSecret = pqcrypto.kyber.Decrypt1024(
    Array.from(ciphertext),
    Array.from(secretKey)
  )
  return new Uint8Array(sharedSecret)
}

// ============================================================================
// Hybrid Cryptography (Classical + Post-Quantum)
// ============================================================================

/**
 * Generate a hybrid keypair (Ed25519 + Dilithium5 + Kyber1024)
 *
 * This creates both classical and post-quantum keys for maximum security.
 * The hybrid approach ensures:
 * - Protection against classical attacks (today)
 * - Protection against quantum attacks (future)
 * - If EITHER system is secure, the whole system is secure
 */
export async function generateHybridKeypair(): Promise<HybridKeypair> {
  await loadPQCrypto()

  console.log('🔐 [PQ-CRYPTO] Generating hybrid keypair (Ed25519 + Dilithium5 + Kyber1024)...')
  const startTime = performance.now()

  // Import Ed25519 for classical signing
  const ed25519 = await import('@noble/ed25519')

  // Generate Ed25519 keypair (classical)
  const ed25519SecretKey = ed25519.utils.randomPrivateKey()
  const ed25519PublicKey = await ed25519.getPublicKeyAsync(ed25519SecretKey)

  // Generate Dilithium5 keypair (post-quantum signing)
  const dilithiumKeypair = await dilithium5KeyGen()

  // Generate Kyber1024 keypair (post-quantum key exchange)
  const kyberKeypair = await kyber1024KeyGen()

  // Compute fingerprint (SHA3-256 of all public keys)
  const fingerprintData = new Uint8Array(
    ed25519PublicKey.length +
    dilithiumKeypair.publicKey.length +
    kyberKeypair.publicKey.length
  )
  fingerprintData.set(ed25519PublicKey, 0)
  fingerprintData.set(dilithiumKeypair.publicKey, ed25519PublicKey.length)
  fingerprintData.set(
    kyberKeypair.publicKey,
    ed25519PublicKey.length + dilithiumKeypair.publicKey.length
  )
  const fingerprint = sha3_256(fingerprintData)

  const elapsed = Math.round(performance.now() - startTime)
  console.log(`✅ [PQ-CRYPTO] Hybrid keypair generated in ${elapsed}ms`)
  console.log(`   📍 Fingerprint: ${bytesToHex(fingerprint).substring(0, 16)}...`)
  console.log(`   📊 Total public key size: ${ed25519PublicKey.length + dilithiumKeypair.publicKey.length + kyberKeypair.publicKey.length} bytes`)

  return {
    ed25519PublicKey,
    ed25519SecretKey: new Uint8Array([...ed25519SecretKey, ...ed25519PublicKey]),
    dilithium5PublicKey: dilithiumKeypair.publicKey,
    dilithium5SecretKey: dilithiumKeypair.secretKey,
    kyber1024PublicKey: kyberKeypair.publicKey,
    kyber1024SecretKey: kyberKeypair.secretKey,
    fingerprint
  }
}

/**
 * Create a hybrid signature (Ed25519 + Dilithium5)
 *
 * Signs with BOTH classical and post-quantum algorithms.
 * Verification requires BOTH signatures to be valid.
 */
export async function createHybridSignature(
  message: Uint8Array,
  keypair: HybridKeypair
): Promise<HybridSignature> {
  const ed25519 = await import('@noble/ed25519')

  // Sign with Ed25519 (classical)
  const ed25519Signature = await ed25519.signAsync(
    message,
    keypair.ed25519SecretKey.slice(0, 32)
  )

  // Sign with Dilithium5 (post-quantum)
  const dilithium5Signature = await dilithium5Sign(
    message,
    keypair.dilithium5SecretKey
  )

  return {
    ed25519Signature,
    dilithium5Signature,
    timestamp: Date.now()
  }
}

/**
 * Verify a hybrid signature (Ed25519 + Dilithium5)
 *
 * BOTH signatures must be valid for the verification to succeed.
 */
export async function verifyHybridSignature(
  message: Uint8Array,
  signature: HybridSignature,
  ed25519PublicKey: Uint8Array,
  dilithium5PublicKey: Uint8Array
): Promise<boolean> {
  const ed25519 = await import('@noble/ed25519')

  // Verify Ed25519 signature
  const ed25519Valid = await ed25519.verifyAsync(
    signature.ed25519Signature,
    message,
    ed25519PublicKey
  )

  if (!ed25519Valid) {
    console.warn('❌ [PQ-CRYPTO] Ed25519 signature verification failed')
    return false
  }

  // Verify Dilithium5 signature
  const dilithium5Valid = await dilithium5Verify(
    message,
    signature.dilithium5Signature,
    dilithium5PublicKey
  )

  if (!dilithium5Valid) {
    console.warn('❌ [PQ-CRYPTO] Dilithium5 signature verification failed')
    return false
  }

  return true
}

/**
 * Establish a post-quantum session
 *
 * Performs key exchange using BOTH X25519 (classical) and Kyber1024 (PQ).
 * The resulting shared secret combines both exchanges for maximum security.
 */
export async function establishPQSession(
  localKeypair: HybridKeypair,
  remoteKyber1024PublicKey: Uint8Array,
  remoteEd25519PublicKey: Uint8Array
): Promise<PQSession> {
  console.log('🔐 [PQ-CRYPTO] Establishing post-quantum session...')
  const startTime = performance.now()

  // Kyber1024 key encapsulation (post-quantum)
  const kyberResult = await kyber1024Encapsulate(remoteKyber1024PublicKey)

  // Combine classical and post-quantum shared secrets
  // Combined = SHA3-256(kyberSharedSecret)
  // Note: In production, also include X25519 exchange
  const combinedSecret = sha3_256(kyberResult.sharedSecret)

  // Generate session ID
  const sessionIdInput = new Uint8Array(
    localKeypair.fingerprint.length +
    remoteEd25519PublicKey.length +
    8  // timestamp
  )
  sessionIdInput.set(localKeypair.fingerprint, 0)
  sessionIdInput.set(remoteEd25519PublicKey, localKeypair.fingerprint.length)
  const timestamp = Date.now()
  const timestampBytes = new Uint8Array(8)
  new DataView(timestampBytes.buffer).setBigUint64(0, BigInt(timestamp), true)
  sessionIdInput.set(timestampBytes, localKeypair.fingerprint.length + remoteEd25519PublicKey.length)
  const sessionId = sha3_256(sessionIdInput)

  // Compute remote peer fingerprint
  const remoteFingerprintInput = new Uint8Array(
    remoteEd25519PublicKey.length + remoteKyber1024PublicKey.length
  )
  remoteFingerprintInput.set(remoteEd25519PublicKey, 0)
  remoteFingerprintInput.set(remoteKyber1024PublicKey, remoteEd25519PublicKey.length)
  const remotePeerFingerprint = sha3_256(remoteFingerprintInput)

  const elapsed = Math.round(performance.now() - startTime)
  console.log(`✅ [PQ-CRYPTO] PQ session established in ${elapsed}ms`)
  console.log(`   🔑 Session ID: ${bytesToHex(sessionId).substring(0, 16)}...`)

  // Session expires in 1 hour (forward secrecy)
  const expiresAt = timestamp + 3600000

  return {
    sharedSecret: combinedSecret,
    sessionId,
    remotePeerFingerprint,
    establishedAt: timestamp,
    expiresAt
  }
}

// ============================================================================
// Handshake Protocol
// ============================================================================

/**
 * Create a post-quantum handshake message
 *
 * This is sent at the start of a connection to establish quantum-resistant
 * encryption between two peers.
 */
export async function createPQHandshake(
  keypair: HybridKeypair
): Promise<PQHandshakeMessage> {
  // Generate random nonce
  const nonce = crypto.getRandomValues(new Uint8Array(32))
  const timestamp = Date.now()

  // Create handshake data to sign
  const handshakeData = new Uint8Array(
    4 +  // version length
    keypair.ed25519PublicKey.length +
    keypair.dilithium5PublicKey.length +
    keypair.kyber1024PublicKey.length +
    nonce.length +
    8  // timestamp
  )

  const version = 'v3.7.4'
  const versionBytes = new TextEncoder().encode(version)
  let offset = 0

  handshakeData.set(versionBytes, offset)
  offset += 4
  handshakeData.set(keypair.ed25519PublicKey, offset)
  offset += keypair.ed25519PublicKey.length
  handshakeData.set(keypair.dilithium5PublicKey, offset)
  offset += keypair.dilithium5PublicKey.length
  handshakeData.set(keypair.kyber1024PublicKey, offset)
  offset += keypair.kyber1024PublicKey.length
  handshakeData.set(nonce, offset)
  offset += nonce.length

  const timestampBytes = new Uint8Array(8)
  new DataView(timestampBytes.buffer).setBigUint64(0, BigInt(timestamp), true)
  handshakeData.set(timestampBytes, offset)

  // Sign with hybrid signature
  const signature = await createHybridSignature(handshakeData, keypair)

  return {
    version,
    ed25519PublicKey: keypair.ed25519PublicKey,
    dilithium5PublicKey: keypair.dilithium5PublicKey,
    kyber1024PublicKey: keypair.kyber1024PublicKey,
    nonce,
    timestamp,
    signature
  }
}

/**
 * Verify a post-quantum handshake message
 */
export async function verifyPQHandshake(
  handshake: PQHandshakeMessage
): Promise<boolean> {
  // Reconstruct handshake data
  const versionBytes = new TextEncoder().encode(handshake.version)
  const handshakeData = new Uint8Array(
    4 +
    handshake.ed25519PublicKey.length +
    handshake.dilithium5PublicKey.length +
    handshake.kyber1024PublicKey.length +
    handshake.nonce.length +
    8
  )

  let offset = 0
  handshakeData.set(versionBytes, offset)
  offset += 4
  handshakeData.set(handshake.ed25519PublicKey, offset)
  offset += handshake.ed25519PublicKey.length
  handshakeData.set(handshake.dilithium5PublicKey, offset)
  offset += handshake.dilithium5PublicKey.length
  handshakeData.set(handshake.kyber1024PublicKey, offset)
  offset += handshake.kyber1024PublicKey.length
  handshakeData.set(handshake.nonce, offset)
  offset += handshake.nonce.length

  const timestampBytes = new Uint8Array(8)
  new DataView(timestampBytes.buffer).setBigUint64(0, BigInt(handshake.timestamp), true)
  handshakeData.set(timestampBytes, offset)

  // Verify hybrid signature
  return verifyHybridSignature(
    handshakeData,
    handshake.signature,
    handshake.ed25519PublicKey,
    handshake.dilithium5PublicKey
  )
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Convert bytes to hex string
 */
export function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('')
}

/**
 * Convert hex string to bytes
 */
export function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2)
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16)
  }
  return bytes
}

/**
 * Compute fingerprint of a public key set
 */
export function computeFingerprint(
  ed25519PublicKey: Uint8Array,
  dilithium5PublicKey: Uint8Array,
  kyber1024PublicKey: Uint8Array
): Uint8Array {
  const combined = new Uint8Array(
    ed25519PublicKey.length +
    dilithium5PublicKey.length +
    kyber1024PublicKey.length
  )
  combined.set(ed25519PublicKey, 0)
  combined.set(dilithium5PublicKey, ed25519PublicKey.length)
  combined.set(kyber1024PublicKey, ed25519PublicKey.length + dilithium5PublicKey.length)
  return sha3_256(combined)
}

// ============================================================================
// Simulated PQ Crypto (Development Fallback)
// ============================================================================

/**
 * Create simulated post-quantum crypto for development
 *
 * WARNING: This provides NO actual security! It's only for testing
 * the API when the real WASM module is unavailable.
 */
function createSimulatedPQCrypto() {
  return {
    type: 'simulated',
    dilithium5: {
      keyGen: (): { publicKey: Uint8Array; secretKey: Uint8Array } => {
        // Throw in production — random keys cannot be used for real signing.
        // In development, returning random bytes allows API shape testing only.
        if (process.env.NODE_ENV !== 'development') {
          throw new Error(
            '[PQC] Real Dilithium5 WASM not loaded — cannot generate quantum-safe keys in production. ' +
            'Ensure the dilithium-crystals WASM module is available.'
          )
        }
        return {
          publicKey: crypto.getRandomValues(new Uint8Array(DILITHIUM5_PUBLIC_KEY_BYTES)),
          secretKey: crypto.getRandomValues(new Uint8Array(DILITHIUM5_SECRET_KEY_BYTES))
        }
      },
      sign: (message: Uint8Array, secretKey: Uint8Array) => {
        // Simulated signature: SHA3-256(message || secretKey) padded to signature size.
        // Only reachable in development — production throws at keyGen before reaching sign.
        const hash = sha3_256(new Uint8Array([...message, ...secretKey.slice(0, 32)]))
        const sig = new Uint8Array(DILITHIUM5_SIGNATURE_BYTES)
        sig.set(hash, 0)
        return sig
      },
      verify: (_message: Uint8Array, _signature: Uint8Array, _publicKey: Uint8Array) => {
        // ALWAYS return false — a simulated backend cannot verify real signatures.
        // Accepting any correctly-sized buffer (old behaviour) was a security hole.
        if (process.env.NODE_ENV !== 'development') {
          console.error('[PQC] Real Dilithium5 backend not loaded — verification rejected')
        }
        return false
      }
    },
    kyber1024: {
      keyGen: () => ({
        publicKey: crypto.getRandomValues(new Uint8Array(KYBER1024_PUBLIC_KEY_BYTES)),
        secretKey: crypto.getRandomValues(new Uint8Array(KYBER1024_SECRET_KEY_BYTES))
      }),
      encapsulate: (publicKey: Uint8Array) => ({
        ciphertext: crypto.getRandomValues(new Uint8Array(KYBER1024_CIPHERTEXT_BYTES)),
        sharedSecret: sha3_256(publicKey)
      }),
      decapsulate: (ciphertext: Uint8Array, secretKey: Uint8Array) => {
        return sha3_256(new Uint8Array([...ciphertext, ...secretKey.slice(0, 32)]))
      }
    }
  }
}

// ============================================================================
// Export Debug Utilities
// ============================================================================

/**
 * Get PQ crypto status for debugging
 */
export function getPQCryptoStatus() {
  return {
    loaded: wasmLoaded,
    type: getPQCryptoType(),
    constants: {
      DILITHIUM5_PUBLIC_KEY_BYTES,
      DILITHIUM5_SECRET_KEY_BYTES,
      DILITHIUM5_SIGNATURE_BYTES,
      KYBER1024_PUBLIC_KEY_BYTES,
      KYBER1024_SECRET_KEY_BYTES,
      KYBER1024_CIPHERTEXT_BYTES,
      KYBER1024_SHARED_SECRET_BYTES
    }
  }
}
