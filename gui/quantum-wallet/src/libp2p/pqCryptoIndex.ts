/**
 * Post-Quantum Cryptography Index
 * v3.7.4: Centralized exports for browser PQ crypto
 *
 * This module re-exports all post-quantum cryptography functionality
 * for easy importing throughout the application.
 *
 * Usage:
 * ```typescript
 * import {
 *   loadPQCrypto,
 *   generateHybridKeypair,
 *   dilithium5Sign,
 *   kyber1024Encapsulate,
 * } from './libp2p/pqCryptoIndex'
 * ```
 *
 * Security Level: NIST Level 5 (256-bit post-quantum)
 * - Dilithium5: Digital signatures (lattice-based)
 * - Kyber1024: Key encapsulation (lattice-based)
 * - Hybrid mode: Classical + PQ for maximum security
 */

// ============================================================================
// Post-Quantum Core Exports
// ============================================================================

export {
  // Initialization
  loadPQCrypto,
  isPQCryptoAvailable,
  getPQCryptoType,
  getPQCryptoStatus,

  // Dilithium5 (Signatures)
  dilithium5KeyGen,
  dilithium5Sign,
  dilithium5Verify,

  // Kyber1024 (Key Encapsulation)
  kyber1024KeyGen,
  kyber1024Encapsulate,
  kyber1024Decapsulate,

  // Hybrid Cryptography
  generateHybridKeypair,
  createHybridSignature,
  verifyHybridSignature,
  establishPQSession,

  // Handshake Protocol
  createPQHandshake,
  verifyPQHandshake,

  // Utilities
  bytesToHex,
  hexToBytes,
  computeFingerprint,

  // Constants
  DILITHIUM5_PUBLIC_KEY_BYTES,
  DILITHIUM5_SECRET_KEY_BYTES,
  DILITHIUM5_SIGNATURE_BYTES,
  KYBER1024_PUBLIC_KEY_BYTES,
  KYBER1024_SECRET_KEY_BYTES,
  KYBER1024_CIPHERTEXT_BYTES,
  KYBER1024_SHARED_SECRET_BYTES,

  // Types
  type Dilithium5Keypair,
  type Kyber1024Keypair,
  type KyberEncapsulation,
  type HybridKeypair,
  type HybridSignature,
  type PQHandshakeMessage as PQHandshakeMessageCore,
  type PQSession as PQSessionCore,
} from './postQuantumCrypto'

// ============================================================================
// Connection Encrypter Exports
// ============================================================================

export {
  pqNoise,
  createHybridEncrypter,
  getPQNoiseStatus,
  PQ_PROTOCOL_NAME,
} from './pqConnectionEncrypter'

// ============================================================================
// Type Re-exports from types.ts
// ============================================================================

export type {
  PQSignedTransaction,
  PQPeerIdentity,
  PQHandshakeMessage,
  PQSession,
  PQCryptoStatus,
} from './types'

// ============================================================================
// Quick Start Guide
// ============================================================================

/**
 * QUICK START: Post-Quantum Cryptography in Q-NarwhalKnight Browser
 *
 * 1. Initialize PQ Crypto (done automatically at node start):
 * ```typescript
 * await loadPQCrypto()
 * ```
 *
 * 2. Generate a hybrid keypair:
 * ```typescript
 * const keypair = await generateHybridKeypair()
 * // keypair.ed25519PublicKey - Classical signing key
 * // keypair.dilithium5PublicKey - PQ signing key (2,592 bytes)
 * // keypair.kyber1024PublicKey - PQ key exchange key (1,568 bytes)
 * // keypair.fingerprint - Unique identifier
 * ```
 *
 * 3. Sign a message with hybrid signature:
 * ```typescript
 * const signature = await createHybridSignature(message, keypair)
 * // signature.ed25519Signature - Classical signature (64 bytes)
 * // signature.dilithium5Signature - PQ signature (4,627 bytes)
 * ```
 *
 * 4. Verify a hybrid signature:
 * ```typescript
 * const valid = await verifyHybridSignature(
 *   message,
 *   signature,
 *   keypair.ed25519PublicKey,
 *   keypair.dilithium5PublicKey
 * )
 * ```
 *
 * 5. Establish a PQ session (key exchange):
 * ```typescript
 * const session = await establishPQSession(
 *   localKeypair,
 *   remoteKyber1024PublicKey,
 *   remoteEd25519PublicKey
 * )
 * // session.sharedSecret - 32-byte key for symmetric encryption
 * ```
 *
 * WHY POST-QUANTUM?
 *
 * Current classical cryptography (RSA, ECDSA, X25519) can be broken by
 * sufficiently powerful quantum computers using Shor's algorithm.
 *
 * Dilithium and Kyber are based on lattice problems (Module-LWE), which
 * operate in 1,024+ dimensional mathematical spaces. Finding the shortest
 * vector in these high-dimensional lattices is believed to be hard even
 * for quantum computers.
 *
 * By using hybrid mode (classical + PQ), we get:
 * - Protection against classical attacks (today)
 * - Protection against quantum attacks (future)
 * - If EITHER system is secure, the whole system is secure
 *
 * PERFORMANCE NOTES:
 *
 * | Operation              | Time (ms) | Size (bytes) |
 * |------------------------|-----------|--------------|
 * | Dilithium5 KeyGen      | ~5-10     | 2,592 (pk)   |
 * | Dilithium5 Sign        | ~2-5      | 4,627        |
 * | Dilithium5 Verify      | ~1-2      | -            |
 * | Kyber1024 KeyGen       | ~2-5      | 1,568 (pk)   |
 * | Kyber1024 Encapsulate  | ~1-2      | 1,568 (ct)   |
 * | Kyber1024 Decapsulate  | ~1-2      | 32 (ss)      |
 * | Hybrid Keypair Gen     | ~10-20    | ~4,192       |
 *
 * The larger key/signature sizes are acceptable for:
 * - One-time operations (key generation, session establishment)
 * - P2P handshakes (done once per connection)
 * - Transaction signing (security > bandwidth)
 *
 * For high-frequency operations, consider SQIsign (204-byte signatures)
 * which trades computation time for smaller sizes.
 */
