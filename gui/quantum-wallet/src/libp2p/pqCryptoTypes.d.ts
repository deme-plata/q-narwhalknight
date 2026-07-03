/**
 * Type declarations for post-quantum cryptography packages
 * v3.7.4: Dilithium5 + Kyber1024 TypeScript support
 */

// ============================================================================
// crystals-kyber types
// ============================================================================

declare module 'crystals-kyber' {
  /**
   * Generate a Kyber1024 keypair
   * @returns [publicKey, secretKey] as number arrays
   */
  export function KeyGen1024(): [number[], number[]]

  /**
   * Encapsulate (encrypt) a shared secret with a public key
   * @param publicKey - Recipient's public key
   * @returns [ciphertext, sharedSecret] as number arrays
   */
  export function Encrypt1024(publicKey: number[]): [number[], number[]]

  /**
   * Decapsulate (decrypt) a shared secret with a secret key
   * @param ciphertext - The ciphertext from encapsulation
   * @param secretKey - The secret key
   * @returns Shared secret as number array
   */
  export function Decrypt1024(ciphertext: number[], secretKey: number[]): number[]

  // Kyber768 variants
  export function KeyGen768(): [number[], number[]]
  export function Encrypt768(publicKey: number[]): [number[], number[]]
  export function Decrypt768(ciphertext: number[], secretKey: number[]): number[]

  // Kyber512 variants
  export function KeyGen512(): [number[], number[]]
  export function Encrypt512(publicKey: number[]): [number[], number[]]
  export function Decrypt512(ciphertext: number[], secretKey: number[]): number[]
}

// ============================================================================
// dilithium-crystals types (from package index.d.ts)
// ============================================================================

declare module 'dilithium-crystals' {
  interface IDilithium {
    /** Signature length. */
    bytes: Promise<number>

    /** Private key length. */
    privateKeyBytes: Promise<number>

    /** Public key length. */
    publicKeyBytes: Promise<number>

    /** Generates key pair. */
    keyPair(): Promise<{ privateKey: Uint8Array; publicKey: Uint8Array }>

    /** Verifies signed message against publicKey and returns it. */
    open(signed: Uint8Array, publicKey: Uint8Array): Promise<Uint8Array>

    /** Signs message with privateKey and returns combined message. */
    sign(message: Uint8Array, privateKey: Uint8Array): Promise<Uint8Array>

    /** Signs message with privateKey and returns signature. */
    signDetached(message: Uint8Array, privateKey: Uint8Array): Promise<Uint8Array>

    /** Verifies detached signature against publicKey. */
    verifyDetached(
      signature: Uint8Array,
      message: Uint8Array,
      publicKey: Uint8Array
    ): Promise<boolean>
  }

  const dilithium: IDilithium
  export default dilithium
  export { IDilithium }
}

// ============================================================================
// liboqs-wasm types (optional, for when available)
// ============================================================================

declare module 'liboqs-wasm' {
  export default function init(): Promise<void>

  export class Signature {
    constructor(algorithm: string)
    generateKeypair(): Uint8Array
    exportSecretKey(): Uint8Array
    importSecretKey(key: Uint8Array): void
    sign(message: Uint8Array): Uint8Array
    verify(message: Uint8Array, signature: Uint8Array, publicKey: Uint8Array): boolean
  }

  export class KEM {
    constructor(algorithm: string)
    generateKeypair(): Uint8Array
    exportSecretKey(): Uint8Array
    importSecretKey(key: Uint8Array): void
    encaps(publicKey: Uint8Array): { ciphertext: Uint8Array; sharedSecret: Uint8Array }
    decaps(ciphertext: Uint8Array): Uint8Array
  }

  export const algorithms: {
    signatures: string[]
    kems: string[]
  }
}
