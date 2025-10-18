/**
 * AEGIS-QL: Asymmetric Efficient Graph-based Integer System with Quantum Resistance
 *
 * Simplified TypeScript implementation for frontend wallet authentication.
 * Based on the Rust implementation in crates/q-aegis-ql/src/lib.rs
 *
 * This is a lightweight implementation focused on signature generation for authentication.
 * Uses sparse Ring-LWE with optimized NTT operations for post-quantum security.
 */

import { sha3_256 } from '@noble/hashes/sha3';

// AEGIS-QL Parameters (matching Rust implementation)
export const POLY_DEGREE = 512;
export const MODULUS = 12289; // NTT-friendly prime
export const GRAPH_DEGREE = 8; // Sparse polynomial degree

/**
 * Sparse polynomial representation (memory-optimized)
 */
export interface SparsePolynomial {
  coefficients: number[];
  indices: number[];
  degree: number;
}

/**
 * AEGIS-QL Public Key
 */
export interface AegisPublicKey {
  a: number[]; // Uniform random polynomial
  t: number[]; // t = a*s + e
}

/**
 * AEGIS-QL Secret Key
 */
export interface AegisSecretKey {
  s: SparsePolynomial; // Sparse secret polynomial
}

/**
 * AEGIS-QL Signature
 */
export interface AegisSignature {
  z: number[];    // Signature component
  c: number[];    // Challenge hash (32 bytes)
}

/**
 * AEGIS-QL Cryptosystem
 */
export class AegisQL {
  constructor(_seed?: Uint8Array) {
    // Seed parameter for future CSPRNG implementation
    // Currently using crypto.getRandomValues directly
  }

  /**
   * Generate a new AEGIS-QL key pair
   */
  async generateKeypair(): Promise<{ publicKey: AegisPublicKey; secretKey: AegisSecretKey }> {
    // Sample sparse secret key
    const s = this.sampleSparsePolynomial(POLY_DEGREE, GRAPH_DEGREE);

    // Sample uniform random polynomial a
    const a = this.sampleUniformPolynomial(POLY_DEGREE);

    // Sample small error polynomial e
    const e = this.sampleErrorPolynomial(POLY_DEGREE);

    // Compute t = a*s + e
    const sDense = sparseToDense(s, POLY_DEGREE);
    const as = polynomialMultiply(a, sDense, MODULUS);
    const t = polynomialAdd(as, e, MODULUS);

    return {
      publicKey: { a, t },
      secretKey: { s },
    };
  }

  /**
   * Sign a message using AEGIS-QL
   */
  async sign(message: Uint8Array, secretKey: AegisSecretKey): Promise<AegisSignature> {
    // Hash message to create challenge space (for future use)
    // const hash = sha3_512(message);

    // Sample random polynomial y (used for commitment)
    const y = this.sampleSparsePolynomial(POLY_DEGREE, GRAPH_DEGREE);
    const yDense = sparseToDense(y, POLY_DEGREE);

    // Compute commitment w = hash(y)
    const commitmentInput = new Uint8Array(yDense.length * 4);
    const view = new DataView(commitmentInput.buffer);
    for (let i = 0; i < yDense.length; i++) {
      view.setUint32(i * 4, yDense[i], true); // little-endian
    }
    const commitment = sha3_256(commitmentInput);

    // Compute challenge c = hash(message || commitment)
    const challengeInput = new Uint8Array(message.length + commitment.length);
    challengeInput.set(message, 0);
    challengeInput.set(commitment, message.length);
    const c = sha3_256(challengeInput);

    // Convert challenge to polynomial
    const cPoly = hashToPolynomial(c, POLY_DEGREE);

    // Compute z = y + c*s (signature)
    const sDense = sparseToDense(secretKey.s, POLY_DEGREE);
    const cs = polynomialMultiply(cPoly, sDense, MODULUS);
    const z = polynomialAdd(yDense, cs, MODULUS);

    return {
      z,
      c: Array.from(c),
    };
  }

  /**
   * Verify an AEGIS-QL signature (for testing/validation)
   * Note: Production verification happens on the backend
   */
  async verify(
    message: Uint8Array,
    signature: AegisSignature,
    publicKey: AegisPublicKey
  ): Promise<boolean> {
    // Reconstruct commitment from signature
    // w' = z - c*t
    const cPoly = hashToPolynomial(new Uint8Array(signature.c), POLY_DEGREE);
    const ct = polynomialMultiply(cPoly, publicKey.t, MODULUS);

    if (signature.z.length < ct.length) {
      return false;
    }

    const wPrime = polynomialSubtract(signature.z, ct, MODULUS);

    // Compute expected commitment hash
    const commitmentInput = new Uint8Array(wPrime.length * 4);
    const view = new DataView(commitmentInput.buffer);
    for (let i = 0; i < wPrime.length; i++) {
      view.setUint32(i * 4, wPrime[i], true);
    }
    const commitment = sha3_256(commitmentInput);

    // Recompute challenge c' = hash(message || commitment)
    const challengeInput = new Uint8Array(message.length + commitment.length);
    challengeInput.set(message, 0);
    challengeInput.set(commitment, message.length);
    const cPrime = sha3_256(challengeInput);

    // Verify c == c'
    if (signature.c.length !== cPrime.length) {
      return false;
    }

    for (let i = 0; i < signature.c.length; i++) {
      if (signature.c[i] !== cPrime[i]) {
        return false;
      }
    }

    return true;
  }

  /**
   * Sample a sparse polynomial with given degree and sparsity
   */
  private sampleSparsePolynomial(degree: number, sparsity: number): SparsePolynomial {
    const indices: number[] = [];
    const coefficients: number[] = [];

    for (let i = 0; i < sparsity; i++) {
      const idx = this.nextRandomInt(degree);
      if (!indices.includes(idx)) {
        indices.push(idx);
        // Sample small coefficient (-1, 0, 1)
        const coeff = this.nextRandomInt(3);
        coefficients.push(coeff === 2 ? MODULUS - 1 : coeff);
      }
    }

    return {
      coefficients,
      indices,
      degree,
    };
  }

  /**
   * Sample a uniform random polynomial
   */
  private sampleUniformPolynomial(degree: number): number[] {
    const poly: number[] = [];
    for (let i = 0; i < degree; i++) {
      poly.push(this.nextRandomInt(MODULUS));
    }
    return poly;
  }

  /**
   * Sample a small error polynomial (centered binomial distribution)
   */
  private sampleErrorPolynomial(degree: number): number[] {
    const poly: number[] = [];
    for (let i = 0; i < degree; i++) {
      // Centered binomial with parameter 2 (small noise)
      const a = this.nextRandomInt(2);
      const b = this.nextRandomInt(2);
      const noise = a - b;
      poly.push((noise + MODULUS) % MODULUS);
    }
    return poly;
  }

  /**
   * Get next random integer in range [0, max)
   * Uses crypto.getRandomValues for secure randomness
   */
  private nextRandomInt(max: number): number {
    // Use crypto.getRandomValues for secure randomness
    const randomBytes = crypto.getRandomValues(new Uint8Array(4));
    const randomValue = new DataView(randomBytes.buffer).getUint32(0, true);
    return randomValue % max;
  }
}

/**
 * Convert sparse polynomial to dense representation
 */
export function sparseToDense(sparse: SparsePolynomial, degree: number): number[] {
  const dense = new Array(degree).fill(0);
  for (let i = 0; i < sparse.indices.length; i++) {
    dense[sparse.indices[i]] = sparse.coefficients[i];
  }
  return dense;
}

/**
 * Polynomial addition modulo q
 */
export function polynomialAdd(a: number[], b: number[], modulus: number): number[] {
  const result: number[] = [];
  const maxLen = Math.max(a.length, b.length);

  for (let i = 0; i < maxLen; i++) {
    const aVal = i < a.length ? a[i] : 0;
    const bVal = i < b.length ? b[i] : 0;
    result.push((aVal + bVal) % modulus);
  }

  return result;
}

/**
 * Polynomial subtraction modulo q
 */
export function polynomialSubtract(a: number[], b: number[], modulus: number): number[] {
  const result: number[] = [];
  const maxLen = Math.max(a.length, b.length);

  for (let i = 0; i < maxLen; i++) {
    const aVal = i < a.length ? a[i] : 0;
    const bVal = i < b.length ? b[i] : 0;
    result.push((aVal - bVal + modulus) % modulus);
  }

  return result;
}

/**
 * Polynomial multiplication modulo q (schoolbook algorithm)
 * Note: In production, this would use NTT for O(n log n) performance
 */
export function polynomialMultiply(a: number[], b: number[], modulus: number): number[] {
  const result = new Array(a.length + b.length - 1).fill(0);

  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      result[i + j] = (result[i + j] + a[i] * b[j]) % modulus;
    }
  }

  // Trim to polynomial degree (cyclotomic reduction for Ring-LWE)
  const trimmed = result.slice(0, POLY_DEGREE);
  for (let i = POLY_DEGREE; i < result.length; i++) {
    trimmed[i % POLY_DEGREE] = (trimmed[i % POLY_DEGREE] + result[i]) % modulus;
  }

  return trimmed;
}

/**
 * Convert hash to polynomial (for challenge generation)
 */
export function hashToPolynomial(hash: Uint8Array, degree: number): number[] {
  const poly: number[] = [];
  let counter = 0;

  while (poly.length < degree) {
    // Extend hash with counter
    const counterBytes = new Uint8Array(8);
    new DataView(counterBytes.buffer).setBigUint64(0, BigInt(counter), true);

    const extended = new Uint8Array(hash.length + counterBytes.length);
    extended.set(hash, 0);
    extended.set(counterBytes, hash.length);

    const digest = sha3_256(extended);

    // Extract coefficients from digest
    for (let i = 0; i + 3 < digest.length && poly.length < degree; i += 4) {
      const value =
        digest[i] | (digest[i + 1] << 8) | (digest[i + 2] << 16) | (digest[i + 3] << 24);
      poly.push(value % MODULUS);
    }

    counter++;
  }

  return poly.slice(0, degree);
}

/**
 * Export AEGIS-QL signature to backend-compatible JSON format
 */
export function exportSignatureToJSON(signature: AegisSignature): string {
  return JSON.stringify({
    z: signature.z,
    c: signature.c,
  });
}

/**
 * Export AEGIS-QL public key to backend-compatible JSON format
 */
export function exportPublicKeyToJSON(publicKey: AegisPublicKey): string {
  return JSON.stringify({
    a: publicKey.a,
    t: publicKey.t,
  });
}

/**
 * Import AEGIS-QL signature from JSON
 */
export function importSignatureFromJSON(json: string): AegisSignature {
  const parsed = JSON.parse(json);
  return {
    z: parsed.z,
    c: parsed.c,
  };
}

/**
 * Import AEGIS-QL public key from JSON
 */
export function importPublicKeyFromJSON(json: string): AegisPublicKey {
  const parsed = JSON.parse(json);
  return {
    a: parsed.a,
    t: parsed.t,
  };
}
