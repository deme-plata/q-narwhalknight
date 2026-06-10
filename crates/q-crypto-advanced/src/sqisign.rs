//! SQIsign: Compact Isogeny-Based Signatures
//!
//! Based on: "SQIsign2D-West: The Full Story" (IACR 2025/847)
//! and "SQIsignHD: New Dimensions in Cryptography" (Eurocrypt 2024)
//!
//! SQIsign is a post-quantum signature scheme based on supersingular isogenies.
//! It achieves the smallest signature sizes among PQ signatures (~204 bytes for
//! NIST Level I security).
//!
//! ## Security Properties
//! - **Post-quantum secure**: Based on hardness of computing isogenies between
//!   supersingular elliptic curves
//! - **Smallest PQ signatures**: 204 bytes at NIST Level I
//! - **Compact public keys**: ~64 bytes
//!
//! ## Performance Characteristics (NIST Level I)
//! - Public key: 64 bytes
//! - Signature: 204 bytes
//! - Key generation: ~1s (with precomputation)
//! - Signing: ~500ms
//! - Verification: ~50ms
//!
//! ## Comparison with Other PQ Signatures
//! | Scheme      | PK size | Sig size | Notes             |
//! |-------------|---------|----------|-------------------|
//! | SQIsign     | 64 B    | 204 B    | Smallest sigs     |
//! | Dilithium3  | 1,952 B | 3,293 B  | NIST standard     |
//! | Falcon-512  | 897 B   | 690 B    | Fast, small       |
//! | SPHINCS+    | 32 B    | 7,856 B  | Hash-based        |

use crate::errors::CryptoError;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256, Sha3_512};
use zeroize::Zeroize;

// When sqisign-ffi feature is enabled, delegate to the real C reference implementation
#[cfg(feature = "sqisign-ffi")]
use q_sqisign;

/// Security level for SQIsign
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SqiSignLevel {
    /// NIST Level I (128-bit classical, 64-bit quantum)
    Level1,
    /// NIST Level III (192-bit classical, 128-bit quantum)
    Level3,
    /// NIST Level V (256-bit classical, 128-bit quantum)
    Level5,
}

impl Default for SqiSignLevel {
    fn default() -> Self {
        SqiSignLevel::Level1
    }
}

/// Parameters for SQIsign based on security level
#[derive(Clone, Debug)]
pub struct SqiSignParams {
    /// Security level
    pub level: SqiSignLevel,
    /// Prime characteristic p (p = f * 2^e₂ * 3^e₃ - 1)
    pub p_bits: usize,
    /// Exponent of 2 in cofactor
    pub e2: usize,
    /// Exponent of 3 in cofactor
    pub e3: usize,
    /// Public key size in bytes
    pub pk_size: usize,
    /// Signature size in bytes
    pub sig_size: usize,
}

impl SqiSignParams {
    /// NIST Level I parameters
    pub fn level_1() -> Self {
        Self {
            level: SqiSignLevel::Level1,
            p_bits: 256,
            e2: 126,
            e3: 80,
            pk_size: 64,
            sig_size: 204,
        }
    }

    /// NIST Level III parameters
    pub fn level_3() -> Self {
        Self {
            level: SqiSignLevel::Level3,
            p_bits: 384,
            e2: 190,
            e3: 120,
            pk_size: 96,
            sig_size: 306,
        }
    }

    /// NIST Level V parameters
    pub fn level_5() -> Self {
        Self {
            level: SqiSignLevel::Level5,
            p_bits: 512,
            e2: 254,
            e3: 160,
            pk_size: 128,
            sig_size: 408,
        }
    }

    /// Get params for a security level
    pub fn for_level(level: SqiSignLevel) -> Self {
        match level {
            SqiSignLevel::Level1 => Self::level_1(),
            SqiSignLevel::Level3 => Self::level_3(),
            SqiSignLevel::Level5 => Self::level_5(),
        }
    }
}

/// Field element in F_p²
/// Represented as a + b*i where i² = -1
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fp2Element {
    /// Real component
    pub a: Vec<u8>,
    /// Imaginary component
    pub b: Vec<u8>,
}

impl Fp2Element {
    /// Create a new Fp2 element
    pub fn new(a: Vec<u8>, b: Vec<u8>) -> Self {
        Self { a, b }
    }

    /// Create zero element
    pub fn zero(size: usize) -> Self {
        Self {
            a: vec![0u8; size],
            b: vec![0u8; size],
        }
    }

    /// Create one element (1 + 0*i)
    pub fn one(size: usize) -> Self {
        let mut a = vec![0u8; size];
        if size > 0 {
            a[0] = 1;
        }
        Self {
            a,
            b: vec![0u8; size],
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.a.len() + self.b.len());
        bytes.extend(&self.a);
        bytes.extend(&self.b);
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8], element_size: usize) -> Result<Self, CryptoError> {
        if bytes.len() != 2 * element_size {
            return Err(CryptoError::DeserializationError(
                "Invalid Fp2 element size".into(),
            ));
        }
        Ok(Self {
            a: bytes[..element_size].to_vec(),
            b: bytes[element_size..].to_vec(),
        })
    }
}

/// Point on a supersingular elliptic curve over Fp²
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CurvePoint {
    /// x-coordinate
    pub x: Fp2Element,
    /// y-coordinate
    pub y: Fp2Element,
    /// Is this the point at infinity?
    pub is_infinity: bool,
}

impl CurvePoint {
    /// Point at infinity (identity)
    pub fn infinity(size: usize) -> Self {
        Self {
            x: Fp2Element::zero(size),
            y: Fp2Element::zero(size),
            is_infinity: true,
        }
    }

    /// Create a new point
    pub fn new(x: Fp2Element, y: Fp2Element) -> Self {
        Self {
            x,
            y,
            is_infinity: false,
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        if self.is_infinity {
            vec![0u8] // Compressed infinity
        } else {
            let mut bytes = vec![1u8]; // Not infinity flag
            bytes.extend(self.x.to_bytes());
            bytes.extend(self.y.to_bytes());
            bytes
        }
    }
}

/// Supersingular elliptic curve E: y² = x³ + ax + b over Fp²
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SupersingularCurve {
    /// Coefficient a (usually 0 for Montgomery form)
    pub a: Fp2Element,
    /// Coefficient b (usually 1)
    pub b: Fp2Element,
    /// j-invariant (uniquely identifies the curve up to isomorphism)
    pub j_invariant: Fp2Element,
}

impl SupersingularCurve {
    /// Create a new curve with given parameters
    pub fn new(a: Fp2Element, b: Fp2Element, j_invariant: Fp2Element) -> Self {
        Self { a, b, j_invariant }
    }

    /// Serialize to bytes (j-invariant is sufficient to identify the curve)
    pub fn to_bytes(&self) -> Vec<u8> {
        self.j_invariant.to_bytes()
    }
}

/// Isogeny between supersingular curves (compact representation)
/// An isogeny φ: E1 → E2 is represented by its kernel generator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Isogeny {
    /// Kernel generator point
    pub kernel: CurvePoint,
    /// Degree of the isogeny (log₂ for 2-isogenies, log₃ for 3-isogenies)
    pub degree_log: u32,
    /// Whether this uses 2-isogenies or 3-isogenies
    pub is_two_isogeny: bool,
}

impl Isogeny {
    /// Serialize to compact bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.kernel.to_bytes();
        bytes.extend_from_slice(&self.degree_log.to_le_bytes());
        bytes.push(self.is_two_isogeny as u8);
        bytes
    }
}

/// SQIsign public key
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SqiSignPublicKey {
    /// Public curve EA (the codomain of the secret isogeny from E0)
    pub curve: SupersingularCurve,
    /// Security level
    pub level: SqiSignLevel,
    /// Compressed representation
    pub compressed: Vec<u8>,
}

impl SqiSignPublicKey {
    /// Create a new public key
    pub fn new(curve: SupersingularCurve, level: SqiSignLevel) -> Self {
        let compressed = curve.to_bytes();
        Self {
            curve,
            level,
            compressed,
        }
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        self.compressed.len()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![self.level as u8];
        bytes.extend(&self.compressed);
        bytes
    }

    /// Compute the key hash
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.to_bytes());
        hasher.finalize().into()
    }
}

/// SQIsign secret key
#[derive(Clone)]
pub struct SqiSignSecretKey {
    /// Secret isogeny τ: E0 → EA
    secret_isogeny: Vec<u8>,
    /// Auxiliary data for efficient signing
    aux_data: Vec<u8>,
    /// Security level
    level: SqiSignLevel,
}

impl Zeroize for SqiSignSecretKey {
    fn zeroize(&mut self) {
        self.secret_isogeny.zeroize();
        self.aux_data.zeroize();
    }
}

impl Drop for SqiSignSecretKey {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// SQIsign signature
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SqiSignature {
    /// Response isogeny (compact representation)
    pub response: Vec<u8>,
    /// Commitment hash
    pub commitment: [u8; 32],
    /// Security level
    pub level: SqiSignLevel,
}

impl SqiSignature {
    /// Get signature size in bytes
    pub fn size(&self) -> usize {
        self.response.len() + 32 + 1
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![self.level as u8];
        bytes.extend(&self.commitment);
        bytes.extend(&self.response);
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() < 34 {
            return Err(CryptoError::DeserializationError(
                "Signature too short".into(),
            ));
        }

        let level = match bytes[0] {
            0 => SqiSignLevel::Level1,
            1 => SqiSignLevel::Level3,
            2 => SqiSignLevel::Level5,
            _ => {
                return Err(CryptoError::DeserializationError(
                    "Invalid security level".into(),
                ))
            }
        };

        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&bytes[1..33]);
        let response = bytes[33..].to_vec();

        Ok(Self {
            response,
            commitment,
            level,
        })
    }
}

/// SQIsign key pair
pub struct SqiSignKeyPair {
    /// Public key
    pub public_key: SqiSignPublicKey,
    /// Secret key
    secret_key: SqiSignSecretKey,
    /// Parameters
    params: SqiSignParams,
    /// When sqisign-ffi is enabled, holds the real C library keypair for signing
    #[cfg(feature = "sqisign-ffi")]
    ffi_keypair: Option<q_sqisign::KeyPair>,
}

impl SqiSignKeyPair {
    /// Generate a new key pair.
    /// When the `sqisign-ffi` feature is enabled and level is Level1,
    /// uses the real C reference implementation via FFI.
    pub fn generate(level: SqiSignLevel) -> Result<Self, CryptoError> {
        // Use real C implementation for Level 1 when FFI is available
        #[cfg(feature = "sqisign-ffi")]
        if level == SqiSignLevel::Level1 {
            return Self::generate_ffi();
        }

        // Hash-based fallback (for non-Level1 or when FFI is disabled)
        let mut seed = vec![0u8; 32];
        getrandom::getrandom(&mut seed).map_err(|_| CryptoError::RngFailed)?;
        Self::from_seed(&seed, level)
    }

    /// Generate a Level 1 keypair using the real SQIsign C reference implementation.
    #[cfg(feature = "sqisign-ffi")]
    fn generate_ffi() -> Result<Self, CryptoError> {
        let params = SqiSignParams::for_level(SqiSignLevel::Level1);
        let ffi_kp = q_sqisign::KeyPair::generate()
            .map_err(|e| CryptoError::KeyGenFailed(format!("SQIsign FFI: {}", e)))?;

        let pk_bytes = ffi_kp.public_key().to_vec();
        // Store raw FFI public key bytes in the scaffold's compressed field.
        // The j_invariant is derived from pk bytes for type compatibility.
        let half = pk_bytes.len() / 2;
        let j_invariant = Fp2Element::new(
            pk_bytes[..half].to_vec(),
            pk_bytes[half..].to_vec(),
        );
        let element_size = params.p_bits / 8;
        let curve = SupersingularCurve::new(
            Fp2Element::zero(element_size / 2),
            Fp2Element::one(element_size / 2),
            j_invariant,
        );
        let public_key = SqiSignPublicKey {
            curve,
            level: SqiSignLevel::Level1,
            compressed: pk_bytes,
        };
        // Secret key bytes are held inside ffi_keypair; scaffold field is a placeholder.
        let secret_key = SqiSignSecretKey {
            secret_isogeny: Vec::new(),
            aux_data: Vec::new(),
            level: SqiSignLevel::Level1,
        };
        Ok(Self {
            public_key,
            secret_key,
            params,
            ffi_keypair: Some(ffi_kp),
        })
    }

    /// Generate key pair from seed (deterministic)
    pub fn from_seed(seed: &[u8], level: SqiSignLevel) -> Result<Self, CryptoError> {
        let params = SqiSignParams::for_level(level);

        // Expand seed using SHAKE256
        let mut hasher = Sha3_512::new();
        hasher.update(seed);
        hasher.update(b"sqisign-keygen");
        hasher.update(&[level as u8]);
        let expanded: [u8; 64] = hasher.finalize().into();

        // Generate secret isogeny representation
        // In full implementation: compute random degree-d isogeny from base curve E0
        let secret_isogeny = expanded[..32].to_vec();

        // Compute auxiliary data for efficient signing
        let aux_data = expanded[32..].to_vec();

        // Compute public curve EA = τ(E0)
        // In full implementation: apply isogeny to base curve
        let element_size = params.p_bits / 8;
        let j_invariant = Fp2Element::new(
            expanded[..element_size / 2].to_vec(),
            expanded[element_size / 2..element_size].to_vec(),
        );

        let curve = SupersingularCurve::new(
            Fp2Element::zero(element_size / 2),
            Fp2Element::one(element_size / 2),
            j_invariant,
        );

        let public_key = SqiSignPublicKey::new(curve, level);
        let secret_key = SqiSignSecretKey {
            secret_isogeny,
            aux_data,
            level,
        };

        Ok(Self {
            public_key,
            secret_key,
            params,
            #[cfg(feature = "sqisign-ffi")]
            ffi_keypair: None,
        })
    }

    /// Get the public key
    pub fn public_key(&self) -> &SqiSignPublicKey {
        &self.public_key
    }

    /// Sign a message.
    /// When `sqisign-ffi` is enabled and the keypair was generated via FFI,
    /// uses the real C implementation. Otherwise falls back to hash-based.
    pub fn sign(&self, message: &[u8]) -> Result<SqiSignature, CryptoError> {
        // Delegate to real C implementation if FFI keypair is available
        #[cfg(feature = "sqisign-ffi")]
        if let Some(ref ffi_kp) = self.ffi_keypair {
            let sig_bytes = ffi_kp.sign(message)
                .map_err(|e| CryptoError::SigningFailed(format!("SQIsign FFI: {}", e)))?;

            // Store raw C signature in `response`; commitment is a hash for identification
            let mut hasher = Sha3_256::new();
            hasher.update(&sig_bytes);
            let commitment: [u8; 32] = hasher.finalize().into();

            return Ok(SqiSignature {
                response: sig_bytes,
                commitment,
                level: self.secret_key.level,
            });
        }

        // Hash-based fallback (existing scaffold implementation)
        let mut commitment_seed = vec![0u8; 32];
        getrandom::getrandom(&mut commitment_seed).map_err(|_| CryptoError::RngFailed)?;

        let mut hasher = Sha3_256::new();
        hasher.update(&commitment_seed);
        hasher.update(message);
        hasher.update(&self.public_key.compressed);
        let commitment: [u8; 32] = hasher.finalize().into();

        let response_size = self.params.sig_size - 32 - 1;
        let mut response = Vec::with_capacity(response_size);
        let mut counter = 0u32;

        while response.len() < response_size {
            let mut response_hasher = Sha3_512::new();
            response_hasher.update(&self.secret_key.secret_isogeny);
            response_hasher.update(&commitment);
            response_hasher.update(&self.secret_key.aux_data);
            response_hasher.update(&counter.to_le_bytes());
            let hash: [u8; 64] = response_hasher.finalize().into();

            let take = std::cmp::min(64, response_size - response.len());
            response.extend_from_slice(&hash[..take]);
            counter += 1;
        }

        Ok(SqiSignature {
            response,
            commitment,
            level: self.secret_key.level,
        })
    }
}

/// SQIsign verifier
pub struct SqiSignVerifier {
    params: SqiSignParams,
}

impl SqiSignVerifier {
    /// Create a new verifier for the given security level
    pub fn new(level: SqiSignLevel) -> Self {
        Self {
            params: SqiSignParams::for_level(level),
        }
    }

    /// Verify a signature.
    /// When `sqisign-ffi` is enabled and the signature is Level1,
    /// uses the real C reference implementation for cryptographic verification.
    pub fn verify(
        &self,
        public_key: &SqiSignPublicKey,
        message: &[u8],
        signature: &SqiSignature,
    ) -> Result<bool, CryptoError> {
        // Verify security levels match
        if public_key.level != signature.level {
            return Ok(false);
        }

        // Use real C verification for Level 1 when FFI is available
        #[cfg(feature = "sqisign-ffi")]
        if signature.level == SqiSignLevel::Level1 {
            // FFI signatures store the raw C output in `response`
            return q_sqisign::verify(
                &public_key.compressed,
                message,
                &signature.response,
            ).map_err(|e| CryptoError::InternalError(format!("SQIsign FFI verify: {}", e)));
        }

        // v10.3.0: SCAFFOLD REMOVED — was returning Ok(true) for all signatures.
        // That was a time bomb: any code path routing consensus verification through
        // the scaffold would silently accept all signatures.
        //
        // The scaffold is now a hard error. To verify real SQIsign signatures,
        // enable the 'sqisign-ffi' feature to link the C reference implementation.
        //
        // See: metzdowd cryptography mailing list, April 2026
        // "A verify function that returns Ok(true) is a time bomb"
        Err(CryptoError::InternalError(
            "SQIsign scaffold verification is disabled (v10.3.0). \
             The hash-based fallback previously returned Ok(true) for ALL signatures — \
             a security vulnerability if used in consensus. \
             Enable 'sqisign-ffi' feature to link the C reference implementation \
             for real isogeny-based verification.".to_string()
        ))
    }
}

/// Batch verifier for SQIsign signatures
pub struct SqiSignBatchVerifier {
    pending: Vec<(SqiSignPublicKey, Vec<u8>, SqiSignature)>,
    level: SqiSignLevel,
}

impl SqiSignBatchVerifier {
    /// Create a new batch verifier
    pub fn new(level: SqiSignLevel) -> Self {
        Self {
            pending: Vec::new(),
            level,
        }
    }

    /// Add a signature to verify
    pub fn add(
        &mut self,
        public_key: SqiSignPublicKey,
        message: Vec<u8>,
        signature: SqiSignature,
    ) {
        self.pending.push((public_key, message, signature));
    }

    /// Verify all pending signatures
    pub fn verify_all(&self) -> Result<Vec<bool>, CryptoError> {
        let verifier = SqiSignVerifier::new(self.level);
        let mut results = Vec::with_capacity(self.pending.len());

        for (pk, msg, sig) in &self.pending {
            results.push(verifier.verify(pk, msg, sig)?);
        }

        Ok(results)
    }

    /// Clear pending signatures
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

/// Aggregate multiple SQIsign signatures (same message)
pub struct SqiSignAggregator {
    level: SqiSignLevel,
    signatures: Vec<(SqiSignPublicKey, SqiSignature)>,
    message: Option<Vec<u8>>,
}

impl SqiSignAggregator {
    /// Create a new aggregator
    pub fn new(level: SqiSignLevel) -> Self {
        Self {
            level,
            signatures: Vec::new(),
            message: None,
        }
    }

    /// Add a signature
    pub fn add(
        &mut self,
        public_key: SqiSignPublicKey,
        message: &[u8],
        signature: SqiSignature,
    ) -> Result<(), CryptoError> {
        if let Some(ref existing_msg) = self.message {
            if existing_msg != message {
                return Err(CryptoError::InvalidParameters(
                    "All signatures must be on the same message".into(),
                ));
            }
        } else {
            self.message = Some(message.to_vec());
        }

        self.signatures.push((public_key, signature));
        Ok(())
    }

    /// Aggregate all signatures into a single proof
    pub fn aggregate(&self) -> Result<AggregatedSqiSign, CryptoError> {
        if self.signatures.is_empty() {
            return Err(CryptoError::InvalidParameters(
                "No signatures to aggregate".into(),
            ));
        }

        // Collect all components
        let mut public_keys = Vec::new();
        let mut responses = Vec::new();
        let mut commitment_aggregate = [0u8; 32];

        for (pk, sig) in &self.signatures {
            public_keys.push(pk.clone());
            responses.extend(&sig.response);

            // XOR commitments for aggregate
            for (i, b) in sig.commitment.iter().enumerate() {
                commitment_aggregate[i] ^= b;
            }
        }

        let message = self.message.clone().unwrap_or_default();

        Ok(AggregatedSqiSign {
            public_keys,
            responses,
            commitment_aggregate,
            message,
            count: self.signatures.len() as u32,
            level: self.level,
        })
    }
}

/// Aggregated SQIsign proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregatedSqiSign {
    /// All public keys
    pub public_keys: Vec<SqiSignPublicKey>,
    /// Concatenated responses
    pub responses: Vec<u8>,
    /// Aggregated commitment
    pub commitment_aggregate: [u8; 32],
    /// The signed message
    pub message: Vec<u8>,
    /// Number of signatures
    pub count: u32,
    /// Security level
    pub level: SqiSignLevel,
}

impl AggregatedSqiSign {
    /// Get the total size in bytes
    pub fn size(&self) -> usize {
        let params = SqiSignParams::for_level(self.level);
        (self.count as usize) * params.pk_size + self.responses.len() + 32 + 4 + 1
    }

    /// Verify the aggregated signature
    pub fn verify(&self) -> Result<bool, CryptoError> {
        let verifier = SqiSignVerifier::new(self.level);
        let params = SqiSignParams::for_level(self.level);
        // Compute per-signature response size from actual data (handles both
        // FFI signatures at 148 bytes and hash-based at sig_size-33).
        let response_size = if self.count > 0 {
            self.responses.len() / self.count as usize
        } else {
            params.sig_size - 32 - 1
        };

        // Verify each individual signature
        for (i, pk) in self.public_keys.iter().enumerate() {
            let start = i * response_size;
            let end = start + response_size;

            if end > self.responses.len() {
                return Ok(false);
            }

            let response = self.responses[start..end].to_vec();
            let sig = SqiSignature {
                response,
                commitment: self.commitment_aggregate, // Simplified
                level: self.level,
            };

            if !verifier.verify(pk, &self.message, &sig)? {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params() {
        let params1 = SqiSignParams::level_1();
        assert_eq!(params1.pk_size, 64);
        assert_eq!(params1.sig_size, 204);

        let params3 = SqiSignParams::level_3();
        assert_eq!(params3.pk_size, 96);
        assert_eq!(params3.sig_size, 306);
    }

    #[test]
    fn test_fp2_element() {
        let elem = Fp2Element::new(vec![1, 2, 3, 4], vec![5, 6, 7, 8]);
        let bytes = elem.to_bytes();
        assert_eq!(bytes.len(), 8);

        let recovered = Fp2Element::from_bytes(&bytes, 4).unwrap();
        assert_eq!(recovered.a, vec![1, 2, 3, 4]);
        assert_eq!(recovered.b, vec![5, 6, 7, 8]);
    }

    #[test]
    fn test_keygen() {
        let keypair = SqiSignKeyPair::generate(SqiSignLevel::Level1).unwrap();
        assert_eq!(keypair.public_key.level, SqiSignLevel::Level1);
    }

    #[test]
    fn test_deterministic_keygen() {
        let seed = b"test seed for sqisign keygen!!!";

        let kp1 = SqiSignKeyPair::from_seed(seed, SqiSignLevel::Level1).unwrap();
        let kp2 = SqiSignKeyPair::from_seed(seed, SqiSignLevel::Level1).unwrap();

        assert_eq!(
            kp1.public_key.compressed,
            kp2.public_key.compressed
        );
    }

    #[test]
    fn test_sign() {
        let keypair = SqiSignKeyPair::generate(SqiSignLevel::Level1).unwrap();
        let message = b"test message for sqisign";

        let signature = keypair.sign(message).unwrap();
        assert_eq!(signature.level, SqiSignLevel::Level1);
        assert!(!signature.response.is_empty());
    }

    #[test]
    fn test_verify() {
        let keypair = SqiSignKeyPair::generate(SqiSignLevel::Level1).unwrap();
        let message = b"test verification message";

        let signature = keypair.sign(message).unwrap();
        let verifier = SqiSignVerifier::new(SqiSignLevel::Level1);

        let valid = verifier.verify(&keypair.public_key, message, &signature).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_signature_serialization() {
        let keypair = SqiSignKeyPair::generate(SqiSignLevel::Level1).unwrap();
        let message = b"serialize test";

        let signature = keypair.sign(message).unwrap();
        let bytes = signature.to_bytes();
        let recovered = SqiSignature::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.commitment, signature.commitment);
        assert_eq!(recovered.level, signature.level);
    }

    #[test]
    fn test_batch_verify() {
        let mut batch = SqiSignBatchVerifier::new(SqiSignLevel::Level1);

        // Generate multiple signatures
        for i in 0..3 {
            let keypair = SqiSignKeyPair::generate(SqiSignLevel::Level1).unwrap();
            let message = format!("message {}", i);
            let signature = keypair.sign(message.as_bytes()).unwrap();

            batch.add(keypair.public_key, message.into_bytes(), signature);
        }

        let results = batch.verify_all().unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|&r| r));
    }

    #[test]
    fn test_aggregation() {
        let message = b"common message for aggregation";
        let mut aggregator = SqiSignAggregator::new(SqiSignLevel::Level1);

        // Add multiple signatures on the same message
        for _ in 0..3 {
            let keypair = SqiSignKeyPair::generate(SqiSignLevel::Level1).unwrap();
            let signature = keypair.sign(message).unwrap();
            aggregator.add(keypair.public_key, message, signature).unwrap();
        }

        let aggregated = aggregator.aggregate().unwrap();
        assert_eq!(aggregated.count, 3);

        let valid = aggregated.verify().unwrap();
        assert!(valid);
    }

    #[test]
    fn test_signature_sizes() {
        // Verify we achieve the target signature sizes
        let params = SqiSignParams::level_1();

        let keypair = SqiSignKeyPair::generate(SqiSignLevel::Level1).unwrap();
        let signature = keypair.sign(b"size test").unwrap();

        // The signature should be close to the expected size
        // (Our simplified version may differ slightly)
        assert!(signature.size() <= params.sig_size + 10);
    }
}
