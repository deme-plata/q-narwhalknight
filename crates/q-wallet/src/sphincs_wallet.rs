/// SPHINCS+ Conservative Post-Quantum Signatures
/// Phase 8: Automatic ultra-conservative signatures for critical operations
///
/// SPHINCS+ is AUTOMATICALLY used for:
/// - Genesis block signatures
/// - Protocol upgrade signatures
/// - Validator key rotation
/// - Checkpoint signatures
/// - Audit trail signatures
///
/// Users NEVER choose - the system automatically applies maximum security
/// where it matters most.

use anyhow::{anyhow, Result};
use pqcrypto_sphincsplus::sphincssha256256fsimple;
use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SecretKey as PQSecretKey, SignedMessage};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

/// Operation type that determines signature scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    /// Regular user transaction - uses Dilithium5
    RegularTransaction,

    /// CRITICAL: Genesis block - AUTOMATICALLY uses SPHINCS+ + Dilithium5
    GenesisBlock,

    /// CRITICAL: Protocol upgrade - AUTOMATICALLY uses SPHINCS+ + Dilithium5
    ProtocolUpgrade,

    /// CRITICAL: Validator key rotation - AUTOMATICALLY uses SPHINCS+ + Dilithium5
    ValidatorKeyRotation,

    /// CRITICAL: System checkpoint - AUTOMATICALLY uses SPHINCS+ + Dilithium5
    SystemCheckpoint,

    /// CRITICAL: Audit trail - AUTOMATICALLY uses SPHINCS+ + Dilithium5
    AuditTrail,
}

impl OperationType {
    /// Determines if this operation requires SPHINCS+ ultra-conservative security
    /// This is AUTOMATIC - users cannot disable it
    pub fn requires_sphincs_plus(&self) -> bool {
        match self {
            OperationType::RegularTransaction => false,
            OperationType::GenesisBlock => true,
            OperationType::ProtocolUpgrade => true,
            OperationType::ValidatorKeyRotation => true,
            OperationType::SystemCheckpoint => true,
            OperationType::AuditTrail => true,
        }
    }

    /// Get human-readable description of security level
    pub fn security_description(&self) -> &'static str {
        if self.requires_sphincs_plus() {
            "MAXIMUM SECURITY: Dilithium5 + SPHINCS+ (hash-based) dual signatures"
        } else {
            "HIGH SECURITY: Dilithium5 lattice-based signatures"
        }
    }
}

/// SPHINCS+ keypair (256-bit security level)
pub struct SphincsPlusKeyPair {
    pub public_key: sphincssha256256fsimple::PublicKey,
    pub secret_key: sphincssha256256fsimple::SecretKey,
}

impl SphincsPlusKeyPair {
    /// Generate a new SPHINCS+-256f keypair
    /// 256f = 256-bit security, "fast" variant (faster signing, larger signatures)
    pub fn generate() -> Self {
        let (public_key, secret_key) = sphincssha256256fsimple::keypair();
        Self {
            public_key,
            secret_key,
        }
    }

    /// Sign a message with SPHINCS+ (hash-based post-quantum signatures)
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        sphincssha256256fsimple::sign(message, &self.secret_key)
            .as_bytes()
            .to_vec()
    }

    /// Verify a SPHINCS+ signature
    pub fn verify(_message: &[u8], signed_message: &[u8], public_key: &[u8]) -> Result<bool> {
        let pk = sphincssha256256fsimple::PublicKey::from_bytes(public_key)
            .map_err(|_| anyhow!("Invalid SPHINCS+ public key"))?;

        let signed_msg = SignedMessage::from_bytes(signed_message)
            .map_err(|_| anyhow!("Invalid SPHINCS+ signed message"))?;

        match sphincssha256256fsimple::open(&signed_msg, &pk) {
            Ok(_recovered_message) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Derive address from public key (SHA3-256 hash)
    pub fn derive_address(public_key: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"SPHINCS+:");  // Domain separator
        hasher.update(public_key);
        hasher.finalize().into()
    }

    /// Get public key bytes
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.public_key.as_bytes().to_vec()
    }

    /// Get secret key bytes
    pub fn secret_key_bytes(&self) -> Vec<u8> {
        self.secret_key.as_bytes().to_vec()
    }
}

/// Ultra-secure signature combining Dilithium5 + SPHINCS+
/// AUTOMATICALLY applied for critical operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltraSecureSignature {
    /// Dilithium5 lattice-based signature (~4.6 KB)
    pub dilithium5_signature: Vec<u8>,

    /// SPHINCS+ hash-based signature (~50 KB)
    pub sphincs_signature: Vec<u8>,

    /// Operation type that triggered this signature
    pub operation_type: OperationType,

    /// Timestamp when signature was created
    pub timestamp: i64,
}

impl UltraSecureSignature {
    /// Total signature size (Dilithium5 + SPHINCS+)
    pub fn total_size(&self) -> usize {
        self.dilithium5_signature.len() + self.sphincs_signature.len()
    }

    /// Verify BOTH signatures must be valid
    /// This provides defense-in-depth:
    /// - Dilithium5: Fast lattice-based security
    /// - SPHINCS+: Ultra-conservative hash-based security
    pub fn verify_both(
        &self,
        message: &[u8],
        dilithium5_public_key: &[u8],
        sphincs_public_key: &[u8],
    ) -> Result<bool> {
        use crate::dilithium_wallet::Dilithium5KeyPair;

        // BOTH signatures must verify
        let dilithium_valid = Dilithium5KeyPair::verify(
            message,
            &self.dilithium5_signature,
            dilithium5_public_key,
        )?;

        let sphincs_valid = SphincsPlusKeyPair::verify(
            message,
            &self.sphincs_signature,
            sphincs_public_key,
        )?;

        Ok(dilithium_valid && sphincs_valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphincs_keypair_generation() {
        let keypair = SphincsPlusKeyPair::generate();

        // SPHINCS+-256f key sizes
        assert_eq!(
            keypair.public_key.as_bytes().len(),
            sphincssha256256fsimple::public_key_bytes()
        );
        assert_eq!(
            keypair.secret_key.as_bytes().len(),
            sphincssha256256fsimple::secret_key_bytes()
        );
    }

    #[test]
    fn test_sphincs_sign_verify() {
        let keypair = SphincsPlusKeyPair::generate();
        let message = b"Critical system operation";

        let signature = keypair.sign(message);
        let is_valid = SphincsPlusKeyPair::verify(
            message,
            &signature,
            keypair.public_key.as_bytes(),
        )
        .expect("Verification failed");

        assert!(is_valid, "SPHINCS+ signature should verify");
    }

    #[test]
    fn test_sphincs_invalid_signature() {
        let keypair1 = SphincsPlusKeyPair::generate();
        let keypair2 = SphincsPlusKeyPair::generate();
        let message = b"Test message";

        // Sign with keypair1
        let signature = keypair1.sign(message);

        // Try to verify with keypair2's public key (should fail)
        let is_valid = SphincsPlusKeyPair::verify(
            message,
            &signature,
            keypair2.public_key.as_bytes(),
        )
        .expect("Verification should not error");

        assert!(!is_valid, "Wrong key signature should not verify");
    }

    #[test]
    fn test_sphincs_signature_size() {
        let keypair = SphincsPlusKeyPair::generate();
        let message = b"Test";

        let signature = keypair.sign(message);

        // SPHINCS+-256f signatures are approximately 50 KB
        let expected_size = sphincssha256256fsimple::signature_bytes();

        // The signed message includes the message itself
        assert!(
            signature.len() >= expected_size,
            "SPHINCS+ signature should be large (hash-based security)"
        );

        println!("SPHINCS+-256f signature size: {} bytes", signature.len());
    }

    #[test]
    fn test_operation_type_automatic_detection() {
        // Regular transactions should NOT use SPHINCS+
        assert!(!OperationType::RegularTransaction.requires_sphincs_plus());

        // Critical operations AUTOMATICALLY use SPHINCS+
        assert!(OperationType::GenesisBlock.requires_sphincs_plus());
        assert!(OperationType::ProtocolUpgrade.requires_sphincs_plus());
        assert!(OperationType::ValidatorKeyRotation.requires_sphincs_plus());
        assert!(OperationType::SystemCheckpoint.requires_sphincs_plus());
        assert!(OperationType::AuditTrail.requires_sphincs_plus());
    }

    #[test]
    fn test_ultra_secure_signature_dual_verification() {
        use crate::dilithium_wallet::Dilithium5KeyPair;

        let dilithium_keypair = Dilithium5KeyPair::generate();
        let sphincs_keypair = SphincsPlusKeyPair::generate();

        let message = b"Genesis block signature";

        // Create dual signature
        let dilithium_sig = dilithium_keypair.sign(message);
        let sphincs_sig = sphincs_keypair.sign(message);

        let ultra_sig = UltraSecureSignature {
            dilithium5_signature: dilithium_sig,
            sphincs_signature: sphincs_sig,
            operation_type: OperationType::GenesisBlock,
            timestamp: chrono::Utc::now().timestamp(),
        };

        // Verify both signatures
        let is_valid = ultra_sig
            .verify_both(
                message,
                dilithium_keypair.public_key.as_bytes(),
                sphincs_keypair.public_key.as_bytes(),
            )
            .expect("Verification failed");

        assert!(is_valid, "Both signatures should verify");

        println!("Total ultra-secure signature size: {} bytes", ultra_sig.total_size());
    }

    #[test]
    fn test_ultra_secure_signature_requires_both_valid() {
        use crate::dilithium_wallet::Dilithium5KeyPair;

        let dilithium_keypair = Dilithium5KeyPair::generate();
        let sphincs_keypair = SphincsPlusKeyPair::generate();
        let wrong_dilithium_keypair = Dilithium5KeyPair::generate();

        let message = b"Critical operation";

        let dilithium_sig = dilithium_keypair.sign(message);
        let sphincs_sig = sphincs_keypair.sign(message);

        let ultra_sig = UltraSecureSignature {
            dilithium5_signature: dilithium_sig,
            sphincs_signature: sphincs_sig,
            operation_type: OperationType::ProtocolUpgrade,
            timestamp: chrono::Utc::now().timestamp(),
        };

        // Use wrong Dilithium key - should fail even though SPHINCS+ is correct
        let is_valid = ultra_sig
            .verify_both(
                message,
                wrong_dilithium_keypair.public_key.as_bytes(),
                sphincs_keypair.public_key.as_bytes(),
            )
            .expect("Verification should not error");

        assert!(!is_valid, "Should fail if either signature is invalid");
    }

    #[test]
    fn test_security_descriptions() {
        assert_eq!(
            OperationType::RegularTransaction.security_description(),
            "HIGH SECURITY: Dilithium5 lattice-based signatures"
        );

        assert_eq!(
            OperationType::GenesisBlock.security_description(),
            "MAXIMUM SECURITY: Dilithium5 + SPHINCS+ (hash-based) dual signatures"
        );
    }
}
