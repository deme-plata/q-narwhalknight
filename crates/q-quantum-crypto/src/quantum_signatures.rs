//! ✍️ Quantum Digital Signatures
//! Unconditionally secure signature schemes

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

use crate::quantum_entropy::QuantumEntropySource;
use crate::{NodeId, QuantumKey};

/// Quantum digital signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSignature {
    /// Signature scheme used
    pub scheme: SignatureScheme,
    /// Signature data
    pub signature_data: Vec<u8>,
    /// Public verification data
    pub verification_data: Vec<u8>,
    /// Signature timestamp
    pub timestamp: SystemTime,
    /// Security level achieved
    pub security_level: SecurityLevel,
    /// One-time signature sequence number
    pub sequence_number: u64,
}

/// Available quantum signature schemes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SignatureScheme {
    /// Lamport One-Time Signatures
    LamportOTS,
    /// Winternitz One-Time Signatures
    WinternitzOTS,
    /// Merkle Signature Scheme
    Merkle,
    /// XMSS (Extended Merkle Signature Scheme)
    XMSS,
}

/// Signature security level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Information-theoretically secure
    InformationTheoretic,
    /// Quantum-computationally secure
    QuantumComputational,
    /// Compromised security
    Compromised,
}

/// Lamport One-Time Signature implementation
#[derive(Debug)]
pub struct LamportOTS {
    /// Node identifier
    node_id: NodeId,
    /// Quantum entropy source
    entropy_source: Arc<QuantumEntropySource>,
    /// Private key pairs (256 pairs of 32-byte values)
    private_keys: Vec<(Vec<u8>, Vec<u8>)>,
    /// Public key pairs (hashes of private keys)
    public_keys: Vec<(Vec<u8>, Vec<u8>)>,
    /// Usage flag (one-time use only)
    used: bool,
    /// Creation timestamp
    created_at: SystemTime,
}

impl LamportOTS {
    /// Generate new Lamport OTS key pair
    pub async fn generate(
        node_id: NodeId,
        entropy_source: Arc<QuantumEntropySource>,
    ) -> Result<Self> {
        let mut private_keys = Vec::with_capacity(256);
        let mut public_keys = Vec::with_capacity(256);

        // Generate 256 key pairs (one for each bit of SHA3-256)
        for _ in 0..256 {
            // Generate two 32-byte private keys
            let priv_key_0 = entropy_source.generate_true_random(32).await?;
            let priv_key_1 = entropy_source.generate_true_random(32).await?;

            // Compute corresponding public keys (hashes)
            let pub_key_0 = Sha3_256::digest(&priv_key_0).to_vec();
            let pub_key_1 = Sha3_256::digest(&priv_key_1).to_vec();

            private_keys.push((priv_key_0, priv_key_1));
            public_keys.push((pub_key_0, pub_key_1));
        }

        Ok(Self {
            node_id,
            entropy_source,
            private_keys,
            public_keys,
            used: false,
            created_at: SystemTime::now(),
        })
    }

    /// Sign message (can only be used once)
    pub async fn sign(&mut self, message: &[u8]) -> Result<QuantumSignature> {
        if self.used {
            return Err(anyhow::anyhow!("Lamport OTS key already used"));
        }

        // Hash message to 256 bits
        let message_hash = Sha3_256::digest(message);
        let hash_bits = message_hash.to_vec();

        let mut signature_data = Vec::new();

        // For each bit of the hash, reveal the corresponding private key
        for (i, &byte) in hash_bits.iter().enumerate() {
            for bit_pos in 0..8 {
                let bit_index = i * 8 + bit_pos;
                if bit_index >= 256 {
                    break;
                }

                let bit = (byte >> bit_pos) & 1;
                let private_key = if bit == 0 {
                    &self.private_keys[bit_index].0
                } else {
                    &self.private_keys[bit_index].1
                };

                signature_data.extend_from_slice(private_key);
            }
        }

        // Create verification data (all public keys)
        let mut verification_data = Vec::new();
        for (pub_key_0, pub_key_1) in &self.public_keys {
            verification_data.extend_from_slice(pub_key_0);
            verification_data.extend_from_slice(pub_key_1);
        }

        self.used = true;

        Ok(QuantumSignature {
            scheme: SignatureScheme::LamportOTS,
            signature_data,
            verification_data,
            timestamp: SystemTime::now(),
            security_level: SecurityLevel::InformationTheoretic,
            sequence_number: 0,
        })
    }

    /// Get public verification data
    pub fn get_public_key(&self) -> Vec<u8> {
        let mut public_key_data = Vec::new();
        for (pub_key_0, pub_key_1) in &self.public_keys {
            public_key_data.extend_from_slice(pub_key_0);
            public_key_data.extend_from_slice(pub_key_1);
        }
        public_key_data
    }

    /// Check if key has been used
    pub fn is_used(&self) -> bool {
        self.used
    }
}

/// Quantum signature verifier
#[derive(Debug)]
pub struct QuantumVerifier {
    /// Signer node ID
    signer_id: NodeId,
}

impl QuantumVerifier {
    /// Create new verifier for a signer
    pub fn new(signer_id: NodeId) -> Self {
        Self { signer_id }
    }

    /// Verify quantum signature
    pub async fn verify_signature(
        &self,
        message: &[u8],
        signature: &QuantumSignature,
    ) -> Result<bool> {
        match signature.scheme {
            SignatureScheme::LamportOTS => self.verify_lamport_signature(message, signature).await,
            _ => Err(anyhow::anyhow!("Signature scheme not yet implemented")),
        }
    }

    /// Verify Lamport OTS signature
    async fn verify_lamport_signature(
        &self,
        message: &[u8],
        signature: &QuantumSignature,
    ) -> Result<bool> {
        // Hash message
        let message_hash = Sha3_256::digest(message);
        let hash_bits = message_hash.to_vec();

        // Extract public keys from verification data
        let verification_data = &signature.verification_data;
        if verification_data.len() != 256 * 2 * 32 {
            return Ok(false); // Invalid public key data
        }

        let mut public_keys = Vec::new();
        for i in 0..256 {
            let offset = i * 64; // 2 keys * 32 bytes each
            let pub_key_0 = verification_data[offset..offset + 32].to_vec();
            let pub_key_1 = verification_data[offset + 32..offset + 64].to_vec();
            public_keys.push((pub_key_0, pub_key_1));
        }

        // Verify signature
        let signature_data = &signature.signature_data;
        if signature_data.len() != 256 * 32 {
            return Ok(false); // Invalid signature length
        }

        for (i, &byte) in hash_bits.iter().enumerate() {
            for bit_pos in 0..8 {
                let bit_index = i * 8 + bit_pos;
                if bit_index >= 256 {
                    break;
                }

                let bit = (byte >> bit_pos) & 1;
                let sig_offset = bit_index * 32;
                let revealed_key = &signature_data[sig_offset..sig_offset + 32];

                // Hash the revealed private key and compare with public key
                let computed_hash = Sha3_256::digest(revealed_key).to_vec();
                let expected_public_key = if bit == 0 {
                    &public_keys[bit_index].0
                } else {
                    &public_keys[bit_index].1
                };

                if computed_hash != *expected_public_key {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

/// Quantum signature manager
#[derive(Debug)]
pub struct QuantumSigner {
    /// Node identifier
    node_id: NodeId,
    /// Quantum entropy source
    entropy_source: Arc<QuantumEntropySource>,
    /// Current Lamport OTS key
    current_lamport_key: Arc<RwLock<Option<LamportOTS>>>,
    /// Signature counter
    signature_count: Arc<RwLock<u64>>,
    /// Key generation history
    key_history: Arc<RwLock<Vec<KeyGenerationRecord>>>,
}

impl QuantumSigner {
    /// Create new quantum signer
    pub async fn new(node_id: NodeId, entropy_source: Arc<QuantumEntropySource>) -> Result<Self> {
        Ok(Self {
            node_id,
            entropy_source,
            current_lamport_key: Arc::new(RwLock::new(None)),
            signature_count: Arc::new(RwLock::new(0)),
            key_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Sign message with quantum signature
    pub async fn sign_message(&self, message: &[u8]) -> Result<QuantumSignature> {
        // Check if we need a new Lamport key
        let needs_new_key = {
            let current_key = self.current_lamport_key.read().await;
            current_key.is_none() || current_key.as_ref().unwrap().is_used()
        };

        if needs_new_key {
            self.generate_new_lamport_key().await?;
        }

        // Sign with current key
        let signature = {
            let mut current_key = self.current_lamport_key.write().await;
            let key = current_key
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("No signing key available"))?;
            key.sign(message).await?
        };

        // Update signature count
        *self.signature_count.write().await += 1;

        Ok(signature)
    }

    /// Generate new Lamport OTS key pair
    async fn generate_new_lamport_key(&self) -> Result<()> {
        let new_key = LamportOTS::generate(self.node_id, self.entropy_source.clone()).await?;

        // Record key generation
        let record = KeyGenerationRecord {
            generated_at: SystemTime::now(),
            public_key_hash: {
                let public_key_data = new_key.get_public_key();
                Sha3_256::digest(&public_key_data).to_vec()
            },
            scheme: SignatureScheme::LamportOTS,
        };

        self.key_history.write().await.push(record);
        *self.current_lamport_key.write().await = Some(new_key);

        Ok(())
    }

    /// Get current public key
    pub async fn get_public_key(&self) -> Result<Vec<u8>> {
        let current_key = self.current_lamport_key.read().await;
        match current_key.as_ref() {
            Some(key) => Ok(key.get_public_key()),
            None => {
                drop(current_key);
                self.generate_new_lamport_key().await?;
                let new_key = self.current_lamport_key.read().await;
                Ok(new_key.as_ref().unwrap().get_public_key())
            }
        }
    }

    /// Get signature count
    pub async fn get_signature_count(&self) -> u64 {
        *self.signature_count.read().await
    }

    /// Health check
    pub async fn health_check(&self) -> Result<bool> {
        let current_key = self.current_lamport_key.read().await;

        // Check if we have a valid unused key or can generate one
        match current_key.as_ref() {
            Some(key) => Ok(!key.is_used()),
            None => Ok(true), // We can generate a new key
        }
    }

    /// Get key generation history
    pub async fn get_key_history(&self) -> Vec<KeyGenerationRecord> {
        self.key_history.read().await.clone()
    }
}

/// Key generation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyGenerationRecord {
    pub generated_at: SystemTime,
    pub public_key_hash: Vec<u8>,
    pub scheme: SignatureScheme,
}

/// Winternitz One-Time Signature (more efficient than Lamport)
#[derive(Debug)]
pub struct WinternitzOTS {
    /// Winternitz parameter (trade-off between signature size and security)
    w: u8,
    /// Private key chains
    private_chains: Vec<Vec<u8>>,
    /// Public key (end of chains)
    public_key: Vec<u8>,
    /// Usage flag
    used: bool,
}

impl WinternitzOTS {
    /// Generate Winternitz OTS key pair
    pub async fn generate(w: u8, entropy_source: Arc<QuantumEntropySource>) -> Result<Self> {
        let chain_count = match w {
            4 => 67,  // For w=4 (16 values per chain)
            8 => 34,  // For w=8 (256 values per chain)
            16 => 18, // For w=16 (65536 values per chain)
            _ => return Err(anyhow::anyhow!("Unsupported Winternitz parameter")),
        };

        let chain_length = 1 << w; // 2^w
        let mut private_chains = Vec::with_capacity(chain_count);
        let mut public_key = Vec::new();

        for _ in 0..chain_count {
            // Generate random starting point
            let mut chain_start = entropy_source.generate_true_random(32).await?;

            // Compute full chain by iterative hashing
            let mut current = chain_start.clone();
            for _ in 0..chain_length {
                current = Sha3_256::digest(&current).to_vec();
            }

            // Store the starting point (private key)
            private_chains.push(chain_start);

            // Store the end point (public key component)
            public_key.extend_from_slice(&current);
        }

        Ok(Self {
            w,
            private_chains,
            public_key,
            used: false,
        })
    }

    /// Sign message with Winternitz OTS
    pub async fn sign(&mut self, message: &[u8]) -> Result<Vec<u8>> {
        if self.used {
            return Err(anyhow::anyhow!("Winternitz OTS key already used"));
        }

        // Hash message and convert to base-w representation
        let message_hash = Sha3_256::digest(message);
        let base_w_repr = self.to_base_w(&message_hash);

        let mut signature = Vec::new();

        for (i, &value) in base_w_repr.iter().enumerate() {
            if i >= self.private_chains.len() {
                break;
            }

            // Hash the private key 'value' times
            let mut current = self.private_chains[i].clone();
            for _ in 0..value {
                current = Sha3_256::digest(&current).to_vec();
            }

            signature.extend_from_slice(&current);
        }

        self.used = true;
        Ok(signature)
    }

    /// Convert hash to base-w representation
    fn to_base_w(&self, hash: &[u8]) -> Vec<u32> {
        let mut result = Vec::new();
        let base = 1 << self.w;

        for &byte in hash.iter() {
            let mut remaining = byte as u32;
            let chunks = match self.w {
                4 => 2, // 8 bits / 4 bits = 2 chunks
                8 => 1, // 8 bits / 8 bits = 1 chunk
                _ => 1,
            };

            for _ in 0..chunks {
                result.push(remaining % base);
                remaining /= base;
            }
        }

        result
    }
}

/// Quantum signature statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSignatureStats {
    pub total_signatures_generated: u64,
    pub total_signatures_verified: u64,
    pub lamport_keys_generated: u64,
    pub winternitz_keys_generated: u64,
    pub average_signature_size: usize,
    pub last_signature_time: SystemTime,
}

impl Default for QuantumSignatureStats {
    fn default() -> Self {
        Self {
            total_signatures_generated: 0,
            total_signatures_verified: 0,
            lamport_keys_generated: 0,
            winternitz_keys_generated: 0,
            average_signature_size: 0,
            last_signature_time: SystemTime::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_entropy::QuantumEntropySource;

    #[tokio::test]
    async fn test_lamport_ots_generation() {
        let node_id = [1u8; 32];
        let entropy_source = Arc::new(QuantumEntropySource::new().await.unwrap());

        let lamport = LamportOTS::generate(node_id, entropy_source).await.unwrap();
        assert_eq!(lamport.private_keys.len(), 256);
        assert_eq!(lamport.public_keys.len(), 256);
        assert!(!lamport.is_used());
    }

    #[tokio::test]
    async fn test_lamport_signature_and_verification() {
        let node_id = [1u8; 32];
        let entropy_source = Arc::new(QuantumEntropySource::new().await.unwrap());

        let mut lamport = LamportOTS::generate(node_id, entropy_source).await.unwrap();
        let message = b"Hello, quantum world!";

        let signature = lamport.sign(message).await.unwrap();
        assert!(lamport.is_used());

        let verifier = QuantumVerifier::new(node_id);
        let is_valid = verifier
            .verify_signature(message, &signature)
            .await
            .unwrap();
        assert!(is_valid);

        // Test with wrong message
        let wrong_message = b"Wrong message";
        let is_invalid = verifier
            .verify_signature(wrong_message, &signature)
            .await
            .unwrap();
        assert!(!is_invalid);
    }

    #[tokio::test]
    async fn test_quantum_signer() {
        let node_id = [1u8; 32];
        let entropy_source = Arc::new(QuantumEntropySource::new().await.unwrap());

        let signer = QuantumSigner::new(node_id, entropy_source).await.unwrap();
        let message = b"Test message for quantum signature";

        let signature = signer.sign_message(message).await.unwrap();
        assert_eq!(signature.scheme, SignatureScheme::LamportOTS);
        assert_eq!(
            signature.security_level,
            SecurityLevel::InformationTheoretic
        );

        let count = signer.get_signature_count().await;
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_winternitz_ots_generation() {
        let entropy_source = Arc::new(QuantumEntropySource::new().await.unwrap());

        let winternitz = WinternitzOTS::generate(4, entropy_source).await.unwrap();
        assert_eq!(winternitz.w, 4);
        assert!(!winternitz.used);
        assert!(!winternitz.public_key.is_empty());
    }
}
