/// Crypto-Agile Framework for Q-NarwhalKnight
/// Enables seamless transitions between cryptographic schemes without hard forks

use q_types::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Cryptographic provider with algorithm agility
pub struct CryptoProvider {
    current_scheme: CryptoScheme,
    supported_schemes: HashMap<CryptoSchemeId, Box<dyn CryptoAlgorithm>>,
    phase: Phase,
}

/// Unique identifier for cryptographic schemes (multicodec-compatible)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CryptoSchemeId {
    // Signature schemes
    Ed25519 = 0x1200,           // Phase 0
    Dilithium5 = 0x1300,        // Phase 1 default
    Falcon1024 = 0x1301,        // Phase 1 fallback
    SQIsign = 0x1302,           // Phase 1+ future
    
    // KEM schemes  
    X25519 = 0x2200,            // Phase 0
    Kyber1024 = 0x2300,         // Phase 1 default
    NTRUPrime = 0x2301,         // Phase 1 fallback
    FrodoKEM = 0x2302,          // Phase 1+ alternative
    
    // Hash functions
    SHA256 = 0x12,              // Phase 0 compat
    SHA3_256 = 0x1f,            // Phase 0+ default
    BLAKE3 = 0x1e,              // Phase 1+ alternative
    
    // VRF schemes
    Ed25519VRF = 0x1400,        // Phase 0
    LatticeVRFDilithium = 0x1401, // Phase 2+
}

/// Combined cryptographic scheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoScheme {
    pub signature: CryptoSchemeId,
    pub kem: CryptoSchemeId,
    pub hash: CryptoSchemeId,
    pub vrf: Option<CryptoSchemeId>,
    pub version: u32,
}

/// Crypto-agile handshake for algorithm negotiation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgileHandshake {
    pub supported_schemes: Vec<CryptoScheme>,
    pub preferred_scheme: CryptoScheme,
    pub phase: Phase,
    pub node_capabilities: Vec<String>,
    pub challenge: [u8; 32],
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Trait for cryptographic algorithms
pub trait CryptoAlgorithm: Send + Sync {
    fn scheme_id(&self) -> CryptoSchemeId;
    fn key_size(&self) -> usize;
    fn signature_size(&self) -> Option<usize>;
    fn is_quantum_resistant(&self) -> bool;
    fn performance_tier(&self) -> PerformanceTier;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTier {
    Fast,      // <1ms operations
    Medium,    // 1-10ms operations  
    Slow,      // 10-100ms operations
    VerySlow,  // >100ms operations
}

impl CryptoProvider {
    /// Create Phase 0 crypto provider (Classical)
    pub fn new_phase0() -> Result<Self> {
        let mut supported_schemes = HashMap::new();
        
        // Register classical algorithms
        supported_schemes.insert(CryptoSchemeId::Ed25519, Box::new(Ed25519Algorithm) as Box<dyn CryptoAlgorithm>);
        supported_schemes.insert(CryptoSchemeId::X25519, Box::new(X25519Algorithm));
        supported_schemes.insert(CryptoSchemeId::SHA3_256, Box::new(SHA3_256Algorithm));
        supported_schemes.insert(CryptoSchemeId::Ed25519VRF, Box::new(Ed25519VRFAlgorithm));

        let current_scheme = CryptoScheme {
            signature: CryptoSchemeId::Ed25519,
            kem: CryptoSchemeId::X25519,
            hash: CryptoSchemeId::SHA3_256,
            vrf: Some(CryptoSchemeId::Ed25519VRF),
            version: 1,
        };

        Ok(Self {
            current_scheme,
            supported_schemes,
            phase: Phase::Phase0,
        })
    }

    /// Create Phase 1 crypto provider (Post-Quantum)
    pub fn new_phase1() -> Result<Self> {
        let mut supported_schemes = HashMap::new();
        
        // Register post-quantum algorithms
        supported_schemes.insert(CryptoSchemeId::Dilithium5, Box::new(Dilithium5Algorithm));
        supported_schemes.insert(CryptoSchemeId::Falcon1024, Box::new(Falcon1024Algorithm));
        supported_schemes.insert(CryptoSchemeId::Kyber1024, Box::new(Kyber1024Algorithm));
        supported_schemes.insert(CryptoSchemeId::NTRUPrime, Box::new(NTRUPrimeAlgorithm));
        supported_schemes.insert(CryptoSchemeId::SHA3_256, Box::new(SHA3_256Algorithm));
        supported_schemes.insert(CryptoSchemeId::BLAKE3, Box::new(BLAKE3Algorithm));
        
        // Keep classical for backward compatibility
        supported_schemes.insert(CryptoSchemeId::Ed25519, Box::new(Ed25519Algorithm));
        supported_schemes.insert(CryptoSchemeId::X25519, Box::new(X25519Algorithm));

        let current_scheme = CryptoScheme {
            signature: CryptoSchemeId::Dilithium5,
            kem: CryptoSchemeId::Kyber1024,
            hash: CryptoSchemeId::SHA3_256,
            vrf: None, // L-VRF not available until Phase 2
            version: 2,
        };

        Ok(Self {
            current_scheme,
            supported_schemes,
            phase: Phase::Phase1,
        })
    }

    /// Get all supported cryptographic schemes
    pub fn get_supported_schemes(&self) -> Vec<CryptoScheme> {
        // Generate all valid combinations
        let mut schemes = Vec::new();
        
        let signatures: Vec<CryptoSchemeId> = self.supported_schemes.keys()
            .filter(|id| matches!(id, CryptoSchemeId::Ed25519 | CryptoSchemeId::Dilithium5 | CryptoSchemeId::Falcon1024))
            .copied()
            .collect();
            
        let kems: Vec<CryptoSchemeId> = self.supported_schemes.keys()
            .filter(|id| matches!(id, CryptoSchemeId::X25519 | CryptoSchemeId::Kyber1024 | CryptoSchemeId::NTRUPrime))
            .copied()
            .collect();
            
        let hashes: Vec<CryptoSchemeId> = self.supported_schemes.keys()
            .filter(|id| matches!(id, CryptoSchemeId::SHA3_256 | CryptoSchemeId::BLAKE3))
            .copied()
            .collect();

        for sig in &signatures {
            for kem in &kems {
                for hash in &hashes {
                    schemes.push(CryptoScheme {
                        signature: *sig,
                        kem: *kem,
                        hash: *hash,
                        vrf: None, // Simplified for now
                        version: self.current_scheme.version,
                    });
                }
            }
        }

        schemes
    }

    /// Negotiate best mutual scheme with peer
    pub fn negotiate_scheme(&self, peer_schemes: &[CryptoScheme]) -> Result<CryptoScheme> {
        let our_schemes = self.get_supported_schemes();
        
        // Find schemes we both support
        let mutual_schemes: Vec<&CryptoScheme> = our_schemes.iter()
            .filter(|our_scheme| {
                peer_schemes.iter().any(|peer_scheme| {
                    our_scheme.signature == peer_scheme.signature &&
                    our_scheme.kem == peer_scheme.kem &&
                    our_scheme.hash == peer_scheme.hash
                })
            })
            .collect();

        if mutual_schemes.is_empty() {
            return Err(CryptoError::NoMutualSchemes.into());
        }

        // Select best scheme based on security and performance
        let best_scheme = mutual_schemes.into_iter()
            .max_by_key(|scheme| self.scheme_priority_score(scheme))
            .unwrap();

        Ok(best_scheme.clone())
    }

    /// Calculate priority score for scheme selection
    fn scheme_priority_score(&self, scheme: &CryptoScheme) -> u32 {
        let mut score = 0u32;
        
        // Prefer quantum-resistant algorithms
        if let Some(sig_alg) = self.supported_schemes.get(&scheme.signature) {
            if sig_alg.is_quantum_resistant() {
                score += 1000;
            }
            
            // Performance bonus
            score += match sig_alg.performance_tier() {
                PerformanceTier::Fast => 100,
                PerformanceTier::Medium => 50,
                PerformanceTier::Slow => 20,
                PerformanceTier::VerySlow => 1,
            };
        }
        
        if let Some(kem_alg) = self.supported_schemes.get(&scheme.kem) {
            if kem_alg.is_quantum_resistant() {
                score += 1000;
            }
        }
        
        // Prefer newer versions
        score += scheme.version * 10;
        
        score
    }

    /// Upgrade to new cryptographic scheme
    pub async fn upgrade_scheme(&mut self, new_scheme: CryptoScheme) -> Result<()> {
        // Validate that we support the new scheme
        if !self.is_scheme_supported(&new_scheme) {
            return Err(CryptoError::UnsupportedScheme.into());
        }

        tracing::info!("🔄 Upgrading crypto scheme: {:?} -> {:?}", 
                      self.current_scheme, new_scheme);

        // TODO: Implement key rotation and secure transition
        self.current_scheme = new_scheme;
        
        tracing::info!("✅ Successfully upgraded to new crypto scheme");
        Ok(())
    }

    /// Check if scheme is supported
    pub fn is_scheme_supported(&self, scheme: &CryptoScheme) -> bool {
        self.supported_schemes.contains_key(&scheme.signature) &&
        self.supported_schemes.contains_key(&scheme.kem) &&
        self.supported_schemes.contains_key(&scheme.hash)
    }

    /// Get current scheme description
    pub fn get_current_scheme(&self) -> String {
        format!("Sig:{:?}/KEM:{:?}/Hash:{:?}", 
                self.current_scheme.signature,
                self.current_scheme.kem, 
                self.current_scheme.hash)
    }

    /// Generate capabilities list for handshake
    pub fn get_capabilities(&self) -> Vec<String> {
        let mut caps = vec![
            format!("phase-{}", self.phase as u8),
            "q-narwhal-knight".to_string(),
            "dag-bft".to_string(),
        ];

        if self.phase >= Phase::Phase1 {
            caps.push("post-quantum".to_string());
        }
        
        if self.phase >= Phase::Phase2 {
            caps.push("quantum-randomness".to_string());
        }

        caps
    }
}

impl AgileHandshake {
    pub fn new(supported_schemes: Vec<CryptoScheme>, phase: Phase) -> Result<Self> {
        let preferred_scheme = supported_schemes.first()
            .ok_or(CryptoError::NoSupportedSchemes)?
            .clone();

        let mut challenge = [0u8; 32];
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut challenge);

        Ok(Self {
            supported_schemes,
            preferred_scheme,
            phase,
            node_capabilities: vec![], // Will be filled by provider
            challenge,
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Algorithm implementations
struct Ed25519Algorithm;
impl CryptoAlgorithm for Ed25519Algorithm {
    fn scheme_id(&self) -> CryptoSchemeId { CryptoSchemeId::Ed25519 }
    fn key_size(&self) -> usize { 32 }
    fn signature_size(&self) -> Option<usize> { Some(64) }
    fn is_quantum_resistant(&self) -> bool { false }
    fn performance_tier(&self) -> PerformanceTier { PerformanceTier::Fast }
}

struct Dilithium5Algorithm;
impl CryptoAlgorithm for Dilithium5Algorithm {
    fn scheme_id(&self) -> CryptoSchemeId { CryptoSchemeId::Dilithium5 }
    fn key_size(&self) -> usize { 2592 }
    fn signature_size(&self) -> Option<usize> { Some(4595) }
    fn is_quantum_resistant(&self) -> bool { true }
    fn performance_tier(&self) -> PerformanceTier { PerformanceTier::Medium }
}

struct Falcon1024Algorithm;
impl CryptoAlgorithm for Falcon1024Algorithm {
    fn scheme_id(&self) -> CryptoSchemeId { CryptoSchemeId::Falcon1024 }
    fn key_size(&self) -> usize { 1793 }
    fn signature_size(&self) -> Option<usize> { Some(1330) }
    fn is_quantum_resistant(&self) -> bool { true }
    fn performance_tier(&self) -> PerformanceTier { PerformanceTier::Slow }
}

struct X25519Algorithm;
impl CryptoAlgorithm for X25519Algorithm {
    fn scheme_id(&self) -> CryptoSchemeId { CryptoSchemeId::X25519 }
    fn key_size(&self) -> usize { 32 }
    fn signature_size(&self) -> Option<usize> { None }
    fn is_quantum_resistant(&self) -> bool { false }
    fn performance_tier(&self) -> PerformanceTier { PerformanceTier::Fast }
}

struct Kyber1024Algorithm;
impl CryptoAlgorithm for Kyber1024Algorithm {
    fn scheme_id(&self) -> CryptoSchemeId { CryptoSchemeId::Kyber1024 }
    fn key_size(&self) -> usize { 1568 }
    fn signature_size(&self) -> Option<usize> { None }
    fn is_quantum_resistant(&self) -> bool { true }
    fn performance_tier(&self) -> PerformanceTier { PerformanceTier::Medium }
}

struct NTRUPrimeAlgorithm;
impl CryptoAlgorithm for NTRUPrimeAlgorithm {
    fn scheme_id(&self) -> CryptoSchemeId { CryptoSchemeId::NTRUPrime }
    fn key_size(&self) -> usize { 1322 }
    fn signature_size(&self) -> Option<usize> { None }
    fn is_quantum_resistant(&self) -> bool { true }
    fn performance_tier(&self) -> PerformanceTier { PerformanceTier::Medium }
}

struct SHA3_256Algorithm;
impl CryptoAlgorithm for SHA3_256Algorithm {
    fn scheme_id(&self) -> CryptoSchemeId { CryptoSchemeId::SHA3_256 }
    fn key_size(&self) -> usize { 0 }
    fn signature_size(&self) -> Option<usize> { None }
    fn is_quantum_resistant(&self) -> bool { true }
    fn performance_tier(&self) -> PerformanceTier { PerformanceTier::Fast }
}

struct BLAKE3Algorithm;
impl CryptoAlgorithm for BLAKE3Algorithm {
    fn scheme_id(&self) -> CryptoSchemeId { CryptoSchemeId::BLAKE3 }
    fn key_size(&self) -> usize { 0 }
    fn signature_size(&self) -> Option<usize> { None }
    fn is_quantum_resistant(&self) -> bool { true }
    fn performance_tier(&self) -> PerformanceTier { PerformanceTier::Fast }
}

struct Ed25519VRFAlgorithm;
impl CryptoAlgorithm for Ed25519VRFAlgorithm {
    fn scheme_id(&self) -> CryptoSchemeId { CryptoSchemeId::Ed25519VRF }
    fn key_size(&self) -> usize { 32 }
    fn signature_size(&self) -> Option<usize> { Some(96) }
    fn is_quantum_resistant(&self) -> bool { false }
    fn performance_tier(&self) -> PerformanceTier { PerformanceTier::Fast }
}

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("No mutual cryptographic schemes found")]
    NoMutualSchemes,
    #[error("No supported schemes available")]
    NoSupportedSchemes,
    #[error("Unsupported cryptographic scheme")]
    UnsupportedScheme,
    #[error("Key generation failed")]
    KeyGenerationFailed,
    #[error("Signature verification failed")]
    SignatureVerificationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase0_provider() {
        let provider = CryptoProvider::new_phase0().unwrap();
        assert_eq!(provider.phase, Phase::Phase0);
        assert_eq!(provider.current_scheme.signature, CryptoSchemeId::Ed25519);
    }

    #[test]
    fn test_phase1_provider() {
        let provider = CryptoProvider::new_phase1().unwrap();
        assert_eq!(provider.phase, Phase::Phase1);
        assert_eq!(provider.current_scheme.signature, CryptoSchemeId::Dilithium5);
        assert_eq!(provider.current_scheme.kem, CryptoSchemeId::Kyber1024);
    }

    #[test]
    fn test_scheme_negotiation() {
        let provider = CryptoProvider::new_phase1().unwrap();
        
        let peer_schemes = vec![
            CryptoScheme {
                signature: CryptoSchemeId::Dilithium5,
                kem: CryptoSchemeId::Kyber1024,
                hash: CryptoSchemeId::SHA3_256,
                vrf: None,
                version: 2,
            }
        ];
        
        let negotiated = provider.negotiate_scheme(&peer_schemes).unwrap();
        assert_eq!(negotiated.signature, CryptoSchemeId::Dilithium5);
    }

    #[test]
    fn test_algorithm_properties() {
        let ed25519 = Ed25519Algorithm;
        assert!(!ed25519.is_quantum_resistant());
        assert_eq!(ed25519.performance_tier(), PerformanceTier::Fast);
        
        let dilithium = Dilithium5Algorithm;
        assert!(dilithium.is_quantum_resistant());
        assert_eq!(dilithium.performance_tier(), PerformanceTier::Medium);
    }

    #[test]
    fn test_handshake_creation() {
        let schemes = vec![
            CryptoScheme {
                signature: CryptoSchemeId::Dilithium5,
                kem: CryptoSchemeId::Kyber1024,
                hash: CryptoSchemeId::SHA3_256,
                vrf: None,
                version: 2,
            }
        ];
        
        let handshake = AgileHandshake::new(schemes, Phase::Phase1);
        assert!(handshake.is_ok());
        
        let handshake = handshake.unwrap();
        assert_eq!(handshake.phase, Phase::Phase1);
        assert_eq!(handshake.supported_schemes.len(), 1);
    }

    #[test]
    fn test_scheme_priority_scoring() {
        let provider = CryptoProvider::new_phase1().unwrap();
        
        let pq_scheme = CryptoScheme {
            signature: CryptoSchemeId::Dilithium5,
            kem: CryptoSchemeId::Kyber1024,
            hash: CryptoSchemeId::SHA3_256,
            vrf: None,
            version: 2,
        };
        
        let classical_scheme = CryptoScheme {
            signature: CryptoSchemeId::Ed25519,
            kem: CryptoSchemeId::X25519,
            hash: CryptoSchemeId::SHA3_256,
            vrf: None,
            version: 1,
        };
        
        let pq_score = provider.scheme_priority_score(&pq_scheme);
        let classical_score = provider.scheme_priority_score(&classical_scheme);
        
        assert!(pq_score > classical_score); // Post-quantum should score higher
    }
}