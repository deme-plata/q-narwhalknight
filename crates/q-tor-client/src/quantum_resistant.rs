/// Quantum-Resistant Upgrades for Q-NarwhalKnight Tor Layer
///
/// This module provides post-quantum cryptographic enhancements for Tor circuits,
/// preparing for the quantum computing threat to current cryptographic algorithms.
///
/// Features:
/// - Hybrid key exchange (classical + post-quantum)
/// - Post-quantum signature verification
/// - Algorithm agility for future upgrades
/// - Gradual migration support
/// - Backward compatibility with classical Tor
///
/// Implements concepts from NIST Post-Quantum Cryptography standardization.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// 🔐 SHA3-256 for key IDs + the hybrid-KEM combiner. The actual Kyber/X25519
// primitives live in the `pq_kem` submodule (whitepaper §8.3 "hybrid Kyber-1024 + X25519").
use sha3::{Digest, Sha3_256};

/// Post-quantum algorithm families
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PQAlgorithm {
    /// Kyber - Lattice-based key encapsulation (NIST standard)
    Kyber512,
    Kyber768,
    Kyber1024,
    /// Dilithium - Lattice-based signatures (NIST standard)
    Dilithium2,
    Dilithium3,
    Dilithium5,
    /// SPHINCS+ - Hash-based signatures (NIST standard)
    SphincsShake128f,
    SphincsShake192f,
    SphincsShake256f,
    /// Classic McEliece - Code-based (conservative choice)
    McEliece348864,
    McEliece460896,
    /// NTRU - Lattice-based (alternative)
    NtruHps2048509,
    NtruHps2048677,
    /// Hybrid classical + post-quantum
    HybridX25519Kyber768,
    HybridP256Kyber768,
    /// Hybrid X25519 + Kyber-1024 (NIST Level 5) — whitepaper §8.3 circuit-level TLS target
    HybridX25519Kyber1024,
}

impl PQAlgorithm {
    pub fn name(&self) -> &'static str {
        match self {
            PQAlgorithm::Kyber512 => "Kyber-512",
            PQAlgorithm::Kyber768 => "Kyber-768",
            PQAlgorithm::Kyber1024 => "Kyber-1024",
            PQAlgorithm::Dilithium2 => "Dilithium-2",
            PQAlgorithm::Dilithium3 => "Dilithium-3",
            PQAlgorithm::Dilithium5 => "Dilithium-5",
            PQAlgorithm::SphincsShake128f => "SPHINCS+-SHAKE-128f",
            PQAlgorithm::SphincsShake192f => "SPHINCS+-SHAKE-192f",
            PQAlgorithm::SphincsShake256f => "SPHINCS+-SHAKE-256f",
            PQAlgorithm::McEliece348864 => "Classic-McEliece-348864",
            PQAlgorithm::McEliece460896 => "Classic-McEliece-460896",
            PQAlgorithm::NtruHps2048509 => "NTRU-HPS-2048-509",
            PQAlgorithm::NtruHps2048677 => "NTRU-HPS-2048-677",
            PQAlgorithm::HybridX25519Kyber768 => "X25519+Kyber-768",
            PQAlgorithm::HybridP256Kyber768 => "P-256+Kyber-768",
            PQAlgorithm::HybridX25519Kyber1024 => "X25519+Kyber-1024",
        }
    }

    /// Security level (NIST levels 1-5)
    pub fn security_level(&self) -> u8 {
        match self {
            PQAlgorithm::Kyber512 | PQAlgorithm::Dilithium2 | PQAlgorithm::SphincsShake128f => 1,
            PQAlgorithm::NtruHps2048509 => 1,
            PQAlgorithm::Kyber768 | PQAlgorithm::Dilithium3 | PQAlgorithm::SphincsShake192f => 3,
            PQAlgorithm::McEliece348864 | PQAlgorithm::NtruHps2048677 => 3,
            PQAlgorithm::HybridX25519Kyber768 | PQAlgorithm::HybridP256Kyber768 => 3,
            PQAlgorithm::Kyber1024 | PQAlgorithm::Dilithium5 | PQAlgorithm::SphincsShake256f => 5,
            PQAlgorithm::McEliece460896 => 5,
            PQAlgorithm::HybridX25519Kyber1024 => 5,
        }
    }

    /// Is this a key encapsulation mechanism (KEM)?
    pub fn is_kem(&self) -> bool {
        matches!(self,
            PQAlgorithm::Kyber512 | PQAlgorithm::Kyber768 | PQAlgorithm::Kyber1024 |
            PQAlgorithm::McEliece348864 | PQAlgorithm::McEliece460896 |
            PQAlgorithm::NtruHps2048509 | PQAlgorithm::NtruHps2048677 |
            PQAlgorithm::HybridX25519Kyber768 | PQAlgorithm::HybridP256Kyber768 |
            PQAlgorithm::HybridX25519Kyber1024
        )
    }

    /// Is this a signature algorithm?
    pub fn is_signature(&self) -> bool {
        matches!(self,
            PQAlgorithm::Dilithium2 | PQAlgorithm::Dilithium3 | PQAlgorithm::Dilithium5 |
            PQAlgorithm::SphincsShake128f | PQAlgorithm::SphincsShake192f | PQAlgorithm::SphincsShake256f
        )
    }

    /// Is this a hybrid algorithm?
    pub fn is_hybrid(&self) -> bool {
        matches!(self,
            PQAlgorithm::HybridX25519Kyber768 | PQAlgorithm::HybridP256Kyber768 |
            PQAlgorithm::HybridX25519Kyber1024
        )
    }

    /// Approximate public key size in bytes
    pub fn public_key_size(&self) -> usize {
        match self {
            PQAlgorithm::Kyber512 => 800,
            PQAlgorithm::Kyber768 => 1184,
            PQAlgorithm::Kyber1024 => 1568,
            PQAlgorithm::Dilithium2 => 1312,
            PQAlgorithm::Dilithium3 => 1952,
            PQAlgorithm::Dilithium5 => 2592,
            PQAlgorithm::SphincsShake128f => 32,
            PQAlgorithm::SphincsShake192f => 48,
            PQAlgorithm::SphincsShake256f => 64,
            PQAlgorithm::McEliece348864 => 261120,
            PQAlgorithm::McEliece460896 => 524160,
            PQAlgorithm::NtruHps2048509 => 699,
            PQAlgorithm::NtruHps2048677 => 930,
            PQAlgorithm::HybridX25519Kyber768 => 32 + 1184,
            PQAlgorithm::HybridP256Kyber768 => 65 + 1184,
            PQAlgorithm::HybridX25519Kyber1024 => 32 + 1568,
        }
    }

    /// Approximate ciphertext/signature size in bytes
    pub fn ciphertext_size(&self) -> usize {
        match self {
            PQAlgorithm::Kyber512 => 768,
            PQAlgorithm::Kyber768 => 1088,
            PQAlgorithm::Kyber1024 => 1568,
            PQAlgorithm::Dilithium2 => 2420,
            PQAlgorithm::Dilithium3 => 3293,
            PQAlgorithm::Dilithium5 => 4595,
            PQAlgorithm::SphincsShake128f => 17088,
            PQAlgorithm::SphincsShake192f => 35664,
            PQAlgorithm::SphincsShake256f => 49856,
            PQAlgorithm::McEliece348864 => 128,
            PQAlgorithm::McEliece460896 => 188,
            PQAlgorithm::NtruHps2048509 => 699,
            PQAlgorithm::NtruHps2048677 => 930,
            PQAlgorithm::HybridX25519Kyber768 => 32 + 1088,
            PQAlgorithm::HybridP256Kyber768 => 65 + 1088,
            PQAlgorithm::HybridX25519Kyber1024 => 32 + 1568,
        }
    }
}

/// Migration phase for quantum-resistant upgrade
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationPhase {
    /// Phase 0: Classical only (current Tor)
    ClassicalOnly,
    /// Phase 1: Hybrid optional (test with early adopters)
    HybridOptional,
    /// Phase 2: Hybrid preferred (encourage adoption)
    HybridPreferred,
    /// Phase 3: Hybrid required (deprecate classical)
    HybridRequired,
    /// Phase 4: Post-quantum only (classical disabled)
    PostQuantumOnly,
}

impl MigrationPhase {
    pub fn name(&self) -> &'static str {
        match self {
            MigrationPhase::ClassicalOnly => "Classical Only",
            MigrationPhase::HybridOptional => "Hybrid Optional",
            MigrationPhase::HybridPreferred => "Hybrid Preferred",
            MigrationPhase::HybridRequired => "Hybrid Required",
            MigrationPhase::PostQuantumOnly => "Post-Quantum Only",
        }
    }

    /// Whether to accept classical-only connections
    pub fn accept_classical(&self) -> bool {
        !matches!(self, MigrationPhase::PostQuantumOnly)
    }

    /// Whether to prefer hybrid connections
    pub fn prefer_hybrid(&self) -> bool {
        matches!(self, MigrationPhase::HybridPreferred | MigrationPhase::HybridRequired)
    }

    /// Whether to require hybrid/PQ connections
    pub fn require_pq(&self) -> bool {
        matches!(self, MigrationPhase::HybridRequired | MigrationPhase::PostQuantumOnly)
    }
}

/// Post-quantum key pair
#[derive(Debug, Clone)]
pub struct PQKeyPair {
    /// Algorithm used
    pub algorithm: PQAlgorithm,
    /// Public key bytes
    pub public_key: Vec<u8>,
    /// Secret key bytes (kept private)
    secret_key: Vec<u8>,
    /// Creation time
    pub created_at: SystemTime,
    /// Key ID (hash of public key)
    pub key_id: [u8; 32],
}

impl PQKeyPair {
    /// Generate a new key pair.
    ///
    /// KEM algorithms (Kyber family + X25519⊕Kyber hybrids) use REAL crypto via
    /// `pq_kem`. Other (signature / not-yet-wired) algorithms keep the previous
    /// random-bytes placeholder so existing callers don't regress.
    pub fn generate(algorithm: PQAlgorithm) -> Result<Self> {
        let (public_key, secret_key) = if algorithm.is_kem() {
            pq_kem::generate(algorithm)?
        } else {
            // Placeholder for signature/unwired algorithms (sign/verify still stubbed).
            let pk_size = algorithm.public_key_size();
            let mut rng = rand::thread_rng();
            use rand::RngCore;
            let mut public_key = vec![0u8; pk_size];
            let mut secret_key = vec![0u8; pk_size * 2];
            rng.fill_bytes(&mut public_key);
            rng.fill_bytes(&mut secret_key);
            (public_key, secret_key)
        };

        // Key ID = SHA3-256(public_key) — stable, collision-resistant.
        let mut key_id = [0u8; 32];
        key_id.copy_from_slice(&Sha3_256::digest(&public_key));

        info!("Generated {} key pair (ID: {})", algorithm.name(), hex::encode(&key_id[0..8]));

        Ok(Self {
            algorithm,
            public_key,
            secret_key,
            created_at: SystemTime::now(),
            key_id,
        })
    }

    /// Get public key
    pub fn public_key(&self) -> &[u8] {
        &self.public_key
    }

    /// Encapsulate a shared secret to a peer's public key (KEM initiator→responder).
    ///
    /// Returns `(ciphertext, shared_secret)`. The ciphertext is sent to the peer, who
    /// recovers the same `shared_secret` via [`decapsulate`]. Real Kyber ⊕ X25519
    /// crypto — no more zero secrets.
    pub fn encapsulate_to(&self, peer_public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        if !self.algorithm.is_kem() {
            return Err(anyhow!("Algorithm is not a KEM"));
        }
        pq_kem::encapsulate(self.algorithm, peer_public_key)
    }

    /// Decapsulate a shared secret from a ciphertext using our secret key (KEM responder→initiator).
    pub fn decapsulate(&self, ciphertext: &[u8]) -> Result<Vec<u8>> {
        if !self.algorithm.is_kem() {
            return Err(anyhow!("Algorithm is not a KEM"));
        }
        pq_kem::decapsulate(self.algorithm, ciphertext, &self.secret_key)
    }

    /// Sign a message (for signature algorithms)
    pub fn sign(&self, _message: &[u8]) -> Result<Vec<u8>> {
        if !self.algorithm.is_signature() {
            return Err(anyhow!("Algorithm is not a signature scheme"));
        }

        // Placeholder - in production use actual signature
        let signature = vec![0u8; self.algorithm.ciphertext_size()];
        Ok(signature)
    }

    /// Verify a signature (for signature algorithms)
    pub fn verify(&self, _message: &[u8], _signature: &[u8]) -> Result<bool> {
        if !self.algorithm.is_signature() {
            return Err(anyhow!("Algorithm is not a signature scheme"));
        }

        // Placeholder - in production use actual verification
        Ok(true)
    }
}

/// Configuration for quantum-resistant upgrades
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResistantConfig {
    /// Current migration phase
    pub migration_phase: MigrationPhase,
    /// Preferred KEM algorithm
    pub preferred_kem: PQAlgorithm,
    /// Preferred signature algorithm
    pub preferred_signature: PQAlgorithm,
    /// Minimum acceptable security level
    pub min_security_level: u8,
    /// Enable algorithm negotiation
    pub enable_negotiation: bool,
    /// Supported algorithms (in preference order)
    pub supported_kems: Vec<PQAlgorithm>,
    /// Supported signature algorithms (in preference order)
    pub supported_signatures: Vec<PQAlgorithm>,
    /// Key rotation interval
    pub key_rotation_interval: Duration,
    /// Enable backward compatibility
    pub backward_compatible: bool,
}

impl Default for QuantumResistantConfig {
    fn default() -> Self {
        Self {
            migration_phase: MigrationPhase::HybridOptional,
            // Whitepaper §8.3 Q3: hybrid Kyber-1024 + X25519 (NIST Level 5) is the target.
            preferred_kem: PQAlgorithm::HybridX25519Kyber1024,
            preferred_signature: PQAlgorithm::Dilithium3,
            min_security_level: 3,
            enable_negotiation: true,
            supported_kems: vec![
                PQAlgorithm::HybridX25519Kyber1024,
                PQAlgorithm::HybridX25519Kyber768,
                PQAlgorithm::Kyber1024,
                PQAlgorithm::Kyber768,
            ],
            supported_signatures: vec![
                PQAlgorithm::Dilithium3,
                PQAlgorithm::Dilithium5,
                PQAlgorithm::SphincsShake192f,
            ],
            key_rotation_interval: Duration::from_secs(86400), // 24 hours
            backward_compatible: true,
        }
    }
}

/// Statistics for quantum-resistant operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantumResistantStats {
    /// Key pairs generated
    pub keys_generated: u64,
    /// KEM encapsulations
    pub encapsulations: u64,
    /// KEM decapsulations
    pub decapsulations: u64,
    /// Signatures created
    pub signatures_created: u64,
    /// Signatures verified
    pub signatures_verified: u64,
    /// Hybrid connections established
    pub hybrid_connections: u64,
    /// Classical fallback connections
    pub classical_fallbacks: u64,
    /// Algorithm negotiation successes
    pub negotiation_successes: u64,
    /// Algorithm negotiation failures
    pub negotiation_failures: u64,
}

/// Quantum-resistant circuit handshake
#[derive(Debug, Clone)]
pub struct PQHandshake {
    /// Our ephemeral KEM key pair
    our_kem_keypair: PQKeyPair,
    /// Our ephemeral signature key pair (for authentication)
    our_sig_keypair: Option<PQKeyPair>,
    /// Peer's public KEM key
    peer_kem_pubkey: Option<Vec<u8>>,
    /// Peer's public signature key
    peer_sig_pubkey: Option<Vec<u8>>,
    /// Derived shared secret
    shared_secret: Option<Vec<u8>>,
    /// Handshake state
    state: HandshakeState,
    /// Creation time
    created_at: Instant,
}

/// Handshake state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeState {
    /// Initial state
    Initial,
    /// Sent our public key
    SentPubkey,
    /// Received peer's public key
    ReceivedPubkey,
    /// Shared secret derived
    Completed,
    /// Handshake failed
    Failed,
}

impl PQHandshake {
    /// Create a new handshake initiator
    pub fn new_initiator(config: &QuantumResistantConfig) -> Result<Self> {
        let kem_keypair = PQKeyPair::generate(config.preferred_kem)?;
        let sig_keypair = PQKeyPair::generate(config.preferred_signature)?;

        Ok(Self {
            our_kem_keypair: kem_keypair,
            our_sig_keypair: Some(sig_keypair),
            peer_kem_pubkey: None,
            peer_sig_pubkey: None,
            shared_secret: None,
            state: HandshakeState::Initial,
            created_at: Instant::now(),
        })
    }

    /// Create a new handshake responder
    pub fn new_responder(config: &QuantumResistantConfig) -> Result<Self> {
        let kem_keypair = PQKeyPair::generate(config.preferred_kem)?;

        Ok(Self {
            our_kem_keypair: kem_keypair,
            our_sig_keypair: None,
            peer_kem_pubkey: None,
            peer_sig_pubkey: None,
            shared_secret: None,
            state: HandshakeState::Initial,
            created_at: Instant::now(),
        })
    }

    /// Get our public key for sending
    pub fn our_public_key(&self) -> &[u8] {
        self.our_kem_keypair.public_key()
    }

    /// Process peer's public key
    pub fn receive_pubkey(&mut self, peer_pubkey: &[u8]) -> Result<()> {
        if self.state != HandshakeState::Initial && self.state != HandshakeState::SentPubkey {
            return Err(anyhow!("Invalid handshake state"));
        }

        self.peer_kem_pubkey = Some(peer_pubkey.to_vec());
        self.state = HandshakeState::ReceivedPubkey;
        Ok(())
    }

    /// Complete the handshake (as initiator)
    pub fn complete_as_initiator(&mut self, ciphertext: &[u8]) -> Result<Vec<u8>> {
        if self.state != HandshakeState::ReceivedPubkey {
            return Err(anyhow!("Invalid handshake state"));
        }

        let shared_secret = self.our_kem_keypair.decapsulate(ciphertext)?;
        self.shared_secret = Some(shared_secret.clone());
        self.state = HandshakeState::Completed;

        info!("PQ handshake completed (initiator) in {:?}", self.created_at.elapsed());
        Ok(shared_secret)
    }

    /// Complete the handshake (as responder)
    pub fn complete_as_responder(&mut self) -> Result<(Vec<u8>, Vec<u8>)> {
        if self.state != HandshakeState::ReceivedPubkey {
            return Err(anyhow!("Invalid handshake state"));
        }

        // Encapsulate to the PEER's (initiator's) public key — not our own.
        let peer_pubkey = self
            .peer_kem_pubkey
            .as_ref()
            .ok_or_else(|| anyhow!("cannot encapsulate: peer KEM public key not received"))?;
        let (ciphertext, shared_secret) = self.our_kem_keypair.encapsulate_to(peer_pubkey)?;
        self.shared_secret = Some(shared_secret.clone());
        self.state = HandshakeState::Completed;

        info!("PQ handshake completed (responder) in {:?}", self.created_at.elapsed());
        Ok((ciphertext, shared_secret))
    }

    /// Get the shared secret (after handshake completion)
    pub fn shared_secret(&self) -> Option<&[u8]> {
        self.shared_secret.as_deref()
    }

    /// Check if handshake is complete
    pub fn is_complete(&self) -> bool {
        self.state == HandshakeState::Completed
    }
}

/// Quantum-resistant manager
pub struct QuantumResistantManager {
    config: QuantumResistantConfig,
    /// Long-term identity key pairs
    identity_keys: Arc<RwLock<HashMap<PQAlgorithm, PQKeyPair>>>,
    /// Statistics
    stats: Arc<RwLock<QuantumResistantStats>>,
    /// Last key rotation
    last_rotation: Arc<RwLock<Option<Instant>>>,
}

impl QuantumResistantManager {
    /// Create a new quantum-resistant manager
    pub fn new(config: QuantumResistantConfig) -> Self {
        info!("🔐 Creating Quantum-Resistant Manager");
        info!("   Migration phase: {}", config.migration_phase.name());
        info!("   Preferred KEM: {}", config.preferred_kem.name());
        info!("   Preferred signature: {}", config.preferred_signature.name());
        info!("   Min security level: NIST Level {}", config.min_security_level);

        Self {
            config,
            identity_keys: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(QuantumResistantStats::default())),
            last_rotation: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize identity keys
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing quantum-resistant identity keys...");

        // Generate identity keys for supported algorithms
        let mut keys = self.identity_keys.write().await;

        for alg in &self.config.supported_kems {
            if !keys.contains_key(alg) {
                let keypair = PQKeyPair::generate(*alg)?;
                keys.insert(*alg, keypair);
            }
        }

        for alg in &self.config.supported_signatures {
            if !keys.contains_key(alg) {
                let keypair = PQKeyPair::generate(*alg)?;
                keys.insert(*alg, keypair);
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.keys_generated = keys.len() as u64;

        // Set rotation time
        let mut rotation = self.last_rotation.write().await;
        *rotation = Some(Instant::now());

        info!("✅ Initialized {} quantum-resistant key pairs", keys.len());
        Ok(())
    }

    /// Rotate identity keys if needed
    pub async fn maybe_rotate_keys(&self) -> Result<bool> {
        let should_rotate = {
            let rotation = self.last_rotation.read().await;
            match *rotation {
                Some(last) => last.elapsed() >= self.config.key_rotation_interval,
                None => true,
            }
        };

        if should_rotate {
            self.rotate_keys().await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Force key rotation
    pub async fn rotate_keys(&self) -> Result<()> {
        info!("🔄 Rotating quantum-resistant keys...");

        let mut keys = self.identity_keys.write().await;
        let algs: Vec<PQAlgorithm> = keys.keys().cloned().collect();

        for alg in algs {
            let keypair = PQKeyPair::generate(alg)?;
            keys.insert(alg, keypair);
        }

        let mut stats = self.stats.write().await;
        stats.keys_generated += keys.len() as u64;

        let mut rotation = self.last_rotation.write().await;
        *rotation = Some(Instant::now());

        info!("✅ Rotated {} keys", keys.len());
        Ok(())
    }

    /// Get identity public key for an algorithm
    pub async fn get_public_key(&self, algorithm: PQAlgorithm) -> Option<Vec<u8>> {
        let keys = self.identity_keys.read().await;
        keys.get(&algorithm).map(|k| k.public_key.clone())
    }

    /// Create a new handshake initiator
    pub fn create_handshake(&self) -> Result<PQHandshake> {
        PQHandshake::new_initiator(&self.config)
    }

    /// Create a new handshake responder
    pub fn create_responder(&self) -> Result<PQHandshake> {
        PQHandshake::new_responder(&self.config)
    }

    /// Negotiate algorithm with peer
    pub fn negotiate_algorithm(&self, peer_supported: &[PQAlgorithm]) -> Option<PQAlgorithm> {
        if !self.config.enable_negotiation {
            return Some(self.config.preferred_kem);
        }

        // Find first mutually supported algorithm with sufficient security level
        for our_alg in &self.config.supported_kems {
            if our_alg.security_level() >= self.config.min_security_level
                && peer_supported.contains(our_alg)
            {
                return Some(*our_alg);
            }
        }

        // Fallback to preferred if backward compatible
        if self.config.backward_compatible {
            return Some(self.config.preferred_kem);
        }

        None
    }

    /// Check if peer is compatible
    pub fn is_peer_compatible(&self, peer_algorithms: &[PQAlgorithm]) -> bool {
        if self.config.migration_phase == MigrationPhase::ClassicalOnly {
            return true; // Always compatible in classical mode
        }

        if !self.config.migration_phase.require_pq() {
            return true; // Don't require PQ, so compatible
        }

        // Require at least one mutually supported PQ algorithm
        peer_algorithms.iter().any(|a| {
            a.is_kem() && self.config.supported_kems.contains(a)
        })
    }

    /// Record encapsulation operation
    pub async fn record_encapsulation(&self) {
        let mut stats = self.stats.write().await;
        stats.encapsulations += 1;
    }

    /// Record decapsulation operation
    pub async fn record_decapsulation(&self) {
        let mut stats = self.stats.write().await;
        stats.decapsulations += 1;
    }

    /// Record hybrid connection
    pub async fn record_hybrid_connection(&self) {
        let mut stats = self.stats.write().await;
        stats.hybrid_connections += 1;
    }

    /// Record classical fallback
    pub async fn record_classical_fallback(&self) {
        let mut stats = self.stats.write().await;
        stats.classical_fallbacks += 1;
    }

    /// Get statistics
    pub async fn get_stats(&self) -> QuantumResistantStats {
        self.stats.read().await.clone()
    }

    /// Get current migration phase
    pub fn migration_phase(&self) -> MigrationPhase {
        self.config.migration_phase
    }

    /// Get config
    pub fn config(&self) -> &QuantumResistantConfig {
        &self.config
    }
}

/// 🔐 Real hybrid KEM crypto: CRYSTALS-Kyber (pqcrypto) ⊕ X25519 (dalek).
///
/// Wire format for a hybrid `X25519+Kyber-N`:
/// - public key  = `x25519_pub(32) || kyber_pub`
/// - secret key  = `x25519_secret(32) || kyber_secret`
/// - ciphertext  = `x25519_ephemeral_pub(32) || kyber_ciphertext`
/// - shared secret = `SHA3-256("qnk-hybrid-kem-v1" || x25519_ss || kyber_ss)` (32 bytes)
///
/// The SHA3 combiner is the standard concatenation KDF: the result stays secret as long
/// as EITHER the classical or the post-quantum half is unbroken — that's the whole point
/// of a hybrid (safe against both a Kyber break and a future quantum X25519 break).
mod pq_kem {
    use super::PQAlgorithm;
    use anyhow::{anyhow, Result};
    use pqcrypto_traits::kem::{
        Ciphertext as _, PublicKey as _, SecretKey as _, SharedSecret as _,
    };
    use rand::RngCore;
    use sha3::{Digest, Sha3_256};
    use x25519_dalek::{PublicKey as XPublicKey, StaticSecret as XStaticSecret};

    const X25519_LEN: usize = 32;

    #[derive(Clone, Copy)]
    enum KyberVariant {
        K512,
        K768,
        K1024,
    }

    /// Generate `(public_key, secret_key)` bytes for a KEM algorithm.
    pub fn generate(alg: PQAlgorithm) -> Result<(Vec<u8>, Vec<u8>)> {
        match alg {
            PQAlgorithm::Kyber512 => kyber_keypair(KyberVariant::K512),
            PQAlgorithm::Kyber768 => kyber_keypair(KyberVariant::K768),
            PQAlgorithm::Kyber1024 => kyber_keypair(KyberVariant::K1024),
            PQAlgorithm::HybridX25519Kyber768 => hybrid_keypair(KyberVariant::K768),
            PQAlgorithm::HybridX25519Kyber1024 => hybrid_keypair(KyberVariant::K1024),
            other => Err(anyhow!("KEM keygen not implemented for {}", other.name())),
        }
    }

    /// Encapsulate to a peer's public key → `(ciphertext, shared_secret)`.
    pub fn encapsulate(alg: PQAlgorithm, peer_pk: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        match alg {
            PQAlgorithm::Kyber512 => kyber_encapsulate(KyberVariant::K512, peer_pk),
            PQAlgorithm::Kyber768 => kyber_encapsulate(KyberVariant::K768, peer_pk),
            PQAlgorithm::Kyber1024 => kyber_encapsulate(KyberVariant::K1024, peer_pk),
            PQAlgorithm::HybridX25519Kyber768 => hybrid_encapsulate(KyberVariant::K768, peer_pk),
            PQAlgorithm::HybridX25519Kyber1024 => hybrid_encapsulate(KyberVariant::K1024, peer_pk),
            other => Err(anyhow!("KEM encapsulate not implemented for {}", other.name())),
        }
    }

    /// Decapsulate `ciphertext` with our `secret_key` → `shared_secret`.
    pub fn decapsulate(alg: PQAlgorithm, ct: &[u8], sk: &[u8]) -> Result<Vec<u8>> {
        match alg {
            PQAlgorithm::Kyber512 => kyber_decapsulate(KyberVariant::K512, ct, sk),
            PQAlgorithm::Kyber768 => kyber_decapsulate(KyberVariant::K768, ct, sk),
            PQAlgorithm::Kyber1024 => kyber_decapsulate(KyberVariant::K1024, ct, sk),
            PQAlgorithm::HybridX25519Kyber768 => hybrid_decapsulate(KyberVariant::K768, ct, sk),
            PQAlgorithm::HybridX25519Kyber1024 => hybrid_decapsulate(KyberVariant::K1024, ct, sk),
            other => Err(anyhow!("KEM decapsulate not implemented for {}", other.name())),
        }
    }

    // ── Kyber (pqcrypto). Each variant is a distinct module/type → dispatch per variant. ──
    fn kyber_keypair(v: KyberVariant) -> Result<(Vec<u8>, Vec<u8>)> {
        Ok(match v {
            KyberVariant::K512 => {
                let (pk, sk) = pqcrypto_kyber::kyber512::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            KyberVariant::K768 => {
                let (pk, sk) = pqcrypto_kyber::kyber768::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
            KyberVariant::K1024 => {
                let (pk, sk) = pqcrypto_kyber::kyber1024::keypair();
                (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
            }
        })
    }

    fn kyber_encapsulate(v: KyberVariant, peer_pk: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        Ok(match v {
            KyberVariant::K512 => {
                let pk = pqcrypto_kyber::kyber512::PublicKey::from_bytes(peer_pk)
                    .map_err(|e| anyhow!("bad kyber512 pk: {e:?}"))?;
                let (ss, ct) = pqcrypto_kyber::kyber512::encapsulate(&pk);
                (ct.as_bytes().to_vec(), ss.as_bytes().to_vec())
            }
            KyberVariant::K768 => {
                let pk = pqcrypto_kyber::kyber768::PublicKey::from_bytes(peer_pk)
                    .map_err(|e| anyhow!("bad kyber768 pk: {e:?}"))?;
                let (ss, ct) = pqcrypto_kyber::kyber768::encapsulate(&pk);
                (ct.as_bytes().to_vec(), ss.as_bytes().to_vec())
            }
            KyberVariant::K1024 => {
                let pk = pqcrypto_kyber::kyber1024::PublicKey::from_bytes(peer_pk)
                    .map_err(|e| anyhow!("bad kyber1024 pk: {e:?}"))?;
                let (ss, ct) = pqcrypto_kyber::kyber1024::encapsulate(&pk);
                (ct.as_bytes().to_vec(), ss.as_bytes().to_vec())
            }
        })
    }

    fn kyber_decapsulate(v: KyberVariant, ct: &[u8], sk: &[u8]) -> Result<Vec<u8>> {
        Ok(match v {
            KyberVariant::K512 => {
                let ct = pqcrypto_kyber::kyber512::Ciphertext::from_bytes(ct)
                    .map_err(|e| anyhow!("bad kyber512 ct: {e:?}"))?;
                let sk = pqcrypto_kyber::kyber512::SecretKey::from_bytes(sk)
                    .map_err(|e| anyhow!("bad kyber512 sk: {e:?}"))?;
                pqcrypto_kyber::kyber512::decapsulate(&ct, &sk).as_bytes().to_vec()
            }
            KyberVariant::K768 => {
                let ct = pqcrypto_kyber::kyber768::Ciphertext::from_bytes(ct)
                    .map_err(|e| anyhow!("bad kyber768 ct: {e:?}"))?;
                let sk = pqcrypto_kyber::kyber768::SecretKey::from_bytes(sk)
                    .map_err(|e| anyhow!("bad kyber768 sk: {e:?}"))?;
                pqcrypto_kyber::kyber768::decapsulate(&ct, &sk).as_bytes().to_vec()
            }
            KyberVariant::K1024 => {
                let ct = pqcrypto_kyber::kyber1024::Ciphertext::from_bytes(ct)
                    .map_err(|e| anyhow!("bad kyber1024 ct: {e:?}"))?;
                let sk = pqcrypto_kyber::kyber1024::SecretKey::from_bytes(sk)
                    .map_err(|e| anyhow!("bad kyber1024 sk: {e:?}"))?;
                pqcrypto_kyber::kyber1024::decapsulate(&ct, &sk).as_bytes().to_vec()
            }
        })
    }

    // ── X25519 (dalek). We use StaticSecret-from-seed to keep keys serializable. ──
    fn x25519_keypair() -> ([u8; X25519_LEN], [u8; X25519_LEN]) {
        let mut seed = [0u8; X25519_LEN];
        rand::thread_rng().fill_bytes(&mut seed);
        let sk = XStaticSecret::from(seed);
        let pk = XPublicKey::from(&sk);
        (*pk.as_bytes(), sk.to_bytes())
    }

    fn arr32(b: &[u8]) -> Result<[u8; X25519_LEN]> {
        if b.len() != X25519_LEN {
            return Err(anyhow!("expected 32-byte x25519 component, got {}", b.len()));
        }
        let mut a = [0u8; X25519_LEN];
        a.copy_from_slice(b);
        Ok(a)
    }

    // ── Hybrid: x25519 ⊕ kyber, SHA3-256 combiner ──
    fn hybrid_keypair(v: KyberVariant) -> Result<(Vec<u8>, Vec<u8>)> {
        let (x_pk, x_sk) = x25519_keypair();
        let (k_pk, k_sk) = kyber_keypair(v)?;
        let mut pk = x_pk.to_vec();
        pk.extend_from_slice(&k_pk);
        let mut sk = x_sk.to_vec();
        sk.extend_from_slice(&k_sk);
        Ok((pk, sk))
    }

    fn hybrid_encapsulate(v: KyberVariant, peer_pk: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        if peer_pk.len() <= X25519_LEN {
            return Err(anyhow!("hybrid peer public key too short"));
        }
        let (x_peer_pk, k_peer_pk) = peer_pk.split_at(X25519_LEN);
        // Classical: ephemeral X25519 DH against the peer's static X25519 key.
        let mut eph_seed = [0u8; X25519_LEN];
        rand::thread_rng().fill_bytes(&mut eph_seed);
        let eph_sk = XStaticSecret::from(eph_seed);
        let eph_pk = XPublicKey::from(&eph_sk);
        let x_ss = eph_sk.diffie_hellman(&XPublicKey::from(arr32(x_peer_pk)?));
        // Post-quantum: Kyber encapsulation against the peer's Kyber key.
        let (k_ct, k_ss) = kyber_encapsulate(v, k_peer_pk)?;
        let mut ct = eph_pk.as_bytes().to_vec();
        ct.extend_from_slice(&k_ct);
        Ok((ct, combine(x_ss.as_bytes(), &k_ss)))
    }

    fn hybrid_decapsulate(v: KyberVariant, ct: &[u8], sk: &[u8]) -> Result<Vec<u8>> {
        if ct.len() <= X25519_LEN || sk.len() <= X25519_LEN {
            return Err(anyhow!("hybrid ciphertext/secret too short"));
        }
        let (x_ct, k_ct) = ct.split_at(X25519_LEN);
        let (x_sk_b, k_sk) = sk.split_at(X25519_LEN);
        let x_sk = XStaticSecret::from(arr32(x_sk_b)?);
        let x_ss = x_sk.diffie_hellman(&XPublicKey::from(arr32(x_ct)?));
        let k_ss = kyber_decapsulate(v, k_ct, k_sk)?;
        Ok(combine(x_ss.as_bytes(), &k_ss))
    }

    /// Concatenation KDF: `SHA3-256(domain || classical_ss || pq_ss)`.
    fn combine(classical_ss: &[u8], pq_ss: &[u8]) -> Vec<u8> {
        let mut h = Sha3_256::new();
        h.update(b"qnk-hybrid-kem-v1");
        h.update(classical_ss);
        h.update(pq_ss);
        h.finalize().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_properties() {
        assert!(PQAlgorithm::Kyber768.is_kem());
        assert!(!PQAlgorithm::Kyber768.is_signature());
        assert!(PQAlgorithm::Dilithium3.is_signature());
        assert!(PQAlgorithm::HybridX25519Kyber768.is_hybrid());
        assert_eq!(PQAlgorithm::Kyber768.security_level(), 3);
    }

    #[test]
    fn test_key_generation() {
        let keypair = PQKeyPair::generate(PQAlgorithm::Kyber768).unwrap();
        assert_eq!(keypair.algorithm, PQAlgorithm::Kyber768);
        assert!(!keypair.public_key.is_empty());
    }

    #[tokio::test]
    async fn test_manager_initialization() {
        let config = QuantumResistantConfig::default();
        let manager = QuantumResistantManager::new(config);

        manager.initialize().await.unwrap();

        let stats = manager.get_stats().await;
        assert!(stats.keys_generated > 0);
    }

    #[test]
    fn test_algorithm_negotiation() {
        let config = QuantumResistantConfig::default();
        let manager = QuantumResistantManager::new(config);

        // Peer supports Kyber768
        let peer_algs = vec![PQAlgorithm::Kyber768, PQAlgorithm::Kyber512];
        let negotiated = manager.negotiate_algorithm(&peer_algs);
        assert!(negotiated.is_some());
    }

    #[test]
    fn test_migration_phases() {
        assert!(MigrationPhase::ClassicalOnly.accept_classical());
        assert!(!MigrationPhase::PostQuantumOnly.accept_classical());
        assert!(MigrationPhase::HybridPreferred.prefer_hybrid());
        assert!(MigrationPhase::HybridRequired.require_pq());
    }

    #[test]
    fn test_handshake_creation() {
        let config = QuantumResistantConfig::default();
        let handshake = PQHandshake::new_initiator(&config).unwrap();
        assert!(!handshake.our_public_key().is_empty());
        assert!(!handshake.is_complete());
    }

    /// Round-trip: initiator and responder MUST derive the same, NON-ZERO shared secret.
    /// This is the regression guard against the old `vec![0u8; 32]` placeholder.
    fn kem_roundtrip(alg: PQAlgorithm) {
        // Initiator generates a keypair and publishes its public key.
        let initiator = PQKeyPair::generate(alg).unwrap();
        assert_eq!(initiator.public_key().len(), alg.public_key_size(),
            "{} public key size mismatch", alg.name());

        // Responder encapsulates to the initiator's public key.
        let responder = PQKeyPair::generate(alg).unwrap();
        let (ciphertext, ss_responder) = responder.encapsulate_to(initiator.public_key()).unwrap();
        assert_eq!(ciphertext.len(), alg.ciphertext_size(),
            "{} ciphertext size mismatch", alg.name());

        // Initiator decapsulates with its secret key.
        let ss_initiator = initiator.decapsulate(&ciphertext).unwrap();

        assert_eq!(ss_initiator, ss_responder, "{} shared secrets differ", alg.name());
        assert_eq!(ss_initiator.len(), 32);
        assert_ne!(ss_initiator, vec![0u8; 32], "{} shared secret is all-zero (placeholder!)", alg.name());
    }

    #[test]
    fn test_hybrid_x25519_kyber1024_roundtrip() {
        // The whitepaper §8.3 target.
        kem_roundtrip(PQAlgorithm::HybridX25519Kyber1024);
        assert_eq!(PQAlgorithm::HybridX25519Kyber1024.security_level(), 5);
        assert!(PQAlgorithm::HybridX25519Kyber1024.is_kem());
        assert!(PQAlgorithm::HybridX25519Kyber1024.is_hybrid());
    }

    #[test]
    fn test_kem_roundtrips_all_variants() {
        kem_roundtrip(PQAlgorithm::Kyber512);
        kem_roundtrip(PQAlgorithm::Kyber768);
        kem_roundtrip(PQAlgorithm::Kyber1024);
        kem_roundtrip(PQAlgorithm::HybridX25519Kyber768);
        kem_roundtrip(PQAlgorithm::HybridX25519Kyber1024);
    }

    /// Full handshake state machine derives a matching secret end-to-end.
    #[test]
    fn test_pq_handshake_end_to_end() {
        let config = QuantumResistantConfig::default(); // preferred_kem = HybridX25519Kyber1024
        let mut initiator = PQHandshake::new_initiator(&config).unwrap();
        let mut responder = PQHandshake::new_responder(&config).unwrap();

        // Initiator → responder: public key.
        responder.receive_pubkey(initiator.our_public_key()).unwrap();
        // Responder → initiator: ciphertext + derives its secret.
        let (ciphertext, ss_responder) = responder.complete_as_responder().unwrap();
        // Initiator needs responder's pubkey state set before completing (state machine).
        initiator.receive_pubkey(responder.our_public_key()).unwrap();
        let ss_initiator = initiator.complete_as_initiator(&ciphertext).unwrap();

        assert_eq!(ss_initiator, ss_responder);
        assert_ne!(ss_initiator, vec![0u8; 32]);
        assert!(initiator.is_complete() && responder.is_complete());
    }
}
