//! Post-Quantum Hybrid Handshake for Q-NarwhalKnight
//! v4.3.0-beta: Application-level Kyber1024 key exchange on top of standard Noise XX
//!
//! Strategy: Classical Noise XX (X25519) handles the libp2p transport layer.
//! After connection, we perform Kyber1024 encapsulation via request-response
//! to establish a post-quantum shared secret. The combined key material provides
//! hybrid security: even if one primitive breaks, the other protects.

use async_trait::async_trait;
use dashmap::DashMap;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use libp2p::request_response::Codec;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::io;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use unsigned_varint::{aio, encode};

/// Protocol ID for PQ handshake request-response
pub const PQ_HANDSHAKE_PROTOCOL: &str = "/qnk/pq-handshake/1.0.0";

/// PQ Handshake configuration
#[derive(Debug, Clone)]
pub struct PQHandshakeConfig {
    /// Whether PQ handshake is enabled (env: Q_PQ_HANDSHAKE=1)
    pub enabled: bool,
    /// Whether to require PQ handshake (reject classical-only peers)
    pub require_pq: bool,
    /// Handshake timeout
    pub timeout: Duration,
}

impl Default for PQHandshakeConfig {
    fn default() -> Self {
        Self {
            enabled: std::env::var("Q_PQ_HANDSHAKE")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false),
            require_pq: false,
            timeout: Duration::from_secs(10),
        }
    }
}

/// Result of a PQ handshake with a peer
#[derive(Debug, Clone)]
pub struct PQHandshakeResult {
    pub peer_id: PeerId,
    /// Whether PQ key exchange was completed
    pub pq_capable: bool,
    /// Combined key material (SHA3-256 of noise_key || kyber_secret), 32 bytes
    pub combined_key: Option<[u8; 32]>,
    /// Timestamp of handshake completion
    pub completed_at: Instant,
}

/// PQ Handshake request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQHandshakeRequest {
    /// Kyber1024 public key (1568 bytes)
    pub kyber_public_key: Vec<u8>,
    /// Node identifier
    pub node_id: String,
    /// Protocol version
    pub version: u32,
}

/// PQ Handshake response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQHandshakeResponse {
    /// Kyber1024 ciphertext (encapsulated shared secret)
    pub kyber_ciphertext: Vec<u8>,
    /// Responder's Kyber1024 public key (for mutual authentication)
    pub responder_public_key: Vec<u8>,
    /// Whether responder supports PQ
    pub pq_supported: bool,
    /// Protocol version
    pub version: u32,
}

// ============================================================================
// Codec Implementation (follows HandshakeCodec pattern from handshake_validator.rs)
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct PQHandshakeCodec;

#[async_trait]
impl Codec for PQHandshakeCodec {
    type Protocol = &'static str;
    type Request = PQHandshakeRequest;
    type Response = PQHandshakeResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let len = aio::read_usize(&mut *io)
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if len > 16_384 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "PQ handshake request too large"));
        }
        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;
        serde_json::from_slice(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let len = aio::read_usize(&mut *io)
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if len > 16_384 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "PQ handshake response too large"));
        }
        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;
        serde_json::from_slice(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let msg_bytes = serde_json::to_vec(&req)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut len_buf = encode::usize_buffer();
        let encoded_len = encode::usize(msg_bytes.len(), &mut len_buf);
        io.write_all(encoded_len).await?;
        io.write_all(&msg_bytes).await?;
        io.close().await?;
        Ok(())
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let msg_bytes = serde_json::to_vec(&res)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut len_buf = encode::usize_buffer();
        let encoded_len = encode::usize(msg_bytes.len(), &mut len_buf);
        io.write_all(encoded_len).await?;
        io.write_all(&msg_bytes).await?;
        io.close().await?;
        Ok(())
    }
}

// ============================================================================
// Session Manager
// ============================================================================

/// Manages PQ handshake sessions per peer
pub struct PQSessionManager {
    sessions: DashMap<PeerId, PQHandshakeResult>,
}

impl PQSessionManager {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
        }
    }

    /// Check if a peer has completed PQ handshake
    pub fn is_pq_secured(&self, peer_id: &PeerId) -> bool {
        self.sessions.get(peer_id)
            .map(|r| r.pq_capable && r.combined_key.is_some())
            .unwrap_or(false)
    }

    /// Get session result for a peer
    pub fn get_session(&self, peer_id: &PeerId) -> Option<PQHandshakeResult> {
        self.sessions.get(peer_id).map(|r| r.value().clone())
    }

    /// Store a completed handshake result
    pub fn store_session(&self, result: PQHandshakeResult) {
        info!("[PQ] Stored session for peer {} (pq_capable={})", result.peer_id, result.pq_capable);
        self.sessions.insert(result.peer_id, result);
    }

    /// Remove session on disconnect
    pub fn remove_session(&self, peer_id: &PeerId) {
        self.sessions.remove(peer_id);
    }

    /// Count of PQ-secured peers
    pub fn pq_peer_count(&self) -> usize {
        self.sessions.iter().filter(|r| r.pq_capable).count()
    }

    /// Total tracked peers
    pub fn total_sessions(&self) -> usize {
        self.sessions.len()
    }
}

impl Default for PQSessionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Kyber1024 Key Operations
// ============================================================================

/// Generate a Kyber1024 key pair
/// Returns (public_key, secret_key) as byte vectors
pub fn create_kyber_keypair() -> (Vec<u8>, Vec<u8>) {
    use pqcrypto_kyber::kyber1024;
    use pqcrypto_traits::kem::{PublicKey, SecretKey};

    let (pk, sk) = kyber1024::keypair();
    (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
}

/// Encapsulate a shared secret using a Kyber1024 public key
/// Returns (ciphertext, shared_secret)
pub fn encapsulate_key(public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    use pqcrypto_kyber::kyber1024;
    use pqcrypto_traits::kem::{PublicKey, SecretKey, SharedSecret, Ciphertext};

    let pk = kyber1024::PublicKey::from_bytes(public_key)
        .map_err(|e| format!("Invalid Kyber1024 public key: {:?}", e))?;
    let (ss, ct) = kyber1024::encapsulate(&pk);
    Ok((ct.as_bytes().to_vec(), ss.as_bytes().to_vec()))
}

/// Decapsulate a shared secret using a Kyber1024 secret key and ciphertext
pub fn decapsulate_key(ciphertext: &[u8], secret_key: &[u8]) -> Result<Vec<u8>, String> {
    use pqcrypto_kyber::kyber1024;
    use pqcrypto_traits::kem::{PublicKey, SecretKey, SharedSecret, Ciphertext};

    let ct = kyber1024::Ciphertext::from_bytes(ciphertext)
        .map_err(|e| format!("Invalid Kyber1024 ciphertext: {:?}", e))?;
    let sk = kyber1024::SecretKey::from_bytes(secret_key)
        .map_err(|e| format!("Invalid Kyber1024 secret key: {:?}", e))?;
    let ss = kyber1024::decapsulate(&ct, &sk);
    Ok(ss.as_bytes().to_vec())
}

/// Combine classical and post-quantum key material using SHA3-256
/// Provides hybrid security: both must be broken to compromise the key
pub fn combine_keys(classical_key: &[u8], pq_secret: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(classical_key);
    hasher.update(pq_secret);
    let result = hasher.finalize();
    let mut combined = [0u8; 32];
    combined.copy_from_slice(&result);
    combined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kyber_roundtrip() {
        let (pk, sk) = create_kyber_keypair();
        assert!(!pk.is_empty(), "Public key should not be empty");
        assert!(!sk.is_empty(), "Secret key should not be empty");

        let (ct, ss_enc) = encapsulate_key(&pk).expect("Encapsulation should succeed");
        let ss_dec = decapsulate_key(&ct, &sk).expect("Decapsulation should succeed");

        assert_eq!(ss_enc, ss_dec, "Shared secrets must match after round-trip");
    }

    #[test]
    fn test_combine_keys_deterministic() {
        let classical = b"classical_key_material_32_bytes!";
        let pq_secret = b"post_quantum_shared_secret_here!";

        let combined1 = combine_keys(classical, pq_secret);
        let combined2 = combine_keys(classical, pq_secret);

        assert_eq!(combined1, combined2, "Key combination must be deterministic");
    }

    #[test]
    fn test_combine_keys_different_inputs() {
        let key_a = combine_keys(b"key_a", b"secret_a");
        let key_b = combine_keys(b"key_b", b"secret_a");
        let key_c = combine_keys(b"key_a", b"secret_b");

        assert_ne!(key_a, key_b, "Different classical keys should produce different combined keys");
        assert_ne!(key_a, key_c, "Different PQ secrets should produce different combined keys");
    }

    #[test]
    fn test_session_manager() {
        let manager = PQSessionManager::new();
        let peer = PeerId::random();

        assert!(!manager.is_pq_secured(&peer));

        manager.store_session(PQHandshakeResult {
            peer_id: peer,
            pq_capable: true,
            combined_key: Some([42u8; 32]),
            completed_at: Instant::now(),
        });

        assert!(manager.is_pq_secured(&peer));
        assert_eq!(manager.pq_peer_count(), 1);

        manager.remove_session(&peer);
        assert!(!manager.is_pq_secured(&peer));
    }

    #[test]
    fn test_pq_config_default() {
        let config = PQHandshakeConfig::default();
        // Default: disabled unless Q_PQ_HANDSHAKE=1
        assert!(!config.require_pq);
        assert_eq!(config.timeout, Duration::from_secs(10));
    }
}
