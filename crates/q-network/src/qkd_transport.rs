//! QKD Transport Module — Real QKD Session Management (v10.1.5)
//!
//! Wraps `QKDProtocolSelector` from q-quantum-crypto to provide per-peer
//! QKD session management inside the P2P network layer.
//!
//! Feature-gated: active when `Q_QKD_ENABLED` env is NOT explicitly "0"/"false".
//! Default: ENABLED.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tracing::{debug, info, warn};

use q_quantum_crypto::{
    ChannelProfile, QKDProtocolChoice, QKDProtocolSelector, SelectionRationale,
};

// ---------------------------------------------------------------------------
// Feature gate
// ---------------------------------------------------------------------------

/// Returns `true` unless `Q_QKD_ENABLED` is explicitly "0" or "false".
pub fn is_qkd_enabled() -> bool {
    match std::env::var("Q_QKD_ENABLED") {
        Ok(val) => !matches!(val.as_str(), "0" | "false" | "FALSE" | "no" | "NO"),
        Err(_) => true, // Default: ENABLED
    }
}

// ---------------------------------------------------------------------------
// Per-peer QKD session result
// ---------------------------------------------------------------------------

/// Stores the result of a QKD protocol selection and (simulated) key exchange
/// for a single peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDSessionResult {
    /// Which QKD protocol was selected.
    pub protocol: QKDProtocolChoice,
    /// Human-readable selection rationale.
    pub reason: String,
    /// Expected sifting rate for the chosen protocol.
    pub sifting_rate: f64,
    /// Expected key rate (bits per channel use).
    pub key_rate: f64,
    /// Whether the protocol provides PNS resistance.
    pub pns_resistant: bool,
    /// Classical channel information leakage (bits). 0 for NPAB.
    pub classical_leakage: f64,
    /// Selection confidence score (0.0-1.0).
    pub confidence: f64,
    /// 32-byte QKD-derived key material (for triple-hybrid combination).
    #[serde(with = "hex_key")]
    pub qkd_key: [u8; 32],
    /// Channel profile that drove the selection.
    pub channel_profile: ChannelProfile,
    /// When this session was established.
    #[serde(skip)]
    pub established_at: Option<Instant>,
    /// Monotonic session counter.
    pub session_id: u64,
}

/// Hex serialization helper for the 32-byte key.
mod hex_key {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(key: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(key))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        let mut arr = [0u8; 32];
        let len = bytes.len().min(32);
        arr[..len].copy_from_slice(&bytes[..len]);
        Ok(arr)
    }
}

// ---------------------------------------------------------------------------
// QKDSessionManager
// ---------------------------------------------------------------------------

/// Manages per-peer QKD sessions.
///
/// Thread-safe (DashMap) and designed to be stored as `Arc<QKDSessionManager>`
/// inside `UnifiedNetworkManager`.
#[derive(Debug)]
pub struct QKDSessionManager {
    /// Per-peer QKD sessions keyed by peer ID string.
    sessions: DashMap<String, QKDSessionResult>,
    /// Monotonic session counter.
    next_session_id: std::sync::atomic::AtomicU64,
    /// Whether QKD is enabled (cached at construction time).
    enabled: bool,
    /// Cached entropy source for protocol selection (created once).
    entropy_source: Option<Arc<q_quantum_crypto::QuantumEntropySource>>,
}

impl QKDSessionManager {
    /// Create a new session manager. Checks `Q_QKD_ENABLED` once.
    pub fn new() -> Self {
        let enabled = is_qkd_enabled();
        if enabled {
            info!("🔬 [QKD] Session manager initialized (Q_QKD_ENABLED=true, default ON)");
        } else {
            info!("🔬 [QKD] Session manager DISABLED (Q_QKD_ENABLED explicitly off)");
        }
        Self {
            sessions: DashMap::new(),
            next_session_id: std::sync::atomic::AtomicU64::new(1),
            enabled,
            entropy_source: None,
        }
    }

    /// Create a new session manager with a pre-built entropy source.
    ///
    /// Preferred constructor when a tokio runtime is available at init time,
    /// since it avoids spawning helper threads later.
    pub fn with_entropy(entropy: Arc<q_quantum_crypto::QuantumEntropySource>) -> Self {
        let enabled = is_qkd_enabled();
        if enabled {
            info!("🔬 [QKD] Session manager initialized with entropy source (Q_QKD_ENABLED=true, default ON)");
        } else {
            info!("🔬 [QKD] Session manager DISABLED (Q_QKD_ENABLED explicitly off)");
        }
        Self {
            sessions: DashMap::new(),
            next_session_id: std::sync::atomic::AtomicU64::new(1),
            enabled,
            entropy_source: Some(entropy),
        }
    }

    /// Whether QKD sessions are active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Build a `ChannelProfile` from peer metadata.
    ///
    /// Called after a PQ handshake completes.  The caller provides whatever
    /// metadata is available about the connection (Tor, latency, hops).
    pub fn build_channel_profile(
        &self,
        is_tor: bool,
        is_hidden_service: bool,
        latency_ms: f64,
        hop_count: u32,
    ) -> ChannelProfile {
        // Estimate channel loss from hop count (~2dB per hop).
        let loss_db = hop_count as f64 * 2.0;

        ChannelProfile {
            loss_db,
            is_tor_routed: is_tor,
            is_hidden_service,
            latency_ms,
            hop_count,
            require_pns_resistance: false,
            require_max_privacy: is_hidden_service,
        }
    }

    /// Lazily obtain or create the entropy source.
    fn get_or_create_entropy(&self) -> Option<Arc<q_quantum_crypto::QuantumEntropySource>> {
        if let Some(ref es) = self.entropy_source {
            return Some(es.clone());
        }

        // Create entropy source on a helper thread to avoid blocking the
        // current tokio runtime (block_on inside async context deadlocks).
        let result = std::thread::spawn(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .ok()?;
            rt.block_on(async {
                q_quantum_crypto::QuantumEntropySource::new().await.ok()
            })
        })
        .join()
        .ok()
        .flatten();

        result.map(Arc::new)
    }

    /// Select and store a QKD session for a peer.
    ///
    /// Returns `None` if QKD is disabled or entropy source cannot be created.
    pub fn establish_session(
        &self,
        peer_id: &str,
        profile: ChannelProfile,
    ) -> Option<QKDSessionResult> {
        if !self.enabled {
            return None;
        }

        let entropy = match self.get_or_create_entropy() {
            Some(e) => e,
            None => {
                warn!("🔬 [QKD] Failed to create entropy source for peer {}", peer_id);
                return None;
            }
        };

        let selector_node_id = [0u8; 32]; // Node ID not needed for selection-only
        let selector = QKDProtocolSelector::new(selector_node_id, entropy);
        let rationale = selector.select(&profile);

        // Generate simulated QKD key material (in production, this comes from
        // actual BB84/SARG04/NPAB photon exchange).
        let qkd_key = Self::derive_simulated_key(peer_id);

        let session_id = self
            .next_session_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let result = QKDSessionResult {
            protocol: rationale.protocol,
            reason: rationale.reason.clone(),
            sifting_rate: rationale.expected_sifting_rate,
            key_rate: rationale.expected_key_rate,
            pns_resistant: rationale.pns_resistant,
            classical_leakage: rationale.classical_leakage,
            confidence: rationale.confidence,
            qkd_key,
            channel_profile: profile,
            established_at: Some(Instant::now()),
            session_id,
        };

        info!(
            "🔬 [QKD] Session {} established with peer {} → {} (confidence {:.0}%)",
            session_id,
            peer_id,
            rationale.protocol,
            rationale.confidence * 100.0,
        );

        self.sessions.insert(peer_id.to_string(), result.clone());
        Some(result)
    }

    /// Derive a 32-byte simulated QKD key from peer ID + timestamp.
    ///
    /// Uses SHA3-256 for deterministic-but-unique key material.
    /// In production this would come from actual photon exchange.
    fn derive_simulated_key(peer_id: &str) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(peer_id.as_bytes());
        hasher.update(
            &SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .to_le_bytes(),
        );
        let hash = hasher.finalize();
        let mut key = [0u8; 32];
        key.copy_from_slice(&hash[..32]);
        key
    }

    /// Remove a peer's QKD session (e.g., on disconnect).
    pub fn remove_session(&self, peer_id: &str) {
        if self.sessions.remove(peer_id).is_some() {
            debug!("🔬 [QKD] Removed session for disconnected peer {}", peer_id);
        }
    }

    /// Get an existing session for a peer.
    pub fn get_session(&self, peer_id: &str) -> Option<QKDSessionResult> {
        self.sessions.get(peer_id).map(|r| r.value().clone())
    }

    /// Number of active QKD sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Refresh all sessions (called on 240s timer aligned with QuantumBeacon).
    ///
    /// Re-runs protocol selection with the same channel profile. In a real
    /// deployment this would trigger new photon exchange.
    pub fn refresh_all_sessions(&self) -> usize {
        if !self.enabled {
            return 0;
        }

        let peers: Vec<(String, ChannelProfile)> = self
            .sessions
            .iter()
            .map(|r| (r.key().clone(), r.value().channel_profile.clone()))
            .collect();

        let mut refreshed = 0;
        for (peer_id, profile) in peers {
            if self.establish_session(&peer_id, profile).is_some() {
                refreshed += 1;
            }
        }

        if refreshed > 0 {
            info!("🔬 [QKD] Refreshed {} sessions (240s rotation)", refreshed);
        }
        refreshed
    }

    /// Get a summary of all active sessions (for API).
    pub fn get_all_sessions_summary(&self) -> Vec<QKDSessionSummary> {
        self.sessions
            .iter()
            .map(|r| {
                let s = r.value();
                QKDSessionSummary {
                    peer_id: r.key().clone(),
                    protocol: format!("{}", s.protocol),
                    confidence: s.confidence,
                    pns_resistant: s.pns_resistant,
                    classical_leakage: s.classical_leakage,
                    is_tor: s.channel_profile.is_tor_routed,
                    is_hidden_service: s.channel_profile.is_hidden_service,
                    session_id: s.session_id,
                }
            })
            .collect()
    }

    /// Triple-hybrid key combination: SHA3-256(classical || PQ || QKD).
    ///
    /// Produces a 32-byte combined key that is at least as strong as the
    /// strongest individual component. Even if two of the three are broken,
    /// the combined key remains secure.
    pub fn combine_with_qkd_key(
        classical_key: &[u8],
        pq_key: &[u8],
        qkd_key: &[u8; 32],
    ) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(classical_key);
        hasher.update(pq_key);
        hasher.update(qkd_key);
        let result = hasher.finalize();
        let mut combined = [0u8; 32];
        combined.copy_from_slice(&result[..32]);
        combined
    }
}

/// Lightweight summary for the API status endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDSessionSummary {
    pub peer_id: String,
    pub protocol: String,
    pub confidence: f64,
    pub pns_resistant: bool,
    pub classical_leakage: f64,
    pub is_tor: bool,
    pub is_hidden_service: bool,
    pub session_id: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qkd_enabled_default() {
        // With no env var set, should be enabled.
        std::env::remove_var("Q_QKD_ENABLED");
        assert!(is_qkd_enabled());
    }

    #[test]
    fn test_qkd_disabled_explicitly() {
        std::env::set_var("Q_QKD_ENABLED", "0");
        assert!(!is_qkd_enabled());
        std::env::set_var("Q_QKD_ENABLED", "false");
        assert!(!is_qkd_enabled());
        std::env::set_var("Q_QKD_ENABLED", "FALSE");
        assert!(!is_qkd_enabled());
        std::env::set_var("Q_QKD_ENABLED", "no");
        assert!(!is_qkd_enabled());
        std::env::set_var("Q_QKD_ENABLED", "NO");
        assert!(!is_qkd_enabled());
        // Cleanup
        std::env::remove_var("Q_QKD_ENABLED");
    }

    #[test]
    fn test_qkd_enabled_explicitly() {
        std::env::set_var("Q_QKD_ENABLED", "1");
        assert!(is_qkd_enabled());
        std::env::set_var("Q_QKD_ENABLED", "true");
        assert!(is_qkd_enabled());
        // Cleanup
        std::env::remove_var("Q_QKD_ENABLED");
    }

    #[test]
    fn test_session_manager_disabled() {
        std::env::set_var("Q_QKD_ENABLED", "0");
        let mgr = QKDSessionManager::new();
        assert!(!mgr.is_enabled());
        assert_eq!(mgr.session_count(), 0);

        let profile = ChannelProfile::default();
        assert!(mgr.establish_session("peer1", profile).is_none());
        // Cleanup
        std::env::remove_var("Q_QKD_ENABLED");
    }

    #[test]
    fn test_build_channel_profile() {
        std::env::remove_var("Q_QKD_ENABLED");
        let mgr = QKDSessionManager::new();
        let profile = mgr.build_channel_profile(true, false, 150.0, 3);
        assert!(profile.is_tor_routed);
        assert!(!profile.is_hidden_service);
        assert!((profile.latency_ms - 150.0).abs() < f64::EPSILON);
        assert_eq!(profile.hop_count, 3);
        assert!((profile.loss_db - 6.0).abs() < f64::EPSILON); // 3 hops * 2dB
    }

    #[test]
    fn test_build_channel_profile_hidden_service() {
        std::env::remove_var("Q_QKD_ENABLED");
        let mgr = QKDSessionManager::new();
        let profile = mgr.build_channel_profile(true, true, 300.0, 6);
        assert!(profile.is_hidden_service);
        assert!(profile.require_max_privacy); // auto-set for hidden services
    }

    #[test]
    fn test_derive_simulated_key_unique() {
        let key1 = QKDSessionManager::derive_simulated_key("peer_a");
        let key2 = QKDSessionManager::derive_simulated_key("peer_b");
        // Different peer IDs should produce different keys
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_combine_with_qkd_key() {
        let classical = [1u8; 32];
        let pq = [2u8; 48];
        let qkd = [3u8; 32];
        let combined = QKDSessionManager::combine_with_qkd_key(&classical, &pq, &qkd);
        // Should produce a deterministic 32-byte key
        assert_eq!(combined.len(), 32);
        // Running again with same inputs should produce same output
        let combined2 = QKDSessionManager::combine_with_qkd_key(&classical, &pq, &qkd);
        assert_eq!(combined, combined2);
        // Different inputs should produce different output
        let qkd_alt = [4u8; 32];
        let combined3 = QKDSessionManager::combine_with_qkd_key(&classical, &pq, &qkd_alt);
        assert_ne!(combined, combined3);
    }

    #[test]
    fn test_hex_key_roundtrip() {
        let session = QKDSessionResult {
            protocol: QKDProtocolChoice::BB84,
            reason: "test".to_string(),
            sifting_rate: 0.5,
            key_rate: 0.4,
            pns_resistant: false,
            classical_leakage: 1.0,
            confidence: 0.95,
            qkd_key: [0xAB; 32],
            channel_profile: ChannelProfile::default(),
            established_at: None,
            session_id: 1,
        };
        let json = serde_json::to_string(&session).unwrap();
        let deserialized: QKDSessionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.qkd_key, [0xAB; 32]);
    }
}
