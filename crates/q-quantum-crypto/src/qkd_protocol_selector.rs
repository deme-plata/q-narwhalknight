//! QKD Protocol Selector
//!
//! Automatically selects the optimal QKD protocol (BB84, SARG04, or NPAB)
//! based on channel characteristics and privacy requirements.
//!
//! Selection criteria:
//! - **BB84**: Default for low-loss, high-throughput channels (direct LAN/WAN).
//!   Sifting rate: 50%. Best key rate.
//! - **SARG04**: Selected for high-loss channels or when PNS resistance is needed
//!   (e.g., Tor circuits with >6dB loss). Sifting rate: 25%.
//! - **NPAB**: Selected when maximum classical-channel privacy is required
//!   (e.g., Tor hidden services where metadata must be minimised).
//!   Sifting rate: ~12.5%. Zero classical leakage.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use crate::bb84_protocol::MeasurementBasis;
use crate::qkd::{QKDKey, SecurityLevel};
use crate::quantum_channels::QuantumChannel;
use crate::quantum_entropy::QuantumEntropySource;
use crate::sarg04_protocol::{SARG04Config, SARG04Protocol};
use crate::npab_protocol::{NPABConfig, NPABProtocol};
use crate::NodeId;

// ---------------------------------------------------------------------------
// Channel profile
// ---------------------------------------------------------------------------

/// Measured or estimated characteristics of the quantum channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelProfile {
    /// Estimated channel loss in dB.
    pub loss_db: f64,
    /// Whether the channel is routed through Tor.
    pub is_tor_routed: bool,
    /// Whether the channel is a hidden service (.onion).
    pub is_hidden_service: bool,
    /// Estimated round-trip latency in milliseconds.
    pub latency_ms: f64,
    /// Number of intermediate hops (0 for direct).
    pub hop_count: u32,
    /// Whether PNS attack resistance is explicitly required.
    pub require_pns_resistance: bool,
    /// Whether maximum classical privacy is explicitly required.
    pub require_max_privacy: bool,
}

impl Default for ChannelProfile {
    fn default() -> Self {
        Self {
            loss_db: 0.0,
            is_tor_routed: false,
            is_hidden_service: false,
            latency_ms: 10.0,
            hop_count: 0,
            require_pns_resistance: false,
            require_max_privacy: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Protocol selection
// ---------------------------------------------------------------------------

/// The selected QKD protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QKDProtocolChoice {
    /// Standard BB84 (highest throughput).
    BB84,
    /// SARG04 (PNS-resistant, for lossy/Tor channels).
    SARG04,
    /// NPAB (maximum privacy, no basis announcement).
    NPAB,
}

impl std::fmt::Display for QKDProtocolChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QKDProtocolChoice::BB84 => write!(f, "BB84"),
            QKDProtocolChoice::SARG04 => write!(f, "SARG04"),
            QKDProtocolChoice::NPAB => write!(f, "NPAB"),
        }
    }
}

/// Explanation of why a particular protocol was selected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionRationale {
    /// The chosen protocol.
    pub protocol: QKDProtocolChoice,
    /// Human-readable reason for the selection.
    pub reason: String,
    /// Expected sifting rate for this protocol.
    pub expected_sifting_rate: f64,
    /// Expected key rate (bits per channel use).
    pub expected_key_rate: f64,
    /// Classical channel leakage (bits).
    pub classical_leakage: f64,
    /// Whether PNS resistance is provided.
    pub pns_resistant: bool,
    /// Confidence score for this selection (0.0–1.0).
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// Protocol selector
// ---------------------------------------------------------------------------

/// Selects and executes the optimal QKD protocol based on channel conditions.
#[derive(Debug)]
pub struct QKDProtocolSelector {
    node_id: NodeId,
    entropy_source: Arc<QuantumEntropySource>,
    /// Loss threshold (dB) above which SARG04 is preferred over BB84.
    loss_threshold_db: f64,
    /// Latency threshold (ms) above which NPAB is considered for privacy.
    latency_threshold_ms: f64,
}

impl QKDProtocolSelector {
    /// Create a new protocol selector with default thresholds.
    pub fn new(node_id: NodeId, entropy_source: Arc<QuantumEntropySource>) -> Self {
        Self {
            node_id,
            entropy_source,
            loss_threshold_db: 6.0,    // Above 6dB, switch to SARG04
            latency_threshold_ms: 200.0, // Above 200ms, consider NPAB for Tor
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(
        node_id: NodeId,
        entropy_source: Arc<QuantumEntropySource>,
        loss_threshold_db: f64,
        latency_threshold_ms: f64,
    ) -> Self {
        Self {
            node_id,
            entropy_source,
            loss_threshold_db,
            latency_threshold_ms,
        }
    }

    /// Select the optimal QKD protocol for the given channel profile.
    ///
    /// Decision tree:
    /// 1. If `require_max_privacy` → NPAB
    /// 2. If `is_hidden_service` → NPAB
    /// 3. If `require_pns_resistance` → SARG04
    /// 4. If `loss_db > threshold` AND `is_tor_routed` → SARG04
    /// 5. If `is_tor_routed` AND high latency → SARG04 (Tor default)
    /// 6. Otherwise → BB84 (maximum throughput)
    pub fn select(&self, profile: &ChannelProfile) -> SelectionRationale {
        // Rule 1: Explicit max privacy request → NPAB
        if profile.require_max_privacy {
            return SelectionRationale {
                protocol: QKDProtocolChoice::NPAB,
                reason: "Maximum classical-channel privacy explicitly requested".to_string(),
                expected_sifting_rate: 0.125,
                expected_key_rate: 0.10,
                classical_leakage: 0.0,
                pns_resistant: true,
                confidence: 0.95,
            };
        }

        // Rule 2: Hidden service → NPAB (metadata minimisation critical)
        if profile.is_hidden_service {
            return SelectionRationale {
                protocol: QKDProtocolChoice::NPAB,
                reason: "Hidden service detected — minimising classical metadata".to_string(),
                expected_sifting_rate: 0.125,
                expected_key_rate: 0.10,
                classical_leakage: 0.0,
                pns_resistant: true,
                confidence: 0.90,
            };
        }

        // Rule 3: Explicit PNS resistance → SARG04
        if profile.require_pns_resistance {
            return SelectionRationale {
                protocol: QKDProtocolChoice::SARG04,
                reason: "PNS attack resistance explicitly requested".to_string(),
                expected_sifting_rate: 0.25,
                expected_key_rate: 0.20,
                classical_leakage: 1.0, // 1 bit per photon (announcement)
                pns_resistant: true,
                confidence: 0.95,
            };
        }

        // Rule 4: High loss + Tor → SARG04
        if profile.loss_db > self.loss_threshold_db && profile.is_tor_routed {
            return SelectionRationale {
                protocol: QKDProtocolChoice::SARG04,
                reason: format!(
                    "High channel loss ({:.1}dB) on Tor circuit — SARG04 for PNS resistance",
                    profile.loss_db
                ),
                expected_sifting_rate: 0.25,
                expected_key_rate: 0.18,
                classical_leakage: 1.0,
                pns_resistant: true,
                confidence: 0.85,
            };
        }

        // Rule 5: Tor + high latency → SARG04 (default for Tor)
        if profile.is_tor_routed && profile.latency_ms > self.latency_threshold_ms {
            return SelectionRationale {
                protocol: QKDProtocolChoice::SARG04,
                reason: format!(
                    "Tor-routed channel with {:.0}ms latency — SARG04 as default Tor protocol",
                    profile.latency_ms
                ),
                expected_sifting_rate: 0.25,
                expected_key_rate: 0.20,
                classical_leakage: 1.0,
                pns_resistant: true,
                confidence: 0.80,
            };
        }

        // Rule 6: Default → BB84 (maximum throughput)
        SelectionRationale {
            protocol: QKDProtocolChoice::BB84,
            reason: "Low-loss direct channel — BB84 for maximum key rate".to_string(),
            expected_sifting_rate: 0.50,
            expected_key_rate: 0.40,
            classical_leakage: 1.0,
            pns_resistant: false,
            confidence: 0.95,
        }
    }

    /// Select protocol and execute key exchange in one call.
    ///
    /// This is the high-level API: provide a channel profile and get back
    /// a QKD key generated with the optimal protocol.
    pub async fn execute_key_exchange(
        &self,
        channel: &QuantumChannel,
        profile: &ChannelProfile,
        key_length: usize,
        role: KeyExchangeRole,
    ) -> Result<(QKDKey, SelectionRationale)> {
        let selection = self.select(profile);

        let key = match (selection.protocol, role) {
            (QKDProtocolChoice::SARG04, KeyExchangeRole::Alice) => {
                let proto = SARG04Protocol::new(self.node_id, self.entropy_source.clone());
                proto.execute_as_alice(channel, key_length).await?
            }
            (QKDProtocolChoice::SARG04, KeyExchangeRole::Bob) => {
                let proto = SARG04Protocol::new(self.node_id, self.entropy_source.clone());
                proto.execute_as_bob(channel).await?
            }
            (QKDProtocolChoice::NPAB, KeyExchangeRole::Alice) => {
                let proto = NPABProtocol::new(self.node_id, self.entropy_source.clone());
                proto.execute_as_alice(channel, key_length).await?
            }
            (QKDProtocolChoice::NPAB, KeyExchangeRole::Bob) => {
                let proto = NPABProtocol::new(self.node_id, self.entropy_source.clone());
                proto.execute_as_bob(channel).await?
            }
            (QKDProtocolChoice::BB84, _) => {
                // Fall back to existing BB84 implementation via QKDEngine.
                let engine = crate::qkd::QKDEngine::new(
                    self.node_id,
                    self.entropy_source.clone(),
                )
                .await?;
                let raw_key = engine.execute_bb84_protocol(self.node_id, channel).await?;
                QKDKey {
                    key_data: raw_key,
                    generated_at: std::time::SystemTime::now(),
                    security_level: SecurityLevel::InformationTheoretic,
                    error_rate: 0.0,
                    usage_count: 0,
                    max_usage: 1,
                }
            }
        };

        Ok((key, selection))
    }

    /// Compare all three protocols for a given channel profile.
    ///
    /// Useful for logging and diagnostics. Returns the rationale for each
    /// protocol without executing any key exchange.
    pub fn compare_protocols(&self, profile: &ChannelProfile) -> Vec<SelectionRationale> {
        let mut results = Vec::with_capacity(3);

        // BB84
        results.push(SelectionRationale {
            protocol: QKDProtocolChoice::BB84,
            reason: "Direct channel, maximum throughput".to_string(),
            expected_sifting_rate: 0.50,
            expected_key_rate: if profile.loss_db < 3.0 { 0.40 } else { 0.30 },
            classical_leakage: 1.0,
            pns_resistant: false,
            confidence: if profile.is_tor_routed { 0.3 } else { 0.95 },
        });

        // SARG04
        results.push(SelectionRationale {
            protocol: QKDProtocolChoice::SARG04,
            reason: "PNS-resistant, Tor-optimised".to_string(),
            expected_sifting_rate: 0.25,
            expected_key_rate: if profile.loss_db < 10.0 { 0.20 } else { 0.12 },
            classical_leakage: 1.0,
            pns_resistant: true,
            confidence: if profile.is_tor_routed { 0.85 } else { 0.60 },
        });

        // NPAB
        results.push(SelectionRationale {
            protocol: QKDProtocolChoice::NPAB,
            reason: "Maximum privacy, zero classical leakage".to_string(),
            expected_sifting_rate: 0.125,
            expected_key_rate: 0.10,
            classical_leakage: 0.0,
            pns_resistant: true,
            confidence: if profile.is_hidden_service { 0.90 } else { 0.40 },
        });

        results
    }
}

/// Role in the key exchange (determines which side of the protocol to run).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyExchangeRole {
    /// Alice: prepares and sends quantum states.
    Alice,
    /// Bob: receives and measures quantum states.
    Bob,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_selector() -> QKDProtocolSelector {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let entropy = rt.block_on(async { Arc::new(QuantumEntropySource::new().await.unwrap()) });
        QKDProtocolSelector::new([0u8; 32], entropy)
    }

    #[test]
    fn test_select_bb84_for_direct_channel() {
        let selector = make_selector();
        let profile = ChannelProfile::default();
        let result = selector.select(&profile);
        assert_eq!(result.protocol, QKDProtocolChoice::BB84);
        assert!((result.expected_sifting_rate - 0.50).abs() < 0.01);
    }

    #[test]
    fn test_select_sarg04_for_tor_high_loss() {
        let selector = make_selector();
        let profile = ChannelProfile {
            loss_db: 10.0,
            is_tor_routed: true,
            ..Default::default()
        };
        let result = selector.select(&profile);
        assert_eq!(result.protocol, QKDProtocolChoice::SARG04);
        assert!(result.pns_resistant);
    }

    #[test]
    fn test_select_sarg04_for_explicit_pns() {
        let selector = make_selector();
        let profile = ChannelProfile {
            require_pns_resistance: true,
            ..Default::default()
        };
        let result = selector.select(&profile);
        assert_eq!(result.protocol, QKDProtocolChoice::SARG04);
    }

    #[test]
    fn test_select_sarg04_for_tor_high_latency() {
        let selector = make_selector();
        let profile = ChannelProfile {
            is_tor_routed: true,
            latency_ms: 300.0,
            ..Default::default()
        };
        let result = selector.select(&profile);
        assert_eq!(result.protocol, QKDProtocolChoice::SARG04);
    }

    #[test]
    fn test_select_npab_for_hidden_service() {
        let selector = make_selector();
        let profile = ChannelProfile {
            is_hidden_service: true,
            is_tor_routed: true,
            ..Default::default()
        };
        let result = selector.select(&profile);
        assert_eq!(result.protocol, QKDProtocolChoice::NPAB);
        assert_eq!(result.classical_leakage, 0.0);
    }

    #[test]
    fn test_select_npab_for_explicit_max_privacy() {
        let selector = make_selector();
        let profile = ChannelProfile {
            require_max_privacy: true,
            ..Default::default()
        };
        let result = selector.select(&profile);
        assert_eq!(result.protocol, QKDProtocolChoice::NPAB);
    }

    #[test]
    fn test_max_privacy_overrides_pns() {
        let selector = make_selector();
        // Both flags set — max_privacy should win (checked first).
        let profile = ChannelProfile {
            require_max_privacy: true,
            require_pns_resistance: true,
            ..Default::default()
        };
        let result = selector.select(&profile);
        assert_eq!(result.protocol, QKDProtocolChoice::NPAB);
    }

    #[test]
    fn test_compare_protocols_returns_three() {
        let selector = make_selector();
        let profile = ChannelProfile::default();
        let results = selector.compare_protocols(&profile);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].protocol, QKDProtocolChoice::BB84);
        assert_eq!(results[1].protocol, QKDProtocolChoice::SARG04);
        assert_eq!(results[2].protocol, QKDProtocolChoice::NPAB);
    }

    #[test]
    fn test_compare_protocols_tor_boosts_sarg04() {
        let selector = make_selector();
        let profile = ChannelProfile {
            is_tor_routed: true,
            ..Default::default()
        };
        let results = selector.compare_protocols(&profile);
        // SARG04 confidence should be higher than BB84 on Tor.
        assert!(
            results[1].confidence > results[0].confidence,
            "SARG04 ({}) should have higher confidence than BB84 ({}) on Tor",
            results[1].confidence,
            results[0].confidence
        );
    }

    #[test]
    fn test_compare_protocols_hidden_service_boosts_npab() {
        let selector = make_selector();
        let profile = ChannelProfile {
            is_hidden_service: true,
            is_tor_routed: true,
            ..Default::default()
        };
        let results = selector.compare_protocols(&profile);
        // NPAB confidence should be highest for hidden services.
        assert!(
            results[2].confidence > results[0].confidence,
            "NPAB ({}) should have higher confidence than BB84 ({}) for hidden service",
            results[2].confidence,
            results[0].confidence
        );
    }

    #[test]
    fn test_sifting_rates_ordering() {
        let selector = make_selector();
        let results = selector.compare_protocols(&ChannelProfile::default());
        // BB84 > SARG04 > NPAB sifting rates.
        assert!(results[0].expected_sifting_rate > results[1].expected_sifting_rate);
        assert!(results[1].expected_sifting_rate > results[2].expected_sifting_rate);
    }

    #[test]
    fn test_protocol_choice_display() {
        assert_eq!(format!("{}", QKDProtocolChoice::BB84), "BB84");
        assert_eq!(format!("{}", QKDProtocolChoice::SARG04), "SARG04");
        assert_eq!(format!("{}", QKDProtocolChoice::NPAB), "NPAB");
    }

    #[test]
    fn test_custom_thresholds() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let entropy = rt.block_on(async { Arc::new(QuantumEntropySource::new().await.unwrap()) });

        // Very high loss threshold → BB84 even at 8dB loss on Tor.
        let selector = QKDProtocolSelector::with_thresholds(
            [0u8; 32],
            entropy,
            10.0,  // loss threshold
            500.0, // latency threshold
        );
        let profile = ChannelProfile {
            loss_db: 8.0,
            is_tor_routed: true,
            latency_ms: 250.0,
            ..Default::default()
        };
        let result = selector.select(&profile);
        // 8dB < 10dB threshold and 250ms < 500ms threshold → BB84
        assert_eq!(result.protocol, QKDProtocolChoice::BB84);
    }
}
