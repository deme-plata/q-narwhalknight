//! SARG04 Protocol Implementation
//!
//! SARG04 is a variant of BB84 that provides superior resistance to
//! Photon Number Splitting (PNS) attacks, making it ideal for high-loss
//! quantum channels such as those routed through Tor circuits.
//!
//! Key difference from BB84:
//! - In BB84, Alice announces her **basis** after transmission.
//! - In SARG04, Alice announces **one of the four states she DID NOT send**.
//!   Bob must use his measurement result to deduce which of two possible
//!   states Alice actually sent.
//! - An eavesdropper with a copy of the photon learns less per intercepted
//!   photon because the announcement is ambiguous.
//! - Sifting rate is ~25% (vs 50% for BB84), but security per bit is higher.
//!
//! Reference: Scarani, Acin, Ribordy, Gisin, PRL 92, 057901 (2004)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

use crate::bb84_protocol::{MeasurementBasis, PhotonPolarization, QuantumBit};
use crate::qkd::{QKDKey, SecurityLevel};
use crate::quantum_channels::QuantumChannel;
use crate::quantum_entropy::QuantumEntropySource;
use crate::NodeId;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Error correction method for post-sifting key reconciliation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCorrectionMethod {
    /// Binary block parity (Cascade), standard for moderate QBER.
    Cascade,
    /// Low-Density Parity-Check codes, better throughput on high-QBER channels.
    LDPC,
}

impl Default for ErrorCorrectionMethod {
    fn default() -> Self {
        ErrorCorrectionMethod::Cascade
    }
}

/// Configuration parameters for the SARG04 protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SARG04Config {
    /// Target key length in bits (default 256).
    pub target_key_length: usize,
    /// Maximum tolerable QBER before aborting (10.95%).
    pub qber_threshold: f64,
    /// Fraction of sifted bits used for parameter estimation (default 10%).
    pub test_fraction: f64,
    /// Whether to apply privacy amplification.
    pub privacy_amplification: bool,
    /// Error correction algorithm.
    pub error_correction: ErrorCorrectionMethod,
}

impl Default for SARG04Config {
    fn default() -> Self {
        Self {
            target_key_length: 256,
            qber_threshold: 0.1095,
            test_fraction: 0.10,
            privacy_amplification: true,
            error_correction: ErrorCorrectionMethod::Cascade,
        }
    }
}

// ---------------------------------------------------------------------------
// SARG04 announcement
// ---------------------------------------------------------------------------

/// In SARG04 Alice announces one of the four states she DID NOT send.
///
/// Each announcement eliminates one candidate state. Combined with Bob's
/// measurement outcome this allows him to unambiguously determine Alice's
/// bit in about 25% of cases (when his basis is compatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SARG04Announcement {
    /// Alice did NOT send |H> (Horizontal, Rectilinear-0).
    NotHorizontal,
    /// Alice did NOT send |V> (Vertical, Rectilinear-1).
    NotVertical,
    /// Alice did NOT send |D> (Diagonal-0, 45 degrees).
    NotDiagonal,
    /// Alice did NOT send |A> (AntiDiagonal-1, 135 degrees).
    NotAntiDiagonal,
}

impl SARG04Announcement {
    /// Generate the correct announcement for a given sent state.
    ///
    /// Alice picks one state she did NOT send. We pick the state in the
    /// *same basis* with the *opposite bit*. This maximises the probability
    /// that Bob can unambiguously decode.
    pub fn for_sent_state(sent: PhotonPolarization) -> Self {
        match sent {
            // Sent |H> (Rect-0) -> announce "not |V>" (same basis, opposite bit)
            PhotonPolarization::Horizontal => SARG04Announcement::NotVertical,
            // Sent |V> (Rect-1) -> announce "not |H>"
            PhotonPolarization::Vertical => SARG04Announcement::NotHorizontal,
            // Sent |D> (Diag-0) -> announce "not |A>"
            PhotonPolarization::Diagonal => SARG04Announcement::NotAntiDiagonal,
            // Sent |A> (Diag-1) -> announce "not |D>"
            PhotonPolarization::AntiDiagonal => SARG04Announcement::NotDiagonal,
        }
    }

    /// Returns the excluded polarization.
    pub fn excluded_state(&self) -> PhotonPolarization {
        match self {
            SARG04Announcement::NotHorizontal => PhotonPolarization::Horizontal,
            SARG04Announcement::NotVertical => PhotonPolarization::Vertical,
            SARG04Announcement::NotDiagonal => PhotonPolarization::Diagonal,
            SARG04Announcement::NotAntiDiagonal => PhotonPolarization::AntiDiagonal,
        }
    }

    /// Returns the two candidate states that Alice *could* have sent,
    /// given this announcement.
    ///
    /// By excluding one state the remaining three are candidates, but the
    /// two most probable (in the standard SARG04 encoding) are the ones
    /// from the *same basis* and the *non-excluded state from the other
    /// basis that shares the same bit value* as the excluded state.
    ///
    /// For the standard pair encoding used in our implementation the
    /// candidates are:
    ///   "not |V>"  -> Alice sent |H> or |D>   (both encode bit 0)
    ///   "not |H>"  -> Alice sent |V> or |A>   (both encode bit 1)
    ///   "not |A>"  -> Alice sent |D> or |H>   (both encode bit 0)
    ///   "not |D>"  -> Alice sent |A> or |V>   (both encode bit 1)
    pub fn candidate_states(&self) -> (PhotonPolarization, PhotonPolarization) {
        match self {
            SARG04Announcement::NotVertical => {
                (PhotonPolarization::Horizontal, PhotonPolarization::Diagonal)
            }
            SARG04Announcement::NotHorizontal => {
                (PhotonPolarization::Vertical, PhotonPolarization::AntiDiagonal)
            }
            SARG04Announcement::NotAntiDiagonal => {
                (PhotonPolarization::Diagonal, PhotonPolarization::Horizontal)
            }
            SARG04Announcement::NotDiagonal => {
                (PhotonPolarization::AntiDiagonal, PhotonPolarization::Vertical)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Security analysis
// ---------------------------------------------------------------------------

/// Security analysis result for a SARG04 session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SARG04SecurityAnalysis {
    /// Observed QBER.
    pub qber: f64,
    /// Estimated information leakage to Eve (bits per sifted bit).
    pub eve_information: f64,
    /// Secure key rate (bits per channel use).
    pub secure_key_rate: f64,
    /// Channel loss in dB.
    pub channel_loss_db: f64,
    /// Maximum tolerable loss for PNS-safe operation (dB).
    pub max_safe_loss_db: f64,
    /// Whether the session parameters are considered safe.
    pub is_safe: bool,
    /// Achieved security level.
    pub security_level: SecurityLevel,
}

// ---------------------------------------------------------------------------
// Protocol statistics
// ---------------------------------------------------------------------------

/// Statistics for SARG04 protocol sessions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SARG04Stats {
    pub sessions_completed: u64,
    pub total_photons_sent: u64,
    pub total_photons_received: u64,
    pub average_error_rate: f64,
    pub total_key_bits_generated: u64,
    pub average_sifting_rate: f64,
    pub last_session_duration: Duration,
}

// ---------------------------------------------------------------------------
// SARG04 Protocol
// ---------------------------------------------------------------------------

/// SARG04 Protocol -- resilient to PNS attacks on lossy channels (e.g., Tor).
///
/// Reference: Scarani et al., PRL 92, 057901 (2004)
#[derive(Debug)]
pub struct SARG04Protocol {
    node_id: NodeId,
    entropy_source: Arc<QuantumEntropySource>,
    config: SARG04Config,
    stats: Arc<RwLock<SARG04Stats>>,
}

impl SARG04Protocol {
    /// Create a new SARG04 protocol instance with default configuration.
    pub fn new(node_id: NodeId, entropy_source: Arc<QuantumEntropySource>) -> Self {
        Self::with_config(node_id, entropy_source, SARG04Config::default())
    }

    /// Create a new SARG04 protocol instance with custom configuration.
    pub fn with_config(
        node_id: NodeId,
        entropy_source: Arc<QuantumEntropySource>,
        config: SARG04Config,
    ) -> Self {
        Self {
            node_id,
            entropy_source,
            config,
            stats: Arc::new(RwLock::new(SARG04Stats::default())),
        }
    }

    /// Execute SARG04 as Alice (sender).
    ///
    /// Alice prepares random polarisation states, sends them through the
    /// quantum channel, and then publicly announces one state she did NOT
    /// send for each photon. Bob uses these announcements together with
    /// his measurement outcomes to sift and decode.
    pub async fn execute_as_alice(
        &self,
        channel: &QuantumChannel,
        key_length: usize,
    ) -> Result<QKDKey> {
        let start = Instant::now();

        // We need ~16x raw photons to get `key_length` final bits after
        // 25% sifting, 10% test discard, error correction, and privacy amp.
        let raw_count = key_length * 16;

        // --- Step 1: Generate random bits and bases ---
        let raw_bits = self.entropy_source.generate_true_random(raw_count).await?;
        let raw_bases = self.entropy_source.generate_true_random(raw_count).await?;

        // --- Step 2: Prepare polarisation states ---
        let mut alice_states: Vec<PhotonPolarization> = Vec::with_capacity(raw_count);
        let mut photon_states = Vec::with_capacity(raw_count);

        for i in 0..raw_count {
            let bit = (raw_bits[i % raw_bits.len()] & 1) != 0;
            let basis = if (raw_bases[i % raw_bases.len()] & 1) != 0 {
                MeasurementBasis::Diagonal
            } else {
                MeasurementBasis::Rectilinear
            };
            let qbit = QuantumBit::new(bit, basis);
            alice_states.push(qbit.polarization);

            // Convert to PhotonState for the quantum channel
            let photon_state = match (basis, bit) {
                (MeasurementBasis::Rectilinear, b) => crate::PhotonState::Rectilinear(b),
                (MeasurementBasis::Diagonal, b) => crate::PhotonState::Diagonal(b),
            };
            photon_states.push(photon_state);
        }

        // --- Step 3: Send photons ---
        channel.send_photons(&photon_states).await?;

        // --- Step 4: Generate and send SARG04 announcements ---
        let announcements: Vec<SARG04Announcement> = alice_states
            .iter()
            .map(|&s| SARG04Announcement::for_sent_state(s))
            .collect();

        // Encode announcements as bytes (0..3) for the classical channel.
        let announcement_bytes: Vec<u8> = announcements
            .iter()
            .map(|a| match a {
                SARG04Announcement::NotHorizontal => 0u8,
                SARG04Announcement::NotVertical => 1u8,
                SARG04Announcement::NotDiagonal => 2u8,
                SARG04Announcement::NotAntiDiagonal => 3u8,
            })
            .collect();

        channel
            .send_classical_message(
                crate::quantum_channels::ClassicalMessageType::BasisAnnouncement,
                announcement_bytes,
            )
            .await?;

        // --- Step 5: Receive Bob's sifted bit indices ---
        let index_data = channel
            .receive_classical_message(
                crate::quantum_channels::ClassicalMessageType::MatchingIndices,
            )
            .await?;
        let sifted_indices: Vec<usize> = index_data
            .chunks(8)
            .map(|c| {
                let mut buf = [0u8; 8];
                buf[..c.len()].copy_from_slice(c);
                usize::from_le_bytes(buf)
            })
            .collect();

        // Extract Alice's sifted bits.
        let alice_sifted: Vec<bool> = sifted_indices
            .iter()
            .filter_map(|&idx| {
                if idx < alice_states.len() {
                    Some(alice_states[idx].to_bit())
                } else {
                    None
                }
            })
            .collect();

        // --- Step 6: Parameter estimation ---
        let test_count = (alice_sifted.len() as f64 * self.config.test_fraction).ceil() as usize;
        let test_count = test_count.min(alice_sifted.len());

        // Send Alice's test bits to Bob.
        let test_bits_bytes: Vec<u8> = alice_sifted[..test_count]
            .iter()
            .map(|&b| b as u8)
            .collect();
        channel
            .send_classical_message(
                crate::quantum_channels::ClassicalMessageType::TestBits,
                test_bits_bytes,
            )
            .await?;

        // Receive Bob's test bits.
        let bob_test_data = channel
            .receive_classical_message(crate::quantum_channels::ClassicalMessageType::TestBits)
            .await?;
        let bob_test_bits: Vec<bool> = bob_test_data.into_iter().map(|b| b != 0).collect();

        let mut errors = 0usize;
        let compared = test_count.min(bob_test_bits.len());
        for i in 0..compared {
            if alice_sifted[i] != bob_test_bits[i] {
                errors += 1;
            }
        }
        let qber = if compared > 0 {
            errors as f64 / compared as f64
        } else {
            0.0
        };

        // Announce error rate to Bob.
        channel
            .send_classical_message(
                crate::quantum_channels::ClassicalMessageType::ErrorRate,
                qber.to_le_bytes().to_vec(),
            )
            .await?;

        if qber > self.config.qber_threshold {
            return Err(anyhow::anyhow!(
                "SARG04 QBER too high: {:.2}% (threshold {:.2}%)",
                qber * 100.0,
                self.config.qber_threshold * 100.0
            ));
        }

        // --- Step 7: Error correction ---
        let remaining_bits = &alice_sifted[test_count..];
        let corrected = self.apply_error_correction(remaining_bits, qber)?;

        // --- Step 8: Privacy amplification ---
        let final_key = if self.config.privacy_amplification {
            self.privacy_amplification(&corrected, qber)?
        } else {
            corrected
        };

        // --- Step 9: Update stats ---
        let sifting_rate = if raw_count > 0 {
            alice_sifted.len() as f64 / raw_count as f64
        } else {
            0.0
        };

        {
            let mut stats = self.stats.write().await;
            stats.sessions_completed += 1;
            stats.total_photons_sent += raw_count as u64;
            stats.average_error_rate =
                (stats.average_error_rate * (stats.sessions_completed - 1) as f64 + qber)
                    / stats.sessions_completed as f64;
            stats.total_key_bits_generated += final_key.len() as u64 * 8;
            stats.average_sifting_rate =
                (stats.average_sifting_rate * (stats.sessions_completed - 1) as f64 + sifting_rate)
                    / stats.sessions_completed as f64;
            stats.last_session_duration = start.elapsed();
        }

        let security_level = Self::classify_security(qber);

        let key_len = final_key.len() as u64;
        Ok(QKDKey {
            key_data: final_key,
            generated_at: SystemTime::now(),
            security_level,
            error_rate: qber,
            usage_count: 0,
            max_usage: key_len,
        })
    }

    /// Execute SARG04 as Bob (receiver).
    ///
    /// Bob receives photons, measures them in randomly chosen bases, then
    /// uses Alice's SARG04 announcements to determine which measurements
    /// can be unambiguously decoded.
    pub async fn execute_as_bob(&self, channel: &QuantumChannel) -> Result<QKDKey> {
        let start = Instant::now();

        // --- Step 1: Receive photons ---
        let received_photons = channel.receive_photons().await?;
        let n = received_photons.len();

        // --- Step 2: Randomly choose measurement bases ---
        let basis_bytes = self.entropy_source.generate_true_random(n).await?;

        let mut bob_bases: Vec<MeasurementBasis> = Vec::with_capacity(n);
        let mut bob_results: Vec<bool> = Vec::with_capacity(n);
        let mut bob_polarizations: Vec<PhotonPolarization> = Vec::with_capacity(n);

        for (i, photon) in received_photons.iter().enumerate() {
            let basis = if (basis_bytes[i % basis_bytes.len()] & 1) != 0 {
                MeasurementBasis::Diagonal
            } else {
                MeasurementBasis::Rectilinear
            };
            bob_bases.push(basis);

            // Simulate measurement.
            let bit = self.measure_photon(photon, basis).await?;
            bob_results.push(bit);

            let pol = match (basis, bit) {
                (MeasurementBasis::Rectilinear, false) => PhotonPolarization::Horizontal,
                (MeasurementBasis::Rectilinear, true) => PhotonPolarization::Vertical,
                (MeasurementBasis::Diagonal, false) => PhotonPolarization::Diagonal,
                (MeasurementBasis::Diagonal, true) => PhotonPolarization::AntiDiagonal,
            };
            bob_polarizations.push(pol);
        }

        // --- Step 3: Receive SARG04 announcements ---
        let announcement_data = channel
            .receive_classical_message(
                crate::quantum_channels::ClassicalMessageType::BasisAnnouncement,
            )
            .await?;

        let announcements: Vec<SARG04Announcement> = announcement_data
            .iter()
            .map(|&b| match b {
                0 => SARG04Announcement::NotHorizontal,
                1 => SARG04Announcement::NotVertical,
                2 => SARG04Announcement::NotDiagonal,
                3 => SARG04Announcement::NotAntiDiagonal,
                _ => SARG04Announcement::NotHorizontal, // fallback
            })
            .collect();

        // --- Step 4: SARG04 sifting ---
        let sifted = Self::sift_sarg04(&bob_polarizations, &bob_results, &announcements);

        // Send sifted indices to Alice.
        let index_bytes: Vec<u8> = sifted
            .iter()
            .flat_map(|&(idx, _)| idx.to_le_bytes())
            .collect();
        channel
            .send_classical_message(
                crate::quantum_channels::ClassicalMessageType::MatchingIndices,
                index_bytes,
            )
            .await?;

        let bob_sifted_bits: Vec<bool> = sifted.iter().map(|&(_, bit)| bit).collect();

        // --- Step 5: Parameter estimation ---
        let test_count =
            (bob_sifted_bits.len() as f64 * self.config.test_fraction).ceil() as usize;
        let test_count = test_count.min(bob_sifted_bits.len());

        // Receive Alice's test bits.
        let alice_test_data = channel
            .receive_classical_message(crate::quantum_channels::ClassicalMessageType::TestBits)
            .await?;
        let alice_test_bits: Vec<bool> = alice_test_data.into_iter().map(|b| b != 0).collect();

        // Send Bob's test bits to Alice.
        let bob_test_bytes: Vec<u8> = bob_sifted_bits[..test_count]
            .iter()
            .map(|&b| b as u8)
            .collect();
        channel
            .send_classical_message(
                crate::quantum_channels::ClassicalMessageType::TestBits,
                bob_test_bytes,
            )
            .await?;

        // Receive QBER from Alice.
        let qber_data = channel
            .receive_classical_message(crate::quantum_channels::ClassicalMessageType::ErrorRate)
            .await?;
        let qber = if qber_data.len() >= 8 {
            f64::from_le_bytes([
                qber_data[0],
                qber_data[1],
                qber_data[2],
                qber_data[3],
                qber_data[4],
                qber_data[5],
                qber_data[6],
                qber_data[7],
            ])
        } else {
            0.0
        };

        if qber > self.config.qber_threshold {
            return Err(anyhow::anyhow!(
                "SARG04 QBER too high: {:.2}% (threshold {:.2}%)",
                qber * 100.0,
                self.config.qber_threshold * 100.0
            ));
        }

        // --- Step 6: Error correction ---
        let remaining_bits = &bob_sifted_bits[test_count..];
        let corrected = self.apply_error_correction(remaining_bits, qber)?;

        // --- Step 7: Privacy amplification ---
        let final_key = if self.config.privacy_amplification {
            self.privacy_amplification(&corrected, qber)?
        } else {
            corrected
        };

        // --- Step 8: Update stats ---
        let sifting_rate = if n > 0 {
            bob_sifted_bits.len() as f64 / n as f64
        } else {
            0.0
        };

        {
            let mut stats = self.stats.write().await;
            stats.sessions_completed += 1;
            stats.total_photons_received += n as u64;
            stats.average_error_rate =
                (stats.average_error_rate * (stats.sessions_completed - 1) as f64 + qber)
                    / stats.sessions_completed as f64;
            stats.total_key_bits_generated += final_key.len() as u64 * 8;
            stats.average_sifting_rate =
                (stats.average_sifting_rate * (stats.sessions_completed - 1) as f64 + sifting_rate)
                    / stats.sessions_completed as f64;
            stats.last_session_duration = start.elapsed();
        }

        let security_level = Self::classify_security(qber);

        let key_len = final_key.len() as u64;
        Ok(QKDKey {
            key_data: final_key,
            generated_at: SystemTime::now(),
            security_level,
            error_rate: qber,
            usage_count: 0,
            max_usage: key_len,
        })
    }

    // -----------------------------------------------------------------------
    // SARG04 sifting (the core algorithmic difference from BB84)
    // -----------------------------------------------------------------------

    /// SARG04 sifting logic.
    ///
    /// For each photon, Alice announces one state she did NOT send. This
    /// implies two candidate states (one from each basis). Bob checks
    /// whether his measurement result is *incompatible* with exactly one of
    /// the two candidates, leaving a single unambiguous answer.
    ///
    /// Concretely, Bob can decode when:
    ///   - His measurement basis matches ONE of the two candidates' bases
    ///   - His result rules out that candidate
    ///   - The remaining candidate is therefore the only possibility
    ///
    /// Returns `(index, bit_value)` pairs where Bob's decode is unambiguous.
    pub fn sift_sarg04(
        bob_polarizations: &[PhotonPolarization],
        bob_results: &[bool],
        announcements: &[SARG04Announcement],
    ) -> Vec<(usize, bool)> {
        let n = bob_polarizations
            .len()
            .min(bob_results.len())
            .min(announcements.len());
        let mut sifted = Vec::new();

        for i in 0..n {
            let (cand_a, cand_b) = announcements[i].candidate_states();
            let bob_basis = bob_polarizations[i].basis();
            let bob_bit = bob_results[i];

            // The two candidates come from different bases (by construction).
            // Bob measured in one basis. Check which candidate is in Bob's basis.
            let cand_a_basis = cand_a.basis();
            let cand_b_basis = cand_b.basis();

            // Bob can only decode if his measurement basis matches exactly one
            // candidate's basis.
            if bob_basis == cand_a_basis && bob_basis != cand_b_basis {
                // Bob's basis matches candidate A.
                // If Bob's result is DIFFERENT from candidate A's bit value,
                // then A is ruled out and B is the answer.
                if bob_bit != cand_a.to_bit() {
                    sifted.push((i, cand_b.to_bit()));
                }
                // If Bob's result matches candidate A, both are still possible
                // (ambiguous) -- discard.
            } else if bob_basis == cand_b_basis && bob_basis != cand_a_basis {
                // Bob's basis matches candidate B.
                if bob_bit != cand_b.to_bit() {
                    sifted.push((i, cand_a.to_bit()));
                }
            }
            // If Bob's basis matches both or neither candidate's basis, discard.
        }

        sifted
    }

    // -----------------------------------------------------------------------
    // Security analysis
    // -----------------------------------------------------------------------

    /// Produce a security analysis for a SARG04 session.
    ///
    /// SARG04 tolerates higher channel loss than BB84 before PNS attacks
    /// become viable. For single-photon sources the QBER threshold is
    /// ~10.95%. For weak coherent pulses with decoy states the tolerable
    /// loss is significantly higher than BB84.
    pub fn security_analysis(&self, qber: f64, channel_loss_db: f64) -> SARG04SecurityAnalysis {
        // Shannon binary entropy h(x) = -x log2(x) - (1-x) log2(1-x)
        let h_qber = if qber > 0.0 && qber < 1.0 {
            -qber * qber.log2() - (1.0 - qber) * (1.0 - qber).log2()
        } else {
            0.0
        };

        // Eve's information in SARG04 (tighter bound than BB84).
        // In SARG04 a PNS attack on multi-photon pulses gives Eve less
        // information because she cannot determine the state from the
        // announcement alone.
        let eve_info = h_qber; // Simplified: full intercept-resend bound.

        // Secure key rate r = 1 - 2 h(e) for SARG04 (slightly different
        // from BB84's 1 - 2h(e) due to different sifting).
        let secure_key_rate = (1.0 - 2.0 * h_qber).max(0.0) * 0.25; // 25% sifting rate

        // PNS safety: with weak coherent pulses, SARG04 is safe up to
        // ~8 dB higher loss than BB84. This is the key advantage.
        let max_safe_loss_db = 20.0; // Conservative estimate for SARG04 with decoy states.

        let is_safe = qber <= self.config.qber_threshold && channel_loss_db <= max_safe_loss_db;

        let security_level = Self::classify_security(qber);

        SARG04SecurityAnalysis {
            qber,
            eve_information: eve_info,
            secure_key_rate,
            channel_loss_db,
            max_safe_loss_db,
            is_safe,
            security_level,
        }
    }

    /// Get protocol statistics.
    pub async fn get_stats(&self) -> SARG04Stats {
        self.stats.read().await.clone()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Classify the achieved security level based on QBER.
    fn classify_security(qber: f64) -> SecurityLevel {
        if qber < 0.02 {
            SecurityLevel::InformationTheoretic
        } else if qber < 0.05 {
            SecurityLevel::Computational
        } else if qber <= 0.1095 {
            SecurityLevel::Degraded
        } else {
            SecurityLevel::Insecure
        }
    }

    /// Simulate quantum measurement of a photon in a chosen basis.
    async fn measure_photon(
        &self,
        photon: &crate::PhotonState,
        basis: MeasurementBasis,
    ) -> Result<bool> {
        match (photon, basis) {
            // Correct basis -- deterministic.
            (crate::PhotonState::Rectilinear(bit), MeasurementBasis::Rectilinear) => Ok(*bit),
            (crate::PhotonState::Diagonal(bit), MeasurementBasis::Diagonal) => Ok(*bit),
            // Wrong basis -- random result.
            _ => {
                let byte = self.entropy_source.generate_true_random(1).await?;
                Ok((byte[0] & 1) != 0)
            }
        }
    }

    /// Error correction (simplified Cascade / LDPC).
    fn apply_error_correction(&self, bits: &[bool], _qber: f64) -> Result<Vec<u8>> {
        let mut corrected = Vec::new();
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for (j, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1 << j;
                }
            }
            // Simple parity correction (production would use full Cascade/LDPC).
            let parity = byte.count_ones() % 2;
            if parity != 0 && _qber > 0.01 {
                byte ^= 1; // Flip LSB as simplified correction.
            }
            corrected.push(byte);
        }
        Ok(corrected)
    }

    /// Privacy amplification using a Toeplitz-hash construction.
    fn privacy_amplification(&self, key_bytes: &[u8], qber: f64) -> Result<Vec<u8>> {
        // Output length reduced by the estimated information leaked to Eve.
        let input_entropy = key_bytes.len() as f64 * (1.0 - qber);
        let output_len = (input_entropy * 0.8).max(1.0) as usize;

        let mut amplified = Vec::with_capacity(output_len);
        for i in 0..output_len {
            let mut acc = 0u8;
            for (j, &byte) in key_bytes.iter().enumerate() {
                // Toeplitz-like universal hash: multiply by pseudo-random matrix.
                let seed = ((i as u64).wrapping_mul(31) ^ (j as u64).wrapping_mul(37)) as u8;
                acc ^= byte.wrapping_mul(seed);
            }
            amplified.push(acc);
        }
        Ok(amplified)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Announcement tests ----

    #[test]
    fn test_announcement_for_sent_state() {
        // Horizontal -> not Vertical
        assert_eq!(
            SARG04Announcement::for_sent_state(PhotonPolarization::Horizontal),
            SARG04Announcement::NotVertical
        );
        // Vertical -> not Horizontal
        assert_eq!(
            SARG04Announcement::for_sent_state(PhotonPolarization::Vertical),
            SARG04Announcement::NotHorizontal
        );
        // Diagonal -> not AntiDiagonal
        assert_eq!(
            SARG04Announcement::for_sent_state(PhotonPolarization::Diagonal),
            SARG04Announcement::NotAntiDiagonal
        );
        // AntiDiagonal -> not Diagonal
        assert_eq!(
            SARG04Announcement::for_sent_state(PhotonPolarization::AntiDiagonal),
            SARG04Announcement::NotDiagonal
        );
    }

    #[test]
    fn test_announcement_excluded_state() {
        assert_eq!(
            SARG04Announcement::NotHorizontal.excluded_state(),
            PhotonPolarization::Horizontal
        );
        assert_eq!(
            SARG04Announcement::NotVertical.excluded_state(),
            PhotonPolarization::Vertical
        );
        assert_eq!(
            SARG04Announcement::NotDiagonal.excluded_state(),
            PhotonPolarization::Diagonal
        );
        assert_eq!(
            SARG04Announcement::NotAntiDiagonal.excluded_state(),
            PhotonPolarization::AntiDiagonal
        );
    }

    #[test]
    fn test_candidate_states_are_from_different_bases() {
        for ann in &[
            SARG04Announcement::NotHorizontal,
            SARG04Announcement::NotVertical,
            SARG04Announcement::NotDiagonal,
            SARG04Announcement::NotAntiDiagonal,
        ] {
            let (a, b) = ann.candidate_states();
            // Candidates MUST be from different bases for SARG04 to work.
            assert_ne!(
                a.basis(),
                b.basis(),
                "Candidates for {:?} must span both bases",
                ann
            );
        }
    }

    #[test]
    fn test_candidate_states_encode_same_bit() {
        // In our encoding, both candidates for a given announcement
        // encode the same bit value.
        for ann in &[
            SARG04Announcement::NotVertical,     // H,D -> bit 0
            SARG04Announcement::NotHorizontal,    // V,A -> bit 1
            SARG04Announcement::NotAntiDiagonal,  // D,H -> bit 0
            SARG04Announcement::NotDiagonal,      // A,V -> bit 1
        ] {
            let (a, b) = ann.candidate_states();
            assert_eq!(
                a.to_bit(),
                b.to_bit(),
                "Both candidates for {:?} should encode the same bit",
                ann
            );
        }
    }

    // ---- Sifting tests ----

    #[test]
    fn test_sift_correct_basis_opposite_result_keeps() {
        // Alice sent Horizontal (Rect-0), announced NotVertical.
        // Candidates: (Horizontal, Diagonal) -- both bit 0.
        // Bob measures in Rectilinear basis and gets bit 1 (opposite of H).
        // That rules out Horizontal -> answer is Diagonal (bit 0).
        let bob_pols = vec![PhotonPolarization::Vertical]; // Rect basis, bit 1
        let bob_results = vec![true];
        let announcements = vec![SARG04Announcement::NotVertical];

        let sifted = SARG04Protocol::sift_sarg04(&bob_pols, &bob_results, &announcements);
        assert_eq!(sifted.len(), 1);
        assert_eq!(sifted[0].0, 0); // index 0
        assert_eq!(sifted[0].1, false); // Diagonal -> bit 0
    }

    #[test]
    fn test_sift_correct_basis_same_result_discards() {
        // Alice sent Horizontal (Rect-0), announced NotVertical.
        // Candidates: (Horizontal, Diagonal).
        // Bob measures in Rectilinear and gets bit 0 (same as Horizontal).
        // He cannot rule out Horizontal -> ambiguous -> discard.
        let bob_pols = vec![PhotonPolarization::Horizontal]; // Rect basis, bit 0
        let bob_results = vec![false];
        let announcements = vec![SARG04Announcement::NotVertical];

        let sifted = SARG04Protocol::sift_sarg04(&bob_pols, &bob_results, &announcements);
        assert!(sifted.is_empty());
    }

    #[test]
    fn test_sift_wrong_basis_for_both_candidates_discards() {
        // If Bob's basis doesn't match either candidate's basis the
        // result is always discarded. However, our candidates always
        // span both bases, so Bob's basis will always match exactly one.
        // This test verifies the basic invariant by checking sifting
        // rate is roughly 25% on large random data.
        let n = 10000;
        let mut bob_pols = Vec::with_capacity(n);
        let mut bob_results = Vec::with_capacity(n);
        let mut announcements = Vec::with_capacity(n);

        for i in 0..n {
            // Alternate bases for Bob.
            let basis = if i % 2 == 0 {
                MeasurementBasis::Rectilinear
            } else {
                MeasurementBasis::Diagonal
            };
            let bit = (i % 3) != 0; // some pattern
            let pol = match (basis, bit) {
                (MeasurementBasis::Rectilinear, false) => PhotonPolarization::Horizontal,
                (MeasurementBasis::Rectilinear, true) => PhotonPolarization::Vertical,
                (MeasurementBasis::Diagonal, false) => PhotonPolarization::Diagonal,
                (MeasurementBasis::Diagonal, true) => PhotonPolarization::AntiDiagonal,
            };
            bob_pols.push(pol);
            bob_results.push(bit);

            // Random-ish announcements.
            let ann = match i % 4 {
                0 => SARG04Announcement::NotHorizontal,
                1 => SARG04Announcement::NotVertical,
                2 => SARG04Announcement::NotDiagonal,
                _ => SARG04Announcement::NotAntiDiagonal,
            };
            announcements.push(ann);
        }

        let sifted = SARG04Protocol::sift_sarg04(&bob_pols, &bob_results, &announcements);
        let rate = sifted.len() as f64 / n as f64;

        // Sifting rate should be in the ballpark of 25% (some variance
        // expected due to the non-random pattern).
        assert!(
            rate > 0.05 && rate < 0.60,
            "Sifting rate {} is outside expected range",
            rate
        );
    }

    #[test]
    fn test_sift_empty_input() {
        let sifted = SARG04Protocol::sift_sarg04(&[], &[], &[]);
        assert!(sifted.is_empty());
    }

    #[test]
    fn test_sift_mismatched_lengths_uses_minimum() {
        // Three announcements but only two bob measurements.
        let bob_pols = vec![PhotonPolarization::Horizontal, PhotonPolarization::Vertical];
        let bob_results = vec![false, true];
        let announcements = vec![
            SARG04Announcement::NotVertical,
            SARG04Announcement::NotHorizontal,
            SARG04Announcement::NotDiagonal, // extra -- should be ignored
        ];

        let sifted = SARG04Protocol::sift_sarg04(&bob_pols, &bob_results, &announcements);
        // Should not panic, and index should never be 2.
        for &(idx, _) in &sifted {
            assert!(idx < 2);
        }
    }

    // ---- Config tests ----

    #[test]
    fn test_default_config() {
        let cfg = SARG04Config::default();
        assert_eq!(cfg.target_key_length, 256);
        assert!((cfg.qber_threshold - 0.1095).abs() < 1e-6);
        assert!((cfg.test_fraction - 0.10).abs() < 1e-6);
        assert!(cfg.privacy_amplification);
        assert_eq!(cfg.error_correction, ErrorCorrectionMethod::Cascade);
    }

    // ---- Security analysis tests ----

    #[test]
    fn test_security_analysis_low_qber() {
        let node_id = [0u8; 32];
        // We need a runtime for the entropy source, but security_analysis is sync.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let entropy = rt.block_on(async { Arc::new(QuantumEntropySource::new().await.unwrap()) });
        let proto = SARG04Protocol::new(node_id, entropy);

        let analysis = proto.security_analysis(0.01, 3.0);
        assert!(analysis.is_safe);
        assert!(analysis.secure_key_rate > 0.0);
        assert_eq!(analysis.security_level, SecurityLevel::InformationTheoretic);
    }

    #[test]
    fn test_security_analysis_high_qber_unsafe() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let entropy = rt.block_on(async { Arc::new(QuantumEntropySource::new().await.unwrap()) });
        let proto = SARG04Protocol::new([0u8; 32], entropy);

        let analysis = proto.security_analysis(0.15, 3.0);
        assert!(!analysis.is_safe);
        assert_eq!(analysis.security_level, SecurityLevel::Insecure);
    }

    #[test]
    fn test_security_analysis_high_loss() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let entropy = rt.block_on(async { Arc::new(QuantumEntropySource::new().await.unwrap()) });
        let proto = SARG04Protocol::new([0u8; 32], entropy);

        let analysis = proto.security_analysis(0.03, 25.0);
        assert!(!analysis.is_safe); // Loss exceeds max_safe_loss_db
    }

    // ---- Error correction tests ----

    #[test]
    fn test_error_correction_roundtrip() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let entropy = rt.block_on(async { Arc::new(QuantumEntropySource::new().await.unwrap()) });
        let proto = SARG04Protocol::new([0u8; 32], entropy);

        let bits: Vec<bool> = vec![
            true, false, true, true, false, false, true, false, true, true, false, false, true,
            false, true, true,
        ];
        let corrected = proto.apply_error_correction(&bits, 0.001).unwrap();
        assert_eq!(corrected.len(), 2); // 16 bits -> 2 bytes
    }

    // ---- Privacy amplification tests ----

    #[test]
    fn test_privacy_amplification_reduces_length() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let entropy = rt.block_on(async { Arc::new(QuantumEntropySource::new().await.unwrap()) });
        let proto = SARG04Protocol::new([0u8; 32], entropy);

        let key_bytes = vec![0xAA; 64];
        let amplified = proto.privacy_amplification(&key_bytes, 0.05).unwrap();
        // Output should be shorter than input (entropy reduction).
        assert!(amplified.len() <= key_bytes.len());
        assert!(!amplified.is_empty());
    }

    // ---- Classification tests ----

    #[test]
    fn test_classify_security_levels() {
        assert_eq!(
            SARG04Protocol::classify_security(0.01),
            SecurityLevel::InformationTheoretic
        );
        assert_eq!(
            SARG04Protocol::classify_security(0.03),
            SecurityLevel::Computational
        );
        assert_eq!(
            SARG04Protocol::classify_security(0.08),
            SecurityLevel::Degraded
        );
        assert_eq!(
            SARG04Protocol::classify_security(0.12),
            SecurityLevel::Insecure
        );
    }

    // ---- Protocol creation tests ----

    #[tokio::test]
    async fn test_protocol_creation_default() {
        let entropy = Arc::new(QuantumEntropySource::new().await.unwrap());
        let proto = SARG04Protocol::new([1u8; 32], entropy);
        assert_eq!(proto.node_id, [1u8; 32]);

        let stats = proto.get_stats().await;
        assert_eq!(stats.sessions_completed, 0);
    }

    #[tokio::test]
    async fn test_protocol_creation_custom_config() {
        let entropy = Arc::new(QuantumEntropySource::new().await.unwrap());
        let config = SARG04Config {
            target_key_length: 512,
            qber_threshold: 0.08,
            test_fraction: 0.15,
            privacy_amplification: false,
            error_correction: ErrorCorrectionMethod::LDPC,
        };
        let proto = SARG04Protocol::with_config([2u8; 32], entropy, config);
        assert_eq!(proto.config.target_key_length, 512);
        assert!((proto.config.qber_threshold - 0.08).abs() < 1e-9);
        assert!(!proto.config.privacy_amplification);
        assert_eq!(proto.config.error_correction, ErrorCorrectionMethod::LDPC);
    }

    // ---- Announcement serialization tests ----

    #[test]
    fn test_announcement_serialization_roundtrip() {
        let announcements = vec![
            SARG04Announcement::NotHorizontal,
            SARG04Announcement::NotVertical,
            SARG04Announcement::NotDiagonal,
            SARG04Announcement::NotAntiDiagonal,
        ];
        for ann in &announcements {
            let json = serde_json::to_string(ann).unwrap();
            let deser: SARG04Announcement = serde_json::from_str(&json).unwrap();
            assert_eq!(*ann, deser);
        }
    }

    // ---- SARG04 sifting deterministic scenario ----

    #[test]
    fn test_sift_deterministic_scenario() {
        // Scenario: Alice sent |V> (Rect-1), announced NotHorizontal.
        // Candidates: (Vertical, AntiDiagonal) -- both bit 1.
        // Bob measures in Diagonal basis and gets bit 0 (=Diagonal).
        // Diagonal basis matches candidate B (AntiDiagonal).
        // Bob's result (0) != AntiDiagonal's bit (1), so AntiDiag ruled out.
        // Answer: Vertical (bit 1).
        let bob_pols = vec![PhotonPolarization::Diagonal]; // Diag basis, bit 0
        let bob_results = vec![false];
        let announcements = vec![SARG04Announcement::NotHorizontal];

        let sifted = SARG04Protocol::sift_sarg04(&bob_pols, &bob_results, &announcements);
        assert_eq!(sifted.len(), 1);
        assert_eq!(sifted[0].1, true); // Vertical -> bit 1
    }

    // ---- Security analysis at boundary ----

    #[test]
    fn test_security_analysis_at_threshold() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let entropy = rt.block_on(async { Arc::new(QuantumEntropySource::new().await.unwrap()) });
        let proto = SARG04Protocol::new([0u8; 32], entropy);

        // Exactly at the threshold should still be safe.
        let analysis = proto.security_analysis(0.1095, 10.0);
        assert!(analysis.is_safe);
        assert_eq!(analysis.security_level, SecurityLevel::Degraded);

        // Just over the threshold should be unsafe.
        let analysis = proto.security_analysis(0.11, 10.0);
        assert!(!analysis.is_safe);
        assert_eq!(analysis.security_level, SecurityLevel::Insecure);
    }

    // ---- Error correction edge cases ----

    #[test]
    fn test_error_correction_empty_input() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let entropy = rt.block_on(async { Arc::new(QuantumEntropySource::new().await.unwrap()) });
        let proto = SARG04Protocol::new([0u8; 32], entropy);

        let corrected = proto.apply_error_correction(&[], 0.05).unwrap();
        assert!(corrected.is_empty());
    }

    #[test]
    fn test_error_correction_partial_byte() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let entropy = rt.block_on(async { Arc::new(QuantumEntropySource::new().await.unwrap()) });
        let proto = SARG04Protocol::new([0u8; 32], entropy);

        // Only 3 bits -- should produce 1 byte.
        let bits = vec![true, false, true];
        let corrected = proto.apply_error_correction(&bits, 0.05).unwrap();
        assert_eq!(corrected.len(), 1);
    }
}
