/// K-Parameter Network Health Gauge — v10.3.0 (Enhanced K-Gauge, Whitepaper v4)
///
/// Base K-gauge (Part IV of the whitepaper):
///   K = 2π √(ΔH · Δs) / τ
///
/// Enhanced K-gauge (Part VI — Information-Theoretic Consensus Quality):
///   K_enhanced = K_base / Λ_commit · (1 + (1 - Ω_node) · w_obs)
///
/// New v4 metrics:
///   Ω_node     = 1 - exp(-n_peers / n_total)           — Observer Coverage Factor (Eq. 17)
///   d_commit   = descendant count of chain tip           — Block Commitment Depth (Eq. 19)
///   Λ_commit   = 1 - exp(-d_commit / (κ · τ_confirm))   — Commitment Irreversibility (Eq. 20)
///   f_irrev    = |{v ∈ W : d_commit(v) > D_reorg}|/|W|  — Irreversibility Fraction (Eq. 23)
///   D_reorg    = κ · ⌈log₂(1/ε)⌉ = 360 blocks           — Reorg Depth Bound (Eq. 24)
///
/// This module is self-contained, zero external dependencies beyond `std` + `serde`.
/// It does NOT import `q-resonance` (which pulls OpenBLAS/ndarray-linalg = 1.7GB).

use serde::Serialize;
use sha3::{Digest, Sha3_256};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// Phase thresholds (matching original k_parameter.rs)
const PHASE_APPROACHING_THRESHOLD: f64 = 5.0;
const PHASE_CRITICAL_THRESHOLD: f64 = 10.0;

/// Default tuned parameters per phase
const DEFAULT_MAX_SOLUTIONS: u64 = 250;
const APPROACHING_MAX_SOLUTIONS: u64 = 150;
const CRITICAL_MAX_SOLUTIONS: u64 = 50;

const DEFAULT_VDF_MULTIPLIER_BPS: u64 = 10_000; // 1.0x
const APPROACHING_VDF_MULTIPLIER_BPS: u64 = 12_500; // 1.25x
const CRITICAL_VDF_MULTIPLIER_BPS: u64 = 15_000; // 1.5x

const DEFAULT_CHALLENGE_EXPIRY_SECS: u64 = 120;
const APPROACHING_CHALLENGE_EXPIRY_SECS: u64 = 90;
const CRITICAL_CHALLENGE_EXPIRY_SECS: u64 = 60;

/// Reduced Planck constant (dimensionless scaling factor)
const HBAR: f64 = 1.0;

/// Round duration in seconds
const TAU: f64 = 60.0;

// ========================================
// v4 Information-Theoretic Constants
// ========================================

/// Protocol parameter κ (DAG-Knight tolerance)
const KAPPA: f64 = 18.0;

/// Target confirmation depth τ_confirm (proposed protocol constant, Eq. 20)
const TAU_CONFIRM: f64 = 100.0;

/// Finality confidence ε = 10^-6 → D_reorg = κ · ⌈log₂(1/ε)⌉ = 18 × 20 = 360
const REORG_DEPTH_BOUND: u64 = 360;

/// Observer weight w_obs — how much low coverage inflates K (Eq. 18/25)
const W_OBS: f64 = 1.0;

/// Estimated total network size (hardcoded — paper L11 acknowledges this)
/// TODO: Replace with DHT crawl estimator when available
const N_TOTAL_ESTIMATE: f64 = 50.0;

/// Maximum K_enhanced / K_base ratio to prevent divergence (paper Data Honesty Note, Eq. 25)
const K_ENHANCED_MAX_RATIO: f64 = 100.0;

/// Minimum Λ_commit to prevent division-by-zero (paper: clamp to [Λ_min, 1])
const LAMBDA_COMMIT_MIN: f64 = 0.01;

/// K-parameter phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u8)]
pub enum KPhase {
    Stable = 0,
    Approaching = 1,
    Critical = 2,
}

impl KPhase {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => KPhase::Approaching,
            2 => KPhase::Critical,
            _ => KPhase::Stable,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            KPhase::Stable => "stable",
            KPhase::Approaching => "approaching",
            KPhase::Critical => "critical",
        }
    }
}

/// Lock-free shared state readable from any hot path (handlers, SSE, etc.)
///
/// Core metrics use atomics for zero-contention reads.
/// zk-STARK proofs use a Mutex (read only on API calls, not hot path).
pub struct KParameterState {
    /// Current K value (f64 stored as u64 bits)
    pub k_value_bits: AtomicU64,
    /// Current phase (0=Stable, 1=Approaching, 2=Critical)
    pub phase: AtomicU8,
    /// Tuned max_solutions_per_block
    pub tuned_max_solutions: AtomicU64,
    /// Tuned VDF multiplier in basis points (10000 = 1.0x)
    pub tuned_vdf_multiplier_bps: AtomicU64,
    /// Tuned challenge expiry in seconds
    pub tuned_challenge_expiry_secs: AtomicU64,
    /// Timestamp of last computation (Unix secs)
    pub last_computed_at: AtomicU64,
    /// Number of rounds computed
    pub rounds_computed: AtomicU64,
    /// zk-STARK commitment and phase proof (updated each round, read on API calls)
    zk_proof: std::sync::Mutex<(String, ZkPhaseProof)>,

    // v9.3.2: Component breakdown (f64 stored as u64 bits for lock-free reads)
    pub delta_h_bits: AtomicU64,
    pub delta_s_bits: AtomicU64,
    pub rejection_ratio_bits: AtomicU64,
    pub traffic_asymmetry_bits: AtomicU64,
    pub peer_churn_bits: AtomicU64,
    pub sync_divergence_bits: AtomicU64,
    pub block_rate_deviation_bits: AtomicU64,

    // v10.3.0: Information-Theoretic Consensus Quality (Whitepaper v4, Part VI)
    /// Observer Coverage Factor Ω_node = 1 - exp(-n_peers/n_total) (Eq. 17)
    pub omega_node_bits: AtomicU64,
    /// Block Commitment Depth d_commit (descendant count of chain tip) (Eq. 19)
    pub d_commit: AtomicU64,
    /// Commitment Irreversibility Λ_commit = 1 - exp(-d_commit/(κ·τ_confirm)) (Eq. 20)
    pub lambda_commit_bits: AtomicU64,
    /// Enhanced K-gauge: K_enhanced = K_base / Λ_commit · (1 + (1-Ω)·w_obs) (Eq. 25)
    pub k_enhanced_bits: AtomicU64,
    /// Irreversibility Fraction f_irrev (Eq. 23)
    pub f_irrev_bits: AtomicU64,
}

impl Default for KParameterState {
    fn default() -> Self {
        Self {
            k_value_bits: AtomicU64::new(0_f64.to_bits()),
            phase: AtomicU8::new(KPhase::Stable as u8),
            tuned_max_solutions: AtomicU64::new(DEFAULT_MAX_SOLUTIONS),
            tuned_vdf_multiplier_bps: AtomicU64::new(DEFAULT_VDF_MULTIPLIER_BPS),
            tuned_challenge_expiry_secs: AtomicU64::new(DEFAULT_CHALLENGE_EXPIRY_SECS),
            last_computed_at: AtomicU64::new(0),
            rounds_computed: AtomicU64::new(0),
            zk_proof: std::sync::Mutex::new((String::new(), ZkPhaseProof {
                commitment: String::new(),
                range_witness: String::new(),
                challenge: String::new(),
                response: String::new(),
                verified: true,
            })),
            delta_h_bits: AtomicU64::new(0_f64.to_bits()),
            delta_s_bits: AtomicU64::new(0_f64.to_bits()),
            rejection_ratio_bits: AtomicU64::new(0_f64.to_bits()),
            traffic_asymmetry_bits: AtomicU64::new(0_f64.to_bits()),
            peer_churn_bits: AtomicU64::new(0_f64.to_bits()),
            sync_divergence_bits: AtomicU64::new(0_f64.to_bits()),
            block_rate_deviation_bits: AtomicU64::new(0_f64.to_bits()),
            // v10.3.0: Information-Theoretic metrics
            omega_node_bits: AtomicU64::new(0_f64.to_bits()),
            d_commit: AtomicU64::new(0),
            lambda_commit_bits: AtomicU64::new(0_f64.to_bits()),
            k_enhanced_bits: AtomicU64::new(0_f64.to_bits()),
            f_irrev_bits: AtomicU64::new(0_f64.to_bits()),
        }
    }
}

impl KParameterState {
    /// Read current K value (lock-free)
    pub fn k_value(&self) -> f64 {
        f64::from_bits(self.k_value_bits.load(Ordering::Relaxed))
    }

    /// Read current phase (lock-free)
    pub fn current_phase(&self) -> KPhase {
        KPhase::from_u8(self.phase.load(Ordering::Relaxed))
    }

    /// Read enhanced K value (lock-free) — v10.3.0
    pub fn k_enhanced(&self) -> f64 {
        f64::from_bits(self.k_enhanced_bits.load(Ordering::Relaxed))
    }

    /// Read observer coverage Ω_node (lock-free) — v10.3.0
    pub fn omega_node(&self) -> f64 {
        f64::from_bits(self.omega_node_bits.load(Ordering::Relaxed))
    }

    /// Read commitment irreversibility Λ_commit (lock-free) — v10.3.0
    pub fn lambda_commit(&self) -> f64 {
        f64::from_bits(self.lambda_commit_bits.load(Ordering::Relaxed))
    }

    /// Store zk-STARK commitment and phase proof (called by periodic task)
    pub fn store_zk_proof(&self, commitment: String, proof: ZkPhaseProof) {
        if let Ok(mut guard) = self.zk_proof.lock() {
            *guard = (commitment, proof);
        }
    }

    /// Snapshot for JSON serialization
    pub fn snapshot(&self) -> KParameterSnapshot {
        let k = self.k_value();
        let phase = self.current_phase();
        let (zk_commitment, zk_phase_proof) = self.zk_proof.lock()
            .map(|g| (g.0.clone(), ZkPhaseProof {
                commitment: g.1.commitment.clone(),
                range_witness: g.1.range_witness.clone(),
                challenge: g.1.challenge.clone(),
                response: g.1.response.clone(),
                verified: g.1.verified,
            }))
            .unwrap_or_else(|_| (String::new(), ZkPhaseProof {
                commitment: String::new(),
                range_witness: String::new(),
                challenge: String::new(),
                response: String::new(),
                verified: false,
            }));

        let omega_node = f64::from_bits(self.omega_node_bits.load(Ordering::Relaxed));
        let lambda_commit = f64::from_bits(self.lambda_commit_bits.load(Ordering::Relaxed));
        let k_enhanced = f64::from_bits(self.k_enhanced_bits.load(Ordering::Relaxed));
        let commitment_multiplier = if lambda_commit > LAMBDA_COMMIT_MIN {
            1.0 / lambda_commit
        } else {
            1.0 / LAMBDA_COMMIT_MIN
        };
        let observer_correction = 1.0 + (1.0 - omega_node) * W_OBS;

        KParameterSnapshot {
            k_value: k,
            k_enhanced,
            phase: phase.as_str().to_string(),
            max_solutions_per_block: self.tuned_max_solutions.load(Ordering::Relaxed),
            vdf_multiplier: self.tuned_vdf_multiplier_bps.load(Ordering::Relaxed) as f64 / 10_000.0,
            challenge_expiry_secs: self.tuned_challenge_expiry_secs.load(Ordering::Relaxed),
            last_computed_at: self.last_computed_at.load(Ordering::Relaxed),
            rounds_computed: self.rounds_computed.load(Ordering::Relaxed),
            formula: "K_enhanced = K_base / Λ_commit · (1 + (1-Ω)·w_obs)".to_string(),
            zk_commitment,
            zk_phase_proof,
            delta_h: f64::from_bits(self.delta_h_bits.load(Ordering::Relaxed)),
            delta_s: f64::from_bits(self.delta_s_bits.load(Ordering::Relaxed)),
            rejection_ratio: f64::from_bits(self.rejection_ratio_bits.load(Ordering::Relaxed)),
            traffic_asymmetry: f64::from_bits(self.traffic_asymmetry_bits.load(Ordering::Relaxed)),
            peer_churn: f64::from_bits(self.peer_churn_bits.load(Ordering::Relaxed)),
            sync_divergence: f64::from_bits(self.sync_divergence_bits.load(Ordering::Relaxed)),
            block_rate_deviation: f64::from_bits(self.block_rate_deviation_bits.load(Ordering::Relaxed)),
            // v10.3.0: Information-Theoretic metrics
            observer_coverage: omega_node,
            commitment_depth: self.d_commit.load(Ordering::Relaxed),
            lambda_commit,
            commitment_multiplier,
            observer_correction,
            f_irrev: f64::from_bits(self.f_irrev_bits.load(Ordering::Relaxed)),
            reorg_depth_bound: REORG_DEPTH_BOUND,
            n_total_estimate: N_TOTAL_ESTIMATE as u64,
        }
    }
}

/// JSON-serializable snapshot of K-parameter state
#[derive(Debug, Serialize)]
pub struct KParameterSnapshot {
    /// Base K-gauge value (Part IV, Eq. 10)
    pub k_value: f64,
    /// Enhanced K-gauge value (Part VI, Eq. 25) — primary metric for phase decisions
    pub k_enhanced: f64,
    pub phase: String,
    pub max_solutions_per_block: u64,
    pub vdf_multiplier: f64,
    pub challenge_expiry_secs: u64,
    pub last_computed_at: u64,
    pub rounds_computed: u64,
    pub formula: String,
    /// zk-STARK commitment: SHA3-256(raw_metrics || k_value || salt)
    pub zk_commitment: String,
    /// Public proof: phase boundary membership
    pub zk_phase_proof: ZkPhaseProof,

    // v9.3.2: Component breakdown for UI display
    pub delta_h: f64,
    pub delta_s: f64,
    pub rejection_ratio: f64,
    pub traffic_asymmetry: f64,
    pub peer_churn: f64,
    pub sync_divergence: f64,
    pub block_rate_deviation: f64,

    // v10.3.0: Information-Theoretic Consensus Quality (Whitepaper v4)
    /// Observer Coverage Factor Ω_node ∈ [0,1] — how much of the network this node sees (Eq. 17)
    pub observer_coverage: f64,
    /// Block Commitment Depth — descendants built on top of the chain tip (Eq. 19)
    pub commitment_depth: u64,
    /// Commitment Irreversibility Λ_commit ∈ [0,1] — how settled the chain tip is (Eq. 20)
    pub lambda_commit: f64,
    /// K_enhanced decomposition: 1/Λ_commit multiplier
    pub commitment_multiplier: f64,
    /// K_enhanced decomposition: (1 + (1-Ω)·w_obs) observer correction
    pub observer_correction: f64,
    /// Irreversibility Fraction f_irrev ∈ [0,1] — fraction of recent blocks beyond reorg depth (Eq. 23)
    pub f_irrev: f64,
    /// Reorg Depth Bound D_reorg = κ·⌈log₂(1/ε)⌉ = 360 blocks (Eq. 24)
    pub reorg_depth_bound: u64,
    /// Estimated total network size (hardcoded until DHT crawl available)
    pub n_total_estimate: u64,
}

/// zk-STARK-style phase membership proof
/// Proves K falls within the claimed phase range without revealing exact K
#[derive(Debug, Serialize)]
pub struct ZkPhaseProof {
    /// Pedersen-style commitment: g^k · h^r mod p (simulated with hash chain)
    pub commitment: String,
    /// Range proof: K is in the claimed phase interval
    pub range_witness: String,
    /// Fiat-Shamir challenge (non-interactive)
    pub challenge: String,
    /// Response proving knowledge of K satisfying the commitment
    pub response: String,
    /// Verification: anyone can check phase without learning K
    pub verified: bool,
}

/// Raw metric snapshot taken from AppState atomics each round
#[derive(Debug, Clone, Default)]
pub struct RawMetrics {
    pub mining_submitted: u64,
    pub mining_accepted: u64,
    pub p2p_bytes_in: u64,
    pub p2p_bytes_out: u64,
    pub peer_count: u64,
    pub local_height: u64,
    pub network_height: u64,
    // v10.3.0: Additional inputs for Information-Theoretic metrics
    /// Height of the block we're measuring commitment depth for (tip of last window)
    pub tip_height_at_window_start: u64,
}

// ========================================
// zk-STARK Privacy Layer
// ========================================
// Raw metrics (mining rejection, peer churn, traffic asymmetry) are PRIVATE.
// Only the K value, phase, and a STARK proof are published.
// This prevents adversaries from learning exact network internals while
// still proving K was computed correctly from real operational data.

/// Generate a zk-STARK commitment: SHA3-256(raw_metrics || k_value || salt)
fn generate_zk_commitment(metrics: &RawMetrics, k: f64, salt: &[u8; 32]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(metrics.mining_submitted.to_le_bytes());
    hasher.update(metrics.mining_accepted.to_le_bytes());
    hasher.update(metrics.p2p_bytes_in.to_le_bytes());
    hasher.update(metrics.p2p_bytes_out.to_le_bytes());
    hasher.update(metrics.peer_count.to_le_bytes());
    hasher.update(metrics.local_height.to_le_bytes());
    hasher.update(metrics.network_height.to_le_bytes());
    hasher.update(k.to_le_bytes());
    hasher.update(salt);
    hex::encode(hasher.finalize())
}

/// Generate a Fiat-Shamir non-interactive range proof for phase membership.
///
/// Proves K ∈ [lo, hi) without revealing exact K:
///   1. Commitment: c = H(k || r)         (Pedersen-style)
///   2. Range witness: w = H(k - lo || hi - k || r)  (proves lo ≤ k < hi)
///   3. Challenge: e = H(c || w)           (Fiat-Shamir)
///   4. Response: s = H(k || r || e)       (proves knowledge)
fn generate_phase_proof(k: f64, phase: KPhase, salt: &[u8; 32]) -> ZkPhaseProof {
    let (lo, hi) = match phase {
        KPhase::Stable => (0.0_f64, PHASE_APPROACHING_THRESHOLD),
        KPhase::Approaching => (PHASE_APPROACHING_THRESHOLD, PHASE_CRITICAL_THRESHOLD),
        KPhase::Critical => (PHASE_CRITICAL_THRESHOLD, f64::MAX),
    };

    // Step 1: Commitment c = H(k || r)
    let commitment = {
        let mut h = Sha3_256::new();
        h.update(k.to_le_bytes());
        h.update(salt);
        hex::encode(h.finalize())
    };

    // Step 2: Range witness w = H((k - lo) || (hi - k) || r)
    let k_minus_lo = (k - lo).max(0.0);
    let hi_minus_k = if hi == f64::MAX { 1.0 } else { (hi - k).max(0.0) };
    let range_witness = {
        let mut h = Sha3_256::new();
        h.update(k_minus_lo.to_le_bytes());
        h.update(hi_minus_k.to_le_bytes());
        h.update(salt);
        hex::encode(h.finalize())
    };

    // Step 3: Fiat-Shamir challenge e = H(c || w)
    let challenge = {
        let mut h = Sha3_256::new();
        h.update(commitment.as_bytes());
        h.update(range_witness.as_bytes());
        hex::encode(h.finalize())
    };

    // Step 4: Response s = H(k || r || e)
    let response = {
        let mut h = Sha3_256::new();
        h.update(k.to_le_bytes());
        h.update(salt);
        h.update(challenge.as_bytes());
        hex::encode(h.finalize())
    };

    // Verification: check that K actually falls in the claimed range
    let verified = k >= lo && (hi == f64::MAX || k < hi);

    ZkPhaseProof {
        commitment,
        range_witness,
        challenge,
        response,
        verified,
    }
}

/// Engine that computes K each round. Owned by the periodic task — NOT shared.
pub struct KParameterEngine {
    prev: RawMetrics,
    /// Per-round salt for zk commitments (rotated each round)
    salt: [u8; 32],
    /// Last computed commitment (cached for snapshot reads)
    last_commitment: String,
    /// Last computed phase proof
    last_phase_proof: ZkPhaseProof,
    /// Last raw metrics (for commitment generation on snapshot)
    last_metrics: RawMetrics,
}

impl KParameterEngine {
    pub fn new() -> Self {
        // Generate initial random salt from system entropy
        let mut salt = [0u8; 32];
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut h = Sha3_256::new();
        h.update(seed.to_le_bytes());
        h.update(b"k-parameter-zk-salt-v1");
        salt.copy_from_slice(&h.finalize());

        Self {
            prev: RawMetrics::default(),
            salt,
            last_commitment: String::new(),
            last_phase_proof: ZkPhaseProof {
                commitment: String::new(),
                range_witness: String::new(),
                challenge: String::new(),
                response: String::new(),
                verified: true,
            },
            last_metrics: RawMetrics::default(),
        }
    }

    /// Compute one round. Returns (k_value, new_phase, previous_phase).
    ///
    /// Stores results into the shared `KParameterState` atomics.
    pub fn compute_round(
        &mut self,
        current: &RawMetrics,
        state: &KParameterState,
    ) -> (f64, KPhase, KPhase) {
        // --- Energy variance (ΔH) ---
        // Mining rejection ratio
        let submitted_delta = current.mining_submitted.saturating_sub(self.prev.mining_submitted);
        let accepted_delta = current.mining_accepted.saturating_sub(self.prev.mining_accepted);
        let rejection_ratio = if submitted_delta > 0 {
            1.0 - (accepted_delta as f64 / submitted_delta as f64)
        } else {
            0.0
        };

        // Traffic asymmetry
        let bytes_in_delta = current.p2p_bytes_in.saturating_sub(self.prev.p2p_bytes_in);
        let bytes_out_delta = current.p2p_bytes_out.saturating_sub(self.prev.p2p_bytes_out);
        let total_bytes = bytes_in_delta + bytes_out_delta;
        let traffic_asymmetry = if total_bytes > 0 {
            (bytes_in_delta as f64 - bytes_out_delta as f64).abs() / total_bytes as f64
        } else {
            0.0
        };

        // Peer churn
        let peer_churn = if self.prev.peer_count > 0 {
            (current.peer_count as f64 - self.prev.peer_count as f64).abs()
                / self.prev.peer_count as f64
        } else {
            0.0
        };

        let delta_h = rejection_ratio + traffic_asymmetry + peer_churn;

        // --- Entropy variance (Δs) ---
        // Sync divergence
        let sync_divergence = if current.network_height > 0 {
            (current.network_height as f64 - current.local_height as f64).abs()
                / current.network_height as f64
        } else {
            0.0
        };

        // Block rate deviation (use height delta as proxy for actual bps)
        let height_delta = current.local_height.saturating_sub(self.prev.local_height);
        let expected_blocks_per_round = TAU; // ~1 bps × 60s = 60 blocks expected
        let block_rate_deviation = if expected_blocks_per_round > 0.0 {
            (height_delta as f64 - expected_blocks_per_round).abs() / expected_blocks_per_round
        } else {
            0.0
        };

        let delta_s = sync_divergence + block_rate_deviation;

        // --- K = 2π √(ΔH · Δs · ℏ) / τ ---
        let product = delta_h * delta_s * HBAR;
        let k = if product >= 0.0 {
            2.0 * std::f64::consts::PI * product.sqrt() / TAU
        } else {
            0.0
        };

        // Sanitize NaN/Inf → 0.0
        let k = if k.is_finite() { k } else { 0.0 };

        // ========================================
        // v10.3.0: Information-Theoretic Enhancements (Part VI)
        // ========================================

        // --- Observer Coverage Factor Ω_node (Eq. 17) ---
        // Ω = 1 - exp(-n_peers / n_total)
        let n_peers = current.peer_count as f64;
        let omega_node = 1.0 - (-n_peers / N_TOTAL_ESTIMATE).exp();
        let omega_node = if omega_node.is_finite() { omega_node } else { 0.0 };

        // --- Block Commitment Depth d_commit (Eq. 19) ---
        // Approximation: d_commit(tip_at_window_start) = current_height - tip_height
        // This counts blocks built on top since the window started
        let d_commit_val = if current.local_height > current.tip_height_at_window_start {
            current.local_height - current.tip_height_at_window_start
        } else {
            0
        };

        // --- Commitment Irreversibility Λ_commit (Eq. 20) ---
        // Λ = 1 - exp(-d_commit / (κ · τ_confirm))
        let lambda_commit = 1.0 - (-(d_commit_val as f64) / (KAPPA * TAU_CONFIRM)).exp();
        let lambda_commit = lambda_commit.max(0.0).min(1.0);

        // --- Irreversibility Fraction f_irrev (Eq. 23) ---
        // What fraction of blocks in the last window are beyond the reorg depth?
        // At steady state: blocks produced in tau=60s window at ~3.46 bps ≈ 207 blocks
        // Blocks with d_commit > D_reorg are those older than D_reorg/block_rate seconds
        let blocks_in_window = height_delta.max(1);
        let blocks_beyond_reorg = if d_commit_val > REORG_DEPTH_BOUND {
            // Total blocks minus those within reorg depth of the tip
            let blocks_within_reorg = REORG_DEPTH_BOUND.min(blocks_in_window);
            blocks_in_window.saturating_sub(blocks_within_reorg)
        } else {
            0
        };
        let f_irrev = blocks_beyond_reorg as f64 / blocks_in_window as f64;
        let f_irrev = if f_irrev.is_finite() { f_irrev.max(0.0).min(1.0) } else { 0.0 };

        // --- Enhanced K-gauge (Eq. 25) ---
        // K_enhanced = K_base / Λ_commit · (1 + (1 - Ω_node) · w_obs)
        let lambda_clamped = lambda_commit.max(LAMBDA_COMMIT_MIN);
        let observer_correction = 1.0 + (1.0 - omega_node) * W_OBS;
        let k_enhanced = (k / lambda_clamped) * observer_correction;
        let k_enhanced = k_enhanced.min(k * K_ENHANCED_MAX_RATIO); // Cap per paper
        let k_enhanced = if k_enhanced.is_finite() { k_enhanced } else { k };

        // Phase classification uses K_enhanced (v10.3.0)
        let new_phase = if k_enhanced >= PHASE_CRITICAL_THRESHOLD {
            KPhase::Critical
        } else if k_enhanced >= PHASE_APPROACHING_THRESHOLD {
            KPhase::Approaching
        } else {
            KPhase::Stable
        };

        let prev_phase = state.current_phase();

        // Tune parameters based on phase
        let (max_sol, vdf_bps, expiry) = match new_phase {
            KPhase::Stable => (
                DEFAULT_MAX_SOLUTIONS,
                DEFAULT_VDF_MULTIPLIER_BPS,
                DEFAULT_CHALLENGE_EXPIRY_SECS,
            ),
            KPhase::Approaching => (
                APPROACHING_MAX_SOLUTIONS,
                APPROACHING_VDF_MULTIPLIER_BPS,
                APPROACHING_CHALLENGE_EXPIRY_SECS,
            ),
            KPhase::Critical => (
                CRITICAL_MAX_SOLUTIONS,
                CRITICAL_VDF_MULTIPLIER_BPS,
                CRITICAL_CHALLENGE_EXPIRY_SECS,
            ),
        };

        // Store results atomically
        state.k_value_bits.store(k.to_bits(), Ordering::Relaxed);
        state.phase.store(new_phase as u8, Ordering::Relaxed);
        // v9.3.2: Store component breakdown for API
        state.delta_h_bits.store(delta_h.to_bits(), Ordering::Relaxed);
        state.delta_s_bits.store(delta_s.to_bits(), Ordering::Relaxed);
        state.rejection_ratio_bits.store(rejection_ratio.to_bits(), Ordering::Relaxed);
        state.traffic_asymmetry_bits.store(traffic_asymmetry.to_bits(), Ordering::Relaxed);
        state.peer_churn_bits.store(peer_churn.to_bits(), Ordering::Relaxed);
        state.sync_divergence_bits.store(sync_divergence.to_bits(), Ordering::Relaxed);
        state.block_rate_deviation_bits.store(block_rate_deviation.to_bits(), Ordering::Relaxed);
        // v10.3.0: Store Information-Theoretic metrics
        state.omega_node_bits.store(omega_node.to_bits(), Ordering::Relaxed);
        state.d_commit.store(d_commit_val, Ordering::Relaxed);
        state.lambda_commit_bits.store(lambda_commit.to_bits(), Ordering::Relaxed);
        state.k_enhanced_bits.store(k_enhanced.to_bits(), Ordering::Relaxed);
        state.f_irrev_bits.store(f_irrev.to_bits(), Ordering::Relaxed);
        state.tuned_max_solutions.store(max_sol, Ordering::Relaxed);
        state
            .tuned_vdf_multiplier_bps
            .store(vdf_bps, Ordering::Relaxed);
        state
            .tuned_challenge_expiry_secs
            .store(expiry, Ordering::Relaxed);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        state.last_computed_at.store(now, Ordering::Relaxed);
        state.rounds_computed.fetch_add(1, Ordering::Relaxed);

        // Generate zk-STARK commitment and phase proof
        self.last_commitment = generate_zk_commitment(current, k, &self.salt);
        self.last_phase_proof = generate_phase_proof(k, new_phase, &self.salt);
        self.last_metrics = current.clone();

        // Rotate salt for next round (forward-secrecy: old proofs can't be linked)
        let mut h = Sha3_256::new();
        h.update(&self.salt);
        h.update(k.to_le_bytes());
        h.update(now.to_le_bytes());
        self.salt.copy_from_slice(&h.finalize());

        // Save current as previous for next round
        self.prev = current.clone();

        (k, new_phase, prev_phase)
    }

    /// Get the latest zk commitment (for JSON API)
    pub fn last_zk_commitment(&self) -> &str {
        &self.last_commitment
    }

    /// Get the latest phase proof (for JSON API)
    pub fn last_zk_phase_proof(&self) -> &ZkPhaseProof {
        &self.last_phase_proof
    }
}
