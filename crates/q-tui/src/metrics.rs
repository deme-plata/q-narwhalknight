use serde::{Deserialize, Serialize};

/// Direction of a top-mover wallet's net balance change over the ring-buffer window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoverDirection {
    Up,
    Down,
    Flat, // zero delta (defensive — shouldn't occur in top 5)
}

/// A wallet whose balance changed rapidly over the recent block window.
/// Rendered in the dashboard "Top Movers" panel. `addr_prefix` holds the first
/// 4 bytes of the 32-byte address (enough for an 8-hex-char display label).
/// `delta_qug` is the signed sum of balance deltas in 24-decimal QUG units
/// (so 1 QUG = 1e24). Positive means net inflow, negative means net outflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopMover {
    pub addr_prefix: [u8; 4],
    pub delta_qug: i128,
    pub direction: MoverDirection,
}

impl TopMover {
    /// Format the signed delta as a human-readable QUG amount with K/M/B suffix.
    /// Internal `delta_qug` is in 24-decimal units, so we divide by 1e24 first.
    pub fn format_delta(&self) -> String {
        let qug = self.delta_qug as f64 / 1e24;
        let abs = qug.abs();
        let sign = if qug >= 0.0 { "+" } else { "-" };
        if abs >= 1e9 {
            format!("{}{:.2}B QUG", sign, abs / 1e9)
        } else if abs >= 1e6 {
            format!("{}{:.2}M QUG", sign, abs / 1e6)
        } else if abs >= 1e3 {
            format!("{}{:.2}K QUG", sign, abs / 1e3)
        } else {
            format!("{}{:.0} QUG", sign, abs)
        }
    }

    /// Format the 4-byte address prefix as 8 hex chars (lowercase).
    /// Hand-rolled to avoid adding a `hex` dependency to q-tui.
    pub fn format_addr(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(8);
        for b in &self.addr_prefix {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
        out
    }
}

/// Network throttle mode — controls P2P aggressiveness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkThrottleMode {
    Conservative, // Low bandwidth: reduce sync concurrency, smaller mesh
    Normal,       // Default settings
    Turbo,        // Max bandwidth: aggressive sync, full mesh
}

impl NetworkThrottleMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            NetworkThrottleMode::Conservative => "Conservative",
            NetworkThrottleMode::Normal => "Normal",
            NetworkThrottleMode::Turbo => "Turbo",
        }
    }

    /// Cycle to next mode
    pub fn next(self) -> Self {
        match self {
            NetworkThrottleMode::Conservative => NetworkThrottleMode::Normal,
            NetworkThrottleMode::Normal => NetworkThrottleMode::Turbo,
            NetworkThrottleMode::Turbo => NetworkThrottleMode::Conservative,
        }
    }
}

impl Default for NetworkThrottleMode {
    fn default() -> Self {
        NetworkThrottleMode::Turbo // v8.5.4: Default to Turbo for max sync speed
    }
}

impl NetworkThrottleMode {
    /// Convert to AtomicU8 value: 0=Conservative, 1=Normal, 2=Turbo
    pub fn to_u8(self) -> u8 {
        match self {
            NetworkThrottleMode::Conservative => 0,
            NetworkThrottleMode::Normal => 1,
            NetworkThrottleMode::Turbo => 2,
        }
    }

    /// Convert from AtomicU8 value
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => NetworkThrottleMode::Conservative,
            1 => NetworkThrottleMode::Normal,
            _ => NetworkThrottleMode::Turbo,
        }
    }
}

// v10.9.19 — Track A: Node readiness mode for the TUI top banner.
/// Each mode corresponds to a specific phase of node bootstrapping and
/// archive construction. The TUI renders a colored banner accordingly.
/// `FastReady` is the post-SNARK-bootstrap state and lights up when
/// `verified_proof_height` becomes Some (currently unwired; future work).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadinessMode {
    Bootstrapping,    // grey
    FastReady,        // green — proof-bootstrap verified
    CheckpointTrust,  // yellow — BAL-001 snapshot accepted
    GenesisSync,      // cyan — sync from height 1
    ArchiveComplete,  // bright green — full history available
}

impl Default for ReadinessMode {
    fn default() -> Self {
        Self::Bootstrapping
    }
}

/// Node metrics for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    // Network metrics
    pub peer_count: usize,
    pub inbound_peers: usize,
    pub outbound_peers: usize,
    pub tor_circuits: usize,
    pub bytes_received: u64,
    pub bytes_sent: u64,

    // Bandwidth analysis
    pub bytes_in_per_sec: u64,
    pub bytes_out_per_sec: u64,
    pub total_bytes_in: u64,
    pub total_bytes_out: u64,
    pub network_throttle_mode: NetworkThrottleMode,

    // Blockchain metrics
    pub block_height: u64,
    pub dag_size_mb: f64,
    pub last_block_secs: u64,
    pub anchor_count: u64,
    pub vertex_count: u64,

    // Performance metrics
    pub current_tps: usize,
    pub latency_p50_ms: u64,
    pub latency_p99_ms: u64,
    pub cpu_usage_percent: f32,
    pub ram_usage_gb: f32,
    pub ram_total_gb: f32,
    pub disk_usage_gb: f64,
    pub disk_total_gb: f64,

    // Uptime
    pub uptime_secs: u64,

    // Mining (if enabled)
    pub mining_enabled: bool,
    pub hashrate: f64,
    pub blocks_mined: u64,
    pub active_miners: usize,
    pub last_block_timestamp: u64,

    // Sync status
    pub is_syncing: bool,
    pub sync_progress_percent: f32,
    pub sync_current_height: u64,
    pub sync_target_height: u64,
    pub sync_speed_blocks_per_sec: f32,

    // Node identity
    pub network_id: String,
    pub version: String,
    pub network_height: u64,
    pub total_supply: f64,
    pub emission_rate: f64,

    // APOLLO sync optimization metrics
    pub apollo_sync_mode: u8,             // 0=idle, 1=turbo, 2=endgame, 3=micro
    pub apollo_chunks_completed: u64,
    pub apollo_chunks_total: u64,
    pub apollo_in_flight: u64,
    pub apollo_queued: u64,
    // Kalman network predictor
    pub apollo_kalman_bandwidth_mbps: f64,
    pub apollo_kalman_latency_ms: f64,
    pub apollo_kalman_confidence: f64,
    pub apollo_kalman_optimal_chunk_kb: u64,
    pub apollo_kalman_loss_pct: f64,      // Predicted loss %
    pub apollo_kalman_timeout_ms: u64,    // Optimal timeout
    pub apollo_kalman_concurrency: usize, // Optimal parallel streams
    // PID rate controller
    pub apollo_pid_target_bps: f64,
    pub apollo_pid_current_bps: f64,
    pub apollo_pid_error: f64,            // Average error
    // Gravity-assist peer momentum
    pub apollo_peers_tracked: usize,
    pub apollo_gravity_best_peer: String, // Short peer ID of best peer
    pub apollo_gravity_best_heat: f64,    // Best peer's cache heat

    // Starship Flight Computer telemetry
    pub starship_phase: String,                  // "PRELAUNCH" / "SUPER_HEAVY" / "STATION_KEEPING" etc.
    pub starship_phase_duration_secs: u64,
    pub starship_orbit_stable: bool,
    pub starship_mission_elapsed_secs: u64,
    pub starship_peer_health: f64,               // 0.0-1.0
    // v8.7.0: Enhanced telemetry
    pub starship_blocks_in_phase: u64,
    pub starship_phase_bps: f64,                 // blocks/sec in current phase
    pub starship_total_synced: u64,              // total blocks synced this mission
    pub starship_mission_avg_bps: f64,           // mission-wide average bps
    pub starship_orbit_decays: u32,              // orbit perturbation count
    pub starship_recommended_throttle: String,   // "turbo" / "normal" / "conservative"

    // Distributed AI metrics
    pub ai_enabled: bool,
    pub ai_nodes_available: usize,
    pub ai_total_requests: u64,
    pub ai_nodes_participated: u64,
    pub ai_avg_nodes_per_request: f64,
    pub ai_layers_processed: u64,
    pub ai_active_requests: usize,

    // Physics Dashboard — Theoretical Consensus Metrics
    // Consensus Hamiltonian: H_dag = H_parent + H_anticone + H_blue + H_vdf
    pub physics_h_total: f64,
    pub physics_h_parent: f64,
    pub physics_h_anticone: f64,
    pub physics_h_blue: f64,
    pub physics_h_vdf: f64,
    // Phase Transition (K-parameter)
    pub physics_kappa: f64,
    pub physics_kappa_c: f64,
    pub physics_phase: String,  // "ordered" / "disordered"
    pub physics_phase_margin: f64,
    pub physics_order_param: f64,
    // Effective Temperature
    pub physics_t_eff: f64,
    // Gossip Diffusion
    pub physics_diffusion_d: f64,
    pub physics_tau_gossip_ms: f64,
    pub physics_mesh_degree: f64,
    pub physics_info_density_200ms: f64,
    pub physics_info_density_1s: f64,
    // Convergence
    pub physics_spectral_gap: f64,
    pub physics_convergence_time_s: f64,
    // Thermodynamics: F = <E> - T_eff * S
    pub physics_free_energy: f64,
    pub physics_entropy: f64,
    // Security bounds (bits)
    pub physics_sig_forgery_bits: f64,
    pub physics_key_recovery_bits: f64,
    pub physics_dag_manipulation_bits: f64,
    // Dandelion++ Privacy
    pub physics_stem_length: f64,
    pub physics_p_deanon: f64,
    // Network parameters
    pub physics_block_rate: f64,
    pub physics_byzantine_fraction: f64,

    // v9.1.0: Compute Power Layer metrics
    pub compute_network_hashrate_hs: f64,   // Total network hashrate (H/s) from P2P announcements
    pub compute_connected_peers: u32,       // Number of peers reporting hashrate
    pub compute_live_security_bits: f64,    // Live security bits (boosted by real-time hashpower)
    pub compute_local_hashrate_hs: f64,     // This node's hashrate contribution
    pub compute_simd_tier: String,          // AVX-512 / AVX2 / SSE2 / NEON / Scalar

    // v9.3.2: K-Parameter Network Health Gauge
    pub kparam_k_value: f64,            // Current K value (0 = healthy, >10 = critical)
    pub kparam_phase: String,           // "stable" / "approaching" / "critical"
    pub kparam_max_solutions: u64,      // Tuned max_solutions_per_block
    pub kparam_vdf_multiplier: f64,     // Tuned VDF difficulty multiplier
    pub kparam_challenge_expiry: u64,   // Tuned challenge expiry (seconds)
    pub kparam_rounds: u64,             // Computation rounds since startup

    // Node operator wallet & fee metrics (v8.6.1)
    pub admin_wallet_address: String,
    pub admin_wallet_balance: f64,
    pub operator_fee_promille: u64,
    pub operator_fee_session_qug: f64,
    pub operator_fee_total_qug: f64,
    pub operator_fee_tx_count: u64,
    pub founder_wallet_balance: f64,

    // v10.9.19 — Engine pulse metrics (sourced from AppState atomics every update tick)
    /// Total API requests served since process start. Compute req/sec by diffing
    /// against previous tick's value.
    pub engine_api_requests_total: u64,
    /// Total P2P bytes in / out since process start.
    pub engine_p2p_bytes_in: u64,
    pub engine_p2p_bytes_out: u64,
    /// True iff data integrity is currently consistent (checkpoint applied OR
    /// contiguous height ≥ tip). Reflects whether local state-root computation
    /// is trustworthy for queries.
    pub engine_data_integrity_ok: bool,
    /// Quorum: current connected peer count, used for BFT-style health display.
    pub engine_quorum_peers: usize,

    // v10.9.19 — Track A: fast-readiness banner state
    pub readiness_mode: ReadinessMode,
    /// Wall-clock when `readiness_mode` last changed. Used for a 300ms flash
    /// animation on transition. `#[serde(skip)]` because Instant isn't serialisable
    /// and the timestamp is purely render-side.
    #[serde(skip)]
    pub readiness_changed_at: Option<std::time::Instant>,
    /// Lowest block height with contiguous archive coverage. Sourced from
    /// `AppState.contiguous_height_atomic`.
    pub archive_lowest_indexed_height: u64,
    /// Network tip height. Sourced from `AppState.current_height_atomic`.
    pub archive_tip_height: u64,
    /// True iff full chain history from genesis to tip is locally indexed.
    pub archive_complete: bool,

    /// 🔥 Top Movers (last 60 blocks): up to 5 wallets ranked by |Δ balance|.
    /// Populated by `update_tui_metrics` from `AppState.recent_balance_deltas`.
    /// Empty until the ring buffer has ingested at least one block.
    pub top_movers: Vec<TopMover>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            peer_count: 0,
            inbound_peers: 0,
            outbound_peers: 0,
            tor_circuits: 0,
            bytes_received: 0,
            bytes_sent: 0,
            bytes_in_per_sec: 0,
            bytes_out_per_sec: 0,
            total_bytes_in: 0,
            total_bytes_out: 0,
            network_throttle_mode: NetworkThrottleMode::Normal,
            block_height: 0,
            dag_size_mb: 0.0,
            last_block_secs: 0,
            anchor_count: 0,
            vertex_count: 0,
            current_tps: 0,
            latency_p50_ms: 0,
            latency_p99_ms: 0,
            cpu_usage_percent: 0.0,
            ram_usage_gb: 0.0,
            ram_total_gb: 8.0,
            disk_usage_gb: 0.0,
            disk_total_gb: 500.0,
            uptime_secs: 0,
            mining_enabled: false,
            hashrate: 0.0,
            blocks_mined: 0,
            active_miners: 0,
            last_block_timestamp: 0,
            is_syncing: false,
            sync_progress_percent: 0.0,
            sync_current_height: 0,
            sync_target_height: 0,
            sync_speed_blocks_per_sec: 0.0,
            network_id: String::new(),
            version: String::new(),
            network_height: 0,
            total_supply: 0.0,
            emission_rate: 0.0,
            // APOLLO sync optimization defaults
            apollo_sync_mode: 0,
            apollo_chunks_completed: 0,
            apollo_chunks_total: 0,
            apollo_in_flight: 0,
            apollo_queued: 0,
            apollo_kalman_bandwidth_mbps: 0.0,
            apollo_kalman_latency_ms: 0.0,
            apollo_kalman_confidence: 0.0,
            apollo_kalman_optimal_chunk_kb: 0,
            apollo_kalman_loss_pct: 0.0,
            apollo_kalman_timeout_ms: 0,
            apollo_kalman_concurrency: 0,
            apollo_pid_target_bps: 0.0,
            apollo_pid_current_bps: 0.0,
            apollo_pid_error: 0.0,
            apollo_peers_tracked: 0,
            apollo_gravity_best_peer: String::new(),
            apollo_gravity_best_heat: 0.0,
            // Starship Flight Computer defaults
            starship_phase: "PRELAUNCH".to_string(),
            starship_phase_duration_secs: 0,
            starship_orbit_stable: false,
            starship_mission_elapsed_secs: 0,
            starship_peer_health: 0.0,
            starship_blocks_in_phase: 0,
            starship_phase_bps: 0.0,
            starship_total_synced: 0,
            starship_mission_avg_bps: 0.0,
            starship_orbit_decays: 0,
            starship_recommended_throttle: "normal".to_string(),
            ai_enabled: false,
            ai_nodes_available: 0,
            ai_total_requests: 0,
            ai_nodes_participated: 0,
            ai_avg_nodes_per_request: 0.0,
            ai_layers_processed: 0,
            ai_active_requests: 0,
            // Physics defaults
            physics_h_total: 0.0,
            physics_h_parent: 0.0,
            physics_h_anticone: 0.0,
            physics_h_blue: 0.0,
            physics_h_vdf: 0.0,
            physics_kappa: 18.0,
            physics_kappa_c: 0.0,
            physics_phase: "ordered".to_string(),
            physics_phase_margin: 0.0,
            physics_order_param: 1.0,
            physics_t_eff: 0.0,
            physics_diffusion_d: 0.0,
            physics_tau_gossip_ms: 0.0,
            physics_mesh_degree: 8.0,
            physics_info_density_200ms: 0.0,
            physics_info_density_1s: 0.0,
            physics_spectral_gap: 0.0,
            physics_convergence_time_s: 0.0,
            physics_free_energy: 0.0,
            physics_entropy: 0.0,
            physics_sig_forgery_bits: 256.0,
            physics_key_recovery_bits: 200.0,
            physics_dag_manipulation_bits: 0.0,
            physics_stem_length: 4.0,
            physics_p_deanon: 0.0,
            physics_block_rate: 0.0,
            physics_byzantine_fraction: 0.0,
            // Compute Power Layer defaults
            compute_network_hashrate_hs: 0.0,
            compute_connected_peers: 0,
            compute_live_security_bits: 0.0,
            compute_local_hashrate_hs: 0.0,
            compute_simd_tier: String::new(),
            // K-Parameter Network Health Gauge defaults
            kparam_k_value: 0.0,
            kparam_phase: "stable".to_string(),
            kparam_max_solutions: 0,
            kparam_vdf_multiplier: 1.0,
            kparam_challenge_expiry: 0,
            kparam_rounds: 0,
            // Operator wallet defaults
            admin_wallet_address: String::new(),
            admin_wallet_balance: 0.0,
            operator_fee_promille: 0,
            operator_fee_session_qug: 0.0,
            operator_fee_total_qug: 0.0,
            operator_fee_tx_count: 0,
            founder_wallet_balance: 0.0,
            // v10.9.19 — Engine pulse defaults
            engine_api_requests_total: 0,
            engine_p2p_bytes_in: 0,
            engine_p2p_bytes_out: 0,
            engine_data_integrity_ok: false,
            engine_quorum_peers: 0,
            // v10.9.19 — Track A defaults
            readiness_mode: ReadinessMode::default(),
            readiness_changed_at: None,
            archive_lowest_indexed_height: 0,
            archive_tip_height: 0,
            archive_complete: false,
            // 🔥 Top Movers defaults — empty until first block ingestion
            top_movers: Vec::new(),
        }
    }
}

impl Metrics {
    /// Format bytes as human-readable string
    pub fn format_bytes(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Format bytes per second as human-readable rate
    pub fn format_bytes_rate(bytes_per_sec: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if bytes_per_sec >= GB {
            format!("{:.1} GB/s", bytes_per_sec as f64 / GB as f64)
        } else if bytes_per_sec >= MB {
            format!("{:.1} MB/s", bytes_per_sec as f64 / MB as f64)
        } else if bytes_per_sec >= KB {
            format!("{:.1} KB/s", bytes_per_sec as f64 / KB as f64)
        } else {
            format!("{} B/s", bytes_per_sec)
        }
    }

    /// Format uptime as human-readable string
    pub fn format_uptime(secs: u64) -> String {
        let days = secs / 86400;
        let hours = (secs % 86400) / 3600;
        let mins = (secs % 3600) / 60;

        if days > 0 {
            format!("{}d {}h {}m", days, hours, mins)
        } else if hours > 0 {
            format!("{}h {}m", hours, mins)
        } else {
            format!("{}m", mins)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1e24 doesn't fit in i128? It does: 1e24 ≈ 10^24, i128::MAX ≈ 1.7e38.
    const QUG_SCALE: i128 = 1_000_000_000_000_000_000_000_000; // 10^24

    fn mover(delta_qug: i128) -> TopMover {
        TopMover {
            addr_prefix: [0xa3, 0xb4, 0xc8, 0xd9],
            delta_qug,
            direction: if delta_qug > 0 {
                MoverDirection::Up
            } else if delta_qug < 0 {
                MoverDirection::Down
            } else {
                MoverDirection::Flat
            },
        }
    }

    #[test]
    fn test_format_delta_billions() {
        // 12.45M (millions of QUG) — task spec example.
        // 12_450_000 QUG * 1e24 = 1.245e31 raw units
        let m = mover(12_450_000_i128.saturating_mul(QUG_SCALE));
        let s = m.format_delta();
        assert!(
            s.starts_with("+12.45M"),
            "expected leading +12.45M, got {}",
            s
        );
    }

    #[test]
    fn test_format_delta_negative() {
        let m = mover(-3_100_000_i128.saturating_mul(QUG_SCALE));
        let s = m.format_delta();
        assert!(s.starts_with("-"), "expected negative sign, got {}", s);
        assert!(s.contains("3.10M"), "expected 3.10M magnitude, got {}", s);
    }

    #[test]
    fn test_format_delta_small() {
        let m = mover(500_i128.saturating_mul(QUG_SCALE));
        let s = m.format_delta();
        assert!(s.starts_with("+500"), "expected +500, got {}", s);
        // 500 is below the 1e3 threshold so no K/M/B suffix.
        assert!(!s.contains("K"), "should not have K suffix, got {}", s);
        assert!(!s.contains("M"), "should not have M suffix, got {}", s);
    }

    #[test]
    fn test_format_delta_zero() {
        let m = mover(0);
        let s = m.format_delta();
        // Zero rendered with '+' sign — defensive case (shouldn't appear in top 5).
        assert!(s.starts_with("+0"), "expected +0 for zero delta, got {}", s);
    }

    #[test]
    fn test_format_delta_billions_huge() {
        // 12.45B QUG = 12_450_000_000 QUG
        let m = mover(12_450_000_000_i128.saturating_mul(QUG_SCALE));
        let s = m.format_delta();
        assert!(
            s.starts_with("+12.45B"),
            "expected leading +12.45B, got {}",
            s
        );
    }

    #[test]
    fn test_format_addr() {
        let m = mover(0);
        // addr_prefix is [0xa3, 0xb4, 0xc8, 0xd9] per the helper above.
        assert_eq!(m.format_addr(), "a3b4c8d9");
    }

    #[test]
    fn test_format_addr_zero_bytes() {
        let m = TopMover {
            addr_prefix: [0x00, 0x00, 0x00, 0x00],
            delta_qug: 1,
            direction: MoverDirection::Up,
        };
        assert_eq!(m.format_addr(), "00000000");
    }

    #[test]
    fn test_format_addr_all_ff() {
        let m = TopMover {
            addr_prefix: [0xff, 0xff, 0xff, 0xff],
            delta_qug: 1,
            direction: MoverDirection::Up,
        };
        assert_eq!(m.format_addr(), "ffffffff");
    }
}
