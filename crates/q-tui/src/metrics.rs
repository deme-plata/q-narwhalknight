use serde::{Deserialize, Serialize};

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
