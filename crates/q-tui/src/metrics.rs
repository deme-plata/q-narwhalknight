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

    // Distributed AI metrics
    pub ai_enabled: bool,
    pub ai_nodes_available: usize,
    pub ai_total_requests: u64,
    pub ai_nodes_participated: u64,
    pub ai_avg_nodes_per_request: f64,
    pub ai_layers_processed: u64,
    pub ai_active_requests: usize,
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
            ai_enabled: false,
            ai_nodes_available: 0,
            ai_total_requests: 0,
            ai_nodes_participated: 0,
            ai_avg_nodes_per_request: 0.0,
            ai_layers_processed: 0,
            ai_active_requests: 0,
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
