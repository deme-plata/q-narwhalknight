// ═══════════════════════════════════════════════════════════════════
// SharedMinerState: Thread-safe shared state for TUI + mining threads
// ═══════════════════════════════════════════════════════════════════

use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// Per-thread status — uses RwLock because enum with data can't be atomic,
/// but status changes are rare (~1/sec vs millions of hashes/sec).
#[derive(Debug, Clone)]
pub enum ThreadStatus {
    Starting,
    FetchingChallenge,
    WaitingForSync { blocks_behind: u64 },
    Mining { block_height: u64 },
    Error { message: String, since: Instant },
    Stopped,
}

impl ThreadStatus {
    pub fn label(&self) -> &'static str {
        match self {
            ThreadStatus::Starting => "starting",
            ThreadStatus::FetchingChallenge => "fetching",
            ThreadStatus::WaitingForSync { .. } => "syncing",
            ThreadStatus::Mining { .. } => "mining",
            ThreadStatus::Error { .. } => "error",
            ThreadStatus::Stopped => "stopped",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, ThreadStatus::Mining { .. })
    }

    pub fn is_error(&self) -> bool {
        matches!(self, ThreadStatus::Error { .. })
    }
}

/// Per-thread state container
pub struct ThreadState {
    pub status: RwLock<ThreadStatus>,
    pub hashes_this_thread: AtomicU64,
    pub solutions_found: AtomicU64,
    pub challenge_fetch_latency_us: AtomicU64,
}

impl ThreadState {
    pub fn new() -> Self {
        Self {
            status: RwLock::new(ThreadStatus::Starting),
            hashes_this_thread: AtomicU64::new(0),
            solutions_found: AtomicU64::new(0),
            challenge_fetch_latency_us: AtomicU64::new(0),
        }
    }

    pub fn set_status(&self, s: ThreadStatus) {
        *self.status.write() = s;
    }

    pub fn get_status(&self) -> ThreadStatus {
        self.status.read().clone()
    }
}

/// Diagnostic events sent from mining threads to TUI
#[derive(Debug, Clone)]
pub enum DiagnosticEvent {
    // Thread lifecycle
    ThreadStarted { thread_id: usize },
    ThreadStopped { thread_id: usize },
    ThreadError { thread_id: usize, message: String },

    // Challenge events
    ChallengeFetched { thread_id: usize, block_height: u64, latency_ms: u64 },
    ChallengeFetchFailed { thread_id: usize, error: String },

    // Solution events
    SolutionFound { thread_id: usize, block_height: u64, nonce: u64 },
    SolutionAccepted { block_height: u64, reward_qnk: f64 },
    SolutionRejected { block_height: u64, reason: String },

    // Connection events
    SseConnected { url: String },
    SseDisconnected { error: String },
    MinerLinkConnected,
    MinerLinkDisconnected,

    // Server notices
    ServerNotice { message: String },
    UpdateAvailable { min_miner_version: String },

    // Sync events — v9.0.4: Enhanced with Starship telemetry
    ServerSyncing { sync_info: StarshipSyncInfo },
    ServerSyncComplete,

    // Block events
    NewBlockSignal { block_height: u64 },
    MiningReward { reward_qnk: f64, block_height: u64 },
    BalanceUpdated { new_balance: f64 },

    // Throttle
    ThrottleChanged { mode: MinerThrottleMode },

    // v9.1.0: Compute Power Layer updates (from challenge response)
    ComputePowerUpdate {
        network_hashrate_hs: f64,
        connected_miners: u32,
        live_security_bits: f64,
    },

    // v9.1.4: Mining mode switch from admin
    MiningModeSwitch {
        target_mode: String,
        pool_url: Option<String>,
        reason: Option<String>,
    },

    // v9.9.0: Auto-update events
    UpdateDownloading { version: String, bytes_downloaded: u64, bytes_total: u64 },
    UpdateReadyToApply { version: String },
    UpdateApplying { version: String },
    UpdateError { version: String, message: String },

    // v10.2.0: Hybrid Quantum Mining — GPU events
    GpuStarted { device_name: String },
    GpuStopped,
    GpuSolutionFound { nonce: u64, block_height: u64 },
    GpuError { message: String },
}

/// v9.0.4: Starship sync telemetry — rich sync progress for TUI
#[derive(Debug, Clone)]
pub struct StarshipSyncInfo {
    pub blocks_behind: u64,
    pub local_height: u64,
    pub network_height: u64,
    pub sync_progress: f32,         // 0.0 - 100.0
    pub sync_speed_bps: f32,        // blocks per second
    pub phase: String,              // Prelaunch, SuperHeavy, HotStaging, StarshipCruise, StationKeeping
    pub phase_duration_secs: u64,
    pub mission_elapsed_secs: u64,
    pub peer_count: u64,
    pub orbit_stable: bool,
    pub eta_secs: u64,              // estimated time to completion
}

impl Default for StarshipSyncInfo {
    fn default() -> Self {
        Self {
            blocks_behind: 0,
            local_height: 0,
            network_height: 0,
            sync_progress: 0.0,
            sync_speed_bps: 0.0,
            phase: "Unknown".to_string(),
            phase_duration_secs: 0,
            mission_elapsed_secs: 0,
            peer_count: 0,
            orbit_stable: false,
            eta_secs: 0,
        }
    }
}

/// Network throttle mode — cycles with `T` key
/// v8.8.3: Added UltraLight mode with LZ4 compression for <10 KB/s on 256 threads
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinerThrottleMode {
    Off,
    /// LZ4-compressed requests + extended refresh intervals. Target: <10 KB/s total.
    UltraLight,
    Light,
    Heavy,
}

impl MinerThrottleMode {
    pub fn next(self) -> Self {
        match self {
            MinerThrottleMode::Off => MinerThrottleMode::UltraLight,
            MinerThrottleMode::UltraLight => MinerThrottleMode::Light,
            MinerThrottleMode::Light => MinerThrottleMode::Heavy,
            MinerThrottleMode::Heavy => MinerThrottleMode::Off,
        }
    }

    pub fn delay_ms(self) -> u64 {
        match self {
            MinerThrottleMode::Off => 0,
            MinerThrottleMode::UltraLight => 0, // no hash-loop delay, bandwidth saved via compression
            MinerThrottleMode::Light => 100,
            MinerThrottleMode::Heavy => 500,
        }
    }

    /// Challenge refresh interval for thread 0 in this throttle mode.
    /// Higher = fewer API calls = less bandwidth.
    pub fn challenge_refresh_secs(self) -> u64 {
        match self {
            MinerThrottleMode::Off => 50,
            MinerThrottleMode::UltraLight => 120,  // 2 min between fetches
            MinerThrottleMode::Light => 50,
            MinerThrottleMode::Heavy => 50,
        }
    }

    /// Whether to use LZ4 compression on API payloads.
    pub fn use_compression(self) -> bool {
        matches!(self, MinerThrottleMode::UltraLight)
    }

    pub fn label(self) -> &'static str {
        match self {
            MinerThrottleMode::Off => "Off",
            MinerThrottleMode::UltraLight => "UltraLight (LZ4+gzip)",
            MinerThrottleMode::Light => "Light (100ms)",
            MinerThrottleMode::Heavy => "Heavy (500ms)",
        }
    }
}

/// Central shared state between mining threads and TUI
pub struct SharedMinerState {
    // Existing atomics (wrapped from main.rs)
    pub hash_counter: Arc<AtomicU64>,
    pub is_running: Arc<AtomicBool>,
    pub new_block_signal: Arc<AtomicU64>,
    pub current_hashrate_khs: Arc<AtomicU64>,
    pub is_paused: Arc<AtomicBool>,
    pub target_threads: Arc<AtomicUsize>,
    pub target_intensity: Arc<AtomicU8>,
    pub solutions_found: Arc<AtomicU64>,
    pub blocks_mined: Arc<AtomicU64>,

    // New TUI fields
    pub thread_states: Vec<Arc<ThreadState>>,
    pub event_tx: mpsc::UnboundedSender<DiagnosticEvent>,
    pub throttle_mode: Arc<RwLock<MinerThrottleMode>>,
    pub start_time: Instant,

    // Connection status atomics
    pub sse_connected: Arc<AtomicBool>,
    pub miner_link_connected: Arc<AtomicBool>,
    pub using_fallback: Arc<AtomicBool>,
    pub last_challenge_latency_us: Arc<AtomicU64>,

    // v8.6.6: Bandwidth tracking (bytes, atomic for lock-free updates)
    pub bytes_downloaded: Arc<AtomicU64>,  // total bytes received (challenges, SSE, etc.)
    pub bytes_uploaded: Arc<AtomicU64>,    // total bytes sent (solutions, heartbeats)
    pub api_requests_total: Arc<AtomicU64>,   // total API calls made
    pub api_requests_failed: Arc<AtomicU64>,  // failed API calls

    // v9.2.6: "Mercedes" balance smoothing — SSE freshness timestamp (epoch secs)
    pub last_balance_sse_epoch: Arc<AtomicU64>,

    // Config strings (read-only after init)
    pub server_url: String,
    pub wallet_address: String,
    pub miner_id: String,
    pub miner_name: Option<String>,
    pub mining_mode: String,
    pub num_threads: usize,
    pub proxy_url: Option<String>,

    // v9.1.4: Dynamic mining mode switch — admin can force mode change at runtime
    // 0 = no switch pending, 1 = switch to solo, 2 = switch to pool
    pub mode_switch_signal: Arc<AtomicU8>,
    pub mode_switch_pool_url: Arc<parking_lot::RwLock<Option<String>>>,

    // v9.1.7: P2P networking status (gossipsub challenge relay + solution broadcast)
    pub p2p_connected: Arc<AtomicBool>,
    pub p2p_peer_count: Arc<AtomicU32>,
    pub p2p_challenges_received: Arc<AtomicU64>,
    pub p2p_solutions_broadcast: Arc<AtomicU64>,

    // v10.2.0: Hybrid Quantum Mining — GPU state
    pub gpu_active: Arc<AtomicBool>,
    pub gpu_hashrate_hs: Arc<AtomicU64>,     // f64 bits stored as u64
    pub gpu_hashes_total: Arc<AtomicU64>,
    pub gpu_device_name: Arc<RwLock<String>>,
    // v10.1.7: Rich GPU device info for TUI display
    pub gpu_devices: Arc<RwLock<Vec<GpuDeviceSnapshot>>>,
}

/// Lightweight snapshot of GPU device info for TUI display (avoids cross-crate deps)
#[derive(Debug, Clone)]
pub struct GpuDeviceSnapshot {
    pub index: usize,
    pub name: String,
    pub vendor: String,
    pub compute_units: u32,
    pub global_memory_mb: u64,
    pub local_memory_kb: u64,
    pub max_clock_mhz: u32,
    pub api: String,  // "OpenCL", "CUDA", "Vulkan"
}

impl SharedMinerState {
    pub fn new(
        hash_counter: Arc<AtomicU64>,
        is_running: Arc<AtomicBool>,
        new_block_signal: Arc<AtomicU64>,
        current_hashrate_khs: Arc<AtomicU64>,
        is_paused: Arc<AtomicBool>,
        target_threads: Arc<AtomicUsize>,
        target_intensity: Arc<AtomicU8>,
        solutions_found: Arc<AtomicU64>,
        blocks_mined: Arc<AtomicU64>,
        num_threads: usize,
        server_url: String,
        wallet_address: String,
        miner_id: String,
        miner_name: Option<String>,
        mining_mode: String,
        proxy_url: Option<String>,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<DiagnosticEvent>) {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let thread_states: Vec<Arc<ThreadState>> = (0..num_threads)
            .map(|_| Arc::new(ThreadState::new()))
            .collect();

        let state = Arc::new(Self {
            hash_counter,
            is_running,
            new_block_signal,
            current_hashrate_khs,
            is_paused,
            target_threads,
            target_intensity,
            solutions_found,
            blocks_mined,
            thread_states,
            event_tx,
            throttle_mode: Arc::new(RwLock::new(MinerThrottleMode::Off)),
            start_time: Instant::now(),
            sse_connected: Arc::new(AtomicBool::new(false)),
            miner_link_connected: Arc::new(AtomicBool::new(false)),
            using_fallback: Arc::new(AtomicBool::new(false)),
            last_challenge_latency_us: Arc::new(AtomicU64::new(0)),
            bytes_downloaded: Arc::new(AtomicU64::new(0)),
            bytes_uploaded: Arc::new(AtomicU64::new(0)),
            api_requests_total: Arc::new(AtomicU64::new(0)),
            api_requests_failed: Arc::new(AtomicU64::new(0)),
            server_url,
            wallet_address,
            miner_id,
            miner_name,
            mining_mode,
            num_threads,
            proxy_url,
            last_balance_sse_epoch: Arc::new(AtomicU64::new(0)),
            mode_switch_signal: Arc::new(AtomicU8::new(0)),
            mode_switch_pool_url: Arc::new(parking_lot::RwLock::new(None)),
            p2p_connected: Arc::new(AtomicBool::new(false)),
            p2p_peer_count: Arc::new(AtomicU32::new(0)),
            p2p_challenges_received: Arc::new(AtomicU64::new(0)),
            p2p_solutions_broadcast: Arc::new(AtomicU64::new(0)),
            gpu_active: Arc::new(AtomicBool::new(false)),
            gpu_hashrate_hs: Arc::new(AtomicU64::new(0)),
            gpu_hashes_total: Arc::new(AtomicU64::new(0)),
            gpu_device_name: Arc::new(RwLock::new(String::new())),
            gpu_devices: Arc::new(RwLock::new(Vec::new())),
        });

        (state, event_rx)
    }

    /// Send a diagnostic event (non-blocking, drops if TUI is gone)
    pub fn send_event(&self, event: DiagnosticEvent) {
        let _ = self.event_tx.send(event);
    }

    pub fn get_hashrate_khs(&self) -> f64 {
        f64::from_bits(self.current_hashrate_khs.load(Ordering::Relaxed))
    }

    pub fn active_thread_count(&self) -> usize {
        self.thread_states
            .iter()
            .filter(|ts| ts.get_status().is_active())
            .count()
    }

    pub fn errored_thread_count(&self) -> usize {
        self.thread_states
            .iter()
            .filter(|ts| ts.get_status().is_error())
            .count()
    }
}
