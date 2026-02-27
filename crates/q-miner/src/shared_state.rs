// ═══════════════════════════════════════════════════════════════════
// SharedMinerState: Thread-safe shared state for TUI + mining threads
// ═══════════════════════════════════════════════════════════════════

use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
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

    // Sync events
    ServerSyncing { blocks_behind: u64 },
    ServerSyncComplete,

    // Block events
    NewBlockSignal { block_height: u64 },
    MiningReward { reward_qnk: f64, block_height: u64 },
    BalanceUpdated { new_balance: f64 },

    // Throttle
    ThrottleChanged { mode: MinerThrottleMode },
}

/// Network throttle mode — cycles with `T` key
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinerThrottleMode {
    Off,
    Light,
    Heavy,
}

impl MinerThrottleMode {
    pub fn next(self) -> Self {
        match self {
            MinerThrottleMode::Off => MinerThrottleMode::Light,
            MinerThrottleMode::Light => MinerThrottleMode::Heavy,
            MinerThrottleMode::Heavy => MinerThrottleMode::Off,
        }
    }

    pub fn delay_ms(self) -> u64 {
        match self {
            MinerThrottleMode::Off => 0,
            MinerThrottleMode::Light => 100,
            MinerThrottleMode::Heavy => 500,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MinerThrottleMode::Off => "Off",
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

    // Config strings (read-only after init)
    pub server_url: String,
    pub wallet_address: String,
    pub miner_id: String,
    pub miner_name: Option<String>,
    pub mining_mode: String,
    pub num_threads: usize,
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
            server_url,
            wallet_address,
            miner_id,
            miner_name,
            mining_mode,
            num_threads,
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
