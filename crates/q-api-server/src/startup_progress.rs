//! Startup Progress Tracker
//!
//! v1.4.15-beta: Provides real-time startup progress for the frontend UI.
//! Shows precise progress during DAG integrity check and initialization phases.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Startup phase enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupPhase {
    /// Initial startup
    Initializing,
    /// Loading configuration
    LoadingConfig,
    /// Opening database
    OpeningDatabase,
    /// Checking DAG integrity (can be slow)
    CheckingDagIntegrity,
    /// Repairing database if needed
    RepairingDatabase,
    /// Initializing network
    InitializingNetwork,
    /// Initializing P2P
    InitializingP2P,
    /// Syncing with network
    SyncingWithNetwork,
    /// Ready to serve
    Ready,
}

impl std::fmt::Display for StartupPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartupPhase::Initializing => write!(f, "Initializing..."),
            StartupPhase::LoadingConfig => write!(f, "Loading configuration..."),
            StartupPhase::OpeningDatabase => write!(f, "Opening database..."),
            StartupPhase::CheckingDagIntegrity => write!(f, "Checking DAG integrity..."),
            StartupPhase::RepairingDatabase => write!(f, "Repairing database..."),
            StartupPhase::InitializingNetwork => write!(f, "Initializing network..."),
            StartupPhase::InitializingP2P => write!(f, "Initializing P2P..."),
            StartupPhase::SyncingWithNetwork => write!(f, "Syncing with network..."),
            StartupPhase::Ready => write!(f, "Ready"),
        }
    }
}

/// Global startup progress tracker
#[derive(Debug)]
pub struct StartupProgress {
    /// Current phase
    phase: RwLock<StartupPhase>,
    /// Current phase message (detailed)
    message: RwLock<String>,
    /// Progress within current phase (0-100)
    phase_progress: AtomicU64,
    /// Total blocks being checked
    total_blocks: AtomicU64,
    /// Blocks checked so far
    blocks_checked: AtomicU64,
    /// Whether startup is complete
    is_ready: AtomicBool,
    /// Start time for elapsed calculation
    start_time: Instant,
    /// Current height (for display)
    current_height: AtomicU64,
    /// Network height (target)
    network_height: AtomicU64,
}

impl Default for StartupProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl StartupProgress {
    pub fn new() -> Self {
        Self {
            phase: RwLock::new(StartupPhase::Initializing),
            message: RwLock::new("Starting up...".to_string()),
            phase_progress: AtomicU64::new(0),
            total_blocks: AtomicU64::new(0),
            blocks_checked: AtomicU64::new(0),
            is_ready: AtomicBool::new(false),
            start_time: Instant::now(),
            current_height: AtomicU64::new(0),
            network_height: AtomicU64::new(0),
        }
    }

    /// Set the current phase
    pub async fn set_phase(&self, phase: StartupPhase, message: &str) {
        *self.phase.write().await = phase;
        *self.message.write().await = message.to_string();
        self.phase_progress.store(0, Ordering::SeqCst);

        if phase == StartupPhase::Ready {
            self.is_ready.store(true, Ordering::SeqCst);
            self.phase_progress.store(100, Ordering::SeqCst);
        }
    }

    /// Set the current phase (sync version for non-async contexts)
    pub fn set_phase_sync(&self, phase: StartupPhase, message: &str) {
        // Use try_write to avoid blocking
        if let Ok(mut guard) = self.phase.try_write() {
            *guard = phase;
        }
        if let Ok(mut guard) = self.message.try_write() {
            *guard = message.to_string();
        }
        self.phase_progress.store(0, Ordering::SeqCst);

        if phase == StartupPhase::Ready {
            self.is_ready.store(true, Ordering::SeqCst);
            self.phase_progress.store(100, Ordering::SeqCst);
        }
    }

    /// Update progress for DAG integrity check
    pub fn update_integrity_check(&self, checked: u64, total: u64) {
        self.blocks_checked.store(checked, Ordering::SeqCst);
        self.total_blocks.store(total, Ordering::SeqCst);

        if total > 0 {
            let progress = (checked * 100) / total;
            self.phase_progress.store(progress.min(100), Ordering::SeqCst);
        }
    }

    /// Update sync progress
    pub fn update_sync_progress(&self, current: u64, network: u64) {
        self.current_height.store(current, Ordering::SeqCst);
        self.network_height.store(network, Ordering::SeqCst);

        if network > 0 && current <= network {
            let progress = (current * 100) / network;
            self.phase_progress.store(progress.min(100), Ordering::SeqCst);
        }
    }

    /// Mark as ready
    pub async fn set_ready(&self) {
        self.set_phase(StartupPhase::Ready, "Node is ready").await;
    }

    /// Get current status as JSON-serializable struct
    pub async fn get_status(&self) -> StartupStatus {
        let phase = *self.phase.read().await;
        let message = self.message.read().await.clone();

        StartupStatus {
            phase,
            message,
            phase_progress: self.phase_progress.load(Ordering::SeqCst),
            total_blocks: self.total_blocks.load(Ordering::SeqCst),
            blocks_checked: self.blocks_checked.load(Ordering::SeqCst),
            is_ready: self.is_ready.load(Ordering::SeqCst),
            elapsed_seconds: self.start_time.elapsed().as_secs(),
            current_height: self.current_height.load(Ordering::SeqCst),
            network_height: self.network_height.load(Ordering::SeqCst),
        }
    }

    /// Check if startup is complete
    pub fn is_ready(&self) -> bool {
        self.is_ready.load(Ordering::SeqCst)
    }
}

/// Serializable startup status for API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupStatus {
    pub phase: StartupPhase,
    pub message: String,
    pub phase_progress: u64,
    pub total_blocks: u64,
    pub blocks_checked: u64,
    pub is_ready: bool,
    pub elapsed_seconds: u64,
    pub current_height: u64,
    pub network_height: u64,
}

/// Global singleton for startup progress
static STARTUP_PROGRESS: std::sync::OnceLock<Arc<StartupProgress>> = std::sync::OnceLock::new();

/// Get or initialize the global startup progress tracker
pub fn get_startup_progress() -> Arc<StartupProgress> {
    STARTUP_PROGRESS.get_or_init(|| Arc::new(StartupProgress::new())).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_startup_progress() {
        let progress = StartupProgress::new();

        // Initial state
        let status = progress.get_status().await;
        assert_eq!(status.phase, StartupPhase::Initializing);
        assert!(!status.is_ready);

        // Update phase
        progress.set_phase(StartupPhase::CheckingDagIntegrity, "Checking blocks...").await;
        progress.update_integrity_check(500, 1000);

        let status = progress.get_status().await;
        assert_eq!(status.phase, StartupPhase::CheckingDagIntegrity);
        assert_eq!(status.phase_progress, 50);
        assert_eq!(status.blocks_checked, 500);
        assert_eq!(status.total_blocks, 1000);

        // Mark ready
        progress.set_ready().await;
        let status = progress.get_status().await;
        assert!(status.is_ready);
        assert_eq!(status.phase, StartupPhase::Ready);
    }
}
