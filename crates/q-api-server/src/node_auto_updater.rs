//! v8.5.0: Node Auto-Update System
//!
//! Enables automatic binary updates via P2P gossipsub announcements with
//! Ed25519 quorum verification (2-of-3 trusted bootstrap signers).
//!
//! State machine:
//!   Disabled → Idle → WaitingForQuorum → Downloading → Verifying →
//!   PreflightCheck → ReadyToRestart → RestartScheduled
//!
//! Safety gates:
//! - Minimum connected peers (default 2)
//! - Node must not be syncing (< 10 blocks behind)
//! - Dual-hash verification (SHA-256 + BLAKE3)
//! - Preflight check on new binary before apply
//! - 60-second rollback watchdog after restart
//! - Announcements expire after 24 hours

use ed25519_dalek::VerifyingKey;
use q_types::update_announcement::{
    load_trusted_signers, UpdateAnnouncement, UpdateQuorum, MAX_ANNOUNCEMENT_AGE_SECS,
    MIN_UPDATE_QUORUM,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

/// Current state of the auto-updater
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "state")]
pub enum NodeUpdateState {
    /// Auto-update is disabled (Q_AUTO_UPDATE=0)
    Disabled,
    /// Idle, waiting for announcements
    Idle,
    /// Received announcement(s), waiting for quorum (2/3 signers)
    WaitingForQuorum {
        version: String,
        signers_so_far: usize,
        signers_needed: usize,
    },
    /// Quorum reached, binary available but auto-update off (notification only)
    Available {
        version: String,
        download_url: String,
    },
    /// Downloading the new binary
    Downloading {
        version: String,
        progress_percent: u8,
    },
    /// Verifying checksums
    Verifying {
        version: String,
    },
    /// Running preflight check on new binary
    PreflightCheck {
        version: String,
    },
    /// Ready to restart with new binary
    ReadyToRestart {
        version: String,
    },
    /// Restart has been scheduled (delay countdown)
    RestartScheduled {
        version: String,
        restart_in_secs: u64,
    },
    /// An error occurred (retryable)
    Error {
        version: String,
        message: String,
        retry_count: u32,
    },
    /// Rolling back to previous binary
    RollingBack {
        version: String,
        reason: String,
    },
}

/// Configuration for the auto-updater
#[derive(Debug, Clone)]
pub struct AutoUpdateConfig {
    /// Whether auto-update is enabled
    pub enabled: bool,
    /// Seconds between HTTP polls for version check
    pub check_interval_secs: u64,
    /// Seconds to wait after download before restarting
    pub restart_delay_secs: u64,
    /// Seconds after restart before marking update as healthy
    pub rollback_timeout_secs: u64,
    /// Minimum connected peers required to accept an update
    pub min_peers: usize,
    /// Only auto-apply mandatory (security) updates
    pub mandatory_only: bool,
    /// Database path (for restart/healthy markers)
    pub db_path: PathBuf,
    /// Current binary version
    pub current_version: String,
    /// Network ID (for filtering announcements)
    pub network_id: String,
    /// Bootstrap URLs for HTTP polling fallback
    pub bootstrap_urls: Vec<String>,
}

impl AutoUpdateConfig {
    /// Load configuration from environment variables with defaults
    pub fn from_env(db_path: &str, current_version: &str, network_id: &str) -> Self {
        Self {
            enabled: std::env::var("Q_AUTO_UPDATE")
                .unwrap_or_else(|_| "0".to_string())
                == "1",
            check_interval_secs: std::env::var("Q_AUTO_UPDATE_CHECK_INTERVAL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(300),
            restart_delay_secs: std::env::var("Q_AUTO_UPDATE_RESTART_DELAY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30),
            rollback_timeout_secs: std::env::var("Q_AUTO_UPDATE_ROLLBACK_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(60),
            min_peers: std::env::var("Q_AUTO_UPDATE_MIN_PEERS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2),
            mandatory_only: std::env::var("Q_AUTO_UPDATE_MANDATORY_ONLY")
                .unwrap_or_else(|_| "0".to_string())
                == "1",
            db_path: PathBuf::from(db_path),
            current_version: current_version.to_string(),
            network_id: network_id.to_string(),
            bootstrap_urls: vec![
                "https://quillon.xyz".to_string(),
                "https://dl.quillon.xyz".to_string(),
            ],
        }
    }
}

/// Restart marker written before graceful restart (used by rollback-check.sh)
#[derive(Debug, Serialize, Deserialize)]
pub struct RestartMarker {
    pub timestamp: u64,
    pub previous_version: String,
    pub new_version: String,
    pub pid: u32,
    pub binary_backup_path: String,
}

/// Maximum announcement messages to process per second (rate limit against spam)
const MAX_ANNOUNCEMENTS_PER_SEC: u32 = 5;

/// Allowed download URL prefixes — defense-in-depth against compromised quorum
/// pointing to a malicious binary host. Only these domains are accepted.
const ALLOWED_DOWNLOAD_PREFIXES: &[&str] = &[
    "https://quillon.xyz/downloads/",
    "https://dl.quillon.xyz/downloads/",
    "https://quillon.xyz/dist-final/downloads/",
];

/// Main auto-updater instance
pub struct NodeAutoUpdater {
    config: AutoUpdateConfig,
    quorum: tokio::sync::Mutex<UpdateQuorum>,
    trusted_signers: Vec<VerifyingKey>,
    state_tx: watch::Sender<NodeUpdateState>,
    state_rx: watch::Receiver<NodeUpdateState>,
    /// Channel to receive gossipsub announcements
    announcement_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    /// Reference to peer count for safety checks
    peer_count: Option<Arc<std::sync::atomic::AtomicUsize>>,
    /// Reference to current height and network height for sync check
    current_height: Arc<std::sync::atomic::AtomicU64>,
    network_height: Arc<std::sync::atomic::AtomicU64>,
    /// Accepted quorum announcement (latest with quorum) for co-signing and download
    quorum_announcement: tokio::sync::Mutex<Option<UpdateAnnouncement>>,
    /// Whether this is a bootstrap node (should co-sign)
    is_bootstrap: bool,
    /// Signing key for co-signing (bootstrap nodes only)
    signing_key: Arc<ed25519_dalek::SigningKey>,
    /// Command sender for publishing gossipsub messages
    libp2p_command_tx: Option<tokio::sync::mpsc::UnboundedSender<q_network::NetworkCommand>>,
    /// Shutting down flag
    shutting_down: Arc<AtomicBool>,
    /// v8.6.5: Shared runtime toggle — admin API can enable/disable auto-update at runtime
    /// This replaces the static config.enabled check so the toggle actually works.
    auto_update_enabled: Arc<AtomicBool>,
    /// Rate limiter: track announcements processed in the current second
    rate_limit_count: u32,
    rate_limit_window: std::time::Instant,
}

impl NodeAutoUpdater {
    /// Create a new auto-updater instance.
    /// Returns the updater + a sender for forwarding gossipsub announcements.
    pub fn new(
        config: AutoUpdateConfig,
        peer_count: Option<Arc<std::sync::atomic::AtomicUsize>>,
        current_height: Arc<std::sync::atomic::AtomicU64>,
        network_height: Arc<std::sync::atomic::AtomicU64>,
        is_bootstrap: bool,
        signing_key: Arc<ed25519_dalek::SigningKey>,
        libp2p_command_tx: Option<tokio::sync::mpsc::UnboundedSender<q_network::NetworkCommand>>,
        auto_update_enabled: Arc<AtomicBool>,
    ) -> (Self, mpsc::UnboundedSender<Vec<u8>>) {
        let initial_state = if config.enabled {
            NodeUpdateState::Idle
        } else {
            NodeUpdateState::Disabled
        };

        let (state_tx, state_rx) = watch::channel(initial_state);
        let (announcement_tx, announcement_rx) = mpsc::unbounded_channel();
        let trusted_signers = load_trusted_signers();

        info!(
            "🔄 [AUTO-UPDATE] Initialized: enabled={}, interval={}s, min_peers={}, mandatory_only={}, trusted_signers={}",
            config.enabled,
            config.check_interval_secs,
            config.min_peers,
            config.mandatory_only,
            trusted_signers.len()
        );

        (
            Self {
                config,
                quorum: tokio::sync::Mutex::new(UpdateQuorum::new()),
                trusted_signers,
                state_tx,
                state_rx,
                announcement_rx,
                peer_count,
                current_height,
                network_height,
                quorum_announcement: tokio::sync::Mutex::new(None),
                is_bootstrap,
                signing_key,
                libp2p_command_tx,
                shutting_down: Arc::new(AtomicBool::new(false)),
                auto_update_enabled,
                rate_limit_count: 0,
                rate_limit_window: std::time::Instant::now(),
            },
            announcement_tx,
        )
    }

    /// Get a receiver for watching state changes (for SSE/API)
    pub fn state_receiver(&self) -> watch::Receiver<NodeUpdateState> {
        self.state_rx.clone()
    }

    /// Spawn the background auto-updater task
    pub fn spawn(mut self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.run().await;
        })
    }

    /// Main event loop: listen for gossipsub announcements + HTTP poll timer
    async fn run(&mut self) {
        let poll_interval =
            tokio::time::Duration::from_secs(self.config.check_interval_secs);
        let mut poll_timer = tokio::time::interval(poll_interval);
        poll_timer.tick().await; // skip first immediate tick

        // Check if we just restarted after an auto-update (write healthy marker after timeout)
        self.check_restart_marker().await;

        info!("🔄 [AUTO-UPDATE] Background task started");

        loop {
            tokio::select! {
                // Gossipsub announcement received
                Some(data) = self.announcement_rx.recv() => {
                    self.handle_raw_announcement(&data).await;
                }

                // HTTP poll timer fired
                _ = poll_timer.tick() => {
                    if !self.shutting_down.load(Ordering::Relaxed) {
                        self.poll_version_endpoint().await;
                    }
                }

                else => {
                    info!("🔄 [AUTO-UPDATE] All channels closed, shutting down");
                    break;
                }
            }
        }
    }

    /// Handle a raw gossipsub announcement (bytes from P2P)
    async fn handle_raw_announcement(&mut self, data: &[u8]) {
        // Rate limit: max N announcements per second to prevent spam/DoS
        let now = std::time::Instant::now();
        if now.duration_since(self.rate_limit_window).as_secs() >= 1 {
            self.rate_limit_count = 0;
            self.rate_limit_window = now;
        }
        self.rate_limit_count += 1;
        if self.rate_limit_count > MAX_ANNOUNCEMENTS_PER_SEC {
            debug!("🔄 [AUTO-UPDATE] Rate limited: {} announcements/sec exceeded", MAX_ANNOUNCEMENTS_PER_SEC);
            return;
        }

        let announcement: UpdateAnnouncement = match serde_json::from_slice(data) {
            Ok(a) => a,
            Err(e) => {
                debug!("🔄 [AUTO-UPDATE] Failed to deserialize announcement: {}", e);
                return;
            }
        };

        self.handle_announcement(announcement).await;
    }

    /// Process a verified or pending announcement
    async fn handle_announcement(&mut self, announcement: UpdateAnnouncement) {
        // 1. Network ID must match
        if announcement.network_id != self.config.network_id {
            debug!(
                "🔄 [AUTO-UPDATE] Ignoring announcement for network {} (we are {})",
                announcement.network_id, self.config.network_id
            );
            return;
        }

        // 2. Version must be newer than current
        if !self.is_newer_version(&announcement.version) {
            debug!(
                "🔄 [AUTO-UPDATE] Ignoring announcement for v{} (current: v{})",
                announcement.version, self.config.current_version
            );
            return;
        }

        // 3. Check expiry
        if announcement.is_expired() {
            warn!(
                "🔄 [AUTO-UPDATE] Rejecting expired announcement for v{} (age > {}s)",
                announcement.version, MAX_ANNOUNCEMENT_AGE_SECS
            );
            return;
        }

        // 4. Verify Ed25519 signature
        if let Err(e) = announcement.verify_signature() {
            warn!(
                "🔄 [AUTO-UPDATE] Rejecting announcement with invalid signature: {}",
                e
            );
            return;
        }

        // 5. Check if signer is in the hardcoded trusted signers list
        let is_trusted = self
            .trusted_signers
            .iter()
            .any(|key| hex::encode(key.as_bytes()) == announcement.signer_pubkey);

        if !is_trusted {
            warn!(
                "🔄 [AUTO-UPDATE] Rejecting announcement from untrusted signer: {}",
                &announcement.signer_pubkey[..16]
            );
            return;
        }

        info!(
            "🔄 [AUTO-UPDATE] Valid announcement: v{} from {} (SHA256: {}...)",
            announcement.version,
            &announcement.signer_peer_id,
            &announcement.sha256_checksum[..16.min(announcement.sha256_checksum.len())]
        );

        // 6. Accumulate for quorum (prune old entries to prevent unbounded growth)
        let quorum_reached = {
            let mut quorum = self.quorum.lock().await;
            quorum.prune_older_than(&self.config.current_version);
            quorum.record(&announcement)
        };

        if quorum_reached {
            info!(
                "🔄 [AUTO-UPDATE] ✅ QUORUM REACHED for v{} (SHA256: {}...)",
                announcement.version,
                &announcement.sha256_checksum[..16.min(announcement.sha256_checksum.len())]
            );

            *self.quorum_announcement.lock().await = Some(announcement.clone());

            // v8.6.5: Check shared runtime toggle (admin API can enable/disable)
            let enabled = self.auto_update_enabled.load(Ordering::Relaxed);
            if enabled && (!self.config.mandatory_only || announcement.mandatory) {
                // Auto-update enabled — proceed to download
                self.download_and_apply(announcement).await;
            } else {
                // Auto-update disabled — just notify
                let _ = self.state_tx.send(NodeUpdateState::Available {
                    version: announcement.version.clone(),
                    download_url: announcement.download_url.clone(),
                });
                info!(
                    "🔄 [AUTO-UPDATE] New version v{} available: {}",
                    announcement.version, announcement.download_url
                );
            }
        } else {
            let signer_count = {
                let quorum = self.quorum.lock().await;
                quorum.signer_count(&announcement.version, &announcement.sha256_checksum)
            };

            let _ = self.state_tx.send(NodeUpdateState::WaitingForQuorum {
                version: announcement.version.clone(),
                signers_so_far: signer_count,
                signers_needed: MIN_UPDATE_QUORUM,
            });

            // Bootstrap nodes: co-sign if we're running the announced version
            if self.is_bootstrap
                && announcement.version == self.config.current_version
            {
                self.co_sign_announcement(&announcement).await;
            }
        }
    }

    /// Bootstrap node co-signs an announcement if it's running the same version
    async fn co_sign_announcement(&self, original: &UpdateAnnouncement) {
        let our_pubkey = hex::encode(self.signing_key.verifying_key().as_bytes());

        // Don't co-sign our own announcements
        if original.signer_pubkey == our_pubkey {
            return;
        }

        info!(
            "🔄 [AUTO-UPDATE] Co-signing announcement for v{} (we are running same version)",
            original.version
        );

        let our_peer_id = ""; // Will be filled from AppState if needed
        let mut co_signed = UpdateAnnouncement::new(
            original.version.clone(),
            original.sha256_checksum.clone(),
            original.blake3_checksum.clone(),
            original.binary_size,
            original.download_url.clone(),
            original.network_id.clone(),
            our_peer_id.to_string(),
            original.mandatory,
            original.release_notes.clone(),
        );

        co_signed.sign(&self.signing_key);

        // Publish via gossipsub
        if let Some(ref cmd_tx) = self.libp2p_command_tx {
            let topic = self.config.network_id.parse::<q_types::NetworkId>()
                .map(|nid| nid.update_announcements_topic())
                .unwrap_or_else(|_| format!("/qnk/{}/update-announcements", self.config.network_id));

            if let Ok(data) = serde_json::to_vec(&co_signed) {
                let _ = cmd_tx.send(q_network::NetworkCommand::PublishMessage {
                    topic,
                    data,
                });
                info!(
                    "🔄 [AUTO-UPDATE] Co-signed announcement published for v{}",
                    original.version
                );
            }
        }
    }

    /// Download, verify, and apply the update
    async fn download_and_apply(&mut self, announcement: UpdateAnnouncement) {
        // Safety gate: minimum peers
        if let Some(ref pc) = self.peer_count {
            let peers = pc.load(Ordering::Relaxed);
            if peers < self.config.min_peers {
                warn!(
                    "🔄 [AUTO-UPDATE] Aborting: only {} peers connected (need {})",
                    peers, self.config.min_peers
                );
                let _ = self.state_tx.send(NodeUpdateState::Error {
                    version: announcement.version.clone(),
                    message: format!(
                        "Insufficient peers: {} < {}",
                        peers, self.config.min_peers
                    ),
                    retry_count: 0,
                });
                return;
            }
        }

        // Safety gate: not syncing (< 10 blocks behind)
        let cur = self.current_height.load(Ordering::Relaxed);
        let net = self.network_height.load(Ordering::Relaxed);
        if net > 0 && net.saturating_sub(cur) > 10 {
            warn!(
                "🔄 [AUTO-UPDATE] Aborting: node is syncing (height {} vs network {})",
                cur, net
            );
            let _ = self.state_tx.send(NodeUpdateState::Error {
                version: announcement.version.clone(),
                message: format!("Node syncing: height {} vs network {}", cur, net),
                retry_count: 0,
            });
            return;
        }

        // 1. Download
        let _ = self.state_tx.send(NodeUpdateState::Downloading {
            version: announcement.version.clone(),
            progress_percent: 0,
        });

        let download_dir = self.config.db_path.join("auto_update");
        if let Err(e) = tokio::fs::create_dir_all(&download_dir).await {
            error!("🔄 [AUTO-UPDATE] Failed to create download dir: {}", e);
            return;
        }

        let temp_path = download_dir.join(format!("q-api-server-v{}.tmp", announcement.version));
        let final_path = download_dir.join(format!("q-api-server-v{}", announcement.version));

        // Clean up any stale temp files from previous failed downloads
        if temp_path.exists() {
            info!("🔄 [AUTO-UPDATE] Removing stale temp file from previous attempt");
            let _ = tokio::fs::remove_file(&temp_path).await;
        }

        // v8.6.5: Validate download URL against allowlist (defense-in-depth)
        let url_allowed = ALLOWED_DOWNLOAD_PREFIXES.iter().any(|prefix| announcement.download_url.starts_with(prefix));
        if !url_allowed {
            error!(
                "🔄 [AUTO-UPDATE] REJECTED: download URL '{}' is not in the allowlist!",
                announcement.download_url
            );
            let _ = self.state_tx.send(NodeUpdateState::Error {
                version: announcement.version.clone(),
                message: format!("Download URL rejected: not in allowlist"),
                retry_count: 0,
            });
            return;
        }

        match self.download_binary(&announcement.download_url, &temp_path).await {
            Ok(()) => {
                info!("🔄 [AUTO-UPDATE] Download complete: {:?}", temp_path);
            }
            Err(e) => {
                error!("🔄 [AUTO-UPDATE] Download failed: {}", e);
                let _ = self.state_tx.send(NodeUpdateState::Error {
                    version: announcement.version.clone(),
                    message: format!("Download failed: {}", e),
                    retry_count: 0,
                });
                return;
            }
        }

        // 2. Verify checksums
        let _ = self.state_tx.send(NodeUpdateState::Verifying {
            version: announcement.version.clone(),
        });

        if let Err(e) =
            self.verify_binary_integrity(&temp_path, &announcement).await
        {
            error!("🔄 [AUTO-UPDATE] Verification failed: {}", e);
            let _ = tokio::fs::remove_file(&temp_path).await;
            let _ = self.state_tx.send(NodeUpdateState::Error {
                version: announcement.version.clone(),
                message: format!("Verification failed: {}", e),
                retry_count: 0,
            });
            return;
        }

        // Make executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Err(e) =
                tokio::fs::set_permissions(&temp_path, std::fs::Permissions::from_mode(0o755))
                    .await
            {
                error!("🔄 [AUTO-UPDATE] Failed to set executable permission: {}", e);
                return;
            }
        }

        // 3. Preflight check
        let _ = self.state_tx.send(NodeUpdateState::PreflightCheck {
            version: announcement.version.clone(),
        });

        if let Err(e) = self.run_preflight_check(&temp_path).await {
            error!("🔄 [AUTO-UPDATE] Preflight check failed: {}", e);
            let _ = tokio::fs::remove_file(&temp_path).await;
            let _ = self.state_tx.send(NodeUpdateState::Error {
                version: announcement.version.clone(),
                message: format!("Preflight check failed: {}", e),
                retry_count: 0,
            });
            return;
        }

        // Move temp to final
        if let Err(e) = tokio::fs::rename(&temp_path, &final_path).await {
            error!("🔄 [AUTO-UPDATE] Failed to rename: {}", e);
            return;
        }

        // 4. Backup current binary
        let current_exe = match std::env::current_exe() {
            Ok(p) => p,
            Err(e) => {
                error!("🔄 [AUTO-UPDATE] Cannot determine current exe: {}", e);
                return;
            }
        };

        let backup_path = download_dir.join(format!(
            "q-api-server-v{}.backup",
            self.config.current_version
        ));

        if let Err(e) = tokio::fs::copy(&current_exe, &backup_path).await {
            error!("🔄 [AUTO-UPDATE] Failed to backup current binary: {}", e);
            return;
        }

        info!(
            "🔄 [AUTO-UPDATE] Backed up current binary to {:?}",
            backup_path
        );

        // 5. Write restart marker (for rollback detection)
        let marker = RestartMarker {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            previous_version: self.config.current_version.clone(),
            new_version: announcement.version.clone(),
            pid: std::process::id(),
            binary_backup_path: backup_path.to_string_lossy().to_string(),
        };

        let marker_path = self.config.db_path.join("restart_marker");
        if let Err(e) = tokio::fs::write(
            &marker_path,
            serde_json::to_string_pretty(&marker).unwrap_or_default(),
        )
        .await
        {
            error!("🔄 [AUTO-UPDATE] Failed to write restart marker: {}", e);
            return;
        }

        // 6. Apply binary swap using self_replace
        let _ = self.state_tx.send(NodeUpdateState::ReadyToRestart {
            version: announcement.version.clone(),
        });

        info!(
            "🔄 [AUTO-UPDATE] Applying binary swap: v{} → v{}",
            self.config.current_version, announcement.version
        );

        match self_replace::self_replace(&final_path) {
            Ok(()) => {
                info!("🔄 [AUTO-UPDATE] ✅ Binary swapped successfully!");
            }
            Err(e) => {
                error!("🔄 [AUTO-UPDATE] Binary swap failed: {}", e);
                let _ = tokio::fs::remove_file(&marker_path).await;
                let _ = self.state_tx.send(NodeUpdateState::Error {
                    version: announcement.version.clone(),
                    message: format!("Binary swap failed: {}", e),
                    retry_count: 0,
                });
                return;
            }
        }

        // 7. Schedule restart
        let delay = self.config.restart_delay_secs;
        let version = announcement.version.clone();

        let _ = self.state_tx.send(NodeUpdateState::RestartScheduled {
            version: version.clone(),
            restart_in_secs: delay,
        });

        info!(
            "🔄 [AUTO-UPDATE] Restart scheduled in {}s for v{}",
            delay, version
        );

        // Wait then send SIGUSR1 to self for graceful restart
        tokio::time::sleep(tokio::time::Duration::from_secs(delay)).await;

        info!("🔄 [AUTO-UPDATE] Sending SIGUSR1 for graceful restart...");

        #[cfg(unix)]
        {
            unsafe {
                libc::kill(libc::getpid(), libc::SIGUSR1);
            }
        }
    }

    /// Download binary from URL to local path (streamed to disk, not loaded into RAM)
    async fn download_binary(&self, url: &str, dest: &Path) -> Result<(), String> {
        info!("🔄 [AUTO-UPDATE] Downloading from {} (streaming to disk)", url);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .map_err(|e| format!("HTTP client error: {}", e))?;

        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("HTTP {} from {}", response.status(), url));
        }

        let total_size = response.content_length().unwrap_or(0);

        // v8.6.5: Stream chunks to disk instead of loading entire binary into memory
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;

        let mut file = tokio::fs::File::create(dest)
            .await
            .map_err(|e| format!("Failed to create file: {}", e))?;

        let mut stream = response.bytes_stream();
        let mut downloaded: u64 = 0;
        let mut last_log = std::time::Instant::now();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| format!("Stream error: {}", e))?;
            downloaded += chunk.len() as u64;
            file.write_all(&chunk)
                .await
                .map_err(|e| format!("Failed to write chunk: {}", e))?;

            // Log progress every 5 seconds
            if last_log.elapsed().as_secs() >= 5 {
                let pct = if total_size > 0 {
                    (downloaded * 100 / total_size) as u8
                } else {
                    0
                };
                info!(
                    "🔄 [AUTO-UPDATE] Download progress: {}/{} bytes ({}%)",
                    downloaded, total_size, pct
                );

                // Update state with progress
                let _ = self.state_tx.send(NodeUpdateState::Downloading {
                    version: String::new(), // Will be overwritten by caller
                    progress_percent: pct,
                });

                last_log = std::time::Instant::now();
            }
        }

        file.flush()
            .await
            .map_err(|e| format!("Failed to flush file: {}", e))?;

        info!(
            "🔄 [AUTO-UPDATE] Download complete: {} bytes written to {:?}",
            downloaded, dest
        );

        Ok(())
    }

    /// Verify binary integrity: size, SHA-256, BLAKE3
    async fn verify_binary_integrity(
        &self,
        path: &Path,
        announcement: &UpdateAnnouncement,
    ) -> Result<(), String> {
        let data = tokio::fs::read(path)
            .await
            .map_err(|e| format!("Failed to read binary: {}", e))?;

        // Size check
        if data.len() as u64 != announcement.binary_size {
            return Err(format!(
                "Size mismatch: got {} bytes, expected {}",
                data.len(),
                announcement.binary_size
            ));
        }

        // BLAKE3 (fast)
        let blake3_hash = blake3::hash(&data);
        let blake3_hex = blake3_hash.to_hex().to_string();
        if blake3_hex != announcement.blake3_checksum {
            return Err(format!(
                "BLAKE3 mismatch: got {}, expected {}",
                &blake3_hex[..16],
                &announcement.blake3_checksum[..16]
            ));
        }

        // SHA-256 (defense-in-depth)
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let sha256_hex = format!("{:x}", hasher.finalize());
        if sha256_hex != announcement.sha256_checksum {
            return Err(format!(
                "SHA-256 mismatch: got {}, expected {}",
                &sha256_hex[..16],
                &announcement.sha256_checksum[..16]
            ));
        }

        info!(
            "🔄 [AUTO-UPDATE] ✅ Checksums verified: BLAKE3={:.16}... SHA256={:.16}...",
            blake3_hex, sha256_hex
        );

        Ok(())
    }

    /// Run Q_PREFLIGHT_ONLY=1 on the new binary to verify database compatibility
    async fn run_preflight_check(&self, binary_path: &Path) -> Result<(), String> {
        info!("🔄 [AUTO-UPDATE] Running preflight check on {:?}", binary_path);

        let output = tokio::process::Command::new(binary_path)
            .env("Q_PREFLIGHT_ONLY", "1")
            .env("Q_DB_PATH", &self.config.db_path)
            .env("Q_NETWORK_ID", &self.config.network_id)
            .output()
            .await
            .map_err(|e| format!("Failed to run preflight: {}", e))?;

        if output.status.success() {
            info!("🔄 [AUTO-UPDATE] ✅ Preflight check PASSED");
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "Preflight check FAILED (exit code {:?}): {}",
                output.status.code(),
                &stderr[..stderr.len().min(500)]
            ))
        }
    }

    /// Poll bootstrap node /api/v1/version endpoint for updates (HTTP fallback)
    async fn poll_version_endpoint(&self) {
        for base_url in &self.config.bootstrap_urls {
            let url = format!("{}/api/v1/version", base_url);
            let client = match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
            {
                Ok(c) => c,
                Err(_) => continue,
            };

            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        if let Some(latest) = json.get("latest_node_version").and_then(|v| v.as_str())
                        {
                            if self.is_newer_version(latest) {
                                info!(
                                    "🔄 [AUTO-UPDATE] HTTP poll: v{} available (current: v{})",
                                    latest, self.config.current_version
                                );
                                // HTTP poll only notifies — quorum still required for auto-update
                                // But it helps nodes that missed gossipsub know about updates
                            }
                        }
                    }
                    break; // One successful poll is enough
                }
                Ok(resp) => {
                    debug!(
                        "🔄 [AUTO-UPDATE] HTTP poll {} returned {}",
                        url,
                        resp.status()
                    );
                }
                Err(e) => {
                    debug!("🔄 [AUTO-UPDATE] HTTP poll {} failed: {}", url, e);
                }
            }
        }
    }

    /// Check if a version string is newer than our current version
    fn is_newer_version(&self, version: &str) -> bool {
        // Simple semver comparison
        let parse = |v: &str| -> (u32, u32, u32) {
            let parts: Vec<&str> = v.trim_start_matches('v').split('.').collect();
            let major = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            let minor = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            let patch = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
            (major, minor, patch)
        };

        let current = parse(&self.config.current_version);
        let candidate = parse(version);

        candidate > current
    }

    /// After auto-update restart, check if we need to write the healthy marker
    /// or if we need to roll back
    async fn check_restart_marker(&self) {
        let marker_path = self.config.db_path.join("restart_marker");
        let healthy_path = self.config.db_path.join("update_healthy");
        let watchdog_lock_path = self.config.db_path.join("watchdog_active");

        if !marker_path.exists() {
            return; // Not a post-update restart
        }

        // Prevent double-watchdog: if a watchdog is already running for this marker,
        // skip. This can happen if the process crashes and restarts within the
        // watchdog timeout window.
        if watchdog_lock_path.exists() {
            // Check if the healthy marker was already written (watchdog passed before crash)
            if healthy_path.exists() {
                info!("🔄 [AUTO-UPDATE] Update already marked healthy — cleaning up stale markers");
                let _ = tokio::fs::remove_file(&marker_path).await;
                let _ = tokio::fs::remove_file(&watchdog_lock_path).await;
            } else {
                warn!("🔄 [AUTO-UPDATE] Watchdog lock exists but no healthy marker — previous watchdog may have failed. Re-running.");
                // Fall through to re-run watchdog
            }
        }

        // Read the marker
        let marker: RestartMarker = match tokio::fs::read_to_string(&marker_path).await {
            Ok(s) => match serde_json::from_str(&s) {
                Ok(m) => m,
                Err(e) => {
                    warn!("🔄 [AUTO-UPDATE] Invalid restart marker: {}", e);
                    let _ = tokio::fs::remove_file(&marker_path).await;
                    return;
                }
            },
            Err(e) => {
                warn!("🔄 [AUTO-UPDATE] Cannot read restart marker: {}", e);
                return;
            }
        };

        info!(
            "🔄 [AUTO-UPDATE] Post-update restart detected: v{} → v{} (writing healthy marker in {}s)",
            marker.previous_version,
            marker.new_version,
            self.config.rollback_timeout_secs
        );

        // Write watchdog lock so we don't re-run on subsequent crashes
        let _ = tokio::fs::write(
            &watchdog_lock_path,
            format!("{}", std::process::id()),
        ).await;

        // Spawn watchdog: after rollback_timeout_secs, write healthy marker
        let rollback_timeout = self.config.rollback_timeout_secs;
        let healthy_path_clone = healthy_path.clone();
        let marker_path_clone = marker_path.clone();
        let watchdog_lock_clone = watchdog_lock_path.clone();

        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_secs(rollback_timeout)).await;

            // If we get here, the node has been running for rollback_timeout_secs
            // without crashing — the update is healthy
            let healthy_data = serde_json::json!({
                "timestamp": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                "version": marker.new_version,
            });

            if let Err(e) = tokio::fs::write(
                &healthy_path_clone,
                serde_json::to_string_pretty(&healthy_data).unwrap_or_default(),
            )
            .await
            {
                error!("🔄 [AUTO-UPDATE] Failed to write healthy marker: {}", e);
                return;
            }

            // Clean up restart marker and watchdog lock
            let _ = tokio::fs::remove_file(&marker_path_clone).await;
            let _ = tokio::fs::remove_file(&watchdog_lock_clone).await;

            info!(
                "🔄 [AUTO-UPDATE] ✅ Update verified healthy after {}s — marker written, watchdog cleared",
                rollback_timeout
            );
        });
    }
}
