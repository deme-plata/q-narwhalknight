// ═══════════════════════════════════════════════════════════════════
// MinerAutoUpdater: Background auto-update for q-miner binary
//
// Flow: check /api/v1/version → compare semver → download to .update-tmp
//       → SHA-256 verify → park until user presses [U] → self_replace → restart
// ═══════════════════════════════════════════════════════════════════

use crate::shared_state::DiagnosticEvent;
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

/// Current state of the auto-updater, observable by TUI via watch channel.
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateState {
    Idle,
    Checking,
    Available { version: String },
    Downloading { version: String, bytes_downloaded: u64, bytes_total: u64 },
    ReadyToApply { version: String },
    Applying { version: String },
    Error { version: String, message: String },
}

impl UpdateState {
    pub fn version(&self) -> Option<&str> {
        match self {
            UpdateState::Available { version }
            | UpdateState::Downloading { version, .. }
            | UpdateState::ReadyToApply { version }
            | UpdateState::Applying { version }
            | UpdateState::Error { version, .. } => Some(version),
            _ => None,
        }
    }
}

pub struct MinerAutoUpdater {
    server_url: String,
    event_tx: mpsc::UnboundedSender<DiagnosticEvent>,
    state_tx: watch::Sender<UpdateState>,
    state_rx: watch::Receiver<UpdateState>,
    http_client: reqwest::Client,
}

impl MinerAutoUpdater {
    pub fn new(
        server_url: String,
        event_tx: mpsc::UnboundedSender<DiagnosticEvent>,
        proxy_url: Option<&str>,
    ) -> Self {
        let (state_tx, state_rx) = watch::channel(UpdateState::Idle);

        let mut builder = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .connect_timeout(std::time::Duration::from_secs(15));
        if let Some(proxy) = proxy_url {
            if let Ok(p) = reqwest::Proxy::all(proxy) {
                builder = builder.proxy(p);
            }
        }
        let http_client = builder.build().unwrap_or_else(|_| reqwest::Client::new());

        Self {
            server_url,
            event_tx,
            state_tx,
            state_rx,
            http_client,
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<UpdateState> {
        self.state_rx.clone()
    }

    fn set_state(&self, state: UpdateState) {
        let _ = self.state_tx.send(state);
    }

    /// Check the server for a newer miner version.
    /// Returns Some((version, sha256, download_url)) if an update is available.
    async fn check_for_update(&self) -> Result<Option<(String, String, String)>> {
        self.set_state(UpdateState::Checking);

        let url = format!("{}/api/v1/version", self.server_url.trim_end_matches('/'));
        let resp = self.http_client.get(&url)
            .send()
            .await
            .context("Failed to fetch version info")?;

        let body: serde_json::Value = resp.json().await.context("Failed to parse version response")?;

        // Navigate into {"data": {...}} wrapper
        let data = body.get("data").unwrap_or(&body);

        let remote_version = match data.get("latest_miner_version").and_then(|v| v.as_str()) {
            Some(v) => v.to_string(),
            None => {
                debug!("Server does not report latest_miner_version");
                self.set_state(UpdateState::Idle);
                return Ok(None);
            }
        };

        let remote_sha256 = match data.get("latest_miner_sha256").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => {
                debug!("Server does not report latest_miner_sha256 — refusing update without checksum");
                self.set_state(UpdateState::Idle);
                return Ok(None);
            }
        };

        // Compare versions using semver
        let current = semver::Version::parse(env!("CARGO_PKG_VERSION"))
            .unwrap_or_else(|_| semver::Version::new(0, 0, 0));
        let remote = match semver::Version::parse(&remote_version) {
            Ok(v) => v,
            Err(e) => {
                warn!("Invalid remote miner version '{}': {}", remote_version, e);
                self.set_state(UpdateState::Idle);
                return Ok(None);
            }
        };

        if remote <= current {
            debug!("Miner is up to date (local={}, remote={})", current, remote);
            self.set_state(UpdateState::Idle);
            return Ok(None);
        }

        info!("Miner update available: v{} → v{}", current, remote);

        // Build download URL
        let download_url = match data.get("latest_miner_download_url").and_then(|v| v.as_str()) {
            Some(u) => u.to_string(),
            None => {
                // Fall back to convention
                let base = self.server_url.trim_end_matches('/');
                format!("{}/downloads/q-miner-v{}", base, remote_version)
            }
        };

        self.set_state(UpdateState::Available { version: remote_version.clone() });
        self.send_event(DiagnosticEvent::UpdateDownloading {
            version: remote_version.clone(),
            bytes_downloaded: 0,
            bytes_total: 0,
        });

        Ok(Some((remote_version, remote_sha256, download_url)))
    }

    /// Download the update binary, verify SHA-256, and mark ready to apply.
    async fn download_update(&self, version: &str, expected_sha256: &str, download_url: &str) -> Result<PathBuf> {
        let current_exe = std::env::current_exe().context("Cannot determine current exe path")?;
        let tmp_path = current_exe.with_extension("update-tmp");

        // Clean up any leftover tmp from previous attempts
        let _ = std::fs::remove_file(&tmp_path);

        info!("Downloading miner v{} from {}", version, download_url);

        let resp = self.http_client.get(download_url)
            .send()
            .await
            .context("Download request failed")?;

        if !resp.status().is_success() {
            bail!("Download failed: HTTP {}", resp.status());
        }

        let total_size = resp.content_length().unwrap_or(0);

        self.set_state(UpdateState::Downloading {
            version: version.to_string(),
            bytes_downloaded: 0,
            bytes_total: total_size,
        });

        // Stream download to tmp file
        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut file = tokio::fs::File::create(&tmp_path).await
            .context("Cannot create temp file for update")?;
        let mut downloaded: u64 = 0;
        let mut hasher = Sha256::new();
        let mut last_progress_event = std::time::Instant::now();

        use tokio::io::AsyncWriteExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Download stream error")?;
            file.write_all(&chunk).await.context("Write to temp file failed")?;
            hasher.update(&chunk);
            downloaded += chunk.len() as u64;

            // Rate-limit progress events to ~4/sec
            if last_progress_event.elapsed() >= std::time::Duration::from_millis(250) {
                self.set_state(UpdateState::Downloading {
                    version: version.to_string(),
                    bytes_downloaded: downloaded,
                    bytes_total: total_size,
                });
                self.send_event(DiagnosticEvent::UpdateDownloading {
                    version: version.to_string(),
                    bytes_downloaded: downloaded,
                    bytes_total: total_size,
                });
                last_progress_event = std::time::Instant::now();
            }
        }
        file.flush().await?;
        drop(file);

        // SHA-256 verification (mandatory)
        let actual_sha256 = format!("{:x}", hasher.finalize());
        if actual_sha256 != expected_sha256 {
            let _ = std::fs::remove_file(&tmp_path);
            bail!(
                "SHA-256 mismatch! expected={}, actual={}",
                expected_sha256, actual_sha256
            );
        }

        info!("SHA-256 verified for miner v{} ({} bytes)", version, downloaded);

        // Set executable permission on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&tmp_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&tmp_path, perms)?;
        }

        self.set_state(UpdateState::ReadyToApply { version: version.to_string() });
        self.send_event(DiagnosticEvent::UpdateReadyToApply { version: version.to_string() });

        Ok(tmp_path)
    }

    /// Apply the update: atomic binary swap via self_replace.
    pub fn apply_update(tmp_path: &std::path::Path, version: &str) -> Result<()> {
        info!("Applying miner update v{} via self_replace", version);
        self_replace::self_replace(tmp_path)
            .context("self_replace failed")?;
        // Clean up tmp file after successful replace
        let _ = std::fs::remove_file(tmp_path);
        Ok(())
    }

    /// Restart the miner process with the same arguments.
    pub fn restart() -> ! {
        // On Linux, after self_replace atomically swaps the binary via rename(),
        // /proc/self/exe shows the OLD inode as "<path> (deleted)".
        // Strip the suffix to get the actual filesystem path, which now
        // points to the NEW binary placed there by the rename.
        let exe_raw = std::env::current_exe().expect("Cannot determine current exe");
        let exe = {
            let s = exe_raw.to_string_lossy();
            if let Some(stripped) = s.strip_suffix(" (deleted)") {
                std::path::PathBuf::from(stripped)
            } else {
                exe_raw
            }
        };
        let args: Vec<String> = std::env::args().skip(1).collect();
        info!("Restarting miner: {:?} {:?}", exe, args);

        // Spawn new process
        let _ = std::process::Command::new(&exe)
            .args(&args)
            .spawn()
            .expect("Failed to restart miner");

        // Exit current process
        std::process::exit(0);
    }

    fn send_event(&self, event: DiagnosticEvent) {
        let _ = self.event_tx.send(event);
    }

    /// Main update loop: runs as a background tokio task.
    /// Checks every 5 minutes, auto-downloads when available, then parks
    /// waiting for user to trigger apply via [U] key.
    pub async fn run_update_loop(self: Arc<Self>) {
        // Initial delay — let miner stabilize before checking
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;

        let check_interval = std::time::Duration::from_secs(300); // 5 minutes

        loop {
            match self.check_for_update().await {
                Ok(Some((version, sha256, url))) => {
                    match self.download_update(&version, &sha256, &url).await {
                        Ok(_tmp_path) => {
                            info!("Update v{} downloaded and verified — press [U] in TUI to apply", version);
                            // Park here — the TUI [U] key handler will call apply_update + restart.
                            // We just keep the state as ReadyToApply and stop checking.
                            // The next check will happen only if the state gets reset to Idle (error recovery).
                            loop {
                                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                                let current_state = self.state_rx.borrow().clone();
                                match current_state {
                                    UpdateState::ReadyToApply { .. } => continue, // still waiting
                                    UpdateState::Error { .. } => break, // retry on error
                                    _ => break,
                                }
                            }
                        }
                        Err(e) => {
                            error!("Update download failed: {}", e);
                            self.set_state(UpdateState::Error {
                                version: version.clone(),
                                message: e.to_string(),
                            });
                            self.send_event(DiagnosticEvent::UpdateError {
                                version,
                                message: e.to_string(),
                            });
                        }
                    }
                }
                Ok(None) => {
                    // No update available, or server doesn't have miner version info yet
                }
                Err(e) => {
                    debug!("Update check failed: {}", e);
                    self.set_state(UpdateState::Idle);
                }
            }

            tokio::time::sleep(check_interval).await;
        }
    }

    /// Get the path to the downloaded update tmp file (if ready).
    pub fn update_tmp_path() -> PathBuf {
        let current_exe = std::env::current_exe().unwrap_or_default();
        current_exe.with_extension("update-tmp")
    }
}
