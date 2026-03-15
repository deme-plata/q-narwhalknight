use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use semver::Version;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tokio::sync::watch;

/// Current binary version (set at compile time from Cargo.toml).
pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// State machine for the auto-update lifecycle.
#[derive(Clone, Debug)]
pub enum UpdateState {
    /// No update available / not checked yet.
    Idle,
    /// A newer version is available on the server.
    Available { version: String },
    /// Currently downloading the update binary.
    Downloading {
        version: String,
        bytes_downloaded: u64,
        bytes_total: u64,
    },
    /// Download complete, ready for user to restart.
    ReadyToRestart { version: String },
    /// An error occurred (user can retry).
    Error { version: String, message: String },
}

impl UpdateState {
    pub fn is_idle(&self) -> bool {
        matches!(self, UpdateState::Idle)
    }
}

/// Manages checking for, downloading, and applying wallet updates.
pub struct Updater {
    server_url: String,
    state_tx: watch::Sender<UpdateState>,
    state_rx: watch::Receiver<UpdateState>,
    /// Path to the downloaded temp file (set after successful download).
    temp_path: std::sync::Mutex<Option<PathBuf>>,
    /// Expected SHA-256 checksum from server (set during version check).
    expected_sha256: std::sync::Mutex<Option<String>>,
}

impl Updater {
    pub fn new(server_url: &str) -> Self {
        let (state_tx, state_rx) = watch::channel(UpdateState::Idle);
        Self {
            server_url: server_url.trim_end_matches('/').to_string(),
            state_tx,
            state_rx,
            temp_path: std::sync::Mutex::new(None),
            expected_sha256: std::sync::Mutex::new(None),
        }
    }

    /// Get a watch receiver for UI to observe state changes.
    pub fn subscribe(&self) -> watch::Receiver<UpdateState> {
        self.state_rx.clone()
    }

    /// Get current state snapshot.
    pub fn state(&self) -> UpdateState {
        self.state_rx.borrow().clone()
    }

    /// Check the server for a newer wallet version.
    /// Returns `Some(version_string)` if an update is available, `None` otherwise.
    pub async fn check_for_update(&self) -> Result<Option<String>> {
        let url = format!("{}/api/v1/version", self.server_url);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(anyhow!("Version check failed: HTTP {}", resp.status()));
        }
        let body: serde_json::Value = resp.json().await?;
        let latest = body
            .get("data")
            .and_then(|d| d.get("latest_wallet_version"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("No latest_wallet_version in response"))?;

        let current = Version::parse(CURRENT_VERSION)
            .map_err(|e| anyhow!("Bad current version '{}': {}", CURRENT_VERSION, e))?;
        let remote = Version::parse(latest)
            .map_err(|e| anyhow!("Bad remote version '{}': {}", latest, e))?;

        if remote > current {
            let ver = latest.to_string();
            // Capture SHA-256 checksum from server for post-download verification
            let sha256 = body
                .get("data")
                .and_then(|d| d.get("latest_wallet_sha256"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            *self.expected_sha256.lock().unwrap() = sha256;
            let _ = self
                .state_tx
                .send(UpdateState::Available { version: ver.clone() });
            Ok(Some(ver))
        } else {
            Ok(None)
        }
    }

    /// Build the platform-specific download URL for a given version.
    fn build_download_url(&self, version: &str) -> String {
        let filename = if cfg!(target_os = "windows") {
            format!("slint-wallet-v{}.exe", version)
        } else {
            format!("slint-wallet-v{}", version)
        };
        format!("{}/downloads/{}", self.server_url, filename)
    }

    /// Download the update binary with progress reporting.
    pub async fn download_update(&self, version: &str) -> Result<()> {
        let url = self.build_download_url(version);
        let version_owned = version.to_string();

        let _ = self.state_tx.send(UpdateState::Downloading {
            version: version_owned.clone(),
            bytes_downloaded: 0,
            bytes_total: 0,
        });

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()?;
        let resp = client.get(&url).send().await?;

        if !resp.status().is_success() {
            let msg = format!("Download failed: HTTP {}", resp.status());
            let _ = self.state_tx.send(UpdateState::Error {
                version: version_owned,
                message: msg.clone(),
            });
            return Err(anyhow!(msg));
        }

        let total = resp.content_length().unwrap_or(0);
        let mut downloaded: u64 = 0;

        // Write to a temp file next to the current executable
        let current_exe = std::env::current_exe()?;
        let temp_path = current_exe.with_extension("update-tmp");

        // Clean up stale temp file from any previous failed download
        if temp_path.exists() {
            let _ = tokio::fs::remove_file(&temp_path).await;
        }

        let mut file = tokio::fs::File::create(&temp_path).await?;

        let mut stream = resp.bytes_stream();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            downloaded += chunk.len() as u64;
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;

            let _ = self.state_tx.send(UpdateState::Downloading {
                version: version_owned.clone(),
                bytes_downloaded: downloaded,
                bytes_total: total,
            });
        }
        tokio::io::AsyncWriteExt::flush(&mut file).await?;
        drop(file);

        // Verify SHA-256 checksum (MANDATORY — prevents MITM / corrupt downloads)
        let expected = self.expected_sha256.lock().unwrap().clone();
        if let Some(expected) = expected.as_ref() {
            let file_data = tokio::fs::read(&temp_path).await?;
            let mut hasher = Sha256::new();
            hasher.update(&file_data);
            let actual = format!("{:x}", hasher.finalize());
            if actual != *expected {
                // Delete the corrupt/tampered binary immediately
                let _ = tokio::fs::remove_file(&temp_path).await;
                let msg = format!(
                    "SHA-256 mismatch! Expected {}..., got {}... — possible tampering",
                    &expected[..16.min(expected.len())],
                    &actual[..16]
                );
                let _ = self.state_tx.send(UpdateState::Error {
                    version: version_owned,
                    message: msg.clone(),
                });
                return Err(anyhow!(msg));
            }
            eprintln!("✅ [UPDATE] SHA-256 verified: {}...", &actual[..16]);
        } else {
            // v8.6.5: Refuse updates without checksum — prevents blind binary replacement
            let _ = tokio::fs::remove_file(&temp_path).await;
            let msg = "Server did not provide SHA-256 checksum — refusing update for safety".to_string();
            let _ = self.state_tx.send(UpdateState::Error {
                version: version_owned,
                message: msg.clone(),
            });
            return Err(anyhow!(msg));
        }

        // Make executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&temp_path, std::fs::Permissions::from_mode(0o755))?;
        }

        // Store temp path for apply step
        *self.temp_path.lock().unwrap() = Some(temp_path);

        let _ = self.state_tx.send(UpdateState::ReadyToRestart {
            version: version_owned,
        });

        Ok(())
    }

    /// Apply the downloaded update by replacing the current binary.
    /// After this call, the caller should spawn the new binary and exit.
    pub fn apply_update(&self) -> Result<()> {
        let temp_path = self
            .temp_path
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| anyhow!("No downloaded update to apply"))?;

        if !temp_path.exists() {
            return Err(anyhow!("Update temp file missing: {:?}", temp_path));
        }

        // self-replace handles cross-platform atomic binary swap:
        // - Linux: rename temp over current exe (atomic)
        // - Windows: rename current aside, rename temp in place
        self_replace::self_replace(&temp_path)?;

        // Clean up temp file (may already be gone after self-replace on Linux)
        let _ = std::fs::remove_file(&temp_path);

        Ok(())
    }

    /// Restart by spawning the new binary and exiting the current process.
    pub fn restart() -> Result<()> {
        let exe = std::env::current_exe()?;
        let args: Vec<String> = std::env::args().skip(1).collect();

        std::process::Command::new(&exe).args(&args).spawn()?;

        std::process::exit(0);
    }

    /// Reset state back to Available (for retry after error).
    pub fn reset_to_available(&self, version: &str) {
        let _ = self.state_tx.send(UpdateState::Available {
            version: version.to_string(),
        });
    }

    /// Reset state to idle (dismiss update notification).
    pub fn reset_to_idle(&self) {
        let _ = self.state_tx.send(UpdateState::Idle);
    }
}
