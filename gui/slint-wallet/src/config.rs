//! User-persistent settings for the Slint wallet.
//!
//! Stored at `~/.config/quillon-wallet/config.toml` on Linux,
//! `%APPDATA%\quillon-wallet\config.toml` on Windows.
//!
//! First-launch defaults are intentionally low-resource: 1 mining thread, 5%
//! GPU intensity. Users dial up via the in-app sliders (miner.slint), and
//! their choice is written back here so the next launch honors it.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletConfig {
    #[serde(default)]
    pub mining: MiningConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningConfig {
    /// Number of CPU mining threads. `None` = use first-launch default (1).
    pub cpu_threads: Option<usize>,
    /// GPU intensity 1..=100 (percent). `None` = use first-launch default (5).
    pub gpu_intensity_pct: Option<u8>,
}

impl Default for WalletConfig {
    fn default() -> Self {
        Self {
            mining: MiningConfig::default(),
        }
    }
}

impl Default for MiningConfig {
    fn default() -> Self {
        Self {
            cpu_threads: None,
            gpu_intensity_pct: None,
        }
    }
}

impl WalletConfig {
    /// Resolve the cpu thread count to use right now: either the user's
    /// persisted choice, or the conservative first-launch default of 1.
    pub fn effective_cpu_threads(&self) -> usize {
        self.mining.cpu_threads.unwrap_or(1).max(1)
    }

    /// Resolve GPU intensity (1..=100). Default 5%.
    pub fn effective_gpu_intensity_pct(&self) -> u8 {
        self.mining
            .gpu_intensity_pct
            .unwrap_or(5)
            .clamp(1, 100)
    }
}

fn config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("quillon-wallet").join("config.toml"))
}

pub fn load() -> WalletConfig {
    let Some(path) = config_path() else {
        return WalletConfig::default();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return WalletConfig::default();
    };
    toml::from_str(&text).unwrap_or_default()
}

pub fn save(cfg: &WalletConfig) -> std::io::Result<()> {
    let path = config_path().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "no config dir on this platform")
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = toml::to_string_pretty(cfg).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    })?;
    std::fs::write(path, text)
}

/// Convenience: update only the cpu thread count and persist.
pub fn set_cpu_threads(threads: usize) {
    let mut cfg = load();
    cfg.mining.cpu_threads = Some(threads.max(1));
    let _ = save(&cfg);
}

/// Convenience: update only the GPU intensity and persist.
pub fn set_gpu_intensity_pct(pct: u8) {
    let mut cfg = load();
    cfg.mining.gpu_intensity_pct = Some(pct.clamp(1, 100));
    let _ = save(&cfg);
}
