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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WalletConfig {
    #[serde(default)]
    pub mining: MiningConfig,
    /// v1.3.0: Settings page additions (Network / Appearance / Updates / etc).
    #[serde(default)]
    pub settings: SettingsConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MiningConfig {
    /// Number of CPU mining threads. `None` = use first-launch default (1).
    pub cpu_threads: Option<usize>,
    /// GPU intensity 1..=100 (percent). `None` = use first-launch default (5).
    pub gpu_intensity_pct: Option<u8>,
}

/// v1.3.0: user-configurable settings exposed by the Settings screen.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SettingsConfig {
    /// API base URL override. `None` = default `https://quillon.xyz`.
    pub rpc_url: Option<String>,
    /// `"dark"` (default) | `"light"`.
    pub theme: Option<String>,
    /// `"cyan"` (default) | `"purple"` | `"green"` | `"yellow"`.
    pub accent_color: Option<String>,
    /// Linux/macOS autostart + Windows Run-key. `None` = installed by default
    /// at first launch via `desktop_integration::install_desktop_integration`.
    pub autostart_enabled: Option<bool>,
    /// `None` (default true) or `Some(false)` to disable check-and-self-replace.
    pub auto_update_enabled: Option<bool>,
    /// `"stable"` (default) | `"beta"`.
    pub update_channel: Option<String>,
    /// Minutes to remember the unlock session after closing the app.
    /// `None` = always require password on launch. `Some(n)` = stay
    /// auto-unlocked for n minutes after last app close.
    pub session_persistence_minutes: Option<u32>,
    /// Disable the quantum-particle background on the home screen for
    /// low-power devices (Pi). Default false (particles on).
    pub disable_particle_bg: Option<bool>,
    /// v1.3.0: skip the password prompt at launch when on. Less secure;
    /// users opt in via Settings explicitly. Default false.
    pub auto_login_enabled: Option<bool>,
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

    /// Resolve RPC base URL. Default `https://quillon.xyz`.
    pub fn effective_rpc_url(&self) -> String {
        self.settings
            .rpc_url
            .clone()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "https://quillon.xyz".to_string())
    }

    /// Resolve theme. Default `"dark"`.
    pub fn effective_theme(&self) -> String {
        self.settings.theme.clone().unwrap_or_else(|| "dark".to_string())
    }

    /// Resolve accent color. Default `"cyan"`.
    pub fn effective_accent(&self) -> String {
        self.settings
            .accent_color
            .clone()
            .unwrap_or_else(|| "cyan".to_string())
    }

    pub fn is_autostart_enabled(&self) -> bool {
        self.settings.autostart_enabled.unwrap_or(true)
    }

    pub fn is_auto_update_enabled(&self) -> bool {
        self.settings.auto_update_enabled.unwrap_or(true)
    }

    pub fn effective_update_channel(&self) -> String {
        self.settings
            .update_channel
            .clone()
            .unwrap_or_else(|| "stable".to_string())
    }

    pub fn session_persistence_minutes(&self) -> u32 {
        self.settings.session_persistence_minutes.unwrap_or(0)
    }

    pub fn particle_bg_enabled(&self) -> bool {
        !self.settings.disable_particle_bg.unwrap_or(false)
    }

    pub fn is_auto_login_enabled(&self) -> bool {
        self.settings.auto_login_enabled.unwrap_or(false)
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

// v1.3.0: Settings-page setters. Each one loads, mutates one field, persists.
// `let _ = save(...)` because we never want a write failure to crash the UI.

pub fn set_rpc_url(url: String) {
    let mut cfg = load();
    let trimmed = url.trim().to_string();
    cfg.settings.rpc_url = if trimmed.is_empty() { None } else { Some(trimmed) };
    let _ = save(&cfg);
}

pub fn set_theme(theme: String) {
    let mut cfg = load();
    cfg.settings.theme = Some(theme);
    let _ = save(&cfg);
}

pub fn set_accent_color(accent: String) {
    let mut cfg = load();
    cfg.settings.accent_color = Some(accent);
    let _ = save(&cfg);
}

pub fn set_autostart_enabled(enabled: bool) {
    let mut cfg = load();
    cfg.settings.autostart_enabled = Some(enabled);
    let _ = save(&cfg);
}

pub fn set_auto_update_enabled(enabled: bool) {
    let mut cfg = load();
    cfg.settings.auto_update_enabled = Some(enabled);
    let _ = save(&cfg);
}

pub fn set_update_channel(channel: String) {
    let mut cfg = load();
    cfg.settings.update_channel = Some(channel);
    let _ = save(&cfg);
}

pub fn set_session_persistence_minutes(minutes: u32) {
    let mut cfg = load();
    cfg.settings.session_persistence_minutes =
        if minutes == 0 { None } else { Some(minutes) };
    let _ = save(&cfg);
}

pub fn set_disable_particle_bg(disabled: bool) {
    let mut cfg = load();
    cfg.settings.disable_particle_bg = Some(disabled);
    let _ = save(&cfg);
}

pub fn set_auto_login_enabled(enabled: bool) {
    let mut cfg = load();
    cfg.settings.auto_login_enabled = Some(enabled);
    let _ = save(&cfg);
}
