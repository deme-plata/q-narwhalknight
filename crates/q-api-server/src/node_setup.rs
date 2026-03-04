//! Automatic Node Setup Wizard via OAuth2 Device Login
//!
//! Provides zero-config first-boot experience for new node operators:
//! 1. Detect hardware (RAM, CPU, SSD/HDD)
//! 2. Fetch recommended config from bootstrap server
//! 3. Authenticate via OAuth2 device login (browser-based)
//! 4. Write .env + optional systemd service file
//!
//! Triggered by `--setup` flag or auto-detected on first boot
//! (no .env, no --admin-wallet, no Q_ADMIN_WALLET).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn, error};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub ram_gb: f64,
    pub cpu_cores: usize,
    pub is_ssd: bool,
    pub tier: HardwareTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HardwareTier {
    Low,
    Medium,
    High,
    XLarge,
}

impl HardwareTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            HardwareTier::Low => "low",
            HardwareTier::Medium => "medium",
            HardwareTier::High => "high",
            HardwareTier::XLarge => "xlarge",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    pub network_id: String,
    pub version: String,
    pub bootstrap_peers: Vec<String>,
    pub recommended: HashMap<String, String>,
    pub hardware_profiles: HashMap<String, HashMap<String, String>>,
}

#[derive(Debug, Clone)]
pub struct SetupResult {
    pub env_path: PathBuf,
    pub admin_wallet: Option<String>,
    pub service_path: Option<PathBuf>,
    pub network_id: String,
}

// ---------------------------------------------------------------------------
// Hardware detection
// ---------------------------------------------------------------------------

pub fn detect_hardware() -> HardwareProfile {
    use sysinfo::System;
    let sys = System::new_all();

    let ram_bytes = sys.total_memory(); // bytes
    let ram_gb = ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let cpu_cores = sys.cpus().len();

    // Detect SSD vs HDD via /sys/block/*/queue/rotational
    let is_ssd = detect_ssd();

    let tier = if ram_gb > 32.0 {
        HardwareTier::XLarge
    } else if ram_gb > 8.0 {
        HardwareTier::High
    } else if ram_gb > 4.0 {
        HardwareTier::Medium
    } else {
        HardwareTier::Low
    };

    HardwareProfile {
        ram_gb,
        cpu_cores,
        is_ssd,
        tier,
    }
}

fn detect_ssd() -> bool {
    // Check /sys/block/*/queue/rotational for each block device
    // 0 = SSD/NVMe, 1 = HDD
    let block_dir = Path::new("/sys/block");
    if !block_dir.exists() {
        return true; // Assume SSD if we can't detect
    }

    if let Ok(entries) = std::fs::read_dir(block_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            // Skip loop, ram, dm- devices
            if name_str.starts_with("loop")
                || name_str.starts_with("ram")
                || name_str.starts_with("dm-")
            {
                continue;
            }
            let rotational_path = entry.path().join("queue/rotational");
            if let Ok(content) = std::fs::read_to_string(&rotational_path) {
                if content.trim() == "1" {
                    return false; // HDD detected
                }
            }
        }
    }
    true // Default to SSD
}

// ---------------------------------------------------------------------------
// Bootstrap config fetch
// ---------------------------------------------------------------------------

pub async fn fetch_bootstrap_config(bootstrap_url: &str) -> Result<BootstrapConfig> {
    let url = format!("{}/api/v1/node-config", bootstrap_url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .context("Failed to create HTTP client")?;

    let resp = client
        .get(&url)
        .send()
        .await
        .context("Failed to connect to bootstrap server")?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "Bootstrap server returned HTTP {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        );
    }

    // The endpoint wraps in ApiResponse { success, data, ... }
    let body: serde_json::Value = resp.json().await.context("Invalid JSON from bootstrap")?;

    if let Some(data) = body.get("data") {
        let config: BootstrapConfig =
            serde_json::from_value(data.clone()).context("Failed to parse bootstrap config")?;
        Ok(config)
    } else {
        anyhow::bail!("Bootstrap response missing 'data' field");
    }
}

// ---------------------------------------------------------------------------
// OAuth2 device login (client side)
// ---------------------------------------------------------------------------

pub async fn run_device_login(bootstrap_url: &str) -> Result<String> {
    let base = bootstrap_url.trim_end_matches('/');
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .context("Failed to create HTTP client")?;

    // Step 1: Request device login code
    let resp = client
        .post(format!("{}/api/v1/miner/device-login", base))
        .send()
        .await
        .context("Failed to request device login code")?;

    let body: serde_json::Value = resp.json().await.context("Invalid response from device login")?;

    let data = body
        .get("data")
        .ok_or_else(|| anyhow::anyhow!("Missing 'data' in device login response"))?;

    let device_code = data["device_code"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing device_code"))?
        .to_string();
    let user_code = data["user_code"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing user_code"))?;
    let verification_url = data["verification_url"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing verification_url"))?;
    let interval = data["interval"].as_u64().unwrap_or(3);

    // Step 2: Display instructions and attempt to open browser
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║                    NODE SETUP — LOGIN                        ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════╣");
    eprintln!("║                                                               ║");
    eprintln!("║  Open this URL in your browser:                               ║");
    eprintln!("║                                                               ║");
    eprintln!("║    {}  ", verification_url);
    eprintln!("║                                                               ║");
    eprintln!("║  Your code: {}                                    ", user_code);
    eprintln!("║                                                               ║");
    eprintln!("║  Waiting for you to log in...                                 ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");
    eprintln!();

    // Try to open browser (best-effort, non-blocking)
    if let Err(e) = open::that(verification_url) {
        info!("Could not auto-open browser: {} — please open the URL manually", e);
    }

    // Step 3: Poll for completion (max 10 minutes)
    let max_attempts = 600 / interval.max(1); // 10 minutes
    for attempt in 0..max_attempts {
        tokio::time::sleep(std::time::Duration::from_secs(interval)).await;

        let poll_resp = client
            .get(format!("{}/api/v1/miner/device-login/{}", base, device_code))
            .send()
            .await;

        let poll_resp = match poll_resp {
            Ok(r) => r,
            Err(e) => {
                if attempt % 10 == 0 {
                    warn!("Poll failed (will retry): {}", e);
                }
                continue;
            }
        };

        let poll_body: serde_json::Value = match poll_resp.json().await {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Check for error (expired, invalid)
        if let Some(err) = poll_body.get("error").and_then(|e| e.as_str()) {
            if !err.is_empty() {
                anyhow::bail!("Device login failed: {}", err);
            }
        }

        if let Some(data) = poll_body.get("data") {
            let status = data["status"].as_str().unwrap_or("");
            if status == "complete" {
                let wallet = data["wallet_address"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("Login complete but no wallet_address"))?
                    .to_string();

                eprintln!();
                eprintln!("  ✅ Login successful! Wallet: {}...{}", &wallet[..wallet.len().min(12)], &wallet[wallet.len().saturating_sub(6)..]);
                eprintln!();
                return Ok(wallet);
            }
            // Still pending — keep polling
        }
    }

    anyhow::bail!("Device login timed out after 10 minutes. Please re-run setup.")
}

// ---------------------------------------------------------------------------
// .env file generation
// ---------------------------------------------------------------------------

pub fn write_env_file(
    env_path: &Path,
    bootstrap_config: Option<&BootstrapConfig>,
    admin_wallet: Option<&str>,
    hardware: &HardwareProfile,
    _port: u16,
) -> Result<PathBuf> {
    // Back up existing .env if present
    if env_path.exists() {
        let backup_path = env_path.with_extension("env.backup");
        std::fs::copy(env_path, &backup_path)
            .context("Failed to backup existing .env")?;
        eprintln!("  📋 Backed up existing .env → .env.backup");
    }

    let network_id = bootstrap_config
        .map(|c| c.network_id.as_str())
        .unwrap_or("mainnet-genesis");

    let db_path = format!("./data-{}", network_id);

    let cache_mb = match hardware.tier {
        HardwareTier::Low => "512",
        HardwareTier::Medium => "1024",
        HardwareTier::High => "2048",
        HardwareTier::XLarge => "4096",
    };

    let mut lines: Vec<String> = Vec::new();
    lines.push("# Q-NarwhalKnight Node Configuration".to_string());
    lines.push(format!("# Auto-generated by setup wizard on {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
    lines.push(String::new());

    // Core config
    lines.push("# Network".to_string());
    lines.push(format!("Q_NETWORK_ID={}", network_id));
    lines.push(format!("Q_DB_PATH={}", db_path));
    lines.push(String::new());

    // Admin wallet
    if let Some(wallet) = admin_wallet {
        lines.push("# Node admin wallet (controls admin panel)".to_string());
        lines.push(format!("Q_ADMIN_WALLET={}", wallet));
        lines.push(String::new());
    }

    // Safety & sync
    lines.push("# Safety & Sync".to_string());
    lines.push("Q_PREFLIGHT_CHECK=1".to_string());
    lines.push("Q_TURBO_SYNC=1".to_string());
    lines.push("Q_BATCHED_WRITES=1".to_string());
    lines.push("Q_STATE_SYNC=1".to_string());
    lines.push("Q_IS_VALIDATOR=true".to_string());
    lines.push(format!("Q_P2P_PORT=9001"));
    lines.push(String::new());

    // Hardware-tuned settings (v9.0.2: added sync tuning to prevent mining 503s during sync)
    lines.push("# Hardware-tuned (auto-detected)".to_string());
    lines.push(format!("ROCKSDB_BLOCK_CACHE_MB={}", cache_mb));
    if !hardware.is_ssd {
        lines.push("Q_CHEAP_SSD=1".to_string());
        lines.push("# HDD detected — SSD-friendly mode enabled to reduce write amplification".to_string());
    }

    // Sync tuning — prevents sync from starving mining handlers
    let (parallel_streams, sync_concurrency, rocksdb_write_rate) = match hardware.tier {
        HardwareTier::Low => ("4", "2", "50"),
        HardwareTier::Medium => ("8", "4", "100"),
        HardwareTier::High => ("16", "8", "200"),
        HardwareTier::XLarge => ("16", "8", "400"),
    };
    lines.push(format!("Q_TURBO_PARALLEL_STREAMS={}", parallel_streams));
    lines.push(format!("Q_SYNC_MAX_CONCURRENCY={}", sync_concurrency));
    lines.push(format!("Q_ROCKSDB_WRITE_RATE_MB={}", rocksdb_write_rate));
    lines.push(String::new());

    // Merge recommended values from bootstrap (skip keys we already set)
    if let Some(config) = bootstrap_config {
        let already_set: std::collections::HashSet<&str> = [
            "Q_NETWORK_ID", "Q_DB_PATH", "Q_ADMIN_WALLET", "Q_PREFLIGHT_CHECK",
            "Q_TURBO_SYNC", "Q_BATCHED_WRITES", "Q_STATE_SYNC", "Q_IS_VALIDATOR",
            "Q_P2P_PORT", "ROCKSDB_BLOCK_CACHE_MB", "Q_CHEAP_SSD",
            "Q_TURBO_PARALLEL_STREAMS", "Q_SYNC_MAX_CONCURRENCY", "Q_ROCKSDB_WRITE_RATE_MB",
        ]
        .iter()
        .copied()
        .collect();

        let mut extra: Vec<String> = Vec::new();
        for (key, value) in &config.recommended {
            if !already_set.contains(key.as_str()) {
                extra.push(format!("{}={}", key, value));
            }
        }

        // Also merge hardware-profile-specific overrides
        if let Some(profile) = config.hardware_profiles.get(hardware.tier.as_str()) {
            for (key, value) in profile {
                if !already_set.contains(key.as_str()) {
                    extra.push(format!("{}={}", key, value));
                }
            }
        }

        if !extra.is_empty() {
            lines.push("# Recommended by bootstrap server".to_string());
            extra.sort();
            lines.extend(extra);
            lines.push(String::new());
        }
    }

    let content = lines.join("\n") + "\n";
    std::fs::write(env_path, &content).context("Failed to write .env file")?;

    eprintln!("  ✅ Wrote {}", env_path.display());
    Ok(env_path.to_path_buf())
}

// ---------------------------------------------------------------------------
// Systemd service file generation
// ---------------------------------------------------------------------------

pub fn generate_service_file(
    binary_path: &Path,
    working_dir: &Path,
    env_path: &Path,
    port: u16,
) -> Result<Option<PathBuf>> {
    // Skip if already running under systemd
    if std::env::var("INVOCATION_ID").is_ok() {
        eprintln!("  ℹ️  Running under systemd — skipping service file generation");
        return Ok(None);
    }

    let binary_abs = std::fs::canonicalize(binary_path)
        .unwrap_or_else(|_| binary_path.to_path_buf());
    let workdir_abs = std::fs::canonicalize(working_dir)
        .unwrap_or_else(|_| working_dir.to_path_buf());
    let env_abs = std::fs::canonicalize(env_path)
        .unwrap_or_else(|_| env_path.to_path_buf());

    let service_content = format!(
        r#"[Unit]
Description=Q-NarwhalKnight Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={workdir}
ExecStart={binary} --port {port}
EnvironmentFile={env_file}
Restart=on-failure
RestartSec=5
LimitNOFILE=65536
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
"#,
        workdir = workdir_abs.display(),
        binary = binary_abs.display(),
        port = port,
        env_file = env_abs.display(),
    );

    #[cfg(unix)]
    let is_root = unsafe { libc::geteuid() } == 0;
    #[cfg(not(unix))]
    let is_root = false;

    if is_root {
        let service_path = Path::new("/etc/systemd/system/q-api-server.service");
        std::fs::write(service_path, &service_content)
            .context("Failed to write systemd service file")?;

        eprintln!("  ✅ Wrote {}", service_path.display());

        // daemon-reload + enable
        let _ = std::process::Command::new("systemctl")
            .args(["daemon-reload"])
            .status();
        let _ = std::process::Command::new("systemctl")
            .args(["enable", "q-api-server.service"])
            .status();
        eprintln!("  ✅ Enabled q-api-server.service (will auto-start on boot)");

        Ok(Some(service_path.to_path_buf()))
    } else {
        let local_path = working_dir.join("q-api-server.service");
        std::fs::write(&local_path, &service_content)
            .context("Failed to write local service file")?;

        eprintln!("  ✅ Wrote {}", local_path.display());
        eprintln!();
        eprintln!("  To install as a system service (requires root):");
        eprintln!("    sudo cp {} /etc/systemd/system/", local_path.display());
        eprintln!("    sudo systemctl daemon-reload");
        eprintln!("    sudo systemctl enable --now q-api-server");

        Ok(Some(local_path))
    }
}

// ---------------------------------------------------------------------------
// Main setup wizard orchestrator
// ---------------------------------------------------------------------------

pub async fn run_setup_wizard(
    binary_path: &Path,
    working_dir: &Path,
    port: u16,
    bootstrap_url: &str,
) -> Result<SetupResult> {
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║              Q-NARWHALKNIGHT NODE SETUP WIZARD               ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════╣");
    eprintln!("║  This wizard will configure your node automatically.         ║");
    eprintln!("║  You can re-run it anytime with: ./q-api-server --setup      ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");
    eprintln!();

    // Step 1: Detect hardware
    eprintln!("  [1/4] Detecting hardware...");
    let hardware = detect_hardware();
    eprintln!(
        "         RAM: {:.1} GB | CPUs: {} | Disk: {} | Tier: {:?}",
        hardware.ram_gb, hardware.cpu_cores,
        if hardware.is_ssd { "SSD" } else { "HDD" },
        hardware.tier,
    );
    eprintln!();

    // Step 2: Fetch bootstrap config
    eprintln!("  [2/4] Fetching network config from {}...", bootstrap_url);
    let bootstrap_config = match fetch_bootstrap_config(bootstrap_url).await {
        Ok(config) => {
            eprintln!(
                "         Network: {} | Version: {} | Peers: {}",
                config.network_id, config.version, config.bootstrap_peers.len()
            );
            Some(config)
        }
        Err(e) => {
            warn!("Could not fetch bootstrap config: {} — using defaults", e);
            eprintln!("         ⚠️  Could not reach bootstrap — using default config");
            None
        }
    };
    eprintln!();

    // Step 3: Device login
    eprintln!("  [3/4] Authenticating via browser...");
    let admin_wallet = match run_device_login(bootstrap_url).await {
        Ok(wallet) => Some(wallet),
        Err(e) => {
            warn!("Device login failed: {} — node will use default admin wallet", e);
            eprintln!("         ⚠️  Login skipped — you can set Q_ADMIN_WALLET in .env later");
            None
        }
    };
    eprintln!();

    // Step 4: Write config files
    eprintln!("  [4/4] Writing configuration...");
    let env_path = working_dir.join(".env");
    write_env_file(
        &env_path,
        bootstrap_config.as_ref(),
        admin_wallet.as_deref(),
        &hardware,
        port,
    )?;

    let service_path = generate_service_file(binary_path, working_dir, &env_path, port)?;

    let network_id = bootstrap_config
        .as_ref()
        .map(|c| c.network_id.clone())
        .unwrap_or_else(|| "mainnet-genesis".to_string());

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║                    SETUP COMPLETE ✅                          ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════╣");
    eprintln!("║  .env written — your node is configured and ready to run.    ║");
    if service_path.is_some() {
        eprintln!("║  systemd service installed — use: systemctl start q-api-server║");
    }
    eprintln!("║                                                               ║");
    eprintln!("║  To reconfigure: ./q-api-server --setup                       ║");
    eprintln!("║  To edit config:  nano .env                                   ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");
    eprintln!();

    Ok(SetupResult {
        env_path,
        admin_wallet,
        service_path,
        network_id,
    })
}

/// Check if this looks like a first-boot scenario (no .env, no admin wallet configured).
pub fn is_first_boot(working_dir: &Path) -> bool {
    let env_path = working_dir.join(".env");
    let has_env_file = env_path.exists();
    let has_admin_wallet_env = std::env::var("Q_ADMIN_WALLET").is_ok();

    !has_env_file && !has_admin_wallet_env
}
