/// Configuration management for Quillon Bank CLI

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    pub board: BoardConfig,
    pub node: NodeConfig,
    pub policies: PoliciesConfig,
    pub stablecoin: StablecoinConfig,
    pub notifications: NotificationsConfig,
    pub claude: ClaudeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardConfig {
    pub member_id: String,
    pub authentication: String,
    pub key_path: PathBuf,
    pub mfa_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub api_endpoint: String,
    pub backup_endpoints: Vec<String>,
    pub timeout: u64,
    pub retry_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoliciesConfig {
    pub auto_approve_loans_under: u64,
    pub auto_liquidate_enabled: bool,
    pub auto_liquidate_threshold: u64,
    pub daily_withdrawal_limit: u64,
    pub min_collateral_ratio: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StablecoinConfig {
    pub target_collateral_ratio: u64,
    pub auto_rebalance: bool,
    pub rebalance_threshold: u64,
    pub peg_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationsConfig {
    pub slack_webhook: Option<String>,
    pub email: Option<String>,
    pub critical_alerts: bool,
    pub daily_summary: bool,
    pub weekly_report: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeConfig {
    pub autonomous_mode: bool,
    pub auto_approve_limit: u64,
    pub require_confirmation: Vec<String>,
}

impl CliConfig {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            return Ok(Self::default());
        }

        let contents = std::fs::read_to_string(&config_path)
            .context("Failed to read config file")?;

        toml::from_str(&contents).context("Failed to parse config file")
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, contents)?;

        Ok(())
    }

    pub fn config_path() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        Ok(home.join(".quillon").join("config.toml"))
    }

    pub fn keys_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        Ok(home.join(".quillon").join("keys"))
    }
}

impl Default for CliConfig {
    fn default() -> Self {
        let keys_dir = CliConfig::keys_dir().unwrap_or_else(|_| PathBuf::from("."));

        Self {
            board: BoardConfig {
                member_id: "board-member-001".to_string(),
                authentication: "key-file".to_string(),
                key_path: keys_dir.join("board-key.pem"),
                mfa_enabled: false,
            },
            node: NodeConfig {
                api_endpoint: "http://localhost:8090".to_string(),
                backup_endpoints: vec![],
                timeout: 30,
                retry_attempts: 3,
            },
            policies: PoliciesConfig {
                auto_approve_loans_under: 10000,
                auto_liquidate_enabled: true,
                auto_liquidate_threshold: 100,
                daily_withdrawal_limit: 5000000,
                min_collateral_ratio: 105,
            },
            stablecoin: StablecoinConfig {
                target_collateral_ratio: 110,
                auto_rebalance: true,
                rebalance_threshold: 5,
                peg_tolerance: 0.005,
            },
            notifications: NotificationsConfig {
                slack_webhook: None,
                email: None,
                critical_alerts: true,
                daily_summary: true,
                weekly_report: true,
            },
            claude: ClaudeConfig {
                autonomous_mode: false,
                auto_approve_limit: 100000,
                require_confirmation: vec![
                    "liquidations".to_string(),
                    "large-mints".to_string(),
                    "policy-changes".to_string(),
                ],
            },
        }
    }
}