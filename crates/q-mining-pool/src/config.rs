//! Pool configuration

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::{
    DEFAULT_MIN_PAYOUT, DEFAULT_POOL_FEE_BPS, DEFAULT_PPLNS_N_FACTOR,
    DEFAULT_STRATUM_PORT, DEFAULT_VARDIFF_TARGET_TIME,
};

/// Pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Pool name
    pub name: String,

    /// Pool operator wallet address
    pub pool_wallet: String,

    /// Stratum server configuration
    pub stratum: StratumConfig,

    /// Fee configuration
    pub fees: FeeConfig,

    /// PPLNS configuration
    pub pplns: PPLNSConfig,

    /// Vardiff configuration
    pub vardiff: VardiffConfig,

    /// Payout configuration
    pub payout: PayoutConfig,

    /// Security configuration
    pub security: SecurityConfig,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            name: "Q-NarwhalKnight Pool".to_string(),
            pool_wallet: String::new(),
            stratum: StratumConfig::default(),
            fees: FeeConfig::default(),
            pplns: PPLNSConfig::default(),
            vardiff: VardiffConfig::default(),
            payout: PayoutConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

/// Stratum server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumConfig {
    /// Stratum server port
    pub port: u16,

    /// Bind address
    pub bind_address: String,

    /// Maximum concurrent connections
    pub max_connections: usize,
}

impl Default for StratumConfig {
    fn default() -> Self {
        Self {
            port: DEFAULT_STRATUM_PORT,
            bind_address: "0.0.0.0".to_string(),
            max_connections: 10_000,
        }
    }
}

/// Fee configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeConfig {
    /// Pool operator fee in basis points (100 = 1%)
    pub pool_fee_bps: u64,

    /// Enable dev fee (1% protocol fee)
    pub dev_fee_enabled: bool,

    /// Enable promotional zero-fee period
    pub promotional_period: bool,

    /// Promotional period end timestamp
    pub promotional_end: Option<u64>,
}

impl Default for FeeConfig {
    fn default() -> Self {
        Self {
            pool_fee_bps: DEFAULT_POOL_FEE_BPS,
            dev_fee_enabled: true,
            promotional_period: false,
            promotional_end: None,
        }
    }
}

impl FeeConfig {
    /// Get effective pool fee considering promotional period
    pub fn effective_fee_bps(&self) -> u64 {
        if self.promotional_period {
            if let Some(end) = self.promotional_end {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if now < end {
                    return 0;
                }
            } else {
                return 0;
            }
        }
        self.pool_fee_bps
    }
}

/// PPLNS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPLNSConfig {
    /// N-factor (window size multiplier)
    pub n_factor: f64,

    /// Maximum shares to keep in memory
    pub max_shares_in_memory: usize,

    /// Enable share persistence to disk
    pub persist_shares: bool,
}

impl Default for PPLNSConfig {
    fn default() -> Self {
        Self {
            n_factor: DEFAULT_PPLNS_N_FACTOR,
            max_shares_in_memory: 1_000_000,
            persist_shares: true,
        }
    }
}

/// Variable difficulty configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VardiffConfig {
    /// Enable vardiff
    pub enabled: bool,

    /// Initial share difficulty
    pub initial_difficulty: f64,

    /// Target time between shares (seconds)
    pub target_time_seconds: f64,

    /// Acceptable variance percentage (0.25 = 25%)
    pub variance_percent: f64,

    /// Minimum share difficulty
    pub min_difficulty: f64,

    /// Maximum share difficulty
    pub max_difficulty: f64,

    /// Retarget interval (seconds)
    pub retarget_interval_seconds: f64,
}

impl Default for VardiffConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_difficulty: 0.001,
            target_time_seconds: DEFAULT_VARDIFF_TARGET_TIME,
            variance_percent: 0.25,
            min_difficulty: 0.0001,
            max_difficulty: 1000.0,
            retarget_interval_seconds: 60.0,
        }
    }
}

/// Payout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoutConfig {
    /// Minimum payout threshold (atomic units)
    pub min_payout: u64,

    /// Payout interval
    pub interval: PayoutInterval,

    /// Maximum payouts per batch transaction
    pub max_batch_size: usize,

    /// Enable automatic payouts
    pub auto_payout: bool,
}

impl Default for PayoutConfig {
    fn default() -> Self {
        Self {
            min_payout: DEFAULT_MIN_PAYOUT,
            interval: PayoutInterval::Hourly,
            max_batch_size: 100,
            auto_payout: true,
        }
    }
}

/// Payout interval configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PayoutInterval {
    /// Payout immediately when block found
    Immediate,

    /// Payout hourly
    Hourly,

    /// Payout daily
    Daily,

    /// Payout every N blocks
    EveryNBlocks(u64),

    /// Payout when threshold reached
    Threshold(u64),
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Rate limit: shares per second per worker
    pub rate_limit_shares_per_second: f64,

    /// Maximum invalid shares before ban
    pub max_invalid_shares_before_ban: u32,

    /// Ban duration (seconds)
    pub ban_duration_seconds: u64,

    /// Require worker names (not just wallet)
    pub require_worker_names: bool,

    /// Maximum connections per IP
    pub max_connections_per_ip: u32,

    /// Enable block withholding detection
    pub detect_block_withholding: bool,

    /// Block withholding detection threshold (standard deviations)
    pub withholding_threshold_sigma: f64,

    /// Require valid wallet address format
    pub validate_wallet_format: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            rate_limit_shares_per_second: 10.0,
            max_invalid_shares_before_ban: 100,
            ban_duration_seconds: 3600,
            require_worker_names: false,
            max_connections_per_ip: 50,
            detect_block_withholding: true,
            withholding_threshold_sigma: 3.0,
            validate_wallet_format: true,
        }
    }
}

impl SecurityConfig {
    /// Get ban duration as Duration
    pub fn ban_duration(&self) -> Duration {
        Duration::from_secs(self.ban_duration_seconds)
    }
}

mod humantime_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PoolConfig::default();
        assert_eq!(config.stratum.port, 3333);
        assert_eq!(config.fees.pool_fee_bps, 250); // v8.6.0: raised to 2.5%
        assert_eq!(config.vardiff.target_time_seconds, 20.0);
    }

    #[test]
    fn test_promotional_fee() {
        let mut config = FeeConfig::default();

        // Normal fee
        assert_eq!(config.effective_fee_bps(), 250); // v8.6.0: raised to 2.5%

        // Promotional with no end date = 0%
        config.promotional_period = true;
        config.promotional_end = None;
        assert_eq!(config.effective_fee_bps(), 0);

        // Promotional with future end date = 0%
        let future = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;
        config.promotional_end = Some(future);
        assert_eq!(config.effective_fee_bps(), 0);

        // Promotional with past end date = normal fee
        config.promotional_end = Some(0);
        assert_eq!(config.effective_fee_bps(), 250); // v8.6.0: raised to 2.5%
    }
}
