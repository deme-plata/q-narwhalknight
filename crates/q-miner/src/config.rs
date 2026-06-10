use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, debug};

/// Complete miner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerConfig {
    pub mining: MiningConfig,
    pub hardware: HardwareConfig,
    pub network: NetworkConfig,
    pub pool: PoolConfig,
    pub wallet: WalletConfig,
    pub ui: UiConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningConfig {
    pub algorithm: String,
    pub intensity: u8,
    pub auto_tune: bool,
    pub difficulty_target: Option<String>,
    pub enable_cpu: bool,
    pub enable_gpu: bool,
    pub max_temperature: f64,
    pub power_limit: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub cpu_threads: usize,  // 0 = auto-detect
    pub gpu_devices: Vec<u32>,
    pub cuda_enabled: bool,
    pub opencl_enabled: bool,
    pub memory_limit_gb: f64,
    pub thermal_throttle: bool,
    pub power_efficiency_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub mode: String,  // "solo" or "pool"
    pub tor_enabled: bool,
    pub p2p_enabled: bool,
    pub max_peers: u32,
    pub connection_timeout: u64,
    pub retry_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub url: Option<String>,
    pub backup_urls: Vec<String>,
    pub worker_name: Option<String>,
    pub failover_enabled: bool,
    pub submit_stale: bool,
    pub difficulty_adjustment: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletConfig {
    pub address: Option<String>,
    pub private_key_encrypted: Option<String>,
    pub derivation_path: Option<String>,
    pub auto_create: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    pub mode: String,  // "cli", "gui", "web", "headless"
    pub web_port: u16,
    pub update_interval: u64,
    pub show_advanced_stats: bool,
    pub theme: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file_enabled: bool,
    pub file_path: Option<PathBuf>,
    pub console_enabled: bool,
    pub json_format: bool,
}

impl Default for MinerConfig {
    fn default() -> Self {
        Self {
            mining: MiningConfig {
                algorithm: "dag-knight-vdf".to_string(),
                intensity: 7,
                auto_tune: true,
                difficulty_target: None,
                enable_cpu: true,
                enable_gpu: true,
                max_temperature: 85.0,
                power_limit: None,
            },
            hardware: HardwareConfig {
                cpu_threads: 0, // Auto-detect
                gpu_devices: vec![0], // Use first GPU
                cuda_enabled: true,
                opencl_enabled: true,
                memory_limit_gb: 8.0,
                thermal_throttle: true,
                power_efficiency_mode: false,
            },
            network: NetworkConfig {
                mode: "pool".to_string(),
                tor_enabled: true,
                p2p_enabled: true,
                max_peers: 32,
                connection_timeout: 30,
                retry_attempts: 3,
            },
            pool: PoolConfig {
                url: Some("stratum+tor://pool.qnarwhal.onion:4444".to_string()),
                backup_urls: vec![
                    "stratum+tor://qmu.onion:3333".to_string(),
                    "stratum+tor://ahc.onion:5555".to_string(),
                ],
                worker_name: None, // Auto-generate
                failover_enabled: true,
                submit_stale: false,
                difficulty_adjustment: true,
            },
            wallet: WalletConfig {
                address: None, // Must be provided by user
                private_key_encrypted: None,
                derivation_path: Some("m/44'/0'/0'/0/0".to_string()),
                auto_create: false,
            },
            ui: UiConfig {
                mode: "cli".to_string(),
                web_port: 8090,
                update_interval: 1000, // 1 second
                show_advanced_stats: false,
                theme: "dark".to_string(),
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                file_enabled: true,
                file_path: None, // Use default
                console_enabled: true,
                json_format: false,
            },
        }
    }
}

impl MinerConfig {
    /// Load configuration from file or create default
    pub async fn load(config_path: Option<&str>) -> Result<Self> {
        let config_file = if let Some(path) = config_path {
            PathBuf::from(path)
        } else {
            Self::default_config_path()?
        };
        
        if config_file.exists() {
            info!("📖 Loading miner configuration from: {}", config_file.display());
            let content = tokio::fs::read_to_string(&config_file).await?;
            let config: MinerConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            info!("📝 Creating default miner configuration");
            let config = Self::default();
            config.save(Some(config_file.to_str().unwrap())).await?;
            Ok(config)
        }
    }
    
    /// Save configuration to file
    pub async fn save(&self, config_path: Option<&str>) -> Result<()> {
        let config_file = if let Some(path) = config_path {
            PathBuf::from(path)
        } else {
            Self::default_config_path()?
        };
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = config_file.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        
        let content = toml::to_string_pretty(self)?;
        tokio::fs::write(&config_file, content).await?;
        
        info!("💾 Configuration saved to: {}", config_file.display());
        Ok(())
    }
    
    /// Get default configuration file path
    fn default_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?;
        
        Ok(config_dir.join("q-miner").join("config.toml"))
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate mining config
        if self.mining.intensity > 10 {
            return Err(anyhow::anyhow!("Mining intensity must be between 1-10"));
        }
        
        if self.mining.max_temperature < 40.0 || self.mining.max_temperature > 100.0 {
            return Err(anyhow::anyhow!("Max temperature must be between 40-100°C"));
        }
        
        // Validate hardware config
        if self.hardware.memory_limit_gb < 0.5 {
            return Err(anyhow::anyhow!("Memory limit must be at least 0.5GB"));
        }
        
        // Validate network config
        if !["solo", "pool"].contains(&self.network.mode.as_str()) {
            return Err(anyhow::anyhow!("Network mode must be 'solo' or 'pool'"));
        }
        
        // Validate pool config for pool mode
        if self.network.mode == "pool" && self.pool.url.is_none() {
            return Err(anyhow::anyhow!("Pool URL required for pool mining"));
        }
        
        // Validate wallet config
        if self.wallet.address.is_none() && !self.wallet.auto_create {
            return Err(anyhow::anyhow!("Wallet address required or enable auto_create"));
        }
        
        Ok(())
    }
    
    /// Auto-tune configuration based on hardware
    pub async fn auto_tune(&mut self) -> Result<()> {
        info!("🔧 Auto-tuning miner configuration...");
        
        // Auto-detect CPU threads
        if self.hardware.cpu_threads == 0 {
            self.hardware.cpu_threads = num_cpus::get();
            debug!("Auto-detected {} CPU threads", self.hardware.cpu_threads);
        }
        
        // Auto-detect GPU devices
        if self.hardware.gpu_devices.is_empty() {
            #[cfg(feature = "cuda-mining")]
            {
                if let Ok(cuda_devices) = crate::gpu::cuda::detect_cuda_devices().await {
                    self.hardware.gpu_devices = cuda_devices.iter()
                        .map(|d| d.device_id)
                        .collect();
                    debug!("Auto-detected {} CUDA devices", cuda_devices.len());
                }
            }
        }
        
        // Adjust memory limit based on available memory
        let system_memory_gb = sysinfo::System::new_all().total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        if self.hardware.memory_limit_gb > system_memory_gb * 0.8 {
            self.hardware.memory_limit_gb = system_memory_gb * 0.8;
            debug!("Adjusted memory limit to {:.1}GB", self.hardware.memory_limit_gb);
        }
        
        // Auto-tune mining intensity based on hardware
        if self.mining.auto_tune {
            self.mining.intensity = self.calculate_optimal_intensity().await;
            debug!("Auto-tuned mining intensity to {}", self.mining.intensity);
        }
        
        info!("✅ Auto-tuning completed");
        Ok(())
    }
    
    async fn calculate_optimal_intensity(&self) -> u8 {
        // Calculate optimal intensity based on hardware capabilities
        let mut intensity: u8 = 5; // Base intensity
        
        // Increase for high-end CPUs
        if self.hardware.cpu_threads >= 16 {
            intensity += 1;
        }
        if self.hardware.cpu_threads >= 32 {
            intensity += 1;
        }
        
        // Increase for high-end GPUs
        if !self.hardware.gpu_devices.is_empty() {
            intensity += 2;
        }
        
        // Adjust for power efficiency mode
        if self.hardware.power_efficiency_mode {
            intensity = intensity.saturating_sub(1);
        }
        
        // Clamp to valid range
        intensity.clamp(1, 10)
    }
    
    /// Create configuration wizard for first-time setup
    pub async fn setup_wizard() -> Result<Self> {
        use dialoguer::{Input, Select, Confirm};
        
        println!("🧙 Quillon Miner Setup Wizard");
        println!("=====================================");
        
        // Mining mode selection
        let mining_modes = &["Pool Mining (Recommended)", "Solo Mining"];
        let mode_selection = Select::new()
            .with_prompt("Select mining mode")
            .items(mining_modes)
            .default(0)
            .interact()?;
        
        let mode = if mode_selection == 0 { "pool" } else { "solo" };
        
        // Hardware configuration
        let enable_cpu = Confirm::new()
            .with_prompt("Enable CPU mining?")
            .default(true)
            .interact()?;
        
        let enable_gpu = Confirm::new()
            .with_prompt("Enable GPU mining?")
            .default(true)
            .interact()?;
        
        // Wallet configuration
        let wallet_address: String = Input::new()
            .with_prompt("Enter your Quillon wallet address")
            .interact_text()?;
        
        // Pool configuration (if pool mode)
        let pool_url = if mode == "pool" {
            let pools = vec![
                "stratum+tor://pool.qnarwhal.onion:4444 (Official)",
                "stratum+tor://qmu.onion:3333 (Quantum Miners United)",
                "stratum+tor://ahc.onion:5555 (Anonymous Hash Collective)",
                "Custom URL..."
            ];
            
            let pool_selection = Select::new()
                .with_prompt("Select mining pool")
                .items(&pools)
                .default(0)
                .interact()?;
            
            if pool_selection == 3 {
                Some(Input::new()
                    .with_prompt("Enter custom pool URL")
                    .interact_text()?)
            } else {
                Some(pools[pool_selection].split(' ').next().unwrap().to_string())
            }
        } else {
            None
        };
        
        // Anonymity settings
        let tor_enabled = Confirm::new()
            .with_prompt("Enable Tor anonymity (recommended)?")
            .default(true)
            .interact()?;
        
        // Create configuration
        let mut config = Self::default();
        config.network.mode = mode.to_string();
        config.mining.enable_cpu = enable_cpu;
        config.mining.enable_gpu = enable_gpu;
        config.wallet.address = Some(wallet_address);
        config.pool.url = pool_url;
        config.network.tor_enabled = tor_enabled;
        
        // Auto-tune based on selections
        config.auto_tune().await?;
        
        println!("\n✅ Configuration wizard completed!");
        println!("💾 Configuration will be saved to: {}", 
            Self::default_config_path()?.display());
        
        Ok(config)
    }
}

/// Environment-specific configuration overrides
impl MinerConfig {
    /// Load configuration with environment variable overrides
    pub async fn load_with_env_overrides(config_path: Option<&str>) -> Result<Self> {
        let mut config = Self::load(config_path).await?;
        
        // Override with environment variables
        if let Ok(intensity) = std::env::var("Q_MINER_INTENSITY") {
            if let Ok(intensity_val) = intensity.parse::<u8>() {
                config.mining.intensity = intensity_val;
                debug!("Environment override: intensity = {}", intensity_val);
            }
        }
        
        if let Ok(threads) = std::env::var("Q_MINER_THREADS") {
            if let Ok(threads_val) = threads.parse::<usize>() {
                config.hardware.cpu_threads = threads_val;
                debug!("Environment override: cpu_threads = {}", threads_val);
            }
        }
        
        if let Ok(pool_url) = std::env::var("Q_MINER_POOL") {
            config.pool.url = Some(pool_url.clone());
            debug!("Environment override: pool_url = {}", pool_url);
        }
        
        if let Ok(wallet) = std::env::var("Q_MINER_WALLET") {
            config.wallet.address = Some(wallet.clone());
            debug!("Environment override: wallet = {}", wallet);
        }
        
        if let Ok(tor_enabled) = std::env::var("Q_MINER_TOR") {
            config.network.tor_enabled = tor_enabled.to_lowercase() == "true";
            debug!("Environment override: tor_enabled = {}", config.network.tor_enabled);
        }
        
        Ok(config)
    }
}

/// Configuration validation and sanitization
impl MinerConfig {
    /// Sanitize configuration for security
    pub fn sanitize(&mut self) {
        // Ensure safe temperature limits
        if self.mining.max_temperature > 95.0 {
            self.mining.max_temperature = 95.0;
        }
        
        // Clamp intensity to safe range
        self.mining.intensity = self.mining.intensity.clamp(1, 10);
        
        // Validate pool URLs
        if let Some(ref pool_url) = self.pool.url {
            if !pool_url.starts_with("stratum+") {
                self.pool.url = Some(format!("stratum+tcp://{}", pool_url));
            }
        }
        
        // Sanitize backup URLs
        self.pool.backup_urls = self.pool.backup_urls
            .iter()
            .filter(|url| url.starts_with("stratum+"))
            .cloned()
            .collect();
        
        // Ensure reasonable peer limits
        self.network.max_peers = self.network.max_peers.clamp(1, 100);
        
        // Validate memory limits
        self.hardware.memory_limit_gb = self.hardware.memory_limit_gb.max(0.5);
    }
    
    /// Export configuration for sharing/backup
    pub async fn export(&self) -> Result<String> {
        let mut export_config = self.clone();
        
        // Remove sensitive information
        export_config.wallet.private_key_encrypted = None;
        export_config.wallet.address = Some("YOUR_WALLET_ADDRESS_HERE".to_string());
        
        Ok(toml::to_string_pretty(&export_config)?)
    }
}