use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use tracing::info;

/// Configuration for robot control system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotConfig {
    /// Individual robot configurations
    pub robots: HashMap<String, RobotConfigEntry>,
    /// Network configuration
    pub network: NetworkConfig,
    /// Quantum system configuration
    pub quantum: QuantumConfig,
    /// Security configuration
    pub security: SecurityConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Configuration for individual robots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotConfigEntry {
    /// Robot type identifier
    pub robot_type: String,
    /// Connection endpoint (IP:port or URL)
    pub endpoint: String,
    /// Authentication credentials
    pub auth: AuthConfig,
    /// Robot-specific capabilities
    pub capabilities: Vec<String>,
    /// Operational parameters
    pub parameters: HashMap<String, f64>,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// libp2p listening addresses
    pub listen_addresses: Vec<String>,
    /// Bootstrap nodes for peer discovery
    pub bootstrap_nodes: Vec<String>,
    /// Maximum number of connections
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connection_timeout: u64,
    /// Enable quantum-secure communication
    pub quantum_secure: bool,
}

/// Quantum system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    /// Enable quantum random number generation
    pub enable_qrng: bool,
    /// Quantum coherence monitoring interval (seconds)
    pub coherence_monitor_interval: f64,
    /// Entanglement network configuration
    pub entanglement: EntanglementConfig,
    /// Quantum visualization settings
    pub visualization: VisualizationConfig,
}

/// Entanglement network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementConfig {
    /// Enable automatic entanglement establishment
    pub auto_establish: bool,
    /// Target entanglement fidelity (0.0-1.0)
    pub target_fidelity: f64,
    /// Decoherence monitoring threshold
    pub decoherence_threshold: f64,
    /// Bell state type for swarm entanglement
    pub bell_state_type: String,
}

/// Quantum visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Enable real-time quantum state visualization
    pub enable_realtime: bool,
    /// Visualization update frequency (Hz)
    pub update_frequency: f64,
    /// Color scheme for quantum states
    pub color_scheme: String,
    /// Export format for visualization data
    pub export_format: String,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Post-quantum cryptography settings
    pub post_quantum: PostQuantumConfig,
    /// Authentication configuration
    pub authentication: AuthenticationConfig,
    /// Access control settings
    pub access_control: AccessControlConfig,
}

/// Post-quantum cryptography configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostQuantumConfig {
    /// Enable post-quantum signatures
    pub enable_signatures: bool,
    /// Signature algorithm (Dilithium5, etc.)
    pub signature_algorithm: String,
    /// Enable post-quantum key exchange
    pub enable_key_exchange: bool,
    /// Key exchange algorithm (Kyber1024, etc.)
    pub key_exchange_algorithm: String,
    /// Hybrid mode with classical algorithms
    pub hybrid_mode: bool,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Authentication method (certificate, key, token)
    pub method: String,
    /// Certificate or key file path
    pub credential_path: String,
    /// Token refresh interval (seconds)
    pub token_refresh_interval: u64,
    /// Enable multi-factor authentication
    pub enable_mfa: bool,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Admin users list
    pub admin_users: Vec<String>,
    /// Operator users list  
    pub operator_users: Vec<String>,
    /// Read-only users list
    pub readonly_users: Vec<String>,
    /// Permission levels for different operations
    pub permissions: HashMap<String, String>,
}

/// Individual authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type
    pub auth_type: String,
    /// Username or identifier
    pub username: Option<String>,
    /// Password or token
    pub password: Option<String>,
    /// Certificate path
    pub certificate: Option<String>,
    /// Private key path
    pub private_key: Option<String>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Log output format (json, text)
    pub format: String,
    /// Log file path (optional)
    pub file_path: Option<String>,
    /// Enable console logging
    pub console: bool,
    /// Enable system metrics logging
    pub metrics: bool,
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self {
            robots: HashMap::new(),
            network: NetworkConfig::default(),
            quantum: QuantumConfig::default(),
            security: SecurityConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addresses: vec![
                "/ip4/0.0.0.0/tcp/0".to_string(),
                "/ip6/::/tcp/0".to_string(),
            ],
            bootstrap_nodes: vec![],
            max_connections: 100,
            connection_timeout: 30,
            quantum_secure: true,
        }
    }
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            enable_qrng: true,
            coherence_monitor_interval: 1.0,
            entanglement: EntanglementConfig::default(),
            visualization: VisualizationConfig::default(),
        }
    }
}

impl Default for EntanglementConfig {
    fn default() -> Self {
        Self {
            auto_establish: true,
            target_fidelity: 0.9,
            decoherence_threshold: 0.1,
            bell_state_type: "PhiPlus".to_string(),
        }
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            update_frequency: 10.0,
            color_scheme: "rainbow".to_string(),
            export_format: "json".to_string(),
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            post_quantum: PostQuantumConfig::default(),
            authentication: AuthenticationConfig::default(),
            access_control: AccessControlConfig::default(),
        }
    }
}

impl Default for PostQuantumConfig {
    fn default() -> Self {
        Self {
            enable_signatures: true,
            signature_algorithm: "Dilithium5".to_string(),
            enable_key_exchange: true,
            key_exchange_algorithm: "Kyber1024".to_string(),
            hybrid_mode: true,
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            method: "certificate".to_string(),
            credential_path: "./certs/robot-client.pem".to_string(),
            token_refresh_interval: 3600,
            enable_mfa: false,
        }
    }
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            admin_users: vec!["claude".to_string()],
            operator_users: vec!["operator".to_string()],
            readonly_users: vec!["observer".to_string()],
            permissions: [
                ("robot_control".to_string(), "admin,operator".to_string()),
                ("swarm_control".to_string(), "admin,operator".to_string()),
                ("quantum_measure".to_string(), "admin,operator,readonly".to_string()),
                ("system_config".to_string(), "admin".to_string()),
            ].iter().cloned().collect(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "text".to_string(),
            file_path: None,
            console: true,
            metrics: true,
        }
    }
}

impl RobotConfig {
    /// Load configuration from file
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading robot configuration from {:?}", path);
        
        if !path.exists() {
            info!("Configuration file not found, creating default configuration");
            let default_config = Self::default();
            default_config.save(path).await
                .context("Failed to save default configuration")?;
            return Ok(default_config);
        }
        
        let config_data = fs::read_to_string(path).await
            .context("Failed to read configuration file")?;
        
        let config: Self = toml::from_str(&config_data)
            .context("Failed to parse configuration file")?;
        
        info!("Successfully loaded configuration with {} robot entries", 
            config.robots.len());
        
        Ok(config)
    }
    
    /// Save configuration to file
    pub async fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        info!("Saving robot configuration to {:?}", path);
        
        let config_data = toml::to_string_pretty(self)
            .context("Failed to serialize configuration")?;
        
        fs::write(path, config_data).await
            .context("Failed to write configuration file")?;
        
        info!("Successfully saved configuration");
        Ok(())
    }
    
    /// Add robot configuration entry
    pub fn add_robot(&mut self, id: String, config: RobotConfigEntry) {
        info!("Adding robot configuration for {}", id);
        self.robots.insert(id, config);
    }
    
    /// Get robot configuration by ID
    pub fn get_robot(&self, id: &str) -> Option<&RobotConfigEntry> {
        self.robots.get(id)
    }
    
    /// Update network bootstrap nodes
    pub fn set_bootstrap_nodes(&mut self, nodes: Vec<String>) {
        self.network.bootstrap_nodes = nodes;
    }
    
    /// Enable/disable quantum features
    pub fn set_quantum_enabled(&mut self, enabled: bool) {
        self.quantum.enable_qrng = enabled;
        self.quantum.entanglement.auto_establish = enabled;
        self.security.post_quantum.enable_signatures = enabled;
        self.security.post_quantum.enable_key_exchange = enabled;
    }
    
    /// Configure post-quantum cryptography
    pub fn configure_post_quantum(
        &mut self, 
        signature_algorithm: String,
        key_exchange_algorithm: String,
        hybrid_mode: bool
    ) {
        self.security.post_quantum.signature_algorithm = signature_algorithm;
        self.security.post_quantum.key_exchange_algorithm = key_exchange_algorithm;
        self.security.post_quantum.hybrid_mode = hybrid_mode;
    }
    
    /// Set logging configuration
    pub fn set_logging_level(&mut self, level: String) {
        self.logging.level = level;
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate network configuration
        if self.network.listen_addresses.is_empty() {
            return Err(anyhow::anyhow!("No listen addresses configured"));
        }
        
        // Validate quantum configuration
        if self.quantum.entanglement.target_fidelity < 0.0 || 
           self.quantum.entanglement.target_fidelity > 1.0 {
            return Err(anyhow::anyhow!("Target entanglement fidelity must be between 0.0 and 1.0"));
        }
        
        // Validate security configuration
        let valid_signature_algorithms = ["Dilithium2", "Dilithium3", "Dilithium5"];
        if !valid_signature_algorithms.contains(&self.security.post_quantum.signature_algorithm.as_str()) {
            return Err(anyhow::anyhow!("Invalid signature algorithm: {}", 
                self.security.post_quantum.signature_algorithm));
        }
        
        let valid_key_exchange_algorithms = ["Kyber512", "Kyber768", "Kyber1024"];
        if !valid_key_exchange_algorithms.contains(&self.security.post_quantum.key_exchange_algorithm.as_str()) {
            return Err(anyhow::anyhow!("Invalid key exchange algorithm: {}", 
                self.security.post_quantum.key_exchange_algorithm));
        }
        
        // Validate logging configuration
        let valid_log_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_log_levels.contains(&self.logging.level.as_str()) {
            return Err(anyhow::anyhow!("Invalid log level: {}", self.logging.level));
        }
        
        info!("Configuration validation passed");
        Ok(())
    }
    
    /// Create example configuration with sample robots
    pub fn create_example() -> Self {
        let mut config = Self::default();
        
        // Add example robots
        config.add_robot("quantum_jelly_001".to_string(), RobotConfigEntry {
            robot_type: "QuantumJellyfish".to_string(),
            endpoint: "tcp://192.168.1.100:8080".to_string(),
            auth: AuthConfig {
                auth_type: "certificate".to_string(),
                username: Some("jelly_001".to_string()),
                password: None,
                certificate: Some("./certs/jelly_001.pem".to_string()),
                private_key: Some("./certs/jelly_001.key".to_string()),
            },
            capabilities: vec![
                "bioluminescence".to_string(),
                "superposition_glow".to_string(),
                "quantum_sensing".to_string(),
            ],
            parameters: [
                ("max_depth".to_string(), 100.0),
                ("bioluminescence_intensity".to_string(), 1000.0),
                ("coherence_time".to_string(), 0.001),
            ].iter().cloned().collect(),
        });
        
        config.add_robot("dolphin_alpha_002".to_string(), RobotConfigEntry {
            robot_type: "EntangledDolphin".to_string(),
            endpoint: "tcp://192.168.1.101:8080".to_string(),
            auth: AuthConfig {
                auth_type: "certificate".to_string(),
                username: Some("dolphin_002".to_string()),
                password: None,
                certificate: Some("./certs/dolphin_002.pem".to_string()),
                private_key: Some("./certs/dolphin_002.key".to_string()),
            },
            capabilities: vec![
                "quantum_echolocation".to_string(),
                "entanglement_comm".to_string(),
                "swarm_leadership".to_string(),
            ],
            parameters: [
                ("max_speed".to_string(), 15.0),
                ("echolocation_range".to_string(), 500.0),
                ("comm_fidelity".to_string(), 0.95),
            ].iter().cloned().collect(),
        });
        
        config.add_robot("octopus_stealth_003".to_string(), RobotConfigEntry {
            robot_type: "TunnelingOctopus".to_string(),
            endpoint: "tcp://192.168.1.102:8080".to_string(),
            auth: AuthConfig {
                auth_type: "certificate".to_string(),
                username: Some("octopus_003".to_string()),
                password: None,
                certificate: Some("./certs/octopus_003.pem".to_string()),
                private_key: Some("./certs/octopus_003.key".to_string()),
            },
            capabilities: vec![
                "quantum_tunneling".to_string(),
                "phase_camouflage".to_string(),
                "precision_manipulation".to_string(),
            ],
            parameters: [
                ("tunneling_probability".to_string(), 0.15),
                ("camouflage_effectiveness".to_string(), 0.9),
                ("manipulation_precision".to_string(), 0.01),
            ].iter().cloned().collect(),
        });
        
        // Configure example bootstrap nodes
        config.set_bootstrap_nodes(vec![
            "/ip4/192.168.1.1/tcp/4001/p2p/12D3KooWExample1".to_string(),
            "/ip4/192.168.1.2/tcp/4001/p2p/12D3KooWExample2".to_string(),
        ]);
        
        // Enable all quantum features
        config.set_quantum_enabled(true);
        
        // Configure advanced post-quantum cryptography
        config.configure_post_quantum(
            "Dilithium5".to_string(),
            "Kyber1024".to_string(),
            true
        );
        
        // Set debug logging for development
        config.set_logging_level("debug".to_string());
        
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_config_save_and_load() -> Result<()> {
        let dir = tempdir()?;
        let config_path = dir.path().join("test_config.toml");
        
        let original_config = RobotConfig::create_example();
        original_config.save(&config_path).await?;
        
        let loaded_config = RobotConfig::load(&config_path).await?;
        
        assert_eq!(original_config.robots.len(), loaded_config.robots.len());
        assert_eq!(original_config.network.max_connections, loaded_config.network.max_connections);
        assert_eq!(original_config.quantum.enable_qrng, loaded_config.quantum.enable_qrng);
        
        Ok(())
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = RobotConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid fidelity
        config.quantum.entanglement.target_fidelity = 1.5;
        assert!(config.validate().is_err());
        
        // Test invalid algorithm
        config.quantum.entanglement.target_fidelity = 0.9;
        config.security.post_quantum.signature_algorithm = "InvalidAlgorithm".to_string();
        assert!(config.validate().is_err());
    }
}