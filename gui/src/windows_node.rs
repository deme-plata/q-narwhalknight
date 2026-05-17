/// Q-NarwhalKnight Windows GUI with Embedded Full Node
/// Complete quantum consensus system with GUI interface for Windows
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};

#[cfg(feature = "embedded-node")]
use {
    q_api_server::ApiServer, q_dag_knight::DAGKnightConsensus, q_mining::MiningNode,
    q_narwhal_core::NarwhalCore, q_network::QuantumNetwork, q_storage::QuantumStorage, q_types::*,
    q_wallet::WalletManager,
};

// Import the GUI components
use crate::{AppState, MainWindow};

/// Windows node with embedded full Q-NarwhalKnight functionality
pub struct WindowsNode {
    #[cfg(feature = "embedded-node")]
    consensus: Arc<DAGKnightConsensus>,
    #[cfg(feature = "embedded-node")]
    mempool: Arc<NarwhalCore>,
    #[cfg(feature = "embedded-node")]
    network: Arc<Mutex<QuantumNetwork>>,
    #[cfg(feature = "embedded-node")]
    api_server: Arc<ApiServer>,
    #[cfg(feature = "embedded-node")]
    wallet_manager: Arc<WalletManager>,
    #[cfg(feature = "embedded-node")]
    storage: Arc<QuantumStorage>,
    #[cfg(feature = "embedded-node")]
    mining_node: Option<Arc<MiningNode>>,

    // GUI state
    gui_state: Arc<Mutex<AppState>>,
    node_id: NodeId,
    config: WindowsNodeConfig,
}

#[derive(Debug, Clone)]
pub struct WindowsNodeConfig {
    pub data_dir: String,
    pub listen_port: u16,
    pub api_port: u16,
    pub mining_enabled: bool,
    pub tor_enabled: bool,
    pub auto_start: bool,
}

impl Default for WindowsNodeConfig {
    fn default() -> Self {
        Self {
            data_dir: std::env::var("APPDATA")
                .map(|appdata| format!("{}\\Q-NarwhalKnight", appdata))
                .unwrap_or_else(|_| ".\\q-narwhal-data".to_string()),
            listen_port: 7000,
            api_port: 3030,
            mining_enabled: true,
            tor_enabled: false, // Default off for Windows
            auto_start: true,
        }
    }
}

impl WindowsNode {
    /// Create new Windows node with embedded functionality
    pub async fn new(config: WindowsNodeConfig) -> Result<Self> {
        info!("🪟 Initializing Q-NarwhalKnight Windows Node");

        // Generate or load node ID
        let node_id = Self::load_or_generate_node_id(&config.data_dir).await?;

        // Create GUI state
        let gui_state = Arc::new(Mutex::new(AppState::new()?));

        #[cfg(feature = "embedded-node")]
        {
            // Initialize storage
            let storage =
                Arc::new(QuantumStorage::new(&format!("{}\\storage", config.data_dir)).await?);

            // Initialize wallet manager
            let wallet_manager = Arc::new(WalletManager::new(storage.clone()).await?);

            // Initialize consensus components
            let consensus = Arc::new(DAGKnightConsensus::new(node_id, 1)?);
            let mempool = Arc::new(NarwhalCore::new(node_id));

            // Initialize networking
            let network = Arc::new(Mutex::new(QuantumNetwork::new_phase0(node_id).await?));

            // Initialize API server with internal address
            let api_server = Arc::new(
                ApiServer::new(
                    format!("127.0.0.1:{}", config.api_port),
                    consensus.clone(),
                    mempool.clone(),
                    wallet_manager.clone(),
                )
                .await?,
            );

            // Initialize mining if enabled
            let mining_node = if config.mining_enabled {
                Some(Arc::new(
                    MiningNode::new(node_id, consensus.clone(), mempool.clone()).await?,
                ))
            } else {
                None
            };

            Ok(Self {
                consensus,
                mempool,
                network,
                api_server,
                wallet_manager,
                storage,
                mining_node,
                gui_state,
                node_id,
                config,
            })
        }

        #[cfg(not(feature = "embedded-node"))]
        {
            // GUI-only mode (connects to external node)
            Ok(Self {
                gui_state,
                node_id,
                config,
            })
        }
    }

    /// Start the Windows node (both backend and GUI)
    pub async fn start(&mut self) -> Result<()> {
        info!("🚀 Starting Q-NarwhalKnight Windows Node");

        #[cfg(feature = "embedded-node")]
        {
            // Start backend services
            self.start_backend_services().await?;

            // Start mining if enabled
            if let Some(mining_node) = &self.mining_node {
                tokio::spawn({
                    let mining_node = mining_node.clone();
                    async move {
                        if let Err(e) = mining_node.start_mining().await {
                            error!("Mining failed: {}", e);
                        }
                    }
                });
            }
        }

        // Start GUI
        self.start_gui().await?;

        Ok(())
    }

    #[cfg(feature = "embedded-node")]
    async fn start_backend_services(&self) -> Result<()> {
        info!("🔧 Starting embedded node services");

        // Start API server
        tokio::spawn({
            let api_server = self.api_server.clone();
            async move {
                if let Err(e) = api_server.start().await {
                    error!("API server failed: {}", e);
                }
            }
        });

        // Start networking
        tokio::spawn({
            let network = self.network.clone();
            async move {
                let mut net = network.lock().await;
                if let Err(e) = net.run().await {
                    error!("Network failed: {}", e);
                }
            }
        });

        // Start consensus engine
        tokio::spawn({
            let consensus = self.consensus.clone();
            async move {
                // Consensus event loop would go here
                loop {
                    if let Err(e) = consensus.advance_round().await {
                        error!("Consensus round failed: {}", e);
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }
        });

        info!("✅ Backend services started");
        Ok(())
    }

    async fn start_gui(&self) -> Result<()> {
        info!("🎨 Starting Windows GUI interface");

        // Set up GUI with local API connection
        {
            let mut gui = self.gui_state.lock().await;
            #[cfg(feature = "embedded-node")]
            {
                // Connect to local embedded API
                gui.api_base = format!("http://127.0.0.1:{}/api/v1", self.config.api_port);
            }
            #[cfg(not(feature = "embedded-node"))]
            {
                // Use external API (configurable)
                gui.api_base = "http://localhost:3030/api/v1".to_string();
            }
        }

        // Run the GUI (this will block)
        let ui = {
            let gui = self.gui_state.lock().await;
            gui.ui.clone_strong()
        };

        // Configure GUI for Windows
        #[cfg(target_os = "windows")]
        {
            ui.set_current_phase("Phase 1 (Windows)".into());
        }

        ui.run()?;

        Ok(())
    }

    async fn load_or_generate_node_id(data_dir: &str) -> Result<NodeId> {
        use std::path::Path;

        let node_id_path = format!("{}\\node_id", data_dir);

        if Path::new(&node_id_path).exists() {
            // Load existing node ID
            let id_bytes = std::fs::read(&node_id_path)?;
            if id_bytes.len() == 32 {
                let mut node_id = [0u8; 32];
                node_id.copy_from_slice(&id_bytes);
                Ok(node_id)
            } else {
                Self::generate_and_save_node_id(&node_id_path).await
            }
        } else {
            // Create data directory if it doesn't exist
            std::fs::create_dir_all(data_dir)?;
            Self::generate_and_save_node_id(&node_id_path).await
        }
    }

    async fn generate_and_save_node_id(path: &str) -> Result<NodeId> {
        use rand::RngCore;

        let mut node_id = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut node_id);

        std::fs::write(path, &node_id)?;
        info!("Generated new node ID: {}", hex::encode(&node_id[..8]));

        Ok(node_id)
    }

    /// Get node status for GUI display
    pub async fn get_node_status(&self) -> NodeStatus {
        #[cfg(feature = "embedded-node")]
        {
            let consensus_status = self.consensus.get_status().await;
            let network_stats = {
                let network = self.network.lock().await;
                network.get_network_stats().await
            };

            NodeStatus {
                node_id: hex::encode(&self.node_id[..8]),
                running: true,
                consensus_round: consensus_status.current_round,
                connected_peers: network_stats.connected_peers,
                mining_active: self.mining_node.is_some(),
                uptime: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default(),
            }
        }

        #[cfg(not(feature = "embedded-node"))]
        {
            NodeStatus {
                node_id: hex::encode(&self.node_id[..8]),
                running: false,
                consensus_round: 0,
                connected_peers: 0,
                mining_active: false,
                uptime: std::time::Duration::default(),
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct NodeStatus {
    pub node_id: String,
    pub running: bool,
    pub consensus_round: u64,
    pub connected_peers: u64,
    pub mining_active: bool,
    pub uptime: std::time::Duration,
}

/// Windows-specific main function for embedded node
#[cfg(feature = "embedded-node")]
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize Windows-specific logging
    tracing_subscriber::fmt()
        .with_env_filter("qnk_windows=info,qnk_gui=debug")
        .init();

    info!("🪟 Starting Q-NarwhalKnight Windows Full Node");

    // Load configuration (could be from registry or config file)
    let config = WindowsNodeConfig::default();

    // Create and start the Windows node
    let mut windows_node = WindowsNode::new(config).await?;
    windows_node.start().await?;

    info!("👋 Q-NarwhalKnight Windows Node shutdown complete");
    Ok(())
}

/// GUI-only main function (for development)
#[cfg(not(feature = "embedded-node"))]
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("qnk_gui=debug,info")
        .init();

    info!("🎨 Starting Q-NarwhalKnight GUI (External Node Mode)");

    let config = WindowsNodeConfig::default();
    let mut windows_node = WindowsNode::new(config).await?;
    windows_node.start().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_windows_node_creation() {
        let config = WindowsNodeConfig::default();
        let result = WindowsNode::new(config).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_defaults() {
        let config = WindowsNodeConfig::default();
        assert_eq!(config.listen_port, 7000);
        assert_eq!(config.api_port, 3030);
        assert!(config.mining_enabled);
        assert!(!config.tor_enabled); // Default off for Windows
    }
}
