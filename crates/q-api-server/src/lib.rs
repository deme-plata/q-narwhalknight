use q_types::*;
use q_wallet::{WalletManager, MemoryWalletStore};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub mod handlers;
pub mod config;
pub mod streaming;

pub use config::Config;
pub use streaming::{EventBroadcaster, HighPerformanceEmitter, StreamEvent};

/// Application state shared across handlers
pub struct AppState {
    pub config: Config,
    pub wallet_manager: WalletManager<MemoryWalletStore>,
    pub node_status: Arc<RwLock<NodeStatus>>,
    pub tx_pool: Arc<RwLock<HashMap<TxHash, Transaction>>>,
    pub tx_status: Arc<RwLock<HashMap<TxHash, TxStatus>>>,
    pub blocks: Arc<RwLock<HashMap<Height, Vec<Transaction>>>>,
    pub event_broadcaster: Arc<EventBroadcaster>,
    pub event_emitter: Arc<HighPerformanceEmitter>,
}

impl AppState {
    pub async fn new(config: Config) -> anyhow::Result<Self> {
        let wallet_store = MemoryWalletStore::new();
        let wallet_manager = WalletManager::new(wallet_store);
        
        let node_status = NodeStatus {
            node_id: [0u8; 32], // TODO: Load from config
            current_round: 0,
            current_height: 0,
            connected_peers: 0,
            tx_pool_size: 0,
            is_validator: config.is_validator,
            uptime: std::time::Duration::from_secs(0),
        };

        // Initialize real-time streaming
        let event_broadcaster = Arc::new(EventBroadcaster::new());
        let event_emitter = Arc::new(HighPerformanceEmitter::new(event_broadcaster.clone()));

        Ok(Self {
            config,
            wallet_manager,
            node_status: Arc::new(RwLock::new(node_status)),
            tx_pool: Arc::new(RwLock::new(HashMap::new())),
            tx_status: Arc::new(RwLock::new(HashMap::new())),
            blocks: Arc::new(RwLock::new(HashMap::new())),
            event_broadcaster,
            event_emitter,
        })
    }
}