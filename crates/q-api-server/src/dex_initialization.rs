//! DEX Component Initialization Module
//!
//! Handles initialization of all DEX components including token registry,
//! price history, DEX manager, and oracle price bridge.

use anyhow::Result;
use q_dex::QuantumDexManager;
use q_storage::{price_history::PriceHistoryManager, token_registry::TokenRegistry, QStorage};
use q_types::{NodeId, Phase};
use std::sync::Arc;
use tracing::info;

/// Container for all initialized DEX components
pub struct DexComponents {
    pub token_registry: Arc<TokenRegistry>,
    pub price_history: Arc<PriceHistoryManager>,
    pub dex_manager: Arc<QuantumDexManager>,
    pub price_bridge: Option<Arc<()>>, // Placeholder - will be implemented with oracle integration
}

/// Initialize all DEX components with proper dependencies
pub async fn initialize_dex_components(
    storage: &Arc<QStorage>,
    _node_id: NodeId,
    _phase: Phase,
    _enable_oracle: bool,
) -> Result<DexComponents> {
    let rocks_db_kv = storage.get_hot_db();
    let raw_db = rocks_db_kv.get_raw_db();
    info!("🌊 Initializing DEX components...");

    // Initialize token registry
    info!("  → Creating token registry...");
    let token_registry = Arc::new(TokenRegistry::new(raw_db.clone()));
    token_registry.initialize().await?;

    // Initialize price history manager
    info!("  → Creating price history manager...");
    let price_history = Arc::new(PriceHistoryManager::new(raw_db.clone()));
    price_history.initialize().await?;

    // Initialize DEX manager
    info!("  → Creating DEX manager...");
    let dex_manager = Arc::new(QuantumDexManager::new(
        token_registry.clone(),
        price_history.clone(),
    )?);
    dex_manager.initialize().await?;

    // Oracle price bridge will be initialized when oracle is enabled
    info!("  → Oracle integration: Not yet implemented");
    let price_bridge = None;

    info!("✅ DEX components initialized successfully");

    Ok(DexComponents {
        token_registry,
        price_history,
        dex_manager,
        price_bridge,
    })
}
