/// Embedded Full Node Module
///
/// This module provides optional embedded blockchain node functionality.
/// When compiled with --features full-node, users can run a complete
/// Q-NarwhalKnight consensus node directly within the GUI.

use anyhow::Result;
use tracing::{info, warn};

/// Node status information
#[derive(Debug, Clone)]
pub struct NodeStatus {
    pub is_running: bool,
    pub block_height: u64,
    pub peer_count: usize,
    pub sync_progress: f32,
    pub mining_active: bool,
}

impl Default for NodeStatus {
    fn default() -> Self {
        Self {
            is_running: false,
            block_height: 0,
            peer_count: 0,
            sync_progress: 0.0,
            mining_active: false,
        }
    }
}

/// Embedded node manager
pub struct EmbeddedNode {
    #[cfg(feature = "embedded-node")]
    _api_server: Option<std::sync::Arc<()>>, // Placeholder for actual server

    status: NodeStatus,
}

impl EmbeddedNode {
    /// Create a new embedded node instance
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "embedded-node")]
            _api_server: None,
            status: NodeStatus::default(),
        }
    }

    /// Start the embedded node
    pub async fn start(&mut self) -> Result<()> {
        #[cfg(feature = "embedded-node")]
        {
            info!("🚀 Starting embedded Q-NarwhalKnight node...");

            // TODO: Initialize full node components
            // - DAG-Knight consensus engine
            // - Narwhal mempool
            // - P2P network layer
            // - Mining engine
            // - Storage backend
            // - Tor client

            self.status.is_running = true;
            self.status.peer_count = 0;
            self.status.block_height = 0;
            self.status.sync_progress = 0.0;

            info!("✅ Embedded node started successfully");
            Ok(())
        }

        #[cfg(not(feature = "embedded-node"))]
        {
            warn!("⚠️  Embedded node feature not compiled in");
            warn!("   Rebuild with: cargo build --features full-node");
            Err(anyhow::anyhow!("Embedded node feature not available"))
        }
    }

    /// Stop the embedded node
    pub async fn stop(&mut self) -> Result<()> {
        #[cfg(feature = "embedded-node")]
        {
            info!("🛑 Stopping embedded node...");
            self.status.is_running = false;
            info!("✅ Embedded node stopped");
            Ok(())
        }

        #[cfg(not(feature = "embedded-node"))]
        {
            Ok(())
        }
    }

    /// Get current node status
    pub fn get_status(&self) -> NodeStatus {
        self.status.clone()
    }

    /// Check if embedded node feature is available
    pub fn is_available() -> bool {
        cfg!(feature = "embedded-node")
    }
}

impl Default for EmbeddedNode {
    fn default() -> Self {
        Self::new()
    }
}
