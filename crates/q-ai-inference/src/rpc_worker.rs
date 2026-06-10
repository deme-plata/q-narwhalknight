//! RPC Worker Manager - Manages llama.cpp RPC server subprocesses for pipeline parallelism
//!
//! v5.1.0: When a node volunteers as an AI worker, this module spawns a `rpc-server`
//! subprocess that exposes GPU/CPU compute to the llama.cpp RPC layer. Remote nodes
//! can then load models with `--rpc worker1:port,worker2:port` to automatically
//! distribute model layers across available workers.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │              Coordinator Node                        │
//! │  LlamaCppEngine::new("model.gguf")                  │
//! │    --rpc worker1:50000,worker2:50001                │
//! │  → Auto-distributes layers by available memory       │
//! └──────────┬───────────────┬──────────────────────────┘
//!            │               │
//!     ┌──────▼───────┐ ┌────▼────────────┐
//!     │ Worker 1     │ │ Worker 2         │
//!     │ rpc-server   │ │ rpc-server       │
//!     │ :50000       │ │ :50001           │
//!     │ Layers 0-15  │ │ Layers 16-31     │
//!     └──────────────┘ └─────────────────┘
//! ```
//!
//! ## llama.cpp RPC Protocol
//!
//! Binary TCP protocol with 17 commands:
//! - `ALLOC_BUFFER`, `GET_ALIGNMENT`, `GET_MAX_SIZE`
//! - `BUFFER_GET_BASE`, `FREE_BUFFER`
//! - `BUFFER_CLEAR`, `SET_TENSOR`, `GET_TENSOR`
//! - `COPY_TENSOR`, `GRAPH_COMPUTE`
//! - `GET_DEVICE_MEMORY`
//!
//! Proven performance: 48 tok/s over Ethernet (vs 50 tok/s local)

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Child;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// RPC worker endpoint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcWorkerInfo {
    /// Peer ID of the worker node
    pub peer_id: String,
    /// IP address of the worker
    pub host: String,
    /// Port the RPC server listens on
    pub port: u16,
    /// Available RAM/VRAM in GB
    pub available_memory_gb: usize,
    /// Whether this is a local worker (same machine)
    pub is_local: bool,
    /// Current status
    pub status: WorkerStatus,
}

/// Worker status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    /// RPC server starting up
    Starting,
    /// Ready to accept connections
    Ready,
    /// Currently processing inference
    Busy,
    /// Worker has stopped
    Stopped,
    /// Worker encountered an error
    Error,
}

/// Manages llama.cpp RPC worker subprocesses.
///
/// Handles spawning, monitoring, and stopping RPC servers that expose local
/// compute resources to the distributed inference network.
pub struct RpcWorkerManager {
    /// Map of peer_id -> subprocess handle for local workers
    local_workers: Arc<RwLock<HashMap<String, RpcWorkerProcess>>>,

    /// Map of peer_id -> worker info for all known workers (local + remote)
    known_workers: Arc<RwLock<HashMap<String, RpcWorkerInfo>>>,

    /// Port allocator (starts at 50000)
    next_port: AtomicU16,

    /// Path to the rpc-server binary
    rpc_server_binary: String,
}

/// Local worker process handle
struct RpcWorkerProcess {
    child: Option<Child>,
    info: RpcWorkerInfo,
}

impl RpcWorkerManager {
    /// Create a new RPC worker manager.
    ///
    /// # Arguments
    /// * `rpc_server_binary` - Path to the llama.cpp `rpc-server` binary.
    ///   If not found, workers can still be registered as remote endpoints.
    pub fn new(rpc_server_binary: Option<String>) -> Self {
        let binary_path = rpc_server_binary.unwrap_or_else(|| {
            // Try common locations
            let candidates = [
                "/opt/orobit/shared/q-narwhalknight/bin/rpc-server",
                "./rpc-server",
                "rpc-server",
            ];
            for candidate in &candidates {
                if std::path::Path::new(candidate).exists() {
                    return candidate.to_string();
                }
            }
            "rpc-server".to_string() // Hope it's in PATH
        });

        Self {
            local_workers: Arc::new(RwLock::new(HashMap::new())),
            known_workers: Arc::new(RwLock::new(HashMap::new())),
            next_port: AtomicU16::new(50000),
            rpc_server_binary: binary_path,
        }
    }

    /// Start a local RPC worker subprocess.
    ///
    /// Spawns a `rpc-server` process that listens on the allocated port.
    /// The worker exposes local compute (CPU/GPU) to the RPC network.
    pub async fn start_local_worker(
        &self,
        peer_id: &str,
        host: &str,
        available_memory_gb: usize,
    ) -> Result<RpcWorkerInfo> {
        let port = self.next_port.fetch_add(1, Ordering::SeqCst);

        info!("🚀 Starting local RPC worker on {}:{}...", host, port);

        // Spawn rpc-server subprocess
        let child = std::process::Command::new(&self.rpc_server_binary)
            .args(&["--host", host, "--port", &port.to_string()])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!(
                "Failed to spawn rpc-server at {}: {}. \
                 Ensure llama.cpp rpc-server binary is installed.",
                self.rpc_server_binary, e
            ))?;

        let info = RpcWorkerInfo {
            peer_id: peer_id.to_string(),
            host: host.to_string(),
            port,
            available_memory_gb,
            is_local: true,
            status: WorkerStatus::Starting,
        };

        info!("✅ Local RPC worker started: {}:{} (PID: {})",
              host, port, child.id());

        let process = RpcWorkerProcess {
            child: Some(child),
            info: info.clone(),
        };

        self.local_workers.write().await.insert(peer_id.to_string(), process);
        self.known_workers.write().await.insert(peer_id.to_string(), info.clone());

        Ok(info)
    }

    /// Register a remote RPC worker (discovered via P2P gossipsub).
    pub async fn register_remote_worker(&self, info: RpcWorkerInfo) {
        info!("📡 Registering remote RPC worker: {}:{} (peer: {})",
              info.host, info.port, info.peer_id);
        self.known_workers.write().await.insert(info.peer_id.clone(), info);
    }

    /// Remove a worker (local or remote).
    pub async fn remove_worker(&self, peer_id: &str) -> Result<()> {
        // Stop local worker if exists
        if let Some(mut process) = self.local_workers.write().await.remove(peer_id) {
            if let Some(ref mut child) = process.child {
                info!("🛑 Stopping local RPC worker for peer {}", peer_id);
                let _ = child.kill();
                let _ = child.wait();
            }
        }

        self.known_workers.write().await.remove(peer_id);
        Ok(())
    }

    /// Build the `--rpc` argument string for llama.cpp model loading.
    ///
    /// Returns a comma-separated list of worker endpoints: "host1:port1,host2:port2"
    /// Only includes workers with status Ready or Busy.
    pub async fn build_rpc_arg(&self) -> Option<String> {
        let workers = self.known_workers.read().await;
        let endpoints: Vec<String> = workers
            .values()
            .filter(|w| matches!(w.status, WorkerStatus::Ready | WorkerStatus::Busy))
            .map(|w| format!("{}:{}", w.host, w.port))
            .collect();

        if endpoints.is_empty() {
            None
        } else {
            Some(endpoints.join(","))
        }
    }

    /// Get all known workers.
    pub async fn get_workers(&self) -> Vec<RpcWorkerInfo> {
        self.known_workers.read().await.values().cloned().collect()
    }

    /// Get count of ready workers.
    pub async fn ready_worker_count(&self) -> usize {
        self.known_workers.read().await
            .values()
            .filter(|w| matches!(w.status, WorkerStatus::Ready))
            .count()
    }

    /// Update worker status.
    pub async fn update_worker_status(&self, peer_id: &str, status: WorkerStatus) {
        if let Some(worker) = self.known_workers.write().await.get_mut(peer_id) {
            worker.status = status;
        }
    }

    /// Stop all local workers (cleanup on shutdown).
    pub async fn stop_all(&self) {
        let mut workers = self.local_workers.write().await;
        for (peer_id, process) in workers.iter_mut() {
            if let Some(ref mut child) = process.child {
                info!("🛑 Stopping RPC worker for peer {}", peer_id);
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        workers.clear();
    }
}

impl Drop for RpcWorkerManager {
    fn drop(&mut self) {
        // Best-effort cleanup of child processes
        if let Ok(mut workers) = self.local_workers.try_write() {
            for (_, process) in workers.iter_mut() {
                if let Some(ref mut child) = process.child {
                    let _ = child.kill();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_info_serialization() {
        let info = RpcWorkerInfo {
            peer_id: "test-peer".to_string(),
            host: "127.0.0.1".to_string(),
            port: 50000,
            available_memory_gb: 16,
            is_local: true,
            status: WorkerStatus::Ready,
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: RpcWorkerInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.port, 50000);
        assert_eq!(parsed.peer_id, "test-peer");
    }

    #[tokio::test]
    async fn test_rpc_arg_building() {
        let manager = RpcWorkerManager::new(None);

        // No workers → None
        assert!(manager.build_rpc_arg().await.is_none());

        // Add a ready worker
        manager.register_remote_worker(RpcWorkerInfo {
            peer_id: "peer1".to_string(),
            host: "192.168.1.10".to_string(),
            port: 50000,
            available_memory_gb: 8,
            is_local: false,
            status: WorkerStatus::Ready,
        }).await;

        manager.register_remote_worker(RpcWorkerInfo {
            peer_id: "peer2".to_string(),
            host: "192.168.1.11".to_string(),
            port: 50001,
            available_memory_gb: 16,
            is_local: false,
            status: WorkerStatus::Ready,
        }).await;

        let rpc_arg = manager.build_rpc_arg().await.unwrap();
        assert!(rpc_arg.contains("192.168.1.10:50000"));
        assert!(rpc_arg.contains("192.168.1.11:50001"));
    }
}
