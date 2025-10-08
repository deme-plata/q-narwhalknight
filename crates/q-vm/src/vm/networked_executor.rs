/// Networked VM Executor - Distributed Smart Contract Execution
///
/// This module provides a VM executor that can execute contracts locally
/// or distribute execution across the P2P network using libp2p.

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::network::{VmNetworkBridge, VmNetworkConfig, VmNetworkMessage, VmExecutionResult};
use crate::state::StateDB;
use crate::vm::{ExecutionResult, VmError};
use crate::vm::ultra_performance_bridge::UltraContractProcessor;

/// Execution strategy for contract calls
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionStrategy {
    /// Execute locally only
    Local,

    /// Execute on remote VM (load balancing)
    Remote,

    /// Execute on both local and remote, compare results (redundancy)
    Replicated,

    /// Execute on fastest available VM (automatic selection)
    Fastest,
}

/// Configuration for networked executor
#[derive(Debug, Clone)]
pub struct NetworkedExecutorConfig {
    /// Default execution strategy
    pub default_strategy: ExecutionStrategy,

    /// Enable automatic fallback to local execution on network failure
    pub fallback_to_local: bool,

    /// Maximum wait time for remote execution (ms)
    pub remote_timeout_ms: u64,

    /// Enable execution result validation across replicas
    pub enable_result_validation: bool,

    /// Minimum number of matching results for validation
    pub min_validation_confirmations: usize,
}

impl Default for NetworkedExecutorConfig {
    fn default() -> Self {
        Self {
            default_strategy: ExecutionStrategy::Local,
            fallback_to_local: true,
            remote_timeout_ms: 5000,
            enable_result_validation: false,
            min_validation_confirmations: 2,
        }
    }
}

/// Statistics for networked execution
#[derive(Debug, Clone, Default)]
pub struct NetworkedExecutorStats {
    pub local_executions: u64,
    pub remote_executions: u64,
    pub replicated_executions: u64,
    pub validation_failures: u64,
    pub network_fallbacks: u64,
    pub average_local_latency_ms: f64,
    pub average_remote_latency_ms: f64,
}

/// Networked VM Executor with libp2p integration
pub struct NetworkedVmExecutor {
    /// Configuration
    config: NetworkedExecutorConfig,

    /// Network bridge for P2P communication
    network_bridge: Arc<RwLock<VmNetworkBridge>>,

    /// Ultra-performance local executor
    local_executor: Arc<UltraContractProcessor>,

    /// State database
    state_db: Arc<StateDB>,

    /// Execution statistics
    stats: Arc<RwLock<NetworkedExecutorStats>>,
}

impl NetworkedVmExecutor {
    /// Create new networked executor
    pub async fn new(
        config: NetworkedExecutorConfig,
        network_config: VmNetworkConfig,
        state_db: Arc<StateDB>,
    ) -> Result<Self> {
        info!("🌐 Initializing Networked VM Executor");

        // Create network bridge
        let network_bridge = VmNetworkBridge::new(network_config, state_db.clone()).await?;
        let network_bridge = Arc::new(RwLock::new(network_bridge));

        // Create ultra-performance local executor
        let ultra_config = crate::vm::ultra_performance_bridge::UltraContractConfig {
            target_tps: 150_000,
            num_shards: num_cpus::get(),
            workers_per_shard: 4,
            batch_size: 10_000,
            contract_cache_size: 100_000,
            pipeline_depth: 8,
            use_simd: true,
            use_zero_copy: true,
            jit_compilation: true,
        };

        let local_state_db = Arc::new(crate::vm::ultra_performance_bridge::StateDB::new());
        let local_executor = Arc::new(UltraContractProcessor::new(ultra_config, local_state_db)?);

        info!("✅ Networked VM Executor initialized");

        Ok(Self {
            config,
            network_bridge,
            local_executor,
            state_db,
            stats: Arc::new(RwLock::new(NetworkedExecutorStats::default())),
        })
    }

    /// Initialize network bridge with libp2p
    pub async fn with_libp2p(self, keypair: libp2p::identity::Keypair) -> Result<Self> {
        let mut bridge = self.network_bridge.write().await;
        let new_bridge = std::mem::replace(&mut *bridge,
            VmNetworkBridge::new(VmNetworkConfig::default(), self.state_db.clone()).await?);

        *bridge = new_bridge.with_libp2p_bridge(keypair).await?;
        drop(bridge);

        info!("✅ Networked executor connected to libp2p");
        Ok(self)
    }

    /// Initialize with unified network manager (zero-config)
    pub async fn with_unified_network(self) -> Result<Self> {
        let mut bridge = self.network_bridge.write().await;
        let new_bridge = std::mem::replace(&mut *bridge,
            VmNetworkBridge::new(VmNetworkConfig::default(), self.state_db.clone()).await?);

        *bridge = new_bridge.with_unified_network().await?;
        drop(bridge);

        info!("✅ Networked executor using unified network");
        Ok(self)
    }

    /// Execute contract with specified strategy
    pub async fn execute(
        &self,
        contract_address: &str,
        function: &str,
        args: &[u8],
        caller: &str,
        gas_limit: u64,
        strategy: Option<ExecutionStrategy>,
    ) -> Result<ExecutionResult, VmError> {
        let strategy = strategy.unwrap_or(self.config.default_strategy);

        debug!(
            contract = %contract_address,
            function = %function,
            strategy = ?strategy,
            "Executing contract with strategy"
        );

        match strategy {
            ExecutionStrategy::Local => {
                self.execute_local(contract_address, function, args, caller, gas_limit).await
            }
            ExecutionStrategy::Remote => {
                self.execute_remote(contract_address, function, args, caller, gas_limit).await
            }
            ExecutionStrategy::Replicated => {
                self.execute_replicated(contract_address, function, args, caller, gas_limit).await
            }
            ExecutionStrategy::Fastest => {
                self.execute_fastest(contract_address, function, args, caller, gas_limit).await
            }
        }
    }

    /// Execute contract locally with ultra-performance
    async fn execute_local(
        &self,
        contract_address: &str,
        function: &str,
        args: &[u8],
        caller: &str,
        gas_limit: u64,
    ) -> Result<ExecutionResult, VmError> {
        let start = Instant::now();

        // Execute using ultra-performance executor
        let result = UltraContractProcessor::execute_contract(
            contract_address,
            function,
            args,
            caller,
            gas_limit,
            1_000_000_000, // 1 gwei gas price
        ).await?;

        let latency = start.elapsed().as_millis() as f64;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.local_executions += 1;

            let total = stats.local_executions as f64;
            stats.average_local_latency_ms =
                (stats.average_local_latency_ms * (total - 1.0) + latency) / total;
        }

        debug!(
            contract = %contract_address,
            function = %function,
            latency_ms = latency,
            gas_used = result.gas_used,
            "Local execution completed"
        );

        Ok(ExecutionResult {
            success: result.success,
            return_data: result.return_data,
            gas_used: result.gas_used,
            logs: result.logs,
            error: result.error_message,
        })
    }

    /// Execute contract on remote VM
    async fn execute_remote(
        &self,
        contract_address: &str,
        function: &str,
        args: &[u8],
        caller: &str,
        gas_limit: u64,
    ) -> Result<ExecutionResult, VmError> {
        let start = Instant::now();

        let bridge = self.network_bridge.read().await;
        let result = bridge.execute_remote_contract(
            contract_address.to_string(),
            function.to_string(),
            args.to_vec(),
            caller.to_string(),
            gas_limit,
        ).await;

        drop(bridge);

        match result {
            Ok(vm_result) => {
                let latency = start.elapsed().as_millis() as f64;

                // Update stats
                {
                    let mut stats = self.stats.write().await;
                    stats.remote_executions += 1;

                    let total = stats.remote_executions as f64;
                    stats.average_remote_latency_ms =
                        (stats.average_remote_latency_ms * (total - 1.0) + latency) / total;
                }

                debug!(
                    contract = %contract_address,
                    function = %function,
                    latency_ms = latency,
                    "Remote execution completed"
                );

                Ok(ExecutionResult {
                    success: vm_result.success,
                    return_data: vm_result.return_data,
                    gas_used: vm_result.gas_used,
                    logs: vm_result.logs,
                    error: vm_result.error,
                })
            }
            Err(e) if self.config.fallback_to_local => {
                warn!(
                    error = %e,
                    "Remote execution failed, falling back to local"
                );

                let mut stats = self.stats.write().await;
                stats.network_fallbacks += 1;
                drop(stats);

                self.execute_local(contract_address, function, args, caller, gas_limit).await
            }
            Err(e) => Err(e),
        }
    }

    /// Execute on both local and remote, validate results
    async fn execute_replicated(
        &self,
        contract_address: &str,
        function: &str,
        args: &[u8],
        caller: &str,
        gas_limit: u64,
    ) -> Result<ExecutionResult, VmError> {
        // Execute locally and remotely in parallel
        let local_future = self.execute_local(contract_address, function, args, caller, gas_limit);
        let remote_future = self.execute_remote(contract_address, function, args, caller, gas_limit);

        let (local_result, remote_result) = tokio::join!(local_future, remote_future);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.replicated_executions += 1;
        }

        // Validate results if enabled
        if self.config.enable_result_validation {
            match (&local_result, &remote_result) {
                (Ok(local), Ok(remote)) => {
                    // Compare results
                    if local.success == remote.success
                        && local.return_data == remote.return_data
                        && local.gas_used == remote.gas_used
                    {
                        info!(
                            contract = %contract_address,
                            "Replicated execution validated successfully"
                        );
                        return Ok(local.clone());
                    } else {
                        warn!(
                            contract = %contract_address,
                            "Replicated execution mismatch detected"
                        );

                        let mut stats = self.stats.write().await;
                        stats.validation_failures += 1;
                    }
                }
                _ => {}
            }
        }

        // Return local result (preferred)
        local_result
    }

    /// Execute on fastest available VM (race condition)
    async fn execute_fastest(
        &self,
        contract_address: &str,
        function: &str,
        args: &[u8],
        caller: &str,
        gas_limit: u64,
    ) -> Result<ExecutionResult, VmError> {
        let local_future = self.execute_local(contract_address, function, args, caller, gas_limit);
        let remote_future = self.execute_remote(contract_address, function, args, caller, gas_limit);

        // Race both executions, return whichever finishes first
        tokio::select! {
            result = local_future => {
                debug!(contract = %contract_address, "Local execution won race");
                result
            }
            result = remote_future => {
                debug!(contract = %contract_address, "Remote execution won race");
                result
            }
        }
    }

    /// Get execution statistics
    pub async fn get_stats(&self) -> NetworkedExecutorStats {
        self.stats.read().await.clone()
    }

    /// Get network statistics
    pub async fn get_network_stats(&self) -> crate::network::VmNetworkStats {
        self.network_bridge.read().await.get_stats().await
    }

    /// Run network bridge event loop (should be spawned as task)
    pub async fn run_network_bridge(&self) -> Result<()> {
        let mut bridge = self.network_bridge.write().await;
        bridge.run().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_networked_executor_creation() {
        let state_db = Arc::new(StateDB::new());
        let exec_config = NetworkedExecutorConfig::default();
        let net_config = VmNetworkConfig::default();

        let executor = NetworkedVmExecutor::new(exec_config, net_config, state_db).await;
        assert!(executor.is_ok());
    }

    #[tokio::test]
    async fn test_local_execution() {
        let state_db = Arc::new(StateDB::new());
        let exec_config = NetworkedExecutorConfig::default();
        let net_config = VmNetworkConfig::default();

        let executor = NetworkedVmExecutor::new(exec_config, net_config, state_db).await.unwrap();

        let result = executor.execute(
            "0xcontract",
            "balanceOf",
            &[1, 2, 3, 4],
            "0xcaller",
            100000,
            Some(ExecutionStrategy::Local),
        ).await;

        assert!(result.is_ok());

        let stats = executor.get_stats().await;
        assert_eq!(stats.local_executions, 1);
    }
}
