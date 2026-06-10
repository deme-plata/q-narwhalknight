//! Unified VM Integration Coordinator
//! Orchestrates all integration subsystems for seamless operation

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use std::time::Instant;

use crate::vm::VirtualMachine;
use crate::state::StateDB;
use crate::types::VMTransaction;

use super::{
    IntegrationConfig, IntegrationResult, IntegrationStatus, IntegrationMetrics, CryptographicPhase,
    dag_consensus::DAGConsensusIntegration,
    narwhal_broadcast::NarwhalBroadcastIntegration,
    quantum_vdf::QuantumVDFIntegration,
    post_quantum::PostQuantumIntegration,
    VMIntegration,
};

/// Comprehensive VM integration coordinator
pub struct VMIntegrationCoordinator {
    /// Core VM engine
    vm: Arc<VirtualMachine>,
    
    /// State database
    state_db: Arc<StateDB>,
    
    /// Integration subsystems
    dag_consensus: Arc<DAGConsensusIntegration>,
    narwhal_broadcast: Arc<NarwhalBroadcastIntegration>,
    quantum_vdf: Arc<QuantumVDFIntegration>,
    post_quantum: Arc<PostQuantumIntegration>,
    
    /// Configuration
    config: Arc<RwLock<IntegrationConfig>>,
    
    /// Coordinator metrics
    coordinator_metrics: Arc<RwLock<CoordinatorMetrics>>,
    
    /// Transaction processing pipeline
    processing_pipeline: Arc<RwLock<ProcessingPipeline>>,
}

#[derive(Debug, Clone, Default)]
pub struct CoordinatorMetrics {
    pub total_transactions_processed: u64,
    pub successful_integrations: u64,
    pub failed_integrations: u64,
    pub average_integration_time_ms: f64,
    pub consensus_integrations: u64,
    pub mempool_broadcasts: u64,
    pub vdf_computations: u64,
    pub crypto_operations: u64,
    pub peak_tps: f64,
    pub current_tps: f64,
    pub uptime_seconds: u64,
}

#[derive(Debug)]
pub struct ProcessingPipeline {
    pub pending_transactions: std::collections::VecDeque<VMTransaction>,
    pub processing_transactions: std::collections::HashMap<String, ProcessingState>,
    pub completed_transactions: std::collections::VecDeque<IntegrationResult>,
    pub max_pipeline_size: usize,
}

#[derive(Debug, Clone)]
pub struct ProcessingState {
    pub transaction: VMTransaction,
    pub start_time: Instant,
    pub current_stage: IntegrationStage,
    pub stage_results: std::collections::HashMap<IntegrationStage, IntegrationResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegrationStage {
    Validation,       // Initial transaction validation
    Cryptography,     // Signature verification and PQ crypto
    VDF,              // Quantum VDF computation
    Consensus,        // DAG consensus ordering
    Broadcast,        // Narwhal reliable broadcast
    Execution,        // VM execution
    Finalization,     // State finalization
}

impl VMIntegrationCoordinator {
    /// Create new integration coordinator
    pub fn new(config: IntegrationConfig) -> Result<Self> {
        // Initialize state database
        let state_db = Arc::new(StateDB::new());
        let vm = Arc::new(VirtualMachine::new(state_db.clone()));
        
        // Initialize all integration subsystems
        let dag_consensus = Arc::new(DAGConsensusIntegration::new(config.node_id.clone())?);
        let narwhal_broadcast = Arc::new(NarwhalBroadcastIntegration::new(config.node_id.clone()));
        let quantum_vdf = Arc::new(QuantumVDFIntegration::new(config.vdf_difficulty));
        let post_quantum = Arc::new(PostQuantumIntegration::new(config.phase)?);
        
        let processing_pipeline = ProcessingPipeline {
            pending_transactions: std::collections::VecDeque::new(),
            processing_transactions: std::collections::HashMap::new(),
            completed_transactions: std::collections::VecDeque::new(),
            max_pipeline_size: 1000,
        };
        
        Ok(Self {
            vm,
            state_db,
            dag_consensus,
            narwhal_broadcast,
            quantum_vdf,
            post_quantum,
            config: Arc::new(RwLock::new(config)),
            coordinator_metrics: Arc::new(RwLock::new(CoordinatorMetrics::default())),
            processing_pipeline: Arc::new(RwLock::new(processing_pipeline)),
        })
    }
    
    /// Initialize all integration subsystems
    pub async fn initialize(&self) -> Result<()> {
        let config = self.config.read().await;
        
        tracing::info!(
            "Initializing VM Integration Coordinator for node {} in phase {:?}",
            config.node_id, config.phase
        );
        
        // Initialize all subsystems
        self.dag_consensus.initialize(&config).await?;
        self.narwhal_broadcast.initialize(&config).await?;
        self.quantum_vdf.initialize(&config).await?;
        self.post_quantum.initialize(&config).await?;
        
        tracing::info!("All integration subsystems initialized successfully");
        Ok(())
    }
    
    /// Process transaction through complete integration pipeline
    pub async fn process_transaction(&self, tx: VMTransaction) -> Result<IntegrationResult> {
        let start_time = Instant::now();
        let tx_id = tx.id.clone();
        
        // Add to processing pipeline
        {
            let mut pipeline = self.processing_pipeline.write().await;
            if pipeline.pending_transactions.len() >= pipeline.max_pipeline_size {
                return Err(anyhow::anyhow!("Processing pipeline full"));
            }
            
            pipeline.processing_transactions.insert(tx_id.clone(), ProcessingState {
                transaction: tx.clone(),
                start_time,
                current_stage: IntegrationStage::Validation,
                stage_results: std::collections::HashMap::new(),
            });
        }
        
        // Execute integration pipeline
        let result = self.execute_integration_pipeline(tx).await;
        
        // Update pipeline state
        {
            let mut pipeline = self.processing_pipeline.write().await;
            pipeline.processing_transactions.remove(&tx_id);
            
            if let Ok(ref integration_result) = result {
                pipeline.completed_transactions.push_back(integration_result.clone());
                
                // Keep completed results limited
                if pipeline.completed_transactions.len() > 1000 {
                    pipeline.completed_transactions.pop_front();
                }
            }
        }
        
        // Update coordinator metrics
        {
            let mut metrics = self.coordinator_metrics.write().await;
            metrics.total_transactions_processed += 1;
            
            if result.is_ok() {
                metrics.successful_integrations += 1;
            } else {
                metrics.failed_integrations += 1;
            }
            
            let processing_time = start_time.elapsed().as_millis() as f64;
            metrics.average_integration_time_ms = 
                (metrics.average_integration_time_ms * (metrics.total_transactions_processed - 1) as f64 
                 + processing_time) / metrics.total_transactions_processed as f64;
        }
        
        result
    }
    
    /// Execute the complete integration pipeline
    async fn execute_integration_pipeline(&self, tx: VMTransaction) -> Result<IntegrationResult> {
        let mut final_result = IntegrationResult {
            success: true,
            transaction_hash: tx.id.clone(),
            execution_result: crate::vm::ExecutionResult {
                success: true,
                return_data: Vec::new(),
                gas_used: 0,
                logs: Vec::new(),
                error: None,
            },
            consensus_round: 0,
            vdf_output: None,
            crypto_phase: CryptographicPhase::Phase1,
            processing_time_ms: 0,
            integration_metrics: IntegrationMetrics::default(),
        };
        
        // Stage 1: Cryptographic validation and signing
        tracing::debug!("Stage 1: Cryptographic processing for transaction {}", tx.id);
        self.update_processing_stage(&tx.id, IntegrationStage::Cryptography).await;
        
        let crypto_result = self.post_quantum.process_transaction(&tx).await?;
        if !crypto_result.success {
            return Ok(crypto_result);
        }
        final_result.execution_result.gas_used += crypto_result.execution_result.gas_used;
        final_result.execution_result.logs.extend(crypto_result.execution_result.logs);
        
        // Stage 2: VDF computation for randomness
        if self.config.read().await.enable_quantum_vdf {
            tracing::debug!("Stage 2: VDF computation for transaction {}", tx.id);
            self.update_processing_stage(&tx.id, IntegrationStage::VDF).await;
            
            let vdf_result = self.quantum_vdf.process_transaction(&tx).await?;
            final_result.vdf_output = vdf_result.vdf_output;
            final_result.execution_result.gas_used += vdf_result.execution_result.gas_used;
            final_result.execution_result.logs.extend(vdf_result.execution_result.logs);
        }
        
        // Stage 3: Narwhal reliable broadcast
        tracing::debug!("Stage 3: Narwhal broadcast for transaction {}", tx.id);
        self.update_processing_stage(&tx.id, IntegrationStage::Broadcast).await;
        
        let broadcast_result = self.narwhal_broadcast.process_transaction(&tx).await?;
        final_result.execution_result.logs.extend(broadcast_result.execution_result.logs);
        
        // Stage 4: DAG consensus ordering
        tracing::debug!("Stage 4: DAG consensus for transaction {}", tx.id);
        self.update_processing_stage(&tx.id, IntegrationStage::Consensus).await;
        
        let consensus_result = self.dag_consensus.process_transaction(&tx).await?;
        final_result.consensus_round = consensus_result.consensus_round;
        final_result.execution_result.logs.extend(consensus_result.execution_result.logs);
        
        // Stage 5: VM execution
        tracing::debug!("Stage 5: VM execution for transaction {}", tx.id);
        self.update_processing_stage(&tx.id, IntegrationStage::Execution).await;
        
        let vm_result = self.vm.execute_transaction(&tx).await
            .map_err(|e| anyhow::anyhow!("VM execution failed: {}", e))?;
        
        // Combine all results
        final_result.execution_result = vm_result;
        final_result.processing_time_ms = Instant::now().duration_since(
            self.get_processing_start_time(&tx.id).await.unwrap_or_else(Instant::now)
        ).as_millis() as u64;
        
        // Stage 6: Finalization
        self.update_processing_stage(&tx.id, IntegrationStage::Finalization).await;
        self.finalize_transaction(&tx, &final_result).await?;
        
        // Update subsystem metrics
        {
            let mut coordinator_metrics = self.coordinator_metrics.write().await;
            coordinator_metrics.consensus_integrations += 1;
            coordinator_metrics.mempool_broadcasts += 1;
            coordinator_metrics.crypto_operations += 1;
            if self.config.read().await.enable_quantum_vdf {
                coordinator_metrics.vdf_computations += 1;
            }
        }
        
        tracing::info!(
            "Successfully processed transaction {} through complete integration pipeline in {}ms",
            tx.id, final_result.processing_time_ms
        );
        
        Ok(final_result)
    }
    
    /// Update processing stage for a transaction
    async fn update_processing_stage(&self, tx_id: &str, stage: IntegrationStage) {
        let mut pipeline = self.processing_pipeline.write().await;
        if let Some(processing_state) = pipeline.processing_transactions.get_mut(tx_id) {
            processing_state.current_stage = stage;
        }
    }
    
    /// Get processing start time for a transaction
    async fn get_processing_start_time(&self, tx_id: &str) -> Option<Instant> {
        let pipeline = self.processing_pipeline.read().await;
        pipeline.processing_transactions.get(tx_id).map(|state| state.start_time)
    }
    
    /// Finalize transaction processing
    async fn finalize_transaction(&self, _tx: &VMTransaction, result: &IntegrationResult) -> Result<()> {
        // Update state if transaction was successful
        if result.success {
            // State would be committed here in production
            tracing::debug!("Transaction {} finalized successfully", result.transaction_hash);
        }
        Ok(())
    }
    
    /// Get comprehensive system status
    pub async fn get_integrated_status(&self) -> Result<IntegratedSystemStatus> {
        let dag_status = self.dag_consensus.get_status().await?;
        let mempool_status = self.narwhal_broadcast.get_status().await?;
        let vdf_status = self.quantum_vdf.get_status().await?;
        let crypto_status = self.post_quantum.get_status().await?;
        
        let coordinator_metrics = self.coordinator_metrics.read().await.clone();
        let pipeline = self.processing_pipeline.read().await;
        
        Ok(IntegratedSystemStatus {
            overall_health: dag_status.is_healthy && mempool_status.is_healthy 
                         && vdf_status.is_healthy && crypto_status.is_healthy,
            dag_consensus: dag_status,
            narwhal_mempool: mempool_status,
            quantum_vdf: vdf_status,
            post_quantum_crypto: crypto_status,
            coordinator_metrics,
            pipeline_status: PipelineStatus {
                pending_transactions: pipeline.pending_transactions.len() as u64,
                processing_transactions: pipeline.processing_transactions.len() as u64,
                completed_transactions: pipeline.completed_transactions.len() as u64,
                max_pipeline_capacity: pipeline.max_pipeline_size as u64,
            },
        })
    }
    
    /// Batch process multiple transactions
    pub async fn batch_process(&self, transactions: Vec<VMTransaction>) -> Result<Vec<IntegrationResult>> {
        let start_time = Instant::now();
        let mut results = Vec::new();
        
        tracing::info!("Starting batch processing of {} transactions", transactions.len());
        
        // Process transactions in parallel batches
        let batch_size = 10; // Process 10 at a time
        for chunk in transactions.chunks(batch_size) {
            let mut batch_futures = Vec::new();
            
            for tx in chunk {
                let future = self.process_transaction(tx.clone());
                batch_futures.push(future);
            }
            
            // Wait for batch completion
            let batch_results = futures::future::join_all(batch_futures).await;
            for result in batch_results {
                match result {
                    Ok(integration_result) => results.push(integration_result),
                    Err(e) => {
                        tracing::error!("Batch transaction failed: {}", e);
                        // Continue processing other transactions
                    }
                }
            }
        }
        
        let total_time = start_time.elapsed();
        let tps = results.len() as f64 / total_time.as_secs_f64();
        
        // Update TPS metrics
        {
            let mut metrics = self.coordinator_metrics.write().await;
            metrics.current_tps = tps;
            if tps > metrics.peak_tps {
                metrics.peak_tps = tps;
            }
        }
        
        tracing::info!(
            "Batch processed {} transactions in {:?} ({:.2} TPS)",
            results.len(), total_time, tps
        );
        
        Ok(results)
    }
    
    /// Shutdown all integration subsystems
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down VM Integration Coordinator");
        
        // Shutdown all subsystems
        self.dag_consensus.shutdown().await?;
        self.narwhal_broadcast.shutdown().await?;
        self.quantum_vdf.shutdown().await?;
        self.post_quantum.shutdown().await?;
        
        tracing::info!("All integration subsystems shut down successfully");
        Ok(())
    }
    
    /// Get coordinator metrics
    pub async fn get_metrics(&self) -> CoordinatorMetrics {
        self.coordinator_metrics.read().await.clone()
    }
}

/// Comprehensive system status
#[derive(Debug, Clone)]
pub struct IntegratedSystemStatus {
    pub overall_health: bool,
    pub dag_consensus: IntegrationStatus,
    pub narwhal_mempool: IntegrationStatus,
    pub quantum_vdf: IntegrationStatus,
    pub post_quantum_crypto: IntegrationStatus,
    pub coordinator_metrics: CoordinatorMetrics,
    pub pipeline_status: PipelineStatus,
}

#[derive(Debug, Clone)]
pub struct PipelineStatus {
    pub pending_transactions: u64,
    pub processing_transactions: u64,
    pub completed_transactions: u64,
    pub max_pipeline_capacity: u64,
}

/// Convenience functions for easy integration

impl VMIntegrationCoordinator {
    /// Create coordinator with default configuration
    pub fn with_defaults(node_id: String, phase: CryptographicPhase) -> Result<Self> {
        let mut config = IntegrationConfig::default();
        config.node_id = node_id;
        config.phase = phase;
        Self::new(config)
    }
    
    /// Quick transaction processing with minimal configuration
    pub async fn quick_process(&self, tx: VMTransaction) -> Result<bool> {
        match self.process_transaction(tx).await {
            Ok(result) => Ok(result.success),
            Err(_) => Ok(false),
        }
    }
    
    /// Health check for all subsystems
    pub async fn health_check(&self) -> bool {
        match self.get_integrated_status().await {
            Ok(status) => status.overall_health,
            Err(_) => false,
        }
    }
}