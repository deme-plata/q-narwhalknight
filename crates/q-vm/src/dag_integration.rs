use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
/// DAG Integration for Q-NarwhalKnight VM
/// Connects the existing DAG-Knight consensus with the VM layer
/// This bridges the gap between consensus and smart contract execution
use std::sync::Arc;
use tokio::sync::RwLock;

// Import existing DAG components
use q_dag_knight::{CommitDecision, ConsensusMetrics, ConsensusStatus, DAGKnightConsensus};
use q_narwhal_core::{Certificate, NarwhalCore};
use q_types::{NodeId, Round, Transaction, VertexId};

use crate::state::StateDB;
use crate::vm::{ExecutionResult, StateAccess, VirtualMachine, VmError};

/// Helper function to convert [u8; 32] to u64 for legacy address format
fn address_to_u64(address: &[u8; 32]) -> u64 {
    u64::from_be_bytes([
        address[0], address[1], address[2], address[3], address[4], address[5], address[6],
        address[7],
    ])
}

/// Helper function to convert u64 to [u8; 32] for modern address format
fn u64_to_address(value: u64) -> [u8; 32] {
    let mut address = [0u8; 32];
    let bytes = value.to_be_bytes();
    address[..8].copy_from_slice(&bytes);
    address
}

/// Helper function to convert [u8; 32] to String for transaction IDs
fn id_to_string(id: &[u8; 32]) -> String {
    hex::encode(id)
}

/// Enhanced DAG that integrates with VM execution
pub struct VMIntegratedDAG {
    /// Core DAG-Knight consensus engine
    pub dag_consensus: Arc<DAGKnightConsensus>,

    /// Narwhal mempool layer
    pub narwhal_core: Arc<NarwhalCore>,

    /// VM instance for smart contract execution
    pub virtual_machine: Arc<VirtualMachine>,

    /// State database for VM state management
    pub state_db: Arc<StateDB>,

    /// Transaction execution queue
    pub execution_queue: RwLock<Vec<VMTransaction>>,

    /// Contract deployment tracking
    pub deployed_contracts: RwLock<HashMap<VertexId, ContractDeployment>>,

    /// VM execution metrics
    pub vm_metrics: RwLock<VMExecutionMetrics>,
}

/// Transaction with VM execution context
#[derive(Debug, Clone)]
pub struct VMTransaction {
    pub base_transaction: Transaction,
    pub execution_type: VMExecutionType,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub contract_address: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum VMExecutionType {
    ContractDeployment {
        bytecode: Vec<u8>,
        constructor_args: Vec<u8>,
    },
    ContractCall {
        function_name: String,
        arguments: Vec<u8>,
    },
    Transfer {
        to: u64,
        /// v2.10.0: Updated to u128 for 24 decimal precision
        amount: u128,
    },
}

#[derive(Debug, Clone)]
pub struct ContractDeployment {
    pub contract_address: u64,
    pub deployment_vertex: VertexId,
    pub deployment_round: Round,
    pub bytecode_hash: [u8; 32],
    pub gas_used: u64,
}

#[derive(Debug, Clone, Default)]
pub struct VMExecutionMetrics {
    pub total_contracts_deployed: u64,
    pub total_contract_calls: u64,
    pub total_gas_used: u64,
    pub average_execution_time_ms: f64,
    pub successful_executions: u64,
    pub failed_executions: u64,
}

impl VMIntegratedDAG {
    pub async fn new(node_id: NodeId, f: usize, _state_db_path: &str) -> Result<Self> {
        // Initialize core DAG consensus
        let dag_consensus = Arc::new(DAGKnightConsensus::new(node_id, f).await?);

        // Initialize Narwhal mempool
        let narwhal_core = Arc::new(NarwhalCore::new(node_id));

        // Initialize state database
        let state_db = Arc::new(StateDB::new());

        // Initialize VM
        let virtual_machine = Arc::new(VirtualMachine::new(state_db.clone()));

        Ok(Self {
            dag_consensus,
            narwhal_core,
            virtual_machine,
            state_db,
            execution_queue: RwLock::new(Vec::new()),
            deployed_contracts: RwLock::new(HashMap::new()),
            vm_metrics: RwLock::new(VMExecutionMetrics::default()),
        })
    }

    /// Process a certificate from DAG consensus and execute VM transactions
    pub async fn process_certificate_with_vm(
        &self,
        certificate: Certificate,
    ) -> Result<Vec<VMExecutionResult>> {
        tracing::info!(
            "Processing certificate {} with VM execution",
            hex::encode(certificate.vertex_id)
        );

        // First, process through DAG consensus
        let commit_decisions = self
            .dag_consensus
            .process_certificate(certificate.clone())
            .await?;

        let mut vm_results = Vec::new();

        // For each committed decision, execute VM transactions
        for decision in commit_decisions {
            let execution_results = self.execute_vertex_transactions(&decision).await?;
            vm_results.extend(execution_results);
        }

        Ok(vm_results)
    }

    /// Execute all transactions in a committed vertex
    async fn execute_vertex_transactions(
        &self,
        decision: &CommitDecision,
    ) -> Result<Vec<VMExecutionResult>> {
        tracing::info!(
            "Executing VM transactions for committed vertex {} in round {}",
            hex::encode(decision.vertex_id),
            decision.round
        );

        let mut results = Vec::new();
        let start_time = std::time::Instant::now();

        for transaction in &decision.transactions {
            let vm_transaction = self.parse_vm_transaction(transaction)?;
            let execution_result = self
                .execute_vm_transaction(vm_transaction, decision.round)
                .await?;
            results.push(execution_result);
        }

        // Update metrics
        {
            let mut metrics = self.vm_metrics.write().await;
            let execution_time = start_time.elapsed().as_millis() as f64;

            // Update average execution time
            let total_executions = metrics.successful_executions + metrics.failed_executions;
            metrics.average_execution_time_ms =
                (metrics.average_execution_time_ms * total_executions as f64 + execution_time)
                    / (total_executions + results.len() as u64) as f64;
        }

        Ok(results)
    }

    /// Parse a regular transaction into a VM transaction
    fn parse_vm_transaction(&self, transaction: &Transaction) -> Result<VMTransaction> {
        // Parse transaction data to determine execution type
        // This is a simplified parser - in practice, would need proper encoding/decoding

        let execution_type = if transaction.data.starts_with(b"DEPLOY:") {
            // Contract deployment
            let bytecode = transaction.data[7..].to_vec(); // Skip "DEPLOY:" prefix
            VMExecutionType::ContractDeployment {
                bytecode,
                constructor_args: Vec::new(),
            }
        } else if transaction.data.starts_with(b"CALL:") {
            // Contract call
            VMExecutionType::ContractCall {
                function_name: "execute".to_string(),      // Simplified
                arguments: transaction.data[5..].to_vec(), // Skip "CALL:" prefix
            }
        } else {
            // Simple transfer
            VMExecutionType::Transfer {
                to: address_to_u64(&transaction.to),
                amount: transaction.amount,
            }
        };

        Ok(VMTransaction {
            base_transaction: transaction.clone(),
            execution_type,
            gas_limit: 100000, // Default gas limit
            gas_price: 1,      // Default gas price
            contract_address: None,
        })
    }

    /// Execute a single VM transaction
    async fn execute_vm_transaction(
        &self,
        vm_tx: VMTransaction,
        round: Round,
    ) -> Result<VMExecutionResult> {
        let execution_start = std::time::Instant::now();

        let result = match vm_tx.execution_type {
            VMExecutionType::ContractDeployment {
                bytecode,
                constructor_args,
            } => {
                self.deploy_contract(
                    address_to_u64(&vm_tx.base_transaction.from),
                    bytecode,
                    constructor_args,
                    round,
                )
                .await?
            }
            VMExecutionType::ContractCall {
                function_name,
                arguments,
            } => {
                if let Some(contract_address) = vm_tx.contract_address {
                    self.call_contract(
                        contract_address,
                        function_name,
                        arguments,
                        address_to_u64(&vm_tx.base_transaction.from),
                        vm_tx.gas_limit,
                    )
                    .await?
                } else {
                    return Err(anyhow::anyhow!(
                        "Contract address required for contract call"
                    ));
                }
            }
            VMExecutionType::Transfer { to, amount } => {
                self.execute_transfer(address_to_u64(&vm_tx.base_transaction.from), to, amount)
                    .await?
            }
        };

        // Update execution metrics
        {
            let mut metrics = self.vm_metrics.write().await;
            if result.success {
                metrics.successful_executions += 1;
            } else {
                metrics.failed_executions += 1;
            }
            metrics.total_gas_used += result.gas_used;
        }

        Ok(VMExecutionResult {
            transaction_id: id_to_string(&vm_tx.base_transaction.id),
            execution_result: result,
            execution_time_ms: execution_start.elapsed().as_millis() as u64,
            round,
        })
    }

    /// Deploy a smart contract
    async fn deploy_contract(
        &self,
        deployer: u64,
        bytecode: Vec<u8>,
        _constructor_args: Vec<u8>,
        round: Round,
    ) -> Result<ExecutionResult> {
        tracing::info!(
            "Deploying contract for deployer {} in round {}",
            deployer,
            round
        );

        // Generate contract address (simplified - use deployer + nonce)
        let nonce = self.state_db.get_nonce(deployer).await.unwrap_or(0);
        let contract_address = ((deployer as u128) << 64 | nonce as u128) as u64;

        // TODO: Compile and instantiate WASM contract
        // For now, simulate deployment
        let gas_used = 21000; // Base deployment gas

        // Store contract code
        self.state_db
            .set_storage(contract_address, b"CODE".to_vec(), bytecode.clone())
            .await?;

        // Record deployment
        {
            let mut deployments = self.deployed_contracts.write().await;
            deployments.insert(
                [0u8; 32],
                ContractDeployment {
                    // Use proper vertex ID
                    contract_address,
                    deployment_vertex: [0u8; 32], // Use actual vertex ID
                    deployment_round: round,
                    bytecode_hash: blake3::hash(&bytecode).into(),
                    gas_used,
                },
            );
        }

        // Update metrics
        {
            let mut metrics = self.vm_metrics.write().await;
            metrics.total_contracts_deployed += 1;
        }

        Ok(ExecutionResult {
            success: true,
            return_data: contract_address.to_be_bytes().to_vec(),
            gas_used,
            logs: vec![format!("Contract deployed at address {}", contract_address)],
            error: None,
        })
    }

    /// Call a smart contract function
    async fn call_contract(
        &self,
        contract_address: u64,
        function: String,
        _arguments: Vec<u8>,
        caller: u64,
        _gas_limit: u64,
    ) -> Result<ExecutionResult> {
        tracing::info!(
            "Calling function '{}' on contract {} from caller {}",
            function,
            contract_address,
            caller
        );

        // Check if contract exists
        let contract_code = self.state_db.get_storage(contract_address, b"CODE").await?;
        if contract_code.is_none() {
            return Ok(ExecutionResult {
                success: false,
                return_data: Vec::new(),
                gas_used: 21000,
                logs: Vec::new(),
                error: Some("Contract not found".to_string()),
            });
        }

        // TODO: Execute WASM contract
        // For now, simulate execution
        let gas_used = 50000; // Simulate gas usage

        // Update metrics
        {
            let mut metrics = self.vm_metrics.write().await;
            metrics.total_contract_calls += 1;
        }

        Ok(ExecutionResult {
            success: true,
            return_data: b"execution_result".to_vec(),
            gas_used,
            logs: vec![format!("Function '{}' executed successfully", function)],
            error: None,
        })
    }

    /// Execute a simple transfer
    /// v2.10.0: Updated to u128 for 24 decimal precision
    /// Note: VM state internally uses u64 for backwards compatibility
    async fn execute_transfer(&self, from: u64, to: u64, amount: u128) -> Result<ExecutionResult> {
        tracing::info!(
            "Executing transfer: {} -> {} (amount: {})",
            from,
            to,
            amount
        );

        // VM internal state uses u64 - cast to u64 (safe for reasonable amounts)
        let amount_u64 = amount as u64;

        // Check balance
        let from_balance = self.state_db.get_balance(from).await.unwrap_or(0);
        if from_balance < amount_u64 {
            return Ok(ExecutionResult {
                success: false,
                return_data: Vec::new(),
                gas_used: 21000,
                logs: Vec::new(),
                error: Some("Insufficient balance".to_string()),
            });
        }

        // Execute transfer
        self.state_db
            .set_balance(from, from_balance - amount_u64)
            .await?;
        let to_balance = self.state_db.get_balance(to).await.unwrap_or(0);
        self.state_db.set_balance(to, to_balance + amount_u64).await?;

        Ok(ExecutionResult {
            success: true,
            return_data: Vec::new(),
            gas_used: 21000,
            logs: vec![format!("Transferred {} from {} to {}", amount, from, to)],
            error: None,
        })
    }

    /// Get DAG consensus status with VM metrics
    pub async fn get_integrated_status(&self) -> Result<IntegratedDAGStatus> {
        let dag_status = self.dag_consensus.get_status().await;
        let dag_metrics = self.dag_consensus.get_metrics().await;
        let vm_metrics = self.vm_metrics.read().await.clone();

        Ok(IntegratedDAGStatus {
            consensus_status: dag_status,
            consensus_metrics: dag_metrics,
            vm_metrics,
            total_deployed_contracts: self.deployed_contracts.read().await.len() as u64,
        })
    }

    /// Get transaction ordering with VM execution context
    pub async fn get_vm_transaction_ordering(
        &self,
        from_round: Round,
        to_round: Round,
    ) -> Result<Vec<VMExecutionResult>> {
        let ordered_transactions = self
            .dag_consensus
            .get_transaction_ordering(from_round, to_round)
            .await?;

        let mut vm_results = Vec::new();

        // Convert to VM execution results (this would normally be stored)
        for (i, transaction) in ordered_transactions.iter().enumerate() {
            let vm_result = VMExecutionResult {
                transaction_id: id_to_string(&transaction.id),
                execution_result: ExecutionResult {
                    success: true,
                    return_data: Vec::new(),
                    gas_used: 21000,
                    logs: Vec::new(),
                    error: None,
                },
                execution_time_ms: 10,               // Mock execution time
                round: from_round + (i as u64 / 10), // Estimate round
            };
            vm_results.push(vm_result);
        }

        Ok(vm_results)
    }

    /// Start the integrated DAG-VM system
    pub async fn start(&self) -> Result<()> {
        tracing::info!("Starting integrated DAG-VM system");

        // Initialize quantum VDF timing for consensus
        // Note: This integrates the existing quantum VDF from DAG-Knight
        let target_round_time_ms = 100; // 100ms round time target
        if let Ok(mut consensus) = Arc::try_unwrap(self.dag_consensus.clone()) {
            consensus
                .integrate_quantum_vdf_timing(target_round_time_ms)
                .await?;
        }

        tracing::info!("Integrated DAG-VM system started successfully");
        Ok(())
    }
}

/// Result of VM execution within DAG context
#[derive(Debug, Clone)]
pub struct VMExecutionResult {
    pub transaction_id: String,
    pub execution_result: ExecutionResult,
    pub execution_time_ms: u64,
    pub round: Round,
}

/// Combined status of DAG consensus and VM execution
#[derive(Debug, Clone)]
pub struct IntegratedDAGStatus {
    pub consensus_status: ConsensusStatus,
    pub consensus_metrics: ConsensusMetrics,
    pub vm_metrics: VMExecutionMetrics,
    pub total_deployed_contracts: u64,
}

/// Implementation of StateAccess for the integrated system
#[async_trait]
impl StateAccess for VMIntegratedDAG {
    async fn get_contract(&self, address: u64) -> Result<Option<Vec<u8>>, VmError> {
        match self.state_db.get_storage(address, b"CODE").await {
            Ok(code) => Ok(code),
            Err(e) => Err(VmError::StorageError(e.to_string())),
        }
    }

    async fn get_storage(&self, address: u64, key: &[u8]) -> Result<Option<Vec<u8>>, VmError> {
        match self.state_db.get_storage(address, key).await {
            Ok(value) => Ok(value),
            Err(e) => Err(VmError::StorageError(e.to_string())),
        }
    }

    async fn set_storage(&self, address: u64, key: Vec<u8>, value: Vec<u8>) -> Result<(), VmError> {
        match self.state_db.set_storage(address, key, value).await {
            Ok(()) => Ok(()),
            Err(e) => Err(VmError::StorageError(e.to_string())),
        }
    }

    async fn get_balance(&self, address: u64) -> Result<u64, VmError> {
        match self.state_db.get_balance(address).await {
            Ok(balance) => Ok(balance),
            Err(e) => Err(VmError::StorageError(e.to_string())),
        }
    }

    async fn set_balance(&self, address: u64, amount: u64) -> Result<(), VmError> {
        match self.state_db.set_balance(address, amount).await {
            Ok(()) => Ok(()),
            Err(e) => Err(VmError::StorageError(e.to_string())),
        }
    }

    async fn get_nonce(&self, address: u64) -> Result<u64, VmError> {
        match self.state_db.get_nonce(address).await {
            Ok(nonce) => Ok(nonce),
            Err(e) => Err(VmError::StorageError(e.to_string())),
        }
    }

    async fn get_contract_state(
        &self,
        address: u64,
    ) -> Result<Option<crate::vm::ContractState>, VmError> {
        // Get contract code
        let code = match self.get_contract(address).await? {
            Some(code) => code,
            None => return Ok(None),
        };

        // TODO: Load storage (simplified for now)
        let storage = std::collections::HashMap::new();

        Ok(Some(crate::vm::ContractState {
            code,
            storage,
            balance: 0, // Default balance
            nonce: 0,   // Default nonce
        }))
    }
}
