//! Enhanced VM Tests - Comprehensive VM functionality testing
//!
//! This test file provides extensive testing of VM operations including:
//! - Basic state management and transfers
//! - Contract execution simulation
//! - Gas metering and performance testing
//! - Edge cases and error handling
//! - Concurrent operations and scalability

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

// Simple mock types to avoid complex dependencies
#[derive(Debug, Clone, Default)]
pub struct SimpleVmState {
    balances: HashMap<u64, u64>,
    nonces: HashMap<u64, u64>,
    storage: HashMap<u64, HashMap<Vec<u8>, Vec<u8>>>,
    state_root: [u8; 32],
    gas_used: u64,
    block_height: u64,
    contracts: HashMap<u64, Contract>,
}

#[derive(Debug, Clone)]
pub struct Contract {
    code: Vec<u8>,
    storage: HashMap<Vec<u8>, Vec<u8>>,
    balance: u64,
    created_at: u64,
}

#[derive(Debug, Clone)]
pub struct Transaction {
    from: u64,
    to: u64,
    value: u64,
    gas_limit: u64,
    gas_price: u64,
    data: Vec<u8>,
    nonce: u64,
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    success: bool,
    gas_used: u64,
    return_data: Vec<u8>,
    logs: Vec<String>,
    state_changes: u32,
}

impl SimpleVmState {
    fn update_state_root(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash balances
        for (account, balance) in &self.balances {
            account.hash(&mut hasher);
            balance.hash(&mut hasher);
        }

        // Hash nonces
        for (account, nonce) in &self.nonces {
            account.hash(&mut hasher);
            nonce.hash(&mut hasher);
        }

        // Hash gas usage and block height
        self.gas_used.hash(&mut hasher);
        self.block_height.hash(&mut hasher);

        // Hash contracts
        for (addr, contract) in &self.contracts {
            addr.hash(&mut hasher);
            contract.code.hash(&mut hasher);
            contract.balance.hash(&mut hasher);
        }

        let hash = hasher.finish();
        self.state_root = hash
            .to_be_bytes()
            .iter()
            .cycle()
            .take(32)
            .copied()
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap_or([0u8; 32]);
    }

    fn advance_block(&mut self) {
        self.block_height += 1;
        self.update_state_root();
    }
}

#[derive(Debug)]
pub struct SimpleStateDB {
    state: Arc<RwLock<SimpleVmState>>,
}

impl SimpleStateDB {
    pub fn new_in_memory() -> Self {
        Self {
            state: Arc::new(RwLock::new(SimpleVmState::default())),
        }
    }

    pub async fn get_balance(&self, account: u64) -> Result<u64> {
        let state = self.state.read().await;
        Ok(state.balances.get(&account).copied().unwrap_or(0))
    }

    pub async fn set_balance(&self, account: u64, balance: u64) -> Result<()> {
        let mut state = self.state.write().await;
        state.balances.insert(account, balance);
        state.update_state_root();
        Ok(())
    }

    pub async fn get_nonce(&self, account: u64) -> Result<u64> {
        let state = self.state.read().await;
        Ok(state.nonces.get(&account).copied().unwrap_or(0))
    }

    pub async fn set_nonce(&self, account: u64, nonce: u64) -> Result<()> {
        let mut state = self.state.write().await;
        state.nonces.insert(account, nonce);
        state.update_state_root();
        Ok(())
    }

    pub async fn get_storage(&self, contract: u64, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let state = self.state.read().await;
        Ok(state
            .storage
            .get(&contract)
            .and_then(|storage| storage.get(key))
            .cloned())
    }

    pub async fn set_storage(&self, contract: u64, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let mut state = self.state.write().await;
        state
            .storage
            .entry(contract)
            .or_insert_with(HashMap::new)
            .insert(key, value);
        state.update_state_root();
        Ok(())
    }

    pub async fn get_state_root(&self) -> Result<[u8; 32]> {
        let state = self.state.read().await;
        Ok(state.state_root)
    }

    pub async fn get_gas_used(&self) -> Result<u64> {
        let state = self.state.read().await;
        Ok(state.gas_used)
    }

    pub async fn add_gas_usage(&self, gas: u64) -> Result<()> {
        let mut state = self.state.write().await;
        state.gas_used = state.gas_used.saturating_add(gas);
        state.update_state_root();
        Ok(())
    }

    pub async fn get_block_height(&self) -> Result<u64> {
        let state = self.state.read().await;
        Ok(state.block_height)
    }

    pub async fn advance_block(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.advance_block();
        Ok(())
    }

    pub async fn deploy_contract(
        &self,
        address: u64,
        code: Vec<u8>,
        initial_balance: u64,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        let contract = Contract {
            code,
            storage: HashMap::new(),
            balance: initial_balance,
            created_at: state.block_height,
        };
        state.contracts.insert(address, contract);
        state.update_state_root();
        Ok(())
    }

    pub async fn get_contract(&self, address: u64) -> Result<Option<Contract>> {
        let state = self.state.read().await;
        Ok(state.contracts.get(&address).cloned())
    }
}

#[derive(Debug, Clone)]
pub struct SimpleVirtualMachine {
    state_db: Arc<SimpleStateDB>,
}

impl SimpleVirtualMachine {
    pub fn new(state_db: Arc<SimpleStateDB>) -> Self {
        Self { state_db }
    }

    pub async fn execute_transfer(&self, from: u64, to: u64, amount: u64) -> Result<bool> {
        let from_balance = self.state_db.get_balance(from).await?;
        if from_balance < amount {
            return Ok(false);
        }

        let to_balance = self.state_db.get_balance(to).await?;

        self.state_db
            .set_balance(from, from_balance - amount)
            .await?;
        self.state_db.set_balance(to, to_balance + amount).await?;

        // Add gas cost for transfer
        self.state_db.add_gas_usage(21000).await?;

        Ok(true)
    }

    pub async fn execute_transaction(&self, tx: Transaction) -> Result<ExecutionResult> {
        let start_time = Instant::now();
        let mut logs = Vec::new();
        let mut state_changes = 0;

        // Check nonce
        let current_nonce = self.state_db.get_nonce(tx.from).await?;
        if tx.nonce != current_nonce {
            return Ok(ExecutionResult {
                success: false,
                gas_used: 21000, // Base transaction cost
                return_data: vec![],
                logs: vec!["Invalid nonce".to_string()],
                state_changes: 0,
            });
        }

        // Check balance for gas
        let from_balance = self.state_db.get_balance(tx.from).await?;
        let gas_cost = tx.gas_limit * tx.gas_price;
        if from_balance < tx.value + gas_cost {
            return Ok(ExecutionResult {
                success: false,
                gas_used: 21000,
                return_data: vec![],
                logs: vec!["Insufficient balance for gas + value".to_string()],
                state_changes: 0,
            });
        }

        let mut gas_used = 21000; // Base cost

        // Execute transfer if value > 0
        if tx.value > 0 {
            if !self.execute_transfer(tx.from, tx.to, tx.value).await? {
                return Ok(ExecutionResult {
                    success: false,
                    gas_used,
                    return_data: vec![],
                    logs: vec!["Transfer failed".to_string()],
                    state_changes: 0,
                });
            }
            state_changes += 2; // from and to balance changes
            logs.push(format!(
                "Transferred {} from {} to {}",
                tx.value, tx.from, tx.to
            ));
        }

        // Simulate contract execution if data is provided
        if !tx.data.is_empty() {
            gas_used += self
                .simulate_contract_execution(&tx.data, tx.gas_limit - gas_used)
                .await?;
            logs.push("Contract execution completed".to_string());
            state_changes += 1;
        }

        // Update nonce
        self.state_db.set_nonce(tx.from, current_nonce + 1).await?;
        state_changes += 1;

        // Charge gas
        let total_gas_cost = gas_used * tx.gas_price;
        let remaining_balance = self.state_db.get_balance(tx.from).await?;
        self.state_db
            .set_balance(tx.from, remaining_balance - total_gas_cost)
            .await?;

        let execution_time = start_time.elapsed();
        logs.push(format!("Execution completed in {:?}", execution_time));

        Ok(ExecutionResult {
            success: true,
            gas_used,
            return_data: vec![42, 0, 1, 2], // Mock return data
            logs,
            state_changes,
        })
    }

    async fn simulate_contract_execution(&self, data: &[u8], gas_limit: u64) -> Result<u64> {
        // Simple opcode simulation
        let mut gas_used = 0;
        let mut pc = 0;

        while pc < data.len() && gas_used < gas_limit {
            match data.get(pc).unwrap_or(&0) {
                0x01 => gas_used += 3,        // ADD
                0x02 => gas_used += 5,        // MUL
                0x03 => gas_used += 5,        // SUB
                0x04 => gas_used += 5,        // DIV
                0x50 => gas_used += 2,        // POP
                0x51 => gas_used += 3,        // MLOAD
                0x52 => gas_used += 3,        // MSTORE
                0x54 => gas_used += 800,      // SLOAD
                0x55 => gas_used += 20000,    // SSTORE
                0x60..=0x7f => gas_used += 3, // PUSH
                _ => gas_used += 1,           // Default opcode cost
            }
            pc += 1;
        }

        Ok(gas_used.min(gas_limit))
    }

    pub async fn deploy_contract(
        &self,
        deployer: u64,
        code: Vec<u8>,
        initial_balance: u64,
    ) -> Result<u64> {
        let contract_address = self.generate_contract_address(deployer).await?;
        self.state_db
            .deploy_contract(contract_address, code, initial_balance)
            .await?;

        // Transfer initial balance to contract
        if initial_balance > 0 {
            self.execute_transfer(deployer, contract_address, initial_balance)
                .await?;
        }

        Ok(contract_address)
    }

    async fn generate_contract_address(&self, deployer: u64) -> Result<u64> {
        let nonce = self.state_db.get_nonce(deployer).await?;
        let block_height = self.state_db.get_block_height().await?;

        // Simple deterministic address generation
        Ok(((deployer * 1000 + nonce) * 1000 + block_height) % u64::MAX)
    }
}

// Basic VM Tests - No heavy dependencies
#[tokio::test]
async fn test_simple_vm_state_management() -> Result<()> {
    println!("🧪 Testing Simple VM State Management");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());

    // Test balance operations
    state_db.set_balance(1, 1000).await?;
    let balance = state_db.get_balance(1).await?;
    assert_eq!(balance, 1000, "Balance should be 1000 after setting");

    // Test nonce operations
    state_db.set_nonce(1, 42).await?;
    let nonce = state_db.get_nonce(1).await?;
    assert_eq!(nonce, 42, "Nonce should be 42 after setting");

    // Test state root calculation
    let root1 = state_db.get_state_root().await?;
    state_db.set_balance(2, 500).await?;
    let root2 = state_db.get_state_root().await?;
    assert_ne!(
        root1, root2,
        "State root should change after state modification"
    );

    println!("✅ Simple VM State Management test passed");
    Ok(())
}

#[tokio::test]
async fn test_simple_vm_storage() -> Result<()> {
    println!("🧪 Testing Simple VM Storage");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());

    // Test storage operations
    let contract_addr = 100;
    let key = b"test_key".to_vec();
    let value = b"test_value".to_vec();

    state_db
        .set_storage(contract_addr, key.clone(), value.clone())
        .await?;
    let retrieved = state_db.get_storage(contract_addr, &key).await?;

    assert_eq!(
        retrieved,
        Some(value),
        "Storage should return the set value"
    );

    // Test non-existent key
    let missing = state_db.get_storage(contract_addr, b"missing_key").await?;
    assert_eq!(missing, None, "Non-existent key should return None");

    println!("✅ Simple VM Storage test passed");
    Ok(())
}

#[tokio::test]
async fn test_simple_vm_transfers() -> Result<()> {
    println!("🧪 Testing Simple VM Transfers");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());
    let vm = SimpleVirtualMachine::new(state_db.clone());

    // Set initial balances
    state_db.set_balance(1, 1000).await?;
    state_db.set_balance(2, 500).await?;

    // Test successful transfer
    let success = vm.execute_transfer(1, 2, 200).await?;
    assert!(success, "Transfer should succeed with sufficient balance");

    let balance1 = state_db.get_balance(1).await?;
    let balance2 = state_db.get_balance(2).await?;
    assert_eq!(balance1, 800, "From balance should decrease");
    assert_eq!(balance2, 700, "To balance should increase");

    // Test insufficient balance
    let failed = vm.execute_transfer(1, 2, 1000).await?;
    assert!(!failed, "Transfer should fail with insufficient balance");

    println!("✅ Simple VM Transfers test passed");
    Ok(())
}

#[tokio::test]
async fn test_simple_vm_concurrent_access() -> Result<()> {
    println!("🧪 Testing Simple VM Concurrent Access");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());

    // Launch multiple concurrent operations
    let mut handles = vec![];

    for i in 0..10 {
        let db = state_db.clone();
        let handle = tokio::spawn(async move {
            db.set_balance(i, i * 100).await.unwrap();
            db.set_nonce(i, i * 10).await.unwrap();
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        handle.await?;
    }

    // Verify all balances were set correctly
    for i in 0..10 {
        let balance = state_db.get_balance(i).await?;
        let nonce = state_db.get_nonce(i).await?;
        assert_eq!(balance, i * 100, "Balance should be set correctly");
        assert_eq!(nonce, i * 10, "Nonce should be set correctly");
    }

    println!("✅ Simple VM Concurrent Access test passed");
    Ok(())
}

#[tokio::test]
async fn test_simple_vm_large_scale() -> Result<()> {
    println!("🧪 Testing Simple VM Large Scale Operations");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());

    // Create 1000 accounts
    for i in 0..1000 {
        state_db.set_balance(i, i * 1000).await?;
        state_db.set_nonce(i, i).await?;
    }

    // Verify state consistency
    let mut total_balance = 0u64;
    for i in 0..1000 {
        let balance = state_db.get_balance(i).await?;
        let nonce = state_db.get_nonce(i).await?;
        assert_eq!(
            balance,
            i * 1000,
            "Balance should be correct for account {}",
            i
        );
        assert_eq!(nonce, i, "Nonce should be correct for account {}", i);
        total_balance = total_balance.saturating_add(balance);
    }

    // Total balance should be sum of arithmetic series
    let expected_total = (0..1000).map(|i| i * 1000).sum::<u64>();
    assert_eq!(
        total_balance, expected_total,
        "Total balance should be correct"
    );

    println!("✅ Simple VM Large Scale test passed");
    Ok(())
}

#[tokio::test]
async fn test_simple_vm_state_root_determinism() -> Result<()> {
    println!("🧪 Testing Simple VM State Root Determinism");

    // Create two identical state databases
    let state_db1 = Arc::new(SimpleStateDB::new_in_memory());
    let state_db2 = Arc::new(SimpleStateDB::new_in_memory());

    // Apply identical operations
    for i in 0..10 {
        state_db1.set_balance(i, i * 100).await?;
        state_db2.set_balance(i, i * 100).await?;

        state_db1.set_nonce(i, i * 10).await?;
        state_db2.set_nonce(i, i * 10).await?;
    }

    // State roots should be identical
    let root1 = state_db1.get_state_root().await?;
    let root2 = state_db2.get_state_root().await?;
    assert_eq!(
        root1, root2,
        "Identical states should have identical state roots"
    );

    // Different state should have different root
    state_db2.set_balance(100, 999).await?;
    let root3 = state_db2.get_state_root().await?;
    assert_ne!(
        root1, root3,
        "Different states should have different state roots"
    );

    println!("✅ Simple VM State Root Determinism test passed");
    Ok(())
}

#[tokio::test]
async fn test_simple_vm_error_handling() -> Result<()> {
    println!("🧪 Testing Simple VM Error Handling");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());
    let vm = SimpleVirtualMachine::new(state_db.clone());

    // Test transfer with zero balance
    let result = vm.execute_transfer(999, 1000, 100).await?;
    assert!(
        !result,
        "Transfer from account with zero balance should fail"
    );

    // Verify balances unchanged
    let balance999 = state_db.get_balance(999).await?;
    let balance1000 = state_db.get_balance(1000).await?;
    assert_eq!(balance999, 0, "Source balance should remain zero");
    assert_eq!(balance1000, 0, "Destination balance should remain zero");

    // Test self-transfer
    state_db.set_balance(50, 500).await?;
    let self_transfer = vm.execute_transfer(50, 50, 100).await?;
    assert!(self_transfer, "Self-transfer should succeed");

    let balance50 = state_db.get_balance(50).await?;
    assert_eq!(balance50, 500, "Self-transfer should not change balance");

    println!("✅ Simple VM Error Handling test passed");
    Ok(())
}

// Advanced VM Tests

#[tokio::test]
async fn test_vm_gas_metering() -> Result<()> {
    println!("🧪 Testing VM Gas Metering");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());
    let vm = SimpleVirtualMachine::new(state_db.clone());

    // Set up account with sufficient balance
    state_db.set_balance(1, 1000000).await?;
    state_db.set_nonce(1, 0).await?;

    // Test simple transfer transaction
    let tx = Transaction {
        from: 1,
        to: 2,
        value: 100,
        gas_limit: 50000,
        gas_price: 20,
        data: vec![],
        nonce: 0,
    };

    let result = vm.execute_transaction(tx).await?;
    assert!(result.success, "Transaction should succeed");
    assert_eq!(result.gas_used, 42000, "Should use base gas + transfer gas"); // 21000 * 2
    assert!(!result.logs.is_empty(), "Should have execution logs");

    // Test contract execution with opcodes
    let contract_data = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x55]; // PUSH1 1 PUSH1 2 ADD SSTORE
    let tx_contract = Transaction {
        from: 1,
        to: 100,
        value: 0,
        gas_limit: 50000,
        gas_price: 20,
        data: contract_data,
        nonce: 1,
    };

    let result = vm.execute_transaction(tx_contract).await?;
    assert!(result.success, "Contract transaction should succeed");
    assert!(
        result.gas_used > 21000,
        "Should use more gas for contract execution"
    );

    println!("✅ VM Gas Metering test passed");
    Ok(())
}

#[tokio::test]
async fn test_vm_contract_deployment() -> Result<()> {
    println!("🧪 Testing VM Contract Deployment");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());
    let vm = SimpleVirtualMachine::new(state_db.clone());

    // Set up deployer account
    state_db.set_balance(1, 2000000).await?;

    // Deploy a contract
    let contract_code = vec![0x60, 0x80, 0x60, 0x40, 0x52]; // Simple initialization code
    let contract_address = vm.deploy_contract(1, contract_code.clone(), 1000).await?;

    assert!(contract_address > 0, "Contract address should be generated");

    // Verify contract was deployed
    let contract = state_db.get_contract(contract_address).await?;
    assert!(contract.is_some(), "Contract should exist");

    let contract = contract.unwrap();
    assert_eq!(contract.code, contract_code, "Contract code should match");
    assert_eq!(
        contract.balance, 1000,
        "Contract should have initial balance"
    );

    // Verify deployer balance was reduced
    let deployer_balance = state_db.get_balance(1).await?;
    assert!(
        deployer_balance < 2000000,
        "Deployer balance should be reduced"
    );

    println!("✅ VM Contract Deployment test passed");
    Ok(())
}

#[tokio::test]
async fn test_vm_transaction_validation() -> Result<()> {
    println!("🧪 Testing VM Transaction Validation");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());
    let vm = SimpleVirtualMachine::new(state_db.clone());

    // Set up account
    state_db.set_balance(1, 1000).await?;
    state_db.set_nonce(1, 5).await?;

    // Test invalid nonce
    let tx_bad_nonce = Transaction {
        from: 1,
        to: 2,
        value: 100,
        gas_limit: 50000,
        gas_price: 20,
        data: vec![],
        nonce: 3, // Wrong nonce
    };

    let result = vm.execute_transaction(tx_bad_nonce).await?;
    assert!(!result.success, "Transaction with bad nonce should fail");
    assert!(
        result.logs.contains(&"Invalid nonce".to_string()),
        "Should report nonce error"
    );

    // Test insufficient balance for gas
    let tx_insufficient = Transaction {
        from: 1,
        to: 2,
        value: 100,
        gas_limit: 50000,
        gas_price: 1000, // Very high gas price
        data: vec![],
        nonce: 5,
    };

    let result = vm.execute_transaction(tx_insufficient).await?;
    assert!(
        !result.success,
        "Transaction with insufficient balance should fail"
    );

    println!("✅ VM Transaction Validation test passed");
    Ok(())
}

#[tokio::test]
async fn test_vm_block_progression() -> Result<()> {
    println!("🧪 Testing VM Block Progression");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());

    // Initial block height should be 0
    let height = state_db.get_block_height().await?;
    assert_eq!(height, 0, "Initial block height should be 0");

    // Advance blocks and verify state root changes
    let root1 = state_db.get_state_root().await?;
    state_db.advance_block().await?;

    let height2 = state_db.get_block_height().await?;
    let root2 = state_db.get_state_root().await?;

    assert_eq!(height2, 1, "Block height should increment");
    assert_ne!(
        root1, root2,
        "State root should change with block advancement"
    );

    // Advance multiple blocks
    for _ in 0..10 {
        state_db.advance_block().await?;
    }

    let final_height = state_db.get_block_height().await?;
    assert_eq!(
        final_height, 11,
        "Block height should be correct after multiple advances"
    );

    println!("✅ VM Block Progression test passed");
    Ok(())
}

#[tokio::test]
async fn test_vm_performance_benchmarks() -> Result<()> {
    println!("🧪 Testing VM Performance Benchmarks");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());
    let vm = SimpleVirtualMachine::new(state_db.clone());

    // Setup accounts for benchmarking
    for i in 0..100 {
        state_db.set_balance(i, 1000000).await?;
        state_db.set_nonce(i, 0).await?;
    }

    // Benchmark simple transfers
    let start = Instant::now();
    let mut successful_txs = 0;

    for i in 0..100 {
        let tx = Transaction {
            from: i,
            to: (i + 1) % 100,
            value: 10,
            gas_limit: 50000,
            gas_price: 20,
            data: vec![],
            nonce: 0,
        };

        let result = vm.execute_transaction(tx).await?;
        if result.success {
            successful_txs += 1;
        }
    }

    let duration = start.elapsed();
    let tps = successful_txs as f64 / duration.as_secs_f64();

    assert_eq!(successful_txs, 100, "All transfers should succeed");
    assert!(tps > 1000.0, "Should achieve >1000 TPS, got {:.2}", tps);

    println!(
        "✅ Processed {} transactions in {:?} ({:.2} TPS)",
        successful_txs, duration, tps
    );

    // Benchmark contract executions
    let contract_data = vec![0x60; 50]; // 50 PUSH1 opcodes
    let start = Instant::now();

    for i in 0..50 {
        state_db.set_nonce(i, 1).await?; // Update nonce for second transaction
        let tx = Transaction {
            from: i,
            to: 200 + i,
            value: 0,
            gas_limit: 100000,
            gas_price: 20,
            data: contract_data.clone(),
            nonce: 1,
        };

        let _ = vm.execute_transaction(tx).await?;
    }

    let contract_duration = start.elapsed();
    let contract_tps = 50.0 / contract_duration.as_secs_f64();

    println!("✅ Contract executions: {:.2} TPS", contract_tps);
    println!("✅ VM Performance Benchmarks test passed");
    Ok(())
}

#[tokio::test]
async fn test_vm_stress_testing() -> Result<()> {
    println!("🧪 Testing VM Stress Testing");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());
    let vm = SimpleVirtualMachine::new(state_db.clone());

    // Create many accounts with concurrent operations
    let mut handles = vec![];

    for i in 0..50 {
        let db = state_db.clone();
        let handle = tokio::spawn(async move {
            // Create account with balance
            db.set_balance(i, (i + 1) * 10000).await.unwrap();
            db.set_nonce(i, 0).await.unwrap();

            // Add some storage data
            for j in 0..10 {
                let key = format!("key_{}", j).into_bytes();
                let value = format!("value_{}_{}", i, j).into_bytes();
                db.set_storage(i, key, value).await.unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for account creation
    for handle in handles {
        handle.await?;
    }

    // Stress test with many concurrent transactions
    let mut tx_handles = vec![];

    for i in 0..50 {
        let vm_clone = vm.clone();
        let handle = tokio::spawn(async move {
            let tx = Transaction {
                from: i,
                to: (i + 25) % 50,
                value: i * 100,
                gas_limit: 50000,
                gas_price: 20,
                data: vec![0x01, 0x02, 0x03], // Some contract data
                nonce: 0,
            };

            vm_clone.execute_transaction(tx).await.unwrap()
        });
        tx_handles.push(handle);
    }

    // Collect results
    let mut successful = 0;
    let mut total_gas = 0;

    for handle in tx_handles {
        let result = handle.await?;
        if result.success {
            successful += 1;
            total_gas += result.gas_used;
        }
    }

    assert!(
        successful >= 45,
        "Most transactions should succeed, got {}",
        successful
    );
    assert!(total_gas > 0, "Should have consumed gas");

    // Verify state consistency
    let final_state_root = state_db.get_state_root().await?;
    assert_ne!(final_state_root, [0u8; 32], "State root should be non-zero");

    println!(
        "✅ Stress test: {}/50 transactions successful, {} total gas",
        successful, total_gas
    );
    println!("✅ VM Stress Testing test passed");
    Ok(())
}

#[tokio::test]
async fn test_vm_edge_cases() -> Result<()> {
    println!("🧪 Testing VM Edge Cases");

    let state_db = Arc::new(SimpleStateDB::new_in_memory());
    let vm = SimpleVirtualMachine::new(state_db.clone());

    // Test maximum value transfers
    state_db.set_balance(1, u64::MAX).await?;
    let success = vm.execute_transfer(1, 2, u64::MAX - 100000).await?;
    assert!(success, "Max value transfer should succeed");

    // Test zero value transfer
    let success = vm.execute_transfer(2, 1, 0).await?;
    assert!(success, "Zero value transfer should succeed");

    // Test gas limit edge cases
    state_db.set_balance(3, 1000000).await?;
    state_db.set_nonce(3, 0).await?;

    let tx_max_gas = Transaction {
        from: 3,
        to: 4,
        value: 100,
        gas_limit: u64::MAX,
        gas_price: 1,
        data: vec![0x54; 1000], // Many SLOAD opcodes
        nonce: 0,
    };

    let result = vm.execute_transaction(tx_max_gas).await?;
    assert!(result.success, "Transaction with max gas should succeed");
    assert!(result.gas_used < u64::MAX, "Gas used should be reasonable");

    // Test empty contract code
    let empty_contract = vm.deploy_contract(3, vec![], 0).await?;
    assert!(
        empty_contract > 0,
        "Empty contract should deploy successfully"
    );

    println!("✅ VM Edge Cases test passed");
    Ok(())
}
