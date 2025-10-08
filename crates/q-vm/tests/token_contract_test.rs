/// Token Contract Transaction Tests
///
/// Tests real smart contract deployment and transaction execution
/// using the token.wat contract with the VM system

use q_vm::vm::{UltraContractProcessor, VmConfig, ExecutionResult};
use q_vm::state::StateDB;
use std::sync::Arc;
use std::fs;

/// Helper to load WASM bytecode from WAT file
fn load_token_contract() -> Vec<u8> {
    let wat_path = "examples/contracts/token.wat";
    let wat_content = fs::read_to_string(wat_path)
        .expect("Failed to read token.wat file");

    wat::parse_str(&wat_content)
        .expect("Failed to parse WAT to WASM")
}

/// Helper to create VM processor
fn create_vm() -> (Arc<UltraContractProcessor>, Arc<StateDB>) {
    let state_db = Arc::new(StateDB::new());
    let config = VmConfig {
        max_memory_pages: 10,
        max_stack_depth: 1024,
        gas_limit: 10_000_000,
        enable_simd: true,
        enable_parallel: true,
    };

    let processor = Arc::new(UltraContractProcessor::new(config, state_db.clone()));
    (processor, state_db)
}

#[tokio::test]
async fn test_deploy_token_contract() {
    let (vm, state_db) = create_vm();
    let bytecode = load_token_contract();

    println!("📦 Deploying token contract...");
    println!("   Bytecode size: {} bytes", bytecode.len());

    // Deploy contract
    let contract_address = "0xtoken_contract_001";
    let result = vm.deploy_contract(
        contract_address,
        &bytecode,
        "deployer_alice",
    ).await;

    assert!(result.is_ok(), "Contract deployment should succeed");
    println!("✅ Token contract deployed at: {}", contract_address);

    // Verify contract is stored
    let stored = state_db.get_contract(contract_address);
    assert!(stored.is_some(), "Contract should be stored in state");
    println!("✅ Contract verified in state database");
}

#[tokio::test]
async fn test_initialize_token_contract() {
    let (vm, _state_db) = create_vm();
    let bytecode = load_token_contract();

    let contract_address = "0xtoken_init_test";

    // Deploy contract
    vm.deploy_contract(contract_address, &bytecode, "deployer").await.unwrap();

    println!("\n💰 Initializing token with 1,000,000 supply...");

    // Initialize with total supply of 1,000,000
    let total_supply: u32 = 1_000_000;
    let args = total_supply.to_le_bytes().to_vec();

    let result = vm.execute_contract(
        contract_address,
        "init",
        &args,
        "deployer",
        1_000_000, // gas limit
    ).await;

    assert!(result.is_ok(), "Contract initialization should succeed");
    let exec_result = result.unwrap();

    println!("   Gas used: {}", exec_result.gas_used);
    println!("   Success: {}", exec_result.success);

    assert!(exec_result.success, "Initialization should return success");
    assert_eq!(
        i32::from_le_bytes(exec_result.return_data.as_slice().try_into().unwrap_or([0; 4])),
        1,
        "Should return 1 for success"
    );

    println!("✅ Token initialized with supply: {}", total_supply);
}

#[tokio::test]
async fn test_check_total_supply() {
    let (vm, _state_db) = create_vm();
    let bytecode = load_token_contract();
    let contract_address = "0xtoken_supply_test";

    // Deploy and initialize
    vm.deploy_contract(contract_address, &bytecode, "deployer").await.unwrap();

    let total_supply: u32 = 5_000_000;
    let args = total_supply.to_le_bytes().to_vec();
    vm.execute_contract(contract_address, "init", &args, "deployer", 1_000_000).await.unwrap();

    println!("\n📊 Checking total supply...");

    // Call totalSupply()
    let result = vm.execute_contract(
        contract_address,
        "totalSupply",
        &[],
        "anyone",
        500_000,
    ).await;

    assert!(result.is_ok(), "totalSupply call should succeed");
    let exec_result = result.unwrap();

    println!("   Gas used: {}", exec_result.gas_used);

    let returned_supply = i32::from_le_bytes(
        exec_result.return_data.as_slice().try_into().unwrap_or([0; 4])
    ) as u32;

    println!("   Returned supply: {}", returned_supply);
    assert_eq!(returned_supply, total_supply, "Total supply should match");

    println!("✅ Total supply verified: {}", returned_supply);
}

#[tokio::test]
async fn test_check_initial_balance() {
    let (vm, _state_db) = create_vm();
    let bytecode = load_token_contract();
    let contract_address = "0xtoken_balance_test";

    // Deploy and initialize
    vm.deploy_contract(contract_address, &bytecode, "deployer").await.unwrap();

    let total_supply: u32 = 10_000_000;
    let init_args = total_supply.to_le_bytes().to_vec();
    vm.execute_contract(contract_address, "init", &init_args, "deployer", 1_000_000).await.unwrap();

    println!("\n💵 Checking deployer's initial balance...");

    // Check balance of deployer (mock address 123456789)
    let owner_address: u32 = 123456789;
    let balance_args = owner_address.to_le_bytes().to_vec();

    let result = vm.execute_contract(
        contract_address,
        "balanceOf",
        &balance_args,
        "anyone",
        500_000,
    ).await;

    assert!(result.is_ok(), "balanceOf call should succeed");
    let exec_result = result.unwrap();

    println!("   Gas used: {}", exec_result.gas_used);

    let balance = i32::from_le_bytes(
        exec_result.return_data.as_slice().try_into().unwrap_or([0; 4])
    ) as u32;

    println!("   Owner balance: {}", balance);
    assert_eq!(balance, total_supply, "Owner should have all tokens initially");

    println!("✅ Initial balance verified: {}", balance);
}

#[tokio::test]
async fn test_transfer_tokens() {
    let (vm, _state_db) = create_vm();
    let bytecode = load_token_contract();
    let contract_address = "0xtoken_transfer_test";

    // Deploy and initialize
    vm.deploy_contract(contract_address, &bytecode, "alice").await.unwrap();

    let total_supply: u32 = 100_000;
    let init_args = total_supply.to_le_bytes().to_vec();
    vm.execute_contract(contract_address, "init", &init_args, "alice", 1_000_000).await.unwrap();

    println!("\n🔄 Testing token transfer...");
    println!("   Alice initial balance: {}", total_supply);

    // Transfer 30,000 tokens from Alice to Bob
    let bob_address: u32 = 987654321;
    let transfer_amount: u32 = 30_000;

    // Construct transfer args: (recipient_address, amount)
    let mut transfer_args = Vec::new();
    transfer_args.extend_from_slice(&bob_address.to_le_bytes());
    transfer_args.extend_from_slice(&transfer_amount.to_le_bytes());

    println!("   Transferring {} tokens to Bob...", transfer_amount);

    let result = vm.execute_contract(
        contract_address,
        "transfer",
        &transfer_args,
        "alice",
        2_000_000,
    ).await;

    assert!(result.is_ok(), "Transfer should succeed");
    let exec_result = result.unwrap();

    println!("   Gas used: {}", exec_result.gas_used);
    println!("   Success: {}", exec_result.success);

    let transfer_result = i32::from_le_bytes(
        exec_result.return_data.as_slice().try_into().unwrap_or([0; 4])
    );

    assert_eq!(transfer_result, 1, "Transfer should return success (1)");
    println!("✅ Transfer completed successfully");

    // Verify Alice's new balance
    let alice_address: u32 = 123456789;
    let alice_balance_args = alice_address.to_le_bytes().to_vec();

    let alice_balance_result = vm.execute_contract(
        contract_address,
        "balanceOf",
        &alice_balance_args,
        "anyone",
        500_000,
    ).await.unwrap();

    let alice_balance = i32::from_le_bytes(
        alice_balance_result.return_data.as_slice().try_into().unwrap_or([0; 4])
    ) as u32;

    println!("   Alice new balance: {}", alice_balance);
    assert_eq!(alice_balance, total_supply - transfer_amount, "Alice balance should be reduced");

    // Verify Bob's balance
    let bob_balance_args = bob_address.to_le_bytes().to_vec();
    let bob_balance_result = vm.execute_contract(
        contract_address,
        "balanceOf",
        &bob_balance_args,
        "anyone",
        500_000,
    ).await.unwrap();

    let bob_balance = i32::from_le_bytes(
        bob_balance_result.return_data.as_slice().try_into().unwrap_or([0; 4])
    ) as u32;

    println!("   Bob new balance: {}", bob_balance);
    assert_eq!(bob_balance, transfer_amount, "Bob should receive tokens");

    println!("✅ Balances verified after transfer");
}

#[tokio::test]
async fn test_transfer_insufficient_balance() {
    let (vm, _state_db) = create_vm();
    let bytecode = load_token_contract();
    let contract_address = "0xtoken_insufficient_test";

    // Deploy and initialize
    vm.deploy_contract(contract_address, &bytecode, "alice").await.unwrap();

    let total_supply: u32 = 50_000;
    let init_args = total_supply.to_le_bytes().to_vec();
    vm.execute_contract(contract_address, "init", &init_args, "alice", 1_000_000).await.unwrap();

    println!("\n❌ Testing transfer with insufficient balance...");
    println!("   Alice balance: {}", total_supply);

    // Try to transfer more than balance
    let bob_address: u32 = 987654321;
    let transfer_amount: u32 = 100_000; // More than Alice has

    let mut transfer_args = Vec::new();
    transfer_args.extend_from_slice(&bob_address.to_le_bytes());
    transfer_args.extend_from_slice(&transfer_amount.to_le_bytes());

    println!("   Attempting to transfer {} tokens (exceeds balance)...", transfer_amount);

    let result = vm.execute_contract(
        contract_address,
        "transfer",
        &transfer_args,
        "alice",
        2_000_000,
    ).await;

    assert!(result.is_ok(), "Call should succeed but return failure");
    let exec_result = result.unwrap();

    let transfer_result = i32::from_le_bytes(
        exec_result.return_data.as_slice().try_into().unwrap_or([0; 4])
    );

    assert_eq!(transfer_result, 0, "Transfer should return failure (0)");
    println!("✅ Transfer correctly rejected (insufficient balance)");

    // Verify Alice still has original balance
    let alice_address: u32 = 123456789;
    let alice_balance_args = alice_address.to_le_bytes().to_vec();

    let alice_balance_result = vm.execute_contract(
        contract_address,
        "balanceOf",
        &alice_balance_args,
        "anyone",
        500_000,
    ).await.unwrap();

    let alice_balance = i32::from_le_bytes(
        alice_balance_result.return_data.as_slice().try_into().unwrap_or([0; 4])
    ) as u32;

    println!("   Alice balance unchanged: {}", alice_balance);
    assert_eq!(alice_balance, total_supply, "Balance should be unchanged");

    println!("✅ Balance protection verified");
}

#[tokio::test]
async fn test_multiple_transactions() {
    let (vm, _state_db) = create_vm();
    let bytecode = load_token_contract();
    let contract_address = "0xtoken_multi_tx_test";

    // Deploy and initialize
    vm.deploy_contract(contract_address, &bytecode, "alice").await.unwrap();

    let total_supply: u32 = 1_000_000;
    let init_args = total_supply.to_le_bytes().to_vec();
    vm.execute_contract(contract_address, "init", &init_args, "alice", 1_000_000).await.unwrap();

    println!("\n🔄 Testing multiple transactions...");
    println!("   Initial supply: {}", total_supply);

    let bob_address: u32 = 100;
    let charlie_address: u32 = 200;

    // Transaction 1: Alice -> Bob (100,000 tokens)
    println!("\n   TX1: Alice -> Bob (100,000)");
    let mut tx1_args = Vec::new();
    tx1_args.extend_from_slice(&bob_address.to_le_bytes());
    tx1_args.extend_from_slice(&100_000u32.to_le_bytes());

    vm.execute_contract(contract_address, "transfer", &tx1_args, "alice", 2_000_000).await.unwrap();

    // Transaction 2: Alice -> Charlie (200,000 tokens)
    println!("   TX2: Alice -> Charlie (200,000)");
    let mut tx2_args = Vec::new();
    tx2_args.extend_from_slice(&charlie_address.to_le_bytes());
    tx2_args.extend_from_slice(&200_000u32.to_le_bytes());

    vm.execute_contract(contract_address, "transfer", &tx2_args, "alice", 2_000_000).await.unwrap();

    // Transaction 3: Bob -> Charlie (50,000 tokens)
    println!("   TX3: Bob -> Charlie (50,000)");
    let mut tx3_args = Vec::new();
    tx3_args.extend_from_slice(&charlie_address.to_le_bytes());
    tx3_args.extend_from_slice(&50_000u32.to_le_bytes());

    vm.execute_contract(contract_address, "transfer", &tx3_args, "bob", 2_000_000).await.unwrap();

    // Verify final balances
    println!("\n   📊 Final Balances:");

    // Alice balance
    let alice_address: u32 = 123456789;
    let alice_result = vm.execute_contract(
        contract_address,
        "balanceOf",
        &alice_address.to_le_bytes().to_vec(),
        "anyone",
        500_000,
    ).await.unwrap();
    let alice_balance = i32::from_le_bytes(alice_result.return_data.as_slice().try_into().unwrap()) as u32;
    println!("      Alice: {} (expected 700,000)", alice_balance);
    assert_eq!(alice_balance, 700_000);

    // Bob balance
    let bob_result = vm.execute_contract(
        contract_address,
        "balanceOf",
        &bob_address.to_le_bytes().to_vec(),
        "anyone",
        500_000,
    ).await.unwrap();
    let bob_balance = i32::from_le_bytes(bob_result.return_data.as_slice().try_into().unwrap()) as u32;
    println!("      Bob: {} (expected 50,000)", bob_balance);
    assert_eq!(bob_balance, 50_000);

    // Charlie balance
    let charlie_result = vm.execute_contract(
        contract_address,
        "balanceOf",
        &charlie_address.to_le_bytes().to_vec(),
        "anyone",
        500_000,
    ).await.unwrap();
    let charlie_balance = i32::from_le_bytes(charlie_result.return_data.as_slice().try_into().unwrap()) as u32;
    println!("      Charlie: {} (expected 250,000)", charlie_balance);
    assert_eq!(charlie_balance, 250_000);

    // Verify total (should still equal supply)
    let total = alice_balance + bob_balance + charlie_balance;
    println!("\n   ✅ Total tokens: {} (supply: {})", total, total_supply);
    assert_eq!(total, total_supply, "Total tokens should equal supply");

    println!("✅ All multiple transactions verified!");
}

#[tokio::test]
async fn test_parallel_contract_execution() {
    let (vm, _state_db) = create_vm();
    let bytecode = load_token_contract();
    let contract_address = "0xtoken_parallel_test";

    // Deploy and initialize
    vm.deploy_contract(contract_address, &bytecode, "deployer").await.unwrap();

    let total_supply: u32 = 10_000_000;
    let init_args = total_supply.to_le_bytes().to_vec();
    vm.execute_contract(contract_address, "init", &init_args, "deployer", 1_000_000).await.unwrap();

    println!("\n⚡ Testing parallel contract execution...");

    // Execute multiple balance queries in parallel
    let mut handles = Vec::new();

    for i in 0..10 {
        let vm_clone = vm.clone();
        let address: u32 = 123456789 + i;

        let handle = tokio::spawn(async move {
            let args = address.to_le_bytes().to_vec();
            vm_clone.execute_contract(
                contract_address,
                "balanceOf",
                &args,
                "anyone",
                500_000,
            ).await
        });

        handles.push(handle);
    }

    // Wait for all parallel executions
    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(_result)) = handle.await {
            success_count += 1;
        }
    }

    println!("   Parallel executions completed: {}/10", success_count);
    assert_eq!(success_count, 10, "All parallel executions should succeed");

    println!("✅ Parallel execution test passed!");
}

#[tokio::test]
async fn test_gas_metering() {
    let (vm, _state_db) = create_vm();
    let bytecode = load_token_contract();
    let contract_address = "0xtoken_gas_test";

    // Deploy and initialize
    vm.deploy_contract(contract_address, &bytecode, "deployer").await.unwrap();

    let total_supply: u32 = 1_000_000;
    let init_args = total_supply.to_le_bytes().to_vec();
    let init_result = vm.execute_contract(
        contract_address,
        "init",
        &init_args,
        "deployer",
        10_000_000,
    ).await.unwrap();

    println!("\n⛽ Gas metering test:");
    println!("   Init gas used: {}", init_result.gas_used);

    // Simple query (should use less gas)
    let query_result = vm.execute_contract(
        contract_address,
        "totalSupply",
        &[],
        "anyone",
        1_000_000,
    ).await.unwrap();

    println!("   Query gas used: {}", query_result.gas_used);

    // Transfer (should use more gas)
    let mut transfer_args = Vec::new();
    transfer_args.extend_from_slice(&999u32.to_le_bytes());
    transfer_args.extend_from_slice(&1000u32.to_le_bytes());

    let transfer_result = vm.execute_contract(
        contract_address,
        "transfer",
        &transfer_args,
        "deployer",
        10_000_000,
    ).await.unwrap();

    println!("   Transfer gas used: {}", transfer_result.gas_used);

    // Verify gas usage pattern
    assert!(init_result.gas_used > 0, "Init should use gas");
    assert!(query_result.gas_used > 0, "Query should use gas");
    assert!(transfer_result.gas_used > query_result.gas_used, "Transfer should use more gas than query");

    println!("✅ Gas metering verified!");
}
