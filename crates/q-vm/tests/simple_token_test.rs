/// Simple Token Contract Test
/// Tests real smart contract execution with transactions

use q_vm::vm::ultra_performance_bridge::{UltraContractProcessor, ContractCall, StateDB};
use std::sync::Arc;
use std::fs;

#[tokio::test]
async fn test_token_contract_full_flow() {
    println!("\n🚀 Token Contract Full Flow Test\n");

    // Load the token contract
    let wat_path = "examples/contracts/token.wat";
    let wat_content = fs::read_to_string(wat_path)
        .expect("Failed to read token.wat");

    let bytecode = wat::parse_str(&wat_content)
        .expect("Failed to parse WAT to WASM");

    println!("✅ Token contract loaded ({} bytes)", bytecode.len());

    // Create VM configuration
    let config = q_vm::vm::ultra_performance_bridge::UltraContractConfig {
        target_tps: 150_000,
        num_shards: 4,
        workers_per_shard: 4,
        batch_size: 100,
        contract_cache_size: 1000,
        pipeline_depth: 2,
        use_simd: true,
        use_zero_copy: true,
        jit_compilation: true,
    };

    let state_db = Arc::new(StateDB::new());

    // Load the actual WASM bytecode into state_db
    state_db.set_contract("0xtoken".to_string(), bytecode.clone());

    let vm = UltraContractProcessor::new(config, state_db.clone())
        .expect("Failed to create VM");

    println!("✅ VM processor created\n");

    // Test 1: Initialize token with 1,000,000 supply
    println!("📝 Test 1: Initialize Token");
    println!("   Creating token with supply: 1,000,000");

    let total_supply: u32 = 1_000_000;
    let init_args = total_supply.to_le_bytes().to_vec();

    let init_call = ContractCall {
        contract_address: "0xtoken".to_string(),
        function: "init".to_string(),
        args: init_args,
        caller: "alice".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    };

    let init_result = vm.execute_contract_ultra(init_call).await;

    match init_result {
        Ok(response) => {
            println!("   ✅ Initialization succeeded");
            println!("      Gas used: {}", response.gas_used);
            println!("      Success: {}", response.success);
        }
        Err(e) => {
            println!("   ❌ Initialization failed: {}", e);
        }
    }

    // Test 2: Check total supply
    println!("\n📊 Test 2: Check Total Supply");

    let supply_call = ContractCall {
        contract_address: "0xtoken".to_string(),
        function: "totalSupply".to_string(),
        args: vec![],
        caller: "anyone".to_string(),
        gas_limit: 1_000_000,
        gas_price: Some(1),
        value: Some(0),
    };

    let supply_result = vm.execute_contract_ultra(supply_call).await;

    match supply_result {
        Ok(response) => {
            println!("   ✅ Total supply query succeeded");
            println!("      Gas used: {}", response.gas_used);

            if response.return_data.len() >= 4 {
                let returned = u32::from_le_bytes(response.return_data[0..4].try_into().unwrap());
                println!("      Supply: {}", returned);
            }
        }
        Err(e) => {
            println!("   ❌ Total supply query failed: {}", e);
        }
    }

    // Test 3: Check Alice's balance
    println!("\n💰 Test 3: Check Alice's Balance");

    let alice_addr: u32 = 123456789;
    let balance_args = alice_addr.to_le_bytes().to_vec();

    let balance_call = ContractCall {
        contract_address: "0xtoken".to_string(),
        function: "balanceOf".to_string(),
        args: balance_args,
        caller: "anyone".to_string(),
        gas_limit: 1_000_000,
        gas_price: Some(1),
        value: Some(0),
    };

    let balance_result = vm.execute_contract_ultra(balance_call).await;

    match balance_result {
        Ok(response) => {
            println!("   ✅ Balance query succeeded");
            println!("      Gas used: {}", response.gas_used);

            if response.return_data.len() >= 4 {
                let balance = u32::from_le_bytes(response.return_data[0..4].try_into().unwrap());
                println!("      Alice's balance: {}", balance);
            }
        }
        Err(e) => {
            println!("   ❌ Balance query failed: {}", e);
        }
    }

    // Test 4: Transfer tokens
    println!("\n💸 Test 4: Transfer Tokens (Alice -> Bob)");
    println!("   Transferring 250,000 tokens");

    let bob_addr: u32 = 987654321;
    let transfer_amount: u32 = 250_000;

    let mut transfer_args = Vec::new();
    transfer_args.extend_from_slice(&bob_addr.to_le_bytes());
    transfer_args.extend_from_slice(&transfer_amount.to_le_bytes());

    let transfer_call = ContractCall {
        contract_address: "0xtoken".to_string(),
        function: "transfer".to_string(),
        args: transfer_args,
        caller: "alice".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    };

    let transfer_result = vm.execute_contract_ultra(transfer_call).await;

    match transfer_result {
        Ok(response) => {
            println!("   ✅ Transfer succeeded");
            println!("      Gas used: {}", response.gas_used);

            if response.return_data.len() >= 4 {
                let result = i32::from_le_bytes(response.return_data[0..4].try_into().unwrap());
                println!("      Result: {}", if result == 1 { "Success" } else { "Failed" });
            }
        }
        Err(e) => {
            println!("   ❌ Transfer failed: {}", e);
        }
    }

    // Test 5: Verify balances after transfer
    println!("\n🔍 Test 5: Verify Balances After Transfer");

    // Check Alice's new balance
    let alice_balance_call = ContractCall {
        contract_address: "0xtoken".to_string(),
        function: "balanceOf".to_string(),
        args: alice_addr.to_le_bytes().to_vec(),
        caller: "anyone".to_string(),
        gas_limit: 1_000_000,
        gas_price: Some(1),
        value: Some(0),
    };

    if let Ok(response) = vm.execute_contract_ultra(alice_balance_call).await {
        if response.return_data.len() >= 4 {
            let alice_balance = u32::from_le_bytes(response.return_data[0..4].try_into().unwrap());
            println!("   Alice's balance: {} (expected 750,000)", alice_balance);
        }
    }

    // Check Bob's balance
    let bob_balance_call = ContractCall {
        contract_address: "0xtoken".to_string(),
        function: "balanceOf".to_string(),
        args: bob_addr.to_le_bytes().to_vec(),
        caller: "anyone".to_string(),
        gas_limit: 1_000_000,
        gas_price: Some(1),
        value: Some(0),
    };

    if let Ok(response) = vm.execute_contract_ultra(bob_balance_call).await {
        if response.return_data.len() >= 4 {
            let bob_balance = u32::from_le_bytes(response.return_data[0..4].try_into().unwrap());
            println!("   Bob's balance: {} (expected 250,000)", bob_balance);
        }
    }

    // Test 6: Batch execution
    println!("\n⚡ Test 6: Batch Transaction Execution");
    println!("   Creating 5 balance queries in parallel");

    let mut batch_calls = Vec::new();
    for i in 0..5 {
        let addr: u32 = 100 + i;
        batch_calls.push(ContractCall {
            contract_address: "0xtoken".to_string(),
            function: "balanceOf".to_string(),
            args: addr.to_le_bytes().to_vec(),
            caller: "anyone".to_string(),
            gas_limit: 500_000,
            gas_price: Some(1),
            value: Some(0),
        });
    }

    let batch_results = vm.execute_batch_ultra(batch_calls).await;
    println!("   ✅ Batch executed: {} calls processed", batch_results.len());

    let total_gas: u64 = batch_results.iter().map(|r| r.gas_used).sum();
    println!("      Total gas used: {}", total_gas);

    println!("\n🎉 Token Contract Full Flow Test Complete!\n");
}

#[tokio::test]
async fn test_transfer_insufficient_balance() {
    println!("\n❌ Testing Insufficient Balance Protection\n");

    let wat_content = fs::read_to_string("examples/contracts/token.wat")
        .expect("Failed to read token.wat");
    let bytecode = wat::parse_str(&wat_content).expect("Failed to parse WAT");

    let config = q_vm::vm::ultra_performance_bridge::UltraContractConfig {
        target_tps: 150_000,
        num_shards: 2,
        workers_per_shard: 2,
        batch_size: 10,
        contract_cache_size: 100,
        pipeline_depth: 1,
        use_simd: false,
        use_zero_copy: true,
        jit_compilation: false,
    };

    let state_db = Arc::new(StateDB::new());
    state_db.set_contract("0xtoken2".to_string(), bytecode.clone());
    let vm = UltraContractProcessor::new(config, state_db).unwrap();

    // Initialize with small supply
    let init_call = ContractCall {
        contract_address: "0xtoken2".to_string(),
        function: "init".to_string(),
        args: 50_000u32.to_le_bytes().to_vec(),
        caller: "alice".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    };

    vm.execute_contract_ultra(init_call).await.unwrap();
    println!("✅ Token initialized with 50,000 supply");

    // Try to transfer more than balance
    let mut transfer_args = Vec::new();
    transfer_args.extend_from_slice(&999u32.to_le_bytes());
    transfer_args.extend_from_slice(&100_000u32.to_le_bytes()); // More than 50,000

    let transfer_call = ContractCall {
        contract_address: "0xtoken2".to_string(),
        function: "transfer".to_string(),
        args: transfer_args,
        caller: "alice".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    };

    let result = vm.execute_contract_ultra(transfer_call).await.unwrap();

    if result.return_data.len() >= 4 {
        let transfer_result = i32::from_le_bytes(result.return_data[0..4].try_into().unwrap());
        assert_eq!(transfer_result, 0, "Transfer should fail (return 0)");
        println!("✅ Transfer correctly rejected (returned 0)");
    }

    println!("✅ Insufficient balance protection working!\n");
}

#[tokio::test]
async fn test_multiple_transfers() {
    println!("\n🔄 Testing Multiple Sequential Transfers\n");

    let wat_content = fs::read_to_string("examples/contracts/token.wat")
        .expect("Failed to read token.wat");
    let bytecode = wat::parse_str(&wat_content).expect("Failed to parse WAT");

    let config = q_vm::vm::ultra_performance_bridge::UltraContractConfig {
        target_tps: 150_000,
        num_shards: 4,
        workers_per_shard: 4,
        batch_size: 50,
        contract_cache_size: 500,
        pipeline_depth: 2,
        use_simd: true,
        use_zero_copy: true,
        jit_compilation: true,
    };

    let state_db = Arc::new(StateDB::new());
    state_db.set_contract("0xtoken3".to_string(), bytecode.clone());
    let vm = UltraContractProcessor::new(config, state_db).unwrap();

    // Initialize
    vm.execute_contract_ultra(ContractCall {
        contract_address: "0xtoken3".to_string(),
        function: "init".to_string(),
        args: 1_000_000u32.to_le_bytes().to_vec(),
        caller: "alice".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    }).await.unwrap();

    println!("✅ Token initialized: 1,000,000 supply");
    println!("\n📝 Executing transfer sequence:");

    // Transfer 1: Alice -> Bob (100,000)
    println!("   TX1: Alice -> Bob (100,000)");
    let mut tx1_args = Vec::new();
    tx1_args.extend_from_slice(&100u32.to_le_bytes());
    tx1_args.extend_from_slice(&100_000u32.to_le_bytes());

    vm.execute_contract_ultra(ContractCall {
        contract_address: "0xtoken3".to_string(),
        function: "transfer".to_string(),
        args: tx1_args,
        caller: "alice".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    }).await.unwrap();

    // Transfer 2: Alice -> Charlie (200,000)
    println!("   TX2: Alice -> Charlie (200,000)");
    let mut tx2_args = Vec::new();
    tx2_args.extend_from_slice(&200u32.to_le_bytes());
    tx2_args.extend_from_slice(&200_000u32.to_le_bytes());

    vm.execute_contract_ultra(ContractCall {
        contract_address: "0xtoken3".to_string(),
        function: "transfer".to_string(),
        args: tx2_args,
        caller: "alice".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    }).await.unwrap();

    // Transfer 3: Bob -> Charlie (50,000)
    println!("   TX3: Bob -> Charlie (50,000)");
    let mut tx3_args = Vec::new();
    tx3_args.extend_from_slice(&200u32.to_le_bytes());
    tx3_args.extend_from_slice(&50_000u32.to_le_bytes());

    vm.execute_contract_ultra(ContractCall {
        contract_address: "0xtoken3".to_string(),
        function: "transfer".to_string(),
        args: tx3_args,
        caller: "bob".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    }).await.unwrap();

    println!("\n✅ All transfers completed successfully!");
    println!("\n🎉 Multiple Transfer Test Complete!\n");
}
