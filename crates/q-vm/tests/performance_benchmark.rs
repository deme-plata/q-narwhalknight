/// Performance Benchmark - Real WASM Smart Contract Execution
/// Tests compilation caching, parallel execution, and TPS measurement

use q_vm::vm::ultra_performance_bridge::{UltraContractProcessor, ContractCall, StateDB, UltraContractConfig};
use std::sync::Arc;
use std::time::Instant;
use std::fs;

#[tokio::test]
async fn benchmark_compilation_caching() {
    println!("\n🚀 WASM Compilation Caching Benchmark\n");

    let wat_content = fs::read_to_string("examples/contracts/token.wat")
        .expect("Failed to read token.wat");
    let bytecode = wat::parse_str(&wat_content).expect("Failed to parse WAT");

    let config = UltraContractConfig {
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
    state_db.set_contract("0xtoken".to_string(), bytecode.clone());
    let vm = UltraContractProcessor::new(config, state_db).unwrap();

    // Initialize token
    let init_call = ContractCall {
        contract_address: "0xtoken".to_string(),
        function: "init".to_string(),
        args: 10_000_000u32.to_le_bytes().to_vec(),
        caller: "alice".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    };

    vm.execute_contract_ultra(init_call).await.unwrap();

    // First execution - cold (compilation required)
    let start = Instant::now();
    let balance_call = ContractCall {
        contract_address: "0xtoken".to_string(),
        function: "balanceOf".to_string(),
        args: 123u32.to_le_bytes().to_vec(),
        caller: "anyone".to_string(),
        gas_limit: 1_000_000,
        gas_price: Some(1),
        value: Some(0),
    };

    vm.execute_contract_ultra(balance_call.clone()).await.unwrap();
    let cold_time = start.elapsed();
    println!("❄️  Cold execution (compile + run): {:?}", cold_time);

    // Second execution - warm (cached module)
    let start = Instant::now();
    vm.execute_contract_ultra(balance_call.clone()).await.unwrap();
    let warm_time = start.elapsed();
    println!("🔥 Warm execution (cached):        {:?}", warm_time);

    let speedup = cold_time.as_micros() as f64 / warm_time.as_micros() as f64;
    println!("\n✨ Cache speedup: {:.2}x faster\n", speedup);

    assert!(warm_time < cold_time, "Cached execution should be faster");
}

#[tokio::test]
async fn benchmark_parallel_execution() {
    println!("\n⚡ Parallel Contract Execution Benchmark\n");

    let wat_content = fs::read_to_string("examples/contracts/token.wat")
        .expect("Failed to read token.wat");
    let bytecode = wat::parse_str(&wat_content).expect("Failed to parse WAT");

    let config = UltraContractConfig {
        target_tps: 150_000,
        num_shards: 16,
        workers_per_shard: 8,
        batch_size: 1000,
        contract_cache_size: 10000,
        pipeline_depth: 4,
        use_simd: true,
        use_zero_copy: true,
        jit_compilation: true,
    };

    let state_db = Arc::new(StateDB::new());
    state_db.set_contract("0xtoken".to_string(), bytecode.clone());
    let vm = UltraContractProcessor::new(config, state_db).unwrap();

    // Initialize token
    vm.execute_contract_ultra(ContractCall {
        contract_address: "0xtoken".to_string(),
        function: "init".to_string(),
        args: 100_000_000u32.to_le_bytes().to_vec(),
        caller: "alice".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    }).await.unwrap();

    // Create batch of balance queries
    let batch_sizes = vec![100, 500, 1000, 5000];

    for batch_size in batch_sizes {
        let mut calls = Vec::new();
        for i in 0..batch_size {
            calls.push(ContractCall {
                contract_address: "0xtoken".to_string(),
                function: "balanceOf".to_string(),
                args: (100 + i).to_le_bytes().to_vec(),
                caller: "anyone".to_string(),
                gas_limit: 500_000,
                gas_price: Some(1),
                value: Some(0),
            });
        }

        let start = Instant::now();
        let results = vm.execute_batch_ultra(calls).await;
        let duration = start.elapsed();

        let tps = (batch_size as f64 / duration.as_secs_f64()) as u64;
        let latency = duration.as_micros() / batch_size as u128;

        println!("📊 Batch size: {}", batch_size);
        println!("   TPS:     {:>10}", format!("{} tx/s", tps));
        println!("   Latency: {:>10} μs/tx", latency);
        println!("   Total:   {:>10?}", duration);
        println!();

        assert_eq!(results.len(), batch_size, "All calls should complete");
    }
}

#[tokio::test]
async fn benchmark_tps_target() {
    println!("\n🎯 150K+ TPS Target Benchmark\n");

    let wat_content = fs::read_to_string("examples/contracts/token.wat")
        .expect("Failed to read token.wat");
    let bytecode = wat::parse_str(&wat_content).expect("Failed to parse WAT");

    let config = UltraContractConfig {
        target_tps: 150_000,
        num_shards: 16,
        workers_per_shard: 8,
        batch_size: 1000,
        contract_cache_size: 10000,
        pipeline_depth: 4,
        use_simd: true,
        use_zero_copy: true,
        jit_compilation: true,
    };

    let state_db = Arc::new(StateDB::new());
    state_db.set_contract("0xtoken".to_string(), bytecode.clone());
    let vm = UltraContractProcessor::new(config, state_db).unwrap();

    // Initialize
    vm.execute_contract_ultra(ContractCall {
        contract_address: "0xtoken".to_string(),
        function: "init".to_string(),
        args: 1_000_000_000u32.to_le_bytes().to_vec(),
        caller: "alice".to_string(),
        gas_limit: 5_000_000,
        gas_price: Some(1),
        value: Some(0),
    }).await.unwrap();

    // Run sustained load test
    let test_duration = std::time::Duration::from_secs(5);
    let start = Instant::now();
    let mut total_txs = 0;

    while start.elapsed() < test_duration {
        let mut batch = Vec::new();
        for i in 0..1000 {
            batch.push(ContractCall {
                contract_address: "0xtoken".to_string(),
                function: "balanceOf".to_string(),
                args: (total_txs + i).to_le_bytes().to_vec(),
                caller: "anyone".to_string(),
                gas_limit: 500_000,
                gas_price: Some(1),
                value: Some(0),
            });
        }

        vm.execute_batch_ultra(batch).await;
        total_txs += 1000;
    }

    let duration = start.elapsed();
    let tps = (total_txs as f64 / duration.as_secs_f64()) as u64;

    println!("🏁 Test Results:");
    println!("   Duration:  {:?}", duration);
    println!("   Total TXs: {}", total_txs);
    println!("   TPS:       {} tx/s", tps);
    println!();

    if tps >= 150_000 {
        println!("✅ SUCCESS: Achieved 150K+ TPS target!");
    } else {
        println!("⚠️  Current TPS: {}, Target: 150K", tps);
    }

    println!();
}
