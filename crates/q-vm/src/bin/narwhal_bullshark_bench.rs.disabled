//! Benchmark tool for Narwhal-Bullshark VM

use clap::Parser;
use dagknight_vm::config;
use dagknight_vm::state::StateDB;
use dagknight_vm::vm::narwhal_bullshark_vm::{NarwhalBullsharkVm, SmartContractTx};
use dagknight_vm::vm::VirtualMachine;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex; // Add this import

#[derive(Parser, Debug)]
#[clap(author, version, about = "Benchmark tool for Narwhal-Bullshark VM")]
struct Args {
    /// Number of transactions to generate
    #[clap(short, long, default_value = "10000")]
    transactions: usize,

    /// Batch size for transactions
    #[clap(short, long, default_value = "100")]
    batch_size: usize,

    /// Number of nodes to simulate
    #[clap(short, long, default_value = "4")]
    nodes: usize,

    /// Run mode (single, multi, stress)
    #[clap(short, long, default_value = "single")]
    mode: String,

    /// Test duration in seconds (for stress test)
    #[clap(short, long, default_value = "60")]
    duration: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Starting Narwhal-Bullshark VM benchmark");
    println!("----------------------------------------");
    println!("Transactions: {}", args.transactions);
    println!("Batch size: {}", args.batch_size);
    println!("Nodes: {}", args.nodes);
    println!("Mode: {}", args.mode);

    // Load configuration
    let config_path = "config/vm_config.toml";
    match config::load_config(config_path) {
        Ok(_) => println!("Loaded configuration from {}", config_path),
        Err(e) => eprintln!("Warning: Failed to load configuration: {}", e),
    }

    // Update batch size from arguments
    config::update_batch_size(args.batch_size);

    // Create nodes
    let mut node_ids = Vec::new();
    for i in 0..args.nodes {
        node_ids.push(format!("node_{}", i));
    }

    // Create virtual machine - Fix the StateDB path
    let vm = Arc::new(VirtualMachine::new(Arc::new(StateDB::new())));

    match args.mode.as_str() {
        "single" => {
            // Run single-node benchmark
            println!("\nRunning single-node benchmark...");
            run_single_node_benchmark(
                node_ids[0].clone(),
                node_ids[1..].to_vec(),
                vm.clone(),
                args.transactions,
                args.batch_size,
            )
            .await?;
        }
        "multi" => {
            // Run multi-node benchmark
            println!("\nRunning multi-node benchmark...");
            run_multi_node_benchmark(
                node_ids.clone(),
                vm.clone(),
                args.transactions,
                args.batch_size,
            )
            .await?;
        }
        "stress" => {
            // Run stress benchmark
            println!(
                "\nRunning stress benchmark for {} seconds...",
                args.duration
            );
            run_stress_benchmark(node_ids.clone(), vm.clone(), args.batch_size, args.duration)
                .await?;
        }
        _ => {
            eprintln!(
                "Unknown mode: {}. Must be 'single', 'multi', or 'stress'.",
                args.mode
            );
            std::process::exit(1);
        }
    }

    Ok(())
}

// Run benchmark with a single node
async fn run_single_node_benchmark(
    node_id: String,
    peers: Vec<String>,
    vm: Arc<VirtualMachine>,
    transaction_count: usize,
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create Narwhal-Bullshark VM
    let nb_vm = Arc::new(NarwhalBullsharkVm::new(node_id, peers, vm));

    // Start VM
    nb_vm.start().await?;

    // Allow time for initialization
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Generate and submit transactions
    let start_time = Instant::now();
    let mut completed = 0;

    println!(
        "Generating and submitting {} transactions...",
        transaction_count
    );

    // Create batches of transactions
    for batch_num in 0..(transaction_count / batch_size + 1) {
        let batch_start = batch_num * batch_size;
        let batch_end = std::cmp::min(batch_start + batch_size, transaction_count);

        if batch_start >= batch_end {
            break;
        }

        let batch_size = batch_end - batch_start;
        println!(
            "Submitting batch {} with {} transactions...",
            batch_num + 1,
            batch_size
        );

        // Submit transactions in parallel
        let mut handles = Vec::new();

        for i in batch_start..batch_end {
            let vm_clone = nb_vm.clone();

            let handle = tokio::spawn(async move {
                // Create a smart contract transaction
                let tx = SmartContractTx {
                    address: 1000, // Example contract address
                    function: "transfer".to_string(),
                    arguments: vec![1, 2, 3, 4], // Example arguments
                    sender: 101,
                    gas_limit: 100000,
                    gas_price: 1,
                    nonce: i as u64,
                    value: 0,
                    signature: [0; 64], // Example signature
                };

                // Submit transaction
                match vm_clone.submit_transaction(tx).await {
                    Ok(_) => true,
                    Err(e) => {
                        eprintln!("Failed to submit transaction {}: {:?}", i, e);
                        false
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all submissions to complete
        for handle in handles {
            if let Ok(success) = handle.await {
                if success {
                    completed += 1;
                }
            }
        }

        // Progress update
        let progress = completed as f64 / transaction_count as f64 * 100.0;
        println!(
            "Progress: {}/{} transactions ({:.1}%)",
            completed, transaction_count, progress
        );

        // Short delay between batches to avoid overwhelming the system
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Calculate throughput
    let elapsed = start_time.elapsed();
    let tps = completed as f64 / elapsed.as_secs_f64();

    println!("\nBenchmark Results:");
    println!("  Transactions submitted: {}", completed);
    println!("  Elapsed time: {:.2} seconds", elapsed.as_secs_f64());
    println!("  Throughput: {:.2} TPS", tps);

    // Allow time for processing to complete
    println!("\nWaiting for all transactions to be processed...");
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Get current TPS from VM
    let vm_tps = nb_vm.get_tps().await;
    println!("  VM reported TPS: {:.2}", vm_tps);

    // Stop VM
    nb_vm.stop().await?;

    Ok(())
}

// Run benchmark with multiple nodes
async fn run_multi_node_benchmark(
    node_ids: Vec<String>,
    vm: Arc<VirtualMachine>,
    transaction_count: usize,
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Creating {} nodes for multi-node benchmark...",
        node_ids.len()
    );

    // Create and start VMs for each node
    let mut vms = Vec::new();

    for (i, node_id) in node_ids.iter().enumerate() {
        // Create peers list (all other nodes)
        let peers: Vec<String> = node_ids
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, id)| id.clone())
            .collect();

        // Create VM
        let nb_vm = Arc::new(NarwhalBullsharkVm::new(node_id.clone(), peers, vm.clone()));

        // Start VM
        nb_vm.start().await?;

        vms.push(nb_vm);

        println!("Started node {} with {} peers", node_id, node_ids.len() - 1);
    }

    // Allow time for nodes to connect
    println!("Allowing time for nodes to connect...");
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Run benchmark on the first node
    println!("Running benchmark on node {}...", node_ids[0]);
    let nb_vm = &vms[0];

    // Generate and submit transactions
    let start_time = Instant::now();
    let mut completed = 0;

    println!(
        "Generating and submitting {} transactions...",
        transaction_count
    );

    // Create batches of transactions
    for batch_num in 0..(transaction_count / batch_size + 1) {
        let batch_start = batch_num * batch_size;
        let batch_end = std::cmp::min(batch_start + batch_size, transaction_count);

        if batch_start >= batch_end {
            break;
        }

        let batch_size = batch_end - batch_start;
        println!(
            "Submitting batch {} with {} transactions...",
            batch_num + 1,
            batch_size
        );

        // Submit transactions in parallel
        let mut handles = Vec::new();

        for i in batch_start..batch_end {
            let vm_clone = nb_vm.clone();

            let handle = tokio::spawn(async move {
                // Create a smart contract transaction
                let tx = SmartContractTx {
                    address: 1000, // Example contract address
                    function: "transfer".to_string(),
                    arguments: vec![1, 2, 3, 4], // Example arguments
                    sender: 101,
                    gas_limit: 100000,
                    gas_price: 1,
                    nonce: i as u64,
                    value: 0,
                    signature: [0; 64], // Example signature
                };

                // Submit transaction
                match vm_clone.submit_transaction(tx).await {
                    Ok(_) => true,
                    Err(e) => {
                        eprintln!("Failed to submit transaction {}: {:?}", i, e);
                        false
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all submissions to complete
        for handle in handles {
            if let Ok(success) = handle.await {
                if success {
                    completed += 1;
                }
            }
        }

        // Progress update
        let progress = completed as f64 / transaction_count as f64 * 100.0;
        println!(
            "Progress: {}/{} transactions ({:.1}%)",
            completed, transaction_count, progress
        );

        // Short delay between batches to avoid overwhelming the system
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Calculate throughput
    let elapsed = start_time.elapsed();
    let tps = completed as f64 / elapsed.as_secs_f64();

    println!("\nBenchmark Results:");
    println!("  Transactions submitted: {}", completed);
    println!("  Elapsed time: {:.2} seconds", elapsed.as_secs_f64());
    println!("  Throughput: {:.2} TPS", tps);

    // Allow time for processing to complete
    println!("\nWaiting for all transactions to be processed...");
    tokio::time::sleep(Duration::from_secs(10)).await;

    // Get TPS from each node
    println!("\nTPS reported by each node:");
    for (i, vm) in vms.iter().enumerate() {
        let node_tps = vm.get_tps().await;
        println!("  Node {}: {:.2} TPS", node_ids[i], node_tps);
    }

    // Stop all VMs
    println!("\nStopping all nodes...");
    for (i, vm) in vms.iter().enumerate() {
        vm.stop().await?;
        println!("Stopped node {}", node_ids[i]);
    }

    Ok(())
}

// Run stress benchmark
async fn run_stress_benchmark(
    node_ids: Vec<String>,
    vm: Arc<VirtualMachine>,
    batch_size: usize,
    duration_secs: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting stress benchmark for {} seconds...", duration_secs);

    // Create and start VM
    let nb_vm = Arc::new(NarwhalBullsharkVm::new(
        node_ids[0].clone(),
        node_ids[1..].to_vec(),
        vm.clone(),
    ));

    // Start VM
    nb_vm.start().await?;

    // Allow time for initialization
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Counters for stress test
    let transaction_counter = Arc::new(Mutex::new(0));
    let stop_flag = Arc::new(Mutex::new(false));

    // Start metrics reporting
    let tc_clone = transaction_counter.clone();
    let metrics_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        let start_time = Instant::now();
        let mut last_count = 0;

        loop {
            interval.tick().await;

            let elapsed = start_time.elapsed();
            let count = *tc_clone.lock().await;

            // Calculate incremental and overall TPS
            let incremental_tps = (count - last_count) as f64;
            let overall_tps = count as f64 / elapsed.as_secs_f64().max(1.0);

            println!(
                "[{:4.1}s] Transactions: {} (+{}), TPS: {:.2} (current: {:.2})",
                elapsed.as_secs_f64(),
                count,
                count - last_count,
                overall_tps,
                incremental_tps
            );

            last_count = count;

            if elapsed.as_secs() >= duration_secs {
                break;
            }
        }
    });

    // Start transaction generation
    let sf_clone = stop_flag.clone();
    let tc_clone = transaction_counter.clone();
    let vm_clone = nb_vm.clone();

    let generator_handle = tokio::spawn(async move {
        let mut batch_num = 0;

        loop {
            // Check if we should stop
            if *sf_clone.lock().await {
                break;
            }

            // Submit a batch of transactions
            let batch_start = batch_num * batch_size;
            let mut submitted = 0;

            // Submit transactions in parallel
            let mut handles = Vec::new();

            for i in 0..batch_size {
                let nonce = (batch_start + i) as u64;
                let vm_clone = vm_clone.clone();

                let handle = tokio::spawn(async move {
                    // Create a smart contract transaction
                    let tx = SmartContractTx {
                        address: 1000, // Example contract address
                        function: "transfer".to_string(),
                        arguments: vec![1, 2, 3, 4], // Example arguments
                        sender: 101,
                        gas_limit: 100000,
                        gas_price: 1,
                        nonce,
                        value: 0,
                        signature: [0; 64], // Example signature
                    };

                    // Submit transaction
                    match vm_clone.submit_transaction(tx).await {
                        Ok(_) => true,
                        Err(_) => false,
                    }
                });

                handles.push(handle);
            }

            // Wait for all submissions to complete
            for handle in handles {
                if let Ok(success) = handle.await {
                    if success {
                        submitted += 1;
                    }
                }
            }

            // Update counter
            let mut counter = tc_clone.lock().await;
            *counter += submitted;

            batch_num += 1;

            // Adaptive backpressure - slow down if transactions are being generated too quickly
            if submitted < batch_size / 2 {
                // If we couldn't submit even half the batch, add more delay
                tokio::time::sleep(Duration::from_millis(100)).await;
            } else {
                // Regular delay between batches
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    });

    // Wait for the duration
    tokio::time::sleep(Duration::from_secs(duration_secs)).await;

    // Stop transaction generation
    {
        let mut stop = stop_flag.lock().await;
        *stop = true;
    }

    // Wait for generator to finish
    let _ = generator_handle.await;

    // Wait for metrics to finish
    let _ = metrics_handle.await;

    // Final statistics
    let total_transactions = *transaction_counter.lock().await;
    let tps = total_transactions as f64 / duration_secs as f64;

    println!("\nStress Benchmark Results:");
    println!("  Total duration: {} seconds", duration_secs);
    println!("  Total transactions: {}", total_transactions);
    println!("  Overall throughput: {:.2} TPS", tps);

    // Get VM's perspective on TPS
    let vm_tps = nb_vm.get_tps().await;
    println!("  VM reported TPS: {:.2}", vm_tps);

    // Stop VM
    nb_vm.stop().await?;

    Ok(())
}
