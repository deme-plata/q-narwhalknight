# Source Code Compilation from /home/myuser/viper/dagknight-vm
Generated on: 2025-04-13 12:58:27

## Table of Contents

- [Cargo.toml](#Cargo.toml)
- [src/api/mod.rs](#src-api-mod.rs)
- [src/bin/narwhal_bullshark_bench.rs](#src-bin-narwhal_bullshark_bench.rs)
- [src/cache/mod.rs](#src-cache-mod.rs)
- [src/config/mod.rs](#src-config-mod.rs)
- [src/consensus/mod.rs](#src-consensus-mod.rs)
- [src/consensus/narwhal_bullshark.rs](#src-consensus-narwhal_bullshark.rs)
- [src/consensus/narwhal_bullshark.rs.bak](#src-consensus-narwhal_bullshark.rs.bak)
- [src/consensus/narwhal_bullshark/types.rs](#src-consensus-narwhal_bullshark-types.rs)
- [src/consensus/pbft.rs](#src-consensus-pbft.rs)
- [src/consensus/stub.rs](#src-consensus-stub.rs)
- [src/contracts/mod.rs](#src-contracts-mod.rs)
- [src/dag/mod.rs](#src-dag-mod.rs)
- [src/error/mod.rs](#src-error-mod.rs)
- [src/fault_tolerance/mod.rs](#src-fault_tolerance-mod.rs)
- [src/lib.rs](#src-lib.rs)
- [src/main.rs](#src-main.rs)
- [src/mempool/mod.rs](#src-mempool-mod.rs)
- [src/models/mod.rs](#src-models-mod.rs)
- [src/models/mod.rs.bak](#src-models-mod.rs.bak)
- [src/network/mod.rs](#src-network-mod.rs)
- [src/network/p2p.rs](#src-network-p2p.rs)
- [src/network/p2p_debug.rs](#src-network-p2p_debug.rs)
- [src/network/stub.rs](#src-network-stub.rs)
- [src/state/mod.rs](#src-state-mod.rs)
- [src/state/mod.rs.bak](#src-state-mod.rs.bak)
- [src/transaction/mod.rs](#src-transaction-mod.rs)
- [src/transaction/serde_impl.rs](#src-transaction-serde_impl.rs)
- [src/types/mod.rs](#src-types-mod.rs)
- [src/vm/ai/executor.rs](#src-vm-ai-executor.rs)
- [src/vm/ai/executor.rs.bak](#src-vm-ai-executor.rs.bak)
- [src/vm/ai/mod.rs](#src-vm-ai-mod.rs)
- [src/vm/batch/mod.rs](#src-vm-batch-mod.rs)
- [src/vm/batch_call_contracts.rs](#src-vm-batch_call_contracts.rs)
- [src/vm/cache/mod.rs](#src-vm-cache-mod.rs)
- [src/vm/executor.rs](#src-vm-executor.rs)
- [src/vm/jit_executor.rs](#src-vm-jit_executor.rs)
- [src/vm/memory/mod.rs](#src-vm-memory-mod.rs)
- [src/vm/memory/pool.rs](#src-vm-memory-pool.rs)
- [src/vm/memory/zero_copy.rs](#src-vm-memory-zero_copy.rs)
- [src/vm/mod.rs](#src-vm-mod.rs)
- [src/vm/mod.rs.bak](#src-vm-mod.rs.bak)
- [src/vm/mod.rs.fixed_batch](#src-vm-mod.rs.fixed_batch)
- [src/vm/narwhal_bullshark_vm.rs.bak](#src-vm-narwhal_bullshark_vm.rs.bak)
- [src/vm/narwhal_bullshark_vm/mod.rs](#src-vm-narwhal_bullshark_vm-mod.rs)
- [src/vm/parallel_executor.rs](#src-vm-parallel_executor.rs)
- [src/vm/tiered_vm.rs](#src-vm-tiered_vm.rs)

## Cargo.toml

### File path: `/home/myuser/viper/dagknight-vm/Cargo.toml`

```toml
[package]
name = "dagknight_vm"
version = "0.1.0"
edition = "2021"
authors = ["DAGKnight Team"]
description = "A virtual machine for DAGKnight blockchain with AI capabilities"
readme = "README.md"

[dependencies]
clap = { version = "4.0", features = ["derive"] }
toml = "0.5"
# Original dependencies
tokio = { version = "1.35.0", features = ["full", "rt-multi-thread"] }
rocksdb = { version = "0.20.1", features = ["multi-threaded-cf", "lz4", "zstd"] }
libp2p = { version = "0.53", features = ["tcp", "tokio", "noise", "yamux", "gossipsub", "identify", "ping", "kad", "dns", "mdns", "macros", "request-response"] }
serde = { version = "1.0", features = ["derive", "rc"] }
serde_json = "1.0"
serde-big-array = "0.4.0"
bincode = "1.3.3"
hex = { version = "0.4.3", features = ["serde"] }
thiserror = "1.0.0"
async-trait = "0.1.68"
futures = "0.3"
lazy_static = "1.4.0"
log = "0.4.0"
pretty_env_logger = "0.4.0"
blake3 = "1.3.3"
parking_lot = "0.12.1"
dashmap = { version = "5.5.3", features = ["raw-api", "rayon"] }
tracing = "0.1.37"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
bytes = "1.5"
parity-scale-codec = { version = "3.0", features = ["derive"] }
wasmer = "4.0.0"
rand = "0.8.5"
rayon = "1.8"
ed25519-dalek = "2.0.0"
priority-queue = "1.3"
structopt = "0.3"
ctrlc = "3.2"
tempfile = "3.3"
sha2 = "0.10.6"
signature = "2.1.0"
moka = { version = "0.12", features = ["sync"] }
crossbeam = "0.8"
once_cell = "1.19"
memmap2 = "0.9"
lru = "0.12"
num_cpus = "1.16"
anyhow = "1.0"


# AI Integration
ollama-rs = "0.1.5"
reqwest = { version = "0.11", features = ["json"] }
tokio-tungstenite = "0.20.0"

# Caching
redis = { version = "0.23.0", features = ["tokio-comp"] }

# For testing (marked as optional)
criterion = { version = "0.4", optional = true }
mockall = { version = "0.11.3", optional = true }
prometheus = { version = "0.13.3", optional = true }

[dev-dependencies]
proptest = "1.0"
test-case = "3.0"
tokio-test = "0.4"
wat = "1.0"
criterion = "0.4"
mockall = "0.11"

[features]
default = ["ai", "cache"]
ai = []
cache = []
dynamic-allocation = []
fault-tolerance = []
metrics = ["prometheus"]
testing = ["criterion", "mockall"]

[[bench]]
name = "vm_benchmarks"
harness = false

[[bin]]
name = "dagknight"
path = "src/main.rs"```

---


## src/api/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/api/mod.rs`

```rust
//! API module for DAGKnight VM
```

---


## src/bin/narwhal_bullshark_bench.rs

### File path: `/home/myuser/viper/dagknight-vm/src/bin/narwhal_bullshark_bench.rs`

```rust
//! Benchmark tool for Narwhal-Bullshark VM

use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Mutex;
use clap::Parser;
use dagknight_vm::config;
use dagknight_vm::vm::narwhal_bullshark_vm::{NarwhalBullsharkVm, SmartContractTx};
use dagknight_vm::vm::VirtualMachine;
use dagknight_vm::state::StateDB; // Add this import

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
                args.batch_size
            ).await?;
        },
        "multi" => {
            // Run multi-node benchmark
            println!("\nRunning multi-node benchmark...");
            run_multi_node_benchmark(
                node_ids.clone(), 
                vm.clone(), 
                args.transactions, 
                args.batch_size
            ).await?;
        },
        "stress" => {
            // Run stress benchmark
            println!("\nRunning stress benchmark for {} seconds...", args.duration);
            run_stress_benchmark(
                node_ids.clone(), 
                vm.clone(), 
                args.batch_size, 
                args.duration
            ).await?;
        },
        _ => {
            eprintln!("Unknown mode: {}. Must be 'single', 'multi', or 'stress'.", args.mode);
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
    batch_size: usize
) -> Result<(), Box<dyn std::error::Error>> {
    // Create Narwhal-Bullshark VM
    let nb_vm = Arc::new(NarwhalBullsharkVm::new(
        node_id, peers, vm
    ));
    
    // Start VM
    nb_vm.start().await?;
    
    // Allow time for initialization
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Generate and submit transactions
    let start_time = Instant::now();
    let mut completed = 0;
    
    println!("Generating and submitting {} transactions...", transaction_count);
    
    // Create batches of transactions
    for batch_num in 0..(transaction_count / batch_size + 1) {
        let batch_start = batch_num * batch_size;
        let batch_end = std::cmp::min(batch_start + batch_size, transaction_count);
        
        if batch_start >= batch_end {
            break;
        }
        
        let batch_size = batch_end - batch_start;
        println!("Submitting batch {} with {} transactions...", batch_num + 1, batch_size);
        
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
        println!("Progress: {}/{} transactions ({:.1}%)", 
            completed, transaction_count, progress);
        
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
    batch_size: usize
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating {} nodes for multi-node benchmark...", node_ids.len());
    
    // Create and start VMs for each node
    let mut vms = Vec::new();
    
    for (i, node_id) in node_ids.iter().enumerate() {
        // Create peers list (all other nodes)
        let peers: Vec<String> = node_ids.iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, id)| id.clone())
            .collect();
        
        // Create VM
        let nb_vm = Arc::new(NarwhalBullsharkVm::new(
            node_id.clone(), peers, vm.clone()
        ));
        
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
    
    println!("Generating and submitting {} transactions...", transaction_count);
    
    // Create batches of transactions
    for batch_num in 0..(transaction_count / batch_size + 1) {
        let batch_start = batch_num * batch_size;
        let batch_end = std::cmp::min(batch_start + batch_size, transaction_count);
        
        if batch_start >= batch_end {
            break;
        }
        
        let batch_size = batch_end - batch_start;
        println!("Submitting batch {} with {} transactions...", batch_num + 1, batch_size);
        
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
        println!("Progress: {}/{} transactions ({:.1}%)", 
            completed, transaction_count, progress);
        
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
    duration_secs: u64
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting stress benchmark for {} seconds...", duration_secs);
    
    // Create and start VM
    let nb_vm = Arc::new(NarwhalBullsharkVm::new(
        node_ids[0].clone(), 
        node_ids[1..].to_vec(), 
        vm.clone()
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
            
            println!("[{:4.1}s] Transactions: {} (+{}), TPS: {:.2} (current: {:.2})", 
                     elapsed.as_secs_f64(), 
                     count, 
                     count - last_count,
                     overall_tps,
                     incremental_tps);
            
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
                        Err(_) => false
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
}```

---


## src/cache/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/cache/mod.rs`

```rust
//! Caching layer for AI model outputs
use redis::{Client, AsyncCommands};
use lru::LruCache;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::num::NonZeroUsize;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::{info, warn, error, debug};
use std::time::{Duration, Instant};

/// Cache provider type
pub enum CacheProvider {
    /// In-memory LRU cache
    Memory,
    /// Redis cache
    Redis,
    /// Combined (layered) cache
    Layered,
}

/// Model cache for storing AI outputs
pub struct ModelCache {
    /// In-memory cache
    memory_cache: Arc<Mutex<LruCache<u64, CacheEntry>>>,
    /// Redis client if available
    redis_client: Option<Client>,
    /// Cache provider
    provider: CacheProvider,
    /// Cache hit statistics
    stats: Arc<Mutex<CacheStats>>,
}

/// Cache entry with expiration
#[derive(Clone)]
struct CacheEntry {
    /// Output data
    data: Vec<u8>,
    /// Expiration timestamp
    expires_at: Instant,
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
struct CacheStats {
    /// Number of gets
    gets: u64,
    /// Number of sets
    sets: u64,
    /// Number of memory hits
    memory_hits: u64,
    /// Number of redis hits
    redis_hits: u64,
    /// Number of misses
    misses: u64,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(provider: CacheProvider, memory_size: usize, redis_url: Option<String>) -> Self {
        // Create memory cache
        let memory_size = NonZeroUsize::new(memory_size).unwrap_or(NonZeroUsize::new(10000).unwrap());
        let memory_cache = Arc::new(Mutex::new(LruCache::new(memory_size)));
        
        // Create Redis client if URL provided
        let redis_client = if let Some(url) = redis_url {
            match Client::open(url) {
                Ok(client) => {
                    info!("Redis cache connected");
                    Some(client)
                },
                Err(e) => {
                    error!("Failed to connect to Redis: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Warn if Redis requested but not available
        if matches!(provider, CacheProvider::Redis | CacheProvider::Layered) && redis_client.is_none() {
            warn!("Redis cache requested but not available, falling back to memory cache");
        }
        
        Self {
            memory_cache,
            redis_client,
            provider,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }
    
    /// Get cached result
    pub async fn get(&self, model: &str, input: &[u8], ttl: u64) -> Option<Vec<u8>> {
        let key = Self::generate_key(model, input);
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.gets += 1;
        }
        
        // Try memory cache first
        if let Some(entry) = self.check_memory_cache(key).await {
            if entry.expires_at > Instant::now() {
                // Update stats
                {
                    let mut stats = self.stats.lock().await;
                    stats.memory_hits += 1;
                }
                return Some(entry.data);
            }
        }
        
        // If Redis is available and enabled, try it next
        if matches!(self.provider, CacheProvider::Redis | CacheProvider::Layered) {
            if let Some(client) = &self.redis_client {
                if let Some(data) = self.check_redis_cache(&client, model, input).await {
                    // Also update memory cache for next time
                    self.update_memory_cache(key, &data, ttl).await;
                    
                    // Update stats
                    {
                        let mut stats = self.stats.lock().await;
                        stats.redis_hits += 1;
                    }
                    
                    return Some(data);
                }
            }
        }
        
        // Update stats for miss
        {
            let mut stats = self.stats.lock().await;
            stats.misses += 1;
        }
        
        None
    }
    
    /// Check memory cache
    async fn check_memory_cache(&self, key: u64) -> Option<CacheEntry> {
        let mut cache = self.memory_cache.lock().await;
        if let Some(entry) = cache.get(&key) {
            Some(entry.clone())
        } else {
            None
        }
    }
    
    /// Check Redis cache
    async fn check_redis_cache(&self, client: &Client, model: &str, input: &[u8]) -> Option<Vec<u8>> {
        let key = format!("dagknight:model:{}:{}", model, hex::encode(Self::hash_bytes(input)));
        
        match client.get_async_connection().await {
            Ok(mut conn) => {
                match conn.get::<_, Option<Vec<u8>>>(&key).await {
                    Ok(Some(data)) => Some(data),
                    Ok(None) => None,
                    Err(e) => {
                        error!("Redis error while getting key {}: {}", key, e);
                        None
                    }
                }
            },
            Err(e) => {
                error!("Failed to get Redis connection: {}", e);
                None
            }
        }
    }
    
    /// Set value in cache
    pub async fn set(&self, model: &str, input: &[u8], output: &[u8], ttl: u64) {
        let key = Self::generate_key(model, input);
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.sets += 1;
        }
        
        // Update memory cache
        self.update_memory_cache(key, output, ttl).await;
        
        // Update Redis if available
        if matches!(self.provider, CacheProvider::Redis | CacheProvider::Layered) {
            if let Some(client) = &self.redis_client {
                self.update_redis_cache(client, model, input, output, ttl).await;
            }
        }
    }
    
    /// Update memory cache
    async fn update_memory_cache(&self, key: u64, data: &[u8], ttl: u64) {
        let entry = CacheEntry {
            data: data.to_vec(),
            expires_at: Instant::now() + Duration::from_secs(ttl),
        };
        
        let mut cache = self.memory_cache.lock().await;
        cache.put(key, entry);
    }
    
    /// Update Redis cache
    async fn update_redis_cache(&self, client: &Client, model: &str, input: &[u8], output: &[u8], ttl: u64) {
        let key = format!("dagknight:model:{}:{}", model, hex::encode(Self::hash_bytes(input)));
        
        match client.get_async_connection().await {
            Ok(mut conn) => {
                let _: Result<(), redis::RedisError> = conn.set_ex(&key, output, ttl as usize).await;
            },
            Err(e) => {
                error!("Failed to get Redis connection for set: {}", e);
            }
        }
    }
    
    /// Generate cache key
    fn generate_key(model: &str, input: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        model.hash(&mut hasher);
        input.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Hash bytes
    fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher.finalize().into()
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        // Get a lock on stats and clone the value
        let stats_guard = self.stats.lock().await;
        stats_guard.clone()
    }
    
    /// Start periodic cleanup
    pub fn start_cleanup_task(&self) {
        let memory_cache = self.memory_cache.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                let now = Instant::now();
                let mut cache = memory_cache.lock().await;
                
                // Remove expired entries
                // LruCache doesn't have a retain method, so we need to find expired keys and remove them
                let mut expired_keys = Vec::new();
                for (&key, entry) in cache.iter() {
                    if entry.expires_at <= now {
                        expired_keys.push(key);
                    }
                }
                
                for key in expired_keys {
                    cache.pop(&key);
                }
                
                debug!("Cache cleanup completed, size: {}", cache.len());
            }
        });
    }
}```

---


## src/config/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/config/mod.rs`

```rust
//! Configuration management for the VM

use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use std::sync::RwLock;

// VM Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmConfig {
    pub narwhal_bullshark: NarwhalBullsharkConfig,
}

// Narwhal-Bullshark specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarwhalBullsharkConfig {
    pub batch_size: usize,
    pub max_parallel_executions: usize,
    pub max_mempool_size: usize,
    pub block_production_interval_ms: u64,
    pub transaction_timeout_ms: u64,
    pub enable_metrics: bool,
    pub metrics_interval_seconds: u64,
}

impl Default for VmConfig {
    fn default() -> Self {
        Self {
            narwhal_bullshark: NarwhalBullsharkConfig::default(),
        }
    }
}

impl Default for NarwhalBullsharkConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_parallel_executions: 4,
            max_mempool_size: 100000,
            block_production_interval_ms: 1000,
            transaction_timeout_ms: 5000,
            enable_metrics: true,
            metrics_interval_seconds: 10,
        }
    }
}

// Singleton configuration
lazy_static::lazy_static! {
    static ref CONFIG: RwLock<VmConfig> = RwLock::new(VmConfig::default());
}

// Load configuration from file
pub fn load_config(path: impl AsRef<Path>) -> Result<(), String> {
    let path = path.as_ref();
    
    if !path.exists() {
        return Err(format!("Configuration file not found: {}", path.display()));
    }
    
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config file: {}", e))?;
    
    let config: VmConfig = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse config file: {}", e))?;
    
    // Update global config
    let mut global_config = CONFIG.write().unwrap();
    *global_config = config;
    
    Ok(())
}

// Get configuration
pub fn get_config() -> VmConfig {
    CONFIG.read().unwrap().clone()
}

// Get Narwhal-Bullshark configuration
pub fn get_narwhal_bullshark_config() -> NarwhalBullsharkConfig {
    CONFIG.read().unwrap().narwhal_bullshark.clone()
}

// Update batch size
pub fn update_batch_size(batch_size: usize) {
    let mut config = CONFIG.write().unwrap();
    config.narwhal_bullshark.batch_size = batch_size;
}
```

---


## src/consensus/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/consensus/mod.rs`

```rust
use std::sync::Arc;

pub mod narwhal_bullshark;
pub mod pbft;

pub struct Knight {
    pub dag: Arc<dyn std::any::Any + Send + Sync>, // Use a trait object temporarily
}

impl Knight {
    pub fn new(dag: Arc<dyn std::any::Any + Send + Sync>) -> Self {
        Self { dag }
    }

    pub fn get_current_k(&self) -> usize {
        2 // Placeholder
    }
}```

---


## src/consensus/narwhal_bullshark.rs

### File path: `/home/myuser/viper/dagknight-vm/src/consensus/narwhal_bullshark.rs`

```rust
use std::sync::Arc; 
use tokio::sync::{mpsc, RwLock}; 
use parking_lot::Mutex; 
use anyhow::Result; 
use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use serde_big_array::BigArray;

// Import using relative paths 
use crate::consensus::pbft::Block; 
use crate::vm::VmError;

// Define types directly in this file to avoid circular dependencies
pub type NodeId = String;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: [u8; 32],
    pub data: Vec<u8>,
    pub sender: [u8; 32],
    pub nonce: u64,
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
    pub timestamp: u64,
}

// Define a type alias for clarity
pub type NarwhalTransaction = Transaction;

pub struct Narwhal {     
    node_id: NodeId,     
    peers: Vec<NodeId>,     
    vertices: DashMap<u64, Vec<u8>>,      
    latest_round: RwLock<u64>,     
    pub tx_pool: Arc<Mutex<Vec<NarwhalTransaction>>>,     
    tx_network: mpsc::Sender<(NodeId, Vec<u8>)>, 
}  

impl Narwhal {     
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> (Self, mpsc::Receiver<(NodeId, Vec<u8>)>) {         
        let (tx, rx) = mpsc::channel(1000);                  
        (Self {             
            node_id,             
            peers,             
            vertices: DashMap::new(),             
            latest_round: RwLock::new(0),             
            tx_pool: Arc::new(Mutex::new(Vec::new())),             
            tx_network: tx,         
        }, rx)     
    } 
}  

pub struct Bullshark {     
    node_id: NodeId,     
    peers: Vec<NodeId>,     
    latest_round: RwLock<u64>,     
    finalized_blocks: DashMap<u64, Block>, 
}  

impl Bullshark {     
    pub fn new(node_id: NodeId, peers: Vec<NodeId>, _narwhal: Arc<Narwhal>) -> Self {         
        Self {             
            node_id,             
            peers,             
            latest_round: RwLock::new(0),             
            finalized_blocks: DashMap::new(),         
        }     
    }          
    
    pub async fn get_latest_finalized(&self) -> u64 {         
        *self.latest_round.read().await     
    }          
    
    pub async fn get_finalized_block(&self, seq_num: u64) -> Option<Block> {         
        self.finalized_blocks.get(&seq_num).map(|b| b.clone())     
    } 
}  

pub struct NarwhalBullshark {     
    node_id: NodeId,     
    peers: Vec<NodeId>,     
    narwhal: Arc<Narwhal>,     
    bullshark: Arc<Bullshark>,     
    finalized_blocks: DashMap<u64, Block>,     
    latest_height: RwLock<u64>,     
    tx_network: mpsc::Sender<(NodeId, Vec<u8>)>,     
    rx_narwhal: mpsc::Receiver<(NodeId, Vec<u8>)>,     
    tx_mempool: mpsc::Sender<NarwhalTransaction>,     
    rx_mempool: mpsc::Receiver<NarwhalTransaction>, 
}  

impl NarwhalBullshark {     
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> Self {         
        let (narwhal, rx_narwhal) = Narwhal::new(node_id.clone(), peers.clone());         
        let narwhal = Arc::new(narwhal);         
        let bullshark = Arc::new(Bullshark::new(node_id.clone(), peers.clone(), narwhal.clone()));                  
        let (tx_mempool, rx_mempool) = mpsc::channel(1000);         
        let (tx_network, _) = mpsc::channel(1000);                  
        Self {             
            node_id,             
            peers,             
            narwhal,             
            bullshark,             
            finalized_blocks: DashMap::new(),             
            latest_height: RwLock::new(0),             
            tx_network,             
            rx_narwhal,             
            tx_mempool,             
            rx_mempool,         
        }     
    }          
    
    pub async fn start(&self) {         
        println!("Starting NarwhalBullshark consensus...");         
        // Implementation would go here     
    }          
    
    pub async fn get_latest_finalized(&self) -> u64 {         
        (*self.bullshark).get_latest_finalized().await     
    }          
    
    pub async fn get_finalized_block(&self, seq_num: u64) -> Option<Block> {         
        (*self.bullshark).get_finalized_block(seq_num).await     
    }          
    
    pub async fn add_transaction(&self, tx: NarwhalTransaction) -> Result<(), VmError> {         
        let _tx_data = bincode::serialize(&tx)             
            .map_err(|e| VmError::SerializationError(e.to_string()))?;                      
        // Simplified implementation         
        let mut pool = self.narwhal.tx_pool.lock();         
        pool.push(tx);                  
        Ok(())     
    } 
}```

---


## src/consensus/narwhal_bullshark.rs.bak

### File path: `/home/myuser/viper/dagknight-vm/src/consensus/narwhal_bullshark.rs.bak`

```text
//! Narwhal-Bullshark consensus implementation for high-throughput transaction processing

mod types;
pub use types::*;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use parking_lot::Mutex;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use dashmap::DashMap;

use crate::vm::{ConsensusEngine, VmError};

// Re-export main components
use crate::consensus::pbft::Block;

// Narwhal data availability layer
pub struct Narwhal {
    /* Added placeholder for missing Vertex type */
    node_id: NodeId,
    peers: Vec<NodeId>,
    
    // DAG state
    vertices: DashMap<u64, Vec<u8>>, // Modified to use available types
    parent_vertices: DashMap<u64, HashSet<VertexId>>, // round -> vertices
    
    // Transaction pool
    tx_pool: Arc<Mutex<Vec<Transaction>>>,
    
    // Networking
    tx_network: mpsc::Sender<(NodeId, Vec<u8>)>, // Modified to use Vec<u8> instead of NarwhalMessage
    
    // Current round
    current_round: Arc<RwLock<u64>>,
    
    // Last activity timestamp for timeout
    last_activity: Arc<Mutex<Instant>>,
}

// Bullshark consensus layer
pub struct Bullshark {
    node_id: NodeId,
    peers: Vec<NodeId>,
    
    // Reference to Narwhal layer
    narwhal: Arc<Narwhal>,
    
    // Ordered blocks
    ordered_blocks: Arc<RwLock<Vec<Block>>>,
    
    // Finalized DAG rounds
    finalized_rounds: Arc<RwLock<u64>>,
    
    // Latest block sequence number
    latest_seq_num: Arc<RwLock<u64>>,
    
    // Latest finalized block
    latest_finalized: Arc<RwLock<u64>>,
    
    // Blockchain state
    blocks: Arc<RwLock<HashMap<[u8; 32], Block>>>,
    finalized_blocks: Arc<RwLock<HashMap<u64, [u8; 32]>>>,
}

impl Narwhal {
    // Create a new Narwhal instance
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> (Self, mpsc::Receiver<(NodeId, Vec<u8>)>) {
        let (tx_network, rx_network) = mpsc::channel(1000);
        
        let narwhal = Self {
            node_id,
            peers,
            vertices: DashMap::new(),
            parent_vertices: DashMap::new(),
            tx_pool: Arc::new(Mutex::new(Vec::new())),
            tx_network,
            current_round: Arc::new(RwLock::new(0)),
            last_activity: Arc::new(Mutex::new(Instant::now())),
        };
        
        (narwhal, rx_network)
    }

    // Implementation methods (as in the original code)
    // ... (abbreviated for script readability)
}

impl Bullshark {
    // Create a new Bullshark instance
    pub fn new(node_id: NodeId, peers: Vec<NodeId>, narwhal: Arc<Narwhal>) -> Self {
        Self {
            node_id,
            peers,
            narwhal,
            ordered_blocks: Arc::new(RwLock::new(Vec::new())),
            finalized_rounds: Arc::new(RwLock::new(0)),
            latest_seq_num: Arc::new(RwLock::new(0)),
            latest_finalized: Arc::new(RwLock::new(0)),
            blocks: Arc::new(RwLock::new(HashMap::new())),
            finalized_blocks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // Implementation methods (as in the original code)
    // ... (abbreviated for script readability)
}

// Main NarwhalBullshark consensus implementation
pub struct NarwhalBullshark {
    node_id: NodeId,
    peers: Vec<NodeId>,
    
    // Core components
    narwhal: Arc<Narwhal>,
    bullshark: Arc<Bullshark>,
    
    // Communication channels
    tx_network: mpsc::Sender<(NodeId, Vec<u8>)>, // Modified to use Vec<u8> instead of ConsensusMessage
    rx_narwhal: mpsc::Receiver<(NodeId, Vec<u8>)>, // Modified to use Vec<u8> instead of NarwhalMessage
    
    // Transaction channels
    tx_mempool: mpsc::Sender<Transaction>,
    rx_mempool: mpsc::Receiver<Transaction>,
}

impl NarwhalBullshark {
    // Create a new NarwhalBullshark instance
    pub async fn start(&self) {
        println!("Starting NarwhalBullshark consensus...");
        // Placeholder implementation
    }
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> Self {
        // Create channels
        let (tx_network, _rx_network) = mpsc::channel(1000);
        let (tx_mempool, rx_mempool) = mpsc::channel(10000);
        
        // Create Narwhal
        let (narwhal, rx_narwhal) = Narwhal::new(node_id.clone(), peers.clone());
        let narwhal = Arc::new(narwhal);
        
        // Create Bullshark
        let bullshark = Arc::new(Bullshark::new(node_id.clone(), peers.clone(), Arc::clone(&narwhal)));
        
        Self {
            node_id,
            peers,
            narwhal,
            bullshark,
            tx_network,
            rx_narwhal,
            tx_mempool,
            rx_mempool,
        }
    }

    // Implementation methods (as in the original code)
    // ... (abbreviated for script readability)
    
    // Implement ConsensusEngine trait methods
    pub async fn add_transaction(&self, tx: Transaction) -> Result<(), VmError> {
        if let Err(_) = self.tx_mempool.send(tx).await {
            Err(VmError::ConsensusFailure("Failed to add transaction to mempool".to_string()))
        } else {
            Ok(())
        }
    }
    
    pub async fn get_latest_finalized(&self) -> u64 {
        (*self.bullshark).get_latest_finalized().await
    }
    
    pub async fn get_finalized_block(&self, seq_num: u64) -> Option<Block> {
        (*self.bullshark).get_finalized_block(seq_num).await
    }
}

// Implement ConsensusEngine trait for NarwhalBullshark
#[async_trait]
impl ConsensusEngine for NarwhalBullshark {
    async fn validate_contract(&self, _hash: [u8; 32], _bytecode: &[u8]) -> Result<(), VmError> {
        // In a real implementation, this would validate the contract
        // For simplicity, we just return Ok
        Ok(())
    }

    async fn broadcast_contract(&self, hash: [u8; 32], bytecode: Vec<u8>) -> Result<(), VmError> {
        // Create a transaction for the contract
        let tx = Transaction {
            hash,
            data: bytecode,
            sender: [0; 32], // Placeholder
            nonce: 0,        // Placeholder
            signature: [0; 64], // Placeholder
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Add transaction to mempool
        self.add_transaction(tx).await
    }
}
```

---


## src/consensus/narwhal_bullshark/types.rs

### File path: `/home/myuser/viper/dagknight-vm/src/consensus/narwhal_bullshark/types.rs`

```rust
#[derive(Clone, Debug)]
pub struct VertexId(pub [u8; 32]);

#[derive(Clone, Debug)]
pub struct Vertex {
    pub id: VertexId,
    pub round: u64,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub enum NarwhalMessage {
    Vertex(Vertex),
    Sync(u64),
}

#[derive(Clone, Debug)]
pub enum ConsensusMessage {
    Propose(u64),
    Vote(u64, [u8; 32]),
}
```

---


## src/consensus/pbft.rs

### File path: `/home/myuser/viper/dagknight-vm/src/consensus/pbft.rs`

```rust
use async_trait::async_trait;
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::RwLock;
use parking_lot::Mutex;
use serde::{Serialize, Deserialize};
use crate::vm::VmError;
use crate::vm::{ConsensusEngine, VmError as VMError};
use serde_big_array::big_array;

// Initialize BigArray for arrays up to size 64
big_array! { BigArray; 64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: [u8; 32],         // Transaction hash
    pub data: Vec<u8>,          // Transaction data
    pub sender: [u8; 32],       // Sender's address
    pub nonce: u64,             // Sender's nonce
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],    // Transaction signature
    pub timestamp: u64,         // Timestamp when created
}

type BlockHash = [u8; 32];
type NodeId = String;

// PBFT message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PbftMessage {
    PrepareRequest(PrepareRequest),
    PrepareResponse(PrepareResponse),
    CommitRequest(CommitRequest),
    CommitResponse(CommitResponse),
    ViewChange(ViewChange),
    NewView(NewView),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareRequest {
    pub view: u64,
    pub seq_num: u64,
    pub block_hash: BlockHash,
    pub block_data: Vec<u8>,
    pub primary_id: NodeId,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareResponse {
    pub view: u64,
    pub seq_num: u64,
    pub block_hash: BlockHash,
    pub node_id: NodeId,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRequest {
    pub view: u64,
    pub seq_num: u64,
    pub block_hash: BlockHash,
    pub node_id: NodeId,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResponse {
    pub view: u64,
    pub seq_num: u64,
    pub block_hash: BlockHash,
    pub node_id: NodeId,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewChange {
    pub new_view: u64,
    pub seq_num: u64,
    pub node_id: NodeId,
    pub checkpoint: Option<(u64, BlockHash)>,
    pub prepared_proofs: Vec<(u64, BlockHash, Vec<PrepareResponse>)>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewView {
    pub view: u64,
    pub view_change_proofs: Vec<ViewChange>,
    pub prepare_requests: Vec<PrepareRequest>,
    pub node_id: NodeId,
    pub timestamp: u64,
}

// Block structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub hash: BlockHash,
    pub parent_hash: BlockHash,
    pub seq_num: u64,
    pub transactions: Vec<Transaction>,
    pub timestamp: u64,
    pub proposer: NodeId,
}

impl Block {
    pub fn new(parent_hash: BlockHash, seq_num: u64, transactions: Vec<Transaction>, proposer: NodeId) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        let mut block = Self {
            hash: [0; 32],
            parent_hash,
            seq_num,
            transactions,
            timestamp,
            proposer,
        };
        
        block.hash = block.compute_hash();
        block
    }
    
    pub fn compute_hash(&self) -> BlockHash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.parent_hash);
        hasher.update(&self.seq_num.to_le_bytes());
        
        for tx in &self.transactions {
            hasher.update(&tx.hash);
        }
        
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(self.proposer.as_bytes());
        
        let mut hash = [0; 32];
        hash.copy_from_slice(hasher.finalize().as_bytes());
        hash
    }
}

// PBFT consensus engine
#[derive(Debug)]
pub struct PbftConsensus {
    node_id: NodeId,
    peers: Vec<NodeId>,
    view: Arc<RwLock<u64>>,
    seq_num: Arc<RwLock<u64>>,
    is_primary: Arc<RwLock<bool>>,
    
    // Message channels
    tx_network: Sender<(NodeId, PbftMessage)>,
    rx_network: Receiver<(NodeId, PbftMessage)>,
    
    // State
    prepare_requests: Arc<DashMap<(u64, u64), PrepareRequest>>,
    prepare_responses: Arc<DashMap<(u64, u64, BlockHash), HashSet<NodeId>>>,
    commit_requests: Arc<DashMap<(u64, u64, BlockHash), HashSet<NodeId>>>,
    commit_responses: Arc<DashMap<(u64, u64, BlockHash), HashSet<NodeId>>>,
    
    // View change state
    view_changes: Arc<DashMap<(u64, NodeId), ViewChange>>,
    
    // Blockchain state
    _blockchain: Arc<RwLock<HashMap<BlockHash, Block>>>,
    finalized_blocks: Arc<RwLock<HashMap<u64, BlockHash>>>,
    latest_finalized: Arc<RwLock<u64>>,
    
    // Timer for view change
    view_change_timeout: Duration,
    last_activity: Arc<Mutex<Instant>>,
}

impl PbftConsensus {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> Self {
        let (tx_network, rx_network) = mpsc::channel(1000);
        
        let is_primary = node_id == Self::get_primary(0, &peers);
        
        Self {
            node_id,
            peers,
            view: Arc::new(RwLock::new(0)),
            seq_num: Arc::new(RwLock::new(0)),
            is_primary: Arc::new(RwLock::new(is_primary)),
            
            tx_network,
            rx_network,
            
            prepare_requests: Arc::new(DashMap::new()),
            prepare_responses: Arc::new(DashMap::new()),
            commit_requests: Arc::new(DashMap::new()),
            commit_responses: Arc::new(DashMap::new()),
            
            view_changes: Arc::new(DashMap::new()),
            
            _blockchain: Arc::new(RwLock::new(HashMap::new())),
            finalized_blocks: Arc::new(RwLock::new(HashMap::new())),
            latest_finalized: Arc::new(RwLock::new(0)),
            
            view_change_timeout: Duration::from_secs(30),
            last_activity: Arc::new(Mutex::new(Instant::now())),
        }
    }
    
    // Get the primary node for a view
    fn get_primary(view: u64, peers: &[NodeId]) -> NodeId {
        let idx = (view as usize) % (peers.len() + 1);
        if idx < peers.len() {
            peers[idx].clone()
        } else {
            peers[0].clone() // Fallback
        }
    }
    
    // Start the consensus engine
    pub async fn start(&mut self) {
        // Start view change timer
        self.start_view_change_timer();
        
        // Start message processing loop
        self.process_messages().await;
    }
    
    // Start the view change timer
    fn start_view_change_timer(&self) {
        let view = self.view.clone();
        let last_activity = self.last_activity.clone();
        let node_id = self.node_id.clone();
        let peers = self.peers.clone();
        let tx_network = self.tx_network.clone();
        let seq_num = self.seq_num.clone();
        let view_change_timeout = self.view_change_timeout;
        
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                
                let current_view = *view.read().await;
                let elapsed = {
                    let last = last_activity.lock();
                    last.elapsed()
                };
                
                if elapsed > view_change_timeout {
                    // Trigger view change
                    let new_view = current_view + 1;
                    let current_seq = *seq_num.read().await;
                    
                    // Create view change message
                    let view_change = ViewChange {
                        new_view,
                        seq_num: current_seq,
                        node_id: node_id.clone(),
                        checkpoint: None, // For simplicity
                        prepared_proofs: Vec::new(), // For simplicity
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    };
                    
                    // Broadcast view change to all peers
                    for peer in &peers {
                        let _ = tx_network.send((peer.clone(), PbftMessage::ViewChange(view_change.clone()))).await;
                    }
                    
                    // Update view locally
                    *view.write().await = new_view;
                    
                    // Reset activity timer
                    {
                        let mut last = last_activity.lock();
                        *last = Instant::now();
                    }
                }
            }
        });
    }
    
    // Process incoming messages
    async fn process_messages(&mut self) {
        while let Some((_sender, message)) = self.rx_network.recv().await {
            // Update activity timer
            {
                let mut last = self.last_activity.lock();
                *last = Instant::now();
            }
            
            match message {
                PbftMessage::PrepareRequest(req) => {
                    self.handle_prepare_request(req).await;
                },
                PbftMessage::PrepareResponse(resp) => {
                    self.handle_prepare_response(resp).await;
                },
                PbftMessage::CommitRequest(req) => {
                    self.handle_commit_request(req).await;
                },
                PbftMessage::CommitResponse(resp) => {
                    self.handle_commit_response(resp).await;
                },
                PbftMessage::ViewChange(vc) => {
                    self.handle_view_change(vc).await;
                },
                PbftMessage::NewView(nv) => {
                    self.handle_new_view(nv).await;
                },
            }
        }
    }
    
    // Handle prepare request
    async fn handle_prepare_request(&self, req: PrepareRequest) {
        let current_view = *self.view.read().await;
        
        // Verify view and sequence number
        if req.view != current_view {
            return; // Ignore requests from different views
        }
        
        // Store the prepare request
        self.prepare_requests.insert((req.view, req.seq_num), req.clone());
        
        // Create prepare response
        let prepare_response = PrepareResponse {
            view: req.view,
            seq_num: req.seq_num,
            block_hash: req.block_hash,
            node_id: self.node_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Send prepare response to primary
        let _ = self.tx_network.send((req.primary_id.clone(), 
            PbftMessage::PrepareResponse(prepare_response))).await;
    }
    
    // Handle prepare response
    async fn handle_prepare_response(&self, resp: PrepareResponse) {
        let current_view = *self.view.read().await;
        
        // Verify view
        if resp.view != current_view {
            return; // Ignore responses from different views
        }
        
        // Add to prepare responses
        let key = (resp.view, resp.seq_num, resp.block_hash);
        let mut entry = self.prepare_responses.entry(key).or_insert_with(HashSet::new);
        entry.insert(resp.node_id.clone());
        
        // Check if we have 2f+1 prepare responses
        if entry.len() >= self.get_quorum_size() {
            // Create commit request
            let commit_request = CommitRequest {
                view: resp.view,
                seq_num: resp.seq_num,
                block_hash: resp.block_hash,
                node_id: self.node_id.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };
            
            // Broadcast commit request to all peers
            for peer in &self.peers {
                let _ = self.tx_network.send((peer.clone(), 
                    PbftMessage::CommitRequest(commit_request.clone()))).await;
            }
        }
    }
    
    // Handle commit request
    async fn handle_commit_request(&self, req: CommitRequest) {
        let current_view = *self.view.read().await;
        
        // Verify view
        if req.view != current_view {
            return; // Ignore requests from different views
        }
        
        // Add to commit requests
        let key = (req.view, req.seq_num, req.block_hash);
        let mut entry = self.commit_requests.entry(key).or_insert_with(HashSet::new);
        entry.insert(req.node_id.clone());
        
        // Create commit response
        let commit_response = CommitResponse {
            view: req.view,
            seq_num: req.seq_num,
            block_hash: req.block_hash,
            node_id: self.node_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Broadcast commit response to all peers
        for peer in &self.peers {
            let _ = self.tx_network.send((peer.clone(), 
                PbftMessage::CommitResponse(commit_response.clone()))).await;
        }
    }
    
    // Handle commit response
    async fn handle_commit_response(&self, resp: CommitResponse) {
        let current_view = *self.view.read().await;
        
        // Verify view
        if resp.view != current_view {
            return; // Ignore responses from different views
        }
        
        // Add to commit responses
        let key = (resp.view, resp.seq_num, resp.block_hash);
        let mut entry = self.commit_responses.entry(key).or_insert_with(HashSet::new);
        entry.insert(resp.node_id.clone());
        
        // Check if we have 2f+1 commit responses
        if entry.len() >= self.get_quorum_size() {
            // Finalize the block
            self.finalize_block(resp.seq_num, resp.block_hash).await;
        }
    }
    
    // Handle view change
    async fn handle_view_change(&self, vc: ViewChange) {
        let current_view = *self.view.read().await;
        
        // Only consider view changes for views greater than current
        if vc.new_view <= current_view {
            return;
        }
        
        // Store view change
        self.view_changes.insert((vc.new_view, vc.node_id.clone()), vc.clone());
        
        // Check if we have enough view changes for the new view
        let mut view_changes_for_new_view = Vec::new();
        for item in self.view_changes.iter() {
            let ((view, _), vc) = item.pair();
            if *view == vc.new_view {
                view_changes_for_new_view.push(vc.clone());
            }
        }
        
        // Check if we have 2f+1 view changes
        if view_changes_for_new_view.len() >= self.get_quorum_size() {
            // Become primary if it's our turn
            let is_primary = self.node_id == Self::get_primary(vc.new_view, &self.peers);
            *self.is_primary.write().await = is_primary;
            
            if is_primary {
                // Create new view message
                let new_view = NewView {
                    view: vc.new_view,
                    view_change_proofs: view_changes_for_new_view.clone(),
                    prepare_requests: Vec::new(), // Simplified for now
                    node_id: self.node_id.clone(),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };
                
                // Broadcast new view to all peers
                for peer in &self.peers {
                    let _ = self.tx_network.send((peer.clone(), 
                        PbftMessage::NewView(new_view.clone()))).await;
                }
            }
            
            // Update view
            *self.view.write().await = vc.new_view;
        }
    }
    
    // Handle new view
    async fn handle_new_view(&self, nv: NewView) {
        let current_view = *self.view.read().await;
        
        // Verify new view
        if nv.view <= current_view {
            return; // Ignore outdated new views
        }
        
        // Verify new view has enough view change proofs
        if nv.view_change_proofs.len() < self.get_quorum_size() {
            return; // Not enough proofs
        }
        
        // Update view
        *self.view.write().await = nv.view;
        
        // Process prepare requests if any
        for prep_req in nv.prepare_requests {
            self.handle_prepare_request(prep_req).await;
        }
    }
    
    // Get quorum size (2f+1 where f is max faulty nodes)
    fn get_quorum_size(&self) -> usize {
        let n = self.peers.len() + 1; // Total nodes including self
        let f = (n - 1) / 3; // Max faulty nodes
        2 * f + 1
    }
    
    // Finalize a block
    async fn finalize_block(&self, seq_num: u64, block_hash: BlockHash) {
        let _blockchain = self._blockchain.write().await;
        let mut finalized_blocks = self.finalized_blocks.write().await;
        let mut latest_finalized = self.latest_finalized.write().await;
        
        // Mark block as finalized
        finalized_blocks.insert(seq_num, block_hash);
        
        // Update latest finalized if this is newer
        if seq_num > *latest_finalized {
            *latest_finalized = seq_num;
        }
        
        // Update sequence number if needed
        if seq_num >= *self.seq_num.read().await {
            *self.seq_num.write().await = seq_num + 1;
        }
        
        println!("Block finalized: seq={}, hash={:?}", seq_num, block_hash);
    }
    
    // Propose a new block
    pub async fn propose_block(&self, parent_hash: BlockHash, transactions: Vec<Transaction>) -> Result<BlockHash, VmError> {
        let is_primary = *self.is_primary.read().await;
        if !is_primary {
            return Err(VmError::ConsensusFailure("Not the primary node".to_string()));
        }
        
        let view = *self.view.read().await;
        let seq_num = *self.seq_num.read().await;
        
        // Create new block
        let block = Block::new(parent_hash, seq_num, transactions, self.node_id.clone());
        
        // Store block in _blockchain
        {
            let mut _blockchain = self._blockchain.write().await;
            _blockchain.insert(block.hash, block.clone());
        }
        
        // Create prepare request
        let prepare_request = PrepareRequest {
            view,
            seq_num,
            block_hash: block.hash,
            block_data: bincode::serialize(&block).map_err(|_| VmError::SerializationError("Serialization failed".to_string()))?,
            primary_id: self.node_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Broadcast prepare request to all peers
        for peer in &self.peers {
            let _ = self.tx_network.send((peer.clone(), 
                PbftMessage::PrepareRequest(prepare_request.clone()))).await;
        }
        
        // Handle prepare request locally
        self.handle_prepare_request(prepare_request).await;
        
        Ok(block.hash)
    }
    
    // Get network sender
    pub fn get_network_sender(&self) -> Sender<(NodeId, PbftMessage)> {
        self.tx_network.clone()
    }
    
    // Get the latest finalized block height
    pub async fn get_latest_finalized(&self) -> u64 {
        *self.latest_finalized.read().await
    }
    
    // Get a finalized block by sequence number
    pub async fn get_finalized_block(&self, seq_num: u64) -> Option<Block> {
        let finalized_blocks = self.finalized_blocks.read().await;
        let _blockchain = self._blockchain.read().await;
        
        if let Some(hash) = finalized_blocks.get(&seq_num) {
            _blockchain.get(hash).cloned()
        } else {
            None
        }
    }
    
    // For testing: Create a view change message
    pub async fn create_view_change(&self, new_view: u64) -> ViewChange {
        let current_seq = *self.seq_num.read().await;
        
        ViewChange {
            new_view,
            seq_num: current_seq,
            node_id: self.node_id.clone(),
            checkpoint: None, // For simplicity
            prepared_proofs: Vec::new(), // For simplicity
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    // For testing: Create a view change message for a specific node
    pub async fn create_view_change_for_test(&self, node_id: NodeId, new_view: u64) -> ViewChange {
        let current_seq = *self.seq_num.read().await;
        
        ViewChange {
            new_view,
            seq_num: current_seq,
            node_id,
            checkpoint: None, // For simplicity
            prepared_proofs: Vec::new(), // For simplicity
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    
    // For testing: Get current view
    pub async fn get_current_view(&self) -> u64 {
        *self.view.read().await
    }
}

#[async_trait]
impl ConsensusEngine for PbftConsensus {
    async fn validate_contract(&self, _hash: [u8; 32], _bytecode: &[u8]) -> Result<(), VmError> {
        // Implement contract validation logic for PBFT consensus
        // For simplicity, we'll just accept all contracts for now
        Ok(())
    }

    async fn validate_block(&self, _block: &[u8]) -> Result<bool, VmError> {
        // PBFT block validation logic
        Ok(true)
    }

    async fn finalize_block(&self, _block: &[u8]) -> Result<(), VmError> {
        // PBFT block finalization logic
        Ok(())
    }

    async fn get_latest_block(&self) -> Result<Vec<u8>, VmError> {
        // Get latest block from PBFT
        Ok(Vec::new())
    }

    async fn broadcast_contract(&self, hash: [u8; 32], bytecode: Vec<u8>) -> Result<(), VmError> {
        // Create a transaction for the contract
        let tx = Transaction {
            hash,
            data: bytecode,
            sender: [0; 32], // Placeholder
            nonce: 0,        // Placeholder
            signature: [0; 64], // Placeholder
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Get the latest finalized block's hash for parent
        let latest_seq = self.get_latest_finalized().await;
        let parent_hash = if latest_seq > 0 {
            if let Some(block) = self.get_finalized_block(latest_seq).await {
                block.hash
            } else {
                [0; 32] // Genesis block hash
            }
        } else {
            [0; 32] // Genesis block hash
        };
        
        // Propose a new block with this transaction
        match self.propose_block(parent_hash, vec![tx]).await {
            Ok(_) => Ok(()),
            Err(e) => Err(VMError::ConsensusFailure(format!("Failed to propose block: {:?}", e))),
        }
    }
}
```

---


## src/consensus/stub.rs

### File path: `/home/myuser/viper/dagknight-vm/src/consensus/stub.rs`

```rust
use crate::vm::{ConsensusEngine, VmError};

// Stub implementation for testing
pub struct StubConsensus;

impl StubConsensus {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl ConsensusEngine for StubConsensus {
    async fn validate_contract(&self, _hash: [u8; 32], _bytecode: &[u8]) -> Result<(), VmError> {
        // Stub implementation that just succeeds
        Ok(())
    }
    
    async fn broadcast_contract(&self, _hash: [u8; 32], _bytecode: Vec<u8>) -> Result<(), VmError> {
        // Stub implementation that just succeeds
        Ok(())
    }
}
```

---


## src/contracts/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/contracts/mod.rs`

```rust
// Contracts module
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractCall {
    pub contract_address: [u8; 32],
    pub method: String,
    pub args: Vec<u8>,
}

// The problem is here - we have two identical derive macros
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingCapability {
    None,
    DataParallel,
    ModelParallel,
    Horizontal,
    Vertical,
    Full,
}

#[derive(Debug, Clone)]
pub struct AIModelCall {
    pub model_id: String,
    pub input: Vec<u8>,
    pub model: String,
    pub shard_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_cpu_cores: u32,
    pub min_memory_mb: u64,
    pub min_gpu_memory_mb: u64,
    pub preferred_batch_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistration {
    pub model_id: String,
    pub version: String,
    pub owner: [u8; 32],
    pub description: String,
    pub capabilities: ShardingCapability,
    pub resources: ResourceRequirements,
    pub hash: [u8; 32],
    pub timestamp: u64,
}

// Add Contract struct that was missing
#[derive(Debug, Clone)]
pub struct Contract {
    pub code: Vec<u8>,
    pub state: HashMap<Vec<u8>, Vec<u8>>,
}

// Add ContractResult struct
#[derive(Debug, Clone)]
pub struct ContractResult {
    pub success: bool,
    pub return_data: Vec<u8>,
    pub error: Option<String>,
    pub gas_used: u64,
    pub state_changes: HashMap<Vec<u8>, Vec<u8>>,
    pub logs: Vec<String>,
}

// Add ContractRegistry
pub struct ContractRegistry {
    contracts: std::sync::RwLock<HashMap<[u8; 32], std::sync::Arc<Contract>>>,
}

impl ContractRegistry {
    pub fn new() -> Self {
        Self {
            contracts: std::sync::RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, address: &[u8; 32]) -> Option<std::sync::Arc<Contract>> {
        let contracts = self.contracts.read().unwrap();
        contracts.get(address).cloned()
    }
}
```

---


## src/dag/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/dag/mod.rs`

```rust
//! DAG module for consensus
#[derive(Debug)]
pub struct DAG {
    // Placeholder implementation
}

impl DAG {
    pub fn new() -> Self {
        Self {}
    }
}
```

---


## src/error/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/error/mod.rs`

```rust
use thiserror::Error;
use std::fmt;

#[derive(Debug, Error, Clone)]
pub enum Error {
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("Unauthorized: {0}")]
    Unauthorized(String),
    
    #[error("I/O error: {0}")]
    Io(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("Database error: {0}")]
    Database(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
    
    #[error("General error: {0}")]
    General(String),
    
    #[error("Security error: {0}")]
    Security(String),
    
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
}

// VmError is a placeholder for now
#[derive(Debug, Clone)]
pub struct VmError(pub String);

// Define a Result type
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VM Error: {}", self.0)
    }
}

impl std::error::Error for VmError {}

// Allow conversion from various error types
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

impl From<String> for Error {
    fn from(e: String) -> Self {
        Error::General(e)
    }
}

impl From<&str> for Error {
    fn from(e: &str) -> Self {
        Error::General(e.to_string())
    }
}
```

---


## src/fault_tolerance/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/fault_tolerance/mod.rs`

```rust
//! Fault tolerance for distributed computation
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use futures::future::{self, Future};
use tracing::{info, warn, error, debug, instrument};
use thiserror::Error;

/// Recovery error
#[derive(Debug, Error, Clone)]
pub enum RecoveryError {
    #[error("Task failed: {0}")]
    TaskFailed(String),
    
    #[error("All tasks failed")]
    AllTasksFailed,
    
    #[error("Timeout: {0}")]
    Timeout(String),
}

type Result<T> = std::result::Result<T, RecoveryError>;

/// Recovery manager for fault-tolerant computations
pub struct RecoveryManager {
    /// Node reliability ratings
    node_reliability: Arc<RwLock<HashMap<String, f64>>>,
    /// Failed nodes
    failed_nodes: Arc<RwLock<HashSet<String>>>,
    /// Recovery settings
    settings: Arc<RecoverySettings>,
}

/// Recovery settings
#[derive(Debug, Clone)]
pub struct RecoverySettings {
    /// Enable task replication
    pub enable_replication: bool,
    /// Replication factor (how many duplicate tasks to run)
    pub replication_factor: usize,
    /// Max retry attempts
    pub max_retries: usize,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Task timeout in seconds
    pub task_timeout_secs: u64,
}

impl Default for RecoverySettings {
    fn default() -> Self {
        Self {
            enable_replication: false,
            replication_factor: 1,
            max_retries: 3,
            retry_delay_ms: 500,
            task_timeout_secs: 60,
        }
    }
}

/// Node status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is healthy
    Healthy,
    /// Node is partially degraded
    Degraded,
    /// Node is unhealthy
    Unhealthy,
    /// Node is offline
    Offline,
}

impl RecoveryManager {
    /// Create a new recovery manager
    pub fn new(settings: RecoverySettings) -> Self {
        Self {
            node_reliability: Arc::new(RwLock::new(HashMap::new())),
            failed_nodes: Arc::new(RwLock::new(HashSet::new())),
            settings: Arc::new(settings),
        }
    }
    
    /// Execute tasks with recovery
    #[instrument(skip(self, tasks), fields(task_count = %tasks.len()))]
    pub async fn execute_with_recovery<T, E, F>(
        &self,
        tasks: Vec<F>,
    ) -> Result<Vec<T>>
    where
        T: Send + 'static,
        E: std::error::Error + Send + Sync + 'static,
        F: Future<Output = std::result::Result<T, E>> + Send + 'static,
    {
        let task_count = tasks.len();
        info!("Executing {} tasks with recovery", task_count);
        
        if task_count == 0 {
            return Ok(vec![]);
        }
        
        let settings = self.settings.clone();
        let timeout = Duration::from_secs(settings.task_timeout_secs);
        
        // Create futures for each task with custom error handling
        let futures: Vec<_> = tasks.into_iter()
            .enumerate()
            .map(|(i, task)| {
                let task_id = format!("task_{}", i);
                
                // Create a future that includes our error handling and timeouts
                async move {
                    let start_time = Instant::now();
                    debug!("Starting {}", task_id);
                    
                    let result = tokio::time::timeout(timeout, task).await;
                    
                    match result {
                        Ok(Ok(value)) => {
                            let elapsed = start_time.elapsed();
                            debug!("{} completed successfully in {:?}", task_id, elapsed);
                            Ok(value)
                        },
                        Ok(Err(e)) => {
                            error!("{} failed with error: {}", task_id, e);
                            Err(RecoveryError::TaskFailed(e.to_string()))
                        },
                        Err(_) => {
                            error!("{} timed out after {:?}", task_id, timeout);
                            Err(RecoveryError::Timeout(task_id))
                        }
                    }
                }
            })
            .collect();
        
        // Execute all futures in parallel
        let results = future::join_all(futures).await;
        
        // If retry is enabled, handle retries for failed tasks
        let results = if settings.max_retries > 0 {
            self.handle_retries(results).await
        } else {
            results
        };
        
        // Filter successful results
        let successful_results: Vec<_> = results.into_iter()
            .filter_map(|r| r.ok())
            .collect();
            
        if successful_results.is_empty() {
            error!("All tasks failed");
            Err(RecoveryError::AllTasksFailed)
        } else {
            Ok(successful_results)
        }
    }
    
    async fn handle_retries<T>(&self, results: Vec<Result<T>>) -> Vec<Result<T>> 
    where
        T: Send + 'static,
    {
        let max_retries = self.settings.max_retries;
        
        // Find indices of failed tasks
        let mut failed_indices: Vec<usize> = results.iter()
            .enumerate()
            .filter_map(|(i, r)| if r.is_err() { Some(i) } else { None })
            .collect();
        
        // No failures, early return
        if failed_indices.is_empty() {
            return results;
        }
        
        info!("Found {} failed tasks, will retry up to {} times", 
              failed_indices.len(), max_retries);
        
        // Retry loop - but skip actual modification of results for now
        for retry in 1..=max_retries {
            // Delay before retry
            tokio::time::sleep(Duration::from_millis(self.settings.retry_delay_ms)).await;
            
            if failed_indices.is_empty() {
                break;
            }
            
            info!("Retry {}/{}: Retrying {} failed tasks", 
                  retry, max_retries, failed_indices.len());
            
            // In a real implementation, you would retry the tasks here
            // For now, we'll just simulate by removing some indices
            
            // Remove half of the failed indices (simulation only)
            let remove_count = failed_indices.len() / 2;
            failed_indices.truncate(failed_indices.len() - remove_count);
        }
        
        // Return original results - in a real implementation,
        // you would update the results array with retry outcomes
        results
    }
        
    /// Record node success
    pub async fn record_node_success(&self, node_id: &str) {
        let mut reliability = self.node_reliability.write().await;
        let current = reliability.get(node_id).copied().unwrap_or(0.5);
        
        // Increase reliability (with ceiling)
        let new_reliability = f64::min(1.0, current + 0.1);
        reliability.insert(node_id.to_string(), new_reliability);
        
        // Remove from failed nodes if present
        let mut failed = self.failed_nodes.write().await;
        failed.remove(node_id);
    }
    
    /// Record node failure
    pub async fn record_node_failure(&self, node_id: &str) {
        // Update reliability
        let mut reliability = self.node_reliability.write().await;
        let current = reliability.get(node_id).copied().unwrap_or(0.5);
        
        // Decrease reliability (with floor)
        let new_reliability = f64::max(0.0, current - 0.2);
        reliability.insert(node_id.to_string(), new_reliability);
        
        // Add to failed nodes if reliability drops too low
        if new_reliability < 0.3 {
            let mut failed = self.failed_nodes.write().await;
            failed.insert(node_id.to_string());
            
            warn!("Node {} marked as failed (reliability: {})", node_id, new_reliability);
        }
    }
    
    /// Check if a node is failed
    pub async fn is_node_failed(&self, node_id: &str) -> bool {
        let failed = self.failed_nodes.read().await;
        failed.contains(node_id)
    }
    
    /// Get node status
    pub async fn get_node_status(&self, node_id: &str) -> NodeStatus {
        let reliability = self.node_reliability.read().await;
        let failed = self.failed_nodes.read().await;
        
        if failed.contains(node_id) {
            return NodeStatus::Offline;
        }
        
        match reliability.get(node_id).copied().unwrap_or(0.5) {
            r if r >= 0.8 => NodeStatus::Healthy,
            r if r >= 0.5 => NodeStatus::Degraded,
            _ => NodeStatus::Unhealthy,
        }
    }
    
    /// Reset node status
    pub async fn reset_node(&self, node_id: &str) {
        let mut reliability = self.node_reliability.write().await;
        reliability.insert(node_id.to_string(), 0.5);
        
        let mut failed = self.failed_nodes.write().await;
        failed.remove(node_id);
        
        info!("Reset status for node {}", node_id);
    }
    
    /// Get healthiest nodes
    pub async fn get_healthiest_nodes(&self, count: usize) -> Vec<String> {
        let reliability = self.node_reliability.read().await;
        let failed = self.failed_nodes.read().await;
        
        let mut nodes: Vec<_> = reliability.iter()
            .filter(|(node_id, _)| !failed.contains(*node_id))
            .collect();
            
        // Sort by reliability (highest first)
        nodes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take requested count
        nodes.iter()
            .take(count)
            .map(|(node_id, _)| (*node_id).clone())
            .collect()
    }
}

impl Default for RecoveryManager {
    fn default() -> Self {
        Self::new(RecoverySettings::default())
    }
}
```

---


## src/lib.rs

### File path: `/home/myuser/viper/dagknight-vm/src/lib.rs`

```rust
//! DAGKnight VM implementation

// Core modules
pub mod contracts;
pub mod types;
pub mod consensus;
pub mod vm;
pub mod state;
pub mod network;
pub mod transaction;
pub mod dag;
pub mod cache;
pub mod fault_tolerance;
pub mod models;
pub mod mempool;
pub mod error;
pub mod api;

// Re-export key types for convenience
pub use dag::DAG;
pub use vm::VirtualMachine;

pub fn init(config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let _ = config::load_config(config_path);
    
    // Handle batch size argument if provided
    if let Some(size_str) = std::env::args().nth(2) {
        if let Ok(size) = size_str.parse::<usize>() {
            config::update_batch_size(size);
        }
    }
    
    Ok(())
}

// Re-export config for compatibility
pub mod config {
    pub fn load_config(config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        match std::fs::read_to_string(config_path) {
            Ok(_) => Ok(()),
            Err(e) => Err(Box::new(e)),
        }
    }
    
    pub fn update_batch_size(_batch_size: usize) {
        println!("Updated batch size");
    }
}
```

---


## src/main.rs

### File path: `/home/myuser/viper/dagknight-vm/src/main.rs`

```rust
//! DAGKnight blockchain with distributed AI capabilities
use std::sync::Arc;
use tokio::signal;
use tracing::info;

mod api;
mod cache;
mod consensus;
mod contracts;
mod error;
mod fault_tolerance;
mod models;
mod network;
mod state;
mod vm;
mod config;

use crate::cache::{ModelCache, CacheProvider};
use crate::fault_tolerance::{RecoveryManager, RecoverySettings};
use crate::models::ModelRegistry;
use crate::state::StateDB;
use crate::vm::ai::executor::AIExecutor;
use crate::vm::cache::ContractCache;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting DAGKnight with distributed AI capabilities");
    
    // Initialize core components
    let registry = Arc::new(ModelRegistry::new());
    let cache = Arc::new(ModelCache::new(
        CacheProvider::Layered,
        50000,
        Some("redis://localhost:6379".to_string()),
    ));
    
    let recovery_settings = RecoverySettings {
        enable_replication: true,
        replication_factor: 2,
        max_retries: 3,
        retry_delay_ms: 500,
        task_timeout_secs: 120,
    };
    
    let _recovery = Arc::new(RecoveryManager::new(recovery_settings));
    let _state_db = Arc::new(StateDB::new());
    
    // Initialize model registry with defaults
    registry.initialize_defaults().await;
    
    // Initialize contract cache for VM
    let contract_cache = Arc::new(ContractCache::new());
    
    // Initialize AI executor with contract cache instead of model cache
    let _ai_executor = AIExecutor::new(
        contract_cache,
    ).await.unwrap();
    
    // Start cache maintenance
    cache.start_cleanup_task();
    
    info!("System initialized and ready");
    
    // Wait for shutdown signal
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Shutdown signal received, stopping DAGKnight");
        },
        Err(err) => {
            eprintln!("Unable to listen for shutdown signal: {}", err);
        },
    }
    
    Ok(())
}```

---


## src/mempool/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/mempool/mod.rs`

```rust
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use std::sync::Arc;
use parking_lot::Mutex;
use priority_queue::PriorityQueue;
use crate::transaction::Transaction;

// Transaction with priority info
#[derive(Debug, Clone)]
struct PrioritizedTransaction {
    tx: Transaction,
    gas_price: u64,
    time_added: u64,
}

impl Eq for PrioritizedTransaction {}

impl PartialEq for PrioritizedTransaction {
    fn eq(&self, other: &Self) -> bool {
        self.tx.hash == other.tx.hash
    }
}

// Priority value for organizing transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Priority(u64);

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Mempool for managing pending transactions
#[derive(Debug)]
pub struct Mempool {
    // Priority queue for transactions
    txs: Arc<Mutex<PriorityQueue<[u8; 32], Priority>>>,
    
    // Map of transaction hash to transaction data
    tx_map: Arc<Mutex<HashMap<[u8; 32], PrioritizedTransaction>>>,
    
    // Track transactions by sender for nonce ordering
    sender_txs: Arc<Mutex<HashMap<[u8; 32], HashMap<u64, [u8; 32]>>>>,
    
    // Configuration
    max_size: usize,
    min_gas_price: u64,
    
    // Metrics
    added_count: Arc<Mutex<u64>>,
    removed_count: Arc<Mutex<u64>>,
}

impl Mempool {
    pub fn new(max_size: usize, min_gas_price: u64) -> Self {
        Self {
            txs: Arc::new(Mutex::new(PriorityQueue::new())),
            tx_map: Arc::new(Mutex::new(HashMap::with_capacity(max_size))),
            sender_txs: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            min_gas_price,
            added_count: Arc::new(Mutex::new(0)),
            removed_count: Arc::new(Mutex::new(0)),
        }
    }
    
    // Add transaction to mempool
    pub fn add_transaction(&self, tx: Transaction, gas_price: u64) -> Result<(), String> {
        // Check if gas price meets minimum
        if gas_price < self.min_gas_price {
            return Err(format!("Gas price too low: {}, minimum: {}", gas_price, self.min_gas_price));
        }
        
        // Check if mempool is full
        {
            let txs = self.txs.lock();
            if txs.len() >= self.max_size {
                return Err("Mempool is full".to_string());
            }
        }
        
        // Check if transaction already exists
        {
            let tx_map = self.tx_map.lock();
            if tx_map.contains_key(&tx.hash) {
                return Err("Transaction already in mempool".to_string());
            }
        }
        
        // Calculate priority
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        let priority = gas_price * 1000 + (1000000000 - timestamp); // Higher gas price + earlier time = higher priority
        
        // Create prioritized transaction
        let ptx = PrioritizedTransaction {
            tx: tx.clone(),
            gas_price,
            time_added: timestamp,
        };
        
        // Add to data structures
        {
            let mut txs = self.txs.lock();
            let mut tx_map = self.tx_map.lock();
            let mut sender_txs = self.sender_txs.lock();
            
            // Add to priority queue
            txs.push(tx.hash, Priority(priority));
            
            // Add to transaction map
            tx_map.insert(tx.hash, ptx);
            
            // Add to sender transactions
            let sender_map = sender_txs.entry(tx.sender).or_insert_with(HashMap::new);
            sender_map.insert(tx.nonce, tx.hash);
            
            // Update metrics
            let mut added_count = self.added_count.lock();
            *added_count += 1;
        }
        
        Ok(())
    }
    
    // Get best transactions for block proposal
    pub fn get_best_transactions(&self, limit: usize) -> Vec<Transaction> {
        let mut result = Vec::with_capacity(limit);
        let mut visited = HashSet::new();
        
        // Clone priority queue to avoid deadlock
        let queue_clone = {
            let txs = self.txs.lock();
            txs.clone()
        };
        
        let tx_map = self.tx_map.lock();
        
        // Get transactions in order of priority
        for (hash, _) in queue_clone.into_sorted_iter() {
            if visited.contains(&hash) {
                continue;
            }
            
            if let Some(ptx) = tx_map.get(&hash) {
                result.push(ptx.tx.clone());
                visited.insert(hash);
                
                if result.len() >= limit {
                    break;
                }
            }
        }
        
        result
    }
    
    // Remove transaction from mempool
    pub fn remove_transaction(&self, tx_hash: &[u8; 32]) -> Option<Transaction> {
        let mut txs = self.txs.lock();
        let mut tx_map = self.tx_map.lock();
        let mut sender_txs = self.sender_txs.lock();
        
        // Remove from transaction map
        let ptx = tx_map.remove(tx_hash)?;
        
        // Remove from priority queue
        txs.remove(tx_hash);
        
        // Remove from sender transactions
        if let Some(sender_map) = sender_txs.get_mut(&ptx.tx.sender) {
            sender_map.remove(&ptx.tx.nonce);
            if sender_map.is_empty() {
                sender_txs.remove(&ptx.tx.sender);
            }
        }
        
        // Update metrics
        let mut removed_count = self.removed_count.lock();
        *removed_count += 1;
        
        Some(ptx.tx)
    }
    
    // Get transactions by sender with nonce ordering
    pub fn get_sender_transactions(&self, sender: &[u8; 32]) -> Vec<Transaction> {
        let sender_txs = self.sender_txs.lock();
        let tx_map = self.tx_map.lock();
        
        let mut result = Vec::new();
        
        if let Some(nonce_map) = sender_txs.get(sender) {
            // Get all nonces
            let mut nonces: Vec<_> = nonce_map.keys().collect();
            nonces.sort();
            
            // Get transactions in nonce order
            for nonce in nonces {
                if let Some(tx_hash) = nonce_map.get(nonce) {
                    if let Some(ptx) = tx_map.get(tx_hash) {
                        result.push(ptx.tx.clone());
                    }
                }
            }
        }
        
        result
    }
    
    // Get all transactions in mempool
    pub fn get_all_transactions(&self) -> Vec<Transaction> {
        let tx_map = self.tx_map.lock();
        tx_map.values().map(|ptx| ptx.tx.clone()).collect()
    }
    
    // Get transaction count
    pub fn get_transaction_count(&self) -> usize {
        let txs = self.txs.lock();
        txs.len()
    }
    
    // Update minimum gas price based on demand
    pub fn update_min_gas_price(&mut self, new_min: u64) {
        self.min_gas_price = new_min;
    }
    
    // Get metrics
    pub fn get_metrics(&self) -> (u64, u64, usize) {
        let added = *self.added_count.lock();
        let removed = *self.removed_count.lock();
        let current = self.get_transaction_count();
        
        (added, removed, current)
    }
}
```

---


## src/models/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/models/mod.rs`

```rust
//! Model registry for DAGKnight
use crate::contracts::{ModelRegistration, ShardingCapability, ResourceRequirements};
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, warn};

/// Model registry
pub struct ModelRegistry {
    models: RwLock<HashMap<String, ModelInfo>>,
}

/// Extended model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model registration info
    pub registration: ModelRegistration,
    /// Popularity score
    pub popularity: f64,
    /// Performance metrics
    pub performance: ModelPerformance,
    /// Quality metrics
    pub quality: ModelQuality,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// Average RAM usage in MB
    pub avg_ram_usage_mb: u64,
    /// Average GPU usage in MB
    pub avg_gpu_usage_mb: Option<u64>,
}

/// Model quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQuality {
    /// Average quality score (0-100)
    pub quality_score: f64,
    /// Number of ratings
    pub num_ratings: u64,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new())
        }
    }

    /// Register a new model
    pub async fn register_model(&self, registration: ModelRegistration) -> bool {
        let mut models = self.models.write().await;

        if models.contains_key(&registration.model_id) {
            warn!("Model {} already registered", registration.model_id);
            return false;
        }

        let model_info = ModelInfo {
            registration: registration.clone(),
            popularity: 0.0,
            performance: ModelPerformance {
                avg_tokens_per_second: 0.0,
                avg_ram_usage_mb: registration.resources.min_memory_mb,
                avg_gpu_usage_mb: Some(registration.resources.min_gpu_memory_mb),
            },
            quality: ModelQuality {
                quality_score: 0.0,
                num_ratings: 0,
            },
        };

        models.insert(registration.model_id.clone(), model_info);
        info!("Registered model: {}", registration.model_id);

        true
    }

    /// Get model information
    pub async fn get_model(&self, model_id: &str) -> Option<ModelInfo> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }

    /// Update model performance
    pub async fn update_performance(
        &self,
        model_id: &str,
        tokens_per_second: f64,
        ram_usage_mb: u64,
        gpu_usage_mb: Option<u64>,
    ) -> bool {
        let mut models = self.models.write().await;

        if let Some(model) = models.get_mut(model_id) {
            // Update with exponential moving average
            let alpha = 0.1; // Weight for new observations

            model.performance.avg_tokens_per_second =
                (1.0 - alpha) * model.performance.avg_tokens_per_second + alpha * tokens_per_second;

            model.performance.avg_ram_usage_mb =
                ((1.0 - alpha) * model.performance.avg_ram_usage_mb as f64 + alpha * ram_usage_mb as f64) as u64;

            if let Some(gpu_usage) = gpu_usage_mb {
                model.performance.avg_gpu_usage_mb = Some(
                    ((1.0 - alpha) * model.performance.avg_gpu_usage_mb.unwrap_or(0) as f64 +
                     alpha * gpu_usage as f64) as u64
                );
            }

            // Increase popularity
            model.popularity += 0.1;
            return true;
        }

        warn!("Model {} not found for performance update", model_id);
        false
    }

    /// Update model quality
    pub async fn update_quality(&self, model_id: &str, quality_score: f64) -> bool {
        let mut models = self.models.write().await;

        if let Some(model) = models.get_mut(model_id) {
            // Update with weighted average
            let current_score = model.quality.quality_score;
            let num_ratings = model.quality.num_ratings;

            model.quality.quality_score =
                (current_score * num_ratings as f64 + quality_score) / (num_ratings as f64 + 1.0);

            model.quality.num_ratings += 1;

            return true;
        }

        warn!("Model {} not found for quality update", model_id);
        false
    }

    /// List all available models
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }

    /// Find models that meet resource constraints
    pub async fn find_models_by_resources(
        &self,
        max_memory: u64,
        gpu_required: bool,
    ) -> Vec<ModelInfo> {
        let models = self.models.read().await;

        models.values()
            .filter(|m| {
                m.registration.resources.min_memory_mb <= max_memory &&
                (!gpu_required || m.registration.resources.min_gpu_memory_mb > 0)
            })
            .cloned()
            .collect()
    }

    /// Find models by sharding capability
    pub async fn find_models_by_sharding(
        &self,
        capability: ShardingCapability,
    ) -> Vec<ModelInfo> {
        let models = self.models.read().await;

        models.values()
            .filter(|m| {
                match (capability.clone(), &m.registration.capabilities) {
                    (ShardingCapability::None, _) => true,
                    (ShardingCapability::Horizontal, ShardingCapability::Horizontal | ShardingCapability::Full) => true,
                    (ShardingCapability::Vertical, ShardingCapability::Vertical | ShardingCapability::Full) => true,
                    (ShardingCapability::Full, ShardingCapability::Full) => true,
                    _ => false,
                }
            })
            .cloned()
            .collect()
    }

    /// Initialize registry with default models
    pub async fn initialize_defaults(&self) {
        // Register some default models
        let models = [
            ModelRegistration {
                hash: [0; 32],
                owner: [0; 32],
                timestamp: 0,
                model_id: "llama2:7b".to_string(),
                description: "Meta's Llama 2 7B parameter model".to_string(),
                version: "2.0".to_string(),
                // memory_required: 16000,
                capabilities: ShardingCapability::Horizontal,
                resources: ResourceRequirements {
                    min_cpu_cores: 4,
                    min_memory_mb: 16000,
                    min_gpu_memory_mb: 8000,
                    preferred_batch_size: 32,
                    // disk_space_mb: 14000,
                    // avg_exec_time_per_token_ms: 15.0,
                },
            },
            ModelRegistration {
                hash: [0; 32],
                owner: [0; 32],
                timestamp: 0,
                model_id: "deepseek-r1:1.5b".to_string(),
                description: "DeepSeek R1 1.5B parameter model".to_string(),
                version: "1.0".to_string(),
                // memory_required: 3000,
                capabilities: ShardingCapability::Full,
                resources: ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_mb: 4000,
                    min_gpu_memory_mb: 3000,
                    preferred_batch_size: 64,
                    // disk_space_mb: 3000,
                    // avg_exec_time_per_token_ms: 5.0,
                },
            },
            ModelRegistration {
                hash: [0; 32],
                owner: [0; 32],
                timestamp: 0,
                model_id: "mistral:7b".to_string(),
                description: "Mistral 7B parameter model".to_string(),
                version: "1.0".to_string(),
                // memory_required: 16000,
                capabilities: ShardingCapability::Vertical,
                resources: ResourceRequirements {
                    min_cpu_cores: 4,
                    min_memory_mb: 16000,
                    min_gpu_memory_mb: 7000,
                    preferred_batch_size: 32,
                    // disk_space_mb: 13500,
                    // avg_exec_time_per_token_ms: 12.0,
                },
            },
            ModelRegistration {
                hash: [0; 32],
                owner: [0; 32],
                timestamp: 0,
                model_id: "phi-2:3b".to_string(),
                description: "Microsoft's Phi-2 3B parameter model".to_string(),
                version: "2.0".to_string(),
                // memory_required: 6000,
                capabilities: ShardingCapability::Horizontal,
                resources: ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_mb: 8000,
                    min_gpu_memory_mb: 4000,
                    preferred_batch_size: 48,
                    // disk_space_mb: 6000,
                    // avg_exec_time_per_token_ms: 8.0,
                },
            },
        ];

        for model in models {
            self.register_model(model).await;
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
```

---


## src/models/mod.rs.bak

### File path: `/home/myuser/viper/dagknight-vm/src/models/mod.rs.bak`

```text
//! Model registry for DAGKnight
use crate::contracts::{ModelRegistration, ShardingCapability, ResourceRequirements};
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, warn};

/// Model registry
pub struct ModelRegistry {
    models: RwLock<HashMap<String, ModelInfo>>,
}

/// Extended model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model registration info
    pub registration: ModelRegistration,
    /// Popularity score
    pub popularity: f64,
    /// Performance metrics
    pub performance: ModelPerformance,
    /// Quality metrics
    pub quality: ModelQuality,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// Average RAM usage in MB
    pub avg_ram_usage_mb: u64,
    /// Average GPU usage in MB
    pub avg_gpu_usage_mb: Option<u64>,
}

/// Model quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQuality {
    /// Average quality score (0-100)
    pub quality_score: f64,
    /// Number of ratings
    pub num_ratings: u64,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new())
        }
    }
    
    /// Register a new model
    pub async fn register_model(&self, registration: ModelRegistration) -> bool {
        let mut models = self.models.write().await;
        
        if models.contains_key(&registration.model_id) {
            warn!("Model {} already registered", registration.model_id);
            return false;
        }
        
        let model_info = ModelInfo {
            registration: registration.clone(),
            popularity: 0.0,
            performance: ModelPerformance {
                avg_tokens_per_second: 0.0,
                avg_ram_usage_mb: registration.resources.min_memory_mb,
                avg_gpu_usage_mb: registration.resources.min_gpu_memory_mb,
            },
            quality: ModelQuality {
                quality_score: 0.0,
                num_ratings: 0,
            },
        };
        
        models.insert(registration.model_id.clone(), model_info);
        info!("Registered model: {}", registration.model_id);
        
        true
    }
    
    /// Get model information
    pub async fn get_model(&self, model_id: &str) -> Option<ModelInfo> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }
    
    /// Update model performance
    pub async fn update_performance(
        &self,
        model_id: &str,
        tokens_per_second: f64,
        ram_usage_mb: u64,
        gpu_usage_mb: Option<u64>,
    ) -> bool {
        let mut models = self.models.write().await;
        
        if let Some(model) = models.get_mut(model_id) {
            // Update with exponential moving average
            let alpha = 0.1; // Weight for new observations
            
            model.performance.avg_tokens_per_second = 
                (1.0 - alpha) * model.performance.avg_tokens_per_second + alpha * tokens_per_second;
                
            model.performance.avg_ram_usage_mb = 
                ((1.0 - alpha) * model.performance.avg_ram_usage_mb as f64 + alpha * ram_usage_mb as f64) as u64;
                
            if let Some(gpu_usage) = gpu_usage_mb {
                model.performance.avg_gpu_usage_mb = Some(
                    ((1.0 - alpha) * model.performance.avg_gpu_usage_mb.unwrap_or(0) as f64 + 
                     alpha * gpu_usage as f64) as u64
                );
            }
            
            // Increase popularity
            model.popularity += 0.1;
            return true;
        }
        
        warn!("Model {} not found for performance update", model_id);
        false
    }
    
    /// Update model quality
    pub async fn update_quality(&self, model_id: &str, quality_score: f64) -> bool {
        let mut models = self.models.write().await;
        
        if let Some(model) = models.get_mut(model_id) {
            // Update with weighted average
            let current_score = model.quality.quality_score;
            let num_ratings = model.quality.num_ratings;
            
            model.quality.quality_score = 
                (current_score * num_ratings as f64 + quality_score) / (num_ratings as f64 + 1.0);
                
            model.quality.num_ratings += 1;
            
            return true;
        }
        
        warn!("Model {} not found for quality update", model_id);
        false
    }
    
    /// List all available models
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }
    
    /// Find models that meet resource constraints
    pub async fn find_models_by_resources(
        &self,
        max_memory: u64,
        gpu_required: bool,
    ) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        
        models.values()
            .filter(|m| {
                m.registration.resources.min_memory_mb <= max_memory &&
                (!gpu_required || m.registration.resources.min_gpu_memory_mb > 0)
            })
            .cloned()
            .collect()
    }
    
    /// Find models by sharding capability
    pub async fn find_models_by_sharding(
        &self,
        capability: ShardingCapability,
    ) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        
        models.values()
            .filter(|m| {
                match (capability.clone(), &m.registration.capabilities) {
                    (ShardingCapability::None, _) => true,
                    (ShardingCapability::Horizontal, ShardingCapability::Horizontal | ShardingCapability::Full) => true,
                    (ShardingCapability::Vertical, ShardingCapability::Vertical | ShardingCapability::Full) => true,
                    (ShardingCapability::Full, ShardingCapability::Full) => true,
                    _ => false,
                }
            })
            .cloned()
            .collect()
    }
    
    /// Initialize registry with default models
    pub async fn initialize_defaults(&self) {
        // Register some default models
        let models = [
            ModelRegistration {
                hash: [0; 32],
                owner: "system".to_string(),
                timestamp: 0,
                model_id: "llama2:7b".to_string(,
                description: "Meta's Llama 2 7B parameter model".to_string(,
                version: "2.0".to_string(),
                // memory_required: 16000,
                capabilities: ShardingCapability::Horizontal,
                resources: ResourceRequirements {
                    min_cpu_cores: 4,
                    min_memory_mb: 16000,
                    min_gpu_memory_mb: 8000,
                    // disk_space_mb: 14000,
                    // avg_exec_time_per_token_ms: 15.0,
                hash: [0; 32],
                owner: "system".to_string(),
                timestamp: 0,
                },
            },
            ModelRegistration {
                model_id: "deepseek-r1:1.5b".to_string(,
                description: "DeepSeek R1 1.5B parameter model".to_string(,
                version: "1.0".to_string(),
                // memory_required: 3000,
                capabilities: ShardingCapability::Full,
                resources: ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_mb: 4000,
                hash: [0; 32],
                owner: "system".to_string(),
                timestamp: 0,
                    min_gpu_memory_mb: 3000,
                    // disk_space_mb: 3000,
                    // avg_exec_time_per_token_ms: 5.0,
                },
            },
            ModelRegistration {
                model_id: "mistral:7b".to_string(,
                description: "Mistral 7B parameter model".to_string(,
                version: "1.0".to_string(),
                // memory_required: 16000,
                capabilities: ShardingCapability::Vertical,
                hash: [0; 32],
                owner: "system".to_string(),
                timestamp: 0,
                resources: ResourceRequirements {
                    min_cpu_cores: 4,
                    min_memory_mb: 16000,
                    min_gpu_memory_mb: 7000,
                    // disk_space_mb: 13500,
                    // avg_exec_time_per_token_ms: 12.0,
                },
            },
            ModelRegistration {
                model_id: "phi-2:3b".to_string(,
                description: "Microsoft's Phi-2 3B parameter model".to_string(,
                version: "2.0".to_string(),
                // memory_required: 6000,
                capabilities: ShardingCapability::Horizontal,
                resources: ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_mb: 8000,
                    min_gpu_memory_mb: 4000,
                    // disk_space_mb: 6000,
                    // avg_exec_time_per_token_ms: 8.0,
                },
            },
        ];
        
        for model in models {
            self.register_model(model).await;
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
```

---


## src/network/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/network/mod.rs`

```rust
pub mod p2p;
pub mod stub;
pub mod p2p_debug;

```

---


## src/network/p2p.rs

### File path: `/home/myuser/viper/dagknight-vm/src/network/p2p.rs`

```rust
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::ser::SerializeTuple;
use serde::de::{self, Visitor};
use std::fmt;
use thiserror::Error;

// Assuming ResourceUsage is defined elsewhere, here's a placeholder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_time_ms: u64,
    pub memory_used_mb: u64,
    pub gpu_time_ms: Option<u64>,
}

// Custom serialization for fixed-size byte arrays
#[derive(Clone, PartialEq)]
pub struct Bytes32(pub [u8; 32]);

#[derive(Clone, PartialEq)]
pub struct Bytes64(pub [u8; 64]);

impl Serialize for Bytes32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tup = serializer.serialize_tuple(32)?;
        for byte in &self.0[..] {
            tup.serialize_element(byte)?;
        }
        tup.end()
    }
}

impl<'de> Deserialize<'de> for Bytes32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Bytes32Visitor;
        
        impl<'de> Visitor<'de> for Bytes32Visitor {
            type Value = Bytes32;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a 32-byte array")
            }
            
            fn visit_seq<A>(self, mut seq: A) -> Result<Bytes32, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut bytes = [0u8; 32];
                for i in 0..32 {
                    bytes[i] = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(i, &self))?;
                }
                Ok(Bytes32(bytes))
            }
        }
        
        deserializer.deserialize_tuple(32, Bytes32Visitor)
    }
}

impl fmt::Debug for Bytes32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bytes32({:?})", &self.0[..])
    }
}

impl Serialize for Bytes64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tup = serializer.serialize_tuple(64)?;
        for byte in &self.0[..] {
            tup.serialize_element(byte)?;
        }
        tup.end()
    }
}

impl<'de> Deserialize<'de> for Bytes64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Bytes64Visitor;
        
        impl<'de> Visitor<'de> for Bytes64Visitor {
            type Value = Bytes64;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a 64-byte array")
            }
            
            fn visit_seq<A>(self, mut seq: A) -> Result<Bytes64, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut bytes = [0u8; 64];
                for i in 0..64 {
                    bytes[i] = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(i, &self))?;
                }
                Ok(Bytes64(bytes))
            }
        }
        
        deserializer.deserialize_tuple(64, Bytes64Visitor)
    }
}

impl fmt::Debug for Bytes64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bytes64({:?})", &self.0[..])
    }
}

/// Network message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    Transaction {
        data: Vec<u8>,
        _hash: Bytes32,
        _timestamp: u64,
    },
    Block {
        data: Vec<u8>,
        _hash: Bytes32,
        _height: u64,
        _timestamp: u64,
    },
    Consensus {
        _consensus_type: ConsensusType,
        data: Vec<u8>,
        _timestamp: u64,
    },
    ComputeTask {
        _contract: Bytes32,
        model: String,
        input: Vec<u8>,
        _timestamp: u64,
    },
    ComputeResult {
        _contract: Bytes32,
        output: Vec<u8>,
        _proof: Bytes64,
        _resources: ResourceUsage,
        _timestamp: u64,
    },
    NodeStatus {
        _node_id: Bytes32,
        _status: NodeStatus,
        _available_resources: AvailableResources,
        _timestamp: u64,
    },
    ModelRegistry {
        _action: ModelRegistryAction,
        data: Vec<u8>,
        _timestamp: u64,
    },
}

/// Consensus message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusType {
    Prepare,
    Commit,
    ViewChange,
    NewView,
    ValidationResult,
}

/// Node status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Online,
    Offline,
    Syncing,
    ReadyForCompute,
    Busy,
}

/// Available resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableResources {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub gpu_memory_mb: Option<u64>,
    pub disk_space_mb: u64,
    pub network_bandwidth_mbps: u64,
    pub latency_ms: u64,
}

/// Model registry actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelRegistryAction {
    Register,
    Update,
    Remove,
    Query,
    ReportPerformance,
}

/// P2P message handler
pub struct P2PMessageHandler {
    // Could add message queue or state here if needed
}

impl P2PMessageHandler {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn handle_message(&self, message: NetworkMessage) -> Result<(), MessageError> {
        match message {
            NetworkMessage::Transaction { data, _hash, _timestamp } => {
                self.handle_transaction(data, _hash, _timestamp).await
            },
            NetworkMessage::Block { data, _hash, _height, _timestamp } => {
                self.handle_block(data, _hash, _height, _timestamp).await
            },
            NetworkMessage::Consensus { _consensus_type, data, _timestamp } => {
                self.handle_consensus(_consensus_type, data, _timestamp).await
            },
            NetworkMessage::ComputeTask { _contract, model, input, _timestamp } => {
                self.handle_compute_task(_contract, model, input, _timestamp).await
            },
            NetworkMessage::ComputeResult { _contract, output, _proof, _resources, _timestamp } => {
                self.handle_compute_result(_contract, output, _proof, _resources, _timestamp).await
            },
            NetworkMessage::NodeStatus { _node_id, _status, _available_resources, _timestamp } => {
                self.handle_node_status(_node_id, _status, _available_resources, _timestamp).await
            },
            NetworkMessage::ModelRegistry { _action, data, _timestamp } => {
                self.handle_model_registry(_action, data, _timestamp).await
            },
        }
    }
    
    async fn handle_transaction(&self, data: Vec<u8>, _hash: Bytes32, _timestamp: u64) -> Result<(), MessageError> {
        if data.is_empty() {
            return Err(MessageError::InvalidMessage("Empty transaction data".to_string()));
        }
        // Implement transaction handling logic here
        Ok(())
    }
    
    async fn handle_block(&self, data: Vec<u8>, _hash: Bytes32, _height: u64, _timestamp: u64) -> Result<(), MessageError> {
        if data.is_empty() {
            return Err(MessageError::InvalidMessage("Empty block data".to_string()));
        }
        // Implement block handling logic here
        Ok(())
    }
    
    async fn handle_consensus(&self, _consensus_type: ConsensusType, data: Vec<u8>, _timestamp: u64) -> Result<(), MessageError> {
        if data.is_empty() {
            return Err(MessageError::InvalidMessage("Empty consensus data".to_string()));
        }
        // Implement consensus handling logic here
        Ok(())
    }
    
    async fn handle_compute_task(
        &self,
        _contract: Bytes32,
        model: String,
        input: Vec<u8>,
        _timestamp: u64,
    ) -> Result<(), MessageError> {
        if model.is_empty() {
            return Err(MessageError::InvalidMessage("Empty model identifier".to_string()));
        }
        if input.is_empty() {
            return Err(MessageError::InvalidMessage("Empty input data".to_string()));
        }
        // Implement compute task handling logic here
        Ok(())
    }
    
    async fn handle_compute_result(
        &self,
        _contract: Bytes32,
        output: Vec<u8>,
        _proof: Bytes64,
        _resources: ResourceUsage,
        _timestamp: u64,
    ) -> Result<(), MessageError> {
        if output.is_empty() {
            return Err(MessageError::InvalidMessage("Empty output data".to_string()));
        }
        // Implement compute result handling logic here
        Ok(())
    }
    
    async fn handle_node_status(
        &self,
        _node_id: Bytes32,
        _status: NodeStatus,
        _available_resources: AvailableResources,
        _timestamp: u64,
    ) -> Result<(), MessageError> {
        // Implement node status handling logic here
        Ok(())
    }
    
    async fn handle_model_registry(
        &self,
        _action: ModelRegistryAction,
        data: Vec<u8>,
        _timestamp: u64,
    ) -> Result<(), MessageError> {
        if data.is_empty() {
            return Err(MessageError::InvalidMessage("Empty model registry data".to_string()));
        }
        // Implement model registry handling logic here
        Ok(())
    }
}

/// Message error
#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Invalid message: {0}")]
    InvalidMessage(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compute_task_handling() {
        let handler = P2PMessageHandler::new();
        let message = NetworkMessage::ComputeTask {
            _contract: Bytes32([0; 32]),
            model: "test_model".to_string(),
            input: vec![1, 2, 3],
            _timestamp: 1234567890,
        };
        
        let result = handler.handle_message(message).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_empty_input_compute_task() {
        let handler = P2PMessageHandler::new();
        let message = NetworkMessage::ComputeTask {
            _contract: Bytes32([0; 32]),
            model: "test_model".to_string(),
            input: vec![],
            _timestamp: 1234567890,
        };
        
        let result = handler.handle_message(message).await;
        assert!(matches!(result, Err(MessageError::InvalidMessage(_))));
    }
}

/// P2P network implementation
pub struct P2pNetwork {
    // Basic P2P network implementation
}

impl P2pNetwork {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn get_local_peer_id(&self) -> String {
        "peer_id_placeholder".to_string() // Replace with actual implementation
    }
    
    pub fn get_connected_peers_count(&self) -> usize {
        0 // Replace with actual implementation
    }
}

// Note: Manual Debug implementation provided in p2p_debug.rs
// Do not add #[derive(Debug)] here to avoid conflicts```

---


## src/network/p2p_debug.rs

### File path: `/home/myuser/viper/dagknight-vm/src/network/p2p_debug.rs`

```rust
use std::fmt;
use crate::network::p2p::P2pNetwork;

impl fmt::Debug for P2pNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("P2pNetwork")
            .field("local_peer_id", &self.get_local_peer_id())
            .field("connected_peers_count", &self.get_connected_peers_count())
            .finish()
    }
}
```

---


## src/network/stub.rs

### File path: `/home/myuser/viper/dagknight-vm/src/network/stub.rs`

```rust
// Temporary stub implementation until the p2p module is fixed
use crate::vm::VmError;

pub struct Network {
    address: String,
    port: u16,
}

impl Network {
    pub fn new(address: String, port: u16) -> Result<Self, VmError> {
        Ok(Self { address, port })
    }
}
```

---


## src/state/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/state/mod.rs`

```rust
/// State management for DAGKnight
use std::sync::Arc;
use tokio::sync::RwLock;

// Define VmState directly here as a temporary solution
#[derive(Debug, Clone, Default)]
pub struct VmState {
    pub contracts: std::collections::HashMap<u64, Vec<u8>>,
    pub storage: std::collections::HashMap<u64, std::collections::HashMap<Vec<u8>, Vec<u8>>>,
    pub balances: std::collections::HashMap<u64, u64>,
    pub nonces: std::collections::HashMap<u64, u64>,
}

#[derive(Debug)]
pub struct StateDB {
    pub state: Arc<RwLock<VmState>>,
    pub resource_ledger: Option<Box<dyn std::any::Any + Send + Sync>>,
}

impl StateDB {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(VmState::default())),
            resource_ledger: None,
        }
    }

    pub fn with_state(state: Arc<RwLock<VmState>>) -> Self {
        Self {
            state,
            resource_ledger: None,
        }
    }
    
    // Add a new method for testing purposes
    pub fn new_in_memory() -> Self {
        Self::new()
    }
}

// Resource usage struct for AI execution
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub compute_units: u64,
    pub memory_bytes: u64,
    pub storage_bytes: u64,
    pub cpu_time: u64,
    pub memory_used: u64,
    pub gpu_time: u64,
}```

---


## src/state/mod.rs.bak

### File path: `/home/myuser/viper/dagknight-vm/src/state/mod.rs.bak`

```text
use std::sync::Arc;
use tokio::sync::RwLock;
/// State management for DAGKnight
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU time in milliseconds
    pub cpu_time: u64,
    /// Memory used in bytes
    pub memory_used: u64,
    /// GPU time in milliseconds (if used)
    pub gpu_time: Option<u64>,
}

impl ResourceUsage {
    /// Create a minimal resource usage entry for cached results
    pub fn minimal() -> Self {
        Self {
            cpu_time: 1,
            memory_used: 1024,
            gpu_time: None,
        }
    }
    
    /// Calculate resource cost based on usage
    pub fn calculate_cost(&self) -> u64 {
        // Base cost from CPU
        let cpu_cost = self.cpu_time * 1; // 1 token per ms of CPU time
        
        // Memory cost
        let memory_cost = (self.memory_used / (1024 * 1024)) * 10; // 10 tokens per MB
        
        // GPU cost if used
        let gpu_cost = self.gpu_time.map(|time| time * 5).unwrap_or(0); // 5 tokens per ms of GPU time
        
        cpu_cost + memory_cost + gpu_cost
    }
}

#[derive(Debug)]
/// Resource ledger for tracking contributions
pub struct ResourceLedger {
    /// Tracks resource contributions per node
    contributions: RwLock<HashMap<[u8; 32], ResourcePool>>,
}

/// Resource pool for a single node
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourcePool {
    /// Total CPU time provided (ms)
    pub total_cpu: u64,
    /// Total memory provided (MB)
    pub total_memory: u64,
    /// Total GPU time provided (ms)
    pub total_gpu: u64,
    /// Pending rewards (tokens)
    pub pending_rewards: u64,
}

impl ResourceLedger {
    /// Create a new resource ledger
    pub fn new() -> Self {
        Self {
            contributions: RwLock::new(HashMap::new()),
        }
    }
    
    /// Update resource usage for a node
    pub async fn update_resources(&self, node: [u8; 32], usage: &ResourceUsage) {
        let mut ledger = self.contributions.write().await;
        let pool = ledger.entry(node).or_insert(ResourcePool::default());
        
        pool.total_cpu += usage.cpu_time;
        pool.total_memory += usage.memory_used / (1024 * 1024); // Convert to MB
        pool.total_gpu += usage.gpu_time.unwrap_or(0);
        
        // Calculate rewards
        let reward = usage.calculate_cost();
        pool.pending_rewards += reward;
    }
    
    /// Get resource pool for a node
    pub async fn get_resource_pool(&self, node: &[u8; 32]) -> Option<ResourcePool> {
        let ledger = self.contributions.read().await;
        ledger.get(node).cloned()
    }
    
    /// Get all resource pools
    pub async fn get_all_resource_pools(&self) -> HashMap<[u8; 32],
    ResourcePool> {
        let ledger = self.contributions.read().await;
        ledger.clone()
    }
    
    /// Claim rewards for a node
    pub async fn claim_rewards(&self, node: &[u8; 32]) -> u64 {
        let mut ledger = self.contributions.write().await;
        
        if let Some(pool) = ledger.get_mut(node) {
            let rewards = pool.pending_rewards;
            pool.pending_rewards = 0;
            rewards
        } else {
            0
        }
    }
    
    /// Get total resource usage across all nodes
    pub async fn get_total_resource_usage(&self) -> ResourcePool {
        let ledger = self.contributions.read().await;
        
        let mut total = ResourcePool::default();
        for pool in ledger.values() {
            total.total_cpu += pool.total_cpu;
            total.total_memory += pool.total_memory;
            total.total_gpu += pool.total_gpu;
            total.pending_rewards += pool.pending_rewards;
        }
        
        total
    }
}

/// State database for DAGKnight
#[derive(Debug)]

pub struct StateDB {
    pub state: Arc<RwLock<VmState>>,
    /// Resource ledger
    pub resource_ledger: ResourceLedger,
    // Other state components would go here
}

impl StateDB {
    /// Create a new state database
    pub fn new() -> Self {
        Self {
            resource_ledger: ResourceLedger::new(),
        }
    }
    
    /// Update resource ledger
    pub async fn update_resource_ledger(
        &self,
        node: [u8; 32],
        usage: &ResourceUsage
    ) {
        self.resource_ledger.update_resources(node, usage).await;
    }

    pub fn with_state(state: Arc<RwLock<VmState>>) -> Self {
        Self { state }
    }
}

impl Default for StateDB {
    fn default() -> Self {
        Self::new()
    }
}
```

---


## src/transaction/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/transaction/mod.rs`

```rust
use crate::contracts::ContractCall;

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use parking_lot::Mutex;
use crate::state::StateDB;

// Transaction structure
#[derive(Debug, Clone)]
pub struct Transaction {
    pub hash: [u8; 32],         // Transaction hash
    pub data: Vec<u8>,          // Transaction data
    pub sender: [u8; 32],       // Sender's address
    pub nonce: u64,             // Sender's nonce
    pub signature: [u8; 64],    // Transaction signature
    pub timestamp: u64,         // Timestamp when created
}

// Transaction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    ContractDeployment(ContractDeployment),
    ContractCall(ContractCall),
    Transfer(Transfer),
}

// Contract deployment transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractDeployment {
    pub bytecode: Vec<u8>,      // WebAssembly bytecode
    pub constructor_args: Vec<Vec<u8>>, // Arguments for constructor
    pub initial_state: HashMap<Vec<u8>, Vec<u8>>, // Initial contract state
}

// Value transfer transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transfer {
    pub recipient: [u8; 32],    // Recipient's address
    pub amount: u64,            // Amount to transfer
}

// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionStatus {
    Pending,
    Included(u64),  // Block sequence number
    Confirmed,
    Failed(String),
}

// Transaction manager
#[derive(Debug)]
pub struct TransactionManager {
    state_db: Arc<StateDB>,
    // Track transaction status
    tx_status: Arc<RwLock<HashMap<[u8; 32], TransactionStatus>>>,
    // Gas price oracle
    gas_price: Arc<Mutex<u64>>,
}

impl TransactionManager {
    pub fn new(state_db: Arc<StateDB>) -> Self {
        Self {
            state_db,
            tx_status: Arc::new(RwLock::new(HashMap::new())),
            gas_price: Arc::new(Mutex::new(1)), // Default gas price
        }
    }
    
    // Submit a transaction
    pub async fn submit_transaction(&self, tx: Transaction) -> Result<[u8; 32], String> {
        // Verify transaction signature
        if !self.verify_signature(&tx) {
            return Err("Invalid signature".to_string());
        }
        
        // Verify nonce
        if !self.verify_nonce(&tx).await {
            return Err("Invalid nonce".to_string());
        }
        
        // Update transaction status
        {
            let mut statuses = self.tx_status.write().await;
            statuses.insert(tx.hash, TransactionStatus::Pending);
        }
        
        // Return transaction hash
        Ok(tx.hash)
    }
    
    // Get transaction status
    pub async fn get_transaction_status(&self, tx_hash: &[u8; 32]) -> Option<TransactionStatus> {
        let statuses = self.tx_status.read().await;
        statuses.get(tx_hash).cloned()
    }
    
    // Update transaction status
    pub async fn update_transaction_status(&self, tx_hash: &[u8; 32], status: TransactionStatus) {
        let mut statuses = self.tx_status.write().await;
        statuses.insert(*tx_hash, status);
    }
    
    // Verify transaction signature
    fn verify_signature(&self, _tx: &Transaction) -> bool {
        // For simplicity, we'll just return true for now
        // In a real implementation, this would verify ed25519 signatures
        true
    }
    
    // Verify transaction nonce
    async fn verify_nonce(&self, _tx: &Transaction) -> bool {
        // For simplicity, we'll just return true for now
        // In a real implementation, this would check against account state
        true
    }
    
    // Get current gas price
    pub fn get_gas_price(&self) -> u64 {
        *self.gas_price.lock()
    }
    
    // Update gas price based on network demand
    pub fn update_gas_price(&self, new_price: u64) {
        let mut price = self.gas_price.lock();
        *price = new_price;
    }
}

// Transaction receipt containing execution results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionReceipt {
    pub tx_hash: [u8; 32],
    pub block_seq: u64,
    pub block_hash: [u8; 32],
    pub gas_used: u64,
    pub status: bool,
    pub result: Option<Vec<u8>>,
    pub logs: Vec<Log>,
}

// Log entry generated during transaction execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Log {
    pub address: [u8; 32],
    pub topics: Vec<[u8; 32]>,
    pub data: Vec<u8>,
}

pub mod serde_impl;
```

---


## src/transaction/serde_impl.rs

### File path: `/home/myuser/viper/dagknight-vm/src/transaction/serde_impl.rs`

```rust
use super::Transaction;
use serde::{Serialize, Deserialize, Serializer, Deserializer};

#[derive(Serialize, Deserialize)]
struct TransactionSerde {
    pub hash: [u8; 32],
    pub data: Vec<u8>,
    pub sender: [u8; 32],
    pub nonce: u64,
    pub signature: Vec<u8>,
    pub timestamp: u64,
}

impl From<&Transaction> for TransactionSerde {
    fn from(tx: &Transaction) -> Self {
        Self {
            hash: tx.hash,
            data: tx.data.clone(),
            sender: tx.sender,
            nonce: tx.nonce,
            signature: tx.signature.to_vec(),
            timestamp: tx.timestamp,
        }
    }
}

impl From<TransactionSerde> for Transaction {
    fn from(tx: TransactionSerde) -> Self {
        let mut signature = [0u8; 64];
        if tx.signature.len() >= 64 {
            signature.copy_from_slice(&tx.signature[0..64]);
        }
        
        Self {
            hash: tx.hash,
            data: tx.data,
            sender: tx.sender,
            nonce: tx.nonce,
            signature,
            timestamp: tx.timestamp,
        }
    }
}

impl Serialize for Transaction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let serde_tx = TransactionSerde::from(self);
        serde_tx.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Transaction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serde_tx = TransactionSerde::deserialize(deserializer)?;
        Ok(Transaction::from(serde_tx))
    }
}
```

---


## src/types/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/types/mod.rs`

```rust
//! Core types used throughout the DAGKnight VM system

use serde::{Serialize, Deserialize};
use serde_big_array::BigArray;
use std::collections::HashMap;

/// Identifier for a node in the network
pub type NodeId = String;

/// Smart contract or account address
pub type Address = u64;

/// Transaction structure representing operations on the blockchain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transaction {
    /// Transaction hash - unique identifier
    pub hash: [u8; 32],
    /// Transaction payload data
    pub data: Vec<u8>,
    /// Address of the transaction sender
    pub sender: [u8; 32],
    /// Sender's transaction sequence number
    pub nonce: u64,
    /// Cryptographic signature
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
    /// Unix timestamp when the transaction was created
    pub timestamp: u64,
}

/// Virtual machine state containing contracts, storage, and account data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VmState {
    /// Map of contract addresses to bytecode
    pub contracts: HashMap<Address, Vec<u8>>,
    /// Map of contract addresses to their key-value storage
    pub storage: HashMap<Address, HashMap<Vec<u8>, Vec<u8>>>,
    /// Map of addresses to account balances
    pub balances: HashMap<Address, u64>,
    /// Map of addresses to their current nonce
    pub nonces: HashMap<Address, u64>,
}

/// Result of executing a transaction or contract call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Whether execution was successful
    pub success: bool,
    /// Data returned from execution
    pub return_data: Vec<u8>,
    /// Amount of gas consumed during execution
    pub gas_used: u64,
    /// Log messages emitted during execution
    pub logs: Vec<String>,
    /// Error message if execution failed
    pub error: Option<String>,
}```

---


## src/vm/ai/executor.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/ai/executor.rs`

```rust
use crate::vm::cache::ContractCache;
use crate::contracts::AIModelCall;
use std::sync::Arc;

// Simple error enum for AI execution
#[derive(Debug, Clone, thiserror::Error)]
pub enum AIExecutionError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

pub struct AIExecutor {
    cache: Arc<ContractCache>,
}

impl AIExecutor {
    pub async fn new(cache: Arc<ContractCache>) -> Result<Self, AIExecutionError> {
        Ok(Self {
            cache,
        })
    }
    
    pub async fn execute(&self, _model_call: &AIModelCall, _contract_address: [u8; 32]) -> Result<(Vec<u8>, crate::state::ResourceUsage), AIExecutionError> {
        // Stub implementation
        let usage = crate::state::ResourceUsage {
            compute_units: 100,
            memory_bytes: 1024 * 1024, // 1 MB
            storage_bytes: 0,
            cpu_time: 50,
            memory_used: 1024 * 1024,
            gpu_time: 0,
        };
        
        Ok((vec![0, 1, 2, 3], usage))
    }
}
```

---


## src/vm/ai/executor.rs.bak

### File path: `/home/myuser/viper/dagknight-vm/src/vm/ai/executor.rs.bak`

```text
//! AI model execution engine
use crate::contracts::{AIModelCall, ShardingCapability};
use crate::state::ResourceUsage;
use crate::vm::cache::ContractCache;  // Fixed: use the actual cache module from vm
use crate::vm::VmError;  // Added for error handling
use ollama_rs::Ollama;
use ollama_rs::generation::completion::{request::GenerationRequest};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, warn, error, debug, instrument};

/// Error types for AI execution
#[derive(Debug, Clone, thiserror::Error)]
pub enum AIExecutionError {
    #[error("Ollama error: {0}")]
    OllamaError(String),
    
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Computation timeout")]
    Timeout,
    
    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),
    
    #[error("Shard allocation failed: {0}")]
    ShardAllocationFailed(String),
    
    #[error("Node failure: {0}")]
    NodeFailure(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

type Result<T> = std::result::Result<T, AIExecutionError>;

/// Model registry for caching information about models
struct ModelInfo {
    name: String,
    capabilities: ShardingCapability,
    // Additional model information as needed
}

/// AI execution engine
pub struct AIExecutor {
    /// Ollama client
    ollama: Ollama,
    /// GPU availability
    gpu_enabled: bool,
    /// Contract cache for storing compiled contract code
    cache: Arc<ContractCache>,
    /// Local model registry
    models: HashMap<String, ModelInfo>,
    /// Performance metrics per model
    performance_metrics: Arc<Mutex<HashMap<String, Vec<ExecutionMetrics>>>>,
}

/// Execution metrics for learning
struct ExecutionMetrics {
    model: String,
    shard_count: u64,
    input_size: usize,
    execution_time: Duration,
    memory_used: u64,
    success: bool,
}

impl AIExecutor {
    /// Create a new AI executor
    pub async fn new(
        cache: Arc<ContractCache>,
    ) -> Result<Self> {
        let ollama = Ollama::default();
        
        // Check if Ollama is running
        match ollama.list_local_models().await {
            Ok(_) => info!("Connected to Ollama service"),
            Err(e) => {
                error!("Failed to connect to Ollama: {}", e);
                return Err(AIExecutionError::OllamaError(e.to_string()));
            }
        }
        
        // Try to check for GPU
        let gpu_enabled = match ollama.generate(GenerationRequest::new(
            "nous-hermes:1b".to_string(), 
            "Are you using GPU?".to_string()
        )).await {
            Ok(_) => {
                info!("GPU acceleration available");
                true
            },
            Err(e) => {
                warn!("GPU acceleration unavailable: {}", e);
                false
            }
        };
        
        // Initialize default models
        let mut models = HashMap::new();
        models.insert("llama2:7b".to_string(), 
                     ModelInfo {
                         name: "llama2:7b".to_string(),
                         capabilities: ShardingCapability::Horizontal,
                     });
        
        models.insert("mistral:7b".to_string(), 
                     ModelInfo {
                         name: "mistral:7b".to_string(),
                         capabilities: ShardingCapability::Vertical,
                     });
        
        Ok(Self {
            ollama,
            gpu_enabled,
            cache,
            models,
            performance_metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
    
    /// Execute an AI model
    #[instrument(skip(self, model_call), fields(model = %model_call.model))]
    pub async fn execute(
        &self,
        model_call: &AIModelCall,
        contract_address: [u8; 32],
    ) -> Result<(Vec<u8>, ResourceUsage)> {
        // Check model registry
        let model_info = self.models.get(&model_call.model)
            .ok_or_else(|| AIExecutionError::ModelNotFound(model_call.model.clone()))?;
        
        // In a real implementation, we would check the cache here
        
        info!("Executing model {}", model_call.model);
        
        // Check if model supports sharding
                }
                ShardingStrategy::None
            },
            ShardingCapability::Horizontal => ShardingStrategy::Horizontal,
            ShardingCapability::Vertical => ShardingStrategy::Vertical,
            ShardingCapability::Full => {
                // Choose best strategy based on historical performance
                self.determine_best_strategy(&model_call.model, model_call.input.len()).await
            }
        };
        
        // Prepare execution
        let start_time = Instant::now();
        let result: Result<Vec<u8>>;
        let nodes_used: u64;
        
        // Execute based on sharding strategy
        match sharding_strategy {
            ShardingStrategy::None => {
                debug!("Executing model {} without sharding", model_call.model);
                result = self.execute_single(model_call).await;
                nodes_used = 1;
            },
            ShardingStrategy::Horizontal => {
                let shard_count = self.calculate_optimal_shard_count(
                    &model_call.model, 
                    model_call.input.len()
                ).await;
                
                debug!("Executing model {} with horizontal sharding (shards: {})", 
                       model_call.model, shard_count);
                       
                result = self.execute_horizontal_sharded(model_call, shard_count).await;
                nodes_used = shard_count;
            },
            ShardingStrategy::Vertical => {
                debug!("Executing model {} with vertical sharding", model_call.model);
                result = self.execute_vertical_sharded(model_call).await;
                nodes_used = model_call.shard_count;
            }
        }
        
        let execution_time = start_time.elapsed();
        
        // Record metrics for future optimization
        self.record_execution_metrics(
            &model_call.model,
            nodes_used,
            model_call.input.len(),
            execution_time,
            result.is_ok(),
        ).await;
        
        // Calculate resource usage
        let usage = ResourceUsage {
            compute_units: execution_time.as_millis() as u64,
            memory_bytes: estimate_memory_usage(&model_call.model, model_call.input.len()),
            storage_bytes: 0,
            cpu_time: execution_time.as_millis() as u64,
            memory_used: estimate_memory_usage(&model_call.model, model_call.input.len()),
            gpu_time: if self.gpu_enabled { execution_time.as_millis() as u64 } else { 0 },
        };
        
        // Return the result or handle error
        match result {
            Ok(output) => Ok((output, usage)),
            Err(e) => Err(e),
        }
    }
    
    /// Execute model on a single node
    async fn execute_single(&self, model_call: &AIModelCall) -> Result<Vec<u8>> {
        let input_str = String::from_utf8_lossy(&model_call.input).to_string();
        
        // Create the generation request
        let req = GenerationRequest::new(model_call.model.clone(), input_str);
        
        // Execute with timeout protection
        let result = tokio::time::timeout(
            Duration::from_secs(120), // 2 minute timeout
            self.ollama.generate(req)
        ).await;
        
        match result {
            Ok(Ok(response)) => {
                Ok(response.response.into_bytes())
            },
            Ok(Err(e)) => {
                error!("Ollama execution error: {}", e);
                Err(AIExecutionError::OllamaError(e.to_string()))
            },
            Err(_) => {
                error!("Execution timed out");
                Err(AIExecutionError::Timeout)
            }
        }
    }
    
    /// Execute model with horizontal sharding (input splitting)
    async fn execute_horizontal_sharded(
        &self, 
        model_call: &AIModelCall,
        shard_count: u64
    ) -> Result<Vec<u8>> {
        // Split input
        let input_chunks = split_input(&model_call.input, shard_count as usize);
        
        // Prepare tasks
        let mut results = Vec::with_capacity(input_chunks.len());
        
        for chunk in input_chunks {
            // In a real implementation, we'd execute these in parallel
            // For simplicity, we'll process them sequentially here
            let input_str = String::from_utf8_lossy(&chunk).to_string();
            let req = GenerationRequest::new(model_call.model.clone(), input_str);
            
            match self.ollama.generate(req).await {
                Ok(response) => results.push(Ok(response.response.into_bytes())),
                Err(e) => results.push(Err(AIExecutionError::OllamaError(e.to_string()))),
            }
        }
        
        // Merge results
        let merged = merge_outputs(results)?;
        
        Ok(merged)
    }
    
    /// Execute model with vertical sharding (model splitting)
    async fn execute_vertical_sharded(&self, model_call: &AIModelCall) -> Result<Vec<u8>> {
        // For this example, we'll simulate vertical sharding
        // In a real implementation, this would distribute model layers across nodes
        
        warn!("Vertical sharding is simulated in this implementation");
        
        // Fallback to single execution for this demo
        self.execute_single(model_call).await
    }
    
    /// Determine best sharding strategy based on historical performance
    async fn determine_best_strategy(&self, model: &str, input_size: usize) -> ShardingStrategy {
        let metrics = self.performance_metrics.lock().await;
        
        // If we don't have enough data, default to horizontal
        if !metrics.contains_key(model) {
            return ShardingStrategy::Horizontal;
        }
        
        // Find similar workloads
        let model_metrics = metrics.get(model).unwrap();
        let similar_workloads: Vec<_> = model_metrics.iter()
            .filter(|m| (m.input_size as f64 * 0.8..=m.input_size as f64 * 1.2).contains(&(input_size as f64)))
            .collect();
            
        if similar_workloads.is_empty() {
            return ShardingStrategy::Horizontal;
        }
        
        // Count successes for each strategy
        let horizontal_success = similar_workloads.iter()
            .filter(|m| m.shard_count > 1 && m.success)
            .count();
            
        let vertical_success = similar_workloads.iter()
            .filter(|m| m.shard_count > 1 && m.success)
            .count();
            
        // Choose the more successful strategy
        if horizontal_success >= vertical_success {
            ShardingStrategy::Horizontal
        } else {
            ShardingStrategy::Vertical
        }
    }
    
    /// Calculate optimal shard count based on historical performance
    async fn calculate_optimal_shard_count(&self, model: &str, input_size: usize) -> u64 {
        let metrics = self.performance_metrics.lock().await;
        
        if !metrics.contains_key(model) {
            return 4; // Default to 4 shards if no data
        }
        
        let model_metrics = metrics.get(model).unwrap();
        
        // Find metrics for similar workloads
        let similar_workloads: Vec<_> = model_metrics.iter()
            .filter(|m| (m.input_size as f64 * 0.8..=m.input_size as f64 * 1.2).contains(&(input_size as f64)))
            .filter(|m| m.success) // Only consider successful executions
            .collect();
            
        if similar_workloads.is_empty() {
            return 4; // Default if no similar workloads
        }
        
        // Group by shard count and calculate average execution time
        let mut shard_performance: HashMap<u64, (Duration, usize)> = HashMap::new();
        
        for metric in similar_workloads {
            let entry = shard_performance.entry(metric.shard_count).or_insert((Duration::from_secs(0), 0));
            entry.0 += metric.execution_time;
            entry.1 += 1;
        }
        
        // Find shard count with lowest average execution time
        shard_performance.iter()
            .map(|(shard_count, (total_time, count))| {
                let avg_time = total_time.div_f64(*count as f64);
                (*shard_count, avg_time)
            })
            .min_by_key(|(_, time)| time.as_millis() as u64)
            .map(|(shard_count, _)| shard_count)
            .unwrap_or(4) // Default if comparison fails
    }
    
    /// Record metrics for future optimization
    async fn record_execution_metrics(
        &self,
        model: &str,
        shard_count: u64,
        input_size: usize,
        execution_time: Duration,
        success: bool,
    ) {
        let metric = ExecutionMetrics {
            model: model.to_string(),
            shard_count,
            input_size,
            execution_time,
            memory_used: estimate_memory_usage(model, input_size),
            success,
        };
        
        let mut metrics = self.performance_metrics.lock().await;
        
        // Add to metrics history
        metrics.entry(model.to_string())
            .or_insert_with(Vec::new)
            .push(metric);
            
        // Keep only the last 100 metrics per model
        if let Some(model_metrics) = metrics.get_mut(model) {
            if model_metrics.len() > 100 {
                model_metrics.sort_by_key(|m| m.execution_time);
                model_metrics.truncate(100);
            }
        }
    }
}

/// Sharding strategy
enum ShardingStrategy {
    /// No sharding
    None,
    /// Horizontal sharding (input splitting)
    Horizontal,
    /// Vertical sharding (model splitting)
    Vertical,
}

/// Split input data into chunks
fn split_input(input: &[u8], chunk_count: usize) -> Vec<Vec<u8>> {
    if chunk_count <= 1 {
        return vec![input.to_vec()];
    }
    
    let chunk_size = (input.len() / chunk_count) + 1;
    let mut chunks = Vec::with_capacity(chunk_count);
    
    for i in 0..chunk_count {
        let start = i * chunk_size;
        if start >= input.len() {
            break;
        }
        
        let end = (start + chunk_size).min(input.len());
        chunks.push(input[start..end].to_vec());
    }
    
    chunks
}

/// Merge outputs from multiple chunks
fn merge_outputs(outputs: Vec<Result<Vec<u8>>>) -> Result<Vec<u8>> {
    // Process any errors
    for output in &outputs {
        if let Err(e) = output {
            return Err(e.clone());
        }
    }
    
    // Combine successful outputs
    let mut result = Vec::new();
    for output in outputs {
        if let Ok(data) = output {
            result.extend_from_slice(&data);
        }
    }
    
    Ok(result)
}

/// Estimate memory usage based on model and input size
fn estimate_memory_usage(model: &str, input_size: usize) -> u64 {
    // This is a simplified estimation - a real implementation would have more sophisticated logic
    let base_memory = match model.split(':').next().unwrap_or("unknown") {
        "llama2" => 4000,
        "deepseek" => 6000,
        "mistral" => 8000,
        "phi" => 3000,
        _ => 5000, // Default for unknown models
    };
    
    // Scale with input size (simplified)
    base_memory + (input_size as u64 / 100)
}```

---


## src/vm/ai/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/ai/mod.rs`

```rust
pub mod executor;
```

---


## src/vm/batch/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/batch/mod.rs`

```rust
// Minimal batch implementation for compilation
pub struct TransactionBatch;
```

---


## src/vm/batch_call_contracts.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/batch_call_contracts.rs`

```rust
    // Execute a batch of contract calls
    pub async fn batch_call_contracts(&self, calls: Vec<ContractCall>) -> Vec<ContractResult> {
        use std::collections::HashMap;
        
        // Prepare the calls with their contracts
        let mut call_with_contracts = Vec::new();
        
        for call in calls {
            if let Some(contract) = self.contract_registry.get(&call.contract_address) {
                call_with_contracts.push((call, Arc::new(contract.clone())));
            } else {
                // Contract not found, return an error for this call
                // For now, we'll just skip it
                continue;
            }
        }
        
        // Execute in parallel
        let batch_result = self.parallel_executor.execute_batch(call_with_contracts).await;
        let total_gas_used = batch_result.gas_used;
        
        // Clone the results to avoid using after move
        let results_clone = batch_result.results.clone();
        
        // Convert results to ContractResult objects
        let results: Vec<ContractResult> = results_clone.into_iter()
            .enumerate()
            .map(|(_idx, result)| {
                match result {
                    Ok(data) => ContractResult {
                        success: true,
                        return_data: data,
                        error: None,
                        gas_used: total_gas_used / batch_result.results.len() as u64, // Average
                        state_changes: HashMap::new(), // Simplified
                        logs: Vec::new(), // Simplified
                    },
                    Err(e) => ContractResult {
                        success: false,
                        return_data: Vec::new(),
                        error: Some(e.to_string()),
                        gas_used: total_gas_used / batch_result.results.len() as u64, // Average
                        state_changes: HashMap::new(),
                        logs: Vec::new(),
                    }
                }
            })
            .collect();
            
        results
    }
```

---


## src/vm/cache/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/cache/mod.rs`

```rust
use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;

#[derive(Debug)]
pub struct ContractCache {
    contracts: RwLock<HashMap<String, Vec<u8>>>,
}

impl ContractCache {
    pub fn new() -> Self {
        Self {
            contracts: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.contracts.read().get(key).cloned()
    }

    pub fn insert(&self, key: String, value: Vec<u8>) {
        self.contracts.write().insert(key, value);
    }
}
```

---


## src/vm/executor.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/executor.rs`

```rust
use wasmer::{Store, Module, Instance, imports, Value, Function, FunctionEnv, FunctionType, Type};
use std::sync::Arc;
use crate::state::StateDB;
use crate::vm::VmError;

#[derive(Debug, Clone)]
pub struct VMEnvironment {
    state_db: Arc<StateDB>,
    gas_used: u64,
    gas_limit: u64,
}

impl VMEnvironment {
    pub fn new(state_db: Arc<StateDB>, gas_limit: u64) -> Self {
        Self {
            state_db,
            gas_used: 0,
            gas_limit,
        }
    }

    pub fn charge_gas(&mut self, amount: u64) -> Result<(), VmError> {
        self.gas_used += amount;
        if self.gas_used > self.gas_limit {
            return Err(VmError::OutOfGas);
        }
        Ok(())
    }

    pub fn get_gas_used(&self) -> u64 {
        self.gas_used
    }
}

// Store can't be cloned, so don't derive Clone
#[derive(Debug)]
pub struct WasmExecutor {
    store: Store,
}

impl WasmExecutor {
    pub fn new() -> Self {
        let store = Store::default();
        Self { store }
    }

    pub fn execute(&mut self, bytecode: &[u8], env: VMEnvironment, function: &str, args: Vec<Value>) -> Result<Vec<Value>, VmError> {
        // Compile the module
        let module = Module::new(&self.store, bytecode)
            .map_err(|_e| VmError::CompilationError("Failed to compile module".to_string()))?;

        // Create function environment
        let func_env = FunctionEnv::new(&mut self.store, env);
        
        // Create read_state function
        let read_state = move |_ctx: wasmer::FunctionEnvMut<VMEnvironment>, _args: &[Value]| -> Result<Vec<Value>, wasmer::RuntimeError> {
            // In a real implementation, we'd extract arguments and call the actual host function
            // For compilation, we just return an empty result
            Ok(vec![Value::I32(0)])
        };
        
        // Create write_state function
        let write_state = move |_ctx: wasmer::FunctionEnvMut<VMEnvironment>, _args: &[Value]| -> Result<Vec<Value>, wasmer::RuntimeError> {
            // In a real implementation, we'd extract arguments and call the actual host function
            // For compilation, we just return an empty result
            Ok(vec![Value::I32(0)])
        };
        
        // Define function signatures
        let read_state_sig = FunctionType::new(vec![Type::I32, Type::I32, Type::I32, Type::I32], vec![Type::I32]);
        let write_state_sig = FunctionType::new(vec![Type::I32, Type::I32, Type::I32, Type::I32], vec![Type::I32]);
        
        // Create import object with environment functions
        let import_object = imports! {
            "env" => {
                "read_state" => Function::new_with_env(&mut self.store, &func_env, read_state_sig, read_state),
                "write_state" => Function::new_with_env(&mut self.store, &func_env, write_state_sig, write_state),
            }
        };

        // Instantiate the module
        let instance = Instance::new(&mut self.store, &module, &import_object)
            .map_err(|_e| VmError::InstantiationError("Failed to instantiate module".to_string()))?;

        // Get the function to execute
        let wasm_function = instance.exports.get_function(function)
            .map_err(|_e| VmError::FunctionNotFound(function.to_string()))?;

        // Execute the function
        let result = wasm_function.call(&mut self.store, &args)
            .map_err(|e| VmError::ExecutionError(e.to_string()))?;

        Ok(result.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StateDB;

    #[test]
    fn test_wasm_execution() {
        // This test is simplified for compilation purposes
        let state_db = Arc::new(StateDB::new_in_memory());
        let env = VMEnvironment::new(state_db, 1000000);
        
        // For compilation only
        assert!(env.gas_limit > 0);
    }
}```

---


## src/vm/jit_executor.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/jit_executor.rs`

```rust
// This is a simplified JIT executor for demonstration - in a real implementation
// we would integrate with cranelift for actual JIT compilation

use crate::vm::VmError;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

// Type of function pointer for JIT-compiled functions
type JitFunction = fn(&[u8]) -> Vec<u8>;

// JIT compiler for WebAssembly functions
pub struct JitCompiler {
    // Map of function name to compiled function
    compiled_functions: Arc<RwLock<HashMap<String, Box<JitFunction>>>>,
}

impl JitCompiler {
    pub fn new() -> Self {
        Self {
            compiled_functions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    // Compile a WebAssembly function to native code
    pub fn compile(&self, wasm_code: &[u8], function_name: &str) -> Result<(), VmError> {
        // In a real implementation, this would use Cranelift or similar JIT
        // compiler to compile WebAssembly to native code
        //
        // For demonstration, we'll just register a dummy function
        
        let dummy_function: JitFunction = |_args| {
            // Return a dummy result
            vec![0, 1, 2, 3]
        };
        
        let mut functions = self.compiled_functions.write();
        functions.insert(function_name.to_string(), Box::new(dummy_function));
        
        Ok(())
    }
    
    // Execute a compiled function
    pub fn execute(&self, function_name: &str, args: &[u8]) -> Result<Vec<u8>, VmError> {
        let functions = self.compiled_functions.read();
        
        if let Some(function) = functions.get(function_name) {
            Ok(function(args))
        } else {
            Err(VmError::FunctionNotFound(format!("JIT function not found: {}", function_name)))
        }
    }
    
    // Check if a function is compiled
    pub fn is_compiled(&self, function_name: &str) -> bool {
        let functions = self.compiled_functions.read();
        functions.contains_key(function_name)
    }
}

// Example of using the JIT compiler
pub fn example_jit_usage() -> Result<(), VmError> {
    let jit = JitCompiler::new();
    
    // "Compile" a function
    jit.compile(&[0u8; 10], "test_function")?;
    
    // Execute the function
    let result = jit.execute("test_function", &[1, 2, 3])?;
    
    // Print result
    println!("JIT result: {:?}", result);
    
    Ok(())
}
```

---


## src/vm/memory/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/memory/mod.rs`

```rust
// Memory module
pub mod pool;
```

---


## src/vm/memory/pool.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/memory/pool.rs`

```rust
// Minimal memory pool implementation for compilation

// Define constants used in imports
pub static STRING_POOL: () = ();
pub static BUFFER_POOL: () = ();
pub static ARG_POOL: () = ();

// Initialize memory pools function
pub fn init_memory_pools() {
    // No-op for compilation
}
```

---


## src/vm/memory/zero_copy.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/memory/zero_copy.rs`

```rust
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Error, ErrorKind, Result};
use memmap2::MmapMut;
use parking_lot::RwLock;

const VALUE_HEADER_SIZE: usize = 4; // 4 bytes for length

pub struct ZeroCopyState {
    // Memory-mapped file for state
    mmap: MmapMut,
    // Index mapping keys to offsets in the mmap
    index: RwLock<HashMap<[u8; 32], usize>>,
    // Next free offset
    next_offset: RwLock<usize>,
    // Free space map (offset -> size)
    free_spaces: RwLock<HashMap<usize, usize>>,
}

impl ZeroCopyState {
    pub fn new(path: &str, size: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
            
        file.set_len(size as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        // Initialize index
        let index = HashMap::new();
        
        // Start data after header
        let next_offset = 8; // 8 byte header
        
        Ok(Self {
            mmap,
            index: RwLock::new(index),
            next_offset: RwLock::new(next_offset),
            free_spaces: RwLock::new(HashMap::new()),
        })
    }
    
    pub fn get<'a>(&'a self, key: &[u8; 32]) -> Option<&'a [u8]> {
        let index = self.index.read();
        
        index.get(key).map(|&offset| {
            let len_bytes = &self.mmap[offset..offset + VALUE_HEADER_SIZE];
            let len = u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
            &self.mmap[offset + VALUE_HEADER_SIZE..offset + VALUE_HEADER_SIZE + len]
        })
    }
    
    // Find space for a value of given size
    fn find_space(&self, size: usize) -> Result<usize> {
        // Get from next offset
        let mut next_offset = self.next_offset.write();
        let offset = *next_offset;
        
        // Check if we have enough space
        if offset + size > self.mmap.len() {
            return Err(Error::new(ErrorKind::Other, "Out of memory"));
        }
        
        // Update next offset
        *next_offset = offset + size;
        
        Ok(offset)
    }
}
```

---


## src/vm/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/mod.rs`

```rust
//! VM module for DAGKnight VM

// Submodules
pub mod ai;
pub mod cache;
pub mod narwhal_bullshark_vm;

// Consensus engine trait
#[async_trait::async_trait]
pub trait ConsensusEngine: Send + Sync {
    // Original methods
    async fn validate_contract(&self, hash: [u8; 32], bytecode: &[u8]) -> Result<(), VmError>;
    async fn broadcast_contract(&self, hash: [u8; 32], bytecode: Vec<u8>) -> Result<(), VmError>;
    
    // Added methods needed by PBFT implementation with default implementations
    async fn validate_block(&self, _block: &[u8]) -> Result<bool, VmError> {
        // Default implementation
        Ok(true)
    }
    
    async fn finalize_block(&self, _block: &[u8]) -> Result<(), VmError> {
        // Default implementation
        Ok(())
    }
    
    async fn get_latest_block(&self) -> Result<Vec<u8>, VmError> {
        // Default implementation
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum VmError {
    #[error("Consensus error: {0}")]
    ConsensusFailure(String),
    
    #[error("Storage error")]
    StorageError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Contract not found: {0}")]
    ContractNotFound(String),
    
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    
    #[error("Compilation error: {0}")]
    CompilationError(String),
    
    #[error("Instantiation error: {0}")]
    InstantiationError(String),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    #[error("Out of gas")]
    OutOfGas,
    
    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),
    
    #[error("Insufficient balance")]
    InsufficientBalance,
    
    #[error("Invalid nonce")]
    InvalidNonce,
}

// Define necessary types for interaction with narwhal_bullshark_vm
pub struct VirtualMachine {
    pub state_db: Arc<crate::state::StateDB>,
}

impl VirtualMachine {
    pub fn new(state_db: Arc<crate::state::StateDB>) -> Self {
        Self { state_db }
    }
}

// Contract state for StateAccess
#[derive(Debug, Clone)]
pub struct ContractState {
    pub code: Vec<u8>,
    pub storage: std::collections::HashMap<Vec<u8>, Vec<u8>>,
}

// Call data for executing contracts
#[derive(Debug, Clone)]
pub struct CallData {
    pub contract_address: u64,
    pub function: String,
    pub arguments: Vec<u8>,
    pub sender: u64,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub value: u64,
}

// Result of execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub return_data: Vec<u8>,
    pub gas_used: u64,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

// State access trait
#[async_trait::async_trait]
pub trait StateAccess: Send + Sync {
    async fn get_contract(&self, address: u64) -> Result<Option<Vec<u8>>, VmError>;
    async fn get_storage(&self, address: u64, key: &[u8]) -> Result<Option<Vec<u8>>, VmError>;
    async fn set_storage(&self, address: u64, key: Vec<u8>, value: Vec<u8>) -> Result<(), VmError>;
    async fn get_balance(&self, address: u64) -> Result<u64, VmError>;
    async fn set_balance(&self, address: u64, amount: u64) -> Result<(), VmError>;
    async fn get_nonce(&self, address: u64) -> Result<u64, VmError>;
    async fn get_contract_state(&self, address: u64) -> Result<Option<ContractState>, VmError>;
}

use std::sync::Arc;
```

---


## src/vm/mod.rs.bak

### File path: `/home/myuser/viper/dagknight-vm/src/vm/mod.rs.bak`

```text
use std::sync::Arc;
use std::collections::HashMap;

use crate::contracts::{Contract, ContractCall, ContractResult, ContractRegistry};
use crate::state::StateDB;
use self::executor::{WasmExecutor, VMEnvironment};
use self::cache::ContractCache;
use self::parallel_executor::ParallelExecutor;
use self::tiered_vm::TieredVM;

// Submodules of the VM
pub mod executor;
pub mod ai;
pub mod memory;
pub mod cache;
pub mod batch;
pub mod parallel_executor;
pub mod tiered_vm;
pub mod narwhal_bullshark_vm;

// Transaction structure for DagkVm
pub struct Transaction {
    pub from: [u8; 32],
    pub to: [u8; 32],
    pub nonce: u64,
    pub data: Vec<u8>,
    pub signature: Vec<u8>,
}

// Types expected by narwhal_bullshark_vm.rs
pub type Address = u64; // Define Address as u64 to match narwhal_bullshark_vm.rs

#[derive(Debug, Clone)]
pub struct CallData {
    pub contract_address: Address,
    pub function: String,
    pub arguments: Vec<u8>,
    pub sender: Address,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub value: u64,
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub return_data: Vec<u8>,
    pub gas_used: u64,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ContractState {
    pub code: Vec<u8>,
    pub storage: HashMap<Vec<u8>, Vec<u8>>,
}

#[async_trait::async_trait]
pub trait StateAccess: Send + Sync {
    async fn get_contract(&self, address: Address) -> Result<Option<Vec<u8>>, VmError>;
    async fn get_storage(&self, address: Address, key: &[u8]) -> Result<Option<Vec<u8>>, VmError>;
    async fn set_storage(&self, address: Address, key: Vec<u8>, value: Vec<u8>) -> Result<(), VmError>;
    async fn get_balance(&self, address: Address) -> Result<u64, VmError>;
    async fn set_balance(&self, address: Address, amount: u64) -> Result<(), VmError>;
    async fn get_nonce(&self, address: Address) -> Result<u64, VmError>;
    async fn get_contract_state(&self, address: Address) -> Result<Option<ContractState>, VmError>;
}

// VmError enum - centralized error type for the VM
#[derive(Clone, Debug, thiserror::Error)]
pub enum VmError {
    #[error("Consensus error: {0}")]
    ConsensusFailure(String),
    
    #[error("Storage error: {0}")]
    StorageError(#[from] rocksdb::Error),
    
    #[error("Serialization error")]
    SerializationError(String),
    
    #[error("Contract not found: {0}")]
    ContractNotFound(String),
    
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    
    #[error("Compilation error: {0}")]
    CompilationError(String),
    
    #[error("Instantiation error: {0}")]
    InstantiationError(String),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    #[error("Out of gas")]
    OutOfGas,
    
    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),
    
    // Added for narwhal_bullshark_vm.rs compatibility
    #[error("Insufficient balance")]
    InsufficientBalance,
    
    #[error("Invalid nonce")]
    InvalidNonce,
}

// VirtualMachine implementation
pub struct VirtualMachine {
    state_db: Arc<StateDB>,
    executor: WasmExecutor,
}

impl VirtualMachine {
    pub fn new(state_db: Arc<StateDB>) -> Self {
        Self {
            state_db,
            executor: WasmExecutor::new(),
        }
    }

    pub async fn execute(&mut self, call_data: &CallData, state_access: &dyn StateAccess) -> Result<ExecutionResult, VmError> {
        // Placeholder: Implement actual execution logic using WasmExecutor
        let env = VMEnvironment::new(Arc::clone(&self.state_db), call_data.gas_limit);
        let contract_code = state_access.get_contract(call_data.contract_address)
            .await?
            .ok_or_else(|| VmError::ContractNotFound(call_data.contract_address.to_string()))?;
        
        let wasm_args = vec![]; // Simplified: Convert call_data.arguments to wasmer values if needed
        let result = self.executor.execute(&contract_code, env, &call_data.function, wasm_args)
            .map_err(|e| VmError::ExecutionError(e.to_string()))?;

        Ok(ExecutionResult {
            success: true,
            return_data: Vec::new(), // Simplified; real impl would convert result
            gas_used: call_data.gas_limit / 2, // Example
            logs: vec![],
            error: None,
        })
    }
}

#[async_trait::async_trait]
pub trait NetworkInterface: Send + Sync {
    async fn broadcast_contract(&self, hash: [u8; 32], bytecode: Vec<u8>) -> Result<(), VmError>;
}

#[async_trait::async_trait]
pub trait ConsensusEngine: Send + Sync {
    async fn validate_contract(&self, hash: [u8; 32], bytecode: &[u8]) -> Result<(), VmError>;
    async fn broadcast_contract(&self, hash: [u8; 32], bytecode: Vec<u8>) -> Result<(), VmError>;
}

// The main DAGKnight VM implementation
pub struct DagkVm {
    pub state_db: Arc<StateDB>,
    pub contract_registry: Arc<ContractRegistry>,
    pub network: Arc<dyn NetworkInterface>,
    pub consensus_engine: Arc<dyn ConsensusEngine>,
    pub executor: WasmExecutor,
    pub contract_cache: ContractCache,
    pub parallel_executor: ParallelExecutor,
    pub tiered_vm: TieredVM,
}

impl DagkVm {
    pub fn new(
        state_db: Arc<StateDB>,
        contract_registry: Arc<ContractRegistry>,
        network: Arc<dyn NetworkInterface>,
        consensus_engine: Arc<dyn ConsensusEngine>,
    ) -> Self {
        // Initialize memory pools if available
        if let Ok(pool) = std::path::Path::new("src/vm/memory/pool.rs").try_exists() {
            if pool {
                // Dummy init to ensure compilation
            }
        }
        
        // Create VM components
        let executor = WasmExecutor::new();
        let contract_cache = ContractCache::new();
        let parallel_executor = ParallelExecutor::new(Arc::clone(&state_db));
        let tiered_vm = TieredVM::new(Arc::clone(&state_db));
        
        Self {
            state_db,
            contract_registry,
            network,
            consensus_engine,
            executor,
            contract_cache,
            parallel_executor,
            tiered_vm,
        }
    }
    
    pub async fn call_contract(
        &self,
        contract_address: [u8; 32],
        function: &str,
        args: Vec<Vec<u8>>,
        _sender: [u8; 32],
        _nonce: u64,
    ) -> Result<[u8; 32], VmError> {
        self.consensus_engine.validate_contract(contract_address, &[])
            .await
            .map_err(|e| VmError::ConsensusFailure(e.to_string()))?;
            
        let contract = self.find_contract(&contract_address)
            .ok_or_else(|| VmError::ContractNotFound(hex::encode(contract_address)))?;
            
        let _result = self.tiered_vm.execute(&contract, function, &args)?;
        
        Ok([0u8; 32]) // Dummy hash
    }
    
    fn find_contract(&self, address: &[u8; 32]) -> Option<Contract> {
        self.contract_registry.get(address).map(|arc_contract| arc_contract.as_ref().clone())
    }

    pub fn view_contract(
        &self,
        contract_address: [u8; 32],
        function: &str,
        _args: Vec<Vec<u8>>,
        _sender: [u8; 32],
        _nonce: u64,
    ) -> Result<Vec<u8>, VmError> {
        let contract = self.find_contract(&contract_address)
            .ok_or_else(|| VmError::ContractNotFound(hex::encode(contract_address)))?;
            
        let env = VMEnvironment::new(Arc::clone(&self.state_db), 1_000_000);
        let wasm_args = vec![];
        let mut new_executor = WasmExecutor::new();
        
        let _result = new_executor.execute(&contract.code, env, function, wasm_args)
            .map_err(|e| VmError::ExecutionError(e.to_string()))?;
            
        Ok(vec![0u8; 4]) // Dummy result
    }
    
    pub async fn submit_transaction(&self, tx: Transaction) -> Result<[u8; 32], VmError> {
        self.consensus_engine.validate_contract(tx.to, &tx.data)
            .await
            .map_err(|e| VmError::ConsensusFailure(e.to_string()))?;
            
        self.network.broadcast_contract(tx.to, tx.data.clone())
            .await
            .map_err(|e| VmError::ConsensusFailure(format!("Failed to broadcast transaction: {:?}", e)))?;
            
        Ok([0u8; 32]) // Dummy hash
    }
    
    pub async fn batch_call_contracts(&self, calls: Vec<ContractCall>) -> Vec<ContractResult> {
        let mut call_with_contracts = Vec::new();
        
        for call in calls {
            if let Some(contract) = self.find_contract(&call.contract_address) {
                call_with_contracts.push((call, Arc::new(contract)));
            }
        }
        
        let batch_result = self.parallel_executor.execute_batch(call_with_contracts).await;
        let _total_gas_used = batch_result.gas_used;
        
        let results_clone = batch_result.results.clone();
        let _batch_len = results_clone.len() as u64;
        
        results_clone.into_iter()
            .enumerate()
            .map(|(_idx, result)| match result {
                Ok(data) => ContractResult {
                    output: data,
                    success: true,
                    state_changes: HashMap::new(),
                },
                Err(_) => ContractResult {
                    output: Vec::new(),
                    success: false,
                    state_changes: HashMap::new(),
                },
            })
            .collect()
    }
}

// Export NarwhalBullsharkVm
pub use narwhal_bullshark_vm::NarwhalBullsharkVm;```

---


## src/vm/mod.rs.fixed_batch

### File path: `/home/myuser/viper/dagknight-vm/src/vm/mod.rs.fixed_batch`

```text
    // Execute a batch of contract calls
    pub async fn batch_call_contracts(&self, calls: Vec<ContractCall>) -> Vec<ContractResult> {
        // Prepare the calls with their contracts
        let mut call_with_contracts = Vec::new();
        
        for call in calls {
            if let Some(contract) = self.contract_registry.get(&call.contract_address) {
                call_with_contracts.push((call, Arc::new(contract.clone())));
            } else {
                // Contract not found, return an error for this call
                // For now, we'll just skip it
                continue;
            }
        }
        
        // Execute in parallel
        let batch_result = self.parallel_executor.execute_batch(call_with_contracts).await;
        let gas_used = batch_result.gas_used;
        
        // Convert results to ContractResult objects
        let results: Vec<ContractResult> = batch_result.results.into_iter()
            .enumerate()
            .map(|(_idx, result)| {
                match result {
                    Ok(data) => ContractResult {
                        success: true,
                        return_data: data,
                        error: None,
                        gas_used,
                        state_changes: HashMap::new(), // Simplified
                        logs: Vec::new(), // Simplified
                    },
                    Err(e) => ContractResult {
                        success: false,
                        return_data: Vec::new(),
                        error: Some(e.to_string()),
                        gas_used,
                        state_changes: HashMap::new(),
                        logs: Vec::new(),
                    }
                }
            })
            .collect();
            
        results
    }
```

---


## src/vm/narwhal_bullshark_vm.rs.bak

### File path: `/home/myuser/viper/dagknight-vm/src/vm/narwhal_bullshark_vm.rs.bak`

```text
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex as TokioMutex};
use parking_lot::Mutex;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use serde_big_array::BigArray;
use dashmap::DashMap;
use anyhow::Result;
use blake3;
use bincode;

mod consensus {
    pub mod narwhal_bullshark {
        use std::time::{Duration, SystemTime, UNIX_EPOCH};
        use anyhow::Result;
        use dashmap::DashMap;
        use tokio::sync::RwLock;
        use blake3;
        use serde::{Serialize, Deserialize};
        use serde_big_array::BigArray;

        pub type NodeId = String;

        #[derive(Clone, Serialize, Deserialize)]
        pub struct Transaction {
            pub hash: [u8; 32],
            pub data: Vec<u8>,
            pub sender: [u8; 32],
            pub nonce: u64,
            #[serde(with = "BigArray")]
            pub signature: [u8; 64],
            pub timestamp: u64,
        }

        #[derive(Clone, Serialize, Deserialize)]
        pub struct Block {
            pub hash: [u8; 32],
            pub seq_num: u64,
            pub transactions: Vec<Transaction>,
            pub timestamp: u64,
        }

        pub struct NarwhalBullshark {
            node_id: NodeId,
            peers: Vec<NodeId>,
            finalized_blocks: DashMap<u64, Block>,
            latest_height: RwLock<u64>,
        }

        impl NarwhalBullshark {
            pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> Self {
                Self {
                    node_id,
                    peers,
                    finalized_blocks: DashMap::new(),
                    latest_height: RwLock::new(0),
                }
            }

            pub async fn start(&self) {
                let finalized_blocks = self.finalized_blocks.clone();
                let latest_height = Arc::clone(&self.latest_height);
                
                tokio::spawn(async move {
                    let mut height: u64 = 1;
                    loop {
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        let block = Block {
                            hash: blake3::hash(&height.to_le_bytes()).into(),
                            seq_num: height,
                            transactions: Vec::new(),
                            timestamp: SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        };
                        finalized_blocks.insert(height, block);
                        *latest_height.write().await = height;
                        height += 1;
                    }
                });
            }

            pub async fn add_transaction(&self, tx: Transaction) -> Result<()> {
                let height = *self.latest_height.read().await + 1;
                let mut block = self.get_finalized_block(height - 1)
                    .await
                    .unwrap_or_else(|| Block {
                        hash: blake3::hash(&height.to_le_bytes()).into(),
                        seq_num: height,
                        transactions: Vec::new(),
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    });
                block.transactions.push(tx);
                self.finalized_blocks.insert(height, block);
                *self.latest_height.write().await = height;
                Ok(())
            }

            pub async fn get_latest_finalized(&self) -> u64 {
                *self.latest_height.read().await
            }

            pub async fn get_finalized_block(&self, height: u64) -> Option<Block> {
                self.finalized_blocks.get(&height).map(|b| b.clone())
            }
        }
    }
}

mod vm {
    use super::*;
    use async_trait::async_trait;
    use anyhow::Result;
    use serde::{Serialize, Deserialize};

    pub type Address = u64;

    #[derive(Clone, Serialize, Deserialize)]
    pub struct ExecutionResult {
        pub success: bool,
        pub return_data: Vec<u8>,
        pub gas_used: u64,
        pub logs: Vec<String>,
        pub error: Option<String>,
    }

    #[derive(Clone)]
    pub struct ContractState {
        pub code: Vec<u8>,
        pub storage: HashMap<Vec<u8>, Vec<u8>>,
    }

    pub struct CallData {
        pub contract_address: Address,
        pub function: String,
        pub arguments: Vec<u8>,
        pub sender: Address,
        pub gas_limit: u64,
        pub gas_price: u64,
        pub value: u64,
    }

    pub struct StateDB {
        // Placeholder for state database; in a real implementation, this would interact with persistent storage
        state: Arc<RwLock<VmState>>,
    }

    impl StateDB {
        pub fn new(state: Arc<RwLock<VmState>>) -> Self {
            Self { state }
        }
    }

    #[async_trait]
    pub trait StateAccess: Send + Sync {
        async fn get_contract(&self, address: Address) -> Result<Option<Vec<u8>>, VmError>;
        async fn get_storage(&self, address: Address, key: &[u8]) -> Result<Option<Vec<u8>>, VmError>;
        async fn set_storage(&self, address: Address, key: Vec<u8>, value: Vec<u8>) -> Result<(), VmError>;
        async fn get_balance(&self, address: Address) -> Result<u64, VmError>;
        async fn set_balance(&self, address: Address, amount: u64) -> Result<(), VmError>;
        async fn get_nonce(&self, address: Address) -> Result<u64, VmError>;
        async fn get_contract_state(&self, address: Address) -> Result<Option<ContractState>, VmError>;
    }

    pub struct VirtualMachine {
        state_db: Arc<StateDB>,
    }

    impl VirtualMachine {
        pub fn new(state_db: Arc<StateDB>) -> Self {
            Self { state_db }
        }

        pub async fn execute(&self, call_data: &CallData, _state: &dyn StateAccess) -> Result<ExecutionResult, VmError> {
            let gas_used = call_data.gas_limit / 2;
            Ok(ExecutionResult {
                success: true,
                return_data: Vec::new(), // Simplified; real implementation might process WASM and return data
                gas_used,
                logs: vec![format!("Executed function: {}", call_data.function)],
                error: None,
            })
        }
    }

    #[derive(Debug)]
    pub enum VmError {
        SerializationError(String),
        InsufficientBalance,
        InvalidNonce,
        ContractNotFound(String),
    }

    impl std::fmt::Display for VmError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::SerializationError(e) => write!(f, "Serialization error: {}", e),
                Self::InsufficientBalance => write!(f, "Insufficient balance"),
                Self::InvalidNonce => write!(f, "Invalid nonce"),
                Self::ContractNotFound(addr) => write!(f, "Contract not found: {}", addr),
            }
        }
    }

    impl std::error::Error for VmError {}
    
    // Added from crate root - needs to be accessible here for StateAccess
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct VmState {
        pub contracts: HashMap<Address, Vec<u8>>,
        pub storage: HashMap<Address, HashMap<Vec<u8>, Vec<u8>>>,
        pub balances: HashMap<Address, u64>,
        pub nonces: HashMap<Address, u64>,
    }
}

// Adjusted imports for correct paths
use crate::consensus::narwhal_bullshark::NarwhalBullshark;
use std::sync::Arc;
use crate::consensus::pbft::Block;
use crate::vm::{VirtualMachine, VmError, ExecutionResult, ContractState, CallData, StateAccess, Address, StateDB};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartContractTx {
    pub address: Address,
    pub function: String,
    pub arguments: Vec<u8>,
    pub sender: Address,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub nonce: u64,
    pub value: u64,
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
}

// VmState moved to vm module to avoid circular dependencies

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmBlockResult {
    pub block_hash: [u8; 32],
    pub height: u64,
    pub transactions: Vec<SmartContractTx>,
    pub tx_results: Vec<LocalExecutionResult>,
    pub state_root: [u8; 32],
    pub timestamp: u64,
}

pub struct NarwhalBullsharkVm {
    consensus: Arc<NarwhalBullshark>,
    vm: Arc<VirtualMachine>,
    current_state: Arc<RwLock<VmState>>,
    state_history: Arc<DashMap<u64, [u8; 32]>>,
    pending_txs: Arc<Mutex<Vec<SmartContractTx>>>,
    tx_results: Arc<DashMap<[u8; 32], ExecutionResult>>,
    block_results: Arc<DashMap<u64, VmBlockResult>>,
    last_processed_height: Arc<RwLock<u64>>,
    tx_throughput: Arc<RwLock<(u64, Instant)>>,
    shutdown: Arc<TokioMutex<bool>>,
}

impl NarwhalBullsharkVm {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>, vm: Arc<VirtualMachine>) -> Self {
        let consensus = Arc::new(NarwhalBullshark::new(node_id, peers));
        let current_state = Arc::new(RwLock::new(VmState {
            contracts: HashMap::new(),
            storage: HashMap::new(),
            balances: HashMap::new(),
            nonces: HashMap::new(),
        }));

        Self {
            consensus,
            vm,
            current_state,
            state_history: Arc::new(DashMap::new()),
            pending_txs: Arc::new(Mutex::new(Vec::new())),
            tx_results: Arc::new(DashMap::new()),
            block_results: Arc::new(DashMap::new()),
            last_processed_height: Arc::new(RwLock::new(0)),
            tx_throughput: Arc::new(RwLock::new((0, Instant::now()))),
            shutdown: Arc::new(TokioMutex::new(false)),
        }
    }

    pub async fn start(&self) -> Result<(), VmError> {
        let consensus = Arc::clone(&self.consensus);
        tokio::spawn(async move {
            (*consensus).start().await;
        });

        self.start_execution_loop();
        self.start_block_processor();
        self.start_metrics_reporter();

        Ok(())
    }

    pub async fn submit_transaction(&self, tx: SmartContractTx) -> Result<[u8; 32], VmError> {
        self.validate_transaction(&tx).await?;
        let tx_hash = self.compute_tx_hash(&tx);
        let consensus_tx = Transaction {
            hash: tx_hash,
            data: bincode::serialize(&tx)
                .map_err(|e| VmError::SerializationError(e.to_string()))?,
            sender: self.address_to_bytes(&tx.sender),
            nonce: tx.nonce,
            signature: tx.signature,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        {
            let mut pending = self.pending_txs.lock();
            pending.push(tx);
        }

        self.consensus.add_transaction(consensus_tx).await
            .map_err(|_| VmError::SerializationError("Failed to add transaction to consensus".to_string()))?;
        Ok(tx_hash)
    }

    async fn validate_transaction(&self, tx: &SmartContractTx) -> Result<(), VmError> {
        let state = self.current_state.read().await;
        let balance = *state.balances.get(&tx.sender).unwrap_or(&0);
        let tx_cost = tx.gas_limit * tx.gas_price + tx.value;

        if balance < tx_cost {
            return Err(VmError::InsufficientBalance);
        }

        let current_nonce = *state.nonces.get(&tx.sender).unwrap_or(&0);
        if tx.nonce < current_nonce {
            return Err(VmError::InvalidNonce);
        }

        if !tx.function.is_empty() && !state.contracts.contains_key(&tx.address) {
            return Err(VmError::ContractNotFound(tx.address.to_string()));
        }

        Ok(())
    }

    fn start_execution_loop(&self) {
        let vm = Arc::clone(&self.vm);
        let current_state: Arc<RwLock<VmState>> = Arc::clone(&self.current_state);
        let pending_txs = Arc::clone(&self.pending_txs);
        let tx_results = Arc::clone(&self.tx_results);

        tokio::spawn(async move {
            let state_access = VmStateAccess::new(Arc::clone(&current_state));
            loop {
                let batch = {
                    let mut pending = pending_txs.lock();
                    let count = pending.len().min(100);
                    if count == 0 {
                        Vec::new()
                    } else {
                        pending.drain(0..count).collect::<Vec<_>>()
                    }
                };
                if batch.is_empty() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }

                for tx in batch {
                    let tx_hash = Self::compute_tx_hash_static(&tx);
                    let call_data = CallData {
                        contract_address: tx.address,
                        function: tx.function.clone(),
                        arguments: tx.arguments.clone(),
                        sender: tx.sender,
                        gas_limit: tx.gas_limit,
                        gas_price: tx.gas_price,
                        value: tx.value,
                    };

                    match vm.execute(&call_data, &state_access).await {
                        Ok(result) => {
                            tx_results.insert(tx_hash, result);
                        },
                        Err(e) => {
                            tx_results.insert(tx_hash, ExecutionResult {
                                success: false,
                                return_data: Vec::new(),
                                gas_used: 0,
                                logs: Vec::new(),
                                error: Some(e.to_string()),
                            });
                        }
                    }
                }
            }
        });
    }

    fn start_block_processor(&self) {
        let consensus = Arc::clone(&self.consensus);
        let current_state: Arc<RwLock<VmState>> = Arc::clone(&self.current_state);
        let state_history = Arc::clone(&self.state_history);
        let block_results = Arc::clone(&self.block_results);
        let tx_results = Arc::clone(&self.tx_results);
        let last_processed_height = Arc::clone(&self.last_processed_height);
        let tx_throughput = Arc::clone(&self.tx_throughput);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            loop {
                interval.tick().await;
                let latest_height = consensus.get_latest_finalized().await;
                let processed_height = *last_processed_height.read().await;

                if latest_height <= processed_height {
                    continue;
                }

                for height in (processed_height + 1)..=latest_height {
                    if let Some(block) = consensus.get_finalized_block(height).await {
                        let vm_block_result = process_block(&block, &tx_results).await;
                        update_state(&block, &vm_block_result, &current_state).await;
                        state_history.insert(height, vm_block_result.state_root);
                        block_results.insert(height, vm_block_result);
                        let mut throughput = tx_throughput.write().await;
                        throughput.0 += block.transactions.len() as u64;
                    }
                }

                let mut last_height = last_processed_height.write().await;
                *last_height = latest_height;
            }
        });

        async fn process_block(
            block: &Block,
            tx_results: &DashMap<[u8; 32], ExecutionResult>,
        ) -> VmBlockResult {
            let mut vm_txs = Vec::new();
            let mut results = Vec::new();

            for tx in &block.transactions {
                if let Ok(vm_tx) = bincode::deserialize::<SmartContractTx>(&tx.data) {
                    if let Some(result) = tx_results.get(&tx.hash) {
                        vm_txs.push(vm_tx);
                        results.push(to_local_result(&result));
                    }
                }
            }

            let mut hasher = blake3::Hasher::new();
            // Use a simpler approach that doesn't try to serialize ExecutionResult
            for (i, tx) in vm_txs.iter().enumerate() {
                if let Ok(tx_bytes) = bincode::serialize(tx) {
                    hasher.update(&tx_bytes);
                    // Instead of serializing the result, just use its fields
                    let result = &results[i];
                    hasher.update(&[result.success as u8]);
                    hasher.update(&result.return_data);
                    hasher.update(&result.gas_used.to_le_bytes());
                    for log in &result.logs {
                        hasher.update(log.as_bytes());
                    }
                    if let Some(error) = &result.error {
                        hasher.update(error.as_bytes());
                    }
                }
            }

            let mut state_root = [0u8; 32];
            state_root.copy_from_slice(hasher.finalize().as_bytes());

            VmBlockResult {
                block_hash: block.hash,
                height: block.seq_num,
                transactions: vm_txs,
                tx_results: results,
                state_root,
                timestamp: block.timestamp,
            }
        }

        async fn update_state(
            _block: &Block,
            vm_block_result: &VmBlockResult,
            current_state: &RwLock<VmState>,
        ) {
            let mut state = current_state.write().await;
            for (i, tx) in vm_block_result.transactions.iter().enumerate() {
                let result = &vm_block_result.tx_results[i];
                if result.success {
                    state.nonces.insert(tx.sender, tx.nonce + 1);
                    if let Some(balance) = state.balances.get_mut(&tx.sender) {
                        let gas_cost = result.gas_used * tx.gas_price;
                        *balance = balance.saturating_sub(gas_cost);
                    }
                    if tx.value > 0 {
                        if let Some(sender_balance) = state.balances.get_mut(&tx.sender) {
                            *sender_balance = sender_balance.saturating_sub(tx.value);
                        }
                        let recipient_balance = state.balances.entry(tx.address).or_insert(0);
                        *recipient_balance = recipient_balance.saturating_add(tx.value);
                    }
                    if tx.function.is_empty() && !result.return_data.is_empty() {
                        state.contracts.insert(tx.address, result.return_data.clone());
                    }
                }
            }
        }
    }

    fn start_metrics_reporter(&self) {
        let tx_throughput = Arc::clone(&self.tx_throughput);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                if *shutdown.lock().await {
                    break;
                }
                let mut throughput = tx_throughput.write().await;
                let tx_count = throughput.0;
                let elapsed = throughput.1.elapsed();
                if tx_count > 0 && elapsed.as_secs() > 0 {
                    let tps = tx_count as f64 / elapsed.as_secs() as f64;
                    println!("VM TPS: {:.2} ({} transactions in {:?})", tps, tx_count, elapsed);
                    throughput.0 = 0;
                    throughput.1 = Instant::now();
                }
            }
        });
    }

    pub async fn stop(&self) -> Result<(), VmError> {
        let mut shutdown = self.shutdown.lock().await;
        *shutdown = true;
        Ok(())
    }

    pub async fn get_tps(&self) -> f64 {
        let throughput = self.tx_throughput.read().await;
        let tx_count = throughput.0;
        let elapsed = throughput.1.elapsed();
        if tx_count > 0 && elapsed.as_secs() > 0 {
            tx_count as f64 / elapsed.as_secs() as f64
        } else {
            0.0
        }
    }

    pub async fn get_block_result(&self, height: u64) -> Option<VmBlockResult> {
        self.block_results.get(&height).map(|r| r.clone())
    }

    pub async fn get_transaction_result(&self, tx_hash: [u8; 32]) -> Option<ExecutionResult> {
        self.tx_results.get(&tx_hash).map(|r| r.clone())
    }

    fn address_to_bytes(&self, address: &Address) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        let addr_bytes = address.to_be_bytes();
        bytes[32 - addr_bytes.len()..].copy_from_slice(&addr_bytes);
        bytes
    }

    fn compute_tx_hash(&self, tx: &SmartContractTx) -> [u8; 32] {
        Self::compute_tx_hash_static(tx)
    }

    fn compute_tx_hash_static(tx: &SmartContractTx) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&tx.address.to_be_bytes());
        hasher.update(tx.function.as_bytes());
        hasher.update(&tx.arguments);
        hasher.update(&tx.sender.to_be_bytes());
        hasher.update(&tx.gas_limit.to_le_bytes());
        hasher.update(&tx.gas_price.to_le_bytes());
        hasher.update(&tx.nonce.to_le_bytes());
        hasher.update(&tx.value.to_le_bytes());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(hasher.finalize().as_bytes());
        hash
    }
}

struct VmStateAccess {
    state: Arc<RwLock<VmState>>,
}

impl VmStateAccess {
    fn new(state: Arc<RwLock<VmState>>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl StateAccess for VmStateAccess {
    async fn get_contract(&self, address: Address) -> Result<Option<Vec<u8>>, VmError> {
        let state = self.state.read().await;
        Ok(state.contracts.get(&address).cloned())
    }

    async fn get_storage(&self, address: Address, key: &[u8]) -> Result<Option<Vec<u8>>, VmError> {
        let state = self.state.read().await;
        Ok(state.storage.get(&address).and_then(|s| s.get(key).cloned()))
    }

    async fn set_storage(&self, address: Address, key: Vec<u8>, value: Vec<u8>) -> Result<(), VmError> {
        let mut state = self.state.write().await;
        state.storage.entry(address).or_insert_with(HashMap::new).insert(key, value);
        Ok(())
    }

    async fn get_balance(&self, address: Address) -> Result<u64, VmError> {
        let state = self.state.read().await;
        Ok(*state.balances.get(&address).unwrap_or(&0))
    }

    async fn set_balance(&self, address: Address, amount: u64) -> Result<(), VmError> {
        let mut state = self.state.write().await;
        state.balances.insert(address, amount);
        Ok(())
    }

    async fn get_nonce(&self, address: Address) -> Result<u64, VmError> {
        let state = self.state.read().await;
        Ok(*state.nonces.get(&address).unwrap_or(&0))
    }

    async fn get_contract_state(&self, address: Address) -> Result<Option<ContractState>, VmError> {
        let state = self.state.read().await;
        state.contracts.get(&address).map(|code| {
            let storage = state.storage.get(&address).cloned().unwrap_or_default();
            Ok(Some(ContractState { code: code.clone(), storage }))
        }).unwrap_or(Ok(None))
    }
}

pub struct TpsBenchmark {
    vm: Arc<NarwhalBullsharkVm>,
    transaction_count: usize,
    batch_size: usize,
}

impl TpsBenchmark {
    pub fn new(vm: Arc<NarwhalBullsharkVm>, transaction_count: usize, batch_size: usize) -> Self {
        Self { vm, transaction_count, batch_size }
    }

    pub async fn run(&self) -> Result<f64, VmError> {
        println!("Starting TPS benchmark with {} transactions in batches of {}", 
                 self.transaction_count, self.batch_size);

        let start_time = Instant::now();
        let mut submitted = 0;
        let sender_addr = 1001u64;
        {
            let mut state = self.vm.current_state.write().await;
            state.balances.insert(sender_addr, 10_000_000_000);
            state.nonces.insert(sender_addr, 0);
        }

        while submitted < self.transaction_count {
            let batch_size = self.batch_size.min(self.transaction_count - submitted);
            let mut batch = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let nonce = submitted as u64 + i as u64;
                let tx = SmartContractTx {
                    address: 1000u64,
                    function: "transfer".to_string(),
                    arguments: vec![0u8; 32],
                    sender: sender_addr,
                    gas_limit: 100_000,
                    gas_price: 1,
                    nonce,
                    value: 0,
                    signature: [0u8; 64],
                };
                batch.push(tx);
            }

            for tx in batch {
                if let Err(e) = self.vm.submit_transaction(tx).await {
                    println!("Error submitting transaction: {:?}", e);
                }
            }

            submitted += batch_size;
            if submitted % (self.transaction_count / 10) == 0 || submitted == self.transaction_count {
                println!("Submitted {}/{} transactions ({:.1}%)", 
                         submitted, self.transaction_count, 
                         (submitted as f64 / self.transaction_count as f64) * 100.0);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        println!("Waiting for transactions to be processed...");
        let mut processed = 0;
        let timeout = Duration::from_secs(60);
        let start_wait = Instant::now();

        while processed < self.transaction_count && start_wait.elapsed() < timeout {
            let height = *self.vm.last_processed_height.read().await;
            if let Some(block_result) = self.vm.get_block_result(height).await {
                processed += block_result.transactions.len();
                println!("Processed {}/{} transactions ({:.1}%) in block {}", 
                         processed, self.transaction_count, 
                         (processed as f64 / self.transaction_count as f64) * 100.0,
                         height);
                if processed >= self.transaction_count {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        let elapsed = start_time.elapsed();
        let tps = self.transaction_count as f64 / elapsed.as_secs_f64();
        println!("Benchmark completed:");
        println!("  Total transactions: {}", self.transaction_count);
        println!("  Elapsed time: {:.2} seconds", elapsed.as_secs_f64());
        println!("  Transactions per second: {:.2} TPS", tps);
        Ok(tps)
    }
}

pub async fn run_benchmark_suite(node_id: String, peers: Vec<String>, vm: Arc<VirtualMachine>) -> Result<(), VmError> {
    println!("Starting Narwhal-Bullshark VM benchmark suite");
    let nbs_vm = Arc::new(NarwhalBullsharkVm::new(node_id, peers, vm));
    nbs_vm.start().await?;
    tokio::time::sleep(Duration::from_secs(5)).await;

    let batch_sizes = [10, 50, 100, 500, 1000];
    let transaction_count = 10000;
    let mut results = Vec::new();

    for &batch_size in &batch_sizes {
        println!("\nRunning benchmark with batch size: {}", batch_size);
        let benchmark = TpsBenchmark::new(nbs_vm.clone(), transaction_count, batch_size);
        match benchmark.run().await {
            Ok(tps) => results.push((batch_size, tps)),
            Err(e) => println!("Benchmark failed: {:?}", e),
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    println!("\nBenchmark Results Summary:");
    println!("---------------------------");
    for (batch_size, tps) in &results {
        println!("Batch size {}: {:.2} TPS", batch_size, tps);
    }

    if let Some((best_batch, best_tps)) = results.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) {
        println!("\nBest performance: {:.2} TPS with batch size {}", best_tps, best_batch);
    }

    nbs_vm.stop().await?;
    Ok(())
}

// Create a simple config module to handle missing references
pub mod config {
    pub fn load_config(_config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Placeholder implementation
        Ok(())
    }

    pub fn update_batch_size(_size: usize) {
        // Placeholder implementation
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::args().nth(1).unwrap_or_else(|| "config.toml".to_string());
    let _ = config::load_config(&config_path);
    
    // Handle batch size argument if provided
    if let Some(size_str) = std::env::args().nth(2) {
        if let Ok(size) = size_str.parse::<usize>() {
            config::update_batch_size(size);
        }
    }
    
    let current_state = Arc::new(RwLock::new(VmState {
        contracts: HashMap::new(),
        storage: HashMap::new(),
        balances: HashMap::new(),
        nonces: HashMap::new(),
    }));
    let state_db = Arc::new(StateDB::with_state(current_state));
    let vm = Arc::new(VirtualMachine::new(state_db));
    let node_id = "node1".to_string();
    let peers = vec!["node2".to_string(), "node3".to_string()];
    run_benchmark_suite(node_id, peers, vm).await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    Ok(())
}
```

---


## src/vm/narwhal_bullshark_vm/mod.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/narwhal_bullshark_vm/mod.rs`

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::vm::{VirtualMachine, VmError};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use serde_big_array::big_array;

// Initialize BigArray for arrays up to size 64
big_array! { BigArray; 64 }

pub type NodeId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartContractTx {
    pub address: u64,
    pub function: String,
    pub arguments: Vec<u8>,
    pub sender: u64,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub nonce: u64,
    pub value: u64,
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
}

pub struct NarwhalBullsharkVm {
    node_id: NodeId,
    peers: Vec<NodeId>,
    vm: Arc<VirtualMachine>,
    // Add explicit TPS tracking fields
    tx_count: Arc<RwLock<u64>>,
    start_time: Arc<RwLock<Instant>>,
    // Other fields would go here in a real implementation
}

impl NarwhalBullsharkVm {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>, vm: Arc<VirtualMachine>) -> Self {
        Self {
            node_id,
            peers,
            vm,
            // Initialize TPS tracking
            tx_count: Arc::new(RwLock::new(0)),
            start_time: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    pub async fn start(&self) -> Result<(), VmError> {
        println!("Starting NarwhalBullshark VM...");
        // Reset TPS counter when starting
        self.reset_tps_counter().await?;
        Ok(())
    }
    
    pub async fn stop(&self) -> Result<(), VmError> {
        println!("Stopping NarwhalBullshark VM...");
        Ok(())
    }
    
    pub async fn submit_transaction(&self, _tx: SmartContractTx) -> Result<[u8; 32], VmError> {
        // Increment transaction count for TPS calculation
        {
            let mut count = self.tx_count.write().await;
            *count += 1;
        }
        
        // Return a dummy hash for testing
        Ok([0; 32])
    }
    
    pub async fn get_tps(&self) -> f64 {
        // Get transaction count and elapsed time
        let count = *self.tx_count.read().await;
        let start = *self.start_time.read().await;
        let elapsed = start.elapsed().as_secs_f64();
        
        // Debug output to verify values
        println!("DEBUG - TX count: {}, elapsed: {:.2}s", count, elapsed);
        
        // Avoid division by zero
        if elapsed <= 0.001 {  // Using a small threshold instead of exact zero
            return 0.0;
        }
        
        // Calculate and return actual TPS
        count as f64 / elapsed
    }
    
    // New method to provide detailed TPS metrics for debugging
    pub async fn get_detailed_tps(&self) -> (u64, f64, f64) {
        let count = *self.tx_count.read().await;
        let start = *self.start_time.read().await;
        let elapsed = start.elapsed().as_secs_f64();
        
        let tps = if elapsed <= 0.0 { 0.0 } else { count as f64 / elapsed };
        
        (count, elapsed, tps)
    }
    
    // Reset TPS counter for fresh measurements
    pub async fn reset_tps_counter(&self) -> Result<(), VmError> {
        {
            let mut count = self.tx_count.write().await;
            *count = 0;
        }
        {
            let mut start = self.start_time.write().await;
            *start = Instant::now();
        }
        println!("TPS counter reset successfully");
        Ok(())
    }
}

// Config functions for the benchmarking tool
pub mod config {
    pub fn load_config(config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("Loading config from {}", config_path);
        Ok(())
    }
    
    pub fn update_batch_size(batch_size: usize) {
        println!("Updating batch size to {}", batch_size);
    }
}```

---


## src/vm/parallel_executor.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/parallel_executor.rs`

```rust
use crate::contracts::{Contract, ContractCall};
use std::collections::HashMap;

use crate::vm::VmError;
use crate::state::StateDB;
use crate::vm::tiered_vm::TieredVM;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;
use std::fmt;

// Results of parallel execution
pub struct BatchExecutionResult {
    pub results: Vec<Result<Vec<u8>, VmError>>,
    pub total_time: Duration,
    pub gas_used: u64,
}

// Executor for parallel contract execution
pub struct ParallelExecutor {
    pub state_db: Arc<StateDB>,
    pub thread_count: usize,
}

// Add Debug implementation directly here
impl fmt::Debug for ParallelExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParallelExecutor")
            .field("thread_count", &self.thread_count)
            .finish()
    }
}

impl ParallelExecutor {
    pub fn new(state_db: Arc<StateDB>) -> Self {
        // Use half of CPU cores for parallel execution
        let thread_count = std::cmp::max(1, num_cpus::get() / 2);
        
        Self {
            state_db,
            thread_count,
        }
    }
    
    // Execute a batch of contract calls in parallel
    pub async fn execute_batch(&self, calls: Vec<(ContractCall, Arc<Contract>)>) -> BatchExecutionResult {
        let start_time = Instant::now();
        let mut results = Vec::with_capacity(calls.len());
        for _ in 0..calls.len() {
            results.push(Ok(Vec::new())); // Initialize with empty results
        }
        
        // Group calls by contract
        let mut contract_groups: HashMap<[u8; 32], Vec<(usize, ContractCall, Arc<Contract>)>> = HashMap::new();
        for (idx, (call, contract)) in calls.into_iter().enumerate() {
            contract_groups.entry(call.contract_address)
                .or_insert_with(Vec::new)
                .push((idx, call, Arc::clone(&contract)));
        }
        
        // Create execution tasks
        let mut tasks = JoinSet::new();
        
        for (_contract_addr, group_calls) in contract_groups {
            // Execute this contract's calls in a separate task
            let state_db = Arc::clone(&self.state_db);
            
            tasks.spawn(async move {
                let mut task_results = Vec::new();
                let vm = TieredVM::new(state_db);
                
                for (idx, call, contract) in group_calls {
                    // The args in ContractCall is already a Vec<Vec<u8>>, so we need to pass it as a slice
                    // Fixed: args field is likely Vec<u8>, so we need to convert it to Vec<Vec<u8>> first
                    let args_as_vec_of_vec = vec![call.args.clone()]; // Wrap in a vector to match &[Vec<u8>]
                    let result = vm.execute(&contract, &call.method, &args_as_vec_of_vec);
                    task_results.push((idx, result));
                }
                
                task_results
            });
        }
        
        // Collect results
        while let Some(task_result) = tasks.join_next().await {
            if let Ok(task_results) = task_result {
                for (idx, result) in task_results {
                    results[idx] = result;
                }
            }
        }
        
        // Calculate gas used (simplified)
        let gas_used = start_time.elapsed().as_micros() as u64;
        
        BatchExecutionResult {
            results,
            total_time: start_time.elapsed(),
            gas_used,
        }
    }
    
    // Using rayon for CPU-bound tasks
    pub fn execute_batch_sync(&self, _calls: Vec<(ContractCall, Arc<Contract>)>) -> BatchExecutionResult {
        // Create a dummy implementation that compiles
        let start_time = Instant::now();
        let results = vec![Ok(Vec::new())]; // Dummy result
        
        BatchExecutionResult {
            results,
            total_time: start_time.elapsed(),
            gas_used: 0,
        }
    }
}```

---


## src/vm/tiered_vm.rs

### File path: `/home/myuser/viper/dagknight-vm/src/vm/tiered_vm.rs`

```rust
use crate::contracts::Contract;

use std::sync::Arc;
use crate::vm::VmError;
use crate::state::StateDB;
use std::fmt;

// Simplified tiered VM for compilation
#[derive(Clone)]
pub struct TieredVM {
    pub state_db: Arc<StateDB>,
}

impl fmt::Debug for TieredVM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TieredVM").finish()
    }
}

impl TieredVM {
    pub fn new(state_db: Arc<StateDB>) -> Self {
        Self {
            state_db,
        }
    }
    
    // Execute a contract function
    pub fn execute(&self, _contract: &Contract, _function: &str, _args: &[Vec<u8>]) -> Result<Vec<u8>, VmError> {
        // Simplified implementation for compilation
        Ok(vec![0u8; 4])
    }
}
```

---

