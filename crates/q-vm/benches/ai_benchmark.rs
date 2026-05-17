use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rocksdb::{ColumnFamilyDescriptor, Options, DB};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

// Import DAGKnight VM components
use dagknight_vm::cache::{CacheProvider, ModelCache};
use dagknight_vm::contracts::{AIModelCall, CachePolicy, Transaction, TransactionType};
use dagknight_vm::fault_tolerance::{RecoveryManager, RecoverySettings};
use dagknight_vm::models::ModelRegistry;
use dagknight_vm::network::p2p::{NetworkMessage, P2PMessageHandler};
use dagknight_vm::state::{ResourceUsage, StateDB};
use dagknight_vm::vm::ai::executor::AIExecutor;

// Test prompt for the model
const TEST_PROMPT: &str = "Explain the concept of a distributed ledger in simple terms.";

// Number of nodes to simulate in the benchmark
const NODE_COUNT: usize = 5;

// Smart contract code for AI model execution
const AI_CONTRACT_CODE: &[u8] = include_bytes!("../fixtures/ai_contract.wasm");

// Benchmark configuration
struct BenchConfig {
    node_count: usize,
    db_paths: Vec<TempDir>,
    contract_address: [u8; 32],
    model_name: String,
    shard_count: u64,
    iterations: usize,
}

// Node simulation
struct NodeSim {
    id: usize,
    db: Arc<DB>,
    executor: Arc<AIExecutor>,
    state: Arc<StateDB>,
    recovery: Arc<RecoveryManager>,
    cache: Arc<ModelCache>,
}

// Setup database for a node
fn setup_db(path: &TempDir) -> Arc<DB> {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.create_missing_column_families(true);

    // Define column families
    let cf_names = vec!["contracts", "state", "transactions", "models", "cache"];
    let cfs: Vec<_> = cf_names
        .iter()
        .map(|name| {
            let mut cf_opts = Options::default();
            cf_opts.set_max_write_buffer_number(4);
            cf_opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
            ColumnFamilyDescriptor::new(*name, cf_opts)
        })
        .collect();

    // Open database with column families
    let db =
        DB::open_cf_descriptors(&opts, path.path(), cfs).expect("Failed to open RocksDB database");

    Arc::new(db)
}

// Initialize a single node for testing
async fn init_node(id: usize, db_path: &TempDir, model_registry: Arc<ModelRegistry>) -> NodeSim {
    // Setup database
    let db = setup_db(db_path);

    // Create components
    let cache = Arc::new(ModelCache::new(CacheProvider::Memory, 10000, None));

    let recovery = Arc::new(RecoveryManager::new(RecoverySettings {
        enable_replication: true,
        replication_factor: 2,
        max_retries: 3,
        retry_delay_ms: 500,
        task_timeout_secs: 60,
    }));

    let state = Arc::new(StateDB::new());

    let executor = Arc::new(
        AIExecutor::new(cache.clone(), model_registry.clone(), recovery.clone())
            .await
            .expect("Failed to create AI executor"),
    );

    NodeSim {
        id,
        db,
        executor,
        state,
        recovery,
        cache,
    }
}

// Create a transaction for AI model execution
fn create_ai_transaction(
    contract_address: [u8; 32],
    model_name: &str,
    input: &str,
    shard_count: u64,
) -> Transaction {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Create AI model call
    let model_call = AIModelCall {
        model: model_name.to_string(),
        input: input.as_bytes().to_vec(),
        gas_limit: 1_000_000,
        shard_count,
        cache_policy: CachePolicy::NoCache,
    };

    // Create transaction
    Transaction {
        hash: [1u8; 32], // Dummy hash
        tx_type: TransactionType::AIModelExecution(model_call),
        sender: [2u8; 32], // Dummy sender
        nonce: 1,
        gas_price: 1000,
        gas_limit: 1_000_000,
        timestamp: now,
        signature: [3u8; 64], // Dummy signature
    }
}

// Deploy AI contract to simulated blockchain
async fn deploy_contract(nodes: &[NodeSim], contract_address: [u8; 32]) {
    for node in nodes {
        // Store contract code in RocksDB
        let cf_contracts = node
            .db
            .cf_handle("contracts")
            .expect("Column family 'contracts' not found");

        node.db
            .put_cf(&cf_contracts, contract_address, AI_CONTRACT_CODE)
            .expect("Failed to store contract code");
    }
}

// Run the benchmark with specified configuration
async fn run_benchmark(config: &BenchConfig) -> Vec<Duration> {
    // Create shared model registry
    let model_registry = Arc::new(ModelRegistry::new());
    model_registry.initialize_defaults().await;

    // Initialize nodes
    let mut nodes = Vec::with_capacity(config.node_count);
    for i in 0..config.node_count {
        let node = init_node(i, &config.db_paths[i], model_registry.clone()).await;
        nodes.push(node);
    }

    // Deploy contract
    deploy_contract(&nodes, config.contract_address).await;

    // Measure execution time for each iteration
    let mut execution_times = Vec::with_capacity(config.iterations);

    for i in 0..config.iterations {
        // Create transaction
        let tx = create_ai_transaction(
            config.contract_address,
            &config.model_name,
            &format!("{} Iteration: {}", TEST_PROMPT, i),
            config.shard_count,
        );

        // Track starting time
        let start_time = Instant::now();

        // Distribute transaction to all nodes
        let mut handles = Vec::new();
        for (node_idx, node) in nodes.iter().enumerate() {
            // Only process the transaction on nodes based on shard assignment
            if node_idx < config.shard_count as usize {
                let executor = node.executor.clone();
                let tx_clone = tx.clone();

                // Process in parallel
                let handle = tokio::spawn(async move {
                    if let TransactionType::AIModelExecution(model_call) = &tx_clone.tx_type {
                        match executor.execute(model_call, config.contract_address).await {
                            Ok((output, resources)) => {
                                // Return the first 100 bytes of output for validation
                                (
                                    output.get(..100.min(output.len())).unwrap_or(&[]).to_vec(),
                                    resources,
                                )
                            }
                            Err(e) => {
                                panic!("Failed to execute AI model: {:?}", e);
                            }
                        }
                    } else {
                        panic!("Invalid transaction type");
                    }
                });

                handles.push(handle);
            }
        }

        // Gather results
        let results = futures::future::join_all(handles).await;

        // Process results (in a real system, this would be consensus)
        let mut outputs = Vec::new();
        let mut total_resources = ResourceUsage::minimal();

        for result in results {
            let (output, resources) = result.expect("Task failed");
            outputs.push(output);

            // Accumulate resources
            total_resources.cpu_time += resources.cpu_time;
            total_resources.memory_used += resources.memory_used;
            if let Some(gpu_time) = resources.gpu_time {
                total_resources.gpu_time = Some(total_resources.gpu_time.unwrap_or(0) + gpu_time);
            }
        }

        // In a real system, we would validate outputs for consensus
        // Here we just take the first one as the "canonical" result

        // Record execution time
        let execution_time = start_time.elapsed();
        execution_times.push(execution_time);

        // Add a small delay between iterations
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    execution_times
}

// Main benchmark function using Criterion
fn benchmark_ai_execution(c: &mut Criterion) {
    // Create runtime for async code
    let rt = Runtime::new().unwrap();

    // Setup database paths
    let db_paths: Vec<_> = (0..NODE_COUNT)
        .map(|_| TempDir::new().expect("Failed to create temporary directory"))
        .collect();

    // Benchmark parameters
    let shard_counts = vec![1, 2, 3, 5];
    let iterations = 5;

    // Create benchmark group
    let mut group = c.benchmark_group("distributed_ai");
    group.sample_size(10); // Reduced sample size due to computational cost
    group.measurement_time(Duration::from_secs(10));

    for shard_count in shard_counts {
        // Create benchmark config
        let config = BenchConfig {
            node_count: NODE_COUNT,
            db_paths: db_paths.clone(),
            contract_address: [42u8; 32],
            model_name: "deepseek-r1:1.5b".to_string(),
            shard_count,
            iterations,
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("shards_{}", shard_count)),
            &shard_count,
            |b, _| {
                b.iter(|| {
                    // Run the benchmark
                    let execution_times = rt.block_on(run_benchmark(&config));

                    // Calculate and print statistics
                    let total_time: Duration = execution_times.iter().sum();
                    let avg_time = total_time / execution_times.len() as u32;

                    println!(
                        "Shards: {}, Avg execution time: {:?} per transaction",
                        shard_count, avg_time
                    );

                    execution_times
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_ai_execution);
criterion_main!(benches);
