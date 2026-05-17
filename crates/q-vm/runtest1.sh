#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
  echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
  echo -e "${RED}[ERROR] $1${NC}"
  exit 1
}

success() {
  echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Create fixtures directory if it doesn't exist
if [ ! -d "fixtures" ]; then
  log "Creating fixtures directory..."
  mkdir -p fixtures
fi

# Create the AI contract WebAssembly file
log "Creating AI contract WASM file..."

# Convert WAT to WASM (requires wat2wasm tool)
if ! command -v wat2wasm &> /dev/null; then
  log "Installing WABT (WebAssembly Binary Toolkit)..."
  if [ -f /etc/debian_version ]; then
    apt-get update && apt-get install -y wabt
  elif [ -f /etc/redhat-release ]; then
    dnf install -y wabt
  else
    error "Please install the WABT toolkit manually: https://github.com/WebAssembly/wabt"
  fi
fi

# Create WAT file
cat > fixtures/ai_contract.wat <<EOL
(module
  ;; Import memory and host functions
  (import "env" "memory" (memory 1))
  (import "env" "console_log" (func \$console_log (param i32 i32)))
  (import "env" "execute_ai_model" (func \$execute_ai_model (param i32 i32 i32 i32) (result i32)))
  (import "env" "store_result" (func \$store_result (param i32 i32)))

  ;; String utility functions
  (func \$strlen (param \$str i32) (result i32)
    (local \$len i32)
    (block \$exit
      (loop \$cont
        (br_if \$exit (i32.eq (i32.load8_u (i32.add (get_local \$str) (get_local \$len))) (i32.const 0)))
        (set_local \$len (i32.add (get_local \$len) (i32.const 1)))
        (br \$cont)
      )
    )
    (get_local \$len)
  )

  ;; Data section for constants
  (data (i32.const 100) "deepseek-r1:1.5b") ;; Model name
  (data (i32.const 200) "Executing AI model request") ;; Log message
  (data (i32.const 300) "Execution completed successfully") ;; Success message

  ;; Memory layout:
  ;; - 0-999: Reserved for internal use
  ;; - 1000-1999: Input string from the transaction
  ;; - 2000-2999: Result of AI model execution
  ;; - 3000+: Temporary storage

  ;; Main entry point for contract execution
  (func \$process_ai_request (export "process_ai_request") (param \$input_ptr i32) (param \$input_len i32) (result i32)
    (local \$model_ptr i32)
    (local \$model_len i32)
    (local \$result_ptr i32)
    (local \$result_len i32)
    (local \$shard_count i32)

    ;; Set model information
    (set_local \$model_ptr (i32.const 100))
    (set_local \$model_len (call \$strlen (get_local \$model_ptr)))

    ;; Set input buffer
    (set_local \$result_ptr (i32.const 2000))
    
    ;; Set shard count - default to 5 nodes
    (set_local \$shard_count (i32.const 5))

    ;; Log start of execution
    (call \$console_log (i32.const 200) (call \$strlen (i32.const 200)))

    ;; Call the execute_ai_model host function
    (set_local \$result_len 
      (call \$execute_ai_model 
        (get_local \$model_ptr)
        (get_local \$model_len)
        (get_local \$input_ptr)
        (get_local \$shard_count)
      )
    )

    ;; Store result for later retrieval
    (call \$store_result (get_local \$result_ptr) (get_local \$result_len))

    ;; Log success
    (call \$console_log (i32.const 300) (call \$strlen (i32.const 300)))

    ;; Return result length
    (get_local \$result_len)
  )

  ;; Function to retrieve the result
  (func \$get_result (export "get_result") (param \$output_ptr i32) (result i32)
    ;; In a real implementation, we would copy the result to the output_ptr
    ;; For simplicity, we just return the fixed location of the result
    (i32.const 2000)
  )
)
EOL

# Convert WAT to WASM
wat2wasm fixtures/ai_contract.wat -o fixtures/ai_contract.wasm
if [ $? -ne 0 ]; then
  error "Failed to compile WebAssembly contract"
fi

success "AI contract WASM file created successfully"

# Create benchmark test file
log "Creating benchmark test file..."

cat > benches/ai_benchmark.rs <<EOL
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use std::time::{Duration, Instant};
use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::path::PathBuf;
use tempfile::TempDir;
use std::collections::HashMap;

// Import DAGKnight VM components
use dagknight_vm::contracts::{AIModelCall, CachePolicy, Transaction, TransactionType};
use dagknight_vm::vm::ai::executor::AIExecutor;
use dagknight_vm::models::ModelRegistry;
use dagknight_vm::cache::{ModelCache, CacheProvider};
use dagknight_vm::fault_tolerance::{RecoveryManager, RecoverySettings};
use dagknight_vm::state::{StateDB, ResourceUsage};
use dagknight_vm::network::p2p::{NetworkMessage, P2PMessageHandler};

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
    let cfs: Vec<_> = cf_names.iter()
        .map(|name| {
            let mut cf_opts = Options::default();
            cf_opts.set_max_write_buffer_number(4);
            cf_opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
            ColumnFamilyDescriptor::new(*name, cf_opts)
        })
        .collect();
    
    // Open database with column families
    let db = DB::open_cf_descriptors(&opts, path.path(), cfs)
        .expect("Failed to open RocksDB database");
    
    Arc::new(db)
}

// Initialize a single node for testing
async fn init_node(id: usize, db_path: &TempDir, model_registry: Arc<ModelRegistry>) -> NodeSim {
    // Setup database
    let db = setup_db(db_path);
    
    // Create components
    let cache = Arc::new(ModelCache::new(
        CacheProvider::Memory,
        10000,
        None
    ));
    
    let recovery = Arc::new(RecoveryManager::new(RecoverySettings {
        enable_replication: true,
        replication_factor: 2,
        max_retries: 3,
        retry_delay_ms: 500,
        task_timeout_secs: 60,
    }));
    
    let state = Arc::new(StateDB::new());
    
    let executor = Arc::new(
        AIExecutor::new(
            cache.clone(),
            model_registry.clone(),
            recovery.clone(),
        ).await.expect("Failed to create AI executor")
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
    shard_count: u64
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
        let cf_contracts = node.db.cf_handle("contracts")
            .expect("Column family 'contracts' not found");
        
        node.db.put_cf(&cf_contracts, contract_address, AI_CONTRACT_CODE)
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
            config.shard_count
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
                                (output.get(..100.min(output.len())).unwrap_or(&[]).to_vec(), resources)
                            },
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
            }
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_ai_execution);
criterion_main!(benches);
EOL

success "Benchmark test file created successfully"

# Ensure benches directory exists
if [ ! -d "benches" ]; then
  log "Creating benches directory..."
  mkdir -p benches
fi

# Update Cargo.toml to include benchmark
log "Updating Cargo.toml to include benchmarks..."

if ! grep -q "\[\[bench\]\]" Cargo.toml; then
  cat >> Cargo.toml <<EOL

[[bench]]
name = "ai_benchmark"
harness = false
EOL
fi

# Install Ollama and pull the model if it's not already installed
if ! command -v ollama &> /dev/null; then
  log "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  if [ $? -ne 0 ]; then
    error "Failed to install Ollama."
  fi
fi

log "Pulling deepseek-r1:1.5b model (this may take a while)..."
ollama pull deepseek-r1:1.5b
if [ $? -ne 0 ]; then
  warning "Failed to pull deepseek-r1:1.5b model. It will be downloaded when the benchmark runs."
else
  success "Model deepseek-r1:1.5b pulled successfully."
fi

# Run the benchmark
log "Running distributed AI benchmark..."
log "This will simulate 5 nodes running the deepseek-r1:1.5b model with varying shard counts."
log "The benchmark may take some time to complete, especially on the first run."

# Check if Redis is running (optional for improved caching)
if ! redis-cli ping &>/dev/null; then
  warning "Redis is not running. The benchmark will use memory-only caching."
  warning "For better performance, consider starting Redis: docker run -d -p 6379:6379 redis:alpine"
fi

# Run the benchmark
RUST_LOG=warn cargo bench --bench ai_benchmark

success "Benchmark completed!"
log "For detailed analysis, check the generated report in target/criterion/"
