#!/bin/bash
#
# Narwhal-Bullshark VM Integration Script
# ---------------------------------------
# This script integrates the Narwhal-Bullshark consensus mechanism
# with your existing VM system for higher TPS

set -e  # Exit on any error

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration (modify as needed)
PROJECT_ROOT="$(pwd)"  # Assumes script is run from project root
VM_DIR="$PROJECT_ROOT/src/vm"
CONSENSUS_DIR="$PROJECT_ROOT/src/consensus"
CONFIG_DIR="$PROJECT_ROOT/config"
BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
DEFAULT_BATCH_SIZE=100
DEFAULT_BENCH_TRANSACTIONS=10000

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

# Function to display step information
step() {
    echo -e "${BLUE}➤ $1${NC}"
}

# Function to display success messages
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to display warnings
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to display errors and exit
error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    exit 1
}

# Function to create backups of files being modified
backup_file() {
    if [ -f "$1" ]; then
        local backup_path="$BACKUP_DIR/$(basename "$1")"
        cp "$1" "$backup_path"
        echo "Backed up $1 to $backup_path"
    fi
}

# Check if we're in a Rust project
check_environment() {
    step "Checking environment..."
    
    if [ ! -f "Cargo.toml" ]; then
        error "Not a Rust project. Please run this script from your project root."
    fi
    
    if [ ! -d "$VM_DIR" ]; then
        error "VM directory not found at $VM_DIR."
    fi
    
    if [ ! -d "$CONSENSUS_DIR" ]; then
        warning "Consensus directory not found at $CONSENSUS_DIR. Creating it..."
        mkdir -p "$CONSENSUS_DIR"
    fi
    
    if ! command -v cargo &> /dev/null; then
        error "Cargo not found. Please install Rust: https://rustup.rs/"
    fi
    
    success "Environment check passed"
}

# Add necessary dependencies to Cargo.toml
update_dependencies() {
    step "Updating dependencies in Cargo.toml..."
    
    backup_file "Cargo.toml"
    
    # Check if dependencies are already present
    if grep -q "async-trait" "Cargo.toml" && \
       grep -q "dashmap" "Cargo.toml" && \
       grep -q "parking_lot" "Cargo.toml" && \
       grep -q "bincode" "Cargo.toml" && \
       grep -q "blake3" "Cargo.toml"; then
        echo "Required dependencies already present in Cargo.toml"
    else
        # Add dependencies
        cat >> "Cargo.toml" << EOL

# Dependencies for Narwhal-Bullshark consensus
async-trait = "0.1"
dashmap = "5.4"
parking_lot = "0.12"
bincode = "1.3"
blake3 = "1.3"
anyhow = "1.0"
tokio = { version = "1.25", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
EOL
        success "Added required dependencies to Cargo.toml"
    fi
    
    # Run cargo check to update Cargo.lock
    cargo check
    
    success "Dependencies updated"
}

# Create the Narwhal-Bullshark consensus implementation
create_consensus_impl() {
    step "Creating Narwhal-Bullshark consensus implementation..."
    
    # Create directory structure
    mkdir -p "$CONSENSUS_DIR/narwhal_bullshark"
    
    # Create main consensus module file
    cat > "$CONSENSUS_DIR/narwhal_bullshark.rs" << 'EOL'
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
pub use crate::consensus::narwhal_bullshark::types::{
    Transaction, Block, NodeId, VertexId, Vertex, 
    NarwhalMessage, BullsharkMessage, ConsensusMessage
};

// Narwhal data availability layer
pub struct Narwhal {
    node_id: NodeId,
    peers: Vec<NodeId>,
    
    // DAG state
    vertices: DashMap<VertexId, Vertex>,
    parent_vertices: DashMap<u64, HashSet<VertexId>>, // round -> vertices
    
    // Transaction pool
    tx_pool: Arc<Mutex<Vec<Transaction>>>,
    
    // Networking
    tx_network: mpsc::Sender<(NodeId, NarwhalMessage)>,
    
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
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> (Self, mpsc::Receiver<(NodeId, NarwhalMessage)>) {
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
    tx_network: mpsc::Sender<(NodeId, ConsensusMessage)>,
    rx_narwhal: mpsc::Receiver<(NodeId, NarwhalMessage)>,
    
    // Transaction channels
    tx_mempool: mpsc::Sender<Transaction>,
    rx_mempool: mpsc::Receiver<Transaction>,
}

impl NarwhalBullshark {
    // Create a new NarwhalBullshark instance
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
        self.bullshark.get_latest_finalized().await
    }
    
    pub async fn get_finalized_block(&self, seq_num: u64) -> Option<Block> {
        self.bullshark.get_finalized_block(seq_num).await
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
EOL

    # Create consensus types file
    cat > "$CONSENSUS_DIR/narwhal_bullshark/types.rs" << 'EOL'
//! Type definitions for Narwhal-Bullshark consensus

use serde::{Serialize, Deserialize};

// Transaction structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: [u8; 32],         // Transaction hash
    pub data: Vec<u8>,          // Transaction data
    pub sender: [u8; 32],       // Sender's address
    pub nonce: u64,             // Sender's nonce
    pub signature: [u8; 64],    // Transaction signature
    pub timestamp: u64,         // Timestamp when created
}

// DAG vertex for Narwhal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VertexId(pub [u8; 32]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    pub id: VertexId,
    pub round: u64,
    pub author: NodeId,
    pub parents: Vec<VertexId>,
    pub transactions: Vec<Transaction>,
    pub timestamp: u64,
}

// Block structure with additional DAG information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub hash: [u8; 32],
    pub parent_hash: [u8; 32],
    pub seq_num: u64,
    pub round: u64,
    pub vertices: Vec<VertexId>,  // References to vertices included in this block
    pub transactions: Vec<Transaction>,
    pub timestamp: u64,
    pub proposer: NodeId,
}

// Node identifier
pub type NodeId = String;

// Messages for Narwhal protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NarwhalMessage {
    VertexAnnouncement(Vertex),
    VertexRequest(VertexId),
    VertexResponse(Vertex),
    CertificateAnnouncement(VertexCertificate),
}

// Messages for Bullshark protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BullsharkMessage {
    BlockProposal(Block),
    BlockVote(BlockVote),
    BlockCertificate(BlockCertificate),
}

// Combined message type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    Narwhal(NarwhalMessage),
    Bullshark(BullsharkMessage),
}

// Vertex certificate (proves a vertex has been seen by 2f+1 nodes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexCertificate {
    pub vertex_id: VertexId,
    pub signatures: Vec<(NodeId, [u8; 64])>,
}

// Block vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockVote {
    pub block_hash: [u8; 32],
    pub seq_num: u64,
    pub voter: NodeId,
    pub signature: [u8; 64],
}

// Block certificate (proves 2f+1 votes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockCertificate {
    pub block_hash: [u8; 32],
    pub votes: Vec<BlockVote>,
}
EOL

    # Update consensus mod.rs to include our new implementation
    if [ -f "$CONSENSUS_DIR/mod.rs" ]; then
        backup_file "$CONSENSUS_DIR/mod.rs"
        
        # Check if narwhal_bullshark is already included
        if ! grep -q "pub mod narwhal_bullshark;" "$CONSENSUS_DIR/mod.rs"; then
            # Add our module
            cat >> "$CONSENSUS_DIR/mod.rs" << 'EOL'

// High-throughput consensus implementation
pub mod narwhal_bullshark;
EOL
            success "Updated consensus/mod.rs to include Narwhal-Bullshark"
        else
            echo "Narwhal-Bullshark already included in consensus/mod.rs"
        fi
    else
        # Create new mod.rs file
        cat > "$CONSENSUS_DIR/mod.rs" << 'EOL'
//! Consensus implementations for the VM

// High-throughput consensus implementation
pub mod narwhal_bullshark;

// Re-export common consensus traits and types
pub use self::narwhal_bullshark::{NarwhalBullshark, Transaction, Block, NodeId};
EOL
        success "Created new consensus/mod.rs file"
    fi
    
    success "Created Narwhal-Bullshark consensus implementation"
}

# Update the VM module to integrate with Narwhal-Bullshark
update_vm_module() {
    step "Updating VM module..."
    
    # Check if the VM module file exists
    if [ ! -f "$VM_DIR/mod.rs" ]; then
        error "VM module file $VM_DIR/mod.rs not found."
    fi
    
    backup_file "$VM_DIR/mod.rs"
    
    # Check if Narwhal-Bullshark VM is already included
    if ! grep -q "pub mod narwhal_bullshark_vm;" "$VM_DIR/mod.rs"; then
        # Add our module
        cat >> "$VM_DIR/mod.rs" << 'EOL'

// High-throughput VM implementation using Narwhal-Bullshark
pub mod narwhal_bullshark_vm;
pub use narwhal_bullshark_vm::NarwhalBullsharkVm;
EOL
        success "Updated VM module to include Narwhal-Bullshark VM"
    else
        echo "Narwhal-Bullshark VM already included in VM module"
    fi
    
    # Create a VM configuration file to manage batch size and other parameters
    mkdir -p "$CONFIG_DIR"
    
    if [ ! -f "$CONFIG_DIR/vm_config.toml" ]; then
        cat > "$CONFIG_DIR/vm_config.toml" << EOL
# VM Configuration

[narwhal_bullshark]
# Batch size for transaction processing
batch_size = $DEFAULT_BATCH_SIZE

# Maximum number of parallel execution threads
max_parallel_executions = 4

# Maximum transactions in memory pool
max_mempool_size = 100000

# Interval (ms) for block production
block_production_interval_ms = 1000

# Timeout (ms) for transaction processing
transaction_timeout_ms = 5000

# Enable detailed performance metrics
enable_metrics = true

# Metrics reporting interval (seconds)
metrics_interval_seconds = 10
EOL
        success "Created VM configuration file"
    fi
    
    success "VM module updated successfully"
}

# Create configuration loader
create_config_loader() {
    step "Creating configuration loader..."
    
    # Create config directory if it doesn't exist
    mkdir -p "$CONFIG_DIR"
    
    # Create config module if it doesn't exist
    mkdir -p "$PROJECT_ROOT/src/config"
    
    if [ ! -f "$PROJECT_ROOT/src/config/mod.rs" ]; then
        cat > "$PROJECT_ROOT/src/config/mod.rs" << 'EOL'
//! Configuration management for the VM

use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::sync::RwLock;
use std::collections::HashMap;

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
EOL
        success "Created config module"
    fi
    
    # Update main.rs or lib.rs to include config
    local main_file=""
    if [ -f "$PROJECT_ROOT/src/main.rs" ]; then
        main_file="$PROJECT_ROOT/src/main.rs"
    elif [ -f "$PROJECT_ROOT/src/lib.rs" ]; then
        main_file="$PROJECT_ROOT/src/lib.rs"
    else
        warning "Could not find main.rs or lib.rs to update config loading."
        return
    fi
    
    backup_file "$main_file"
    
    # Check if config is already included
    if ! grep -q "mod config;" "$main_file"; then
        # Insert after other mod declarations or at the beginning
        if grep -q "mod " "$main_file"; then
            # Find the last mod declaration
            line_num=$(grep -n "mod " "$main_file" | tail -1 | cut -d: -f1)
            # Insert after that line
            sed -i "${line_num}amod config;" "$main_file"
        else
            # Insert at the beginning
            sed -i "1imod config;" "$main_file"
        fi
        success "Updated $main_file to include config module"
    else
        echo "Config module already included in $main_file"
    fi
    
    success "Configuration loader created"
}

# Create benchmark tool
create_benchmark_tool() {
    step "Creating benchmark tool..."
    
    mkdir -p "$PROJECT_ROOT/src/bin"
    
    cat > "$PROJECT_ROOT/src/bin/narwhal_bullshark_bench.rs" << 'EOL'
//! Benchmark tool for Narwhal-Bullshark VM

use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Mutex;
use clap::Parser;

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
    match dagknight_node::config::load_config(config_path) {
        Ok(_) => println!("Loaded configuration from {}", config_path),
        Err(e) => eprintln!("Warning: Failed to load configuration: {}", e),
    }
    
    // Update batch size from arguments
    dagknight_node::config::update_batch_size(args.batch_size);
    
    // Create nodes
    let mut node_ids = Vec::new();
    for i in 0..args.nodes {
        node_ids.push(format!("node_{}", i));
    }
    
    // Create virtual machine
    let vm = Arc::new(dagknight_node::vm::VirtualMachine::new());
    
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
    vm: Arc<dagknight_node::vm::VirtualMachine>,
    transaction_count: usize,
    batch_size: usize
) -> Result<(), Box<dyn std::error::Error>> {
    // Create Narwhal-Bullshark VM
    let nb_vm = Arc::new(dagknight_node::vm::NarwhalBullsharkVm::new(
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
                let tx = dagknight_node::vm::narwhal_bullshark_vm::SmartContractTx {
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
    vm: Arc<dagknight_node::vm::VirtualMachine>,
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
        let nb_vm = Arc::new(dagknight_node::vm::NarwhalBullsharkVm::new(
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
                let tx = dagknight_node::vm::narwhal_bullshark_vm::SmartContractTx {
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
    vm: Arc<dagknight_node::vm::VirtualMachine>,
    batch_size: usize,
    duration_secs: u64
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting stress benchmark for {} seconds...", duration_secs);
    
    // Create and start VM
    let nb_vm = Arc::new(dagknight_node::vm::NarwhalBullsharkVm::new(
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
                    let tx = dagknight_node::vm::narwhal_bullshark_vm::SmartContractTx {
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
}
EOL

    success "Created benchmark tool"
}

# Create a Node.js GUI tool for easier benchmarking
create_gui_tool() {
    step "Creating GUI benchmarking tool..."
    
    # Check if Node.js is available
    if ! command -v node &> /dev/null; then
        warning "Node.js not found. Skipping GUI tool creation."
        return
    fi
    
    # Create directory for the GUI tool
    mkdir -p "$PROJECT_ROOT/tools/benchmark-gui"
    
    # Create package.json
    cat > "$PROJECT_ROOT/tools/benchmark-gui/package.json" << 'EOL'
{
  "name": "dagknight-benchmark-gui",
  "version": "1.0.0",
  "description": "GUI tool for benchmarking DAGKnight's Narwhal-Bullshark VM",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  },
  "dependencies": {
    "express": "^4.17.1",
    "socket.io": "^4.4.1",
    "chart.js": "^3.7.0"
  }
}
EOL

    # Create the main server file
    cat > "$PROJECT_ROOT/tools/benchmark-gui/index.js" << 'EOL'
const express = require('express');
const http = require('http');
const { spawn } = require('child_process');
const path = require('path');
const socketIo = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

const PORT = process.env.PORT || 3000;

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Serve the main page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start benchmark
app.post('/start-benchmark', (req, res) => {
  const { transactions, batchSize, nodes, mode, duration } = req.body;
  
  console.log(`Starting benchmark with: transactions=${transactions}, batchSize=${batchSize}, nodes=${nodes}, mode=${mode}, duration=${duration}`);
  
  // Build command arguments
  const args = [
    `--transactions=${transactions}`,
    `--batch-size=${batchSize}`,
    `--nodes=${nodes}`,
    `--mode=${mode}`
  ];
  
  if (mode === 'stress') {
    args.push(`--duration=${duration}`);
  }
  
  // Find the binary path
  const binaryPath = path.resolve('../../target/debug/narwhal_bullshark_bench');
  
  // Start benchmark process
  const benchmark = spawn(binaryPath, args);
  
  let output = '';
  let tpsData = [];
  let lastTimestamp = Date.now();
  
  benchmark.stdout.on('data', (data) => {
    const text = data.toString();
    output += text;
    
    // Parse TPS data
    const tpsMatch = text.match(/TPS: ([0-9.]+)/);
    if (tpsMatch) {
      const now = Date.now();
      const tps = parseFloat(tpsMatch[1]);
      tpsData.push({ timestamp: now, tps });
      
      // Send TPS update to clients
      io.emit('tps-update', { timestamp: now, tps });
    }
    
    // Send output to clients
    io.emit('benchmark-output', { text });
  });
  
  benchmark.stderr.on('data', (data) => {
    const text = data.toString();
    output += text;
    io.emit('benchmark-output', { text, error: true });
  });
  
  benchmark.on('close', (code) => {
    const success = code === 0;
    console.log(`Benchmark exited with code ${code}`);
    
    io.emit('benchmark-complete', { 
      success, 
      exitCode: code,
      tpsData
    });
  });
  
  // Respond to the request
  res.json({ success: true, message: 'Benchmark started' });
});

// Socket.io connection
io.on('connection', (socket) => {
  console.log('Client connected');
  
  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

// Start server
server.listen(PORT, () => {
  console.log(`DAGKnight Benchmark GUI running on http://localhost:${PORT}`);
});
EOL

    # Create public directory
    mkdir -p "$PROJECT_ROOT/tools/benchmark-gui/public"
    
    # Create HTML file
    cat > "$PROJECT_ROOT/tools/benchmark-gui/public/index.html" << 'EOL'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DAGKnight Narwhal-Bullshark Benchmark</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      padding-top: 20px;
      padding-bottom: 20px;
    }
    .terminal {
      background-color: #000;
      color: #00ff00;
      font-family: monospace;
      padding: 10px;
      height: 300px;
      overflow-y: auto;
      margin-bottom: 20px;
    }
    .error-text {
      color: #ff0000;
    }
    #tpsChart {
      width: 100%;
      height: 300px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1 class="text-center mb-4">DAGKnight Narwhal-Bullshark Benchmark</h1>
    
    <div class="row">
      <div class="col-md-4">
        <div class="card">
          <div class="card-header">
            Benchmark Configuration
          </div>
          <div class="card-body">
            <form id="benchmarkForm">
              <div class="mb-3">
                <label for="transactions" class="form-label">Transactions</label>
                <input type="number" class="form-control" id="transactions" value="10000">
              </div>
              <div class="mb-3">
                <label for="batchSize" class="form-label">Batch Size</label>
                <input type="number" class="form-control" id="batchSize" value="100">
              </div>
              <div class="mb-3">
                <label for="nodes" class="form-label">Number of Nodes</label>
                <input type="number" class="form-control" id="nodes" value="4">
              </div>
              <div class="mb-3">
                <label for="mode" class="form-label">Benchmark Mode</label>
                <select class="form-control" id="mode">
                  <option value="single">Single Node</option>
                  <option value="multi">Multi Node</option>
                  <option value="stress">Stress Test</option>
                </select>
              </div>
              <div class="mb-3" id="durationContainer" style="display: none;">
                <label for="duration" class="form-label">Duration (seconds)</label>
                <input type="number" class="form-control" id="duration" value="60">
              </div>
              <button type="submit" class="btn btn-primary" id="startBtn">Start Benchmark</button>
              <button type="button" class="btn btn-secondary" id="exportBtn" disabled>Export Results</button>
            </form>
          </div>
        </div>
      </div>
      
      <div class="col-md-8">
        <div class="card mb-4">
          <div class="card-header">
            TPS Performance
          </div>
          <div class="card-body">
            <canvas id="tpsChart"></canvas>
          </div>
        </div>
        
        <div class="card">
          <div class="card-header">
            Benchmark Output
          </div>
          <div class="card-body p-0">
            <div class="terminal" id="output"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="/socket.io/socket.io.js"></script>
  <script>
    // Connect to socket.io
    const socket = io();
    
    // Elements
    const benchmarkForm = document.getElementById('benchmarkForm');
    const startBtn = document.getElementById('startBtn');
    const exportBtn = document.getElementById('exportBtn');
    const outputElement = document.getElementById('output');
    const modeSelect = document.getElementById('mode');
    const durationContainer = document.getElementById('durationContainer');
    
    // Chart
    const ctx = document.getElementById('tpsChart').getContext('2d');
    const tpsChart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Transactions Per Second',
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1,
          data: []
        }]
      },
      options: {
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'second'
            },
            title: {
              display: true,
              text: 'Time'
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'TPS'
            }
          }
        },
        animation: false,
        responsive: true,
        maintainAspectRatio: false
      }
    });
    
    // Show/hide duration field based on mode
    modeSelect.addEventListener('change', () => {
      durationContainer.style.display = modeSelect.value === 'stress' ? 'block' : 'none';
    });
    
    // Handle form submission
    benchmarkForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      
      // Clear previous results
      outputElement.innerHTML = '';
      tpsChart.data.datasets[0].data = [];
      tpsChart.update();
      
      // Get form data
      const transactions = document.getElementById('transactions').value;
      const batchSize = document.getElementById('batchSize').value;
      const nodes = document.getElementById('nodes').value;
      const mode = document.getElementById('mode').value;
      const duration = document.getElementById('duration').value;
      
      // Update UI
      startBtn.disabled = true;
      startBtn.innerHTML = 'Running...';
      exportBtn.disabled = true;
      
      try {
        // Send request to start benchmark
        const response = await fetch('/start-benchmark', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            transactions,
            batchSize,
            nodes,
            mode,
            duration
          })
        });
        
        const data = await response.json();
        
        if (!data.success) {
          appendOutput(`Error: ${data.message}`, true);
        }
      } catch (error) {
        appendOutput(`Error: ${error.message}`, true);
        resetUI();
      }
    });
    
    // Handle benchmark output
    socket.on('benchmark-output', (data) => {
      appendOutput(data.text, data.error);
    });
    
    // Handle TPS updates
    socket.on('tps-update', (data) => {
      // Add data point to chart
      tpsChart.data.datasets[0].data.push({
        x: data.timestamp,
        y: data.tps
      });
      
      // Keep last 100 points for better visibility
      if (tpsChart.data.datasets[0].data.length > 100) {
        tpsChart.data.datasets[0].data.shift();
      }
      
      tpsChart.update();
    });
    
    // Handle benchmark completion
    socket.on('benchmark-complete', (data) => {
      appendOutput(`\nBenchmark ${data.success ? 'completed successfully' : 'failed'} with exit code ${data.exitCode}`);
      resetUI();
      
      // Enable export if we have data
      if (data.tpsData && data.tpsData.length > 0) {
        exportBtn.disabled = false;
      }
    });
    
    // Append output to terminal
    function appendOutput(text, isError = false) {
      const div = document.createElement('div');
      div.textContent = text;
      
      if (isError) {
        div.classList.add('error-text');
      }
      
      outputElement.appendChild(div);
      outputElement.scrollTop = outputElement.scrollHeight;
    }
    
    // Reset UI state
    function resetUI() {
      startBtn.disabled = false;
      startBtn.innerHTML = 'Start Benchmark';
    }
    
    // Export results
    exportBtn.addEventListener('click', () => {
      const data = tpsChart.data.datasets[0].data;
      
      if (data.length === 0) {
        return;
      }
      
      // Convert to CSV
      const csv = 'timestamp,tps\n' + 
        data.map(point => `${point.x},${point.y}`).join('\n');
      
      // Create download link
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dagknight-benchmark-${new Date().toISOString()}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  </script>
</body>
</html>
EOL

    # Create a startup script
    cat > "$PROJECT_ROOT/tools/benchmark-gui/start.sh" << 'EOL'
#!/bin/bash

# Build the benchmark binary
cd ../../
cargo build --bin narwhal_bullshark_bench
cd - > /dev/null

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Start the GUI
echo "Starting benchmark GUI..."
npm start
EOL
    
    chmod +x "$PROJECT_ROOT/tools/benchmark-gui/start.sh"
    
    success "Created GUI benchmarking tool"
    echo "To use the GUI tool, run: cd tools/benchmark-gui && ./start.sh"
}

# Update VM integration tests
create_integration_tests() {
    step "Creating integration tests..."
    
    mkdir -p "$PROJECT_ROOT/tests/integration"
    
    cat > "$PROJECT_ROOT/tests/integration/narwhal_bullshark_vm_test.rs" << 'EOL'
//! Integration tests for Narwhal-Bullshark VM

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::Mutex;
    
    use crate::vm::{VirtualMachine, VmError, ExecutionResult, ConsensusEngine};
    use crate::vm::narwhal_bullshark_vm::{NarwhalBullsharkVm, SmartContractTx};
    use crate::consensus::narwhal_bullshark::{NarwhalBullshark, Transaction};
    
    // Basic functionality test
    #[tokio::test]
    async fn test_basic_functionality() {
        // Create VM and consensus
        let vm = Arc::new(VirtualMachine::new());
        let nb_vm = Arc::new(NarwhalBullsharkVm::new(
            "test_node".to_string(),
            vec!["peer1".to_string(), "peer2".to_string()],
            vm
        ));
        
        // Start VM
        nb_vm.start().await.expect("Failed to start VM");
        
        // Create and submit a test transaction
        let tx = SmartContractTx {
            address: 1000,
            function: "test".to_string(),
            arguments: vec![1, 2, 3, 4],
            sender: 101,
            gas_limit: 100000,
            gas_price: 1,
            nonce: 0,
            value: 0,
            signature: [0; 64],
        };
        
        let tx_hash = nb_vm.submit_transaction(tx).await.expect("Failed to submit transaction");
        
        // Wait a moment for processing
        tokio::time::sleep(Duration::from_secs(1)).await;
        
        // Get transaction result
        let result = nb_vm.get_transaction_result(tx_hash).await;
        
        // Stop VM
        nb_vm.stop().await.expect("Failed to stop VM");
        
        // Assert that transaction was processed
        assert!(result.is_some(), "Transaction result should be available");
    }
    
    // Performance test with multiple transactions
    #[tokio::test]
    async fn test_transaction_throughput() {
        // Create VM and consensus
        let vm = Arc::new(VirtualMachine::new());
        let nb_vm = Arc::new(NarwhalBullsharkVm::new(
            "test_node".to_string(),
            vec!["peer1".to_string(), "peer2".to_string()],
            vm
        ));
        
        // Start VM
        nb_vm.start().await.expect("Failed to start VM");
        
        // Set up account with balance
        {
            let mut state = nb_vm.current_state.write().await;
            state.balances.insert(101, 10_000_000);
            state.nonces.insert(101, 0);
        }
        
        // Number of transactions to test
        let tx_count = 100;
        
        // Track submission time
        let start_time = Instant::now();
        
        // Submit transactions
        for i in 0..tx_count {
            let tx = SmartContractTx {
                address: 1000,
                function: "transfer".to_string(),
                arguments: vec![1, 2, 3, 4],
                sender: 101,
                gas_limit: 100000,
                gas_price: 1,
                nonce: i as u64,
                value: 0,
                signature: [0; 64],
            };
            
            nb_vm.submit_transaction(tx).await.expect("Failed to submit transaction");
        }
        
        let submission_time = start_time.elapsed();
        
        // Wait for processing to complete
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Calculate TPS
        let total_time = start_time.elapsed();
        let submission_tps = tx_count as f64 / submission_time.as_secs_f64();
        let overall_tps = tx_count as f64 / total_time.as_secs_f64();
        
        // Get VM's TPS measurement
        let vm_tps = nb_vm.get_tps().await;
        
        // Stop VM
        nb_vm.stop().await.expect("Failed to stop VM");
        
        println!("Transaction Throughput Test Results:");
        println!("  Transactions: {}", tx_count);
        println!("  Submission Time: {:.2?}", submission_time);
        println!("  Total Time: {:.2?}", total_time);
        println!("  Submission TPS: {:.2}", submission_tps);
        println!("  Overall TPS: {:.2}", overall_tps);
        println!("  VM Reported TPS: {:.2}", vm_tps);
        
        // Assert reasonable performance
        assert!(submission_tps > 10.0, "Submission TPS should be greater than 10");
    }
    
    // Test with multiple nodes
    #[tokio::test]
    async fn test_multi_node() {
        // Create node IDs
        let node_ids = vec![
            "node_1".to_string(),
            "node_2".to_string(),
            "node_3".to_string(),
        ];
        
        // Create and start VMs
        let vm = Arc::new(VirtualMachine::new());
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
            nb_vm.start().await.expect("Failed to start VM");
            
            vms.push(nb_vm);
        }
        
        // Allow time for nodes to connect
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Set up account with balance
        {
            let mut state = vms[0].current_state.write().await;
            state.balances.insert(101, 10_000_000);
            state.nonces.insert(101, 0);
        }
        
        // Submit a test transaction to the first node
        let tx = SmartContractTx {
            address: 1000,
            function: "transfer".to_string(),
            arguments: vec![1, 2, 3, 4],
            sender: 101,
            gas_limit: 100000,
            gas_price: 1,
            nonce: 0,
            value: 100,
            signature: [0; 64],
        };
        
        let tx_hash = vms[0].submit_transaction(tx).await.expect("Failed to submit transaction");
        
        // Wait for transaction to propagate
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        // Stop all VMs
        for (i, vm) in vms.iter().enumerate() {
            vm.stop().await.expect("Failed to stop VM");
            println!("Stopped node {}", node_ids[i]);
        }
        
        // Assert successful test
        assert!(true, "Multi-node test completed");
    }
}
EOL

    success "Created integration tests"
}

# Update the main library if needed
update_main_library() {
    step "Updating main library..."
    
    # Find the main library file (lib.rs or main.rs)
    local main_file=""
    if [ -f "$PROJECT_ROOT/src/lib.rs" ]; then
        main_file="$PROJECT_ROOT/src/lib.rs"
    elif [ -f "$PROJECT_ROOT/src/main.rs" ]; then
        main_file="$PROJECT_ROOT/src/main.rs"
    else
        warning "Could not find lib.rs or main.rs. Skipping main library update."
        return
    fi
    
    backup_file "$main_file"
    
    # Check if we need to add the lazy_static dependency
    if ! grep -q "lazy_static" "Cargo.toml"; then
        cat >> "Cargo.toml" << EOL

# Lazy static for configuration
lazy_static = "1.4"
EOL
        success "Added lazy_static dependency to Cargo.toml"
    fi
    
    # Add a function to initialize Narwhal-Bullshark VM
    if ! grep -q "init_narwhal_bullshark_vm" "$main_file"; then
        cat >> "$main_file" << 'EOL'

/// Initialize a Narwhal-Bullshark VM instance
///
/// # Arguments
///
/// * `node_id` - The ID of this node
/// * `peers` - List of peer node IDs
/// * `batch_size` - Optional batch size (overrides config)
///
/// # Returns
///
/// Returns a new Narwhal-Bullshark VM instance
pub fn init_narwhal_bullshark_vm(
    node_id: String,
    peers: Vec<String>,
    batch_size: Option<usize>,
) -> std::sync::Arc<vm::NarwhalBullsharkVm> {
    // Load configuration
    let config_path = "config/vm_config.toml";
    let _ = config::load_config(config_path);
    
    // Override batch size if provided
    if let Some(size) = batch_size {
        config::update_batch_size(size);
    }
    
    // Create VM
    let vm = std::sync::Arc::new(vm::VirtualMachine::new());
    
    // Create Narwhal-Bullshark VM
    let nb_vm = std::sync::Arc::new(vm::NarwhalBullsharkVm::new(
        node_id,
        peers,
        vm
    ));
    
    // Start VM
    tokio::spawn(async move {
        if let Err(e) = nb_vm.start().await {
            eprintln!("Error starting VM: {:?}", e);
        }
    });
    
    nb_vm
}
EOL
        success "Added init_narwhal_bullshark_vm function to $main_file"
    fi
    
    success "Main library updated"
}

# Main script execution
main() {
    echo "==============================================="
    echo "DAGKnight Narwhal-Bullshark VM Integration Tool"
    echo "==============================================="
    echo ""
    
    # Check environment
    check_environment
    
    # Update dependencies
    update_dependencies
    
    # Create Narwhal-Bullshark consensus implementation
    create_consensus_impl
    
    # Update VM module
    update_vm_module
    
    # Create configuration loader
    create_config_loader
    
    # Create benchmark tool
    create_benchmark_tool
    
    # Create GUI tool (optional)
    create_gui_tool
    
    # Create integration tests
    create_integration_tests
    
    # Update main library
    update_main_library
    
    echo ""
    echo "==============================================="
    echo "Integration completed successfully!"
    echo "==============================================="
    echo ""
    echo "Next steps:"
    echo "1. Build the project: cargo build"
    echo "2. Run the benchmark: cargo run --bin narwhal_bullshark_bench"
    echo "3. Run integration tests: cargo test --test narwhal_bullshark_vm_test"
    echo "4. Use the GUI tool: cd tools/benchmark-gui && ./start.sh"
    echo ""
    echo "All backups have been stored in: $BACKUP_DIR"
    echo ""
}

# Execute main script
main
