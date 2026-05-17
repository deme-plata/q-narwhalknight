#!/bin/bash

# Fix DAGKnight VM compilation issues
# This script creates necessary directories, fixes module structure, and corrects code issues

set -e

echo "Starting DAGKnight VM fixes..."

# Navigate to project root directory
# Assuming the script is run from the project root
# If not, uncomment and modify the line below:
# cd /path/to/dagknight-vm

# Create lib.rs file
echo "Creating lib.rs..."
cat > src/lib.rs << 'EOF'
pub mod vm;
pub mod network;
pub mod consensus;
pub mod mempool;
pub mod state;
pub mod transaction;
pub mod contracts;
EOF

# Create consensus module structure
echo "Creating consensus module structure..."
mkdir -p src/consensus
cat > src/consensus/mod.rs << 'EOF'
pub mod pbft;
EOF

# Fix pbft.rs by moving it to the proper location
if [ -f "src/consensus/pbft.rs" ]; then
    echo "PBFT file already exists"
else
    echo "Creating pbft.rs from source..."
    # Copy from the paste.txt document
    # This assumes your pbft.rs content is correct and in the paste
    sed -n '/## src\/consensus\/pbft\.rs/,/---/p' paste.txt | grep -v "^---$" | grep -v "^##" | grep -v "^### File path:" > src/consensus/pbft.rs
fi

# Fix network module structure
echo "Fixing network module structure..."
cat > src/network/mod.rs << 'EOF'
pub mod p2p;
pub mod stub;

pub use stub::Network;
EOF

# Fix wasmer imports in vm/executor.rs
echo "Fixing wasmer imports in executor.rs..."
sed -i 's/use wasmer::{Store, Module, Instance, imports, Value, WasmPtr, Memory, MemoryType, WasmerEnv, Function};/use wasmer::{Store, Module, Instance, imports, Value, WasmPtr, Function};\nuse wasmer::FunctionEnv;/' src/vm/executor.rs
sed -i 's/use wasmer_compiler_cranelift::Cranelift;/\/\/ Wasmer 4.0 has integrated Cranelift\n\/\/ use wasmer_compiler_cranelift::Cranelift;/' src/vm/executor.rs

# Replace WasmerEnv with FunctionEnv
echo "Replacing WasmerEnv with FunctionEnv..."
sed -i 's/impl WasmerEnv for VMEnvironment {}/impl VMEnvironment {}/g' src/vm/executor.rs
sed -i 's/WasmerEnv/FunctionEnv/g' src/vm/executor.rs

# Fix mutable reference issue in vm/mod.rs
echo "Fixing mutable reference issue in vm/mod.rs..."
sed -i 's/pub async fn call_contract(&self, contract_address: \[u8; 32\], function: &str, args: Vec<Vec<u8>>,/pub async fn call_contract(\&mut self, contract_address: \[u8; 32\], function: \&str, args: Vec<Vec<u8>>,/g' src/vm/executor.rs

# Fix compiler issues in Store initialization
sed -i 's/let compiler = Cranelift::default();/\/\/ Wasmer 4.0 has a simplified API\nlet compiler = wasmer::compiler_cranelift::Cranelift::default();/' src/vm/executor.rs
sed -i 's/let store = Store::new(&compiler);/let store = Store::new(compiler);/' src/vm/executor.rs

# Fix unused variables warnings
echo "Fixing unused variables warnings..."
sed -i 's/let tx_hash = {/let _tx_hash = {/g' src/vm/mod.rs
sed -i 's/fn verify_nonce(&self, tx: &Transaction)/fn verify_nonce(\&self, _tx: \&Transaction)/g' src/transaction/mod.rs

# Update Cargo.toml
echo "Updating Cargo.toml dependencies..."
# Use awk to update wasmer dependency
awk '/wasmer = / { print "wasmer = { version = \"4.0.0\", features = [\"cranelift\"] }"; next } 1' Cargo.toml > Cargo.toml.new
mv Cargo.toml.new Cargo.toml

# Fix main.rs
echo "Fixing main.rs imports..."
cat > src/main.rs << 'EOF'
use std::sync::Arc;
use std::path::Path;
use structopt::StructOpt;
use tokio::sync::mpsc;

mod vm;
mod network;
mod consensus;
mod mempool;
mod state;
mod transaction;
mod contracts;

use vm::DagkVm;
use network::p2p::P2pNetwork;
use consensus::pbft::PbftConsensus;

#[derive(Debug, StructOpt)]
#[structopt(name = "dagknight-vm", about = "DAGKnight Virtual Machine")]
struct Opt {
    /// Node ID
    #[structopt(short, long)]
    node_id: usize,
    
    /// Peers to connect to
    #[structopt(short, long)]
    peers: Vec<String>,
    
    /// Listen address
    #[structopt(short, long, default_value = "/ip4/0.0.0.0/tcp/4001")]
    listen: String,
    
    /// Data directory
    #[structopt(short, long, default_value = "./dagknight_data")]
    data_dir: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    pretty_env_logger::init();
    
    // Parse command line arguments
    let opt = Opt::from_args();
    
    // Create data directory if it doesn't exist
    let data_dir = Path::new(&opt.data_dir);
    if !data_dir.exists() {
        std::fs::create_dir_all(data_dir)?;
    }
    
    // Create RocksDB path
    let db_path = data_dir.join("db").to_str().unwrap().to_string();
    
    // Initialize P2P network
    let mut network = P2pNetwork::new().await?;
    
    // Listen for incoming connections
    network.listen(&opt.listen).await?;
    
    // Connect to peers
    for peer in &opt.peers {
        match network.connect(peer).await {
            Ok(_) => println!("Connected to peer: {}", peer),
            Err(e) => eprintln!("Failed to connect to peer {}: {:?}", peer, e),
        }
    }
    
    // Start network
    network.start().await;
    
    // Initialize consensus
    let node_id = format!("node-{}", opt.node_id);
    let mut peer_ids = Vec::new();
    
    // Extract peer IDs from peer addresses (simplified)
    for (i, _) in opt.peers.iter().enumerate() {
        if i != opt.node_id {
            peer_ids.push(format!("node-{}", i));
        }
    }
    
    let consensus = Arc::new(PbftConsensus::new(node_id, peer_ids));
    
    // Initialize VM
    let vm = DagkVm::new(&db_path, Arc::new(network), consensus.clone());
    
    // Handle Ctrl+C
    let (tx, mut rx) = mpsc::channel(1);
    ctrlc::set_handler(move || {
        let _ = tx.try_send(());
    })?;
    
    println!("DAGKnight VM started");
    println!("Press Ctrl+C to exit");
    
    // Wait for Ctrl+C
    rx.recv().await;
    
    println!("Shutting down...");
    
    Ok(())
}
EOF

# Fix imports in vm/mod.rs
echo "Fixing imports in vm/mod.rs..."
sed -i 's/use crate::network::p2p::P2pNetwork;/use super::network::p2p::P2pNetwork;/' src/vm/mod.rs
sed -i 's/use crate::consensus::pbft::PbftConsensus;/use super::consensus::pbft::PbftConsensus;/' src/vm/mod.rs
sed -i 's/use crate::mempool::Mempool;/use super::mempool::Mempool;/' src/vm/mod.rs
sed -i 's/use crate::state::StateDB;/use super::state::StateDB;/' src/vm/mod.rs
sed -i 's/use crate::transaction::{Transaction, TransactionManager};/use super::transaction::{Transaction, TransactionManager};/' src/vm/mod.rs
sed -i 's/use crate::contracts::{Contract, ContractRegistry};/use super::contracts::{Contract, ContractRegistry};/' src/vm/mod.rs
sed -i 's/use crate::vm::executor::{WasmExecutor, VMEnvironment};/use self::executor::{WasmExecutor, VMEnvironment};/' src/vm/mod.rs

# Fix other files with crate references
find src -type f -name "*.rs" -not -path "src/lib.rs" -exec sed -i 's/use crate::/use super::super::/g' {} \;

echo "Fixes completed. Try building with 'cargo build' now."
