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
