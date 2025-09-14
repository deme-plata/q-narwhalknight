use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use std::path::Path;
use std::fs;
use std::error::Error;
use structopt::StructOpt;
use parking_lot::RwLock as PLRwLock;

// Import necessary components from dagknight_vm
use dagknight_vm::config;
use dagknight_vm::vm::narwhal_bullshark_vm::{NarwhalBullsharkVm, SmartContractTx};
use dagknight_vm::vm::VirtualMachine;
use dagknight_vm::state::StateDB;
use dagknight_vm::vm::VmError;
use dagknight_vm::transaction::Transaction;
use dagknight_vm::transaction::TransactionStatus;

// Define the CLI arguments for our test
#[derive(StructOpt, Debug)]
#[structopt(name = "dagknight-consensus-test", about = "Test DagKnight VM Narwhal-Bullshark consensus")]
struct Args {
    /// Number of nodes to start
    #[structopt(short, long, default_value = "4")]
    nodes: usize,
    
    /// Number of wallets to create
    #[structopt(short, long, default_value = "10")]
    wallets: usize,
    
    /// Number of transactions to generate per wallet
    #[structopt(short, long, default_value = "1000")]
    transactions: usize,
    
    /// Batch size for transactions
    #[structopt(short, long, default_value = "100")]
    batch_size: usize,
    
    /// Test runtime in seconds
    #[structopt(short = "d", long, default_value = "60")]
    duration: u64,
    
    /// Output directory for test results
    #[structopt(short, long, default_value = "test_results")]
    output_dir: String,
}

// Wallet structure
#[derive(Debug, Clone)]
struct Wallet {
    pub address: [u8; 32],
    pub private_key: [u8; 64],
    pub balance: u64,
    pub nonce: u64,
}

impl Wallet {
    // Create a new wallet with random keys
    fn new() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut address = [0u8; 32];
        let mut private_key = [0u8; 64];
        
        rng.fill(&mut address);
        rng.fill(&mut private_key);
        
        Self {
            address,
            private_key,
            balance: 0,
            nonce: 0,
        }
    }
    
    // Sign a transaction
    fn sign_transaction(&self, tx_data: &[u8]) -> [u8; 64] {
        use ed25519_dalek::{Keypair, Signer, SecretKey};
        
        // Convert our private key to an ed25519 secret key
        let secret = match SecretKey::from_bytes(&self.private_key[0..32]) {
            Ok(secret) => secret,
            Err(_) => {
                // In a real implementation, handle this error properly
                return [0u8; 64];
            }
        };
        
        // Create a keypair from the secret key
        let keypair = Keypair {
            secret,
            public: secret.into(),
        };
        
        // Sign the data
        let signature = keypair.sign(tx_data);
        
        // Convert to our fixed-size array
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(signature.as_ref());
        sig_bytes
    }
}

// Smart contract token definition
#[derive(Debug)]
struct TokenContract {
    pub address: u64,
    pub bytecode: Vec<u8>,
    pub total_supply: u64,
}

impl TokenContract {
    // Deploy a new token contract
    async fn deploy(vm: Arc<NarwhalBullsharkVm>, sender: &Wallet, total_supply: u64) -> Result<Self, Box<dyn Error>> {
        println!("Deploying token contract...");
        
        // In a real implementation, this would be actual token contract bytecode
        // For this test, we're just creating a placeholder
        let bytecode = vec![0, 1, 2, 3, 4]; // Placeholder contract bytecode
        
        // Generate a contract address (in production this would be derived from the deployer's address and nonce)
        let contract_address = 1000; // Using a fixed address for simplicity
        
        // Create the deployment transaction
        let deploy_tx = SmartContractTx {
            address: contract_address,
            function: "constructor".to_string(),
            arguments: total_supply.to_le_bytes().to_vec(),
            sender: u64::from_le_bytes(sender.address[0..8].try_into().unwrap()),
            gas_limit: 1000000,
            gas_price: 1,
            nonce: sender.nonce,
            value: 0,
            signature: sender.sign_transaction(&bytecode),
        };
        
        // Submit the transaction to deploy the contract
        vm.submit_transaction(deploy_tx).await?;
        
        println!("Token contract deployed at address: {}", contract_address);
        
        Ok(Self {
            address: contract_address,
            bytecode,
            total_supply,
        })
    }
    
    // Create a transfer transaction from one wallet to another
    fn create_transfer_tx(&self, from: &mut Wallet, to: &Wallet, amount: u64) -> SmartContractTx {
        // Create the transaction
        let tx = SmartContractTx {
            address: self.address,
            function: "transfer".to_string(),
            arguments: {
                // Encode recipient address and amount
                let mut args = Vec::with_capacity(40);
                args.extend_from_slice(&to.address);
                args.extend_from_slice(&amount.to_le_bytes());
                args
            },
            sender: u64::from_le_bytes(from.address[0..8].try_into().unwrap()),
            gas_limit: 100000,
            gas_price: 1,
            nonce: from.nonce,
            value: 0,
            signature: [0; 64], // Will be filled later
        };
        
        // Sign the transaction
        let mut tx_data = Vec::new();
        tx_data.extend_from_slice(&self.address.to_le_bytes());
        tx_data.extend_from_slice(tx.function.as_bytes());
        tx_data.extend_from_slice(&tx.arguments);
        tx_data.extend_from_slice(&tx.sender.to_le_bytes());
        tx_data.extend_from_slice(&tx.nonce.to_le_bytes());
        
        let mut signed_tx = tx;
        signed_tx.signature = from.sign_transaction(&tx_data);
        
        // Update sender's nonce
        from.nonce += 1;
        
        signed_tx
    }
}

// Test results structure
#[derive(Debug)]
struct TestResults {
    pub nodes: usize,
    pub wallets: usize,
    pub total_transactions: usize,
    pub successful_transactions: usize,
    pub elapsed_time: Duration,
    pub tps: f64,
    pub node_tps: Vec<(String, f64)>,
}

// Initialize test environment
async fn init_test_env(args: &Args) -> Result<(Vec<Arc<NarwhalBullsharkVm>>, Vec<Wallet>), Box<dyn Error>> {
    println!("Initializing test environment with {} nodes and {} wallets", args.nodes, args.wallets);
    
    // Create output directory if it doesn't exist
    if !Path::new(&args.output_dir).exists() {
        fs::create_dir_all(&args.output_dir)?;
    }
    
    // Load configuration
    let config_path = "config/vm_config.toml";
    match config::load_config(config_path) {
        Ok(_) => println!("Loaded configuration from {}", config_path),
        Err(e) => eprintln!("Warning: Failed to load configuration: {}", e),
    }
    
    // Update batch size from arguments
    config::update_batch_size(args.batch_size);
    
    // Create node IDs
    let mut node_ids = Vec::new();
    for i in 0..args.nodes {
        node_ids.push(format!("node_{}", i));
    }
    
    // Create virtual machines for each node
    let mut vms = Vec::new();
    
    for (i, node_id) in node_ids.iter().enumerate() {
        // Create a state database for this node
        let vm_state = Arc::new(tokio::sync::RwLock::new(dagknight_vm::state::VmState::default()));
        let state_db = Arc::new(StateDB::with_state(vm_state));
        
        // Create base VM
        let vm = Arc::new(VirtualMachine::new(state_db));
        
        // Create peers list (all other nodes)
        let peers: Vec<String> = node_ids.iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, id)| id.clone())
            .collect();
        
        // Create Narwhal-Bullshark VM
        let nb_vm = Arc::new(NarwhalBullsharkVm::new(
            node_id.clone(), peers, vm.clone()
        ));
        
        // Start VM
        nb_vm.start().await?;
        
        vms.push(nb_vm);
        
        println!("Started node {} with {} peers", node_id, node_ids.len() - 1);
    }
    
    // Create wallets
    let mut wallets = Vec::new();
    for _ in 0..args.wallets {
        let wallet = Wallet::new();
        wallets.push(wallet);
    }
    
    // Allow time for nodes to connect
    println!("Allowing time for nodes to connect...");
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    Ok((vms, wallets))
}

// Run the token contract deployment test
async fn run_deploy_test(
    vms: &[Arc<NarwhalBullsharkVm>], 
    wallets: &mut [Wallet]
) -> Result<TokenContract, Box<dyn Error>> {
    println!("\nRunning token contract deployment test...");
    
    // Choose the first VM for deployment
    let vm = &vms[0];
    
    // Use the first wallet as the contract deployer
    let deployer = &mut wallets[0];
    
    // Fund the deployer wallet with some initial balance
    // In a real network, this would come from a faucet or genesis allocation
    deployer.balance = 1_000_000;
    
    // Deploy the token contract with a supply of 1,000,000 tokens
    let token = TokenContract::deploy(vm.clone(), deployer, 1_000_000).await?;
    
    // Wait for the contract deployment to be processed
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    Ok(token)
}

// Run the transaction throughput test
async fn run_tps_test(
    vms: &[Arc<NarwhalBullsharkVm>],
    wallets: &mut [Wallet],
    token: &TokenContract,
    args: &Args
) -> Result<TestResults, Box<dyn Error>> {
    println!("\nRunning transaction throughput test...");
    
    // Choose the primary VM for transaction submission
    let primary_vm = &vms[0];
    
    // Initialize the first wallet with all tokens
    wallets[0].balance = token.total_supply;
    
    // Distribute some tokens to each wallet
    let tokens_per_wallet = token.total_supply / (wallets.len() as u64);
    
    println!("Distributing {} tokens to each wallet...", tokens_per_wallet);
    
    // First, send tokens from wallet[0] to all other wallets
    for i in 1..wallets.len() {
        let tx = token.create_transfer_tx(&mut wallets[0], &wallets[i], tokens_per_wallet);
        primary_vm.submit_transaction(tx).await?;
    }
    
    // Allow some time for the initial distribution to complete
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // Now run the actual TPS test
    println!("Starting TPS test with {} transactions per wallet...", args.transactions);
    
    // Create transaction counter
    let transaction_counter = Arc::new(Mutex::new((0usize, 0usize))); // (submitted, successful)
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
            let (count, _) = *tc_clone.lock().await;
            
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
            
            if elapsed.as_secs() >= args.duration {
                break;
            }
        }
    });
    
    // Start transaction generation
    let sf_clone = stop_flag.clone();
    let tc_clone = transaction_counter.clone();
    let vm_clone = primary_vm.clone();
    let wallets_clone = wallets.to_vec();
    let token_clone = token.clone();
    let batch_size = args.batch_size;
    let total_tx = args.transactions * (wallets.len() - 1);
    
    let generator_handle = tokio::spawn(async move {
        let mut batch_num = 0;
        let mut local_wallets = wallets_clone;
        
        loop {
            // Check if we should stop
            if *sf_clone.lock().await {
                break;
            }
            
            let mut submitted = 0;
            let mut successful = 0;
            
            // Choose random sender and recipient wallets
            for sender_idx in 1..local_wallets.len() {
                // Get a batch of transactions from this sender to other wallets
                let mut handles = Vec::new();
                
                for _ in 0..batch_size.min(total_tx - (batch_num * batch_size)) {
                    let mut sender = local_wallets[sender_idx].clone();
                    
                    // Choose a random recipient (not the sender)
                    let recipient_idx = if sender_idx == 1 { 2 } else { 1 };
                    let recipient = local_wallets[recipient_idx].clone();
                    
                    // Create a token transfer transaction
                    let tx = token_clone.create_transfer_tx(&mut sender, &recipient, 1);
                    
                    // Update the local wallet's nonce
                    local_wallets[sender_idx].nonce = sender.nonce;
                    
                    let vm = vm_clone.clone();
                    
                    // Submit the transaction
                    let handle = tokio::spawn(async move {
                        match vm.submit_transaction(tx).await {
                            Ok(_) => (true, true),  // Submitted and successful
                            Err(_) => (true, false), // Submitted but failed
                        }
                    });
                    
                    handles.push(handle);
                }
                
                // Wait for all submissions to complete
                for handle in handles {
                    if let Ok((submitted_ok, successful_ok)) = handle.await {
                        if submitted_ok {
                            submitted += 1;
                        }
                        if successful_ok {
                            successful += 1;
                        }
                    }
                }
            }
            
            // Update counters
            {
                let mut counter = tc_clone.lock().await;
                counter.0 += submitted;
                counter.1 += successful;
            }
            
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
    
    // Wait for the test duration
    tokio::time::sleep(Duration::from_secs(args.duration)).await;
    
    // Stop transaction generation
    {
        let mut stop = stop_flag.lock().await;
        *stop = true;
    }
    
    // Wait for generator to finish
    let _ = generator_handle.await;
    
    // Wait for metrics to finish
    let _ = metrics_handle.await;
    
    // Collect final statistics
    let (submitted, successful) = *transaction_counter.lock().await;
    let elapsed = Duration::from_secs(args.duration);
    let tps = submitted as f64 / elapsed.as_secs_f64();
    
    // Get TPS from each node
    let mut node_tps = Vec::new();
    for (i, vm) in vms.iter().enumerate() {
        let node_tps_value = vm.get_tps().await;
        node_tps.push((format!("node_{}", i), node_tps_value));
    }
    
    // Compile results
    let results = TestResults {
        nodes: vms.len(),
        wallets: wallets.len(),
        total_transactions: submitted,
        successful_transactions: successful,
        elapsed_time: elapsed,
        tps,
        node_tps,
    };
    
    Ok(results)
}

// Write test results to file
fn write_results(results: &TestResults, output_dir: &str) -> Result<(), Box<dyn Error>> {
    let output_path = Path::new(output_dir).join("narwhal_bullshark_results.txt");
    
    let mut output = String::new();
    
    output.push_str(&format!("DAGKnight Narwhal-Bullshark Consensus Test Results\n"));
    output.push_str(&format!("=================================================\n\n"));
    
    output.push_str(&format!("Test Configuration:\n"));
    output.push_str(&format!("  Nodes: {}\n", results.nodes));
    output.push_str(&format!("  Wallets: {}\n", results.wallets));
    output.push_str(&format!("  Test Duration: {:?}\n\n", results.elapsed_time));
    
    output.push_str(&format!("Performance Results:\n"));
    output.push_str(&format!("  Total Transactions: {}\n", results.total_transactions));
    output.push_str(&format!("  Successful Transactions: {}\n", results.successful_transactions));
    output.push_str(&format!("  Success Rate: {:.2}%\n", 
                             (results.successful_transactions as f64 / results.total_transactions as f64) * 100.0));
    output.push_str(&format!("  Overall TPS: {:.2}\n\n", results.tps));
    
    output.push_str(&format!("Node Performance:\n"));
    for (node, tps) in &results.node_tps {
        output.push_str(&format!("  {}: {:.2} TPS\n", node, tps));
    }
    
    // Write to file
    fs::write(&output_path, output)?;
    
    println!("Results written to {}", output_path.display());
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args = Args::from_args();
    
    println!("DAGKnight Narwhal-Bullshark Consensus Test");
    println!("=========================================");
    
    // Initialize test environment
    let (vms, mut wallets) = init_test_env(&args).await?;
    
    // Deploy the token contract
    let token = run_deploy_test(&vms, &mut wallets).await?;
    
    // Run the TPS test
    let results = run_tps_test(&vms, &mut wallets, &token, &args).await?;
    
    // Display results
    println!("\nTest Results:");
    println!("  Nodes: {}", results.nodes);
    println!("  Wallets: {}", results.wallets);
    println!("  Total Transactions: {}", results.total_transactions);
    println!("  Successful Transactions: {}", results.successful_transactions);
    println!("  Test Duration: {:?}", results.elapsed_time);
    println!("  Overall TPS: {:.2}", results.tps);
    
    println!("\nNode Performance:");
    for (node, tps) in &results.node_tps {
        println!("  {}: {:.2} TPS", node, tps);
    }
    
    // Write results to file
    write_results(&results, &args.output_dir)?;
    
    // Stop all VMs
    println!("\nStopping all nodes...");
    for (i, vm) in vms.iter().enumerate() {
        vm.stop().await?;
        println!("Stopped node_{}", i);
    }
    
    Ok(())
}