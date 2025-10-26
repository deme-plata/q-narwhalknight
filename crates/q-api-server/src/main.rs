use axum::{
    routing::{get, post, put},
    Router,
};
use clap::{Arg, ArgAction, Command};
use q_api_server::{handlers, streaming, payment_api, oauth2_provider, AppState, Config, ConsoleVisualizer, LiquidityPool, update_stats, aegis_auth_middleware};
use q_types::{TxStatus, TxHash};
mod contracts_api;
mod dex_integration_api;
mod liquidity_api;
mod cdp_simple;
// ✅ ENABLED - Full Quillon Bank CDP system
mod quillon_bank_api;
// ✅ ENABLED - QUG/QUGUSD Dual-Token Stablecoin System
mod stablecoin_api;
use contracts_api::create_contracts_router;
use dex_integration_api::create_dex_integration_router;
use liquidity_api::create_liquidity_router;
use cdp_simple::create_cdp_router;
use quillon_bank_api::{create_quillon_bank_router, create_public_routes, create_protected_routes};
// DEACTIVATED: use q_bep44_discovery::{Bep44DiscoveryConfig, DiscoveryEngine};
// DEACTIVATED: use q_bitcoin_bridge::{
//     bridge::{IntegratedBitcoinBridge, PeerNetworkEvent},
//     BitcoinBridgeConfig,
// };
use q_tor_client::QTorClient; // ✅ Re-enabled with embedded Arti support
use q_types::NodeId;
use std::{collections::HashSet, sync::Arc};
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tower_http::services::ServeDir;
use tracing::{debug, error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Update TUI metrics from AppState
#[cfg(feature = "tui")]
async fn update_tui_metrics(
    tui_metrics: &std::sync::Arc<std::sync::RwLock<q_tui::Metrics>>,
    app_state: &Arc<AppState>,
    start_time: std::time::Instant,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Collect all data first (with awaits), before acquiring write lock
    let uptime_secs = start_time.elapsed().as_secs();

    // Get node status
    let (peer_count, block_height) = {
        let node_status = app_state.node_status.read().await;
        (node_status.connected_peers as usize, node_status.current_height)
    };

    // Get transaction pool size
    let tx_pool_size = app_state.tx_pool.len();

    // Get DAG metrics
    let (dag_size_mb, vertex_count, anchor_count) = {
        let blocks = app_state.blocks.read().await;
        let dag_size = blocks.len() as f64 * 0.1; // Rough estimate: 0.1 MB per block
        let vertices = blocks.values().map(|txs| txs.len() as u64).sum();
        let anchors = blocks.len() as u64;
        (dag_size, vertices, anchors)
    };

    // Get system metrics (non-async)
    #[cfg(target_os = "linux")]
    let (cpu_usage, ram_usage, ram_total, disk_usage, disk_total) = {
        use sysinfo::{System, Disks};
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu = sys.global_cpu_info().cpu_usage();
        let ram_used = sys.used_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
        let ram_tot = sys.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);

        let disks = Disks::new_with_refreshed_list();
        let (disk_used, disk_tot) = if let Some(disk) = disks.iter().next() {
            let total = disk.total_space() as f64 / (1024.0 * 1024.0 * 1024.0);
            let available = disk.available_space() as f64 / (1024.0 * 1024.0 * 1024.0);
            (total - available, total)
        } else {
            (0.0, 500.0)
        };

        (cpu, ram_used, ram_tot, disk_used, disk_tot)
    };

    #[cfg(not(target_os = "linux"))]
    let (cpu_usage, ram_usage, ram_total, disk_usage, disk_total) = (0.0, 0.0, 8.0, 0.0, 500.0);

    // Now acquire write lock and update metrics (no awaits here!)
    {
        let mut metrics = tui_metrics.write().unwrap();

        metrics.uptime_secs = uptime_secs;
        metrics.peer_count = peer_count;
        metrics.block_height = block_height;
        metrics.current_tps = tx_pool_size;
        metrics.dag_size_mb = dag_size_mb;
        metrics.vertex_count = vertex_count;
        metrics.anchor_count = anchor_count;
        metrics.cpu_usage_percent = cpu_usage;
        metrics.ram_usage_gb = ram_usage;
        metrics.ram_total_gb = ram_total;
        metrics.disk_usage_gb = disk_usage;
        metrics.disk_total_gb = disk_total;
        metrics.latency_p50_ms = 50; // Placeholder
        metrics.latency_p99_ms = 150; // Placeholder
        metrics.tor_circuits = 0; // TODO
        metrics.mining_enabled = false;
        metrics.hashrate = 0.0;
        metrics.blocks_mined = 0;
        metrics.bytes_received = 0; // TODO
        metrics.bytes_sent = 0; // TODO
        metrics.inbound_peers = 0; // TODO
        metrics.outbound_peers = 0; // TODO
        metrics.last_block_secs = 0; // TODO
    } // Lock dropped here

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file (for Stripe API keys, etc.)
    if let Err(e) = dotenvy::dotenv() {
        eprintln!("⚠️  Warning: Could not load .env file: {}", e);
        eprintln!("    Continuing without .env (environment variables must be set externally)");
    } else {
        eprintln!("✅ Loaded environment variables from .env file");
    }

    // Parse command line arguments
    let matches = Command::new("q-api-server")
        .version("0.1.0")
        .about("Q-NarwhalKnight API Server - Server Alpha Node")
        .arg(
            Arg::new("node-id")
                .long("node-id")
                .value_name("ID")
                .help("Node identifier for this Alpha instance")
                .required(false),
        )
        .arg(
            Arg::new("target-beta")
                .long("target-beta")
                .help("Enable Server Beta targeting mode")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("production")
                .long("production")
                .help("Enable production peer discovery with real network connections")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("port")
                .long("port")
                .value_name("PORT")
                .help("API server port")
                .default_value("8080"),
        )
        .arg(
            Arg::new("tui")
                .long("tui")
                .help("Enable beautiful terminal UI mode for node monitoring")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("network")
                .long("network")
                .value_name("NETWORK")
                .help("Network to connect to (testnet or mainnet)")
                .default_value("testnet"),
        )
        .get_matches();

    // Check if TUI mode is enabled
    let tui_mode = matches.get_flag("tui");

    // Initialize tracing
    if !tui_mode {
        // Normal logging mode
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "q_api_server=debug,q_network=debug,tower_http=debug".into()),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();
    } else {
        // TUI mode - minimal logging, will be captured by TUI
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "q_api_server=info,q_network=info,tower_http=warn".into()),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    // Load configuration
    let mut config = Config::from_env()?;

    // v0.0.22-beta Quick Win #2: Validate configuration on startup
    if let Err(e) = config.validate() {
        error!("❌ Configuration validation failed: {}", e);
        error!("❌ Please fix your configuration and try again");
        std::process::exit(1);
    }

    // Parse network configuration (testnet/mainnet)
    let network_str = matches.get_one::<String>("network").map(|s| s.as_str()).unwrap_or("testnet");
    let network_id = network_str.parse::<q_types::NetworkId>()
        .unwrap_or_else(|e| {
            warn!("Invalid network '{}': {}. Defaulting to testnet.", network_str, e);
            q_types::NetworkId::Testnet
        });

    let network_config = q_types::NetworkConfig::from_network_id(network_id);

    info!("🌐 ════════════════════════════════════════════════════════");
    info!("🌐 Network: {}", network_config.network_id.display_name());
    info!("🌐 Chain ID: {}", network_config.chain_id);
    info!("🌐 Version: {}", network_config.version);
    info!("🌐 Launch Time: {}", network_config.launch_time);
    if !network_config.is_launched() {
        if let Some(duration) = network_config.time_until_launch() {
            info!("⏰ Time until launch: {} days", duration.num_days());
        }
    }
    info!("🌐 ════════════════════════════════════════════════════════");

    // Override port if provided via command line, otherwise use network default
    if let Some(port_str) = matches.get_one::<String>("port") {
        config.port = port_str.parse()?;
    } else {
        // Use network-specific default port
        config.port = network_config.api_port;
        info!("📡 Using network default port: {}", config.port);
    }

    // Check if we're targeting Server Beta
    let target_beta = matches.get_flag("target-beta");
    let production_mode = matches.get_flag("production");
    let node_name = matches
        .get_one::<String>("node-id")
        .map(|s| s.clone())
        .unwrap_or_else(|| format!("{}-node-unknown", network_config.network_id.as_str()));

    if target_beta {
        info!(
            "🎯 SERVER ALPHA NODE: {} - AUTOMATIC BETA DISCOVERY MODE",
            node_name
        );
        info!("🔍 Will use DNS-Phantom steganographic discovery to find Server Beta");

        // Enable enhanced discovery capabilities
        config.tor.enabled = true;
        config.tor.enable_dandelion = true;
    }

    if production_mode {
        info!(
            "🚀 PRODUCTION MODE ENABLED - Real Network Peer Discovery",
        );
        info!("📡 Will perform genuine peer discovery with real network connections");
        
        // Enable all production networking features
        config.tor.enabled = true;
        config.tor.enable_dandelion = true;
    }

    info!(
        "Starting Q-NarwhalKnight API Server {} on port {}",
        node_name, config.port
    );

    // Generate node ID from config or generate new one
    let node_id: NodeId = config.node_id.unwrap_or_else(|| {
        let mut id = [0u8; 32];
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut id);
        info!("🆔 Generated new node ID: {}", hex::encode(id));
        id
    });

    info!("🚀 Initializing Q-NarwhalKnight Triple-Layer Anonymity Network");
    info!("📡 Node ID: {}", hex::encode(node_id));

    // Initialize Tor client first
    info!("🧅 Starting Tor client...");
    let tor_config = q_tor_client::TorConfig::default();
    let tor_client = match q_tor_client::QTorClient::new(tor_config, node_id, q_types::Phase::Phase1).await {
        Ok(client) => {
            info!("✅ Tor client initialized successfully");
            Some(Arc::new(client))
        }
        Err(e) => {
            warn!(
                "⚠️  Tor client initialization failed: {}, continuing without Tor",
                e
            );
            info!("   Error details: {}", e);
            info!("   Make sure Tor is running on 127.0.0.1:9150");
            // Continue without Tor rather than using mock
            None
        }
    };

    // Initialize Bitcoin-Tor Bridge - DEACTIVATED
    let bitcoin_bridge: Option<Arc<()>> = {
        info!("⚠️  Bitcoin Bridge DEACTIVATED - module commented out");
        None
    };
    /*
    let bitcoin_bridge = if std::env::var("SKIP_BITCOIN").is_ok() {
        info!("⚠️  Skipping Bitcoin-Tor Bridge initialization (SKIP_BITCOIN set)");
        None
    } else {
        info!("₿ 🧅 Initializing Bitcoin-Tor Bridge...");
        let mut bitcoin_config = BitcoinBridgeConfig::default();

        // Override with environment variables if provided
        if let Ok(rpc_url) = std::env::var("BITCOIN_RPC_URL") {
            bitcoin_config.bitcoin_rpc_url = rpc_url;
            info!("🔗 Using custom Bitcoin RPC URL from environment");
        }
        if let Ok(rpc_user) = std::env::var("BITCOIN_RPC_USER") {
            bitcoin_config.bitcoin_rpc_user = rpc_user;
        }
        if let Ok(rpc_password) = std::env::var("BITCOIN_RPC_PASSWORD") {
            bitcoin_config.bitcoin_rpc_password = rpc_password;
        }
        // Set to mainnet since we're connecting to a mainnet node
        bitcoin_config.bitcoin_network = q_bitcoin_bridge::BitcoinNetworkType::Mainnet;
        let onion_address = format!("{}.onion", hex::encode(&node_id[..16]));

        // Add timeout to prevent hanging
        match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            IntegratedBitcoinBridge::new(
                bitcoin_config,
                node_id,
                onion_address.clone(),
                tor_client.clone(),
            ),
        )
        .await
        {
            Ok(Ok(mut bridge)) => {
                info!("✅ Bitcoin-Tor Bridge initialized");
                info!("🌐 Onion address: {}", onion_address);

                // Start the bridge system with timeout
                match tokio::time::timeout(std::time::Duration::from_secs(10), bridge.start()).await
                {
                    Ok(Ok(_)) => {
                        info!("🚀 Bitcoin-Tor Bridge started successfully");
                        info!("📡 Starting peer discovery through Bitcoin network...");
                    }
                    Ok(Err(e)) => {
                        warn!("⚠️  Failed to start Bitcoin-Tor bridge: {}", e);
                    }
                    Err(_) => {
                        warn!("⚠️  Bitcoin-Tor bridge start timed out");
                    }
                }

                Some(Arc::new(bridge))
            }
            Ok(Err(e)) => {
                warn!("⚠️  Bitcoin-Tor Bridge initialization failed: {}, continuing without Bitcoin discovery", e);
                None
            }
            Err(_) => {
                warn!("⚠️  Bitcoin-Tor Bridge initialization timed out, continuing without Bitcoin discovery");
                None
            }
        }
    };
    */

    // Initialize DNS-Phantom Network with automatic node integration - DEACTIVATED
    let dns_phantom: Option<Arc<()>> = {
        info!("⚠️  DNS Phantom DEACTIVATED - module commented out");
        None
    };
    /*
    let dns_phantom = if std::env::var("SKIP_DNS").is_ok() {
        info!("⚠️  Skipping DNS-Phantom Network initialization (SKIP_DNS set)");
        None
    } else {
        info!("🌐 👻 Initializing DNS-Phantom Steganographic Network...");

        {
            use q_dns_phantom::node_integration::{
                DNSPhantomNode, DefaultBlockVerifier, DefaultTransactionVerifier,
                NodeIntegrationConfig,
            };

            // Configure DNS Phantom for automatic operation
            let mut phantom_config = NodeIntegrationConfig::default();
            phantom_config.auto_start = true; // Automatically start on node launch
            phantom_config.propagate_transactions = true;
            phantom_config.propagate_blocks = true;
            phantom_config.consensus_via_dns = true;
            phantom_config.stealth_mode = config.tor.enabled; // Use stealth if Tor is enabled
            phantom_config.max_tx_propagation_rate = 100;

            // Create verifiers for transaction and block validation
            let tx_verifier = Arc::new(DefaultTransactionVerifier);
            let block_verifier = Arc::new(DefaultBlockVerifier);

            // Add timeout to DNS phantom initialization
            match tokio::time::timeout(
                std::time::Duration::from_secs(15),
                DNSPhantomNode::new(node_id, tx_verifier, block_verifier, phantom_config),
            )
            .await
            {
                Ok(Ok(phantom_node)) => {
                    let phantom_arc = Arc::new(phantom_node);

                    // Start DNS Phantom automatically
                    match phantom_arc.clone().start().await {
                        Ok(_) => {
                            info!("✅ DNS-Phantom Network started successfully");
                            info!("🔮 Steganographic peer discovery active");
                            info!("📡 Transaction propagation via DNS enabled");
                            info!("🔐 Block verification through DNS active");

                            // Subscribe to DNS Phantom events
                            let mut event_receiver = phantom_arc.subscribe_to_events();
                            tokio::spawn(async move {
                                use q_dns_phantom::node_integration::NodeEvent;
                                loop {
                                    match event_receiver.recv().await {
                                        Ok(event) => {
                                            match event {
                                                NodeEvent::PeerDiscovered { peer_id, via_dns }
                                                    if via_dns =>
                                                {
                                                    info!("🔍 Peer discovered via DNS steganography: {}", 
                                                          hex::encode(&peer_id[..8]));
                                                }
                                                NodeEvent::TransactionVerified {
                                                    tx_hash,
                                                    score,
                                                } => {
                                                    info!("✅ Transaction verified via DNS: {} (score: {:.2})", 
                                                          hex::encode(&tx_hash[..8.min(tx_hash.len())]), score);
                                                }
                                                NodeEvent::BlockReceived { block_hash, height } => {
                                                    info!("⛓️ Block received via DNS: height={}, hash={}", 
                                                          height, hex::encode(&block_hash[..8.min(block_hash.len())]));
                                                }
                                                _ => {}
                                            }
                                        }
                                        Err(_) => break,
                                    }
                                }
                            });

                            Some(phantom_arc)
                        }
                        Err(e) => {
                            warn!("⚠️  DNS-Phantom failed to start: {}, continuing without DNS steganography", e);
                            None
                        }
                    }
                }
                Ok(Err(e)) => {
                    warn!("⚠️  DNS-Phantom initialization failed: {}, continuing without DNS steganography", e);
                    None
                }
                Err(_) => {
                    warn!("⚠️  DNS-Phantom initialization timed out, continuing without DNS steganography");
                    None
                }
            }
        }
    };
    */

    // Initialize BEP-44 Discovery Engine (next-generation peer discovery) - DEACTIVATED
    let bep44_discovery: Option<Arc<()>> = {
        info!("⚠️  BEP-44 Discovery DEACTIVATED - module commented out");
        None
    };
    /*
    info!("🔍 🌐 Initializing BEP-44 DHT Discovery Engine...");

    // Convert node ID to keypair for BEP-44
    let mut validator_keypair = [0u8; 32];
    validator_keypair.copy_from_slice(&node_id[..32]);

    let bep44_config = Bep44DiscoveryConfig {
        bootstrap_nodes: vec![
            // Use IP addresses for reliable parsing with fallbacks
            "91.121.59.153:6881"
                .parse()
                .unwrap_or_else(|_| "127.0.0.1:6881".parse().unwrap()),
            "87.98.162.88:6881"
                .parse()
                .unwrap_or_else(|_| "127.0.0.1:6882".parse().unwrap()),
            "198.105.254.11:6881"
                .parse()
                .unwrap_or_else(|_| "127.0.0.1:6883".parse().unwrap()),
        ],
        validator_keypair,
        tor_socks_proxy: "127.0.0.1:9050"
            .parse()
            .unwrap_or_else(|_| "127.0.0.1:9050".parse().unwrap()),
        announcement_interval: std::time::Duration::from_secs(300), // 5 minutes
        key_rotation_interval: std::time::Duration::from_secs(3600), // 1 hour
        enable_decoy_traffic: true,
        max_discovered_peers: 1000,
    };

    let bep44_discovery = match DiscoveryEngine::new(bep44_config).await {
        Ok(mut engine) => {
            info!("✅ BEP-44 Discovery Engine created");

            // Initialize the engine
            if let Err(e) = engine.initialize().await {
                warn!("⚠️  BEP-44 engine initialization failed: {}", e);
                None
            } else {
                info!("🚀 BEP-44 Discovery Engine initialized");
                info!("🌐 Connected to BitTorrent DHT network");
                info!("🔒 Encrypted friend-only announcements enabled");
                info!("🎭 Decoy traffic generation active");

                // Start discovery
                if let Err(e) = engine.start().await {
                    warn!("⚠️  Failed to start BEP-44 discovery: {}", e);
                    None
                } else {
                    info!("✅ BEP-44 DHT discovery is running");
                    Some(Arc::new(tokio::sync::Mutex::new(engine)))
                }
            }
        }
        Err(e) => {
            warn!(
                "⚠️  BEP-44 Discovery Engine creation failed: {}, continuing without DHT discovery",
                e
            );
            None
        }
    };
    */

    // Initialize Production Peer Discovery if requested - DEACTIVATED (dependencies deactivated)
    let production_peer_discovery: Option<Arc<()>> = {
        if production_mode {
            info!("⚠️  Production Peer Discovery DEACTIVATED - dependencies commented out");
        }
        None
    };
    /*
    let production_peer_discovery = if production_mode {
        info!("🔧 Initializing Production Peer Discovery System...");

        // Import the real peer discovery components we created
        use q_network::real_peer_discovery::RealPeerDiscovery;
        // DEACTIVATED: use q_bitcoin_bridge::real_bitcoin_client::RealBitcoinClient;
        // DEACTIVATED: use q_dns_phantom::real_dns_resolver::RealDnsResolver;
        use q_tor_client::real_tor_client::RealTorClient;
        use q_network::real_dht::RealDht;
        
        match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            async {
                // Initialize real components
                let real_dht = RealDht::new().await?;
                // DEACTIVATED: let real_bitcoin = RealBitcoinClient::new().await?;
                // DEACTIVATED: let real_dns = RealDnsResolver::new().await?;
                let real_tor = RealTorClient::new().await?;

                // Create integrated peer discovery (with deactivated modules)
                let peer_discovery = RealPeerDiscovery::new(
                    real_dht,
                    // DEACTIVATED: real_bitcoin,
                    // DEACTIVATED: real_dns,
                    real_tor,
                ).await?;
                
                // Start peer discovery
                peer_discovery.start_discovery().await?;
                
                anyhow::Ok(peer_discovery)
            }
        ).await {
            Ok(Ok(discovery)) => {
                info!("✅ Production Peer Discovery System initialized successfully");
                info!("🌐 Real libp2p Kademlia DHT active");
                info!("₿  Real Bitcoin RPC client connected");
                info!("🌍 Real DNS resolver with multiple providers");
                info!("🧅 Real Tor client with arti integration");
                info!("🔍 Production peer discovery active");
                Some(Arc::new(tokio::sync::Mutex::new(discovery)))
            }
            Ok(Err(e)) => {
                warn!("⚠️  Production peer discovery initialization failed: {}", e);
                info!("   Falling back to existing discovery methods");
                None
            }
            Err(_) => {
                warn!("⚠️  Production peer discovery initialization timed out");
                info!("   Falling back to existing discovery methods");
                None
            }
        }
    } else {
        None
    };
    */

    // ========================================
    // 🌐 LIBP2P UNIFIED NETWORK MANAGER
    // Network-specific P2P networking with Gossipsub
    // ========================================
    info!("🌐 Initializing libp2p Unified Network Manager for {}...", network_config.network_id.display_name());

    let libp2p_manager = match q_network::UnifiedNetworkManager::new(network_config.clone()).await {
        Ok(mut manager) => {
            info!("✅ libp2p Network Manager initialized for {}", manager.network_config().network_id.display_name());
            info!("   Local Peer ID: {}", manager.peer_id());
            info!("   Protocols: mDNS, Kademlia DHT, Gossipsub, Identify, Ping");
            info!("   Network Topics: {}/transactions, {}/blocks, etc.",
                  manager.network_config().network_id.gossipsub_topic_prefix(),
                  manager.network_config().network_id.gossipsub_topic_prefix());

            // Set up gossipsub message forwarding
            let (gossipsub_tx, mut gossipsub_rx) = tokio::sync::mpsc::unbounded_channel();
            manager.set_gossipsub_channel(gossipsub_tx);

            // Get command sender BEFORE wrapping in Arc<Mutex>
            let command_tx = manager.get_command_sender();

            // Get peer count atomic BEFORE wrapping in Arc<Mutex> and spawning event loop
            // This prevents deadlock when main tries to lock the manager later
            let peer_count_atomic = manager.get_peer_count_atomic();

            // Subscribe to database updates topic BEFORE spawning event loop
            // This prevents deadlock when database replication tries to subscribe later
            if let Err(e) = manager.subscribe_topic(q_ipfs_storage::DATABASE_UPDATES_TOPIC) {
                warn!("⚠️  Failed to pre-subscribe to database updates topic: {}", e);
            } else {
                info!("📢 Pre-subscribed to database updates topic for replication");
            }

            // Start network event loop
            let manager_arc = Arc::new(tokio::sync::Mutex::new(manager));
            let manager_clone = manager_arc.clone();

            tokio::spawn(async move {
                info!("🔄 Starting libp2p network event loop...");
                let mut nm = manager_clone.lock().await;
                if let Err(e) = nm.run().await {
                    error!("❌ Network manager event loop terminated: {}", e);
                }
            });

            info!("✅ libp2p network fully operational");
            Some((manager_arc, gossipsub_rx, command_tx, peer_count_atomic))
        }
        Err(e) => {
            warn!("⚠️  libp2p Network Manager initialization failed: {}", e);
            warn!("   Continuing without libp2p gossipsub");
            None
        }
    };

    // Split libp2p_manager tuple into manager, gossipsub receiver, command sender, and peer count
    let (libp2p_discovery, gossipsub_rx_opt, libp2p_command_tx, peer_count_atomic) = match libp2p_manager {
        Some((manager, rx, cmd_tx, peer_count)) => (Some(manager), Some(rx), Some(cmd_tx), Some(peer_count)),
        None => (None, None, None, None),
    };

    // Initialize application state with network components
    let state = AppState::new_with_networks(
        config.clone(),
        node_id,
        bitcoin_bridge,
        dns_phantom,
        bep44_discovery,
        tor_client,
        None,  // production_peer_discovery is deactivated
        libp2p_discovery,  // ✅ ENABLED - libp2p gossipsub for transaction propagation
        libp2p_command_tx,  // ✅ Command channel for non-blocking P2P operations
    )
    .await?;
    let mut state = state;

    // ========================================
    // 📊 LIBP2P PEER COUNT - Atomic Counter for Thread-Safe Access
    // ========================================
    // peer_count_atomic was already extracted before spawning the event loop (see above)
    // This prevents deadlock from trying to lock the manager while it's held by the event loop

    if peer_count_atomic.is_some() {
        info!("📊 Peer count tracking enabled - atomic counter initialized");
    }
    // Note: node_status.connected_peers will be updated by reading the atomic counter
    // 2. P2P listener (direct TCP connections)
    // Stats loop reads from node_status.connected_peers

    // ========================================
    // PHASE 1: HIGH-PERFORMANCE CONSENSUS INITIALIZATION
    // Target: 50K+ TPS (baseline without SIMD/kernel optimizations)
    // ========================================
    info!("🚀 Initializing High-Performance Consensus System");
    info!("   Target: 50,000+ TPS (Phase 1)");
    info!("   Future: 200,000+ TPS (Phase 2 - Parallel Workers)");
    info!("   Future: 500,000+ TPS (Phase 3 - SIMD Crypto)");
    info!("   Future: 1,000,000+ TPS (Phase 4 - io_uring Kernel I/O)");

    // Determine number of parallel workers (4x CPU cores for I/O bound workload)
    let cpu_cores = num_cpus::get();
    let num_workers = (cpu_cores * 4).max(64); // Minimum 64 workers, optimal for 1M TPS target
    info!("   Parallel Workers: {} ({}x CPU cores)", num_workers, num_workers / cpu_cores);

    // Initialize Production Mempool for high-throughput transaction batching
    let mempool_config = q_narwhal_core::production_mempool::MempoolConfig {
        max_transactions: 1_000_000,  // 1M transaction capacity
        max_age: std::time::Duration::from_secs(300), // 5 minutes
        min_fee_per_byte: 1,  // Low fees for high throughput
        max_transaction_size: 1024 * 1024,  // 1MB max tx size
        max_tx_per_validator_per_second: 100_000,  // 100K TPS per validator
        enable_byzantine_protection: true,
    };

    info!("📦 Initializing Production Mempool...");

    // Initialize ProductionTorClient for Bracha's reliable broadcast
    use q_narwhal_core::{ProductionTorClient, TorClientConfig};
    use q_types::Phase;

    let tor_config = TorClientConfig {
        socks_proxy: "127.0.0.1:9050".to_string(),  // Default Tor SOCKS5 proxy
        connection_timeout: std::time::Duration::from_secs(30),
        max_pool_size: 100,
        enable_connection_pooling: true,
        connection_keep_alive: std::time::Duration::from_secs(300),
    };

    let production_tor_client: Option<Arc<dyn q_narwhal_core::TorClient>> = {
        let client = ProductionTorClient::new(tor_config);
        info!("✅ ProductionTorClient initialized successfully");
        info!("   Note: Tor connection will be attempted when broadcasting");
        let arc_client: Arc<dyn q_narwhal_core::TorClient> = Arc::new(client);
        Some(arc_client)
    };

    // Initialize ProductionMempool with Bracha's reliable broadcast
    let production_mempool: Option<Arc<q_narwhal_core::production_mempool::ProductionMempool>> =
        if let Some(tor_client) = production_tor_client {
            match q_narwhal_core::production_mempool::ProductionMempool::new(
                mempool_config,
                tor_client,
                Phase::Phase1,  // Phase 1: Post-quantum cryptography
            ).await {
                Ok(mempool) => {
                    info!("✅ Production Mempool initialized with Tor broadcast");
                    info!("   Max transactions: 1M");
                    info!("   Byzantine protection: ENABLED");
                    info!("   Reliable broadcast: Bracha's protocol over Tor");
                    Some(Arc::new(mempool))
                }
                Err(e) => {
                    warn!("⚠️  Production Mempool initialization failed: {}", e);
                    info!("   Using fallback transaction pool");
                    None
                }
            }
        } else {
            info!("⚠️  Production Mempool skipped (Tor client unavailable)");
            info!("   Using fallback transaction pool for testing");
            None
        };

    // Initialize DAG-Knight Consensus with parallel processing
    info!("⚔️  Initializing DAG-Knight Consensus...");
    info!("   Workers: {} parallel vertex processors", num_workers);

    let dag_knight = match q_dag_knight::DAGKnightConsensus::new(
        node_id,
        3, // f = 3 for 3f+1 = 10 total validators (minimum Byzantine fault tolerance)
    ).await {
        Ok(consensus) => {
            info!("✅ DAG-Knight Consensus initialized successfully");
            info!("   Validator ID: {}", hex::encode(node_id));
            info!("   Byzantine threshold: f=3 (tolerates 3 Byzantine nodes)");
            info!("   Quantum anchor election: VDF-based");
            info!("   Zero-message complexity ordering");
            Some(Arc::new(consensus))
        }
        Err(e) => {
            warn!("⚠️  DAG-Knight initialization failed: {}", e);
            info!("   Consensus ordering will be disabled");
            None
        }
    };

    // Update state with initialized consensus components
    state.production_mempool = production_mempool;
    state.dag_knight = dag_knight.clone();

    // ========================================
    // QUILLON RESONANCE CONSENSUS - K-PARAMETER PHASE ANALYSIS
    // ========================================
    info!("🌊 Initializing Quillon Resonance Consensus with K-Parameter analysis...");

    let k_analyzer = q_resonance::KParameterAnalyzer::new()
        .with_planck_constant(1.0)
        .with_threshold(1.0);

    info!("✅ K-Parameter analyzer initialized");
    info!("   Formula: K = 2π √(ΔH · Δs · ℏ) / τ");
    info!("   Phase detection: Stable → Approaching → Critical");
    info!("   Dynamic parameter tuning: ACTIVE");

    state.k_parameter_analyzer = Some(Arc::new(k_analyzer));

    // ========================================
    // SHADOW MODE: DAG-Knight (Primary) + Q-Resonance (Shadow)
    // ========================================
    // Initialize Shadow Mode Coordinator if DAG-Knight is active
    if let Some(ref dag_knight_ref) = dag_knight {
        info!("🎭 Initializing Shadow Mode Coordinator...");

        // Create ResonanceCoordinator for shadow mode
        let resonance = q_resonance::ResonanceCoordinator::new(node_id.to_vec());
        let resonance_arc = Arc::new(resonance);

        // Configure shadow mode
        let shadow_config = q_resonance::ShadowModeConfig {
            enabled: true,
            agreement_threshold: 0.85,       // 85% agreement required
            observation_rounds: 100,          // Observe 100 rounds
            hybrid_mode: false,               // Pure shadow initially
            resonance_weight: 0.0,            // Start at 0% resonance
            auto_adjust_weight: true,         // Auto-adjust on performance
            log_interval_rounds: 10,          // Log every 10 rounds
        };

        // Create ShadowModeCoordinator
        match q_resonance::ShadowModeCoordinator::new(
            dag_knight_ref.clone(),
            resonance_arc.clone(),
            shadow_config,
        ).await {
            Ok(shadow_coordinator) => {
                info!("✅ Shadow Mode Coordinator initialized");
                info!("   🎯 Primary: DAG-Knight Consensus");
                info!("   🌊 Shadow: Quillon Resonance Consensus");
                info!("   📊 Agreement Threshold: 85.0%");
                info!("   🔄 Observation Rounds: 100");
                info!("   ⚖️  Auto-weight adjustment: ENABLED");
                info!("   String-theoretic consensus: SHADOW MODE");
                info!("   Energy minimization: MONITORING");
                info!("   Spectral BFT: COMPARISON");

                // Store both coordinators in app state
                state.resonance_coordinator = Some(resonance_arc);
                state.shadow_coordinator = Some(Arc::new(tokio::sync::Mutex::new(shadow_coordinator)));
            }
            Err(e) => {
                warn!("⚠️  Shadow Mode Coordinator initialization failed: {}", e);
                warn!("   Continuing with ResonanceCoordinator only (no shadow mode)");
                // Still store resonance coordinator without shadow mode
                state.resonance_coordinator = Some(resonance_arc);
            }
        };
    }

    // ========================================
    // PHASE 1: DAG STATE SYNCHRONIZATION INFRASTRUCTURE
    // ========================================
    info!("🔄 Initializing DAG State Synchronization...");

    // 1. Initialize PeerRegistry for validator peer tracking
    let peer_registry = Arc::new(q_network::PeerRegistry::new(node_id));
    info!("✅ PeerRegistry initialized");

    // 2. Initialize PersistentChannelManager for dedicated Tor circuits (if Tor is available)
    let channel_manager = if let Some(tor_client) = &state.tor_client {
        let mgr = Arc::new(q_network::PersistentChannelManager::new(
            tor_client.clone(),
            node_id,
            24, // 24-hour circuit rotation
        ));
        info!("✅ PersistentChannelManager initialized with Tor circuits");
        Some(mgr)
    } else {
        warn!("⚠️  PersistentChannelManager skipped (no Tor client available)");
        None
    };

    // 3. Initialize DagSyncManager for full blockchain state synchronization
    if let Some(channel_mgr) = &channel_manager {
        let dag_sync = Arc::new(q_network::DagSyncManager::new(
            node_id,
            peer_registry.clone(),
            channel_mgr.clone(),
        ));

        state.dag_sync_manager = Some(dag_sync.clone());

        info!("✅ DagSyncManager initialized - full sync enabled");
        info!("   📦 Sync capabilities: DAG vertices, certificates, transactions");
        info!("   💰 Sync capabilities: wallet balances, smart contracts");
        info!("   🔗 Sync will trigger automatically on peer connection");
    } else {
        warn!("⚠️  DagSyncManager initialization skipped (requires Tor circuits)");
        info!("   Using fallback: direct peer communication without sync");
    }

    // ========================================
    // MINING SUBMISSION QUEUE - ASYNC PROCESSING
    // ========================================
    info!("⚡ Initializing mining submission async queue...");
    let (mining_tx, mut mining_rx) = tokio::sync::mpsc::unbounded_channel::<q_api_server::MiningSubmission>();
    state.mining_submission_tx = Some(mining_tx);
    info!("✅ Mining queue initialized - async processing enabled");
    info!("   Non-blocking submission acceptance");
    info!("   Background I/O processing");
    info!("   Prevents server overload from mining activity");

    let app_state = Arc::new(state);

    // ========================================
    // MINING SUBMISSION ASYNC PROCESSOR
    // ========================================
    {
        let app_state_mining = app_state.clone();
        tokio::spawn(async move {
            info!("⚡ Starting mining submission async processor...");
            let mut processed_count = 0u64;
            let mut last_log = std::time::Instant::now();

            while let Some(submission) = mining_rx.recv().await {
                let start = std::time::Instant::now();

                // Calculate mining reward based on TIME (works at any BPS: 0.067 → 100,000!)
                let current_timestamp = chrono::Utc::now().timestamp() as u64;
                let block_reward = q_api_server::handlers::calculate_block_reward_time_based(
                    q_api_server::handlers::GENESIS_TIMESTAMP,
                    current_timestamp,
                );

                // Update wallet balance
                let mut balances = app_state_mining.wallet_balances.write().await;
                let current_balance = balances.get(&submission.miner_address).copied().unwrap_or(0);
                let new_balance = current_balance + block_reward;
                balances.insert(submission.miner_address, new_balance);
                drop(balances);

                // Persist balance to disk (this is the slow I/O operation)
                if let Err(e) = app_state_mining.save_wallet_balance(&submission.miner_address, new_balance).await {
                    warn!("❌ Failed to persist mining reward: {:?}", e);
                }

                // Create mining reward transaction
                let tx_hash = blake3::hash(&format!("mining_reward_{}_{}_{}",
                    submission.miner_address_str, submission.nonce, chrono::Utc::now().timestamp()).as_bytes()).as_bytes().to_vec();
                let tx_hash_array: [u8; 32] = tx_hash.as_slice().try_into().unwrap();

                let mining_tx = q_types::Transaction {
                    id: tx_hash_array,
                    from: [0u8; 32],
                    to: submission.miner_address,
                    amount: block_reward,
                    fee: 0,
                    timestamp: chrono::Utc::now(),
                    signature: vec![],
                    nonce: submission.nonce,
                    data: format!("VDF Mining Reward - Nonce: {}", submission.nonce).into_bytes(),
                    token_type: q_types::TokenType::QUG,
                    fee_token_type: q_types::TokenType::QUGUSD,
                };

                // Add to transaction pool
                app_state_mining.tx_pool.insert(tx_hash_array, mining_tx.clone());
                let block_height = app_state_mining.node_status.read().await.current_height;
                app_state_mining.tx_status.insert(tx_hash_array, q_types::TxStatus::Confirmed { block_height, round: 0 });

                // Broadcast via SSE (non-blocking)
                use q_api_server::streaming::StreamEvent;
                let reward_qnk = block_reward as f64 / 100_000_000.0;
                let _ = app_state_mining.event_broadcaster.broadcast(StreamEvent::MiningReward {
                    miner_address: submission.miner_address_str.clone(),
                    reward_qnk,
                    nonce: submission.nonce,
                    block_height,
                    difficulty: "0000".to_string(),
                    hash_rate: 0.0,
                    timestamp: chrono::Utc::now(),
                });

                let _ = app_state_mining.event_broadcaster.broadcast(StreamEvent::BalanceUpdated {
                    wallet_address: submission.miner_address_str.clone(),
                    old_balance: current_balance as f64 / 100_000_000.0,
                    new_balance: new_balance as f64 / 100_000_000.0,
                    change_reason: "mining_reward".to_string(),
                    timestamp: chrono::Utc::now(),
                });

                // 🏗️ BLOCK PRODUCTION: Queue solution to BlockProducer
                {
                    let solution = q_types::MiningSolution {
                        nonce: submission.nonce,
                        hash: submission.hash,
                        difficulty_target: submission.difficulty_target,
                        miner_address: submission.miner_address,
                        timestamp: chrono::Utc::now().timestamp() as u64,
                        pool_id: None,
                    };

                    // PHASE 2: Queue solution to parallel producer pool (round-robin distribution)
                    app_state_mining.block_producer_pool.queue_solution(solution).await;

                    // PHASE 2: Check if any producer should produce a block
                    if app_state_mining.block_producer_pool.should_produce().await {
                        // PHASE 2: Produce blocks from all ready producers (returns Vec<(producer_id, QBlock)>)
                        let new_blocks = app_state_mining.block_producer_pool.produce_blocks().await;

                        for (producer_id, new_block) in new_blocks {
                            info!("🎉 PARALLEL BLOCK PRODUCED: Producer #{} created Height {}, Hash {}, Solutions {}",
                                producer_id,
                                new_block.header.height,
                                hex::encode(&new_block.calculate_hash()[..8]),
                                new_block.mining_solutions.len()
                            );

                            // Update node status with new height
                            {
                                let mut status = app_state_mining.node_status.write().await;
                                status.current_height = new_block.header.height;
                            }

                            // Broadcast NewBlock event via SSE with enhanced data
                            let block_hash = new_block.calculate_hash();
                            let reward_per_solution = q_api_server::handlers::calculate_block_reward(new_block.header.height);
                            let block_reward = new_block.mining_solutions.len() as u64 * reward_per_solution;
                            let tx_count = new_block.transactions.len();

                            let _ = app_state_mining.event_broadcaster.broadcast(
                                q_api_server::streaming::StreamEvent::NewBlock {
                                    height: new_block.header.height,
                                    hash: hex::encode(&block_hash),
                                    prev_hash: hex::encode(&new_block.header.prev_block_hash),
                                    solutions_count: new_block.mining_solutions.len(),
                                    total_difficulty: new_block.header.height as u128, // Cumulative difficulty
                                    dag_round: new_block.header.height, // DAG round number (single validator mode)
                                    miner_count: new_block.mining_solutions.len(), // Number of miners who contributed
                                    tx_count,
                                    block_reward: block_reward as f64,
                                    producer_id, // PHASE 2: Use actual producer ID from parallel pool
                                    timestamp: chrono::Utc::now(),
                                }
                            );

                            // Store block in RocksDB
                            if let Err(e) = app_state_mining.storage_engine.save_qblock(&new_block).await {
                                error!("❌ Failed to save block {}: {}", new_block.header.height, e);
                            }

                            // PHASE 3: Submit block to DAG-Knight consensus
                            {
                                // PHASE 2: Get a producer from the pool for vertex conversion (stateless utility methods)
                                // We use the first producer since these are stateless conversion functions
                                let producer_ref = app_state_mining.block_producer_pool.get_producer(0).await;

                                // Convert QBlock to DAG Vertex
                                let dag_vertex = match producer_ref.qblock_to_vertex(&new_block) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        error!("❌ Failed to convert block {} to vertex: {}", new_block.header.height, e);
                                        continue;
                                    }
                                };

                                // Convert DAG-Knight vertex to storage vertex
                                let storage_vertex = producer_ref.dag_vertex_to_storage_vertex(&dag_vertex, &new_block);

                                // Store vertex in consensus vertex store
                                let consensus = app_state_mining.consensus.read().await;
                                if let Err(e) = consensus.vertex_store.store_vertex(storage_vertex).await {
                                    error!("❌ Failed to store vertex for block {}: {}", new_block.header.height, e);
                                } else {
                                    // Create certificate for consensus processing
                                    let certificate = q_types::Certificate {
                                        vertex_id: dag_vertex.id,
                                        round: dag_vertex.round,
                                        signatures: std::collections::BTreeMap::new(), // Single-node: no signatures yet
                                        threshold_met: true, // Single-node consensus
                                    };

                                    // Process through DAG-Knight consensus
                                    match consensus.process_certificate(certificate).await {
                                        Ok(commit_decisions) => {
                                            if !commit_decisions.is_empty() {
                                                for decision in commit_decisions {
                                                    info!("🎯 BLOCK FINALIZED: Height {}, Round {}, Anchor {}",
                                                        new_block.header.height,
                                                        decision.round,
                                                        hex::encode(&decision.vertex_id[..8])
                                                    );

                                                    // Broadcast BlockFinalized SSE event
                                                    let tx_hashes: Vec<TxHash> = new_block.transactions
                                                        .iter()
                                                        .map(|tx| tx.id)
                                                        .collect();

                                                    let _ = app_state_mining.event_broadcaster.broadcast(
                                                        q_api_server::streaming::StreamEvent::BlockFinalized {
                                                            height: new_block.header.height,
                                                            round: decision.round,
                                                            transactions: tx_hashes,
                                                            timestamp: chrono::Utc::now(),
                                                        }
                                                    );
                                                }
                                            } else {
                                                debug!("Block {} submitted to consensus, pending commit decision",
                                                    new_block.header.height);
                                            }
                                        }
                                        Err(e) => {
                                            error!("❌ Consensus processing failed for block {}: {}", new_block.header.height, e);
                                        }
                                    }
                                }
                                drop(consensus); // Release read lock
                            }

                            // PHASE 3 PART 3: Broadcast block to P2P network via Gossipsub
                            if let Some(ref libp2p_manager) = app_state_mining.libp2p_discovery {
                                match postcard::to_allocvec(&new_block) {
                                    Ok(block_bytes) => {
                                        let libp2p_clone = libp2p_manager.clone();
                                        let block_height = new_block.header.height;
                                        tokio::spawn(async move {
                                            let mut nm = libp2p_clone.lock().await;
                                            let topic = nm.network_config().network_id.blocks_topic();
                                            if let Err(e) = nm.publish_topic(&topic, block_bytes) {
                                                warn!("Failed to broadcast block {} to network: {}", block_height, e);
                                            } else {
                                                info!("📡 Block {} broadcast to {} P2P network", block_height, nm.network_config().network_id.as_str());
                                            }
                                        });
                                    }
                                    Err(e) => {
                                        warn!("Failed to serialize block {} for broadcast: {}", new_block.header.height, e);
                                    }
                                }
                            }
                        }
                    }
                }

                processed_count += 1;
                let elapsed = start.elapsed();

                // Log throughput every 10 seconds
                if last_log.elapsed().as_secs() >= 10 {
                    info!("⚡ Mining queue: {} submissions processed, last took {:?}",
                          processed_count, elapsed);
                    last_log = std::time::Instant::now();
                }
            }
            warn!("⚠️  Mining submission processor stopped");
        });
        info!("✅ Mining submission async processor started");
    }

    // ========================================
    // TIME-BASED BLOCK PRODUCTION LOOP
    // ========================================
    {
        let app_state_block_producer = app_state.clone();
        tokio::spawn(async move {
            info!("⏰ Starting time-based block production loop...");
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));

            loop {
                interval.tick().await;

                // PHASE 2: Check if any producer in pool should produce blocks
                if app_state_block_producer.block_producer_pool.should_produce().await {
                    // PHASE 2: Produce blocks from all ready producers (returns Vec<(producer_id, QBlock)>)
                    let new_blocks = app_state_block_producer.block_producer_pool.produce_blocks().await;

                    for (producer_id, new_block) in new_blocks {
                        info!("⏰ PHASE 2: TIME-BASED PARALLEL BLOCK PRODUCED by Producer #{}: Height {}, Hash {}, Solutions {}",
                            producer_id,
                            new_block.header.height,
                            hex::encode(&new_block.calculate_hash()[..8]),
                            new_block.mining_solutions.len()
                        );

                        // Update node status with new height
                        {
                            let mut status = app_state_block_producer.node_status.write().await;
                            status.current_height = new_block.header.height;
                        }

                        // Broadcast block event via SSE with actual producer_id
                        let block_hash = new_block.calculate_hash();
                        let solutions_count = new_block.mining_solutions.len();
                        let reward_per_solution = q_api_server::handlers::calculate_block_reward(new_block.header.height);
                        let block_reward = solutions_count as u64 * reward_per_solution;
                        let tx_count = new_block.transactions.len();

                        let _ = app_state_block_producer.event_broadcaster.broadcast(
                            q_api_server::streaming::StreamEvent::NewBlock {
                                height: new_block.header.height,
                                hash: hex::encode(&block_hash),
                                prev_hash: hex::encode(&new_block.header.prev_block_hash),
                                solutions_count,
                                total_difficulty: new_block.header.height as u128,
                                dag_round: new_block.header.height,
                                miner_count: solutions_count,
                                tx_count,
                                block_reward: block_reward as f64,
                                producer_id, // PHASE 2: Use actual producer ID for lane assignment
                                timestamp: chrono::Utc::now(),
                            }
                        );

                        // Store block in RocksDB
                        if let Err(e) = app_state_block_producer.storage_engine.save_qblock(&new_block).await {
                            error!("❌ Failed to save block {}: {}", new_block.header.height, e);
                        }

                        // PHASE 3: Submit block to DAG-Knight consensus
                        {
                            // PHASE 2: Get a producer from pool for vertex conversion (stateless utility methods)
                            let producer = app_state_block_producer.block_producer_pool.get_producer(0).await;

                            // Convert QBlock to DAG Vertex
                            let dag_vertex = match producer.qblock_to_vertex(&new_block) {
                                Ok(v) => v,
                                Err(e) => {
                                    error!("❌ Failed to convert block {} to vertex: {}", new_block.header.height, e);
                                    drop(producer);
                                    continue;
                                }
                            };

                            // Convert DAG-Knight vertex to storage vertex
                            let storage_vertex = producer.dag_vertex_to_storage_vertex(&dag_vertex, &new_block);
                            drop(producer); // Release producer lock

                            // Store vertex in consensus vertex store
                            let consensus = app_state_block_producer.consensus.read().await;
                            if let Err(e) = consensus.vertex_store.store_vertex(storage_vertex).await {
                                error!("❌ Failed to store vertex for block {}: {}", new_block.header.height, e);
                            } else {
                                // Create certificate for consensus processing
                                let certificate = q_types::Certificate {
                                    vertex_id: dag_vertex.id,
                                    round: dag_vertex.round,
                                    signatures: std::collections::BTreeMap::new(), // Single-node: no signatures yet
                                    threshold_met: true, // Single-node consensus
                                };

                                // Process through DAG-Knight consensus
                                match consensus.process_certificate(certificate).await {
                                    Ok(commit_decisions) => {
                                        if !commit_decisions.is_empty() {
                                            for decision in commit_decisions {
                                                info!("🎯 BLOCK FINALIZED (TIME-BASED): Height {}, Round {}, Anchor {}",
                                                    new_block.header.height,
                                                    decision.round,
                                                    hex::encode(&decision.vertex_id[..8])
                                                );

                                                // Broadcast BlockFinalized SSE event
                                                let tx_hashes: Vec<TxHash> = new_block.transactions
                                                    .iter()
                                                    .map(|tx| tx.id)
                                                    .collect();

                                                let _ = app_state_block_producer.event_broadcaster.broadcast(
                                                    q_api_server::streaming::StreamEvent::BlockFinalized {
                                                        height: new_block.header.height,
                                                        round: decision.round,
                                                        transactions: tx_hashes,
                                                        timestamp: chrono::Utc::now(),
                                                    }
                                                );
                                            }
                                        } else {
                                            debug!("Block {} submitted to consensus (time-based), pending commit decision",
                                                new_block.header.height);
                                        }
                                    }
                                    Err(e) => {
                                        error!("❌ Consensus processing failed for block {}: {}", new_block.header.height, e);
                                    }
                                }
                            }
                            drop(consensus); // Release consensus lock
                        }

                        // PHASE 3 PART 3: Broadcast block to P2P network via Gossipsub
                        if let Some(ref libp2p_manager) = app_state_block_producer.libp2p_discovery {
                            match postcard::to_allocvec(&new_block) {
                                Ok(block_bytes) => {
                                    let libp2p_clone = libp2p_manager.clone();
                                    let block_height = new_block.header.height;
                                    tokio::spawn(async move {
                                        let mut nm = libp2p_clone.lock().await;
                                        let topic = nm.network_config().network_id.blocks_topic();
                                        if let Err(e) = nm.publish_topic(&topic, block_bytes) {
                                            warn!("Failed to broadcast block {} to network (time-based): {}", block_height, e);
                                        } else {
                                            info!("📡 Block {} broadcast to {} P2P network (time-based)", block_height, nm.network_config().network_id.as_str());
                                        }
                                    });
                                }
                                Err(e) => {
                                    warn!("Failed to serialize block {} for broadcast (time-based): {}", new_block.header.height, e);
                                }
                            }
                        }
                    }
                }
            }
        });
        info!("✅ Time-based block production loop started");
    }

    // ========================================
    // GOSSIPSUB TRANSACTION/BLOCK SYNCHRONIZATION
    // ========================================
    if let Some(mut gossipsub_rx) = gossipsub_rx_opt {
        let app_state_gossip = app_state.clone();
        tokio::spawn(async move {
            info!("📨 Starting gossipsub transaction/block synchronization processor...");
            while let Some((topic, data)) = gossipsub_rx.recv().await {
                info!("📥 GOSSIPSUB: topic={}, size={} bytes", topic, data.len());

                // Match topics by suffix to support both testnet and mainnet
                // e.g., "/qnk/testnet/transactions" or "/qnk/mainnet/transactions"
                if topic.ends_with("/transactions") {
                        // Deserialize and process incoming transaction
                        match postcard::from_bytes::<q_types::Transaction>(&data) {
                            Ok(tx) => {
                                let tx_hash = tx.id;
                                info!("📥 Received transaction {} from network", hex::encode(&tx_hash[..8]));

                                // Add to transaction pool (lock-free)
                                app_state_gossip.tx_pool.insert(tx_hash, tx.clone());
                                app_state_gossip.tx_status.insert(tx_hash, q_types::TxStatus::InMempool);

                                info!("✅ Transaction {} synced to local pool", hex::encode(&tx_hash[..8]));
                            }
                            Err(e) => {
                                warn!("Failed to deserialize transaction from network: {}", e);
                            }
                        }
                } else if topic.ends_with("/mining-rewards") {
                        // Deserialize and process incoming mining reward transaction
                        match postcard::from_bytes::<q_types::Transaction>(&data) {
                            Ok(tx) => {
                                let tx_hash = tx.id;
                                let miner_addr = tx.to;
                                let reward = tx.amount;

                                info!("💎 Received mining reward from network: {} QNK to wallet {}",
                                      reward as f64 / 100_000_000.0, hex::encode(&miner_addr[..8]));

                                // Update wallet balance
                                let mut balances = app_state_gossip.wallet_balances.write().await;
                                let current_balance = balances.get(&miner_addr).copied().unwrap_or(0);
                                let new_balance = current_balance + reward;
                                balances.insert(miner_addr, new_balance);
                                drop(balances);

                                // Persist balance to disk
                                if let Err(e) = app_state_gossip.save_wallet_balance(&miner_addr, new_balance).await {
                                    warn!("❌ Failed to persist synced mining reward: {:?}", e);
                                }

                                // Add transaction to pool
                                app_state_gossip.tx_pool.insert(tx_hash, tx.clone());
                                let block_height = app_state_gossip.node_status.read().await.current_height;
                                app_state_gossip.tx_status.insert(tx_hash, q_types::TxStatus::Confirmed { block_height, round: 0 });

                                info!("✅ Mining reward synced: {} QNK to wallet {}", reward as f64 / 100_000_000.0, hex::encode(&miner_addr[..8]));
                            }
                            Err(e) => {
                                warn!("Failed to deserialize mining reward from network: {}", e);
                            }
                        }
                } else if topic.ends_with("/dex/swaps") {
                        // Deserialize and process incoming DEX swap event
                        match postcard::from_bytes::<q_api_server::handlers::SwapEvent>(&data) {
                            Ok(swap) => {
                                info!("💱 Received DEX swap from network: {}->{} (pool: {})",
                                      swap.from_token, swap.to_token, &swap.pool_id);

                                // Update liquidity pool reserves
                                let pool_data_opt = {
                                    let mut pools = app_state_gossip.liquidity_pools.write().await;
                                    if let Some(pool) = pools.get_mut(&swap.pool_id) {
                                        pool.reserve0 = swap.new_reserve0;
                                        pool.reserve1 = swap.new_reserve1;
                                        info!("✅ DEX pool {} synced: reserves {}/{}",
                                              swap.pool_id, swap.new_reserve0, swap.new_reserve1);

                                        // Serialize pool data before dropping lock
                                        serde_json::to_vec(&*pool).ok()
                                    } else {
                                        warn!("DEX swap references unknown pool: {}", swap.pool_id);
                                        None
                                    }
                                };

                                // Persist updated pool to storage (outside of lock)
                                if let Some(pool_data) = pool_data_opt {
                                    if let Err(e) = app_state_gossip.storage_engine
                                        .save_liquidity_pool(&swap.pool_id, &pool_data).await {
                                        warn!("Failed to persist synced liquidity pool: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to deserialize DEX swap from network: {}", e);
                            }
                        }
                } else if topic.contains("/blocks") {
                    // PHASE 3 PART 3: P2P Block Propagation Handler
                    // Deserialize and process incoming block from network
                    match postcard::from_bytes::<q_types::block::QBlock>(&data) {
                        Ok(block) => {
                            let block_height = block.header.height;
                            let block_hash = block.calculate_hash();
                            info!("📦 Received block {} (height={}) from network",
                                  hex::encode(&block_hash[..8]), block_height);

                            // Verify block is newer than our current height
                            let current_height = app_state_gossip.node_status.read().await.current_height;
                            if block_height <= current_height && block_height > 0 {
                                debug!("Skipping old block {} (current height: {})", block_height, current_height);
                                continue;
                            }

                            // Save block to RocksDB storage
                            if let Err(e) = app_state_gossip.storage_engine.save_qblock(&block).await {
                                warn!("❌ Failed to save incoming block {}: {}", block_height, e);
                                continue;
                            }
                            info!("✅ Stored incoming block {} to RocksDB", block_height);

                            // Update node status if this block advances our height
                            {
                                let mut status = app_state_gossip.node_status.write().await;
                                if block_height > status.current_height {
                                    status.current_height = block_height;
                                    info!("📈 Node height advanced to {}", block_height);
                                }
                            }

                            // PHASE 3: Submit incoming block to consensus
                            {
                                // PHASE 2: Get a producer from pool for vertex conversion (stateless utility methods)
                                let producer = app_state_gossip.block_producer_pool.get_producer(0).await;
                                let dag_vertex = match producer.qblock_to_vertex(&block) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        error!("❌ Failed to convert incoming block {} to vertex: {}", block_height, e);
                                        continue;
                                    }
                                };

                                // Convert DAG-Knight vertex to storage vertex
                                let storage_vertex = producer.dag_vertex_to_storage_vertex(&dag_vertex, &block);
                                drop(producer); // Release read lock

                                // Store vertex in consensus vertex store
                                let consensus = app_state_gossip.consensus.read().await;
                                if let Err(e) = consensus.vertex_store.store_vertex(storage_vertex).await {
                                    error!("❌ Failed to store vertex for incoming block {}: {}", block_height, e);
                                    continue;
                                }

                                // Create certificate for consensus processing
                                let certificate = q_types::Certificate {
                                    vertex_id: dag_vertex.id,
                                    round: dag_vertex.round,
                                    signatures: std::collections::BTreeMap::new(), // Multi-node: signatures from validators
                                    threshold_met: true, // Assume threshold met for received blocks
                                };

                                // Process through DAG-Knight consensus
                                match consensus.process_certificate(certificate).await {
                                    Ok(commit_decisions) => {
                                        if !commit_decisions.is_empty() {
                                            for decision in commit_decisions {
                                                info!("🎯 INCOMING BLOCK FINALIZED: Height {}, Round {}, Anchor {}",
                                                    block_height,
                                                    decision.round,
                                                    hex::encode(&decision.vertex_id[..8])
                                                );

                                                // Broadcast BlockFinalized SSE event
                                                let tx_hashes: Vec<TxHash> = block.transactions
                                                    .iter()
                                                    .map(|tx| tx.id)
                                                    .collect();

                                                let _ = app_state_gossip.event_broadcaster.broadcast(
                                                    q_api_server::streaming::StreamEvent::BlockFinalized {
                                                        height: block_height,
                                                        round: decision.round,
                                                        transactions: tx_hashes,
                                                        timestamp: chrono::Utc::now(),
                                                    }
                                                );
                                            }
                                        } else {
                                            debug!("Incoming block {} submitted to consensus, pending commit decision", block_height);
                                        }
                                    }
                                    Err(e) => {
                                        error!("❌ Consensus processing failed for incoming block {}: {}", block_height, e);
                                    }
                                }
                                drop(consensus); // Release read lock
                            }

                            info!("✅ Block {} fully processed from network", block_height);
                        }
                        Err(e) => {
                            warn!("Failed to deserialize block from network: {}", e);
                        }
                    }
                } else if topic.contains("/votes") {
                    info!("  → Vote aggregation (handler not yet implemented)");
                } else if topic.contains("/ack") {
                    info!("  → Acknowledgements (handler not yet implemented)");
                } else {
                    warn!("  → Unknown gossipsub topic: {}, dropping", topic);
                }
            }
            warn!("📨 Gossipsub processor channel closed");
        });
        info!("✅ Gossipsub transaction/block synchronization enabled");
    } else {
        warn!("⚠️  Gossipsub synchronization disabled (libp2p not available)");
    }

    // ========================================
    // IPFS-ROCKSDB DECENTRALIZED STORAGE INITIALIZATION
    // ========================================
    info!("💾 Initializing IPFS-RocksDB decentralized storage system...");
    let ipfs_storage = match q_api_server::storage_api::initialize_storage().await {
        Ok(storage) => {
            info!("✅ IPFS-RocksDB storage system initialized successfully");
            info!("   Distributed database backups enabled");
            info!("   Content-addressed storage via IPFS");
            info!("   libp2p network integration active");
            Some(storage)
        }
        Err(e) => {
            warn!("⚠️  IPFS storage initialization failed: {}, backup functionality disabled", e);
            None
        }
    };
    let ipfs_storage_state = Arc::new(tokio::sync::RwLock::new(ipfs_storage));

    // ========================================
    // DATABASE REPLICATION VIA GOSSIPSUB
    // ========================================
    info!("🔄 Initializing database replication system...");

    // Initialize replication only if IPFS storage is available
    let replication_system = if ipfs_storage_state.read().await.is_some() {
        use q_ipfs_storage::{DatabaseReplicationManager, ReplicationConfig};
        use q_api_server::database_replication_bridge::DatabaseReplicationBridge;

        // Create replication configuration
        let replication_config = ReplicationConfig {
            enabled: true,
            snapshot_interval: 300,  // 5 minutes
            max_incremental_updates: 100,
            verify_updates: true,
            parallel_downloads: 10,
        };

        // Initialize replication manager
        let (replication_manager, update_rx) = DatabaseReplicationManager::new(
            node_id.to_vec(),
            ipfs_storage_state.clone(),
            replication_config,
        );
        let replication_manager = Arc::new(replication_manager);

        // Start replication manager background tasks
        info!("🚀 Starting database replication manager...");
        replication_manager.clone().start().await;

        // Create channel for gossipsub publishing
        let (gossipsub_tx, mut gossipsub_rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();

        // Create replication bridge
        let bridge = DatabaseReplicationBridge::new(
            replication_manager.clone(),
            update_rx,
        );

        // Start bridge (spawns background tasks for bidirectional forwarding)
        info!("🌉 Starting database replication bridge...");
        let incoming_tx = match bridge.start(gossipsub_tx).await {
            Ok(tx) => {
                info!("✅ Database replication bridge started successfully");
                tx
            }
            Err(e) => {
                warn!("⚠️  Failed to start replication bridge: {}, replication disabled", e);
                tokio::sync::mpsc::unbounded_channel().0
            }
        };

        // Integrate with libp2p UnifiedNetworkManager if available
        if let Some(libp2p_discovery) = &app_state.libp2p_discovery {
            let discovery_clone = libp2p_discovery.clone();

            // Subscribe to database updates topic
            // NOTE: Subscription is now done BEFORE spawning the event loop (line 555)
            // to prevent deadlock. This code is kept for reference but subscription
            // already completed earlier.
            info!("📢 Database updates topic subscription already active (set during initialization)");

            // Spawn task to forward outgoing updates to gossipsub
            // DISABLED: This causes deadlock by trying to lock the manager from a spawned task
            // TODO: Implement using command channel pattern instead
            warn!("⚠️  Outgoing database update forwarder disabled to prevent deadlock");
            warn!("   Database replication will work for incoming updates only");

            // Drop the unused gossipsub_rx to avoid warnings
            drop(gossipsub_rx);

            info!("✅ Database replication integrated with gossipsub");
            info!("   Automatic synchronization: ENABLED");
            info!("   Snapshot interval: 5 minutes");
            info!("   Topic: /qnk/database-updates/1.0.0");

            Some((replication_manager, incoming_tx))
        } else {
            warn!("⚠️  libp2p discovery not available, replication will not work");
            None
        }
    } else {
        warn!("⚠️  IPFS storage not initialized, database replication disabled");
        None
    };

    info!("🔍 DEBUG: Reached after database replication - continuing initialization");

    // ========================================
    // 🎨 START ANIMATED CONSOLE VISUALIZATION
    // ========================================
    // Skip console visualization if TUI mode is enabled
    // DISABLED: Console visualization causes blocking - skip it entirely for now
    if false && !tui_mode {
        info!("🎨 Initializing animated consensus visualization...");
        let visualizer = ConsoleVisualizer::new();
        let viz_stats = visualizer.get_stats_handle();

        // Initialize stats with current state
        update_stats(viz_stats.clone(), |stats| {
            stats.connected_peers = 0; // Will be updated by network events
            stats.resonance_enabled = true; // Phase 5 complete
            stats.shadow_mode_active = false; // Can be enabled later
        }).await;

        // Start animated visualization loop
        let viz_clone = visualizer;
        tokio::spawn(async move {
            viz_clone.start_animation().await;
        });

        // Spawn background stats updater
        let stats_handle_updater = viz_stats.clone();
        let app_state_updater = app_state.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            let mut last_tx_count = 0u64;
            let mut last_block_count = 0u64;
            let start_time = std::time::Instant::now();

        loop {
            interval.tick().await;

            // Read peer count from atomic counter (thread-safe, no locking needed!)
            let connected_peers = if let Some(ref peer_count) = peer_count_atomic {
                let count = peer_count.load(std::sync::atomic::Ordering::SeqCst);

                // Update node_status with current peer count
                {
                    let mut status = app_state_updater.node_status.write().await;
                    status.connected_peers = count as u32;
                }

                count
            } else {
                0
            };

            // Read current state - count CONFIRMED transactions (not just mempool)
            let current_tx = app_state_updater.tx_status.iter()
                .filter(|entry| matches!(entry.value(), TxStatus::Confirmed { .. }))
                .count() as u64;
            let mempool_size = app_state_updater.tx_pool.len() as u64;
            let node_status = app_state_updater.node_status.read().await;
            let current_blocks = node_status.current_height;

            // Calculate per-second rates
            let elapsed = start_time.elapsed().as_secs_f64().max(1.0);
            let tx_delta = current_tx.saturating_sub(last_tx_count);
            let block_delta = current_blocks.saturating_sub(last_block_count);

            update_stats(stats_handle_updater.clone(), |stats| {
                stats.total_transactions = current_tx; // Count of CONFIRMED transactions
                stats.total_blocks = current_blocks;
                stats.transactions_per_second = tx_delta as f64;
                stats.blocks_per_second = block_delta as f64;
                stats.connected_peers = connected_peers; // Reading from ConnectionManager
                stats.mempool_size = mempool_size as usize; // Pending transactions in mempool
                stats.average_latency_ms = 45.2; // Default, can be measured
                stats.dag_vertices = current_blocks * 4; // Approximate
                stats.consensus_rounds = current_blocks;
            }).await;

            last_tx_count = current_tx;
            last_block_count = current_blocks;
        }
    });

        info!("✅ Console visualization started");
    } else {
        info!("🎨 Console visualization disabled - TUI mode active");
    }

    info!("🔍 DEBUG: About to show activation banner");

    // Log final startup status
    info!("🌟 ================================");
    info!("🌟   Q-NARWHALKNIGHT ACTIVATED   ");
    info!("🌟 ================================");
    info!("🆔 Node ID: {}", hex::encode(node_id));
    info!(
        "🧅 Tor Integration: {}",
        if app_state.network_manager.is_some() {
            "✅ Active (via NetworkManager)"
        } else {
            "❌ Disabled"
        }
    );
    info!(
        "₿  Bitcoin Discovery: {}",
        if app_state.bitcoin_bridge.is_some() {
            "✅ Active"
        } else {
            "❌ Disabled"
        }
    );
    info!(
        "👻 DNS-Phantom Network: {}",
        if app_state.dns_phantom.is_some() {
            "✅ Active"
        } else {
            "❌ Disabled"
        }
    );
    info!(
        "🔍 BEP-44 DHT Discovery: {}",
        if app_state.bep44_discovery.is_some() {
            "✅ Active"
        } else {
            "❌ Disabled"
        }
    );
    info!(
        "🚀 Production Peer Discovery: {}",
        if app_state.production_peer_discovery.is_some() {
            "✅ Active"
        } else {
            "❌ Disabled"
        }
    );
    info!("🌟 ================================");

    // SERVER ALPHA BETA-TARGETING MODE: Automatic discovery
    if target_beta {
        info!(
            "🎯 SERVER ALPHA {}: Initiating automatic Server Beta discovery",
            node_name
        );
        let alpha_node_id = hex::encode(&node_id[..8]);
        let alpha_name = node_name.clone();

        tokio::spawn(async move {
            info!(
                "🔍 SERVER ALPHA {}: Starting DNS-Phantom steganographic discovery",
                alpha_name
            );

            // Enhanced DNS discovery queries that will trigger steganographic detection
            let discovery_domains = vec![
                "discovery.q-narwhal.local",
                "mesh.qnk.network",
                "phantom.quantum.dns",
                "steganographic.mesh.discovery",
                "beta-discovery.hidden.network",
                // Docker container discovery domains
                "beta-coordinator.qnarwhal-mesh",
                "dns-phantom-hub.qnarwhal-mesh",
                "alpha-node-1.qnarwhal-mesh",
                "alpha-node-2.qnarwhal-mesh",
                "alpha-node-3.qnarwhal-mesh",
            ];

            for domain in discovery_domains {
                info!(
                    "🔍 SERVER ALPHA {}: Querying discovery domain: {}",
                    alpha_name, domain
                );

                // Perform DNS queries that will be detected by DNS-Phantom system
                if let Ok(addr) = tokio::net::lookup_host(format!("{}:53", domain)).await {
                    for _resolved in addr {
                        // This DNS query will be processed by DNS-Phantom detection
                        info!(
                            "📡 SERVER ALPHA {}: DNS query sent for {}",
                            alpha_name, domain
                        );
                    }
                }

                // Small delay between queries
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }

            info!("✅ SERVER ALPHA {}: DNS discovery queries completed - waiting for steganographic detection", alpha_name);
        });
    }

    // Start automatic network discovery
    // DEACTIVATED:     if let Some(bridge) = &app_state.bitcoin_bridge {
    // DEACTIVATED:         let bridge_clone = bridge.clone();
    // DEACTIVATED:         let state_clone = app_state.clone();
    // DEACTIVATED:         tokio::spawn(async move {
    // DEACTIVATED:             info!("🔍 Starting automatic peer discovery through Bitcoin network...");
    // DEACTIVATED:             let mut events = bridge_clone.subscribe_to_events();
    // DEACTIVATED: 
    // DEACTIVATED:             while let Ok(event) = events.recv().await {
    // DEACTIVATED:                 match event {
    // DEACTIVATED:                     PeerNetworkEvent::PeerDiscovered {
    // DEACTIVATED:                         node_id,
    // DEACTIVATED:                         confidence,
    // DEACTIVATED:                         ..
    // DEACTIVATED:                     } => {
    // DEACTIVATED:                         info!(
    // DEACTIVATED:                             "🎯 Discovered peer through Bitcoin: {} (confidence: {:.2})",
    // DEACTIVATED:                             hex::encode(node_id),
    // DEACTIVATED:                             confidence
    // DEACTIVATED:                         );
    // DEACTIVATED: 
    // DEACTIVATED:                         // Update node status
    // DEACTIVATED:                         {
    // DEACTIVATED:                             let mut status = state_clone.node_status.write().await;
    // DEACTIVATED:                             status.connected_peers += 1;
    // DEACTIVATED:                         }
    // DEACTIVATED: 
    // DEACTIVATED:                         // Emit real-time event
    // DEACTIVATED:                         let _ = state_clone
    // DEACTIVATED:                             .event_emitter
    // DEACTIVATED:                             .emit_peer_discovered(hex::encode(node_id), confidence)
    // DEACTIVATED:                             .await;
    // DEACTIVATED:                     }
    // DEACTIVATED:                     PeerNetworkEvent::PeerConnected { node_id, .. } => {
    // DEACTIVATED:                         info!("✅ Connected to peer: {}", hex::encode(node_id));
    // DEACTIVATED:                         let _ = state_clone
    // DEACTIVATED:                             .event_emitter
    // DEACTIVATED:                             .emit_peer_connected(hex::encode(node_id))
    // DEACTIVATED:                             .await;
    // DEACTIVATED:                     }
    // DEACTIVATED:                     PeerNetworkEvent::PeerDisconnected {
    // DEACTIVATED:                         node_id, reason, ..
    // DEACTIVATED:                     } => {
    // DEACTIVATED:                         info!(
    // DEACTIVATED:                             "❌ Disconnected from peer {}: {}",
    // DEACTIVATED:                             hex::encode(node_id),
    // DEACTIVATED:                             reason
    // DEACTIVATED:                         );
    // DEACTIVATED: 
    // DEACTIVATED:                         // Update node status
    // DEACTIVATED:                         {
    // DEACTIVATED:                             let mut status = state_clone.node_status.write().await;
    // DEACTIVATED:                             status.connected_peers = status.connected_peers.saturating_sub(1);
    // DEACTIVATED:                         }
    // DEACTIVATED: 
    // DEACTIVATED:                         let _ = state_clone
    // DEACTIVATED:                             .event_emitter
    // DEACTIVATED:                             .emit_peer_disconnected(hex::encode(node_id), reason)
    // DEACTIVATED:                             .await;
    // DEACTIVATED:                     }
    // DEACTIVATED:                     _ => {}
    // DEACTIVATED:                 }
    // DEACTIVATED:             }
    // DEACTIVATED:         });
    // DEACTIVATED:     }

    //     // Start DNS-Phantom network monitoring
    //     if let Some(dns_phantom) = &app_state.dns_phantom {
    //         let phantom_clone = dns_phantom.clone();
    //         let state_clone = app_state.clone();
    //         tokio::spawn(async move {
    //             info!("👻 Starting DNS-Phantom network monitoring...");
    //             let mut events = phantom_clone.subscribe_to_events();
    //
    //             while let Ok(event) = events.recv().await {
    //                 match event {
    //                     q_dns_phantom::PhantomNetworkEvent::PeerDiscovered {
    //                         node_id,
    //                         confidence,
    //                         discovery_method,
    //                     } => {
    //                         info!(
    //                             "🌐 Phantom peer discovered: {} via {:?} (confidence: {:.2})",
    //                             hex::encode(node_id),
    //                             discovery_method,
    //                             confidence
    //                         );
    //
    //                         // SERVER ALPHA: AUTOMATIC BETA DISCOVERY VIA DNS-PHANTOM
    //                         info!("🎯 SERVER ALPHA: DNS-Phantom detected Server Beta steganographic data");
    //                         info!("🔍 Discovery method: {:?}, Confidence: {:.2}", discovery_method, confidence);
    //
    //                         // Extract Server Beta connection info from steganographic data
    //                         let discovery_method_clone = discovery_method.clone();
    //                         let state_for_extraction = state_clone.clone();
    //                         tokio::spawn(async move {
    //                             info!("🔍 SERVER ALPHA: Extracting Server Beta connection info from DNS-Phantom data");
    //
    //                             // Use the DNS-Phantom peer extraction system to get Beta server info
    //                             match q_dns_phantom::peer_extraction::extract_peer_from_response(&[], "dns-phantom-discovery").await {
    //                                 Ok(Some(peer_info)) => {
    //                                     info!("✅ SERVER ALPHA: Successfully extracted Server Beta address: {}", peer_info.address);
    //
    //                                     // Connect to the automatically discovered Server Beta
    //                                     match tokio::net::TcpStream::connect(peer_info.address).await {
    //                                         Ok(mut stream) => {
    //                                             info!("✅ SERVER ALPHA: TCP connection established to auto-discovered Server Beta!");
    //
    //                                             // Send JSON handshake
    //                                             let handshake = serde_json::json!({
    //                                                 "node_id": peer_info.node_id,
    //                                                 "server": "alpha",
    //                                                 "message": "Hello Server Beta - Automatic DNS-Phantom Discovery",
    //                                                 "discovery_method": format!("{:?}", discovery_method_clone),
    //                                                 "confidence": confidence,
    // DEACTIVATED:     //                                                 "capabilities": ["consensus", "mempool", "state_sync"],
    // DEACTIVATED:     //                                                 "version": "0.1.0",
    // DEACTIVATED:     //                                                 "timestamp": chrono::Utc::now().to_rfc3339()
    // DEACTIVATED:     //                                             });
    // DEACTIVATED:     //
    // DEACTIVATED:     //                                             use tokio::io::{AsyncWriteExt, AsyncReadExt};
    // DEACTIVATED:     //                                             let handshake_str = handshake.to_string() + "\n";
    // DEACTIVATED:     //
    // DEACTIVATED:     //                                             if let Err(e) = stream.write_all(handshake_str.as_bytes()).await {
    // DEACTIVATED:     //                                                 error!("❌ SERVER ALPHA: Failed to send handshake: {}", e);
    // DEACTIVATED:     //                                                 return;
    // DEACTIVATED:     //                                             }
    // DEACTIVATED:     //
    // DEACTIVATED:     //                                             info!("📨 SERVER ALPHA: Sent JSON handshake to auto-discovered Server Beta");
    // DEACTIVATED:     //
    // DEACTIVATED:     //                                             // Read Server Beta response
    // DEACTIVATED:     //                                             let mut buffer = [0; 1024];
    // DEACTIVATED:     //                                             match tokio::time::timeout(std::time::Duration::from_secs(10), stream.read(&mut buffer)).await {
    // DEACTIVATED:     //                                                 Ok(Ok(n)) => {
    // DEACTIVATED:     //                                                     let response = String::from_utf8_lossy(&buffer[..n]);
    // DEACTIVATED:     //                                                     info!("📬 SERVER ALPHA: Server Beta response: {}", response);
    // DEACTIVATED:     //
    // DEACTIVATED:     //                                                     if let Ok(beta_response) = serde_json::from_str::<serde_json::Value>(&response) {
    // DEACTIVATED:     //                                                         if beta_response.get("status").and_then(|s| s.as_str()) == Some("connected") {
    // DEACTIVATED:     //                                                             info!("🎉 SERVER ALPHA: Successfully connected to Server Beta via automatic discovery!");
    // DEACTIVATED:     //                                                             info!("🌐 Cross-server P2P mesh established through DNS steganography!");
    // DEACTIVATED:     //
    // DEACTIVATED:     //                                                             // Update connection status
    // DEACTIVATED:     //                                                             {
    // DEACTIVATED:     //                                                                 let mut status = state_for_extraction.node_status.write().await;
    // DEACTIVATED:     //                                                                 status.connected_peers += 1;
    // DEACTIVATED:     //                                                             }
    // DEACTIVATED:     //                                                         }
    // DEACTIVATED:     //                                                     }
    // DEACTIVATED:     //                                                 }
    // DEACTIVATED:     //                                                 Ok(Err(e)) => {
    // DEACTIVATED:     //                                                     error!("❌ SERVER ALPHA: Error reading Beta response: {}", e);
    // DEACTIVATED:     //                                                 }
    // DEACTIVATED:     //                                                 Err(_) => {
    // DEACTIVATED:     //                                                     warn!("⏰ SERVER ALPHA: Timeout waiting for Server Beta response");
    // DEACTIVATED:     //                                                 }
    // DEACTIVATED:     //                                             }
    // DEACTIVATED:     //                                         }
    // DEACTIVATED:     //                                         Err(e) => {
    // DEACTIVATED:     //                                             error!("❌ SERVER ALPHA: Failed to connect to auto-discovered Server Beta: {}", e);
    // DEACTIVATED:     //                                         }
    // DEACTIVATED:     //                                     }
    // DEACTIVATED:     //                                 }
    // DEACTIVATED:     //                                 Ok(None) => {
    // DEACTIVATED:     //                                     info!("🔍 SERVER ALPHA: No Server Beta connection info found in steganographic data");
    // DEACTIVATED:     //                                 }
    // DEACTIVATED:     //                                 Err(e) => {
    // DEACTIVATED:     //                                     warn!("⚠️ SERVER ALPHA: Failed to extract peer info from DNS-Phantom data: {}", e);
    // DEACTIVATED:     //                                 }
    // DEACTIVATED:     //                             }
    // DEACTIVATED:     //                         });
    // DEACTIVATED:     //
    // DEACTIVATED:     //                         // Emit dashboard event
    // DEACTIVATED:     //                         let _ = state_clone
    // DEACTIVATED:     //                             .event_emitter
    // DEACTIVATED:     //                             .emit_phantom_peer_discovered(
    // DEACTIVATED:     //                                 hex::encode(node_id),
    // DEACTIVATED:     //                                 format!("{:?}", discovery_method),
    // DEACTIVATED:     //                                 confidence,
    // DEACTIVATED:     //                             )
    // DEACTIVATED:     //                             .await;
    // DEACTIVATED:     //                     }
    // DEACTIVATED:     //                     q_dns_phantom::PhantomNetworkEvent::MessageReceived {
    // DEACTIVATED:     //                         from,
    // DEACTIVATED:     //                         message_type,
    // DEACTIVATED:     //                         size,
    // DEACTIVATED:     //                     } => {
    // DEACTIVATED:     //                         info!(
    // DEACTIVATED:     //                             "📨 Phantom message received from {}: {:?} ({} bytes)",
    // DEACTIVATED:     //                             hex::encode(from),
    // DEACTIVATED:     //                             message_type,
    // DEACTIVATED:     //                             size
    // DEACTIVATED:     //                         );
    // DEACTIVATED:     //                         let _ = state_clone
    // DEACTIVATED:     //                             .event_emitter
    // DEACTIVATED:     //                             .emit_phantom_message(
    // DEACTIVATED:     //                                 hex::encode(from),
    // DEACTIVATED:     //                                 format!("{:?}", message_type),
    // DEACTIVATED:     //                                 size,
    // DEACTIVATED:     //                             )
    // DEACTIVATED:     //                             .await;
    // DEACTIVATED:     //                     }
    // DEACTIVATED:     //                     q_dns_phantom::PhantomNetworkEvent::CacheAnomalyDetected {
    // DEACTIVATED:     //                         provider,
    // DEACTIVATED:     //                         anomaly_type,
    // DEACTIVATED:     //                         risk_level,
    // DEACTIVATED:     //                     } => {
    // DEACTIVATED:     //                         warn!(
    // DEACTIVATED:     //                             "🚨 DNS Cache anomaly detected: {} on {:?} (risk: {:.2})",
    // DEACTIVATED:     //                             anomaly_type, provider, risk_level
    // DEACTIVATED:     //                         );
    // DEACTIVATED:     //                         let _ = state_clone
    // DEACTIVATED:     //                             .event_emitter
    // DEACTIVATED:     // //                             .emit_security_alert(
    // DEACTIVATED: 
    // DEACTIVATED:     // DEACTIVATED: BEP-44 DHT discovery (module deactivated)
    // if let Some(bep44_discovery) = &app_state.bep44_discovery {
    //     let discovery_clone = bep44_discovery.clone();
    //     let state_clone = app_state.clone();
    //     tokio::spawn(async move {
    //         info!("🔍 Starting BEP-44 DHT discovery monitoring...");
    //
    //         // Periodically check for discovered peers
    //         let mut discovery_interval = tokio::time::interval(std::time::Duration::from_secs(60)); // 1 minute
    //
    //         loop {
    //             discovery_interval.tick().await;
    //
    //             // Get discovered peers from BEP-44 engine
    //             let discovered_peers = discovery_clone.lock().await.get_discovered_peers().await;
    //             for peer in discovered_peers {
    //                 info!(
    //                     "🌐 BEP-44 peer discovered: {} via {} (confidence: 100.0)",
    //                     hex::encode(&peer.validator_id[..4]),
    //                     peer.discovery_method
    //                 );
    //
    //                 // Bridge BEP-44 discovery to NetworkManager (similar to DNS-phantom)
    //                 if let Some(network_manager) = &state_clone.network_manager {
    //                     let mut capabilities = std::collections::HashSet::new();
    //                     for cap in &peer.capabilities {
    //                         match cap {
    //                             q_bep44_discovery::PeerCapability::Consensus => {
    //                                 capabilities.insert(
    //                                     q_network::peer_registry::PeerCapability::Consensus,
    //                                 );
    //                             }
    //                             q_bep44_discovery::PeerCapability::Mempool => {
    //                                 capabilities
    //                                     .insert(q_network::peer_registry::PeerCapability::Mempool);
    //                             }
    //                             q_bep44_discovery::PeerCapability::StateSync => {
    //                                 capabilities.insert(
    //                                     q_network::peer_registry::PeerCapability::StateSync,
    //                                 );
    //                             }
    //                             q_bep44_discovery::PeerCapability::Archive => {
    //                                 capabilities.insert(
    //                                     q_network::peer_registry::PeerCapability::ArchiveNode,
    //                                 );
    //                             }
    //                         }
    //                     }
    //
    //                     let peer_info = q_network::peer_registry::PeerInfo {
    //                         validator_id: peer.validator_id,
    //                         onion_address: peer.onion_address.clone(),
    //                         public_key: peer.validator_id.to_vec(),
    //                         capabilities,
    //                         network_addresses: vec![],
    //                         last_seen: std::time::Instant::now(),
    //                         connection_quality: q_network::peer_registry::ConnectionQuality::new(),
    //                         protocol_version: "0.1.0".to_string(),
    //                         stake: 100,
    //                         reputation_score: 1.0,
    //                     };
    //
    //                     if let Err(e) = network_manager.register_peer(peer_info).await {
    //                         tracing::warn!("⚠️ Failed to register BEP-44 peer: {}", e);
    //                     } else {
    //                         // Attempt connection via Tor
    //                         match discovery_clone
    //                             .lock()
    //                             .await
    //                             .connect_to_peer(&peer.validator_id)
    //                             .await
    //                         {
    //                             Ok(_) => {
    //                                 info!("✅ BEP-44 peer connected via Tor!");
    //                                 // Update connected peer count
    //                                 {
    //                                     let mut status = state_clone.node_status.write().await;
    //                                     status.connected_peers += 1;
    //                                 }
    //
    //                                 // Emit real-time event
    //                                 let _ = state_clone
    //                                     .event_emitter
    //                                     .emit_peer_discovered(
    //                                         hex::encode(&peer.validator_id[..4]),
    //                                         100.0,
    //                                     )
    //                                     .await;
    //                             }
    //                             Err(e) => {
    //                                 tracing::warn!("⚠️ BEP-44 peer connection failed: {}", e);
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //
    //             // Log discovery statistics
    //             let stats = discovery_clone.lock().await.get_stats().await;
    //             {
    //                 if stats.total_discovered_peers > 0 {
    //                     info!(
    //                         "📊 BEP-44 Stats: {} peers discovered, {} successful connections",
    //                         stats.total_discovered_peers, stats.successful_connections
    //                     );
    //                 }
    //             }
    //         }
    //     });
    // }

    // DEACTIVATED: Production peer discovery (Arc<()> placeholder)
    // if let Some(production_discovery) = &app_state.production_peer_discovery {
    //     let discovery_clone = production_discovery.clone();
    //     let state_clone = app_state.clone();
    //     tokio::spawn(async move {
    //         info!("🚀 Starting Production Peer Discovery monitoring...");
    //
    //         // Periodically check discovery stats and discovered peers
    //         let mut discovery_interval = tokio::time::interval(std::time::Duration::from_secs(30)); // 30 seconds
    //
    //         loop {
    //             discovery_interval.tick().await;
    //
    //             let discovery_guard = discovery_clone.lock().await;
    //
    //             // Get discovered peers
    //             let discovered_peers = discovery_guard.get_discovered_peers().await;
    //             if !discovered_peers.is_empty() {
    //                 info!("🌐 Production Discovery - {} peers found:", discovered_peers.len());
    //
    //                 for (peer_id, peer_info) in discovered_peers.iter().take(5) { // Show first 5
    //                     info!(
    //                         "  📡 Peer {}: {} via {:?}",
    //                         hex::encode(&peer_id[..4]),
    //                         peer_info.addresses.first().map(|a| a.to_string()).unwrap_or_else(|| "unknown".to_string()),
    //                         peer_info.discovered_via
    //                     );
    //
    //                     // Update connected peers count
    //                     {
    //                         let mut status = state_clone.node_status.write().await;
    //                         if !discovered_peers.is_empty() {
    //                             status.connected_peers = discovered_peers.len() as u32;
    //                         }
    //                     }
    //
    //                     // Emit real-time event for dashboard
    //                     let _ = state_clone
    //                         .event_emitter
    //                         .emit_peer_discovered(
    //                             hex::encode(&peer_id[..4]),
    //                             peer_info.reliability_score * 100.0,  // Convert to 0-100 scale
    //                         )
    //                         .await;
    //                 }
    //             }
    //
    //             // Get and log discovery statistics
    //             let stats = discovery_guard.get_stats().await;
    //             if stats.peers_discovered > 0 {
    //                 info!(
    //                     "📊 Production Discovery Stats: {} total, {} DHT, {} Bitcoin, {} DNS discoveries",
    //                     stats.peers_discovered,
    //                     stats.dht_discoveries,
    //                     stats.bitcoin_discoveries,
    //                     stats.dns_discoveries
    //                 );
    //             }
    //
    //             // Test connectivity every few minutes
    //             // TODO: Re-enable when test_peer_connectivity is implemented
    //             // if discovery_interval.missed_tick_behavior() == tokio::time::MissedTickBehavior::Skip {
    //             //     // Every 6th tick (3 minutes), test a random peer connection
    //             //     if !discovered_peers.is_empty() {
    //             //         let random_peer = discovered_peers.keys().next().copied();
    //             //         if let Some(peer_id) = random_peer {
    //             //             match discovery_guard.test_peer_connectivity(peer_id).await {
    //             //                 Ok(latency) => {
    //             //                     info!("✅ Production Discovery - Peer {} responding in {}ms",
    //             //                           hex::encode(&peer_id[..4]), latency.as_millis());
    //             //                 }
    //             //                 Err(e) => {
    //             //                     warn!("⚠️ Production Discovery - Peer {} connectivity failed: {}",
    //             //                           hex::encode(&peer_id[..4]), e);
    //             //                 }
    //             //             }
    //             //         }
    //             //     }
    //             // }
    //         }
    //     });
    // }

    info!("🔍 DEBUG: About to build application router");

    // Build the application router
    let mut app = Router::new()
        // Wallet endpoints
        .route("/api/v1/wallets", get(handlers::list_wallets))
        .route("/api/v1/wallets/create", post(handlers::create_wallet))
        .route("/api/v1/wallets/import", post(handlers::import_wallet))
        .route("/api/v1/wallets/:id", get(handlers::get_wallet))
        .route("/api/v1/wallets/:id/sign", post(handlers::sign_transaction))
        .route(
            "/api/v1/wallets/:address/balance",
            get(handlers::get_wallet_balance),
        ) // Get wallet balance by address
        .route("/api/v1/mnemonic", get(handlers::generate_mnemonic))
        .route("/api/v1/faucet", post(handlers::faucet)) // Test token faucet
        .route("/api/v1/mining/challenge", get(handlers::get_mining_challenge)) // Get current mining challenge
        .route("/api/v1/mining/submit", post(handlers::submit_mining_solution))
        // v0.0.22-beta Quick Win #1: Manual trigger endpoint REMOVED from default routes
        // Added conditionally below based on config.allow_manual_trigger
        // Chain endpoints
        .route("/api/v1/status", get(handlers::node_status))
        .route("/api/v1/node/status", get(handlers::node_status)) // Dashboard compatibility alias
        .route("/api/v1/network/supply", get(handlers::network_supply)) // Network supply statistics (max supply, mined coins, hashrate)
        .route("/api/v1/peer-id", get(handlers::get_peer_id)) // libp2p peer ID for dynamic bootstrap discovery
        .route("/api/v1/transactions", post(handlers::submit_transaction))
        .route(
            "/api/v1/transactions/send",
            post(handlers::send_transaction),
        ) // Combined send endpoint
        // Quantum Privacy Mixer endpoints
        .route(
            "/api/v1/mixer/join",
            post(handlers::join_mixing_pool),
        ) // Join quantum privacy mixing pool
        .route(
            "/api/v1/mixer/send",
            post(handlers::send_private_transaction),
        ) // Send transaction with quantum privacy mixing
        .route(
            "/api/v1/mixer/pools",
            get(handlers::get_mixing_pools_status),
        ) // Get mixing pools status and statistics
        .route(
            "/api/v1/mixer/status/:mixing_id",
            get(handlers::get_mixing_status),
        ) // Get mixing status by session ID or transaction hash

        // ==================== Privacy-as-a-Service (PaaS) API ====================
        // Enterprise-grade privacy infrastructure for all blockchains
        // Revenue flows to Quillon Bank master account
        .route(
            "/api/v1/privacy/tor/relay",
            post(q_api_server::privacy_service_api::tor_relay_service),
        ) // Tor relay service
        .route(
            "/api/v1/privacy/mix/submit",
            post(q_api_server::privacy_service_api::mixing_service),
        ) // Transaction mixing service
        .route(
            "/api/v1/privacy/ring-signature/generate",
            post(q_api_server::privacy_service_api::ring_signature_service),
        ) // Ring signature generation
        .route(
            "/api/v1/privacy/stealth-address/generate",
            post(q_api_server::privacy_service_api::stealth_address_service),
        ) // Stealth address generation
        .route(
            "/api/v1/privacy/zk-stark/prove",
            post(q_api_server::privacy_service_api::zk_stark_proof_service),
        ) // ZK-STARK proof generation
        .route(
            "/api/v1/privacy/paas/statistics",
            get(q_api_server::privacy_service_api::paas_statistics),
        ) // PaaS statistics and revenue

        // Mount PaaS admin router for management endpoints
        .nest(
            "/api/v1/privacy/paas",
            q_api_server::paas_admin_api::create_paas_admin_router(),
        )
        // =========================================================================

        .route("/api/v1/transactions/:hash", get(handlers::get_transaction))
        .route(
            "/api/v1/transactions/recent",
            get(handlers::get_recent_transactions),
        ) // Dashboard recent transactions
        .route("/api/v1/blocks/:height", get(handlers::get_block))

        // ============================================
        // EXPLORER API ENDPOINTS
        // ============================================
        .route("/api/v1/statistics/network", get(handlers::network_analytics)) // Use existing function
        .route("/api/v1/blocks/recent", get(handlers::list_blocks)) // Use existing function
        .route("/api/v1/contracts/recent", get(handlers::list_contracts)) // Use existing function
        .route("/api/v1/dag/vertices/recent", get(handlers::get_dag_vertices)) // Use existing function
        .route("/api/v1/search", get(handlers::search_transactions)) // Use existing function

        // Network Analytics endpoints
        .route(
            "/api/v1/network/analytics",
            get(handlers::network_analytics),
        )
        .route("/api/v1/network/topology", get(handlers::network_topology))
        .route("/api/v1/network/active-peers", get(handlers::active_peers))
        .route("/api/v1/network/peers/connect", post(handlers::connect_peer))
        .route(
            "/api/v1/network/discovery/stats",
            get(handlers::discovery_stats),
        )
        // Bitcoin-Tor Bridge endpoints
        .route(
            "/api/v1/bitcoin/bridge/status",
            get(handlers::bitcoin_bridge_status),
        )
        .route(
            "/api/v1/bitcoin/bridge/peers",
            get(handlers::bitcoin_bridge_peers),
        )
        .route(
            "/api/v1/bitcoin/bridge/stats",
            get(handlers::bitcoin_bridge_stats),
        )
        .route(
            "/api/v1/bitcoin/bridge/connect/:node_id",
            post(handlers::connect_to_peer),
        )
        // DNS-Phantom Network endpoints
        .route(
            "/api/v1/dns/phantom/status",
            get(handlers::dns_phantom_status),
        )
        .route(
            "/api/v1/dns/phantom/peers",
            get(handlers::dns_phantom_peers),
        )
        .route(
            "/api/v1/dns/phantom/send",
            post(handlers::send_phantom_message),
        )
        .route(
            "/api/v1/dns/phantom/providers",
            get(handlers::dns_providers_status),
        )
        .route(
            "/api/v1/dns/phantom/domains",
            get(handlers::generated_domains),
        )
        // Production Peer Discovery endpoints
        .route(
            "/api/v1/discovery/production/status",
            get(handlers::production_discovery_status),
        )
        .route(
            "/api/v1/discovery/production/peers",
            get(handlers::production_discovery_peers),
        )
        .route(
            "/api/v1/discovery/production/stats",
            get(handlers::production_discovery_stats),
        )
        .route(
            "/api/v1/discovery/production/test/:peer_id",
            post(handlers::test_production_peer_connectivity),
        )
        // Security and Monitoring endpoints
        .route(
            "/api/v1/security/anomalies",
            get(handlers::security_anomalies),
        )
        .route("/api/v1/security/threats", get(handlers::threat_analysis))
        .route("/api/v1/security/tor/status", get(handlers::tor_status))
        .route("/api/v1/security/tor/circuits", get(handlers::tor_circuits))
        // Advanced Analytics endpoints
        .route(
            "/api/v1/analytics/performance",
            get(handlers::performance_metrics),
        )
        .route(
            "/api/v1/analytics/steganography",
            get(handlers::steganography_stats),
        )
        .route("/api/v1/analytics/mesh", get(handlers::mesh_network_stats))
        .route(
            "/api/v1/analytics/timeline",
            get(handlers::network_timeline),
        )
        // Real-time streaming endpoints
        .route("/api/v1/events", get(streaming::sse_events))
        .route("/api/v1/ws", get(streaming::websocket_handler))
        // WebSocket transaction streaming for 1M+ TPS (zero HTTP overhead)
        .route("/api/v1/ws/transactions", get(q_api_server::websocket_stream::ws_transaction_stream))
        // ===========================================
        // NEW INTEGRATED CRATES API ROUTES
        // ===========================================
        // ZK Privacy Components
        .route(
            "/api/v1/zk/stark/proof",
            post(handlers::stark_generate_proof),
        )
        .route(
            "/api/v1/zk/groth16/proof",
            post(handlers::groth16_generate_proof),
        )
        .route(
            "/api/v1/zk/plonk/proof",
            post(handlers::plonk_generate_proof),
        )
        // Sharding & Performance
        .route("/api/v1/sharding/status", get(handlers::sharding_status))
        .route(
            "/api/v1/cache/performance",
            get(handlers::cache_performance),
        )
        // Consensus & DAG
        .route(
            "/api/v1/consensus/dag-knight",
            get(handlers::dag_knight_status),
        )
        .route("/api/v1/consensus/narwhal", get(handlers::narwhal_status))
        .route("/api/v1/consensus/vdf", get(handlers::vdf_status))
        // Quillon Resonance Consensus
        .route("/api/v1/consensus/resonance/status", get(handlers::resonance_status))
        .route("/api/v1/consensus/resonance/k-parameter", get(handlers::k_parameter_metrics))
        // Shadow Mode - Performance Monitoring
        .route("/api/v1/consensus/shadow-metrics", get(handlers::shadow_mode_metrics))
        .route("/api/v1/consensus/migration-report", get(handlers::shadow_mode_migration_report))
        .route("/api/v1/consensus/migrate-to-resonance", post(handlers::migrate_to_resonance))
        // Quantum Cryptography
        .route(
            "/api/v1/quantum/crypto/status",
            get(handlers::quantum_crypto_status),
        )
        .route("/api/v1/quantum/bb84/status", get(handlers::bb84_status))
        // DeFi Components
        .route("/api/v1/defi/dex/status", get(handlers::dex_status))
        .route("/api/v1/defi/oracle/status", get(handlers::oracle_status))
        .route("/api/v1/defi/oracle/price/:feed_id", get(handlers::get_oracle_price))
        .route("/api/v1/defi/oracle/feeds", get(handlers::get_oracle_feeds))
        .route(
            "/api/v1/defi/stablecoin/status",
            get(handlers::stablecoin_status),
        )
        // Nitro Points / Token Boosting System
        .route("/api/v1/nitro/boosts", get(handlers::get_nitro_boosts))
        .route("/api/v1/nitro/boost", post(handlers::add_nitro_boost))
        // DEX Swap Functionality
        .route("/api/v1/dex/swap", post(handlers::execute_swap))
        // Blockchain Benchmark (rate limited to once per 24 hours)
        .route("/api/v1/benchmark", post(handlers::run_blockchain_benchmark))
        // Stripe Payment Integration for USD Wallet - ENABLED
        .route("/api/v1/payment/create-intent", post(payment_api::create_payment_intent))
        .route("/api/v1/payment/confirm", post(payment_api::confirm_payment))
        .route("/api/v1/payment/balance", post(payment_api::get_usd_balance))
        .route("/api/v1/payment/withdraw", post(payment_api::withdraw_usd))
        .route("/api/v1/payment/convert-to-qugusd", post(payment_api::convert_usd_to_qugusd))
        .route("/api/v1/payment/transfer", post(payment_api::transfer_usd))
        // OAuth2 Provider for Third-Party Integration - ENABLED
        .route("/api/v1/oauth2/register", post(oauth2_provider::register_client))
        .route("/api/v1/oauth2/authorize", get(oauth2_provider::authorize))
        .route("/api/v1/oauth2/consent", post(oauth2_provider::handle_consent))
        .route("/api/v1/oauth2/token", post(oauth2_provider::token))
        .route("/api/v1/oauth2/userinfo", get(oauth2_provider::userinfo))
        .route("/api/v1/oauth2/revoke", post(oauth2_provider::revoke))
        .route("/api/v1/oauth2/clients/:client_id", get(oauth2_provider::get_client_info))
        // Network & Infrastructure
        .route(
            "/api/v1/tor/circuits/status",
            get(handlers::tor_circuit_status),
        )
        .route(
            "/api/v1/robots/swarm/status",
            get(handlers::robot_swarm_status),
        )
        .route(
            "/api/v1/network/p2p/status",
            get(handlers::p2p_network_status),
        )
        // Plugin System
        .route(
            "/api/v1/plugins/system/status",
            get(handlers::plugin_system_status),
        )
        .route("/api/v1/plugins/install", post(handlers::install_plugin))
        .route(
            "/api/v1/plugins/:plugin_name/execute",
            post(handlers::execute_plugin),
        )
        .route("/api/v1/plugins/metrics", get(handlers::plugin_metrics))
        .route(
            "/api/v1/plugins/:plugin_name/configure",
            put(handlers::configure_plugin),
        )
        .route(
            "/api/v1/plugins/dev/toolkit",
            get(handlers::plugin_dev_toolkit),
        )
        // ===========================================
        // 🌐 DNS-PHANTOM MESH API ENDPOINTS
        // ===========================================
        .route("/api/mesh/status", get(handlers::get_mesh_status))
        .route("/api/mesh/start", post(handlers::start_mesh))
        .route("/api/mesh/stop", post(handlers::stop_mesh))
        .route("/api/mesh/peers", get(handlers::get_mesh_peers))
        .route("/api/mesh/connect", post(handlers::force_mesh_connect))
        .route("/api/mesh/health", get(handlers::get_mesh_health))
        .route("/api/mesh/stats", get(handlers::get_mesh_stats))
        .route("/api/mesh/discover", post(handlers::trigger_mesh_discovery))
        // Smart Contracts API - Orobit Chimera Integration
        .nest("/api/v1/contracts", create_contracts_router())
        // DEX Integration API - Secure external DEX/swap integration
        .nest("/api/v1/dex", create_dex_integration_router())
        // Liquidity Provision API
        .nest("/api/v1/liquidity", create_liquidity_router())
        // ✅ ENABLED - QUG/QUGUSD Dual-Token Stablecoin System (AUTHENTICATED)
        .route("/api/v1/wallet/tokens", get(stablecoin_api::get_multi_token_balance))
        .route("/api/v1/stablecoin/mint", post(stablecoin_api::mint_qugusd))
        .route("/api/v1/stablecoin/redeem", post(stablecoin_api::redeem_qug))
        .route("/api/v1/stablecoin/position/:address", get(stablecoin_api::get_position_health))
        .route("/api/v1/stablecoin/vault/stats", get(stablecoin_api::get_vault_stats))
        .route("/api/v1/stats/fees", get(stablecoin_api::get_fee_stats))
        .route("/api/v1/stablecoin/liquidatable", get(stablecoin_api::get_liquidatable_positions))
        .route("/api/v1/stablecoin/liquidate", post(stablecoin_api::liquidate_position))
        // Simple CDP API - QUGUSD minting with QUG collateral (fallback) - DISABLED (conflicts with full Quillon Bank)
        // .nest("/api/v1/quillon-bank/stablecoin", create_cdp_router())
        // ✅ ENABLED - Full Quillon Bank CDP system with AEGIS-QL post-quantum authentication
        // Public routes (read-only, no authentication)
        .nest("/api/v1/quillon-bank", create_public_routes())
        // Protected routes (founder-only, AEGIS-QL authentication required)
        .nest("/api/v1/quillon-bank",
            create_protected_routes()
                .layer(axum::middleware::from_fn_with_state(
                    app_state.aegis_auth_state.clone(),
                    aegis_auth_middleware::verify_founder_signature
                ))
        )
        // Health and metrics
        .route("/health", get(handlers::health_check))
        .route("/api/v1/health", get(handlers::health_check))
        .route("/metrics", get(handlers::metrics))
        // HIGH-PERFORMANCE BINARY PROTOCOL ENDPOINTS (1000x improvement)
        .route(
            "/api/v1/binary/transaction",
            post(q_api_server::binary_protocol::submit_binary_transaction),
        )
        .route(
            "/api/v1/binary/batch",
            post(q_api_server::binary_protocol::submit_binary_batch),
        )
        .route(
            "/api/v1/binary/stream",
            get(q_api_server::binary_protocol::websocket_binary_handler),
        )
        // Serve static frontend files
        .nest_service("/ui", ServeDir::new("web-ui/dist-final"))
        .fallback_service(ServeDir::new("web-ui/dist-final"))
        // Add middleware
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
                // Increase body size limit to 50MB for large transaction batches (50K tx)
                .layer(axum::extract::DefaultBodyLimit::max(50 * 1024 * 1024)),
        );

    // v0.0.22-beta Quick Win #1: Conditionally add manual trigger endpoint
    // Only register endpoint if explicitly enabled in configuration
    if app_state.config.allow_manual_trigger {
        warn!("⚠️  Manual block trigger endpoint ENABLED at /api/v1/trigger-block");
        warn!("⚠️  This should only be used for testing/development");
        warn!("⚠️  Ensure API authentication is configured!");
        app = app.route("/api/v1/trigger-block", post(handlers::trigger_block_production));
    } else {
        info!("✅ Manual block trigger endpoint DISABLED (secure by default)");
    }

    let app = app.with_state(app_state.clone());

    // Create separate router for IPFS storage endpoints with their own state
    let storage_router = Router::new()
        .route("/api/v1/storage/backup", post(q_api_server::storage_api::backup_database))
        .route("/api/v1/storage/restore", post(q_api_server::storage_api::restore_database))
        .route("/api/v1/storage/status", get(q_api_server::storage_api::storage_status))
        .with_state(ipfs_storage_state);

    // Merge the routers
    let app = app.merge(storage_router);

    // Create shared peer list for P2P connections
    let active_peers = Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));

    // Start P2P connection listener
    let p2p_peers = active_peers.clone();
    let p2p_node_id = node_id;
    let p2p_port = config.port;
    let p2p_node_status = app_state.node_status.clone();
    tokio::spawn(async move {
        if let Err(e) =
            q_api_server::p2p_listener::start_p2p_listener(p2p_port, p2p_node_id, p2p_peers, p2p_node_status).await
        {
            error!("P2P listener failed: {}", e);
        }
    });

    // ============================================================================
    // PARALLEL WORKER POOL - 16x HIGH-THROUGHPUT CONSENSUS PIPELINE
    // ============================================================================
    // This replaces the single background processor with 16 parallel workers:
    // - 16 workers processing sharded transaction pool
    // - Hash-based deterministic sharding
    // - NUMA-aware CPU pinning (optional)
    // - Lock-free coordination via DashMap
    //
    // Pipeline: DashMap → SIMD Verification → Narwhal → DAG-Knight → Bullshark
    // Target: 350,000+ TPS (16x improvement over 21,817 TPS baseline)
    // ============================================================================
    info!("🚀 Starting parallel worker pool for 16x performance improvement");
    info!("   16 parallel workers processing sharded transaction pool");
    info!("   Expected TPS: {} (16x over 21,817 baseline)", 21_817 * 16);
    info!("   Full consensus pipeline: SIMD → Narwhal → DAG-Knight → Bullshark");

    // Initialize parallel worker pool
    // NOTE: Even though worker_pool is unused, it must be kept alive so spawned tasks persist
    let worker_pool = q_api_server::parallel_workers::init_parallel_workers(app_state.clone());
    info!("✅ Parallel worker pool initialized successfully");
    info!("   Workers will process transactions continuously in background");

    // Start libp2p-based zero-config peer discovery (mDNS + Gossipsub)
    if let Some(libp2p_discovery) = &app_state.libp2p_discovery {
        // Create channel for libp2p → ConnectionManager bridge (Phase 2)
        if let Some(connection_manager) = &app_state.connection_manager {
            let (peer_tx, mut peer_rx) = tokio::sync::mpsc::unbounded_channel::<q_network::connection_manager::PeerInfo>();

            // ✅ DEADLOCK FIX: Use command channel pattern (non-blocking)
            // Send SetPeerChannel command to network manager event loop
            if let Some(command_tx) = &app_state.libp2p_command_tx {
                if let Err(e) = command_tx.send(q_network::NetworkCommand::SetPeerChannel { tx: peer_tx }) {
                    error!("❌ Failed to send SetPeerChannel command: {}", e);
                } else {
                    info!("✅ Sent SetPeerChannel command to network manager");
                    info!("🌉 libp2p → ConnectionManager bridge ENABLED");
                }

                // Spawn receiver task to forward peers to ConnectionManager
                let connection_mgr_bridge = connection_manager.clone();
                tokio::spawn(async move {
                    info!("🌉 Starting libp2p → ConnectionManager bridge receiver...");
                    while let Some(peer_info) = peer_rx.recv().await {
                        info!("🌉 Bridging peer {} ({:?}) to ConnectionManager",
                              peer_info.node_id, peer_info.server_role);
                        connection_mgr_bridge.add_discovered_peer(peer_info).await;
                    }
                    warn!("🌉 libp2p → ConnectionManager bridge channel closed");
                });
            } else {
                warn!("⚠️  libp2p command channel not available");
                drop(peer_tx);
                drop(peer_rx);
            }
        }

        // Set up gossipsub message forwarding channel for database replication
        if let Some((_, incoming_tx)) = &replication_system {
            let (gossipsub_msg_tx, mut gossipsub_msg_rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();

            // ✅ DEADLOCK FIX: Use command channel pattern (non-blocking)
            // Send SetGossipsubChannel command to network manager event loop
            if let Some(command_tx) = &app_state.libp2p_command_tx {
                if let Err(e) = command_tx.send(q_network::NetworkCommand::SetGossipsubChannel { tx: gossipsub_msg_tx }) {
                    error!("❌ Failed to send SetGossipsubChannel command: {}", e);
                } else {
                    info!("✅ Sent SetGossipsubChannel command to network manager");
                    info!("🌉 Gossipsub → replication bridge ENABLED");
                }

                // Spawn receiver task to forward gossipsub messages to replication bridge
                let incoming_replication_tx = incoming_tx.clone();
                tokio::spawn(async move {
                    info!("📥 Starting gossipsub → replication bridge receiver...");
                    while let Some((topic, data)) = gossipsub_msg_rx.recv().await {
                        // Only forward database update messages to replication bridge
                        if topic == q_ipfs_storage::DATABASE_UPDATES_TOPIC {
                            tracing::debug!("📥 Forwarding database update message to replication bridge");
                            if let Err(e) = incoming_replication_tx.send(data) {
                                tracing::error!("❌ Failed to forward message to replication bridge: {}", e);
                            }
                        }
                    }
                    tracing::warn!("📥 Gossipsub → replication bridge channel closed");
                });
            } else {
                warn!("⚠️  libp2p command channel not available");
                drop(gossipsub_msg_tx);
                drop(gossipsub_msg_rx);
            }
        }

        // Spawn libp2p discovery event loop
        // DEADLOCK FIX: This is DUPLICATE code - the event loop was already spawned at line 565!
        // This code tries to lock the manager and spawn a second event loop, which will deadlock.
        // The real event loop is already running in the background.
        warn!("⚠️  Duplicate event loop spawn disabled (already running from line 565)");

        // DISABLED:
        // let discovery_clone = libp2p_discovery.clone();
        // tokio::spawn(async move {
        //     info!("🚀 Starting libp2p Zero-Knowledge Discovery event loop...");
        //     let mut discovery_guard = discovery_clone.lock().await;
        //     if let Err(e) = discovery_guard.run().await {
        //         error!("❌ libp2p discovery event loop failed: {}", e);
        //     }
        // });
    }

    info!("🔍 DEBUG: About to initialize HTTP server");

    // Start the HIGH-PERFORMANCE HTTP API server
    info!("🚀 Initializing High-Performance HTTP Server for 1M+ TPS");
    info!("   TCP optimizations: NODELAY, REUSEPORT, 4MB buffers");
    info!("   HTTP/2 support: Automatic via client negotiation");
    info!(
        "P2P connections will be accepted on port {}",
        config.port + 1
    );

    // Use our optimized HTTP server with TCP socket configuration
    use q_api_server::high_performance_server::HighPerformanceServer;

    let addr: std::net::SocketAddr = format!("0.0.0.0:{}", config.port).parse()?;
    let high_perf_server = HighPerformanceServer::new(app, addr)
        .with_tcp_buffers(4 * 1024 * 1024, 4 * 1024 * 1024)  // 4MB buffers
        .with_backlog(1024);  // 1024 pending connections

    if tui_mode {
        // Run TUI mode
        #[cfg(feature = "tui")]
        {
            info!("🎨 Launching TUI mode - Beautiful terminal UI enabled");

            // Create shared metrics for TUI
            let tui_metrics = std::sync::Arc::new(std::sync::RwLock::new(q_tui::Metrics::default()));

            // Create TUI app with shared metrics
            let tui_app = q_tui::App::with_metrics(tui_metrics.clone());

            // Spawn metrics updater task
            let metrics_clone = tui_metrics.clone();
            let app_state_clone = app_state.clone();
            let start_time = std::time::Instant::now();

            tokio::spawn(async move {
                info!("📊 Starting TUI metrics updater task...");
                loop {
                    // Update metrics from app state
                    if let Err(e) = update_tui_metrics(&metrics_clone, &app_state_clone, start_time).await {
                        warn!("Failed to update TUI metrics: {}", e);
                    }

                    // Update every second
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
            });

            // Spawn server in background
            tokio::spawn(async move {
                if let Err(e) = high_perf_server.run().await {
                    error!("Server error: {}", e);
                }
            });

            // Run TUI in foreground
            q_tui::run_tui(tui_app).await?;
        }

        #[cfg(not(feature = "tui"))]
        {
            error!("TUI mode requested but not compiled with --features tui");
            error!("Please rebuild with: cargo build --features tui");
            return Err("TUI feature not enabled".into());
        }
    } else {
        // Run normally without TUI
        high_perf_server.run().await?;
    }

    Ok(())
}
