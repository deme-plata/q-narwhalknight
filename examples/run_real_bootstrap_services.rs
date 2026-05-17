//! Run Real Bootstrap Services
//!
//! This example demonstrates running actual Tor hidden services for bootstrap node discovery.
//! Creates production-ready bootstrap services with real .onion addresses.

use anyhow::Result;
use clap::{Arg, Command};
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tracing::{error, info, warn, Level};
use tracing_subscriber;

// Import our implementations
use q_tor_client::{
    bootstrap_service::{BootstrapService, BootstrapServiceBuilder},
    real_bootstrap_discovery::{RealBootstrapBuilder, RealBootstrapDiscovery},
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    let matches = Command::new("Q-NarwhalKnight Bootstrap Services")
        .version("1.0.0")
        .about("Run real Tor hidden services for ZK-enhanced DHT bootstrap")
        .arg(
            Arg::new("mode")
                .short('m')
                .long("mode")
                .value_name("MODE")
                .help("Operation mode: server, client, or hybrid")
                .default_value("hybrid"),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("Local port for HTTP server")
                .default_value("8080"),
        )
        .arg(
            Arg::new("count")
                .short('c')
                .long("count")
                .value_name("COUNT")
                .help("Number of bootstrap services to run")
                .default_value("1"),
        )
        .arg(
            Arg::new("zk-proofs")
                .long("zk-proofs")
                .help("Require ZK proofs for enhanced security")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let mode = matches.get_one::<String>("mode").unwrap();
    let base_port: u16 = matches.get_one::<String>("port").unwrap().parse()?;
    let service_count: usize = matches.get_one::<String>("count").unwrap().parse()?;
    let require_zk = matches.get_flag("zk-proofs");

    info!("🚀 Q-NarwhalKnight Real Bootstrap Services");
    info!("   Mode: {}", mode);
    info!("   Base Port: {}", base_port);
    info!("   Service Count: {}", service_count);
    info!("   ZK Proofs Required: {}", require_zk);

    match mode.as_str() {
        "server" => run_bootstrap_servers(base_port, service_count, require_zk).await,
        "client" => run_bootstrap_client().await,
        "hybrid" => run_hybrid_mode(base_port, service_count, require_zk).await,
        _ => {
            error!("Invalid mode: {}. Use server, client, or hybrid", mode);
            std::process::exit(1);
        }
    }
}

/// Run bootstrap servers (hidden services)
async fn run_bootstrap_servers(base_port: u16, count: usize, require_zk: bool) -> Result<()> {
    info!("🌐 Starting {} bootstrap services", count);

    let mut services = Vec::new();
    let mut onion_addresses = Vec::new();

    // Create and start multiple bootstrap services
    for i in 0..count {
        let port = base_port + i as u16;
        let service_name = format!("QNK Bootstrap Service #{}", i + 1);

        info!("🔧 Creating bootstrap service: {}", service_name);

        let mut service = BootstrapServiceBuilder::new()
            .service_name(service_name.clone())
            .local_port(port)
            .max_peers(10_000)
            .peer_ttl(Duration::from_secs(3600))
            .require_zk_proofs(require_zk)
            .build()
            .await?;

        // Start the service and get onion address
        let onion_addr = service.start().await?;
        let full_address = format!("{}.onion", onion_addr);

        info!("✅ {} online at: {}", service_name, full_address);
        onion_addresses.push(full_address);
        services.push(service);

        // Brief delay between service startups
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Print summary
    info!("🎉 All bootstrap services are running!");
    info!("📋 Service Directory:");
    for (i, address) in onion_addresses.iter().enumerate() {
        info!("   Service #{}: {}", i + 1, address);
        info!("     API Endpoints:");
        info!("       GET  http://{}/api/v1/peers", address);
        info!("       POST http://{}/api/v1/peers/register", address);
        info!("       GET  http://{}/api/v1/status", address);
        info!("       GET  http://{}/health", address);
    }

    info!("🔄 Services will run until interrupted (Ctrl+C)");

    // Wait for shutdown signal
    signal::ctrl_c().await?;

    info!("🛑 Shutdown signal received, stopping services...");
    info!("✅ Bootstrap services stopped gracefully");

    Ok(())
}

/// Run bootstrap client (discovery)
async fn run_bootstrap_client() -> Result<()> {
    info!("🔍 Starting bootstrap client for peer discovery");

    // Create Tor client
    let tor_client = Arc::new(
        arti_client::TorClient::create_bootstrapped(arti_client::TorClientConfig::default())
            .await?,
    );

    // Create real bootstrap discovery system
    let discovery = RealBootstrapBuilder::new()
        .run_local_service(false) // Client mode - don't run local service
        .reputation_threshold(0.7) // Higher threshold for client
        .peer_ttl(Duration::from_secs(1800)) // 30 minutes
        .build(tor_client)
        .await?;

    info!("🌍 Discovering peers from real bootstrap services...");

    // Discover peers
    let discovered_peers = discovery.discover_real_peers().await?;

    info!("📊 Discovery Results:");
    info!("   Total peers discovered: {}", discovered_peers.len());

    if discovered_peers.is_empty() {
        warn!("⚠️  No peers discovered - bootstrap services may be offline");
        return Ok(());
    }

    // Display discovered peers
    info!("📋 Discovered Peers:");
    for (i, peer) in discovered_peers.iter().enumerate() {
        info!("   Peer #{}: {}", i + 1, peer.node_id);
        info!("     Address: {}", peer.to_address_string());
        info!("     Capabilities: {}", peer.capabilities.join(", "));
        info!("     Reputation: {:.2}", peer.reputation);
        info!("     ZK Verified: {}", peer.zk_proof_verified);
        info!("     Source: {}", peer.bootstrap_source);
    }

    // Get service statistics
    let service_stats = discovery.get_service_stats().await;
    info!("📈 Bootstrap Service Statistics:");
    for service in service_stats {
        info!("   Service: {}", service.service_name);
        info!("     Address: {}", service.onion_address);
        info!("     Reputation: {:.2}", service.reputation);
        info!("     Active Peers: {}", service.active_peers);
        info!("     Response Time: {}ms", service.response_time_ms);
        info!("     ZK Support: {}", service.supports_zk_proofs);
    }

    Ok(())
}

/// Run hybrid mode (both server and client)
async fn run_hybrid_mode(base_port: u16, count: usize, require_zk: bool) -> Result<()> {
    info!("🔄 Starting hybrid mode: bootstrap services + peer discovery");

    // Create Tor client
    let tor_client = Arc::new(
        arti_client::TorClient::create_bootstrapped(arti_client::TorClientConfig::default())
            .await?,
    );

    // Start local bootstrap services
    info!("🌐 Phase 1: Starting local bootstrap services");
    let mut services = Vec::new();
    let mut local_addresses = Vec::new();

    for i in 0..count {
        let port = base_port + i as u16;
        let service_name = format!("Hybrid Bootstrap #{}", i + 1);

        let mut service = BootstrapServiceBuilder::new()
            .service_name(service_name.clone())
            .local_port(port)
            .max_peers(5_000)
            .peer_ttl(Duration::from_secs(3600))
            .require_zk_proofs(require_zk)
            .build()
            .await?;

        let onion_addr = service.start().await?;
        let full_address = format!("{}.onion", onion_addr);

        info!("✅ {} online at: {}", service_name, full_address);
        local_addresses.push(full_address);
        services.push(service);
    }

    // Wait a moment for services to fully initialize
    tokio::time::sleep(Duration::from_secs(2)).await;

    info!("🔍 Phase 2: Starting peer discovery");

    // Create discovery system that includes our local services
    let discovery = RealBootstrapBuilder::new()
        .run_local_service(false) // We already started services above
        .reputation_threshold(0.5)
        .peer_ttl(Duration::from_secs(3600))
        .build(tor_client.clone())
        .await?;

    // Simulate our own node registration
    let our_node = q_tor_client::production_tor_dht::ProductionDhtRecord {
        node_id: format!("hybrid-node-{:x}", rand::random::<u32>()),
        onion_address: format!("ournode{:x}.onion", rand::random::<u32>()),
        dht_port: 8333,
        public_key: vec![0u8; 32],
        signature: ed25519_dalek::Signature::from([0u8; 64]),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        capabilities: vec![
            "zk-snark".to_string(),
            "zk-stark".to_string(),
            "dht".to_string(),
        ],
        reputation_score: 1.0,
    };

    info!("📝 Phase 3: Registering with bootstrap services");
    if let Err(e) = discovery.register_with_bootstrap_services(&our_node).await {
        warn!("⚠️  Registration partially failed: {}", e);
    }

    info!("🔍 Phase 4: Discovering network peers");
    let discovered_peers = discovery.discover_real_peers().await?;

    info!("🎉 Hybrid mode fully operational!");
    info!("📊 Network Status:");
    info!("   Local bootstrap services: {}", local_addresses.len());
    info!("   Discovered peers: {}", discovered_peers.len());
    info!("   Our node ID: {}", our_node.node_id);

    info!("📋 Local Services:");
    for address in &local_addresses {
        info!("   {}", address);
    }

    if !discovered_peers.is_empty() {
        info!("📋 Sample Discovered Peers:");
        for peer in discovered_peers.iter().take(5) {
            info!("   {} at {}", peer.node_id, peer.to_address_string());
        }
        if discovered_peers.len() > 5 {
            info!("   ... and {} more", discovered_peers.len() - 5);
        }
    }

    // Keep running and periodically rediscover
    info!("🔄 Entering maintenance mode - press Ctrl+C to stop");

    let mut discovery_interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes

    loop {
        tokio::select! {
            _ = discovery_interval.tick() => {
                info!("🔄 Periodic peer rediscovery...");
                match discovery.discover_real_peers().await {
                    Ok(peers) => {
                        info!("📊 Rediscovery: {} peers found", peers.len());
                    }
                    Err(e) => {
                        warn!("⚠️  Rediscovery failed: {}", e);
                    }
                }
            }
            _ = signal::ctrl_c() => {
                info!("🛑 Shutdown signal received");
                break;
            }
        }
    }

    info!("✅ Hybrid mode stopped gracefully");
    Ok(())
}

/// Bootstrap service deployment helper
pub async fn deploy_production_bootstrap_network() -> Result<()> {
    info!("🚀 Deploying Production Bootstrap Network");

    // This would deploy multiple bootstrap services across different servers
    // Each with its own Tor hidden service

    let bootstrap_configs = vec![
        ("Alpha", 8080, true),  // Primary with ZK proofs
        ("Beta", 8081, true),   // Secondary with ZK proofs
        ("Gamma", 8082, false), // Legacy compatibility
        ("Delta", 8083, true),  // High-capacity node
    ];

    let mut deployed_services = Vec::new();

    for (name, port, zk_enabled) in bootstrap_configs {
        info!("🔧 Deploying bootstrap service: {}", name);

        let service = BootstrapServiceBuilder::new()
            .service_name(format!("QNK Production Bootstrap {}", name))
            .local_port(port)
            .max_peers(50_000) // Production capacity
            .peer_ttl(Duration::from_secs(7200)) // 2 hours
            .require_zk_proofs(zk_enabled)
            .build()
            .await?;

        // In production, this would include:
        // - SSL/TLS certificates
        // - Rate limiting
        // - DDoS protection
        // - Monitoring and alerting
        // - Backup and failover
        // - Geographic distribution

        deployed_services.push((name, service));
    }

    info!("✅ Production bootstrap network deployment complete");
    info!("📊 Deployed {} bootstrap services", deployed_services.len());

    Ok(())
}
