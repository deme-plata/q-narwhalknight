//! Minimal libp2p dialer to test transport in isolation
//!
//! This example creates the simplest possible libp2p node that:
//! 1. Uses the exact same transport stack as q-api-server
//! 2. Dials Server Beta's bootstrap peer
//! 3. Reports success or detailed failure
//!
//! **Purpose**: Isolate whether connection failures are due to:
//! - Transport configuration issues (this will fail)
//! - QNarwhalBehaviour complexity (this will succeed, main fails)
//!
//! **Usage**:
//! ```bash
//! cargo run --example minimal_dialer
//! ```
//!
//! **Expected output on success**:
//! ```
//! ✅ SUCCESS: ConnectionEstablished
//!    Peer: 12D3KooWC688bzHi7djbkensGQMABzX9tY41LNasgd3g3FdwqQn7
//! ```
//!
//! **On failure**, you'll see detailed transport errors that pinpoint the issue.

use anyhow::Result;
use futures::StreamExt;
use libp2p::{
    core::{transport::Transport, upgrade},
    identity, noise, tcp, yamux,
    swarm::{Swarm, SwarmEvent},
    Multiaddr, PeerId,
};
use std::time::Duration;
use tracing::{debug, error, info};

/// Server Beta's bootstrap peer (as of 2025-11-18)
const TARGET_PEER: &str = "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWC688bzHi7djbkensGQMABzX9tY41LNasgd3g3FdwqQn7";

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("minimal_dialer=debug".parse()?)
                .add_directive("libp2p_swarm=debug".parse()?)
                .add_directive("libp2p_tcp=debug".parse()?),
        )
        .init();

    info!("🚀 Minimal libp2p Dialer - Transport Isolation Test");
    info!("");

    // Generate ephemeral identity for this test
    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    info!("🔑 Local peer ID: {}", local_peer_id);
    info!("");

    // Build transport - EXACT SAME as q-api-server
    info!("🔧 Building transport stack:");
    info!("   - TCP transport (async_io)");
    info!("   - Upgrade: Version::V1");
    info!("   - Auth: Noise (XX handshake)");
    info!("   - Multiplex: Yamux");

    let tcp_transport = tcp::async_io::Transport::new(tcp::Config::default());

    let transport = tcp_transport
        .upgrade(upgrade::Version::V1)
        .authenticate(noise::Config::new(&local_key)?)
        .multiplex(yamux::Config::default())
        .boxed();

    info!("✅ Transport built successfully");
    info!("");

    // Use simplest possible behaviour (just ping)
    let behaviour = libp2p::ping::Behaviour::default();

    // Create swarm
    let mut swarm = Swarm::new(
        transport,
        behaviour,
        local_peer_id,
        libp2p::swarm::Config::with_tokio_executor(),
    );

    // Parse target multiaddr
    let target: Multiaddr = TARGET_PEER.parse()?;
    info!("🎯 Target: {}", target);
    info!("   IP: 185.182.185.227");
    info!("   Port: 9001");
    info!("   PeerID: 12D3KooWC688bzHi7djbkensGQMABzX9tY41LNasgd3g3FdwqQn7");
    info!("");

    // Dial the target
    info!("📡 Initiating dial...");
    swarm.dial(target.clone())?;

    // Set timeout
    let timeout_duration = Duration::from_secs(30);
    let timeout = tokio::time::sleep(timeout_duration);
    tokio::pin!(timeout);

    info!("⏳ Waiting for connection (30s timeout)...");
    info!("");

    // Event loop
    loop {
        tokio::select! {
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                        info!("✅ SUCCESS: ConnectionEstablished");
                        info!("   Peer: {}", peer_id);
                        info!("   Endpoint: {:?}", endpoint);
                        info!("");
                        info!("🎉 RESULT: Transport layer is working correctly!");
                        info!("   → If main q-api-server still fails, the issue is in:");
                        info!("      - QNarwhalBehaviour configuration");
                        info!("      - Connection limits");
                        info!("      - Some other behaviour blocking dials");
                        return Ok(());
                    }

                    SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                        error!("❌ FAILURE: OutgoingConnectionError");
                        error!("   Peer: {:?}", peer_id);
                        error!("");
                        error!("   Detailed error breakdown:");

                        // Detailed error analysis
                        match &error {
                            libp2p::swarm::DialError::Transport(addrs) => {
                                error!("   🚨 TRANSPORT ERROR:");
                                for (addr, transport_err) in addrs {
                                    error!("      Failed address: {}", addr);
                                    error!("      Transport error: {:?}", transport_err);

                                    // Further breakdown
                                    use libp2p::TransportError;
                                    match transport_err {
                                        TransportError::MultiaddrNotSupported(a) => {
                                            error!("         → Multiaddr not supported: {}", a);
                                            error!("         → FIX: Check transport supports this address type");
                                        }
                                        TransportError::Other(e) => {
                                            error!("         → IO Error: {}", e);

                                            // Common IO errors
                                            let err_str = format!("{}", e);
                                            if err_str.contains("Connection refused") {
                                                error!("         → FIX: Server Beta not listening on port 9001");
                                                error!("                Check: ss -tlnp | grep 9001 on Beta");
                                            } else if err_str.contains("timeout") {
                                                error!("         → FIX: Firewall blocking connection");
                                                error!("                Check: iptables/ufw on both servers");
                                            } else if err_str.contains("DNS") {
                                                error!("         → FIX: DNS resolution failed");
                                                error!("                Use /ip4/ instead of /dns/ in multiaddr");
                                            }
                                        }
                                    }
                                }
                            }

                            libp2p::swarm::DialError::ConnectionLimit(limit) => {
                                error!("   🚨 CONNECTION LIMIT: {:?}", limit);
                                error!("      → FIX: Increase or disable connection limits");
                            }

                            libp2p::swarm::DialError::Denied { cause } => {
                                error!("   🚨 CONNECTION DENIED: {:?}", cause);
                                error!("      → FIX: Check connection gating logic");
                            }

                            libp2p::swarm::DialError::NoAddresses => {
                                error!("   🚨 NO ADDRESSES TO DIAL");
                                error!("      → FIX: Bootstrap multiaddr is empty or invalid");
                            }

                            libp2p::swarm::DialError::WrongPeerId { obtained, endpoint } => {
                                error!("   🚨 WRONG PEER ID:");
                                error!("      Expected: 12D3KooWC688bzHi7djbkensGQMABzX9tY41LNasgd3g3FdwqQn7");
                                error!("      Obtained: {}", obtained);
                                error!("      Endpoint: {:?}", endpoint);
                                error!("      → FIX: Update BOOTSTRAP_PEERS with correct PeerID");
                            }

                            libp2p::swarm::DialError::Aborted => {
                                error!("   🚨 DIAL ABORTED");
                                error!("      → Dial was cancelled before completion");
                            }

                            libp2p::swarm::DialError::DialPeerConditionFalse(_) => {
                                error!("   🚨 DIAL PEER CONDITION FALSE");
                                error!("      → Some pre-dial condition check failed");
                            }

                            other => {
                                error!("   🚨 OTHER ERROR: {:?}", other);
                            }
                        }

                        error!("");
                        error!("🔍 DIAGNOSIS COMPLETE");
                        error!("   → Use error details above to identify fix");
                        return Err(anyhow::anyhow!("Connection failed: {:?}", error));
                    }

                    SwarmEvent::Dialing { peer_id, .. } => {
                        debug!("📞 Dialing peer: {:?}", peer_id);
                    }

                    other => {
                        debug!("Event: {:?}", other);
                    }
                }
            }

            _ = &mut timeout => {
                error!("❌ TIMEOUT: No connection after 30 seconds");
                error!("");
                error!("Possible causes:");
                error!("   1. Firewall blocking outbound connections");
                error!("   2. Server Beta not reachable");
                error!("   3. Port 9001 not open");
                error!("");
                error!("Quick tests:");
                error!("   nc -zv 185.182.185.227 9001  # Should succeed");
                error!("   curl telnet://185.182.185.227:9001  # Should connect");
                return Err(anyhow::anyhow!("Connection timeout"));
            }
        }
    }
}
