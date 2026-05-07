// UDP server main loop.

use std::sync::Arc;
use tokio::net::UdpSocket;
use tracing::{error, info};

use crate::config::Config;
use crate::turn::{self, TurnState};

const RECV_BUF: usize = 65536;

pub async fn run(config: Config) -> anyhow::Result<()> {
    let bind_addr = config.server.bind;
    let sock = UdpSocket::bind(bind_addr).await?;
    info!("q-turn listening on udp://{}", bind_addr);
    info!("  realm:      {}", config.server.realm);
    info!("  public_ip:  {}", config.relay.public_ip);
    info!("  relay ports: {}–{}", config.relay.min_port, config.relay.max_port);
    info!("  max_allocations: {}", config.limits.max_allocations);

    let sock = Arc::new(sock);
    let state = TurnState::new(config, sock.clone());

    // Background nonce eviction every 60 seconds
    {
        let auth = state.auth.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                auth.evict_nonces();
            }
        });
    }

    let mut buf = vec![0u8; RECV_BUF];
    loop {
        let (len, src) = match sock.recv_from(&mut buf).await {
            Ok(v)  => v,
            Err(e) => { error!("recv_from error: {}", e); continue; }
        };

        let raw = buf[..len].to_vec();
        let state_ref = state.clone();

        // Spawn per-packet task so the receive loop is never blocked.
        // For high-throughput deployments this can be replaced with a worker pool.
        tokio::spawn(async move {
            if let Some(response) = turn::handle_packet(&raw, src, &state_ref).await {
                state_ref.main_sock.send_to(&response, src).await.ok();
            }
        });
    }
}
