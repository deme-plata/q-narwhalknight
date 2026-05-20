// v10.10.9 Phase C prep — libp2p Transport adapter over QTorClient.
//
// The existing `libp2p_transport::TorTransport` takes `Arc<DedicatedCircuitManager>`
// which is a different type from `QTorClient.circuit_manager: Arc<Mutex<CircuitManager>>`.
// Wiring it would either double-bootstrap Arti (TorMessageRouter creates its own
// DedicatedCircuitManager) or require extracting a shared circuit manager.
//
// This adapter sidesteps both: it holds `Arc<QTorClient>` and delegates dial
// to `QTorClient::connect_to_peer`, which uses the existing CircuitManager
// (single Arti instance). On dial the result is a `TcpStream` (extracted from
// `TorConnection::into_stream`), which libp2p can pipe Noise+Yamux over.
//
// Wire-up (v10.10.10): in unified_network_manager.rs:1819 area, when
// `tor_policy.outbound_via_tor`, replace the plain `.with_tcp(...)` step with
// `.with_other_transport(|key| { OrTransport::new(QTorTransport::new(qtor),
//   tcp_transport).authenticate(Noise).multiplex(Yamux).boxed() })`.

use crate::QTorClient;
use futures::future::BoxFuture;
use libp2p::{
    core::{
        transport::{DialOpts, ListenerId, Transport, TransportError, TransportEvent},
    },
    Multiaddr,
};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::net::TcpStream;
use tracing::{debug, warn};

/// Outbound-only libp2p Transport that dials via the agent's Tor circuits.
///
/// Accepts `/onion3/<base32>:<port>` multiaddrs natively. With clearnet
/// fallback enabled, also handles `/ip4/.../tcp/<port>` by routing through
/// the embedded Arti SOCKS5 proxy (so peers see the Tor exit, not us).
///
/// listen_on / poll are no-ops — incoming connections need the Phase A++
/// onion-service ↔ libp2p delivery pipeline which isn't built yet
/// (see plans/v10.10.10-tor-phase-c.md if present).
pub struct QTorTransport {
    qtor: Arc<QTorClient>,
    allow_clearnet_fallback: bool,
}

impl QTorTransport {
    pub fn new(qtor: Arc<QTorClient>) -> Self {
        Self {
            qtor,
            allow_clearnet_fallback: false,
        }
    }

    /// Enable wrapping plain `/ip4/.../tcp/...` dials through Tor too.
    /// Without this, only onion3 addresses are dialled via Tor; other
    /// addresses fail with MultiaddrNotSupported so the OrTransport
    /// composition can fall through to the clearnet TCP transport.
    pub fn with_clearnet_fallback(mut self, allow: bool) -> Self {
        self.allow_clearnet_fallback = allow;
        self
    }

    /// Extract `<onion-pubkey>.onion:<port>` from a `/onion3/<base32>:<port>` multiaddr.
    fn extract_onion_target(addr: &Multiaddr) -> Option<String> {
        use libp2p::multiaddr::Protocol;
        for proto in addr.iter() {
            if let Protocol::Onion3(onion) = proto {
                let hash_b32 = base32::encode(
                    base32::Alphabet::Rfc4648Lower { padding: false },
                    onion.hash(),
                );
                return Some(format!("{}.onion:{}", hash_b32, onion.port()));
            }
        }
        None
    }

    /// Extract `<host>:<port>` from a `/ip4/<addr>/tcp/<port>` multiaddr.
    fn extract_tcp_target(addr: &Multiaddr) -> Option<String> {
        use libp2p::multiaddr::Protocol;
        let mut host: Option<String> = None;
        let mut port: Option<u16> = None;
        for proto in addr.iter() {
            match proto {
                Protocol::Ip4(ip) => host = Some(ip.to_string()),
                Protocol::Ip6(ip) => host = Some(format!("[{}]", ip)),
                Protocol::Dns(name) | Protocol::Dns4(name) | Protocol::Dns6(name) => {
                    host = Some(name.to_string())
                }
                Protocol::Tcp(p) => port = Some(p),
                _ => {}
            }
        }
        match (host, port) {
            (Some(h), Some(p)) => Some(format!("{}:{}", h, p)),
            _ => None,
        }
    }
}

impl Transport for QTorTransport {
    type Output = TcpStream;
    type Error = std::io::Error;
    type ListenerUpgrade = futures::future::Ready<Result<Self::Output, Self::Error>>;
    type Dial = BoxFuture<'static, Result<Self::Output, Self::Error>>;

    fn listen_on(
        &mut self,
        _id: ListenerId,
        addr: Multiaddr,
    ) -> Result<(), TransportError<Self::Error>> {
        // Outbound-only. Inbound requires Phase A++ work (Arti
        // launch_onion_service → libp2p TransportEvent::Incoming).
        Err(TransportError::MultiaddrNotSupported(addr))
    }

    fn remove_listener(&mut self, _id: ListenerId) -> bool {
        false
    }

    fn dial(
        &mut self,
        addr: Multiaddr,
        _opts: DialOpts,
    ) -> Result<Self::Dial, TransportError<Self::Error>> {
        // Pick a target string for connect_to_peer.
        let target = if let Some(t) = Self::extract_onion_target(&addr) {
            t
        } else if self.allow_clearnet_fallback {
            match Self::extract_tcp_target(&addr) {
                Some(t) => t,
                None => return Err(TransportError::MultiaddrNotSupported(addr)),
            }
        } else {
            // Let the OrTransport peer (clearnet TCP) handle non-onion addrs.
            return Err(TransportError::MultiaddrNotSupported(addr));
        };

        debug!("🧅 [QTorTransport] dial via Tor: {} ({})", addr, target);
        let qtor = Arc::clone(&self.qtor);
        let fut = async move {
            match qtor.connect_to_peer(&target).await {
                Ok(conn) => Ok(conn.into_stream()),
                Err(e) => {
                    warn!("🧅 [QTorTransport] connect_to_peer({}) failed: {}", target, e);
                    Err(std::io::Error::new(std::io::ErrorKind::ConnectionRefused, e.to_string()))
                }
            }
        };
        Ok(Box::pin(fut))
    }

    fn poll(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        // Outbound-only — no incoming connections from this Transport.
        Poll::Pending
    }
}
