//! 🧅 SOCKS5-beneath-TCP libp2p transport (Tor Phase C — outbound onion dialing)
//!
//! This is the "Tor-native" piece the privacy whitepaper (§8.3 roadmap, Q2 2026)
//! and CLAUDE.md ("Phase C: outbound libp2p dials through Tor SOCKS5") call out as
//! the remaining work. The earlier `.with_other_transport(QTorTransport)` splice was
//! reverted (see `unified_network_manager.rs` ~:1822) because the embedded-Arti
//! transport's `Output` was a raw `tokio::net::TcpStream` that didn't satisfy the
//! `futures::AsyncRead + AsyncWrite` bound `.authenticate()` needs.
//!
//! This transport sidesteps all three of those root causes by being a *thin dialer*:
//! it connects to Arti's SOCKS5 proxy (`Q_TOR_SOCKS5_ADDR`, default 127.0.0.1:9050),
//! issues a SOCKS5 CONNECT to the target (`.onion:port` or `ip:port`), and hands the
//! resulting stream back as a `tokio_util::compat::Compat<TcpStream>` — which *does*
//! implement `futures` AsyncRead/AsyncWrite. libp2p then applies its **normal**
//! Noise + Yamux upgrade on top, so security/muxing are unchanged; only the bytes
//! underneath travel through a Tor circuit.
//!
//! ## Gating — zero regression by default
//! The transport is always spliced into the `SwarmBuilder` chain (the builder is
//! type-level; we can't conditionally skip a phase), but `dial()` returns
//! `MultiaddrNotSupported` whenever Tor-SOCKS dialing is disabled or the address
//! shouldn't be Tor-routed. When it returns that, libp2p simply falls through to the
//! TCP/QUIC transports — i.e. *exactly* today's behaviour. So with the feature off,
//! nothing changes.
//!
//! ## Routing policy
//! - **disabled**         → never handles any dial (pure passthrough to TCP/QUIC).
//! - **onion address**    → always SOCKS5-dialed through Tor (the primary win: as
//!   peers announce their `.onion` via Identify/Phase-A, those dials become onion
//!   streams automatically).
//! - **clearnet + tor_only** → SOCKS5-dialed through a Tor exit (stealth mode).
//! - **clearnet + hybrid**  → not handled here → libp2p dials it directly (TCP).
//!
//! `listen_on` is intentionally unsupported: inbound is served by the existing TCP
//! listener + the Phase-A onion hidden service, not by this dialer.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::future::{BoxFuture, Ready};
use libp2p::core::transport::{DialOpts, ListenerId, Transport, TransportError, TransportEvent};
use libp2p::Multiaddr;
use tokio_socks::tcp::Socks5Stream;
use tokio_util::compat::{Compat, TokioAsyncReadCompatExt};
use tracing::{debug, info, warn};

const ONION3_PROTOCOL: &str = "onion3";

/// Runtime configuration for the SOCKS5 dialer, built from env once at startup.
#[derive(Debug, Clone)]
pub struct Socks5DialConfig {
    /// Arti / tor SOCKS5 proxy address (Q_TOR_SOCKS5_ADDR, default 127.0.0.1:9050).
    pub socks_addr: SocketAddr,
    /// Master switch. When false the transport is inert (pure passthrough).
    pub enabled: bool,
    /// If true, clearnet (ip/dns) dials are ALSO routed through Tor (stealth mode).
    /// If false (hybrid), only `.onion` targets are Tor-routed and clearnet stays direct.
    pub tor_only: bool,
    /// Per-dial SOCKS connect timeout.
    pub timeout: Duration,
}

impl Default for Socks5DialConfig {
    fn default() -> Self {
        Self {
            socks_addr: "127.0.0.1:9050".parse().unwrap(),
            enabled: false,
            tor_only: false,
            timeout: Duration::from_secs(30),
        }
    }
}

impl Socks5DialConfig {
    /// Build from environment, matching the whitepaper's documented variables:
    ///   Q_ENABLE_TOR (master), Q_TOR_DISABLED (force off),
    ///   Q_TOR_SOCKS5_ADDR (proxy), Q_TOR_SOCKS5_DIAL (opt-in for THIS transport),
    ///   Q_TOR_ONLY (stealth — route clearnet through Tor too).
    ///
    /// Default-OFF: outbound SOCKS dialing only activates when explicitly opted in
    /// via `Q_TOR_SOCKS5_DIAL=1` (or `Q_TOR_ONLY=1`), so existing deployments are
    /// untouched until an operator turns it on.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("Q_TOR_SOCKS5_ADDR") {
            match v.parse::<SocketAddr>() {
                Ok(a) => cfg.socks_addr = a,
                Err(e) => warn!("🧅 Q_TOR_SOCKS5_ADDR='{v}' unparseable ({e}); using {}", cfg.socks_addr),
            }
        }

        let tor_disabled = env_bool("Q_TOR_DISABLED", false);
        let tor_enabled = env_bool("Q_ENABLE_TOR", true) && !tor_disabled;
        cfg.tor_only = env_bool("Q_TOR_ONLY", false);
        // Opt-in for outbound SOCKS dialing. tor_only implies it.
        let opt_in = env_bool("Q_TOR_SOCKS5_DIAL", false) || cfg.tor_only;
        cfg.enabled = tor_enabled && opt_in;

        if let Ok(v) = std::env::var("Q_TOR_BOOTSTRAP_TIMEOUT") {
            if let Ok(secs) = v.parse::<u64>() {
                if secs > 0 {
                    cfg.timeout = Duration::from_secs(secs);
                }
            }
        }

        if cfg.enabled {
            info!(
                "🧅 SOCKS5 outbound transport ENABLED — proxy={} tor_only={} timeout={:?}",
                cfg.socks_addr, cfg.tor_only, cfg.timeout
            );
        } else {
            debug!("🧅 SOCKS5 outbound transport inert (Q_TOR_SOCKS5_DIAL unset / Tor off)");
        }
        cfg
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => default,
    }
}

/// A dial-only libp2p `Transport` that connects through a Tor SOCKS5 proxy.
///
/// `Output = Compat<TcpStream>` so the standard `.upgrade().authenticate(noise)
/// .multiplex(yamux)` chain applies unchanged.
#[derive(Debug, Clone)]
pub struct Socks5DialTransport {
    config: Socks5DialConfig,
}

impl Socks5DialTransport {
    pub fn new(config: Socks5DialConfig) -> Self {
        Self { config }
    }

    fn is_onion_address(addr: &Multiaddr) -> bool {
        let s = addr.to_string();
        s.contains(ONION3_PROTOCOL) || s.contains(".onion")
    }

    /// Resolve the SOCKS5 CONNECT target ("host:port") for a dial, or `None` if
    /// this transport should NOT handle the address (→ libp2p uses TCP/QUIC).
    ///
    /// Mirrors `q_tor_client::libp2p_transport`'s string-based extraction so the two
    /// transports agree on the multiaddr format announced via Identify.
    fn resolve_target(&self, addr: &Multiaddr) -> Option<String> {
        if Self::is_onion_address(addr) {
            return Self::extract_onion_target(addr);
        }
        if self.config.tor_only {
            return Self::extract_tcp_target(addr);
        }
        // hybrid mode: leave clearnet to the direct TCP transport
        None
    }

    /// `/onion3/<base32>:<port>` or any `.onion` component → "<host>.onion:<port>".
    fn extract_onion_target(addr: &Multiaddr) -> Option<String> {
        let addr_str = addr.to_string();
        if addr_str.contains(ONION3_PROTOCOL) {
            let parts: Vec<&str> = addr_str.split('/').collect();
            for (i, part) in parts.iter().enumerate() {
                if *part == ONION3_PROTOCOL {
                    if let Some(onion_part) = parts.get(i + 1) {
                        if let Some((host, port)) = onion_part.split_once(':') {
                            return Some(format!("{host}.onion:{port}"));
                        }
                        return Some(format!("{onion_part}.onion:9001"));
                    }
                }
            }
        }
        // Direct ".onion" components (e.g. /dns4/<x>.onion/tcp/<port>)
        let mut host: Option<String> = None;
        let mut port: Option<u16> = None;
        for protocol in addr.iter() {
            use libp2p::multiaddr::Protocol::*;
            match protocol {
                Dns(h) | Dns4(h) | Dns6(h) if h.contains(".onion") => host = Some(h.to_string()),
                Tcp(p) => port = Some(p),
                _ => {}
            }
        }
        if let Some(h) = host {
            return Some(format!("{}:{}", h, port.unwrap_or(9001)));
        }
        // Fall back to grabbing any ".onion[:port]" substring from the string form.
        for component in addr_str.split('/') {
            if component.contains(".onion") {
                return Some(component.to_string());
            }
        }
        None
    }

    /// `/ip4|ip6|dns*/.../tcp/<port>` → "host:port".
    fn extract_tcp_target(addr: &Multiaddr) -> Option<String> {
        let mut host: Option<String> = None;
        let mut port: Option<u16> = None;
        for protocol in addr.iter() {
            use libp2p::multiaddr::Protocol::*;
            match protocol {
                Ip4(ip) => host = Some(ip.to_string()),
                Ip6(ip) => host = Some(ip.to_string()),
                Dns(h) | Dns4(h) | Dns6(h) => host = Some(h.to_string()),
                Tcp(p) => port = Some(p),
                _ => {}
            }
        }
        match (host, port) {
            (Some(h), Some(p)) => Some(format!("{h}:{p}")),
            _ => None,
        }
    }
}

impl Transport for Socks5DialTransport {
    type Output = Compat<tokio::net::TcpStream>;
    type Error = io::Error;
    type ListenerUpgrade = Ready<Result<Self::Output, Self::Error>>;
    type Dial = BoxFuture<'static, Result<Self::Output, Self::Error>>;

    fn listen_on(
        &mut self,
        _id: ListenerId,
        addr: Multiaddr,
    ) -> Result<(), TransportError<Self::Error>> {
        // Dial-only: inbound is the TCP listener + Phase-A onion hidden service.
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
        if !self.config.enabled {
            // Inert → libp2p tries the next transport (direct TCP/QUIC).
            return Err(TransportError::MultiaddrNotSupported(addr));
        }
        let target = match self.resolve_target(&addr) {
            Some(t) => t,
            None => return Err(TransportError::MultiaddrNotSupported(addr)),
        };
        let socks = self.config.socks_addr;
        let timeout = self.config.timeout;
        debug!("🧅 SOCKS5 dial {} → {} (via {})", addr, target, socks);

        Ok(Box::pin(async move {
            let connect = Socks5Stream::connect(socks, target.as_str());
            let stream = tokio::time::timeout(timeout, connect)
                .await
                .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "tor socks5 dial timeout"))?
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("tor socks5: {e}")))?;
            // Socks5Stream<TcpStream> → inner TcpStream → futures-compatible Compat.
            Ok(stream.into_inner().compat())
        }))
    }

    fn poll(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        // No listeners → never produces events.
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_by_default_and_inert() {
        let cfg = Socks5DialConfig::default();
        assert!(!cfg.enabled);
        let mut t = Socks5DialTransport::new(cfg);
        let addr: Multiaddr = "/ip4/1.2.3.4/tcp/9001".parse().unwrap();
        // Inert transport refuses every dial → libp2p falls through to TCP.
        assert!(matches!(
            t.dial(addr, DialOpts::default()),
            Err(TransportError::MultiaddrNotSupported(_))
        ));
    }

    #[test]
    fn hybrid_handles_onion_skips_clearnet() {
        let cfg = Socks5DialConfig { enabled: true, tor_only: false, ..Default::default() };
        let t = Socks5DialTransport::new(cfg);
        let onion: Multiaddr =
            "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd:9001"
                .parse()
                .unwrap_or_else(|_| "/dns4/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd.onion/tcp/9001".parse().unwrap());
        assert!(t.resolve_target(&onion).is_some());
        let clearnet: Multiaddr = "/ip4/1.2.3.4/tcp/9001".parse().unwrap();
        assert!(t.resolve_target(&clearnet).is_none(), "hybrid must leave clearnet to TCP");
    }

    #[test]
    fn tor_only_routes_clearnet_too() {
        let cfg = Socks5DialConfig { enabled: true, tor_only: true, ..Default::default() };
        let t = Socks5DialTransport::new(cfg);
        let clearnet: Multiaddr = "/ip4/1.2.3.4/tcp/9001".parse().unwrap();
        assert_eq!(t.resolve_target(&clearnet).as_deref(), Some("1.2.3.4:9001"));
    }

    #[test]
    fn extract_tcp_target_dns() {
        let a: Multiaddr = "/dns4/example.com/tcp/8080".parse().unwrap();
        assert_eq!(Socks5DialTransport::extract_tcp_target(&a).as_deref(), Some("example.com:8080"));
    }
}
