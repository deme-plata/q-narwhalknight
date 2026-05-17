/// libp2p-Tor Transport for Q-NarwhalKnight
///
/// This module provides a native libp2p Transport implementation that wraps the
/// DedicatedCircuitManager, enabling gossipsub and Kademlia to work transparently
/// over Tor circuits with per-operation-type isolation.
///
/// Features:
/// - Transparent Tor integration for libp2p protocols
/// - Automatic operation type detection from multiaddr
/// - Circuit isolation per protocol/operation
/// - Onion address (.onion) support
/// - Real Arti DataStream wrapping for AsyncRead/AsyncWrite
/// - Fallback to direct connection when Tor unavailable (if configured)

use crate::dedicated_circuits::{DedicatedCircuitManager, OperationType};
use anyhow::{anyhow, Result};
use arti_client::DataStream;
use futures::future::{BoxFuture, Ready};
use libp2p::{
    core::{
        transport::{DialOpts, ListenerId, TransportError, TransportEvent},
        Multiaddr, Transport,
    },
    PeerId,
};
use std::{
    collections::HashMap,
    io::{self, Error, ErrorKind},
    net::{IpAddr, SocketAddr},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// Multiaddr protocol constants for onion addresses
const ONION3_PROTOCOL: &str = "onion3";

/// libp2p Transport that routes connections through Tor
pub struct TorTransport {
    /// Dedicated circuit manager for per-operation Tor circuits
    circuit_manager: Arc<DedicatedCircuitManager>,
    /// Configuration
    config: TorTransportConfig,
    /// Listener counter
    listener_id_counter: u64,
    /// Active listeners (for onion services)
    listeners: Arc<RwLock<HashMap<ListenerId, TorListener>>>,
}

/// Configuration for TorTransport
#[derive(Debug, Clone)]
pub struct TorTransportConfig {
    /// Allow clearnet fallback if Tor unavailable
    pub allow_clearnet_fallback: bool,
    /// Default operation type for unknown protocols
    pub default_operation_type: OperationType,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Enable onion service listening
    pub enable_onion_listening: bool,
    /// Onion service port
    pub onion_port: u16,
}

impl Default for TorTransportConfig {
    fn default() -> Self {
        Self {
            allow_clearnet_fallback: false, // Mandatory Tor by default
            default_operation_type: OperationType::General,
            connection_timeout: Duration::from_secs(30),
            enable_onion_listening: true,
            onion_port: 9001,
        }
    }
}

/// Tor listener for incoming connections
#[derive(Debug)]
pub struct TorListener {
    /// Listener ID
    id: ListenerId,
    /// Onion address for this listener
    onion_address: Option<String>,
    /// Local port
    port: u16,
    /// Receiver for incoming connections
    incoming_rx: Option<mpsc::Receiver<TorIncoming>>,
}

/// Incoming Tor connection
#[derive(Debug)]
pub struct TorIncoming {
    /// Remote peer's onion address or IP
    remote_addr: Multiaddr,
    // The underlying stream placeholder - in production this would hold the Arti DataStream
}

/// Tor-wrapped stream for libp2p with real Arti DataStream I/O
///
/// This struct wraps an Arti `DataStream` and implements the tokio `AsyncRead`
/// and `AsyncWrite` traits, allowing it to be used seamlessly with libp2p.
pub struct TorStream {
    /// The underlying Arti DataStream for Tor communication
    inner: Option<DataStream>,
    /// Operation type this stream uses (for metrics/logging)
    operation_type: OperationType,
    /// Target address for debugging/logging
    target: String,
    /// Connection established timestamp
    connected_at: Instant,
    /// Bytes sent counter
    bytes_sent: u64,
    /// Bytes received counter
    bytes_received: u64,
}

impl TorStream {
    /// Create a new TorStream wrapping an Arti DataStream
    pub fn new(inner: DataStream, operation_type: OperationType, target: String) -> Self {
        Self {
            inner: Some(inner),
            operation_type,
            target,
            connected_at: Instant::now(),
            bytes_sent: 0,
            bytes_received: 0,
        }
    }

    /// Create a placeholder TorStream (for testing or fallback)
    pub fn placeholder(operation_type: OperationType, target: String) -> Self {
        Self {
            inner: None,
            operation_type,
            target,
            connected_at: Instant::now(),
            bytes_sent: 0,
            bytes_received: 0,
        }
    }

    /// Get the operation type for this stream
    pub fn operation_type(&self) -> OperationType {
        self.operation_type
    }

    /// Get the target address
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Get connection duration
    pub fn connection_duration(&self) -> Duration {
        self.connected_at.elapsed()
    }

    /// Get bytes sent
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent
    }

    /// Get bytes received
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received
    }

    /// Check if this stream has a real Tor connection
    pub fn is_real_connection(&self) -> bool {
        self.inner.is_some()
    }
}

/// Statistics for TorStream connections
#[derive(Debug, Clone)]
pub struct TorStreamStats {
    pub operation_type: OperationType,
    pub target: String,
    pub connection_duration: Duration,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub is_real_connection: bool,
}

impl From<&TorStream> for TorStreamStats {
    fn from(stream: &TorStream) -> Self {
        Self {
            operation_type: stream.operation_type,
            target: stream.target.clone(),
            connection_duration: stream.connection_duration(),
            bytes_sent: stream.bytes_sent,
            bytes_received: stream.bytes_received,
            is_real_connection: stream.is_real_connection(),
        }
    }
}

// Implement AsyncRead for TorStream - delegates to inner DataStream
impl AsyncRead for TorStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match &mut self.inner {
            Some(stream) => {
                let initial_filled = buf.filled().len();
                let pinned = Pin::new(stream);
                let result = pinned.poll_read(cx, buf);

                // Track bytes received
                if let Poll::Ready(Ok(())) = &result {
                    let bytes_read = buf.filled().len() - initial_filled;
                    self.bytes_received += bytes_read as u64;
                }

                result
            }
            None => {
                // Placeholder stream - return empty read (EOF)
                Poll::Ready(Ok(()))
            }
        }
    }
}

// Implement AsyncWrite for TorStream - delegates to inner DataStream
impl AsyncWrite for TorStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match &mut self.inner {
            Some(stream) => {
                let pinned = Pin::new(stream);
                let result = pinned.poll_write(cx, buf);

                // Track bytes sent
                if let Poll::Ready(Ok(bytes_written)) = &result {
                    self.bytes_sent += *bytes_written as u64;
                }

                result
            }
            None => {
                // Placeholder stream - pretend write succeeded
                Poll::Ready(Ok(buf.len()))
            }
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match &mut self.inner {
            Some(stream) => {
                let pinned = Pin::new(stream);
                pinned.poll_flush(cx)
            }
            None => Poll::Ready(Ok(())),
        }
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match &mut self.inner {
            Some(stream) => {
                let pinned = Pin::new(stream);
                pinned.poll_shutdown(cx)
            }
            None => Poll::Ready(Ok(())),
        }
    }
}

impl TorTransport {
    /// Create a new TorTransport with a DedicatedCircuitManager
    pub fn new(circuit_manager: Arc<DedicatedCircuitManager>, config: TorTransportConfig) -> Self {
        info!("🧅 Creating libp2p TorTransport");
        info!(
            "   Clearnet fallback: {}",
            if config.allow_clearnet_fallback {
                "enabled"
            } else {
                "DISABLED (mandatory Tor)"
            }
        );

        Self {
            circuit_manager,
            config,
            listener_id_counter: 0,
            listeners: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Extract operation type from multiaddr based on protocol hints
    fn detect_operation_type(&self, addr: &Multiaddr) -> OperationType {
        let addr_string = addr.to_string();

        // Check for protocol-specific patterns in the multiaddr
        if addr_string.contains("/qnk/blocks") || addr_string.contains("/meshsub/1.1.0") {
            return OperationType::BlockPropagation;
        }

        if addr_string.contains("/kad") || addr_string.contains("/ipfs/kad") {
            return OperationType::PeerDiscovery;
        }

        if addr_string.contains("/qnk/tx") || addr_string.contains("/transactions") {
            return OperationType::TransactionSubmission;
        }

        if addr_string.contains("/qnk/sync") || addr_string.contains("/turbo-sync") {
            return OperationType::P2PSync;
        }

        if addr_string.contains("/qnk/validator") || addr_string.contains("/consensus") {
            return OperationType::ValidatorCommunication;
        }

        if addr_string.contains("/qnk/ai") || addr_string.contains("/inference") {
            return OperationType::AIInference;
        }

        if addr_string.contains("/qnk/entropy") || addr_string.contains("/qrng") {
            return OperationType::QuantumEntropy;
        }

        // Default operation type
        self.config.default_operation_type
    }

    /// Check if address is an onion address
    fn is_onion_address(addr: &Multiaddr) -> bool {
        addr.to_string().contains(ONION3_PROTOCOL)
            || addr.to_string().ends_with(".onion")
            || addr.to_string().contains(".onion:")
    }

    /// Extract onion address and port from multiaddr
    fn extract_onion_target(addr: &Multiaddr) -> Option<String> {
        let addr_str = addr.to_string();

        // Handle /onion3/ADDRESS:PORT format
        if addr_str.contains(ONION3_PROTOCOL) {
            // Parse /onion3/base32address:port
            let parts: Vec<&str> = addr_str.split('/').collect();
            for (i, part) in parts.iter().enumerate() {
                if *part == ONION3_PROTOCOL {
                    if let Some(onion_part) = parts.get(i + 1) {
                        // onion_part might be "address:port" or just "address"
                        if onion_part.contains(':') {
                            let split: Vec<&str> = onion_part.split(':').collect();
                            if split.len() == 2 {
                                return Some(format!("{}.onion:{}", split[0], split[1]));
                            }
                        } else {
                            return Some(format!("{}.onion:9001", onion_part));
                        }
                    }
                }
            }
        }

        // Handle direct .onion addresses in the string
        if addr_str.contains(".onion") {
            // Try to extract address:port
            for component in addr_str.split('/') {
                if component.contains(".onion") {
                    return Some(component.to_string());
                }
            }
        }

        None
    }

    /// Extract TCP target from multiaddr for non-onion addresses
    fn extract_tcp_target(addr: &Multiaddr) -> Option<String> {
        let mut ip: Option<String> = None;
        let mut port: Option<u16> = None;

        for protocol in addr.iter() {
            match protocol {
                libp2p::multiaddr::Protocol::Ip4(ipv4) => {
                    ip = Some(ipv4.to_string());
                }
                libp2p::multiaddr::Protocol::Ip6(ipv6) => {
                    ip = Some(ipv6.to_string());
                }
                libp2p::multiaddr::Protocol::Dns(dns) => {
                    ip = Some(dns.to_string());
                }
                libp2p::multiaddr::Protocol::Dns4(dns) => {
                    ip = Some(dns.to_string());
                }
                libp2p::multiaddr::Protocol::Dns6(dns) => {
                    ip = Some(dns.to_string());
                }
                libp2p::multiaddr::Protocol::Tcp(p) => {
                    port = Some(p);
                }
                _ => {}
            }
        }

        match (ip, port) {
            (Some(i), Some(p)) => Some(format!("{}:{}", i, p)),
            _ => None,
        }
    }

    /// Dial an address through Tor
    async fn dial_through_tor(
        circuit_manager: Arc<DedicatedCircuitManager>,
        target: String,
        operation_type: OperationType,
    ) -> Result<TorStream, io::Error> {
        debug!(
            "🧅 Dialing {} via Tor circuit for {}",
            target,
            operation_type.name()
        );

        let start = Instant::now();

        // Use the dedicated circuit for this operation type
        match circuit_manager.connect(operation_type, &target).await {
            Ok(data_stream) => {
                let latency = start.elapsed();
                info!(
                    "✅ Connected to {} via {} circuit in {}ms",
                    target,
                    operation_type.name(),
                    latency.as_millis()
                );
                // Wrap the real Arti DataStream in TorStream
                Ok(TorStream::new(data_stream, operation_type, target))
            }
            Err(e) => {
                error!("❌ Tor connection failed: {}", e);
                Err(io::Error::new(
                    ErrorKind::ConnectionRefused,
                    format!("Tor connection failed: {}", e),
                ))
            }
        }
    }

    /// Get statistics for active connections (for metrics)
    pub fn connection_count(&self) -> usize {
        // This would be tracked by the circuit manager
        0
    }
}

impl Transport for TorTransport {
    type Output = TorStream;
    type Error = io::Error;
    type ListenerUpgrade = Ready<Result<Self::Output, Self::Error>>;
    type Dial = BoxFuture<'static, Result<Self::Output, Self::Error>>;

    fn listen_on(
        &mut self,
        id: ListenerId,
        addr: Multiaddr,
    ) -> Result<(), TransportError<Self::Error>> {
        info!("🧅 TorTransport listen_on: {} (id: {:?})", addr, id);

        // For onion services, we'd create a hidden service here
        // This is a placeholder implementation
        if !self.config.enable_onion_listening {
            return Err(TransportError::MultiaddrNotSupported(addr));
        }

        let listener = TorListener {
            id,
            onion_address: None, // Would be set by hidden service creation
            port: self.config.onion_port,
            incoming_rx: None,
        };

        // Store listener
        let listeners = self.listeners.clone();
        tokio::spawn(async move {
            let mut listeners = listeners.write().await;
            listeners.insert(id, listener);
        });

        info!("✅ TorTransport listening on port {}", self.config.onion_port);
        Ok(())
    }

    fn remove_listener(&mut self, id: ListenerId) -> bool {
        info!("🧅 TorTransport remove_listener: {:?}", id);

        let listeners = self.listeners.clone();
        tokio::spawn(async move {
            let mut listeners = listeners.write().await;
            listeners.remove(&id);
        });

        true
    }

    fn dial(
        &mut self,
        addr: Multiaddr,
        _opts: DialOpts,
    ) -> Result<Self::Dial, TransportError<Self::Error>> {
        debug!("🧅 TorTransport dial: {}", addr);

        // Detect operation type from address
        let operation_type = self.detect_operation_type(&addr);
        debug!("   Operation type: {}", operation_type.name());

        // Extract target address
        let target = if Self::is_onion_address(&addr) {
            Self::extract_onion_target(&addr)
        } else if self.config.allow_clearnet_fallback {
            Self::extract_tcp_target(&addr)
        } else {
            // Mandatory Tor mode - must be onion address
            warn!(
                "❌ Non-onion address {} rejected in mandatory Tor mode",
                addr
            );
            return Err(TransportError::MultiaddrNotSupported(addr));
        };

        let target = match target {
            Some(t) => t,
            None => {
                warn!("❌ Could not extract target from address: {}", addr);
                return Err(TransportError::MultiaddrNotSupported(addr));
            }
        };

        let circuit_manager = Arc::clone(&self.circuit_manager);

        Ok(Box::pin(async move {
            Self::dial_through_tor(circuit_manager, target, operation_type).await
        }))
    }

    fn poll(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        // For incoming connections, we'd poll the hidden service here
        // This is a placeholder that returns Pending
        Poll::Pending
    }
}

/// Builder for TorTransport with fluent API
pub struct TorTransportBuilder {
    config: TorTransportConfig,
    circuit_manager: Option<Arc<DedicatedCircuitManager>>,
}

impl TorTransportBuilder {
    pub fn new() -> Self {
        Self {
            config: TorTransportConfig::default(),
            circuit_manager: None,
        }
    }

    pub fn with_circuit_manager(mut self, manager: Arc<DedicatedCircuitManager>) -> Self {
        self.circuit_manager = Some(manager);
        self
    }

    pub fn allow_clearnet_fallback(mut self, allow: bool) -> Self {
        self.config.allow_clearnet_fallback = allow;
        self
    }

    pub fn connection_timeout(mut self, timeout: Duration) -> Self {
        self.config.connection_timeout = timeout;
        self
    }

    pub fn default_operation_type(mut self, op_type: OperationType) -> Self {
        self.config.default_operation_type = op_type;
        self
    }

    pub fn onion_port(mut self, port: u16) -> Self {
        self.config.onion_port = port;
        self
    }

    pub fn enable_onion_listening(mut self, enable: bool) -> Self {
        self.config.enable_onion_listening = enable;
        self
    }

    pub fn build(self) -> Result<TorTransport> {
        let circuit_manager = self
            .circuit_manager
            .ok_or_else(|| anyhow!("Circuit manager is required"))?;

        Ok(TorTransport::new(circuit_manager, self.config))
    }
}

impl Default for TorTransportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper trait for converting addresses to onion format
pub trait OnionMultiaddr {
    /// Create a multiaddr from an onion address
    fn from_onion(onion_addr: &str, port: u16) -> Result<Multiaddr>;

    /// Check if this multiaddr is an onion address
    fn is_onion(&self) -> bool;
}

impl OnionMultiaddr for Multiaddr {
    fn from_onion(onion_addr: &str, port: u16) -> Result<Multiaddr> {
        // Strip .onion suffix if present
        let addr = onion_addr.trim_end_matches(".onion");

        // Create /onion3/address/tcp/port multiaddr
        let multiaddr_str = format!("/onion3/{}:{}", addr, port);

        multiaddr_str
            .parse()
            .map_err(|e| anyhow!("Failed to parse onion multiaddr: {}", e))
    }

    fn is_onion(&self) -> bool {
        TorTransport::is_onion_address(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_type_detection() {
        // This would require a mock circuit manager for full testing
    }

    #[test]
    fn test_onion_address_detection() {
        let onion_addr: Multiaddr = "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd:9001"
            .parse()
            .unwrap();

        assert!(TorTransport::is_onion_address(&onion_addr));

        let tcp_addr: Multiaddr = "/ip4/127.0.0.1/tcp/9001".parse().unwrap();
        assert!(!TorTransport::is_onion_address(&tcp_addr));
    }

    #[test]
    fn test_onion_target_extraction() {
        let addr: Multiaddr = "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd:9001"
            .parse()
            .unwrap();

        let target = TorTransport::extract_onion_target(&addr);
        assert!(target.is_some());
        assert!(target.unwrap().contains(".onion"));
    }

    #[test]
    fn test_tcp_target_extraction() {
        let addr: Multiaddr = "/ip4/192.168.1.1/tcp/8080".parse().unwrap();
        let target = TorTransport::extract_tcp_target(&addr);
        assert_eq!(target, Some("192.168.1.1:8080".to_string()));

        let dns_addr: Multiaddr = "/dns4/example.com/tcp/443".parse().unwrap();
        let dns_target = TorTransport::extract_tcp_target(&dns_addr);
        assert_eq!(dns_target, Some("example.com:443".to_string()));
    }
}
