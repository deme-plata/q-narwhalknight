use anyhow::Result;
use parking_lot::RwLock;
use rustls::ServerConfig;
use std::io::BufReader;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;

use crate::config::TlsConfig;

// ─── TLS Hot-Reload (Issue #10) + OCSP Stapling + Drain ─────────────────

/// Inner state for SharedTlsConfig, behind a single Arc.
struct SharedTlsInner {
    config: RwLock<Arc<ServerConfig>>,
    /// Notifier fired on every TLS reload so interested parties (e.g. monitoring)
    /// can observe when old configs begin draining. Wrapped in Arc so callers
    /// can obtain a cheap handle via `subscribe_drain()`.
    drain_notify: Arc<tokio::sync::Notify>,
    /// Monotonic counter of how many TLS reloads have occurred.
    reload_count: AtomicU64,
}

/// Shared TLS config that can be atomically swapped for hot-reload.
/// Workers read the current config on each new TLS handshake.
///
/// On reload, existing connections continue using their captured `Arc<ServerConfig>`
/// (no forced termination). New connections automatically get the updated config.
/// The `drain_notify` is signalled so monitoring/logging can track when a reload happened.
#[derive(Clone)]
pub struct SharedTlsConfig {
    inner: Arc<SharedTlsInner>,
}

impl SharedTlsConfig {
    pub fn new(config: Arc<ServerConfig>) -> Self {
        Self {
            inner: Arc::new(SharedTlsInner {
                config: RwLock::new(config),
                drain_notify: Arc::new(tokio::sync::Notify::new()),
                reload_count: AtomicU64::new(0),
            }),
        }
    }

    /// Get the current TLS config (fast read lock).
    #[inline]
    pub fn load(&self) -> Arc<ServerConfig> {
        self.inner.config.read().clone()
    }

    /// Hot-reload TLS certificates from disk. Returns Ok with info string on success.
    ///
    /// After swapping the config, fires the drain notifier so watchers know that
    /// old connections are now using a stale config. Increments the reload counter.
    pub fn reload(&self, tls: &TlsConfig) -> Result<String> {
        let new_config = build_tls_config(tls)?;
        {
            let mut guard = self.inner.config.write();
            *guard = new_config;
        }
        // Increment reload counter and notify drain watchers
        let count = self.inner.reload_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.inner.drain_notify.notify_waiters();
        tracing::info!(
            reload_count = count,
            "TLS certificates hot-reloaded from {:?} (reload #{})",
            tls.cert,
            count,
        );
        Ok(format!("TLS reloaded from {} (reload #{})", tls.cert.display(), count))
    }

    /// Subscribe to drain notifications. The returned `Arc<tokio::sync::Notify>` is
    /// signalled every time `reload()` swaps in a new TLS config.
    ///
    /// Callers can `notified().await` in a select! to log when a reload occurs.
    /// Old connections are NOT forcefully closed -- they naturally drain as the old
    /// `Arc<ServerConfig>` reference count drops to zero.
    #[allow(dead_code)] // Public API for future worker drain integration
    pub fn subscribe_drain(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.inner.drain_notify)
    }

    /// Await a drain notification. Resolves when `reload()` is called.
    /// Can be used in `tokio::select!` to react to TLS config changes.
    pub async fn drain_notified(&self) {
        self.inner.drain_notify.notified().await;
    }

    /// Returns the number of TLS reloads that have occurred since startup.
    pub fn reload_count(&self) -> u64 {
        self.inner.reload_count.load(Ordering::Relaxed)
    }
}

/// Build a rustls ServerConfig from cert/key files.
/// The Arc<ServerConfig> is shared across all workers (rustls is thread-safe).
///
/// When `tls.ocsp_staple` is set, the OCSP response (DER-encoded) is read from disk
/// and stapled into the TLS handshake. This eliminates the 50-100ms OCSP lookup
/// penalty that clients would otherwise incur on first connection.
pub fn build_tls_config(tls: &TlsConfig) -> Result<Arc<ServerConfig>> {
    // Load certificate chain
    let cert_file = std::fs::File::open(&tls.cert)
        .map_err(|e| anyhow::anyhow!("Cannot open cert {}: {}", tls.cert.display(), e))?;
    let mut cert_reader = BufReader::new(cert_file);
    let certs: Vec<_> = rustls_pemfile::certs(&mut cert_reader)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Failed to parse certs: {}", e))?;

    if certs.is_empty() {
        anyhow::bail!("No certificates found in {}", tls.cert.display());
    }

    // Load private key
    let key_file = std::fs::File::open(&tls.key)
        .map_err(|e| anyhow::anyhow!("Cannot open key {}: {}", tls.key.display(), e))?;
    let mut key_reader = BufReader::new(key_file);
    let key = rustls_pemfile::private_key(&mut key_reader)
        .map_err(|e| anyhow::anyhow!("Failed to parse key: {}", e))?
        .ok_or_else(|| anyhow::anyhow!("No private key found in {}", tls.key.display()))?;

    // Build server config with session resumption for fast miner reconnects.
    // Session tickets: resumed handshake ~0.5ms vs ~2ms full handshake (75% faster).
    // Session cache: 1M entries shared across all workers via Arc<ServerConfig>.
    //
    // OCSP stapling: if an OCSP response file is configured, include it in the
    // TLS handshake via with_single_cert_with_ocsp(). Clients that support OCSP
    // stapling will use the stapled response instead of making a separate OCSP
    // request, saving 50-100ms on new connections.
    let mut config = if let Some(ref ocsp_path) = tls.ocsp_staple {
        let ocsp_der = std::fs::read(ocsp_path)
            .map_err(|e| anyhow::anyhow!("Cannot read OCSP staple {}: {}", ocsp_path.display(), e))?;

        if ocsp_der.is_empty() {
            tracing::warn!("OCSP staple file {} is empty, proceeding without OCSP stapling", ocsp_path.display());
            ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(certs, key)
                .map_err(|e| anyhow::anyhow!("TLS config error: {}", e))?
        } else {
            tracing::info!(
                "OCSP stapling enabled: {} ({} bytes)",
                ocsp_path.display(),
                ocsp_der.len(),
            );
            ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert_with_ocsp(certs, key, ocsp_der)
                .map_err(|e| anyhow::anyhow!("TLS config error (with OCSP): {}", e))?
        }
    } else {
        ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| anyhow::anyhow!("TLS config error: {}", e))?
    };

    // Enable TLS session tickets (key rotation is automatic)
    config.ticketer = rustls::crypto::ring::Ticketer::new()
        .map_err(|e| anyhow::anyhow!("Failed to create TLS ticketer: {}", e))?;

    // Shared session cache -- 1M sessions across all workers.
    // Each entry ~256 bytes -> ~256MB at full capacity.
    // With billions of miners cycling, a large cache means more TLS resumptions
    // (0.5ms resumed vs 2ms full handshake = 4x faster).
    config.session_storage = rustls::server::ServerSessionMemoryCache::new(1_048_576);

    // ALPN: advertise both HTTP/2 and HTTP/1.1 (Phase 3).
    // H2 listed first — modern browsers prefer it for multiplexing.
    // Miners using raw HTTP/1.1 clients negotiate h1 as fallback.
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

    tracing::info!(
        "TLS config: session tickets enabled, session cache 1M, ALPN [h2, http/1.1], OCSP {}",
        if tls.ocsp_staple.is_some() { "stapled" } else { "disabled" },
    );

    Ok(Arc::new(config))
}

/// Build a TLS config that uses SNI-based certificate selection for virtual hosting.
///
/// When the client sends an SNI extension in the ClientHello, rustls calls
/// `VhostRouter::resolve()` which returns the matching vhost cert. Falls back to
/// the default cert if no vhost matches. Inherits all other settings (session tickets,
/// ALPN, session cache) from the standard build.
pub fn build_tls_config_with_vhosts(
    vhost_router: Arc<crate::vhost::VhostRouter>,
) -> Result<Arc<ServerConfig>> {
    let mut config = ServerConfig::builder()
        .with_no_client_auth()
        .with_cert_resolver(vhost_router);

    config.ticketer = rustls::crypto::ring::Ticketer::new()
        .map_err(|e| anyhow::anyhow!("Failed to create TLS ticketer: {}", e))?;
    config.session_storage = rustls::server::ServerSessionMemoryCache::new(1_048_576);
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

    tracing::info!("TLS config with vhost SNI routing: session tickets, ALPN [h2, http/1.1]");
    Ok(Arc::new(config))
}

/// Build a TLS config that only advertises HTTP/1.1 (no h2).
/// Used for the libp2p WebSocket proxy port where the browser must use
/// HTTP/1.1 for the WebSocket upgrade handshake.
pub fn build_tls_config_h1_only(tls: &TlsConfig) -> Result<Arc<ServerConfig>> {
    let base = build_tls_config(tls)?;
    // Clone the config and override ALPN to h1-only
    let mut config = ServerConfig::clone(&base);
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    tracing::info!("LibP2P WS TLS config: ALPN [http/1.1] only (no h2)");
    Ok(Arc::new(config))
}

/// Create a TCP listener with SO_REUSEPORT + SO_REUSEADDR.
/// With SO_REUSEPORT, multiple workers can bind to the same port and the kernel
/// distributes incoming connections across them (no thundering herd).
pub fn create_listener(addr: &str) -> Result<socket2::Socket> {
    let sock_addr: SocketAddr = addr.parse()
        .map_err(|e| anyhow::anyhow!("Invalid listen address '{}': {}", addr, e))?;

    let domain = if sock_addr.is_ipv6() {
        socket2::Domain::IPV6
    } else {
        socket2::Domain::IPV4
    };

    let socket = socket2::Socket::new(domain, socket2::Type::STREAM, Some(socket2::Protocol::TCP))?;

    // Socket options for billion-miner scale
    socket.set_reuse_address(true)?;
    #[cfg(target_os = "linux")]
    {
        unsafe {
            let fd = socket.as_raw_fd();
            let one: libc::c_int = 1;
            let optlen = std::mem::size_of::<libc::c_int>() as libc::socklen_t;

            // SO_REUSEPORT: kernel distributes connections across workers (no thundering herd)
            libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEPORT,
                &one as *const _ as *const libc::c_void, optlen);

            // TCP_FASTOPEN: allow data in SYN packet (saves 1 RTT for repeat clients).
            // Queue length 256 = pending TFO connections before falling back to normal 3WHS.
            let tfo_qlen: libc::c_int = 256;
            libc::setsockopt(fd, libc::IPPROTO_TCP, libc::TCP_FASTOPEN,
                &tfo_qlen as *const _ as *const libc::c_void, optlen);

            // TCP_DEFER_ACCEPT: don't wake worker until client sends data (reduces accept() overhead).
            // Timeout in seconds -- kernel drops connections that send nothing within this window.
            let defer_secs: libc::c_int = 10;
            libc::setsockopt(fd, libc::IPPROTO_TCP, libc::TCP_DEFER_ACCEPT,
                &defer_secs as *const _ as *const libc::c_void, optlen);

            // Increase socket receive buffer (256KB for high-throughput mining submissions)
            let rcvbuf: libc::c_int = 262144;
            libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_RCVBUF,
                &rcvbuf as *const _ as *const libc::c_void, optlen);
        }
    }
    socket.set_nonblocking(true)?;
    socket.set_nodelay(true)?;

    socket.bind(&sock_addr.into())?;
    // Listen backlog 65535: max pending connections in the kernel queue.
    // At burst rates of 100K+ connections/sec, a small backlog drops connections.
    socket.listen(65535)?;

    tracing::info!(addr = %sock_addr, "Listener created with SO_REUSEPORT");
    Ok(socket)
}

/// Convert a socket2::Socket into a tokio TcpListener.
pub fn into_tokio_listener(socket: socket2::Socket) -> Result<tokio::net::TcpListener> {
    let std_listener: std::net::TcpListener = socket.into();
    let listener = tokio::net::TcpListener::from_std(std_listener)?;
    Ok(listener)
}

// ─── kTLS Kernel TLS Offload (Issue #017) ────────────────────────────────

/// kTLS feature detection result.
#[derive(Debug, Clone, Copy)]
pub struct KtlsFeatures {
    /// Whether the kernel supports kTLS (SOL_TLS exists).
    pub available: bool,
    /// Whether TLS_TX (transmit offload) is supported.
    pub tx_offload: bool,
    /// Whether TLS_RX (receive offload) is supported.
    pub rx_offload: bool,
}

impl Default for KtlsFeatures {
    fn default() -> Self {
        Self { available: false, tx_offload: false, rx_offload: false }
    }
}

impl std::fmt::Display for KtlsFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.available {
            write!(f, "kTLS available (TX: {}, RX: {})",
                if self.tx_offload { "yes" } else { "no" },
                if self.rx_offload { "yes" } else { "no" })
        } else {
            write!(f, "kTLS not available")
        }
    }
}

/// Probe the kernel for kTLS support.
///
/// Creates a temporary TCP socket and attempts `setsockopt(SOL_TLS, TLS_TX)`.
/// This is a non-destructive check — the socket is immediately dropped.
///
/// On non-Linux platforms, always returns `KtlsFeatures::default()` (not available).
pub fn probe_ktls() -> KtlsFeatures {
    #[cfg(target_os = "linux")]
    {
        // SOL_TLS = 282, TLS_TX = 1, TLS_RX = 2 (from linux/tls.h)
        const SOL_TLS: libc::c_int = 282;
        const TLS_TX: libc::c_int = 1;
        const TLS_RX: libc::c_int = 2;

        // Check if kTLS module is loaded by attempting setsockopt on a dummy socket.
        // If SOL_TLS doesn't exist (kernel < 4.13 or tls module not loaded),
        // setsockopt returns ENOPROTOOPT.
        let sock = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
        if sock < 0 {
            return KtlsFeatures::default();
        }

        // Try to set SOL_TLS level — just check if the protocol level exists.
        // We pass a minimal struct that will fail validation, but the error code
        // tells us if SOL_TLS is available:
        // - ENOPROTOOPT = kTLS not available
        // - EINVAL = kTLS available but bad params (this is what we want)
        // - ENOENT = kTLS available but TLS not established
        let dummy: [u8; 1] = [0];
        let tx_result = unsafe {
            libc::setsockopt(
                sock, SOL_TLS, TLS_TX,
                dummy.as_ptr() as *const libc::c_void, 1,
            )
        };
        let tx_errno = if tx_result < 0 {
            std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
        } else {
            0
        };

        let rx_result = unsafe {
            libc::setsockopt(
                sock, SOL_TLS, TLS_RX,
                dummy.as_ptr() as *const libc::c_void, 1,
            )
        };
        let rx_errno = if rx_result < 0 {
            std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
        } else {
            0
        };

        unsafe { libc::close(sock); }

        // ENOPROTOOPT (92) means SOL_TLS doesn't exist (kTLS not available).
        // Any other error (EINVAL, ENOENT) means kTLS IS available.
        let available = tx_errno != libc::ENOPROTOOPT;
        KtlsFeatures {
            available,
            tx_offload: available && tx_errno != libc::ENOPROTOOPT,
            rx_offload: available && rx_errno != libc::ENOPROTOOPT,
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        KtlsFeatures::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ktls_probe_returns_valid_features() {
        let features = probe_ktls();
        // Just verify it doesn't panic and returns a valid struct
        let desc = format!("{}", features);
        assert!(!desc.is_empty());
        // If TX is available, kTLS must be available
        if features.tx_offload {
            assert!(features.available);
        }
        if features.rx_offload {
            assert!(features.available);
        }
    }

    #[test]
    fn test_ktls_features_display_not_available() {
        let f = KtlsFeatures::default();
        assert_eq!(format!("{}", f), "kTLS not available");
    }

    #[test]
    fn test_ktls_features_display_available() {
        let f = KtlsFeatures { available: true, tx_offload: true, rx_offload: false };
        assert_eq!(format!("{}", f), "kTLS available (TX: yes, RX: no)");
    }
}
