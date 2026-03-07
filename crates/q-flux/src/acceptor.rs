use anyhow::Result;
use parking_lot::RwLock;
use rustls::ServerConfig;
use std::io::BufReader;
use std::net::SocketAddr;
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;

use crate::config::TlsConfig;

// ─── TLS Hot-Reload (Issue #10) ───────────────────────────────────────────

/// Shared TLS config that can be atomically swapped for hot-reload.
/// Workers read the current config on each new TLS handshake.
#[derive(Clone)]
pub struct SharedTlsConfig {
    inner: Arc<RwLock<Arc<ServerConfig>>>,
}

impl SharedTlsConfig {
    pub fn new(config: Arc<ServerConfig>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(config)),
        }
    }

    /// Get the current TLS config (fast read lock).
    #[inline]
    pub fn load(&self) -> Arc<ServerConfig> {
        self.inner.read().clone()
    }

    /// Hot-reload TLS certificates from disk. Returns Ok with info string on success.
    pub fn reload(&self, tls: &TlsConfig) -> Result<String> {
        let new_config = build_tls_config(tls)?;
        let mut guard = self.inner.write();
        *guard = new_config;
        tracing::info!("TLS certificates hot-reloaded from {:?}", tls.cert);
        Ok(format!("TLS reloaded from {}", tls.cert.display()))
    }
}

/// Build a rustls ServerConfig from cert/key files.
/// The Arc<ServerConfig> is shared across all workers (rustls is thread-safe).
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
    // Session cache: 65536 entries shared across all workers via Arc<ServerConfig>.
    let mut config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| anyhow::anyhow!("TLS config error: {}", e))?;

    // Enable TLS session tickets (key rotation is automatic)
    config.ticketer = rustls::crypto::ring::Ticketer::new()
        .map_err(|e| anyhow::anyhow!("Failed to create TLS ticketer: {}", e))?;

    // Shared session cache — 65536 sessions across all workers
    config.session_storage = rustls::server::ServerSessionMemoryCache::new(65536);

    // ALPN: advertise HTTP/1.1 (HTTP/2 in Phase 3)
    config.alpn_protocols = vec![b"http/1.1".to_vec()];

    tracing::info!(
        "TLS config: session tickets enabled, session cache 65536, ALPN [http/1.1]"
    );

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

    // Socket options for high-performance
    socket.set_reuse_address(true)?;
    #[cfg(target_os = "linux")]
    {
        // SO_REUSEPORT: kernel distributes connections across workers
        unsafe {
            let optval: libc::c_int = 1;
            libc::setsockopt(
                socket.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_REUSEPORT,
                &optval as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
        }
    }
    socket.set_nonblocking(true)?;
    socket.set_nodelay(true)?;

    socket.bind(&sock_addr.into())?;
    socket.listen(4096)?;

    tracing::info!(addr = %sock_addr, "Listener created with SO_REUSEPORT");
    Ok(socket)
}

/// Convert a socket2::Socket into a tokio TcpListener.
pub fn into_tokio_listener(socket: socket2::Socket) -> Result<tokio::net::TcpListener> {
    let std_listener: std::net::TcpListener = socket.into();
    let listener = tokio::net::TcpListener::from_std(std_listener)?;
    Ok(listener)
}
