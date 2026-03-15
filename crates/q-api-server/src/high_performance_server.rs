/// High-Performance HTTP Server for 1M+ TPS
///
/// This module provides optimized TCP socket configuration for Axum
/// designed for extreme transaction throughput:
///
/// - Optimized TCP socket configuration (NODELAY, keepalive, REUSEADDR)
/// - v1.0.53: Automatic port detection if requested port is in use
/// - v8.9.8: Reduced TCP buffers (4MB→256KB) to prevent 17GB kernel memory
///           usage with 75K+ connections. Added connection-aware backlog.
/// - v9.1.2: Manual accept loop with hyper HTTP/1.1 header_read_timeout (30s)
///           to auto-close idle kept-alive connections. Prevents connection
///           pileup from reverse proxies (Caddy/nginx).
///
/// Target Performance: 1,000,000+ TPS with binary protocol
use axum::Router;
use axum::extract::connect_info::ConnectInfo;
use hyper::body::Incoming;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto::Builder as AutoBuilder;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::net::TcpListener;
use tower::Service;
use tracing::{error, info, warn};

/// High-performance HTTP server configuration
pub struct HighPerformanceServer {
    app: Router,
    addr: SocketAddr,
    tcp_recv_buffer_size: usize,
    tcp_send_buffer_size: usize,
    tcp_backlog: u32,
    /// v1.0.53: Enable automatic port detection if requested port is in use
    auto_port_detection: bool,
    /// Maximum number of ports to try when auto-detecting
    max_port_attempts: u16,
    /// v9.1.2: HTTP/1.1 keepalive idle timeout (seconds)
    http_keepalive_timeout_secs: u64,
    /// v9.8.4: Hard cap on concurrent connections (env: MAX_HTTP_CONNECTIONS)
    max_connections: usize,
}

impl HighPerformanceServer {
    /// Create a new high-performance server with optimal defaults
    pub fn new(app: Router, addr: SocketAddr) -> Self {
        Self {
            app,
            addr,
            // v8.9.8: Reduced from 4MB to 256KB per connection
            // 4MB × 75K connections = 300GB kernel pressure (caused 20GB RSS, load 52)
            // 256KB is more than enough for JSON API responses (typically < 1KB)
            tcp_recv_buffer_size: 256 * 1024,  // 256KB receive buffer
            tcp_send_buffer_size: 256 * 1024,  // 256KB send buffer
            tcp_backlog: 4096,                 // v8.9.8: 4096 pending connections (was 1024)
            auto_port_detection: true,         // v1.0.53: Enable by default
            max_port_attempts: 10,             // Try up to 10 ports
            http_keepalive_timeout_secs: 30,   // v9.1.2: Close idle connections after 30s
            max_connections: std::env::var("MAX_HTTP_CONNECTIONS")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(8192),
        }
    }

    /// Configure TCP buffer sizes
    pub fn with_tcp_buffers(mut self, recv_size: usize, send_size: usize) -> Self {
        self.tcp_recv_buffer_size = recv_size;
        self.tcp_send_buffer_size = send_size;
        self
    }

    /// Configure TCP backlog (pending connection queue)
    pub fn with_backlog(mut self, backlog: u32) -> Self {
        self.tcp_backlog = backlog;
        self
    }

    /// v1.0.53: Enable or disable automatic port detection
    /// When enabled, if the requested port is in use, the server will try
    /// subsequent ports until it finds an available one.
    pub fn with_auto_port_detection(mut self, enabled: bool) -> Self {
        self.auto_port_detection = enabled;
        self
    }

    /// Configure maximum number of ports to try when auto-detecting
    pub fn with_max_port_attempts(mut self, attempts: u16) -> Self {
        self.max_port_attempts = attempts;
        self
    }

    /// Run the high-performance server
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚀 Starting High-Performance HTTP Server");
        info!("   Requested address: {}", self.addr);
        info!(
            "   TCP buffer size: {} KB recv, {} KB send",
            self.tcp_recv_buffer_size / 1024,
            self.tcp_send_buffer_size / 1024
        );
        info!("   TCP backlog: {} pending connections", self.tcp_backlog);
        info!("   HTTP/1.1 keepalive timeout: {}s (idle connections auto-close)", self.http_keepalive_timeout_secs);
        info!("   Max connections: {} (env: MAX_HTTP_CONNECTIONS)", self.max_connections);
        info!("   Target throughput: 1,000,000+ TPS");
        if self.auto_port_detection {
            info!("   Auto port detection: ENABLED (will try up to {} ports)", self.max_port_attempts);
        }

        // v1.0.53: Try to bind with automatic port detection
        let (listener, actual_addr) = self.try_bind_with_auto_detection().await?;

        info!("✅ TCP listener configured and ready");
        info!("🌟 High-Performance Server READY - accepting connections...");
        info!("   Listening on: http://{}", actual_addr);

        // v9.1.2: Manual accept loop with hyper HTTP/1.1 header_read_timeout
        // This prevents idle connection accumulation from reverse proxies.
        // axum::serve() doesn't expose this setting, so we use hyper-util directly.
        let keepalive_timeout = std::time::Duration::from_secs(self.http_keepalive_timeout_secs);

        let app = self.app;
        let active_connections = Arc::new(AtomicUsize::new(0));

        // Create shutdown signal
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let shutdown_trigger = shutdown.clone();

        // Spawn signal handler
        tokio::spawn(async move {
            let ctrl_c = async {
                tokio::signal::ctrl_c()
                    .await
                    .expect("failed to install CTRL+C signal handler");
            };

            #[cfg(unix)]
            let terminate = async {
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to install SIGTERM signal handler")
                    .recv()
                    .await;
            };

            #[cfg(not(unix))]
            let terminate = std::future::pending::<()>();

            #[cfg(unix)]
            let usr1 = async {
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined1())
                    .expect("failed to install SIGUSR1 signal handler")
                    .recv()
                    .await;
            };

            #[cfg(not(unix))]
            let usr1 = std::future::pending::<()>();

            #[cfg(unix)]
            let usr2 = async {
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined2())
                    .expect("failed to install SIGUSR2 signal handler")
                    .recv()
                    .await;
            };

            #[cfg(not(unix))]
            let usr2 = std::future::pending::<()>();

            tokio::select! {
                _ = ctrl_c => {
                    info!("🛑 Received CTRL+C signal - initiating graceful shutdown");
                },
                _ = terminate => {
                    info!("🛑 Received SIGTERM signal - initiating graceful shutdown");
                },
                _ = usr1 => {
                    info!("🔄 Received SIGUSR1 signal - initiating graceful restart (auto-update)");
                },
                _ = usr2 => {
                    warn!("⏪ Received SIGUSR2 signal - initiating rollback restart");
                },
            }

            shutdown_trigger.notify_waiters();
        });

        // Log connection count periodically
        let active_conns_log = active_connections.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                let count = active_conns_log.load(Ordering::Relaxed);
                if count > 1000 {
                    warn!("⚠️  Active HTTP connections: {} (high)", count);
                } else {
                    info!("📊 Active HTTP connections: {}", count);
                }
            }
        });

        // v9.8.4: Capture max_connections for the accept loop
        let max_connections = self.max_connections;

        // Accept loop
        loop {
            tokio::select! {
                result = listener.accept() => {
                    let (stream, remote_addr) = match result {
                        Ok(conn) => conn,
                        Err(e) => {
                            // Accept errors are usually transient (too many FDs, etc.)
                            warn!("Failed to accept connection: {}", e);
                            continue;
                        }
                    };

                    // v9.8.4: Hard connection cap — reject if above limit
                    let current = active_connections.load(Ordering::Relaxed);
                    if current >= max_connections {
                        warn!("🚫 Connection limit reached ({}/{}), rejecting {}", current, max_connections, remote_addr);
                        drop(stream);
                        continue;
                    }

                    let io = TokioIo::new(stream);
                    let app = app.clone();
                    let conns = active_connections.clone();
                    conns.fetch_add(1, Ordering::Relaxed);

                    tokio::spawn(async move {
                        // Build hyper HTTP server with keepalive timeout
                        let mut builder = AutoBuilder::new(TokioExecutor::new());
                        builder.http1()
                            .keep_alive(true)
                            .timer(TokioTimer::new())
                            .header_read_timeout(keepalive_timeout);

                        // Convert axum Router to hyper service, injecting ConnectInfo
                        let tower_service = app;
                        let hyper_service = hyper::service::service_fn(move |mut req: hyper::Request<Incoming>| {
                            // Inject ConnectInfo so extractors like ConnectInfo<SocketAddr> work
                            req.extensions_mut().insert(ConnectInfo(remote_addr));
                            tower_service.clone().call(req)
                        });

                        if let Err(e) = builder
                            .serve_connection_with_upgrades(io, hyper_service)
                            .await
                        {
                            // Don't log normal connection closes
                            let msg = e.to_string();
                            if !msg.contains("connection closed")
                                && !msg.contains("broken pipe")
                                && !msg.contains("reset by peer")
                                && !msg.contains("timed out")
                            {
                                warn!("Connection error from {}: {}", remote_addr, msg);
                            }
                        }

                        conns.fetch_sub(1, Ordering::Relaxed);
                    });
                }
                _ = shutdown.notified() => {
                    info!("🛑 Stopping accept loop...");
                    break;
                }
            }
        }

        // Wait briefly for in-flight connections to finish
        let remaining = active_connections.load(Ordering::Relaxed);
        if remaining > 0 {
            info!("⏳ Waiting for {} active connections to drain...", remaining);
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let still_remaining = active_connections.load(Ordering::Relaxed);
            if still_remaining > 0 {
                warn!("⚠️  {} connections still active after 5s drain, shutting down anyway", still_remaining);
            }
        }

        info!("✅ Server shutdown completed");
        Ok(())
    }

    /// v1.0.53: Try to bind to the requested port, with automatic fallback to alternative ports
    async fn try_bind_with_auto_detection(&self) -> Result<(TcpListener, SocketAddr), Box<dyn std::error::Error>> {
        let base_port = self.addr.port();
        let ip = self.addr.ip();

        for attempt in 0..self.max_port_attempts {
            let try_port = base_port.saturating_add(attempt);
            let try_addr = SocketAddr::new(ip, try_port);

            match self.try_bind_single_port(try_addr).await {
                Ok((listener, addr)) => {
                    if attempt > 0 {
                        warn!("⚠️  Port {} was in use, bound to port {} instead", base_port, try_port);
                        warn!("   To avoid this, ensure no other process is using port {}", base_port);
                        warn!("   Or specify a different port with --port <PORT>");
                    }
                    return Ok((listener, addr));
                }
                Err(e) => {
                    // Check if it's an "address in use" error
                    let err_string = e.to_string();
                    let is_addr_in_use = err_string.contains("Address already in use")
                        || err_string.contains("AddrInUse")
                        || err_string.contains("os error 98")
                        || err_string.contains("os error 48"); // macOS

                    if is_addr_in_use && self.auto_port_detection && attempt < self.max_port_attempts - 1 {
                        warn!("⚠️  Port {} is in use, trying port {}...", try_port, try_port + 1);
                        continue;
                    } else if is_addr_in_use && !self.auto_port_detection {
                        error!("❌ Port {} is already in use!", try_port);
                        error!("   Solutions:");
                        error!("   1. Kill the process using the port: sudo lsof -i :{} | grep LISTEN", try_port);
                        error!("   2. Use a different port: --port {}", try_port + 1);
                        error!("   3. Wait for the previous process to fully terminate");
                        return Err(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(format!(
            "Failed to bind to any port in range {}-{}. All ports are in use.",
            base_port,
            base_port.saturating_add(self.max_port_attempts - 1)
        ).into())
    }

    /// Try to bind to a single port with all TCP optimizations
    async fn try_bind_single_port(&self, addr: SocketAddr) -> Result<(TcpListener, SocketAddr), Box<dyn std::error::Error>> {
        // Create socket using socket2 for advanced configuration
        let socket = socket2::Socket::new(
            socket2::Domain::IPV4,
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        )?;

        // TCP_NODELAY - Disable Nagle's algorithm for low latency
        socket.set_nodelay(true)?;
        info!("   ✓ TCP_NODELAY enabled (eliminates 40ms delay)");

        // SO_REUSEADDR - Allow rapid restart
        socket.set_reuse_address(true)?;
        info!("   ✓ SO_REUSEADDR enabled");

        // v8.2.2: SO_REUSEPORT DISABLED — it prevents binding over TIME_WAIT sockets,
        // which causes the server to silently move to a different port after restarts.
        // We only run one server instance, so kernel load balancing isn't needed.
        // SO_REUSEADDR alone is sufficient for rapid restart.

        // v8.2.3: TCP keepalive — detect dead connections (SSE CLOSE-WAIT leak fix)
        // Without this, disconnected SSE clients leave CLOSE-WAIT sockets that accumulate
        // until the accept queue is saturated and the server stops accepting new connections.
        socket.set_keepalive(true)?;
        {
            let mut keepalive = socket2::TcpKeepalive::new()
                .with_time(std::time::Duration::from_secs(30))      // Start probing after 30s idle
                .with_interval(std::time::Duration::from_secs(10)); // Probe every 10s
            #[cfg(target_os = "linux")]
            {
                keepalive = keepalive.with_retries(3);               // Give up after 3 failed probes
            }
            let _ = socket.set_tcp_keepalive(&keepalive);
        }
        info!("   ✓ TCP keepalive enabled (30s idle, 10s interval)");

        // Set TCP buffer sizes
        socket.set_recv_buffer_size(self.tcp_recv_buffer_size)?;
        socket.set_send_buffer_size(self.tcp_send_buffer_size)?;
        info!("   ✓ TCP buffers configured");

        // Bind to address
        socket.bind(&addr.into())?;

        // Listen with custom backlog
        socket.listen(self.tcp_backlog as i32)?;
        info!(
            "   ✓ Listening with {} connection backlog",
            self.tcp_backlog
        );

        // Convert to non-blocking
        socket.set_nonblocking(true)?;

        // Convert to tokio TcpListener
        let std_listener: std::net::TcpListener = socket.into();
        let listener = TcpListener::from_std(std_listener)?;
        let actual_addr = listener.local_addr()?;

        Ok((listener, actual_addr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;

    #[tokio::test]
    async fn test_server_creation() {
        let app = Router::new().route("/health", get(|| async { "OK" }));
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();

        let server = HighPerformanceServer::new(app, addr);
        assert_eq!(server.addr, addr);
        assert_eq!(server.tcp_recv_buffer_size, 256 * 1024);
    }

    #[test]
    fn test_builder_pattern() {
        let app = Router::new();
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();

        let server = HighPerformanceServer::new(app, addr)
            .with_tcp_buffers(8 * 1024 * 1024, 8 * 1024 * 1024)
            .with_backlog(2048);

        assert_eq!(server.tcp_recv_buffer_size, 8 * 1024 * 1024);
        assert_eq!(server.tcp_send_buffer_size, 8 * 1024 * 1024);
        assert_eq!(server.tcp_backlog, 2048);
    }
}
