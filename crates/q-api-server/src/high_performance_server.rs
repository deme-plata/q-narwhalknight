/// High-Performance HTTP Server for 1M+ TPS
///
/// This module provides optimized TCP socket configuration for Axum
/// designed for extreme transaction throughput:
///
/// - Optimized TCP socket configuration (NODELAY, large buffers, REUSEPORT)
/// - Connection pooling and backlog management
/// - Zero-copy request handling via Axum's default behavior
/// - v1.0.53: Automatic port detection if requested port is in use
///
/// Target Performance: 1,000,000+ TPS with binary protocol
///
/// Note: HTTP/2 support is enabled automatically by Axum when the client
/// requests it via ALPN negotiation.
use axum::Router;
use std::net::SocketAddr;
use tokio::net::TcpListener;
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
}

impl HighPerformanceServer {
    /// Create a new high-performance server with optimal defaults
    pub fn new(app: Router, addr: SocketAddr) -> Self {
        Self {
            app,
            addr,
            tcp_recv_buffer_size: 4 * 1024 * 1024, // 4MB receive buffer
            tcp_send_buffer_size: 4 * 1024 * 1024, // 4MB send buffer
            tcp_backlog: 1024,                     // 1024 pending connections (up from default 128)
            auto_port_detection: true,             // v1.0.53: Enable by default
            max_port_attempts: 10,                 // Try up to 10 ports
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
            "   TCP buffer size: {} MB recv, {} MB send",
            self.tcp_recv_buffer_size / (1024 * 1024),
            self.tcp_send_buffer_size / (1024 * 1024)
        );
        info!("   TCP backlog: {} pending connections", self.tcp_backlog);
        info!("   Target throughput: 1,000,000+ TPS");
        if self.auto_port_detection {
            info!("   Auto port detection: ENABLED (will try up to {} ports)", self.max_port_attempts);
        }

        // v1.0.53: Try to bind with automatic port detection
        let (listener, actual_addr) = self.try_bind_with_auto_detection().await?;

        info!("✅ TCP listener configured and ready");
        info!("🌟 High-Performance Server READY - accepting connections...");
        info!("   Listening on: http://{}", actual_addr);
        info!("   HTTP/2 will be negotiated automatically per connection");

        // Create shutdown signal handler for both SIGTERM (systemd) and CTRL+C
        let shutdown_signal = async {
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

            tokio::select! {
                _ = ctrl_c => {
                    info!("🛑 Received CTRL+C signal - initiating graceful shutdown");
                },
                _ = terminate => {
                    info!("🛑 Received SIGTERM signal - initiating graceful shutdown");
                },
            }
        };

        // Use Axum's optimized serve function with graceful shutdown
        axum::serve(
            listener,
            self.app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown_signal)
        .await?;

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

        // SO_REUSEPORT - Kernel load balancing (Linux only)
        #[cfg(target_os = "linux")]
        {
            if let Err(e) = socket.set_reuse_port(true) {
                warn!("   ⚠️  SO_REUSEPORT not supported: {}", e);
            } else {
                info!("   ✓ SO_REUSEPORT enabled (kernel load balancing)");
            }
        }

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
        assert_eq!(server.tcp_recv_buffer_size, 4 * 1024 * 1024);
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
