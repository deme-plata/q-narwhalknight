pub mod access_log;
pub mod admin;
pub mod config;
pub mod acceptor;
pub mod worker;
pub mod proxy;
pub mod upstream;
pub mod metrics;
pub mod health;
pub mod static_serve;
pub mod tui;

// Phase 2: io_uring + SIMD
#[cfg(target_os = "linux")]
pub mod io_uring_loop;
pub mod simd_parse;

// Phase 3: HTTP/2 + QUIC
pub mod h2_proxy;
#[cfg(feature = "quic")]
pub mod quic_proxy;

// Phase 4: libp2p awareness
pub mod libp2p_aware;
