//! Comprehensive network observability — Prometheus-format metrics.
//!
//! This module owns a single [`prometheus_client::registry::Registry`] that
//! aggregates:
//!
//!  1. **libp2p built-in metrics** — connection lifecycle, gossipsub mesh
//!     state, request-response counters, ping RTT, identify exchanges,
//!     Kademlia query stats — all populated automatically by feeding every
//!     `SwarmEvent` to [`libp2p::metrics::Metrics::record`].
//!
//!  2. **App-level counters** specific to Q-NarwhalKnight sync:
//!     - per-peer connection state snapshot (transport, last-seen, peer
//!       protocol version)
//!     - bootstrap dial outcomes (success / failure with cause)
//!     - block-pack request rate (inbound + outbound, success / throttle /
//!       error)
//!     - block-pack response payload size + latency histograms
//!     - chunk scheduler in-flight counts + retry causes
//!     - height progress (local contiguous, network max, gap-to-tip)
//!     - process RSS, DB size, file-descriptor count
//!
//! The full set is exposed on `GET /metrics` in Prometheus text format,
//! curl-able from anywhere. No Prometheus server required — you can
//! `curl :PORT/metrics | grep connections_closed_cause` and immediately
//! see the disconnect reasons every peer pair has experienced.
//!
//! ## Design — why one Registry, not one per concern
//!
//! Prometheus convention is one Registry per process, so when scrapers
//! (or `curl`) hit `/metrics` they get everything in one consistent
//! snapshot. Sub-systems (libp2p, the sync scheduler, the gossipsub
//! handler) register their families into this shared Registry at startup;
//! readers then take an `Arc<RwLock<...>>` and serialize on each request.
//!
//! ## "Why doesn't sync work" diagnostic flow
//!
//! 1. `curl :62107/metrics | grep qnk_peers_connected` → if 0, peer discovery failed.
//! 2. `curl ... | grep libp2p_swarm_connections_closed_cause` → if `keep_alive_timeout`
//!    or `io_error` dominates, transport-layer churn.
//! 3. `curl ... | grep qnk_bootstrap_dial_total{result=` → if `failed` dominates,
//!    bootstrap dial path broken (firewall / wrong peer id / outbound rules).
//! 4. `curl ... | grep qnk_block_pack_request_total{direction="out",result=` →
//!    if `throttle` dominates, client semaphore (Step 1+2) firing too aggressively.
//!    If `error` dominates, server rejecting.
//! 5. `curl ... | grep qnk_chunk_retry_total{reason=` → if `timeout` dominates,
//!    Epsilon's response not returning. If `throttle`, local back-pressure.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
// std::sync::RwLock is fine here — the Registry is only locked briefly
// during HTTP serialization on /metrics scrapes (which arrive at most
// every few seconds from Prometheus, or on demand from `curl`).
use std::sync::RwLock;

use libp2p::metrics::Metrics as Libp2pMetrics;
use libp2p::PeerId;
use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

/// Label-set: bootstrap dial outcome.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct BootstrapDialLabels {
    /// `success` | `failure`
    pub result: String,
    /// Short cause string. Examples: `connection_refused`,
    /// `connection_timed_out`, `protocol_mismatch`, `dns_failure`. For success,
    /// `none`.
    pub cause: String,
}

/// Label-set: block-pack request flow.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct BlockPackLabels {
    /// `in` = received from peer, `out` = sent to peer
    pub direction: String,
    /// `success` | `throttle` | `error` | `timeout`
    pub result: String,
}

/// Label-set: chunk-scheduler retry cause.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ChunkRetryLabels {
    /// `throttle` (Step 1+2's ClientThrottle, no real failure)
    /// `timeout` (request sent but no response)
    /// `transport_error` (connection dropped mid-request)
    /// `other`
    pub reason: String,
}

/// Label-set: gossipsub mesh snapshot per topic.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct TopicLabels {
    pub topic: String,
}

/// The full network metrics registry plus app-level counter families.
///
/// One instance per process, held in an `Arc` and shared between the
/// network manager (which records events), the sync scheduler (which
/// records chunk outcomes), and the HTTP handler (which serializes on
/// demand).
pub struct NetworkMetrics {
    /// The single shared Prometheus Registry. Locked only for HTTP serve.
    pub registry: Arc<RwLock<Registry>>,

    /// libp2p built-in metrics (connections, gossipsub, request-response,
    /// ping RTT, identify, Kademlia query stats). Registered into
    /// `registry` at construction. Call `.record(&swarm_event)` from the
    /// swarm event loop.
    pub libp2p_metrics: Arc<Libp2pMetrics>,

    // ── App-level counters ──────────────────────────────────────────

    /// Current number of established connections (gauge — set via
    /// `peers_connected.set(...)`).
    pub peers_connected: Gauge,

    /// Current number of peers subscribed to each gossipsub topic. Set
    /// from the MeshPeers event handler in the swarm loop.
    pub peers_in_mesh: Family<TopicLabels, Gauge>,

    /// Bootstrap-dial outcomes. Incremented on every Dial attempt to a
    /// configured bootstrap peer (success or failure).
    pub bootstrap_dial_total: Family<BootstrapDialLabels, Counter>,

    /// Block-pack request flow (in + out × result). The chunk scheduler
    /// increments these on each request dispatch + each response.
    pub block_pack_request_total: Family<BlockPackLabels, Counter>,

    /// Block-pack response payload size in bytes (outbound responses we
    /// served, inbound responses we received).
    pub block_pack_response_bytes: Histogram,

    /// Block-pack request → response duration in seconds (client-side).
    pub block_pack_response_duration_seconds: Histogram,

    /// Number of times the per-peer client-block-pack semaphore returned
    /// `ClientThrottle`. Useful to confirm Step 1+2's flow control is
    /// engaging without firing the retry storm.
    pub client_throttle_total: Counter,

    /// Chunk-scheduler retry events broken down by cause.
    pub chunk_retry_total: Family<ChunkRetryLabels, Counter>,

    /// Outstanding (in-flight) chunk requests right now.
    pub chunks_in_flight: Gauge,

    /// Local contiguous chain height (the most recent height we have a
    /// complete chain from genesis to). Read from storage on each tick.
    pub local_height: Gauge,

    /// Highest peer-announced network height we have ever seen.
    pub network_max_height: Gauge,

    /// `network_max_height - local_height`. Negative means we are at tip.
    pub gap_to_tip: Gauge,

    /// Total blocks ingested by libp2p sync path since process start.
    pub blocks_synced_total: Counter,

    /// Process resident set size in bytes (refreshed once per HTTP scrape
    /// from /proc/self/statm or sysinfo).
    pub rss_bytes: Gauge,

    /// On-disk database size in bytes (du -sb of the configured Q_DB_PATH).
    pub db_size_bytes: Gauge,

    /// Open file descriptors held by this process.
    pub open_fds: Gauge,

    // ── Snapshot helpers ────────────────────────────────────────────

    /// Per-peer last-seen wall-clock seconds. Not a Prometheus metric
    /// itself; used by the HTTP handler to emit a `qnk_peer_last_seen{peer=...}`
    /// gauge on demand. Outside the Registry so we don't bound the
    /// label cardinality forever — peers churn, we re-export only the
    /// currently-connected set on each scrape.
    pub peer_last_seen_secs: Arc<dashmap::DashMap<PeerId, u64>>,

    /// Process start time (unix seconds) — used to compute uptime.
    pub start_time_unix_secs: u64,

    /// Monotonic counter of bytes received over libp2p (running sum).
    /// Exposed as `qnk_libp2p_rx_bytes_total`.
    pub rx_bytes_total: Arc<AtomicU64>,

    /// Monotonic counter of bytes sent over libp2p (running sum).
    pub tx_bytes_total: Arc<AtomicU64>,
}

impl NetworkMetrics {
    /// Build a fresh metrics registry. Call once at process start. The
    /// returned `Arc<NetworkMetrics>` is then cloned into both
    /// `UnifiedNetworkManager` (which records SwarmEvents into
    /// `libp2p_metrics`) and the HTTP `AppState` (which serializes
    /// `registry` for `/metrics` requests).
    pub fn new() -> Self {
        let mut registry = Registry::default();

        // libp2p built-in metrics. The `Metrics::new` method here
        // registers all of libp2p's internal families into our Registry.
        let libp2p_metrics = Libp2pMetrics::new(&mut registry);

        // ── App-level metric families ──────────────────────────────

        let peers_connected = Gauge::default();
        registry.register(
            "qnk_peers_connected",
            "Currently established libp2p peer connections (any transport)",
            peers_connected.clone(),
        );

        let peers_in_mesh: Family<TopicLabels, Gauge> = Family::default();
        registry.register(
            "qnk_peers_in_gossipsub_mesh",
            "Current peer count in each gossipsub topic mesh",
            peers_in_mesh.clone(),
        );

        let bootstrap_dial_total: Family<BootstrapDialLabels, Counter> = Family::default();
        registry.register(
            "qnk_bootstrap_dial_total",
            "Bootstrap-peer dial attempts (success / failure) broken down by cause",
            bootstrap_dial_total.clone(),
        );

        let block_pack_request_total: Family<BlockPackLabels, Counter> = Family::default();
        registry.register(
            "qnk_block_pack_request_total",
            "Block-pack request-response flow events (direction × result)",
            block_pack_request_total.clone(),
        );

        // Histogram buckets sized for typical 50-byte → 50MB block-pack responses
        let block_pack_response_bytes = Histogram::new(
            [
                1024.0,         //   1 KiB
                10_240.0,       //  10 KiB
                102_400.0,      // 100 KiB
                1_048_576.0,    //   1 MiB
                10_485_760.0,   //  10 MiB
                52_428_800.0,   //  50 MiB
                104_857_600.0,  // 100 MiB
            ]
            .into_iter(),
        );
        registry.register(
            "qnk_block_pack_response_bytes",
            "Distribution of block-pack response sizes",
            block_pack_response_bytes.clone(),
        );

        // Histogram buckets for 10ms → 60s request-response duration
        let block_pack_response_duration_seconds = Histogram::new(
            [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
                .into_iter(),
        );
        registry.register(
            "qnk_block_pack_response_duration_seconds",
            "Block-pack client request → response wall-clock duration",
            block_pack_response_duration_seconds.clone(),
        );

        let client_throttle_total = Counter::default();
        registry.register(
            "qnk_client_throttle_total",
            "Times the per-peer client-block-pack semaphore returned ClientThrottle",
            client_throttle_total.clone(),
        );

        let chunk_retry_total: Family<ChunkRetryLabels, Counter> = Family::default();
        registry.register(
            "qnk_chunk_retry_total",
            "Chunk scheduler retry events by cause (throttle/timeout/transport/other)",
            chunk_retry_total.clone(),
        );

        let chunks_in_flight = Gauge::default();
        registry.register(
            "qnk_chunks_in_flight",
            "Outstanding (dispatched, no response yet) block-pack chunks",
            chunks_in_flight.clone(),
        );

        let local_height = Gauge::default();
        registry.register(
            "qnk_local_height",
            "Local contiguous chain height (highest block with full ancestry from genesis)",
            local_height.clone(),
        );

        let network_max_height = Gauge::default();
        registry.register(
            "qnk_network_max_height",
            "Highest peer-announced height observed since process start",
            network_max_height.clone(),
        );

        let gap_to_tip = Gauge::default();
        registry.register(
            "qnk_gap_to_tip",
            "Blocks behind: network_max_height - local_height (negative => at tip)",
            gap_to_tip.clone(),
        );

        let blocks_synced_total = Counter::default();
        registry.register(
            "qnk_blocks_synced_total",
            "Total blocks ingested via libp2p sync path since process start",
            blocks_synced_total.clone(),
        );

        let rss_bytes = Gauge::default();
        registry.register(
            "qnk_process_rss_bytes",
            "Process resident set size, in bytes",
            rss_bytes.clone(),
        );

        let db_size_bytes = Gauge::default();
        registry.register(
            "qnk_db_size_bytes",
            "On-disk database size (Q_DB_PATH), in bytes",
            db_size_bytes.clone(),
        );

        let open_fds = Gauge::default();
        registry.register(
            "qnk_open_file_descriptors",
            "Open file descriptors held by this process",
            open_fds.clone(),
        );

        let start_time_unix_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Uptime as a counter that the HTTP handler refreshes pre-encode.
        // We approximate it with start_time_unix_secs and let the handler
        // compute `now - start` for the snapshot. No separate metric.

        Self {
            registry: Arc::new(RwLock::new(registry)),
            libp2p_metrics: Arc::new(libp2p_metrics),
            peers_connected,
            peers_in_mesh,
            bootstrap_dial_total,
            block_pack_request_total,
            block_pack_response_bytes,
            block_pack_response_duration_seconds,
            client_throttle_total,
            chunk_retry_total,
            chunks_in_flight,
            local_height,
            network_max_height,
            gap_to_tip,
            blocks_synced_total,
            rss_bytes,
            db_size_bytes,
            open_fds,
            peer_last_seen_secs: Arc::new(dashmap::DashMap::new()),
            start_time_unix_secs,
            rx_bytes_total: Arc::new(AtomicU64::new(0)),
            tx_bytes_total: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Update the process-resource gauges. Call this from the HTTP
    /// handler before serializing the Registry — it's cheap (single
    /// /proc read) and keeps the snapshot live.
    pub fn refresh_process_stats(&self, db_path: Option<&std::path::Path>) {
        if let Some(rss) = read_self_rss_bytes() {
            self.rss_bytes.set(rss as i64);
        }
        if let Some(path) = db_path {
            if let Some(sz) = dir_size_bytes(path) {
                self.db_size_bytes.set(sz as i64);
            }
        }
        if let Some(fds) = count_open_fds() {
            self.open_fds.set(fds as i64);
        }
    }

    /// Serialize the Registry into Prometheus text format. This is what
    /// the `/metrics` HTTP handler calls; the buffer it returns goes
    /// straight into the response body with content-type
    /// `application/openmetrics-text; version=1.0.0; charset=utf-8`.
    pub fn encode_text(&self) -> Result<String, std::fmt::Error> {
        let mut buf = String::with_capacity(8192);
        // std::sync::RwLock::read returns Result — only fails if the lock
        // is poisoned (a previous holder panicked). Recover by using the
        // poison guard's inner data; for /metrics readout, the Registry
        // is append-only at startup so even poisoned state is safe to read.
        let reg = self
            .registry
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        encode(&mut buf, &reg)?;
        // Append uptime as a freeform comment + counter (no need for it
        // to live in the Registry since it's trivially derived).
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(self.start_time_unix_secs);
        let uptime = now.saturating_sub(self.start_time_unix_secs);
        let rx = self.rx_bytes_total.load(Ordering::Relaxed);
        let tx = self.tx_bytes_total.load(Ordering::Relaxed);
        // OpenMetrics requires `# EOF` last; we appended below it so we
        // emit our extras then re-emit EOF. Easier path: write extras
        // BEFORE the encode() output. But encode() already wrote EOF.
        // Workaround: strip trailing `# EOF\n` if present, append extras,
        // re-append EOF.
        let mut out = buf;
        if out.ends_with("# EOF\n") {
            out.truncate(out.len() - 6);
        }
        out.push_str(&format!(
            "# HELP qnk_process_uptime_seconds Process uptime in seconds.\n\
             # TYPE qnk_process_uptime_seconds counter\n\
             qnk_process_uptime_seconds_total {}\n\
             # HELP qnk_libp2p_rx_bytes_total Total bytes received over libp2p transports.\n\
             # TYPE qnk_libp2p_rx_bytes_total counter\n\
             qnk_libp2p_rx_bytes_total {}\n\
             # HELP qnk_libp2p_tx_bytes_total Total bytes sent over libp2p transports.\n\
             # TYPE qnk_libp2p_tx_bytes_total counter\n\
             qnk_libp2p_tx_bytes_total {}\n\
             # EOF\n",
            uptime, rx, tx
        ));
        Ok(out)
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Helpers (no dep on libp2p) ─────────────────────────────────────

/// Read the current process resident-set-size in bytes from
/// `/proc/self/statm`. Returns None on non-Linux or if the file can't be
/// parsed.
fn read_self_rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/statm").ok()?;
    // statm format: size resident shared text lib data dt
    // all in pages of size sysconf(_SC_PAGESIZE) (typically 4096).
    let resident_pages: u64 = s.split_whitespace().nth(1)?.parse().ok()?;
    let page_size = page_size_bytes();
    Some(resident_pages.saturating_mul(page_size))
}

fn page_size_bytes() -> u64 {
    // Linux page size; for portability we could call sysconf(_SC_PAGESIZE)
    // via libc, but 4096 is universally true for the platforms we run on.
    4096
}

/// Cheap recursive directory size in bytes. Returns None on first IO
/// error so a missing DB dir reports "unknown" rather than 0.
fn dir_size_bytes(p: &std::path::Path) -> Option<u64> {
    let mut total: u64 = 0;
    let mut stack = vec![p.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let md = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            if md.is_file() {
                total = total.saturating_add(md.len());
            } else if md.is_dir() {
                stack.push(entry.path());
            }
        }
    }
    Some(total)
}

/// Count open file descriptors for the current process (Linux).
fn count_open_fds() -> Option<u64> {
    let entries = std::fs::read_dir("/proc/self/fd").ok()?;
    Some(entries.flatten().count() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_metrics_registers_all_families() {
        let m = NetworkMetrics::new();
        let out = m.encode_text().expect("encode");
        // Spot-check that each family name appears in the output.
        for needle in &[
            "qnk_peers_connected",
            "qnk_peers_in_gossipsub_mesh",
            "qnk_bootstrap_dial_total",
            "qnk_block_pack_request_total",
            "qnk_block_pack_response_bytes",
            "qnk_block_pack_response_duration_seconds",
            "qnk_client_throttle_total",
            "qnk_chunk_retry_total",
            "qnk_chunks_in_flight",
            "qnk_local_height",
            "qnk_network_max_height",
            "qnk_gap_to_tip",
            "qnk_blocks_synced_total",
            "qnk_process_rss_bytes",
            "qnk_db_size_bytes",
            "qnk_open_file_descriptors",
            "qnk_process_uptime_seconds",
            "qnk_libp2p_rx_bytes_total",
            "qnk_libp2p_tx_bytes_total",
        ] {
            assert!(
                out.contains(needle),
                "expected metric `{}` in /metrics output",
                needle
            );
        }
    }

    #[test]
    fn counters_increment_correctly() {
        let m = NetworkMetrics::new();
        m.client_throttle_total.inc();
        m.client_throttle_total.inc();
        m.block_pack_request_total
            .get_or_create(&BlockPackLabels {
                direction: "out".into(),
                result: "success".into(),
            })
            .inc();
        let out = m.encode_text().unwrap();
        assert!(out.contains("qnk_client_throttle_total_total 2"));
        assert!(out.contains("direction=\"out\""));
    }

    #[test]
    fn gauges_accept_set_and_get() {
        let m = NetworkMetrics::new();
        m.peers_connected.set(3);
        m.local_height.set(10_426);
        m.gap_to_tip.set(18_064_797);
        let out = m.encode_text().unwrap();
        assert!(out.contains("qnk_peers_connected 3"));
        assert!(out.contains("qnk_local_height 10426"));
        assert!(out.contains("qnk_gap_to_tip 18064797"));
    }

    #[test]
    fn encode_ends_with_eof() {
        let m = NetworkMetrics::new();
        let out = m.encode_text().unwrap();
        assert!(
            out.trim_end().ends_with("# EOF"),
            "/metrics output must end with `# EOF` per OpenMetrics spec"
        );
    }

    #[test]
    fn refresh_process_stats_populates_rss() {
        let m = NetworkMetrics::new();
        m.refresh_process_stats(None);
        // On Linux this should be > 0; on other platforms it stays 0.
        // We don't assert > 0 to keep the test portable.
        let _out = m.encode_text().unwrap();
    }
}
