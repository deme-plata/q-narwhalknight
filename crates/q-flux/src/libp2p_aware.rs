//! libp2p-aware proxy layer (Phase 4).
//!
//! Detects libp2p WebSocket connections at the proxy layer and applies
//! per-peer policies before forwarding to the backend. This offloads
//! work from the backend node.
//!
//! Detection: libp2p peers connect via WebSocket and immediately send
//! a multistream-select header (`/multistream/1.0.0\n`). The proxy
//! detects this pattern in the first bytes after WebSocket upgrade.
//!
//! Capabilities:
//! 1. Per-peer bandwidth tiers (bootstrap nodes get more bandwidth)
//! 2. Circuit breaking for slow/abusive peers
//! 3. Per-peer connection limiting (max 4 per peer ID)
//! 4. Gossipsub message dedup at proxy layer (reduces backend load)
//! 5. Peer metrics (connections, bandwidth, latency per peer)
//!
//! # Integration
//!
//! In the WebSocket upgrade handler (`proxy.rs`), after the WS handshake:
//! ```ignore
//! // Peek at first bytes on the WS data stream
//! if libp2p_aware::is_libp2p_handshake(&first_bytes) {
//!     let info = LibP2pDetector::detect(&first_bytes);
//!     if let Some(ref info) = info {
//!         if !should_allow_peer(&tracker, &info.peer_id) {
//!             // Reject: too many connections or circuit breaker open
//!             return;
//!         }
//!     }
//!     // Continue with WS splice, applying bandwidth limits
//! }
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Multistream-select detection
// ---------------------------------------------------------------------------

/// The multistream-select 1.0.0 protocol identifier.
/// libp2p sends this as the first message after a transport connection is
/// established (after TLS/Noise handshake, after WebSocket upgrade).
///
/// Wire format: varint-length-prefixed string followed by `\n`.
/// The varint for 20 bytes is `0x14`, so the full wire bytes are:
/// `\x14/multistream/1.0.0\n`
const MULTISTREAM_PREFIX: &[u8] = b"/multistream/1.0.0\n";

/// Minimum bytes we need to peek at to detect a multistream-select handshake.
/// The varint prefix byte + the protocol string.
const MULTISTREAM_DETECT_LEN: usize = 1 + MULTISTREAM_PREFIX.len(); // 21 bytes

/// Check if a byte buffer starts with the multistream-select 1.0.0 handshake.
///
/// The multistream-select wire format is:
///   `<varint length><protocol-id>\n`
///
/// For `/multistream/1.0.0\n` (20 bytes), the varint is `0x14` (single byte).
/// We also check for the raw protocol string without the varint prefix,
/// because some WebSocket-framed transports skip the length prefix.
pub fn is_libp2p_handshake(buf: &[u8]) -> bool {
    if buf.len() < MULTISTREAM_PREFIX.len() {
        return false;
    }

    // Check with varint prefix: 0x14 = 20 (length of "/multistream/1.0.0\n")
    if buf.len() >= MULTISTREAM_DETECT_LEN
        && buf[0] == 0x14
        && &buf[1..MULTISTREAM_DETECT_LEN] == MULTISTREAM_PREFIX
    {
        return true;
    }

    // Check without varint prefix (WebSocket-framed transport)
    if buf.starts_with(MULTISTREAM_PREFIX) {
        return true;
    }

    // Check for the pattern anywhere in the first 128 bytes (may be after
    // WebSocket frame headers or noise handshake remnants)
    let search_len = buf.len().min(128);
    for window in buf[..search_len].windows(MULTISTREAM_PREFIX.len()) {
        if window == MULTISTREAM_PREFIX {
            return true;
        }
    }

    false
}

/// Information extracted from a detected libp2p connection.
#[derive(Debug, Clone)]
pub struct LibP2pInfo {
    /// The peer ID as a string (e.g., "12D3KooW...").
    /// Empty if we could not extract it from the handshake.
    pub peer_id: String,

    /// Byte offset where the multistream-select header was found.
    pub header_offset: usize,
}

/// Detector for libp2p traffic patterns.
pub struct LibP2pDetector;

impl LibP2pDetector {
    /// Attempt to detect a libp2p connection and extract peer information
    /// from the initial bytes of a WebSocket data stream.
    ///
    /// Returns `None` if the bytes do not match a libp2p handshake pattern.
    pub fn detect(first_bytes: &[u8]) -> Option<LibP2pInfo> {
        if !is_libp2p_handshake(first_bytes) {
            return None;
        }

        // Find the offset
        let header_offset = if (first_bytes.len() >= MULTISTREAM_DETECT_LEN
            && first_bytes[0] == 0x14
            && &first_bytes[1..MULTISTREAM_DETECT_LEN] == MULTISTREAM_PREFIX)
            || first_bytes.starts_with(MULTISTREAM_PREFIX)
        {
            0
        } else {
            let search_len = first_bytes.len().min(128);
            let mut offset = 0;
            for i in 0..search_len.saturating_sub(MULTISTREAM_PREFIX.len()) {
                if &first_bytes[i..i + MULTISTREAM_PREFIX.len()] == MULTISTREAM_PREFIX {
                    offset = i;
                    break;
                }
            }
            offset
        };

        // Try to extract peer ID from Noise handshake payload.
        // The peer ID typically appears after the multistream negotiation
        // and Noise IX/XX handshake. In practice, extracting it requires
        // decoding the Noise handshake which is not feasible at the proxy
        // layer without the session keys. We leave peer_id empty here;
        // the actual peer ID is learned from the backend after the full
        // handshake completes.
        let peer_id = extract_peer_id(first_bytes).unwrap_or_default();

        Some(LibP2pInfo {
            peer_id,
            header_offset,
        })
    }
}

/// Try to extract a peer ID from the handshake bytes.
///
/// libp2p peer IDs are base58-encoded multihashes. In the Noise handshake
/// payload, the peer's public key is sent in a protobuf envelope. At the
/// proxy layer we cannot decode the Noise handshake (no session keys), so
/// this function looks for the well-known peer ID prefix pattern:
/// - Ed25519 peer IDs start with `\x00\x24\x08\x01\x12\x20` (protobuf key type + length)
/// - The resulting peer ID starts with "12D3KooW" when base58-encoded
///
/// Returns None if no peer ID pattern is found. This is expected for most
/// connections since the peer ID is encrypted within the Noise handshake.
pub fn extract_peer_id(buf: &[u8]) -> Option<String> {
    // Look for Ed25519 public key protobuf envelope
    // Field 1 (key_type): varint = 1 (Ed25519)
    // Field 2 (data): 32 bytes
    // Protobuf: 08 01 12 20 <32 bytes>
    let ed25519_prefix: &[u8] = &[0x08, 0x01, 0x12, 0x20];
    let search_len = buf.len().min(512);

    for i in 0..search_len.saturating_sub(ed25519_prefix.len() + 32) {
        if &buf[i..i + ed25519_prefix.len()] == ed25519_prefix {
            // Found a potential Ed25519 public key
            let key_bytes = &buf[i..i + ed25519_prefix.len() + 32];

            // Hash to produce a multihash-style identifier.
            // In a real implementation we would compute the proper
            // identity/sha256 multihash and base58-encode it. For the
            // proxy layer, a hex fingerprint is sufficient for tracking.
            let mut hasher = DefaultHasher::new();
            key_bytes.hash(&mut hasher);
            let fingerprint = hasher.finish();
            return Some(format!("peer-{:016x}", fingerprint));
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Peer tiers and policies
// ---------------------------------------------------------------------------

/// Bandwidth tier for a peer. Determines resource allocation priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PeerTier {
    /// 10Gbit supernode (e.g., Epsilon). Highest priority, most bandwidth.
    Supernode,
    /// Bootstrap/seed node. High priority.
    Bootstrap,
    /// Active validator. Medium-high priority.
    Validator,
    /// Mining client. Standard priority.
    Miner,
    /// Unknown/new peer. Lowest priority until classified.
    Unknown,
}

impl PeerTier {
    /// Default bandwidth limit in Mbps for this tier.
    pub fn default_bandwidth_mbps(&self) -> u64 {
        match self {
            PeerTier::Supernode => 1000,  // 1 Gbps
            PeerTier::Bootstrap => 500,   // 500 Mbps
            PeerTier::Validator => 200,   // 200 Mbps
            PeerTier::Miner => 50,        // 50 Mbps
            PeerTier::Unknown => 10,      // 10 Mbps
        }
    }

    /// Default maximum connections for this tier.
    pub fn default_max_connections(&self) -> u32 {
        match self {
            PeerTier::Supernode => 16,
            PeerTier::Bootstrap => 8,
            PeerTier::Validator => 4,
            PeerTier::Miner => 2,
            PeerTier::Unknown => 1,
        }
    }

    /// Priority value (higher = more priority in resource allocation).
    pub fn priority(&self) -> u8 {
        match self {
            PeerTier::Supernode => 5,
            PeerTier::Bootstrap => 4,
            PeerTier::Validator => 3,
            PeerTier::Miner => 2,
            PeerTier::Unknown => 1,
        }
    }
}

impl std::fmt::Display for PeerTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PeerTier::Supernode => write!(f, "supernode"),
            PeerTier::Bootstrap => write!(f, "bootstrap"),
            PeerTier::Validator => write!(f, "validator"),
            PeerTier::Miner => write!(f, "miner"),
            PeerTier::Unknown => write!(f, "unknown"),
        }
    }
}

/// Policy applied to a specific peer or tier.
#[derive(Debug, Clone)]
pub struct PeerPolicy {
    /// The peer's bandwidth tier.
    pub tier: PeerTier,
    /// Maximum bandwidth in Mbps.
    pub max_bandwidth_mbps: u64,
    /// Maximum concurrent connections from this peer.
    pub max_connections: u32,
    /// Priority value for resource allocation.
    pub priority: u8,
}

impl PeerPolicy {
    /// Create a policy from a tier using default values.
    pub fn from_tier(tier: PeerTier) -> Self {
        Self {
            tier,
            max_bandwidth_mbps: tier.default_bandwidth_mbps(),
            max_connections: tier.default_max_connections(),
            priority: tier.priority(),
        }
    }
}

// ---------------------------------------------------------------------------
// Circuit breaker
// ---------------------------------------------------------------------------

/// Circuit breaker state machine.
///
/// Protects the proxy from repeatedly forwarding traffic to/from a misbehaving
/// peer. After `threshold` consecutive failures, the breaker opens (blocks all
/// traffic from the peer). After `reset_after_secs`, it transitions to
/// half-open and allows one probe request. If the probe succeeds, the breaker
/// closes; if it fails, it opens again.
///
/// State transitions:
/// ```text
/// Closed ---[threshold failures]--> Open ---[timeout]--> HalfOpen
///   ^                                                       |
///   |                                                       |
///   +---[success]---<------<------<------<------<------<-----+
///                                                           |
///   Open <---[failure]---<------<------<------<------<-------+
/// ```
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    /// Number of consecutive failures.
    pub failures: u32,
    /// Current state of the breaker.
    pub state: BreakerState,
    /// Number of failures before the breaker opens.
    pub threshold: u32,
    /// Seconds after which an open breaker transitions to half-open.
    pub reset_after_secs: u64,
}

/// The three states of a circuit breaker.
#[derive(Debug, Clone, PartialEq)]
pub enum BreakerState {
    /// Normal operation. Traffic flows freely.
    Closed,
    /// Breaker is open. All traffic is blocked. Contains the timestamp when
    /// the breaker opened.
    Open(Instant),
    /// Breaker is half-open. One probe request is allowed.
    HalfOpen,
}

impl CircuitBreakerState {
    /// Create a new circuit breaker in the closed state.
    pub fn new(threshold: u32, reset_after_secs: u64) -> Self {
        Self {
            failures: 0,
            state: BreakerState::Closed,
            threshold,
            reset_after_secs,
        }
    }

    /// Record a failure. If the failure count reaches the threshold,
    /// the breaker transitions to open.
    pub fn record_failure(&mut self) {
        self.failures += 1;
        if self.failures >= self.threshold {
            self.state = BreakerState::Open(Instant::now());
            debug!(
                failures = self.failures,
                threshold = self.threshold,
                "Circuit breaker opened"
            );
        }
    }

    /// Record a success. Resets the failure count and closes the breaker.
    pub fn record_success(&mut self) {
        self.failures = 0;
        self.state = BreakerState::Closed;
    }

    /// Check if a request should be allowed through.
    ///
    /// - Closed: always allow.
    /// - Open: block unless the reset timeout has elapsed, in which case
    ///   transition to half-open and allow one probe.
    /// - HalfOpen: allow (the probe request).
    pub fn should_allow(&mut self) -> bool {
        match &self.state {
            BreakerState::Closed => true,
            BreakerState::Open(opened_at) => {
                if opened_at.elapsed() >= Duration::from_secs(self.reset_after_secs) {
                    self.state = BreakerState::HalfOpen;
                    debug!("Circuit breaker transitioned to half-open");
                    true // Allow the probe
                } else {
                    false // Still blocked
                }
            }
            BreakerState::HalfOpen => {
                // Allow the probe through. The caller must call
                // record_success() or record_failure() based on the result.
                true
            }
        }
    }

    /// Whether the breaker is currently in the open state.
    pub fn is_open(&self) -> bool {
        matches!(self.state, BreakerState::Open(_))
    }
}

// ---------------------------------------------------------------------------
// Per-peer state tracking
// ---------------------------------------------------------------------------

/// Per-peer connection state tracked by the proxy.
#[derive(Debug)]
pub struct PeerState {
    /// Number of currently active connections from this peer.
    pub active_connections: AtomicU32,
    /// Total bytes received from this peer.
    pub bytes_in: AtomicU64,
    /// Total bytes sent to this peer.
    pub bytes_out: AtomicU64,
    /// Last time we saw traffic from this peer.
    pub last_seen: RwLock<Instant>,
    /// The peer's bandwidth tier.
    pub tier: PeerTier,
    /// Circuit breaker for this peer.
    pub circuit_breaker: RwLock<CircuitBreakerState>,
}

impl PeerState {
    /// Create a new peer state with the given tier.
    pub fn new(tier: PeerTier) -> Self {
        Self {
            active_connections: AtomicU32::new(0),
            bytes_in: AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
            last_seen: RwLock::new(Instant::now()),
            tier,
            circuit_breaker: RwLock::new(CircuitBreakerState::new(
                10,  // 10 failures before opening
                60,  // Reset after 60 seconds
            )),
        }
    }

    /// Record incoming bytes.
    pub fn record_rx(&self, n: u64) {
        self.bytes_in.fetch_add(n, Ordering::Relaxed);
        *self.last_seen.write() = Instant::now();
    }

    /// Record outgoing bytes.
    pub fn record_tx(&self, n: u64) {
        self.bytes_out.fetch_add(n, Ordering::Relaxed);
    }

    /// Increment the active connection count.
    pub fn conn_opened(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
        *self.last_seen.write() = Instant::now();
    }

    /// Decrement the active connection count.
    pub fn conn_closed(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Tracker for all known peers. Thread-safe via DashMap.
pub struct PeerTracker {
    peers: DashMap<String, PeerState>,
    /// Known bootstrap peer IDs for automatic tier classification.
    bootstrap_peers: Vec<String>,
    /// Known supernode peer IDs.
    supernode_peers: Vec<String>,
}

impl PeerTracker {
    /// Create a new peer tracker.
    ///
    /// `bootstrap_peers` and `supernode_peers` are peer ID strings (or
    /// prefixes) used to automatically classify connecting peers.
    pub fn new(bootstrap_peers: Vec<String>, supernode_peers: Vec<String>) -> Self {
        Self {
            peers: DashMap::with_capacity(256),
            bootstrap_peers,
            supernode_peers,
        }
    }

    /// Get or create a peer entry. Automatically classifies the peer tier
    /// based on known peer ID lists.
    ///
    /// Uses DashMap `entry()` API to avoid TOCTOU races — a separate
    /// `contains_key()` + `get().unwrap()` pattern can panic if another
    /// thread removes the key between the two calls.
    pub fn get_or_create(&self, peer_id: &str) -> dashmap::mapref::one::Ref<'_, String, PeerState> {
        // Insert if missing (entry API is atomic — no TOCTOU race).
        let tier = self.classify_peer(peer_id);
        self.peers.entry(peer_id.to_string()).or_insert_with(|| {
            info!(peer = peer_id, tier = %tier, "New peer detected");
            PeerState::new(tier)
        });
        // Safe: we just ensured the key exists via entry(), and only
        // cleanup_stale removes entries (which runs on an interval, not
        // concurrently with this call in the hot path).
        self.peers.get(peer_id).expect("peer just inserted via entry()")
    }

    /// Classify a peer based on known peer ID lists.
    fn classify_peer(&self, peer_id: &str) -> PeerTier {
        if self.supernode_peers.iter().any(|id| peer_id.contains(id.as_str())) {
            PeerTier::Supernode
        } else if self.bootstrap_peers.iter().any(|id| peer_id.contains(id.as_str())) {
            PeerTier::Bootstrap
        } else {
            PeerTier::Unknown
        }
    }

    /// Total number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Return a reference to the underlying peer map.
    /// Used by `BandwidthLimiter::cleanup()` to evict buckets for peers
    /// that are no longer tracked.
    pub fn peer_map(&self) -> &DashMap<String, PeerState> {
        &self.peers
    }

    /// Remove stale peers that haven't been seen for the given duration.
    pub fn cleanup_stale(&self, max_idle: Duration) {
        self.peers.retain(|_id, state| {
            let idle = state.last_seen.read().elapsed();
            let active = state.active_connections.load(Ordering::Relaxed);
            // Keep if still has active connections or was seen recently
            active > 0 || idle < max_idle
        });
    }

    /// Scan all peers and auto-promote based on observed traffic patterns.
    ///
    /// Heuristics (only upgrade, never downgrade from pre-seeded tiers):
    /// - Active > `min_duration` with > `min_bytes` transferred → promote to Miner
    /// - Peer ID prefix matches known validator patterns → promote to Validator
    /// - Never auto-promote to Bootstrap or Supernode (pre-seeded only)
    pub fn auto_classify(&self, min_duration: Duration, min_bytes: u64) {
        // Phase 1: Collect promotion candidates (read-only scan).
        // Must release all DashMap refs before Phase 2 to avoid deadlock.
        let mut promote_to_miner: Vec<String> = Vec::new();

        for entry in self.peers.iter() {
            let state = entry.value();

            // Skip peers already at Miner or higher — never downgrade
            match state.tier {
                PeerTier::Supernode | PeerTier::Bootstrap | PeerTier::Validator | PeerTier::Miner => continue,
                PeerTier::Unknown => {}
            }

            let idle = state.last_seen.read().elapsed();
            let total_bytes = state.bytes_in.load(Ordering::Relaxed)
                + state.bytes_out.load(Ordering::Relaxed);

            // Heuristic: long-lived connection + significant traffic = miner
            if idle < min_duration && total_bytes >= min_bytes {
                promote_to_miner.push(entry.key().clone());
            }
        }
        // All DashMap refs dropped here (for loop ended).

        // Phase 2: Apply promotions (write operations, no concurrent reads).
        for peer_id in &promote_to_miner {
            info!(
                peer = peer_id.as_str(),
                "Auto-promoting peer Unknown → Miner (traffic heuristic)"
            );
            self.set_tier(peer_id, PeerTier::Miner);
        }
    }

    /// Manually set a peer's tier (e.g., after validator registration).
    pub fn set_tier(&self, peer_id: &str, tier: PeerTier) {
        if let Some(entry) = self.peers.get(peer_id) {
            // PeerState.tier is not atomic, but since we only upgrade
            // tiers (never downgrade), reading a stale value is safe.
            // For a proper implementation, tier would be AtomicU8.
            drop(entry);
        }
        // Re-insert with new tier (simpler than interior mutability for tier)
        let old_state = self.peers.remove(peer_id);
        let new_state = PeerState::new(tier);
        if let Some((_, old)) = old_state {
            new_state.bytes_in.store(old.bytes_in.load(Ordering::Relaxed), Ordering::Relaxed);
            new_state.bytes_out.store(old.bytes_out.load(Ordering::Relaxed), Ordering::Relaxed);
            new_state
                .active_connections
                .store(old.active_connections.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        self.peers.insert(peer_id.to_string(), new_state);
    }
}

/// Check if a peer should be allowed to connect based on connection limits
/// and circuit breaker state.
pub fn should_allow_peer(tracker: &PeerTracker, peer_id: &str) -> bool {
    let peer = tracker.get_or_create(peer_id);
    let policy = PeerPolicy::from_tier(peer.tier);

    // Check connection limit
    let active = peer.active_connections.load(Ordering::Relaxed);
    if active >= policy.max_connections {
        warn!(
            peer = peer_id,
            active = active,
            max = policy.max_connections,
            tier = %peer.tier,
            "Peer connection limit reached"
        );
        return false;
    }

    // Check circuit breaker
    let mut cb = peer.circuit_breaker.write();
    if !cb.should_allow() {
        warn!(
            peer = peer_id,
            failures = cb.failures,
            "Peer circuit breaker is open"
        );
        return false;
    }

    true
}

// ---------------------------------------------------------------------------
// Bandwidth limiter (token bucket per peer)
// ---------------------------------------------------------------------------

/// Per-peer bandwidth limiter using token buckets.
///
/// Each peer gets a token bucket based on their tier's bandwidth limit.
/// Tokens represent bytes (1 token = 1 byte). The bucket refills at
/// `max_bandwidth_mbps * 125_000` bytes per second (Mbps -> bytes/s).
pub struct BandwidthLimiter {
    /// Token buckets keyed by peer ID.
    buckets: DashMap<String, BandwidthBucket>,
}

/// A single peer's bandwidth token bucket.
struct BandwidthBucket {
    /// Available tokens (bytes).
    tokens: AtomicU64,
    /// Last refill timestamp in microseconds since epoch.
    last_refill_us: AtomicU64,
    /// Refill rate in bytes per second.
    rate_bytes_per_sec: u64,
    /// Maximum burst capacity in bytes.
    capacity: u64,
}

impl BandwidthLimiter {
    /// Create a new bandwidth limiter.
    pub fn new() -> Self {
        Self {
            buckets: DashMap::with_capacity(256),
        }
    }

    /// Ensure a bucket exists for the given peer with the specified bandwidth.
    fn ensure_bucket(&self, peer_id: &str, max_bandwidth_mbps: u64) {
        self.buckets.entry(peer_id.to_string()).or_insert_with(|| {
            let rate = max_bandwidth_mbps * 125_000; // Mbps -> bytes/sec
            let capacity = rate * 2; // 2-second burst
            BandwidthBucket {
                tokens: AtomicU64::new(capacity),
                last_refill_us: AtomicU64::new(0),
                rate_bytes_per_sec: rate,
                capacity,
            }
        });
    }

    /// Try to consume `bytes` tokens from the peer's bucket.
    ///
    /// Returns `true` if the transfer is allowed, `false` if the peer has
    /// exceeded their bandwidth limit.
    pub fn try_consume(&self, peer_id: &str, bytes: usize, max_bandwidth_mbps: u64) -> bool {
        self.ensure_bucket(peer_id, max_bandwidth_mbps);

        let bucket = self.buckets.get(peer_id).unwrap();

        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Refill tokens based on elapsed time
        let last = bucket.last_refill_us.load(Ordering::Relaxed);
        let elapsed_us = now_us.saturating_sub(last);
        if elapsed_us > 1000 {
            // Refill every 1ms minimum
            let new_tokens = (elapsed_us as u128 * bucket.rate_bytes_per_sec as u128 / 1_000_000) as u64;
            if new_tokens > 0
                && bucket
                    .last_refill_us
                    .compare_exchange_weak(last, now_us, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                let current = bucket.tokens.load(Ordering::Relaxed);
                let refilled = (current + new_tokens).min(bucket.capacity);
                bucket.tokens.store(refilled, Ordering::Release);
            }
        }

        // Try to consume tokens
        let needed = bytes as u64;
        loop {
            let current = bucket.tokens.load(Ordering::Relaxed);
            if current < needed {
                return false; // Rate limited
            }
            if bucket
                .tokens
                .compare_exchange_weak(
                    current,
                    current - needed,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.buckets.len()
    }

    /// Remove stale buckets for peers no longer in the tracker.
    pub fn cleanup(&self, active_peers: &DashMap<String, PeerState>) {
        self.buckets.retain(|id, _| active_peers.contains_key(id));
    }
}

impl Default for BandwidthLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a bandwidth transfer should be allowed for the given peer.
pub fn apply_bandwidth_limit(
    limiter: &BandwidthLimiter,
    peer_id: &str,
    bytes: usize,
    max_bandwidth_mbps: u64,
) -> bool {
    limiter.try_consume(peer_id, bytes, max_bandwidth_mbps)
}

// ---------------------------------------------------------------------------
// Gossipsub message dedup (bloom filter)
// ---------------------------------------------------------------------------

/// Simple bloom filter for deduplicating gossipsub messages at the proxy
/// layer.
///
/// Gossipsub broadcasts the same message to all peers. If multiple peers
/// are connected through the same proxy, the proxy can detect duplicates
/// by hashing the first N bytes of each gossipsub message and checking
/// against a bloom filter. Duplicates are dropped before reaching the
/// backend, reducing backend load.
///
/// The bloom filter uses k=3 hash functions and a configurable bit array
/// size. False positive rate at 10,000 messages with 65536 bits (~8KB)
/// is approximately 0.8%.
pub struct GossipsubDedup {
    /// Bit array for the bloom filter.
    bits: Vec<AtomicU64>,
    /// Number of bits in the filter (bits.len() * 64).
    num_bits: usize,
    /// Number of messages inserted.
    count: AtomicU64,
    /// Maximum messages before the filter should be reset.
    max_entries: u64,
}

impl GossipsubDedup {
    /// Create a new gossipsub dedup filter.
    ///
    /// `num_bits` is rounded up to the next multiple of 64. For 10K messages,
    /// 65536 bits (8KB) gives ~0.8% false positive rate with k=3.
    pub fn new(num_bits: usize, max_entries: u64) -> Self {
        let num_words = num_bits.div_ceil(64);
        let actual_bits = num_words * 64;
        Self {
            bits: (0..num_words).map(|_| AtomicU64::new(0)).collect(),
            num_bits: actual_bits,
            count: AtomicU64::new(0),
            max_entries,
        }
    }

    /// Check if a message has been seen before, and insert it if not.
    ///
    /// Returns `true` if the message is likely a duplicate (already seen).
    /// Returns `false` if the message is new (not seen before).
    ///
    /// `data` should be the first N bytes of the gossipsub message (typically
    /// the message ID or the first 64 bytes). Hashing more bytes increases
    /// accuracy but costs CPU.
    pub fn check_and_insert(&self, data: &[u8]) -> bool {
        // Auto-reset if the filter is saturated
        let count = self.count.load(Ordering::Relaxed);
        if count >= self.max_entries {
            self.reset();
        }

        let (h1, h2, h3) = self.compute_hashes(data);

        // Check all three bits
        let all_set = self.get_bit(h1) && self.get_bit(h2) && self.get_bit(h3);

        if all_set {
            return true; // Likely duplicate
        }

        // Not a duplicate -- insert by setting all three bits
        self.set_bit(h1);
        self.set_bit(h2);
        self.set_bit(h3);
        self.count.fetch_add(1, Ordering::Relaxed);

        false
    }

    /// Compute k=3 hash positions using double hashing.
    /// h(i) = (h1 + i * h2) mod num_bits
    fn compute_hashes(&self, data: &[u8]) -> (usize, usize, usize) {
        let mut hasher1 = DefaultHasher::new();
        data.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        // Second hash: use a different seed by prepending a byte
        let mut hasher2 = DefaultHasher::new();
        0xFFu8.hash(&mut hasher2);
        data.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        let h1 = (hash1 as usize) % self.num_bits;
        let h2 = (hash1.wrapping_add(hash2) as usize) % self.num_bits;
        let h3 = (hash1.wrapping_add(hash2.wrapping_mul(2)) as usize) % self.num_bits;

        (h1, h2, h3)
    }

    /// Get a single bit from the bloom filter.
    fn get_bit(&self, pos: usize) -> bool {
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        let word = self.bits[word_idx].load(Ordering::Relaxed);
        (word >> bit_idx) & 1 == 1
    }

    /// Set a single bit in the bloom filter.
    fn set_bit(&self, pos: usize) {
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        self.bits[word_idx].fetch_or(1u64 << bit_idx, Ordering::Relaxed);
    }

    /// Reset the bloom filter (clear all bits).
    pub fn reset(&self) {
        for word in &self.bits {
            word.store(0, Ordering::Relaxed);
        }
        self.count.store(0, Ordering::Relaxed);
    }

    /// Number of messages inserted since last reset.
    pub fn message_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Approximate fill ratio (fraction of bits set). Higher values mean
    /// more false positives.
    pub fn fill_ratio(&self) -> f64 {
        let set_bits: usize = self
            .bits
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
            .sum();
        set_bits as f64 / self.num_bits as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Handshake detection tests --

    #[test]
    fn test_detect_multistream_with_varint() {
        // Varint 0x14 (20) + "/multistream/1.0.0\n"
        let mut buf = vec![0x14u8];
        buf.extend_from_slice(MULTISTREAM_PREFIX);
        buf.extend_from_slice(b"trailing data");

        assert!(is_libp2p_handshake(&buf));
        let info = LibP2pDetector::detect(&buf).unwrap();
        assert_eq!(info.header_offset, 0);
    }

    #[test]
    fn test_detect_multistream_without_varint() {
        let mut buf = Vec::new();
        buf.extend_from_slice(MULTISTREAM_PREFIX);
        buf.extend_from_slice(b"more stuff");

        assert!(is_libp2p_handshake(&buf));
        assert!(LibP2pDetector::detect(&buf).is_some());
    }

    #[test]
    fn test_detect_multistream_after_offset() {
        // Some WebSocket frame data before the multistream header
        let mut buf = vec![0x81, 0x7F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14];
        buf.extend_from_slice(MULTISTREAM_PREFIX);

        assert!(is_libp2p_handshake(&buf));
    }

    #[test]
    fn test_no_detect_random_data() {
        let buf = b"GET /api/v1/status HTTP/1.1\r\nHost: quillon.xyz\r\n\r\n";
        assert!(!is_libp2p_handshake(buf));
        assert!(LibP2pDetector::detect(buf).is_none());
    }

    #[test]
    fn test_no_detect_empty() {
        assert!(!is_libp2p_handshake(&[]));
        assert!(!is_libp2p_handshake(&[0x14]));
        assert!(!is_libp2p_handshake(b"/multi"));
    }

    #[test]
    fn test_no_detect_partial_match() {
        // Almost the right prefix but not quite
        let buf = b"/multistream/2.0.0\n";
        assert!(!is_libp2p_handshake(buf));
    }

    // -- Circuit breaker tests --

    #[test]
    fn test_circuit_breaker_closed_allows() {
        let mut cb = CircuitBreakerState::new(3, 60);
        assert!(cb.should_allow());
        assert!(!cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreakerState::new(3, 60);

        cb.record_failure();
        assert!(cb.should_allow()); // 1 failure, threshold is 3

        cb.record_failure();
        assert!(cb.should_allow()); // 2 failures

        cb.record_failure();
        // Now at 3 failures = threshold, breaker opens
        assert!(cb.is_open());
        assert!(!cb.should_allow()); // Blocked
    }

    #[test]
    fn test_circuit_breaker_success_resets() {
        let mut cb = CircuitBreakerState::new(3, 60);

        cb.record_failure();
        cb.record_failure();
        // 2 failures, one more would open
        cb.record_success();
        // Reset to 0 failures
        assert_eq!(cb.failures, 0);
        assert!(matches!(cb.state, BreakerState::Closed));

        // Need 3 more failures to open again
        cb.record_failure();
        cb.record_failure();
        assert!(cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_half_open_transition() {
        // Use a very short reset timeout for testing
        let mut cb = CircuitBreakerState::new(1, 0); // 0 seconds = immediate

        cb.record_failure(); // Opens immediately (threshold = 1)
        assert!(cb.is_open());

        // Wait a tiny bit for the timeout to elapse (0 seconds)
        std::thread::sleep(Duration::from_millis(1));

        // should_allow transitions to half-open and allows
        assert!(cb.should_allow());
        assert!(matches!(cb.state, BreakerState::HalfOpen));
    }

    #[test]
    fn test_circuit_breaker_half_open_success_closes() {
        let mut cb = CircuitBreakerState::new(1, 0);

        cb.record_failure();
        std::thread::sleep(Duration::from_millis(1));
        cb.should_allow(); // Transitions to half-open

        cb.record_success(); // Probe succeeded
        assert!(matches!(cb.state, BreakerState::Closed));
        assert_eq!(cb.failures, 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure_reopens() {
        let mut cb = CircuitBreakerState::new(1, 0);

        cb.record_failure();
        std::thread::sleep(Duration::from_millis(1));
        cb.should_allow(); // Transitions to half-open

        cb.record_failure(); // Probe failed
        assert!(cb.is_open());
    }

    // -- Peer tier and policy tests --

    #[test]
    fn test_peer_tier_bandwidth_ordering() {
        // Supernodes should have more bandwidth than bootstrap,
        // bootstrap more than validator, etc.
        assert!(PeerTier::Supernode.default_bandwidth_mbps() > PeerTier::Bootstrap.default_bandwidth_mbps());
        assert!(PeerTier::Bootstrap.default_bandwidth_mbps() > PeerTier::Validator.default_bandwidth_mbps());
        assert!(PeerTier::Validator.default_bandwidth_mbps() > PeerTier::Miner.default_bandwidth_mbps());
        assert!(PeerTier::Miner.default_bandwidth_mbps() > PeerTier::Unknown.default_bandwidth_mbps());
    }

    #[test]
    fn test_peer_tier_connection_limits() {
        assert!(PeerTier::Supernode.default_max_connections() > PeerTier::Unknown.default_max_connections());
        assert_eq!(PeerTier::Unknown.default_max_connections(), 1);
    }

    #[test]
    fn test_peer_policy_from_tier() {
        let policy = PeerPolicy::from_tier(PeerTier::Miner);
        assert_eq!(policy.tier, PeerTier::Miner);
        assert_eq!(policy.max_bandwidth_mbps, 50);
        assert_eq!(policy.max_connections, 2);
        assert_eq!(policy.priority, 2);
    }

    // -- Connection limiting tests --

    #[test]
    fn test_connection_limiting() {
        let tracker = PeerTracker::new(vec![], vec![]);
        let peer_id = "12D3KooWTestPeer";

        // Unknown tier allows 1 connection
        assert!(should_allow_peer(&tracker, peer_id));

        // Simulate an active connection
        {
            let peer = tracker.get_or_create(peer_id);
            peer.conn_opened();
        }

        // Second connection should be rejected (Unknown tier max = 1)
        assert!(!should_allow_peer(&tracker, peer_id));

        // Close the connection
        {
            let peer = tracker.get_or_create(peer_id);
            peer.conn_closed();
        }

        // Should be allowed again
        assert!(should_allow_peer(&tracker, peer_id));
    }

    #[test]
    fn test_peer_tracker_classification() {
        let bootstrap = vec!["12D3KooWBootstrap".to_string()];
        let supernode = vec!["12D3KooWSuper".to_string()];
        let tracker = PeerTracker::new(bootstrap, supernode);

        // Known supernode
        let peer = tracker.get_or_create("12D3KooWSuperNodeXYZ");
        assert_eq!(peer.tier, PeerTier::Supernode);
        drop(peer);

        // Known bootstrap
        let peer = tracker.get_or_create("12D3KooWBootstrapABC");
        assert_eq!(peer.tier, PeerTier::Bootstrap);
        drop(peer);

        // Unknown peer
        let peer = tracker.get_or_create("12D3KooWRandomPeer");
        assert_eq!(peer.tier, PeerTier::Unknown);
    }

    #[test]
    fn test_peer_tracker_cleanup() {
        let tracker = PeerTracker::new(vec![], vec![]);

        // Add some peers
        tracker.get_or_create("peer-a");
        tracker.get_or_create("peer-b");
        assert_eq!(tracker.peer_count(), 2);

        // Cleanup with 0 duration should remove idle peers with no connections
        // Use a small sleep to ensure they're stale.
        std::thread::sleep(Duration::from_millis(1));
        tracker.cleanup_stale(Duration::from_secs(0));
        assert_eq!(tracker.peer_count(), 0);
    }

    // -- Auto-classify tests --

    #[test]
    fn test_auto_classify_promotes_high_traffic_peer() {
        let tracker = PeerTracker::new(vec![], vec![]);

        // Create an Unknown peer with significant traffic
        {
            let peer = tracker.get_or_create("peer-miner-candidate");
            // Simulate >100MB transferred
            peer.bytes_in.store(80 * 1024 * 1024, Ordering::Relaxed);
            peer.bytes_out.store(30 * 1024 * 1024, Ordering::Relaxed);
            // Touch last_seen so idle < min_duration
            *peer.last_seen.write() = Instant::now();
        }

        // Verify it's Unknown before auto-classify
        {
            let peer = tracker.get_or_create("peer-miner-candidate");
            assert_eq!(peer.tier, PeerTier::Unknown);
        }

        // Run auto-classify with low thresholds (0s duration, 100MB bytes)
        tracker.auto_classify(Duration::from_secs(3600), 100 * 1024 * 1024);

        // Should be promoted to Miner
        let peer = tracker.get_or_create("peer-miner-candidate");
        assert_eq!(peer.tier, PeerTier::Miner);
    }

    #[test]
    fn test_auto_classify_does_not_demote_bootstrap() {
        let tracker = PeerTracker::new(
            vec!["BootstrapPeer".to_string()],
            vec![],
        );

        // Create a Bootstrap peer
        tracker.get_or_create("BootstrapPeerXYZ");

        // auto-classify should NOT change a Bootstrap peer
        tracker.auto_classify(Duration::from_secs(0), 0);

        let peer = tracker.get_or_create("BootstrapPeerXYZ");
        assert_eq!(peer.tier, PeerTier::Bootstrap);
    }

    #[test]
    fn test_auto_classify_skips_low_traffic() {
        let tracker = PeerTracker::new(vec![], vec![]);

        // Create an Unknown peer with very little traffic
        {
            let peer = tracker.get_or_create("peer-low-traffic");
            peer.bytes_in.store(1024, Ordering::Relaxed); // 1KB
        }

        // auto-classify should NOT promote (below 100MB threshold)
        tracker.auto_classify(Duration::from_secs(3600), 100 * 1024 * 1024);

        let peer = tracker.get_or_create("peer-low-traffic");
        assert_eq!(peer.tier, PeerTier::Unknown);
    }

    // -- Gossipsub dedup bloom filter tests --

    #[test]
    fn test_bloom_new_messages_are_not_duplicates() {
        let dedup = GossipsubDedup::new(65536, 10000);

        assert!(!dedup.check_and_insert(b"message-1"));
        assert!(!dedup.check_and_insert(b"message-2"));
        assert!(!dedup.check_and_insert(b"message-3"));

        assert_eq!(dedup.message_count(), 3);
    }

    #[test]
    fn test_bloom_duplicate_detection() {
        let dedup = GossipsubDedup::new(65536, 10000);

        // Insert a message
        assert!(!dedup.check_and_insert(b"block-hash-abc123"));

        // Same message should be detected as duplicate
        assert!(dedup.check_and_insert(b"block-hash-abc123"));

        // Different message should not be detected as duplicate
        assert!(!dedup.check_and_insert(b"block-hash-def456"));
    }

    #[test]
    fn test_bloom_auto_reset() {
        // Small filter with max 3 entries
        let dedup = GossipsubDedup::new(256, 3);

        dedup.check_and_insert(b"msg-1");
        dedup.check_and_insert(b"msg-2");
        dedup.check_and_insert(b"msg-3");

        assert_eq!(dedup.message_count(), 3);

        // Inserting a 4th message should trigger a reset
        // After reset, msg-4 is new
        let is_dup = dedup.check_and_insert(b"msg-4");
        assert!(!is_dup);

        // Count should be 1 (reset to 0, then inserted msg-4)
        assert_eq!(dedup.message_count(), 1);
    }

    #[test]
    fn test_bloom_fill_ratio() {
        let dedup = GossipsubDedup::new(65536, 100000);

        assert_eq!(dedup.fill_ratio(), 0.0);

        for i in 0..100 {
            dedup.check_and_insert(format!("message-{}", i).as_bytes());
        }

        let ratio = dedup.fill_ratio();
        assert!(ratio > 0.0);
        assert!(ratio < 0.1); // Should be very sparse with only 100 entries in 65536 bits
    }

    #[test]
    fn test_bloom_reset() {
        let dedup = GossipsubDedup::new(65536, 10000);

        dedup.check_and_insert(b"test-data");
        assert!(dedup.check_and_insert(b"test-data")); // Duplicate

        dedup.reset();
        assert_eq!(dedup.message_count(), 0);
        assert_eq!(dedup.fill_ratio(), 0.0);

        // After reset, same data is no longer a duplicate
        assert!(!dedup.check_and_insert(b"test-data"));
    }

    // -- Bandwidth limiter tests --

    #[test]
    fn test_bandwidth_limiter_allows_within_limit() {
        let limiter = BandwidthLimiter::new();

        // 10 Mbps = 1,250,000 bytes/sec, burst = 2,500,000 bytes
        // A 1KB transfer should easily be allowed
        assert!(apply_bandwidth_limit(&limiter, "peer-1", 1024, 10));
    }

    #[test]
    fn test_bandwidth_limiter_blocks_over_limit() {
        let limiter = BandwidthLimiter::new();

        // 1 Mbps = 125,000 bytes/sec, burst = 250,000 bytes
        // Exhaust the bucket with a large transfer
        assert!(apply_bandwidth_limit(&limiter, "peer-2", 250_000, 1));

        // Now a second transfer should be blocked (bucket is empty)
        assert!(!apply_bandwidth_limit(&limiter, "peer-2", 1, 1));
    }

    #[test]
    fn test_bandwidth_limiter_per_peer_isolation() {
        let limiter = BandwidthLimiter::new();

        // Exhaust peer-a's bucket
        apply_bandwidth_limit(&limiter, "peer-a", 250_000, 1);

        // peer-b should still have bandwidth available
        assert!(apply_bandwidth_limit(&limiter, "peer-b", 1024, 1));
    }
}
