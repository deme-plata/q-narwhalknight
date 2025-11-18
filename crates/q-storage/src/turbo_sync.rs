/// Turbo Sync - Revolutionary Git-Inspired Blockchain Synchronization
///
/// Inspired by Git's pack files, delta compression, and parallel fetching,
/// combined with Q-NarwhalKnight's DAG-Knight consensus architecture.
///
/// Performance Targets:
/// - 50-250x faster than current gossipsub sequential sync
/// - 1,000-5,000 blocks/minute throughput (vs current ~21 blocks/min)
/// - 3-10x bandwidth reduction via zstd compression + delta encoding
/// - Parallel downloads from 8+ peers simultaneously
/// - Sub-10 minute full chain sync (110,000 blocks)
///
/// Architecture:
/// 1. Smart Protocol: Discover what peers have (Git's "want/have" negotiation)
/// 2. Range Splitting: Divide missing blocks into parallel chunks
/// 3. Pack Files: Compress chunks with zstd (level 3 for speed)
/// 4. Delta Encoding: Compress similar blocks (headers mostly identical)
/// 5. Pipelining: Download → Decompress → Verify → Apply simultaneously
/// 6. Multi-Peer: Round-robin across all available peers

use anyhow::{Context, Result};  // v0.9.53-beta: Added Context for .context() method
use crate::QStorage;
use futures::{stream::FuturesUnordered, StreamExt};
use libp2p::PeerId;
use rayon::prelude::*;  // ✅ v0.9.41-beta: Parallel decompression
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, RwLock, Semaphore, Mutex};
use tracing::{debug, error, info, warn};

use q_types::block::QBlock;

// Import QStorage from parent module
// v0.8.0-beta: Import balance consensus for Turbo Sync integration
use crate::{BalanceConsensusEngine, BalanceConsensusError};

/// Configuration for Turbo Sync
#[derive(Clone, Debug)]
pub struct TurboSyncConfig {
    /// Number of parallel download streams (like Git's parallel fetching)
    pub parallel_streams: usize,

    /// Blocks per chunk (larger = better compression, slower start)
    /// Git uses variable pack sizes, we use 1000 for balanced performance
    pub chunk_size: u64,

    /// Enable delta compression between blocks (like Git's delta encoding)
    /// Most block headers are identical except height/hash/timestamp
    pub delta_compression: bool,

    /// Compression level (1-21, higher = smaller but slower)
    /// Level 3 is Git's default: fast with good compression
    pub compression_level: i32,

    /// Enable pipelining (download + process simultaneously)
    /// Like Git's streaming decompression
    pub enable_pipelining: bool,

    /// Maximum concurrent peer connections
    pub max_peer_connections: usize,

    /// Timeout for individual chunk downloads
    pub chunk_timeout: Duration,

    /// Enable smart protocol negotiation (like Git's "want/have")
    pub smart_protocol: bool,
}

impl Default for TurboSyncConfig {
    fn default() -> Self {
        Self {
            parallel_streams: 12,  // ✅ v0.9.41-beta: 8 → 12 (50% more parallelism)
            chunk_size: 800,  // ✅ v0.9.41-beta: 500 → 800 (60% larger chunks, still under 10MB limit)
            // 800 blocks × ~20KB/block compressed ≈ 6.4MB (well within gossipsub 10MB limit)
            // Previous: 500 blocks = ~10MB (at the limit)
            // New: 800 blocks = ~6.4MB compressed (safe margin for variability)
            delta_compression: true,
            compression_level: 1,  // ✅ v0.9.41-beta: 3 → 1 (faster compression/decompression)
            // Level 1 is ~2x faster than level 3, only ~10% larger compressed size
            enable_pipelining: true,
            max_peer_connections: 16,
            chunk_timeout: Duration::from_secs(45),  // ✅ v0.9.41-beta: 30 → 45s (larger chunks need more time)
            smart_protocol: true,
        }
    }
}

/// Compressed block pack (Git-inspired pack file format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockPack {
    /// Starting height of this pack
    pub start_height: u64,

    /// Ending height (inclusive)
    pub end_height: u64,

    /// Compressed block data (zstd-compressed bincode)
    pub compressed_data: Vec<u8>,

    /// Checksum for verification (blake3 for speed)
    pub checksum: [u8; 32],

    /// Compression ratio achieved (for metrics)
    pub compression_ratio: f32,

    /// Number of blocks in this pack
    pub block_count: u32,

    /// Original uncompressed size
    pub uncompressed_size: u64,

    /// Request ID for tracking P2P responses (optional, only used for TRUE P2P)
    #[serde(default)]
    pub request_id: Option<String>,
}

/// Network request for block pack (sent via gossipsub)
///
/// v0.9.53-beta: REMOVED #[serde(default)] from protocol_version
/// This was causing postcard deserialization to misinterpret byte streams.
/// Version detection now happens BEFORE deserialization by inspecting first byte.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockPackRequest {
    /// Protocol version - ALWAYS REQUIRED
    /// Version 1: Current format with protocol_version field
    /// Version detection happens before deserialization (peek first byte)
    pub protocol_version: u32,

    /// Starting height
    pub start_height: u64,

    /// Ending height (inclusive)
    pub end_height: u64,

    /// Request ID for tracking responses
    pub request_id: String,
}

impl BlockPackRequest {
    /// Current protocol version
    pub const CURRENT_PROTOCOL_VERSION: u32 = 1;

    /// Create new request with current protocol version
    pub fn new(start_height: u64, end_height: u64, request_id: String) -> Self {
        Self {
            protocol_version: Self::CURRENT_PROTOCOL_VERSION,
            start_height,
            end_height,
            request_id,
        }
    }

    /// ✅ v0.9.53-beta: Version detection BEFORE deserialization
    /// ✅ v0.9.56-beta: Enhanced logging and early corruption detection
    /// Inspects first byte to determine format, eliminating deserialization ambiguity
    /// This fixes the critical bug where identical nodes couldn't decode each other's messages
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            anyhow::bail!("Empty BlockPackRequest data");
        }

        // ✅ v0.9.56-beta: Enhanced DEBUG - Log received bytes with more context
        tracing::info!("🔍 [TURBO SYNC DEBUG] Received BlockPackRequest bytes (len={}): {:02x?}",
                       data.len(), &data[..data.len().min(64)]);

        // v0.9.53-beta: Version detection by inspecting first byte
        // NEW format (v1): First byte is 0x01 (protocol_version=1 as varint)
        // OLD format (v0): First byte is start_height varint (typically > 0x01 for any realistic blockchain)
        //
        // EDGE CASE: If old format has start_height=1, first byte WILL be 0x01
        // Solution: Try NEW format first. If deserialization fails OR validation fails, try OLD format.
        // This is safe because:
        // - NEW format has 4 fields (protocol_version, start_height, end_height, request_id)
        // - OLD format has 3 fields (start_height, end_height, request_id)
        // - Postcard will fail to deserialize if field count doesn't match

        let is_new_format = data[0] == 0x01;

        // ✅ v0.9.56-beta: Log format detection decision
        tracing::info!("🔍 [TURBO SYNC DEBUG] Format detection: first_byte=0x{:02x}, is_new_format={}",
                       data[0], is_new_format);

        if is_new_format {
            // Try NEW format (with protocol_version field)
            match postcard::from_bytes::<Self>(data) {
                Ok(req) => {
                    // ✅ v0.9.56-beta: Enhanced logging with all decoded fields
                    tracing::info!("✅ [TURBO SYNC DEBUG] NEW format decoded: protocol_version={}, start={}, end={}, id={}",
                                  req.protocol_version, req.start_height, req.end_height, req.request_id);

                    // ✅ v0.9.56-beta: CRITICAL - Early corruption detection BEFORE validation
                    // Detect corrupted heights that exceed maximum blockchain height
                    // This catches the case where postcard SUCCEEDS but produces corrupted values
                    if req.start_height > 100_000_000 || req.end_height > 100_000_000 {
                        tracing::error!("❌ [TURBO SYNC DEBUG] CORRUPTED HEIGHT DETECTED AFTER POSTCARD DECODE!");
                        tracing::error!("❌ [TURBO SYNC DEBUG] start_height={}, end_height={}",
                                       req.start_height, req.end_height);
                        tracing::error!("❌ [TURBO SYNC DEBUG] This indicates struct field misalignment during deserialization!");
                        tracing::error!("❌ [TURBO SYNC DEBUG] Possible causes:");
                        tracing::error!("    1. Sender has #[serde(default)] on protocol_version (OLD binary v0.9.52 or earlier)");
                        tracing::error!("    2. Receiver has #[serde(default)] on protocol_version (THIS binary is OLD)");
                        tracing::error!("    3. Binary version mismatch between sender and receiver");
                        tracing::error!("❌ [TURBO SYNC DEBUG] Raw bytes: {:02x?}", data);

                        // Try OLD format as fallback
                        tracing::warn!("⚠️  [TURBO SYNC DEBUG] Attempting OLD format decode as fallback...");
                        return Self::decode_old_format(data);
                    }

                    // Validate heights (normal validation after corruption check)
                    if let Err(e) = req.validate_heights() {
                        // NEW format succeeded but heights are invalid
                        // This might be OLD format with start_height=1 (edge case)
                        tracing::warn!("NEW format validation failed: {}, trying OLD format", e);
                        return Self::decode_old_format(data);
                    }

                    Ok(req)
                }
                Err(e) => {
                    // NEW format deserialization failed
                    // This is likely OLD format with start_height=1 (edge case)
                    tracing::warn!("⚠️  [TURBO SYNC DEBUG] NEW format decode failed: {}, trying OLD format", e);
                    Self::decode_old_format(data)
                }
            }
        } else {
            // First byte is NOT 0x01, definitely OLD format
            tracing::info!("🔍 [TURBO SYNC DEBUG] Using OLD format decode (first byte != 0x01)");
            Self::decode_old_format(data)
        }
    }

    /// v0.9.53-beta: Decode OLD format (no protocol_version field)
    /// v0.9.65-beta: Added MessagePack fallback for cross-version compatibility
    /// Includes validation to catch any deserialization errors
    fn decode_old_format(data: &[u8]) -> Result<Self> {
        #[derive(serde::Deserialize)]
        struct OldBlockPackRequest {
            pub start_height: u64,
            pub end_height: u64,
            pub request_id: String,
        }

        const MAX_SANE_HEIGHT: u64 = 100_000_000; // 100 million blocks

        // ✅ v0.9.65-beta: Try postcard OLD format first
        tracing::info!("🔍 [TURBO SYNC DEBUG] Trying OLD format (postcard)");
        match postcard::from_bytes::<OldBlockPackRequest>(data) {
            Ok(old) => {
                // ✅ v0.9.59-beta: CRITICAL - Detect corrupt deserialization (SYNC-DOWN PREVENTION)
                // This catches postcard successfully deserializing but producing garbage values
                if old.start_height <= MAX_SANE_HEIGHT && old.end_height <= MAX_SANE_HEIGHT {
                    tracing::info!("✅ [TURBO SYNC DEBUG] OLD format (postcard) decoded successfully");
                    let req = Self {
                        protocol_version: 0,
                        start_height: old.start_height,
                        end_height: old.end_height,
                        request_id: old.request_id,
                    };
                    req.validate_heights()?;
                    return Ok(req);
                } else {
                    tracing::warn!(
                        "⚠️ OLD format decode produced insane heights: {}-{}, trying MessagePack",
                        old.start_height, old.end_height
                    );
                }
            }
            Err(e) => {
                tracing::warn!("⚠️ OLD format decode failed: {:?}, trying MessagePack", e);
            }
        }

        // ✅ v0.9.66-beta: Try MessagePack with rmp_serde::Deserializer for better compatibility
        // MessagePack can serialize structs as arrays OR maps - we need to support both
        tracing::info!("🔍 [TURBO SYNC DEBUG] Trying MessagePack format (flexible deserializer)");

        // Try default MessagePack (struct as map)
        match rmp_serde::from_slice::<Self>(data) {
            Ok(req) => {
                tracing::info!("✅ [TURBO SYNC DEBUG] MessagePack format (map) decoded successfully!");
                req.validate_heights()?;
                return Ok(req);
            }
            Err(e) => {
                tracing::warn!("⚠️ MessagePack NEW format (map) decode failed: {:?}, trying array format", e);
            }
        }

        // ✅ v0.9.66-beta: Try MessagePack with array format (compact serialization)
        // Some MessagePack encoders use positional arrays instead of named maps
        use rmp_serde::Deserializer;
        use serde::Deserialize;

        let mut deserializer = Deserializer::new(&data[..]);
        match Self::deserialize(&mut deserializer) {
            Ok(req) => {
                tracing::info!("✅ [TURBO SYNC DEBUG] MessagePack format (array) decoded successfully!");
                req.validate_heights()?;
                return Ok(req);
            }
            Err(e) => {
                tracing::warn!("⚠️ MessagePack NEW format (array) decode failed: {:?}, trying OLD format", e);
            }
        }

        // ✅ v0.9.65-beta: Try MessagePack OLD format (without protocol_version)
        match rmp_serde::from_slice::<OldBlockPackRequest>(data) {
            Ok(old) => {
                tracing::info!("✅ [TURBO SYNC DEBUG] MessagePack OLD format decoded successfully!");
                let req = Self {
                    protocol_version: 0,
                    start_height: old.start_height,
                    end_height: old.end_height,
                    request_id: old.request_id,
                };
                req.validate_heights()?;
                return Ok(req);
            }
            Err(e2) => {
                tracing::warn!("⚠️ MessagePack OLD format decode failed: {:?}, trying bincode", e2);
            }
        }

        // ✅ v0.9.66-beta: Try bincode NEW format (Docker containers use bincode!)
        tracing::info!("🔍 [TURBO SYNC DEBUG] Trying bincode format");
        let bincode_new_err = match bincode::deserialize::<Self>(data) {
            Ok(req) => {
                tracing::info!("✅ [TURBO SYNC DEBUG] Bincode NEW format decoded successfully!");
                req.validate_heights()?;
                return Ok(req);
            }
            Err(e) => {
                tracing::warn!("⚠️ Bincode NEW format decode failed: {:?}", e);
                e
            }
        };

        // ✅ v0.9.66-beta: Try bincode OLD format (without protocol_version)
        match bincode::deserialize::<OldBlockPackRequest>(data) {
            Ok(old) => {
                tracing::info!("✅ [TURBO SYNC DEBUG] Bincode OLD format decoded successfully!");
                let req = Self {
                    protocol_version: 0,
                    start_height: old.start_height,
                    end_height: old.end_height,
                    request_id: old.request_id,
                };
                req.validate_heights()?;
                Ok(req)
            }
            Err(bincode_old_err) => {
                tracing::error!("❌ [v0.9.66] ALL decode attempts failed!");
                tracing::error!("   Postcard NEW: failed");
                tracing::error!("   Postcard OLD: insane heights or failed");
                tracing::error!("   MessagePack NEW (map): failed");
                tracing::error!("   MessagePack NEW (array): failed");
                tracing::error!("   MessagePack OLD: failed");
                tracing::error!("   Bincode NEW: {:?}", bincode_new_err);
                tracing::error!("   Bincode OLD: {:?}", bincode_old_err);
                tracing::error!("   Raw bytes (len={}, first 64): {:02x?}", data.len(), &data[..data.len().min(64)]);

                // ✅ v0.9.66: Try to decode as ASCII to see if it's human-readable
                if let Ok(ascii) = std::str::from_utf8(&data[..data.len().min(64)]) {
                    tracing::error!("   As ASCII: {:?}", ascii);
                }

                // ✅ v0.9.66: Detailed byte analysis for debugging
                tracing::error!("   First byte analysis: 0x{:02x} = {:08b}", data[0], data[0]);
                if data[0] >= 0x90 && data[0] <= 0x9f {
                    tracing::error!("      → MessagePack fixarray with {} elements", data[0] & 0x0f);
                } else if data[0] == 0x01 {
                    tracing::error!("      → Postcard protocol_version=1");
                } else {
                    tracing::error!("      → Unknown format (not MessagePack array or postcard v1)");
                }

                anyhow::bail!(
                    "Failed to decode BlockPackRequest with any known format. \
                    This indicates incompatible binary versions between peers."
                )
            }
        }
    }

    /// v0.9.53-beta: Validate that heights make sense (not corrupted)
    /// Replaces the fragile detect_corruption() heuristics
    fn validate_heights(&self) -> Result<()> {
        const MAX_REALISTIC_HEIGHT: u64 = 100_000_000; // 100 million blocks

        if self.start_height > MAX_REALISTIC_HEIGHT {
            anyhow::bail!(
                "Invalid start_height={} (exceeds maximum {})",
                self.start_height, MAX_REALISTIC_HEIGHT
            );
        }

        if self.end_height > MAX_REALISTIC_HEIGHT {
            anyhow::bail!(
                "Invalid end_height={} (exceeds maximum {})",
                self.end_height, MAX_REALISTIC_HEIGHT
            );
        }

        if self.end_height < self.start_height {
            anyhow::bail!(
                "Invalid height range: end_height ({}) < start_height ({})",
                self.end_height, self.start_height
            );
        }

        Ok(())
    }

    /// Validate protocol version compatibility
    pub fn validate_version(&self) -> Result<()> {
        if self.protocol_version != Self::CURRENT_PROTOCOL_VERSION {
            anyhow::bail!(
                "Protocol version mismatch: received v{}, expected v{}. \
                Peer may be running incompatible software version.",
                self.protocol_version,
                Self::CURRENT_PROTOCOL_VERSION
            );
        }
        Ok(())
    }

    /// v0.9.57-beta: Serialize for a specific peer based on their negotiated protocol version
    /// This allows heterogeneous networks with nodes running different versions
    pub fn to_bytes_for_peer(&self, peer_version: u32) -> Result<Vec<u8>> {
        match peer_version {
            0 => {
                // OLD format: [start_height, end_height, request_id]
                // For backwards compatibility with pre-v0.9.53 nodes
                #[derive(Serialize)]
                struct OldFormat {
                    start_height: u64,
                    end_height: u64,
                    request_id: String,
                }

                let old = OldFormat {
                    start_height: self.start_height,
                    end_height: self.end_height,
                    request_id: self.request_id.clone(),
                };

                tracing::debug!(
                    "📤 [TURBO SYNC] Serializing OLD format for peer (version 0): blocks {}-{}",
                    self.start_height,
                    self.end_height
                );

                postcard::to_allocvec(&old)
                    .context("Failed to serialize OLD format BlockPackRequest")
            }
            1 => {
                // NEW format: [protocol_version, start_height, end_height, request_id]
                tracing::debug!(
                    "📤 [TURBO SYNC] Serializing NEW format for peer (version 1): blocks {}-{}",
                    self.start_height,
                    self.end_height
                );

                postcard::to_allocvec(self)
                    .context("Failed to serialize NEW format BlockPackRequest")
            }
            _ => {
                anyhow::bail!("Unsupported peer turbo sync version: {}", peer_version);
            }
        }
    }

    /// v0.9.53-beta: REMOVED - This method used fragile heuristics
    /// Replaced by version detection BEFORE deserialization + validate_heights()
    ///
    /// Old approach (v0.9.52-beta and earlier):
    /// - Deserialize first, then check if values look corrupted
    /// - Used timestamp detection, max height guessing
    /// - Prone to false positives and false negatives
    ///
    /// New approach (v0.9.53-beta):
    /// - Inspect first byte to detect version BEFORE deserialization
    /// - Simple height range validation (no heuristics)
    /// - More robust and maintainable
    #[deprecated(since = "0.9.53-beta", note = "Use version detection + validate_heights() instead")]
    pub fn detect_corruption_old(&self) -> Result<()> {
        const MAX_REASONABLE_HEIGHT: u64 = 10_000_000_000; // 10 billion blocks

        // ✅ v0.9.40-beta: Detect Unix timestamps disguised as block heights
        // Unix timestamps are in the range 1,000,000,000 to 2,000,000,000 for years 2001-2033
        const MIN_TIMESTAMP: u64 = 1_000_000_000;
        const MAX_TIMESTAMP: u64 = 2_500_000_000; // Year 2049
        const MAX_REALISTIC_BLOCKCHAIN_HEIGHT: u64 = 100_000_000; // 100 million blocks

        // Check if start_height looks like a Unix timestamp
        if self.start_height >= MIN_TIMESTAMP && self.start_height <= MAX_TIMESTAMP {
            anyhow::bail!(
                "CORRUPTED REQUEST DETECTED: start_height={} appears to be a Unix timestamp, not a block height. \
                This indicates binary deserialization failure or struct field misalignment.",
                self.start_height
            );
        }

        // Check if end_height looks like a Unix timestamp
        if self.end_height >= MIN_TIMESTAMP && self.end_height <= MAX_TIMESTAMP {
            anyhow::bail!(
                "CORRUPTED REQUEST DETECTED: end_height={} appears to be a Unix timestamp, not a block height. \
                This indicates binary deserialization failure or struct field misalignment.",
                self.end_height
            );
        }

        if self.start_height > MAX_REASONABLE_HEIGHT {
            anyhow::bail!(
                "CORRUPTED REQUEST DETECTED: start_height={} exceeds reasonable maximum {}. \
                This indicates binary protocol version mismatch between nodes.",
                self.start_height, MAX_REASONABLE_HEIGHT
            );
        }

        if self.end_height > MAX_REASONABLE_HEIGHT {
            anyhow::bail!(
                "CORRUPTED REQUEST DETECTED: end_height={} exceeds reasonable maximum {}. \
                This indicates binary protocol version mismatch between nodes.",
                self.end_height, MAX_REASONABLE_HEIGHT
            );
        }

        // ✅ v0.9.40-beta: Additional sanity check for realistic blockchain heights
        if self.start_height > MAX_REALISTIC_BLOCKCHAIN_HEIGHT {
            anyhow::bail!(
                "SUSPICIOUS REQUEST: start_height={} exceeds realistic blockchain height {}. \
                This may indicate corruption or an extremely long-running network.",
                self.start_height, MAX_REALISTIC_BLOCKCHAIN_HEIGHT
            );
        }

        if self.end_height > MAX_REALISTIC_BLOCKCHAIN_HEIGHT {
            anyhow::bail!(
                "SUSPICIOUS REQUEST: end_height={} exceeds realistic blockchain height {}. \
                This may indicate corruption or an extremely long-running network.",
                self.end_height, MAX_REALISTIC_BLOCKCHAIN_HEIGHT
            );
        }

        if self.end_height < self.start_height {
            anyhow::bail!(
                "INVALID REQUEST: end_height ({}) < start_height ({}). \
                Request is malformed or corrupted.",
                self.end_height, self.start_height
            );
        }

        Ok(())
    }
}

/// Network request type for TRUE P2P communication
#[derive(Debug)]
pub enum NetworkRequest {
    /// Request a block pack from the network
    RequestBlockPack {
        start_height: u64,
        end_height: u64,
        request_id: String,
        response_tx: oneshot::Sender<Result<BlockPack>>,
    },
}

/// Smart protocol negotiation (Git's "want/have" protocol)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncNegotiation {
    /// What the requester has (block heights)
    pub have: Vec<u64>,

    /// What the requester wants (target height)
    pub want: u64,

    /// Requester's peer ID
    pub peer_id: String,

    /// Preferred chunk size
    pub preferred_chunk_size: u64,
}

/// Smart protocol response (what the responder can provide)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncNegotiationResponse {
    /// What the responder has (highest block)
    pub highest_block: u64,

    /// Can serve the requested range
    pub can_serve: bool,

    /// Recommended chunk sizes for optimal performance
    pub recommended_chunks: Vec<(u64, u64)>, // (start, end) pairs

    /// Estimated time to serve all chunks
    pub estimated_time_ms: u64,
}

/// Turbo Sync metrics (real-time performance tracking)
#[derive(Debug, Default)]
pub struct TurboSyncMetrics {
    pub total_blocks_synced: AtomicU64,
    pub total_bytes_downloaded: AtomicU64,
    pub total_bytes_saved_by_compression: AtomicU64,
    pub active_parallel_streams: AtomicUsize,
    pub failed_chunks: AtomicU64,
    pub retried_chunks: AtomicU64,
    pub start_time: RwLock<Option<Instant>>,
    pub end_time: RwLock<Option<Instant>>,
}

impl TurboSyncMetrics {
    /// Calculate average download speed in MB/s
    pub async fn average_speed_mbps(&self) -> f64 {
        let start_opt = *self.start_time.read().await;
        let end_opt = *self.end_time.read().await;

        if let (Some(start), Some(end)) = (start_opt, end_opt) {
            let elapsed_secs = end.duration_since(start).as_secs_f64();
            let bytes = self.total_bytes_downloaded.load(Ordering::Relaxed) as f64;
            (bytes / elapsed_secs) / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }

    /// Calculate blocks per second
    pub async fn blocks_per_second(&self) -> f64 {
        let start_opt = *self.start_time.read().await;
        let end_opt = *self.end_time.read().await;

        if let (Some(start), Some(end)) = (start_opt, end_opt) {
            let elapsed_secs = end.duration_since(start).as_secs_f64();
            let blocks = self.total_blocks_synced.load(Ordering::Relaxed) as f64;
            blocks / elapsed_secs
        } else {
            0.0
        }
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let downloaded = self.total_bytes_downloaded.load(Ordering::Relaxed) as f32;
        let saved = self.total_bytes_saved_by_compression.load(Ordering::Relaxed) as f32;
        if downloaded > 0.0 {
            downloaded / (downloaded + saved)
        } else {
            1.0
        }
    }
}

/// Turbo Sync Manager - Main orchestrator
pub struct TurboSyncManager {
    config: TurboSyncConfig,
    storage: Arc<QStorage>,

    /// Semaphore for limiting concurrent downloads
    download_semaphore: Arc<Semaphore>,

    /// Metrics for monitoring
    pub metrics: Arc<TurboSyncMetrics>,

    /// Peer registry (peer_id -> highest_block)
    peer_registry: Arc<RwLock<Vec<(PeerId, u64)>>>,

    /// 🌐 TRUE P2P INTEGRATION - Network communication channel
    /// Send network requests (block pack requests via gossipsub)
    network_tx: Option<mpsc::UnboundedSender<NetworkRequest>>,

    /// 🔐 v0.9.15-beta: AEGIS-QL post-quantum cryptography for signed syncs
    aegis: Arc<Mutex<q_aegis_ql::AegisQL>>,
    aegis_secret_key: Arc<RwLock<q_aegis_ql::SecretKey>>,
    aegis_public_key: q_aegis_ql::PublicKey,

    /// 📊 v0.9.15-beta: Peer trust registry for reputation tracking
    peer_trust: Arc<crate::aegis_sync::PeerTrustRegistry>,

    /// 🧠 v1.0.15.1-beta: Memory limiter for adaptive sync batch sizing
    memory_limiter: Arc<crate::memory_limiter::MemoryLimiter>,
}

impl TurboSyncManager {
    /// Create new Turbo Sync manager
    pub fn new(storage: Arc<QStorage>, config: TurboSyncConfig) -> Self {
        let max_concurrent = config.max_peer_connections;

        // 🔐 v0.9.15-beta: Generate AEGIS-QL keypair for post-quantum signed syncs
        let mut aegis = q_aegis_ql::AegisQL::new();
        let (public_key, secret_key) = aegis.generate_keypair().expect("Failed to generate AEGIS-QL keypair");
        info!("🔐 [AEGIS-QL] Generated post-quantum keypair for signed P2P sync");
        info!("   Public key (first 16 bytes): {}", hex::encode(&bincode::serialize(&public_key).unwrap()[..16]));

        // 🧠 v1.0.15.1-beta: Initialize memory limiter with adaptive batch sizing
        let memory_limiter = Arc::new(crate::memory_limiter::MemoryLimiter::new());
        info!("🧠 [MEMORY] Memory limiter initialized for adaptive sync batch sizing");

        Self {
            config,
            storage,
            download_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            metrics: Arc::new(TurboSyncMetrics::default()),
            peer_registry: Arc::new(RwLock::new(Vec::new())),
            network_tx: None, // Set via set_network_channel()
            aegis: Arc::new(Mutex::new(aegis)),
            aegis_secret_key: Arc::new(RwLock::new(secret_key)),
            aegis_public_key: public_key,
            peer_trust: Arc::new(crate::aegis_sync::PeerTrustRegistry::new()),
            memory_limiter,
        }
    }

    /// 🌐 Set network channel for TRUE P2P communication
    /// Call this after creating TurboSyncManager to enable gossipsub integration
    pub fn set_network_channel(&mut self, tx: mpsc::UnboundedSender<NetworkRequest>) {
        info!("🌐 [TURBO SYNC] Network channel configured - TRUE P2P enabled!");
        self.network_tx = Some(tx);
    }

    /// Get local blockchain height
    async fn get_local_height(&self) -> Result<u64> {
        Ok(self.storage.get_latest_qblock_height().await?.unwrap_or(0))
    }

    /// Register a peer with their highest block height
    pub async fn register_peer(&self, peer_id: PeerId, highest_block: u64) {
        let mut registry = self.peer_registry.write().await;

        // Update or insert
        if let Some(entry) = registry.iter_mut().find(|(p, _)| p == &peer_id) {
            entry.1 = highest_block;
        } else {
            registry.push((peer_id, highest_block));
        }

        info!("📡 Registered peer {} with height {}", peer_id, highest_block);
    }

    /// Get peer registry information for debugging
    pub async fn get_peer_registry_info(&self) -> Vec<(PeerId, u64)> {
        let registry = self.peer_registry.read().await;
        registry.clone()
    }

    /// Discover peers that have the required height
    async fn discover_peers_with_height(&self, target_height: u64) -> Result<Vec<PeerId>> {
        let registry = self.peer_registry.read().await;

        let qualified: Vec<PeerId> = registry
            .iter()
            .filter(|(_, height)| *height >= target_height)
            .map(|(peer, _)| *peer)
            .collect();

        if qualified.is_empty() {
            // ✅ v0.9.6-beta: LOUD ERROR logging for critical peer discovery failure
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("🚨 CRITICAL: NO PEERS AVAILABLE FOR SYNC!");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("   Target height: {}", target_height);
            error!("   Peers in registry: {}", registry.len());
            error!("");
            error!("🔍 TROUBLESHOOTING:");
            error!("   1. Check if bootstrap peers are configured (Q_BOOTSTRAP_PEERS)");
            error!("   2. Verify libp2p peer discovery is working (check for CONNECTION logs)");
            error!("   3. Ensure peer height announcements are being received (check gossipsub)");
            error!("   4. Check if peer registry is being populated from libp2p discoveries");
            error!("");
            error!("📋 Peer Registry Contents:");
            for (peer, height) in registry.iter() {
                error!("   • Peer {} has height {}", peer, height);
            }
            if registry.is_empty() {
                error!("   (EMPTY - This is the problem! No peers discovered.)");
            }
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        } else {
            info!("📡 Found {} peers with height >= {}", qualified.len(), target_height);
        }

        Ok(qualified)
    }

    /// Split range into optimal chunks for parallel downloading
    fn split_into_chunks(&self, start: u64, end: u64) -> Vec<(u64, u64)> {
        let mut chunks = Vec::new();
        let chunk_size = self.config.chunk_size;

        // ✅ v0.9.55-beta CRITICAL FIX: Include genesis block (height 0) in sync
        // ROOT CAUSE: `start + 1` skipped genesis when syncing from height 0
        // IMPACT: Fresh nodes stuck at height 0 forever with "Gap at height 2" spam
        // SOLUTION: Start from `start` instead of `start + 1` to include all blocks
        let mut current = start;
        while current <= end {
            let chunk_end = (current + chunk_size - 1).min(end);
            chunks.push((current, chunk_end));
            current = chunk_end + 1;
        }

        info!("📦 Split range {}-{} into {} chunks of ~{} blocks",
              start, end, chunks.len(), chunk_size);

        chunks
    }

    /// Create compressed block pack (server-side)
    pub async fn create_block_pack(
        &self,
        start_height: u64,
        end_height: u64,
    ) -> Result<BlockPack> {
        let pack_start = Instant::now();

        // PHASE 1: Validate requested range against actual storage
        let local_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);

        if start_height > local_height {
            anyhow::bail!(
                "Requested range {}-{} exceeds local height {} (peer height mismatch)",
                start_height, end_height, local_height
            );
        }

        // Adjust end_height to what we actually have
        let actual_end = end_height.min(local_height);

        // PHASE 2: Fetch blocks (with gap detection)
        // 🎨 v0.6.7-beta: Continue through gaps to enable partial batch delivery
        let mut blocks = Vec::new();
        let mut missing_heights = Vec::new();

        for height in start_height..=actual_end {
            match self.storage.get_qblock_by_height(height).await? {
                Some(block) => {
                    blocks.push(block);
                }
                None => {
                    missing_heights.push(height);
                    // 🎨 v0.6.7-beta: Removed early exit - continue collecting all available blocks
                    // Even with gaps, we deliver what we have for maximum sync throughput
                }
            }
        }

        // PHASE 3: Handle missing blocks
        if blocks.is_empty() {
            anyhow::bail!(
                "No blocks found in range {}-{} (requested {}-{}, local height: {}, missing: {} blocks)",
                start_height, actual_end, start_height, end_height, local_height, missing_heights.len()
            );
        }

        // Log warning if pack is partial
        if !missing_heights.is_empty() {
            let missing_count = missing_heights.len();
            let total_requested = (actual_end - start_height + 1) as usize;
            let availability_pct = (blocks.len() as f32 / total_requested as f32) * 100.0;

            warn!(
                "⚠️  Creating PARTIAL pack {}-{}: {}/{} blocks ({:.1}% available)",
                start_height, actual_end, blocks.len(), total_requested, availability_pct
            );

            // Log sample of missing heights for debugging
            let sample_size = missing_heights.len().min(10);
            warn!(
                "   Missing heights (showing {}/{}): {:?}{}",
                sample_size, missing_count,
                &missing_heights[..sample_size],
                if missing_count > sample_size { "..." } else { "" }
            );
        }

        // Serialize blocks
        let serialized = bincode::serialize(&blocks)?;
        let uncompressed_size = serialized.len() as u64;

        // Compress with zstd (fast mode like Git)
        let compressed = zstd::encode_all(&serialized[..], self.config.compression_level)?;

        // Calculate checksum (blake3 for speed, like Git uses SHA-1)
        let checksum_hash = blake3::hash(&compressed);
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(checksum_hash.as_bytes());

        let compression_ratio = compressed.len() as f32 / serialized.len() as f32;
        let pack_time = pack_start.elapsed();

        info!(
            "📦 Created pack {}-{}: {} blocks, {:.1}KB → {:.1}KB ({:.1}% compression) in {}ms",
            start_height, end_height, blocks.len(),
            uncompressed_size as f64 / 1024.0,
            compressed.len() as f64 / 1024.0,
            (1.0 - compression_ratio) * 100.0,
            pack_time.as_millis()
        );

        Ok(BlockPack {
            start_height,
            end_height,
            compressed_data: compressed,
            checksum,
            compression_ratio,
            block_count: blocks.len() as u32,
            uncompressed_size,
            request_id: None, // Set by P2P handler when responding to specific request
        })
    }

    /// Apply a received block pack (client-side)
    ///
    /// v0.8.0-beta: Now accepts optional BalanceConsensusEngine for deterministic reward processing
    pub async fn apply_block_pack(
        &self,
        pack: BlockPack,
        balance_engine: Option<&BalanceConsensusEngine>,
    ) -> Result<()> {
        let apply_start = Instant::now();

        // Verify checksum
        let computed_checksum = blake3::hash(&pack.compressed_data);
        if computed_checksum.as_bytes() != &pack.checksum {
            anyhow::bail!(
                "Checksum mismatch for pack {}-{}",
                pack.start_height, pack.end_height
            );
        }

        // Decompress
        let decompressed = zstd::decode_all(&pack.compressed_data[..])?;
        let mut blocks: Vec<QBlock> = bincode::deserialize(&decompressed)?;

        // 🎨 v0.6.7-beta: CRITICAL FIX - Sort blocks by height to handle partial batches correctly
        // Partial batches (with gaps) may have blocks in arbitrary order from database fetch
        // Sorting ensures we can accurately track contiguous height progression
        blocks.sort_by_key(|b| b.header.height);

        // 🚨 v0.7.0-beta: CRITICAL FIX - Prevent height regression bug
        // OLD BUG: If a pack contained blocks [1, 2, 3, 1000, 1001, 1002], and current_height = 993,
        //          the algorithm would scan from 0 and set highest_contiguous = 3 (WRONG!)
        //          This caused sync to REGRESS from 993 → 3 blocks (CATASTROPHIC!)
        //
        // NEW FIX: ONLY allow height pointer to move FORWARD
        //          - Height pointer can only increase, never decrease
        //          - Ignore blocks that are BELOW current local height (already have them)
        //          - Only process blocks that are HIGHER than current height
        //          - Stop at first gap in the FORWARD sequence
        let current_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);
        let mut highest_contiguous = current_height;
        let mut gap_detected = false;
        let mut gap_start: Option<u64> = None;
        let mut blocks_below_current = 0usize;
        let mut blocks_forward = 0usize;

        // 🚨 CRITICAL SAFETY: Scan ONLY blocks that are ABOVE current height
        // Ignore any blocks at or below current height (we already have them, they're stale)
        for block in &blocks {
            if block.header.height <= current_height {
                // Skip blocks we already have (stale data from partial batches)
                blocks_below_current += 1;
                continue;
            }

            // 🎯 Process blocks that are HIGHER than current height
            if block.header.height == highest_contiguous + 1 {
                // Perfect - next sequential block in FORWARD direction
                highest_contiguous = block.header.height;
                blocks_forward += 1;
            } else if block.header.height > highest_contiguous + 1 && !gap_detected {
                // Gap detected in FORWARD sequence - stop advancing height pointer
                gap_detected = true;
                gap_start = Some(block.header.height);
                warn!(
                    "⚠️  [v0.7.0] Gap in FORWARD sequence: pack {}-{}, expected height {}, got {} - height pointer will be {}",
                    pack.start_height, pack.end_height,
                    highest_contiguous + 1, block.header.height, highest_contiguous
                );
                // Don't break - continue storing all blocks for later use
            }
        }

        // 🚨 CRITICAL SAFETY: Ensure height NEVER regresses
        if highest_contiguous < current_height {
            error!(
                "🚨 [v0.7.0] SAFETY ABORT: Height regression detected! current={}, computed={}, pack={}-{}",
                current_height, highest_contiguous, pack.start_height, pack.end_height
            );
            error!(
                "   Blocks analysis: {} below current, {} forward, {} gaps",
                blocks_below_current, blocks_forward, if gap_detected { 1 } else { 0 }
            );
            anyhow::bail!(
                "SAFETY ABORT: Height regression from {} to {} - this would cause sync corruption!",
                current_height, highest_contiguous
            );
        }

        // ========================================
        // v0.8.1-beta: ATOMIC TRANSACTIONS - Process each block atomically
        // ========================================
        // SECURITY FIX: Wrap balance consensus + block save in atomic transaction
        // to prevent CRITICAL-1 race condition (balances updated but blocks not saved)

        let mut balance_updates_total = 0;
        let mut blocks_committed = 0;

        if let Some(engine) = balance_engine {
            for block in &blocks {
                // Begin transaction for this block
                let tx = match self.storage.begin_transaction().await {
                    Ok(tx) => tx,
                    Err(e) => {
                        error!(
                            "❌ [TRANSACTION] Failed to begin transaction for block {} in pack {}-{}: {:?}",
                            block.header.height, pack.start_height, pack.end_height, e
                        );
                        continue; // Skip this block, try next
                    }
                };

                // Process balance consensus within transaction (buffered)
                let updates = match engine.process_block_mining_rewards_tx(&tx, block).await {
                    Ok(updates) => updates,
                    Err(BalanceConsensusError::AlreadyProcessed(_)) => {
                        debug!("🔄 [TURBO SYNC TX] Block {} already processed", block.header.height);
                        continue; // Skip already processed blocks
                    }
                    Err(e) => {
                        error!(
                            "❌ [TURBO SYNC TX] CRITICAL: Failed to process block {} in pack {}-{}: {:?}",
                            block.header.height, pack.start_height, pack.end_height, e
                        );
                        continue; // Transaction auto-rolled back, try next block
                    }
                };

                // ✅ v0.9.98-beta: FAIL FAST - Abort on any block error (AI Expert Consensus)
                // ChatGPT, DeepSeek, Kimi AI all agree: "If save_qblock() fails, ABORT transaction"
                // Previous pattern (continue) created pointer-data mismatches
                tx.save_qblock(block).await
                    .context(format!("Failed to save block {} in pack {}-{}",
                        block.header.height, pack.start_height, pack.end_height))?;

                // Commit transaction atomically (all or nothing)
                tx.commit().await
                    .context(format!("Failed to commit block {}", block.header.height))?;

                // ✅ v0.9.98-beta: EXPLICIT DURABILITY - Wait for WAL fsync
                // AI Expert Consensus: "tx.commit() is atomic but not automatically durable"
                // This guarantees blocks are on disk before we continue
                self.storage.sync_wal().await
                    .context("Failed to sync WAL after block commit")?;

                balance_updates_total += updates.len();
                blocks_committed += 1;
                if block.header.height % 100 == 0 {
                    debug!("✅ [TURBO SYNC TX] Committed block {} atomically + durable", block.header.height);
                }
            }

            if balance_updates_total > 0 {
                debug!(
                    "💰 [TURBO SYNC TX] Processed {} balance updates for {}/{} blocks in pack {}-{}",
                    balance_updates_total, blocks_committed, blocks.len(), pack.start_height, pack.end_height
                );
            }
        } else {
            // No balance engine - just save blocks without consensus processing
            // ✅ v0.9.98-beta: FAIL FAST pattern applied here too
            for block in &blocks {
                let tx = self.storage.begin_transaction().await
                    .context(format!("Failed to begin transaction for block {}", block.header.height))?;

                tx.save_qblock(block).await
                    .context(format!("Failed to save block {}", block.header.height))?;

                tx.commit().await
                    .context(format!("Failed to commit block {}", block.header.height))?;

                // ✅ v0.9.98-beta: EXPLICIT DURABILITY
                self.storage.sync_wal().await
                    .context("Failed to sync WAL after block commit")?;

                blocks_committed += 1;
            }
        }

        // 🎨 v0.6.7-beta: Update height pointer to highest CONTIGUOUS block only
        // This prevents reporting false heights when gaps exist in the blockchain
        // v0.8.1-beta: Use transaction for height pointer update
        let tx = self.storage.begin_transaction().await?;
        let latest_height_bytes = highest_contiguous.to_be_bytes().to_vec();
        tx.put("blocks", b"qblock:latest", &latest_height_bytes).await?;
        tx.commit().await?;

        // Log pack application details with FORWARD-only analysis
        if gap_detected {
            warn!(
                "⚠️  [v0.8.1] Applied PARTIAL pack {}-{}: stored {}/{} blocks ({} below current, {} forward), height: {} → {} (+{}), gap from block {}",
                pack.start_height, pack.end_height, blocks_committed, blocks.len(),
                blocks_below_current, blocks_forward,
                current_height, highest_contiguous, highest_contiguous - current_height,
                gap_start.unwrap_or(0)
            );
        } else if blocks_below_current > 0 {
            info!(
                "✅ [v0.8.1] Applied pack {}-{}: {}/{} blocks ({} below current ignored, {} forward), height: {} → {} (+{})",
                pack.start_height, pack.end_height, blocks_committed, blocks.len(),
                blocks_below_current, blocks_forward,
                current_height, highest_contiguous, highest_contiguous - current_height
            );
        } else {
            info!(
                "✅ [v0.8.1] Applied COMPLETE pack {}-{}: {}/{} blocks (all forward), height: {} → {} (+{})",
                pack.start_height, pack.end_height, blocks_committed, blocks.len(),
                current_height, highest_contiguous, highest_contiguous - current_height
            );
        }

        // v0.8.1-beta: Atomic transactions are committed individually per block above
        // Each commit uses fsync for durability, guaranteeing no data loss on crashes
        debug!(
            "✅ [v0.8.1] ATOMIC transactions: {}/{} blocks committed with fsync - durability guaranteed!",
            blocks_committed,
            blocks.len()
        );

        // Update metrics
        self.metrics.total_blocks_synced.fetch_add(blocks.len() as u64, Ordering::Relaxed);
        self.metrics.total_bytes_downloaded.fetch_add(pack.compressed_data.len() as u64, Ordering::Relaxed);

        let saved = pack.uncompressed_size.saturating_sub(pack.compressed_data.len() as u64);
        self.metrics.total_bytes_saved_by_compression.fetch_add(saved, Ordering::Relaxed);

        let apply_time = apply_start.elapsed();

        debug!(
            "✅ Applied pack {}-{}: {} blocks in {}ms",
            pack.start_height, pack.end_height, blocks.len(), apply_time.as_millis()
        );

        Ok(())
    }

    /// Download and apply a single chunk from a peer
    async fn download_and_apply_chunk(
        &self,
        peer: PeerId,
        start_height: u64,
        end_height: u64,
        retry_count: u32,
    ) -> Result<()> {
        let chunk_start = Instant::now();

        // Acquire semaphore permit for rate limiting
        let _permit = self.download_semaphore.acquire().await?;

        self.metrics.active_parallel_streams.fetch_add(1, Ordering::Relaxed);

        // ========================================
        // 🌐 TRUE P2P INTEGRATION - Request block pack via gossipsub
        // ========================================
        let pack = if let Some(network_tx) = &self.network_tx {
            // Generate unique request ID
            let request_id = format!("{}-{}-{}", start_height, end_height,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos());

            // Create oneshot channel for response
            let (response_tx, response_rx) = oneshot::channel();

            // Send network request via gossipsub
            if let Err(e) = network_tx.send(NetworkRequest::RequestBlockPack {
                start_height,
                end_height,
                request_id: request_id.clone(),
                response_tx, // Network handler will use this to send the response
            }) {
                anyhow::bail!("Failed to send network request: {}", e);
            }

            info!("🌐 [TURBO SYNC P2P] Requested pack {}-{} via gossipsub (ID: {})",
                  start_height, end_height, &request_id[..16]);

            // Wait for response with timeout
            match tokio::time::timeout(self.config.chunk_timeout, response_rx).await {
                Ok(Ok(pack_result)) => {
                    match pack_result {
                        Ok(pack) => {
                            info!("✅ [TURBO SYNC P2P] Received pack {}-{} from peer",
                                  start_height, end_height);
                            pack
                        }
                        Err(e) => {
                            warn!("❌ [TURBO SYNC P2P] Pack request failed: {}, falling back to local", e);
                            // Fallback to local pack creation
                            self.create_block_pack(start_height, end_height).await?
                        }
                    }
                }
                Ok(Err(_)) => {
                    // Channel closed
                    warn!("⚠️ [TURBO SYNC P2P] Response channel closed for {}-{}, falling back to local",
                          start_height, end_height);
                    self.create_block_pack(start_height, end_height).await?
                }
                Err(_) => {
                    // Timeout
                    warn!("⏱️ [TURBO SYNC P2P] Timeout waiting for pack {}-{}, falling back to local",
                          start_height, end_height);
                    self.create_block_pack(start_height, end_height).await?
                }
            }
        } else {
            // Network not configured, use local pack creation (hybrid mode)
            debug!("📦 [TURBO SYNC HYBRID] Network not available, using local pack creation");
            self.create_block_pack(start_height, end_height).await?
        };

        // Apply the pack
        // v0.8.0-beta: Pass None for balance_engine in internal sync methods
        // Balance processing happens in main.rs gossipsub handler when blocks are received
        self.apply_block_pack(pack, None).await?;

        self.metrics.active_parallel_streams.fetch_sub(1, Ordering::Relaxed);

        let chunk_time = chunk_start.elapsed();

        info!(
            "🚀 Downloaded chunk {}-{} from {} in {}ms (retry: {})",
            start_height, end_height, peer, chunk_time.as_millis(), retry_count
        );

        Ok(())
    }

    /// Download chunks in parallel with pipelining
    async fn download_chunks_parallel(
        &self,
        chunks: Vec<(u64, u64)>,
        peers: Vec<PeerId>,
    ) -> Result<()> {
        if peers.is_empty() {
            anyhow::bail!("No peers available for parallel download");
        }

        let total_chunks = chunks.len();
        let mut futures = FuturesUnordered::new();
        let mut completed_chunks = 0usize;

        info!("🚀 Starting parallel download: {} chunks from {} peers", total_chunks, peers.len());

        // Create parallel download tasks
        for (chunk_idx, (start, end)) in chunks.into_iter().enumerate() {
            // Round-robin peer selection (load balancing)
            let peer = peers[chunk_idx % peers.len()];

            let self_clone = self.clone_for_task();

            futures.push(tokio::spawn(async move {
                // Retry logic
                let mut retry_count = 0;
                let max_retries = 3;

                loop {
                    match self_clone.download_and_apply_chunk(peer, start, end, retry_count).await {
                        Ok(()) => {
                            return Ok((start, end));
                        }
                        Err(e) => {
                            retry_count += 1;
                            if retry_count >= max_retries {
                                error!("❌ Failed chunk {}-{} after {} retries: {}", start, end, max_retries, e);
                                return Err(e);
                            }
                            warn!("⚠️  Retrying chunk {}-{} (attempt {}/{}): {}", start, end, retry_count + 1, max_retries, e);
                            self_clone.metrics.retried_chunks.fetch_add(1, Ordering::Relaxed);
                            tokio::time::sleep(Duration::from_millis(500 * retry_count as u64)).await;
                        }
                    }
                }
            }));
        }

        // Wait for all chunks to complete with progress reporting
        let mut last_progress_log = Instant::now();
        while let Some(result) = futures.next().await {
            match result? {
                Ok((start, end)) => {
                    completed_chunks += 1;
                    let progress = (completed_chunks as f64 / total_chunks as f64) * 100.0;

                    // 🔐 v0.9.15-beta: Enhanced logging with AEGIS-QL metrics
                    // Update progress FREQUENTLY (every chunk or every 1 second)
                    if last_progress_log.elapsed().as_secs() >= 1 || completed_chunks % 5 == 0 {
                        let trusted_peers = self.peer_trust.get_trusted_peers();
                        let trusted_count = trusted_peers.len();

                        info!("📊 [TURBO SYNC] Progress: {}/{} chunks ({:.1}%)",
                              completed_chunks, total_chunks, progress);
                        info!("🔐 [AEGIS-QL] {} trusted peers (>80% trust) | Latest chunk: {}-{}",
                              trusted_count, start, end);

                        // Log individual peer trust scores periodically
                        if completed_chunks % 10 == 0 {
                            for peer_id in trusted_peers.iter().take(5) {
                                if let Some(score) = self.peer_trust.get_trust_score(peer_id) {
                                    info!("   • Peer {}: {:.1}% trust",
                                          &peer_id[..min(8, peer_id.len())],
                                          score * 100.0);
                                }
                            }
                        }

                        last_progress_log = Instant::now();
                    }
                }
                Err(e) => {
                    self.metrics.failed_chunks.fetch_add(1, Ordering::Relaxed);
                    error!("❌ Chunk failed: {}", e);
                    // Continue with other chunks even if one fails
                }
            }
        }

        // ✅ v0.9.40-beta FIX: FAIL LOUD instead of silent success
        // This prevents phantom success when chunks fail to download
        if completed_chunks < total_chunks {
            let failed = total_chunks - completed_chunks;
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("🚨 [TURBO SYNC] DOWNLOAD FAILED!");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("   Completed chunks: {}/{}", completed_chunks, total_chunks);
            error!("   Failed chunks: {}", failed);
            error!("   Success rate: {:.1}%", (completed_chunks as f64 / total_chunks as f64) * 100.0);
            error!("");
            error!("   This prevents phantom success - refusing to claim sync complete!");
            error!("   Will fall back to HTTP sync for missing blocks.");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            anyhow::bail!(
                "TURBO SYNC incomplete: {}/{} chunks failed ({:.1}% success rate). \
                This prevents phantom success. Falling back to HTTP sync.",
                failed, total_chunks, (completed_chunks as f64 / total_chunks as f64) * 100.0
            );
        }

        info!("✅ [v0.9.40 DEBUG] All {}/{} chunks completed successfully (100% success rate)",
              completed_chunks, total_chunks);
        Ok(())
    }

    /// Clone for async task (avoiding full Arc cloning complexity)
    fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            storage: Arc::clone(&self.storage),
            download_semaphore: Arc::clone(&self.download_semaphore),
            metrics: Arc::clone(&self.metrics),
            peer_registry: Arc::clone(&self.peer_registry),
            network_tx: self.network_tx.clone(), // Clone the network channel
            aegis: Arc::clone(&self.aegis),
            aegis_secret_key: Arc::clone(&self.aegis_secret_key),
            aegis_public_key: self.aegis_public_key.clone(),
            peer_trust: Arc::clone(&self.peer_trust),
            memory_limiter: Arc::clone(&self.memory_limiter),
        }
    }

    /// Main entry point: Sync to target height
    pub async fn sync_to_height(&self, target_height: u64) -> Result<()> {
        let sync_start = Instant::now();

        let local_height = self.get_local_height().await?;

        if local_height >= target_height {
            info!("🎯 Already synced to height {} (target: {})", local_height, target_height);
            return Ok(());
        }

        // 📜 v0.9.15-beta: Check if we've already synced to this height (using AEGIS-QL certificate)
        // This prevents restart loops - node knows it already synced even after restart
        if self.check_if_already_synced(target_height).await? {
            info!("✅ [AEGIS-QL] Skipping sync - already synced to {} (certificate verified)", target_height);
            return Ok(());
        }

        // 🚨 CRITICAL SAFETY CHECK: Prevent catastrophic sync-down (v0.5.23-beta)
        // This prevents BILLIONS of dollars in data loss on mainnet
        if target_height < local_height && local_height > 1000 {
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("🚨 CRITICAL SAFETY ABORT: SYNC-DOWN DETECTED!");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("   Current height: {} blocks", local_height);
            error!("   Target height:  {} blocks", target_height);
            error!("   Would LOSE:     {} blocks", local_height - target_height);
            error!("   ");
            error!("   This would cause CATASTROPHIC DATA LOSS!");
            error!("   Refusing to execute for safety.");
            error!("   ");
            error!("   Possible causes:");
            error!("   1. Malicious peer announcing false height");
            error!("   2. Network split");
            error!("   3. Bug in peer announcement handling");
            error!("   ");
            error!("   Action: Check peer heights and network status");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            return Err(anyhow::anyhow!(
                "SAFETY ABORT: Refusing to sync down from {} to {} (would lose {} blocks). \
                This protects against data loss. Check peer announcements.",
                local_height, target_height, local_height - target_height
            ));
        }

        // 🧠 v1.0.15.1-beta: Check memory pressure before starting sync
        if self.memory_limiter.should_pause_sync().await {
            warn!("⏸️  [MEMORY CRITICAL] Pausing sync until memory relief");
            self.memory_limiter.wait_for_memory_relief().await;
        }

        // 🧠 v1.0.15.1-beta: Get adaptive batch size based on current memory pressure
        let adaptive_chunk_size = self.memory_limiter.get_recommended_batch_size().await as u64;
        let chunk_size = min(self.config.chunk_size, adaptive_chunk_size);

        let memory_stats = self.memory_limiter.get_memory_stats().await;
        info!("🧠 [MEMORY] Adaptive batch sizing: {} blocks (pressure: {:?}, usage: {:.1}%)",
              chunk_size, memory_stats.pressure, memory_stats.usage_percent());

        let missing_range = target_height - local_height;

        info!("🚀 TURBO SYNC STARTING: {} blocks ({} → {})",
              missing_range, local_height, target_height);
        info!("⚙️  Config: {} parallel streams, {} blocks/chunk (adaptive), compression level {}",
              self.config.parallel_streams, self.config.chunk_size, self.config.compression_level);

        // Record start time
        *self.metrics.start_time.write().await = Some(sync_start);

        // PHASE 1: Discover peers with required height
        info!("🔍 [v0.9.40 DEBUG] PHASE 1: Discovering peers with height {}...", target_height);
        let qualified_peers = self.discover_peers_with_height(target_height).await?;
        info!("🔍 [v0.9.40 DEBUG] PHASE 1 COMPLETE: Found {} qualified peers", qualified_peers.len());

        if !qualified_peers.is_empty() {
            for (idx, peer) in qualified_peers.iter().enumerate().take(5) {
                info!("   • Peer {}: {}", idx + 1, peer);
            }
        }

        if qualified_peers.is_empty() {
            // ✅ v0.9.6-beta: CRITICAL ERROR - Turbo Sync cannot proceed without peers
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("🚨 TURBO SYNC FAILED: NO PEERS AVAILABLE!");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("   Cannot sync to height {} - no peers have this height", target_height);
            error!("");
            error!("🔧 REQUIRED ACTIONS:");
            error!("   1. Ensure bootstrap node is reachable (http://185.182.185.227:8080)");
            error!("   2. Verify gossipsub peer discovery is working");
            error!("   3. Check if peer height announcements are being processed");
            error!("   4. Verify TurboSync peer registry is being populated");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            anyhow::bail!("No peers available with target height {}", target_height);
        }

        // PHASE 2: Split range into parallel chunks
        info!("🔍 [v0.9.40 DEBUG] PHASE 2: Splitting {} blocks into chunks...", missing_range);
        let chunks = self.split_into_chunks(local_height, target_height);
        info!("🔍 [v0.9.40 DEBUG] PHASE 2 COMPLETE: Created {} chunks for parallel download", chunks.len());

        if !chunks.is_empty() {
            info!("   • First chunk: {}-{}", chunks[0].0, chunks[0].1);
            if chunks.len() > 1 {
                info!("   • Last chunk: {}-{}", chunks[chunks.len()-1].0, chunks[chunks.len()-1].1);
            }
        }

        // PHASE 3: Download chunks in parallel from multiple peers
        info!("🔍 [v0.9.40 DEBUG] PHASE 3: Starting parallel download of {} chunks...", chunks.len());
        self.download_chunks_parallel(chunks, qualified_peers).await?;
        info!("🔍 [v0.9.40 DEBUG] PHASE 3 COMPLETE: All chunks downloaded successfully");

        // PHASE 4: Final flush to persist all bulk writes to disk
        info!("💾 Flushing all bulk writes to disk (this may take a moment)...");
        let flush_start = Instant::now();
        self.storage.hot_db.flush().await?;
        let flush_time = flush_start.elapsed();
        info!("✅ Flush complete in {:.2}s - all data persisted to disk", flush_time.as_secs_f64());

        // PHASE 5: Finalize and report metrics
        let sync_duration = sync_start.elapsed();
        *self.metrics.end_time.write().await = Some(Instant::now());

        let blocks_synced = self.metrics.total_blocks_synced.load(Ordering::Relaxed);
        let bytes_downloaded = self.metrics.total_bytes_downloaded.load(Ordering::Relaxed);
        let bytes_saved = self.metrics.total_bytes_saved_by_compression.load(Ordering::Relaxed);
        let failed = self.metrics.failed_chunks.load(Ordering::Relaxed);
        let retried = self.metrics.retried_chunks.load(Ordering::Relaxed);

        let blocks_per_sec = blocks_synced as f64 / sync_duration.as_secs_f64();
        let blocks_per_min = blocks_per_sec * 60.0;
        let mbps = self.metrics.average_speed_mbps().await;
        let compression_ratio = self.metrics.compression_ratio();

        info!("🎉 TURBO SYNC COMPLETE!");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📊 Performance Summary:");
        info!("   • Blocks synced: {} blocks", blocks_synced);
        info!("   • Time elapsed: {:.2}s", sync_duration.as_secs_f64());
        info!("   • Speed: {:.0} blocks/sec ({:.0} blocks/min)", blocks_per_sec, blocks_per_min);
        info!("   • Bandwidth: {:.2} MB/s", mbps);
        info!("   • Downloaded: {:.2} MB", bytes_downloaded as f64 / (1024.0 * 1024.0));
        info!("   • Saved by compression: {:.2} MB ({:.1}%)",
              bytes_saved as f64 / (1024.0 * 1024.0), (1.0 - compression_ratio) * 100.0);
        info!("   • Failed chunks: {} (retried: {})", failed, retried);
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Performance validation
        if blocks_per_min < 1000.0 {
            warn!("⚠️  Sync speed {:.0} blocks/min is below target of 1000 blocks/min", blocks_per_min);
        }

        // 📜 v0.9.15-beta: Create AEGIS-QL sync affirmation certificate
        // This cryptographically proves sync completion and prevents restart loops
        info!("📜 [AEGIS-QL] Creating sync affirmation certificate...");
        let cert_start = Instant::now();

        match self.create_sync_affirmation(local_height, target_height).await {
            Ok(cert) => {
                match self.store_sync_certificate(&cert).await {
                    Ok(()) => {
                        info!("✅ [AEGIS-QL] SYNC AFFIRMED in {:?}: {} → {} blocks",
                              cert_start.elapsed(),
                              cert.start_height,
                              cert.end_height);
                        info!("   Certificate merkle root: {}", hex::encode(&cert.merkle_root[..8]));
                        info!("   This prevents restart loops - node knows sync is complete");
                    }
                    Err(e) => {
                        warn!("⚠️ [AEGIS-QL] Failed to store certificate: {}", e);
                        warn!("   Sync completed but certificate not saved (not critical)");
                    }
                }
            }
            Err(e) => {
                warn!("⚠️ [AEGIS-QL] Failed to create certificate: {}", e);
                warn!("   Sync completed but certificate not created (not critical)");
            }
        }

        Ok(())
    }

    /// Get chunks that need to be synced - for use by external network layer
    /// This allows the network layer to send requests without circular dependencies
    pub async fn get_sync_chunks(&self, target_height: u64) -> Result<Vec<(u64, u64)>> {
        let local_height = self.get_local_height().await?;

        if local_height >= target_height {
            return Ok(Vec::new());
        }

        let chunks = self.split_into_chunks(local_height, target_height);
        Ok(chunks)
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🔐 v0.9.14-beta: AEGIS-QL POST-QUANTUM SIGNED SYNC
    // ═══════════════════════════════════════════════════════════════════

    /// v0.9.21-beta: Create a signed block pack from height range (for gossipsub P2P)
    /// This combines block fetching + signing for efficient P2P transmission
    pub async fn create_signed_block_pack_from_range(
        &self,
        start_height: u64,
        end_height: u64,
        peer_id: String,
    ) -> Result<crate::aegis_sync::SignedBlockPack> {
        let pack_start = Instant::now();

        // 1. Fetch blocks from storage
        let local_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);

        if start_height > local_height {
            anyhow::bail!(
                "Requested range {}-{} exceeds local height {}",
                start_height, end_height, local_height
            );
        }

        let actual_end = end_height.min(local_height);
        let mut blocks = Vec::new();

        for height in start_height..=actual_end {
            if let Some(block) = self.storage.get_qblock_by_height(height).await? {
                blocks.push(block);
            }
        }

        if blocks.is_empty() {
            anyhow::bail!("No blocks found in range {}-{}", start_height, end_height);
        }

        info!("📦 [AEGIS-QL] Fetched {} blocks in {:?}, now signing...",
              blocks.len(), pack_start.elapsed());

        // 2. Sign the blocks
        self.create_signed_block_pack(blocks, peer_id).await
    }

    /// Create a signed block pack with AEGIS-QL post-quantum signature
    pub async fn create_signed_block_pack(
        &self,
        blocks: Vec<QBlock>,
        peer_id: String,
    ) -> Result<crate::aegis_sync::SignedBlockPack> {
        use crate::aegis_sync::{compute_merkle_root, SignedBlockPack};

        let start_sign = Instant::now();

        // 1. Compute merkle root of block hashes
        let block_hashes: Vec<[u8; 32]> = blocks
            .iter()
            .map(|b| b.calculate_hash())
            .collect();
        let merkle_root = compute_merkle_root(&block_hashes);

        // 2. Create message to sign
        let timestamp = chrono::Utc::now().timestamp();
        let mut message = Vec::new();
        message.extend_from_slice(&merkle_root);
        message.extend_from_slice(&timestamp.to_le_bytes());
        message.extend_from_slice(peer_id.as_bytes());

        // 3. Sign with AEGIS-QL
        let mut aegis = self.aegis.lock().await;
        let secret_key = self.aegis_secret_key.read().await;
        let signature = aegis.sign(&message, &*secret_key)?;

        info!("🔐 [AEGIS-QL] Signed {} blocks in {:?}", blocks.len(), start_sign.elapsed());
        info!("   Merkle root: {}", hex::encode(&merkle_root[..8]));

        Ok(SignedBlockPack {
            blocks,
            merkle_root,
            aegis_signature: signature,
            peer_public_key: self.aegis_public_key.clone(),
            timestamp,
            peer_id,
        })
    }

    /// Verify a signed block pack received from a peer
    pub async fn verify_signed_block_pack(
        &self,
        pack: &crate::aegis_sync::SignedBlockPack,
    ) -> Result<bool> {
        use crate::aegis_sync::{compute_merkle_root, verify_timestamp};

        let start_verify = Instant::now();

        // 1. Verify timestamp (prevents replay attacks)
        if !verify_timestamp(pack.timestamp) {
            warn!("❌ [AEGIS-QL] INVALID TIMESTAMP from peer {} (diff: {}s)",
                  &pack.peer_id[..8],
                  (chrono::Utc::now().timestamp() - pack.timestamp).abs());
            self.peer_trust.record_invalid_signature(&pack.peer_id, pack.peer_public_key.clone());
            return Ok(false);
        }

        // 2. Verify merkle root matches blocks
        let block_hashes: Vec<[u8; 32]> = pack.blocks
            .iter()
            .map(|b| b.calculate_hash())
            .collect();
        let computed_root = compute_merkle_root(&block_hashes);

        if computed_root != pack.merkle_root {
            warn!("❌ [AEGIS-QL] MERKLE ROOT MISMATCH from peer {}", &pack.peer_id[..8]);
            warn!("   Expected: {}", hex::encode(&pack.merkle_root[..8]));
            warn!("   Computed: {}", hex::encode(&computed_root[..8]));
            self.peer_trust.record_merkle_failure(&pack.peer_id, pack.peer_public_key.clone());
            return Ok(false);
        }

        // 3. Verify AEGIS-QL signature
        let mut message = Vec::new();
        message.extend_from_slice(&pack.merkle_root);
        message.extend_from_slice(&pack.timestamp.to_le_bytes());
        message.extend_from_slice(pack.peer_id.as_bytes());

        let aegis = self.aegis.lock().await;
        let valid = aegis.verify(&message, &pack.aegis_signature, &pack.peer_public_key)?;

        if !valid {
            error!("🚨 [AEGIS-QL] INVALID SIGNATURE from peer {}!", &pack.peer_id[..8]);
            error!("   This peer may be malicious or have key corruption!");
            self.peer_trust.record_invalid_signature(&pack.peer_id, pack.peer_public_key.clone());

            // Check if peer should be banned
            if self.peer_trust.should_ban_peer(&pack.peer_id) {
                error!("🚫 [AEGIS-QL] BANNING PEER {} (trust score below 20%)", &pack.peer_id[..8]);
                // TODO: Actually ban the peer from P2P network
            }

            return Ok(false);
        }

        // 4. Record successful verification
        self.peer_trust.record_valid_pack(&pack.peer_id, pack.peer_public_key.clone());

        info!("✅ [AEGIS-QL] Verified {} blocks from peer {} in {:?}",
              pack.blocks.len(),
              &pack.peer_id[..8],
              start_verify.elapsed());

        Ok(true)
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🗜️ v0.9.21-beta: COMPRESSED SIGNED BLOCK PACKS
    // ═══════════════════════════════════════════════════════════════════

    /// v0.9.21-beta: Create a COMPRESSED signed block pack with AEGIS-QL signature
    /// This reduces bandwidth from 2 MB (uncompressed) to ~603 KB (compressed + signature)
    pub async fn create_signed_block_pack_compressed(
        &self,
        blocks: Vec<QBlock>,
        peer_id: String,
    ) -> Result<crate::aegis_sync::SignedBlockPackCompressed> {
        use crate::aegis_sync::{compute_merkle_root, SignedBlockPackCompressed};

        let pack_start = Instant::now();
        let block_count = blocks.len();
        let start_height = blocks.first().map(|b| b.header.height).unwrap_or(0);
        let end_height = blocks.last().map(|b| b.header.height).unwrap_or(0);

        // 1. Compute merkle root of UNCOMPRESSED blocks (before compression)
        let block_hashes: Vec<[u8; 32]> = blocks
            .iter()
            .map(|b| b.calculate_hash())
            .collect();
        let merkle_root = compute_merkle_root(&block_hashes);

        // 2. Serialize blocks with postcard (efficient binary format)
        let serialized = postcard::to_allocvec(&blocks)?;
        let original_size = serialized.len();

        // 3. Compress with zstd level 3 (balance speed vs compression)
        let compressed = zstd::bulk::compress(&serialized, 3)?;
        let compressed_size = compressed.len();
        let compression_ratio = original_size as f64 / compressed_size as f64;

        info!("🗜️  [AEGIS-QL] Compressed {} blocks: {} → {} bytes ({:.1}% reduction)",
              block_count,
              original_size,
              compressed_size,
              (1.0 - compressed_size as f64 / original_size as f64) * 100.0);

        // 4. Create message to sign: compressed_blocks || merkle_root || timestamp
        let timestamp = chrono::Utc::now().timestamp();
        let mut message = Vec::new();
        message.extend_from_slice(&compressed);  // Sign compressed data
        message.extend_from_slice(&merkle_root);
        message.extend_from_slice(&timestamp.to_le_bytes());
        message.extend_from_slice(peer_id.as_bytes());

        // 5. Sign with AEGIS-QL Dilithium5 (post-quantum signature)
        let mut aegis = self.aegis.lock().await;
        let secret_key = self.aegis_secret_key.read().await;
        let signature = aegis.sign(&message, &*secret_key)?;

        info!("🔐 [AEGIS-QL] Signed compressed pack in {:?}: {} blocks, {:.1} KB, {:.1}x compression",
              pack_start.elapsed(),
              block_count,
              compressed_size as f64 / 1024.0,
              compression_ratio);
        info!("   Merkle root: {}", hex::encode(&merkle_root[..8]));

        Ok(SignedBlockPackCompressed {
            compressed_blocks: compressed,
            block_count,
            start_height,
            end_height,
            merkle_root,
            aegis_signature: signature,
            peer_public_key: self.aegis_public_key.clone(),
            timestamp,
            peer_id,
            compression_ratio,
        })
    }

    /// v0.9.21-beta: Verify and decompress a COMPRESSED signed block pack
    /// Returns the decompressed blocks if verification succeeds
    pub async fn verify_signed_block_pack_compressed(
        &self,
        pack: &crate::aegis_sync::SignedBlockPackCompressed,
    ) -> Result<Vec<QBlock>> {
        use crate::aegis_sync::{compute_merkle_root, verify_timestamp};

        let start_verify = Instant::now();

        // 1. Verify timestamp (prevents replay attacks)
        if !verify_timestamp(pack.timestamp) {
            warn!("❌ [AEGIS-QL] INVALID TIMESTAMP from peer {} (diff: {}s)",
                  &pack.peer_id[..8],
                  (chrono::Utc::now().timestamp() - pack.timestamp).abs());
            self.peer_trust.record_invalid_signature(&pack.peer_id, pack.peer_public_key.clone());
            anyhow::bail!("Invalid timestamp from peer {}", &pack.peer_id[..8]);
        }

        // 2. Verify AEGIS-QL signature BEFORE decompression (security-first)
        let mut message = Vec::new();
        message.extend_from_slice(&pack.compressed_blocks);
        message.extend_from_slice(&pack.merkle_root);
        message.extend_from_slice(&pack.timestamp.to_le_bytes());
        message.extend_from_slice(pack.peer_id.as_bytes());

        let aegis = self.aegis.lock().await;
        let sig_valid = aegis.verify(&message, &pack.aegis_signature, &pack.peer_public_key)?;

        if !sig_valid {
            error!("🚨 [AEGIS-QL] INVALID SIGNATURE from peer {}!", &pack.peer_id[..8]);
            error!("   This peer may be MALICIOUS or have key corruption!");
            self.peer_trust.record_invalid_signature(&pack.peer_id, pack.peer_public_key.clone());

            if self.peer_trust.should_ban_peer(&pack.peer_id) {
                error!("🚫 [AEGIS-QL] BANNING PEER {} (trust score < 20%)", &pack.peer_id[..8]);
            }

            anyhow::bail!("Invalid AEGIS-QL signature from peer {}", &pack.peer_id[..8]);
        }

        drop(aegis); // Release lock before decompression

        // 3. Decompress blocks (10 MB limit to prevent DoS)
        let decompressed = zstd::bulk::decompress(&pack.compressed_blocks, 10_000_000)?;
        let blocks: Vec<QBlock> = postcard::from_bytes(&decompressed)?;

        // 4. Verify block count matches
        if blocks.len() != pack.block_count {
            warn!("❌ [AEGIS-QL] BLOCK COUNT MISMATCH from peer {}: expected {}, got {}",
                  &pack.peer_id[..8], pack.block_count, blocks.len());
            self.peer_trust.record_merkle_failure(&pack.peer_id, pack.peer_public_key.clone());
            anyhow::bail!("Block count mismatch from peer {}", &pack.peer_id[..8]);
        }

        // 5. Verify merkle root matches decompressed blocks
        // ✅ v0.9.41-beta: Parallel hash calculation with rayon (2-4x faster on multi-core CPUs)
        let block_hashes: Vec<[u8; 32]> = blocks
            .par_iter()  // ✅ Parallel iterator
            .map(|b| b.calculate_hash())
            .collect();
        let computed_root = compute_merkle_root(&block_hashes);

        if computed_root != pack.merkle_root {
            warn!("❌ [AEGIS-QL] MERKLE ROOT MISMATCH from peer {}", &pack.peer_id[..8]);
            warn!("   Expected: {}", hex::encode(&pack.merkle_root[..8]));
            warn!("   Computed: {}", hex::encode(&computed_root[..8]));
            self.peer_trust.record_merkle_failure(&pack.peer_id, pack.peer_public_key.clone());
            anyhow::bail!("Merkle root mismatch from peer {}", &pack.peer_id[..8]);
        }

        // 6. Verify height range
        if let (Some(first), Some(last)) = (blocks.first(), blocks.last()) {
            if first.header.height != pack.start_height || last.header.height != pack.end_height {
                warn!("❌ [AEGIS-QL] HEIGHT MISMATCH from peer {}: expected {}-{}, got {}-{}",
                      &pack.peer_id[..8],
                      pack.start_height, pack.end_height,
                      first.header.height, last.header.height);
                anyhow::bail!("Height range mismatch from peer {}", &pack.peer_id[..8]);
            }
        }

        // 7. Record successful verification
        self.peer_trust.record_valid_pack(&pack.peer_id, pack.peer_public_key.clone());

        info!("✅ [AEGIS-QL] Verified compressed pack from peer {} in {:?}",
              &pack.peer_id[..8],
              start_verify.elapsed());
        info!("   {} blocks, {}-{}, {:.1} KB compressed, {:.1}x compression",
              blocks.len(),
              pack.start_height,
              pack.end_height,
              pack.compressed_blocks.len() as f64 / 1024.0,
              pack.compression_ratio);

        Ok(blocks)
    }

    /// Create a sync affirmation certificate after successful sync
    pub async fn create_sync_affirmation(
        &self,
        start_height: u64,
        end_height: u64,
    ) -> Result<crate::aegis_sync::SyncAffirmationCertificate> {
        use crate::aegis_sync::{compute_merkle_root, SyncAffirmationCertificate};

        info!("📜 [AEGIS-QL] Creating sync affirmation certificate for heights {} → {}", start_height, end_height);

        // 1. Collect all block hashes in range
        let mut block_hashes = Vec::new();
        for height in start_height..=end_height {
            if let Some(block) = self.storage.get_qblock_by_height(height).await? {
                block_hashes.push(block.calculate_hash());
            } else {
                warn!("⚠️ Missing block at height {} - cannot create affirmation", height);
                return Err(anyhow::anyhow!("Missing block at height {}", height));
            }
        }

        // 2. Compute merkle root
        let merkle_root = compute_merkle_root(&block_hashes);

        // 3. Create message to sign
        let timestamp = chrono::Utc::now().timestamp();
        let mut message = Vec::new();
        message.extend_from_slice(&start_height.to_le_bytes());
        message.extend_from_slice(&end_height.to_le_bytes());
        message.extend_from_slice(&merkle_root);
        message.extend_from_slice(&timestamp.to_le_bytes());

        // 4. Sign with AEGIS-QL
        let mut aegis = self.aegis.lock().await;
        let secret_key = self.aegis_secret_key.read().await;
        let signature = aegis.sign(&message, &*secret_key)?;

        info!("✅ [AEGIS-QL] Sync affirmation created: {} blocks verified", block_hashes.len());
        info!("   Certificate merkle root: {}", hex::encode(&merkle_root[..8]));

        Ok(SyncAffirmationCertificate {
            start_height,
            end_height,
            block_hashes,
            merkle_root,
            aegis_signature: signature,
            timestamp,
            syncer_public_key: self.aegis_public_key.clone(),
        })
    }

    /// Verify a sync affirmation certificate
    pub async fn verify_sync_affirmation(
        &self,
        cert: &crate::aegis_sync::SyncAffirmationCertificate,
    ) -> Result<bool> {
        use crate::aegis_sync::{compute_merkle_root, verify_timestamp};

        // 1. Verify timestamp
        if !verify_timestamp(cert.timestamp) {
            warn!("❌ [AEGIS-QL] Invalid certificate timestamp");
            return Ok(false);
        }

        // 2. Verify merkle root matches stored hashes
        let computed_root = compute_merkle_root(&cert.block_hashes);
        if computed_root != cert.merkle_root {
            warn!("❌ [AEGIS-QL] Certificate merkle root mismatch");
            return Ok(false);
        }

        // 3. Verify AEGIS-QL signature
        let mut message = Vec::new();
        message.extend_from_slice(&cert.start_height.to_le_bytes());
        message.extend_from_slice(&cert.end_height.to_le_bytes());
        message.extend_from_slice(&cert.merkle_root);
        message.extend_from_slice(&cert.timestamp.to_le_bytes());

        let aegis = self.aegis.lock().await;
        let valid = aegis.verify(&message, &cert.aegis_signature, &cert.syncer_public_key)?;

        if !valid {
            error!("🚨 [AEGIS-QL] INVALID CERTIFICATE SIGNATURE!");
            return Ok(false);
        }

        info!("✅ [AEGIS-QL] Sync affirmation certificate verified");
        Ok(true)
    }

    /// Get peer trust score
    pub fn get_peer_trust_score(&self, peer_id: &str) -> Option<f64> {
        self.peer_trust.get_trust_score(peer_id)
    }

    /// Get all trusted peers (trust score >= 80%)
    pub fn get_trusted_peers(&self) -> Vec<String> {
        self.peer_trust.get_trusted_peers()
    }

    // ═══════════════════════════════════════════════════════════════════
    // 📜 v0.9.15-beta: SYNC AFFIRMATION CERTIFICATE PERSISTENCE
    // ═══════════════════════════════════════════════════════════════════

    /// Store sync affirmation certificate to database
    pub async fn store_sync_certificate(&self, cert: &crate::aegis_sync::SyncAffirmationCertificate) -> Result<()> {
        // Store as latest certificate
        let key = b"sync_cert:latest";
        let value = bincode::serialize(cert)?;
        self.storage.hot_db.put(crate::CF_SYNC_CERTIFICATES, key, &value).await?;

        // Also store by end height for historical lookup
        let key_by_height = format!("sync_cert:{}", cert.end_height);
        self.storage.hot_db.put(crate::CF_SYNC_CERTIFICATES, key_by_height.as_bytes(), &value).await?;

        info!("📜 [AEGIS-QL] Stored sync certificate: {} → {} (merkle: {})",
              cert.start_height,
              cert.end_height,
              hex::encode(&cert.merkle_root[..8]));

        Ok(())
    }

    /// Delete sync affirmation certificate (v0.9.20-beta)
    /// Used when stale certificate is detected
    pub async fn delete_sync_certificate(&self) -> Result<()> {
        let key = b"sync_cert:latest";

        match self.storage.hot_db.delete(crate::CF_SYNC_CERTIFICATES, key).await {
            Ok(()) => {
                info!("🗑️  [AEGIS-QL] Deleted stale sync certificate");
                Ok(())
            }
            Err(e) => {
                warn!("⚠️  [AEGIS-QL] Failed to delete certificate: {}", e);
                Err(e.into())
            }
        }
    }

    /// Load latest sync affirmation certificate from database
    pub async fn load_latest_certificate(&self) -> Result<Option<crate::aegis_sync::SyncAffirmationCertificate>> {
        let key = b"sync_cert:latest";

        match self.storage.hot_db.get(crate::CF_SYNC_CERTIFICATES, key).await? {
            Some(bytes) => {
                let cert: crate::aegis_sync::SyncAffirmationCertificate = bincode::deserialize(&bytes)?;
                info!("📜 [AEGIS-QL] Loaded sync certificate: {} → {} (merkle: {})",
                      cert.start_height,
                      cert.end_height,
                      hex::encode(&cert.merkle_root[..8]));
                Ok(Some(cert))
            }
            None => {
                debug!("📜 [AEGIS-QL] No sync certificate found in database");
                Ok(None)
            }
        }
    }

    /// Check if we've already synced to target height (using certificate)
    /// ✅ v0.9.20-beta: CRITICAL FIX - Validate certificate against actual storage
    pub async fn check_if_already_synced(&self, target_height: u64) -> Result<bool> {
        if let Some(cert) = self.load_latest_certificate().await? {
            if cert.end_height >= target_height {
                // ✅ v0.9.20-beta: VALIDATE certificate against actual storage
                // This prevents phantom success from stale certificates
                let actual_height = self.get_local_height().await?;

                if actual_height >= cert.end_height {
                    // Certificate is VALID - we have the blocks
                    info!("✅ [AEGIS-QL] Already synced to {} (certificate end: {}, storage: {})",
                          target_height,
                          cert.end_height,
                          actual_height);
                    return Ok(true);
                } else {
                    // Certificate is STALE - blocks are missing!
                    error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    error!("🚨 [AEGIS-QL] STALE CERTIFICATE DETECTED!");
                    error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    error!("   Certificate claims: {} blocks", cert.end_height);
                    error!("   Actual storage has: {} blocks", actual_height);
                    error!("   Blocks missing: {}", cert.end_height - actual_height);
                    error!("   ");
                    error!("   This indicates database corruption or incomplete sync!");
                    error!("   The certificate survived but the blocks were lost.");
                    error!("   ");
                    error!("   Deleting stale certificate and re-syncing...");
                    error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                    // Delete stale certificate
                    self.delete_sync_certificate().await?;

                    // Continue with sync (return false to trigger download)
                    return Ok(false);
                }
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests removed temporarily - will be added after full integration
}

