// ✅ v0.9.68-beta: Proper libp2p request-response protocol for block sync
// Replaces broken gossipsub-based turbo sync with proper request/response pattern

use crate::QBlock;
use anyhow::Result;
use async_trait::async_trait;
use futures::prelude::*;
use libp2p::request_response::Codec;
use serde::{Deserialize, Serialize};
use std::io;

/// Maximum blocks per request to prevent DoS and oversized responses
/// v8.1.5: Reduced from 5000 to 1000 — 5000 blocks serialized to ~211MB in bincode
/// which caused "Failed to parse response" errors on client nodes.
/// 1000 blocks ≈ 40-50MB per response, much more reliable over the network.
pub const MAX_BLOCKS_PER_REQUEST: usize = 1000;

/// Maximum response size in bytes (100MB safety limit)
/// v8.1.5: Prevents OOM from malicious or oversized responses
pub const MAX_RESPONSE_BYTES: usize = 100 * 1024 * 1024;

/// Block pack request for efficient blockchain sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockPackRequest {
    /// Starting block height (inclusive)
    pub start_height: u64,

    /// Ending block height (inclusive)
    pub end_height: u64,

    /// Maximum blocks to return (DoS protection)
    pub max_blocks: usize,
}

impl BlockPackRequest {
    /// Create a new block pack request
    pub fn new(start_height: u64, end_height: u64) -> Self {
        let block_count = end_height.saturating_sub(start_height) + 1;
        Self {
            start_height,
            end_height,
            max_blocks: MAX_BLOCKS_PER_REQUEST.min(block_count as usize),
        }
    }

    /// Validate request
    pub fn validate(&self) -> Result<()> {
        if self.start_height > self.end_height {
            anyhow::bail!("Invalid range: start > end");
        }
        if self.max_blocks > MAX_BLOCKS_PER_REQUEST {
            anyhow::bail!("Exceeds max blocks per request");
        }
        Ok(())
    }
}

/// v10.9.43 item 12: wire-format version tag for the parallel codec.
///
/// The legacy bincode format has no version prefix; its first byte is the
/// LE u64 vector length of `blocks`, which for typical packs is 0xC8, 0xE8,
/// or similar. By choosing `0x02` (which never appears as the first byte
/// of a u64-LE vec length < 2^56), we get a non-ambiguous sniff:
///
/// - First byte == 0x02 → v2 parallel-decode format
/// - First byte != 0x02 → fall through legacy parse chain (bincode → CBOR → ...)
///
/// v2 wire format (little-endian):
/// ```text
/// [version: u8 = 0x02]
/// [block_count: u32 LE]
/// [offsets: block_count × u32 LE]    // byte offset of each block within blocks_buf
/// [meta: bincode-encoded (start_height, end_height, has_more, peer_height, permanent_gap)]
/// [blocks_buf: concatenated bincode-encoded QBlock entries]
/// ```
///
/// Index overhead for 2000 blocks: 1 + 4 + 8000 = ~8 KB (negligible vs the
/// 40-50 MB body).
pub const BLOCK_PACK_V2_TAG: u8 = 0x02;

/// v10.9.43 item 12: codec versions a node can advertise / negotiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockPackWireVersion {
    /// Legacy bincode of the whole `BlockPackResponse` struct (current
    /// default). Decoded sequentially.
    LegacyBincode,
    /// v10.9.43 parallel-decode format with per-block offset index.
    V2Parallel,
}

/// Error type for v2 codec.
#[derive(Debug, thiserror::Error)]
pub enum BlockPackCodecError {
    #[error("buffer too short: got {got}, need at least {need}")]
    TooShort { got: usize, need: usize },
    #[error("unsupported wire format version: 0x{0:02x} (this client supports up to 0x02)")]
    UnsupportedVersion(u8),
    #[error("block_count {count} exceeds maximum {max}")]
    TooManyBlocks { count: usize, max: usize },
    #[error("offsets table truncated: need {need} bytes, have {have}")]
    TruncatedOffsets { need: usize, have: usize },
    #[error("invalid offset {offset} at index {index} (blocks_buf is {len} bytes)")]
    InvalidOffset { offset: usize, index: usize, len: usize },
    #[error("meta decode failed: {0}")]
    MetaDecode(String),
    #[error("block decode failed at index {index}: {reason}")]
    BlockDecode { index: usize, reason: String },
}

/// Block pack response containing requested blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockPackResponse {
    /// Blocks in the requested range
    pub blocks: Vec<QBlock>,

    /// Actual start height of returned blocks
    pub start_height: u64,

    /// Actual end height of returned blocks
    pub end_height: u64,

    /// Whether more blocks are available beyond this response
    pub has_more: bool,

    /// v1.0.45-beta: Peer's highest block height for progress tracking
    /// This allows the requesting node to show accurate sync progress
    #[serde(default)]
    pub peer_height: u64,

    /// v10.9.41: Permanent storage gap declaration.
    /// When the server's forward-seek finds blocks AT OR AFTER `start_height` but
    /// the first available block is more than the safety threshold past it (e.g.
    /// blocks 26,001..=100,440 are missing — a known network-wide pruning-bug
    /// loss), the server returns empty `blocks` but sets `permanent_gap =
    /// Some((26001, 100440))` so the client can advance its contiguous height to
    /// 100,441 instead of stalling forever.
    ///
    /// Client safety: only honour `permanent_gap` if the SAME gap is reported by
    /// at least N independent peers (configurable via Q_GAP_QUORUM, default 2),
    /// or trust a single peer when Q_GAP_TRUST_SINGLE_PEER=1 (operator override).
    /// `serde(default)` ensures older servers/clients ignore this field.
    #[serde(default)]
    pub permanent_gap: Option<(u64, u64)>,
}

impl BlockPackResponse {
    /// Create response from blocks with peer's current height for progress tracking
    /// v1.0.45-beta: Added peer_height parameter for sync progress display
    pub fn from_blocks(blocks: Vec<QBlock>, requested_end: u64, peer_height: u64) -> Self {
        if blocks.is_empty() {
            return Self {
                blocks: vec![],
                start_height: 0,
                end_height: 0,
                has_more: false,
                peer_height,
                permanent_gap: None,
            };
        }

        let start_height = blocks.first().unwrap().header.height;
        let end_height = blocks.last().unwrap().header.height;
        let has_more = end_height < requested_end;

        Self {
            blocks,
            start_height,
            end_height,
            has_more,
            peer_height,
            permanent_gap: None,
        }
    }

    /// v10.9.41: Constructor for the "we have a permanent gap, advance past it"
    /// response. Used by the server-side forward-seek clamp when the missing
    /// range is wider than the safety threshold. The client treats this as a
    /// signed-by-server claim "blocks (gap_start..=gap_end) are unrecoverable;
    /// advance your contiguous tip to gap_end+1". Quorum gate is the client's
    /// responsibility — see `permanent_gap` field doc.
    pub fn with_permanent_gap(peer_height: u64, gap_start: u64, gap_end: u64) -> Self {
        Self {
            blocks: vec![],
            start_height: 0,
            end_height: 0,
            has_more: true,
            peer_height,
            permanent_gap: Some((gap_start, gap_end)),
        }
    }

    /// Legacy constructor without peer_height (for backward compatibility)
    pub fn from_blocks_legacy(blocks: Vec<QBlock>, requested_end: u64) -> Self {
        Self::from_blocks(blocks, requested_end, 0)
    }

    // ─────────────────────────────────────────────────────────────────────
    // v10.9.43 item 12 — parallel chunk-response codec (v2 wire format)
    // ─────────────────────────────────────────────────────────────────────

    /// Encode the response in the v2 wire format (per-block offset index +
    /// parallel decode-ready layout).
    ///
    /// Wire layout: see `BLOCK_PACK_V2_TAG` doc comment.
    ///
    /// Backwards compat: this is an OPT-IN encoder. The default codec
    /// (`BlockPackCodec::write_response`) still emits legacy bincode unless
    /// the operator sets `Q_BLOCK_PACK_V2=1`. Old peers that receive a v2
    /// response see first byte 0x02, fail their bincode/CBOR/JSON/postcard
    /// chain, and return a clean `UnsupportedVersion`-style parse error
    /// (no panic, no corruption).
    pub fn to_bytes_v2(&self) -> Result<Vec<u8>, BlockPackCodecError> {
        let count = self.blocks.len();
        // Sanity cap — we already have MAX_BLOCKS_PER_REQUEST=1000, but
        // allow a 2x headroom in case future versions increase it. Hard
        // limit at u32::MAX since the count field is u32.
        let max_count = (MAX_BLOCKS_PER_REQUEST * 2).min(u32::MAX as usize);
        if count > max_count {
            return Err(BlockPackCodecError::TooManyBlocks {
                count,
                max: max_count,
            });
        }

        // Estimate: 1 byte version + 4 bytes count + 4 bytes per offset +
        // ~64B meta + ~25 KB per block.
        let mut out = Vec::with_capacity(9 + count * (4 + 25_000));

        out.push(BLOCK_PACK_V2_TAG);
        out.extend_from_slice(&(count as u32).to_le_bytes());

        // Reserve the offsets table — we backfill once we know the offsets.
        let offsets_start = out.len();
        out.resize(offsets_start + count * 4, 0);

        // Meta: encode the five non-blocks fields as a tuple.
        let meta = (
            self.start_height,
            self.end_height,
            self.has_more,
            self.peer_height,
            self.permanent_gap,
        );
        let meta_bytes = bincode::serialize(&meta)
            .map_err(|e| BlockPackCodecError::MetaDecode(e.to_string()))?;
        out.extend_from_slice(&meta_bytes);

        // Per-block: record offset (relative to blocks_base), then write
        // bincode-encoded QBlock.
        let blocks_base = out.len();
        for (i, b) in self.blocks.iter().enumerate() {
            let off = (out.len() - blocks_base) as u32;
            out[offsets_start + i * 4..offsets_start + i * 4 + 4]
                .copy_from_slice(&off.to_le_bytes());
            let block_bytes = bincode::serialize(b).map_err(|e| {
                BlockPackCodecError::BlockDecode {
                    index: i,
                    reason: e.to_string(),
                }
            })?;
            out.extend_from_slice(&block_bytes);
        }

        Ok(out)
    }

    /// Decode a v2-encoded buffer with rayon-parallel per-block deserialize.
    ///
    /// Wire layout: see `BLOCK_PACK_V2_TAG` doc comment.
    ///
    /// Returns `UnsupportedVersion(first_byte)` if the buffer is not v2 —
    /// callers should fall through to their legacy parse chain on that
    /// error.
    pub fn from_bytes_v2(buf: &[u8]) -> Result<Self, BlockPackCodecError> {
        use rayon::prelude::*;

        // Minimum: version + count = 5 bytes.
        if buf.len() < 5 {
            return Err(BlockPackCodecError::TooShort {
                got: buf.len(),
                need: 5,
            });
        }
        let version = buf[0];
        if version != BLOCK_PACK_V2_TAG {
            return Err(BlockPackCodecError::UnsupportedVersion(version));
        }

        // Block count (u32 LE).
        let count = u32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
        let max_count = (MAX_BLOCKS_PER_REQUEST * 2).min(u32::MAX as usize);
        if count > max_count {
            return Err(BlockPackCodecError::TooManyBlocks {
                count,
                max: max_count,
            });
        }

        // Offsets table.
        let offsets_start: usize = 5;
        let offsets_bytes = count
            .checked_mul(4usize)
            .ok_or(BlockPackCodecError::TruncatedOffsets {
                need: usize::MAX,
                have: buf.len() - offsets_start,
            })?;
        let offsets_end = offsets_start
            .checked_add(offsets_bytes)
            .ok_or(BlockPackCodecError::TruncatedOffsets {
                need: usize::MAX,
                have: buf.len() - offsets_start,
            })?;
        if offsets_end > buf.len() {
            return Err(BlockPackCodecError::TruncatedOffsets {
                need: offsets_bytes,
                have: buf.len() - offsets_start,
            });
        }
        let offsets: Vec<u32> = buf[offsets_start..offsets_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Meta: decode in place via a Cursor so we can find blocks_base.
        let mut cursor = std::io::Cursor::new(&buf[offsets_end..]);
        let meta: (u64, u64, bool, u64, Option<(u64, u64)>) =
            bincode::deserialize_from(&mut cursor)
                .map_err(|e| BlockPackCodecError::MetaDecode(e.to_string()))?;
        let (start_height, end_height, has_more, peer_height, permanent_gap) = meta;
        let blocks_base = offsets_end + cursor.position() as usize;

        if blocks_base > buf.len() {
            return Err(BlockPackCodecError::TooShort {
                got: buf.len(),
                need: blocks_base,
            });
        }
        let blocks_buf = &buf[blocks_base..];

        // Compute per-block byte ranges from the offset table.
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(count);
        for i in 0..count {
            let start = offsets[i] as usize;
            let end = if i + 1 < count {
                offsets[i + 1] as usize
            } else {
                blocks_buf.len()
            };
            if start > end || end > blocks_buf.len() {
                return Err(BlockPackCodecError::InvalidOffset {
                    offset: start,
                    index: i,
                    len: blocks_buf.len(),
                });
            }
            ranges.push((start, end));
        }

        // Parallel decode. Each block is independent; rayon's
        // par_iter().collect::<Result<Vec, _>>() short-circuits on the first
        // error and reports its index.
        let decoded: Result<Vec<QBlock>, BlockPackCodecError> = ranges
            .par_iter()
            .enumerate()
            .map(|(i, (s, e))| {
                bincode::deserialize::<QBlock>(&blocks_buf[*s..*e]).map_err(|err| {
                    BlockPackCodecError::BlockDecode {
                        index: i,
                        reason: err.to_string(),
                    }
                })
            })
            .collect();
        let blocks = decoded?;

        Ok(Self {
            blocks,
            start_height,
            end_height,
            has_more,
            peer_height,
            permanent_gap,
        })
    }
}

/// Protocol identifier for block pack requests
#[derive(Debug, Clone)]
pub struct BlockPackProtocol;

impl AsRef<str> for BlockPackProtocol {
    fn as_ref(&self) -> &str {
        "/qnk/block-pack/1.0.0"
    }
}

/// v3.4.15-beta: Bincode codec for block pack request/response
/// CRITICAL FIX: Switched from CBOR to bincode because CBOR cannot serialize u128 values.
/// The migration from u64 to u128 for token amounts caused "The number can't be stored in CBOR"
/// errors which completely broke sync for blocks 199,002+.
///
/// Bincode benefits:
/// - Native u128 support (CBOR lacks this!)
/// - Compact binary format (more efficient than CBOR for numeric data)
/// - Already used for block storage (proven to work)
/// - Faster serialization/deserialization
#[derive(Debug, Clone, Default)]
pub struct BlockPackCodec;

impl BlockPackCodec {
    /// Parse request - try bincode first, then CBOR/JSON for backward compatibility
    fn parse_request(buf: &[u8]) -> io::Result<BlockPackRequest> {
        if buf.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Empty request buffer",
            ));
        }

        // Try bincode first (v3.4.15+ format with u128 support)
        if let Ok(req) = bincode::deserialize::<BlockPackRequest>(buf) {
            return Ok(req);
        }

        // Fall back to CBOR for legacy peers (pre-v3.4.15)
        // Note: CBOR works for requests since BlockPackRequest only has u64 fields
        if let Ok(req) = serde_cbor::from_slice::<BlockPackRequest>(buf) {
            return Ok(req);
        }

        // Fall back to JSON for very old peers
        let first_byte = buf[0];
        if first_byte == b'{' || first_byte == b'[' {
            if let Ok(req) = serde_json::from_slice::<BlockPackRequest>(buf) {
                return Ok(req);
            }
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to parse request: not valid bincode, CBOR or JSON (first byte: 0x{:02x}, len: {})", buf[0], buf.len()),
        ))
    }

    /// Parse response - try multiple formats for backward compatibility with older peers
    fn parse_response(buf: &[u8]) -> io::Result<BlockPackResponse> {
        if buf.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Empty response buffer - peer may have closed connection prematurely",
            ));
        }

        // v10.9.43 item 12: try v2 parallel-decode codec first when the
        // version byte matches. v2 emission is opt-in (Q_BLOCK_PACK_V2=1
        // on the server), so legacy peers never see this branch. Any v2
        // decode error short-circuits — we don't fall through to legacy
        // parsers because a v2 buffer that's malformed isn't a v1 buffer.
        if buf[0] == BLOCK_PACK_V2_TAG {
            return BlockPackResponse::from_bytes_v2(buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
        }

        // Try bincode first (v3.4.15+ format with u128 support)
        if let Ok(res) = bincode::deserialize::<BlockPackResponse>(buf) {
            return Ok(res);
        }

        // v10.2.8: Try deserializing as raw Vec<QBlock> (no wrapper struct).
        // Older peers send blocks directly without BlockPackResponse wrapper fields.
        // Header pattern: c8 00 00 00 00 00 00 00 = bincode u64 LE vec length (200 blocks),
        // followed immediately by QBlock data (height, "mainnet-genesis", etc.)
        if let Ok(blocks) = bincode::deserialize::<Vec<QBlock>>(buf) {
            if !blocks.is_empty() && blocks[0].header.height > 0 {
                let start_height = blocks.first().unwrap().header.height;
                let end_height = blocks.last().unwrap().header.height;
                eprintln!("📦 [BLOCK-PACK] Parsed {} blocks via raw Vec<QBlock> (heights {}-{})", blocks.len(), start_height, end_height);
                return Ok(BlockPackResponse {
                    blocks, start_height, end_height, has_more: false, peer_height: 0, permanent_gap: None,
                });
            }
        }

        // v10.2.8: Try legacy QBlock formats for older peers.
        // The codebase has 3 legacy struct versions with proven From<Legacy*> for QBlock
        // conversions (used by storage layer on 13M+ blocks). Try each as Vec<Legacy*>.
        use crate::legacy::{LegacyQBlock, LegacyQBlockV2, LegacyQBlockV3};

        // LegacyQBlockV2: most likely format for recent-but-old peers (v1.0.60-v1.0.85)
        if let Ok(legacy_blocks) = bincode::deserialize::<Vec<LegacyQBlockV2>>(buf) {
            if !legacy_blocks.is_empty() {
                let blocks: Vec<QBlock> = legacy_blocks.into_iter().map(|b| b.into()).collect();
                if blocks[0].header.height > 0 && blocks[0].header.timestamp < 2000000000 {
                    let start_height = blocks.first().unwrap().header.height;
                    let end_height = blocks.last().unwrap().header.height;
                    eprintln!("📦 [BLOCK-PACK LEGACY] Parsed {} blocks via Vec<LegacyQBlockV2> (heights {}-{})", blocks.len(), start_height, end_height);
                    return Ok(BlockPackResponse {
                        blocks, start_height, end_height, has_more: false, peer_height: 0, permanent_gap: None,
                    });
                }
            }
        }

        // LegacyQBlockV3: older format with old quantum metadata
        if let Ok(legacy_blocks) = bincode::deserialize::<Vec<LegacyQBlockV3>>(buf) {
            if !legacy_blocks.is_empty() {
                let blocks: Vec<QBlock> = legacy_blocks.into_iter().map(|b| b.into()).collect();
                if blocks[0].header.height > 0 && blocks[0].header.timestamp < 2000000000 {
                    let start_height = blocks.first().unwrap().header.height;
                    let end_height = blocks.last().unwrap().header.height;
                    eprintln!("📦 [BLOCK-PACK LEGACY] Parsed {} blocks via Vec<LegacyQBlockV3> (heights {}-{})", blocks.len(), start_height, end_height);
                    return Ok(BlockPackResponse {
                        blocks, start_height, end_height, has_more: false, peer_height: 0, permanent_gap: None,
                    });
                }
            }
        }

        // LegacyQBlock: oldest format (pre-v1.0.60)
        if let Ok(legacy_blocks) = bincode::deserialize::<Vec<LegacyQBlock>>(buf) {
            if !legacy_blocks.is_empty() {
                let blocks: Vec<QBlock> = legacy_blocks.into_iter().map(|b| b.into()).collect();
                if blocks[0].header.height > 0 && blocks[0].header.timestamp < 2000000000 {
                    let start_height = blocks.first().unwrap().header.height;
                    let end_height = blocks.last().unwrap().header.height;
                    eprintln!("📦 [BLOCK-PACK LEGACY] Parsed {} blocks via Vec<LegacyQBlock> (heights {}-{})", blocks.len(), start_height, end_height);
                    return Ok(BlockPackResponse {
                        blocks, start_height, end_height, has_more: false, peer_height: 0, permanent_gap: None,
                    });
                }
            }
        }

        // v10.2.8: Try MessagePack (rmp_serde)
        if let Ok(res) = rmp_serde::from_slice::<BlockPackResponse>(buf) {
            return Ok(res);
        }

        // v10.2.8: Try postcard
        if let Ok(res) = postcard::from_bytes::<BlockPackResponse>(buf) {
            return Ok(res);
        }

        // v10.2.8: Try bincode with a minimal struct (peers before v1.0.45 lack peer_height)
        #[derive(serde::Deserialize)]
        struct BlockPackResponseLegacy {
            blocks: Vec<QBlock>,
            start_height: u64,
            end_height: u64,
            has_more: bool,
        }
        if let Ok(res) = bincode::deserialize::<BlockPackResponseLegacy>(buf) {
            return Ok(BlockPackResponse {
                blocks: res.blocks,
                start_height: res.start_height,
                end_height: res.end_height,
                has_more: res.has_more,
                peer_height: 0,
                permanent_gap: None,
            });
        }
        // Also try MessagePack with legacy struct
        if let Ok(res) = rmp_serde::from_slice::<BlockPackResponseLegacy>(buf) {
            return Ok(BlockPackResponse {
                blocks: res.blocks,
                start_height: res.start_height,
                end_height: res.end_height,
                has_more: res.has_more,
                peer_height: 0,
                permanent_gap: None,
            });
        }
        // Also try postcard with legacy struct
        if let Ok(res) = postcard::from_bytes::<BlockPackResponseLegacy>(buf) {
            return Ok(BlockPackResponse {
                blocks: res.blocks,
                start_height: res.start_height,
                end_height: res.end_height,
                has_more: res.has_more,
                peer_height: 0,
                permanent_gap: None,
            });
        }

        // Fall back to CBOR for legacy peers (only works for old blocks without u128)
        if let Ok(res) = serde_cbor::from_slice::<BlockPackResponse>(buf) {
            return Ok(res);
        }

        // Fall back to JSON for very old peers
        let first_byte = buf[0];
        if first_byte == b'{' || first_byte == b'[' {
            if let Ok(res) = serde_json::from_slice::<BlockPackResponse>(buf) {
                return Ok(res);
            }
        }

        // v10.2.8: Dump first 64 bytes for debugging unknown format
        let header_hex: String = buf[..std::cmp::min(64, buf.len())]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ");
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to parse response: not valid bincode, rmp, CBOR, postcard or JSON (first byte: 0x{:02x}, len: {}, header: {})", buf[0], buf.len(), header_hex),
        ))
    }
}

#[async_trait]
impl Codec for BlockPackCodec {
    type Protocol = BlockPackProtocol;
    type Request = BlockPackRequest;
    type Response = BlockPackResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut buf = Vec::new();
        io.read_to_end(&mut buf).await?;

        Self::parse_request(&buf)
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        // v8.1.5: Bounded read — reject responses over MAX_RESPONSE_BYTES (100MB)
        // Previously unbounded read_to_end() allowed 211MB+ responses that failed to parse
        let mut buf = Vec::new();
        let mut limited = io.take(MAX_RESPONSE_BYTES as u64 + 1);
        limited.read_to_end(&mut buf).await?;

        if buf.len() > MAX_RESPONSE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Response too large: {} bytes (max {}MB). Peer may need to reduce batch size.",
                    buf.len(), MAX_RESPONSE_BYTES / 1024 / 1024),
            ));
        }

        Self::parse_response(&buf)
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        // v3.4.15-beta: Use bincode (native u128 support)
        let bytes = bincode::serialize(&req)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        io.write_all(&bytes).await?;
        io.flush().await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        // v3.4.15-beta: CRITICAL FIX - Use bincode instead of CBOR
        // CBOR cannot serialize u128 values, causing "The number can't be stored in CBOR"
        // errors for blocks 199,002+ after the u64→u128 migration.
        // Bincode natively supports u128 and is already used for block storage.
        //
        // v10.9.43 item 12: opt-in v2 parallel-decode format via env knob.
        // Set `Q_BLOCK_PACK_V2=1` on the server to emit the v2 wire format.
        // Old clients see first byte 0x02 and reject with a clean parse
        // error (no panic, no corruption). New clients sniff it and use
        // the rayon-parallel decoder.
        let block_count = res.blocks.len();
        let use_v2 = std::env::var("Q_BLOCK_PACK_V2")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .map(|n| n != 0)
            .unwrap_or(false);
        let bytes = if use_v2 {
            res.to_bytes_v2().map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("v2 encode failed: {}", e))
            })?
        } else {
            bincode::serialize(&res)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        };

        // v8.1.5: Log response size for debugging oversized responses
        let size_mb = bytes.len() as f64 / (1024.0 * 1024.0);
        if size_mb > 10.0 {
            eprintln!("⚠️ [BLOCK-PACK] Large response: {} blocks = {:.1}MB bincode", block_count, size_mb);
        }

        io.write_all(&bytes).await?;
        io.flush().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_pack_request_validation() {
        let valid = BlockPackRequest::new(0, 100);
        assert!(valid.validate().is_ok());
        assert_eq!(valid.max_blocks, 101);

        let mut invalid = BlockPackRequest::new(100, 50);
        assert!(invalid.validate().is_err());

        invalid.max_blocks = MAX_BLOCKS_PER_REQUEST + 1;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_block_pack_request_limits() {
        let large_request = BlockPackRequest::new(0, 5000);
        assert_eq!(large_request.max_blocks, MAX_BLOCKS_PER_REQUEST);
    }

    // ─────────────────────────────────────────────────────────────────────
    // v10.9.43 item 12: v2 wire format tests
    // ─────────────────────────────────────────────────────────────────────

    use crate::block::{BlockHeader, QuantumMetadata, VDFProof};

    fn make_test_block(height: u64) -> QBlock {
        QBlock {
            header: BlockHeader {
                height,
                phase: 5,
                network_id: "mainnet-genesis".to_string(),
                prev_block_hash: {
                    let mut p = [0u8; 32];
                    p[0..8].copy_from_slice(&height.to_le_bytes());
                    p
                },
                solutions_root: [0u8; 32],
                tx_root: [0u8; 32],
                state_root: [(height & 0xff) as u8; 32],
                timestamp: 1_700_000_000 + height,
                dag_round: height,
                vdf_proof: VDFProof::default(),
                anchor_validator: None,
                proposer: [(height ^ 0x11) as u8; 32],
                producer_id: 0,
                total_difficulty: 1000u128 + height as u128,
                producer_public_key: None,
                producer_signature: None,
                coinbase_merkle_root: None,
                total_coinbase_reward: None,
                coinbase_count: None,
            },
            mining_solutions: vec![],
            dag_parents: vec![],
            quantum_metadata: QuantumMetadata::default(),
            transactions: vec![],
            balance_updates: vec![],
            size_bytes: 0,
        }
    }

    fn make_response(n: u64) -> BlockPackResponse {
        let blocks: Vec<QBlock> = (1..=n).map(make_test_block).collect();
        BlockPackResponse {
            blocks,
            start_height: 1,
            end_height: n,
            has_more: false,
            peer_height: n + 100,
            permanent_gap: Some((42, 99)),
        }
    }

    /// v2 round-trip: serialize and deserialize must preserve all fields
    /// and all blocks bit-identically.
    #[test]
    fn test_v2_codec_roundtrip_100_blocks() {
        let resp = make_response(100);
        let bytes = resp.to_bytes_v2().expect("v2 encode");
        assert_eq!(bytes[0], BLOCK_PACK_V2_TAG, "version tag missing");

        let decoded = BlockPackResponse::from_bytes_v2(&bytes).expect("v2 decode");
        assert_eq!(decoded.blocks.len(), resp.blocks.len());
        assert_eq!(decoded.start_height, resp.start_height);
        assert_eq!(decoded.end_height, resp.end_height);
        assert_eq!(decoded.has_more, resp.has_more);
        assert_eq!(decoded.peer_height, resp.peer_height);
        assert_eq!(decoded.permanent_gap, resp.permanent_gap);
        // Per-block: hashes must match (verifies block bytes survived
        // the round trip).
        for i in 0..resp.blocks.len() {
            assert_eq!(
                decoded.blocks[i].header.height,
                resp.blocks[i].header.height,
                "height mismatch at {}",
                i
            );
            assert_eq!(
                decoded.blocks[i].calculate_hash(),
                resp.blocks[i].calculate_hash(),
                "hash mismatch at index {}",
                i
            );
        }
    }

    /// v2 single-block decode equivalence: each block decoded via the v2
    /// parallel path must equal the same block round-tripped via plain
    /// bincode.
    #[test]
    fn test_v2_per_block_matches_sequential_bincode() {
        let resp = make_response(20);
        let bytes = resp.to_bytes_v2().unwrap();
        let decoded = BlockPackResponse::from_bytes_v2(&bytes).unwrap();
        for i in 0..resp.blocks.len() {
            let seq = bincode::serialize(&resp.blocks[i]).unwrap();
            let seq_decoded: QBlock = bincode::deserialize(&seq).unwrap();
            assert_eq!(
                decoded.blocks[i].calculate_hash(),
                seq_decoded.calculate_hash(),
                "v2 vs sequential bincode mismatch at {}",
                i
            );
        }
    }

    /// v1 client (legacy parse chain) reading a v2 buffer must fail
    /// cleanly. We simulate by calling `from_bytes_v2` on a legacy
    /// bincode payload and asserting `UnsupportedVersion`.
    #[test]
    fn test_v1_client_rejects_v2_with_unsupported_version() {
        // Build a buffer that does NOT start with 0x02: a legitimate
        // bincode-encoded BlockPackResponse starts with the u64-LE vec
        // length of `blocks`.
        let resp = make_response(3);
        let legacy_bytes = bincode::serialize(&resp).unwrap();
        // Sanity: legacy bincode does NOT have version 0x02 as first byte.
        assert_ne!(legacy_bytes[0], BLOCK_PACK_V2_TAG);

        // A v2 decoder seeing a legacy buffer must report UnsupportedVersion.
        let err = BlockPackResponse::from_bytes_v2(&legacy_bytes).unwrap_err();
        match err {
            BlockPackCodecError::UnsupportedVersion(v) => {
                assert_eq!(v, legacy_bytes[0]);
            }
            other => panic!("expected UnsupportedVersion, got {:?}", other),
        }
    }

    /// v2 client reading a v1 payload via the unified `parse_response`
    /// must succeed via the legacy bincode branch.
    #[test]
    fn test_v2_client_reads_legacy_bincode() {
        let resp = make_response(3);
        let legacy_bytes = bincode::serialize(&resp).unwrap();
        let decoded = BlockPackCodec::parse_response(&legacy_bytes).expect("legacy parse");
        assert_eq!(decoded.blocks.len(), resp.blocks.len());
    }

    /// v2 client reading a v2 payload via the unified `parse_response`
    /// must succeed via the v2 branch.
    #[test]
    fn test_v2_client_reads_v2_payload() {
        let resp = make_response(5);
        let v2_bytes = resp.to_bytes_v2().unwrap();
        let decoded = BlockPackCodec::parse_response(&v2_bytes).expect("v2 parse");
        assert_eq!(decoded.blocks.len(), resp.blocks.len());
        assert_eq!(decoded.peer_height, resp.peer_height);
    }

    /// Truncated v2 buffer must error with TooShort, not panic.
    #[test]
    fn test_v2_truncated_buffer_reports_too_short() {
        let resp = make_response(2);
        let bytes = resp.to_bytes_v2().unwrap();
        // Truncate the offsets table.
        let truncated = &bytes[..6];
        let err = BlockPackResponse::from_bytes_v2(truncated).unwrap_err();
        matches!(err, BlockPackCodecError::TooShort { .. } | BlockPackCodecError::TruncatedOffsets { .. });
    }

    /// Empty pack round-trip — edge case.
    #[test]
    fn test_v2_empty_pack_roundtrip() {
        let resp = BlockPackResponse {
            blocks: vec![],
            start_height: 0,
            end_height: 0,
            has_more: false,
            peer_height: 0,
            permanent_gap: None,
        };
        let bytes = resp.to_bytes_v2().expect("encode empty");
        assert_eq!(bytes[0], BLOCK_PACK_V2_TAG);
        let decoded = BlockPackResponse::from_bytes_v2(&bytes).expect("decode empty");
        assert_eq!(decoded.blocks.len(), 0);
    }
}
