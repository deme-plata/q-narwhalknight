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
        }
    }

    /// Legacy constructor without peer_height (for backward compatibility)
    pub fn from_blocks_legacy(blocks: Vec<QBlock>, requested_end: u64) -> Self {
        Self::from_blocks(blocks, requested_end, 0)
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
                    blocks, start_height, end_height, has_more: false, peer_height: 0,
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
                        blocks, start_height, end_height, has_more: false, peer_height: 0,
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
                        blocks, start_height, end_height, has_more: false, peer_height: 0,
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
                        blocks, start_height, end_height, has_more: false, peer_height: 0,
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
        let block_count = res.blocks.len();
        let bytes = bincode::serialize(&res)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

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
}
