// ✅ v0.9.68-beta: Proper libp2p request-response protocol for block sync
// Replaces broken gossipsub-based turbo sync with proper request/response pattern

use crate::QBlock;
use anyhow::Result;
use async_trait::async_trait;
use futures::prelude::*;
use libp2p::request_response::Codec;
use serde::{Deserialize, Serialize};
use std::io;

/// Maximum blocks per request to prevent DoS
/// v1.0.46-beta: Increased from 1000 to 5000 for faster sync
/// Q-NarwhalKnight blocks are small (~2-5KB), so 5000 blocks = ~10-25MB per response
pub const MAX_BLOCKS_PER_REQUEST: usize = 5000;

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

/// v1.1.28-beta: Hybrid CBOR/JSON codec for block pack request/response
/// Supports both CBOR (preferred) and JSON (legacy) for backward compatibility
#[derive(Debug, Clone, Default)]
pub struct BlockPackCodec;

impl BlockPackCodec {
    /// Try to parse as CBOR first, then JSON for backward compatibility
    fn parse_request(buf: &[u8]) -> io::Result<BlockPackRequest> {
        if buf.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Empty request buffer",
            ));
        }

        // CBOR messages typically start with 0xa0-0xbf (map) or 0x80-0x9f (array)
        // JSON messages typically start with '{' (0x7b) or '[' (0x5b)
        let first_byte = buf[0];

        // Try CBOR first (preferred format)
        if let Ok(req) = serde_cbor::from_slice::<BlockPackRequest>(buf) {
            return Ok(req);
        }

        // Fall back to JSON for legacy compatibility
        if first_byte == b'{' || first_byte == b'[' {
            if let Ok(req) = serde_json::from_slice::<BlockPackRequest>(buf) {
                // Log legacy format usage for debugging
                #[cfg(feature = "tracing")]
                tracing::debug!("[BLOCK-PACK] Received legacy JSON request, consider upgrading peer");
                return Ok(req);
            }
        }

        // Neither format worked
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to parse request: not valid CBOR or JSON (first byte: 0x{:02x}, len: {})", first_byte, buf.len()),
        ))
    }

    /// Try to parse as CBOR first, then JSON for backward compatibility
    fn parse_response(buf: &[u8]) -> io::Result<BlockPackResponse> {
        if buf.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Empty response buffer - peer may have closed connection prematurely",
            ));
        }

        let first_byte = buf[0];

        // Try CBOR first (preferred format)
        if let Ok(res) = serde_cbor::from_slice::<BlockPackResponse>(buf) {
            return Ok(res);
        }

        // Fall back to JSON for legacy compatibility
        if first_byte == b'{' || first_byte == b'[' {
            if let Ok(res) = serde_json::from_slice::<BlockPackResponse>(buf) {
                #[cfg(feature = "tracing")]
                tracing::debug!("[BLOCK-PACK] Received legacy JSON response, consider upgrading peer");
                return Ok(res);
            }
        }

        // Neither format worked
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to parse response: not valid CBOR or JSON (first byte: 0x{:02x}, len: {})", first_byte, buf.len()),
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
        let mut buf = Vec::new();
        io.read_to_end(&mut buf).await?;

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
        // v1.1.28-beta: Use CBOR (compact, efficient)
        let bytes = serde_cbor::to_vec(&req)
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
        // 🚀 v1.6.0-SCRAMJET: Switch to CBOR for ~40% bandwidth reduction
        // BREAKING CHANGE: Old clients (pre-v1.1.28) will fail to parse CBOR responses.
        // Network-wide upgrade required. Legacy JSON support removed for performance.
        // CBOR is more compact than JSON (no field names repeated, binary encoding).
        let bytes = serde_cbor::to_vec(&res)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

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
