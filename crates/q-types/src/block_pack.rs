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

    /// Parse response - try bincode first, then CBOR/JSON for backward compatibility
    fn parse_response(buf: &[u8]) -> io::Result<BlockPackResponse> {
        if buf.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Empty response buffer - peer may have closed connection prematurely",
            ));
        }

        // Try bincode first (v3.4.15+ format with u128 support)
        // This is the only format that can handle blocks with u128 token amounts
        if let Ok(res) = bincode::deserialize::<BlockPackResponse>(buf) {
            return Ok(res);
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

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to parse response: not valid bincode, CBOR or JSON (first byte: 0x{:02x}, len: {})", buf[0], buf.len()),
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
        let bytes = bincode::serialize(&res)
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
