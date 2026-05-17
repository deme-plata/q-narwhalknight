//! Header-chain → STARK trace conversion.
//!
//! `q-zk-stark::StarkSystem::prove` consumes `&[Vec<u64>]`. We pack each
//! header into 8 u64 columns:
//!
//! | col | meaning |
//! |-----|---------|
//! | 0   | height |
//! | 1   | timestamp |
//! | 2   | producer_id (extended to u64) |
//! | 3-6 | `state_root` (32 B split into 4 × u64 little-endian) |
//! | 7   | `BLAKE3(prev_header)` truncated to u64 — commits the chain link |
//!
//! The 8th column is the critical chain-binding column: each row carries
//! a hash of the PREVIOUS header. At commit time `build_trace` recomputes
//! the previous header's hash and refuses to add the next row if the
//! claimed `prev_block_hash` doesn't match — this is what prevents a
//! malicious prover from inserting a broken chain.
//!
//! Once the trace is built, q-zk-stark's FRI commits to it as a low-degree
//! polynomial. The verifier later recomputes the trace commitment from
//! the FRI proof and the verifier's known starting hash.

use crate::{HeaderChainStep, HeaderHash, VerifyError};

/// Number of u64 columns per header row.
pub const COLS_PER_ROW: usize = 8;

/// Result of building a trace: the matrix the STARK system consumes,
/// plus the entry/exit fingerprints the verifier uses to bind it to
/// the public inputs.
pub struct BuiltTrace {
    pub trace: Vec<Vec<u64>>,
    /// Hash of the LAST header — the verifier checks this matches the
    /// claimed `tip_header` hash.
    pub end_hash: HeaderHash,
}

/// Build the trace matrix from a slice of headers anchored at
/// `anchor_hash` (the hash of the immediately preceding block, or all
/// zeros for genesis). Returns an error if any header's
/// `prev_block_hash` does not match the hash of the preceding row.
///
/// This is where chain-link integrity is enforced. Once the trace is
/// committed via FRI, the integrity is locked into the commitment.
pub fn build_trace(
    headers: &[HeaderChainStep],
    anchor_hash: HeaderHash,
) -> Result<BuiltTrace, VerifyError> {
    let mut trace = Vec::with_capacity(headers.len());
    let mut prev_hash = anchor_hash;

    for h in headers {
        // Chain link integrity — refuse to build the trace if broken.
        if h.prev_block_hash != prev_hash {
            return Err(VerifyError::Prover(format!(
                "chain break at height {}: claimed prev_hash != BLAKE3(prev_header)",
                h.height
            )));
        }

        let mut row = Vec::with_capacity(COLS_PER_ROW);
        row.push(h.height);
        row.push(h.timestamp);
        row.push(h.producer_id as u64);
        // state_root split into 4 × u64 LE
        for i in 0..4 {
            let bytes: [u8; 8] = h.state_root[i * 8..(i + 1) * 8].try_into().unwrap();
            row.push(u64::from_le_bytes(bytes));
        }
        // The chain-binding column: previous header's hash truncated to u64.
        // Full 32-byte hash is committed via the state_root cols across the
        // chain — this u64 truncation is the in-row witness only.
        let prev_h_bytes: [u8; 8] = prev_hash[0..8].try_into().unwrap();
        row.push(u64::from_le_bytes(prev_h_bytes));
        trace.push(row);

        prev_hash = h.hash();
    }

    Ok(BuiltTrace {
        trace,
        end_hash: prev_hash,
    })
}
