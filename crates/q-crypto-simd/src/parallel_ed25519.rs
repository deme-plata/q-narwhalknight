// TRUE Parallel SIMD Ed25519 Signature Verification
// Fixes the sequential bottleneck in batch_verification.rs

use ed25519_dalek::{Verifier, VerifyingKey, Signature as Ed25519Signature};
use rayon::prelude::*;
use anyhow::Result;
use tracing::{debug, info};

/// Parallel signature verification result
#[derive(Debug, Clone)]
pub struct ParallelVerificationResult {
    pub total: usize,
    pub valid: usize,
    pub invalid: usize,
    pub verification_times_us: Vec<u64>,
    pub throughput_sigs_per_sec: f64,
    /// v10.9.43: per-index validity. `results[i] == true` ⇔ signature `i`
    /// verified successfully. Length matches the input slices. Callers in
    /// the chunk-ingest path use this to attribute failures back to the
    /// originating block (see sync_pipeline.rs::validation_stage).
    pub results: Vec<bool>,
}

/// True parallel Ed25519 batch verifier using CPU threading
#[derive(Debug)]
pub struct ParallelEd25519Verifier {
    num_threads: usize,
    chunk_size: usize,
}

impl ParallelEd25519Verifier {
    /// Create new parallel verifier
    pub fn new(num_threads: usize) -> Self {
        let chunk_size = 32; // Optimal chunk size for cache locality

        info!("Initializing parallel Ed25519 verifier with {} threads, chunk size {}",
              num_threads, chunk_size);

        Self {
            num_threads,
            chunk_size,
        }
    }

    /// Verify batch of signatures in parallel using all CPU cores.
    /// Takes owned vectors to avoid lifetime issues.
    ///
    /// v10.9.43: now populates `results: Vec<bool>` with per-index validity
    /// (load-bearing for chunk-ingest failure attribution — see
    /// `crates/q-storage/src/sync_pipeline.rs::validation_stage`).
    pub fn verify_batch_parallel(
        &self,
        messages: &[Vec<u8>],
        signatures: &[Vec<u8>],
        public_keys: &[Vec<u8>],
    ) -> Result<ParallelVerificationResult> {
        let total = messages.len();

        if total != signatures.len() || total != public_keys.len() {
            return Err(anyhow::anyhow!("Batch size mismatch"));
        }

        debug!("Parallel verification of {} signatures using {} threads",
               total, self.num_threads);

        let start = std::time::Instant::now();

        // v10.9.43: collect per-index validity to support failure attribution.
        // into_par_iter().collect() preserves index order.
        let results: Vec<bool> = (0..total)
            .into_par_iter()
            .map(|i| verify_single(&messages[i], &signatures[i], &public_keys[i]))
            .collect();

        let elapsed = start.elapsed();
        let valid = results.iter().filter(|b| **b).count();
        let throughput = (total as f64) / elapsed.as_secs_f64();

        Ok(ParallelVerificationResult {
            total,
            valid,
            invalid: total - valid,
            verification_times_us: Vec::new(), // Don't collect per-sig times for performance
            throughput_sigs_per_sec: throughput,
            results,
        })
    }

    /// Verify batch with SIMD-optimized chunking.
    ///
    /// v10.9.43: populates `results` for per-index attribution. The chunk
    /// layout is preserved so cache-line locality wins still apply.
    pub fn verify_batch_chunked(
        &self,
        messages: &[Vec<u8>],
        signatures: &[Vec<u8>],
        public_keys: &[Vec<u8>],
    ) -> Result<ParallelVerificationResult> {
        let total = messages.len();

        if total != signatures.len() || total != public_keys.len() {
            return Err(anyhow::anyhow!("Batch size mismatch"));
        }

        debug!("Chunked parallel verification of {} signatures", total);

        let start = std::time::Instant::now();

        // Process in cache-friendly chunks; collect (index, valid) tuples
        // so we can reconstruct per-index ordering.
        let chunks: Vec<Vec<usize>> = (0..total)
            .collect::<Vec<_>>()
            .chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Build per-chunk results, then flatten in index order.
        let chunk_results: Vec<Vec<(usize, bool)>> = chunks
            .par_iter()
            .map(|chunk_indices| {
                chunk_indices
                    .iter()
                    .map(|&i| (i, verify_single(&messages[i], &signatures[i], &public_keys[i])))
                    .collect()
            })
            .collect();

        // Flatten — chunks are produced in order, indices within a chunk
        // are sequential, so the flatten preserves global index ordering.
        let mut results: Vec<bool> = vec![false; total];
        for chunk in chunk_results {
            for (i, ok) in chunk {
                results[i] = ok;
            }
        }

        let elapsed = start.elapsed();
        let valid = results.iter().filter(|b| **b).count();
        let throughput = (total as f64) / elapsed.as_secs_f64();

        Ok(ParallelVerificationResult {
            total,
            valid,
            invalid: total - valid,
            verification_times_us: Vec::new(),
            throughput_sigs_per_sec: throughput,
            results,
        })
    }
}

/// Single Ed25519 verify with size-tolerant input handling.
///
/// Wrong-length pubkey/signature or malformed pubkey are reported as
/// invalid (false), not panicked or errored. This is the same contract as
/// the old inline rayon loop in `sync_pipeline.rs::validation_stage`.
#[inline]
fn verify_single(message: &[u8], signature: &[u8], pubkey: &[u8]) -> bool {
    let Ok(pk_bytes): std::result::Result<&[u8; 32], _> = pubkey.try_into() else {
        return false;
    };
    let Ok(sig_bytes): std::result::Result<&[u8; 64], _> = signature.try_into() else {
        return false;
    };
    let Ok(pubkey) = VerifyingKey::from_bytes(pk_bytes) else {
        return false;
    };
    let sig = Ed25519Signature::from_bytes(sig_bytes);
    pubkey.verify(message, &sig).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{SigningKey, Signer};

    // Deterministic per-iteration keypair (ed25519-dalek 2.x dropped
    // SigningKey::generate; we use from_bytes with a counter-derived seed).
    fn signing_key_from_index(i: usize) -> SigningKey {
        let mut seed = [0u8; 32];
        seed[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        SigningKey::from_bytes(&seed)
    }

    #[test]
    fn test_parallel_verification() {
        let num_sigs = 100;
        let verifier = ParallelEd25519Verifier::new(num_cpus::get());

        // Generate test data
        let mut messages = Vec::new();
        let mut signatures = Vec::new();
        let mut public_keys = Vec::new();

        for i in 0..num_sigs {
            let signing_key = signing_key_from_index(i);
            let message = format!("test message {}", i).into_bytes();
            let signature = signing_key.sign(&message);

            messages.push(message);
            signatures.push(signature.to_bytes().to_vec());
            public_keys.push(signing_key.verifying_key().to_bytes().to_vec());
        }

        // Verify in parallel
        let result = verifier.verify_batch_parallel(&messages, &signatures, &public_keys)
            .expect("Verification failed");

        assert_eq!(result.valid, num_sigs);
        assert_eq!(result.invalid, 0);
        assert!(result.throughput_sigs_per_sec > 1000.0);
    }

    #[test]
    fn test_chunked_verification() {
        let num_sigs = 256;
        let verifier = ParallelEd25519Verifier::new(num_cpus::get());

        let mut messages = Vec::new();
        let mut signatures = Vec::new();
        let mut public_keys = Vec::new();

        for i in 0..num_sigs {
            let signing_key = signing_key_from_index(i);
            let message = format!("test message {}", i).into_bytes();
            let signature = signing_key.sign(&message);

            messages.push(message);
            signatures.push(signature.to_bytes().to_vec());
            public_keys.push(signing_key.verifying_key().to_bytes().to_vec());
        }

        let result = verifier.verify_batch_chunked(&messages, &signatures, &public_keys)
            .expect("Verification failed");

        assert_eq!(result.valid, num_sigs);
        assert_eq!(result.invalid, 0);
    }

    /// v10.9.43 item 11: verify per-index attribution. The chunk-ingest path
    /// uses `results[i]` to map a failed signature back to the block it
    /// belongs to, so this MUST report which specific indices failed.
    #[test]
    fn test_per_index_attribution_parallel() {
        let num_sigs = 50;
        let verifier = ParallelEd25519Verifier::new(num_cpus::get());

        let mut messages = Vec::new();
        let mut signatures = Vec::new();
        let mut public_keys = Vec::new();
        let mut expected = Vec::new();

        for i in 0..num_sigs {
            let signing_key = signing_key_from_index(i);
            let message = format!("attribution test {}", i).into_bytes();
            let mut signature = signing_key.sign(&message).to_bytes().to_vec();

            // Corrupt indices 3, 7, 11, 23, 42 — non-aligned to chunk_size=32
            // boundary to make sure attribution survives chunking.
            let corrupted = matches!(i, 3 | 7 | 11 | 23 | 42);
            if corrupted {
                signature[0] ^= 0xFF;
            }
            expected.push(!corrupted);

            messages.push(message);
            signatures.push(signature);
            public_keys.push(signing_key.verifying_key().to_bytes().to_vec());
        }

        let res = verifier
            .verify_batch_parallel(&messages, &signatures, &public_keys)
            .expect("verify failed");
        assert_eq!(res.results.len(), num_sigs);
        assert_eq!(res.results, expected, "per-index attribution mismatch");
        assert_eq!(res.valid, num_sigs - 5);
        assert_eq!(res.invalid, 5);
    }

    /// Same attribution test for the chunked path — verify index ordering
    /// is preserved across chunk boundaries.
    #[test]
    fn test_per_index_attribution_chunked() {
        let num_sigs = 80; // crosses 2 chunk_size=32 boundaries
        let verifier = ParallelEd25519Verifier::new(num_cpus::get());

        let mut messages = Vec::new();
        let mut signatures = Vec::new();
        let mut public_keys = Vec::new();
        let mut expected = Vec::new();

        for i in 0..num_sigs {
            let signing_key = signing_key_from_index(i);
            let message = format!("chunked {}", i).into_bytes();
            let mut signature = signing_key.sign(&message).to_bytes().to_vec();

            let corrupted = matches!(i, 0 | 31 | 32 | 63 | 64 | 79);
            if corrupted {
                signature[0] ^= 0xFF;
            }
            expected.push(!corrupted);

            messages.push(message);
            signatures.push(signature);
            public_keys.push(signing_key.verifying_key().to_bytes().to_vec());
        }

        let res = verifier
            .verify_batch_chunked(&messages, &signatures, &public_keys)
            .expect("verify failed");
        assert_eq!(res.results, expected, "chunked per-index attribution mismatch");
    }

    #[test]
    fn test_invalid_signatures() {
        let num_sigs = 50;
        let verifier = ParallelEd25519Verifier::new(num_cpus::get());

        let mut messages = Vec::new();
        let mut signatures = Vec::new();
        let mut public_keys = Vec::new();

        for i in 0..num_sigs {
            let signing_key = signing_key_from_index(i);
            let message = format!("test message {}", i).into_bytes();
            let mut signature = signing_key.sign(&message).to_bytes().to_vec();

            // Corrupt every other signature
            if i % 2 == 0 {
                signature[0] ^= 0xFF;
            }

            messages.push(message);
            signatures.push(signature);
            public_keys.push(signing_key.verifying_key().to_bytes().to_vec());
        }

        let result = verifier.verify_batch_parallel(&messages, &signatures, &public_keys)
            .expect("Verification failed");

        // Should have ~50% valid
        assert!(result.valid >= 20 && result.valid <= 30);
        assert!(result.invalid >= 20 && result.invalid <= 30);
    }
}
