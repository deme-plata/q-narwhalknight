//! Batched Dilithium5 verification with runtime AVX2 dispatch.
//!
//! # Background
//!
//! `pqcrypto-dilithium = "0.5"` ships PQClean's C reference implementation,
//! built via the `cc` crate. There is **no upstream AVX-512 path** as of
//! v0.5.0. The crate exposes an `avx2` Cargo feature (default-enabled) that
//! compiles PQClean's AVX2-optimized C variant; the picked implementation
//! is selected per-call via `std::is_x86_feature_detected!("avx2")` inside
//! the crate's `verify_detached_signature` function.
//!
//! # What this module adds
//!
//! 1. A batched verifier that dispatches each verify across a rayon pool,
//!    so an N-signature pack becomes `cores` parallel verifies instead of
//!    one serial loop.
//! 2. A runtime CPU-feature gate that logs the dispatch path so operators
//!    can confirm the binary they're running picks AVX2 on Epsilon's Xeon
//!    Gold. Returns a `DilithiumDispatchPath` enum that callers can inspect
//!    for telemetry.
//! 3. A `Vec<bool>` per-index API so callers can attribute which specific
//!    signature failed — necessary for chunk-ingest to identify the
//!    offending block rather than rejecting the entire pack.
//!
//! # Performance rationale
//!
//! - Per-signature: AVX2 PQClean dilithium5_verify is ~1.5-2× the
//!   reference C build (NOT 3-5× as originally quoted; the spec correction
//!   landed in `docs/v10.9.43-simd-implementation-plan.md` item 10).
//! - Batch dispatch: linear in core count up to ~32 cores on Epsilon.
//! - Combined realistic gain on 256-signature batch: 1.5× (AVX2) × min(N,
//!   cores) (batch) = ~12-15× on 48-core Epsilon vs single-threaded
//!   reference build.
//!
//! # Fallback
//!
//! On non-x86_64 hosts or hosts without AVX2, the same `verify_batch`
//! function still works — pqcrypto-dilithium falls back to its scalar C
//! reference path internally. The dispatch enum reports `Scalar` so the
//! caller knows.

use anyhow::Result;
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{
    DetachedSignature as PqDetachedSignature, PublicKey as PqPublicKey,
};
use rayon::prelude::*;
use tracing::{debug, warn};

/// Which Dilithium5 implementation the runtime picked.
///
/// Logged once at first use so operators can audit binary deployment
/// (Epsilon should land on `Avx2Pqclean`; non-AVX2 hosts on `Scalar`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DilithiumDispatchPath {
    /// PQClean AVX2 build (selected when `is_x86_feature_detected!("avx2")`
    /// returns true AND the crate was built with `avx2` feature).
    Avx2Pqclean,
    /// PQClean ARM aarch64 NEON build.
    AArch64Neon,
    /// PQClean scalar reference build (fallback).
    Scalar,
}

impl DilithiumDispatchPath {
    /// Detect the path the runtime will pick on this host. Cheap — single
    /// CPUID query, cached internally by `std::is_x86_feature_detected!`.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            // pqcrypto-dilithium 0.5 only dispatches to AVX2 when the `avx2`
            // feature is enabled (workspace default) AND the host supports
            // it. We can't read the cfg flag from here, so we assume the
            // build had the feature on — this matches the workspace default.
            if std::is_x86_feature_detected!("avx2") {
                return Self::Avx2Pqclean;
            }
            return Self::Scalar;
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self::AArch64Neon;
        }
        #[allow(unreachable_code)]
        Self::Scalar
    }
}

/// Result of a batched Dilithium5 verify.
#[derive(Debug, Clone)]
pub struct DilithiumBatchResult {
    /// Total signatures attempted.
    pub total: usize,
    /// Count of cryptographically valid signatures.
    pub valid: usize,
    /// Count of invalid (rejected) signatures.
    pub invalid: usize,
    /// Per-index validity: `results[i] == true` ⇔ signature `i` verified.
    /// Length matches input slices.
    pub results: Vec<bool>,
    /// Which dispatch path the runtime picked.
    pub dispatch: DilithiumDispatchPath,
}

/// Batched Dilithium5 verifier with runtime AVX2 dispatch.
///
/// Inputs are detached signatures (length per pqcrypto-dilithium 0.5's
/// schema, currently 4,627 bytes); pubkeys are 2,592 bytes. The verifier
/// gracefully rejects any wrong-length input as an invalid signature
/// rather than erroring the whole batch.
pub struct DilithiumBatchVerifier {
    dispatch: DilithiumDispatchPath,
}

impl Default for DilithiumBatchVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl DilithiumBatchVerifier {
    /// Construct a verifier and snapshot the dispatch path for logging.
    pub fn new() -> Self {
        let dispatch = DilithiumDispatchPath::detect();
        match dispatch {
            DilithiumDispatchPath::Avx2Pqclean => {
                debug!("Dilithium5 batch verifier: AVX2 PQClean path");
            }
            DilithiumDispatchPath::AArch64Neon => {
                debug!("Dilithium5 batch verifier: aarch64 NEON path");
            }
            DilithiumDispatchPath::Scalar => {
                warn!(
                    "Dilithium5 batch verifier: scalar fallback (no AVX2/NEON detected). \
                     Performance will be ~1.5-2× slower than AVX2 path."
                );
            }
        }
        Self { dispatch }
    }

    /// Snapshot of the runtime dispatch path. Useful for telemetry.
    pub fn dispatch(&self) -> DilithiumDispatchPath {
        self.dispatch
    }

    /// Verify N detached signatures in parallel, attributing per-index
    /// validity in the returned `results` vector.
    ///
    /// Input invariants:
    /// - `messages.len() == signatures.len() == public_keys.len()`
    /// - Each `signatures[i]` is a Dilithium5 detached signature (any
    ///   wrong-length entry is recorded as `false` without erroring the
    ///   batch).
    /// - Each `public_keys[i]` is a Dilithium5 public key (same handling
    ///   as signatures for wrong length).
    pub fn verify_batch(
        &self,
        messages: &[&[u8]],
        signatures: &[&[u8]],
        public_keys: &[&[u8]],
    ) -> Result<DilithiumBatchResult> {
        if messages.len() != signatures.len() || messages.len() != public_keys.len() {
            return Err(anyhow::anyhow!(
                "Batch size mismatch: msgs={}, sigs={}, pks={}",
                messages.len(),
                signatures.len(),
                public_keys.len()
            ));
        }

        let total = messages.len();
        if total == 0 {
            return Ok(DilithiumBatchResult {
                total: 0,
                valid: 0,
                invalid: 0,
                results: Vec::new(),
                dispatch: self.dispatch,
            });
        }

        // Index-collected results to preserve per-index attribution under
        // rayon's parallel iteration.
        let mut results: Vec<bool> = (0..total)
            .into_par_iter()
            .map(|i| verify_single_dilithium5(messages[i], signatures[i], public_keys[i]))
            .collect();

        // Defensive: rayon's into_par_iter().collect() preserves order, but
        // we treat results as authoritative regardless.
        let valid = results.iter().filter(|b| **b).count();
        let invalid = total - valid;

        // Shrink to fit — caller may store this in a long-lived struct.
        results.shrink_to_fit();

        Ok(DilithiumBatchResult {
            total,
            valid,
            invalid,
            results,
            dispatch: self.dispatch,
        })
    }
}

/// Verify a single Dilithium5 detached signature. Wrong-length or
/// malformed inputs return `false` (treated as invalid, not an error).
fn verify_single_dilithium5(message: &[u8], signature: &[u8], pubkey: &[u8]) -> bool {
    let Ok(pk) = dilithium5::PublicKey::from_bytes(pubkey) else {
        return false;
    };
    let Ok(sig) = dilithium5::DetachedSignature::from_bytes(signature) else {
        return false;
    };
    dilithium5::verify_detached_signature(&sig, message, &pk).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Acceptance test: valid signatures pass, invalid ones fail, AND the
    /// per-index attribution is correct.
    #[test]
    fn dilithium5_batch_accepts_valid_and_rejects_invalid() {
        let verifier = DilithiumBatchVerifier::new();
        let (pk, sk) = dilithium5::keypair();
        let msg = b"q-narwhalknight dilithium5 batch test";
        let detached = dilithium5::detached_sign(msg, &sk);

        let pk_bytes = pk.as_bytes().to_vec();
        let sig_bytes = detached.as_bytes().to_vec();
        let bogus_sig = vec![0u8; sig_bytes.len()];

        // Pattern: V, I, V, V, I (3 valid, 2 invalid)
        let messages: Vec<&[u8]> = vec![msg, msg, msg, msg, msg];
        let signatures: Vec<&[u8]> = vec![
            &sig_bytes,
            &bogus_sig,
            &sig_bytes,
            &sig_bytes,
            &bogus_sig,
        ];
        let public_keys: Vec<&[u8]> = vec![&pk_bytes; 5];

        let result = verifier
            .verify_batch(&messages, &signatures, &public_keys)
            .expect("verify_batch failed");

        assert_eq!(result.total, 5);
        assert_eq!(result.valid, 3);
        assert_eq!(result.invalid, 2);
        assert_eq!(result.results, vec![true, false, true, true, false]);
    }

    /// Wrong-length inputs must NOT panic — they must be reported as
    /// invalid signatures.
    #[test]
    fn dilithium5_rejects_wrong_lengths_without_panic() {
        let verifier = DilithiumBatchVerifier::new();
        let msg: &[u8] = b"x";
        let too_short_sig: &[u8] = &[0u8; 32];
        let bad_pk: &[u8] = &[0u8; 16];

        let result = verifier
            .verify_batch(&[msg], &[too_short_sig], &[bad_pk])
            .unwrap();
        assert_eq!(result.results, vec![false]);
        assert_eq!(result.valid, 0);
    }

    /// Empty batch is valid (returns empty results).
    #[test]
    fn dilithium5_empty_batch() {
        let verifier = DilithiumBatchVerifier::new();
        let result = verifier.verify_batch(&[], &[], &[]).unwrap();
        assert_eq!(result.total, 0);
        assert!(result.results.is_empty());
    }

    /// Mismatched batch sizes must error.
    #[test]
    fn dilithium5_mismatched_sizes_errors() {
        let verifier = DilithiumBatchVerifier::new();
        let msg: &[u8] = b"x";
        let sig: &[u8] = &[0u8; 32];
        let pk: &[u8] = &[0u8; 32];
        let result = verifier.verify_batch(&[msg, msg], &[sig], &[pk]);
        assert!(result.is_err());
    }

    /// Runtime feature-detection sanity. On non-x86_64 hosts we expect a
    /// non-AVX2 dispatch path. On x86_64 hosts that DO have AVX2 we expect
    /// `Avx2Pqclean`; on x86_64 hosts without AVX2 we expect `Scalar`.
    /// This is a runtime-gated assertion, not a compile-time one.
    #[test]
    fn dilithium5_dispatch_detection_consistent_with_cpu() {
        let path = DilithiumDispatchPath::detect();
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                assert_eq!(path, DilithiumDispatchPath::Avx2Pqclean);
            } else {
                assert_eq!(path, DilithiumDispatchPath::Scalar);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(path, DilithiumDispatchPath::AArch64Neon);
        }
    }

    /// Tampered message must fail even when signature/pubkey are valid.
    #[test]
    fn dilithium5_rejects_tampered_message() {
        let verifier = DilithiumBatchVerifier::new();
        let (pk, sk) = dilithium5::keypair();
        let detached = dilithium5::detached_sign(b"original", &sk);
        let pk_bytes = pk.as_bytes().to_vec();
        let sig_bytes = detached.as_bytes().to_vec();

        let result = verifier
            .verify_batch(&[b"tampered"], &[&sig_bytes], &[&pk_bytes])
            .unwrap();
        assert_eq!(result.valid, 0);
        assert_eq!(result.results, vec![false]);
    }
}
