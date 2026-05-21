//! `TipProofClient` — verifier-side counterpart to [`TipProofService`].
//!
//! [`TipProofService`]: crate::tip_proof_service::TipProofService
//!
//! A fresh node bootstrapping from `/api/v1/proof/tip`:
//!
//! ```text
//!     wallet / fresh node                ┌─ TipProofClient ──────────────┐
//!     ───────────────────                │                                │
//!                                        │  anchor_height                 │
//!     1. HTTP GET /proof/tip ──────────▶ │  anchor_state (hardcoded       │
//!                                        │     genesis trust root)        │
//!     2. body: bincode(LatticeTipProofV2)│                                │
//!                                        │  RwLock<Option<cached_proof>>  │
//!     3. client.ingest_bytes(body) ────▶ │  RwLock<TipProofClientStats>   │
//!                                        │                                │
//!     4. client.current_tip_height() ───▶│  → trustless chain tip         │
//!                                        └────────────────────────────────┘
//! ```
//!
//! # Trust model
//!
//! The client is anchored at a **verifier-hardcoded trust root** —
//! genesis `(0, [0u8; 32])` for fresh nodes, or a post-genesis
//! checkpoint for nodes that opted into a later anchor. Every ingested
//! proof must verify against this anchor; if it doesn't, the proof is
//! rejected and the cached proof remains unchanged.
//!
//! # Tip-height monotonicity
//!
//! The client refuses to accept a proof whose `tip_height` is below the
//! last-accepted proof's tip. This protects against a malicious upstream
//! serving an older proof (the kind of "rollback attack" classic
//! light-client literature spends pages on). The anti-rollback is
//! per-client-instance — if the wallet restarts and reloads from disk,
//! the cached last-accepted height needs to be persisted alongside.
//!
//! # What the client does NOT do (Phase B1 boundary)
//!
//! Per-step **cryptographic** verification (calling `LatticeStepFolder`'s
//! `verify_step` on each `delta_proofs[i]`) is gated behind the
//! caller's choice. The client's default `ingest_bytes` path runs
//! **structural** verification only ([`tip_verify_v2`]) — chain
//! integrity, anchor binding, monotonicity, version pinning. That's
//! the right default for wallets bootstrapping over flaky networks where
//! the per-step crypto cost would block the UI.
//!
//! Wallets that need crypto verification (paranoid mode, or post-Phase-C
//! when the cost drops to constant ≤10ms) call the
//! [`TipProofClient::ingest_proof_with_folder`] path instead.
//!
//! [`tip_verify_v2`]: crate::tip_verify_v2

use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

use q_ivc::recursion::LatticeStepFolder;

use crate::tip_proof_persistence::{self, TipProofPersistence};
use crate::tip_proof_v2::{
    self, LatticeTipProofV2, StateRoot, VerifyErrorV2, PROOF_VERSION,
};

// ════════════════════════════════════════════════════════════════════════════
// Errors
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Error)]
pub enum TipProofClientError {
    #[error("failed to deserialize proof: {0}")]
    Deserialize(#[from] bincode::Error),

    #[error("structural verification failed: {0}")]
    Verify(#[from] VerifyErrorV2),

    /// The newly-fetched proof's tip is at or below the cached proof's
    /// tip — possible rollback attack from a malicious upstream.
    #[error("tip-height monotonicity violated: fetched {fetched} <= cached {cached}")]
    RollbackRejected { fetched: u64, cached: u64 },
}

// ════════════════════════════════════════════════════════════════════════════
// Stats
// ════════════════════════════════════════════════════════════════════════════

/// Client-side counters. Snapshot via [`TipProofClient::stats`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TipProofClientStats {
    /// Total `ingest_*` calls (success + failure).
    pub total_proofs_fetched: u64,
    /// Proofs that passed verification + monotonicity.
    pub total_proofs_accepted: u64,
    /// Proofs that failed structural verification.
    pub total_proofs_rejected: u64,
    /// Proofs rejected by anti-rollback (tip ≤ cached tip).
    pub total_rollbacks_rejected: u64,
    /// Tip height of the most recently accepted proof. `None` if no
    /// proof has been accepted yet (fresh client state).
    pub last_accepted_tip_height: Option<u64>,
    /// Unix-epoch seconds at the last accepted proof.
    pub last_accepted_at_unix: Option<u64>,
    /// Stringified error from the most recent rejection.
    pub last_failure_reason: Option<String>,
}

impl TipProofClientStats {
    fn record_success(&mut self, tip_height: u64) {
        self.total_proofs_fetched += 1;
        self.total_proofs_accepted += 1;
        self.last_accepted_tip_height = Some(tip_height);
        self.last_accepted_at_unix = Some(now_unix());
        self.last_failure_reason = None;
    }

    fn record_failure(&mut self, reason: String) {
        self.total_proofs_fetched += 1;
        self.total_proofs_rejected += 1;
        self.last_failure_reason = Some(reason);
    }

    fn record_rollback(&mut self, reason: String) {
        self.total_proofs_fetched += 1;
        self.total_rollbacks_rejected += 1;
        self.last_failure_reason = Some(reason);
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ════════════════════════════════════════════════════════════════════════════
// Client
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
struct ClientState {
    cached: Option<LatticeTipProofV2>,
    stats: TipProofClientStats,
}

/// Thread-safe verifier-side wrapper. Clone is cheap (`Arc<...>`
/// internally); pass to UI handlers freely.
#[derive(Clone)]
pub struct TipProofClient {
    anchor_height: u64,
    anchor_state: StateRoot,
    state: Arc<RwLock<ClientState>>,
    persistence: Option<Arc<dyn TipProofPersistence>>,
}

impl TipProofClient {
    /// Construct a client anchored at the given trust root. No
    /// persistence — anti-rollback cache resets on every restart. Use
    /// [`Self::from_persistence`] for the production wallet path.
    pub fn new(anchor_height: u64, anchor_state: StateRoot) -> Self {
        Self {
            anchor_height,
            anchor_state,
            state: Arc::new(RwLock::new(ClientState {
                cached: None,
                stats: TipProofClientStats::default(),
            })),
            persistence: None,
        }
    }

    /// Anchor at genesis (the typical wallet bootstrap). Same as
    /// `TipProofClient::new(0, [0u8; 32])`.
    pub fn genesis() -> Self {
        Self::new(0, [0u8; 32])
    }

    /// Construct a client backed by `persistence`. On startup, if the
    /// backend has a saved proof whose anchor matches `anchor_height`
    /// + `anchor_state`, the cache is hydrated and anti-rollback
    /// monotonicity is enforced from that height across the restart.
    /// A mismatched anchor (typical after an upgrade-gate flip) is
    /// discarded with a warning.
    ///
    /// Every accepted proof is then saved through the backend so the
    /// cache survives the next restart.
    pub fn from_persistence(
        anchor_height: u64,
        anchor_state: StateRoot,
        persistence: Arc<dyn TipProofPersistence>,
    ) -> Self {
        let cached =
            match tip_proof_persistence::load_or_warn(&persistence, "TipProofClient") {
                Some(saved)
                    if saved.anchor_height == anchor_height
                        && saved.anchor_state == anchor_state =>
                {
                    info!(
                        "TipProofClient: restored cached proof at tip {}",
                        saved.tip_height
                    );
                    let mut stats = TipProofClientStats::default();
                    stats.last_accepted_tip_height = Some(saved.tip_height);
                    Some((saved, stats))
                }
                Some(saved) => {
                    warn!(
                        "TipProofClient: persisted proof anchor ({}, …) differs from configured anchor ({}, …) — discarding",
                        saved.anchor_height, anchor_height
                    );
                    None
                }
                None => None,
            };

        let (cached, stats) = match cached {
            Some((p, s)) => (Some(p), s),
            None => (None, TipProofClientStats::default()),
        };

        Self {
            anchor_height,
            anchor_state,
            state: Arc::new(RwLock::new(ClientState { cached, stats })),
            persistence: Some(persistence),
        }
    }

    /// The verifier-known anchor `(height, state)` this client is
    /// pinned against. Surfaced for the UI banner that displays
    /// "verified from height N against root XX...".
    pub fn anchor(&self) -> (u64, StateRoot) {
        (self.anchor_height, self.anchor_state)
    }

    /// Ingest serialized proof bytes (typically the body of
    /// `GET /api/v1/proof/tip`). Runs:
    ///
    ///   1. bincode deserialize
    ///   2. structural verify against the client's anchor
    ///   3. anti-rollback (new tip > cached tip)
    ///
    /// On success: caches the proof, updates stats, returns the new tip
    /// height. On failure: cached proof unchanged, failure recorded in
    /// stats.
    pub fn ingest_bytes(&self, bytes: &[u8]) -> Result<u64, TipProofClientError> {
        let proof: LatticeTipProofV2 = bincode::deserialize(bytes)
            .map_err(TipProofClientError::Deserialize)?;
        self.ingest_proof(proof)
    }

    /// Ingest an already-deserialized proof. Same semantics as
    /// [`Self::ingest_bytes`] minus the bincode hop.
    pub fn ingest_proof(&self, proof: LatticeTipProofV2) -> Result<u64, TipProofClientError> {
        // Structural verification first — cheap and defends against the
        // bulk of malformed inputs.
        if let Err(err) = tip_proof_v2::verify_chain_structure(
            &proof,
            self.anchor_height,
            self.anchor_state,
        ) {
            let mut guard = self.state.write().expect("RwLock poisoned");
            let reason = err.to_string();
            guard.stats.record_failure(reason);
            warn!("TipProofClient: structural verify failed: {}", err);
            return Err(TipProofClientError::Verify(err));
        }

        // Anti-rollback — refuse to downgrade the cached tip.
        {
            let guard = self.state.read().expect("RwLock poisoned");
            if let Some(ref cached) = guard.cached {
                if proof.tip_height <= cached.tip_height {
                    drop(guard);
                    let mut wg = self.state.write().expect("RwLock poisoned");
                    let reason = format!(
                        "rollback rejected: fetched tip {} <= cached tip {}",
                        proof.tip_height, cached.tip_height
                    );
                    wg.stats.record_rollback(reason);
                    warn!(
                        "TipProofClient: rejected rollback {} -> {}",
                        cached.tip_height, proof.tip_height
                    );
                    return Err(TipProofClientError::RollbackRejected {
                        fetched: proof.tip_height,
                        cached: cached.tip_height,
                    });
                }
            }
        }

        // Accept.
        let tip_height = proof.tip_height;
        let mut guard = self.state.write().expect("RwLock poisoned");
        guard.cached = Some(proof.clone());
        guard.stats.record_success(tip_height);
        drop(guard);

        // Persist outside the write lock so file I/O doesn't block
        // concurrent readers. Save failures are logged but do NOT
        // override the in-memory acceptance — the in-memory state IS
        // the authoritative cache for this process lifetime.
        if let Some(backend) = self.persistence.as_ref() {
            if let Err(e) = tip_proof_persistence::save_with_retry(backend, &proof) {
                warn!(
                    "TipProofClient: persistence save failed after accepting tip {} ({}): {}",
                    tip_height,
                    backend.backend_id(),
                    e
                );
            }
        }

        debug!("TipProofClient: accepted tip at height {}", tip_height);
        Ok(tip_height)
    }

    /// Ingest with full per-step crypto verification via a supplied
    /// `LatticeStepFolder`. Same flow as [`Self::ingest_proof`] but
    /// additionally calls
    /// [`verify_with_folder`](crate::tip_proof_v2::verify_with_folder)
    /// to crypto-verify each step.
    ///
    /// `circuit_builder` reconstructs the verification-side R1CS shape
    /// from `(z_in, z_out)` — see the
    /// [`verify_with_folder`](crate::tip_proof_v2::verify_with_folder)
    /// docstring for shape constraints.
    pub fn ingest_proof_with_folder<F, B, C>(
        &self,
        proof: LatticeTipProofV2,
        folder: Arc<LatticeStepFolder>,
        circuit_builder: B,
    ) -> Result<u64, TipProofClientError>
    where
        F: ark_ff::PrimeField,
        B: FnMut(q_ivc::recursion::StepIO, q_ivc::recursion::StepIO) -> C,
        C: ark_relations::r1cs::ConstraintSynthesizer<F>,
    {
        // Anti-rollback first (cheap, before crypto).
        {
            let guard = self.state.read().expect("RwLock poisoned");
            if let Some(ref cached) = guard.cached {
                if proof.tip_height <= cached.tip_height {
                    drop(guard);
                    let mut wg = self.state.write().expect("RwLock poisoned");
                    let reason = format!(
                        "rollback rejected: fetched tip {} <= cached tip {}",
                        proof.tip_height, cached.tip_height
                    );
                    wg.stats.record_rollback(reason);
                    return Err(TipProofClientError::RollbackRejected {
                        fetched: proof.tip_height,
                        cached: cached.tip_height,
                    });
                }
            }
        }

        // Full crypto verification — calls structure check internally.
        if let Err(err) = tip_proof_v2::verify_with_folder::<F, B, C>(
            &proof,
            self.anchor_height,
            self.anchor_state,
            folder,
            circuit_builder,
        ) {
            let mut guard = self.state.write().expect("RwLock poisoned");
            let reason = err.to_string();
            guard.stats.record_failure(reason);
            return Err(TipProofClientError::Verify(err));
        }

        let tip_height = proof.tip_height;
        let mut guard = self.state.write().expect("RwLock poisoned");
        guard.cached = Some(proof);
        guard.stats.record_success(tip_height);
        info!(
            "TipProofClient: accepted crypto-verified tip at height {}",
            tip_height
        );
        Ok(tip_height)
    }

    /// Current cached tip height. `None` if no proof has been accepted.
    pub fn current_tip_height(&self) -> Option<u64> {
        self.state
            .read()
            .expect("RwLock poisoned")
            .cached
            .as_ref()
            .map(|p| p.tip_height)
    }

    /// Current cached tip state root. `None` if no proof has been
    /// accepted.
    pub fn current_tip_state(&self) -> Option<StateRoot> {
        self.state
            .read()
            .expect("RwLock poisoned")
            .cached
            .as_ref()
            .map(|p| p.tip_state)
    }

    /// Current cached proof (clone). `None` if no proof has been
    /// accepted.
    pub fn current_proof(&self) -> Option<LatticeTipProofV2> {
        self.state
            .read()
            .expect("RwLock poisoned")
            .cached
            .clone()
    }

    /// Snapshot client stats.
    pub fn stats(&self) -> TipProofClientStats {
        self.state.read().expect("RwLock poisoned").stats.clone()
    }

    /// Discard the cached proof + reset stats. Used when the wallet
    /// changes anchors (upgrade-gate flip) or on explicit user reset.
    ///
    /// If persistence is configured, the backend is cleared.
    pub fn reset(&self) {
        let mut guard = self.state.write().expect("RwLock poisoned");
        guard.cached = None;
        guard.stats = TipProofClientStats::default();
        drop(guard);

        if let Some(backend) = self.persistence.as_ref() {
            if let Err(e) = backend.clear() {
                warn!(
                    "TipProofClient: persistence clear failed during reset ({}): {}",
                    backend.backend_id(),
                    e
                );
            }
        }
        info!("TipProofClient: cache + stats reset");
    }

    /// Persistence backend identifier (if configured). `None` means no
    /// persistence is active.
    pub fn persistence_backend_id(&self) -> Option<String> {
        self.persistence.as_ref().map(|b| b.backend_id().to_string())
    }

    /// Proof-version this client is configured to accept. Pinned to
    /// [`PROOF_VERSION`]; surfaced so the wallet UI can render the
    /// correct verifier banner.
    pub fn accepted_version(&self) -> &'static str {
        PROOF_VERSION
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use q_ivc::recursion::{LatticeStepProof, StepIO};
    use q_lattice_guard::{params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof};

    fn dummy_lattice_proof() -> LatticeGuardProof {
        LatticeGuardProof {
            commitments: Vec::new(),
            evaluations: (0, 0, 0),
            product_proofs: Vec::new(),
            transcript_state: [0u8; 32],
            metadata: ProofMetadata {
                num_constraints: 0,
                num_public_inputs: 0,
                security_level: SecurityLevel::PQ128,
                generation_time_ms: 0,
            },
        }
    }

    fn step(z_in: StepIO, z_out: StepIO) -> LatticeStepProof {
        LatticeStepProof {
            proof: dummy_lattice_proof(),
            z_in: z_in.pack(),
            z_out: z_out.pack(),
            public_input_count: 9,
        }
    }

    fn root(seed: u8) -> StateRoot {
        let mut r = [0u8; 32];
        for (i, b) in r.iter_mut().enumerate() {
            *b = (seed.wrapping_mul(i as u8 + 1)).wrapping_add(seed);
        }
        r
    }

    fn honest_proof_to_height(n: u64) -> LatticeTipProofV2 {
        let mut p = tip_proof_v2::anchor(0, [0u8; 32]);
        let mut prev = [0u8; 32];
        for h in 0..n {
            let next = root((h + 1) as u8);
            p = tip_proof_v2::extend_with_step_proof(
                &p,
                step(StepIO::new(prev, h), StepIO::new(next, h + 1)),
            )
            .expect("honest extend");
            prev = next;
        }
        p
    }

    #[test]
    fn fresh_client_has_no_cached_tip() {
        let c = TipProofClient::genesis();
        assert_eq!(c.current_tip_height(), None);
        assert_eq!(c.current_tip_state(), None);
        assert!(c.current_proof().is_none());

        let stats = c.stats();
        assert_eq!(stats.total_proofs_fetched, 0);
        assert_eq!(stats.total_proofs_accepted, 0);
    }

    #[test]
    fn ingest_honest_proof_caches_and_returns_tip_height() {
        let c = TipProofClient::genesis();
        let p = honest_proof_to_height(5);
        let tip = c.ingest_proof(p).expect("honest proof must verify");
        assert_eq!(tip, 5);
        assert_eq!(c.current_tip_height(), Some(5));
        let stats = c.stats();
        assert_eq!(stats.total_proofs_accepted, 1);
        assert_eq!(stats.total_proofs_rejected, 0);
        assert_eq!(stats.last_accepted_tip_height, Some(5));
    }

    #[test]
    fn ingest_bytes_round_trip() {
        let c = TipProofClient::genesis();
        let p = honest_proof_to_height(3);
        let bytes = bincode::serialize(&p).expect("serialise");
        let tip = c.ingest_bytes(&bytes).expect("ingest bytes");
        assert_eq!(tip, 3);
        assert_eq!(c.current_tip_height(), Some(3));
    }

    #[test]
    fn ingest_malformed_bytes_rejected_without_corrupting_cache() {
        let c = TipProofClient::genesis();
        c.ingest_proof(honest_proof_to_height(2)).expect("seed");

        let bad_bytes = b"not a valid bincode payload";
        let err = c.ingest_bytes(bad_bytes).unwrap_err();
        assert!(matches!(err, TipProofClientError::Deserialize(_)));

        // Cache unchanged.
        assert_eq!(c.current_tip_height(), Some(2));
        let stats = c.stats();
        assert_eq!(stats.total_proofs_rejected, 1);
        assert_eq!(stats.total_proofs_accepted, 1);
    }

    #[test]
    fn ingest_proof_against_wrong_anchor_rejected() {
        // Client anchored at a different state — honest genesis proof
        // doesn't verify against it.
        let c = TipProofClient::new(0, [0xAB; 32]);
        let p = honest_proof_to_height(3);
        let err = c.ingest_proof(p).unwrap_err();
        assert!(matches!(
            err,
            TipProofClientError::Verify(VerifyErrorV2::AnchorStateMismatch)
        ));
        assert_eq!(c.current_tip_height(), None);
    }

    #[test]
    fn rollback_attack_rejected() {
        // Client accepts proof to height 10; subsequent proof at lower
        // height is rejected (anti-rollback).
        let c = TipProofClient::genesis();
        let high = honest_proof_to_height(10);
        c.ingest_proof(high).expect("accept high tip");
        assert_eq!(c.current_tip_height(), Some(10));

        let low = honest_proof_to_height(5);
        let err = c.ingest_proof(low).unwrap_err();
        assert!(matches!(
            err,
            TipProofClientError::RollbackRejected { fetched: 5, cached: 10 }
        ));
        // Cache unchanged.
        assert_eq!(c.current_tip_height(), Some(10));

        let stats = c.stats();
        assert_eq!(stats.total_rollbacks_rejected, 1);
        assert_eq!(stats.total_proofs_accepted, 1);
    }

    #[test]
    fn equal_tip_height_is_also_rolled_back() {
        // tip_height == cached_tip is treated as rollback (no new info).
        let c = TipProofClient::genesis();
        let p1 = honest_proof_to_height(5);
        c.ingest_proof(p1.clone()).expect("accept");
        let err = c.ingest_proof(p1).unwrap_err();
        assert!(matches!(
            err,
            TipProofClientError::RollbackRejected { fetched: 5, cached: 5 }
        ));
    }

    #[test]
    fn monotonic_extends_accepted() {
        let c = TipProofClient::genesis();
        for tip in [1, 3, 5, 7, 10, 50, 100, 200] {
            let p = honest_proof_to_height(tip);
            let h = c.ingest_proof(p).expect("monotonic ingest");
            assert_eq!(h, tip);
            assert_eq!(c.current_tip_height(), Some(tip));
        }
        let stats = c.stats();
        assert_eq!(stats.total_proofs_accepted, 8);
        assert_eq!(stats.total_rollbacks_rejected, 0);
    }

    #[test]
    fn reset_clears_cache_and_stats() {
        let c = TipProofClient::genesis();
        c.ingest_proof(honest_proof_to_height(3)).expect("seed");
        c.reset();
        assert_eq!(c.current_tip_height(), None);
        let stats = c.stats();
        assert_eq!(stats.total_proofs_accepted, 0);
        assert_eq!(stats.total_proofs_fetched, 0);
    }

    #[test]
    fn accepted_version_is_pinned() {
        let c = TipProofClient::genesis();
        assert_eq!(c.accepted_version(), "latticeguard-rlwe-v1");
    }

    #[test]
    fn client_clone_shares_state() {
        let a = TipProofClient::genesis();
        let b = a.clone();
        a.ingest_proof(honest_proof_to_height(3)).expect("ingest via a");
        assert_eq!(b.current_tip_height(), Some(3));
    }

    #[test]
    fn anchor_getter_returns_constructor_values() {
        let c = TipProofClient::new(7777, [0xCD; 32]);
        assert_eq!(c.anchor(), (7777, [0xCD; 32]));
    }

    #[test]
    fn stats_record_success_clears_failure_reason() {
        let mut s = TipProofClientStats::default();
        s.record_failure("bad".to_string());
        s.record_success(5);
        assert!(s.last_failure_reason.is_none());
        assert_eq!(s.total_proofs_fetched, 2);
        assert_eq!(s.total_proofs_accepted, 1);
        assert_eq!(s.total_proofs_rejected, 1);
    }
}
