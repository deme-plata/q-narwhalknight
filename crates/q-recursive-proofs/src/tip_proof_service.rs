//! `TipProofService` — production wrapper around the Phase B1 stack.
//!
//! Wraps a [`LatticeStepFolder`](q_ivc::recursion::LatticeStepFolder) and an
//! evolving [`LatticeTipProofV2`](crate::tip_proof_v2::LatticeTipProofV2)
//! behind a thread-safe API the q-api-server can call directly:
//!
//! ```text
//!  /api/v1/proof/tip handler           ┌─ TipProofService ─────────────┐
//!     │                                 │                                │
//!     ▼                                 │  ┌──────────────────────┐     │
//!  service.current_proof_bincode() ────▶│  │ RwLock<state>        │     │
//!                                       │  │   .proof: V2          │     │
//!  block-producer task                  │  │   .stats: Stats       │     │
//!     │                                 │  └──────────────────────┘     │
//!     ▼ step_proof                       │                                │
//!  service.extend(step_proof) ─────────▶│  Arc<LatticeStepFolder>       │
//!                                       └────────────────────────────────┘
//! ```
//!
//! # Why a service layer
//!
//! - **Synchronisation:** multiple readers (API server's `/proof/tip`
//!   handler runs once per request) + at-most-one writer (the block-producer
//!   task or the IVC fold worker). `RwLock` is the right primitive.
//! - **Observability:** [`TipProofServiceStats`] captures rate, last-failure
//!   reason, and chain-length progression — surfaceable via Prometheus,
//!   logged on each `extend()`, embedded in the `/proof/tip` response
//!   header.
//! - **Serialization shape stability:** the API server doesn't want to
//!   know about `bincode` versions. [`current_proof_bytes`] returns the
//!   canonical wire format and pins serializer choice in one place.
//! - **Failure mode:** extends can fail (chain discontinuity, height jump);
//!   the service records the failure but keeps the existing tip available
//!   so reads continue to serve the last-known-good proof rather than 500ing.
//! - **Anchor reset:** at upgrade-gate flips, the verifier-known trust root
//!   changes. [`reset_anchor`] discards the proof and starts a fresh chain
//!   from the new anchor — used during the `Upgrade::TipProofV2` activation
//!   window.
//!
//! # Phase boundary
//!
//! Phase B1 chain is linear (proof size grows with chain length).
//! [`TipProofServiceConfig::max_steps_retained`] is the **backpressure
//! valve** during the advisory window: keep at most N step proofs to
//! cap memory. Phase C's Module-SIS folder makes this obsolete by
//! storing a single constant-size accumulator.

use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use bincode;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use q_ivc::recursion::LatticeStepProof;

use crate::tip_proof_persistence::{
    self, TipProofPersistence,
};
use crate::tip_proof_v2::{
    self, LatticeTipProofV2, StateRoot, VerifyErrorV2, PROOF_VERSION,
};

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

/// Tunables for [`TipProofService`].
#[derive(Clone, Debug)]
pub struct TipProofServiceConfig {
    /// Initial anchor — the verifier-known trust root every fresh node
    /// expects. Genesis is `(0, [0u8; 32])`.
    pub anchor_height: u64,
    pub anchor_state: StateRoot,

    /// Cap on retained step proofs. `None` means unlimited; `Some(N)`
    /// drops the **oldest** steps when the chain grows past N, advancing
    /// the anchor to match. Used during the Phase B1 advisory window to
    /// keep server memory bounded.
    ///
    /// **Soundness implication:** the dropped-prefix region is no longer
    /// proven to fresh nodes — they trust the new advanced anchor like a
    /// checkpoint. Set to `None` for genuine genesis-bootstrap; `Some(N)`
    /// is an operational backpressure valve, NOT a soundness primitive.
    pub max_steps_retained: Option<usize>,

    /// Whether to log a TRACE-level line for every successful extend.
    /// Default false (low-cardinality production logging).
    pub trace_each_extend: bool,
}

impl TipProofServiceConfig {
    /// Genesis-anchored service with unbounded retention.
    pub fn genesis() -> Self {
        Self {
            anchor_height: 0,
            anchor_state: [0u8; 32],
            max_steps_retained: None,
            trace_each_extend: false,
        }
    }

    /// Anchor at a specific checkpoint instead of genesis. Used when
    /// the verifier's trust root is a hardcoded post-genesis state.
    pub fn anchored_at(height: u64, state: StateRoot) -> Self {
        Self {
            anchor_height: height,
            anchor_state: state,
            max_steps_retained: None,
            trace_each_extend: false,
        }
    }

    /// Cap retained steps. See `max_steps_retained` docstring for the
    /// soundness caveat.
    pub fn with_max_steps_retained(mut self, n: usize) -> Self {
        self.max_steps_retained = Some(n);
        self
    }

    /// Enable per-extend trace logging.
    pub fn with_trace_each_extend(mut self) -> Self {
        self.trace_each_extend = true;
        self
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Observability
// ════════════════════════════════════════════════════════════════════════════

/// Service-level counters + last-known status. Snapshot via
/// [`TipProofService::stats`]; suitable for Prometheus export.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TipProofServiceStats {
    /// Total number of `extend` calls (success + failure).
    pub total_extends_attempted: u64,
    /// `extend` calls that produced a new tip.
    pub total_extends_succeeded: u64,
    /// `extend` calls rejected by chain integrity (caller's mistake or
    /// adversarial input).
    pub total_extends_rejected: u64,
    /// Number of times an anchor reset happened (upgrade-gate flips,
    /// max-retention prefix-drop, explicit ops intervention).
    pub total_anchor_resets: u64,
    /// Number of times the max-retention prefix-drop fired.
    pub total_prefix_drops: u64,
    /// Unix-epoch seconds at the time of the last successful extend.
    /// `None` if no extends have succeeded since the last reset.
    pub last_extend_at_unix: Option<u64>,
    /// Stringified `VerifyErrorV2` from the most recent rejected extend.
    pub last_failure_reason: Option<String>,
    /// Current chain length (number of step proofs in `delta_proofs`).
    pub current_step_count: usize,
    /// Current chain tip height (matches `proof.tip_height`).
    pub current_tip_height: u64,
}

impl TipProofServiceStats {
    fn record_success(&mut self, new_tip_height: u64, new_step_count: usize) {
        self.total_extends_attempted += 1;
        self.total_extends_succeeded += 1;
        self.current_tip_height = new_tip_height;
        self.current_step_count = new_step_count;
        self.last_extend_at_unix = Some(now_unix());
        self.last_failure_reason = None;
    }

    fn record_failure(&mut self, reason: String) {
        self.total_extends_attempted += 1;
        self.total_extends_rejected += 1;
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
// Service state
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
struct ServiceState {
    proof: LatticeTipProofV2,
    stats: TipProofServiceStats,
}

// ════════════════════════════════════════════════════════════════════════════
// TipProofService
// ════════════════════════════════════════════════════════════════════════════

/// Thread-safe service wrapping the Phase B1 tip-proof stack.
///
/// Clone is cheap (`Arc<...>` internally); pass to handlers freely.
///
/// **By design, the service does NOT own a `LatticeStepFolder`** — block
/// producers (or proof-generation workers) own the folder, generate step
/// proofs themselves, and hand the results to [`Self::extend`]. This split
/// lets the service be Sync without paying SRS-generation cost upfront,
/// keeps unit tests fast (no SRS dependency), and means proving infra
/// can scale horizontally (many folders → one service).
#[derive(Clone)]
pub struct TipProofService {
    state: Arc<RwLock<ServiceState>>,
    config: Arc<TipProofServiceConfig>,
    /// Optional pluggable persistence — if `Some`, the service hydrates
    /// on construction from `from_persistence` and saves on every
    /// successful `extend` / `reset_anchor`.
    persistence: Option<Arc<dyn TipProofPersistence>>,
}

impl TipProofService {
    /// Construct a service anchored at the configured trust root with
    /// an empty (anchor-only) tip proof. No persistence — state is lost
    /// on restart. Use [`Self::from_persistence`] for production.
    pub fn new(config: TipProofServiceConfig) -> Self {
        let proof = tip_proof_v2::anchor(config.anchor_height, config.anchor_state);
        let mut stats = TipProofServiceStats::default();
        stats.current_tip_height = config.anchor_height;
        Self {
            state: Arc::new(RwLock::new(ServiceState { proof, stats })),
            config: Arc::new(config),
            persistence: None,
        }
    }

    /// Construct a service backed by `persistence`. On startup:
    ///   - If the backend has a saved proof whose anchor matches
    ///     `config.anchor_height` + `config.anchor_state`, restore it.
    ///   - If the backend's saved proof has a DIFFERENT anchor (typical
    ///     after an upgrade-gate flip with stale on-disk state), discard
    ///     the saved proof and start fresh from the config's anchor —
    ///     this is the safe default; the operator can call `reset_anchor`
    ///     explicitly to make the discard observable in metrics.
    ///   - If the backend is empty, start anchor-only.
    ///
    /// In all cases the persistence backend is retained for subsequent
    /// saves.
    pub fn from_persistence(
        config: TipProofServiceConfig,
        persistence: Arc<dyn TipProofPersistence>,
    ) -> Self {
        let proof = match tip_proof_persistence::load_or_warn(&persistence, "TipProofService") {
            Some(saved) if saved.anchor_height == config.anchor_height
                && saved.anchor_state == config.anchor_state =>
            {
                info!(
                    "TipProofService: restored chain at tip {}",
                    saved.tip_height
                );
                saved
            }
            Some(saved) => {
                warn!(
                    "TipProofService: persisted proof anchor ({}, …) differs from configured anchor ({}, …) — starting fresh",
                    saved.anchor_height, config.anchor_height
                );
                tip_proof_v2::anchor(config.anchor_height, config.anchor_state)
            }
            None => tip_proof_v2::anchor(config.anchor_height, config.anchor_state),
        };
        let mut stats = TipProofServiceStats::default();
        stats.current_tip_height = proof.tip_height;
        stats.current_step_count = proof.delta_proofs.len();

        Self {
            state: Arc::new(RwLock::new(ServiceState { proof, stats })),
            config: Arc::new(config),
            persistence: Some(persistence),
        }
    }

    /// Snapshot the current proof. Cheap clone (proof shape is `Vec`-backed
    /// but the Vec elements are bincode-friendly proofs).
    pub fn current_proof(&self) -> LatticeTipProofV2 {
        self.state.read().expect("RwLock poisoned").proof.clone()
    }

    /// Snapshot the current proof as canonical wire bytes (bincode).
    /// Used by the q-api-server `/api/v1/proof/tip` handler — returns
    /// `application/octet-stream`.
    pub fn current_proof_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        let proof = self.current_proof();
        bincode::serialize(&proof)
    }

    /// Approximate on-wire size of the current proof in bytes — `O(1)`
    /// header + linear delta-proofs size. Used for telemetry and `Content-Length`
    /// estimation without paying the serialisation cost.
    pub fn current_proof_size_estimate(&self) -> usize {
        let proof = self.current_proof();
        approximate_wire_size(&proof)
    }

    /// Snapshot service stats. Cheap (clones a small struct).
    pub fn stats(&self) -> TipProofServiceStats {
        self.state.read().expect("RwLock poisoned").stats.clone()
    }

    /// Service configuration (Arc-shared, never mutates after construction).
    pub fn config(&self) -> Arc<TipProofServiceConfig> {
        Arc::clone(&self.config)
    }

    /// Attempt to extend the tip by one step. On success, returns a
    /// snapshot of the post-extend stats; on failure, the existing tip
    /// is untouched and the failure is recorded in stats.
    ///
    /// If a persistence backend is configured, the post-extend proof is
    /// saved asynchronously after the write lock is released — a save
    /// failure is logged but does NOT mark the extend as failed (the
    /// in-memory state is already valid; the next successful save will
    /// catch the persistence layer up).
    pub fn extend(
        &self,
        step_proof: LatticeStepProof,
    ) -> Result<TipProofServiceStats, VerifyErrorV2> {
        let mut guard = self.state.write().expect("RwLock poisoned");

        let result = tip_proof_v2::extend_with_step_proof(&guard.proof, step_proof);
        let (stats_snapshot, proof_to_persist) = match result {
            Ok(mut new_proof) => {
                // Apply max-steps-retained backpressure if configured.
                let mut prefix_dropped = false;
                if let Some(cap) = self.config.max_steps_retained {
                    if new_proof.delta_proofs.len() > cap {
                        let to_drop = new_proof.delta_proofs.len() - cap;
                        prefix_dropped = true;
                        new_proof = drop_prefix(new_proof, to_drop);
                    }
                }

                let new_tip = new_proof.tip_height;
                let new_step_count = new_proof.delta_proofs.len();

                guard.proof = new_proof;
                guard.stats.record_success(new_tip, new_step_count);

                if prefix_dropped {
                    guard.stats.total_prefix_drops += 1;
                    guard.stats.total_anchor_resets += 1;
                    warn!(
                        "TipProofService: max_steps_retained={} hit, dropped prefix, new anchor at height {}",
                        self.config.max_steps_retained.unwrap_or(0),
                        guard.proof.anchor_height
                    );
                }

                if self.config.trace_each_extend {
                    debug!(
                        "TipProofService: extended to height={} steps={}",
                        new_tip, new_step_count
                    );
                }

                let snapshot = guard.stats.clone();
                let proof_clone = self.persistence.as_ref().map(|_| guard.proof.clone());
                (Ok(snapshot), proof_clone)
            }
            Err(err) => {
                let reason = err.to_string();
                guard.stats.record_failure(reason);
                warn!("TipProofService: extend rejected: {}", err);
                (Err(err), None)
            }
        };

        // Drop the write lock BEFORE persisting — file I/O can take
        // milliseconds and we don't want to block readers.
        drop(guard);

        if let (Some(backend), Some(proof)) = (self.persistence.as_ref(), proof_to_persist) {
            if let Err(e) = tip_proof_persistence::save_with_retry(backend, &proof) {
                warn!(
                    "TipProofService: persistence save failed after successful extend ({}): {}",
                    backend.backend_id(),
                    e
                );
            }
        }

        stats_snapshot
    }

    /// Reset the chain to a new anchor, discarding all step proofs.
    /// Used during upgrade-gate activation when the verifier's trust
    /// root changes.
    ///
    /// If persistence is configured, the backend is cleared (subsequent
    /// extends will save fresh state). A clear failure is logged but
    /// the in-memory reset still takes effect — the next successful
    /// extend's save will overwrite stale on-disk state.
    pub fn reset_anchor(&self, new_anchor_height: u64, new_anchor_state: StateRoot) {
        let mut guard = self.state.write().expect("RwLock poisoned");
        guard.proof = tip_proof_v2::anchor(new_anchor_height, new_anchor_state);
        guard.stats.current_step_count = 0;
        guard.stats.current_tip_height = new_anchor_height;
        guard.stats.last_extend_at_unix = None;
        guard.stats.last_failure_reason = None;
        guard.stats.total_anchor_resets += 1;
        info!(
            "TipProofService: anchor reset to height={}",
            new_anchor_height
        );
        drop(guard);

        if let Some(backend) = self.persistence.as_ref() {
            if let Err(e) = backend.clear() {
                warn!(
                    "TipProofService: persistence clear failed after reset_anchor ({}): {}",
                    backend.backend_id(),
                    e
                );
            }
        }
    }

    /// Persistence backend identifier (if configured). `None` means no
    /// persistence is active. Surfaced for diagnostics + Prometheus
    /// label rendering.
    pub fn persistence_backend_id(&self) -> Option<String> {
        self.persistence.as_ref().map(|b| b.backend_id().to_string())
    }

    /// Self-check: verify the service's chain is internally consistent
    /// against its own anchor. Useful for periodic integrity audits and
    /// post-extend smoke tests.
    pub fn verify_chain_self_consistent(&self) -> Result<(), VerifyErrorV2> {
        let guard = self.state.read().expect("RwLock poisoned");
        tip_proof_v2::verify_chain_structure(
            &guard.proof,
            guard.proof.anchor_height,
            guard.proof.anchor_state,
        )
    }

    /// Current chain length. Cheap (read-lock + integer load).
    pub fn step_count(&self) -> usize {
        self.state.read().expect("RwLock poisoned").proof.delta_proofs.len()
    }

    /// Current tip height. Cheap.
    pub fn tip_height(&self) -> u64 {
        self.state.read().expect("RwLock poisoned").proof.tip_height
    }

    /// Current anchor `(height, state)`. Cheap.
    pub fn anchor(&self) -> (u64, StateRoot) {
        let g = self.state.read().expect("RwLock poisoned");
        (g.proof.anchor_height, g.proof.anchor_state)
    }

    /// Proof-version string. Pinned to [`PROOF_VERSION`]; surfaced so
    /// the API server can echo it in response headers without taking
    /// a dep on `tip_proof_v2`.
    pub fn proof_version(&self) -> &'static str {
        PROOF_VERSION
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

/// Drop the first `n` step proofs from the chain, advancing the anchor
/// to the first retained step's `z_in`. Used by the max-steps-retained
/// backpressure valve.
///
/// Invariant: the resulting proof is still self-consistent under
/// [`verify_chain_structure`](crate::tip_proof_v2::verify_chain_structure)
/// against its new anchor.
fn drop_prefix(mut proof: LatticeTipProofV2, n: usize) -> LatticeTipProofV2 {
    if n == 0 || n >= proof.delta_proofs.len() {
        return proof;
    }
    let new_anchor = proof.delta_proofs[n].z_in_unpacked();
    proof.delta_proofs.drain(..n);
    proof.anchor_height = new_anchor.height;
    proof.anchor_state = new_anchor.state_root;
    proof
}

/// Cheap estimate of `bincode::serialize(&proof).unwrap().len()` without
/// actually serialising. Used for `Content-Length` headers and telemetry.
///
/// Per-step is dominated by `LatticeGuardProof` payloads. At pq128 each
/// proof is ~10-50 KB depending on commitment count; we conservatively
/// model 30 KB / step. Header is fixed-shape.
fn approximate_wire_size(proof: &LatticeTipProofV2) -> usize {
    const PER_STEP_BYTES: usize = 30 * 1024;
    const HEADER_BYTES: usize =
        32 /* anchor_state */ + 8 /* anchor_height */
            + 32 /* tip_state */ + 8 /* tip_height */
            + 32 /* version string typical len + framing */;
    HEADER_BYTES + proof.delta_proofs.len() * PER_STEP_BYTES
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

    #[test]
    fn service_genesis_then_extend_and_stats() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());

        assert_eq!(service.tip_height(), 0);
        assert_eq!(service.step_count(), 0);
        assert_eq!(service.anchor(), (0, [0u8; 32]));

        let s1 = step(StepIO::new([0u8; 32], 0), StepIO::new(root(1), 1));
        let stats = service.extend(s1).expect("first extend must succeed");
        assert_eq!(stats.current_tip_height, 1);
        assert_eq!(stats.current_step_count, 1);
        assert_eq!(stats.total_extends_succeeded, 1);
        assert_eq!(stats.total_extends_rejected, 0);
        assert!(stats.last_extend_at_unix.is_some());

        let s2 = step(StepIO::new(root(1), 1), StepIO::new(root(2), 2));
        service.extend(s2).expect("second extend must succeed");

        assert_eq!(service.tip_height(), 2);
        assert_eq!(service.step_count(), 2);
        service
            .verify_chain_self_consistent()
            .expect("self-check must pass on healthy chain");
    }

    #[test]
    fn service_rejects_height_jump_and_records_failure() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());

        let bad = step(StepIO::new(root(99), 5), StepIO::new(root(100), 6));
        let result = service.extend(bad);
        assert!(result.is_err(), "height jump must be rejected");

        let stats = service.stats();
        assert_eq!(stats.total_extends_attempted, 1);
        assert_eq!(stats.total_extends_succeeded, 0);
        assert_eq!(stats.total_extends_rejected, 1);
        assert!(stats.last_failure_reason.is_some());

        // Failed extend must not have mutated the tip.
        assert_eq!(service.tip_height(), 0);
        assert_eq!(service.step_count(), 0);
    }

    #[test]
    fn service_drops_prefix_when_max_steps_retained_exceeded() {
        let config = TipProofServiceConfig::genesis().with_max_steps_retained(2);
        let service = TipProofService::new(config);

        for i in 0..5u8 {
            let z_in = if i == 0 {
                StepIO::new([0u8; 32], 0)
            } else {
                StepIO::new(root(i), i as u64)
            };
            let z_out = StepIO::new(root(i + 1), (i + 1) as u64);
            service.extend(step(z_in, z_out)).expect("extend");
        }

        // After 5 extends with cap=2, only the last 2 steps survive.
        // anchor advanced to z_in of the third extend.
        assert_eq!(service.step_count(), 2);
        assert_eq!(service.tip_height(), 5);
        assert_eq!(service.anchor().0, 3, "anchor advanced past dropped prefix");

        let stats = service.stats();
        assert!(stats.total_prefix_drops >= 1, "prefix drops must have fired");
        assert!(stats.total_anchor_resets >= 1);

        service
            .verify_chain_self_consistent()
            .expect("self-check must pass after prefix drop");
    }

    #[test]
    fn service_reset_anchor_clears_chain() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());

        let s1 = step(StepIO::new([0u8; 32], 0), StepIO::new(root(1), 1));
        service.extend(s1).expect("extend");

        service.reset_anchor(100, root(50));
        assert_eq!(service.tip_height(), 100);
        assert_eq!(service.step_count(), 0);
        assert_eq!(service.anchor(), (100, root(50)));

        let stats = service.stats();
        assert!(stats.total_anchor_resets >= 1);
        assert!(
            stats.last_extend_at_unix.is_none(),
            "reset must clear last_extend timestamp"
        );
    }

    #[test]
    fn service_current_proof_bytes_round_trips_through_bincode() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        let s = step(StepIO::new([0u8; 32], 0), StepIO::new(root(1), 1));
        service.extend(s).expect("extend");

        let bytes = service.current_proof_bytes().expect("serialise");
        let parsed: LatticeTipProofV2 =
            bincode::deserialize(&bytes).expect("deserialise");
        assert_eq!(parsed.tip_height, 1);
        assert_eq!(parsed.delta_proofs.len(), 1);
        assert_eq!(parsed.version, "latticeguard-rlwe-v1");
    }

    #[test]
    fn service_clone_shares_underlying_state() {
        let a = TipProofService::new(TipProofServiceConfig::genesis());
        let b = a.clone();
        let s = step(StepIO::new([0u8; 32], 0), StepIO::new(root(1), 1));
        a.extend(s).expect("extend via a");
        // b must observe the extend through the shared Arc<RwLock<...>>.
        assert_eq!(b.tip_height(), 1);
        assert_eq!(b.step_count(), 1);
    }

    #[test]
    fn service_proof_version_is_pinned() {
        let s = TipProofService::new(TipProofServiceConfig::genesis());
        assert_eq!(s.proof_version(), "latticeguard-rlwe-v1");
    }

    #[test]
    fn service_anchored_at_custom_height_starts_at_that_tip() {
        let s = TipProofService::new(TipProofServiceConfig::anchored_at(7777, [0xAB; 32]));
        assert_eq!(s.tip_height(), 7777);
        assert_eq!(s.anchor(), (7777, [0xAB; 32]));
        assert_eq!(s.step_count(), 0);
        // Stats should reflect the anchor's height as the current tip.
        let stats = s.stats();
        assert_eq!(stats.current_tip_height, 7777);
    }

    #[test]
    fn drop_prefix_handles_drop_count_at_or_above_chain_length() {
        // drop_prefix is a pure helper — exercise directly without a
        // folder so this test stays fast.
        let mut p = tip_proof_v2::anchor(0, [1u8; 32]);
        let s1 = step(StepIO::new([1u8; 32], 0), StepIO::new(root(1), 1));
        p = tip_proof_v2::extend_with_step_proof(&p, s1).unwrap();

        // n >= chain length: no-op (we don't drop everything because that
        // would orphan the anchor).
        let unchanged = drop_prefix(p.clone(), 5);
        assert_eq!(unchanged.delta_proofs.len(), p.delta_proofs.len());
        assert_eq!(unchanged.anchor_height, p.anchor_height);

        // n = 0: no-op.
        let unchanged_zero = drop_prefix(p.clone(), 0);
        assert_eq!(unchanged_zero.delta_proofs.len(), p.delta_proofs.len());
    }

    #[test]
    fn approximate_wire_size_scales_linearly_with_step_count() {
        let p0 = tip_proof_v2::anchor(0, [0u8; 32]);
        let size_0 = approximate_wire_size(&p0);

        let s1 = step(StepIO::new([0u8; 32], 0), StepIO::new(root(1), 1));
        let p1 = tip_proof_v2::extend_with_step_proof(&p0, s1).unwrap();
        let size_1 = approximate_wire_size(&p1);

        // Each step should add roughly the per-step constant.
        assert!(
            size_1 > size_0,
            "size estimate must grow with step count: {} → {}",
            size_0,
            size_1
        );
        let per_step = size_1 - size_0;
        // Sanity bound: per-step contribution is in the 10K-100K ballpark.
        assert!(per_step >= 10 * 1024 && per_step <= 100 * 1024);
    }

    #[test]
    fn stats_record_success_updates_fields() {
        let mut s = TipProofServiceStats::default();
        s.record_success(42, 7);
        assert_eq!(s.total_extends_attempted, 1);
        assert_eq!(s.total_extends_succeeded, 1);
        assert_eq!(s.total_extends_rejected, 0);
        assert_eq!(s.current_tip_height, 42);
        assert_eq!(s.current_step_count, 7);
        assert!(s.last_extend_at_unix.is_some());
        assert!(s.last_failure_reason.is_none());
    }

    #[test]
    fn stats_record_failure_does_not_clear_prior_success_timestamp() {
        let mut s = TipProofServiceStats::default();
        s.record_success(1, 1);
        let prior_ts = s.last_extend_at_unix;
        s.record_failure("bad".to_string());
        assert_eq!(s.total_extends_attempted, 2);
        assert_eq!(s.total_extends_succeeded, 1);
        assert_eq!(s.total_extends_rejected, 1);
        assert_eq!(
            s.last_extend_at_unix, prior_ts,
            "failure must not clear the prior success timestamp"
        );
        assert_eq!(s.last_failure_reason.as_deref(), Some("bad"));
    }

    #[test]
    fn config_builder_chain() {
        let c = TipProofServiceConfig::anchored_at(100, [9u8; 32])
            .with_max_steps_retained(50)
            .with_trace_each_extend();
        assert_eq!(c.anchor_height, 100);
        assert_eq!(c.anchor_state, [9u8; 32]);
        assert_eq!(c.max_steps_retained, Some(50));
        assert!(c.trace_each_extend);
    }
}
