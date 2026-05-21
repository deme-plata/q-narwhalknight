//! Background task that produces step proofs and feeds a
//! [`TipProofService`].
//!
//! [`TipProofService`]: crate::TipProofService
//!
//! # The pipeline
//!
//! ```text
//!      block-producer (consensus / mempool)
//!                 │
//!                 │ mpsc<BlockEvent>
//!                 ▼
//!           BlockProducerTask     ┌─ owns ─┐
//!         ┌──────────────────────┐│         │
//!         │ 1. recv BlockEvent   ││ folder  │
//!         │ 2. call folder.prove ├┘         │
//!         │ 3. service.extend    ┐          │
//!         │ 4. emit telemetry    ├─ owns ──▶│ service
//!         │ 5. loop              ┘          │
//!         └──────────────────────┘          │
//!                                            │
//! API server reads service.current_proof_bytes() ◀──┘
//! ```
//!
//! # Design notes
//!
//! - **Decoupling.** Block production (1 bps consensus) and proof
//!   generation (~1s/step at pq128) run independently. The mpsc buffer
//!   absorbs proof-gen latency without back-pressuring consensus.
//! - **Graceful shutdown.** The task exits when the sender side of the
//!   channel drops OR a shutdown signal is received via the
//!   `cancellation_token` (caller-provided). On exit it flushes any
//!   in-flight extend before returning.
//! - **Failure tolerance.** A failed `folder.prove_step` (e.g. circuit
//!   too large for SRS) increments a counter and is logged, but the
//!   task continues — the next block gets a fresh attempt. The user
//!   sees the failure surface as `TipProofServiceStats.last_failure_reason`.
//! - **Testability.** [`BlockProducerTask`] is generic over the
//!   step-proof source. Tests use a closure that produces stub
//!   `LatticeStepProof`s without paying SRS cost; production uses a
//!   `LatticeStepFolder` via [`folder_strategy`].

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

use q_ivc::recursion::{LatticeStepFolder, LatticeStepProof, StepIO};

use crate::tip_proof_service::TipProofService;
use crate::tip_proof_v2::VerifyErrorV2;

// ════════════════════════════════════════════════════════════════════════════
// Block event
// ════════════════════════════════════════════════════════════════════════════

/// Notification of a new block ready for proof generation.
///
/// Producer sends one per block; the task reads one, generates a
/// `LatticeStepProof` for that block's δ-transition, and feeds the
/// result into the service.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockEvent {
    /// The state at the START of this block (i.e. the previous block's
    /// state_root_next). Becomes the new step's `z_in`.
    pub z_in: StepIO,
    /// The state at the END of this block. Becomes the new step's `z_out`.
    pub z_out: StepIO,
}

impl BlockEvent {
    pub fn new(z_in: StepIO, z_out: StepIO) -> Self {
        Self { z_in, z_out }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Step-proof source
// ════════════════════════════════════════════════════════════════════════════

/// Pluggable step-proof generation strategy. Production uses
/// [`folder_strategy`]; tests use a closure that returns stub proofs
/// instantly.
pub trait StepProofSource: Send + Sync {
    /// Produce a step proof for the given `(z_in, z_out)` transition.
    fn prove_step(
        &self,
        event: &BlockEvent,
    ) -> Result<LatticeStepProof, StepProofError>;
}

#[derive(Debug, Error)]
pub enum StepProofError {
    #[error("folder error: {0}")]
    Folder(#[from] q_ivc::recursion::FolderError),

    #[error("custom strategy error: {0}")]
    Custom(String),
}

/// Strategy that drives a [`LatticeStepFolder`] over a caller-supplied
/// circuit builder. The circuit builder produces an arkworks
/// `ConstraintSynthesizer` from the block event; the folder bridges it
/// into `q-lattice-guard` and proves.
pub fn folder_strategy<F, B, C>(
    folder: Arc<LatticeStepFolder>,
    circuit_builder: B,
) -> Arc<dyn StepProofSource>
where
    F: ark_ff::PrimeField + Send + Sync + 'static,
    B: Fn(&BlockEvent) -> C + Send + Sync + 'static,
    C: ark_relations::r1cs::ConstraintSynthesizer<F> + 'static,
{
    Arc::new(FolderStrategy {
        folder,
        circuit_builder,
        _f: std::marker::PhantomData,
        _c: std::marker::PhantomData,
    })
}

struct FolderStrategy<F, B, C> {
    folder: Arc<LatticeStepFolder>,
    circuit_builder: B,
    // `fn() -> T` variant: PhantomData is always `Send + Sync` regardless
    // of whether `T` is. The `StepProofSource` supertrait requires this
    // struct to be `Send + Sync`, but `C: ConstraintSynthesizer<F>` does
    // not (and cannot — arkworks constraint synthesisers are not
    // Send/Sync in general). The fn-pointer indirection erases the
    // auto-trait dependency on `C`.
    _f: std::marker::PhantomData<fn() -> F>,
    _c: std::marker::PhantomData<fn() -> C>,
}

impl<F, B, C> StepProofSource for FolderStrategy<F, B, C>
where
    F: ark_ff::PrimeField + Send + Sync + 'static,
    B: Fn(&BlockEvent) -> C + Send + Sync,
    C: ark_relations::r1cs::ConstraintSynthesizer<F>,
{
    fn prove_step(
        &self,
        event: &BlockEvent,
    ) -> Result<LatticeStepProof, StepProofError> {
        let circuit = (self.circuit_builder)(event);
        self.folder
            .prove_step::<F, _>(circuit, event.z_in, event.z_out)
            .map_err(StepProofError::Folder)
    }
}

/// Stub strategy that returns a caller-supplied proof verbatim — used
/// by tests to bypass real SRS-backed proving. Wrap with `Arc::new(...)`
/// to satisfy the `Arc<dyn StepProofSource>` argument shape.
pub struct StubStepProofSource<F>
where
    F: Fn(&BlockEvent) -> LatticeStepProof + Send + Sync,
{
    f: F,
}

impl<F> StubStepProofSource<F>
where
    F: Fn(&BlockEvent) -> LatticeStepProof + Send + Sync,
{
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F> StepProofSource for StubStepProofSource<F>
where
    F: Fn(&BlockEvent) -> LatticeStepProof + Send + Sync,
{
    fn prove_step(
        &self,
        event: &BlockEvent,
    ) -> Result<LatticeStepProof, StepProofError> {
        Ok((self.f)(event))
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BlockProducerTask
// ════════════════════════════════════════════════════════════════════════════

/// Configuration for [`BlockProducerTask::spawn`].
#[derive(Clone, Debug)]
pub struct BlockProducerTaskConfig {
    /// Capacity of the BlockEvent mpsc channel. Larger = more burst
    /// absorption; smaller = more back-pressure to the producer. Default 64.
    pub channel_capacity: usize,
}

impl Default for BlockProducerTaskConfig {
    fn default() -> Self {
        Self { channel_capacity: 64 }
    }
}

/// Counters exposed by [`BlockProducerTask::stats`]. Snapshot is atomic
/// across reads; cloning is cheap.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BlockProducerTaskStats {
    pub events_received: u64,
    pub events_proven: u64,
    pub prove_failures: u64,
    pub extends_succeeded: u64,
    pub extends_rejected: u64,
}

/// Handle to a spawned background task. Drop to signal shutdown;
/// `await` the join_handle to wait for clean exit.
pub struct BlockProducerHandle {
    /// Submit new block events through this sender. Drop the last
    /// sender clone to signal "no more events" and the task will exit
    /// after draining.
    pub sender: mpsc::Sender<BlockEvent>,
    /// Read live stats. Updated atomically through a `watch` channel
    /// so readers never block writers.
    pub stats_rx: watch::Receiver<BlockProducerTaskStats>,
    /// Set to `true` to request shutdown. The task checks this between
    /// events and exits after the current in-flight extend.
    pub shutdown_tx: watch::Sender<bool>,
    /// JoinHandle for the spawned task. Await it after sending shutdown
    /// to confirm clean exit.
    pub join_handle: tokio::task::JoinHandle<()>,
}

/// The background runner. Use [`Self::spawn`] to start it.
pub struct BlockProducerTask {
    service: TipProofService,
    source: Arc<dyn StepProofSource>,
    config: BlockProducerTaskConfig,
}

impl BlockProducerTask {
    /// Spawn the task on the current tokio runtime. Returns a
    /// [`BlockProducerHandle`] for sending events + observing stats +
    /// requesting shutdown.
    pub fn spawn(
        service: TipProofService,
        source: Arc<dyn StepProofSource>,
        config: BlockProducerTaskConfig,
    ) -> BlockProducerHandle {
        let (event_tx, mut event_rx) = mpsc::channel::<BlockEvent>(config.channel_capacity);
        let (stats_tx, stats_rx) = watch::channel(BlockProducerTaskStats::default());
        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

        let task = Self {
            service,
            source,
            config: config.clone(),
        };

        let join_handle = tokio::spawn(async move {
            info!("BlockProducerTask: started, channel_capacity={}", task.config.channel_capacity);
            let mut stats = BlockProducerTaskStats::default();

            loop {
                tokio::select! {
                    biased;
                    // Shutdown takes priority — but still drain any
                    // events that arrived between the signal and the
                    // recv() returning None.
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            info!("BlockProducerTask: shutdown requested, draining channel");
                            while let Ok(event) = event_rx.try_recv() {
                                Self::process_event(&task, &event, &mut stats, &stats_tx);
                            }
                            break;
                        }
                    }
                    maybe_event = event_rx.recv() => {
                        match maybe_event {
                            Some(event) => {
                                Self::process_event(&task, &event, &mut stats, &stats_tx);
                            }
                            None => {
                                info!("BlockProducerTask: sender dropped, exiting");
                                break;
                            }
                        }
                    }
                }
            }

            info!(
                "BlockProducerTask: exited cleanly (events_received={}, extends_succeeded={})",
                stats.events_received, stats.extends_succeeded
            );
        });

        BlockProducerHandle {
            sender: event_tx,
            stats_rx,
            shutdown_tx,
            join_handle,
        }
    }

    fn process_event(
        task: &Self,
        event: &BlockEvent,
        stats: &mut BlockProducerTaskStats,
        stats_tx: &watch::Sender<BlockProducerTaskStats>,
    ) {
        stats.events_received += 1;
        debug!(
            "BlockProducerTask: received block event z_in.height={} z_out.height={}",
            event.z_in.height, event.z_out.height
        );

        let proof = match task.source.prove_step(event) {
            Ok(p) => {
                stats.events_proven += 1;
                p
            }
            Err(e) => {
                stats.prove_failures += 1;
                error!(
                    "BlockProducerTask: prove_step failed for block at z_out.height={}: {}",
                    event.z_out.height, e
                );
                let _ = stats_tx.send(stats.clone());
                return;
            }
        };

        match task.service.extend(proof) {
            Ok(_) => stats.extends_succeeded += 1,
            Err(VerifyErrorV2::HeightDiscontinuity { .. })
            | Err(VerifyErrorV2::ChainDiscontinuity { .. }) => {
                stats.extends_rejected += 1;
                warn!(
                    "BlockProducerTask: service rejected extend (out-of-order block at z_in.height={})",
                    event.z_in.height
                );
            }
            Err(e) => {
                stats.extends_rejected += 1;
                warn!(
                    "BlockProducerTask: service rejected extend: {} (z_in.height={})",
                    e, event.z_in.height
                );
            }
        }

        let _ = stats_tx.send(stats.clone());
    }
}

impl BlockProducerHandle {
    /// Send one block event. Returns an error if the task has exited.
    pub async fn submit(&self, event: BlockEvent) -> Result<(), mpsc::error::SendError<BlockEvent>> {
        self.sender.send(event).await
    }

    /// Snapshot current stats (cheap, lock-free).
    pub fn stats(&self) -> BlockProducerTaskStats {
        self.stats_rx.borrow().clone()
    }

    /// Request graceful shutdown. Subsequent `submit` calls may succeed
    /// or fail depending on timing; safest to drop the sender before
    /// awaiting the join_handle.
    pub fn request_shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Convenience: signal shutdown + drop sender + await join_handle.
    /// Returns the final stats snapshot.
    pub async fn shutdown_and_join(self) -> BlockProducerTaskStats {
        self.request_shutdown();
        drop(self.sender);
        let _ = self.join_handle.await;
        self.stats_rx.borrow().clone()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TipProofService, TipProofServiceConfig};
    use q_lattice_guard::{
        params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof,
    };

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

    fn root(seed: u8) -> [u8; 32] {
        let mut r = [0u8; 32];
        for (i, b) in r.iter_mut().enumerate() {
            *b = (seed.wrapping_mul(i as u8 + 1)).wrapping_add(seed);
        }
        r
    }

    /// Stub strategy that produces a step proof matching the event's
    /// `(z_in, z_out)` — used to exercise the task plumbing without
    /// SRS-backed proving.
    fn stub_source() -> Arc<dyn StepProofSource> {
        Arc::new(StubStepProofSource::new(|event: &BlockEvent| {
            LatticeStepProof {
                proof: dummy_lattice_proof(),
                z_in: event.z_in.pack(),
                z_out: event.z_out.pack(),
                public_input_count: 9,
            }
        }))
    }

    #[tokio::test]
    async fn task_processes_events_in_order() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        let handle = BlockProducerTask::spawn(
            service.clone(),
            stub_source(),
            BlockProducerTaskConfig::default(),
        );

        // Submit 5 honest block events.
        let mut prev = [0u8; 32];
        for h in 0u64..5 {
            let next = root((h + 1) as u8);
            handle
                .submit(BlockEvent::new(
                    StepIO::new(prev, h),
                    StepIO::new(next, h + 1),
                ))
                .await
                .expect("submit");
            prev = next;
        }

        let stats = handle.shutdown_and_join().await;
        assert_eq!(stats.events_received, 5);
        assert_eq!(stats.events_proven, 5);
        assert_eq!(stats.extends_succeeded, 5);
        assert_eq!(stats.extends_rejected, 0);
        assert_eq!(service.tip_height(), 5);
    }

    #[tokio::test]
    async fn task_records_rejection_when_service_extends_fail() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        let handle = BlockProducerTask::spawn(
            service.clone(),
            stub_source(),
            BlockProducerTaskConfig::default(),
        );

        // Submit an out-of-order event (height 5 with no prior chain).
        handle
            .submit(BlockEvent::new(
                StepIO::new(root(99), 5),
                StepIO::new(root(100), 6),
            ))
            .await
            .expect("submit");

        let stats = handle.shutdown_and_join().await;
        assert_eq!(stats.events_received, 1);
        assert_eq!(stats.events_proven, 1);
        assert_eq!(stats.extends_rejected, 1, "service must reject out-of-order extend");
        assert_eq!(service.tip_height(), 0, "service tip must be unchanged on rejection");
    }

    #[tokio::test]
    async fn task_exits_when_sender_dropped() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        let handle = BlockProducerTask::spawn(
            service,
            stub_source(),
            BlockProducerTaskConfig::default(),
        );

        // Drop the sender — task should exit promptly.
        drop(handle.sender);
        let _ = handle.join_handle.await;
        // If we got here, the task exited cleanly.
    }

    #[tokio::test]
    async fn task_drains_pending_events_on_shutdown() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        let handle = BlockProducerTask::spawn(
            service.clone(),
            stub_source(),
            BlockProducerTaskConfig { channel_capacity: 16 },
        );

        // Submit a burst before requesting shutdown.
        let mut prev = [0u8; 32];
        for h in 0u64..3 {
            let next = root((h + 1) as u8);
            handle
                .submit(BlockEvent::new(
                    StepIO::new(prev, h),
                    StepIO::new(next, h + 1),
                ))
                .await
                .expect("submit");
            prev = next;
        }

        let stats = handle.shutdown_and_join().await;
        // All 3 events should have been processed before the task exited.
        assert_eq!(stats.events_received, 3);
        assert_eq!(service.tip_height(), 3);
    }

    #[tokio::test]
    async fn task_stats_observable_via_watch_channel() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        let handle = BlockProducerTask::spawn(
            service,
            stub_source(),
            BlockProducerTaskConfig::default(),
        );

        // Initially stats are default.
        let initial = handle.stats();
        assert_eq!(initial.events_received, 0);

        handle
            .submit(BlockEvent::new(
                StepIO::new([0u8; 32], 0),
                StepIO::new(root(1), 1),
            ))
            .await
            .expect("submit");

        let _ = handle.shutdown_and_join().await;
    }

    #[test]
    fn block_event_round_trip() {
        let e = BlockEvent::new(
            StepIO::new([0xAA; 32], 7),
            StepIO::new([0xBB; 32], 8),
        );
        assert_eq!(e.z_in.height, 7);
        assert_eq!(e.z_out.height, 8);
        assert_eq!(e.z_in.state_root, [0xAA; 32]);
    }

    #[test]
    fn config_default_channel_capacity_is_sane() {
        let c = BlockProducerTaskConfig::default();
        assert!(
            c.channel_capacity > 0 && c.channel_capacity <= 1024,
            "channel_capacity should be in a sane range, got {}",
            c.channel_capacity
        );
    }

    #[test]
    fn stub_step_proof_source_returns_provided_proof() {
        let s = StubStepProofSource::new(|event: &BlockEvent| LatticeStepProof {
            proof: dummy_lattice_proof(),
            z_in: event.z_in.pack(),
            z_out: event.z_out.pack(),
            public_input_count: 9,
        });
        let event = BlockEvent::new(
            StepIO::new([0u8; 32], 0),
            StepIO::new(root(1), 1),
        );
        let p = s.prove_step(&event).expect("stub must succeed");
        assert_eq!(p.z_in_unpacked().height, 0);
        assert_eq!(p.z_out_unpacked().height, 1);
    }
}
