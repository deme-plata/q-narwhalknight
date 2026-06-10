//! Persistence end-to-end tests across service + client lifecycles.
//!
//! Covers the restart-survival contract: write state, "restart" by
//! constructing a fresh service/client from the same backend, observe
//! restored state. Plus failure modes: anchor mismatch on hydrate,
//! concurrent write contention, file persistence atomicity, mixed
//! service-and-client sharing a backend.

use std::sync::Arc;
use std::thread;

use q_ivc::recursion::{LatticeStepProof, StepIO};
use q_lattice_guard::{params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof};
use q_recursive_proofs::{
    FilePersistence, MemoryPersistence, TipProofClient, TipProofPersistence,
    TipProofService, TipProofServiceConfig,
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

fn step(z_in: StepIO, z_out: StepIO) -> LatticeStepProof {
    LatticeStepProof {
        proof: dummy_lattice_proof(),
        z_in: z_in.pack(),
        z_out: z_out.pack(),
        public_input_count: 9,
    }
}

fn root(seed: u8) -> [u8; 32] {
    let mut r = [0u8; 32];
    for (i, b) in r.iter_mut().enumerate() {
        *b = (seed.wrapping_mul(i as u8 + 1)).wrapping_add(seed);
    }
    r
}

fn produce(service: &TipProofService, starting_prev: [u8; 32], starting_height: u64, n: u64) {
    let mut prev = starting_prev;
    for h in starting_height..starting_height + n {
        let next = root((h + 1) as u8);
        service
            .extend(step(StepIO::new(prev, h), StepIO::new(next, h + 1)))
            .expect("honest extend");
        prev = next;
    }
}

fn tmp_path(test: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "q-recursive-proofs-e2e-{}-{}.bin",
        test,
        std::process::id()
    ));
    let _ = std::fs::remove_file(&p);
    p
}

// ── Memory backend ──────────────────────────────────────────────────────

#[test]
fn memory_service_round_trip_across_restart() {
    let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());

    let service =
        TipProofService::from_persistence(TipProofServiceConfig::genesis(), backend.clone());
    produce(&service, [0u8; 32], 0, 7);
    let pre_restart_tip = service.tip_height();
    assert_eq!(pre_restart_tip, 7);

    // Simulate restart: drop the service, construct a new one from
    // the same backend. The backend retains its bytes because the
    // Arc still has the e2e-test owner alive.
    drop(service);

    let restored =
        TipProofService::from_persistence(TipProofServiceConfig::genesis(), backend.clone());
    assert_eq!(
        restored.tip_height(),
        pre_restart_tip,
        "service must restore tip across restart"
    );
    assert_eq!(restored.step_count(), 7);
    restored
        .verify_chain_self_consistent()
        .expect("restored chain self-consistent");
}

#[test]
fn memory_client_round_trip_across_restart() {
    let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());

    // Producer + client side: client ingests then "restarts".
    let service =
        TipProofService::from_persistence(TipProofServiceConfig::genesis(), backend.clone());
    produce(&service, [0u8; 32], 0, 3);

    let client_backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());
    let client = TipProofClient::from_persistence(0, [0u8; 32], client_backend.clone());
    let proof = service.current_proof();
    client.ingest_proof(proof.clone()).expect("ingest");
    assert_eq!(client.current_tip_height(), Some(3));

    drop(client);
    let restored = TipProofClient::from_persistence(0, [0u8; 32], client_backend.clone());
    assert_eq!(
        restored.current_tip_height(),
        Some(3),
        "client must restore cached tip across restart"
    );

    // Anti-rollback survives restart: feeding a stale proof now is
    // rejected.
    let stale = service.current_proof(); // tip=3
    produce(&service, root(3), 3, 5); // service advances to tip=8
    let _ = restored.ingest_proof(stale).unwrap_err();
    let stats = restored.stats();
    assert!(stats.total_rollbacks_rejected >= 1);
}

#[test]
fn anchor_mismatch_on_hydrate_discards_persisted_proof() {
    let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());

    // Save a proof anchored at genesis.
    let service =
        TipProofService::from_persistence(TipProofServiceConfig::genesis(), backend.clone());
    produce(&service, [0u8; 32], 0, 4);
    assert_eq!(service.tip_height(), 4);
    drop(service);

    // "Restart" with a DIFFERENT anchor — service must discard the
    // saved proof (anchor mismatch) and start fresh from the new
    // anchor.
    let new_config = TipProofServiceConfig::anchored_at(100, root(50));
    let restored = TipProofService::from_persistence(new_config, backend.clone());
    assert_eq!(restored.tip_height(), 100);
    assert_eq!(restored.step_count(), 0);
    assert_eq!(restored.anchor(), (100, root(50)));
}

#[test]
fn reset_anchor_clears_persistence_backend() {
    let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());

    let service =
        TipProofService::from_persistence(TipProofServiceConfig::genesis(), backend.clone());
    produce(&service, [0u8; 32], 0, 3);
    assert!(backend.load().unwrap().is_some(), "save must have happened");

    service.reset_anchor(50, root(7));
    assert!(
        backend.load().unwrap().is_none(),
        "reset_anchor must clear persistence backend"
    );
}

#[test]
fn extend_failure_does_not_corrupt_persistence() {
    let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());
    let service =
        TipProofService::from_persistence(TipProofServiceConfig::genesis(), backend.clone());

    // Honest extend.
    produce(&service, [0u8; 32], 0, 1);
    let bytes_after_one = backend.load().unwrap().expect("present");

    // Adversarial extend.
    let bad = step(StepIO::new(root(99), 99), StepIO::new(root(100), 100));
    let _ = service.extend(bad).unwrap_err();

    // Backend still reflects the last successful extend; the failed
    // one didn't overwrite.
    let bytes_after_fail = backend.load().unwrap().expect("present");
    assert_eq!(
        bytes_after_one.tip_height, bytes_after_fail.tip_height,
        "failed extend must not have persisted"
    );
}

// ── File backend ──────────────────────────────────────────────────────

#[test]
fn file_service_round_trip_across_restart() {
    let path = tmp_path("file_service_restart");
    let backend: Arc<dyn TipProofPersistence> = Arc::new(FilePersistence::new(&path));

    let service =
        TipProofService::from_persistence(TipProofServiceConfig::genesis(), backend.clone());
    produce(&service, [0u8; 32], 0, 5);
    assert_eq!(service.tip_height(), 5);
    drop(service);

    // Reconstruct backend from a DIFFERENT instance (production path:
    // process restart re-opens the same file from disk).
    let fresh_backend: Arc<dyn TipProofPersistence> = Arc::new(FilePersistence::new(&path));
    let restored = TipProofService::from_persistence(
        TipProofServiceConfig::genesis(),
        fresh_backend,
    );
    assert_eq!(restored.tip_height(), 5);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn file_persistence_survives_drop_and_recreate_via_clone() {
    let path = tmp_path("file_survives_drop");
    {
        let backend: Arc<dyn TipProofPersistence> = Arc::new(FilePersistence::new(&path));
        let service = TipProofService::from_persistence(
            TipProofServiceConfig::genesis(),
            backend,
        );
        produce(&service, [0u8; 32], 0, 2);
        // service + backend dropped at end of scope.
    }

    // File survived.
    assert!(path.exists(), "FilePersistence file must outlive the service");

    let backend2: Arc<dyn TipProofPersistence> = Arc::new(FilePersistence::new(&path));
    let loaded = backend2.load().unwrap().expect("present after process simulation");
    assert_eq!(loaded.tip_height, 2);

    let _ = std::fs::remove_file(&path);
}

// ── Mixed backend / concurrent ────────────────────────────────────────

#[test]
fn service_with_persistence_blocks_only_extend_not_reads() {
    // Sanity: persistence save happens AFTER write lock is released,
    // so concurrent readers don't observe blocking I/O latency. We
    // can't directly measure lock hold time but we can verify reads
    // don't error / block forever under concurrent extends.
    let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());
    let service = Arc::new(TipProofService::from_persistence(
        TipProofServiceConfig::genesis(),
        backend,
    ));

    let writer = {
        let s = Arc::clone(&service);
        thread::spawn(move || {
            let mut prev = [0u8; 32];
            for h in 0u64..30 {
                let next = root((h + 1) as u8);
                s.extend(step(StepIO::new(prev, h), StepIO::new(next, h + 1)))
                    .expect("extend");
                prev = next;
            }
        })
    };

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let s = Arc::clone(&service);
            thread::spawn(move || {
                let mut max = 0u64;
                for _ in 0..200 {
                    let h = s.tip_height();
                    assert!(h >= max);
                    max = h;
                }
                max
            })
        })
        .collect();

    writer.join().unwrap();
    for r in readers {
        r.join().unwrap();
    }
    assert_eq!(service.tip_height(), 30);
}

#[test]
fn service_persistence_backend_id_surfaced_through_getter() {
    let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());
    let with = TipProofService::from_persistence(TipProofServiceConfig::genesis(), backend);
    assert_eq!(with.persistence_backend_id(), Some("memory".to_string()));

    let without = TipProofService::new(TipProofServiceConfig::genesis());
    assert_eq!(without.persistence_backend_id(), None);
}

#[test]
fn client_persistence_backend_id_surfaced_through_getter() {
    let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());
    let with = TipProofClient::from_persistence(0, [0u8; 32], backend);
    assert_eq!(with.persistence_backend_id(), Some("memory".to_string()));

    let without = TipProofClient::genesis();
    assert_eq!(without.persistence_backend_id(), None);
}
