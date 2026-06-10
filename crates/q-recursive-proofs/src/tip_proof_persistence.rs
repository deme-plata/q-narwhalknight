//! Pluggable persistence backend for [`TipProofService`] and
//! [`TipProofClient`].
//!
//! [`TipProofService`]: crate::TipProofService
//! [`TipProofClient`]: crate::TipProofClient
//!
//! Without persistence the Phase B1 stack loses everything on restart:
//!
//!   * A producer's chain shrinks to anchor-only — every fold step done
//!     between the anchor and the crash is gone.
//!   * A consumer's anti-rollback cache resets to `None`, leaving the
//!     wallet vulnerable to a one-time rollback by a malicious upstream
//!     across restart.
//!
//! Both are unacceptable for production. This module defines the
//! pluggable [`TipProofPersistence`] trait and ships two concrete impls:
//!
//! - [`MemoryPersistence`] — for unit + integration tests. Backed by an
//!   `Arc<Mutex<Option<Vec<u8>>>>`; survives across clones but vanishes
//!   when the last reference drops.
//! - [`FilePersistence`] — for production. Atomic write via
//!   `tmp + rename`; `std::fs` only (no RocksDB / sled / heavy deps).
//!   Stores one bincode'd `LatticeTipProofV2` per file. Tolerant of
//!   missing files (returns `Ok(None)`).
//!
//! # Atomicity
//!
//! `FilePersistence::save` writes to `<path>.tmp` then renames to
//! `<path>`. On crash mid-write the original `<path>` is unchanged —
//! the rename is the atomic commit point. A subsequent `load` either
//! sees the old or the new payload, never a torn write.
//!
//! # Why not RocksDB
//!
//! q-storage already uses RocksDB for chain state. Adding a second
//! RocksDB dep here would double the embedded-db footprint for what's
//! a single small blob. The file-based backend is enough for ~50 KB
//! proofs at Phase B1; Phase C's constant-size accumulator brings the
//! size down further, making heavyweight storage even more unjustified.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use bincode;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::tip_proof_v2::LatticeTipProofV2;

// ════════════════════════════════════════════════════════════════════════════
// Errors
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("bincode error: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("backend lock poisoned")]
    LockPoisoned,
}

pub type PersistenceResult<T> = Result<T, PersistenceError>;

// ════════════════════════════════════════════════════════════════════════════
// Trait
// ════════════════════════════════════════════════════════════════════════════

/// Backend-agnostic persistence for one `LatticeTipProofV2`.
///
/// Implementors MUST be thread-safe (`Send + Sync`); both
/// [`TipProofService`](crate::TipProofService) and
/// [`TipProofClient`](crate::TipProofClient) share their backend via
/// `Arc<dyn TipProofPersistence>`.
///
/// **Concurrency contract:** `save` and `load` MAY interleave from
/// different threads. Implementors are responsible for serialising
/// writes if their backing store doesn't support concurrent writers
/// (the file backend uses an internal mutex + atomic rename).
pub trait TipProofPersistence: Send + Sync + std::fmt::Debug {
    /// Persist the proof bytes. Returns when the bytes are durably
    /// committed (file fsync'd, db transaction completed, etc.).
    fn save(&self, proof: &LatticeTipProofV2) -> PersistenceResult<()>;

    /// Load the most recently persisted proof. Returns `Ok(None)` if
    /// the backend is empty (fresh install — never written to).
    fn load(&self) -> PersistenceResult<Option<LatticeTipProofV2>>;

    /// Drop the persisted proof. After this, `load` returns `Ok(None)`.
    /// Used during anchor resets when the cached chain is no longer
    /// trusted.
    fn clear(&self) -> PersistenceResult<()>;

    /// A human-readable identifier for logging / diagnostics. Should
    /// not contain secrets.
    fn backend_id(&self) -> &str;
}

// ════════════════════════════════════════════════════════════════════════════
// MemoryPersistence — for tests
// ════════════════════════════════════════════════════════════════════════════

/// In-memory backend. Cloning shares the underlying `Arc<Mutex<...>>`
/// so multiple services / clients in the same test process see the
/// same persisted bytes.
#[derive(Clone, Debug)]
pub struct MemoryPersistence {
    bytes: Arc<Mutex<Option<Vec<u8>>>>,
}

impl Default for MemoryPersistence {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPersistence {
    pub fn new() -> Self {
        Self {
            bytes: Arc::new(Mutex::new(None)),
        }
    }

    /// Snapshot the currently persisted bytes for inspection (testing).
    pub fn snapshot_bytes(&self) -> Option<Vec<u8>> {
        self.bytes.lock().ok().and_then(|g| g.clone())
    }
}

impl TipProofPersistence for MemoryPersistence {
    fn save(&self, proof: &LatticeTipProofV2) -> PersistenceResult<()> {
        let bytes = bincode::serialize(proof)?;
        let mut g = self.bytes.lock().map_err(|_| PersistenceError::LockPoisoned)?;
        *g = Some(bytes);
        Ok(())
    }

    fn load(&self) -> PersistenceResult<Option<LatticeTipProofV2>> {
        let g = self.bytes.lock().map_err(|_| PersistenceError::LockPoisoned)?;
        match g.as_ref() {
            None => Ok(None),
            Some(bytes) => Ok(Some(bincode::deserialize(bytes)?)),
        }
    }

    fn clear(&self) -> PersistenceResult<()> {
        let mut g = self.bytes.lock().map_err(|_| PersistenceError::LockPoisoned)?;
        *g = None;
        Ok(())
    }

    fn backend_id(&self) -> &str {
        "memory"
    }
}

// ════════════════════════════════════════════════════════════════════════════
// FilePersistence — for production
// ════════════════════════════════════════════════════════════════════════════

/// File-system backend. One file per backend instance. Atomic writes
/// via tmp+rename so a crash mid-write leaves the original intact.
///
/// Recommended path: `/var/lib/quillon/tip_proof_v2.bin` for the
/// producer service; per-user `~/.quillon/wallet_tip_cache.bin` for
/// wallet clients.
#[derive(Debug)]
pub struct FilePersistence {
    /// The committed path. Reads from here; writes go through `<path>.tmp`
    /// first then rename.
    path: PathBuf,
    /// Serialises concurrent `save` calls — the OS handles concurrent
    /// reads safely but the tmp-file dance needs at-most-one writer.
    write_lock: Mutex<()>,
    /// `backend_id` string formed from the path; cached so the getter
    /// doesn't allocate.
    id: String,
}

impl FilePersistence {
    /// Create a backend at `path`. Does NOT touch the filesystem;
    /// `load` will succeed (returning `None`) if the file doesn't exist.
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        let path = path.into();
        let id = format!("file:{}", path.display());
        Self {
            path,
            write_lock: Mutex::new(()),
            id,
        }
    }

    fn tmp_path(&self) -> PathBuf {
        self.path.with_extension("tmp")
    }
}

impl TipProofPersistence for FilePersistence {
    fn save(&self, proof: &LatticeTipProofV2) -> PersistenceResult<()> {
        let bytes = bincode::serialize(proof)?;
        let tmp = self.tmp_path();

        // Serialise writes — single writer at a time.
        let _guard = self.write_lock.lock().map_err(|_| PersistenceError::LockPoisoned)?;

        // Ensure the parent dir exists; first-run friendliness.
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Write to tmp, fsync, rename.
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)?;

        debug!(
            "FilePersistence: saved {} bytes to {}",
            bytes.len(),
            self.path.display()
        );
        Ok(())
    }

    fn load(&self) -> PersistenceResult<Option<LatticeTipProofV2>> {
        match std::fs::read(&self.path) {
            Ok(bytes) => {
                let proof: LatticeTipProofV2 = bincode::deserialize(&bytes)?;
                debug!(
                    "FilePersistence: loaded proof at tip {} from {}",
                    proof.tip_height,
                    self.path.display()
                );
                Ok(Some(proof))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                debug!(
                    "FilePersistence: no file at {} — fresh state",
                    self.path.display()
                );
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    fn clear(&self) -> PersistenceResult<()> {
        let _guard = self.write_lock.lock().map_err(|_| PersistenceError::LockPoisoned)?;
        match std::fs::remove_file(&self.path) {
            Ok(()) => {
                info!("FilePersistence: cleared {}", self.path.display());
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()), // already absent
            Err(e) => Err(e.into()),
        }
    }

    fn backend_id(&self) -> &str {
        &self.id
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Persistence-aware service/client constructors
// ════════════════════════════════════════════════════════════════════════════

/// Convenience: load + delegate. Used by [`TipProofService::from_persistence`]
/// and [`TipProofClient::from_persistence`] in their respective modules.
///
/// If the backend has a saved proof, returns `Some(proof)`. If load
/// fails (corruption, schema drift), logs a warning and returns `None`
/// — the caller decides whether to anchor-only or to refuse to boot.
///
/// [`TipProofService::from_persistence`]: crate::TipProofService::from_persistence
/// [`TipProofClient::from_persistence`]: crate::TipProofClient::from_persistence
pub fn load_or_warn(
    backend: &Arc<dyn TipProofPersistence>,
    context: &str,
) -> Option<LatticeTipProofV2> {
    match backend.load() {
        Ok(Some(proof)) => {
            info!(
                "{}: restored from persistence ({}) at tip {}",
                context,
                backend.backend_id(),
                proof.tip_height
            );
            Some(proof)
        }
        Ok(None) => {
            debug!("{}: persistence ({}) is empty", context, backend.backend_id());
            None
        }
        Err(e) => {
            warn!(
                "{}: persistence ({}) load failed: {} — starting fresh",
                context,
                backend.backend_id(),
                e
            );
            None
        }
    }
}

/// Wrapper that retries `save` once before surfacing the error — covers
/// transient EBUSY / disk-full hiccups without aborting an extend.
pub fn save_with_retry(
    backend: &Arc<dyn TipProofPersistence>,
    proof: &LatticeTipProofV2,
) -> PersistenceResult<()> {
    match backend.save(proof) {
        Ok(()) => Ok(()),
        Err(first_err) => {
            warn!(
                "persistence save failed ({}); retrying once: {}",
                backend.backend_id(),
                first_err
            );
            // Brief yield so a momentary disk-busy condition has a
            // chance to clear. `std::thread::yield_now()` is std-only
            // and doesn't pull in tokio.
            std::thread::yield_now();
            backend.save(proof)
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Marker: persistence backends serialize as backend IDs
// ════════════════════════════════════════════════════════════════════════════

/// Stats hook: emit one of these from a service/client whenever a save
/// or load happens. Surfaced through `TipProofServiceStats` /
/// `TipProofClientStats` in a future revision.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PersistenceStats {
    pub total_saves: u64,
    pub total_loads: u64,
    pub total_save_failures: u64,
    pub total_load_failures: u64,
    pub last_backend_id: Option<String>,
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tip_proof_v2;
    use q_ivc::recursion::{LatticeStepProof, StepIO};
    use q_lattice_guard::{params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof};

    fn dummy_proof() -> LatticeTipProofV2 {
        tip_proof_v2::anchor(0, [0u8; 32])
    }

    fn dummy_step(z_in: StepIO, z_out: StepIO) -> LatticeStepProof {
        LatticeStepProof {
            proof: LatticeGuardProof {
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
            },
            z_in: z_in.pack(),
            z_out: z_out.pack(),
            public_input_count: 9,
        }
    }

    #[test]
    fn memory_save_and_load_round_trip() {
        let m = MemoryPersistence::new();
        assert!(m.load().unwrap().is_none(), "fresh backend is empty");

        let p = dummy_proof();
        m.save(&p).unwrap();

        let loaded = m.load().unwrap().expect("must be present");
        assert_eq!(loaded.tip_height, p.tip_height);
        assert_eq!(loaded.anchor_state, p.anchor_state);
    }

    #[test]
    fn memory_clear_makes_load_empty() {
        let m = MemoryPersistence::new();
        m.save(&dummy_proof()).unwrap();
        assert!(m.load().unwrap().is_some());
        m.clear().unwrap();
        assert!(m.load().unwrap().is_none());
    }

    #[test]
    fn memory_clone_shares_state() {
        let a = MemoryPersistence::new();
        let b = a.clone();
        a.save(&dummy_proof()).unwrap();
        // b sees a's write via the shared Arc<Mutex<...>>.
        assert!(b.load().unwrap().is_some());
    }

    #[test]
    fn memory_overwrites_on_repeated_save() {
        let m = MemoryPersistence::new();
        let mut p = tip_proof_v2::anchor(0, [0u8; 32]);
        m.save(&p).unwrap();
        let s = dummy_step(StepIO::new([0u8; 32], 0), StepIO::new([1u8; 32], 1));
        p = tip_proof_v2::extend_with_step_proof(&p, s).unwrap();
        m.save(&p).unwrap();
        assert_eq!(m.load().unwrap().unwrap().tip_height, 1);
    }

    #[test]
    fn memory_backend_id_is_constant() {
        let m = MemoryPersistence::new();
        assert_eq!(m.backend_id(), "memory");
    }

    #[test]
    fn memory_snapshot_bytes_matches_load() {
        let m = MemoryPersistence::new();
        let p = dummy_proof();
        m.save(&p).unwrap();

        let snapped = m.snapshot_bytes().expect("must be present");
        let loaded = m.load().unwrap().unwrap();
        let reserialized = bincode::serialize(&loaded).unwrap();
        assert_eq!(snapped, reserialized, "snapshot bytes equal re-serialised load");
    }

    // ── FilePersistence tests ──────────────────────────────────────────────

    /// Per-test unique tmpdir under the OS temp dir.
    fn tmp_path(test_name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "q-recursive-proofs-test-{}-{}.bin",
            test_name,
            std::process::id()
        ));
        // Clean any leftover from a previous run.
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn file_save_and_load_round_trip() {
        let path = tmp_path("file_save_and_load");
        let f = FilePersistence::new(&path);
        assert!(f.load().unwrap().is_none(), "missing file is empty");

        let p = dummy_proof();
        f.save(&p).unwrap();

        let loaded = f.load().unwrap().expect("present");
        assert_eq!(loaded.tip_height, p.tip_height);

        // Cleanup.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn file_atomic_write_leaves_no_tmp_residue() {
        let path = tmp_path("file_atomic_write");
        let f = FilePersistence::new(&path);
        f.save(&dummy_proof()).unwrap();

        // .tmp must NOT exist after a successful save.
        let tmp = f.tmp_path();
        assert!(!tmp.exists(), "tmp file must be cleaned up via rename");
        assert!(path.exists(), "committed file must exist");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn file_overwrite_is_atomic() {
        let path = tmp_path("file_overwrite");
        let f = FilePersistence::new(&path);

        let p0 = tip_proof_v2::anchor(0, [0u8; 32]);
        f.save(&p0).unwrap();
        let bytes_v0 = std::fs::read(&path).unwrap();

        let s = dummy_step(StepIO::new([0u8; 32], 0), StepIO::new([1u8; 32], 1));
        let p1 = tip_proof_v2::extend_with_step_proof(&p0, s).unwrap();
        f.save(&p1).unwrap();
        let bytes_v1 = std::fs::read(&path).unwrap();

        assert_ne!(bytes_v0, bytes_v1, "overwrite must change content");
        assert_eq!(f.load().unwrap().unwrap().tip_height, 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn file_clear_removes_file() {
        let path = tmp_path("file_clear");
        let f = FilePersistence::new(&path);
        f.save(&dummy_proof()).unwrap();
        assert!(path.exists());
        f.clear().unwrap();
        assert!(!path.exists());
        assert!(f.load().unwrap().is_none());
    }

    #[test]
    fn file_clear_on_missing_file_is_ok() {
        let path = tmp_path("file_clear_missing");
        let f = FilePersistence::new(&path);
        // Never saved — clear on missing must not error.
        f.clear().unwrap();
        f.clear().unwrap(); // idempotent
    }

    #[test]
    fn file_backend_id_includes_path() {
        let path = tmp_path("file_backend_id");
        let f = FilePersistence::new(&path);
        let id = f.backend_id();
        assert!(id.starts_with("file:"));
        assert!(id.contains(&path.display().to_string()));
    }

    #[test]
    fn file_creates_parent_directory_on_first_save() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "q-recursive-proofs-test-newdir-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);

        let mut full = path.clone();
        full.push("nested");
        full.push("tip_proof.bin");

        let f = FilePersistence::new(&full);
        f.save(&dummy_proof())
            .expect("save must create parent dirs");
        assert!(full.exists());

        let _ = std::fs::remove_dir_all(&path);
    }

    // ── trait-object usage ──────────────────────────────────────────────

    #[test]
    fn trait_object_dispatch_works() {
        let backends: Vec<Arc<dyn TipProofPersistence>> = vec![
            Arc::new(MemoryPersistence::new()),
            Arc::new(FilePersistence::new(tmp_path("trait_object"))),
        ];

        for b in &backends {
            let p = dummy_proof();
            b.save(&p).unwrap();
            assert!(b.load().unwrap().is_some());
            b.clear().unwrap();
            assert!(b.load().unwrap().is_none());
        }
    }

    #[test]
    fn load_or_warn_returns_none_on_empty_backend() {
        let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());
        let result = load_or_warn(&backend, "test_context");
        assert!(result.is_none());
    }

    #[test]
    fn load_or_warn_returns_proof_when_present() {
        let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());
        backend.save(&dummy_proof()).unwrap();
        let result = load_or_warn(&backend, "test_context");
        assert!(result.is_some());
    }

    #[test]
    fn save_with_retry_succeeds_on_healthy_backend() {
        let backend: Arc<dyn TipProofPersistence> = Arc::new(MemoryPersistence::new());
        save_with_retry(&backend, &dummy_proof()).expect("healthy backend never fails");
        assert!(backend.load().unwrap().is_some());
    }
}
