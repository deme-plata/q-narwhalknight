//! Single-instance enforcement.
//!
//! Acquires a system-wide lock at startup. If another wallet is already running
//! this returns `None` and the caller should exit early (a follow-up commit will
//! add IPC to forward CLI args / quillon:// URLs to the running instance and
//! raise its window).
//!
//! Cross-platform via the `single-instance` crate (Windows named mutex, Linux
//! abstract socket, macOS named semaphore).

use single_instance::SingleInstance;

const LOCK_ID: &str = "quillon-wallet-single-instance-v1";

/// Holds the lock for the lifetime of the wallet process. Drop it to release.
pub struct InstanceLock(SingleInstance);

/// Try to acquire the single-instance lock.
///
/// Returns `Some(lock)` if this is the only running wallet. Returns `None` if
/// another wallet instance is already running.
pub fn acquire() -> Option<InstanceLock> {
    match SingleInstance::new(LOCK_ID) {
        Ok(instance) if instance.is_single() => Some(InstanceLock(instance)),
        Ok(_) => None,
        Err(e) => {
            eprintln!("[single-instance] WARN: failed to create lock ({e}); proceeding without enforcement");
            None
        }
    }
}
