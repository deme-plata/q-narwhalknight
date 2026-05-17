//! Core Enforcer — Real CPU affinity enforcement via sched_setaffinity
//!
//! Issue #013: The orchestrator assigns core budgets to layers, but without
//! actual OS-level enforcement the kernel scheduler can freely move threads
//! between cores, defeating cache locality and isolation.
//!
//! This module provides `CoreEnforcer` which uses raw `libc::sched_setaffinity`
//! on Linux to pin threads to specific core sets. On non-Linux platforms it
//! logs a warning and degrades gracefully (advisory-only).

use crate::ComputeLayer;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Result of an affinity operation
#[derive(Debug, Clone, PartialEq)]
pub enum AffinityResult {
    /// Successfully pinned to the requested cores
    Enforced { cores: Vec<usize> },
    /// Affinity was released (reset to all cores)
    Released,
    /// Platform does not support enforcement; advisory only
    UnsupportedPlatform,
    /// The syscall failed with this OS error code
    Failed { errno: i32 },
    /// Empty core set requested — no-op
    EmptyCoreSet,
}

/// Tracks per-layer affinity enforcement state
pub struct CoreEnforcer {
    /// Which cores each layer is currently pinned to (empty = not pinned)
    layer_cores: HashMap<ComputeLayer, Vec<usize>>,
    /// Total cores on this system
    total_cores: usize,
}

impl CoreEnforcer {
    /// Create a new CoreEnforcer
    pub fn new() -> Self {
        let total_cores = num_cpus::get();
        info!(
            "🔧 [CORE ENFORCER] Initialized — {} logical cores available",
            total_cores
        );
        Self {
            layer_cores: HashMap::new(),
            total_cores,
        }
    }

    /// Total logical cores on this system
    pub fn total_cores(&self) -> usize {
        self.total_cores
    }

    /// Get the cores currently pinned for a given layer (empty if none)
    pub fn get_layer_cores(&self, layer: &ComputeLayer) -> &[usize] {
        self.layer_cores
            .get(layer)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Pin the **calling thread** to the specified cores for the given layer.
    ///
    /// On Linux this calls `sched_setaffinity` with a `cpu_set_t` containing
    /// all cores in `cores`. On other platforms it returns
    /// `AffinityResult::UnsupportedPlatform`.
    ///
    /// Cores that exceed the system's logical core count are silently skipped.
    pub fn enforce_layer_affinity(
        &mut self,
        layer: ComputeLayer,
        cores: &[usize],
    ) -> AffinityResult {
        if cores.is_empty() {
            return AffinityResult::EmptyCoreSet;
        }

        // Filter out cores beyond what the system actually has
        let valid_cores: Vec<usize> = cores
            .iter()
            .copied()
            .filter(|&c| c < self.total_cores)
            .collect();

        if valid_cores.is_empty() {
            warn!(
                "🔧 [CORE ENFORCER] All requested cores for {} are out of range (max={})",
                layer.name(),
                self.total_cores
            );
            return AffinityResult::EmptyCoreSet;
        }

        let result = set_thread_affinity(&valid_cores);

        match &result {
            AffinityResult::Enforced { cores: pinned } => {
                info!(
                    "🔧 [CORE ENFORCER] {} pinned to cores {:?} (count={})",
                    layer.name(),
                    pinned,
                    pinned.len()
                );
                self.layer_cores.insert(layer, pinned.clone());
            }
            AffinityResult::UnsupportedPlatform => {
                warn!(
                    "🔧 [CORE ENFORCER] {} — platform does not support sched_setaffinity, advisory only",
                    layer.name()
                );
                // Still record the intent for status reporting
                self.layer_cores.insert(layer, valid_cores);
            }
            AffinityResult::Failed { errno } => {
                warn!(
                    "🔧 [CORE ENFORCER] {} — sched_setaffinity failed (errno={}), falling back to advisory",
                    layer.name(),
                    errno
                );
            }
            _ => {}
        }

        result
    }

    /// Release affinity for a layer — resets the calling thread to be
    /// schedulable on all cores.
    pub fn release_affinity(&mut self, layer: ComputeLayer) -> AffinityResult {
        self.layer_cores.remove(&layer);

        let all_cores: Vec<usize> = (0..self.total_cores).collect();
        let result = set_thread_affinity(&all_cores);

        match &result {
            AffinityResult::Enforced { .. } => {
                debug!(
                    "🔧 [CORE ENFORCER] {} affinity released — thread free on all {} cores",
                    layer.name(),
                    self.total_cores
                );
                AffinityResult::Released
            }
            AffinityResult::UnsupportedPlatform => {
                debug!(
                    "🔧 [CORE ENFORCER] {} release — platform unsupported, no-op",
                    layer.name()
                );
                AffinityResult::Released
            }
            other => other.clone(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Linux implementation — real sched_setaffinity
// ═══════════════════════════════════════════════════════════════════

#[cfg(target_os = "linux")]
fn set_thread_affinity(cores: &[usize]) -> AffinityResult {
    use std::mem;

    unsafe {
        // Get the current thread ID via syscall (gettid is not always
        // exposed as a libc wrapper on all glibc versions)
        let tid = libc::syscall(libc::SYS_gettid) as libc::pid_t;

        // Zero-initialise the cpu_set_t
        let mut cpuset: libc::cpu_set_t = mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);

        // Set each requested core
        for &core in cores {
            // CPU_SET may panic/UB if core >= CPU_SETSIZE (typically 1024)
            if core < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(core, &mut cpuset);
            }
        }

        // Apply affinity
        let ret = libc::sched_setaffinity(
            tid,
            mem::size_of::<libc::cpu_set_t>(),
            &cpuset,
        );

        if ret == 0 {
            AffinityResult::Enforced {
                cores: cores.to_vec(),
            }
        } else {
            let errno = *libc::__errno_location();
            AffinityResult::Failed { errno }
        }
    }
}

/// Read the current affinity mask for the calling thread.
/// Returns the set of cores the thread is allowed to run on.
#[cfg(target_os = "linux")]
pub fn get_thread_affinity() -> Result<Vec<usize>, i32> {
    use std::mem;

    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::pid_t;
        let mut cpuset: libc::cpu_set_t = mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);

        let ret = libc::sched_getaffinity(
            tid,
            mem::size_of::<libc::cpu_set_t>(),
            &mut cpuset,
        );

        if ret != 0 {
            return Err(*libc::__errno_location());
        }

        let num_cpus = num_cpus::get();
        let mut cores = Vec::new();
        for i in 0..num_cpus.min(libc::CPU_SETSIZE as usize) {
            if libc::CPU_ISSET(i, &cpuset) {
                cores.push(i);
            }
        }
        Ok(cores)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Non-Linux fallback — graceful degradation
// ═══════════════════════════════════════════════════════════════════

#[cfg(not(target_os = "linux"))]
fn set_thread_affinity(_cores: &[usize]) -> AffinityResult {
    AffinityResult::UnsupportedPlatform
}

#[cfg(not(target_os = "linux"))]
pub fn get_thread_affinity() -> Result<Vec<usize>, i32> {
    // On non-Linux, return all cores as "allowed"
    Ok((0..num_cpus::get()).collect())
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_enforcer_creation() {
        let enforcer = CoreEnforcer::new();
        assert!(enforcer.total_cores() > 0);
        assert!(enforcer.get_layer_cores(&ComputeLayer::Mining).is_empty());
    }

    #[test]
    fn test_enforce_empty_core_set() {
        let mut enforcer = CoreEnforcer::new();
        let result = enforcer.enforce_layer_affinity(ComputeLayer::Mining, &[]);
        assert_eq!(result, AffinityResult::EmptyCoreSet);
    }

    #[test]
    fn test_enforce_out_of_range_cores() {
        let mut enforcer = CoreEnforcer::new();
        // Request cores way beyond what the system has
        let bogus = vec![99999, 100000, 100001];
        let result = enforcer.enforce_layer_affinity(ComputeLayer::Mining, &bogus);
        assert_eq!(result, AffinityResult::EmptyCoreSet);
    }

    #[test]
    fn test_release_without_prior_enforce() {
        let mut enforcer = CoreEnforcer::new();
        // Releasing a layer that was never pinned should not panic
        let result = enforcer.release_affinity(ComputeLayer::AiInference);
        // On Linux: Released, on other platforms: Released (from UnsupportedPlatform path)
        match result {
            AffinityResult::Released => {}
            AffinityResult::Failed { .. } => {
                // May fail in containers with restricted permissions — acceptable
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_enforce_and_release_core0() {
        let mut enforcer = CoreEnforcer::new();

        // Pin to core 0 (always exists on any system)
        let result = enforcer.enforce_layer_affinity(ComputeLayer::Mining, &[0]);
        match result {
            AffinityResult::Enforced { ref cores } => {
                assert_eq!(cores, &[0]);
            }
            AffinityResult::UnsupportedPlatform => {
                // Non-Linux — acceptable
            }
            AffinityResult::Failed { .. } => {
                // Container restrictions — acceptable
            }
            other => panic!("Unexpected result: {:?}", other),
        }

        // Verify layer tracking
        let tracked = enforcer.get_layer_cores(&ComputeLayer::Mining);
        assert!(!tracked.is_empty(), "Layer should be tracked after enforce");

        // Release
        let release = enforcer.release_affinity(ComputeLayer::Mining);
        match release {
            AffinityResult::Released => {}
            AffinityResult::Failed { .. } => {
                // Container restrictions — acceptable
            }
            other => panic!("Unexpected release result: {:?}", other),
        }

        // Layer tracking should be cleared
        assert!(enforcer.get_layer_cores(&ComputeLayer::Mining).is_empty());
    }

    #[test]
    fn test_enforce_multiple_cores() {
        let mut enforcer = CoreEnforcer::new();
        if enforcer.total_cores() < 2 {
            // Single-core system, skip multi-core test
            return;
        }

        let cores: Vec<usize> = (0..enforcer.total_cores().min(4)).collect();
        let result = enforcer.enforce_layer_affinity(ComputeLayer::AiInference, &cores);

        match result {
            AffinityResult::Enforced { cores: pinned } => {
                assert_eq!(pinned.len(), cores.len());
            }
            AffinityResult::UnsupportedPlatform | AffinityResult::Failed { .. } => {
                // Acceptable fallback
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_different_layers_independent() {
        let mut enforcer = CoreEnforcer::new();
        if enforcer.total_cores() < 4 {
            return;
        }

        // Pin Mining to cores 0-1
        enforcer.enforce_layer_affinity(ComputeLayer::Mining, &[0, 1]);
        // Pin AI to cores 2-3
        enforcer.enforce_layer_affinity(ComputeLayer::AiInference, &[2, 3]);

        let mining_cores = enforcer.get_layer_cores(&ComputeLayer::Mining);
        let ai_cores = enforcer.get_layer_cores(&ComputeLayer::AiInference);

        // Both should be tracked independently
        assert!(!mining_cores.is_empty());
        assert!(!ai_cores.is_empty());

        // Release mining only
        enforcer.release_affinity(ComputeLayer::Mining);
        assert!(enforcer.get_layer_cores(&ComputeLayer::Mining).is_empty());
        assert!(!enforcer.get_layer_cores(&ComputeLayer::AiInference).is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_linux_cpu_set_operations() {
        use std::mem;

        unsafe {
            let mut cpuset: libc::cpu_set_t = mem::zeroed();
            libc::CPU_ZERO(&mut cpuset);

            // Verify CPU_ZERO cleared everything
            assert!(
                !libc::CPU_ISSET(0, &cpuset),
                "CPU 0 should be clear after CPU_ZERO"
            );

            // Set core 0
            libc::CPU_SET(0, &mut cpuset);
            assert!(
                libc::CPU_ISSET(0, &cpuset),
                "CPU 0 should be set after CPU_SET"
            );
            assert!(
                !libc::CPU_ISSET(1, &cpuset),
                "CPU 1 should still be clear"
            );

            // Set core 1
            libc::CPU_SET(1, &mut cpuset);
            assert!(libc::CPU_ISSET(0, &cpuset));
            assert!(libc::CPU_ISSET(1, &cpuset));

            // Clear core 0
            libc::CPU_CLR(0, &mut cpuset);
            assert!(!libc::CPU_ISSET(0, &cpuset));
            assert!(libc::CPU_ISSET(1, &cpuset));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_linux_sched_getaffinity_reflects_set() {
        // Pin current thread to core 0, then read back and verify
        let mut enforcer = CoreEnforcer::new();

        let result = enforcer.enforce_layer_affinity(ComputeLayer::Mining, &[0]);
        if let AffinityResult::Enforced { .. } = result {
            // Read back with sched_getaffinity
            match get_thread_affinity() {
                Ok(affinity_cores) => {
                    assert!(
                        affinity_cores.contains(&0),
                        "After pinning to core 0, sched_getaffinity should report core 0"
                    );
                    // Should NOT contain cores we did not request (unless 1-core system)
                    if enforcer.total_cores() > 1 {
                        assert_eq!(
                            affinity_cores.len(),
                            1,
                            "Should be pinned to exactly 1 core, got {:?}",
                            affinity_cores
                        );
                    }
                }
                Err(errno) => {
                    // sched_getaffinity failed — container restriction
                    warn!("sched_getaffinity failed with errno={}, skipping assertion", errno);
                }
            }

            // Release and verify all cores are restored
            enforcer.release_affinity(ComputeLayer::Mining);
            if let Ok(all) = get_thread_affinity() {
                assert!(
                    all.len() >= enforcer.total_cores(),
                    "After release, thread should be free on all {} cores, got {}",
                    enforcer.total_cores(),
                    all.len()
                );
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_non_linux_graceful_degradation() {
        let mut enforcer = CoreEnforcer::new();
        let result = enforcer.enforce_layer_affinity(ComputeLayer::Mining, &[0, 1]);
        assert_eq!(result, AffinityResult::UnsupportedPlatform);

        // get_thread_affinity should return all cores
        let cores = get_thread_affinity().unwrap();
        assert_eq!(cores.len(), enforcer.total_cores());
    }
}
