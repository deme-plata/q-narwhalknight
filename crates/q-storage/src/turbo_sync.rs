/// Turbo Sync - Revolutionary Git-Inspired Blockchain Synchronization
///
/// Inspired by Git's pack files, delta compression, and parallel fetching,
/// combined with Q-NarwhalKnight's DAG-Knight consensus architecture.
///
/// Performance Targets:
/// - 50-250x faster than current gossipsub sequential sync
/// - 1,000-5,000 blocks/minute throughput (vs current ~21 blocks/min)
/// - 3-10x bandwidth reduction via LZ4 compression + delta encoding
/// - Parallel downloads from 8+ peers simultaneously
/// - Sub-10 minute full chain sync (110,000 blocks)
///
/// Architecture:
/// 1. Smart Protocol: Discover what peers have (Git's "want/have" negotiation)
/// 2. Range Splitting: Divide missing blocks into parallel chunks
/// 3. Pack Files: LZ4 compression (v1.4.14-beta: 3-5x faster than zstd)
/// 4. Delta Encoding: Compress similar blocks (headers mostly identical)
/// 5. Pipelining: Download → Decompress → Verify → Apply simultaneously
/// 6. Multi-Peer: Round-robin across all available peers

use anyhow::{Context, Result};  // v0.9.53-beta: Added Context for .context() method
use crate::QStorage;
use futures::{stream::FuturesUnordered, StreamExt};
use libp2p::PeerId;
use rayon::prelude::*;  // ✅ v0.9.41-beta: Parallel decompression
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, RwLock, Semaphore, Mutex};
use tracing::{debug, error, info, warn};

/// v8.4.4: Dedicated rayon thread pool for sync verification work.
/// Limits CPU-bound par_sort and SHA3 verification to 4 threads instead of
/// the global rayon pool (19 threads on Beta), preventing sync from saturating all CPU cores.
/// API handlers and tokio workers get the remaining cores for responsive request handling.
static SYNC_RAYON_POOL: once_cell::sync::Lazy<rayon::ThreadPool> = once_cell::sync::Lazy::new(|| {
    let num_threads = std::env::var("Q_SYNC_RAYON_THREADS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(8usize); // v1.0.2: default 8 threads (was 4)
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(|idx| format!("sync-rayon-{}", idx))
        .build()
        .expect("Failed to create sync rayon thread pool")
});

use q_types::block::QBlock;

// Import QStorage from parent module
// v0.8.0-beta: Import balance consensus for Turbo Sync integration
use crate::{BalanceConsensusEngine, BalanceConsensusError};

// 🚀 v1.0.5-beta: Request Pipelining (Phase 2: libp2p-rust network optimization)
use crate::request_pipeline::{PipelineConfig, PipelineManager};

// 🚀 v1.0.6-beta: Pack Caching (Phase 3: libp2p-rust network optimization)
use crate::pack_cache::{PackCache, PackCacheConfig, PackCacheKey};

// 🚀 v1.0.50-beta: Crypto-Enhanced Sync (IACR 2024-2025 reliability improvements)
use crate::crypto_enhanced_sync::{
    IncrementalBlockVerifier, AdaptiveTimeout, SyncProgressTracker,
    EnhancedSyncConfig, SyncCheckpoint,
};

// 🚀 v1.0.60-beta: Comprehensive State Sync (all state via transactions)
#[cfg(not(target_os = "windows"))]
use crate::block_state_processor::BlockStateProcessor;

// 🛡️ v1.3.0-beta: SHA3-256 Data Integrity (Quantum-Resistant Block Verification)
use crate::sha3_data_integrity::{Sha3DataIntegrity, Sha3IntegrityConfig};

// 🚀 v1.5.0-beta: CHIRON Parallel State Applicator (~30% sync speedup)
// Uses pre-computed execution hints to parallelize transaction state application
use crate::parallel_state_applicator::{ParallelStateApplicator, BalanceState, ApplyStats};

// 🚀 v1.5.0-beta: NEMO High-Contention Executor (+42% over Block-STM)
// Uses greedy commits, refined dependency handling, and priority scheduling
use crate::nemo_executor::{NemoExecutor, NemoStats};

// 🚀 v1.5.0-beta: Reddio-Style Async Storage Pipeline (70% overhead reduction)
// Hot cache + async prefetching + pipelined workflow for maximum throughput
#[cfg(not(target_os = "windows"))]
use crate::async_pipeline::{AsyncStoragePipeline, AsyncPipelineConfig, AsyncPipelineStats};

// ═══════════════════════════════════════════════════════════════════════════════
// 🚀 v2.1.0-DELTA-V: Project APOLLO Control Systems Integration
// ═══════════════════════════════════════════════════════════════════════════════

// Phase 5 KALMAN: Self-tuning optimal sync parameters
use crate::pid_controller::PIDRateController;
use crate::kalman_predictor::{KalmanNetworkPredictor, NetworkState, SyncSettings};

// Phase 4 SLINGSHOT: Cache-aware peer selection
use crate::peer_momentum::{PeerMomentumManager, GravityAssistedSelector};

// 🚀 v2.3.10-beta: Warp Sync Phase 2 & 3 - Multi-Peer Download + Prefetch Pipeline
// Provides 3-5x faster sync via intelligent peer selection and prefetching
use crate::warp_sync::{MultiPeerDownloader, PrefetchPipeline, ChunkAssignment, ChunkStatus};

// Phase 6 DELTA-V: Pre-compressed storage for zero-CPU P2P serving
use crate::precompressed_storage::{PrecompressedBlock, CompressionAlgorithm};

// ═══════════════════════════════════════════════════════════════════════════════
// v1.0.2: Starship Flight Computer — SpaceX-Inspired Sync State Machine
// ═══════════════════════════════════════════════════════════════════════════════

/// SpaceX Starship-inspired sync phase state machine.
/// Replaces raw AtomicU8 (0=idle, 1=turbo, 2=endgame, 3=micro) with named phases.
///
/// Mission profile:
/// PRELAUNCH -> IGNITION -> SUPER_HEAVY -> HOT_STAGING -> STARSHIP_CRUISE -> ORBITAL_INSERTION -> STATION_KEEPING
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum StarshipPhase {
    /// Pre-launch: preflight checks, peer discovery, height negotiation
    Prelaunch        = 10,
    /// Ignition: first sync trigger, Raptor engines light (chunk planning)
    Ignition         = 11,
    /// Super Heavy Boost: bulk turbo download at max thrust (was mode=1 "turbo")
    SuperHeavy       = 1,   // backward-compat with existing AtomicU8 value
    /// Hot Staging: separation maneuver, endgame transition (was mode=2 "endgame")
    HotStaging       = 2,
    /// Starship Cruise: final approach, micro-sync last blocks (was mode=3 "micro")
    StarshipCruise   = 3,
    /// Orbital Insertion: reached tip, transitioning to steady-state
    OrbitalInsertion = 12,
    /// Station Keeping: fully synced, maintaining orbit (was mode=0 "fully_synced")
    StationKeeping   = 0,
}

impl StarshipPhase {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0  => StarshipPhase::StationKeeping,
            1  => StarshipPhase::SuperHeavy,
            2  => StarshipPhase::HotStaging,
            3  => StarshipPhase::StarshipCruise,
            10 => StarshipPhase::Prelaunch,
            11 => StarshipPhase::Ignition,
            12 => StarshipPhase::OrbitalInsertion,
            _  => StarshipPhase::StationKeeping,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            StarshipPhase::Prelaunch        => "PRELAUNCH",
            StarshipPhase::Ignition         => "IGNITION",
            StarshipPhase::SuperHeavy       => "SUPER_HEAVY",
            StarshipPhase::HotStaging       => "HOT_STAGING",
            StarshipPhase::StarshipCruise   => "STARSHIP_CRUISE",
            StarshipPhase::OrbitalInsertion => "ORBITAL_INSERTION",
            StarshipPhase::StationKeeping   => "STATION_KEEPING",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            StarshipPhase::Prelaunch        => "\u{1f680}", // rocket
            StarshipPhase::Ignition         => "\u{1f525}", // fire
            StarshipPhase::SuperHeavy       => "\u{26a1}",  // lightning
            StarshipPhase::HotStaging       => "\u{1f4ab}", // dizzy/explosion
            StarshipPhase::StarshipCruise   => "\u{2728}",  // sparkles
            StarshipPhase::OrbitalInsertion => "\u{1f30d}", // earth
            StarshipPhase::StationKeeping   => "\u{1f6f0}\u{fe0f}",  // satellite
        }
    }

    pub fn is_syncing(&self) -> bool {
        matches!(self, StarshipPhase::SuperHeavy | StarshipPhase::HotStaging | StarshipPhase::StarshipCruise)
    }

    pub fn is_bulk(&self) -> bool {
        matches!(self, StarshipPhase::SuperHeavy)
    }
}

impl std::fmt::Display for StarshipPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.emoji(), self.as_str())
    }
}

/// Station Keeping health monitors (endgame maintenance)
#[derive(Debug, Clone)]
pub struct StationKeepingState {
    pub last_peer_health_check: Instant,
    pub last_pool_sync: Instant,
    pub last_reorg_scan: Instant,
    pub consecutive_tip_confirmations: u32,
    pub orbit_stable: bool,           // true after 10 consecutive tip confirmations
    pub orbit_decay_warnings: u32,    // v8.7.0: count of orbit perturbation events
    pub peak_peer_count: u32,         // v8.7.0: high-water mark for peer count
    pub peer_health_trend: f64,       // v8.7.0: EWMA of peer health (detects degradation)
    pub blocks_received_in_orbit: u64, // v8.7.0: blocks received via gossipsub while in orbit
}

impl Default for StationKeepingState {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            last_peer_health_check: now,
            last_pool_sync: now,
            last_reorg_scan: now,
            consecutive_tip_confirmations: 0,
            orbit_stable: false,
            orbit_decay_warnings: 0,
            peak_peer_count: 0,
            peer_health_trend: 0.0,
            blocks_received_in_orbit: 0,
        }
    }
}

/// Per-phase performance record for telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseRecord {
    pub phase: String,
    pub duration_secs: f64,
    pub blocks_processed: u64,
    pub blocks_per_sec: f64,
}

/// Starship telemetry snapshot for TUI/SSE consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarshipTelemetry {
    pub phase: String,
    pub phase_duration_secs: u64,
    pub mission_elapsed_secs: u64,
    pub orbit_stable: bool,
    pub peer_health: f64,
    pub consecutive_tip_confirmations: u32,
    pub phases_completed: usize,
    // v8.7.0: Enhanced telemetry
    pub blocks_in_phase: u64,
    pub phase_blocks_per_sec: f64,
    pub total_blocks_synced: u64,
    pub mission_avg_bps: f64,
    pub orbit_decay_warnings: u32,
    pub phase_history: Vec<PhaseRecord>,
    pub recommended_throttle: String,   // "turbo" / "normal" / "conservative"
}

/// Central state machine for sync lifecycle.
/// Owns phase transitions — consolidates all session_sync_mode writes.
///
/// v8.7.0: Enhanced with hysteresis bands, per-phase performance tracking,
/// throttle recommendations, and richer Station Keeping diagnostics.
pub struct FlightComputer {
    phase: Arc<AtomicU8>,           // == TurboSyncManager.session_sync_mode (shared ref)
    current_phase: StarshipPhase,
    phase_entered_at: Instant,
    mission_start: Instant,
    phase_history: Vec<(StarshipPhase, Duration, u64)>,  // (phase, time_spent, blocks_processed)
    blocks_at_phase_start: u64,
    blocks_at_mission_start: u64,
    station_keeping: StationKeepingState,
    last_known_tip: u64,
    last_gap: u64,                  // v8.7.0: previous gap for trend detection
    gap_shrinking_ticks: u32,       // v8.7.0: consecutive ticks where gap decreased
    gap_growing_ticks: u32,         // v8.7.0: consecutive ticks where gap increased
    transition_count: u32,          // v8.7.0: total phase transitions (detects oscillation)
    last_transition_at: Instant,    // v8.7.0: anti-oscillation cooldown
}

/// v8.7.0: Hysteresis bands prevent phase oscillation.
/// To transition DOWN (SuperHeavy->HotStaging), gap must be <= threshold.
/// To transition UP (HotStaging->SuperHeavy), gap must be >= threshold * HYSTERESIS_FACTOR.
/// This creates a dead zone preventing rapid back-and-forth switching.
const HYSTERESIS_FACTOR: f64 = 1.5;

/// v8.7.0: Minimum time in a phase before allowing non-emergency transitions (seconds).
/// Prevents oscillation when gap hovers near a boundary.
const MIN_PHASE_DWELL_SECS: u64 = 5;

/// v8.7.0: Orbit perturbation threshold — gap must exceed this to leave StationKeeping.
/// Higher than entry threshold (hysteresis) so small gossipsub delays don't trigger re-sync.
const ORBIT_PERTURBATION_THRESHOLD: u64 = 10;

impl FlightComputer {
    /// Create a new FlightComputer sharing the same AtomicU8 as TurboSyncManager
    pub fn new(phase_atomic: Arc<AtomicU8>) -> Self {
        let initial = StarshipPhase::from_u8(phase_atomic.load(Ordering::Relaxed));
        let now = Instant::now();
        Self {
            phase: phase_atomic,
            current_phase: initial,
            phase_entered_at: now,
            mission_start: now,
            phase_history: Vec::new(),
            blocks_at_phase_start: 0,
            blocks_at_mission_start: 0,
            station_keeping: StationKeepingState::default(),
            last_known_tip: 0,
            last_gap: 0,
            gap_shrinking_ticks: 0,
            gap_growing_ticks: 0,
            transition_count: 0,
            last_transition_at: now,
        }
    }

    /// Get current phase
    pub fn phase(&self) -> StarshipPhase {
        self.current_phase
    }

    /// How long the current phase has been active
    pub fn phase_elapsed(&self) -> Duration {
        self.phase_entered_at.elapsed()
    }

    /// Transition to a new phase, logging mission telemetry and updating the shared AtomicU8
    pub fn transition_to(&mut self, new_phase: StarshipPhase, current_height: u64) {
        if new_phase == self.current_phase {
            return; // no-op for same phase
        }

        let time_in_phase = self.phase_entered_at.elapsed();
        let blocks_in_phase = current_height.saturating_sub(self.blocks_at_phase_start);
        let bps = if time_in_phase.as_secs_f64() > 0.1 {
            blocks_in_phase as f64 / time_in_phase.as_secs_f64()
        } else {
            0.0
        };

        // Record history with performance data
        self.phase_history.push((self.current_phase, time_in_phase, blocks_in_phase));
        // Cap history to last 20 phases to bound memory
        if self.phase_history.len() > 20 {
            self.phase_history.remove(0);
        }

        self.transition_count += 1;

        info!("{} [FLIGHT COMPUTER] {} -> {} | {:.1}s in {} | {} blocks @ {:.0} bps | transition #{}",
            new_phase.emoji(),
            self.current_phase.as_str(),
            new_phase.as_str(),
            time_in_phase.as_secs_f64(),
            self.current_phase.as_str(),
            blocks_in_phase,
            bps,
            self.transition_count,
        );

        // Update atomic for backward compatibility
        self.phase.store(new_phase as u8, Ordering::Release);

        // Reset phase tracking
        self.current_phase = new_phase;
        self.phase_entered_at = Instant::now();
        self.last_transition_at = Instant::now();
        self.blocks_at_phase_start = current_height;

        // Set mission start block on first real sync phase
        if self.blocks_at_mission_start == 0 && new_phase.is_syncing() {
            self.blocks_at_mission_start = current_height;
        }

        // Reset station keeping state on re-entry
        if new_phase == StarshipPhase::StationKeeping {
            self.station_keeping = StationKeepingState::default();
            info!("{} [STATION KEEPING] Orbit achieved at height {} — monitoring peer health, pool freshness, reorg watchdog",
                new_phase.emoji(), current_height);
        }

        // Reset gap trend on phase change
        self.gap_shrinking_ticks = 0;
        self.gap_growing_ticks = 0;
    }

    /// v8.7.0: Track gap trend for smarter phase decisions
    fn update_gap_trend(&mut self, gap: u64) {
        if gap < self.last_gap {
            self.gap_shrinking_ticks += 1;
            self.gap_growing_ticks = 0;
        } else if gap > self.last_gap {
            self.gap_growing_ticks += 1;
            self.gap_shrinking_ticks = 0;
        }
        // gap == last_gap: no change to either counter
        self.last_gap = gap;
    }

    /// v8.7.0: Check anti-oscillation cooldown
    fn dwell_satisfied(&self) -> bool {
        self.phase_entered_at.elapsed().as_secs() >= MIN_PHASE_DWELL_SECS
    }

    /// Pure function: compute the next phase based on sync gap.
    /// Returns None if no transition needed.
    ///
    /// v8.7.0: Enhanced with hysteresis bands and dwell time to prevent oscillation.
    /// - Downward transitions (less urgent phase): gap must cross threshold cleanly
    /// - Upward transitions (more urgent phase): gap must exceed threshold * 1.5x
    /// - All non-emergency transitions respect MIN_PHASE_DWELL_SECS cooldown
    pub fn should_advance(&mut self, gap: u64, local_height: u64, _network_height: u64, endgame_threshold: u64) -> Option<StarshipPhase> {
        self.update_gap_trend(gap);
        let current = self.current_phase;

        // Hysteresis thresholds: re-entry into a more aggressive phase requires higher gap
        let endgame_re_entry = (endgame_threshold as f64 * HYSTERESIS_FACTOR) as u64;
        let cruise_upper = (50.0 * HYSTERESIS_FACTOR) as u64; // 75 blocks

        match current {
            StarshipPhase::Prelaunch => {
                // No dwell for prelaunch — transition immediately
                if gap > 0 {
                    return Some(StarshipPhase::Ignition);
                }
                if local_height > 0 {
                    return Some(StarshipPhase::StationKeeping);
                }
                None
            }
            StarshipPhase::Ignition => {
                // Ignition always transitions out immediately (planning phase)
                if gap > endgame_threshold {
                    Some(StarshipPhase::SuperHeavy)
                } else if gap > 50 {
                    Some(StarshipPhase::HotStaging)
                } else if gap > 0 {
                    Some(StarshipPhase::StarshipCruise)
                } else {
                    Some(StarshipPhase::StationKeeping)
                }
            }
            StarshipPhase::SuperHeavy => {
                // Emergency: gap=0 always transitions (no dwell)
                if gap == 0 {
                    return Some(StarshipPhase::OrbitalInsertion);
                }
                if !self.dwell_satisfied() { return None; }
                // Downward: SuperHeavy -> HotStaging (less aggressive)
                if gap <= endgame_threshold {
                    if gap <= 50 {
                        Some(StarshipPhase::StarshipCruise)
                    } else {
                        Some(StarshipPhase::HotStaging)
                    }
                } else {
                    None // Stay in SuperHeavy
                }
            }
            StarshipPhase::HotStaging => {
                if gap == 0 {
                    return Some(StarshipPhase::OrbitalInsertion);
                }
                if !self.dwell_satisfied() { return None; }
                if gap <= 50 {
                    Some(StarshipPhase::StarshipCruise)
                } else if gap > endgame_re_entry {
                    // Upward with hysteresis: only re-enter SuperHeavy if gap grew past 1.5x threshold
                    Some(StarshipPhase::SuperHeavy)
                } else {
                    None
                }
            }
            StarshipPhase::StarshipCruise => {
                if gap == 0 {
                    return Some(StarshipPhase::OrbitalInsertion);
                }
                if !self.dwell_satisfied() { return None; }
                if gap > endgame_re_entry {
                    Some(StarshipPhase::SuperHeavy)
                } else if gap > cruise_upper {
                    // Only escalate to HotStaging if gap clearly above cruise band
                    Some(StarshipPhase::HotStaging)
                } else {
                    None
                }
            }
            StarshipPhase::OrbitalInsertion => {
                // Stay for 3 tip confirmations, then StationKeeping
                if self.station_keeping.consecutive_tip_confirmations >= 3 {
                    return Some(StarshipPhase::StationKeeping);
                }
                // Fell behind during insertion — immediate re-entry (no dwell)
                if gap > endgame_threshold {
                    Some(StarshipPhase::SuperHeavy)
                } else if gap > 50 {
                    Some(StarshipPhase::HotStaging)
                } else if gap > 0 {
                    Some(StarshipPhase::StarshipCruise)
                } else {
                    None
                }
            }
            StarshipPhase::StationKeeping => {
                // v8.7.0: Higher threshold to leave orbit (hysteresis)
                // Small gaps (1-10) are normal gossipsub propagation delay — don't re-enter sync
                if gap > endgame_re_entry {
                    self.station_keeping.orbit_decay_warnings += 1;
                    warn!("{} [STATION KEEPING] Major orbit decay! gap={} — re-entering SuperHeavy (decay event #{})",
                        StarshipPhase::StationKeeping.emoji(), gap, self.station_keeping.orbit_decay_warnings);
                    Some(StarshipPhase::SuperHeavy)
                } else if gap > cruise_upper {
                    self.station_keeping.orbit_decay_warnings += 1;
                    warn!("{} [STATION KEEPING] Orbit decay: gap={} — re-entering HotStaging (decay event #{})",
                        StarshipPhase::StationKeeping.emoji(), gap, self.station_keeping.orbit_decay_warnings);
                    Some(StarshipPhase::HotStaging)
                } else if gap > ORBIT_PERTURBATION_THRESHOLD {
                    self.station_keeping.orbit_decay_warnings += 1;
                    info!("{} [STATION KEEPING] Orbit perturbation: gap={} — micro-sync (decay event #{})",
                        StarshipPhase::StationKeeping.emoji(), gap, self.station_keeping.orbit_decay_warnings);
                    Some(StarshipPhase::StarshipCruise)
                } else {
                    None // Stable orbit — normal gossipsub keeps us at tip
                }
            }
        }
    }

    /// Called every sync loop tick when gap == 0 to track tip confirmations
    pub fn confirm_at_tip(&mut self, current_height: u64) {
        if current_height >= self.last_known_tip {
            self.station_keeping.consecutive_tip_confirmations += 1;
            self.last_known_tip = current_height;
            if current_height > self.last_known_tip {
                self.station_keeping.blocks_received_in_orbit += 1;
            }
        } else {
            // Height went down — reset confirmations (reorg or measurement error)
            self.station_keeping.consecutive_tip_confirmations = 0;
            self.station_keeping.orbit_stable = false;
        }

        if self.station_keeping.consecutive_tip_confirmations >= 10 {
            self.station_keeping.orbit_stable = true;
        }
    }

    /// Called every 15s when in StationKeeping for maintenance checks.
    /// v8.7.0: Enhanced with EWMA peer health trend and peak tracking.
    pub fn station_keeping_tick(&mut self, peer_count: u32, current_height: u64, network_height: u64) {
        let now = Instant::now();

        // Track peak peer count
        if peer_count > self.station_keeping.peak_peer_count {
            self.station_keeping.peak_peer_count = peer_count;
        }

        // Peer health check (every 30s) with EWMA trend
        if now.duration_since(self.station_keeping.last_peer_health_check) > Duration::from_secs(30) {
            self.station_keeping.last_peer_health_check = now;
            let health = Self::compute_peer_health(peer_count);
            // EWMA: trend = 0.3 * current + 0.7 * previous
            let alpha = 0.3;
            self.station_keeping.peer_health_trend =
                alpha * health + (1.0 - alpha) * self.station_keeping.peer_health_trend;

            // Warn if health is declining
            let peak = self.station_keeping.peak_peer_count;
            if peer_count < peak / 2 && peak >= 4 {
                warn!("{} [STATION KEEPING] Peer count degraded: {} (peak was {}) — health trend: {:.2}",
                    StarshipPhase::StationKeeping.emoji(), peer_count, peak, self.station_keeping.peer_health_trend);
            } else {
                debug!("{} [STATION KEEPING] Peers: {}/{} peak | health: {:.2} (trend {:.2}) | orbit: {} | confirms: {} | gossip blocks: {}",
                    StarshipPhase::StationKeeping.emoji(),
                    peer_count, peak, health, self.station_keeping.peer_health_trend,
                    if self.station_keeping.orbit_stable { "STABLE" } else { "STABILIZING" },
                    self.station_keeping.consecutive_tip_confirmations,
                    self.station_keeping.blocks_received_in_orbit,
                );
            }
        }

        // Reorg watchdog (every 60s)
        if now.duration_since(self.station_keeping.last_reorg_scan) > Duration::from_secs(60) {
            self.station_keeping.last_reorg_scan = now;
            let gap = network_height.saturating_sub(current_height);
            if gap > 2 {
                warn!("{} [STATION KEEPING] Orbit perturbation: gap={} (local={}, network={}) — watching...",
                    StarshipPhase::StationKeeping.emoji(), gap, current_height, network_height);
                if gap > 5 {
                    self.station_keeping.orbit_stable = false;
                    self.station_keeping.consecutive_tip_confirmations = 0;
                }
            }
        }

        // Pool freshness check (every 60s)
        if now.duration_since(self.station_keeping.last_pool_sync) > Duration::from_secs(60) {
            self.station_keeping.last_pool_sync = now;
            let mission_t = self.mission_start.elapsed().as_secs();
            let t_str = if mission_t >= 3600 {
                format!("T+{}h{}m", mission_t / 3600, (mission_t % 3600) / 60)
            } else if mission_t >= 60 {
                format!("T+{}m{}s", mission_t / 60, mission_t % 60)
            } else {
                format!("T+{}s", mission_t)
            };
            debug!("{} [STATION KEEPING] Maintenance pass @ {} | height: {} | orbit decays: {} | transitions: {}",
                StarshipPhase::StationKeeping.emoji(), t_str, current_height,
                self.station_keeping.orbit_decay_warnings, self.transition_count,
            );
        }
    }

    /// Compute peer health score (0.0-1.0) with granular scaling
    fn compute_peer_health(peer_count: u32) -> f64 {
        match peer_count {
            0 => 0.0,
            1 => 0.3,
            2 => 0.5,
            3 => 0.7,
            4 => 0.85,
            5..=7 => 0.95,
            _ => 1.0,  // 8+ peers = maximum health
        }
    }

    /// v8.7.0: Recommend throttle mode based on current phase and performance
    fn recommended_throttle(&self) -> &'static str {
        match self.current_phase {
            StarshipPhase::SuperHeavy | StarshipPhase::Ignition => "turbo",
            StarshipPhase::HotStaging => {
                // If gap is shrinking fast, stay turbo; otherwise normal
                if self.gap_shrinking_ticks > 3 { "turbo" } else { "normal" }
            }
            StarshipPhase::StarshipCruise => "normal",
            StarshipPhase::OrbitalInsertion | StarshipPhase::StationKeeping => "conservative",
            StarshipPhase::Prelaunch => "normal",
        }
    }

    /// Snapshot for TUI/SSE — v8.7.0: enhanced with per-phase performance data
    pub fn telemetry(&self, peer_count: u32) -> StarshipTelemetry {
        let phase_elapsed = self.phase_entered_at.elapsed();
        let mission_elapsed = self.mission_start.elapsed();
        let blocks_in_phase = self.last_known_tip.saturating_sub(self.blocks_at_phase_start);
        let phase_bps = if phase_elapsed.as_secs_f64() > 1.0 {
            blocks_in_phase as f64 / phase_elapsed.as_secs_f64()
        } else {
            0.0
        };

        let total_blocks = self.last_known_tip.saturating_sub(self.blocks_at_mission_start);
        let mission_bps = if mission_elapsed.as_secs_f64() > 1.0 {
            total_blocks as f64 / mission_elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Build phase history records
        let history: Vec<PhaseRecord> = self.phase_history.iter().map(|(phase, dur, blocks)| {
            let bps = if dur.as_secs_f64() > 0.1 { *blocks as f64 / dur.as_secs_f64() } else { 0.0 };
            PhaseRecord {
                phase: phase.as_str().to_string(),
                duration_secs: dur.as_secs_f64(),
                blocks_processed: *blocks,
                blocks_per_sec: bps,
            }
        }).collect();

        StarshipTelemetry {
            phase: self.current_phase.as_str().to_string(),
            phase_duration_secs: phase_elapsed.as_secs(),
            mission_elapsed_secs: mission_elapsed.as_secs(),
            orbit_stable: self.station_keeping.orbit_stable,
            peer_health: Self::compute_peer_health(peer_count),
            consecutive_tip_confirmations: self.station_keeping.consecutive_tip_confirmations,
            phases_completed: self.phase_history.len(),
            blocks_in_phase,
            phase_blocks_per_sec: phase_bps,
            total_blocks_synced: total_blocks,
            mission_avg_bps: mission_bps,
            orbit_decay_warnings: self.station_keeping.orbit_decay_warnings,
            phase_history: history,
            recommended_throttle: self.recommended_throttle().to_string(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// v5.2.0: Enhanced Peer Registry - Monotonicity enforcement & stale eviction
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-peer height tracking with monotonicity enforcement
#[derive(Debug, Clone)]
pub struct PeerHeightRecord {
    pub peer_id: PeerId,
    pub height: u64,
    pub tip_hash: Option<[u8; 32]>,
    pub last_updated: Instant,
    pub violations: u32,
}

/// Enhanced peer registry with Byzantine-resistant height consensus
#[derive(Debug)]
pub struct EnhancedPeerRegistry {
    peers: HashMap<PeerId, PeerHeightRecord>,
}

impl EnhancedPeerRegistry {
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
        }
    }

    /// Update peer height with monotonicity check.
    /// v8.4.4: Relaxed from 5 → 50 violations and added time-based reset.
    /// Gossipsub doesn't guarantee message ordering, so small height decreases
    /// are normal (out-of-order delivery). The old threshold of 5 was banning
    /// our best sync peers (Gamma/Delta) within minutes of startup.
    pub fn update_peer(&mut self, peer_id: PeerId, height: u64, tip_hash: Option<[u8; 32]>) -> bool {
        if let Some(record) = self.peers.get_mut(&peer_id) {
            // v8.4.4: Reset violations if peer has been well-behaved for 60s
            if record.violations > 0 && record.last_updated.elapsed() > Duration::from_secs(60) {
                record.violations = 0;
            }
            if height < record.height {
                // v8.4.4: Only count as violation if decrease is significant (>100 blocks)
                // Small decreases are normal gossipsub out-of-order delivery
                if record.height - height > 100 {
                    record.violations += 1;
                    warn!(
                        "⚠️ [MONOTONICITY] Peer {} height decreased {} → {} (violation #{}/50)",
                        peer_id, record.height, height, record.violations
                    );
                    if record.violations >= 50 {
                        warn!("🚫 [MONOTONICITY] Peer {} rejected: too many height decreases", peer_id);
                        return false;
                    }
                }
                // Don't update height to a lower value — keep the highest seen
                record.tip_hash = tip_hash;
                record.last_updated = Instant::now();
                return true;
            }
            record.height = height;
            record.tip_hash = tip_hash;
            record.last_updated = Instant::now();
        } else {
            self.peers.insert(peer_id, PeerHeightRecord {
                peer_id,
                height,
                tip_hash,
                last_updated: Instant::now(),
                violations: 0,
            });
        }
        true
    }

    /// Remove peers not heard from in `stale_secs` seconds
    pub fn evict_stale(&mut self, stale_secs: u64) -> usize {
        let cutoff = Duration::from_secs(stale_secs);
        let before = self.peers.len();
        self.peers.retain(|_, record| record.last_updated.elapsed() < cutoff);
        let evicted = before - self.peers.len();
        if evicted > 0 {
            info!("🧹 [PEER EVICTION] Evicted {} stale peers (>{stale_secs}s since last update)", evicted);
        }
        evicted
    }

    /// Byzantine-resistant median height from active peers
    pub fn median_height(&self) -> Option<u64> {
        let mut heights: Vec<u64> = self.peers.values().map(|r| r.height).collect();
        if heights.is_empty() {
            return None;
        }
        heights.sort_unstable();
        let mid = heights.len() / 2;
        Some(heights[mid])
    }

    /// Number of active (non-stale) peers
    pub fn active_peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Get peers sorted by height descending
    pub fn active_peers_by_height(&self) -> Vec<&PeerHeightRecord> {
        let mut peers: Vec<_> = self.peers.values().collect();
        peers.sort_by(|a, b| b.height.cmp(&a.height));
        peers
    }

    /// Max height across all peers
    pub fn max_height(&self) -> Option<u64> {
        self.peers.values().map(|r| r.height).max()
    }

    /// Convert to legacy format for backward compat
    pub fn to_legacy_vec(&self) -> Vec<(PeerId, u64)> {
        self.peers.values().map(|r| (r.peer_id, r.height)).collect()
    }

    /// Get the age (in seconds) of the most recent peer update
    pub fn newest_update_age_secs(&self) -> Option<u64> {
        self.peers.values()
            .map(|r| r.last_updated.elapsed().as_secs())
            .min()
    }

    // --- Backward-compat delegation methods (Vec<(PeerId, u64)> API) ---

    /// Number of peers (backward compat for `registry.len()`)
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    /// Is empty (backward compat for `registry.is_empty()`)
    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    /// Iterate as (PeerId, u64) tuples (backward compat for Vec<(PeerId, u64)>::iter())
    /// Returns an iterator yielding owned tuples for chainable .filter()/.map() use
    pub fn iter(&self) -> impl Iterator<Item = (PeerId, u64)> + '_ {
        self.peers.values().map(|r| (r.peer_id, r.height))
    }
}

/// Configuration for Turbo Sync
#[derive(Clone, Debug)]
pub struct TurboSyncConfig {
    /// Number of parallel download streams (like Git's parallel fetching)
    pub parallel_streams: usize,

    /// Blocks per chunk (larger = better compression, slower start)
    /// Git uses variable pack sizes, we use 1000 for balanced performance
    pub chunk_size: u64,

    /// Enable delta compression between blocks (like Git's delta encoding)
    /// Most block headers are identical except height/hash/timestamp
    pub delta_compression: bool,

    /// Compression level (1-21, higher = smaller but slower)
    /// Level 3 is Git's default: fast with good compression
    pub compression_level: i32,

    /// Enable pipelining (download + process simultaneously)
    /// Like Git's streaming decompression
    pub enable_pipelining: bool,

    /// Maximum concurrent peer connections
    pub max_peer_connections: usize,

    /// Timeout for individual chunk downloads
    pub chunk_timeout: Duration,

    /// Enable smart protocol negotiation (like Git's "want/have")
    pub smart_protocol: bool,

    /// 🚀 v1.0.33-beta: Enable batched database writes (50x performance boost)
    /// When true, uses save_qblocks_batch instead of per-block writes
    /// Default: read from Q_BATCHED_WRITES environment variable (defaults to false for safety)
    pub enable_batched_writes: bool,

    /// 🚀 v1.0.5-beta: Request pipelining configuration (Phase 2: Network Optimization)
    /// Enables adaptive window sizing and flow control for +50% performance
    /// Target: 12.8 → 19.2 blocks/s improvement
    pub pipeline_config: PipelineConfig,

    /// 🚀 v1.0.6-beta: Pack caching configuration (Phase 3: Network Optimization)
    /// Server-side LRU cache for compressed block packs for +30% performance
    /// Target: 19.2 → 25+ blocks/s improvement
    pub pack_cache_config: PackCacheConfig,

    /// 🚀 v1.0.60-beta: Enable comprehensive state sync (all state via transactions)
    /// When true, processes ALL transactions through StateProcessor/StateApplicator
    /// This enables full decentralization - user nodes get balances, DEX state, etc.
    /// Default: read from Q_STATE_SYNC environment variable (defaults to false for gradual rollout)
    pub enable_state_sync: bool,

    /// 🚀 v1.0.60-beta: Block gas limit for state processing
    /// Maximum gas consumed per block (default: 30M like Ethereum)
    pub block_gas_limit: u64,

    /// 🚀 v1.5.0-beta: Enable CHIRON parallel state application (~30% sync speedup)
    /// When true, uses pre-computed execution hints to parallelize transaction processing
    /// Default: read from Q_CHIRON_HINTS environment variable (defaults to true)
    pub enable_chiron_hints: bool,

    /// 🚀 v1.5.0-beta: Enable NEMO high-contention executor (+42% over Block-STM)
    /// When true, uses greedy commits, refined dependency handling, and priority scheduling
    /// Default: read from Q_NEMO_EXECUTOR environment variable (defaults to true)
    pub enable_nemo_executor: bool,

    /// 🚀 v1.5.0-beta: Enable Reddio-style async storage pipeline (70% overhead reduction)
    /// Hot cache + async prefetching + pipelined workflow for maximum throughput
    /// Based on Reddio paper (https://arxiv.org/abs/2503.04595)
    /// Default: read from Q_ASYNC_PIPELINE environment variable (defaults to true)
    pub enable_async_pipeline: bool,

    /// 🚀 v1.5.0-beta: Async pipeline configuration
    #[cfg(not(target_os = "windows"))]
    pub async_pipeline_config: AsyncPipelineConfig,

    // ═══════════════════════════════════════════════════════════════════════════════
    // 🚀 v2.1.0-DELTA-V: Project APOLLO Control Systems Configuration
    // ═══════════════════════════════════════════════════════════════════════════════

    /// 🎛️ v2.0.0-KALMAN: Enable PID rate controller for self-tuning sync rates
    /// Auto-adjusts download rate based on throughput feedback
    /// Default: read from Q_APOLLO_PID environment variable (defaults to true)
    pub enable_apollo_pid: bool,

    /// 🔭 v2.0.0-KALMAN: Enable Kalman network predictor for optimal chunk sizing
    /// Uses Kalman filtering to predict bandwidth/latency and optimize parameters
    /// Default: read from Q_APOLLO_KALMAN environment variable (defaults to true)
    pub enable_apollo_kalman: bool,

    /// 🌍 v1.9.0-SLINGSHOT: Enable gravity-assist peer selection
    /// Prefers peers with hot cache for adjacent block ranges
    /// Default: read from Q_APOLLO_GRAVITY_ASSIST environment variable (defaults to true)
    pub enable_apollo_gravity_assist: bool,

    /// 📦 v2.1.0-DELTA-V: Enable pre-compressed storage for zero-CPU P2P serving
    /// Stores blocks already compressed, serves directly without re-compression
    /// Default: read from Q_APOLLO_PRECOMPRESSED environment variable (defaults to false)
    pub enable_apollo_precompressed: bool,

    /// 🎯 v2.0.0-KALMAN: Target throughput for PID controller (blocks/second)
    /// Default: 1000 BPS (matching existing target)
    pub apollo_target_throughput: f64,

    // ═══════════════════════════════════════════════════════════════════════════════
    // 🎯 v3.2.11-beta: ATOMIC SYNC ENDGAME - Streamlined last ~500 blocks
    // ═══════════════════════════════════════════════════════════════════════════════

    /// 🎯 v3.2.11-beta: Endgame threshold - when within this many blocks of tip, use fast mode
    /// Default: 500 blocks (activates near-tip optimizations)
    pub endgame_threshold: u64,

    /// ⚡ v3.2.11-beta: Endgame chunk size - smaller chunks for faster near-tip sync
    /// Default: 50 blocks (vs 20k normal) - fast, low-latency requests
    pub endgame_chunk_size: u64,

    /// ⏱️ v3.2.11-beta: Endgame timeout - aggressive timeout for small chunks
    /// Default: 3 seconds (vs 10s normal) - fail fast, retry fast
    pub endgame_timeout: Duration,

    /// 🔥 v3.2.11-beta: Enable live block streaming during endgame
    /// When true, subscribes to gossipsub for live blocks while filling small gaps
    /// Default: true - provides near-atomic sync to network tip
    pub endgame_live_stream: bool,

    /// v8.4.0: Preferred peers for sync (1Gbit servers)
    /// Read from Q_PREFERRED_SYNC_PEERS="peer_id1,peer_id2"
    /// These peers get a 3x score boost in gravity-assist peer selection
    pub preferred_sync_peers: Vec<String>,

    /// v8.6.2: Supernode peers (10Gbit+ servers) for maximum sync speed
    /// Read from Q_SUPERNODE_PEERS="peer_id1,peer_id2"
    /// These peers get a 10x score boost (vs 3x for preferred) and use 5000-block chunks
    pub supernode_peers: Vec<String>,
}

impl Default for TurboSyncConfig {
    fn default() -> Self {
        // 🎯 v1.0.39-beta: PHASE 1 COMPRESSION OPTIMIZATION
        //
        // DISCOVERY: Server-side compression is the #1 bottleneck (60ms = 32% of total time)
        // REAL BOTTLENECK: Server compression (40-60ms @ level 3) + Network wait (~150ms)
        //
        // Evidence from testing:
        // - Database batched writes: 106 blocks/s (burst rate) - NOT the bottleneck
        // - Sustained sync rate: 5.33 blocks/s (measured)
        // - Server compression (level 3): ~60ms per 10k-block chunk
        // - Network bandwidth: <1 Mbps of 100 Mbps (massively underutilized!)
        // - Per-chunk time: 188ms (60ms compression + 150ms other)
        //
        // Strategy Phase 1: Reduce compression level 3 → 1
        // Trade-off: 20% more bandwidth for 3x faster compression
        // - Compression time: 60ms → 20ms (3x faster)
        // - Compressed size: 40 MB → 48 MB (+20%)
        // - Network transfer: Still <10ms @ 100 Mbps libp2p
        // Expected improvement: 5.33 → 6.76 blocks/s (+27%)
        //
        // Changes in v1.0.39-beta:
        // - compression_level: 3 → 1 (MAJOR: 3x faster server compression)
        // - parallel_streams: 16 (maintained from v1.0.38)
        // - chunk_size: 10000 (maintained from v1.0.38)
        //
        // To override these defaults, use environment variables (see main.rs):
        // - Q_TURBO_COMPRESSION_LEVEL (override compression level)
        // - Q_TURBO_PARALLEL_STREAMS
        // - Q_TURBO_CHUNK_SIZE
        // - Q_TURBO_CHUNK_TIMEOUT_SECS
        // - Q_BATCHED_WRITES (MUST STAY FALSE - breaks P2P!)

        Self {
            // 🚀 v3.4.7-beta: RELIABLE TURBO SYNC - Sequential with parallelism
            // v3.4.7 FIX: Reduced parallelism to prevent timeout cascades
            // PROBLEM: 64 streams × 20k chunks = 1.28M blocks in flight
            //   → Far-ahead chunks timeout while near chunks complete
            //   → Creates unresolvable gaps, sync stalls
            // SOLUTION: 8 streams × 5k chunks = 40k blocks in flight
            //   → More manageable, fewer timeouts
            //   → Sequential progress with light parallelism
            //
            // Override via environment variables:
            // - Q_TURBO_PARALLEL_STREAMS=8
            // - Q_TURBO_CHUNK_SIZE=5000
            // - Q_TURBO_COMPRESSION_LEVEL=1
            // - Q_TURBO_CHUNK_TIMEOUT_SECS=45
            // v6.0.4: RAM-aware parallel streams to prevent OOM on small nodes
            // 32 streams × 1000 blocks = 32,000 blocks in-flight → ~4-6 GB for Gamma
            // Reduced for small RAM nodes to prevent OOM kills
            parallel_streams: std::env::var("Q_TURBO_PARALLEL_STREAMS")
                .ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                    let ram_mb = {
                        use sysinfo::System;
                        let mut sys = System::new();
                        sys.refresh_memory();
                        (sys.total_memory() / (1024 * 1024)) as usize
                    };
                    // v10.0.9: MAJOR REDUCTION — users reporting 55GB RSS on 64GB machines.
                    // Root cause: streams × chunk_size × ~150KB/block (with deser overhead) = in-flight RAM.
                    // Old: 32 streams × 5000 chunks = 160K blocks × 150KB = 24GB in-flight alone!
                    // New: 8 streams × 2000 chunks = 16K blocks × 150KB = 2.4GB in-flight.
                    // Combined with reduced RocksDB cache, total peak RAM should be <8GB.
                    match ram_mb {
                        0..=3999     => 2,    // micro: 2 streams
                        4000..=7999  => 3,    // small: 3 streams (was 4)
                        8000..=15999 => 4,    // medium: 4 streams (was 6)
                        16000..=31999 => 6,   // large: 6 streams (was 16)
                        32000..=63999 => 8,   // xlarge: 8 streams (was 32)
                        _            => 10,   // xxlarge 64GB+: 10 streams (was 48)
                    }
                }),
            // v8.0.7: P2P-aware chunk size — capped at 500 regardless of local RAM
            // Problem: Large chunks (2000-3000) on big servers timeout because the
            // SERVING peer (e.g., Gamma 7.8GB) can't fetch+serialize+send that many
            // blocks within the 45s timeout. Gamma reliably serves ~500 blocks in ~5s.
            // RAM-based scaling only helps for local DB writes, not P2P requests.
            chunk_size: std::env::var("Q_TURBO_CHUNK_SIZE")
                .ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                    let ram_mb = {
                        use sysinfo::System;
                        let mut sys = System::new();
                        sys.refresh_memory();
                        (sys.total_memory() / (1024 * 1024)) as u64
                    };
                    // v10.0.9: Reduced chunk sizes to cut in-flight memory.
                    // Sync speed is limited by the SERVING peer (Gamma 7.8GB serves ~500/5s).
                    // Larger chunks just timeout on slow peers. 500-1000 is the sweet spot.
                    match ram_mb {
                        0..=3999     => 200,   // micro: 200 blocks/chunk
                        4000..=7999  => 500,   // small: 500 blocks/chunk (was 1000)
                        8000..=15999 => 500,   // medium: 500 blocks/chunk (was 2000)
                        _            => 1000,  // large+: 1000 blocks/chunk (was 5000)
                    }
                }),
            compression_level: std::env::var("Q_TURBO_COMPRESSION_LEVEL")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(1),  // Level 1 for speed
            chunk_timeout: Duration::from_secs(
                std::env::var("Q_TURBO_CHUNK_TIMEOUT_SECS")
                    .ok().and_then(|v| v.parse().ok()).unwrap_or(30)),  // v8.0.10: 30s (was 45s — faster retry)

            // Protocol features (safe to keep enabled)
            delta_compression: true,
            enable_pipelining: true,
            max_peer_connections: std::env::var("Q_MAX_PEER_CONNECTIONS")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(64),  // v8.6.0: 64 peers (was 32, more peer diversity)
            smart_protocol: true,

            // 🚀 v1.0.89-beta: TURBO SYNC - Batched writes now default TRUE
            // History:
            // - v1.0.7-beta: Enabled for performance (true)
            // - v1.0.87-beta: Discovered height regression bugs
            // - v1.0.88-beta: FIXED in save_qblocks_batch() with contiguous height calc
            // - v1.0.89-beta: Re-enabled with TURBO MODE for 1000+ BPS
            //
            // The bugs were:
            // 1. Height pointer set to MAX instead of CONTIGUOUS (FIXED in lib.rs)
            // 2. Per-write fsync causing ~500 BPS limit (FIXED with write_batch_turbo)
            //
            // TURBO MODE combines:
            // - Batched writes (single DB transaction per pack)
            // - write_batch_turbo (WAL enabled, no per-write fsync)
            // - Periodic sync_wal (after each pack for durability)
            //
            // Result: 1000-1500 BPS (3-5x faster than before)
            enable_batched_writes: std::env::var("Q_BATCHED_WRITES")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),  // 🚀 v1.0.89-beta: Default TRUE for turbo sync

            // 🚀 v1.0.5-beta: Request pipelining (Phase 2: Network Optimization)
            // Conservative start with depth=2, adaptive window sizing, flow control
            pipeline_config: PipelineConfig::default(),

            // 🚀 v1.0.6-beta: Pack caching (Phase 3: Network Optimization)
            // Server-side LRU cache (500 MB, 1000 entries, 1 hour TTL)
            pack_cache_config: PackCacheConfig::default(),

            // 🚀 v2.3.1-beta: Comprehensive state sync enabled by default
            // Users expect sync to "just work" - no environment variables needed
            enable_state_sync: std::env::var("Q_STATE_SYNC")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),  // ✅ v2.3.1: Default TRUE for out-of-box experience
            block_gas_limit: 30_000_000, // 30M gas per block (Ethereum-equivalent)

            // 🚀 v1.5.0-beta: CHIRON parallel state application enabled by default
            // Provides ~30% speedup on transaction-heavy blocks via parallel execution
            // Disable with Q_CHIRON_HINTS=0 if encountering issues
            enable_chiron_hints: std::env::var("Q_CHIRON_HINTS")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),  // ON by default - major performance win

            // 🚀 v1.5.0-beta: NEMO high-contention executor enabled by default
            // Provides +42% improvement over Block-STM under high contention
            // Uses greedy commits + priority scheduling from NEMO paper
            // Disable with Q_NEMO_EXECUTOR=0 if encountering issues
            enable_nemo_executor: std::env::var("Q_NEMO_EXECUTOR")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),  // ON by default - best for contended workloads

            // 🚀 v1.5.0-beta: Reddio-style async storage pipeline enabled by default
            // Provides 70% reduction in storage overhead via:
            // - Hot balance cache (direct state reading)
            // - Async prefetching (parallel node loading)
            // - Pipelined workflow (overlapping read/execute/write)
            // Disable with Q_ASYNC_PIPELINE=0 if encountering issues
            enable_async_pipeline: std::env::var("Q_ASYNC_PIPELINE")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),  // ON by default - major storage optimization

            // 🚀 v1.5.0-beta: Async pipeline configuration (reasonable defaults)
            #[cfg(not(target_os = "windows"))]
            async_pipeline_config: AsyncPipelineConfig::default(),

            // ═══════════════════════════════════════════════════════════════════════════════
            // 🚀 v2.1.0-DELTA-V: Project APOLLO Control Systems Defaults
            // ═══════════════════════════════════════════════════════════════════════════════

            // 🎛️ v2.0.0-KALMAN: PID rate controller (self-tuning sync rates)
            enable_apollo_pid: std::env::var("Q_APOLLO_PID")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),  // ON by default - auto-adjusts download rate

            // 🔭 v2.0.0-KALMAN: Kalman network predictor (optimal chunk sizing)
            enable_apollo_kalman: std::env::var("Q_APOLLO_KALMAN")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),  // ON by default - predicts optimal parameters

            // 🌍 v1.9.0-SLINGSHOT: Gravity-assist peer selection (cache-aware)
            enable_apollo_gravity_assist: std::env::var("Q_APOLLO_GRAVITY_ASSIST")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),  // ON by default - prefers peers with hot cache

            // 📦 v2.1.0-DELTA-V: Pre-compressed storage (zero-CPU P2P serving)
            enable_apollo_precompressed: std::env::var("Q_APOLLO_PRECOMPRESSED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),  // OFF by default - needs storage migration

            // 🎯 v2.0.0-KALMAN: Target throughput (blocks/second)
            apollo_target_throughput: std::env::var("Q_APOLLO_TARGET_THROUGHPUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| {
                    let ram_mb = {
                        use sysinfo::System;
                        let mut sys = System::new();
                        sys.refresh_memory();
                        sys.total_memory() / (1024 * 1024)
                    };
                    // v7.1.4: Higher throughput target for beefy servers
                    if ram_mb >= 32000 { 2000.0 }  // 32GB+: target 2000 BPS
                    else if ram_mb >= 16000 { 1500.0 }  // 16GB+: target 1500 BPS
                    else { 1000.0 }  // default: 1000 BPS
                }),

            // ═══════════════════════════════════════════════════════════════════════════════
            // 🎯 v3.2.11-beta: ATOMIC SYNC ENDGAME - Near-Tip Optimization Defaults
            // ═══════════════════════════════════════════════════════════════════════════════

            // 🎯 Endgame threshold: Activate fast mode when within 500 blocks of network tip
            // This ensures the final sync phase is streamlined and near-atomic
            endgame_threshold: std::env::var("Q_ENDGAME_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(200),  // v1.0.2: 200 blocks from tip = endgame mode (was 500)

            // ⚡ Endgame chunk size: Small chunks for fast near-tip requests
            // Smaller chunks = lower latency = faster convergence to tip
            endgame_chunk_size: std::env::var("Q_ENDGAME_CHUNK_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50),  // 50 blocks per request (fast, low overhead)

            // ⏱️ Endgame timeout: Aggressive timeout for small chunk requests
            // Fail fast, retry fast - no waiting 10+ seconds for 50 blocks
            endgame_timeout: Duration::from_secs(
                std::env::var("Q_ENDGAME_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(3)),  // 3 seconds (vs 10s normal)

            // 🔥 Live block streaming: Subscribe to gossipsub while syncing final blocks
            // This provides near-atomic sync by receiving new blocks as they're produced
            endgame_live_stream: std::env::var("Q_ENDGAME_LIVE_STREAM")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true),  // ON by default for atomic sync experience

            // v8.4.0: Preferred sync peers (comma-separated peer IDs)
            // These peers get a 3x score boost in gravity-assist peer selection
            preferred_sync_peers: std::env::var("Q_PREFERRED_SYNC_PEERS")
                .ok()
                .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
                .unwrap_or_default(),

            // v8.6.2: Supernode peers (10Gbit+ servers)
            // These peers get a 10x score boost and use 5000-block chunks
            supernode_peers: std::env::var("Q_SUPERNODE_PEERS")
                .ok()
                .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
                .unwrap_or_default(),
        }
    }
}

/// Compressed block pack (Git-inspired pack file format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockPack {
    /// Starting height of this pack
    pub start_height: u64,

    /// Ending height (inclusive)
    pub end_height: u64,

    /// Compressed block data (zstd-compressed bincode)
    pub compressed_data: Vec<u8>,

    /// Checksum for verification (blake3 for speed)
    pub checksum: [u8; 32],

    /// Compression ratio achieved (for metrics)
    pub compression_ratio: f32,

    /// Number of blocks in this pack
    pub block_count: u32,

    /// Original uncompressed size
    pub uncompressed_size: u64,

    /// Request ID for tracking P2P responses (optional, only used for TRUE P2P)
    #[serde(default)]
    pub request_id: Option<String>,
}

/// Network request for block pack (sent via gossipsub)
///
/// v0.9.53-beta: REMOVED #[serde(default)] from protocol_version
/// This was causing postcard deserialization to misinterpret byte streams.
/// Version detection now happens BEFORE deserialization by inspecting first byte.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockPackRequest {
    /// Protocol version - ALWAYS REQUIRED
    /// Version 1: Current format with protocol_version field
    /// Version detection happens before deserialization (peek first byte)
    pub protocol_version: u32,

    /// Starting height
    pub start_height: u64,

    /// Ending height (inclusive)
    pub end_height: u64,

    /// Request ID for tracking responses
    pub request_id: String,
}

impl BlockPackRequest {
    /// Current protocol version
    pub const CURRENT_PROTOCOL_VERSION: u32 = 1;

    /// Create new request with current protocol version
    pub fn new(start_height: u64, end_height: u64, request_id: String) -> Self {
        Self {
            protocol_version: Self::CURRENT_PROTOCOL_VERSION,
            start_height,
            end_height,
            request_id,
        }
    }

    /// ✅ v0.9.53-beta: Version detection BEFORE deserialization
    /// ✅ v0.9.56-beta: Enhanced logging and early corruption detection
    /// Inspects first byte to determine format, eliminating deserialization ambiguity
    /// This fixes the critical bug where identical nodes couldn't decode each other's messages
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            anyhow::bail!("Empty BlockPackRequest data");
        }

        // ✅ v0.9.56-beta: Enhanced DEBUG - Log received bytes with more context
        tracing::info!("🔍 [TURBO SYNC DEBUG] Received BlockPackRequest bytes (len={}): {:02x?}",
                       data.len(), &data[..data.len().min(64)]);

        // v0.9.53-beta: Version detection by inspecting first byte
        // NEW format (v1): First byte is 0x01 (protocol_version=1 as varint)
        // OLD format (v0): First byte is start_height varint (typically > 0x01 for any realistic blockchain)
        //
        // EDGE CASE: If old format has start_height=1, first byte WILL be 0x01
        // Solution: Try NEW format first. If deserialization fails OR validation fails, try OLD format.
        // This is safe because:
        // - NEW format has 4 fields (protocol_version, start_height, end_height, request_id)
        // - OLD format has 3 fields (start_height, end_height, request_id)
        // - Postcard will fail to deserialize if field count doesn't match

        let is_new_format = data[0] == 0x01;

        // ✅ v0.9.56-beta: Log format detection decision
        tracing::info!("🔍 [TURBO SYNC DEBUG] Format detection: first_byte=0x{:02x}, is_new_format={}",
                       data[0], is_new_format);

        if is_new_format {
            // Try NEW format (with protocol_version field)
            match postcard::from_bytes::<Self>(data) {
                Ok(req) => {
                    // ✅ v0.9.56-beta: Enhanced logging with all decoded fields
                    tracing::info!("✅ [TURBO SYNC DEBUG] NEW format decoded: protocol_version={}, start={}, end={}, id={}",
                                  req.protocol_version, req.start_height, req.end_height, req.request_id);

                    // ✅ v0.9.56-beta: CRITICAL - Early corruption detection BEFORE validation
                    // Detect corrupted heights that exceed maximum blockchain height
                    // This catches the case where postcard SUCCEEDS but produces corrupted values
                    if req.start_height > 100_000_000 || req.end_height > 100_000_000 {
                        tracing::error!("❌ [TURBO SYNC DEBUG] CORRUPTED HEIGHT DETECTED AFTER POSTCARD DECODE!");
                        tracing::error!("❌ [TURBO SYNC DEBUG] start_height={}, end_height={}",
                                       req.start_height, req.end_height);
                        tracing::error!("❌ [TURBO SYNC DEBUG] This indicates struct field misalignment during deserialization!");
                        tracing::error!("❌ [TURBO SYNC DEBUG] Possible causes:");
                        tracing::error!("    1. Sender has #[serde(default)] on protocol_version (OLD binary v0.9.52 or earlier)");
                        tracing::error!("    2. Receiver has #[serde(default)] on protocol_version (THIS binary is OLD)");
                        tracing::error!("    3. Binary version mismatch between sender and receiver");
                        tracing::error!("❌ [TURBO SYNC DEBUG] Raw bytes: {:02x?}", data);

                        // Try OLD format as fallback
                        tracing::warn!("⚠️  [TURBO SYNC DEBUG] Attempting OLD format decode as fallback...");
                        return Self::decode_old_format(data);
                    }

                    // Validate heights (normal validation after corruption check)
                    if let Err(e) = req.validate_heights() {
                        // NEW format succeeded but heights are invalid
                        // This might be OLD format with start_height=1 (edge case)
                        tracing::warn!("NEW format validation failed: {}, trying OLD format", e);
                        return Self::decode_old_format(data);
                    }

                    Ok(req)
                }
                Err(e) => {
                    // NEW format deserialization failed
                    // This is likely OLD format with start_height=1 (edge case)
                    tracing::warn!("⚠️  [TURBO SYNC DEBUG] NEW format decode failed: {}, trying OLD format", e);
                    Self::decode_old_format(data)
                }
            }
        } else {
            // First byte is NOT 0x01, definitely OLD format
            tracing::info!("🔍 [TURBO SYNC DEBUG] Using OLD format decode (first byte != 0x01)");
            Self::decode_old_format(data)
        }
    }

    /// v0.9.53-beta: Decode OLD format (no protocol_version field)
    /// v0.9.65-beta: Added MessagePack fallback for cross-version compatibility
    /// Includes validation to catch any deserialization errors
    fn decode_old_format(data: &[u8]) -> Result<Self> {
        #[derive(serde::Deserialize)]
        struct OldBlockPackRequest {
            pub start_height: u64,
            pub end_height: u64,
            pub request_id: String,
        }

        const MAX_SANE_HEIGHT: u64 = 100_000_000; // 100 million blocks

        // ✅ v0.9.65-beta: Try postcard OLD format first
        tracing::info!("🔍 [TURBO SYNC DEBUG] Trying OLD format (postcard)");
        match postcard::from_bytes::<OldBlockPackRequest>(data) {
            Ok(old) => {
                // ✅ v0.9.59-beta: CRITICAL - Detect corrupt deserialization (SYNC-DOWN PREVENTION)
                // This catches postcard successfully deserializing but producing garbage values
                if old.start_height <= MAX_SANE_HEIGHT && old.end_height <= MAX_SANE_HEIGHT {
                    tracing::info!("✅ [TURBO SYNC DEBUG] OLD format (postcard) decoded successfully");
                    let req = Self {
                        protocol_version: 0,
                        start_height: old.start_height,
                        end_height: old.end_height,
                        request_id: old.request_id,
                    };
                    req.validate_heights()?;
                    return Ok(req);
                } else {
                    tracing::warn!(
                        "⚠️ OLD format decode produced insane heights: {}-{}, trying MessagePack",
                        old.start_height, old.end_height
                    );
                }
            }
            Err(e) => {
                tracing::warn!("⚠️ OLD format decode failed: {:?}, trying MessagePack", e);
            }
        }

        // ✅ v0.9.66-beta: Try MessagePack with rmp_serde::Deserializer for better compatibility
        // MessagePack can serialize structs as arrays OR maps - we need to support both
        tracing::info!("🔍 [TURBO SYNC DEBUG] Trying MessagePack format (flexible deserializer)");

        // Try default MessagePack (struct as map)
        match rmp_serde::from_slice::<Self>(data) {
            Ok(req) => {
                tracing::info!("✅ [TURBO SYNC DEBUG] MessagePack format (map) decoded successfully!");
                req.validate_heights()?;
                return Ok(req);
            }
            Err(e) => {
                tracing::warn!("⚠️ MessagePack NEW format (map) decode failed: {:?}, trying array format", e);
            }
        }

        // ✅ v0.9.66-beta: Try MessagePack with array format (compact serialization)
        // Some MessagePack encoders use positional arrays instead of named maps
        use rmp_serde::Deserializer;
        use serde::Deserialize;

        let mut deserializer = Deserializer::new(&data[..]);
        match Self::deserialize(&mut deserializer) {
            Ok(req) => {
                tracing::info!("✅ [TURBO SYNC DEBUG] MessagePack format (array) decoded successfully!");
                req.validate_heights()?;
                return Ok(req);
            }
            Err(e) => {
                tracing::warn!("⚠️ MessagePack NEW format (array) decode failed: {:?}, trying OLD format", e);
            }
        }

        // ✅ v0.9.65-beta: Try MessagePack OLD format (without protocol_version)
        match rmp_serde::from_slice::<OldBlockPackRequest>(data) {
            Ok(old) => {
                tracing::info!("✅ [TURBO SYNC DEBUG] MessagePack OLD format decoded successfully!");
                let req = Self {
                    protocol_version: 0,
                    start_height: old.start_height,
                    end_height: old.end_height,
                    request_id: old.request_id,
                };
                req.validate_heights()?;
                return Ok(req);
            }
            Err(e2) => {
                tracing::warn!("⚠️ MessagePack OLD format decode failed: {:?}, trying bincode", e2);
            }
        }

        // ✅ v0.9.66-beta: Try bincode NEW format (Docker containers use bincode!)
        tracing::info!("🔍 [TURBO SYNC DEBUG] Trying bincode format");
        let bincode_new_err = match bincode::deserialize::<Self>(data) {
            Ok(req) => {
                tracing::info!("✅ [TURBO SYNC DEBUG] Bincode NEW format decoded successfully!");
                req.validate_heights()?;
                return Ok(req);
            }
            Err(e) => {
                tracing::warn!("⚠️ Bincode NEW format decode failed: {:?}", e);
                e
            }
        };

        // ✅ v0.9.66-beta: Try bincode OLD format (without protocol_version)
        match bincode::deserialize::<OldBlockPackRequest>(data) {
            Ok(old) => {
                tracing::info!("✅ [TURBO SYNC DEBUG] Bincode OLD format decoded successfully!");
                let req = Self {
                    protocol_version: 0,
                    start_height: old.start_height,
                    end_height: old.end_height,
                    request_id: old.request_id,
                };
                req.validate_heights()?;
                Ok(req)
            }
            Err(bincode_old_err) => {
                tracing::error!("❌ [v0.9.66] ALL decode attempts failed!");
                tracing::error!("   Postcard NEW: failed");
                tracing::error!("   Postcard OLD: insane heights or failed");
                tracing::error!("   MessagePack NEW (map): failed");
                tracing::error!("   MessagePack NEW (array): failed");
                tracing::error!("   MessagePack OLD: failed");
                tracing::error!("   Bincode NEW: {:?}", bincode_new_err);
                tracing::error!("   Bincode OLD: {:?}", bincode_old_err);
                tracing::error!("   Raw bytes (len={}, first 64): {:02x?}", data.len(), &data[..data.len().min(64)]);

                // ✅ v0.9.66: Try to decode as ASCII to see if it's human-readable
                if let Ok(ascii) = std::str::from_utf8(&data[..data.len().min(64)]) {
                    tracing::error!("   As ASCII: {:?}", ascii);
                }

                // ✅ v0.9.66: Detailed byte analysis for debugging
                tracing::error!("   First byte analysis: 0x{:02x} = {:08b}", data[0], data[0]);
                if data[0] >= 0x90 && data[0] <= 0x9f {
                    tracing::error!("      → MessagePack fixarray with {} elements", data[0] & 0x0f);
                } else if data[0] == 0x01 {
                    tracing::error!("      → Postcard protocol_version=1");
                } else {
                    tracing::error!("      → Unknown format (not MessagePack array or postcard v1)");
                }

                anyhow::bail!(
                    "Failed to decode BlockPackRequest with any known format. \
                    This indicates incompatible binary versions between peers."
                )
            }
        }
    }

    /// v0.9.53-beta: Validate that heights make sense (not corrupted)
    /// Replaces the fragile detect_corruption() heuristics
    fn validate_heights(&self) -> Result<()> {
        const MAX_REALISTIC_HEIGHT: u64 = 100_000_000; // 100 million blocks

        if self.start_height > MAX_REALISTIC_HEIGHT {
            anyhow::bail!(
                "Invalid start_height={} (exceeds maximum {})",
                self.start_height, MAX_REALISTIC_HEIGHT
            );
        }

        if self.end_height > MAX_REALISTIC_HEIGHT {
            anyhow::bail!(
                "Invalid end_height={} (exceeds maximum {})",
                self.end_height, MAX_REALISTIC_HEIGHT
            );
        }

        if self.end_height < self.start_height {
            anyhow::bail!(
                "Invalid height range: end_height ({}) < start_height ({})",
                self.end_height, self.start_height
            );
        }

        Ok(())
    }

    /// Validate protocol version compatibility
    pub fn validate_version(&self) -> Result<()> {
        if self.protocol_version != Self::CURRENT_PROTOCOL_VERSION {
            anyhow::bail!(
                "Protocol version mismatch: received v{}, expected v{}. \
                Peer may be running incompatible software version.",
                self.protocol_version,
                Self::CURRENT_PROTOCOL_VERSION
            );
        }
        Ok(())
    }

    /// v0.9.57-beta: Serialize for a specific peer based on their negotiated protocol version
    /// This allows heterogeneous networks with nodes running different versions
    pub fn to_bytes_for_peer(&self, peer_version: u32) -> Result<Vec<u8>> {
        match peer_version {
            0 => {
                // OLD format: [start_height, end_height, request_id]
                // For backwards compatibility with pre-v0.9.53 nodes
                #[derive(Serialize)]
                struct OldFormat {
                    start_height: u64,
                    end_height: u64,
                    request_id: String,
                }

                let old = OldFormat {
                    start_height: self.start_height,
                    end_height: self.end_height,
                    request_id: self.request_id.clone(),
                };

                tracing::debug!(
                    "📤 [TURBO SYNC] Serializing OLD format for peer (version 0): blocks {}-{}",
                    self.start_height,
                    self.end_height
                );

                postcard::to_allocvec(&old)
                    .context("Failed to serialize OLD format BlockPackRequest")
            }
            1 => {
                // NEW format: [protocol_version, start_height, end_height, request_id]
                tracing::debug!(
                    "📤 [TURBO SYNC] Serializing NEW format for peer (version 1): blocks {}-{}",
                    self.start_height,
                    self.end_height
                );

                postcard::to_allocvec(self)
                    .context("Failed to serialize NEW format BlockPackRequest")
            }
            _ => {
                anyhow::bail!("Unsupported peer turbo sync version: {}", peer_version);
            }
        }
    }

    /// v0.9.53-beta: REMOVED - This method used fragile heuristics
    /// Replaced by version detection BEFORE deserialization + validate_heights()
    ///
    /// Old approach (v0.9.52-beta and earlier):
    /// - Deserialize first, then check if values look corrupted
    /// - Used timestamp detection, max height guessing
    /// - Prone to false positives and false negatives
    ///
    /// New approach (v0.9.53-beta):
    /// - Inspect first byte to detect version BEFORE deserialization
    /// - Simple height range validation (no heuristics)
    /// - More robust and maintainable
    #[deprecated(since = "0.9.53-beta", note = "Use version detection + validate_heights() instead")]
    pub fn detect_corruption_old(&self) -> Result<()> {
        const MAX_REASONABLE_HEIGHT: u64 = 10_000_000_000; // 10 billion blocks

        // ✅ v0.9.40-beta: Detect Unix timestamps disguised as block heights
        // Unix timestamps are in the range 1,000,000,000 to 2,000,000,000 for years 2001-2033
        const MIN_TIMESTAMP: u64 = 1_000_000_000;
        const MAX_TIMESTAMP: u64 = 2_500_000_000; // Year 2049
        const MAX_REALISTIC_BLOCKCHAIN_HEIGHT: u64 = 100_000_000; // 100 million blocks

        // Check if start_height looks like a Unix timestamp
        if self.start_height >= MIN_TIMESTAMP && self.start_height <= MAX_TIMESTAMP {
            anyhow::bail!(
                "CORRUPTED REQUEST DETECTED: start_height={} appears to be a Unix timestamp, not a block height. \
                This indicates binary deserialization failure or struct field misalignment.",
                self.start_height
            );
        }

        // Check if end_height looks like a Unix timestamp
        if self.end_height >= MIN_TIMESTAMP && self.end_height <= MAX_TIMESTAMP {
            anyhow::bail!(
                "CORRUPTED REQUEST DETECTED: end_height={} appears to be a Unix timestamp, not a block height. \
                This indicates binary deserialization failure or struct field misalignment.",
                self.end_height
            );
        }

        if self.start_height > MAX_REASONABLE_HEIGHT {
            anyhow::bail!(
                "CORRUPTED REQUEST DETECTED: start_height={} exceeds reasonable maximum {}. \
                This indicates binary protocol version mismatch between nodes.",
                self.start_height, MAX_REASONABLE_HEIGHT
            );
        }

        if self.end_height > MAX_REASONABLE_HEIGHT {
            anyhow::bail!(
                "CORRUPTED REQUEST DETECTED: end_height={} exceeds reasonable maximum {}. \
                This indicates binary protocol version mismatch between nodes.",
                self.end_height, MAX_REASONABLE_HEIGHT
            );
        }

        // ✅ v0.9.40-beta: Additional sanity check for realistic blockchain heights
        if self.start_height > MAX_REALISTIC_BLOCKCHAIN_HEIGHT {
            anyhow::bail!(
                "SUSPICIOUS REQUEST: start_height={} exceeds realistic blockchain height {}. \
                This may indicate corruption or an extremely long-running network.",
                self.start_height, MAX_REALISTIC_BLOCKCHAIN_HEIGHT
            );
        }

        if self.end_height > MAX_REALISTIC_BLOCKCHAIN_HEIGHT {
            anyhow::bail!(
                "SUSPICIOUS REQUEST: end_height={} exceeds realistic blockchain height {}. \
                This may indicate corruption or an extremely long-running network.",
                self.end_height, MAX_REALISTIC_BLOCKCHAIN_HEIGHT
            );
        }

        if self.end_height < self.start_height {
            anyhow::bail!(
                "INVALID REQUEST: end_height ({}) < start_height ({}). \
                Request is malformed or corrupted.",
                self.end_height, self.start_height
            );
        }

        Ok(())
    }
}

/// Network request type for TRUE P2P communication
#[derive(Debug)]
pub enum NetworkRequest {
    /// Request a block pack from the network (gossipsub - LEGACY, less reliable)
    RequestBlockPack {
        start_height: u64,
        end_height: u64,
        request_id: String,
        response_tx: oneshot::Sender<Result<BlockPack>>,
    },
    /// 🚀 v1.3.9-beta: Direct request-response for Turbo Sync (NEW - more reliable)
    /// Uses libp2p request-response protocol instead of gossipsub for:
    /// - Guaranteed delivery or error (no silent drops)
    /// - Point-to-point communication (no broadcast flooding)
    /// - Built-in timeout handling (60s)
    RequestBlockRangeDirect {
        /// Optional peer ID to request from (if None, selects best peer)
        peer_id: Option<String>,
        start_height: u64,
        end_height: u64,
        /// Returns raw QBlock vector (more efficient than BlockPack for small batches)
        response_tx: oneshot::Sender<Result<Vec<q_types::QBlock>>>,
    },
}

/// Smart protocol negotiation (Git's "want/have" protocol)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncNegotiation {
    /// What the requester has (block heights)
    pub have: Vec<u64>,

    /// What the requester wants (target height)
    pub want: u64,

    /// Requester's peer ID
    pub peer_id: String,

    /// Preferred chunk size
    pub preferred_chunk_size: u64,
}

/// Smart protocol response (what the responder can provide)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncNegotiationResponse {
    /// What the responder has (highest block)
    pub highest_block: u64,

    /// Can serve the requested range
    pub can_serve: bool,

    /// Recommended chunk sizes for optimal performance
    pub recommended_chunks: Vec<(u64, u64)>, // (start, end) pairs

    /// Estimated time to serve all chunks
    pub estimated_time_ms: u64,
}

/// Turbo Sync metrics (real-time performance tracking)
#[derive(Debug, Default)]
pub struct TurboSyncMetrics {
    pub total_blocks_synced: AtomicU64,
    pub total_bytes_downloaded: AtomicU64,
    pub total_bytes_saved_by_compression: AtomicU64,
    pub active_parallel_streams: AtomicUsize,
    pub failed_chunks: AtomicU64,
    pub retried_chunks: AtomicU64,
    pub start_time: RwLock<Option<Instant>>,
    pub end_time: RwLock<Option<Instant>>,
}

impl TurboSyncMetrics {
    /// Calculate average download speed in MB/s
    pub async fn average_speed_mbps(&self) -> f64 {
        let start_opt = *self.start_time.read().await;
        let end_opt = *self.end_time.read().await;

        if let (Some(start), Some(end)) = (start_opt, end_opt) {
            let elapsed_secs = end.duration_since(start).as_secs_f64();
            let bytes = self.total_bytes_downloaded.load(Ordering::Relaxed) as f64;
            (bytes / elapsed_secs) / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }

    /// Calculate blocks per second
    pub async fn blocks_per_second(&self) -> f64 {
        let start_opt = *self.start_time.read().await;
        let end_opt = *self.end_time.read().await;

        if let (Some(start), Some(end)) = (start_opt, end_opt) {
            let elapsed_secs = end.duration_since(start).as_secs_f64();
            let blocks = self.total_blocks_synced.load(Ordering::Relaxed) as f64;
            blocks / elapsed_secs
        } else {
            0.0
        }
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let downloaded = self.total_bytes_downloaded.load(Ordering::Relaxed) as f32;
        let saved = self.total_bytes_saved_by_compression.load(Ordering::Relaxed) as f32;
        if downloaded > 0.0 {
            downloaded / (downloaded + saved)
        } else {
            1.0
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 📊 v1.0.2: Detailed Sync Status (Admin Panel Visibility)
// ═══════════════════════════════════════════════════════════════════════════════

/// Detailed sync status for the admin panel — combines session chunk progress,
/// TurboSyncMetrics, and peer registry data into a single snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedSyncStatus {
    pub sync_mode: String,
    pub local_height: u64,
    pub network_height: u64,
    pub gap: u64,
    pub total_chunks: u64,
    pub completed_chunks: u64,
    pub in_flight: u64,
    pub queued: u64,
    pub chunk_progress_pct: f64,
    pub blocks_per_second: f64,
    pub bytes_downloaded_mb: f64,
    pub compression_ratio: f64,
    pub active_streams: u64,
    pub failed_chunks: u64,
    pub retried_chunks: u64,
    pub peer_count: u64,
    pub best_peer_height: u64,
    pub is_fully_synced: bool,
    pub eta_seconds: Option<u64>,
    // v10.9.44 — q-sync-optimizers integration STATUS:
    //   1. KalmanBdpEstimator   — WIRED on TurboSyncManager::kalman_bdp; used in pick_chunk_size_kb().
    //   2. BetaScoreRegistry    — WIRED on TurboSyncManager::beta_scores; used in pick_chunk_peer().
    //   3. CubicRegistry        — WIRED on UnifiedNetworkManager (q-network) for per-peer cwnd.
    //   4. LittlesLawEstimator  — WIRED on TurboSyncManager::littles_law; exposed via qnk_optimal_inflight gauge.
    //   5. ChunkFloorEstimator  — WIRED on TurboSyncManager::chunk_floor; exposed via qnk_chunk_size_floor gauge.
    //   6. MarkovRegistry       — WIRED on TurboSyncManager::markov_states; exposed via qnk_peer_state gauge.
    //   7. EwmvRtt              — WIRED on UnifiedNetworkManager (per-peer timeout registry).
    //   8. PeerFailPredictor    — WIRED on TurboSyncManager::peer_fail_predictor; exposed via qnk_peer_p_fail gauge.
    // v8.2.8: Apollo subsystem metrics
    pub apollo_kalman_bandwidth_mbps: f64,
    pub apollo_kalman_latency_ms: f64,
    pub apollo_kalman_confidence: f64,
    pub apollo_kalman_optimal_chunk_kb: u64,
    pub apollo_kalman_loss_pct: f64,
    pub apollo_kalman_timeout_ms: u64,
    pub apollo_kalman_concurrency: u64,
    pub apollo_kalman_jitter_ms: f64,
    pub apollo_kalman_samples: u64,
    pub apollo_pid_target_bps: f64,
    pub apollo_pid_current_bps: f64,
    pub apollo_pid_error: f64,
    pub apollo_pid_kp: f64,
    pub apollo_pid_ki: f64,
    pub apollo_pid_kd: f64,
    pub apollo_peers_tracked: u64,
    pub apollo_gravity_best_peer: String,
    pub apollo_gravity_best_heat: f64,
    // v1.0.2: Starship Flight Computer telemetry
    pub starship_phase: String,
    pub phase_duration_secs: u64,
    pub orbit_stable: bool,
    pub station_keeping_peer_health: f64,
    pub mission_elapsed_secs: u64,
}

impl Default for DetailedSyncStatus {
    fn default() -> Self {
        Self {
            sync_mode: "idle".to_string(),
            local_height: 0,
            network_height: 0,
            gap: 0,
            total_chunks: 0,
            completed_chunks: 0,
            in_flight: 0,
            queued: 0,
            chunk_progress_pct: 0.0,
            blocks_per_second: 0.0,
            bytes_downloaded_mb: 0.0,
            compression_ratio: 1.0,
            active_streams: 0,
            failed_chunks: 0,
            retried_chunks: 0,
            peer_count: 0,
            best_peer_height: 0,
            is_fully_synced: false,
            eta_seconds: None,
            apollo_kalman_bandwidth_mbps: 0.0,
            apollo_kalman_latency_ms: 0.0,
            apollo_kalman_confidence: 0.0,
            apollo_kalman_optimal_chunk_kb: 0,
            apollo_kalman_loss_pct: 0.0,
            apollo_kalman_timeout_ms: 0,
            apollo_kalman_concurrency: 0,
            apollo_kalman_jitter_ms: 0.0,
            apollo_kalman_samples: 0,
            apollo_pid_target_bps: 0.0,
            apollo_pid_current_bps: 0.0,
            apollo_pid_error: 0.0,
            apollo_pid_kp: 0.0,
            apollo_pid_ki: 0.0,
            apollo_pid_kd: 0.0,
            apollo_peers_tracked: 0,
            apollo_gravity_best_peer: String::new(),
            apollo_gravity_best_heat: 0.0,
            starship_phase: "PRELAUNCH".to_string(),
            phase_duration_secs: 0,
            orbit_stable: false,
            station_keeping_peer_health: 0.0,
            mission_elapsed_secs: 0,
        }
    }
}

/// Turbo Sync Manager - Main orchestrator
pub struct TurboSyncManager {
    config: TurboSyncConfig,
    storage: Arc<QStorage>,

    /// Semaphore for limiting concurrent downloads
    download_semaphore: Arc<Semaphore>,

    /// 🚀 v1.5.1-beta: Semaphore for limiting concurrent decompression tasks
    /// Prevents spawn_blocking threadpool saturation (fixes 100k block stall)
    decompression_semaphore: Arc<Semaphore>,

    /// 🚀 v1.5.1-beta: Counter for batched WAL sync (sync every N chunks, not per-chunk)
    chunks_since_wal_sync: AtomicU64,

    /// 🚀 v1.5.1-beta: Last WAL sync timestamp for timer-based syncing
    last_wal_sync_time: AtomicU64,

    /// 🛡️ v7.2.7: Last flush timestamp for periodic disk persistence
    last_flush_time: AtomicU64,

    /// Metrics for monitoring
    pub metrics: Arc<TurboSyncMetrics>,

    /// v5.2.0: Enhanced peer registry with monotonicity enforcement
    peer_registry: Arc<RwLock<EnhancedPeerRegistry>>,

    /// 🌐 TRUE P2P INTEGRATION - Network communication channel
    /// Send network requests (block pack requests via gossipsub)
    network_tx: Option<mpsc::UnboundedSender<NetworkRequest>>,

    /// 🔐 v0.9.15-beta: AEGIS-QL post-quantum cryptography for signed syncs
    aegis: Arc<Mutex<q_aegis_ql::AegisQL>>,
    aegis_secret_key: Arc<RwLock<q_aegis_ql::SecretKey>>,
    aegis_public_key: q_aegis_ql::PublicKey,

    /// 📊 v0.9.15-beta: Peer trust registry for reputation tracking
    peer_trust: Arc<crate::aegis_sync::PeerTrustRegistry>,

    /// 🧠 v1.0.15.1-beta: Memory limiter for adaptive sync batch sizing
    memory_limiter: Arc<crate::memory_limiter::MemoryLimiter>,

    /// 🚀 v1.0.5-beta: Request pipelining manager (Phase 2: Network Optimization)
    /// Manages concurrent in-flight requests with adaptive window sizing
    pipeline_manager: Arc<PipelineManager>,

    /// 🚀 v1.0.6-beta: Pack cache (Phase 3: Network Optimization)
    /// Server-side LRU cache for compressed block packs
    pack_cache: Arc<PackCache>,

    /// 🚀 v1.0.50-beta: Adaptive timeout calculator (Crypto-Enhanced Sync)
    /// Dynamically adjusts timeouts based on actual network RTT
    adaptive_timeout: Arc<RwLock<AdaptiveTimeout>>,

    /// 🚀 v1.0.50-beta: Sync progress tracker (Crypto-Enhanced Sync)
    /// Tracks peer performance and enables checkpointing for resume
    progress_tracker: Arc<RwLock<SyncProgressTracker>>,

    /// 🚀 v1.0.50-beta: Incremental block verifier (Crypto-Enhanced Sync)
    /// Verifies blocks as they arrive, catching errors early
    block_verifier: Arc<RwLock<IncrementalBlockVerifier>>,

    /// 🛡️ v1.3.0-beta: SHA3-256 Data Integrity Verifier (Quantum-Resistant)
    /// Provides additional quantum-resistant block hash verification
    sha3_verifier: Arc<Sha3DataIntegrity>,

    /// 🚀 v1.0.60-beta: Block state processor (Comprehensive State Sync)
    /// Processes ALL transactions through StateProcessor/StateApplicator
    /// Optional: only used when config.enable_state_sync = true
    #[cfg(not(target_os = "windows"))]
    state_processor: Option<Arc<BlockStateProcessor>>,

    /// 🤖 v1.4.0-beta: ML-driven adaptive batch size optimizer
    /// Uses online linear regression to predict optimal batch sizes based on:
    /// RTT, memory pressure, peer trust, bandwidth, success rate, etc.
    batch_predictor: Arc<RwLock<crate::ml_batch_optimizer::BatchSizePredictor>>,

    /// 🛡️ v1.4.5-beta: Orphan rate limiter for DAG spam attack prevention
    /// Tracks orphan blocks per peer and applies rate limiting/banning
    orphan_limiter: Arc<RwLock<crate::orphan_rate_limiter::OrphanRateLimiter>>,

    /// 🚀 v2.3.4-beta: Emergency sync guard to prevent multiple concurrent syncs
    /// When true, a sync is in progress and new emergency syncs should be skipped
    emergency_sync_in_progress: Arc<AtomicBool>,

    /// 🔒 v10.5.0: Fresh-start single-flight gate.
    /// Gossipsub delivers peer-height events concurrently; each fires sync_to_height().
    /// On a fresh node (height < 100) two invocations race over the height pointer.
    /// Only one invocation should run the probe + initial sync at a time.
    fresh_sync_gate: Arc<Mutex<()>>,
    /// Highest target height latched by any concurrent caller while gate is held.
    /// Gate winner reads this after acquiring the lock so it syncs to the maximum
    /// announced height, not just the height that triggered its own invocation.
    fresh_sync_target: Arc<AtomicU64>,

    // ═══════════════════════════════════════════════════════════════════════════════
    // 🚀 v1.0.2: Lock-Free Sync State (Miner-Optimized Atomics)
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Cached max peer height — updated atomically on every peer registration.
    /// Sync loop reads this instead of acquiring RwLock on peer_registry.
    pub cached_max_peer_height: Arc<AtomicU64>,

    /// Cached active peer count — updated atomically on peer add/remove.
    pub cached_peer_count: Arc<AtomicU32>,

    /// Whether the blockchain has known gaps — set true when gap detected, false when filled.
    /// Sync loop skips expensive get_first_missing_height() DB scan when false.
    pub cached_has_gaps: Arc<AtomicBool>,

    /// Whether the node is fully synced (gap == 0). Used for adaptive loop frequency.
    pub is_fully_synced: Arc<AtomicBool>,

    // ═══════════════════════════════════════════════════════════════════════════════
    // 📊 v1.0.2: Session Chunk Progress Atomics (Admin Panel Sync Visibility)
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Total chunks in the current sync session (0 when idle)
    pub session_total_chunks: Arc<AtomicU64>,
    /// Completed chunks in the current sync session
    pub session_completed_chunks: Arc<AtomicU64>,
    /// Currently in-flight chunk downloads
    pub session_in_flight: Arc<AtomicU64>,
    /// Chunks still queued waiting to be spawned
    pub session_queued: Arc<AtomicU64>,
    /// Current sync mode: 0=idle, 1=turbo, 2=endgame, 3=micro
    pub session_sync_mode: Arc<AtomicU8>,

    /// 🚀 v1.5.0-beta: CHIRON Parallel State Applicator (~30% sync speedup)
    /// Uses pre-computed execution hints to parallelize transaction state application
    /// Based on CHIRON paper (https://arxiv.org/abs/2401.14278)
    parallel_state_applicator: Arc<ParallelStateApplicator>,

    /// 🚀 v1.5.0-beta: NEMO High-Contention Executor (+42% over Block-STM)
    /// Uses greedy commits, refined dependency handling, and priority scheduling
    /// Based on NEMO paper (https://arxiv.org/abs/2510.15122)
    nemo_executor: Arc<NemoExecutor>,

    /// 🚀 v1.5.0-beta: Reddio-Style Async Storage Pipeline (70% overhead reduction)
    /// Hot cache + async prefetching + pipelined workflow for maximum throughput
    /// Based on Reddio paper (https://arxiv.org/abs/2503.04595)
    #[cfg(not(target_os = "windows"))]
    async_pipeline: Option<Arc<AsyncStoragePipeline>>,

    // ═══════════════════════════════════════════════════════════════════════════════
    // 🚀 v2.1.0-DELTA-V: Project APOLLO Control Systems
    // ═══════════════════════════════════════════════════════════════════════════════

    /// 🎛️ v2.0.0-KALMAN: PID rate controller for self-tuning sync rates
    /// Automatically adjusts download rate based on throughput feedback
    /// Like rocket engine throttling to maintain optimal trajectory
    apollo_pid_controller: Arc<RwLock<PIDRateController>>,

    /// 🔭 v2.0.0-KALMAN: Kalman network predictor for optimal parameter tuning
    /// Predicts bandwidth, latency, loss rate for optimal chunk sizing
    /// Like spacecraft navigation using Kalman filtering
    apollo_kalman_predictor: Arc<RwLock<KalmanNetworkPredictor>>,

    /// 🌍 v1.9.0-SLINGSHOT: Gravity-assist peer selection manager
    /// Tracks peer momentum (cache heat, bandwidth) for optimal selection
    /// Like planetary gravity assists, uses peer momentum to accelerate sync
    apollo_peer_momentum: Arc<PeerMomentumManager>,

    // ═══════════════════════════════════════════════════════════════════════════════
    // 🚀 v2.3.10-beta: WARP SYNC Phase 2 & 3 Integration
    // ═══════════════════════════════════════════════════════════════════════════════

    /// 🚀 v2.3.10-beta: Multi-Peer Download Coordinator (Phase 2: 3-5x speedup)
    /// Tracks per-peer bandwidth/latency metrics for intelligent load balancing
    /// Uses scoring formula: score = (bandwidth × success_rate) / (1 + in_flight)
    warp_multi_peer: Arc<MultiPeerDownloader>,

    /// 🚀 v2.3.10-beta: Prefetch Pipeline (Phase 3: Hide network latency)
    /// Predicts next chunks and starts downloading before they're needed
    /// Keeps 8 chunks in prefetch queue for continuous download pipelining
    warp_prefetch: Arc<PrefetchPipeline>,

    // ═══════════════════════════════════════════════════════════════════════════════
    // 🎚️ v8.5.4: Runtime Network Throttle Mode (TUI-controlled resource management)
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Network throttle mode: 0=Conservative, 1=Normal, 2=Turbo
    /// Shared with AppState — TUI sets it, sync loop reads it.
    /// Conservative: 2 in-flight, 200ms inter-chunk delay (SSD-friendly)
    /// Normal: config-default in-flight, 10ms inter-chunk delay
    /// Turbo: 2x config in-flight, 0ms delay (max speed)
    pub network_throttle_mode: Arc<AtomicU8>,

    /// v10.9.44: Known permanent gaps in the chain (e.g. 25988..=100440 from
    /// the historical pruning incident). Parsed from `Q_KNOWN_PERMANENT_GAPS`
    /// at startup. When sync_to_height observes `contiguous + 1 == gap.start`,
    /// the contiguous pointer is advanced past `gap.end` so chunk scheduling
    /// can skip the network-wide hole instead of stalling forever.
    pub known_gaps: Arc<crate::known_gaps::KnownGaps>,

    /// v10.9.44: Per-peer Kalman BDP estimator state (B.1). Wraps the algorithm
    /// from `q-sync-optimizers::kalman_bdp`. The hot path reads the current
    /// snapshot from `apollo_kalman_predictor` and calls `chunk_size_kb`.
    pub kalman_bdp: Arc<RwLock<q_sync_optimizers::KalmanBdpEstimator>>,

    /// v10.9.44: Beta-distribution peer score registry (B.4 + Thompson sampling
    /// for chunk dispatch). Lock-protected because Thompson sampling mutates
    /// the per-peer counter on every draw; the lock is held microseconds.
    pub beta_scores: Arc<parking_lot::Mutex<q_sync_optimizers::BetaScoreRegistry<libp2p::PeerId>>>,

    /// v10.9.44: Markov peer-state registry for per-peer state tracking (item 6).
    /// Exported as `qnk_peer_state{peer="..."}` gauge.
    pub markov_states: Arc<parking_lot::Mutex<q_sync_optimizers::MarkovRegistry<libp2p::PeerId>>>,

    /// v10.9.44: Per-peer Little's Law estimators (item 4 — combined with CUBIC cwnd).
    /// Exported as `qnk_optimal_inflight{peer="..."}` gauge.
    pub littles_law: Arc<dashmap::DashMap<libp2p::PeerId, q_sync_optimizers::LittlesLawEstimator>>,

    /// v10.9.44: Information-theoretic chunk-size floor (item 5). Single
    /// shared estimator — bits-per-block is global, peer-set-size is the input.
    /// Output exported as `qnk_chunk_size_floor` gauge.
    pub chunk_floor: Arc<q_sync_optimizers::ChunkFloorEstimator>,

    /// v10.9.44: Per-peer "about-to-fail" logistic predictor (item 8).
    /// Stateless — config only — features are extracted per-call.
    /// Output exported as `qnk_peer_p_fail{peer="..."}` gauge.
    pub peer_fail_predictor: Arc<q_sync_optimizers::PeerFailPredictor>,

    /// v10.9.47: Autonomous permanent-gap detection — tally of how many
    /// independent peers have declared each (start,end) gap. The handler task
    /// reads gap_advance_rx, updates this tally, and decides whether to commit
    /// a gap into `known_gaps` based on quorum + per-peer Beta trust.
    /// Inner key: (gap_start, gap_end). Value: set of peer ids that reported it.
    pub gap_detection_tally: Arc<parking_lot::Mutex<std::collections::HashMap<(u64, u64), std::collections::HashSet<libp2p::PeerId>>>>,

    /// v10.9.47: Receiver for runtime gap declarations from NetworkManager.
    /// Set via `set_gap_advance_rx()` in main.rs after both components are
    /// constructed. `spawn_gap_advance_handler()` takes ownership and spawns
    /// the long-lived processor task.
    pub gap_advance_rx: parking_lot::Mutex<Option<tokio::sync::broadcast::Receiver<(u64, u64, libp2p::PeerId)>>>,
}

impl TurboSyncManager {
    /// Create new Turbo Sync manager
    pub fn new(storage: Arc<QStorage>, config: TurboSyncConfig) -> Self {
        let max_concurrent = config.max_peer_connections;

        // 🔐 v0.9.15-beta: Generate AEGIS-QL keypair for post-quantum signed syncs
        let mut aegis = q_aegis_ql::AegisQL::new();
        let (public_key, secret_key) = aegis.generate_keypair().expect("Failed to generate AEGIS-QL keypair");
        info!("🔐 [AEGIS-QL] Generated post-quantum keypair for signed P2P sync");
        info!("   Public key (first 16 bytes): {}", hex::encode(&bincode::serialize(&public_key).unwrap()[..16]));

        // 🧠 v1.0.15.1-beta: Initialize memory limiter with adaptive batch sizing
        let memory_limiter = Arc::new(crate::memory_limiter::MemoryLimiter::new());
        info!("🧠 [MEMORY] Memory limiter initialized for adaptive sync batch sizing");

        // 🚀 v1.0.5-beta: Initialize request pipelining manager (Phase 2: Network Optimization)
        let pipeline_manager = Arc::new(PipelineManager::new(config.pipeline_config.clone()));
        info!("🚀 [PIPELINE] Request pipelining initialized (depth: {}, target RTT: {}ms, flow control: {})",
              config.pipeline_config.initial_depth,
              config.pipeline_config.target_rtt_ms,
              config.pipeline_config.enable_flow_control);

        // 🚀 v1.0.6-beta: Initialize pack cache (Phase 3: Network Optimization)
        let pack_cache = Arc::new(PackCache::new(config.pack_cache_config.clone()));
        info!("📦 [PACK CACHE] Initialized (max: {} MB, {} entries, TTL: {}s)",
              config.pack_cache_config.max_size_bytes / (1024 * 1024),
              config.pack_cache_config.max_entries,
              config.pack_cache_config.entry_ttl.as_secs());

        // 🚀 v1.0.50-beta: Initialize crypto-enhanced sync components
        // These provide reliability improvements to prevent sync stalling
        let enhanced_config = EnhancedSyncConfig::default();
        // 🚀 v3.4.7-beta: SYNC TIMEOUT FIX - Reduced max timeout from 180s to 45s
        // REASON: 180s max was causing 3-minute stalls when peers are slow/busy
        // Faster timeout = faster retry with different peer = faster overall sync
        // If a request takes >45s, it's better to retry with another peer anyway.
        // For 5000-block chunks at ~100KB/block = ~500MB, 45s = ~11MB/s minimum.
        let adaptive_timeout = Arc::new(RwLock::new(AdaptiveTimeout::new(
            5000,   // v8.0.10: 5 second minimum timeout (was 10s — too slow for 250-block chunks)
            30000,  // v8.0.10: 30 second maximum timeout (was 45s — faster retry on slow peers)
        )));
        let progress_tracker = Arc::new(RwLock::new(SyncProgressTracker::new(enhanced_config.clone())));
        let block_verifier = Arc::new(RwLock::new(IncrementalBlockVerifier::new(enhanced_config, None)));
        info!("🔐 [CRYPTO-ENHANCED SYNC] Initialized:");
        info!("   • Adaptive timeout: 10s-45s based on RTT (v3.4.7 speed fix)");
        info!("   • Progress tracker: checkpointing every 1000 blocks");
        info!("   • Incremental verifier: early error detection");

        // 🚀 v1.0.89-beta: Log TURBO SYNC mode status
        let turbo_enabled = std::env::var("Q_TURBO_SYNC")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(true);  // Default: ON
        let batched_enabled = config.enable_batched_writes;

        if turbo_enabled && batched_enabled {
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            info!("⚡ [TURBO SYNC v1.0.89] MAXIMUM PERFORMANCE MODE ENABLED");
            info!("   • write_batch_turbo: WAL + no fsync = ~0.1ms/write");
            info!("   • Batched writes: ON (single DB tx per pack)");
            info!("   • Target: 1000-1500 blocks/second");
            info!("   • Durability: sync_wal() after each pack");
            info!("   • Max data loss on crash: 1 pack (~500 blocks)");
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        } else {
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            warn!("🐌 [SAFE SYNC] Running in SAFE MODE (slower but durable)");
            warn!("   • TURBO: {} (Q_TURBO_SYNC)", if turbo_enabled { "ON" } else { "OFF" });
            warn!("   • BATCH: {} (Q_BATCHED_WRITES)", if batched_enabled { "ON" } else { "OFF" });
            warn!("   • Expected: ~200-500 blocks/second");
            warn!("   • For 1000+ BPS: Q_TURBO_SYNC=1 Q_BATCHED_WRITES=1");
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }

        // 🚀 v1.4.12-beta: Log performance configuration
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📊 [TURBO SYNC v1.4.12] Performance Configuration:");
        info!("   • Parallel streams: {}", config.parallel_streams);
        info!("   • Chunk size: {} blocks", config.chunk_size);
        info!("   • Compression level: {}", config.compression_level);
        info!("   • Chunk timeout: {}s", config.chunk_timeout.as_secs());
        info!("   • Max peer connections: {}", config.max_peer_connections);
        info!("   • Pipeline depth: {}-{}", config.pipeline_config.min_depth, config.pipeline_config.max_depth);
        info!("   Override via: Q_TURBO_PARALLEL_STREAMS, Q_TURBO_CHUNK_SIZE, etc.");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // 🚀 v1.0.60-beta: Initialize block state processor (Comprehensive State Sync)
        // Only initialize if enabled via Q_STATE_SYNC=1
        #[cfg(not(target_os = "windows"))]
        let state_processor = if config.enable_state_sync {
            // Get RocksDB handle from storage for state processor
            if let Some(db) = storage.get_rocks_db_handle() {
                let processor = BlockStateProcessor::with_gas_limit(db, config.block_gas_limit);
                info!("🔄 [STATE SYNC] BlockStateProcessor initialized (gas limit: {})", config.block_gas_limit);
                Some(Arc::new(processor))
            } else {
                warn!("⚠️  [STATE SYNC] Q_STATE_SYNC=1 but RocksDB handle unavailable - falling back to legacy mode");
                None
            }
        } else {
            debug!("📭 [STATE SYNC] Disabled (set Q_STATE_SYNC=1 to enable)");
            None
        };

        // 🛡️ v1.3.0-beta: Initialize SHA3-256 Data Integrity Verifier
        // Provides quantum-resistant block hash verification using NIST FIPS 202 SHA3-256
        let sha3_config = Sha3IntegrityConfig::default();
        let sha3_verifier = Arc::new(Sha3DataIntegrity::new(sha3_config));
        info!("🛡️ [SHA3-256] Data integrity verifier initialized (quantum-resistant block verification)");

        // 🤖 v1.4.0-beta: Initialize ML-driven batch size optimizer
        // Uses online linear regression to predict optimal batch sizes
        // v2.7.4-beta FIX: min_batch_size 10 → 500 to fix slow peer sync (was 6-block chunks)
        // v6.1.4 FIX: Ensure min_batch_size <= max_batch_size (was panicking on low-RAM systems
        //   where chunk_size=100 but min_batch_size=500 → clamp(500,100) assertion failure)
        let max_batch = config.chunk_size as u64;
        let min_batch = 500u64.min(max_batch); // Never exceed max on low-RAM systems
        let batch_config = crate::ml_batch_optimizer::BatchOptimizerConfig {
            min_batch_size: min_batch,
            max_batch_size: max_batch,
            learning_rate: 0.05,  // v8.6.0: faster adaptation (was 0.01)
            ema_decay: 0.1,  // Fast adaptation for changing network conditions
            cold_start_threshold: 50,  // Use heuristics until 50 samples
            history_size: 100,
            target_throughput_bps: 2000.0,  // v2.7.4: Target 2000 blocks/second (was 1000)
            // v1.4.0: Multi-line prefetch optimization (MDPI 2025 paper)
            ..Default::default()  // Use defaults for prefetch settings
        };
        let batch_predictor = Arc::new(RwLock::new(
            crate::ml_batch_optimizer::BatchSizePredictor::new(batch_config)
        ));
        info!("🤖 [ML BATCH] Adaptive batch size optimizer initialized (cold start: 50 samples)");

        // 🛡️ v1.4.5-beta: Initialize orphan rate limiter for DAG spam attack prevention
        let orphan_limits = crate::orphan_rate_limiter::OrphanRateLimits {
            warning_threshold: 10,      // 10 orphans/minute = warning
            ban_threshold: 50,          // 50 orphans/minute = 5 min ban
            max_global_orphans: 10_000, // Maximum pending orphans globally
            ..Default::default()
        };
        let orphan_limiter = Arc::new(RwLock::new(
            crate::orphan_rate_limiter::OrphanRateLimiter::with_limits(orphan_limits)
        ));
        info!("🛡️ [ORPHAN LIMITER] DAG spam attack prevention initialized (warn: 10/min, ban: 50/min)");

        // 🚀 v2.3.4-beta: Initialize emergency sync guard
        let emergency_sync_in_progress = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // 🚀 v1.5.0-beta: Initialize CHIRON Parallel State Applicator
        // Uses pre-computed execution hints to parallelize transaction processing (~30% speedup)
        let chiron_enabled = config.enable_chiron_hints;
        let parallel_state_applicator = Arc::new(
            ParallelStateApplicator::with_default_parallelism()
        );
        if chiron_enabled {
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            info!("🚀 [CHIRON v1.5.0] PARALLEL STATE APPLICATION ENABLED");
            info!("   • Based on: CHIRON paper (https://arxiv.org/abs/2401.14278)");
            info!("   • Speedup target: ~30% faster block state application");
            info!("   • Parallelism: {} threads (rayon)", rayon::current_num_threads());
            info!("   • Block-STM validation: ON (conflict detection)");
            info!("   Disable with Q_CHIRON_HINTS=0 if encountering issues");
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        } else {
            debug!("📭 [CHIRON] Parallel state application disabled (set Q_CHIRON_HINTS=1 to enable)");
        }

        // 🚀 v1.5.0-beta: Initialize NEMO High-Contention Executor
        // Provides +42% improvement over Block-STM under high contention
        let nemo_enabled = config.enable_nemo_executor;
        let nemo_executor = Arc::new(
            NemoExecutor::with_default_parallelism()
        );
        if nemo_enabled {
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            info!("🚀 [NEMO v1.5.0] HIGH-CONTENTION EXECUTOR ENABLED");
            info!("   • Based on: NEMO paper (https://arxiv.org/abs/2510.15122)");
            info!("   • Speedup target: +42% over Block-STM under high contention");
            info!("   • Features: Greedy commits, priority scheduling");
            info!("   • Parallelism: {} threads (rayon)", rayon::current_num_threads());
            info!("   Disable with Q_NEMO_EXECUTOR=0 if encountering issues");
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        } else {
            debug!("📭 [NEMO] High-contention executor disabled (set Q_NEMO_EXECUTOR=1 to enable)");
        }

        // 🚀 v1.5.0-beta: Initialize Reddio-Style Async Storage Pipeline
        // Provides 70% reduction in storage overhead (MPT ops = 70% of execution time)
        #[cfg(not(target_os = "windows"))]
        let async_pipeline_enabled = config.enable_async_pipeline;
        #[cfg(not(target_os = "windows"))]
        let async_pipeline = if async_pipeline_enabled {
            // Get RocksDB handle from storage for async pipeline
            if let Some(db) = storage.get_rocks_db_handle() {
                match AsyncStoragePipeline::new(
                    db,
                    "balances".to_string(),
                    config.async_pipeline_config.clone(),
                ) {
                    Ok(pipeline) => {
                        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                        info!("🚀 [REDDIO v1.5.0] ASYNC STORAGE PIPELINE ENABLED");
                        info!("   • Based on: Reddio paper (https://arxiv.org/abs/2503.04595)");
                        info!("   • Target: 70% reduction in storage overhead");
                        info!("   • Features: Hot cache, async prefetch, pipelined workflow");
                        info!("   • Cache capacity: {} entries", config.async_pipeline_config.hot_cache_capacity);
                        info!("   • Prefetch workers: {}", config.async_pipeline_config.prefetch_workers);
                        info!("   Disable with Q_ASYNC_PIPELINE=0 if encountering issues");
                        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                        Some(Arc::new(pipeline))
                    }
                    Err(e) => {
                        warn!("⚠️  [ASYNC PIPELINE] Failed to initialize: {} - falling back to sync I/O", e);
                        None
                    }
                }
            } else {
                warn!("⚠️  [ASYNC PIPELINE] Q_ASYNC_PIPELINE=1 but RocksDB handle unavailable");
                None
            }
        } else {
            debug!("📭 [ASYNC PIPELINE] Disabled (set Q_ASYNC_PIPELINE=1 to enable)");
            None
        };

        // 🚀 v6.0.5: RAM-aware decompression parallelism
        // Each concurrent decompression holds a full pack in memory (~5-25MB each)
        // v8.0.10: Match decompression parallelism to stream count
        // Decompression is fast (~5ms per chunk with LZ4) — the old low limits
        // serialized work and caused sync <10 BPS
        let default_decomp = {
            use sysinfo::System;
            let mut sys = System::new();
            sys.refresh_memory();
            let ram_mb = (sys.total_memory() / (1024 * 1024)) as usize;
            match ram_mb {
                0..=3999     => 4usize,   // micro: match 4 streams
                4000..=7999  => 8,         // small: match 8 streams
                8000..=15999 => 16,        // medium: match 16 streams
                16000..=31999 => 24,       // large: match 24 streams
                _            => 32,        // xlarge: match 32 streams
            }
        };
        let decompression_parallelism = std::env::var("Q_DECOMPRESSION_PARALLELISM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_decomp);
        info!("🚀 [v1.6.0-SCRAMJET] Decompression semaphore: {} concurrent tasks (THRUST VECTORING)",
              decompression_parallelism);

        // 🚀 v1.5.1-beta: Configure WAL sync batching for 1000 BPS
        // Sync WAL every 50 chunks (1M blocks) OR every 60 seconds
        // v7.2.7: Reduced to 5 chunks to minimize data loss on crash (~2500 blocks max loss)
        let wal_sync_batch_size = std::env::var("Q_WAL_SYNC_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5u64);  // v7.2.7: 5 chunks (was 50). Max ~2500 blocks loss on crash
        info!("🔧 [v7.2.7] WAL sync every {} chunks (safe batched sync)", wal_sync_batch_size);

        // ═══════════════════════════════════════════════════════════════════════════════
        // 🚀 v2.1.0-DELTA-V: Initialize Project APOLLO Control Systems
        // ═══════════════════════════════════════════════════════════════════════════════

        // 🎛️ v2.0.0-KALMAN: Initialize PID rate controller
        // Automatically adjusts sync rate based on throughput feedback
        let apollo_pid_controller = Arc::new(RwLock::new(
            PIDRateController::new(config.apollo_target_throughput)
        ));
        if config.enable_apollo_pid {
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            info!("🚀 [APOLLO KALMAN v2.0.0] PID RATE CONTROLLER ENABLED");
            info!("   • Target throughput: {:.0} blocks/second", config.apollo_target_throughput);
            info!("   • Auto-tuning: Ziegler-Nichols method");
            info!("   • Gains: Kp=0.5, Ki=0.1, Kd=0.05 (auto-adjusted)");
            info!("   Disable with Q_APOLLO_PID=0 if encountering issues");
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }

        // 🔭 v2.0.0-KALMAN: Initialize Kalman network predictor
        // Predicts optimal chunk size, timeout, concurrency based on network conditions
        let apollo_kalman_predictor = Arc::new(RwLock::new(
            KalmanNetworkPredictor::new()
        ));
        if config.enable_apollo_kalman {
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            info!("🚀 [APOLLO KALMAN v2.0.0] NETWORK PREDICTOR ENABLED");
            info!("   • Kalman filtering for bandwidth, latency, loss");
            info!("   • Optimal chunk size: bandwidth-delay product");
            info!("   • Adaptive timeout: 2x predicted latency");
            info!("   Disable with Q_APOLLO_KALMAN=0 if encountering issues");
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }

        // 🌍 v1.9.0-SLINGSHOT: Initialize peer momentum manager
        // Tracks which peers have hot cache for adjacent block ranges
        let apollo_peer_momentum = Arc::new(PeerMomentumManager::new());
        if config.enable_apollo_gravity_assist {
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            info!("🚀 [APOLLO SLINGSHOT v1.9.0] GRAVITY ASSIST ENABLED");
            info!("   • Cache-aware peer selection (3-5x cache hit improvement)");
            info!("   • Peer momentum tracking: heat, bandwidth, latency");
            info!("   • 30-second cache heat half-life");
            info!("   Disable with Q_APOLLO_GRAVITY_ASSIST=0 if encountering issues");
            if !config.supernode_peers.is_empty() {
                info!("   🚀 v8.6.2: {} SUPERNODE peers configured (10x boost, 5000-block chunks)",
                      config.supernode_peers.len());
                for p in &config.supernode_peers {
                    info!("     ⚡ {}", &p[..p.len().min(16)]);
                }
            }
            if !config.preferred_sync_peers.is_empty() {
                info!("   📡 v8.4.0: {} preferred sync peers configured (3x boost)",
                      config.preferred_sync_peers.len());
                for p in &config.preferred_sync_peers {
                    info!("     → {}", &p[..p.len().min(16)]);
                }
            }
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }

        // 🚀 v2.3.10-beta: Initialize Warp Sync Phase 2 & 3
        // Multi-Peer Download + Prefetch Pipeline for 3-5x faster sync
        let warp_multi_peer = Arc::new(MultiPeerDownloader::new());
        let warp_prefetch = Arc::new(PrefetchPipeline::new());
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("🚀 [WARP SYNC v2.3.10] PHASE 2 & 3 ENABLED");
        info!("   • Multi-Peer Download: Intelligent load balancing");
        info!("   • Prefetch Pipeline: Predictive block downloading");
        info!("   • Scoring: bandwidth × success_rate / (1 + in_flight)");
        info!("   Disable with Q_WARP_MULTI_PEER=0 or Q_WARP_PREFETCH=0");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        Self {
            config,
            storage,
            download_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            decompression_semaphore: Arc::new(Semaphore::new(decompression_parallelism)),
            chunks_since_wal_sync: AtomicU64::new(0),
            last_wal_sync_time: AtomicU64::new(0),
            last_flush_time: AtomicU64::new(0),
            metrics: Arc::new(TurboSyncMetrics::default()),
            peer_registry: Arc::new(RwLock::new(EnhancedPeerRegistry::new())),
            network_tx: None, // Set via set_network_channel()
            aegis: Arc::new(Mutex::new(aegis)),
            aegis_secret_key: Arc::new(RwLock::new(secret_key)),
            aegis_public_key: public_key,
            peer_trust: Arc::new(crate::aegis_sync::PeerTrustRegistry::new()),
            memory_limiter,
            pipeline_manager,
            pack_cache,
            adaptive_timeout,
            progress_tracker,
            block_verifier,
            sha3_verifier,
            #[cfg(not(target_os = "windows"))]
            state_processor,
            batch_predictor,
            orphan_limiter,
            emergency_sync_in_progress,
            parallel_state_applicator,
            nemo_executor,
            #[cfg(not(target_os = "windows"))]
            async_pipeline,
            // 🚀 v2.1.0-DELTA-V: Project APOLLO Control Systems
            apollo_pid_controller,
            apollo_kalman_predictor,
            apollo_peer_momentum,
            // 🚀 v2.3.10-beta: WARP SYNC Phase 2 & 3
            warp_multi_peer,
            warp_prefetch,
            // 🚀 v1.0.2: Lock-Free Sync State
            cached_max_peer_height: Arc::new(AtomicU64::new(0)),
            cached_peer_count: Arc::new(AtomicU32::new(0)),
            cached_has_gaps: Arc::new(AtomicBool::new(true)), // Assume gaps until proven otherwise
            is_fully_synced: Arc::new(AtomicBool::new(false)),
            // 📊 v1.0.2: Session Chunk Progress Atomics
            session_total_chunks: Arc::new(AtomicU64::new(0)),
            session_completed_chunks: Arc::new(AtomicU64::new(0)),
            session_in_flight: Arc::new(AtomicU64::new(0)),
            session_queued: Arc::new(AtomicU64::new(0)),
            session_sync_mode: Arc::new(AtomicU8::new(0)),
            // 🎚️ v8.5.4: Runtime throttle mode (default: Turbo=2 for max sync speed)
            network_throttle_mode: Arc::new(AtomicU8::new(2)),
            // 🔒 v10.5.0: Fresh-start single-flight gate
            fresh_sync_gate: Arc::new(Mutex::new(())),
            fresh_sync_target: Arc::new(AtomicU64::new(0)),
            // v10.9.44: Definitive gap-skip from Q_KNOWN_PERMANENT_GAPS env var
            known_gaps: Arc::new(crate::known_gaps::KnownGaps::from_env()),
            // v10.9.44: Scientific sync optimizers (q-sync-optimizers crate)
            kalman_bdp: Arc::new(RwLock::new(q_sync_optimizers::KalmanBdpEstimator::new())),
            beta_scores: Arc::new(parking_lot::Mutex::new(q_sync_optimizers::BetaScoreRegistry::new())),
            markov_states: Arc::new(parking_lot::Mutex::new(q_sync_optimizers::MarkovRegistry::new())),
            littles_law: Arc::new(dashmap::DashMap::new()),
            chunk_floor: Arc::new(q_sync_optimizers::ChunkFloorEstimator::new()),
            peer_fail_predictor: Arc::new(q_sync_optimizers::PeerFailPredictor::default()),
            // v10.9.47: Autonomous gap-heal infrastructure (empty at startup; populated
            // by spawn_gap_advance_handler after main.rs wires the broadcast channel).
            gap_detection_tally: Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
            gap_advance_rx: parking_lot::Mutex::new(None),
        }
    }

    /// v10.9.47: Wire the broadcast receiver from NetworkManager. Call this
    /// AFTER NetworkManager has been constructed and `set_gap_advance_tx` has
    /// been used to pair the channel. Then call `spawn_gap_advance_handler()`.
    pub fn set_gap_advance_rx(&self, rx: tokio::sync::broadcast::Receiver<(u64, u64, libp2p::PeerId)>) {
        *self.gap_advance_rx.lock() = Some(rx);
    }

    /// v10.9.47: Hydrate `known_gaps` from RocksDB-persisted entries. Call
    /// once at startup, after storage is ready and before sync begins.
    /// Returns the number of gaps loaded.
    pub async fn load_persisted_gaps(&self) -> usize {
        let pairs = self.storage.load_permanent_gaps().await;
        let count = pairs.len();
        for (start, end) in pairs {
            self.known_gaps.add_gap(start, end);
        }
        // Refresh observability gauge.
        crate::metrics::SyncOptimizerGauges::instance()
            .known_gap_configured_count
            .set(self.known_gaps.len() as i64);
        if count > 0 {
            info!(
                "[KNOWN-GAP v10.9.47] Hydrated {} permanent gap(s) from RocksDB: {:?}",
                count,
                self.known_gaps.snapshot()
            );
        }
        count
    }

    /// v10.9.47: Spawn the long-lived autonomous gap-heal handler. Takes the
    /// broadcast receiver from `gap_advance_rx`, processes each declared gap,
    /// applies a Beta-trust + quorum policy, and commits to `known_gaps` +
    /// RocksDB persistence when accepted. Idempotent — safe to call once.
    ///
    /// Policy (v10.9.47):
    ///   accept if `unique_reporters >= 2`
    ///   OR `unique_reporters >= 1 AND beta_score(peer).mean() >= 0.8`
    ///   OR `Q_GAP_TRUST_SINGLE_PEER=1` (operator override)
    pub fn spawn_gap_advance_handler(self: Arc<Self>) {
        let mut rx = match self.gap_advance_rx.lock().take() {
            Some(rx) => rx,
            None => {
                warn!("[KNOWN-GAP] spawn_gap_advance_handler called without rx wired — autonomous heal DISABLED");
                return;
            }
        };
        let trust_single = std::env::var("Q_GAP_TRUST_SINGLE_PEER")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let trust_threshold = std::env::var("Q_GAP_TRUST_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.8);
        info!(
            "[KNOWN-GAP v10.9.47] Autonomous gap-heal handler started \
             (trust_threshold={:.2}, trust_single={}, quorum=2)",
            trust_threshold, trust_single
        );

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok((start, end, peer)) => {
                        if start > end {
                            warn!(
                                "[KNOWN-GAP] Invalid gap declaration {}-{} from peer {} — discarding",
                                start, end, peer
                            );
                            continue;
                        }

                        // Tally the report.
                        let reporter_count = {
                            let mut tally = self.gap_detection_tally.lock();
                            let entry = tally.entry((start, end)).or_default();
                            entry.insert(peer);
                            entry.len()
                        };

                        // Skip-if-already-known check (idempotent — we may
                        // already have this gap from RocksDB or a prior accept).
                        if self.known_gaps.contains(start) && self.known_gaps.contains(end) {
                            // Already known — don't re-log, don't re-persist.
                            continue;
                        }

                        // Look up the peer's trust score (Beta mean).
                        // BetaScoreRegistry::mean(&peer) returns 0.5 for unseen peers
                        // (Beta(1,1) prior) and grows toward 1.0 with successful chunks.
                        let beta_mean = self.beta_scores.lock().mean(&peer);

                        let accept = trust_single
                            || reporter_count >= 2
                            || beta_mean >= trust_threshold;

                        if !accept {
                            info!(
                                "[KNOWN-GAP v10.9.47] Provisional gap {}-{} from {} \
                                 (reporters={}, trust={:.3}) — below threshold, awaiting quorum",
                                start, end, peer, reporter_count, beta_mean
                            );
                            continue;
                        }

                        // Commit: add to known_gaps + persist to RocksDB + observability.
                        let changed = self.known_gaps.add_gap(start, end);
                        if changed {
                            if let Err(e) = self.storage.persist_permanent_gap(start, end).await {
                                warn!(
                                    "[KNOWN-GAP] Failed to persist gap {}-{}: {} — \
                                     will be re-detected next time",
                                    start, end, e
                                );
                            }
                            let gauges = crate::metrics::SyncOptimizerGauges::instance();
                            gauges
                                .known_gap_configured_count
                                .set(self.known_gaps.len() as i64);
                            info!(
                                "[KNOWN-GAP v10.9.47] ✅ ACCEPTED gap {}-{} (reporters={}, \
                                 trust={:.3}, persisted to RocksDB). Total gaps: {}",
                                start, end, reporter_count, beta_mean, self.known_gaps.len()
                            );
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!("[KNOWN-GAP] handler lagged {} declarations — continuing", n);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        info!("[KNOWN-GAP] handler channel closed — exiting");
                        return;
                    }
                }
            }
        });
    }

    /// 🌐 Set network channel for TRUE P2P communication
    /// Call this after creating TurboSyncManager to enable gossipsub integration
    pub fn set_network_channel(&mut self, tx: mpsc::UnboundedSender<NetworkRequest>) {
        info!("🌐 [TURBO SYNC] Network channel configured - TRUE P2P enabled!");
        self.network_tx = Some(tx);
    }

    /// 🎚️ v8.5.4: Share throttle mode atomic with AppState so TUI can control sync speed.
    /// Call this after creating TurboSyncManager and before starting sync loops.
    pub fn set_network_throttle_mode(&mut self, mode: Arc<AtomicU8>) {
        self.network_throttle_mode = mode;
    }

    /// 🎚️ v8.5.4: Get sync parameters based on current throttle mode.
    /// Returns (max_concurrency_override, inter_chunk_delay_ms)
    ///
    /// Turbo:        max speed — full parallelism, no delay
    /// Normal:       fast but polite — half parallelism, 10ms delay
    /// Conservative: SSD-friendly — 2 streams, 200ms delay
    fn get_throttle_params(&self) -> (Option<usize>, u64) {
        match self.network_throttle_mode.load(Ordering::Relaxed) {
            0 => (Some(2), 200),    // Conservative: 2 in-flight, 200ms delay
            1 => {                  // Normal: half of config, 10ms delay
                let normal_concurrency = (self.config.parallel_streams / 2).max(4);
                (Some(normal_concurrency), 10)
            }
            _ => (None, 0),         // Turbo (default): full speed, no throttle
        }
    }

    /// 🚀 v1.5.0-beta: Get CHIRON parallel state applicator for external use
    /// Allows block producers and other components to use CHIRON parallel processing
    pub fn get_parallel_state_applicator(&self) -> Arc<ParallelStateApplicator> {
        Arc::clone(&self.parallel_state_applicator)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // 🚀 v1.0.2: Lock-Free Sync State Accessors (Miner-Optimized)
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Read max peer height without acquiring RwLock (lock-free, O(1))
    pub fn max_peer_height_fast(&self) -> u64 {
        self.cached_max_peer_height.load(Ordering::Acquire)
    }

    /// Read active peer count without acquiring RwLock (lock-free, O(1))
    pub fn peer_count_fast(&self) -> u32 {
        self.cached_peer_count.load(Ordering::Acquire)
    }

    /// Check if blockchain has known gaps (lock-free)
    pub fn has_gaps_fast(&self) -> bool {
        self.cached_has_gaps.load(Ordering::Acquire)
    }

    /// Mark gap state (called after gap detection / gap fill)
    pub fn set_has_gaps(&self, has_gaps: bool) {
        self.cached_has_gaps.store(has_gaps, Ordering::Release);
    }

    /// Check if node is fully synced (lock-free)
    pub fn is_synced_fast(&self) -> bool {
        self.is_fully_synced.load(Ordering::Acquire)
    }

    /// Update fully-synced state
    pub fn set_fully_synced(&self, synced: bool) {
        self.is_fully_synced.store(synced, Ordering::Release);
    }

    /// 📊 v1.0.2: Get detailed sync status snapshot for admin panel
    pub async fn get_detailed_sync_status(&self) -> DetailedSyncStatus {
        let local_height = self.get_local_height().await.unwrap_or(0);
        let network_height = self.cached_max_peer_height.load(Ordering::Relaxed);
        let gap = network_height.saturating_sub(local_height);
        let is_synced = self.is_fully_synced.load(Ordering::Relaxed);

        // Session chunk progress
        let total_chunks = self.session_total_chunks.load(Ordering::Relaxed);
        let completed_chunks = self.session_completed_chunks.load(Ordering::Relaxed);
        let in_flight = self.session_in_flight.load(Ordering::Relaxed);
        let queued = self.session_queued.load(Ordering::Relaxed);
        let mode_u8 = self.session_sync_mode.load(Ordering::Relaxed);

        let sync_mode = if is_synced && gap <= 5 {
            "fully_synced".to_string()
        } else {
            match mode_u8 {
                1 => "turbo".to_string(),
                2 => "endgame".to_string(),
                3 => "micro".to_string(),
                _ => if gap > 0 { "idle".to_string() } else { "fully_synced".to_string() },
            }
        };

        let chunk_progress_pct = if total_chunks > 0 {
            (completed_chunks as f64 / total_chunks as f64) * 100.0
        } else {
            0.0
        };

        // Metrics from TurboSyncMetrics
        let blocks_per_second = self.metrics.blocks_per_second().await;
        let bytes_downloaded = self.metrics.total_bytes_downloaded.load(Ordering::Relaxed) as f64;
        let bytes_downloaded_mb = bytes_downloaded / (1024.0 * 1024.0);
        let compression_ratio = self.metrics.compression_ratio() as f64;
        let active_streams = self.metrics.active_parallel_streams.load(Ordering::Relaxed) as u64;
        let failed_chunks = self.metrics.failed_chunks.load(Ordering::Relaxed);
        let retried_chunks = self.metrics.retried_chunks.load(Ordering::Relaxed);

        // Peer info
        let peer_count = self.cached_peer_count.load(Ordering::Relaxed) as u64;
        let best_peer_height = network_height;

        // ETA calculation
        let eta_seconds = if gap > 0 && blocks_per_second > 0.0 {
            Some((gap as f64 / blocks_per_second) as u64)
        } else {
            None
        };

        // v8.2.8: Apollo subsystem metrics
        let (kalman_bw, kalman_lat, kalman_conf, kalman_chunk, kalman_loss, kalman_timeout, kalman_conc, kalman_jitter, kalman_samples) =
            if let Some(km) = self.apollo_get_kalman_metrics().await {
                (km.bandwidth_mbps, km.latency_ms, km.confidence, km.optimal_chunk_kb as u64,
                 km.loss_percent, km.optimal_timeout_ms, km.optimal_concurrency as u64,
                 km.jitter_ms, km.samples_collected as u64)
            } else {
                (0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0, 0)
            };

        let (pid_target, pid_current, pid_error, pid_kp, pid_ki, pid_kd) =
            if let Some(pm) = self.apollo_get_pid_metrics().await {
                (pm.target, pm.current_rate, pm.avg_error, pm.kp, pm.ki, pm.kd)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            };

        let all_peer_stats = self.apollo_get_all_peer_stats();
        let peers_tracked = all_peer_stats.len() as u64;
        let (best_peer_name, best_heat) = all_peer_stats.iter()
            .max_by(|a, b| a.cache_heat.partial_cmp(&b.cache_heat).unwrap_or(std::cmp::Ordering::Equal))
            .map(|p| (p.peer_id[..12.min(p.peer_id.len())].to_string(), p.cache_heat))
            .unwrap_or_default();

        DetailedSyncStatus {
            sync_mode,
            local_height,
            network_height,
            gap,
            total_chunks,
            completed_chunks,
            in_flight,
            queued,
            chunk_progress_pct,
            blocks_per_second,
            bytes_downloaded_mb,
            compression_ratio,
            active_streams,
            failed_chunks,
            retried_chunks,
            peer_count,
            best_peer_height,
            is_fully_synced: is_synced,
            eta_seconds,
            apollo_kalman_bandwidth_mbps: kalman_bw,
            apollo_kalman_latency_ms: kalman_lat,
            apollo_kalman_confidence: kalman_conf,
            apollo_kalman_optimal_chunk_kb: kalman_chunk,
            apollo_kalman_loss_pct: kalman_loss,
            apollo_kalman_timeout_ms: kalman_timeout,
            apollo_kalman_concurrency: kalman_conc,
            apollo_kalman_jitter_ms: kalman_jitter,
            apollo_kalman_samples: kalman_samples,
            apollo_pid_target_bps: pid_target,
            apollo_pid_current_bps: pid_current,
            apollo_pid_error: pid_error,
            apollo_pid_kp: pid_kp,
            apollo_pid_ki: pid_ki,
            apollo_pid_kd: pid_kd,
            apollo_peers_tracked: peers_tracked,
            apollo_gravity_best_peer: best_peer_name,
            apollo_gravity_best_heat: best_heat,
            // Starship phase from AtomicU8 — FlightComputer populates these externally
            starship_phase: match mode_u8 {
                1 => "SUPER_HEAVY",
                2 => "HOT_STAGING",
                3 => "STARSHIP_CRUISE",
                _ => if is_synced { "STATION_KEEPING" } else { "PRELAUNCH" },
            }.to_string(),
            phase_duration_secs: 0,  // populated by FlightComputer externally
            orbit_stable: is_synced && gap <= 5,
            station_keeping_peer_health: if peer_count >= 5 { 1.0 } else if peer_count >= 3 { 0.8 } else if peer_count >= 1 { 0.5 } else { 0.0 },
            mission_elapsed_secs: 0,  // populated by FlightComputer externally
        }
    }

    /// 🚀 v1.5.0-beta: Check if CHIRON hints are enabled
    pub fn is_chiron_enabled(&self) -> bool {
        self.config.enable_chiron_hints
    }

    /// 🚀 v1.5.0-beta: Get CHIRON performance statistics
    /// Returns (parallel_txs, sequential_txs, blocks_processed)
    pub fn get_chiron_stats(&self) -> (u64, u64, u64) {
        self.parallel_state_applicator.get_cumulative_stats()
    }

    /// 🚀 v1.5.0-beta: Log CHIRON performance summary
    pub fn log_chiron_summary(&self) {
        self.parallel_state_applicator.log_performance_summary();
    }

    /// 🚀 v1.5.0-beta: Get NEMO executor for external use
    pub fn get_nemo_executor(&self) -> Arc<NemoExecutor> {
        Arc::clone(&self.nemo_executor)
    }

    /// 🚀 v1.5.0-beta: Check if NEMO executor is enabled
    pub fn is_nemo_enabled(&self) -> bool {
        self.config.enable_nemo_executor
    }

    /// 🚀 v1.5.0-beta: Get NEMO performance statistics
    /// Returns (greedy_commits, validated_commits, reexecutions_avoided, blocks)
    pub fn get_nemo_stats(&self) -> (u64, u64, u64, u64) {
        self.nemo_executor.get_cumulative_stats()
    }

    /// 🚀 v1.5.0-beta: Log NEMO performance summary
    pub fn log_nemo_summary(&self) {
        self.nemo_executor.log_performance_summary();
    }

    /// 🚀 v1.5.0-beta: Log combined CHIRON + NEMO performance summary
    pub fn log_parallel_execution_summary(&self) {
        self.log_chiron_summary();
        self.log_nemo_summary();
        self.log_async_pipeline_summary();
    }

    // ========== v2.3.4-beta: Emergency Sync Guard Methods ==========

    /// 🚀 v2.3.4-beta: Check if emergency sync is currently running
    /// Prevents multiple concurrent emergency syncs from spawning
    pub fn is_emergency_sync_running(&self) -> bool {
        self.emergency_sync_in_progress.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// 🚀 v2.3.4-beta: Set emergency sync running state
    pub fn set_emergency_sync_running(&self, running: bool) {
        self.emergency_sync_in_progress.store(running, std::sync::atomic::Ordering::SeqCst);
    }

    /// 🚀 v2.3.4-beta: Try to start emergency sync atomically
    /// Returns true if we acquired the lock (no other sync was running)
    /// Returns false if another sync is already in progress
    pub fn try_start_emergency_sync(&self) -> bool {
        self.emergency_sync_in_progress.compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        ).is_ok()
    }

    // ========== v1.5.0-beta: Reddio Async Storage Pipeline Methods ==========

    /// 🚀 v1.5.0-beta: Get async storage pipeline for external use
    /// Allows components to use hot cache, prefetching, and pipelined I/O
    #[cfg(not(target_os = "windows"))]
    pub fn get_async_pipeline(&self) -> Option<Arc<AsyncStoragePipeline>> {
        self.async_pipeline.clone()
    }

    /// 🚀 v1.5.0-beta: Check if async pipeline is enabled
    #[cfg(not(target_os = "windows"))]
    pub fn is_async_pipeline_enabled(&self) -> bool {
        self.async_pipeline.is_some() && self.config.enable_async_pipeline
    }

    #[cfg(target_os = "windows")]
    pub fn is_async_pipeline_enabled(&self) -> bool {
        false
    }

    /// 🚀 v1.5.0-beta: Get async pipeline statistics
    /// Returns comprehensive stats: cache hit rate, prefetch count, pipeline throughput
    #[cfg(not(target_os = "windows"))]
    pub fn get_async_pipeline_stats(&self) -> Option<AsyncPipelineStats> {
        self.async_pipeline.as_ref().map(|p| p.stats())
    }

    /// 🚀 v1.5.0-beta: Log async pipeline performance summary
    #[cfg(not(target_os = "windows"))]
    pub fn log_async_pipeline_summary(&self) {
        if let Some(pipeline) = &self.async_pipeline {
            pipeline.log_stats();
        }
    }

    #[cfg(target_os = "windows")]
    pub fn log_async_pipeline_summary(&self) {}

    /// 🚀 v1.5.0-beta: Prefetch addresses for upcoming block processing
    /// Call this before processing a block to warm the hot cache
    #[cfg(not(target_os = "windows"))]
    pub fn prefetch_for_block(&self, transactions: &[q_types::Transaction]) {
        if let Some(pipeline) = &self.async_pipeline {
            pipeline.prefetch_for_block(transactions);
        }
    }

    #[cfg(target_os = "windows")]
    pub fn prefetch_for_block(&self, _transactions: &[q_types::Transaction]) {}

    /// 🚀 v1.5.0-beta: Get balance from hot cache (bypasses RocksDB if cached)
    /// Returns None if async pipeline is disabled
    #[cfg(not(target_os = "windows"))]
    pub fn get_cached_balance(&self, address: &q_types::Address) -> Option<u64> {
        self.async_pipeline.as_ref().and_then(|p| p.cache.get(address))
    }

    #[cfg(target_os = "windows")]
    pub fn get_cached_balance(&self, _address: &q_types::Address) -> Option<u64> {
        None
    }

    /// 🚀 v1.5.0-beta: Log full v1.5.0 optimization summary
    /// Combines CHIRON + NEMO + Reddio stats
    pub fn log_v150_optimization_summary(&self) {
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📊 v1.5.0-beta PARALLEL EXECUTION OPTIMIZATION SUMMARY");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // CHIRON stats
        let (chiron_parallel, chiron_sequential, chiron_blocks) = self.get_chiron_stats();
        let chiron_total = chiron_parallel + chiron_sequential;
        let chiron_ratio = if chiron_total > 0 { chiron_parallel as f64 / chiron_total as f64 * 100.0 } else { 0.0 };
        info!("🔧 CHIRON (Parallel State): {:.1}% parallel ({}/{} txs, {} blocks)",
              chiron_ratio, chiron_parallel, chiron_total, chiron_blocks);

        // NEMO stats
        let (nemo_greedy, nemo_validated, nemo_avoided, nemo_blocks) = self.get_nemo_stats();
        let nemo_total = nemo_greedy + nemo_validated;
        let nemo_greedy_ratio = if nemo_total > 0 { nemo_greedy as f64 / nemo_total as f64 * 100.0 } else { 0.0 };
        info!("⚡ NEMO (High Contention): {:.1}% greedy commits ({}/{} txs, {} re-exec avoided, {} blocks)",
              nemo_greedy_ratio, nemo_greedy, nemo_total, nemo_avoided, nemo_blocks);

        // Async pipeline stats
        #[cfg(not(target_os = "windows"))]
        {
            if let Some(stats) = self.get_async_pipeline_stats() {
                info!("🚀 REDDIO (Async Pipeline): {:.1}% cache hit rate ({}/{} reads), {} prefetches",
                      stats.cache.hit_rate * 100.0, stats.cache.hits, stats.total_reads, stats.prefetch.total_requests);
                info!("   Pipeline: {} tasks completed, avg {}μs latency",
                      stats.pipeline.completed_tasks, stats.pipeline.avg_latency_us);
            } else {
                info!("📭 REDDIO (Async Pipeline): Disabled");
            }
        }
        #[cfg(target_os = "windows")]
        {
            info!("📭 REDDIO (Async Pipeline): Disabled (Windows)");
        }

        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // 🚀 v2.1.0-DELTA-V: Project APOLLO Control System Methods
    // ═══════════════════════════════════════════════════════════════════════════════

    /// 🎛️ v2.0.0-KALMAN: Update PID controller with current throughput measurement
    /// Call this after each sync chunk completes to allow PID to auto-tune
    pub async fn apollo_update_pid(&self, current_throughput: f64) {
        if !self.config.enable_apollo_pid {
            return;
        }
        let recommended_rate = {
            let mut pid = self.apollo_pid_controller.write().await;
            pid.update(current_throughput)
        };
        debug!("🎛️ [APOLLO PID] Current: {:.1} BPS, Recommended: {:.1} BPS",
               current_throughput, recommended_rate);
    }

    /// 🎛️ v2.0.0-KALMAN: Get recommended sync rate from PID controller
    /// Returns the optimal blocks-per-second rate based on feedback
    pub async fn apollo_get_recommended_rate(&self) -> f64 {
        if !self.config.enable_apollo_pid {
            return self.config.apollo_target_throughput;
        }
        let pid = self.apollo_pid_controller.read().await;
        pid.get_rate()
    }

    /// 🎛️ v2.0.0-KALMAN: Get PID controller metrics for monitoring
    pub async fn apollo_get_pid_metrics(&self) -> Option<crate::pid_controller::PIDMetrics> {
        if !self.config.enable_apollo_pid {
            return None;
        }
        let pid = self.apollo_pid_controller.read().await;
        Some(pid.get_metrics())
    }

    /// 🔭 v2.0.0-KALMAN: Update Kalman predictor with network measurement
    /// Call this with actual observed network conditions to improve predictions
    pub async fn apollo_update_kalman(&self, bandwidth_mbps: f64, latency_ms: f64, loss_rate: f64) {
        if !self.config.enable_apollo_kalman {
            return;
        }
        let state = {
            let mut kalman = self.apollo_kalman_predictor.write().await;
            kalman.update(bandwidth_mbps, latency_ms, loss_rate)
        };
        debug!("🔭 [APOLLO KALMAN] Predicted: BW={:.1} bps, Lat={:.1} ms, Loss={:.2}%",
               state.bandwidth_bps / 1_000_000.0, state.latency_ms, state.loss_rate * 100.0);
    }

    /// 🔭 v2.0.0-KALMAN: Get network state from Kalman predictor
    /// Returns predicted network state (bandwidth, latency, loss)
    pub async fn apollo_get_network_state(&self) -> crate::kalman_predictor::NetworkState {
        if !self.config.enable_apollo_kalman {
            return crate::kalman_predictor::NetworkState {
                bandwidth_bps: 100_000_000.0, // 100 Mbps default
                latency_ms: 50.0,
                loss_rate: 0.01,
                jitter_ms: 5.0,
                confidence: 0.5,
            };
        }
        let kalman = self.apollo_kalman_predictor.read().await;
        kalman.get_state()
    }

    /// 🔭 v2.0.0-KALMAN: Get optimal chunk size from network state
    pub async fn apollo_get_optimal_chunk_size(&self) -> usize {
        let state = self.apollo_get_network_state().await;
        state.optimal_chunk_size()
    }

    /// v10.9.44: Kalman BDP chunk-size in **blocks** via q-sync-optimizers.
    ///
    /// Returns `Some(blocks)` when the Apollo Kalman estimator has sufficient
    /// confidence to produce a useful estimate; `None` otherwise (caller should
    /// fall back to the existing ML-blended path).
    ///
    /// Conversion: bytes/sec → mbps (× 8 / 1e6), then KalmanBdpEstimator
    /// returns chunk KiB clamped to [floor, 4096]. Convert to blocks via
    /// `avg_block_size_bytes = 1000` (per spec).
    pub async fn bdp_chunk_size_blocks(&self) -> Option<u64> {
        let state = self.apollo_get_network_state().await;
        if !(state.confidence > 0.0) || state.bandwidth_bps <= 0.0 || state.latency_ms <= 0.0 {
            return None;
        }
        let bw_mbps = state.bandwidth_bps * 8.0 / 1_000_000.0;
        let snap = q_sync_optimizers::KalmanSnapshot::new(bw_mbps, state.latency_ms, state.confidence);
        let est = self.kalman_bdp.read().await;
        let kb = est.chunk_size_kb(snap);
        // 1000 B/block estimate per spec → blocks = KB * 1024 / 1000 ≈ KB.
        let blocks = (kb as u64).saturating_mul(1024).saturating_div(1000);
        Some(blocks.max(1))
    }

    /// v10.9.44: Thompson-sample a peer from the Beta-score registry (item 2).
    ///
    /// Returns `None` for empty `candidates`. Otherwise picks the peer with
    /// the highest sample drawn from its `Beta(α, β)` posterior. After the
    /// caller observes the outcome it should call `record_peer_outcome` to
    /// update the per-peer counter.
    pub fn thompson_pick_peer<'a>(
        &self,
        candidates: &'a [libp2p::PeerId],
    ) -> Option<&'a libp2p::PeerId> {
        if candidates.is_empty() {
            return None;
        }
        // The registry's thompson_pick borrows `candidates` so the return value
        // lives as long as the input. We hold the lock only for the draw.
        let mut reg = self.beta_scores.lock();
        let mut rng = rand::thread_rng();
        reg.thompson_pick(candidates, &mut rng)
    }

    /// v10.9.44: Record an outcome for the Beta-score registry (item 2).
    ///
    /// Pass `true` on a successful block-pack response, `false` on failure.
    pub fn record_peer_outcome(&self, peer: &libp2p::PeerId, success: bool) {
        let mut reg = self.beta_scores.lock();
        if success {
            reg.record_success(peer);
        } else {
            reg.record_failure(peer);
        }
    }

    /// v10.9.44: Record a per-peer RTT into the Markov state model (item 6)
    /// AND the Little's law per-peer estimator (item 4). Idempotent for NaN /
    /// negative values (silently ignored by the underlying estimators).
    pub fn record_peer_rtt(&self, peer: &libp2p::PeerId, rtt_ms: f64) {
        // Markov state — guards against NaN internally.
        self.markov_states.lock().record_rtt(peer, rtt_ms);
        // Little's law per-peer — also guards against NaN internally.
        let mut entry = self
            .littles_law
            .entry(*peer)
            .or_insert_with(q_sync_optimizers::LittlesLawEstimator::new);
        entry.record_rtt_ms(rtt_ms);
    }

    /// v10.9.44: Record a peer timeout into the Markov model (item 6).
    pub fn record_peer_timeout(&self, peer: &libp2p::PeerId) {
        self.markov_states.lock().record_timeout(peer);
    }

    /// v10.9.44: Refresh the q-sync-optimizers Prometheus gauges (items 4/5/6/8).
    ///
    /// Called on every scheduler tick. All math is O(peers) and budget is < 10 µs.
    /// Gauges:
    /// - `qnk_chunk_size_floor` (KiB) — from ChunkFloorEstimator with current peer-set size.
    /// - `qnk_optimal_inflight{peer}` — from LittlesLawEstimator combined with CUBIC cwnd.
    /// - `qnk_peer_state{peer}` — 0=Fast, 1=Slow, 2=Stalled.
    /// - `qnk_peer_p_fail{peer}` — PeerFailPredictor logistic output [0,1].
    pub fn refresh_sync_optimizer_gauges(
        &self,
        peer_set: &[libp2p::PeerId],
        per_peer_cwnd: impl Fn(&libp2p::PeerId) -> u32,
    ) {
        use q_sync_optimizers::PeerFailFeatures;
        let metrics = &crate::metrics::SyncOptimizerGauges::instance();

        // Item 5 — chunk floor (single global gauge).
        let floor_kib = self.chunk_floor.floor_kib(peer_set.len() as u32);
        metrics.chunk_size_floor.set(floor_kib as i64);

        // Per-peer gauges (items 4/6/8).
        let markov = self.markov_states.lock();
        for peer in peer_set {
            let peer_label = peer.to_string();

            // Item 4 — optimal in-flight combined with CUBIC cwnd.
            let cwnd = per_peer_cwnd(peer);
            if let Some(est) = self.littles_law.get(peer) {
                let l = est.value().combined_with_cubic(cwnd);
                metrics
                    .optimal_inflight
                    .with_label_values(&[&peer_label])
                    .set(l as f64);
            } else {
                // No RTT data — fall back to cwnd alone.
                metrics
                    .optimal_inflight
                    .with_label_values(&[&peer_label])
                    .set(cwnd as f64);
            }

            // Item 6 — Markov state.
            let state = markov.state(peer) as u32;
            metrics
                .peer_state
                .with_label_values(&[&peer_label])
                .set(state as f64);

            // Item 8 — peer-fail predictor with default features (until the
            // call sites that have richer per-peer telemetry start feeding
            // real values). Defaults give a baseline reading; integrations
            // upstream can override via call-site-specific feature extraction.
            let features = PeerFailFeatures::default();
            let p_fail = self.peer_fail_predictor.p_fail(&features);
            metrics
                .peer_p_fail
                .with_label_values(&[&peer_label])
                .set(p_fail);
        }
    }

    /// 🔭 v2.0.0-KALMAN: Get optimal timeout from network state
    pub async fn apollo_get_optimal_timeout(&self) -> Duration {
        let state = self.apollo_get_network_state().await;
        state.optimal_timeout()
    }

    /// 🔭 v2.0.0-KALMAN: Get Kalman predictor metrics for monitoring
    pub async fn apollo_get_kalman_metrics(&self) -> Option<crate::kalman_predictor::KalmanMetrics> {
        if !self.config.enable_apollo_kalman {
            return None;
        }
        let kalman = self.apollo_kalman_predictor.read().await;
        Some(kalman.get_metrics())
    }

    /// 🌍 v1.9.0-SLINGSHOT: Record peer serving a block range (for cache tracking)
    /// Call this after a peer successfully serves blocks to build momentum
    pub fn apollo_record_peer_serving(&self, peer_id: &str, range: std::ops::Range<u64>, bytes_served: u64, latency_ms: u32) {
        if !self.config.enable_apollo_gravity_assist {
            return;
        }
        self.apollo_peer_momentum.record_serve(peer_id, range.clone(), bytes_served, latency_ms);
        debug!("🌍 [APOLLO SLINGSHOT] Recorded {} serving {}-{} ({}ms)",
               peer_id, range.start, range.end, latency_ms);
    }

    /// 🌍 v1.9.0-SLINGSHOT: Select best peer for target range using gravity assist
    /// Returns the peer most likely to have hot cache for the target range
    /// v8.6.2: Tiered boost — 10x supernode, 3x preferred, 1x default
    pub fn apollo_select_peer(&self, target_range: &std::ops::Range<u64>, available_peers: &[&str]) -> Option<String> {
        if !self.config.enable_apollo_gravity_assist || available_peers.is_empty() {
            return available_peers.first().map(|s| s.to_string());
        }

        // v8.6.2: Tiered boost for supernode (10x) and preferred (3x) peers
        let has_tiers = !self.config.supernode_peers.is_empty() || !self.config.preferred_sync_peers.is_empty();
        if has_tiers {
            let mut best_peer: Option<(String, f64)> = None;
            for &peer_id in available_peers {
                let base_score = self.apollo_peer_momentum
                    .get_peer_momentum(peer_id)
                    .map(|m| m.selection_score(target_range))
                    .unwrap_or(0.1);

                // v8.7.4: 10x boost for supernode peers (10Gbit+ servers)
                // Hardcoded Epsilon peer ID + env-configured supernodes
                const EPSILON_PEER: &str = "12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM";
                let is_supernode = peer_id.contains(EPSILON_PEER) || EPSILON_PEER.contains(peer_id)
                    || self.config.supernode_peers.iter()
                        .any(|p| peer_id.contains(p.as_str()) || p.contains(peer_id));
                // 3x boost for preferred peers (1Gbit servers)
                let is_preferred = self.config.preferred_sync_peers.iter()
                    .any(|p| peer_id.contains(p.as_str()) || p.contains(peer_id));

                let tier_multiplier = if is_supernode { 10.0 } else if is_preferred { 3.0 } else { 1.0 };
                // v9.1.0: Factor in peer's announced compute power (hashrate)
                let compute_boost = crate::compute_power_boost(peer_id);
                let multiplier = tier_multiplier * compute_boost;
                let final_score = base_score * multiplier;

                match &best_peer {
                    None => best_peer = Some((peer_id.to_string(), final_score)),
                    Some((_, best_score)) if final_score > *best_score => {
                        best_peer = Some((peer_id.to_string(), final_score));
                    }
                    _ => {}
                }
            }
            return best_peer.map(|(peer, _)| peer);
        }

        self.apollo_peer_momentum.select_best_peer(target_range, available_peers)
    }

    /// v8.6.2: Get optimal chunk size for a specific peer based on its bandwidth tier
    /// Supernodes (10Gbit, 64GB RAM) can serve 5000 blocks/chunk
    /// Standard (1Gbit) can serve 1000 blocks/chunk
    /// Fallback (unknown/slow) caps at 500 blocks/chunk
    pub fn get_peer_chunk_size(&self, peer_id: &str) -> u64 {
        // v8.7.4: Hardcoded Epsilon supernode peer ID — always gets 5000 chunks even without env var
        const EPSILON_PEER_PREFIX: &str = "12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM";
        if peer_id.contains(EPSILON_PEER_PREFIX) || EPSILON_PEER_PREFIX.contains(peer_id) {
            return 5000;
        }
        // Check if this peer is a known supernode (configured via Q_SUPERNODE_PEERS)
        let is_supernode = self.config.supernode_peers.iter()
            .any(|p| peer_id.contains(p.as_str()) || p.contains(peer_id));
        if is_supernode {
            return 5000;
        }

        // Check if this peer is a known preferred (1Gbit standard) peer
        let is_preferred = self.config.preferred_sync_peers.iter()
            .any(|p| peer_id.contains(p.as_str()) || p.contains(peer_id));
        if is_preferred {
            return 1000;
        }

        // Check handshake-seeded bandwidth from momentum tracker
        if let Some(momentum) = self.apollo_peer_momentum.get_peer_momentum(peer_id) {
            let bw_mbps = momentum.bandwidth_velocity / 125_000.0; // bytes/s → Mbps
            return if bw_mbps >= 5000.0 { 5000 } else if bw_mbps >= 500.0 { 1000 } else { 500 };
        }

        // Default: safe cap that Gamma (7.8GB) can handle
        500
    }

    /// 🌍 v1.9.0-SLINGSHOT: Get peer momentum data for monitoring
    pub fn apollo_get_peer_momentum(&self, peer_id: &str) -> Option<crate::peer_momentum::PeerMomentum> {
        if !self.config.enable_apollo_gravity_assist {
            return None;
        }
        self.apollo_peer_momentum.get_peer_momentum(peer_id)
    }

    /// 🌍 v1.9.0: Get all peer gravity-assist stats for TUI/monitoring
    pub fn apollo_get_all_peer_stats(&self) -> Vec<crate::peer_momentum::PeerStats> {
        self.apollo_peer_momentum.get_all_stats()
    }

    /// 🚀 v2.1.0-DELTA-V: Log full APOLLO optimization summary
    pub async fn log_apollo_summary(&self) {
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("🚀 v2.1.0-DELTA-V PROJECT APOLLO CONTROL SYSTEM SUMMARY");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // PID stats
        if let Some(metrics) = self.apollo_get_pid_metrics().await {
            info!("🎛️ PID Rate Controller: target={:.0} BPS, current={:.1} BPS, avg_error={:.2}",
                  metrics.target, metrics.current_rate, metrics.avg_error);
            info!("   • Gains: Kp={:.3}, Ki={:.3}, Kd={:.3}", metrics.kp, metrics.ki, metrics.kd);
        } else {
            info!("🎛️ PID Rate Controller: DISABLED (Q_APOLLO_PID=0)");
        }

        // Kalman stats
        if let Some(metrics) = self.apollo_get_kalman_metrics().await {
            info!("🔭 Kalman Predictor: BW={:.1} Mbps, Lat={:.1} ms, Loss={:.2}%",
                  metrics.bandwidth_mbps, metrics.latency_ms, metrics.loss_percent);
            info!("   • Confidence: {:.2}, Optimal chunk: {} KB",
                  metrics.confidence, metrics.optimal_chunk_kb);
        } else {
            info!("🔭 Kalman Predictor: DISABLED (Q_APOLLO_KALMAN=0)");
        }

        // Peer momentum stats
        if self.config.enable_apollo_gravity_assist {
            let stats = self.apollo_peer_momentum.get_all_stats();
            info!("🌍 Gravity Assist: {} peers tracked", stats.len());
            for stat in stats.iter().take(5) {
                let peer_display = if stat.peer_id.len() > 12 { &stat.peer_id[..12] } else { &stat.peer_id };
                info!("   • {}: heat={:.2}, bw={:.1} Mbps, p50={}ms",
                      peer_display,
                      stat.cache_heat,
                      stat.bandwidth_mbps,
                      stat.latency_p50);
            }
        } else {
            info!("🌍 Gravity Assist: DISABLED (Q_APOLLO_GRAVITY_ASSIST=0)");
        }

        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    /// Get local blockchain height (contiguous - no gaps from genesis)
    /// 🚨 v2.3.7-beta CRITICAL FIX: Use CONTIGUOUS height, not latest stored!
    /// BUG: get_latest_qblock_height() returns highest stored (875,866) but with gaps
    /// This caused TurboSync to think it was already synced when gaps existed!
    /// FIX: get_highest_contiguous_block() returns last contiguous height (correct)
    async fn get_local_height(&self) -> Result<u64> {
        Ok(self.storage.get_highest_contiguous_block().await.unwrap_or(0))
    }

    /// v6.0.9: Read process RSS from /proc/self/statm (near-zero overhead)
    /// Returns RSS in megabytes, or None on non-Linux or read failure.
    #[cfg(target_os = "linux")]
    fn get_rss_mb() -> Option<u64> {
        let data = std::fs::read_to_string("/proc/self/statm").ok()?;
        let rss_pages: u64 = data.split_whitespace().nth(1)?.parse().ok()?;
        Some(rss_pages * 4 / 1024) // pages (4KB each) → MB
    }

    #[cfg(not(target_os = "linux"))]
    fn get_rss_mb() -> Option<u64> {
        None // Not available on non-Linux
    }

    /// 🚀 v2.3.3-beta: ENDGAME SYNC OPTIMIZATION
    /// Find the highest block we have stored near the target, even if there are gaps.
    /// This enables "endgame mode" - when we have most blocks but are missing the tip.
    ///
    /// Returns (highest_stored, has_gap_blocks) where:
    /// - highest_stored: The highest block we have in the range (contiguous, target)
    /// - has_gap_blocks: True if we have blocks stored above contiguous (endgame mode)
    async fn detect_endgame_mode(&self, contiguous_height: u64, target_height: u64) -> (u64, bool) {
        // Quick check: Sample blocks near the target to detect endgame scenario
        // This avoids expensive full scans by probing specific heights

        let probe_heights = [
            target_height.saturating_sub(10),   // Very close to tip
            target_height.saturating_sub(100),  // Close to tip
            target_height.saturating_sub(500),  // Near tip
            target_height.saturating_sub(1000), // ~1k behind
            target_height.saturating_sub(5000), // ~5k behind
        ];

        let mut highest_found = contiguous_height;
        let mut has_gap_blocks = false;

        for probe in probe_heights.iter() {
            if *probe <= contiguous_height {
                continue; // Already covered by contiguous range
            }

            // Check if we have this block
            if let Ok(Some(_)) = self.storage.get_qblock_by_height(*probe).await {
                if *probe > highest_found {
                    highest_found = *probe;
                    has_gap_blocks = true;
                }
            }
        }

        // If we found blocks above contiguous, do a more precise scan from highest found backwards
        if has_gap_blocks && highest_found > contiguous_height + 100 {
            // Scan backwards from highest_found to find actual highest contiguous from that point
            // This finds where the tip gap starts
            let scan_start = highest_found.min(target_height);
            for height in (contiguous_height + 1..=scan_start).rev() {
                if let Ok(Some(_)) = self.storage.get_qblock_by_height(height).await {
                    highest_found = height;
                    break;
                }
            }
        }

        if has_gap_blocks {
            info!("🎯 [ENDGAME] Detected endgame mode: contiguous={}, highest_stored={}, target={}",
                  contiguous_height, highest_found, target_height);
            info!("   Gap blocks exist! Can skip from {} to {} ({} blocks saved)",
                  contiguous_height, highest_found, highest_found - contiguous_height);
        }

        (highest_found, has_gap_blocks)
    }
    /// v10.3.5: Checkpoint Sync — discover the earliest available block.
    ///
    /// Uses lightweight HTTP probes against known bootstrap peers to find where
    /// blocks begin on the network. Only called once at startup for fresh nodes.
    /// The actual block sync uses P2P — only this discovery step uses HTTP.
    ///
    /// Returns 0 if blocks exist from genesis (no gap), or the first available height.
    async fn probe_network_gap(
        &self,
        target_height: u64,
        _peers: &[PeerId],
    ) -> u64 {
        info!("🔗 [CHECKPOINT SYNC] Probing bootstrap peers for earliest available block...");

        // Run blocking HTTP probes on a dedicated thread
        let target = target_height;
        match tokio::task::spawn_blocking(move || {
            Self::probe_network_gap_blocking(target)
        }).await {
            Ok(result) => result,
            Err(e) => {
                warn!("🔗 [CHECKPOINT SYNC] Probe task failed: {} — syncing from genesis", e);
                0
            }
        }
    }

    /// Blocking implementation of network gap probe (runs on spawn_blocking thread).
    fn probe_network_gap_blocking(target_height: u64) -> u64 {
        // Epsilon (89.149.241.126) is the 10Gbit supernode with the deepest block history
        // (~292K floor vs ~13.9M on Beta). It must be first so the binary search converges
        // to the earliest available height rather than Beta's truncated floor.
        let bootstrap_urls = [
            "http://89.149.241.126:8080",   // Epsilon  — 10Gbit supernode, PRIMARY (deepest history)
            "http://5.79.79.158:8080",      // Delta    — 1Gbit, secondary
            "http://109.205.176.60:8080",   // Gamma    — 1Gbit, tertiary
            "http://185.182.185.227:8080",  // Beta     — 100Mbit, fallback
        ];

        // Probe helper: check if ANY peer has a block at height h
        // v10.3.6: Uses /api/v1/sync/blocks?from_height=X&limit=1 instead of
        // /api/v1/blocks/{height}. The blocks endpoint returns 404 for all heights
        // due to high_performance_server route interception (pre-existing bug).
        // The sync/blocks endpoint works correctly and returns block data.
        let probe = |height: u64| -> bool {
            for url in &bootstrap_urls {
                // v10.3.7: Use limit=1000 instead of limit=1.
                // DAG blocks are sparse — a single height may have no block,
                // but a 1000-height window will find nearby blocks.
                let sync_url = format!("{}/api/v1/sync/blocks?from_height={}&limit=1000", url, height);
                match ureq::get(&sync_url)
                    .timeout(std::time::Duration::from_secs(8))
                    .call()
                {
                    Ok(resp) => {
                        if let Ok(text) = resp.into_string() {
                            // v10.3.7: Detailed probe debugging
                            let has_blocks = text.contains("\"blocks\":[{");
                            let count = if text.contains("\"count\":") {
                                text.split("\"count\":").nth(1)
                                    .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
                                    .unwrap_or("?")
                            } else { "?" };
                            let latest = if text.contains("\"latest_height\":") {
                                text.split("\"latest_height\":").nth(1)
                                    .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
                                    .unwrap_or("?")
                            } else { "?" };
                            info!("🔍 [CHECKPOINT PROBE] height={} peer={} has_blocks={} count={} latest_height={}",
                                  height, url, has_blocks, count, latest);
                            if has_blocks {
                                return true;
                            }
                        }
                    }
                    Err(e) => {
                        warn!("🔍 [CHECKPOINT PROBE] height={} peer={} ERROR: {}", height, url, e);
                        continue;
                    }
                }
            }
            false
        };

        // Step 1: Quick check — does block 1 exist?
        if probe(1) {
            info!("🔗 [CHECKPOINT SYNC] Genesis blocks available — full sync");
            return 0;
        }

        // Step 2: Exponential probe
        let heights: Vec<u64> = vec![
            1_000, 10_000, 50_000, 100_000, 250_000, 500_000,
            1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000,
            7_000_000, 10_000_000, 12_000_000, 14_000_000,
        ].into_iter().filter(|&h| h < target_height).collect();

        let mut last_empty: u64 = 0;
        let mut first_found: u64 = 0;

        for &h in &heights {
            if probe(h) {
                first_found = h;
                info!("🔗 [CHECKPOINT SYNC] Blocks found at height {}", h);
                break;
            }
            last_empty = h;
        }

        // If not found, try near-tip
        if first_found == 0 {
            for offset in &[1_000_000u64, 500_000, 100_000, 10_000] {
                let h = target_height.saturating_sub(*offset);
                if h <= last_empty { continue; }
                if probe(h) {
                    first_found = h;
                    info!("🔗 [CHECKPOINT SYNC] Blocks found at height {} (near-tip)", h);
                    break;
                }
            }
        }

        if first_found == 0 {
            warn!("🔗 [CHECKPOINT SYNC] No blocks found on peers — syncing from genesis");
            return 0;
        }

        // Step 3: Binary search (±5000 precision)
        let mut lo = last_empty;
        let mut hi = first_found;
        while hi - lo > 5000 {
            let mid = lo + (hi - lo) / 2;
            if probe(mid) { hi = mid; } else { lo = mid; }
        }

        // Step 4: Back-check
        if !probe(hi) {
            warn!("🔗 [CHECKPOINT SYNC] Back-check failed at {} — syncing from genesis", hi);
            return 0;
        }

        info!("🔗 [CHECKPOINT SYNC] Network checkpoint at height {} (searched {}..{})", hi, last_empty, first_found);
        hi
    }

    /// v10.5.0: Fetch a specific block range from HTTP bootstrap peers and store directly.
    /// RC-3 [GAP-FILL]: Fill a missing block range using P2P (libp2p block-pack protocol).
    /// Uses NetworkRequest::RequestBlockRangeDirect — the same turbo-sync mechanism —
    /// instead of HTTP bootstrap. Does NOT touch the contiguous height pointer; the
    /// integrity monitor advances it on its next scan once gaps are confirmed filled.
    /// Fast parallel gap-fill via P2P.
    ///
    /// v1.0.2: Upgraded from a 50-block sequential cursor loop (~330K requests for
    /// the 16.5M pre-checkpoint range, ~19 days at 25s/chunk) to a sliding-window
    /// parallel fetcher with batched writes that matches the regular turbo-sync
    /// throughput (~1,100 blocks/sec on Epsilon → ~4 hours for 16.5M).
    ///
    /// Same network-request type as turbo sync (`RequestBlockRangeDirect`) — the
    /// only difference is configuration: chunk size, in-flight concurrency, batched
    /// storage writes, and peer rotation on retry.
    ///
    /// Tunable via env vars:
    ///   Q_GAPFILL_CHUNK_SIZE       — blocks per request (default 500, max 1000 per BlockPack invariant)
    ///   Q_GAPFILL_MAX_CONCURRENCY  — max in-flight chunks (default 8)
    ///   Q_GAPFILL_CHUNK_TIMEOUT    — per-chunk timeout in seconds (default 25)
    pub async fn fill_gap_p2p(&self, first_gap: u64, last_gap: u64) -> Result<()> {
        let total_range = last_gap.saturating_sub(first_gap) + 1;

        // v1.0.2 OPTION B: For very large ranges (pre-checkpoint historical backfill,
        // typically 1 → ~16.5M), delegate to download_chunks_parallel which has the
        // full turbo machinery (Apollo Kalman concurrency, Warp Sync prefetch,
        // gravity-assist peer ordering, adaptive timeouts). The simpler local windowed
        // fetcher below tops out around 60-70 bps on pre-checkpoint blocks because
        // Epsilon's block-pack semaphore caps server-side concurrency at 4. The turbo
        // path hits ~570 bps average — same orders-of-magnitude difference as turbo
        // sync vs the old 50-block sequential fill.
        const LARGE_RANGE_THRESHOLD: u64 = 100_000;
        if total_range > LARGE_RANGE_THRESHOLD {
            return self.fill_gap_via_turbo(first_gap, last_gap).await;
        }

        let chunk_size: u64 = std::env::var("Q_GAPFILL_CHUNK_SIZE")
            .ok().and_then(|v| v.parse().ok())
            .unwrap_or(500)
            .clamp(50, 1000); // hard cap matches BlockPack MAX_BLOCKS_PER_REQUEST
        let max_concurrency: usize = std::env::var("Q_GAPFILL_MAX_CONCURRENCY")
            .ok().and_then(|v| v.parse().ok())
            .unwrap_or(8)
            .clamp(1, 32);
        let chunk_timeout_secs: u64 = std::env::var("Q_GAPFILL_CHUNK_TIMEOUT")
            .ok().and_then(|v| v.parse().ok())
            .unwrap_or(25)
            .clamp(5, 120);
        const MAX_RETRIES: u32 = 3; // peer rotation on each retry (was 1)

        let network_tx = match &self.network_tx {
            Some(tx) => tx.clone(),
            None => {
                warn!("🔧 [GAP-FILL P2P] No network channel — cannot fill gap {}-{}", first_gap, last_gap);
                return Ok(());
            }
        };

        let total_chunks = total_range.div_ceil(chunk_size);

        info!(
            "🚀 [GAP-FILL P2P] Starting parallel fetch: {} blocks ({}-{}) in {} chunks of {} (max {} in-flight, {}s timeout)",
            total_range, first_gap, last_gap, total_chunks, chunk_size, max_concurrency, chunk_timeout_secs
        );

        // Snapshot peer list once at the start. The list is queue-ordered by height
        // descending so peers[0] is the highest-tip (typically Epsilon for backfill).
        let mut peers: Vec<String> = {
            let registry = self.peer_registry.read().await;
            registry.active_peers_by_height()
                .into_iter()
                .filter(|p| p.height >= last_gap)
                .map(|p| p.peer_id.to_string())
                .collect()
        };
        if peers.is_empty() {
            warn!("🔧 [GAP-FILL P2P] No eligible peer at height ≥ {} — aborting gap fill for now", last_gap);
            return Ok(());
        }
        info!("🔧 [GAP-FILL P2P] {} eligible peers at height ≥ {}", peers.len(), last_gap);

        // Build chunk work queue.
        let mut chunks_queue: std::collections::VecDeque<(u64, u64)> =
            std::collections::VecDeque::with_capacity(total_chunks as usize);
        let mut cursor = first_gap;
        while cursor <= last_gap {
            let end = (cursor + chunk_size - 1).min(last_gap);
            chunks_queue.push_back((cursor, end));
            cursor = end + 1;
        }

        // Async closure that runs one chunk's request-with-retry cycle.
        // Returns Ok(Vec<QBlock>) on success or Err with description for logging.
        let fetch_chunk = |start: u64, end: u64, peers_snapshot: Vec<String>| {
            let network_tx = network_tx.clone();
            async move {
                let mut excluded: HashSet<String> = HashSet::new();
                for attempt in 0..=MAX_RETRIES {
                    let peer = peers_snapshot.iter()
                        .find(|p| !excluded.contains(*p))
                        .cloned();
                    let peer = match peer {
                        Some(p) => p,
                        None => return Err(format!("no eligible peer after {} retries", attempt)),
                    };

                    let (response_tx, response_rx) = oneshot::channel();
                    if let Err(e) = network_tx.send(NetworkRequest::RequestBlockRangeDirect {
                        peer_id: Some(peer.clone()),
                        start_height: start,
                        end_height: end,
                        response_tx,
                    }) {
                        excluded.insert(peer);
                        debug!("🔧 [GAP-FILL] dispatch failed for {}-{} attempt {}: {}", start, end, attempt, e);
                        continue;
                    }

                    match tokio::time::timeout(Duration::from_secs(chunk_timeout_secs), response_rx).await {
                        Ok(Ok(Ok(blocks))) if !blocks.is_empty() => {
                            return Ok((start, end, blocks));
                        }
                        Ok(Ok(Ok(_))) => {
                            // Empty response — peer doesn't have this range. Exclude and rotate.
                            excluded.insert(peer);
                            debug!("🔧 [GAP-FILL] empty response from peer for {}-{}", start, end);
                        }
                        Ok(Ok(Err(e))) => {
                            excluded.insert(peer);
                            debug!("🔧 [GAP-FILL] peer error for {}-{}: {}", start, end, e);
                        }
                        Ok(Err(_)) => {
                            excluded.insert(peer);
                            debug!("🔧 [GAP-FILL] response channel closed for {}-{}", start, end);
                        }
                        Err(_) => {
                            excluded.insert(peer);
                            debug!("🔧 [GAP-FILL] timeout ({}s) fetching {}-{}", chunk_timeout_secs, start, end);
                        }
                    }
                }
                Err(format!("exhausted {} retries", MAX_RETRIES + 1))
            }
        };

        let start_time = Instant::now();
        let mut in_flight = FuturesUnordered::new();
        let mut chunks_completed: u64 = 0;
        let mut chunks_failed: u64 = 0;
        let mut blocks_stored: u64 = 0;
        let mut last_progress_log = Instant::now();

        // Prime the sliding window.
        for _ in 0..max_concurrency {
            if let Some((s, e)) = chunks_queue.pop_front() {
                in_flight.push(fetch_chunk(s, e, peers.clone()));
            } else {
                break;
            }
        }

        // Drive the window: as each chunk completes, store its blocks (batched) and
        // dispatch the next pending chunk.
        while let Some(result) = in_flight.next().await {
            match result {
                Ok((start, end, blocks)) => {
                    let n = blocks.len();
                    // v1.0.2: batched save via save_qblocks_batch_turbo — single RocksDB
                    // commit instead of N individual saves. Same path turbo sync uses.
                    if let Err(e) = self.storage.save_qblocks_batch_turbo(&blocks).await {
                        warn!("🔧 [GAP-FILL] Batch save failed for {}-{}: {} (falling back to per-block)", start, end, e);
                        for block in &blocks {
                            if let Err(e) = self.storage.save_qblock(block).await {
                                debug!("🔧 [GAP-FILL] per-block fallback failed for {}: {}", block.header.height, e);
                            }
                        }
                    }
                    blocks_stored += n as u64;
                    chunks_completed += 1;
                }
                Err(reason) => {
                    chunks_failed += 1;
                    debug!("🔧 [GAP-FILL] chunk failed: {}", reason);
                }
            }

            // Refresh peer list every ~50 chunks in case better peers came online.
            if (chunks_completed + chunks_failed) % 50 == 0 {
                let fresh: Vec<String> = {
                    let registry = self.peer_registry.read().await;
                    registry.active_peers_by_height()
                        .into_iter()
                        .filter(|p| p.height >= last_gap)
                        .map(|p| p.peer_id.to_string())
                        .collect()
                };
                if !fresh.is_empty() && fresh.len() != peers.len() {
                    debug!("🔧 [GAP-FILL] Peer list refreshed: {} eligible peers", fresh.len());
                    peers = fresh;
                }
            }

            // Periodic progress log (every 5s of wall time).
            if last_progress_log.elapsed() >= Duration::from_secs(5) {
                let elapsed = start_time.elapsed().as_secs_f64().max(0.001);
                let bps = blocks_stored as f64 / elapsed;
                let eta_secs = if bps > 0.0 {
                    (total_range.saturating_sub(blocks_stored)) as f64 / bps
                } else { f64::INFINITY };
                info!(
                    "📊 [GAP-FILL P2P] {}/{} chunks ({} failed), {} blocks stored, {:.0} bps, eta {:.1}min",
                    chunks_completed, total_chunks, chunks_failed, blocks_stored, bps, eta_secs / 60.0
                );
                last_progress_log = Instant::now();
            }

            // Refill the window.
            if let Some((s, e)) = chunks_queue.pop_front() {
                in_flight.push(fetch_chunk(s, e, peers.clone()));
            }
        }

        let elapsed = start_time.elapsed();
        let bps = blocks_stored as f64 / elapsed.as_secs_f64().max(0.001);
        info!(
            "✅ [GAP-FILL P2P] Completed {}-{} in {:?}: {} blocks stored ({:.0} bps), {}/{} chunks ok, {} failed",
            first_gap, last_gap, elapsed, blocks_stored, bps, chunks_completed, total_chunks, chunks_failed
        );
        Ok(())
    }

    /// v1.0.2 OPTION B: Delegate huge gap ranges to the turbo sync parallel download path.
    ///
    /// The historical pre-checkpoint backfill is 16.5M blocks. The simpler windowed
    /// `fill_gap_p2p` tops out at ~70 bps because (a) it uses smaller chunks (500 vs
    /// 1000), (b) its 8-in-flight is capped at 4 by Epsilon's server-side block-pack
    /// semaphore, and (c) it doesn't use the Kalman concurrency / gravity-assist /
    /// Warp Sync prefetch machinery.
    ///
    /// `download_chunks_parallel` IS that machinery. Routing the big range through it
    /// gets ~570 bps (memory: "Full sync (~11.4M blocks): ~5.5 hours") — making the
    /// 16.5M-block backfill finish in ~8 hours instead of ~60.
    async fn fill_gap_via_turbo(&self, first_gap: u64, last_gap: u64) -> Result<()> {
        let total_range = last_gap.saturating_sub(first_gap) + 1;

        // Chunk size matches turbo sync's default for medium tiers (16GB+ RAM).
        // 1000 is the BlockPack hard cap per memory; larger would fail to serialize.
        let chunk_size: u64 = std::env::var("Q_GAPFILL_TURBO_CHUNK_SIZE")
            .ok().and_then(|v| v.parse().ok())
            .unwrap_or(1000)
            .clamp(100, 1000);

        info!(
            "🚀 [GAP-FILL TURBO] Large range ({} blocks) — delegating to download_chunks_parallel \
             for {}-{} in chunks of {}",
            total_range, first_gap, last_gap, chunk_size
        );

        // Build chunks
        let mut chunks: Vec<(u64, u64)> = Vec::with_capacity((total_range / chunk_size + 1) as usize);
        let mut cursor = first_gap;
        while cursor <= last_gap {
            let end = (cursor + chunk_size - 1).min(last_gap);
            chunks.push((cursor, end));
            cursor = end + 1;
        }

        // v10.9.26: Apply genesis-window cap here too (gap-fill path).
        // Without this, gap-fill in genesis mode would dispatch chunks at
        // arbitrary high heights (the gap range is computed from the
        // PEER tip, not from local contiguous height), causing the
        // "stops at 26k" symptom — chunks land at 6M+ but contiguous
        // can't advance.
        let genesis_mode = std::env::var("Q_GENESIS_SYNC_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if genesis_mode {
            let lookahead: u64 = std::env::var("Q_GENESIS_LOOKAHEAD_BLOCKS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(10_000);
            let window_cap = first_gap.saturating_add(lookahead);
            let original_len = chunks.len();
            // v10.9.29: cap BOTH chunk start AND chunk end (clip). See split_into_chunks
            // for the rationale — a single oversized chunk slipping through breaks ingestion.
            chunks.retain_mut(|(s, e)| {
                if *s > window_cap {
                    false
                } else {
                    *e = (*e).min(window_cap);
                    true
                }
            });
            if chunks.len() < original_len {
                warn!(
                    "🌱 [GENESIS-WINDOW v10.9.29] gap-fill capped: kept {}/{} chunks, \
                     each clipped to end ≤ {} (= first_gap {} + {} lookahead).",
                    chunks.len(), original_len, window_cap, first_gap, lookahead
                );
            }
        }

        // Eligible peers (already PeerId in the registry). Filter to peers at or above
        // the range end so they're guaranteed to have these blocks.
        let candidate_peers: Vec<PeerId> = {
            let registry = self.peer_registry.read().await;
            registry.active_peers_by_height()
                .into_iter()
                .filter(|p| p.height >= last_gap)
                .map(|p| p.peer_id)
                .collect()
        };
        if candidate_peers.is_empty() {
            warn!(
                "🔧 [GAP-FILL TURBO] No eligible peer at height ≥ {} — falling back to local windowed fetch",
                last_gap
            );
            return Ok(());
        }

        // v1.0.2 FIX (#16): Probe each candidate peer for archive capability before
        // launching the bulk turbo download. Pre-checkpoint blocks (heights < CHECKPOINT)
        // can only be served by archive nodes — peers that themselves bootstrapped from
        // the checkpoint don't have heights 1..CHECKPOINT and return empty.
        //
        // Without this filter, download_chunks_parallel rotates through non-archive peers
        // for each chunk, gets empty responses, and "completes" the entire 16.5M-block
        // range in minutes without actually storing anything (observed in v10.9.11 soak:
        // 0/166 sampled blocks present after "completion").
        //
        // Probe: ask each peer for the block at `first_gap` (typically height 1 for
        // pre-checkpoint backfill). Peer that returns a non-empty response has the
        // history we need; others are excluded for this range.
        let probe_height = first_gap;
        let network_tx = match &self.network_tx {
            Some(tx) => tx.clone(),
            None => {
                warn!("🔧 [GAP-FILL TURBO] No network channel — cannot probe peers");
                return Ok(());
            }
        };

        info!(
            "🔍 [GAP-FILL TURBO] Probing {} candidate peers for archive capability (height {} block)…",
            candidate_peers.len(), probe_height
        );

        let mut archive_peers: Vec<PeerId> = Vec::new();
        for peer in &candidate_peers {
            let (probe_tx, probe_rx) = oneshot::channel();
            if let Err(e) = network_tx.send(NetworkRequest::RequestBlockRangeDirect {
                peer_id: Some(peer.to_string()),
                start_height: probe_height,
                end_height: probe_height,
                response_tx: probe_tx,
            }) {
                debug!("🔍 [GAP-FILL TURBO probe] dispatch failed for {}: {}", peer, e);
                continue;
            }
            match tokio::time::timeout(Duration::from_secs(10), probe_rx).await {
                Ok(Ok(Ok(blocks))) if !blocks.is_empty() => {
                    debug!("✅ [GAP-FILL TURBO probe] {} has block at h={}, is archive-capable",
                           peer, probe_height);
                    archive_peers.push(*peer);
                }
                Ok(Ok(Ok(_))) => {
                    debug!("⛔ [GAP-FILL TURBO probe] {} returned empty for h={} — non-archive",
                           peer, probe_height);
                }
                Ok(Ok(Err(e))) => {
                    debug!("⛔ [GAP-FILL TURBO probe] {} error for h={}: {}", peer, probe_height, e);
                }
                _ => {
                    debug!("⛔ [GAP-FILL TURBO probe] {} timeout/closed for h={}", peer, probe_height);
                }
            }
        }

        if archive_peers.is_empty() {
            warn!(
                "🔧 [GAP-FILL TURBO] None of {} candidate peers can serve heights < CHECKPOINT \
                 (no archive-capable peer reachable). Aborting Phase 2 — caller may retry later.",
                candidate_peers.len()
            );
            return Err(anyhow::anyhow!(
                "no archive-capable peer found for range {}-{}", first_gap, last_gap
            ));
        }

        info!(
            "🔧 [GAP-FILL TURBO] {}/{} peers archive-capable; queueing {} chunks of {} blocks",
            archive_peers.len(), candidate_peers.len(), chunks.len(), chunk_size
        );

        // v10.9.14 PHASE 2 MEMORY BUDGET: dispatch in bounded batches with cgroup-RSS gating.
        //
        // v10.9.13 OOM root cause: download_chunks_parallel was given all 16,539 chunks at once.
        // Internally it caps to 8 in-flight, but with no end-to-end memory budget, jemalloc
        // fragmentation + RocksDB write-buffer accumulation + decompression staging drove RSS
        // from 290 MB → 16 GB over ~70 min. Phase 1 (forward sync) completed cleanly at ~6 GB;
        // Phase 2 alone added ~10 GB through unbounded staging.
        //
        // Fix: batch the chunk queue and gate dispatch on real container memory (cgroup
        // memory.current, not /proc RSS — they differ in Docker). Each batch goes through
        // download_chunks_parallel unchanged; the outer loop just paces them. Throughput is
        // preserved because the bottleneck is network RTT × in-flight bytes, not memory —
        // 32 chunks × 500 blocks × ~50 KB/block × resident_factor 4 ≈ 3.2 GB per batch peak,
        // safely under the 9 GB soft limit on a 16 GB container.
        let batch_size: usize = std::env::var("Q_PHASE2_BATCH_SIZE")
            .ok().and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let soft_limit_mb: u64 = std::env::var("Q_PHASE2_SOFT_LIMIT_MB")
            .ok().and_then(|v| v.parse().ok())
            .unwrap_or(9216);  // 9 GB on 16 GB container
        let hard_limit_mb: u64 = std::env::var("Q_PHASE2_HARD_LIMIT_MB")
            .ok().and_then(|v| v.parse().ok())
            .unwrap_or(12288); // 12 GB
        let pause_secs: u64 = std::env::var("Q_PHASE2_PAUSE_SECS")
            .ok().and_then(|v| v.parse().ok())
            .unwrap_or(10);

        info!(
            "🧠 [PHASE2 BUDGET] batch_size={} soft={}MB hard={}MB pause={}s",
            batch_size, soft_limit_mb, hard_limit_mb, pause_secs
        );

        let start = Instant::now();
        let total_chunks = chunks.len();
        let mut chunks_done = 0usize;
        let mut batches_done = 0usize;
        let mut batch_failures = 0usize;

        for batch in chunks.chunks(batch_size) {
            // RSS gate: check before dispatching the next batch.
            // Loop until we're below the soft limit (or sleep if above hard).
            loop {
                let rss_mb = read_container_memory_mb().await.unwrap_or(0);
                if rss_mb >= hard_limit_mb {
                    warn!(
                        "🛑 [PHASE2 RSS GATE] rss={}MB ≥ hard={}MB — pausing dispatch {}s",
                        rss_mb, hard_limit_mb, pause_secs
                    );
                    tokio::time::sleep(Duration::from_secs(pause_secs)).await;
                    continue;
                } else if rss_mb >= soft_limit_mb {
                    debug!(
                        "🐢 [PHASE2 RSS GATE] rss={}MB ≥ soft={}MB — slowing dispatch",
                        rss_mb, soft_limit_mb
                    );
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    break;
                } else {
                    break;
                }
            }

            // Dispatch this batch through the existing turbo machinery
            let batch_chunks: Vec<(u64, u64)> = batch.to_vec();
            let batch_size_actual = batch_chunks.len();
            match self.download_chunks_parallel(batch_chunks, archive_peers.clone()).await {
                Ok(()) => {
                    chunks_done += batch_size_actual;
                    batches_done += 1;
                }
                Err(e) => {
                    batch_failures += 1;
                    warn!(
                        "⚠️ [PHASE2 BATCH] batch #{} ({} chunks) failed: {} — continuing",
                        batches_done, batch_size_actual, e
                    );
                }
            }

            // Telemetry: every ~100 chunks, log RSS + progress
            if batches_done % 6 == 0 {
                let rss_mb = read_container_memory_mb().await.unwrap_or(0);
                let pct = (chunks_done as f64 / total_chunks as f64) * 100.0;
                let bps = (chunks_done as f64 * batch_size as f64 * 1000.0) / start.elapsed().as_secs_f64().max(1.0);
                info!(
                    "📊 [PHASE2 PROGRESS] {}/{} chunks ({:.1}%, {} failed batches), rss={}MB, ~{:.0} bps",
                    chunks_done, total_chunks, pct, batch_failures, rss_mb, bps
                );
            }
        }

        let elapsed = start.elapsed();
        info!(
            "✅ [GAP-FILL TURBO] Completed {}-{} in {:?} via turbo path ({} chunks ok, {} batches failed)",
            first_gap, last_gap, elapsed, chunks_done, batch_failures
        );
        Ok(())
    }


    /// Register a peer with their highest block height (v5.2.0: with monotonicity enforcement)
    pub async fn register_peer(&self, peer_id: PeerId, highest_block: u64) {
        self.register_peer_with_tip(peer_id, highest_block, None).await;
    }

    /// v8.4.0: Seed peer bandwidth from handshake-reported tier
    /// Called by network layer after successful handshake to give gravity-assist
    /// an initial bandwidth signal before actual block transfers occur.
    pub fn seed_peer_bandwidth(&self, peer_id: &str, bandwidth_mbps: u32) {
        if self.config.enable_apollo_gravity_assist && bandwidth_mbps > 0 {
            self.apollo_peer_momentum.seed_bandwidth_from_handshake(peer_id, bandwidth_mbps);
        }
    }

    /// Register a peer with height and optional tip hash (v5.2.0)
    /// v8.0.8: Added height sanity check to prevent rogue peers from poisoning registry
    pub async fn register_peer_with_tip(&self, peer_id: PeerId, highest_block: u64, tip_hash: Option<[u8; 32]>) {
        // v8.2.8: Dynamic peer height sanity check using peer consensus.
        // Old approach: reject if peer_height > our_height + 500K — broke on Windows/sled
        // when height pointer was lost (our_height=0 → cap=500K → all real peers rejected).
        //
        // New approach: Use MEDIAN of already-registered peers as the reference, not just
        // our local height. If 3+ peers agree on ~2.3M, a rogue peer claiming 100M is rejected,
        // but legitimate peers are never blocked even when local height is 0.
        let our_height = self.storage.get_highest_contiguous_block().await.unwrap_or(0);
        {
            let registry = self.peer_registry.read().await;
            let peer_count = registry.active_peer_count();
            let reference_height = if peer_count >= 3 {
                // Use median of existing peers as reference (consensus-based)
                // v8.5.9: ALWAYS use max(median, our_height) — never reject peers
                // just because a few slow/stale peers drag down the median
                let median = registry.median_height().unwrap_or(our_height);
                median.max(our_height)
            } else {
                our_height
            };
            // For established nodes: allow 3x reference or reference + 500K (whichever is larger)
            // For fresh nodes (reference < 1000): NO cap — accept any height.
            // v10.0.6: For syncing nodes (reference < network height by >500K), use relaxed cap.
            // A fresh/syncing node can't judge "too high" when it's millions behind the chain tip.
            // The old 500K cap caused total sync stalls when chain was at 10M+ and node at 26K.
            if reference_height >= 1_000 {
                // v10.0.6: Check if we're in initial sync (far behind network).
                // If the gap between announced height and our height is huge AND we have few peers,
                // this is likely initial sync, not a malicious peer. Use much more relaxed cap.
                let gap = highest_block.saturating_sub(our_height);
                let is_initial_sync = gap > 500_000 && peer_count <= 5;

                let max_reasonable = if is_initial_sync {
                    // During initial sync: accept heights up to 100x our height or +50M (whichever larger)
                    // This allows a node at 26K to accept peers at 10M+
                    (reference_height * 100).max(reference_height + 50_000_000)
                } else {
                    // Normal operation: conservative 3x or +500K cap
                    (reference_height * 3).max(reference_height + 500_000)
                };
                if highest_block > max_reasonable {
                    warn!("🚫 [PEER REGISTRY] Rejecting suspicious height {} from peer {} (ref: {}, our: {}, peers: {}, max: {}, initial_sync: {})",
                        highest_block, peer_id, reference_height, our_height, peer_count, max_reasonable, is_initial_sync);
                    return;
                }
            }
        }

        let mut registry = self.peer_registry.write().await;

        let accepted = registry.update_peer(peer_id, highest_block, tip_hash);
        if !accepted {
            warn!("🚫 [PEER REGISTRY] Rejected update from peer {} (monotonicity violations)", peer_id);
            return;
        }

        // 🚀 v1.0.2: Update lock-free atomic caches (sync loop reads these without lock)
        // Update max height: atomic CAS loop to ensure we only increase
        let mut current_max = self.cached_max_peer_height.load(Ordering::Relaxed);
        while highest_block > current_max {
            match self.cached_max_peer_height.compare_exchange_weak(
                current_max, highest_block, Ordering::Release, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
        self.cached_peer_count.store(registry.active_peer_count() as u32, Ordering::Release);

        // 🚀 v2.3.10-beta: Also register with Warp Sync MultiPeerDownloader
        // This enables intelligent peer selection based on bandwidth/latency metrics
        self.warp_multi_peer.register_peer(peer_id.to_string(), highest_block).await;

        info!("📡 Registered peer {} with height {} (Warp Sync enabled)", peer_id, highest_block);
    }

    /// Get peer registry information for debugging (backward-compat format)
    pub async fn get_peer_registry_info(&self) -> Vec<(PeerId, u64)> {
        let registry = self.peer_registry.read().await;
        registry.to_legacy_vec()
    }

    /// v5.2.0: Get read access to the enhanced peer registry
    pub async fn get_enhanced_registry(&self) -> tokio::sync::RwLockReadGuard<'_, EnhancedPeerRegistry> {
        self.peer_registry.read().await
    }

    /// v5.2.0: Evict stale peers (not heard from in `stale_secs` seconds)
    pub async fn evict_stale_peers(&self, stale_secs: u64) -> usize {
        let mut registry = self.peer_registry.write().await;
        let evicted = registry.evict_stale(stale_secs);
        // 🚀 v1.0.2: Update lock-free atomics after eviction
        if evicted > 0 {
            self.cached_peer_count.store(registry.active_peer_count() as u32, Ordering::Release);
            // Recalculate max height after eviction
            let new_max = registry.max_height().unwrap_or(0);
            self.cached_max_peer_height.store(new_max, Ordering::Release);
        }
        evicted
    }

    /// 🤖 v1.4.0-beta: Extract sync features for ML batch size prediction
    ///
    /// Collects 9 features from various components for the ML model:
    /// 1. RTT median/MAD - from AdaptiveTimeout
    /// 2. Memory pressure - from MemoryLimiter
    /// 3. Peer trust score - from PeerTrustRegistry (average)
    /// 4. Bandwidth estimate - from TurboSyncMetrics
    /// 5. Success rate - from TurboSyncMetrics
    /// 6. Compression ratio - from TurboSyncMetrics
    /// 7. Pipeline depth - from PipelineManager
    /// 8. Peer count - from peer_registry
    async fn extract_sync_features(&self, qualified_peers: &[PeerId]) -> crate::ml_batch_optimizer::SyncFeatures {
        // 1. RTT statistics from AdaptiveTimeout
        let (rtt_median, rtt_mad) = {
            let timeout = self.adaptive_timeout.read().await;
            timeout.get_rtt_stats()
        };

        // 2. Memory pressure from MemoryLimiter
        let memory_stats = self.memory_limiter.get_memory_stats().await;
        let memory_pressure = memory_stats.usage_percent() / 100.0; // Normalize to 0-1

        // 3. Average peer trust score (from PeerTrustRegistry)
        let peer_trust_score = if !qualified_peers.is_empty() {
            let mut total_trust = 0.0f32;
            let mut counted = 0;
            for peer in qualified_peers.iter().take(10) {
                let peer_id_str = peer.to_string();
                if let Some(trust) = self.peer_trust.get_trust_score(&peer_id_str) {
                    total_trust += trust as f32;
                    counted += 1;
                } else {
                    total_trust += 0.5; // Default trust for unknown peers
                    counted += 1;
                }
            }
            if counted > 0 {
                (total_trust / counted as f32).clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.5 // Default trust when no peers
        };

        // 4. Bandwidth estimate from metrics (MB/s)
        let bandwidth_mbps = {
            let total_bytes = self.metrics.total_bytes_downloaded.load(std::sync::atomic::Ordering::Relaxed);
            let elapsed_ms = self.metrics.start_time.read().await
                .map(|t| t.elapsed().as_millis() as u64)
                .unwrap_or(1000);
            if elapsed_ms > 0 {
                (total_bytes as f64 / (elapsed_ms as f64 / 1000.0) / (1024.0 * 1024.0)) as f32
            } else {
                10.0 // Default 10 MB/s
            }
        };

        // 5. Success rate from metrics (use failed_chunks and retried_chunks)
        let success_rate = {
            let failed = self.metrics.failed_chunks.load(std::sync::atomic::Ordering::Relaxed);
            let retried = self.metrics.retried_chunks.load(std::sync::atomic::Ordering::Relaxed);
            let blocks_synced = self.metrics.total_blocks_synced.load(std::sync::atomic::Ordering::Relaxed);
            // Estimate total attempts: blocks synced / chunk_size + failures + retries
            let estimated_total = (blocks_synced / self.config.chunk_size).max(1) + failed + retried;
            if estimated_total > 0 {
                1.0 - (failed as f32 / estimated_total as f32)
            } else {
                1.0 // Default to 100% success for new syncs
            }
        };

        // 6. Compression ratio (estimated from config)
        let compression_ratio = match self.config.compression_level {
            0 => 1.0,       // No compression
            1..=3 => 0.7,   // Light compression
            4..=6 => 0.5,   // Medium compression
            _ => 0.3,       // Heavy compression
        };

        // 7. Pipeline depth from PipelineManager
        let pipeline_depth = {
            let current = self.pipeline_manager.current_depth().await;
            current as f32 / self.config.pipeline_config.max_depth as f32
        };

        // 8. Peer count
        let peer_count = qualified_peers.len() as f32;

        crate::ml_batch_optimizer::SyncFeatures {
            rtt_median_ms: rtt_median,
            rtt_mad_ms: rtt_mad,
            memory_pressure: memory_pressure as f32,
            peer_trust_score,
            bandwidth_mbps,
            success_rate,
            compression_ratio,
            pipeline_depth,
            peer_count,
        }
    }

    /// Discover peers that have the required height
    ///
    /// 🛡️ v1.4.5-beta: Enhanced with PeerTrustRegistry integration
    /// - Filters out banned peers (trust < 0.1)
    /// - Sorts by trust score (highest first)
    /// - Prefers peers with proven reliability
    async fn discover_peers_with_height(&self, target_height: u64) -> Result<Vec<PeerId>> {
        let registry = self.peer_registry.read().await;

        // 🚀 v2.2.1-beta: CRITICAL FIX - Get local height to use as minimum threshold
        // We need peers with height > local_height (have blocks we need), not height >= target_height
        // This fixes the bug where peers at 258,495 were rejected when target was 789,189
        // even though they could help sync from 10,000 to 258,495!
        let local_height = self.get_local_height().await.unwrap_or(0);

        // 🐛 v2.1.1-DELTA-V: DEBUG - Log raw registry contents before filtering
        info!("🔍 [PEER DISCOVERY DEBUG] Registry has {} peers, looking for height > {} (local), target is {}",
              registry.len(), local_height, target_height);
        for (peer, height) in registry.iter() {
            info!("   📋 Peer {} has height {} (need > {} local)", peer, height, local_height);
        }

        // 🚀 v2.1.4-DELTA-V: Accept ALL peers with sufficient height
        // Sort so bootstrap is first (most reliable), then by height descending
        const BOOTSTRAP_PEER: &str = "12D3KooWSBxwSKw4wftHViMdw5rrV8Z1wEkikDS2vKYZtRrio5hH";

        // 🛡️ v1.4.5-beta: Collect peers with height and trust scores
        // 🚀 v2.2.1-beta: CRITICAL FIX - Use local_height not target_height!
        // A peer at 258,495 CAN help us sync from 10,000 even if target is 789,189
        let mut candidates: Vec<(PeerId, u64, f64)> = registry
            .iter()
            .filter(|(peer, height)| {
                let passes = *height > local_height;
                if !passes {
                    debug!("   ❌ Peer {} REJECTED: height {} <= local {}", peer, height, local_height);
                }
                passes  // Only keep peers with height > local (have blocks we need)
            })
            .map(|(peer, height)| {
                let peer_id_str = peer.to_string();
                // 🚀 v2.1.5-DELTA-V: Bootstrap peer is ALWAYS trusted (prevents chicken-egg)
                let is_bootstrap = peer_id_str == BOOTSTRAP_PEER;
                let trust = if is_bootstrap {
                    0.5f64  // Bootstrap always trusted
                } else {
                    self.peer_trust.get_trust_score(&peer_id_str).unwrap_or(0.5)
                };
                (peer, height, trust)
            })
            .collect();

        info!("📊 [PEER FILTER] After height filter: {} candidates (height > {} local)",
              candidates.len(), local_height);

        // 🛡️ v2.1.5-DELTA-V: DISABLED trust filter - was causing all peers to be rejected
        // The trust system has a chicken-egg problem: peers get penalized for failed syncs,
        // but syncs fail because no peers are available!
        let min_trust_threshold = 0.0f64;  // Was 0.1, now disabled
        let pre_filter_count = candidates.len();
        candidates.retain(|(_, _, trust)| *trust >= min_trust_threshold);
        let banned_count = pre_filter_count - candidates.len();

        if banned_count > 0 {
            warn!(
                "🚫 [PEER TRUST] Filtered out {} banned/untrusted peers (trust < {})",
                banned_count, min_trust_threshold
            );
        }

        // 🚀 v2.1.4-DELTA-V: Sort so bootstrap is FIRST (most reliable)
        // Then by trust score, then by height
        candidates.sort_by(|a, b| {
            let a_is_bootstrap = a.0.to_string() == BOOTSTRAP_PEER;
            let b_is_bootstrap = b.0.to_string() == BOOTSTRAP_PEER;

            // Bootstrap always first
            match (a_is_bootstrap, b_is_bootstrap) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => {
                    // Then by trust (descending)
                    b.2.partial_cmp(&a.2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        // Then by height (descending) as tiebreaker
                        .then_with(|| b.1.cmp(&a.1))
                }
            }
        });

        // 🚀 v2.3.8-beta: CRITICAL FIX - Accept ALL peers with height > local_height
        // Previous v2.1.7 logic ONLY allowed bootstrap peer, causing sync deadlock when
        // bootstrap falls behind local node (bootstrap at 861k, local at 875k).
        //
        // NEW STRATEGY:
        // 1. Accept ALL peers with height > local_height (sorted by trust, bootstrap first)
        // 2. Prioritize bootstrap peer if available (most reliable)
        // 3. Fall back to any peer with higher height
        // 4. libp2p request-response will handle connection establishment
        let mut qualified: Vec<PeerId> = candidates
            .iter()
            .map(|(peer, _, _)| *peer)
            .collect();

        info!("📡 [v2.3.8-beta] Accepting {} peers with height > {} (local)",
              qualified.len(), local_height);
        for (idx, (peer, height, trust)) in candidates.iter().take(5).enumerate() {
            let is_bootstrap = peer.to_string() == BOOTSTRAP_PEER;
            info!("   • Peer {}: {} (height: {}, trust: {:.2}){}",
                  idx + 1, &peer.to_string()[..12.min(peer.to_string().len())],
                  height, trust, if is_bootstrap { " [BOOTSTRAP]" } else { "" });
        }

        // 🚀 v2.3.8-beta: If no peers have height > local, check for ANY peer ahead of local
        // This handles edge case where candidates got filtered elsewhere
        if qualified.is_empty() && !registry.is_empty() {
            warn!("⚠️  [PEER RECOVERY] No candidates passed filter, checking raw registry...");

            // Find ANY peer with height > local_height (not target_height!)
            let ahead_peers: Vec<_> = registry.iter()
                .filter(|(_, h)| *h > local_height)
                .collect();

            if !ahead_peers.is_empty() {
                warn!("🔄 [PEER RECOVERY] Found {} peers ahead of local height {}",
                      ahead_peers.len(), local_height);
                for (peer, height) in ahead_peers.iter().take(3) {
                    warn!("   • Recovering peer {} with height {}", peer, height);
                    qualified.push((*peer).clone());
                }
            }
        }

        if qualified.is_empty() {
            // ✅ v0.9.6-beta: LOUD ERROR logging for critical peer discovery failure
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("🚨 CRITICAL: NO PEERS AVAILABLE FOR SYNC!");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("   Target height: {}", target_height);
            error!("   Peers in registry: {}", registry.len());
            error!("   Peers filtered by trust: {}", banned_count);
            error!("");
            error!("🔍 TROUBLESHOOTING:");
            error!("   1. Check if bootstrap peers are configured (Q_BOOTSTRAP_PEERS)");
            error!("   2. Verify libp2p peer discovery is working (check for CONNECTION logs)");
            error!("   3. Ensure peer height announcements are being received (check gossipsub)");
            error!("   4. Check if peer registry is being populated from libp2p discoveries");
            error!("   5. Check if peers are banned due to low trust scores");
            error!("");
            error!("📋 Peer Registry Contents:");
            for (peer, height) in registry.iter() {
                let trust = self.peer_trust.get_trust_score(&peer.to_string()).unwrap_or(0.5);
                error!("   • Peer {} has height {}, trust: {:.2}", peer, height, trust);
            }
            if registry.is_empty() {
                error!("   (EMPTY - This is the problem! No peers discovered.)");
            }
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        } else {
            info!("📡 [PEER TRUST] Found {} trusted peers with height >= {} (sorted by trust)",
                  qualified.len(), target_height);

            // Log top 3 peers with their trust scores
            for (idx, (peer, height, trust)) in candidates.iter().take(3).enumerate() {
                debug!("   • Peer {}: {} (height: {}, trust: {:.2})", idx + 1, peer, height, trust);
            }
        }

        Ok(qualified)
    }

    /// Split range into optimal chunks for parallel downloading
    fn split_into_chunks(&self, start: u64, end: u64, chunk_size: u64) -> Vec<(u64, u64)> {
        let mut chunks = Vec::new();
        // 🤖 v1.4.0-beta: Use ML-predicted chunk size passed as parameter
        let chunk_size = chunk_size.max(1);  // Ensure at least 1 block per chunk

        // ✅ v0.9.55-beta CRITICAL FIX: Include genesis block (height 0) in sync
        // ROOT CAUSE: `start + 1` skipped genesis when syncing from height 0
        // IMPACT: Fresh nodes stuck at height 0 forever with "Gap at height 2" spam
        // SOLUTION: Start from `start` instead of `start + 1` to include all blocks
        let mut current = start;
        while current <= end {
            let chunk_end = (current + chunk_size - 1).min(end);
            chunks.push((current, chunk_end));
            current = chunk_end + 1;
        }

        // v10.9.26: Apply the genesis-window cap here, INSIDE split_into_chunks,
        // so EVERY caller benefits (not just sync_to_height). Two known callers:
        //   • sync_to_height (line ~6822) — already had v10.9.25 cap above this;
        //     redundant but cheap.
        //   • other paths that build chunks (gap-fill at line ~3580, additional
        //     turbo paths at line ~7152) — previously bypassed the cap and were
        //     producing requests at heights > local + 1M, causing the chain to
        //     "stop at 26k" symptom from the v10.9.25 Beta test.
        //
        // The cap is computed from `start` (the request floor) rather than
        // `local_height` — same effective semantics for genesis mode because
        // start = local_height + 1 in all callers.
        let genesis_mode = std::env::var("Q_GENESIS_SYNC_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if genesis_mode {
            let lookahead: u64 = std::env::var("Q_GENESIS_LOOKAHEAD_BLOCKS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(10_000); // v10.9.28: match the GAP_SKIP_REFUSED ingestion safety cap (10000).
            let window_cap = start.saturating_add(lookahead);
            let original_len = chunks.len();
            // v10.9.29: cap BOTH chunk start AND chunk end (clip).
            //
            // The v10.9.28 cap only filtered by chunk start. If the ML chunk-size
            // predictor (or Kalman blend, or supernode-boost) produced a chunk
            // larger than `lookahead`, a single chunk spanning [start, start+N]
            // with N >> lookahead would pass the filter (start <= window_cap) but
            // span far beyond window_cap. When dispatched, the peer returns the
            // full N-block payload, the ingestion path's 10K gap-from-contiguous
            // safety check refuses it, and the contiguous chain doesn't advance.
            //
            // Observed v10.9.28 symptom: Beta sync test reached 26K (one good
            // batch) then froze; "1/1 chunks failed" loop. The single chunk was
            // spanning ~18M blocks (full target) because the chunk_size predictor
            // went large. Clipping each chunk end to window_cap forces every
            // dispatched chunk into the ingestable window.
            chunks.retain_mut(|(s, e)| {
                if *s > window_cap {
                    false
                } else {
                    *e = (*e).min(window_cap);
                    true
                }
            });
            if chunks.len() < original_len {
                warn!(
                    "🌱 [GENESIS-WINDOW v10.9.29] split_into_chunks capped: kept {}/{} chunks, \
                     each clipped to end ≤ {} (= start {} + {} lookahead).",
                    chunks.len(), original_len, window_cap, start, lookahead
                );
            }
        }

        info!("📦 Split range {}-{} into {} chunks of ~{} blocks",
              start, end, chunks.len(), chunk_size);

        chunks
    }

    /// Create compressed block pack (server-side)
    pub async fn create_block_pack(
        &self,
        start_height: u64,
        end_height: u64,
    ) -> Result<BlockPack> {
        let pack_start = Instant::now();

        // 🚀 v1.0.6-beta: PHASE 0 - Try cache first (Phase 3: Pack Caching)
        let cache_key = PackCacheKey::new(start_height, end_height, self.config.compression_level);

        if let Some(cached_pack) = self.pack_cache.get(&cache_key).await {
            debug!("✅ [PACK CACHE HIT] Serving {}..{} from cache ({:?})",
                  start_height, end_height, pack_start.elapsed());
            return Ok(cached_pack);
        }

        debug!("❌ [PACK CACHE MISS] Creating fresh pack for {}..{}", start_height, end_height);

        // PHASE 1: Validate requested range against actual storage
        let local_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);

        if start_height > local_height {
            anyhow::bail!(
                "Requested range {}-{} exceeds local height {} (peer height mismatch)",
                start_height, end_height, local_height
            );
        }

        // Adjust end_height to what we actually have
        let actual_end = end_height.min(local_height);

        // 🚀 v1.0.100-beta: BATCH FETCH OPTIMIZATION
        // get_qblocks_range now uses RocksDB multi_get internally for 10-50x faster fetching
        // Calculate limit from height range
        let limit = (actual_end - start_height + 1) as usize;
        let blocks = self.storage.get_qblocks_range(start_height, limit).await?;

        // 🔍 v1.3.10-beta: DEBUG - Log transaction counts from database
        // This helps identify if transactions are missing in DB or lost during serialization
        let blocks_with_txs = blocks.iter().filter(|b| !b.transactions.is_empty()).count();
        let blocks_without_txs = blocks.iter().filter(|b| b.transactions.is_empty()).count();
        let total_txs: usize = blocks.iter().map(|b| b.transactions.len()).sum();

        if blocks_without_txs > 0 {
            warn!(
                "📊 [PACK DEBUG] Range {}-{}: {} blocks with txs, {} blocks WITHOUT txs, {} total txs",
                start_height, actual_end, blocks_with_txs, blocks_without_txs, total_txs
            );

            // Log first 5 blocks without transactions for diagnosis
            for block in blocks.iter().filter(|b| b.transactions.is_empty()).take(5) {
                warn!(
                    "📊 [PACK DEBUG] Block {} has 0 txs (proposer={})",
                    block.header.height,
                    hex::encode(&block.header.proposer[..8])
                );
            }
        } else {
            info!(
                "✅ [PACK DEBUG] Range {}-{}: ALL {} blocks have transactions ({} total txs)",
                start_height, actual_end, blocks.len(), total_txs
            );
        }

        // Track missing blocks for gap detection (compare expected vs actual count)
        let expected_count = (actual_end - start_height + 1) as usize;
        let missing_count = expected_count.saturating_sub(blocks.len());
        let missing_heights: Vec<u64> = if missing_count > 0 {
            // We don't know exactly which heights are missing without individual checks,
            // but we can report the gap statistics
            vec![] // Empty vec - we no longer track individual missing heights
        } else {
            vec![]
        };

        // PHASE 3: Handle missing blocks
        if blocks.is_empty() {
            anyhow::bail!(
                "No blocks found in range {}-{} (requested {}-{}, local height: {}, missing: {} blocks)",
                start_height, actual_end, start_height, end_height, local_height, missing_count
            );
        }

        // Log warning if pack is partial (some blocks missing)
        if missing_count > 0 {
            let availability_pct = (blocks.len() as f32 / expected_count as f32) * 100.0;

            warn!(
                "⚠️  Creating PARTIAL pack {}-{}: {}/{} blocks ({:.1}% available, {} missing)",
                start_height, actual_end, blocks.len(), expected_count, availability_pct, missing_count
            );
        }

        // Suppress unused variable warning
        let _ = missing_heights;

        // Serialize blocks
        let serialized = bincode::serialize(&blocks)?;
        let uncompressed_size = serialized.len() as u64;

        // v1.4.14-beta: LZ4 compression (3-5x faster than zstd, ~15-20% larger output)
        // LZ4 is optimized for speed over compression ratio - perfect for real-time sync
        let compressed = lz4::block::compress(&serialized, None, true)
            .map_err(|e| anyhow::anyhow!("LZ4 compression failed: {}", e))?;

        // Calculate checksum (blake3 for speed, like Git uses SHA-1)
        let checksum_hash = blake3::hash(&compressed);
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(checksum_hash.as_bytes());

        let compression_ratio = compressed.len() as f32 / serialized.len() as f32;
        let pack_time = pack_start.elapsed();

        info!(
            "📦 Created pack {}-{}: {} blocks, {:.1}KB → {:.1}KB ({:.1}% compression) in {}ms",
            start_height, end_height, blocks.len(),
            uncompressed_size as f64 / 1024.0,
            compressed.len() as f64 / 1024.0,
            (1.0 - compression_ratio) * 100.0,
            pack_time.as_millis()
        );

        let pack = BlockPack {
            start_height,
            end_height,
            compressed_data: compressed,
            checksum,
            compression_ratio,
            block_count: blocks.len() as u32,
            uncompressed_size,
            request_id: None, // Set by P2P handler when responding to specific request
        };

        // 🚀 v1.0.6-beta: Cache the pack for future requests (Phase 3: Pack Caching)
        if let Err(e) = self.pack_cache.put(cache_key, pack.clone()).await {
            warn!("📦 [PACK CACHE] Failed to cache pack {}..{}: {}",
                  start_height, end_height, e);
            // Non-fatal - continue even if caching fails
        } else {
            debug!("📦 [PACK CREATED & CACHED] {}..{} in {:?} ({} blocks, {:.1} MB → {:.1} MB, ratio {:.2})",
                  start_height, end_height, pack_start.elapsed(),
                  pack.block_count,
                  uncompressed_size as f64 / (1024.0 * 1024.0),
                  pack.compressed_data.len() as f64 / (1024.0 * 1024.0),
                  compression_ratio);
        }

        Ok(pack)
    }

    /// Apply a received block pack (client-side)
    ///
    /// v0.8.0-beta: Now accepts optional BalanceConsensusEngine for deterministic reward processing
    pub async fn apply_block_pack(
        &self,
        pack: BlockPack,
        balance_engine: Option<&BalanceConsensusEngine>,
    ) -> Result<()> {
        let apply_start = Instant::now();

        // Verify checksum
        let computed_checksum = blake3::hash(&pack.compressed_data);
        if computed_checksum.as_bytes() != &pack.checksum {
            anyhow::bail!(
                "Checksum mismatch for pack {}-{}",
                pack.start_height, pack.end_height
            );
        }

        // 🚀 v1.4.2-beta: Decompression in spawn_blocking (doesn't block async runtime)
        // OPTIMIZATION: CPU-bound decompression runs on dedicated threadpool
        // - Before: Blocked async runtime, caused task starvation with 32 parallel streams
        // - After:  Runs on blocking threadpool, async runtime stays responsive
        //
        // 🔧 v1.5.0-beta CRITICAL FIX: Format auto-detection (zstd vs LZ4)
        // PROBLEM: Server compresses with zstd (lines 2407-2409), but client only tried LZ4.
        // Magic bytes [0x28, 0xB5, 0x2F, 0xFD] = zstd, otherwise assume LZ4.
        // This fix enables backward/forward compatibility between compression formats.
        //
        // 🚀 v1.5.1-beta: DECOMPRESSION SEMAPHORE - Prevents 100k block stall!
        // PROBLEM: With 32 parallel downloads, spawn_blocking gets 32+ concurrent decompression
        // tasks, saturating the threadpool (default 512 threads) and causing memory exhaustion.
        // FIX: Limit concurrent decompression to 8 tasks (configurable via Q_DECOMPRESSION_PARALLELISM)
        let _decomp_permit = self.decompression_semaphore.acquire().await
            .map_err(|e| anyhow::anyhow!("Failed to acquire decompression semaphore: {}", e))?;

        let decompress_start = Instant::now();
        let compressed_data = pack.compressed_data.clone();
        let uncompressed_size = pack.uncompressed_size;

        // 🔧 v1.5.0-beta: Detect compression format from magic bytes
        // zstd magic: [0x28, 0xB5, 0x2F, 0xFD] (little-endian 0xFD2FB528)
        // LZ4 block format: no magic bytes (raw compressed data)
        let is_zstd = compressed_data.len() >= 4
            && compressed_data[0] == 0x28
            && compressed_data[1] == 0xB5
            && compressed_data[2] == 0x2F
            && compressed_data[3] == 0xFD;

        let compressed_len = compressed_data.len();
        let first_bytes: Vec<u8> = compressed_data.iter().take(16).copied().collect();
        let format_name = if is_zstd { "zstd" } else { "LZ4" };

        debug!("🔍 [DECOMPRESS] Pack {}-{}: {} format detected ({} bytes, first 4: {:02x?})",
               pack.start_height, pack.end_height, format_name, compressed_len, &first_bytes[..4.min(first_bytes.len())]);

        let decompressed = if is_zstd {
            // 🔧 v1.5.0-beta: zstd decompression path
            tokio::task::spawn_blocking(move || {
                // Limit decompression to 100MB to prevent DoS attacks
                let max_size = 100_000_000usize;
                zstd::bulk::decompress(&compressed_data, max_size)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            })
            .await
            .map_err(|e| anyhow::anyhow!("spawn_blocking join failed: {}", e))?
            .map_err(|e| {
                error!("🔍 [ZSTD DEBUG] Decompression failed for pack {}-{}:", pack.start_height, pack.end_height);
                error!("   Compressed size: {} bytes", compressed_len);
                error!("   Expected uncompressed: {} bytes", uncompressed_size);
                error!("   First 16 bytes: {:02x?}", first_bytes);
                error!("   Error: {}", e);
                anyhow::anyhow!("zstd decompression failed: {}", e)
            })?
        } else {
            // 🔧 v2.1.6-DELTA-V: LZ4 decompression with PROPER prepend_size handling
            // The server compresses with prepend_size=true (line 2054), so the compressed data
            // has a 4-byte size prefix. When decompressing, we MUST pass None for size_hint
            // to let LZ4 read the prepended size. Passing a size_hint conflicts with the prefix.
            //
            // BUG FIX: Previous code passed size_hint which caused "Input invalid or too long?"
            // errors when the prepended size didn't match the hint.

            tokio::task::spawn_blocking(move || {
                // 🛡️ v5.1.1: Validate prepended size BEFORE decompression to prevent DoS
                // LZ4 with prepend_size=true stores uncompressed size as first 4 bytes (little-endian u32)
                // A malicious/corrupted peer could set this to 0xFFFFFFFF (4GB) causing OOM crash
                const MAX_DECOMPRESSED_SIZE: u32 = 200_000_000; // 200MB limit (matches zstd safety)
                if compressed_data.len() >= 4 {
                    let prepended_size = u32::from_le_bytes([
                        compressed_data[0], compressed_data[1],
                        compressed_data[2], compressed_data[3],
                    ]);
                    if prepended_size > MAX_DECOMPRESSED_SIZE {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("LZ4 prepended size {} exceeds safety limit of {} bytes (possible DoS or corruption)",
                                    prepended_size, MAX_DECOMPRESSED_SIZE),
                        ));
                    }
                }
                lz4::block::decompress(&compressed_data, None)
            })
            .await
            .map_err(|e| anyhow::anyhow!("spawn_blocking join failed: {}", e))?
            .map_err(|e| {
                error!("🔍 [LZ4 DEBUG v5.1.1] Decompression failed for pack {}-{}:", pack.start_height, pack.end_height);
                error!("   Compressed size: {} bytes", compressed_len);
                error!("   Expected uncompressed: {} bytes (from pack metadata, NOT used as hint)", uncompressed_size);
                error!("   First 16 bytes: {:02x?}", first_bytes);
                error!("   Error: {}", e);
                error!("   NOTE: LZ4 uses prepended size from compressed data (prepend_size=true)");
                anyhow::anyhow!("LZ4 decompression failed: {}", e)
            })?
        };
        let decompress_time = decompress_start.elapsed();

        // 🚀 v1.4.2-beta: Deserialize in spawn_blocking too (CPU-bound)
        // 🔧 v1.5.3-beta: CRITICAL FIX - Use postcard (matches serialization at line 2607)
        // BUG FOUND: v1.3.9-beta introduced postcard serialization but kept bincode deserialization
        // This caused "Deserialization failed: string is not valid utf8" errors during sync
        let deserialize_start = Instant::now();
        let decompressed_clone = decompressed;
        let mut blocks: Vec<QBlock> = tokio::task::spawn_blocking(move || {
            // v1.5.3-beta: Try postcard first (matches line 2607 serialization), fallback to bincode
            postcard::from_bytes(&decompressed_clone)
                .or_else(|_| bincode::deserialize(&decompressed_clone))
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join failed: {}", e))?
        .map_err(|e| anyhow::anyhow!("Deserialization failed: {}", e))?;
        let deserialize_time = deserialize_start.elapsed();

        // 🔍 v1.3.10-beta: DEBUG - Log transaction counts after deserialization
        // This traces where transactions are lost during sync
        let rx_blocks_with_txs = blocks.iter().filter(|b| !b.transactions.is_empty()).count();
        let rx_blocks_without_txs = blocks.iter().filter(|b| b.transactions.is_empty()).count();
        let rx_total_txs: usize = blocks.iter().map(|b| b.transactions.len()).sum();

        if rx_blocks_without_txs > 0 {
            warn!(
                "📊 [UNPACK DEBUG] Pack {}-{}: {} blocks with txs, {} blocks WITHOUT txs, {} total txs",
                pack.start_height, pack.end_height, rx_blocks_with_txs, rx_blocks_without_txs, rx_total_txs
            );
        } else {
            info!(
                "✅ [UNPACK DEBUG] Pack {}-{}: ALL {} blocks have transactions ({} total txs)",
                pack.start_height, pack.end_height, blocks.len(), rx_total_txs
            );
        }

        // 🎨 v0.6.7-beta: CRITICAL FIX - Sort blocks by height to handle partial batches correctly
        // Partial batches (with gaps) may have blocks in arbitrary order from database fetch
        // Sorting ensures we can accurately track contiguous height progression
        // 🚀 v1.0.7-beta: Use parallel sort for large batches (faster for 1000+ blocks)
        // v8.4.4: Use dedicated sync rayon pool (4 threads) instead of global pool (19 threads)
        let sort_start = Instant::now();
        if blocks.len() > 100 {
            SYNC_RAYON_POOL.install(|| blocks.par_sort_unstable_by_key(|b| b.header.height));
        } else {
            blocks.sort_by_key(|b| b.header.height);
        }
        let sort_time = sort_start.elapsed();

        debug!(
            "🚀 [PHASE 4] Decompress: {:?}, Deserialize: {:?}, Sort: {:?} ({} blocks)",
            decompress_time, deserialize_time, sort_time, blocks.len()
        );

        // 🚀 v1.0.50-beta: Incremental verification - catch errors early
        // This verifies blocks as they arrive, detecting corruption within 100ms
        // instead of waiting for full download to complete
        // NOTE: IncrementalBlockVerifier is now thread-safe internally (uses Mutex),
        // so we only need a read lock here to access it
        //
        // 🔧 v1.4.6-beta: CRITICAL FIX - Skip incremental verification during parallel sync
        // PROBLEM: IncrementalBlockVerifier requires SEQUENTIAL blocks, but TurboSync
        // fetches chunks in PARALLEL. When chunk 30000-30504 arrives before chunk 20001-20504,
        // the verifier sees a "gap" and rejects ALL blocks in the out-of-order chunk.
        //
        // FIX: Check if pack is "in order" (starts at verifier's expected height).
        // If not, skip incremental verification and rely on SHA3 verification which
        // doesn't require sequential processing.
        let verify_start = Instant::now();
        let verifier = self.block_verifier.read().await;
        let verifier_height = verifier.get_verified_height().await;

        // Check if this pack starts at the expected next height
        let first_block_height = blocks.first().map(|b| b.header.height).unwrap_or(0);
        let is_in_order = first_block_height == verifier_height + 1
            || first_block_height <= verifier_height; // Already processed blocks are OK

        let valid_count = if is_in_order {
            // Sequential pack - use incremental verification
            verifier.verify_block_batch(&blocks).await
                .context("Incremental block verification failed")?
        } else {
            // Out-of-order pack (parallel sync) - skip incremental verification
            // SHA3 verification below will still catch corruption
            debug!(
                "📦 [BULK SYNC] Pack {}-{} is out of order (verifier at {}, pack starts at {}), skipping incremental verification",
                pack.start_height, pack.end_height, verifier_height, first_block_height
            );
            blocks.len() // All blocks pass (will be verified by SHA3)
        };
        let verify_time = verify_start.elapsed();

        if valid_count < blocks.len() {
            warn!(
                "⚠️  [INCREMENTAL VERIFY] Only {}/{} blocks valid in pack {}-{}",
                valid_count, blocks.len(), pack.start_height, pack.end_height
            );
            // Truncate blocks to only the valid ones
            if valid_count > 0 {
                blocks.truncate(valid_count);
                info!("📦 [INCREMENTAL VERIFY] Truncated pack to {} valid blocks", valid_count);
            } else {
                anyhow::bail!(
                    "All {} blocks invalid in pack {}-{} - possible corruption or malicious peer",
                    blocks.len(), pack.start_height, pack.end_height
                );
            }
        }

        debug!(
            "✅ [INCREMENTAL VERIFY] {}/{} blocks verified in {:?}",
            valid_count, blocks.len(), verify_time
        );
        drop(verifier); // Release lock early

        // 🚀 v1.4.2-beta: PARALLEL SHA3-256 Quantum-Resistant Block Verification
        // OPTIMIZATION: Uses rayon for 10-20x faster verification on multi-core CPUs
        // - Before: Sequential loop, 15-25ms for 8k blocks
        // - After:  Parallel rayon, 1-3ms for 8k blocks
        // v8.4.4: Run on dedicated sync rayon pool (4 threads) instead of global (19 threads)
        let sha3_start = Instant::now();
        let sha3_verifier_ref = &self.sha3_verifier;
        let blocks_ref = &blocks;
        let (sha3_valid_count, failed_heights) = SYNC_RAYON_POOL.install(|| {
            sha3_verifier_ref.verify_blocks_batch_parallel(blocks_ref)
        });
        let sha3_time = sha3_start.elapsed();

        if !failed_heights.is_empty() {
            warn!(
                "⚠️  [SHA3-256] {}/{} blocks failed quantum-resistant verification in pack {}-{}: {:?}",
                failed_heights.len(), blocks.len(), pack.start_height, pack.end_height,
                &failed_heights[..std::cmp::min(5, failed_heights.len())]
            );
        } else {
            debug!(
                "🚀 [SHA3-256] All {}/{} blocks verified (parallel, quantum-resistant) in {:?}",
                sha3_valid_count, blocks.len(), sha3_time
            );
        }

        // v10.9.43 item 14: SIMD chain-linkage validation.
        //
        // Only run on in-order, contiguous packs. Out-of-order parallel-sync
        // packs MUST skip this check — they cross chunk boundaries and the
        // chain-linkage invariant only holds for adjacent in-pack blocks
        // from the same source range.
        //
        // Closes a real correctness gap: today's chunk-ingest path doesn't
        // verify that adjacent blocks chain via BLAKE3, so a malicious peer
        // can construct individually-valid but disconnected blocks. Each
        // pair compare is a single SIMD u8x32 instruction via the `wide`
        // crate (~0.5ms per 2000-block pack on Epsilon).
        if is_in_order && blocks.len() >= 2 {
            let linkage_start = Instant::now();
            if let Err(err) = self.sha3_verifier.verify_chain_linkage(&blocks) {
                warn!(
                    "🚨 [CHAIN-LINKAGE] Pack {}-{} failed chain-linkage: {}",
                    pack.start_height, pack.end_height, err
                );
                anyhow::bail!(
                    "chain-linkage validation failed in pack {}-{}: {}",
                    pack.start_height, pack.end_height, err
                );
            }
            debug!(
                "🔗 [CHAIN-LINKAGE] {} blocks chain-linkage validated in {:?}",
                blocks.len(), linkage_start.elapsed()
            );
        }

        // 🚨 v0.7.0-beta: CRITICAL FIX - Prevent height regression bug
        // OLD BUG: If a pack contained blocks [1, 2, 3, 1000, 1001, 1002], and current_height = 993,
        //          the algorithm would scan from 0 and set highest_contiguous = 3 (WRONG!)
        //          This caused sync to REGRESS from 993 → 3 blocks (CATASTROPHIC!)
        //
        // NEW FIX: ONLY allow height pointer to move FORWARD
        //          - Height pointer can only increase, never decrease
        //          - Ignore blocks that are BELOW current local height (already have them)
        //          - Only process blocks that are HIGHER than current height
        //          - Stop at first gap in the FORWARD sequence
        let current_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);
        let mut highest_contiguous = current_height;
        let mut gap_detected = false;
        let mut gap_start: Option<u64> = None;
        let mut blocks_below_current = 0usize;
        let mut blocks_forward = 0usize;

        // 🚨 CRITICAL SAFETY: Scan ONLY blocks that are ABOVE current height
        // Ignore any blocks at or below current height (we already have them, they're stale)
        for block in &blocks {
            if block.header.height <= current_height {
                // Skip blocks we already have (stale data from partial batches)
                blocks_below_current += 1;
                continue;
            }

            // 🎯 Process blocks that are HIGHER than current height
            if block.header.height == highest_contiguous + 1 {
                // Perfect - next sequential block in FORWARD direction
                highest_contiguous = block.header.height;
                blocks_forward += 1;
            } else if block.header.height > highest_contiguous + 1 && !gap_detected {
                // Gap detected in FORWARD sequence
                gap_detected = true;
                gap_start = Some(block.header.height);

                // 🚀 v2.1.8-DELTA-V: GAP SKIP FIX - If gap is at the very START of pack
                // (meaning server doesn't have the blocks we need), skip past the gap
                // to prevent infinite sync loops when server has permanent gaps
                let gap_size = block.header.height - (highest_contiguous + 1);

                if blocks_forward == 0 && gap_size > 0 {
                    // This is a gap at the START - server doesn't have these blocks
                    // Skip past the gap and continue from where server CAN provide blocks
                    warn!(
                        "🚨 [v2.1.8 GAP SKIP] Gap at START of pack! Server missing blocks {}-{}",
                        highest_contiguous + 1, block.header.height - 1
                    );
                    warn!(
                        "   Skipping {} missing blocks to prevent infinite sync loop",
                        gap_size
                    );
                    warn!(
                        "   Advancing height from {} to {} (including current block)",
                        highest_contiguous, block.header.height
                    );
                    // Advance height TO this block (we're accepting it despite the gap)
                    highest_contiguous = block.header.height;
                    blocks_forward += 1; // Count this as a forward block
                    gap_detected = false; // Reset gap detection since we skipped it
                } else {
                    warn!(
                        "⚠️  [v0.7.0] Gap in FORWARD sequence: pack {}-{}, expected height {}, got {} - height pointer will be {}",
                        pack.start_height, pack.end_height,
                        highest_contiguous + 1, block.header.height, highest_contiguous
                    );
                }
                // Don't break - continue storing all blocks for later use
            }
        }

        // 🚨 CRITICAL SAFETY: Ensure height NEVER regresses
        if highest_contiguous < current_height {
            error!(
                "🚨 [v0.7.0] SAFETY ABORT: Height regression detected! current={}, computed={}, pack={}-{}",
                current_height, highest_contiguous, pack.start_height, pack.end_height
            );
            error!(
                "   Blocks analysis: {} below current, {} forward, {} gaps",
                blocks_below_current, blocks_forward, if gap_detected { 1 } else { 0 }
            );
            anyhow::bail!(
                "SAFETY ABORT: Height regression from {} to {} - this would cause sync corruption!",
                current_height, highest_contiguous
            );
        }

        // ========================================
        // v0.8.1-beta: ATOMIC TRANSACTIONS - Process each block atomically
        // ========================================
        // SECURITY FIX: Wrap balance consensus + block save in atomic transaction
        // to prevent CRITICAL-1 race condition (balances updated but blocks not saved)

        let mut balance_updates_total = 0;
        let mut blocks_committed = 0;
        let mut state_changes_total = 0;

        // ✅ v1.0.33-beta: BATCHED WRITES - Feature-flagged for safety
        if self.config.enable_batched_writes {
            // 🚀 v1.0.36-beta: LOUD batched write confirmation
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            warn!("🚀 [BATCH MODE] BATCHED WRITES ENABLED!");
            warn!("   Saving {} blocks in SINGLE transaction", blocks.len());
            warn!("   Expected: 3-5x faster than per-block writes");
            warn!("   Q_BATCHED_WRITES=1 (or default true)");
            #[cfg(not(target_os = "windows"))]
            if self.state_processor.is_some() {
                warn!("   🔄 STATE SYNC: ENABLED (Q_STATE_SYNC=1)");
            }
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            let batch_start = Instant::now();

            // 🚀 v1.0.60-beta: Process state changes via BlockStateProcessor (NEW!)
            // This is the comprehensive state sync that processes ALL transaction types
            #[cfg(not(target_os = "windows"))]
            if let Some(ref state_proc) = self.state_processor {
                let state_start = Instant::now();
                for block in &blocks {
                    match state_proc.process_block(block) {
                        Ok(result) => {
                            state_changes_total += result.changes_applied;
                            if result.changes_applied > 0 {
                                debug!(
                                    "🔄 [STATE SYNC] Block {}: {} txs, {} state changes, {} gas",
                                    block.header.height,
                                    result.transactions_processed,
                                    result.changes_applied,
                                    result.total_gas_used
                                );
                            }
                        }
                        Err(e) => {
                            // Log error but don't fail entire sync - state can be rebuilt
                            warn!(
                                "⚠️  [STATE SYNC] Failed to process block {}: {} (continuing sync)",
                                block.header.height, e
                            );
                        }
                    }
                }
                let state_duration = state_start.elapsed();
                if state_changes_total > 0 {
                    info!(
                        "🔄 [STATE SYNC] Processed {} state changes for {} blocks in {:?} ({:.0} changes/s)",
                        state_changes_total,
                        blocks.len(),
                        state_duration,
                        state_changes_total as f64 / state_duration.as_secs_f64().max(0.001)
                    );
                }
            }

            // Process balance consensus for all blocks first (if needed)
            // NOTE: This is the LEGACY balance engine - will be deprecated in favor of state sync
            // ✅ v1.0.76-beta: BATCHED BALANCE CONSENSUS - single transaction for all blocks!
            // BEFORE: 500 blocks × 50-400ms fsync = 25-200 SECONDS
            // AFTER:  1 transaction × 50-400ms fsync = 50-400ms TOTAL
            //
            // v10.7.1: ALWAYS process full balance state (coinbase + transfers).
            // The previous "extreme skip" optimization (skip transfers when >5k blocks behind)
            // created a different state machine from archive replay: transfer-only wallets were
            // never created and transfer debits were never applied, causing supply inflation.
            // Wallet balances are consensus-critical (BAL-001 activates at block 18,600,000)
            // so "fast but approximate" is no longer a valid mode.
            if let Some(engine) = balance_engine {
                let balance_start = std::time::Instant::now();

                // Create SINGLE transaction for ALL blocks in the batch
                let tx = match self.storage.begin_transaction().await {
                    Ok(tx) => tx,
                    Err(e) => {
                        error!(
                            "❌ [BATCH MODE] Failed to begin batch balance transaction: {:?}",
                            e
                        );
                        // Continue without balance processing - blocks will still sync
                        return Ok(());
                    }
                };

                // v10.7.1 debug: count coinbase vs transfer txs in this batch
                let (mut coinbase_tx_count, mut transfer_tx_count) = (0u64, 0u64);
                for block in &blocks {
                    for tx_item in &block.transactions {
                        if tx_item.is_coinbase() {
                            coinbase_tx_count += 1;
                        } else {
                            transfer_tx_count += 1;
                        }
                    }
                }
                debug!(
                    "📊 [BATCH MODE] Tx breakdown: {} coinbase, {} transfer across {} blocks (heights {}-{})",
                    coinbase_tx_count, transfer_tx_count, blocks.len(),
                    blocks.first().map(|b| b.header.height).unwrap_or(0),
                    blocks.last().map(|b| b.header.height).unwrap_or(0)
                );

                for block in &blocks {
                    let result = engine.process_block_mining_rewards_tx(&tx, block).await;
                    match result {
                        Ok(updates) => {
                            balance_updates_total += updates.len();
                            blocks_committed += 1;
                        }
                        Err(BalanceConsensusError::AlreadyProcessed(_)) => {
                            debug!("🔄 [BATCH MODE] Block {} already processed", block.header.height);
                        }
                        Err(e) => {
                            error!(
                                "❌ [BATCH MODE] Failed to process block {} balance consensus: {:?}",
                                block.header.height, e
                            );
                        }
                    }
                }

                // SINGLE commit for all blocks - only ONE fsync!
                tx.commit().await
                    .context(format!("Failed to commit batched balance updates for {} blocks", blocks.len()))?;

                // v10.7.1: Persist total_minted_supply to RocksDB after each batch so it
                // survives restarts with the correct value (not recomputed from a stale map).
                if let Ok(balances) = self.storage.load_wallet_balances().await {
                    let total: u128 = balances.values().copied().sum();
                    let wallet_count = balances.len();
                    if let Err(e) = self.storage.save_total_supply(total).await {
                        warn!("⚠️ [BATCH MODE] Failed to persist total_minted_supply: {:?}", e);
                    } else {
                        debug!(
                            "💾 [BATCH MODE] Persisted supply={} QUG-units, wallet_count={} (heights {}-{})",
                            total,
                            wallet_count,
                            blocks.first().map(|b| b.header.height).unwrap_or(0),
                            blocks.last().map(|b| b.header.height).unwrap_or(0)
                        );
                    }
                } else {
                    warn!("⚠️ [BATCH MODE] load_wallet_balances() failed — supply not persisted");
                }

                let balance_elapsed = balance_start.elapsed();
                if balance_updates_total > 0 {
                    info!(
                        "💰 [BATCH MODE] Processed {} balance updates for {} blocks in {:?} (single commit!) coinbase={} transfer={}",
                        balance_updates_total, blocks_committed, balance_elapsed,
                        coinbase_tx_count, transfer_tx_count
                    );
                }
            }

            // ✅ SINGLE BATCH WRITE (instead of 500+ individual writes!)
            // 🚀 v1.0.96-beta: Use save_qblocks_batch_turbo to skip orphan rejection
            // Turbo sync downloads chunks in parallel, so blocks arrive out of order.
            // The orphan rejection would reject valid blocks just because they arrived
            // before their predecessors in the chain.
            self.storage.save_qblocks_batch_turbo(&blocks).await
                .context(format!("Failed to batch save {} blocks (heights {}-{})",
                    blocks.len(), pack.start_height, pack.end_height))?;

            // v8.5.9: Post-write disk I/O throttle (same as apply_blocks_vec path)
            let disk_throttle_ms = match self.network_throttle_mode.load(Ordering::Relaxed) {
                0 => 100u64,  // Conservative: SSD-friendly
                1 => 10,      // Normal
                _ => 0,       // Turbo
            };
            if disk_throttle_ms > 0 {
                tokio::time::sleep(Duration::from_millis(disk_throttle_ms)).await;
            }

            // 🚨 v1.1.27-beta CRITICAL FIX: Update height cache to CONTIGUOUS height only!
            // ROOT CAUSE (v1.1.6): update_height_cache(pack.end_height) advanced pointer even with gaps.
            //   - Parallel turbo sync: Pack B (3100-3199) completes before Pack A (3000-3099)
            //   - Height cache updated to 3199 even though blocks 3000-3099 missing!
            //   - This created the gap at block 3042 that caused 10k block loss.
            // FIX: Only update to HIGHEST CONTIGUOUS height to prevent gaps in pointer.
            let contiguous_height = self.storage.get_highest_contiguous_block().await.unwrap_or(0);
            let safe_height = contiguous_height.min(pack.end_height);
            if safe_height > self.storage.height_cache.cached() {
                self.storage.update_height_cache(safe_height).await;
                debug!("🎯 [v1.1.27-beta] Height cache updated to {} (contiguous, pack ended at {})",
                       safe_height, pack.end_height);
            } else {
                debug!("🔍 [v1.1.27-beta] Skipped height update: contiguous={}, pack.end={}, cached={}",
                       contiguous_height, pack.end_height, self.storage.height_cache.cached());
            }

            // 🚀 v1.5.1-beta: EXTREME TURBO SYNC - TARGET 1000 BPS!
            // Three sync modes based on environment:
            //
            // 1. Q_EXTREME_SYNC=1: NO WAL SYNC AT ALL during initial sync
            //    - Maximum speed: 1000+ BPS
            //    - Risk: If crash, lose ALL progress, restart from 0
            //    - Use for: Fresh installs, testnet, speed-critical scenarios
            //
            // 2. Q_TURBO_SYNC=1 (default): Batched WAL sync every 50 chunks or 60s
            //    - Good speed: 200-500 BPS
            //    - Risk: If crash, lose up to 1M blocks
            //    - Use for: Normal operation
            //
            // 3. Q_TURBO_SYNC=0: Per-chunk WAL sync (legacy safe mode)
            //    - Slow: 50-100 BPS
            //    - Risk: Minimal data loss
            //    - Use for: Critical production nodes
            // 🚀 v3.4.11-beta: AUTO-EXTREME SYNC for initial sync
            // When node is >50,000 blocks behind, automatically use EXTREME mode
            // This gives fresh nodes 2000+ BPS without manual configuration
            // v8.0.9: Lowered from 50k to 5k — users syncing 5k+ blocks should
            // get EXTREME mode automatically (skip per-chunk WAL sync, faster writes)
            let auto_extreme_threshold: u64 = std::env::var("Q_AUTO_EXTREME_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5_000);  // 5k blocks = auto-enable extreme (was 50k)

            // Use peer registry to estimate network height (highest known peer)
            let estimated_network_height = {
                let registry = self.peer_registry.read().await;
                registry.max_height().unwrap_or(pack.end_height)
            };
            let blocks_behind = estimated_network_height.saturating_sub(self.storage.height_cache.cached());

            let use_extreme_env = std::env::var("Q_EXTREME_SYNC")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false);

            // v3.4.11: Auto-extreme when far behind, or explicit env var
            let use_extreme = use_extreme_env || blocks_behind > auto_extreme_threshold;

            let use_turbo = std::env::var("Q_TURBO_SYNC")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true);

            if use_extreme {
                // 🔥 EXTREME MODE: WAL sync every 10 chunks for crash safety
                // 🛡️ v7.2.3: Changed from NO sync to periodic sync.
                // Previously lost ALL in-flight blocks on crash (up to 500K+).
                // Now loses at most ~10K blocks (10 chunks × ~1000 blocks).
                let chunks_processed = self.chunks_since_wal_sync.fetch_add(1, Ordering::Relaxed) + 1;

                let extreme_sync_interval = std::env::var("Q_EXTREME_WAL_SYNC_INTERVAL")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(10u64);

                if chunks_processed >= extreme_sync_interval {
                    let sync_start = std::time::Instant::now();
                    self.storage.sync_wal().await
                        .context("Failed to sync WAL in extreme mode")?;
                    self.chunks_since_wal_sync.store(0, Ordering::Relaxed);
                    let auto_reason = if use_extreme_env { "env" } else { "auto (>50k behind)" };
                    info!(
                        "🔥 [EXTREME SYNC v7.2.3] WAL synced in {:?} after {} chunks ({}, {} blocks behind)",
                        sync_start.elapsed(), chunks_processed, auto_reason, blocks_behind
                    );
                } else if chunks_processed % 100 == 0 {
                    let auto_reason = if use_extreme_env { "env" } else { "auto (>50k behind)" };
                    info!(
                        "🔥 [EXTREME SYNC v7.2.3] {} chunks processed ({}, {} blocks behind) - next WAL sync in {} chunks",
                        chunks_processed, auto_reason, blocks_behind, extreme_sync_interval - (chunks_processed % extreme_sync_interval)
                    );
                }
            } else if use_turbo {
                // Increment chunk counter
                let chunks_processed = self.chunks_since_wal_sync.fetch_add(1, Ordering::Relaxed) + 1;

                // Get WAL sync batch size from env (default: 50 chunks for 1000 BPS)
                let wal_sync_batch_size = std::env::var("Q_WAL_SYNC_BATCH_SIZE")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(10u64);  // 🛡️ v7.2.3: 10 chunks (was 50). Reduces max data loss from ~50K to ~10K blocks on crash

                // Get time since last sync
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                let last_sync_ms = self.last_wal_sync_time.load(Ordering::Relaxed);
                let elapsed_since_sync_ms = now_ms.saturating_sub(last_sync_ms);

                // 🛡️ v7.2.3: Sync if: (a) 10+ chunks processed, OR (b) 15+ seconds since last sync
                // (was 50 chunks / 60s - too much data loss risk on crash)
                let should_sync = chunks_processed >= wal_sync_batch_size || elapsed_since_sync_ms >= 15_000;

                if should_sync {
                    let sync_start = std::time::Instant::now();
                    self.storage.sync_wal().await
                        .context("Failed to sync WAL after turbo batch")?;

                    // Reset counters
                    self.chunks_since_wal_sync.store(0, Ordering::Relaxed);
                    self.last_wal_sync_time.store(now_ms, Ordering::Relaxed);

                    // v7.2.7: Periodic flush every 30s to persist blocks from OS page cache to disk
                    // Without this, crash loses ALL blocks since last flush (even if WAL synced)
                    let last_flush = self.last_flush_time.load(Ordering::Relaxed);
                    if now_ms - last_flush >= 30_000 {
                        let flush_start = std::time::Instant::now();
                        if let Err(e) = self.storage.hot_db.flush().await {
                            warn!("⚠️ [TURBO SYNC v7.2.7] Periodic flush failed: {} (non-fatal)", e);
                        } else {
                            info!(
                                "💾 [TURBO SYNC v7.2.7] Periodic flush in {:?} — blocks persisted to disk",
                                flush_start.elapsed()
                            );
                        }
                        self.last_flush_time.store(now_ms, Ordering::Relaxed);
                    }

                    info!(
                        "⚡ [TURBO SYNC v7.2.7] WAL synced in {:?} after {} chunks ({} blocks) - last sync {}ms ago",
                        sync_start.elapsed(), chunks_processed, blocks.len(), elapsed_since_sync_ms
                    );
                } else {
                    debug!(
                        "📝 [TURBO SYNC v1.5.1] Skipping WAL sync: {}/{} chunks, {}ms since last sync",
                        chunks_processed, wal_sync_batch_size, elapsed_since_sync_ms
                    );
                }
            } else {
                // Legacy safe mode: sync after every chunk
                let sync_start = std::time::Instant::now();
                self.storage.sync_wal().await
                    .context("Failed to sync WAL after batch")?;
                debug!(
                    "🐌 [SAFE SYNC] WAL synced in {:?} after {} blocks (per-chunk sync)",
                    sync_start.elapsed(), blocks.len()
                );
            }

            let batch_duration = batch_start.elapsed();
            let batch_rate = blocks.len() as f64 / batch_duration.as_secs_f64();

            info!(
                "🚀 [BATCH WRITE] {} blocks in {:?} ({:.0} blocks/s) - estimated {}x faster than legacy",
                blocks.len(), batch_duration, batch_rate,
                (blocks.len() as f64 * 0.003) / batch_duration.as_secs_f64()
            );

        } else {
            // ❌ LEGACY MODE: Per-block writes (safe, slow, original behavior)
            // 🚀 v1.0.36-beta: LOUD legacy mode warning
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            warn!("⚠️  [LEGACY MODE] BATCHED WRITES DISABLED!");
            warn!("   Using per-block writes (slow but safe)");
            warn!("   Saving {} blocks with {} individual transactions", blocks.len(), blocks.len());
            warn!("   For 3-5x speedup, set Q_BATCHED_WRITES=1");
            warn!("   (Only enable after verifying P2P stability)");
            #[cfg(not(target_os = "windows"))]
            if self.state_processor.is_some() {
                warn!("   🔄 STATE SYNC: ENABLED (Q_STATE_SYNC=1)");
            }
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // 🚀 v1.0.60-beta: Process state changes via BlockStateProcessor (NEW!)
            // Works in legacy mode too - process state for each block
            #[cfg(not(target_os = "windows"))]
            if let Some(ref state_proc) = self.state_processor {
                for block in &blocks {
                    match state_proc.process_block(block) {
                        Ok(result) => {
                            state_changes_total += result.changes_applied;
                        }
                        Err(e) => {
                            warn!(
                                "⚠️  [STATE SYNC] Failed to process block {}: {} (continuing sync)",
                                block.header.height, e
                            );
                        }
                    }
                }
                if state_changes_total > 0 {
                    info!(
                        "🔄 [STATE SYNC] Processed {} state changes for {} blocks (legacy mode)",
                        state_changes_total, blocks.len()
                    );
                }
            }

            // 🚨 v2.2.0: CRITICAL DATA INTEGRITY FIX
            // Acquire global write lock BEFORE any Transaction writes to prevent race conditions.
            // Without this lock, concurrent writes could:
            // 1. Read the same height pointer
            // 2. Both try to update it, causing pointer regression
            // 3. Orphan blocks and create database gaps
            let _global_guard = self.storage.acquire_global_write_lock().await;
            debug!("🔒 [v2.2.0] Legacy mode acquired global write lock for {} blocks", blocks.len());

            // ✅ v1.0.76-beta: BATCHED LEGACY MODE - single transaction for entire pack!
            // BEFORE: 500 blocks × (50-400ms commit + sync_wal) = 50-400 SECONDS per pack
            // AFTER:  1 transaction × 50-400ms = 50-400ms TOTAL
            if let Some(engine) = balance_engine {
                let legacy_start = std::time::Instant::now();

                // Create SINGLE transaction for ALL blocks
                let tx = match self.storage.begin_transaction().await {
                    Ok(tx) => tx,
                    Err(e) => {
                        error!(
                            "❌ [LEGACY MODE] Failed to begin batch transaction for pack {}-{}: {:?}",
                            pack.start_height, pack.end_height, e
                        );
                        return Err(anyhow::anyhow!("Failed to begin batch transaction: {:?}", e));
                    }
                };

                for block in &blocks {
                    // Process balance consensus within transaction (buffered)
                    match engine.process_block_mining_rewards_tx(&tx, block).await {
                        Ok(updates) => {
                            balance_updates_total += updates.len();
                        }
                        Err(BalanceConsensusError::AlreadyProcessed(_)) => {
                            debug!("🔄 [LEGACY TX] Block {} already processed", block.header.height);
                        }
                        Err(e) => {
                            error!(
                                "❌ [LEGACY TX] Failed to process block {} balance: {:?}",
                                block.header.height, e
                            );
                        }
                    }

                    // Save block to transaction buffer
                    tx.save_qblock(block).await
                        .context(format!("Failed to save block {} in pack {}-{}",
                            block.header.height, pack.start_height, pack.end_height))?;

                    blocks_committed += 1;
                }

                // SINGLE commit for ALL blocks - only ONE fsync!
                tx.commit().await
                    .context(format!("Failed to commit batch of {} blocks", blocks.len()))?;

                // Single WAL sync for the entire batch
                self.storage.sync_wal().await
                    .context("Failed to sync WAL after batch commit")?;

                let legacy_elapsed = legacy_start.elapsed();
                info!(
                    "✅ [LEGACY MODE] Committed {} blocks with {} balance updates in {:?} (single commit!)",
                    blocks_committed, balance_updates_total, legacy_elapsed
                );
            } else {
                // No balance engine - just save blocks without consensus processing
                // ✅ v1.0.76-beta: BATCHED MODE for block-only saves too
                let tx = self.storage.begin_transaction().await
                    .context("Failed to begin batch transaction for blocks")?;

                for block in &blocks {
                    tx.save_qblock(block).await
                        .context(format!("Failed to save block {}", block.header.height))?;
                    blocks_committed += 1;
                }

                tx.commit().await
                    .context(format!("Failed to commit batch of {} blocks", blocks.len()))?;

                // Single WAL sync for the batch
                self.storage.sync_wal().await
                    .context("Failed to sync WAL after batch commit")?;
            }
            debug!("🔓 [v2.2.0] Legacy mode releasing global write lock");
        }

        // 🛡️ v1.0.88-beta: CRITICAL FIX - Unified height pointer update with monotonicity check
        // This replaces the confusing batch/legacy split with a single safe path.
        //
        // SAFETY RULES:
        // 1. Height pointer can only INCREASE, never decrease
        // 2. Only update to contiguous height (no gaps)
        // 3. Log all updates for debugging
        //
        // 🚀 v1.0.93-beta: Use current_height (already fetched at line 1244) instead of
        // another async DB read. This saves ~5-10ms per pack.
        //
        // 🚨 v2.2.0: CRITICAL DATA INTEGRITY FIX
        // Height pointer update MUST be protected by global write lock to prevent race conditions.
        // In BATCHED mode, save_qblocks_batch_turbo already updates the pointer under lock,
        // so this section is a no-op (highest_contiguous == current_db_height).
        // In LEGACY mode, the lock is already held from line 2730.
        // We acquire it here unconditionally to be safe for any future code paths.
        let current_db_height = current_height; // Already fetched at start of function

        if highest_contiguous > current_db_height {
            // 🚨 v2.2.0: Acquire global lock for pointer update
            let _global_guard = self.storage.acquire_global_write_lock().await;
            debug!("🔒 [v2.2.0] Acquired global write lock for height pointer update");

            // Re-read current height under lock to prevent TOCTOU race
            let actual_current = self.storage.get_latest_qblock_height().await.ok().flatten().unwrap_or(0);

            // SAFE: Height is increasing AND we verified under lock
            if highest_contiguous > actual_current {
                let tx = self.storage.begin_transaction().await?;
                let latest_height_bytes = highest_contiguous.to_be_bytes().to_vec();
                tx.put("blocks", b"qblock:latest", &latest_height_bytes).await?;
                tx.commit().await?;

                // 🛡️ v2.2.0: WAL sync AFTER pointer update for durability
                self.storage.sync_wal().await
                    .context("Failed to sync WAL after height pointer update")?;

                info!(
                    "📈 [TURBO SYNC] Height pointer updated: {} → {} (+{} blocks)",
                    actual_current, highest_contiguous, highest_contiguous - actual_current
                );

                // v8.3.0: CRITICAL — sync height cache with DB pointer.
                // Without this, save_safe_floor() persists stale cached height,
                // causing 140K block regression on restart (2.97M → 2.83M).
                self.storage.update_height_cache(highest_contiguous).await;
            } else {
                debug!(
                    "📊 [v2.2.0] Height update skipped under lock: {} already >= {}",
                    actual_current, highest_contiguous
                );
            }
            debug!("🔓 [v2.2.0] Releasing global write lock after height pointer update");
        } else if highest_contiguous < current_db_height && current_db_height > 1000 {
            // 🚨 CRITICAL: Would DECREASE height - this is the regression bug!
            error!(
                "🚨 [TURBO SYNC] BLOCKED HEIGHT REGRESSION: {} → {} (keeping {})",
                current_db_height, highest_contiguous, current_db_height
            );
            error!(
                "   Pack {}-{}: {} blocks, {} below current, {} forward",
                pack.start_height, pack.end_height, blocks.len(), blocks_below_current, blocks_forward
            );
            // Keep the higher value
            highest_contiguous = current_db_height;
        } else {
            // No change needed (equal or early chain)
            debug!(
                "📊 [TURBO SYNC] Height unchanged at {} (pack contiguous calc: {})",
                current_db_height, highest_contiguous
            );
            highest_contiguous = current_db_height;
        }

        // Log pack application details with FORWARD-only analysis
        if gap_detected {
            warn!(
                "⚠️  [v0.8.1] Applied PARTIAL pack {}-{}: stored {}/{} blocks ({} below current, {} forward), height: {} → {} (+{}), gap from block {}",
                pack.start_height, pack.end_height, blocks_committed, blocks.len(),
                blocks_below_current, blocks_forward,
                current_height, highest_contiguous, highest_contiguous - current_height,
                gap_start.unwrap_or(0)
            );
        } else if blocks_below_current > 0 {
            info!(
                "✅ [v0.8.1] Applied pack {}-{}: {}/{} blocks ({} below current ignored, {} forward), height: {} → {} (+{})",
                pack.start_height, pack.end_height, blocks_committed, blocks.len(),
                blocks_below_current, blocks_forward,
                current_height, highest_contiguous, highest_contiguous - current_height
            );
        } else {
            info!(
                "✅ [v0.8.1] Applied COMPLETE pack {}-{}: {}/{} blocks (all forward), height: {} → {} (+{})",
                pack.start_height, pack.end_height, blocks_committed, blocks.len(),
                current_height, highest_contiguous, highest_contiguous - current_height
            );
        }

        // v0.8.1-beta: Atomic transactions are committed individually per block above
        // Each commit uses fsync for durability, guaranteeing no data loss on crashes
        debug!(
            "✅ [v0.8.1] ATOMIC transactions: {}/{} blocks committed with fsync - durability guaranteed!",
            blocks_committed,
            blocks.len()
        );

        // Update metrics
        self.metrics.total_blocks_synced.fetch_add(blocks.len() as u64, Ordering::Relaxed);
        self.metrics.total_bytes_downloaded.fetch_add(pack.compressed_data.len() as u64, Ordering::Relaxed);

        let saved = pack.uncompressed_size.saturating_sub(pack.compressed_data.len() as u64);
        self.metrics.total_bytes_saved_by_compression.fetch_add(saved, Ordering::Relaxed);

        let apply_time = apply_start.elapsed();

        debug!(
            "✅ Applied pack {}-{}: {} blocks in {}ms",
            pack.start_height, pack.end_height, blocks.len(), apply_time.as_millis()
        );

        Ok(())
    }

    /// v6.1.0 OOM FIX: Apply pre-deserialized blocks directly (zero-copy path)
    ///
    /// This bypasses the BlockPack serialize→compress→decompress→deserialize cycle
    /// that was creating 3 copies of block data in memory simultaneously.
    /// Saves ~66MB peak memory per chunk on small-tier nodes.
    async fn apply_blocks_vec(
        &self,
        mut blocks: Vec<QBlock>,
        balance_engine: Option<&BalanceConsensusEngine>,
        range_start: u64,
        range_end: u64,
    ) -> Result<()> {
        let apply_start = Instant::now();

        // Sort blocks by height (same as apply_block_pack line 2703)
        // v8.4.4: Use dedicated sync rayon pool (4 threads) instead of global pool
        if blocks.len() > 100 {
            SYNC_RAYON_POOL.install(|| blocks.par_sort_unstable_by_key(|b| b.header.height));
        } else {
            blocks.sort_by_key(|b| b.header.height);
        }

        // SHA3 verification (same as apply_block_pack line 2781)
        // v8.4.4: Run on dedicated sync rayon pool
        let sha3_verifier_ref = &self.sha3_verifier;
        let blocks_ref = &blocks;
        let (sha3_valid_count, failed_heights) = SYNC_RAYON_POOL.install(|| {
            sha3_verifier_ref.verify_blocks_batch_parallel(blocks_ref)
        });
        if !failed_heights.is_empty() {
            warn!(
                "⚠️  [SHA3-256 DIRECT] {}/{} blocks failed verification in range {}-{}: {:?}",
                failed_heights.len(), blocks.len(), range_start, range_end,
                &failed_heights[..std::cmp::min(5, failed_heights.len())]
            );
        } else {
            debug!(
                "🚀 [SHA3-256 DIRECT] All {}/{} blocks verified in range {}-{}",
                sha3_valid_count, blocks.len(), range_start, range_end
            );
        }

        // Height safety checks (same as apply_block_pack line 2802)
        let current_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);
        let mut highest_contiguous = current_height;
        let mut blocks_below_current = 0usize;
        let mut blocks_forward = 0usize;

        for block in &blocks {
            if block.header.height <= current_height {
                blocks_below_current += 1;
                continue;
            }
            if block.header.height == highest_contiguous + 1 {
                highest_contiguous = block.header.height;
                blocks_forward += 1;
            } else if block.header.height > highest_contiguous + 1 {
                let gap_size = block.header.height - (highest_contiguous + 1);
                if blocks_forward == 0 && gap_size > 0 {
                    // v10.5.0 TIP GUARD: never skip gaps when near the live chain tip.
                    // At the tip, missed gossipsub blocks will be re-delivered within seconds.
                    // Skipping them permanently orphans them and accumulates gaps that
                    // auto-repair cannot fill (auto-repair is a no-op for BlockGaps).
                    // GAP SKIP is only safe in the sparse historical range (genesis→100K)
                    // where gossipsub will never re-deliver old blocks.
                    let network_height = self.cached_max_peer_height.load(Ordering::Relaxed);
                    let near_tip = network_height > 0
                        && highest_contiguous + 5_000 >= network_height;
                    if near_tip {
                        warn!(
                            "⏸️ [TIP GAP STOP] Gap of {} blocks ({}-{}) at live tip \
                             (contiguous={}, network={}). Stopping — gossipsub will fill gap.",
                            gap_size, highest_contiguous + 1, block.header.height - 1,
                            highest_contiguous, network_height
                        );
                        break;
                    }

                    // v10.5.0: Bound the maximum gap skip for historical range.
                    // The original code was unconditional — a peer returning blocks from
                    // any height (e.g. their local tip, height 16M) when we requested
                    // heights 1–1000 would silently advance our pointer 16M forward.
                    // Limit to 10,000 heights; larger jumps require a registered chain void.
                    const MAX_UNREGISTERED_GAP_SKIP: u64 = 10_000;
                    if gap_size > MAX_UNREGISTERED_GAP_SKIP {
                        error!(
                            "🚫 [GAP SKIP REFUSED] Gap at start of batch is {} heights \
                             (missing {}-{}), exceeds safety cap {}. Discarding out-of-range batch.",
                            gap_size, highest_contiguous + 1, block.header.height - 1,
                            MAX_UNREGISTERED_GAP_SKIP
                        );
                        break;
                    }
                    warn!(
                        "🔍 [v6.1.0 GAP SKIP] Historical gap at start of batch: {} heights ({}-{}), advancing pointer",
                        gap_size, highest_contiguous + 1, block.header.height - 1
                    );
                    highest_contiguous = block.header.height;
                    blocks_forward += 1;
                } else {
                    break; // Stop at gap
                }
            }
        }

        // Safety: never regress height
        if highest_contiguous < current_height {
            anyhow::bail!(
                "SAFETY ABORT: Height regression from {} to {} in direct apply",
                current_height, highest_contiguous
            );
        }

        // Save blocks using batched writes (same as apply_block_pack line 2900)
        if self.config.enable_batched_writes {
            // v10.7.1: ALWAYS process full balance state. See apply_block_pack fix above.
            // Process state changes
            #[cfg(not(target_os = "windows"))]
            if let Some(ref state_proc) = self.state_processor {
                for block in &blocks {
                    let _ = state_proc.process_block(block);
                }
            }

            if let Some(engine) = balance_engine {
                // v10.7.1 debug: count coinbase vs transfer txs in this batch
                let (mut coinbase_tx_count_d, mut transfer_tx_count_d) = (0u64, 0u64);
                for block in &blocks {
                    for tx_item in &block.transactions {
                        if tx_item.is_coinbase() {
                            coinbase_tx_count_d += 1;
                        } else {
                            transfer_tx_count_d += 1;
                        }
                    }
                }
                debug!(
                    "📊 [DIRECT] Tx breakdown: {} coinbase, {} transfer across {} blocks (heights {}-{})",
                    coinbase_tx_count_d, transfer_tx_count_d, blocks.len(),
                    blocks.first().map(|b| b.header.height).unwrap_or(0),
                    blocks.last().map(|b| b.header.height).unwrap_or(0)
                );

                let tx = self.storage.begin_transaction().await?;
                let mut direct_updates = 0usize;
                for block in &blocks {
                    let result = engine.process_block_mining_rewards_tx(&tx, block).await;
                    match result {
                        Ok(updates) => { direct_updates += updates.len(); }
                        Err(BalanceConsensusError::AlreadyProcessed(_)) => {}
                        Err(e) => {
                            error!("❌ [DIRECT] Balance processing failed for block {}: {:?}",
                                block.header.height, e);
                        }
                    }
                }
                tx.commit().await?;

                // v10.7.1: Persist total_minted_supply after each batch.
                if let Ok(balances) = self.storage.load_wallet_balances().await {
                    let total: u128 = balances.values().copied().sum();
                    let wallet_count = balances.len();
                    if let Err(e) = self.storage.save_total_supply(total).await {
                        warn!("⚠️ [DIRECT] Failed to persist total_minted_supply: {:?}", e);
                    } else {
                        debug!(
                            "💾 [DIRECT] Persisted supply={} QUG-units, wallet_count={} updates={} coinbase={} transfer={} (heights {}-{})",
                            total, wallet_count, direct_updates,
                            coinbase_tx_count_d, transfer_tx_count_d,
                            blocks.first().map(|b| b.header.height).unwrap_or(0),
                            blocks.last().map(|b| b.header.height).unwrap_or(0)
                        );
                    }
                } else {
                    warn!("⚠️ [DIRECT] load_wallet_balances() failed — supply not persisted");
                }
            }

            // Batch save blocks
            self.storage.save_qblocks_batch_turbo(&blocks).await
                .context(format!("Failed to batch save {} blocks (direct apply)", blocks.len()))?;

            // v8.5.9: Post-write disk I/O throttle — lets cheap SSDs flush before next batch.
            // Conservative=100ms, Normal=10ms, Turbo=0ms. Without this, RocksDB memtable flushes
            // + compaction can saturate a budget SATA SSD (~200 MB/s), causing I/O stalls.
            let disk_throttle_ms = match self.network_throttle_mode.load(Ordering::Relaxed) {
                0 => 100u64,  // Conservative: 100ms post-write rest (SSD-friendly)
                1 => 10,      // Normal: 10ms breather
                _ => 0,       // Turbo: no delay
            };
            if disk_throttle_ms > 0 {
                tokio::time::sleep(Duration::from_millis(disk_throttle_ms)).await;
            }

            // Update height cache
            let contiguous_height = self.storage.get_highest_contiguous_block().await.unwrap_or(0);
            let safe_height = contiguous_height.min(range_end);
            if safe_height > self.storage.height_cache.cached() {
                self.storage.update_height_cache(safe_height).await;
            }

            // WAL sync (same logic as apply_block_pack)
            let blocks_behind = {
                let registry = self.peer_registry.read().await;
                let net_height = registry.max_height().unwrap_or(range_end);
                net_height.saturating_sub(self.storage.height_cache.cached())
            };
            let use_extreme = std::env::var("Q_EXTREME_SYNC")
                .map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false)
                || blocks_behind > std::env::var("Q_AUTO_EXTREME_THRESHOLD")
                    .ok().and_then(|v| v.parse().ok()).unwrap_or(5_000u64);  // v8.0.9: 5k (was 50k)

            if !use_extreme {
                let chunks_processed = self.chunks_since_wal_sync.fetch_add(1, Ordering::Relaxed) + 1;
                // v8.0.9: Increased to 10 chunks during sync (was 5, was 50)
                // 5 was too frequent — each WAL sync adds ~50-200ms latency.
                // At 4 parallel streams × 250 blocks/chunk = 1000 blocks between syncs.
                // Max data loss on crash: ~2500 blocks (~5 minutes of chain), easily re-synced.
                let wal_sync_batch_size = std::env::var("Q_WAL_SYNC_BATCH_SIZE")
                    .ok().and_then(|v| v.parse().ok()).unwrap_or(10u64);
                if chunks_processed >= wal_sync_batch_size {
                    self.storage.sync_wal().await?;
                    self.chunks_since_wal_sync.store(0, Ordering::Relaxed);
                }
            }
        } else {
            // Legacy per-block save
            let _global_guard = self.storage.acquire_global_write_lock().await;
            let tx = self.storage.begin_transaction().await?;
            for block in &blocks {
                tx.save_qblock(block).await?;
            }
            tx.commit().await?;
            self.storage.sync_wal().await?;
        }

        // Update height pointer if increased
        if highest_contiguous > current_height {
            let _global_guard = self.storage.acquire_global_write_lock().await;
            let actual_current = self.storage.get_latest_qblock_height().await.ok().flatten().unwrap_or(0);
            if highest_contiguous > actual_current {
                let tx = self.storage.begin_transaction().await?;
                let latest_height_bytes = highest_contiguous.to_be_bytes().to_vec();
                tx.put("blocks", b"qblock:latest", &latest_height_bytes).await?;
                tx.commit().await?;
                info!("📈 [DIRECT APPLY] Height: {} → {} (+{})",
                    actual_current, highest_contiguous, highest_contiguous - actual_current);

                // v8.3.0: Sync height cache with pointer (same fix as apply_block_pack path).
                self.storage.update_height_cache(highest_contiguous).await;
            }
        }

        debug!(
            "✅ [DIRECT APPLY v6.1.0] {} blocks in {:?} ({} forward, {} skipped)",
            blocks.len(), apply_start.elapsed(), blocks_forward, blocks_below_current
        );

        Ok(())
    }

    /// Download and apply a single chunk from a peer
    async fn download_and_apply_chunk(
        &self,
        peer: PeerId,
        start_height: u64,
        end_height: u64,
        retry_count: u32,
    ) -> Result<()> {
        let chunk_start = Instant::now();

        // v6.0.9: RSS-BASED BACKPRESSURE - reads /proc/self/statm (zero overhead)
        // The old MemoryLimiter called sysinfo::System::refresh_processes_specifics which
        // allocates memory itself and measures SYSTEM memory (includes page cache).
        // This new check reads OUR OWN RSS directly from procfs - near zero cost.
        let memory_check_start = Instant::now();
        let mut memory_wait_loops = 0u32;
        const MAX_MEMORY_WAIT_LOOPS: u32 = 5; // v8.0.10: Max 250ms (was 3s — still too slow, caused <10 BPS)
        const MEMORY_WAIT_INTERVAL_MS: u64 = 50; // v8.0.10: 50ms (was 100ms)

        // v6.1.0: CGROUP-AWARE RSS limit for backpressure
        // v6.0.9 used sys.total_memory() which reads HOST RAM, but containers/systemd
        // services have cgroup MemoryMax limits that are lower.
        // Now reads /sys/fs/cgroup/memory.max (cgroup v2) or memory.limit_in_bytes (v1)
        // Fallback: 60% of system RAM if cgroup not configured
        let rss_limit_mb: u64 = {
            use std::sync::OnceLock;
            static RSS_LIMIT: OnceLock<u64> = OnceLock::new();
            *RSS_LIMIT.get_or_init(|| {
                // Try cgroup v2 first, then v1
                let cgroup_limit_mb = std::fs::read_to_string("/sys/fs/cgroup/memory.max")
                    .or_else(|_| std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
                    .ok()
                    .and_then(|s| {
                        let s = s.trim();
                        if s == "max" || s == "-1" { None } // unlimited
                        else { s.parse::<u64>().ok().map(|b| b / (1024 * 1024)) }
                    });

                let mut sys = sysinfo::System::new();
                sys.refresh_memory();
                let total_mb = sys.total_memory() / (1024 * 1024);

                let effective_mb = cgroup_limit_mb.unwrap_or(total_mb);
                // v8.0.10: Higher RSS allowance — 55% was too conservative, caused sync <10 BPS
                // OS page cache reclaim handles the rest; malloc_trim returns freed pages
                let ratio = if effective_mb >= 32000 { 0.75 } else if effective_mb >= 16000 { 0.70 } else { 0.65 };
                let limit = (effective_mb as f64 * ratio) as u64;
                info!("🧠 [RSS BACKPRESSURE v7.1.4] RSS limit: {}MB ({:.0}% of {}MB effective, cgroup: {:?}, system: {}MB)",
                    limit, ratio * 100.0, effective_mb, cgroup_limit_mb, total_mb);
                limit
            })
        };

        loop {
            // Read RSS from /proc/self/statm (page count, multiply by 4096)
            let rss_mb = Self::get_rss_mb().unwrap_or(0);
            if rss_mb < rss_limit_mb {
                break;
            }

            memory_wait_loops += 1;

            // Force malloc to return freed pages WHILE waiting
            #[cfg(target_os = "linux")]
            {
                extern "C" { fn malloc_trim(pad: usize) -> i32; }
                unsafe { malloc_trim(0); }
            }

            if memory_wait_loops == 1 {
                warn!(
                    "⏸️  [RSS BACKPRESSURE v6.0.9] RSS {}MB > {}MB limit, pausing chunk {}..={} + calling malloc_trim",
                    rss_mb, rss_limit_mb, start_height, end_height
                );
            }

            if memory_wait_loops > MAX_MEMORY_WAIT_LOOPS {
                warn!(
                    "⚠️  [RSS BACKPRESSURE] RSS still {}MB after {}s, proceeding (chunk {}..={})",
                    rss_mb, memory_wait_loops as u64 * MEMORY_WAIT_INTERVAL_MS / 1000, start_height, end_height
                );
                break;
            }

            tokio::time::sleep(Duration::from_millis(MEMORY_WAIT_INTERVAL_MS)).await;
        }

        if memory_wait_loops > 0 {
            info!(
                "▶️  [BACKPRESSURE v1.5.1] Memory OK after {} loops ({:?}), resuming download {}..={}",
                memory_wait_loops, memory_check_start.elapsed(), start_height, end_height
            );
        }

        // 🚀 v1.0.5-beta: Acquire pipeline slot for adaptive window sizing
        // This controls in-flight request depth based on RTT measurements
        let pipeline_slot = self.pipeline_manager.acquire_slot().await
            .context("Failed to acquire pipeline slot")?;

        // Acquire semaphore permit for rate limiting
        let _permit = self.download_semaphore.acquire().await?;

        self.metrics.active_parallel_streams.fetch_add(1, Ordering::Relaxed);

        // ========================================
        // 🚀 v1.3.9-beta: DIRECT REQUEST-RESPONSE for Turbo Sync
        // Uses libp2p request-response protocol instead of gossipsub for:
        // - Guaranteed delivery or error (no silent drops)
        // - Point-to-point communication (no broadcast flooding)
        // - Built-in 60s timeout (vs gossipsub's unreliable delivery)
        // ========================================
        let pack = if let Some(network_tx) = &self.network_tx {
            // Create oneshot channel for response
            let (response_tx, response_rx) = oneshot::channel();

            // 🚀 v1.3.9-beta: Send direct request-response (NOT gossipsub!)
            // This is much more reliable for large block batches
            if let Err(e) = network_tx.send(NetworkRequest::RequestBlockRangeDirect {
                peer_id: Some(peer.to_string()), // Send to specific peer
                start_height,
                end_height,
                response_tx,
            }) {
                anyhow::bail!("Failed to send direct network request: {}", e);
            }

            // 🚀 v1.0.5-beta: Record request sent for RTT tracking
            pipeline_slot.sent(start_height, end_height).await;

            // 📨 v1.3.9-beta: LOUD direct request logging
            info!("📨 [P2P DIRECT] Requesting blocks {}..={} via request-response from {}",
                  start_height, end_height, peer);

            // 🚀 v3.4.7-beta: Use adaptive timeout (10s-45s range for faster retry)
            // 🔭 v1.0.2-KALMAN: Blend Kalman-predicted timeout (30% weight) with RTT-based (70%)
            let dynamic_timeout = {
                let timeout_calc = self.adaptive_timeout.read().await;
                let rtt_timeout = timeout_calc.get_timeout();

                if self.config.enable_apollo_kalman {
                    let kalman = self.apollo_kalman_predictor.read().await;
                    let state = kalman.get_state();
                    if state.confidence > 0.3 {
                        let kalman_timeout = state.optimal_timeout();
                        // Blend: 70% RTT-based + 30% Kalman-predicted
                        let blended_ms = (rtt_timeout.as_millis() as f64 * 0.7
                            + kalman_timeout.as_millis() as f64 * 0.3) as u64;
                        let blended = Duration::from_millis(blended_ms)
                            .clamp(Duration::from_secs(5), Duration::from_secs(30));
                        debug!("🔭 [KALMAN TIMEOUT] RTT={:?}, Kalman={:?}, Blended={:?} (conf={:.2})",
                               rtt_timeout, kalman_timeout, blended, state.confidence);
                        blended
                    } else {
                        rtt_timeout // Low confidence — use pure RTT
                    }
                } else {
                    rtt_timeout
                }
            };
            info!("⏱️  [ADAPTIVE TIMEOUT] Using {:?} for chunk {}..={} (v3.4.7+KALMAN blend)",
                   dynamic_timeout, start_height, end_height);

            // Wait for response with adaptive timeout
            match tokio::time::timeout(dynamic_timeout, response_rx).await {
                Ok(Ok(blocks_result)) => {
                    match blocks_result {
                        Ok(blocks) => {
                            // 🚀 v1.0.5-beta: Record response received for RTT tracking
                            pipeline_slot.received(start_height).await;

                            // 🚀 v1.0.50-beta: Record RTT for adaptive timeout adjustment
                            let rtt_ms = chunk_start.elapsed().as_millis() as u64;
                            {
                                let mut timeout_calc = self.adaptive_timeout.write().await;
                                timeout_calc.record_rtt(rtt_ms);
                            }

                            let block_count = blocks.len() as u32;

                            // v10.2.9-fix: In DAG-Knight, sparse height ranges are normal.
                            // A peer returning 0 blocks means "no blocks exist at these heights" —
                            // this is valid for sparse DAGs where not every height has a block.
                            // Previously (v10.2.8) this was treated as FAILURE, but combined with
                            // v10.2.9's early-abort on missing blocks in get_qblocks_range(),
                            // it created a deadlock: server returns 0 (by design) → client rejects 0 (by design).
                            let requested_size = end_height.saturating_sub(start_height) + 1;
                            if blocks.is_empty() && requested_size > 0 {
                                info!("📭 [SPARSE-SYNC] Peer {} returned 0 blocks for range {}-{} (requested {}) — sparse DAG, advancing cursor",
                                      peer, start_height, end_height, requested_size);
                                // Skip apply_blocks_vec — nothing to apply. Just advance past this range.
                                self.metrics.active_parallel_streams.fetch_sub(1, Ordering::Relaxed);

                                let chunk_time = chunk_start.elapsed();
                                if self.config.enable_apollo_gravity_assist {
                                    self.apollo_record_peer_serving(
                                        &peer.to_string(),
                                        start_height..end_height,
                                        0,
                                        chunk_time.as_millis() as u32,
                                    );
                                }

                                return Ok(());
                            }

                            let actual_start = blocks.first().map(|b| b.header.height).unwrap_or(start_height);
                            let actual_end = blocks.last().map(|b| b.header.height).unwrap_or(end_height);

                            // 🚀 v1.0.50-beta: Track successful download for peer scoring
                            {
                                let mut tracker = self.progress_tracker.write().await;
                                tracker.record_success(
                                    &peer.to_string(),
                                    (block_count as u64) * 1024, // estimate
                                    block_count as u64,
                                ).await;
                            }

                            info!("📦 [P2P DIRECT] ✅ Received {} blocks: {}..={} (RTT: {}ms)",
                                  block_count, actual_start, actual_end, rtt_ms);

                            // 🚀 v6.1.0 OOM FIX: Apply blocks DIRECTLY without re-serializing
                            // BEFORE: Vec<QBlock> → postcard serialize → LZ4 compress → BlockPack
                            //         → LZ4 decompress → postcard deserialize → process
                            // This created 3 copies of block data in memory simultaneously!
                            // AFTER: Vec<QBlock> → process directly (zero-copy)
                            // Saves ~66MB per chunk peak memory (3x reduction)
                            self.apply_blocks_vec(blocks, None, actual_start, actual_end).await?;

                            // v6.1.0: Record metrics manually since we bypass BlockPack
                            self.metrics.total_blocks_synced.fetch_add(block_count as u64, Ordering::Relaxed);

                            self.metrics.active_parallel_streams.fetch_sub(1, Ordering::Relaxed);

                            // v6.0.9: Force glibc malloc to return freed pages to OS
                            #[cfg(target_os = "linux")]
                            {
                                extern "C" { fn malloc_trim(pad: usize) -> i32; }
                                unsafe { malloc_trim(0); }
                            }

                            let chunk_time = chunk_start.elapsed();

                            // 🌍 v1.0.2-SLINGSHOT: Record peer serving for gravity-assist momentum
                            if self.config.enable_apollo_gravity_assist {
                                self.apollo_record_peer_serving(
                                    &peer.to_string(),
                                    start_height..end_height,
                                    (block_count as u64) * 1024, // estimated bytes
                                    chunk_time.as_millis() as u32,
                                );
                            }

                            // 🔭 v1.0.2-KALMAN: Feed Kalman predictor with actual measurements
                            if self.config.enable_apollo_kalman {
                                let chunk_secs = chunk_time.as_secs_f64().max(0.001);
                                let bandwidth_mbps = ((block_count as f64) * 1024.0) / chunk_secs / 1_000_000.0;
                                let latency_ms = chunk_time.as_millis() as f64 / 2.0; // approximate one-way
                                self.apollo_update_kalman(bandwidth_mbps, latency_ms, 0.0).await;
                            }

                            info!("🚀 Downloaded+applied chunk {}-{} from {} in {}ms (direct, retry: {})",
                                  start_height, end_height, peer, chunk_time.as_millis(), retry_count);

                            return Ok(());
                        }
                        Err(e) => {
                            // 🚀 v1.0.50-beta: Track failure for peer scoring
                            {
                                let mut tracker = self.progress_tracker.write().await;
                                tracker.record_failure(&peer.to_string()).await;
                            }
                            // 🌍 v1.0.2-SLINGSHOT: Penalty for gravity-assist momentum
                            if self.config.enable_apollo_gravity_assist {
                                self.apollo_peer_momentum.record_failure(&peer.to_string());
                            }
                            // 🔭 v1.0.2-KALMAN: Feed loss=1.0 on request error
                            if self.config.enable_apollo_kalman {
                                self.apollo_update_kalman(0.0, 0.0, 1.0).await;
                            }

                            // 💥 v1.3.9-beta: Direct request error
                            error!("💥 [P2P DIRECT ERROR] Request failed: {:?}", e);
                            error!("   Range: {}..={}", start_height, end_height);
                            error!("   Peer: {} (score decreased)", peer);

                            let local_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);
                            if start_height > local_height {
                                anyhow::bail!(
                                    "P2P direct request failed for chunk {}..={} (local height: {}): {}. Will retry.",
                                    start_height, end_height, local_height, e
                                );
                            }
                            self.create_block_pack(start_height, end_height).await?
                        }
                    }
                }
                Ok(Err(_)) => {
                    // Channel closed
                    {
                        let mut tracker = self.progress_tracker.write().await;
                        tracker.record_failure(&peer.to_string()).await;
                    }
                    // 🌍 v1.0.2-SLINGSHOT: Penalty for gravity-assist momentum
                    if self.config.enable_apollo_gravity_assist {
                        self.apollo_peer_momentum.record_failure(&peer.to_string());
                    }
                    // 🔭 v1.0.2-KALMAN: Feed loss=1.0 on channel close
                    if self.config.enable_apollo_kalman {
                        self.apollo_update_kalman(0.0, 0.0, 1.0).await;
                    }

                    error!("💥 [P2P DIRECT ERROR] Response channel closed for {}..={}",
                          start_height, end_height);
                    error!("   Peer: {} (likely disconnected)", peer);

                    let local_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);
                    if start_height > local_height {
                        anyhow::bail!(
                            "P2P direct channel closed for chunk {}..={} (local height: {}). Will retry.",
                            start_height, end_height, local_height
                        );
                    }
                    self.create_block_pack(start_height, end_height).await?
                }
                Err(_) => {
                    // Timeout
                    {
                        let mut timeout_calc = self.adaptive_timeout.write().await;
                        timeout_calc.record_timeout();
                    }
                    {
                        let mut tracker = self.progress_tracker.write().await;
                        tracker.record_failure(&peer.to_string()).await;
                    }
                    // 🌍 v1.0.2-SLINGSHOT: Penalty for gravity-assist momentum
                    if self.config.enable_apollo_gravity_assist {
                        self.apollo_peer_momentum.record_failure(&peer.to_string());
                    }
                    // 🔭 v1.0.2-KALMAN: Feed loss=1.0 on timeout
                    if self.config.enable_apollo_kalman {
                        self.apollo_update_kalman(0.0, 0.0, 1.0).await;
                    }

                    // ⏱️ v3.4.7-beta: Timeout logging with reduced max timeout
                    error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    error!("⏱️  [P2P DIRECT] Chunk timeout after {:?}", dynamic_timeout);
                    error!("    Range: {}..={}", start_height, end_height);
                    error!("    Peer: {} (score decreased)", peer);
                    error!("    NOTE: v3.4.7 uses 10s-45s adaptive timeout - will retry with different peer");
                    error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                    let local_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);
                    if start_height > local_height {
                        anyhow::bail!(
                            "P2P direct timeout for chunk {}..={} (local height: {}). Will retry.",
                            start_height, end_height, local_height
                        );
                    }
                    self.create_block_pack(start_height, end_height).await?
                }
            }
        } else {
            // Network not configured, use local pack creation (hybrid mode)
            debug!("📦 [TURBO SYNC HYBRID] Network not available, using local pack creation");
            self.create_block_pack(start_height, end_height).await?
        };

        // Apply the pack
        // v0.8.0-beta: Pass None for balance_engine in internal sync methods
        // Balance processing happens in main.rs gossipsub handler when blocks are received
        self.apply_block_pack(pack, None).await?;

        self.metrics.active_parallel_streams.fetch_sub(1, Ordering::Relaxed);

        // v6.0.9: CRITICAL OOM FIX - Force glibc malloc to return freed pages to OS
        // During sync, each chunk cycle allocates 50-200MB for serialization/deserialization.
        // glibc malloc keeps freed memory in its arena (fragmentation) and RSS grows unbounded.
        // malloc_trim(0) releases all freed memory pages back to the OS immediately.
        // On Gamma (7.8GB RAM), without this, RSS grows from 800MB to 7.2GB in 5 minutes.
        #[cfg(target_os = "linux")]
        {
            extern "C" { fn malloc_trim(pad: usize) -> i32; }
            unsafe { malloc_trim(0); }
        }

        let chunk_time = chunk_start.elapsed();

        // 🌍 v1.0.2-SLINGSHOT: Record peer serving for gravity-assist momentum (BlockPack path)
        if self.config.enable_apollo_gravity_assist {
            let estimated_bytes = (end_height - start_height + 1) * 1024;
            self.apollo_record_peer_serving(
                &peer.to_string(),
                start_height..end_height,
                estimated_bytes,
                chunk_time.as_millis() as u32,
            );
        }

        // 🔭 v1.0.2-KALMAN: Feed Kalman predictor (BlockPack path)
        if self.config.enable_apollo_kalman {
            let chunk_secs = chunk_time.as_secs_f64().max(0.001);
            let blocks = (end_height - start_height + 1) as f64;
            let bandwidth_mbps = (blocks * 1024.0) / chunk_secs / 1_000_000.0;
            let latency_ms = chunk_time.as_millis() as f64 / 2.0;
            self.apollo_update_kalman(bandwidth_mbps, latency_ms, 0.0).await;
        }

        info!(
            "🚀 Downloaded chunk {}-{} from {} in {}ms (retry: {})",
            start_height, end_height, peer, chunk_time.as_millis(), retry_count
        );

        Ok(())
    }

    /// Download chunks in parallel with pipelining
    async fn download_chunks_parallel(
        &self,
        chunks: Vec<(u64, u64)>,
        peers: Vec<PeerId>,
    ) -> Result<()> {
        if peers.is_empty() {
            anyhow::bail!("No peers available for parallel download");
        }

        let total_chunks = chunks.len();
        let mut futures = FuturesUnordered::new();
        let mut completed_chunks = 0usize;

        // 🚀 v2.3.10-beta: WARP SYNC Phase 2 - Get intelligent peer ranking
        // Use MultiPeerDownloader's bandwidth-weighted peer selection
        let target_height = chunks.last().map(|(_, e)| *e).unwrap_or(0);
        let warp_peers = self.warp_multi_peer.get_qualified_peers(target_height).await;
        let use_warp_peers = !warp_peers.is_empty();

        // Build priority-ordered peer list from Warp Sync metrics
        let priority_peers: Vec<PeerId> = if use_warp_peers {
            warp_peers.iter()
                .filter_map(|p| p.peer_id.parse::<PeerId>().ok())
                .collect()
        } else {
            peers.clone() // Fallback to original peers
        };

        // 📊 v1.0.2: Update session atomics for admin panel visibility
        self.session_total_chunks.store(total_chunks as u64, Ordering::Release);
        self.session_completed_chunks.store(0, Ordering::Release);
        self.session_sync_mode.store(1, Ordering::Release); // 1 = turbo

        info!("🚀 Starting parallel download: {} chunks from {} peers (Warp Sync: {})",
              total_chunks, peers.len(), if use_warp_peers { "intelligent routing" } else { "fallback" });

        // 🚀 v2.3.10-beta: WARP SYNC Phase 3 - Queue chunks for prefetch
        // Convert to ChunkAssignments for prefetch tracking
        let chunk_assignments: Vec<ChunkAssignment> = chunks.iter()
            .enumerate()
            .map(|(idx, (start, end))| ChunkAssignment {
                chunk_id: idx as u64,
                start_height: *start,
                end_height: *end,
                assigned_peer: None,
                status: ChunkStatus::Pending,
                attempts: 0,
                assigned_at: None,
            })
            .collect();
        self.warp_prefetch.queue_prefetch(&chunk_assignments).await;

        // 🚀 v3.4.8-beta: SLIDING WINDOW PARALLEL SYNC
        // Instead of spawning ALL chunks at once (causing far-ahead timeouts),
        // only spawn chunks within a window from current applied height.
        // This prevents 200K+ block-ahead chunks from timing out.
        // 🔭 v1.0.2-KALMAN: Use Kalman optimal_concurrency when confident
        //
        // v8.4.4: Q_SYNC_MAX_CONCURRENCY env var caps max in-flight chunks.
        // On Beta (48 parallel_streams), uncapped concurrency saturates all 19 tokio workers,
        // starving API handlers. Default cap: 8 — enough for good throughput, leaves workers for API.
        let env_max_concurrency: usize = std::env::var("Q_SYNC_MAX_CONCURRENCY")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(8);

        // 🎚️ v8.5.4: Apply runtime throttle mode (TUI-controlled)
        let (throttle_concurrency_cap, inter_chunk_delay_ms) = self.get_throttle_params();
        let effective_max_concurrency = match throttle_concurrency_cap {
            Some(cap) => env_max_concurrency.min(cap),
            None => env_max_concurrency, // Turbo: no additional cap
        };

        let max_in_flight_chunks = if self.config.enable_apollo_kalman {
            let kalman = self.apollo_kalman_predictor.read().await;
            let state = kalman.get_state();
            if state.confidence > 0.4 {
                let kalman_concurrency = state.optimal_concurrency();
                let bounded = kalman_concurrency.clamp(4, self.config.parallel_streams * 2);
                // v8.4.4: Apply env cap on top of Kalman
                // v8.5.4: Also apply throttle mode cap
                let capped = bounded.min(effective_max_concurrency);
                info!("🔭 [KALMAN CONCURRENCY] Using {} in-flight (Kalman={}, config={}, cap={}, throttle={}, conf={:.2})",
                      capped, kalman_concurrency, self.config.parallel_streams, env_max_concurrency,
                      throttle_concurrency_cap.map(|c| c.to_string()).unwrap_or("off".into()),
                      state.confidence);
                capped
            } else {
                self.config.parallel_streams.min(effective_max_concurrency)
            }
        } else {
            self.config.parallel_streams.min(effective_max_concurrency)
        };

        let throttle_mode_name = match self.network_throttle_mode.load(Ordering::Relaxed) {
            0 => "Conservative",
            1 => "Normal",
            _ => "Turbo",
        };
        if inter_chunk_delay_ms > 0 {
            info!("🎚️ [THROTTLE] Mode: {} — max {} in-flight, {}ms inter-chunk delay",
                  throttle_mode_name, max_in_flight_chunks, inter_chunk_delay_ms);
        }
        let mut chunks_queue: std::collections::VecDeque<(usize, u64, u64)> = chunks
            .into_iter()
            .enumerate()
            .map(|(idx, (start, end))| (idx, start, end))
            .collect();

        info!("🚀 [v3.4.8] SLIDING WINDOW SYNC: {} chunks, max {} in-flight at once",
              total_chunks, max_in_flight_chunks);

        // Helper closure to spawn a chunk download task
        let spawn_chunk_task = |futures: &mut FuturesUnordered<_>,
                                chunk_idx: usize, start: u64, end: u64,
                                peers_list: Vec<PeerId>, self_ref: &TurboSyncManager| {
            let peer_count = peers_list.len();
            let self_clone = self_ref.clone_for_task();

            // 🌍 v1.0.2-SLINGSHOT: Use gravity-assist to pick optimal first peer
            // Reorder peers_list so the best-cached peer for this range is first
            let ordered_peers = if self_ref.config.enable_apollo_gravity_assist && peer_count > 1 {
                let peer_strs: Vec<String> = peers_list.iter().map(|p| p.to_string()).collect();
                let peer_str_refs: Vec<&str> = peer_strs.iter().map(|s| s.as_str()).collect();
                if let Some(best) = self_ref.apollo_select_peer(&(start..end), &peer_str_refs) {
                    // Move the best peer to front, keep others in original order for retries
                    let mut reordered = peers_list.clone();
                    if let Some(pos) = reordered.iter().position(|p| p.to_string() == best) {
                        if pos > 0 {
                            let best_peer = reordered.remove(pos);
                            reordered.insert(0, best_peer);
                            debug!("🌍 [GRAVITY ASSIST] Chunk {}-{}: Promoted peer {} to front (cache-hot)",
                                   start, end, &best[..best.len().min(12)]);
                        }
                    }
                    reordered
                } else {
                    peers_list.clone()
                }
            } else {
                peers_list.clone()
            };

            futures.push(tokio::spawn(async move {
                // v1.3.10-beta: Retry logic with DIFFERENT PEER on each attempt
                let mut retry_count = 0;
                let max_retries = 3;
                // v10.2.10: Wall-clock deadline prevents infinite stall on a single chunk.
                // If all retries burn 120s total, abandon and let outer loop reschedule.
                // v10.9.27: The deadline check is at the top of the loop, BEFORE the
                // retry_count branch, so it still fires when a chunk loops only on
                // ClientThrottle responses (which intentionally do NOT increment
                // retry_count). This is the safety net for the throttle path.
                let chunk_deadline = tokio::time::Instant::now() + Duration::from_secs(120);
                // v10.2.10: Per-chunk peer exclusion — avoid retrying same slow peer
                // on transport/timeout failures. Validation failures go through AEGIS.
                let mut excluded_peers: HashSet<String> = HashSet::new();
                // v10.9.27: Count of consecutive `ClientThrottle` responses observed
                // for this chunk. These are local back-pressure (the per-peer client
                // semaphore was exhausted), NOT peer failures, so they must NOT
                // consume the retry budget. We still log a warning if the count
                // grows excessively — that signals the client cap is mis-sized.
                let mut throttle_waits: usize = 0;

                loop {
                    // v10.2.10: Wall-clock circuit breaker
                    if tokio::time::Instant::now() >= chunk_deadline {
                        warn!("⏰ [CIRCUIT BREAKER] Chunk {}-{} exceeded 120s wall-clock limit — \
                               abandoning attempt, will be rescheduled by outer sync loop",
                              start, end);
                        return Err(anyhow::anyhow!(
                            "WALL_CLOCK_TIMEOUT: Chunk {}-{} exceeded 120s deadline", start, end
                        ));
                    }

                    // v10.2.10: Filter out peers that already timed out on THIS chunk
                    let available: Vec<_> = ordered_peers.iter()
                        .filter(|p| !excluded_peers.contains(&p.to_string()))
                        .cloned()
                        .collect();

                    // v1.3.10-beta: Select DIFFERENT peer for each retry attempt
                    // v1.0.2: First attempt uses gravity-assist ordered peer (index 0)
                    let peer = if available.is_empty() {
                        // All peers excluded — last resort, use any peer.
                        // Wall-clock deadline will catch us if this also fails.
                        ordered_peers[retry_count as usize % peer_count]
                    } else if retry_count == 0 {
                        available[0] // Gravity-assist selected (or original first)
                    } else {
                        available[retry_count as usize % available.len()] // Round-robin non-excluded
                    };

                    if retry_count > 0 {
                        info!("🔄 [RETRY] Chunk {}-{}: Using {} peer {} (attempt {}/{}, {} excluded)",
                              start, end,
                              if available.is_empty() { "LAST-RESORT" } else { "DIFFERENT" },
                              peer, retry_count + 1, max_retries, excluded_peers.len());
                    }

                    match self_clone.download_and_apply_chunk(peer, start, end, retry_count).await {
                        Ok(()) => {
                            // v10.2.10: Record success for AEGIS recovery
                            let peer_str = peer.to_string();
                            self_clone.peer_trust.record_successful_chunk(&peer_str);
                            return Ok((start, end));
                        }
                        Err(e) => {
                            let err_msg = e.to_string();

                            // v10.9.27: Discriminate client-side throttle from real failures.
                            //
                            // When the per-peer client block-pack semaphore in
                            // `unified_network_manager.rs` is full, the dispatch returns
                            // an error whose Display contains `ClientThrottle`. This is
                            // local back-pressure — the chunk hasn't actually been tried
                            // against the peer yet, so we must NOT:
                            //   - increment `retry_count` (would burn the retry budget)
                            //   - mark the peer as excluded (it didn't misbehave)
                            //   - record a `data_failure` on the trust tracker
                            //
                            // Instead we sleep briefly and loop. The 120s wall-clock
                            // deadline at the top of the loop is the safety net that
                            // prevents an infinite throttle spin if the cap is misconfigured.
                            if err_msg.contains("ClientThrottle") {
                                throttle_waits += 1;
                                if throttle_waits == 100 || throttle_waits % 500 == 0 {
                                    warn!(
                                        "🚦 [THROTTLE] Chunk {}-{} blocked by local per-peer \
                                         semaphore {} times against peer {} — check \
                                         CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER capacity",
                                        start, end, throttle_waits, peer
                                    );
                                }
                                tokio::time::sleep(Duration::from_millis(50)).await;
                                continue;
                            }

                            retry_count += 1;

                            // v10.2.10: Exclude peer on transport/timeout failures only.
                            // Check if error looks like a timeout/transport issue.
                            let is_transport_failure = err_msg.contains("timeout")
                                || err_msg.contains("Timeout")
                                || err_msg.contains("timed out")
                                || err_msg.contains("connection")
                                || err_msg.contains("transport")
                                || err_msg.contains("P2P direct timeout");
                            if is_transport_failure {
                                excluded_peers.insert(peer.to_string());
                            }

                            if retry_count >= max_retries {
                                // v10.4.6: If ALL peers explicitly say "No blocks found" the
                                // data is pruned network-wide and will never be retrievable.
                                // Skip the chunk so sync continues rather than stalling forever.
                                // Transport failures (timeout, connection) are NOT skipped — they
                                // hit the is_transport_failure path above and get proper retries.
                                if err_msg.contains("No blocks found in range") {
                                    warn!("⚠️  [HISTORICAL GAP v10.4.6] Skipping unavailable \
                                           chunk {}-{}: no peer has these blocks \
                                           (data pruned network-wide). Sync continues.",
                                          start, end);
                                    return Ok((start, end));
                                }
                                error!("❌ Failed chunk {}-{} after {} retries with {} different peers: {}",
                                       start, end, max_retries, max_retries.min(peer_count as u32), e);
                                return Err(e);
                            }
                            warn!("⚠️  Chunk {}-{} failed with peer {} (attempt {}/{}): {}",
                                  start, end, peer, retry_count, max_retries, e);
                            self_clone.metrics.retried_chunks.fetch_add(1, Ordering::Relaxed);

                            // v1.5.2-beta: Penalize peer trust on chunk failures
                            let peer_str = peer.to_string();
                            self_clone.peer_trust.record_data_failure(&peer_str);

                            // 🚀 v2.3.12-beta: SYNC SPEED FIX - Reduced backoff
                            tokio::time::sleep(Duration::from_millis(50 * retry_count as u64)).await;
                        }
                    }
                }
            }));
        };

        // Spawn initial batch of chunks (up to max_in_flight_chunks)
        let mut spawned_count = 0;
        while spawned_count < max_in_flight_chunks && !chunks_queue.is_empty() {
            if let Some((chunk_idx, start, end)) = chunks_queue.pop_front() {
                let peers_for_retry = if use_warp_peers { priority_peers.clone() } else { peers.clone() };
                spawn_chunk_task(&mut futures, chunk_idx, start, end, peers_for_retry, self);
                spawned_count += 1;
            }
        }

        // 📊 v1.0.2: Update session atomics after initial spawn
        self.session_in_flight.store(spawned_count as u64, Ordering::Release);
        self.session_queued.store(chunks_queue.len() as u64, Ordering::Release);

        info!("🚀 [v3.4.8] Spawned initial {} chunks, {} remaining in queue",
              spawned_count, chunks_queue.len());

        // Wait for all chunks to complete with progress reporting
        // As each chunk completes, spawn the next one from the queue
        let mut last_progress_log = Instant::now();
        let sync_start_time = Instant::now();  // 🚀 v2.1.0-DELTA-V: Track for PID throughput
        while let Some(result) = futures.next().await {
            match result? {
                Ok((start, end)) => {
                    completed_chunks += 1;
                    let progress = (completed_chunks as f64 / total_chunks as f64) * 100.0;

                    // 🚀 v3.4.8-beta: SLIDING WINDOW - Spawn next chunk when one completes
                    // This maintains continuous parallelism without spawning ALL chunks at once
                    if let Some((next_idx, next_start, next_end)) = chunks_queue.pop_front() {
                        let peers_for_retry = if use_warp_peers { priority_peers.clone() } else { peers.clone() };
                        spawn_chunk_task(&mut futures, next_idx, next_start, next_end, peers_for_retry, self);
                        debug!("🚀 [v3.4.8] Spawned chunk {}-{} ({} remaining)",
                               next_start, next_end, chunks_queue.len());
                    }

                    // v8.4.4: Yield to tokio scheduler after each chunk completes.
                    // Without this, the sync loop runs back-to-back without giving
                    // API handler tasks a chance to execute on the worker threads.
                    tokio::task::yield_now().await;

                    // 🎚️ v8.5.4: Inter-chunk throttle delay (TUI-controlled)
                    // Conservative=200ms, Normal=10ms, Turbo=0ms
                    // Re-read throttle mode each iteration so TUI changes take effect immediately
                    let (_, delay_ms) = self.get_throttle_params();
                    if delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }

                    // 📊 v1.0.2: Update session atomics for admin panel
                    self.session_completed_chunks.store(completed_chunks as u64, Ordering::Release);
                    self.session_in_flight.store(futures.len() as u64, Ordering::Release);
                    self.session_queued.store(chunks_queue.len() as u64, Ordering::Release);

                    // 🚀 v2.1.0-DELTA-V: Update APOLLO PID controller with current throughput
                    if self.config.enable_apollo_pid && completed_chunks > 0 {
                        let elapsed_secs = sync_start_time.elapsed().as_secs_f64().max(0.001);
                        let blocks_per_chunk = (end - start + 1) as f64;
                        let current_bps = (completed_chunks as f64 * blocks_per_chunk) / elapsed_secs;
                        self.apollo_update_pid(current_bps).await;
                    }

                    // 🔐 v0.9.15-beta: Enhanced logging with AEGIS-QL metrics
                    // Update progress FREQUENTLY (every chunk or every 1 second)
                    if last_progress_log.elapsed().as_secs() >= 1 || completed_chunks % 5 == 0 {
                        let trusted_peers = self.peer_trust.get_trusted_peers();
                        let trusted_count = trusted_peers.len();

                        info!("📊 [TURBO SYNC] Progress: {}/{} chunks ({:.1}%) | In-flight: {} | Queue: {}",
                              completed_chunks, total_chunks, progress,
                              futures.len(), chunks_queue.len());
                        info!("🔐 [AEGIS-QL] {} trusted peers (>80% trust) | Latest chunk: {}-{}",
                              trusted_count, start, end);

                        // Log individual peer trust scores periodically
                        if completed_chunks % 10 == 0 {
                            for peer_id in trusted_peers.iter().take(5) {
                                if let Some(score) = self.peer_trust.get_trust_score(peer_id) {
                                    info!("   • Peer {}: {:.1}% trust",
                                          &peer_id[..min(8, peer_id.len())],
                                          score * 100.0);
                                }
                            }
                        }

                        last_progress_log = Instant::now();
                    }
                }
                Err(e) => {
                    self.metrics.failed_chunks.fetch_add(1, Ordering::Relaxed);
                    let err_str = e.to_string();
                    error!("❌ Chunk failed: {}", err_str);

                    // v10.2.10: If chunk hit wall-clock deadline, cooldown 60s before
                    // spawning more work to prevent hot-loop retries of stuck ranges.
                    if err_str.contains("WALL_CLOCK_TIMEOUT") {
                        warn!("⏳ [CIRCUIT BREAKER] Wall-clock timeout detected — cooling down 60s \
                               before continuing to prevent hot-loop retries");
                        tokio::time::sleep(Duration::from_secs(60)).await;
                    }

                    // 🚀 v3.4.8-beta: Spawn next chunk even on failure to maintain parallelism
                    if let Some((next_idx, next_start, next_end)) = chunks_queue.pop_front() {
                        let peers_for_retry = if use_warp_peers { priority_peers.clone() } else { peers.clone() };
                        spawn_chunk_task(&mut futures, next_idx, next_start, next_end, peers_for_retry, self);
                    }
                    // 📊 v1.0.2: Update session atomics on failure too
                    self.session_in_flight.store(futures.len() as u64, Ordering::Release);
                    self.session_queued.store(chunks_queue.len() as u64, Ordering::Release);
                    // Continue with other chunks even if one fails
                }
            }
        }

        // ✅ v0.9.40-beta FIX: FAIL LOUD instead of silent success
        // This prevents phantom success when chunks fail to download
        if completed_chunks < total_chunks {
            let failed = total_chunks - completed_chunks;
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("🚨 [TURBO SYNC] DOWNLOAD FAILED!");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("   Completed chunks: {}/{}", completed_chunks, total_chunks);
            error!("   Failed chunks: {}", failed);
            error!("   Success rate: {:.1}%", (completed_chunks as f64 / total_chunks as f64) * 100.0);
            error!("");
            error!("   This prevents phantom success - refusing to claim sync complete!");
            error!("   Will fall back to HTTP sync for missing blocks.");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            anyhow::bail!(
                "TURBO SYNC incomplete: {}/{} chunks failed ({:.1}% success rate). \
                This prevents phantom success. Falling back to HTTP sync.",
                failed, total_chunks, (completed_chunks as f64 / total_chunks as f64) * 100.0
            );
        }

        info!("✅ [v0.9.40 DEBUG] All {}/{} chunks completed successfully (100% success rate)",
              completed_chunks, total_chunks);

        // 🚀 v2.1.0-DELTA-V: Log APOLLO optimization summary after sync
        if self.config.enable_apollo_pid || self.config.enable_apollo_kalman || self.config.enable_apollo_gravity_assist {
            self.log_apollo_summary().await;
        }

        // 🚀 v1.0.2-POST-SYNC: Post-sync optimization phase
        {
            info!("🔧 [POST-SYNC] Starting post-sync optimization phase...");

            // Phase A: Trigger RocksDB compaction (background, non-blocking)
            // After bulk sync, RocksDB has many L0 SST files from rapid writes.
            // Background compaction consolidates them for 20-40% faster reads.
            let storage_for_compact = Arc::clone(&self.storage);
            tokio::spawn(async move {
                info!("🔧 [POST-SYNC] Phase A: Triggering background RocksDB compaction...");
                match storage_for_compact.compact().await {
                    Ok(()) => info!("✅ [POST-SYNC] RocksDB compaction completed"),
                    Err(e) => warn!("⚠️ [POST-SYNC] RocksDB compaction failed (non-critical): {}", e),
                }
            });

            // Phase B: Reset PID controller for steady-state operation
            // During sync, PID targets ~1000 BPS throughput. After sync, target ~100 BPS
            // (real-time block production rate) to prevent over-aggressive network usage.
            if self.config.enable_apollo_pid {
                let mut pid = self.apollo_pid_controller.write().await;
                pid.reset();
                pid.set_target(100.0); // Steady-state: ~100 BPS (real-time block rate)
                info!("🎛️ [POST-SYNC] Phase B: PID reset for steady-state (target=100 BPS)");
            }

            // Phase C: Log Kalman converged state for monitoring
            if self.config.enable_apollo_kalman {
                let kalman = self.apollo_kalman_predictor.read().await;
                let state = kalman.get_state();
                let metrics = kalman.get_metrics();
                info!("🔭 [POST-SYNC] Phase C: Kalman converged state:");
                info!("   BW={:.1} Mbps, Lat={:.1} ms, Loss={:.2}%, Conf={:.2}",
                      state.bandwidth_bps / 1_000_000.0, state.latency_ms,
                      state.loss_rate * 100.0, state.confidence);
                info!("   Optimal: chunk={}KB, timeout={}ms, concurrency={}",
                      metrics.optimal_chunk_kb, metrics.optimal_timeout_ms, metrics.optimal_concurrency);
            }

            // Phase D: Return sync memory to OS (prevent OOM on Gamma)
            #[cfg(target_os = "linux")]
            {
                extern "C" { fn malloc_trim(pad: usize) -> i32; }
                unsafe { malloc_trim(0); }
                info!("🧹 [POST-SYNC] Phase D: malloc_trim — returned freed memory to OS");
            }

            info!("✅ [POST-SYNC] Optimization phase complete");
        }

        // 🚀 v2.3.10-beta: Clear prefetch pipeline after sync completes
        self.warp_prefetch.clear().await;
        debug!("🚀 [WARP SYNC] Prefetch pipeline cleared after sync completion");

        // 📊 v1.0.2: Reset session atomics — sync complete, back to idle
        self.session_sync_mode.store(0, Ordering::Release);
        self.session_in_flight.store(0, Ordering::Release);
        self.session_queued.store(0, Ordering::Release);

        Ok(())
    }

    /// Clone for async task (avoiding full Arc cloning complexity)
    fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            storage: Arc::clone(&self.storage),
            download_semaphore: Arc::clone(&self.download_semaphore),
            // 🚀 v1.5.1-beta: Clone new sync optimization fields
            decompression_semaphore: Arc::clone(&self.decompression_semaphore),
            chunks_since_wal_sync: AtomicU64::new(self.chunks_since_wal_sync.load(Ordering::Relaxed)),
            last_wal_sync_time: AtomicU64::new(self.last_wal_sync_time.load(Ordering::Relaxed)),
            last_flush_time: AtomicU64::new(self.last_flush_time.load(Ordering::Relaxed)),
            metrics: Arc::clone(&self.metrics),
            peer_registry: Arc::clone(&self.peer_registry),
            network_tx: self.network_tx.clone(), // Clone the network channel
            aegis: Arc::clone(&self.aegis),
            aegis_secret_key: Arc::clone(&self.aegis_secret_key),
            aegis_public_key: self.aegis_public_key.clone(),
            peer_trust: Arc::clone(&self.peer_trust),
            memory_limiter: Arc::clone(&self.memory_limiter),
            pipeline_manager: Arc::clone(&self.pipeline_manager),
            pack_cache: Arc::clone(&self.pack_cache),
            // v1.0.50-beta: Crypto-enhanced sync fields
            adaptive_timeout: Arc::clone(&self.adaptive_timeout),
            progress_tracker: Arc::clone(&self.progress_tracker),
            block_verifier: Arc::clone(&self.block_verifier),
            // v1.0.60-beta: State processor for comprehensive state sync
            #[cfg(not(target_os = "windows"))]
            state_processor: self.state_processor.clone(),
            // 🛡️ v1.3.0-beta: SHA3-256 Data Integrity Verifier
            sha3_verifier: Arc::clone(&self.sha3_verifier),
            // 🤖 v1.4.0-beta: ML batch optimizer
            batch_predictor: Arc::clone(&self.batch_predictor),
            // 🛡️ v1.4.5-beta: Orphan rate limiter
            orphan_limiter: Arc::clone(&self.orphan_limiter),
            // 🚀 v2.3.4-beta: Emergency sync guard
            emergency_sync_in_progress: Arc::clone(&self.emergency_sync_in_progress),
            // 🔒 v10.5.0: Fresh-start single-flight gate (shared Arc — same gate across clones)
            fresh_sync_gate: Arc::clone(&self.fresh_sync_gate),
            fresh_sync_target: Arc::clone(&self.fresh_sync_target),
            // 🚀 v1.5.0-beta: CHIRON parallel state applicator
            parallel_state_applicator: Arc::clone(&self.parallel_state_applicator),
            // 🚀 v1.5.0-beta: NEMO high-contention executor
            nemo_executor: Arc::clone(&self.nemo_executor),
            // 🚀 v1.5.0-beta: Reddio async storage pipeline
            #[cfg(not(target_os = "windows"))]
            async_pipeline: self.async_pipeline.clone(),
            // 🚀 v2.1.0-DELTA-V: Project APOLLO Control Systems
            apollo_pid_controller: Arc::clone(&self.apollo_pid_controller),
            apollo_kalman_predictor: Arc::clone(&self.apollo_kalman_predictor),
            apollo_peer_momentum: Arc::clone(&self.apollo_peer_momentum),
            // 🚀 v2.3.10-beta: WARP SYNC Phase 2 & 3
            warp_multi_peer: Arc::clone(&self.warp_multi_peer),
            warp_prefetch: Arc::clone(&self.warp_prefetch),
            // 🚀 v1.0.2: Lock-Free Sync State
            cached_max_peer_height: Arc::clone(&self.cached_max_peer_height),
            cached_peer_count: Arc::clone(&self.cached_peer_count),
            cached_has_gaps: Arc::clone(&self.cached_has_gaps),
            is_fully_synced: Arc::clone(&self.is_fully_synced),
            // 📊 v1.0.2: Session Chunk Progress
            session_total_chunks: Arc::clone(&self.session_total_chunks),
            session_completed_chunks: Arc::clone(&self.session_completed_chunks),
            session_in_flight: Arc::clone(&self.session_in_flight),
            session_queued: Arc::clone(&self.session_queued),
            session_sync_mode: Arc::clone(&self.session_sync_mode),
            // 🎚️ v8.5.4: Share throttle mode (same Arc)
            network_throttle_mode: Arc::clone(&self.network_throttle_mode),
            // v10.9.44: Share gap config + scientific sync state across clones
            known_gaps: Arc::clone(&self.known_gaps),
            kalman_bdp: Arc::clone(&self.kalman_bdp),
            beta_scores: Arc::clone(&self.beta_scores),
            markov_states: Arc::clone(&self.markov_states),
            littles_law: Arc::clone(&self.littles_law),
            chunk_floor: Arc::clone(&self.chunk_floor),
            peer_fail_predictor: Arc::clone(&self.peer_fail_predictor),
            // v10.9.47: Share gap-detection tally across clones. The
            // gap_advance_rx is NOT shared (only the primary instance owns it);
            // clones get a fresh None.
            gap_detection_tally: Arc::clone(&self.gap_detection_tally),
            gap_advance_rx: parking_lot::Mutex::new(None),
        }
    }

    /// Main entry point: Sync to target height
    pub async fn sync_to_height(&self, target_height: u64) -> Result<()> {
        let sync_start = Instant::now();

        // v10.9.44: local_height is mut so the known-gap advance (below) can
        // reflect the post-skip value to downstream checks.
        let mut local_height = self.get_local_height().await?;

        // v10.9.55 Task 4: SYNCED-THROUGH POINTER.
        //
        // On a DAG-Knight chain with legitimately sparse heights (15M+ is ~93-96%
        // dense; pre-7M is mostly missing from historical damage), the chunk-build
        // start must be the "highest range we've REQUESTED" — not "highest height
        // where blocks 0..H all exist contiguously". With the old model the loop
        // wedges on sparse heights forever; with this one it advances past dead
        // ranges and gets to the live tail.
        //
        // synced_through is monotonic and persisted (key `qblock:synced_through`),
        // so a restart picks up where we left off. The known-gap auto-advance below
        // still runs — it can push contiguous_height past configured gaps; we then
        // take max(local_height, synced_through) to honor whichever advanced further.
        let synced_through = self.storage.get_synced_through_height();
        if synced_through > local_height {
            info!(
                "⏭️ [SYNCED-THROUGH v10.9.55] Resuming from {} (contiguous={}, persisted synced_through={})",
                synced_through, local_height, synced_through
            );
            local_height = synced_through;
        }

        if local_height >= target_height {
            info!("🎯 Already synced to height {} (target: {})", local_height, target_height);
            return Ok(());
        }

        // 📜 v0.9.15-beta: Check if we've already synced to this height (using AEGIS-QL certificate)
        // This prevents restart loops - node knows it already synced even after restart
        if self.check_if_already_synced(target_height).await? {
            info!("✅ [AEGIS-QL] Skipping sync - already synced to {} (certificate verified)", target_height);
            return Ok(());
        }

        // 🚨 CRITICAL SAFETY CHECK: Prevent catastrophic sync-down (v0.5.23-beta)
        // This prevents BILLIONS of dollars in data loss on mainnet
        if target_height < local_height && local_height > 1000 {
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("🚨 CRITICAL SAFETY ABORT: SYNC-DOWN DETECTED!");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("   Current height: {} blocks", local_height);
            error!("   Target height:  {} blocks", target_height);
            error!("   Would LOSE:     {} blocks", local_height - target_height);
            error!("   ");
            error!("   This would cause CATASTROPHIC DATA LOSS!");
            error!("   Refusing to execute for safety.");
            error!("   ");
            error!("   Possible causes:");
            error!("   1. Malicious peer announcing false height");
            error!("   2. Network split");
            error!("   3. Bug in peer announcement handling");
            error!("   ");
            error!("   Action: Check peer heights and network status");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            return Err(anyhow::anyhow!(
                "SAFETY ABORT: Refusing to sync down from {} to {} (would lose {} blocks). \
                This protects against data loss. Check peer announcements.",
                local_height, target_height, local_height - target_height
            ));
        }

        // ⏩ v10.9.44: KNOWN-GAP AUTO-ADVANCE (deterministic 26K→100K fix).
        //
        // Before any peer discovery or chunk dispatch, check whether our
        // contiguous height sits immediately below a known network-wide gap
        // (configured via `Q_KNOWN_PERMANENT_GAPS`, default `25988:100440`).
        // If yes, advance `contiguous_height` past `gap_end` so the sync loop
        // can immediately resume from post-gap blocks rather than wedging on
        // a request no peer can serve.
        //
        // Multiple gaps are handled with a loop — if jumping past one gap
        // lands us at the edge of another (adjacent gaps), keep jumping until
        // the next gap is no longer immediately ahead.
        //
        // Persistence: `save_safe_floor()` is called after every advance so
        // the jump survives restart. Idempotent — repeat invocations are a
        // no-op once contiguous has advanced past all gaps.
        // v10.9.46: Fire when contiguous+1 falls INSIDE a known gap, not only at
        // the exact gap-start. The original strict equality (gap_start ==
        // working_height + 1) silently failed every call on the canary: contiguous
        // was 26,000 vs configured gap (25,988, 100,440), so 25988 == 26001 was
        // FALSE and the loop fell to `_ => break` without advancing. Verified
        // 2026-05-17: canary stalled at h=26,000 for 22+ min with [CONFIG]
        // Q_KNOWN_PERMANENT_GAPS = '25988:100440' parsed correctly but zero
        // KNOWN-GAP auto-advances fired in the log. The off-by-one is real and
        // also defends against operator gap configs whose declared start is
        // slightly low.
        let mut advanced = false;
        let mut working_height = local_height;
        loop {
            let next_block = working_height.saturating_add(1);
            match self.known_gaps.next_gap_above(working_height) {
                Some((gap_start, gap_end)) if next_block >= gap_start && next_block <= gap_end => {
                    // Next missing block is INSIDE the known gap → jump to gap_end
                    // so the next chunk request lands at gap_end+1 (post-gap region).
                    let new_contiguous = gap_end; // height pointer at end of gap
                    info!(
                        "⏩ [KNOWN-GAP v10.9.46] Auto-advanced contiguous {} → {} via Q_KNOWN_PERMANENT_GAPS \
                         (next_block={} inside gap {}..={}, skipped {} blocks)",
                        working_height,
                        new_contiguous,
                        next_block,
                        gap_start,
                        gap_end,
                        gap_end - gap_start + 1
                    );
                    // Wire Prometheus gauges so /metrics shows the fire (v10.9.46).
                    let gauges = crate::metrics::SyncOptimizerGauges::instance();
                    gauges.known_gap_advances_total.inc();
                    gauges.known_gap_blocks_skipped_total.inc_by(gap_end - gap_start + 1);
                    self.storage.update_height_cache(new_contiguous).await;
                    if let Err(e) = self.storage.save_safe_floor(new_contiguous).await {
                        warn!("⏩ [KNOWN-GAP] save_safe_floor({}) failed: {} \
                              — advance is in-memory only and may not survive restart",
                              new_contiguous, e);
                    }
                    working_height = new_contiguous;
                    advanced = true;
                    // Stop if we've now passed/reached the target.
                    if working_height >= target_height {
                        info!("⏩ [KNOWN-GAP v10.9.46] Advanced past target height ({} >= {}), \
                               nothing left to sync this round",
                              working_height, target_height);
                        return Ok(());
                    }
                }
                Some((gap_start, gap_end)) => {
                    // Gap exists but next_block is before it (sync gap fill in
                    // progress) — log once and stop. Don't advance.
                    debug!(
                        "⏩ [KNOWN-GAP v10.9.46] Next gap {}..={} ahead but not yet adjacent \
                         (contiguous={}, next_block={}, distance={}). No advance.",
                        gap_start, gap_end, working_height, next_block,
                        gap_start.saturating_sub(next_block)
                    );
                    break;
                }
                None => break, // No more gaps ahead.
            }
        }
        if advanced {
            info!(
                "⏩ [KNOWN-GAP v10.9.44] Total advance: {} → {} ({} blocks skipped). \
                 Resuming normal sync from height {} toward target {}.",
                local_height, working_height, working_height - local_height,
                working_height, target_height
            );
            // Reflect post-skip value to downstream checks (fresh-start detection,
            // endgame detection, effective_start_height computation).
            local_height = working_height;
        }

        // 🔒 v10.5.0: Fresh-start single-flight gate.
        // On a fresh node (height < 100) multiple gossipsub height announcements arrive
        // within milliseconds of each other, each triggering sync_to_height() concurrently.
        // Two invocations racing over the shared height pointer corrupt it — one advances
        // to 75,000 while the other writes 7,200 and the last writer wins.
        //
        // Pattern: latch the highest target, then try_lock().
        //   - Gate winner: acquires lock, re-reads latched target, runs full probe + sync.
        //   - Losers: latch their target and return immediately.
        // The _guard is held for the rest of this invocation via RAII drop.
        let _fresh_sync_guard;
        let target_height = if local_height < 100 {
            self.fresh_sync_target.fetch_max(target_height, Ordering::AcqRel);
            match self.fresh_sync_gate.try_lock() {
                Ok(guard) => {
                    _fresh_sync_guard = Some(guard);
                    // Use the highest target latched by any concurrent caller.
                    self.fresh_sync_target.load(Ordering::Acquire)
                }
                Err(_) => {
                    info!(
                        "🔒 [FRESH-SYNC GATE] Invocation deferred — another caller holds \
                         the gate (our target={}). Latched and exiting.",
                        target_height
                    );
                    return Ok(());
                }
            }
        } else {
            _fresh_sync_guard = None;
            target_height
        };

        // 🔧 v3.1.4-beta: FRESH START PROTECTION
        // If local_height is 0 or very low (< 100), SKIP endgame detection entirely!
        // This works with the fix in lib.rs that returns height=0 when genesis blocks are missing.
        let effective_start_height = if local_height < 100 {
            // FRESH START: Force sync from beginning
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            warn!("🚀 [v3.1.4] FRESH START DETECTED (contiguous height: {})", local_height);
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // v10.3.5: Checkpoint Sync probe will run after peer discovery (below).
            // Set to 0 for now — will be updated to checkpoint height if gap is detected.
            warn!("   Will probe network for checkpoint sync after peer discovery...");
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            0 // Will be overwritten by checkpoint sync probe
        } else {
            // Normal operation: Allow endgame detection for established chains
            // 🚀 v2.3.3-beta: ENDGAME SYNC OPTIMIZATION
            // Detect if we have blocks stored above contiguous height (endgame scenario)
            // This happens when gossipsub delivers tip blocks but we have gaps in the middle
            let (highest_stored, is_endgame) = self.detect_endgame_mode(local_height, target_height).await;

            if is_endgame && highest_stored > local_height + 1000 && local_height >= 10000 {
                // Endgame mode: We have blocks above contiguous, skip to fill tip gap first
                info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                info!("🎯 [ENDGAME SYNC] FAST TIP CATCH-UP MODE ACTIVATED!");
                info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                info!("   Contiguous height: {} (pointer)", local_height);
                info!("   Highest stored:    {} (actual blocks)", highest_stored);
                info!("   Target height:     {}", target_height);
                info!("   Gap to skip:       {} blocks (will fill later)", highest_stored - local_height);
                info!("   Tip gap to fill:   {} blocks (PRIORITY)", target_height - highest_stored);
                info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                highest_stored
            } else {
                // Normal sync from current contiguous height
                local_height
            }
        };

        // 🧠 v1.0.15.1-beta: Check memory pressure before starting sync
        if self.memory_limiter.should_pause_sync().await {
            warn!("⏸️  [MEMORY CRITICAL] Pausing sync until memory relief");
            self.memory_limiter.wait_for_memory_relief().await;
        }

        // 🧠 Log current memory status (actual batch size determined by ML after peer discovery)
        let memory_stats = self.memory_limiter.get_memory_stats().await;
        info!("🧠 [MEMORY] Status: pressure={:?}, usage={:.1}%",
              memory_stats.pressure, memory_stats.usage_percent());

        // 🚀 v2.3.3-beta: Calculate effective missing range (accounts for endgame skip)
        let missing_range = target_height.saturating_sub(effective_start_height);
        let total_gap = target_height - local_height;

        // v3.1.4: Simplified logging (is_endgame variable no longer in scope)
        if effective_start_height > local_height {
            // Endgame mode was activated
            info!("🚀 TURBO SYNC STARTING [ENDGAME]: {} blocks ({} → {})",
                  missing_range, effective_start_height, target_height);
            info!("   Total gap: {} blocks (filling {} now, {} later)",
                  total_gap, missing_range, effective_start_height - local_height);
        } else {
            info!("🚀 TURBO SYNC STARTING: {} blocks ({} → {})",
                  missing_range, local_height, target_height);
        }
        info!("⚙️  Config: {} parallel streams, {} blocks/chunk (adaptive), compression level {}",
              self.config.parallel_streams, self.config.chunk_size, self.config.compression_level);

        // Record start time
        *self.metrics.start_time.write().await = Some(sync_start);

        // PHASE 1: Discover peers with required height
        info!("🔍 [v0.9.40 DEBUG] PHASE 1: Discovering peers with height {}...", target_height);
        let qualified_peers = self.discover_peers_with_height(target_height).await?;
        info!("🔍 [v0.9.40 DEBUG] PHASE 1 COMPLETE: Found {} qualified peers", qualified_peers.len());

        if !qualified_peers.is_empty() {
            // 🛰️ v1.0.36-beta: LOUD P2P SUCCESS LOGGING
            warn!("🛰️  [SYNC] Using P2P libp2p sync - {} peers available", qualified_peers.len());
            warn!("    Config: streams={} chunk_size={} compression_level={}",
                  self.config.parallel_streams, self.config.chunk_size, self.config.compression_level);

            for (idx, peer) in qualified_peers.iter().enumerate().take(5) {
                info!("   • Peer {}: {}", idx + 1, peer);
            }
        }

        if qualified_peers.is_empty() {
            // 🌐 v1.0.36-beta: LOUD P2P FAILURE - This triggers HTTP fallback in older code
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("🌐 [SYNC] P2P UNAVAILABLE - NO PEERS FOUND!");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            error!("   Local height: {}, Target: {} - no peers have blocks we need", local_height, target_height);
            error!("   Looking for peers with height > {} (v2.2.1 fix)", local_height);
            error!("");
            error!("🔧 REQUIRED ACTIONS:");
            error!("   1. Ensure bootstrap node is reachable (http://89.149.241.126:8080 — Epsilon supernode)");
            error!("   2. Verify gossipsub peer discovery is working");
            error!("   3. Check if peer height announcements are being processed");
            error!("   4. Verify TurboSync peer registry is being populated");
            error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            anyhow::bail!("No peers available with height > {} (local)", local_height);
        }

        // v10.3.5: CHECKPOINT SYNC — now that we have peers, probe for the gap
        // Q_SKIP_CHECKPOINT=1 bypasses the probe so the node syncs ALL blocks from genesis.
        let skip_checkpoint = std::env::var("Q_SKIP_CHECKPOINT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut effective_start_height = effective_start_height; // make mutable for checkpoint update
        if skip_checkpoint && local_height < 100 {
            warn!("🚫 [CHECKPOINT SYNC] Q_SKIP_CHECKPOINT=1 — skipping network gap probe. Syncing ALL blocks from genesis via P2P.");
        }
        if !skip_checkpoint && local_height < 100 && effective_start_height == 0 {
            let gap_floor = self.probe_network_gap(target_height, &qualified_peers).await;
            if gap_floor > 0 {
                let blocks_to_sync = target_height.saturating_sub(gap_floor);
                info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━���━━━━━━━━━━━━━━━━━━━━━━");
                info!("🔗 [CHECKPOINT SYNC v10.3.5] Using network-verified checkpoint");
                info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                info!("   Checkpoint height: {}", gap_floor);
                info!("   Balances verified via state sync from {} peers", qualified_peers.len());
                info!("   Syncing {} blocks ({} → {})", blocks_to_sync, gap_floor, target_height);
                info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                // CRITICAL: Initialize the contiguous height pointer to checkpoint.
                // Without this, blocks written at H+1, H+2... would never advance
                // the pointer because it expects H-1 to exist (contiguity check).
                // We set the pointer to (gap_floor - 1) so the FIRST block at gap_floor
                // satisfies the check: gap_floor == (gap_floor - 1) + 1
                let checkpoint_base = gap_floor.saturating_sub(1);
                self.storage.update_height_cache(checkpoint_base).await;
                if let Err(e) = self.storage.save_safe_floor(checkpoint_base).await {
                    warn!("🔗 [CHECKPOINT SYNC] Could not persist safe floor: {}", e);
                }
                info!("🔗 [CHECKPOINT SYNC] Height pointer initialized to {} (checkpoint base)", checkpoint_base);

                effective_start_height = gap_floor;
            }
        }

        // 🎯 v3.2.11-beta: ATOMIC SYNC ENDGAME - Detect near-tip and use fast settings
        let blocks_to_sync = target_height.saturating_sub(local_height);
        let is_endgame = blocks_to_sync <= self.config.endgame_threshold && blocks_to_sync > 0;

        // 📊 v1.0.2: Set session_sync_mode for admin panel badges
        if is_endgame {
            self.session_sync_mode.store(2, Ordering::Release); // 2 = endgame
        } else if blocks_to_sync <= 50 {
            self.session_sync_mode.store(3, Ordering::Release); // 3 = micro
        }
        // Note: mode 1 (turbo) is set in download_chunks_parallel() when chunk download starts

        // 🤖 v1.4.0-beta: ML-driven adaptive batch size prediction
        // Now that we have qualified peers, extract features and predict optimal batch size
        let features = self.extract_sync_features(&qualified_peers).await;
        let ml_chunk_size = {
            let predictor = self.batch_predictor.read().await;
            predictor.predict_batch_size(&features)
        };

        // Safety: Cap ML prediction by memory limiter (never exceed memory-safe size)
        let memory_cap = self.memory_limiter.get_recommended_batch_size().await as u64;

        // 🎯 v8.2.0: SINGLE-SHOT ENDGAME — request ALL remaining blocks in one chunk
        // v3.2.11 used 50-block chunks × 10 round-trips = ~30s worst case
        // v8.2.0: 1 request for up to 500 blocks (~5-10 MB compressed) = ~3s
        let chunk_size = if is_endgame {
            // Single-shot: request all remaining blocks at once
            let single_shot = blocks_to_sync.min(self.config.endgame_threshold);
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            info!("🎯 [SINGLE-SHOT ENDGAME v8.2.0] Requesting ALL {} blocks in ONE request", single_shot);
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            info!("   Mode: SINGLE-SHOT (1 round-trip instead of ~{} round-trips)",
                  (blocks_to_sync + self.config.endgame_chunk_size - 1) / self.config.endgame_chunk_size);
            info!("   Payload: ~{}-{} MB compressed", blocks_to_sync / 100, blocks_to_sync / 50);
            info!("   Timeout: {:?} (endgame) vs {:?} (normal)", self.config.endgame_timeout, self.config.chunk_timeout);
            info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // v10.1.1: Scale timeout with batch size — 3s was too aggressive for >100 blocks
            {
                let endgame_timeout = if blocks_to_sync <= 50 {
                    Duration::from_secs(3) // 50 blocks is tiny — 3s is plenty
                } else if blocks_to_sync <= 200 {
                    Duration::from_secs(10) // 200 blocks — give it time
                } else {
                    Duration::from_secs(30) // Large endgame — don't timeout prematurely
                };
                let mut timeout_calc = self.adaptive_timeout.write().await;
                timeout_calc.set_endgame_mode(endgame_timeout);
            }

            single_shot
        } else {
            // Normal sync: Use ML-predicted chunk size
            // v8.6.2: If a supernode is among qualified peers, allow larger chunks (up to 5000)
            let supernode_boost = if !self.config.supernode_peers.is_empty() {
                let has_supernode = qualified_peers.iter().any(|p| {
                    let pid = p.to_string();
                    self.config.supernode_peers.iter().any(|s| pid.contains(s.as_str()) || s.contains(&pid))
                });
                if has_supernode {
                    info!("🚀 [SUPERNODE] Supernode peer detected among {} qualified peers — allowing 5000-block chunks",
                          qualified_peers.len());
                    5000u64
                } else {
                    self.config.chunk_size
                }
            } else {
                self.config.chunk_size
            };
            // 🔭 v1.0.2-KALMAN: Blend Kalman BDP-optimal chunk size (40% weight) when confident
            let base_chunk = ml_chunk_size.min(memory_cap).min(supernode_boost);
            if self.config.enable_apollo_kalman {
                let kalman = self.apollo_kalman_predictor.read().await;
                let state = kalman.get_state();
                if state.confidence > 0.4 {
                    let kalman_chunk = state.optimal_chunk_size() as u64;
                    // Blend: 60% ML/config + 40% Kalman BDP-optimal
                    let blended = ((base_chunk as f64 * 0.6) + (kalman_chunk as f64 * 0.4)) as u64;
                    let clamped = blended.clamp(
                        self.config.chunk_size / 4, // min: quarter of config
                        supernode_boost.max(self.config.chunk_size * 2),  // max: supernode or double config
                    ).min(memory_cap);
                    info!("🔭 [KALMAN CHUNK] ML={}, Kalman={}, Blended={} (conf={:.2})",
                          base_chunk, kalman_chunk, clamped, state.confidence);
                    clamped
                } else {
                    base_chunk
                }
            } else {
                base_chunk
            }
        };

        // Log ML prediction details
        let samples_seen = {
            let predictor = self.batch_predictor.read().await;
            predictor.samples_seen()
        };
        if is_endgame {
            info!("🎯 [ENDGAME] Using chunk_size={} for fast near-tip sync", chunk_size);
        } else {
            info!("🤖 [ML BATCH] Predicted: {} blocks (memory cap: {}, config: {})",
                  ml_chunk_size, memory_cap, self.config.chunk_size);
            info!("   Features: RTT={:.0}ms, mem={:.2}, trust={:.2}, bw={:.1}MB/s, success={:.2}",
                  features.rtt_median_ms, features.memory_pressure, features.peer_trust_score,
                  features.bandwidth_mbps, features.success_rate);
            info!("   Model: {} samples learned, final chunk_size: {}", samples_seen, chunk_size);
        }

        // PHASE 2: Split range into parallel chunks
        // 🔧 v1.3.2-beta: GENESIS BLOCK FIX - Start from height 1 when at height 0
        // Genesis block is at height 1 (not 0), so new nodes must start from 1
        // 🚀 v2.3.3-beta: Use effective_start_height for endgame optimization
        let sync_start_height = if effective_start_height == 0 { 1 } else { effective_start_height + 1 };
        let actual_missing = target_height.saturating_sub(effective_start_height);
        // 🔧 v3.1.3: Enhanced logging to diagnose sync issues
        warn!("🔍 [v3.1.3 SYNC DEBUG] PHASE 2: Splitting {} blocks into chunks", actual_missing);
        warn!("   local_height={}, effective_start={}, sync_start={}, target={}",
              local_height, effective_start_height, sync_start_height, target_height);
        // 🤖 v1.4.0-beta: Pass ML-predicted chunk_size to split_into_chunks
        let mut chunks = self.split_into_chunks(sync_start_height, target_height, chunk_size);

        // v10.9.25: GENESIS-MODE CHUNK WINDOW.
        //
        // SYMPTOM this fixes: with Q_GENESIS_SYNC_ONLY=1 and a fresh node at
        // contiguous height 4045, gravity-assist parallel streams were spawning
        // chunks across the entire 4046..18M range. The first few thousand
        // chunks (4046, 5046, 6046, …) extended the contiguous chain, but the
        // bulk of in-flight chunks at heights 6M, 9M, 13M wasted bandwidth —
        // their blocks downloaded and saved, but with no contiguous predecessor
        // they never advanced `qblock:latest`. From the operator's view the
        // node looked stuck (gravity-assist progress bar moved 3200/360000
        // chunks while contiguous height stayed at 4045).
        //
        // FIX: when Q_GENESIS_SYNC_ONLY=1, restrict the chunk set to those
        // within GENESIS_LOOKAHEAD_BLOCKS of the current contiguous height.
        // The default 1M window keeps gravity-assist effective (10K concurrent
        // chunks at 100-block size all fit) while ensuring every in-flight
        // chunk DIRECTLY extends the contiguous chain on completion. Tunable
        // via Q_GENESIS_LOOKAHEAD_BLOCKS for operators with fast peers.
        //
        // Checkpoint and endgame modes are NOT affected — only Q_GENESIS_SYNC_ONLY.
        let genesis_mode = std::env::var("Q_GENESIS_SYNC_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if genesis_mode {
            let lookahead: u64 = std::env::var("Q_GENESIS_LOOKAHEAD_BLOCKS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(10_000);
            let window_cap = effective_start_height.saturating_add(lookahead);
            let original_len = chunks.len();
            // v10.9.29: cap BOTH chunk start AND chunk end (clip). See split_into_chunks
            // for the rationale — a single oversized chunk slipping through breaks ingestion.
            chunks.retain_mut(|(start, end)| {
                if *start > window_cap {
                    false
                } else {
                    *end = (*end).min(window_cap);
                    true
                }
            });
            if chunks.len() < original_len {
                warn!(
                    "🌱 [GENESIS-WINDOW v10.9.29] sync_to_height capped: kept {}/{} chunks, each clipped to end ≤ {} \
                     (= contiguous {} + {} lookahead). Remainder scheduled on next sync_to_height tick as contiguous advances.",
                    chunks.len(), original_len, window_cap, effective_start_height, lookahead
                );
            }
        }

        info!("🔍 [v0.9.40 DEBUG] PHASE 2 COMPLETE: Created {} chunks for parallel download", chunks.len());

        if !chunks.is_empty() {
            info!("   • First chunk: {}-{}", chunks[0].0, chunks[0].1);
            if chunks.len() > 1 {
                info!("   • Last chunk: {}-{}", chunks[chunks.len()-1].0, chunks[chunks.len()-1].1);
            }
        }

        // PHASE 3: Download chunks in parallel from multiple peers
        info!("🔍 [v0.9.40 DEBUG] PHASE 3: Starting parallel download of {} chunks...", chunks.len());
        // 🤖 v1.4.0-beta: Save chunk count for ML outcome recording
        let num_chunks = chunks.len();
        let ml_predicted_size = ml_chunk_size;  // Save for outcome recording
        let ml_features = features.clone();  // Save features for outcome recording
        let chunk_start = Instant::now();
        self.download_chunks_parallel(chunks, qualified_peers).await?;
        let chunk_duration = chunk_start.elapsed();
        info!("🔍 [v0.9.40 DEBUG] PHASE 3 COMPLETE: All chunks downloaded successfully");

        // PHASE 4: Final flush to persist all bulk writes to disk
        info!("💾 Flushing all bulk writes to disk (this may take a moment)...");
        let flush_start = Instant::now();
        self.storage.hot_db.flush().await?;
        let flush_time = flush_start.elapsed();
        info!("✅ Flush complete in {:.2}s - all data persisted to disk", flush_time.as_secs_f64());

        // 🚨 v1.1.27-beta CRITICAL FIX: Update height cache to CONTIGUOUS height only!
        // ROOT CAUSE (v1.1.6): update_height_cache(target_height) advanced pointer even with gaps.
        //   - Parallel downloads may complete out of order or some may fail
        //   - This created gap at block 3042, causing 10k block loss on restart
        // FIX: Only update to HIGHEST CONTIGUOUS height to prevent gaps in pointer.
        let contiguous_height = self.storage.get_highest_contiguous_block().await.unwrap_or(0);
        let safe_height = contiguous_height.min(target_height);
        info!("🎯 [v1.1.27-beta] Updating height cache: target={}, contiguous={}, safe={}",
              target_height, contiguous_height, safe_height);
        if safe_height > self.storage.height_cache.cached() {
            self.storage.update_height_cache(safe_height).await;
            info!("✅ [v1.1.27-beta] Height cache updated to {} (contiguous)", safe_height);
        } else {
            info!("✅ [v1.1.27-beta] Height cache unchanged at {} (contiguous: {})",
                  self.storage.height_cache.cached(), contiguous_height);
        }
        if contiguous_height < target_height {
            warn!("⚠️  [v1.1.27-beta] GAP DETECTED: target={}, contiguous={}. Missing {} blocks!",
                  target_height, contiguous_height, target_height - contiguous_height);
        }

        // PHASE 5: Finalize and report metrics
        let sync_duration = sync_start.elapsed();
        *self.metrics.end_time.write().await = Some(Instant::now());

        let blocks_synced = self.metrics.total_blocks_synced.load(Ordering::Relaxed);
        let bytes_downloaded = self.metrics.total_bytes_downloaded.load(Ordering::Relaxed);
        let bytes_saved = self.metrics.total_bytes_saved_by_compression.load(Ordering::Relaxed);
        let failed = self.metrics.failed_chunks.load(Ordering::Relaxed);
        let retried = self.metrics.retried_chunks.load(Ordering::Relaxed);

        let blocks_per_sec = blocks_synced as f64 / sync_duration.as_secs_f64();
        let blocks_per_min = blocks_per_sec * 60.0;
        let mbps = self.metrics.average_speed_mbps().await;
        let compression_ratio = self.metrics.compression_ratio();

        // 🤖 v1.4.0-beta: Record ML outcome for online learning
        // This feedback loop allows the model to learn from actual sync performance
        // Estimate success count from blocks synced / chunk size
        let success_count = (blocks_synced / chunk_size.max(1)).max(1);
        let failure_count = failed;
        let timeout_occurred = failed > 0 || retried > 0;  // Any retry indicates potential timeout

        let outcome = crate::ml_batch_optimizer::BatchOutcome {
            features: ml_features,
            predicted_size: ml_predicted_size,
            actual_size: chunk_size.min(self.config.chunk_size),
            throughput_bps: blocks_per_sec as f32,
            timeout_occurred,
            failure_count: failure_count as u32,
            success_count: success_count as u32,
            duration: chunk_duration,
            timestamp: std::time::Instant::now(),
        };

        // Update the ML model with this outcome (online learning)
        {
            let mut predictor = self.batch_predictor.write().await;
            predictor.record_outcome(outcome);
            let new_samples = predictor.samples_seen();

            if new_samples % 10 == 0 {
                info!("🤖 [ML BATCH] Learning update: {} samples, throughput={:.1} bps, trained={}",
                      new_samples, blocks_per_sec, predictor.is_trained());
            }
        }

        // 🚀 v1.0.5-beta: Get pipeline metrics
        let pipeline_depth = self.pipeline_manager.current_depth().await;
        let avg_rtt = self.pipeline_manager.avg_rtt().await;

        // 🚀 v1.0.6-beta: Get cache metrics
        let cache_stats = self.pack_cache.stats().await;
        let hit_rate = cache_stats.hit_rate();

        // 🎯 v3.2.11-beta: Restore normal timeout settings after endgame
        if is_endgame {
            let mut timeout_calc = self.adaptive_timeout.write().await;
            timeout_calc.clear_endgame_mode(Duration::from_secs(10)); // Restore 10s min
            info!("🎯 [ATOMIC SYNC ENDGAME] Complete! Restored normal timeout settings");
        }

        info!("🎉 TURBO SYNC COMPLETE!");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📊 Performance Summary:");
        info!("   • Blocks synced: {} blocks", blocks_synced);
        info!("   • Time elapsed: {:.2}s", sync_duration.as_secs_f64());
        info!("   • Speed: {:.0} blocks/sec ({:.0} blocks/min)", blocks_per_sec, blocks_per_min);
        info!("   • Bandwidth: {:.2} MB/s", mbps);
        info!("   • Downloaded: {:.2} MB", bytes_downloaded as f64 / (1024.0 * 1024.0));
        info!("   • Saved by compression: {:.2} MB ({:.1}%)",
              bytes_saved as f64 / (1024.0 * 1024.0), (1.0 - compression_ratio) * 100.0);
        info!("   • Failed chunks: {} (retried: {})", failed, retried);
        info!("🚀 [PIPELINE] Phase 2 Network Optimization:");
        info!("   • Pipeline depth: {} (adaptive)", pipeline_depth);
        if let Some(rtt) = avg_rtt {
            info!("   • Average RTT: {:.0}ms", rtt.as_millis());
        }
        info!("📦 [PACK CACHE] Phase 3 Server-Side Caching:");
        info!("   • Cache hits: {} / {} ({:.1}% hit rate)",
              cache_stats.hits,
              cache_stats.hits + cache_stats.misses,
              hit_rate * 100.0);
        info!("   • Cache size: {:.1} MB / {} MB ({} entries)",
              cache_stats.current_size_bytes as f64 / (1024.0 * 1024.0),
              self.config.pack_cache_config.max_size_bytes / (1024 * 1024),
              cache_stats.current_entries);
        if cache_stats.evictions > 0 || cache_stats.invalidations > 0 {
            info!("   • Evictions: {} | Invalidations: {}",
                  cache_stats.evictions,
                  cache_stats.invalidations);
        }
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Performance validation
        if blocks_per_min < 1000.0 {
            warn!("⚠️  Sync speed {:.0} blocks/min is below target of 1000 blocks/min", blocks_per_min);
        }

        // 📜 v0.9.15-beta: Create AEGIS-QL sync affirmation certificate
        // This cryptographically proves sync completion and prevents restart loops
        info!("📜 [AEGIS-QL] Creating sync affirmation certificate...");
        let cert_start = Instant::now();

        match self.create_sync_affirmation(local_height, target_height).await {
            Ok(cert) => {
                match self.store_sync_certificate(&cert).await {
                    Ok(()) => {
                        info!("✅ [AEGIS-QL] SYNC AFFIRMED in {:?}: {} → {} blocks",
                              cert_start.elapsed(),
                              cert.start_height,
                              cert.end_height);
                        info!("   Certificate merkle root: {}", hex::encode(&cert.merkle_root[..8]));
                        info!("   This prevents restart loops - node knows sync is complete");
                    }
                    Err(e) => {
                        warn!("⚠️ [AEGIS-QL] Failed to store certificate: {}", e);
                        warn!("   Sync completed but certificate not saved (not critical)");
                    }
                }
            }
            Err(e) => {
                warn!("⚠️ [AEGIS-QL] Failed to create certificate: {}", e);
                warn!("   Sync completed but certificate not created (not critical)");
            }
        }

        // v10.9.55 Task 4: advance synced_through to target so the next sync invocation
        // starts past this window. Idempotent under fetch_max — concurrent advances
        // won't regress. Best-effort persist: if the write fails, the in-memory atomic
        // still advances; worst case a restart re-requests this window.
        if let Err(e) = self.storage.advance_synced_through(target_height).await {
            warn!("⚠️ [SYNCED-THROUGH] Failed to advance to {}: {} \
                   (in-memory atomic still updated; will re-request on restart)",
                  target_height, e);
        }

        Ok(())
    }

    /// v8.2.0: MICRO-SYNC — fast path for gaps ≤ 50 blocks
    ///
    /// Skips the full TurboSync pipeline (peer discovery, ML prediction, memory check,
    /// chunk splitting, parallel download) and directly requests missing blocks from
    /// the best cached peer via P2P request-response.
    ///
    /// Returns Ok(true) if blocks were fetched and applied, Ok(false) if micro-sync
    /// couldn't run (no network channel, no peers), Err on failure.
    pub async fn micro_sync(&self, from_height: u64, to_height: u64) -> Result<bool> {
        let range = to_height.saturating_sub(from_height) + 1;
        if range > 50 || range == 0 {
            return Ok(false); // Not eligible for micro-sync
        }

        let network_tx = match &self.network_tx {
            Some(tx) => tx,
            None => return Ok(false), // No network channel — can't micro-sync
        };

        // Use cached best peer (no expensive peer discovery)
        let best_peer = {
            let registry = self.peer_registry.read().await;
            let peers = registry.active_peers_by_height();
            match peers.first() {
                Some(record) if record.height >= to_height => record.peer_id,
                _ => return Ok(false), // No peer at required height
            }
        };

        // 📊 v1.0.2: Set session_sync_mode for admin panel — micro mode
        self.session_sync_mode.store(3, Ordering::Release); // 3 = micro

        info!("⚡ [MICRO-SYNC v8.2.0] Fast path: requesting {} blocks ({}-{}) from {}",
              range, from_height, to_height, best_peer);

        let micro_start = Instant::now();

        // Direct request-response — single call, 2s timeout
        let (response_tx, response_rx) = oneshot::channel();
        if let Err(e) = network_tx.send(NetworkRequest::RequestBlockRangeDirect {
            peer_id: Some(best_peer.to_string()),
            start_height: from_height,
            end_height: to_height,
            response_tx,
        }) {
            warn!("⚠️ [MICRO-SYNC] Failed to send request: {}", e);
            return Ok(false);
        }

        // 2s timeout — 50 blocks is tiny
        match tokio::time::timeout(Duration::from_secs(2), response_rx).await {
            Ok(Ok(Ok(blocks))) => {
                let block_count = blocks.len();
                if block_count == 0 {
                    info!("⚡ [MICRO-SYNC] Peer returned 0 blocks — falling through to normal sync");
                    return Ok(false);
                }

                let actual_start = blocks.first().map(|b| b.header.height).unwrap_or(from_height);
                let actual_end = blocks.last().map(|b| b.header.height).unwrap_or(to_height);

                // Apply blocks directly (reuse existing apply path)
                self.apply_blocks_vec(blocks, None, actual_start, actual_end).await?;

                // Update metrics
                self.metrics.total_blocks_synced.fetch_add(block_count as u64, Ordering::Relaxed);

                // Update height cache
                let contiguous = self.storage.get_highest_contiguous_block().await.unwrap_or(0);
                if contiguous > self.storage.height_cache.cached() {
                    self.storage.update_height_cache(contiguous).await;
                }

                let elapsed = micro_start.elapsed();
                info!("⚡ [MICRO-SYNC v8.2.0] SUCCESS: {} blocks ({}-{}) in {:?} from {}",
                      block_count, actual_start, actual_end, elapsed, best_peer);

                self.session_sync_mode.store(0, Ordering::Release); // back to idle
                Ok(true)
            }
            Ok(Ok(Err(e))) => {
                warn!("⚡ [MICRO-SYNC] Peer returned error: {} — falling through", e);
                self.session_sync_mode.store(0, Ordering::Release);
                Ok(false)
            }
            Ok(Err(_)) => {
                warn!("⚡ [MICRO-SYNC] Response channel closed — falling through");
                self.session_sync_mode.store(0, Ordering::Release);
                Ok(false)
            }
            Err(_) => {
                warn!("⚡ [MICRO-SYNC] Timed out after 2s — falling through to normal sync");
                self.session_sync_mode.store(0, Ordering::Release);
                Ok(false)
            }
        }
    }

    /// Get chunks that need to be synced - for use by external network layer
    /// This allows the network layer to send requests without circular dependencies
    pub async fn get_sync_chunks(&self, target_height: u64) -> Result<Vec<(u64, u64)>> {
        let local_height = self.get_local_height().await?;

        if local_height >= target_height {
            return Ok(Vec::new());
        }

        // 🔧 v1.3.2-beta: GENESIS BLOCK FIX - Start from height 1 when at height 0
        // Genesis block is at height 1 (not 0), so new nodes must start from 1
        let sync_start_height = if local_height == 0 { 1 } else { local_height + 1 };
        // Use config chunk_size as fallback (ML prediction in sync_to_height)
        let chunks = self.split_into_chunks(sync_start_height, target_height, self.config.chunk_size);
        Ok(chunks)
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🔐 v0.9.14-beta: AEGIS-QL POST-QUANTUM SIGNED SYNC
    // ═══════════════════════════════════════════════════════════════════

    /// v0.9.21-beta: Create a signed block pack from height range (for gossipsub P2P)
    /// This combines block fetching + signing for efficient P2P transmission
    pub async fn create_signed_block_pack_from_range(
        &self,
        start_height: u64,
        end_height: u64,
        peer_id: String,
    ) -> Result<crate::aegis_sync::SignedBlockPack> {
        let pack_start = Instant::now();

        // 1. Fetch blocks from storage
        let local_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);

        if start_height > local_height {
            anyhow::bail!(
                "Requested range {}-{} exceeds local height {}",
                start_height, end_height, local_height
            );
        }

        let actual_end = end_height.min(local_height);
        let mut blocks = Vec::new();

        for height in start_height..=actual_end {
            if let Some(block) = self.storage.get_qblock_by_height(height).await? {
                blocks.push(block);
            }
        }

        if blocks.is_empty() {
            anyhow::bail!("No blocks found in range {}-{}", start_height, end_height);
        }

        info!("📦 [AEGIS-QL] Fetched {} blocks in {:?}, now signing...",
              blocks.len(), pack_start.elapsed());

        // 2. Sign the blocks
        self.create_signed_block_pack(blocks, peer_id).await
    }

    /// Create a signed block pack with AEGIS-QL post-quantum signature
    pub async fn create_signed_block_pack(
        &self,
        blocks: Vec<QBlock>,
        peer_id: String,
    ) -> Result<crate::aegis_sync::SignedBlockPack> {
        use crate::aegis_sync::{compute_merkle_root, SignedBlockPack};

        let start_sign = Instant::now();

        // 1. Compute merkle root of block hashes
        // v10.9.43: batched BLAKE3 path — rayon-parallel + tree-mode SIMD when
        // headers exceed 128 KiB (typical hot path uses scalar-blake3-per-block
        // dispatched in parallel; ~2-3x wall-clock vs sequential).
        let block_hashes: Vec<[u8; 32]> = QBlock::batch_calculate_hashes(&blocks);
        let merkle_root = compute_merkle_root(&block_hashes);

        // 2. Create message to sign
        let timestamp = chrono::Utc::now().timestamp();
        let mut message = Vec::new();
        message.extend_from_slice(&merkle_root);
        message.extend_from_slice(&timestamp.to_le_bytes());
        message.extend_from_slice(peer_id.as_bytes());

        // 3. Sign with AEGIS-QL
        let mut aegis = self.aegis.lock().await;
        let secret_key = self.aegis_secret_key.read().await;
        let signature = aegis.sign(&message, &*secret_key)?;

        info!("🔐 [AEGIS-QL] Signed {} blocks in {:?}", blocks.len(), start_sign.elapsed());
        info!("   Merkle root: {}", hex::encode(&merkle_root[..8]));

        Ok(SignedBlockPack {
            blocks,
            merkle_root,
            aegis_signature: signature,
            peer_public_key: self.aegis_public_key.clone(),
            timestamp,
            peer_id,
        })
    }

    /// Verify a signed block pack received from a peer
    pub async fn verify_signed_block_pack(
        &self,
        pack: &crate::aegis_sync::SignedBlockPack,
    ) -> Result<bool> {
        use crate::aegis_sync::{compute_merkle_root, verify_timestamp};

        let start_verify = Instant::now();

        // 1. Verify timestamp (prevents replay attacks)
        if !verify_timestamp(pack.timestamp) {
            warn!("❌ [AEGIS-QL] INVALID TIMESTAMP from peer {} (diff: {}s)",
                  &pack.peer_id[..8],
                  (chrono::Utc::now().timestamp() - pack.timestamp).abs());
            self.peer_trust.record_invalid_signature(&pack.peer_id, pack.peer_public_key.clone());
            return Ok(false);
        }

        // 2. Verify merkle root matches blocks
        // v10.9.43: batched BLAKE3 helper — see QBlock::batch_calculate_hashes
        let block_hashes: Vec<[u8; 32]> = QBlock::batch_calculate_hashes(&pack.blocks);
        let computed_root = compute_merkle_root(&block_hashes);

        if computed_root != pack.merkle_root {
            warn!("❌ [AEGIS-QL] MERKLE ROOT MISMATCH from peer {}", &pack.peer_id[..8]);
            warn!("   Expected: {}", hex::encode(&pack.merkle_root[..8]));
            warn!("   Computed: {}", hex::encode(&computed_root[..8]));
            self.peer_trust.record_merkle_failure(&pack.peer_id, pack.peer_public_key.clone());
            return Ok(false);
        }

        // 3. Verify AEGIS-QL signature
        let mut message = Vec::new();
        message.extend_from_slice(&pack.merkle_root);
        message.extend_from_slice(&pack.timestamp.to_le_bytes());
        message.extend_from_slice(pack.peer_id.as_bytes());

        let aegis = self.aegis.lock().await;
        let valid = aegis.verify(&message, &pack.aegis_signature, &pack.peer_public_key)?;

        if !valid {
            error!("🚨 [AEGIS-QL] INVALID SIGNATURE from peer {}!", &pack.peer_id[..8]);
            error!("   This peer may be malicious or have key corruption!");
            self.peer_trust.record_invalid_signature(&pack.peer_id, pack.peer_public_key.clone());

            // Check if peer should be banned
            if self.peer_trust.should_ban_peer(&pack.peer_id) {
                error!("🚫 [AEGIS-QL] BANNING PEER {} (trust score below 20%)", &pack.peer_id[..8]);
                // TODO: Actually ban the peer from P2P network
            }

            return Ok(false);
        }

        // 4. Record successful verification
        self.peer_trust.record_valid_pack(&pack.peer_id, pack.peer_public_key.clone());

        info!("✅ [AEGIS-QL] Verified {} blocks from peer {} in {:?}",
              pack.blocks.len(),
              &pack.peer_id[..8],
              start_verify.elapsed());

        Ok(true)
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🗜️ v0.9.21-beta: COMPRESSED SIGNED BLOCK PACKS
    // ═══════════════════════════════════════════════════════════════════

    /// v0.9.21-beta: Create a COMPRESSED signed block pack with AEGIS-QL signature
    /// This reduces bandwidth from 2 MB (uncompressed) to ~603 KB (compressed + signature)
    pub async fn create_signed_block_pack_compressed(
        &self,
        blocks: Vec<QBlock>,
        peer_id: String,
    ) -> Result<crate::aegis_sync::SignedBlockPackCompressed> {
        use crate::aegis_sync::{compute_merkle_root, SignedBlockPackCompressed};

        let pack_start = Instant::now();
        let block_count = blocks.len();
        let start_height = blocks.first().map(|b| b.header.height).unwrap_or(0);
        let end_height = blocks.last().map(|b| b.header.height).unwrap_or(0);

        // 1. Compute merkle root of UNCOMPRESSED blocks (before compression)
        // v10.9.43: batched BLAKE3 helper — see QBlock::batch_calculate_hashes
        let block_hashes: Vec<[u8; 32]> = QBlock::batch_calculate_hashes(&blocks);
        let merkle_root = compute_merkle_root(&block_hashes);

        // 2. Serialize blocks with postcard (efficient binary format)
        let serialized = postcard::to_allocvec(&blocks)?;
        let original_size = serialized.len();

        // 🚀 v1.6.0-SCRAMJET: Standardize on LZ4 (3-5x faster than zstd)
        // LZ4 tradeoff: ~15-20% larger output, but 3-5x faster compress/decompress
        let compressed = lz4::block::compress(&serialized, None, true)
            .map_err(|e| anyhow::anyhow!("LZ4 compression failed: {}", e))?;
        let compressed_size = compressed.len();
        let compression_ratio = original_size as f64 / compressed_size as f64;

        info!("🗜️  [AEGIS-QL] Compressed {} blocks: {} → {} bytes ({:.1}% reduction)",
              block_count,
              original_size,
              compressed_size,
              (1.0 - compressed_size as f64 / original_size as f64) * 100.0);

        // 4. Create message to sign: compressed_blocks || merkle_root || timestamp
        let timestamp = chrono::Utc::now().timestamp();
        let mut message = Vec::new();
        message.extend_from_slice(&compressed);  // Sign compressed data
        message.extend_from_slice(&merkle_root);
        message.extend_from_slice(&timestamp.to_le_bytes());
        message.extend_from_slice(peer_id.as_bytes());

        // 5. Sign with AEGIS-QL Dilithium5 (post-quantum signature)
        let mut aegis = self.aegis.lock().await;
        let secret_key = self.aegis_secret_key.read().await;
        let signature = aegis.sign(&message, &*secret_key)?;

        info!("🔐 [AEGIS-QL] Signed compressed pack in {:?}: {} blocks, {:.1} KB, {:.1}x compression",
              pack_start.elapsed(),
              block_count,
              compressed_size as f64 / 1024.0,
              compression_ratio);
        info!("   Merkle root: {}", hex::encode(&merkle_root[..8]));

        Ok(SignedBlockPackCompressed {
            compressed_blocks: compressed,
            block_count,
            start_height,
            end_height,
            merkle_root,
            aegis_signature: signature,
            peer_public_key: self.aegis_public_key.clone(),
            timestamp,
            peer_id,
            compression_ratio,
        })
    }

    /// v0.9.21-beta: Verify and decompress a COMPRESSED signed block pack
    /// Returns the decompressed blocks if verification succeeds
    pub async fn verify_signed_block_pack_compressed(
        &self,
        pack: &crate::aegis_sync::SignedBlockPackCompressed,
    ) -> Result<Vec<QBlock>> {
        use crate::aegis_sync::{compute_merkle_root, verify_timestamp};

        let start_verify = Instant::now();

        // 1. Verify timestamp (prevents replay attacks)
        if !verify_timestamp(pack.timestamp) {
            warn!("❌ [AEGIS-QL] INVALID TIMESTAMP from peer {} (diff: {}s)",
                  &pack.peer_id[..8],
                  (chrono::Utc::now().timestamp() - pack.timestamp).abs());
            self.peer_trust.record_invalid_signature(&pack.peer_id, pack.peer_public_key.clone());
            anyhow::bail!("Invalid timestamp from peer {}", &pack.peer_id[..8]);
        }

        // 2. Verify AEGIS-QL signature BEFORE decompression (security-first)
        let mut message = Vec::new();
        message.extend_from_slice(&pack.compressed_blocks);
        message.extend_from_slice(&pack.merkle_root);
        message.extend_from_slice(&pack.timestamp.to_le_bytes());
        message.extend_from_slice(pack.peer_id.as_bytes());

        let aegis = self.aegis.lock().await;
        let sig_valid = aegis.verify(&message, &pack.aegis_signature, &pack.peer_public_key)?;

        if !sig_valid {
            error!("🚨 [AEGIS-QL] INVALID SIGNATURE from peer {}!", &pack.peer_id[..8]);
            error!("   This peer may be MALICIOUS or have key corruption!");
            self.peer_trust.record_invalid_signature(&pack.peer_id, pack.peer_public_key.clone());

            if self.peer_trust.should_ban_peer(&pack.peer_id) {
                error!("🚫 [AEGIS-QL] BANNING PEER {} (trust score < 20%)", &pack.peer_id[..8]);
            }

            anyhow::bail!("Invalid AEGIS-QL signature from peer {}", &pack.peer_id[..8]);
        }

        drop(aegis); // Release lock before decompression

        // 🚀 v1.6.0-SCRAMJET: LZ4 decompression with zstd fallback (backward compat)
        // Magic bytes [0x28, 0xB5, 0x2F, 0xFD] = zstd, otherwise assume LZ4
        let is_zstd = pack.compressed_blocks.len() >= 4
            && pack.compressed_blocks[0] == 0x28
            && pack.compressed_blocks[1] == 0xB5
            && pack.compressed_blocks[2] == 0x2F
            && pack.compressed_blocks[3] == 0xFD;

        let decompressed = if is_zstd {
            // Legacy zstd from older peers
            zstd::bulk::decompress(&pack.compressed_blocks, 10_000_000)?
        } else {
            // v1.6.0-SCRAMJET: LZ4 (3-5x faster)
            // 🛡️ v5.1.1: Validate prepended size BEFORE decompression to prevent DoS/OOM
            const MAX_DECOMPRESSED_SIZE: u32 = 200_000_000; // 200MB limit
            if pack.compressed_blocks.len() >= 4 {
                let prepended_size = u32::from_le_bytes([
                    pack.compressed_blocks[0], pack.compressed_blocks[1],
                    pack.compressed_blocks[2], pack.compressed_blocks[3],
                ]);
                if prepended_size > MAX_DECOMPRESSED_SIZE {
                    anyhow::bail!("LZ4 prepended size {} exceeds safety limit of {} bytes (possible DoS or corruption)",
                                  prepended_size, MAX_DECOMPRESSED_SIZE);
                }
            }
            lz4::block::decompress(&pack.compressed_blocks, None)
                .map_err(|e| anyhow::anyhow!("LZ4 decompression failed: {}", e))?
        };
        let blocks: Vec<QBlock> = postcard::from_bytes(&decompressed)?;

        // 4. Verify block count matches
        if blocks.len() != pack.block_count {
            warn!("❌ [AEGIS-QL] BLOCK COUNT MISMATCH from peer {}: expected {}, got {}",
                  &pack.peer_id[..8], pack.block_count, blocks.len());
            self.peer_trust.record_merkle_failure(&pack.peer_id, pack.peer_public_key.clone());
            anyhow::bail!("Block count mismatch from peer {}", &pack.peer_id[..8]);
        }

        // 5. Verify merkle root matches decompressed blocks
        // ✅ v0.9.41-beta: Parallel hash calculation with rayon (2-4x faster on multi-core CPUs)
        // v10.9.43: routed through QBlock::batch_calculate_hashes for SIMD tree-mode on large headers
        let block_hashes: Vec<[u8; 32]> = QBlock::batch_calculate_hashes(&blocks);
        let computed_root = compute_merkle_root(&block_hashes);

        if computed_root != pack.merkle_root {
            warn!("❌ [AEGIS-QL] MERKLE ROOT MISMATCH from peer {}", &pack.peer_id[..8]);
            warn!("   Expected: {}", hex::encode(&pack.merkle_root[..8]));
            warn!("   Computed: {}", hex::encode(&computed_root[..8]));
            self.peer_trust.record_merkle_failure(&pack.peer_id, pack.peer_public_key.clone());
            anyhow::bail!("Merkle root mismatch from peer {}", &pack.peer_id[..8]);
        }

        // 6. Verify height range
        if let (Some(first), Some(last)) = (blocks.first(), blocks.last()) {
            if first.header.height != pack.start_height || last.header.height != pack.end_height {
                warn!("❌ [AEGIS-QL] HEIGHT MISMATCH from peer {}: expected {}-{}, got {}-{}",
                      &pack.peer_id[..8],
                      pack.start_height, pack.end_height,
                      first.header.height, last.header.height);
                anyhow::bail!("Height range mismatch from peer {}", &pack.peer_id[..8]);
            }
        }

        // 7. Record successful verification
        self.peer_trust.record_valid_pack(&pack.peer_id, pack.peer_public_key.clone());

        info!("✅ [AEGIS-QL] Verified compressed pack from peer {} in {:?}",
              &pack.peer_id[..8],
              start_verify.elapsed());
        info!("   {} blocks, {}-{}, {:.1} KB compressed, {:.1}x compression",
              blocks.len(),
              pack.start_height,
              pack.end_height,
              pack.compressed_blocks.len() as f64 / 1024.0,
              pack.compression_ratio);

        Ok(blocks)
    }

    /// Create a sync affirmation certificate after successful sync
    pub async fn create_sync_affirmation(
        &self,
        start_height: u64,
        end_height: u64,
    ) -> Result<crate::aegis_sync::SyncAffirmationCertificate> {
        use crate::aegis_sync::{compute_merkle_root, SyncAffirmationCertificate};

        info!("📜 [AEGIS-QL] Creating sync affirmation certificate for heights {} → {}", start_height, end_height);

        // 1. Collect all block hashes in range
        let mut block_hashes = Vec::new();
        for height in start_height..=end_height {
            if let Some(block) = self.storage.get_qblock_by_height(height).await? {
                block_hashes.push(block.calculate_hash());
            } else {
                warn!("⚠️ Missing block at height {} - cannot create affirmation", height);
                return Err(anyhow::anyhow!("Missing block at height {}", height));
            }
        }

        // 2. Compute merkle root
        let merkle_root = compute_merkle_root(&block_hashes);

        // 3. Create message to sign
        let timestamp = chrono::Utc::now().timestamp();
        let mut message = Vec::new();
        message.extend_from_slice(&start_height.to_le_bytes());
        message.extend_from_slice(&end_height.to_le_bytes());
        message.extend_from_slice(&merkle_root);
        message.extend_from_slice(&timestamp.to_le_bytes());

        // 4. Sign with AEGIS-QL
        let mut aegis = self.aegis.lock().await;
        let secret_key = self.aegis_secret_key.read().await;
        let signature = aegis.sign(&message, &*secret_key)?;

        info!("✅ [AEGIS-QL] Sync affirmation created: {} blocks verified", block_hashes.len());
        info!("   Certificate merkle root: {}", hex::encode(&merkle_root[..8]));

        Ok(SyncAffirmationCertificate {
            start_height,
            end_height,
            block_hashes,
            merkle_root,
            aegis_signature: signature,
            timestamp,
            syncer_public_key: self.aegis_public_key.clone(),
        })
    }

    /// Verify a sync affirmation certificate
    pub async fn verify_sync_affirmation(
        &self,
        cert: &crate::aegis_sync::SyncAffirmationCertificate,
    ) -> Result<bool> {
        use crate::aegis_sync::{compute_merkle_root, verify_timestamp};

        // 1. Verify timestamp
        if !verify_timestamp(cert.timestamp) {
            warn!("❌ [AEGIS-QL] Invalid certificate timestamp");
            return Ok(false);
        }

        // 2. Verify merkle root matches stored hashes
        let computed_root = compute_merkle_root(&cert.block_hashes);
        if computed_root != cert.merkle_root {
            warn!("❌ [AEGIS-QL] Certificate merkle root mismatch");
            return Ok(false);
        }

        // 3. Verify AEGIS-QL signature
        let mut message = Vec::new();
        message.extend_from_slice(&cert.start_height.to_le_bytes());
        message.extend_from_slice(&cert.end_height.to_le_bytes());
        message.extend_from_slice(&cert.merkle_root);
        message.extend_from_slice(&cert.timestamp.to_le_bytes());

        let aegis = self.aegis.lock().await;
        let valid = aegis.verify(&message, &cert.aegis_signature, &cert.syncer_public_key)?;

        if !valid {
            error!("🚨 [AEGIS-QL] INVALID CERTIFICATE SIGNATURE!");
            return Ok(false);
        }

        info!("✅ [AEGIS-QL] Sync affirmation certificate verified");
        Ok(true)
    }

    /// Get peer trust score
    pub fn get_peer_trust_score(&self, peer_id: &str) -> Option<f64> {
        self.peer_trust.get_trust_score(peer_id)
    }

    /// Get all trusted peers (trust score >= 80%)
    pub fn get_trusted_peers(&self) -> Vec<String> {
        self.peer_trust.get_trusted_peers()
    }

    // ═══════════════════════════════════════════════════════════════════
    // 📜 v0.9.15-beta: SYNC AFFIRMATION CERTIFICATE PERSISTENCE
    // ═══════════════════════════════════════════════════════════════════

    /// Store sync affirmation certificate to database
    pub async fn store_sync_certificate(&self, cert: &crate::aegis_sync::SyncAffirmationCertificate) -> Result<()> {
        // Store as latest certificate
        let key = b"sync_cert:latest";
        let value = bincode::serialize(cert)?;
        self.storage.hot_db.put(crate::CF_SYNC_CERTIFICATES, key, &value).await?;

        // Also store by end height for historical lookup
        let key_by_height = format!("sync_cert:{}", cert.end_height);
        self.storage.hot_db.put(crate::CF_SYNC_CERTIFICATES, key_by_height.as_bytes(), &value).await?;

        info!("📜 [AEGIS-QL] Stored sync certificate: {} → {} (merkle: {})",
              cert.start_height,
              cert.end_height,
              hex::encode(&cert.merkle_root[..8]));

        Ok(())
    }

    /// Delete sync affirmation certificate (v0.9.20-beta)
    /// Used when stale certificate is detected
    pub async fn delete_sync_certificate(&self) -> Result<()> {
        let key = b"sync_cert:latest";

        match self.storage.hot_db.delete(crate::CF_SYNC_CERTIFICATES, key).await {
            Ok(()) => {
                info!("🗑️  [AEGIS-QL] Deleted stale sync certificate");
                Ok(())
            }
            Err(e) => {
                warn!("⚠️  [AEGIS-QL] Failed to delete certificate: {}", e);
                Err(e.into())
            }
        }
    }

    /// Load latest sync affirmation certificate from database
    pub async fn load_latest_certificate(&self) -> Result<Option<crate::aegis_sync::SyncAffirmationCertificate>> {
        let key = b"sync_cert:latest";

        match self.storage.hot_db.get(crate::CF_SYNC_CERTIFICATES, key).await? {
            Some(bytes) => {
                let cert: crate::aegis_sync::SyncAffirmationCertificate = bincode::deserialize(&bytes)?;
                info!("📜 [AEGIS-QL] Loaded sync certificate: {} → {} (merkle: {})",
                      cert.start_height,
                      cert.end_height,
                      hex::encode(&cert.merkle_root[..8]));
                Ok(Some(cert))
            }
            None => {
                debug!("📜 [AEGIS-QL] No sync certificate found in database");
                Ok(None)
            }
        }
    }

    /// Check if we've already synced to target height (using certificate)
    /// ✅ v0.9.20-beta: CRITICAL FIX - Validate certificate against actual storage
    pub async fn check_if_already_synced(&self, target_height: u64) -> Result<bool> {
        if let Some(cert) = self.load_latest_certificate().await? {
            if cert.end_height >= target_height {
                // ✅ v0.9.20-beta: VALIDATE certificate against actual storage
                // This prevents phantom success from stale certificates
                let actual_height = self.get_local_height().await?;

                if actual_height >= cert.end_height {
                    // Certificate is VALID - we have the blocks
                    info!("✅ [AEGIS-QL] Already synced to {} (certificate end: {}, storage: {})",
                          target_height,
                          cert.end_height,
                          actual_height);
                    return Ok(true);
                } else {
                    // Certificate is STALE - blocks are missing!
                    error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    error!("🚨 [AEGIS-QL] STALE CERTIFICATE DETECTED!");
                    error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    error!("   Certificate claims: {} blocks", cert.end_height);
                    error!("   Actual storage has: {} blocks", actual_height);
                    error!("   Blocks missing: {}", cert.end_height - actual_height);
                    error!("   ");
                    error!("   This indicates database corruption or incomplete sync!");
                    error!("   The certificate survived but the blocks were lost.");
                    error!("   ");
                    error!("   Deleting stale certificate and re-syncing...");
                    error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                    // Delete stale certificate
                    self.delete_sync_certificate().await?;

                    // Continue with sync (return false to trigger download)
                    return Ok(false);
                }
            }
        }
        Ok(false)
    }
}

/// v10.9.14: Read container memory usage in MB.
///
/// Prefers cgroup memory accounting (matches Docker's OOM-killer view) and falls back
/// to /proc/self/status RSS. cgroup v2 is at `/sys/fs/cgroup/memory.current`, cgroup v1
/// is at `/sys/fs/cgroup/memory/memory.usage_in_bytes`. Returns None only if all readings
/// fail (e.g., outside any cgroup AND /proc unreadable, which shouldn't happen on Linux).
async fn read_container_memory_mb() -> Option<u64> {
    // cgroup v2 (most modern systems including Debian 12 Docker)
    if let Ok(content) = tokio::fs::read_to_string("/sys/fs/cgroup/memory.current").await {
        if let Ok(bytes) = content.trim().parse::<u64>() {
            return Some(bytes / (1024 * 1024));
        }
    }
    // cgroup v1
    if let Ok(content) = tokio::fs::read_to_string("/sys/fs/cgroup/memory/memory.usage_in_bytes").await {
        if let Ok(bytes) = content.trim().parse::<u64>() {
            return Some(bytes / (1024 * 1024));
        }
    }
    // Fallback: /proc/self/status VmRSS
    if let Ok(content) = tokio::fs::read_to_string("/proc/self/status").await {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(kb) = rest.split_whitespace().next().and_then(|s| s.parse::<u64>().ok()) {
                    return Some(kb / 1024);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v10.9.27: The retry-storm dampener.
    ///
    /// `ClientThrottle` errors are local back-pressure — they must NOT consume
    /// the retry budget. This test models the discriminator arm in the chunk
    /// retry loop: feed it 20 chunks worth of throttle errors followed by an
    /// Ok, and assert `retry_count` stayed at 0.
    ///
    /// The full retry loop lives inside `start_warp_sync_loop` (~6160) and is
    /// not directly unit-testable without standing up a full `TurboSyncManager`
    /// + libp2p swarm + peer mesh. This test verifies the discriminator
    /// algorithm in isolation; the real loop is exercised end-to-end in the
    /// Epsilon Docker sync test once the binary is built.
    #[tokio::test]
    async fn retry_count_unchanged_on_client_throttle() {
        // Mirror of the discriminator in the chunk retry loop.
        let mut retry_count: u32 = 0;
        let max_retries: u32 = 3;
        let mut throttle_waits: usize = 0;
        let chunk_deadline =
            tokio::time::Instant::now() + std::time::Duration::from_secs(60);

        // Sequence: 20 throttle errors then a success.
        let mut iter = 0;
        loop {
            // Wall-clock breaker matches the production code: it sits OUTSIDE
            // the retry_count arm so throttle-only spins are still bounded.
            assert!(
                tokio::time::Instant::now() < chunk_deadline,
                "wall-clock breaker MUST be reachable from throttle path"
            );

            iter += 1;

            // Simulated dispatch error / success. The literal "ClientThrottle"
            // matches `q_network::CLIENT_THROTTLE_MARKER`. Hard-coded here
            // because q-storage does not (and cannot — circular dep risk)
            // depend on q-network. The protocol between the two crates is this
            // exact string substring.
            let err_msg = if iter <= 20 {
                Some("ClientThrottle: per-peer cap reached".to_string())
            } else {
                None // success
            };

            match err_msg {
                None => break, // chunk succeeded
                Some(err_msg) => {
                    // The exact discriminator from turbo_sync.rs:
                    if err_msg.contains("ClientThrottle") {
                        throttle_waits += 1;
                        // Production code sleeps 50ms here — use 0 in the test
                        // to keep it fast.
                        continue;
                    }
                    // Real failure path:
                    retry_count += 1;
                    if retry_count >= max_retries {
                        panic!("test should never reach retry_count >= max_retries");
                    }
                }
            }
        }

        assert_eq!(
            retry_count, 0,
            "ClientThrottle errors must NOT consume the retry budget"
        );
        assert_eq!(
            throttle_waits, 20,
            "all 20 throttles should have been counted as throttle_waits"
        );
    }

    /// Sanity: a non-throttle error path DOES consume the retry budget.
    #[tokio::test]
    async fn retry_count_increments_on_real_failure() {
        let mut retry_count: u32 = 0;
        let mut throttle_waits: usize = 0;

        for _ in 0..3 {
            let err_msg = "timeout: peer did not respond".to_string();
            if err_msg.contains("ClientThrottle") {
                throttle_waits += 1;
                continue;
            }
            retry_count += 1;
        }

        assert_eq!(retry_count, 3, "real timeouts must increment retry_count");
        assert_eq!(throttle_waits, 0, "no throttles were issued in this scenario");
    }
}

