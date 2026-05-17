//! Game Trainer — Performance Cheat Engine
//!
//! Like playing on EXTREME difficulty with all cheats enabled.
//! Each "cheat" is a real OS/runtime optimization that squeezes
//! maximum performance from the hardware.
//!
//! F1: INFINITE CORES    — Core pinning, no idle cores
//! F2: GOD MODE MEMORY   — Huge pages, mlock, zero swap
//! F3: SPEED HACK x100   — SIMD + GPU + io_uring
//! F4: WALL HACK         — See all peer compute capacity
//! F5: AIM BOT           — Auto-assign optimal tasks
//! F6: NO CLIP           — Bypass OS scheduler limits
//! F7: INFINITE AMMO     — Never-empty work queue
//! F8: RAPID FIRE        — Batch submit mining solutions
//! F9: TELEPORT          — Zero-copy data paths
//! F10: PRESTIGE MODE    — Overclock safely
//! F11: NUKE             — All cheats at once
//! F12: TRAINER MENU     — Status display + toggle interface

use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{info, warn};

/// Performance boost percentages per cheat (used for display and calculation).
/// These represent the theoretical maximum contribution of each cheat.
const BOOST_F1_INFINITE_CORES: f64 = 120.0;
const BOOST_F2_GOD_MODE_MEMORY: f64 = 80.0;
const BOOST_F3_SPEED_HACK: f64 = 100.0;
const BOOST_F4_WALL_HACK: f64 = 20.0;
const BOOST_F5_AIM_BOT: f64 = 50.0;
const BOOST_F6_NO_CLIP: f64 = 100.0;
const BOOST_F7_INFINITE_AMMO: f64 = 40.0;
const BOOST_F8_RAPID_FIRE: f64 = 30.0;
const BOOST_F9_TELEPORT: f64 = 60.0;
const BOOST_F10_PRESTIGE_MODE: f64 = 80.0;
const BOOST_F11_NUKE: f64 = 200.0;
const BOOST_F12_TRAINER_MENU: f64 = 0.0;

/// Trainer status summary — returned by `status_summary()`
#[derive(Debug, Clone)]
pub struct TrainerStatus {
    /// Names of currently active cheats
    pub active_cheats: Vec<String>,
    /// Total number of cheats available (12)
    pub total_cheats: usize,
    /// Calculated performance boost percentage based on active cheats
    pub performance_boost_percent: f64,
    /// Whether NUKE mode (F11) is currently active
    pub nuke_active: bool,
}

/// Cheat descriptor for the trainer menu
struct CheatInfo {
    key: &'static str,
    name: &'static str,
    boost: f64,
}

/// All 12 cheats in order
const CHEAT_TABLE: [CheatInfo; 12] = [
    CheatInfo { key: "F1",  name: "INFINITE CORES",  boost: BOOST_F1_INFINITE_CORES },
    CheatInfo { key: "F2",  name: "GOD MODE MEMORY", boost: BOOST_F2_GOD_MODE_MEMORY },
    CheatInfo { key: "F3",  name: "SPEED HACK x100", boost: BOOST_F3_SPEED_HACK },
    CheatInfo { key: "F4",  name: "WALL HACK",       boost: BOOST_F4_WALL_HACK },
    CheatInfo { key: "F5",  name: "AIM BOT",         boost: BOOST_F5_AIM_BOT },
    CheatInfo { key: "F6",  name: "NO CLIP",         boost: BOOST_F6_NO_CLIP },
    CheatInfo { key: "F7",  name: "INFINITE AMMO",   boost: BOOST_F7_INFINITE_AMMO },
    CheatInfo { key: "F8",  name: "RAPID FIRE",      boost: BOOST_F8_RAPID_FIRE },
    CheatInfo { key: "F9",  name: "TELEPORT",        boost: BOOST_F9_TELEPORT },
    CheatInfo { key: "F10", name: "PRESTIGE MODE",   boost: BOOST_F10_PRESTIGE_MODE },
    CheatInfo { key: "F11", name: "NUKE",            boost: BOOST_F11_NUKE },
    CheatInfo { key: "F12", name: "TRAINER MENU",    boost: BOOST_F12_TRAINER_MENU },
];

/// Individual trainer cheats
pub struct Trainer {
    // Enable flags (user intent)
    pub infinite_cores: AtomicBool,     // F1
    pub god_mode_memory: AtomicBool,    // F2
    pub speed_hack: AtomicBool,         // F3
    pub wall_hack: AtomicBool,          // F4
    pub aim_bot: AtomicBool,            // F5
    pub no_clip: AtomicBool,            // F6
    pub infinite_ammo: AtomicBool,      // F7
    pub rapid_fire: AtomicBool,         // F8
    pub teleport: AtomicBool,           // F9
    pub prestige_mode: AtomicBool,      // F10
    pub nuke_active: AtomicBool,        // F11

    // Applied flags (true = OS-level side effect actually succeeded)
    applied_infinite_cores: AtomicBool,
    applied_god_mode_memory: AtomicBool,
    applied_speed_hack: AtomicBool,
    applied_no_clip: AtomicBool,
    applied_prestige_mode: AtomicBool,
}

impl Trainer {
    pub fn new() -> Self {
        Self {
            infinite_cores: AtomicBool::new(false),
            god_mode_memory: AtomicBool::new(false),
            speed_hack: AtomicBool::new(false),
            wall_hack: AtomicBool::new(false),
            aim_bot: AtomicBool::new(false),
            no_clip: AtomicBool::new(false),
            infinite_ammo: AtomicBool::new(false),
            rapid_fire: AtomicBool::new(false),
            teleport: AtomicBool::new(false),
            prestige_mode: AtomicBool::new(false),
            nuke_active: AtomicBool::new(false),
            applied_infinite_cores: AtomicBool::new(false),
            applied_god_mode_memory: AtomicBool::new(false),
            applied_speed_hack: AtomicBool::new(false),
            applied_no_clip: AtomicBool::new(false),
            applied_prestige_mode: AtomicBool::new(false),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // F11: NUKE — Enable ALL cheats simultaneously
    // ═══════════════════════════════════════════════════════════════

    /// F11: NUKE — Activate ALL cheats (F1-F10) simultaneously.
    /// Sets compute mode to maximum, calls each activate function in sequence,
    /// and marks nuke as active.
    pub fn activate_nuke(&self) {
        info!("==========================================================");
        info!("  NUKE MODE ACTIVATED -- ALL CHEATS ENABLED");
        info!("==========================================================");
        info!("🎮 [TRAINER] F11 NUKE — Enabling all cheats F1 through F10...");

        // Set all enable flags to true
        self.infinite_cores.store(true, Ordering::SeqCst);
        self.god_mode_memory.store(true, Ordering::SeqCst);
        self.speed_hack.store(true, Ordering::SeqCst);
        self.wall_hack.store(true, Ordering::SeqCst);
        self.aim_bot.store(true, Ordering::SeqCst);
        self.no_clip.store(true, Ordering::SeqCst);
        self.infinite_ammo.store(true, Ordering::SeqCst);
        self.rapid_fire.store(true, Ordering::SeqCst);
        self.teleport.store(true, Ordering::SeqCst);
        self.prestige_mode.store(true, Ordering::SeqCst);
        self.nuke_active.store(true, Ordering::SeqCst);

        // Set compute mode to maximum via environment variable
        std::env::set_var("Q_COMPUTE_MODE", "nuke");

        // Apply each cheat in sequence (OS-level side effects) — track success
        info!("🎮 [TRAINER] NUKE: Applying F1 INFINITE CORES...");
        self.applied_infinite_cores.store(self.apply_infinite_cores(), Ordering::SeqCst);

        info!("🎮 [TRAINER] NUKE: Applying F2 GOD MODE MEMORY...");
        self.applied_god_mode_memory.store(self.apply_god_mode_memory(), Ordering::SeqCst);

        info!("🎮 [TRAINER] NUKE: Applying F3 SPEED HACK...");
        self.applied_speed_hack.store(self.apply_speed_hack(), Ordering::SeqCst);

        info!("🎮 [TRAINER] NUKE: Applying F4 WALL HACK...");
        self.apply_wall_hack();     // Flag-only, always "succeeds"

        info!("🎮 [TRAINER] NUKE: Applying F5 AIM BOT...");
        self.apply_aim_bot();       // Flag-only, always "succeeds"

        info!("🎮 [TRAINER] NUKE: Applying F6 NO CLIP...");
        self.applied_no_clip.store(self.apply_no_clip(), Ordering::SeqCst);

        info!("🎮 [TRAINER] NUKE: Applying F7 INFINITE AMMO...");
        self.apply_infinite_ammo(); // Flag-only

        info!("🎮 [TRAINER] NUKE: Applying F8 RAPID FIRE...");
        self.apply_rapid_fire();    // Flag-only

        info!("🎮 [TRAINER] NUKE: Applying F9 TELEPORT...");
        self.apply_teleport();      // Flag-only

        info!("🎮 [TRAINER] NUKE: Applying F10 PRESTIGE MODE...");
        self.applied_prestige_mode.store(self.apply_prestige_mode(), Ordering::SeqCst);

        let active_count = self.active_cheats().len();
        let boost = self.calculate_display_boost();
        info!("==========================================================");
        info!(
            "  NUKE COMPLETE: {}/10 cheats active, {:.0}% performance boost",
            active_count, boost
        );
        info!("==========================================================");
    }

    /// F11 (legacy alias): activate_all calls activate_nuke
    pub fn activate_all(&self) {
        self.activate_nuke();
    }

    // ═══════════════════════════════════════════════════════════════
    // F12: TRAINER MENU — Formatted status display
    // ═══════════════════════════════════════════════════════════════

    /// F12: TRAINER MENU — Returns a formatted string showing trainer status
    /// as a text UI with all 12 cheats, their ON/OFF status, and aggregate stats.
    pub fn get_trainer_menu(&self) -> String {
        let statuses = self.all_cheat_statuses();
        let active_count = statuses.iter().filter(|&&s| s).count();
        let boost = self.calculate_display_boost();

        let mut menu = String::with_capacity(2048);

        menu.push_str("+--------------------- QNK TRAINER v1.0 -----------------------+\n");

        for (i, cheat) in CHEAT_TABLE.iter().enumerate() {
            let on_off = if statuses[i] { "ON " } else { "OFF" };
            // Pad name to 20 chars for alignment
            menu.push_str(&format!(
                "|  [{}] {:20} [{}]                          |\n",
                cheat.key, cheat.name, on_off
            ));
        }

        menu.push_str("|                                                              |\n");
        menu.push_str(&format!(
            "|  STATUS: {}/12 ACTIVE     Performance: {:.0}% boost              |\n",
            active_count, boost
        ));
        menu.push_str("+--------------------------------------------------------------+\n");

        menu
    }

    // ═══════════════════════════════════════════════════════════════
    // Toggle individual cheats
    // ═══════════════════════════════════════════════════════════════

    /// Toggle an individual cheat on/off by name (F1-F12 or snake_case name).
    /// Returns `true` if the cheat is now ON after toggling, `false` if now OFF.
    /// Returns `false` if the cheat name is not recognized.
    pub fn toggle_cheat(&self, cheat_name: &str) -> bool {
        match cheat_name.to_lowercase().as_str() {
            "f1" | "infinite_cores" => {
                let new_val = !self.infinite_cores.load(Ordering::SeqCst);
                self.infinite_cores.store(new_val, Ordering::SeqCst);
                if new_val {
                    self.applied_infinite_cores.store(self.apply_infinite_cores(), Ordering::SeqCst);
                } else {
                    self.applied_infinite_cores.store(false, Ordering::SeqCst);
                }
                info!("🎮 [TRAINER] F1 INFINITE CORES toggled {}", if new_val { "ON" } else { "OFF" });
                new_val
            }
            "f2" | "god_mode_memory" => {
                let new_val = !self.god_mode_memory.load(Ordering::SeqCst);
                self.god_mode_memory.store(new_val, Ordering::SeqCst);
                if new_val {
                    self.applied_god_mode_memory.store(self.apply_god_mode_memory(), Ordering::SeqCst);
                } else {
                    self.applied_god_mode_memory.store(false, Ordering::SeqCst);
                }
                info!("🎮 [TRAINER] F2 GOD MODE MEMORY toggled {}", if new_val { "ON" } else { "OFF" });
                new_val
            }
            "f3" | "speed_hack" => {
                let new_val = !self.speed_hack.load(Ordering::SeqCst);
                self.speed_hack.store(new_val, Ordering::SeqCst);
                if new_val {
                    self.applied_speed_hack.store(self.apply_speed_hack(), Ordering::SeqCst);
                } else {
                    self.applied_speed_hack.store(false, Ordering::SeqCst);
                }
                info!("🎮 [TRAINER] F3 SPEED HACK toggled {}", if new_val { "ON" } else { "OFF" });
                new_val
            }
            "f4" | "wall_hack" => {
                let new_val = !self.wall_hack.load(Ordering::SeqCst);
                self.wall_hack.store(new_val, Ordering::SeqCst);
                if new_val { self.apply_wall_hack(); }
                info!("🎮 [TRAINER] F4 WALL HACK toggled {}", if new_val { "ON" } else { "OFF" });
                new_val
            }
            "f5" | "aim_bot" => {
                let new_val = !self.aim_bot.load(Ordering::SeqCst);
                self.aim_bot.store(new_val, Ordering::SeqCst);
                if new_val { self.apply_aim_bot(); }
                info!("🎮 [TRAINER] F5 AIM BOT toggled {}", if new_val { "ON" } else { "OFF" });
                new_val
            }
            "f6" | "no_clip" => {
                let new_val = !self.no_clip.load(Ordering::SeqCst);
                self.no_clip.store(new_val, Ordering::SeqCst);
                if new_val {
                    self.applied_no_clip.store(self.apply_no_clip(), Ordering::SeqCst);
                } else {
                    self.applied_no_clip.store(false, Ordering::SeqCst);
                }
                info!("🎮 [TRAINER] F6 NO CLIP toggled {}", if new_val { "ON" } else { "OFF" });
                new_val
            }
            "f7" | "infinite_ammo" => {
                let new_val = !self.infinite_ammo.load(Ordering::SeqCst);
                self.infinite_ammo.store(new_val, Ordering::SeqCst);
                if new_val { self.apply_infinite_ammo(); }
                info!("🎮 [TRAINER] F7 INFINITE AMMO toggled {}", if new_val { "ON" } else { "OFF" });
                new_val
            }
            "f8" | "rapid_fire" => {
                let new_val = !self.rapid_fire.load(Ordering::SeqCst);
                self.rapid_fire.store(new_val, Ordering::SeqCst);
                if new_val { self.apply_rapid_fire(); }
                info!("🎮 [TRAINER] F8 RAPID FIRE toggled {}", if new_val { "ON" } else { "OFF" });
                new_val
            }
            "f9" | "teleport" => {
                let new_val = !self.teleport.load(Ordering::SeqCst);
                self.teleport.store(new_val, Ordering::SeqCst);
                if new_val { self.apply_teleport(); }
                info!("🎮 [TRAINER] F9 TELEPORT toggled {}", if new_val { "ON" } else { "OFF" });
                new_val
            }
            "f10" | "prestige_mode" => {
                let new_val = !self.prestige_mode.load(Ordering::SeqCst);
                self.prestige_mode.store(new_val, Ordering::SeqCst);
                if new_val {
                    self.applied_prestige_mode.store(self.apply_prestige_mode(), Ordering::SeqCst);
                } else {
                    self.applied_prestige_mode.store(false, Ordering::SeqCst);
                }
                info!("🎮 [TRAINER] F10 PRESTIGE MODE toggled {}", if new_val { "ON" } else { "OFF" });
                new_val
            }
            "f11" | "nuke" => {
                let currently_active = self.nuke_active.load(Ordering::SeqCst);
                if currently_active {
                    // Deactivate nuke = deactivate all
                    self.deactivate_all();
                    info!("🎮 [TRAINER] F11 NUKE toggled OFF — all cheats disabled");
                    false
                } else {
                    self.activate_nuke();
                    info!("🎮 [TRAINER] F11 NUKE toggled ON — all cheats enabled");
                    true
                }
            }
            "f12" | "trainer_menu" => {
                // F12 is display-only; toggling it is a no-op that returns false
                info!("🎮 [TRAINER] F12 TRAINER MENU is display-only");
                false
            }
            _ => {
                warn!("🎮 [TRAINER] Unknown cheat name: '{}'", cheat_name);
                false
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Status summary
    // ═══════════════════════════════════════════════════════════════

    /// Returns a `TrainerStatus` struct summarizing the trainer state.
    pub fn status_summary(&self) -> TrainerStatus {
        let active = self.active_cheats();
        let boost = self.calculate_display_boost();
        let nuke = self.nuke_active.load(Ordering::Relaxed);

        TrainerStatus {
            active_cheats: active,
            total_cheats: 12,
            performance_boost_percent: boost,
            nuke_active: nuke,
        }
    }

    /// Deactivate all cheats
    pub fn deactivate_all(&self) {
        info!("🎮 [TRAINER] All cheats deactivated");
        self.infinite_cores.store(false, Ordering::SeqCst);
        self.god_mode_memory.store(false, Ordering::SeqCst);
        self.speed_hack.store(false, Ordering::SeqCst);
        self.wall_hack.store(false, Ordering::SeqCst);
        self.aim_bot.store(false, Ordering::SeqCst);
        self.no_clip.store(false, Ordering::SeqCst);
        self.infinite_ammo.store(false, Ordering::SeqCst);
        self.rapid_fire.store(false, Ordering::SeqCst);
        self.teleport.store(false, Ordering::SeqCst);
        self.prestige_mode.store(false, Ordering::SeqCst);
        self.nuke_active.store(false, Ordering::SeqCst);
        // Clear applied flags
        self.applied_infinite_cores.store(false, Ordering::SeqCst);
        self.applied_god_mode_memory.store(false, Ordering::SeqCst);
        self.applied_speed_hack.store(false, Ordering::SeqCst);
        self.applied_no_clip.store(false, Ordering::SeqCst);
        self.applied_prestige_mode.store(false, Ordering::SeqCst);
    }

    /// Get list of active cheats (F1-F10 + F11 nuke)
    pub fn active_cheats(&self) -> Vec<String> {
        let mut cheats = Vec::new();
        if self.infinite_cores.load(Ordering::Relaxed) { cheats.push("F1:INFINITE_CORES".to_string()); }
        if self.god_mode_memory.load(Ordering::Relaxed) { cheats.push("F2:GOD_MODE_MEMORY".to_string()); }
        if self.speed_hack.load(Ordering::Relaxed) { cheats.push("F3:SPEED_HACK_x100".to_string()); }
        if self.wall_hack.load(Ordering::Relaxed) { cheats.push("F4:WALL_HACK".to_string()); }
        if self.aim_bot.load(Ordering::Relaxed) { cheats.push("F5:AIM_BOT".to_string()); }
        if self.no_clip.load(Ordering::Relaxed) { cheats.push("F6:NO_CLIP".to_string()); }
        if self.infinite_ammo.load(Ordering::Relaxed) { cheats.push("F7:INFINITE_AMMO".to_string()); }
        if self.rapid_fire.load(Ordering::Relaxed) { cheats.push("F8:RAPID_FIRE".to_string()); }
        if self.teleport.load(Ordering::Relaxed) { cheats.push("F9:TELEPORT".to_string()); }
        if self.prestige_mode.load(Ordering::Relaxed) { cheats.push("F10:PRESTIGE_MODE".to_string()); }
        if self.nuke_active.load(Ordering::Relaxed) { cheats.push("F11:NUKE".to_string()); }
        cheats
    }

    /// Estimated performance boost percentage — only counts cheats that actually applied.
    /// Cheats that only set environment variables (F4, F5, F7, F8, F9) don't claim
    /// a boost because nothing reads those variables yet.
    pub fn estimated_boost_pct(&self) -> f32 {
        let mut boost: f32 = 0.0;
        // F1: Core pinning — only if core_affinity succeeded
        if self.infinite_cores.load(Ordering::Relaxed) && self.applied_infinite_cores.load(Ordering::Relaxed) {
            boost += 150.0; // Core pinning = ~150% mining boost
        }
        // F2: Huge pages / mlock — only if mlockall succeeded
        if self.god_mode_memory.load(Ordering::Relaxed) && self.applied_god_mode_memory.load(Ordering::Relaxed) {
            boost += 30.0;  // Huge pages = ~30% less TLB misses
        }
        // F3: SIMD detection — no GPU/SIMD hashing code exists, so this is informational only.
        // Real boost is 0% until SIMD hash kernels are implemented.
        // F6: RT scheduler — only if sched_setscheduler succeeded
        if self.no_clip.load(Ordering::Relaxed) && self.applied_no_clip.load(Ordering::Relaxed) {
            boost += 50.0;  // RT scheduler = ~50% less jitter
        }
        // F10: CPU governor — only if governor write succeeded
        if self.prestige_mode.load(Ordering::Relaxed) && self.applied_prestige_mode.load(Ordering::Relaxed) {
            boost += 15.0;  // Max turbo = ~15% clock boost
        }
        // F4, F5, F7, F8, F9 only set env vars — no measurable boost until consumer code reads them
        boost
    }

    /// Check if a specific cheat was actually applied (not just enabled)
    pub fn is_cheat_applied(&self, name: &str) -> bool {
        match name {
            "F1" | "infinite_cores" => self.applied_infinite_cores.load(Ordering::Relaxed),
            "F2" | "god_mode_memory" => self.applied_god_mode_memory.load(Ordering::Relaxed),
            "F3" | "speed_hack" => self.applied_speed_hack.load(Ordering::Relaxed),
            "F6" | "no_clip" => self.applied_no_clip.load(Ordering::Relaxed),
            "F10" | "prestige_mode" => self.applied_prestige_mode.load(Ordering::Relaxed),
            // Flag-only cheats: "applied" = enabled (they always succeed)
            "F4" | "wall_hack" => self.wall_hack.load(Ordering::Relaxed),
            "F5" | "aim_bot" => self.aim_bot.load(Ordering::Relaxed),
            "F7" | "infinite_ammo" => self.infinite_ammo.load(Ordering::Relaxed),
            "F8" | "rapid_fire" => self.rapid_fire.load(Ordering::Relaxed),
            "F9" | "teleport" => self.teleport.load(Ordering::Relaxed),
            "F11" | "nuke" => self.nuke_active.load(Ordering::Relaxed),
            _ => false,
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Internal helpers
    // ═══════════════════════════════════════════════════════════════

    /// Get the ON/OFF status of all 12 cheats as a fixed-size array.
    /// Order: F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12
    /// F12 (trainer menu) is always considered "OFF" when queried (it's display-only).
    fn all_cheat_statuses(&self) -> [bool; 12] {
        [
            self.infinite_cores.load(Ordering::Relaxed),   // F1
            self.god_mode_memory.load(Ordering::Relaxed),  // F2
            self.speed_hack.load(Ordering::Relaxed),       // F3
            self.wall_hack.load(Ordering::Relaxed),        // F4
            self.aim_bot.load(Ordering::Relaxed),          // F5
            self.no_clip.load(Ordering::Relaxed),          // F6
            self.infinite_ammo.load(Ordering::Relaxed),    // F7
            self.rapid_fire.load(Ordering::Relaxed),       // F8
            self.teleport.load(Ordering::Relaxed),         // F9
            self.prestige_mode.load(Ordering::Relaxed),    // F10
            self.nuke_active.load(Ordering::Relaxed),      // F11
            false,                                          // F12 (display-only, not a toggle)
        ]
    }

    /// Calculate the display performance boost percentage based on enabled cheats.
    /// Uses the per-cheat boost constants (F1=120%, F2=80%, etc.).
    /// This is the "theoretical maximum" boost for the trainer menu display —
    /// different from `estimated_boost_pct()` which only counts actually-applied cheats.
    fn calculate_display_boost(&self) -> f64 {
        let statuses = self.all_cheat_statuses();
        let mut boost = 0.0;
        for (i, cheat) in CHEAT_TABLE.iter().enumerate() {
            if statuses[i] {
                boost += cheat.boost;
            }
        }
        boost
    }

    // ═══════════════════════════════════════════════════════════════
    // Cheat implementations
    // ═══════════════════════════════════════════════════════════════

    /// F1: Pin all threads to cores, no idle allowed. Returns true if pinning succeeded.
    fn apply_infinite_cores(&self) -> bool {
        let total = num_cpus::get();
        info!("🎮 [F1] INFINITE CORES — Pinning {} cores", total);

        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        if core_ids.is_empty() {
            warn!("🎮 [F1] Could not get core IDs — skipping core pinning");
            return false;
        }

        if let Some(core) = core_ids.first() {
            core_affinity::set_for_current(*core);
        }

        info!("🎮 [F1] Core pinning active — {} cores available", core_ids.len());
        true
    }

    /// F2: Enable huge pages + mlock all memory. Returns true if mlockall succeeded.
    fn apply_god_mode_memory(&self) -> bool {
        info!("🎮 [F2] GOD MODE MEMORY — Huge pages + mlock");

        #[cfg(target_os = "linux")]
        {
            let result = unsafe {
                libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE)
            };
            if result == 0 {
                info!("🎮 [F2] mlockall() success — all memory pinned to RAM");
                info!("🎮 [F2] Huge pages requested via madvise");
                return true;
            } else {
                warn!("🎮 [F2] mlockall() failed (need CAP_IPC_LOCK or root) — continuing without");
                return false;
            }
        }

        #[cfg(target_os = "windows")]
        {
            info!("🎮 [F2] Windows: Large pages require SeLockMemoryPrivilege — skipping auto-apply");
            return false;
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        { false }
    }

    /// F6: Set real-time scheduler priority. Returns true if scheduler change succeeded.
    fn apply_no_clip(&self) -> bool {
        info!("🎮 [F6] NO CLIP — Bypassing OS scheduler limits");

        #[cfg(target_os = "linux")]
        {
            unsafe {
                let param = libc::sched_param { sched_priority: 50 };
                let result = libc::sched_setscheduler(0, libc::SCHED_FIFO, &param);
                if result == 0 {
                    info!("🎮 [F6] SCHED_FIFO priority 50 — mining threads preempt everything");
                    return true;
                } else {
                    warn!("🎮 [F6] SCHED_FIFO failed (need root/CAP_SYS_NICE) — using nice -20 fallback");
                    let nice_result = libc::setpriority(libc::PRIO_PROCESS, 0, -20);
                    return nice_result == 0;
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            info!("🎮 [F6] Windows: Setting HIGH_PRIORITY_CLASS");
            unsafe {
                let handle = windows_sys::Win32::System::Threading::GetCurrentProcess();
                windows_sys::Win32::System::Threading::SetPriorityClass(
                    handle,
                    windows_sys::Win32::System::Threading::HIGH_PRIORITY_CLASS,
                );
            }
            return true;
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        { false }
    }

    /// F3: SPEED HACK — Detect SIMD capabilities and set environment for acceleration.
    fn apply_speed_hack(&self) -> bool {
        let mut detected_any = false;
        info!("🎮 [F3] SPEED HACK x100 — Detecting SIMD + acceleration capabilities");

        #[cfg(target_arch = "x86_64")]
        {
            let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
            let has_avx512f = std::arch::is_x86_feature_detected!("avx512f");
            let has_aes = std::arch::is_x86_feature_detected!("aes");
            let has_sse42 = std::arch::is_x86_feature_detected!("sse4.2");

            if has_avx512f {
                info!("🎮 [F3] AVX-512 detected (no SIMD hash kernels yet — informational)");
                std::env::set_var("Q_SIMD_LEVEL", "avx512");
                detected_any = true;
            } else if has_avx2 {
                info!("🎮 [F3] AVX2 detected (no SIMD hash kernels yet — informational)");
                std::env::set_var("Q_SIMD_LEVEL", "avx2");
                detected_any = true;
            } else if has_sse42 {
                info!("🎮 [F3] SSE4.2 detected (no SIMD hash kernels yet — informational)");
                std::env::set_var("Q_SIMD_LEVEL", "sse42");
                detected_any = true;
            } else {
                info!("🎮 [F3] No SIMD extensions detected — scalar fallback");
                std::env::set_var("Q_SIMD_LEVEL", "scalar");
            }

            if has_aes {
                info!("🎮 [F3] AES-NI detected — hardware AES acceleration");
                std::env::set_var("Q_AES_NI", "1");
                detected_any = true;
            }

            let _ = has_avx2;
            let _ = has_avx512f;
            let _ = has_sse42;
        }

        #[cfg(target_arch = "aarch64")]
        {
            info!("🎮 [F3] ARM64 NEON SIMD detected (informational)");
            std::env::set_var("Q_SIMD_LEVEL", "neon");
            detected_any = true;
        }

        #[cfg(target_os = "linux")]
        {
            if std::path::Path::new("/dev/nvidia0").exists() {
                info!("🎮 [F3] NVIDIA GPU detected (no GPU compute code yet — informational)");
                std::env::set_var("Q_GPU_AVAILABLE", "nvidia");
            } else if std::path::Path::new("/dev/dri/renderD128").exists() {
                info!("🎮 [F3] GPU render node detected (informational)");
                std::env::set_var("Q_GPU_AVAILABLE", "opencl");
            } else {
                info!("🎮 [F3] No GPU detected — CPU-only mode");
                std::env::set_var("Q_GPU_AVAILABLE", "none");
            }
        }

        info!("🎮 [F3] Speed hack active — acceleration environment configured");
        detected_any
    }

    /// F4: WALL HACK — Enable peer compute visibility
    fn apply_wall_hack(&self) {
        info!("🎮 [F4] WALL HACK — Peer compute visibility enabled");
        std::env::set_var("Q_COMPUTE_WALL_HACK", "1");
        info!("🎮 [F4] Wall hack active — compute-tunnel topic subscription requested");
    }

    /// F5: AIM BOT — Enable optimal task assignment scoring
    fn apply_aim_bot(&self) {
        info!("🎮 [F5] AIM BOT — Optimal task assignment enabled");
        std::env::set_var("Q_COMPUTE_AIM_BOT", "1");
        info!("🎮 [F5] Aim bot active — score-based task routing enabled");
    }

    /// F7: INFINITE AMMO — Enable work queue prefetch and pipelining
    fn apply_infinite_ammo(&self) {
        info!("🎮 [F7] INFINITE AMMO — Work queue prefetch enabled");
        std::env::set_var("Q_MINING_PREFETCH", "1");
        std::env::set_var("Q_MINING_BUFFER_DEPTH", "4");
        info!("🎮 [F7] Infinite ammo active — mining work pipeline depth = 4");
    }

    /// F8: RAPID FIRE — Enable batch mining solution submission
    fn apply_rapid_fire(&self) {
        info!("🎮 [F8] RAPID FIRE — Batch submit mode enabled");
        std::env::set_var("Q_MINING_BATCH_SIZE", "8");
        std::env::set_var("Q_MINING_BATCH_FLUSH_MS", "50");
        info!("🎮 [F8] Rapid fire active — batch size=8, flush every 50ms");
    }

    /// F9: TELEPORT — Enable zero-copy data paths
    fn apply_teleport(&self) {
        info!("🎮 [F9] TELEPORT — Zero-copy data paths enabled");

        #[cfg(target_os = "linux")]
        {
            std::env::set_var("Q_ROCKSDB_MMAP_READS", "1");
            std::env::set_var("Q_SPLICE_ENABLED", "1");
            std::env::set_var("Q_READAHEAD_KB", "256");
            info!("🎮 [F9] Teleport active — mmap reads + splice + 256KB readahead");
        }

        #[cfg(not(target_os = "linux"))]
        {
            std::env::set_var("Q_ROCKSDB_MMAP_READS", "1");
            info!("🎮 [F9] Teleport active — mmap reads enabled (splice unavailable)");
        }
    }

    /// F10: Set CPU governor to performance. Returns true if at least one governor was set.
    fn apply_prestige_mode(&self) -> bool {
        info!("🎮 [F10] PRESTIGE MODE — Maximum clock speed");

        #[cfg(target_os = "linux")]
        {
            let total_cores = num_cpus::get();
            let mut governors_set = 0;
            for i in 0..total_cores {
                let path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor", i);
                if std::fs::write(&path, "performance").is_ok() {
                    governors_set += 1;
                }
            }
            if governors_set > 0 {
                info!("🎮 [F10] CPU governor -> performance on {}/{} cores", governors_set, total_cores);
            } else {
                info!("🎮 [F10] CPU governor change failed (need root) — may already be 'performance'");
            }

            let _ = std::fs::write("/sys/devices/system/cpu/intel_pstate/no_turbo", "0");
            info!("🎮 [F10] Turbo boost enabled (if available)");
            return governors_set > 0;
        }

        #[cfg(target_os = "windows")]
        {
            info!("🎮 [F10] Windows: Set power plan to High Performance via powercfg");
            return false;
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        { false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let trainer = Trainer::new();
        assert!(trainer.active_cheats().is_empty());
        assert_eq!(trainer.estimated_boost_pct(), 0.0);
        assert!(!trainer.nuke_active.load(Ordering::Relaxed));
    }

    #[test]
    fn test_boost_requires_applied() {
        let trainer = Trainer::new();
        trainer.infinite_cores.store(true, Ordering::SeqCst);
        assert_eq!(trainer.estimated_boost_pct(), 0.0);
        trainer.applied_infinite_cores.store(true, Ordering::SeqCst);
        assert_eq!(trainer.estimated_boost_pct(), 150.0);
    }

    #[test]
    fn test_f3_speed_hack_no_boost() {
        let trainer = Trainer::new();
        trainer.speed_hack.store(true, Ordering::SeqCst);
        trainer.applied_speed_hack.store(true, Ordering::SeqCst);
        assert_eq!(trainer.estimated_boost_pct(), 0.0);
    }

    #[test]
    fn test_deactivate() {
        let trainer = Trainer::new();
        trainer.infinite_cores.store(true, Ordering::SeqCst);
        trainer.god_mode_memory.store(true, Ordering::SeqCst);
        trainer.applied_infinite_cores.store(true, Ordering::SeqCst);
        trainer.nuke_active.store(true, Ordering::SeqCst);
        assert_eq!(trainer.active_cheats().len(), 3); // F1 + F2 + F11
        assert_eq!(trainer.estimated_boost_pct(), 150.0);
        trainer.deactivate_all();
        assert!(trainer.active_cheats().is_empty());
        assert_eq!(trainer.estimated_boost_pct(), 0.0);
        assert!(!trainer.nuke_active.load(Ordering::Relaxed));
    }

    #[test]
    fn test_all_cheats_counted() {
        let trainer = Trainer::new();
        trainer.infinite_cores.store(true, Ordering::SeqCst);
        trainer.god_mode_memory.store(true, Ordering::SeqCst);
        trainer.speed_hack.store(true, Ordering::SeqCst);
        trainer.wall_hack.store(true, Ordering::SeqCst);
        trainer.aim_bot.store(true, Ordering::SeqCst);
        trainer.no_clip.store(true, Ordering::SeqCst);
        trainer.infinite_ammo.store(true, Ordering::SeqCst);
        trainer.rapid_fire.store(true, Ordering::SeqCst);
        trainer.teleport.store(true, Ordering::SeqCst);
        trainer.prestige_mode.store(true, Ordering::SeqCst);
        assert_eq!(trainer.active_cheats().len(), 10);
        assert_eq!(trainer.estimated_boost_pct(), 0.0);

        trainer.applied_infinite_cores.store(true, Ordering::SeqCst);
        trainer.applied_god_mode_memory.store(true, Ordering::SeqCst);
        trainer.applied_no_clip.store(true, Ordering::SeqCst);
        trainer.applied_prestige_mode.store(true, Ordering::SeqCst);
        assert_eq!(trainer.estimated_boost_pct(), 245.0);
    }

    #[test]
    fn test_is_cheat_applied() {
        let trainer = Trainer::new();
        assert!(!trainer.is_cheat_applied("F1"));
        trainer.applied_infinite_cores.store(true, Ordering::SeqCst);
        assert!(trainer.is_cheat_applied("F1"));
        assert!(trainer.is_cheat_applied("infinite_cores"));

        assert!(!trainer.is_cheat_applied("F4"));
        trainer.wall_hack.store(true, Ordering::SeqCst);
        assert!(trainer.is_cheat_applied("F4"));

        assert!(!trainer.is_cheat_applied("F11"));
        trainer.nuke_active.store(true, Ordering::SeqCst);
        assert!(trainer.is_cheat_applied("F11"));
        assert!(trainer.is_cheat_applied("nuke"));
    }

    // F11 NUKE tests

    #[test]
    fn test_nuke_activates_all_cheats() {
        let trainer = Trainer::new();
        assert!(trainer.active_cheats().is_empty());
        trainer.activate_nuke();
        let active = trainer.active_cheats();
        assert_eq!(active.len(), 11);
        assert!(trainer.nuke_active.load(Ordering::Relaxed));
        assert!(trainer.infinite_cores.load(Ordering::Relaxed));
        assert!(trainer.god_mode_memory.load(Ordering::Relaxed));
        assert!(trainer.speed_hack.load(Ordering::Relaxed));
        assert!(trainer.wall_hack.load(Ordering::Relaxed));
        assert!(trainer.aim_bot.load(Ordering::Relaxed));
        assert!(trainer.no_clip.load(Ordering::Relaxed));
        assert!(trainer.infinite_ammo.load(Ordering::Relaxed));
        assert!(trainer.rapid_fire.load(Ordering::Relaxed));
        assert!(trainer.teleport.load(Ordering::Relaxed));
        assert!(trainer.prestige_mode.load(Ordering::Relaxed));
    }

    #[test]
    fn test_nuke_sets_compute_mode_env() {
        let trainer = Trainer::new();
        trainer.activate_nuke();
        assert_eq!(std::env::var("Q_COMPUTE_MODE").unwrap_or_default(), "nuke");
    }

    // F12 TRAINER MENU tests

    #[test]
    fn test_trainer_menu_all_off() {
        let trainer = Trainer::new();
        let menu = trainer.get_trainer_menu();
        assert!(menu.contains("QNK TRAINER v1.0"));
        assert!(menu.contains("[F1]"));
        assert!(menu.contains("[F12]"));
        assert!(menu.contains("0/12 ACTIVE"));
        assert!(menu.contains("0% boost"));
    }

    #[test]
    fn test_trainer_menu_some_on() {
        let trainer = Trainer::new();
        trainer.infinite_cores.store(true, Ordering::SeqCst);
        trainer.speed_hack.store(true, Ordering::SeqCst);
        trainer.teleport.store(true, Ordering::SeqCst);
        let menu = trainer.get_trainer_menu();
        assert!(menu.contains("3/12 ACTIVE"));
        assert!(menu.contains("280% boost"));
    }

    #[test]
    fn test_trainer_menu_nuke_on() {
        let trainer = Trainer::new();
        trainer.activate_nuke();
        let menu = trainer.get_trainer_menu();
        assert!(menu.contains("11/12 ACTIVE"));
    }

    // Toggle tests

    #[test]
    fn test_toggle_individual_cheat() {
        let trainer = Trainer::new();
        assert!(!trainer.wall_hack.load(Ordering::Relaxed));
        let result = trainer.toggle_cheat("F4");
        assert!(result);
        assert!(trainer.wall_hack.load(Ordering::Relaxed));
        let result = trainer.toggle_cheat("F4");
        assert!(!result);
        assert!(!trainer.wall_hack.load(Ordering::Relaxed));
    }

    #[test]
    fn test_toggle_by_name() {
        let trainer = Trainer::new();
        let result = trainer.toggle_cheat("aim_bot");
        assert!(result);
        assert!(trainer.aim_bot.load(Ordering::Relaxed));
        let result = trainer.toggle_cheat("aim_bot");
        assert!(!result);
        assert!(!trainer.aim_bot.load(Ordering::Relaxed));
    }

    #[test]
    fn test_toggle_nuke_on_off() {
        let trainer = Trainer::new();
        let result = trainer.toggle_cheat("F11");
        assert!(result);
        assert!(trainer.nuke_active.load(Ordering::Relaxed));
        assert!(trainer.infinite_cores.load(Ordering::Relaxed));
        let result = trainer.toggle_cheat("F11");
        assert!(!result);
        assert!(!trainer.nuke_active.load(Ordering::Relaxed));
        assert!(!trainer.infinite_cores.load(Ordering::Relaxed));
    }

    #[test]
    fn test_toggle_unknown_cheat() {
        let trainer = Trainer::new();
        let result = trainer.toggle_cheat("nonexistent");
        assert!(!result);
    }

    #[test]
    fn test_toggle_f12_is_noop() {
        let trainer = Trainer::new();
        let result = trainer.toggle_cheat("F12");
        assert!(!result);
    }

    // Status summary tests

    #[test]
    fn test_status_summary_empty() {
        let trainer = Trainer::new();
        let status = trainer.status_summary();
        assert!(status.active_cheats.is_empty());
        assert_eq!(status.total_cheats, 12);
        assert_eq!(status.performance_boost_percent, 0.0);
        assert!(!status.nuke_active);
    }

    #[test]
    fn test_status_summary_with_cheats() {
        let trainer = Trainer::new();
        trainer.infinite_cores.store(true, Ordering::SeqCst);
        trainer.god_mode_memory.store(true, Ordering::SeqCst);
        let status = trainer.status_summary();
        assert_eq!(status.active_cheats.len(), 2);
        assert_eq!(status.total_cheats, 12);
        assert_eq!(status.performance_boost_percent, 200.0);
        assert!(!status.nuke_active);
    }

    #[test]
    fn test_status_summary_nuke() {
        let trainer = Trainer::new();
        trainer.activate_nuke();
        let status = trainer.status_summary();
        assert_eq!(status.active_cheats.len(), 11);
        assert_eq!(status.total_cheats, 12);
        assert!(status.nuke_active);
        // F1(120)+F2(80)+F3(100)+F4(20)+F5(50)+F6(100)+F7(40)+F8(30)+F9(60)+F10(80)+F11(200) = 880
        assert_eq!(status.performance_boost_percent, 880.0);
    }

    #[test]
    fn test_display_boost_vs_estimated_boost() {
        let trainer = Trainer::new();
        trainer.infinite_cores.store(true, Ordering::SeqCst);
        trainer.god_mode_memory.store(true, Ordering::SeqCst);
        let display = trainer.calculate_display_boost();
        assert_eq!(display, 200.0);
        assert_eq!(trainer.estimated_boost_pct(), 0.0);
        trainer.applied_infinite_cores.store(true, Ordering::SeqCst);
        assert_eq!(trainer.estimated_boost_pct(), 150.0);
    }
}
