//! World — top-level metadata for the game state.

use serde::{Deserialize, Serialize};

/// Blocks between game ticks.
pub const BLOCKS_PER_TURN: u64 = 10;

/// Maximum number of factions.
pub const MAX_FACTIONS: u8 = 7;

/// Maximum provinces on the map.
pub const MAX_PROVINCES: u16 = 25;

/// Simulation version — **MUST be bumped whenever tick logic changes.**
/// Nodes with mismatched SIM_VERSION reject each other's game state.
pub const SIM_VERSION: &str = "1.2.0";

/// Per-step work caps to prevent gas exhaustion on pathological turns.
pub const MAX_BATTLES_PER_TURN: usize = 5;
pub const MAX_EVENTS_PER_TURN: usize = 10;
pub const MAX_SUCCESSIONS_PER_TURN: usize = 3;
pub const MAX_BIRTHS_PER_TURN: usize = 5;
pub const MAX_DEATHS_PER_TURN: usize = 5;

/// World metadata persisted in plugin storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldMeta {
    /// Current game turn (increments each tick).
    pub turn: u32,
    /// Block height at which the world was initialized.
    pub genesis_block: u64,
    /// Number of players currently joined.
    pub player_count: u8,
    /// Whether world generation has completed.
    pub initialized: bool,
    /// Version of world generation (for future upgrades).
    pub world_version: u8,
    /// Simulation version stamp — must match SIM_VERSION to process ticks.
    pub sim_version: String,
}

impl Default for WorldMeta {
    fn default() -> Self {
        Self {
            turn: 0,
            genesis_block: 0,
            player_count: 0,
            initialized: false,
            world_version: 1,
            sim_version: SIM_VERSION.to_string(),
        }
    }
}

/// Configuration for world generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    pub province_count: u16,
    pub faction_count: u8,
    pub starting_characters_per_faction: u8,
    pub starting_population_per_province: u32,
    pub starting_treasury: i64,
    pub blocks_per_turn: u64,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            province_count: 25,
            faction_count: 7,
            starting_characters_per_faction: 5,
            starting_population_per_province: 10_000,
            starting_treasury: 500_000, // 500.000 in FixedPoint
            blocks_per_turn: BLOCKS_PER_TURN,
        }
    }
}
