//! Crown & Ash -- RocksDB persistence for game state.
//!
//! Saves and loads [`CrownAshGameState`] checkpoints to/from the main
//! blockchain RocksDB instance.  This allows the game world to survive
//! server restarts without requiring a separate database.
//!
//! # Storage layout
//!
//! All keys live in the `CF_MANIFEST` column family, prefixed with
//! `crown_ash:` to avoid collisions with other metadata stored there.
//!
//! | Key                          | Value (bincode)                   |
//! |------------------------------|-----------------------------------|
//! | `crown_ash:world`            | Full [`GameWorld`] snapshot        |
//! | `crown_ash:turn`             | `u64` -- current game turn        |
//! | `crown_ash:history`          | `Vec<TurnSummary>` -- turn log    |
//! | `crown_ash:action_queue`     | `Vec<QueuedAction>` -- pending    |
//!
//! # Design notes
//!
//! - Serialization uses **bincode** for compact, fast encoding.
//! - All functions accept `&dyn KVStore` (the async trait from `q-storage`)
//!   and use the `CF_MANIFEST` column family.
//! - Deserialization errors are logged and mapped to `None` rather than
//!   panicking, so a corrupted checkpoint degrades gracefully to a fresh
//!   world instead of crashing the server.

use std::sync::Arc;

use anyhow::Result;
use tracing::{error, info, warn};

use crown_ash_sim::GameWorld;
use crown_ash_types::{QueuedAction, TurnSummary};
use q_storage::KVStore;

use crate::CrownAshGameState;

// ---------------------------------------------------------------------------
// RocksDB column family and key constants
// ---------------------------------------------------------------------------

/// Column family used for all Crown & Ash keys.
/// We share the general-purpose `manifest` CF rather than registering a new
/// one, keeping the persistence layer zero-config.
const CF: &str = "manifest";

/// Key for the full [`GameWorld`] checkpoint.
const KEY_WORLD: &[u8] = b"crown_ash:world";

/// Key for the persisted turn counter (`u64`).
const KEY_TURN: &[u8] = b"crown_ash:turn";

/// Key for the accumulated turn history (`Vec<TurnSummary>`).
const KEY_HISTORY: &[u8] = b"crown_ash:history";

/// Key for the pending action queue (`Vec<QueuedAction>`).
const KEY_ACTION_QUEUE: &[u8] = b"crown_ash:action_queue";

// ---------------------------------------------------------------------------
// Save helpers
// ---------------------------------------------------------------------------

/// Persist the full game state (world + action queue + turn history) to RocksDB.
///
/// This is the primary save entry-point.  Call it after every game tick or
/// significant state mutation to ensure durability.
///
/// Uses `put` (WAL-protected but not per-write fsync) for throughput.  The
/// main server's periodic `sync_wal()` call covers crash durability.
pub async fn save_game_state(db: &Arc<dyn KVStore>, state: &CrownAshGameState) -> Result<()> {
    // World snapshot (the largest payload -- typically 50-200 KB for a 25-province game).
    save_world(db, &state.world).await?;

    // Turn counter (8 bytes -- trivial).
    save_turn(db, state.world.meta.turn as u64).await?;

    // Action queue (usually small; empty between ticks).
    save_action_queue(db, &state.action_queue).await?;

    // Turn history (grows over time; capped by caller if needed).
    save_history(db, &state.turn_history).await?;

    info!(
        turn = state.world.meta.turn,
        provinces = state.world.provinces.len(),
        history_len = state.turn_history.len(),
        queue_len = state.action_queue.len(),
        "crown-ash: game state persisted to RocksDB"
    );

    Ok(())
}

/// Persist only the [`GameWorld`] snapshot.
///
/// Useful when you want to checkpoint the world without touching the
/// turn history (e.g. after processing a player action mid-tick).
pub async fn save_world(db: &Arc<dyn KVStore>, world: &GameWorld) -> Result<()> {
    let encoded = bincode::serialize(world)
        .map_err(|e| anyhow::anyhow!("crown-ash: failed to serialize GameWorld: {e}"))?;
    db.put(CF, KEY_WORLD, &encoded).await?;
    Ok(())
}

/// Persist the current turn number.
pub async fn save_turn(db: &Arc<dyn KVStore>, turn: u64) -> Result<()> {
    db.put(CF, KEY_TURN, &turn.to_le_bytes()).await?;
    Ok(())
}

/// Persist the accumulated turn history.
pub async fn save_history(db: &Arc<dyn KVStore>, history: &[TurnSummary]) -> Result<()> {
    let encoded = bincode::serialize(history)
        .map_err(|e| anyhow::anyhow!("crown-ash: failed to serialize turn history: {e}"))?;
    db.put(CF, KEY_HISTORY, &encoded).await?;
    Ok(())
}

/// Persist the pending action queue.
pub async fn save_action_queue(db: &Arc<dyn KVStore>, queue: &[QueuedAction]) -> Result<()> {
    let encoded = bincode::serialize(queue)
        .map_err(|e| anyhow::anyhow!("crown-ash: failed to serialize action queue: {e}"))?;
    db.put(CF, KEY_ACTION_QUEUE, &encoded).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Load helpers
// ---------------------------------------------------------------------------

/// Load the full game state from RocksDB.
///
/// Returns `Ok(Some(state))` if a valid checkpoint exists, `Ok(None)` if no
/// checkpoint has been saved yet, or `Ok(None)` if the stored data is corrupt
/// (with a warning logged).
///
/// This is the primary load entry-point, intended to be called once at server
/// startup before the game tick loop begins.
pub async fn load_game_state(db: &Arc<dyn KVStore>) -> Result<Option<CrownAshGameState>> {
    let world = match load_world(db).await? {
        Some(w) => w,
        None => return Ok(None),
    };

    let action_queue = load_action_queue(db).await?.unwrap_or_default();
    let turn_history = load_history(db).await?.unwrap_or_default();

    info!(
        turn = world.meta.turn,
        provinces = world.provinces.len(),
        characters = world.characters.len(),
        factions = world.factions.len(),
        history_len = turn_history.len(),
        queue_len = action_queue.len(),
        "crown-ash: game state loaded from RocksDB"
    );

    Ok(Some(CrownAshGameState {
        world,
        action_queue,
        turn_history,
    }))
}

/// Load the [`GameWorld`] snapshot from RocksDB.
///
/// Returns `None` if the key does not exist or if deserialization fails
/// (the latter case logs a warning).
pub async fn load_world(db: &Arc<dyn KVStore>) -> Result<Option<GameWorld>> {
    match db.get(CF, KEY_WORLD).await? {
        Some(bytes) => match bincode::deserialize::<GameWorld>(&bytes) {
            Ok(world) => Ok(Some(world)),
            Err(e) => {
                warn!(
                    "crown-ash: failed to deserialize GameWorld ({} bytes): {e}. \
                     Starting fresh.",
                    bytes.len()
                );
                Ok(None)
            }
        },
        None => Ok(None),
    }
}

/// Load the persisted turn number.
///
/// Returns `0` if the key is absent or the stored value has an unexpected
/// length.
pub async fn load_turn(db: &Arc<dyn KVStore>) -> Result<u64> {
    match db.get(CF, KEY_TURN).await? {
        Some(bytes) if bytes.len() == 8 => {
            let arr: [u8; 8] = bytes[..8]
                .try_into()
                .expect("slice length verified above");
            Ok(u64::from_le_bytes(arr))
        }
        Some(bytes) => {
            warn!(
                "crown-ash: turn key has unexpected length {} (expected 8). Defaulting to 0.",
                bytes.len()
            );
            Ok(0)
        }
        None => Ok(0),
    }
}

/// Load the accumulated turn history.
///
/// Returns `None` if the key is absent.  Deserialization errors are logged
/// and mapped to `None`.
pub async fn load_history(db: &Arc<dyn KVStore>) -> Result<Option<Vec<TurnSummary>>> {
    match db.get(CF, KEY_HISTORY).await? {
        Some(bytes) => match bincode::deserialize::<Vec<TurnSummary>>(&bytes) {
            Ok(history) => Ok(Some(history)),
            Err(e) => {
                warn!(
                    "crown-ash: failed to deserialize turn history ({} bytes): {e}",
                    bytes.len()
                );
                Ok(None)
            }
        },
        None => Ok(None),
    }
}

/// Load the pending action queue.
///
/// Returns `None` if the key is absent.  Deserialization errors are logged
/// and mapped to `None`.
pub async fn load_action_queue(db: &Arc<dyn KVStore>) -> Result<Option<Vec<QueuedAction>>> {
    match db.get(CF, KEY_ACTION_QUEUE).await? {
        Some(bytes) => match bincode::deserialize::<Vec<QueuedAction>>(&bytes) {
            Ok(queue) => Ok(Some(queue)),
            Err(e) => {
                warn!(
                    "crown-ash: failed to deserialize action queue ({} bytes): {e}",
                    bytes.len()
                );
                Ok(None)
            }
        },
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Delete / reset
// ---------------------------------------------------------------------------

/// Remove all Crown & Ash keys from RocksDB.
///
/// Use this when the game world is reset or the game feature is disabled.
pub async fn delete_game_state(db: &Arc<dyn KVStore>) -> Result<()> {
    // Best-effort deletion -- log but don't fail on individual key errors.
    for key in [KEY_WORLD, KEY_TURN, KEY_HISTORY, KEY_ACTION_QUEUE] {
        if let Err(e) = db.delete(CF, key).await {
            error!(
                "crown-ash: failed to delete key {:?}: {e}",
                String::from_utf8_lossy(key)
            );
        }
    }
    info!("crown-ash: game state deleted from RocksDB");
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crown_ash_types::WorldMeta;

    /// Build a minimal GameWorld for testing serialization round-trips.
    fn test_world() -> GameWorld {
        GameWorld {
            meta: WorldMeta {
                turn: 42,
                genesis_block: 1000,
                player_count: 2,
                initialized: true,
                world_version: 1,
                sim_version: crown_ash_types::SIM_VERSION.to_string(),
            },
            provinces: Vec::new(),
            characters: Vec::new(),
            factions: Vec::new(),
            realms: Vec::new(),
            armies: Vec::new(),
            dynasties: Vec::new(),
            diplomacy: Vec::new(),
            action_queue: Vec::new(),
            plots: Vec::new(),
            trade_routes: Vec::new(),
            tombstones: Vec::new(),
            next_character_id: 100,
            next_army_id: 50,
            next_plot_id: 0,
            next_trade_route_id: 0,
            dirty: Default::default(),
        }
    }

    fn test_state() -> CrownAshGameState {
        CrownAshGameState {
            world: test_world(),
            action_queue: Vec::new(),
            turn_history: vec![TurnSummary {
                turn: 41,
                block_height: 990,
                events: Vec::new(),
                active_factions: 7,
                total_armies: 12,
                total_population: 250_000,
            }],
        }
    }

    /// Bincode round-trip for GameWorld (does not require a real KV store).
    #[test]
    fn world_bincode_round_trip() {
        let world = test_world();
        let encoded = bincode::serialize(&world).unwrap();
        let decoded: GameWorld = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded.meta.turn, 42);
        assert_eq!(decoded.meta.genesis_block, 1000);
        assert_eq!(decoded.next_character_id, 100);
        assert_eq!(decoded.next_army_id, 50);
    }

    /// Bincode round-trip for the full CrownAshGameState.
    #[test]
    fn game_state_bincode_round_trip() {
        let state = test_state();

        // World
        let world_bytes = bincode::serialize(&state.world).unwrap();
        let world_back: GameWorld = bincode::deserialize(&world_bytes).unwrap();
        assert_eq!(world_back.meta.turn, 42);

        // History
        let history_bytes = bincode::serialize(&state.turn_history).unwrap();
        let history_back: Vec<TurnSummary> = bincode::deserialize(&history_bytes).unwrap();
        assert_eq!(history_back.len(), 1);
        assert_eq!(history_back[0].turn, 41);

        // Action queue
        let queue_bytes = bincode::serialize(&state.action_queue).unwrap();
        let queue_back: Vec<QueuedAction> = bincode::deserialize(&queue_bytes).unwrap();
        assert!(queue_back.is_empty());
    }

    /// Turn counter encoding round-trip.
    #[test]
    fn turn_encoding_round_trip() {
        let turn: u64 = 12345;
        let bytes = turn.to_le_bytes();
        assert_eq!(bytes.len(), 8);
        let decoded = u64::from_le_bytes(bytes);
        assert_eq!(decoded, 12345);
    }

    /// Verify key constants are distinct and properly prefixed.
    #[test]
    fn key_constants_are_distinct() {
        let keys: &[&[u8]] = &[KEY_WORLD, KEY_TURN, KEY_HISTORY, KEY_ACTION_QUEUE];
        for (i, a) in keys.iter().enumerate() {
            // All start with "crown_ash:"
            assert!(
                a.starts_with(b"crown_ash:"),
                "key {:?} missing crown_ash: prefix",
                String::from_utf8_lossy(a)
            );
            for (j, b) in keys.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "keys at index {i} and {j} collide");
                }
            }
        }
    }
}
