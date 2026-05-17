//! Serialize / deserialize the [`GameWorld`] to and from plugin storage.
//!
//! The world is split across many small keys so that the host only has to
//! read/write the pieces that actually changed.  This is critical for gas
//! efficiency: a full-world write costs O(N) storage ops, but most ticks
//! only touch a handful of provinces and a few characters.
//!
//! # Key layout
//!
//! | Key pattern                       | Value (bincode)          |
//! |-----------------------------------|--------------------------|
//! | `crown_ash:meta`                  | `WorldMeta`              |
//! | `crown_ash:province_{id}`         | `Province`               |
//! | `crown_ash:character_{id}`        | `Character`              |
//! | `crown_ash:faction_{id}`          | `Faction`                |
//! | `crown_ash:army_{id}`             | `Army`                   |
//! | `crown_ash:dynasty_{id}`          | `Dynasty`                |
//! | `crown_ash:diplomacy_{a}_{b}`     | `DiplomaticRelation`     |
//! | `crown_ash:realm_{faction_id}`    | `Realm`                  |
//! | `crown_ash:action_queue`          | `Vec<QueuedAction>`      |
//! | `crown_ash:index_provinces`       | `Vec<ProvinceId>`        |
//! | `crown_ash:index_characters`      | `Vec<CharacterId>`       |
//! | `crown_ash:index_factions`        | `Vec<FactionId>`         |
//! | `crown_ash:index_armies`          | `Vec<ArmyId>`            |
//! | `crown_ash:index_dynasties`       | `Vec<DynastyId>`         |
//! | `crown_ash:counters`              | `(next_char, next_army)` |

use crown_ash_sim::GameWorld;
use crown_ash_types::*;
use crate::host;

// ---------------------------------------------------------------------------
// Key helpers
// ---------------------------------------------------------------------------

const PREFIX: &str = "crown_ash";

fn key_meta() -> String { format!("{PREFIX}:meta") }
fn key_province(id: ProvinceId) -> String { format!("{PREFIX}:province_{id}") }
fn key_character(id: CharacterId) -> String { format!("{PREFIX}:character_{id}") }
fn key_faction(id: FactionId) -> String { format!("{PREFIX}:faction_{id}") }
fn key_army(id: ArmyId) -> String { format!("{PREFIX}:army_{id}") }
fn key_dynasty(id: DynastyId) -> String { format!("{PREFIX}:dynasty_{id}") }
fn key_diplomacy(a: u8, b: u8) -> String {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    format!("{PREFIX}:diplomacy_{lo}_{hi}")
}
fn key_realm(faction_id: u8) -> String { format!("{PREFIX}:realm_{faction_id}") }
fn key_action_queue() -> String { format!("{PREFIX}:action_queue") }
fn key_index_provinces() -> String { format!("{PREFIX}:index_provinces") }
fn key_index_characters() -> String { format!("{PREFIX}:index_characters") }
fn key_index_factions() -> String { format!("{PREFIX}:index_factions") }
fn key_index_armies() -> String { format!("{PREFIX}:index_armies") }
fn key_index_dynasties() -> String { format!("{PREFIX}:index_dynasties") }
fn key_counters() -> String { format!("{PREFIX}:counters") }

// ---------------------------------------------------------------------------
// Serialisation helpers (bincode)
// ---------------------------------------------------------------------------

fn ser<T: serde::Serialize>(val: &T) -> Vec<u8> {
    bincode::serialize(val).expect("bincode serialize should not fail for game types")
}

fn de<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    bincode::deserialize(bytes).ok()
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

/// Persist the entire `GameWorld` to plugin storage.
///
/// This writes every entity plus the index vectors.  For a partial-save
/// strategy, call the individual `save_*` helpers instead.
pub fn save_world(world: &GameWorld) {
    // Meta
    host::storage_write(&key_meta(), &ser(&world.meta));

    // Counters
    let counters: (u32, u32) = (world.next_character_id, world.next_army_id);
    host::storage_write(&key_counters(), &ser(&counters));

    // Provinces
    let province_ids: Vec<ProvinceId> = world.provinces.iter().map(|p| p.id).collect();
    host::storage_write(&key_index_provinces(), &ser(&province_ids));
    for p in &world.provinces {
        host::storage_write(&key_province(p.id), &ser(p));
    }

    // Characters
    let char_ids: Vec<CharacterId> = world.characters.iter().map(|c| c.id).collect();
    host::storage_write(&key_index_characters(), &ser(&char_ids));
    for c in &world.characters {
        host::storage_write(&key_character(c.id), &ser(c));
    }

    // Factions
    let faction_ids: Vec<FactionId> = world.factions.iter().map(|f| f.id).collect();
    host::storage_write(&key_index_factions(), &ser(&faction_ids));
    for f in &world.factions {
        host::storage_write(&key_faction(f.id), &ser(f));
    }

    // Armies
    let army_ids: Vec<ArmyId> = world.armies.iter().map(|a| a.id).collect();
    host::storage_write(&key_index_armies(), &ser(&army_ids));
    for a in &world.armies {
        host::storage_write(&key_army(a.id), &ser(a));
    }

    // Dynasties
    let dynasty_ids: Vec<DynastyId> = world.dynasties.iter().map(|d| d.id).collect();
    host::storage_write(&key_index_dynasties(), &ser(&dynasty_ids));
    for d in &world.dynasties {
        host::storage_write(&key_dynasty(d.id), &ser(d));
    }

    // Diplomacy
    for rel in &world.diplomacy {
        host::storage_write(&key_diplomacy(rel.faction_a, rel.faction_b), &ser(rel));
    }

    // Realms
    for r in &world.realms {
        host::storage_write(&key_realm(r.faction), &ser(r));
    }

    // Action queue
    host::storage_write(&key_action_queue(), &ser(&world.action_queue));

    host::log_info(&format!(
        "crown-ash: saved world turn={} provinces={} characters={} armies={}",
        world.meta.turn,
        world.provinces.len(),
        world.characters.len(),
        world.armies.len(),
    ));
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Reconstruct the full `GameWorld` from plugin storage.
///
/// Returns `None` if the world has not been initialized (no `meta` key).
pub fn load_world() -> Option<GameWorld> {
    // Meta (mandatory -- if absent the world was never initialized)
    let meta: WorldMeta = de(&host::storage_read(&key_meta())?)?;
    if !meta.initialized {
        return None;
    }

    let counters: (u32, u32) = de(&host::storage_read(&key_counters())?)?;
    let (next_character_id, next_army_id) = counters;

    // Provinces
    let province_ids: Vec<ProvinceId> =
        de(&host::storage_read(&key_index_provinces())?)?;
    let mut provinces = Vec::with_capacity(province_ids.len());
    for id in &province_ids {
        let p: Province = de(&host::storage_read(&key_province(*id))?)?;
        provinces.push(p);
    }

    // Characters
    let char_ids: Vec<CharacterId> =
        de(&host::storage_read(&key_index_characters())?)?;
    let mut characters = Vec::with_capacity(char_ids.len());
    for id in &char_ids {
        let c: Character = de(&host::storage_read(&key_character(*id))?)?;
        characters.push(c);
    }

    // Factions
    let faction_ids: Vec<FactionId> =
        de(&host::storage_read(&key_index_factions())?)?;
    let mut factions = Vec::with_capacity(faction_ids.len());
    for id in &faction_ids {
        let f: Faction = de(&host::storage_read(&key_faction(*id))?)?;
        factions.push(f);
    }

    // Armies
    let army_ids: Vec<ArmyId> =
        de(&host::storage_read(&key_index_armies())?)?;
    let mut armies = Vec::with_capacity(army_ids.len());
    for id in &army_ids {
        let a: Army = de(&host::storage_read(&key_army(*id))?)?;
        armies.push(a);
    }

    // Dynasties
    let dynasty_ids: Vec<DynastyId> =
        de(&host::storage_read(&key_index_dynasties())?)?;
    let mut dynasties = Vec::with_capacity(dynasty_ids.len());
    for id in &dynasty_ids {
        let d: Dynasty = de(&host::storage_read(&key_dynasty(*id))?)?;
        dynasties.push(d);
    }

    // Diplomacy -- reconstructed from all faction pairs
    let mut diplomacy = Vec::new();
    for (i, &a) in faction_ids.iter().enumerate() {
        for &b in &faction_ids[i + 1..] {
            if let Some(bytes) = host::storage_read(&key_diplomacy(a, b)) {
                if let Some(rel) = de::<DiplomaticRelation>(&bytes) {
                    diplomacy.push(rel);
                }
            }
        }
    }

    // Realms
    let mut realms = Vec::with_capacity(faction_ids.len());
    for &fid in &faction_ids {
        if let Some(bytes) = host::storage_read(&key_realm(fid)) {
            if let Some(r) = de::<Realm>(&bytes) {
                realms.push(r);
            }
        }
    }

    // Action queue (may be empty or absent for new worlds)
    let action_queue: Vec<QueuedAction> =
        host::storage_read(&key_action_queue())
            .and_then(|b| de(&b))
            .unwrap_or_default();

    host::log_info(&format!(
        "crown-ash: loaded world turn={} provinces={} characters={} armies={}",
        meta.turn,
        provinces.len(),
        characters.len(),
        armies.len(),
    ));

    Some(GameWorld {
        meta,
        provinces,
        characters,
        factions,
        realms,
        armies,
        dynasties,
        diplomacy,
        action_queue,
        plots: Vec::new(),
        trade_routes: Vec::new(),
        tombstones: Vec::new(),
        next_character_id,
        next_army_id,
        next_plot_id: 0,
        next_trade_route_id: 0,
        dirty: Default::default(),
    })
}

// ---------------------------------------------------------------------------
// Granular save helpers (for gas-efficient partial updates)
// ---------------------------------------------------------------------------

/// Save only the world metadata (turn counter, player count, etc.).
pub fn save_meta(meta: &WorldMeta) {
    host::storage_write(&key_meta(), &ser(meta));
}

/// Save a single province.
pub fn save_province(province: &Province) {
    host::storage_write(&key_province(province.id), &ser(province));
}

/// Save a single character.
pub fn save_character(character: &Character) {
    host::storage_write(&key_character(character.id), &ser(character));
}

/// Save a single faction.
pub fn save_faction(faction: &Faction) {
    host::storage_write(&key_faction(faction.id), &ser(faction));
}

/// Save a single army.  Also updates the army index.
pub fn save_army(army: &Army, all_army_ids: &[ArmyId]) {
    host::storage_write(&key_index_armies(), &ser(&all_army_ids.to_vec()));
    host::storage_write(&key_army(army.id), &ser(army));
}

/// Remove an army from storage and update the index.
pub fn remove_army(army_id: ArmyId, remaining_ids: &[ArmyId]) {
    host::storage_delete(&key_army(army_id));
    host::storage_write(&key_index_armies(), &ser(&remaining_ids.to_vec()));
}

/// Save a single realm.
pub fn save_realm(realm: &Realm) {
    host::storage_write(&key_realm(realm.faction), &ser(realm));
}

/// Save a diplomatic relation.
pub fn save_diplomacy(rel: &DiplomaticRelation) {
    host::storage_write(&key_diplomacy(rel.faction_a, rel.faction_b), &ser(rel));
}

/// Save the ID counters.
pub fn save_counters(next_character_id: u32, next_army_id: u32) {
    let counters: (u32, u32) = (next_character_id, next_army_id);
    host::storage_write(&key_counters(), &ser(&counters));
}

// ---------------------------------------------------------------------------
// Delta save (dirty-tracking-aware partial persist)
// ---------------------------------------------------------------------------

/// Persist only entities that were modified during the current tick.
///
/// Uses the `DirtyTracker` in `world.dirty` to determine which records to
/// write.  Falls back to [`save_world`] when the dirty set exceeds a
/// threshold (more than 60% of entities dirty — not worth the bookkeeping).
///
/// The army index is re-written whenever armies were added or removed.
/// Counters are re-written whenever armies or characters were added.
pub fn save_world_delta(world: &GameWorld) {
    let dirty = &world.dirty;

    // Heuristic: if most entities are dirty, a full save is cheaper than
    // iterating both the dirty set and the entity list.
    let total_entities = world.provinces.len()
        + world.characters.len()
        + world.factions.len()
        + world.realms.len()
        + world.armies.len()
        + world.dynasties.len()
        + world.diplomacy.len()
        + 1; // meta
    if dirty.dirty_count() > total_entities * 6 / 10 {
        save_world(world);
        return;
    }

    // --- Meta ---
    if dirty.meta_dirty {
        host::storage_write(&key_meta(), &ser(&world.meta));
    }

    // --- Provinces ---
    for &pid in &dirty.dirty_provinces {
        if let Some(p) = world.provinces.iter().find(|p| p.id == pid) {
            host::storage_write(&key_province(pid), &ser(p));
        }
    }

    // --- Characters ---
    for &cid in &dirty.dirty_characters {
        if let Some(c) = world.characters.iter().find(|c| c.id == cid) {
            host::storage_write(&key_character(cid), &ser(c));
        }
    }

    // --- Characters added (update index) ---
    if !dirty.characters_added.is_empty() {
        let char_ids: Vec<u32> = world.characters.iter().map(|c| c.id).collect();
        host::storage_write(&key_index_characters(), &ser(&char_ids));
    }

    // --- Factions ---
    for &fid in &dirty.dirty_factions {
        if let Some(f) = world.factions.iter().find(|f| f.id == fid) {
            host::storage_write(&key_faction(fid), &ser(f));
        }
    }

    // --- Realms ---
    for &fid in &dirty.dirty_realms {
        if let Some(r) = world.realms.iter().find(|r| r.faction == fid) {
            host::storage_write(&key_realm(fid), &ser(r));
        }
    }

    // --- Armies ---
    for &aid in &dirty.dirty_armies {
        if let Some(a) = world.armies.iter().find(|a| a.id == aid) {
            host::storage_write(&key_army(aid), &ser(a));
        }
    }

    // If armies were added or removed, update the army index and clean up
    // removed entries from storage.
    if !dirty.armies_added.is_empty() || !dirty.armies_removed.is_empty() {
        let army_ids: Vec<u32> = world.armies.iter().map(|a| a.id).collect();
        host::storage_write(&key_index_armies(), &ser(&army_ids));

        for &removed_id in &dirty.armies_removed {
            host::storage_delete(&key_army(removed_id));
        }
    }

    // --- Dynasties ---
    for &did in &dirty.dirty_dynasties {
        if let Some(d) = world.dynasties.iter().find(|d| d.id == did) {
            host::storage_write(&key_dynasty(did), &ser(d));
        }
    }

    // --- Diplomacy ---
    for &(a, b) in &dirty.dirty_diplomacy {
        if let Some(rel) = world.diplomacy.iter().find(|r| {
            (r.faction_a == a && r.faction_b == b) || (r.faction_a == b && r.faction_b == a)
        }) {
            host::storage_write(&key_diplomacy(a, b), &ser(rel));
        }
    }

    // --- Counters (always save if armies/characters were added) ---
    if !dirty.armies_added.is_empty() || !dirty.characters_added.is_empty() {
        save_counters(world.next_character_id, world.next_army_id);
    }

    // --- Action queue (always save — it was drained or modified) ---
    host::storage_write(&key_action_queue(), &ser(&world.action_queue));

    host::log_info(&format!(
        "crown-ash: delta-saved turn={} dirty: provinces={} chars={} armies={} realms={} factions={} diplomacy={}",
        world.meta.turn,
        dirty.dirty_provinces.len(),
        dirty.dirty_characters.len(),
        dirty.dirty_armies.len(),
        dirty.dirty_realms.len(),
        dirty.dirty_factions.len(),
        dirty.dirty_diplomacy.len(),
    ));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crown_ash_sim::init_world;

    fn test_seed() -> [u8; 32] {
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(7);
        }
        seed
    }

    #[test]
    fn round_trip_full_world() {
        host::reset();

        let config = WorldConfig::default();
        let world = init_world(&config, test_seed());

        save_world(&world);
        let loaded = load_world().expect("load_world should succeed after save_world");

        // Verify key structural properties match.
        assert_eq!(loaded.meta.turn, world.meta.turn);
        assert_eq!(loaded.meta.genesis_block, world.meta.genesis_block);
        assert_eq!(loaded.provinces.len(), world.provinces.len());
        assert_eq!(loaded.characters.len(), world.characters.len());
        assert_eq!(loaded.factions.len(), world.factions.len());
        assert_eq!(loaded.armies.len(), world.armies.len());
        assert_eq!(loaded.dynasties.len(), world.dynasties.len());
        assert_eq!(loaded.diplomacy.len(), world.diplomacy.len());
        assert_eq!(loaded.realms.len(), world.realms.len());
        assert_eq!(loaded.next_character_id, world.next_character_id);
        assert_eq!(loaded.next_army_id, world.next_army_id);
        assert_eq!(loaded.action_queue.len(), world.action_queue.len());

        // Verify province data preserved.
        for (orig, loaded) in world.provinces.iter().zip(loaded.provinces.iter()) {
            assert_eq!(orig.id, loaded.id);
            assert_eq!(orig.name, loaded.name);
            assert_eq!(orig.controller, loaded.controller);
            assert_eq!(orig.population, loaded.population);
        }
    }

    #[test]
    fn load_before_save_returns_none() {
        host::reset();
        assert!(load_world().is_none());
    }

    #[test]
    fn granular_save_province() {
        host::reset();

        let config = WorldConfig::default();
        let mut world = init_world(&config, test_seed());

        // Full save first.
        save_world(&world);

        // Mutate a province and do a granular save.
        world.provinces[0].population = 99_999;
        save_province(&world.provinces[0]);

        // Load and verify the mutation persisted.
        let loaded = load_world().unwrap();
        assert_eq!(loaded.provinces[0].population, 99_999);
    }

    #[test]
    fn delta_save_persists_dirty_province() {
        host::reset();

        let config = WorldConfig::default();
        let mut world = init_world(&config, test_seed());

        // Full save first (creates all keys in storage).
        save_world(&world);

        // Mutate province 0 via dirty accessor.
        let pid = world.provinces[0].id;
        if let Some(prov) = world.province_mut_dirty(pid) {
            prov.population = 77_777;
        }
        // Also mark meta dirty (turn changes every tick).
        world.mark_meta_dirty();
        world.meta.turn = 42;

        // Delta save.
        save_world_delta(&world);

        // Load and verify the dirty mutation persisted, and untouched data
        // is still intact.
        let loaded = load_world().unwrap();
        assert_eq!(loaded.provinces[0].population, 77_777, "Dirty province should be persisted");
        assert_eq!(loaded.meta.turn, 42, "Dirty meta should be persisted");
        // Untouched provinces retain their original values.
        assert_eq!(loaded.provinces.len(), world.provinces.len());
        assert_eq!(loaded.characters.len(), world.characters.len());
    }

    #[test]
    fn delta_save_after_tick_round_trips() {
        host::reset();

        let config = WorldConfig::default();
        let mut world = init_world(&config, test_seed());

        // Full save first.
        save_world(&world);

        // Run a tick (populates dirty tracker).
        let hash = [0xAB; 32];
        let _summary = crown_ash_sim::tick(&mut world, &hash);

        // Delta save.
        save_world_delta(&world);

        // Load and verify turn advanced.
        let loaded = load_world().unwrap();
        assert_eq!(loaded.meta.turn, 1, "Turn should have advanced");
        assert_eq!(loaded.provinces.len(), world.provinces.len());
        assert_eq!(loaded.characters.len(), world.characters.len());
        assert_eq!(loaded.factions.len(), world.factions.len());
    }
}
