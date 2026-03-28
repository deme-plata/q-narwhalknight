//! GameWorld — the complete simulation state held in-memory.
//!
//! `GameWorld` is the single root of all mutable game state.
//! It is serialised to bincode for on-chain plugin storage.
//!
//! # Dirty Tracking
//!
//! The `DirtyTracker` records which entities were modified during a tick.
//! Only dirty entities need to be persisted, enabling delta-write persistence
//! that dramatically reduces storage gas vs full-world serialization.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crown_ash_types::{
    Army, Character, CharacterTombstone, DiplomaticRelation, Dynasty, Faction,
    Plot, Province, QueuedAction, Realm, TradeRoute, WorldMeta,
};

/// Tracks which entities were modified during a tick.
/// Used by the persistence layer to write only changed records (delta writes).
#[derive(Debug, Clone, Default)]
pub struct DirtyTracker {
    pub meta_dirty: bool,
    pub dirty_provinces: HashSet<u16>,
    pub dirty_characters: HashSet<u32>,
    pub dirty_factions: HashSet<u8>,
    pub dirty_realms: HashSet<u8>,       // keyed by faction_id
    pub dirty_armies: HashSet<u32>,
    pub dirty_dynasties: HashSet<u16>,
    pub dirty_diplomacy: HashSet<(u8, u8)>, // canonical (min, max) pairs
    pub armies_added: Vec<u32>,
    pub armies_removed: Vec<u32>,
    pub characters_added: Vec<u32>,
    pub characters_removed: Vec<u32>,
}

impl DirtyTracker {
    /// Reset all dirty flags (call at start of each tick).
    pub fn clear(&mut self) {
        self.meta_dirty = false;
        self.dirty_provinces.clear();
        self.dirty_characters.clear();
        self.dirty_factions.clear();
        self.dirty_realms.clear();
        self.dirty_armies.clear();
        self.dirty_dynasties.clear();
        self.dirty_diplomacy.clear();
        self.armies_added.clear();
        self.armies_removed.clear();
        self.characters_added.clear();
        self.characters_removed.clear();
    }

    /// Number of dirty entities (for deciding full-checkpoint vs delta-write).
    pub fn dirty_count(&self) -> usize {
        self.dirty_provinces.len()
            + self.dirty_characters.len()
            + self.dirty_factions.len()
            + self.dirty_realms.len()
            + self.dirty_armies.len()
            + self.dirty_dynasties.len()
            + self.dirty_diplomacy.len()
            + if self.meta_dirty { 1 } else { 0 }
    }

    /// Mark a diplomacy pair dirty (canonicalizes order).
    pub fn mark_diplomacy(&mut self, a: u8, b: u8) {
        self.dirty_diplomacy.insert((a.min(b), a.max(b)));
    }
}

/// Complete game world state.
///
/// Every field must be deterministic and serialisable.
/// **No floating point** anywhere in this struct tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameWorld {
    pub meta: WorldMeta,
    pub provinces: Vec<Province>,
    pub characters: Vec<Character>,
    pub factions: Vec<Faction>,
    pub realms: Vec<Realm>,
    pub armies: Vec<Army>,
    pub dynasties: Vec<Dynasty>,
    pub diplomacy: Vec<DiplomaticRelation>,
    pub action_queue: Vec<QueuedAction>,
    pub plots: Vec<Plot>,
    pub trade_routes: Vec<TradeRoute>,
    /// Compact records of dead characters, capped at 200.
    /// Replaces full Character structs after a grace period to prevent unbounded growth.
    #[serde(default)]
    pub tombstones: Vec<CharacterTombstone>,
    pub next_character_id: u32,
    pub next_army_id: u32,
    pub next_plot_id: u32,
    pub next_trade_route_id: u32,
    /// Tracks which entities were modified during the current tick.
    /// Not serialized — rebuilt each tick.
    #[serde(skip)]
    pub dirty: DirtyTracker,
}

impl GameWorld {
    /// Find a character by ID.
    pub fn character(&self, id: u32) -> Option<&Character> {
        self.characters.iter().find(|c| c.id == id)
    }

    /// Find a mutable character by ID.
    pub fn character_mut(&mut self, id: u32) -> Option<&mut Character> {
        self.characters.iter_mut().find(|c| c.id == id)
    }

    /// Find an army by ID.
    pub fn army(&self, id: u32) -> Option<&Army> {
        self.armies.iter().find(|a| a.id == id)
    }

    /// Find a mutable army by ID.
    pub fn army_mut(&mut self, id: u32) -> Option<&mut Army> {
        self.armies.iter_mut().find(|a| a.id == id)
    }

    /// Find a province by ID.
    pub fn province(&self, id: u16) -> Option<&Province> {
        self.provinces.iter().find(|p| p.id == id)
    }

    /// Find a mutable province by ID.
    pub fn province_mut(&mut self, id: u16) -> Option<&mut Province> {
        self.provinces.iter_mut().find(|p| p.id == id)
    }

    /// Find the faction by ID.
    pub fn faction(&self, id: u8) -> Option<&Faction> {
        self.factions.iter().find(|f| f.id == id)
    }

    /// Find the mutable faction by ID.
    pub fn faction_mut(&mut self, id: u8) -> Option<&mut Faction> {
        self.factions.iter_mut().find(|f| f.id == id)
    }

    /// Find the realm owned by a wallet.
    pub fn realm_by_wallet(&self, wallet: &str) -> Option<&Realm> {
        self.realms.iter().find(|r| r.owner_wallet == wallet)
    }

    /// Find the mutable realm owned by a wallet.
    pub fn realm_by_wallet_mut(&mut self, wallet: &str) -> Option<&mut Realm> {
        self.realms.iter_mut().find(|r| r.owner_wallet == wallet)
    }

    /// Find the realm for a faction.
    pub fn realm_for_faction(&self, faction_id: u8) -> Option<&Realm> {
        self.realms.iter().find(|r| r.faction == faction_id)
    }

    /// Find the mutable realm for a faction.
    pub fn realm_for_faction_mut(&mut self, faction_id: u8) -> Option<&mut Realm> {
        self.realms.iter_mut().find(|r| r.faction == faction_id)
    }

    /// Get diplomatic relation between two factions (order-independent).
    pub fn relation(&self, a: u8, b: u8) -> Option<&DiplomaticRelation> {
        self.diplomacy.iter().find(|r| {
            (r.faction_a == a && r.faction_b == b) || (r.faction_a == b && r.faction_b == a)
        })
    }

    /// Get mutable diplomatic relation between two factions (order-independent).
    pub fn relation_mut(&mut self, a: u8, b: u8) -> Option<&mut DiplomaticRelation> {
        self.diplomacy.iter_mut().find(|r| {
            (r.faction_a == a && r.faction_b == b) || (r.faction_a == b && r.faction_b == a)
        })
    }

    /// Are two factions at war?
    pub fn at_war(&self, a: u8, b: u8) -> bool {
        self.relation(a, b).map_or(false, |r| r.at_war)
    }

    /// Count provinces controlled by a faction.
    pub fn faction_province_count(&self, faction_id: u8) -> usize {
        self.provinces.iter().filter(|p| p.controller == faction_id).count()
    }

    /// Allocate a new character ID and increment the counter.
    pub fn alloc_character_id(&mut self) -> u32 {
        let id = self.next_character_id;
        self.next_character_id += 1;
        id
    }

    /// Allocate a new army ID and increment the counter.
    pub fn alloc_army_id(&mut self) -> u32 {
        let id = self.next_army_id;
        self.next_army_id += 1;
        id
    }

    /// Allocate a new plot ID and increment the counter.
    pub fn alloc_plot_id(&mut self) -> u32 {
        let id = self.next_plot_id;
        self.next_plot_id += 1;
        id
    }

    /// Allocate a new trade route ID and increment the counter.
    pub fn alloc_trade_route_id(&mut self) -> u32 {
        let id = self.next_trade_route_id;
        self.next_trade_route_id += 1;
        id
    }

    /// List armies in a given province belonging to a specific faction.
    pub fn armies_in_province(&self, province_id: u16, faction_id: u8) -> Vec<u32> {
        self.armies
            .iter()
            .filter(|a| a.location == province_id && a.owner_faction == faction_id && !a.is_moving())
            .map(|a| a.id)
            .collect()
    }

    /// Total military power of a faction across all armies.
    pub fn faction_total_power(&self, faction_id: u8) -> i64 {
        self.armies
            .iter()
            .filter(|a| a.owner_faction == faction_id)
            .map(|a| a.attack_power().raw())
            .sum()
    }

    // -----------------------------------------------------------------------
    // Dirty-tracking wrappers
    // -----------------------------------------------------------------------

    /// Get a mutable province and mark it dirty.
    pub fn province_mut_dirty(&mut self, id: u16) -> Option<&mut Province> {
        self.dirty.dirty_provinces.insert(id);
        self.province_mut(id)
    }

    /// Get a mutable character and mark it dirty.
    pub fn character_mut_dirty(&mut self, id: u32) -> Option<&mut Character> {
        self.dirty.dirty_characters.insert(id);
        self.character_mut(id)
    }

    /// Get a mutable army and mark it dirty.
    pub fn army_mut_dirty(&mut self, id: u32) -> Option<&mut Army> {
        self.dirty.dirty_armies.insert(id);
        self.army_mut(id)
    }

    /// Get a mutable realm for a faction and mark it dirty.
    pub fn realm_for_faction_mut_dirty(&mut self, faction_id: u8) -> Option<&mut Realm> {
        self.dirty.dirty_realms.insert(faction_id);
        self.realm_for_faction_mut(faction_id)
    }

    /// Get a mutable diplomatic relation and mark it dirty.
    pub fn relation_mut_dirty(&mut self, a: u8, b: u8) -> Option<&mut DiplomaticRelation> {
        self.dirty.mark_diplomacy(a, b);
        self.relation_mut(a, b)
    }

    /// Mark meta as dirty (turn increment, player count change, etc.).
    pub fn mark_meta_dirty(&mut self) {
        self.dirty.meta_dirty = true;
    }

    /// Reset dirty tracker (call at start of each tick).
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    // -----------------------------------------------------------------------
    // Post-tick invariant checks (debug builds only)
    // -----------------------------------------------------------------------

    /// Validate world state invariants. Panics in debug builds if any fail.
    /// Call after every tick in tests to catch determinism bugs early.
    #[cfg(any(debug_assertions, test))]
    pub fn assert_invariants(&self) {
        let max_faction = self.factions.len() as u8;
        let max_province = self.provinces.len() as u16;

        // Province controllers are valid faction IDs; province values in bounds
        for p in &self.provinces {
            assert!(
                p.controller < max_faction,
                "Province {} has invalid controller {}",
                p.id, p.controller
            );
            assert!(p.id < max_province, "Province ID {} out of range", p.id);
            // Unrest must be in [0, 1000]
            assert!(
                p.unrest.raw() >= 0 && p.unrest.raw() <= 1_000_000,
                "Province {} unrest out of bounds: {}",
                p.id, p.unrest.raw()
            );
            // Prosperity must be in [0, 1000]
            assert!(
                p.prosperity.raw() >= 0 && p.prosperity.raw() <= 1_000_000,
                "Province {} prosperity out of bounds: {}",
                p.id, p.prosperity.raw()
            );
            // No negative population
            // population is u32, always >= 0
            // Neighbors reference valid provinces
            for &n in &p.neighbors {
                assert!(
                    self.province(n).is_some(),
                    "Province {} has invalid neighbor {}",
                    p.id, n
                );
            }
        }

        // Army locations are valid provinces
        for a in &self.armies {
            assert!(
                self.province(a.location).is_some(),
                "Army {} at invalid province {}",
                a.id, a.location
            );
            assert!(
                a.owner_faction < max_faction,
                "Army {} has invalid owner faction {}",
                a.id, a.owner_faction
            );
            if let Some(dest) = a.destination {
                assert!(
                    self.province(dest).is_some(),
                    "Army {} moving to invalid province {}",
                    a.id, dest
                );
            }
        }

        // Characters reference valid factions
        for c in &self.characters {
            if c.alive {
                assert!(
                    c.faction < max_faction,
                    "Character {} ({}) has invalid faction {}",
                    c.id, c.name, c.faction
                );
            }
        }

        // Unique army IDs
        let mut army_ids: Vec<u32> = self.armies.iter().map(|a| a.id).collect();
        army_ids.sort_unstable();
        army_ids.dedup();
        assert_eq!(
            army_ids.len(),
            self.armies.len(),
            "Duplicate army IDs found"
        );

        // Diplomacy pairs are canonical (no duplicates, no self-relations)
        for r in &self.diplomacy {
            assert_ne!(
                r.faction_a, r.faction_b,
                "Self-diplomatic relation found: {}",
                r.faction_a
            );
        }

        // Realm province ownership matches province controller data
        for realm in &self.realms {
            for &prov_id in &realm.provinces {
                if let Some(p) = self.province(prov_id) {
                    assert_eq!(
                        p.controller, realm.faction,
                        "Realm faction {} claims province {} but controller is {}",
                        realm.faction, prov_id, p.controller
                    );
                }
            }
        }

        // Sim version matches
        assert_eq!(
            self.meta.sim_version,
            crown_ash_types::SIM_VERSION,
            "WorldMeta sim_version mismatch: {} vs {}",
            self.meta.sim_version,
            crown_ash_types::SIM_VERSION,
        );
    }

    /// No-op in release builds.
    #[cfg(not(any(debug_assertions, test)))]
    pub fn assert_invariants(&self) {}
}
