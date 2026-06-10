//! Entity lifecycle management — tombstones and caps.
//!
//! Runs at the end of each tick to:
//! 1. Convert dead characters to compact tombstones after a grace period
//! 2. Enforce per-faction army caps
//! 3. Disband zero-troop armies (safety net)
//!
//! ## Why tombstones?
//!
//! Without pruning, the character list grows unboundedly (births > deaths).
//! Each `Character` struct is ~300 bytes with traits, stats, and family links.
//! After 1000 turns that could mean 500+ dead characters weighing on
//! serialization, iteration, and gas cost. Tombstones are ~100 bytes and
//! capped at 200.
//!
//! ## Why army caps?
//!
//! Without caps, AI factions can raise armies every turn, creating hundreds
//! of armies over a long game. Each army participates in battle resolution
//! (O(n^2) scans per province), adding gas creep.

use crown_ash_types::{
    CharacterRole, CharacterTombstone, FixedPoint, GameEvent,
};
use crate::world_state::GameWorld;

/// Maximum number of armies any single faction may have at once.
pub const MAX_ARMIES_PER_FACTION: usize = 5;

/// Maximum number of tombstones retained (oldest are evicted when exceeded).
pub const MAX_TOMBSTONES: usize = 200;

/// Number of turns a dead character is kept in the full characters list
/// before being converted to a tombstone. This grace period allows other
/// systems (succession, events, UI) to reference the recently-deceased.
pub const TOMBSTONE_GRACE_TURNS: u32 = 10;

/// Run end-of-tick lifecycle cleanup.
///
/// Returns events for any tombstoning or auto-disbanding that occurred.
pub fn process_lifecycle(world: &mut GameWorld, current_turn: u32) -> Vec<GameEvent> {
    let mut events = Vec::new();

    // Step 1: Convert dead characters to tombstones after grace period.
    tombstone_dead_characters(world, current_turn, &mut events);

    // Step 2: Enforce army caps per faction.
    enforce_army_caps(world, current_turn, &mut events);

    // Step 3: Remove zero-troop armies (safety net).
    cleanup_empty_armies(world);

    events
}

/// Count how many armies a faction currently controls.
pub fn faction_army_count(world: &GameWorld, faction_id: u8) -> usize {
    world.armies.iter().filter(|a| a.owner_faction == faction_id).count()
}

// ---------------------------------------------------------------------------
// Step 1: Tombstone dead characters
// ---------------------------------------------------------------------------

fn tombstone_dead_characters(
    world: &mut GameWorld,
    current_turn: u32,
    events: &mut Vec<GameEvent>,
) {
    // Collect IDs of characters eligible for tombstoning:
    // - Not alive
    // - Died more than TOMBSTONE_GRACE_TURNS ago (or death_turn unknown and
    //   we've been running long enough that it's safe)
    let mut to_tombstone: Vec<usize> = Vec::new();

    for (idx, ch) in world.characters.iter().enumerate() {
        if ch.alive {
            continue;
        }
        let died_turn = ch.death_turn.unwrap_or(0);
        // Grace period: keep for at least TOMBSTONE_GRACE_TURNS after death.
        if current_turn >= died_turn + TOMBSTONE_GRACE_TURNS {
            to_tombstone.push(idx);
        }
    }

    if to_tombstone.is_empty() {
        return;
    }

    // Before removing, clear spouse references that point to tombstoned characters.
    let tombstone_ids: Vec<u32> = to_tombstone.iter()
        .map(|&idx| world.characters[idx].id)
        .collect();

    // Clear spouse references from living characters pointing to tombstoned IDs.
    for ch in &mut world.characters {
        if let Some(spouse_id) = ch.spouse {
            if tombstone_ids.contains(&spouse_id) {
                ch.spouse = None;
                world.dirty.dirty_characters.insert(ch.id);
            }
        }
        // Clear heir references pointing to tombstoned characters.
        if let Some(heir_id) = ch.heir {
            if tombstone_ids.contains(&heir_id) {
                ch.heir = None;
                world.dirty.dirty_characters.insert(ch.id);
            }
        }
    }

    // Build tombstones and collect removal data.
    // Process in reverse index order so removal doesn't shift earlier indices.
    to_tombstone.sort_unstable();
    to_tombstone.reverse();

    for &idx in &to_tombstone {
        let ch = &world.characters[idx];
        let was_ruler = ch.role == CharacterRole::Ruler;

        let tombstone = CharacterTombstone {
            id: ch.id,
            name: ch.name.clone(),
            dynasty: ch.dynasty,
            faction: ch.faction,
            cause_of_death: ch.death_cause.clone().unwrap_or_else(|| "Unknown".to_string()),
            death_turn: ch.death_turn.unwrap_or(current_turn.saturating_sub(TOMBSTONE_GRACE_TURNS)),
            prestige: ch.prestige,
            age_at_death: ch.age,
            was_ruler,
        };

        events.push(GameEvent::CharacterTombstoned {
            character_id: ch.id,
            character_name: ch.name.clone(),
            turn: current_turn,
        });

        world.dirty.characters_removed.push(ch.id);
        world.tombstones.push(tombstone);
        world.characters.remove(idx);
    }

    // Enforce tombstone cap — remove oldest (lowest death_turn) first.
    if world.tombstones.len() > MAX_TOMBSTONES {
        // Sort by death_turn ascending so we can truncate from the front.
        world.tombstones.sort_by_key(|t| t.death_turn);
        let excess = world.tombstones.len() - MAX_TOMBSTONES;
        world.tombstones.drain(0..excess);
    }
}

// ---------------------------------------------------------------------------
// Step 2: Enforce army caps
// ---------------------------------------------------------------------------

fn enforce_army_caps(
    world: &mut GameWorld,
    current_turn: u32,
    events: &mut Vec<GameEvent>,
) {
    // Collect faction IDs that are alive.
    let faction_ids: Vec<u8> = world.factions.iter()
        .filter(|f| f.alive)
        .map(|f| f.id)
        .collect();

    for faction_id in faction_ids {
        let army_count = faction_army_count(world, faction_id);
        if army_count <= MAX_ARMIES_PER_FACTION {
            continue;
        }

        let excess = army_count - MAX_ARMIES_PER_FACTION;

        // Find the weakest armies (lowest total troops) for this faction.
        let mut faction_armies: Vec<(u32, u32, u16)> = world.armies.iter()
            .filter(|a| a.owner_faction == faction_id)
            .map(|a| (a.id, a.total_soldiers(), a.location))
            .collect();

        // Sort by total soldiers ascending (weakest first).
        faction_armies.sort_by_key(|&(_, soldiers, _)| soldiers);

        // Disband the weakest `excess` armies — return troops to garrison.
        let to_disband: Vec<(u32, u16)> = faction_armies.iter()
            .take(excess)
            .map(|&(id, _, loc)| (id, loc))
            .collect();

        for (army_id, location) in to_disband {
            // Find army data before removal.
            let troops_returned;
            if let Some(army) = world.army(army_id) {
                troops_returned = army.total_soldiers();
                let levy = army.troops.levy;
                let maa = army.troops.men_at_arms;
                let knights = army.troops.knights;

                // Return troops to province garrison.
                if let Some(prov) = world.province_mut_dirty(location) {
                    prov.garrison.levy += levy;
                    prov.garrison.men_at_arms += maa;
                    prov.garrison.knights += knights;
                }
            } else {
                troops_returned = 0;
            }

            // Remove the army.
            if let Some(pos) = world.armies.iter().position(|a| a.id == army_id) {
                world.armies.remove(pos);
                world.dirty.armies_removed.push(army_id);
            }

            events.push(GameEvent::ArmyAutoDisbanded {
                army_id,
                faction: faction_id,
                troops_returned,
                province: location,
                turn: current_turn,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3: Clean up zero-troop armies (safety net)
// ---------------------------------------------------------------------------

fn cleanup_empty_armies(world: &mut GameWorld) {
    let empty_ids: Vec<u32> = world.armies.iter()
        .filter(|a| a.total_soldiers() == 0)
        .map(|a| a.id)
        .collect();

    for id in &empty_ids {
        world.dirty.armies_removed.push(*id);
    }

    world.armies.retain(|a| a.total_soldiers() > 0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::{Army, WorldConfig};
    use crown_ash_types::province::Troops;

    fn test_world() -> GameWorld {
        let config = WorldConfig::default();
        init_world(&config, [0x42; 32])
    }

    #[test]
    fn test_dead_characters_tombstoned() {
        let mut world = test_world();
        let turn = 50u32;
        world.meta.turn = turn;

        // Kill two characters with death_turn well in the past.
        let id0 = world.characters[0].id;
        let id1 = world.characters[1].id;
        world.characters[0].alive = false;
        world.characters[0].death_turn = Some(10);
        world.characters[0].death_cause = Some("OldAge".to_string());
        world.characters[1].alive = false;
        world.characters[1].death_turn = Some(15);
        world.characters[1].death_cause = Some("Assassination".to_string());

        let initial_char_count = world.characters.len();

        let events = process_lifecycle(&mut world, turn);

        // Characters should be removed.
        assert_eq!(world.characters.len(), initial_char_count - 2);
        assert!(world.character(id0).is_none(), "Dead char 0 should be removed");
        assert!(world.character(id1).is_none(), "Dead char 1 should be removed");

        // Tombstones should be created.
        assert_eq!(world.tombstones.len(), 2);

        // Events should be emitted.
        let tombstone_events: Vec<_> = events.iter()
            .filter(|e| matches!(e, GameEvent::CharacterTombstoned { .. }))
            .collect();
        assert_eq!(tombstone_events.len(), 2);
    }

    #[test]
    fn test_tombstone_preserves_key_info() {
        let mut world = test_world();
        let turn = 100u32;
        world.meta.turn = turn;

        // Set up a dead character with specific info.
        let idx = 0;
        let expected_name = world.characters[idx].name.clone();
        let expected_dynasty = world.characters[idx].dynasty;
        let expected_faction = world.characters[idx].faction;
        let expected_prestige = world.characters[idx].prestige;
        let expected_age = world.characters[idx].age;

        world.characters[idx].alive = false;
        world.characters[idx].death_turn = Some(50);
        world.characters[idx].death_cause = Some("Battle".to_string());
        world.characters[idx].role = crown_ash_types::CharacterRole::Ruler;

        let _ = process_lifecycle(&mut world, turn);

        assert!(!world.tombstones.is_empty());
        let tombstone = &world.tombstones[0];
        assert_eq!(tombstone.name, expected_name);
        assert_eq!(tombstone.dynasty, expected_dynasty);
        assert_eq!(tombstone.faction, expected_faction);
        assert_eq!(tombstone.prestige, expected_prestige);
        assert_eq!(tombstone.age_at_death, expected_age);
        assert_eq!(tombstone.cause_of_death, "Battle");
        assert_eq!(tombstone.death_turn, 50);
        assert!(tombstone.was_ruler);
    }

    #[test]
    fn test_tombstone_cap_enforced() {
        let mut world = test_world();
        let turn = 500u32;
        world.meta.turn = turn;

        // Pre-fill tombstones to just under the cap.
        for i in 0..195 {
            world.tombstones.push(CharacterTombstone {
                id: 10000 + i,
                name: format!("OldDead_{}", i),
                dynasty: 0,
                faction: 0,
                cause_of_death: "OldAge".to_string(),
                death_turn: i,
                prestige: FixedPoint::ZERO,
                age_at_death: 70,
                was_ruler: false,
            });
        }

        // Kill 10 characters (will push past cap of 200).
        for idx in 0..10.min(world.characters.len()) {
            world.characters[idx].alive = false;
            world.characters[idx].death_turn = Some(100);
            world.characters[idx].death_cause = Some("OldAge".to_string());
        }

        let _ = process_lifecycle(&mut world, turn);

        // Should be capped at MAX_TOMBSTONES.
        assert!(
            world.tombstones.len() <= MAX_TOMBSTONES,
            "Tombstones should be capped at {}, got {}",
            MAX_TOMBSTONES,
            world.tombstones.len()
        );
    }

    #[test]
    fn test_army_cap_per_faction() {
        let mut world = test_world();
        let turn = 10u32;
        world.meta.turn = turn;

        // Clear existing armies.
        world.armies.clear();

        // Add 8 armies for faction 0 (exceeds cap of 5).
        for i in 0..8 {
            let aid = world.alloc_army_id();
            world.armies.push(Army {
                id: aid,
                owner_faction: 0,
                commander: None,
                troops: Troops { levy: 100 + i * 10, men_at_arms: 20, knights: 5 },
                morale: FixedPoint::from_int(800),
                location: 7,
                destination: None,
                movement_queue: Vec::new(),
                raised_turn: turn,
                supply: FixedPoint::from_int(100),
                siege: None,
            });
        }

        let events = process_lifecycle(&mut world, turn);

        // Should have been reduced to MAX_ARMIES_PER_FACTION.
        let faction_0_armies = faction_army_count(&world, 0);
        assert_eq!(
            faction_0_armies, MAX_ARMIES_PER_FACTION,
            "Faction 0 should have exactly {} armies, got {}",
            MAX_ARMIES_PER_FACTION, faction_0_armies
        );

        // Should have auto-disband events for the 3 excess armies.
        let disband_events: Vec<_> = events.iter()
            .filter(|e| matches!(e, GameEvent::ArmyAutoDisbanded { .. }))
            .collect();
        assert_eq!(disband_events.len(), 3);
    }

    #[test]
    fn test_excess_armies_auto_disbanded_weakest_first() {
        let mut world = test_world();
        let turn = 10u32;
        world.meta.turn = turn;

        // Clear existing armies.
        world.armies.clear();

        // Create armies with specific troop counts so we know which are weakest.
        let troop_counts = [500u32, 100, 300, 50, 200, 150, 400]; // 7 armies
        let mut army_ids = Vec::new();
        for &troops in &troop_counts {
            let aid = world.alloc_army_id();
            army_ids.push(aid);
            world.armies.push(Army {
                id: aid,
                owner_faction: 0,
                commander: None,
                troops: Troops { levy: troops, men_at_arms: 0, knights: 0 },
                morale: FixedPoint::from_int(800),
                location: 7,
                destination: None,
                movement_queue: Vec::new(),
                raised_turn: turn,
                supply: FixedPoint::from_int(100),
                siege: None,
            });
        }

        // Province 7 garrison before.
        let garrison_before = world.province(7).map(|p| p.garrison.levy).unwrap_or(0);

        let _ = process_lifecycle(&mut world, turn);

        // 2 weakest should be disbanded: 50 and 100 troop armies.
        assert_eq!(faction_army_count(&world, 0), MAX_ARMIES_PER_FACTION);

        // The remaining armies should be the 5 strongest.
        let remaining: Vec<u32> = world.armies.iter()
            .filter(|a| a.owner_faction == 0)
            .map(|a| a.troops.levy)
            .collect();
        // Weakest (50, 100) gone. Remaining sorted: 150, 200, 300, 400, 500.
        for &r in &remaining {
            assert!(r >= 150, "Army with {} levy should have been disbanded", r);
        }

        // Troops should return to garrison.
        let garrison_after = world.province(7).map(|p| p.garrison.levy).unwrap_or(0);
        assert_eq!(
            garrison_after,
            garrison_before + 50 + 100,
            "Disbanded troops should return to garrison"
        );
    }

    #[test]
    fn test_zero_troop_armies_removed() {
        let mut world = test_world();
        let turn = 10u32;
        world.meta.turn = turn;

        // Add an empty army.
        let aid = world.alloc_army_id();
        world.armies.push(Army {
            id: aid,
            owner_faction: 0,
            commander: None,
            troops: Troops { levy: 0, men_at_arms: 0, knights: 0 },
            morale: FixedPoint::from_int(800),
            location: 7,
            destination: None,
            movement_queue: Vec::new(),
            raised_turn: turn,
            supply: FixedPoint::from_int(100),
            siege: None,
        });

        let initial = world.armies.len();
        let _ = process_lifecycle(&mut world, turn);

        // The empty army should be removed.
        assert_eq!(world.armies.len(), initial - 1);
        assert!(world.army(aid).is_none(), "Zero-troop army should be removed");
    }

    #[test]
    fn test_grace_period_respected() {
        let mut world = test_world();
        let turn = 15u32;
        world.meta.turn = turn;

        // Kill a character who died very recently (within grace period).
        let id0 = world.characters[0].id;
        world.characters[0].alive = false;
        world.characters[0].death_turn = Some(turn - 2); // Only 2 turns ago.
        world.characters[0].death_cause = Some("OldAge".to_string());

        let initial_count = world.characters.len();
        let _ = process_lifecycle(&mut world, turn);

        // Character should still be in the list (grace period not elapsed).
        assert_eq!(world.characters.len(), initial_count);
        assert!(world.character(id0).is_some(), "Character should still exist during grace period");
        assert!(world.tombstones.is_empty(), "No tombstones yet during grace period");
    }
}
