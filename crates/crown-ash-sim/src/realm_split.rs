//! Realm Split -- partitions a faction's provinces among succession claimants.
//!
//! When a succession crisis fires with `realm_split = true` and 2+ claimants,
//! the realm is partitioned: the winner keeps the capital region, and the
//! strongest remaining claimant gets a breakaway faction with nearby provinces.
//!
//! Province partition uses BFS from the capital province. The winner keeps
//! provinces closest to capital (up to ceil(total/2)). The rebel gets the rest.
//!
//! Design constraints:
//! - All math uses `FixedPoint` -- no floating point.
//! - New faction ID = `factions.len() as u8` (must stay < 255).
//! - Dirty tracking must be maintained for all modified entities.
//! - Realm province lists must stay consistent with province controllers.

use std::collections::VecDeque;

use crown_ash_types::{
    CharacterRole, DiplomaticRelation, Dynasty, Faction,
    FixedPoint, GameEvent, Realm, RealmCohesion,
};
use crown_ash_types::dynasty::SuccessionRule;

use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Only 1 realm can split per turn to cap computational cost.
pub const MAX_SPLITS_PER_TURN: usize = 1;

/// Minimum number of provinces a faction must control for a split to occur.
/// With fewer than 3, there is not enough territory to meaningfully divide.
const MIN_PROVINCES_FOR_SPLIT: usize = 3;

/// Starting opinion between the original and rebel faction (hostile).
const SPLIT_HOSTILE_OPINION: i64 = -200;

/// Process a realm split for a faction undergoing a succession crisis.
///
/// The `winner_id` keeps the capital region. The second-strongest claimant
/// (first entry in `claimants` that is not `winner_id`) becomes the rebel
/// leader of a new breakaway faction.
///
/// Returns a list of `GameEvent`s describing the split.
pub fn process_realm_split(
    world: &mut GameWorld,
    faction_id: u8,
    winner_id: u32,
    claimants: &[u32],
    _rng: &mut DeterministicRng,
) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;

    // -----------------------------------------------------------------------
    // 1. Find the rebel leader (second-strongest claimant, not the winner).
    // -----------------------------------------------------------------------
    let rebel_leader_id = match claimants.iter().find(|&&cid| cid != winner_id) {
        Some(&id) => id,
        None => return events, // No rebel candidate -- cannot split.
    };

    // -----------------------------------------------------------------------
    // 2. Identify all provinces controlled by this faction.
    // -----------------------------------------------------------------------
    let faction_provinces: Vec<u16> = world
        .provinces
        .iter()
        .filter(|p| p.controller == faction_id)
        .map(|p| p.id)
        .collect();

    if faction_provinces.len() < MIN_PROVINCES_FOR_SPLIT {
        return events; // Not enough territory to split.
    }

    // -----------------------------------------------------------------------
    // 3. Find the capital province (first in realm's province list, or first
    //    faction province as fallback).
    // -----------------------------------------------------------------------
    let capital_id = world
        .realm_for_faction(faction_id)
        .and_then(|r| r.provinces.first().copied())
        .or_else(|| faction_provinces.first().copied());

    let capital_id = match capital_id {
        Some(id) => id,
        None => return events, // No provinces at all.
    };

    // -----------------------------------------------------------------------
    // 4. BFS from capital to partition provinces.
    //    Winner keeps the first ceil(total/2) provinces (closest to capital).
    //    Rebel gets the remainder.
    // -----------------------------------------------------------------------
    let winner_count = (faction_provinces.len() + 1) / 2; // ceil division
    let bfs_order = bfs_province_order(world, capital_id, &faction_provinces);

    let winner_provinces: Vec<u16> = bfs_order.iter().take(winner_count).copied().collect();
    let rebel_provinces: Vec<u16> = bfs_order.iter().skip(winner_count).copied().collect();

    if rebel_provinces.is_empty() {
        return events; // Nothing for the rebel to take.
    }

    // -----------------------------------------------------------------------
    // 5. Create new faction + realm for the rebel.
    // -----------------------------------------------------------------------
    let new_faction_id = world.factions.len() as u8;

    // Derive faction properties from the original.
    let (original_name, original_culture, original_religion, original_bonuses, original_color) = {
        let f = match world.faction(faction_id) {
            Some(f) => f,
            None => return events,
        };
        (
            f.name.clone(),
            f.culture,
            f.religion,
            f.bonuses.clone(),
            f.color_rgb,
        )
    };

    // Generate a rebel faction name from the rebel leader's dynasty, if available.
    let rebel_leader_dynasty = world.character(rebel_leader_id).map(|c| c.dynasty);
    let rebel_faction_name = rebel_leader_dynasty
        .and_then(|did| world.dynasties.iter().find(|d| d.id == did))
        .map(|d| format!("{} Rebellion", d.name))
        .unwrap_or_else(|| format!("Rebel {}", original_name));

    // Slightly mutate the color so it is visually distinct.
    let rebel_color = [
        original_color[0].wrapping_add(40),
        original_color[1].wrapping_sub(20),
        original_color[2].wrapping_add(60),
    ];

    let rebel_faction = Faction {
        id: new_faction_id,
        name: rebel_faction_name,
        alive: true,
        culture: original_culture,
        religion: original_religion,
        bonuses: original_bonuses,
        color_rgb: rebel_color,
        player_wallet: None,
    };
    world.factions.push(rebel_faction);
    world.dirty.dirty_factions.insert(new_faction_id);

    // Copy succession rule from the original dynasty (or default to Primogeniture).
    let succession_rule = rebel_leader_dynasty
        .and_then(|did| world.dynasties.iter().find(|d| d.id == did))
        .map(|d| d.succession_rule)
        .unwrap_or(SuccessionRule::Primogeniture);

    // Create a new dynasty for the rebel faction.
    let rebel_dynasty_id = world.dynasties.len() as u16;
    let rebel_leader_name = world
        .character(rebel_leader_id)
        .map(|c| c.name.clone())
        .unwrap_or_else(|| "Unknown".to_string());

    world.dynasties.push(Dynasty {
        id: rebel_dynasty_id,
        name: format!("{}'s Line", rebel_leader_name),
        founder: rebel_leader_id,
        succession_rule,
        prestige: 0,
        members: vec![rebel_leader_id],
        founded_turn: turn,
    });
    world.dirty.dirty_dynasties.insert(rebel_dynasty_id);

    // Split treasury: rebel gets a proportional share.
    let original_treasury = world
        .realm_for_faction(faction_id)
        .map(|r| r.treasury)
        .unwrap_or(FixedPoint::ZERO);
    let total_prov = faction_provinces.len() as i64;
    let rebel_prov_count = rebel_provinces.len() as i64;
    let rebel_treasury_raw = if total_prov > 0 {
        original_treasury.raw() * rebel_prov_count / total_prov
    } else {
        0
    };
    let rebel_treasury = FixedPoint::from_raw(rebel_treasury_raw);
    let winner_treasury = original_treasury - rebel_treasury;

    // Update original realm's treasury.
    if let Some(realm) = world.realm_for_faction_mut_dirty(faction_id) {
        realm.treasury = winner_treasury;
    }

    // Create rebel realm.
    let rebel_realm = Realm {
        owner_wallet: String::new(),
        faction: new_faction_id,
        ruler: rebel_leader_id,
        provinces: rebel_provinces.clone(),
        vassals: Vec::new(),
        treasury: rebel_treasury,
        cohesion: RealmCohesion {
            legitimacy: FixedPoint::from_int(300), // low -- just rebelled
            fealty: FixedPoint::from_int(400),
            clerical_favor: FixedPoint::from_int(400),
            commoner_mood: FixedPoint::from_int(350),
            regional_identity: FixedPoint::from_int(600), // high -- they identify with rebellion
        },
        age: 0,
        at_war_with: vec![faction_id], // At war with the original.
        allies: Vec::new(),
        religious_authority: FixedPoint::from_int(500),
    };
    world.realms.push(rebel_realm);
    world.dirty.dirty_realms.insert(new_faction_id);

    // -----------------------------------------------------------------------
    // 6. Reassign province controllers.
    // -----------------------------------------------------------------------
    let prov_count = world.provinces.len();
    for idx in 0..prov_count {
        if rebel_provinces.contains(&world.provinces[idx].id) {
            world.provinces[idx].controller = new_faction_id;
            world.dirty.dirty_provinces.insert(world.provinces[idx].id);
        }
    }

    // Update original realm's province list to only winner provinces.
    if let Some(realm) = world.realm_for_faction_mut_dirty(faction_id) {
        realm.provinces = winner_provinces;
        // Original faction is now at war with the rebel.
        if !realm.at_war_with.contains(&new_faction_id) {
            realm.at_war_with.push(new_faction_id);
        }
    }

    // -----------------------------------------------------------------------
    // 7. Reassign characters.
    //    Characters whose faction matches the original and who are "located"
    //    in rebel provinces join the rebel faction.
    //    Heuristic: assign characters round-robin based on index to rebel
    //    provinces. The rebel leader always switches. Other characters with
    //    roles Duke/Marshal/Courtier in excess of what the winner needs may
    //    switch. Ruler and Heir stay with the winner.
    // -----------------------------------------------------------------------
    let rebel_prov_set: std::collections::HashSet<u16> =
        rebel_provinces.iter().copied().collect();

    // First, promote the rebel leader.
    if let Some(c) = world.character_mut_dirty(rebel_leader_id) {
        c.faction = new_faction_id;
        c.role = CharacterRole::Ruler;
        c.legitimacy = FixedPoint::from_int(200);
        c.prestige += FixedPoint::from_int(100);
    }

    // Distribute other characters: those whose index mod total_prov falls
    // in the rebel province range switch to rebel faction. Skip the winner
    // and the rebel leader (already handled).
    let char_count = world.characters.len();
    let mut rebel_char_count = 0usize;
    for cidx in 0..char_count {
        let c = &world.characters[cidx];
        if !c.alive || c.faction != faction_id || c.id == winner_id || c.id == rebel_leader_id {
            continue;
        }

        // Ruler and Heir stay with the winner.
        if c.role == CharacterRole::Ruler || c.role == CharacterRole::Heir {
            continue;
        }

        // Deterministic assignment: use character ID to decide.
        // Characters with even IDs in the "rebel half" of the faction go to rebel.
        // This gives a roughly proportional split.
        let assign_to_rebel = (c.id as usize % total_prov as usize) >= winner_count;

        if assign_to_rebel {
            world.characters[cidx].faction = new_faction_id;
            world.dirty.dirty_characters.insert(world.characters[cidx].id);
            rebel_char_count += 1;
        }
    }

    // If no characters were assigned to the rebel beyond the leader, assign
    // at least one courtier so the faction is not barren.
    if rebel_char_count == 0 {
        for cidx in 0..char_count {
            let c = &world.characters[cidx];
            if c.alive
                && c.faction == faction_id
                && c.id != winner_id
                && c.id != rebel_leader_id
                && c.role == CharacterRole::Courtier
            {
                world.characters[cidx].faction = new_faction_id;
                world.dirty.dirty_characters.insert(world.characters[cidx].id);
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // 8. Reassign armies in rebel provinces to rebel faction.
    // -----------------------------------------------------------------------
    let army_count = world.armies.len();
    for aidx in 0..army_count {
        if world.armies[aidx].owner_faction == faction_id
            && rebel_prov_set.contains(&world.armies[aidx].location)
        {
            world.armies[aidx].owner_faction = new_faction_id;
            world.dirty.dirty_armies.insert(world.armies[aidx].id);
        }
    }

    // -----------------------------------------------------------------------
    // 9. Initialize diplomacy between old and new faction (hostile).
    //    Also create neutral relations with all other existing factions.
    // -----------------------------------------------------------------------
    // Hostile relation with original.
    let mut hostile_relation = DiplomaticRelation::new(
        faction_id.min(new_faction_id),
        faction_id.max(new_faction_id),
    );
    hostile_relation.at_war = true;
    hostile_relation.opinion = FixedPoint::from_int(SPLIT_HOSTILE_OPINION);
    hostile_relation.grievances.push(
        crown_ash_types::diplomacy::Grievance {
            reason: "Realm split -- civil war".to_string(),
            opinion_modifier: FixedPoint::from_int(SPLIT_HOSTILE_OPINION),
            inflicted_turn: turn,
            decay_turns_remaining: 500, // Long-lasting grudge.
        },
    );
    world.diplomacy.push(hostile_relation);
    world.dirty.mark_diplomacy(faction_id, new_faction_id);

    // Neutral relations with all other factions.
    let existing_faction_count = world.factions.len() as u8; // includes the new faction
    for fid in 0..existing_faction_count {
        if fid == faction_id || fid == new_faction_id {
            continue;
        }
        // Only add if relation does not already exist.
        let exists = world.relation(fid, new_faction_id).is_some();
        if !exists {
            world.diplomacy.push(DiplomaticRelation::new(
                fid.min(new_faction_id),
                fid.max(new_faction_id),
            ));
            world.dirty.mark_diplomacy(fid, new_faction_id);
        }
    }

    // -----------------------------------------------------------------------
    // 10. Emit RealmSplit event.
    // -----------------------------------------------------------------------
    events.push(GameEvent::RealmSplit {
        original_faction: faction_id,
        new_faction: new_faction_id,
        rebel_leader: rebel_leader_id,
        provinces_lost: rebel_provinces.len() as u32,
        turn,
    });

    events
}

/// BFS traversal from `start` across provinces in `allowed_set`.
///
/// Returns province IDs in BFS order (closest to `start` first).
/// Only traverses provinces whose ID is in `allowed_set`.
fn bfs_province_order(
    world: &GameWorld,
    start: u16,
    allowed_set: &[u16],
) -> Vec<u16> {
    let allowed: std::collections::HashSet<u16> = allowed_set.iter().copied().collect();
    let mut visited: std::collections::HashSet<u16> = std::collections::HashSet::new();
    let mut order = Vec::with_capacity(allowed_set.len());
    let mut queue = VecDeque::new();

    if allowed.contains(&start) {
        queue.push_back(start);
        visited.insert(start);
    }

    while let Some(pid) = queue.pop_front() {
        order.push(pid);

        if let Some(prov) = world.province(pid) {
            for &neighbor in &prov.neighbors {
                if allowed.contains(&neighbor) && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    // If some provinces were not reachable via BFS (disconnected graph),
    // append them at the end so nothing is lost.
    for &pid in allowed_set {
        if !visited.contains(&pid) {
            order.push(pid);
        }
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::WorldConfig;

    /// Helper: set up a world and kill the ruler of faction 0 to trigger succession,
    /// then manually invoke process_realm_split.
    fn setup_realm_split() -> (GameWorld, u32, u32, Vec<u32>) {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0xBB; 32]);

        let ruler_id = world.realms[0].ruler;

        // Remove heir so succession crisis fires.
        if let Some(ruler) = world.character_mut(ruler_id) {
            ruler.heir = None;
            ruler.alive = false;
        }

        // Find claimants for faction 0.
        let claimants: Vec<u32> = world
            .characters
            .iter()
            .filter(|c| c.alive && c.faction == 0 && c.is_adult() && c.id != ruler_id)
            .map(|c| c.id)
            .collect();

        assert!(
            claimants.len() >= 2,
            "Need at least 2 claimants for split, got {}",
            claimants.len()
        );

        let winner_id = claimants[0];
        let rebel_leader_id = claimants[1];

        (world, winner_id, rebel_leader_id, claimants)
    }

    #[test]
    fn test_realm_split_creates_new_faction() {
        let (mut world, winner_id, _rebel_id, claimants) = setup_realm_split();
        let initial_faction_count = world.factions.len();

        let mut rng = DeterministicRng::new([0xAA; 32], "split_test");
        let events = process_realm_split(&mut world, 0, winner_id, &claimants, &mut rng);

        assert!(
            !events.is_empty(),
            "Should produce at least one RealmSplit event"
        );
        assert_eq!(
            world.factions.len(),
            initial_faction_count + 1,
            "Should create one new faction"
        );
        assert!(
            world.factions.last().unwrap().alive,
            "New faction should be alive"
        );
        assert_eq!(
            world.realms.len(),
            initial_faction_count + 1,
            "Should create one new realm"
        );

        // Verify the event payload.
        let split_event = events
            .iter()
            .find(|e| matches!(e, GameEvent::RealmSplit { .. }))
            .expect("Should have a RealmSplit event");
        if let GameEvent::RealmSplit {
            original_faction,
            new_faction,
            rebel_leader,
            provinces_lost,
            ..
        } = split_event
        {
            assert_eq!(*original_faction, 0);
            assert_eq!(*new_faction, initial_faction_count as u8);
            assert!(*provinces_lost > 0, "Rebel should get at least 1 province");
            assert_ne!(*rebel_leader, winner_id, "Rebel leader should not be the winner");
        }
    }

    #[test]
    fn test_provinces_partitioned() {
        let (mut world, winner_id, _rebel_id, claimants) = setup_realm_split();

        // Count original provinces for faction 0.
        let original_count = world
            .provinces
            .iter()
            .filter(|p| p.controller == 0)
            .count();
        assert!(
            original_count >= MIN_PROVINCES_FOR_SPLIT,
            "Faction 0 needs >= {} provinces, has {}",
            MIN_PROVINCES_FOR_SPLIT,
            original_count
        );

        let mut rng = DeterministicRng::new([0xAA; 32], "split_test");
        let _events = process_realm_split(&mut world, 0, winner_id, &claimants, &mut rng);

        let new_faction_id = (world.factions.len() - 1) as u8;

        // Winner provinces.
        let winner_provinces: Vec<u16> = world
            .provinces
            .iter()
            .filter(|p| p.controller == 0)
            .map(|p| p.id)
            .collect();

        // Rebel provinces.
        let rebel_provinces: Vec<u16> = world
            .provinces
            .iter()
            .filter(|p| p.controller == new_faction_id)
            .map(|p| p.id)
            .collect();

        assert!(
            !winner_provinces.is_empty(),
            "Winner should keep at least 1 province"
        );
        assert!(
            !rebel_provinces.is_empty(),
            "Rebel should get at least 1 province"
        );
        assert_eq!(
            winner_provinces.len() + rebel_provinces.len(),
            original_count,
            "Total provinces should be conserved"
        );

        // Winner keeps the majority (ceil(total/2)).
        let expected_winner = (original_count + 1) / 2;
        assert_eq!(
            winner_provinces.len(),
            expected_winner,
            "Winner should keep ceil(total/2) provinces"
        );

        // Verify realm province lists match controllers.
        let winner_realm = world.realm_for_faction(0).unwrap();
        assert_eq!(
            winner_realm.provinces.len(),
            winner_provinces.len(),
            "Winner realm province list should match controller data"
        );
        let rebel_realm = world.realm_for_faction(new_faction_id).unwrap();
        assert_eq!(
            rebel_realm.provinces.len(),
            rebel_provinces.len(),
            "Rebel realm province list should match controller data"
        );
    }

    #[test]
    fn test_characters_reassigned() {
        let (mut world, winner_id, _rebel_id, claimants) = setup_realm_split();

        let mut rng = DeterministicRng::new([0xAA; 32], "split_test");
        let _events = process_realm_split(&mut world, 0, winner_id, &claimants, &mut rng);

        let new_faction_id = (world.factions.len() - 1) as u8;

        // The rebel leader should be in the new faction.
        let rebel_leader_id = claimants[1]; // second-strongest
        let rebel_leader = world.character(rebel_leader_id).unwrap();
        assert_eq!(
            rebel_leader.faction, new_faction_id,
            "Rebel leader should be in the new faction"
        );
        assert_eq!(
            rebel_leader.role,
            CharacterRole::Ruler,
            "Rebel leader should be the ruler of the new faction"
        );

        // At least one character (the rebel leader) should be in the new faction.
        let rebel_chars: Vec<_> = world
            .characters
            .iter()
            .filter(|c| c.alive && c.faction == new_faction_id)
            .collect();
        assert!(
            !rebel_chars.is_empty(),
            "New faction should have at least one character"
        );

        // The winner should still be in faction 0.
        let winner = world.character(winner_id).unwrap();
        assert_eq!(
            winner.faction, 0,
            "Winner should remain in the original faction"
        );
    }

    #[test]
    fn test_armies_reassigned() {
        let (mut world, winner_id, _rebel_id, claimants) = setup_realm_split();

        // Place an army in one of faction 0's provinces that will become rebel territory.
        // First, find which provinces faction 0 owns.
        let faction_provinces: Vec<u16> = world
            .provinces
            .iter()
            .filter(|p| p.controller == 0)
            .map(|p| p.id)
            .collect();

        // The rebel will get the provinces furthest from capital (last in BFS order).
        // Place an army in the last province.
        let last_province = *faction_provinces.last().unwrap();
        let army_id = world.alloc_army_id();
        world.armies.push(crown_ash_types::Army {
            id: army_id,
            owner_faction: 0,
            commander: None,
            troops: crown_ash_types::province::Troops {
                levy: 200,
                men_at_arms: 50,
                knights: 5,
            },
            morale: FixedPoint::from_int(700),
            location: last_province,
            destination: None,
            movement_queue: Vec::new(),
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
            siege: None,
        });

        let mut rng = DeterministicRng::new([0xAA; 32], "split_test");
        let _events = process_realm_split(&mut world, 0, winner_id, &claimants, &mut rng);

        let new_faction_id = (world.factions.len() - 1) as u8;

        // Check if the province is now rebel territory.
        let prov = world.province(last_province).unwrap();
        if prov.controller == new_faction_id {
            // Army should have been reassigned to the rebel faction.
            let army = world.army(army_id).unwrap();
            assert_eq!(
                army.owner_faction, new_faction_id,
                "Army in rebel territory should switch to rebel faction"
            );
        }
        // If the province stayed with the winner (BFS ordering), the army stays too.
        // Either way, the army's faction should match the province controller.
        let army = world.army(army_id).unwrap();
        let prov_controller = world.province(army.location).unwrap().controller;
        assert_eq!(
            army.owner_faction, prov_controller,
            "Army faction should match the province controller"
        );
    }

    #[test]
    fn test_diplomacy_initialized() {
        let (mut world, winner_id, _rebel_id, claimants) = setup_realm_split();
        let initial_diplomacy_count = world.diplomacy.len();

        let mut rng = DeterministicRng::new([0xAA; 32], "split_test");
        let _events = process_realm_split(&mut world, 0, winner_id, &claimants, &mut rng);

        let new_faction_id = (world.factions.len() - 1) as u8;

        // Should have a hostile relation between original and rebel.
        let rel = world
            .relation(0, new_faction_id)
            .expect("Should have diplomatic relation between original and rebel");
        assert!(rel.at_war, "Original and rebel should be at war");
        assert!(
            rel.opinion.raw() <= SPLIT_HOSTILE_OPINION * 1000,
            "Opinion should be hostile (<= {}), got {}",
            SPLIT_HOSTILE_OPINION,
            rel.opinion.raw()
        );

        // Should have neutral relations with all other factions.
        let _other_faction_count = world.factions.len() - 2; // exclude faction 0 and new
        let new_relations = world.diplomacy.len() - initial_diplomacy_count;
        // 1 hostile (with original) + N neutral (with others)
        assert!(
            new_relations >= 1,
            "Should have at least the hostile relation, got {}",
            new_relations
        );

        // Check that the original realm is also at war with the rebel.
        let original_realm = world.realm_for_faction(0).unwrap();
        assert!(
            original_realm.at_war_with.contains(&new_faction_id),
            "Original realm should be at war with rebel"
        );
    }

    #[test]
    fn test_small_realm_no_split() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0xCC; 32]);

        // Give faction 2 (Ember Church) only 2 provinces by reassigning one.
        // Ember Church has provinces 11, 12, 13 by default.
        // Reassign province 13 to faction 1.
        if let Some(p) = world.province_mut(13) {
            p.controller = 1;
        }
        // Update realm province lists.
        if let Some(r) = world.realm_for_faction_mut(2) {
            r.provinces.retain(|&pid| pid != 13);
        }
        if let Some(r) = world.realm_for_faction_mut(1) {
            r.provinces.push(13);
        }

        // Verify faction 2 now has only 2 provinces.
        let f2_province_count = world
            .provinces
            .iter()
            .filter(|p| p.controller == 2)
            .count();
        assert_eq!(f2_province_count, 2, "Faction 2 should have 2 provinces");

        // Find claimants for faction 2.
        let claimants: Vec<u32> = world
            .characters
            .iter()
            .filter(|c| c.alive && c.faction == 2 && c.is_adult())
            .map(|c| c.id)
            .collect();

        let winner_id = claimants[0];

        let mut rng = DeterministicRng::new([0xDD; 32], "split_test");
        let initial_faction_count = world.factions.len();

        let events = process_realm_split(&mut world, 2, winner_id, &claimants, &mut rng);

        assert!(
            events.is_empty(),
            "Should not split a realm with only 2 provinces"
        );
        assert_eq!(
            world.factions.len(),
            initial_faction_count,
            "No new faction should be created"
        );
    }
}
