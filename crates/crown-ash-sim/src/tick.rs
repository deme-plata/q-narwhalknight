//! Tick — the 10-step turn pipeline.
//!
//! Each game turn processes the following phases in order:
//!
//! 1. Age characters (1 day per turn; increment year every 365 turns)
//! 2. Process queued player actions
//! 3. Resolve battles (armies in same province, factions at war)
//! 3b. Process sieges (fortified province captures)
//! 4. Move armies (destination -> location)
//! 5. Economy (tax collection, improvement effects)
//! 5b. Trade (inter-province trade routes, prosperity bonuses, income)
//! 6. Unrest calculation
//! 7. Cohesion decay
//! 7a. Religion (authority, conversion progress, religious events)
//! 7b. Intrigue (advance plots, check discovery, execute)
//! 7c. Diplomacy (tribute, treaty expiry, grievance decay, coalitions)
//! 8. Random events (plague, famine, harvest, rebellion)
//! 9. Succession check
//! 9b. Lifecycle cleanup (tombstone dead characters, enforce army caps)
//! 10. Build TurnSummary and return
//!
//! All state transitions are deterministic given the same `block_hash`.

use crown_ash_types::{
    FixedPoint, GameEvent, TurnSummary,
};
use crate::ai;
use crate::birth;
use crate::cohesion;
use crate::combat;
use crate::diplomacy;
use crate::economy;
use crate::education;
use crate::events;
use crate::intrigue;
use crate::lifecycle;
use crate::process_action_internal;
use crate::random::DeterministicRng;
use crate::relationships;
use crate::religion;
use crate::succession;
use crate::trade;
use crate::world_state::GameWorld;

/// Execute one game turn.
///
/// `block_hash` is the Q-NarwhalKnight block hash that seeds all RNG for this turn.
/// Returns a `TurnSummary` describing what happened.
pub fn tick(world: &mut GameWorld, block_hash: [u8; 32]) -> TurnSummary {
    // Verify sim version before processing any state changes.
    assert_eq!(
        world.meta.sim_version,
        crown_ash_types::SIM_VERSION,
        "Cannot process tick: sim version mismatch ({} vs {})",
        world.meta.sim_version,
        crown_ash_types::SIM_VERSION,
    );

    // Clear dirty tracker from previous tick.
    world.clear_dirty();

    let mut all_events: Vec<GameEvent> = Vec::new();
    world.meta.turn += 1;
    world.mark_meta_dirty();
    let turn = world.meta.turn;

    // -----------------------------------------------------------------------
    // Step 1: Age characters and natural death checks.
    // -----------------------------------------------------------------------
    let mut rng_aging = DeterministicRng::new(block_hash, "aging");
    let death_events = succession::age_and_death_check(world, &mut rng_aging);
    all_events.extend(death_events);

    // -----------------------------------------------------------------------
    // Step 1b: Birth — married couples may produce children.
    // -----------------------------------------------------------------------
    let mut rng_birth = DeterministicRng::new(block_hash, "birth");
    let birth_events = birth::process_births(world, &mut rng_birth);
    all_events.extend(birth_events);

    // -----------------------------------------------------------------------
    // Step 1c: Education — age-gated stat growth and graduation.
    // -----------------------------------------------------------------------
    let mut rng_education = DeterministicRng::new(block_hash, "education");
    let education_events = education::process_education(world, &mut rng_education);
    all_events.extend(education_events);

    // -----------------------------------------------------------------------
    // Step 2: Process queued player actions.
    // -----------------------------------------------------------------------
    let queued = std::mem::take(&mut world.action_queue);
    for qa in queued {
        let _ = process_action_internal(world, &qa.wallet, qa.action);
    }

    // Also process AI actions for NPC factions.
    let mut rng_ai = DeterministicRng::new(block_hash, "ai");
    let ai_actions = ai::generate_ai_actions(world, &mut rng_ai);
    for (faction_id, action) in ai_actions {
        // Find the wallet for this faction (empty string for NPC).
        let wallet = world.realm_for_faction(faction_id)
            .map(|r| r.owner_wallet.clone())
            .unwrap_or_default();
        let _ = process_action_internal(world, &wallet, action);
    }

    // -----------------------------------------------------------------------
    // Step 3: Resolve battles.
    // -----------------------------------------------------------------------
    let mut rng_combat = DeterministicRng::new(block_hash, "combat");
    let battle_results = combat::resolve_battles(world, &mut rng_combat);
    for br in &battle_results {
        all_events.push(GameEvent::Battle(br.clone()));
    }

    // Check province captures after battles (unfortified provinces only).
    let captures = combat::check_province_captures(world);
    for (pid, old_ctrl, new_ctrl) in &captures {
        all_events.push(GameEvent::ProvinceConquered {
            province: *pid,
            old_controller: *old_ctrl,
            new_controller: *new_ctrl,
            turn,
        });
        // Apply cohesion penalty for conquest.
        cohesion::apply_conquest_penalty(world, *new_ctrl);
    }

    // Step 3b: Process sieges (fortified provinces).
    let siege_events = combat::process_sieges(world);
    for ev in &siege_events {
        if let GameEvent::SiegeCompleted { province, old_controller, new_controller, .. } = ev {
            all_events.push(GameEvent::ProvinceConquered {
                province: *province,
                old_controller: *old_controller,
                new_controller: *new_controller,
                turn,
            });
            cohesion::apply_conquest_penalty(world, *new_controller);
        }
    }
    all_events.extend(siege_events);

    // -----------------------------------------------------------------------
    // Step 4: Move armies (destination -> location).
    // -----------------------------------------------------------------------
    move_armies(world);

    // -----------------------------------------------------------------------
    // Step 5: Economy (tax collection, improvement effects).
    // -----------------------------------------------------------------------
    economy::run_economy(world);

    // -----------------------------------------------------------------------
    // Step 5b: Trade (inter-province trade routes, prosperity, income).
    // -----------------------------------------------------------------------
    let mut rng_trade = DeterministicRng::new(block_hash, "trade");
    let trade_events = trade::process_trade(world, &mut rng_trade);
    all_events.extend(trade_events);

    // -----------------------------------------------------------------------
    // Step 6: Unrest calculation.
    // -----------------------------------------------------------------------
    update_unrest(world);

    // -----------------------------------------------------------------------
    // Step 7: Cohesion decay.
    // -----------------------------------------------------------------------
    cohesion::update_cohesion(world);

    // -----------------------------------------------------------------------
    // Step 7a: Religion — authority, conversion progress, cohesion effects.
    // -----------------------------------------------------------------------
    religion::update_religious_authority(world);
    let conversion_events = religion::process_conversions(world);
    all_events.extend(conversion_events);
    religion::authority_cohesion_effect(world);

    let mut rng_religion = DeterministicRng::new(block_hash, "religion");
    let religion_events = religion::roll_religious_events(world, &mut rng_religion);
    all_events.extend(religion_events);

    // -----------------------------------------------------------------------
    // Step 7b: Intrigue — advance plots, check discovery, execute.
    // -----------------------------------------------------------------------
    let mut rng_intrigue = DeterministicRng::new(block_hash, "intrigue");
    let intrigue_events = intrigue::process_intrigue(world, &mut rng_intrigue);
    all_events.extend(intrigue_events);

    // -----------------------------------------------------------------------
    // Step 7c: Character relationships — friendships, rivalries, alliances.
    // -----------------------------------------------------------------------
    let relationship_events = relationships::process_relationships(world);
    all_events.extend(relationship_events);

    // -----------------------------------------------------------------------
    // Step 7d: Diplomacy — tribute, treaty expiry, grievance decay, coalitions.
    // -----------------------------------------------------------------------
    let mut rng_diplomacy = DeterministicRng::new(block_hash, "diplomacy");
    let diplomacy_events = diplomacy::process_diplomacy(world, &mut rng_diplomacy);
    all_events.extend(diplomacy_events);

    // -----------------------------------------------------------------------
    // Step 8: Random events (plague, famine, harvest, rebellion).
    // -----------------------------------------------------------------------
    let mut rng_events = DeterministicRng::new(block_hash, "events");
    let random_events = events::roll_events(world, &mut rng_events);
    all_events.extend(random_events);

    // Decay scars and grudges.
    events::decay_scars(world);

    // -----------------------------------------------------------------------
    // Step 8b: Clamp province values (unrest, prosperity) after all modifiers.
    // -----------------------------------------------------------------------
    // Random events (famine, plague) can push unrest/prosperity outside [0, 1000]
    // after the step-6 clamp. This sweep enforces hard bounds at end of pipeline.
    clamp_province_values(world);

    // -----------------------------------------------------------------------
    // Step 9: Succession check.
    // -----------------------------------------------------------------------
    let mut rng_succession = DeterministicRng::new(block_hash, "succession");
    let succession_events = succession::check_succession(world, &mut rng_succession);
    all_events.extend(succession_events);

    // Check for eliminated factions.
    check_faction_elimination(world, &mut all_events, turn);

    // Increment realm ages.
    let realm_count = world.realms.len();
    for idx in 0..realm_count {
        world.realms[idx].age += 1;
        world.dirty.dirty_realms.insert(world.realms[idx].faction);
    }

    // -----------------------------------------------------------------------
    // Step 9b: Lifecycle cleanup — tombstone dead characters, enforce army caps.
    // -----------------------------------------------------------------------
    let lifecycle_events = lifecycle::process_lifecycle(world, turn);
    all_events.extend(lifecycle_events);

    // -----------------------------------------------------------------------
    // Post-tick invariant check (debug/test builds only).
    // -----------------------------------------------------------------------
    world.assert_invariants();

    // -----------------------------------------------------------------------
    // Step 10: Build TurnSummary.
    // -----------------------------------------------------------------------
    let active_factions = world.factions.iter().filter(|f| f.alive).count() as u8;
    let total_armies = world.armies.len() as u32;
    let total_population: u64 = world.provinces.iter()
        .map(|p| p.population as u64)
        .sum();

    TurnSummary {
        turn,
        block_height: 0, // Set by caller.
        events: all_events,
        active_factions,
        total_armies,
        total_population,
    }
}

/// Move armies that have a destination or movement queue.
///
/// Multi-hop: pops the front of `movement_queue` each turn.
/// Single-hop: consumes `destination` in one turn.
fn move_armies(world: &mut GameWorld) {
    let army_count = world.armies.len();
    for idx in 0..army_count {
        // Besieging armies cannot move (siege cancelled automatically if they try).
        if world.armies[idx].siege.is_some() {
            continue;
        }

        // Multi-hop path takes priority over single-hop destination.
        if !world.armies[idx].movement_queue.is_empty() {
            let next = world.armies[idx].movement_queue.remove(0);
            world.armies[idx].location = next;
            // If we also had a single-hop destination, clear it.
            world.armies[idx].destination = None;
            world.dirty.dirty_armies.insert(world.armies[idx].id);
        } else if let Some(dest) = world.armies[idx].destination.take() {
            world.armies[idx].location = dest;
            world.dirty.dirty_armies.insert(world.armies[idx].id);
        }
    }
}

/// Update unrest for all provinces.
///
/// Unrest increases from:
/// - High tax rate (> 30%)
/// - Low prosperity (< 300)
/// - War (province controller at war)
/// - Province not of controller's culture/religion
///
/// Unrest decreases from:
/// - Low tax rate (< 20%)
/// - High prosperity (> 600)
/// - Temple improvement (+religious unity)
/// - Garrison presence
fn update_unrest(world: &mut GameWorld) {
    let province_count = world.provinces.len();

    for idx in 0..province_count {
        let faction_id = world.provinces[idx].controller;

        // Check if faction is at war.
        let at_war = world.realms.iter()
            .find(|r| r.faction == faction_id)
            .map(|r| !r.at_war_with.is_empty())
            .unwrap_or(false);

        let faction_culture = world.faction(faction_id)
            .map(|f| f.culture);
        let faction_religion = world.faction(faction_id)
            .map(|f| f.religion);

        let prov = &world.provinces[idx];

        let mut unrest_delta = FixedPoint::ZERO;

        // High tax → +unrest.
        if prov.tax_rate.raw() > 300 {
            unrest_delta += FixedPoint::from_raw(
                (prov.tax_rate.raw() - 300) / 10
            );
        }

        // Low tax → -unrest.
        if prov.tax_rate.raw() < 200 {
            unrest_delta -= FixedPoint::from_raw(
                (200 - prov.tax_rate.raw()) / 10
            );
        }

        // Low prosperity → +unrest.
        if prov.prosperity.raw() < 300_000 {
            unrest_delta += FixedPoint::from_int(5);
        }

        // High prosperity → -unrest.
        if prov.prosperity.raw() > 600_000 {
            unrest_delta -= FixedPoint::from_int(3);
        }

        // War → +unrest.
        if at_war {
            unrest_delta += FixedPoint::from_int(10);
        }

        // Cultural mismatch → +unrest.
        if faction_culture.map_or(false, |fc| fc != prov.culture) {
            unrest_delta += FixedPoint::from_int(5);
        }

        // Religious mismatch → +unrest.
        if faction_religion.map_or(false, |fr| fr != prov.religion) {
            unrest_delta += FixedPoint::from_int(5);
        }

        // Temple → -unrest.
        if prov.improvements.contains(&crown_ash_types::Improvement::Temple) {
            unrest_delta -= FixedPoint::from_int(5);
        }

        // Garrison → -unrest (1 per 100 garrison troops).
        let garrison_total = prov.garrison.total();
        if garrison_total > 0 {
            let garrison_effect = (garrison_total as i64 / 100).min(20);
            unrest_delta -= FixedPoint::from_int(garrison_effect);
        }

        // Natural decay toward 100.
        let equilibrium = FixedPoint::from_int(100);
        let current = world.provinces[idx].unrest;
        if current > equilibrium {
            unrest_delta -= FixedPoint::from_int(2); // slow decay
        }

        // Apply delta.
        world.provinces[idx].unrest += unrest_delta;

        // Clamp to 0..1000.
        world.provinces[idx].unrest = world.provinces[idx].unrest
            .clamp(FixedPoint::ZERO, FixedPoint::from_int(1000));

        // Mark province dirty — unrest changed.
        world.dirty.dirty_provinces.insert(world.provinces[idx].id);
    }
}

/// Enforce hard bounds on all province FixedPoint values.
///
/// Called after all modifier steps (economy, events, intrigue) to guarantee
/// invariants hold before the post-tick assertion.
fn clamp_province_values(world: &mut GameWorld) {
    let zero = FixedPoint::ZERO;
    let max = FixedPoint::from_int(1000);
    let province_count = world.provinces.len();

    for idx in 0..province_count {
        let p = &mut world.provinces[idx];
        let old_unrest = p.unrest;
        let old_prosperity = p.prosperity;

        p.unrest = p.unrest.clamp(zero, max);
        p.prosperity = p.prosperity.clamp(zero, max);

        if p.unrest != old_unrest || p.prosperity != old_prosperity {
            world.dirty.dirty_provinces.insert(p.id);
        }
    }
}

/// Check for factions that have lost all provinces.
fn check_faction_elimination(world: &mut GameWorld, events: &mut Vec<GameEvent>, turn: u32) {
    let faction_count = world.factions.len();
    for idx in 0..faction_count {
        if !world.factions[idx].alive {
            continue;
        }
        let fid = world.factions[idx].id;
        let province_count = world.provinces.iter()
            .filter(|p| p.controller == fid)
            .count();
        if province_count == 0 {
            world.factions[idx].alive = false;
            // Mark faction dirty — eliminated.
            world.dirty.dirty_factions.insert(fid);
            events.push(GameEvent::FactionEliminated {
                faction: fid,
                turn,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::WorldConfig;

    #[test]
    fn tick_advances_turn() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);

        assert_eq!(world.meta.turn, 0);

        let summary = tick(&mut world, [0x01; 32]);
        assert_eq!(summary.turn, 1);
        assert_eq!(world.meta.turn, 1);
    }

    #[test]
    fn tick_is_deterministic() {
        let config = WorldConfig::default();
        let block = [0xAB; 32];

        let mut world1 = init_world(&config, [0x42; 32]);
        let mut world2 = init_world(&config, [0x42; 32]);

        let summary1 = tick(&mut world1, block);
        let summary2 = tick(&mut world2, block);

        assert_eq!(summary1.turn, summary2.turn);
        assert_eq!(summary1.events.len(), summary2.events.len());
        assert_eq!(summary1.active_factions, summary2.active_factions);
        assert_eq!(summary1.total_population, summary2.total_population);
    }

    #[test]
    fn multiple_ticks_produce_valid_state() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);

        for i in 0u8..50 {
            let block = [i; 32];
            let summary = tick(&mut world, block);
            assert_eq!(summary.turn, (i as u32) + 1);
            assert!(summary.active_factions > 0, "At least one faction should be alive");
            assert!(summary.total_population > 0, "Total population should be positive");
        }
    }

    #[test]
    fn army_movement_completes_in_one_turn() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);

        // Create an army with a destination.
        use crown_ash_types::{Army, FixedPoint};
        use crown_ash_types::province::Troops;

        let aid = world.alloc_army_id();
        world.armies.push(Army {
            id: aid,
            owner_faction: 0,
            commander: None,
            troops: Troops { levy: 100, men_at_arms: 20, knights: 5 },
            morale: FixedPoint::from_int(800),
            location: 7,         // Ashenmere
            destination: Some(9), // Embervale (adjacent)
            movement_queue: Vec::new(),
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
            siege: None,
        });

        tick(&mut world, [0x01; 32]);

        let army = world.army(aid).unwrap();
        assert_eq!(army.location, 9, "Army should have moved to destination");
        assert!(army.destination.is_none(), "Destination should be cleared");
    }
}
