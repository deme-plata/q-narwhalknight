//! Combat resolution — auto-resolved battles between armies.
//!
//! ## Formula
//!
//! ```text
//! attack_power = levy * 1 + men_at_arms * 3 + knights * 10
//! defense_power = attack_power * (1000 + terrain_bonus + fort_bonus) / 1000
//! commander_bonus = character.martial * 10
//! random_factor = range(850, 1150) from block hash
//! final = (power + cmd_bonus) * random_factor / 1000
//! casualties proportional to power ratio
//! morale collapse if casualties > 40%
//! ```
//!
//! All math uses `FixedPoint` — no floating point.

use crown_ash_types::{Army, ArmyId, BattleResult, FixedPoint};
use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Fortification defense bonus per level (100 = +10% per level).
const FORT_BONUS_PER_LEVEL: i64 = 100;

/// Casualty ratio threshold for morale collapse (400 = 40%).
const MORALE_COLLAPSE_THRESHOLD: FixedPoint = FixedPoint::from_raw(400);

/// Minimum morale after collapse.
const MORALE_COLLAPSE_FLOOR: FixedPoint = FixedPoint::from_raw(100);

/// Resolve all battles in the world for this turn.
///
/// A battle occurs when armies of factions at war occupy the same province
/// (and are not currently moving).
///
/// Capped at [`MAX_BATTLES_PER_TURN`](crown_ash_types::MAX_BATTLES_PER_TURN)
/// to prevent gas exhaustion on pathological turns.
pub fn resolve_battles(world: &mut GameWorld, rng: &mut DeterministicRng) -> Vec<BattleResult> {
    let mut results = Vec::new();

    // Collect potential battle sites: provinces where multiple factions have armies.
    let province_ids: Vec<u16> = world.provinces.iter().map(|p| p.id).collect();

    for &pid in &province_ids {
        if results.len() >= crown_ash_types::MAX_BATTLES_PER_TURN {
            break; // Work cap reached — remaining battles deferred to next turn.
        }
        loop {
            if results.len() >= crown_ash_types::MAX_BATTLES_PER_TURN {
                break;
            }

            // Find two armies in this province from factions at war.
            let battle_pair = find_battle_pair(world, pid);
            let (attacker_id, defender_id) = match battle_pair {
                Some(pair) => pair,
                None => break,
            };

            let result = resolve_single_battle(world, attacker_id, defender_id, pid, rng);
            results.push(result);

            // Remove destroyed armies and track removals.
            let before_ids: Vec<u32> = world.armies.iter()
                .filter(|a| a.troops.total() == 0)
                .map(|a| a.id)
                .collect();
            for &removed_id in &before_ids {
                world.dirty.armies_removed.push(removed_id);
            }
            world.armies.retain(|a| a.troops.total() > 0);
        }
    }

    results
}

/// Find two armies in the same province belonging to factions at war.
fn find_battle_pair(world: &GameWorld, province_id: u16) -> Option<(ArmyId, ArmyId)> {
    let stationary_armies: Vec<&Army> = world.armies.iter()
        .filter(|a| a.location == province_id && !a.is_moving() && a.troops.total() > 0)
        .collect();

    for (i, a) in stationary_armies.iter().enumerate() {
        for b in stationary_armies.iter().skip(i + 1) {
            if world.at_war(a.owner_faction, b.owner_faction) {
                return Some((a.id, b.id));
            }
        }
    }
    None
}

/// Resolve a single battle between two armies.
fn resolve_single_battle(
    world: &mut GameWorld,
    attacker_id: ArmyId,
    defender_id: ArmyId,
    province_id: u16,
    rng: &mut DeterministicRng,
) -> BattleResult {
    // Gather data (we need immutable access first, then mutable).
    let att_power = world.army(attacker_id).map(|a| a.attack_power()).unwrap_or(FixedPoint::ZERO);
    let def_power = world.army(defender_id).map(|a| a.attack_power()).unwrap_or(FixedPoint::ZERO);
    let att_commander = world.army(attacker_id).and_then(|a| a.commander);
    let def_commander = world.army(defender_id).and_then(|a| a.commander);
    let att_total = world.army(attacker_id).map(|a| a.total_soldiers()).unwrap_or(0);
    let def_total = world.army(defender_id).map(|a| a.total_soldiers()).unwrap_or(0);

    // Terrain and fortification bonuses apply to defender.
    let terrain_bonus = world.province(province_id)
        .map(|p| p.terrain.defense_bonus())
        .unwrap_or(FixedPoint::ZERO);
    let fort_bonus = world.province(province_id)
        .map(|p| FixedPoint::from_raw(p.fortification as i64 * FORT_BONUS_PER_LEVEL))
        .unwrap_or(FixedPoint::ZERO);

    // defense_power = attack_power * (1000 + terrain_bonus + fort_bonus) / 1000
    let def_modified = FixedPoint::from_raw(
        def_power.raw() * (1000 + terrain_bonus.raw() + fort_bonus.raw()) / 1000
    );

    // Commander bonuses: martial * 10.
    let att_cmd_bonus = att_commander
        .and_then(|cid| world.character(cid))
        .map(|c| c.effective_stats().martial * 10)
        .unwrap_or(FixedPoint::ZERO);
    let def_cmd_bonus = def_commander
        .and_then(|cid| world.character(cid))
        .map(|c| c.effective_stats().martial * 10)
        .unwrap_or(FixedPoint::ZERO);

    // Random factors (850-1150).
    let att_random = FixedPoint::from_raw(rng.range(850, 1150));
    let def_random = FixedPoint::from_raw(rng.range(850, 1150));

    // Final power: (power + cmd_bonus) * random_factor / 1000
    let att_final_raw = (att_power.raw() + att_cmd_bonus.raw()) * att_random.raw() / 1000;
    let def_final_raw = (def_modified.raw() + def_cmd_bonus.raw()) * def_random.raw() / 1000;

    let att_final = FixedPoint::from_raw(att_final_raw);
    let def_final = FixedPoint::from_raw(def_final_raw);

    let attacker_won = att_final.raw() >= def_final.raw();

    // Casualties proportional to enemy power.
    // Winner takes 20-30% casualties, loser takes 40-60%.
    let (att_casualty_rate, def_casualty_rate) = if attacker_won {
        // Attacker won: attacker loses less, defender loses more.
        let ratio = if att_final.raw() > 0 {
            def_final.raw() * 1000 / att_final.raw()
        } else {
            500
        };
        // Attacker casualties: 200-300 scaled by ratio.
        let att_rate = (200 * ratio / 1000).max(50).min(400);
        // Defender casualties: 400-600.
        let def_rate = (400 + (1000 - ratio) * 200 / 1000).max(300).min(700);
        (att_rate, def_rate)
    } else {
        let ratio = if def_final.raw() > 0 {
            att_final.raw() * 1000 / def_final.raw()
        } else {
            500
        };
        let def_rate = (200 * ratio / 1000).max(50).min(400);
        let att_rate = (400 + (1000 - ratio) * 200 / 1000).max(300).min(700);
        (att_rate, def_rate)
    };

    let att_casualties = ((att_total as i64 * att_casualty_rate) / 1000).max(0) as u32;
    let def_casualties = ((def_total as i64 * def_casualty_rate) / 1000).max(0) as u32;

    // Apply casualties to armies (marks them dirty).
    apply_casualties(world, attacker_id, att_casualties);
    apply_casualties(world, defender_id, def_casualties);

    // Morale collapse check: if casualties > 40% of original, morale tanks.
    let att_collapse_threshold = (att_total as i64 * MORALE_COLLAPSE_THRESHOLD.raw()) / 1000;
    let def_collapse_threshold = (def_total as i64 * MORALE_COLLAPSE_THRESHOLD.raw()) / 1000;

    if att_casualties as i64 > att_collapse_threshold {
        if let Some(army) = world.army_mut_dirty(attacker_id) {
            if army.morale > MORALE_COLLAPSE_FLOOR {
                army.morale = MORALE_COLLAPSE_FLOOR;
            }
        }
    }
    if def_casualties as i64 > def_collapse_threshold {
        if let Some(army) = world.army_mut_dirty(defender_id) {
            if army.morale > MORALE_COLLAPSE_FLOOR {
                army.morale = MORALE_COLLAPSE_FLOOR;
            }
        }
    }

    // Losing army retreats: set destination to a random neighboring province owned by their faction.
    let loser_id = if attacker_won { defender_id } else { attacker_id };
    if let Some(loser) = world.army(loser_id) {
        let loser_faction = loser.owner_faction;
        let location = loser.location;
        let neighbors = world.province(location)
            .map(|p| p.neighbors.clone())
            .unwrap_or_default();
        // Prefer a province the loser controls.
        let retreat_target = neighbors.iter()
            .find(|&&n| world.province(n).map_or(false, |p| p.controller == loser_faction))
            .or_else(|| neighbors.first())
            .copied();
        if let Some(target) = retreat_target {
            if let Some(army) = world.army_mut_dirty(loser_id) {
                army.destination = Some(target);
            }
        }
    }

    BattleResult {
        attacker_army: attacker_id,
        defender_army: Some(defender_id),
        province: province_id,
        attacker_casualties: att_casualties,
        defender_casualties: def_casualties,
        attacker_won,
        random_factor: att_random,
        turn: world.meta.turn,
    }
}

/// Remove casualties from an army proportionally across troop types.
fn apply_casualties(world: &mut GameWorld, army_id: ArmyId, total_casualties: u32) {
    if let Some(army) = world.army_mut_dirty(army_id) {
        let total = army.troops.total();
        if total == 0 || total_casualties == 0 {
            return;
        }
        // Distribute proportionally.
        let levy_share = (total_casualties as u64 * army.troops.levy as u64 / total as u64) as u32;
        let maa_share = (total_casualties as u64 * army.troops.men_at_arms as u64 / total as u64) as u32;
        let knight_share = total_casualties.saturating_sub(levy_share).saturating_sub(maa_share);

        army.troops.levy = army.troops.levy.saturating_sub(levy_share);
        army.troops.men_at_arms = army.troops.men_at_arms.saturating_sub(maa_share);
        army.troops.knights = army.troops.knights.saturating_sub(knight_share.min(u16::MAX as u32) as u16);
    }
}

/// Province capture: if an army occupies an unfortified enemy province
/// unopposed, capture it immediately.  Fortified provinces require a siege
/// (see [`process_sieges`]).
pub fn check_province_captures(world: &mut GameWorld) -> Vec<(u16, u8, u8)> {
    let mut captures = Vec::new();

    let province_ids: Vec<u16> = world.provinces.iter().map(|p| p.id).collect();

    for &pid in &province_ids {
        let (controller, fortification) = match world.province(pid) {
            Some(p) => (p.controller, p.fortification),
            None => continue,
        };

        // Find any enemy army stationary in this province.
        let enemy_army = world.armies.iter()
            .find(|a| {
                a.location == pid
                    && !a.is_moving()
                    && a.owner_faction != controller
                    && a.troops.total() > 0
                    && a.siege.is_none() // not already besieging
                    && world.at_war(a.owner_faction, controller)
            });

        if let Some(army) = enemy_army {
            // Check if there are any defender armies present.
            let defenders_present = world.armies.iter().any(|a| {
                a.location == pid
                    && !a.is_moving()
                    && a.owner_faction == controller
                    && a.troops.total() > 0
            });

            // Check garrison strength.
            let garrison_strength = world.province(pid)
                .map(|p| p.garrison.total())
                .unwrap_or(0);

            if !defenders_present && garrison_strength == 0 && fortification == 0 {
                // No walls, no defenders — instant capture.
                let new_controller = army.owner_faction;
                captures.push((pid, controller, new_controller));
            }
            // Fortified provinces are handled by process_sieges().
        }
    }

    // Apply captures.
    for &(pid, old_ctrl, new_ctrl) in &captures {
        apply_province_capture(world, pid, old_ctrl, new_ctrl);
    }

    captures
}

// ─── Siege Processing ────────────────────────────────────────────────────────

/// Siege duration per fortification level (turns).
const SIEGE_TURNS_PER_FORT_LEVEL: u32 = 3;

/// Prosperity lost per turn of siege.
const SIEGE_PROSPERITY_DRAIN: i64 = 20_000; // 20.000

/// Unrest gained per turn of siege.
const SIEGE_UNREST_GAIN: i64 = 15_000; // 15.000

/// Population attrition per turn of siege (per-mille: 5 = 0.5%).
const SIEGE_POPULATION_ATTRITION_PERMILLE: u32 = 5;

/// Attacker casualty rate on siege completion (per-mille of garrison strength).
const SIEGE_ASSAULT_CASUALTY_RATE: u32 = 200; // 20% of garrison lost as attacker casualties

/// Process all active sieges and start new ones.
///
/// Called after battles and province captures.  For each enemy army
/// stationary in a fortified province:
/// 1. If already besieging: tick progress, apply attrition, check completion.
/// 2. If new arrival: start siege.
///
/// Returns siege-related events (SiegeStarted, SiegeCompleted).
pub fn process_sieges(world: &mut GameWorld) -> Vec<crown_ash_types::GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;

    // --- Phase 1: Tick existing sieges. ---

    // Cancel sieges where the army moved away, died, or defenders arrived.
    let army_count = world.armies.len();
    for idx in 0..army_count {
        if let Some(ref siege) = world.armies[idx].siege {
            let pid = siege.target_province;
            let still_there = world.armies[idx].location == pid
                && !world.armies[idx].is_moving()
                && world.armies[idx].troops.total() > 0;

            if !still_there {
                // Army moved or destroyed — cancel siege.
                world.armies[idx].siege = None;
                world.dirty.dirty_armies.insert(world.armies[idx].id);
                continue;
            }

            // Check if a defending army arrived (would have fought in battle step).
            let defender_faction = siege.defender_faction;
            let defenders_present = world.armies.iter().any(|a| {
                a.location == pid
                    && !a.is_moving()
                    && a.owner_faction == defender_faction
                    && a.troops.total() > 0
            });
            if defenders_present {
                // Defenders broke through — cancel siege.
                world.armies[idx].siege = None;
                world.dirty.dirty_armies.insert(world.armies[idx].id);
            }
        }
    }

    // Tick progress on remaining sieges.
    let army_count = world.armies.len();
    let mut completed_sieges: Vec<(usize, u16, u8, u8, u32)> = Vec::new();

    for idx in 0..army_count {
        let army_id = world.armies[idx].id;
        let owner = world.armies[idx].owner_faction;
        if let Some(ref mut siege) = world.armies[idx].siege {
            siege.turns_besieged += 1;
            let besieged = siege.turns_besieged;
            let required = siege.turns_required;
            let target = siege.target_province;
            let defender = siege.defender_faction;
            world.dirty.dirty_armies.insert(army_id);

            if besieged >= required {
                completed_sieges.push((idx, target, defender, owner, besieged));
            }
        }
    }

    // Apply siege attrition to besieged provinces.
    let mut besieged_provinces: Vec<u16> = Vec::new();
    for army in &world.armies {
        if let Some(ref siege) = army.siege {
            if !besieged_provinces.contains(&siege.target_province) {
                besieged_provinces.push(siege.target_province);
            }
        }
    }
    for &pid in &besieged_provinces {
        if let Some(prov) = world.province_mut_dirty(pid) {
            prov.prosperity -= FixedPoint::from_raw(SIEGE_PROSPERITY_DRAIN);
            prov.unrest += FixedPoint::from_raw(SIEGE_UNREST_GAIN);
            let pop_loss = (prov.population as u64 * SIEGE_POPULATION_ATTRITION_PERMILLE as u64
                / 1000) as u32;
            prov.population = prov.population.saturating_sub(pop_loss.max(1));
            prov.last_siege_turn = Some(turn);
        }
    }

    // Apply completed sieges.
    for (army_idx, pid, old_ctrl, new_ctrl, turns_lasted) in completed_sieges {
        // Attacker takes casualties proportional to garrison strength.
        let garrison_total = world.province(pid)
            .map(|p| p.garrison.total())
            .unwrap_or(0);
        let assault_casualties = garrison_total * SIEGE_ASSAULT_CASUALTY_RATE / 1000;

        // Apply casualties to the besieging army.
        let army_id = world.armies[army_idx].id;
        apply_casualties(world, army_id, assault_casualties);

        // Clear garrison.
        if let Some(prov) = world.province_mut_dirty(pid) {
            prov.garrison = crown_ash_types::province::Troops {
                levy: 0,
                men_at_arms: 0,
                knights: 0,
            };
            // Add siege scar.
            prov.add_scar(crown_ash_types::province::ProvinceScar {
                turn_inflicted: turn,
                scar_type: crown_ash_types::province::ScarType::WarDamage,
                severity: FixedPoint::from_int(300),
            });
        }

        // Transfer province control.
        apply_province_capture(world, pid, old_ctrl, new_ctrl);

        // Clear siege state from army.
        if army_idx < world.armies.len() {
            world.armies[army_idx].siege = None;
            world.dirty.dirty_armies.insert(world.armies[army_idx].id);
        }

        events.push(crown_ash_types::GameEvent::SiegeCompleted {
            province: pid,
            old_controller: old_ctrl,
            new_controller: new_ctrl,
            turns_lasted,
            attacker_casualties: assault_casualties,
            turn,
        });
    }

    // --- Phase 2: Start new sieges. ---

    let province_ids: Vec<u16> = world.provinces.iter().map(|p| p.id).collect();
    for &pid in &province_ids {
        let (controller, fortification) = match world.province(pid) {
            Some(p) => (p.controller, p.fortification),
            None => continue,
        };
        if fortification == 0 {
            continue; // No walls — handled by check_province_captures.
        }

        // Find an enemy army that could start a siege.
        let candidate_idx = world.armies.iter().position(|a| {
            a.location == pid
                && !a.is_moving()
                && a.owner_faction != controller
                && a.troops.total() > 0
                && a.siege.is_none()
                && world.at_war(a.owner_faction, controller)
        });

        if let Some(idx) = candidate_idx {
            // Check no defending army present.
            let defenders_present = world.armies.iter().any(|a| {
                a.location == pid
                    && !a.is_moving()
                    && a.owner_faction == controller
                    && a.troops.total() > 0
            });
            if defenders_present {
                continue;
            }

            let turns_required = (fortification as u32 + 1) * SIEGE_TURNS_PER_FORT_LEVEL;
            let attacker_faction = world.armies[idx].owner_faction;

            world.armies[idx].siege = Some(crown_ash_types::SiegeProgress {
                target_province: pid,
                defender_faction: controller,
                turns_besieged: 0,
                turns_required,
            });
            world.dirty.dirty_armies.insert(world.armies[idx].id);

            events.push(crown_ash_types::GameEvent::SiegeStarted {
                province: pid,
                attacker_faction,
                defender_faction: controller,
                turns_required,
                turn,
            });
        }
    }

    events
}

/// Transfer province control and update realm province lists.
fn apply_province_capture(world: &mut GameWorld, pid: u16, old_ctrl: u8, new_ctrl: u8) {
    if let Some(province) = world.province_mut_dirty(pid) {
        province.controller = new_ctrl;
    }
    if let Some(old_realm) = world.realm_for_faction_mut_dirty(old_ctrl) {
        old_realm.provinces.retain(|&p| p != pid);
    }
    if let Some(new_realm) = world.realm_for_faction_mut_dirty(new_ctrl) {
        if !new_realm.provinces.contains(&pid) {
            new_realm.provinces.push(pid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::{WorldConfig, FixedPoint};
    use crown_ash_types::province::Troops;

    fn test_world() -> GameWorld {
        let config = WorldConfig::default();
        init_world(&config, [0xAA; 32])
    }

    #[test]
    fn no_battles_when_no_wars() {
        let mut world = test_world();
        let mut rng = DeterministicRng::new([0x01; 32], "combat_test");
        let results = resolve_battles(&mut world, &mut rng);
        assert!(results.is_empty(), "No battles should occur without wars");
    }

    #[test]
    fn battle_resolves_when_at_war() {
        let mut world = test_world();

        // Declare war between faction 0 and faction 4 (share border at province 1/7).
        if let Some(rel) = world.relation_mut(0, 4) {
            rel.at_war = true;
        }

        // Place armies in the same province (province 7, owned by faction 0).
        let aid1 = world.alloc_army_id();
        world.armies.push(Army {
            id: aid1,
            owner_faction: 0,
            commander: None,
            troops: Troops { levy: 500, men_at_arms: 100, knights: 10 },
            morale: FixedPoint::from_int(800),
            location: 7,
            destination: None,
            movement_queue: Vec::new(),
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
            siege: None,
        });

        let aid2 = world.alloc_army_id();
        world.armies.push(Army {
            id: aid2,
            owner_faction: 4,
            commander: None,
            troops: Troops { levy: 400, men_at_arms: 80, knights: 8 },
            morale: FixedPoint::from_int(800),
            location: 7,
            destination: None,
            movement_queue: Vec::new(),
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
            siege: None,
        });

        let mut rng = DeterministicRng::new([0x42; 32], "combat_test");
        let results = resolve_battles(&mut world, &mut rng);

        assert_eq!(results.len(), 1);
        assert!(results[0].attacker_casualties > 0 || results[0].defender_casualties > 0);
    }

    #[test]
    fn siege_starts_on_fortified_province() {
        let mut world = test_world();

        // Declare war between faction 0 and faction 4.
        if let Some(rel) = world.relation_mut(0, 4) {
            rel.at_war = true;
        }

        // Fortify a province controlled by faction 0.
        let target_pid = world.provinces.iter()
            .find(|p| p.controller == 0)
            .map(|p| p.id)
            .unwrap();
        if let Some(prov) = world.province_mut_dirty(target_pid) {
            prov.fortification = 2; // Level 2 walls.
            prov.garrison = Troops { levy: 0, men_at_arms: 0, knights: 0 }; // empty garrison
        }

        // Place enemy army in that province.
        let aid = world.alloc_army_id();
        world.armies.push(Army {
            id: aid,
            owner_faction: 4,
            commander: None,
            troops: Troops { levy: 500, men_at_arms: 100, knights: 10 },
            morale: FixedPoint::from_int(800),
            location: target_pid,
            destination: None,
            movement_queue: Vec::new(),
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
            siege: None,
        });

        // Remove any defender armies from that province.
        world.armies.retain(|a| !(a.owner_faction == 0 && a.location == target_pid && a.id != aid));

        let events = process_sieges(&mut world);

        // Should have started a siege.
        let siege_started = events.iter().any(|e| matches!(e, crown_ash_types::GameEvent::SiegeStarted { .. }));
        assert!(siege_started, "Should emit SiegeStarted event");

        let army = world.army(aid).unwrap();
        assert!(army.siege.is_some(), "Army should have siege progress");
        let siege = army.siege.as_ref().unwrap();
        assert_eq!(siege.target_province, target_pid);
        // (fort_level + 1) * 3 = (2 + 1) * 3 = 9
        assert_eq!(siege.turns_required, 9);
        assert_eq!(siege.turns_besieged, 0);
    }

    #[test]
    fn siege_ticks_to_completion() {
        let mut world = test_world();
        world.meta.turn = 1;

        if let Some(rel) = world.relation_mut(0, 4) {
            rel.at_war = true;
        }

        let target_pid = world.provinces.iter()
            .find(|p| p.controller == 0)
            .map(|p| p.id)
            .unwrap();
        if let Some(prov) = world.province_mut_dirty(target_pid) {
            prov.fortification = 1; // Level 1 → (1+1)*3 = 6 turns
            prov.garrison = Troops { levy: 200, men_at_arms: 0, knights: 0 };
        }

        let aid = world.alloc_army_id();
        world.armies.push(Army {
            id: aid,
            owner_faction: 4,
            commander: None,
            troops: Troops { levy: 1000, men_at_arms: 200, knights: 20 },
            morale: FixedPoint::from_int(800),
            location: target_pid,
            destination: None,
            movement_queue: Vec::new(),
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
            siege: Some(crown_ash_types::SiegeProgress {
                target_province: target_pid,
                defender_faction: 0,
                turns_besieged: 5, // One tick away from completion (need 6).
                turns_required: 6,
            }),
        });

        world.armies.retain(|a| !(a.owner_faction == 0 && a.location == target_pid && a.id != aid));

        let events = process_sieges(&mut world);

        let completed = events.iter().any(|e| matches!(e, crown_ash_types::GameEvent::SiegeCompleted { .. }));
        assert!(completed, "Siege should complete on 6th tick");

        // Province should now belong to faction 4.
        let prov = world.province(target_pid).unwrap();
        assert_eq!(prov.controller, 4, "Province should be captured");
        assert_eq!(prov.garrison.total(), 0, "Garrison should be destroyed");

        // Army siege should be cleared.
        let army = world.army(aid).unwrap();
        assert!(army.siege.is_none(), "Siege should be cleared after completion");
    }

    #[test]
    fn unfortified_province_instant_capture() {
        let mut world = test_world();

        if let Some(rel) = world.relation_mut(0, 4) {
            rel.at_war = true;
        }

        let target_pid = world.provinces.iter()
            .find(|p| p.controller == 0)
            .map(|p| p.id)
            .unwrap();
        if let Some(prov) = world.province_mut_dirty(target_pid) {
            prov.fortification = 0; // No walls.
            prov.garrison = Troops { levy: 0, men_at_arms: 0, knights: 0 };
        }

        let aid = world.alloc_army_id();
        world.armies.push(Army {
            id: aid,
            owner_faction: 4,
            commander: None,
            troops: Troops { levy: 500, men_at_arms: 100, knights: 10 },
            morale: FixedPoint::from_int(800),
            location: target_pid,
            destination: None,
            movement_queue: Vec::new(),
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
            siege: None,
        });

        world.armies.retain(|a| !(a.owner_faction == 0 && a.location == target_pid && a.id != aid));

        let captures = check_province_captures(&mut world);
        assert!(!captures.is_empty(), "Unfortified province should be captured instantly");
        assert_eq!(captures[0].2, 4, "New controller should be faction 4");

        // No siege should have started.
        let army = world.army(aid).unwrap();
        assert!(army.siege.is_none(), "No siege needed for unfortified");
    }

    #[test]
    fn besieging_army_stays_put() {
        use crate::world_gen::init_world;
        use crate::tick::tick;

        let config = crown_ash_types::WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);
        world.armies.clear();

        // Army is besieging with NO destination — should not move.
        let aid = world.alloc_army_id();
        world.armies.push(Army {
            id: aid,
            owner_faction: 0,
            commander: None,
            troops: Troops { levy: 500, men_at_arms: 100, knights: 10 },
            morale: FixedPoint::from_int(800),
            location: 7,
            destination: None,
            movement_queue: Vec::new(),
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
            siege: Some(crown_ash_types::SiegeProgress {
                target_province: 7,
                defender_faction: 4,
                turns_besieged: 2,
                turns_required: 9,
            }),
        });

        let _ = tick(&mut world, [0x01; 32]);

        let army = world.army(aid).unwrap();
        assert_eq!(army.location, 7, "Besieging army should not move");
        assert!(army.siege.is_some(), "Siege should still be active");
    }

    #[test]
    fn siege_cancelled_when_army_moves() {
        let mut world = test_world();

        if let Some(rel) = world.relation_mut(0, 4) {
            rel.at_war = true;
        }

        let target_pid = world.provinces.iter()
            .find(|p| p.controller == 4)
            .map(|p| p.id)
            .unwrap();

        // Army besieging but has a destination — siege should cancel.
        let aid = world.alloc_army_id();
        world.armies.push(Army {
            id: aid,
            owner_faction: 0,
            commander: None,
            troops: Troops { levy: 500, men_at_arms: 100, knights: 10 },
            morale: FixedPoint::from_int(800),
            location: target_pid,
            destination: Some(7),
            movement_queue: Vec::new(),
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
            siege: Some(crown_ash_types::SiegeProgress {
                target_province: target_pid,
                defender_faction: 4,
                turns_besieged: 2,
                turns_required: 9,
            }),
        });

        let events = process_sieges(&mut world);

        let army = world.army(aid).unwrap();
        assert!(army.siege.is_none(), "Siege should be cancelled when army has destination");
        // No SiegeCompleted event.
        assert!(!events.iter().any(|e| matches!(e, crown_ash_types::GameEvent::SiegeCompleted { .. })));
    }

    #[test]
    fn casualty_distribution_preserves_types() {
        let mut world = test_world();
        let aid = world.alloc_army_id();
        world.armies.push(Army {
            id: aid,
            owner_faction: 0,
            commander: None,
            troops: Troops { levy: 1000, men_at_arms: 200, knights: 50 },
            morale: FixedPoint::from_int(800),
            location: 7,
            destination: None,
            movement_queue: Vec::new(),
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
            siege: None,
        });

        apply_casualties(&mut world, aid, 500);

        let army = world.army(aid).unwrap();
        let remaining = army.troops.total();
        // Should have lost roughly 500 soldiers (proportional distribution).
        assert!(remaining < 1250, "Should have lost soldiers");
        assert!(remaining > 600, "Should not have lost too many");
    }
}
