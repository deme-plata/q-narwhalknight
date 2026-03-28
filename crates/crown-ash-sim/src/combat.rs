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

/// Province siege: if an army occupies an enemy province unopposed for a turn, capture it.
pub fn check_province_captures(world: &mut GameWorld) -> Vec<(u16, u8, u8)> {
    let mut captures = Vec::new();

    let province_ids: Vec<u16> = world.provinces.iter().map(|p| p.id).collect();

    for &pid in &province_ids {
        let controller = match world.province(pid) {
            Some(p) => p.controller,
            None => continue,
        };

        // Find any enemy army stationary in this province.
        let enemy_army = world.armies.iter()
            .find(|a| {
                a.location == pid
                    && !a.is_moving()
                    && a.owner_faction != controller
                    && a.troops.total() > 0
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

            if !defenders_present && garrison_strength == 0 {
                let new_controller = army.owner_faction;
                captures.push((pid, controller, new_controller));
            }
        }
    }

    // Apply captures.
    for &(pid, old_ctrl, new_ctrl) in &captures {
        if let Some(province) = world.province_mut_dirty(pid) {
            province.controller = new_ctrl;
        }
        // Update realm province lists.
        if let Some(old_realm) = world.realm_for_faction_mut_dirty(old_ctrl) {
            old_realm.provinces.retain(|&p| p != pid);
        }
        if let Some(new_realm) = world.realm_for_faction_mut_dirty(new_ctrl) {
            if !new_realm.provinces.contains(&pid) {
                new_realm.provinces.push(pid);
            }
        }
    }

    captures
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
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
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
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
        });

        let mut rng = DeterministicRng::new([0x42; 32], "combat_test");
        let results = resolve_battles(&mut world, &mut rng);

        assert_eq!(results.len(), 1);
        assert!(results[0].attacker_casualties > 0 || results[0].defender_casualties > 0);
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
            raised_turn: 0,
            supply: FixedPoint::from_int(100),
        });

        apply_casualties(&mut world, aid, 500);

        let army = world.army(aid).unwrap();
        let remaining = army.troops.total();
        // Should have lost roughly 500 soldiers (proportional distribution).
        assert!(remaining < 1250, "Should have lost soldiers");
        assert!(remaining > 600, "Should not have lost too many");
    }
}
