//! AI — simple heuristic decision-making for NPC factions.
//!
//! ## Strategy
//!
//! - **Strong** (army > neighbors): expand — declare war on weakest neighbor.
//! - **Weak**: build economy, raise armies.
//! - **At war and losing**: propose peace.
//! - **Always**: raise armies if below military threshold.
//!
//! AI decisions are deterministic given the same world state and RNG.
//! All math uses `FixedPoint` — no floating point.

use crown_ash_types::{
    CasusBelli, CharacterRole, FixedPoint, GameAction, PlotType, TreatyType,
};
use crate::map;
use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Minimum army strength ratio before AI considers expanding.
const EXPANSION_STRENGTH_RATIO: i64 = 1500; // 1.500x enemy = "strong"

/// Army-to-province ratio threshold below which AI raises more troops.
const MIN_ARMY_PER_PROVINCE: u32 = 200;

/// Minimum treasury to start building improvements.
const BUILD_TREASURY_THRESHOLD: FixedPoint = FixedPoint::from_int(200);

/// Losing ratio threshold — AI proposes peace if power < 60% of enemy.
const LOSING_RATIO: i64 = 600; // 0.600

/// Generate AI actions for all NPC factions.
///
/// A faction is NPC if its realm has no `player_wallet` set.
/// Returns a list of `GameAction`s to be queued.
pub fn generate_ai_actions(world: &GameWorld, rng: &mut DeterministicRng) -> Vec<(u8, GameAction)> {
    let mut actions: Vec<(u8, GameAction)> = Vec::new();

    for realm in &world.realms {
        // Skip player-controlled factions.
        if realm.owner_wallet.is_empty() {
            // NPC faction
        } else {
            continue;
        }

        let faction_id = realm.faction;

        // Skip dead factions.
        if !world.faction(faction_id).map_or(false, |f| f.alive) {
            continue;
        }
        if realm.provinces.is_empty() {
            continue;
        }

        let our_power = world.faction_total_power(faction_id);
        let our_army_count: u32 = world.armies.iter()
            .filter(|a| a.owner_faction == faction_id)
            .map(|a| a.total_soldiers())
            .sum();

        // --- Raise armies if below threshold ---
        let min_troops = realm.provinces.len() as u32 * MIN_ARMY_PER_PROVINCE;
        if our_army_count < min_troops {
            // Raise from the most populated province.
            if let Some(best_province) = realm.provinces.iter()
                .filter_map(|&pid| world.province(pid))
                .max_by_key(|p| p.population)
            {
                actions.push((faction_id, GameAction::RaiseArmy {
                    province: best_province.id,
                }));
            }
        }

        // --- Check war status ---
        if !realm.at_war_with.is_empty() {
            // At war — evaluate if winning or losing.
            for &enemy_id in &realm.at_war_with {
                let enemy_power = world.faction_total_power(enemy_id);

                if enemy_power > 0 && our_power * 1000 / enemy_power < LOSING_RATIO {
                    // We are losing — propose peace.
                    actions.push((faction_id, GameAction::ProposeTreaty {
                        target: enemy_id,
                        treaty: TreatyType::WhitePeace,
                    }));
                } else {
                    // We are winning or even — continue fighting.
                    // Move armies toward enemy provinces.
                    let enemy_provinces: Vec<u16> = world.provinces.iter()
                        .filter(|p| p.controller == enemy_id)
                        .map(|p| p.id)
                        .collect();

                    // Find our idle armies.
                    let idle_armies: Vec<u32> = world.armies.iter()
                        .filter(|a| a.owner_faction == faction_id && !a.is_moving())
                        .map(|a| a.id)
                        .collect();

                    for army_id in idle_armies.iter().take(3) {
                        let army_loc = world.army(*army_id).map(|a| a.location).unwrap_or(0);

                        // Find adjacent enemy province or a step toward one.
                        let target = find_attack_target(world, army_loc, &enemy_provinces);
                        if let Some(target_id) = target {
                            actions.push((faction_id, GameAction::MoveArmy {
                                army: *army_id,
                                target: target_id,
                            }));
                        }
                    }
                }
            }
        } else {
            // Not at war — consider expansion or economy.

            // Find neighboring factions.
            let neighbor_factions = find_neighbor_factions(world, faction_id);

            // Evaluate if we are strong enough to expand.
            let weakest_neighbor = neighbor_factions.iter()
                .filter(|&&nf| nf != faction_id)
                .min_by_key(|&&nf| world.faction_total_power(nf));

            if let Some(&weakest) = weakest_neighbor {
                let their_power = world.faction_total_power(weakest);
                let ratio = if their_power > 0 {
                    our_power * 1000 / their_power
                } else {
                    2000 // infinite advantage
                };

                if ratio > EXPANSION_STRENGTH_RATIO && rng.chance(1, 5) {
                    // Declare war on weakest neighbor.
                    actions.push((faction_id, GameAction::DeclareWar {
                        target: weakest,
                        casus_belli: CasusBelli::Conquest,
                    }));
                }
            }

            // Build improvements if treasury allows.
            if realm.treasury > BUILD_TREASURY_THRESHOLD {
                // Find a province without a Market.
                if let Some(build_province) = realm.provinces.iter()
                    .filter_map(|&pid| world.province(pid))
                    .find(|p| {
                        !p.improvements.contains(&crown_ash_types::Improvement::Market)
                            && p.construction_queue.is_empty()
                    })
                {
                    actions.push((faction_id, GameAction::BuildImprovement {
                        province: build_province.id,
                        improvement: crown_ash_types::Improvement::Market,
                    }));
                }
            }
        }

        // --- Intrigue AI: plots and espionage ---
        generate_intrigue_actions(world, faction_id, rng, &mut actions);
    }

    actions
}

/// Generate intrigue-related AI actions for an NPC faction.
///
/// Rules:
/// - At war + enemy ruler has low intrigue (< 8): 10% chance to launch assassination.
/// - Rival has rich province (prosperity > 600): 5% chance to launch sabotage.
/// - Back existing friendly plots against enemies: 15% chance per eligible character.
fn generate_intrigue_actions(
    world: &GameWorld,
    faction_id: u8,
    rng: &mut DeterministicRng,
    actions: &mut Vec<(u8, GameAction)>,
) {
    let realm = match world.realm_for_faction(faction_id) {
        Some(r) => r,
        None => return,
    };

    // --- Assassination: at war, enemy ruler is weak ---
    for &enemy_id in &realm.at_war_with {
        let enemy_ruler = world
            .realms
            .iter()
            .find(|r| r.faction == enemy_id)
            .map(|r| r.ruler);
        if let Some(ruler_id) = enemy_ruler {
            let ruler_intrigue = world
                .character(ruler_id)
                .map(|c| c.effective_stats().intrigue.integer())
                .unwrap_or(0);
            // Low intrigue = vulnerable target.
            if ruler_intrigue < 8 && rng.chance(1, 10) {
                // Only if we don't already have an assassination plot against this ruler.
                let already_plotting = world.plots.iter().any(|p| {
                    p.target == ruler_id
                        && p.plot_type == PlotType::Assassination
                        && world
                            .character(p.instigator)
                            .map(|c| c.faction == faction_id)
                            .unwrap_or(false)
                });
                if !already_plotting {
                    actions.push((faction_id, GameAction::LaunchPlot {
                        target: ruler_id,
                        plot_type: PlotType::Assassination,
                    }));
                }
            }
        }
    }

    // --- Sabotage: rival has rich province ---
    let neighbor_factions = find_neighbor_factions(world, faction_id);
    for &nf in &neighbor_factions {
        let rich_province = world.provinces.iter().find(|p| {
            p.controller == nf && p.prosperity > FixedPoint::from_int(600)
        });
        if rich_province.is_some() && rng.chance(1, 20) {
            // Find the enemy ruler as the target.
            let enemy_ruler = world
                .realms
                .iter()
                .find(|r| r.faction == nf)
                .map(|r| r.ruler);
            if let Some(ruler_id) = enemy_ruler {
                let already_plotting = world.plots.iter().any(|p| {
                    p.target == ruler_id
                        && p.plot_type == PlotType::Sabotage
                        && world
                            .character(p.instigator)
                            .map(|c| c.faction == faction_id)
                            .unwrap_or(false)
                });
                if !already_plotting {
                    actions.push((faction_id, GameAction::LaunchPlot {
                        target: ruler_id,
                        plot_type: PlotType::Sabotage,
                    }));
                }
            }
        }
    }

    // --- Back existing plots against enemies ---
    let friendly_plots: Vec<u32> = world
        .plots
        .iter()
        .filter(|p| {
            // Plot instigator is from our faction.
            world
                .character(p.instigator)
                .map(|c| c.faction == faction_id)
                .unwrap_or(false)
        })
        .map(|p| p.id)
        .collect();

    for plot_id in &friendly_plots {
        // 15% chance to assign a backer.
        if rng.chance(15, 100) {
            // Find a character in our faction who isn't already backing this plot
            // and isn't the ruler or heir.
            let existing_backers: Vec<u32> = world
                .plots
                .iter()
                .find(|p| p.id == *plot_id)
                .map(|p| p.backers.clone())
                .unwrap_or_default();
            let instigator_id = world
                .plots
                .iter()
                .find(|p| p.id == *plot_id)
                .map(|p| p.instigator);

            let backer_candidate = world
                .characters
                .iter()
                .find(|c| {
                    c.faction == faction_id
                        && c.alive
                        && c.role != CharacterRole::Ruler
                        && c.role != CharacterRole::Heir
                        && Some(c.id) != instigator_id
                        && !existing_backers.contains(&c.id)
                })
                .map(|c| c.id);

            if let Some(_backer_id) = backer_candidate {
                actions.push((faction_id, GameAction::BackPlot { plot_id: *plot_id }));
            }
        }
    }
}

/// Find factions that share a border with the given faction.
fn find_neighbor_factions(world: &GameWorld, faction_id: u8) -> Vec<u8> {
    let mut neighbors = Vec::new();

    for province in &world.provinces {
        if province.controller != faction_id {
            continue;
        }
        for &neighbor_id in &province.neighbors {
            if let Some(neighbor_prov) = world.province(neighbor_id) {
                let nf = neighbor_prov.controller;
                if nf != faction_id && !neighbors.contains(&nf) {
                    neighbors.push(nf);
                }
            }
        }
    }

    neighbors
}

/// Find the best adjacent province to move toward when attacking.
///
/// If an enemy province is directly adjacent, return it.
/// Otherwise, return the neighbor that gets us closer.
fn find_attack_target(_world: &GameWorld, from: u16, enemy_provinces: &[u16]) -> Option<u16> {
    let neighbors = map::neighbors(from);

    // Direct attack: adjacent enemy province.
    for &n in neighbors {
        if enemy_provinces.contains(&n) {
            return Some(n);
        }
    }

    // Step toward: pick neighbor that has the most adjacencies to enemy provinces.
    let mut best: Option<(u16, usize)> = None;
    for &n in neighbors {
        let n_neighbors = map::neighbors(n);
        let enemy_adj_count = n_neighbors.iter()
            .filter(|&&nn| enemy_provinces.contains(&nn))
            .count();
        if enemy_adj_count > 0 {
            if best.map_or(true, |(_, bc)| enemy_adj_count > bc) {
                best = Some((n, enemy_adj_count));
            }
        }
    }

    best.map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::WorldConfig;

    #[test]
    fn npc_factions_generate_actions() {
        let config = WorldConfig::default();
        let world = init_world(&config, [0x42; 32]);

        let mut rng = DeterministicRng::new([0x42; 32], "ai");
        let actions = generate_ai_actions(&world, &mut rng);

        // All factions are NPC by default, so there should be some actions.
        assert!(!actions.is_empty(), "NPC factions should generate AI actions");
    }

    #[test]
    fn player_factions_skipped() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);

        // Set all factions as player-controlled.
        for realm in &mut world.realms {
            realm.owner_wallet = format!("player_{}", realm.faction);
        }

        let mut rng = DeterministicRng::new([0x42; 32], "ai");
        let actions = generate_ai_actions(&world, &mut rng);

        assert!(actions.is_empty(), "Player factions should not have AI actions");
    }

    #[test]
    fn neighbor_factions_found() {
        let config = WorldConfig::default();
        let world = init_world(&config, [0x42; 32]);

        // Ashen Crown (faction 0) should border Frost Marches (4), Vale Princes (1),
        // Ember Church (2), Black Abbey (6), and Red Steppe (5) via Crownspire.
        let neighbors = find_neighbor_factions(&world, 0);
        assert!(neighbors.contains(&4), "Ashen Crown should border Frost Marches");
        assert!(neighbors.contains(&1), "Ashen Crown should border Vale Princes");
    }
}
