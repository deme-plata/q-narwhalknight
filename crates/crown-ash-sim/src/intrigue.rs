//! Intrigue system -- plots, espionage, and counter-intelligence.
//!
//! Each turn, active plots advance based on the instigator's intrigue stat.
//! Plots can be detected by the target's faction spymaster.
//! Success triggers the plot effect; discovery causes opinion penalties.
//!
//! # Mechanics
//!
//! - **Progress**: `instigator.intrigue * 20 + sum(backer.intrigue * 5)` per turn.
//!   Fires when progress >= 1000.
//! - **Secrecy**: Starts at `500 + instigator.intrigue * 30`. Decays by 10/turn.
//!   Boosted by +100 for `Deceitful` trait.
//! - **Discovery**: Spymaster's `intrigue * 10` vs plot secrecy. If discovery
//!   power > secrecy, the plot is found.
//! - **Assassination success**: 70% + (intrigue * 3)% when firing. On failure,
//!   the plot is automatically discovered.
//! - **Opinion penalty**: -100 on discovery. +50 for target faction against
//!   instigator's faction.
//! - **Trait interaction**: `Paranoid` gives +200 discovery bonus. `Deceitful`
//!   gives +100 secrecy.
//!
//! All math uses `FixedPoint` -- no floating point.

use crown_ash_types::{
    CharacterId, CharacterRole, DeathCause, FixedPoint, GameEvent, Plot, PlotType, Trait,
};
use crown_ash_types::diplomacy::Grievance;
use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Maximum plots that can fire (execute) in a single turn.
pub const MAX_PLOTS_PER_TURN: usize = 3;

/// Global cap on active plots to prevent unbounded growth.
pub const MAX_ACTIVE_PLOTS: usize = 20;

/// Secrecy decay per turn (FixedPoint integer units).
const SECRECY_DECAY_PER_TURN: i64 = 10;

/// Progress threshold at which a plot fires.
const PLOT_FIRE_THRESHOLD: FixedPoint = FixedPoint::from_int(1000);

/// Base assassination success chance (per mille): 700 = 70%.
const ASSASSINATION_BASE_CHANCE: u32 = 700;

/// Assassination chance bonus per intrigue point (per mille): 30 = 3%.
const ASSASSINATION_INTRIGUE_BONUS: u32 = 30;

/// Opinion penalty when a plot is discovered.
const DISCOVERY_OPINION_PENALTY: FixedPoint = FixedPoint::from_int(100);

/// Opinion modifier applied to the target faction against the instigator's faction.
const DISCOVERY_FACTION_OPINION: FixedPoint = FixedPoint::from_int(50);

/// Gold stolen per successful Steal plot (FixedPoint raw).
const STEAL_GOLD_AMOUNT: FixedPoint = FixedPoint::from_int(100);

/// Prosperity reduction for a successful Sabotage plot.
const SABOTAGE_PROSPERITY_LOSS: FixedPoint = FixedPoint::from_int(200);

/// Paranoid trait discovery bonus (FixedPoint integer).
const PARANOID_DISCOVERY_BONUS: i64 = 200;

/// Deceitful trait secrecy bonus (FixedPoint integer) -- applied at plot creation.
const DECEITFUL_SECRECY_BONUS: i64 = 100;

// ---------------------------------------------------------------------------
// Per-turn processing
// ---------------------------------------------------------------------------

/// Advance all active plots, check for discovery and execution.
///
/// Called once per tick. Returns a list of narrative events.
pub fn process_intrigue(world: &mut GameWorld, rng: &mut DeterministicRng) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;

    // Collect IDs of plots to process (avoid borrow issues).
    let plot_ids: Vec<u32> = world.plots.iter().map(|p| p.id).collect();

    // Track which plots to remove after processing.
    let mut plots_to_remove: Vec<u32> = Vec::new();
    let mut fired_count: usize = 0;

    for plot_id in &plot_ids {
        // --- 1. Advance progress ---
        let advance_data = compute_advance_data(world, *plot_id);
        let (instigator_intrigue, backer_intrigue_sum, _instigator_secrecy_bonus) =
            match advance_data {
                Some(d) => d,
                None => {
                    plots_to_remove.push(*plot_id);
                    continue;
                }
            };

        // Apply progress: instigator.intrigue * 20 + sum(backer.intrigue * 5)
        let progress_gain = FixedPoint::from_int(instigator_intrigue * 20 + backer_intrigue_sum * 5);

        // Apply secrecy decay and instigator's intrigue-based secrecy maintenance.
        let secrecy_maintenance = FixedPoint::from_int(instigator_intrigue * 5);
        let secrecy_delta =
            secrecy_maintenance - FixedPoint::from_int(SECRECY_DECAY_PER_TURN);

        if let Some(plot) = world.plots.iter_mut().find(|p| p.id == *plot_id) {
            plot.progress += progress_gain;
            plot.secrecy += secrecy_delta;
            // Clamp secrecy to [0, 1000].
            plot.secrecy = plot
                .secrecy
                .clamp(FixedPoint::ZERO, FixedPoint::from_int(1000));
        }

        // --- 2. Check for discovery (counter-intelligence) ---
        let discovery_result = check_discovery(world, *plot_id, rng);
        if let Some(discovery_event) = discovery_result {
            events.push(discovery_event);
            // Apply opinion penalty.
            apply_discovery_penalty(world, *plot_id, turn);
            plots_to_remove.push(*plot_id);
            continue;
        }

        // --- 3. Check for plot execution (progress >= 1000) ---
        let should_fire = world
            .plots
            .iter()
            .find(|p| p.id == *plot_id)
            .map(|p| p.progress >= PLOT_FIRE_THRESHOLD)
            .unwrap_or(false);

        if should_fire && fired_count < MAX_PLOTS_PER_TURN {
            let fire_events = execute_plot(world, *plot_id, rng, turn);
            events.extend(fire_events);
            plots_to_remove.push(*plot_id);
            fired_count += 1;
        }
    }

    // --- 4. Remove completed / discovered plots ---
    world.plots.retain(|p| !plots_to_remove.contains(&p.id));

    // --- 5. Cap total active plots ---
    while world.plots.len() > MAX_ACTIVE_PLOTS {
        // Remove the oldest plot (lowest started_turn).
        if let Some(oldest_idx) = world
            .plots
            .iter()
            .enumerate()
            .min_by_key(|(_, p)| p.started_turn)
            .map(|(i, _)| i)
        {
            world.plots.remove(oldest_idx);
        } else {
            break;
        }
    }

    events
}

// ---------------------------------------------------------------------------
// Advance data computation (reads world immutably)
// ---------------------------------------------------------------------------

/// Returns (instigator_intrigue_int, sum_backer_intrigue_int, instigator_secrecy_bonus).
/// None if the plot or instigator no longer exists / is dead.
fn compute_advance_data(world: &GameWorld, plot_id: u32) -> Option<(i64, i64, i64)> {
    let plot = world.plots.iter().find(|p| p.id == plot_id)?;
    let instigator = world.character(plot.instigator)?;
    if !instigator.alive {
        return None;
    }

    let eff = instigator.effective_stats();
    let instigator_intrigue = eff.intrigue.integer();

    // Sum backer intrigue.
    let backer_sum: i64 = plot
        .backers
        .iter()
        .filter_map(|&bid| world.character(bid))
        .filter(|c| c.alive)
        .map(|c| c.effective_stats().intrigue.integer())
        .sum();

    let secrecy_bonus = if instigator.traits.contains(&Trait::Deceitful) {
        DECEITFUL_SECRECY_BONUS
    } else {
        0
    };

    Some((instigator_intrigue, backer_sum, secrecy_bonus))
}

// ---------------------------------------------------------------------------
// Discovery check
// ---------------------------------------------------------------------------

/// Check if the target faction's spymaster discovers the plot.
/// Returns Some(GameEvent::PlotDiscovered) if found, None otherwise.
fn check_discovery(
    world: &GameWorld,
    plot_id: u32,
    _rng: &mut DeterministicRng,
) -> Option<GameEvent> {
    let plot = world.plots.iter().find(|p| p.id == plot_id)?;
    let target = world.character(plot.target)?;
    if !target.alive {
        return None;
    }

    let target_faction = target.faction;

    // Find the spymaster for the target's faction.
    let spymaster = world
        .characters
        .iter()
        .find(|c| c.faction == target_faction && c.role == CharacterRole::Spymaster && c.alive);

    let spymaster = match spymaster {
        Some(s) => s,
        None => return None, // No spymaster, no discovery.
    };

    let spy_eff = spymaster.effective_stats();
    let mut discovery_power = spy_eff.intrigue.integer() * 10;

    // Paranoid target gets bonus to discovery.
    if target.traits.contains(&Trait::Paranoid) {
        discovery_power += PARANOID_DISCOVERY_BONUS;
    }

    let secrecy = plot.secrecy.integer();

    if discovery_power > secrecy {
        let instigator_name = world
            .character(plot.instigator)
            .map(|c| c.name.clone())
            .unwrap_or_else(|| "Unknown".to_string());
        let target_name = target.name.clone();
        let discovered_by = spymaster.name.clone();

        Some(GameEvent::PlotDiscovered {
            instigator_name,
            target_name,
            discovered_by,
            turn: world.meta.turn,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Discovery penalty
// ---------------------------------------------------------------------------

/// Apply opinion penalties when a plot is discovered.
fn apply_discovery_penalty(world: &mut GameWorld, plot_id: u32, turn: u32) {
    // Read plot data.
    let (instigator_id, target_id) = {
        let plot = match world.plots.iter().find(|p| p.id == plot_id) {
            Some(p) => p,
            None => return,
        };
        (plot.instigator, plot.target)
    };

    let instigator_faction = world.character(instigator_id).map(|c| c.faction);
    let target_faction = world.character(target_id).map(|c| c.faction);

    if let (Some(ins_f), Some(tar_f)) = (instigator_faction, target_faction) {
        if ins_f != tar_f {
            if let Some(rel) = world.relation_mut_dirty(ins_f, tar_f) {
                rel.opinion -= DISCOVERY_OPINION_PENALTY;
                // Extra penalty from target faction's perspective.
                rel.opinion -= DISCOVERY_FACTION_OPINION;
                rel.grievances.push(Grievance {
                    reason: "Plot discovered against our ruler".to_string(),
                    opinion_modifier: -(DISCOVERY_OPINION_PENALTY + DISCOVERY_FACTION_OPINION),
                    inflicted_turn: turn,
                    decay_turns_remaining: 100,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Plot execution
// ---------------------------------------------------------------------------

/// Execute a plot that has reached the progress threshold.
///
/// Returns events describing the outcome.
fn execute_plot(
    world: &mut GameWorld,
    plot_id: u32,
    rng: &mut DeterministicRng,
    turn: u32,
) -> Vec<GameEvent> {
    let mut events = Vec::new();

    // Read plot data.
    let (plot_type, instigator_id, target_id) = {
        let plot = match world.plots.iter().find(|p| p.id == plot_id) {
            Some(p) => p,
            None => return events,
        };
        (plot.plot_type, plot.instigator, plot.target)
    };

    let instigator_name = world
        .character(instigator_id)
        .map(|c| c.name.clone())
        .unwrap_or_else(|| "Unknown".to_string());
    let target_name = world
        .character(target_id)
        .map(|c| c.name.clone())
        .unwrap_or_else(|| "Unknown".to_string());
    let instigator_intrigue = world
        .character(instigator_id)
        .map(|c| c.effective_stats().intrigue.integer())
        .unwrap_or(0);
    let instigator_faction = world.character(instigator_id).map(|c| c.faction);
    let target_faction = world.character(target_id).map(|c| c.faction);

    match plot_type {
        PlotType::Assassination => {
            // Success chance: 70% + (intrigue * 3)%.
            let success_chance =
                ASSASSINATION_BASE_CHANCE + (instigator_intrigue as u32 * ASSASSINATION_INTRIGUE_BONUS);
            let success_chance = success_chance.min(950); // Cap at 95%.

            if rng.chance(success_chance, 1000) {
                // Kill the target.
                if let Some(c) = world.character_mut_dirty(target_id) {
                    c.alive = false;
                    c.death_turn = Some(turn);
                    c.death_cause = Some("Assassination".to_string());
                }
                events.push(GameEvent::CharacterDied {
                    character_id: target_id,
                    character_name: target_name.clone(),
                    cause: DeathCause::Assassination,
                    turn,
                });
                events.push(GameEvent::PlotSucceeded {
                    instigator_name,
                    target_name,
                    plot_type: plot_type.label().to_string(),
                    turn,
                });
            } else {
                // Failed -- plot is automatically discovered.
                events.push(GameEvent::PlotFoiled {
                    instigator_name: instigator_name.clone(),
                    target_name: target_name.clone(),
                    turn,
                });
                // Apply discovery penalty on failure.
                apply_discovery_penalty(world, plot_id, turn);
            }
        }

        PlotType::Fabricate => {
            // Instigator gains a claim -- represented by a prestige boost.
            // In a full implementation this would create an actual claim record;
            // here we give +200 prestige (unlocking Reconquest casus belli).
            if let Some(c) = world.character_mut_dirty(instigator_id) {
                c.prestige += FixedPoint::from_int(200);
            }
            events.push(GameEvent::PlotSucceeded {
                instigator_name,
                target_name,
                plot_type: plot_type.label().to_string(),
                turn,
            });
        }

        PlotType::Sabotage => {
            // Reduce prosperity of a province controlled by the target's faction.
            if let Some(tar_f) = target_faction {
                let target_prov_id = world
                    .provinces
                    .iter()
                    .find(|p| p.controller == tar_f)
                    .map(|p| p.id);
                if let Some(pid) = target_prov_id {
                    if let Some(prov) = world.province_mut_dirty(pid) {
                        prov.prosperity -= SABOTAGE_PROSPERITY_LOSS;
                        prov.prosperity = prov
                            .prosperity
                            .clamp(FixedPoint::ZERO, FixedPoint::from_int(1000));
                    }
                }
            }
            events.push(GameEvent::PlotSucceeded {
                instigator_name,
                target_name,
                plot_type: plot_type.label().to_string(),
                turn,
            });
        }

        PlotType::Steal => {
            // Transfer gold from target faction to instigator faction.
            if let (Some(ins_f), Some(tar_f)) = (instigator_faction, target_faction) {
                let stolen = {
                    let tar_realm = world.realm_for_faction(tar_f);
                    let available = tar_realm
                        .map(|r| r.treasury)
                        .unwrap_or(FixedPoint::ZERO);
                    if available >= STEAL_GOLD_AMOUNT {
                        STEAL_GOLD_AMOUNT
                    } else {
                        available
                    }
                };
                if stolen > FixedPoint::ZERO {
                    if let Some(tar_realm) = world.realm_for_faction_mut_dirty(tar_f) {
                        tar_realm.treasury -= stolen;
                    }
                    if let Some(ins_realm) = world.realm_for_faction_mut_dirty(ins_f) {
                        ins_realm.treasury += stolen;
                    }
                }
            }
            events.push(GameEvent::PlotSucceeded {
                instigator_name,
                target_name,
                plot_type: plot_type.label().to_string(),
                turn,
            });
        }
    }

    events
}

// ---------------------------------------------------------------------------
// Player-facing plot management (called from process_action)
// ---------------------------------------------------------------------------

/// Launch a new plot.
///
/// Validates that instigator and target belong to different factions and
/// that the instigator is alive. Creates the plot and adds it to the world.
///
/// Returns the newly created plot ID on success, or an error string.
pub fn launch_plot(
    world: &mut GameWorld,
    instigator_id: CharacterId,
    target_id: CharacterId,
    plot_type: PlotType,
) -> Result<u32, String> {
    let instigator = world
        .character(instigator_id)
        .ok_or("Instigator not found")?;
    if !instigator.alive {
        return Err("Instigator is dead".to_string());
    }
    let instigator_faction = instigator.faction;
    let instigator_intrigue = instigator.effective_stats().intrigue.integer();
    let has_deceitful = instigator.traits.contains(&Trait::Deceitful);

    let target = world.character(target_id).ok_or("Target not found")?;
    if !target.alive {
        return Err("Target is dead".to_string());
    }
    if target.faction == instigator_faction {
        return Err("Cannot plot against your own faction".to_string());
    }

    // Compute starting secrecy: 500 + intrigue * 30 + deceitful bonus.
    let mut starting_secrecy = 500 + instigator_intrigue * 30;
    if has_deceitful {
        starting_secrecy += DECEITFUL_SECRECY_BONUS;
    }
    let starting_secrecy =
        FixedPoint::from_int(starting_secrecy.clamp(0, 1000));

    let plot_id = world.alloc_plot_id();
    let turn = world.meta.turn;

    world.plots.push(Plot {
        id: plot_id,
        plot_type,
        instigator: instigator_id,
        target: target_id,
        backers: Vec::new(),
        progress: FixedPoint::ZERO,
        secrecy: starting_secrecy,
        started_turn: turn,
    });

    Ok(plot_id)
}

/// Add a backer to an existing plot.
///
/// The backer must be alive and from the same faction as the instigator.
pub fn back_plot(
    world: &mut GameWorld,
    plot_id: u32,
    backer_id: CharacterId,
) -> Result<(), String> {
    let backer = world.character(backer_id).ok_or("Backer not found")?;
    if !backer.alive {
        return Err("Backer is dead".to_string());
    }
    let backer_faction = backer.faction;

    let plot = world
        .plots
        .iter()
        .find(|p| p.id == plot_id)
        .ok_or("Plot not found")?;

    let instigator_faction = world
        .character(plot.instigator)
        .map(|c| c.faction)
        .ok_or("Instigator not found")?;

    if backer_faction != instigator_faction {
        return Err("Backer must be from the instigator's faction".to_string());
    }

    if plot.backers.contains(&backer_id) {
        return Err("Already backing this plot".to_string());
    }

    // We need to mutably borrow the plot.
    let plot = world
        .plots
        .iter_mut()
        .find(|p| p.id == plot_id)
        .ok_or("Plot not found")?;
    plot.backers.push(backer_id);

    Ok(())
}

/// Boost discovery chance for this turn by using a spymaster to investigate.
///
/// This immediately runs a discovery check with a boosted spymaster.
/// Returns any events generated (discovery events).
pub fn investigate_plot(
    world: &mut GameWorld,
    spymaster_id: CharacterId,
    rng: &mut DeterministicRng,
) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;

    let spymaster = match world.character(spymaster_id) {
        Some(c) if c.alive && c.role == CharacterRole::Spymaster => c,
        _ => return events,
    };
    let spy_faction = spymaster.faction;
    let spy_eff = spymaster.effective_stats();
    let mut discovery_power = spy_eff.intrigue.integer() * 10;

    // Check if the faction ruler is Paranoid (bonus to discovery).
    let ruler_paranoid = world
        .realms
        .iter()
        .find(|r| r.faction == spy_faction)
        .and_then(|r| world.character(r.ruler))
        .map(|c| c.traits.contains(&Trait::Paranoid))
        .unwrap_or(false);
    if ruler_paranoid {
        discovery_power += PARANOID_DISCOVERY_BONUS;
    }

    // Investigation doubles the discovery power for this check.
    discovery_power *= 2;

    // Check all active plots targeting this faction.
    let plots_targeting_us: Vec<u32> = world
        .plots
        .iter()
        .filter(|p| {
            world
                .character(p.target)
                .map(|c| c.faction == spy_faction)
                .unwrap_or(false)
        })
        .map(|p| p.id)
        .collect();

    let mut discovered_ids: Vec<u32> = Vec::new();

    for pid in &plots_targeting_us {
        let secrecy = world
            .plots
            .iter()
            .find(|p| p.id == *pid)
            .map(|p| p.secrecy.integer())
            .unwrap_or(i64::MAX);

        if discovery_power > secrecy {
            let instigator_name = world
                .plots
                .iter()
                .find(|p| p.id == *pid)
                .and_then(|p| world.character(p.instigator))
                .map(|c| c.name.clone())
                .unwrap_or_else(|| "Unknown".to_string());
            let target_name = world
                .plots
                .iter()
                .find(|p| p.id == *pid)
                .and_then(|p| world.character(p.target))
                .map(|c| c.name.clone())
                .unwrap_or_else(|| "Unknown".to_string());
            let discovered_by = world
                .character(spymaster_id)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| "Spymaster".to_string());

            events.push(GameEvent::PlotDiscovered {
                instigator_name,
                target_name,
                discovered_by,
                turn,
            });
            apply_discovery_penalty(world, *pid, turn);
            discovered_ids.push(*pid);
        }
    }

    // Remove discovered plots.
    world.plots.retain(|p| !discovered_ids.contains(&p.id));

    events
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::{CharacterRole, FixedPoint, Trait, WorldConfig};

    /// Helper: create a world and set up an instigator and target in different factions.
    fn setup_intrigue_world(seed: [u8; 32]) -> GameWorld {
        let config = WorldConfig::default();
        init_world(&config, seed)
    }

    /// Find a ruler of the given faction.
    fn find_ruler(world: &GameWorld, faction: u8) -> u32 {
        world
            .characters
            .iter()
            .find(|c| c.faction == faction && c.role == CharacterRole::Ruler && c.alive)
            .map(|c| c.id)
            .expect("Faction should have a ruler")
    }

    /// Find a spymaster of the given faction.
    fn find_spymaster(world: &GameWorld, faction: u8) -> u32 {
        world
            .characters
            .iter()
            .find(|c| c.faction == faction && c.role == CharacterRole::Spymaster && c.alive)
            .map(|c| c.id)
            .expect("Faction should have a spymaster")
    }

    #[test]
    fn test_plot_advances_each_turn() {
        let mut world = setup_intrigue_world([0x42; 32]);
        let instigator = find_ruler(&world, 0);
        let target = find_ruler(&world, 1);

        launch_plot(&mut world, instigator, target, PlotType::Assassination).unwrap();
        assert_eq!(world.plots.len(), 1);
        assert_eq!(world.plots[0].progress, FixedPoint::ZERO);

        // Process one turn of intrigue.
        let mut rng = DeterministicRng::new([0x01; 32], "intrigue");
        let _events = process_intrigue(&mut world, &mut rng);

        // Progress should have advanced (instigator intrigue * 20).
        // Even if the plot was discovered/removed, if it's still there it advanced.
        if !world.plots.is_empty() {
            assert!(
                world.plots[0].progress > FixedPoint::ZERO,
                "Plot progress should advance after one turn"
            );
        }
        // (If the plot was discovered that's also a valid outcome -- the spymaster found it.)
    }

    #[test]
    fn test_plot_fires_at_threshold() {
        let mut world = setup_intrigue_world([0x42; 32]);
        let instigator = find_ruler(&world, 0);
        let target = find_ruler(&world, 1);

        launch_plot(&mut world, instigator, target, PlotType::Sabotage).unwrap();

        // Manually set progress to just below threshold.
        world.plots[0].progress = FixedPoint::from_int(999);
        // Set very high secrecy to prevent discovery.
        world.plots[0].secrecy = FixedPoint::from_int(1000);

        let mut rng = DeterministicRng::new([0x02; 32], "intrigue");
        let events = process_intrigue(&mut world, &mut rng);

        // The plot should have fired (Sabotage always succeeds once it fires).
        let succeeded = events
            .iter()
            .any(|e| matches!(e, GameEvent::PlotSucceeded { .. }));
        assert!(succeeded, "Sabotage plot should succeed when progress >= 1000");

        // Plot should be removed.
        assert!(world.plots.is_empty(), "Fired plot should be removed");
    }

    #[test]
    fn test_assassination_kills_target() {
        let mut world = setup_intrigue_world([0x42; 32]);
        let instigator = find_ruler(&world, 0);
        let target = find_ruler(&world, 1);

        // Give instigator very high intrigue for near-certain success.
        if let Some(c) = world.character_mut(instigator) {
            c.stats.intrigue = FixedPoint::from_int(20);
        }

        launch_plot(&mut world, instigator, target, PlotType::Assassination).unwrap();
        // Set progress to firing threshold.
        world.plots[0].progress = FixedPoint::from_int(999);
        world.plots[0].secrecy = FixedPoint::from_int(1000);

        // Run multiple attempts to account for RNG.
        let mut target_died = false;
        for seed_byte in 0u8..50 {
            // Reset target alive state if previous attempt killed them
            // (we want a fresh attempt each time).
            if let Some(c) = world.character_mut(target) {
                c.alive = true;
            }
            // Re-add plot if removed.
            if world.plots.is_empty() {
                launch_plot(&mut world, instigator, target, PlotType::Assassination).unwrap();
                world.plots.last_mut().unwrap().progress = FixedPoint::from_int(999);
                world.plots.last_mut().unwrap().secrecy = FixedPoint::from_int(1000);
            }

            let mut rng = DeterministicRng::new([seed_byte; 32], "intrigue");
            let events = process_intrigue(&mut world, &mut rng);

            let died = events.iter().any(|e| {
                matches!(
                    e,
                    GameEvent::CharacterDied {
                        cause: DeathCause::Assassination,
                        ..
                    }
                )
            });
            if died {
                target_died = true;
                break;
            }
        }

        assert!(
            target_died,
            "With intrigue=20 (95% chance), assassination should succeed at least once in 50 tries"
        );
    }

    #[test]
    fn test_plot_discovered_by_spymaster() {
        let mut world = setup_intrigue_world([0x42; 32]);
        let instigator = find_ruler(&world, 0);
        let target = find_ruler(&world, 1);

        // Give target faction's spymaster very high intrigue.
        let spymaster = find_spymaster(&world, 1);
        if let Some(c) = world.character_mut(spymaster) {
            c.stats.intrigue = FixedPoint::from_int(20);
        }

        // Give instigator low intrigue (low secrecy).
        if let Some(c) = world.character_mut(instigator) {
            c.stats.intrigue = FixedPoint::from_int(1);
        }

        launch_plot(&mut world, instigator, target, PlotType::Assassination).unwrap();
        // Low secrecy plot.
        world.plots[0].secrecy = FixedPoint::from_int(50);

        let mut rng = DeterministicRng::new([0x03; 32], "intrigue");
        let events = process_intrigue(&mut world, &mut rng);

        let discovered = events
            .iter()
            .any(|e| matches!(e, GameEvent::PlotDiscovered { .. }));
        assert!(
            discovered,
            "High-intrigue spymaster should discover low-secrecy plot"
        );
    }

    #[test]
    fn test_discovery_removes_plot() {
        let mut world = setup_intrigue_world([0x42; 32]);
        let instigator = find_ruler(&world, 0);
        let target = find_ruler(&world, 1);

        // Make discovery guaranteed.
        let spymaster = find_spymaster(&world, 1);
        if let Some(c) = world.character_mut(spymaster) {
            c.stats.intrigue = FixedPoint::from_int(20);
        }

        launch_plot(&mut world, instigator, target, PlotType::Steal).unwrap();
        world.plots[0].secrecy = FixedPoint::from_int(10); // Very low secrecy.

        assert_eq!(world.plots.len(), 1);

        let mut rng = DeterministicRng::new([0x04; 32], "intrigue");
        let _events = process_intrigue(&mut world, &mut rng);

        assert!(
            world.plots.is_empty(),
            "Discovered plot should be removed from world"
        );
    }

    #[test]
    fn test_max_plots_cap() {
        let mut world = setup_intrigue_world([0x42; 32]);
        let instigator = find_ruler(&world, 0);

        // Create MAX_ACTIVE_PLOTS + 5 plots.
        for i in 0..(MAX_ACTIVE_PLOTS + 5) as u32 {
            // Target different factions' rulers in a cycle.
            let target_faction = ((i % 6) + 1) as u8;
            let target = find_ruler(&world, target_faction);

            let plot_id = world.alloc_plot_id();
            world.plots.push(Plot {
                id: plot_id,
                plot_type: PlotType::Steal,
                instigator,
                target,
                backers: Vec::new(),
                progress: FixedPoint::ZERO,
                secrecy: FixedPoint::from_int(1000), // Very high so they don't get discovered.
                started_turn: i,
            });
        }

        assert!(world.plots.len() > MAX_ACTIVE_PLOTS);

        let mut rng = DeterministicRng::new([0x05; 32], "intrigue");
        let _events = process_intrigue(&mut world, &mut rng);

        assert!(
            world.plots.len() <= MAX_ACTIVE_PLOTS,
            "Active plots should be capped at MAX_ACTIVE_PLOTS ({}), got {}",
            MAX_ACTIVE_PLOTS,
            world.plots.len()
        );
    }

    #[test]
    fn test_opinion_penalty_on_discovery() {
        let mut world = setup_intrigue_world([0x42; 32]);
        let instigator = find_ruler(&world, 0);
        let target = find_ruler(&world, 1);

        let opinion_before = world
            .relation(0, 1)
            .map(|r| r.opinion)
            .unwrap_or(FixedPoint::ZERO);

        // Force discovery.
        let spymaster = find_spymaster(&world, 1);
        if let Some(c) = world.character_mut(spymaster) {
            c.stats.intrigue = FixedPoint::from_int(20);
        }

        launch_plot(&mut world, instigator, target, PlotType::Assassination).unwrap();
        world.plots[0].secrecy = FixedPoint::from_int(10);

        let mut rng = DeterministicRng::new([0x06; 32], "intrigue");
        let events = process_intrigue(&mut world, &mut rng);

        let discovered = events
            .iter()
            .any(|e| matches!(e, GameEvent::PlotDiscovered { .. }));

        if discovered {
            let opinion_after = world
                .relation(0, 1)
                .map(|r| r.opinion)
                .unwrap_or(FixedPoint::ZERO);
            assert!(
                opinion_after < opinion_before,
                "Opinion should decrease after plot discovery: before={}, after={}",
                opinion_before,
                opinion_after
            );
        }
    }

    #[test]
    fn test_deterministic_intrigue() {
        let mut world1 = setup_intrigue_world([0x42; 32]);
        let mut world2 = setup_intrigue_world([0x42; 32]);

        let instigator = find_ruler(&world1, 0);
        let target = find_ruler(&world1, 1);

        launch_plot(&mut world1, instigator, target, PlotType::Sabotage).unwrap();
        launch_plot(&mut world2, instigator, target, PlotType::Sabotage).unwrap();

        let block_hash = [0x07; 32];
        let mut rng1 = DeterministicRng::new(block_hash, "intrigue");
        let mut rng2 = DeterministicRng::new(block_hash, "intrigue");

        let events1 = process_intrigue(&mut world1, &mut rng1);
        let events2 = process_intrigue(&mut world2, &mut rng2);

        assert_eq!(
            events1.len(),
            events2.len(),
            "Same seed should produce same number of intrigue events"
        );
        assert_eq!(
            world1.plots.len(),
            world2.plots.len(),
            "Same seed should produce same number of remaining plots"
        );
    }
}
