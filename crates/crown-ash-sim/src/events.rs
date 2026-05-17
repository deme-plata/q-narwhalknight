//! Random events — plague, famine, harvest, rebellion.
//!
//! Events are rolled each turn using the `DeterministicRng` derived from
//! the block hash. All probabilities use integer per-mille (1/1000) chances.
//!
//! ## Event Probabilities (per province per turn)
//!
//! | Event     | Base Chance | Condition                          |
//! |-----------|-------------|------------------------------------|
//! | Plague    | 1/1000      | Always possible                    |
//! | Famine    | 2/1000      | food < 100                         |
//! | Harvest   | 5/1000      | Has Farmstead improvement          |
//! | Rebellion | variable    | unrest > 700                       |
//!
//! All math uses `FixedPoint` — no floating point.

use crown_ash_types::{
    FixedPoint, GameEvent, Improvement,
};
use crown_ash_types::province::{ProvinceScar, ScarType};
use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Plague base chance: 1 per 1000 provinces per turn.
const PLAGUE_CHANCE_NUM: u32 = 1;
const PLAGUE_CHANCE_DEN: u32 = 100;

/// Famine chance when food is low: 2 per 1000.
const FAMINE_CHANCE_NUM: u32 = 2;
const FAMINE_CHANCE_DEN: u32 = 100;

/// Harvest chance when province has Farmstead: 5 per 1000.
const HARVEST_CHANCE_NUM: u32 = 5;
const HARVEST_CHANCE_DEN: u32 = 100;

/// Unrest threshold for rebellion chance.
const REBELLION_UNREST_THRESHOLD: FixedPoint = FixedPoint::from_int(700);

/// Plague population loss: 5-15% of population.
const PLAGUE_POP_LOSS_MIN_PCT: i64 = 5;
const PLAGUE_POP_LOSS_MAX_PCT: i64 = 15;

/// Plague prosperity penalty.
const PLAGUE_PROSPERITY_PENALTY: FixedPoint = FixedPoint::from_int(50);

/// Famine prosperity penalty.
const FAMINE_PROSPERITY_PENALTY: FixedPoint = FixedPoint::from_int(30);

/// Famine unrest increase.
const FAMINE_UNREST_INCREASE: FixedPoint = FixedPoint::from_int(50);

/// Harvest prosperity bonus.
const HARVEST_PROSPERITY_BONUS: FixedPoint = FixedPoint::from_int(20);

/// Hospital reduces plague severity by 50%.
const HOSPITAL_PLAGUE_REDUCTION: i64 = 500; // 50% of 1000

/// Granary reduces famine chance by 50%.
const GRANARY_FAMINE_REDUCTION: u32 = 2; // divide chance by 2

/// Roll random events for all provinces.
///
/// Capped at [`MAX_EVENTS_PER_TURN`](crown_ash_types::MAX_EVENTS_PER_TURN)
/// to prevent gas exhaustion on pathological turns.
pub fn roll_events(world: &mut GameWorld, rng: &mut DeterministicRng) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;
    let province_count = world.provinces.len();

    for idx in 0..province_count {
        if events.len() >= crown_ash_types::MAX_EVENTS_PER_TURN {
            break; // Work cap reached — remaining events deferred to next turn.
        }

        let province_id = world.provinces[idx].id;

        // --- Plague ---
        let has_hospital = world.provinces[idx].improvements.contains(&Improvement::Hospital);

        if rng.chance(PLAGUE_CHANCE_NUM, PLAGUE_CHANCE_DEN) {
            let severity = rng.range(300, 800);
            let effective_severity = if has_hospital {
                severity * HOSPITAL_PLAGUE_REDUCTION / 1000
            } else {
                severity
            };

            let pop_loss_pct = rng.range(PLAGUE_POP_LOSS_MIN_PCT, PLAGUE_POP_LOSS_MAX_PCT);
            let effective_loss_pct = if has_hospital {
                (pop_loss_pct * HOSPITAL_PLAGUE_REDUCTION / 1000).max(1)
            } else {
                pop_loss_pct
            };

            let pop = world.provinces[idx].population;
            let pop_lost = ((pop as i64 * effective_loss_pct) / 100).max(1) as u32;

            // Mark province dirty — plague modifies population, prosperity, and scars.
            world.dirty.dirty_provinces.insert(province_id);

            world.provinces[idx].population = pop.saturating_sub(pop_lost);
            world.provinces[idx].prosperity =
                world.provinces[idx].prosperity.saturating_sub(PLAGUE_PROSPERITY_PENALTY);
            if world.provinces[idx].prosperity.raw() < 0 {
                world.provinces[idx].prosperity = FixedPoint::ZERO;
            }

            // Add scar.
            world.provinces[idx].add_scar(ProvinceScar {
                turn_inflicted: turn,
                scar_type: ScarType::Plague,
                severity: FixedPoint::from_raw(effective_severity),
            });

            events.push(GameEvent::PlagueOutbreak {
                province: world.provinces[idx].id,
                severity: effective_severity,
                population_lost: pop_lost,
                turn,
            });
        }

        // --- Famine ---
        let food_low = world.provinces[idx].resources.food.raw() < 100_000; // < 100.000
        let has_granary = world.provinces[idx].improvements.contains(&Improvement::Granary);
        let famine_den = if has_granary {
            FAMINE_CHANCE_DEN * GRANARY_FAMINE_REDUCTION
        } else {
            FAMINE_CHANCE_DEN
        };

        if food_low && rng.chance(FAMINE_CHANCE_NUM, famine_den) {
            let severity = rng.range(200, 600);

            // Mark province dirty — famine modifies prosperity, unrest, scars.
            world.dirty.dirty_provinces.insert(province_id);

            world.provinces[idx].prosperity =
                world.provinces[idx].prosperity.saturating_sub(FAMINE_PROSPERITY_PENALTY);
            world.provinces[idx].unrest += FAMINE_UNREST_INCREASE;

            if world.provinces[idx].prosperity.raw() < 0 {
                world.provinces[idx].prosperity = FixedPoint::ZERO;
            }

            world.provinces[idx].last_famine_turn = Some(turn);

            world.provinces[idx].add_scar(ProvinceScar {
                turn_inflicted: turn,
                scar_type: ScarType::Famine,
                severity: FixedPoint::from_raw(severity),
            });

            events.push(GameEvent::Famine {
                province: world.provinces[idx].id,
                severity,
                turn,
            });
        }

        // --- Harvest ---
        let has_farmstead = world.provinces[idx].improvements.contains(&Improvement::Farmstead);
        if has_farmstead && rng.chance(HARVEST_CHANCE_NUM, HARVEST_CHANCE_DEN) {
            // Mark province dirty — harvest modifies prosperity.
            world.dirty.dirty_provinces.insert(province_id);

            world.provinces[idx].prosperity += HARVEST_PROSPERITY_BONUS;

            let prosperity_gain = HARVEST_PROSPERITY_BONUS.raw();

            // Clamp.
            let max = FixedPoint::from_int(1000);
            if world.provinces[idx].prosperity > max {
                world.provinces[idx].prosperity = max;
            }

            events.push(GameEvent::Harvest {
                province: world.provinces[idx].id,
                prosperity_gain,
                turn,
            });
        }

        // --- Rebellion ---
        if world.provinces[idx].unrest > REBELLION_UNREST_THRESHOLD {
            // Rebellion chance scales with unrest: (unrest - 700) / 10 per mille.
            let excess = (world.provinces[idx].unrest.raw() - REBELLION_UNREST_THRESHOLD.raw()).max(0);
            let chance_num = (excess / 10_000).max(1) as u32; // per 10.000 raw = per 10 unrest
            let chance_den = 1000u32;

            if rng.chance(chance_num, chance_den) {
                // Rebel army spawns from discontented population.
                let rebel_count = (world.provinces[idx].population / 20).max(50);

                // Mark province dirty — rebellion modifies unrest.
                world.dirty.dirty_provinces.insert(province_id);

                // Reduce unrest after rebellion (they let off steam).
                world.provinces[idx].unrest =
                    world.provinces[idx].unrest.saturating_sub(FixedPoint::from_int(200));

                events.push(GameEvent::Rebellion {
                    province: world.provinces[idx].id,
                    rebels: rebel_count,
                    turn,
                });
            }
        }
    }

    events
}

/// Decay province scars over time.
///
/// Each scar loses 1.000 severity per turn. Removed when severity reaches 0.
pub fn decay_scars(world: &mut GameWorld) {
    let decay_amount = FixedPoint::from_int(1);
    let province_count = world.provinces.len();

    // Decay scars.
    for idx in 0..province_count {
        let had_scars = !world.provinces[idx].scars.is_empty();
        for scar in &mut world.provinces[idx].scars {
            scar.severity = scar.severity.saturating_sub(decay_amount);
        }
        world.provinces[idx].scars.retain(|s| s.severity.raw() > 0);
        if had_scars {
            world.dirty.dirty_provinces.insert(world.provinces[idx].id);
        }
    }

    // Also decay grudges.
    let grudge_decay = FixedPoint::from_raw(500); // 0.500 per turn
    for idx in 0..province_count {
        let had_grudges = !world.provinces[idx].grudges.is_empty();
        for grudge in &mut world.provinces[idx].grudges {
            grudge.intensity = grudge.intensity.saturating_sub(grudge_decay);
        }
        world.provinces[idx].grudges.retain(|g| g.intensity.raw() > 0);
        if had_grudges {
            world.dirty.dirty_provinces.insert(world.provinces[idx].id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::WorldConfig;

    #[test]
    fn events_are_deterministic() {
        let config = WorldConfig::default();

        let mut world1 = init_world(&config, [0xBB; 32]);
        let mut world2 = init_world(&config, [0xBB; 32]);

        let mut rng1 = DeterministicRng::new([0x11; 32], "events");
        let mut rng2 = DeterministicRng::new([0x11; 32], "events");

        let events1 = roll_events(&mut world1, &mut rng1);
        let events2 = roll_events(&mut world2, &mut rng2);

        assert_eq!(events1.len(), events2.len(), "Same seed should produce same events");
    }

    #[test]
    fn scar_decay_removes_old_scars() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0xCC; 32]);

        // Add a weak scar.
        world.provinces[0].add_scar(ProvinceScar {
            turn_inflicted: 0,
            scar_type: ScarType::Plague,
            severity: FixedPoint::from_raw(500), // 0.500
        });

        assert_eq!(world.provinces[0].scars.len(), 1);

        // Decay once — should still exist (0.500 - 1.000 = 0, but saturating_sub → 0, removed).
        decay_scars(&mut world);

        assert_eq!(world.provinces[0].scars.len(), 0, "Weak scar should be removed after decay");
    }

    #[test]
    fn high_unrest_province_can_rebel() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0xDD; 32]);

        // Set very high unrest on province 0.
        world.provinces[0].unrest = FixedPoint::from_int(900);

        // Run many turns to trigger rebellion.
        let mut rebellion_found = false;
        for seed_byte in 0u8..100 {
            let mut rng = DeterministicRng::new([seed_byte; 32], "events");
            let events = roll_events(&mut world, &mut rng);
            if events.iter().any(|e| matches!(e, GameEvent::Rebellion { .. })) {
                rebellion_found = true;
                break;
            }
        }

        assert!(rebellion_found, "High unrest should eventually cause rebellion");
    }
}
