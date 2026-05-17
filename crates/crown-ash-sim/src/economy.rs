//! Economy — tax collection, prosperity, and improvement effects.
//!
//! ## Tax Formula
//!
//! ```text
//! gold = population * tax_rate * prosperity / 1_000_000
//! ```
//!
//! Prosperity changes based on peace/war/improvements.
//! All values use `FixedPoint` (i64 x 1000) — no floating point.

use crown_ash_types::{FixedPoint, Improvement};
use crate::world_state::GameWorld;

/// Prosperity gain per turn from peace (no war, no unrest > 500).
const PEACE_PROSPERITY_GAIN: FixedPoint = FixedPoint::from_raw(2000); // +2.000 per turn

/// Prosperity loss per turn from war (faction at war).
const WAR_PROSPERITY_LOSS: FixedPoint = FixedPoint::from_raw(5000); // -5.000 per turn

/// Prosperity gain from Farmstead improvement.
const FARMSTEAD_PROSPERITY: FixedPoint = FixedPoint::from_raw(1000); // +1.000

/// Prosperity gain from Market improvement.
const MARKET_PROSPERITY: FixedPoint = FixedPoint::from_raw(2000); // +2.000

/// Maximum prosperity a province can reach.
const MAX_PROSPERITY: FixedPoint = FixedPoint::from_int(1000);

/// Minimum prosperity.
const MIN_PROSPERITY: FixedPoint = FixedPoint::from_raw(50000); // 50.000

/// Population growth rate per turn when prosperity > 500 (raw: 1 = 0.1%).
const POP_GROWTH_RATE: i64 = 1;

/// Population loss rate when prosperity < 200.
const POP_DECLINE_RATE: i64 = 2;

/// Run the economy phase for one turn.
///
/// For each province:
/// 1. Collect taxes into the controlling realm's treasury.
/// 2. Update prosperity based on peace/war and improvements.
/// 3. Apply improvement resource bonuses.
/// 4. Advance construction queue.
/// 5. Population growth/decline.
pub fn run_economy(world: &mut GameWorld) {
    let province_count = world.provinces.len();

    for idx in 0..province_count {
        let faction_id = world.provinces[idx].controller;
        let province_id = world.provinces[idx].id;

        // --- 1. Tax Collection ---
        let tax_gold = compute_tax(&world.provinces[idx]);
        if let Some(realm) = world.realm_for_faction_mut_dirty(faction_id) {
            realm.treasury += tax_gold;
        }

        // --- 2. Prosperity Update ---
        let at_war = world.realms.iter()
            .find(|r| r.faction == faction_id)
            .map(|r| !r.at_war_with.is_empty())
            .unwrap_or(false);

        // Mark province dirty (tax, prosperity, resources, population all change).
        world.dirty.dirty_provinces.insert(province_id);
        let prov = &mut world.provinces[idx];

        if at_war {
            prov.prosperity = prov.prosperity.saturating_sub(WAR_PROSPERITY_LOSS);
        } else if prov.unrest.raw() <= 500_000 {
            prov.prosperity = prov.prosperity.saturating_add(PEACE_PROSPERITY_GAIN);
        }

        // Improvement bonuses to prosperity.
        for imp in &prov.improvements.clone() {
            match imp {
                Improvement::Farmstead => {
                    prov.prosperity = prov.prosperity.saturating_add(FARMSTEAD_PROSPERITY);
                }
                Improvement::Market => {
                    prov.prosperity = prov.prosperity.saturating_add(MARKET_PROSPERITY);
                }
                _ => {}
            }
        }

        // Clamp prosperity.
        prov.prosperity = prov.prosperity.clamp(MIN_PROSPERITY, MAX_PROSPERITY);

        // --- 3. Improvement Resource Bonuses ---
        apply_improvement_resources(prov);

        // --- 4. Construction Queue ---
        advance_construction(&mut world.provinces[idx]);

        // --- 5. Population Growth/Decline ---
        let prov = &mut world.provinces[idx];
        if prov.prosperity.raw() > 500_000 {
            // Growth: population * POP_GROWTH_RATE / 1000
            let growth = (prov.population as i64 * POP_GROWTH_RATE / 1000).max(1) as u32;
            prov.population = prov.population.saturating_add(growth);
        } else if prov.prosperity.raw() < 200_000 {
            // Decline.
            let loss = (prov.population as i64 * POP_DECLINE_RATE / 1000).max(1) as u32;
            prov.population = prov.population.saturating_sub(loss);
            if prov.population < 100 {
                prov.population = 100; // Minimum population.
            }
        }
    }
}

/// Compute tax income for a single province.
///
/// `gold = population * tax_rate * prosperity / 1_000_000`
///
/// `tax_rate` is in FixedPoint (0..1000 raw = 0%..100%).
/// `prosperity` is in FixedPoint (0..1000000 raw for 0..1000.000).
pub fn compute_tax(province: &crown_ash_types::Province) -> FixedPoint {
    // All in raw i64 to avoid overflow: use i128 intermediate.
    let pop = province.population as i128;
    let rate = province.tax_rate.raw() as i128;
    let prosperity = province.prosperity.raw() as i128;

    // gold = pop * rate * prosperity / 1_000_000
    // rate is scaled x1000, prosperity is scaled x1000, so /1000000 normalises.
    let gold_raw = (pop * rate * prosperity / 1_000_000) as i64;
    FixedPoint::from_raw(gold_raw)
}

/// Apply per-turn resource generation from improvements.
fn apply_improvement_resources(province: &mut crown_ash_types::Province) {
    let improvements = province.improvements.clone();
    for imp in &improvements {
        match imp {
            Improvement::Farmstead => {
                province.resources.food += FixedPoint::from_int(10);
            }
            Improvement::Mine => {
                province.resources.iron += FixedPoint::from_int(5);
                province.resources.gold += FixedPoint::from_int(3);
            }
            Improvement::Lumbercamp => {
                province.resources.timber += FixedPoint::from_int(8);
            }
            Improvement::Quarry => {
                province.resources.stone += FixedPoint::from_int(6);
            }
            Improvement::Stables => {
                province.resources.horses += FixedPoint::from_int(4);
            }
            Improvement::Market => {
                province.resources.trade_goods += FixedPoint::from_int(5);
                province.resources.gold += FixedPoint::from_int(5);
            }
            Improvement::Port => {
                province.resources.trade_goods += FixedPoint::from_int(8);
                province.resources.gold += FixedPoint::from_int(4);
            }
            Improvement::Temple
            | Improvement::Fortification
            | Improvement::University
            | Improvement::Granary
            | Improvement::Hospital => {
                // These have non-resource effects handled elsewhere.
            }
        }
    }
}

/// Advance construction queues: decrement turns, complete if done.
fn advance_construction(province: &mut crown_ash_types::Province) {
    let mut completed = Vec::new();

    for entry in province.construction_queue.iter_mut() {
        if entry.1 > 0 {
            entry.1 -= 1;
        }
        if entry.1 == 0 {
            completed.push(entry.0);
        }
    }

    // Remove completed from queue and add to improvements.
    province.construction_queue.retain(|e| e.1 > 0);
    for imp in completed {
        if !province.improvements.contains(&imp) {
            province.improvements.push(imp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::WorldConfig;

    #[test]
    fn tax_collection_positive() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x33; 32]);

        // Record starting treasury.
        let starting_treasury = world.realms[0].treasury;

        run_economy(&mut world);

        // Treasury should have increased.
        assert!(
            world.realms[0].treasury.raw() > starting_treasury.raw(),
            "Treasury should grow from taxes"
        );
    }

    #[test]
    fn compute_tax_formula() {
        use crown_ash_types::Province;
        use crown_ash_types::province::{Resources, Troops, Terrain, Religion, Culture};

        let p = Province {
            id: 0,
            name: "Test".into(),
            terrain: Terrain::Plains,
            controller: 0,
            population: 10_000,
            prosperity: FixedPoint::from_int(500), // raw 500_000
            unrest: FixedPoint::ZERO,
            fortification: 0,
            religion: Religion::EmberChurch,
            culture: Culture::Imperial,
            resources: Resources::default(),
            garrison: Troops::default(),
            improvements: vec![],
            construction_queue: vec![],
            scars: vec![],
            grudges: vec![],
            last_famine_turn: None,
            last_siege_turn: None,
            tax_rate: FixedPoint::from_raw(200), // 20%
            neighbors: vec![],
            conversion_progress: None,
        };

        let gold = compute_tax(&p);
        // gold = 10000 * 200 * 500000 / 1_000_000 = 1_000_000_000 / 1_000_000 = 1000 raw
        // = 1.000 FixedPoint
        assert_eq!(gold.raw(), 1_000_000);
    }

    #[test]
    fn construction_completes() {
        use crown_ash_types::Province;
        use crown_ash_types::province::{Resources, Troops, Terrain, Religion, Culture};

        let mut p = Province {
            id: 0,
            name: "Test".into(),
            terrain: Terrain::Plains,
            controller: 0,
            population: 5000,
            prosperity: FixedPoint::from_int(500),
            unrest: FixedPoint::ZERO,
            fortification: 0,
            religion: Religion::EmberChurch,
            culture: Culture::Imperial,
            resources: Resources::default(),
            garrison: Troops::default(),
            improvements: vec![],
            construction_queue: vec![(Improvement::Farmstead, 1)],
            scars: vec![],
            grudges: vec![],
            last_famine_turn: None,
            last_siege_turn: None,
            tax_rate: FixedPoint::from_raw(200),
            neighbors: vec![],
            conversion_progress: None,
        };

        advance_construction(&mut p);
        assert!(p.improvements.contains(&Improvement::Farmstead));
        assert!(p.construction_queue.is_empty());
    }
}
