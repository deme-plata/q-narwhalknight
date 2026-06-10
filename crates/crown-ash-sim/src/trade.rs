//! Trade route system -- inter-province economic connections.
//!
//! Trade routes connect adjacent provinces and provide mutual prosperity
//! bonuses. Routes grow in volume over peaceful turns and are disrupted
//! by war between the controlling factions.
//!
//! ## Mechanics
//!
//! - A trade route links two adjacent provinces.
//! - Volume starts at 100 and grows by [`ROUTE_GROWTH_PER_TURN`] each peaceful turn,
//!   up to [`MAX_ROUTE_VOLUME`].
//! - When the controlling factions are at war, the route is disrupted and volume decays.
//! - Each non-disrupted route grants a prosperity bonus scaled by volume.
//! - Trade income (gold) flows into the realm treasury based on total route volume.
//! - Adjacent provinces that both have a Market improvement auto-establish routes.
//! - Routes with volume <= 0 are removed.
//!
//! All math uses `FixedPoint` (i64 x 1000) -- no floating point.

use crown_ash_types::{
    FixedPoint, GameEvent, Improvement, TradeGood, TradeRoute,
};
use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Maximum number of trade routes globally.
pub const MAX_TRADE_ROUTES: usize = 50;

/// Maximum trade routes per individual province.
pub const MAX_ROUTES_PER_PROVINCE: usize = 3;

/// Volume growth per peaceful turn (FixedPoint integer units).
pub const ROUTE_GROWTH_PER_TURN: i64 = 20;

/// Maximum route volume (FixedPoint integer units).
pub const MAX_ROUTE_VOLUME: i64 = 1000;

/// Volume decay per disrupted turn (FixedPoint integer units).
pub const ROUTE_DECAY_PER_TURN: i64 = 50;

/// Base prosperity bonus per active (non-disrupted) route per turn.
pub const PROSPERITY_BONUS_PER_ROUTE: i64 = 3;

/// Gold income factor: gold_per_route = volume * TRADE_INCOME_FACTOR / 1000.
/// At max volume (1000), this yields 10.000 gold per route per turn.
pub const TRADE_INCOME_FACTOR: i64 = 10;

/// Initial volume when a route is first established.
pub const INITIAL_VOLUME: i64 = 100;

/// Process trade routes for one turn.
///
/// 1. Update disruption status based on war state.
/// 2. Grow or decay volume.
/// 3. Apply prosperity bonuses to provinces with active routes.
/// 4. Generate trade income for realm treasuries.
/// 5. Auto-establish routes between adjacent Market provinces.
/// 6. Remove fully decayed routes (volume <= 0).
///
/// Returns a list of trade-related events.
pub fn process_trade(world: &mut GameWorld, _rng: &mut DeterministicRng) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;

    // -----------------------------------------------------------------------
    // 1 & 2. Update disruption status and grow/decay volume.
    // -----------------------------------------------------------------------
    let route_count = world.trade_routes.len();
    for idx in 0..route_count {
        let province_a_id = world.trade_routes[idx].province_a;
        let province_b_id = world.trade_routes[idx].province_b;

        // Look up controllers of both provinces.
        let controller_a = world.province(province_a_id).map(|p| p.controller);
        let controller_b = world.province(province_b_id).map(|p| p.controller);

        let was_disrupted = world.trade_routes[idx].disrupted;

        // Determine if the route should be disrupted.
        if let (Some(fa), Some(fb)) = (controller_a, controller_b) {
            if fa != fb {
                // Different factions -- check if at war.
                let at_war = world.at_war(fa, fb);
                world.trade_routes[idx].disrupted = at_war;
            } else {
                // Same faction -- always peaceful.
                world.trade_routes[idx].disrupted = false;
            }
        }

        // Grow or decay volume.
        if !world.trade_routes[idx].disrupted {
            let new_vol = world.trade_routes[idx].volume
                + FixedPoint::from_int(ROUTE_GROWTH_PER_TURN);
            world.trade_routes[idx].volume = new_vol
                .clamp(FixedPoint::ZERO, FixedPoint::from_int(MAX_ROUTE_VOLUME));
        } else {
            let new_vol = world.trade_routes[idx].volume
                - FixedPoint::from_int(ROUTE_DECAY_PER_TURN);
            world.trade_routes[idx].volume = if new_vol < FixedPoint::ZERO {
                FixedPoint::ZERO
            } else {
                new_vol
            };

            // Emit disruption event on transition from active to disrupted.
            if !was_disrupted {
                events.push(GameEvent::TradeRouteDisrupted {
                    from: province_a_id,
                    to: province_b_id,
                    reason: "War between controlling factions".to_string(),
                    turn,
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. Apply prosperity bonuses to provinces with active routes.
    // -----------------------------------------------------------------------
    // Collect bonuses per province to avoid multiple mutable borrows.
    let mut prosperity_bonuses: Vec<(u16, FixedPoint)> = Vec::new();
    for route in &world.trade_routes {
        if route.disrupted || route.volume <= FixedPoint::ZERO {
            continue;
        }
        // Bonus scales with volume: actual = base * volume / MAX_ROUTE_VOLUME
        let volume_fraction = route.volume.div_fp(FixedPoint::from_int(MAX_ROUTE_VOLUME));
        let bonus = FixedPoint::from_int(PROSPERITY_BONUS_PER_ROUTE).mul_fp(volume_fraction);

        prosperity_bonuses.push((route.province_a, bonus));
        prosperity_bonuses.push((route.province_b, bonus));
    }

    // Apply accumulated prosperity bonuses.
    for (prov_id, bonus) in &prosperity_bonuses {
        if let Some(prov) = world.province_mut_dirty(*prov_id) {
            prov.prosperity = prov.prosperity.saturating_add(*bonus);
            // Clamp to [0, 1000].
            prov.prosperity = prov.prosperity
                .clamp(FixedPoint::ZERO, FixedPoint::from_int(1000));
        }
    }

    // -----------------------------------------------------------------------
    // 4. Generate trade income for realm treasuries.
    // -----------------------------------------------------------------------
    // Accumulate gold per faction from all routes touching their provinces.
    let mut faction_income: Vec<(u8, FixedPoint)> = Vec::new();
    for route in &world.trade_routes {
        if route.disrupted || route.volume <= FixedPoint::ZERO {
            continue;
        }
        // Gold = volume * TRADE_INCOME_FACTOR / 1000
        let income = route.volume.mul_fp(FixedPoint::from_int(TRADE_INCOME_FACTOR))
            .div_fp(FixedPoint::from_int(1000));

        if let Some(prov_a) = world.province(route.province_a) {
            faction_income.push((prov_a.controller, income));
        }
        if let Some(prov_b) = world.province(route.province_b) {
            faction_income.push((prov_b.controller, income));
        }
    }

    // Apply trade income to realm treasuries.
    for (faction_id, gold) in &faction_income {
        if let Some(realm) = world.realm_for_faction_mut_dirty(*faction_id) {
            realm.treasury += *gold;
        }
    }

    // -----------------------------------------------------------------------
    // 5. Auto-establish routes between adjacent provinces with Market.
    // -----------------------------------------------------------------------
    if world.trade_routes.len() < MAX_TRADE_ROUTES {
        let mut new_routes: Vec<(u16, u16)> = Vec::new();

        // Find all provinces with a Market improvement.
        let market_provinces: Vec<u16> = world.provinces.iter()
            .filter(|p| p.improvements.contains(&Improvement::Market))
            .map(|p| p.id)
            .collect();

        for &prov_a_id in &market_provinces {
            // Count existing routes for this province.
            let routes_a = count_routes_for_province(&world.trade_routes, prov_a_id)
                + new_routes.iter().filter(|&&(a, b)| a == prov_a_id || b == prov_a_id).count();
            if routes_a >= MAX_ROUTES_PER_PROVINCE {
                continue;
            }

            let neighbors: Vec<u16> = world.province(prov_a_id)
                .map(|p| p.neighbors.clone())
                .unwrap_or_default();

            for &prov_b_id in &neighbors {
                // Check that neighbor also has Market.
                if !market_provinces.contains(&prov_b_id) {
                    continue;
                }

                // Check no existing route between these two.
                let already_exists = world.trade_routes.iter().any(|r| {
                    (r.province_a == prov_a_id && r.province_b == prov_b_id)
                        || (r.province_a == prov_b_id && r.province_b == prov_a_id)
                });
                if already_exists {
                    continue;
                }

                // Avoid duplicate new routes (canonicalize order).
                let canonical = (prov_a_id.min(prov_b_id), prov_a_id.max(prov_b_id));
                if new_routes.contains(&canonical) {
                    continue;
                }

                // Check per-province cap for b.
                let routes_b = count_routes_for_province(&world.trade_routes, prov_b_id)
                    + new_routes.iter().filter(|&&(a, b)| a == prov_b_id || b == prov_b_id).count();
                if routes_b >= MAX_ROUTES_PER_PROVINCE {
                    continue;
                }

                // Check global cap.
                if world.trade_routes.len() + new_routes.len() >= MAX_TRADE_ROUTES {
                    break;
                }

                new_routes.push(canonical);
            }

            if world.trade_routes.len() + new_routes.len() >= MAX_TRADE_ROUTES {
                break;
            }
        }

        // Actually create the new routes.
        for (a_id, b_id) in new_routes {
            let goods_a = determine_trade_good_for_province(world, a_id);
            let route_id = world.alloc_trade_route_id();
            let route = TradeRoute {
                id: route_id,
                province_a: a_id,
                province_b: b_id,
                goods: goods_a,
                volume: FixedPoint::from_int(INITIAL_VOLUME),
                established_turn: turn,
                disrupted: false,
            };
            world.trade_routes.push(route);

            let goods_label = goods_a.label().to_string();
            events.push(GameEvent::TradeRouteEstablished {
                from: a_id,
                to: b_id,
                goods: goods_label,
                turn,
            });
        }
    }

    // -----------------------------------------------------------------------
    // 6. Remove fully decayed routes (volume <= 0).
    // -----------------------------------------------------------------------
    world.trade_routes.retain(|r| r.volume > FixedPoint::ZERO);

    events
}

/// Count the number of existing trade routes touching a given province.
fn count_routes_for_province(routes: &[TradeRoute], province_id: u16) -> usize {
    routes.iter()
        .filter(|r| r.province_a == province_id || r.province_b == province_id)
        .count()
}

/// Determine the primary trade good exported by a province based on its improvements.
fn determine_trade_good_for_province(world: &GameWorld, province_id: u16) -> TradeGood {
    let improvements = match world.province(province_id) {
        Some(p) => &p.improvements,
        None => return TradeGood::Luxuries,
    };

    // Priority order: specific resources first, generic last.
    if improvements.contains(&Improvement::Farmstead) {
        TradeGood::Grain
    } else if improvements.contains(&Improvement::Mine) {
        TradeGood::Iron
    } else if improvements.contains(&Improvement::Lumbercamp) {
        TradeGood::Timber
    } else if improvements.contains(&Improvement::Quarry) {
        TradeGood::Stone
    } else if improvements.contains(&Improvement::Stables) {
        TradeGood::Horses
    } else if improvements.contains(&Improvement::Temple) {
        TradeGood::Faith
    } else {
        TradeGood::Luxuries
    }
}

/// Count routes touching a specific province (public for action validation).
pub fn routes_for_province(world: &GameWorld, province_id: u16) -> usize {
    count_routes_for_province(&world.trade_routes, province_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crate::random::DeterministicRng;
    use crown_ash_types::{FixedPoint, Improvement, WorldConfig};

    /// Helper: create a default world and wire two adjacent provinces for trade testing.
    fn setup_trade_world() -> GameWorld {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);

        // Give provinces 0 and 1 Markets (they should be neighbors from the map).
        // First, verify they are neighbors.
        let neighbors_of_0: Vec<u16> = world.province(0)
            .map(|p| p.neighbors.clone())
            .unwrap_or_default();

        // Pick the first neighbor of province 0 as our trading partner.
        let partner_id = neighbors_of_0[0];

        // Give both provinces a Market improvement.
        if let Some(prov) = world.province_mut(0) {
            if !prov.improvements.contains(&Improvement::Market) {
                prov.improvements.push(Improvement::Market);
            }
        }
        if let Some(prov) = world.province_mut(partner_id) {
            if !prov.improvements.contains(&Improvement::Market) {
                prov.improvements.push(Improvement::Market);
            }
        }

        world
    }

    /// Helper: manually create a trade route between two provinces.
    fn add_route(world: &mut GameWorld, a: u16, b: u16, volume: i64) -> u32 {
        let route_id = world.alloc_trade_route_id();
        world.trade_routes.push(TradeRoute {
            id: route_id,
            province_a: a,
            province_b: b,
            goods: TradeGood::Luxuries,
            volume: FixedPoint::from_int(volume),
            established_turn: world.meta.turn,
            disrupted: false,
        });
        route_id
    }

    #[test]
    fn test_trade_route_grows_when_peaceful() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);

        // Provinces 0 and 1 are controlled by the same faction (faction 0 usually).
        // Add a route between two provinces of the same faction.
        let prov0_ctrl = world.province(0).unwrap().controller;
        // Find a neighbor of province 0 controlled by the same faction.
        let neighbors: Vec<u16> = world.province(0).unwrap().neighbors.clone();
        let same_faction_neighbor = neighbors.iter()
            .find(|&&n| world.province(n).map(|p| p.controller) == Some(prov0_ctrl))
            .copied()
            .unwrap_or(neighbors[0]);

        let initial_volume = 100;
        let route_id = add_route(&mut world, 0, same_faction_neighbor, initial_volume);

        // Process trade.
        let mut rng = DeterministicRng::new([0x01; 32], "trade");
        let _events = process_trade(&mut world, &mut rng);

        // Our manually-added route should still exist and have grown.
        let route = world.trade_routes.iter().find(|r| r.id == route_id)
            .expect("Manually-added route should still exist");
        assert!(
            route.volume.raw() > FixedPoint::from_int(initial_volume).raw(),
            "Volume should grow when peaceful: was {}, now {}",
            initial_volume,
            route.volume
        );
        assert!(!route.disrupted, "Route should not be disrupted");
    }

    #[test]
    fn test_trade_route_disrupted_by_war() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);

        // Find two provinces controlled by different factions.
        let prov0_ctrl = world.province(0).unwrap().controller;
        let neighbors: Vec<u16> = world.province(0).unwrap().neighbors.clone();
        let diff_faction_neighbor = neighbors.iter()
            .find(|&&n| {
                world.province(n)
                    .map(|p| p.controller != prov0_ctrl)
                    .unwrap_or(false)
            })
            .copied();

        // If all neighbors belong to the same faction, skip this test.
        // (This shouldn't happen with the 25-province map, but be safe.)
        let partner = match diff_faction_neighbor {
            Some(p) => p,
            None => return, // cannot test war disruption if no cross-faction neighbors
        };

        let partner_ctrl = world.province(partner).unwrap().controller;

        // Set factions at war.
        let realm_count = world.realms.len();
        for idx in 0..realm_count {
            if world.realms[idx].faction == prov0_ctrl && !world.realms[idx].at_war_with.contains(&partner_ctrl) {
                world.realms[idx].at_war_with.push(partner_ctrl);
            }
            if world.realms[idx].faction == partner_ctrl && !world.realms[idx].at_war_with.contains(&prov0_ctrl) {
                world.realms[idx].at_war_with.push(prov0_ctrl);
            }
        }
        if let Some(rel) = world.relation_mut(prov0_ctrl, partner_ctrl) {
            rel.at_war = true;
        }

        // Add a route.
        add_route(&mut world, 0, partner, 500);

        // Process trade.
        let mut rng = DeterministicRng::new([0x02; 32], "trade");
        let events = process_trade(&mut world, &mut rng);

        // Our manually-added route should be disrupted.
        let route = world.trade_routes.iter().find(|r| r.province_a == 0 && r.province_b == partner)
            .expect("Manually-added route should still exist");
        assert!(route.disrupted, "Route should be disrupted when factions at war");
        assert!(
            route.volume.raw() < FixedPoint::from_int(500).raw(),
            "Volume should decay when disrupted"
        );

        // Should have a disruption event.
        let disruption_events: Vec<_> = events.iter()
            .filter(|e| matches!(e, GameEvent::TradeRouteDisrupted { .. }))
            .collect();
        assert!(!disruption_events.is_empty(), "Should emit disruption event");
    }

    #[test]
    fn test_prosperity_bonus_applied() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);

        // Find a same-faction neighbor pair.
        let prov0_ctrl = world.province(0).unwrap().controller;
        let neighbors: Vec<u16> = world.province(0).unwrap().neighbors.clone();
        let partner = neighbors.iter()
            .find(|&&n| world.province(n).map(|p| p.controller) == Some(prov0_ctrl))
            .copied()
            .unwrap_or(neighbors[0]);

        // Record starting prosperity.
        let start_prosperity_a = world.province(0).unwrap().prosperity;
        let start_prosperity_b = world.province(partner).unwrap().prosperity;

        // Add a high-volume route.
        add_route(&mut world, 0, partner, 800);

        // Process trade.
        let mut rng = DeterministicRng::new([0x03; 32], "trade");
        let _events = process_trade(&mut world, &mut rng);

        // Both provinces should have gained prosperity.
        let end_prosperity_a = world.province(0).unwrap().prosperity;
        let end_prosperity_b = world.province(partner).unwrap().prosperity;

        assert!(
            end_prosperity_a.raw() > start_prosperity_a.raw(),
            "Province A prosperity should increase from trade: was {}, now {}",
            start_prosperity_a, end_prosperity_a
        );
        assert!(
            end_prosperity_b.raw() > start_prosperity_b.raw(),
            "Province B prosperity should increase from trade: was {}, now {}",
            start_prosperity_b, end_prosperity_b
        );
    }

    #[test]
    fn test_auto_establish_market_routes() {
        let mut world = setup_trade_world();

        // No trade routes yet.
        assert!(world.trade_routes.is_empty(), "Should start with no routes");

        // Process trade -- should auto-establish routes between Market provinces.
        let mut rng = DeterministicRng::new([0x04; 32], "trade");
        let events = process_trade(&mut world, &mut rng);

        // Should have at least one new route.
        assert!(
            !world.trade_routes.is_empty(),
            "Should auto-establish at least one trade route between Market provinces"
        );

        // Should have establishment events.
        let established_events: Vec<_> = events.iter()
            .filter(|e| matches!(e, GameEvent::TradeRouteEstablished { .. }))
            .collect();
        assert!(
            !established_events.is_empty(),
            "Should emit TradeRouteEstablished events"
        );

        // New routes should have initial volume.
        for route in &world.trade_routes {
            assert_eq!(
                route.volume,
                FixedPoint::from_int(INITIAL_VOLUME),
                "New routes should start with initial volume"
            );
        }
    }

    #[test]
    fn test_max_routes_per_province_cap() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);

        let neighbors: Vec<u16> = world.province(0).unwrap().neighbors.clone();

        // Add MAX_ROUTES_PER_PROVINCE routes from province 0.
        for (i, &neighbor) in neighbors.iter().take(MAX_ROUTES_PER_PROVINCE).enumerate() {
            add_route(&mut world, 0, neighbor, 100 + i as i64 * 50);
        }

        assert_eq!(
            count_routes_for_province(&world.trade_routes, 0),
            MAX_ROUTES_PER_PROVINCE,
            "Should have exactly MAX_ROUTES_PER_PROVINCE routes"
        );

        // Now give province 0 and another neighbor (beyond the cap) Markets.
        if let Some(prov) = world.province_mut(0) {
            if !prov.improvements.contains(&Improvement::Market) {
                prov.improvements.push(Improvement::Market);
            }
        }
        // Find a neighbor NOT already in a route with province 0.
        let already_routed: Vec<u16> = world.trade_routes.iter()
            .filter(|r| r.province_a == 0 || r.province_b == 0)
            .map(|r| if r.province_a == 0 { r.province_b } else { r.province_a })
            .collect();
        let extra_neighbor = neighbors.iter()
            .find(|&&n| !already_routed.contains(&n))
            .copied();

        if let Some(extra) = extra_neighbor {
            if let Some(prov) = world.province_mut(extra) {
                if !prov.improvements.contains(&Improvement::Market) {
                    prov.improvements.push(Improvement::Market);
                }
            }
        }

        let routes_before = world.trade_routes.len();

        // Process trade -- should NOT create a new route from province 0 (at cap).
        let mut rng = DeterministicRng::new([0x05; 32], "trade");
        let _events = process_trade(&mut world, &mut rng);

        let routes_from_0 = count_routes_for_province(&world.trade_routes, 0);
        assert!(
            routes_from_0 <= MAX_ROUTES_PER_PROVINCE,
            "Province 0 should not exceed MAX_ROUTES_PER_PROVINCE ({}), got {}",
            MAX_ROUTES_PER_PROVINCE,
            routes_from_0
        );
    }

    #[test]
    fn test_trade_income_to_treasury() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x42; 32]);

        let faction_id = world.province(0).unwrap().controller;

        // Find a same-faction neighbor.
        let neighbors: Vec<u16> = world.province(0).unwrap().neighbors.clone();
        let partner = neighbors.iter()
            .find(|&&n| world.province(n).map(|p| p.controller) == Some(faction_id))
            .copied()
            .unwrap_or(neighbors[0]);

        // Record starting treasury.
        let start_treasury = world.realm_for_faction(faction_id)
            .map(|r| r.treasury)
            .unwrap_or(FixedPoint::ZERO);

        // Add a high-volume route.
        add_route(&mut world, 0, partner, 500);

        // Process trade (but NOT economy -- we only want trade income).
        let mut rng = DeterministicRng::new([0x06; 32], "trade");
        let _events = process_trade(&mut world, &mut rng);

        let end_treasury = world.realm_for_faction(faction_id)
            .map(|r| r.treasury)
            .unwrap_or(FixedPoint::ZERO);

        assert!(
            end_treasury.raw() > start_treasury.raw(),
            "Treasury should increase from trade income: was {}, now {}",
            start_treasury, end_treasury
        );
    }
}
