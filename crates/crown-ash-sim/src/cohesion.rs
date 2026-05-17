//! Cohesion — realm stability mechanics.
//!
//! Each of the 5 cohesion components decays toward equilibrium (500)
//! at a faction-specific rate.
//!
//! External modifiers:
//! - War reduces `commoner_mood`.
//! - Conquest reduces `regional_identity`.
//! - Strong ruler boosts `legitimacy`.
//! - Pious ruler boosts `clerical_favor`.
//!
//! All math uses `FixedPoint` — no floating point.

use crown_ash_types::FixedPoint;
use crate::world_state::GameWorld;

/// Base decay rate toward equilibrium (per turn).
/// Faction bonuses modify this: lower `cohesion_decay` bonus = slower decay.
const BASE_DECAY_RATE: FixedPoint = FixedPoint::from_raw(10); // 0.010 per turn

/// War commoner_mood penalty per turn.
const WAR_MOOD_PENALTY: FixedPoint = FixedPoint::from_raw(3000); // -3.000

/// Conquest regional_identity penalty (one-time, applied by caller).
pub const CONQUEST_IDENTITY_PENALTY: FixedPoint = FixedPoint::from_raw(30000); // -30.000

/// Strong ruler legitimacy bonus per turn (martial > 10).
const STRONG_RULER_LEGITIMACY: FixedPoint = FixedPoint::from_raw(2000); // +2.000

/// Pious ruler clerical_favor bonus per turn (learning > 10).
const PIOUS_RULER_CLERICAL: FixedPoint = FixedPoint::from_raw(2000); // +2.000

/// Diplomatic ruler fealty bonus per turn (diplomacy > 10).
const DIPLOMATIC_RULER_FEALTY: FixedPoint = FixedPoint::from_raw(1500); // +1.500

/// Good steward commoner_mood bonus per turn (stewardship > 10).
const STEWARD_MOOD_BONUS: FixedPoint = FixedPoint::from_raw(1000); // +1.000

/// High-stat threshold for ruler bonuses (10.000 in FixedPoint).
const HIGH_STAT_THRESHOLD: FixedPoint = FixedPoint::from_int(10);

/// Run the cohesion phase for one turn.
///
/// For each realm:
/// 1. Apply decay toward equilibrium (500) using the faction's decay rate.
/// 2. Apply war penalty to `commoner_mood` if at war.
/// 3. Apply ruler stat bonuses.
/// 4. Clamp all components to 0..1000.
pub fn update_cohesion(world: &mut GameWorld) {
    let realm_count = world.realms.len();

    for idx in 0..realm_count {
        let faction_id = world.realms[idx].faction;
        let at_war = !world.realms[idx].at_war_with.is_empty();
        let ruler_id = world.realms[idx].ruler;

        // Mark realm dirty — cohesion is updated every turn.
        world.dirty.dirty_realms.insert(faction_id);

        // Get faction decay modifier.
        let decay_modifier = world.faction(faction_id)
            .map(|f| f.bonuses.cohesion_decay)
            .unwrap_or(FixedPoint::from_int(1));

        // Effective decay rate = base * faction modifier / 1000.
        let effective_rate = BASE_DECAY_RATE.mul_fp(decay_modifier);

        // 1. Decay toward equilibrium.
        world.realms[idx].cohesion.decay_toward_equilibrium(effective_rate);

        // 2. War penalty.
        if at_war {
            world.realms[idx].cohesion.commoner_mood =
                world.realms[idx].cohesion.commoner_mood.saturating_sub(WAR_MOOD_PENALTY);
        }

        // 3. Ruler stat bonuses.
        let ruler_stats = world.character(ruler_id)
            .map(|c| c.effective_stats())
            .unwrap_or_default();

        if ruler_stats.martial > HIGH_STAT_THRESHOLD {
            world.realms[idx].cohesion.legitimacy =
                world.realms[idx].cohesion.legitimacy.saturating_add(STRONG_RULER_LEGITIMACY);
        }
        if ruler_stats.learning > HIGH_STAT_THRESHOLD {
            world.realms[idx].cohesion.clerical_favor =
                world.realms[idx].cohesion.clerical_favor.saturating_add(PIOUS_RULER_CLERICAL);
        }
        if ruler_stats.diplomacy > HIGH_STAT_THRESHOLD {
            world.realms[idx].cohesion.fealty =
                world.realms[idx].cohesion.fealty.saturating_add(DIPLOMATIC_RULER_FEALTY);
        }
        if ruler_stats.stewardship > HIGH_STAT_THRESHOLD {
            world.realms[idx].cohesion.commoner_mood =
                world.realms[idx].cohesion.commoner_mood.saturating_add(STEWARD_MOOD_BONUS);
        }

        // 4. Clamp.
        world.realms[idx].cohesion.clamp_all();
    }
}

/// Apply a one-time cohesion penalty for conquering a province.
///
/// Called after a province capture. Reduces `regional_identity` for the
/// conquering realm.
pub fn apply_conquest_penalty(world: &mut GameWorld, conquering_faction: u8) {
    if let Some(realm) = world.realm_for_faction_mut_dirty(conquering_faction) {
        realm.cohesion.regional_identity =
            realm.cohesion.regional_identity.saturating_sub(CONQUEST_IDENTITY_PENALTY);
        realm.cohesion.clamp_all();
    }
}

/// Check if any realm is in a critical cohesion state.
///
/// Returns a list of faction IDs whose cohesion has a component below 200.
pub fn find_critical_realms(world: &GameWorld) -> Vec<u8> {
    world.realms.iter()
        .filter(|r| r.cohesion.is_critical())
        .map(|r| r.faction)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::WorldConfig;

    #[test]
    fn cohesion_stays_in_range() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x55; 32]);

        // Run 100 turns of cohesion updates.
        for _ in 0..100 {
            update_cohesion(&mut world);
        }

        for realm in &world.realms {
            let c = &realm.cohesion;
            assert!(c.legitimacy.raw() >= 0 && c.legitimacy.raw() <= 1_000_000);
            assert!(c.fealty.raw() >= 0 && c.fealty.raw() <= 1_000_000);
            assert!(c.clerical_favor.raw() >= 0 && c.clerical_favor.raw() <= 1_000_000);
            assert!(c.commoner_mood.raw() >= 0 && c.commoner_mood.raw() <= 1_000_000);
            assert!(c.regional_identity.raw() >= 0 && c.regional_identity.raw() <= 1_000_000);
        }
    }

    #[test]
    fn war_reduces_mood() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x66; 32]);

        let mood_before = world.realms[0].cohesion.commoner_mood;

        // Set faction 0 at war.
        world.realms[0].at_war_with.push(1);

        update_cohesion(&mut world);

        let mood_after = world.realms[0].cohesion.commoner_mood;
        assert!(mood_after < mood_before, "War should reduce commoner mood");
    }

    #[test]
    fn conquest_penalty_applied() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0x77; 32]);

        let identity_before = world.realms[0].cohesion.regional_identity;

        apply_conquest_penalty(&mut world, 0);

        let identity_after = world.realms[0].cohesion.regional_identity;
        assert!(
            identity_after < identity_before,
            "Conquest should reduce regional identity"
        );
    }
}
