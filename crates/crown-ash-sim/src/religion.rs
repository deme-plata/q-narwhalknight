//! Religion — religious authority, gradual conversion, and holy events.
//!
//! ## Religious Authority
//!
//! Each realm has a `religious_authority` score (0-1000) based on:
//! - % of realm provinces sharing the faction's religion
//! - Temple count across realm
//! - Chaplain learning stat
//! - Ruler's Pious/Theologian traits
//!
//! ## Gradual Conversion
//!
//! `ConvertProvince` starts a multi-turn process. Progress per turn:
//! ```text
//! base = religious_authority / 100
//! chaplain_bonus = chaplain.learning * 2 (if assigned)
//! progress_per_turn = base + chaplain_bonus
//! complete when progress >= 1000
//! ```
//!
//! ## Religious Events
//!
//! - **Heresy**: chance if authority < 300, reduces clerical_favor
//! - **Miracle**: chance if authority > 700, boosts province prosperity
//!
//! All math uses `FixedPoint` — no floating point.

use crown_ash_types::{
    CharacterRole, FixedPoint, GameEvent, Improvement, Trait,
};
use crown_ash_types::province::{ProvinceScar, Religion, ScarType};
use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Temple religious authority bonus per temple in realm.
const TEMPLE_AUTHORITY_BONUS: i64 = 30_000; // +30

/// Pious trait authority bonus.
const PIOUS_TRAIT_BONUS: i64 = 50_000; // +50

/// Theologian trait authority bonus.
const THEOLOGIAN_TRAIT_BONUS: i64 = 40_000; // +40

/// Chaplain learning authority bonus per point above 5.
const CHAPLAIN_LEARNING_BONUS_PER_POINT: i64 = 5_000; // +5 per point

/// Base conversion progress divisor.
const CONVERSION_AUTHORITY_DIVISOR: i64 = 100;

/// Chaplain conversion bonus: learning * this.
const CHAPLAIN_CONVERSION_MULTIPLIER: i64 = 2_000; // 2.000

/// Heresy chance when authority < 300 (per realm per turn, 1/50).
const HERESY_CHANCE_NUM: u32 = 1;
const HERESY_CHANCE_DEN: u32 = 50;

/// Heresy authority threshold.
const HERESY_THRESHOLD: i64 = 300_000; // 300.000

/// Miracle chance when authority > 700 (per province per turn, 1/100).
const MIRACLE_CHANCE_NUM: u32 = 1;
const MIRACLE_CHANCE_DEN: u32 = 100;

/// Miracle authority threshold.
const MIRACLE_THRESHOLD: i64 = 700_000; // 700.000

/// Miracle prosperity bonus.
const MIRACLE_PROSPERITY_BONUS: FixedPoint = FixedPoint::from_int(25);

/// Heresy clerical_favor penalty.
const HERESY_CLERICAL_PENALTY: FixedPoint = FixedPoint::from_int(40);

/// Unrest added when conversion completes.
const CONVERSION_UNREST: FixedPoint = FixedPoint::from_int(80);

/// Conversion completion scar severity.
const CONVERSION_SCAR_SEVERITY: FixedPoint = FixedPoint::from_int(250);

/// Recalculate religious authority for all realms.
///
/// Authority = base(500) + unity_bonus + temple_bonus + ruler_bonus + chaplain_bonus
/// unity_bonus = (same_religion_provinces / total_provinces - 0.5) * 600
pub fn update_religious_authority(world: &mut GameWorld) {
    let realm_count = world.realms.len();

    for idx in 0..realm_count {
        let faction_id = world.realms[idx].faction;
        let faction_religion = world.faction(faction_id)
            .map(|f| f.religion)
            .unwrap_or(Religion::EmberChurch);

        let realm_provinces = &world.realms[idx].provinces;
        let total = realm_provinces.len() as i64;
        if total == 0 {
            world.realms[idx].religious_authority = FixedPoint::ZERO;
            continue;
        }

        // Count provinces with matching religion and temples.
        let mut same_religion = 0i64;
        let mut temple_count = 0i64;
        for &pid in realm_provinces {
            if let Some(p) = world.province(pid) {
                if p.religion == faction_religion {
                    same_religion += 1;
                }
                if p.improvements.contains(&Improvement::Temple) {
                    temple_count += 1;
                }
            }
        }

        // Unity bonus: scale from -300 to +300 based on religious unity.
        // At 50% unity: 0 bonus. At 100%: +300. At 0%: -300.
        let unity_ratio_1000 = same_religion * 1000 / total; // 0-1000
        let unity_bonus = (unity_ratio_1000 - 500) * 600; // scaled to FixedPoint raw

        // Temple bonus.
        let temple_bonus = temple_count * TEMPLE_AUTHORITY_BONUS;

        // Ruler trait bonus.
        let ruler_id = world.realms[idx].ruler;
        let mut ruler_bonus = 0i64;
        if let Some(ruler) = world.character(ruler_id) {
            if ruler.traits.contains(&Trait::Pious) {
                ruler_bonus += PIOUS_TRAIT_BONUS;
            }
            if ruler.traits.contains(&Trait::Theologian) {
                ruler_bonus += THEOLOGIAN_TRAIT_BONUS;
            }
        }

        // Chaplain bonus: learning stat above 5 adds authority.
        let mut chaplain_bonus = 0i64;
        for c in &world.characters {
            if c.faction == faction_id && c.alive && c.role == CharacterRole::Chaplain {
                let effective = c.effective_stats();
                let learning_above_5 = (effective.learning.raw() - 5_000).max(0);
                chaplain_bonus += learning_above_5 * CHAPLAIN_LEARNING_BONUS_PER_POINT / 1_000;
                break; // Only one chaplain per faction.
            }
        }

        // Final authority = 500 + bonuses, clamped to [0, 1000].
        let authority_raw = 500_000 + unity_bonus + temple_bonus + ruler_bonus + chaplain_bonus;
        world.realms[idx].religious_authority = FixedPoint::from_raw(authority_raw)
            .clamp(FixedPoint::ZERO, FixedPoint::from_int(1000));
    }
}

/// Advance active province conversions.
///
/// Each province with an active `conversion_progress` gains progress per turn
/// based on the controlling faction's religious authority and chaplain stats.
/// Completes when progress reaches 1000.
pub fn process_conversions(world: &mut GameWorld) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;
    let province_count = world.provinces.len();

    for idx in 0..province_count {
        let (target_religion, current_progress) = match world.provinces[idx].conversion_progress {
            Some((rel, prog)) => (rel, prog),
            None => continue,
        };

        let controller = world.provinces[idx].controller;

        // Get faction's religious authority.
        let authority = world.realm_for_faction(controller)
            .map(|r| r.religious_authority.raw())
            .unwrap_or(500_000);

        // Base progress = authority / 100 (at 500 authority → +5 per turn).
        let base_progress = authority / CONVERSION_AUTHORITY_DIVISOR;

        // Chaplain bonus.
        let mut chaplain_progress = 0i64;
        for c in &world.characters {
            if c.faction == controller && c.alive && c.role == CharacterRole::Chaplain {
                let effective = c.effective_stats();
                chaplain_progress = effective.learning.raw() * CHAPLAIN_CONVERSION_MULTIPLIER / 1_000;
                break;
            }
        }

        let progress_gain = base_progress + chaplain_progress;
        let new_progress = current_progress.raw() + progress_gain;

        if new_progress >= 1_000_000 {
            // Conversion complete!
            let old_religion = world.provinces[idx].religion;
            let old_name = format!("{:?}", old_religion);
            let new_name = format!("{:?}", target_religion);

            world.provinces[idx].religion = target_religion;
            world.provinces[idx].conversion_progress = None;
            world.provinces[idx].unrest += CONVERSION_UNREST;
            world.provinces[idx].add_scar(ProvinceScar {
                turn_inflicted: turn,
                scar_type: ScarType::ForcedConversion,
                severity: CONVERSION_SCAR_SEVERITY,
            });
            world.dirty.dirty_provinces.insert(world.provinces[idx].id);

            events.push(GameEvent::ReligiousConversion {
                province: world.provinces[idx].id,
                old_religion: old_name,
                new_religion: new_name,
                turn,
            });
        } else {
            // Update progress.
            world.provinces[idx].conversion_progress =
                Some((target_religion, FixedPoint::from_raw(new_progress)));
            world.dirty.dirty_provinces.insert(world.provinces[idx].id);
        }
    }

    events
}

/// Roll religious events: heresy and miracles.
pub fn roll_religious_events(
    world: &mut GameWorld,
    rng: &mut DeterministicRng,
) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;

    // --- Heresy: low-authority realms risk heresy ---
    let realm_count = world.realms.len();
    for idx in 0..realm_count {
        let authority = world.realms[idx].religious_authority.raw();
        if authority < HERESY_THRESHOLD && rng.chance(HERESY_CHANCE_NUM, HERESY_CHANCE_DEN) {
            let faction_id = world.realms[idx].faction;
            let severity = rng.range(200, 500);

            // Heresy reduces clerical_favor.
            world.realms[idx].cohesion.clerical_favor =
                world.realms[idx].cohesion.clerical_favor.saturating_sub(HERESY_CLERICAL_PENALTY);
            world.realms[idx].cohesion.clamp_all();
            world.dirty.dirty_realms.insert(faction_id);

            // Pick a random province in the realm for the event location.
            let province = world.realms[idx].provinces.first().copied().unwrap_or(0);

            events.push(GameEvent::Heresy {
                faction: faction_id,
                province,
                severity,
                turn,
            });
        }
    }

    // --- Miracles: high-authority provinces with temples ---
    let province_count = world.provinces.len();
    for idx in 0..province_count {
        if !world.provinces[idx].improvements.contains(&Improvement::Temple) {
            continue;
        }

        let controller = world.provinces[idx].controller;
        let authority = world.realm_for_faction(controller)
            .map(|r| r.religious_authority.raw())
            .unwrap_or(0);

        if authority > MIRACLE_THRESHOLD && rng.chance(MIRACLE_CHANCE_NUM, MIRACLE_CHANCE_DEN) {
            let prosperity_gain = MIRACLE_PROSPERITY_BONUS.raw();
            world.provinces[idx].prosperity += MIRACLE_PROSPERITY_BONUS;
            let max = FixedPoint::from_int(1000);
            if world.provinces[idx].prosperity > max {
                world.provinces[idx].prosperity = max;
            }
            world.dirty.dirty_provinces.insert(world.provinces[idx].id);

            events.push(GameEvent::Miracle {
                province: world.provinces[idx].id,
                prosperity_gain,
                turn,
            });
        }
    }

    events
}

/// Apply clerical_favor bonus/penalty based on religious authority.
///
/// Called during cohesion update phase. High authority boosts clerical_favor,
/// low authority drains it.
pub fn authority_cohesion_effect(world: &mut GameWorld) {
    let realm_count = world.realms.len();
    for idx in 0..realm_count {
        let authority = world.realms[idx].religious_authority.raw();

        // High authority (>600): +2 clerical_favor per turn.
        if authority > 600_000 {
            world.realms[idx].cohesion.clerical_favor += FixedPoint::from_int(2);
        }
        // Low authority (<400): -2 clerical_favor per turn.
        else if authority < 400_000 {
            world.realms[idx].cohesion.clerical_favor =
                world.realms[idx].cohesion.clerical_favor.saturating_sub(FixedPoint::from_int(2));
        }

        world.realms[idx].cohesion.clamp_all();
        world.dirty.dirty_realms.insert(world.realms[idx].faction);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::{WorldConfig, FixedPoint};

    fn test_world() -> GameWorld {
        let config = WorldConfig::default();
        init_world(&config, [0xAA; 32])
    }

    #[test]
    fn authority_starts_at_reasonable_level() {
        let mut world = test_world();
        update_religious_authority(&mut world);

        for realm in &world.realms {
            assert!(realm.religious_authority.raw() >= 0,
                "Authority should be non-negative for faction {}", realm.faction);
            assert!(realm.religious_authority.raw() <= 1_000_000,
                "Authority should be <= 1000 for faction {}", realm.faction);
        }
    }

    #[test]
    fn high_unity_gives_high_authority() {
        let mut world = test_world();
        let faction_religion = world.faction(0).unwrap().religion;

        // Set all provinces of faction 0 to its religion.
        for &pid in &world.realms[0].provinces.clone() {
            if let Some(p) = world.province_mut(pid) {
                p.religion = faction_religion;
            }
        }

        update_religious_authority(&mut world);
        assert!(world.realms[0].religious_authority.raw() > 500_000,
            "Full unity should give above-average authority");
    }

    #[test]
    fn low_unity_reduces_authority() {
        let mut world = test_world();

        // Set all provinces of faction 0 to a different religion.
        let foreign_religion = Religion::FrostSpirits;
        for &pid in &world.realms[0].provinces.clone() {
            if let Some(p) = world.province_mut(pid) {
                p.religion = foreign_religion;
            }
        }

        update_religious_authority(&mut world);
        assert!(world.realms[0].religious_authority.raw() < 500_000,
            "Zero unity should give below-average authority");
    }

    #[test]
    fn conversion_progresses_over_turns() {
        let mut world = test_world();
        update_religious_authority(&mut world);

        // Start a conversion on province 0.
        let target_religion = Religion::SaltCult;
        world.provinces[0].conversion_progress =
            Some((target_religion, FixedPoint::ZERO));

        // Run several turns of conversion.
        for _ in 0..10 {
            let _ = process_conversions(&mut world);
        }

        // Should have made some progress.
        match world.provinces[0].conversion_progress {
            Some((_, progress)) => {
                assert!(progress.raw() > 0, "Conversion should have made progress");
            }
            None => {
                // Conversion completed (unlikely in 10 turns but possible).
            }
        }
    }

    #[test]
    fn conversion_completes_and_changes_religion() {
        let mut world = test_world();
        update_religious_authority(&mut world);

        let original_religion = world.provinces[0].religion;
        let target_religion = if original_religion == Religion::SaltCult {
            Religion::EmberChurch
        } else {
            Religion::SaltCult
        };

        // Set progress near completion.
        world.provinces[0].conversion_progress =
            Some((target_religion, FixedPoint::from_int(990)));

        let events = process_conversions(&mut world);

        assert_eq!(world.provinces[0].religion, target_religion,
            "Province should have converted");
        assert!(world.provinces[0].conversion_progress.is_none(),
            "Progress should be cleared");
        assert!(!events.is_empty(), "Should emit ReligiousConversion event");
    }

    #[test]
    fn heresy_can_occur_at_low_authority() {
        let mut world = test_world();

        // Set low authority.
        world.realms[0].religious_authority = FixedPoint::from_int(100);
        let favor_before = world.realms[0].cohesion.clerical_favor;

        // Try many RNG seeds (1/50 chance per realm per roll).
        let mut heresy_found = false;
        for seed_hi in 0u8..5 {
            for seed_lo in 0u8..255 {
                let mut seed = [seed_hi; 32];
                seed[0] = seed_lo;
                let mut rng = DeterministicRng::new(seed, "religion");
                // Reset clerical_favor each time.
                world.realms[0].cohesion.clerical_favor = favor_before;

                let events = roll_religious_events(&mut world, &mut rng);
                if events.iter().any(|e| matches!(e, GameEvent::Heresy { .. })) {
                    heresy_found = true;
                    break;
                }
            }
            if heresy_found { break; }
        }
        assert!(heresy_found, "Low authority should eventually cause heresy");
    }

    #[test]
    fn miracle_can_occur_at_high_authority() {
        let mut world = test_world();

        // Set high authority and add temples.
        world.realms[0].religious_authority = FixedPoint::from_int(800);
        for &pid in &world.realms[0].provinces.clone() {
            if let Some(p) = world.province_mut(pid) {
                if !p.improvements.contains(&Improvement::Temple) {
                    p.improvements.push(Improvement::Temple);
                }
            }
        }

        let mut miracle_found = false;
        for seed in 0u8..200 {
            let mut rng = DeterministicRng::new([seed; 32], "religion");
            let events = roll_religious_events(&mut world, &mut rng);
            if events.iter().any(|e| matches!(e, GameEvent::Miracle { .. })) {
                miracle_found = true;
                break;
            }
        }
        assert!(miracle_found, "High authority with temples should eventually produce miracles");
    }

    #[test]
    fn authority_cohesion_effect_modifies_favor() {
        let mut world = test_world();

        // High authority should boost clerical_favor.
        world.realms[0].religious_authority = FixedPoint::from_int(800);
        let favor_before = world.realms[0].cohesion.clerical_favor;

        authority_cohesion_effect(&mut world);

        assert!(world.realms[0].cohesion.clerical_favor > favor_before,
            "High authority should boost clerical_favor");

        // Low authority should reduce it.
        world.realms[1].religious_authority = FixedPoint::from_int(200);
        let favor_before_1 = world.realms[1].cohesion.clerical_favor;

        authority_cohesion_effect(&mut world);

        assert!(world.realms[1].cohesion.clerical_favor < favor_before_1,
            "Low authority should reduce clerical_favor");
    }
}
