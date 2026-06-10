//! Education — age-gated skill progression for young characters.
//!
//! ## Education Phases
//!
//! | Age    | Phase          | Effect                                    |
//! |--------|----------------|-------------------------------------------|
//! | 0-5    | Infant         | No education, stats frozen                |
//! | 6-15   | Schooling      | +1-3 stat points per year, mentored       |
//! | 16+    | Adult          | Graduation bonus: gain 1 trait             |
//!
//! ## Mentorship
//!
//! Children are auto-assigned the highest-learning adult in their faction
//! as mentor. Mentor's best stat determines the child's education focus.
//!
//! ## Stat Growth
//!
//! Each year (every 365 turns), children aged 6-15 gain:
//! ```text
//! focus_stat += 2 + mentor_bonus (0-3 based on mentor's stat / 5)
//! off_stats  += 1 (random selection)
//! ```
//!
//! ## Graduation (Age 16)
//!
//! When a child turns 16, they receive a trait based on their education:
//! - Highest stat is martial → Strategist or Brave
//! - Highest stat is diplomacy → Gregarious or Just
//! - Highest stat is stewardship → Administrator or Diligent
//! - Highest stat is intrigue → Schemer or Deceitful
//! - Highest stat is learning → Scholar or Theologian
//!
//! All math uses `FixedPoint` — no floating point.

use crown_ash_types::{CharacterRole, FixedPoint, GameEvent, Trait};
use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Minimum age for education to begin.
const EDUCATION_START_AGE: u8 = 6;

/// Age at which education ends and graduation occurs.
const EDUCATION_END_AGE: u8 = 16;

/// Base stat gain per year for the focus stat.
const FOCUS_STAT_BASE_GAIN: i64 = 2_000; // 2.000

/// Off-stat gain per year.
const OFF_STAT_GAIN: i64 = 1_000; // 1.000

/// Mentor bonus divisor: mentor_stat / this = bonus.
const MENTOR_BONUS_DIVISOR: i64 = 5_000; // mentor's stat / 5

/// Process education for all characters.
///
/// Called once per turn. Stat growth only happens on yearly boundaries
/// (every 365 turns) for characters aged 6-15.
pub fn process_education(
    world: &mut GameWorld,
    rng: &mut DeterministicRng,
) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;

    // Only process on yearly boundaries (same as aging).
    if turn == 0 || turn % 365 != 0 {
        return events;
    }

    let char_count = world.characters.len();

    // First pass: find best mentor per faction (highest learning adult).
    let faction_count = world.factions.len();
    let mut best_mentors: Vec<Option<(u32, FixedPoint, FixedPoint, FixedPoint, FixedPoint, FixedPoint)>> =
        vec![None; faction_count];

    for c in &world.characters {
        if !c.alive || c.age < EDUCATION_END_AGE {
            continue;
        }
        let faction = c.faction as usize;
        if faction >= faction_count {
            continue;
        }
        let stats = c.effective_stats();
        let learning = stats.learning;

        let dominated = best_mentors[faction]
            .map(|(_, _, _, _, _, prev_learning)| learning > prev_learning)
            .unwrap_or(true);

        if dominated {
            best_mentors[faction] = Some((
                c.id,
                stats.martial,
                stats.diplomacy,
                stats.stewardship,
                stats.intrigue,
                stats.learning,
            ));
        }
    }

    // Second pass: educate children.
    for idx in 0..char_count {
        if !world.characters[idx].alive {
            continue;
        }
        let age = world.characters[idx].age;
        let faction = world.characters[idx].faction as usize;

        if age >= EDUCATION_START_AGE && age < EDUCATION_END_AGE {
            // Active education — grow stats.
            let mentor = if faction < faction_count {
                best_mentors[faction]
            } else {
                None
            };

            // Determine focus stat from mentor's best stat.
            let (focus, mentor_bonus) = if let Some((_, mart, dipl, stew, intr, learn)) = mentor {
                let stats = [mart, dipl, stew, intr, learn];
                let max_idx = stats.iter()
                    .enumerate()
                    .max_by_key(|(_, s)| s.raw())
                    .map(|(i, _)| i)
                    .unwrap_or(4); // default to learning

                let mentor_stat = stats[max_idx];
                let bonus = (mentor_stat.raw() / MENTOR_BONUS_DIVISOR).max(0).min(3_000);
                (max_idx, bonus)
            } else {
                // No mentor — learning focus with no bonus.
                (4, 0)
            };

            // Apply focus stat gain.
            let focus_gain = FOCUS_STAT_BASE_GAIN + mentor_bonus;
            match focus {
                0 => world.characters[idx].stats.martial += FixedPoint::from_raw(focus_gain),
                1 => world.characters[idx].stats.diplomacy += FixedPoint::from_raw(focus_gain),
                2 => world.characters[idx].stats.stewardship += FixedPoint::from_raw(focus_gain),
                3 => world.characters[idx].stats.intrigue += FixedPoint::from_raw(focus_gain),
                _ => world.characters[idx].stats.learning += FixedPoint::from_raw(focus_gain),
            }

            // Apply off-stat gain to a random stat (not the focus).
            let off_stat = rng.range(0, 4) as usize;
            let actual_off = if off_stat >= focus { off_stat + 1 } else { off_stat };
            let actual_off = actual_off % 5;
            match actual_off {
                0 => world.characters[idx].stats.martial += FixedPoint::from_raw(OFF_STAT_GAIN),
                1 => world.characters[idx].stats.diplomacy += FixedPoint::from_raw(OFF_STAT_GAIN),
                2 => world.characters[idx].stats.stewardship += FixedPoint::from_raw(OFF_STAT_GAIN),
                3 => world.characters[idx].stats.intrigue += FixedPoint::from_raw(OFF_STAT_GAIN),
                _ => world.characters[idx].stats.learning += FixedPoint::from_raw(OFF_STAT_GAIN),
            }

            world.dirty.dirty_characters.insert(world.characters[idx].id);
        }

        // Graduation check — character just turned 16.
        if age == EDUCATION_END_AGE {
            let stats = &world.characters[idx].stats;
            let raw = [
                stats.martial.raw(),
                stats.diplomacy.raw(),
                stats.stewardship.raw(),
                stats.intrigue.raw(),
                stats.learning.raw(),
            ];

            let best_idx = raw.iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i)
                .unwrap_or(0);

            // Pick trait based on best stat, with RNG to choose variant.
            let coin = rng.range(0, 2);
            let new_trait = match best_idx {
                0 => if coin == 0 { Trait::Strategist } else { Trait::Brave },
                1 => if coin == 0 { Trait::Gregarious } else { Trait::Just },
                2 => if coin == 0 { Trait::Administrator } else { Trait::Diligent },
                3 => if coin == 0 { Trait::Schemer } else { Trait::Deceitful },
                _ => if coin == 0 { Trait::Scholar } else { Trait::Theologian },
            };

            // Only add if they don't already have it.
            if !world.characters[idx].traits.contains(&new_trait) {
                world.characters[idx].traits.push(new_trait);
                world.dirty.dirty_characters.insert(world.characters[idx].id);

                events.push(GameEvent::CharacterBorn {
                    character_id: world.characters[idx].id,
                    character_name: format!("{} graduated ({:?})", world.characters[idx].name, new_trait),
                    parent: world.characters[idx].parent.unwrap_or(0),
                    dynasty: world.characters[idx].dynasty,
                    turn,
                });
            }
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::{WorldConfig, FixedPoint, Character, CharacterStats};

    fn test_world() -> GameWorld {
        let config = WorldConfig::default();
        init_world(&config, [0xCC; 32])
    }

    #[test]
    fn no_education_before_age_6() {
        let mut world = test_world();

        // Add a young child.
        let cid = world.alloc_character_id();
        world.characters.push(Character {
            id: cid,
            name: "Baby".into(),
            dynasty: 0,
            faction: 0,
            role: CharacterRole::Courtier,
            age: 3,
            alive: true,
            traits: vec![],
            stats: CharacterStats::default(),
            health: FixedPoint::from_int(1000),
            legitimacy: FixedPoint::ZERO,
            prestige: FixedPoint::ZERO,
            heir: None,
            spouse: None,
            children: vec![],
            relations: Vec::new(),
            parent: None,
            death_turn: None,
            death_cause: None,
        });

        let stats_before = world.characters.last().unwrap().stats.clone();

        // Simulate yearly boundary.
        world.meta.turn = 365;
        let mut rng = DeterministicRng::new([0x11; 32], "education");
        let _ = process_education(&mut world, &mut rng);

        let child = world.character(cid).unwrap();
        assert_eq!(child.stats.martial.raw(), stats_before.martial.raw(),
            "Baby should not gain stats");
    }

    #[test]
    fn education_grows_stats() {
        let mut world = test_world();

        // Add a school-age child.
        let cid = world.alloc_character_id();
        world.characters.push(Character {
            id: cid,
            name: "Student".into(),
            dynasty: 0,
            faction: 0,
            role: CharacterRole::Courtier,
            age: 10,
            alive: true,
            traits: vec![],
            stats: CharacterStats {
                martial: FixedPoint::from_int(1),
                diplomacy: FixedPoint::from_int(1),
                stewardship: FixedPoint::from_int(1),
                intrigue: FixedPoint::from_int(1),
                learning: FixedPoint::from_int(1),
            },
            health: FixedPoint::from_int(1000),
            legitimacy: FixedPoint::ZERO,
            prestige: FixedPoint::ZERO,
            heir: None,
            spouse: None,
            children: vec![],
            relations: Vec::new(),
            parent: None,
            death_turn: None,
            death_cause: None,
        });

        let total_before: i64 = {
            let s = &world.character(cid).unwrap().stats;
            s.martial.raw() + s.diplomacy.raw() + s.stewardship.raw()
                + s.intrigue.raw() + s.learning.raw()
        };

        // Simulate yearly boundary.
        world.meta.turn = 365;
        let mut rng = DeterministicRng::new([0x22; 32], "education");
        let _ = process_education(&mut world, &mut rng);

        let total_after: i64 = {
            let s = &world.character(cid).unwrap().stats;
            s.martial.raw() + s.diplomacy.raw() + s.stewardship.raw()
                + s.intrigue.raw() + s.learning.raw()
        };

        assert!(total_after > total_before,
            "Student should have gained stats: before={} after={}", total_before, total_after);
    }

    #[test]
    fn graduation_grants_trait() {
        let mut world = test_world();

        // Add a character about to graduate.
        let cid = world.alloc_character_id();
        world.characters.push(Character {
            id: cid,
            name: "Graduate".into(),
            dynasty: 0,
            faction: 0,
            role: CharacterRole::Courtier,
            age: 16, // Just turned 16.
            alive: true,
            traits: vec![],
            stats: CharacterStats {
                martial: FixedPoint::from_int(15), // Best stat.
                diplomacy: FixedPoint::from_int(5),
                stewardship: FixedPoint::from_int(5),
                intrigue: FixedPoint::from_int(5),
                learning: FixedPoint::from_int(5),
            },
            health: FixedPoint::from_int(1000),
            legitimacy: FixedPoint::ZERO,
            prestige: FixedPoint::ZERO,
            heir: None,
            spouse: None,
            children: vec![],
            relations: Vec::new(),
            parent: None,
            death_turn: None,
            death_cause: None,
        });

        world.meta.turn = 365;
        let mut rng = DeterministicRng::new([0x33; 32], "education");
        let events = process_education(&mut world, &mut rng);

        let graduate = world.character(cid).unwrap();
        assert!(!graduate.traits.is_empty(),
            "Graduate should have received a trait");
        // Martial is highest → should be Strategist or Brave.
        assert!(
            graduate.traits.contains(&Trait::Strategist) || graduate.traits.contains(&Trait::Brave),
            "Martial-focused graduate should get Strategist or Brave, got {:?}",
            graduate.traits
        );
    }

    #[test]
    fn no_education_on_non_yearly_turns() {
        let mut world = test_world();

        let cid = world.alloc_character_id();
        world.characters.push(Character {
            id: cid,
            name: "Student".into(),
            dynasty: 0,
            faction: 0,
            role: CharacterRole::Courtier,
            age: 10,
            alive: true,
            traits: vec![],
            stats: CharacterStats::default(),
            health: FixedPoint::from_int(1000),
            legitimacy: FixedPoint::ZERO,
            prestige: FixedPoint::ZERO,
            heir: None,
            spouse: None,
            children: vec![],
            relations: Vec::new(),
            parent: None,
            death_turn: None,
            death_cause: None,
        });

        world.meta.turn = 100; // Not a yearly boundary.
        let mut rng = DeterministicRng::new([0x44; 32], "education");
        let events = process_education(&mut world, &mut rng);

        assert!(events.is_empty(), "No education events on non-yearly turns");
    }
}
