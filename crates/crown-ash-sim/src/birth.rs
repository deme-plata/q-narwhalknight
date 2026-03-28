//! Birth system — produces new characters from married couples.
//!
//! Each turn, married couples (both alive, fertile age) have a small
//! chance of producing a child.  Children inherit averaged stats from
//! parents with random variation.
//!
//! Capped at [`MAX_BIRTHS_PER_TURN`](crown_ash_types::MAX_BIRTHS_PER_TURN)
//! to prevent gas exhaustion on pathological turns.
//!
//! All math uses `FixedPoint` — no floating point.

use crown_ash_types::{
    Character, CharacterId, CharacterRole, CharacterStats, DynastyId,
    FixedPoint, GameEvent, Trait,
};
use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Conception chance per turn, expressed as numerator/denominator.
/// 15 / 1000 = 1.5% per turn per couple.
const CONCEPTION_CHANCE_NUM: u32 = 15;
const CONCEPTION_CHANCE_DEN: u32 = 1000;

/// Minimum age for fertility (both parents).
const MIN_FERTILE_AGE: u8 = 16;

/// Maximum age for fertility (both parents).
const MAX_FERTILE_AGE: u8 = 45;

/// Random variation applied to each inherited stat (±STAT_VARIATION as FixedPoint integer).
const STAT_VARIATION: i64 = 3;

/// Character name pools per faction culture (mirrored from world_gen).
const IMPERIAL_NAMES: &[&str] = &[
    "Aldric", "Cedric", "Elowen", "Theron", "Giselle",
    "Hadrian", "Isolde", "Roderic", "Seraphina", "Valentin",
];
const FEUDAL_NAMES: &[&str] = &[
    "Gareth", "Lysandra", "Baldwin", "Rowena", "Percival",
    "Elara", "Godfrey", "Matilda", "Tristan", "Beatrice",
];
const CLERICAL_NAMES: &[&str] = &[
    "Alaric", "Cassandra", "Dorian", "Evangeline", "Ignatius",
    "Mirabel", "Peregrine", "Theodora", "Ambrose", "Celestine",
];
const MERCANTILE_NAMES: &[&str] = &[
    "Lorenzo", "Vivienne", "Prospero", "Daria", "Silvius",
    "Ophelia", "Cassius", "Julianna", "Renato", "Isabetta",
];
const NORDIC_NAMES: &[&str] = &[
    "Bjorn", "Astrid", "Gunnar", "Sigrid", "Eirik",
    "Freya", "Ragnar", "Thyra", "Leif", "Ingrid",
];
const NOMADIC_NAMES: &[&str] = &[
    "Temur", "Altani", "Borte", "Kublai", "Yesugei",
    "Mandukhai", "Batu", "Sorghaghtani", "Ariq", "Toregene",
];
const MONASTIC_NAMES: &[&str] = &[
    "Corvus", "Nocturna", "Vesper", "Morrigan", "Cinis",
    "Tenebris", "Silentio", "Umbra", "Ashen", "Griselda",
];

/// All possible traits for random selection (same pool as world_gen).
const ALL_TRAITS: &[Trait] = &[
    Trait::Brave, Trait::Just, Trait::Pious, Trait::Temperate,
    Trait::Kind, Trait::Diligent, Trait::Patient, Trait::Honest,
    Trait::Craven, Trait::Cruel, Trait::Cynical, Trait::Gluttonous,
    Trait::Wrathful, Trait::Slothful, Trait::Impatient, Trait::Deceitful,
    Trait::Ambitious, Trait::Content, Trait::Gregarious, Trait::Shy,
    Trait::Paranoid, Trait::Trusting,
    Trait::Strategist, Trait::Administrator, Trait::Theologian,
    Trait::Schemer, Trait::Scholar,
];

/// Get name pool for a given faction ID.
fn name_pool(faction_id: u8) -> &'static [&'static str] {
    match faction_id {
        0 => IMPERIAL_NAMES,
        1 => FEUDAL_NAMES,
        2 => CLERICAL_NAMES,
        3 => MERCANTILE_NAMES,
        4 => NORDIC_NAMES,
        5 => NOMADIC_NAMES,
        6 => MONASTIC_NAMES,
        _ => IMPERIAL_NAMES,
    }
}

/// Pick a deterministic name from the faction's pool.
fn pick_child_name(rng: &mut DeterministicRng, faction_id: u8) -> String {
    let pool = name_pool(faction_id);
    let idx = rng.next_u32() as usize % pool.len();
    pool[idx].to_string()
}

/// Average two `FixedPoint` values and add random variation in [-STAT_VARIATION, +STAT_VARIATION].
/// Result is clamped to [1, 20] (as FixedPoint integers).
fn inherit_stat(a: FixedPoint, b: FixedPoint, rng: &mut DeterministicRng) -> FixedPoint {
    // Average the two parent stats (integer division on raw values).
    let avg_raw = (a.raw() + b.raw()) / 2;
    let variation = rng.range(-STAT_VARIATION, STAT_VARIATION);
    let result_raw = avg_raw + variation * 1000; // variation is in integer units, raw is *1000
    // Clamp to [1.000, 20.000] in raw.
    let clamped = result_raw.clamp(1_000, 20_000);
    FixedPoint::from_raw(clamped)
}

/// Pick 1-2 traits from the combined parent traits, potentially adding one random trait.
fn inherit_traits(
    parent_a_traits: &[Trait],
    parent_b_traits: &[Trait],
    rng: &mut DeterministicRng,
) -> Vec<Trait> {
    let mut picked: Vec<Trait> = Vec::with_capacity(3);

    // Combine parent traits into a pool (deduplicated).
    let mut parent_pool: Vec<Trait> = Vec::new();
    for &t in parent_a_traits.iter().chain(parent_b_traits.iter()) {
        if !parent_pool.contains(&t) {
            parent_pool.push(t);
        }
    }

    // Pick 1-2 traits from parent pool.
    let inherit_count = if parent_pool.is_empty() {
        0
    } else {
        rng.range(1, 2.min(parent_pool.len() as i64)) as usize
    };

    for _ in 0..inherit_count {
        if parent_pool.is_empty() {
            break;
        }
        let idx = rng.next_u32() as usize % parent_pool.len();
        let t = parent_pool.remove(idx);
        picked.push(t);
    }

    // 30% chance to gain one random trait not already picked.
    if rng.chance(300, 1000) {
        for _ in 0..20 {
            let idx = rng.next_u32() as usize % ALL_TRAITS.len();
            let t = ALL_TRAITS[idx];
            if !picked.contains(&t) {
                picked.push(t);
                break;
            }
        }
    }

    picked
}

/// Process births for all eligible married couples.
///
/// A couple is eligible if:
/// - Both partners are alive.
/// - Both partners are within fertile age range (16..=45).
/// - The lower-ID partner initiates (prevents double-counting).
///
/// Returns a list of `GameEvent::CharacterBorn` events.
pub fn process_births(
    world: &mut GameWorld,
    rng: &mut DeterministicRng,
) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;

    // Collect eligible couples.  Only the partner with the lower ID initiates,
    // preventing the same couple from being processed twice.
    let couples: Vec<(CharacterId, CharacterId)> = world
        .characters
        .iter()
        .filter(|c| {
            c.alive
                && c.spouse.is_some()
                && c.age >= MIN_FERTILE_AGE
                && c.age <= MAX_FERTILE_AGE
        })
        .filter_map(|c| {
            let spouse_id = c.spouse.unwrap();
            // Only lower-ID initiates.
            if c.id >= spouse_id {
                return None;
            }
            // Verify spouse is alive and fertile.
            world.character(spouse_id).and_then(|s| {
                if s.alive && s.age >= MIN_FERTILE_AGE && s.age <= MAX_FERTILE_AGE {
                    Some((c.id, spouse_id))
                } else {
                    None
                }
            })
        })
        .collect();

    for (parent_a_id, parent_b_id) in couples {
        if events.len() >= crown_ash_types::MAX_BIRTHS_PER_TURN {
            break;
        }

        // Roll conception chance.
        if !rng.chance(CONCEPTION_CHANCE_NUM, CONCEPTION_CHANCE_DEN) {
            continue;
        }

        // Gather parent data (immutable borrows).
        let (a_stats, a_traits, a_dynasty, a_faction) = {
            let a = match world.character(parent_a_id) {
                Some(c) => c,
                None => continue,
            };
            (a.stats.clone(), a.traits.clone(), a.dynasty, a.faction)
        };
        let (b_stats, b_traits) = {
            let b = match world.character(parent_b_id) {
                Some(c) => c,
                None => continue,
            };
            (b.stats.clone(), b.traits.clone())
        };

        // Dynasty comes from parent_a (lower ID, conventionally the "father").
        let child_dynasty: DynastyId = a_dynasty;
        let child_faction: u8 = a_faction;

        // Generate child stats by averaging parents with variation.
        let child_stats = CharacterStats {
            martial: inherit_stat(a_stats.martial, b_stats.martial, rng),
            diplomacy: inherit_stat(a_stats.diplomacy, b_stats.diplomacy, rng),
            stewardship: inherit_stat(a_stats.stewardship, b_stats.stewardship, rng),
            intrigue: inherit_stat(a_stats.intrigue, b_stats.intrigue, rng),
            learning: inherit_stat(a_stats.learning, b_stats.learning, rng),
        };

        // Inherit traits from parents.
        let child_traits = inherit_traits(&a_traits, &b_traits, rng);

        // Generate child name.
        let child_name = pick_child_name(rng, child_faction);

        // Allocate a new character ID.
        let child_id = world.alloc_character_id();

        let child = Character {
            id: child_id,
            name: child_name.clone(),
            dynasty: child_dynasty,
            faction: child_faction,
            role: CharacterRole::Courtier,
            age: 0,
            alive: true,
            traits: child_traits,
            stats: child_stats,
            health: FixedPoint::from_int(1000), // newborns are healthy
            legitimacy: FixedPoint::from_int(200),
            prestige: FixedPoint::ZERO,
            heir: None,
            spouse: None,
            children: Vec::new(),
            relations: Vec::new(),
            parent: Some(parent_a_id),
            death_turn: None,
            death_cause: None,
        };

        // Add child to world.
        world.characters.push(child);

        // Track dirty state.
        world.dirty.characters_added.push(child_id);
        world.dirty.dirty_characters.insert(child_id);

        // Update parent_a's children list.
        if let Some(pa) = world.character_mut(parent_a_id) {
            pa.children.push(child_id);
        }
        world.dirty.dirty_characters.insert(parent_a_id);

        // Update parent_b's children list.
        if let Some(pb) = world.character_mut(parent_b_id) {
            pb.children.push(child_id);
        }
        world.dirty.dirty_characters.insert(parent_b_id);

        // Update dynasty members list.
        if let Some(dynasty) = world.dynasties.iter_mut().find(|d| d.id == child_dynasty) {
            dynasty.members.push(child_id);
        }
        world.dirty.dirty_dynasties.insert(child_dynasty);

        // Emit birth event.
        events.push(GameEvent::CharacterBorn {
            character_id: child_id,
            character_name: child_name,
            parent: parent_a_id,
            dynasty: child_dynasty,
            turn,
        });
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::WorldConfig;

    /// Helper: create a world and marry two characters of the same faction,
    /// both at a fertile age.
    fn setup_married_world(seed: [u8; 32]) -> GameWorld {
        let config = WorldConfig::default();
        let mut world = init_world(&config, seed);

        // Find two alive, adult, unmarried characters in faction 0.
        let candidates: Vec<CharacterId> = world
            .characters
            .iter()
            .filter(|c| {
                c.faction == 0
                    && c.alive
                    && c.age >= MIN_FERTILE_AGE
                    && c.age <= MAX_FERTILE_AGE
                    && c.spouse.is_none()
            })
            .map(|c| c.id)
            .collect();

        assert!(
            candidates.len() >= 2,
            "Need at least 2 marriageable characters in faction 0, found {}",
            candidates.len()
        );

        let a = candidates[0];
        let b = candidates[1];

        // Marry them.
        if let Some(ca) = world.character_mut(a) {
            ca.spouse = Some(b);
            ca.age = 25; // ensure fertile
        }
        if let Some(cb) = world.character_mut(b) {
            cb.spouse = Some(a);
            cb.age = 25; // ensure fertile
        }

        world
    }

    #[test]
    fn test_married_couples_produce_children() {
        let mut world = setup_married_world([0x42; 32]);
        let initial_char_count = world.characters.len();

        // Run many ticks — at 1.5% chance, we should see at least one birth in 200 tries.
        let mut total_births = 0;
        for seed_byte in 0u8..200 {
            let mut rng = DeterministicRng::new([seed_byte; 32], "birth");
            let events = process_births(&mut world, &mut rng);
            total_births += events.len();
        }

        assert!(
            total_births > 0,
            "Over 200 ticks, at least one birth should occur (1.5% chance per tick)"
        );
        assert!(
            world.characters.len() > initial_char_count,
            "New characters should have been added"
        );
    }

    #[test]
    fn test_max_births_cap() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0xBB; 32]);

        // Create many married couples to exceed the cap.
        for i in 0..20u32 {
            let a_id = world.alloc_character_id();
            let b_id = world.alloc_character_id();

            let a = Character {
                id: a_id,
                name: format!("TestParentA_{}", i),
                dynasty: 0,
                faction: 0,
                role: CharacterRole::Courtier,
                age: 25,
                alive: true,
                traits: vec![Trait::Brave],
                stats: CharacterStats {
                    martial: FixedPoint::from_int(10),
                    diplomacy: FixedPoint::from_int(10),
                    stewardship: FixedPoint::from_int(10),
                    intrigue: FixedPoint::from_int(10),
                    learning: FixedPoint::from_int(10),
                },
                health: FixedPoint::from_int(800),
                legitimacy: FixedPoint::from_int(300),
                prestige: FixedPoint::from_int(100),
                heir: None,
                spouse: Some(b_id),
                children: Vec::new(),
                relations: Vec::new(),
                parent: None,
                death_turn: None,
                death_cause: None,
            };

            let b = Character {
                id: b_id,
                name: format!("TestParentB_{}", i),
                dynasty: 0,
                faction: 0,
                role: CharacterRole::Courtier,
                age: 25,
                alive: true,
                traits: vec![Trait::Kind],
                stats: CharacterStats {
                    martial: FixedPoint::from_int(8),
                    diplomacy: FixedPoint::from_int(12),
                    stewardship: FixedPoint::from_int(8),
                    intrigue: FixedPoint::from_int(8),
                    learning: FixedPoint::from_int(12),
                },
                health: FixedPoint::from_int(800),
                legitimacy: FixedPoint::from_int(300),
                prestige: FixedPoint::from_int(100),
                heir: None,
                spouse: Some(a_id),
                children: Vec::new(),
                relations: Vec::new(),
                parent: None,
                death_turn: None,
                death_cause: None,
            };

            world.characters.push(a);
            world.characters.push(b);
        }

        // Use a seed that maximises births (try many seeds, pick the one with most).
        // With 20 couples and 100% chance hack (we can't hack chance, but with many seeds
        // we should find one that produces >=5).
        // Actually, just run many iterations and check the cap.
        for seed_byte in 0u8..255 {
            let mut rng = DeterministicRng::new([seed_byte; 32], "birth");
            let events = process_births(&mut world, &mut rng);
            assert!(
                events.len() <= crown_ash_types::MAX_BIRTHS_PER_TURN,
                "Birth count {} exceeded MAX_BIRTHS_PER_TURN {}",
                events.len(),
                crown_ash_types::MAX_BIRTHS_PER_TURN
            );
        }
    }

    #[test]
    fn test_child_stats_in_range() {
        let mut world = setup_married_world([0xCC; 32]);

        // Find the married couple.
        let couple: Option<(CharacterId, CharacterId)> = world
            .characters
            .iter()
            .filter(|c| c.alive && c.spouse.is_some())
            .find_map(|c| {
                let sid = c.spouse.unwrap();
                if c.id < sid {
                    Some((c.id, sid))
                } else {
                    None
                }
            });
        let (parent_a_id, parent_b_id) = couple.expect("Should have a married couple");

        let parent_a_stats = world.character(parent_a_id).unwrap().stats.clone();
        let parent_b_stats = world.character(parent_b_id).unwrap().stats.clone();

        // Run until we get a birth.
        let mut child_id = None;
        for seed_byte in 0u8..255 {
            let mut rng = DeterministicRng::new([seed_byte; 32], "birth");
            let events = process_births(&mut world, &mut rng);
            if let Some(GameEvent::CharacterBorn { character_id, .. }) = events.first() {
                child_id = Some(*character_id);
                break;
            }
        }

        let cid = child_id.expect("Should have produced at least one child in 255 attempts");
        let child = world.character(cid).expect("Child should exist in world");

        // Check that each stat is within expected range:
        // min(parent) - STAT_VARIATION <= child_stat <= max(parent) + STAT_VARIATION
        // (in raw terms, with 1000 multiplier for FixedPoint)
        let check_stat = |child_val: FixedPoint, a_val: FixedPoint, b_val: FixedPoint, name: &str| {
            let min_parent = a_val.raw().min(b_val.raw());
            let max_parent = a_val.raw().max(b_val.raw());
            // The average can be anywhere between min and max parent, plus variation.
            // Lower bound: (min_parent + min_parent)/2 - variation = min_parent - variation
            // Upper bound: (max_parent + max_parent)/2 + variation = max_parent + variation
            // But also clamped to [1_000, 20_000].
            let lower = (min_parent - STAT_VARIATION * 1000).max(1_000);
            let upper = (max_parent + STAT_VARIATION * 1000).min(20_000);
            assert!(
                child_val.raw() >= lower && child_val.raw() <= upper,
                "Child {} stat {} out of range [{}, {}]",
                name,
                child_val.raw(),
                lower,
                upper
            );
        };

        check_stat(child.stats.martial, parent_a_stats.martial, parent_b_stats.martial, "martial");
        check_stat(child.stats.diplomacy, parent_a_stats.diplomacy, parent_b_stats.diplomacy, "diplomacy");
        check_stat(child.stats.stewardship, parent_a_stats.stewardship, parent_b_stats.stewardship, "stewardship");
        check_stat(child.stats.intrigue, parent_a_stats.intrigue, parent_b_stats.intrigue, "intrigue");
        check_stat(child.stats.learning, parent_a_stats.learning, parent_b_stats.learning, "learning");
    }

    #[test]
    fn test_dynasty_membership_updated() {
        let mut world = setup_married_world([0xDD; 32]);

        // Find the dynasty of the first parent in the married couple.
        let parent_dynasty = world
            .characters
            .iter()
            .find(|c| c.alive && c.spouse.is_some() && c.faction == 0)
            .map(|c| c.dynasty)
            .expect("Should have a married character");

        let initial_members = world
            .dynasties
            .iter()
            .find(|d| d.id == parent_dynasty)
            .map(|d| d.members.len())
            .unwrap_or(0);

        // Run until a birth occurs.
        let mut births = 0;
        for seed_byte in 0u8..255 {
            let mut rng = DeterministicRng::new([seed_byte; 32], "birth");
            let events = process_births(&mut world, &mut rng);
            births += events.len();
            if births > 0 {
                break;
            }
        }

        assert!(births > 0, "Should have produced at least one birth");

        let final_members = world
            .dynasties
            .iter()
            .find(|d| d.id == parent_dynasty)
            .map(|d| d.members.len())
            .unwrap_or(0);

        assert!(
            final_members > initial_members,
            "Dynasty should have gained members: initial={}, final={}",
            initial_members,
            final_members
        );
    }

    #[test]
    fn test_parent_child_bidirectional() {
        let mut world = setup_married_world([0xEE; 32]);

        // Run until a birth occurs.
        let mut birth_event = None;
        for seed_byte in 0u8..255 {
            let mut rng = DeterministicRng::new([seed_byte; 32], "birth");
            let events = process_births(&mut world, &mut rng);
            if let Some(ev) = events.into_iter().next() {
                birth_event = Some(ev);
                break;
            }
        }

        let ev = birth_event.expect("Should have produced a birth");
        if let GameEvent::CharacterBorn {
            character_id,
            parent,
            ..
        } = ev
        {
            // Child references parent.
            let child = world.character(character_id).expect("Child should exist");
            assert_eq!(
                child.parent,
                Some(parent),
                "Child's parent field should reference parent_a"
            );

            // Parent_a references child.
            let pa = world.character(parent).expect("Parent A should exist");
            assert!(
                pa.children.contains(&character_id),
                "Parent A's children list should contain the child"
            );

            // Parent_b (spouse of parent_a) also references child.
            let spouse_id = pa.spouse.expect("Parent A should have a spouse");
            let pb = world.character(spouse_id).expect("Parent B should exist");
            assert!(
                pb.children.contains(&character_id),
                "Parent B's children list should contain the child"
            );
        } else {
            panic!("Expected CharacterBorn event");
        }
    }

    #[test]
    fn test_deterministic_births() {
        // Two identical worlds with the same seed should produce the same births.
        let mut world1 = setup_married_world([0xFF; 32]);
        let mut world2 = setup_married_world([0xFF; 32]);

        let block_hash = [0x42; 32];
        let mut rng1 = DeterministicRng::new(block_hash, "birth");
        let mut rng2 = DeterministicRng::new(block_hash, "birth");

        let events1 = process_births(&mut world1, &mut rng1);
        let events2 = process_births(&mut world2, &mut rng2);

        assert_eq!(
            events1.len(),
            events2.len(),
            "Same seed should produce same number of births"
        );

        // Verify same character IDs and names.
        for (e1, e2) in events1.iter().zip(events2.iter()) {
            match (e1, e2) {
                (
                    GameEvent::CharacterBorn {
                        character_id: id1,
                        character_name: name1,
                        parent: p1,
                        dynasty: d1,
                        ..
                    },
                    GameEvent::CharacterBorn {
                        character_id: id2,
                        character_name: name2,
                        parent: p2,
                        dynasty: d2,
                        ..
                    },
                ) => {
                    assert_eq!(id1, id2, "Character IDs should match");
                    assert_eq!(name1, name2, "Character names should match");
                    assert_eq!(p1, p2, "Parent IDs should match");
                    assert_eq!(d1, d2, "Dynasty IDs should match");
                }
                _ => panic!("Expected CharacterBorn events"),
            }
        }
    }
}
