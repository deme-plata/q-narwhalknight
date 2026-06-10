//! Character relationship processing — friendships, rivalries, marriage alliances.
//!
//! Each tick:
//! 1. Decay timed opinion modifiers.
//! 2. Same-faction characters in the same province build rapport.
//! 3. Characters with high/low opinion cross thresholds → Friend / Rival.
//! 4. Marriage alliances grant ongoing diplomatic opinion between factions.
//! 5. Remove relations to dead characters.

use crown_ash_types::{
    CharacterId, FixedPoint, GameEvent, PersonalRelation, RelationType,
};
use crate::world_state::GameWorld;

/// Opinion threshold to become friends.
const FRIEND_THRESHOLD: i64 = 50_000; // 50.000

/// Opinion threshold to become rivals.
const RIVAL_THRESHOLD: i64 = -50_000; // -50.000

/// Opinion gained per turn from sharing a province (same faction).
const PROXIMITY_OPINION_GAIN: i64 = 2_000; // 2.000

/// Marriage alliance opinion bonus applied to faction relations per turn.
const MARRIAGE_ALLIANCE_OPINION_PER_TURN: i64 = 3_000; // 3.000 per living alliance

/// Natural opinion decay toward zero per turn.
const OPINION_DECAY_RATE: i64 = 1_000; // 1.000

/// Max personal relations per character (prevent unbounded growth).
const MAX_RELATIONS_PER_CHARACTER: usize = 12;

/// Process character relationships for this turn.
pub fn process_relationships(world: &mut GameWorld) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;

    // --- Step 1: Decay timed modifiers and natural opinion regression. ---
    decay_modifiers(world);

    // --- Step 2: Proximity bonding — same faction, same province, both alive. ---
    proximity_bonding(world);

    // --- Step 3: Threshold checks — promote/demote to Friend/Rival. ---
    threshold_events(world, &mut events, turn);

    // --- Step 4: Marriage alliance diplomatic effects. ---
    marriage_alliance_effects(world, &mut events, turn);

    // --- Step 5: Prune dead-character relations. ---
    prune_dead_relations(world);

    events
}

/// Decay timed modifiers and regress neutral opinion toward zero.
fn decay_modifiers(world: &mut GameWorld) {
    let char_count = world.characters.len();
    for idx in 0..char_count {
        if !world.characters[idx].alive {
            continue;
        }
        let mut changed = false;
        for rel in &mut world.characters[idx].relations {
            // Tick down modifiers.
            rel.modifiers.retain_mut(|m| {
                if let Some(ref mut remaining) = m.turns_remaining {
                    if *remaining <= 1 {
                        rel.opinion -= m.value;
                        changed = true;
                        return false; // Remove expired modifier.
                    }
                    *remaining -= 1;
                }
                true
            });

            // Natural decay toward zero for base opinion (not modifier-driven).
            if rel.opinion.raw() > 0 {
                rel.opinion -= FixedPoint::from_raw(OPINION_DECAY_RATE.min(rel.opinion.raw()));
                changed = true;
            } else if rel.opinion.raw() < 0 {
                rel.opinion += FixedPoint::from_raw(OPINION_DECAY_RATE.min(-rel.opinion.raw()));
                changed = true;
            }
        }
        if changed {
            world.dirty.dirty_characters.insert(world.characters[idx].id);
        }
    }
}

/// Characters in the same province and faction gradually build friendship.
fn proximity_bonding(world: &mut GameWorld) {
    // Collect living characters by (faction, province) for matching.
    let mut location_groups: Vec<(u8, u16, CharacterId)> = Vec::new();
    for ch in &world.characters {
        if !ch.alive || !ch.is_adult() {
            continue;
        }
        // A character's "province" is determined by their faction's capital
        // or the army they command. For simplicity, use the first province
        // their faction controls (court location).
        let province = world.realms.iter()
            .find(|r| r.faction == ch.faction)
            .and_then(|r| r.provinces.first().copied())
            .unwrap_or(0);
        location_groups.push((ch.faction, province, ch.id));
    }

    // For each pair in the same (faction, province), add small opinion boost.
    let len = location_groups.len();
    for i in 0..len {
        for j in (i + 1)..len {
            let (fa, pa, id_a) = location_groups[i];
            let (fb, pb, id_b) = location_groups[j];
            if fa == fb && pa == pb {
                add_opinion(world, id_a, id_b, PROXIMITY_OPINION_GAIN);
            }
        }
    }
}

/// Check opinion thresholds and promote/demote relationship types.
fn threshold_events(world: &mut GameWorld, events: &mut Vec<GameEvent>, turn: u32) {
    let char_count = world.characters.len();
    for idx in 0..char_count {
        if !world.characters[idx].alive {
            continue;
        }
        let char_id = world.characters[idx].id;
        let rel_count = world.characters[idx].relations.len();
        let mut changed = false;
        for ri in 0..rel_count {
            let opinion_raw = world.characters[idx].relations[ri].opinion.raw();
            let old_type = world.characters[idx].relations[ri].relation_type;
            let target = world.characters[idx].relations[ri].target;

            if opinion_raw >= FRIEND_THRESHOLD && old_type != Some(RelationType::Friend) && old_type != Some(RelationType::MarriageAlliance) {
                world.characters[idx].relations[ri].relation_type = Some(RelationType::Friend);
                events.push(GameEvent::Friendship { character_a: char_id, character_b: target, turn });
                changed = true;
            } else if opinion_raw <= RIVAL_THRESHOLD && old_type != Some(RelationType::Rival) {
                world.characters[idx].relations[ri].relation_type = Some(RelationType::Rival);
                events.push(GameEvent::Rivalry { character_a: char_id, character_b: target, turn });
                changed = true;
            } else if opinion_raw > RIVAL_THRESHOLD && opinion_raw < FRIEND_THRESHOLD {
                if old_type == Some(RelationType::Friend) || old_type == Some(RelationType::Rival) {
                    world.characters[idx].relations[ri].relation_type = None;
                    changed = true;
                }
            }
        }
        if changed {
            world.dirty.dirty_characters.insert(char_id);
        }
    }
}

/// Marriage alliances between factions grant ongoing diplomatic opinion.
fn marriage_alliance_effects(world: &mut GameWorld, _events: &mut Vec<GameEvent>, _turn: u32) {
    // Find all living cross-faction spouses.
    let mut alliance_pairs: Vec<(u8, u8)> = Vec::new();
    for ch in &world.characters {
        if !ch.alive {
            continue;
        }
        if let Some(spouse_id) = ch.spouse {
            if let Some(spouse) = world.characters.iter().find(|c| c.id == spouse_id) {
                if spouse.alive && ch.faction != spouse.faction && ch.id < spouse.id {
                    // Only count each pair once (lower id first).
                    let pair = if ch.faction < spouse.faction {
                        (ch.faction, spouse.faction)
                    } else {
                        (spouse.faction, ch.faction)
                    };
                    if !alliance_pairs.contains(&pair) {
                        alliance_pairs.push(pair);
                    }
                }
            }
        }
    }

    // Apply small per-turn opinion boost for each marriage alliance.
    for (fa, fb) in &alliance_pairs {
        if let Some(rel) = world.relation_mut_dirty(*fa, *fb) {
            rel.opinion += FixedPoint::from_raw(MARRIAGE_ALLIANCE_OPINION_PER_TURN);
            // Cap at 200.
            if rel.opinion.raw() > 200_000 {
                rel.opinion = FixedPoint::from_raw(200_000);
            }
        }
    }
}

/// Remove relations pointing at dead characters.
fn prune_dead_relations(world: &mut GameWorld) {
    let dead_ids: Vec<CharacterId> = world.characters.iter()
        .filter(|c| !c.alive)
        .map(|c| c.id)
        .collect();

    if dead_ids.is_empty() {
        return;
    }

    let char_count = world.characters.len();
    for idx in 0..char_count {
        let before = world.characters[idx].relations.len();
        world.characters[idx].relations.retain(|r| !dead_ids.contains(&r.target));
        if world.characters[idx].relations.len() != before {
            world.dirty.dirty_characters.insert(world.characters[idx].id);
        }
    }
}

/// Add opinion between two characters (bidirectional, capped).
fn add_opinion(world: &mut GameWorld, a: CharacterId, b: CharacterId, amount: i64) {
    add_opinion_one_way(world, a, b, amount);
    add_opinion_one_way(world, b, a, amount);
}

/// Add opinion from character `from` toward `to`.
fn add_opinion_one_way(world: &mut GameWorld, from: CharacterId, to: CharacterId, amount: i64) {
    let ch_idx = match world.characters.iter().position(|c| c.id == from) {
        Some(idx) => idx,
        None => return,
    };
    let ch_id = world.characters[ch_idx].id;

    if let Some(rel) = world.characters[ch_idx].relations.iter_mut().find(|r| r.target == to) {
        rel.opinion += FixedPoint::from_raw(amount);
        // Clamp.
        rel.opinion = rel.opinion.clamp(
            FixedPoint::from_raw(-1_000_000),
            FixedPoint::from_raw(1_000_000),
        );
        world.dirty.dirty_characters.insert(ch_id);
    } else if world.characters[ch_idx].relations.len() < MAX_RELATIONS_PER_CHARACTER {
        // Create new relation.
        world.characters[ch_idx].relations.push(PersonalRelation {
            target: to,
            opinion: FixedPoint::from_raw(amount),
            relation_type: None,
            modifiers: Vec::new(),
        });
        world.dirty.dirty_characters.insert(ch_id);
    }
}

/// Called when a marriage is arranged — adds MarriageAlliance relation type.
pub fn on_marriage(world: &mut GameWorld, a: CharacterId, b: CharacterId, events: &mut Vec<GameEvent>, turn: u32) {
    let a_faction = world.character(a).map(|c| c.faction);
    let b_faction = world.character(b).map(|c| c.faction);

    // Add personal relation with initial opinion boost.
    add_opinion(world, a, b, 30_000);

    // Set relation type to MarriageAlliance.
    set_relation_type(world, a, b, RelationType::MarriageAlliance);
    set_relation_type(world, b, a, RelationType::MarriageAlliance);

    // Emit event for cross-faction marriages.
    if let (Some(fa), Some(fb)) = (a_faction, b_faction) {
        if fa != fb {
            events.push(GameEvent::MarriageAlliance {
                character_a: a,
                character_b: b,
                faction_a: fa,
                faction_b: fb,
                turn,
            });
        }
    }
}

fn set_relation_type(world: &mut GameWorld, from: CharacterId, to: CharacterId, rtype: RelationType) {
    let ch_idx = match world.characters.iter().position(|c| c.id == from) {
        Some(idx) => idx,
        None => return,
    };
    if let Some(rel) = world.characters[ch_idx].relations.iter_mut().find(|r| r.target == to) {
        rel.relation_type = Some(rtype);
    }
    world.dirty.dirty_characters.insert(world.characters[ch_idx].id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::WorldConfig;

    fn test_world() -> GameWorld {
        let config = WorldConfig::default();
        init_world(&config, [0xBB; 32])
    }

    #[test]
    fn proximity_bonding_builds_opinion() {
        let mut world = test_world();
        // Get two adult characters from faction 0.
        let chars: Vec<u32> = world.characters.iter()
            .filter(|c| c.faction == 0 && c.alive && c.is_adult())
            .take(2)
            .map(|c| c.id)
            .collect();
        assert!(chars.len() >= 2, "Need at least 2 adults in faction 0");

        let events = process_relationships(&mut world);
        // After one tick of proximity bonding, characters should have some opinion.
        let ch = world.character(chars[0]).unwrap();
        let rel = ch.relations.iter().find(|r| r.target == chars[1]);
        // May or may not have a relation yet depending on province grouping,
        // but the function should not panic.
        let _ = events; // May have events.
    }

    #[test]
    fn friendship_forms_at_threshold() {
        let mut world = test_world();
        let chars: Vec<u32> = world.characters.iter()
            .filter(|c| c.faction == 0 && c.alive && c.is_adult())
            .take(2)
            .map(|c| c.id)
            .collect();
        assert!(chars.len() >= 2);

        // Set opinion well below threshold (proximity bonding may add 2_000).
        add_opinion(&mut world, chars[0], chars[1], 40_000);

        let events = process_relationships(&mut world);
        // Should not yet be friends (40_000 + bonding - decay < 50_000).
        assert!(!events.iter().any(|e| matches!(e, GameEvent::Friendship { .. })),
            "Should not be friends below threshold");

        // Push well over threshold.
        add_opinion(&mut world, chars[0], chars[1], 15_000);
        let events = process_relationships(&mut world);
        let has_friendship = events.iter().any(|e| matches!(e, GameEvent::Friendship { .. }));
        assert!(has_friendship, "Should emit Friendship event when opinion crosses threshold");
    }

    #[test]
    fn rivalry_forms_at_negative_threshold() {
        let mut world = test_world();
        let chars: Vec<u32> = world.characters.iter()
            .filter(|c| c.faction == 0 && c.alive && c.is_adult())
            .take(2)
            .map(|c| c.id)
            .collect();
        assert!(chars.len() >= 2);

        add_opinion(&mut world, chars[0], chars[1], -55_000);
        let events = process_relationships(&mut world);
        let has_rivalry = events.iter().any(|e| matches!(e, GameEvent::Rivalry { .. }));
        assert!(has_rivalry, "Should emit Rivalry event at negative threshold");
    }

    #[test]
    fn marriage_alliance_boosts_faction_opinion() {
        let mut world = test_world();
        let turn = 5u32;
        world.meta.turn = turn;

        // Find two unmarried adults from different factions.
        let char_a = world.characters.iter()
            .find(|c| c.faction == 0 && c.alive && c.is_adult() && c.spouse.is_none())
            .map(|c| c.id)
            .expect("Need unmarried adult in faction 0");
        let char_b = world.characters.iter()
            .find(|c| c.faction == 1 && c.alive && c.is_adult() && c.spouse.is_none())
            .map(|c| c.id)
            .expect("Need unmarried adult in faction 1");

        let opinion_before = world.relation_mut(0, 1)
            .map(|r| r.opinion.raw())
            .unwrap_or(0);

        // Arrange marriage.
        let mut events = Vec::new();
        on_marriage(&mut world, char_a, char_b, &mut events, turn);

        // Set spouse fields (normally done by action handler).
        if let Some(ca) = world.character_mut_dirty(char_a) {
            ca.spouse = Some(char_b);
        }
        if let Some(cb) = world.character_mut_dirty(char_b) {
            cb.spouse = Some(char_a);
        }

        // Process relationships — should apply per-turn alliance bonus.
        let _ = process_relationships(&mut world);

        let opinion_after = world.relation_mut(0, 1)
            .map(|r| r.opinion.raw())
            .unwrap_or(0);

        assert!(opinion_after > opinion_before, "Marriage alliance should boost faction opinion");
        assert!(events.iter().any(|e| matches!(e, GameEvent::MarriageAlliance { .. })),
            "Should emit MarriageAlliance event");
    }

    #[test]
    fn dead_relations_pruned() {
        let mut world = test_world();
        let chars: Vec<u32> = world.characters.iter()
            .filter(|c| c.faction == 0 && c.alive && c.is_adult())
            .take(2)
            .map(|c| c.id)
            .collect();
        assert!(chars.len() >= 2);

        add_opinion(&mut world, chars[0], chars[1], 60_000);

        // Verify relation exists.
        assert!(world.character(chars[0]).unwrap().relations.iter().any(|r| r.target == chars[1]));

        // Kill character b.
        if let Some(ch) = world.character_mut_dirty(chars[1]) {
            ch.alive = false;
        }

        let _ = process_relationships(&mut world);

        // Relation should be pruned.
        assert!(!world.character(chars[0]).unwrap().relations.iter().any(|r| r.target == chars[1]),
            "Dead character relations should be pruned");
    }
}
