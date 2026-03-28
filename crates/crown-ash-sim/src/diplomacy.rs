//! Diplomacy — vassal tribute, treaty expiration, grievance decay, coalitions.
//!
//! ## Vassal Tribute
//!
//! Vassals pay 10% of their per-turn income to their liege each turn.
//! Low opinion (<-300) risks vassal revolt.
//!
//! ## Treaty Expiration
//!
//! Treaties with `expires_turn` are removed when their expiration is reached.
//! Expired DefensiveAlliance also removes from `allies` list.
//!
//! ## Grievance Decay
//!
//! Grievances tick down `decay_turns_remaining` each turn and are removed at 0.
//!
//! ## Coalitions
//!
//! If any faction controls >40% of all provinces, all other factions at peace
//! with each other get a DefensiveAlliance against the dominant faction.
//!
//! All math uses `FixedPoint` — no floating point.

use crown_ash_types::{FixedPoint, GameEvent};
use crown_ash_types::diplomacy::ActiveTreaty;
use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Vassal tribute rate (100 = 10% of income per turn).
const VASSAL_TRIBUTE_RATE: i64 = 100; // 10% in FixedPoint raw (100/1000)

/// Opinion threshold below which vassal may revolt.
const VASSAL_REVOLT_THRESHOLD: i64 = -300_000; // -300.000

/// Vassal revolt chance per turn when opinion < threshold (1/20).
const VASSAL_REVOLT_CHANCE_NUM: u32 = 1;
const VASSAL_REVOLT_CHANCE_DEN: u32 = 20;

/// Province count percentage to trigger coalition (400 = 40%).
const COALITION_TRIGGER_PCT: i64 = 400;

/// Per-turn tribute amount per vassal province.
const TRIBUTE_PER_PROVINCE: FixedPoint = FixedPoint::from_int(5);

/// Process all diplomacy mechanics for one turn.
pub fn process_diplomacy(world: &mut GameWorld, rng: &mut DeterministicRng) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;

    // 1. Collect vassal tribute.
    collect_tribute(world, &mut events, turn);

    // 2. Check for vassal revolts.
    check_vassal_revolts(world, rng, &mut events, turn);

    // 3. Expire treaties.
    expire_treaties(world, turn);

    // 4. Decay grievances.
    decay_grievances(world);

    // 5. Check coalition triggers.
    check_coalitions(world, &mut events, turn);

    events
}

/// Vassals pay tribute to their liege.
fn collect_tribute(world: &mut GameWorld, events: &mut Vec<GameEvent>, turn: u32) {
    let realm_count = world.realms.len();

    // Collect (liege_faction, vassal_faction) pairs first.
    let mut tribute_pairs: Vec<(u8, u8)> = Vec::new();
    for idx in 0..realm_count {
        let liege_faction = world.realms[idx].faction;
        for &vassal_faction in &world.realms[idx].vassals.clone() {
            tribute_pairs.push((liege_faction, vassal_faction));
        }
    }

    for (liege_faction, vassal_faction) in tribute_pairs {
        // Tribute = TRIBUTE_PER_PROVINCE * vassal's province count.
        let vassal_provinces = world.faction_province_count(vassal_faction) as i64;
        let tribute = FixedPoint::from_raw(TRIBUTE_PER_PROVINCE.raw() * vassal_provinces);

        if tribute.raw() <= 0 {
            continue;
        }

        // Deduct from vassal treasury.
        let vassal_can_pay = world.realm_for_faction(vassal_faction)
            .map(|r| r.treasury.raw() >= tribute.raw())
            .unwrap_or(false);

        let actual_tribute = if vassal_can_pay {
            tribute
        } else {
            // Pay what they can.
            world.realm_for_faction(vassal_faction)
                .map(|r| r.treasury.max(FixedPoint::ZERO))
                .unwrap_or(FixedPoint::ZERO)
        };

        if actual_tribute.raw() <= 0 {
            continue;
        }

        if let Some(vassal_realm) = world.realm_for_faction_mut_dirty(vassal_faction) {
            vassal_realm.treasury = vassal_realm.treasury.saturating_sub(actual_tribute);
        }
        if let Some(liege_realm) = world.realm_for_faction_mut_dirty(liege_faction) {
            liege_realm.treasury += actual_tribute;
        }

        events.push(GameEvent::TreatySigned {
            faction_a: vassal_faction,
            faction_b: liege_faction,
            treaty_type: format!("Tribute: {:.3}", actual_tribute.raw() as f64 / 1000.0),
            turn,
        });
    }
}

/// Check for vassal revolts when opinion is very negative.
fn check_vassal_revolts(
    world: &mut GameWorld,
    rng: &mut DeterministicRng,
    events: &mut Vec<GameEvent>,
    turn: u32,
) {
    let realm_count = world.realms.len();

    // Collect potential revolts.
    let mut revolts: Vec<(u8, u8)> = Vec::new(); // (liege, vassal)
    for idx in 0..realm_count {
        let liege = world.realms[idx].faction;
        for &vassal in &world.realms[idx].vassals.clone() {
            let opinion = world.relation(liege, vassal)
                .map(|r| r.opinion.raw())
                .unwrap_or(0);

            if opinion < VASSAL_REVOLT_THRESHOLD
                && rng.chance(VASSAL_REVOLT_CHANCE_NUM, VASSAL_REVOLT_CHANCE_DEN)
            {
                revolts.push((liege, vassal));
            }
        }
    }

    // Apply revolts.
    for (liege, vassal) in revolts {
        // Remove vassalage.
        if let Some(realm) = world.realm_for_faction_mut_dirty(liege) {
            realm.vassals.retain(|&v| v != vassal);
            if !realm.at_war_with.contains(&vassal) {
                realm.at_war_with.push(vassal);
            }
        }
        if let Some(realm) = world.realm_for_faction_mut_dirty(vassal) {
            if !realm.at_war_with.contains(&liege) {
                realm.at_war_with.push(liege);
            }
        }

        // Set war in diplomatic relation.
        if let Some(rel) = world.relation_mut_dirty(liege, vassal) {
            rel.at_war = true;
            // Remove vassalization treaty.
            rel.treaties.retain(|t| t.treaty_type != "Vassalization");
        }

        events.push(GameEvent::WarDeclared {
            attacker: vassal,
            defender: liege,
            casus_belli: "Rebellion".into(),
            turn,
        });
    }
}

/// Remove expired treaties.
fn expire_treaties(world: &mut GameWorld, turn: u32) {
    let diplomacy_count = world.diplomacy.len();

    for idx in 0..diplomacy_count {
        let fa = world.diplomacy[idx].faction_a;
        let fb = world.diplomacy[idx].faction_b;
        let before = world.diplomacy[idx].treaties.len();

        // Check for expired defensive alliances specifically.
        let had_alliance = world.diplomacy[idx].treaties.iter()
            .any(|t| t.treaty_type == "DefensiveAlliance");

        world.diplomacy[idx].treaties.retain(|t| {
            t.expires_turn.map_or(true, |exp| exp > turn)
        });

        let lost_alliance = had_alliance && !world.diplomacy[idx].treaties.iter()
            .any(|t| t.treaty_type == "DefensiveAlliance");

        if world.diplomacy[idx].treaties.len() != before {
            world.dirty.mark_diplomacy(fa, fb);
        }

        // If defensive alliance expired, remove from allies lists.
        if lost_alliance {
            if let Some(realm) = world.realm_for_faction_mut_dirty(fa) {
                realm.allies.retain(|&a| a != fb);
            }
            if let Some(realm) = world.realm_for_faction_mut_dirty(fb) {
                realm.allies.retain(|&a| a != fa);
            }
        }
    }
}

/// Decay grievances by ticking down their remaining turns.
fn decay_grievances(world: &mut GameWorld) {
    let diplomacy_count = world.diplomacy.len();

    for idx in 0..diplomacy_count {
        let had_grievances = !world.diplomacy[idx].grievances.is_empty();

        for grievance in &mut world.diplomacy[idx].grievances {
            grievance.decay_turns_remaining = grievance.decay_turns_remaining.saturating_sub(1);
        }

        // Remove expired grievances and restore opinion.
        let expired: Vec<FixedPoint> = world.diplomacy[idx].grievances.iter()
            .filter(|g| g.decay_turns_remaining == 0)
            .map(|g| g.opinion_modifier)
            .collect();

        for modifier in &expired {
            // Restore the opinion penalty that was applied.
            world.diplomacy[idx].opinion -= *modifier; // modifier is negative, so -= adds
        }

        world.diplomacy[idx].grievances.retain(|g| g.decay_turns_remaining > 0);

        if had_grievances {
            let fa = world.diplomacy[idx].faction_a;
            let fb = world.diplomacy[idx].faction_b;
            world.dirty.mark_diplomacy(fa, fb);
        }
    }
}

/// If any faction controls >40% of provinces, form defensive coalitions.
fn check_coalitions(world: &mut GameWorld, events: &mut Vec<GameEvent>, turn: u32) {
    let total_provinces = world.provinces.len() as i64;
    if total_provinces == 0 {
        return;
    }
    let threshold = total_provinces * COALITION_TRIGGER_PCT / 1000;

    // Find dominant factions.
    let faction_count = world.factions.len();
    for f_idx in 0..faction_count {
        if !world.factions[f_idx].alive {
            continue;
        }
        let fid = world.factions[f_idx].id;
        let province_count = world.faction_province_count(fid) as i64;

        if province_count > threshold {
            // Form coalition: all other alive factions get defensive alliance with each other.
            let other_factions: Vec<u8> = world.factions.iter()
                .filter(|f| f.alive && f.id != fid)
                .map(|f| f.id)
                .collect();

            let mut coalition_formed = false;
            for i in 0..other_factions.len() {
                for j in (i + 1)..other_factions.len() {
                    let a = other_factions[i];
                    let b = other_factions[j];

                    // Only add if not already allied and not at war.
                    let already_allied = world.relation(a, b)
                        .map(|r| r.treaties.iter().any(|t| t.treaty_type == "DefensiveAlliance"))
                        .unwrap_or(false);
                    let at_war = world.at_war(a, b);

                    if !already_allied && !at_war {
                        if let Some(rel) = world.relation_mut_dirty(a, b) {
                            rel.treaties.push(ActiveTreaty {
                                treaty_type: "DefensiveAlliance".into(),
                                signed_turn: turn,
                                expires_turn: Some(turn + 50), // Coalition expires in 50 turns.
                            });
                            rel.opinion += FixedPoint::from_int(20);
                        }
                        if let Some(realm) = world.realm_for_faction_mut_dirty(a) {
                            if !realm.allies.contains(&b) {
                                realm.allies.push(b);
                            }
                        }
                        if let Some(realm) = world.realm_for_faction_mut_dirty(b) {
                            if !realm.allies.contains(&a) {
                                realm.allies.push(a);
                            }
                        }
                        coalition_formed = true;
                    }
                }
            }

            if coalition_formed {
                events.push(GameEvent::TreatySigned {
                    faction_a: other_factions[0],
                    faction_b: fid,
                    treaty_type: format!("Coalition against dominant faction {}", fid),
                    turn,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::{WorldConfig, FixedPoint};

    fn test_world() -> GameWorld {
        let config = WorldConfig::default();
        init_world(&config, [0xBB; 32])
    }

    #[test]
    fn tribute_transfers_gold() {
        let mut world = test_world();

        // Make faction 1 a vassal of faction 0.
        world.realms[0].vassals.push(1);

        let liege_gold_before = world.realms[0].treasury;
        let vassal_gold_before = world.realms[1].treasury;

        let mut rng = DeterministicRng::new([0x11; 32], "diplomacy");
        let events = process_diplomacy(&mut world, &mut rng);

        let liege_gold_after = world.realms[0].treasury;
        let vassal_gold_after = world.realms[1].treasury;

        assert!(liege_gold_after > liege_gold_before,
            "Liege should gain gold from tribute");
        assert!(vassal_gold_after < vassal_gold_before,
            "Vassal should lose gold from tribute");
    }

    #[test]
    fn treaties_expire() {
        let mut world = test_world();

        // Add a treaty that expires at turn 5.
        if let Some(rel) = world.relation_mut(0, 1) {
            rel.treaties.push(ActiveTreaty {
                treaty_type: "NonAggression".into(),
                signed_turn: 0,
                expires_turn: Some(5),
            });
        }

        world.meta.turn = 10;
        let mut rng = DeterministicRng::new([0x22; 32], "diplomacy");
        let _ = process_diplomacy(&mut world, &mut rng);

        let treaties = world.relation(0, 1)
            .map(|r| r.treaties.len())
            .unwrap_or(0);
        assert_eq!(treaties, 0, "Expired treaty should be removed");
    }

    #[test]
    fn grievances_decay() {
        let mut world = test_world();

        // Add a grievance with 3 turns remaining.
        if let Some(rel) = world.relation_mut(0, 1) {
            rel.grievances.push(crown_ash_types::diplomacy::Grievance {
                reason: "Border raid".into(),
                opinion_modifier: FixedPoint::from_int(-50),
                inflicted_turn: 0,
                decay_turns_remaining: 3,
            });
            rel.opinion = FixedPoint::from_int(-50);
        }

        let mut rng = DeterministicRng::new([0x33; 32], "diplomacy");

        // Tick 3 times.
        for _ in 0..3 {
            let _ = process_diplomacy(&mut world, &mut rng);
        }

        let grievance_count = world.relation(0, 1)
            .map(|r| r.grievances.len())
            .unwrap_or(0);
        assert_eq!(grievance_count, 0, "Grievance should have decayed");
    }

    #[test]
    fn vassal_revolt_with_bad_opinion() {
        let mut world = test_world();

        // Make faction 1 a vassal of faction 0 with terrible opinion.
        world.realms[0].vassals.push(1);
        if let Some(rel) = world.relation_mut(0, 1) {
            rel.opinion = FixedPoint::from_int(-500);
        }

        // Try many seeds for the 1/20 chance.
        let mut revolt_found = false;
        for seed in 0u8..200 {
            // Reset state.
            world.realms[0].vassals = vec![1];
            world.realms[0].at_war_with.clear();
            world.realms[1].at_war_with.clear();
            if let Some(rel) = world.relation_mut(0, 1) {
                rel.at_war = false;
                rel.opinion = FixedPoint::from_int(-500);
            }

            let mut rng = DeterministicRng::new([seed; 32], "diplomacy");
            let events = process_diplomacy(&mut world, &mut rng);
            if events.iter().any(|e| matches!(e, GameEvent::WarDeclared { casus_belli, .. } if casus_belli == "Rebellion")) {
                revolt_found = true;
                break;
            }
        }
        assert!(revolt_found, "Very negative opinion should eventually cause vassal revolt");
    }

    #[test]
    fn coalition_forms_against_dominant_faction() {
        let mut world = test_world();

        // Give faction 0 control of >40% of provinces (25 provinces total, need >10).
        // Move 7 extra provinces to faction 0 (it already has ~4).
        let provinces_to_take: Vec<u16> = world.provinces.iter()
            .filter(|p| p.controller != 0)
            .map(|p| p.id)
            .take(7)
            .collect();

        for pid in &provinces_to_take {
            if let Some(p) = world.province_mut(*pid) {
                let old_ctrl = p.controller;
                p.controller = 0;
                // Update realm province lists.
                if let Some(old_realm) = world.realm_for_faction_mut(old_ctrl) {
                    old_realm.provinces.retain(|&p| p != *pid);
                }
            }
            if let Some(realm) = world.realm_for_faction_mut(0) {
                if !realm.provinces.contains(pid) {
                    realm.provinces.push(*pid);
                }
            }
        }

        let faction0_count = world.faction_province_count(0);
        assert!(faction0_count > 10, "Faction 0 should control >40% of provinces");

        let mut rng = DeterministicRng::new([0x44; 32], "diplomacy");
        let events = process_diplomacy(&mut world, &mut rng);

        // At least one coalition event should have fired.
        let coalition_events = events.iter()
            .filter(|e| matches!(e, GameEvent::TreatySigned { treaty_type, .. } if treaty_type.contains("Coalition")))
            .count();
        assert!(coalition_events > 0, "Coalition should form against dominant faction");
    }
}
