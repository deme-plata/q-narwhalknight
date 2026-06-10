//! Succession — ruler death and heir resolution.
//!
//! When a ruler dies:
//! 1. Has designated heir with legitimacy > 300? -> Clean succession.
//! 2. No heir / low legitimacy? -> Crisis (score claimants, possibly split realm).
//! 3. Cohesion drops -300 during crisis.
//!
//! All math uses `FixedPoint` — no floating point.

use crown_ash_types::{
    CharacterId, CharacterRole, DeathCause, FixedPoint, GameEvent,
};
use crate::random::DeterministicRng;
use crate::realm_split;
use crate::world_state::GameWorld;

/// Minimum heir legitimacy for a clean succession.
const CLEAN_SUCCESSION_LEGITIMACY: FixedPoint = FixedPoint::from_int(300);

/// Cohesion penalty during a succession crisis.
const CRISIS_COHESION_PENALTY: FixedPoint = FixedPoint::from_int(300);

/// Base death chance per turn for characters aged 60+ (per mille per year above 60).
const OLD_AGE_DEATH_RATE_PER_YEAR: i64 = 30; // 3% per year above 60

/// Base death chance per turn for characters aged 40-59 (per mille per year).
const MIDDLE_AGE_DEATH_RATE: i64 = 3; // 0.3% per year

/// Health threshold below which death chance increases.
const LOW_HEALTH_THRESHOLD: FixedPoint = FixedPoint::from_int(300);

/// Run aging and natural death checks for all characters.
///
/// Characters age 1 day per turn. Every 365 turns, their age increments by 1.
/// Death checks run every turn based on age and health.
///
/// Capped at [`MAX_DEATHS_PER_TURN`](crown_ash_types::MAX_DEATHS_PER_TURN)
/// to prevent gas exhaustion on pathological turns.
///
/// Returns list of game events for deaths.
pub fn age_and_death_check(
    world: &mut GameWorld,
    rng: &mut DeterministicRng,
) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let turn = world.meta.turn;
    let char_count = world.characters.len();

    for idx in 0..char_count {
        if !world.characters[idx].alive {
            continue;
        }

        // Age increment: every 365 turns = 1 year.
        if turn > 0 && turn % 365 == 0 {
            world.characters[idx].age = world.characters[idx].age.saturating_add(1);
            // Mark character dirty — age changed.
            world.dirty.dirty_characters.insert(world.characters[idx].id);
        }

        let age = world.characters[idx].age;
        let health = world.characters[idx].health;

        // Death chance calculation.
        let death_chance = if age >= 60 {
            let years_above_60 = (age - 60) as i64;
            // Base rate scaled by years above 60, per turn (divide by 365 for daily).
            let annual_rate = OLD_AGE_DEATH_RATE_PER_YEAR * (years_above_60 + 1);
            // Convert annual rate to per-turn: rate / 365, min 1 per mille.
            (annual_rate / 365).max(1) as u32
        } else if age >= 40 {
            // Small chance of natural death.
            let annual_rate = MIDDLE_AGE_DEATH_RATE;
            (annual_rate / 365).max(0) as u32
        } else {
            0
        };

        // Low health modifier.
        let health_modifier = if health < LOW_HEALTH_THRESHOLD {
            2u32
        } else {
            1u32
        };

        let total_chance = death_chance * health_modifier;

        if total_chance > 0 && rng.chance(total_chance, 1000) {
            world.characters[idx].alive = false;
            world.characters[idx].death_turn = Some(turn);
            world.characters[idx].death_cause = Some("OldAge".to_string());
            // Mark character dirty — died.
            world.dirty.dirty_characters.insert(world.characters[idx].id);
            events.push(GameEvent::CharacterDied {
                character_id: world.characters[idx].id,
                character_name: world.characters[idx].name.clone(),
                cause: DeathCause::OldAge,
                turn,
            });
            if events.len() >= crown_ash_types::MAX_DEATHS_PER_TURN {
                break; // Work cap reached.
            }
        }
    }

    events
}

/// Check if any rulers have died and handle succession.
///
/// Capped at [`MAX_SUCCESSIONS_PER_TURN`](crown_ash_types::MAX_SUCCESSIONS_PER_TURN)
/// to prevent gas exhaustion on pathological turns.
///
/// Returns game events for succession crises and new rulers.
pub fn check_succession(
    world: &mut GameWorld,
    rng: &mut DeterministicRng,
) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let mut successions_processed = 0usize;

    let realm_count = world.realms.len();
    for ridx in 0..realm_count {
        if successions_processed >= crown_ash_types::MAX_SUCCESSIONS_PER_TURN {
            break; // Work cap reached — remaining successions deferred to next turn.
        }

        let ruler_id = world.realms[ridx].ruler;
        let faction_id = world.realms[ridx].faction;

        let ruler_alive = world.character(ruler_id)
            .map(|c| c.alive)
            .unwrap_or(false);

        if ruler_alive {
            continue;
        }

        // Ruler is dead — check for heir.
        let heir_id = world.character(ruler_id).and_then(|c| c.heir);

        let clean_succession = heir_id
            .and_then(|hid| world.character(hid))
            .filter(|h| h.alive && h.legitimacy >= CLEAN_SUCCESSION_LEGITIMACY)
            .map(|h| h.id);

        successions_processed += 1;

        // Mark realm dirty — ruler succession changes the realm.
        world.dirty.dirty_realms.insert(faction_id);

        if let Some(new_ruler_id) = clean_succession {
            // Clean succession.
            promote_to_ruler(world, new_ruler_id, faction_id);
            world.realms[ridx].ruler = new_ruler_id;
        } else {
            // Succession crisis!
            let claimants = find_claimants(world, faction_id);
            let realm_split = claimants.len() >= 3 && rng.chance(1, 3);

            events.push(GameEvent::SuccessionCrisis {
                faction: faction_id,
                dead_ruler: ruler_id,
                claimants: claimants.clone(),
                realm_split,
                turn: world.meta.turn,
            });

            // Apply crisis cohesion penalty.
            world.realms[ridx].cohesion.legitimacy =
                world.realms[ridx].cohesion.legitimacy.saturating_sub(CRISIS_COHESION_PENALTY);
            world.realms[ridx].cohesion.fealty =
                world.realms[ridx].cohesion.fealty.saturating_sub(CRISIS_COHESION_PENALTY);
            world.realms[ridx].cohesion.clamp_all();

            // Pick the best claimant (highest score).
            if let Some(&best) = claimants.first() {
                promote_to_ruler(world, best, faction_id);
                world.realms[ridx].ruler = best;

                // Realm split: if crisis triggers a split and there are enough claimants,
                // partition the realm between the winner and the strongest rebel.
                if realm_split && claimants.len() >= 2 {
                    let split_events = realm_split::process_realm_split(
                        world, faction_id, best, &claimants, rng,
                    );
                    events.extend(split_events);
                }
            }
        }
    }

    events
}

/// Score and rank claimants for succession.
///
/// Eligible claimants are living characters of the same faction with a role
/// that indicates seniority (Ruler, Heir, Duke, Marshal).
///
/// Score: `legitimacy + prestige + martial*2 + diplomacy*2`.
fn find_claimants(world: &GameWorld, faction_id: u8) -> Vec<CharacterId> {
    let mut candidates: Vec<(CharacterId, i64)> = world.characters.iter()
        .filter(|c| {
            c.alive
                && c.faction == faction_id
                && c.is_adult()
                && matches!(c.role,
                    CharacterRole::Heir
                    | CharacterRole::Duke
                    | CharacterRole::Marshal
                    | CharacterRole::Courtier
                )
        })
        .map(|c| {
            let stats = c.effective_stats();
            let score = c.legitimacy.raw()
                + c.prestige.raw()
                + stats.martial.raw() * 2
                + stats.diplomacy.raw() * 2;
            (c.id, score)
        })
        .collect();

    // Sort by score descending.
    candidates.sort_by(|a, b| b.1.cmp(&a.1));

    candidates.into_iter().map(|(id, _)| id).collect()
}

/// Promote a character to ruler of their faction.
fn promote_to_ruler(world: &mut GameWorld, character_id: CharacterId, faction_id: u8) {
    // Demote any existing ruler.
    let char_count = world.characters.len();
    for idx in 0..char_count {
        if world.characters[idx].faction == faction_id
            && world.characters[idx].role == CharacterRole::Ruler
            && world.characters[idx].id != character_id
        {
            world.characters[idx].role = CharacterRole::Courtier;
            world.dirty.dirty_characters.insert(world.characters[idx].id);
        }
    }
    // Promote the new ruler.
    if let Some(c) = world.character_mut_dirty(character_id) {
        c.role = CharacterRole::Ruler;
        // Boost legitimacy on succession.
        c.legitimacy += FixedPoint::from_int(100);
    }
}

/// Designate a character as heir to a ruler.
///
/// Returns an error if the character doesn't belong to the same faction.
pub fn designate_heir(
    world: &mut GameWorld,
    ruler_id: CharacterId,
    heir_id: CharacterId,
) -> Result<(), String> {
    let ruler_faction = world.character(ruler_id)
        .map(|c| c.faction)
        .ok_or_else(|| "Ruler not found".to_string())?;
    let heir_faction = world.character(heir_id)
        .map(|c| c.faction)
        .ok_or_else(|| "Heir not found".to_string())?;

    if ruler_faction != heir_faction {
        return Err("Heir must be from the same faction".to_string());
    }

    let heir_alive = world.character(heir_id).map(|c| c.alive).unwrap_or(false);
    if !heir_alive {
        return Err("Heir must be alive".to_string());
    }

    // Set heir on ruler.
    if let Some(ruler) = world.character_mut_dirty(ruler_id) {
        ruler.heir = Some(heir_id);
    }

    // Set heir role.
    if let Some(heir) = world.character_mut_dirty(heir_id) {
        heir.role = CharacterRole::Heir;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_gen::init_world;
    use crown_ash_types::WorldConfig;

    #[test]
    fn clean_succession_when_heir_exists() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0xCC; 32]);

        let ruler_id = world.realms[0].ruler;
        let heir_id = world.character(ruler_id)
            .and_then(|c| c.heir)
            .expect("Ruler should have an heir");

        // Ensure heir has enough legitimacy.
        if let Some(heir) = world.character_mut(heir_id) {
            heir.legitimacy = FixedPoint::from_int(500);
        }

        // Kill the ruler.
        if let Some(ruler) = world.character_mut(ruler_id) {
            ruler.alive = false;
        }

        let mut rng = DeterministicRng::new([0xDD; 32], "succession_test");
        let events = check_succession(&mut world, &mut rng);

        // Should be clean succession (no crisis event).
        let crises: Vec<_> = events.iter()
            .filter(|e| matches!(e, GameEvent::SuccessionCrisis { .. }))
            .collect();
        assert!(crises.is_empty(), "Should be clean succession with legitimate heir");

        // New ruler should be the heir.
        assert_eq!(world.realms[0].ruler, heir_id);
    }

    #[test]
    fn crisis_when_no_heir() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0xEE; 32]);

        let ruler_id = world.realms[0].ruler;

        // Remove heir.
        if let Some(ruler) = world.character_mut(ruler_id) {
            ruler.heir = None;
        }

        // Kill the ruler.
        if let Some(ruler) = world.character_mut(ruler_id) {
            ruler.alive = false;
        }

        let mut rng = DeterministicRng::new([0xFF; 32], "succession_test");
        let events = check_succession(&mut world, &mut rng);

        // Should have a succession crisis.
        let crises: Vec<_> = events.iter()
            .filter(|e| matches!(e, GameEvent::SuccessionCrisis { .. }))
            .collect();
        assert!(!crises.is_empty(), "Should trigger succession crisis without heir");
    }

    #[test]
    fn designate_heir_works() {
        let config = WorldConfig::default();
        let mut world = init_world(&config, [0xAA; 32]);

        let ruler_id = world.realms[0].ruler;
        // Find any courtier/councillor of faction 0 that isn't the current heir.
        let candidate = world.characters.iter()
            .find(|c| c.faction == 0 && c.alive && c.id != ruler_id)
            .map(|c| c.id)
            .unwrap();

        let result = designate_heir(&mut world, ruler_id, candidate);
        assert!(result.is_ok());

        assert_eq!(
            world.character(ruler_id).unwrap().heir,
            Some(candidate)
        );
        assert_eq!(
            world.character(candidate).unwrap().role,
            CharacterRole::Heir
        );
    }
}
