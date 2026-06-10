//! NPC Personality System — trait-driven dialog tone and LLM prompting.
//!
//! Each character has a personality profile derived from their traits.
//! This module maps trait combinations to speech patterns, tone descriptors,
//! and system prompts used by the LLM (Tier 2/3) to generate in-character
//! dialog and speeches.
//!
//! # Design
//!
//! Personality is expressed through three layers:
//!
//! 1. **Archetype** — A dominant personality pattern derived from the
//!    character's strongest trait axes (e.g., "Tyrant", "Saint", "Schemer").
//!
//! 2. **Tone Descriptors** — Adjectives that color the character's speech
//!    (e.g., "menacing", "pious", "calculating"). Used in LLM system prompts.
//!
//! 3. **Speech Patterns** — Short template phrases the character uses in
//!    Tier 1 (template) dialog. No LLM needed — just trait-flavored text.

use crown_ash_types::character::Trait;
use serde::{Deserialize, Serialize};

// ─── Archetype ──────────────────────────────────────────────────────────────

/// A character archetype derived from their dominant traits.
/// Used to select dialog templates and LLM persona prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Archetype {
    /// Brave + Cruel / Wrathful — rules through fear and force.
    Tyrant,
    /// Pious + Kind / Just — devoted ruler, morally upright.
    Saint,
    /// Schemer + Deceitful / Ambitious — cunning manipulator.
    Schemer,
    /// Brave + Just / Strategist — noble warrior leader.
    Commander,
    /// Gregarious + Kind / Honest — beloved by the people.
    Diplomat,
    /// Scholar + Patient / Diligent — wise advisor type.
    Scholar,
    /// Ambitious + Wrathful / Brave — bold and reckless.
    Conqueror,
    /// Content + Slothful / Gluttonous — passive, pleasure-seeking.
    Hedonist,
    /// Paranoid + Cruel / Cynical — suspicious of everyone.
    Inquisitor,
    /// No dominant pattern — balanced or contradictory traits.
    Balanced,
}

impl Archetype {
    /// Human-readable name for display.
    pub fn label(&self) -> &'static str {
        match self {
            Archetype::Tyrant => "Tyrant",
            Archetype::Saint => "Saint",
            Archetype::Schemer => "Schemer",
            Archetype::Commander => "Commander",
            Archetype::Diplomat => "Diplomat",
            Archetype::Scholar => "Scholar",
            Archetype::Conqueror => "Conqueror",
            Archetype::Hedonist => "Hedonist",
            Archetype::Inquisitor => "Inquisitor",
            Archetype::Balanced => "Balanced",
        }
    }
}

// ─── Personality Profile ────────────────────────────────────────────────────

/// A character's full personality profile — derived from traits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityProfile {
    /// The dominant archetype.
    pub archetype: Archetype,
    /// Tone descriptors for LLM prompting (e.g., ["menacing", "authoritative"]).
    pub tone: Vec<String>,
    /// Short speech flavor phrases for Tier 1 template dialog.
    pub speech_flavors: Vec<String>,
}

// ─── Derive Archetype from Traits ───────────────────────────────────────────

/// Derive the dominant archetype from a character's trait list.
pub fn derive_archetype(traits: &[Trait]) -> Archetype {
    let has = |t: Trait| traits.contains(&t);

    // Check combinations in priority order (most specific first)
    if has(Trait::Schemer) || (has(Trait::Deceitful) && has(Trait::Ambitious)) {
        return Archetype::Schemer;
    }
    if has(Trait::Pious) && (has(Trait::Kind) || has(Trait::Just)) {
        return Archetype::Saint;
    }
    if (has(Trait::Cruel) || has(Trait::Wrathful)) && has(Trait::Brave) {
        return Archetype::Tyrant;
    }
    if has(Trait::Paranoid) && (has(Trait::Cruel) || has(Trait::Cynical)) {
        return Archetype::Inquisitor;
    }
    if has(Trait::Brave) && (has(Trait::Just) || has(Trait::Strategist)) {
        return Archetype::Commander;
    }
    if has(Trait::Ambitious) && (has(Trait::Wrathful) || has(Trait::Brave)) {
        return Archetype::Conqueror;
    }
    if has(Trait::Gregarious) && (has(Trait::Kind) || has(Trait::Honest)) {
        return Archetype::Diplomat;
    }
    if has(Trait::Scholar) || (has(Trait::Patient) && has(Trait::Diligent)) {
        return Archetype::Scholar;
    }
    if has(Trait::Content) && (has(Trait::Slothful) || has(Trait::Gluttonous)) {
        return Archetype::Hedonist;
    }

    Archetype::Balanced
}

// ─── Build Full Profile ─────────────────────────────────────────────────────

/// Build a full personality profile from a character's traits.
pub fn build_profile(traits: &[Trait]) -> PersonalityProfile {
    let archetype = derive_archetype(traits);

    let tone: Vec<String> = match archetype {
        Archetype::Tyrant => vec!["menacing", "authoritative", "cold"],
        Archetype::Saint => vec!["gentle", "pious", "compassionate"],
        Archetype::Schemer => vec!["calculating", "smooth", "insinuating"],
        Archetype::Commander => vec!["decisive", "honorable", "firm"],
        Archetype::Diplomat => vec!["warm", "persuasive", "charming"],
        Archetype::Scholar => vec!["measured", "thoughtful", "precise"],
        Archetype::Conqueror => vec!["bold", "impatient", "commanding"],
        Archetype::Hedonist => vec!["languid", "indifferent", "amused"],
        Archetype::Inquisitor => vec!["suspicious", "sharp", "accusatory"],
        Archetype::Balanced => vec!["measured", "calm", "pragmatic"],
    }.into_iter().map(String::from).collect();

    let speech_flavors: Vec<String> = match archetype {
        Archetype::Tyrant => vec![
            "You dare question me?",
            "Obedience or death — choose wisely.",
            "The weak serve. The strong rule.",
            "I did not ask for your opinion.",
        ],
        Archetype::Saint => vec![
            "May the gods guide our path.",
            "Mercy is the mark of a true ruler.",
            "We must care for the least among us.",
            "I pray this brings peace to our people.",
        ],
        Archetype::Schemer => vec![
            "Every player has a price...",
            "Patience. The game is long.",
            "Trust is a weapon best used sparingly.",
            "Let them think they've won. For now.",
        ],
        Archetype::Commander => vec![
            "Stand firm! We hold this ground!",
            "Honor demands we fight.",
            "I lead from the front, not the rear.",
            "Our cause is just. Victory will follow.",
        ],
        Archetype::Diplomat => vec![
            "Surely we can find common ground.",
            "War benefits no one — let us talk.",
            "A friend today is an ally tomorrow.",
            "I bring greetings and an open hand.",
        ],
        Archetype::Scholar => vec![
            "The texts are clear on this matter.",
            "Let us consider the evidence.",
            "Knowledge is the truest power.",
            "History teaches us caution here.",
        ],
        Archetype::Conqueror => vec![
            "More! I want it all!",
            "They will kneel, or they will fall.",
            "Hesitation is defeat.",
            "The world is mine for the taking.",
        ],
        Archetype::Hedonist => vec![
            "Pour another cup — governance can wait.",
            "Why trouble ourselves with such matters?",
            "Life is short. Enjoy what you can.",
            "Let my steward handle the details.",
        ],
        Archetype::Inquisitor => vec![
            "I trust no one. Least of all you.",
            "Who sent you? Speak truthfully.",
            "Every smile hides a dagger.",
            "I see plots everywhere — and I am usually right.",
        ],
        Archetype::Balanced => vec![
            "Let us weigh our options carefully.",
            "There is wisdom in moderation.",
            "I will consider your counsel.",
            "We proceed with caution.",
        ],
    }.into_iter().map(String::from).collect();

    PersonalityProfile {
        archetype,
        tone,
        speech_flavors,
    }
}

// ─── LLM System Prompt ──────────────────────────────────────────────────────

/// Build an LLM system prompt for in-character dialog generation.
///
/// This is the "persona" prompt that precedes the user/event prompt.
/// It instructs the LLM to speak as this specific character with their
/// personality traits.
pub fn build_persona_prompt(
    character_name: &str,
    role: &str,
    faction_name: &str,
    profile: &PersonalityProfile,
) -> String {
    let tone_str = profile.tone.join(", ");

    format!(
        "You are {name}, {role} of {faction}. \
         Your personality archetype is {archetype}. \
         Your tone is {tone}. \
         Speak in the first person as {name}. \
         Use medieval-flavored English — formal but readable. \
         Keep responses to 1-3 sentences unless instructed otherwise. \
         Never break character. Never use modern language or references.",
        name = character_name,
        role = role,
        faction = faction_name,
        archetype = profile.archetype.label(),
        tone = tone_str,
    )
}

/// Generate a Tier 1 (template) dialog line for a character based on context.
///
/// Uses the character's speech_flavors to pick a contextually appropriate
/// line. The `seed` parameter (typically turn number) selects deterministically.
pub fn template_dialog(profile: &PersonalityProfile, seed: u32) -> &str {
    let flavors = &profile.speech_flavors;
    if flavors.is_empty() {
        return "...";
    }
    let idx = (seed as usize * 7919) % flavors.len();
    &flavors[idx]
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemer_archetype() {
        let traits = vec![Trait::Schemer, Trait::Deceitful, Trait::Ambitious];
        assert_eq!(derive_archetype(&traits), Archetype::Schemer);
    }

    #[test]
    fn saint_archetype() {
        let traits = vec![Trait::Pious, Trait::Kind];
        assert_eq!(derive_archetype(&traits), Archetype::Saint);
    }

    #[test]
    fn commander_archetype() {
        let traits = vec![Trait::Brave, Trait::Just, Trait::Strategist];
        assert_eq!(derive_archetype(&traits), Archetype::Commander);
    }

    #[test]
    fn tyrant_archetype() {
        let traits = vec![Trait::Brave, Trait::Cruel, Trait::Wrathful];
        assert_eq!(derive_archetype(&traits), Archetype::Tyrant);
    }

    #[test]
    fn empty_traits_give_balanced() {
        let traits: Vec<Trait> = vec![];
        assert_eq!(derive_archetype(&traits), Archetype::Balanced);
    }

    #[test]
    fn profile_has_tone_and_flavors() {
        let traits = vec![Trait::Gregarious, Trait::Kind];
        let profile = build_profile(&traits);
        assert_eq!(profile.archetype, Archetype::Diplomat);
        assert!(!profile.tone.is_empty());
        assert!(!profile.speech_flavors.is_empty());
    }

    #[test]
    fn persona_prompt_contains_character_info() {
        let traits = vec![Trait::Pious, Trait::Just];
        let profile = build_profile(&traits);
        let prompt = build_persona_prompt("King Aldric", "Ruler", "Ashen Crown", &profile);
        assert!(prompt.contains("King Aldric"));
        assert!(prompt.contains("Ashen Crown"));
        assert!(prompt.contains("Saint"));
        assert!(prompt.contains("medieval"));
    }

    #[test]
    fn template_dialog_deterministic() {
        let traits = vec![Trait::Schemer, Trait::Deceitful];
        let profile = build_profile(&traits);

        let line1 = template_dialog(&profile, 42);
        let line2 = template_dialog(&profile, 42);
        assert_eq!(line1, line2); // Same seed → same line

        // Different seed may give different line
        let line3 = template_dialog(&profile, 43);
        // At least one of these should be a real phrase
        assert!(!line1.is_empty());
        assert!(!line3.is_empty());
    }

    #[test]
    fn all_archetypes_have_speech() {
        let test_cases: Vec<Vec<Trait>> = vec![
            vec![Trait::Brave, Trait::Cruel],       // Tyrant
            vec![Trait::Pious, Trait::Kind],         // Saint
            vec![Trait::Schemer],                    // Schemer
            vec![Trait::Brave, Trait::Just],         // Commander
            vec![Trait::Gregarious, Trait::Kind],    // Diplomat
            vec![Trait::Scholar],                    // Scholar
            vec![Trait::Ambitious, Trait::Wrathful], // Conqueror
            vec![Trait::Content, Trait::Slothful],   // Hedonist
            vec![Trait::Paranoid, Trait::Cruel],     // Inquisitor
            vec![],                                  // Balanced
        ];

        for traits in test_cases {
            let profile = build_profile(&traits);
            assert!(!profile.speech_flavors.is_empty(),
                "Archetype {:?} has no speech flavors", profile.archetype);
            assert!(!profile.tone.is_empty(),
                "Archetype {:?} has no tone", profile.archetype);
        }
    }
}
