//! Crown & Ash — Narrative text generation engine.
//!
//! Two-tier architecture for rich game storytelling:
//!
//! - **Tier 1 (Templates)**: Handwritten narrative templates with variable substitution.
//!   Every event gets rich prose instantly — no LLM, no latency, deterministic.
//!   Multiple variants per event type selected by context (culture, terrain, severity).
//!
//! - **Tier 2 (LLM)**: Deep narrative for important moments via `q-ai-inference`.
//!   Feature-gated behind `llm`. Generates character speeches, battle epics,
//!   and chronicle summaries using Mistral-7B or Nemotron models.
//!
//! # Usage
//!
//! ```ignore
//! use crown_ash_narrative::{NarrativeEngine, WorldContext};
//!
//! let engine = NarrativeEngine::new();
//! let context = WorldContext::from_world(&game_world);
//! let prose = engine.narrate(&event, &context);
//! ```

pub mod templates;
pub mod chronicle;
pub mod personality;
pub mod history;
pub mod llm;
pub mod cascade;

use crown_ash_types::GameEvent;
use serde::{Deserialize, Serialize};

// ─── World Context ───────────────────────────────────────────────────────────

/// Contextual information passed to the narrative engine for richer text.
///
/// Built from the current `GameWorld` snapshot. Provides names, cultures,
/// and relationships that templates use to fill in variables.
#[derive(Debug, Clone, Default)]
pub struct WorldContext {
    /// Province ID → name mapping.
    pub province_names: Vec<(u16, String)>,
    /// Faction ID → name mapping.
    pub faction_names: Vec<(u8, String)>,
    /// Character ID → name mapping.
    pub character_names: Vec<(u32, String)>,
    /// Faction ID → culture string.
    pub faction_cultures: Vec<(u8, String)>,
    /// Army ID → owner faction ID mapping (for battle narrative).
    pub army_factions: Vec<(u32, u8)>,
    /// Current game turn.
    pub current_turn: u32,
}

impl WorldContext {
    /// Look up a province name by ID.
    pub fn province_name(&self, id: u16) -> &str {
        self.province_names.iter()
            .find(|(pid, _)| *pid == id)
            .map(|(_, name)| name.as_str())
            .unwrap_or("an unknown province")
    }

    /// Look up a faction name by ID.
    pub fn faction_name(&self, id: u8) -> &str {
        self.faction_names.iter()
            .find(|(fid, _)| *fid == id)
            .map(|(_, name)| name.as_str())
            .unwrap_or("an unknown faction")
    }

    /// Look up a character name by ID.
    pub fn character_name(&self, id: u32) -> &str {
        self.character_names.iter()
            .find(|(cid, _)| *cid == id)
            .map(|(_, name)| name.as_str())
            .unwrap_or("a forgotten soul")
    }

    /// Look up the faction that owns an army.
    pub fn army_faction(&self, army_id: u32) -> Option<u8> {
        self.army_factions.iter()
            .find(|(aid, _)| *aid == army_id)
            .map(|(_, fid)| *fid)
    }

    /// Look up the faction name for an army (convenience).
    pub fn army_faction_name(&self, army_id: u32) -> &str {
        self.army_faction(army_id)
            .map(|fid| self.faction_name(fid))
            .unwrap_or("an unknown host")
    }
}

// ─── Narrative Output ────────────────────────────────────────────────────────

/// The importance level of a narrative event — determines which tiers fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Importance {
    /// Routine event — Tier 1 template only.
    Minor,
    /// Notable event — Tier 1 + Tier 2 short dialog.
    Notable,
    /// Epic event — all tiers, including deep LLM narrative.
    Epic,
}

/// Output from the narrative engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Narrative {
    /// The importance classification of this event.
    pub importance: Importance,
    /// Tier 1: Template-generated prose (always present).
    pub prose: String,
    /// Short one-line summary for the event feed.
    pub summary: String,
    /// Optional Tier 2 prompt for LLM generation (only for Notable/Epic).
    /// The caller can send this to q-ai-inference for deep narrative.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_prompt: Option<String>,
}

// ─── Narrative Engine ────────────────────────────────────────────────────────

/// The core narrative engine. Stateless — all context is passed per-call.
pub struct NarrativeEngine {
    _private: (),
}

impl NarrativeEngine {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Generate narrative text for a game event.
    ///
    /// Returns a `Narrative` with template prose (always) and an optional
    /// LLM prompt (for Notable/Epic events).
    pub fn narrate(&self, event: &GameEvent, ctx: &WorldContext) -> Narrative {
        let importance = classify_importance(event);
        let prose = templates::render(event, ctx);
        let summary = templates::render_summary(event, ctx);

        let llm_prompt = match importance {
            Importance::Minor => None,
            Importance::Notable | Importance::Epic => {
                Some(templates::build_llm_prompt(event, ctx, importance))
            }
        };

        Narrative {
            importance,
            prose,
            summary,
            llm_prompt,
        }
    }

    /// Generate narratives for all events in a turn summary.
    pub fn narrate_turn(&self, events: &[GameEvent], ctx: &WorldContext) -> Vec<Narrative> {
        events.iter().map(|e| self.narrate(e, ctx)).collect()
    }
}

impl Default for NarrativeEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Importance Classification ───────────────────────────────────────────────

fn classify_importance(event: &GameEvent) -> Importance {
    match event {
        // Epic — realm-shaking events
        GameEvent::FactionEliminated { .. } => Importance::Epic,
        GameEvent::RealmSplit { .. } => Importance::Epic,
        GameEvent::SuccessionCrisis { realm_split: true, .. } => Importance::Epic,

        // Notable — significant but not earth-shattering
        GameEvent::Battle(result) => {
            if result.attacker_casualties + result.defender_casualties > 500 {
                Importance::Epic
            } else {
                Importance::Notable
            }
        }
        GameEvent::ProvinceConquered { .. } => Importance::Notable,
        GameEvent::WarDeclared { .. } => Importance::Notable,
        GameEvent::TreatySigned { .. } => Importance::Notable,
        GameEvent::SuccessionCrisis { .. } => Importance::Notable,
        GameEvent::PlotSucceeded { .. } => Importance::Notable,
        GameEvent::PlotDiscovered { .. } => Importance::Notable,
        GameEvent::Rebellion { rebels, .. } if *rebels > 200 => Importance::Notable,
        GameEvent::PlagueOutbreak { population_lost, .. } if *population_lost > 500 => Importance::Notable,
        GameEvent::SiegeCompleted { .. } => Importance::Notable,
        GameEvent::SiegeStarted { .. } => Importance::Notable,
        GameEvent::MarriageAlliance { .. } => Importance::Notable,

        // Minor — routine events
        _ => Importance::Minor,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context() -> WorldContext {
        WorldContext {
            province_names: vec![
                (0, "Frosthold".into()),
                (7, "Ashenmere".into()),
                (14, "Saltmere".into()),
            ],
            faction_names: vec![
                (0, "Ashen Crown".into()),
                (1, "Vale Princes".into()),
                (3, "Salt League".into()),
            ],
            character_names: vec![
                (1, "King Aldric".into()),
                (2, "Queen Isolde".into()),
            ],
            faction_cultures: vec![],
            army_factions: vec![
                (100, 0), // army 100 → Ashen Crown
                (200, 1), // army 200 → Vale Princes
            ],
            current_turn: 50,
        }
    }

    #[test]
    fn narrate_plague_returns_prose() {
        let engine = NarrativeEngine::new();
        let ctx = test_context();
        let event = GameEvent::PlagueOutbreak {
            province: 7,
            severity: 500,
            population_lost: 234,
            turn: 50,
        };

        let narrative = engine.narrate(&event, &ctx);
        assert!(!narrative.prose.is_empty());
        assert!(narrative.prose.contains("Ashenmere"));
        assert_eq!(narrative.importance, Importance::Minor);
    }

    #[test]
    fn epic_battle_triggers_llm_prompt() {
        let engine = NarrativeEngine::new();
        let ctx = test_context();
        let event = GameEvent::Battle(crown_ash_types::army::BattleResult {
            attacker_army: 100,
            defender_army: Some(200),
            province: 7,
            attacker_casualties: 400,
            defender_casualties: 350,
            attacker_won: true,
            random_factor: crown_ash_types::fixed_point::FixedPoint::from_int(1000),
            turn: 50,
        });

        let narrative = engine.narrate(&event, &ctx);
        assert_eq!(narrative.importance, Importance::Epic);
        assert!(narrative.llm_prompt.is_some());
    }

    #[test]
    fn faction_eliminated_is_epic() {
        let engine = NarrativeEngine::new();
        let ctx = test_context();
        let event = GameEvent::FactionEliminated { faction: 1, turn: 50 };

        let narrative = engine.narrate(&event, &ctx);
        assert_eq!(narrative.importance, Importance::Epic);
        assert!(narrative.prose.contains("Vale Princes"));
    }

    #[test]
    fn minor_harvest_no_llm() {
        let engine = NarrativeEngine::new();
        let ctx = test_context();
        let event = GameEvent::Harvest {
            province: 7,
            prosperity_gain: 20000,
            turn: 50,
        };

        let narrative = engine.narrate(&event, &ctx);
        assert_eq!(narrative.importance, Importance::Minor);
        assert!(narrative.llm_prompt.is_none());
    }
}
