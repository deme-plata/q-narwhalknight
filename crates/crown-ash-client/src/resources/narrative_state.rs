//! Narrative state resource — holds chronicles, history cache, and LLM output.
//!
//! Updated by the narrative update system each frame. Provides rich text
//! for the detail panel (chronicle tab, province/faction history) and the
//! event feed (narrative prose instead of raw format_event).

use bevy::prelude::*;
use crown_ash_narrative::chronicle::CharacterChronicle;
use crown_ash_narrative::cascade::CascadeResult;
use std::collections::HashMap;

/// Cached narrative state for the client UI.
#[derive(Resource)]
pub struct NarrativeState {
    /// Character chronicles stored as a flat Vec (matches chronicle::update_chronicles API).
    pub chronicles_vec: Vec<CharacterChronicle>,

    /// Recent cascade results for the event feed (last N events with prose).
    /// Each entry has the narrative prose + summary from the cascade engine.
    pub event_narratives: Vec<EventNarrative>,

    /// Province ID → cached history text (regenerated when events change).
    pub province_histories: HashMap<u16, String>,

    /// Faction ID → cached history text.
    pub faction_histories: HashMap<u8, String>,

    /// LLM-generated text that arrived via SSE (keyed by generation ID).
    pub llm_results: Vec<LlmNarrativeResult>,

    /// Whether narrative state needs rebuilding (set when new events arrive).
    pub dirty: bool,
}

/// A single event with its narrative prose for the event feed.
pub struct EventNarrative {
    /// Turn number.
    pub turn: u32,
    /// Template-generated prose (Tier 1).
    pub prose: String,
    /// One-line summary.
    pub summary: String,
    /// Importance level as string for color coding.
    pub importance: NarrativeImportance,
}

/// Importance level for color-coding in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrativeImportance {
    Minor,
    Notable,
    Epic,
}

/// LLM-generated text received via SSE.
pub struct LlmNarrativeResult {
    /// Which tier produced this (2 = dialog, 3 = deep narrative).
    pub tier: u8,
    /// The generated text.
    pub text: String,
    /// Speaker name (for dialog).
    pub speaker: Option<String>,
    /// Timestamp when received.
    pub received_turn: u32,
}

// ---------------------------------------------------------------------------
// Dialog speech bubbles — floating overlay for Tier 2/3 LLM-generated text
// ---------------------------------------------------------------------------

/// A single active dialog bubble floating on the game screen.
pub struct DialogBubble {
    /// Speaker name (character, faction, narrator).
    pub speaker: String,
    /// The dialog / narrative text.
    pub text: String,
    /// Which tier generated this (2 = dialog, 3 = epic).
    pub tier: u8,
    /// Seconds remaining before auto-dismiss.
    pub timer: f32,
    /// Turn number when this dialog was received.
    pub turn: u32,
}

/// Resource tracking all active speech bubbles on screen.
#[derive(Resource)]
pub struct DialogState {
    /// Active bubbles, newest last.
    pub bubbles: Vec<DialogBubble>,
    /// Maximum simultaneous bubbles on screen.
    pub max_visible: usize,
}

impl Default for DialogState {
    fn default() -> Self {
        Self {
            bubbles: Vec::new(),
            max_visible: 4,
        }
    }
}

impl DialogState {
    /// Add a new speech bubble. Evicts the oldest if at capacity.
    pub fn push_dialog(&mut self, speaker: String, text: String, tier: u8, turn: u32) {
        let duration = match tier {
            3 => 12.0, // Epic narratives stay longer
            _ => 8.0,  // Dialog stays 8 seconds
        };
        self.bubbles.push(DialogBubble {
            speaker,
            text,
            tier,
            timer: duration,
            turn,
        });
        // Keep only max_visible bubbles — evict oldest.
        while self.bubbles.len() > self.max_visible {
            self.bubbles.remove(0);
        }
    }

    /// Tick all timers, remove expired bubbles. Returns true if any were removed.
    pub fn tick(&mut self, dt: f32) -> bool {
        let before = self.bubbles.len();
        for b in &mut self.bubbles {
            b.timer -= dt;
        }
        self.bubbles.retain(|b| b.timer > 0.0);
        self.bubbles.len() != before
    }
}

impl Default for NarrativeState {
    fn default() -> Self {
        Self {
            chronicles_vec: Vec::new(),
            event_narratives: Vec::new(),
            province_histories: HashMap::new(),
            faction_histories: HashMap::new(),
            llm_results: Vec::new(),
            dirty: true,
        }
    }
}

impl NarrativeState {
    /// Get the chronicle for a character, if it exists.
    pub fn chronicle_text(&self, character_id: u32) -> Option<String> {
        self.chronicles_vec.iter()
            .find(|c| c.character_id == character_id)
            .map(|c| c.render_full())
    }

    /// Get cached province history text.
    pub fn province_history(&self, province_id: u16) -> Option<&str> {
        self.province_histories.get(&province_id).map(|s| s.as_str())
    }

    /// Get cached faction history text.
    pub fn faction_history(&self, faction_id: u8) -> Option<&str> {
        self.faction_histories.get(&faction_id).map(|s| s.as_str())
    }
}

impl From<&CascadeResult> for EventNarrative {
    fn from(result: &CascadeResult) -> Self {
        let turn = match &result.event {
            crown_ash_types::GameEvent::Battle(r) => r.turn,
            crown_ash_types::GameEvent::Harvest { turn, .. } => *turn,
            crown_ash_types::GameEvent::Famine { turn, .. } => *turn,
            crown_ash_types::GameEvent::PlagueOutbreak { turn, .. } => *turn,
            crown_ash_types::GameEvent::WarDeclared { turn, .. } => *turn,
            crown_ash_types::GameEvent::TreatySigned { turn, .. } => *turn,
            crown_ash_types::GameEvent::CharacterBorn { turn, .. } => *turn,
            crown_ash_types::GameEvent::CharacterDied { turn, .. } => *turn,
            crown_ash_types::GameEvent::ProvinceConquered { turn, .. } => *turn,
            crown_ash_types::GameEvent::SuccessionCrisis { turn, .. } => *turn,
            crown_ash_types::GameEvent::FactionEliminated { turn, .. } => *turn,
            crown_ash_types::GameEvent::Rebellion { turn, .. } => *turn,
            crown_ash_types::GameEvent::RealmSplit { turn, .. } => *turn,
            _ => 0,
        };

        let importance = match result.importance {
            crown_ash_narrative::Importance::Minor => NarrativeImportance::Minor,
            crown_ash_narrative::Importance::Notable => NarrativeImportance::Notable,
            crown_ash_narrative::Importance::Epic => NarrativeImportance::Epic,
        };

        EventNarrative {
            turn,
            prose: result.prose.clone(),
            summary: result.summary.clone(),
            importance,
        }
    }
}
