//! Narrative state resource — holds chronicles, history cache, and LLM output.
//!
//! Updated by the narrative update system each frame. Provides rich text
//! for the detail panel (chronicle tab, province/faction history) and the
//! event feed (narrative prose instead of raw format_event).

use bevy::prelude::*;
use crown_ash_narrative::chronicle::CharacterChronicle;
use crown_ash_narrative::cascade::CascadeResult;
use std::collections::{HashMap, VecDeque};

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

    /// Per-turn summary sentences (turn number → prose).
    pub turn_summaries: Vec<(u32, String)>,

    /// War summaries keyed by (faction_a, faction_b) — cached prose for active wars.
    pub war_summaries: HashMap<(u8, u8), String>,

    /// Realm prosperity narrative per faction.
    pub realm_prosperity: HashMap<u8, String>,

    /// Intrigue narrative entries (turn, prose).
    pub intrigue_narratives: Vec<(u32, String)>,

    /// Era overview text (refreshed every 10 turns).
    pub era_summary_text: String,

    /// Province religion narrative cache.
    pub province_religion: HashMap<u16, String>,

    /// Diplomacy narrative cache keyed by (faction_a, faction_b).
    pub diplomacy_narratives: HashMap<(u8, u8), String>,

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
    /// Active bubbles, newest last. VecDeque for O(1) front eviction.
    pub bubbles: VecDeque<DialogBubble>,
    /// Maximum simultaneous bubbles on screen.
    pub max_visible: usize,
}

impl Default for DialogState {
    fn default() -> Self {
        Self {
            bubbles: VecDeque::new(),
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
        self.bubbles.push_back(DialogBubble {
            speaker,
            text,
            tier,
            timer: duration,
            turn,
        });
        // Keep only max_visible bubbles — evict oldest (O(1) with VecDeque).
        while self.bubbles.len() > self.max_visible {
            self.bubbles.pop_front();
        }
    }

    /// Tick all timers, remove expired bubbles. Returns true if any were removed.
    pub fn tick(&mut self, dt: f32) -> bool {
        let before = self.bubbles.len();
        for b in self.bubbles.iter_mut() {
            b.timer -= dt;
        }
        self.bubbles.retain(|b| b.timer > 0.0);
        self.bubbles.len() != before
    }
}

// ---------------------------------------------------------------------------
// Notification toasts — brief auto-dismiss popups for Epic/Notable events
// ---------------------------------------------------------------------------

/// A single notification toast on the left side of the screen.
pub struct NotificationToast {
    /// Short summary text (one line).
    pub text: String,
    /// Importance level for color coding.
    pub importance: NarrativeImportance,
    /// Seconds remaining before auto-dismiss.
    pub timer: f32,
    /// Turn when this event occurred.
    pub turn: u32,
}

/// Resource tracking active notification toasts.
#[derive(Resource)]
pub struct ToastState {
    /// Active toasts, newest last. VecDeque for O(1) front eviction.
    pub toasts: VecDeque<NotificationToast>,
    /// Maximum simultaneous toasts.
    pub max_visible: usize,
}

impl Default for ToastState {
    fn default() -> Self {
        Self {
            toasts: VecDeque::new(),
            max_visible: 5,
        }
    }
}

impl ToastState {
    /// Push a new notification toast. Evicts oldest if at capacity.
    pub fn push(&mut self, text: String, importance: NarrativeImportance, turn: u32) {
        let duration = match importance {
            NarrativeImportance::Epic => 6.0,
            NarrativeImportance::Notable => 4.0,
            NarrativeImportance::Minor => 3.0,
        };
        self.toasts.push_back(NotificationToast {
            text,
            importance,
            timer: duration,
            turn,
        });
        while self.toasts.len() > self.max_visible {
            self.toasts.pop_front();
        }
    }

    /// Tick all timers, remove expired. Returns true if any were removed.
    pub fn tick(&mut self, dt: f32) -> bool {
        let before = self.toasts.len();
        for t in self.toasts.iter_mut() {
            t.timer -= dt;
        }
        self.toasts.retain(|t| t.timer > 0.0);
        self.toasts.len() != before
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
            turn_summaries: Vec::new(),
            war_summaries: HashMap::new(),
            realm_prosperity: HashMap::new(),
            intrigue_narratives: Vec::new(),
            era_summary_text: String::new(),
            province_religion: HashMap::new(),
            diplomacy_narratives: HashMap::new(),
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

    /// Get turn summary for a given turn number.
    pub fn turn_summary(&self, turn: u32) -> Option<&str> {
        self.turn_summaries.iter()
            .find(|(t, _)| *t == turn)
            .map(|(_, text)| text.as_str())
    }

    /// Get cached war summary between two factions (order-independent).
    pub fn war_summary(&self, a: u8, b: u8) -> Option<&str> {
        let key = if a <= b { (a, b) } else { (b, a) };
        self.war_summaries.get(&key).map(|s| s.as_str())
    }

    /// Get realm prosperity narrative for a faction.
    pub fn realm_prosperity_text(&self, faction_id: u8) -> Option<&str> {
        self.realm_prosperity.get(&faction_id).map(|s| s.as_str())
    }

    /// Get the era overview text.
    pub fn era_summary(&self) -> Option<&str> {
        if self.era_summary_text.is_empty() { None } else { Some(&self.era_summary_text) }
    }

    /// Get province religion narrative.
    pub fn province_religion_text(&self, province_id: u16) -> Option<&str> {
        self.province_religion.get(&province_id).map(|s| s.as_str())
    }

    /// Get diplomacy narrative between two factions (order-independent).
    pub fn diplomacy_text(&self, a: u8, b: u8) -> Option<&str> {
        let key = if a <= b { (a, b) } else { (b, a) };
        self.diplomacy_narratives.get(&key).map(|s| s.as_str())
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
