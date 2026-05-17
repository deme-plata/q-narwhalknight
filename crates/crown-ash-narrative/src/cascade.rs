//! Cascading Text Generation — Progressive narrative enrichment.
//!
//! Implements the Nemotron Cascade Pattern for optimal quality/speed tradeoff:
//!
//! ```text
//! Event occurs
//!   → Tier 0: Structured data (instant, always)     → API/SSE
//!   → Tier 1: Template narrative (instant, always)   → Event feed
//!   → Tier 2: Short LLM dialog (1-3s, if notable)   → Speech bubble
//!   → Tier 3: Deep LLM narrative (5-15s, if epic)   → Chronicle panel
//! ```
//!
//! Each tier fires independently. Lower tiers never wait for higher tiers.
//! The UI progressively enriches as higher-tier text arrives via SSE.
//!
//! # Architecture
//!
//! The `CascadeEngine` produces a `CascadeResult` for each event:
//!
//! - **Tier 0** (`structured`): Always present. Raw event data as JSON for API consumers.
//! - **Tier 1** (`prose` + `summary`): Always present. Template-generated text.
//! - **Tier 2** (`dialog_requests`): Zero or more `NarrativeRequest`s for short dialog.
//!   Only for Notable+ events. The server sends these to Mistral-7B.
//! - **Tier 3** (`deep_requests`): Zero or more `NarrativeRequest`s for epic narrative.
//!   Only for Epic events. The server sends these to Nemotron or large model.
//!
//! The server processes tiers 2-3 asynchronously and pushes results via SSE as they arrive.
//! The client shows template text instantly, then replaces/enriches with LLM text.
//!
//! # SSE Event Flow
//!
//! ```text
//! Client sees:
//!
//! t=0ms    crown_ash_event  { tier: 0, data: { ... } }           ← structured
//! t=0ms    crown_ash_prose  { tier: 1, prose: "Steel met...", summary: "Battle..." }
//! t=1500ms crown_ash_dialog { tier: 2, speaker: "King Aldric", text: "Victory!" }
//! t=8000ms crown_ash_epic   { tier: 3, type: "battle_epic", text: "The dawn broke..." }
//! ```

use crate::llm::{self, GenerationType, NarrativeRequest};
use crate::{Importance, NarrativeEngine, WorldContext};
use crown_ash_types::GameEvent;
use serde::{Deserialize, Serialize};

// ─── Cascade Result ─────────────────────────────────────────────────────────

/// The complete output of the cascade engine for a single event.
///
/// Tier 0 + Tier 1 are always populated (instant).
/// Tier 2 + Tier 3 contain deferred LLM requests (may be empty).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeResult {
    /// The original game event (Tier 0 — structured data).
    pub event: GameEvent,
    /// Event importance classification.
    pub importance: Importance,

    // ── Tier 1: Template (instant) ──
    /// Template-generated prose narrative.
    pub prose: String,
    /// One-line event summary.
    pub summary: String,

    // ── Tier 2: Short dialog (deferred, 1-3s) ──
    /// LLM requests for short in-character dialog. Empty for Minor events.
    pub dialog_requests: Vec<NarrativeRequest>,

    // ── Tier 3: Deep narrative (deferred, 5-15s) ──
    /// LLM requests for deep narrative (battle epics, speeches). Empty for non-Epic.
    pub deep_requests: Vec<NarrativeRequest>,
}

impl CascadeResult {
    /// Total number of deferred LLM requests (Tier 2 + Tier 3).
    pub fn pending_llm_count(&self) -> usize {
        self.dialog_requests.len() + self.deep_requests.len()
    }

    /// Whether any LLM generation is needed.
    pub fn has_llm_work(&self) -> bool {
        self.pending_llm_count() > 0
    }

    /// All LLM requests in priority order (dialog first, then deep narrative).
    pub fn all_requests(&self) -> Vec<&NarrativeRequest> {
        self.dialog_requests.iter()
            .chain(self.deep_requests.iter())
            .collect()
    }

    /// Estimated total generation time in seconds (rough, for UI progress).
    pub fn estimated_generation_secs(&self) -> f32 {
        let dialog_time: f32 = self.dialog_requests.iter()
            .map(|r| r.max_tokens as f32 / 25.0) // ~25 tok/s for Mistral
            .sum();
        let deep_time: f32 = self.deep_requests.iter()
            .map(|r| r.max_tokens as f32 / 15.0) // ~15 tok/s for Nemotron
            .sum();
        dialog_time + deep_time
    }
}

// ─── Cascade Turn Summary ───────────────────────────────────────────────────

/// Summary of all cascade results for a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnCascade {
    /// The game turn number.
    pub turn: u32,
    /// Cascade results for each event this turn.
    pub results: Vec<CascadeResult>,
    /// Total deferred LLM requests across all events.
    pub total_llm_requests: usize,
    /// Events classified as Epic this turn.
    pub epic_count: usize,
    /// Events classified as Notable this turn.
    pub notable_count: usize,
}

impl TurnCascade {
    /// Get only the events that need LLM generation.
    pub fn llm_events(&self) -> Vec<&CascadeResult> {
        self.results.iter().filter(|r| r.has_llm_work()).collect()
    }

    /// Get all LLM requests across all events, ordered by priority.
    /// Epic events come first, then Notable. Within each, dialog before deep.
    pub fn prioritized_requests(&self) -> Vec<(&CascadeResult, &NarrativeRequest)> {
        let mut requests: Vec<(&CascadeResult, &NarrativeRequest)> = Vec::new();

        // Epic dialog first (highest priority — most impactful + fast)
        for result in &self.results {
            if result.importance == Importance::Epic {
                for req in &result.dialog_requests {
                    requests.push((result, req));
                }
            }
        }
        // Notable dialog
        for result in &self.results {
            if result.importance == Importance::Notable {
                for req in &result.dialog_requests {
                    requests.push((result, req));
                }
            }
        }
        // Epic deep narrative (slower but important)
        for result in &self.results {
            if result.importance == Importance::Epic {
                for req in &result.deep_requests {
                    requests.push((result, req));
                }
            }
        }
        // Notable deep narrative (lowest priority)
        for result in &self.results {
            if result.importance == Importance::Notable {
                for req in &result.deep_requests {
                    requests.push((result, req));
                }
            }
        }

        requests
    }
}

// ─── Cascade Engine ─────────────────────────────────────────────────────────

/// The cascade engine — orchestrates multi-tier narrative generation.
///
/// Stateless. Takes events + context, produces `CascadeResult`s with
/// instant template text and deferred LLM requests.
pub struct CascadeEngine {
    narrative: NarrativeEngine,
}

impl CascadeEngine {
    pub fn new() -> Self {
        Self {
            narrative: NarrativeEngine::new(),
        }
    }

    /// Process a single event through all cascade tiers.
    ///
    /// Returns a `CascadeResult` with instant Tier 0/1 text and deferred
    /// Tier 2/3 LLM requests that the server processes asynchronously.
    pub fn process_event(&self, event: &GameEvent, ctx: &WorldContext) -> CascadeResult {
        // Tier 0 + 1: Instant template generation
        let narrative = self.narrative.narrate(event, ctx);
        let importance = narrative.importance;

        // Tier 2 + 3: Build LLM requests based on importance
        let gen_types = llm::classify_generation_types(event, importance);

        let mut dialog_requests = Vec::new();
        let mut deep_requests = Vec::new();

        for gen_type in gen_types {
            if let Some(req) = self.build_request(gen_type, event, ctx) {
                if gen_type.prefers_large_model() {
                    deep_requests.push(req);
                } else {
                    dialog_requests.push(req);
                }
            }
        }

        CascadeResult {
            event: event.clone(),
            importance,
            prose: narrative.prose,
            summary: narrative.summary,
            dialog_requests,
            deep_requests,
        }
    }

    /// Process all events for a turn.
    pub fn process_turn(&self, turn: u32, events: &[GameEvent], ctx: &WorldContext) -> TurnCascade {
        let results: Vec<CascadeResult> = events.iter()
            .map(|e| self.process_event(e, ctx))
            .collect();

        let total_llm_requests: usize = results.iter()
            .map(|r| r.pending_llm_count())
            .sum();
        let epic_count = results.iter()
            .filter(|r| r.importance == Importance::Epic)
            .count();
        let notable_count = results.iter()
            .filter(|r| r.importance == Importance::Notable)
            .count();

        TurnCascade {
            turn,
            results,
            total_llm_requests,
            epic_count,
            notable_count,
        }
    }

    /// Build an LLM request for a specific generation type.
    ///
    /// Returns `None` if the generation type doesn't apply to this event.
    fn build_request(
        &self,
        gen_type: GenerationType,
        event: &GameEvent,
        ctx: &WorldContext,
    ) -> Option<NarrativeRequest> {
        match gen_type {
            GenerationType::BattleEpic => {
                llm::build_battle_epic_request(event, ctx)
            }
            GenerationType::SuccessionSpeech => {
                match event {
                    GameEvent::SuccessionCrisis { faction, dead_ruler, claimants, realm_split, .. } => {
                        Some(llm::build_succession_request(
                            ctx.faction_name(*faction),
                            ctx.character_name(*dead_ruler),
                            claimants.len() as u8,
                            *realm_split,
                        ))
                    }
                    GameEvent::RealmSplit { original_faction, .. } => {
                        Some(llm::build_succession_request(
                            ctx.faction_name(*original_faction),
                            "the old ruler",
                            2,
                            true,
                        ))
                    }
                    _ => None,
                }
            }
            GenerationType::DiplomaticSpeech => {
                // For diplomatic events, we don't have speaker info at this level.
                // Return a generic chronicler request instead.
                let summary = crate::templates::render_summary(event, ctx);
                Some(NarrativeRequest {
                    generation_type: GenerationType::DiplomaticSpeech,
                    system_prompt: "You are a medieval chronicler recording diplomatic events. \
                        Write formal, dramatic prose.".to_string(),
                    user_prompt: format!(
                        "Write a 2-3 sentence account of this diplomatic event:\n\n{}\n\n\
                         Capture the gravity and political implications.",
                        summary
                    ),
                    max_tokens: gen_type.max_tokens(),
                    temperature: 0.7,
                    deferrable: true,
                })
            }
            GenerationType::ShortDialog => {
                // Without specific speaker info, build a general reaction.
                let summary = crate::templates::render_summary(event, ctx);
                Some(NarrativeRequest {
                    generation_type: GenerationType::ShortDialog,
                    system_prompt: "You are a medieval ruler reacting to news. \
                        Speak in the first person, 1-2 sentences. \
                        Use medieval-flavored English.".to_string(),
                    user_prompt: format!(
                        "React to this news:\n\n{}\n\n\
                         Show emotion appropriate to the gravity of the event.",
                        summary
                    ),
                    max_tokens: gen_type.max_tokens(),
                    temperature: 0.7,
                    deferrable: true,
                })
            }
            GenerationType::ChronicleEntry => {
                let summary = crate::templates::render_summary(event, ctx);
                Some(NarrativeRequest {
                    generation_type: GenerationType::ChronicleEntry,
                    system_prompt: "You are a medieval chronicler. Write one paragraph \
                        recording this event for the realm's official history. \
                        Formal, evocative, medieval English.".to_string(),
                    user_prompt: format!(
                        "Record this event in the chronicle:\n\n{}\n\n\
                         Write 2-3 sentences of dramatic prose.",
                        summary
                    ),
                    max_tokens: gen_type.max_tokens(),
                    temperature: 0.6,
                    deferrable: true,
                })
            }
            // BiographySummary and HistoryNarrative are on-demand, not event-triggered
            GenerationType::BiographySummary | GenerationType::HistoryNarrative => None,
        }
    }
}

impl Default for CascadeEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─── SSE Event Types ────────────────────────────────────────────────────────

/// SSE event types for the cascade system.
///
/// The server emits these as SSE events, each with a `tier` field so the
/// client knows where to display them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CascadeSseEvent {
    /// Tier 0: Structured event data.
    #[serde(rename = "crown_ash_event")]
    StructuredEvent {
        tier: u8,
        event: GameEvent,
        importance: Importance,
    },
    /// Tier 1: Template prose (instant).
    #[serde(rename = "crown_ash_prose")]
    TemplateProse {
        tier: u8,
        prose: String,
        summary: String,
        importance: Importance,
    },
    /// Tier 2: Short LLM dialog (1-3s).
    #[serde(rename = "crown_ash_dialog")]
    Dialog {
        tier: u8,
        speaker: String,
        text: String,
        generation_type: GenerationType,
    },
    /// Tier 3: Deep LLM narrative (5-15s).
    #[serde(rename = "crown_ash_epic")]
    DeepNarrative {
        tier: u8,
        text: String,
        generation_type: GenerationType,
    },
    /// Streaming token for Tier 2/3 (progressive display).
    #[serde(rename = "crown_ash_token")]
    StreamingToken {
        tier: u8,
        token: String,
        generation_type: GenerationType,
    },
}

impl CascadeSseEvent {
    /// Create Tier 0 + Tier 1 SSE events from a cascade result.
    pub fn from_cascade_result(result: &CascadeResult) -> Vec<Self> {
        vec![
            CascadeSseEvent::StructuredEvent {
                tier: 0,
                event: result.event.clone(),
                importance: result.importance,
            },
            CascadeSseEvent::TemplateProse {
                tier: 1,
                prose: result.prose.clone(),
                summary: result.summary.clone(),
                importance: result.importance,
            },
        ]
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crown_ash_types::army::BattleResult;
    use crown_ash_types::fixed_point::FixedPoint;

    fn ctx() -> WorldContext {
        WorldContext {
            province_names: vec![
                (0, "Frosthold".into()),
                (7, "Ashenmere".into()),
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
            army_factions: vec![(100, 0), (200, 1)],
            current_turn: 50,
        }
    }

    #[test]
    fn minor_event_no_llm_requests() {
        let engine = CascadeEngine::new();
        let event = GameEvent::Harvest { province: 7, prosperity_gain: 200, turn: 50 };
        let result = engine.process_event(&event, &ctx());

        assert_eq!(result.importance, Importance::Minor);
        assert!(!result.prose.is_empty());
        assert!(!result.summary.is_empty());
        assert!(result.dialog_requests.is_empty());
        assert!(result.deep_requests.is_empty());
        assert!(!result.has_llm_work());
    }

    #[test]
    fn notable_battle_gets_tier2_only() {
        let engine = CascadeEngine::new();
        let event = GameEvent::Battle(BattleResult {
            attacker_army: 100,
            defender_army: Some(200),
            province: 7,
            attacker_casualties: 100,
            defender_casualties: 80,
            attacker_won: true,
            random_factor: FixedPoint::from_int(1000),
            turn: 50,
        });
        let result = engine.process_event(&event, &ctx());

        assert_eq!(result.importance, Importance::Notable);
        assert!(!result.dialog_requests.is_empty(), "Should have Tier 2 dialog");
        assert!(result.deep_requests.is_empty(), "Notable should not have Tier 3");
        assert!(result.has_llm_work());
    }

    #[test]
    fn epic_battle_gets_all_tiers() {
        let engine = CascadeEngine::new();
        let event = GameEvent::Battle(BattleResult {
            attacker_army: 100,
            defender_army: Some(200),
            province: 7,
            attacker_casualties: 400,
            defender_casualties: 350,
            attacker_won: true,
            random_factor: FixedPoint::from_int(1000),
            turn: 50,
        });
        let result = engine.process_event(&event, &ctx());

        assert_eq!(result.importance, Importance::Epic);
        assert!(!result.prose.is_empty(), "Tier 1 always present");
        assert!(!result.dialog_requests.is_empty(), "Should have Tier 2");
        assert!(!result.deep_requests.is_empty(), "Epic should have Tier 3");

        // Tier 3 should include battle epic
        let has_battle_epic = result.deep_requests.iter()
            .any(|r| r.generation_type == GenerationType::BattleEpic);
        assert!(has_battle_epic, "Epic battle should trigger BattleEpic");
    }

    #[test]
    fn succession_crisis_epic_gets_speech() {
        let engine = CascadeEngine::new();
        let event = GameEvent::SuccessionCrisis {
            faction: 0,
            dead_ruler: 1,
            claimants: vec![2, 3, 4],
            realm_split: true,
            turn: 50,
        };
        let result = engine.process_event(&event, &ctx());

        assert_eq!(result.importance, Importance::Epic);
        let has_speech = result.deep_requests.iter()
            .any(|r| r.generation_type == GenerationType::SuccessionSpeech);
        assert!(has_speech, "Epic succession should trigger SuccessionSpeech");
    }

    #[test]
    fn realm_split_gets_succession_speech() {
        let engine = CascadeEngine::new();
        let event = GameEvent::RealmSplit {
            original_faction: 0,
            new_faction: 3,
            rebel_leader: 2,
            provinces_lost: 2,
            turn: 50,
        };
        let result = engine.process_event(&event, &ctx());

        assert_eq!(result.importance, Importance::Epic);
        let has_speech = result.deep_requests.iter()
            .any(|r| r.generation_type == GenerationType::SuccessionSpeech);
        assert!(has_speech);
    }

    #[test]
    fn process_turn_summary() {
        let engine = CascadeEngine::new();
        let events = vec![
            GameEvent::Harvest { province: 7, prosperity_gain: 200, turn: 50 },
            GameEvent::Battle(BattleResult {
                attacker_army: 100,
                defender_army: Some(200),
                province: 7,
                attacker_casualties: 400,
                defender_casualties: 350,
                attacker_won: true,
                random_factor: FixedPoint::from_int(1000),
                turn: 50,
            }),
            GameEvent::FactionEliminated { faction: 1, turn: 50 },
        ];

        let turn = engine.process_turn(50, &events, &ctx());

        assert_eq!(turn.turn, 50);
        assert_eq!(turn.results.len(), 3);
        assert_eq!(turn.epic_count, 2); // battle (>500 casualties) + faction eliminated
        assert_eq!(turn.notable_count, 0);
        assert!(turn.total_llm_requests > 0);
    }

    #[test]
    fn prioritized_requests_order() {
        let engine = CascadeEngine::new();
        let events = vec![
            // Notable event
            GameEvent::WarDeclared {
                attacker: 0,
                defender: 1,
                casus_belli: "honor".into(),
                turn: 50,
            },
            // Epic event
            GameEvent::FactionEliminated { faction: 1, turn: 50 },
        ];

        let turn = engine.process_turn(50, &events, &ctx());
        let prioritized = turn.prioritized_requests();

        if prioritized.len() >= 2 {
            // Epic requests should come before Notable
            let first_importance = prioritized[0].0.importance;
            assert_eq!(first_importance, Importance::Epic,
                "Epic events should be prioritized first");
        }
    }

    #[test]
    fn cascade_sse_events_from_result() {
        let engine = CascadeEngine::new();
        let event = GameEvent::Harvest { province: 7, prosperity_gain: 200, turn: 50 };
        let result = engine.process_event(&event, &ctx());

        let sse_events = CascadeSseEvent::from_cascade_result(&result);
        assert_eq!(sse_events.len(), 2); // Tier 0 + Tier 1
    }

    #[test]
    fn estimated_generation_time() {
        let engine = CascadeEngine::new();
        let event = GameEvent::Battle(BattleResult {
            attacker_army: 100,
            defender_army: Some(200),
            province: 7,
            attacker_casualties: 400,
            defender_casualties: 350,
            attacker_won: true,
            random_factor: FixedPoint::from_int(1000),
            turn: 50,
        });
        let result = engine.process_event(&event, &ctx());
        let est = result.estimated_generation_secs();
        assert!(est > 0.0, "Epic event should have non-zero generation time estimate");
    }

    #[test]
    fn war_declared_gets_diplomatic_speech() {
        let engine = CascadeEngine::new();
        let event = GameEvent::WarDeclared {
            attacker: 0,
            defender: 1,
            casus_belli: "territorial claim".into(),
            turn: 50,
        };
        let result = engine.process_event(&event, &ctx());

        assert_eq!(result.importance, Importance::Notable);
        let has_diplo = result.dialog_requests.iter()
            .any(|r| r.generation_type == GenerationType::DiplomaticSpeech);
        assert!(has_diplo, "War declared should trigger DiplomaticSpeech");
    }

    #[test]
    fn marriage_alliance_gets_diplomatic() {
        let engine = CascadeEngine::new();
        let event = GameEvent::MarriageAlliance {
            character_a: 1,
            character_b: 2,
            faction_a: 0,
            faction_b: 1,
            turn: 50,
        };
        let result = engine.process_event(&event, &ctx());

        assert_eq!(result.importance, Importance::Notable);
        let has_diplo = result.dialog_requests.iter()
            .any(|r| r.generation_type == GenerationType::DiplomaticSpeech
                   || r.generation_type == GenerationType::ChronicleEntry);
        assert!(has_diplo);
    }
}
