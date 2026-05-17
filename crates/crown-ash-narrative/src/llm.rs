//! LLM Integration — Deep narrative generation via q-ai-inference.
//!
//! This module provides structured prompt engineering for multi-tier LLM
//! narrative generation. It builds context-rich prompts that the server
//! sends to Mistral-7B (fast dialog) or Nemotron (epic narratives).
//!
//! # Architecture
//!
//! The narrative crate itself does NOT depend on q-ai-inference directly.
//! Instead, it provides:
//!
//! 1. **Prompt builders** — construct optimized prompts for each narrative tier
//! 2. **Output parsers** — clean/format raw LLM output for display
//! 3. **Token budgets** — max_tokens for each generation type
//! 4. **System prompts** — character personas, chronicler voice, etc.
//!
//! The server (crown-ash-api) wires these prompts to the actual inference engine.
//!
//! # Generation Types
//!
//! | Type              | Model    | Tokens | Latency | Trigger              |
//! |-------------------|----------|--------|---------|----------------------|
//! | Short Dialog      | Mistral  | 60     | 1-3s    | Notable events       |
//! | Battle Epic       | Nemotron | 200    | 5-10s   | Epic battles         |
//! | Succession Speech | Nemotron | 150    | 4-8s    | Succession crises    |
//! | Chronicle Entry   | Mistral  | 100    | 3-5s    | Notable+ events      |
//! | Biography Summary | Nemotron | 250    | 8-15s   | On-demand (detail)   |
//! | Faction History   | Nemotron | 300    | 10-15s  | On-demand (detail)   |

use crate::{Importance, WorldContext, personality};
use crown_ash_types::character::Trait;
use crown_ash_types::GameEvent;
use serde::{Deserialize, Serialize};

// ─── Generation Request ─────────────────────────────────────────────────────

/// The type of LLM narrative to generate. Determines prompt style and token budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationType {
    /// Short 1-2 sentence in-character dialog (Mistral, 60 tokens).
    ShortDialog,
    /// Dramatic battle narrative (Nemotron, 200 tokens).
    BattleEpic,
    /// Ruler's speech during succession crisis (Nemotron, 150 tokens).
    SuccessionSpeech,
    /// Single chronicle paragraph for a character's history (Mistral, 100 tokens).
    ChronicleEntry,
    /// Full biography summary for a character (Nemotron, 250 tokens).
    BiographySummary,
    /// Province or faction history prose (Nemotron, 300 tokens).
    HistoryNarrative,
    /// War declaration or treaty speech (Mistral, 80 tokens).
    DiplomaticSpeech,
}

impl GenerationType {
    /// Maximum tokens for this generation type.
    pub fn max_tokens(&self) -> usize {
        match self {
            GenerationType::ShortDialog => 60,
            GenerationType::BattleEpic => 200,
            GenerationType::SuccessionSpeech => 150,
            GenerationType::ChronicleEntry => 100,
            GenerationType::BiographySummary => 250,
            GenerationType::HistoryNarrative => 300,
            GenerationType::DiplomaticSpeech => 80,
        }
    }

    /// Whether this type benefits from a larger model (Nemotron vs Mistral).
    pub fn prefers_large_model(&self) -> bool {
        matches!(
            self,
            GenerationType::BattleEpic
                | GenerationType::SuccessionSpeech
                | GenerationType::BiographySummary
                | GenerationType::HistoryNarrative
        )
    }
}

/// A complete request for LLM narrative generation.
///
/// The server takes this and feeds `system_prompt` + `user_prompt` to the
/// inference engine with `max_tokens` as the generation limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeRequest {
    /// What kind of narrative this is.
    pub generation_type: GenerationType,
    /// System prompt (persona, writing style instructions).
    pub system_prompt: String,
    /// User prompt (the actual event description and task).
    pub user_prompt: String,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Temperature suggestion (0.0 = deterministic, 0.7 = creative).
    pub temperature: f32,
    /// Whether this request can be deferred (non-blocking).
    pub deferrable: bool,
}

// ─── System Prompts ─────────────────────────────────────────────────────────

/// The base chronicler system prompt — used for most narrative generation.
const CHRONICLER_SYSTEM: &str = "\
You are a medieval chronicler recording the history of a feudal realm. \
Write in the style of a medieval chronicle — formal, dramatic, evocative. \
Use archaic-flavored English but keep it readable. \
Never use modern language, slang, or anachronisms. \
Never break character. Never reference anything outside the medieval world.";

/// System prompt for battle narratives — more visceral and cinematic.
const BATTLE_NARRATOR_SYSTEM: &str = "\
You are a battle poet chronicling the great conflicts of the realm. \
Write vivid, cinematic descriptions of combat — the clash of steel, \
the thunder of cavalry, the cries of the fallen. \
Use dramatic imagery and evocative language. \
Name specific combatants and factions when provided. \
Keep the tone somber and respectful of the dead, even for villains.";

/// System prompt for diplomatic speeches.
const DIPLOMAT_SYSTEM: &str = "\
You are recording the words of rulers and diplomats in a medieval court. \
Write formal speeches appropriate to a throne room or war council. \
Each speaker has a distinct personality — match their tone to their traits. \
Keep speeches concise and impactful. Use medieval-flavored English.";

// ─── Prompt Builders ────────────────────────────────────────────────────────

/// Build a short dialog request for a Notable event.
///
/// This generates 1-2 sentences spoken by a character reacting to an event.
/// Used for speech bubbles in the UI.
pub fn build_dialog_request(
    event: &GameEvent,
    ctx: &WorldContext,
    speaker_name: &str,
    speaker_traits: &[Trait],
    speaker_role: &str,
    faction_name: &str,
) -> NarrativeRequest {
    let profile = personality::build_profile(speaker_traits);
    let persona = personality::build_persona_prompt(speaker_name, speaker_role, faction_name, &profile);

    let event_desc = crate::templates::render_summary(event, ctx);

    let user_prompt = format!(
        "React to this event in 1-2 sentences, speaking as yourself:\n\n\
         Event: {}\n\n\
         Speak naturally in character. Show emotion appropriate to the situation.",
        event_desc
    );

    NarrativeRequest {
        generation_type: GenerationType::ShortDialog,
        system_prompt: persona,
        user_prompt,
        max_tokens: GenerationType::ShortDialog.max_tokens(),
        temperature: 0.7,
        deferrable: true,
    }
}

/// Build a battle epic request for an Epic battle event.
pub fn build_battle_epic_request(
    event: &GameEvent,
    ctx: &WorldContext,
) -> Option<NarrativeRequest> {
    let GameEvent::Battle(result) = event else {
        return None;
    };

    let attacker_name = ctx.army_faction_name(result.attacker_army);
    let defender_name = result.defender_army
        .map(|did| ctx.army_faction_name(did))
        .unwrap_or("the garrison");
    let province_name = ctx.province_name(result.province);
    let total_dead = result.attacker_casualties + result.defender_casualties;
    let victor = if result.attacker_won { attacker_name } else { defender_name };

    let user_prompt = format!(
        "Write a dramatic 3-4 sentence account of this battle:\n\n\
         Location: the fields of {province}\n\
         Attacker: {attacker} host\n\
         Defender: {defender} forces\n\
         Attacker losses: {att_dead}\n\
         Defender losses: {def_dead}\n\
         Total fallen: {total}\n\
         Victor: {victor}\n\n\
         Describe the clash, the turning point, and the aftermath. \
         Name the factions. Honor the fallen.",
        province = province_name,
        attacker = attacker_name,
        defender = defender_name,
        att_dead = result.attacker_casualties,
        def_dead = result.defender_casualties,
        total = total_dead,
        victor = victor,
    );

    Some(NarrativeRequest {
        generation_type: GenerationType::BattleEpic,
        system_prompt: BATTLE_NARRATOR_SYSTEM.to_string(),
        user_prompt,
        max_tokens: GenerationType::BattleEpic.max_tokens(),
        temperature: 0.8,
        deferrable: true,
    })
}

/// Build a succession speech request.
pub fn build_succession_request(
    faction_name: &str,
    dead_ruler_name: &str,
    heir_count: u8,
    realm_split: bool,
) -> NarrativeRequest {
    let situation = if realm_split {
        format!(
            "The realm of {} has shattered! With {}'s death, {} claimants \
             tear the kingdom apart. The realm splits — civil war is inevitable.",
            faction_name, dead_ruler_name, heir_count
        )
    } else {
        format!(
            "A succession crisis grips {}. {} is dead, and {} claimants \
             eye the throne. The court holds its breath.",
            faction_name, dead_ruler_name, heir_count
        )
    };

    let user_prompt = format!(
        "Write a dramatic 2-3 sentence chronicle entry about this succession crisis:\n\n\
         {}\n\n\
         Capture the tension, the uncertainty, and the stakes.",
        situation
    );

    NarrativeRequest {
        generation_type: GenerationType::SuccessionSpeech,
        system_prompt: CHRONICLER_SYSTEM.to_string(),
        user_prompt,
        max_tokens: GenerationType::SuccessionSpeech.max_tokens(),
        temperature: 0.7,
        deferrable: true,
    }
}

/// Build a diplomatic speech request (war declaration or treaty).
pub fn build_diplomatic_request(
    event: &GameEvent,
    ctx: &WorldContext,
    speaker_name: &str,
    speaker_traits: &[Trait],
    speaker_role: &str,
    faction_name: &str,
) -> NarrativeRequest {
    let profile = personality::build_profile(speaker_traits);
    let persona = personality::build_persona_prompt(speaker_name, speaker_role, faction_name, &profile);

    let event_desc = crate::templates::render_summary(event, ctx);

    let user_prompt = match event {
        GameEvent::WarDeclared { .. } => format!(
            "Announce this declaration of war in 2-3 sentences, \
             speaking to your court and your enemy:\n\n\
             Event: {}\n\n\
             Be commanding and resolute.",
            event_desc
        ),
        GameEvent::TreatySigned { .. } => format!(
            "Announce this peace treaty in 2-3 sentences, \
             speaking to your court:\n\n\
             Event: {}\n\n\
             Match your personality — are you relieved? Resentful? Magnanimous?",
            event_desc
        ),
        _ => format!(
            "React to this diplomatic event in 2-3 sentences:\n\n\
             Event: {}\n\n\
             Speak in character.",
            event_desc
        ),
    };

    NarrativeRequest {
        generation_type: GenerationType::DiplomaticSpeech,
        system_prompt: format!("{}\n\n{}", DIPLOMAT_SYSTEM, persona),
        user_prompt,
        max_tokens: GenerationType::DiplomaticSpeech.max_tokens(),
        temperature: 0.7,
        deferrable: true,
    }
}

/// Build a character biography summary request.
///
/// Takes the character's chronicle entries (template prose) and asks the LLM
/// to weave them into a cohesive biography paragraph.
pub fn build_biography_request(
    character_name: &str,
    faction_name: &str,
    chronicle_entries: &[String],
) -> NarrativeRequest {
    let events_text = chronicle_entries.join("\n- ");

    let user_prompt = format!(
        "Write a 3-5 sentence biography of {name}, a figure of {faction}, \
         based on these life events:\n\n\
         - {events}\n\n\
         Weave these events into a cohesive narrative. \
         Highlight the most dramatic moments. \
         End with their legacy or current situation.",
        name = character_name,
        faction = faction_name,
        events = events_text,
    );

    NarrativeRequest {
        generation_type: GenerationType::BiographySummary,
        system_prompt: CHRONICLER_SYSTEM.to_string(),
        user_prompt,
        max_tokens: GenerationType::BiographySummary.max_tokens(),
        temperature: 0.6,
        deferrable: false, // On-demand, user is waiting
    }
}

/// Build a province or faction history narrative request.
///
/// Takes the template-generated history text and asks the LLM to
/// rewrite it as flowing medieval chronicle prose.
pub fn build_history_request(
    _entity_name: &str,
    template_history: &str,
    is_faction: bool,
) -> NarrativeRequest {
    let entity_type = if is_faction { "faction" } else { "province" };

    let user_prompt = format!(
        "Rewrite this {entity_type} history as flowing medieval chronicle prose \
         (4-6 sentences). Make it dramatic and evocative:\n\n\
         {history}\n\n\
         Keep all the facts but elevate the language. \
         Use vivid imagery and a sense of the sweep of history.",
        entity_type = entity_type,
        history = template_history,
    );

    NarrativeRequest {
        generation_type: GenerationType::HistoryNarrative,
        system_prompt: CHRONICLER_SYSTEM.to_string(),
        user_prompt,
        max_tokens: GenerationType::HistoryNarrative.max_tokens(),
        temperature: 0.6,
        deferrable: false,
    }
}

// ─── Output Processing ──────────────────────────────────────────────────────

/// Clean raw LLM output for display.
///
/// Strips common artifacts: leading/trailing whitespace, incomplete sentences,
/// markdown formatting, and repeated phrases.
pub fn clean_output(raw: &str) -> String {
    let mut text = raw.trim().to_string();

    // Remove markdown artifacts that LLMs sometimes inject
    text = text.replace("**", "");
    text = text.replace("##", "");
    text = text.replace("# ", "");

    // Remove any "as a medieval chronicler" or similar meta-commentary
    if let Some(idx) = text.find("As a medieval") {
        text = text[..idx].trim().to_string();
    }
    if let Some(idx) = text.find("As a chronicler") {
        text = text[..idx].trim().to_string();
    }

    // Ensure text ends with proper punctuation
    if !text.is_empty() && !text.ends_with('.') && !text.ends_with('!') && !text.ends_with('?') {
        // Find the last complete sentence
        if let Some(last_period) = text.rfind(". ") {
            text.truncate(last_period + 1);
        } else if let Some(last_period) = text.rfind('.') {
            text.truncate(last_period + 1);
        }
    }

    text
}

/// Classify which LLM generation requests an event should trigger.
///
/// Returns a list of `GenerationType`s appropriate for this event at the
/// given importance level. The caller builds the actual requests.
pub fn classify_generation_types(
    event: &GameEvent,
    importance: Importance,
) -> Vec<GenerationType> {
    match importance {
        Importance::Minor => vec![],
        Importance::Notable => {
            let mut types = vec![GenerationType::ChronicleEntry];
            match event {
                GameEvent::Battle(_) => types.push(GenerationType::ShortDialog),
                GameEvent::WarDeclared { .. } => types.push(GenerationType::DiplomaticSpeech),
                GameEvent::TreatySigned { .. } => types.push(GenerationType::DiplomaticSpeech),
                GameEvent::PlotSucceeded { .. } | GameEvent::PlotDiscovered { .. } => {
                    types.push(GenerationType::ShortDialog);
                }
                GameEvent::SiegeStarted { .. } | GameEvent::SiegeCompleted { .. } => {
                    types.push(GenerationType::ShortDialog);
                }
                GameEvent::MarriageAlliance { .. } => {
                    types.push(GenerationType::DiplomaticSpeech);
                }
                _ => {}
            }
            types
        }
        Importance::Epic => {
            let mut types = vec![GenerationType::ChronicleEntry];
            match event {
                GameEvent::Battle(_) => {
                    types.push(GenerationType::BattleEpic);
                    types.push(GenerationType::ShortDialog);
                }
                GameEvent::FactionEliminated { .. } => {
                    types.push(GenerationType::ShortDialog);
                }
                GameEvent::RealmSplit { .. } | GameEvent::SuccessionCrisis { .. } => {
                    types.push(GenerationType::SuccessionSpeech);
                }
                _ => {
                    types.push(GenerationType::ShortDialog);
                }
            }
            types
        }
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
            province_names: vec![(7, "Ashenmere".into())],
            faction_names: vec![(0, "Ashen Crown".into()), (1, "Vale Princes".into())],
            character_names: vec![(1, "King Aldric".into())],
            faction_cultures: vec![],
            army_factions: vec![(100, 0), (200, 1)],
            current_turn: 50,
        }
    }

    #[test]
    fn dialog_request_contains_persona() {
        let event = GameEvent::WarDeclared {
            attacker: 0,
            defender: 1,
            casus_belli: "territorial claim".into(),
            turn: 50,
        };
        let req = build_dialog_request(
            &event,
            &ctx(),
            "King Aldric",
            &[Trait::Brave, Trait::Just],
            "Ruler",
            "Ashen Crown",
        );
        assert!(req.system_prompt.contains("King Aldric"));
        assert!(req.system_prompt.contains("Commander"));
        assert_eq!(req.max_tokens, 60);
        assert!(req.deferrable);
    }

    #[test]
    fn battle_epic_request_details() {
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
        let req = build_battle_epic_request(&event, &ctx()).unwrap();
        assert!(req.user_prompt.contains("Ashenmere"));
        assert!(req.user_prompt.contains("Ashen Crown"));
        assert!(req.user_prompt.contains("Vale Princes"));
        assert!(req.user_prompt.contains("750")); // total dead
        assert_eq!(req.max_tokens, 200);
        assert!(req.system_prompt.contains("battle poet"));
    }

    #[test]
    fn battle_epic_returns_none_for_non_battle() {
        let event = GameEvent::Harvest {
            province: 7,
            prosperity_gain: 100,
            turn: 50,
        };
        assert!(build_battle_epic_request(&event, &ctx()).is_none());
    }

    #[test]
    fn succession_request_realm_split() {
        let req = build_succession_request("Ashen Crown", "King Aldric", 3, true);
        assert!(req.user_prompt.contains("shattered"));
        assert!(req.user_prompt.contains("3 claimants"));
        assert_eq!(req.generation_type, GenerationType::SuccessionSpeech);
    }

    #[test]
    fn succession_request_no_split() {
        let req = build_succession_request("Vale Princes", "Duke Varen", 2, false);
        assert!(req.user_prompt.contains("succession crisis"));
        assert!(!req.user_prompt.contains("shattered"));
    }

    #[test]
    fn biography_request_includes_events() {
        let entries = vec![
            "Born in Frosthold, Turn 1".to_string(),
            "Won the Battle of Ashenmere, Turn 30".to_string(),
            "Crowned Ruler of the Ashen Crown, Turn 35".to_string(),
        ];
        let req = build_biography_request("King Aldric", "Ashen Crown", &entries);
        assert!(req.user_prompt.contains("Born in Frosthold"));
        assert!(req.user_prompt.contains("Battle of Ashenmere"));
        assert!(!req.deferrable); // On-demand, user waiting
        assert_eq!(req.max_tokens, 250);
    }

    #[test]
    fn history_request_province() {
        let req = build_history_request(
            "Ashenmere",
            "Ashenmere has changed hands 3 times and weathered 2 plagues.",
            false,
        );
        assert!(req.user_prompt.contains("province"));
        assert!(req.user_prompt.contains("changed hands"));
        assert_eq!(req.generation_type, GenerationType::HistoryNarrative);
    }

    #[test]
    fn history_request_faction() {
        let req = build_history_request(
            "Ashen Crown",
            "The Ashen Crown conquered 5 provinces and fought 3 wars.",
            true,
        );
        assert!(req.user_prompt.contains("faction"));
    }

    #[test]
    fn clean_output_strips_artifacts() {
        assert_eq!(
            clean_output("  **The battle** was fierce.  "),
            "The battle was fierce."
        );
        assert_eq!(
            clean_output("The king spoke. As a medieval chronicler, I must note"),
            "The king spoke."
        );
        // Incomplete sentence gets truncated to last complete one
        assert_eq!(
            clean_output("The army marched. The battle raged. Then the ki"),
            "The army marched. The battle raged."
        );
    }

    #[test]
    fn clean_output_preserves_good_text() {
        let good = "The crown sat heavy on his brow.";
        assert_eq!(clean_output(good), good);
    }

    #[test]
    fn classify_minor_returns_empty() {
        let event = GameEvent::Harvest { province: 0, prosperity_gain: 100, turn: 1 };
        assert!(classify_generation_types(&event, Importance::Minor).is_empty());
    }

    #[test]
    fn classify_notable_battle_gets_dialog() {
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
        let types = classify_generation_types(&event, Importance::Notable);
        assert!(types.contains(&GenerationType::ChronicleEntry));
        assert!(types.contains(&GenerationType::ShortDialog));
    }

    #[test]
    fn classify_epic_battle_gets_all_tiers() {
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
        let types = classify_generation_types(&event, Importance::Epic);
        assert!(types.contains(&GenerationType::ChronicleEntry));
        assert!(types.contains(&GenerationType::BattleEpic));
        assert!(types.contains(&GenerationType::ShortDialog));
    }

    #[test]
    fn classify_epic_succession_gets_speech() {
        let event = GameEvent::SuccessionCrisis {
            faction: 0,
            dead_ruler: 1,
            claimants: vec![2, 3, 4],
            realm_split: true,
            turn: 50,
        };
        let types = classify_generation_types(&event, Importance::Epic);
        assert!(types.contains(&GenerationType::SuccessionSpeech));
    }

    #[test]
    fn generation_type_model_preference() {
        assert!(GenerationType::BattleEpic.prefers_large_model());
        assert!(GenerationType::BiographySummary.prefers_large_model());
        assert!(!GenerationType::ShortDialog.prefers_large_model());
        assert!(!GenerationType::ChronicleEntry.prefers_large_model());
    }

    #[test]
    fn diplomatic_request_war() {
        let event = GameEvent::WarDeclared {
            attacker: 0,
            defender: 1,
            casus_belli: "honor".into(),
            turn: 50,
        };
        let req = build_diplomatic_request(
            &event, &ctx(), "King Aldric",
            &[Trait::Brave, Trait::Just], "Ruler", "Ashen Crown",
        );
        assert!(req.user_prompt.contains("declaration of war"));
        assert_eq!(req.generation_type, GenerationType::DiplomaticSpeech);
    }

    #[test]
    fn diplomatic_request_treaty() {
        let event = GameEvent::TreatySigned {
            faction_a: 0,
            faction_b: 1,
            treaty_type: "non-aggression".into(),
            turn: 50,
        };
        let req = build_diplomatic_request(
            &event, &ctx(), "King Aldric",
            &[Trait::Gregarious, Trait::Kind], "Ruler", "Ashen Crown",
        );
        assert!(req.user_prompt.contains("peace treaty"));
        assert!(req.system_prompt.contains("Diplomat"));
    }
}
