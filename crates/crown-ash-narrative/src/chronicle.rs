//! Character Chronicle — accumulated narrative history per character.
//!
//! Every character builds a chronicle of their life events. When a player
//! clicks a character in the detail panel, they see their story: born,
//! fought battles, gained traits, plotted assassinations, died.
//!
//! Chronicles are built incrementally — each turn, relevant events are
//! filtered per-character and appended as prose entries.

use crown_ash_types::GameEvent;
use crown_ash_types::event::DeathCause;
use serde::{Deserialize, Serialize};

use crate::WorldContext;

// ─── Chronicle Entry ────────────────────────────────────────────────────────

/// A single entry in a character's chronicle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronicleEntry {
    /// Game turn when this entry was recorded.
    pub turn: u32,
    /// Prose paragraph describing the event.
    pub prose: String,
    /// Short tag for categorization (e.g., "battle", "birth", "death", "plot").
    pub tag: String,
}

// ─── Character Chronicle ────────────────────────────────────────────────────

/// The full chronicle of a character's life.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CharacterChronicle {
    /// Character ID this chronicle belongs to.
    pub character_id: u32,
    /// Character name (cached for display when character is tombstoned).
    pub character_name: String,
    /// Ordered list of chronicle entries (oldest first).
    pub entries: Vec<ChronicleEntry>,
}

impl CharacterChronicle {
    /// Create a new empty chronicle for a character.
    pub fn new(character_id: u32, name: String) -> Self {
        Self {
            character_id,
            character_name: name,
            entries: Vec::new(),
        }
    }

    /// Append an entry to the chronicle.
    pub fn append(&mut self, entry: ChronicleEntry) {
        self.entries.push(entry);
    }

    /// Get the number of recorded events.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is this chronicle empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Render the full chronicle as a single prose text (paragraphs separated
    /// by double newlines). Used for the Chronicle tab in the UI.
    pub fn render_full(&self) -> String {
        if self.entries.is_empty() {
            return format!(
                "The chronicle of {} is yet unwritten. Their story awaits.",
                self.character_name
            );
        }

        let mut text = format!("The Chronicle of {}\n\n", self.character_name);
        for entry in &self.entries {
            text.push_str(&format!("Turn {} — {}\n\n", entry.turn, entry.prose));
        }
        text
    }
}

// ─── Event → Chronicle Mapping ──────────────────────────────────────────────

/// Extract chronicle entries from a game event for a specific character.
///
/// Returns `Some(ChronicleEntry)` if the event involves this character,
/// `None` otherwise. A single event may produce entries for multiple
/// characters (e.g., a battle involves attacker and defender faction rulers).
pub fn extract_for_character(
    character_id: u32,
    character_name: &str,
    event: &GameEvent,
    ctx: &WorldContext,
) -> Option<ChronicleEntry> {
    match event {
        // Character was born
        GameEvent::CharacterBorn { character_id: cid, character_name: cname, parent, turn, .. } => {
            if *cid == character_id {
                let parent_name = ctx.character_name(*parent);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} was born, child of {}. A new thread woven into \
                         the tapestry of the realm.",
                        cname, parent_name
                    ),
                    tag: "birth".into(),
                })
            } else if *parent == character_id {
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "A child was born to {} — {} enters the world. \
                         The dynasty's future is secured for another generation.",
                        character_name, cname
                    ),
                    tag: "dynasty".into(),
                })
            } else {
                None
            }
        }

        // Character died
        GameEvent::CharacterDied { character_id: cid, character_name: cname, cause, turn } => {
            if *cid == character_id {
                let cause_prose = match cause {
                    DeathCause::OldAge => format!(
                        "{} drew their final breath, taken by the weight of years. \
                         A long life, now ended. The realm mourns the passing.",
                        cname
                    ),
                    DeathCause::Battle => format!(
                        "{} fell in battle, their blood mingling with the mud of \
                         the battlefield. A warrior's end — sudden and brutal.",
                        cname
                    ),
                    DeathCause::Disease => format!(
                        "A cruel sickness claimed {}. No healer's art could \
                         overcome the pestilence that consumed them.",
                        cname
                    ),
                    DeathCause::Assassination => format!(
                        "{} was murdered — a blade in the night silenced them \
                         forever. The assassin's identity remains shrouded in shadow.",
                        cname
                    ),
                    DeathCause::Execution => format!(
                        "{} was executed. The headsman's axe fell, and a life \
                         was ended by decree. Justice or cruelty — history will judge.",
                        cname
                    ),
                    DeathCause::Accident => format!(
                        "A tragic accident took {}. Whether fate or conspiracy, \
                         the truth died with them.",
                        cname
                    ),
                };
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: cause_prose,
                    tag: "death".into(),
                })
            } else {
                None
            }
        }

        // Succession crisis — relevant to all claimants and the dead ruler
        GameEvent::SuccessionCrisis { faction, dead_ruler, claimants, turn, .. } => {
            if *dead_ruler == character_id {
                let fac = ctx.faction_name(*faction);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "With {}'s death, {} was plunged into a succession crisis. \
                         {} claimants vied for the throne they left behind.",
                        character_name, fac, claimants.len()
                    ),
                    tag: "succession".into(),
                })
            } else if claimants.contains(&character_id) {
                let fac = ctx.faction_name(*faction);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} emerged as a claimant to the throne of {}. \
                         Ambition stirred — the crown was within reach.",
                        character_name, fac
                    ),
                    tag: "succession".into(),
                })
            } else {
                None
            }
        }

        // Realm split — relevant to rebel leader
        GameEvent::RealmSplit { rebel_leader, original_faction, provinces_lost, turn, .. } => {
            if *rebel_leader == character_id {
                let orig = ctx.faction_name(*original_faction);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} led a rebellion against {}, carving out a new domain \
                         from {} provinces. A bold gambit — a new power rises.",
                        character_name, orig, provinces_lost
                    ),
                    tag: "rebellion".into(),
                })
            } else {
                None
            }
        }

        // Plot launched — relevant to instigator and target
        GameEvent::PlotLaunched { instigator, target, plot_type, turn } => {
            if *instigator == character_id {
                let tgt = ctx.character_name(*target);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} began plotting a {} against {}. \
                         Shadows and whispers — the game of intrigue had begun.",
                        character_name, plot_type, tgt
                    ),
                    tag: "intrigue".into(),
                })
            } else if *target == character_id {
                let ins = ctx.character_name(*instigator);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "Unknown to {}, {} had begun plotting against them. \
                         A {} scheme was set in motion.",
                        character_name, ins, plot_type
                    ),
                    tag: "intrigue".into(),
                })
            } else {
                None
            }
        }

        // Plot succeeded — relevant to named instigator and target
        GameEvent::PlotSucceeded { instigator_name, target_name, plot_type, turn } => {
            if instigator_name == character_name {
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{}'s {} against {} succeeded. The schemer's patience \
                         bore fruit — dark fruit, perhaps, but fruit nonetheless.",
                        character_name, plot_type, target_name
                    ),
                    tag: "intrigue".into(),
                })
            } else if target_name == character_name {
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} fell victim to a {} plot orchestrated by {}. \
                         The blow struck true, and the consequences were swift.",
                        character_name, plot_type, instigator_name
                    ),
                    tag: "intrigue".into(),
                })
            } else {
                None
            }
        }

        // Plot discovered — relevant to named parties
        GameEvent::PlotDiscovered { instigator_name, target_name, discovered_by, turn } => {
            if instigator_name == character_name {
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{}'s plot against {} was uncovered by {}! \
                         The schemer was exposed — trust shattered, reputation in tatters.",
                        character_name, target_name, discovered_by
                    ),
                    tag: "intrigue".into(),
                })
            } else if target_name == character_name {
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "A plot against {} was discovered by {}. The would-be \
                         victim was saved by vigilance and good intelligence.",
                        character_name, discovered_by
                    ),
                    tag: "intrigue".into(),
                })
            } else if discovered_by == character_name {
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} uncovered a plot by {} against {}. \
                         The spymaster's keen eye protected the realm from treachery.",
                        character_name, instigator_name, target_name
                    ),
                    tag: "intrigue".into(),
                })
            } else {
                None
            }
        }

        // Plot foiled
        GameEvent::PlotFoiled { instigator_name, target_name, turn } => {
            if instigator_name == character_name {
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{}'s scheme against {} was foiled at the last moment. \
                         A humiliating failure — but the schemer lives to plot again.",
                        character_name, target_name
                    ),
                    tag: "intrigue".into(),
                })
            } else if target_name == character_name {
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "An attempt on {} was thwarted. Fortune smiled, \
                         and the intended victim escaped unscathed.",
                        character_name
                    ),
                    tag: "intrigue".into(),
                })
            } else {
                None
            }
        }

        // Character tombstoned — their living chronicle concludes
        GameEvent::CharacterTombstoned { character_id: cid, character_name: cname, turn } => {
            if *cid == character_id {
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} passed from living memory into legend. \
                         Their deeds are recorded in the great chronicle \
                         for future generations.",
                        cname
                    ),
                    tag: "legacy".into(),
                })
            } else {
                None
            }
        }

        // Friendship — both characters get an entry
        GameEvent::Friendship { character_a, character_b, turn, .. } => {
            if *character_a == character_id {
                let other = ctx.character_name(*character_b);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} found a true companion in {}. In a court of \
                         serpents, a genuine bond formed — rare and precious.",
                        character_name, other
                    ),
                    tag: "friendship".into(),
                })
            } else if *character_b == character_id {
                let other = ctx.character_name(*character_a);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "A friendship blossomed between {} and {}. \
                         Trust, once given, is not easily forgotten.",
                        character_name, other
                    ),
                    tag: "friendship".into(),
                })
            } else {
                None
            }
        }

        // Rivalry — both characters get an entry
        GameEvent::Rivalry { character_a, character_b, turn, .. } => {
            if *character_a == character_id {
                let other = ctx.character_name(*character_b);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} and {} became bitter rivals. Their enmity \
                         would color every interaction that followed.",
                        character_name, other
                    ),
                    tag: "rivalry".into(),
                })
            } else if *character_b == character_id {
                let other = ctx.character_name(*character_a);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "A rivalry took root between {} and {}. \
                         Cold stares and cutting words — the prelude \
                         to darker deeds.",
                        character_name, other
                    ),
                    tag: "rivalry".into(),
                })
            } else {
                None
            }
        }

        // Marriage Alliance — both characters get an entry
        GameEvent::MarriageAlliance { character_a, character_b, faction_a, faction_b, turn } => {
            if *character_a == character_id {
                let other = ctx.character_name(*character_b);
                let fb = ctx.faction_name(*faction_b);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} married {} of {}, forging a marriage alliance \
                         that bound two houses together. Love or politics? \
                         Perhaps both.",
                        character_name, other, fb
                    ),
                    tag: "marriage".into(),
                })
            } else if *character_b == character_id {
                let other = ctx.character_name(*character_a);
                let fa = ctx.faction_name(*faction_a);
                Some(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} was wed to {} of {}. The marriage alliance \
                         would reshape the balance of power between their \
                         two realms.",
                        character_name, other, fa
                    ),
                    tag: "marriage".into(),
                })
            } else {
                None
            }
        }

        // All other events don't generate per-character chronicle entries
        _ => None,
    }
}

// ─── Chronicle Builder ──────────────────────────────────────────────────────

/// Process a batch of events and update all relevant character chronicles.
///
/// Call this once per turn with the turn's events. Characters not already
/// in the map are auto-created when they appear in events.
pub fn update_chronicles(
    chronicles: &mut Vec<CharacterChronicle>,
    events: &[GameEvent],
    ctx: &WorldContext,
) {
    for event in events {
        // Collect (id, name) pairs of characters we know about
        let character_ids: Vec<(u32, String)> = chronicles
            .iter()
            .map(|c| (c.character_id, c.character_name.clone()))
            .collect();

        // Check each known character for relevance to this event
        for (cid, cname) in &character_ids {
            if let Some(entry) = extract_for_character(*cid, cname, event, ctx) {
                if let Some(chronicle) = chronicles.iter_mut().find(|c| c.character_id == *cid) {
                    chronicle.append(entry);
                }
            }
        }

        // Auto-create chronicles for newly born characters
        if let GameEvent::CharacterBorn { character_id, character_name, parent, turn, .. } = event {
            if !chronicles.iter().any(|c| c.character_id == *character_id) {
                let parent_name = ctx.character_name(*parent);
                let mut chronicle = CharacterChronicle::new(*character_id, character_name.clone());
                chronicle.append(ChronicleEntry {
                    turn: *turn,
                    prose: format!(
                        "{} was born, child of {}. A new thread woven into \
                         the tapestry of the realm.",
                        character_name, parent_name
                    ),
                    tag: "birth".into(),
                });
                chronicles.push(chronicle);
            }
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ctx() -> WorldContext {
        WorldContext {
            province_names: vec![(0, "Frosthold".into()), (7, "Ashenmere".into())],
            faction_names: vec![(0, "Ashen Crown".into()), (1, "Vale Princes".into())],
            character_names: vec![
                (1, "King Aldric".into()),
                (2, "Queen Isolde".into()),
                (5, "Duke Varen".into()),
            ],
            faction_cultures: vec![],
            army_factions: vec![],
            current_turn: 50,
        }
    }

    #[test]
    fn birth_creates_chronicle_entry() {
        let ctx = test_ctx();
        let event = GameEvent::CharacterBorn {
            character_id: 10,
            character_name: "Prince Edwin".into(),
            parent: 1,
            dynasty: 0,
            turn: 50,
        };

        let entry = extract_for_character(10, "Prince Edwin", &event, &ctx);
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.tag, "birth");
        assert!(entry.prose.contains("King Aldric"));
    }

    #[test]
    fn death_creates_chronicle_entry() {
        let ctx = test_ctx();
        let event = GameEvent::CharacterDied {
            character_id: 1,
            character_name: "King Aldric".into(),
            cause: DeathCause::Assassination,
            turn: 55,
        };

        let entry = extract_for_character(1, "King Aldric", &event, &ctx);
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.tag, "death");
        assert!(entry.prose.contains("murdered"));
    }

    #[test]
    fn unrelated_character_gets_no_entry() {
        let ctx = test_ctx();
        let event = GameEvent::CharacterDied {
            character_id: 1,
            character_name: "King Aldric".into(),
            cause: DeathCause::OldAge,
            turn: 55,
        };

        let entry = extract_for_character(99, "Nobody", &event, &ctx);
        assert!(entry.is_none());
    }

    #[test]
    fn succession_claimant_gets_entry() {
        let ctx = test_ctx();
        let event = GameEvent::SuccessionCrisis {
            faction: 0,
            dead_ruler: 1,
            claimants: vec![2, 5],
            realm_split: false,
            turn: 60,
        };

        // Claimant should get an entry
        let entry = extract_for_character(2, "Queen Isolde", &event, &ctx);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().tag, "succession");

        // Dead ruler should also get an entry
        let entry = extract_for_character(1, "King Aldric", &event, &ctx);
        assert!(entry.is_some());
    }

    #[test]
    fn chronicle_render_full() {
        let mut chronicle = CharacterChronicle::new(1, "King Aldric".into());
        chronicle.append(ChronicleEntry {
            turn: 1,
            prose: "Born to rule.".into(),
            tag: "birth".into(),
        });
        chronicle.append(ChronicleEntry {
            turn: 50,
            prose: "Fell in battle.".into(),
            tag: "death".into(),
        });

        let text = chronicle.render_full();
        assert!(text.contains("Chronicle of King Aldric"));
        assert!(text.contains("Born to rule"));
        assert!(text.contains("Fell in battle"));
    }

    #[test]
    fn empty_chronicle_renders_placeholder() {
        let chronicle = CharacterChronicle::new(1, "King Aldric".into());
        let text = chronicle.render_full();
        assert!(text.contains("yet unwritten"));
    }

    #[test]
    fn update_chronicles_auto_creates_for_birth() {
        let ctx = test_ctx();
        let mut chronicles: Vec<CharacterChronicle> = Vec::new();

        let events = vec![GameEvent::CharacterBorn {
            character_id: 10,
            character_name: "Prince Edwin".into(),
            parent: 1,
            dynasty: 0,
            turn: 50,
        }];

        update_chronicles(&mut chronicles, &events, &ctx);

        assert_eq!(chronicles.len(), 1);
        assert_eq!(chronicles[0].character_id, 10);
        assert_eq!(chronicles[0].entries.len(), 1);
    }

    #[test]
    fn plot_entries_for_both_parties() {
        let ctx = test_ctx();
        let event = GameEvent::PlotLaunched {
            instigator: 5,
            target: 1,
            plot_type: "Assassination".into(),
            turn: 45,
        };

        let ins_entry = extract_for_character(5, "Duke Varen", &event, &ctx);
        assert!(ins_entry.is_some());
        assert!(ins_entry.unwrap().prose.contains("plotting"));

        let tgt_entry = extract_for_character(1, "King Aldric", &event, &ctx);
        assert!(tgt_entry.is_some());
        assert!(tgt_entry.unwrap().prose.contains("plotting against"));
    }
}
