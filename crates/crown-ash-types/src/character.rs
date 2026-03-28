//! Character — rulers, heirs, councilors, and nobles.

use serde::{Deserialize, Serialize};
use crate::fixed_point::FixedPoint;

// ---------------------------------------------------------------------------
// Plot types — used by the intrigue system
// ---------------------------------------------------------------------------

/// The type of an intrigue plot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlotType {
    /// Kill the target character.
    Assassination,
    /// Fabricate a claim on one of the target's provinces.
    Fabricate,
    /// Reduce prosperity of one of the target's provinces.
    Sabotage,
    /// Steal gold from the target faction's treasury.
    Steal,
}

impl PlotType {
    /// Human-readable label for events.
    pub fn label(&self) -> &'static str {
        match self {
            PlotType::Assassination => "Assassination",
            PlotType::Fabricate => "Fabricate Claim",
            PlotType::Sabotage => "Sabotage",
            PlotType::Steal => "Steal Gold",
        }
    }
}

/// An active intrigue plot against a character.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plot {
    /// Unique plot identifier (monotonically increasing).
    pub id: u32,
    /// What kind of plot this is.
    pub plot_type: PlotType,
    /// The character who started the plot.
    pub instigator: CharacterId,
    /// The character being targeted.
    pub target: CharacterId,
    /// Characters backing / supporting the plot.
    pub backers: Vec<CharacterId>,
    /// Progress toward execution — fires at 1000 (FixedPoint integer scale).
    pub progress: FixedPoint,
    /// Secrecy level — higher means harder to detect. Decays over time.
    pub secrecy: FixedPoint,
    /// The turn the plot was launched.
    pub started_turn: u32,
}

pub type CharacterId = u32;
pub type DynastyId = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CharacterRole {
    Ruler,
    Heir,
    Duke,
    Marshal,     // Military advisor
    Chaplain,    // Religious advisor
    Steward,     // Economic advisor
    Spymaster,   // Intrigue advisor
    Courtier,    // No official role
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Trait {
    // Virtues
    Brave,
    Just,
    Pious,
    Temperate,
    Kind,
    Diligent,
    Patient,
    Honest,
    // Vices
    Craven,
    Cruel,
    Cynical,
    Gluttonous,
    Wrathful,
    Slothful,
    Impatient,
    Deceitful,
    // Personality
    Ambitious,
    Content,
    Gregarious,
    Shy,
    Paranoid,
    Trusting,
    // Skills
    Strategist,
    Administrator,
    Theologian,
    Schemer,
    Scholar,
}

impl Trait {
    /// Modifier to martial stat (×1000).
    pub fn martial_mod(&self) -> i64 {
        match self {
            Trait::Brave => 3000,
            Trait::Strategist => 5000,
            Trait::Craven => -3000,
            Trait::Wrathful => 2000,
            _ => 0,
        }
    }

    /// Modifier to diplomacy stat.
    pub fn diplomacy_mod(&self) -> i64 {
        match self {
            Trait::Just => 2000,
            Trait::Kind => 3000,
            Trait::Gregarious => 4000,
            Trait::Cruel => -3000,
            Trait::Shy => -2000,
            Trait::Honest => 1000,
            _ => 0,
        }
    }

    /// Modifier to stewardship stat.
    pub fn stewardship_mod(&self) -> i64 {
        match self {
            Trait::Diligent => 3000,
            Trait::Administrator => 5000,
            Trait::Temperate => 2000,
            Trait::Gluttonous => -2000,
            Trait::Slothful => -3000,
            _ => 0,
        }
    }

    /// Modifier to intrigue stat.
    pub fn intrigue_mod(&self) -> i64 {
        match self {
            Trait::Schemer => 5000,
            Trait::Deceitful => 3000,
            Trait::Paranoid => 2000,
            Trait::Trusting => -3000,
            Trait::Honest => -2000,
            _ => 0,
        }
    }

    /// Modifier to learning stat.
    pub fn learning_mod(&self) -> i64 {
        match self {
            Trait::Scholar => 5000,
            Trait::Theologian => 3000,
            Trait::Patient => 2000,
            Trait::Impatient => -2000,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CharacterStats {
    pub martial: FixedPoint,
    pub diplomacy: FixedPoint,
    pub stewardship: FixedPoint,
    pub intrigue: FixedPoint,
    pub learning: FixedPoint,
}

impl CharacterStats {
    /// Compute effective stats with trait modifiers applied.
    pub fn effective(&self, traits: &[Trait]) -> Self {
        let mut result = self.clone();
        for t in traits {
            result.martial += FixedPoint::from_raw(t.martial_mod());
            result.diplomacy += FixedPoint::from_raw(t.diplomacy_mod());
            result.stewardship += FixedPoint::from_raw(t.stewardship_mod());
            result.intrigue += FixedPoint::from_raw(t.intrigue_mod());
            result.learning += FixedPoint::from_raw(t.learning_mod());
        }
        result
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Character {
    pub id: CharacterId,
    pub name: String,
    pub dynasty: DynastyId,
    pub faction: u8,
    pub role: CharacterRole,
    pub age: u8,
    pub alive: bool,
    pub traits: Vec<Trait>,
    pub stats: CharacterStats,
    pub health: FixedPoint,
    pub legitimacy: FixedPoint,
    pub prestige: FixedPoint,
    pub heir: Option<CharacterId>,
    pub spouse: Option<CharacterId>,
    pub children: Vec<CharacterId>,
    pub parent: Option<CharacterId>,
    /// Turn on which this character died (set when alive becomes false).
    #[serde(default)]
    pub death_turn: Option<u32>,
    /// Cause of death, if dead.
    #[serde(default)]
    pub death_cause: Option<String>,
}

impl Character {
    /// Is this character of ruling age? (>= 16)
    pub fn is_adult(&self) -> bool {
        self.age >= 16
    }

    /// Effective stats including trait bonuses.
    pub fn effective_stats(&self) -> CharacterStats {
        self.stats.effective(&self.traits)
    }
}

/// Compact record of a dead character for historical reference.
///
/// When a character dies and the grace period elapses, their full `Character`
/// struct is removed from the world and replaced by a lightweight tombstone.
/// This prevents unbounded growth of the character list over long games.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterTombstone {
    pub id: CharacterId,
    pub name: String,
    pub dynasty: DynastyId,
    pub faction: u8,
    pub cause_of_death: String,
    pub death_turn: u32,
    pub prestige: FixedPoint,
    pub age_at_death: u8,
    pub was_ruler: bool,
}
