//! Province — the fundamental territorial unit on the campaign map.

use serde::{Deserialize, Serialize};
use crate::fixed_point::FixedPoint;

pub type ProvinceId = u16;
pub type FactionId = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Terrain {
    Plains,
    Hills,
    Mountains,
    Forest,
    Marsh,
    Desert,
    Coastal,
    River,
}

impl Terrain {
    /// Defense bonus multiplier (scaled ×1000).
    pub fn defense_bonus(&self) -> FixedPoint {
        match self {
            Terrain::Plains => FixedPoint::from_raw(0),
            Terrain::Hills => FixedPoint::from_raw(200),      // +20%
            Terrain::Mountains => FixedPoint::from_raw(400),   // +40%
            Terrain::Forest => FixedPoint::from_raw(250),      // +25%
            Terrain::Marsh => FixedPoint::from_raw(300),       // +30%
            Terrain::Desert => FixedPoint::from_raw(100),      // +10%
            Terrain::Coastal => FixedPoint::from_raw(150),     // +15%
            Terrain::River => FixedPoint::from_raw(200),       // +20%
        }
    }

    /// Movement cost multiplier (1000 = normal).
    pub fn movement_cost(&self) -> FixedPoint {
        match self {
            Terrain::Plains => FixedPoint::ONE,
            Terrain::Hills => FixedPoint::from_raw(1500),
            Terrain::Mountains => FixedPoint::from_raw(2000),
            Terrain::Forest => FixedPoint::from_raw(1300),
            Terrain::Marsh => FixedPoint::from_raw(1800),
            Terrain::Desert => FixedPoint::from_raw(1400),
            Terrain::Coastal => FixedPoint::ONE,
            Terrain::River => FixedPoint::from_raw(1200),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Religion {
    OldFaith,       // Pagan forest religion
    EmberChurch,    // Dominant organized religion
    SaltCult,       // Merchant trade-god
    FrostSpirits,   // Northern animism
    BlackOrder,     // Monastic secret sect
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Culture {
    Imperial,    // Ashen Crown heartland
    Feudal,      // Vale Princes chivalric
    Clerical,    // Ember Church theocratic
    Mercantile,  // Salt League trading
    Nordic,      // Frost Marches hardy
    Nomadic,     // Red Steppe horsemen
    Monastic,    // Black Abbey secretive
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Resources {
    pub food: FixedPoint,
    pub gold: FixedPoint,
    pub iron: FixedPoint,
    pub timber: FixedPoint,
    pub stone: FixedPoint,
    pub horses: FixedPoint,
    pub trade_goods: FixedPoint,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Troops {
    pub levy: u32,
    pub men_at_arms: u32,
    pub knights: u16,
}

impl Troops {
    pub fn total(&self) -> u32 {
        self.levy + self.men_at_arms + self.knights as u32
    }
}

/// Permanent wound on a province from historical events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvinceScar {
    pub turn_inflicted: u32,
    pub scar_type: ScarType,
    /// Severity 0-1000 (decays over time).
    pub severity: FixedPoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScarType {
    WarDamage,
    Famine,
    Plague,
    ForcedConversion,
    Massacre,
    Pillage,
}

/// Long-memory grudge between province population and a faction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grudge {
    pub against_faction: FactionId,
    pub reason: ScarType,
    pub turn_caused: u32,
    /// Intensity 0-1000 (decays very slowly).
    pub intensity: FixedPoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Improvement {
    Farmstead,       // +food
    Mine,            // +iron, +gold
    Lumbercamp,      // +timber
    Quarry,          // +stone
    Stables,         // +horses
    Market,          // +trade_goods, +gold
    Temple,          // +clerical_favor
    Fortification,   // +defense
    University,      // +learning
    Port,            // +trade (coastal only)
    Granary,         // famine resistance
    Hospital,        // plague resistance
}

impl Improvement {
    pub fn build_cost_gold(&self) -> FixedPoint {
        match self {
            Improvement::Farmstead => FixedPoint::from_int(50),
            Improvement::Mine => FixedPoint::from_int(80),
            Improvement::Lumbercamp => FixedPoint::from_int(40),
            Improvement::Quarry => FixedPoint::from_int(60),
            Improvement::Stables => FixedPoint::from_int(70),
            Improvement::Market => FixedPoint::from_int(100),
            Improvement::Temple => FixedPoint::from_int(120),
            Improvement::Fortification => FixedPoint::from_int(150),
            Improvement::University => FixedPoint::from_int(200),
            Improvement::Port => FixedPoint::from_int(130),
            Improvement::Granary => FixedPoint::from_int(45),
            Improvement::Hospital => FixedPoint::from_int(90),
        }
    }

    pub fn build_turns(&self) -> u32 {
        match self {
            Improvement::Farmstead | Improvement::Lumbercamp | Improvement::Granary => 3,
            Improvement::Mine | Improvement::Quarry | Improvement::Stables => 5,
            Improvement::Market | Improvement::Temple | Improvement::Hospital => 7,
            Improvement::Port | Improvement::University => 10,
            Improvement::Fortification => 12,
        }
    }
}

/// A province on the campaign map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Province {
    pub id: ProvinceId,
    pub name: String,
    pub terrain: Terrain,
    pub controller: FactionId,
    pub population: u32,
    pub prosperity: FixedPoint,
    pub unrest: FixedPoint,
    pub fortification: u16,
    pub religion: Religion,
    pub culture: Culture,
    pub resources: Resources,
    pub garrison: Troops,
    pub improvements: Vec<Improvement>,
    /// Buildings under construction: (improvement, turns_remaining).
    pub construction_queue: Vec<(Improvement, u32)>,
    /// Permanent marks from war/famine/plague (capped at 20).
    pub scars: Vec<ProvinceScar>,
    /// Long-memory grudges against factions.
    pub grudges: Vec<Grudge>,
    pub last_famine_turn: Option<u32>,
    pub last_siege_turn: Option<u32>,
    /// Tax rate set by controller (0-1000 = 0%-100%).
    pub tax_rate: FixedPoint,
    /// Adjacency list (province IDs reachable in 1 move).
    pub neighbors: Vec<ProvinceId>,
    /// Active conversion: (target_religion, progress 0-1000). Completes when progress >= 1000.
    #[serde(default)]
    pub conversion_progress: Option<(Religion, FixedPoint)>,
}

impl Province {
    /// Cap scars to 20 entries, removing oldest.
    pub fn add_scar(&mut self, scar: ProvinceScar) {
        self.scars.push(scar);
        if self.scars.len() > 20 {
            self.scars.remove(0);
        }
    }
}
