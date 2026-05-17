//! Faction — the 7 playable factions of Crown & Ash.

use serde::{Deserialize, Serialize};
use crate::fixed_point::FixedPoint;
use crate::province::{Culture, Religion};

pub type FactionId = u8;

/// Template for creating the 7 starting factions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionTemplate {
    pub id: FactionId,
    pub name: String,
    pub motto: String,
    pub culture: Culture,
    pub religion: Religion,
    /// Starting province count.
    pub start_provinces: u8,
    /// Faction-specific bonuses (×1000 scale).
    pub bonuses: FactionBonuses,
    pub color_rgb: [u8; 3],
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactionBonuses {
    /// Military strength multiplier (1000 = neutral).
    pub military: FixedPoint,
    /// Economic income multiplier.
    pub economy: FixedPoint,
    /// Diplomacy effectiveness multiplier.
    pub diplomacy: FixedPoint,
    /// Intrigue power multiplier.
    pub intrigue: FixedPoint,
    /// Clerical influence multiplier.
    pub clerical: FixedPoint,
    /// Starting legitimacy bonus.
    pub legitimacy: FixedPoint,
    /// Cohesion decay rate modifier (lower = slower decay).
    pub cohesion_decay: FixedPoint,
}

/// Runtime faction state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Faction {
    pub id: FactionId,
    pub name: String,
    pub alive: bool,
    pub culture: Culture,
    pub religion: Religion,
    pub bonuses: FactionBonuses,
    pub color_rgb: [u8; 3],
    /// Player wallet (None if NPC).
    pub player_wallet: Option<String>,
}

/// Build all 7 faction templates.
pub fn default_faction_templates() -> Vec<FactionTemplate> {
    vec![
        FactionTemplate {
            id: 0,
            name: "Ashen Crown".into(),
            motto: "From ashes, authority.".into(),
            culture: Culture::Imperial,
            religion: Religion::EmberChurch,
            start_provinces: 4,
            bonuses: FactionBonuses {
                military: FixedPoint::from_raw(900),
                economy: FixedPoint::from_raw(1000),
                diplomacy: FixedPoint::from_raw(1200),
                intrigue: FixedPoint::from_raw(1000),
                clerical: FixedPoint::from_raw(1100),
                legitimacy: FixedPoint::from_raw(1500),
                cohesion_decay: FixedPoint::from_raw(900),
            },
            color_rgb: [180, 140, 60],
        },
        FactionTemplate {
            id: 1,
            name: "Vale Princes".into(),
            motto: "Gold sharpens every blade.".into(),
            culture: Culture::Feudal,
            religion: Religion::EmberChurch,
            start_provinces: 4,
            bonuses: FactionBonuses {
                military: FixedPoint::from_raw(1100),
                economy: FixedPoint::from_raw(1300),
                diplomacy: FixedPoint::from_raw(1000),
                intrigue: FixedPoint::from_raw(1000),
                clerical: FixedPoint::from_raw(900),
                legitimacy: FixedPoint::from_raw(1000),
                cohesion_decay: FixedPoint::from_raw(1200),
            },
            color_rgb: [60, 100, 180],
        },
        FactionTemplate {
            id: 2,
            name: "Ember Church".into(),
            motto: "The flame reveals all truth.".into(),
            culture: Culture::Clerical,
            religion: Religion::EmberChurch,
            start_provinces: 3,
            bonuses: FactionBonuses {
                military: FixedPoint::from_raw(800),
                economy: FixedPoint::from_raw(900),
                diplomacy: FixedPoint::from_raw(1100),
                intrigue: FixedPoint::from_raw(900),
                clerical: FixedPoint::from_raw(1500),
                legitimacy: FixedPoint::from_raw(1200),
                cohesion_decay: FixedPoint::from_raw(800),
            },
            color_rgb: [200, 60, 40],
        },
        FactionTemplate {
            id: 3,
            name: "Salt League".into(),
            motto: "Coin speaks louder than steel.".into(),
            culture: Culture::Mercantile,
            religion: Religion::SaltCult,
            start_provinces: 3,
            bonuses: FactionBonuses {
                military: FixedPoint::from_raw(800),
                economy: FixedPoint::from_raw(1500),
                diplomacy: FixedPoint::from_raw(1200),
                intrigue: FixedPoint::from_raw(1100),
                clerical: FixedPoint::from_raw(700),
                legitimacy: FixedPoint::from_raw(800),
                cohesion_decay: FixedPoint::from_raw(1100),
            },
            color_rgb: [220, 200, 160],
        },
        FactionTemplate {
            id: 4,
            name: "Frost Marches".into(),
            motto: "Winter hardens the worthy.".into(),
            culture: Culture::Nordic,
            religion: Religion::FrostSpirits,
            start_provinces: 4,
            bonuses: FactionBonuses {
                military: FixedPoint::from_raw(1400),
                economy: FixedPoint::from_raw(700),
                diplomacy: FixedPoint::from_raw(800),
                intrigue: FixedPoint::from_raw(900),
                clerical: FixedPoint::from_raw(800),
                legitimacy: FixedPoint::from_raw(1000),
                cohesion_decay: FixedPoint::from_raw(1000),
            },
            color_rgb: [140, 180, 220],
        },
        FactionTemplate {
            id: 5,
            name: "Red Steppe".into(),
            motto: "The horizon belongs to the swift.".into(),
            culture: Culture::Nomadic,
            religion: Religion::OldFaith,
            start_provinces: 4,
            bonuses: FactionBonuses {
                military: FixedPoint::from_raw(1300),
                economy: FixedPoint::from_raw(600),
                diplomacy: FixedPoint::from_raw(700),
                intrigue: FixedPoint::from_raw(1000),
                clerical: FixedPoint::from_raw(600),
                legitimacy: FixedPoint::from_raw(800),
                cohesion_decay: FixedPoint::from_raw(1300),
            },
            color_rgb: [180, 50, 30],
        },
        FactionTemplate {
            id: 6,
            name: "Black Abbey".into(),
            motto: "Silence is the sharpest weapon.".into(),
            culture: Culture::Monastic,
            religion: Religion::BlackOrder,
            start_provinces: 3,
            bonuses: FactionBonuses {
                military: FixedPoint::from_raw(900),
                economy: FixedPoint::from_raw(1000),
                diplomacy: FixedPoint::from_raw(800),
                intrigue: FixedPoint::from_raw(1500),
                clerical: FixedPoint::from_raw(1100),
                legitimacy: FixedPoint::from_raw(900),
                cohesion_decay: FixedPoint::from_raw(1000),
            },
            color_rgb: [40, 40, 50],
        },
    ]
}
