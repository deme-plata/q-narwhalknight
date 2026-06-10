//! Realm — a player's domain, anchored to their QNK wallet.

use serde::{Deserialize, Serialize};
use crate::fixed_point::FixedPoint;
use crate::province::ProvinceId;

/// The 5 components of realm cohesion — the core "holding it together" mechanic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealmCohesion {
    /// Ruler's perceived right to rule (blood, conquest, election).
    pub legitimacy: FixedPoint,
    /// Vassal lords' loyalty to the crown.
    pub fealty: FixedPoint,
    /// Church/religious institution support.
    pub clerical_favor: FixedPoint,
    /// Common people's happiness (taxation, war weariness, prosperity).
    pub commoner_mood: FixedPoint,
    /// How unified the realm's provinces feel culturally.
    pub regional_identity: FixedPoint,
}

impl Default for RealmCohesion {
    fn default() -> Self {
        Self {
            legitimacy: FixedPoint::from_int(500),
            fealty: FixedPoint::from_int(500),
            clerical_favor: FixedPoint::from_int(500),
            commoner_mood: FixedPoint::from_int(500),
            regional_identity: FixedPoint::from_int(500),
        }
    }
}

impl RealmCohesion {
    /// Average of all 5 components (0-1000 range).
    pub fn average(&self) -> FixedPoint {
        let sum = self.legitimacy.raw()
            + self.fealty.raw()
            + self.clerical_favor.raw()
            + self.commoner_mood.raw()
            + self.regional_identity.raw();
        FixedPoint::from_raw(sum / 5)
    }

    /// Is the realm in danger of fracturing? (any component below 200)
    pub fn is_critical(&self) -> bool {
        let threshold = FixedPoint::from_int(200);
        self.legitimacy < threshold
            || self.fealty < threshold
            || self.clerical_favor < threshold
            || self.commoner_mood < threshold
            || self.regional_identity < threshold
    }

    /// Apply uniform decay toward equilibrium (500).
    pub fn decay_toward_equilibrium(&mut self, rate: FixedPoint) {
        let eq = FixedPoint::from_int(500);
        self.legitimacy = decay_component(self.legitimacy, eq, rate);
        self.fealty = decay_component(self.fealty, eq, rate);
        self.clerical_favor = decay_component(self.clerical_favor, eq, rate);
        self.commoner_mood = decay_component(self.commoner_mood, eq, rate);
        self.regional_identity = decay_component(self.regional_identity, eq, rate);
    }

    /// Clamp all components to 0-1000 range.
    pub fn clamp_all(&mut self) {
        let lo = FixedPoint::ZERO;
        let hi = FixedPoint::from_int(1000);
        self.legitimacy = self.legitimacy.clamp(lo, hi);
        self.fealty = self.fealty.clamp(lo, hi);
        self.clerical_favor = self.clerical_favor.clamp(lo, hi);
        self.commoner_mood = self.commoner_mood.clamp(lo, hi);
        self.regional_identity = self.regional_identity.clamp(lo, hi);
    }
}

fn decay_component(current: FixedPoint, equilibrium: FixedPoint, rate: FixedPoint) -> FixedPoint {
    let diff = equilibrium - current;
    current + diff.mul_fp(rate)
}

/// A player's realm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Realm {
    /// QNK wallet address of the player who owns this realm.
    pub owner_wallet: String,
    pub faction: u8,
    pub ruler: u32,
    pub provinces: Vec<ProvinceId>,
    pub vassals: Vec<u8>,
    pub treasury: FixedPoint,
    pub cohesion: RealmCohesion,
    /// Turns this realm has existed.
    pub age: u32,
    /// Active wars (enemy faction IDs).
    pub at_war_with: Vec<u8>,
    /// Active treaties (ally faction IDs).
    pub allies: Vec<u8>,
    /// Religious authority: how dominant this faction's religion is within its realm (0-1000).
    /// Higher authority = faster conversions, resistance to heresy, clerical_favor bonus.
    #[serde(default = "default_authority")]
    pub religious_authority: FixedPoint,
}

fn default_authority() -> FixedPoint {
    FixedPoint::from_int(500)
}
