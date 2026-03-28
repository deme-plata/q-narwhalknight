//! Army — military units that move across the campaign map.

use serde::{Deserialize, Serialize};
use crate::fixed_point::FixedPoint;
use crate::province::{ProvinceId, Troops};

pub type ArmyId = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Army {
    pub id: ArmyId,
    pub owner_faction: u8,
    pub commander: Option<u32>,
    pub troops: Troops,
    pub morale: FixedPoint,
    pub location: ProvinceId,
    /// If moving, the destination province. Arrives next turn.
    pub destination: Option<ProvinceId>,
    pub raised_turn: u32,
    /// Supplies remaining before attrition kicks in.
    pub supply: FixedPoint,
}

impl Army {
    /// Combat power calculation (all fixed-point).
    pub fn attack_power(&self) -> FixedPoint {
        let levy = FixedPoint::from_int(self.troops.levy as i64);
        let maa = FixedPoint::from_int(self.troops.men_at_arms as i64) * 3;
        let knights = FixedPoint::from_int(self.troops.knights as i64) * 10;
        levy + maa + knights
    }

    pub fn is_moving(&self) -> bool {
        self.destination.is_some()
    }

    pub fn total_soldiers(&self) -> u32 {
        self.troops.total()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BattleResult {
    pub attacker_army: ArmyId,
    pub defender_army: Option<ArmyId>,
    pub province: ProvinceId,
    pub attacker_casualties: u32,
    pub defender_casualties: u32,
    pub attacker_won: bool,
    /// Random factor used (850-1150).
    pub random_factor: FixedPoint,
    pub turn: u32,
}
