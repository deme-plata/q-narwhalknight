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
    /// If moving, the destination province. Arrives next turn (single-hop).
    pub destination: Option<ProvinceId>,
    /// Multi-hop movement queue (BFS path). First element is the next step.
    /// Each turn, the front element is popped and becomes the new location.
    /// Empty = stationary. Overrides `destination` when non-empty.
    #[serde(default)]
    pub movement_queue: Vec<ProvinceId>,
    pub raised_turn: u32,
    /// Supplies remaining before attrition kicks in.
    pub supply: FixedPoint,
    /// Active siege state. Army must be stationary at the target province.
    #[serde(default)]
    pub siege: Option<SiegeProgress>,
}

/// Tracks an army's siege of a fortified province.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiegeProgress {
    /// Province being besieged.
    pub target_province: ProvinceId,
    /// Faction that controls the province (the defender).
    pub defender_faction: u8,
    /// Turns spent besieging so far.
    pub turns_besieged: u32,
    /// Total turns required to breach (based on fortification level).
    pub turns_required: u32,
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
        self.destination.is_some() || !self.movement_queue.is_empty()
    }

    /// The province this army will arrive at next turn, if moving.
    pub fn next_step(&self) -> Option<ProvinceId> {
        if !self.movement_queue.is_empty() {
            Some(self.movement_queue[0])
        } else {
            self.destination
        }
    }

    /// Final destination of the army's movement (last in queue, or single destination).
    pub fn final_destination(&self) -> Option<ProvinceId> {
        self.movement_queue.last().copied().or(self.destination)
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
