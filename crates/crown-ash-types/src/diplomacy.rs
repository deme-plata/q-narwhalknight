//! Diplomacy — relations, treaties, and casus belli between factions.

use serde::{Deserialize, Serialize};
use crate::fixed_point::FixedPoint;

/// Diplomatic standing between two factions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiplomaticRelation {
    pub faction_a: u8,
    pub faction_b: u8,
    /// Opinion score: -1000 (mortal enemies) to +1000 (blood brothers).
    pub opinion: FixedPoint,
    pub at_war: bool,
    pub treaties: Vec<ActiveTreaty>,
    /// Recent offenses that decay over time.
    pub grievances: Vec<Grievance>,
}

impl DiplomaticRelation {
    pub fn new(a: u8, b: u8) -> Self {
        Self {
            faction_a: a,
            faction_b: b,
            opinion: FixedPoint::ZERO,
            at_war: false,
            treaties: Vec::new(),
            grievances: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTreaty {
    pub treaty_type: String,
    pub signed_turn: u32,
    /// Some treaties expire.
    pub expires_turn: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grievance {
    pub reason: String,
    pub opinion_modifier: FixedPoint,
    pub inflicted_turn: u32,
    /// Turns until this grievance expires.
    pub decay_turns_remaining: u32,
}
