//! Dynasty and succession rules.

use serde::{Deserialize, Serialize};

pub type DynastyId = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuccessionRule {
    /// First-born child inherits (male-preference).
    Primogeniture,
    /// Elective — council votes among eligible claimants.
    Elective,
    /// Strongest warrior claims the throne.
    TrialByCombat,
    /// Eldest living member of dynasty.
    Seniority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dynasty {
    pub id: DynastyId,
    pub name: String,
    pub founder: u32,
    pub succession_rule: SuccessionRule,
    pub prestige: i64,
    /// All living members (character IDs).
    pub members: Vec<u32>,
    pub founded_turn: u32,
}
