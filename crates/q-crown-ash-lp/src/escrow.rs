//! Inter-agent escrow — conditional QUG transfers triggered by game state.
//!
//! Example: Agent A deposits 100 QUG into escrow with the condition
//! "release to Agent B if their Black Abbey faction holds Lakeshore
//! province for ≥100 turns from now." The chain itself adjudicates
//! when the condition fires (read province controller at turn N+100,
//! compare to required controller).
//!
//! Trustless — no human or oracle intermediary. Composable with treaty
//! mechanics: e.g., "release iff you sign DefensiveAlliance with me
//! within 10 turns AND honor it for 1000 turns."
//!
//! # Status
//!
//! Stub. Implementation gated on Path A. Then designs the condition
//! grammar — small DSL over GameState predicates, evaluated each
//! turn-tick on each open escrow.

use crate::CrownAshLpError;

#[derive(Debug, Clone, Copy)]
pub struct EscrowId(pub u64);

/// Condition that fires the escrow.
#[derive(Debug, Clone)]
pub enum EscrowCondition {
    /// Faction F holds province P at turn T.
    ProvinceHeld { faction: u8, province: u32, by_turn: u64 },
    /// Active treaty between A and B at turn T.
    TreatyActive { a: u8, b: u8, by_turn: u64 },
    /// Faction F's army count ≥ N at turn T.
    ArmyCount { faction: u8, min: u32, by_turn: u64 },
    /// Logical AND of multiple conditions.
    AllOf(Vec<EscrowCondition>),
    /// Logical OR of multiple conditions.
    AnyOf(Vec<EscrowCondition>),
}

/// Lock QUG in escrow. Returns the escrow ID.
pub fn create(
    _from: [u8; 32],
    _to: [u8; 32],
    _amount_qug: u128,
    _condition: EscrowCondition,
    _expires_at_turn: u64,
) -> Result<EscrowId, CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("escrow::create"))
}

/// Called automatically each turn-tick on each open escrow. If the
/// condition fires, transfers the locked QUG to recipient. If expiry
/// reached without firing, refunds to sender.
pub fn tick(_id: EscrowId) -> Result<bool, CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("escrow::tick"))
}
