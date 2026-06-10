//! War prediction pools — bet on whether an aggressor wins a war.
//!
//! Each `DeclareWar` action automatically opens a `WAR-<aggressor>-
//! <defender>-<deadline>` pool. Speculators stake on either side.
//! Outcome resolves automatically when the war ends (peace, surrender,
//! or vassal-ization) — the chain's own consensus is the oracle.
//!
//! LPs in these pools earn the spread between bet-side stakes vs the
//! actual outcome. Losing-side stake is redistributed to winning-side
//! stakeholders pro-rata, minus a small operator/burn slice.
//!
//! # Status
//!
//! Stub. Implementation gated on Path A validation. The mechanic
//! itself is straightforward but the UX (how speculators see open
//! wars, current odds, expected return) needs the wallet UI work too.

use crate::CrownAshLpError;

/// Identifier for a war (matches GameState wars vec index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WarId(pub u32);

/// Side of the war.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarSide {
    Aggressor,
    Defender,
}

/// Outcome of a resolved war.
#[derive(Debug, Clone, Copy)]
pub enum WarOutcome {
    AggressorWins,
    DefenderWins,
    WhitePeace,    // Both sides survive, stalemate
    Vassalization, // One side becomes vassal of the other
}

/// Stake QUG on a war's outcome. Locks until the war resolves.
pub fn stake(_war: WarId, _side: WarSide, _amount_qug: u128) -> Result<(), CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("war_prediction::stake"))
}

/// Called automatically when a war resolves. Distributes pool to
/// winning-side stakeholders. Returns the QUG burned + operator fee.
pub fn resolve(_war: WarId, _outcome: WarOutcome) -> Result<(u128, u128), CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("war_prediction::resolve"))
}

/// Current odds (each side's pool TVL as basis points).
pub fn odds(_war: WarId) -> Result<(u32, u32), CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("war_prediction::odds"))
}
