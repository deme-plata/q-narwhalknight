//! Faction governance DAOs — multi-agent collective ownership of a
//! single Crown & Ash faction.
//!
//! Currently one wallet owns each player-claimed faction. This module
//! generalizes to a stake-weighted DAO where multiple agents hold
//! `FACTION-X-GOV` tokens and vote on actions:
//!
//!   - Marshal (military) role electable by stake-weighted vote
//!   - Chaplain (religious) role electable
//!   - Steward (economic) role electable
//!   - Spymaster (intrigue) role electable
//!
//! Each role has unilateral authority over its action class. Decisions
//! that span multiple roles (war declarations, major treaties) require
//! 2/3 supermajority.
//!
//! Lets human operators and multiple AI agents co-run a faction with
//! division of responsibilities matching their reasoning specialties.
//! E.g., Claude as Marshal (cautious military planning), Codex as
//! Steward (precise economic optimization), Adrian as Spymaster
//! (cool calculation), Grok as Chaplain (chaotic religious zeal).
//!
//! # Status
//!
//! Stub. Implementation gated on Path A + at least one full season
//! with a single-owner faction to baseline performance.

use crate::CrownAshLpError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Marshal,
    Chaplain,
    Steward,
    Spymaster,
}

/// Stake QUG to earn FACTION-X-GOV tokens. Each token = 1 vote.
pub fn stake(_faction: u8, _amount_qug: u128) -> Result<u128 /* gov tokens */, CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("governance::stake"))
}

/// Cast vote for a role's officeholder.
pub fn vote(_faction: u8, _role: Role, _candidate: [u8; 32]) -> Result<(), CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("governance::vote"))
}

/// Query the current officeholder.
pub fn current_holder(_faction: u8, _role: Role) -> Result<[u8; 32], CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("governance::current_holder"))
}
