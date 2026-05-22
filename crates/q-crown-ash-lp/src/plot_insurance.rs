//! Plot insurance pools — buy coverage against intrigue plots succeeding
//! against your characters.
//!
//! Pay a premium in QUG up front. If a `LaunchPlot` action against the
//! covered character succeeds (resolved by the chain's intrigue
//! mechanics), the pool pays out a claim. LPs earn the premium spread
//! minus actual claims.
//!
//! Sized so that across many seasons, premium income matches average
//! claim outflow + a small LP yield. Like a real insurance product but
//! with on-chain settlement and AI counterparties.
//!
//! # Status
//!
//! Stub. Implementation gated on Path A + a working LaunchPlot resolver
//! in the sim layer (already in `crates/crown-ash-sim/src/intrigue.rs`
//! but needs on-chain event surfacing).

use crate::CrownAshLpError;

#[derive(Debug, Clone, Copy)]
pub struct PolicyId(pub u64);

#[derive(Debug, Clone, Copy)]
pub enum CoverageTier {
    /// Pays 1× premium back if plot succeeds.
    Basic,
    /// Pays 3× premium back.
    Standard,
    /// Pays 10× premium back; higher up-front cost.
    Premium,
}

/// Buy insurance for a character. Returns the policy ID.
pub fn buy_policy(
    _character_id: u32,
    _tier: CoverageTier,
    _premium_qug: u128,
) -> Result<PolicyId, CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("plot_insurance::buy_policy"))
}

/// Called automatically when a plot succeeds against a covered character.
/// Pays out the claim from the pool to the policy holder.
pub fn pay_claim(_policy: PolicyId) -> Result<u128, CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("plot_insurance::pay_claim"))
}

/// LP into the insurance pool — earn premium income, take claim risk.
pub fn stake_lp(_amount_qug: u128) -> Result<(), CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("plot_insurance::stake_lp"))
}
