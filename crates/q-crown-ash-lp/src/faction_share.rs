//! Faction-share index tokens (FACTION-SALT, FACTION-CROWN, etc.)
//!
//! Each Crown & Ash faction gets an on-chain ERC-20-style token whose
//! supply tracks the faction's "fundamental value" — a function of
//! (treasury + provinces × constant + military_strength + legitimacy).
//!
//! LPs can hold the token, providing liquidity in pools like
//! `QUG/FACTION-SALT`. When the faction prospers, the underlying value
//! grows and the token price appreciates. LPs earn DEX trading fees
//! from speculators betting on faction performance.
//!
//! # Status
//!
//! Stub. Implementation gated on Path A validation + at least one full
//! season of LP TVL data so we can size the mint+burn schedule.

use crate::CrownAshLpError;

/// Identifier for a faction's index token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FactionShareId(pub u8); // 0..6 — matches GameAction faction IDs

/// Compute the current "fundamental value" of a faction in QUG-equivalent terms.
///
/// Formula (proposed):
/// ```text
/// fundamental = treasury_qug
///             + (provinces * BASELINE_PROVINCE_VALUE_QUG)
///             + (military_score * MILITARY_QUG_MULTIPLIER)
///             + (legitimacy * LEGITIMACY_QUG_MULTIPLIER)
/// ```
///
/// Constants are tunable per season.
pub fn fundamental_value_qug(_faction: FactionShareId) -> Result<u128, CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("faction_share::fundamental_value_qug"))
}

/// Mint additional FACTION-X tokens when the faction's fundamental value
/// grows. Called automatically at season-tick by the protocol; not
/// callable by users directly.
pub fn mint_on_growth(_faction: FactionShareId, _delta_value: u128) -> Result<(), CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("faction_share::mint_on_growth"))
}

/// Burn FACTION-X tokens proportional to faction's loss of value.
/// Called automatically when a faction loses provinces or treasury.
pub fn burn_on_loss(_faction: FactionShareId, _delta_value: u128) -> Result<(), CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("faction_share::burn_on_loss"))
}
