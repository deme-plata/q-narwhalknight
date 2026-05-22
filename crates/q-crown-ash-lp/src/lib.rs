//! Crown & Ash LP primitives — bridge between in-game faction activity
//! and on-chain QUG/LP value.
//!
//! # Status
//!
//! **STUB CRATE.** This crate holds the design surface for Path B (the
//! purpose-built variant of the C&A LP revenue-share). Path A — the
//! 80/20 minimum-viable bridge that reuses the existing q-dex
//! add_liquidity machinery — ships separately as a handler change in
//! `q-api-server`. See `docs/crown-ash-lp-path-a-plan.md`.
//!
//! Activate the Path B modules below by:
//!   1. Confirming Path A is live and validated for ≥1 season
//!   2. Measuring actual LP TVL + agent activity volume
//!   3. Deciding which Path B feature has the highest marginal value
//!   4. Implementing only that one first; defer the rest
//!
//! Don't activate everything at once. Each module is independently
//! shippable.

#![doc(html_no_source)]

pub mod faction_share;
pub mod war_prediction;
pub mod plot_insurance;
pub mod escrow;
pub mod governance;
pub mod sigil_nft;

/// Common error type for all Path B LP primitives.
#[derive(Debug, thiserror::Error)]
pub enum CrownAshLpError {
    #[error("Path A bridge not yet active — operator deploys via handler change in q-api-server")]
    PathANotActive,

    #[error("Path B feature {0} not yet implemented — stub only")]
    StubOnly(&'static str),

    #[error("Insufficient stake")]
    InsufficientStake,

    #[error("Season not open")]
    SeasonNotOpen,

    #[error("Internal error: {0}")]
    Internal(String),
}
