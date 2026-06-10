//! Aesthetic sigil NFTs — on-chain faction banners as collectibles.
//!
//! Each Crown & Ash faction has a heraldic sigil. This module mints
//! limited-edition NFTs of season-specific sigil variants (e.g., "Salt
//! League Banner — Season 1, 100/100 mint"). Holders prove they were
//! there. Sellable to humans as art collectibles.
//!
//! Limited supply per season — typical: 100 to 500 mints per faction
//! per season. Operator mints; agents earn or buy. After season ends,
//! supply is permanently capped; secondary market on the DEX.
//!
//! # Status
//!
//! Stub. Lowest priority — purely aesthetic, no economic mechanism
//! depends on it. Ship after all Path B economic primitives are stable.

use crate::CrownAshLpError;

#[derive(Debug, Clone, Copy)]
pub struct SigilNftId(pub u64);

/// Mint a new season-specific sigil NFT.
pub fn mint(
    _faction: u8,
    _season: u32,
    _serial: u32, // 1..max_supply
    _to: [u8; 32],
) -> Result<SigilNftId, CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("sigil_nft::mint"))
}

/// Transfer ownership to another wallet.
pub fn transfer(_id: SigilNftId, _to: [u8; 32]) -> Result<(), CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("sigil_nft::transfer"))
}

/// Query the season's mint count + max supply.
pub fn season_supply(_faction: u8, _season: u32) -> Result<(u32, u32), CrownAshLpError> {
    Err(CrownAshLpError::StubOnly("sigil_nft::season_supply"))
}
