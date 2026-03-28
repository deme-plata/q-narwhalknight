//! Economy — resources, trade routes, and market mechanics.

use serde::{Deserialize, Serialize};
use crate::fixed_point::FixedPoint;
use crate::province::ProvinceId;

/// Goods exchanged along a trade route, determined by province improvements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradeGood {
    /// Farmstead provinces export grain.
    Grain,
    /// Mine provinces export iron.
    Iron,
    /// Lumbercamp provinces export timber.
    Timber,
    /// Quarry provinces export stone.
    Stone,
    /// Stables provinces export horses.
    Horses,
    /// Market provinces export luxuries.
    Luxuries,
    /// Temple provinces generate pilgrimage traffic.
    Faith,
}

impl TradeGood {
    /// Human-readable label for events.
    pub fn label(&self) -> &'static str {
        match self {
            TradeGood::Grain => "Grain",
            TradeGood::Iron => "Iron",
            TradeGood::Timber => "Timber",
            TradeGood::Stone => "Stone",
            TradeGood::Horses => "Horses",
            TradeGood::Luxuries => "Luxuries",
            TradeGood::Faith => "Faith",
        }
    }
}

/// A trade route between two provinces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRoute {
    /// Unique route identifier.
    pub id: u32,
    /// First endpoint province.
    pub province_a: ProvinceId,
    /// Second endpoint province.
    pub province_b: ProvinceId,
    /// Primary good exchanged on this route.
    pub goods: TradeGood,
    /// Trade volume (0-1000 in FixedPoint). Grows when peaceful, decays when disrupted.
    pub volume: FixedPoint,
    /// Turn this route was established.
    pub established_turn: u32,
    /// True if the route is disrupted (provinces at war).
    pub disrupted: bool,
}
