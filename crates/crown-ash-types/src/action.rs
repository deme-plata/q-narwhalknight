//! GameAction — player commands submitted as signed transactions.

use serde::{Deserialize, Serialize};
use crate::character::PlotType;
use crate::fixed_point::FixedPoint;
use crate::province::{Improvement, ProvinceId, Religion};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CasusBelli {
    Conquest,          // Generic territorial claim
    HolyWar,           // Religious difference
    Reconquest,        // Reclaim lost province
    Rebellion,         // Vassal uprising
    Succession,        // Disputed inheritance
    Insult,            // Diplomatic offense
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TreatyType {
    NonAggression,
    DefensiveAlliance,
    TradeAgreement,
    Marriage,          // Sealed with royal marriage
    Vassalization,     // Submit as vassal
    WhitePeace,        // End war with no demands
    Surrender,         // Full surrender with demands
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CouncilRole {
    Marshal,
    Chaplain,
    Steward,
    Spymaster,
}

/// Actions a player can submit (arrive as signed blockchain transactions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameAction {
    /// Raise troops from a province's population.
    RaiseArmy { province: ProvinceId },
    /// Move army to adjacent province (1 province per turn).
    MoveArmy { army: u32, target: ProvinceId },
    /// Move army along a multi-hop BFS path to a distant province.
    /// The server computes the shortest path; army advances 1 step per turn.
    MoveArmyPath { army: u32, target: ProvinceId },
    /// Disband army back to garrison.
    DisbandArmy { army: u32 },
    /// Declare war on another faction.
    DeclareWar { target: u8, casus_belli: CasusBelli },
    /// Propose treaty/peace to another faction.
    ProposeTreaty { target: u8, treaty: TreatyType },
    /// Accept a proposed treaty.
    AcceptTreaty { from: u8, treaty: TreatyType },
    /// Build an improvement in a province.
    BuildImprovement { province: ProvinceId, improvement: Improvement },
    /// Set tax rate (0-1000 = 0.000 to 1.000).
    SetTaxRate { province: ProvinceId, rate: FixedPoint },
    /// Assign a character to a council role.
    AssignCouncilor { character: u32, role: CouncilRole },
    /// Designate an heir for succession.
    DesignateHeir { character: u32 },
    /// Arrange marriage between two characters.
    ArrangeMarriage { a: u32, b: u32 },
    /// Begin religious conversion of a province.
    ConvertProvince { province: ProvinceId, religion: Religion },
    /// Launch an intrigue plot against a target character.
    LaunchPlot { target: u32, plot_type: PlotType },
    /// Back (support) an existing plot by its ID.
    BackPlot { plot_id: u32 },
    /// Use a spymaster to investigate and detect enemy plots.
    InvestigatePlot { spymaster: u32 },
    /// Establish a trade route between two adjacent provinces.
    EstablishTradeRoute { from: ProvinceId, to: ProvinceId },
    /// Disrupt (raid/blockade) an existing trade route.
    DisruptTradeRoute { route_id: u32 },
}

/// A queued action with sender wallet and turn submitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedAction {
    pub wallet: String,
    pub action: GameAction,
    pub submitted_turn: u32,
}
