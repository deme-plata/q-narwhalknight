//! Types for QNK-INDEX smart contract

use serde::{Deserialize, Serialize};

/// Index fund data structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QnkIndex {
    /// Unique index identifier (blake3 hash)
    pub index_id: [u8; 32],

    /// Human-readable name (e.g., "QNK Top 10")
    pub name: String,

    /// Trading symbol (e.g., "QNK10")
    pub symbol: String,

    /// Total supply of index shares
    pub total_supply: u64,

    /// Current component tokens
    pub components: Vec<IndexComponent>,

    /// Net Asset Value per share (in qug, 8 decimals)
    pub nav_per_share: u64,

    /// Last rebalance block height
    pub last_rebalance_height: u64,

    /// Blocks between rebalances
    pub rebalance_interval: u64,

    /// Annual management fee in basis points (100 = 1%)
    pub management_fee_bps: u16,

    /// Performance fee in basis points
    pub performance_fee_bps: u16,

    /// Minimum market cap for inclusion (in qug)
    pub min_market_cap: u64,

    /// Maximum number of components
    pub max_components: u8,

    /// Fund manager wallet
    pub manager: [u8; 32],

    /// Enable governance voting
    pub governance_enabled: bool,

    /// Weighting methodology
    pub methodology: IndexMethodology,

    /// Creation block height
    pub creation_block: u64,

    /// Total fees accrued
    pub total_fees_accrued: u64,

    /// High water mark for performance fees
    pub high_water_mark: u64,

    /// Emergency pause states
    pub paused_mint: bool,
    pub paused_redeem: bool,
    pub paused_rebalance: bool,
    pub emergency_paused_at: Option<u64>,
}

/// Index component (a token in the index)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexComponent {
    /// Token contract address
    pub token_address: [u8; 32],

    /// Token symbol
    pub symbol: String,

    /// Target weight in basis points (10000 = 100%)
    pub target_weight_bps: u16,

    /// Actual current weight
    pub actual_weight_bps: u16,

    /// Amount of token held
    pub holdings: u64,

    /// Current price in QUG (8 decimals)
    pub price_qug: u64,

    /// Rank by market cap (1 = highest)
    pub rank: u8,
}

/// Shareholder record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexShareHolder {
    /// Wallet address
    pub wallet: [u8; 32],

    /// Number of shares held
    pub shares: u64,

    /// NAV when entered
    pub entry_nav: u64,

    /// Block when entered
    pub entry_block: u64,
}

/// Weighting methodology
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IndexMethodology {
    /// Market cap weighted with caps
    MarketCapWeighted {
        max_component_weight: u16,
        min_component_weight: u16,
    },
    /// Equal weight across all components
    EqualWeight {
        components_count: u8,
    },
    /// Custom weights (must sum to 10000)
    CustomWeights {
        weights: Vec<u16>,
    },
    /// Risk-adjusted (inverse volatility)
    RiskAdjusted {
        volatility_window: u64,
        max_volatility_bps: u16,
    },
}

/// Price feed data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriceFeed {
    /// Token address
    pub token_address: [u8; 32],

    /// Current price in QUG (8 decimals)
    pub current_price: u64,

    /// 24h TWAP
    pub twap_24h: u64,

    /// Last update block
    pub last_update_block: u64,

    /// Oracle sources
    pub oracle_sources: Vec<OracleSource>,

    /// Confidence (0-100)
    pub confidence: u8,
}

/// Oracle data source
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OracleSource {
    /// Source identifier
    pub source_id: String,

    /// Price from this source
    pub price: u64,

    /// Weight for aggregation
    pub weight: u8,

    /// Last update timestamp
    pub last_update: u64,
}

/// Governance proposal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GovernanceProposal {
    /// Proposal ID
    pub proposal_id: u64,

    /// Index this proposal is for
    pub index_id: [u8; 32],

    /// Proposer wallet
    pub proposer: [u8; 32],

    /// Proposal type
    pub proposal_type: ProposalType,

    /// Description
    pub description: String,

    /// Voting start block
    pub start_block: u64,

    /// Voting end block
    pub end_block: u64,

    /// Votes for (in shares)
    pub votes_for: u64,

    /// Votes against (in shares)
    pub votes_against: u64,

    /// Voters who have voted
    pub voters: Vec<[u8; 32]>,

    /// Proposal status
    pub status: ProposalStatus,

    /// Execution data (encoded)
    pub execution_data: Vec<u8>,
}

/// Types of governance proposals
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProposalType {
    /// Add a new component
    AddComponent {
        token_address: [u8; 32],
        symbol: String,
        initial_weight_bps: u16,
    },
    /// Remove a component
    RemoveComponent {
        token_address: [u8; 32],
    },
    /// Change weights
    ChangeWeights {
        new_weights: Vec<(u8, u16)>, // (rank, new_weight_bps)
    },
    /// Change management fee
    ChangeFee {
        new_fee_bps: u16,
    },
    /// Change manager
    ChangeManager {
        new_manager: [u8; 32],
    },
    /// Emergency pause
    EmergencyPause {
        pause_mint: bool,
        pause_redeem: bool,
        pause_rebalance: bool,
    },
    /// Update methodology
    ChangeMethodology {
        new_methodology: IndexMethodology,
    },
}

/// Proposal status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProposalStatus {
    Active,
    Passed,
    Rejected,
    Executed,
    Cancelled,
}

/// Operation type for rate limiting
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperationType {
    Mint,
    Redeem,
    Rebalance,
}

/// Result of minting shares
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MintResult {
    pub shares_minted: u64,
    pub qug_deposited: u64,
    pub fee_paid: u64,
    pub nav_per_share: u64,
}

/// Result of redeeming shares
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RedeemResult {
    pub shares_redeemed: u64,
    pub qug_returned: u64,
    pub fee_paid: u64,
    pub nav_per_share: u64,
}

/// Rebalance result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RebalanceResult {
    /// Trades executed
    pub trades: Vec<RebalanceTrade>,

    /// Total slippage incurred
    pub total_slippage_bps: u16,

    /// Block executed at
    pub execution_block: u64,

    /// New NAV per share after rebalance
    pub new_nav_per_share: u64,
}

/// Individual trade during rebalance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RebalanceTrade {
    /// Token being traded
    pub token_address: [u8; 32],

    /// Buy or sell
    pub side: TradeSide,

    /// Amount traded
    pub amount: u64,

    /// Execution price
    pub price: u64,

    /// Slippage incurred
    pub slippage_bps: u16,
}

/// Trade direction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Index fund errors
#[derive(Clone, Debug, thiserror::Error)]
pub enum IndexError {
    #[error("Index not found")]
    IndexNotFound,

    #[error("Component not found")]
    ComponentNotFound,

    #[error("Component already exists in index")]
    ComponentAlreadyExists,

    #[error("Index is full")]
    IndexFull,

    #[error("Unauthorized operation")]
    Unauthorized,

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Insufficient balance")]
    InsufficientBalance,

    #[error("Insufficient shares")]
    InsufficientShares,

    #[error("Slippage exceeded")]
    SlippageExceeded,

    #[error("Contract is paused")]
    ContractPaused,

    #[error("Rate limited")]
    RateLimited,

    #[error("Fee too high (max 5%)")]
    FeeTooHigh,

    #[error("Invalid weight")]
    InvalidWeight,

    #[error("Weights must sum to 100%")]
    WeightsNot100Percent,

    #[error("Invalid component count")]
    InvalidComponentCount,

    #[error("Invalid parameter")]
    InvalidParameter,

    #[error("Arithmetic overflow")]
    ArithmeticOverflow,

    #[error("Arithmetic underflow")]
    ArithmeticUnderflow,

    #[error("Rebalance not needed yet")]
    RebalanceNotNeeded,

    #[error("Price data stale")]
    StalePriceData,

    #[error("Oracle error: {0}")]
    OracleError(String),

    #[error("Governance error: {0}")]
    GovernanceError(String),

    #[error("Proposal not found")]
    ProposalNotFound,

    #[error("Voting ended")]
    VotingEnded,

    #[error("Already voted")]
    AlreadyVoted,

    #[error("Quorum not reached")]
    QuorumNotReached,
}

/// API response for index info
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexInfoResponse {
    pub index_id: String,
    pub name: String,
    pub symbol: String,
    pub nav_per_share: f64,
    pub total_supply: u64,
    pub total_value_locked: u64,
    pub components: Vec<ComponentInfo>,
    pub management_fee_percent: f64,
    pub methodology: String,
    pub last_rebalance: u64,
    pub next_rebalance: u64,
    pub creation_block: u64,
}

/// Component info for API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentInfo {
    pub token_address: String,
    pub symbol: String,
    pub weight_percent: f64,
    pub target_weight_percent: f64,
    pub price_qug: f64,
    pub holdings: u64,
    pub rank: u8,
}

/// Shareholder info for API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShareholderInfo {
    pub wallet: String,
    pub shares: u64,
    pub value_qug: f64,
    pub unrealized_pnl: f64,
    pub entry_nav: f64,
    pub entry_block: u64,
}
