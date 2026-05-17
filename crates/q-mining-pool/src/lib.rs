//! Q-NarwhalKnight Mining Pool
//!
//! v2.3.0-beta: Fully Decentralized P2P Mining Pool
//!
//! A comprehensive mining pool implementation with:
//! - Stratum V1 protocol support
//! - PPLNS reward distribution with CRDT-based synchronization
//! - Variable difficulty (Vardiff)
//! - Share validation with VDF proofs
//! - Automated payouts with threshold signatures
//! - P2P decentralization via gossipsub
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    POOL OPERATOR NODE                    │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
//! │  │   Stratum   │  │    Share    │  │   Reward    │      │
//! │  │   Server    │  │  Validator  │  │ Calculator  │      │
//! │  └─────────────┘  └─────────────┘  └─────────────┘      │
//! │         │                │                │              │
//! │         ▼                ▼                ▼              │
//! │  ┌─────────────────────────────────────────────────┐    │
//! │  │              Pool State Manager                  │    │
//! │  │  - Workers, Shares, Jobs, Payouts                │    │
//! │  └─────────────────────────────────────────────────┘    │
//! │                          │                              │
//! │  ┌─────────────────────────────────────────────────┐    │
//! │  │         Distributed Pool Coordinator             │    │
//! │  │  - CRDT PPLNS, Gossipsub, Threshold Sigs        │    │
//! │  └─────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────┘
//!                          │
//!           ┌──────────────┼──────────────┐
//!           ▼              ▼              ▼
//!      ┌────────┐    ┌────────┐    ┌────────┐
//!      │Worker 1│    │Worker 2│    │Worker N│
//!      └────────┘    └────────┘    └────────┘
//!           │              │              │
//!           ▼              ▼              ▼
//!      ┌────────────────────────────────────┐
//!      │       P2P Gossipsub Network         │
//!      │  - Share propagation                │
//!      │  - Block consensus                  │
//!      │  - PPLNS synchronization            │
//!      └────────────────────────────────────┘
//! ```

pub mod config;
pub mod error;
pub mod job;
pub mod payout;
pub mod pplns;
pub mod share;
pub mod stratum;
pub mod vardiff;
pub mod worker;
pub mod pool;
pub mod distributed;

pub use config::PoolConfig;
pub use error::{PoolError, PoolResult};
pub use job::{MiningJob, JobManager};
pub use payout::{Payout, PayoutProcessor, PayoutStatus};
pub use pplns::PPLNSCalculator;
pub use share::{Share, ShareValidator, ShareValidationResult};
pub use stratum::{StratumServer, StratumMessage, StratumResponse};
pub use vardiff::VardiffController;
pub use worker::{Worker, WorkerId, WorkerManager};
pub use pool::MiningPool;

// v2.3.0-beta: Distributed P2P Pool exports
pub use distributed::{
    DistributedShare, DistributedPPLNS, DistributedPoolCoordinator,
    PoolTopics, PoolNodeDiscovery, PoolNodeInfo,
    BlockFoundAnnouncement, NodeAttestation,
    PayoutBatch, PayoutVote, ThresholdSignature,
    PoolMessage, PeerIdBytes, ShareId,
};

/// Mining pool version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default pool fee in basis points (100 = 1%)
pub const DEFAULT_POOL_FEE_BPS: u64 = 250; // v8.6.0: raised from 1.5% to 2.5%

/// Development fee in basis points (immutable, protocol-level)
pub const DEV_FEE_BPS: u64 = 175; // v8.6.0: raised from 1% to 1.75%

/// Default minimum payout threshold in atomic units
pub const DEFAULT_MIN_PAYOUT: u64 = 50_000_000; // v8.6.0: raised from 0.01 to 0.05 QUG (reduce payout frequency 5x)

/// Default PPLNS N-factor
pub const DEFAULT_PPLNS_N_FACTOR: f64 = 2.0;

/// Default vardiff target time (seconds between shares)
pub const DEFAULT_VARDIFF_TARGET_TIME: f64 = 20.0;

/// Default stratum port
pub const DEFAULT_STRATUM_PORT: u16 = 3333;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fee_calculation() {
        let block_reward = 2_000_000_000u64; // 2.0 QUG in atomic units

        // v8.6.0: Dev fee (1.75%)
        let dev_fee = block_reward * DEV_FEE_BPS / 10_000;
        assert_eq!(dev_fee, 35_000_000); // 0.035 QUG

        // v8.6.0: Pool fee (2.5%)
        let pool_fee = block_reward * DEFAULT_POOL_FEE_BPS / 10_000;
        assert_eq!(pool_fee, 50_000_000); // 0.05 QUG

        // Miner rewards
        let miner_rewards = block_reward - dev_fee - pool_fee;
        assert_eq!(miner_rewards, 1_915_000_000); // 1.915 QUG
    }
}
