# Q-NarwhalKnight Endgame Solution - Technical Review

**Version**: 1.5.0-beta
**Date**: December 2024
**Authors**: Q-NarwhalKnight Core Team
**Status**: Design Specification

---

## Executive Summary

This document presents a comprehensive technical design for solving the "endgame problem" - ensuring perpetual network security without compromising the 21M QUG supply cap. Our solution combines four synergistic layers that preserve scarcity while creating sustainable security funding.

**Key Principles**:
- **21M Cap Preserved**: Zero inflation, no tail emission
- **Decentralized**: No trusted parties, on-chain governance
- **Adaptive**: ML-driven optimization of all parameters
- **User-Friendly**: Intuitive staking with real-time yield visualization

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Layer 1: VDF Security Floor](#2-layer-1-vdf-security-floor)
3. [Layer 2: Multi-Source Fee Generation](#3-layer-2-multi-source-fee-generation)
4. [Layer 3: Fee Smoothing Reserve](#4-layer-3-fee-smoothing-reserve)
5. [Layer 4: Productive Staking System](#5-layer-4-productive-staking-system)
6. [ML-Driven Adaptive Optimization](#6-ml-driven-adaptive-optimization)
7. [Decentralization Mechanisms](#7-decentralization-mechanisms)
8. [Frontend UI/UX Design](#8-frontend-uiux-design)
9. [Security Analysis](#9-security-analysis)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Q-NARWHALKNIGHT ENDGAME ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     LAYER 4: STAKING SYSTEM                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │   Staking   │  │   Yield     │  │  Validator  │  │    ML      │ │   │
│  │  │   Pools     │  │ Calculator  │  │  Selection  │  │ Optimizer  │ │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │   │
│  └─────────┼────────────────┼────────────────┼───────────────┼────────┘   │
│            │                │                │               │             │
│  ┌─────────▼────────────────▼────────────────▼───────────────▼────────┐   │
│  │                   LAYER 3: FEE SMOOTHING RESERVE                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │  Reserve    │  │  Adaptive   │  │  Emergency  │  │  DAO       │ │   │
│  │  │  Pool       │  │  Thresholds │  │  Buffer     │  │ Governance │ │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │   │
│  └─────────┼────────────────┼────────────────┼───────────────┼────────┘   │
│            │                │                │               │             │
│  ┌─────────▼────────────────▼────────────────▼───────────────▼────────┐   │
│  │                 LAYER 2: MULTI-SOURCE FEE GENERATION               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │  DEX Swaps  │  │  CDP Ops    │  │  Smart      │  │  Transfer  │ │   │
│  │  │  (0.3%)     │  │  (0.5%)     │  │  Contracts  │  │  Fees      │ │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │   │
│  └─────────┼────────────────┼────────────────┼───────────────┼────────┘   │
│            │                │                │               │             │
│  ┌─────────▼────────────────▼────────────────▼───────────────▼────────┐   │
│  │                    LAYER 1: VDF SECURITY FLOOR                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │  Sequential │  │  Adaptive   │  │  Attack     │  │  Hashpower │ │   │
│  │  │  VDF Chain  │  │  Iterations │  │  Time-Lock  │  │  Baseline  │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                     CONSENSUS LAYER (DAG-KNIGHT)                   │    │
│  │            Parallel Block Production + VDF Anchoring               │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Design Goals

| Goal | Mechanism | Metric |
|------|-----------|--------|
| **Scarcity** | Fixed 21M cap, no inflation | Supply ≤ 21,000,000 QUG |
| **Security** | VDF + Staking + PoW hybrid | Attack cost > 10× market cap |
| **Decentralization** | On-chain governance, permissionless staking | Nakamoto coefficient > 20 |
| **Sustainability** | Fee-funded security, no subsidy dependency | 100% fee-funded by 2140 |
| **Adaptivity** | ML-optimized parameters | <5% fee volatility |

---

## 2. Layer 1: VDF Security Floor

### 2.1 Current Implementation (Already Complete)

The VDF (Verifiable Delay Function) provides a time-based security floor independent of hashpower economics.

```rust
// crates/q-api-server/src/block_producer.rs

/// VDF Chain Binding - Sequential computation that cannot be parallelized
pub struct VDFSecurityFloor {
    /// Base iterations (scales with network maturity)
    base_iterations: u64,

    /// Height-based scaling (+1 per 1000 blocks)
    height_scaling: u64,

    /// Peer-based scaling (+10 per connected peer)
    peer_scaling: u64,

    /// Security tier multiplier
    security_multiplier: f64,
}

impl VDFSecurityFloor {
    /// Calculate minimum attack time in seconds
    pub fn minimum_attack_time(&self, attacker_hashpower_ratio: f64) -> Duration {
        // Even with 100% hashpower, attacker must complete VDF sequentially
        let iterations = self.total_iterations();

        // Assuming ~1M iterations/second on modern hardware
        let vdf_time_seconds = iterations as f64 / 1_000_000.0;

        // Attack requires building alternative chain
        // Each block needs VDF completion
        let blocks_to_reorg = 6; // Standard confirmation depth

        Duration::from_secs_f64(vdf_time_seconds * blocks_to_reorg as f64)
    }

    fn total_iterations(&self) -> u64 {
        self.base_iterations + self.height_scaling + self.peer_scaling
    }
}
```

### 2.2 Security Guarantees

```
┌──────────────────────────────────────────────────────────────┐
│                VDF SECURITY ANALYSIS                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Traditional PoW (Bitcoin):                                  │
│    Attack Time = Hashpower Acquisition Time                  │
│    With enough capital: INSTANT                              │
│                                                              │
│  VDF-Enhanced PoW (Q-NarwhalKnight):                        │
│    Attack Time = Hashpower Acquisition + VDF Sequential Time │
│    With unlimited capital: STILL TAKES HOURS                 │
│                                                              │
│  Example at Height 1,000,000 with 100 peers:                │
│    VDF iterations = 1000 + 1000 + 1000 = 3000               │
│    Per-block VDF time = 3ms                                  │
│    6-block reorg = 18ms minimum (+ hashpower time)          │
│                                                              │
│  Enhanced Security Mode (future):                            │
│    VDF iterations = 100,000 + scaling                        │
│    Per-block VDF time = 100ms                                │
│    6-block reorg = 600ms minimum wall-clock time            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 Adaptive VDF Scaling

```rust
/// ML-driven VDF iteration adjustment
pub struct AdaptiveVDFController {
    /// Historical attack attempts (detected anomalies)
    attack_history: VecDeque<AttackAttempt>,

    /// Network security target (bits)
    target_security_bits: u32,

    /// ML model for iteration prediction
    iteration_predictor: VDFIterationPredictor,
}

impl AdaptiveVDFController {
    /// Adjust VDF iterations based on network conditions
    pub async fn recommend_iterations(&self, context: &NetworkContext) -> u64 {
        let features = VDFFeatures {
            current_hashrate: context.network_hashrate,
            market_cap: context.market_cap_usd,
            recent_attacks: self.attack_history.len(),
            peer_count: context.connected_peers,
            block_height: context.current_height,
        };

        // ML prediction with safety bounds
        let ml_recommendation = self.iteration_predictor.predict(&features);

        // Safety bounds: 1000 minimum, 1M maximum
        ml_recommendation.clamp(1_000, 1_000_000)
    }
}
```

---

## 3. Layer 2: Multi-Source Fee Generation

### 3.1 Fee Sources Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEE GENERATION SOURCES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   DEX SWAPS     │    │   CDP OPERATIONS │                   │
│  │                 │    │                  │                    │
│  │  Fee: 0.30%     │    │  Mint: 0.50%     │                   │
│  │  LP Share: 0.25%│    │  Burn: 0.25%     │                   │
│  │  Protocol: 0.05%│    │  Protocol: 0.25% │                   │
│  └────────┬────────┘    └────────┬─────────┘                   │
│           │                      │                              │
│  ┌────────▼──────────────────────▼─────────┐                   │
│  │           PROTOCOL FEE COLLECTOR         │                   │
│  └────────┬──────────────────────┬─────────┘                   │
│           │                      │                              │
│  ┌────────▼────────┐    ┌────────▼─────────┐                   │
│  │  SMART CONTRACT │    │   TRANSFER FEES  │                   │
│  │     GAS FEES    │    │                  │                   │
│  │                 │    │  Base: 0.001 QUG │                   │
│  │  Base: 21 gas   │    │  Per KB: 0.0001  │                   │
│  │  Per op: varies │    │  Priority: bid   │                   │
│  └────────┬────────┘    └────────┬─────────┘                   │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      ▼                                          │
│  ┌─────────────────────────────────────────┐                   │
│  │         FEE DISTRIBUTION ENGINE          │                   │
│  │                                          │                   │
│  │   ┌─────────┐ ┌─────────┐ ┌──────────┐  │                   │
│  │   │ Miners  │ │ Stakers │ │ Reserve  │  │                   │
│  │   │  30%    │ │  50%    │ │   20%    │  │                   │
│  │   └─────────┘ └─────────┘ └──────────┘  │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Fee Distribution Contract

```rust
// crates/q-vm/src/contracts/fee_distribution.rs

/// Decentralized fee distribution with no trusted parties
pub struct FeeDistributionEngine {
    /// Distribution ratios (must sum to 100%)
    pub miner_share: u8,      // 30% - Block producers
    pub staker_share: u8,     // 50% - Security stakers
    pub reserve_share: u8,    // 20% - Fee smoothing reserve

    /// Accumulated fees awaiting distribution
    pending_fees: u64,

    /// Distribution frequency (blocks)
    distribution_interval: u64,

    /// Governance-controlled parameters
    governance: GovernanceConfig,
}

impl FeeDistributionEngine {
    /// Process fees from a block
    pub fn collect_fees(&mut self, block_fees: BlockFees) -> Result<()> {
        // Aggregate all fee sources
        let total_fees = block_fees.transfer_fees
            + block_fees.dex_fees
            + block_fees.cdp_fees
            + block_fees.contract_gas;

        self.pending_fees += total_fees;

        // Distribute if interval reached
        if self.should_distribute() {
            self.distribute_fees()?;
        }

        Ok(())
    }

    /// Distribute accumulated fees
    fn distribute_fees(&mut self) -> Result<()> {
        let total = self.pending_fees;

        // Calculate shares
        let miner_amount = total * self.miner_share as u64 / 100;
        let staker_amount = total * self.staker_share as u64 / 100;
        let reserve_amount = total - miner_amount - staker_amount;

        // Distribute to miners (proportional to recent hashpower)
        self.distribute_to_miners(miner_amount)?;

        // Distribute to stakers (proportional to stake)
        self.distribute_to_stakers(staker_amount)?;

        // Send to fee reserve
        self.send_to_reserve(reserve_amount)?;

        self.pending_fees = 0;

        emit_event!(FeeDistribution {
            total,
            miner_amount,
            staker_amount,
            reserve_amount,
            timestamp: current_timestamp(),
        });

        Ok(())
    }
}
```

### 3.3 Fee Estimation API

```rust
/// Real-time fee estimation for wallet integration
pub struct FeeEstimator {
    /// Recent fee history for prediction
    fee_history: VecDeque<FeeDataPoint>,

    /// ML model for fee prediction
    fee_predictor: FeePredictionModel,

    /// Current network congestion
    congestion_level: CongestionLevel,
}

impl FeeEstimator {
    /// Estimate fee for transaction
    pub async fn estimate_fee(&self, tx: &Transaction) -> FeeEstimate {
        let base_fee = self.calculate_base_fee(tx);
        let priority_fee = self.calculate_priority_fee();

        // ML-predicted optimal fee for confirmation time targets
        let predictions = self.fee_predictor.predict_confirmation_times(base_fee);

        FeeEstimate {
            base_fee,
            priority_fee,
            total_recommended: base_fee + priority_fee,
            confirmation_predictions: ConfirmationPredictions {
                fast_1_block: predictions.fee_for_blocks(1),
                standard_3_blocks: predictions.fee_for_blocks(3),
                economy_10_blocks: predictions.fee_for_blocks(10),
            },
            network_congestion: self.congestion_level,
        }
    }
}
```

---

## 4. Layer 3: Fee Smoothing Reserve

### 4.1 Reserve Architecture

The Fee Smoothing Reserve acts as a decentralized insurance pool that buffers fee volatility, ensuring miners and stakers receive stable rewards even during low-activity periods.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEE SMOOTHING RESERVE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                     ┌─────────────────┐                        │
│                     │  RESERVE POOL   │                        │
│                     │                 │                        │
│                     │  Balance: X QUG │                        │
│                     │  Target: Y QUG  │                        │
│                     │  Health: Z%     │                        │
│                     └────────┬────────┘                        │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │  INFLOW     │     │  OUTFLOW    │     │  EMERGENCY  │      │
│  │             │     │             │     │             │      │
│  │ High fees   │     │ Low fees    │     │ Crisis mode │      │
│  │ → Save 20%  │     │ → Release   │     │ → Unlock    │      │
│  └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    ML CONTROLLER                         │   │
│  │  • Predicts fee trends                                   │   │
│  │  • Adjusts save/release thresholds                       │   │
│  │  • Optimizes reserve target                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Reserve Contract Implementation

```rust
// crates/q-vm/src/contracts/fee_reserve.rs

use crate::ml::ReserveOptimizer;

/// Decentralized Fee Smoothing Reserve
///
/// Invariants:
/// - Reserve balance >= 0
/// - No new coins created (redistribution only)
/// - Governance-controlled parameters
pub struct FeeSmoothingReserve {
    /// Current reserve balance (in satoshis)
    balance: u64,

    /// Target reserve size (ML-optimized)
    target_balance: u64,

    /// Minimum guaranteed miner reward per block
    min_block_reward: u64,

    /// Maximum reserve contribution per block (% of fees)
    max_contribution_rate: u8,

    /// ML optimizer for adaptive thresholds
    optimizer: ReserveOptimizer,

    /// Historical data for ML training
    history: ReserveHistory,

    /// DAO governance reference
    governance: GovernanceRef,
}

impl FeeSmoothingReserve {
    /// Process block fees through the reserve
    pub fn process_block(&mut self, block_fees: u64, block_height: u64) -> ReserveDecision {
        // Update ML model with new data
        self.history.record(block_fees, block_height);

        // Get ML recommendation
        let ml_decision = self.optimizer.recommend_action(&self.history, self.balance);

        // Calculate thresholds
        let avg_fees = self.history.moving_average(100); // 100-block average
        let target_reward = self.min_block_reward.max((avg_fees * 80) / 100);

        if block_fees >= target_reward * 2 {
            // High fee period: save surplus to reserve
            let surplus = block_fees - target_reward;
            let contribution = surplus.min(
                (block_fees * self.max_contribution_rate as u64) / 100
            );

            // Apply ML-recommended contribution rate
            let adjusted_contribution = ml_decision.adjust_contribution(contribution);

            self.balance += adjusted_contribution;

            ReserveDecision::Save {
                amount: adjusted_contribution,
                to_miners: block_fees - adjusted_contribution,
                reserve_health: self.health_ratio(),
            }
        } else if block_fees < target_reward {
            // Low fee period: supplement from reserve
            let deficit = target_reward - block_fees;
            let supplement = deficit.min(self.balance).min(
                self.max_withdrawal_rate()
            );

            // Apply ML-recommended supplement rate
            let adjusted_supplement = ml_decision.adjust_supplement(supplement);

            self.balance -= adjusted_supplement;

            ReserveDecision::Supplement {
                amount: adjusted_supplement,
                to_miners: block_fees + adjusted_supplement,
                reserve_health: self.health_ratio(),
            }
        } else {
            // Normal fees: pass through
            ReserveDecision::PassThrough {
                to_miners: block_fees,
                reserve_health: self.health_ratio(),
            }
        }
    }

    /// Calculate reserve health (0-100%)
    fn health_ratio(&self) -> u8 {
        if self.target_balance == 0 {
            return 100;
        }
        ((self.balance * 100) / self.target_balance).min(100) as u8
    }

    /// Maximum withdrawal rate to prevent reserve depletion
    fn max_withdrawal_rate(&self) -> u64 {
        // Never withdraw more than 1% of reserve per block
        self.balance / 100
    }

    /// Emergency mode: governance can unlock additional funds
    pub fn emergency_unlock(&mut self, amount: u64, governance_proof: GovernanceProof) -> Result<()> {
        // Verify governance approval (requires 2/3 vote)
        self.governance.verify_emergency_unlock(&governance_proof)?;

        let unlock_amount = amount.min(self.balance);
        self.balance -= unlock_amount;

        emit_event!(EmergencyUnlock {
            amount: unlock_amount,
            governance_proof_hash: governance_proof.hash(),
            remaining_balance: self.balance,
        });

        Ok(())
    }
}

#[derive(Debug)]
pub enum ReserveDecision {
    Save {
        amount: u64,
        to_miners: u64,
        reserve_health: u8,
    },
    Supplement {
        amount: u64,
        to_miners: u64,
        reserve_health: u8,
    },
    PassThrough {
        to_miners: u64,
        reserve_health: u8,
    },
}
```

### 4.3 ML Reserve Optimizer

```rust
// crates/q-storage/src/ml/reserve_optimizer.rs

/// Online learning optimizer for fee reserve management
pub struct ReserveOptimizer {
    /// Weights for contribution rate prediction
    contribution_weights: [f32; NUM_FEATURES],

    /// Weights for supplement rate prediction
    supplement_weights: [f32; NUM_FEATURES],

    /// Learning rate
    learning_rate: f32,

    /// Prediction history for validation
    prediction_history: VecDeque<PredictionOutcome>,
}

impl ReserveOptimizer {
    /// Features for reserve decision
    fn extract_features(&self, history: &ReserveHistory, balance: u64) -> [f32; NUM_FEATURES] {
        [
            // Fee trend (positive = increasing)
            history.fee_trend_slope(),

            // Fee volatility
            history.fee_volatility(),

            // Reserve health (0-1)
            (balance as f32) / (history.avg_daily_fees() * 30.0), // 30-day coverage

            // Network activity level
            history.transaction_volume_trend(),

            // Time of day/week patterns
            history.cyclical_pattern_strength(),

            // Market conditions (from oracle)
            history.market_sentiment_score(),
        ]
    }

    /// Recommend reserve action
    pub fn recommend_action(
        &self,
        history: &ReserveHistory,
        balance: u64,
    ) -> ReserveRecommendation {
        let features = self.extract_features(history, balance);

        // Linear prediction + sigmoid for bounded output
        let contribution_rate = self.predict_contribution(&features);
        let supplement_rate = self.predict_supplement(&features);

        ReserveRecommendation {
            contribution_multiplier: contribution_rate,
            supplement_multiplier: supplement_rate,
            confidence: self.prediction_confidence(),
        }
    }

    /// Online learning update
    pub fn learn(&mut self, outcome: &ReserveOutcome) {
        // Calculate reward: stability of miner income
        let reward = outcome.calculate_stability_reward();

        // Gradient update
        let features = self.extract_features(&outcome.history, outcome.balance);

        for i in 0..NUM_FEATURES {
            let gradient = reward * features[i];
            self.contribution_weights[i] += self.learning_rate * gradient;
            self.supplement_weights[i] += self.learning_rate * gradient;
        }
    }
}
```

---

## 5. Layer 4: Productive Staking System

### 5.1 Staking Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PRODUCTIVE STAKING SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        STAKING POOLS                                 │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  FLEXIBLE    │  │  30-DAY      │  │  365-DAY     │              │   │
│  │  │              │  │              │  │              │              │   │
│  │  │  Lock: None  │  │  Lock: 30d   │  │  Lock: 365d  │              │   │
│  │  │  Boost: 1.0x │  │  Boost: 1.5x │  │  Boost: 3.0x │              │   │
│  │  │  APY: ~2%    │  │  APY: ~3%    │  │  APY: ~6%    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      YIELD CALCULATION                               │   │
│  │                                                                      │   │
│  │   Your Yield = (Your Stake × Boost) / Total Weighted Stake × Fees  │   │
│  │                                                                      │   │
│  │   Example:                                                           │   │
│  │     Stake: 10,000 QUG in 365-day pool (3.0x boost)                 │   │
│  │     Weighted Stake: 30,000                                          │   │
│  │     Total Weighted Stake: 10,000,000                                │   │
│  │     Daily Fees: 1,000 QUG                                           │   │
│  │     Staker Share (50%): 500 QUG                                     │   │
│  │     Your Daily Yield: 500 × (30,000/10,000,000) = 1.5 QUG          │   │
│  │     Annual Yield: 1.5 × 365 = 547.5 QUG (5.47% APY)                │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     VALIDATOR SELECTION                              │   │
│  │                                                                      │   │
│  │  Stakers vote on transaction ordering (MEV protection)              │   │
│  │  Weight = Stake × Lock Duration × Reputation Score                  │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Staking Contract Implementation

```rust
// crates/q-vm/src/contracts/staking.rs

use std::collections::HashMap;

/// Staking pool configuration
#[derive(Debug, Clone)]
pub struct StakingPoolConfig {
    pub pool_id: PoolId,
    pub lock_duration: Duration,
    pub boost_multiplier: f64,
    pub early_unstake_penalty: u8, // Percentage
    pub min_stake: u64,
}

/// Individual stake position
#[derive(Debug, Clone)]
pub struct StakePosition {
    pub staker: Address,
    pub amount: u64,
    pub pool_id: PoolId,
    pub staked_at: u64,        // Block height
    pub unlock_at: u64,        // Block height
    pub accumulated_rewards: u64,
    pub last_claim_height: u64,
}

/// Main staking contract
pub struct StakingContract {
    /// Pool configurations
    pools: HashMap<PoolId, StakingPoolConfig>,

    /// All stake positions
    positions: HashMap<PositionId, StakePosition>,

    /// Staker -> Position IDs mapping
    staker_positions: HashMap<Address, Vec<PositionId>>,

    /// Total staked per pool
    pool_totals: HashMap<PoolId, u64>,

    /// Total weighted stake (for yield calculation)
    total_weighted_stake: u64,

    /// Pending rewards for distribution
    pending_rewards: u64,

    /// ML optimizer for APY prediction
    apy_optimizer: APYOptimizer,
}

impl StakingContract {
    /// Initialize with default pools
    pub fn new() -> Self {
        let mut pools = HashMap::new();

        // Flexible pool - no lock, 1x boost
        pools.insert(PoolId::Flexible, StakingPoolConfig {
            pool_id: PoolId::Flexible,
            lock_duration: Duration::ZERO,
            boost_multiplier: 1.0,
            early_unstake_penalty: 0,
            min_stake: 100_000_000, // 1 QUG minimum
        });

        // 30-day pool - 1.5x boost
        pools.insert(PoolId::ThirtyDay, StakingPoolConfig {
            pool_id: PoolId::ThirtyDay,
            lock_duration: Duration::from_days(30),
            boost_multiplier: 1.5,
            early_unstake_penalty: 10,
            min_stake: 100_000_000,
        });

        // 90-day pool - 2.0x boost
        pools.insert(PoolId::NinetyDay, StakingPoolConfig {
            pool_id: PoolId::NinetyDay,
            lock_duration: Duration::from_days(90),
            boost_multiplier: 2.0,
            early_unstake_penalty: 15,
            min_stake: 100_000_000,
        });

        // 365-day pool - 3.0x boost
        pools.insert(PoolId::OneYear, StakingPoolConfig {
            pool_id: PoolId::OneYear,
            lock_duration: Duration::from_days(365),
            boost_multiplier: 3.0,
            early_unstake_penalty: 25,
            min_stake: 100_000_000,
        });

        Self {
            pools,
            positions: HashMap::new(),
            staker_positions: HashMap::new(),
            pool_totals: HashMap::new(),
            total_weighted_stake: 0,
            pending_rewards: 0,
            apy_optimizer: APYOptimizer::new(),
        }
    }

    /// Stake QUG into a pool
    pub fn stake(
        &mut self,
        staker: Address,
        amount: u64,
        pool_id: PoolId,
        current_height: u64,
    ) -> Result<PositionId> {
        let pool = self.pools.get(&pool_id)
            .ok_or(StakingError::InvalidPool)?;

        // Validate minimum stake
        if amount < pool.min_stake {
            return Err(StakingError::BelowMinimum);
        }

        // Calculate unlock height
        let blocks_per_day = 43200; // ~2 second blocks
        let lock_blocks = pool.lock_duration.as_days() as u64 * blocks_per_day;
        let unlock_at = current_height + lock_blocks;

        // Create position
        let position_id = PositionId::generate();
        let position = StakePosition {
            staker,
            amount,
            pool_id,
            staked_at: current_height,
            unlock_at,
            accumulated_rewards: 0,
            last_claim_height: current_height,
        };

        // Update state
        self.positions.insert(position_id, position);
        self.staker_positions.entry(staker).or_default().push(position_id);
        *self.pool_totals.entry(pool_id).or_insert(0) += amount;

        // Update weighted stake
        self.total_weighted_stake += (amount as f64 * pool.boost_multiplier) as u64;

        emit_event!(Staked {
            staker,
            amount,
            pool_id,
            position_id,
            unlock_at,
        });

        Ok(position_id)
    }

    /// Unstake from a position
    pub fn unstake(
        &mut self,
        staker: Address,
        position_id: PositionId,
        current_height: u64,
    ) -> Result<UnstakeResult> {
        let position = self.positions.get(&position_id)
            .ok_or(StakingError::PositionNotFound)?;

        // Verify ownership
        if position.staker != staker {
            return Err(StakingError::NotOwner);
        }

        let pool = &self.pools[&position.pool_id];

        // Calculate penalty for early unstake
        let (amount_returned, penalty) = if current_height < position.unlock_at {
            let penalty_amount = (position.amount * pool.early_unstake_penalty as u64) / 100;
            (position.amount - penalty_amount, penalty_amount)
        } else {
            (position.amount, 0)
        };

        // Claim pending rewards first
        let rewards = self.calculate_pending_rewards(position_id, current_height)?;

        // Update state
        self.total_weighted_stake -= (position.amount as f64 * pool.boost_multiplier) as u64;
        *self.pool_totals.get_mut(&position.pool_id).unwrap() -= position.amount;
        self.positions.remove(&position_id);

        // Remove from staker's positions
        if let Some(positions) = self.staker_positions.get_mut(&staker) {
            positions.retain(|&id| id != position_id);
        }

        // Penalty goes to reserve (not burned - preserves supply)
        if penalty > 0 {
            self.pending_rewards += penalty;
        }

        emit_event!(Unstaked {
            staker,
            position_id,
            amount_returned,
            penalty,
            rewards_claimed: rewards,
        });

        Ok(UnstakeResult {
            amount_returned,
            penalty,
            rewards_claimed: rewards,
        })
    }

    /// Distribute fee rewards to stakers
    pub fn distribute_rewards(&mut self, fee_amount: u64, current_height: u64) -> Result<()> {
        if self.total_weighted_stake == 0 {
            // No stakers - fees go to reserve
            self.pending_rewards += fee_amount;
            return Ok(());
        }

        // Distribute proportionally to weighted stake
        for (position_id, position) in &mut self.positions {
            let pool = &self.pools[&position.pool_id];
            let weighted_stake = (position.amount as f64 * pool.boost_multiplier) as u64;

            let share = (fee_amount * weighted_stake) / self.total_weighted_stake;
            position.accumulated_rewards += share;
        }

        emit_event!(RewardsDistributed {
            amount: fee_amount,
            total_staked: self.total_weighted_stake,
            block_height: current_height,
        });

        Ok(())
    }

    /// Calculate pending rewards for a position
    pub fn calculate_pending_rewards(
        &self,
        position_id: PositionId,
        current_height: u64,
    ) -> Result<u64> {
        let position = self.positions.get(&position_id)
            .ok_or(StakingError::PositionNotFound)?;

        Ok(position.accumulated_rewards)
    }

    /// Claim rewards without unstaking
    pub fn claim_rewards(
        &mut self,
        staker: Address,
        position_id: PositionId,
        current_height: u64,
    ) -> Result<u64> {
        let position = self.positions.get_mut(&position_id)
            .ok_or(StakingError::PositionNotFound)?;

        if position.staker != staker {
            return Err(StakingError::NotOwner);
        }

        let rewards = position.accumulated_rewards;
        position.accumulated_rewards = 0;
        position.last_claim_height = current_height;

        emit_event!(RewardsClaimed {
            staker,
            position_id,
            amount: rewards,
        });

        Ok(rewards)
    }

    /// Get estimated APY for a pool
    pub fn estimate_apy(&self, pool_id: PoolId) -> f64 {
        let pool = match self.pools.get(&pool_id) {
            Some(p) => p,
            None => return 0.0,
        };

        // Use ML optimizer for prediction
        self.apy_optimizer.predict_apy(
            self.total_weighted_stake,
            pool.boost_multiplier,
            self.pending_rewards,
        )
    }

    /// Get staker dashboard data
    pub fn get_staker_dashboard(&self, staker: Address, current_height: u64) -> StakerDashboard {
        let positions = self.staker_positions.get(&staker)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.positions.get(id))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        let total_staked: u64 = positions.iter().map(|p| p.amount).sum();
        let total_rewards: u64 = positions.iter().map(|p| p.accumulated_rewards).sum();

        let positions_detail: Vec<PositionDetail> = positions.iter().map(|p| {
            let pool = &self.pools[&p.pool_id];
            let is_locked = current_height < p.unlock_at;
            let blocks_remaining = if is_locked { p.unlock_at - current_height } else { 0 };

            PositionDetail {
                position_id: p.staker.into(), // Simplified
                pool_name: pool.pool_id.to_string(),
                staked_amount: p.amount,
                pending_rewards: p.accumulated_rewards,
                boost_multiplier: pool.boost_multiplier,
                is_locked,
                blocks_until_unlock: blocks_remaining,
                estimated_unlock_time: blocks_remaining * 2, // 2 sec blocks
            }
        }).collect();

        StakerDashboard {
            staker,
            total_staked,
            total_pending_rewards: total_rewards,
            positions: positions_detail,
            estimated_daily_yield: self.estimate_daily_yield(staker),
            estimated_annual_yield: self.estimate_annual_yield(staker),
        }
    }

    fn estimate_daily_yield(&self, staker: Address) -> u64 {
        // Calculate based on current fee rate and staker's weighted position
        let staker_weighted = self.staker_positions.get(&staker)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.positions.get(id))
                    .map(|p| {
                        let pool = &self.pools[&p.pool_id];
                        (p.amount as f64 * pool.boost_multiplier) as u64
                    })
                    .sum::<u64>()
            })
            .unwrap_or(0);

        if self.total_weighted_stake == 0 {
            return 0;
        }

        // Use historical average daily fees
        let avg_daily_fees = self.apy_optimizer.average_daily_fees();
        let staker_share = 50; // 50% to stakers

        (avg_daily_fees * staker_share / 100 * staker_weighted) / self.total_weighted_stake
    }

    fn estimate_annual_yield(&self, staker: Address) -> u64 {
        self.estimate_daily_yield(staker) * 365
    }
}
```

### 5.3 APY Optimizer (ML-Driven)

```rust
// crates/q-storage/src/ml/apy_optimizer.rs

/// ML model for APY prediction and optimization
pub struct APYOptimizer {
    /// Historical fee data
    fee_history: VecDeque<DailyFeeData>,

    /// Staking participation history
    staking_history: VecDeque<StakingSnapshot>,

    /// Linear regression weights
    weights: [f32; 6],

    /// Running average of daily fees
    avg_daily_fees: f64,
}

impl APYOptimizer {
    pub fn new() -> Self {
        Self {
            fee_history: VecDeque::with_capacity(365),
            staking_history: VecDeque::with_capacity(365),
            weights: [0.3, 0.2, 0.15, 0.15, 0.1, 0.1], // Initial weights
            avg_daily_fees: 0.0,
        }
    }

    /// Predict APY for given conditions
    pub fn predict_apy(
        &self,
        total_staked: u64,
        boost_multiplier: f64,
        pending_rewards: u64,
    ) -> f64 {
        if total_staked == 0 {
            return 0.0;
        }

        // Feature extraction
        let features = [
            self.avg_daily_fees / 1e8,                    // Daily fees (QUG)
            total_staked as f64 / 1e16,                   // Total stake (normalized)
            self.fee_trend(),                              // Fee trend
            self.staking_ratio(),                          // Participation rate
            boost_multiplier,                              // Pool multiplier
            pending_rewards as f64 / 1e8,                 // Pending (QUG)
        ];

        // Linear prediction
        let mut prediction: f64 = 0.0;
        for (i, &feat) in features.iter().enumerate() {
            prediction += self.weights[i] as f64 * feat;
        }

        // Clamp to reasonable range (0.1% to 50% APY)
        (prediction * 100.0).clamp(0.1, 50.0)
    }

    /// Record daily data for learning
    pub fn record_daily(&mut self, fees: u64, total_staked: u64) {
        self.fee_history.push_back(DailyFeeData {
            fees,
            timestamp: current_timestamp(),
        });

        if self.fee_history.len() > 365 {
            self.fee_history.pop_front();
        }

        // Update running average
        let sum: u64 = self.fee_history.iter().map(|d| d.fees).sum();
        self.avg_daily_fees = sum as f64 / self.fee_history.len() as f64;

        self.staking_history.push_back(StakingSnapshot {
            total_staked,
            timestamp: current_timestamp(),
        });

        if self.staking_history.len() > 365 {
            self.staking_history.pop_front();
        }
    }

    pub fn average_daily_fees(&self) -> u64 {
        self.avg_daily_fees as u64
    }

    fn fee_trend(&self) -> f64 {
        if self.fee_history.len() < 7 {
            return 0.0;
        }

        let recent: f64 = self.fee_history.iter()
            .rev()
            .take(7)
            .map(|d| d.fees as f64)
            .sum::<f64>() / 7.0;

        let older: f64 = self.fee_history.iter()
            .rev()
            .skip(7)
            .take(7)
            .map(|d| d.fees as f64)
            .sum::<f64>() / 7.0;

        if older > 0.0 {
            (recent - older) / older
        } else {
            0.0
        }
    }

    fn staking_ratio(&self) -> f64 {
        // Estimate based on recent staking levels
        // This would need total supply context
        0.3 // Placeholder - 30% staked
    }
}
```

---

## 6. ML-Driven Adaptive Optimization

### 6.1 Unified ML Controller

```rust
// crates/q-storage/src/ml/unified_controller.rs

/// Central ML controller for all adaptive systems
pub struct UnifiedMLController {
    /// VDF iteration optimizer
    vdf_optimizer: VDFIterationPredictor,

    /// Fee reserve optimizer
    reserve_optimizer: ReserveOptimizer,

    /// Batch size predictor (existing)
    batch_predictor: BatchSizePredictor,

    /// APY predictor
    apy_optimizer: APYOptimizer,

    /// Fee distribution optimizer
    fee_optimizer: FeeDistributionOptimizer,

    /// Global learning rate controller
    learning_controller: AdaptiveLearningRate,
}

impl UnifiedMLController {
    /// Process block and update all models
    pub async fn process_block(&mut self, block_data: &BlockMLData) {
        // Update all models with new data
        self.vdf_optimizer.update(&block_data.vdf_metrics);
        self.reserve_optimizer.update(&block_data.fee_data);
        self.batch_predictor.record_outcome(block_data.sync_outcome.clone());
        self.apy_optimizer.record_daily(block_data.fees, block_data.total_staked);
        self.fee_optimizer.update(&block_data.distribution_outcome);

        // Adjust global learning rate based on prediction accuracy
        self.learning_controller.adjust(&self.get_prediction_accuracies());
    }

    /// Get all predictions for next block
    pub fn predict_next_block(&self, context: &NetworkContext) -> BlockPredictions {
        BlockPredictions {
            vdf_iterations: self.vdf_optimizer.predict(context),
            reserve_action: self.reserve_optimizer.predict(context),
            optimal_batch_size: self.batch_predictor.predict_batch_size(&context.sync_features),
            expected_apy: self.apy_optimizer.predict_apy(
                context.total_staked,
                1.0, // Average multiplier
                context.pending_rewards,
            ),
            fee_distribution: self.fee_optimizer.predict(context),
        }
    }

    fn get_prediction_accuracies(&self) -> Vec<f32> {
        vec![
            self.vdf_optimizer.accuracy(),
            self.reserve_optimizer.accuracy(),
            self.batch_predictor.prediction_quality_ema,
            self.apy_optimizer.accuracy(),
            self.fee_optimizer.accuracy(),
        ]
    }
}
```

### 6.2 Feature Engineering

```rust
/// Features extracted for ML predictions
#[derive(Debug, Clone)]
pub struct NetworkMLFeatures {
    // Market features
    pub market_cap_usd: f64,
    pub price_24h_change: f64,
    pub trading_volume_24h: f64,

    // Network features
    pub block_height: u64,
    pub connected_peers: usize,
    pub network_hashrate: f64,
    pub transaction_rate: f64,

    // Fee features
    pub avg_fee_24h: f64,
    pub fee_volatility: f64,
    pub pending_tx_count: usize,

    // Staking features
    pub total_staked: u64,
    pub staking_ratio: f64,
    pub avg_lock_duration: f64,

    // Security features
    pub vdf_iterations: u64,
    pub reserve_health: f64,
    pub attack_attempts_30d: u32,
}

impl NetworkMLFeatures {
    /// Normalize to [0, 1] for ML models
    pub fn to_tensor(&self) -> [f32; 16] {
        [
            (self.market_cap_usd / 1e12).min(1.0) as f32,
            ((self.price_24h_change + 50.0) / 100.0).clamp(0.0, 1.0) as f32,
            (self.trading_volume_24h / 1e9).min(1.0) as f32,
            (self.block_height as f64 / 1e8).min(1.0) as f32,
            (self.connected_peers as f64 / 1000.0).min(1.0) as f32,
            (self.network_hashrate.log10() / 20.0).clamp(0.0, 1.0) as f32,
            (self.transaction_rate / 10000.0).min(1.0) as f32,
            (self.avg_fee_24h / 1e8).min(1.0) as f32,
            self.fee_volatility.min(1.0) as f32,
            (self.pending_tx_count as f64 / 100000.0).min(1.0) as f32,
            (self.total_staked as f64 / 1e16).min(1.0) as f32,
            self.staking_ratio.min(1.0) as f32,
            (self.avg_lock_duration / 365.0).min(1.0) as f32,
            (self.vdf_iterations as f64 / 1e6).min(1.0) as f32,
            self.reserve_health.min(1.0) as f32,
            (self.attack_attempts_30d as f64 / 100.0).min(1.0) as f32,
        ]
    }
}
```

---

## 7. Decentralization Mechanisms

### 7.1 On-Chain Governance

```rust
// crates/q-vm/src/contracts/governance.rs

/// Decentralized governance for system parameters
pub struct GovernanceContract {
    /// Active proposals
    proposals: HashMap<ProposalId, Proposal>,

    /// Voting power = staked QUG × lock duration multiplier
    voting_power: HashMap<Address, u64>,

    /// Executed changes
    execution_history: Vec<ExecutedProposal>,

    /// Governance parameters
    config: GovernanceConfig,
}

#[derive(Debug, Clone)]
pub struct GovernanceConfig {
    /// Minimum stake to create proposal
    pub min_proposal_stake: u64,

    /// Voting period in blocks
    pub voting_period: u64,

    /// Quorum required (% of voting power)
    pub quorum_percent: u8,

    /// Approval threshold (% of votes)
    pub approval_threshold: u8,

    /// Timelock before execution (blocks)
    pub execution_delay: u64,
}

#[derive(Debug, Clone)]
pub struct Proposal {
    pub id: ProposalId,
    pub proposer: Address,
    pub description: String,
    pub changes: Vec<ParameterChange>,
    pub created_at: u64,
    pub voting_ends_at: u64,
    pub votes_for: u64,
    pub votes_against: u64,
    pub status: ProposalStatus,
}

#[derive(Debug, Clone)]
pub enum ParameterChange {
    /// Change fee distribution ratios
    FeeDistribution {
        miner_share: u8,
        staker_share: u8,
        reserve_share: u8,
    },

    /// Change staking pool parameters
    StakingPool {
        pool_id: PoolId,
        new_boost: f64,
        new_penalty: u8,
    },

    /// Change reserve parameters
    ReserveConfig {
        target_balance: u64,
        max_contribution_rate: u8,
    },

    /// Change VDF parameters
    VDFConfig {
        base_iterations: u64,
        security_multiplier: f64,
    },

    /// Emergency action
    Emergency {
        action: EmergencyAction,
        justification: String,
    },
}

impl GovernanceContract {
    /// Create new proposal
    pub fn create_proposal(
        &mut self,
        proposer: Address,
        description: String,
        changes: Vec<ParameterChange>,
        current_height: u64,
    ) -> Result<ProposalId> {
        // Check proposer has sufficient stake
        let voting_power = self.voting_power.get(&proposer).copied().unwrap_or(0);
        if voting_power < self.config.min_proposal_stake {
            return Err(GovernanceError::InsufficientStake);
        }

        let proposal_id = ProposalId::generate();
        let proposal = Proposal {
            id: proposal_id,
            proposer,
            description,
            changes,
            created_at: current_height,
            voting_ends_at: current_height + self.config.voting_period,
            votes_for: 0,
            votes_against: 0,
            status: ProposalStatus::Active,
        };

        self.proposals.insert(proposal_id, proposal);

        emit_event!(ProposalCreated {
            id: proposal_id,
            proposer,
            voting_ends_at: current_height + self.config.voting_period,
        });

        Ok(proposal_id)
    }

    /// Cast vote
    pub fn vote(
        &mut self,
        voter: Address,
        proposal_id: ProposalId,
        support: bool,
        current_height: u64,
    ) -> Result<()> {
        let proposal = self.proposals.get_mut(&proposal_id)
            .ok_or(GovernanceError::ProposalNotFound)?;

        if current_height >= proposal.voting_ends_at {
            return Err(GovernanceError::VotingEnded);
        }

        let voting_power = self.voting_power.get(&voter).copied().unwrap_or(0);
        if voting_power == 0 {
            return Err(GovernanceError::NoVotingPower);
        }

        if support {
            proposal.votes_for += voting_power;
        } else {
            proposal.votes_against += voting_power;
        }

        emit_event!(VoteCast {
            proposal_id,
            voter,
            support,
            voting_power,
        });

        Ok(())
    }

    /// Execute passed proposal
    pub fn execute_proposal(
        &mut self,
        proposal_id: ProposalId,
        current_height: u64,
    ) -> Result<Vec<ParameterChange>> {
        let proposal = self.proposals.get_mut(&proposal_id)
            .ok_or(GovernanceError::ProposalNotFound)?;

        // Check voting ended
        if current_height < proposal.voting_ends_at {
            return Err(GovernanceError::VotingNotEnded);
        }

        // Check timelock
        if current_height < proposal.voting_ends_at + self.config.execution_delay {
            return Err(GovernanceError::TimelockNotExpired);
        }

        // Check quorum
        let total_votes = proposal.votes_for + proposal.votes_against;
        let total_voting_power: u64 = self.voting_power.values().sum();
        let quorum = (total_votes * 100) / total_voting_power;

        if quorum < self.config.quorum_percent as u64 {
            proposal.status = ProposalStatus::Failed;
            return Err(GovernanceError::QuorumNotMet);
        }

        // Check approval
        let approval = (proposal.votes_for * 100) / total_votes;
        if approval < self.config.approval_threshold as u64 {
            proposal.status = ProposalStatus::Failed;
            return Err(GovernanceError::NotApproved);
        }

        // Execute changes
        proposal.status = ProposalStatus::Executed;

        emit_event!(ProposalExecuted {
            id: proposal_id,
            votes_for: proposal.votes_for,
            votes_against: proposal.votes_against,
        });

        Ok(proposal.changes.clone())
    }
}
```

### 7.2 Validator Selection (Decentralized)

```rust
/// Decentralized validator selection based on stake and reputation
pub struct ValidatorSelector {
    /// Validator scores (stake × reputation × uptime)
    scores: HashMap<Address, ValidatorScore>,

    /// Reputation tracking
    reputation: ReputationTracker,

    /// VRF for random selection
    vrf: VRFInstance,
}

#[derive(Debug, Clone)]
pub struct ValidatorScore {
    pub address: Address,
    pub staked_amount: u64,
    pub reputation_score: f64,  // 0-1
    pub uptime_score: f64,      // 0-1
    pub total_score: f64,
}

impl ValidatorSelector {
    /// Select validators for block validation
    pub fn select_validators(
        &self,
        count: usize,
        vrf_seed: &[u8],
    ) -> Vec<Address> {
        // Weight by total score
        let total_weight: f64 = self.scores.values()
            .map(|s| s.total_score)
            .sum();

        let mut selected = Vec::with_capacity(count);
        let mut remaining_scores = self.scores.clone();

        for i in 0..count {
            // VRF-based random selection weighted by score
            let vrf_output = self.vrf.generate(&[vrf_seed, &[i as u8]].concat());
            let random_point = bytes_to_f64(&vrf_output) * total_weight;

            let mut cumulative = 0.0;
            for (addr, score) in &remaining_scores {
                cumulative += score.total_score;
                if cumulative >= random_point {
                    selected.push(*addr);
                    remaining_scores.remove(addr);
                    break;
                }
            }
        }

        selected
    }

    /// Update validator score after block
    pub fn update_score(&mut self, validator: Address, performance: ValidatorPerformance) {
        if let Some(score) = self.scores.get_mut(&validator) {
            // Update reputation based on performance
            self.reputation.record_performance(&validator, &performance);
            score.reputation_score = self.reputation.get_score(&validator);

            // Update uptime
            score.uptime_score = self.calculate_uptime(&validator);

            // Recalculate total
            score.total_score = (score.staked_amount as f64 / 1e8)
                * score.reputation_score
                * score.uptime_score;
        }
    }
}
```

---

## 8. Frontend UI/UX Design

### 8.1 Staking Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUANTUM WALLET                                    │
│  ┌─────────┐                                                   ┌─────────┐ │
│  │ < Back  │                     STAKING                       │  ⚙️  👤  │ │
│  └─────────┘                                                   └─────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        MY WALLET CARD                                 │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │  ╔═══════════════════════════════════════════════════════════╗ │ │ │
│  │  │  ║  ◈ Q-NARWHALKNIGHT                              ⟐ QUANTUM ║ │ │ │
│  │  │  ║                                                           ║ │ │ │
│  │  │  ║  AVAILABLE BALANCE                                        ║ │ │ │
│  │  │  ║  ┌─────────────────────────────────────────────────────┐  ║ │ │ │
│  │  │  ║  │     ◆ 12,458.75 QUG                                │  ║ │ │ │
│  │  │  ║  │       ≈ $529,496.87 USD                            │  ║ │ │ │
│  │  │  ║  └─────────────────────────────────────────────────────┘  ║ │ │ │
│  │  │  ║                                                           ║ │ │ │
│  │  │  ║  STAKED BALANCE                        PENDING REWARDS   ║ │ │ │
│  │  │  ║  ┌───────────────────┐                 ┌───────────────┐  ║ │ │ │
│  │  │  ║  │ ◆ 5,000.00 QUG   │                 │ ◆ 47.25 QUG   │  ║ │ │ │
│  │  │  ║  │   ≈ $212,500.00  │                 │  ≈ $2,008.12  │  ║ │ │ │
│  │  │  ║  └───────────────────┘                 └───────────────┘  ║ │ │ │
│  │  │  ║                                                           ║ │ │ │
│  │  │  ║  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ║ │ │ │
│  │  │  ║  qug1qnk7x4m5...8f3j2k                    COPY ADDRESS ⧉ ║ │ │ │
│  │  │  ╚═══════════════════════════════════════════════════════════╝ │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                       │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │ │
│  │  │   📤 STAKE     │  │  📥 UNSTAKE    │  │   💰 CLAIM REWARDS     │ │ │
│  │  │                │  │                │  │                        │ │ │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Staking Calculator Component

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAKING CALCULATOR                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  STAKE AMOUNT                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │  ◆ 1,000                                                   QUG │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                       │ │
│  │  ├────────────────●────────────────────────────────────────────────┤ │ │
│  │  MIN                              1,000                          MAX │ │
│  │                                                                       │ │
│  │  Quick amounts:  [100] [500] [1K] [5K] [10K] [MAX]                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  SELECT POOL                                                          │ │
│  │                                                                       │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │ │
│  │  │    FLEXIBLE     │  │    30 DAYS      │  │    90 DAYS      │      │ │
│  │  │                 │  │                 │  │   ★ POPULAR     │      │ │
│  │  │   No Lock       │  │   1.5x Boost    │  │   2.0x Boost    │      │ │
│  │  │   1.0x Boost    │  │   ~3.2% APY     │  │   ~4.3% APY     │      │ │
│  │  │   ~2.1% APY     │  │   10% penalty   │  │   15% penalty   │      │ │
│  │  │   0% penalty    │  │   if early      │  │   if early      │      │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘      │ │
│  │                                                                       │ │
│  │  ┌─────────────────┐                                                 │ │
│  │  │   365 DAYS      │                                                 │ │
│  │  │   ⭐ BEST APY   │                                                 │ │
│  │  │   3.0x Boost    │                                                 │ │
│  │  │   ~6.4% APY     │  ◄─── SELECTED                                 │ │
│  │  │   25% penalty   │                                                 │ │
│  │  │   if early      │                                                 │ │
│  │  └─────────────────┘                                                 │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  📊 PROJECTED EARNINGS                                                │ │
│  │  ─────────────────────────────────────────────────────────────────── │ │
│  │                                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │                                                                 │ │ │
│  │  │    EARNINGS CHART (Interactive)                                 │ │ │
│  │  │                                                                 │ │ │
│  │  │         ╱─────────────────────────────── 1,064 QUG (365d)      │ │ │
│  │  │        ╱                                                        │ │ │
│  │  │       ╱                                                         │ │ │
│  │  │      ╱─────────────────────── 532 QUG (180d)                   │ │ │
│  │  │     ╱                                                           │ │ │
│  │  │    ╱                                                            │ │ │
│  │  │   ╱─────────────── 266 QUG (90d)                               │ │ │
│  │  │  ╱                                                              │ │ │
│  │  │ ╱───── 88.67 QUG (30d)                                         │ │ │
│  │  │╱                                                                │ │ │
│  │  └─┴────────────┴────────────┴────────────┴────────────┴──────────┘ │ │
│  │   0d          90d         180d         270d        365d             │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  📋 SUMMARY                                                           │ │
│  │  ─────────────────────────────────────────────────────────────────── │ │
│  │                                                                       │ │
│  │  Stake Amount:        1,000.00 QUG                                   │ │
│  │  Selected Pool:       365-Day (3.0x boost)                           │ │
│  │  Lock Period:         365 days                                       │ │
│  │  Unlock Date:         December 19, 2025                              │ │
│  │                                                                       │ │
│  │  ─────────────────────────────────────────────────────────────────── │ │
│  │                                                                       │ │
│  │  Estimated APY:       6.4%                                           │ │
│  │  Daily Earnings:      ≈ 0.175 QUG ($7.44)                           │ │
│  │  Monthly Earnings:    ≈ 5.33 QUG ($226.53)                          │ │
│  │  Yearly Earnings:     ≈ 64.00 QUG ($2,720.00)                       │ │
│  │                                                                       │ │
│  │  Early Unstake Fee:   25% (250 QUG if withdrawn before unlock)      │ │
│  │                                                                       │ │
│  │  ─────────────────────────────────────────────────────────────────── │ │
│  │                                                                       │ │
│  │  ⚠️  Yields are estimates based on current network activity.         │ │
│  │     Actual rewards depend on total staked and fee volume.            │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                                                                       │ │
│  │  ╔═══════════════════════════════════════════════════════════════╗   │ │
│  │  ║                    🔒 STAKE 1,000 QUG                         ║   │ │
│  │  ║                                                               ║   │ │
│  │  ║              Lock for 365 days to earn ~6.4% APY              ║   │ │
│  │  ╚═══════════════════════════════════════════════════════════════╝   │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 React Component Implementation

```typescript
// gui/quantum-wallet/src/components/StakingDashboard.tsx

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';

interface StakingPool {
  id: string;
  name: string;
  lockDays: number;
  boostMultiplier: number;
  estimatedApy: number;
  earlyPenalty: number;
  isPopular?: boolean;
  isBestApy?: boolean;
}

interface StakePosition {
  id: string;
  poolId: string;
  amount: number;
  stakedAt: number;
  unlockAt: number;
  pendingRewards: number;
}

interface StakingDashboardProps {
  walletAddress: string;
  availableBalance: number;
  qugPrice: number;
}

const STAKING_POOLS: StakingPool[] = [
  {
    id: 'flexible',
    name: 'Flexible',
    lockDays: 0,
    boostMultiplier: 1.0,
    estimatedApy: 2.1,
    earlyPenalty: 0,
  },
  {
    id: '30day',
    name: '30 Days',
    lockDays: 30,
    boostMultiplier: 1.5,
    estimatedApy: 3.2,
    earlyPenalty: 10,
  },
  {
    id: '90day',
    name: '90 Days',
    lockDays: 90,
    boostMultiplier: 2.0,
    estimatedApy: 4.3,
    earlyPenalty: 15,
    isPopular: true,
  },
  {
    id: '365day',
    name: '365 Days',
    lockDays: 365,
    boostMultiplier: 3.0,
    estimatedApy: 6.4,
    earlyPenalty: 25,
    isBestApy: true,
  },
];

export const StakingDashboard: React.FC<StakingDashboardProps> = ({
  walletAddress,
  availableBalance,
  qugPrice,
}) => {
  const [stakeAmount, setStakeAmount] = useState<number>(1000);
  const [selectedPool, setSelectedPool] = useState<StakingPool>(STAKING_POOLS[3]);
  const [positions, setPositions] = useState<StakePosition[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Calculate projected earnings
  const projectedEarnings = useMemo(() => {
    const dailyRate = selectedPool.estimatedApy / 365 / 100;
    const data = [];

    for (let day = 0; day <= 365; day += 30) {
      const earnings = stakeAmount * dailyRate * day;
      data.push({
        day,
        earnings: parseFloat(earnings.toFixed(2)),
        earningsUsd: parseFloat((earnings * qugPrice).toFixed(2)),
      });
    }

    return data;
  }, [stakeAmount, selectedPool, qugPrice]);

  // Summary calculations
  const summary = useMemo(() => {
    const dailyEarnings = (stakeAmount * selectedPool.estimatedApy / 365 / 100);
    const monthlyEarnings = dailyEarnings * 30;
    const yearlyEarnings = dailyEarnings * 365;

    return {
      dailyEarnings,
      dailyEarningsUsd: dailyEarnings * qugPrice,
      monthlyEarnings,
      monthlyEarningsUsd: monthlyEarnings * qugPrice,
      yearlyEarnings,
      yearlyEarningsUsd: yearlyEarnings * qugPrice,
      unlockDate: new Date(Date.now() + selectedPool.lockDays * 24 * 60 * 60 * 1000),
      penaltyAmount: stakeAmount * selectedPool.earlyPenalty / 100,
    };
  }, [stakeAmount, selectedPool, qugPrice]);

  // Handle stake
  const handleStake = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/staking/stake', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          amount: Math.floor(stakeAmount * 1e8), // Convert to satoshis
          pool_id: selectedPool.id,
        }),
      });

      if (response.ok) {
        // Refresh positions
        await fetchPositions();
      }
    } catch (error) {
      console.error('Staking failed:', error);
    }
    setIsLoading(false);
  };

  const fetchPositions = async () => {
    try {
      const response = await fetch(`/api/v1/staking/positions/${walletAddress}`);
      const data = await response.json();
      setPositions(data.positions || []);
    } catch (error) {
      console.error('Failed to fetch positions:', error);
    }
  };

  useEffect(() => {
    fetchPositions();
  }, [walletAddress]);

  return (
    <div className="staking-dashboard">
      {/* Wallet Card */}
      <motion.div
        className="wallet-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="wallet-card-inner">
          <div className="wallet-header">
            <span className="logo">◈ Q-NARWHALKNIGHT</span>
            <span className="network">⟐ QUANTUM</span>
          </div>

          <div className="balance-section">
            <div className="balance-label">AVAILABLE BALANCE</div>
            <div className="balance-amount">
              <span className="symbol">◆</span>
              <span className="value">{availableBalance.toLocaleString()}</span>
              <span className="currency">QUG</span>
            </div>
            <div className="balance-usd">
              ≈ ${(availableBalance * qugPrice).toLocaleString()}
            </div>
          </div>

          <div className="staking-balances">
            <div className="staked-balance">
              <div className="label">STAKED BALANCE</div>
              <div className="amount">◆ {positions.reduce((sum, p) => sum + p.amount / 1e8, 0).toLocaleString()} QUG</div>
            </div>
            <div className="pending-rewards">
              <div className="label">PENDING REWARDS</div>
              <div className="amount">◆ {positions.reduce((sum, p) => sum + p.pendingRewards / 1e8, 0).toFixed(2)} QUG</div>
            </div>
          </div>

          <div className="wallet-address">
            <code>{walletAddress.slice(0, 12)}...{walletAddress.slice(-8)}</code>
            <button className="copy-btn">⧉</button>
          </div>
        </div>
      </motion.div>

      {/* Staking Calculator */}
      <motion.div
        className="staking-calculator"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h2>STAKING CALCULATOR</h2>

        {/* Amount Input */}
        <div className="stake-amount-section">
          <label>STAKE AMOUNT</label>
          <div className="amount-input">
            <span className="symbol">◆</span>
            <input
              type="number"
              value={stakeAmount}
              onChange={(e) => setStakeAmount(parseFloat(e.target.value) || 0)}
              max={availableBalance}
            />
            <span className="currency">QUG</span>
          </div>

          <input
            type="range"
            min={100}
            max={availableBalance}
            value={stakeAmount}
            onChange={(e) => setStakeAmount(parseFloat(e.target.value))}
            className="amount-slider"
          />

          <div className="quick-amounts">
            {[100, 500, 1000, 5000, 10000].map(amount => (
              <button
                key={amount}
                onClick={() => setStakeAmount(Math.min(amount, availableBalance))}
                className={stakeAmount === amount ? 'active' : ''}
              >
                {amount >= 1000 ? `${amount / 1000}K` : amount}
              </button>
            ))}
            <button onClick={() => setStakeAmount(availableBalance)}>MAX</button>
          </div>
        </div>

        {/* Pool Selection */}
        <div className="pool-selection">
          <label>SELECT POOL</label>
          <div className="pools-grid">
            {STAKING_POOLS.map(pool => (
              <motion.div
                key={pool.id}
                className={`pool-card ${selectedPool.id === pool.id ? 'selected' : ''}`}
                onClick={() => setSelectedPool(pool)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="pool-name">{pool.name}</div>
                {pool.isPopular && <span className="badge popular">★ POPULAR</span>}
                {pool.isBestApy && <span className="badge best">⭐ BEST APY</span>}

                <div className="pool-details">
                  <div className="boost">{pool.boostMultiplier}x Boost</div>
                  <div className="apy">~{pool.estimatedApy}% APY</div>
                  <div className="penalty">
                    {pool.earlyPenalty > 0
                      ? `${pool.earlyPenalty}% penalty if early`
                      : 'No penalty'}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Earnings Chart */}
        <div className="earnings-chart">
          <h3>📊 PROJECTED EARNINGS</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={projectedEarnings}>
              <defs>
                <linearGradient id="earningsGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis
                dataKey="day"
                tickFormatter={(day) => `${day}d`}
                stroke="#666"
              />
              <YAxis
                tickFormatter={(val) => `${val} QUG`}
                stroke="#666"
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="chart-tooltip">
                        <div>Day {payload[0].payload.day}</div>
                        <div>◆ {payload[0].payload.earnings} QUG</div>
                        <div>${payload[0].payload.earningsUsd}</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Area
                type="monotone"
                dataKey="earnings"
                stroke="#00d4ff"
                strokeWidth={2}
                fill="url(#earningsGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Summary */}
        <div className="staking-summary">
          <h3>📋 SUMMARY</h3>

          <div className="summary-row">
            <span>Stake Amount:</span>
            <span>{stakeAmount.toLocaleString()} QUG</span>
          </div>
          <div className="summary-row">
            <span>Selected Pool:</span>
            <span>{selectedPool.name} ({selectedPool.boostMultiplier}x boost)</span>
          </div>
          <div className="summary-row">
            <span>Lock Period:</span>
            <span>{selectedPool.lockDays} days</span>
          </div>
          <div className="summary-row">
            <span>Unlock Date:</span>
            <span>{summary.unlockDate.toLocaleDateString()}</span>
          </div>

          <div className="summary-divider" />

          <div className="summary-row highlight">
            <span>Estimated APY:</span>
            <span className="apy-value">{selectedPool.estimatedApy}%</span>
          </div>
          <div className="summary-row">
            <span>Daily Earnings:</span>
            <span>≈ {summary.dailyEarnings.toFixed(3)} QUG (${summary.dailyEarningsUsd.toFixed(2)})</span>
          </div>
          <div className="summary-row">
            <span>Monthly Earnings:</span>
            <span>≈ {summary.monthlyEarnings.toFixed(2)} QUG (${summary.monthlyEarningsUsd.toFixed(2)})</span>
          </div>
          <div className="summary-row">
            <span>Yearly Earnings:</span>
            <span>≈ {summary.yearlyEarnings.toFixed(2)} QUG (${summary.yearlyEarningsUsd.toFixed(2)})</span>
          </div>

          {selectedPool.earlyPenalty > 0 && (
            <div className="early-penalty-warning">
              ⚠️ Early Unstake Fee: {selectedPool.earlyPenalty}% ({summary.penaltyAmount.toFixed(2)} QUG if withdrawn before unlock)
            </div>
          )}

          <div className="disclaimer">
            ⚠️ Yields are estimates based on current network activity.
            Actual rewards depend on total staked and fee volume.
          </div>
        </div>

        {/* Stake Button */}
        <motion.button
          className="stake-button"
          onClick={handleStake}
          disabled={isLoading || stakeAmount <= 0 || stakeAmount > availableBalance}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {isLoading ? (
            <span className="loading">Processing...</span>
          ) : (
            <>
              🔒 STAKE {stakeAmount.toLocaleString()} QUG
              <div className="button-subtitle">
                Lock for {selectedPool.lockDays} days to earn ~{selectedPool.estimatedApy}% APY
              </div>
            </>
          )}
        </motion.button>
      </motion.div>
    </div>
  );
};
```

### 8.4 Staking Dashboard CSS

```css
/* gui/quantum-wallet/src/styles/staking.css */

.staking-dashboard {
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Wallet Card */
.wallet-card {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border-radius: 20px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(0, 212, 255, 0.2);
}

.wallet-card-inner {
  position: relative;
}

.wallet-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.wallet-header .logo {
  font-size: 14px;
  font-weight: 600;
  color: #00d4ff;
  letter-spacing: 1px;
}

.wallet-header .network {
  font-size: 12px;
  color: #888;
  background: rgba(0, 212, 255, 0.1);
  padding: 4px 12px;
  border-radius: 12px;
}

.balance-section {
  text-align: center;
  margin-bottom: 24px;
}

.balance-label {
  font-size: 11px;
  color: #888;
  letter-spacing: 2px;
  margin-bottom: 8px;
}

.balance-amount {
  font-size: 36px;
  font-weight: 700;
  color: #fff;
}

.balance-amount .symbol {
  color: #00d4ff;
  margin-right: 8px;
}

.balance-usd {
  font-size: 14px;
  color: #888;
  margin-top: 4px;
}

.staking-balances {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}

.staking-balances > div {
  background: rgba(255, 255, 255, 0.05);
  padding: 16px;
  border-radius: 12px;
}

.staking-balances .label {
  font-size: 10px;
  color: #888;
  letter-spacing: 1px;
  margin-bottom: 8px;
}

.staking-balances .amount {
  font-size: 16px;
  font-weight: 600;
  color: #fff;
}

.wallet-address {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
}

.wallet-address code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: #888;
}

.copy-btn {
  background: none;
  border: none;
  color: #00d4ff;
  cursor: pointer;
  font-size: 14px;
}

/* Staking Calculator */
.staking-calculator {
  background: #1a1a2e;
  border-radius: 20px;
  padding: 24px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.staking-calculator h2 {
  font-size: 18px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 24px;
  letter-spacing: 1px;
}

/* Amount Input */
.stake-amount-section {
  margin-bottom: 24px;
}

.stake-amount-section label {
  display: block;
  font-size: 11px;
  color: #888;
  letter-spacing: 1px;
  margin-bottom: 12px;
}

.amount-input {
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 16px;
}

.amount-input .symbol {
  color: #00d4ff;
  font-size: 20px;
  margin-right: 12px;
}

.amount-input input {
  flex: 1;
  background: none;
  border: none;
  color: #fff;
  font-size: 24px;
  font-weight: 600;
  outline: none;
}

.amount-input .currency {
  color: #888;
  font-size: 14px;
}

.amount-slider {
  width: 100%;
  margin: 16px 0;
  -webkit-appearance: none;
  height: 4px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
}

.amount-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 20px;
  height: 20px;
  background: #00d4ff;
  border-radius: 50%;
  cursor: pointer;
}

.quick-amounts {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.quick-amounts button {
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  color: #888;
  cursor: pointer;
  transition: all 0.2s;
}

.quick-amounts button:hover,
.quick-amounts button.active {
  background: rgba(0, 212, 255, 0.2);
  border-color: #00d4ff;
  color: #00d4ff;
}

/* Pool Selection */
.pool-selection {
  margin-bottom: 24px;
}

.pool-selection label {
  display: block;
  font-size: 11px;
  color: #888;
  letter-spacing: 1px;
  margin-bottom: 12px;
}

.pools-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.pool-card {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 16px;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.pool-card:hover {
  background: rgba(255, 255, 255, 0.05);
  border-color: rgba(0, 212, 255, 0.3);
}

.pool-card.selected {
  background: rgba(0, 212, 255, 0.1);
  border-color: #00d4ff;
}

.pool-card .pool-name {
  font-size: 16px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 8px;
}

.pool-card .badge {
  position: absolute;
  top: 12px;
  right: 12px;
  font-size: 9px;
  padding: 2px 6px;
  border-radius: 4px;
}

.pool-card .badge.popular {
  background: rgba(255, 193, 7, 0.2);
  color: #ffc107;
}

.pool-card .badge.best {
  background: rgba(76, 175, 80, 0.2);
  color: #4caf50;
}

.pool-details {
  font-size: 12px;
  color: #888;
  line-height: 1.6;
}

.pool-details .boost {
  color: #00d4ff;
}

.pool-details .apy {
  color: #4caf50;
  font-weight: 600;
}

/* Earnings Chart */
.earnings-chart {
  margin-bottom: 24px;
}

.earnings-chart h3 {
  font-size: 14px;
  color: #888;
  margin-bottom: 16px;
}

.chart-tooltip {
  background: #1a1a2e;
  border: 1px solid rgba(0, 212, 255, 0.3);
  border-radius: 8px;
  padding: 12px;
  font-size: 12px;
  color: #fff;
}

/* Summary */
.staking-summary {
  background: rgba(255, 255, 255, 0.03);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 24px;
}

.staking-summary h3 {
  font-size: 14px;
  color: #888;
  margin-bottom: 16px;
}

.summary-row {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  font-size: 13px;
  color: #ccc;
}

.summary-row.highlight {
  font-weight: 600;
}

.summary-row .apy-value {
  color: #4caf50;
  font-size: 18px;
}

.summary-divider {
  height: 1px;
  background: rgba(255, 255, 255, 0.1);
  margin: 16px 0;
}

.early-penalty-warning {
  background: rgba(255, 152, 0, 0.1);
  border: 1px solid rgba(255, 152, 0, 0.3);
  border-radius: 8px;
  padding: 12px;
  font-size: 12px;
  color: #ff9800;
  margin-top: 16px;
}

.disclaimer {
  font-size: 11px;
  color: #666;
  margin-top: 16px;
  padding: 12px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
}

/* Stake Button */
.stake-button {
  width: 100%;
  padding: 20px;
  background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
  border: none;
  border-radius: 12px;
  color: #fff;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.stake-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 10px 20px rgba(0, 212, 255, 0.3);
}

.stake-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.stake-button .button-subtitle {
  font-size: 12px;
  font-weight: 400;
  opacity: 0.8;
  margin-top: 4px;
}
```

---

## 9. Security Analysis

### 9.1 Attack Vectors & Mitigations

| Attack Vector | Mitigation |
|---------------|------------|
| **Stake Grinding** | VRF-based validator selection, reputation decay |
| **Long-Range Attack** | VDF time-lock prevents pre-computation |
| **Reserve Drain** | Max withdrawal rate (1% per block), governance timelock |
| **Flash Loan Staking** | Minimum lock period, snapshot-based voting |
| **Governance Attack** | Quorum requirements, execution delay, emergency pause |
| **MEV Extraction** | Staker-weighted transaction ordering |

### 9.2 Economic Security

```
At $1B market cap with 30% staked:
- Staked value: $300M
- Annual fee revenue (1% of tx volume): ~$10M
- Staker APY: 10M / 300M = 3.3%
- Attack cost: 300M × 1.5 (slashing) = $450M
- Attack return: Must exceed $450M to be profitable
```

---

## 10. Implementation Roadmap

### Phase 1: Fee Infrastructure (2 weeks)
- [ ] Fee collection contract
- [ ] Fee distribution engine
- [ ] Fee smoothing reserve

### Phase 2: Staking Core (3 weeks)
- [ ] Staking contract with pools
- [ ] Position management
- [ ] Reward distribution

### Phase 3: ML Integration (2 weeks)
- [ ] APY optimizer
- [ ] Reserve optimizer
- [ ] Unified ML controller

### Phase 4: Governance (2 weeks)
- [ ] Proposal system
- [ ] Voting mechanism
- [ ] Execution timelock

### Phase 5: Frontend (2 weeks)
- [ ] Staking dashboard
- [ ] Calculator component
- [ ] Earnings visualization

### Phase 6: Testing & Audit (3 weeks)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Security audit
- [ ] Mainnet deployment

---

## Appendix A: API Endpoints

```
POST   /api/v1/staking/stake          - Stake QUG
POST   /api/v1/staking/unstake        - Unstake position
POST   /api/v1/staking/claim          - Claim rewards
GET    /api/v1/staking/positions/:addr - Get positions
GET    /api/v1/staking/pools          - List pools
GET    /api/v1/staking/apy/:pool      - Get estimated APY
GET    /api/v1/staking/stats          - Global staking stats
GET    /api/v1/reserve/health         - Reserve health
GET    /api/v1/governance/proposals   - Active proposals
POST   /api/v1/governance/vote        - Cast vote
```

---

**Document Version**: 1.5.0-beta
**Last Updated**: December 2024
**Status**: Ready for Implementation Review
