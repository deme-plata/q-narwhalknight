// ═══════════════════════════════════════════════════════════════════════════════
// WATER ROBOT FINANCIAL INTELLIGENCE SYSTEM
// K-Law Adoption Monitoring for Q-NarwhalKnight Native Coin (QNK)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Based on the Kristensen K-Law: A*_t = K / (1 + μe^(-λΩ_t))
//
// Where:
//   A*_t = Equilibrium adoption ceiling at time t
//   K = Carrying capacity (maximum adoption ~1.0 or 100%)
//   μ = Initial adoption friction coefficient
//   λ = Flow sensitivity parameter
//   Ω_t = Composite flow density at time t
//
// For QNK, we adapt the Bitcoin flow components to:
//   Ω_t = w₁·F_Staking + w₂·F_DeFi + w₃·F_Treasury + w₄·S_Unlock + w₅·ΔX_Exchange
//
// ═══════════════════════════════════════════════════════════════════════════════

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn, error};

/// K-Law parameters for QNK adoption model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KLawParameters {
    /// Carrying capacity K (maximum adoption, typically 1.0)
    pub carrying_capacity: f64,
    /// Initial friction coefficient μ
    pub friction_mu: f64,
    /// Flow sensitivity λ
    pub flow_sensitivity_lambda: f64,
    /// Flow component weights
    pub weights: FlowWeights,
}

impl Default for KLawParameters {
    fn default() -> Self {
        Self {
            carrying_capacity: 1.0,           // 100% maximum adoption
            friction_mu: 150.0,                // High initial friction (early stage)
            flow_sensitivity_lambda: 0.08,     // Moderate flow sensitivity
            weights: FlowWeights::default(),
        }
    }
}

/// Weights for QNK-specific flow components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowWeights {
    /// Weight for staking flows
    pub staking: f64,
    /// Weight for DeFi TVL flows
    pub defi: f64,
    /// Weight for treasury/DAO flows
    pub treasury: f64,
    /// Weight for token unlock schedule
    pub unlock_schedule: f64,
    /// Weight for exchange net flows
    pub exchange: f64,
}

impl Default for FlowWeights {
    fn default() -> Self {
        Self {
            staking: 0.30,      // 30% weight - staking is crucial for PoS
            defi: 0.25,         // 25% weight - DeFi TVL indicates utility
            treasury: 0.20,     // 20% weight - treasury health
            unlock_schedule: 0.15, // 15% weight - supply dynamics
            exchange: 0.10,     // 10% weight - liquidity & accessibility
        }
    }
}

/// QNK Flow Density components at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QNKFlowDensity {
    /// Timestamp of measurement
    pub timestamp: u64,
    /// Staking flow (normalized 0-1)
    pub staking_flow: f64,
    /// DeFi TVL flow (normalized 0-1)
    pub defi_flow: f64,
    /// Treasury/DAO flow (normalized 0-1)
    pub treasury_flow: f64,
    /// Token unlock schedule impact (normalized 0-1)
    pub unlock_flow: f64,
    /// Exchange net flow (normalized -1 to 1, positive = inflow)
    pub exchange_flow: f64,
    /// Composite flow density Ω_t
    pub composite_omega: f64,
}

impl QNKFlowDensity {
    /// Calculate composite flow density using weights
    pub fn calculate_composite(&mut self, weights: &FlowWeights) {
        self.composite_omega =
            weights.staking * self.staking_flow +
            weights.defi * self.defi_flow +
            weights.treasury * self.treasury_flow +
            weights.unlock_schedule * self.unlock_flow +
            weights.exchange * (self.exchange_flow + 1.0) / 2.0; // Normalize -1..1 to 0..1
    }
}

/// Three-layer adoption framework for QNK
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeLayerAdoption {
    /// Layer 1: Savings/Staking (50% weight) - Long-term holders
    pub layer1_savings: f64,
    /// Layer 2: Settlement/Payments (30% weight) - Transaction utility
    pub layer2_settlement: f64,
    /// Layer 3: Collateral/DeFi (20% weight) - Financial infrastructure
    pub layer3_collateral: f64,
    /// Composite adoption score A_t
    pub composite_adoption: f64,
}

impl ThreeLayerAdoption {
    /// Calculate composite adoption from three layers
    pub fn calculate_composite(&mut self) {
        self.composite_adoption =
            0.50 * self.layer1_savings +
            0.30 * self.layer2_settlement +
            0.20 * self.layer3_collateral;
    }
}

/// Kristensen Ratio - Health gauge for adoption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KristensenRatio {
    /// Current adoption A_t
    pub current_adoption: f64,
    /// Equilibrium adoption ceiling A*_t
    pub equilibrium_ceiling: f64,
    /// Kristensen ratio K_t = A_t / A*_t
    pub ratio: f64,
    /// Health assessment
    pub health: AdoptionHealth,
    /// Timestamp
    pub timestamp: u64,
}

/// Adoption health status based on Kristensen Ratio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdoptionHealth {
    /// K_t > 1.1 - Adoption exceeds equilibrium (potential correction)
    Overheated,
    /// K_t in [0.95, 1.1] - Healthy adoption tracking equilibrium
    Healthy,
    /// K_t in [0.7, 0.95] - Lagging but catching up
    Recovering,
    /// K_t in [0.5, 0.7] - Significant underadoption
    Underperforming,
    /// K_t < 0.5 - Critical underadoption
    Critical,
}

impl AdoptionHealth {
    pub fn from_ratio(ratio: f64) -> Self {
        if ratio > 1.1 {
            Self::Overheated
        } else if ratio >= 0.95 {
            Self::Healthy
        } else if ratio >= 0.7 {
            Self::Recovering
        } else if ratio >= 0.5 {
            Self::Underperforming
        } else {
            Self::Critical
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Overheated => "🔥",
            Self::Healthy => "✅",
            Self::Recovering => "📈",
            Self::Underperforming => "⚠️",
            Self::Critical => "🚨",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Overheated => "Adoption exceeds equilibrium - potential correction ahead",
            Self::Healthy => "Adoption tracking equilibrium - optimal state",
            Self::Recovering => "Adoption lagging but momentum positive",
            Self::Underperforming => "Significant gap to equilibrium - action needed",
            Self::Critical => "Critical underadoption - ecosystem risk",
        }
    }
}

/// Financial Intelligence Robot Roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinancialRobotRole {
    /// Flow Monitor - Tracks Ω_t flow components (QuantumJellyfish)
    FlowMonitor {
        monitored_flows: Vec<String>,
        update_interval_secs: u64,
    },
    /// Holder Analyst - Tracks holder distribution (EntangledDolphin)
    HolderAnalyst {
        cohort_boundaries: Vec<f64>,
        track_whale_movements: bool,
    },
    /// On-Chain Oracle - Analyzes transaction patterns (TunnelingOctopus)
    OnChainOracle {
        metrics: Vec<String>,
        anomaly_detection: bool,
    },
    /// Whale Watcher - Large holder monitoring (WaveParticleWhale)
    WhaleWatcher {
        whale_threshold_qnk: f64,
        alert_on_movement: bool,
    },
    /// Adoption Tracker - Computes K-Law metrics (CyberCetus)
    AdoptionTracker {
        k_law_params: KLawParameters,
        three_layer_enabled: bool,
    },
    /// Swarm Coordinator - Aggregates financial intelligence (SchoolingRobotichthys)
    SwarmCoordinator {
        aggregation_method: String,
        consensus_threshold: f64,
    },
}

/// Financial metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialSnapshot {
    /// Block height at snapshot
    pub block_height: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Total QNK supply
    pub total_supply: f64,
    /// Circulating supply
    pub circulating_supply: f64,
    /// Total staked QNK
    pub total_staked: f64,
    /// Staking percentage
    pub staking_percentage: f64,
    /// Total holders count
    pub total_holders: u64,
    /// Active addresses (24h)
    pub active_addresses_24h: u64,
    /// Transaction count (24h)
    pub transactions_24h: u64,
    /// Average transaction value
    pub avg_transaction_value: f64,
    /// DeFi TVL in QNK
    pub defi_tvl: f64,
    /// Treasury balance
    pub treasury_balance: f64,
    /// Exchange reserves
    pub exchange_reserves: f64,
}

/// Holder distribution cohorts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolderDistribution {
    /// Timestamp
    pub timestamp: u64,
    /// Shrimp: < 1 QNK
    pub shrimp_count: u64,
    pub shrimp_total_balance: f64,
    /// Crab: 1-10 QNK
    pub crab_count: u64,
    pub crab_total_balance: f64,
    /// Fish: 10-100 QNK
    pub fish_count: u64,
    pub fish_total_balance: f64,
    /// Dolphin: 100-1,000 QNK
    pub dolphin_count: u64,
    pub dolphin_total_balance: f64,
    /// Whale: 1,000-10,000 QNK
    pub whale_count: u64,
    pub whale_total_balance: f64,
    /// Mega Whale: > 10,000 QNK
    pub mega_whale_count: u64,
    pub mega_whale_total_balance: f64,
    /// Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    pub gini_coefficient: f64,
}

/// Adoption checkpoint for falsifiable predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdoptionCheckpoint {
    /// Target date (Unix timestamp)
    pub target_date: u64,
    /// Target year (for display)
    pub target_year: f64,
    /// Predicted adoption percentage
    pub predicted_adoption: f64,
    /// Actual adoption (filled when date passes)
    pub actual_adoption: Option<f64>,
    /// Predicted holder count
    pub predicted_holders: u64,
    /// Actual holders (filled when date passes)
    pub actual_holders: Option<u64>,
    /// Predicted QNK price in USD
    pub predicted_price_usd: Option<f64>,
    /// Checkpoint status
    pub status: CheckpointStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointStatus {
    Future,
    Active,
    Met,
    Missed,
    Exceeded,
}

/// Main Financial Intelligence Engine
pub struct FinancialIntelligenceEngine {
    /// K-Law parameters
    pub k_law_params: KLawParameters,
    /// Historical flow density data
    pub flow_history: Vec<QNKFlowDensity>,
    /// Historical adoption data
    pub adoption_history: Vec<ThreeLayerAdoption>,
    /// Kristensen ratio history
    pub kristensen_history: Vec<KristensenRatio>,
    /// Current financial snapshot
    pub current_snapshot: Option<FinancialSnapshot>,
    /// Holder distribution
    pub holder_distribution: Option<HolderDistribution>,
    /// Adoption checkpoints
    pub checkpoints: Vec<AdoptionCheckpoint>,
    /// Assigned robot roles
    pub robot_roles: HashMap<String, FinancialRobotRole>,
}

impl FinancialIntelligenceEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            k_law_params: KLawParameters::default(),
            flow_history: Vec::new(),
            adoption_history: Vec::new(),
            kristensen_history: Vec::new(),
            current_snapshot: None,
            holder_distribution: None,
            checkpoints: Vec::new(),
            robot_roles: HashMap::new(),
        };

        // Initialize default checkpoints based on K-Law projections
        engine.initialize_checkpoints();
        engine
    }

    /// Calculate K-Law equilibrium adoption ceiling
    /// A*_t = K / (1 + μ·e^(-λ·Ω_t))
    pub fn calculate_equilibrium_ceiling(&self, omega: f64) -> f64 {
        let k = self.k_law_params.carrying_capacity;
        let mu = self.k_law_params.friction_mu;
        let lambda = self.k_law_params.flow_sensitivity_lambda;

        k / (1.0 + mu * (-lambda * omega).exp())
    }

    /// Calculate critical flow density where adoption accelerates
    /// Ω^crit = ln(μ) / λ
    pub fn calculate_critical_flow_density(&self) -> f64 {
        let mu = self.k_law_params.friction_mu;
        let lambda = self.k_law_params.flow_sensitivity_lambda;

        mu.ln() / lambda
    }

    /// Calculate Kristensen Ratio from current adoption and flow
    pub fn calculate_kristensen_ratio(&self, current_adoption: f64, omega: f64) -> KristensenRatio {
        let equilibrium = self.calculate_equilibrium_ceiling(omega);
        let ratio = if equilibrium > 0.0 {
            current_adoption / equilibrium
        } else {
            0.0
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        KristensenRatio {
            current_adoption,
            equilibrium_ceiling: equilibrium,
            ratio,
            health: AdoptionHealth::from_ratio(ratio),
            timestamp,
        }
    }

    /// Update flow density from current metrics
    pub fn update_flow_density(&mut self, snapshot: &FinancialSnapshot) -> QNKFlowDensity {
        let timestamp = snapshot.timestamp;

        // Normalize metrics to 0-1 scale
        // Staking flow: % of supply staked
        let staking_flow = snapshot.staking_percentage / 100.0;

        // DeFi flow: TVL as % of circulating supply (capped at 1.0)
        let defi_flow = (snapshot.defi_tvl / snapshot.circulating_supply).min(1.0);

        // Treasury flow: Treasury as % of total supply
        let treasury_flow = (snapshot.treasury_balance / snapshot.total_supply).min(1.0);

        // Unlock flow: inverse of circulating ratio (more locked = higher flow)
        let unlock_flow = 1.0 - (snapshot.circulating_supply / snapshot.total_supply);

        // Exchange flow: normalized exchange reserves change
        // Negative = outflow (good for adoption), Positive = inflow (selling pressure)
        let exchange_flow = if let Some(prev) = self.flow_history.last() {
            let prev_exchange = prev.exchange_flow;
            let current_ratio = snapshot.exchange_reserves / snapshot.circulating_supply;
            (prev_exchange - current_ratio).clamp(-1.0, 1.0)
        } else {
            0.0 // Neutral on first measurement
        };

        let mut flow = QNKFlowDensity {
            timestamp,
            staking_flow,
            defi_flow,
            treasury_flow,
            unlock_flow,
            exchange_flow,
            composite_omega: 0.0,
        };

        flow.calculate_composite(&self.k_law_params.weights);

        self.flow_history.push(flow.clone());
        flow
    }

    /// Update three-layer adoption from snapshot
    pub fn update_three_layer_adoption(&mut self, snapshot: &FinancialSnapshot) -> ThreeLayerAdoption {
        // Layer 1: Savings/Staking - measured by staking ratio
        let layer1_savings = snapshot.staking_percentage / 100.0;

        // Layer 2: Settlement/Payments - measured by transaction activity
        // Normalize by assuming 100k daily txs = full adoption
        let layer2_settlement = (snapshot.transactions_24h as f64 / 100_000.0).min(1.0);

        // Layer 3: Collateral/DeFi - measured by DeFi TVL ratio
        let layer3_collateral = (snapshot.defi_tvl / snapshot.circulating_supply).min(1.0);

        let mut adoption = ThreeLayerAdoption {
            layer1_savings,
            layer2_settlement,
            layer3_collateral,
            composite_adoption: 0.0,
        };

        adoption.calculate_composite();

        self.adoption_history.push(adoption.clone());
        adoption
    }

    /// Assign robot role for financial monitoring
    pub fn assign_robot_role(&mut self, robot_id: &str, role: FinancialRobotRole) {
        info!("Assigning financial role to robot {}: {:?}", robot_id, role);
        self.robot_roles.insert(robot_id.to_string(), role);
    }

    /// Get robots by role type
    pub fn get_robots_by_role(&self, role_type: &str) -> Vec<&str> {
        self.robot_roles.iter()
            .filter(|(_, role)| {
                match (role_type, role) {
                    ("flow", FinancialRobotRole::FlowMonitor { .. }) => true,
                    ("holder", FinancialRobotRole::HolderAnalyst { .. }) => true,
                    ("onchain", FinancialRobotRole::OnChainOracle { .. }) => true,
                    ("whale", FinancialRobotRole::WhaleWatcher { .. }) => true,
                    ("adoption", FinancialRobotRole::AdoptionTracker { .. }) => true,
                    ("coordinator", FinancialRobotRole::SwarmCoordinator { .. }) => true,
                    _ => false,
                }
            })
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Initialize adoption checkpoints with falsifiable predictions
    fn initialize_checkpoints(&mut self) {
        // QNK Adoption Checkpoints (inspired by K-Law Bitcoin projections)
        // Adjusted for a new blockchain launch timeline

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Year 1: Early adopters
        self.checkpoints.push(AdoptionCheckpoint {
            target_date: now + 365 * 24 * 3600,
            target_year: 2027.0,
            predicted_adoption: 0.05,  // 5% of target market
            actual_adoption: None,
            predicted_holders: 10_000,
            actual_holders: None,
            predicted_price_usd: None,
            status: CheckpointStatus::Future,
        });

        // Year 2: Initial growth
        self.checkpoints.push(AdoptionCheckpoint {
            target_date: now + 2 * 365 * 24 * 3600,
            target_year: 2028.0,
            predicted_adoption: 0.15,  // 15%
            actual_adoption: None,
            predicted_holders: 50_000,
            actual_holders: None,
            predicted_price_usd: None,
            status: CheckpointStatus::Future,
        });

        // Year 3: Acceleration phase
        self.checkpoints.push(AdoptionCheckpoint {
            target_date: now + 3 * 365 * 24 * 3600,
            target_year: 2029.0,
            predicted_adoption: 0.35,  // 35%
            actual_adoption: None,
            predicted_holders: 200_000,
            actual_holders: None,
            predicted_price_usd: None,
            status: CheckpointStatus::Future,
        });

        // Year 5: Mainstream adoption
        self.checkpoints.push(AdoptionCheckpoint {
            target_date: now + 5 * 365 * 24 * 3600,
            target_year: 2031.0,
            predicted_adoption: 0.60,  // 60%
            actual_adoption: None,
            predicted_holders: 1_000_000,
            actual_holders: None,
            predicted_price_usd: None,
            status: CheckpointStatus::Future,
        });

        // Year 10: Maturity
        self.checkpoints.push(AdoptionCheckpoint {
            target_date: now + 10 * 365 * 24 * 3600,
            target_year: 2036.0,
            predicted_adoption: 0.85,  // 85%
            actual_adoption: None,
            predicted_holders: 5_000_000,
            actual_holders: None,
            predicted_price_usd: None,
            status: CheckpointStatus::Future,
        });
    }

    /// Generate financial intelligence report
    pub fn generate_report(&self) -> FinancialIntelligenceReport {
        let current_flow = self.flow_history.last().cloned();
        let current_adoption = self.adoption_history.last().cloned();
        let current_kristensen = self.kristensen_history.last().cloned();

        let critical_flow = self.calculate_critical_flow_density();

        let flow_to_critical = current_flow.as_ref()
            .map(|f| f.composite_omega / critical_flow)
            .unwrap_or(0.0);

        FinancialIntelligenceReport {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            k_law_params: self.k_law_params.clone(),
            current_flow,
            current_adoption,
            kristensen_ratio: current_kristensen,
            critical_flow_density: critical_flow,
            flow_to_critical_ratio: flow_to_critical,
            holder_distribution: self.holder_distribution.clone(),
            next_checkpoint: self.checkpoints.iter()
                .find(|c| matches!(c.status, CheckpointStatus::Future))
                .cloned(),
            robot_assignments: self.robot_roles.len(),
        }
    }
}

/// Comprehensive financial intelligence report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialIntelligenceReport {
    pub timestamp: u64,
    pub k_law_params: KLawParameters,
    pub current_flow: Option<QNKFlowDensity>,
    pub current_adoption: Option<ThreeLayerAdoption>,
    pub kristensen_ratio: Option<KristensenRatio>,
    pub critical_flow_density: f64,
    pub flow_to_critical_ratio: f64,
    pub holder_distribution: Option<HolderDistribution>,
    pub next_checkpoint: Option<AdoptionCheckpoint>,
    pub robot_assignments: usize,
}

impl std::fmt::Display for FinancialIntelligenceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║     🐳 WATER ROBOT FINANCIAL INTELLIGENCE REPORT 🐳             ║")?;
        writeln!(f, "║              K-Law Adoption Analysis for QNK                     ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;

        writeln!(f, "║ 📊 K-LAW PARAMETERS                                              ║")?;
        writeln!(f, "║   Carrying Capacity (K): {:.2}                                    ║",
            self.k_law_params.carrying_capacity)?;
        writeln!(f, "║   Friction (μ): {:.2}                                            ║",
            self.k_law_params.friction_mu)?;
        writeln!(f, "║   Flow Sensitivity (λ): {:.4}                                     ║",
            self.k_law_params.flow_sensitivity_lambda)?;
        writeln!(f, "║   Critical Flow Ω^crit: {:.4}                                     ║",
            self.critical_flow_density)?;

        if let Some(ref flow) = self.current_flow {
            writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
            writeln!(f, "║ 🌊 FLOW DENSITY (Ω_t)                                            ║")?;
            writeln!(f, "║   Staking Flow: {:.4}                                            ║", flow.staking_flow)?;
            writeln!(f, "║   DeFi Flow: {:.4}                                               ║", flow.defi_flow)?;
            writeln!(f, "║   Treasury Flow: {:.4}                                           ║", flow.treasury_flow)?;
            writeln!(f, "║   Unlock Flow: {:.4}                                             ║", flow.unlock_flow)?;
            writeln!(f, "║   Exchange Flow: {:.4}                                           ║", flow.exchange_flow)?;
            writeln!(f, "║   ──────────────────────────────────────────                    ║")?;
            writeln!(f, "║   Composite Ω_t: {:.4}                                           ║", flow.composite_omega)?;
            writeln!(f, "║   Flow to Critical: {:.2}%                                       ║",
                self.flow_to_critical_ratio * 100.0)?;
        }

        if let Some(ref adoption) = self.current_adoption {
            writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
            writeln!(f, "║ 📈 THREE-LAYER ADOPTION                                          ║")?;
            writeln!(f, "║   Layer 1 (Savings/Staking): {:.2}%                              ║",
                adoption.layer1_savings * 100.0)?;
            writeln!(f, "║   Layer 2 (Settlement): {:.2}%                                   ║",
                adoption.layer2_settlement * 100.0)?;
            writeln!(f, "║   Layer 3 (Collateral/DeFi): {:.2}%                              ║",
                adoption.layer3_collateral * 100.0)?;
            writeln!(f, "║   ──────────────────────────────────────────                    ║")?;
            writeln!(f, "║   Composite Adoption A_t: {:.2}%                                 ║",
                adoption.composite_adoption * 100.0)?;
        }

        if let Some(ref kr) = self.kristensen_ratio {
            writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
            writeln!(f, "║ 🎯 KRISTENSEN RATIO (K_t = A_t / A*_t)                           ║")?;
            writeln!(f, "║   Current Adoption A_t: {:.2}%                                   ║",
                kr.current_adoption * 100.0)?;
            writeln!(f, "║   Equilibrium Ceiling A*_t: {:.2}%                               ║",
                kr.equilibrium_ceiling * 100.0)?;
            writeln!(f, "║   ──────────────────────────────────────────                    ║")?;
            writeln!(f, "║   Kristensen Ratio: {:.4}                                        ║", kr.ratio)?;
            writeln!(f, "║   Health Status: {} {}                                          ║",
                kr.health.emoji(), kr.health.description())?;
        }

        if let Some(ref checkpoint) = self.next_checkpoint {
            writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
            writeln!(f, "║ 🎯 NEXT CHECKPOINT ({:.1})                                       ║",
                checkpoint.target_year)?;
            writeln!(f, "║   Target Adoption: {:.0}%                                        ║",
                checkpoint.predicted_adoption * 100.0)?;
            writeln!(f, "║   Target Holders: {}                                           ║",
                format_with_commas(checkpoint.predicted_holders))?;
        }

        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ 🤖 ROBOT SWARM STATUS                                            ║")?;
        writeln!(f, "║   Active Financial Robots: {}                                     ║",
            self.robot_assignments)?;

        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;

        Ok(())
    }
}

/// Map water robot types to financial roles
pub fn get_financial_role_for_robot_type(robot_type: &str) -> Option<FinancialRobotRole> {
    match robot_type.to_lowercase().as_str() {
        "jellyfish" | "quantum-jellyfish" => Some(FinancialRobotRole::FlowMonitor {
            monitored_flows: vec![
                "staking".to_string(),
                "defi".to_string(),
                "treasury".to_string(),
            ],
            update_interval_secs: 60,
        }),
        "dolphin" | "entangled-dolphin" => Some(FinancialRobotRole::HolderAnalyst {
            cohort_boundaries: vec![1.0, 10.0, 100.0, 1000.0, 10000.0],
            track_whale_movements: true,
        }),
        "octopus" | "tunneling-octopus" => Some(FinancialRobotRole::OnChainOracle {
            metrics: vec![
                "transaction_volume".to_string(),
                "active_addresses".to_string(),
                "gas_usage".to_string(),
            ],
            anomaly_detection: true,
        }),
        "whale" | "wave-particle-whale" => Some(FinancialRobotRole::WhaleWatcher {
            whale_threshold_qnk: 10000.0,
            alert_on_movement: true,
        }),
        "guardian" | "cybercetus" => Some(FinancialRobotRole::AdoptionTracker {
            k_law_params: KLawParameters::default(),
            three_layer_enabled: true,
        }),
        "school" | "robotichthys" => Some(FinancialRobotRole::SwarmCoordinator {
            aggregation_method: "weighted_consensus".to_string(),
            consensus_threshold: 0.67,
        }),
        _ => None,
    }
}

/// Helper function to format numbers with thousand separators
fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_law_calculation() {
        let engine = FinancialIntelligenceEngine::new();

        // At zero flow, adoption ceiling should be very low
        let ceiling_zero = engine.calculate_equilibrium_ceiling(0.0);
        assert!(ceiling_zero < 0.01, "Zero flow should give ~0% ceiling");

        // At critical flow, adoption should accelerate
        let critical = engine.calculate_critical_flow_density();
        let ceiling_critical = engine.calculate_equilibrium_ceiling(critical);
        assert!(ceiling_critical > 0.4, "Critical flow should give >40% ceiling");

        // At high flow, should approach carrying capacity
        let ceiling_high = engine.calculate_equilibrium_ceiling(100.0);
        assert!(ceiling_high > 0.95, "High flow should approach K");
    }

    #[test]
    fn test_kristensen_ratio() {
        let engine = FinancialIntelligenceEngine::new();

        // Healthy ratio
        let kr = engine.calculate_kristensen_ratio(0.50, 50.0);
        assert!(matches!(kr.health, AdoptionHealth::Healthy | AdoptionHealth::Recovering));

        // Critical underadoption
        let kr_critical = engine.calculate_kristensen_ratio(0.01, 50.0);
        assert!(matches!(kr_critical.health, AdoptionHealth::Critical | AdoptionHealth::Underperforming));
    }

    #[test]
    fn test_flow_weights() {
        let weights = FlowWeights::default();
        let total = weights.staking + weights.defi + weights.treasury +
                   weights.unlock_schedule + weights.exchange;
        assert!((total - 1.0).abs() < 0.001, "Weights should sum to 1.0");
    }
}
