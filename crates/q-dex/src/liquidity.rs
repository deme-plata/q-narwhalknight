//! Quantum Liquidity Manager
//!
//! Quantum-entangled liquidity pools with physics-inspired algorithms,
//! automated market making with uncertainty principle, and post-quantum security.

use anyhow::Result;
use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, trace, warn};

use crate::types::*;

/// Minimum reserve floor — prevents dust-pool near-zero division and drain attacks (DEX-004).
/// 1000 display-unit tokens keeps pools functional while guarding against total drainage.
const MIN_POOL_RESERVE: &str = "1000";

/// Log amount tier without exposing exact values (privacy-safe debug helper).
fn log_amount_tier(v: &BigDecimal) -> &'static str {
    if v < &BigDecimal::from(1i64)              { "dust(<1)" }
    else if v < &BigDecimal::from(1_000i64)     { "small(<1K)" }
    else if v < &BigDecimal::from(1_000_000i64) { "medium(<1M)" }
    else                                         { "large(>=1M)" }
}

/// Quantum Liquidity Manager with entangled pool states
#[derive(Clone)]
pub struct QuantumLiquidityManager {
    /// Active quantum liquidity positions
    pub liquidity_positions: Arc<RwLock<HashMap<String, QuantumLiquidityPosition>>>,
    /// Quantum liquidity pools with physics properties
    pub quantum_pools: Arc<RwLock<HashMap<String, QuantumLiquidityPool>>>,
    /// Automated market maker with quantum algorithms
    pub quantum_amm: Arc<RwLock<QuantumAutomatedMarketMaker>>,
    /// Liquidity provision statistics
    pub liquidity_stats: Arc<RwLock<QuantumLiquidityStats>>,
}

/// Quantum liquidity pool with physics-based properties
#[derive(Debug, Clone)]
pub struct QuantumLiquidityPool {
    pub pool_id: String,
    pub pair_id: String,
    pub token_a_symbol: String,
    pub token_b_symbol: String,
    pub token_a_reserve: BigDecimal,
    pub token_b_reserve: BigDecimal,
    pub total_shares: BigDecimal,
    pub fee_rate: BigDecimal,
    pub quantum_k_invariant: BigDecimal, // x * y = k with quantum enhancements
    pub wave_function_state: QuantumState,
    pub entanglement_strength: f64,
    pub price_uncertainty: BigDecimal,
    pub liquidity_depth_quantum: BigDecimal,
    pub providers_count: u32,
    pub created_at: DateTime<Utc>,
    pub last_interaction: DateTime<Utc>,
}

/// Quantum Automated Market Maker with physics algorithms
#[derive(Debug, Clone)]
pub struct QuantumAutomatedMarketMaker {
    /// Golden ratio constant for optimal pricing
    pub golden_ratio: BigDecimal,
    /// Euler's number for exponential bonding curves
    pub euler_constant: BigDecimal,
    /// Pi constant for wave function calculations
    pub pi_constant: BigDecimal,
    /// Quantum slippage reduction factor in basis points (e.g., 618 = 6.18%)
    pub quantum_slippage_reduction_bps: u16,
    /// Impermanent loss protection coefficient in basis points (e.g., 8500 = 85%)
    pub impermanent_loss_protection_bps: u16,
    /// Maximum allowed price impact in basis points (e.g., 500 = 5%)
    pub max_price_impact_bps: u16,
    /// Quantum yield farming multiplier in basis points (e.g., 16180 = 161.80% = 1.618x)
    pub quantum_yield_multiplier_bps: u16,
}

impl Default for QuantumAutomatedMarketMaker {
    fn default() -> Self {
        Self {
            golden_ratio: "1.618033988749895".parse().unwrap(),
            euler_constant: "2.718281828459045".parse().unwrap(),
            pi_constant: "3.141592653589793".parse().unwrap(),
            quantum_slippage_reduction_bps: 350,   // v8.6.0: 3.50% slippage reduction (was 6.18%)
            impermanent_loss_protection_bps: 6000, // v8.6.0: 60% protection (was 85%)
            max_price_impact_bps: 800,             // v8.6.0: 8% maximum impact (was 5%)
            quantum_yield_multiplier_bps: 11000,   // v8.6.0: 110% = 1.1x yield boost (was 1.618x)
        }
    }
}

/// Quantum liquidity statistics
#[derive(Debug, Clone, Default)]
pub struct QuantumLiquidityStats {
    pub total_liquidity_usd: BigDecimal,
    pub total_pools: u32,
    pub total_providers: u64,
    pub average_position_size: BigDecimal,
    pub quantum_entangled_pools: u32,
    pub impermanent_loss_protected: BigDecimal,
    pub quantum_yield_generated: BigDecimal,
    pub wave_function_collapses: u64,
    pub superposition_positions: u32,
    pub last_update: DateTime<Utc>,
}

impl QuantumLiquidityManager {
    /// Create a new quantum liquidity manager
    pub fn new() -> Self {
        Self {
            liquidity_positions: Arc::new(RwLock::new(HashMap::new())),
            quantum_pools: Arc::new(RwLock::new(HashMap::new())),
            quantum_amm: Arc::new(RwLock::new(QuantumAutomatedMarketMaker::default())),
            liquidity_stats: Arc::new(RwLock::new(QuantumLiquidityStats::default())),
        }
    }

    /// Start quantum liquidity tracking with physics-based algorithms
    pub async fn start_quantum_tracking(&self) -> Result<()> {
        info!("⚛️ Starting Quantum Liquidity Tracking");
        info!("🌊 Physics-inspired AMM algorithms activated");
        info!("🔗 Quantum-entangled liquidity pools enabled");
        info!("🛡️ Impermanent loss protection with uncertainty principle");

        // Initialize quantum liquidity pools
        self.initialize_quantum_pools().await?;

        // Start quantum tracking tasks
        self.start_quantum_tracking_tasks().await?;

        info!("✅ Quantum liquidity tracking initialized successfully");
        Ok(())
    }

    /// Initialize quantum liquidity pools with physics properties
    async fn initialize_quantum_pools(&self) -> Result<()> {
        let mut pools = self.quantum_pools.write().await;

        // Create ORB/ORBUSD quantum-entangled pool
        let orb_orbusd_pool = QuantumLiquidityPool {
            pool_id: "ORB-ORBUSD-QUANTUM".to_string(),
            pair_id: "ORB/ORBUSD".to_string(),
            token_a_symbol: "ORB".to_string(),
            token_b_symbol: "ORBUSD".to_string(),
            token_a_reserve: BigDecimal::from(618034), // Golden ratio * 1000 * 618
            token_b_reserve: BigDecimal::from(1000000), // 1M ORBUSD
            total_shares: BigDecimal::from(785398),    // √(618034 * 1000000)
            fee_rate: "0.010".parse().unwrap(),         // v8.6.0: 1.0% protocol fee (was 0.3%)
            quantum_k_invariant: BigDecimal::from(618034000000i64), // x * y constant
            wave_function_state: QuantumState::Entangled,
            entanglement_strength: 0.707, // Maximum quantum correlation
            price_uncertainty: "0.01".parse().unwrap(), // 1% Heisenberg uncertainty
            liquidity_depth_quantum: BigDecimal::from(2000000), // 2M total depth
            providers_count: 42,          // Initial quantum number
            created_at: Utc::now(),
            last_interaction: Utc::now(),
        };

        pools.insert("ORB/ORBUSD".to_string(), orb_orbusd_pool);

        info!("🏊 Quantum liquidity pools initialized with physics constants");
        Ok(())
    }

    /// Start quantum tracking background tasks
    async fn start_quantum_tracking_tasks(&self) -> Result<()> {
        let pools = self.quantum_pools.clone();
        let stats = self.liquidity_stats.clone();
        let amm = self.quantum_amm.clone();

        // Quantum pool state monitoring
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));

            loop {
                interval.tick().await;

                let mut pools_guard = pools.write().await;
                let amm_guard = amm.read().await;
                let mut stats_guard = stats.write().await;

                let now = Utc::now();

                for pool in pools_guard.values_mut() {
                    // Update quantum K invariant with golden ratio optimization
                    let optimal_k =
                        &pool.token_a_reserve * &pool.token_b_reserve * &amm_guard.golden_ratio;
                    pool.quantum_k_invariant = optimal_k;

                    // Apply wave function evolution
                    if pool.wave_function_state == QuantumState::Superposition {
                        let time_elapsed =
                            now.timestamp() as u64 - pool.last_interaction.timestamp() as u64;
                        if time_elapsed > 300 {
                            // 5 minutes decoherence
                            pool.wave_function_state = QuantumState::Collapsed;
                            stats_guard.wave_function_collapses += 1;
                        }
                    }

                    pool.last_interaction = now;
                }

                // Update global statistics
                stats_guard.total_pools = pools_guard.len() as u32;
                stats_guard.quantum_entangled_pools = pools_guard
                    .values()
                    .filter(|p| p.entanglement_strength > 0.5)
                    .count() as u32;
                stats_guard.superposition_positions = pools_guard
                    .values()
                    .filter(|p| p.wave_function_state == QuantumState::Superposition)
                    .count() as u32;
                stats_guard.last_update = now;
            }
        });

        Ok(())
    }

    /// Add quantum-entangled liquidity to a pool
    pub async fn add_quantum_liquidity(
        &self,
        request: &QuantumTradeRequest,
    ) -> Result<QuantumLiquidityPosition> {
        info!("💧 Adding quantum-entangled liquidity with physics optimization");

        let position_id = uuid::Uuid::new_v4().to_string();
        let pair_id = &request.pair_id;

        // Calculate optimal liquidity amounts using golden ratio
        let (optimal_amount_a, optimal_amount_b) = self
            .calculate_optimal_liquidity_amounts(&request.amount, pair_id)
            .await?;

        // Calculate shares with quantum enhancement
        let shares = self
            .calculate_quantum_shares(&optimal_amount_a, &optimal_amount_b, pair_id)
            .await?;

        // Create quantum liquidity position
        let position = QuantumLiquidityPosition {
            position_id: position_id.clone(),
            provider_id: request.trader_id.clone(),
            pair_id: pair_id.clone(),
            shares: shares.clone(),
            token_a_amount: optimal_amount_a.clone(),
            token_b_amount: optimal_amount_b.clone(),
            fees_earned: BigDecimal::from(0),
            privacy_level: request.privacy_level.clone(),
            zk_proof: None,
            created_at: Utc::now(),
            locked_until: None,
        };

        // Update pool reserves
        self.update_pool_reserves(pair_id, &optimal_amount_a, &optimal_amount_b, true)
            .await?;

        // Store position
        self.liquidity_positions
            .write()
            .await
            .insert(position_id.clone(), position.clone());

        // Update statistics
        self.update_liquidity_stats(&optimal_amount_a, &optimal_amount_b, true)
            .await?;

        info!("✅ Quantum liquidity position created: {} shares", shares);
        Ok(position)
    }

    /// Remove quantum liquidity with wave function collapse
    pub async fn remove_quantum_liquidity(
        &self,
        position_id: &str,
        shares_to_remove: &BigDecimal,
    ) -> Result<(BigDecimal, BigDecimal)> {
        info!("🌊 Removing quantum liquidity with wave function collapse");

        let mut positions = self.liquidity_positions.write().await;

        if let Some(position) = positions.get_mut(position_id) {
            // Calculate proportional amounts to withdraw
            let share_percentage = shares_to_remove / &position.shares;
            let amount_a_withdraw = &position.token_a_amount * &share_percentage;
            let amount_b_withdraw = &position.token_b_amount * &share_percentage;

            // Update position
            position.shares = &position.shares - shares_to_remove;
            position.token_a_amount = &position.token_a_amount - &amount_a_withdraw;
            position.token_b_amount = &position.token_b_amount - &amount_b_withdraw;

            // Clone pair_id before potentially removing position
            let pair_id = position.pair_id.clone();
            let amounts_to_withdraw = (amount_a_withdraw.clone(), amount_b_withdraw.clone());

            // Remove position if fully withdrawn
            if position.shares == BigDecimal::from(0) {
                positions.remove(position_id);
            }

            // Update pool reserves
            self.update_pool_reserves(
                &pair_id,
                &amount_a_withdraw,
                &amount_b_withdraw,
                false,
            )
            .await?;

            // Update statistics
            self.update_liquidity_stats(&amount_a_withdraw, &amount_b_withdraw, false)
                .await?;

            info!(
                "✅ Quantum liquidity removed: {} A, {} B",
                amount_a_withdraw, amount_b_withdraw
            );
            Ok(amounts_to_withdraw)
        } else {
            Err(anyhow::anyhow!(
                "Quantum liquidity position not found: {}",
                position_id
            ))
        }
    }

    /// Calculate optimal liquidity amounts using golden ratio
    async fn calculate_optimal_liquidity_amounts(
        &self,
        input_amount: &BigDecimal,
        pair_id: &str,
    ) -> Result<(BigDecimal, BigDecimal)> {
        let pools = self.quantum_pools.read().await;
        let amm = self.quantum_amm.read().await;

        if let Some(pool) = pools.get(pair_id) {
            // Current pool ratio
            let current_ratio = &pool.token_a_reserve / &pool.token_b_reserve;

            // Apply golden ratio optimization for balanced liquidity
            let golden_ratio_adjustment =
                &current_ratio * &amm.golden_ratio / BigDecimal::from(100);
            let optimized_ratio = &current_ratio + &golden_ratio_adjustment;

            // Calculate amounts
            let amount_a = input_amount.clone();
            let amount_b = &amount_a * &optimized_ratio;

            Ok((amount_a, amount_b))
        } else {
            // Default 1:1 ratio for new pools
            Ok((input_amount.clone(), input_amount.clone()))
        }
    }

    /// Calculate quantum-enhanced shares with physics constants
    async fn calculate_quantum_shares(
        &self,
        amount_a: &BigDecimal,
        amount_b: &BigDecimal,
        pair_id: &str,
    ) -> Result<BigDecimal> {
        let pools = self.quantum_pools.read().await;

        if let Some(pool) = pools.get(pair_id) {
            // Quantum geometric mean with golden ratio enhancement
            let liquidity_added = (amount_a * amount_b)
                .sqrt()
                .ok_or_else(|| anyhow::anyhow!("Cannot calculate square root for liquidity"))?;

            let total_liquidity = (&pool.token_a_reserve * &pool.token_b_reserve)
                .sqrt()
                .ok_or_else(|| anyhow::anyhow!("Cannot calculate square root for pool"))?;

            // Share calculation with quantum enhancement
            let share_percentage = &liquidity_added / &total_liquidity;
            let quantum_shares = &pool.total_shares * share_percentage;

            // Apply golden ratio optimization
            let golden_ratio: BigDecimal = "1.618033988749895".parse().unwrap();
            let optimized_shares = &quantum_shares * &golden_ratio / BigDecimal::from(10);

            Ok(optimized_shares)
        } else {
            // Initial shares for new pool
            let initial_shares = (amount_a * amount_b)
                .sqrt()
                .ok_or_else(|| anyhow::anyhow!("Cannot calculate initial shares"))?;
            Ok(initial_shares)
        }
    }

    /// Update pool reserves with quantum effects (add/remove liquidity — not swap path)
    async fn update_pool_reserves(
        &self,
        pair_id: &str,
        amount_a: &BigDecimal,
        amount_b: &BigDecimal,
        is_addition: bool,
    ) -> Result<()> {
        let mut pools = self.quantum_pools.write().await;

        if let Some(pool) = pools.get_mut(pair_id) {
            if is_addition {
                pool.token_a_reserve = &pool.token_a_reserve + amount_a;
                pool.token_b_reserve = &pool.token_b_reserve + amount_b;
                pool.providers_count += 1;

                // Transition to superposition state for new liquidity
                pool.wave_function_state = QuantumState::Superposition;
            } else {
                // DEX-004: enforce minimum reserve floor before removal
                let min_reserve = BigDecimal::from_str(MIN_POOL_RESERVE)
                    .unwrap_or_else(|_| BigDecimal::from(0i64));
                let new_a = &pool.token_a_reserve - amount_a;
                let new_b = &pool.token_b_reserve - amount_b;
                if new_a < min_reserve || new_b < min_reserve {
                    return Err(anyhow::anyhow!(
                        "Cannot remove liquidity: reserves would fall below minimum floor for pair {}",
                        pair_id
                    ));
                }
                pool.token_a_reserve = new_a;
                pool.token_b_reserve = new_b;
                if pool.providers_count > 0 {
                    pool.providers_count -= 1;
                }

                // Wave function collapse on liquidity removal
                pool.wave_function_state = QuantumState::Collapsed;
            }

            // Update quantum K invariant
            pool.quantum_k_invariant = &pool.token_a_reserve * &pool.token_b_reserve;
            pool.liquidity_depth_quantum = &pool.token_a_reserve + &pool.token_b_reserve;
            pool.last_interaction = Utc::now();
        }

        Ok(())
    }

    /// Atomically execute a constant-product swap, update reserves, verify k-invariant,
    /// enforce slippage floor (DEX-003), and minimum reserve floor (DEX-004).
    ///
    /// Holds the pool write lock for the entire read→compute→write cycle (DEX-001/002).
    /// Returns (amount_out, new_reserve_in, new_reserve_out).
    pub async fn execute_atomic_swap(
        &self,
        pair_id: &str,
        amount_in: &BigDecimal,
        min_amount_out: &BigDecimal,  // slippage floor; zero or negative means no check
    ) -> Result<(BigDecimal, BigDecimal, BigDecimal)> {
        trace!("[DEX] swap enter pair={}", pair_id);

        let min_reserve = BigDecimal::from_str(MIN_POOL_RESERVE)
            .map_err(|_| anyhow::anyhow!("Invalid MIN_POOL_RESERVE constant"))?;

        let mut pools = self.quantum_pools.write().await;
        let pool = pools.get_mut(pair_id)
            .ok_or_else(|| anyhow::anyhow!("Quantum pool not found: {}", pair_id))?;

        let reserve_in  = pool.token_a_reserve.clone();
        let reserve_out = pool.token_b_reserve.clone();

        // Constant-product formula with fee: out = (amount_in * (1 - fee) * reserve_out)
        //                                              / (reserve_in + amount_in * (1 - fee))
        let one = BigDecimal::from(1i64);
        let fee_factor = &one - &pool.fee_rate;
        let amount_in_with_fee = amount_in * &fee_factor;
        let amount_out = &amount_in_with_fee * &reserve_out
            / (&reserve_in + &amount_in_with_fee);

        debug!("[DEX] swap pair={} fee_rate={} amount_out_tier={} min_out_tier={}",
            pair_id, pool.fee_rate,
            log_amount_tier(&amount_out), log_amount_tier(min_amount_out));

        // DEX-003: enforce slippage floor
        if min_amount_out > &BigDecimal::from(0i64) && amount_out < *min_amount_out {
            debug!("[DEX] swap REJECTED slippage pair={} reason=amount_out_below_floor", pair_id);
            return Err(anyhow::anyhow!(
                "Slippage exceeded: expected at least {} out, got {} for pair {}",
                min_amount_out, amount_out, pair_id
            ));
        }

        let new_reserve_in  = &reserve_in  + amount_in;
        let new_reserve_out = &reserve_out - &amount_out;

        // DEX-004: enforce minimum reserve floor
        if new_reserve_out < min_reserve {
            debug!("[DEX] swap REJECTED reserve_floor pair={} reason=post_swap_reserve_below_min", pair_id);
            return Err(anyhow::anyhow!(
                "Reserve too low after swap: {} for pair {}",
                new_reserve_out, pair_id
            ));
        }

        // DEX-002: verify k-invariant (fee guarantees new_k >= old_k; this is a safety guard)
        let old_k = &reserve_in * &reserve_out;
        let new_k = &new_reserve_in * &new_reserve_out;
        if new_k < old_k {
            error!("[DEX] INVARIANT VIOLATION pair={} k_decreased=true", pair_id);
            return Err(anyhow::anyhow!(
                "K-invariant violation for pair {}: new_k < old_k",
                pair_id
            ));
        }

        // DEX-001: atomically write updated reserves under the write lock
        pool.token_a_reserve = new_reserve_in.clone();
        pool.token_b_reserve = new_reserve_out.clone();
        pool.quantum_k_invariant = new_k;
        pool.last_interaction = Utc::now();

        debug!("[DEX] swap OK pair={} reserve_in_tier={} reserve_out_tier={}",
            pair_id,
            log_amount_tier(&new_reserve_in),
            log_amount_tier(&new_reserve_out));
        trace!("[DEX] swap exit pair={}", pair_id);

        Ok((amount_out, new_reserve_in, new_reserve_out))
    }

    /// Update liquidity statistics
    async fn update_liquidity_stats(
        &self,
        amount_a: &BigDecimal,
        amount_b: &BigDecimal,
        is_addition: bool,
    ) -> Result<()> {
        let mut stats = self.liquidity_stats.write().await;

        // Assume 1:1 USD value for simplicity (would use oracle in production)
        let usd_value = amount_a + amount_b;

        if is_addition {
            stats.total_liquidity_usd = &stats.total_liquidity_usd + &usd_value;
            stats.total_providers += 1;
        } else {
            stats.total_liquidity_usd = &stats.total_liquidity_usd - &usd_value;
            if stats.total_providers > 0 {
                stats.total_providers -= 1;
            }
        }

        // Calculate average position size
        if stats.total_providers > 0 {
            stats.average_position_size =
                &stats.total_liquidity_usd / BigDecimal::from(stats.total_providers);
        }

        stats.last_update = Utc::now();
        Ok(())
    }

    /// Get quantum price with uncertainty principle
    pub async fn get_quantum_price(
        &self,
        pair_id: &str,
        amount_in: &BigDecimal,
        token_in: &str,
    ) -> Result<BigDecimal> {
        let pools = self.quantum_pools.read().await;
        let amm = self.quantum_amm.read().await;

        if let Some(pool) = pools.get(pair_id) {
            let (reserve_in, reserve_out) = if token_in == &pool.token_a_symbol {
                (&pool.token_a_reserve, &pool.token_b_reserve)
            } else {
                (&pool.token_b_reserve, &pool.token_a_reserve)
            };

            // Quantum-enhanced constant product formula with slippage reduction
            let amount_in_with_fee = amount_in
                * (BigDecimal::from(1000) - &pool.fee_rate * BigDecimal::from(1000))
                / BigDecimal::from(1000);
            let numerator = &amount_in_with_fee * reserve_out;
            let denominator = reserve_in + &amount_in_with_fee;
            let base_amount_out = numerator / denominator;

            // Apply quantum slippage reduction using integer basis points
            // slippage_reduction_bps is in basis points (e.g., 618 = 6.18%)
            // Formula: amount * (10000 + bps) / 10000
            let bps_denominator = BigDecimal::from(10000);
            let slippage_numerator = BigDecimal::from(10000 + amm.quantum_slippage_reduction_bps as i64);
            let quantum_amount_out = &base_amount_out * slippage_numerator / &bps_denominator;

            // Apply price uncertainty (Heisenberg principle)
            let uncertainty_factor = &pool.price_uncertainty;
            let price_with_uncertainty =
                &quantum_amount_out * (BigDecimal::from(1) + uncertainty_factor);

            Ok(price_with_uncertainty)
        } else {
            Err(anyhow::anyhow!("Quantum pool not found: {}", pair_id))
        }
    }

    /// Get liquidity position information
    pub async fn get_liquidity_position(
        &self,
        position_id: &str,
    ) -> Result<QuantumLiquidityPosition> {
        self.liquidity_positions
            .read()
            .await
            .get(position_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Quantum liquidity position not found: {}", position_id))
    }

    /// Get quantum liquidity statistics
    pub async fn get_quantum_liquidity_stats(&self) -> QuantumLiquidityStats {
        self.liquidity_stats.read().await.clone()
    }

    /// Get quantum pool information
    pub async fn get_quantum_pool(&self, pair_id: &str) -> Result<QuantumLiquidityPool> {
        self.quantum_pools
            .read()
            .await
            .get(pair_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Quantum pool not found: {}", pair_id))
    }

    /// Calculate impermanent loss with quantum protection
    ///
    /// Uses high-precision integer arithmetic for basis points to avoid f64 rounding errors.
    /// The impermanent loss formula: IL = 1 - (2 * sqrt(r)) / (1 + r)
    /// where r = current_price / initial_price
    pub async fn calculate_quantum_impermanent_loss(
        &self,
        position_id: &str,
        current_price_ratio: &BigDecimal,
    ) -> Result<BigDecimal> {
        let positions = self.liquidity_positions.read().await;
        let amm = self.quantum_amm.read().await;

        if let Some(position) = positions.get(position_id) {
            let initial_price_ratio = &position.token_a_amount / &position.token_b_amount;
            let price_change_ratio = current_price_ratio / initial_price_ratio;

            // BigDecimal sqrt with maintained precision
            let sqrt_ratio = price_change_ratio
                .sqrt()
                .ok_or_else(|| anyhow::anyhow!("Cannot calculate square root"))?;

            // Calculate IL multiplier: 2 * sqrt(r) / (1 + r)
            // All operations maintain BigDecimal precision
            let two = BigDecimal::from(2);
            let one = BigDecimal::from(1);
            let il_multiplier = &two * &sqrt_ratio / (&one + &price_change_ratio);

            // Impermanent loss as percentage: (1 - multiplier) * 100
            let hundred = BigDecimal::from(100);
            let impermanent_loss = (&one - &il_multiplier) * &hundred;

            // Apply quantum protection using integer basis points
            // protection_bps is in basis points (e.g., 8500 = 85%)
            // Formula: loss * (10000 - protection_bps) / 10000
            let bps_denominator = BigDecimal::from(10000);
            let protection_numerator = BigDecimal::from(10000 - amm.impermanent_loss_protection_bps as i64);
            let protected_loss = &impermanent_loss * protection_numerator / bps_denominator;

            Ok(protected_loss)
        } else {
            Err(anyhow::anyhow!("Position not found: {}", position_id))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_liquidity_manager_creation() {
        let manager = QuantumLiquidityManager::new();
        assert!(manager.start_quantum_tracking().await.is_ok());
    }

    #[tokio::test]
    async fn test_optimal_liquidity_calculation() {
        let manager = QuantumLiquidityManager::new();
        manager.start_quantum_tracking().await.unwrap();

        let (amount_a, amount_b) = manager
            .calculate_optimal_liquidity_amounts(&BigDecimal::from(1000), "ORB/ORBUSD")
            .await
            .unwrap();

        assert!(amount_a > BigDecimal::from(0));
        assert!(amount_b > BigDecimal::from(0));
    }

    #[tokio::test]
    async fn test_quantum_price_calculation() {
        let manager = QuantumLiquidityManager::new();
        manager.start_quantum_tracking().await.unwrap();

        let price = manager
            .get_quantum_price("ORB/ORBUSD", &BigDecimal::from(100), "ORB")
            .await
            .unwrap();

        assert!(price > BigDecimal::from(0));
    }

    #[tokio::test]
    async fn test_impermanent_loss_protection() {
        let manager = QuantumLiquidityManager::new();
        manager.start_quantum_tracking().await.unwrap();

        // Create test position
        let request = QuantumTradeRequest {
            user: "test".to_string(),
            trader_id: "test_provider".to_string(),
            pair_id: "ORB/ORBUSD".to_string(),
            side: TradeSide::Buy,
            amount: BigDecimal::from(1000),
            price: None,
            order_type: OrderType::Market,
            privacy_level: QuantumPrivacyTier::Basic,
            zk_proof_required: false,
            max_slippage_bps: 50, // 0.5%
            expires_at: None,
            quantum_signature: vec![0u8; 64],
            entanglement_proof: None,
        };

        let position = manager.add_quantum_liquidity(&request).await.unwrap();

        // Test impermanent loss calculation
        let il = manager
            .calculate_quantum_impermanent_loss(
                &position.position_id,
                &BigDecimal::from(2), // 100% price change
            )
            .await
            .unwrap();

        // Should be less than 5.72% due to quantum protection
        assert!(il < "5.72".parse().unwrap());
    }
}
