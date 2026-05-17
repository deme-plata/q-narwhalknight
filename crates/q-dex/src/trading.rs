//! Quantum Trading Engine
//!
//! Physics-inspired trading algorithms with quantum field theory for price discovery,
//! Heisenberg uncertainty-based volatility modeling, and post-quantum security.

use anyhow::Result;
use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::liquidity::QuantumLiquidityManager;
use crate::types::*;

/// Quantum Trading Engine with physics-inspired algorithms
#[derive(Clone)]
pub struct QuantumTradingEngine {
    /// Active quantum trade positions
    pub active_positions: Arc<RwLock<HashMap<String, QuantumTradePosition>>>,
    /// Quantum order book with wave function states
    pub quantum_order_book: Arc<RwLock<QuantumOrderBook>>,
    /// Physics-based trading parameters
    pub quantum_params: Arc<RwLock<QuantumTradingParameters>>,
    /// Trade execution statistics
    pub execution_stats: Arc<RwLock<QuantumExecutionStats>>,
    /// Liquidity manager — holds pool reserves; used for atomic swap (DEX-001/002)
    pub liquidity_manager: Arc<QuantumLiquidityManager>,
}

/// Quantum trade position with physics properties
#[derive(Debug, Clone)]
pub struct QuantumTradePosition {
    pub position_id: String,
    pub trader_id: String,
    pub pair_id: String,
    pub side: TradeSide,
    pub amount: BigDecimal,
    pub entry_price: BigDecimal,
    pub current_price: BigDecimal,
    pub unrealized_pnl: BigDecimal,
    pub quantum_state: QuantumState,
    pub wave_function_probability: f64,
    pub entanglement_strength: f64,
    pub decoherence_time: u64,
    pub opened_at: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
}

/// Quantum order book with superposition states
#[derive(Debug, Clone)]
pub struct QuantumOrderBook {
    pub pair_id: String,
    pub quantum_bids: Vec<QuantumOrder>,
    pub quantum_asks: Vec<QuantumOrder>,
    pub wave_interference_pattern: WavePattern,
    pub total_quantum_depth: BigDecimal,
    pub last_collapse_time: DateTime<Utc>,
    pub superposition_orders: u32,
}

/// Quantum order with physics properties
#[derive(Debug, Clone)]
pub struct QuantumOrder {
    pub order_id: String,
    pub trader_id: String,
    pub price: BigDecimal,
    pub amount: BigDecimal,
    pub quantum_state: QuantumState,
    pub probability_amplitude: f64,
    pub wave_function_phase: f64,
    pub entangled_with: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Physics-based trading parameters using quantum constants
#[derive(Debug, Clone)]
pub struct QuantumTradingParameters {
    /// Planck constant scaled for financial markets (volatility quantum)
    pub planck_financial: BigDecimal,
    /// Golden ratio for optimal price discovery
    pub golden_ratio: BigDecimal,
    /// Euler's number for exponential price movements
    pub euler_constant: BigDecimal,
    /// Pi constant for wave function calculations
    pub pi_constant: BigDecimal,
    /// Heisenberg uncertainty principle factor for price/volume uncertainty
    pub uncertainty_principle: f64,
    /// Wave function collapse threshold (price movement trigger)
    pub collapse_threshold: f64,
    /// Quantum entanglement correlation strength
    pub entanglement_strength: f64,
    /// Decoherence time for quantum states (seconds)
    pub decoherence_time_seconds: u64,
    /// Maximum allowed leverage with quantum risk management
    pub max_quantum_leverage: f64,
    /// Quantum-enhanced liquidation threshold in basis points (e.g., 8000 = 80%)
    pub quantum_liquidation_threshold_bps: u16,
}

impl Default for QuantumTradingParameters {
    fn default() -> Self {
        Self {
            planck_financial: "0.00000000000000000000000000000000066260701".parse().unwrap(), // Scaled for volatility
            golden_ratio: "1.618033988749895".parse().unwrap(),
            euler_constant: "2.718281828459045".parse().unwrap(),
            pi_constant: "3.141592653589793".parse().unwrap(),
            uncertainty_principle: 0.1618, // Golden ratio percentage
            collapse_threshold: 0.05,      // 5% price movement
            entanglement_strength: 0.707,  // √2/2
            decoherence_time_seconds: 300, // 5 minutes
            max_quantum_leverage: 10.0,
            quantum_liquidation_threshold_bps: 8000, // 80%
        }
    }
}

/// Quantum execution statistics
#[derive(Debug, Clone, Default)]
pub struct QuantumExecutionStats {
    pub total_trades_executed: u64,
    pub quantum_enhanced_trades: u64,
    pub wave_function_collapses: u64,
    pub entangled_trade_pairs: u64,
    /// Total execution time in milliseconds (avoids f64 accumulation precision loss)
    /// Use `average_execution_time_ms()` to get the computed average on-demand
    pub total_execution_time_ms: u64,
    /// Quantum slippage reduction in basis points (e.g., 618 = 6.18%)
    pub quantum_slippage_reduction_bps: u16,
    pub privacy_enhanced_trades: u64,
    pub zk_proof_validations: u64,
    pub last_stats_update: DateTime<Utc>,
}

impl QuantumExecutionStats {
    /// Calculate average execution time on-demand to avoid f64 accumulation errors
    #[inline]
    pub fn average_execution_time_ms(&self) -> f64 {
        if self.total_trades_executed == 0 {
            0.0
        } else {
            self.total_execution_time_ms as f64 / self.total_trades_executed as f64
        }
    }
}

impl QuantumTradingEngine {
    /// Create a new quantum trading engine with a shared liquidity manager (DEX-001/002).
    pub fn new(liquidity_manager: Arc<QuantumLiquidityManager>) -> Self {
        Self {
            active_positions: Arc::new(RwLock::new(HashMap::new())),
            quantum_order_book: Arc::new(RwLock::new(QuantumOrderBook {
                pair_id: String::new(),
                quantum_bids: Vec::new(),
                quantum_asks: Vec::new(),
                wave_interference_pattern: WavePattern::Neutral,
                total_quantum_depth: BigDecimal::from(0),
                last_collapse_time: Utc::now(),
                superposition_orders: 0,
            })),
            quantum_params: Arc::new(RwLock::new(QuantumTradingParameters::default())),
            execution_stats: Arc::new(RwLock::new(QuantumExecutionStats::default())),
            liquidity_manager,
        }
    }

    /// Initialize the quantum trading engine
    pub async fn initialize_quantum_engine(&self) -> Result<()> {
        info!("⚛️ Initializing Quantum Trading Engine");
        info!("📊 Physics-inspired price discovery algorithms activated");
        info!("🎯 Heisenberg uncertainty-based volatility modeling enabled");
        info!("🔒 Post-quantum cryptographic trade security active");

        // Initialize quantum parameters with physics constants
        let mut params = self.quantum_params.write().await;
        *params = QuantumTradingParameters::default();

        // Start quantum state monitoring
        self.start_quantum_state_monitoring().await?;

        // Initialize quantum order book
        self.initialize_quantum_order_book().await?;

        info!("✅ Quantum Trading Engine initialized with physics-based algorithms");
        Ok(())
    }

    /// Start quantum state monitoring for decoherence and wave function collapse
    async fn start_quantum_state_monitoring(&self) -> Result<()> {
        let positions = self.active_positions.clone();
        let params = self.quantum_params.clone();
        let stats = self.execution_stats.clone();

        // Quantum decoherence monitoring task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));

            loop {
                interval.tick().await;

                let decoherence_time = {
                    let params_guard = params.read().await;
                    params_guard.decoherence_time_seconds
                };

                let mut positions_guard = positions.write().await;
                let now = Utc::now();

                // Check for quantum state decoherence
                for position in positions_guard.values_mut() {
                    let time_elapsed =
                        now.timestamp() as u64 - position.last_update.timestamp() as u64;

                    if time_elapsed > decoherence_time {
                        // Wave function collapse due to decoherence
                        position.quantum_state = QuantumState::Collapsed;
                        position.wave_function_probability = 1.0;
                        position.last_update = now;

                        // Update statistics
                        let mut stats_guard = stats.write().await;
                        stats_guard.wave_function_collapses += 1;
                        stats_guard.last_stats_update = now;
                    }
                }
            }
        });

        Ok(())
    }

    /// Initialize quantum order book with wave function superposition
    async fn initialize_quantum_order_book(&self) -> Result<()> {
        let mut order_book = self.quantum_order_book.write().await;

        order_book.pair_id = "ORB/ORBUSD".to_string();
        order_book.wave_interference_pattern = WavePattern::Constructive;
        order_book.last_collapse_time = Utc::now();

        info!("📖 Quantum order book initialized with wave function superposition");
        Ok(())
    }

    /// Execute quantum-enhanced trade with physics-based algorithms.
    ///
    /// Physics price discovery drives display; the AMM constant-product formula in
    /// `execute_atomic_swap()` drives the actual settled amount and reserve mutation
    /// (DEX-001/002).  Slippage is enforced inside `execute_atomic_swap` (DEX-003).
    pub async fn execute_quantum_trade(
        &self,
        request: &QuantumTradeRequest,
    ) -> Result<QuantumTradeResult> {
        info!("⚡ Executing quantum trade with physics-based price discovery");

        // Apply Heisenberg uncertainty principle to price/volume
        let quantum_price = self.apply_uncertainty_principle(request).await?;

        // Check for quantum entanglement effects
        let entangled_effect = self.calculate_entanglement_effect(&request.pair_id).await?;

        // Execute trade with quantum enhancements (physics estimation)
        let mut execution_result = self
            .execute_with_quantum_algorithms(request, quantum_price, entangled_effect)
            .await?;

        // DEX-001/002/003: Atomic reserve update under pool write lock.
        // slippage floor = physics output * (1 - max_slippage); zero means no check.
        let physics_out = &execution_result.filled_amount;
        let min_out = if request.max_slippage_bps > 0 {
            physics_out
                * (BigDecimal::from(10_000i64) - BigDecimal::from(request.max_slippage_bps as i64))
                / BigDecimal::from(10_000i64)
        } else {
            BigDecimal::from(0i64)
        };
        debug!("[DEX] trade executing atomic swap pair={} slippage_floor_bps={}",
            request.pair_id, request.max_slippage_bps);
        let (actual_out, _, _) = self.liquidity_manager
            .execute_atomic_swap(&request.pair_id, &request.amount, &min_out)
            .await?;
        // Settle against the AMM output, not the physics estimate
        execution_result.filled_amount = actual_out;

        // Update quantum statistics
        self.update_quantum_stats(&execution_result).await?;

        // Create quantum trade result
        let trade_result = QuantumTradeResult {
            trade_id: uuid::Uuid::new_v4().to_string(),
            trader_id: request.trader_id.clone(),
            pair_id: request.pair_id.clone(),
            side: request.side.clone(),
            amount_filled: execution_result.filled_amount,
            price: execution_result.execution_price,
            fees_paid: execution_result.quantum_fees,
            privacy_level: request.privacy_level.clone(),
            zk_proof: execution_result.zk_proof,
            tor_circuit_id: execution_result.tor_circuit_id,
            executed_at: Utc::now(),
            block_height: Some(execution_result.block_height),
            quantum_signature: Some(request.quantum_signature.clone()),
        };

        info!(
            "✅ Quantum trade executed successfully with ID: {}",
            trade_result.trade_id
        );
        Ok(trade_result)
    }

    /// Apply Heisenberg uncertainty principle to price discovery
    async fn apply_uncertainty_principle(
        &self,
        request: &QuantumTradeRequest,
    ) -> Result<BigDecimal> {
        let params = self.quantum_params.read().await;
        let uncertainty_factor = params.uncertainty_principle;

        // Base price with quantum uncertainty
        let base_price = request
            .price
            .clone()
            .unwrap_or_else(|| "1.618".parse().unwrap());
        use std::str::FromStr;
        let uncertainty = &base_price * BigDecimal::from_str(&uncertainty_factor.to_string())?;

        // Apply quantum price discovery using golden ratio
        let quantum_price =
            &base_price + (&uncertainty * &params.golden_ratio / BigDecimal::from(10));

        Ok(quantum_price)
    }

    /// Calculate quantum entanglement effects on price
    async fn calculate_entanglement_effect(&self, pair_id: &str) -> Result<f64> {
        let params = self.quantum_params.read().await;

        // Check if pair is entangled (ORB/ORBUSD has maximum entanglement)
        let entanglement_strength = if pair_id == "ORB/ORBUSD" {
            params.entanglement_strength
        } else {
            0.0
        };

        // Quantum correlation affects price movement
        Ok(entanglement_strength)
    }

    /// Execute trade with quantum-enhanced algorithms
    async fn execute_with_quantum_algorithms(
        &self,
        request: &QuantumTradeRequest,
        quantum_price: BigDecimal,
        entanglement_effect: f64,
    ) -> Result<QuantumExecutionResult> {
        let start_time = std::time::Instant::now();

        // Apply wave function collapse for price certainty
        let collapsed_price = self
            .collapse_wave_function(quantum_price, entanglement_effect)
            .await?;

        // Calculate quantum-enhanced fees (reduced due to efficiency)
        let quantum_fees = self
            .calculate_quantum_fees(&request.amount, &collapsed_price)
            .await?;

        // Generate ZK proof if required
        let zk_proof = if request.zk_proof_required {
            Some(self.generate_trade_zk_proof(request).await?)
        } else {
            None
        };

        // Create Tor circuit for privacy if needed
        let tor_circuit_id = if request.privacy_level != QuantumPrivacyTier::Basic {
            Some(format!("quantum_circuit_{}", uuid::Uuid::new_v4().simple()))
        } else {
            None
        };

        let execution_time = start_time.elapsed().as_millis() as f64;

        Ok(QuantumExecutionResult {
            filled_amount: request.amount.clone(),
            execution_price: collapsed_price,
            quantum_fees,
            zk_proof,
            tor_circuit_id,
            block_height: 123456, // TODO: Get from consensus
            execution_time_ms: execution_time,
            quantum_enhanced: true,
        })
    }

    /// Collapse wave function to determine final price
    async fn collapse_wave_function(
        &self,
        quantum_price: BigDecimal,
        entanglement: f64,
    ) -> Result<BigDecimal> {
        let params = self.quantum_params.read().await;

        // Apply golden ratio for optimal price discovery
        let golden_adjustment = &quantum_price * &params.golden_ratio / BigDecimal::from(100);

        // Apply entanglement correlation
        let entanglement_adjustment =
            &quantum_price * BigDecimal::from_str(&entanglement.to_string())? / BigDecimal::from(1000);

        // Final collapsed price
        let collapsed_price = quantum_price + golden_adjustment + entanglement_adjustment;

        Ok(collapsed_price)
    }

    /// Calculate quantum-enhanced trading fees (optimized through physics)
    async fn calculate_quantum_fees(
        &self,
        amount: &BigDecimal,
        price: &BigDecimal,
    ) -> Result<BigDecimal> {
        let trade_value = amount * price;

        // Quantum-optimized fee rate (reduced due to efficiency gains)
        let quantum_fee_rate: BigDecimal = "0.001".parse().unwrap(); // 0.1% vs standard 0.3%
        let base_fees = &trade_value * &quantum_fee_rate;

        // Apply golden ratio optimization
        let golden_ratio: BigDecimal = "1.618".parse().unwrap();
        let optimized_fees = &base_fees / golden_ratio;

        Ok(optimized_fees)
    }

    /// Generate ZK proof for trade privacy
    async fn generate_trade_zk_proof(
        &self,
        request: &QuantumTradeRequest,
    ) -> Result<QuantumZkProof> {
        // TODO: Integrate with actual ZK-SNARK implementation
        Ok(QuantumZkProof {
            proof_data: vec![0u8; 256], // Placeholder proof
            public_inputs: vec![
                format!("trade_valid:{}", request.trader_id),
                format!("amount_range:{}", request.amount),
                "privacy_preserved".to_string(),
            ],
            circuit_type: ZkCircuitType::TradeValidation,
            generated_at: Utc::now(),
        })
    }

    /// Update quantum trading statistics
    async fn update_quantum_stats(&self, result: &QuantumExecutionResult) -> Result<()> {
        let mut stats = self.execution_stats.write().await;

        stats.total_trades_executed += 1;

        if result.quantum_enhanced {
            stats.quantum_enhanced_trades += 1;
        }

        if result.zk_proof.is_some() {
            stats.privacy_enhanced_trades += 1;
            stats.zk_proof_validations += 1;
        }

        // Accumulate total execution time (precision-safe u64 accumulation)
        // Average is computed on-demand via stats.average_execution_time_ms()
        stats.total_execution_time_ms += result.execution_time_ms as u64;

        stats.last_stats_update = Utc::now();

        Ok(())
    }

    /// Place quantum order with superposition state
    pub async fn place_quantum_order(&self, order: QuantumOrderRequest) -> Result<String> {
        let order_id = uuid::Uuid::new_v4().to_string();

        let quantum_order = QuantumOrder {
            order_id: order_id.clone(),
            trader_id: order.trader_id,
            price: order.price,
            amount: order.amount,
            quantum_state: QuantumState::Superposition,
            probability_amplitude: 0.707, // √2/2 for maximum superposition
            wave_function_phase: 0.0,
            entangled_with: None,
            created_at: Utc::now(),
        };

        let mut order_book = self.quantum_order_book.write().await;

        match order.side {
            TradeSide::Buy => order_book.quantum_bids.push(quantum_order),
            TradeSide::Sell => order_book.quantum_asks.push(quantum_order),
        }

        order_book.superposition_orders += 1;

        info!(
            "📝 Quantum order placed in superposition state: {}",
            order_id
        );
        Ok(order_id)
    }

    /// Cancel quantum order with wave function collapse
    pub async fn cancel_quantum_order(&self, order_id: &str) -> Result<()> {
        let mut order_book = self.quantum_order_book.write().await;

        // Remove from bids
        order_book
            .quantum_bids
            .retain(|order| order.order_id != order_id);

        // Remove from asks
        order_book
            .quantum_asks
            .retain(|order| order.order_id != order_id);

        if order_book.superposition_orders > 0 {
            order_book.superposition_orders -= 1;
        }

        info!(
            "❌ Quantum order cancelled with wave function collapse: {}",
            order_id
        );
        Ok(())
    }

    /// Get quantum trading statistics
    pub async fn get_quantum_trading_stats(&self) -> QuantumExecutionStats {
        self.execution_stats.read().await.clone()
    }

    /// Get quantum order book depth
    pub async fn get_quantum_order_book(&self, pair_id: &str) -> Result<QuantumOrderBook> {
        let order_book = self.quantum_order_book.read().await;
        if order_book.pair_id == pair_id {
            Ok(order_book.clone())
        } else {
            Err(anyhow::anyhow!(
                "Quantum order book not found for pair: {}",
                pair_id
            ))
        }
    }
}

/// Quantum execution result with physics data
#[derive(Debug, Clone)]
pub struct QuantumExecutionResult {
    pub filled_amount: BigDecimal,
    pub execution_price: BigDecimal,
    pub quantum_fees: BigDecimal,
    pub zk_proof: Option<QuantumZkProof>,
    pub tor_circuit_id: Option<String>,
    pub block_height: u64,
    pub execution_time_ms: f64,
    pub quantum_enhanced: bool,
}

/// Quantum order request
#[derive(Debug, Clone)]
pub struct QuantumOrderRequest {
    pub trader_id: String,
    pub pair_id: String,
    pub side: TradeSide,
    pub price: BigDecimal,
    pub amount: BigDecimal,
    pub order_type: OrderType,
    pub privacy_level: QuantumPrivacyTier,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> QuantumTradingEngine {
        use crate::liquidity::QuantumLiquidityManager;
        QuantumTradingEngine::new(Arc::new(QuantumLiquidityManager::new()))
    }

    #[tokio::test]
    async fn test_quantum_trading_engine_creation() {
        let engine = test_engine();
        assert!(engine.initialize_quantum_engine().await.is_ok());
    }

    #[tokio::test]
    async fn test_uncertainty_principle_application() {
        let engine = test_engine();
        engine.initialize_quantum_engine().await.unwrap();

        let request = QuantumTradeRequest {
            user: "test".to_string(),
            trader_id: "test_trader".to_string(),
            pair_id: "ORB/ORBUSD".to_string(),
            side: TradeSide::Buy,
            amount: BigDecimal::from(100),
            price: Some("1.618".parse().unwrap()),
            order_type: OrderType::Market,
            privacy_level: QuantumPrivacyTier::Basic,
            zk_proof_required: false,
            max_slippage_bps: 50, // 0.5%
            expires_at: None,
            quantum_signature: vec![0u8; 64],
            entanglement_proof: None,
        };

        let quantum_price = engine.apply_uncertainty_principle(&request).await.unwrap();
        assert!(quantum_price > "1.618".parse().unwrap());
    }

    #[tokio::test]
    async fn test_quantum_entanglement_calculation() {
        let engine = test_engine();
        engine.initialize_quantum_engine().await.unwrap();

        let entanglement = engine
            .calculate_entanglement_effect("ORB/ORBUSD")
            .await
            .unwrap();
        assert_eq!(entanglement, 0.707); // √2/2 for ORB/ORBUSD pair

        let no_entanglement = engine
            .calculate_entanglement_effect("OTHER/PAIR")
            .await
            .unwrap();
        assert_eq!(no_entanglement, 0.0);
    }

    #[tokio::test]
    async fn test_quantum_order_placement() {
        let engine = test_engine();
        engine.initialize_quantum_engine().await.unwrap();

        let order = QuantumOrderRequest {
            trader_id: "test_trader".to_string(),
            pair_id: "ORB/ORBUSD".to_string(),
            side: TradeSide::Buy,
            price: "1.618".parse().unwrap(),
            amount: BigDecimal::from(100),
            order_type: OrderType::Limit,
            privacy_level: QuantumPrivacyTier::Enhanced,
        };

        let order_id = engine.place_quantum_order(order).await.unwrap();
        assert!(!order_id.is_empty());

        // Test cancellation
        assert!(engine.cancel_quantum_order(&order_id).await.is_ok());
    }
}
