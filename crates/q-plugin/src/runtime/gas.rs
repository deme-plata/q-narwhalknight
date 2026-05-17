//! Gas Metering Module
//!
//! Provides gas metering for WASM plugin execution using wasmtime's fuel mechanism.
//! Each operation has a defined cost, and execution halts when fuel is exhausted.
//!
//! # Gas Cost Model
//!
//! Operations are categorized by computational complexity:
//! - **Base operations**: Simple memory access, arithmetic (1-10 gas)
//! - **Storage operations**: Read/write to persistent storage (100-1000 gas)
//! - **Crypto operations**: Hash computation, signature verification (1000-10000 gas)
//! - **System operations**: Block info, timestamp (10-50 gas)
//!
//! # Example
//!
//! ```rust,ignore
//! use q_plugin::runtime::gas::{GasCosts, GasMeter};
//!
//! let costs = GasCosts::default();
//! let mut meter = GasMeter::new(1_000_000);
//!
//! // Consume gas for a storage read
//! meter.consume(costs.storage_read)?;
//!
//! // Check remaining gas
//! println!("Remaining: {}", meter.remaining());
//! ```

use crate::{PluginError, PluginResult};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// Gas costs for various operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasCosts {
    // Storage operations
    /// Cost to read from storage (base + per byte)
    pub storage_read_base: u64,
    pub storage_read_per_byte: u64,
    /// Cost to write to storage (base + per byte)
    pub storage_write_base: u64,
    pub storage_write_per_byte: u64,
    /// Cost to delete from storage
    pub storage_delete: u64,
    /// Cost to check existence
    pub storage_exists: u64,

    // Event operations
    /// Cost to emit an event (base + per byte)
    pub emit_event_base: u64,
    pub emit_event_per_byte: u64,

    // System operations
    /// Cost to get block height
    pub get_block_height: u64,
    /// Cost to get timestamp
    pub get_timestamp: u64,
    /// Cost to log a message (base + per byte)
    pub log_base: u64,
    pub log_per_byte: u64,

    // Crypto operations
    /// Cost for SHA3-256 hash (base + per byte)
    pub sha3_base: u64,
    pub sha3_per_byte: u64,
    /// Cost for Ed25519 signature verification
    pub verify_signature: u64,

    // Memory operations
    /// Cost to allocate memory (per page, 64KB)
    pub memory_alloc_per_page: u64,
    /// Cost to copy memory (per byte)
    pub memory_copy_per_byte: u64,

    // WASM execution
    /// Base cost per WASM instruction (used by wasmtime fuel)
    pub wasm_instruction: u64,
}

impl Default for GasCosts {
    fn default() -> Self {
        Self {
            // Storage: expensive due to persistence
            storage_read_base: 100,
            storage_read_per_byte: 1,
            storage_write_base: 500,
            storage_write_per_byte: 5,
            storage_delete: 200,
            storage_exists: 50,

            // Events: moderate cost
            emit_event_base: 200,
            emit_event_per_byte: 2,

            // System: cheap
            get_block_height: 10,
            get_timestamp: 10,
            log_base: 20,
            log_per_byte: 1,

            // Crypto: expensive due to computation
            sha3_base: 100,
            sha3_per_byte: 5,
            verify_signature: 5000,

            // Memory: moderate
            memory_alloc_per_page: 1000,
            memory_copy_per_byte: 1,

            // WASM: 1 gas per instruction (wasmtime default)
            wasm_instruction: 1,
        }
    }
}

impl GasCosts {
    /// Create minimal gas costs (for testing)
    pub fn minimal() -> Self {
        Self {
            storage_read_base: 1,
            storage_read_per_byte: 0,
            storage_write_base: 1,
            storage_write_per_byte: 0,
            storage_delete: 1,
            storage_exists: 1,
            emit_event_base: 1,
            emit_event_per_byte: 0,
            get_block_height: 1,
            get_timestamp: 1,
            log_base: 1,
            log_per_byte: 0,
            sha3_base: 1,
            sha3_per_byte: 0,
            verify_signature: 1,
            memory_alloc_per_page: 1,
            memory_copy_per_byte: 0,
            wasm_instruction: 1,
        }
    }

    /// Calculate storage read cost
    pub fn storage_read(&self, value_size: usize) -> u64 {
        self.storage_read_base + (value_size as u64 * self.storage_read_per_byte)
    }

    /// Calculate storage write cost
    pub fn storage_write(&self, value_size: usize) -> u64 {
        self.storage_write_base + (value_size as u64 * self.storage_write_per_byte)
    }

    /// Calculate emit event cost
    pub fn emit_event(&self, data_size: usize) -> u64 {
        self.emit_event_base + (data_size as u64 * self.emit_event_per_byte)
    }

    /// Calculate log cost
    pub fn log(&self, message_size: usize) -> u64 {
        self.log_base + (message_size as u64 * self.log_per_byte)
    }

    /// Calculate SHA3 hash cost
    pub fn sha3(&self, data_size: usize) -> u64 {
        self.sha3_base + (data_size as u64 * self.sha3_per_byte)
    }

    /// Calculate memory allocation cost
    pub fn memory_alloc(&self, pages: u32) -> u64 {
        (pages as u64) * self.memory_alloc_per_page
    }

    /// Calculate memory copy cost
    pub fn memory_copy(&self, bytes: usize) -> u64 {
        (bytes as u64) * self.memory_copy_per_byte
    }
}

/// Gas meter for tracking consumption during execution
#[derive(Debug)]
pub struct GasMeter {
    /// Initial gas limit
    limit: u64,
    /// Gas consumed so far (atomic for thread safety)
    consumed: AtomicU64,
    /// Gas costs configuration
    costs: GasCosts,
}

impl GasMeter {
    /// Create a new gas meter with the specified limit
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            consumed: AtomicU64::new(0),
            costs: GasCosts::default(),
        }
    }

    /// Create a gas meter with custom costs
    pub fn with_costs(limit: u64, costs: GasCosts) -> Self {
        Self {
            limit,
            consumed: AtomicU64::new(0),
            costs,
        }
    }

    /// Get the gas limit
    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// Get gas consumed so far
    pub fn consumed(&self) -> u64 {
        self.consumed.load(Ordering::Relaxed)
    }

    /// Get remaining gas
    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.consumed())
    }

    /// Get the costs configuration
    pub fn costs(&self) -> &GasCosts {
        &self.costs
    }

    /// Try to consume gas, returns error if insufficient
    pub fn consume(&self, amount: u64) -> PluginResult<()> {
        let current = self.consumed.fetch_add(amount, Ordering::Relaxed);
        let new_total = current + amount;

        if new_total > self.limit {
            // Revert the addition
            self.consumed.fetch_sub(amount, Ordering::Relaxed);
            Err(PluginError::OutOfGas {
                used: current,
                limit: self.limit,
            })
        } else {
            Ok(())
        }
    }

    /// Consume gas for storage read
    pub fn consume_storage_read(&self, value_size: usize) -> PluginResult<()> {
        self.consume(self.costs.storage_read(value_size))
    }

    /// Consume gas for storage write
    pub fn consume_storage_write(&self, value_size: usize) -> PluginResult<()> {
        self.consume(self.costs.storage_write(value_size))
    }

    /// Consume gas for storage delete
    pub fn consume_storage_delete(&self) -> PluginResult<()> {
        self.consume(self.costs.storage_delete)
    }

    /// Consume gas for storage exists check
    pub fn consume_storage_exists(&self) -> PluginResult<()> {
        self.consume(self.costs.storage_exists)
    }

    /// Consume gas for emitting event
    pub fn consume_emit_event(&self, data_size: usize) -> PluginResult<()> {
        self.consume(self.costs.emit_event(data_size))
    }

    /// Consume gas for getting block height
    pub fn consume_get_block_height(&self) -> PluginResult<()> {
        self.consume(self.costs.get_block_height)
    }

    /// Consume gas for getting timestamp
    pub fn consume_get_timestamp(&self) -> PluginResult<()> {
        self.consume(self.costs.get_timestamp)
    }

    /// Consume gas for logging
    pub fn consume_log(&self, message_size: usize) -> PluginResult<()> {
        self.consume(self.costs.log(message_size))
    }

    /// Consume gas for SHA3 hash
    pub fn consume_sha3(&self, data_size: usize) -> PluginResult<()> {
        self.consume(self.costs.sha3(data_size))
    }

    /// Consume gas for signature verification
    pub fn consume_verify_signature(&self) -> PluginResult<()> {
        self.consume(self.costs.verify_signature)
    }

    /// Reset the meter (for reuse)
    pub fn reset(&self) {
        self.consumed.store(0, Ordering::Relaxed);
    }

    /// Set a new limit
    pub fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
        self.reset();
    }

    /// Get metrics snapshot
    pub fn metrics(&self) -> GasMetrics {
        GasMetrics {
            limit: self.limit,
            consumed: self.consumed(),
            remaining: self.remaining(),
        }
    }
}

impl Clone for GasMeter {
    fn clone(&self) -> Self {
        Self {
            limit: self.limit,
            consumed: AtomicU64::new(self.consumed.load(Ordering::Relaxed)),
            costs: self.costs.clone(),
        }
    }
}

/// Gas metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasMetrics {
    /// Gas limit
    pub limit: u64,
    /// Gas consumed
    pub consumed: u64,
    /// Gas remaining
    pub remaining: u64,
}

impl GasMetrics {
    /// Calculate utilization percentage
    pub fn utilization(&self) -> f64 {
        if self.limit == 0 {
            0.0
        } else {
            (self.consumed as f64 / self.limit as f64) * 100.0
        }
    }
}

/// Convert gas to wasmtime fuel
///
/// Wasmtime uses "fuel" for execution limiting. This function converts
/// our gas units to fuel units. Currently 1:1 mapping.
pub fn gas_to_fuel(gas: u64) -> u64 {
    gas
}

/// Convert wasmtime fuel to gas
pub fn fuel_to_gas(fuel: u64) -> u64 {
    fuel
}

/// Calculate fuel consumed from remaining fuel
pub fn calculate_fuel_consumed(initial_fuel: u64, remaining_fuel: u64) -> u64 {
    initial_fuel.saturating_sub(remaining_fuel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gas_costs_default() {
        let costs = GasCosts::default();
        assert!(costs.storage_write_base > costs.storage_read_base);
        assert!(costs.verify_signature > costs.sha3_base);
    }

    #[test]
    fn test_gas_costs_calculation() {
        let costs = GasCosts::default();

        // Storage read cost
        let read_cost = costs.storage_read(100);
        assert_eq!(read_cost, costs.storage_read_base + 100 * costs.storage_read_per_byte);

        // Storage write cost
        let write_cost = costs.storage_write(100);
        assert_eq!(write_cost, costs.storage_write_base + 100 * costs.storage_write_per_byte);

        // SHA3 cost
        let sha3_cost = costs.sha3(64);
        assert_eq!(sha3_cost, costs.sha3_base + 64 * costs.sha3_per_byte);
    }

    #[test]
    fn test_gas_meter_basic() {
        let meter = GasMeter::new(1000);
        assert_eq!(meter.limit(), 1000);
        assert_eq!(meter.consumed(), 0);
        assert_eq!(meter.remaining(), 1000);

        meter.consume(100).unwrap();
        assert_eq!(meter.consumed(), 100);
        assert_eq!(meter.remaining(), 900);
    }

    #[test]
    fn test_gas_meter_out_of_gas() {
        let meter = GasMeter::new(100);

        // Should succeed
        meter.consume(50).unwrap();

        // Should fail - would exceed limit
        let result = meter.consume(100);
        assert!(matches!(result, Err(PluginError::OutOfGas { .. })));

        // Consumed should be unchanged after failed attempt
        assert_eq!(meter.consumed(), 50);
    }

    #[test]
    fn test_gas_meter_reset() {
        let meter = GasMeter::new(1000);
        meter.consume(500).unwrap();
        assert_eq!(meter.consumed(), 500);

        meter.reset();
        assert_eq!(meter.consumed(), 0);
    }

    #[test]
    fn test_gas_meter_operations() {
        let meter = GasMeter::new(100_000);

        meter.consume_get_block_height().unwrap();
        meter.consume_get_timestamp().unwrap();
        meter.consume_storage_read(100).unwrap();
        meter.consume_sha3(64).unwrap();

        assert!(meter.consumed() > 0);
        assert!(meter.remaining() < 100_000);
    }

    #[test]
    fn test_gas_metrics() {
        let meter = GasMeter::new(1000);
        meter.consume(250).unwrap();

        let metrics = meter.metrics();
        assert_eq!(metrics.limit, 1000);
        assert_eq!(metrics.consumed, 250);
        assert_eq!(metrics.remaining, 750);
        assert!((metrics.utilization() - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_gas_to_fuel_conversion() {
        assert_eq!(gas_to_fuel(1000), 1000);
        assert_eq!(fuel_to_gas(1000), 1000);
        assert_eq!(calculate_fuel_consumed(1000, 300), 700);
    }

    #[test]
    fn test_gas_meter_clone() {
        let meter = GasMeter::new(1000);
        meter.consume(100).unwrap();

        let cloned = meter.clone();
        assert_eq!(cloned.limit(), 1000);
        assert_eq!(cloned.consumed(), 100);

        // Modifications should be independent
        cloned.consume(50).unwrap();
        assert_eq!(meter.consumed(), 100);
        assert_eq!(cloned.consumed(), 150);
    }
}
