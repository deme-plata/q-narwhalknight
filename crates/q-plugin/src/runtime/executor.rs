//! Plugin Executor Module
//!
//! The main WASM executor for the Q-NarwhalKnight plugin system. This module provides:
//!
//! - Deterministic WASM execution for consensus compatibility
//! - Plugin loading, validation, and instantiation
//! - Entry point execution with gas metering
//! - Plugin lifecycle management
//!
//! # Deterministic Execution
//!
//! The executor is configured for deterministic execution:
//! - Fuel-based execution limiting (no wall-clock timeouts)
//! - Disabled SIMD for consistent behavior across platforms
//! - Disabled multi-threading within WASM
//! - Cranelift compiler with consistent settings
//!
//! # Example
//!
//! ```rust,ignore
//! use q_plugin::{PluginExecutor, PluginManifest, CapabilitySet};
//!
//! let executor = PluginExecutor::new()?;
//!
//! // Load plugin
//! let manifest = PluginManifest::new("my-plugin", "My Plugin");
//! let plugin_id = executor.load_plugin(&wasm_bytes, manifest)?;
//!
//! // Execute
//! let result = executor.execute(&plugin_id, "process", &input, 100_000)?;
//! println!("Output: {:?}", result.output);
//! println!("Gas used: {}", result.gas_used);
//!
//! // Unload when done
//! executor.unload_plugin(&plugin_id)?;
//! ```

use crate::{
    CapabilitySet, ExecutionResult, PluginError, PluginEvent, PluginId, PluginManifest,
    PluginResult, PluginStorage,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use wasmtime::{Config, Engine, Linker, Module, Store};

use super::gas::{fuel_to_gas, gas_to_fuel, GasCosts, GasMeter};
use super::host_functions::{HostContext, HostFunctionRegistry};

/// Plugin state during execution
#[derive(Debug, Clone)]
pub struct PluginState {
    /// Plugin identifier
    pub plugin_id: PluginId,
    /// Granted capabilities
    pub capabilities: CapabilitySet,
    /// Remaining gas
    pub gas_remaining: u64,
    /// Plugin storage
    pub storage: Arc<RwLock<PluginStorage>>,
    /// Emitted events
    pub events: Vec<PluginEvent>,
}

impl PluginState {
    /// Create new plugin state
    pub fn new(plugin_id: PluginId, capabilities: CapabilitySet, gas_limit: u64) -> Self {
        Self {
            plugin_id: plugin_id.clone(),
            capabilities,
            gas_remaining: gas_limit,
            storage: Arc::new(RwLock::new(PluginStorage::new(&plugin_id))),
            events: Vec::new(),
        }
    }
}

/// A loaded plugin instance
pub struct PluginInstance {
    /// Plugin manifest
    pub manifest: PluginManifest,
    /// Compiled WASM module
    pub module: Module,
    /// Plugin storage
    pub storage: Arc<RwLock<PluginStorage>>,
    /// Load timestamp
    pub loaded_at: std::time::Instant,
    /// Execution count
    pub execution_count: u64,
    /// Total gas consumed
    pub total_gas_consumed: u64,
}

impl PluginInstance {
    /// Get plugin statistics
    pub fn stats(&self) -> PluginInstanceStats {
        PluginInstanceStats {
            plugin_id: self.manifest.id.clone(),
            version: self.manifest.version.to_string(),
            execution_count: self.execution_count,
            total_gas_consumed: self.total_gas_consumed,
            uptime_secs: self.loaded_at.elapsed().as_secs(),
        }
    }
}

/// Statistics for a plugin instance
#[derive(Debug, Clone)]
pub struct PluginInstanceStats {
    pub plugin_id: PluginId,
    pub version: String,
    pub execution_count: u64,
    pub total_gas_consumed: u64,
    pub uptime_secs: u64,
}

/// Main WASM plugin executor
pub struct PluginExecutor {
    /// WASM engine with deterministic configuration
    engine: Engine,
    /// Linker with host functions
    linker: Linker<HostContext>,
    /// Loaded plugin instances
    instances: RwLock<HashMap<PluginId, PluginInstance>>,
    /// Gas costs configuration
    gas_costs: GasCosts,
    /// Current block height (for context)
    block_height: RwLock<u64>,
    /// Current timestamp (for context)
    timestamp: RwLock<u64>,
}

impl PluginExecutor {
    /// Create a new plugin executor with deterministic configuration
    pub fn new() -> PluginResult<Self> {
        // Configure engine for deterministic execution
        let mut config = Config::new();

        // Enable fuel-based execution limiting
        config.consume_fuel(true);

        // Deterministic settings
        config.cranelift_opt_level(wasmtime::OptLevel::Speed);
        config.wasm_simd(false); // Disable SIMD for determinism
        config.wasm_threads(false); // Disable threads for determinism
        config.wasm_multi_memory(false);
        config.wasm_memory64(false);

        // Memory settings
        config.max_wasm_stack(1024 * 1024); // 1MB stack limit
        config.static_memory_maximum_size(100 * 1024 * 1024); // 100MB max
        config.dynamic_memory_guard_size(64 * 1024); // 64KB guard

        // Enable reference types for modern WASM
        config.wasm_reference_types(true);
        config.wasm_bulk_memory(true);

        let engine =
            Engine::new(&config).map_err(|e| PluginError::Internal(format!("Engine creation failed: {}", e)))?;

        // Create linker and register host functions
        let mut linker = Linker::new(&engine);
        HostFunctionRegistry::register(&mut linker)?;

        info!("Plugin executor initialized with deterministic configuration");

        Ok(Self {
            engine,
            linker,
            instances: RwLock::new(HashMap::new()),
            gas_costs: GasCosts::default(),
            block_height: RwLock::new(0),
            timestamp: RwLock::new(0),
        })
    }

    /// Create executor with custom gas costs
    pub fn with_gas_costs(gas_costs: GasCosts) -> PluginResult<Self> {
        let mut executor = Self::new()?;
        executor.gas_costs = gas_costs;
        Ok(executor)
    }

    /// Set the current block height for plugin context
    pub fn set_block_height(&self, height: u64) {
        *self.block_height.write() = height;
    }

    /// Set the current timestamp for plugin context
    pub fn set_timestamp(&self, timestamp: u64) {
        *self.timestamp.write() = timestamp;
    }

    /// Load and validate a WASM plugin
    pub fn load_plugin(
        &self,
        wasm_bytes: &[u8],
        manifest: PluginManifest,
    ) -> PluginResult<PluginId> {
        let plugin_id = manifest.id.clone();

        // Check if plugin already exists
        {
            let instances = self.instances.read();
            if instances.contains_key(&plugin_id) {
                return Err(PluginError::AlreadyExists(plugin_id));
            }
        }

        // Validate WASM module
        self.validate_wasm(wasm_bytes)?;

        // Compile module
        let module = Module::new(&self.engine, wasm_bytes).map_err(|e| {
            PluginError::Compilation(format!("Failed to compile WASM: {}", e))
        })?;

        // Validate exports match manifest entry points
        self.validate_exports(&module, &manifest)?;

        // Create instance
        let instance = PluginInstance {
            manifest: manifest.clone(),
            module,
            storage: Arc::new(RwLock::new(PluginStorage::new(&plugin_id))),
            loaded_at: std::time::Instant::now(),
            execution_count: 0,
            total_gas_consumed: 0,
        };

        // Store instance
        {
            let mut instances = self.instances.write();
            instances.insert(plugin_id.clone(), instance);
        }

        info!(
            plugin_id = %plugin_id,
            version = %manifest.version,
            "Plugin loaded successfully"
        );

        Ok(plugin_id)
    }

    /// Execute a plugin entry point
    pub fn execute(
        &self,
        plugin_id: &PluginId,
        entry_point: &str,
        input: &[u8],
        gas_limit: u64,
    ) -> PluginResult<ExecutionResult> {
        // Get plugin instance
        let (module, manifest, storage) = {
            let instances = self.instances.read();
            let instance = instances
                .get(plugin_id)
                .ok_or_else(|| PluginError::NotFound(plugin_id.clone()))?;

            // Check if entry point is valid
            if !instance.manifest.entry_points.contains(&entry_point.to_string()) {
                return Err(PluginError::InvalidEntryPoint(entry_point.to_string()));
            }

            (
                instance.module.clone(),
                instance.manifest.clone(),
                Arc::clone(&instance.storage),
            )
        };

        // Create execution context
        let block_height = *self.block_height.read();
        let timestamp = *self.timestamp.read();

        let host_ctx = HostContext::new(
            plugin_id.clone(),
            manifest.capabilities.clone(),
            gas_limit,
            block_height,
            timestamp,
        );

        // Replace storage with persistent one
        {
            let storage_data = storage.read().clone();
            let mut host_storage = host_ctx.storage.write().unwrap();
            *host_storage = storage_data;
        }

        // Create store with fuel
        let mut store = Store::new(&self.engine, host_ctx);
        store
            .set_fuel(gas_to_fuel(gas_limit))
            .map_err(|e| PluginError::Internal(format!("Failed to set fuel: {}", e)))?;

        // Instantiate module
        let instance = self
            .linker
            .instantiate(&mut store, &module)
            .map_err(|e| PluginError::Instantiation(format!("Failed to instantiate: {}", e)))?;

        // Get memory for input/output
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| PluginError::Memory("No memory export found".to_string()))?;

        // Allocate input buffer in WASM memory
        let alloc_fn = instance
            .get_typed_func::<u32, u32>(&mut store, "alloc")
            .map_err(|e| PluginError::InvalidEntryPoint(format!("alloc function not found: {}", e)))?;

        let input_ptr = alloc_fn
            .call(&mut store, input.len() as u32)
            .map_err(|e| PluginError::Execution(format!("alloc failed: {}", e)))?;

        // Write input to WASM memory
        memory
            .write(&mut store, input_ptr as usize, input)
            .map_err(|e| PluginError::Memory(format!("Failed to write input: {}", e)))?;

        // Get entry point function
        let entry_fn = instance
            .get_typed_func::<(u32, u32, u32), u32>(&mut store, entry_point)
            .map_err(|e| PluginError::InvalidEntryPoint(format!("{}: {}", entry_point, e)))?;

        // Allocate output length pointer
        let out_len_ptr = alloc_fn
            .call(&mut store, 4)
            .map_err(|e| PluginError::Execution(format!("alloc for output length failed: {}", e)))?;

        // Execute entry point: fn(input_ptr, input_len, out_len_ptr) -> output_ptr
        let result = entry_fn.call(&mut store, (input_ptr, input.len() as u32, out_len_ptr));

        // Calculate gas consumed
        let remaining_fuel = store.get_fuel().unwrap_or(0);
        let gas_used = fuel_to_gas(gas_to_fuel(gas_limit).saturating_sub(remaining_fuel));

        // Get events from context
        let events = store.data().take_events();

        // Persist storage changes
        {
            let host_storage = store.data().storage.read().unwrap();
            let mut persistent_storage = storage.write();
            *persistent_storage = host_storage.clone();
        }

        // Update instance statistics
        {
            let mut instances = self.instances.write();
            if let Some(instance) = instances.get_mut(plugin_id) {
                instance.execution_count += 1;
                instance.total_gas_consumed += gas_used;
            }
        }

        match result {
            Ok(output_ptr) => {
                // Read output length
                let mut len_bytes = [0u8; 4];
                memory
                    .read(&store, out_len_ptr as usize, &mut len_bytes)
                    .map_err(|e| PluginError::Memory(format!("Failed to read output length: {}", e)))?;
                let output_len = u32::from_le_bytes(len_bytes) as usize;

                // Read output data
                let mut output = vec![0u8; output_len];
                if output_len > 0 {
                    memory
                        .read(&store, output_ptr as usize, &mut output)
                        .map_err(|e| PluginError::Memory(format!("Failed to read output: {}", e)))?;
                }

                debug!(
                    plugin_id = %plugin_id,
                    entry_point = %entry_point,
                    gas_used = gas_used,
                    output_len = output_len,
                    "Plugin execution completed successfully"
                );

                Ok(ExecutionResult::success(output, gas_used, events))
            }
            Err(e) => {
                // Check if it was an out-of-gas error
                let error_msg = e.to_string();
                if error_msg.contains("fuel") || error_msg.contains("out of gas") {
                    warn!(
                        plugin_id = %plugin_id,
                        entry_point = %entry_point,
                        gas_limit = gas_limit,
                        "Plugin execution ran out of gas"
                    );
                    Err(PluginError::OutOfGas {
                        used: gas_used,
                        limit: gas_limit,
                    })
                } else {
                    error!(
                        plugin_id = %plugin_id,
                        entry_point = %entry_point,
                        error = %e,
                        "Plugin execution failed"
                    );
                    Ok(ExecutionResult::failure(error_msg, gas_used))
                }
            }
        }
    }

    /// Unload a plugin
    pub fn unload_plugin(&self, plugin_id: &PluginId) -> PluginResult<()> {
        let mut instances = self.instances.write();
        if instances.remove(plugin_id).is_some() {
            info!(plugin_id = %plugin_id, "Plugin unloaded");
            Ok(())
        } else {
            Err(PluginError::NotFound(plugin_id.clone()))
        }
    }

    /// Check if a plugin is loaded
    pub fn is_loaded(&self, plugin_id: &PluginId) -> bool {
        self.instances.read().contains_key(plugin_id)
    }

    /// Get list of loaded plugins
    pub fn loaded_plugins(&self) -> Vec<PluginId> {
        self.instances.read().keys().cloned().collect()
    }

    /// Get plugin statistics
    pub fn get_stats(&self, plugin_id: &PluginId) -> Option<PluginInstanceStats> {
        self.instances.read().get(plugin_id).map(|i| i.stats())
    }

    /// Get all plugin statistics
    pub fn all_stats(&self) -> Vec<PluginInstanceStats> {
        self.instances
            .read()
            .values()
            .map(|i| i.stats())
            .collect()
    }

    /// Validate WASM bytes
    fn validate_wasm(&self, wasm_bytes: &[u8]) -> PluginResult<()> {
        // Check magic number
        if wasm_bytes.len() < 8 {
            return Err(PluginError::InvalidModule("WASM too short".to_string()));
        }

        let magic = &wasm_bytes[0..4];
        if magic != b"\0asm" {
            return Err(PluginError::InvalidModule(
                "Invalid WASM magic number".to_string(),
            ));
        }

        // Check version (should be 1)
        let version = u32::from_le_bytes([wasm_bytes[4], wasm_bytes[5], wasm_bytes[6], wasm_bytes[7]]);
        if version != 1 {
            return Err(PluginError::InvalidModule(format!(
                "Unsupported WASM version: {}",
                version
            )));
        }

        Ok(())
    }

    /// Validate that module exports match manifest entry points
    fn validate_exports(&self, module: &Module, manifest: &PluginManifest) -> PluginResult<()> {
        let exports: Vec<_> = module.exports().map(|e| e.name().to_string()).collect();

        // Check memory export
        if !exports.contains(&"memory".to_string()) {
            return Err(PluginError::InvalidModule(
                "Module must export 'memory'".to_string(),
            ));
        }

        // Check alloc function
        if !exports.contains(&"alloc".to_string()) {
            return Err(PluginError::InvalidModule(
                "Module must export 'alloc' function".to_string(),
            ));
        }

        // Check entry points
        for entry_point in &manifest.entry_points {
            if !exports.contains(entry_point) {
                return Err(PluginError::InvalidModule(format!(
                    "Module missing entry point: {}",
                    entry_point
                )));
            }
        }

        Ok(())
    }

    /// Get the gas costs configuration
    pub fn gas_costs(&self) -> &GasCosts {
        &self.gas_costs
    }
}

impl Default for PluginExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create default plugin executor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Capability;

    // Minimal WASM module for testing (WAT format converted to bytes)
    // This is a simple module that exports memory, alloc, and a "process" function
    const MINIMAL_WASM: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, // magic: \0asm
        0x01, 0x00, 0x00, 0x00, // version: 1
    ];

    #[test]
    fn test_executor_creation() {
        let executor = PluginExecutor::new();
        assert!(executor.is_ok());
    }

    #[test]
    fn test_validate_wasm_magic() {
        let executor = PluginExecutor::new().unwrap();

        // Valid magic
        assert!(executor.validate_wasm(MINIMAL_WASM).is_ok());

        // Invalid magic
        let invalid = &[0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];
        assert!(executor.validate_wasm(invalid).is_err());

        // Too short
        let short = &[0x00, 0x61, 0x73, 0x6d];
        assert!(executor.validate_wasm(short).is_err());
    }

    #[test]
    fn test_load_invalid_wasm() {
        let executor = PluginExecutor::new().unwrap();
        let manifest = PluginManifest::new("test", "Test Plugin")
            .with_entry_point("process");

        // Try to load invalid WASM
        let result = executor.load_plugin(&[0x00, 0x00, 0x00, 0x00], manifest);
        assert!(result.is_err());
    }

    #[test]
    fn test_plugin_not_found() {
        let executor = PluginExecutor::new().unwrap();

        let result = executor.execute(&"nonexistent".to_string(), "process", &[], 1000);
        assert!(matches!(result, Err(PluginError::NotFound(_))));
    }

    #[test]
    fn test_set_context() {
        let executor = PluginExecutor::new().unwrap();

        executor.set_block_height(12345);
        executor.set_timestamp(1234567890);

        assert_eq!(*executor.block_height.read(), 12345);
        assert_eq!(*executor.timestamp.read(), 1234567890);
    }

    #[test]
    fn test_plugin_state() {
        let state = PluginState::new(
            "test-plugin".to_string(),
            CapabilitySet::new(&[Capability::StorageRead, Capability::StorageWrite]),
            1_000_000,
        );

        assert_eq!(state.plugin_id, "test-plugin");
        assert!(state.capabilities.has(Capability::StorageRead));
        assert!(state.capabilities.has(Capability::StorageWrite));
        assert!(!state.capabilities.has(Capability::EmitEvent));
        assert_eq!(state.gas_remaining, 1_000_000);
    }

    #[test]
    fn test_gas_costs_default() {
        let executor = PluginExecutor::new().unwrap();
        let costs = executor.gas_costs();

        assert!(costs.storage_write_base > costs.storage_read_base);
        assert!(costs.verify_signature > costs.sha3_base);
    }

    #[test]
    fn test_loaded_plugins_list() {
        let executor = PluginExecutor::new().unwrap();

        // Initially empty
        assert!(executor.loaded_plugins().is_empty());

        // After failed load, still empty
        let manifest = PluginManifest::new("test", "Test");
        let _ = executor.load_plugin(&[], manifest);
        assert!(executor.loaded_plugins().is_empty());
    }
}
