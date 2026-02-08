//! WASM Runtime Module
//!
//! This module provides the core WASM execution infrastructure for the plugin system:
//!
//! - `executor`: Main WASM executor with deterministic configuration
//! - `host_functions`: Host function definitions accessible from WASM
//! - `gas`: Gas metering and fuel tracking

pub mod executor;
pub mod gas;
pub mod host_functions;

pub use executor::{PluginExecutor, PluginInstance, PluginState};
pub use gas::{GasCosts, GasMeter, GasMetrics};
pub use host_functions::HostFunctionRegistry;
