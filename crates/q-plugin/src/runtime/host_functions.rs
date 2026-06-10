//! Host Functions Module
//!
//! Defines the host functions that WASM plugins can call. All host functions
//! check capabilities before executing and consume appropriate gas.
//!
//! # Available Host Functions
//!
//! ## Storage
//! - `plugin_storage_read(key_ptr, key_len) -> (ptr, len)` - Read from namespaced storage
//! - `plugin_storage_write(key_ptr, key_len, val_ptr, val_len)` - Write to namespaced storage
//! - `plugin_storage_delete(key_ptr, key_len)` - Delete from storage
//! - `plugin_storage_exists(key_ptr, key_len) -> bool` - Check if key exists
//!
//! ## Events
//! - `plugin_emit_event(topic_ptr, topic_len, data_ptr, data_len)` - Emit blockchain event
//!
//! ## System
//! - `plugin_get_block_height() -> u64` - Get current block height
//! - `plugin_get_timestamp() -> u64` - Get current timestamp
//! - `plugin_log(level, msg_ptr, msg_len)` - Log message
//!
//! ## Cryptography
//! - `plugin_sha3_256(data_ptr, data_len, out_ptr)` - Compute SHA3-256 hash
//! - `plugin_verify_signature(msg_ptr, msg_len, sig_ptr, pubkey_ptr) -> bool` - Verify Ed25519
//!
//! # Security Model
//!
//! All host functions:
//! 1. Check if the plugin has the required capability
//! 2. Consume appropriate gas before execution
//! 3. Validate memory bounds
//! 4. Use namespaced storage to isolate plugin data

use crate::{
    Capability, CapabilitySet, PluginError, PluginEvent, PluginId, PluginResult, PluginStorage,
};
use sha3::{Digest, Sha3_256};
use std::sync::{Arc, RwLock};
use tracing::{debug, error, info, trace, warn};
use wasmtime::{Caller, Linker, Memory};

use super::gas::GasMeter;

/// Log levels for plugin logging
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

impl From<u32> for LogLevel {
    fn from(value: u32) -> Self {
        match value {
            0 => LogLevel::Trace,
            1 => LogLevel::Debug,
            2 => LogLevel::Info,
            3 => LogLevel::Warn,
            _ => LogLevel::Error,
        }
    }
}

/// Host context provided to WASM execution
#[derive(Debug)]
pub struct HostContext {
    /// Plugin identifier
    pub plugin_id: PluginId,
    /// Granted capabilities
    pub capabilities: CapabilitySet,
    /// Gas meter for tracking consumption
    pub gas_meter: Arc<GasMeter>,
    /// Plugin storage
    pub storage: Arc<RwLock<PluginStorage>>,
    /// Emitted events
    pub events: Arc<RwLock<Vec<PluginEvent>>>,
    /// Current block height
    pub block_height: u64,
    /// Current timestamp
    pub timestamp: u64,
}

impl HostContext {
    /// Create a new host context
    pub fn new(
        plugin_id: PluginId,
        capabilities: CapabilitySet,
        gas_limit: u64,
        block_height: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            plugin_id: plugin_id.clone(),
            capabilities,
            gas_meter: Arc::new(GasMeter::new(gas_limit)),
            storage: Arc::new(RwLock::new(PluginStorage::new(&plugin_id))),
            events: Arc::new(RwLock::new(Vec::new())),
            block_height,
            timestamp,
        }
    }

    /// Check if capability is granted
    pub fn has_capability(&self, capability: Capability) -> bool {
        self.capabilities.has(capability)
    }

    /// Require capability, return error if not granted
    pub fn require_capability(&self, capability: Capability) -> PluginResult<()> {
        if self.has_capability(capability) {
            Ok(())
        } else {
            Err(PluginError::CapabilityDenied(capability))
        }
    }

    /// Get events and clear the buffer
    pub fn take_events(&self) -> Vec<PluginEvent> {
        let mut events = self.events.write().unwrap();
        std::mem::take(&mut *events)
    }

    /// Get gas consumed
    pub fn gas_consumed(&self) -> u64 {
        self.gas_meter.consumed()
    }
}

impl Clone for HostContext {
    fn clone(&self) -> Self {
        Self {
            plugin_id: self.plugin_id.clone(),
            capabilities: self.capabilities.clone(),
            gas_meter: Arc::clone(&self.gas_meter),
            storage: Arc::clone(&self.storage),
            events: Arc::clone(&self.events),
            block_height: self.block_height,
            timestamp: self.timestamp,
        }
    }
}

/// Registry for host functions
pub struct HostFunctionRegistry;

impl HostFunctionRegistry {
    /// Register all host functions with the linker
    pub fn register(linker: &mut Linker<HostContext>) -> PluginResult<()> {
        Self::register_storage_functions(linker)?;
        Self::register_event_functions(linker)?;
        Self::register_system_functions(linker)?;
        Self::register_crypto_functions(linker)?;
        Ok(())
    }

    /// Register storage-related host functions
    fn register_storage_functions(linker: &mut Linker<HostContext>) -> PluginResult<()> {
        // plugin_storage_read: Read value from storage
        // Returns: (ptr, len) where ptr is 0 if not found
        linker
            .func_wrap(
                "env",
                "plugin_storage_read",
                |mut caller: Caller<'_, HostContext>,
                 key_ptr: u32,
                 key_len: u32,
                 out_ptr: u32,
                 out_len_ptr: u32|
                 -> Result<u32, wasmtime::Error> {
                    let ctx = caller.data();

                    // Check capability
                    if !ctx.has_capability(Capability::StorageRead) {
                        return Ok(0); // Return null pointer on capability denial
                    }

                    // Read key from WASM memory
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;

                    let key = read_string(&memory, &caller, key_ptr, key_len)?;

                    // Get storage and read
                    let ctx = caller.data();
                    let storage = ctx.storage.read().unwrap();
                    let value = storage.read(&key);

                    // Consume gas based on value size
                    drop(storage);
                    let ctx = caller.data();
                    let value_len = value.as_ref().map(|v| v.len()).unwrap_or(0);
                    if ctx.gas_meter.consume_storage_read(value_len).is_err() {
                        return Err(wasmtime::Error::msg("out of gas"));
                    }

                    match value {
                        Some(data) => {
                            // Write value to output buffer
                            let memory = caller
                                .get_export("memory")
                                .and_then(|e| e.into_memory())
                                .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;

                            write_bytes(&memory, &mut caller, out_ptr, &data)?;

                            // Write length
                            let len_bytes = (data.len() as u32).to_le_bytes();
                            write_bytes(&memory, &mut caller, out_len_ptr, &len_bytes)?;

                            Ok(1) // Success
                        }
                        None => Ok(0), // Not found
                    }
                },
            )
            .map_err(|e| PluginError::Internal(e.to_string()))?;

        // plugin_storage_write: Write value to storage
        linker
            .func_wrap(
                "env",
                "plugin_storage_write",
                |mut caller: Caller<'_, HostContext>,
                 key_ptr: u32,
                 key_len: u32,
                 val_ptr: u32,
                 val_len: u32|
                 -> Result<u32, wasmtime::Error> {
                    let ctx = caller.data();

                    // Check capability
                    if !ctx.has_capability(Capability::StorageWrite) {
                        return Ok(0);
                    }

                    // Consume gas first
                    if ctx
                        .gas_meter
                        .consume_storage_write(val_len as usize)
                        .is_err()
                    {
                        return Err(wasmtime::Error::msg("out of gas"));
                    }

                    // Read key and value from WASM memory
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;

                    let key = read_string(&memory, &caller, key_ptr, key_len)?;
                    let value = read_bytes(&memory, &caller, val_ptr, val_len)?;

                    // Write to storage
                    let ctx = caller.data();
                    let mut storage = ctx.storage.write().unwrap();
                    storage.write(&key, value);

                    Ok(1) // Success
                },
            )
            .map_err(|e| PluginError::Internal(e.to_string()))?;

        // plugin_storage_delete: Delete key from storage
        linker
            .func_wrap(
                "env",
                "plugin_storage_delete",
                |mut caller: Caller<'_, HostContext>,
                 key_ptr: u32,
                 key_len: u32|
                 -> Result<u32, wasmtime::Error> {
                    let ctx = caller.data();

                    // Check capability
                    if !ctx.has_capability(Capability::StorageWrite) {
                        return Ok(0);
                    }

                    // Consume gas
                    if ctx.gas_meter.consume_storage_delete().is_err() {
                        return Err(wasmtime::Error::msg("out of gas"));
                    }

                    // Read key
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;

                    let key = read_string(&memory, &caller, key_ptr, key_len)?;

                    // Delete from storage
                    let ctx = caller.data();
                    let mut storage = ctx.storage.write().unwrap();
                    let existed = storage.delete(&key).is_some();

                    Ok(if existed { 1 } else { 0 })
                },
            )
            .map_err(|e| PluginError::Internal(e.to_string()))?;

        // plugin_storage_exists: Check if key exists
        linker
            .func_wrap(
                "env",
                "plugin_storage_exists",
                |mut caller: Caller<'_, HostContext>,
                 key_ptr: u32,
                 key_len: u32|
                 -> Result<u32, wasmtime::Error> {
                    let ctx = caller.data();

                    // Check capability
                    if !ctx.has_capability(Capability::StorageRead) {
                        return Ok(0);
                    }

                    // Consume gas
                    if ctx.gas_meter.consume_storage_exists().is_err() {
                        return Err(wasmtime::Error::msg("out of gas"));
                    }

                    // Read key
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;

                    let key = read_string(&memory, &caller, key_ptr, key_len)?;

                    // Check existence
                    let ctx = caller.data();
                    let storage = ctx.storage.read().unwrap();
                    let exists = storage.exists(&key);

                    Ok(if exists { 1 } else { 0 })
                },
            )
            .map_err(|e| PluginError::Internal(e.to_string()))?;

        Ok(())
    }

    /// Register event-related host functions
    fn register_event_functions(linker: &mut Linker<HostContext>) -> PluginResult<()> {
        // plugin_emit_event: Emit a blockchain event
        linker
            .func_wrap(
                "env",
                "plugin_emit_event",
                |mut caller: Caller<'_, HostContext>,
                 topic_ptr: u32,
                 topic_len: u32,
                 data_ptr: u32,
                 data_len: u32|
                 -> Result<u32, wasmtime::Error> {
                    let ctx = caller.data();

                    // Check capability
                    if !ctx.has_capability(Capability::EmitEvent) {
                        return Ok(0);
                    }

                    // Consume gas
                    if ctx
                        .gas_meter
                        .consume_emit_event(data_len as usize)
                        .is_err()
                    {
                        return Err(wasmtime::Error::msg("out of gas"));
                    }

                    // Read topic and data
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;

                    let topic = read_string(&memory, &caller, topic_ptr, topic_len)?;
                    let data = read_bytes(&memory, &caller, data_ptr, data_len)?;

                    // Create and store event
                    let ctx = caller.data();
                    let event = PluginEvent::new(&topic, data, &ctx.plugin_id);

                    let mut events = ctx.events.write().unwrap();
                    events.push(event);

                    debug!(
                        plugin_id = %ctx.plugin_id,
                        topic = %topic,
                        "Plugin emitted event"
                    );

                    Ok(1) // Success
                },
            )
            .map_err(|e| PluginError::Internal(e.to_string()))?;

        Ok(())
    }

    /// Register system-related host functions
    fn register_system_functions(linker: &mut Linker<HostContext>) -> PluginResult<()> {
        // plugin_get_block_height: Get current block height
        linker
            .func_wrap(
                "env",
                "plugin_get_block_height",
                |caller: Caller<'_, HostContext>| -> Result<u64, wasmtime::Error> {
                    let ctx = caller.data();

                    // Check capability
                    if !ctx.has_capability(Capability::GetBlockHeight) {
                        return Ok(0);
                    }

                    // Consume gas
                    if ctx.gas_meter.consume_get_block_height().is_err() {
                        return Err(wasmtime::Error::msg("out of gas"));
                    }

                    Ok(ctx.block_height)
                },
            )
            .map_err(|e| PluginError::Internal(e.to_string()))?;

        // plugin_get_timestamp: Get current timestamp
        linker
            .func_wrap(
                "env",
                "plugin_get_timestamp",
                |caller: Caller<'_, HostContext>| -> Result<u64, wasmtime::Error> {
                    let ctx = caller.data();

                    // Check capability
                    if !ctx.has_capability(Capability::GetTimestamp) {
                        return Ok(0);
                    }

                    // Consume gas
                    if ctx.gas_meter.consume_get_timestamp().is_err() {
                        return Err(wasmtime::Error::msg("out of gas"));
                    }

                    Ok(ctx.timestamp)
                },
            )
            .map_err(|e| PluginError::Internal(e.to_string()))?;

        // plugin_log: Log a message
        linker
            .func_wrap(
                "env",
                "plugin_log",
                |mut caller: Caller<'_, HostContext>,
                 level: u32,
                 msg_ptr: u32,
                 msg_len: u32|
                 -> Result<(), wasmtime::Error> {
                    let ctx = caller.data();

                    // Check capability
                    if !ctx.has_capability(Capability::Log) {
                        return Ok(());
                    }

                    // Consume gas
                    if ctx.gas_meter.consume_log(msg_len as usize).is_err() {
                        return Err(wasmtime::Error::msg("out of gas"));
                    }

                    // Read message
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;

                    let message = read_string(&memory, &caller, msg_ptr, msg_len)?;
                    let plugin_id = caller.data().plugin_id.clone();

                    // Log with appropriate level
                    match LogLevel::from(level) {
                        LogLevel::Trace => trace!(plugin = %plugin_id, "{}", message),
                        LogLevel::Debug => debug!(plugin = %plugin_id, "{}", message),
                        LogLevel::Info => info!(plugin = %plugin_id, "{}", message),
                        LogLevel::Warn => warn!(plugin = %plugin_id, "{}", message),
                        LogLevel::Error => error!(plugin = %plugin_id, "{}", message),
                    }

                    Ok(())
                },
            )
            .map_err(|e| PluginError::Internal(e.to_string()))?;

        Ok(())
    }

    /// Register cryptography-related host functions
    fn register_crypto_functions(linker: &mut Linker<HostContext>) -> PluginResult<()> {
        // plugin_sha3_256: Compute SHA3-256 hash
        linker
            .func_wrap(
                "env",
                "plugin_sha3_256",
                |mut caller: Caller<'_, HostContext>,
                 data_ptr: u32,
                 data_len: u32,
                 out_ptr: u32|
                 -> Result<u32, wasmtime::Error> {
                    let ctx = caller.data();

                    // Check capability
                    if !ctx.has_capability(Capability::ComputeHash) {
                        return Ok(0);
                    }

                    // Consume gas
                    if ctx.gas_meter.consume_sha3(data_len as usize).is_err() {
                        return Err(wasmtime::Error::msg("out of gas"));
                    }

                    // Read data
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;

                    let data = read_bytes(&memory, &caller, data_ptr, data_len)?;

                    // Compute hash
                    let mut hasher = Sha3_256::new();
                    hasher.update(&data);
                    let hash: [u8; 32] = hasher.finalize().into();

                    // Write hash to output
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;

                    write_bytes(&memory, &mut caller, out_ptr, &hash)?;

                    Ok(1) // Success
                },
            )
            .map_err(|e| PluginError::Internal(e.to_string()))?;

        // plugin_verify_signature: Verify Ed25519 signature
        linker
            .func_wrap(
                "env",
                "plugin_verify_signature",
                |mut caller: Caller<'_, HostContext>,
                 msg_ptr: u32,
                 msg_len: u32,
                 sig_ptr: u32,
                 pubkey_ptr: u32|
                 -> Result<u32, wasmtime::Error> {
                    let ctx = caller.data();

                    // Check capability
                    if !ctx.has_capability(Capability::VerifySignature) {
                        return Ok(0);
                    }

                    // Consume gas
                    if ctx.gas_meter.consume_verify_signature().is_err() {
                        return Err(wasmtime::Error::msg("out of gas"));
                    }

                    // Read message, signature (64 bytes), and public key (32 bytes)
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;

                    let message = read_bytes(&memory, &caller, msg_ptr, msg_len)?;
                    let sig_bytes = read_bytes(&memory, &caller, sig_ptr, 64)?;
                    let pubkey_bytes = read_bytes(&memory, &caller, pubkey_ptr, 32)?;

                    // Verify signature
                    let result = verify_ed25519(&message, &sig_bytes, &pubkey_bytes);

                    Ok(if result { 1 } else { 0 })
                },
            )
            .map_err(|e| PluginError::Internal(e.to_string()))?;

        Ok(())
    }
}

/// Read bytes from WASM memory
fn read_bytes(
    memory: &Memory,
    caller: &Caller<'_, HostContext>,
    ptr: u32,
    len: u32,
) -> Result<Vec<u8>, wasmtime::Error> {
    let data = memory.data(caller);
    let start = ptr as usize;
    let end = start + len as usize;

    if end > data.len() {
        return Err(wasmtime::Error::msg("memory access out of bounds"));
    }

    Ok(data[start..end].to_vec())
}

/// Read string from WASM memory
fn read_string(
    memory: &Memory,
    caller: &Caller<'_, HostContext>,
    ptr: u32,
    len: u32,
) -> Result<String, wasmtime::Error> {
    let bytes = read_bytes(memory, caller, ptr, len)?;
    String::from_utf8(bytes).map_err(|e| wasmtime::Error::msg(format!("invalid UTF-8: {}", e)))
}

/// Write bytes to WASM memory
fn write_bytes(
    memory: &Memory,
    caller: &mut Caller<'_, HostContext>,
    ptr: u32,
    data: &[u8],
) -> Result<(), wasmtime::Error> {
    let mem_data = memory.data_mut(caller);
    let start = ptr as usize;
    let end = start + data.len();

    if end > mem_data.len() {
        return Err(wasmtime::Error::msg("memory write out of bounds"));
    }

    mem_data[start..end].copy_from_slice(data);
    Ok(())
}

/// Verify Ed25519 signature
fn verify_ed25519(message: &[u8], signature: &[u8], public_key: &[u8]) -> bool {
    use ed25519_dalek::{Signature, VerifyingKey};

    // Parse public key
    let pubkey_array: [u8; 32] = match public_key.try_into() {
        Ok(arr) => arr,
        Err(_) => return false,
    };

    let verifying_key = match VerifyingKey::from_bytes(&pubkey_array) {
        Ok(key) => key,
        Err(_) => return false,
    };

    // Parse signature
    let sig_array: [u8; 64] = match signature.try_into() {
        Ok(arr) => arr,
        Err(_) => return false,
    };

    let signature = Signature::from_bytes(&sig_array);

    // Verify
    use ed25519_dalek::Verifier;
    verifying_key.verify(message, &signature).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_context_creation() {
        let ctx = HostContext::new(
            "test-plugin".to_string(),
            CapabilitySet::basic(),
            1_000_000,
            100,
            1234567890,
        );

        assert_eq!(ctx.plugin_id, "test-plugin");
        assert!(ctx.has_capability(Capability::StorageRead));
        assert!(!ctx.has_capability(Capability::StorageWrite));
        assert_eq!(ctx.block_height, 100);
        assert_eq!(ctx.timestamp, 1234567890);
    }

    #[test]
    fn test_capability_check() {
        let ctx = HostContext::new(
            "test".to_string(),
            CapabilitySet::new(&[Capability::StorageRead]),
            1000,
            0,
            0,
        );

        assert!(ctx.require_capability(Capability::StorageRead).is_ok());
        assert!(ctx.require_capability(Capability::StorageWrite).is_err());
    }

    #[test]
    fn test_log_level_conversion() {
        assert_eq!(LogLevel::from(0), LogLevel::Trace);
        assert_eq!(LogLevel::from(1), LogLevel::Debug);
        assert_eq!(LogLevel::from(2), LogLevel::Info);
        assert_eq!(LogLevel::from(3), LogLevel::Warn);
        assert_eq!(LogLevel::from(4), LogLevel::Error);
        assert_eq!(LogLevel::from(99), LogLevel::Error); // Default to error
    }

    #[test]
    fn test_ed25519_verify() {
        use ed25519_dalek::{Signer, SigningKey};

        // Generate key pair
        let mut rng = rand::thread_rng();
        let signing_key = SigningKey::generate(&mut rng);
        let verifying_key = signing_key.verifying_key();

        // Sign message
        let message = b"test message";
        let signature = signing_key.sign(message);

        // Verify
        let result = verify_ed25519(
            message,
            signature.to_bytes().as_ref(),
            verifying_key.as_bytes(),
        );
        assert!(result);

        // Verify with wrong message fails
        let result = verify_ed25519(
            b"wrong message",
            signature.to_bytes().as_ref(),
            verifying_key.as_bytes(),
        );
        assert!(!result);
    }

    #[test]
    fn test_events() {
        let ctx = HostContext::new(
            "test".to_string(),
            CapabilitySet::all(),
            1_000_000,
            100,
            1234567890,
        );

        // Add events
        {
            let mut events = ctx.events.write().unwrap();
            events.push(PluginEvent::new("topic1", vec![1, 2, 3], "test"));
            events.push(PluginEvent::new("topic2", vec![4, 5, 6], "test"));
        }

        // Take events
        let events = ctx.take_events();
        assert_eq!(events.len(), 2);

        // Should be empty now
        let events = ctx.take_events();
        assert!(events.is_empty());
    }
}
