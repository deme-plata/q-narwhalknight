// Q-NarwhalKnight Plugin SDK
//
// This module provides the SDK types and utilities for plugin authors.
// Plugins can import these types to interact with the host environment
// through well-defined interfaces for storage, events, and blockchain data.
//
// # Plugin Development Guide
//
// Plugins are WASM modules that export specific functions and import
// host-provided functions for interacting with the blockchain.
//
// ## Required Exports
//
// Every plugin must export:
// - `plugin_init()` - Called once when the plugin is loaded
// - `plugin_handle_event(event_type: u32, data_ptr: *const u8, data_len: u32) -> i32`
//
// ## Available Imports
//
// Plugins can import functions from the host:
// - `storage_read`, `storage_write` - Key-value storage
// - `emit_event` - Publish events to subscribers
// - `get_block_height`, `get_timestamp` - Blockchain state
// - `log_message` - Logging

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// HOST IMPORTS (FFI declarations for WASM plugins)
// ============================================================================

/// Host-provided functions that plugins can import
///
/// These declarations are used by plugin authors when compiling to WASM.
/// The actual implementations are provided by the plugin executor.
pub mod imports {
    /// Read a value from plugin storage
    ///
    /// # Arguments
    /// * `key_ptr` - Pointer to key bytes
    /// * `key_len` - Length of key
    /// * `out_ptr` - Pointer to output buffer
    /// * `out_max` - Maximum bytes to write
    ///
    /// # Returns
    /// * Positive value: Number of bytes written
    /// * -1: Key not found
    /// * -2: Buffer too small
    /// * -3: Internal error
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn storage_read(
            key_ptr: *const u8,
            key_len: u32,
            out_ptr: *mut u8,
            out_max: u32,
        ) -> i32;
    }

    /// Write a value to plugin storage
    ///
    /// # Arguments
    /// * `key_ptr` - Pointer to key bytes
    /// * `key_len` - Length of key
    /// * `val_ptr` - Pointer to value bytes
    /// * `val_len` - Length of value
    ///
    /// # Returns
    /// * 0: Success
    /// * -1: Storage quota exceeded
    /// * -2: Key too large
    /// * -3: Value too large
    /// * -4: Internal error
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn storage_write(
            key_ptr: *const u8,
            key_len: u32,
            val_ptr: *const u8,
            val_len: u32,
        ) -> i32;
    }

    /// Delete a value from plugin storage
    ///
    /// # Arguments
    /// * `key_ptr` - Pointer to key bytes
    /// * `key_len` - Length of key
    ///
    /// # Returns
    /// * 0: Success (or key didn't exist)
    /// * -1: Internal error
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn storage_delete(key_ptr: *const u8, key_len: u32) -> i32;
    }

    /// Emit an event to subscribers
    ///
    /// # Arguments
    /// * `topic_ptr` - Pointer to topic string
    /// * `topic_len` - Length of topic
    /// * `data_ptr` - Pointer to event data
    /// * `data_len` - Length of event data
    ///
    /// # Returns
    /// * 0: Success
    /// * -1: Topic invalid
    /// * -2: Data too large
    /// * -3: Rate limit exceeded
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn emit_event(
            topic_ptr: *const u8,
            topic_len: u32,
            data_ptr: *const u8,
            data_len: u32,
        ) -> i32;
    }

    /// Get the current block height
    ///
    /// # Returns
    /// Current blockchain height
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn get_block_height() -> u64;
    }

    /// Get the current timestamp (Unix seconds)
    ///
    /// # Returns
    /// Current Unix timestamp
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn get_timestamp() -> u64;
    }

    /// Log a message
    ///
    /// # Arguments
    /// * `level` - Log level (0=trace, 1=debug, 2=info, 3=warn, 4=error)
    /// * `msg_ptr` - Pointer to message string
    /// * `msg_len` - Length of message
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn log_message(level: u32, msg_ptr: *const u8, msg_len: u32);
    }

    /// Get the current network ID
    ///
    /// # Arguments
    /// * `out_ptr` - Pointer to output buffer
    /// * `out_max` - Maximum bytes to write
    ///
    /// # Returns
    /// Number of bytes written, or -1 on error
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn get_network_id(out_ptr: *mut u8, out_max: u32) -> i32;
    }

    /// Get the plugin's own ID
    ///
    /// # Arguments
    /// * `out_ptr` - Pointer to output buffer
    /// * `out_max` - Maximum bytes to write
    ///
    /// # Returns
    /// Number of bytes written, or -1 on error
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn get_plugin_id(out_ptr: *mut u8, out_max: u32) -> i32;
    }

    /// Call another plugin
    ///
    /// # Arguments
    /// * `plugin_id_ptr` - Pointer to target plugin ID
    /// * `plugin_id_len` - Length of plugin ID
    /// * `method_ptr` - Pointer to method name
    /// * `method_len` - Length of method name
    /// * `args_ptr` - Pointer to arguments data
    /// * `args_len` - Length of arguments
    /// * `out_ptr` - Pointer to output buffer
    /// * `out_max` - Maximum bytes to write
    ///
    /// # Returns
    /// Number of bytes written, or negative error code
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn call_plugin(
            plugin_id_ptr: *const u8,
            plugin_id_len: u32,
            method_ptr: *const u8,
            method_len: u32,
            args_ptr: *const u8,
            args_len: u32,
            out_ptr: *mut u8,
            out_max: u32,
        ) -> i32;
    }

    /// Get a block by height
    ///
    /// # Arguments
    /// * `height` - Block height to fetch
    /// * `out_ptr` - Pointer to output buffer
    /// * `out_max` - Maximum bytes to write
    ///
    /// # Returns
    /// Number of bytes written, or -1 if block not found
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn get_block_by_height(height: u64, out_ptr: *mut u8, out_max: u32) -> i32;
    }

    /// Get a transaction by hash
    ///
    /// # Arguments
    /// * `hash_ptr` - Pointer to transaction hash (32 bytes)
    /// * `out_ptr` - Pointer to output buffer
    /// * `out_max` - Maximum bytes to write
    ///
    /// # Returns
    /// Number of bytes written, or -1 if transaction not found
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn get_transaction(hash_ptr: *const u8, out_ptr: *mut u8, out_max: u32) -> i32;
    }

    /// Get an account balance
    ///
    /// # Arguments
    /// * `address_ptr` - Pointer to address (32 bytes)
    /// * `token_ptr` - Pointer to token address (32 bytes, null for native)
    ///
    /// # Returns
    /// Balance as u64 (lower 64 bits of u128)
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn get_balance(address_ptr: *const u8, token_ptr: *const u8) -> u64;
    }

    /// Get high 64 bits of a balance (for u128 support)
    ///
    /// # Arguments
    /// * `address_ptr` - Pointer to address (32 bytes)
    /// * `token_ptr` - Pointer to token address (32 bytes, null for native)
    ///
    /// # Returns
    /// Balance high bits as u64 (upper 64 bits of u128)
    #[link(wasm_import_module = "q_plugin_host")]
    extern "C" {
        pub fn get_balance_high(address_ptr: *const u8, token_ptr: *const u8) -> u64;
    }
}

// ============================================================================
// SAFE WRAPPERS (for use in Rust plugins)
// ============================================================================

/// Plugin context providing safe access to host functions
///
/// This is the main interface plugins use to interact with the host.
pub struct PluginContext {
    /// Storage wrapper
    storage: Storage,
    /// Event emitter wrapper
    events: EventEmitter,
    /// Whether we're in a read-only context
    read_only: bool,
}

impl PluginContext {
    /// Create a new plugin context
    ///
    /// # Safety
    /// This should only be called from plugin initialization code.
    pub fn new() -> Self {
        Self {
            storage: Storage::new(),
            events: EventEmitter::new(),
            read_only: false,
        }
    }

    /// Create a read-only plugin context
    pub fn read_only() -> Self {
        Self {
            storage: Storage::new(),
            events: EventEmitter::new(),
            read_only: true,
        }
    }

    /// Get the storage interface
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Get mutable storage interface (fails if read-only)
    pub fn storage_mut(&mut self) -> Result<&mut Storage, PluginSdkError> {
        if self.read_only {
            Err(PluginSdkError::ReadOnlyContext)
        } else {
            Ok(&mut self.storage)
        }
    }

    /// Get the event emitter interface
    pub fn events(&self) -> &EventEmitter {
        &self.events
    }

    /// Get event emitter (fails if read-only)
    pub fn events_mut(&mut self) -> Result<&mut EventEmitter, PluginSdkError> {
        if self.read_only {
            Err(PluginSdkError::ReadOnlyContext)
        } else {
            Ok(&mut self.events)
        }
    }

    /// Get current block height
    pub fn block_height(&self) -> u64 {
        unsafe { imports::get_block_height() }
    }

    /// Get current timestamp
    pub fn timestamp(&self) -> u64 {
        unsafe { imports::get_timestamp() }
    }

    /// Log a message at info level
    pub fn log_info(&self, msg: &str) {
        log(LogLevel::Info, msg);
    }

    /// Log a message at debug level
    pub fn log_debug(&self, msg: &str) {
        log(LogLevel::Debug, msg);
    }

    /// Log a message at warning level
    pub fn log_warn(&self, msg: &str) {
        log(LogLevel::Warn, msg);
    }

    /// Log a message at error level
    pub fn log_error(&self, msg: &str) {
        log(LogLevel::Error, msg);
    }

    /// Get an account balance (full u128)
    pub fn get_balance(&self, address: &[u8; 32], token: Option<&[u8; 32]>) -> u128 {
        unsafe {
            let token_ptr = token.map(|t| t.as_ptr()).unwrap_or(std::ptr::null());
            let low = imports::get_balance(address.as_ptr(), token_ptr);
            let high = imports::get_balance_high(address.as_ptr(), token_ptr);
            ((high as u128) << 64) | (low as u128)
        }
    }
}

impl Default for PluginContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// STORAGE INTERFACE
// ============================================================================

/// Storage interface for plugin data persistence
///
/// Provides key-value storage scoped to the plugin.
pub struct Storage {
    /// In-memory cache for reads
    cache: HashMap<Vec<u8>, Vec<u8>>,
}

impl Storage {
    /// Create a new storage interface
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Read a value from storage
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        // Check cache first
        if let Some(value) = self.cache.get(key) {
            return Some(value.clone());
        }

        // Read from host
        let mut buffer = vec![0u8; 65536]; // 64KB max value
        let result = unsafe {
            imports::storage_read(
                key.as_ptr(),
                key.len() as u32,
                buffer.as_mut_ptr(),
                buffer.len() as u32,
            )
        };

        if result > 0 {
            buffer.truncate(result as usize);
            Some(buffer)
        } else {
            None
        }
    }

    /// Read and deserialize a value
    pub fn get_json<T: for<'de> Deserialize<'de>>(&self, key: &[u8]) -> Option<T> {
        self.get(key)
            .and_then(|data| serde_json::from_slice(&data).ok())
    }

    /// Write a value to storage
    pub fn set(&mut self, key: &[u8], value: &[u8]) -> Result<(), PluginSdkError> {
        let result = unsafe {
            imports::storage_write(
                key.as_ptr(),
                key.len() as u32,
                value.as_ptr(),
                value.len() as u32,
            )
        };

        match result {
            0 => {
                // Update cache
                self.cache.insert(key.to_vec(), value.to_vec());
                Ok(())
            }
            -1 => Err(PluginSdkError::StorageQuotaExceeded),
            -2 => Err(PluginSdkError::KeyTooLarge),
            -3 => Err(PluginSdkError::ValueTooLarge),
            _ => Err(PluginSdkError::InternalError),
        }
    }

    /// Serialize and write a value
    pub fn set_json<T: Serialize>(&mut self, key: &[u8], value: &T) -> Result<(), PluginSdkError> {
        let data = serde_json::to_vec(value).map_err(|_| PluginSdkError::SerializationError)?;
        self.set(key, &data)
    }

    /// Delete a value from storage
    pub fn delete(&mut self, key: &[u8]) -> Result<(), PluginSdkError> {
        let result = unsafe { imports::storage_delete(key.as_ptr(), key.len() as u32) };

        if result == 0 {
            self.cache.remove(key);
            Ok(())
        } else {
            Err(PluginSdkError::InternalError)
        }
    }

    /// Check if a key exists
    pub fn exists(&self, key: &[u8]) -> bool {
        self.get(key).is_some()
    }

    /// Clear the in-memory cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for Storage {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EVENT EMITTER
// ============================================================================

/// Event emitter for publishing plugin events
pub struct EventEmitter {
    /// Number of events emitted this execution
    events_emitted: u32,
    /// Maximum events per execution
    max_events: u32,
}

impl EventEmitter {
    /// Create a new event emitter
    pub fn new() -> Self {
        Self {
            events_emitted: 0,
            max_events: 100, // Default limit
        }
    }

    /// Emit an event
    pub fn emit(&mut self, topic: &str, data: &[u8]) -> Result<(), PluginSdkError> {
        if self.events_emitted >= self.max_events {
            return Err(PluginSdkError::RateLimitExceeded);
        }

        let result = unsafe {
            imports::emit_event(
                topic.as_ptr(),
                topic.len() as u32,
                data.as_ptr(),
                data.len() as u32,
            )
        };

        match result {
            0 => {
                self.events_emitted += 1;
                Ok(())
            }
            -1 => Err(PluginSdkError::InvalidTopic),
            -2 => Err(PluginSdkError::DataTooLarge),
            -3 => Err(PluginSdkError::RateLimitExceeded),
            _ => Err(PluginSdkError::InternalError),
        }
    }

    /// Emit a JSON event
    pub fn emit_json<T: Serialize>(&mut self, topic: &str, data: &T) -> Result<(), PluginSdkError> {
        let bytes = serde_json::to_vec(data).map_err(|_| PluginSdkError::SerializationError)?;
        self.emit(topic, &bytes)
    }

    /// Get number of events emitted
    pub fn events_count(&self) -> u32 {
        self.events_emitted
    }

    /// Reset event counter (called between executions)
    pub fn reset(&mut self) {
        self.events_emitted = 0;
    }
}

impl Default for EventEmitter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LOGGING
// ============================================================================

/// Log level for plugin logging
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

/// Log a message at the specified level
pub fn log(level: LogLevel, message: &str) {
    unsafe {
        imports::log_message(level as u32, message.as_ptr(), message.len() as u32);
    }
}

/// Convenience macros for logging
#[macro_export]
macro_rules! log_trace {
    ($($arg:tt)*) => {
        $crate::sdk::log($crate::sdk::LogLevel::Trace, &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {
        $crate::sdk::log($crate::sdk::LogLevel::Debug, &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        $crate::sdk::log($crate::sdk::LogLevel::Info, &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        $crate::sdk::log($crate::sdk::LogLevel::Warn, &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        $crate::sdk::log($crate::sdk::LogLevel::Error, &format!($($arg)*))
    };
}

// ============================================================================
// SDK ERROR TYPES
// ============================================================================

/// Errors that can occur when using SDK functions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginSdkError {
    /// Operation not allowed in read-only context
    ReadOnlyContext,
    /// Storage quota exceeded
    StorageQuotaExceeded,
    /// Key is too large
    KeyTooLarge,
    /// Value is too large
    ValueTooLarge,
    /// Data too large for operation
    DataTooLarge,
    /// Invalid topic name
    InvalidTopic,
    /// Rate limit exceeded
    RateLimitExceeded,
    /// Serialization/deserialization error
    SerializationError,
    /// Internal host error
    InternalError,
    /// Key not found
    KeyNotFound,
    /// Plugin not found
    PluginNotFound,
    /// Method not found
    MethodNotFound,
}

impl std::fmt::Display for PluginSdkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadOnlyContext => write!(f, "Operation not allowed in read-only context"),
            Self::StorageQuotaExceeded => write!(f, "Storage quota exceeded"),
            Self::KeyTooLarge => write!(f, "Key too large"),
            Self::ValueTooLarge => write!(f, "Value too large"),
            Self::DataTooLarge => write!(f, "Data too large"),
            Self::InvalidTopic => write!(f, "Invalid topic name"),
            Self::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            Self::SerializationError => write!(f, "Serialization error"),
            Self::InternalError => write!(f, "Internal error"),
            Self::KeyNotFound => write!(f, "Key not found"),
            Self::PluginNotFound => write!(f, "Plugin not found"),
            Self::MethodNotFound => write!(f, "Method not found"),
        }
    }
}

impl std::error::Error for PluginSdkError {}

// ============================================================================
// EVENT TYPES FOR PLUGINS
// ============================================================================

/// Event type discriminants passed to plugins
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum EventType {
    /// Block received event
    BlockReceived = 1,
    /// Transaction received event
    TransactionReceived = 2,
    /// Consensus round started
    ConsensusRound = 3,
    /// Before transaction execution
    BeforeTxExecution = 4,
    /// After transaction execution
    AfterTxExecution = 5,
    /// Peer connected
    PeerConnected = 6,
    /// Peer disconnected
    PeerDisconnected = 7,
    /// Timer event
    Timer = 8,
    /// Node starting
    NodeStart = 9,
    /// Node shutting down
    NodeShutdown = 10,
    /// Custom event (check data for type)
    Custom = 100,
}

impl TryFrom<u32> for EventType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(EventType::BlockReceived),
            2 => Ok(EventType::TransactionReceived),
            3 => Ok(EventType::ConsensusRound),
            4 => Ok(EventType::BeforeTxExecution),
            5 => Ok(EventType::AfterTxExecution),
            6 => Ok(EventType::PeerConnected),
            7 => Ok(EventType::PeerDisconnected),
            8 => Ok(EventType::Timer),
            9 => Ok(EventType::NodeStart),
            10 => Ok(EventType::NodeShutdown),
            100 => Ok(EventType::Custom),
            _ => Err(()),
        }
    }
}

// ============================================================================
// PLUGIN RETURN CODES
// ============================================================================

/// Return codes from plugin execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum PluginReturnCode {
    /// Success
    Ok = 0,
    /// Generic error
    Error = -1,
    /// Event should be propagated to other plugins
    Propagate = 1,
    /// Event should not be propagated (consumed)
    Consume = 2,
    /// Transaction should be rejected
    RejectTransaction = -10,
    /// Block should be rejected
    RejectBlock = -11,
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Get the plugin's own ID
pub fn get_plugin_id() -> Option<String> {
    let mut buffer = vec![0u8; 256];
    let result =
        unsafe { imports::get_plugin_id(buffer.as_mut_ptr(), buffer.len() as u32) };

    if result > 0 {
        buffer.truncate(result as usize);
        String::from_utf8(buffer).ok()
    } else {
        None
    }
}

/// Get the network ID
pub fn get_network_id() -> Option<String> {
    let mut buffer = vec![0u8; 256];
    let result =
        unsafe { imports::get_network_id(buffer.as_mut_ptr(), buffer.len() as u32) };

    if result > 0 {
        buffer.truncate(result as usize);
        String::from_utf8(buffer).ok()
    } else {
        None
    }
}

/// Call another plugin
pub fn call_plugin(
    plugin_id: &str,
    method: &str,
    args: &[u8],
) -> Result<Vec<u8>, PluginSdkError> {
    let mut buffer = vec![0u8; 65536]; // 64KB max response

    let result = unsafe {
        imports::call_plugin(
            plugin_id.as_ptr(),
            plugin_id.len() as u32,
            method.as_ptr(),
            method.len() as u32,
            args.as_ptr(),
            args.len() as u32,
            buffer.as_mut_ptr(),
            buffer.len() as u32,
        )
    };

    if result > 0 {
        buffer.truncate(result as usize);
        Ok(buffer)
    } else {
        match result {
            -1 => Err(PluginSdkError::PluginNotFound),
            -2 => Err(PluginSdkError::MethodNotFound),
            _ => Err(PluginSdkError::InternalError),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_conversion() {
        assert_eq!(EventType::try_from(1), Ok(EventType::BlockReceived));
        assert_eq!(EventType::try_from(2), Ok(EventType::TransactionReceived));
        assert_eq!(EventType::try_from(100), Ok(EventType::Custom));
        assert_eq!(EventType::try_from(999), Err(()));
    }

    #[test]
    fn test_log_level_repr() {
        assert_eq!(LogLevel::Trace as u32, 0);
        assert_eq!(LogLevel::Debug as u32, 1);
        assert_eq!(LogLevel::Info as u32, 2);
        assert_eq!(LogLevel::Warn as u32, 3);
        assert_eq!(LogLevel::Error as u32, 4);
    }

    #[test]
    fn test_plugin_return_code() {
        assert_eq!(PluginReturnCode::Ok as i32, 0);
        assert_eq!(PluginReturnCode::Error as i32, -1);
        assert_eq!(PluginReturnCode::Propagate as i32, 1);
        assert_eq!(PluginReturnCode::RejectTransaction as i32, -10);
    }

    #[test]
    fn test_sdk_error_display() {
        let err = PluginSdkError::StorageQuotaExceeded;
        assert_eq!(err.to_string(), "Storage quota exceeded");

        let err = PluginSdkError::ReadOnlyContext;
        assert_eq!(err.to_string(), "Operation not allowed in read-only context");
    }

    #[test]
    fn test_storage_new() {
        let storage = Storage::new();
        assert!(storage.cache.is_empty());
    }

    #[test]
    fn test_event_emitter_new() {
        let emitter = EventEmitter::new();
        assert_eq!(emitter.events_count(), 0);
    }

    #[test]
    fn test_plugin_context_new() {
        let ctx = PluginContext::new();
        assert!(!ctx.read_only);

        let ctx = PluginContext::read_only();
        assert!(ctx.read_only);
    }
}
