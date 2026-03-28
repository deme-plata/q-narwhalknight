//! Safe Rust wrappers around the raw host-function FFI provided by the
//! Q-NarwhalKnight `PluginExecutor`.
//!
//! On `wasm32` targets the extern "C" block resolves to the real host imports
//! injected by wasmtime.  On native targets a thread-local `HashMap` stands in
//! so that unit tests (and `cargo check`) work without a WASM runtime.

// ---------------------------------------------------------------------------
// Platform-specific FFI
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
extern "C" {
    /// Read a value from namespaced plugin storage.
    /// On success the host writes the value bytes starting at `out_ptr` and
    /// stores the byte-length as a little-endian u32 at `out_len_ptr`.
    /// Returns 1 on success, 0 if the key was not found.
    fn plugin_storage_read(
        key_ptr: u32,
        key_len: u32,
        out_ptr: u32,
        out_len_ptr: u32,
    ) -> u32;

    /// Write a value into namespaced plugin storage.
    /// Returns 1 on success, 0 on capability denial.
    fn plugin_storage_write(
        key_ptr: u32,
        key_len: u32,
        val_ptr: u32,
        val_len: u32,
    ) -> u32;

    /// Delete a key from plugin storage.
    /// Returns 1 if the key existed, 0 otherwise.
    fn plugin_storage_delete(key_ptr: u32, key_len: u32) -> u32;

    /// Check whether a key exists in plugin storage.
    /// Returns 1 if it exists, 0 otherwise.
    fn plugin_storage_exists(key_ptr: u32, key_len: u32) -> u32;

    /// Emit a blockchain event with the given topic and data payload.
    /// Returns 1 on success, 0 on capability denial.
    fn plugin_emit_event(
        topic_ptr: u32,
        topic_len: u32,
        data_ptr: u32,
        data_len: u32,
    ) -> u32;

    /// Return the current block height as seen by the executor.
    fn plugin_get_block_height() -> u64;

    /// Return the current block timestamp (seconds since UNIX epoch).
    fn plugin_get_timestamp() -> u64;

    /// Compute the SHA3-256 digest of `data_ptr[..data_len]` and write the
    /// 32-byte result to `out_ptr`.  Returns 1 on success.
    fn plugin_sha3_256(data_ptr: u32, data_len: u32, out_ptr: u32) -> u32;

    /// Log a message at the given level (0=trace .. 4=error).
    fn plugin_log(level: u32, msg_ptr: u32, msg_len: u32);
}

// ---------------------------------------------------------------------------
// Native fallback (thread-local HashMap)
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
mod native_fallback {
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        /// In-memory storage used during native tests.
        pub static STORAGE: RefCell<HashMap<String, Vec<u8>>> = RefCell::new(HashMap::new());

        /// Simulated block height for native tests.
        pub static BLOCK_HEIGHT: RefCell<u64> = const { RefCell::new(0) };

        /// Simulated timestamp for native tests.
        pub static TIMESTAMP: RefCell<u64> = const { RefCell::new(0) };

        /// Collected log messages for test assertions.
        pub static LOG_MESSAGES: RefCell<Vec<(u32, String)>> = RefCell::new(Vec::new());

        /// Collected events for test assertions.
        pub static EVENTS: RefCell<Vec<(String, Vec<u8>)>> = RefCell::new(Vec::new());
    }

    /// Reset all native fallback state (call between tests).
    pub fn reset() {
        STORAGE.with(|s| s.borrow_mut().clear());
        BLOCK_HEIGHT.with(|h| *h.borrow_mut() = 0);
        TIMESTAMP.with(|t| *t.borrow_mut() = 0);
        LOG_MESSAGES.with(|l| l.borrow_mut().clear());
        EVENTS.with(|e| e.borrow_mut().clear());
    }

    /// Set the simulated block height (test helper).
    pub fn set_block_height(h: u64) {
        BLOCK_HEIGHT.with(|bh| *bh.borrow_mut() = h);
    }

    /// Set the simulated timestamp (test helper).
    pub fn set_timestamp(t: u64) {
        TIMESTAMP.with(|ts| *ts.borrow_mut() = t);
    }

    /// Read collected log messages (test helper).
    pub fn take_logs() -> Vec<(u32, String)> {
        LOG_MESSAGES.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }

    /// Read collected events (test helper).
    pub fn take_events() -> Vec<(String, Vec<u8>)> {
        EVENTS.with(|e| std::mem::take(&mut *e.borrow_mut()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native_fallback::{reset, set_block_height, set_timestamp, take_events, take_logs};

// ---------------------------------------------------------------------------
// Safe public API
// ---------------------------------------------------------------------------

/// Maximum value size we will attempt to read back from the host (1 MiB).
/// Prevents a buggy host from causing unbounded allocation.
#[cfg(target_arch = "wasm32")]
const MAX_VALUE_SIZE: usize = 1 << 20;

/// Read a value from plugin storage.  Returns `None` if the key does not
/// exist.
pub fn storage_read(key: &str) -> Option<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    {
        // Allocate a generous output buffer.  If the value is larger than
        // this the host will simply not write it (returns 0).
        let buf_size: usize = MAX_VALUE_SIZE;
        let mut buf = vec![0u8; buf_size];
        let mut out_len: u32 = 0;

        let ok = unsafe {
            plugin_storage_read(
                key.as_ptr() as u32,
                key.len() as u32,
                buf.as_mut_ptr() as u32,
                &mut out_len as *mut u32 as u32,
            )
        };

        if ok == 0 {
            return None;
        }
        let len = out_len as usize;
        if len > buf_size {
            return None;
        }
        buf.truncate(len);
        Some(buf)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native_fallback::STORAGE.with(|s| s.borrow().get(key).cloned())
    }
}

/// Write a value to plugin storage.
pub fn storage_write(key: &str, value: &[u8]) {
    #[cfg(target_arch = "wasm32")]
    {
        unsafe {
            plugin_storage_write(
                key.as_ptr() as u32,
                key.len() as u32,
                value.as_ptr() as u32,
                value.len() as u32,
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native_fallback::STORAGE.with(|s| {
            s.borrow_mut().insert(key.to_string(), value.to_vec());
        });
    }
}

/// Delete a key from plugin storage.  Returns `true` if the key existed.
pub fn storage_delete(key: &str) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let r = unsafe {
            plugin_storage_delete(key.as_ptr() as u32, key.len() as u32)
        };
        r != 0
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native_fallback::STORAGE.with(|s| s.borrow_mut().remove(key).is_some())
    }
}

/// Check whether a key exists in plugin storage.
pub fn storage_exists(key: &str) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let r = unsafe {
            plugin_storage_exists(key.as_ptr() as u32, key.len() as u32)
        };
        r != 0
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native_fallback::STORAGE.with(|s| s.borrow().contains_key(key))
    }
}

/// Emit a blockchain event with the given topic and arbitrary data payload.
pub fn emit_event(topic: &str, data: &[u8]) {
    #[cfg(target_arch = "wasm32")]
    {
        unsafe {
            plugin_emit_event(
                topic.as_ptr() as u32,
                topic.len() as u32,
                data.as_ptr() as u32,
                data.len() as u32,
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native_fallback::EVENTS.with(|e| {
            e.borrow_mut().push((topic.to_string(), data.to_vec()));
        });
    }
}

/// Return the current block height.
pub fn get_block_height() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        unsafe { plugin_get_block_height() }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native_fallback::BLOCK_HEIGHT.with(|h| *h.borrow())
    }
}

/// Return the current block timestamp (seconds since UNIX epoch).
pub fn get_timestamp() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        unsafe { plugin_get_timestamp() }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native_fallback::TIMESTAMP.with(|t| *t.borrow())
    }
}

/// Compute the SHA3-256 digest of `data` and return the 32-byte hash.
pub fn sha3_256(data: &[u8]) -> [u8; 32] {
    #[cfg(target_arch = "wasm32")]
    {
        let mut out = [0u8; 32];
        unsafe {
            plugin_sha3_256(
                data.as_ptr() as u32,
                data.len() as u32,
                out.as_mut_ptr() as u32,
            );
        }
        out
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Native fallback: use a simple stand-in.  We do NOT pull in sha3
        // as a dependency for the plugin crate -- the real hash comes from
        // the host.  For tests we just hash via a basic mixing function so
        // the output is deterministic and non-trivial.
        let mut h = [0u8; 32];
        let mut state: u64 = 0xcafe_babe_dead_beef;
        for (i, &b) in data.iter().enumerate() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(b as u64);
            h[i % 32] ^= (state >> 32) as u8;
        }
        // Spread entropy across all 32 bytes.
        for i in 0..32 {
            h[i] = h[i].wrapping_add(state as u8);
            state = state.rotate_left(7).wrapping_add(h[i] as u64);
        }
        h
    }
}

/// Log a message at the INFO level (level = 2).
pub fn log_info(msg: &str) {
    log_at_level(2, msg);
}

/// Log a message at the WARN level (level = 3).
pub fn log_warn(msg: &str) {
    log_at_level(3, msg);
}

/// Log a message at the ERROR level (level = 4).
pub fn log_error(msg: &str) {
    log_at_level(4, msg);
}

/// Log a message at the DEBUG level (level = 1).
pub fn log_debug(msg: &str) {
    log_at_level(1, msg);
}

/// Log a message at an explicit numeric level (0=trace .. 4=error).
fn log_at_level(level: u32, msg: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        unsafe {
            plugin_log(level, msg.as_ptr() as u32, msg.len() as u32);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native_fallback::LOG_MESSAGES.with(|l| {
            l.borrow_mut().push((level, msg.to_string()));
        });
    }
}

// ---------------------------------------------------------------------------
// Tests (native only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_round_trip() {
        reset();
        assert!(storage_read("k1").is_none());
        assert!(!storage_exists("k1"));

        storage_write("k1", b"hello");
        assert!(storage_exists("k1"));
        assert_eq!(storage_read("k1").unwrap(), b"hello");

        assert!(storage_delete("k1"));
        assert!(!storage_exists("k1"));
        assert!(!storage_delete("k1"));
    }

    #[test]
    fn block_height_and_timestamp() {
        reset();
        assert_eq!(get_block_height(), 0);
        assert_eq!(get_timestamp(), 0);

        set_block_height(42);
        set_timestamp(1_700_000_000);
        assert_eq!(get_block_height(), 42);
        assert_eq!(get_timestamp(), 1_700_000_000);
    }

    #[test]
    fn sha3_deterministic() {
        reset();
        let h1 = sha3_256(b"test data");
        let h2 = sha3_256(b"test data");
        assert_eq!(h1, h2);
        // Different input produces different output.
        let h3 = sha3_256(b"other data");
        assert_ne!(h1, h3);
    }

    #[test]
    fn logging_collects_messages() {
        reset();
        log_info("hello");
        log_warn("careful");
        log_error("boom");

        let logs = take_logs();
        assert_eq!(logs.len(), 3);
        assert_eq!(logs[0], (2, "hello".to_string()));
        assert_eq!(logs[1], (3, "careful".to_string()));
        assert_eq!(logs[2], (4, "boom".to_string()));
    }

    #[test]
    fn events_collected() {
        reset();
        emit_event("crown_ash:battle", b"payload");

        let evts = take_events();
        assert_eq!(evts.len(), 1);
        assert_eq!(evts[0].0, "crown_ash:battle");
        assert_eq!(evts[0].1, b"payload");
    }
}
