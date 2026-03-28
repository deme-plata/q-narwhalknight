//! Crown & Ash -- WASM plugin for on-chain game simulation.
//!
//! This crate compiles to a `cdylib` that the Q-NarwhalKnight `PluginExecutor`
//! loads via wasmtime.  It wraps the pure simulation engine (`crown-ash-sim`)
//! and exposes four entry points that the executor calls:
//!
//! | Entry point      | Purpose                                         |
//! |------------------|-------------------------------------------------|
//! | `on_init`        | Create a new game world from a `WorldConfig`.   |
//! | `on_tick`        | Advance the simulation by one game turn.         |
//! | `process_action` | Execute a player command (`QueuedAction`).      |
//! | `query_state`    | Read-only query; returns JSON.                   |
//!
//! Each entry point follows the plugin ABI:
//!
//! ```text
//! fn(input_ptr: u32, input_len: u32, out_len_ptr: u32) -> u32
//! ```
//!
//! where:
//! - `input_ptr` / `input_len` point to the JSON-encoded request body in
//!   WASM linear memory.
//! - The function returns `output_ptr`, which is the start of the
//!   JSON-encoded response in WASM linear memory.
//! - `out_len_ptr` is written with the response length (little-endian u32).
//!
//! The plugin also exports:
//! - `alloc(size: u32) -> u32` -- bump allocator used by the host to place
//!   input data and the output-length slot.
//! - `memory` -- the default WASM linear memory (exported automatically by
//!   the Rust compiler when targeting `wasm32`).

pub mod host;
pub mod storage;

use crown_ash_sim;
use crown_ash_types::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Bump allocator
// ---------------------------------------------------------------------------

/// Simple bump allocator for WASM memory management.
///
/// The host calls `alloc(size)` to reserve space in the WASM linear memory
/// before writing input data.  This allocator never frees -- each plugin
/// invocation gets a fresh WASM instance, so fragmentation is irrelevant.
///
/// On native builds this is unused; it exists only to satisfy the linker when
/// compiling tests.
static mut BUMP_OFFSET: u32 = 0;

/// Exported allocator.  The host calls this before writing input data or
/// allocating the output-length slot.
#[no_mangle]
pub extern "C" fn alloc(size: u32) -> u32 {
    // Align to 8 bytes for safe access to u64/f64.
    let align = 8u32;
    unsafe {
        // On the very first call, start past the first 64 KiB to avoid
        // clobbering the stack and low-address data.
        if BUMP_OFFSET == 0 {
            BUMP_OFFSET = 65536;
        }
        let aligned = (BUMP_OFFSET + align - 1) & !(align - 1);
        BUMP_OFFSET = aligned + size;
        aligned
    }
}

// ---------------------------------------------------------------------------
// Request / response types for each entry point
// ---------------------------------------------------------------------------

/// Input to `on_init`.
#[derive(Debug, Serialize, Deserialize)]
pub struct InitRequest {
    pub config: WorldConfig,
    /// Seed bytes (hex-encoded 64 chars = 32 bytes).  Derived from the block
    /// hash at the transaction that creates the game.
    pub seed_hex: String,
}

/// Output from `on_init`.
#[derive(Debug, Serialize, Deserialize)]
pub struct InitResponse {
    pub success: bool,
    pub turn: u32,
    pub province_count: usize,
    pub faction_count: usize,
    pub character_count: usize,
}

/// Input to `on_tick`.
#[derive(Debug, Serialize, Deserialize)]
pub struct TickRequest {
    /// Block hash (hex-encoded 64 chars = 32 bytes) used as RNG seed.
    pub block_hash_hex: String,
}

/// Output from `on_tick`.  Wraps the `TurnSummary` from the sim.
#[derive(Debug, Serialize, Deserialize)]
pub struct TickResponse {
    pub success: bool,
    pub summary: Option<TurnSummary>,
    pub error: Option<String>,
}

/// Input to `process_action`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ActionRequest {
    pub action: QueuedAction,
}

/// Output from `process_action`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ActionResponse {
    pub success: bool,
    pub events: Vec<GameEvent>,
    pub error: Option<String>,
}

/// Input to `query_state`.
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: crown_ash_sim::WorldQuery,
}

/// Output from `query_state`.
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub success: bool,
    /// JSON-encoded result (query-dependent shape).
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Decode a hex string into a 32-byte array.  Returns zeroed array on error.
fn hex_to_32(hex: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    let hex = hex.trim();
    if hex.len() < 64 {
        return out;
    }
    for i in 0..32 {
        let byte_hex = &hex[i * 2..i * 2 + 2];
        out[i] = u8::from_str_radix(byte_hex, 16).unwrap_or(0);
    }
    out
}

/// Read the input slice from WASM memory.
///
/// On `wasm32` the pointers are real; on native we treat them as indices into
/// a test buffer (but the entry points are only called via the host, so the
/// native path just needs to compile).
fn read_input(input_ptr: u32, input_len: u32) -> Vec<u8> {
    #[cfg(target_arch = "wasm32")]
    {
        unsafe {
            let slice = core::slice::from_raw_parts(input_ptr as *const u8, input_len as usize);
            slice.to_vec()
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // In native tests we pass the data through a thread-local.
        let _ = (input_ptr, input_len);
        NATIVE_INPUT.with(|ni| ni.borrow().clone())
    }
}

/// Write the output bytes into WASM memory and set the output-length slot.
/// Returns the pointer to the output bytes.
fn write_output(data: &[u8], out_len_ptr: u32) -> u32 {
    #[cfg(target_arch = "wasm32")]
    {
        let ptr = alloc(data.len() as u32);
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            let len_bytes = (data.len() as u32).to_le_bytes();
            core::ptr::copy_nonoverlapping(
                len_bytes.as_ptr(),
                out_len_ptr as *mut u8,
                4,
            );
        }
        ptr
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = out_len_ptr;
        // In native tests, stash the output for the test to retrieve.
        NATIVE_OUTPUT.with(|no| {
            *no.borrow_mut() = data.to_vec();
        });
        0
    }
}

// Thread-local buffers for native testing of entry points.
#[cfg(not(target_arch = "wasm32"))]
use std::cell::RefCell;

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static NATIVE_INPUT: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    static NATIVE_OUTPUT: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}

/// Helper for native tests: set the input that the next entry-point call
/// will receive.
#[cfg(not(target_arch = "wasm32"))]
pub fn test_set_input(data: &[u8]) {
    NATIVE_INPUT.with(|ni| *ni.borrow_mut() = data.to_vec());
}

/// Helper for native tests: retrieve the output written by the last
/// entry-point call.
#[cfg(not(target_arch = "wasm32"))]
pub fn test_get_output() -> Vec<u8> {
    NATIVE_OUTPUT.with(|no| no.borrow().clone())
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// **on_init** -- Create and persist a new game world.
///
/// Input: JSON-encoded `InitRequest`.
/// Output: JSON-encoded `InitResponse`.
#[no_mangle]
pub extern "C" fn on_init(input_ptr: u32, input_len: u32, out_len_ptr: u32) -> u32 {
    let input_bytes = read_input(input_ptr, input_len);

    let response = match serde_json::from_slice::<InitRequest>(&input_bytes) {
        Ok(req) => {
            let genesis_block = host::get_block_height();
            let seed = hex_to_32(&req.seed_hex);
            let mut world = crown_ash_sim::init_world(&req.config, seed);

            // Set the genesis block from the host's current block height.
            world.meta.genesis_block = genesis_block;

            let resp = InitResponse {
                success: true,
                turn: world.meta.turn,
                province_count: world.provinces.len(),
                faction_count: world.factions.len(),
                character_count: world.characters.len(),
            };

            // Persist to storage.
            storage::save_world(&world);

            // Emit an initialization event.
            let event_data = serde_json::to_vec(&resp).unwrap_or_default();
            host::emit_event("crown_ash:world_init", &event_data);
            host::log_info(&format!(
                "crown-ash: world initialized at block {} with {} provinces, {} factions",
                genesis_block, resp.province_count, resp.faction_count,
            ));

            resp
        }
        Err(e) => {
            host::log_error(&format!("crown-ash on_init: bad input: {e}"));
            InitResponse {
                success: false,
                turn: 0,
                province_count: 0,
                faction_count: 0,
                character_count: 0,
            }
        }
    };

    let out = serde_json::to_vec(&response).unwrap_or_default();
    write_output(&out, out_len_ptr)
}

/// **on_tick** -- Advance the game by one turn.
///
/// Input: JSON-encoded `TickRequest` (contains the block hash for RNG).
/// Output: JSON-encoded `TickResponse` with the `TurnSummary`.
#[no_mangle]
pub extern "C" fn on_tick(input_ptr: u32, input_len: u32, out_len_ptr: u32) -> u32 {
    let input_bytes = read_input(input_ptr, input_len);

    let response = match serde_json::from_slice::<TickRequest>(&input_bytes) {
        Ok(req) => {
            match storage::load_world() {
                Some(mut world) => {
                    let block_hash = hex_to_32(&req.block_hash_hex);
                    let summary = crown_ash_sim::tick(&mut world, &block_hash);

                    // Persist only modified entities (delta write).
                    storage::save_world_delta(&world);

                    // Emit a turn event with the summary.
                    let event_data = serde_json::to_vec(&summary).unwrap_or_default();
                    host::emit_event("crown_ash:turn", &event_data);

                    TickResponse {
                        success: true,
                        summary: Some(summary),
                        error: None,
                    }
                }
                None => {
                    host::log_error("crown-ash on_tick: world not initialized");
                    TickResponse {
                        success: false,
                        summary: None,
                        error: Some("World not initialized. Call on_init first.".into()),
                    }
                }
            }
        }
        Err(e) => {
            host::log_error(&format!("crown-ash on_tick: bad input: {e}"));
            TickResponse {
                success: false,
                summary: None,
                error: Some(format!("Invalid input: {e}")),
            }
        }
    };

    let out = serde_json::to_vec(&response).unwrap_or_default();
    write_output(&out, out_len_ptr)
}

/// **process_action** -- Execute a player command.
///
/// Input: JSON-encoded `ActionRequest`.
/// Output: JSON-encoded `ActionResponse`.
#[no_mangle]
pub extern "C" fn process_action(input_ptr: u32, input_len: u32, out_len_ptr: u32) -> u32 {
    let input_bytes = read_input(input_ptr, input_len);

    let response = match serde_json::from_slice::<ActionRequest>(&input_bytes) {
        Ok(req) => {
            match storage::load_world() {
                Some(mut world) => {
                    // Clear dirty tracker before action processing.
                    world.clear_dirty();
                    let events = crown_ash_sim::process_action(&mut world, &req.action);

                    // Persist only modified entities (delta write).
                    storage::save_world_delta(&world);

                    // Emit action events.
                    if !events.is_empty() {
                        let event_data = serde_json::to_vec(&events).unwrap_or_default();
                        host::emit_event("crown_ash:action_events", &event_data);
                    }

                    host::log_info(&format!(
                        "crown-ash: processed action from wallet={} events={}",
                        req.action.wallet,
                        events.len(),
                    ));

                    ActionResponse {
                        success: true,
                        events,
                        error: None,
                    }
                }
                None => {
                    host::log_error("crown-ash process_action: world not initialized");
                    ActionResponse {
                        success: false,
                        events: Vec::new(),
                        error: Some("World not initialized. Call on_init first.".into()),
                    }
                }
            }
        }
        Err(e) => {
            host::log_error(&format!("crown-ash process_action: bad input: {e}"));
            ActionResponse {
                success: false,
                events: Vec::new(),
                error: Some(format!("Invalid input: {e}")),
            }
        }
    };

    let out = serde_json::to_vec(&response).unwrap_or_default();
    write_output(&out, out_len_ptr)
}

/// **query_state** -- Read-only query against the game world.
///
/// Input: JSON-encoded `QueryRequest`.
/// Output: JSON-encoded `QueryResponse`.
#[no_mangle]
pub extern "C" fn query_state(input_ptr: u32, input_len: u32, out_len_ptr: u32) -> u32 {
    let input_bytes = read_input(input_ptr, input_len);

    let response = match serde_json::from_slice::<QueryRequest>(&input_bytes) {
        Ok(req) => {
            match storage::load_world() {
                Some(world) => {
                    let result_bytes = crown_ash_sim::query_world(&world, &req.query);
                    let data: Option<serde_json::Value> =
                        serde_json::from_slice(&result_bytes).ok();

                    QueryResponse {
                        success: true,
                        data,
                        error: None,
                    }
                }
                None => {
                    host::log_error("crown-ash query_state: world not initialized");
                    QueryResponse {
                        success: false,
                        data: None,
                        error: Some("World not initialized. Call on_init first.".into()),
                    }
                }
            }
        }
        Err(e) => {
            host::log_error(&format!("crown-ash query_state: bad input: {e}"));
            QueryResponse {
                success: false,
                data: None,
                error: Some(format!("Invalid input: {e}")),
            }
        }
    };

    let out = serde_json::to_vec(&response).unwrap_or_default();
    write_output(&out, out_len_ptr)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_seed_hex() -> String {
        "07260745086409a80be70c060d2e0e550f7c10a311ca12f1141815311658177f".to_string()
    }

    #[test]
    fn on_init_creates_world() {
        host::reset();
        host::set_block_height(500);

        let req = InitRequest {
            config: WorldConfig::default(),
            seed_hex: default_seed_hex(),
        };
        let input = serde_json::to_vec(&req).unwrap();
        test_set_input(&input);

        on_init(0, 0, 0);

        let out = test_get_output();
        let resp: InitResponse = serde_json::from_slice(&out).unwrap();
        assert!(resp.success);
        assert_eq!(resp.province_count, 25);
        assert_eq!(resp.faction_count, 7);
        assert!(resp.character_count > 0);

        // Verify world was persisted.
        let world = storage::load_world().expect("world should be loadable after on_init");
        assert_eq!(world.meta.genesis_block, 500);
    }

    #[test]
    fn on_tick_advances_turn() {
        host::reset();
        host::set_block_height(100);

        // Initialize first.
        let init_req = InitRequest {
            config: WorldConfig::default(),
            seed_hex: default_seed_hex(),
        };
        test_set_input(&serde_json::to_vec(&init_req).unwrap());
        on_init(0, 0, 0);

        // Now tick.
        let tick_req = TickRequest {
            block_hash_hex: "aabbccdd".to_string().repeat(8),
        };
        test_set_input(&serde_json::to_vec(&tick_req).unwrap());
        on_tick(0, 0, 0);

        let out = test_get_output();
        let resp: TickResponse = serde_json::from_slice(&out).unwrap();
        assert!(resp.success);
        let summary = resp.summary.unwrap();
        assert_eq!(summary.turn, 1);
    }

    #[test]
    fn on_tick_without_init_fails() {
        host::reset();

        let tick_req = TickRequest {
            block_hash_hex: "00".repeat(32),
        };
        test_set_input(&serde_json::to_vec(&tick_req).unwrap());
        on_tick(0, 0, 0);

        let out = test_get_output();
        let resp: TickResponse = serde_json::from_slice(&out).unwrap();
        assert!(!resp.success);
        assert!(resp.error.is_some());
    }

    #[test]
    fn process_action_declare_war() {
        host::reset();
        host::set_block_height(200);

        // Initialize world.
        let init_req = InitRequest {
            config: WorldConfig::default(),
            seed_hex: default_seed_hex(),
        };
        test_set_input(&serde_json::to_vec(&init_req).unwrap());
        on_init(0, 0, 0);

        // Assign a wallet to faction 0's realm so the action is accepted.
        let mut world = storage::load_world().unwrap();
        world.realms[0].owner_wallet = "wallet_abc".to_string();
        storage::save_world(&world);

        // Declare war.
        let action_req = ActionRequest {
            action: QueuedAction {
                wallet: "wallet_abc".to_string(),
                action: GameAction::DeclareWar {
                    target: 1,
                    casus_belli: CasusBelli::Conquest,
                },
                submitted_turn: 0,
            },
        };
        test_set_input(&serde_json::to_vec(&action_req).unwrap());
        process_action(0, 0, 0);

        let out = test_get_output();
        let resp: ActionResponse = serde_json::from_slice(&out).unwrap();
        assert!(resp.success);
        assert_eq!(resp.events.len(), 1);
        // Verify war is recorded.
        let world = storage::load_world().unwrap();
        assert!(world.realms[0].at_war_with.contains(&1));
    }

    #[test]
    fn query_state_meta() {
        host::reset();
        host::set_block_height(300);

        // Initialize.
        let init_req = InitRequest {
            config: WorldConfig::default(),
            seed_hex: default_seed_hex(),
        };
        test_set_input(&serde_json::to_vec(&init_req).unwrap());
        on_init(0, 0, 0);

        // Query meta.
        let query_req = QueryRequest {
            query: crown_ash_sim::WorldQuery::Meta,
        };
        test_set_input(&serde_json::to_vec(&query_req).unwrap());
        query_state(0, 0, 0);

        let out = test_get_output();
        let resp: QueryResponse = serde_json::from_slice(&out).unwrap();
        assert!(resp.success);
        let data = resp.data.unwrap();
        assert_eq!(data["genesis_block"], 300);
        assert_eq!(data["initialized"], true);
    }

    #[test]
    fn query_state_all_provinces() {
        host::reset();
        host::set_block_height(400);

        let init_req = InitRequest {
            config: WorldConfig {
                province_count: 10,
                faction_count: 3,
                ..WorldConfig::default()
            },
            seed_hex: default_seed_hex(),
        };
        test_set_input(&serde_json::to_vec(&init_req).unwrap());
        on_init(0, 0, 0);

        let query_req = QueryRequest {
            query: crown_ash_sim::WorldQuery::AllProvinces,
        };
        test_set_input(&serde_json::to_vec(&query_req).unwrap());
        query_state(0, 0, 0);

        let out = test_get_output();
        let resp: QueryResponse = serde_json::from_slice(&out).unwrap();
        assert!(resp.success);
        let arr = resp.data.unwrap();
        assert_eq!(arr.as_array().unwrap().len(), 10);
    }

    #[test]
    fn hex_to_32_round_trip() {
        let input = "0102030405060708091011121314151617181920212223242526272829303132";
        let bytes = hex_to_32(input);
        assert_eq!(bytes[0], 0x01);
        assert_eq!(bytes[31], 0x32);
    }

    #[test]
    fn hex_to_32_short_input_returns_zeroes() {
        let bytes = hex_to_32("abcd");
        assert_eq!(bytes, [0u8; 32]);
    }

    #[test]
    fn alloc_returns_aligned_offsets() {
        // Reset bump offset for this test.
        unsafe { BUMP_OFFSET = 0; }
        let a = alloc(10);
        let b = alloc(3);
        // Both should be 8-byte aligned.
        assert_eq!(a % 8, 0);
        assert_eq!(b % 8, 0);
        // b should be after a + 10 bytes.
        assert!(b >= a + 10);
    }
}
