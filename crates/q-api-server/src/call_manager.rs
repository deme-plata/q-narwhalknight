// Call lifecycle state machine — adapted from nova-chat src/media/signaling.rs.
//
// Manages concurrent call state server-side:
// - Capacity enforcement (MAX_CONCURRENT_CALLS)
// - Per-peer deduplication (one active call per peer)
// - Timeout detection (calls stuck in Initiating → emit CallEnd)
//
// Thread-safe via DashMap — no Mutex, no blocking.

use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;

pub const MAX_CONCURRENT_CALLS: usize = 50;
pub const CALL_INITIATION_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone, PartialEq)]
pub enum CallRecordState {
    Initiating,
    Connected,
    Ended,
}

#[derive(Debug, Clone)]
pub struct CallRecord {
    pub call_id: String,
    pub caller: String,
    pub callee: String,
    pub call_type: String,
    pub state: CallRecordState,
    pub started_at: Instant,
}

#[derive(Clone)]
pub struct CallManager {
    /// call_id → CallRecord
    active_calls: Arc<DashMap<String, CallRecord>>,
    /// peer_id → call_id (one active call per peer)
    peer_calls: Arc<DashMap<String, String>>,
}

impl CallManager {
    pub fn new() -> Self {
        Self {
            active_calls: Arc::new(DashMap::new()),
            peer_calls: Arc::new(DashMap::new()),
        }
    }

    /// Register a new call. Returns Err("capacity") if at limit, Err("busy") if either
    /// peer already has an active call.
    pub fn register_call(
        &self,
        call_id: &str,
        caller: &str,
        callee: &str,
        call_type: &str,
    ) -> Result<(), &'static str> {
        if self.active_calls.len() >= MAX_CONCURRENT_CALLS {
            return Err("capacity");
        }
        if self.peer_calls.contains_key(caller) {
            return Err("busy");
        }
        if !callee.is_empty() && self.peer_calls.contains_key(callee) {
            return Err("busy");
        }

        let record = CallRecord {
            call_id: call_id.to_string(),
            caller: caller.to_string(),
            callee: callee.to_string(),
            call_type: call_type.to_string(),
            state: CallRecordState::Initiating,
            started_at: Instant::now(),
        };

        self.active_calls.insert(call_id.to_string(), record);
        self.peer_calls.insert(caller.to_string(), call_id.to_string());
        if !callee.is_empty() {
            self.peer_calls.insert(callee.to_string(), call_id.to_string());
        }

        Ok(())
    }

    pub fn mark_connected(&self, call_id: &str) {
        if let Some(mut record) = self.active_calls.get_mut(call_id) {
            record.state = CallRecordState::Connected;
        }
    }

    pub fn end_call(&self, call_id: &str) {
        if let Some((_, record)) = self.active_calls.remove(call_id) {
            self.peer_calls.remove(&record.caller);
            if !record.callee.is_empty() {
                self.peer_calls.remove(&record.callee);
            }
        }
    }

    pub fn find_call_for_peer(&self, peer_id: &str) -> Option<String> {
        self.peer_calls.get(peer_id).map(|r| r.value().clone())
    }

    pub fn active_call_count(&self) -> usize {
        self.active_calls.len()
    }

    /// Returns (caller, callee) for a call, or None if not found.
    pub fn get_call_peers(&self, call_id: &str) -> Option<(String, String)> {
        self.active_calls.get(call_id).map(|r| (r.caller.clone(), r.callee.clone()))
    }

    /// Returns calls that have been in Initiating state longer than `timeout_secs`.
    /// Returns Vec<(call_id, caller, callee)> so the caller can send CallEnd to both peers.
    pub fn get_timed_out_calls(&self, timeout_secs: u64) -> Vec<(String, String, String)> {
        self.active_calls
            .iter()
            .filter(|r| {
                r.state == CallRecordState::Initiating
                    && r.started_at.elapsed().as_secs() >= timeout_secs
            })
            .map(|r| (r.call_id.clone(), r.caller.clone(), r.callee.clone()))
            .collect()
    }

    /// Remove a peer from any active call without knowing the call_id.
    /// Used during peer disconnect cleanup.
    pub fn remove_peer(&self, peer_id: &str) {
        if let Some((_, call_id)) = self.peer_calls.remove(peer_id) {
            self.end_call(&call_id);
        }
    }
}

impl Default for CallManager {
    fn default() -> Self {
        Self::new()
    }
}
