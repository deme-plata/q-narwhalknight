//! Guards for the "Format 4" archive HTTP fallback in `StorageEngine::get_qblock_any_format`.
//!
//! The fallback exists for nodes that warp-synced from the ~16.5 M checkpoint and so hold no
//! early blocks: on a local miss they ask an archive node over HTTP. Three things went wrong
//! when the node being asked was *this* node (Epsilon, whose default archive URL is its own
//! public IP):
//!
//! 1. every missing early height made Epsilon call itself, blocking a thread for the 10 s
//!    timeout (measured 2026-10-01: 404s took 10-18 s, load average 18 -> 59 under 7 parallel
//!    probes) — and the internal block-pack serving loops call the same function per height,
//!    which is why Epsilon took ~25 s to serve a 200-block chunk to a syncing node;
//! 2. nothing remembered a miss, so the same hole was re-asked on every request;
//! 3. nothing bounded how many such fallbacks ran at once, so any anonymous caller of
//!    `GET /api/v1/blocks/<old height>` could pin blocking threads.
//!
//! This module holds the pure, unit-tested pieces of the fix: self-detection, the loop-guard
//! header, a negative cache and the on/off switch.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::{IpAddr, ToSocketAddrs, UdpSocket};
use std::time::{Duration, Instant};

/// Request header the fallback sets on its outgoing call. A node that receives it must answer
/// from LOCAL storage only — never start a fallback of its own — so a loop (or a chain of
/// archive nodes pointing at each other) can never form.
pub const LOOP_GUARD_HEADER: &str = "X-Qnk-Archive-Fallback";

/// How long a "the archive has no such block" answer is remembered.
pub const MISS_TTL: Duration = Duration::from_secs(300);
/// Hard bound on remembered misses (the whole table is dropped when it fills).
pub const MISS_CAP: usize = 100_000;
/// Maximum archive HTTP fetches in flight at once; further callers get a miss immediately.
pub const MAX_INFLIGHT: usize = 8;

/// `Q_ARCHIVE_FALLBACK=0` switches the fallback off entirely (default: on).
pub fn fallback_enabled() -> bool {
    std::env::var("Q_ARCHIVE_FALLBACK").map(|v| v != "0").unwrap_or(true)
}

/// Extract the bare host from an URL-ish base such as `http://1.2.3.4:8080` or
/// `https://quillon.xyz/path` or `http://[::1]:8080`.
pub fn host_of(base: &str) -> String {
    let rest = base.split("://").nth(1).unwrap_or(base);
    let hostport = rest.split('/').next().unwrap_or(rest);
    if let Some(stripped) = hostport.strip_prefix('[') {
        // IPv6 literal: [::1]:8080
        return stripped.split(']').next().unwrap_or(stripped).to_string();
    }
    match hostport.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) => h.to_string(),
        _ => hostport.to_string(),
    }
}

/// True when `ip` is an address of THIS machine. Binding a UDP socket to an address that is
/// not assigned to any local interface fails with EADDRNOTAVAIL, so a successful bind proves
/// the address is ours. Nothing is sent.
pub fn ip_is_local(ip: IpAddr) -> bool {
    ip.is_loopback() || UdpSocket::bind((ip, 0)).is_ok()
}

fn compute_base_is_self(base: &str) -> bool {
    let host = host_of(base);
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return ip_is_local(ip);
    }
    // A hostname (e.g. quillon.xyz): self if ANY address it resolves to is ours.
    match (host.as_str(), 80u16).to_socket_addrs() {
        Ok(addrs) => addrs.into_iter().any(|a| ip_is_local(a.ip())),
        Err(_) => false,
    }
}

static SELF_CACHE: Lazy<Mutex<HashMap<String, bool>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// The cached answer for `base`, if `base_is_self` has already been computed — lets async
/// callers skip the blocking pool entirely on every call after the first.
pub fn cached_base_is_self(base: &str) -> Option<bool> {
    SELF_CACHE.lock().get(base).copied()
}

/// Does `base` point at this very node? Cached per distinct base (DNS is resolved once).
/// May block briefly on the first call for a hostname — call from `spawn_blocking`.
pub fn base_is_self(base: &str) -> bool {
    if let Some(v) = SELF_CACHE.lock().get(base) {
        return *v;
    }
    let v = compute_base_is_self(base);
    SELF_CACHE.lock().insert(base.to_string(), v);
    v
}

static MISSES: Lazy<Mutex<HashMap<u64, Instant>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Was `height` recently reported absent by the archive?
pub fn recent_miss(height: u64) -> bool {
    recent_miss_at(height, Instant::now())
}

pub fn recent_miss_at(height: u64, now: Instant) -> bool {
    match MISSES.lock().get(&height) {
        Some(t) => now.saturating_duration_since(*t) < MISS_TTL,
        None => false,
    }
}

/// Remember that the archive has no block at `height`.
pub fn note_miss(height: u64) {
    let mut m = MISSES.lock();
    if m.len() >= MISS_CAP {
        m.clear();
    }
    m.insert(height, Instant::now());
}

// Tests: crates/q-storage/tests/integrity_guards_tests.rs (the lib unit-test target does not
// compile at 1e36503e4, so in-module tests would never run).
