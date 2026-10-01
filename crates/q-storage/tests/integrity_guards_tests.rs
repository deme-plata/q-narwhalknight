//! v10.11.109 integrity guards — run as an INTEGRATION test target on purpose: the q-storage
//! lib unit-test target does not compile at 1e36503e4 (68 pre-existing errors in older test
//! modules), so tests placed inside the lib would never execute.
//!
//!   cargo test -p q-storage --test integrity_guards_tests

use q_storage::archive_guard::{
    base_is_self, cached_base_is_self, fallback_enabled, host_of, ip_is_local, note_miss, recent_miss,
    recent_miss_at, MISS_TTL,
};
use q_storage::balance_root::balance_root_v1;
use q_storage::log_ratelimit::RateLimiter;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------- archive_guard

#[test]
fn host_of_handles_the_url_shapes_we_use() {
    assert_eq!(host_of("http://89.149.241.126:8080"), "89.149.241.126");
    assert_eq!(host_of("https://quillon.xyz"), "quillon.xyz");
    assert_eq!(host_of("https://quillon.xyz/api/v1"), "quillon.xyz");
    assert_eq!(host_of("http://[::1]:8080/x"), "::1");
    assert_eq!(host_of("localhost:9000"), "localhost");
}

#[test]
fn loopback_and_localhost_are_self() {
    assert!(base_is_self("http://127.0.0.1:8080"));
    assert!(base_is_self("http://localhost:8080"));
    assert!(base_is_self("http://[::1]:8080"));
}

#[test]
fn an_address_not_assigned_to_this_machine_is_not_self() {
    // 203.0.113.0/24 is TEST-NET-3 (RFC 5737): never assigned to a real interface.
    assert!(!ip_is_local("203.0.113.77".parse().unwrap()));
    assert!(!base_is_self("http://203.0.113.77:8080"));
}

#[test]
fn unresolvable_host_is_not_treated_as_self() {
    assert!(!base_is_self("http://nonexistent.invalid:8080"));
}

#[test]
fn self_answer_is_cached_after_the_first_lookup() {
    let base = "http://127.0.0.1:65001";
    assert_eq!(cached_base_is_self(base), None);
    assert!(base_is_self(base));
    assert_eq!(cached_base_is_self(base), Some(true));
}

#[test]
fn miss_cache_remembers_then_expires() {
    let h = 4_242_424_242_u64;
    assert!(!recent_miss(h));
    note_miss(h);
    assert!(recent_miss(h));
    let later = Instant::now() + MISS_TTL + Duration::from_secs(1);
    assert!(!recent_miss_at(h, later));
}

#[test]
fn fallback_switch_defaults_on() {
    if std::env::var("Q_ARCHIVE_FALLBACK").is_err() {
        assert!(fallback_enabled());
    }
}

// ---------------------------------------------------------------- log_ratelimit

#[test]
fn first_line_logs_then_quiet_until_the_interval_passes() {
    let r = RateLimiter::new(30);
    assert_eq!(r.check_at(1_000), Some(0));
    assert_eq!(r.check_at(1_001), None);
    assert_eq!(r.check_at(1_010), None);
    assert_eq!(r.check_at(1_029), None);
    assert_eq!(r.check_at(1_030), Some(3)); // reports the three swallowed lines
    assert_eq!(r.check_at(1_031), None);
}

#[test]
fn twenty_lines_a_second_collapse_to_one_per_interval() {
    let r = RateLimiter::new(30);
    let mut emitted = 0;
    for tick in 0..(20 * 90) {
        if r.check_at(10_000 + tick / 20).is_some() {
            emitted += 1;
        }
    }
    assert_eq!(emitted, 3, "1800 attempts over 90 s at a 30 s interval -> 3 lines");
}

// ---------------------------------------------------------------- balance_root

/// Straight-line re-statement of the BAL-001 spec, independent of the implementation.
fn spec_root(mut v: Vec<([u8; 32], u128)>) -> [u8; 32] {
    v.retain(|(_, a)| *a > 0);
    if v.is_empty() {
        return [0u8; 32];
    }
    v.sort_by(|x, y| x.0.cmp(&y.0));
    let mut root = blake3::Hasher::new();
    root.update(b"balance_root_v1");
    for (addr, amount) in v {
        let mut leaf = blake3::Hasher::new();
        leaf.update(&addr);
        leaf.update(&amount.to_be_bytes());
        root.update(leaf.finalize().as_bytes());
    }
    *root.finalize().as_bytes()
}

fn addr(b: u8) -> [u8; 32] {
    [b; 32]
}

#[test]
fn empty_and_all_zero_sets_give_the_zero_root() {
    assert_eq!(balance_root_v1(Vec::new()), [0u8; 32]);
    assert_eq!(balance_root_v1(vec![(addr(1), 0), (addr(2), 0)]), [0u8; 32]);
}

#[test]
fn root_matches_the_spec_and_ignores_order_and_zero_entries() {
    let a = vec![(addr(9), 5u128), (addr(1), 7), (addr(4), 0), (addr(3), u128::MAX)];
    let b = vec![(addr(3), u128::MAX), (addr(4), 0), (addr(1), 7), (addr(9), 5u128)];
    let want = spec_root(a.clone());
    assert_eq!(balance_root_v1(a), want);
    assert_eq!(balance_root_v1(b), want);
    assert_eq!(balance_root_v1(vec![(addr(9), 5u128), (addr(1), 7), (addr(3), u128::MAX)]), want);
}

#[test]
fn one_unit_moved_between_wallets_changes_the_root() {
    let base = vec![(addr(1), 1_000u128), (addr(2), 2_000)];
    let moved = vec![(addr(1), 999u128), (addr(2), 2_001)]; // same total supply
    assert_ne!(balance_root_v1(base), balance_root_v1(moved));
}

#[test]
fn amount_is_hashed_big_endian() {
    let v = vec![(addr(7), 0x0102u128)];
    let mut leaf = blake3::Hasher::new();
    leaf.update(&addr(7));
    leaf.update(&[0u8; 14]);
    leaf.update(&[0x01, 0x02]);
    let mut root = blake3::Hasher::new();
    root.update(b"balance_root_v1");
    root.update(leaf.finalize().as_bytes());
    assert_eq!(balance_root_v1(v), *root.finalize().as_bytes());
}
