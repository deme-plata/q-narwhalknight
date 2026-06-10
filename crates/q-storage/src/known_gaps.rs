//! v10.9.44 — Definitive permanent-gap skip via env var `Q_KNOWN_PERMANENT_GAPS`.
//!
//! # Why this exists
//! Blocks `25,988..=100,440` are physically missing **network-wide** as a result
//! of a historical pruning incident. Every fresh node stalls at
//! `h=25,987` because no peer can serve the next chunk — the v10.9.41 quorum
//! negotiation only fires when ≥2 independent peers actively *declare* the gap,
//! and live-prod testing on 2026-05-17 showed zero such declarations because
//! the server-side `forward-seek` returns an empty response before reaching the
//! declaration code path on most peers.
//!
//! This module provides a **deterministic** override: a single env var the
//! operator sets (or that ships with a sane default), parsed at startup, that
//! tells the chunk scheduler to advance `contiguous_height` past each known
//! gap as soon as `contiguous_height + 1 == gap_start`.
//!
//! # Format
//! `Q_KNOWN_PERMANENT_GAPS = "start1:end1,start2:end2,..."`
//!
//! - Both `start` and `end` are inclusive heights.
//! - Whitespace around entries / separators is tolerated.
//! - Empty entries are skipped.
//! - Malformed entries are **logged and dropped** — never panic on operator input.
//! - When the env var is unset, the [`DEFAULT_GAPS`] value (the known
//!   25,988..=100,440 prod gap) is used.
//! - To opt out entirely, set the var to an empty string or `none`.
//!
//! # Hot-path contract
//! - [`KnownGaps::next_gap_above`] is `O(log n)` (binary search on a
//!   sorted-merged interval list, n = number of gaps; in practice n = 1).
//! - Construction is one-shot at startup. The returned `KnownGaps` is `Send +
//!   Sync` and intended to be wrapped in an `Arc` and read-shared.

use std::env;
use tracing::{info, warn};

/// Default empty — v10.9.47 made permanent-gap detection AUTONOMOUS via the
/// runtime broadcast path (peers declare gaps, client validates via Beta-trust
/// + Markov state, persists to RocksDB). Operators no longer need to set this.
/// The env var stays as an ops-override escape hatch but should never be
/// surfaced to end users.
pub const DEFAULT_GAPS: &str = "";

/// Env var name. Operator override only — leave unset for autonomous heal.
pub const ENV_VAR: &str = "Q_KNOWN_PERMANENT_GAPS";

/// A list of `[start, end]` inclusive gap intervals, normalised (sorted + merged).
///
/// v10.9.47: uses interior mutability so the autonomous detection task can
/// add new gaps at runtime without requiring every reader to take a write
/// lock. The hot path `next_gap_above` takes a `parking_lot::RwLock` read,
/// which is microseconds.
#[derive(Debug, Default)]
pub struct KnownGaps {
    /// Invariant: sorted by `start`, disjoint, non-overlapping.
    gaps: parking_lot::RwLock<Vec<(u64, u64)>>,
}

impl KnownGaps {
    /// Build from a pre-validated `(start, end)` list. Used to seed the
    /// runtime list from RocksDB-persisted gaps at startup. Sorts + merges
    /// before storing so the invariants hold.
    pub fn from_pairs(pairs: Vec<(u64, u64)>) -> Self {
        let mut sane: Vec<(u64, u64)> = pairs
            .into_iter()
            .filter(|(s, e)| s <= e)
            .collect();
        sane.sort_by_key(|(s, _)| *s);
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(sane.len());
        for (s, e) in sane {
            match merged.last_mut() {
                Some(last) if s <= last.1.saturating_add(1) => {
                    if e > last.1 {
                        last.1 = e;
                    }
                }
                _ => merged.push((s, e)),
            }
        }
        Self {
            gaps: parking_lot::RwLock::new(merged),
        }
    }

    /// v10.9.47: add a runtime-detected gap (called by the autonomous heal
    /// task). Idempotent — adding an identical or overlapping gap merges.
    /// Returns true if this changed the gap list (caller persists in that case).
    pub fn add_gap(&self, start: u64, end: u64) -> bool {
        if start > end {
            return false;
        }
        let mut guard = self.gaps.write();
        // Reject if already covered.
        for (s, e) in guard.iter() {
            if *s <= start && end <= *e {
                return false;
            }
        }
        guard.push((start, end));
        guard.sort_by_key(|(s, _)| *s);
        // Re-merge in place.
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(guard.len());
        for (s, e) in guard.drain(..) {
            match merged.last_mut() {
                Some(last) if s <= last.1.saturating_add(1) => {
                    if e > last.1 {
                        last.1 = e;
                    }
                }
                _ => merged.push((s, e)),
            }
        }
        *guard = merged;
        true
    }

    /// Build from a raw env-value-shaped string. Tolerates whitespace, drops
    /// malformed entries with a `warn!`. Returns an empty list for empty
    /// input or the sentinel `"none"`.
    pub fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
            return Self::default();
        }

        let mut gaps: Vec<(u64, u64)> = Vec::new();
        for (idx, entry) in trimmed.split(',').enumerate() {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let Some((start_s, end_s)) = entry.split_once(':') else {
                warn!(
                    "[known_gaps] entry #{} '{}': missing ':' separator (expected start:end) — skipped",
                    idx, entry
                );
                continue;
            };
            let start = match start_s.trim().parse::<u64>() {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        "[known_gaps] entry #{} '{}': invalid start '{}' ({}) — skipped",
                        idx, entry, start_s, e
                    );
                    continue;
                }
            };
            let end = match end_s.trim().parse::<u64>() {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        "[known_gaps] entry #{} '{}': invalid end '{}' ({}) — skipped",
                        idx, entry, end_s, e
                    );
                    continue;
                }
            };
            if start > end {
                warn!(
                    "[known_gaps] entry #{} '{}': start ({}) > end ({}) — skipped",
                    idx, entry, start, end
                );
                continue;
            }
            gaps.push((start, end));
        }

        // Sort by start, merge overlaps / adjacents.
        gaps.sort_by_key(|(s, _)| *s);
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(gaps.len());
        for (s, e) in gaps {
            match merged.last_mut() {
                Some(last) if s <= last.1.saturating_add(1) => {
                    if e > last.1 {
                        last.1 = e;
                    }
                }
                _ => merged.push((s, e)),
            }
        }

        Self {
            gaps: parking_lot::RwLock::new(merged),
        }
    }

    /// Construct from environment, falling back to [`DEFAULT_GAPS`] (empty in v10.9.47+).
    ///
    /// Logs the effective gap list at INFO level so operators can verify at
    /// startup what gap-skip behaviour is active.
    pub fn from_env() -> Self {
        let (source, raw) = match env::var(ENV_VAR) {
            Ok(v) => ("env", v),
            Err(_) => ("default", DEFAULT_GAPS.to_string()),
        };
        let parsed = Self::parse(&raw);
        info!(
            "[CONFIG] {} = '{}' (source={}, parsed={} gap(s): {:?})",
            ENV_VAR, raw, source, parsed.len(), parsed.snapshot()
        );
        parsed
    }

    /// Number of gaps configured.
    pub fn len(&self) -> usize {
        self.gaps.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.gaps.read().is_empty()
    }

    /// Snapshot of the gap list. Allocates — use sparingly.
    pub fn snapshot(&self) -> Vec<(u64, u64)> {
        self.gaps.read().clone()
    }

    /// Find the next gap whose `start` is strictly greater than `contiguous`,
    /// OR whose range contains `contiguous + 1`.
    ///
    /// Returns `None` when there is no gap at or above `contiguous + 1`.
    ///
    /// This is the hot-path method called on every scheduler tick.
    pub fn next_gap_above(&self, contiguous: u64) -> Option<(u64, u64)> {
        let next_block = contiguous.saturating_add(1);
        let guard = self.gaps.read();
        // Binary search by start. We want the first gap whose end >= next_block.
        // The gap list is sorted by start and disjoint, so equivalently: the
        // first gap whose start >= next_block, OR the predecessor if it still
        // covers next_block.
        match guard.binary_search_by(|(s, _)| s.cmp(&next_block)) {
            Ok(idx) => Some(guard[idx]),
            Err(idx) => {
                // idx is the insertion point — check predecessor too.
                if idx > 0 {
                    let prev = guard[idx - 1];
                    if prev.1 >= next_block {
                        return Some(prev);
                    }
                }
                guard.get(idx).copied()
            }
        }
    }

    /// Convenience: does the given height fall inside any known gap?
    pub fn contains(&self, height: u64) -> bool {
        self.next_gap_above(height.saturating_sub(1))
            .map(|(s, e)| s <= height && height <= e)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_default_string_is_empty_v10_9_47() {
        // v10.9.47: DEFAULT_GAPS is now empty — autonomous detection handles
        // permanent gaps at runtime via the broadcast path.
        let gaps = KnownGaps::parse(DEFAULT_GAPS);
        assert!(gaps.is_empty());
    }

    #[test]
    fn parse_empty_is_empty() {
        let gaps = KnownGaps::parse("");
        assert!(gaps.is_empty());
    }

    #[test]
    fn parse_whitespace_is_empty() {
        let gaps = KnownGaps::parse("   \t\n  ");
        assert!(gaps.is_empty());
    }

    #[test]
    fn parse_none_sentinel_is_empty() {
        assert!(KnownGaps::parse("none").is_empty());
        assert!(KnownGaps::parse("NONE").is_empty());
        assert!(KnownGaps::parse("None").is_empty());
    }

    #[test]
    fn parse_multiple_gaps() {
        let gaps = KnownGaps::parse("100:200, 300:400, 1000:2000");
        assert_eq!(gaps.snapshot().as_slice(), &[(100, 200), (300, 400), (1000, 2000)]);
    }

    #[test]
    fn parse_multiple_gaps_unsorted_input_sorts_them() {
        let gaps = KnownGaps::parse("1000:2000, 100:200, 300:400");
        assert_eq!(gaps.snapshot().as_slice(), &[(100, 200), (300, 400), (1000, 2000)]);
    }

    #[test]
    fn parse_merges_overlapping_gaps() {
        let gaps = KnownGaps::parse("100:200, 150:300");
        assert_eq!(gaps.snapshot().as_slice(), &[(100, 300)]);
    }

    #[test]
    fn parse_merges_adjacent_gaps() {
        // 100..=200 and 201..=300 are adjacent → merge.
        let gaps = KnownGaps::parse("100:200, 201:300");
        assert_eq!(gaps.snapshot().as_slice(), &[(100, 300)]);
    }

    #[test]
    fn parse_does_not_merge_disjoint_gaps_with_a_gap_block_between() {
        // 100..=200 then 202..=300 have a single block 201 in between → NOT merged.
        let gaps = KnownGaps::parse("100:200, 202:300");
        assert_eq!(gaps.snapshot().as_slice(), &[(100, 200), (202, 300)]);
    }

    #[test]
    fn parse_malformed_entries_are_skipped() {
        // Missing colon, bad number, end < start, plus empty entries.
        let gaps = KnownGaps::parse(",100:200,,abc:def,500:foo,bar:600,800:700,,300:400,");
        assert_eq!(gaps.snapshot().as_slice(), &[(100, 200), (300, 400)]);
    }

    #[test]
    fn parse_extra_whitespace_inside_entry() {
        let gaps = KnownGaps::parse("  100 : 200 ,  300:400  ");
        assert_eq!(gaps.snapshot().as_slice(), &[(100, 200), (300, 400)]);
    }

    #[test]
    fn parse_single_block_gap() {
        let gaps = KnownGaps::parse("42:42");
        assert_eq!(gaps.snapshot().as_slice(), &[(42, 42)]);
        assert!(gaps.contains(42));
    }

    #[test]
    fn next_gap_above_at_stall_point_with_explicit_config() {
        // Simulate the prod scenario via explicit config rather than DEFAULT
        // (which is now empty in v10.9.47): contiguous=25987, gap (25988, 100440).
        let gaps = KnownGaps::parse("25988:100440");
        let g = gaps.next_gap_above(25_987).expect("must find the gap");
        assert_eq!(g, (25_988, 100_440));
    }

    #[test]
    fn next_gap_above_returns_none_when_past_all_gaps() {
        let gaps = KnownGaps::parse("25988:100440");
        // contiguous past the gap end.
        assert!(gaps.next_gap_above(100_441).is_none());
        assert!(gaps.next_gap_above(u64::MAX).is_none());
    }

    #[test]
    fn next_gap_above_returns_current_gap_if_inside() {
        // If contiguous is somehow mid-gap (shouldn't happen but be defensive),
        // next_gap_above still returns the enclosing gap.
        let gaps = KnownGaps::parse("100:200");
        // contiguous=150 → next_block=151 → still inside gap → return (100,200)
        assert_eq!(gaps.next_gap_above(150), Some((100, 200)));
    }

    #[test]
    fn next_gap_above_returns_next_with_multiple_gaps() {
        let gaps = KnownGaps::parse("100:200, 500:600, 1000:2000");
        // Below first.
        assert_eq!(gaps.next_gap_above(50), Some((100, 200)));
        // After first, before second.
        assert_eq!(gaps.next_gap_above(300), Some((500, 600)));
        // After second, before third.
        assert_eq!(gaps.next_gap_above(700), Some((1000, 2000)));
        // After third.
        assert_eq!(gaps.next_gap_above(2000), None);
    }

    #[test]
    fn next_gap_above_zero_contiguous_finds_low_gap() {
        let gaps = KnownGaps::parse("1:5");
        assert_eq!(gaps.next_gap_above(0), Some((1, 5)));
    }

    #[test]
    fn contains_check() {
        let gaps = KnownGaps::parse("100:200");
        assert!(!gaps.contains(99));
        assert!(gaps.contains(100));
        assert!(gaps.contains(150));
        assert!(gaps.contains(200));
        assert!(!gaps.contains(201));
    }

    #[test]
    fn from_env_falls_back_to_empty_default_when_unset() {
        // v10.9.47: default is empty — runtime detection handles gaps.
        let prev = env::var(ENV_VAR).ok();
        env::remove_var(ENV_VAR);
        let gaps = KnownGaps::from_env();
        if let Some(v) = prev {
            env::set_var(ENV_VAR, v);
        } else {
            env::remove_var(ENV_VAR);
        }
        assert!(gaps.is_empty());
    }

    #[test]
    fn from_env_uses_var_value_when_set() {
        let prev = env::var(ENV_VAR).ok();
        env::set_var(ENV_VAR, "777:888");
        let gaps = KnownGaps::from_env();
        if let Some(v) = prev {
            env::set_var(ENV_VAR, v);
        } else {
            env::remove_var(ENV_VAR);
        }
        assert_eq!(gaps.snapshot().as_slice(), &[(777, 888)]);
    }

    #[test]
    fn from_env_can_disable_via_none_sentinel() {
        let prev = env::var(ENV_VAR).ok();
        env::set_var(ENV_VAR, "none");
        let gaps = KnownGaps::from_env();
        if let Some(v) = prev {
            env::set_var(ENV_VAR, v);
        } else {
            env::remove_var(ENV_VAR);
        }
        assert!(gaps.is_empty());
    }

    #[test]
    fn hot_path_under_10us() {
        // Sanity: 100k lookups against a representative gap list must complete fast.
        // v10.9.47: explicit config since DEFAULT_GAPS is now empty.
        let gaps = KnownGaps::parse("25988:100440");
        let start = std::time::Instant::now();
        let mut hits = 0u64;
        for h in 0..100_000u64 {
            if gaps.next_gap_above(h).is_some() {
                hits += 1;
            }
        }
        let elapsed = start.elapsed();
        // Loose: 100k ops in well under 1 s on any sensible machine.
        // Per-op budget ≪ 10 µs (target was per-call).
        assert!(elapsed.as_secs() < 5, "100k lookups took {:?}", elapsed);
        // We hit a gap for every contiguous <= 100440.
        assert!(hits >= 100_000);
    }

    #[test]
    fn add_gap_inserts_and_merges_v10_9_47() {
        // v10.9.47: runtime add_gap path used by autonomous detection.
        let gaps = KnownGaps::from_pairs(vec![(100, 200)]);
        assert!(gaps.add_gap(500, 600));               // new range
        assert!(gaps.add_gap(201, 400));               // bridges (100,200) and (500,600)? No — 201..400 + 500..600 with 401..499 gap → 3 ranges
        // After: (100, 400) and (500, 600) — merge of (100,200) + (201,400) since 201 == 200+1
        let snap = gaps.snapshot();
        assert_eq!(snap, vec![(100, 400), (500, 600)]);
        // Idempotent: re-adding a contained range returns false.
        assert!(!gaps.add_gap(250, 300));
        assert!(!gaps.add_gap(100, 400));
        // Backing onto (500, 600) with (700, 800) → 3 ranges if 601..699 between
        assert!(gaps.add_gap(700, 800));
        assert_eq!(gaps.snapshot(), vec![(100, 400), (500, 600), (700, 800)]);
    }

    #[test]
    fn from_pairs_normalises_and_merges_v10_9_47() {
        let gaps = KnownGaps::from_pairs(vec![(500, 600), (100, 200), (201, 400)]);
        // Expect (100, 400) and (500, 600) after sort+merge.
        assert_eq!(gaps.snapshot(), vec![(100, 400), (500, 600)]);
    }
}
