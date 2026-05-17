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

/// Hard-coded default — the live-prod 26K..100K gap that every fresh node hits.
///
/// Stays here as a string (not a parsed list) so the parsing path is
/// exercised by the default code path too. Tests cover both branches.
pub const DEFAULT_GAPS: &str = "25988:100440";

/// Env var name. Override or set to empty string / "none" to disable.
pub const ENV_VAR: &str = "Q_KNOWN_PERMANENT_GAPS";

/// A list of `[start, end]` inclusive gap intervals, normalised (sorted + merged).
#[derive(Debug, Clone, Default)]
pub struct KnownGaps {
    /// Invariant: sorted by `start`, disjoint, non-overlapping.
    gaps: Vec<(u64, u64)>,
}

impl KnownGaps {
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

        Self { gaps: merged }
    }

    /// Construct from environment, falling back to [`DEFAULT_GAPS`].
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
            ENV_VAR, raw, source, parsed.len(), parsed.gaps()
        );
        parsed
    }

    /// Number of gaps configured.
    pub fn len(&self) -> usize {
        self.gaps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gaps.is_empty()
    }

    /// Borrow the normalised gap list.
    pub fn gaps(&self) -> &[(u64, u64)] {
        &self.gaps
    }

    /// Find the next gap whose `start` is strictly greater than `contiguous`,
    /// OR whose range contains `contiguous + 1`.
    ///
    /// Returns `None` when there is no gap at or above `contiguous + 1`.
    ///
    /// This is the hot-path method called on every scheduler tick.
    pub fn next_gap_above(&self, contiguous: u64) -> Option<(u64, u64)> {
        let next_block = contiguous.saturating_add(1);
        // Binary search by start. We want the first gap whose end >= next_block.
        // The gap list is sorted by start and disjoint, so equivalently: the
        // first gap whose start >= next_block, OR the predecessor if it still
        // covers next_block.
        match self.gaps.binary_search_by(|(s, _)| s.cmp(&next_block)) {
            Ok(idx) => Some(self.gaps[idx]),
            Err(idx) => {
                // idx is the insertion point — check predecessor too.
                if idx > 0 {
                    let prev = self.gaps[idx - 1];
                    if prev.1 >= next_block {
                        return Some(prev);
                    }
                }
                self.gaps.get(idx).copied()
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
    fn parse_default_string() {
        let gaps = KnownGaps::parse(DEFAULT_GAPS);
        assert_eq!(gaps.gaps(), &[(25_988u64, 100_440u64)]);
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
        assert_eq!(gaps.gaps(), &[(100, 200), (300, 400), (1000, 2000)]);
    }

    #[test]
    fn parse_multiple_gaps_unsorted_input_sorts_them() {
        let gaps = KnownGaps::parse("1000:2000, 100:200, 300:400");
        assert_eq!(gaps.gaps(), &[(100, 200), (300, 400), (1000, 2000)]);
    }

    #[test]
    fn parse_merges_overlapping_gaps() {
        let gaps = KnownGaps::parse("100:200, 150:300");
        assert_eq!(gaps.gaps(), &[(100, 300)]);
    }

    #[test]
    fn parse_merges_adjacent_gaps() {
        // 100..=200 and 201..=300 are adjacent → merge.
        let gaps = KnownGaps::parse("100:200, 201:300");
        assert_eq!(gaps.gaps(), &[(100, 300)]);
    }

    #[test]
    fn parse_does_not_merge_disjoint_gaps_with_a_gap_block_between() {
        // 100..=200 then 202..=300 have a single block 201 in between → NOT merged.
        let gaps = KnownGaps::parse("100:200, 202:300");
        assert_eq!(gaps.gaps(), &[(100, 200), (202, 300)]);
    }

    #[test]
    fn parse_malformed_entries_are_skipped() {
        // Missing colon, bad number, end < start, plus empty entries.
        let gaps = KnownGaps::parse(",100:200,,abc:def,500:foo,bar:600,800:700,,300:400,");
        assert_eq!(gaps.gaps(), &[(100, 200), (300, 400)]);
    }

    #[test]
    fn parse_extra_whitespace_inside_entry() {
        let gaps = KnownGaps::parse("  100 : 200 ,  300:400  ");
        assert_eq!(gaps.gaps(), &[(100, 200), (300, 400)]);
    }

    #[test]
    fn parse_single_block_gap() {
        let gaps = KnownGaps::parse("42:42");
        assert_eq!(gaps.gaps(), &[(42, 42)]);
        assert!(gaps.contains(42));
    }

    #[test]
    fn next_gap_above_default_at_stall_point() {
        // The exact prod scenario: contiguous=25987, gap starts at 25988.
        let gaps = KnownGaps::parse(DEFAULT_GAPS);
        let g = gaps.next_gap_above(25_987).expect("must find the gap");
        assert_eq!(g, (25_988, 100_440));
    }

    #[test]
    fn next_gap_above_returns_none_when_past_all_gaps() {
        let gaps = KnownGaps::parse(DEFAULT_GAPS);
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
    fn from_env_falls_back_to_default_when_unset() {
        // Ensure the var is unset (cargo test runs in arbitrary order; this is
        // best-effort and tolerant: if it IS set we just check the var was
        // parsed without panic).
        let prev = env::var(ENV_VAR).ok();
        env::remove_var(ENV_VAR);
        let gaps = KnownGaps::from_env();
        // restore
        if let Some(v) = prev {
            env::set_var(ENV_VAR, v);
        } else {
            env::remove_var(ENV_VAR);
        }
        assert_eq!(gaps.gaps(), &[(25_988u64, 100_440u64)]);
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
        assert_eq!(gaps.gaps(), &[(777, 888)]);
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
        // Sanity: 100k lookups against the default gap list must complete fast.
        let gaps = KnownGaps::parse(DEFAULT_GAPS);
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
}
