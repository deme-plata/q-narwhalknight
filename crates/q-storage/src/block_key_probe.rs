//! v10.11.111: the three on-disk block key formats, as presence probes.
//!
//! A height holds a block if CF_BLOCKS has ANY of:
//!   1. `qblock:height:{N}`            — exact key (current format)
//!   2. `qblock:dag:{N}:{proposer}`    — prefix `qblock:dag:{N}:` (early DAG layer)
//!   3. `N.to_be_bytes() ++ hash`      — prefix = the 8 big-endian height bytes (old finalize_block)
//!
//! Formats 1 and 2 are decimal strings, so RocksDB orders them as text ("100441" < "26001"): the
//! next block above a height cannot be found by walking those keys. Probing each candidate height
//! in all three formats is exact, which is what a permanent-gap declaration needs.

/// Probes for `candidates`, grouped by format (all format-1 probes, then 2, then 3) so a single
/// iterator's consecutive seeks stay inside one key region. `(bytes, is_prefix)`.
pub fn block_key_probes(candidates: &[u64]) -> Vec<(Vec<u8>, bool)> {
    let mut probes = Vec::with_capacity(candidates.len() * 3);
    probes.extend(candidates.iter().map(|h| (format!("qblock:height:{}", h).into_bytes(), false)));
    probes.extend(candidates.iter().map(|h| (format!("qblock:dag:{}:", h).into_bytes(), true)));
    probes.extend(candidates.iter().map(|h| (h.to_be_bytes().to_vec(), true)));
    probes
}

/// Folds `probe_keys` results for [`block_key_probes`] back into the heights that hold a block in
/// any format, in input order.
pub fn heights_from_probe_hits(candidates: &[u64], hit: &[bool]) -> anyhow::Result<Vec<u64>> {
    let n = candidates.len();
    if hit.len() != 3 * n {
        anyhow::bail!("probe_keys returned {} results for {} probes", hit.len(), 3 * n);
    }
    Ok(candidates
        .iter()
        .enumerate()
        .filter(|(i, _)| hit[*i] || hit[n + *i] || hit[2 * n + *i])
        .map(|(_, &h)| h)
        .collect())
}

// Tests: crates/q-storage/tests/block_key_probe_tests.rs (real RocksDB; the lib unit-test target
// does not compile at 1e36503e4).
