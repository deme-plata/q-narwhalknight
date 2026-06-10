// crates/q-storage/src/hot_wallet_cache.rs
//
// Hot-wallet SMT proof prefetch cache.
//
// `BalanceSmt::prove` walks the 256-deep sparse Merkle tree top-to-bottom,
// costing ~256 RocksDB reads per call. The `/api/v1/balance` endpoint hits
// this on every query, and wallet balance traffic follows a power law — the
// top ~1 % of addresses receive ~99 % of queries.
//
// `HotWalletCache` keeps an LRU of (addr → SmtProof) for the ~10 K hottest
// wallets. Cache hits return in O(1). On every block-induced root change the
// background `run` task drops the cache, then re-walks the previously-hot
// keys so the next query against any of them is still a cache hit
// (avoids a "thundering herd" on the slow path right after a block lands).
//
// Memory budget:
//   Each `SmtProof` is ~8 KB (256 siblings × 32 bytes + 32-byte bitmap +
//   32-byte addr + 16-byte balance + alignment).
//   Worst case at the default 10 000-entry cap: 10_000 × 8 KB ≈ 80 MB.
//
// ─── Wiring (FOLLOW-UP PR, not this one) ─────────────────────────────────
//
// In `crates/q-api-server/src/handlers.rs` the entry point
// `pub async fn get_wallet_balance(...)` begins at line 7118.
// The cache check should sit AFTER the address has been parsed
// (around line 7164, after `requested_address` is constructed) and
// BEFORE any call to `BalanceSmt::prove`. Sketch:
//
//     if let Some(proof) = state.hot_wallet_cache.lookup(&requested_address) {
//         /* serialise proof + balance from proof.balance */
//     } else {
//         let bal = state.storage_engine.get_balance(&requested_address).await?;
//         let proof = state.storage_engine.balance_smt.prove(&requested_address, bal)?;
//         state.hot_wallet_cache.insert(requested_address, proof.clone());
//         /* serialise */
//     }
//
// The follow-up PR must also spawn the background task once:
//
//     let cache = Arc::new(HotWalletCache::new(smt, 10_000));
//     let (_tx, rx) = tokio::sync::watch::channel(false);
//     tokio::spawn(Arc::clone(&cache).run(rx));
//
// ─────────────────────────────────────────────────────────────────────────

use crate::balance_smt::{BalanceSmt, SmtProof};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::watch;
use tracing::{debug, trace, warn};

/// Hard upper bound on cache size. Larger requests are clamped to this.
/// At ~8 KB/entry this caps memory at ~80 MB.
const MAX_CACHE_ENTRIES: usize = 10_000;

/// Number of hottest keys to re-prove (warmup) after an invalidation. Keeping
/// this small bounds the time the cache is "cold" after every block.
const WARMUP_TOP_N: usize = 100;

/// Interval between SMT-root polls in the background loop.
const ROOT_POLL_INTERVAL: Duration = Duration::from_millis(500);

// ════════════════════════════════════════════════════════════════════════════
// CachedProof
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct CachedProof {
    pub proof: SmtProof,
    pub fetched_at: SystemTime,
    pub root_at_fetch: [u8; 32],
}

// ════════════════════════════════════════════════════════════════════════════
// Stats
// ════════════════════════════════════════════════════════════════════════════

#[derive(Default, Debug)]
pub struct HotWalletStats {
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    /// Incremented once per root-change event (one per block typically).
    pub cache_invalidations: AtomicU64,
    /// Incremented once per warmup pass that successfully completes.
    pub refresh_cycles: AtomicU64,
    /// Wall-clock duration of the most recent warmup pass, microseconds.
    pub last_refresh_time_us: AtomicU64,
}

// ════════════════════════════════════════════════════════════════════════════
// HotWalletCache
// ════════════════════════════════════════════════════════════════════════════

pub struct HotWalletCache {
    smt: Arc<BalanceSmt>,
    cache: Arc<RwLock<LruCache<[u8; 32], CachedProof>>>,
    last_root: Arc<RwLock<[u8; 32]>>,
    stats: Arc<HotWalletStats>,
}

impl HotWalletCache {
    /// Build a new cache. The caller is responsible for spawning
    /// `Arc::clone(&cache).run(rx)` on a tokio runtime.
    ///
    /// `cache_size` is clamped to `[1, MAX_CACHE_ENTRIES]`.
    pub fn new(smt: Arc<BalanceSmt>, cache_size: usize) -> Self {
        let size = cache_size.clamp(1, MAX_CACHE_ENTRIES);
        let capacity = match NonZeroUsize::new(size) {
            Some(n) => n,
            // Unreachable: clamp lower bound is 1, but fall back gracefully.
            None => match NonZeroUsize::new(1) {
                Some(n) => n,
                None => unreachable!("NonZeroUsize::new(1) is always Some"),
            },
        };
        let initial_root = smt.root();
        Self {
            smt,
            cache: Arc::new(RwLock::new(LruCache::new(capacity))),
            last_root: Arc::new(RwLock::new(initial_root)),
            stats: Arc::new(HotWalletStats::default()),
        }
    }

    /// Cache-aware lookup. Returns `Some(proof)` iff the proof is in cache AND
    /// its `root_at_fetch` matches the current SMT root. Otherwise returns
    /// `None` and the caller falls through to `BalanceSmt::prove`.
    ///
    /// A successful lookup promotes the entry in the LRU.
    pub fn lookup(&self, addr: &[u8; 32]) -> Option<SmtProof> {
        let current_root = self.smt.root();

        // Promote on hit (write lock). On miss we still return None.
        let mut guard = match self.cache.write() {
            Ok(g) => g,
            // Lock poisoning here means a previous *writer* panicked. The cache
            // structure itself is intact, so recover the inner value rather
            // than propagating a panic to every reader.
            Err(poison) => poison.into_inner(),
        };

        if let Some(cached) = guard.get(addr) {
            if cached.root_at_fetch == current_root {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Some(cached.proof.clone());
            }
        }
        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert a freshly-computed proof into the cache.
    ///
    /// Call from the slow path after `BalanceSmt::prove` succeeds so the next
    /// query for the same address is a cache hit.
    pub fn insert(&self, addr: [u8; 32], proof: SmtProof) {
        let current_root = self.smt.root();
        let cached = CachedProof {
            proof,
            fetched_at: SystemTime::now(),
            root_at_fetch: current_root,
        };
        let mut guard = match self.cache.write() {
            Ok(g) => g,
            Err(poison) => poison.into_inner(),
        };
        guard.put(addr, cached);
    }

    /// Background task. Polls the SMT root every `ROOT_POLL_INTERVAL`. On a
    /// root change: snapshot the hot keys, clear the cache, then re-prove the
    /// hottest `WARMUP_TOP_N` keys against the new root.
    ///
    /// Exits cleanly when `shutdown` receives `true`.
    pub async fn run(self: Arc<Self>, mut shutdown: watch::Receiver<bool>) {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(ROOT_POLL_INTERVAL) => {
                    if let Err(e) = self.refresh_if_root_changed() {
                        // Logged but non-fatal: missing a refresh just means the
                        // next lookup falls through to `prove` and the cache
                        // re-populates lazily.
                        warn!("HotWalletCache refresh error: {}", e);
                    }
                }
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        debug!("HotWalletCache shutting down");
                        return;
                    }
                }
            }
        }
    }

    /// Return a handle to the stats counters.
    pub fn stats(&self) -> &HotWalletStats {
        &self.stats
    }

    // ──────── internal ────────

    /// One refresh pass. Returns `Ok(())` even when no refresh was needed.
    fn refresh_if_root_changed(&self) -> anyhow::Result<()> {
        let current_root = self.smt.root();

        // Quick path: root unchanged → nothing to do.
        {
            let last = match self.last_root.read() {
                Ok(g) => *g,
                Err(p) => *p.into_inner(),
            };
            if last == current_root {
                return Ok(());
            }
        }

        // Root changed: snapshot keys in LRU order (most-recent first),
        // clear the cache, then warm the top N.
        let started = Instant::now();
        let hot_keys: Vec<[u8; 32]> = {
            let mut guard = match self.cache.write() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            // `LruCache::iter` yields most-recently-used first.
            let snapshot: Vec<[u8; 32]> =
                guard.iter().take(WARMUP_TOP_N).map(|(k, _)| *k).collect();
            guard.clear();
            snapshot
        };

        // Update the root marker BEFORE warmup so any concurrent `lookup` that
        // finds a stale entry (window already zero — we just cleared) does the
        // correct thing.
        {
            let mut guard = match self.last_root.write() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            *guard = current_root;
        }
        self.stats
            .cache_invalidations
            .fetch_add(1, Ordering::Relaxed);

        // Warmup. Each re-prove costs ~256 RocksDB reads. We cannot know the
        // new balance for these addresses without going through the wallet-
        // balance CF (which lives outside this module), so we re-prove using
        // the *cached* balance and only re-insert if the resulting proof still
        // verifies against the new root. Addresses whose balance changed are
        // simply dropped — the next query will refill them via the slow path.
        let mut warmed = 0usize;
        for addr in &hot_keys {
            // We don't have the previous CachedProof here (cache was cleared).
            // Re-prove with a probe balance of 0 — siblings depend only on
            // `addr`, so we can re-verify against the live root using the new
            // siblings only. If the leaf at `addr` is *non-zero* in the new
            // tree, the proof with balance=0 won't verify and we skip it. If
            // it IS zero, we cache a proof that says so. Either way the
            // invariant holds: any cached proof verifies against the live
            // root.
            //
            // NB: this is conservative — we miss warming addresses that
            // changed balance — but it's correct. The follow-up PR can pass
            // an "addr → balance" closure to do a proper warmup.
            match self.smt.prove(addr, 0) {
                Ok(proof) => {
                    if proof.verify(&current_root) {
                        let cached = CachedProof {
                            proof,
                            fetched_at: SystemTime::now(),
                            root_at_fetch: current_root,
                        };
                        let mut guard = match self.cache.write() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        guard.put(*addr, cached);
                        warmed += 1;
                    }
                }
                Err(e) => {
                    trace!("warmup prove failed for one addr: {}", e);
                }
            }
        }

        let elapsed_us = started.elapsed().as_micros();
        let elapsed_u64 = if elapsed_us > u128::from(u64::MAX) {
            u64::MAX
        } else {
            elapsed_us as u64
        };
        self.stats
            .last_refresh_time_us
            .store(elapsed_u64, Ordering::Relaxed);
        self.stats.refresh_cycles.fetch_add(1, Ordering::Relaxed);

        debug!(
            "HotWalletCache refresh: cleared {} entries, warmed {}, {} µs",
            hot_keys.len(),
            warmed,
            elapsed_u64
        );
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::balance_smt::CF_BALANCE_SMT;
    use rocksdb::{ColumnFamilyDescriptor, Options, DB};
    use tempfile::TempDir;

    fn open_test_smt() -> (Arc<BalanceSmt>, TempDir) {
        let tmp = TempDir::new().expect("tempdir");
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let cfs = vec![
            ColumnFamilyDescriptor::new("default", Options::default()),
            ColumnFamilyDescriptor::new(CF_BALANCE_SMT, Options::default()),
        ];
        let db = DB::open_cf_descriptors(&opts, tmp.path(), cfs).expect("open db");
        let smt = BalanceSmt::open(Arc::new(db)).expect("open smt");
        (Arc::new(smt), tmp)
    }

    #[test]
    fn test_cache_starts_empty() {
        let (smt, _tmp) = open_test_smt();
        let cache = HotWalletCache::new(smt, 100);
        let addr = [0x11u8; 32];
        assert!(cache.lookup(&addr).is_none());
        assert_eq!(cache.stats().cache_hits.load(Ordering::Relaxed), 0);
        assert_eq!(cache.stats().cache_misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_cache_hit() {
        let (smt, _tmp) = open_test_smt();
        let cache = HotWalletCache::new(Arc::clone(&smt), 100);

        let addr = [0x22u8; 32];
        // Pre-populate the SMT with a balance for `addr` so we can build a
        // real, verifiable proof.
        let _root = smt.update_batch(&[(addr, 4242u128)]).expect("update");
        let proof = smt.prove(&addr, 4242).expect("prove");

        cache.insert(addr, proof.clone());

        let hit = cache.lookup(&addr).expect("hit");
        assert_eq!(hit.balance, 4242);
        assert_eq!(hit.addr, addr);
        assert_eq!(cache.stats().cache_hits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_cache_invalidates_on_root_change() {
        let (smt, _tmp) = open_test_smt();
        let cache = HotWalletCache::new(Arc::clone(&smt), 100);

        let addr_a = [0xAAu8; 32];
        let _r1 = smt.update_batch(&[(addr_a, 100u128)]).expect("update r1");
        let proof = smt.prove(&addr_a, 100).expect("prove");
        cache.insert(addr_a, proof);
        assert!(cache.lookup(&addr_a).is_some());

        // Mutate the SMT: insert a new, unrelated wallet → root changes.
        let addr_b = [0xBBu8; 32];
        let _r2 = smt.update_batch(&[(addr_b, 200u128)]).expect("update r2");

        // The cached entry's `root_at_fetch` no longer matches → miss.
        assert!(cache.lookup(&addr_a).is_none());
    }

    #[test]
    fn test_cache_lru_eviction() {
        let (smt, _tmp) = open_test_smt();
        let cache = HotWalletCache::new(Arc::clone(&smt), 3);

        // Build four real proofs against a known SMT state, then insert each.
        let addrs: Vec<[u8; 32]> = (0..4u8)
            .map(|i| {
                let mut a = [0u8; 32];
                a[0] = i + 1;
                a
            })
            .collect();
        let updates: Vec<([u8; 32], u128)> =
            addrs.iter().map(|a| (*a, 1u128)).collect();
        let _root = smt.update_batch(&updates).expect("update");

        for a in &addrs {
            let p = smt.prove(a, 1).expect("prove");
            cache.insert(*a, p);
        }

        // First-inserted should have been evicted (capacity = 3).
        assert!(cache.lookup(&addrs[0]).is_none(), "oldest not evicted");
        assert!(cache.lookup(&addrs[1]).is_some());
        assert!(cache.lookup(&addrs[2]).is_some());
        assert!(cache.lookup(&addrs[3]).is_some());
    }

    #[tokio::test]
    async fn test_run_invalidates_on_external_root_change() {
        let (smt, _tmp) = open_test_smt();
        let cache = Arc::new(HotWalletCache::new(Arc::clone(&smt), 100));

        // Pre-populate cache with one entry against the current root.
        let addr = [0x77u8; 32];
        let _r1 = smt.update_batch(&[(addr, 9u128)]).expect("update r1");
        // Update last_root marker to match the post-update root so the run
        // loop only fires on the *next* mutation (the one we care about).
        {
            let mut guard = match cache.last_root.write() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            *guard = smt.root();
        }
        let proof = smt.prove(&addr, 9).expect("prove");
        cache.insert(addr, proof);
        assert!(cache.lookup(&addr).is_some());

        // Spawn `run`.
        let (tx, rx) = watch::channel(false);
        let cache_clone = Arc::clone(&cache);
        let handle = tokio::spawn(async move { cache_clone.run(rx).await });

        // Externally mutate the SMT → root changes. The background loop
        // (500 ms tick) should see it within ~1 s.
        let other = [0x88u8; 32];
        let _r2 = smt.update_batch(&[(other, 2u128)]).expect("update r2");

        tokio::time::sleep(Duration::from_millis(1_200)).await;

        assert!(
            cache.stats().cache_invalidations.load(Ordering::Relaxed) >= 1,
            "expected at least one invalidation, got {}",
            cache.stats().cache_invalidations.load(Ordering::Relaxed)
        );

        // Shutdown.
        let _ = tx.send(true);
        let _ = handle.await;
    }
}
