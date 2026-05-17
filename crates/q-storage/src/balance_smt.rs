// crates/q-storage/src/balance_smt.rs
//
// Sparse Merkle Tree over (wallet_addr → balance) — depth 256, BLAKE3 internal hashes.
// Underpins `balance_root_v2` and the recursive-SNARK Merkle-path gadget.
//
// Storage model:
//   • Column family `cf_balance_smt` holds non-empty internal nodes and leaves.
//   • A key encodes (depth, path_prefix). Missing key ⇒ that subtree is empty
//     and its hash is the precomputed `empty_subtree[depth]`.
//   • Persisted root pointer at `KEY_PERSISTED_ROOT` for restart consistency.
//
// Bit convention: MSB-first. `addr_bit(addr, i)` returns bit `i` where i=0 is the
// most significant bit of `addr[0]`. The path from root (depth 0) to leaf (depth 256)
// consumes bits 0..256 in order.

use anyhow::{anyhow, Context, Result};
use blake3::Hasher;
use parking_lot::RwLock;
use rocksdb::{WriteBatch, DB};
use std::collections::HashMap;
use std::sync::Arc;

// ════════════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════════════

/// Column family name for the SMT.
pub const CF_BALANCE_SMT: &str = "cf_balance_smt";

/// Tree depth. 256-bit addresses ⇒ 256 path bits ⇒ leaf lives at depth 256.
pub const SMT_DEPTH: usize = 256;

/// Domain separator for leaf hash: BLAKE3("smt_leaf_v2" || addr || balance_le).
const LEAF_TAG: &[u8] = b"smt_leaf_v2";

/// Domain separator for internal node: BLAKE3("smt_node_v2" || left || right).
const NODE_TAG: &[u8] = b"smt_node_v2";

/// Reserved key for the persisted root pointer. v10.9.55: uses a distinct domain
/// byte ('R'=0x52) that cannot collide with the new `node_key` encoding ('N' for
/// internal nodes, 'L' for leaves).
const KEY_PERSISTED_ROOT: &[u8] = b"R__root__";

/// Reserved sentinel marking an in-progress rebuild. Set BEFORE the CF truncate
/// in `rebuild_from_balances` and CLEARED in the same WriteBatch as
/// KEY_PERSISTED_ROOT. If `BalanceSmt::open` finds this key, the prior rebuild
/// was interrupted (crash between truncate and persist) and the SMT state is
/// inconsistent — open refuses rather than silently initializing to genesis on
/// top of a partially-cleared CF. v10.9.55 Codex review (HIGH).
const KEY_REBUILD_IN_PROGRESS: &[u8] = b"R__rebuilding__";

/// v10.9.55 (C2, Codex review 2026-05-17): key-encoding domain bytes.
///
/// PRE-v10.9.55 BUG: `node_key(depth=254, addr)` and `node_key(depth=256, addr)`
/// (leaf) both produced 33-byte keys whose first byte was 0xFE (depth=254 → u8 byte
/// 0xFE, leaf sentinel also 0xFE). For any address whose lowest 2 bits are 00
/// (≈25% of addresses), the depth-254 internal-node key and the leaf key were
/// byte-identical → silently collided in RocksDB. Apply order would decide which
/// one "won", producing nondeterministic SMT state across nodes — a consensus-fork
/// hazard at BalanceRootV2 activation (or at 100M-wallet scale, ~25M proofs would
/// be silently wrong).
///
/// FIX: explicit domain bytes for node vs leaf, and a u16-BE depth field. Now
/// node_key(depth, addr) = [b'N', depth_hi, depth_lo, prefix_bytes...]
/// leaf_key(addr)        = [b'L', addr_full_32_bytes...]
/// These can never collide regardless of depth or address.
///
/// Migration: BalanceRootV2 is currently DORMANT on mainnet (u64::MAX activation).
/// Any persisted SMT data from before this fix is per-node shadow-mode only; can
/// be discarded by deleting `cf_balance_smt` contents and letting the SMT rebuild
/// from the wallet table at next access. See `rebuild_from_balances()`.
const NODE_DOMAIN: u8 = b'N';
const LEAF_DOMAIN: u8 = b'L';

// ════════════════════════════════════════════════════════════════════════════
// Pure helpers (deterministic, no DB access)
// ════════════════════════════════════════════════════════════════════════════

/// BLAKE3("smt_leaf_v2" || addr || balance_le).
#[inline]
fn leaf_hash_raw(addr: &[u8; 32], balance: u128) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(LEAF_TAG);
    h.update(addr);
    h.update(&balance.to_le_bytes());
    *h.finalize().as_bytes()
}

/// BLAKE3("smt_node_v2" || left || right).
#[inline]
fn node_hash_raw(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(NODE_TAG);
    h.update(left);
    h.update(right);
    *h.finalize().as_bytes()
}

/// Precompute empty-subtree hashes for depths 0..=SMT_DEPTH.
///
/// `empty[SMT_DEPTH]` is the empty-leaf hash (`leaf_hash(0_addr, 0_balance)`).
/// `empty[d]` for `d < SMT_DEPTH` is `node_hash(empty[d+1], empty[d+1])`.
fn precompute_empty_subtrees() -> [[u8; 32]; SMT_DEPTH + 1] {
    let mut e = [[0u8; 32]; SMT_DEPTH + 1];
    e[SMT_DEPTH] = leaf_hash_raw(&[0u8; 32], 0);
    for d in (0..SMT_DEPTH).rev() {
        e[d] = node_hash_raw(&e[d + 1], &e[d + 1]);
    }
    e
}

/// Read MSB-first bit `i` from `addr` (0 = most significant bit of `addr[0]`).
#[inline]
fn addr_bit(addr: &[u8; 32], i: usize) -> bool {
    debug_assert!(i < 256);
    (addr[i / 8] >> (7 - (i % 8))) & 1 == 1
}

/// Return `addr` with bit `i` flipped.
#[inline]
fn flip_bit(addr: &[u8; 32], i: usize) -> [u8; 32] {
    debug_assert!(i < 256);
    let mut out = *addr;
    out[i / 8] ^= 1 << (7 - (i % 8));
    out
}

/// Encode a node key for the given depth and path-prefix (the first `depth` bits of `addr`).
/// v10.9.55: domain-separated key encoding (closes the depth-254/leaf collision).
///
/// Key layout:
/// - Internal node at depth `d` ∈ [0, SMT_DEPTH-1]:
///     `[NODE_DOMAIN | depth_be_u16 | prefix_bytes]`
///   where `prefix_bytes` = first `ceil(d/8)` bytes of `addr` with low bits beyond
///   `depth` zeroed.
/// - Leaf at depth SMT_DEPTH (=256):
///     `[LEAF_DOMAIN | addr_full_32_bytes]`
///
/// Examples:
/// - depth=0 root:  `[b'N', 0x00, 0x00]`
/// - depth=8:       `[b'N', 0x00, 0x08, addr[0]]`
/// - depth=12:      `[b'N', 0x00, 0x0C, addr[0], addr[1] & 0xF0]`
/// - depth=254:     `[b'N', 0x00, 0xFE, addr[0..31], addr[31] & 0xFC]`
/// - depth=256:     `[b'L', addr[0..32]]`  ← distinct domain, can never collide
fn node_key(depth: usize, addr: &[u8; 32]) -> Vec<u8> {
    debug_assert!(depth <= SMT_DEPTH);
    if depth == SMT_DEPTH {
        // Leaf row — distinct domain byte, never collides with internal-node keys.
        let mut key = Vec::with_capacity(1 + 32);
        key.push(LEAF_DOMAIN);
        key.extend_from_slice(addr);
        return key;
    }
    let full_bytes = depth / 8;
    let extra_bits = depth % 8;
    let prefix_len = full_bytes + (if extra_bits > 0 { 1 } else { 0 });
    // 3-byte header: domain + u16 BE depth. Header is fixed-length so any two
    // (depth, addr) pairs produce distinct keys iff (depth, prefix_bytes) differ.
    let mut key = Vec::with_capacity(3 + prefix_len);
    key.push(NODE_DOMAIN);
    key.extend_from_slice(&(depth as u16).to_be_bytes());
    key.extend_from_slice(&addr[..full_bytes]);
    if extra_bits > 0 {
        let mask = 0xFFu8 << (8 - extra_bits);
        key.push(addr[full_bytes] & mask);
    }
    key
}

/// Key of the sibling at depth `d+1` reached from the node at depth `d`
/// when descending toward `addr`.
fn sibling_key(d: usize, addr: &[u8; 32]) -> Vec<u8> {
    let flipped = flip_bit(addr, d);
    node_key(d + 1, &flipped)
}

// ════════════════════════════════════════════════════════════════════════════
// Public types
// ════════════════════════════════════════════════════════════════════════════

/// Merkle proof for `(addr → balance)` against an SMT root.
///
/// `empty_bitmap` packs 256 bits MSB-first by byte: bit `i` (byte `i/8`,
/// bit-within-byte `7 - (i%8)`) is set ⇔ `siblings[i]` is the empty subtree
/// hash for depth `i+1`. Lets the in-circuit verifier substitute the precomputed
/// constant instead of carrying the explicit hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmtProof {
    pub addr: [u8; 32],
    pub balance: u128,
    pub siblings: [[u8; 32]; SMT_DEPTH],
    pub empty_bitmap: [u8; 32],
}

impl SmtProof {
    /// Verify against an expected root. Independent of any DB state — uses only
    /// the precomputed empty-subtree constants. Suitable for the wire-protocol
    /// `/api/v1/proof/balance/:addr` consumer side.
    pub fn verify(&self, expected_root: &[u8; 32]) -> bool {
        let empty = precompute_empty_subtrees();
        let mut current = leaf_hash_raw(&self.addr, self.balance);

        // Walk leaf → root: depth 255 (parent of leaf) up to depth 0 (root).
        // At iteration with index `d_from_leaf` (0..256), we're computing the
        // parent at depth `SMT_DEPTH - 1 - d_from_leaf` from its two children at
        // depth `SMT_DEPTH - d_from_leaf`. The bit that decides left/right is
        // `addr_bit(SMT_DEPTH - 1 - d_from_leaf)`.
        for d_from_leaf in 0..SMT_DEPTH {
            let depth = SMT_DEPTH - 1 - d_from_leaf;
            let bit = addr_bit(&self.addr, depth);
            let sib_is_empty =
                (self.empty_bitmap[depth / 8] >> (7 - (depth % 8))) & 1 == 1;
            let sibling = if sib_is_empty {
                &empty[depth + 1]
            } else {
                &self.siblings[depth]
            };
            let (left, right) = if bit {
                (sibling, &current)
            } else {
                (&current, sibling)
            };
            current = node_hash_raw(left, right);
        }
        &current == expected_root
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BalanceSmt
// ════════════════════════════════════════════════════════════════════════════

/// Sparse Merkle Tree backing `balance_root_v2`.
///
/// Thread-safe: the cached root is guarded by `RwLock`. Updates compose into an
/// external `WriteBatch` via `apply_to_batch` so the SMT and the wallet-balance
/// column family commit atomically.
pub struct BalanceSmt {
    db: Arc<DB>,
    cf_name: String,
    /// Cached current root. Updated atomically with `commit_root` AFTER the
    /// caller's WriteBatch has been written to disk.
    cached_root: RwLock<[u8; 32]>,
    /// Precomputed `empty_subtree[d]` for d ∈ 0..=SMT_DEPTH.
    empty_subtree: [[u8; 32]; SMT_DEPTH + 1],
    /// `empty_subtree[0]` cached for convenience — the root of an empty tree.
    genesis_root: [u8; 32],
}

impl BalanceSmt {
    /// Open the SMT on top of an existing RocksDB instance. Caller must have
    /// created `CF_BALANCE_SMT` before opening (see `add_smt_cf_to_descriptors`).
    pub fn open(db: Arc<DB>) -> Result<Self> {
        // Verify the CF exists.
        let _ = db
            .cf_handle(CF_BALANCE_SMT)
            .ok_or_else(|| anyhow!("column family {} missing — was DB opened with it?", CF_BALANCE_SMT))?;

        // v10.9.55 Codex HIGH: refuse to open if rebuild was interrupted.
        // KEY_REBUILD_IN_PROGRESS is set inside the truncate batch and cleared
        // atomically with KEY_PERSISTED_ROOT. Its presence means rebuild crashed
        // between the two writes and CF_BALANCE_SMT is in an inconsistent state.
        // Returning Err here forces the operator to rerun rebuild rather than
        // silently initializing to a possibly-wrong genesis root.
        {
            let cf = db.cf_handle(CF_BALANCE_SMT).unwrap();
            if db.get_cf(&cf, KEY_REBUILD_IN_PROGRESS).context("reading SMT rebuild sentinel")?.is_some() {
                return Err(anyhow!(
                    "BalanceSmt CF_BALANCE_SMT was left in an interrupted-rebuild state \
                     (sentinel KEY_REBUILD_IN_PROGRESS present). Crash between truncate \
                     and persist. Rerun `rebuild_balance_smt_from_wallet_table()` via the \
                     admin endpoint, or clear the sentinel manually after confirming \
                     consistency. Refusing to open with possibly-corrupt SMT state."
                ));
            }
        }

        let empty = precompute_empty_subtrees();
        let genesis_root = empty[0];

        // Read persisted root (if any). Missing ⇒ fresh DB ⇒ root = genesis.
        let persisted = {
            let cf = db.cf_handle(CF_BALANCE_SMT).unwrap();
            db.get_cf(&cf, KEY_PERSISTED_ROOT)
                .context("reading persisted SMT root")?
        };
        let initial_root = match persisted {
            Some(bytes) if bytes.len() == 32 => {
                let mut r = [0u8; 32];
                r.copy_from_slice(&bytes);
                r
            }
            Some(_) => return Err(anyhow!("persisted SMT root has wrong length")),
            None => genesis_root,
        };

        Ok(Self {
            db,
            cf_name: CF_BALANCE_SMT.to_string(),
            cached_root: RwLock::new(initial_root),
            empty_subtree: empty,
            genesis_root,
        })
    }

    /// Current cached root.
    pub fn root(&self) -> [u8; 32] {
        *self.cached_root.read()
    }

    /// Root of an empty tree — the genesis root constant.
    pub fn genesis_root(&self) -> [u8; 32] {
        self.genesis_root
    }

    /// Update the cached root. **MUST** be called after the caller's `WriteBatch`
    /// containing `apply_to_batch`'s writes has been committed to disk.
    pub fn commit_root(&self, new_root: [u8; 32]) {
        *self.cached_root.write() = new_root;
    }

    /// Rebuild from a complete `(addr → balance)` snapshot. Used at fresh-start
    /// from a balance checkpoint, and at BalanceRootV2 activation.
    ///
    /// v10.9.55 (C3 + H4, Codex review 2026-05-17): pure function of `balances`.
    /// 1. Truncate the SMT CF before rebuild — eliminates stale-node contamination.
    ///    apply_to_batch_internal reads siblings via `pending → DB → empty_subtree`,
    ///    so any leftover node from a prior shadow rebuild could fold into the new
    ///    root. Now wiped first.
    /// 2. Sort updates by address — defensive against any subtle ordering dependency
    ///    in the pending-map propagation. Guarantees byte-identical root across N
    ///    nodes that may hash-randomize HashMaps differently.
    pub fn rebuild_from_balances(
        &self,
        balances: &HashMap<[u8; 32], u128>,
    ) -> Result<[u8; 32]> {
        let cf = self.db.cf_handle(&self.cf_name).unwrap();

        // (1) Truncate + sentinel in ONE atomic batch.
        //
        // Crash-atomicity: there are two RocksDB writes in this function (truncate,
        // then rebuilt-state). A crash between them used to leave the CF empty with
        // no persisted root — BalanceSmt::open() would then silently initialize to
        // genesis (Codex 2026-05-18 HIGH). The sentinel `R__rebuilding__` is written
        // INSIDE the truncate batch (after delete_range_cf in batch insertion order,
        // so it survives the broad-range delete) and CLEARED in the second batch.
        // If a crash occurs between the two writes, the sentinel persists and
        // BalanceSmt::open() refuses to load until rebuild is rerun.
        {
            let mut clear_batch = WriteBatch::default();
            clear_batch.delete_range_cf(&cf, &[0x00u8], &[0xFFu8; 64]);
            clear_batch.put_cf(&cf, KEY_REBUILD_IN_PROGRESS, &[1u8]);
            self.db.write(clear_batch).context("truncating SMT CF before rebuild")?;
        }

        // Reset cache to genesis so apply_one starts from empty.
        *self.cached_root.write() = self.genesis_root;

        // (2) Sort updates by address for determinism. `[u8; 32]` has total ordering.
        let mut updates: Vec<([u8; 32], u128)> =
            balances.iter().map(|(a, b)| (*a, *b)).collect();
        updates.sort_unstable_by_key(|(addr, _)| *addr);

        let mut batch = WriteBatch::default();
        let new_root = self.apply_to_batch_internal(&cf, &mut batch, &updates)?;

        // (3) Persist new root + CLEAR the in-progress sentinel atomically. After
        // db.write succeeds, BalanceSmt::open() will load the new root cleanly.
        batch.put_cf(&cf, KEY_PERSISTED_ROOT, new_root);
        batch.delete_cf(&cf, KEY_REBUILD_IN_PROGRESS);
        self.db.write(batch).context("writing SMT rebuild batch")?;
        *self.cached_root.write() = new_root;
        Ok(new_root)
    }

    /// Convenience: apply a batch of updates and commit immediately.
    /// Returns the new root.
    pub fn update_batch(&self, updates: &[([u8; 32], u128)]) -> Result<[u8; 32]> {
        let mut batch = WriteBatch::default();
        let cf = self.db.cf_handle(&self.cf_name).unwrap();
        let new_root = self.apply_to_batch_internal(&cf, &mut batch, updates)?;
        batch.put_cf(&cf, KEY_PERSISTED_ROOT, new_root);
        self.db.write(batch).context("writing SMT update batch")?;
        *self.cached_root.write() = new_root;
        Ok(new_root)
    }

    /// Compose with an external `WriteBatch` (e.g., the one used by
    /// `save_wallet_balances`). Returns the new root that will be live AFTER
    /// the caller's batch is committed.
    ///
    /// **Required follow-up** after `db.write(batch)`:
    ///   `smt.commit_root(returned_root);`
    ///
    /// Also writes `KEY_PERSISTED_ROOT → new_root` into the batch so the root
    /// pointer is consistent with the leaves on disk.
    pub fn apply_to_batch(
        &self,
        batch: &mut WriteBatch,
        updates: &[([u8; 32], u128)],
    ) -> Result<[u8; 32]> {
        let cf = self.db.cf_handle(&self.cf_name).unwrap();
        let new_root = self.apply_to_batch_internal(&cf, batch, updates)?;
        batch.put_cf(&cf, KEY_PERSISTED_ROOT, new_root);
        Ok(new_root)
    }

    /// Internal: apply updates without writing the persisted-root pointer.
    /// Maintains a `pending` map so updates within the same batch see each
    /// other's intermediate node writes.
    fn apply_to_batch_internal(
        &self,
        cf: &Arc<rocksdb::BoundColumnFamily<'_>>,
        batch: &mut WriteBatch,
        updates: &[([u8; 32], u128)],
    ) -> Result<[u8; 32]> {
        // pending: in-flight node writes for this batch.
        // Lookup priority: pending → DB → empty_subtree.
        let mut pending: HashMap<Vec<u8>, [u8; 32]> = HashMap::new();
        let mut current_root = self.root();

        for (addr, balance) in updates {
            current_root = self.apply_one(cf, &mut pending, current_root, addr, *balance)?;
        }

        // Flush all pending nodes to the batch.
        for (key, hash) in &pending {
            batch.put_cf(cf, key, hash);
        }
        Ok(current_root)
    }

    /// Apply one (addr → balance) update. Reads siblings (pending or DB), writes
    /// the new leaf + 256 internal nodes into `pending`, returns the new root.
    fn apply_one(
        &self,
        cf: &Arc<rocksdb::BoundColumnFamily<'_>>,
        pending: &mut HashMap<Vec<u8>, [u8; 32]>,
        prev_root: [u8; 32],
        addr: &[u8; 32],
        balance: u128,
    ) -> Result<[u8; 32]> {
        // Collect siblings on the descent path.
        // siblings[d] is the sibling at depth d+1 reached when descending past depth d.
        let mut siblings = [[0u8; 32]; SMT_DEPTH];
        for d in 0..SMT_DEPTH {
            siblings[d] = self.load_node(cf, pending, &sibling_key(d, addr), d + 1)?;
        }
        // (We don't actually need the old leaf hash — we overwrite with the new one.)
        // But we sanity-check that `prev_root` matches what siblings + old leaf reproduce
        // ONLY in debug builds, since this is an O(256) extra hash chain per update.
        #[cfg(debug_assertions)]
        {
            let leaf_key = node_key(SMT_DEPTH, addr);
            let old_leaf = self.load_node(cf, pending, &leaf_key, SMT_DEPTH)?;
            let recomputed = self.fold_to_root(addr, &old_leaf, &siblings);
            if recomputed != prev_root {
                return Err(anyhow!(
                    "SMT invariant violation: descent root mismatch for addr {:x?} \
                     (expected {:x?}, got {:x?})",
                    addr,
                    prev_root,
                    recomputed
                ));
            }
        }

        // Compute new leaf hash.
        let new_leaf = leaf_hash_raw(addr, balance);
        pending.insert(node_key(SMT_DEPTH, addr), new_leaf);

        // Walk leaf → root, writing each new internal node into `pending`.
        let new_root = self.fold_to_root_writing(addr, &new_leaf, &siblings, pending);
        Ok(new_root)
    }

    /// Read a node hash: pending (in-flight) → DB → empty_subtree fallback.
    fn load_node(
        &self,
        cf: &Arc<rocksdb::BoundColumnFamily<'_>>,
        pending: &HashMap<Vec<u8>, [u8; 32]>,
        key: &[u8],
        node_depth: usize,
    ) -> Result<[u8; 32]> {
        if let Some(h) = pending.get(key) {
            return Ok(*h);
        }
        match self.db.get_cf(cf, key).context("SMT node read")? {
            Some(v) if v.len() == 32 => {
                let mut h = [0u8; 32];
                h.copy_from_slice(&v);
                Ok(h)
            }
            Some(_) => Err(anyhow!("SMT node has wrong byte length at depth {}", node_depth)),
            None => Ok(self.empty_subtree[node_depth]),
        }
    }

    /// Fold leaf → root using the supplied siblings. Pure computation, no writes.
    fn fold_to_root(
        &self,
        addr: &[u8; 32],
        leaf: &[u8; 32],
        siblings: &[[u8; 32]; SMT_DEPTH],
    ) -> [u8; 32] {
        let mut current = *leaf;
        for d_from_leaf in 0..SMT_DEPTH {
            let depth = SMT_DEPTH - 1 - d_from_leaf;
            let sibling = &siblings[depth];
            let (left, right) = if addr_bit(addr, depth) {
                (sibling, &current)
            } else {
                (&current, sibling)
            };
            current = node_hash_raw(left, right);
        }
        current
    }

    /// Same as `fold_to_root` but also writes each computed parent into `pending`.
    fn fold_to_root_writing(
        &self,
        addr: &[u8; 32],
        leaf: &[u8; 32],
        siblings: &[[u8; 32]; SMT_DEPTH],
        pending: &mut HashMap<Vec<u8>, [u8; 32]>,
    ) -> [u8; 32] {
        let mut current = *leaf;
        for d_from_leaf in 0..SMT_DEPTH {
            let depth = SMT_DEPTH - 1 - d_from_leaf;
            let sibling = &siblings[depth];
            let (left, right) = if addr_bit(addr, depth) {
                (sibling, &current)
            } else {
                (&current, sibling)
            };
            let parent = node_hash_raw(left, right);
            // Write parent at this depth, addressed by `addr`'s prefix of length `depth`.
            pending.insert(node_key(depth, addr), parent);
            current = parent;
        }
        current
    }

    /// Generate a Merkle proof for `(addr → balance)`. The proof is verifiable
    /// against the current `root()` (or any historical root, if the caller has
    /// a different one to check against — the proof itself is just structural).
    pub fn prove(&self, addr: &[u8; 32], balance: u128) -> Result<SmtProof> {
        let cf = self.db.cf_handle(&self.cf_name).unwrap();
        let pending: HashMap<Vec<u8>, [u8; 32]> = HashMap::new();
        let mut siblings = [[0u8; 32]; SMT_DEPTH];
        let mut empty_bitmap = [0u8; 32];
        for d in 0..SMT_DEPTH {
            let sib = self.load_node(&cf, &pending, &sibling_key(d, addr), d + 1)?;
            siblings[d] = sib;
            if sib == self.empty_subtree[d + 1] {
                empty_bitmap[d / 8] |= 1 << (7 - (d % 8));
            }
        }
        Ok(SmtProof {
            addr: *addr,
            balance,
            siblings,
            empty_bitmap,
        })
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use rocksdb::{ColumnFamilyDescriptor, Options, DB};
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn open_test_db() -> (Arc<DB>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new("default", Options::default()),
            ColumnFamilyDescriptor::new(CF_BALANCE_SMT, Options::default()),
        ];
        let db = DB::open_cf_descriptors(&opts, tmp.path(), cf_descriptors).unwrap();
        (Arc::new(db), tmp)
    }

    #[test]
    fn empty_tree_root_matches_genesis_constant() {
        let (db, _tmp) = open_test_db();
        let smt = BalanceSmt::open(db).unwrap();
        assert_eq!(smt.root(), smt.genesis_root());
    }

    #[test]
    fn empty_tree_genesis_root_is_deterministic() {
        // Run twice on fresh DBs; root must be byte-identical.
        let (db1, _t1) = open_test_db();
        let s1 = BalanceSmt::open(db1).unwrap();
        let (db2, _t2) = open_test_db();
        let s2 = BalanceSmt::open(db2).unwrap();
        assert_eq!(s1.genesis_root(), s2.genesis_root());
    }

    #[test]
    fn single_insert_changes_root_and_verifies() {
        let (db, _tmp) = open_test_db();
        let smt = BalanceSmt::open(db).unwrap();
        let addr = [0x42u8; 32];
        let bal = 12345u128;
        let r0 = smt.root();
        let r1 = smt.update_batch(&[(addr, bal)]).unwrap();
        assert_ne!(r0, r1);
        let proof = smt.prove(&addr, bal).unwrap();
        assert!(proof.verify(&r1));
        assert!(!proof.verify(&r0));
    }

    #[test]
    fn batch_update_matches_sequential() {
        let (db_a, _ta) = open_test_db();
        let smt_a = BalanceSmt::open(db_a).unwrap();

        let mut updates = Vec::new();
        for i in 0..50u8 {
            let mut addr = [0u8; 32];
            addr[0] = i;
            addr[31] = i.wrapping_mul(7);
            updates.push((addr, (i as u128) * 1_000));
        }

        // Batched
        let root_batch = smt_a.update_batch(&updates).unwrap();

        // Sequential
        let (db_b, _tb) = open_test_db();
        let smt_b = BalanceSmt::open(db_b).unwrap();
        for u in &updates {
            smt_b.update_batch(&[*u]).unwrap();
        }
        assert_eq!(root_batch, smt_b.root());
    }

    #[test]
    fn overlapping_batch_updates_produce_correct_root() {
        // Two updates touching addresses with shared prefix bits.
        let (db_a, _ta) = open_test_db();
        let smt_a = BalanceSmt::open(db_a).unwrap();

        let mut a1 = [0u8; 32];
        a1[0] = 0b1010_0000;
        let mut a2 = [0u8; 32];
        a2[0] = 0b1010_1000; // shares first 4 bits with a1
        let updates = vec![(a1, 100u128), (a2, 200u128)];

        let root_batch = smt_a.update_batch(&updates).unwrap();

        let (db_b, _tb) = open_test_db();
        let smt_b = BalanceSmt::open(db_b).unwrap();
        smt_b.update_batch(&[(a1, 100)]).unwrap();
        smt_b.update_batch(&[(a2, 200)]).unwrap();
        assert_eq!(root_batch, smt_b.root());
    }

    #[test]
    fn proof_against_wrong_root_fails() {
        let (db, _tmp) = open_test_db();
        let smt = BalanceSmt::open(db).unwrap();
        let addr = [0x77u8; 32];
        let root = smt.update_batch(&[(addr, 999)]).unwrap();
        let proof = smt.prove(&addr, 999).unwrap();
        let mut bad = root;
        bad[0] ^= 1;
        assert!(!proof.verify(&bad));
    }

    #[test]
    fn proof_with_tampered_sibling_fails() {
        let (db, _tmp) = open_test_db();
        let smt = BalanceSmt::open(db).unwrap();
        let addr = [0x55u8; 32];
        let root = smt.update_batch(&[(addr, 42)]).unwrap();
        let mut proof = smt.prove(&addr, 42).unwrap();
        // Flip one byte of one sibling (one whose empty_bitmap bit is 0,
        // i.e. the real sibling, not the empty placeholder)
        for i in 0..SMT_DEPTH {
            let bit_set = (proof.empty_bitmap[i / 8] >> (7 - (i % 8))) & 1 == 1;
            if !bit_set {
                proof.siblings[i][0] ^= 1;
                break;
            }
        }
        assert!(!proof.verify(&root));
    }

    #[test]
    fn proof_with_wrong_balance_fails() {
        let (db, _tmp) = open_test_db();
        let smt = BalanceSmt::open(db).unwrap();
        let addr = [0xAAu8; 32];
        let root = smt.update_batch(&[(addr, 500)]).unwrap();
        // Generate proof but lie about the balance
        let lying = SmtProof {
            addr,
            balance: 501, // wrong
            siblings: smt.prove(&addr, 500).unwrap().siblings,
            empty_bitmap: smt.prove(&addr, 500).unwrap().empty_bitmap,
        };
        assert!(!lying.verify(&root));
    }

    #[test]
    fn rebuild_from_balances_idempotent() {
        let (db, _tmp) = open_test_db();
        let smt = BalanceSmt::open(db).unwrap();
        let mut balances = HashMap::new();
        for i in 0..30u8 {
            let mut a = [0u8; 32];
            a[0] = i;
            balances.insert(a, (i as u128) + 1);
        }
        let r1 = smt.rebuild_from_balances(&balances).unwrap();
        let r2 = smt.rebuild_from_balances(&balances).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn restart_loads_persisted_root() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Phase 1: open, write some balances, close.
        let r_before = {
            let cfs = vec![
                ColumnFamilyDescriptor::new("default", Options::default()),
                ColumnFamilyDescriptor::new(CF_BALANCE_SMT, Options::default()),
            ];
            let db = Arc::new(DB::open_cf_descriptors(&opts, &path, cfs).unwrap());
            let smt = BalanceSmt::open(db).unwrap();
            let mut updates = Vec::new();
            for i in 0..10u8 {
                let mut a = [0u8; 32];
                a[i as usize % 32] = i;
                updates.push((a, (i as u128) + 100));
            }
            smt.update_batch(&updates).unwrap()
        };

        // Phase 2: reopen, root must be the persisted value, not genesis.
        let cfs = vec![
            ColumnFamilyDescriptor::new("default", Options::default()),
            ColumnFamilyDescriptor::new(CF_BALANCE_SMT, Options::default()),
        ];
        let db = Arc::new(DB::open_cf_descriptors(&opts, &path, cfs).unwrap());
        let smt = BalanceSmt::open(db).unwrap();
        assert_eq!(smt.root(), r_before);
        assert_ne!(smt.root(), smt.genesis_root());
    }

    #[test]
    fn apply_to_batch_composes_with_external_writebatch() {
        let (db, _tmp) = open_test_db();
        let smt = BalanceSmt::open(db.clone()).unwrap();

        let updates = vec![
            ([1u8; 32], 100u128),
            ([2u8; 32], 200u128),
        ];

        let mut external_batch = WriteBatch::default();
        // Imagine: also writing wallet balances to another CF.
        let new_root = smt.apply_to_batch(&mut external_batch, &updates).unwrap();
        db.write(external_batch).unwrap();
        smt.commit_root(new_root);

        // Verify both updates are reflected.
        for (addr, bal) in &updates {
            let p = smt.prove(addr, *bal).unwrap();
            assert!(p.verify(&new_root));
        }
    }

    #[test]
    fn proof_compresses_empty_siblings() {
        // For a tree with a single entry, ~all siblings should be empty.
        let (db, _tmp) = open_test_db();
        let smt = BalanceSmt::open(db).unwrap();
        let addr = [0x33u8; 32];
        smt.update_batch(&[(addr, 1)]).unwrap();
        let proof = smt.prove(&addr, 1).unwrap();
        let empty_count = (0..SMT_DEPTH)
            .filter(|d| (proof.empty_bitmap[d / 8] >> (7 - (d % 8))) & 1 == 1)
            .count();
        // With one entry, all 256 siblings are empty.
        assert_eq!(empty_count, SMT_DEPTH);
    }

    #[test]
    fn flip_bit_round_trip() {
        let a = [0xABu8; 32];
        for i in 0..256 {
            let f = flip_bit(&a, i);
            assert_ne!(a, f);
            assert_eq!(flip_bit(&f, i), a);
        }
    }

    #[test]
    fn node_key_uniqueness_at_different_depths() {
        let a = [0xFFu8; 32];
        let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        for d in 0..=SMT_DEPTH {
            let k = node_key(d, &a);
            assert!(seen.insert(k.clone()), "duplicate key at depth {}", d);
        }
    }

    /// v10.9.55 Codex MEDIUM: explicit regression for the pre-v10.9.55 depth-254
    /// vs leaf collision class. Pre-fix, `node_key(254, addr)` produced
    /// `[0xFE, addr[0..31], addr[31] & 0xFC]` and `node_key(256, addr)` (leaf)
    /// produced `[0xFE, addr[0..32]]` — byte-identical for any address with
    /// low 2 bits = 00. This test exercises that exact shape across many addresses
    /// AND across multiple depth pairs (254/256, 255/256, KEY_PERSISTED_ROOT).
    #[test]
    fn node_key_no_collision_in_old_failure_shape_v10955() {
        // Addresses that would have collided pre-fix (last byte's low 2 bits = 00).
        let collision_addrs: Vec<[u8; 32]> = vec![
            {
                let mut a = [0u8; 32];
                a[31] = 0x00; // 0b00000000
                a
            },
            {
                let mut a = [0u8; 32];
                a[31] = 0x04; // 0b00000100
                a
            },
            {
                let mut a = [0xAAu8; 32];
                a[31] = 0xFC; // 0b11111100
                a
            },
            {
                let mut a = [0x55u8; 32];
                a[31] = 0x50; // 0b01010000
                a
            },
        ];

        for addr in &collision_addrs {
            let k254 = node_key(254, addr);
            let k255 = node_key(255, addr);
            let kleaf = node_key(SMT_DEPTH, addr);
            assert_ne!(
                k254, kleaf,
                "depth-254 and leaf MUST NOT collide for addr {:?}", addr
            );
            assert_ne!(
                k255, kleaf,
                "depth-255 and leaf MUST NOT collide for addr {:?}", addr
            );
            assert_ne!(
                k254, k255,
                "depth-254 and depth-255 MUST be distinct for addr {:?}", addr
            );
        }

        // Sentinel keys must not collide with any node_key output either.
        let any_addr = [0u8; 32];
        for d in 0..=SMT_DEPTH {
            let k = node_key(d, &any_addr);
            assert_ne!(k.as_slice(), KEY_PERSISTED_ROOT, "node_key collides with persisted-root at depth {}", d);
            assert_ne!(k.as_slice(), KEY_REBUILD_IN_PROGRESS, "node_key collides with rebuild-sentinel at depth {}", d);
        }
    }

    /// v10.9.55 Codex MEDIUM: rebuild determinism under DIVERGENT prior SMT CF
    /// histories. Two nodes can have byte-identical wallet tables but different
    /// stale SMT data from earlier shadow-mode runs. After rebuild, both must
    /// produce the same root (otherwise V2 activation forks).
    #[test]
    fn rebuild_independent_of_prior_smt_cf_contents() {
        use std::collections::HashMap;

        let (db_a, _t_a) = open_test_db();
        let (db_b, _t_b) = open_test_db();

        let smt_a = BalanceSmt::open(db_a).unwrap();
        let smt_b = BalanceSmt::open(db_b).unwrap();

        // Step 1: populate DB A with wallet set X. DB B stays empty.
        let mut wallet_set_x: HashMap<[u8; 32], u128> = HashMap::new();
        for i in 0u8..50 {
            let mut a = [0u8; 32];
            a[0] = i;
            a[31] = i.wrapping_mul(7); // mix in some low-bit variation
            wallet_set_x.insert(a, 1000 + i as u128);
        }
        smt_a.rebuild_from_balances(&wallet_set_x).unwrap();

        // Step 2: build wallet set Y (disjoint from X, fewer entries).
        let mut wallet_set_y: HashMap<[u8; 32], u128> = HashMap::new();
        for i in 100u8..130 {
            let mut a = [0u8; 32];
            a[0] = i;
            a[31] = i.wrapping_mul(11);
            wallet_set_y.insert(a, 5000 + i as u128);
        }

        // Step 3: rebuild BOTH databases with set Y.
        let root_a = smt_a.rebuild_from_balances(&wallet_set_y).unwrap();
        let root_b = smt_b.rebuild_from_balances(&wallet_set_y).unwrap();

        // Step 4: roots must be byte-identical despite different prior histories.
        assert_eq!(
            root_a, root_b,
            "rebuild_from_balances must be a pure function of the input map; \
             prior CF contents must not affect the result"
        );
    }
}
