//! v10.11.114f-sync5 — AUTHORITY IMPORT: make a FOLLOWER's balances equal to the authority's
//! (Epsilon's) at one height H, once, from a snapshot FILE — then never pay a block ≤ H again.
//!
//! Why (measured 2026-10-30 … 11-02 on the Stockholm follower): its balances came from a one-time
//! P2P import on trust, then older blocks were replayed over them; 13 wallets ended 486,831 QUG
//! apart from Epsilon at equal height while every block since is credited exactly. The past cannot
//! be recomputed from blocks (swaps, liquidity, bank loans and contract deploys never enter one),
//! so the honest repair is: take the authority's state at a stated height, from a file whose
//! sha256 the operator confirms, and say plainly that the follower agrees BY IMPORT up to H and
//! BY REPLAY after H.
//!
//! What it writes, in ONE RocksDB batch (all or nothing):
//!   * `wallet_balance_<hex>` = the snapshot's amount for every snapshot wallet; local wallets the
//!     snapshot does not hold are deleted (the authority has no such key).
//!   * `token_balance_<w>_<t>` = the snapshot's amount; local token entries the snapshot does not
//!     hold are set to 0 (not deleted: a deleted manifest key lets the older cf_token_balances copy
//!     reappear on the next load). The "QUG" TOKEN copy (token id "QUG\0…") is never touched in
//!     either direction — it is a separate store that holds garbage-shaped values on Epsilon.
//!   * DEX counters `dex_qug_debited:` / `dex_qug_credited:` = the snapshot's, and
//!     `dex_applied_net:` = credited − debited, so the 5-second DEX reconciler finds NOTHING left to
//!     apply (the imported balances already contain every trade). Local counters the snapshot does
//!     not hold are deleted.
//!   * `authority_import_credit_floor` = H and `authority_import_done:<sha256>` = the report.
//!
//! Refusals (nothing written): a snapshot below the local block pointer (blocks between would be
//! lost), an empty wallet set, a network id that differs, a wrong-looking amount or key.
//! A second run with the same file is a no-op (`already_applied`).

use crate::{QStorage, CF_BLOCKS, CF_MANIFEST, CF_TOKEN_BALANCES};
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Token id prefix of the "QUG" token-store copy: ASCII "QUG" then zeros.
const QUG_TOKEN_COPY_PREFIX: [u8; 4] = [0x51, 0x55, 0x47, 0x00];
pub const DONE_KEY_PREFIX: &str = "authority_import_done:";

pub fn is_qug_token_copy(token: &[u8; 32]) -> bool {
    token[..4] == QUG_TOKEN_COPY_PREFIX && token[4..].iter().all(|b| *b == 0)
}

#[derive(Debug, Clone, Default)]
pub struct AuthoritySnapshot {
    pub height: u64,
    pub network_id: Option<String>,
    pub wallets: BTreeMap<[u8; 32], u128>,
    pub tokens: BTreeMap<([u8; 32], [u8; 32]), u128>,
    pub dex_debited: BTreeMap<String, u128>,
    pub dex_credited: BTreeMap<String, u128>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ImportReport {
    pub file_sha256: String,
    pub height: u64,
    pub local_pointer: u64,
    pub already_applied: bool,
    pub wallets_written: usize,
    pub wallets_changed: usize,
    pub wallets_removed: usize,
    pub tokens_written: usize,
    pub tokens_zeroed: usize,
    pub token_copies_skipped: usize,
    pub dex_wallets: usize,
    pub dex_keys_removed: usize,
    pub supply_before: String,
    pub supply_after: String,
}

fn hex32(s: &str, what: &str) -> Result<[u8; 32]> {
    let v = hex::decode(s).with_context(|| format!("{what}: not hex: {s}"))?;
    if v.len() != 32 {
        bail!("{what}: expected 32 bytes, got {} ({s})", v.len());
    }
    let mut a = [0u8; 32];
    a.copy_from_slice(&v);
    Ok(a)
}

fn amount(v: &serde_json::Value, what: &str) -> Result<u128> {
    match v {
        serde_json::Value::String(s) => s.parse::<u128>().with_context(|| format!("{what}: bad amount {s}")),
        serde_json::Value::Number(n) => n
            .as_u64()
            .map(|x| x as u128)
            .with_context(|| format!("{what}: amount {n} is not an exact integer (export amounts as strings)")),
        other => bail!("{what}: amount is {other}"),
    }
}

impl AuthoritySnapshot {
    /// Parse the authority's own `GET /api/v1/sync/full-state` export (`{"data": {...}}` or bare).
    pub fn from_full_state_json(bytes: &[u8]) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_slice(bytes).context("snapshot is not JSON")?;
        let d = root.get("data").unwrap_or(&root);
        let height = d.get("block_height").and_then(|h| h.as_u64()).context("snapshot has no block_height")?;
        let network_id = d.get("network_id").and_then(|n| n.as_str()).map(|s| s.to_string());
        let mut snap = AuthoritySnapshot { height, network_id, ..Default::default() };
        let obj = |k: &str| d.get(k).and_then(|v| v.as_object());
        for (k, v) in obj("wallet_balances").context("snapshot has no wallet_balances")? {
            snap.wallets.insert(hex32(k, "wallet")?, amount(v, k)?);
        }
        if let Some(m) = obj("token_balances") {
            for (k, v) in m {
                let (w, t) = k.split_once('_').with_context(|| format!("token key without '_': {k}"))?;
                snap.tokens.insert((hex32(w, "token wallet")?, hex32(t, "token id")?), amount(v, k)?);
            }
        }
        for (field, into) in [("dex_qug_debited", &mut snap.dex_debited), ("dex_qug_credited", &mut snap.dex_credited)] {
            if let Some(m) = obj(field) {
                for (k, v) in m {
                    hex32(k, field)?; // same 64-hex form as the wallet keys
                    into.insert(k.to_lowercase(), amount(v, k)?);
                }
            }
        }
        if snap.wallets.is_empty() {
            bail!("snapshot holds 0 wallet balances");
        }
        Ok(snap)
    }

    pub fn wallet_supply(&self) -> u128 {
        self.wallets.values().fold(0u128, |a, b| a.saturating_add(*b))
    }
}

fn u128_le(v: &[u8]) -> Option<u128> {
    match v.len() {
        16 => Some(u128::from_le_bytes(v[..16].try_into().ok()?)),
        8 => Some(u64::from_le_bytes(v[..8].try_into().ok()?) as u128),
        _ => None,
    }
}

impl QStorage {
    /// Load the persisted credit floor (0 when this node never imported) and arm it for every
    /// pay path. Call at boot BEFORE any block is applied.
    pub async fn load_credit_floor(&self) -> Result<u64> {
        let floor = match self.hot_db.get(CF_MANIFEST, crate::balance_consensus::CREDIT_FLOOR_KEY).await? {
            Some(v) if v.len() == 8 => u64::from_le_bytes(v[..8].try_into().unwrap()),
            _ => 0,
        };
        crate::balance_consensus::set_credit_floor(floor);
        Ok(floor)
    }

    /// The raw persisted block pointer `qblock:latest` (big-endian u64), 0 if absent.
    pub async fn raw_block_pointer(&self) -> Result<u64> {
        Ok(match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(v) if v.len() == 8 => u64::from_be_bytes(v[..8].try_into().unwrap()),
            _ => 0,
        })
    }

    /// Apply an authority snapshot (see the module docs). `expected_network` must equal the
    /// snapshot's network id when both are known.
    pub async fn apply_authority_import(
        &self,
        snap: &AuthoritySnapshot,
        file_sha256: &str,
        expected_network: &str,
    ) -> Result<ImportReport> {
        let done_key = format!("{DONE_KEY_PREFIX}{file_sha256}");
        let local_pointer = self.raw_block_pointer().await?;
        let mut rep = ImportReport {
            file_sha256: file_sha256.to_string(),
            height: snap.height,
            local_pointer,
            ..Default::default()
        };
        if self.hot_db.get(CF_MANIFEST, done_key.as_bytes()).await?.is_some() {
            rep.already_applied = true;
            self.load_credit_floor().await?;
            return Ok(rep);
        }
        if let Some(n) = &snap.network_id {
            if n != expected_network {
                bail!("snapshot network {n} != this node's {expected_network} — refusing");
            }
        }
        if snap.height < local_pointer {
            bail!(
                "snapshot height {} is BELOW this node's block pointer {} — blocks in between would be \
                 lost; stop the node first, then take the snapshot — refusing",
                snap.height, local_pointer
            );
        }

        // ---- read what is there now ----
        let mut local_wallets: BTreeMap<[u8; 32], u128> = BTreeMap::new();
        for (k, v) in self.hot_db.scan_prefix(CF_MANIFEST, b"wallet_balance_").await? {
            if let (Ok(ks), Some(a)) = (String::from_utf8(k), u128_le(&v)) {
                if let Some(h) = ks.strip_prefix("wallet_balance_") {
                    if let Ok(w) = hex32(h, "local wallet") {
                        local_wallets.insert(w, a);
                    }
                }
            }
        }
        let mut local_tokens: BTreeMap<([u8; 32], [u8; 32]), u128> = BTreeMap::new();
        for (k, v) in self.hot_db.scan_prefix(CF_MANIFEST, b"token_balance_").await? {
            if let (Ok(ks), Some(a)) = (String::from_utf8(k), u128_le(&v)) {
                if let Some((w, t)) = ks.strip_prefix("token_balance_").and_then(|r| r.split_once('_')) {
                    if let (Ok(w), Ok(t)) = (hex32(w, "w"), hex32(t, "t")) {
                        local_tokens.insert((w, t), a);
                    }
                }
            }
        }
        // the older state-sync token store fills keys the manifest lacks — those need an explicit 0
        let mut state_sync_tokens: BTreeSet<([u8; 32], [u8; 32])> = BTreeSet::new();
        #[cfg(not(target_os = "windows"))]
        if let Some(db) = self.get_rocks_db_handle() {
            if let Some(cf) = db.cf_handle(CF_TOKEN_BALANCES) {
                for item in db.iterator_cf(&cf, rocksdb::IteratorMode::Start) {
                    if let Ok((k, _)) = item {
                        if k.len() == 64 {
                            let (mut w, mut t) = ([0u8; 32], [0u8; 32]);
                            w.copy_from_slice(&k[..32]);
                            t.copy_from_slice(&k[32..]);
                            state_sync_tokens.insert((w, t));
                        }
                    }
                }
            }
        }
        let mut local_dex_keys: Vec<Vec<u8>> = Vec::new();
        for p in [&b"dex_qug_debited:"[..], &b"dex_qug_credited:"[..], &b"dex_applied_net:"[..]] {
            for (k, _) in self.hot_db.scan_prefix(CF_MANIFEST, p).await? {
                local_dex_keys.push(k);
            }
        }

        // ---- build the batch ----
        let db = self.get_rocks_db_handle().context("no RocksDB handle")?;
        let cf = db.cf_handle(CF_MANIFEST).context("no manifest column family")?;
        let mut batch = rocksdb::WriteBatch::default();
        let mut smt: Vec<([u8; 32], u128)> = Vec::new();

        for (w, a) in &snap.wallets {
            batch.put_cf(&cf, format!("wallet_balance_{}", hex::encode(w)).as_bytes(), a.to_le_bytes());
            rep.wallets_written += 1;
            if local_wallets.get(w) != Some(a) {
                rep.wallets_changed += 1;
                smt.push((*w, *a));
            }
        }
        for w in local_wallets.keys().filter(|w| !snap.wallets.contains_key(*w)) {
            batch.delete_cf(&cf, format!("wallet_balance_{}", hex::encode(w)).as_bytes());
            rep.wallets_removed += 1;
            smt.push((*w, 0));
        }

        for ((w, t), a) in &snap.tokens {
            if is_qug_token_copy(t) {
                rep.token_copies_skipped += 1;
                continue;
            }
            batch.put_cf(&cf, format!("token_balance_{}_{}", hex::encode(w), hex::encode(t)).as_bytes(), a.to_le_bytes());
            rep.tokens_written += 1;
        }
        let stale: BTreeSet<([u8; 32], [u8; 32])> = local_tokens
            .iter()
            .filter(|(_, a)| **a != 0)
            .map(|(k, _)| *k)
            .chain(state_sync_tokens.iter().copied().filter(|k| !local_tokens.contains_key(k)))
            .filter(|(w, t)| !is_qug_token_copy(t) && !snap.tokens.contains_key(&(*w, *t)))
            .collect();
        for (w, t) in &stale {
            batch.put_cf(&cf, format!("token_balance_{}_{}", hex::encode(w), hex::encode(t)).as_bytes(), 0u128.to_le_bytes());
            rep.tokens_zeroed += 1;
        }

        for k in &local_dex_keys {
            batch.delete_cf(&cf, k);
        }
        let dex_wallets: BTreeSet<&String> = snap.dex_debited.keys().chain(snap.dex_credited.keys()).collect();
        for w in &dex_wallets {
            let deb = snap.dex_debited.get(*w).copied().unwrap_or(0);
            let cre = snap.dex_credited.get(*w).copied().unwrap_or(0);
            if deb > 0 {
                batch.put_cf(&cf, format!("dex_qug_debited:{w}").as_bytes(), deb.to_le_bytes());
            }
            if cre > 0 {
                batch.put_cf(&cf, format!("dex_qug_credited:{w}").as_bytes(), cre.to_le_bytes());
            }
            let applied: i128 = cre as i128 - deb as i128;
            batch.put_cf(&cf, format!("dex_applied_net:{w}").as_bytes(), applied.to_le_bytes());
        }
        rep.dex_wallets = dex_wallets.len();
        rep.dex_keys_removed = local_dex_keys.len();

        let before = local_wallets.values().fold(0u128, |a, b| a.saturating_add(*b));
        rep.supply_before = before.to_string();
        rep.supply_after = snap.wallet_supply().to_string();

        batch.put_cf(&cf, crate::balance_consensus::CREDIT_FLOOR_KEY, snap.height.to_le_bytes());
        batch.put_cf(&cf, done_key.as_bytes(), serde_json::to_vec(&rep)?);

        // One synced batch at boot, before the network starts: all or nothing.
        let mut opts = rocksdb::WriteOptions::default();
        opts.set_sync(true);
        db.write_opt(batch, &opts).context("authority import batch failed — nothing written")?;

        crate::balance_consensus::set_credit_floor(snap.height);
        self.update_balance_smt_best_effort(&smt);
        Ok(rep)
    }
}
