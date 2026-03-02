/// Dune sync cursor persistence via RocksDB.
/// Tracks what has been pushed so sync can resume after restarts.

use anyhow::Result;
use q_storage::kv::KVStore;
use q_storage::StorageEngine;
use std::sync::Arc;

const CF_MANIFEST: &str = "manifest";
const KEY_LAST_BLOCK_HEIGHT: &[u8] = b"dune:last_pushed_height";
const KEY_LAST_DAILY_DATE: &[u8] = b"dune:last_daily_date";
const KEY_LAST_SUPPLY_TS: &[u8] = b"dune:last_supply_timestamp";
const KEY_LAST_HOLDERS_DATE: &[u8] = b"dune:last_holders_date";
const KEY_LAST_NETWORK_TS: &[u8] = b"dune:last_network_stats_ts";
const KEY_EMISSION_PUSHED: &[u8] = b"dune:emission_schedule_pushed";
const KEY_LAST_MINER_ECON_DATE: &[u8] = b"dune:last_miner_economics_date";
const KEY_LAST_WEALTH_DATE: &[u8] = b"dune:last_wealth_distribution_date";
const KEY_LAST_BLOCK_TIME_DATE: &[u8] = b"dune:last_block_time_analysis_date";

#[derive(Clone)]
pub struct SyncState {
    kv: Arc<dyn KVStore>,
}

impl SyncState {
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        Self { kv: storage.get_kv() }
    }

    pub async fn get_last_pushed_height(&self) -> Result<u64> {
        match self.kv.get(CF_MANIFEST, KEY_LAST_BLOCK_HEIGHT).await? {
            Some(bytes) if bytes.len() == 8 => {
                Ok(u64::from_be_bytes(bytes[..8].try_into().unwrap()))
            }
            _ => Ok(0),
        }
    }

    pub async fn set_last_pushed_height(&self, height: u64) -> Result<()> {
        self.kv.put_sync(CF_MANIFEST, KEY_LAST_BLOCK_HEIGHT, &height.to_be_bytes()).await?;
        Ok(())
    }

    pub async fn get_last_daily_date(&self) -> Result<Option<String>> {
        match self.kv.get(CF_MANIFEST, KEY_LAST_DAILY_DATE).await? {
            Some(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).to_string())),
            None => Ok(None),
        }
    }

    pub async fn set_last_daily_date(&self, date: &str) -> Result<()> {
        self.kv.put_sync(CF_MANIFEST, KEY_LAST_DAILY_DATE, date.as_bytes()).await?;
        Ok(())
    }

    pub async fn get_last_supply_timestamp(&self) -> Result<u64> {
        match self.kv.get(CF_MANIFEST, KEY_LAST_SUPPLY_TS).await? {
            Some(bytes) if bytes.len() == 8 => {
                Ok(u64::from_be_bytes(bytes[..8].try_into().unwrap()))
            }
            _ => Ok(0),
        }
    }

    pub async fn set_last_supply_timestamp(&self, ts: u64) -> Result<()> {
        self.kv.put_sync(CF_MANIFEST, KEY_LAST_SUPPLY_TS, &ts.to_be_bytes()).await?;
        Ok(())
    }

    pub async fn get_last_holders_date(&self) -> Result<Option<String>> {
        match self.kv.get(CF_MANIFEST, KEY_LAST_HOLDERS_DATE).await? {
            Some(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).to_string())),
            None => Ok(None),
        }
    }

    pub async fn set_last_holders_date(&self, date: &str) -> Result<()> {
        self.kv.put_sync(CF_MANIFEST, KEY_LAST_HOLDERS_DATE, date.as_bytes()).await?;
        Ok(())
    }

    pub async fn get_last_network_stats_ts(&self) -> Result<u64> {
        match self.kv.get(CF_MANIFEST, KEY_LAST_NETWORK_TS).await? {
            Some(bytes) if bytes.len() == 8 => {
                Ok(u64::from_be_bytes(bytes[..8].try_into().unwrap()))
            }
            _ => Ok(0),
        }
    }

    pub async fn set_last_network_stats_ts(&self, ts: u64) -> Result<()> {
        self.kv.put_sync(CF_MANIFEST, KEY_LAST_NETWORK_TS, &ts.to_be_bytes()).await?;
        Ok(())
    }

    pub async fn is_emission_schedule_pushed(&self) -> Result<bool> {
        match self.kv.get(CF_MANIFEST, KEY_EMISSION_PUSHED).await? {
            Some(bytes) => Ok(!bytes.is_empty() && bytes[0] == 1),
            None => Ok(false),
        }
    }

    pub async fn set_emission_schedule_pushed(&self) -> Result<()> {
        self.kv.put_sync(CF_MANIFEST, KEY_EMISSION_PUSHED, &[1]).await?;
        Ok(())
    }

    pub async fn get_last_miner_economics_date(&self) -> Result<Option<String>> {
        match self.kv.get(CF_MANIFEST, KEY_LAST_MINER_ECON_DATE).await? {
            Some(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).to_string())),
            None => Ok(None),
        }
    }

    pub async fn set_last_miner_economics_date(&self, date: &str) -> Result<()> {
        self.kv.put_sync(CF_MANIFEST, KEY_LAST_MINER_ECON_DATE, date.as_bytes()).await?;
        Ok(())
    }

    pub async fn get_last_wealth_distribution_date(&self) -> Result<Option<String>> {
        match self.kv.get(CF_MANIFEST, KEY_LAST_WEALTH_DATE).await? {
            Some(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).to_string())),
            None => Ok(None),
        }
    }

    pub async fn set_last_wealth_distribution_date(&self, date: &str) -> Result<()> {
        self.kv.put_sync(CF_MANIFEST, KEY_LAST_WEALTH_DATE, date.as_bytes()).await?;
        Ok(())
    }

    pub async fn get_last_block_time_analysis_date(&self) -> Result<Option<String>> {
        match self.kv.get(CF_MANIFEST, KEY_LAST_BLOCK_TIME_DATE).await? {
            Some(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).to_string())),
            None => Ok(None),
        }
    }

    pub async fn set_last_block_time_analysis_date(&self, date: &str) -> Result<()> {
        self.kv.put_sync(CF_MANIFEST, KEY_LAST_BLOCK_TIME_DATE, date.as_bytes()).await?;
        Ok(())
    }
}
