/// v2.9.2-beta: Contract Event Persistence for Q-VM
///
/// This module provides RocksDB persistence for contract events,
/// enabling event querying, subscription, and historical analysis.
///
/// Key format: `{contract_address}:{block_height:08x}:{event_index:04x}`
/// Value: JSON-serialized ContractEvent

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::{CF_CONTRACT_EVENTS, KVStore};

/// A contract event emitted during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractEvent {
    /// Contract that emitted the event
    pub contract_address: String,
    /// Block height when event was emitted
    pub block_height: u64,
    /// Index within the block (for ordering)
    pub event_index: u16,
    /// Event topic (e.g., "Transfer", "Approval", "Swap")
    pub topic: String,
    /// Event data (arbitrary bytes, usually ABI-encoded)
    pub data: Vec<u8>,
    /// Transaction hash that triggered the event
    pub tx_hash: String,
    /// Timestamp when event was emitted
    pub timestamp: u64,
    /// Additional indexed parameters for filtering
    pub indexed_params: Vec<IndexedParam>,
}

/// Indexed parameter for efficient event filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedParam {
    pub name: String,
    pub value: Vec<u8>,
    pub param_type: String,
}

impl ContractEvent {
    /// Generate storage key for this event
    pub fn storage_key(&self) -> Vec<u8> {
        format!(
            "{}:{:08x}:{:04x}",
            self.contract_address,
            self.block_height,
            self.event_index
        ).into_bytes()
    }
}

/// Contract event storage manager
pub struct ContractEventStorage<S: KVStore> {
    db: Arc<S>,
}

impl<S: KVStore> ContractEventStorage<S> {
    /// Create new event storage
    pub fn new(db: Arc<S>) -> Self {
        Self { db }
    }

    /// Persist a contract event to RocksDB
    pub async fn store_event(&self, event: &ContractEvent) -> Result<()> {
        let key = event.storage_key();
        let value = serde_json::to_vec(event)?;

        self.db.put(CF_CONTRACT_EVENTS, &key, &value).await?;

        debug!(
            "📝 Stored contract event: topic={} contract={} height={}",
            event.topic,
            &event.contract_address[..16.min(event.contract_address.len())],
            event.block_height
        );

        Ok(())
    }

    /// Store multiple events in a batch (more efficient)
    pub async fn store_events_batch(&self, events: &[ContractEvent]) -> Result<usize> {
        if events.is_empty() {
            return Ok(0);
        }

        let batch: Vec<(&str, Vec<u8>, Vec<u8>)> = events
            .iter()
            .map(|e| {
                let key = e.storage_key();
                let value = serde_json::to_vec(e).unwrap_or_default();
                (CF_CONTRACT_EVENTS, key, value)
            })
            .collect();

        self.db.write_batch(batch).await?;

        info!(
            "📝 Stored {} contract events in batch",
            events.len()
        );

        Ok(events.len())
    }

    /// Retrieve a specific event by key
    pub async fn get_event(&self, key: &[u8]) -> Result<Option<ContractEvent>> {
        match self.db.get(CF_CONTRACT_EVENTS, key).await? {
            Some(bytes) => {
                let event: ContractEvent = serde_json::from_slice(&bytes)?;
                Ok(Some(event))
            }
            None => Ok(None),
        }
    }

    /// Get all events for a contract
    pub async fn get_events_for_contract(
        &self,
        contract_address: &str,
    ) -> Result<Vec<ContractEvent>> {
        let prefix = format!("{}:", contract_address);
        let entries = self.db.scan_prefix(CF_CONTRACT_EVENTS, prefix.as_bytes()).await?;

        let mut events = Vec::with_capacity(entries.len());
        for (_, value) in entries {
            if let Ok(event) = serde_json::from_slice::<ContractEvent>(&value) {
                events.push(event);
            }
        }

        debug!(
            "📖 Retrieved {} events for contract {}",
            events.len(),
            &contract_address[..16.min(contract_address.len())]
        );

        Ok(events)
    }

    /// Get events for a contract in a specific block range
    pub async fn get_events_in_range(
        &self,
        contract_address: &str,
        start_height: u64,
        end_height: u64,
    ) -> Result<Vec<ContractEvent>> {
        let prefix = format!("{}:", contract_address);
        let entries = self.db.scan_prefix(CF_CONTRACT_EVENTS, prefix.as_bytes()).await?;

        let mut events = Vec::new();
        for (_, value) in entries {
            if let Ok(event) = serde_json::from_slice::<ContractEvent>(&value) {
                if event.block_height >= start_height && event.block_height <= end_height {
                    events.push(event);
                }
            }
        }

        // Sort by height and index
        events.sort_by(|a, b| {
            a.block_height
                .cmp(&b.block_height)
                .then(a.event_index.cmp(&b.event_index))
        });

        debug!(
            "📖 Retrieved {} events in range [{}, {}] for {}",
            events.len(),
            start_height,
            end_height,
            &contract_address[..16.min(contract_address.len())]
        );

        Ok(events)
    }

    /// Get events by topic across all contracts
    pub async fn get_events_by_topic(&self, topic: &str) -> Result<Vec<ContractEvent>> {
        // This requires a full scan - in production, we'd add a secondary index
        let entries = self.db.scan_all(CF_CONTRACT_EVENTS).await?;

        let mut events = Vec::new();
        for (_, value) in entries {
            if let Ok(event) = serde_json::from_slice::<ContractEvent>(&value) {
                if event.topic == topic {
                    events.push(event);
                }
            }
        }

        debug!("📖 Retrieved {} events with topic '{}'", events.len(), topic);

        Ok(events)
    }

    /// Get latest events across all contracts (for dashboard)
    pub async fn get_latest_events(&self, limit: usize) -> Result<Vec<ContractEvent>> {
        let entries = self.db.scan_all(CF_CONTRACT_EVENTS).await?;

        let mut events: Vec<ContractEvent> = entries
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_slice(&value).ok())
            .collect();

        // Sort by block height descending, then event index
        events.sort_by(|a, b| {
            b.block_height
                .cmp(&a.block_height)
                .then(b.event_index.cmp(&a.event_index))
        });

        events.truncate(limit);

        Ok(events)
    }

    /// Count events for a contract
    pub async fn count_events(&self, contract_address: &str) -> Result<u64> {
        let prefix = format!("{}:", contract_address);
        let entries = self.db.scan_prefix(CF_CONTRACT_EVENTS, prefix.as_bytes()).await?;
        Ok(entries.len() as u64)
    }

    /// Delete events for a contract (for cleanup)
    pub async fn delete_contract_events(&self, contract_address: &str) -> Result<u64> {
        let prefix = format!("{}:", contract_address);
        let entries = self.db.scan_prefix(CF_CONTRACT_EVENTS, prefix.as_bytes()).await?;

        let count = entries.len() as u64;

        for (key, _) in entries {
            self.db.delete(CF_CONTRACT_EVENTS, &key).await?;
        }

        info!("🗑️ Deleted {} events for contract {}", count, contract_address);

        Ok(count)
    }

    /// Get event statistics
    pub async fn get_stats(&self) -> Result<ContractEventStats> {
        let entries = self.db.scan_all(CF_CONTRACT_EVENTS).await?;

        let mut stats = ContractEventStats::default();
        let mut contracts = std::collections::HashSet::new();
        let mut topics = std::collections::HashMap::new();

        for (_, value) in entries {
            if let Ok(event) = serde_json::from_slice::<ContractEvent>(&value) {
                stats.total_events += 1;
                contracts.insert(event.contract_address.clone());
                *topics.entry(event.topic.clone()).or_insert(0u64) += 1;

                if event.block_height > stats.latest_block_height {
                    stats.latest_block_height = event.block_height;
                }
            }
        }

        stats.unique_contracts = contracts.len() as u64;
        stats.unique_topics = topics.len() as u64;

        Ok(stats)
    }
}

/// Statistics for contract events
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContractEventStats {
    pub total_events: u64,
    pub unique_contracts: u64,
    pub unique_topics: u64,
    pub latest_block_height: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_storage_key() {
        let event = ContractEvent {
            contract_address: "0xabcdef".to_string(),
            block_height: 1000,
            event_index: 5,
            topic: "Transfer".to_string(),
            data: vec![1, 2, 3],
            tx_hash: "0x123".to_string(),
            timestamp: 1234567890,
            indexed_params: vec![],
        };

        let key = String::from_utf8(event.storage_key()).unwrap();
        assert_eq!(key, "0xabcdef:000003e8:0005");
    }

    #[test]
    fn test_event_serialization() {
        let event = ContractEvent {
            contract_address: "0xabcdef".to_string(),
            block_height: 1000,
            event_index: 5,
            topic: "Transfer".to_string(),
            data: vec![1, 2, 3],
            tx_hash: "0x123".to_string(),
            timestamp: 1234567890,
            indexed_params: vec![IndexedParam {
                name: "from".to_string(),
                value: vec![1, 2, 3, 4],
                param_type: "address".to_string(),
            }],
        };

        let json = serde_json::to_string(&event).unwrap();
        let parsed: ContractEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.topic, "Transfer");
        assert_eq!(parsed.block_height, 1000);
    }
}
