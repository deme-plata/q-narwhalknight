use std::collections::{BTreeMap, HashMap, HashSet};
use std::cmp::Ordering;
use std::sync::Arc;
use tokio::sync::RwLock;
use parking_lot::Mutex;
use priority_queue::PriorityQueue;
use crate::transaction::Transaction;

// Transaction with priority info
#[derive(Debug, Clone)]
struct PrioritizedTransaction {
    tx: Transaction,
    gas_price: u64,
    time_added: u64,
}

impl Eq for PrioritizedTransaction {}

impl PartialEq for PrioritizedTransaction {
    fn eq(&self, other: &Self) -> bool {
        self.tx.hash == other.tx.hash
    }
}

// Priority value for organizing transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Priority(u64);

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Mempool for managing pending transactions
#[derive(Debug)]
pub struct Mempool {
    // Priority queue for transactions
    txs: Arc<Mutex<PriorityQueue<[u8; 32], Priority>>>,
    
    // Map of transaction hash to transaction data
    tx_map: Arc<Mutex<HashMap<[u8; 32], PrioritizedTransaction>>>,
    
    // Track transactions by sender for nonce ordering
    sender_txs: Arc<Mutex<HashMap<[u8; 32], HashMap<u64, [u8; 32]>>>>,
    
    // Configuration
    max_size: usize,
    min_gas_price: u64,
    
    // Metrics
    added_count: Arc<Mutex<u64>>,
    removed_count: Arc<Mutex<u64>>,
}

impl Mempool {
    pub fn new(max_size: usize, min_gas_price: u64) -> Self {
        Self {
            txs: Arc::new(Mutex::new(PriorityQueue::new())),
            tx_map: Arc::new(Mutex::new(HashMap::with_capacity(max_size))),
            sender_txs: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            min_gas_price,
            added_count: Arc::new(Mutex::new(0)),
            removed_count: Arc::new(Mutex::new(0)),
        }
    }
    
    // Add transaction to mempool
    pub fn add_transaction(&self, tx: Transaction, gas_price: u64) -> Result<(), String> {
        // Check if gas price meets minimum
        if gas_price < self.min_gas_price {
            return Err(format!("Gas price too low: {}, minimum: {}", gas_price, self.min_gas_price));
        }
        
        // Check if mempool is full
        {
            let txs = self.txs.lock();
            if txs.len() >= self.max_size {
                return Err("Mempool is full".to_string());
            }
        }
        
        // Check if transaction already exists
        {
            let tx_map = self.tx_map.lock();
            if tx_map.contains_key(&tx.hash) {
                return Err("Transaction already in mempool".to_string());
            }
        }
        
        // Calculate priority
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        let priority = gas_price * 1000 + (1000000000 - timestamp); // Higher gas price + earlier time = higher priority
        
        // Create prioritized transaction
        let ptx = PrioritizedTransaction {
            tx: tx.clone(),
            gas_price,
            time_added: timestamp,
        };
        
        // Add to data structures
        {
            let mut txs = self.txs.lock();
            let mut tx_map = self.tx_map.lock();
            let mut sender_txs = self.sender_txs.lock();
            
            // Add to priority queue
            txs.push(tx.hash, Priority(priority));
            
            // Add to transaction map
            tx_map.insert(tx.hash, ptx);
            
            // Add to sender transactions
            let sender_map = sender_txs.entry(tx.sender).or_insert_with(HashMap::new);
            sender_map.insert(tx.nonce, tx.hash);
            
            // Update metrics
            let mut added_count = self.added_count.lock();
            *added_count += 1;
        }
        
        Ok(())
    }
    
    // Get best transactions for block proposal
    pub fn get_best_transactions(&self, limit: usize) -> Vec<Transaction> {
        let mut result = Vec::with_capacity(limit);
        let mut visited = HashSet::new();
        
        // Clone priority queue to avoid deadlock
        let queue_clone = {
            let txs = self.txs.lock();
            txs.clone()
        };
        
        let tx_map = self.tx_map.lock();
        
        // Get transactions in order of priority
        for (hash, _) in queue_clone.into_sorted_iter() {
            if visited.contains(&hash) {
                continue;
            }
            
            if let Some(ptx) = tx_map.get(&hash) {
                result.push(ptx.tx.clone());
                visited.insert(hash);
                
                if result.len() >= limit {
                    break;
                }
            }
        }
        
        result
    }
    
    // Remove transaction from mempool
    pub fn remove_transaction(&self, tx_hash: &[u8; 32]) -> Option<Transaction> {
        let mut txs = self.txs.lock();
        let mut tx_map = self.tx_map.lock();
        let mut sender_txs = self.sender_txs.lock();
        
        // Remove from transaction map
        let ptx = tx_map.remove(tx_hash)?;
        
        // Remove from priority queue
        txs.remove(tx_hash);
        
        // Remove from sender transactions
        if let Some(sender_map) = sender_txs.get_mut(&ptx.tx.sender) {
            sender_map.remove(&ptx.tx.nonce);
            if sender_map.is_empty() {
                sender_txs.remove(&ptx.tx.sender);
            }
        }
        
        // Update metrics
        let mut removed_count = self.removed_count.lock();
        *removed_count += 1;
        
        Some(ptx.tx)
    }
    
    // Get transactions by sender with nonce ordering
    pub fn get_sender_transactions(&self, sender: &[u8; 32]) -> Vec<Transaction> {
        let sender_txs = self.sender_txs.lock();
        let tx_map = self.tx_map.lock();
        
        let mut result = Vec::new();
        
        if let Some(nonce_map) = sender_txs.get(sender) {
            // Get all nonces
            let mut nonces: Vec<_> = nonce_map.keys().collect();
            nonces.sort();
            
            // Get transactions in nonce order
            for nonce in nonces {
                if let Some(tx_hash) = nonce_map.get(nonce) {
                    if let Some(ptx) = tx_map.get(tx_hash) {
                        result.push(ptx.tx.clone());
                    }
                }
            }
        }
        
        result
    }
    
    // Get all transactions in mempool
    pub fn get_all_transactions(&self) -> Vec<Transaction> {
        let tx_map = self.tx_map.lock();
        tx_map.values().map(|ptx| ptx.tx.clone()).collect()
    }
    
    // Get transaction count
    pub fn get_transaction_count(&self) -> usize {
        let txs = self.txs.lock();
        txs.len()
    }
    
    // Update minimum gas price based on demand
    pub fn update_min_gas_price(&mut self, new_min: u64) {
        self.min_gas_price = new_min;
    }
    
    // Get metrics
    pub fn get_metrics(&self) -> (u64, u64, usize) {
        let added = *self.added_count.lock();
        let removed = *self.removed_count.lock();
        let current = self.get_transaction_count();
        
        (added, removed, current)
    }
}
