use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use crate::vm::VmError;
use rocksdb::{DB, WriteBatch, Options};
use std::path::Path;

#[derive(Debug)]
// State Database interface
pub struct StateDB {
    db: Option<Arc<DB>>,
    in_memory: bool,
    cache: Arc<RwLock<HashMap<Vec<u8>, Vec<u8>>>>,
}

impl StateDB {
    pub fn new(path: &str) -> Self {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = DB::open(&opts, Path::new(path)).expect("Failed to open RocksDB");
        
        Self {
            db: Some(Arc::new(db)),
            in_memory: false,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn new_in_memory() -> Self {
        Self {
            db: None,
            in_memory: true,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(value) = cache.get(key) {
                return Some(value.clone());
            }
        }
        
        // If not in cache, check DB (if not in-memory)
        if !self.in_memory {
            if let Some(db) = &self.db {
                if let Ok(Some(value)) = db.get(key) {
                    // Update cache
                    let mut cache = self.cache.write();
                    cache.insert(key.to_vec(), value.clone());
                    return Some(value);
                }
            }
        }
        
        None
    }
    
    pub fn put(&self, key: Vec<u8>, value: Vec<u8>) {
        // Update cache
        {
            let mut cache = self.cache.write();
            cache.insert(key.clone(), value.clone());
        }
        
        // If not in-memory, update DB
        if !self.in_memory {
            if let Some(db) = &self.db {
                let _ = db.put(key, value);
            }
        }
    }
    
    pub fn delete(&self, key: &[u8]) {
        // Update cache
        {
            let mut cache = self.cache.write();
            cache.remove(key);
        }
        
        // If not in-memory, update DB
        if !self.in_memory {
            if let Some(db) = &self.db {
                let _ = db.delete(key);
            }
        }
    }
    
    pub fn commit_batch(&self, batch: StateBatch) -> Result<(), VmError> {
        // Apply changes to cache
        {
            let mut cache = self.cache.write();
            for (key, value_opt) in batch.changes.iter() {
                match value_opt {
                    Some(value) => cache.insert(key.clone(), value.clone()),
                    None => cache.remove(key),
                };
            }
        }
        
        // If not in-memory, apply changes to DB
        if !self.in_memory {
            if let Some(db) = &self.db {
                let mut wb = WriteBatch::default();
                
                for (key, value_opt) in batch.changes.iter() {
                    match value_opt {
                        Some(value) => wb.put(key, value),
                        None => wb.delete(key),
                    }
                }
                
                db.write(wb).map_err(|e| VmError::StorageError(e))?;
            }
        }
        
        Ok(())
    }
    
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }
}

// State batch for atomic changes
#[derive(Clone, Default)]
pub struct StateBatch {
    pub changes: HashMap<Vec<u8>, Option<Vec<u8>>>,
}

impl StateBatch {
    pub fn new() -> Self {
        Self {
            changes: HashMap::new(),
        }
    }
    
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.changes.insert(key, Some(value));
    }
    
    pub fn delete(&mut self, key: Vec<u8>) {
        self.changes.insert(key, None);
    }
}

// State root calculation
pub fn compute_state_root(state: &StateDB) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    
    // Get a sorted list of all keys
    let cache = state.cache.read();
    let mut keys: Vec<_> = cache.keys().collect();
    keys.sort();
    
    // Hash all key-value pairs
    for key in keys {
        if let Some(value) = cache.get(key) {
            hasher.update(key);
            hasher.update(value);
        }
    }
    
    let mut root = [0u8; 32];
    root.copy_from_slice(hasher.finalize().as_bytes());
    root
}

// State transition functions
pub struct StateTransition {
    initial_state: Arc<StateDB>,
    final_state: Option<Arc<StateDB>>,
    batch: StateBatch,
}

impl StateTransition {
    pub fn new(state: Arc<StateDB>) -> Self {
        Self {
            initial_state: state,
            final_state: None,
            batch: StateBatch::new(),
        }
    }
    
    pub fn apply(&mut self, key: Vec<u8>, value: Option<Vec<u8>>) {
        match value {
            Some(v) => self.batch.put(key, v),
            None => self.batch.delete(key),
        }
    }
    
    pub fn commit(&mut self) -> Result<Arc<StateDB>, VmError> {
        // Commit batch to initial state
        self.initial_state.commit_batch(self.batch.clone())?;
        
        // Store as final state
        self.final_state = Some(self.initial_state.clone());
        
        Ok(self.initial_state.clone())
    }
    
    pub fn rollback(&mut self) {
        // Clear batch
        self.batch = StateBatch::new();
    }
}
