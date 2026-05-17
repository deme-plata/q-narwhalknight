use std::collections::HashMap;
use parking_lot::RwLock;

#[derive(Debug)]
pub struct ContractCache {
    contracts: RwLock<HashMap<String, Vec<u8>>>,
}

impl ContractCache {
    pub fn new() -> Self {
        Self {
            contracts: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.contracts.read().get(key).cloned()
    }

    pub fn insert(&self, key: String, value: Vec<u8>) {
        self.contracts.write().insert(key, value);
    }
}
