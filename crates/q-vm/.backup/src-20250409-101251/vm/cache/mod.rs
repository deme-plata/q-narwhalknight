use std::fmt;

// Simplified cache implementation for compilation
pub struct ContractCache;

impl fmt::Debug for ContractCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContractCache").finish()
    }
}

impl ContractCache {
    pub fn new() -> Self {
        Self {}
    }
}
