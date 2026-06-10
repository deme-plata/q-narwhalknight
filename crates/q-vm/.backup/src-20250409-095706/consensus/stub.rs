use crate::vm::{ConsensusEngine, VmError};

// Stub implementation for testing
pub struct StubConsensus;

impl StubConsensus {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl ConsensusEngine for StubConsensus {
    async fn validate_contract(&self, _hash: [u8; 32], _bytecode: &[u8]) -> Result<(), VmError> {
        // Stub implementation that just succeeds
        Ok(())
    }
    
    async fn broadcast_contract(&self, _hash: [u8; 32], _bytecode: Vec<u8>) -> Result<(), VmError> {
        // Stub implementation that just succeeds
        Ok(())
    }
}
