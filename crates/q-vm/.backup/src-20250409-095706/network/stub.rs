// Temporary stub implementation until the p2p module is fixed
use std::sync::Arc;
use crate::vm::VmError;

pub struct Network {
    address: String,
    port: u16,
}

impl Network {
    pub fn new(address: String, port: u16) -> Result<Self, VmError> {
        Ok(Self { address, port })
    }
}
