// Temporary stub implementation until the p2p module is fixed
use crate::vm::VmError;

pub struct Network {
    _address: String,
    _port: u16,
}

impl Network {
    pub fn new(address: String, port: u16) -> Result<Self, VmError> {
        Ok(Self {
            _address: address,
            _port: port,
        })
    }
}
