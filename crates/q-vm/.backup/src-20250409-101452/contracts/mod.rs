use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::ser::SerializeTuple;
use serde::de::{self, Visitor};
use std::fmt;

// Custom serialization for [u8; 32] and [u8; 64]
#[derive(Clone, PartialEq)]
pub struct Bytes32(pub [u8; 32]);

#[derive(Clone, PartialEq)]
pub struct Bytes64(pub [u8; 64]);

impl Serialize for Bytes32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tup = serializer.serialize_tuple(32)?;
        for byte in &self.0[..] {
            tup.serialize_element(byte)?;
        }
        tup.end()
    }
}

impl<'de> Deserialize<'de> for Bytes32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Bytes32Visitor;
        
        impl<'de> Visitor<'de> for Bytes32Visitor {
            type Value = Bytes32;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a 32-byte array")
            }
            
            fn visit_seq<A>(self, mut seq: A) -> Result<Bytes32, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut bytes = [0u8; 32];
                for i in 0..32 {
                    bytes[i] = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(i, &self))?;
                }
                Ok(Bytes32(bytes))
            }
        }
        
        deserializer.deserialize_tuple(32, Bytes32Visitor)
    }
}

impl fmt::Debug for Bytes32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bytes32({:?})", &self.0[..])
    }
}

impl Serialize for Bytes64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tup = serializer.serialize_tuple(64)?;
        for byte in &self.0[..] {
            tup.serialize_element(byte)?;
        }
        tup.end()
    }
}

impl<'de> Deserialize<'de> for Bytes64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Bytes64Visitor;
        
        impl<'de> Visitor<'de> for Bytes64Visitor {
            type Value = Bytes64;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a 64-byte array")
            }
            
            fn visit_seq<A>(self, mut seq: A) -> Result<Bytes64, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut bytes = [0u8; 64];
                for i in 0..64 {
                    bytes[i] = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(i, &self))?;
                }
                Ok(Bytes64(bytes))
            }
        }
        
        deserializer.deserialize_tuple(64, Bytes64Visitor)
    }
}

impl fmt::Debug for Bytes64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bytes64({:?})", &self.0[..])
    }
}

/// AI model execution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIModelCall {
    pub model: String,
    pub input: Vec<u8>, // Changed from Vec to Vec<u8> for clarity
    pub gas_limit: u64,
    pub shard_count: u64,
    pub cache_policy: CachePolicy,
}

/// Cache policy for AI model executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachePolicy {
    NoCache,
    UseCache(u64),
    ForceRefresh,
}

/// Transaction types supported by DAGKnight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    Transfer {
        recipient: Bytes32,
        amount: u64,
    },
    ContractExecution {
        contract: Bytes32,
        method: String,
        args: Vec<u8>, // Changed from Vec to Vec<u8>
    },
    AIModelExecution(AIModelCall),
    RegisterModel(ModelRegistration),
}

/// Model registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistration {
    pub model_id: String,
    pub description: String,
    pub version: String,
    pub memory_required: u64,
    pub sharding_capability: ShardingCapability,
    pub resource_requirements: ResourceRequirements,
}

/// Model sharding capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingCapability {
    None,
    Horizontal,
    Vertical,
    Full,
}

/// Resource requirements for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_cpu_cores: u32,
    pub min_memory_mb: u64,
    pub gpu_memory_mb: Option<u64>, // Made Option more specific
    pub disk_space_mb: u64,
    pub avg_exec_time_per_token_ms: f64,
}

/// Transaction with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: Bytes32,
    pub tx_type: TransactionType,
    pub sender: Bytes32,
    pub nonce: u64,
    pub gas_price: u64,
    pub gas_limit: u64,
    pub timestamp: u64,
    pub signature: Bytes64,
}

// Basic implementation with error handling
impl Transaction {
    pub fn new(
        tx_type: TransactionType,
        sender: Bytes32,
        nonce: u64,
        gas_price: u64,
        gas_limit: u64,
        timestamp: u64,
        signature: Bytes64,
    ) -> Result<Self, &'static str> {
        if gas_limit == 0 {
            return Err("Gas limit cannot be zero");
        }
        Ok(Transaction {
            hash: Bytes32([0; 32]), // Should be calculated in real implementation
            tx_type,
            sender,
            nonce,
            gas_price,
            gas_limit,
            timestamp,
            signature,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_creation() {
        let sender = Bytes32([1; 32]);
        let signature = Bytes64([2; 64]);
        let tx = Transaction::new(
            TransactionType::Transfer {
                recipient: Bytes32([3; 32]),
                amount: 100,
            },
            sender,
            1,
            10,
            1000,
            1234567890,
            signature,
        );
        assert!(tx.is_ok());
    }

    #[test]
    fn test_zero_gas_limit() {
        let sender = Bytes32([1; 32]);
        let signature = Bytes64([2; 64]);
        let tx = Transaction::new(
            TransactionType::Transfer {
                recipient: Bytes32([3; 32]),
                amount: 100,
            },
            sender,
            1,
            10,
            0,
            1234567890,
            signature,
        );
        assert!(tx.is_err());
    }
}