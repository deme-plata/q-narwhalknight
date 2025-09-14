use super::Transaction;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize)]
struct TransactionSerde {
    pub hash: [u8; 32],
    pub data: Vec<u8>,
    pub sender: [u8; 32],
    pub nonce: u64,
    pub signature: Vec<u8>,
    pub timestamp: u64,
}

impl From<&Transaction> for TransactionSerde {
    fn from(tx: &Transaction) -> Self {
        Self {
            hash: tx.hash,
            data: tx.data.clone(),
            sender: tx.sender,
            nonce: tx.nonce,
            signature: tx.signature.to_vec(),
            timestamp: tx.timestamp,
        }
    }
}

impl From<TransactionSerde> for Transaction {
    fn from(tx: TransactionSerde) -> Self {
        let mut signature = [0u8; 64];
        if tx.signature.len() >= 64 {
            signature.copy_from_slice(&tx.signature[0..64]);
        }

        Self {
            hash: tx.hash,
            data: tx.data,
            sender: tx.sender,
            nonce: tx.nonce,
            signature,
            timestamp: tx.timestamp,
        }
    }
}

impl Serialize for Transaction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let serde_tx = TransactionSerde::from(self);
        serde_tx.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Transaction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serde_tx = TransactionSerde::deserialize(deserializer)?;
        Ok(Transaction::from(serde_tx))
    }
}
