use crate::{ChunkMetadata, Result, SnapshotMetadata};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    pub cid: String,
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageManifest {
    pub version: String,
    pub snapshot: SnapshotMetadata,
    pub chunks: Vec<ChunkInfo>,
    pub total_chunks: usize,
    pub manifest_cid: Option<String>,
}

impl StorageManifest {
    pub fn new(snapshot: SnapshotMetadata) -> Self {
        Self {
            version: "1.0.0".to_string(),
            snapshot,
            chunks: Vec::new(),
            total_chunks: 0,
            manifest_cid: None,
        }
    }

    pub fn add_chunk(&mut self, cid: String, metadata: ChunkMetadata) {
        self.chunks.push(ChunkInfo { cid, metadata });
        self.total_chunks = self.chunks.len();
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}
