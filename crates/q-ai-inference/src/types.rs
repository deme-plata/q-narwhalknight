use serde::{Deserialize, Serialize};
use std::time::Instant;

/// AI inference request from user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub model: String, // "mistral-7b-instruct-v0.3"
    #[serde(skip)]
    pub timestamp: Option<Instant>,

    /// v6.0.0: Deterministic seed for reproducible inference (opML verification).
    /// When set, forces greedy decoding (temperature=0, top_k=1) so that
    /// two nodes with the same model produce identical output for the same input.
    #[serde(default)]
    pub deterministic_seed: Option<u64>,

    /// v6.0.0: Required model hash (SHA3-256 of GGUF file) for integrity verification.
    /// Workers must serve this exact model or the request is rejected.
    #[serde(default)]
    pub required_model_hash: Option<[u8; 32]>,

    /// v6.0.0: Maximum price per token the user is willing to pay (in QUG base units, 24-decimal).
    #[serde(default)]
    pub max_price_per_token: Option<u128>,
}

/// Response from distributed inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub generated_text: String,
    pub tokens_generated: usize,
    pub latency_ms: u64,
    pub nodes_participated: Vec<String>,
}

/// Layer assignment for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAssignment {
    pub node_id: String,
    pub peer_id: String,
    pub layer_start: usize, // e.g., 0
    pub layer_end: usize,   // e.g., 10 (layers 0-10 = 11 layers)
    pub device_capability: DeviceCapability,
    pub last_seen: i64, // Unix timestamp
}

/// Device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceCapability {
    CPU {
        cores: usize,
        ram_gb: usize,
    },
    CUDA {
        vram_gb: usize,
        compute_capability: String,
    },
    Metal {
        vram_gb: usize,
    },
}

impl DeviceCapability {
    /// Get a score for prioritizing nodes (higher is better)
    pub fn score(&self) -> u64 {
        match self {
            DeviceCapability::CPU { cores, ram_gb } => {
                (*cores as u64) * 10 + (*ram_gb as u64)
            }
            DeviceCapability::CUDA { vram_gb, .. } => {
                (*vram_gb as u64) * 1000 // Heavily prioritize CUDA
            }
            DeviceCapability::Metal { vram_gb } => {
                (*vram_gb as u64) * 800 // Prioritize Metal but less than CUDA
            }
        }
    }

    /// Estimate how many layers this device can handle
    pub fn estimate_layer_capacity(&self) -> usize {
        match self {
            DeviceCapability::CPU { cores: _, ram_gb } => {
                // Conservative estimate: ~1 layer per 4GB RAM, max 8 layers
                (*ram_gb / 4).min(8).max(1)
            }
            DeviceCapability::CUDA { vram_gb, .. } => {
                // ~1 layer per 1GB VRAM for Q4 quantization
                (*vram_gb).min(32).max(2)
            }
            DeviceCapability::Metal { vram_gb } => {
                // Similar to CUDA
                (*vram_gb).min(32).max(2)
            }
        }
    }
}

/// Tensor data for layer communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub request_id: String,
    pub layer_index: usize,
    pub data: Vec<u8>, // Compressed serialized tensor
    pub shape: Vec<usize>,
    pub is_compressed: bool,
}

impl TensorData {
    /// Create a new tensor data with compression
    pub fn new(request_id: String, layer_index: usize, data: Vec<f32>, shape: Vec<usize>) -> Self {
        let compressed = compress_tensor(&data);
        Self {
            request_id,
            layer_index,
            data: compressed,
            shape,
            is_compressed: true,
        }
    }

    /// Decompress and convert to f32 vector
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if self.is_compressed {
            decompress_tensor(&self.data)
        } else {
            // Convert bytes directly to f32
            Ok(self
                .data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
    }
}

/// Messages for Gossipsub pub/sub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AIMessage {
    /// Request for distributed inference
    InferenceRequest(InferenceRequest),

    /// Final response from inference
    InferenceResponse(InferenceResponse),

    /// Output from a specific layer (sent to next layer)
    LayerOutput(TensorData),

    /// Node announcing its capabilities
    NodeCapability(LayerAssignment),

    /// Coordinator election message
    CoordinatorElection { node_id: String, score: u64 },

    /// Heartbeat from active nodes
    Heartbeat { node_id: String, timestamp: i64 },

    /// v5.1.0: Node announcing its RPC worker endpoint for pipeline parallelism
    /// When a node starts a llama.cpp rpc-server, it broadcasts this so the
    /// coordinator can build `--rpc worker1:port,worker2:port` for distributed inference.
    RpcWorkerAvailable(crate::rpc_worker::RpcWorkerInfo),

    /// v5.1.0: Node announcing its RPC worker has stopped
    RpcWorkerStopped { peer_id: String },
}

/// Compress tensor data using gzip
pub fn compress_tensor(data: &[f32]) -> Vec<u8> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(&bytes).expect("Compression failed");
    encoder.finish().expect("Compression finish failed")
}

/// Decompress tensor data from gzip
pub fn decompress_tensor(compressed: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut decoder = GzDecoder::new(compressed);
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;

    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(floats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_compression() {
        let original_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![1, 5];

        let tensor = TensorData::new("test-req".to_string(), 0, original_data.clone(), shape);

        assert!(tensor.is_compressed);

        let decompressed = tensor.to_f32_vec().unwrap();
        assert_eq!(decompressed, original_data);
    }

    #[test]
    fn test_device_capability_score() {
        let cpu = DeviceCapability::CPU {
            cores: 8,
            ram_gb: 16,
        };
        let cuda = DeviceCapability::CUDA {
            vram_gb: 12,
            compute_capability: "8.0".to_string(),
        };

        assert!(cuda.score() > cpu.score());
    }

    #[test]
    fn test_layer_capacity_estimation() {
        let cpu = DeviceCapability::CPU {
            cores: 8,
            ram_gb: 16,
        };
        let cuda = DeviceCapability::CUDA {
            vram_gb: 12,
            compute_capability: "8.0".to_string(),
        };

        assert_eq!(cpu.estimate_layer_capacity(), 4); // 16/4 = 4
        assert_eq!(cuda.estimate_layer_capacity(), 12); // 12GB VRAM
    }
}
