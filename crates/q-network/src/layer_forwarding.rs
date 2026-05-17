/// Layer Output Forwarding - Forwards intermediate tensors between nodes during distributed inference
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Tensor data with shape information and KV-cache for autoregressive generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,

    /// Key cache for attention layers [num_layers, batch, num_heads, seq_len, head_dim]
    /// Enables incremental token generation without re-computing past tokens
    pub key_cache: Option<Vec<f32>>,

    /// Value cache for attention layers [num_layers, batch, num_heads, seq_len, head_dim]
    /// Paired with key_cache for efficient autoregressive generation
    pub value_cache: Option<Vec<f32>>,

    /// Shape of KV-cache tensors (if present)
    pub kv_cache_shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorDType {
    Float32,
    Float16,
    BFloat16,
}

/// Layer output manager for distributed inference
pub struct LayerOutputManager {
    /// Pending layer outputs waiting to be forwarded
    pending_outputs: Arc<RwLock<HashMap<String, LayerOutput>>>,

    /// Received layer inputs from previous nodes
    received_inputs: Arc<RwLock<HashMap<String, TensorData>>>,

    /// Compression settings
    compression_enabled: bool,
    compression_level: i32,
}

#[derive(Debug, Clone)]
pub struct LayerOutput {
    pub request_id: String,
    pub layer_index: usize,
    pub tensor: TensorData,
    pub timestamp: i64,
    pub next_node_id: Option<String>,
}

impl LayerOutputManager {
    /// Create new layer output manager
    pub fn new(compression_enabled: bool) -> Self {
        Self {
            pending_outputs: Arc::new(RwLock::new(HashMap::new())),
            received_inputs: Arc::new(RwLock::new(HashMap::new())),
            compression_enabled,
            compression_level: 3, // zstd compression level (0-22, 3 is balanced)
        }
    }

    /// Store layer output for forwarding
    pub async fn store_layer_output(
        &self,
        request_id: String,
        layer_index: usize,
        tensor: TensorData,
        next_node_id: Option<String>,
    ) -> Result<()> {
        let output = LayerOutput {
            request_id: request_id.clone(),
            layer_index,
            tensor,
            timestamp: chrono::Utc::now().timestamp(),
            next_node_id,
        };

        let key = format!("{}:{}", request_id, layer_index);
        self.pending_outputs.write().await.insert(key, output);

        debug!("📦 Stored layer {} output for request {}", layer_index, request_id);
        Ok(())
    }

    /// Get pending layer output for forwarding
    pub async fn get_pending_output(&self, request_id: &str, layer_index: usize) -> Option<LayerOutput> {
        let key = format!("{}:{}", request_id, layer_index);
        self.pending_outputs.read().await.get(&key).cloned()
    }

    /// Remove pending output after successful forwarding
    pub async fn remove_pending_output(&self, request_id: &str, layer_index: usize) -> Result<()> {
        let key = format!("{}:{}", request_id, layer_index);
        self.pending_outputs.write().await.remove(&key);
        Ok(())
    }

    /// Store received layer input from previous node
    pub async fn store_received_input(
        &self,
        request_id: String,
        layer_index: usize,
        tensor: TensorData,
    ) -> Result<()> {
        let key = format!("{}:{}", request_id, layer_index);
        self.received_inputs.write().await.insert(key, tensor);

        info!("📥 Received layer {} input for request {}", layer_index, request_id);
        Ok(())
    }

    /// Get received layer input
    pub async fn get_received_input(&self, request_id: &str, layer_index: usize) -> Option<TensorData> {
        let key = format!("{}:{}", request_id, layer_index);
        self.received_inputs.read().await.get(&key).cloned()
    }

    /// Wait for layer input with timeout
    pub async fn wait_for_layer_input(
        &self,
        request_id: &str,
        layer_index: usize,
        timeout_secs: u64,
    ) -> Result<TensorData> {
        let key = format!("{}:{}", request_id, layer_index);
        let start = std::time::Instant::now();

        loop {
            if let Some(tensor) = self.received_inputs.read().await.get(&key) {
                return Ok(tensor.clone());
            }

            if start.elapsed().as_secs() >= timeout_secs {
                return Err(anyhow!(
                    "Timeout waiting for layer {} input (request {})",
                    layer_index,
                    request_id
                ));
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }

    /// Compress tensor data using zstd
    pub fn compress_tensor(&self, tensor: &TensorData) -> Result<Vec<u8>> {
        if !self.compression_enabled {
            return Ok(bincode::serialize(tensor)?);
        }

        // Serialize tensor to bytes
        let serialized = bincode::serialize(tensor)?;

        // Compress with zstd
        let compressed = zstd::encode_all(&serialized[..], self.compression_level)?;

        let ratio = serialized.len() as f64 / compressed.len() as f64;
        debug!(
            "🗜️ Compressed tensor: {} bytes → {} bytes ({:.2}× reduction)",
            serialized.len(),
            compressed.len(),
            ratio
        );

        Ok(compressed)
    }

    /// Decompress tensor data
    pub fn decompress_tensor(&self, compressed: &[u8]) -> Result<TensorData> {
        if !self.compression_enabled {
            return Ok(bincode::deserialize(compressed)?);
        }

        // Decompress with zstd
        let decompressed = zstd::decode_all(compressed)?;

        // Deserialize tensor
        let tensor: TensorData = bincode::deserialize(&decompressed)?;

        Ok(tensor)
    }

    /// Validate tensor shape matches expected
    pub fn validate_tensor_shape(&self, tensor: &TensorData, expected_shape: &[usize]) -> Result<()> {
        if tensor.shape != expected_shape {
            return Err(anyhow!(
                "Tensor shape mismatch: expected {:?}, got {:?}",
                expected_shape,
                tensor.shape
            ));
        }
        Ok(())
    }

    /// Calculate tensor size in bytes
    pub fn tensor_size_bytes(&self, tensor: &TensorData) -> usize {
        let element_size = match tensor.dtype {
            TensorDType::Float32 => 4,
            TensorDType::Float16 => 2,
            TensorDType::BFloat16 => 2,
        };

        tensor.data.len() * element_size
    }

    /// Get statistics
    pub async fn get_stats(&self) -> LayerForwardingStats {
        LayerForwardingStats {
            pending_outputs: self.pending_outputs.read().await.len(),
            received_inputs: self.received_inputs.read().await.len(),
            compression_enabled: self.compression_enabled,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerForwardingStats {
    pub pending_outputs: usize,
    pub received_inputs: usize,
    pub compression_enabled: bool,
}

/// Tensor utilities
impl TensorData {
    /// Create new tensor data without KV-cache
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            dtype: TensorDType::Float32,
            key_cache: None,
            value_cache: None,
            kv_cache_shape: None,
        }
    }

    /// Create tensor data with KV-cache
    pub fn with_kv_cache(
        data: Vec<f32>,
        shape: Vec<usize>,
        key_cache: Vec<f32>,
        value_cache: Vec<f32>,
        kv_cache_shape: Vec<usize>,
    ) -> Self {
        Self {
            data,
            shape,
            dtype: TensorDType::Float32,
            key_cache: Some(key_cache),
            value_cache: Some(value_cache),
            kv_cache_shape: Some(kv_cache_shape),
        }
    }

    /// Set KV-cache for this tensor
    pub fn set_kv_cache(
        &mut self,
        key_cache: Vec<f32>,
        value_cache: Vec<f32>,
        kv_cache_shape: Vec<usize>,
    ) {
        self.key_cache = Some(key_cache);
        self.value_cache = Some(value_cache);
        self.kv_cache_shape = Some(kv_cache_shape);
    }

    /// Check if this tensor has KV-cache
    pub fn has_kv_cache(&self) -> bool {
        self.key_cache.is_some() && self.value_cache.is_some()
    }

    /// Get KV-cache size in bytes
    pub fn kv_cache_size_bytes(&self) -> usize {
        if let (Some(key), Some(value)) = (&self.key_cache, &self.value_cache) {
            (key.len() + value.len()) * 4  // f32 = 4 bytes
        } else {
            0
        }
    }

    /// Extract KV-cache from tensor (for passing to model layers)
    pub fn extract_kv_cache(&self) -> Option<(Vec<f32>, Vec<f32>, Vec<usize>)> {
        if let (Some(key), Some(value), Some(shape)) =
            (&self.key_cache, &self.value_cache, &self.kv_cache_shape) {
            Some((key.clone(), value.clone(), shape.clone()))
        } else {
            None
        }
    }

    /// Create from flat array with shape
    pub fn from_array(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(anyhow!(
                "Data size {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_size
            ));
        }

        Ok(Self::new(data, shape))
    }

    /// Get total number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Convert to float16 for reduced bandwidth
    pub fn to_float16(&self) -> Result<Self> {
        // TODO: Implement float16 conversion
        // For now, just return clone
        warn!("Float16 conversion not yet implemented, using Float32");
        Ok(self.clone())
    }

    /// Validate tensor integrity
    pub fn validate(&self) -> Result<()> {
        // Check shape matches data length
        let expected_size: usize = self.shape.iter().product();
        if self.data.len() != expected_size {
            return Err(anyhow!(
                "Tensor data size {} doesn't match shape {:?} (expected {})",
                self.data.len(),
                self.shape,
                expected_size
            ));
        }

        // Check for NaN or Inf
        if self.data.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(anyhow!("Tensor contains NaN or Inf values"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_layer_output_storage() {
        let manager = LayerOutputManager::new(false);

        let tensor = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        manager
            .store_layer_output("req1".to_string(), 5, tensor.clone(), Some("node2".to_string()))
            .await
            .unwrap();

        let output = manager.get_pending_output("req1", 5).await.unwrap();
        assert_eq!(output.layer_index, 5);
        assert_eq!(output.tensor.data.len(), 4);
    }

    #[tokio::test]
    async fn test_tensor_compression() {
        let manager = LayerOutputManager::new(true);

        // Create large tensor (1024 elements)
        let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
        let tensor = TensorData::new(data, vec![32, 32]);

        let compressed = manager.compress_tensor(&tensor).unwrap();
        let decompressed = manager.decompress_tensor(&compressed).unwrap();

        assert_eq!(tensor.data.len(), decompressed.data.len());
        assert_eq!(tensor.shape, decompressed.shape);

        // Verify compression achieved size reduction
        let original_size = bincode::serialize(&tensor).unwrap().len();
        assert!(compressed.len() < original_size);
    }

    #[tokio::test]
    async fn test_wait_for_input_timeout() {
        let manager = LayerOutputManager::new(false);

        let start = std::time::Instant::now();
        let result = manager.wait_for_layer_input("req1", 0, 1).await;

        assert!(result.is_err());
        assert!(start.elapsed().as_secs() >= 1);
    }

    #[tokio::test]
    async fn test_wait_for_input_success() {
        let manager = LayerOutputManager::new(false);
        let manager_clone = manager.clone();

        // Spawn task to store input after 500ms
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            let tensor = TensorData::new(vec![1.0, 2.0], vec![2]);
            manager_clone
                .store_received_input("req1".to_string(), 0, tensor)
                .await
                .unwrap();
        });

        // Wait for input (should succeed before timeout)
        let tensor = manager.wait_for_layer_input("req1", 0, 2).await.unwrap();
        assert_eq!(tensor.data.len(), 2);
    }

    #[test]
    fn test_tensor_validation() {
        let valid_tensor = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(valid_tensor.validate().is_ok());

        let invalid_shape = TensorData::new(vec![1.0, 2.0], vec![2, 2]); // Wrong size
        assert!(invalid_shape.validate().is_err());

        let nan_tensor = TensorData::new(vec![1.0, f32::NAN, 3.0], vec![3]);
        assert!(nan_tensor.validate().is_err());
    }

    #[test]
    fn test_tensor_from_array() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = TensorData::from_array(data, vec![2, 3]).unwrap();

        assert_eq!(tensor.num_elements(), 6);
        assert_eq!(tensor.shape, vec![2, 3]);
    }
}
