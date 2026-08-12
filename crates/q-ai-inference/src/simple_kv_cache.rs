/// Simple KV-Cache for Autoregressive Generation
///
/// This module provides a straightforward KV-cache implementation to avoid
/// recomputing attention keys and values for previous tokens during autoregressive
/// text generation.
///
/// Expected speedup: 3-5x for multi-token generation

use candle_core::{Result, Tensor};

/// Per-layer KV cache storing key and value tensors
///
/// Shape: [batch_size, num_kv_heads, seq_len, head_dim]
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// Cached key tensor: [batch, num_kv_heads, cached_seq_len, head_dim]
    pub k_cache: Option<Tensor>,
    /// Cached value tensor: [batch, num_kv_heads, cached_seq_len, head_dim]
    pub v_cache: Option<Tensor>,
}

impl LayerKVCache {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            k_cache: None,
            v_cache: None,
        }
    }

    /// Get the current cache size (number of cached tokens)
    pub fn cache_size(&self) -> usize {
        match &self.k_cache {
            Some(k) => k.dims()[2], // seq_len dimension
            None => 0,
        }
    }

    /// Update cache with new K/V tensors
    ///
    /// For the first token: Store K/V directly
    /// For subsequent tokens: Concatenate new K/V with cached K/V
    ///
    /// # Arguments
    /// * `k` - New key tensor [batch, num_kv_heads, new_seq_len, head_dim]
    /// * `v` - New value tensor [batch, num_kv_heads, new_seq_len, head_dim]
    ///
    /// # Returns
    /// Complete K/V tensors including cache: [batch, num_kv_heads, total_seq_len, head_dim]
    pub fn update(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let (k_full, v_full) = match (&self.k_cache, &self.v_cache) {
            (None, None) => {
                // First token: no cache yet, store current K/V
                (k.clone(), v.clone())
            }
            (Some(k_cached), Some(v_cached)) => {
                // Subsequent tokens: concatenate along seq_len dimension (dim 2)
                let k_full = Tensor::cat(&[k_cached, &k], 2)?;
                let v_full = Tensor::cat(&[v_cached, &v], 2)?;
                (k_full, v_full)
            }
            _ => {
                return Err(candle_core::Error::Msg(
                    "Inconsistent cache state: K and V must both be cached or both empty".to_string()
                ));
            }
        };

        // Update cache
        self.k_cache = Some(k_full.clone());
        self.v_cache = Some(v_full.clone());

        Ok((k_full, v_full))
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }
}

impl Default for LayerKVCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_layer_kv_cache_basic() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = LayerKVCache::new();

        // First token: [1, 8, 1, 128]
        let k1 = Tensor::zeros((1, 8, 1, 128), candle_core::DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 8, 1, 128), candle_core::DType::F32, &device)?;

        let (k_full, v_full) = cache.update(k1, v1)?;
        assert_eq!(k_full.dims(), &[1, 8, 1, 128]);
        assert_eq!(cache.cache_size(), 1);

        // Second token: [1, 8, 1, 128]
        let k2 = Tensor::zeros((1, 8, 1, 128), candle_core::DType::F32, &device)?;
        let v2 = Tensor::zeros((1, 8, 1, 128), candle_core::DType::F32, &device)?;

        let (k_full, v_full) = cache.update(k2, v2)?;
        assert_eq!(k_full.dims(), &[1, 8, 2, 128]);  // Concatenated!
        assert_eq!(cache.cache_size(), 2);

        Ok(())
    }
}
