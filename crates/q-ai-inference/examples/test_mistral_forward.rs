//! Test Mistral-7B forward pass with actual GGUF model
//!
//! This test validates:
//! - Loading layer weights from GGUF
//! - Constructing MistralLayer from weights
//! - Running forward pass with test input
//! - Validating output shapes and values
//! - Performance measurement

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use q_ai_inference::capability_detector::CapabilityDetector;
use q_ai_inference::{GGUFModelLoader, MistralConfig, MistralLayer};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🧪 Mistral-7B Forward Pass Test");
    println!("================================\n");

    // Detect hardware capabilities
    println!("🔍 Step 1: Hardware Detection");
    let mut detector = CapabilityDetector::new();
    let capability = detector.detect()?;
    println!("   ✅ Detected: {:?}\n", capability);

    // Initialize device
    let device = match &capability {
        q_ai_inference::DeviceCapability::CUDA { .. } => {
            #[cfg(feature = "cuda")]
            {
                println!("   🚀 Using CUDA GPU acceleration");
                Device::new_cuda(0)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                println!("   ⚠️  CUDA not available, using CPU");
                Device::Cpu
            }
        }
        q_ai_inference::DeviceCapability::Metal { .. } => {
            #[cfg(feature = "metal")]
            {
                println!("   🚀 Using Metal GPU acceleration");
                Device::new_metal(0)?
            }
            #[cfg(not(feature = "metal"))]
            {
                println!("   ⚠️  Metal not available, using CPU");
                Device::Cpu
            }
        }
        q_ai_inference::DeviceCapability::CPU { cores, .. } => {
            println!("   💻 Using CPU with {} cores", cores);
            Device::Cpu
        }
    };

    // Model path
    let model_path =
        "/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf";

    println!("\n📦 Step 2: GGUF Model Loading");
    println!("   Model: {}", model_path);

    // Check if model exists
    if !std::path::Path::new(model_path).exists() {
        println!("   ❌ Model file not found!");
        println!("   Please download the model first.");
        return Ok(());
    }

    let file_size = std::fs::metadata(model_path)?.len();
    println!("   ✅ Model found: {:.2} GB", file_size as f64 / 1_073_741_824.0);

    // Create GGUF loader
    let loader = GGUFModelLoader::new(model_path, &capability)?;

    // Load first layer only (for testing)
    println!("\n🔨 Step 3: Loading Layer 0");
    let start = std::time::Instant::now();
    let layers = loader.load_layer_range(0, 0)?;
    let load_time = start.elapsed();

    println!("   ✅ Loaded {} layer in {:.2}s", layers.len(), load_time.as_secs_f64());

    // Verify layer weights
    let layer_weights = &layers[0];
    println!("\n   Layer 0 Components:");
    println!("   - Q projection: {}", layer_weights.attn_q.is_some());
    println!("   - K projection: {}", layer_weights.attn_k.is_some());
    println!("   - V projection: {}", layer_weights.attn_v.is_some());
    println!("   - O projection: {}", layer_weights.attn_output.is_some());
    println!("   - FFN gate: {}", layer_weights.ffn_gate.is_some());
    println!("   - FFN up: {}", layer_weights.ffn_up.is_some());
    println!("   - FFN down: {}", layer_weights.ffn_down.is_some());
    println!("   - Attn norm: {}", layer_weights.attn_norm.is_some());
    println!("   - FFN norm: {}", layer_weights.ffn_norm.is_some());

    // Create Mistral config
    println!("\n⚙️  Step 4: Model Configuration");
    let config = MistralConfig::mistral_7b_v0_3();
    println!("   Architecture: Mistral-7B-Instruct-v0.3");
    println!("   Layers: {}", config.num_hidden_layers);
    println!("   Hidden size: {}", config.hidden_size);
    println!("   Attention heads: {} (query), {} (KV)",
             config.num_attention_heads, config.num_key_value_heads);
    println!("   FFN intermediate: {}", config.intermediate_size);
    println!("   Max sequence length: {}", config.max_position_embeddings);

    // Construct MistralLayer
    println!("\n🏗️  Step 5: Constructing MistralLayer");
    let start = std::time::Instant::now();
    let mistral_layer = MistralLayer::from_weights(layer_weights, &config, &device)?;
    let construct_time = start.elapsed();
    println!("   ✅ Layer constructed in {:.2}s", construct_time.as_secs_f64());
    println!("   (Includes weight dequantization)");

    // Create test input
    println!("\n🧪 Step 6: Creating Test Input");
    let batch_size = 1;
    let seq_len = 10;
    let hidden_size = config.hidden_size;

    println!("   Input shape: [{}, {}, {}]", batch_size, seq_len, hidden_size);
    println!("   (batch_size, sequence_length, hidden_size)");

    let input = Tensor::randn(0f32, 1.0f32, (batch_size, seq_len, hidden_size), &device)?;
    println!("   ✅ Test input created");

    // Create position IDs
    let position_ids = Tensor::arange(0u32, seq_len as u32, &device)?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?; // [1, seq_len]

    println!("\n🚀 Step 7: Running Forward Pass");
    println!("   This will execute:");
    println!("   1. Pre-attention RMSNorm");
    println!("   2. Grouped-Query Attention (32 query heads, 8 KV heads)");
    println!("   3. Residual connection");
    println!("   4. Pre-FFN RMSNorm");
    println!("   5. SwiGLU FFN");
    println!("   6. Residual connection");

    let start = std::time::Instant::now();
    let output = mistral_layer.forward(&input, None, &position_ids)?;
    let forward_time = start.elapsed();

    println!("   ✅ Forward pass complete in {:.2}ms", forward_time.as_millis());

    // Validate output
    println!("\n✅ Step 8: Output Validation");
    let output_shape = output.dims();
    println!("   Output shape: {:?}", output_shape);

    // Check shape
    assert_eq!(output_shape.len(), 3, "Output should be 3D");
    assert_eq!(output_shape[0], batch_size, "Batch size mismatch");
    assert_eq!(output_shape[1], seq_len, "Sequence length mismatch");
    assert_eq!(output_shape[2], hidden_size, "Hidden size mismatch");
    println!("   ✅ Shape validation: PASSED");

    // Check for NaN/Inf
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    let has_nan = output_vec.iter().any(|x| x.is_nan());
    let has_inf = output_vec.iter().any(|x| x.is_infinite());

    if has_nan {
        println!("   ❌ Output contains NaN values!");
        return Err(anyhow::anyhow!("Forward pass produced NaN"));
    }
    if has_inf {
        println!("   ❌ Output contains Inf values!");
        return Err(anyhow::anyhow!("Forward pass produced Inf"));
    }
    println!("   ✅ NaN/Inf check: PASSED");

    // Compute output statistics
    let output_mean = output_vec.iter().sum::<f32>() / output_vec.len() as f32;
    let output_min = output_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let output_max = output_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("   📊 Output Statistics:");
    println!("      Mean: {:.6}", output_mean);
    println!("      Min:  {:.6}", output_min);
    println!("      Max:  {:.6}", output_max);

    // Performance summary
    println!("\n⚡ Performance Summary");
    println!("   Layer loading: {:.2}s", load_time.as_secs_f64());
    println!("   Layer construction: {:.2}s", construct_time.as_secs_f64());
    println!("   Forward pass: {:.2}ms", forward_time.as_millis());
    println!("   Total: {:.2}s",
             (load_time + construct_time + forward_time).as_secs_f64());

    // Extrapolate to full model
    println!("\n🔮 Full Model Projection (32 layers):");
    let total_load = load_time.as_secs_f64() * 32.0;
    let total_construct = construct_time.as_secs_f64() * 32.0;
    let total_forward = forward_time.as_millis() as f64 * 32.0;

    println!("   Loading all layers: {:.2}s", total_load);
    println!("   Constructing all layers: {:.2}s", total_construct);
    println!("   Single token forward pass: {:.2}ms", total_forward);
    println!("   Throughput: {:.2} tokens/second", 1000.0 / total_forward);

    println!("\n🎉 TEST COMPLETE - ALL VALIDATIONS PASSED");
    println!("   ✅ GGUF loading works");
    println!("   ✅ Layer construction works");
    println!("   ✅ Forward pass executes successfully");
    println!("   ✅ Output shapes are correct");
    println!("   ✅ No NaN or Inf values");
    println!("   ✅ Ready for distributed inference!");

    Ok(())
}
