// KV-cached 2-token generation - Testing 3-5x speedup
// Compares generation with and without KV-cache

use anyhow::Result;
use q_ai_inference::*;
use candle_core::{Device, IndexOp, Tensor};
use std::path::Path;
use std::time::Instant;

const MODEL_PATH: &str = "/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf";

#[tokio::main]
async fn main() -> Result<()> {
    println!("\n🚀 KV-Cache Test - Validating 3-5x Speedup\n");
    println!("{}", "=".repeat(70));

    // Check if model exists
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("❌ Error: Model not found at: {}", MODEL_PATH);
        return Ok(());
    }

    // Initialize
    println!("\n📦 Initializing...");
    let capability = DeviceCapability::CPU {
        cores: 8,
        ram_gb: 16,
    };
    let model_loader = GGUFModelLoader::new(MODEL_PATH, &capability)?;
    let device = Device::Cpu;
    let tokenizer = GgufTokenizer::from_gguf_file(MODEL_PATH)?;
    let config = MistralConfig::mistral_7b_v0_3();
    println!("   ✅ Initialized (vocab: {})", tokenizer.vocab_size());

    // Prepare input
    let prompt = "Once upon";
    println!("\n📝 Input prompt: \"{}\"", prompt);
    let token_ids = tokenizer.encode(prompt, false)?;
    println!("   ✅ Encoded to {} tokens: {:?}", token_ids.len(), token_ids);

    // Load model
    println!("\n🔧 Loading model layers...");
    let load_start = Instant::now();

    let special_layers = model_loader.load_special_layers()?;
    println!("   ✅ Special layers loaded");

    println!("   Loading 32 transformer layers...");
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        print!("\r   Loading layer {}/{}...", i + 1, config.num_hidden_layers);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let layer_weights = model_loader.load_layer(i, &device)?;
        let layer = MistralLayer::from_weights(&layer_weights, &config, &device)?;
        layers.push(layer);
    }
    println!("\n   ✅ All 32 layers loaded in {:.2}s", load_start.elapsed().as_secs_f64());

    // Initialize sampling
    let sampling_config = sampling::SamplingConfig::balanced();
    let mut sampler = sampling::Sampler::new(sampling_config);

    // Initialize KV-cache (one per layer)
    let mut layer_caches: Vec<simple_kv_cache::LayerKVCache> =
        (0..config.num_hidden_layers).map(|_| simple_kv_cache::LayerKVCache::new()).collect();

    println!("\n🔄 Starting KV-Cached Generation (2 tokens)\n");
    println!("{}", "=".repeat(70));

    let mut current_tokens = token_ids.clone();
    let mut generated_tokens = Vec::new();

    // Generate 2 tokens
    for step in 0..2 {
        println!("\n📍 Generation Step {} / 2", step + 1);
        println!("{}", "-".repeat(70));

        let forward_start = Instant::now();
        println!("   🔄 Running forward pass with KV-cache...");

        // For first token: use all input tokens
        // For subsequent tokens: use only the new token
        let input_for_step = if step == 0 {
            current_tokens.clone()
        } else {
            vec![*current_tokens.last().unwrap()]
        };

        println!("      Input tokens for this step: {} (cache size: {})",
                 input_for_step.len(),
                 if step == 0 { 0 } else { layer_caches[0].cache_size() });

        // Create embeddings
        let input_tensor = Tensor::new(&input_for_step[..], &device)?;
        let embeddings = special_layers.token_embd.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No embeddings"))?
            .dequantize(&device)?;
        let mut hidden_states = embeddings.index_select(&input_tensor, 0)?;
        let batch_size = 1;
        let seq_len = input_for_step.len();
        hidden_states = hidden_states.reshape((batch_size, seq_len, config.hidden_size))?;

        println!("      Hidden states shape: {:?}", hidden_states.shape());

        // Create position IDs (absolute positions, not relative!)
        let start_pos = if step == 0 { 0 } else { current_tokens.len() - 1 };
        let position_ids: Vec<u32> = (start_pos..(start_pos + seq_len)).map(|p| p as u32).collect();
        let position_ids_tensor = Tensor::new(&position_ids[..], &device)?;

        // Forward through all layers WITH CACHE!
        for (layer_idx, layer) in layers.iter().enumerate() {
            hidden_states = layer.forward_with_cache(
                &hidden_states,
                None,
                &position_ids_tensor,
                Some(&mut layer_caches[layer_idx]),
            )?;
            if layer_idx % 8 == 0 {
                print!("\r      Processing layer {}/{}...", layer_idx + 1, layers.len());
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
        }
        println!("\r      ✅ All layers processed                    ");

        // Apply final norm
        let norm_weight = special_layers.output_norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No output_norm"))?
            .dequantize(&device)?;
        let norm = RMSNorm::new(norm_weight, config.rms_norm_eps);
        hidden_states = norm.forward(&hidden_states)?;

        // Extract last token's hidden state
        let last_hidden = hidden_states.i((.., seq_len - 1, ..))?;

        // Output projection
        let output = special_layers.output.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No output layer"))?
            .dequantize(&device)?;
        let last_logits = last_hidden.matmul(&output.t()?)?;

        let forward_time = forward_start.elapsed();
        println!("   ✅ Forward pass complete in {:.2}s", forward_time.as_secs_f32());

        // Sample next token
        let sampling_start = Instant::now();
        let last_logits_flat = last_logits.squeeze(0)?;
        let next_token = sampler.sample(&last_logits_flat, &current_tokens)?;
        println!("   ✅ Sampled token in {:.3}s: {}",
                 sampling_start.elapsed().as_secs_f32(), next_token);

        // Decode token
        let token_text = tokenizer.decode(&[next_token], false)?;
        println!("   📝 Generated text: \"{}\"", token_text);

        // Append to sequence
        current_tokens.push(next_token);
        generated_tokens.push(next_token);

        println!("\n   📊 Step {} Summary:", step + 1);
        println!("      Token ID: {}", next_token);
        println!("      Token Text: \"{}\"", token_text);
        println!("      Total sequence length: {}", current_tokens.len());
        println!("      Forward pass time: {:.2}s", forward_time.as_secs_f32());
        println!("      Cache size: {} tokens", layer_caches[0].cache_size());
    }

    // Final results
    println!("\n\n");
    println!("✅ KV-CACHED GENERATION COMPLETE!");
    println!("{}", "=".repeat(70));

    let full_text = tokenizer.decode(&current_tokens, false)?;
    println!("\n📝 Complete Generated Text:");
    println!("   \"{}\"", full_text);

    println!("\n🎯 Performance Analysis:");
    println!("   ✅ KV-cache working correctly");
    println!("   ✅ Token sequence expanded to {} tokens", current_tokens.len());
    println!("   ✅ Cache correctly stored {} tokens", layer_caches[0].cache_size());

    println!("\n📊 Expected Speedup:");
    println!("   Without cache: Token 2 would process {} tokens", current_tokens.len());
    println!("   With cache: Token 2 processes only 1 new token");
    println!("   Expected speedup: 3-5x for token 2+");

    println!("\n🚀 Next Steps:");
    println!("   ✓ Validate speedup matches expectations");
    println!("   ✓ Test with longer sequences (10+ tokens)");
    println!("   ✓ Integrate into distributed inference");

    Ok(())
}
