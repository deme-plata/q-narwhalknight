// Integration test for complete AI inference pipeline
//
// This test validates the entire flow:
// 1. Load GGUF model and tokenizer
// 2. Tokenize input text
// 3. Run distributed inference (mocked for now)
// 4. Apply sampling
// 5. Generate output text
// 6. Validate with privacy layer

use anyhow::Result;
use q_ai_inference::*;
use std::path::Path;
use candle_core::{Device, Tensor};

/// Test tokenizer loading from GGUF file
#[test]
fn test_gguf_tokenizer_loading() -> Result<()> {
    // This test requires an actual GGUF file
    // Skip if not available
    let model_path = "/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf";

    if !Path::new(model_path).exists() {
        println!("⚠️  Skipping test: GGUF model not found at {}", model_path);
        println!("   Download model to run full pipeline tests");
        return Ok(());
    }

    println!("🔄 Loading tokenizer from GGUF file...");
    let tokenizer = GgufTokenizer::from_gguf_file(model_path)?;

    // Test encoding
    let text = "Hello, world!";
    let tokens = tokenizer.encode(text, false)?;
    println!("✅ Encoded '{}' to {} tokens", text, tokens.len());
    assert!(!tokens.is_empty(), "Tokenization should produce tokens");

    // Test decoding
    let decoded = tokenizer.decode(&tokens, true)?;
    println!("✅ Decoded back to: '{}'", decoded);

    // Test special tokens
    if let Some(bos) = tokenizer.bos_token() {
        println!("   BOS token: {}", bos);
    }
    if let Some(eos) = tokenizer.eos_token() {
        println!("   EOS token: {}", eos);
    }

    println!("✅ Tokenizer test passed!");
    Ok(())
}

/// Test sampling with mock logits
#[test]
fn test_sampling_strategies() -> Result<()> {
    println!("🔄 Testing sampling strategies...");

    let vocab_size = 32000; // Typical for Mistral
    let device = Device::Cpu;

    // Create mock logits (uniform distribution)
    let logits_data: Vec<f32> = (0..vocab_size).map(|i| (i as f32).sin()).collect();
    let logits = Tensor::from_vec(logits_data, &[vocab_size], &device)?;

    // Test greedy sampling
    println!("   Testing greedy sampling...");
    let config = sampling::SamplingConfig::greedy();
    let mut sampler = sampling::Sampler::new(config);
    let token = sampler.sample(&logits, &[])?;
    println!("   ✅ Greedy token: {}", token);

    // Test balanced sampling
    println!("   Testing balanced sampling...");
    let config = sampling::SamplingConfig::balanced();
    let mut sampler = sampling::Sampler::new(config);
    let token = sampler.sample(&logits, &[])?;
    println!("   ✅ Balanced token: {}", token);

    // Test creative sampling
    println!("   Testing creative sampling...");
    let config = sampling::SamplingConfig::creative();
    let mut sampler = sampling::Sampler::new(config);
    let token = sampler.sample(&logits, &[])?;
    println!("   ✅ Creative token: {}", token);

    // Test with repetition penalty
    println!("   Testing repetition penalty...");
    let mut config = sampling::SamplingConfig::balanced();
    config.repetition_penalty = 1.5;
    let mut sampler = sampling::Sampler::new(config);

    let previous_tokens = vec![100, 200, 300];
    let token = sampler.sample(&logits, &previous_tokens)?;
    println!("   ✅ Token with penalty: {} (avoiding {:?})", token, previous_tokens);

    println!("✅ Sampling test passed!");
    Ok(())
}

/// Test generation loop with mock model
#[test]
fn test_generation_loop() -> Result<()> {
    println!("🔄 Testing generation loop...");

    // Create a mock tokenizer for testing
    // In practice, use GgufTokenizer::from_gguf_file()
    let tokenizer_json = std::env::var("TEST_TOKENIZER_PATH")
        .unwrap_or_else(|_| "/path/to/tokenizer.json".to_string());

    if !Path::new(&tokenizer_json).exists() {
        println!("⚠️  Skipping test: tokenizer.json not found");
        println!("   Set TEST_TOKENIZER_PATH environment variable to run");
        return Ok(());
    }

    let tokenizer = GgufTokenizer::from_pretrained(&tokenizer_json)?;

    // Create generation config
    let config = generation::GenerationConfig {
        max_tokens: 10,
        sampling: sampling::SamplingConfig::balanced(),
        echo_prompt: false,
        ..Default::default()
    };

    let mut generator = generation::Generator::new(config);

    // Mock prompt
    let prompt_tokens = vec![1, 2, 3]; // [BOS, token1, token2]

    // Mock forward function that returns constant logits
    let vocab_size = 32000;
    let device = Device::Cpu;

    let forward_fn = |_tokens: &[u32]| -> Result<Tensor> {
        // Return logits favoring specific tokens for testing
        let mut logits = vec![0.0f32; vocab_size];
        logits[42] = 10.0;  // Strongly favor token 42
        logits[43] = 9.0;
        logits[44] = 8.0;

        Tensor::from_vec(logits, &[vocab_size], &device)
    };

    println!("   Generating with mock model...");
    let (text, stats) = generator.generate(
        &prompt_tokens,
        &tokenizer,
        forward_fn,
    )?;

    println!("   ✅ Generated {} tokens in {:.2}s",
        stats.tokens_generated,
        stats.generation_time.as_secs_f64()
    );
    println!("   ✅ Throughput: {:.2} tokens/sec", stats.tokens_per_second);
    println!("   ✅ Stop reason: {:?}", stats.stop_reason);
    println!("   ✅ Output length: {} chars", text.len());

    assert!(stats.tokens_generated > 0, "Should generate at least one token");
    assert!(stats.tokens_generated <= 10, "Should not exceed max_tokens");

    println!("✅ Generation test passed!");
    Ok(())
}

/// Test chat template formatting
#[test]
fn test_chat_template() -> Result<()> {
    println!("🔄 Testing chat template...");

    // This test doesn't require an actual tokenizer
    // Just validates the template logic

    let messages = vec![
        ("user", "Explain quantum consensus"),
    ];

    // Mock tokenizer (we just need the API structure)
    let formatted = format_mistral_chat(&messages);

    println!("   Chat template output:");
    println!("   {}", formatted);

    assert!(formatted.contains("[INST]"), "Should contain instruction marker");
    assert!(formatted.contains("[/INST]"), "Should contain instruction end marker");
    assert!(formatted.contains("quantum consensus"), "Should contain user message");

    println!("✅ Chat template test passed!");
    Ok(())
}

/// Helper function to format chat messages (Mistral style)
fn format_mistral_chat(messages: &[(&str, &str)]) -> String {
    let mut formatted = String::new();

    for (role, content) in messages {
        match *role {
            "user" => {
                formatted.push_str("[INST] ");
                formatted.push_str(content);
                formatted.push_str(" [/INST]");
            }
            "assistant" => {
                formatted.push(' ');
                formatted.push_str(content);
                formatted.push_str("</s>");
            }
            "system" => {
                formatted.push_str("[INST] <<SYS>>\n");
                formatted.push_str(content);
                formatted.push_str("\n<</SYS>>\n\n");
            }
            _ => {}
        }
    }

    formatted
}

/// Test complete pipeline with privacy layer (mock)
#[test]
fn test_pipeline_with_privacy() -> Result<()> {
    println!("🔄 Testing pipeline with privacy layer...");

    // Create privacy configuration
    let privacy_config = PrivacyConfig {
        encryption_enabled: true,
        use_aegis_ql: true,
        use_zk_proofs: true,
        proof_complexity: 64,
    };

    println!("   Privacy config:");
    println!("   - Encryption: {}", privacy_config.encryption_enabled);
    println!("   - AEGIS-QL: {}", privacy_config.use_aegis_ql);
    println!("   - ZK proofs: {}", privacy_config.use_zk_proofs);

    // Create privacy layer
    let privacy_layer = PrivacyLayer::new(privacy_config)?;

    // Mock tensor data
    let device = Device::Cpu;
    let tensor_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(tensor_data.clone(), &[4], &device)?;

    // Encrypt tensor
    println!("   Encrypting tensor...");
    let encrypted = privacy_layer.encrypt_tensor(&tensor)?;
    println!("   ✅ Encrypted: {} bytes", encrypted.ciphertext.len());

    // Decrypt tensor
    println!("   Decrypting tensor...");
    let decrypted = privacy_layer.decrypt_tensor(&encrypted)?;
    let decrypted_data = decrypted.to_vec1::<f32>()?;
    println!("   ✅ Decrypted: {:?}", &decrypted_data[..4.min(decrypted_data.len())]);

    // Verify data integrity
    assert_eq!(
        &tensor_data[..],
        &decrypted_data[..],
        "Decrypted data should match original"
    );

    // Generate ZK proof (if enabled)
    if privacy_config.use_zk_proofs {
        println!("   Generating ZK proof...");
        let proof = privacy_layer.generate_computation_proof(&tensor, &decrypted)?;
        println!("   ✅ Proof generated: {} bytes", proof.proof_data.len());

        // Verify proof
        println!("   Verifying ZK proof...");
        let valid = privacy_layer.verify_computation_proof(&proof)?;
        assert!(valid, "Proof should be valid");
        println!("   ✅ Proof verified successfully!");
    }

    println!("✅ Privacy layer test passed!");
    Ok(())
}

/// Test KV-cache integration
#[test]
fn test_kv_cache_coordination() -> Result<()> {
    println!("🔄 Testing KV-cache coordination...");

    use std::sync::Arc;
    use parking_lot::RwLock;

    // Create KV-cache coordinator
    let kv_cache = KVCacheCoordinator::new();

    // Mock conversation
    let conversation_id = "test-conv-123".to_string();
    let user_id = "user-alice".to_string();

    // Store cache entry
    println!("   Storing cache entry...");
    let device = Device::Cpu;
    let key_cache = Tensor::from_vec(vec![1.0f32; 100], &[10, 10], &device)?;
    let value_cache = Tensor::from_vec(vec![2.0f32; 100], &[10, 10], &device)?;

    let entry = KVCacheEntry {
        conversation_id: conversation_id.clone(),
        user_id: user_id.clone(),
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        sequence_length: 10,
        timestamp: std::time::SystemTime::now(),
    };

    kv_cache.store_cache(entry)?;
    println!("   ✅ Cache stored");

    // Retrieve cache entry
    println!("   Retrieving cache entry...");
    if let Some(retrieved) = kv_cache.get_cache(&conversation_id)? {
        println!("   ✅ Cache retrieved");
        println!("   - Sequence length: {}", retrieved.sequence_length);
        println!("   - User: {}", retrieved.user_id);
        assert_eq!(retrieved.sequence_length, 10);
    } else {
        panic!("Cache should be retrievable");
    }

    // Get statistics
    let stats = kv_cache.get_statistics();
    println!("   Cache statistics:");
    println!("   - Total entries: {}", stats.total_entries);
    println!("   - Total size: {} bytes", stats.total_size_bytes);
    println!("   - Hit rate: {:.2}%", stats.hit_rate * 100.0);

    println!("✅ KV-cache test passed!");
    Ok(())
}

/// Full integration test (requires GGUF model)
#[test]
#[ignore] // Run with: cargo test --test integration_full_pipeline -- --ignored
fn test_full_inference_pipeline() -> Result<()> {
    println!("🚀 FULL INTEGRATION TEST");
    println!("========================");

    let model_path = "/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf";

    if !Path::new(model_path).exists() {
        println!("❌ Model not found at: {}", model_path);
        println!("   Download Mistral-7B-Instruct GGUF to run this test");
        return Ok(());
    }

    // Step 1: Load tokenizer
    println!("\n1️⃣  Loading tokenizer from GGUF...");
    let tokenizer = GgufTokenizer::from_gguf_file(model_path)?;
    println!("   ✅ Tokenizer loaded");
    println!("   - Vocab size: {}", tokenizer.vocab_size());

    // Step 2: Prepare prompt
    println!("\n2️⃣  Preparing prompt...");
    let messages = vec![
        ("user", "Explain quantum consensus in one sentence"),
    ];
    let prompt = tokenizer.apply_chat_template(&messages)?;
    println!("   Prompt: {}", prompt);

    let prompt_tokens = tokenizer.encode(&prompt, true)?;
    println!("   ✅ Tokenized: {} tokens", prompt_tokens.len());

    // Step 3: Configure generation
    println!("\n3️⃣  Configuring generation...");
    let gen_config = generation::GenerationConfig {
        max_tokens: 50,
        stop_tokens: tokenizer.eos_token_id().into_iter().collect(),
        sampling: sampling::SamplingConfig::balanced(),
        echo_prompt: false,
        ..Default::default()
    };
    println!("   ✅ Generation config ready");
    println!("   - Max tokens: {}", gen_config.max_tokens);
    println!("   - Temperature: {}", gen_config.sampling.temperature);

    // Step 4: Load model (would happen here in real test)
    println!("\n4️⃣  Loading model weights...");
    println!("   ⚠️  TODO: Implement GGUF weight loading");
    println!("   (Using mock forward function for now)");

    // Mock forward function for testing
    let vocab_size = tokenizer.vocab_size();
    let device = Device::Cpu;
    let forward_fn = |_tokens: &[u32]| -> Result<Tensor> {
        let logits = vec![0.0f32; vocab_size];
        Tensor::from_vec(logits, &[vocab_size], &device)
    };

    // Step 5: Generate response
    println!("\n5️⃣  Generating response...");
    let mut generator = generation::Generator::new(gen_config);

    let (response, stats) = generator.generate(
        &prompt_tokens,
        &tokenizer,
        forward_fn,
    )?;

    // Step 6: Display results
    println!("\n6️⃣  Results:");
    println!("   Response: {}", response);
    println!("   Statistics:");
    println!("   - Tokens generated: {}", stats.tokens_generated);
    println!("   - Time: {:.2}s", stats.generation_time.as_secs_f64());
    println!("   - Throughput: {:.2} tokens/sec", stats.tokens_per_second);
    println!("   - Stop reason: {:?}", stats.stop_reason);

    println!("\n✅ FULL PIPELINE TEST COMPLETE!");
    Ok(())
}
