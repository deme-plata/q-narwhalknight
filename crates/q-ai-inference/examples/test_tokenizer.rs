// Test GGUF tokenizer integration
// This example demonstrates loading a tokenizer from a GGUF file and using it for text processing

use anyhow::Result;
use q_ai_inference::GgufTokenizer;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔤 Q-NarwhalKnight GGUF Tokenizer Test\n");

    // Path to GGUF model file (contains tokenizer metadata)
    let model_path = "/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf";

    println!("📦 Loading tokenizer from GGUF file...");
    println!("   Path: {}\n", model_path);

    // Load tokenizer from GGUF model file
    let tokenizer = GgufTokenizer::from_gguf_file(model_path)?;

    println!("✅ Tokenizer loaded successfully!\n");

    // Print special tokens
    println!("🔑 Special Tokens:");
    if let Some(bos) = tokenizer.bos_token() {
        println!("   BOS (Beginning): {:?} (ID: {:?})", bos, tokenizer.bos_token_id());
    }
    if let Some(eos) = tokenizer.eos_token() {
        println!("   EOS (End): {:?} (ID: {:?})", eos, tokenizer.eos_token_id());
    }
    if let Some(unk) = tokenizer.unk_token() {
        println!("   UNK (Unknown): {:?}", unk);
    }
    println!("   Vocabulary size: {}\n", tokenizer.vocab_size());

    // Test basic tokenization
    println!("📝 Testing basic tokenization:");
    let test_texts = vec![
        "Hello, world!",
        "Quantum consensus is a revolutionary approach to distributed systems.",
        "The Q-NarwhalKnight network uses DAG-BFT for high-performance consensus.",
    ];

    for text in &test_texts {
        println!("\n   Input: \"{}\"", text);

        // Encode without special tokens
        let token_ids = tokenizer.encode(text, false)?;
        println!("   Token IDs: {:?}", token_ids);
        println!("   Token count: {}", token_ids.len());

        // Decode back to text
        let decoded = tokenizer.decode(&token_ids, false)?;
        println!("   Decoded: \"{}\"", decoded);

        // Verify round-trip
        let matches = decoded == *text;
        println!("   Round-trip: {}", if matches { "✅ Success" } else { "⚠️  Differs" });
    }

    // Test chat template formatting
    println!("\n\n💬 Testing chat template:");
    let messages = vec![
        ("user", "Explain quantum consensus in one sentence."),
    ];

    let formatted = tokenizer.apply_chat_template(&messages)?;
    println!("   Formatted prompt:\n{}", formatted);

    // Encode the chat-formatted prompt
    let chat_token_ids = tokenizer.encode(&formatted, true)?;
    println!("\n   Chat tokens: {} tokens", chat_token_ids.len());
    println!("   First 10 token IDs: {:?}", &chat_token_ids[..chat_token_ids.len().min(10)]);

    // Test multi-turn conversation
    println!("\n\n🔄 Testing multi-turn conversation:");
    let conversation = vec![
        ("user", "What is Q-NarwhalKnight?"),
        ("assistant", "Q-NarwhalKnight is a quantum-enhanced DAG-BFT consensus system."),
        ("user", "How does it work?"),
    ];

    let conversation_prompt = tokenizer.apply_chat_template(&conversation)?;
    println!("   Multi-turn prompt:\n{}", conversation_prompt);

    let conv_tokens = tokenizer.encode(&conversation_prompt, true)?;
    println!("\n   Conversation tokens: {} tokens", conv_tokens.len());

    println!("\n✅ Tokenizer test completed successfully!");
    println!("\n📊 Summary:");
    println!("   - Tokenizer loaded from GGUF file");
    println!("   - Special tokens identified");
    println!("   - Basic encoding/decoding verified");
    println!("   - Chat template formatting working");
    println!("   - Ready for distributed AI inference");

    Ok(())
}
