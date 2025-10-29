//! Test GGUF Model Loading
//!
//! This example validates that we can successfully load and parse
//! the Mistral-7B-Instruct-v0.3 GGUF model file.

use anyhow::Result;
use q_ai_inference::{DeviceCapability, ModelConfig, ModelLoader};
use std::time::Instant;
use tracing::{info, warn};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🚀 GGUF Model Loading Test");
    info!("─────────────────────────────────────────────────────");

    // Detect hardware capability
    info!("🔍 Detecting hardware capabilities...");
    let capability = detect_capability()?;
    info!("✅ Detected capability: {:?}", capability);

    // Create model config
    let model_config = ModelConfig::mistral_7b_instruct();
    info!("📋 Model configuration:");
    info!("   - Name: {}", model_config.model_name);
    info!("   - Path: {}", model_config.model_path);
    info!("   - Layers: {}", model_config.num_layers);
    info!("   - Hidden size: {}", model_config.hidden_size);
    info!("   - Vocab size: {}", model_config.vocab_size);

    // Check if model file exists
    let model_path = std::path::Path::new(&model_config.model_path);
    if !model_path.exists() {
        warn!("⚠️  Model file not found at: {}", model_config.model_path);
        warn!("   Please ensure the GGUF file is downloaded.");
        return Ok(());
    }

    let file_size = std::fs::metadata(model_path)?.len();
    info!("✅ Model file found: {:.2} GB", file_size as f64 / 1_073_741_824.0);

    // Create model loader
    let loader = ModelLoader::new(capability.clone());
    info!("🔧 Model loader created");

    // Test loading different layer ranges
    let test_cases = vec![
        (0, 3, "First 4 layers"),
        (10, 15, "Middle layers"),
        (29, 31, "Last 3 layers"),
    ];

    for (layer_start, layer_end, description) in test_cases {
        info!("");
        info!("📦 Testing: {} (layers {}-{})", description, layer_start, layer_end);

        let start_time = Instant::now();

        match loader.load_model(model_config.clone(), layer_start, layer_end) {
            Ok(loaded_model) => {
                let elapsed = start_time.elapsed();
                let num_layers = loaded_model.layer_end - loaded_model.layer_start + 1;
                info!("✅ Successfully loaded {} layers in {:?}", num_layers, elapsed);
                info!("   - Layer range: {}-{}",
                    loaded_model.layer_start, loaded_model.layer_end);
                info!("   - Device: {:?}", loaded_model.device);

                // Estimate memory usage
                let memory_mb = loader.estimate_memory_usage(
                    &model_config,
                    num_layers
                ) / (1024 * 1024);
                info!("   - Estimated memory: {} MB", memory_mb);
            }
            Err(e) => {
                warn!("⚠️  Failed to load layers {}-{}: {}",
                    layer_start, layer_end, e);
                warn!("   This is expected as GGUF parsing is not yet fully implemented");
            }
        }
    }

    info!("");
    info!("─────────────────────────────────────────────────────");
    info!("✅ GGUF loading test complete");
    info!("");
    info!("📝 Next steps:");
    info!("   1. Implement actual GGUF file parsing");
    info!("   2. Load model weights into Candle tensors");
    info!("   3. Test forward pass through loaded layers");
    info!("   4. Validate inference output quality");

    Ok(())
}

fn detect_capability() -> Result<DeviceCapability> {
    // Try CUDA first
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if output.status.success() {
            if let Ok(vram_str) = String::from_utf8(output.stdout) {
                if let Ok(vram_mb) = vram_str.trim().parse::<f64>() {
                    let vram_gb = (vram_mb / 1024.0).round() as usize;
                    info!("🎮 CUDA GPU detected: {} GB VRAM", vram_gb);
                    return Ok(DeviceCapability::CUDA {
                        vram_gb,
                        compute_capability: "8.0".to_string(), // Default
                    });
                }
            }
        }
    }

    // Try Metal (macOS)
    if cfg!(target_os = "macos") {
        if let Ok(output) = std::process::Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
        {
            if output.status.success() {
                // Assume 8GB for Metal GPUs (conservative estimate)
                info!("🍎 Metal GPU detected (macOS)");
                return Ok(DeviceCapability::Metal { vram_gb: 8 });
            }
        }
    }

    // Fallback to CPU
    let sys = sysinfo::System::new_all();
    let cores = sys.cpus().len();
    let ram_bytes = sys.total_memory();
    let ram_gb = (ram_bytes / 1_073_741_824) as usize;

    info!("💻 CPU detected: {} cores, {} GB RAM", cores, ram_gb);
    Ok(DeviceCapability::CPU { cores, ram_gb })
}
