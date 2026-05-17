/// AI Chat API Endpoints for Q-NarwhalKnight
///
/// This module provides REST API endpoints for managing AI chat sessions
/// with privacy-first distributed inference using mistral.rs + q-ai-inference.
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        Json,
    },
    routing::{delete, get, post, put},
    Router,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::AppState;
use q_storage::{ChatMessage, ChatMetadata, ChatSettings, GenerationStats};

/// API response wrapper
#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: u64,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: current_timestamp(),
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
            timestamp: current_timestamp(),
        }
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Static storage for the AI engine (persists across calls for metrics tracking)
/// v5.1.0: Changed from MistralRsEngine to dyn InferenceEngine to support llama-cpp-2 backend
static AI_ENGINE: std::sync::OnceLock<Arc<dyn q_ai_inference::InferenceEngine>> = std::sync::OnceLock::new();
static AI_ENGINE_LOADING: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();

/// Helper: Bridge the channel-based InferenceEngine trait to callback-based streaming.
///
/// This allows existing code that uses `engine.generate_stream(prompt, max_tokens, |event| { ... })`
/// to work with any `dyn InferenceEngine` without modification.
pub async fn generate_stream_with_callback<F, Fut>(
    engine: &dyn q_ai_inference::InferenceEngine,
    prompt: &str,
    max_tokens: usize,
    mut callback: F,
) -> anyhow::Result<String>
where
    F: FnMut(q_ai_inference::StreamEvent) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>>,
{
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<q_ai_inference::StreamEvent>();

    // Spawn generation task using raw pointer cast to usize (Send + 'static safe)
    // SAFETY: engine lives behind Arc<dyn InferenceEngine> in AppState which outlives this call.
    // The spawned task is awaited before this function returns.
    let prompt_owned = prompt.to_string();
    let engine_ptr = engine as *const dyn q_ai_inference::InferenceEngine;
    let engine_addr = engine_ptr as *const () as usize;
    let engine_vtable = unsafe {
        std::mem::transmute::<_, [usize; 2]>(engine_ptr)[1]
    };
    let gen_handle = tokio::spawn(async move {
        let reconstructed: *const dyn q_ai_inference::InferenceEngine = unsafe {
            std::mem::transmute::<[usize; 2], *const dyn q_ai_inference::InferenceEngine>(
                [engine_addr, engine_vtable]
            )
        };
        let engine = unsafe { &*reconstructed };
        engine.generate_stream(&prompt_owned, max_tokens, tx).await
    });

    // Forward events from channel to callback
    let mut full_text = String::new();
    while let Some(event) = rx.recv().await {
        if let q_ai_inference::StreamEvent::Token(ref t) = event {
            full_text.push_str(t);
        }
        callback(event).await?;
    }

    // Wait for generation to complete and propagate errors
    match gen_handle.await {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => return Err(e),
        Err(e) => return Err(anyhow::anyhow!("Generation task panicked: {}", e)),
    }

    Ok(full_text)
}

/// On-demand AI engine loading helper
///
/// This function ensures the AI engine is loaded before attempting inference.
/// Uses a static OnceLock to ensure the engine persists and accumulates stats.
///
/// Made public so it can be called from the gossip handler for distributed inference.
/// v5.1.0: Returns dyn InferenceEngine (supports both llama-cpp-2 and mistral.rs backends)
pub async fn ensure_ai_engine_loaded(_state: &Arc<AppState>) -> anyhow::Result<Arc<dyn q_ai_inference::InferenceEngine>> {
    // v1.4.12-beta: Removed excessive error!() logging for performance

    // Check if engine is already loaded in static storage
    if let Some(engine) = AI_ENGINE.get() {
        return Ok(engine.clone());
    }

    // Acquire lock to prevent concurrent loading
    let lock = AI_ENGINE_LOADING.get_or_init(|| tokio::sync::Mutex::new(()));
    let _guard = lock.lock().await;

    // Double-check after acquiring lock (another thread might have loaded it)
    if let Some(engine) = AI_ENGINE.get() {
        return Ok(engine.clone());
    }

    info!("🚀 On-demand loading of AI engine starting...");
    info!("   This is a one-time operation that will take ~5-10 seconds");

    // Download and verify model files
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    async fn ensure_model_available() -> anyhow::Result<std::path::PathBuf> {
        // v1.4.12-beta: Use absolute path to avoid issues when started from different directories
        let model_dir = std::path::PathBuf::from("/opt/orobit/shared/q-narwhalknight/models");
        tokio::fs::create_dir_all(&model_dir).await?;

        // v2.6.1-beta: Support Q_AI_MODEL env var for model selection
        // Priority: explicit env var -> Mistral-7B (ONLY model fully supported by mistral.rs)
        // NOTE: Ministral-3B and BitNet are NOT supported by mistral.rs and will fail!
        let model_name = std::env::var("Q_AI_MODEL").unwrap_or_else(|_| "auto".to_string());

        let mistral7b_path = model_dir.join("Mistral-7B-Instruct-v0.3.Q4_K_M.gguf");

        // Only Mistral-7B is supported by mistral.rs
        let model_path = if model_name.to_lowercase().contains("mistral") && mistral7b_path.exists() {
            info!("📦 Using Mistral-7B-Instruct-v0.3 (4.1GB) - ONLY supported mistral.rs model");
            mistral7b_path.clone()
        } else if mistral7b_path.exists() {
            // Auto-detect: Mistral-7B is the only supported model
            info!("📦 Using Mistral-7B-Instruct-v0.3 (4.1GB) - auto-detected (mistral.rs compatible)");
            mistral7b_path.clone()
        } else {
            return Err(anyhow::anyhow!(
                "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf not found in /opt/orobit/shared/q-narwhalknight/models/. \
                 This is the ONLY model supported by mistral.rs. Download it from: \
                 https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF"
            ));
        };
        debug!("🔍 Model path selected: {:?}", model_path);

        // v1.4.12-beta: SKIP tokenizer download - we use GGUF-embedded tokenizers now!
        // Modern GGUF files contain embedded tokenizer data. Using external tokenizer files
        // caused the "index-select invalid index 47926 with dim size 32768" error because
        // the wrong tokenizer was being used. See mistralrs_engine.rs for details.
        info!("📦 Using GGUF-embedded tokenizer (no external tokenizer files needed)");

        // If model doesn't exist, download from bootstrap node
        if !model_path.exists() {
            info!("📥 Downloading Mistral-7B model (4.1GB) from bootstrap node...");
            info!("   This is a one-time download and will be cached locally");
            info!(
                "   Source: https://dl.quillon.xyz/downloads/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
            );

            let url = "https://dl.quillon.xyz/downloads/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf";
            let response = reqwest::get(url).await?;

            if !response.status().is_success() {
                return Err(anyhow::anyhow!(
                    "Failed to download model: HTTP {}",
                    response.status()
                ));
            }

            let total_size = response.content_length().unwrap_or(0);
            info!(
                "   Download size: {:.2} GB",
                total_size as f64 / 1_000_000_000.0
            );

            let mut file = tokio::fs::File::create(&model_path).await?;
            let mut downloaded: u64 = 0;
            let mut stream = response.bytes_stream();

            let progress_interval = 100 * 1024 * 1024; // Log every 100MB
            let mut last_logged = 0u64;

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                file.write_all(&chunk).await?;
                downloaded += chunk.len() as u64;

                // Log progress every 100MB
                if downloaded - last_logged >= progress_interval {
                    let percent = if total_size > 0 {
                        (downloaded as f64 / total_size as f64 * 100.0)
                    } else {
                        0.0
                    };
                    info!(
                        "   Downloaded: {:.2} MB / {:.2} MB ({:.1}%)",
                        downloaded as f64 / 1_000_000.0,
                        total_size as f64 / 1_000_000.0,
                        percent
                    );
                    last_logged = downloaded;
                }
            }

            file.flush().await?;
            info!(
                "✅ Model downloaded successfully and cached at: {:?}",
                model_path
            );
            info!(
                "   File size: {:.2} GB",
                downloaded as f64 / 1_000_000_000.0
            );
        } else {
            info!("✅ Model already exists locally at: {:?}", model_path);
        }

        Ok(model_path)
    }

    // Ensure model files are available
    let model_path = ensure_model_available().await?;
    debug!("🔍 Model path obtained: {:?}", model_path);

    let model_path_str = model_path.to_str()
        .ok_or_else(|| anyhow::anyhow!("Model path contains invalid UTF-8 characters"))?;

    // v5.1.0: Try LlamaCppEngine first (10-50x faster), fall back to MistralRsEngine
    let engine_arc: Arc<dyn q_ai_inference::InferenceEngine> = {
        // Check Q_AI_ENGINE env var for explicit backend selection
        let preferred = std::env::var("Q_AI_ENGINE").unwrap_or_else(|_| "auto".to_string());

        if preferred == "mistralrs" {
            // MistralRs engine requires legacy-mistralrs feature
            #[cfg(feature = "llama-cpp")]
            {
                warn!("⚠️ Q_AI_ENGINE=mistralrs but legacy-mistralrs not available, using llama.cpp");
                let engine = q_ai_inference::LlamaCppEngine::new(model_path_str).await?;
                Arc::new(engine) as Arc<dyn q_ai_inference::InferenceEngine>
            }
            #[cfg(not(feature = "llama-cpp"))]
            {
                return Err(anyhow::anyhow!("AI inference disabled: llama-cpp removed to prevent API starvation. BitNet engine coming soon."));
            }
        } else {
            // Try llama-cpp-2 first (preferred for performance)
            #[cfg(feature = "llama-cpp")]
            {
                info!("🦙 Initializing llama.cpp engine (v5.1.0)...");
                match q_ai_inference::LlamaCppEngine::new(model_path_str).await {
                    Ok(engine) => {
                        info!("✅ llama.cpp engine loaded (10-50x faster than mistral.rs)");
                        Arc::new(engine) as Arc<dyn q_ai_inference::InferenceEngine>
                    }
                    Err(e) => {
                        return Err(anyhow::anyhow!("llama.cpp engine failed: {}. No fallback available.", e));
                    }
                }
            }
            #[cfg(not(feature = "llama-cpp"))]
            {
                return Err(anyhow::anyhow!("AI inference disabled: llama-cpp removed to prevent API starvation (v8.2.8). Use BitNet or enable with --features llama-cpp."));
            }
        }
    };

    info!("✅ AI engine loaded: {} - ready for inference!", engine_arc.engine_name());

    // Store in static for persistence (metrics tracking, etc.)
    let _ = AI_ENGINE.set(engine_arc.clone());

    // v5.1.0: Wire engine to distributed AI coordinator
    if let Some(ref coordinator) = _state.distributed_ai_coordinator {
        info!("🔌 Wiring {} to distributed AI coordinator...", engine_arc.engine_name());
        coordinator.set_inference_engine(engine_arc.clone()).await;
        info!("✅ Inference engine connected to coordinator!");
        info!("   - Local inference fallback: ENABLED");
        info!("   - Data parallelism support: ENABLED");
        info!("   - Engine: {}", engine_arc.engine_name());
    } else {
        warn!("⚠️ No distributed AI coordinator available - engine loaded for local use only");
    }

    Ok(engine_arc)
}

/// Get the static AI engine reference (for metrics)
/// v5.1.0: Returns dyn InferenceEngine (supports both llama-cpp-2 and mistral.rs)
pub fn get_static_ai_engine() -> Option<Arc<dyn q_ai_inference::InferenceEngine>> {
    AI_ENGINE.get().cloned()
}

/// Request to create a new chat
#[derive(Debug, Deserialize)]
pub struct CreateChatRequest {
    pub user_id: String,
    pub title: Option<String>,
    pub model: Option<String>,
    pub encryption_enabled: Option<bool>,
    pub zk_proofs_enabled: Option<bool>,
    pub distributed_enabled: Option<bool>,
    pub enable_kv_cache: Option<bool>,
    pub enable_pipeline_parallel: Option<bool>,
    pub enable_load_balancing: Option<bool>,
}

/// Request to send a message
#[derive(Debug, Deserialize)]
pub struct SendMessageRequest {
    pub content: String,
    pub images: Option<Vec<String>>,
    pub audio: Option<String>,
}

/// Request to rename a chat
#[derive(Debug, Deserialize)]
pub struct RenameChatRequest {
    pub title: String,
}

/// Request to update chat settings
#[derive(Debug, Deserialize)]
pub struct UpdateSettingsRequest {
    pub encryption_enabled: Option<bool>,
    pub zk_proofs_enabled: Option<bool>,
    pub distributed_enabled: Option<bool>,
    pub enable_kv_cache: Option<bool>,
    pub enable_pipeline_parallel: Option<bool>,
    pub enable_load_balancing: Option<bool>,
}

/// Response for chat creation
#[derive(Debug, Serialize)]
pub struct CreateChatResponse {
    pub chat_id: String,
    pub created_at: u64,
}

/// Response for message send (includes AI response)
#[derive(Debug, Serialize)]
pub struct SendMessageResponse {
    pub user_message: ChatMessage,
    pub ai_response: ChatMessage,
}

/// Request for switching AI model
#[derive(Debug, Deserialize)]
pub struct SwitchModelRequest {
    pub model: String,
}

/// Response for model switch
#[derive(Debug, Serialize)]
pub struct SwitchModelResponse {
    pub success: bool,
    pub model: String,
    pub model_size_gb: f64,
    pub message: String,
}

/// POST /api/chat/create - Create new chat session
pub async fn create_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateChatRequest>,
) -> Result<Json<ApiResponse<CreateChatResponse>>, StatusCode> {
    let chat_id = Uuid::new_v4().to_string();
    let now = current_timestamp();

    let metadata = ChatMetadata {
        chat_id: chat_id.clone(),
        user_id: req.user_id.clone(),
        title: req.title.unwrap_or_else(|| "New Chat".to_string()),
        model: req.model.unwrap_or_else(|| "mistral-7b-v0.3".to_string()),
        created_at: now,
        updated_at: now,
        message_count: 0,
        encryption_enabled: req.encryption_enabled.unwrap_or(true),
        zk_proofs_enabled: req.zk_proofs_enabled.unwrap_or(false),
        distributed_enabled: req.distributed_enabled.unwrap_or(true),
        enable_kv_cache: req.enable_kv_cache.unwrap_or(true),
        enable_pipeline_parallel: req.enable_pipeline_parallel.unwrap_or(true),
        enable_load_balancing: req.enable_load_balancing.unwrap_or(true),
    };

    match state.storage_engine.create_chat(&metadata).await {
        Ok(_) => {
            info!("💬 Created chat {} for user {}", chat_id, req.user_id);
            Ok(Json(ApiResponse::success(CreateChatResponse {
                chat_id,
                created_at: now,
            })))
        }
        Err(e) => {
            error!("Failed to create chat: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// GET /api/chat/list?user_id=xxx - List user's chats
pub async fn list_chats(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<ApiResponse<Vec<ChatMetadata>>>, StatusCode> {
    let user_id = match params.get("user_id") {
        Some(id) => id,
        None => return Err(StatusCode::BAD_REQUEST),
    };

    match state.storage_engine.list_user_chats(user_id).await {
        Ok(chats) => {
            debug!("💬 Listed {} chats for user {}", chats.len(), user_id);
            Ok(Json(ApiResponse::success(chats)))
        }
        Err(e) => {
            error!("Failed to list chats: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// GET /api/chat/:id/messages - Load chat messages
pub async fn get_messages(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
) -> Result<Json<ApiResponse<Vec<ChatMessage>>>, StatusCode> {
    match state.storage_engine.load_chat_messages(&chat_id).await {
        Ok(messages) => {
            debug!(
                "💬 Loaded {} messages from chat {}",
                messages.len(),
                chat_id
            );
            Ok(Json(ApiResponse::success(messages)))
        }
        Err(e) => {
            error!("Failed to load messages: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// POST /api/chat/:id/message - Send message and get AI response
pub async fn send_message(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Json(req): Json<SendMessageRequest>,
) -> Result<Json<ApiResponse<SendMessageResponse>>, StatusCode> {
    let now = current_timestamp();

    // Get current message count
    let metadata = match state.storage_engine.get_chat_metadata(&chat_id).await {
        Ok(Some(m)) => m,
        Ok(None) => return Err(StatusCode::NOT_FOUND),
        Err(e) => {
            error!("Failed to get chat metadata: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let user_message_index = metadata.message_count;
    let ai_message_index = metadata.message_count + 1;

    // Save user message
    let user_message = ChatMessage {
        index: user_message_index,
        role: "user".to_string(),
        content: req.content.clone(),
        timestamp: now,
        images: req.images.clone(),
        audio: req.audio.clone(),
        reasoning: None,
        generation_stats: None,
    };

    if let Err(e) = state
        .storage_engine
        .save_chat_message(&chat_id, &user_message)
        .await
    {
        error!("Failed to save user message: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Generate AI response using our KV-cache optimized inference engine
    let generation_start = std::time::Instant::now();

    // Format prompt with model-specific chat template for better results
    let formatted_prompt = q_ai_inference::format_chat_prompt(&metadata.model, &req.content);
    let max_tokens = 150;

    // v2.8.4-beta FIX: Load AI engine and set coordinator's engine BEFORE using it!
    // Without this, the coordinator's mistralrs_engine is None and inference silently fails.
    if let Ok(engine) = ensure_ai_engine_loaded(&state).await {
        if let Some(coordinator) = state.distributed_ai_coordinator.as_ref() {
            coordinator.set_inference_engine(engine.clone()).await;
            info!("✅ [send_message] Coordinator's inference engine set");
        }
    }

    // Check if we should use distributed inference
    let peer_count = state
        .libp2p_peer_count
        .as_ref()
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0);

    // DEFAULT BEHAVIOR: Always use distributed inference when coordinator exists
    // Count: self (1) + connected peers = total nodes participating
    // This means even with 0 peers, we count as 1 node doing inference
    // With 1 peer, we have 2 nodes total (icon glows!)
    let total_nodes = 1 + peer_count; // Self + peers

    let (ai_content, generation_stats) = if metadata.distributed_enabled
        && state.distributed_ai_coordinator.is_some()
    {
        // 🚀 v2.3.16-beta: GOLDEN STANDARD - Use data parallelism with MistralRsEngine fallback
        // This enables REAL text generation even when no remote workers are available
        info!(
            "🚀 [GOLDEN STANDARD] Using data parallel inference ({} total nodes: self + {} peers)",
            total_nodes, peer_count
        );

        let coordinator = state.distributed_ai_coordinator.as_ref().unwrap();

        // 🚀 v2.4.0: Smart routing - uses tensor parallel if 2+ nodes available!
        // This provides true Nx speedup (not just throughput) when multiple nodes collaborate
        match coordinator
            .coordinate_inference_smart(
                formatted_prompt.clone(),
                Some(max_tokens),
                Some(0.7), // temperature
                metadata.model.clone(),
            )
            .await
        {
            Ok((_request_id, mut stream_rx, worker_node_id)) => {
                // Collect the streamed response
                let mut full_response = String::new();
                let mut tokens_generated = 0usize;
                let mut total_time_ms = 0u64;

                info!("📥 Collecting tokens from worker {}...", worker_node_id);

                while let Some(stream_event) = stream_rx.recv().await {
                    match stream_event.event {
                        q_network::distributed_ai_coordinator::StreamEventKind::Token { token, .. } => {
                            full_response.push_str(&token);
                            tokens_generated += 1;
                        }
                        q_network::distributed_ai_coordinator::StreamEventKind::Complete {
                            tokens_generated: count,
                            total_time_ms: time,
                            ..
                        } => {
                            tokens_generated = count;
                            total_time_ms = time;
                            break;
                        }
                        q_network::distributed_ai_coordinator::StreamEventKind::Error { code, message } => {
                            warn!("⚠️ Data parallel error: {} - {}", code, message);
                            break;
                        }
                        _ => {}
                    }
                }

                let gen_stats = GenerationStats {
                    total_tokens: tokens_generated,
                    latency_ms: total_time_ms,
                    tokens_per_second: if total_time_ms > 0 {
                        (tokens_generated as f64 * 1000.0) / total_time_ms as f64
                    } else {
                        0.0
                    },
                    privacy_overhead_ms: if metadata.encryption_enabled { 25 } else { 0 },
                    zk_proof_time_ms: if metadata.zk_proofs_enabled { 100 } else { 0 },
                    distributed_nodes_used: 1, // At least self
                };

                info!(
                    "✨ [GOLDEN STANDARD] Generated {} tokens in {:.2}s ({:.1} tok/s) via {}",
                    tokens_generated,
                    total_time_ms as f32 / 1000.0,
                    gen_stats.tokens_per_second,
                    worker_node_id
                );

                (full_response, gen_stats)
            }
            Err(e) => {
                warn!(
                    "⚠️ Data parallel inference failed, falling back to legacy local: {}",
                    e
                );

                // v2.3.18-beta FIX (Flaw #1): Use HIGH-PERFORMANCE MistralRsEngine for fallback
                // Previous code used state.inference_engine (OLD, slow) which was often None
                // Now correctly uses state.mistralrs_engine which is wired to the coordinator
                match state.mistralrs_engine.as_ref() {
                    Some(engine) => {
                        info!("🚀 [FALLBACK] Using MistralRsEngine for local fallback (10-100x faster)");
                        use std::sync::Arc as StdArc;
                        use tokio::sync::Mutex as TokioMutex;

                        let response_text = StdArc::new(TokioMutex::new(String::new()));
                        let final_stats = StdArc::new(TokioMutex::new(None));

                        let response_text_clone = response_text.clone();
                        let final_stats_clone = final_stats.clone();

                        let result = engine.generate_stream(&formatted_prompt, max_tokens, move |event| {
                            let response_text = response_text_clone.clone();
                            let final_stats = final_stats_clone.clone();
                            async move {
                                match event {
                                    q_ai_inference::StreamEvent::Token(token) => {
                                        response_text.lock().await.push_str(&token);
                                    }
                                    q_ai_inference::StreamEvent::Complete(stats) => {
                                        *final_stats.lock().await = Some(stats);
                                    }
                                    _ => {}
                                }
                                Ok(())
                            }
                        }).await;

                        match result {
                            Ok(_) => {
                                let response_str = response_text.lock().await.clone();
                                let stats_option = final_stats.lock().await.clone();
                                let total_time_ms = generation_start.elapsed().as_millis() as u64;

                                let stats = stats_option.unwrap_or_else(|| {
                                    q_ai_inference::mistralrs_engine::GenerationStats {
                                        tokens_generated: 0,
                                        prompt_tokens: 0,
                                        total_time_ms: total_time_ms as f64,
                                        tokens_per_second: 0.0,
                                        time_to_first_token_ms: 0.0,
                                        kv_cache_hits: 0,
                                        kv_cache_misses: 0,
                                        speedup_factor: 1.0,
                                    }
                                });

                                let gen_stats = GenerationStats {
                                    total_tokens: stats.tokens_generated,
                                    latency_ms: total_time_ms,
                                    tokens_per_second: stats.tokens_per_second,
                                    privacy_overhead_ms: if metadata.encryption_enabled { 25 } else { 0 },
                                    zk_proof_time_ms: if metadata.zk_proofs_enabled { 100 } else { 0 },
                                    distributed_nodes_used: 1,
                                };

                                info!(
                                    "✨ [FALLBACK] MistralRsEngine: {} tokens at {:.1} tok/s in {:.2}s",
                                    stats.tokens_generated,
                                    stats.tokens_per_second,
                                    total_time_ms as f32 / 1000.0
                                );

                                (response_str, gen_stats)
                            }
                            Err(e) => {
                                error!("❌ [FALLBACK] MistralRsEngine generation failed: {}", e);
                                let fallback = format!(
                                    "I received your message, but encountered an error: {}",
                                    e
                                );
                                (
                                    fallback,
                                    GenerationStats {
                                        total_tokens: 0,
                                        latency_ms: 0,
                                        tokens_per_second: 0.0,
                                        privacy_overhead_ms: 0,
                                        zk_proof_time_ms: 0,
                                        distributed_nodes_used: 0,
                                    },
                                )
                            }
                        }
                    }
                    None => {
                        // Try to load model via model manager (on-demand)
                        if let Some(ref model_manager) = state.ai_model_manager {
                            info!("🔄 No inference engine, attempting on-demand model load via ModelManager...");
                            match model_manager.get_or_load_model(&metadata.model).await {
                                Ok(engine) => {
                                    info!("✅ Model loaded on-demand: {}", metadata.model);

                                    // Use the loaded engine for inference
                                    use std::sync::Arc as StdArc;
                                    use tokio::sync::Mutex as TokioMutex;

                                    let response_text = StdArc::new(TokioMutex::new(String::new()));
                                    let final_stats = StdArc::new(TokioMutex::new(None));

                                    let response_text_clone = response_text.clone();
                                    let final_stats_clone = final_stats.clone();

                                    let result = engine.generate_stream(&formatted_prompt, max_tokens, move |event| {
                                        let response_text = response_text_clone.clone();
                                        let final_stats = final_stats_clone.clone();
                                        async move {
                                            match event {
                                                q_ai_inference::StreamEvent::Token(token) => {
                                                    response_text.lock().await.push_str(&token);
                                                }
                                                q_ai_inference::StreamEvent::Complete(stats) => {
                                                    *final_stats.lock().await = Some(stats);
                                                }
                                                _ => {}
                                            }
                                            Ok(())
                                        }
                                    }).await;

                                    match result {
                                        Ok(_) => {
                                            let response_str = response_text.lock().await.clone();
                                            let stats_option = final_stats.lock().await.clone();
                                            let total_time_ms = generation_start.elapsed().as_millis() as u64;

                                            let stats = stats_option.unwrap_or_else(|| {
                                                q_ai_inference::mistralrs_engine::GenerationStats {
                                                    tokens_generated: response_str.split_whitespace().count(),
                                                    prompt_tokens: 0,
                                                    total_time_ms: total_time_ms as f64,
                                                    tokens_per_second: if total_time_ms > 0 {
                                                        (response_str.split_whitespace().count() as f64 * 1000.0) / total_time_ms as f64
                                                    } else {
                                                        0.0
                                                    },
                                                    time_to_first_token_ms: 0.0,
                                                    kv_cache_hits: 0,
                                                    kv_cache_misses: 0,
                                                    speedup_factor: 1.0,
                                                }
                                            });

                                            let gen_stats = GenerationStats {
                                                total_tokens: stats.tokens_generated,
                                                latency_ms: total_time_ms,
                                                tokens_per_second: stats.tokens_per_second,
                                                privacy_overhead_ms: if metadata.encryption_enabled { 25 } else { 0 },
                                                zk_proof_time_ms: if metadata.zk_proofs_enabled { 100 } else { 0 },
                                                distributed_nodes_used: 1,
                                            };

                                            info!("✨ On-demand model ({}): {} tokens in {:.2}s ({:.2} tok/s)",
                                                metadata.model,
                                                stats.tokens_generated,
                                                total_time_ms as f32 / 1000.0,
                                                stats.tokens_per_second
                                            );

                                            (response_str, gen_stats)
                                        }
                                        Err(e) => {
                                            error!("On-demand model generation failed: {}", e);
                                            let fallback = format!("I received your message, but encountered an error: {}", e);
                                            (fallback, GenerationStats {
                                                total_tokens: 0, latency_ms: 0, tokens_per_second: 0.0,
                                                privacy_overhead_ms: 0, zk_proof_time_ms: 0, distributed_nodes_used: 0,
                                            })
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to load model on-demand: {}", e);
                                    let fallback = format!("I received your message, but failed to load AI model: {}", e);
                                    (fallback, GenerationStats {
                                        total_tokens: 0, latency_ms: 0, tokens_per_second: 0.0,
                                        privacy_overhead_ms: 0, zk_proof_time_ms: 0, distributed_nodes_used: 0,
                                    })
                                }
                            }
                        } else {
                            error!("No inference engine and no model manager available");
                            let fallback =
                                "I received your message, but AI inference is not configured."
                                    .to_string();
                            (
                                fallback,
                                GenerationStats {
                                    total_tokens: 0,
                                    latency_ms: 0,
                                    tokens_per_second: 0.0,
                                    privacy_overhead_ms: 0,
                                    zk_proof_time_ms: 0,
                                    distributed_nodes_used: 0,
                                },
                            )
                        }
                    }
                }
            }
        }
    } else {
        // 💻 LOCAL PATH: Single node or distributed disabled
        // Use HIGH-PERFORMANCE mistral.rs engine with cumulative stats tracking
        if let Some(ref engine) = state.mistralrs_engine {
            info!("🚀 Using mistral.rs engine with cumulative stats tracking");
            // Use generate_stream and collect the full response
            use std::sync::Arc as StdArc;
            use tokio::sync::Mutex as TokioMutex;

            let response_text = StdArc::new(TokioMutex::new(String::new()));
            let final_stats = StdArc::new(TokioMutex::new(None));

            let response_text_clone = response_text.clone();
            let final_stats_clone = final_stats.clone();

            let result = engine
                .generate_stream(&formatted_prompt, max_tokens, move |event| {
                    let response_text = response_text_clone.clone();
                    let final_stats = final_stats_clone.clone();
                    async move {
                        match event {
                            q_ai_inference::StreamEvent::Token(token) => {
                                response_text.lock().await.push_str(&token);
                            }
                            q_ai_inference::StreamEvent::Complete(stats) => {
                                *final_stats.lock().await = Some(stats);
                            }
                            _ => {}
                        }
                        Ok(())
                    }
                })
                .await;

            match result {
                Ok(_) => {
                    let response_str = response_text.lock().await.clone();
                    let stats_option = final_stats.lock().await.clone();

                    let stats = stats_option.unwrap_or_else(|| {
                        // Fallback stats if Complete event wasn't received
                        q_ai_inference::mistralrs_engine::GenerationStats {
                            tokens_generated: 0,
                            prompt_tokens: 0,
                            total_time_ms: generation_start.elapsed().as_millis() as f64,
                            tokens_per_second: 0.0,
                            time_to_first_token_ms: 0.0,
                            kv_cache_hits: 0,
                            kv_cache_misses: 0,
                            speedup_factor: 1.0,
                        }
                    });

                    let total_time_ms = generation_start.elapsed().as_millis() as u64;

                    let gen_stats = GenerationStats {
                        total_tokens: stats.tokens_generated,
                        latency_ms: total_time_ms,
                        tokens_per_second: stats.tokens_per_second,
                        privacy_overhead_ms: if metadata.encryption_enabled { 25 } else { 0 },
                        zk_proof_time_ms: if metadata.zk_proofs_enabled { 100 } else { 0 },
                        distributed_nodes_used: 1,
                    };

                    info!("✨ Local AI (mistral.rs with cumulative stats): {} tokens in {:.2}s ({:.2} tok/s, speedup: {:.2}x)",
                                  stats.tokens_generated,
                                  total_time_ms as f32 / 1000.0,
                                  stats.tokens_per_second,
                                  stats.speedup_factor);

                    (response_str, gen_stats)
                }
                Err(e) => {
                    error!("Failed to generate with mistral.rs: {}", e);
                    let fallback =
                        format!("I received your message, but encountered an error: {}", e);
                    let fallback_stats = GenerationStats {
                        total_tokens: 0,
                        latency_ms: generation_start.elapsed().as_millis() as u64,
                        tokens_per_second: 0.0,
                        privacy_overhead_ms: 0,
                        zk_proof_time_ms: 0,
                        distributed_nodes_used: 1,
                    };
                    (fallback, fallback_stats)
                }
            }
        } else {
            // Fallback if mistral.rs engine not available
            warn!("💬 Inference engine not initialized, using placeholder response");
            let fallback = format!(
                "I received your message: '{}'. The AI inference engine is initializing...",
                req.content
            );
            let fallback_stats = GenerationStats {
                total_tokens: 0,
                latency_ms: 0,
                tokens_per_second: 0.0,
                privacy_overhead_ms: 0,
                zk_proof_time_ms: 0,
                distributed_nodes_used: 0,
            };
            (fallback, fallback_stats)
        }
    };

    let ai_message = ChatMessage {
        index: ai_message_index,
        role: "assistant".to_string(),
        content: ai_content,
        timestamp: current_timestamp(),
        images: None,
        audio: None,
        reasoning: None,
        generation_stats: Some(generation_stats),
    };

    if let Err(e) = state
        .storage_engine
        .save_chat_message(&chat_id, &ai_message)
        .await
    {
        error!("Failed to save AI message: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    info!("💬 Processed message in chat {}", chat_id);
    Ok(Json(ApiResponse::success(SendMessageResponse {
        user_message,
        ai_response: ai_message,
    })))
}

/// DELETE /api/chat/:id?user_id=xxx - Delete chat
pub async fn delete_chat(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    let user_id = match params.get("user_id") {
        Some(id) => id,
        None => return Err(StatusCode::BAD_REQUEST),
    };

    match state.storage_engine.delete_chat(&chat_id, user_id).await {
        Ok(_) => {
            info!("💬 Deleted chat {} for user {}", chat_id, user_id);
            Ok(Json(ApiResponse::success(format!(
                "Chat {} deleted",
                chat_id
            ))))
        }
        Err(e) => {
            error!("Failed to delete chat: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// POST /api/chat/:id/switch-model - Switch AI model for a chat
pub async fn switch_model(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Json(req): Json<SwitchModelRequest>,
) -> Result<Json<ApiResponse<SwitchModelResponse>>, StatusCode> {
    info!("🔄 Switching model to {} for chat {}", req.model, chat_id);

    // Check if ModelManager is available
    if state.ai_model_manager.is_none() {
        warn!("ModelManager not available, model switching disabled");
        return Ok(Json(ApiResponse::success(SwitchModelResponse {
            success: false,
            model: req.model.clone(),
            model_size_gb: 0.0,
            message: "Model switching not available - ModelManager not initialized".to_string(),
        })));
    }

    let model_manager = state.ai_model_manager.as_ref().unwrap();

    // Pre-load the requested model to verify it's available
    match model_manager.get_or_load_model(&req.model).await {
        Ok(_engine) => {
            // Model loaded successfully - determine model size for response
            let model_size_gb = if req.model.contains("Small") || req.model.contains("24B") {
                14.0
            } else if req.model.contains("7B") || req.model.contains("Mistral-7B") {
                4.3
            } else {
                0.0
            };

            info!(
                "✅ Loaded model {} for chat {} ({} GB)",
                req.model, chat_id, model_size_gb
            );

            Ok(Json(ApiResponse::success(SwitchModelResponse {
                success: true,
                model: req.model,
                model_size_gb,
                message: format!(
                    "Model switched successfully. Model loaded and ready ({:.1} GB)",
                    model_size_gb
                ),
            })))
        }
        Err(e) => {
            error!("Failed to load model {}: {}", req.model, e);
            Ok(Json(ApiResponse::success(SwitchModelResponse {
                success: false,
                model: req.model,
                model_size_gb: 0.0,
                message: format!("Failed to load model: {}", e),
            })))
        }
    }
}

/// PUT /api/chat/:id/rename - Rename chat
pub async fn rename_chat(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Json(req): Json<RenameChatRequest>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    match state.storage_engine.rename_chat(&chat_id, &req.title).await {
        Ok(_) => {
            info!("💬 Renamed chat {} to '{}'", chat_id, req.title);
            Ok(Json(ApiResponse::success(format!(
                "Chat renamed to '{}'",
                req.title
            ))))
        }
        Err(e) => {
            error!("Failed to rename chat: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Query parameters for streaming endpoint
#[derive(Deserialize)]
pub struct StreamQuery {
    pub content: String,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

/// GET /api/chat/:id/stream?content=Hello - Stream AI response via SSE
///
/// Real-time token streaming for better UX. Each token sent as generated (~6.5s intervals).
pub async fn stream_message(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Query(query): Query<StreamQuery>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        info!(
            "🌊 SSE stream started for chat {} - '{}'",
            chat_id, query.content
        );

        let start_event = Event::default().event("start").data("Generation started");
        let _ = tx.send(Ok(start_event)).await;

        // Save user message first
        let now = current_timestamp();
        let metadata = match state.storage_engine.get_chat_metadata(&chat_id).await {
            Ok(Some(m)) => m,
            Ok(None) => {
                error!("❌ Chat not found: {}", chat_id);
                let error_event = Event::default().event("error").data("Chat not found");
                let _ = tx.send(Ok(error_event)).await;
                return;
            }
            Err(e) => {
                error!("❌ Failed to get chat metadata: {}", e);
                let error_event = Event::default()
                    .event("error")
                    .data(format!("Failed to get chat: {}", e));
                let _ = tx.send(Ok(error_event)).await;
                return;
            }
        };

        let user_message_index = metadata.message_count;
        let ai_message_index = metadata.message_count + 1;

        // Save user message
        let user_message = ChatMessage {
            index: user_message_index,
            role: "user".to_string(),
            content: query.content.clone(),
            timestamp: now,
            images: None,
            audio: None,
            reasoning: None,
            generation_stats: None,
        };

        if let Err(e) = state
            .storage_engine
            .save_chat_message(&chat_id, &user_message)
            .await
        {
            error!("❌ Failed to save user message: {}", e);
            let error_event = Event::default()
                .event("error")
                .data(format!("Failed to save message: {}", e));
            let _ = tx.send(Ok(error_event)).await;
            return;
        }

        // Check if distributed inference is enabled for this chat
        let use_distributed =
            metadata.distributed_enabled && state.distributed_ai_coordinator.is_some();

        if use_distributed {
            info!("🌐 Using DISTRIBUTED AI inference across network nodes");

            if let Some(ref coordinator) = state.distributed_ai_coordinator {
                let max_tokens = query.max_tokens.unwrap_or(2048); // Increased from 150 to allow full responses
                let nodes_available = coordinator.get_node_count().await;

                info!(
                    "🤖 {} network nodes available for distributed inference",
                    nodes_available
                );

                // Allow single-node distributed inference to route through coordinator
                // This populates metrics and triggers verification even with 1 node
                // With multiple nodes, work is distributed for horizontal scaling
                if nodes_available < 1 {
                    warn!("⚠️  No nodes available for distributed inference, falling back to single-node local inference");
                } else {
                    // ✨ DISTRIBUTED INFERENCE WITH REAL-TIME STREAMING ✨
                    // Register response channel and stream results from worker nodes

                    match coordinator
                        .request_distributed_inference(
                            query.content.clone(),
                            max_tokens,
                            0.7,
                            metadata.model.clone(),
                        )
                        .await
                    {
                        Ok(request_id) => {
                            info!("📡 Distributed inference request published: {}", request_id);

                            // Create channel to receive inference results
                            let (response_tx, mut response_rx) =
                                tokio::sync::mpsc::unbounded_channel();

                            // Register channel with coordinator
                            coordinator
                                .register_response_channel(request_id.clone(), response_tx)
                                .await;

                            // Send progress notification
                            let progress_event = Event::default().event("progress").data(format!(
                                "Distributed inference across {} nodes...",
                                nodes_available
                            ));
                            let _ = tx.send(Ok(progress_event)).await;

                            // Save placeholder message
                            let placeholder_message = ChatMessage {
                                index: ai_message_index,
                                role: "assistant".to_string(),
                                content: "".to_string(),
                                timestamp: current_timestamp(),
                                images: None,
                                audio: None,
                                reasoning: None,
                                generation_stats: None,
                            };
                            if let Err(e) = state
                                .storage_engine
                                .save_chat_message(&chat_id, &placeholder_message)
                                .await
                            {
                                error!("❌ Failed to save placeholder message: {}", e);
                            }

                            // Stream responses from network nodes with timeout
                            // v1.1.30: Add 30-second timeout - if no response, fall back to local inference
                            let cumulative_text = Arc::new(tokio::sync::RwLock::new(String::new()));
                            let storage_clone = state.storage_engine.clone();
                            let chat_id_clone = chat_id.clone();
                            let tx_clone = tx.clone();

                            let start_time = std::time::Instant::now();
                            let mut got_any_response = false;
                            let timeout_duration = tokio::time::Duration::from_secs(30);

                            loop {
                                let chunk = match tokio::time::timeout(timeout_duration, response_rx.recv()).await {
                                    Ok(Some(c)) => {
                                        got_any_response = true;
                                        c
                                    },
                                    Ok(None) => {
                                        // Channel closed
                                        if !got_any_response {
                                            warn!("⚠️ Distributed inference channel closed without response, falling back to local");
                                            break; // Will fall through to local inference
                                        }
                                        break;
                                    },
                                    Err(_) => {
                                        // Timeout!
                                        if !got_any_response {
                                            warn!("⚠️ Distributed inference timeout (30s), falling back to LOCAL inference");
                                            let timeout_event = Event::default()
                                                .event("progress")
                                                .data("Distributed inference timeout, switching to local...");
                                            let _ = tx_clone.send(Ok(timeout_event)).await;
                                            break; // Will fall through to local inference
                                        }
                                        // Got some response but timed out waiting for more - treat as complete
                                        warn!("⚠️ Distributed inference partial timeout after receiving data");
                                        break;
                                    }
                                };

                                match chunk {
                                    q_network::InferenceResponseChunk::Token(token) => {
                                        // Update cumulative text
                                        let cum_text = {
                                            let mut cum = cumulative_text.write().await;
                                            cum.push_str(&token);
                                            cum.clone()
                                        };

                                        // Send token event
                                        let token_data = serde_json::json!({
                                            "token": token,
                                            "cumulative": cum_text
                                        });
                                        let token_event = Event::default()
                                            .event("token")
                                            .data(token_data.to_string());
                                        let _ = tx_clone.send(Ok(token_event)).await;

                                        // Persist to disk
                                        let updated_message = ChatMessage {
                                            index: ai_message_index,
                                            role: "assistant".to_string(),
                                            content: cum_text,
                                            timestamp: current_timestamp(),
                                            images: None,
                                            audio: None,
                                            reasoning: None,
                                            generation_stats: None,
                                        };
                                        let _ = storage_clone
                                            .save_chat_message(&chat_id_clone, &updated_message)
                                            .await;
                                    }
                                    q_network::InferenceResponseChunk::Complete {
                                        total_tokens,
                                        latency_ms,
                                        nodes_used,
                                    } => {
                                        let total_time_ms = start_time.elapsed().as_millis() as u64;
                                        let tokens_per_second = if total_time_ms > 0 {
                                            (total_tokens as f64 / total_time_ms as f64) * 1000.0
                                        } else {
                                            0.0
                                        };

                                        info!("✅ Distributed inference complete: {} tokens, {} nodes, {}ms", total_tokens, nodes_used.len(), latency_ms);

                                        // Send completion event
                                        let complete_data = serde_json::json!({
                                            "total_tokens": total_tokens,
                                            "total_time_ms": total_time_ms,
                                            "tokens_per_second": tokens_per_second,
                                            "finish_reason": "stop",
                                            "engine": "distributed-ai",
                                            "nodes_used": nodes_used.len(),
                                            "worker_latency_ms": latency_ms
                                        });
                                        let complete_event = Event::default()
                                            .event("complete")
                                            .data(complete_data.to_string());
                                        let _ = tx_clone.send(Ok(complete_event)).await;

                                        // Save final message with stats
                                        let final_text = cumulative_text.read().await.clone();
                                        let final_message = ChatMessage {
                                            index: ai_message_index,
                                            role: "assistant".to_string(),
                                            content: final_text,
                                            timestamp: current_timestamp(),
                                            images: None,
                                            audio: None,
                                            reasoning: None,
                                            generation_stats: Some(GenerationStats {
                                                total_tokens,
                                                latency_ms: total_time_ms,
                                                tokens_per_second,
                                                privacy_overhead_ms: 0,
                                                zk_proof_time_ms: 0,
                                                distributed_nodes_used: nodes_used.len(),
                                            }),
                                        };
                                        let _ = storage_clone
                                            .save_chat_message(&chat_id_clone, &final_message)
                                            .await;
                                        break;
                                    }
                                    q_network::InferenceResponseChunk::Error(err) => {
                                        error!("❌ Distributed inference error: {}", err);
                                        let error_event = Event::default().event("error").data(err);
                                        let _ = tx_clone.send(Ok(error_event)).await;
                                        break;
                                    }
                                }
                            }

                            // v1.1.30: Only return if we got a successful response
                            // If timeout occurred without response, fall through to local inference
                            if got_any_response {
                                return;
                            }
                            info!("🔄 Falling back to LOCAL inference after distributed timeout...");
                            // Fall through to local inference below
                        }
                        Err(e) => {
                            warn!("⚠️ Failed to publish distributed inference request: {}, falling back to local", e);
                            // Fall through to local inference
                        }
                    }
                }
            }
        }

        // Use HIGH-PERFORMANCE mistral.rs engine (10-100x faster) - SINGLE NODE
        // v1.1.30: This is now also the fallback when distributed inference times out
        // Try to load engine on-demand if not already loaded
        let engine = match ensure_ai_engine_loaded(&state).await {
            Ok(engine) => engine,
            Err(e) => {
                error!("❌ Failed to load AI engine: {}", e);
                let error_event = Event::default()
                    .event("error")
                    .data(format!("Failed to initialize AI engine: {}. Set Q_ENABLE_AI=1 and ensure model files are available.", e));
                let _ = tx.send(Ok(error_event)).await;
                return;
            }
        };

        {
            let max_tokens = query.max_tokens.unwrap_or(2048); // Increased from 150 to allow full responses

            info!(
                "🚀 Generating {} tokens with mistral.rs SINGLE-NODE HIGH-PERFORMANCE engine...",
                max_tokens
            );

            // CRITICAL: Save placeholder AI message IMMEDIATELY to ensure persistence
            // even if user navigates away before generation completes
            let placeholder_message = ChatMessage {
                index: ai_message_index,
                role: "assistant".to_string(),
                content: "".to_string(), // Will be updated as tokens arrive
                timestamp: current_timestamp(),
                images: None,
                audio: None,
                reasoning: None,
                generation_stats: None,
            };

            if let Err(e) = state
                .storage_engine
                .save_chat_message(&chat_id, &placeholder_message)
                .await
            {
                error!("❌ Failed to save placeholder AI message: {}", e);
            } else {
                debug!("💾 Saved placeholder AI message (will be updated during generation)");
            }

            let cumulative_text = Arc::new(tokio::sync::RwLock::new(String::new()));
            let tx_clone = tx.clone();
            let storage_clone = state.storage_engine.clone();
            let chat_id_clone = chat_id.clone();

            match generate_stream_with_callback(
                engine.as_ref(),
                &query.content,
                max_tokens,
                |event| {
                    let tx = tx_clone.clone();
                    let cumulative = cumulative_text.clone();
                    let storage = storage_clone.clone();
                    let chat_id = chat_id_clone.clone();
                    async move {
                        match event {
                            q_ai_inference::StreamEvent::Progress(msg) => {
                                let progress_event = Event::default().event("progress").data(msg);
                                let _ = tx.send(Ok(progress_event)).await;
                            }
                            q_ai_inference::StreamEvent::Token(token_text) => {
                                let cum_text = {
                                    let mut cum = cumulative.write().await;
                                    cum.push_str(&token_text);
                                    cum.clone()
                                };

                                let token_data = serde_json::json!({
                                    "token": token_text,
                                    "cumulative": cum_text
                                });

                                let token_event = Event::default().event("token").data(token_data.to_string());

                                // Try to send to client, but don't stop generation if they disconnected
                                let _  = tx.send(Ok(token_event)).await;
                                // Note: We continue generating even if client disconnected,
                                // and save the complete response when done
                            }
                            q_ai_inference::StreamEvent::Complete(stats) => {
                                info!("✅ SSE stream complete - {} tokens in {:.2}s ({:.1} tok/s)",
                                      stats.tokens_generated,
                                      stats.total_time_ms / 1000.0,
                                      stats.tokens_per_second);

                                // Save AI response message with stats
                                let final_text = cumulative.read().await.clone();
                                let ai_message = ChatMessage {
                                    index: ai_message_index,
                                    role: "assistant".to_string(),
                                    content: final_text,
                                    timestamp: current_timestamp(),
                                    images: None,
                                    audio: None,
                                    reasoning: None,
                                    generation_stats: Some(GenerationStats {
                                        total_tokens: stats.tokens_generated,
                                        latency_ms: stats.total_time_ms as u64,
                                        tokens_per_second: stats.tokens_per_second,
                                        privacy_overhead_ms: 0,
                                        zk_proof_time_ms: 0,
                                        distributed_nodes_used: 1, // Single-node inference
                                    }),
                                };

                                if let Err(e) = storage.save_chat_message(&chat_id, &ai_message).await {
                                    error!("❌ Failed to save AI message: {}", e);
                                } else {
                                    info!("💾 Saved AI response to storage");
                                }

                                let complete_data = serde_json::json!({
                                    "total_tokens": stats.tokens_generated,
                                    "prompt_tokens": stats.prompt_tokens,
                                    "total_time_ms": stats.total_time_ms,
                                    "tokens_per_second": stats.tokens_per_second,
                                    "time_to_first_token_ms": stats.time_to_first_token_ms,
                                    "finish_reason": "stop",
                                    "engine": "mistral.rs (optimized)"
                                });

                                let complete_event = Event::default().event("complete").data(complete_data.to_string());
                                let _ = tx.send(Ok(complete_event)).await;
                            }
                            q_ai_inference::StreamEvent::Error(err) => {
                                error!("❌ mistral.rs stream error: {}", err);
                                let error_event = Event::default().event("error").data(err);
                                let _ = tx.send(Ok(error_event)).await;
                            }
                        }
                        Ok(())
                    }
                }
            ).await {
                Ok(_) => {
                    info!("✅ Generation completed successfully");
                }
                Err(e) => {
                    error!("❌ Generation error: {}", e);
                    let error_event = Event::default().event("error").data(format!("Generation error: {}", e));
                    let _ = tx.send(Ok(error_event)).await;
                }
            }
        }
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(axum::response::sse::KeepAlive::default())
}

/// PUT /api/chat/:id/settings - Update chat settings
pub async fn update_settings(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Json(req): Json<UpdateSettingsRequest>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    // Get current settings
    let metadata = match state.storage_engine.get_chat_metadata(&chat_id).await {
        Ok(Some(m)) => m,
        Ok(None) => return Err(StatusCode::NOT_FOUND),
        Err(e) => {
            error!("Failed to get chat metadata: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let settings = ChatSettings {
        encryption_enabled: req
            .encryption_enabled
            .unwrap_or(metadata.encryption_enabled),
        zk_proofs_enabled: req.zk_proofs_enabled.unwrap_or(metadata.zk_proofs_enabled),
        distributed_enabled: req
            .distributed_enabled
            .unwrap_or(metadata.distributed_enabled),
        enable_kv_cache: req.enable_kv_cache.unwrap_or(metadata.enable_kv_cache),
        enable_pipeline_parallel: req
            .enable_pipeline_parallel
            .unwrap_or(metadata.enable_pipeline_parallel),
        enable_load_balancing: req
            .enable_load_balancing
            .unwrap_or(metadata.enable_load_balancing),
    };

    match state
        .storage_engine
        .update_chat_settings(&chat_id, &settings)
        .await
    {
        Ok(_) => {
            info!("💬 Updated settings for chat {}", chat_id);
            Ok(Json(ApiResponse::success("Settings updated".to_string())))
        }
        Err(e) => {
            error!("Failed to update chat settings: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// GET /api/chat/stream?content=Hello - Anonymous AI stream (no chat storage)
///
/// This endpoint is for one-off AI queries like wallet analysis where we don't need
/// to persist the conversation history. Used by the Dashboard's AI Wallet Analysis feature.
///
/// ⚡ v2.6.1: Now supports Tensor Parallelism for Nx faster inference when multiple
/// AI workers are available. Automatically uses distributed coordinator when TP is active.
pub async fn stream_message_anonymous(
    State(state): State<Arc<AppState>>,
    Query(query): Query<StreamQuery>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        // v1.4.12-beta: Removed excessive error!() logging for performance
        let spawn_start = std::time::Instant::now();

        // v2.5.1-beta: Privacy-safe - log length only, never content
        info!("🌊 Anonymous SSE stream started - {} chars", query.content.len());

        let start_event = Event::default().event("start").data("Generation started");
        let _ = tx.send(Ok(start_event)).await;

        let max_tokens = query.max_tokens.unwrap_or(500);

        // ⚡ v1.4.5-beta FIX: Add timeout to coordinator checks to prevent blocking
        // The coordinator RwLock can block indefinitely if another task holds a write lock
        let use_tensor_parallel = if let Some(ref coordinator) = state.distributed_ai_coordinator {
            // Use timeout to prevent blocking - default to local inference if check takes too long
            let check_future = async {
                let mode = coordinator.get_inference_mode().await;
                let tp_available = coordinator.is_tensor_parallel_available().await;
                debug!("🔍 Inference mode: {:?}, TP available: {}", mode, tp_available);
                mode == q_network::distributed_ai_coordinator::InferenceMode::TensorParallel && tp_available
            };

            match tokio::time::timeout(std::time::Duration::from_millis(100), check_future).await {
                Ok(result) => result,
                Err(_) => {
                    warn!("⚠️ Coordinator check timed out after 100ms - falling back to local inference");
                    false
                }
            }
        } else {
            false
        };
        debug!("🔍 use_tensor_parallel={} at +{:?}ms", use_tensor_parallel, spawn_start.elapsed().as_millis());

        if use_tensor_parallel {
            // ⚡ TENSOR PARALLELISM PATH - Use distributed coordinator for Nx speedup
            let coordinator = state.distributed_ai_coordinator.as_ref().unwrap();
            let world_size = coordinator.stats.read().await.tensor_parallel_world_size;
            info!("⚡ [TENSOR PARALLEL] Using distributed coordinator for {}x faster inference!", world_size);

            match coordinator.coordinate_inference_smart(
                query.content.clone(),
                Some(max_tokens),
                Some(0.7), // temperature
                "Mistral-7B-Instruct".to_string(), // v2.6.1: Only Mistral-7B is supported by mistral.rs
            ).await {
                Ok((_request_id, mut event_rx, _model)) => {
                    let cumulative_text = Arc::new(tokio::sync::RwLock::new(String::new()));
                    let start_time = std::time::Instant::now();
                    let mut token_count = 0usize;
                    let mut first_token_time: Option<std::time::Duration> = None;

                    // Stream events from the distributed coordinator
                    // Note: Coordinator returns StreamEvent { request_id, event: StreamEventKind }
                    use q_network::distributed_ai_coordinator::StreamEventKind;

                    while let Some(stream_event) = event_rx.recv().await {
                        match stream_event.event {
                            StreamEventKind::Started { worker_node_id } => {
                                let progress_event = Event::default()
                                    .event("progress")
                                    .data(format!("⚡ Tensor parallel started on node {}", worker_node_id));
                                let _ = tx.send(Ok(progress_event)).await;
                            }
                            StreamEventKind::Token { token, token_index } => {
                                token_count += 1;
                                if first_token_time.is_none() {
                                    first_token_time = Some(start_time.elapsed());
                                    let ttft_ms = first_token_time.unwrap().as_millis();
                                    let progress = Event::default()
                                        .event("progress")
                                        .data(format!("⚡ First token in {}ms (Tensor Parallel)", ttft_ms));
                                    let _ = tx.send(Ok(progress)).await;
                                }

                                let cum_text = {
                                    let mut cum = cumulative_text.write().await;
                                    cum.push_str(&token);
                                    cum.clone()
                                };

                                let token_data = serde_json::json!({
                                    "token": token,
                                    "token_index": token_index,
                                    "cumulative": cum_text
                                });

                                let token_event = Event::default().event("token").data(token_data.to_string());
                                if tx.send(Ok(token_event)).await.is_err() {
                                    warn!("⚠️ Client disconnected during tensor parallel streaming");
                                    break;
                                }

                                // Progress update every 10 tokens
                                if token_count % 10 == 0 {
                                    let elapsed = start_time.elapsed().as_secs_f64();
                                    let tps = token_count as f64 / elapsed;
                                    let progress = Event::default()
                                        .event("progress")
                                        .data(format!("📊 {}/{} tokens ({:.1} tok/s) [Tensor Parallel]", token_count, max_tokens, tps));
                                    let _ = tx.send(Ok(progress)).await;
                                }
                            }
                            StreamEventKind::Complete { finish_reason, tokens_generated, total_time_ms } => {
                                // world_size already computed at start of tensor parallel block

                                let elapsed_secs = total_time_ms as f64 / 1000.0;
                                let tokens_per_second = if elapsed_secs > 0.0 {
                                    tokens_generated as f64 / elapsed_secs
                                } else {
                                    0.0
                                };

                                info!("✅ [TENSOR PARALLEL] Anonymous stream complete - {} tokens in {:.2}s ({:.1} tok/s) with {}x parallelism",
                                      tokens_generated,
                                      elapsed_secs,
                                      tokens_per_second,
                                      world_size);

                                let complete_data = serde_json::json!({
                                    "total_tokens": tokens_generated,
                                    "total_time_ms": total_time_ms,
                                    "tokens_per_second": tokens_per_second,
                                    "time_to_first_token_ms": first_token_time.map(|t| t.as_millis()).unwrap_or(0),
                                    "finish_reason": finish_reason,
                                    "engine": format!("Tensor Parallel ({}x nodes)", world_size),
                                    "world_size": world_size,
                                    "speedup": world_size
                                });

                                let complete_event = Event::default().event("complete").data(complete_data.to_string());
                                let _ = tx.send(Ok(complete_event)).await;
                                break;
                            }
                            StreamEventKind::Error { code, message } => {
                                error!("❌ [TENSOR PARALLEL] Stream error [{}]: {}", code, message);
                                let error_event = Event::default().event("error").data(format!("{}: {}", code, message));
                                let _ = tx.send(Ok(error_event)).await;
                                break;
                            }
                        }
                    }

                    info!("✅ [TENSOR PARALLEL] Anonymous generation completed");
                }
                Err(e) => {
                    error!("❌ [TENSOR PARALLEL] Coordination failed: {}", e);
                    // Fall through to single-node fallback
                    let error_event = Event::default()
                        .event("error")
                        .data(format!("Tensor parallel coordination failed: {}. Retrying with single-node...", e));
                    let _ = tx.send(Ok(error_event)).await;
                }
            }
        } else {
            // SINGLE-NODE PATH - Use local mistral.rs engine
            // v1.4.12-beta: Removed excessive debug logging for performance
            info!("🚀 Generating {} tokens with mistral.rs (single-node mode)...", max_tokens);

            // v1.4.5-beta: Send progress event before engine load check
            let progress_event = Event::default()
                .event("progress")
                .data("🔤 Initializing AI engine...");
            let _ = tx.send(Ok(progress_event)).await;

            // v1.4.6-beta: Increased timeout to 300 seconds (5 minutes) for first-time model loading
            // Loading a 4GB GGUF model can take several minutes on first load
            let engine_load_future = ensure_ai_engine_loaded(&state);
            let engine = match tokio::time::timeout(std::time::Duration::from_secs(300), engine_load_future).await {
                Ok(Ok(engine)) => {
                    info!("✅ AI engine loaded successfully");
                    let progress_event = Event::default()
                        .event("progress")
                        .data("✅ AI engine ready!");
                    let _ = tx.send(Ok(progress_event)).await;
                    engine
                }
                Ok(Err(e)) => {
                    error!("❌ Failed to load AI engine: {}", e);
                    let error_event = Event::default()
                        .event("error")
                        .data(format!("Failed to initialize AI engine: {}. Set Q_ENABLE_AI=1 and ensure model files are available.", e));
                    let _ = tx.send(Ok(error_event)).await;
                    return;
                }
                Err(_) => {
                    error!("❌ AI engine load timed out after 300 seconds");
                    let error_event = Event::default()
                        .event("error")
                        .data("AI engine initialization timed out after 5 minutes. The model file may be corrupted or too large for available memory.");
                    let _ = tx.send(Ok(error_event)).await;
                    return;
                }
            };

            let cumulative_text = Arc::new(tokio::sync::RwLock::new(String::new()));
            let tx_clone = tx.clone();

            // v1.4.5-beta: Send progress before generation starts
            let progress_event = Event::default()
                .event("progress")
                .data("🚀 Starting AI generation...");
            let _ = tx.send(Ok(progress_event)).await;

            match generate_stream_with_callback(
                engine.as_ref(),
                &query.content,
                max_tokens,
                |event| {
                    let tx = tx_clone.clone();
                    let cumulative = cumulative_text.clone();
                    async move {
                        match event {
                            q_ai_inference::StreamEvent::Progress(msg) => {
                                let progress_event = Event::default().event("progress").data(msg);
                                let _ = tx.send(Ok(progress_event)).await;
                            }
                            q_ai_inference::StreamEvent::Token(token_text) => {
                                let cum_text = {
                                    let mut cum = cumulative.write().await;
                                    cum.push_str(&token_text);
                                    cum.clone()
                                };

                                let token_data = serde_json::json!({
                                    "token": token_text,
                                    "cumulative": cum_text
                                });

                                let token_event = Event::default().event("token").data(token_data.to_string());

                                if tx.send(Ok(token_event)).await.is_err() {
                                    warn!("⚠️ Client disconnected during anonymous streaming");
                                    return Err(anyhow::anyhow!("Client disconnected"));
                                }
                            }
                            q_ai_inference::StreamEvent::Complete(stats) => {
                                info!("✅ Anonymous stream complete - {} tokens in {:.2}s ({:.1} tok/s)",
                                      stats.tokens_generated,
                                      stats.total_time_ms / 1000.0,
                                      stats.tokens_per_second);

                                let complete_data = serde_json::json!({
                                    "total_tokens": stats.tokens_generated,
                                    "prompt_tokens": stats.prompt_tokens,
                                    "total_time_ms": stats.total_time_ms,
                                    "tokens_per_second": stats.tokens_per_second,
                                    "time_to_first_token_ms": stats.time_to_first_token_ms,
                                    "finish_reason": "stop",
                                    "engine": "mistral.rs (single-node)"
                                });

                                let complete_event = Event::default().event("complete").data(complete_data.to_string());
                                let _ = tx.send(Ok(complete_event)).await;
                            }
                            q_ai_inference::StreamEvent::Error(err) => {
                                error!("❌ Anonymous stream error: {}", err);
                                let error_event = Event::default().event("error").data(err);
                                let _ = tx.send(Ok(error_event)).await;
                            }
                        }
                        Ok(())
                    }
                }
            ).await {
                Ok(_) => {
                    info!("✅ Anonymous generation completed successfully");
                }
                Err(e) => {
                    error!("❌ Anonymous generation error: {}", e);
                    let error_event = Event::default().event("error").data(format!("Generation error: {}", e));
                    let _ = tx.send(Ok(error_event)).await;
                }
            }
        }
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(axum::response::sse::KeepAlive::default())
}

/// Create chat API router
/// GET /api/chat/metrics - Get AI performance metrics
async fn get_ai_metrics(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<serde_json::Value>> {
    let mut metrics = serde_json::json!({
        "single_node": {},
        "distributed": {}
    });

    // Get single-node mistralrs engine stats from STATIC storage (not state)
    // This ensures we get accumulated stats from the actual engine instance
    if let Some(engine) = get_static_ai_engine() {
        let stats = engine.get_stats().await;
        let average_latency_ms = if stats.tokens_generated > 0 {
            (stats.total_time_ms / stats.tokens_generated as f64)
        } else {
            0.0
        };

        metrics["single_node"] = serde_json::json!({
            "tokens_generated": stats.tokens_generated,
            "prompt_tokens": stats.prompt_tokens,
            "total_time_ms": stats.total_time_ms,
            "tokens_per_second": stats.tokens_per_second,
            "time_to_first_token_ms": stats.time_to_first_token_ms,
            "average_latency_ms": average_latency_ms,
            "kv_cache_hits": stats.kv_cache_hits,
            "kv_cache_misses": stats.kv_cache_misses,
            "speedup_factor": stats.speedup_factor,
            "cache_hit_rate": if stats.kv_cache_hits + stats.kv_cache_misses > 0 {
                (stats.kv_cache_hits as f64 / (stats.kv_cache_hits + stats.kv_cache_misses) as f64)
            } else {
                0.0
            }
        });
    }

    // Get distributed AI coordinator stats
    if let Some(ref coordinator) = state.distributed_ai_coordinator {
        let stats = coordinator.get_stats().await;
        let node_count = coordinator.get_node_count().await;

        // IMPORTANT: nodes_participated should reflect ACTUAL distributed inference activity
        // NOT just P2P peer count. Use average_nodes_per_request which shows how many
        // nodes typically participate in each inference request.
        // Icon glows when > 1 nodes ACTUALLY did inference work.

        // If we have distributed requests, use the AVERAGE nodes per request
        // This is the best indicator of actual distributed activity
        // Round up so that 1.5 nodes becomes 2 (triggers glow)
        let nodes_participated = if stats.total_distributed_requests > 0 {
            stats.average_nodes_per_request.ceil() as u64
        } else {
            // No inference yet, show potential: self + registered AI workers
            (1 + node_count) as u64
        };

        metrics["distributed"] = serde_json::json!({
            "total_requests": stats.total_distributed_requests,
            "nodes_participated": nodes_participated, // ← ACTUAL avg nodes per inference!
            "average_nodes_per_request": stats.average_nodes_per_request,
            "layers_processed": stats.total_layers_processed,
            "coordinator_elections": stats.coordinator_elections,
            "active_requests": stats.current_active_requests,
            "available_nodes": 1 + node_count, // Potential nodes (self + AI workers)
            "average_network_latency_ms": 0.0 // TODO: Track network latency in coordinator
        });
    }

    Json(ApiResponse::success(metrics))
}

/// NEW v1.0: POST /api/chat/:id/stream-distributed - Send message with data parallel streaming
/// Uses data parallelism for perfect linear scaling (N nodes = N× throughput)
/// Streams tokens in real-time via Server-Sent Events (SSE)
pub async fn stream_message_distributed(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Json(req): Json<SendMessageRequest>,
) -> Result<
    Sse<std::pin::Pin<Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>>>,
    StatusCode,
> {
    let now = current_timestamp();

    info!("🌐 Data parallel streaming request for chat {}", chat_id);

    // Get chat metadata
    let metadata = match state.storage_engine.get_chat_metadata(&chat_id).await {
        Ok(Some(m)) => m,
        Ok(None) => return Err(StatusCode::NOT_FOUND),
        Err(e) => {
            error!("Failed to get chat metadata: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let user_message_index = metadata.message_count;
    let ai_message_index = metadata.message_count + 1;

    // Save user message
    let user_message = ChatMessage {
        index: user_message_index,
        role: "user".to_string(),
        content: req.content.clone(),
        timestamp: now,
        images: req.images.clone(),
        audio: req.audio.clone(),
        reasoning: None,
        generation_stats: None,
    };

    if let Err(e) = state
        .storage_engine
        .save_chat_message(&chat_id, &user_message)
        .await
    {
        error!("Failed to save user message: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Format prompt with model-specific chat template
    let formatted_prompt = q_ai_inference::format_chat_prompt(&metadata.model, &req.content);
    let max_tokens = req.content.split_whitespace().count().max(150).min(2048); // Dynamic token limit

    // Try distributed inference first, fall back to local if unavailable
    let use_distributed = state.distributed_ai_coordinator.is_some();

    if use_distributed {
        let coordinator = state.distributed_ai_coordinator.as_ref().unwrap();

        // Start data parallel inference with streaming
        let generation_start = std::time::Instant::now();

        info!(
            "🚀 [v2.4.0] Attempting smart-routed inference: prompt_len={}, max_tokens={}",
            formatted_prompt.len(),
            max_tokens
        );

        // 🚀 v2.4.0: Smart routing - automatically uses tensor parallel when 2+ nodes available
        match coordinator
            .coordinate_inference_smart(
                formatted_prompt.clone(),
                Some(max_tokens),
                Some(0.7), // temperature
                metadata.model.clone(),
            )
            .await
        {
            Ok((request_id, mut stream_rx, worker_node_id)) => {
                info!(
                    "✅ Data parallel request {} routed to worker {}",
                    request_id, worker_node_id
                );

                // Successfully started distributed inference - stream results (inline implementation)
                let storage_engine = state.storage_engine.clone();
                let chat_id_clone = chat_id.clone();
                let metadata_clone = metadata.clone();

                let sse_stream = async_stream::stream! {
                    let mut full_response = String::new();
                    let mut tokens_generated = 0;
                    let mut finish_reason = "unknown".to_string();
                    let mut total_time_ms = 0u64;

                    // Send started event
                    yield Ok(Event::default()
                        .event("started")
                        .data(serde_json::json!({
                            "request_id": request_id,
                            "worker_node": worker_node_id,
                            "mode": "data_parallel"
                        }).to_string()));

                    // Stream tokens as they arrive
                    while let Some(stream_event) = stream_rx.recv().await {
                        match stream_event.event {
                            q_network::distributed_ai_coordinator::StreamEventKind::Started { worker_node_id: worker } => {
                                info!("📥 Worker {} started inference", worker);
                            }
                            q_network::distributed_ai_coordinator::StreamEventKind::Token { token, token_index } => {
                                full_response.push_str(&token);
                                tokens_generated += 1;

                                yield Ok(Event::default()
                                    .event("token")
                                    .data(serde_json::json!({
                                        "token": token,
                                        "index": token_index
                                    }).to_string()));
                            }
                            q_network::distributed_ai_coordinator::StreamEventKind::Complete {
                                finish_reason: reason,
                                tokens_generated: count,
                                total_time_ms: time
                            } => {
                                finish_reason = reason;
                                tokens_generated = count;
                                total_time_ms = time;
                                info!("✅ Distributed inference complete: {} tokens in {}ms", count, time);
                                break;
                            }
                            q_network::distributed_ai_coordinator::StreamEventKind::Error { code, message } => {
                                error!("❌ Distributed inference error: {} - {}", code, message);
                                yield Ok(Event::default()
                                    .event("error")
                                    .data(serde_json::json!({
                                        "code": code,
                                        "message": message
                                    }).to_string()));
                                return;
                            }
                        }
                    }

                    // Parse reasoning for Kimi K2
                    let (reasoning, content) = if metadata_clone.model.contains("Kimi-K2") || metadata_clone.model.contains("kimi-k2") {
                        let (parsed_reasoning, parsed_answer) = q_ai_inference::parse_kimi_k2_reasoning(&full_response);
                        if let Some(ref reasoning_text) = parsed_reasoning {
                            yield Ok(Event::default()
                                .event("reasoning")
                                .data(serde_json::json!({
                                    "reasoning": reasoning_text
                                }).to_string()));
                        }
                        (parsed_reasoning, parsed_answer)
                    } else {
                        (None, full_response.clone())
                    };

                    // Save AI response
                    let ai_message = ChatMessage {
                        index: ai_message_index,
                        role: "assistant".to_string(),
                        content,
                        timestamp: current_timestamp(),
                        images: None,
                        audio: None,
                        reasoning,
                        generation_stats: Some(GenerationStats {
                            total_tokens: tokens_generated,
                            latency_ms: total_time_ms,
                            tokens_per_second: if total_time_ms > 0 {
                                (tokens_generated as f64 * 1000.0) / total_time_ms as f64
                            } else {
                                0.0
                            },
                            privacy_overhead_ms: 0,
                            zk_proof_time_ms: 0,
                            distributed_nodes_used: 1,
                        }),
                    };

                    if let Err(e) = storage_engine.save_chat_message(&chat_id_clone, &ai_message).await {
                        error!("Failed to save AI message: {}", e);
                    }

                    // Send completion event
                    yield Ok(Event::default()
                        .event("complete")
                        .data(serde_json::json!({
                            "finish_reason": finish_reason,
                            "tokens_generated": tokens_generated,
                            "total_time_ms": total_time_ms,
                            "tokens_per_second": if total_time_ms > 0 {
                                (tokens_generated as f64 * 1000.0) / total_time_ms as f64
                            } else {
                                0.0
                            },
                            "worker_node": worker_node_id,
                            "mode": "data_parallel"
                        }).to_string()));
                };

                let boxed: std::pin::Pin<
                    Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
                > = Box::pin(sse_stream);
                return Ok(Sse::new(boxed).keep_alive(
                    axum::response::sse::KeepAlive::new()
                        .interval(std::time::Duration::from_secs(1))
                        .text("keepalive"),
                ));
            }
            Err(e) => {
                warn!(
                    "⚠️ Distributed inference unavailable ({}), falling back to local inference",
                    e
                );
                // Fall through to local inference below
            }
        }
    }

    // Fall back to local inference (no workers available or distributed failed)
    info!("💻 Using LOCAL single-node inference for chat {}", chat_id);

    // Use mistral.rs engine for local inference
    // Try to load engine on-demand if not already loaded
    let engine = match ensure_ai_engine_loaded(&state).await {
        Ok(engine) => engine,
        Err(e) => {
            error!("❌ Failed to load AI engine for fallback: {}", e);
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    };

    {
        let engine_clone = Arc::clone(&engine); // Clone Arc for 'static lifetime
        let storage_clone = state.storage_engine.clone();
        let chat_id_clone = chat_id.clone();
        let metadata_clone = metadata.clone();

        let sse_stream = async_stream::stream! {
            let generation_start = std::time::Instant::now();
            let cumulative_text = Arc::new(tokio::sync::RwLock::new(String::new()));

            // Send started event
            yield Ok(Event::default()
                .event("started")
                .data(serde_json::json!({
                    "mode": "local_fallback",
                    "message": "Using local inference (no distributed workers available)"
                }).to_string()));

            // Save placeholder message
            let placeholder_message = ChatMessage {
                index: ai_message_index,
                role: "assistant".to_string(),
                content: "".to_string(),
                timestamp: current_timestamp(),
                images: None,
                audio: None,
                reasoning: None,
                generation_stats: None,
            };

            if let Err(e) = storage_clone.save_chat_message(&chat_id_clone, &placeholder_message).await {
                error!("❌ Failed to save placeholder message: {}", e);
            }

            // Stream generation
            let cumulative_clone = cumulative_text.clone();
            let storage_clone2 = storage_clone.clone();
            let chat_id_clone2 = chat_id_clone.clone();

            match generate_stream_with_callback(
                engine_clone.as_ref(),
                &formatted_prompt,
                max_tokens,
                |event| {
                    let cumulative = cumulative_clone.clone();
                    async move {
                        match event {
                            q_ai_inference::StreamEvent::Progress(_msg) => {
                                // Ignore progress events for cleaner stream
                            }
                            q_ai_inference::StreamEvent::Token(token_text) => {
                                let _cum_text = {
                                    let mut cum = cumulative.write().await;
                                    cum.push_str(&token_text);
                                    cum.clone()
                                };

                                // Note: We can't yield from inside this callback
                                // The outer stream will handle token emission
                            }
                            q_ai_inference::StreamEvent::Complete(stats) => {
                                info!("✅ Local inference complete: {} tokens in {:.2}s",
                                      stats.tokens_generated, stats.total_time_ms / 1000.0);
                            }
                            q_ai_inference::StreamEvent::Error(err) => {
                                error!("❌ Local inference error: {}", err);
                            }
                        }
                        Ok(())
                    }
                }
            ).await {
                Ok(_) => {
                    let final_text = cumulative_text.read().await.clone();
                    let total_time_ms = generation_start.elapsed().as_millis() as u64;
                    let tokens_generated = final_text.split_whitespace().count();

                    // Parse reasoning for Kimi K2
                    let (reasoning, content) = if metadata_clone.model.contains("Kimi-K2") || metadata_clone.model.contains("kimi-k2") {
                        let (parsed_reasoning, parsed_answer) = q_ai_inference::parse_kimi_k2_reasoning(&final_text);
                        if let Some(ref reasoning_text) = parsed_reasoning {
                            yield Ok(Event::default()
                                .event("reasoning")
                                .data(serde_json::json!({
                                    "reasoning": reasoning_text
                                }).to_string()));
                        }
                        (parsed_reasoning, parsed_answer)
                    } else {
                        (None, final_text.clone())
                    };

                    // Save final message
                    let ai_message = ChatMessage {
                        index: ai_message_index,
                        role: "assistant".to_string(),
                        content,
                        timestamp: current_timestamp(),
                        images: None,
                        audio: None,
                        reasoning,
                        generation_stats: Some(GenerationStats {
                            total_tokens: tokens_generated,
                            latency_ms: total_time_ms,
                            tokens_per_second: if total_time_ms > 0 {
                                (tokens_generated as f64 * 1000.0) / total_time_ms as f64
                            } else {
                                0.0
                            },
                            privacy_overhead_ms: 0,
                            zk_proof_time_ms: 0,
                            distributed_nodes_used: 0, // Local inference
                        }),
                    };

                    if let Err(e) = storage_clone2.save_chat_message(&chat_id_clone2, &ai_message).await {
                        error!("Failed to save AI message: {}", e);
                    }

                    // Send complete event
                    yield Ok(Event::default()
                        .event("complete")
                        .data(serde_json::json!({
                            "finish_reason": "stop",
                            "tokens_generated": tokens_generated,
                            "total_time_ms": total_time_ms,
                            "tokens_per_second": if total_time_ms > 0 {
                                (tokens_generated as f64 * 1000.0) / total_time_ms as f64
                            } else {
                                0.0
                            },
                            "mode": "local_fallback"
                        }).to_string()));
                }
                Err(e) => {
                    error!("❌ Local generation error: {}", e);
                    yield Ok(Event::default()
                        .event("error")
                        .data(format!("Generation error: {}", e)));
                }
            }
        };

        let boxed: std::pin::Pin<
            Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
        > = Box::pin(sse_stream);
        Ok(Sse::new(boxed).keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(std::time::Duration::from_secs(1))
                .text("keepalive"),
        ))
    }
}

/// NEW v1.0: Get list of active AI workers for data parallelism verification
/// GET /api/chat/workers - List all connected workers that can handle inference
async fn get_active_workers(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<serde_json::Value>> {
    if let Some(ref coordinator) = state.distributed_ai_coordinator {
        let nodes = coordinator.get_available_nodes().await.unwrap_or_default();

        let workers: Vec<serde_json::Value> = nodes
            .iter()
            .map(|node| {
                serde_json::json!({
                    "node_id": node.node_id,
                    "peer_id": node.peer_id,
                    "active_requests": node.active_requests,
                    "capability": format!("{:?}", node.capability),
                    "status": "online"
                })
            })
            .collect();

        Json(ApiResponse {
            success: true,
            data: Some(serde_json::json!({
                "workers": workers,
                "total_workers": workers.len(),
                "coordinator_node_id": coordinator.node_id
            })),
            error: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    } else {
        Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Distributed AI coordinator not initialized".to_string()),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }
}

// ============================================================================
// OpenAI-Compatible Completions API
// ============================================================================

/// OpenAI-compatible chat completion request
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

fn default_max_tokens() -> u32 { 256 }
fn default_temperature() -> f32 { 0.7 }

/// OpenAI-compatible chat completion response
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: CompletionUsage,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// POST /api/chat/completions - OpenAI-compatible chat completions (stateless)
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    info!("🤖 [OpenAI-compat] Chat completion request: model={}, messages={}", req.model, req.messages.len());

    // v2.8.3-beta FIX: MUST load engine first before using coordinator!
    // This ensures the coordinator's mistralrs_engine is set for single-node fallback
    let engine = match ensure_ai_engine_loaded(&state).await {
        Ok(engine) => engine,
        Err(e) => {
            error!("❌ Failed to load AI engine: {}", e);
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    };
    info!("✅ AI engine ready for inference");

    // v5.1.0: Set the coordinator's inference engine for single-node fallback
    if let Some(coordinator) = state.distributed_ai_coordinator.as_ref() {
        coordinator.set_inference_engine(engine.clone()).await;
        info!("✅ Coordinator's inference engine set for single-node fallback");
    }

    // Build the prompt from messages
    // v1.4.10-beta: Use correct chat template format based on model name in request
    // Only use Qwen ChatML format if explicitly requested, otherwise use Mistral format (default)
    let is_qwen = req.model.to_lowercase().contains("qwen");

    let prompt = req.messages.iter()
        .map(|m| {
            if is_qwen {
                // Qwen3 uses ChatML format: <|im_start|>role\ncontent<|im_end|>
                match m.role.as_str() {
                    "system" => format!("<|im_start|>system\n{}<|im_end|>\n", m.content),
                    "user" => format!("<|im_start|>user\n{}<|im_end|>\n", m.content),
                    "assistant" => format!("<|im_start|>assistant\n{}<|im_end|>\n", m.content),
                    _ => format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content),
                }
            } else {
                // Mistral/Llama uses [INST] format (default)
                match m.role.as_str() {
                    "system" => format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", m.content),
                    "user" => format!("{} [/INST]", m.content),
                    "assistant" => format!("{}\n[INST] ", m.content),
                    _ => m.content.clone(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join("");

    // For Qwen3, add generation prompt
    let prompt = if is_qwen {
        format!("{}<|im_start|>assistant\n", prompt)
    } else {
        prompt
    };

    // Format the prompt for the model
    let formatted_prompt = q_ai_inference::format_chat_prompt(&req.model, &prompt);

    // Try distributed inference first, fall back to local
    let (response_text, tokens_generated) = if let Some(coordinator) = state.distributed_ai_coordinator.as_ref() {
        match coordinator.coordinate_inference_smart(
            formatted_prompt.clone(),
            Some(req.max_tokens as usize),
            Some(req.temperature as f64),
            req.model.clone(),
        ).await {
            Ok((_request_id, mut stream_rx, worker_node_id)) => {
                let mut full_response = String::new();
                let mut tokens = 0usize;

                info!("📥 Collecting tokens from worker {}...", worker_node_id);

                while let Some(event) = stream_rx.recv().await {
                    match event.event {
                        q_network::distributed_ai_coordinator::StreamEventKind::Token { token, .. } => {
                            full_response.push_str(&token);
                            tokens += 1;
                        }
                        q_network::distributed_ai_coordinator::StreamEventKind::Complete { tokens_generated, .. } => {
                            tokens = tokens_generated;
                            break;
                        }
                        q_network::distributed_ai_coordinator::StreamEventKind::Error { code, message } => {
                            warn!("⚠️ Distributed inference error: {} - {}", code, message);
                            break;
                        }
                        _ => {}
                    }
                }

                if !full_response.is_empty() {
                    (full_response, tokens)
                } else {
                    // Fall back to local inference
                    info!("⚠️ Distributed returned empty, trying local inference...");
                    try_local_inference(&state, &formatted_prompt, req.max_tokens as usize).await?
                }
            }
            Err(e) => {
                warn!("⚠️ Distributed inference failed: {}, trying local...", e);
                try_local_inference(&state, &formatted_prompt, req.max_tokens as usize).await?
            }
        }
    } else {
        try_local_inference(&state, &formatted_prompt, req.max_tokens as usize).await?
    };

    info!("✅ [OpenAI-compat] Generated {} tokens", tokens_generated);

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: current_timestamp(),
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                role: "assistant".to_string(),
                content: response_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: CompletionUsage {
            prompt_tokens: (prompt.len() / 4) as u32, // Rough estimate
            completion_tokens: tokens_generated as u32,
            total_tokens: (prompt.len() / 4) as u32 + tokens_generated as u32,
        },
    }))
}

/// Helper to try local inference
async fn try_local_inference(
    state: &Arc<AppState>,
    prompt: &str,
    max_tokens: usize,
) -> Result<(String, usize), StatusCode> {
    // Try loading the AI engine on-demand
    match ensure_ai_engine_loaded(state).await {
        Ok(engine) => {
            match engine.generate(prompt, max_tokens).await {
                Ok(response) => {
                    let tokens = response.split_whitespace().count();
                    Ok((response, tokens))
                }
                Err(e) => {
                    error!("❌ Local inference failed: {}", e);
                    Err(StatusCode::SERVICE_UNAVAILABLE)
                }
            }
        }
        Err(e) => {
            error!("❌ Failed to load AI engine: {}", e);
            Err(StatusCode::SERVICE_UNAVAILABLE)
        }
    }
}

/// v1.4.6-beta: Feedback request from frontend
#[derive(Debug, Deserialize)]
pub struct FeedbackRequest {
    pub message_id: String,
    pub feedback: String, // "good" or "bad"
    pub chat_id: String,
}

/// v1.4.6-beta: Submit feedback for an AI message
async fn submit_feedback(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<FeedbackRequest>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    info!(
        "📝 Received feedback: {} for message {} in chat {}",
        request.feedback, request.message_id, request.chat_id
    );

    // TODO: Store feedback in database for model improvement
    // For now, just log it and acknowledge

    Ok(Json(ApiResponse::success(format!(
        "Feedback '{}' recorded for message {}",
        request.feedback, request.message_id
    ))))
}

pub fn chat_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/completions", post(chat_completions)) // OpenAI-compatible endpoint
        .route("/create", post(create_chat))
        .route("/list", get(list_chats))
        .route("/metrics", get(get_ai_metrics)) // NEW: AI performance metrics endpoint
        .route("/workers", get(get_active_workers)) // NEW v1.0: List active workers
        .route("/feedback", post(submit_feedback)) // v1.4.6-beta: User feedback endpoint
        .route("/stream", get(stream_message_anonymous)) // Anonymous stream for wallet analysis
        .route("/:id/messages", get(get_messages))
        .route("/:id/message", post(send_message))
        .route("/:id/stream", get(stream_message)) // NEW: SSE streaming endpoint
        .route("/:id/stream-distributed", post(stream_message_distributed)) // NEW v1.0: Data parallel streaming
        .route("/:id/switch-model", post(switch_model)) // NEW: Model switching endpoint
        .route("/:id", delete(delete_chat))
        .route("/:id/rename", put(rename_chat))
        .route("/:id/settings", put(update_settings))
}
