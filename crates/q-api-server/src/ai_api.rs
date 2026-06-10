//! OpenAI-Compatible AI Inference API
//!
//! v9.6.0: Provides `/api/v1/ai/chat/completions` and related endpoints
//! compatible with the OpenAI API format, so existing tools (LangChain,
//! OpenAI SDK, etc.) can use QNK nodes as inference backends.
//!
//! Revenue flows back to node operators via the distributed fee system.

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        Json,
    },
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::AppState;
use crate::handlers::ApiResponse;

// ═══════════════════════════════════════════════════════════════════
// REQUEST/RESPONSE TYPES (OpenAI-compatible)
// ═══════════════════════════════════════════════════════════════════

/// OpenAI-compatible chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// POST /api/v1/ai/chat/completions request
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
    /// Wallet address for billing (optional, for pay-per-token)
    pub wallet: Option<String>,
}

fn default_max_tokens() -> usize { 512 }
fn default_temperature() -> f32 { 0.7 }

/// OpenAI-compatible chat completion response
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming chunk (OpenAI SSE format)
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: DeltaContent,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DeltaContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// POST /api/v1/ai/completions request (text completion, non-chat)
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    pub wallet: Option<String>,
}

/// Text completion response
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

/// GET /api/v1/ai/models response
#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub ready: bool,
}

/// GET /api/v1/ai/stats response
#[derive(Debug, Serialize)]
pub struct InferenceStatsResponse {
    pub engine: String,
    pub model: String,
    pub total_requests: u64,
    pub total_tokens: u64,
    pub avg_tokens_per_second: f32,
    pub revenue_micro_qug: u64,
    pub pool_active: bool,
    pub pool_accepting: bool,
    pub tasks_in_queue: u32,
    pub active_tasks: u32,
}

// ═══════════════════════════════════════════════════════════════════
// HANDLERS
// ═══════════════════════════════════════════════════════════════════

/// POST /api/v1/ai/chat/completions — OpenAI-compatible chat endpoint
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<ApiResponse<()>>)> {
    // Get engine from chat_api's static engine (shared singleton)
    let engine = match crate::chat_api::get_static_ai_engine() {
        Some(e) => e,
        None => {
            // Try to load engine on demand
            match crate::chat_api::ensure_ai_engine_loaded(&state).await {
                Ok(e) => e,
                Err(e) => {
                    return Err((
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(ApiResponse::error(format!("AI engine not available: {}", e))),
                    ));
                }
            }
        }
    };

    // Build prompt from messages
    let last_user_msg = req.messages.iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .unwrap_or("");

    let model_name = req.model.as_deref().unwrap_or("auto");
    let formatted_prompt = q_ai_inference::format_chat_prompt(model_name, last_user_msg);

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..24].to_string());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Generate (non-streaming for now)
    match engine.generate(&formatted_prompt, req.max_tokens).await {
        Ok(text) => {
            let stats = engine.get_stats().await;

            // Record to inference pool if available
            if let Some(ref orch) = state.compute_orchestrator {
                orch.record_task(
                    q_compute::ComputeLayer::AiInference,
                    stats.tokens_generated as u64 * q_compute::inference_pool::DEFAULT_PRICE_PER_TOKEN_MICRO_QUG,
                );
            }

            Ok(Json(ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".to_string(),
                created,
                model: engine.engine_name().to_string(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: text,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: UsageInfo {
                    prompt_tokens: stats.prompt_tokens,
                    completion_tokens: stats.tokens_generated,
                    total_tokens: stats.prompt_tokens + stats.tokens_generated,
                },
            }))
        }
        Err(e) => {
            error!("AI inference failed: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(format!("Inference failed: {}", e))),
            ))
        }
    }
}

/// POST /api/v1/ai/completions — Text completion endpoint
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, Json<ApiResponse<()>>)> {
    let engine = match crate::chat_api::get_static_ai_engine() {
        Some(e) => e,
        None => {
            match crate::chat_api::ensure_ai_engine_loaded(&state).await {
                Ok(e) => e,
                Err(e) => {
                    return Err((
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(ApiResponse::error(format!("AI engine not available: {}", e))),
                    ));
                }
            }
        }
    };

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..24].to_string());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    match engine.generate(&req.prompt, req.max_tokens).await {
        Ok(text) => {
            let stats = engine.get_stats().await;

            if let Some(ref orch) = state.compute_orchestrator {
                orch.record_task(
                    q_compute::ComputeLayer::AiInference,
                    stats.tokens_generated as u64 * q_compute::inference_pool::DEFAULT_PRICE_PER_TOKEN_MICRO_QUG,
                );
            }

            Ok(Json(CompletionResponse {
                id: request_id,
                object: "text_completion".to_string(),
                created,
                model: engine.engine_name().to_string(),
                choices: vec![CompletionChoice {
                    index: 0,
                    text,
                    finish_reason: "stop".to_string(),
                }],
                usage: UsageInfo {
                    prompt_tokens: stats.prompt_tokens,
                    completion_tokens: stats.tokens_generated,
                    total_tokens: stats.prompt_tokens + stats.tokens_generated,
                },
            }))
        }
        Err(e) => {
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(format!("Inference failed: {}", e))),
            ))
        }
    }
}

/// GET /api/v1/ai/models — List available models
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<ModelListResponse> {
    let catalog = q_ai_inference::model_catalog::ModelCatalog::new(q_ai_inference::model_catalog::ModelCatalog::default_dir());
    let models = catalog.list_models().await;

    let entries: Vec<ModelEntry> = models.iter().map(|m| ModelEntry {
        id: m.name.clone(),
        object: "model".to_string(),
        created: 1709251200, // Feb 2024
        owned_by: "qnk-network".to_string(),
        ready: m.downloaded,
    }).collect();

    Json(ModelListResponse {
        object: "list".to_string(),
        data: entries,
    })
}

/// GET /api/v1/ai/workers — List network inference workers
pub async fn list_workers(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<serde_json::Value>> {
    // Get compute peer info from orchestrator
    let peers = if let Some(ref orch) = state.compute_orchestrator {
        let status = orch.status();
        status.cluster_peers
    } else {
        vec![]
    };

    let ai_peers: Vec<_> = peers.iter()
        .filter(|p| p.active_layers.iter().any(|l| l.contains("AiInference") || l.contains("ai_inference")))
        .collect();

    Json(ApiResponse::success(serde_json::json!({
        "total_workers": ai_peers.len(),
        "workers": ai_peers,
    })))
}

/// GET /api/v1/ai/stats — Inference statistics
pub async fn inference_stats(
    State(state): State<Arc<AppState>>,
) -> Json<InferenceStatsResponse> {
    let (pool_stats, pool_active, pool_accepting) = if let Some(ref orch) = state.compute_orchestrator {
        let pool = orch.inference_pool();
        (pool.stats(), pool.has_engine(), pool.is_accepting())
    } else {
        (q_compute::inference_pool::AIInferenceStats::default(), false, false)
    };

    let engine_name = crate::chat_api::get_static_ai_engine()
        .map(|e| e.engine_name().to_string())
        .unwrap_or_else(|| "none".to_string());

    Json(InferenceStatsResponse {
        engine: engine_name,
        model: pool_stats.model_loaded.clone(),
        total_requests: pool_stats.total_requests_served,
        total_tokens: pool_stats.total_tokens_generated,
        avg_tokens_per_second: pool_stats.avg_tokens_per_second,
        revenue_micro_qug: pool_stats.revenue_earned_micro_qug,
        pool_active,
        pool_accepting,
        tasks_in_queue: pool_stats.tasks_in_queue,
        active_tasks: pool_stats.active_tasks,
    })
}
