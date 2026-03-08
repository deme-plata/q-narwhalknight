/// Web Search API — GLM-4-Flash with built-in web_search tool
///
/// POST /api/v1/web-search
/// Proxies to Zhipu AI's GLM-4-Flash model, which has a native `web_search` tool.
/// Streams back AI-summarized answers with source citations via SSE.

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        Json,
    },
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::StreamExt;
use tracing::{debug, error, info, warn};

use crate::AppState;

const ZHIPU_API_URL: &str = "https://open.bigmodel.cn/api/paas/v4/chat/completions";
const SYSTEM_PROMPT: &str = "You are a helpful search assistant. Answer the user's query using web search results. Always cite your sources with URLs when available. Be concise and informative.";

#[derive(Deserialize)]
pub struct WebSearchRequest {
    pub query: String,
    /// Optional recency filter: "day", "week", "month", or "any" (default)
    pub recency: Option<String>,
    /// Whether to stream (default true)
    pub stream: Option<bool>,
}

#[derive(Serialize)]
struct ZhipuRequest {
    model: String,
    messages: Vec<ZhipuMessage>,
    tools: Vec<ZhipuTool>,
    stream: bool,
}

#[derive(Serialize)]
struct ZhipuMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ZhipuTool {
    #[serde(rename = "type")]
    tool_type: String,
    web_search: ZhipuWebSearch,
}

#[derive(Serialize)]
struct ZhipuWebSearch {
    enable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_query: Option<String>,
}

/// SSE event data sent to the frontend
#[derive(Serialize)]
struct TokenEvent {
    content: String,
}

#[derive(Serialize, Clone)]
pub struct SearchResultItem {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Serialize)]
struct SearchResultsEvent {
    results: Vec<SearchResultItem>,
}

#[derive(Serialize)]
struct DoneEvent {
    total_tokens: Option<u64>,
}

#[derive(Serialize)]
struct ErrorEvent {
    message: String,
}

/// Non-streaming error response
#[derive(Serialize)]
pub struct ErrorResponse {
    pub success: bool,
    pub error: String,
}

pub async fn web_search_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WebSearchRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, (StatusCode, Json<ErrorResponse>)> {
    let api_key = std::env::var("ZHIPU_API_KEY").map_err(|_| {
        error!("ZHIPU_API_KEY not set");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                success: false,
                error: "Web search is not configured (missing API key)".to_string(),
            }),
        )
    })?;

    let query = req.query.trim().to_string();
    if query.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                success: false,
                error: "Query cannot be empty".to_string(),
            }),
        ));
    }

    info!("[WebSearch] Query: {}", query);

    let zhipu_req = ZhipuRequest {
        model: "glm-4-flash".to_string(),
        messages: vec![
            ZhipuMessage {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            ZhipuMessage {
                role: "user".to_string(),
                content: query.clone(),
            },
        ],
        tools: vec![ZhipuTool {
            tool_type: "web_search".to_string(),
            web_search: ZhipuWebSearch {
                enable: true,
                search_query: None, // Let the model decide
            },
        }],
        stream: true,
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| {
            error!("[WebSearch] Failed to create HTTP client: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    success: false,
                    error: "Internal error".to_string(),
                }),
            )
        })?;

    let response = client
        .post(ZHIPU_API_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&zhipu_req)
        .send()
        .await
        .map_err(|e| {
            error!("[WebSearch] Failed to call Zhipu API: {}", e);
            (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse {
                    success: false,
                    error: format!("Search API unavailable: {}", e),
                }),
            )
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        error!("[WebSearch] Zhipu API error {}: {}", status, body);
        return Err((
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                success: false,
                error: format!("Search API returned error: {}", status),
            }),
        ));
    }

    // Stream the response as SSE
    let byte_stream = response.bytes_stream();

    let stream = async_stream::stream! {
        let mut pinned = std::pin::pin!(byte_stream);
        let mut buffer = String::new();
        let mut collected_citations: Vec<SearchResultItem> = Vec::new();
        let mut total_tokens: Option<u64> = None;

        while let Some(chunk_result) = pinned.next().await {
            match chunk_result {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));

                    // Process complete SSE lines from buffer
                    while let Some(line_end) = buffer.find('\n') {
                        let line = buffer[..line_end].trim().to_string();
                        buffer = buffer[line_end + 1..].to_string();

                        if line.is_empty() || line == ":" {
                            continue;
                        }

                        // Parse SSE data lines
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data.trim() == "[DONE]" {
                                // Send citations if we collected any
                                if !collected_citations.is_empty() {
                                    let citations_event = SearchResultsEvent {
                                        results: collected_citations.clone(),
                                    };
                                    if let Ok(json) = serde_json::to_string(&citations_event) {
                                        yield Ok(Event::default().event("search_results").data(json));
                                    }
                                }

                                // Send done event
                                let done = DoneEvent { total_tokens };
                                if let Ok(json) = serde_json::to_string(&done) {
                                    yield Ok(Event::default().event("done").data(json));
                                }
                                break;
                            }

                            // Parse Zhipu SSE chunk
                            if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
                                // Extract token content from choices[0].delta.content
                                if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                                    for choice in choices {
                                        if let Some(delta) = choice.get("delta") {
                                            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                                if !content.is_empty() {
                                                    let token = TokenEvent { content: content.to_string() };
                                                    if let Ok(json) = serde_json::to_string(&token) {
                                                        yield Ok(Event::default().event("token").data(json));
                                                    }
                                                }
                                            }
                                        }

                                        // Extract web_search results from tool calls or annotations
                                        // GLM-4-Flash returns web_search results in the choice metadata
                                        if let Some(tool_calls) = choice.get("delta")
                                            .and_then(|d| d.get("tool_calls"))
                                            .and_then(|t| t.as_array())
                                        {
                                            for tc in tool_calls {
                                                if let Some(search_results) = tc.get("web_search")
                                                    .and_then(|ws| ws.get("search_results"))
                                                    .and_then(|sr| sr.as_array())
                                                {
                                                    for result in search_results {
                                                        let item = SearchResultItem {
                                                            title: result.get("title")
                                                                .and_then(|t| t.as_str())
                                                                .unwrap_or("")
                                                                .to_string(),
                                                            url: result.get("link")
                                                                .and_then(|l| l.as_str())
                                                                .unwrap_or("")
                                                                .to_string(),
                                                            snippet: result.get("content")
                                                                .and_then(|c| c.as_str())
                                                                .unwrap_or("")
                                                                .to_string(),
                                                        };
                                                        if !item.url.is_empty() {
                                                            collected_citations.push(item);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // Also check top-level web_search field (some API versions)
                                if let Some(web_search) = chunk.get("web_search").and_then(|ws| ws.as_array()) {
                                    for result in web_search {
                                        let item = SearchResultItem {
                                            title: result.get("title")
                                                .and_then(|t| t.as_str())
                                                .unwrap_or("")
                                                .to_string(),
                                            url: result.get("link")
                                                .and_then(|l| l.as_str())
                                                .unwrap_or("")
                                                .to_string(),
                                            snippet: result.get("content")
                                                .and_then(|c| c.as_str())
                                                .unwrap_or("")
                                                .to_string(),
                                        };
                                        if !item.url.is_empty() && !collected_citations.iter().any(|c| c.url == item.url) {
                                            collected_citations.push(item);
                                        }
                                    }
                                }

                                // Extract usage stats
                                if let Some(usage) = chunk.get("usage") {
                                    total_tokens = usage.get("total_tokens")
                                        .and_then(|t| t.as_u64());
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("[WebSearch] Stream error: {}", e);
                    let err = ErrorEvent { message: format!("Stream error: {}", e) };
                    if let Ok(json) = serde_json::to_string(&err) {
                        yield Ok(Event::default().event("error").data(json));
                    }
                    break;
                }
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15))))
}
