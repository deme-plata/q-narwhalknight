/// Web Search API — GLM-4.7-Flash via local Ollama + DuckDuckGo
///
/// POST /api/v1/web-search
/// 1. Scrapes DuckDuckGo for search results
/// 2. Feeds results as context to GLM-4.7-Flash running on Ollama
/// 3. Streams back AI-summarized answers with source citations via SSE
///
/// Environment variables:
///   OLLAMA_URL  — Ollama base URL (default: http://localhost:11434)
///   OLLAMA_MODEL — Model name (default: glm-4.7-flash)

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

// Use 127.0.0.1 instead of localhost to avoid IPv6 ::1 resolution failures
const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MODEL: &str = "nemotron-cascade-2";

const SYSTEM_PROMPT: &str = "You are a helpful search assistant. The user asked a question and web search results are provided below as context. \
Answer the user's query using the search results. Always cite your sources by referencing [Source N] where N corresponds to the numbered search results. \
Be concise, informative, and accurate. If the search results don't contain relevant information, say so honestly.";

/// v1.0.5: Comprehensive system prompt for direct chat mode (no web search).
/// This is the knowledge base that makes Nemotron "know" Quillon Graph without training.
/// Keep under ~2500 tokens so query + response fit in 4K context window.
const DIRECT_CHAT_PROMPT: &str = "\
You are **Quillon Graph AI**, the built-in assistant for the Quillon blockchain. You know everything about this project.

## CRITICAL NAMING
- Blockchain: **Quillon** (also called Quillon Graph, Q-NarwhalKnight internally)
- Native coin: **QUG** (NOT QNG, NOT QNK). Ticker symbol on exchanges: QUG
- Wallet addresses start with `qnk` followed by 64 hex chars (67 total)
- Website: quillon.xyz | Download: quillon.xyz/downloads

## WHAT IS QUILLON?
Quillon is a **Layer-1 proof-of-work blockchain** with post-quantum cryptography and DAG-Knight consensus. \
It is NOT a token on another chain — it has its own native blockchain, miners, and peer-to-peer network. \
QUG is the native coin used for transactions, mining rewards, gas fees, and DEX trading.

## ECONOMICS
- Max supply: **21,000,000 QUG** (21 million, like Bitcoin)
- Decimals: **24** (extreme precision for micropayments and DeFi)
- Emission: ~2,625,000 QUG/year (Era 0), halving every 4 years
- Block time: ~1 second (DAG-Knight consensus)
- Current era: Era 0 (highest emission rate)
- Dev fee: 1% of mining rewards → founder wallet

## MINING
- Algorithm: **BLAKE3 VDF** (100 sequential BLAKE3 hash rounds per nonce)
- Anyone can mine with CPU or GPU (download q-miner from quillon.xyz/downloads)
- GPU mining uses OpenCL (works with AMD and NVIDIA GPUs)
- To mine: download binary, run `./q-miner --server https://quillon.xyz --wallet qnk<your-address>`
- Mining rewards appear in your wallet within seconds via SSE (real-time)
- Difficulty adjusts dynamically based on network hashrate
- Upcoming: Genus-2 Jacobian VDF upgrade (quantum-resistant sequential PoW)

## WALLET
- Create wallet: Click 'Create Wallet' in the app — generates Ed25519 keypair + mnemonic phrase
- **SAVE YOUR MNEMONIC** — it's the only way to recover your wallet
- Send QUG: Enter recipient qnk-address, amount, and click Send
- Receive: Share your qnk-address with the sender
- Transaction fees: ~0.00002 QUG per transfer (very cheap)

## DEX (Decentralized Exchange)
- Built-in AMM (Automated Market Maker) like Uniswap
- Trade QUG for tokens and vice versa
- Create liquidity pools for any token pair
- Deploy your own token via the Token Factory (ERC-20 style on Quillon)
- Token deployment costs a small QUG fee

## SMART CONTRACTS
- Quillon has a **WASM-based smart contract VM**
- Developers can deploy contracts written in Rust (compiled to WASM)
- Token standard: QRC-20 (similar to ERC-20)
- Crown & Ash: Built-in blockchain game using smart contracts

## NETWORK
- Consensus: **DAG-Knight** — a DAG-based BFT protocol (zero-message-complexity)
- P2P: libp2p with gossipsub + Kademlia DHT
- Bootstrap nodes: Epsilon (10Gbit), Beta, Delta, Gamma
- Tor support: Built-in Tor circuits for privacy (optional)
- Post-quantum crypto: Dilithium5 signatures, Kyber1024 key exchange (Phase 1)

## CROWN & ASH
Crown & Ash is a **blockchain strategy game** built on Quillon. Players compete for \
territory, form alliances, and battle for control. All game state is on-chain. \
Game narratives are AI-generated using the built-in AI inference engine.

## KEY LINKS
- Download node: quillon.xyz/downloads
- Block explorer: built into the wallet (Explorer tab)
- Mining: download q-miner or use the built-in wallet miner
- Source: code.quillon.xyz (self-hosted git)

## HOW TO ANSWER
- Be **concise** and **accurate** about Quillon-specific facts
- Use **QUG** for the coin (NEVER say QNG or QNK for the coin)
- When blockchain data is provided as context, use it for accurate answers
- Format with markdown. Use bullet points for lists.
- If you don't know something specific, say so honestly
- For technical questions about mining or running a node, provide step-by-step instructions";

#[derive(Deserialize)]
pub struct WebSearchRequest {
    pub query: String,
    /// Optional recency filter: "day", "week", "month", or "any" (default)
    pub recency: Option<String>,
    /// Whether to stream (default true)
    pub stream: Option<bool>,
    /// Optional pre-fetched context. When provided, skips DuckDuckGo search entirely
    /// and uses this context directly. Used for smart command routing (blockchain API data).
    pub context: Option<String>,
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

/// Ollama chat request
#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    think: bool,
}

#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

/// Scrape DuckDuckGo HTML lite for search results
async fn fetch_duckduckgo_results(query: &str, max_results: usize) -> Vec<SearchResultItem> {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0")
        .redirect(reqwest::redirect::Policy::limited(3))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            warn!("[WebSearch] Failed to create DDG client: {}", e);
            return Vec::new();
        }
    };

    let url = format!(
        "https://html.duckduckgo.com/html/?q={}",
        urlencoding::encode(query)
    );

    let response = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            warn!("[WebSearch] DDG request failed: {}", e);
            return Vec::new();
        }
    };

    let html = match response.text().await {
        Ok(t) => t,
        Err(e) => {
            warn!("[WebSearch] DDG body read failed: {}", e);
            return Vec::new();
        }
    };

    parse_ddg_html(&html, max_results)
}

/// Parse DuckDuckGo HTML lite results using string matching
fn parse_ddg_html(html: &str, max_results: usize) -> Vec<SearchResultItem> {
    let mut results = Vec::new();

    // DuckDuckGo HTML results are in <a class="result__a" href="...">title</a>
    // and <a class="result__snippet" ...>snippet</a>
    // We parse by looking for result__a and result__snippet class markers

    let mut search_pos = 0;
    while results.len() < max_results {
        // Find the next result link
        let link_marker = "class=\"result__a\"";
        let link_start = match html[search_pos..].find(link_marker) {
            Some(pos) => search_pos + pos,
            None => break,
        };

        // Extract href from the <a> tag
        let tag_start = html[..link_start].rfind('<').unwrap_or(link_start);
        let tag_content = &html[tag_start..];

        let href = extract_attr(tag_content, "href").unwrap_or_default();

        // Extract title (text between > and </a>)
        let title_start = match html[link_start..].find('>') {
            Some(pos) => link_start + pos + 1,
            None => {
                search_pos = link_start + link_marker.len();
                continue;
            }
        };
        let title_end = match html[title_start..].find("</a>") {
            Some(pos) => title_start + pos,
            None => {
                search_pos = title_start;
                continue;
            }
        };
        let title = strip_html_tags(&html[title_start..title_end]);

        // Find snippet after this result
        let snippet_marker = "class=\"result__snippet\"";
        let snippet = if let Some(snippet_pos) = html[title_end..].find(snippet_marker) {
            let snippet_abs = title_end + snippet_pos;
            let snippet_start = match html[snippet_abs..].find('>') {
                Some(pos) => snippet_abs + pos + 1,
                None => snippet_abs,
            };
            let snippet_end_markers = ["</a>", "</td>", "</div>"];
            let mut snippet_end = html.len();
            for marker in &snippet_end_markers {
                if let Some(pos) = html[snippet_start..].find(marker) {
                    let candidate = snippet_start + pos;
                    if candidate < snippet_end {
                        snippet_end = candidate;
                    }
                }
            }
            strip_html_tags(&html[snippet_start..snippet_end])
        } else {
            String::new()
        };

        // Resolve DDG redirect URL
        let clean_url = resolve_ddg_url(&href);

        if !clean_url.is_empty() && !title.is_empty() {
            results.push(SearchResultItem {
                title: html_decode(&title),
                url: clean_url,
                snippet: html_decode(&snippet),
            });
        }

        search_pos = title_end;
    }

    results
}

/// Extract an attribute value from an HTML tag
fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let pattern = format!("{}=\"", attr);
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

/// Strip HTML tags from text
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    result.trim().to_string()
}

/// Decode common HTML entities
fn html_decode(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ")
}

/// Resolve DuckDuckGo redirect URLs to actual URLs
fn resolve_ddg_url(href: &str) -> String {
    // DDG uses //duckduckgo.com/l/?uddg=ENCODED_URL&rut=...
    if href.contains("uddg=") {
        if let Some(start) = href.find("uddg=") {
            let encoded = &href[start + 5..];
            let end = encoded.find('&').unwrap_or(encoded.len());
            let encoded_url = &encoded[..end];
            return urlencoding::decode(encoded_url)
                .unwrap_or_else(|_| encoded_url.into())
                .to_string();
        }
    }
    // Direct URL
    if href.starts_with("http") {
        return href.to_string();
    }
    // Relative URL
    if href.starts_with("//") {
        return format!("https:{}", href);
    }
    href.to_string()
}

/// v1.0.5: RAG (Retrieval-Augmented Generation) for Quillon knowledge base.
/// Keyword-matches the user query against pre-written topic chunks and returns
/// the most relevant chunks (max 2) to inject into the prompt.
/// This gives Nemotron deep domain knowledge within its 4K context window.
fn retrieve_knowledge_chunks(query: &str) -> String {
    // Topic keywords → chunk file paths
    // Each chunk is ~300-500 tokens, so 2 chunks ≈ 600-1000 tokens
    let chunk_dir = std::path::Path::new("docs/nemotron-chunks");
    if !chunk_dir.exists() {
        return String::new();
    }

    let query_lower = query.to_lowercase();

    // Score each chunk by keyword matches
    let chunk_keywords: &[(&str, &[&str])] = &[
        ("mining.txt", &["mine", "mining", "miner", "hashrate", "hash rate", "gpu", "cpu", "nonce", "vdf", "blake3", "difficulty", "block reward", "q-miner"]),
        ("wallet.txt", &["wallet", "send", "receive", "mnemonic", "address", "qnk", "transfer", "balance", "create wallet", "restore", "private key", "backup"]),
        ("dex.txt", &["dex", "swap", "trade", "trading", "liquidity", "pool", "amm", "token", "exchange", "uniswap", "deploy token", "token factory", "lp"]),
        ("economics.txt", &["supply", "emission", "halving", "reward", "tokenomics", "21 million", "price", "value", "era", "inflation", "deflationary", "max supply", "fee", "cost"]),
        ("network.txt", &["node", "sync", "peer", "p2p", "bootstrap", "network", "consensus", "dag", "libp2p", "gossipsub", "tor", "quantum", "post-quantum", "dilithium", "kyber"]),
        ("crown_ash.txt", &["crown", "ash", "game", "strategy", "territory", "battle", "alliance", "narrative", "play"]),
        ("smart_contracts.txt", &["contract", "smart contract", "wasm", "qrc-20", "deploy", "developer", "gas", "abi", "rust"]),
        ("troubleshooting.txt", &["error", "problem", "issue", "fix", "stuck", "crash", "oom", "won't", "can't", "help", "not working", "failed"]),
    ];

    let mut scored: Vec<(&str, usize)> = chunk_keywords.iter().map(|(file, keywords)| {
        let score = keywords.iter().filter(|kw| query_lower.contains(*kw)).count();
        (*file, score)
    }).collect();

    // Sort by score descending, take top 2
    scored.sort_by(|a, b| b.1.cmp(&a.1));

    let mut context = String::new();
    let mut chunks_loaded = 0;
    for (file, score) in &scored {
        if *score == 0 || chunks_loaded >= 2 {
            break;
        }
        let path = chunk_dir.join(file);
        if let Ok(content) = std::fs::read_to_string(&path) {
            if !context.is_empty() {
                context.push_str("\n---\n");
            }
            context.push_str(&content);
            chunks_loaded += 1;
            tracing::debug!("[RAG] Loaded chunk {} (score: {})", file, score);
        }
    }

    if chunks_loaded > 0 {
        tracing::info!("[RAG] Injected {} knowledge chunk(s) for query: {}", chunks_loaded, &query[..query.len().min(60)]);
    }

    context
}

/// Build the user prompt with search context
fn build_context_prompt(query: &str, results: &[SearchResultItem]) -> String {
    if results.is_empty() {
        return format!(
            "The user asked: \"{}\"\n\nNo web search results were found. Please answer based on your knowledge and clearly state that no web sources were available.",
            query
        );
    }

    let mut prompt = format!("The user asked: \"{}\"\n\nHere are the web search results:\n\n", query);
    for (i, result) in results.iter().enumerate() {
        prompt.push_str(&format!(
            "[Source {}] {}\nURL: {}\n{}\n\n",
            i + 1,
            result.title,
            result.url,
            result.snippet
        ));
    }
    prompt.push_str("Please provide a comprehensive answer using the search results above. Cite sources using [Source N] notation.");
    prompt
}

pub async fn web_search_handler(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<WebSearchRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, (StatusCode, Json<ErrorResponse>)> {
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

    let ollama_url = std::env::var("OLLAMA_URL")
        .unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
    let model = std::env::var("OLLAMA_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    let has_context = req.context.is_some();
    let provided_context = req.context.unwrap_or_default();

    if has_context {
        info!("[WebSearch] Direct chat mode — query: {} (context: {} bytes, model: {})", query, provided_context.len(), model);
    } else {
        info!("[WebSearch] Web search mode — query: {} (model: {}, ollama: {})", query, model, ollama_url);
    }

    // v9.3.3: Return SSE stream IMMEDIATELY so tower TimeoutLayer doesn't kill us.
    // All slow work (DDG scraping, Ollama inference) happens inside the stream.
    // This means the HTTP 200 + SSE headers go back in <1ms, and the frontend
    // can show a loading state while we scrape + infer.
    let stream = async_stream::stream! {
        let (system_prompt, user_prompt) = if has_context {
            // Direct chat mode: skip DuckDuckGo, use provided context
            // v1.0.5: RAG — inject relevant knowledge chunks for deeper answers
            let rag_context = retrieve_knowledge_chunks(&query);

            let prompt = if !provided_context.is_empty() {
                // Smart command: blockchain data + RAG chunks + question
                if rag_context.is_empty() {
                    format!("{}\n\nUser question: {}", provided_context, query)
                } else {
                    format!("Blockchain data:\n{}\n\nReference knowledge:\n{}\n\nUser question: {}", provided_context, rag_context, query)
                }
            } else if !rag_context.is_empty() {
                // Pure chat with RAG context
                format!("Reference knowledge:\n{}\n\nUser question: {}", rag_context, query)
            } else {
                // Pure chat, no extra context
                query.clone()
            };
            (DIRECT_CHAT_PROMPT.to_string(), prompt)
        } else {
            // Web search mode: fetch from DuckDuckGo first
            let search_results = fetch_duckduckgo_results(&query, 8).await;
            info!("[WebSearch] Got {} DDG results", search_results.len());

            // Send search results immediately so frontend can display them
            if !search_results.is_empty() {
                let citations_event = SearchResultsEvent {
                    results: search_results.clone(),
                };
                if let Ok(json) = serde_json::to_string(&citations_event) {
                    yield Ok(Event::default().event("search_results").data(json));
                }
            }

            (SYSTEM_PROMPT.to_string(), build_context_prompt(&query, &search_results))
        };

        // Call Ollama streaming API
        let ollama_req = OllamaChatRequest {
            model,
            messages: vec![
                OllamaMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                OllamaMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            stream: true,
            think: false,
        };

        // Use connect_timeout (not global timeout) because the response is streamed.
        // A global timeout would kill the stream after N seconds even while tokens flow.
        let client = match reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(300))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                error!("[WebSearch] Failed to create HTTP client: {}", e);
                let err = ErrorEvent { message: "Internal error creating HTTP client".to_string() };
                if let Ok(json) = serde_json::to_string(&err) {
                    yield Ok(Event::default().event("error").data(json));
                }
                return;
            }
        };

        let chat_url = format!("{}/api/chat", ollama_url);
        let response = match client
            .post(&chat_url)
            .header("Content-Type", "application/json")
            .json(&ollama_req)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!("[WebSearch] Failed to call Ollama at {}: {:?}", chat_url, e);
                let err = ErrorEvent { message: format!("AI model unavailable: {}", e) };
                if let Ok(json) = serde_json::to_string(&err) {
                    yield Ok(Event::default().event("error").data(json));
                }
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            error!("[WebSearch] Ollama error {}: {}", status, body);
            let err = ErrorEvent { message: format!("AI model returned error: {}", status) };
            if let Ok(json) = serde_json::to_string(&err) {
                yield Ok(Event::default().event("error").data(json));
            }
            return;
        }

        // Stream the Ollama response as SSE tokens
        let byte_stream = response.bytes_stream();
        let mut pinned = std::pin::pin!(byte_stream);
        let mut buffer = String::new();
        let mut total_tokens: Option<u64> = None;

        while let Some(chunk_result) = pinned.next().await {
            match chunk_result {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));

                    // Ollama streams one JSON object per line
                    while let Some(line_end) = buffer.find('\n') {
                        let line = buffer[..line_end].trim().to_string();
                        buffer = buffer[line_end + 1..].to_string();

                        if line.is_empty() {
                            continue;
                        }

                        // Parse Ollama streaming JSON
                        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&line) {
                            // Extract content from message.content
                            if let Some(content) = chunk
                                .get("message")
                                .and_then(|m| m.get("content"))
                                .and_then(|c| c.as_str())
                            {
                                if !content.is_empty() {
                                    let token = TokenEvent { content: content.to_string() };
                                    if let Ok(json) = serde_json::to_string(&token) {
                                        yield Ok(Event::default().event("token").data(json));
                                    }
                                }
                            }

                            // Check if done
                            if chunk.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                                total_tokens = chunk
                                    .get("eval_count")
                                    .and_then(|c| c.as_u64())
                                    .map(|eval| {
                                        let prompt = chunk
                                            .get("prompt_eval_count")
                                            .and_then(|p| p.as_u64())
                                            .unwrap_or(0);
                                        prompt + eval
                                    });

                                let done = DoneEvent { total_tokens };
                                if let Ok(json) = serde_json::to_string(&done) {
                                    yield Ok(Event::default().event("done").data(json));
                                }
                                break;
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
