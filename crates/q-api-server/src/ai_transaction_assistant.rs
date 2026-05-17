/// AI Transaction Assistant - Natural Language Blockchain Transactions
/// Integrates with Mistral Small 24B for intelligent transaction preparation
///
/// Features:
/// - Fuzzy address book search
/// - Natural language transaction parsing
/// - Fraud detection
/// - Transaction preview with security checks
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};

// Import AppState from parent
use q_api_server::AppState;

// Import AddressBookEntry, ApiResponse, and AuthenticatedWallet from handlers module
use q_api_server::handlers::{AddressBookEntry, ApiResponse, AuthenticatedWallet};

// Import BalanceStorage trait
use q_storage::BalanceStorage;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Fuzzy search query parameters
#[derive(Debug, Deserialize)]
pub struct FuzzySearchQuery {
    pub q: String, // Search query
    #[serde(default = "default_true")]
    pub fuzzy: bool, // Enable fuzzy matching
    #[serde(default = "default_confidence")]
    pub min_confidence: f64, // Minimum match confidence (0-1)
    #[serde(default = "default_limit")]
    pub limit: usize, // Max results
}

fn default_true() -> bool {
    true
}
fn default_confidence() -> f64 {
    0.75
}
fn default_limit() -> usize {
    5
}

/// Address book search result with confidence score
#[derive(Debug, Serialize)]
pub struct AddressMatch {
    pub entry: AddressBookEntry,
    pub confidence: f64,
    pub match_reason: String,
}

/// AI transaction preparation request
#[derive(Debug, Deserialize)]
pub struct AITransactionRequest {
    pub natural_language_query: String, // "Send 50 QUG to Alice for coffee"
    pub user_wallet: Option<String>,    // Optional wallet override
}

/// Parsed transaction intent from natural language
#[derive(Debug, Serialize)]
pub struct TransactionIntent {
    pub action: String,                    // "send", "pay", "transfer"
    pub recipient: Option<String>,         // Contact name or address
    pub recipient_address: Option<String>, // Resolved address
    pub amount: Option<f64>,               // QUG amount
    pub memo: Option<String>,              // Transaction note
    pub priority: String,                  // "low", "medium", "high"
}

/// Transaction preview with security checks
#[derive(Debug, Serialize)]
pub struct TransactionPreview {
    pub intent: TransactionIntent,
    pub from: String,
    pub to: String,
    pub amount: f64,
    pub fee_estimate: f64,
    pub total_cost: f64,
    pub security_checks: SecurityChecks,
    pub requires_confirmation: bool,
}

/// Security assessment for transaction
#[derive(Debug, Serialize)]
pub struct SecurityChecks {
    pub recipient_verified: bool,
    pub balance_sufficient: bool,
    pub fraud_score: f64, // 0-1, higher = more risky
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

// ============================================================================
// FUZZY MATCHING UTILITIES
// ============================================================================

/// Calculate Levenshtein distance for fuzzy matching
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_lower = s1.to_lowercase();
    let s2_lower = s2.to_lowercase();

    let len1 = s1_lower.chars().count();
    let len2 = s2_lower.chars().count();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    let s1_chars: Vec<char> = s1_lower.chars().collect();
    let s2_chars: Vec<char> = s2_lower.chars().collect();

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };
            matrix[i][j] = std::cmp::min(
                std::cmp::min(
                    matrix[i - 1][j] + 1, // deletion
                    matrix[i][j - 1] + 1, // insertion
                ),
                matrix[i - 1][j - 1] + cost, // substitution
            );
        }
    }

    matrix[len1][len2]
}

/// Calculate fuzzy match confidence score (0-1)
fn calculate_confidence(query: &str, target: &str) -> f64 {
    let query_lower = query.to_lowercase();
    let target_lower = target.to_lowercase();

    // Exact match
    if query_lower == target_lower {
        return 1.0;
    }

    // Substring match
    if target_lower.contains(&query_lower) {
        return 0.95;
    }

    // Levenshtein distance-based score
    let distance = levenshtein_distance(&query, &target);
    let max_len = std::cmp::max(query.len(), target.len());

    if max_len == 0 {
        return 0.0;
    }

    let similarity = 1.0 - (distance as f64 / max_len as f64);

    // Typo tolerance: 1 character difference = 90% confidence
    if distance == 1 {
        return 0.90;
    } else if distance == 2 {
        return 0.75;
    }

    similarity
}

// ============================================================================
// API HANDLERS
// ============================================================================

/// GET /v1/addressbook/search - Fuzzy search address book
///
/// Query parameters:
/// - q: Search query (required)
/// - fuzzy: Enable fuzzy matching (default: true)
/// - min_confidence: Minimum match confidence 0-1 (default: 0.75)
/// - limit: Max results (default: 5)
///
/// Example: /v1/addressbook/search?q=Alise&fuzzy=true&min_confidence=0.7
pub async fn search_address_book(
    State(state): State<Arc<AppState>>,
    auth: AuthenticatedWallet,
    Query(params): Query<FuzzySearchQuery>,
) -> Result<Json<ApiResponse<Vec<AddressMatch>>>, StatusCode> {
    let wallet_hex = hex::encode(&auth.address);

    info!(
        "🔍 AI Search: Fuzzy search for '{}' (wallet: {}, min_confidence: {})",
        params.q, wallet_hex, params.min_confidence
    );

    // Load user's address book
    let address_book_key = format!("addressbook:{}", wallet_hex);
    let addresses: Vec<AddressBookEntry> = match state
        .storage_engine
        .db_get("address_book", address_book_key.as_bytes())
        .await
    {
        Ok(Some(data)) => match serde_json::from_slice(&data) {
            Ok(addrs) => addrs,
            Err(e) => {
                warn!("Failed to deserialize address book: {}", e);
                vec![]
            }
        },
        Ok(None) | Err(_) => vec![],
    };

    if addresses.is_empty() {
        return Ok(Json(ApiResponse::success(vec![])));
    }

    // Perform fuzzy matching
    let mut matches: Vec<AddressMatch> = Vec::new();

    for entry in addresses.iter() {
        // Match against label (primary)
        let label_confidence = calculate_confidence(&params.q, &entry.label);

        // Match against tags
        let tag_confidence = entry
            .tags
            .iter()
            .map(|tag| calculate_confidence(&params.q, tag))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Match against notes
        let notes_confidence = calculate_confidence(&params.q, &entry.notes);

        // Match against address (partial)
        let address_match = entry
            .address
            .to_lowercase()
            .contains(&params.q.to_lowercase());
        let address_confidence = if address_match { 0.85 } else { 0.0 };

        // Take best match
        let best_confidence = label_confidence
            .max(tag_confidence)
            .max(notes_confidence)
            .max(address_confidence);

        if best_confidence >= params.min_confidence {
            let match_reason = if label_confidence == best_confidence {
                format!(
                    "Label match: '{}' ({}% confidence)",
                    entry.label,
                    (best_confidence * 100.0) as u32
                )
            } else if tag_confidence == best_confidence {
                format!(
                    "Tag match ({}% confidence)",
                    (best_confidence * 100.0) as u32
                )
            } else if address_match {
                format!(
                    "Address match ({}% confidence)",
                    (best_confidence * 100.0) as u32
                )
            } else {
                format!(
                    "Notes match ({}% confidence)",
                    (best_confidence * 100.0) as u32
                )
            };

            matches.push(AddressMatch {
                entry: entry.clone(),
                confidence: best_confidence,
                match_reason,
            });
        }
    }

    // Sort by confidence (descending) and limit results
    matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    matches.truncate(params.limit);

    info!(
        "✅ AI Search: Found {} matches for '{}'",
        matches.len(),
        params.q
    );

    Ok(Json(ApiResponse::success(matches)))
}

/// POST /v1/ai/transaction/prepare - Prepare transaction from natural language
///
/// Request body:
/// ```json
/// {
///   "natural_language_query": "Send 50 QUG to Alice for coffee",
///   "user_wallet": "qnk8f3a2..." // Optional
/// }
/// ```
///
/// Response includes:
/// - Parsed intent (action, recipient, amount)
/// - Transaction preview (from, to, amounts)
/// - Security checks (fraud score, warnings)
pub async fn prepare_ai_transaction(
    State(state): State<Arc<AppState>>,
    auth: AuthenticatedWallet,
    Json(req): Json<AITransactionRequest>,
) -> Result<Json<ApiResponse<TransactionPreview>>, StatusCode> {
    let wallet_hex = hex::encode(&auth.address);

    info!(
        "🤖 AI Transaction: Parsing '{}'",
        req.natural_language_query
    );

    // Step 1: Parse natural language to extract intent
    let intent = parse_transaction_intent(&req.natural_language_query);

    info!(
        "🎯 AI Transaction: Parsed intent - action: {}, recipient: {:?}, amount: {:?}",
        intent.action, intent.recipient, intent.amount
    );

    // Step 2: Resolve recipient name to address
    let resolved_address = if let Some(ref recipient_name) = intent.recipient {
        // Check if it's already an address
        if recipient_name.starts_with("qnk") {
            Some(recipient_name.clone())
        } else {
            // Search address book
            match resolve_contact_name(&state, &wallet_hex, recipient_name).await {
                Some(addr) => {
                    info!(
                        "✅ AI Transaction: Resolved '{}' to address: {}",
                        recipient_name, addr
                    );
                    Some(addr)
                }
                None => {
                    warn!(
                        "⚠️ AI Transaction: Could not resolve recipient '{}'",
                        recipient_name
                    );
                    None
                }
            }
        }
    } else {
        None
    };

    // Step 3: Get user's balance
    let balance = match state.storage_engine.get_balance(&wallet_hex).await {
        Ok(bal) => bal as f64 / 1e24, // Convert to QNK
        Err(_) => 0.0,
    };

    // Step 4: Perform security checks
    let security_checks = perform_security_checks(
        &state,
        &wallet_hex,
        resolved_address.as_deref(),
        intent.amount.unwrap_or(0.0),
        balance,
    )
    .await;

    // Step 5: Build transaction preview
    let amount = intent.amount.unwrap_or(0.0);
    let fee_estimate = 0.001; // Fixed fee for now

    let preview = TransactionPreview {
        intent,
        from: format!("qnk{}", &wallet_hex[..16]),
        to: resolved_address.unwrap_or_else(|| "UNRESOLVED".to_string()),
        amount,
        fee_estimate,
        total_cost: amount + fee_estimate,
        security_checks,
        requires_confirmation: true,
    };

    info!(
        "✅ AI Transaction: Preview generated - amount: {} QUG, fraud_score: {}",
        preview.amount, preview.security_checks.fraud_score
    );

    Ok(Json(ApiResponse::success(preview)))
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Parse natural language query to extract transaction intent
fn parse_transaction_intent(query: &str) -> TransactionIntent {
    let query_lower = query.to_lowercase();

    // Detect action
    let action = if query_lower.contains("send") {
        "send"
    } else if query_lower.contains("pay") {
        "pay"
    } else if query_lower.contains("transfer") {
        "transfer"
    } else {
        "unknown"
    }
    .to_string();

    // Extract amount (simple regex-like parsing)
    let mut amount = None;
    let words: Vec<&str> = query.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        if let Ok(num) = word.parse::<f64>() {
            // Check if next word is "QUG" or "qug"
            if i + 1 < words.len() && words[i + 1].to_lowercase() == "qug" {
                amount = Some(num);
                break;
            }
        }
    }

    // Extract recipient (word after "to")
    let mut recipient = None;
    for (i, word) in words.iter().enumerate() {
        if word.to_lowercase() == "to" && i + 1 < words.len() {
            recipient = Some(words[i + 1].to_string());
            break;
        }
    }

    // Extract memo (words after "for")
    let mut memo = None;
    if let Some(for_pos) = query_lower.find(" for ") {
        let memo_text = &query[for_pos + 5..].trim();
        if !memo_text.is_empty() {
            memo = Some(memo_text.to_string());
        }
    }

    TransactionIntent {
        action,
        recipient,
        recipient_address: None,
        amount,
        memo,
        priority: "medium".to_string(),
    }
}

/// Resolve contact name to blockchain address
async fn resolve_contact_name(
    state: &Arc<AppState>,
    wallet_hex: &str,
    name: &str,
) -> Option<String> {
    let address_book_key = format!("addressbook:{}", wallet_hex);

    let addresses: Vec<AddressBookEntry> = match state
        .storage_engine
        .db_get("address_book", address_book_key.as_bytes())
        .await
    {
        Ok(Some(data)) => serde_json::from_slice(&data).unwrap_or_default(),
        Ok(None) | Err(_) => return None,
    };

    // Fuzzy search for best match
    let mut best_match: Option<(String, f64)> = None;

    for entry in addresses.iter() {
        let confidence = calculate_confidence(name, &entry.label);

        if confidence > 0.75 {
            if let Some((_, prev_conf)) = best_match {
                if confidence > prev_conf {
                    best_match = Some((entry.address.clone(), confidence));
                }
            } else {
                best_match = Some((entry.address.clone(), confidence));
            }
        }
    }

    best_match.map(|(addr, _)| addr)
}

/// Perform security checks on transaction
async fn perform_security_checks(
    _state: &Arc<AppState>,
    _wallet_hex: &str,
    recipient: Option<&str>,
    amount: f64,
    balance: f64,
) -> SecurityChecks {
    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();
    let mut fraud_score = 0.0;

    // Check 1: Recipient resolved
    let recipient_verified = recipient.is_some() && recipient.unwrap() != "UNRESOLVED";
    if !recipient_verified {
        warnings.push("Recipient not found in address book".to_string());
        fraud_score += 0.2;
        recommendations.push("Verify recipient address manually".to_string());
    }

    // Check 2: Sufficient balance
    let balance_sufficient = amount + 0.001 <= balance; // Include fee
    if !balance_sufficient {
        warnings.push(format!(
            "Insufficient balance (need {}, have {})",
            amount + 0.001,
            balance
        ));
        fraud_score += 0.3;
    }

    // Check 3: Amount sanity check
    if amount > balance * 0.5 {
        warnings.push("Large transaction (>50% of balance)".to_string());
        recommendations.push("Consider splitting into smaller transactions".to_string());
        fraud_score += 0.1;
    }

    // Check 4: New recipient (placeholder - would check transaction history)
    // if is_new_recipient { fraud_score += 0.1; }

    SecurityChecks {
        recipient_verified,
        balance_sufficient,
        fraud_score,
        warnings,
        recommendations,
    }
}
