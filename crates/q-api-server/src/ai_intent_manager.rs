//! AI Intent Manager - Unified Intent Processing for Q-NarwhalKnight DEX
//!
//! This module ties together:
//! - Intent parsing (AI → structured intent)
//! - Intent execution (intent → result/preview)
//! - Confirmation management (preview → confirmed execution)
//! - Response generation (result → user-friendly message)
//!
//! It provides a single entry point for the chat API to process user messages
//! through the safe intent architecture.

use crate::ai_intent::*;
use crate::ai_intent_parser::*;
use crate::ai_intent_executor::*;
use anyhow::{anyhow, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ============================================================================
// INTENT MANAGER - Main entry point
// ============================================================================

/// Unified intent manager that orchestrates the full pipeline
pub struct IntentManager {
    /// Intent parser (uses Mistral 7B)
    parser: Arc<RwLock<IntentParser>>,
    /// Token registry
    token_registry: Arc<TokenRegistry>,
    /// Executor context
    executor_context: Arc<ExecutorContext>,
    /// Session state (confirmation states, etc.)
    sessions: Arc<RwLock<HashMap<String, SessionState>>>,
}

/// Per-session state
#[derive(Debug, Clone)]
pub struct SessionState {
    /// Wallet address for this session
    pub wallet_address: Option<String>,
    /// Pending confirmation if any
    pub pending_confirmation: Option<String>,
    /// Conversation context
    pub context: ConversationContext,
    /// Last activity timestamp
    pub last_activity: i64,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            wallet_address: None,
            pending_confirmation: None,
            context: ConversationContext::new(),
            last_activity: Utc::now().timestamp(),
        }
    }
}

/// Response from the intent manager to the chat API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentManagerResponse {
    /// User-friendly message to display
    pub message: String,
    /// Whether this needs user confirmation
    pub awaiting_confirmation: bool,
    /// Confirmation ID if awaiting confirmation
    pub confirmation_id: Option<String>,
    /// Confirmation code required (for critical transactions)
    pub confirmation_code_required: Option<String>,
    /// Structured data (for UI rendering)
    pub data: Option<ResponseData>,
    /// Warnings to display
    pub warnings: Vec<String>,
    /// Original parsed intent (for debugging)
    pub debug_intent: Option<String>,
}

/// Structured response data for UI rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseData {
    Balances {
        balances: Vec<TokenBalanceDisplay>,
        total_value_usd: String,
    },
    SwapQuote {
        input: String,
        output: String,
        rate: String,
        price_impact: String,
        fee: String,
    },
    SwapPreview {
        input: String,
        output: String,
        min_output: String,
        rate: String,
        price_impact: String,
        fee: String,
    },
    TransferPreview {
        amount: String,
        token: String,
        to: String,
        fee: String,
    },
    LiquidityPreview {
        token_a: String,
        amount_a: String,
        token_b: String,
        amount_b: String,
        lp_tokens: String,
        share_of_pool: String,
    },
    PoolInfo {
        pools: Vec<PoolInfoDisplay>,
    },
    Price {
        token: String,
        price: String,
        change_24h: Option<String>,
    },
    History {
        transactions: Vec<TransactionDisplay>,
    },
    TransactionSuccess {
        confirmation_id: String,
        tx_hash: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBalanceDisplay {
    pub token: String,
    pub balance: String,
    pub value_usd: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolInfoDisplay {
    pub pool_id: String,
    pub pair: String,
    pub tvl: String,
    pub apy: String,
    pub volume_24h: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionDisplay {
    pub tx_hash: String,
    pub tx_type: String,
    pub time_ago: String,
    pub details: String,
}

impl IntentManager {
    /// Create a new intent manager
    pub fn new(executor_context: Arc<ExecutorContext>) -> Self {
        let token_registry = Arc::new(TokenRegistry::new());

        Self {
            parser: Arc::new(RwLock::new(IntentParser::new(token_registry.clone()))),
            token_registry,
            executor_context,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a token to the registry
    pub async fn add_token(&self, info: TokenInfo) {
        // Update the shared registry
        let mut parser = self.parser.write().await;
        // Note: Would need to update the registry through the parser
        // For now, tokens are hardcoded in TokenRegistry::new()
        info!("[INTENT MANAGER] Token added: {}", info.symbol);
    }

    /// Get or create a session
    async fn get_session(&self, session_id: &str) -> SessionState {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned().unwrap_or_default()
    }

    /// Update a session
    async fn update_session(&self, session_id: &str, session: SessionState) {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.to_string(), session);
    }

    /// Set wallet address for a session
    pub async fn set_wallet(&self, session_id: &str, wallet: String) {
        let mut session = self.get_session(session_id).await;
        session.wallet_address = Some(wallet.clone());
        session.context.wallet_address = Some(wallet);
        self.update_session(session_id, session).await;
    }

    /// Get the system prompt for Mistral 7B
    pub async fn get_system_prompt(&self, session_id: &str) -> String {
        let session = self.get_session(session_id).await;
        generate_system_prompt(&self.token_registry, &session.context)
    }

    /// Process a user message and return a response
    /// This is the main entry point for the chat API
    pub async fn process_message(
        &self,
        session_id: &str,
        user_message: &str,
        ai_response: &str,
    ) -> Result<IntentManagerResponse> {
        let mut session = self.get_session(session_id).await;
        session.last_activity = Utc::now().timestamp();

        // Check if user is responding to a pending confirmation
        if let Some(confirmation_id) = &session.pending_confirmation {
            let confirmation_id = confirmation_id.clone();

            if is_confirmation_message(user_message) {
                // User confirmed - execute the transaction
                return self.handle_confirmation(session_id, &confirmation_id, user_message).await;
            } else if is_cancellation_message(user_message) {
                // User cancelled
                return self.handle_cancellation(session_id, &confirmation_id).await;
            }
            // Otherwise, treat as a new request (cancels the pending confirmation)
            session.pending_confirmation = None;
        }

        // Parse the AI response into an intent
        let parser = self.parser.read().await;
        let parsed = parser.parse_ai_response(ai_response)?;

        debug!("[INTENT MANAGER] Parsed intent: {:?}", parsed);

        // Create executor with wallet context
        let executor = match &session.wallet_address {
            Some(wallet) => IntentExecutor::new(self.executor_context.clone())
                .with_wallet(wallet.clone()),
            None => IntentExecutor::new(self.executor_context.clone()),
        };

        // Execute the intent
        let result = executor.execute(&parsed).await?;

        // Generate response
        let response = self.generate_response(&parsed, result, &mut session).await?;

        // Update session
        self.update_session(session_id, session).await;

        Ok(response)
    }

    /// Process without AI - for testing or when AI is unavailable
    pub async fn process_fallback(
        &self,
        session_id: &str,
        user_message: &str,
    ) -> Result<IntentManagerResponse> {
        // Simple keyword-based fallback
        let lower = user_message.to_lowercase();

        let intent = if lower.contains("balance") {
            if let Some(token) = self.extract_token(&lower) {
                UserIntent::CheckBalance { token: Some(token) }
            } else {
                UserIntent::CheckBalance { token: None }
            }
        } else if lower.contains("price") {
            if let Some(token) = self.extract_token(&lower) {
                UserIntent::GetPrice {
                    token,
                    in_currency: "USD".to_string(),
                }
            } else {
                return Ok(IntentManagerResponse {
                    message: "Which token's price would you like to check?".to_string(),
                    awaiting_confirmation: false,
                    confirmation_id: None,
                    confirmation_code_required: None,
                    data: None,
                    warnings: vec![],
                    debug_intent: None,
                });
            }
        } else if lower.contains("hello") || lower.contains("hi") || lower.contains("hey") {
            UserIntent::Chat {
                response: "Hello! I'm your DEX assistant. I can help you swap tokens, check balances, and more. What would you like to do?".to_string(),
            }
        } else {
            UserIntent::Chat {
                response: "I'm sorry, I didn't understand that. Try asking to check your balance, get a price, or swap tokens.".to_string(),
            }
        };

        let parsed = ParsedIntent {
            intent,
            confidence: 0.5, // Lower confidence for fallback
            requires_confirmation: false,
            risk_level: RiskLevel::None,
            extracted_entities: vec![],
            reasoning: Some("Keyword-based fallback parsing".to_string()),
        };

        let mut session = self.get_session(session_id).await;
        let executor = match &session.wallet_address {
            Some(wallet) => IntentExecutor::new(self.executor_context.clone())
                .with_wallet(wallet.clone()),
            None => IntentExecutor::new(self.executor_context.clone()),
        };

        let result = executor.execute(&parsed).await?;
        self.generate_response(&parsed, result, &mut session).await
    }

    /// Extract a token symbol from text
    fn extract_token(&self, text: &str) -> Option<String> {
        for symbol in self.token_registry.all_symbols() {
            if text.to_uppercase().contains(&symbol) {
                return Some(symbol);
            }
        }
        None
    }

    /// Handle a confirmation response
    async fn handle_confirmation(
        &self,
        session_id: &str,
        confirmation_id: &str,
        user_message: &str,
    ) -> Result<IntentManagerResponse> {
        let mut session = self.get_session(session_id).await;

        // Check if there's a confirmation code in the message
        let confirmation_code = self.extract_confirmation_code(user_message);

        // Create executor
        let executor = match &session.wallet_address {
            Some(wallet) => IntentExecutor::new(self.executor_context.clone())
                .with_wallet(wallet.clone()),
            None => IntentExecutor::new(self.executor_context.clone()),
        };

        // Attempt to confirm
        match executor.confirm_transaction(confirmation_id, confirmation_code.as_deref()).await {
            Ok(result) => {
                session.pending_confirmation = None;
                self.update_session(session_id, session).await;

                Ok(IntentManagerResponse {
                    message: "Transaction confirmed and submitted successfully!".to_string(),
                    awaiting_confirmation: false,
                    confirmation_id: Some(confirmation_id.to_string()),
                    confirmation_code_required: None,
                    data: Some(ResponseData::TransactionSuccess {
                        confirmation_id: confirmation_id.to_string(),
                        tx_hash: None, // Would be filled in by actual execution
                    }),
                    warnings: vec![],
                    debug_intent: None,
                })
            }
            Err(e) => {
                // Check if it's asking for confirmation code
                let error_msg = e.to_string();
                if error_msg.contains("requires confirmation code") {
                    // Extract the required code from error message
                    let code = error_msg.split(": ").last().unwrap_or("").to_string();
                    return Ok(IntentManagerResponse {
                        message: format!("Please type the confirmation code to proceed: {}", code),
                        awaiting_confirmation: true,
                        confirmation_id: Some(confirmation_id.to_string()),
                        confirmation_code_required: Some(code),
                        data: None,
                        warnings: vec!["This is a high-value transaction requiring extra confirmation.".to_string()],
                        debug_intent: None,
                    });
                }

                Err(e)
            }
        }
    }

    /// Handle a cancellation response
    async fn handle_cancellation(
        &self,
        session_id: &str,
        confirmation_id: &str,
    ) -> Result<IntentManagerResponse> {
        let mut session = self.get_session(session_id).await;

        // Create executor
        let executor = IntentExecutor::new(self.executor_context.clone());
        let _ = executor.cancel_transaction(confirmation_id).await;

        session.pending_confirmation = None;
        self.update_session(session_id, session).await;

        Ok(IntentManagerResponse {
            message: "Transaction cancelled. Is there anything else I can help you with?".to_string(),
            awaiting_confirmation: false,
            confirmation_id: None,
            confirmation_code_required: None,
            data: None,
            warnings: vec![],
            debug_intent: None,
        })
    }

    /// Extract confirmation code from user message
    fn extract_confirmation_code(&self, message: &str) -> Option<String> {
        // Look for 6-character alphanumeric codes
        let words: Vec<&str> = message.split_whitespace().collect();
        for word in words {
            let cleaned = word.to_uppercase();
            if cleaned.len() == 6 && cleaned.chars().all(|c| c.is_alphanumeric()) {
                return Some(cleaned);
            }
        }
        None
    }

    /// Generate user-friendly response from execution result
    async fn generate_response(
        &self,
        parsed: &ParsedIntent,
        result: ExecutionResult,
        session: &mut SessionState,
    ) -> Result<IntentManagerResponse> {
        match result {
            ExecutionResult::Immediate { response } => {
                self.format_immediate_response(response).await
            }
            ExecutionResult::RequiresConfirmation { preview } => {
                self.format_confirmation_request(preview, session).await
            }
            ExecutionResult::Error { code, message, suggestion } => {
                Ok(IntentManagerResponse {
                    message: format!("{}{}", message,
                        suggestion.map(|s| format!("\n\n{}", s)).unwrap_or_default()),
                    awaiting_confirmation: false,
                    confirmation_id: None,
                    confirmation_code_required: None,
                    data: None,
                    warnings: vec![format!("Error: {}", code)],
                    debug_intent: Some(format!("{:?}", parsed.intent)),
                })
            }
        }
    }

    /// Format an immediate (non-confirmation) response
    async fn format_immediate_response(
        &self,
        response: IntentResponse,
    ) -> Result<IntentManagerResponse> {
        match response {
            IntentResponse::Balances { balances, total_value_usd } => {
                let mut lines = vec!["Your balances:".to_string()];
                let display_balances: Vec<TokenBalanceDisplay> = balances.iter()
                    .map(|b| {
                        lines.push(format!("  {} {}: ${:.2}", b.balance, b.token, b.value_usd));
                        TokenBalanceDisplay {
                            token: b.token.clone(),
                            balance: format!("{:.4}", b.balance),
                            value_usd: format!("${:.2}", b.value_usd),
                        }
                    })
                    .collect();

                lines.push(format!("\nTotal value: ${:.2}", total_value_usd));

                Ok(IntentManagerResponse {
                    message: lines.join("\n"),
                    awaiting_confirmation: false,
                    confirmation_id: None,
                    confirmation_code_required: None,
                    data: Some(ResponseData::Balances {
                        balances: display_balances,
                        total_value_usd: format!("${:.2}", total_value_usd),
                    }),
                    warnings: vec![],
                    debug_intent: None,
                })
            }

            IntentResponse::SwapQuote { quote } => {
                let message = format!(
                    "Swap Quote:\n\
                    Input: {} {}\n\
                    Output: {:.6} {}\n\
                    Rate: 1 {} = {:.6} {}\n\
                    Price Impact: {:.2}%\n\
                    Fee: {:.6} {}\n\n\
                    This quote expires in 60 seconds.",
                    quote.input_amount, quote.input_token,
                    quote.output_amount, quote.output_token,
                    quote.input_token, quote.effective_rate, quote.output_token,
                    quote.price_impact_percent,
                    quote.fee, quote.fee_token,
                );

                Ok(IntentManagerResponse {
                    message,
                    awaiting_confirmation: false,
                    confirmation_id: None,
                    confirmation_code_required: None,
                    data: Some(ResponseData::SwapQuote {
                        input: format!("{} {}", quote.input_amount, quote.input_token),
                        output: format!("{:.6} {}", quote.output_amount, quote.output_token),
                        rate: format!("1 {} = {:.6} {}", quote.input_token, quote.effective_rate, quote.output_token),
                        price_impact: format!("{:.2}%", quote.price_impact_percent),
                        fee: format!("{:.6} {}", quote.fee, quote.fee_token),
                    }),
                    warnings: vec![],
                    debug_intent: None,
                })
            }

            IntentResponse::PoolInfo { pools } => {
                let mut lines = vec!["Liquidity Pools:".to_string()];
                let display_pools: Vec<PoolInfoDisplay> = pools.iter()
                    .map(|p| {
                        lines.push(format!(
                            "\n{}: {}/{}\n  TVL: ${:.2}M\n  APY: {:.1}%\n  24h Volume: ${:.2}K",
                            p.pool_id, p.token_a, p.token_b,
                            p.total_liquidity_usd / 1_000_000.0,
                            p.apy_24h.unwrap_or(0.0),
                            p.volume_24h / 1_000.0,
                        ));
                        PoolInfoDisplay {
                            pool_id: p.pool_id.clone(),
                            pair: format!("{}/{}", p.token_a, p.token_b),
                            tvl: format!("${:.2}M", p.total_liquidity_usd / 1_000_000.0),
                            apy: format!("{:.1}%", p.apy_24h.unwrap_or(0.0)),
                            volume_24h: format!("${:.2}K", p.volume_24h / 1_000.0),
                        }
                    })
                    .collect();

                Ok(IntentManagerResponse {
                    message: lines.join(""),
                    awaiting_confirmation: false,
                    confirmation_id: None,
                    confirmation_code_required: None,
                    data: Some(ResponseData::PoolInfo { pools: display_pools }),
                    warnings: vec![],
                    debug_intent: None,
                })
            }

            IntentResponse::Price { token, price, currency, change_24h } => {
                let change_str = change_24h
                    .map(|c| format!(" ({:+.2}% 24h)", c))
                    .unwrap_or_default();

                Ok(IntentManagerResponse {
                    message: format!("{} price: ${:.2} {}{}", token, price, currency, change_str),
                    awaiting_confirmation: false,
                    confirmation_id: None,
                    confirmation_code_required: None,
                    data: Some(ResponseData::Price {
                        token,
                        price: format!("${:.2}", price),
                        change_24h: change_24h.map(|c| format!("{:+.2}%", c)),
                    }),
                    warnings: vec![],
                    debug_intent: None,
                })
            }

            IntentResponse::History { transactions } => {
                let mut lines = vec!["Recent transactions:".to_string()];
                let display_txs: Vec<TransactionDisplay> = transactions.iter()
                    .take(10)
                    .map(|tx| {
                        let time_ago = self.format_time_ago(tx.timestamp);
                        let details = tx.details.iter()
                            .map(|(k, v)| format!("{}: {}", k, v))
                            .collect::<Vec<_>>()
                            .join(", ");

                        lines.push(format!(
                            "\n{} - {} ({})\n  {}",
                            tx.tx_type, tx.status, time_ago, details
                        ));

                        TransactionDisplay {
                            tx_hash: tx.tx_hash.clone(),
                            tx_type: tx.tx_type.clone(),
                            time_ago,
                            details,
                        }
                    })
                    .collect();

                Ok(IntentManagerResponse {
                    message: lines.join(""),
                    awaiting_confirmation: false,
                    confirmation_id: None,
                    confirmation_code_required: None,
                    data: Some(ResponseData::History { transactions: display_txs }),
                    warnings: vec![],
                    debug_intent: None,
                })
            }

            IntentResponse::Explanation { topic, explanation } => {
                Ok(IntentManagerResponse {
                    message: format!("**{}**\n\n{}", topic, explanation),
                    awaiting_confirmation: false,
                    confirmation_id: None,
                    confirmation_code_required: None,
                    data: None,
                    warnings: vec![],
                    debug_intent: None,
                })
            }

            IntentResponse::ChatResponse { message } => {
                Ok(IntentManagerResponse {
                    message,
                    awaiting_confirmation: false,
                    confirmation_id: None,
                    confirmation_code_required: None,
                    data: None,
                    warnings: vec![],
                    debug_intent: None,
                })
            }
        }
    }

    /// Format a confirmation request
    async fn format_confirmation_request(
        &self,
        preview: TransactionPreview,
        session: &mut SessionState,
    ) -> Result<IntentManagerResponse> {
        let (message, data) = match &preview.summary {
            TransactionSummary::Swap { input, output, rate, price_impact, min_output } => {
                let msg = format!(
                    "Swap Preview:\n\
                    You send: {}\n\
                    You receive: {} (min: {:.6})\n\
                    Rate: 1 token = {:.6}\n\
                    Price Impact: {:.2}%\n\
                    Fee: {:.6} {}\n\n\
                    Reply 'yes' to confirm or 'no' to cancel.",
                    input, output, min_output, rate, price_impact,
                    preview.estimated_fee, preview.fee_token,
                );

                let d = ResponseData::SwapPreview {
                    input: input.clone(),
                    output: output.clone(),
                    min_output: format!("{:.6}", min_output),
                    rate: format!("{:.6}", rate),
                    price_impact: format!("{:.2}%", price_impact),
                    fee: format!("{:.6} {}", preview.estimated_fee, preview.fee_token),
                };

                (msg, d)
            }

            TransactionSummary::Transfer { amount, token, to, memo } => {
                let memo_str = memo.as_ref()
                    .map(|m| format!("\nMemo: {}", m))
                    .unwrap_or_default();

                let msg = format!(
                    "Transfer Preview:\n\
                    Amount: {} {}\n\
                    To: {}{}\n\
                    Fee: {:.6} {}\n\n\
                    Reply 'yes' to confirm or 'no' to cancel.",
                    amount, token, to, memo_str,
                    preview.estimated_fee, preview.fee_token,
                );

                let d = ResponseData::TransferPreview {
                    amount: amount.clone(),
                    token: token.clone(),
                    to: to.clone(),
                    fee: format!("{:.6} {}", preview.estimated_fee, preview.fee_token),
                };

                (msg, d)
            }

            TransactionSummary::AddLiquidity { token_a, amount_a, token_b, amount_b, lp_tokens, share_of_pool } => {
                let msg = format!(
                    "Add Liquidity Preview:\n\
                    Deposit: {:.4} {} + {:.4} {}\n\
                    Receive: {:.4} LP tokens\n\
                    Pool share: {:.2}%\n\
                    Fee: {:.6} {}\n\n\
                    Reply 'yes' to confirm or 'no' to cancel.",
                    amount_a, token_a, amount_b, token_b,
                    lp_tokens, share_of_pool,
                    preview.estimated_fee, preview.fee_token,
                );

                let d = ResponseData::LiquidityPreview {
                    token_a: token_a.clone(),
                    amount_a: format!("{:.4}", amount_a),
                    token_b: token_b.clone(),
                    amount_b: format!("{:.4}", amount_b),
                    lp_tokens: format!("{:.4}", lp_tokens),
                    share_of_pool: format!("{:.2}%", share_of_pool),
                };

                (msg, d)
            }

            TransactionSummary::RemoveLiquidity { lp_tokens, token_a, amount_a, token_b, amount_b } => {
                let msg = format!(
                    "Remove Liquidity Preview:\n\
                    Burn: {:.4} LP tokens\n\
                    Receive: {:.4} {} + {:.4} {}\n\
                    Fee: {:.6} {}\n\n\
                    Reply 'yes' to confirm or 'no' to cancel.",
                    lp_tokens, amount_a, token_a, amount_b, token_b,
                    preview.estimated_fee, preview.fee_token,
                );

                let d = ResponseData::LiquidityPreview {
                    token_a: token_a.clone(),
                    amount_a: format!("{:.4}", amount_a),
                    token_b: token_b.clone(),
                    amount_b: format!("{:.4}", amount_b),
                    lp_tokens: format!("{:.4}", lp_tokens),
                    share_of_pool: "N/A".to_string(),
                };

                (msg, d)
            }
        };

        // Collect warnings
        let warnings: Vec<String> = preview.warnings.iter()
            .map(|w| w.message.clone())
            .collect();

        // Set pending confirmation in session
        session.pending_confirmation = Some(preview.confirmation_id.clone());

        // Check if confirmation code is required
        let confirmation_code_required = if preview.confirmation_code.is_some() {
            Some(format!(
                "This is a high-value transaction. Type the code {} to confirm.",
                preview.confirmation_code.as_ref().unwrap()
            ))
        } else {
            None
        };

        Ok(IntentManagerResponse {
            message,
            awaiting_confirmation: true,
            confirmation_id: Some(preview.confirmation_id),
            confirmation_code_required,
            data: Some(data),
            warnings,
            debug_intent: None,
        })
    }

    /// Format a timestamp as "X ago"
    fn format_time_ago(&self, timestamp: i64) -> String {
        let now = Utc::now().timestamp();
        let diff = now - timestamp;

        if diff < 60 {
            "just now".to_string()
        } else if diff < 3600 {
            format!("{}m ago", diff / 60)
        } else if diff < 86400 {
            format!("{}h ago", diff / 3600)
        } else {
            format!("{}d ago", diff / 86400)
        }
    }

    /// Cleanup expired sessions
    pub async fn cleanup_expired_sessions(&self, max_age_seconds: i64) {
        let now = Utc::now().timestamp();
        let mut sessions = self.sessions.write().await;

        sessions.retain(|_, session| {
            now - session.last_activity < max_age_seconds
        });
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> Arc<ExecutorContext> {
        Arc::new(ExecutorContext {
            balance_provider: Arc::new(MockBalanceProvider),
            pool_provider: Arc::new(MockPoolProvider),
            price_provider: Arc::new(MockPriceProvider),
            history_provider: Arc::new(MockHistoryProvider),
            pending_previews: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    #[tokio::test]
    async fn test_process_balance_check() {
        let manager = IntentManager::new(create_test_context());
        manager.set_wallet("test-session", "qnk1test".to_string()).await;

        // Simulate AI response
        let ai_response = r#"{"action": "check_balance", "token": "QUG"}"#;

        let result = manager.process_message("test-session", "check my QUG balance", ai_response).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(!response.awaiting_confirmation);
        assert!(response.message.contains("QUG"));
    }

    #[tokio::test]
    async fn test_process_swap_requires_confirmation() {
        let manager = IntentManager::new(create_test_context());
        manager.set_wallet("test-session", "qnk1test".to_string()).await;

        let ai_response = r#"{"action": "swap", "from_token": "QUG", "to_token": "USDT", "amount": "100", "max_slippage_percent": 1.0, "deadline_seconds": 300}"#;

        let result = manager.process_message("test-session", "swap 100 QUG for USDT", ai_response).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.awaiting_confirmation);
        assert!(response.confirmation_id.is_some());
    }

    #[tokio::test]
    async fn test_fallback_processing() {
        let manager = IntentManager::new(create_test_context());
        manager.set_wallet("test-session", "qnk1test".to_string()).await;

        let result = manager.process_fallback("test-session", "check my balance").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_confirmation_flow() {
        let manager = IntentManager::new(create_test_context());
        manager.set_wallet("test-session", "qnk1test".to_string()).await;

        // First, create a swap request
        let ai_response = r#"{"action": "swap", "from_token": "QUG", "to_token": "USDT", "amount": "50", "max_slippage_percent": 1.0, "deadline_seconds": 300}"#;
        let response = manager.process_message("test-session", "swap 50 QUG for USDT", ai_response).await.unwrap();

        assert!(response.awaiting_confirmation);
        let confirmation_id = response.confirmation_id.unwrap();

        // Now confirm it
        let confirm_response = manager.handle_confirmation("test-session", &confirmation_id, "yes").await;
        assert!(confirm_response.is_ok());
    }
}
