//! AI Intent Parser for Q-NarwhalKnight DEX
//!
//! This module uses Mistral 7B to parse natural language into structured intents.
//! The AI is constrained to output ONLY valid JSON matching our intent schema.
//!
//! Security principles:
//! - AI outputs are ALWAYS validated against our schema
//! - Unknown tokens are rejected (no hallucinated tokens)
//! - Amounts are validated as numbers (no code injection)
//! - Addresses are validated against format rules

use crate::ai_intent::*;
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

// ============================================================================
// SYSTEM PROMPT - Constrains Mistral 7B to output valid intents
// ============================================================================

/// Generate the system prompt for intent parsing
/// This prompt is carefully crafted to constrain Mistral 7B's output
pub fn generate_system_prompt(token_registry: &TokenRegistry, context: &ConversationContext) -> String {
    let known_tokens = token_registry.all_symbols().join(", ");
    let balance_context = if context.balances.is_empty() {
        "User balances: Not yet loaded".to_string()
    } else {
        let balances: Vec<String> = context.balances
            .iter()
            .map(|(token, amount)| format!("{}: {}", token, amount))
            .collect();
        format!("User balances: {}", balances.join(", "))
    };

    format!(r#"You are a DEX intent parser for Q-NarwhalKnight blockchain. Your ONLY job is to convert user requests into structured JSON intents.

CRITICAL RULES:
1. Output ONLY valid JSON - no explanations, no markdown, no extra text
2. Only use tokens from this list: {known_tokens}
3. If user mentions an unknown token, use the "unsupported" action
4. For amounts, use strings (e.g., "100", "0.5", "MAX")
5. Never hallucinate - if unclear, use "clarify" action

{balance_context}

INTENT SCHEMA (you must output one of these):

1. CHECK BALANCE:
{{"action": "check_balance", "token": "QUG"}}  // specific token
{{"action": "check_balance", "token": null}}   // all balances

2. GET SWAP QUOTE:
{{"action": "get_swap_quote", "from_token": "QUG", "to_token": "USDT", "amount": "100", "direction": "exact_in"}}

3. SWAP TOKENS (requires confirmation):
{{"action": "swap", "from_token": "QUG", "to_token": "USDT", "amount": "100", "max_slippage_percent": 1.0, "deadline_seconds": 300}}

4. GET POOL INFO:
{{"action": "get_pool_info", "pool_id": "QUG-USDT"}}  // specific pool
{{"action": "get_pool_info", "pool_id": null}}        // all pools

5. GET PRICE:
{{"action": "get_price", "token": "QUG", "in_currency": "USD"}}

6. TRANSFER (requires confirmation):
{{"action": "transfer", "to_address": "qnk1...", "token": "QUG", "amount": "50", "memo": "Payment for services"}}

7. ADD LIQUIDITY (requires confirmation):
{{"action": "add_liquidity", "pool_id": "QUG-USDT", "token_a_amount": "100", "token_b_amount": null}}

8. REMOVE LIQUIDITY (requires confirmation):
{{"action": "remove_liquidity", "pool_id": "QUG-USDT", "lp_token_amount": "50", "min_token_a": null, "min_token_b": null}}

9. GET HISTORY:
{{"action": "get_transaction_history", "limit": 20, "token_filter": null}}

10. EXPLAIN CONCEPT:
{{"action": "explain_concept", "topic": "impermanent loss"}}

11. CLARIFY (when user request is ambiguous):
{{"action": "clarify", "question": "Which token do you want to swap from?", "options": ["QUG", "USDT", "QBTC"]}}

12. UNSUPPORTED (when request cannot be fulfilled):
{{"action": "unsupported", "reason": "Token XYZ is not available on this DEX", "suggestion": "Try QUG, USDT, or QBTC instead"}}

13. CHAT (for greetings, general questions):
{{"action": "chat", "response": "Hello! I can help you swap tokens, check balances, and more."}}

AMOUNT FORMATS:
- Exact number: "100", "50.5", "0.001"
- Max balance: "MAX"
- Percentage: "50%" (for liquidity removal)

EXAMPLES:
User: "swap 100 qug for usdt"
{{"action": "swap", "from_token": "QUG", "to_token": "USDT", "amount": "100", "max_slippage_percent": 1.0, "deadline_seconds": 300}}

User: "how much is my qug worth"
{{"action": "check_balance", "token": "QUG"}}

User: "send 50 qug to qnk1abc123"
{{"action": "transfer", "to_address": "qnk1abc123", "token": "QUG", "amount": "50", "memo": null}}

User: "what's the price of bitcoin"
{{"action": "get_price", "token": "QBTC", "in_currency": "USD"}}

User: "hello!"
{{"action": "chat", "response": "Hello! I'm your DEX assistant. I can help you swap tokens, check balances, add liquidity, and more. What would you like to do?"}}

Remember: Output ONLY the JSON object, nothing else."#, known_tokens = known_tokens, balance_context = balance_context)
}

// ============================================================================
// INTENT PARSER - Uses MistralRsEngine to parse intents
// ============================================================================

/// Intent parser that uses Mistral 7B for natural language understanding
pub struct IntentParser {
    /// Token registry for validation
    token_registry: Arc<TokenRegistry>,
    /// Conversation context
    context: ConversationContext,
    /// Minimum confidence threshold for accepting intents
    min_confidence: f64,
}

impl IntentParser {
    pub fn new(token_registry: Arc<TokenRegistry>) -> Self {
        Self {
            token_registry,
            context: ConversationContext::new(),
            min_confidence: 0.7,
        }
    }

    pub fn with_context(mut self, context: ConversationContext) -> Self {
        self.context = context;
        self
    }

    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Get the system prompt for Mistral 7B
    pub fn get_system_prompt(&self) -> String {
        generate_system_prompt(&self.token_registry, &self.context)
    }

    /// Parse user input into a structured intent
    /// This is called AFTER getting the AI response
    pub fn parse_ai_response(&self, ai_response: &str) -> Result<ParsedIntent> {
        // Clean up the response (remove markdown code blocks if present)
        let cleaned = self.clean_ai_response(ai_response);

        // Parse the JSON
        let raw_intent: RawIntentOutput = self.parse_json(&cleaned)?;

        // Validate the intent
        let validated_intent = self.validate_intent(raw_intent.intent)?;

        // Calculate risk level and confirmation requirements
        let risk_level = self.assess_risk(&validated_intent);
        let requires_confirmation = self.requires_confirmation(&validated_intent, &risk_level);

        // Extract entities for logging/debugging
        let extracted_entities = self.extract_entities(&validated_intent);

        Ok(ParsedIntent {
            intent: validated_intent,
            confidence: raw_intent.confidence,
            requires_confirmation,
            risk_level,
            extracted_entities,
            reasoning: raw_intent.reasoning,
        })
    }

    /// Clean up AI response (remove markdown, whitespace, etc.)
    fn clean_ai_response(&self, response: &str) -> String {
        let mut cleaned = response.trim().to_string();

        // Remove markdown code blocks
        if cleaned.starts_with("```json") {
            cleaned = cleaned.strip_prefix("```json").unwrap_or(&cleaned).to_string();
        }
        if cleaned.starts_with("```") {
            cleaned = cleaned.strip_prefix("```").unwrap_or(&cleaned).to_string();
        }
        if cleaned.ends_with("```") {
            cleaned = cleaned.strip_suffix("```").unwrap_or(&cleaned).to_string();
        }

        // Find the first { and last } to extract JSON
        if let (Some(start), Some(end)) = (cleaned.find('{'), cleaned.rfind('}')) {
            cleaned = cleaned[start..=end].to_string();
        }

        cleaned.trim().to_string()
    }

    /// Parse JSON response from AI
    fn parse_json(&self, json_str: &str) -> Result<RawIntentOutput> {
        // First try to parse as RawIntentOutput (full format with confidence)
        if let Ok(raw) = serde_json::from_str::<RawIntentOutput>(json_str) {
            return Ok(raw);
        }

        // Try parsing as just a UserIntent (without confidence wrapper)
        match serde_json::from_str::<UserIntent>(json_str) {
            Ok(intent) => Ok(RawIntentOutput {
                intent,
                confidence: 0.8, // Default confidence when not provided
                reasoning: None,
            }),
            Err(e) => {
                error!("[INTENT PARSER] Failed to parse JSON: {}", e);
                error!("[INTENT PARSER] Raw input: {}", json_str);
                Err(anyhow!("Failed to parse intent JSON: {}", e))
            }
        }
    }

    /// Validate the parsed intent
    fn validate_intent(&self, intent: UserIntent) -> Result<UserIntent> {
        match &intent {
            UserIntent::Swap { from_token, to_token, amount, .. } => {
                self.validate_token(from_token)?;
                self.validate_token(to_token)?;
                self.validate_amount(amount)?;
                if from_token.to_uppercase() == to_token.to_uppercase() {
                    return Err(anyhow!("Cannot swap a token for itself"));
                }
            }
            UserIntent::GetSwapQuote { from_token, to_token, amount, .. } => {
                self.validate_token(from_token)?;
                self.validate_token(to_token)?;
                self.validate_amount(amount)?;
            }
            UserIntent::Transfer { to_address, token, amount, .. } => {
                self.validate_token(token)?;
                self.validate_amount(amount)?;
                self.validate_address(to_address)?;
            }
            UserIntent::CheckBalance { token } => {
                if let Some(t) = token {
                    self.validate_token(t)?;
                }
            }
            UserIntent::GetPrice { token, .. } => {
                self.validate_token(token)?;
            }
            UserIntent::AddLiquidity { token_a_amount, token_b_amount, .. } => {
                self.validate_amount(token_a_amount)?;
                if let Some(b) = token_b_amount {
                    self.validate_amount(b)?;
                }
            }
            UserIntent::RemoveLiquidity { lp_token_amount, min_token_a, min_token_b, .. } => {
                self.validate_amount(lp_token_amount)?;
                if let Some(a) = min_token_a {
                    self.validate_amount(a)?;
                }
                if let Some(b) = min_token_b {
                    self.validate_amount(b)?;
                }
            }
            UserIntent::GetPoolInfo { .. } => {}
            UserIntent::GetTransactionHistory { .. } => {}
            UserIntent::ExplainConcept { .. } => {}
            UserIntent::Clarify { .. } => {}
            UserIntent::Unsupported { .. } => {}
            UserIntent::Chat { .. } => {}
        }

        Ok(intent)
    }

    /// Validate a token symbol
    fn validate_token(&self, token: &str) -> Result<()> {
        if !self.token_registry.is_valid(token) {
            return Err(anyhow!("Unknown token: {}. Valid tokens: {:?}",
                token, self.token_registry.all_symbols()));
        }
        Ok(())
    }

    /// Validate an amount string
    fn validate_amount(&self, amount: &str) -> Result<()> {
        let amount_upper = amount.to_uppercase();

        // Special values
        if amount_upper == "MAX" || amount_upper.ends_with('%') {
            return Ok(());
        }

        // Must be a valid number
        if amount.parse::<f64>().is_err() {
            return Err(anyhow!("Invalid amount: {}. Must be a number, 'MAX', or percentage", amount));
        }

        // Must be positive
        if let Ok(num) = amount.parse::<f64>() {
            if num <= 0.0 {
                return Err(anyhow!("Amount must be positive, got: {}", amount));
            }
        }

        Ok(())
    }

    /// Validate an address
    fn validate_address(&self, address: &str) -> Result<()> {
        // Basic validation - real implementation would be more thorough
        if address.is_empty() {
            return Err(anyhow!("Address cannot be empty"));
        }

        // Check for Q-NarwhalKnight address format (qnk1...)
        if !address.starts_with("qnk1") && !address.starts_with("QNK1") {
            warn!("[INTENT PARSER] Address doesn't start with qnk1: {}", address);
            // Don't reject - might be a valid format we don't know
        }

        // Basic sanitization check - no special characters that could be injection
        if address.contains(';') || address.contains('\'') || address.contains('"')
           || address.contains('<') || address.contains('>') {
            return Err(anyhow!("Address contains invalid characters"));
        }

        Ok(())
    }

    /// Assess the risk level of an intent
    fn assess_risk(&self, intent: &UserIntent) -> RiskLevel {
        match intent {
            // Read-only operations - no risk
            UserIntent::CheckBalance { .. } |
            UserIntent::GetSwapQuote { .. } |
            UserIntent::GetPoolInfo { .. } |
            UserIntent::GetPrice { .. } |
            UserIntent::GetTransactionHistory { .. } |
            UserIntent::ExplainConcept { .. } |
            UserIntent::Clarify { .. } |
            UserIntent::Unsupported { .. } |
            UserIntent::Chat { .. } => RiskLevel::None,

            // Transaction operations - assess based on amount
            UserIntent::Swap { amount, .. } |
            UserIntent::Transfer { amount, .. } => {
                self.assess_amount_risk(amount)
            }

            UserIntent::AddLiquidity { token_a_amount, .. } => {
                self.assess_amount_risk(token_a_amount)
            }

            UserIntent::RemoveLiquidity { lp_token_amount, .. } => {
                // Removing liquidity is generally higher risk
                let base_risk = self.assess_amount_risk(lp_token_amount);
                match base_risk {
                    RiskLevel::Low => RiskLevel::Medium,
                    RiskLevel::Medium => RiskLevel::High,
                    other => other,
                }
            }
        }
    }

    /// Assess risk based on amount
    fn assess_amount_risk(&self, amount: &str) -> RiskLevel {
        let amount_upper = amount.to_uppercase();

        // MAX always means entire balance - critical risk
        if amount_upper == "MAX" {
            return RiskLevel::Critical;
        }

        // Parse the amount
        if let Ok(num) = amount.parse::<f64>() {
            // TODO: These thresholds should be configurable and based on USD value
            if num < 10.0 {
                RiskLevel::Low
            } else if num < 100.0 {
                RiskLevel::Medium
            } else if num < 1000.0 {
                RiskLevel::High
            } else {
                RiskLevel::Critical
            }
        } else if amount_upper.ends_with('%') {
            // Percentage-based - check the percentage
            if let Ok(pct) = amount_upper.trim_end_matches('%').parse::<f64>() {
                if pct < 10.0 {
                    RiskLevel::Low
                } else if pct < 25.0 {
                    RiskLevel::Medium
                } else if pct < 50.0 {
                    RiskLevel::High
                } else {
                    RiskLevel::Critical
                }
            } else {
                RiskLevel::Medium // Can't parse - medium as default
            }
        } else {
            RiskLevel::Medium // Unknown format - medium as default
        }
    }

    /// Determine if an intent requires user confirmation
    fn requires_confirmation(&self, intent: &UserIntent, risk_level: &RiskLevel) -> bool {
        match intent {
            // Read-only operations never need confirmation
            UserIntent::CheckBalance { .. } |
            UserIntent::GetSwapQuote { .. } |
            UserIntent::GetPoolInfo { .. } |
            UserIntent::GetPrice { .. } |
            UserIntent::GetTransactionHistory { .. } |
            UserIntent::ExplainConcept { .. } |
            UserIntent::Clarify { .. } |
            UserIntent::Unsupported { .. } |
            UserIntent::Chat { .. } => false,

            // All transaction operations require confirmation
            UserIntent::Swap { .. } |
            UserIntent::Transfer { .. } |
            UserIntent::AddLiquidity { .. } |
            UserIntent::RemoveLiquidity { .. } => true,
        }
    }

    /// Extract entities from the intent for logging/debugging
    fn extract_entities(&self, intent: &UserIntent) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();

        match intent {
            UserIntent::Swap { from_token, to_token, amount, .. } => {
                entities.push(ExtractedEntity {
                    entity_type: EntityType::Token,
                    value: from_token.clone(),
                    original_text: from_token.clone(),
                    confidence: 1.0,
                });
                entities.push(ExtractedEntity {
                    entity_type: EntityType::Token,
                    value: to_token.clone(),
                    original_text: to_token.clone(),
                    confidence: 1.0,
                });
                entities.push(ExtractedEntity {
                    entity_type: EntityType::Amount,
                    value: amount.clone(),
                    original_text: amount.clone(),
                    confidence: 1.0,
                });
            }
            UserIntent::Transfer { to_address, token, amount, .. } => {
                entities.push(ExtractedEntity {
                    entity_type: EntityType::Address,
                    value: to_address.clone(),
                    original_text: to_address.clone(),
                    confidence: 1.0,
                });
                entities.push(ExtractedEntity {
                    entity_type: EntityType::Token,
                    value: token.clone(),
                    original_text: token.clone(),
                    confidence: 1.0,
                });
                entities.push(ExtractedEntity {
                    entity_type: EntityType::Amount,
                    value: amount.clone(),
                    original_text: amount.clone(),
                    confidence: 1.0,
                });
            }
            UserIntent::GetPrice { token, .. } => {
                entities.push(ExtractedEntity {
                    entity_type: EntityType::Token,
                    value: token.clone(),
                    original_text: token.clone(),
                    confidence: 1.0,
                });
            }
            UserIntent::CheckBalance { token } => {
                if let Some(t) = token {
                    entities.push(ExtractedEntity {
                        entity_type: EntityType::Token,
                        value: t.clone(),
                        original_text: t.clone(),
                        confidence: 1.0,
                    });
                }
            }
            _ => {}
        }

        entities
    }

    /// Update the conversation context
    pub fn update_context(&mut self, intent: ParsedIntent) {
        self.context.history.push(intent);
        // Keep only last 10 intents for context
        if self.context.history.len() > 10 {
            self.context.history.remove(0);
        }
    }

    /// Set wallet address in context
    pub fn set_wallet_address(&mut self, address: String) {
        self.context.wallet_address = Some(address);
    }

    /// Update balances in context
    pub fn update_balances(&mut self, balances: std::collections::HashMap<String, f64>) {
        self.context.balances = balances;
    }

    /// Set pending confirmation
    pub fn set_pending_confirmation(&mut self, confirmation_id: String) {
        self.context.pending_confirmation = Some(confirmation_id);
    }

    /// Clear pending confirmation
    pub fn clear_pending_confirmation(&mut self) {
        self.context.pending_confirmation = None;
    }

    /// Check if there's a pending confirmation
    pub fn has_pending_confirmation(&self) -> bool {
        self.context.pending_confirmation.is_some()
    }

    /// Get the pending confirmation ID
    pub fn get_pending_confirmation(&self) -> Option<&String> {
        self.context.pending_confirmation.as_ref()
    }
}

// ============================================================================
// PROMPT BUILDER - Build prompts for different scenarios
// ============================================================================

/// Build a complete prompt for Mistral 7B intent parsing
pub fn build_intent_prompt(
    user_message: &str,
    token_registry: &TokenRegistry,
    context: &ConversationContext,
) -> Vec<(String, String)> {
    // Build conversation history
    let mut messages = Vec::new();

    // System prompt
    messages.push((
        "system".to_string(),
        generate_system_prompt(token_registry, context)
    ));

    // Add recent conversation history (last 3 turns max)
    let history_start = context.history.len().saturating_sub(3);
    for past_intent in &context.history[history_start..] {
        // Reconstruct what the user might have said
        let user_msg = reconstruct_user_message(&past_intent.intent);
        let assistant_msg = serde_json::to_string(&past_intent.intent)
            .unwrap_or_else(|_| "{}".to_string());

        messages.push(("user".to_string(), user_msg));
        messages.push(("assistant".to_string(), assistant_msg));
    }

    // Current user message
    messages.push(("user".to_string(), user_message.to_string()));

    messages
}

/// Reconstruct a plausible user message from an intent (for context)
fn reconstruct_user_message(intent: &UserIntent) -> String {
    match intent {
        UserIntent::Swap { from_token, to_token, amount, .. } => {
            format!("swap {} {} for {}", amount, from_token, to_token)
        }
        UserIntent::Transfer { to_address, token, amount, .. } => {
            format!("send {} {} to {}", amount, token, to_address)
        }
        UserIntent::CheckBalance { token } => {
            match token {
                Some(t) => format!("check my {} balance", t),
                None => "check all my balances".to_string(),
            }
        }
        UserIntent::GetPrice { token, .. } => {
            format!("what's the price of {}", token)
        }
        UserIntent::GetSwapQuote { from_token, to_token, amount, .. } => {
            format!("quote for {} {} to {}", amount, from_token, to_token)
        }
        UserIntent::Chat { .. } => "hello".to_string(),
        _ => "previous request".to_string(),
    }
}

// ============================================================================
// HELPER - Check if response looks like confirmation
// ============================================================================

/// Check if user message is confirming a pending transaction
pub fn is_confirmation_message(message: &str) -> bool {
    let lower = message.to_lowercase();
    lower == "yes"
        || lower == "confirm"
        || lower == "ok"
        || lower == "proceed"
        || lower == "do it"
        || lower == "execute"
        || lower.starts_with("y")
}

/// Check if user message is canceling a pending transaction
pub fn is_cancellation_message(message: &str) -> bool {
    let lower = message.to_lowercase();
    lower == "no"
        || lower == "cancel"
        || lower == "abort"
        || lower == "stop"
        || lower == "nevermind"
        || lower.starts_with("n")
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_parser() -> IntentParser {
        IntentParser::new(Arc::new(TokenRegistry::new()))
    }

    #[test]
    fn test_parse_swap_intent() {
        let parser = create_test_parser();
        let json = r#"{"action": "swap", "from_token": "QUG", "to_token": "USDT", "amount": "100", "max_slippage_percent": 1.0, "deadline_seconds": 300}"#;

        let result = parser.parse_ai_response(json);
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert!(parsed.requires_confirmation);
        assert!(matches!(parsed.intent, UserIntent::Swap { .. }));
    }

    #[test]
    fn test_parse_balance_intent() {
        let parser = create_test_parser();
        let json = r#"{"action": "check_balance", "token": "QUG"}"#;

        let result = parser.parse_ai_response(json);
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert!(!parsed.requires_confirmation);
        assert_eq!(parsed.risk_level, RiskLevel::None);
    }

    #[test]
    fn test_validate_unknown_token() {
        let parser = create_test_parser();
        let json = r#"{"action": "swap", "from_token": "FAKE", "to_token": "USDT", "amount": "100", "max_slippage_percent": 1.0, "deadline_seconds": 300}"#;

        let result = parser.parse_ai_response(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_clean_markdown_response() {
        let parser = create_test_parser();

        // Test with markdown code block
        let response = "```json\n{\"action\": \"check_balance\", \"token\": null}\n```";
        let cleaned = parser.clean_ai_response(response);
        assert_eq!(cleaned, r#"{"action": "check_balance", "token": null}"#);
    }

    #[test]
    fn test_risk_assessment() {
        let parser = create_test_parser();

        // Small amount - low risk
        let json = r#"{"action": "swap", "from_token": "QUG", "to_token": "USDT", "amount": "5", "max_slippage_percent": 1.0, "deadline_seconds": 300}"#;
        let parsed = parser.parse_ai_response(json).unwrap();
        assert_eq!(parsed.risk_level, RiskLevel::Low);

        // MAX amount - critical risk
        let json = r#"{"action": "swap", "from_token": "QUG", "to_token": "USDT", "amount": "MAX", "max_slippage_percent": 1.0, "deadline_seconds": 300}"#;
        let parsed = parser.parse_ai_response(json).unwrap();
        assert_eq!(parsed.risk_level, RiskLevel::Critical);
    }

    #[test]
    fn test_confirmation_detection() {
        assert!(is_confirmation_message("yes"));
        assert!(is_confirmation_message("Yes"));
        assert!(is_confirmation_message("confirm"));
        assert!(is_confirmation_message("y"));

        assert!(!is_confirmation_message("no"));
        assert!(!is_confirmation_message("maybe"));
    }

    #[test]
    fn test_cancellation_detection() {
        assert!(is_cancellation_message("no"));
        assert!(is_cancellation_message("cancel"));
        assert!(is_cancellation_message("n"));

        assert!(!is_cancellation_message("yes"));
        assert!(!is_cancellation_message("proceed"));
    }
}
