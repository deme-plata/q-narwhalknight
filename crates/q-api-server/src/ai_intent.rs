//! AI Intent Parsing System for Q-NarwhalKnight DEX
//!
//! This module implements a safe intent-parsing architecture where:
//! - AI (Mistral 7B) parses natural language into structured intents
//! - Deterministic Rust code executes the intents
//! - AI NEVER touches private keys or constructs transactions directly
//!
//! Security principle: "The AI is a TRANSLATOR, not an EXECUTOR"

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// INTENT SCHEMA - All possible user intents
// ============================================================================

/// All possible user intents - AI can ONLY output these
/// Anything outside this schema is rejected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum UserIntent {
    // ===== QUERY INTENTS (Read-only, always safe) =====

    /// Check token balance(s)
    CheckBalance {
        /// Specific token to check, or None for all tokens
        token: Option<String>,
    },

    /// Get a swap quote without executing
    GetSwapQuote {
        from_token: String,
        to_token: String,
        /// Amount as string to avoid float precision issues
        amount: String,
        #[serde(default)]
        direction: SwapDirection,
    },

    /// Get information about liquidity pool(s)
    GetPoolInfo {
        /// Specific pool ID, or None for all pools
        pool_id: Option<String>,
    },

    /// Get transaction history
    GetTransactionHistory {
        #[serde(default = "default_history_limit")]
        limit: u32,
        token_filter: Option<String>,
    },

    /// Get current price of a token
    GetPrice {
        token: String,
        /// Currency to quote in (default: USD)
        #[serde(default = "default_currency")]
        in_currency: String,
    },

    /// Explain a DeFi concept
    ExplainConcept {
        topic: String,
    },

    // ===== TRANSACTION INTENTS (Require confirmation) =====

    /// Swap tokens
    Swap {
        from_token: String,
        to_token: String,
        /// Amount as string - can be number or "MAX" for entire balance
        amount: String,
        /// Maximum slippage tolerance (default: 1.0%)
        #[serde(default = "default_slippage")]
        max_slippage_percent: f64,
        /// Transaction deadline in seconds (default: 300)
        #[serde(default = "default_deadline")]
        deadline_seconds: u64,
    },

    /// Add liquidity to a pool
    AddLiquidity {
        pool_id: String,
        token_a_amount: String,
        /// If None, auto-calculate to maintain pool ratio
        token_b_amount: Option<String>,
    },

    /// Remove liquidity from a pool
    RemoveLiquidity {
        pool_id: String,
        /// Amount of LP tokens to burn
        lp_token_amount: String,
        /// Minimum token A to receive (slippage protection)
        min_token_a: Option<String>,
        /// Minimum token B to receive (slippage protection)
        min_token_b: Option<String>,
    },

    /// Transfer tokens to another address
    Transfer {
        to_address: String,
        token: String,
        amount: String,
        memo: Option<String>,
    },

    // ===== META INTENTS =====

    /// Ask user for clarification
    Clarify {
        question: String,
        options: Vec<String>,
    },

    /// User requested something unsupported
    Unsupported {
        reason: String,
        suggestion: Option<String>,
    },

    /// General chat/greeting (not a DEX operation)
    Chat {
        response: String,
    },
}

// Default value functions
fn default_history_limit() -> u32 { 20 }
fn default_currency() -> String { "USD".to_string() }
fn default_slippage() -> f64 { 1.0 }
fn default_deadline() -> u64 { 300 }

/// Direction of swap amount specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SwapDirection {
    #[default]
    ExactIn,   // "I want to swap exactly 100 QUG"
    ExactOut,  // "I want to receive exactly 100 USDT"
}

// ============================================================================
// PARSED INTENT - Result of AI parsing
// ============================================================================

/// Result of parsing user input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedIntent {
    /// The structured intent
    pub intent: UserIntent,
    /// AI's confidence in the parsing (0.0 - 1.0)
    pub confidence: f64,
    /// Whether this intent requires user confirmation
    pub requires_confirmation: bool,
    /// Risk level assessment
    pub risk_level: RiskLevel,
    /// Entities extracted from the input
    pub extracted_entities: Vec<ExtractedEntity>,
    /// AI's reasoning (for debugging/logging)
    pub reasoning: Option<String>,
}

/// Risk level of an operation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    /// Read-only queries - no risk
    None = 0,
    /// Small amounts, known tokens
    Low = 1,
    /// Larger amounts, standard operations
    Medium = 2,
    /// Very large amounts, new tokens, complex operations
    High = 3,
    /// Anything involving > 10% of portfolio
    Critical = 4,
}

impl Default for RiskLevel {
    fn default() -> Self {
        RiskLevel::Medium
    }
}

/// Entity extracted from user input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub entity_type: EntityType,
    pub value: String,
    pub original_text: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Token,
    Amount,
    Address,
    Percentage,
    Duration,
    PoolId,
    Unknown,
}

// ============================================================================
// EXECUTION RESULTS
// ============================================================================

/// Result of intent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutionResult {
    /// Immediate result (for queries)
    Immediate {
        response: IntentResponse,
    },
    /// Requires user confirmation (for transactions)
    RequiresConfirmation {
        preview: TransactionPreview,
    },
    /// Error occurred
    Error {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
}

/// Response data for different intent types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "response_type", rename_all = "snake_case")]
pub enum IntentResponse {
    /// Balance check result
    Balances {
        balances: Vec<TokenBalance>,
        total_value_usd: f64,
    },
    /// Swap quote result
    SwapQuote {
        quote: SwapQuote,
    },
    /// Pool information
    PoolInfo {
        pools: Vec<PoolInfo>,
    },
    /// Price information
    Price {
        token: String,
        price: f64,
        currency: String,
        change_24h: Option<f64>,
    },
    /// Transaction history
    History {
        transactions: Vec<TransactionRecord>,
    },
    /// Explanation of a concept
    Explanation {
        topic: String,
        explanation: String,
    },
    /// General chat response
    ChatResponse {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBalance {
    pub token: String,
    pub balance: f64,
    pub value_usd: f64,
    pub price_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapQuote {
    pub input_amount: f64,
    pub input_token: String,
    pub output_amount: f64,
    pub output_token: String,
    pub effective_rate: f64,
    pub price_impact_percent: f64,
    pub fee: f64,
    pub fee_token: String,
    pub route: Vec<String>,
    pub expires_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolInfo {
    pub pool_id: String,
    pub token_a: String,
    pub token_b: String,
    pub reserve_a: f64,
    pub reserve_b: f64,
    pub total_liquidity_usd: f64,
    pub apy_24h: Option<f64>,
    pub volume_24h: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionRecord {
    pub tx_hash: String,
    pub tx_type: String,
    pub timestamp: i64,
    pub status: String,
    pub details: HashMap<String, String>,
}

// ============================================================================
// TRANSACTION PREVIEW (for confirmation)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionPreview {
    /// Unique confirmation ID
    pub confirmation_id: String,
    /// Original parsed intent
    pub intent: UserIntent,
    /// Human-readable summary
    pub summary: TransactionSummary,
    /// Warnings to show user
    pub warnings: Vec<Warning>,
    /// When this preview expires (Unix timestamp)
    pub expires_at: i64,
    /// Required confirmation code for critical transactions
    pub confirmation_code: Option<String>,
    /// Estimated gas/fee
    pub estimated_fee: f64,
    pub fee_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TransactionSummary {
    Swap {
        input: String,
        output: String,
        rate: f64,
        price_impact: f64,
        min_output: f64,
    },
    Transfer {
        amount: String,
        token: String,
        to: String,
        memo: Option<String>,
    },
    AddLiquidity {
        token_a: String,
        amount_a: f64,
        token_b: String,
        amount_b: f64,
        lp_tokens: f64,
        share_of_pool: f64,
    },
    RemoveLiquidity {
        lp_tokens: f64,
        token_a: String,
        amount_a: f64,
        token_b: String,
        amount_b: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Warning {
    pub level: WarningLevel,
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum WarningLevel {
    Info = 0,
    Caution = 1,
    Warning = 2,
    Danger = 3,
}

// ============================================================================
// RAW AI OUTPUT (for parsing)
// ============================================================================

/// Raw output structure from Mistral 7B
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawIntentOutput {
    pub intent: UserIntent,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    pub reasoning: Option<String>,
}

fn default_confidence() -> f64 { 0.8 }

// ============================================================================
// KNOWN TOKENS REGISTRY
// ============================================================================

/// Registry of known valid tokens
#[derive(Debug, Clone)]
pub struct TokenRegistry {
    tokens: HashMap<String, TokenInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub symbol: String,
    pub name: String,
    pub decimals: u8,
    pub address: String,
    pub is_native: bool,
}

impl TokenRegistry {
    pub fn new() -> Self {
        let mut tokens = HashMap::new();

        // Add known tokens
        tokens.insert("QUG".to_string(), TokenInfo {
            symbol: "QUG".to_string(),
            name: "Quillon".to_string(),
            decimals: 8,
            address: "native".to_string(),
            is_native: true,
        });

        tokens.insert("USDT".to_string(), TokenInfo {
            symbol: "USDT".to_string(),
            name: "Tether USD".to_string(),
            decimals: 6,
            address: "qnk1usdt...".to_string(),
            is_native: false,
        });

        tokens.insert("QBTC".to_string(), TokenInfo {
            symbol: "QBTC".to_string(),
            name: "Quantum Bitcoin".to_string(),
            decimals: 8,
            address: "qnk1qbtc...".to_string(),
            is_native: false,
        });

        tokens.insert("QETH".to_string(), TokenInfo {
            symbol: "QETH".to_string(),
            name: "Quantum Ethereum".to_string(),
            decimals: 18,
            address: "qnk1qeth...".to_string(),
            is_native: false,
        });

        Self { tokens }
    }

    /// Check if a token symbol is valid
    pub fn is_valid(&self, symbol: &str) -> bool {
        self.tokens.contains_key(&symbol.to_uppercase())
    }

    /// Get token info
    pub fn get(&self, symbol: &str) -> Option<&TokenInfo> {
        self.tokens.get(&symbol.to_uppercase())
    }

    /// Get all token symbols (for AI context)
    pub fn all_symbols(&self) -> Vec<String> {
        self.tokens.keys().cloned().collect()
    }

    /// Add a new token (e.g., from on-chain discovery)
    pub fn add_token(&mut self, info: TokenInfo) {
        self.tokens.insert(info.symbol.to_uppercase(), info);
    }
}

impl Default for TokenRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CONVERSATION CONTEXT
// ============================================================================

/// Context for multi-turn conversations
#[derive(Debug, Clone, Default)]
pub struct ConversationContext {
    /// User's wallet address
    pub wallet_address: Option<String>,
    /// Current token balances (cached)
    pub balances: HashMap<String, f64>,
    /// Previous intents in this conversation
    pub history: Vec<ParsedIntent>,
    /// Pending confirmation if any
    pub pending_confirmation: Option<String>,
    /// User preferences
    pub preferences: UserPreferences,
}

#[derive(Debug, Clone, Default)]
pub struct UserPreferences {
    /// Default slippage tolerance
    pub default_slippage: f64,
    /// Preferred quote currency
    pub quote_currency: String,
    /// Whether to show detailed warnings
    pub verbose_warnings: bool,
}

impl ConversationContext {
    pub fn new() -> Self {
        Self {
            preferences: UserPreferences {
                default_slippage: 1.0,
                quote_currency: "USD".to_string(),
                verbose_warnings: true,
            },
            ..Default::default()
        }
    }

    /// Get total portfolio value in USD
    pub fn total_value_usd(&self) -> f64 {
        // This would be calculated from actual prices
        self.balances.values().sum()
    }

    /// Calculate what percentage of portfolio an amount represents
    pub fn portfolio_percentage(&self, token: &str, amount: f64) -> f64 {
        let total = self.total_value_usd();
        if total <= 0.0 { return 0.0; }

        // Get token value (simplified - would use real prices)
        let token_value = amount; // Placeholder
        (token_value / total) * 100.0
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_serialization() {
        let intent = UserIntent::Swap {
            from_token: "QUG".to_string(),
            to_token: "USDT".to_string(),
            amount: "100".to_string(),
            max_slippage_percent: 1.0,
            deadline_seconds: 300,
        };

        let json = serde_json::to_string(&intent).unwrap();
        println!("Serialized: {}", json);

        let parsed: UserIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(intent, parsed);
    }

    #[test]
    fn test_token_registry() {
        let registry = TokenRegistry::new();
        assert!(registry.is_valid("QUG"));
        assert!(registry.is_valid("qug")); // Case insensitive
        assert!(!registry.is_valid("INVALID"));
    }

    #[test]
    fn test_raw_intent_parsing() {
        let json = r#"{
            "intent": {
                "action": "swap",
                "from_token": "QUG",
                "to_token": "USDT",
                "amount": "50",
                "max_slippage_percent": 0.5,
                "deadline_seconds": 300
            },
            "confidence": 0.95,
            "reasoning": "User wants to swap QUG for USDT"
        }"#;

        let parsed: RawIntentOutput = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.confidence, 0.95);

        if let UserIntent::Swap { from_token, amount, .. } = parsed.intent {
            assert_eq!(from_token, "QUG");
            assert_eq!(amount, "50");
        } else {
            panic!("Expected Swap intent");
        }
    }
}
