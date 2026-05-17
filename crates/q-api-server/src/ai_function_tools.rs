//! Native Function Calling for Ministral-3B
//!
//! Ministral-3B supports native function/tool calling which allows us to:
//! - Define structured tool schemas that map directly to UserIntent
//! - Guarantee valid JSON output (no parsing failures)
//! - Much smaller prompts (model understands tools natively)
//!
//! This replaces the 150-line system prompt with clean tool definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tool definition for Ministral-3B function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: ToolParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameters {
    #[serde(rename = "type")]
    pub param_type: String,
    pub properties: HashMap<String, ParameterProperty>,
    pub required: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterProperty {
    #[serde(rename = "type")]
    pub prop_type: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
}

/// Generate all available tools for the DEX
///
/// Ministral-3B will call these tools directly with structured parameters
/// No more parsing JSON from free-form text!
pub fn get_dex_tools(known_tokens: &[String]) -> Vec<Tool> {
    vec![
        // ===== TRANSACTION TOOLS (Require confirmation) =====
        Tool {
            name: "transfer".into(),
            description: "Send tokens to another wallet address".into(),
            parameters: ToolParameters {
                param_type: "object".into(),
                properties: [
                    ("to_address".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Recipient wallet address (qnk1...)".into(),
                        enum_values: None,
                        default: None,
                    }),
                    ("token".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Token symbol to send".into(),
                        enum_values: Some(known_tokens.to_vec()),
                        default: None,
                    }),
                    ("amount".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Amount to send (number or 'MAX')".into(),
                        enum_values: None,
                        default: None,
                    }),
                    ("memo".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Optional transaction memo".into(),
                        enum_values: None,
                        default: None,
                    }),
                ].into(),
                required: vec!["to_address".into(), "token".into(), "amount".into()],
            },
        },
        Tool {
            name: "swap".into(),
            description: "Swap one token for another on the DEX".into(),
            parameters: ToolParameters {
                param_type: "object".into(),
                properties: [
                    ("from_token".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Token to swap from".into(),
                        enum_values: Some(known_tokens.to_vec()),
                        default: None,
                    }),
                    ("to_token".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Token to receive".into(),
                        enum_values: Some(known_tokens.to_vec()),
                        default: None,
                    }),
                    ("amount".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Amount to swap (number or 'MAX')".into(),
                        enum_values: None,
                        default: None,
                    }),
                    ("max_slippage_percent".into(), ParameterProperty {
                        prop_type: "number".into(),
                        description: "Maximum slippage tolerance".into(),
                        enum_values: None,
                        default: Some(serde_json::json!(1.0)),
                    }),
                ].into(),
                required: vec!["from_token".into(), "to_token".into(), "amount".into()],
            },
        },

        // ===== QUERY TOOLS (Read-only) =====
        Tool {
            name: "check_balance".into(),
            description: "Check token balance(s) in wallet".into(),
            parameters: ToolParameters {
                param_type: "object".into(),
                properties: [
                    ("token".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Token to check, or omit for all balances".into(),
                        enum_values: Some(known_tokens.to_vec()),
                        default: None,
                    }),
                ].into(),
                required: vec![],
            },
        },
        Tool {
            name: "get_price".into(),
            description: "Get current price of a token".into(),
            parameters: ToolParameters {
                param_type: "object".into(),
                properties: [
                    ("token".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Token to get price for".into(),
                        enum_values: Some(known_tokens.to_vec()),
                        default: None,
                    }),
                    ("in_currency".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Currency to quote in".into(),
                        enum_values: Some(vec!["USD".into(), "BTC".into(), "ETH".into()]),
                        default: Some(serde_json::json!("USD")),
                    }),
                ].into(),
                required: vec!["token".into()],
            },
        },
        Tool {
            name: "get_pool_info".into(),
            description: "Get liquidity pool information".into(),
            parameters: ToolParameters {
                param_type: "object".into(),
                properties: [
                    ("pool_id".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Pool ID (e.g., QUG-USDT), or omit for all pools".into(),
                        enum_values: None,
                        default: None,
                    }),
                ].into(),
                required: vec![],
            },
        },

        // ===== NEW: MARKET ANALYSIS TOOLS =====
        Tool {
            name: "analyze_market".into(),
            description: "Analyze market conditions for a token or pool".into(),
            parameters: ToolParameters {
                param_type: "object".into(),
                properties: [
                    ("target".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Token symbol or pool ID to analyze".into(),
                        enum_values: None,
                        default: None,
                    }),
                    ("metrics".into(), ParameterProperty {
                        prop_type: "array".into(),
                        description: "Metrics to analyze".into(),
                        enum_values: Some(vec![
                            "volume_24h".into(),
                            "price_change_24h".into(),
                            "liquidity_depth".into(),
                            "volatility".into(),
                            "whale_activity".into(),
                        ]),
                        default: None,
                    }),
                    ("timeframe".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Analysis timeframe".into(),
                        enum_values: Some(vec!["1h".into(), "24h".into(), "7d".into(), "30d".into()]),
                        default: Some(serde_json::json!("24h")),
                    }),
                ].into(),
                required: vec!["target".into()],
            },
        },
        Tool {
            name: "find_opportunities".into(),
            description: "Find trading opportunities based on market conditions".into(),
            parameters: ToolParameters {
                param_type: "object".into(),
                properties: [
                    ("strategy".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Strategy type to look for".into(),
                        enum_values: Some(vec![
                            "arbitrage".into(),
                            "high_yield_pools".into(),
                            "undervalued_tokens".into(),
                            "trending".into(),
                        ]),
                        default: None,
                    }),
                    ("min_profit_percent".into(), ParameterProperty {
                        prop_type: "number".into(),
                        description: "Minimum profit threshold".into(),
                        enum_values: None,
                        default: Some(serde_json::json!(0.5)),
                    }),
                ].into(),
                required: vec!["strategy".into()],
            },
        },
        Tool {
            name: "predict_price_impact".into(),
            description: "Predict price impact of a large trade".into(),
            parameters: ToolParameters {
                param_type: "object".into(),
                properties: [
                    ("from_token".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Token to swap from".into(),
                        enum_values: Some(known_tokens.to_vec()),
                        default: None,
                    }),
                    ("to_token".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Token to swap to".into(),
                        enum_values: Some(known_tokens.to_vec()),
                        default: None,
                    }),
                    ("amount".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Trade amount".into(),
                        enum_values: None,
                        default: None,
                    }),
                ].into(),
                required: vec!["from_token".into(), "to_token".into(), "amount".into()],
            },
        },
        Tool {
            name: "set_price_alert".into(),
            description: "Set an alert for when a token reaches a target price".into(),
            parameters: ToolParameters {
                param_type: "object".into(),
                properties: [
                    ("token".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Token to monitor".into(),
                        enum_values: Some(known_tokens.to_vec()),
                        default: None,
                    }),
                    ("condition".into(), ParameterProperty {
                        prop_type: "string".into(),
                        description: "Alert condition".into(),
                        enum_values: Some(vec!["above".into(), "below".into()]),
                        default: None,
                    }),
                    ("target_price".into(), ParameterProperty {
                        prop_type: "number".into(),
                        description: "Target price in USD".into(),
                        enum_values: None,
                        default: None,
                    }),
                ].into(),
                required: vec!["token".into(), "condition".into(), "target_price".into()],
            },
        },
    ]
}

/// System prompt for Ministral-3B (MUCH shorter with native function calling!)
pub fn get_ministral_system_prompt() -> String {
    r#"You are a helpful DEX assistant for Q-NarwhalKnight blockchain.

Use the available tools to help users:
- Check balances and prices
- Swap tokens and make transfers
- Analyze market conditions
- Find trading opportunities
- Set price alerts

Always use tools for actions. Be concise and helpful.
If a user's request is unclear, ask for clarification.
For transactions, show the details before executing."#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_generation() {
        let tokens = vec!["QUG".into(), "USDT".into(), "QBTC".into()];
        let tools = get_dex_tools(&tokens);

        assert!(tools.len() > 5);
        assert!(tools.iter().any(|t| t.name == "transfer"));
        assert!(tools.iter().any(|t| t.name == "swap"));
        assert!(tools.iter().any(|t| t.name == "analyze_market"));
    }
}
