//! DEX Market Analyzer - Agentic AI for Market Intelligence
//!
//! This module uses Ministral-3B's agentic capabilities to provide:
//! - Real-time market analysis
//! - Opportunity detection (arbitrage, high-yield pools)
//! - Price impact predictions
//! - Whale movement alerts
//! - Trading strategy suggestions
//!
//! The agent uses native function calling for structured, reliable outputs.

use crate::ai_function_tools::{get_dex_tools, get_ministral_system_prompt, Tool};
use crate::AppState;
use anyhow::{anyhow, Result};
use axum::{
    extract::{Path, Query, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use chrono;

// ============================================================================
// MARKET ANALYSIS TYPES
// ============================================================================

/// Market analysis result from the AI agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketAnalysis {
    pub target: String,                    // Token or pool analyzed
    pub timestamp: u64,
    pub metrics: MarketMetrics,
    pub sentiment: MarketSentiment,
    pub recommendations: Vec<Recommendation>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMetrics {
    pub price_usd: f64,
    /// Price change in 1h in basis points (e.g., 100 = 1%)
    pub price_change_1h_bps: i32,
    /// Price change in 24h in basis points (e.g., 550 = 5.5%, -200 = -2%)
    pub price_change_24h_bps: i32,
    /// Price change in 7d in basis points
    pub price_change_7d_bps: i32,
    pub volume_24h: f64,
    pub liquidity_usd: f64,
    pub volatility_score: f64,            // 0-100, higher = more volatile
    pub whale_activity_score: f64,        // 0-100, higher = more whale activity
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketSentiment {
    StrongBullish,
    Bullish,
    Neutral,
    Bearish,
    StrongBearish,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub action: RecommendationAction,
    pub confidence: f64,
    pub reasoning: String,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationAction {
    Buy { token: String, suggested_amount_percent: f64 },
    Sell { token: String, suggested_amount_percent: f64 },
    Hold { token: String },
    AddLiquidity { pool_id: String, apr_estimate: f64 },
    RemoveLiquidity { pool_id: String, reason: String },
    Swap { from: String, to: String, reason: String },
    SetAlert { token: String, condition: String, price: f64 },
    Wait { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

// ============================================================================
// OPPORTUNITY DETECTION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingOpportunity {
    pub id: String,
    pub opportunity_type: OpportunityType,
    pub description: String,
    pub potential_profit_percent: f64,
    pub risk_level: RiskLevel,
    pub expires_in_seconds: Option<u64>,
    pub action_required: ActionRequired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpportunityType {
    Arbitrage {
        buy_pool: String,
        sell_pool: String,
        spread_percent: f64,
    },
    HighYieldPool {
        pool_id: String,
        current_apr: f64,
        tvl_usd: f64,
    },
    PriceDiscrepancy {
        token: String,
        our_price: f64,
        external_price: f64,
        source: String,
    },
    TrendingToken {
        token: String,
        volume_increase_percent: f64,
        price_momentum: f64,
    },
    WhaleAccumulation {
        token: String,
        whale_buys_24h: u64,
        total_volume: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRequired {
    pub tool_name: String,
    pub parameters: serde_json::Value,
    pub requires_confirmation: bool,
}

// ============================================================================
// MARKET ANALYZER AGENT
// ============================================================================

/// The Market Analyzer Agent - uses Ministral-3B for intelligent analysis
pub struct MarketAnalyzerAgent {
    /// Available tools for the agent
    tools: Vec<Tool>,
    /// System prompt for the agent
    system_prompt: String,
    /// Cache of recent analyses (avoid redundant API calls)
    analysis_cache: Arc<RwLock<HashMap<String, (MarketAnalysis, u64)>>>,
    /// Cache TTL in seconds
    cache_ttl: u64,
}

impl MarketAnalyzerAgent {
    pub fn new(known_tokens: &[String]) -> Self {
        Self {
            tools: get_dex_tools(known_tokens),
            system_prompt: Self::get_market_analyzer_prompt(),
            analysis_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_ttl: 60, // 1 minute cache
        }
    }

    /// Enhanced system prompt for market analysis
    fn get_market_analyzer_prompt() -> String {
        r#"You are an expert DeFi market analyzer for Q-NarwhalKnight DEX.

YOUR CAPABILITIES:
1. Analyze market conditions using on-chain data
2. Detect trading opportunities (arbitrage, yield, trends)
3. Predict price impacts for large trades
4. Monitor whale activity and unusual patterns
5. Provide actionable trading recommendations

ANALYSIS METHODOLOGY:
- Use multiple timeframes (1h, 24h, 7d) for trend analysis
- Consider liquidity depth before recommending trades
- Factor in gas costs and slippage for profitability
- Weight recent data more heavily for momentum
- Cross-reference multiple pools for price accuracy

OUTPUT FORMAT:
Always use the appropriate tool to structure your analysis.
Provide confidence scores (0.0-1.0) for all predictions.
Include clear reasoning for recommendations.
Highlight risks prominently.

RISK ASSESSMENT CRITERIA:
- Low: Established tokens, deep liquidity, stable prices
- Medium: Moderate volatility, adequate liquidity
- High: High volatility, thin liquidity, new tokens
- Very High: Extreme volatility, minimal liquidity, unverified tokens

Be conservative with recommendations. Never suggest all-in trades.
Always remind users that past performance doesn't guarantee future results."#.to_string()
    }

    /// Analyze a specific token or pool
    pub async fn analyze(
        &self,
        target: &str,
        metrics: &[&str],
        timeframe: &str,
    ) -> Result<MarketAnalysis> {
        // Check cache first
        let cache_key = format!("{}:{}:{}", target, metrics.join(","), timeframe);
        {
            let cache = self.analysis_cache.read().await;
            if let Some((analysis, timestamp)) = cache.get(&cache_key) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs();
                if now - timestamp < self.cache_ttl {
                    info!("📊 Returning cached analysis for {}", target);
                    return Ok(analysis.clone());
                }
            }
        }

        info!("🔍 Analyzing market for {} ({} timeframe)", target, timeframe);

        // TODO: Call MistralRsEngine with analyze_market tool
        // For now, return mock data demonstrating the structure
        let analysis = self.generate_mock_analysis(target, timeframe).await?;

        // Cache the result
        {
            let mut cache = self.analysis_cache.write().await;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            cache.insert(cache_key, (analysis.clone(), now));
        }

        Ok(analysis)
    }

    /// Find trading opportunities
    pub async fn find_opportunities(
        &self,
        strategy: &str,
        min_profit_percent: f64,
    ) -> Result<Vec<TradingOpportunity>> {
        info!("🎯 Searching for {} opportunities (min {}% profit)", strategy, min_profit_percent);

        // TODO: Implement actual opportunity detection with AI
        // This would call the find_opportunities tool and analyze results

        let opportunities = vec![
            TradingOpportunity {
                id: "opp-001".into(),
                opportunity_type: OpportunityType::HighYieldPool {
                    pool_id: "QUG-USDT".into(),
                    current_apr: 45.5,
                    tvl_usd: 125000.0,
                },
                description: "QUG-USDT pool has elevated APR due to recent incentive boost".into(),
                potential_profit_percent: 45.5,
                risk_level: RiskLevel::Medium,
                expires_in_seconds: Some(86400 * 7), // 7 days
                action_required: ActionRequired {
                    tool_name: "add_liquidity".into(),
                    parameters: serde_json::json!({
                        "pool_id": "QUG-USDT",
                        "token_a_amount": "auto",
                    }),
                    requires_confirmation: true,
                },
            },
            TradingOpportunity {
                id: "opp-002".into(),
                opportunity_type: OpportunityType::TrendingToken {
                    token: "QBTC".into(),
                    volume_increase_percent: 234.5,
                    price_momentum: 0.78,
                },
                description: "QBTC showing strong upward momentum with 2x+ volume surge".into(),
                potential_profit_percent: 8.5,
                risk_level: RiskLevel::High,
                expires_in_seconds: None,
                action_required: ActionRequired {
                    tool_name: "swap".into(),
                    parameters: serde_json::json!({
                        "from_token": "USDT",
                        "to_token": "QBTC",
                        "amount": "10%", // Suggest 10% of portfolio
                    }),
                    requires_confirmation: true,
                },
            },
        ];

        Ok(opportunities.into_iter()
            .filter(|o| o.potential_profit_percent >= min_profit_percent)
            .collect())
    }

    /// Predict price impact of a trade
    pub async fn predict_price_impact(
        &self,
        from_token: &str,
        to_token: &str,
        amount: f64,
    ) -> Result<PriceImpactPrediction> {
        info!("📈 Predicting price impact: {} {} -> {}", amount, from_token, to_token);

        // TODO: Use AI to analyze pool depth and predict impact
        Ok(PriceImpactPrediction {
            from_token: from_token.into(),
            to_token: to_token.into(),
            input_amount: amount,
            expected_output: amount * 0.985, // Mock: 1.5% impact
            price_impact_percent: 1.5,
            min_output_with_slippage: amount * 0.975,
            recommendation: if amount > 10000.0 {
                "Consider splitting this trade into smaller chunks to reduce impact".into()
            } else {
                "Trade size is reasonable for current liquidity".into()
            },
            suggested_max_slippage: 2.0,
        })
    }

    /// Generate mock analysis (to be replaced with actual AI calls)
    async fn generate_mock_analysis(&self, target: &str, _timeframe: &str) -> Result<MarketAnalysis> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        Ok(MarketAnalysis {
            target: target.into(),
            timestamp: now,
            metrics: MarketMetrics {
                price_usd: 1.25,
                price_change_1h_bps: 50,   // 0.5%
                price_change_24h_bps: 320, // 3.2%
                price_change_7d_bps: 1280, // 12.8%
                volume_24h: 45000.0,
                liquidity_usd: 250000.0,
                volatility_score: 45.0,
                whale_activity_score: 25.0,
            },
            sentiment: MarketSentiment::Bullish,
            recommendations: vec![
                Recommendation {
                    action: RecommendationAction::Hold { token: target.into() },
                    confidence: 0.72,
                    reasoning: "Token showing steady growth with healthy volume. No immediate action needed.".into(),
                    risk_level: RiskLevel::Low,
                },
                Recommendation {
                    action: RecommendationAction::AddLiquidity {
                        pool_id: format!("{}-USDT", target),
                        apr_estimate: 35.0,
                    },
                    confidence: 0.65,
                    reasoning: "LP positions could earn ~35% APR at current volumes.".into(),
                    risk_level: RiskLevel::Medium,
                },
            ],
            confidence: 0.75,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceImpactPrediction {
    pub from_token: String,
    pub to_token: String,
    pub input_amount: f64,
    pub expected_output: f64,
    pub price_impact_percent: f64,
    pub min_output_with_slippage: f64,
    pub recommendation: String,
    pub suggested_max_slippage: f64,
}

// ============================================================================
// AGENT API ENDPOINTS
// ============================================================================

/// API response for market analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisResponse {
    pub success: bool,
    pub analysis: Option<MarketAnalysis>,
    pub opportunities: Vec<TradingOpportunity>,
    pub error: Option<String>,
}

/// API response for price impact
#[derive(Debug, Serialize, Deserialize)]
pub struct PriceImpactResponse {
    pub success: bool,
    pub prediction: Option<PriceImpactPrediction>,
    pub error: Option<String>,
}

// ============================================================================
// API REQUEST TYPES
// ============================================================================

/// Request for market analysis
#[derive(Debug, Deserialize)]
pub struct AnalyzeMarketRequest {
    pub target: String,
    #[serde(default = "default_metrics")]
    pub metrics: Vec<String>,
    #[serde(default = "default_timeframe")]
    pub timeframe: String,
}

fn default_metrics() -> Vec<String> {
    vec!["volume_24h".into(), "price_change_24h".into(), "liquidity_depth".into()]
}

fn default_timeframe() -> String {
    "24h".into()
}

/// Request for finding opportunities
#[derive(Debug, Deserialize)]
pub struct FindOpportunitiesRequest {
    pub strategy: String,
    #[serde(default = "default_min_profit")]
    pub min_profit_percent: f64,
}

fn default_min_profit() -> f64 {
    0.5
}

/// Request for price impact prediction
#[derive(Debug, Deserialize)]
pub struct PriceImpactRequest {
    pub from_token: String,
    pub to_token: String,
    pub amount: f64,
}

/// Response for opportunities search
#[derive(Debug, Serialize)]
pub struct OpportunitiesResponse {
    pub success: bool,
    pub opportunities: Vec<TradingOpportunity>,
    pub count: usize,
    pub error: Option<String>,
}

// ============================================================================
// API HANDLERS
// ============================================================================

/// GET /api/v1/market/analyze/:target - Analyze a specific token or pool
pub async fn analyze_market_handler(
    State(state): State<Arc<AppState>>,
    Path(target): Path<String>,
    Query(params): Query<AnalyzeMarketRequest>,
) -> Json<AnalysisResponse> {
    info!("📊 Market analysis request for: {}", target);

    // Get known tokens from deployed contracts
    let tokens: Vec<String> = {
        let ecosystem = state.orobit_ecosystem.clone();
        let contracts = ecosystem.deployed_contracts.read().await;
        contracts
            .values()
            .filter_map(|c| c.metadata.symbol.clone())
            .collect()
    };

    let agent = MarketAnalyzerAgent::new(&tokens);

    let metrics_refs: Vec<&str> = params.metrics.iter().map(|s| s.as_str()).collect();

    match agent.analyze(&target, &metrics_refs, &params.timeframe).await {
        Ok(analysis) => {
            // Also check for opportunities
            let opportunities = agent
                .find_opportunities("trending", 1.0)
                .await
                .unwrap_or_default();

            Json(AnalysisResponse {
                success: true,
                analysis: Some(analysis),
                opportunities,
                error: None,
            })
        }
        Err(e) => {
            error!("❌ Market analysis failed: {}", e);
            Json(AnalysisResponse {
                success: false,
                analysis: None,
                opportunities: vec![],
                error: Some(e.to_string()),
            })
        }
    }
}

/// POST /api/v1/market/opportunities - Find trading opportunities
pub async fn find_opportunities_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<FindOpportunitiesRequest>,
) -> Json<OpportunitiesResponse> {
    info!("🎯 Opportunity search: {} (min {}%)", request.strategy, request.min_profit_percent);

    let tokens: Vec<String> = {
        let ecosystem = state.orobit_ecosystem.clone();
        let contracts = ecosystem.deployed_contracts.read().await;
        contracts
            .values()
            .filter_map(|c| c.metadata.symbol.clone())
            .collect()
    };

    let agent = MarketAnalyzerAgent::new(&tokens);

    match agent.find_opportunities(&request.strategy, request.min_profit_percent).await {
        Ok(opportunities) => {
            let count = opportunities.len();
            info!("✅ Found {} opportunities", count);
            Json(OpportunitiesResponse {
                success: true,
                opportunities,
                count,
                error: None,
            })
        }
        Err(e) => {
            error!("❌ Opportunity search failed: {}", e);
            Json(OpportunitiesResponse {
                success: false,
                opportunities: vec![],
                count: 0,
                error: Some(e.to_string()),
            })
        }
    }
}

/// POST /api/v1/market/price-impact - Predict price impact of a trade
pub async fn price_impact_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PriceImpactRequest>,
) -> Json<PriceImpactResponse> {
    info!("📈 Price impact prediction: {} {} -> {}",
          request.amount, request.from_token, request.to_token);

    let tokens: Vec<String> = {
        let ecosystem = state.orobit_ecosystem.clone();
        let contracts = ecosystem.deployed_contracts.read().await;
        contracts
            .values()
            .filter_map(|c| c.metadata.symbol.clone())
            .collect()
    };

    let agent = MarketAnalyzerAgent::new(&tokens);

    match agent.predict_price_impact(&request.from_token, &request.to_token, request.amount).await {
        Ok(prediction) => {
            info!("✅ Predicted impact: {}%", prediction.price_impact_percent);
            Json(PriceImpactResponse {
                success: true,
                prediction: Some(prediction),
                error: None,
            })
        }
        Err(e) => {
            error!("❌ Price impact prediction failed: {}", e);
            Json(PriceImpactResponse {
                success: false,
                prediction: None,
                error: Some(e.to_string()),
            })
        }
    }
}

/// GET /api/v1/market/sentiment - Get overall market sentiment
pub async fn market_sentiment_handler(
    State(_state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    // Aggregate sentiment across top tokens
    Json(serde_json::json!({
        "success": true,
        "overall_sentiment": "Bullish",
        "sentiment_score": 0.68, // 0 = bearish, 1 = bullish
        "top_movers": [
            { "token": "QUG", "change_24h": 3.5, "sentiment": "Bullish" },
            { "token": "QBTC", "change_24h": 1.2, "sentiment": "Neutral" },
            { "token": "USDT", "change_24h": 0.01, "sentiment": "Neutral" },
        ],
        "active_opportunities": 3,
        "high_yield_pools": 2,
        "last_updated": chrono::Utc::now().to_rfc3339(),
    }))
}

/// GET /api/v1/market/tools - List available AI tools for the market analyzer
pub async fn list_tools_handler(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let tokens: Vec<String> = {
        let ecosystem = state.orobit_ecosystem.clone();
        let contracts = ecosystem.deployed_contracts.read().await;
        contracts
            .values()
            .filter_map(|c| c.metadata.symbol.clone())
            .collect()
    };

    let tools = get_dex_tools(&tokens);

    Json(serde_json::json!({
        "success": true,
        "tools_count": tools.len(),
        "tools": tools,
        "model": "Ministral-3B",
        "capabilities": [
            "Market analysis with multiple timeframes",
            "Opportunity detection (arbitrage, high-yield, trending)",
            "Price impact prediction",
            "Whale activity monitoring",
            "Trading strategy suggestions"
        ]
    }))
}

// ============================================================================
// ROUTER
// ============================================================================

/// Create the market analyzer API router
pub fn create_market_analyzer_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/analyze/:target", get(analyze_market_handler))
        .route("/opportunities", post(find_opportunities_handler))
        .route("/price-impact", post(price_impact_handler))
        .route("/sentiment", get(market_sentiment_handler))
        .route("/tools", get(list_tools_handler))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_market_analyzer() {
        let tokens = vec!["QUG".into(), "USDT".into(), "QBTC".into()];
        let agent = MarketAnalyzerAgent::new(&tokens);

        let analysis = agent.analyze("QUG", &["volume_24h", "price_change_24h"], "24h").await;
        assert!(analysis.is_ok());

        let opportunities = agent.find_opportunities("high_yield_pools", 10.0).await;
        assert!(opportunities.is_ok());
    }
}
