//! AI Intent Executor for Q-NarwhalKnight DEX
//!
//! This module contains DETERMINISTIC Rust code that executes parsed intents.
//! The AI NEVER directly executes transactions - this code does.
//!
//! Security principles:
//! - All execution is deterministic (same input = same output)
//! - Transaction intents return PREVIEWS, not actual transactions
//! - Only after explicit user confirmation does execution happen
//! - Private keys are NEVER touched by AI code paths

use crate::ai_intent::*;
use anyhow::{anyhow, Result};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// ============================================================================
// EXECUTOR CONTEXT - Access to blockchain state
// ============================================================================

/// Context needed for intent execution
/// This provides read-only access to blockchain state
pub struct ExecutorContext {
    /// Access to wallet balances
    pub balance_provider: Arc<dyn BalanceProvider + Send + Sync>,
    /// Access to DEX pools
    pub pool_provider: Arc<dyn PoolProvider + Send + Sync>,
    /// Access to price oracle
    pub price_provider: Arc<dyn PriceProvider + Send + Sync>,
    /// Access to transaction history
    pub history_provider: Arc<dyn HistoryProvider + Send + Sync>,
    /// Pending transaction previews (confirmation_id -> preview)
    pub pending_previews: Arc<RwLock<HashMap<String, TransactionPreview>>>,
}

// ============================================================================
// PROVIDER TRAITS - Abstract access to blockchain data
// ============================================================================

/// Provider for wallet balances
#[async_trait::async_trait]
pub trait BalanceProvider {
    /// Get balance for a specific token
    async fn get_balance(&self, wallet: &str, token: &str) -> Result<f64>;
    /// Get all balances for a wallet
    async fn get_all_balances(&self, wallet: &str) -> Result<Vec<TokenBalance>>;
}

/// Provider for DEX pool information
#[async_trait::async_trait]
pub trait PoolProvider {
    /// Get info for a specific pool
    async fn get_pool(&self, pool_id: &str) -> Result<Option<PoolInfo>>;
    /// Get all pools
    async fn get_all_pools(&self) -> Result<Vec<PoolInfo>>;
    /// Calculate swap quote
    async fn get_swap_quote(
        &self,
        from_token: &str,
        to_token: &str,
        amount: f64,
        direction: SwapDirection,
    ) -> Result<SwapQuote>;
}

/// Provider for token prices
#[async_trait::async_trait]
pub trait PriceProvider {
    /// Get price of a token in specified currency
    async fn get_price(&self, token: &str, currency: &str) -> Result<(f64, Option<f64>)>;
}

/// Provider for transaction history
#[async_trait::async_trait]
pub trait HistoryProvider {
    /// Get transaction history
    async fn get_history(
        &self,
        wallet: &str,
        limit: u32,
        token_filter: Option<&str>,
    ) -> Result<Vec<TransactionRecord>>;
}

// ============================================================================
// INTENT EXECUTOR
// ============================================================================

/// Executor that processes validated intents
pub struct IntentExecutor {
    context: Arc<ExecutorContext>,
    wallet_address: Option<String>,
}

impl IntentExecutor {
    pub fn new(context: Arc<ExecutorContext>) -> Self {
        Self {
            context,
            wallet_address: None,
        }
    }

    pub fn with_wallet(mut self, wallet: String) -> Self {
        self.wallet_address = Some(wallet);
        self
    }

    /// Execute a parsed intent
    pub async fn execute(&self, parsed: &ParsedIntent) -> Result<ExecutionResult> {
        // Check confidence threshold
        if parsed.confidence < 0.5 {
            return Ok(ExecutionResult::Error {
                code: "LOW_CONFIDENCE".to_string(),
                message: format!(
                    "AI confidence too low ({:.0}%). Please rephrase your request.",
                    parsed.confidence * 100.0
                ),
                suggestion: Some("Try being more specific about what you want to do.".to_string()),
            });
        }

        // Execute based on intent type
        match &parsed.intent {
            // === READ-ONLY INTENTS (always safe) ===
            UserIntent::CheckBalance { token } => {
                self.execute_check_balance(token.as_deref()).await
            }
            UserIntent::GetSwapQuote { from_token, to_token, amount, direction } => {
                self.execute_get_quote(from_token, to_token, amount, direction.clone()).await
            }
            UserIntent::GetPoolInfo { pool_id } => {
                self.execute_get_pool_info(pool_id.as_deref()).await
            }
            UserIntent::GetPrice { token, in_currency } => {
                self.execute_get_price(token, in_currency).await
            }
            UserIntent::GetTransactionHistory { limit, token_filter } => {
                self.execute_get_history(*limit, token_filter.as_deref()).await
            }
            UserIntent::ExplainConcept { topic } => {
                self.execute_explain_concept(topic).await
            }

            // === TRANSACTION INTENTS (require confirmation) ===
            UserIntent::Swap { from_token, to_token, amount, max_slippage_percent, deadline_seconds } => {
                self.create_swap_preview(
                    from_token, to_token, amount,
                    *max_slippage_percent, *deadline_seconds,
                    &parsed.risk_level,
                ).await
            }
            UserIntent::Transfer { to_address, token, amount, memo } => {
                self.create_transfer_preview(
                    to_address, token, amount, memo.as_deref(),
                    &parsed.risk_level,
                ).await
            }
            UserIntent::AddLiquidity { pool_id, token_a_amount, token_b_amount } => {
                self.create_add_liquidity_preview(
                    pool_id, token_a_amount, token_b_amount.as_deref(),
                    &parsed.risk_level,
                ).await
            }
            UserIntent::RemoveLiquidity { pool_id, lp_token_amount, min_token_a, min_token_b } => {
                self.create_remove_liquidity_preview(
                    pool_id, lp_token_amount,
                    min_token_a.as_deref(), min_token_b.as_deref(),
                    &parsed.risk_level,
                ).await
            }

            // === META INTENTS ===
            UserIntent::Clarify { question, options } => {
                Ok(ExecutionResult::Immediate {
                    response: IntentResponse::ChatResponse {
                        message: format!("{}\nOptions: {}", question, options.join(", ")),
                    },
                })
            }
            UserIntent::Unsupported { reason, suggestion } => {
                Ok(ExecutionResult::Error {
                    code: "UNSUPPORTED".to_string(),
                    message: reason.clone(),
                    suggestion: suggestion.clone(),
                })
            }
            UserIntent::Chat { response } => {
                Ok(ExecutionResult::Immediate {
                    response: IntentResponse::ChatResponse {
                        message: response.clone(),
                    },
                })
            }
        }
    }

    // ========================================================================
    // READ-ONLY INTENT EXECUTION
    // ========================================================================

    async fn execute_check_balance(&self, token: Option<&str>) -> Result<ExecutionResult> {
        let wallet = self.wallet_address.as_ref()
            .ok_or_else(|| anyhow!("Wallet address not set"))?;

        match token {
            Some(t) => {
                let balance = self.context.balance_provider.get_balance(wallet, t).await?;
                let (price, _) = self.context.price_provider.get_price(t, "USD").await
                    .unwrap_or((0.0, None));

                Ok(ExecutionResult::Immediate {
                    response: IntentResponse::Balances {
                        balances: vec![TokenBalance {
                            token: t.to_string(),
                            balance,
                            value_usd: balance * price,
                            price_usd: price,
                        }],
                        total_value_usd: balance * price,
                    },
                })
            }
            None => {
                let balances = self.context.balance_provider.get_all_balances(wallet).await?;
                let total_value: f64 = balances.iter().map(|b| b.value_usd).sum();

                Ok(ExecutionResult::Immediate {
                    response: IntentResponse::Balances {
                        balances,
                        total_value_usd: total_value,
                    },
                })
            }
        }
    }

    async fn execute_get_quote(
        &self,
        from_token: &str,
        to_token: &str,
        amount: &str,
        direction: SwapDirection,
    ) -> Result<ExecutionResult> {
        let amount_f64 = self.parse_amount(amount, from_token).await?;

        let quote = self.context.pool_provider
            .get_swap_quote(from_token, to_token, amount_f64, direction)
            .await?;

        Ok(ExecutionResult::Immediate {
            response: IntentResponse::SwapQuote { quote },
        })
    }

    async fn execute_get_pool_info(&self, pool_id: Option<&str>) -> Result<ExecutionResult> {
        match pool_id {
            Some(id) => {
                let pool = self.context.pool_provider.get_pool(id).await?
                    .ok_or_else(|| anyhow!("Pool {} not found", id))?;

                Ok(ExecutionResult::Immediate {
                    response: IntentResponse::PoolInfo {
                        pools: vec![pool],
                    },
                })
            }
            None => {
                let pools = self.context.pool_provider.get_all_pools().await?;

                Ok(ExecutionResult::Immediate {
                    response: IntentResponse::PoolInfo { pools },
                })
            }
        }
    }

    async fn execute_get_price(&self, token: &str, currency: &str) -> Result<ExecutionResult> {
        let (price, change_24h) = self.context.price_provider.get_price(token, currency).await?;

        Ok(ExecutionResult::Immediate {
            response: IntentResponse::Price {
                token: token.to_string(),
                price,
                currency: currency.to_string(),
                change_24h,
            },
        })
    }

    async fn execute_get_history(
        &self,
        limit: u32,
        token_filter: Option<&str>,
    ) -> Result<ExecutionResult> {
        let wallet = self.wallet_address.as_ref()
            .ok_or_else(|| anyhow!("Wallet address not set"))?;

        let transactions = self.context.history_provider
            .get_history(wallet, limit, token_filter)
            .await?;

        Ok(ExecutionResult::Immediate {
            response: IntentResponse::History { transactions },
        })
    }

    async fn execute_explain_concept(&self, topic: &str) -> Result<ExecutionResult> {
        // Static explanations for common DeFi concepts
        // AI can provide more dynamic explanations through chat
        let explanation = match topic.to_lowercase().as_str() {
            "impermanent loss" | "il" => {
                "Impermanent loss occurs when you provide liquidity to a pool and the price ratio \
                of the tokens changes. If you had simply held the tokens, you might have more value \
                than what you can withdraw from the pool. It's 'impermanent' because the loss only \
                becomes permanent when you withdraw. The larger the price change, the larger the loss."
            }
            "slippage" => {
                "Slippage is the difference between the expected price of a trade and the actual \
                price when the trade executes. It happens because the market moves between when you \
                submit a transaction and when it confirms. Setting a max slippage tolerance protects \
                you from unexpectedly bad trades."
            }
            "liquidity pool" | "lp" => {
                "A liquidity pool is a collection of tokens locked in a smart contract. It enables \
                decentralized trading by providing the tokens needed for swaps. Liquidity providers \
                earn fees from trades in proportion to their share of the pool."
            }
            "apy" | "apr" => {
                "APY (Annual Percentage Yield) includes compound interest, while APR (Annual \
                Percentage Rate) does not. For liquidity pools, APY estimates your yearly return \
                from trading fees and rewards, assuming you reinvest earnings."
            }
            "amm" | "automated market maker" => {
                "An Automated Market Maker (AMM) is a type of DEX protocol that prices assets using \
                a mathematical formula instead of an order book. The most common formula is x*y=k, \
                where x and y are the amounts of two tokens in a pool."
            }
            _ => {
                "I don't have a built-in explanation for that topic. Try asking in a more \
                conversational way and I'll do my best to explain it."
            }
        };

        Ok(ExecutionResult::Immediate {
            response: IntentResponse::Explanation {
                topic: topic.to_string(),
                explanation: explanation.to_string(),
            },
        })
    }

    // ========================================================================
    // TRANSACTION PREVIEW CREATION
    // ========================================================================

    async fn create_swap_preview(
        &self,
        from_token: &str,
        to_token: &str,
        amount: &str,
        max_slippage: f64,
        deadline_seconds: u64,
        risk_level: &RiskLevel,
    ) -> Result<ExecutionResult> {
        let wallet = self.wallet_address.as_ref()
            .ok_or_else(|| anyhow!("Wallet address not set"))?;

        // Parse the amount
        let amount_f64 = self.parse_amount(amount, from_token).await?;

        // Check balance
        let balance = self.context.balance_provider.get_balance(wallet, from_token).await?;
        if amount_f64 > balance {
            return Ok(ExecutionResult::Error {
                code: "INSUFFICIENT_BALANCE".to_string(),
                message: format!(
                    "Insufficient {} balance. You have {}, but trying to swap {}.",
                    from_token, balance, amount_f64
                ),
                suggestion: Some(format!("Try swapping {} or less.", balance)),
            });
        }

        // Get quote
        let quote = self.context.pool_provider
            .get_swap_quote(from_token, to_token, amount_f64, SwapDirection::ExactIn)
            .await?;

        // Calculate min output with slippage
        let min_output = quote.output_amount * (1.0 - max_slippage / 100.0);

        // Generate warnings
        let mut warnings = Vec::new();

        if quote.price_impact_percent > 1.0 {
            warnings.push(Warning {
                level: WarningLevel::Caution,
                message: format!("Price impact is {:.2}%", quote.price_impact_percent),
            });
        }
        if quote.price_impact_percent > 5.0 {
            warnings.push(Warning {
                level: WarningLevel::Warning,
                message: "High price impact! Consider splitting into smaller trades.".to_string(),
            });
        }
        if quote.price_impact_percent > 10.0 {
            warnings.push(Warning {
                level: WarningLevel::Danger,
                message: "Very high price impact! You will lose significant value.".to_string(),
            });
        }

        // Create confirmation ID
        let confirmation_id = Uuid::new_v4().to_string();

        // Generate confirmation code for critical transactions
        let confirmation_code = if *risk_level >= RiskLevel::Critical {
            Some(self.generate_confirmation_code())
        } else {
            None
        };

        let preview = TransactionPreview {
            confirmation_id: confirmation_id.clone(),
            intent: UserIntent::Swap {
                from_token: from_token.to_string(),
                to_token: to_token.to_string(),
                amount: amount.to_string(),
                max_slippage_percent: max_slippage,
                deadline_seconds,
            },
            summary: TransactionSummary::Swap {
                input: format!("{} {}", amount_f64, from_token),
                output: format!("{:.6} {}", quote.output_amount, to_token),
                rate: quote.effective_rate,
                price_impact: quote.price_impact_percent,
                min_output,
            },
            warnings,
            expires_at: Utc::now().timestamp() + (deadline_seconds as i64),
            confirmation_code,
            estimated_fee: quote.fee,
            fee_token: quote.fee_token,
        };

        // Store the preview
        {
            let mut previews = self.context.pending_previews.write().await;
            previews.insert(confirmation_id.clone(), preview.clone());
        }

        Ok(ExecutionResult::RequiresConfirmation { preview })
    }

    async fn create_transfer_preview(
        &self,
        to_address: &str,
        token: &str,
        amount: &str,
        memo: Option<&str>,
        risk_level: &RiskLevel,
    ) -> Result<ExecutionResult> {
        let wallet = self.wallet_address.as_ref()
            .ok_or_else(|| anyhow!("Wallet address not set"))?;

        // Parse the amount
        let amount_f64 = self.parse_amount(amount, token).await?;

        // Check balance
        let balance = self.context.balance_provider.get_balance(wallet, token).await?;
        if amount_f64 > balance {
            return Ok(ExecutionResult::Error {
                code: "INSUFFICIENT_BALANCE".to_string(),
                message: format!(
                    "Insufficient {} balance. You have {}, but trying to send {}.",
                    token, balance, amount_f64
                ),
                suggestion: Some(format!("Try sending {} or less.", balance)),
            });
        }

        // Generate warnings
        let mut warnings = Vec::new();

        // Check if sending to self
        if to_address == wallet {
            warnings.push(Warning {
                level: WarningLevel::Caution,
                message: "You are sending to your own address.".to_string(),
            });
        }

        // Large transfer warning
        if amount_f64 > balance * 0.5 {
            warnings.push(Warning {
                level: WarningLevel::Warning,
                message: format!("This transfer is more than 50% of your {} balance.", token),
            });
        }

        // Create confirmation ID
        let confirmation_id = Uuid::new_v4().to_string();

        // Generate confirmation code for critical transactions
        let confirmation_code = if *risk_level >= RiskLevel::Critical {
            Some(self.generate_confirmation_code())
        } else {
            None
        };

        // Estimate fee (simplified - real implementation would calculate properly)
        let estimated_fee = 0.001; // 0.001 QUG base fee

        let preview = TransactionPreview {
            confirmation_id: confirmation_id.clone(),
            intent: UserIntent::Transfer {
                to_address: to_address.to_string(),
                token: token.to_string(),
                amount: amount.to_string(),
                memo: memo.map(|s| s.to_string()),
            },
            summary: TransactionSummary::Transfer {
                amount: format!("{}", amount_f64),
                token: token.to_string(),
                to: to_address.to_string(),
                memo: memo.map(|s| s.to_string()),
            },
            warnings,
            expires_at: Utc::now().timestamp() + 300, // 5 minute expiry
            confirmation_code,
            estimated_fee,
            fee_token: "QUG".to_string(),
        };

        // Store the preview
        {
            let mut previews = self.context.pending_previews.write().await;
            previews.insert(confirmation_id.clone(), preview.clone());
        }

        Ok(ExecutionResult::RequiresConfirmation { preview })
    }

    async fn create_add_liquidity_preview(
        &self,
        pool_id: &str,
        token_a_amount: &str,
        token_b_amount: Option<&str>,
        risk_level: &RiskLevel,
    ) -> Result<ExecutionResult> {
        let wallet = self.wallet_address.as_ref()
            .ok_or_else(|| anyhow!("Wallet address not set"))?;

        // Get pool info
        let pool = self.context.pool_provider.get_pool(pool_id).await?
            .ok_or_else(|| anyhow!("Pool {} not found", pool_id))?;

        // Parse amount A
        let amount_a = self.parse_amount(token_a_amount, &pool.token_a).await?;

        // Calculate amount B if not provided (maintain pool ratio)
        let amount_b = match token_b_amount {
            Some(b) => self.parse_amount(b, &pool.token_b).await?,
            None => {
                // Calculate based on pool ratio
                let ratio = pool.reserve_b / pool.reserve_a;
                amount_a * ratio
            }
        };

        // Check balances
        let balance_a = self.context.balance_provider.get_balance(wallet, &pool.token_a).await?;
        let balance_b = self.context.balance_provider.get_balance(wallet, &pool.token_b).await?;

        if amount_a > balance_a {
            return Ok(ExecutionResult::Error {
                code: "INSUFFICIENT_BALANCE".to_string(),
                message: format!("Insufficient {} balance.", pool.token_a),
                suggestion: None,
            });
        }
        if amount_b > balance_b {
            return Ok(ExecutionResult::Error {
                code: "INSUFFICIENT_BALANCE".to_string(),
                message: format!("Insufficient {} balance.", pool.token_b),
                suggestion: None,
            });
        }

        // Calculate LP tokens and pool share
        let total_value = (amount_a / pool.reserve_a) * pool.total_liquidity_usd;
        let lp_tokens = (amount_a / pool.reserve_a) * 1000.0; // Simplified
        let share_of_pool = (amount_a / (pool.reserve_a + amount_a)) * 100.0;

        // Warnings
        let mut warnings = Vec::new();
        warnings.push(Warning {
            level: WarningLevel::Info,
            message: format!("You may experience impermanent loss if {} or {} prices change.", pool.token_a, pool.token_b),
        });

        // Create confirmation
        let confirmation_id = Uuid::new_v4().to_string();
        let confirmation_code = if *risk_level >= RiskLevel::Critical {
            Some(self.generate_confirmation_code())
        } else {
            None
        };

        let preview = TransactionPreview {
            confirmation_id: confirmation_id.clone(),
            intent: UserIntent::AddLiquidity {
                pool_id: pool_id.to_string(),
                token_a_amount: token_a_amount.to_string(),
                token_b_amount: Some(format!("{}", amount_b)),
            },
            summary: TransactionSummary::AddLiquidity {
                token_a: pool.token_a.clone(),
                amount_a,
                token_b: pool.token_b.clone(),
                amount_b,
                lp_tokens,
                share_of_pool,
            },
            warnings,
            expires_at: Utc::now().timestamp() + 300,
            confirmation_code,
            estimated_fee: 0.001,
            fee_token: "QUG".to_string(),
        };

        // Store the preview
        {
            let mut previews = self.context.pending_previews.write().await;
            previews.insert(confirmation_id.clone(), preview.clone());
        }

        Ok(ExecutionResult::RequiresConfirmation { preview })
    }

    async fn create_remove_liquidity_preview(
        &self,
        pool_id: &str,
        lp_token_amount: &str,
        min_token_a: Option<&str>,
        min_token_b: Option<&str>,
        risk_level: &RiskLevel,
    ) -> Result<ExecutionResult> {
        // Get pool info
        let pool = self.context.pool_provider.get_pool(pool_id).await?
            .ok_or_else(|| anyhow!("Pool {} not found", pool_id))?;

        // Parse LP amount (simplified - real implementation would check LP token balance)
        let lp_amount = if lp_token_amount.to_uppercase() == "MAX" {
            100.0 // Placeholder - would get actual LP balance
        } else if lp_token_amount.ends_with('%') {
            let pct = lp_token_amount.trim_end_matches('%').parse::<f64>()
                .map_err(|_| anyhow!("Invalid percentage"))?;
            100.0 * (pct / 100.0) // Placeholder
        } else {
            lp_token_amount.parse::<f64>()
                .map_err(|_| anyhow!("Invalid LP token amount"))?
        };

        // Calculate expected returns (simplified)
        let share = lp_amount / 1000.0; // Simplified
        let amount_a = pool.reserve_a * share;
        let amount_b = pool.reserve_b * share;

        // Warnings
        let mut warnings = Vec::new();
        warnings.push(Warning {
            level: WarningLevel::Info,
            message: "Removing liquidity is irreversible. Make sure you want to do this.".to_string(),
        });

        // Create confirmation
        let confirmation_id = Uuid::new_v4().to_string();
        let confirmation_code = if *risk_level >= RiskLevel::Critical {
            Some(self.generate_confirmation_code())
        } else {
            None
        };

        let preview = TransactionPreview {
            confirmation_id: confirmation_id.clone(),
            intent: UserIntent::RemoveLiquidity {
                pool_id: pool_id.to_string(),
                lp_token_amount: lp_token_amount.to_string(),
                min_token_a: min_token_a.map(|s| s.to_string()),
                min_token_b: min_token_b.map(|s| s.to_string()),
            },
            summary: TransactionSummary::RemoveLiquidity {
                lp_tokens: lp_amount,
                token_a: pool.token_a.clone(),
                amount_a,
                token_b: pool.token_b.clone(),
                amount_b,
            },
            warnings,
            expires_at: Utc::now().timestamp() + 300,
            confirmation_code,
            estimated_fee: 0.001,
            fee_token: "QUG".to_string(),
        };

        // Store the preview
        {
            let mut previews = self.context.pending_previews.write().await;
            previews.insert(confirmation_id.clone(), preview.clone());
        }

        Ok(ExecutionResult::RequiresConfirmation { preview })
    }

    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    /// Parse an amount string, handling MAX and percentages
    async fn parse_amount(&self, amount: &str, token: &str) -> Result<f64> {
        let amount_upper = amount.to_uppercase();

        if amount_upper == "MAX" {
            // Get full balance
            let wallet = self.wallet_address.as_ref()
                .ok_or_else(|| anyhow!("Wallet address not set"))?;
            let balance = self.context.balance_provider.get_balance(wallet, token).await?;
            return Ok(balance);
        }

        if amount_upper.ends_with('%') {
            // Percentage of balance
            let pct = amount_upper.trim_end_matches('%').parse::<f64>()
                .map_err(|_| anyhow!("Invalid percentage: {}", amount))?;

            let wallet = self.wallet_address.as_ref()
                .ok_or_else(|| anyhow!("Wallet address not set"))?;
            let balance = self.context.balance_provider.get_balance(wallet, token).await?;
            return Ok(balance * (pct / 100.0));
        }

        // Regular number
        amount.parse::<f64>()
            .map_err(|_| anyhow!("Invalid amount: {}", amount))
    }

    /// Generate a 6-character confirmation code for critical transactions
    fn generate_confirmation_code(&self) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let chars: Vec<char> = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789".chars().collect();
        (0..6).map(|_| chars[rng.gen_range(0..chars.len())]).collect()
    }

    /// Confirm and execute a pending transaction
    pub async fn confirm_transaction(
        &self,
        confirmation_id: &str,
        confirmation_code: Option<&str>,
    ) -> Result<ExecutionResult> {
        // Get the preview
        let preview = {
            let previews = self.context.pending_previews.read().await;
            previews.get(confirmation_id).cloned()
        };

        let preview = preview.ok_or_else(|| anyhow!("Transaction not found or expired"))?;

        // Check expiration
        if Utc::now().timestamp() > preview.expires_at {
            // Remove expired preview
            let mut previews = self.context.pending_previews.write().await;
            previews.remove(confirmation_id);
            return Err(anyhow!("Transaction preview has expired. Please start over."));
        }

        // Check confirmation code if required
        if let Some(required_code) = &preview.confirmation_code {
            match confirmation_code {
                Some(code) if code == required_code => {
                    // Code matches - proceed
                }
                Some(code) => {
                    return Err(anyhow!(
                        "Invalid confirmation code. Expected: {}, Got: {}",
                        required_code, code
                    ));
                }
                None => {
                    return Err(anyhow!(
                        "This transaction requires confirmation code: {}",
                        required_code
                    ));
                }
            }
        }

        // Remove the preview (one-time use)
        {
            let mut previews = self.context.pending_previews.write().await;
            previews.remove(confirmation_id);
        }

        // Execute the actual transaction
        // This would call into the actual DEX/blockchain execution layer
        // For now, return success
        info!("[INTENT EXECUTOR] Transaction confirmed and executed: {}", confirmation_id);

        Ok(ExecutionResult::Immediate {
            response: IntentResponse::ChatResponse {
                message: format!("Transaction submitted! Confirmation ID: {}", confirmation_id),
            },
        })
    }

    /// Cancel a pending transaction
    pub async fn cancel_transaction(&self, confirmation_id: &str) -> Result<()> {
        let mut previews = self.context.pending_previews.write().await;
        if previews.remove(confirmation_id).is_some() {
            info!("[INTENT EXECUTOR] Transaction cancelled: {}", confirmation_id);
            Ok(())
        } else {
            Err(anyhow!("Transaction not found"))
        }
    }
}

// ============================================================================
// MOCK PROVIDERS (for testing)
// ============================================================================

/// Mock balance provider for testing
pub struct MockBalanceProvider;

#[async_trait::async_trait]
impl BalanceProvider for MockBalanceProvider {
    async fn get_balance(&self, _wallet: &str, token: &str) -> Result<f64> {
        // Return mock balances
        match token.to_uppercase().as_str() {
            "QUG" => Ok(1000.0),
            "USDT" => Ok(500.0),
            "QBTC" => Ok(0.1),
            "QETH" => Ok(5.0),
            _ => Ok(0.0),
        }
    }

    async fn get_all_balances(&self, _wallet: &str) -> Result<Vec<TokenBalance>> {
        Ok(vec![
            TokenBalance { token: "QUG".to_string(), balance: 1000.0, value_usd: 10000.0, price_usd: 10.0 },
            TokenBalance { token: "USDT".to_string(), balance: 500.0, value_usd: 500.0, price_usd: 1.0 },
            TokenBalance { token: "QBTC".to_string(), balance: 0.1, value_usd: 4500.0, price_usd: 45000.0 },
            TokenBalance { token: "QETH".to_string(), balance: 5.0, value_usd: 15000.0, price_usd: 3000.0 },
        ])
    }
}

/// Mock pool provider for testing
pub struct MockPoolProvider;

#[async_trait::async_trait]
impl PoolProvider for MockPoolProvider {
    async fn get_pool(&self, pool_id: &str) -> Result<Option<PoolInfo>> {
        Ok(Some(PoolInfo {
            pool_id: pool_id.to_string(),
            token_a: "QUG".to_string(),
            token_b: "USDT".to_string(),
            reserve_a: 100000.0,
            reserve_b: 1000000.0,
            total_liquidity_usd: 2000000.0,
            apy_24h: Some(25.5),
            volume_24h: 500000.0,
        }))
    }

    async fn get_all_pools(&self) -> Result<Vec<PoolInfo>> {
        Ok(vec![
            PoolInfo {
                pool_id: "QUG-USDT".to_string(),
                token_a: "QUG".to_string(),
                token_b: "USDT".to_string(),
                reserve_a: 100000.0,
                reserve_b: 1000000.0,
                total_liquidity_usd: 2000000.0,
                apy_24h: Some(25.5),
                volume_24h: 500000.0,
            },
            PoolInfo {
                pool_id: "QBTC-USDT".to_string(),
                token_a: "QBTC".to_string(),
                token_b: "USDT".to_string(),
                reserve_a: 10.0,
                reserve_b: 450000.0,
                total_liquidity_usd: 900000.0,
                apy_24h: Some(15.0),
                volume_24h: 200000.0,
            },
        ])
    }

    async fn get_swap_quote(
        &self,
        from_token: &str,
        to_token: &str,
        amount: f64,
        _direction: SwapDirection,
    ) -> Result<SwapQuote> {
        // Simplified AMM calculation
        let rate = if from_token == "QUG" { 10.0 } else { 0.1 };
        let output = amount * rate * 0.997; // 0.3% fee
        let price_impact = (amount / 100000.0) * 100.0; // Simplified

        Ok(SwapQuote {
            input_amount: amount,
            input_token: from_token.to_string(),
            output_amount: output,
            output_token: to_token.to_string(),
            effective_rate: rate,
            price_impact_percent: price_impact.min(50.0),
            fee: amount * 0.003,
            fee_token: from_token.to_string(),
            route: vec![from_token.to_string(), to_token.to_string()],
            expires_at: Utc::now().timestamp() + 60,
        })
    }
}

/// Mock price provider for testing
pub struct MockPriceProvider;

#[async_trait::async_trait]
impl PriceProvider for MockPriceProvider {
    async fn get_price(&self, token: &str, _currency: &str) -> Result<(f64, Option<f64>)> {
        match token.to_uppercase().as_str() {
            "QUG" => Ok((10.0, Some(5.5))),
            "USDT" => Ok((1.0, Some(0.0))),
            "QBTC" => Ok((45000.0, Some(-2.3))),
            "QETH" => Ok((3000.0, Some(3.8))),
            _ => Err(anyhow!("Unknown token: {}", token)),
        }
    }
}

/// Mock history provider for testing
pub struct MockHistoryProvider;

#[async_trait::async_trait]
impl HistoryProvider for MockHistoryProvider {
    async fn get_history(
        &self,
        _wallet: &str,
        limit: u32,
        _token_filter: Option<&str>,
    ) -> Result<Vec<TransactionRecord>> {
        let mut records = Vec::new();
        let limit = limit.min(10) as usize;

        for i in 0..limit {
            let mut details = HashMap::new();
            details.insert("from_token".to_string(), "QUG".to_string());
            details.insert("to_token".to_string(), "USDT".to_string());
            details.insert("amount".to_string(), format!("{}", 100 + i * 50));

            records.push(TransactionRecord {
                tx_hash: format!("0x{:064x}", i),
                tx_type: "swap".to_string(),
                timestamp: Utc::now().timestamp() - (i as i64 * 3600),
                status: "confirmed".to_string(),
                details,
            });
        }

        Ok(records)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_executor() -> IntentExecutor {
        let context = Arc::new(ExecutorContext {
            balance_provider: Arc::new(MockBalanceProvider),
            pool_provider: Arc::new(MockPoolProvider),
            price_provider: Arc::new(MockPriceProvider),
            history_provider: Arc::new(MockHistoryProvider),
            pending_previews: Arc::new(RwLock::new(HashMap::new())),
        });

        IntentExecutor::new(context).with_wallet("qnk1test123".to_string())
    }

    #[tokio::test]
    async fn test_execute_check_balance() {
        let executor = create_test_executor().await;

        let intent = ParsedIntent {
            intent: UserIntent::CheckBalance { token: Some("QUG".to_string()) },
            confidence: 0.9,
            requires_confirmation: false,
            risk_level: RiskLevel::None,
            extracted_entities: vec![],
            reasoning: None,
        };

        let result = executor.execute(&intent).await.unwrap();

        match result {
            ExecutionResult::Immediate { response: IntentResponse::Balances { balances, .. } } => {
                assert_eq!(balances.len(), 1);
                assert_eq!(balances[0].token, "QUG");
                assert_eq!(balances[0].balance, 1000.0);
            }
            _ => panic!("Expected Immediate Balances response"),
        }
    }

    #[tokio::test]
    async fn test_execute_swap_preview() {
        let executor = create_test_executor().await;

        let intent = ParsedIntent {
            intent: UserIntent::Swap {
                from_token: "QUG".to_string(),
                to_token: "USDT".to_string(),
                amount: "100".to_string(),
                max_slippage_percent: 1.0,
                deadline_seconds: 300,
            },
            confidence: 0.9,
            requires_confirmation: true,
            risk_level: RiskLevel::Medium,
            extracted_entities: vec![],
            reasoning: None,
        };

        let result = executor.execute(&intent).await.unwrap();

        match result {
            ExecutionResult::RequiresConfirmation { preview } => {
                assert!(!preview.confirmation_id.is_empty());
                assert!(matches!(preview.summary, TransactionSummary::Swap { .. }));
            }
            _ => panic!("Expected RequiresConfirmation response"),
        }
    }

    #[tokio::test]
    async fn test_insufficient_balance() {
        let executor = create_test_executor().await;

        let intent = ParsedIntent {
            intent: UserIntent::Swap {
                from_token: "QUG".to_string(),
                to_token: "USDT".to_string(),
                amount: "10000".to_string(), // More than balance
                max_slippage_percent: 1.0,
                deadline_seconds: 300,
            },
            confidence: 0.9,
            requires_confirmation: true,
            risk_level: RiskLevel::Critical,
            extracted_entities: vec![],
            reasoning: None,
        };

        let result = executor.execute(&intent).await.unwrap();

        match result {
            ExecutionResult::Error { code, .. } => {
                assert_eq!(code, "INSUFFICIENT_BALANCE");
            }
            _ => panic!("Expected Error response"),
        }
    }
}
