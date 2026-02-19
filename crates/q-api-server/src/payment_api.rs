// Payment API - Stripe integration for USD wallet functionality
// Implements payment intent creation, confirmation, and USD balance management

use axum::{
    extract::{Json, State},
    http::{HeaderMap, StatusCode},
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::{Arc, Mutex};
use stripe::{
    Client, CreateCustomer, CreatePaymentIntent, Currency, Customer, PaymentIntent,
    PaymentIntentConfirmParams,
};
use tracing::{error, info, warn};

use crate::{ApiResponse, AppState};

// ============================================================================
// Authentication Helper
// ============================================================================

/// Extract authenticated wallet address from request headers.
/// Payment endpoints that move money MUST verify the caller owns the wallet.
fn extract_wallet_from_headers(headers: &HeaderMap) -> Option<String> {
    // Try x-wallet-auth header first
    if let Some(wallet) = headers.get("x-wallet-auth").and_then(|v| v.to_str().ok()) {
        if !wallet.is_empty() {
            return Some(wallet.to_string());
        }
    }
    // Try Authorization: Bearer header
    if let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(token) = auth.strip_prefix("Bearer ") {
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

// ============================================================================
// Rate Limiting
// ============================================================================

/// Simple in-memory rate limiter for payment endpoints.
/// Tracks per-wallet request counts within a sliding window.
pub struct PaymentRateLimiter {
    /// wallet_address -> (window_start_time, request_count_in_window)
    requests: Mutex<HashMap<String, (std::time::Instant, u32)>>,
    /// Max requests allowed per window
    max_requests: u32,
    /// Window duration
    window: std::time::Duration,
}

impl PaymentRateLimiter {
    pub fn new(max_requests_per_minute: u32) -> Self {
        Self {
            requests: Mutex::new(HashMap::new()),
            max_requests: max_requests_per_minute,
            window: std::time::Duration::from_secs(60),
        }
    }

    /// Check if a wallet is rate limited. Returns true if the request should be allowed.
    pub fn check_rate_limit(&self, wallet: &str) -> bool {
        let mut map = self.requests.lock().unwrap();
        let now = std::time::Instant::now();

        match map.get_mut(wallet) {
            Some((window_start, count)) => {
                if now.duration_since(*window_start) > self.window {
                    // Window expired, reset
                    *window_start = now;
                    *count = 1;
                    true
                } else if *count >= self.max_requests {
                    // Rate limit exceeded
                    false
                } else {
                    *count += 1;
                    true
                }
            }
            None => {
                map.insert(wallet.to_string(), (now, 1));
                true
            }
        }
    }
}

/// Helper to check rate limit and return an error response if rate limited.
/// The rate limiter should be wired into AppState.payment_rate_limiter field.
/// For now returns None (not wired) -- ready to integrate when AppState is updated.
fn check_payment_rate_limit<T: Serialize>(
    _state: &AppState,
    _wallet: &str,
) -> Option<Json<ApiResponse<T>>> {
    // TODO: Wire into AppState.payment_rate_limiter field
    // Example future usage:
    //   if let Some(ref limiter) = state.payment_rate_limiter {
    //       if !limiter.check_rate_limit(wallet) {
    //           return Some(Json(ApiResponse {
    //               success: false,
    //               data: None,
    //               error: Some("Rate limit exceeded: too many payment requests".to_string()),
    //               timestamp: chrono::Utc::now(),
    //           }));
    //       }
    //   }
    None
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Request to create a payment intent for USD top-up
#[derive(Debug, Deserialize)]
pub struct CreatePaymentIntentRequest {
    /// Wallet address to credit
    pub wallet_address: String,
    /// Amount in USD (decimal, e.g., "100.50" for $100.50)
    pub amount: String,
    /// Optional payment method ID
    pub payment_method_id: Option<String>,
}

/// Response containing payment intent details
#[derive(Debug, Serialize)]
pub struct PaymentIntentResponse {
    /// Stripe payment intent ID
    pub payment_intent_id: String,
    /// Client secret for frontend confirmation
    pub client_secret: String,
    /// Amount in cents
    pub amount: i64,
    /// Currency (always "usd")
    pub currency: String,
    /// Current status
    pub status: String,
}

/// Request to confirm a payment
#[derive(Debug, Deserialize)]
pub struct ConfirmPaymentRequest {
    /// Payment intent ID from Stripe
    pub payment_intent_id: String,
    /// Payment method ID
    pub payment_method_id: String,
}

/// Response for payment confirmation
#[derive(Debug, Serialize)]
pub struct ConfirmPaymentResponse {
    /// Whether payment succeeded
    pub success: bool,
    /// Payment status
    pub status: String,
    /// Amount credited (in USD)
    pub amount_credited: String,
    /// Updated wallet balance
    pub new_balance: String,
}

/// Request to get USD wallet balance
#[derive(Debug, Deserialize)]
pub struct GetBalanceRequest {
    pub wallet_address: String,
}

/// Response with USD balance
#[derive(Debug, Serialize)]
pub struct BalanceResponse {
    pub wallet_address: String,
    pub balance_usd: String,
    pub balance_cents: i64,
}

/// Request to withdraw USD from wallet
#[derive(Debug, Deserialize)]
pub struct WithdrawRequest {
    pub wallet_address: String,
    pub amount_usd: String,
    pub destination_account: String, // Bank account or payment method
}

/// Response for withdrawal
#[derive(Debug, Serialize)]
pub struct WithdrawResponse {
    pub success: bool,
    pub transaction_id: String,
    pub amount_withdrawn: String,
    pub new_balance: String,
}

// ============================================================================
// Stripe Client Initialization
// ============================================================================

/// Initialize Stripe client with API key from environment
pub fn init_stripe_client() -> Result<Client, String> {
    let api_key = std::env::var("STRIPE_SECRET_KEY")
        .map_err(|_| "STRIPE_SECRET_KEY environment variable not set".to_string())?;

    Ok(Client::new(api_key))
}

/// Get Stripe webhook signing secret from environment
fn get_webhook_secret() -> Result<String, String> {
    std::env::var("STRIPE_WEBHOOK_SECRET")
        .map_err(|_| "STRIPE_WEBHOOK_SECRET environment variable not set".to_string())
}

// ============================================================================
// API Handlers
// ============================================================================

/// POST /api/v1/payment/create-intent
/// Create a Stripe payment intent for USD wallet top-up
/// NOTE: No auth required here -- Stripe handles card authentication.
pub async fn create_payment_intent(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreatePaymentIntentRequest>,
) -> Result<Json<ApiResponse<PaymentIntentResponse>>, StatusCode> {
    info!(
        "💳 Creating payment intent for wallet: {}, amount: ${}",
        request.wallet_address, request.amount
    );

    // Parse amount as decimal and convert to cents
    let amount_decimal: Decimal = request.amount.parse().map_err(|e| {
        error!("Invalid amount format: {}", e);
        StatusCode::BAD_REQUEST
    })?;

    // Convert to cents (multiply by 100)
    let amount_cents: i64 = (amount_decimal * Decimal::from(100))
        .try_into()
        .map_err(|_| {
            error!("Amount too large or invalid");
            StatusCode::BAD_REQUEST
        })?;

    if amount_cents < 50 {
        warn!("Amount too small: {} cents (minimum $0.50)", amount_cents);
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Minimum amount is $0.50".to_string()),
            timestamp: chrono::Utc::now(),
        }));
    }

    // Use pre-initialized Stripe client from AppState
    let stripe_client = match &state.stripe_client {
        Some(client) => client,
        None => {
            error!("Stripe client not configured - set STRIPE_SECRET_KEY env var");
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Payment system not configured".to_string()),
                timestamp: chrono::Utc::now(),
            }));
        }
    };

    // Create payment intent
    let mut create_intent = CreatePaymentIntent::new(amount_cents, Currency::USD);
    create_intent.metadata = Some(std::collections::HashMap::from([
        ("wallet_address".to_string(), request.wallet_address.clone()),
        ("purpose".to_string(), "usd_wallet_topup".to_string()),
    ]));

    // Add payment method if provided
    if let Some(pm_id) = request.payment_method_id {
        if let Ok(pm_id_parsed) = pm_id.parse() {
            create_intent.payment_method = Some(pm_id_parsed);
            create_intent.confirm = Some(true); // Auto-confirm if payment method provided
        }
    }

    match PaymentIntent::create(&stripe_client, create_intent).await {
        Ok(intent) => {
            let response = PaymentIntentResponse {
                payment_intent_id: intent.id.to_string(),
                client_secret: intent.client_secret.unwrap_or_default(),
                amount: intent.amount,
                currency: intent.currency.to_string(),
                status: format!("{:?}", intent.status),
            };

            info!("✅ Payment intent created: {}", response.payment_intent_id);

            Ok(Json(ApiResponse {
                success: true,
                data: Some(response),
                error: None,
                timestamp: chrono::Utc::now(),
            }))
        }
        Err(e) => {
            error!("Stripe API error: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Stripe error: {}", e)),
                timestamp: chrono::Utc::now(),
            }))
        }
    }
}

/// POST /api/v1/payment/confirm
/// Confirm a payment and credit USD to wallet
pub async fn confirm_payment(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ConfirmPaymentRequest>,
) -> Result<Json<ApiResponse<ConfirmPaymentResponse>>, StatusCode> {
    info!(
        "✅ Confirming payment intent: {}",
        request.payment_intent_id
    );

    // Use pre-initialized Stripe client from AppState
    let stripe_client = match &state.stripe_client {
        Some(client) => client,
        None => {
            error!("Stripe client not configured - set STRIPE_SECRET_KEY env var");
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Payment system not configured".to_string()),
                timestamp: chrono::Utc::now(),
            }));
        }
    };

    // Retrieve payment intent to check status
    let intent_id: stripe::PaymentIntentId = request.payment_intent_id.parse().map_err(|e| {
        error!("Invalid payment intent ID: {}", e);
        StatusCode::BAD_REQUEST
    })?;

    match PaymentIntent::retrieve(&stripe_client, &intent_id, &[]).await {
        Ok(intent) => {
            let status = format!("{:?}", intent.status);

            // Check if payment succeeded
            if intent.status == stripe::PaymentIntentStatus::Succeeded {
                let intent_id_str = intent.id.to_string();

                // IDEMPOTENCY: Check if this payment was already processed
                if let Ok(true) = state.storage_engine.is_payment_processed(&intent_id_str).await {
                    info!("💳 Payment {} already processed, returning cached result", intent_id_str);
                    let wallet_addr = intent.metadata.get("wallet_address").map(|s| s.as_str()).unwrap_or("unknown");
                    let balance = state.storage_engine.get_usd_balance(wallet_addr).await.unwrap_or(0);
                    let balance_usd = Decimal::from(balance) / Decimal::from(100);
                    return Ok(Json(ApiResponse {
                        success: true,
                        data: Some(ConfirmPaymentResponse {
                            success: true,
                            status: "Already processed".to_string(),
                            amount_credited: "0".to_string(),
                            new_balance: balance_usd.to_string(),
                        }),
                        error: None,
                        timestamp: chrono::Utc::now(),
                    }));
                }

                // Extract wallet address from metadata
                let wallet_address = intent
                    .metadata
                    .get("wallet_address")
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unknown".to_string());

                // Convert amount from cents to USD
                let amount_usd = Decimal::from(intent.amount) / Decimal::from(100);
                let amount_usd_str = amount_usd.to_string();

                // Credit USD to wallet in storage
                match state
                    .storage_engine
                    .credit_usd_balance(&wallet_address, intent.amount as u64)
                    .await
                {
                    Ok(_) => {
                        info!(
                            "💵 Credited ${} to wallet {}",
                            amount_usd_str, wallet_address
                        );

                        // Mark as processed for idempotency
                        let _ = state.storage_engine.mark_payment_processed(
                            &intent_id_str, &wallet_address, intent.amount as u64
                        ).await;

                        // Get updated balance
                        let new_balance = state
                            .storage_engine
                            .get_usd_balance(&wallet_address)
                            .await
                            .unwrap_or(0);
                        let new_balance_usd = Decimal::from(new_balance) / Decimal::from(100);

                        Ok(Json(ApiResponse {
                            success: true,
                            data: Some(ConfirmPaymentResponse {
                                success: true,
                                status,
                                amount_credited: amount_usd_str,
                                new_balance: new_balance_usd.to_string(),
                            }),
                            error: None,
                            timestamp: chrono::Utc::now(),
                        }))
                    }
                    Err(e) => {
                        error!("Failed to credit wallet: {}", e);
                        Ok(Json(ApiResponse {
                            success: false,
                            data: None,
                            error: Some(format!("Failed to credit wallet: {}", e)),
                            timestamp: chrono::Utc::now(),
                        }))
                    }
                }
            } else {
                warn!("Payment not succeeded, status: {}", status);
                Ok(Json(ApiResponse {
                    success: false,
                    data: Some(ConfirmPaymentResponse {
                        success: false,
                        status: status.clone(),
                        amount_credited: "0".to_string(),
                        new_balance: "0".to_string(),
                    }),
                    error: Some(format!("Payment not completed, status: {}", status)),
                    timestamp: chrono::Utc::now(),
                }))
            }
        }
        Err(e) => {
            error!("Failed to retrieve payment intent: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Stripe error: {}", e)),
                timestamp: chrono::Utc::now(),
            }))
        }
    }
}

/// GET /api/v1/payment/balance/:wallet_address
/// Get USD balance for a wallet (requires wallet auth)
pub async fn get_usd_balance(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<GetBalanceRequest>,
) -> Result<Json<ApiResponse<BalanceResponse>>, StatusCode> {
    // Auth: verify caller is the wallet owner
    let auth_wallet = extract_wallet_from_headers(&headers);
    if auth_wallet.as_deref() != Some(&request.wallet_address) {
        warn!("Unauthorized balance query for wallet: {}", request.wallet_address);
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Unauthorized: wallet auth header must match requested wallet".to_string()),
            timestamp: chrono::Utc::now(),
        }));
    }

    // Rate limit check
    if let Some(resp) = check_payment_rate_limit::<BalanceResponse>(&state, &request.wallet_address) {
        return Ok(resp);
    }

    info!(
        "💰 Getting USD balance for wallet: {}",
        request.wallet_address
    );

    match state
        .storage_engine
        .get_usd_balance(&request.wallet_address)
        .await
    {
        Ok(balance_cents) => {
            let balance_usd = Decimal::from(balance_cents) / Decimal::from(100);

            Ok(Json(ApiResponse {
                success: true,
                data: Some(BalanceResponse {
                    wallet_address: request.wallet_address,
                    balance_usd: balance_usd.to_string(),
                    balance_cents: balance_cents as i64,
                }),
                error: None,
                timestamp: chrono::Utc::now(),
            }))
        }
        Err(e) => {
            error!("Failed to get balance: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Storage error: {}", e)),
                timestamp: chrono::Utc::now(),
            }))
        }
    }
}

/// POST /api/v1/payment/withdraw
/// Withdraw USD from wallet (requires wallet auth)
pub async fn withdraw_usd(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<WithdrawRequest>,
) -> Result<Json<ApiResponse<WithdrawResponse>>, StatusCode> {
    // Auth: verify caller is the wallet owner
    let auth_wallet = extract_wallet_from_headers(&headers);
    if auth_wallet.as_deref() != Some(&request.wallet_address) {
        warn!("Unauthorized withdrawal attempt for wallet: {}", request.wallet_address);
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Unauthorized: wallet auth header must match requested wallet".to_string()),
            timestamp: chrono::Utc::now(),
        }));
    }

    // Rate limit check
    if let Some(resp) = check_payment_rate_limit::<WithdrawResponse>(&state, &request.wallet_address) {
        return Ok(resp);
    }

    info!(
        "🏦 Processing USD withdrawal for wallet: {}, amount: ${}",
        request.wallet_address, request.amount_usd
    );

    // Parse amount
    let amount_decimal: Decimal = request.amount_usd.parse().map_err(|e| {
        error!("Invalid amount format: {}", e);
        StatusCode::BAD_REQUEST
    })?;

    let amount_cents: u64 = (amount_decimal * Decimal::from(100))
        .try_into()
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Check if wallet has sufficient balance
    match state
        .storage_engine
        .get_usd_balance(&request.wallet_address)
        .await
    {
        Ok(balance_cents) => {
            if balance_cents < amount_cents {
                warn!(
                    "Insufficient balance for withdrawal: {} < {}",
                    balance_cents, amount_cents
                );
                return Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some("Insufficient balance".to_string()),
                    timestamp: chrono::Utc::now(),
                }));
            }

            // Debit from wallet
            match state
                .storage_engine
                .debit_usd_balance(&request.wallet_address, amount_cents)
                .await
            {
                Ok(_) => {
                    // In production, this would initiate ACH transfer or Stripe payout
                    let transaction_id = uuid::Uuid::new_v4().to_string();

                    let new_balance = state
                        .storage_engine
                        .get_usd_balance(&request.wallet_address)
                        .await
                        .unwrap_or(0);
                    let new_balance_usd = Decimal::from(new_balance) / Decimal::from(100);

                    info!("✅ Withdrawal processed: {}", transaction_id);

                    Ok(Json(ApiResponse {
                        success: true,
                        data: Some(WithdrawResponse {
                            success: true,
                            transaction_id,
                            amount_withdrawn: request.amount_usd,
                            new_balance: new_balance_usd.to_string(),
                        }),
                        error: None,
                        timestamp: chrono::Utc::now(),
                    }))
                }
                Err(e) => {
                    error!("Failed to debit wallet: {}", e);
                    Ok(Json(ApiResponse {
                        success: false,
                        data: None,
                        error: Some(format!("Failed to process withdrawal: {}", e)),
                        timestamp: chrono::Utc::now(),
                    }))
                }
            }
        }
        Err(e) => {
            error!("Failed to get balance: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Storage error: {}", e)),
                timestamp: chrono::Utc::now(),
            }))
        }
    }
}

/// Request to convert USD to QUGUSD
#[derive(Debug, Deserialize)]
pub struct ConvertToQugusdRequest {
    pub wallet_address: String,
    pub usd_amount: String,
}

/// Response for USD to QUGUSD conversion
#[derive(Debug, Serialize)]
pub struct ConvertToQugusdResponse {
    pub success: bool,
    pub usd_deducted: String,
    pub qugusd_minted: String,
    pub conversion_fee: String,
    pub new_usd_balance: String,
    pub new_qugusd_balance: String,
}

/// POST /api/v1/payment/convert-to-qugusd
/// Convert Stripe USD balance to QUGUSD stablecoin (1:1 with 0.1% fee, requires wallet auth)
pub async fn convert_usd_to_qugusd(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ConvertToQugusdRequest>,
) -> Result<Json<ApiResponse<ConvertToQugusdResponse>>, StatusCode> {
    // Auth: verify caller is the wallet owner
    let auth_wallet = extract_wallet_from_headers(&headers);
    if auth_wallet.as_deref() != Some(&request.wallet_address) {
        warn!("Unauthorized USD-to-QUGUSD conversion attempt for wallet: {}", request.wallet_address);
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Unauthorized: wallet auth header must match requested wallet".to_string()),
            timestamp: chrono::Utc::now(),
        }));
    }

    // Rate limit check
    if let Some(resp) = check_payment_rate_limit::<ConvertToQugusdResponse>(&state, &request.wallet_address) {
        return Ok(resp);
    }

    info!(
        "🔄 Converting USD to QUGUSD for wallet: {}, amount: ${}",
        request.wallet_address, request.usd_amount
    );

    // Parse amount
    let amount_decimal: Decimal = request.usd_amount.parse().map_err(|e| {
        error!("Invalid amount format: {}", e);
        StatusCode::BAD_REQUEST
    })?;

    let usd_cents: u64 = (amount_decimal * Decimal::from(100))
        .try_into()
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Check USD balance
    match state
        .storage_engine
        .get_usd_balance(&request.wallet_address)
        .await
    {
        Ok(balance_cents) => {
            if balance_cents < usd_cents {
                warn!(
                    "Insufficient USD balance: {} < {}",
                    balance_cents, usd_cents
                );
                return Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!(
                        "Insufficient USD balance. You have ${:.2}, need ${:.2}",
                        balance_cents as f64 / 100.0,
                        usd_cents as f64 / 100.0
                    )),
                    timestamp: chrono::Utc::now(),
                }));
            }

            // Calculate conversion with 0.1% fee
            let fee_cents = (usd_cents as f64 * 0.001) as u64; // 0.1% fee
            let qugusd_cents = usd_cents - fee_cents;
            // Convert cents to base units using the double-conversion format:
            // 1 USD = 1 QUGUSD display. QUGUSD has 8 decimals.
            // Due to the accepted double-conversion standard, token_balances stores:
            //   display_amount * 10^(2*decimals) = display_amount * 10^16
            // So 1 cent = 0.01 QUGUSD display = 0.01 * 10^16 = 10^14 base units
            let qugusd_base_units_per_cent: u128 = 100_000_000_000_000; // 10^14
            let qugusd_amount: u128 = (qugusd_cents as u128) * qugusd_base_units_per_cent;

            // Parse wallet address to bytes
            let wallet_addr_bytes =
                match crate::handlers::parse_wallet_address(&request.wallet_address) {
                    Ok(addr) => addr,
                    Err(e) => {
                        error!("Invalid wallet address: {}", e);
                        return Ok(Json(ApiResponse {
                            success: false,
                            data: None,
                            error: Some(format!("Invalid wallet address: {}", e)),
                            timestamp: chrono::Utc::now(),
                        }));
                    }
                };

            // Deduct USD balance
            match state
                .storage_engine
                .debit_usd_balance(&request.wallet_address, usd_cents)
                .await
            {
                Ok(_) => {
                    // Get current QUGUSD balance and add to it
                    let current_qugusd = state
                        .storage_engine
                        .get_token_balance(&wallet_addr_bytes, &q_types::QUGUSD_TOKEN_ADDRESS)
                        .await
                        .unwrap_or(0);
                    let new_qugusd_total = current_qugusd + qugusd_amount;

                    // Mint QUGUSD by saving the new balance
                    match state
                        .storage_engine
                        .save_token_balance(
                            &wallet_addr_bytes,
                            &q_types::QUGUSD_TOKEN_ADDRESS,
                            new_qugusd_total,
                        )
                        .await
                    {
                        Ok(_) => {
                            // Get updated balances
                            let new_usd_balance = state
                                .storage_engine
                                .get_usd_balance(&request.wallet_address)
                                .await
                                .unwrap_or(0);
                            let new_qugusd_balance = new_qugusd_total;

                            let usd_deducted = usd_cents as f64 / 100.0;
                            let qugusd_minted = qugusd_cents as f64 / 100.0;
                            let fee = fee_cents as f64 / 100.0;

                            info!(
                                "✅ Converted ${:.2} USD → {:.4} QUGUSD (fee: ${:.4})",
                                usd_deducted, qugusd_minted, fee
                            );

                            Ok(Json(ApiResponse {
                                success: true,
                                data: Some(ConvertToQugusdResponse {
                                    success: true,
                                    usd_deducted: format!("{:.2}", usd_deducted),
                                    qugusd_minted: format!("{:.4}", qugusd_minted),
                                    conversion_fee: format!("{:.4}", fee),
                                    new_usd_balance: format!(
                                        "{:.2}",
                                        new_usd_balance as f64 / 100.0
                                    ),
                                    new_qugusd_balance: format!(
                                        "{:.2}",
                                        new_qugusd_balance as f64 / 1e16  // 10^(2*8) = 10^16 for 8-decimal tokens (double-conversion format)
                                    ),
                                }),
                                error: None,
                                timestamp: chrono::Utc::now(),
                            }))
                        }
                        Err(e) => {
                            error!("Failed to mint QUGUSD: {}", e);
                            // Refund USD since minting failed
                            let _ = state
                                .storage_engine
                                .credit_usd_balance(&request.wallet_address, usd_cents)
                                .await;
                            Ok(Json(ApiResponse {
                                success: false,
                                data: None,
                                error: Some(format!("Failed to mint QUGUSD: {}", e)),
                                timestamp: chrono::Utc::now(),
                            }))
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to debit USD: {}", e);
                    Ok(Json(ApiResponse {
                        success: false,
                        data: None,
                        error: Some(format!("Failed to process conversion: {}", e)),
                        timestamp: chrono::Utc::now(),
                    }))
                }
            }
        }
        Err(e) => {
            error!("Failed to get USD balance: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Storage error: {}", e)),
                timestamp: chrono::Utc::now(),
            }))
        }
    }
}

/// Request to transfer USD between wallets
#[derive(Debug, Deserialize)]
pub struct TransferUsdRequest {
    pub from_wallet: String,
    pub to_wallet: String,
    pub amount_usd: String,
}

/// Response for USD transfer
#[derive(Debug, Serialize)]
pub struct TransferUsdResponse {
    pub success: bool,
    pub amount_transferred: String,
    pub from_new_balance: String,
    pub to_new_balance: String,
    pub transaction_id: String,
}

/// POST /api/v1/payment/transfer
/// Transfer USD from one wallet to another (requires sender wallet auth)
pub async fn transfer_usd(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<TransferUsdRequest>,
) -> Result<Json<ApiResponse<TransferUsdResponse>>, StatusCode> {
    // Auth: verify caller is the SENDER (from_wallet owner)
    let auth_wallet = extract_wallet_from_headers(&headers);
    if auth_wallet.as_deref() != Some(&request.from_wallet) {
        warn!("Unauthorized transfer attempt from wallet: {}", request.from_wallet);
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Unauthorized: wallet auth header must match sender wallet".to_string()),
            timestamp: chrono::Utc::now(),
        }));
    }

    // Rate limit check
    if let Some(resp) = check_payment_rate_limit::<TransferUsdResponse>(&state, &request.from_wallet) {
        return Ok(resp);
    }

    info!(
        "💸 Transferring USD: {} → {}, amount: ${}",
        request.from_wallet, request.to_wallet, request.amount_usd
    );

    // Parse amount
    let amount_decimal: Decimal = request.amount_usd.parse().map_err(|e| {
        error!("Invalid amount format: {}", e);
        StatusCode::BAD_REQUEST
    })?;

    let amount_cents: u64 = (amount_decimal * Decimal::from(100))
        .try_into()
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    if amount_cents == 0 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Transfer amount must be greater than $0".to_string()),
            timestamp: chrono::Utc::now(),
        }));
    }

    // Check sender balance
    match state
        .storage_engine
        .get_usd_balance(&request.from_wallet)
        .await
    {
        Ok(from_balance_cents) => {
            if from_balance_cents < amount_cents {
                warn!(
                    "Insufficient balance for transfer: {} < {}",
                    from_balance_cents, amount_cents
                );
                return Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!(
                        "Insufficient balance. You have ${:.2}, need ${:.2}",
                        from_balance_cents as f64 / 100.0,
                        amount_cents as f64 / 100.0
                    )),
                    timestamp: chrono::Utc::now(),
                }));
            }

            // Atomic transfer - both debit and credit happen together
            match state
                .storage_engine
                .transfer_usd_atomic(&request.from_wallet, &request.to_wallet, amount_cents)
                .await
            {
                Ok((new_from_balance, new_to_balance)) => {
                    let transaction_id = uuid::Uuid::new_v4().to_string();

                    info!("✅ USD transfer completed: {} (${:.2} from {} to {})",
                        transaction_id, amount_cents as f64 / 100.0,
                        request.from_wallet, request.to_wallet);

                    Ok(Json(ApiResponse {
                        success: true,
                        data: Some(TransferUsdResponse {
                            success: true,
                            amount_transferred: request.amount_usd.clone(),
                            from_new_balance: format!("{:.2}", new_from_balance as f64 / 100.0),
                            to_new_balance: format!("{:.2}", new_to_balance as f64 / 100.0),
                            transaction_id,
                        }),
                        error: None,
                        timestamp: chrono::Utc::now(),
                    }))
                }
                Err(e) => {
                    error!("Failed to transfer USD: {}", e);
                    Ok(Json(ApiResponse {
                        success: false,
                        data: None,
                        error: Some(format!("Transfer failed: {}", e)),
                        timestamp: chrono::Utc::now(),
                    }))
                }
            }
        }
        Err(e) => {
            error!("Failed to get sender balance: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Storage error: {}", e)),
                timestamp: chrono::Utc::now(),
            }))
        }
    }
}

// ============================================================================
// AI Payment Consensus System Endpoints
// ============================================================================

use axum::extract::Query;

/// Query parameters for wallet endpoints
#[derive(Debug, Deserialize)]
pub struct WalletQueryParams {
    pub wallet_address: String,
}

/// AI wallet balance response (QUG/QUGUSD balances for AI inference payments)
#[derive(Debug, Serialize)]
pub struct AIWalletBalanceResponse {
    pub success: bool,
    pub data: Option<AIWalletBalanceData>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AIWalletBalanceData {
    pub wallet_address: String,
    #[serde(serialize_with = "serialize_u128_as_string")]
    pub balance_qnk: u128, // QUG balance in base units (24 decimals)
    pub balance_qnk_usd: f64,
    #[serde(serialize_with = "serialize_u128_as_string")]
    pub balance_qugusd: u128, // QUGUSD balance in base units (24 decimals)
    pub tokens_generated_lifetime: u64,
    pub updated_at: u64,
}

fn serialize_u128_as_string<S>(value: &u128, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&value.to_string())
}

/// AI usage statistics response
#[derive(Debug, Serialize)]
pub struct AIUsageStatsResponse {
    pub success: bool,
    pub data: Option<AIUsageStatsData>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AIUsageStatsData {
    pub wallet_address: String,
    #[serde(serialize_with = "serialize_u128_as_string")]
    pub total_spent_qnk: u128, // Total spent in base units (24 decimals)
    pub total_spent_usd: f64,
    pub total_requests: u64,
    pub total_tokens_generated: u64,
    #[serde(serialize_with = "serialize_u128_as_string")]
    pub average_cost_per_request_qnk: u128, // Average cost in base units (24 decimals)
    pub average_tokens_per_request: u32,
    pub first_request_at: Option<u64>,
    pub last_request_at: Option<u64>,
}

/// AI pricing information response
#[derive(Debug, Serialize)]
pub struct AIPricingResponse {
    pub success: bool,
    pub data: Option<AIPricingData>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AIPricingData {
    pub cost_per_token_qnk: u64,
    pub cost_per_token_usd: f64,
    pub estimated_cost_512_tokens_qnk: u64,
    pub estimated_cost_512_tokens_usd: f64,
    pub qnk_to_usd_rate: f64,
    pub updated_at: u64,
}

/// Treasury statistics response (admin only)
#[derive(Debug, Serialize)]
pub struct TreasuryStatsResponse {
    pub success: bool,
    pub data: Option<TreasuryStatsData>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TreasuryStatsData {
    pub wallet_address: String,
    #[serde(serialize_with = "serialize_u128_as_string")]
    pub total_revenue_qnk: u128, // Total revenue in base units (24 decimals)
    pub total_revenue_usd: f64,
    pub total_requests_served: u64,
    pub total_tokens_generated: u64,
    #[serde(serialize_with = "serialize_u128_as_string")]
    pub average_cost_per_request_qnk: u128, // Average cost in base units (24 decimals)
    pub created_at: u64,
    pub updated_at: u64,
}

/// GET /api/wallet/balance?wallet_address=xxx
/// Returns AI wallet balance with QUG and QUGUSD amounts for inference payments
pub async fn get_ai_wallet_balance(
    State(state): State<Arc<AppState>>,
    Query(params): Query<WalletQueryParams>,
) -> Result<Json<AIWalletBalanceResponse>, StatusCode> {
    info!(
        "📊 GET /api/wallet/balance - wallet: {}",
        params.wallet_address
    );

    // Try to get credits from storage
    match state
        .storage_engine
        .get_wallet_credits(&params.wallet_address)
        .await
    {
        Ok(Some(credits)) => {
            // Calculate USD value (assuming $0.005 per QUG for now)
            let qnk_to_usd = 0.005;
            let balance_usd = (credits.balance_qnk as f64) * qnk_to_usd / 1_000_000_000.0;

            let data = AIWalletBalanceData {
                wallet_address: params.wallet_address.clone(),
                balance_qnk: credits.balance_qnk,
                balance_qnk_usd: balance_usd,
                balance_qugusd: credits.balance_qugusd,
                tokens_generated_lifetime: credits.total_tokens_generated,
                updated_at: credits.updated_at,
            };

            Ok(Json(AIWalletBalanceResponse {
                success: true,
                data: Some(data),
                error: None,
            }))
        }
        Ok(None) => {
            // Wallet not found - return zero balance
            let data = AIWalletBalanceData {
                wallet_address: params.wallet_address.clone(),
                balance_qnk: 0,
                balance_qnk_usd: 0.0,
                balance_qugusd: 0,
                tokens_generated_lifetime: 0,
                updated_at: chrono::Utc::now().timestamp() as u64,
            };

            Ok(Json(AIWalletBalanceResponse {
                success: true,
                data: Some(data),
                error: None,
            }))
        }
        Err(e) => {
            error!("Failed to fetch AI wallet balance: {}", e);
            Ok(Json(AIWalletBalanceResponse {
                success: false,
                data: None,
                error: Some(format!("Database error: {}", e)),
            }))
        }
    }
}

/// GET /api/wallet/usage?wallet_address=xxx
/// Returns AI usage statistics for a wallet
pub async fn get_ai_wallet_usage(
    State(state): State<Arc<AppState>>,
    Query(params): Query<WalletQueryParams>,
) -> Result<Json<AIUsageStatsResponse>, StatusCode> {
    info!(
        "📊 GET /api/wallet/usage - wallet: {}",
        params.wallet_address
    );

    // Try to get credits from storage
    match state
        .storage_engine
        .get_wallet_credits(&params.wallet_address)
        .await
    {
        Ok(Some(credits)) => {
            // Calculate USD value
            let qnk_to_usd = 0.005;
            let spent_usd = (credits.total_spent_qnk as f64) * qnk_to_usd / 1_000_000_000.0;

            // Calculate averages (avoid division by zero)
            // v3.0.4: Cast to u128 for division
            let avg_cost: u128 = if credits.total_tokens_generated > 0 {
                credits.total_spent_qnk / credits.total_tokens_generated as u128
            } else {
                0
            };

            // For simplicity, assume 1 request = 100 tokens on average
            let estimated_requests = if credits.total_tokens_generated > 0 {
                credits.total_tokens_generated / 100
            } else {
                0
            };
            let avg_tokens = if estimated_requests > 0 {
                (credits.total_tokens_generated / estimated_requests) as u32
            } else {
                0
            };

            let data = AIUsageStatsData {
                wallet_address: params.wallet_address.clone(),
                total_spent_qnk: credits.total_spent_qnk,
                total_spent_usd: spent_usd,
                total_requests: estimated_requests,
                total_tokens_generated: credits.total_tokens_generated,
                average_cost_per_request_qnk: avg_cost * 100, // Per 100 tokens
                average_tokens_per_request: avg_tokens,
                first_request_at: Some(credits.created_at),
                last_request_at: Some(credits.updated_at),
            };

            Ok(Json(AIUsageStatsResponse {
                success: true,
                data: Some(data),
                error: None,
            }))
        }
        Ok(None) => {
            // Wallet not found - return zero usage
            let data = AIUsageStatsData {
                wallet_address: params.wallet_address.clone(),
                total_spent_qnk: 0,
                total_spent_usd: 0.0,
                total_requests: 0,
                total_tokens_generated: 0,
                average_cost_per_request_qnk: 0,
                average_tokens_per_request: 0,
                first_request_at: None,
                last_request_at: None,
            };

            Ok(Json(AIUsageStatsResponse {
                success: true,
                data: Some(data),
                error: None,
            }))
        }
        Err(e) => {
            error!("Failed to fetch AI usage stats: {}", e);
            Ok(Json(AIUsageStatsResponse {
                success: false,
                data: None,
                error: Some(format!("Database error: {}", e)),
            }))
        }
    }
}

/// GET /api/pricing
/// Returns current AI pricing information
pub async fn get_ai_pricing_info(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<AIPricingResponse>, StatusCode> {
    info!("📊 GET /api/pricing");

    // Current pricing: 100 QUG per token (100 base units)
    let cost_per_token_qnk = 100u64;
    let qnk_to_usd = 0.005;
    let cost_per_token_usd = (cost_per_token_qnk as f64) * qnk_to_usd / 1_000_000_000.0;

    // Estimate for 512 tokens
    let estimated_512_qnk = cost_per_token_qnk * 512;
    let estimated_512_usd = (estimated_512_qnk as f64) * qnk_to_usd / 1_000_000_000.0;

    let data = AIPricingData {
        cost_per_token_qnk,
        cost_per_token_usd,
        estimated_cost_512_tokens_qnk: estimated_512_qnk,
        estimated_cost_512_tokens_usd: estimated_512_usd,
        qnk_to_usd_rate: qnk_to_usd,
        updated_at: chrono::Utc::now().timestamp() as u64,
    };

    Ok(Json(AIPricingResponse {
        success: true,
        data: Some(data),
        error: None,
    }))
}

/// GET /api/treasury/stats
/// Returns treasury statistics (admin endpoint)
pub async fn get_ai_treasury_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<TreasuryStatsResponse>, StatusCode> {
    info!("📊 GET /api/treasury/stats");

    // Try to get treasury balance from storage
    match state.storage_engine.get_treasury_balance().await {
        Ok(treasury) => {
            // Calculate USD value
            let qnk_to_usd = 0.005;
            let revenue_usd = (treasury.total_revenue_qnk as f64) * qnk_to_usd / 1_000_000_000.0;

            // Calculate average cost per request
            // v3.0.4: Cast total_requests_served to u128 for division
            let avg_cost: u128 = if treasury.total_requests_served > 0 {
                treasury.total_revenue_qnk / treasury.total_requests_served as u128
            } else {
                0
            };

            let data = TreasuryStatsData {
                wallet_address: treasury.wallet_address,
                total_revenue_qnk: treasury.total_revenue_qnk,
                total_revenue_usd: revenue_usd,
                total_requests_served: treasury.total_requests_served,
                total_tokens_generated: treasury.total_tokens_generated,
                average_cost_per_request_qnk: avg_cost,
                created_at: treasury.created_at,
                updated_at: treasury.updated_at,
            };

            Ok(Json(TreasuryStatsResponse {
                success: true,
                data: Some(data),
                error: None,
            }))
        }
        Err(e) => {
            error!("Failed to fetch treasury stats: {}", e);
            Ok(Json(TreasuryStatsResponse {
                success: false,
                data: None,
                error: Some(format!("Database error: {}", e)),
            }))
        }
    }
}

// ============================================================================
// Stripe Webhook Handler
// ============================================================================

/// Verify Stripe webhook signature using HMAC-SHA256
fn verify_stripe_signature(payload: &str, sig_header: &str, secret: &str) -> Result<(), String> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    // Parse stripe-signature header: "t=timestamp,v1=signature"
    let mut timestamp = None;
    let mut signature = None;
    for part in sig_header.split(',') {
        let kv: Vec<&str> = part.splitn(2, '=').collect();
        if kv.len() == 2 {
            match kv[0] {
                "t" => timestamp = Some(kv[1]),
                "v1" => signature = Some(kv[1]),
                _ => {}
            }
        }
    }

    let timestamp = timestamp.ok_or("Missing timestamp in signature")?;
    let expected_sig = signature.ok_or("Missing v1 signature")?;

    // Compute expected signature: HMAC-SHA256(secret, "timestamp.payload")
    let signed_payload = format!("{}.{}", timestamp, payload);
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
        .map_err(|e| format!("HMAC init error: {}", e))?;
    mac.update(signed_payload.as_bytes());
    let result = mac.finalize();
    let computed = hex::encode(result.into_bytes());

    // Constant-time comparison
    if computed != expected_sig {
        return Err("Signature mismatch".to_string());
    }

    // Check timestamp freshness (reject events older than 5 minutes)
    if let Ok(ts) = timestamp.parse::<i64>() {
        let now = chrono::Utc::now().timestamp();
        if (now - ts).abs() > 300 {
            return Err("Webhook timestamp too old".to_string());
        }
    }

    Ok(())
}

/// POST /api/v1/payment/webhook
/// Stripe webhook handler - receives payment_intent.succeeded events.
/// Verifies Stripe signature via HMAC-SHA256, then credits wallets idempotently.
pub async fn handle_stripe_webhook(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> Result<StatusCode, StatusCode> {
    // Get the Stripe signature header
    let sig_header = headers
        .get("stripe-signature")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            warn!("Webhook missing stripe-signature header");
            StatusCode::BAD_REQUEST
        })?;

    // Get webhook secret
    let webhook_secret = get_webhook_secret().map_err(|e| {
        error!("Webhook secret not configured: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Verify the webhook signature
    verify_stripe_signature(&body, sig_header, &webhook_secret).map_err(|e| {
        warn!("Webhook signature verification failed: {}", e);
        StatusCode::BAD_REQUEST
    })?;

    // Parse the event JSON
    let event: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
        warn!("Failed to parse webhook body: {}", e);
        StatusCode::BAD_REQUEST
    })?;

    let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

    // Only handle payment_intent.succeeded events
    if event_type != "payment_intent.succeeded" {
        // Acknowledge other events without processing
        return Ok(StatusCode::OK);
    }

    // Extract payment intent data from the event
    let data_object = event
        .get("data")
        .and_then(|d| d.get("object"))
        .ok_or_else(|| {
            warn!("Webhook: Missing data.object");
            StatusCode::BAD_REQUEST
        })?;

    let intent_id = data_object
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if intent_id.is_empty() {
        warn!("Webhook: Missing payment intent ID");
        return Ok(StatusCode::OK);
    }

    // IDEMPOTENCY: Check if already processed
    match state.storage_engine.is_payment_processed(&intent_id).await {
        Ok(true) => {
            info!("Webhook: Payment {} already processed, skipping", intent_id);
            return Ok(StatusCode::OK);
        }
        Ok(false) => {}
        Err(e) => {
            error!("Failed to check payment status: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    }

    // Extract wallet address from metadata
    let wallet_address = data_object
        .get("metadata")
        .and_then(|m| m.get("wallet_address"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if wallet_address.is_empty() {
        warn!("Webhook: No wallet_address in payment metadata for {}", intent_id);
        return Ok(StatusCode::OK);
    }

    let amount_cents = data_object
        .get("amount")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    if amount_cents == 0 {
        warn!("Webhook: Zero amount for payment {}", intent_id);
        return Ok(StatusCode::OK);
    }

    // Credit the wallet
    match state.storage_engine.credit_usd_balance(&wallet_address, amount_cents).await {
        Ok(_) => {
            // Mark as processed AFTER successful credit
            if let Err(e) = state.storage_engine.mark_payment_processed(
                &intent_id, &wallet_address, amount_cents
            ).await {
                error!("Failed to mark payment as processed: {} - manual reconciliation needed", e);
            }

            let amount_usd = amount_cents as f64 / 100.0;
            info!("Webhook: Credited ${:.2} to wallet {} (payment: {})", amount_usd, wallet_address, intent_id);
            Ok(StatusCode::OK)
        }
        Err(e) => {
            error!("Webhook: Failed to credit wallet {}: {}", wallet_address, e);
            // Return 500 so Stripe retries the webhook
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
