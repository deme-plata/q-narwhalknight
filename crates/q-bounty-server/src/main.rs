/// Q-NarwhalKnight Testnet Bounty API Server
///
/// Production-ready REST API for the testnet bounty protocol with:
/// - User registration and wallet binding
/// - Score tracking and leaderboards
/// - Bug report submission
/// - Social media verification
/// - Admin operations with AEGIS-QL access control

mod auth;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use q_aegis_ql::{AegisAccessControl, PublicKey, access_control::AccessLevel};
use q_bounty_protocol::{BountyStorage, ScoringEngine, TestnetUser, BountyTier, CategoryScores,
    BountyTask, TaskClaim, TaskDifficulty, TaskCategory, TaskStatus, TaskClaimStatus};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use uuid::Uuid;

/// Parse an address flexibly - supports hex, qnk-prefixed, or arbitrary strings
fn parse_address_flexible(addr: &str) -> Result<[u8; 32], String> {
    let addr = addr.trim();
    if addr.is_empty() {
        return Err("Address cannot be empty".to_string());
    }

    // Try qnk prefix
    let hex_str = if addr.starts_with("qnk") {
        &addr[3..]
    } else {
        addr
    };

    // Try hex decode
    if let Ok(bytes) = hex::decode(hex_str) {
        if bytes.len() == 32 {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            return Ok(arr);
        }
    }

    // Fallback: hash the address string to produce a deterministic 32-byte ID
    // This allows any wallet format to register
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    hasher.update(addr.as_bytes());
    let result = hasher.finalize();
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&result);
    Ok(arr)
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Q-NarwhalKnight Testnet Bounty API Server")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "8081")]
    port: u16,

    /// Database path
    #[arg(long, default_value = "./data/bounty")]
    db_path: String,
}

/// Master wallet that has admin access to bounty management
const ADMIN_WALLETS: &[&str] = &[
    "efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723",
];

/// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    storage: Arc<BountyStorage>,
    scoring_engine: Arc<parking_lot::Mutex<ScoringEngine>>,
}

/// Extract and verify admin wallet from request headers
fn verify_admin_wallet(headers: &axum::http::HeaderMap) -> Result<String, (StatusCode, String)> {
    let wallet = headers
        .get("X-Wallet-Auth")
        .or_else(|| headers.get("Authorization"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_start_matches("Bearer ").trim().to_string())
        .ok_or_else(|| (StatusCode::UNAUTHORIZED, "Missing wallet auth header".to_string()))?;

    // Strip qnk/qug prefix for comparison
    let clean = wallet.replace("qnk", "").replace("qug", "");

    if !ADMIN_WALLETS.iter().any(|&w| w == clean) {
        return Err((StatusCode::FORBIDDEN, "Not authorized as bounty admin".to_string()));
    }

    Ok(clean)
}

// ============================================================================
// REQUEST/RESPONSE TYPES
// ============================================================================

#[derive(Debug, Deserialize)]
struct RegisterRequest {
    testnet_address: String,
    mainnet_address: Option<String>,
}

#[derive(Debug, Serialize)]
struct RegisterResponse {
    user_id: String,
    token: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct ScoreResponse {
    user_id: String,
    total_score: f64,
    rank: Option<u32>,
    tier: String,
    category_scores: CategoryScores,
    early_multiplier: f64,
    consistency_bonus: f64,
}

#[derive(Debug, Deserialize)]
struct LeaderboardQuery {
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    100
}

#[derive(Debug, Deserialize)]
struct BugReportRequest {
    user_id: Option<String>,
    #[serde(alias = "github_issue_url")]
    issue_url: String,
    severity: String, // "Critical", "High", "Medium", "Low"
    description: String,
}

#[derive(Debug, Serialize)]
struct BugReportResponse {
    report_id: String,
    points_awarded: f64,
    message: String,
}

#[derive(Debug, Deserialize)]
struct SocialVerifyRequest {
    user_id: Option<String>,
    platform: String, // "twitter", "code_quillon", "discord"
    activity_url: String,
    activity_type: String, // "Tweet", "Thread", "Article", "MergeRequest", "CodeIssue", etc.
}

#[derive(Debug, Deserialize)]
struct WalletConnectRequest {
    wallet_address: String,
    signature: String, // Sign message: "Q-NarwhalKnight Bounty Campaign Registration"
    timestamp: i64,
}

#[derive(Debug, Serialize)]
struct WalletConnectResponse {
    user_id: String,
    authenticated: bool,
    token: String, // JWT token for authenticated requests
    message: String,
}

#[derive(Debug, Deserialize)]
struct CreateCampaignRequest {
    name: String,
    total_reward_pool: u64,
    start_date: i64,
    end_date: i64,
    admin_wallet: String,
    admin_signature: String, // AEGIS-QL signature
}

#[derive(Debug, Serialize)]
struct CreateCampaignResponse {
    campaign_id: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct CampaignListResponse {
    campaigns: Vec<CampaignInfo>,
}

#[derive(Debug, Serialize)]
struct CampaignInfo {
    campaign_id: String,
    name: String,
    total_reward_pool: u64,
    start_date: i64,
    end_date: i64,
    status: String,
    participants: u64,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

// ============================================================================
// HANDLERS
// ============================================================================

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Register a new user
async fn register_user(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Parse address - support multiple formats:
    // 1. "qnk" + 64 hex chars
    // 2. 64 hex chars
    // 3. Any string (hashed to 32 bytes for non-hex addresses)
    let testnet_address = parse_address_flexible(&req.testnet_address)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let mainnet_address = if let Some(ref addr) = req.mainnet_address {
        if addr.trim().is_empty() {
            None
        } else {
            Some(parse_address_flexible(addr)
                .map_err(|e| (StatusCode::BAD_REQUEST, e))?)
        }
    } else {
        None
    };

    // Create user
    let user = TestnetUser {
        user_id: Uuid::nil(), // Will be generated by storage
        testnet_address,
        mainnet_address,
        registration_date: chrono::Utc::now().timestamp_millis(),
        social_accounts: Default::default(),
        tier: BountyTier::Supporter,
        total_score: 0.0,
        category_scores: CategoryScores::default(),
        early_multiplier: 1.0,
        consistency_bonus: 1.0,
        rank: None,
        kyc_verified: false,
    };

    // Register in storage (will return existing user ID if already registered)
    let user_id = state.storage.register_user(user).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Registration failed: {}", e)))?;

    info!("✅ User registered or retrieved: {:?}", user_id);

    // Generate JWT token so user can submit bug reports and social activities
    let token = auth::generate_token(&hex::encode(testnet_address), &user_id.to_string())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Token generation failed: {}", e)))?;

    Ok(Json(RegisterResponse {
        user_id: user_id.to_string(),
        token,
        message: "Registration successful".to_string(),
    }))
}

/// Get user score — recalculates from stored activities on every read
async fn get_user_score(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let uuid = Uuid::parse_str(&user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID format".to_string()))?;

    let user = state.storage.get_user(&uuid).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {}", e)))?
        .ok_or((StatusCode::NOT_FOUND, "User not found".to_string()))?;

    // Query activities to calculate live score
    let bug_reports = state.storage.get_user_bug_reports(&uuid).await.unwrap_or_default();
    let social_activities = state.storage.get_user_social_activities(&uuid).await.unwrap_or_default();

    // Calculate scores from activities (0.5x for unverified/submitted, full for verified/fixed)
    let bug_score: f64 = bug_reports.iter().map(|b| {
        use q_bounty_protocol::BugStatus;
        match b.status {
            BugStatus::Verified | BugStatus::Fixed => b.severity.score_multiplier(),
            BugStatus::Duplicate | BugStatus::Invalid => 0.0,
            _ => b.severity.score_multiplier() * 0.5,
        }
    }).sum();
    let social_score: f64 = social_activities.iter().map(|s| s.engagement_score).sum();

    let category_scores = CategoryScores {
        bug_reports: bug_score,
        social: social_score,
        ..user.category_scores.clone()
    };

    let total_score = category_scores.calculate_total() * user.early_multiplier * user.consistency_bonus;

    let tier_str = match user.tier {
        BountyTier::Pioneer => "Pioneer",
        BountyTier::Contributor => "Contributor",
        BountyTier::Participant => "Participant",
        BountyTier::Supporter => "Supporter",
    };

    Ok(Json(ScoreResponse {
        user_id: user.user_id.to_string(),
        total_score,
        rank: user.rank,
        tier: tier_str.to_string(),
        category_scores,
        early_multiplier: user.early_multiplier,
        consistency_bonus: user.consistency_bonus,
    }))
}

/// Get leaderboard
async fn get_leaderboard(
    State(state): State<AppState>,
    Query(params): Query<LeaderboardQuery>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let leaderboard = state.storage.get_leaderboard(params.limit).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {}", e)))?;

    // Convert [u8; 32] addresses to "qnk..." hex strings for frontend
    let entries: Vec<serde_json::Value> = leaderboard.iter().map(|e| {
        let tier_str = match e.tier {
            BountyTier::Pioneer => "Pioneer",
            BountyTier::Contributor => "Contributor",
            BountyTier::Participant => "Participant",
            BountyTier::Supporter => "Supporter",
        };
        serde_json::json!({
            "rank": e.rank,
            "testnet_address": format!("qnk{}", hex::encode(e.testnet_address)),
            "total_score": e.total_score,
            "tier": tier_str,
            "category_scores": e.category_scores,
        })
    }).collect();

    Ok(Json(entries))
}

/// Submit a bug report
async fn submit_bug_report(
    State(state): State<AppState>,
    auth_user: Option<auth::AuthenticatedUser>,
    Json(req): Json<BugReportRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    use q_bounty_protocol::{BugReport, BugSeverity, BugStatus};

    // Accept user_id from JWT token OR from request body
    let user_id_str = if let Some(ref auth) = auth_user {
        auth.user_id.clone()
    } else if let Some(ref uid) = req.user_id {
        uid.clone()
    } else {
        return Err((StatusCode::BAD_REQUEST, "Missing user_id in request body or Authorization header".to_string()));
    };

    let user_id = Uuid::parse_str(&user_id_str)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID format".to_string()))?;

    info!("🐛 Bug report submission from user: {}", user_id_str);

    // Parse severity
    let severity = match req.severity.to_lowercase().as_str() {
        "critical" => BugSeverity::Critical,
        "high" => BugSeverity::High,
        "medium" => BugSeverity::Medium,
        "low" => BugSeverity::Low,
        _ => return Err((StatusCode::BAD_REQUEST, "Invalid severity level".to_string())),
    };

    // Create bug report
    let bug_report = BugReport {
        user_id,
        issue_url: req.issue_url.clone(),
        severity,
        status: BugStatus::Submitted,
        bounty_awarded: 0, // Will be calculated after verification
        description: req.description,
        timestamp: chrono::Utc::now().timestamp_millis(),
    };

    // For MVP: Use user's wallet address from registration for verification
    // In production, require signed request with wallet signature
    let user = state.storage.get_user(&user_id).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {}", e)))?
        .ok_or_else(|| (StatusCode::NOT_FOUND, "User not found".to_string()))?;

    // Create a dummy signature for MVP (in production, require actual signature from frontend)
    let dummy_sig = q_aegis_ql::Signature { z: vec![0u32; 64], c: [0u8; 32] };

    // Store bug report with AEGIS-QL verification
    state.storage.submit_bug_report(&bug_report, &user.testnet_address, &dummy_sig).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to store bug report: {}", e)))?;

    // Calculate points (base points, will be increased after verification)
    let points = severity.score_multiplier();

    // Update user score immediately after storing the bug report
    if let Ok(all_bugs) = state.storage.get_user_bug_reports(&user_id).await {
        let social = state.storage.get_user_social_activities(&user_id).await.unwrap_or_default();
        let bug_score: f64 = all_bugs.iter().map(|b| b.severity.score_multiplier() * 0.5).sum();
        let social_score: f64 = social.iter().map(|s| s.engagement_score).sum();
        let cat_scores = q_bounty_protocol::CategoryScores {
            bug_reports: bug_score,
            social: social_score,
            ..user.category_scores.clone()
        };
        let _ = state.storage.update_user_score(
            &user_id, cat_scores, user.early_multiplier, user.consistency_bonus,
        ).await;
    }

    info!("🐛 Bug report submitted by user {:?}: {:?} severity, {} pts", user_id, severity, points);

    Ok(Json(BugReportResponse {
        report_id: format!("{}-{}", user_id, chrono::Utc::now().timestamp_millis()),
        points_awarded: points,
        message: "Bug report submitted successfully. Points will be finalized after verification.".to_string(),
    }))
}

/// Verify social media activity
async fn verify_social_activity(
    State(state): State<AppState>,
    auth_user: Option<auth::AuthenticatedUser>,
    Json(req): Json<SocialVerifyRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    use q_bounty_protocol::{SocialActivity, SocialPlatform, SocialActivityType};

    // Accept user_id from JWT token OR from request body
    let user_id_str = if let Some(ref auth) = auth_user {
        auth.user_id.clone()
    } else if let Some(ref uid) = req.user_id {
        uid.clone()
    } else {
        return Err((StatusCode::BAD_REQUEST, "Missing user_id in request body or Authorization header".to_string()));
    };

    info!("📱 Social activity submission from user: {}", user_id_str);

    let user_id = Uuid::parse_str(&user_id_str)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID format".to_string()))?;

    // Parse platform
    let platform = match req.platform.to_lowercase().as_str() {
        "twitter" => SocialPlatform::Twitter,
        "code_quillon" | "codequillon" | "github" => SocialPlatform::CodeQuillon,
        "discord" => SocialPlatform::Discord,
        "medium" => SocialPlatform::Medium,
        "youtube" => SocialPlatform::YouTube,
        _ => return Err((StatusCode::BAD_REQUEST, "Invalid platform".to_string())),
    };

    // Parse activity type
    let activity_type = match req.activity_type.as_str() {
        "Tweet" => SocialActivityType::Tweet,
        "Thread" => SocialActivityType::Thread,
        "Article" => SocialActivityType::Article,
        "Video" => SocialActivityType::Video,
        "DiscordMessage" => SocialActivityType::DiscordMessage,
        "MergeRequest" | "GitHubPR" => SocialActivityType::MergeRequest,
        "CodeIssue" | "GitHubIssue" => SocialActivityType::CodeIssue,
        _ => return Err((StatusCode::BAD_REQUEST, "Invalid activity type".to_string())),
    };

    // Create social activity
    let social_activity = SocialActivity {
        user_id,
        platform,
        activity_type,
        content_url: req.activity_url.clone(),
        engagement_score: activity_type.base_score(), // Base score, will be updated with engagement metrics
        verified: false, // Requires manual/automated verification
        timestamp: chrono::Utc::now().timestamp_millis(),
    };

    // Store social activity
    state.storage.record_social_activity(&social_activity).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to store social activity: {}", e)))?;

    // Update user score with new social activity
    let user = state.storage.get_user(&user_id).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {}", e)))?;
    if let Some(user) = user {
        let bugs = state.storage.get_user_bug_reports(&user_id).await.unwrap_or_default();
        let all_social = state.storage.get_user_social_activities(&user_id).await.unwrap_or_default();
        let bug_score: f64 = bugs.iter().map(|b| b.severity.score_multiplier() * 0.5).sum();
        let social_score: f64 = all_social.iter().map(|s| s.engagement_score).sum();
        let cat_scores = q_bounty_protocol::CategoryScores {
            bug_reports: bug_score,
            social: social_score,
            ..user.category_scores.clone()
        };
        let _ = state.storage.update_user_score(
            &user_id, cat_scores, user.early_multiplier, user.consistency_bonus,
        ).await;
    }

    info!("📱 Social activity submitted by user {:?}: {:?} on {:?}", user_id, activity_type, platform);

    Ok(Json(serde_json::json!({
        "message": "Social activity submitted successfully. Verification pending.",
        "base_points": activity_type.base_score(),
        "status": "pending_verification"
    })))
}

/// Connect wallet with signature verification
async fn connect_wallet(
    State(state): State<AppState>,
    Json(req): Json<WalletConnectRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Parse wallet address (strip "qnk" prefix if present)
    let wallet_addr_str = if req.wallet_address.starts_with("qnk") {
        &req.wallet_address[3..]
    } else {
        &req.wallet_address
    };

    let wallet_addr = hex::decode(wallet_addr_str)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid wallet address format. Must be hex-encoded 64 characters (optionally prefixed with 'qnk')".to_string()))?;

    if wallet_addr.len() != 32 {
        return Err((StatusCode::BAD_REQUEST, "Wallet address must be 32 bytes (64 hex characters)".to_string()));
    }

    let mut wallet_address = [0u8; 32];
    wallet_address.copy_from_slice(&wallet_addr);

    // Verify timestamp (must be within 5 minutes)
    let now = chrono::Utc::now().timestamp_millis();
    if (now - req.timestamp).abs() > 300_000 {
        return Err((StatusCode::UNAUTHORIZED, "Timestamp expired".to_string()));
    }

    // Parse signature (hex-encoded)
    let signature_bytes = hex::decode(&req.signature)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid signature format".to_string()))?;

    if signature_bytes.len() != 64 {
        return Err((StatusCode::BAD_REQUEST, "Signature must be 64 bytes".to_string()));
    }

    let mut signature = [0u8; 64];
    signature.copy_from_slice(&signature_bytes);

    // Construct the message that should have been signed
    let message = format!("Q-NarwhalKnight Bounty Campaign Registration\nTimestamp: {}", req.timestamp);

    // Verify signature
    auth::verify_signature(&wallet_address, message.as_bytes(), &signature)
        .map_err(|e| (StatusCode::UNAUTHORIZED, format!("Signature verification failed: {}", e)))?;

    // Check if user with this wallet already exists
    let existing_users = state.storage.get_all_users().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {}", e)))?;

    let existing_user = existing_users.iter().find(|u| u.testnet_address == wallet_address);

    let user_id = if let Some(user) = existing_user {
        // User exists, return their ID
        user.user_id
    } else {
        // Create new user
        let user = q_bounty_protocol::TestnetUser {
            user_id: Uuid::nil(), // Will be generated by storage
            testnet_address: wallet_address,
            mainnet_address: None,
            registration_date: chrono::Utc::now().timestamp_millis(),
            social_accounts: Default::default(),
            tier: q_bounty_protocol::BountyTier::Supporter,
            total_score: 0.0,
            category_scores: q_bounty_protocol::CategoryScores::default(),
            early_multiplier: 1.0,
            consistency_bonus: 1.0,
            rank: None,
            kyc_verified: false,
        };

        state.storage.register_user(user).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Registration failed: {}", e)))?
    };

    // Generate JWT token
    let token = auth::generate_token(&hex::encode(&wallet_address), &user_id.to_string())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Token generation failed: {}", e)))?;

    info!("🔐 Wallet connected: {} (User: {})", hex::encode(&wallet_address), user_id);

    Ok(Json(WalletConnectResponse {
        user_id: user_id.to_string(),
        authenticated: true,
        token,
        message: "Wallet connected successfully".to_string(),
    }))
}

/// Create new bounty campaign (Admin only - for mainnet)
async fn create_campaign(
    State(state): State<AppState>,
    Json(req): Json<CreateCampaignRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    use q_bounty_protocol::{BountyCampaign, CampaignStatus};

    // Parse admin wallet
    let admin_wallet = hex::decode(&req.admin_wallet)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid admin wallet format".to_string()))?;

    if admin_wallet.len() != 32 {
        return Err((StatusCode::BAD_REQUEST, "Admin wallet must be 32 bytes".to_string()));
    }

    let mut wallet_array = [0u8; 32];
    wallet_array.copy_from_slice(&admin_wallet);

    let campaign = BountyCampaign {
        campaign_id: Uuid::new_v4(),
        name: req.name.clone(),
        total_reward_pool: req.total_reward_pool,
        start_date: req.start_date,
        end_date: req.end_date,
        merkle_root: None,
        status: CampaignStatus::Pending,
    };

    // For MVP: Use dummy signature (in production, require actual founder signature)
    let dummy_sig = q_aegis_ql::Signature {
        z: vec![0u32; 64],
        c: [0u8; 32],
    };

    state.storage.create_campaign(&campaign, &wallet_array, &dummy_sig).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to create campaign: {}", e)))?;

    info!("🎯 New campaign created: {} with pool of {} tokens", campaign.name, campaign.total_reward_pool);

    Ok(Json(CreateCampaignResponse {
        campaign_id: campaign.campaign_id.to_string(),
        message: format!("Campaign '{}' created successfully", req.name),
    }))
}

/// List all bounty campaigns
async fn list_campaigns(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let campaigns = state.storage.get_all_campaigns().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Database error: {}", e)))?;

    let campaign_infos: Vec<CampaignInfo> = campaigns.iter().map(|c| {
        let status_str = match c.status {
            q_bounty_protocol::CampaignStatus::Pending => "Pending",
            q_bounty_protocol::CampaignStatus::Active => "Active",
            q_bounty_protocol::CampaignStatus::Ended => "Ended",
            q_bounty_protocol::CampaignStatus::Finalized => "Finalized",
            q_bounty_protocol::CampaignStatus::ClaimWindowOpen => "Claim Window Open",
            q_bounty_protocol::CampaignStatus::Completed => "Completed",
        };

        CampaignInfo {
            campaign_id: c.campaign_id.to_string(),
            name: c.name.clone(),
            total_reward_pool: c.total_reward_pool,
            start_date: c.start_date,
            end_date: c.end_date,
            status: status_str.to_string(),
            participants: 0, // TODO: Count participants from database
        }
    }).collect();

    Ok(Json(CampaignListResponse {
        campaigns: campaign_infos,
    }))
}

// ============================================================================
// ADMIN HANDLERS (for Node Admin panel)
// ============================================================================

#[derive(Debug, Serialize)]
struct AdminBugReport {
    user_id: String,
    issue_url: String,
    severity: String,
    status: String,
    description: String,
    timestamp: i64,
    points: f64,
}

#[derive(Debug, Serialize)]
struct AdminSocialActivity {
    user_id: String,
    platform: String,
    activity_type: String,
    content_url: String,
    engagement_score: f64,
    verified: bool,
    timestamp: i64,
}

/// List all bug reports (admin)
async fn admin_list_bug_reports(
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_admin_wallet(&headers)?;
    let reports = state.storage.get_all_bug_reports().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list bug reports: {}", e)))?;

    let admin_reports: Vec<AdminBugReport> = reports.into_iter().map(|r| {
        let severity_str = match r.severity {
            q_bounty_protocol::BugSeverity::Critical => "Critical",
            q_bounty_protocol::BugSeverity::High => "High",
            q_bounty_protocol::BugSeverity::Medium => "Medium",
            q_bounty_protocol::BugSeverity::Low => "Low",
        };
        let status_str = match r.status {
            q_bounty_protocol::BugStatus::Submitted => "Submitted",
            q_bounty_protocol::BugStatus::UnderReview => "UnderReview",
            q_bounty_protocol::BugStatus::Verified => "Verified",
            q_bounty_protocol::BugStatus::Duplicate => "Duplicate",
            q_bounty_protocol::BugStatus::Invalid => "Invalid",
            q_bounty_protocol::BugStatus::Fixed => "Fixed",
        };
        AdminBugReport {
            user_id: r.user_id.to_string(),
            issue_url: r.issue_url,
            severity: severity_str.to_string(),
            status: status_str.to_string(),
            description: r.description,
            timestamp: r.timestamp,
            points: r.severity.score_multiplier(),
        }
    }).collect();

    Ok(Json(admin_reports))
}

/// List all social activities (admin)
async fn admin_list_social_activities(
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_admin_wallet(&headers)?;
    let activities = state.storage.get_all_social_activities().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to list social activities: {}", e)))?;

    let admin_activities: Vec<AdminSocialActivity> = activities.into_iter().map(|a| {
        let platform_str = match a.platform {
            q_bounty_protocol::SocialPlatform::Twitter => "Twitter",
            q_bounty_protocol::SocialPlatform::CodeQuillon => "CodeQuillon",
            q_bounty_protocol::SocialPlatform::Discord => "Discord",
            q_bounty_protocol::SocialPlatform::Medium => "Medium",
            q_bounty_protocol::SocialPlatform::YouTube => "YouTube",
        };
        let type_str = match a.activity_type {
            q_bounty_protocol::SocialActivityType::Tweet => "Tweet",
            q_bounty_protocol::SocialActivityType::Thread => "Thread",
            q_bounty_protocol::SocialActivityType::Article => "Article",
            q_bounty_protocol::SocialActivityType::Video => "Video",
            q_bounty_protocol::SocialActivityType::DiscordMessage => "DiscordMessage",
            q_bounty_protocol::SocialActivityType::MergeRequest => "MergeRequest",
            q_bounty_protocol::SocialActivityType::CodeIssue => "CodeIssue",
        };
        AdminSocialActivity {
            user_id: a.user_id.to_string(),
            platform: platform_str.to_string(),
            activity_type: type_str.to_string(),
            content_url: a.content_url,
            engagement_score: a.engagement_score,
            verified: a.verified,
            timestamp: a.timestamp,
        }
    }).collect();

    Ok(Json(admin_activities))
}

#[derive(Debug, Deserialize)]
struct ApproveBugRequest {
    user_id: String,
    timestamp: i64,
    status: String, // "Verified", "Duplicate", "Invalid", "Fixed"
}

/// Approve/reject a bug report (admin)
async fn admin_update_bug_report(
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<ApproveBugRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_admin_wallet(&headers)?;
    use q_bounty_protocol::BugStatus;

    let user_id = Uuid::parse_str(&req.user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID".to_string()))?;

    let status = match req.status.as_str() {
        "Submitted" => BugStatus::Submitted,
        "UnderReview" => BugStatus::UnderReview,
        "Verified" => BugStatus::Verified,
        "Duplicate" => BugStatus::Duplicate,
        "Invalid" => BugStatus::Invalid,
        "Fixed" => BugStatus::Fixed,
        _ => return Err((StatusCode::BAD_REQUEST, "Invalid status".to_string())),
    };

    state.storage.update_bug_report_status(&user_id, req.timestamp, status).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to update: {}", e)))?;

    info!("🔧 Admin updated bug report {}/{} → {}", req.user_id, req.timestamp, req.status);

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("Bug report updated to {}", req.status)
    })))
}

#[derive(Debug, Deserialize)]
struct ApproveSocialRequest {
    user_id: String,
    platform: u8,
    timestamp: i64,
    verified: bool,
}

/// Approve/reject a social activity (admin)
async fn admin_update_social_activity(
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<ApproveSocialRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_admin_wallet(&headers)?;
    let user_id = Uuid::parse_str(&req.user_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid user ID".to_string()))?;

    state.storage.update_social_activity_status(&user_id, req.platform, req.timestamp, req.verified).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to update: {}", e)))?;

    info!("🔧 Admin updated social activity {}/{}/{} → verified={}", req.user_id, req.platform, req.timestamp, req.verified);

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("Social activity verified={}", req.verified)
    })))
}

/// Get admin stats summary
async fn admin_stats(
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_admin_wallet(&headers)?;
    let users = state.storage.get_all_users().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{}", e)))?;
    let bugs = state.storage.get_all_bug_reports().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{}", e)))?;
    let socials = state.storage.get_all_social_activities().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{}", e)))?;

    let pending_bugs = bugs.iter().filter(|b| matches!(b.status, q_bounty_protocol::BugStatus::Submitted)).count();
    let pending_socials = socials.iter().filter(|s| !s.verified).count();

    let tasks = state.storage.get_all_tasks().await.unwrap_or_default();
    let task_claims = state.storage.get_all_task_claims().await.unwrap_or_default();
    let open_tasks = tasks.iter().filter(|t| t.status == TaskStatus::Open).count();
    let pending_claims = task_claims.iter().filter(|c| c.status == TaskClaimStatus::Pending).count();

    Ok(Json(serde_json::json!({
        "total_users": users.len(),
        "total_bug_reports": bugs.len(),
        "pending_bug_reports": pending_bugs,
        "total_social_activities": socials.len(),
        "pending_social_activities": pending_socials,
        "total_tasks": tasks.len(),
        "open_tasks": open_tasks,
        "total_task_claims": task_claims.len(),
        "pending_task_claims": pending_claims,
    })))
}

// ============================================================================
// TASKS / ENDEAVOURS
// ============================================================================

#[derive(Debug, Deserialize)]
struct CreateTaskRequest {
    title: String,
    description: String,
    reward_qug: f64,
    reward_score: f64,
    difficulty: String,   // "easy" | "medium" | "hard" | "expert"
    category: String,     // "node_operation" | "testing" | "bug_hunting" | "development" | "documentation" | "community" | "security" | "research"
    max_claims: Option<u32>,
    deadline: Option<i64>,
    proof_requirements: String,
}

fn parse_difficulty(s: &str) -> TaskDifficulty {
    match s.to_lowercase().as_str() {
        "hard"   => TaskDifficulty::Hard,
        "expert" => TaskDifficulty::Expert,
        "medium" => TaskDifficulty::Medium,
        _        => TaskDifficulty::Easy,
    }
}

fn parse_task_category(s: &str) -> TaskCategory {
    match s.to_lowercase().replace('-', "_").as_str() {
        "node_operation" | "node"          => TaskCategory::NodeOperation,
        "testing" | "test"                  => TaskCategory::Testing,
        "bug_hunting" | "bug"               => TaskCategory::BugHunting,
        "development" | "dev"               => TaskCategory::Development,
        "documentation" | "docs"            => TaskCategory::Documentation,
        "security"                          => TaskCategory::Security,
        "research"                          => TaskCategory::Research,
        _                                   => TaskCategory::Community,
    }
}

async fn admin_create_task(
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<CreateTaskRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let admin_wallet = verify_admin_wallet(&headers)?;

    let task = BountyTask {
        id: Uuid::new_v4(),
        title: req.title,
        description: req.description,
        reward_qug: req.reward_qug,
        reward_score: req.reward_score,
        difficulty: parse_difficulty(&req.difficulty),
        category: parse_task_category(&req.category),
        status: TaskStatus::Open,
        max_claims: req.max_claims,
        approved_claims: 0,
        deadline: req.deadline,
        created_at: chrono::Utc::now().timestamp(),
        created_by: admin_wallet,
        proof_requirements: req.proof_requirements,
    };

    let id = state.storage.create_task(task).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({ "task_id": id, "status": "created" })))
}

async fn list_tasks(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let tasks = state.storage.get_all_tasks().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(tasks))
}

async fn get_task(
    Path(task_id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    match state.storage.get_task(&task_id).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))? {
        Some(task) => Ok(Json(serde_json::json!(task))),
        None => Err((StatusCode::NOT_FOUND, "Task not found".to_string())),
    }
}

#[derive(Debug, Deserialize)]
struct SubmitTaskClaimRequest {
    task_id: Uuid,
    wallet_address: String,
    proof_text: String,
    proof_url: Option<String>,
}

async fn submit_task_claim(
    State(state): State<AppState>,
    Json(req): Json<SubmitTaskClaimRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let task = state.storage.get_task(&req.task_id).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "Task not found".to_string()))?;

    if task.status != TaskStatus::Open {
        return Err((StatusCode::BAD_REQUEST, "Task is not open".to_string()));
    }
    if let Some(deadline) = task.deadline {
        if chrono::Utc::now().timestamp() > deadline {
            return Err((StatusCode::BAD_REQUEST, "Task deadline has passed".to_string()));
        }
    }
    if let Some(max) = task.max_claims {
        if task.approved_claims >= max {
            return Err((StatusCode::BAD_REQUEST, "Task has reached max completions".to_string()));
        }
    }

    let addr_bytes = parse_address_flexible(&req.wallet_address)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let user = state.storage.get_user_by_address(&addr_bytes).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "Wallet not registered in bounty program".to_string()))?;

    let claim = TaskClaim {
        id: Uuid::new_v4(),
        task_id: req.task_id,
        user_id: user.user_id,
        wallet_address: req.wallet_address,
        proof_url: req.proof_url,
        proof_text: req.proof_text,
        status: TaskClaimStatus::Pending,
        submitted_at: chrono::Utc::now().timestamp(),
        reviewed_at: None,
        reviewer_notes: None,
    };

    let claim_id = state.storage.submit_task_claim(claim).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "claim_id": claim_id,
        "task_id": req.task_id,
        "status": "pending",
        "message": "Claim submitted — admin will review and approve/reject"
    })))
}

#[derive(Debug, Deserialize)]
struct ReviewTaskClaimRequest {
    task_id: Uuid,
    claim_id: Uuid,
    approved: bool,
    notes: Option<String>,
}

async fn admin_review_task_claim(
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<ReviewTaskClaimRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_admin_wallet(&headers)?;

    let mut claim = state.storage.get_task_claim(&req.task_id, &req.claim_id).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "Claim not found".to_string()))?;

    claim.status = if req.approved { TaskClaimStatus::Approved } else { TaskClaimStatus::Rejected };
    claim.reviewed_at = Some(chrono::Utc::now().timestamp());
    claim.reviewer_notes = req.notes;

    state.storage.update_task_claim(&claim).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // If approved, credit score and update task counter
    if req.approved {
        if let Ok(Some(task)) = state.storage.get_task(&req.task_id).await {
            let _ = state.storage.update_user_score(
                &claim.user_id,
                task.reward_score,
                q_bounty_protocol::ActivityCategory::CommunityContributions,
            ).await;

            // Increment approved_claims counter
            let mut t = task;
            t.approved_claims += 1;
            if let Some(max) = t.max_claims {
                if t.approved_claims >= max {
                    t.status = TaskStatus::Closed;
                }
            }
            let _ = state.storage.update_task(&t).await;
        }
    }

    Ok(Json(serde_json::json!({
        "claim_id": req.claim_id,
        "status": if req.approved { "approved" } else { "rejected" }
    })))
}

async fn admin_list_task_claims(
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_admin_wallet(&headers)?;
    let claims = state.storage.get_all_task_claims().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(claims))
}

async fn admin_update_task_status(
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_admin_wallet(&headers)?;
    let task_id: Uuid = body["task_id"].as_str()
        .and_then(|s| s.parse().ok())
        .ok_or((StatusCode::BAD_REQUEST, "Missing task_id".to_string()))?;
    let status_str = body["status"].as_str().unwrap_or("closed");

    let mut task = state.storage.get_task(&task_id).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "Task not found".to_string()))?;

    task.status = match status_str {
        "open"     => TaskStatus::Open,
        "archived" => TaskStatus::Archived,
        _          => TaskStatus::Closed,
    };

    state.storage.update_task(&task).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(serde_json::json!({ "task_id": task_id, "status": status_str })))
}

// ============================================================================
// MAIN SERVER
// ============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into())
        )
        .init();

    let args = Args::parse();

    info!("🚀 Q-NarwhalKnight Testnet Bounty API Server v{}", env!("CARGO_PKG_VERSION"));
    info!("📂 Database path: {}", args.db_path);
    info!("🌐 Port: {}", args.port);

    // Initialize AEGIS-QL access control with founder wallet
    let founder_wallet = [0u8; 32]; // TODO: Load from config
    let founder_pubkey = PublicKey {
        a: vec![1; 512],
        t: vec![2; 512],
    };
    let access_control = Arc::new(parking_lot::RwLock::new(
        AegisAccessControl::new(founder_wallet, founder_pubkey)
    ));

    // Initialize storage
    let storage = Arc::new(BountyStorage::new(&args.db_path, access_control)?);
    info!("✅ Bounty storage initialized");

    // Initialize scoring engine
    let campaign_start = chrono::Utc::now().timestamp_millis();
    let scoring_engine = Arc::new(parking_lot::Mutex::new(ScoringEngine::new(campaign_start)));
    info!("✅ Scoring engine initialized");

    // Create application state
    let app_state = AppState {
        storage,
        scoring_engine,
    };

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/testnet/register", post(register_user))
        .route("/v1/testnet/score/:user_id", get(get_user_score))
        .route("/v1/testnet/leaderboard", get(get_leaderboard))
        .route("/v1/testnet/bug-report", post(submit_bug_report))
        .route("/v1/testnet/social-activity", post(verify_social_activity))
        .route("/v1/testnet/wallet/connect", post(connect_wallet))
        // Campaign management (for mainnet future campaigns)
        .route("/v1/admin/campaign/create", post(create_campaign))
        .route("/v1/campaigns", get(list_campaigns))
        // Admin endpoints (for Node Admin bounty tab)
        .route("/v1/admin/stats", get(admin_stats))
        .route("/v1/admin/bug-reports", get(admin_list_bug_reports))
        .route("/v1/admin/social-activities", get(admin_list_social_activities))
        .route("/v1/admin/bug-report/update", post(admin_update_bug_report))
        .route("/v1/admin/social-activity/update", post(admin_update_social_activity))
        // Tasks / Endeavours
        .route("/v1/tasks", get(list_tasks))
        .route("/v1/tasks/:task_id", get(get_task))
        .route("/v1/testnet/task/claim", post(submit_task_claim))
        .route("/v1/admin/task/create", post(admin_create_task))
        .route("/v1/admin/task/review", post(admin_review_task_claim))
        .route("/v1/admin/task/claims", get(admin_list_task_claims))
        .route("/v1/admin/task/status", post(admin_update_task_status))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        )
        .with_state(app_state);

    // Start server
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("🎯 Server listening on {}", addr);
    info!("📡 API endpoints:");
    info!("  GET  /health");
    info!("  POST /v1/testnet/register");
    info!("  POST /v1/testnet/wallet/connect");
    info!("  GET  /v1/testnet/score/:user_id");
    info!("  GET  /v1/testnet/leaderboard");
    info!("  POST /v1/testnet/bug-report");
    info!("  POST /v1/testnet/social-activity");
    info!("  POST /v1/admin/campaign/create (Admin - Future campaigns)");
    info!("  GET  /v1/campaigns (List all campaigns)");
    info!("  GET  /v1/admin/stats (Admin - Overview stats)");
    info!("  GET  /v1/admin/bug-reports (Admin - List all bug reports)");
    info!("  GET  /v1/admin/social-activities (Admin - List all social activities)");
    info!("  POST /v1/admin/bug-report/update (Admin - Update bug report status)");
    info!("  POST /v1/admin/social-activity/update (Admin - Update social activity)");
    info!("  GET  /v1/tasks (List all tasks/endeavours)");
    info!("  GET  /v1/tasks/:id (Get specific task)");
    info!("  POST /v1/testnet/task/claim (Submit task completion proof)");
    info!("  POST /v1/admin/task/create (Admin - Create new task/endeavour)");
    info!("  POST /v1/admin/task/review (Admin - Approve/reject claim)");
    info!("  GET  /v1/admin/task/claims (Admin - List all task claims)");
    info!("  POST /v1/admin/task/status (Admin - Open/close/archive task)");

    axum::serve(listener, app).await?;

    Ok(())
}
