/// Decentralized Blockchain Calendar API
/// v7.3.3: Personal events, scheduled transactions, network milestones, community events
///
/// Features:
/// - Per-wallet calendar events stored in RocksDB
/// - Scheduled transactions with background executor
/// - Community events shared via P2P gossipsub
/// - Network milestones (halvings, upgrades) hardcoded
/// - SSE push notifications for reminders

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{delete, get, post, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::streaming::StreamEvent;
use crate::wallet_auth::AuthenticatedWallet;
use crate::AppState;
use q_types::*;

// ============================================================================
// Router
// ============================================================================

pub fn calendar_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/events", post(create_event))
        .route("/events", get(get_events))
        .route("/events/:id", get(get_event))
        .route("/events/:id", put(update_event))
        .route("/events/:id", delete(delete_event))
        .route("/events/:id/share", post(share_event))
        .route("/scheduled-tx", post(create_scheduled_tx))
        .route("/scheduled-tx", get(get_scheduled_txs))
        .route("/scheduled-tx/:id", delete(cancel_scheduled_tx))
        .route("/network-events", get(get_network_events))
        .route("/community-events", get(get_community_events_handler))
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct CreateEventRequest {
    pub title: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub event_type: Option<String>,
    pub start_time: u64,
    #[serde(default)]
    pub end_time: Option<u64>,
    #[serde(default)]
    pub all_day: bool,
    #[serde(default)]
    pub recurring: Option<RecurrenceRule>,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub reminder_minutes: Option<Vec<u32>>,
    #[serde(default)]
    pub shared: bool,
}

#[derive(Debug, Deserialize)]
pub struct CreateScheduledTxRequest {
    pub title: String,
    #[serde(default)]
    pub description: Option<String>,
    pub start_time: u64,
    pub to_wallet: String,
    pub token: String,
    pub amount: String,
    #[serde(default)]
    pub reminder_minutes: Option<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
pub struct EventQueryParams {
    #[serde(default)]
    pub start_date: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(default)]
    pub event_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EventResponse {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub event_type: String,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub all_day: bool,
    pub recurring: Option<RecurrenceRule>,
    pub color: Option<String>,
    pub reminder_minutes: Option<Vec<u32>>,
    pub scheduled_tx: Option<ScheduledTransaction>,
    pub shared: bool,
    pub created_at: u64,
    pub updated_at: Option<u64>,
    pub cancelled: bool,
    pub source_peer: Option<String>,
}

impl From<CalendarEvent> for EventResponse {
    fn from(e: CalendarEvent) -> Self {
        EventResponse {
            id: e.id,
            title: e.title,
            description: e.description,
            event_type: e.event_type.to_string(),
            start_time: e.start_time,
            end_time: e.end_time,
            all_day: e.all_day,
            recurring: e.recurring,
            color: e.color,
            reminder_minutes: e.reminder_minutes,
            scheduled_tx: e.scheduled_tx,
            shared: e.shared,
            created_at: e.created_at,
            updated_at: e.updated_at,
            cancelled: e.cancelled,
            source_peer: e.source_peer,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct NetworkEvent {
    pub id: String,
    pub title: String,
    pub description: String,
    pub event_type: String,
    pub start_time: u64,
    pub color: String,
    pub icon: String,
}

// Uses q_types::ApiResponse (from `use q_types::*`)

fn parse_event_type(s: &str) -> CalendarEventType {
    match s.to_lowercase().as_str() {
        "scheduled_tx" | "scheduledtransaction" => CalendarEventType::ScheduledTransaction,
        "vesting_unlock" | "vestingunlock" => CalendarEventType::VestingUnlock,
        "governance_vote" | "governancevote" => CalendarEventType::GovernanceVote,
        "network_milestone" | "networkmilestone" => CalendarEventType::NetworkMilestone,
        "community_event" | "communityevent" => CalendarEventType::CommunityEvent,
        "price_alert" | "pricealert" => CalendarEventType::PriceAlert,
        _ => CalendarEventType::Personal,
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// POST /api/v1/calendar/events — Create a calendar event
async fn create_event(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateEventRequest>,
) -> Result<Json<ApiResponse<EventResponse>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let event_id = Uuid::new_v4().to_string();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let event_type = req.event_type.as_deref().map(parse_event_type).unwrap_or(CalendarEventType::Personal);

    let event = CalendarEvent {
        id: event_id.clone(),
        wallet: auth.address,
        title: req.title.clone(),
        description: req.description,
        event_type: event_type.clone(),
        start_time: req.start_time,
        end_time: req.end_time,
        all_day: req.all_day,
        recurring: req.recurring,
        color: req.color,
        reminder_minutes: req.reminder_minutes,
        scheduled_tx: None,
        shared: req.shared,
        created_at: now,
        updated_at: None,
        source_peer: None,
        cancelled: false,
    };

    if let Err(e) = state.storage_engine.save_calendar_event(&event).await {
        error!("Failed to save calendar event: {}", e);
        return Ok(Json(ApiResponse::error(format!("Storage error: {}", e))));
    }

    // Emit SSE
    let _ = state.event_broadcaster.broadcast(StreamEvent::CalendarEventCreated {
        event_id: event_id.clone(),
        title: req.title,
        event_type: event_type.to_string(),
        start_time: req.start_time,
        has_scheduled_tx: false,
        timestamp: chrono::Utc::now(),
    });

    if event.shared {
        publish_calendar_event_p2p(&state, &event).await;
    }

    info!("📅 Created calendar event {} for wallet {}", event_id, hex::encode(&auth.address[..4]));
    Ok(Json(ApiResponse::success(EventResponse::from(event))))
}

/// GET /api/v1/calendar/events — Get events in date range
async fn get_events(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<EventQueryParams>,
) -> Result<Json<ApiResponse<Vec<EventResponse>>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let now = chrono::Utc::now();
    let start_date = params.start_date.unwrap_or_else(|| {
        now.format("%Y%m%d").to_string()
    });
    let end_date = params.end_date.unwrap_or_else(|| {
        (now + chrono::Duration::days(90)).format("%Y%m%d").to_string()
    });

    let type_filter = params.event_type.as_deref().map(parse_event_type);

    match state.storage_engine.get_calendar_events_by_date_range(&auth.address, &start_date, &end_date).await {
        Ok(events) => {
            let mut results: Vec<EventResponse> = events.into_iter()
                .filter(|e| {
                    if let Some(ref tf) = type_filter {
                        &e.event_type == tf
                    } else {
                        true
                    }
                })
                .map(EventResponse::from)
                .collect();

            // Also include community events in the range
            if let Ok(community) = state.storage_engine.get_community_events(100).await {
                for ce in community {
                    let ce_date = chrono::DateTime::from_timestamp(ce.start_time as i64, 0)
                        .map(|dt| dt.format("%Y%m%d").to_string())
                        .unwrap_or_default();
                    if ce_date >= start_date && ce_date <= end_date {
                        results.push(EventResponse::from(ce));
                    }
                }
            }

            results.sort_by_key(|e| e.start_time);
            Ok(Json(ApiResponse::success(results)))
        }
        Err(e) => {
            error!("Failed to get calendar events: {}", e);
            Ok(Json(ApiResponse::error(format!("Storage error: {}", e))))
        }
    }
}

/// GET /api/v1/calendar/events/:id — Get single event
async fn get_event(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<EventResponse>>, StatusCode> {
    let _auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    match state.storage_engine.get_calendar_event(&id).await {
        Ok(Some(event)) => Ok(Json(ApiResponse::success(EventResponse::from(event)))),
        Ok(None) => Ok(Json(ApiResponse::error("Event not found".to_string()))),
        Err(e) => Ok(Json(ApiResponse::error(format!("Storage error: {}", e)))),
    }
}

/// PUT /api/v1/calendar/events/:id — Update event
async fn update_event(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<CreateEventRequest>,
) -> Result<Json<ApiResponse<EventResponse>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let existing = match state.storage_engine.get_calendar_event(&id).await {
        Ok(Some(e)) => e,
        Ok(None) => return Ok(Json(ApiResponse::error("Event not found".to_string()))),
        Err(e) => return Ok(Json(ApiResponse::error(format!("Storage error: {}", e)))),
    };

    if existing.wallet != auth.address {
        return Ok(Json(ApiResponse::error("Not authorized to edit this event".to_string())));
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let event_type = req.event_type.as_deref().map(parse_event_type).unwrap_or(existing.event_type);

    let updated = CalendarEvent {
        id: existing.id,
        wallet: existing.wallet,
        title: req.title,
        description: req.description,
        event_type,
        start_time: req.start_time,
        end_time: req.end_time,
        all_day: req.all_day,
        recurring: req.recurring,
        color: req.color,
        reminder_minutes: req.reminder_minutes,
        scheduled_tx: existing.scheduled_tx,
        shared: existing.shared,
        created_at: existing.created_at,
        updated_at: Some(now),
        source_peer: existing.source_peer,
        cancelled: false,
    };

    if let Err(e) = state.storage_engine.update_calendar_event(&updated).await {
        return Ok(Json(ApiResponse::error(format!("Storage error: {}", e))));
    }

    Ok(Json(ApiResponse::success(EventResponse::from(updated))))
}

/// DELETE /api/v1/calendar/events/:id — Soft-delete event
async fn delete_event(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<()>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let existing = match state.storage_engine.get_calendar_event(&id).await {
        Ok(Some(e)) => e,
        Ok(None) => return Ok(Json(ApiResponse::error("Event not found".to_string()))),
        Err(e) => return Ok(Json(ApiResponse::error(format!("Storage error: {}", e)))),
    };

    if existing.wallet != auth.address {
        return Ok(Json(ApiResponse::error("Not authorized to delete this event".to_string())));
    }

    if let Err(e) = state.storage_engine.delete_calendar_event(&id).await {
        return Ok(Json(ApiResponse::error(format!("Storage error: {}", e))));
    }

    Ok(Json(ApiResponse::success(())))
}

/// POST /api/v1/calendar/events/:id/share — Share event to P2P network
async fn share_event(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<()>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let mut event = match state.storage_engine.get_calendar_event(&id).await {
        Ok(Some(e)) => e,
        Ok(None) => return Ok(Json(ApiResponse::error("Event not found".to_string()))),
        Err(e) => return Ok(Json(ApiResponse::error(format!("Storage error: {}", e)))),
    };

    if event.wallet != auth.address {
        return Ok(Json(ApiResponse::error("Not authorized".to_string())));
    }

    // Spam limit: 5 shares per day
    match state.storage_engine.count_shared_events_today(&auth.address).await {
        Ok(count) if count >= 5 => {
            return Ok(Json(ApiResponse::error("Daily share limit reached (5/day)".to_string())));
        }
        Err(e) => {
            warn!("Failed to check share count: {}", e);
        }
        _ => {}
    }

    event.shared = true;
    event.event_type = CalendarEventType::CommunityEvent;
    event.updated_at = Some(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );

    // Save as community event
    if let Err(e) = state.storage_engine.save_community_event(&event).await {
        return Ok(Json(ApiResponse::error(format!("Storage error: {}", e))));
    }

    // Update the original event
    if let Err(e) = state.storage_engine.update_calendar_event(&event).await {
        warn!("Failed to update shared flag: {}", e);
    }

    // Publish via P2P
    publish_calendar_event_p2p(&state, &event).await;

    info!("📅 Shared community event {} via P2P", id);
    Ok(Json(ApiResponse::success(())))
}

/// POST /api/v1/calendar/scheduled-tx — Create scheduled transaction
async fn create_scheduled_tx(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateScheduledTxRequest>,
) -> Result<Json<ApiResponse<EventResponse>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let event_id = Uuid::new_v4().to_string();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if req.start_time <= now {
        return Ok(Json(ApiResponse::error("Scheduled time must be in the future".to_string())));
    }

    let event = CalendarEvent {
        id: event_id.clone(),
        wallet: auth.address,
        title: req.title.clone(),
        description: req.description,
        event_type: CalendarEventType::ScheduledTransaction,
        start_time: req.start_time,
        end_time: None,
        all_day: false,
        recurring: None,
        color: Some("#f59e0b".to_string()), // amber for scheduled tx
        reminder_minutes: req.reminder_minutes,
        scheduled_tx: Some(ScheduledTransaction {
            to_wallet: req.to_wallet.clone(),
            token: req.token.clone(),
            amount: req.amount.clone(),
            executed: false,
            tx_hash: None,
            error: None,
        }),
        shared: false,
        created_at: now,
        updated_at: None,
        source_peer: None,
        cancelled: false,
    };

    if let Err(e) = state.storage_engine.save_calendar_event(&event).await {
        error!("Failed to save scheduled tx: {}", e);
        return Ok(Json(ApiResponse::error(format!("Storage error: {}", e))));
    }

    // Emit SSE
    let _ = state.event_broadcaster.broadcast(StreamEvent::CalendarEventCreated {
        event_id: event_id.clone(),
        title: req.title,
        event_type: "scheduled_tx".to_string(),
        start_time: req.start_time,
        has_scheduled_tx: true,
        timestamp: chrono::Utc::now(),
    });

    info!("📅 Created scheduled TX {} → {} {} {} at {}", event_id, req.to_wallet, req.amount, req.token, req.start_time);
    Ok(Json(ApiResponse::success(EventResponse::from(event))))
}

/// GET /api/v1/calendar/scheduled-tx — List scheduled transactions
async fn get_scheduled_txs(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<EventResponse>>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    // Get all events and filter for scheduled transactions belonging to this wallet
    let start = "19700101";
    let end = "29991231";
    match state.storage_engine.get_calendar_events_by_date_range(&auth.address, start, end).await {
        Ok(events) => {
            let results: Vec<EventResponse> = events.into_iter()
                .filter(|e| e.scheduled_tx.is_some() && !e.cancelled)
                .map(EventResponse::from)
                .collect();
            Ok(Json(ApiResponse::success(results)))
        }
        Err(e) => Ok(Json(ApiResponse::error(format!("Storage error: {}", e)))),
    }
}

/// DELETE /api/v1/calendar/scheduled-tx/:id — Cancel scheduled transaction
async fn cancel_scheduled_tx(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<()>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let event = match state.storage_engine.get_calendar_event(&id).await {
        Ok(Some(e)) => e,
        Ok(None) => return Ok(Json(ApiResponse::error("Event not found".to_string()))),
        Err(e) => return Ok(Json(ApiResponse::error(format!("Storage error: {}", e)))),
    };

    if event.wallet != auth.address {
        return Ok(Json(ApiResponse::error("Not authorized".to_string())));
    }

    if let Some(ref tx) = event.scheduled_tx {
        if tx.executed {
            return Ok(Json(ApiResponse::error("Transaction already executed".to_string())));
        }
    }

    if let Err(e) = state.storage_engine.delete_calendar_event(&id).await {
        return Ok(Json(ApiResponse::error(format!("Storage error: {}", e))));
    }

    info!("📅 Cancelled scheduled TX {}", id);
    Ok(Json(ApiResponse::success(())))
}

/// GET /api/v1/calendar/network-events — Hardcoded network milestones
async fn get_network_events(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<NetworkEvent>>>, StatusCode> {
    let events = vec![
        NetworkEvent {
            id: "net-mainnet-2026-2".to_string(),
            title: "Mainnet 2026.2 Launch".to_string(),
            description: "Q-NarwhalKnight mainnet 2026.2 launch with fresh genesis block and new emission schedule".to_string(),
            event_type: "network_milestone".to_string(),
            start_time: 1771761600, // Feb 22, 2026 12:00 UTC
            color: "#f43f5e".to_string(),
            icon: "rocket".to_string(),
        },
        NetworkEvent {
            id: "net-halving-1".to_string(),
            title: "First Halving".to_string(),
            description: "QUG mining reward halves from 2,625,000 to 1,312,500 QUG/year (Era 1)".to_string(),
            event_type: "network_milestone".to_string(),
            start_time: 1771761600 + (4 * 365 * 24 * 3600), // +4 years
            color: "#f43f5e".to_string(),
            icon: "scissors".to_string(),
        },
        NetworkEvent {
            id: "net-halving-2".to_string(),
            title: "Second Halving".to_string(),
            description: "QUG mining reward halves from 1,312,500 to 656,250 QUG/year (Era 2)".to_string(),
            event_type: "network_milestone".to_string(),
            start_time: 1771761600 + (8 * 365 * 24 * 3600), // +8 years
            color: "#f43f5e".to_string(),
            icon: "scissors".to_string(),
        },
        NetworkEvent {
            id: "net-max-supply".to_string(),
            title: "21M QUG Max Supply".to_string(),
            description: "Estimated date when total QUG supply approaches 21M cap".to_string(),
            event_type: "network_milestone".to_string(),
            start_time: 1771761600 + (50 * 365 * 24 * 3600), // ~50 years
            color: "#f59e0b".to_string(),
            icon: "trophy".to_string(),
        },
        NetworkEvent {
            id: "net-pq-upgrade".to_string(),
            title: "Post-Quantum Signature Upgrade".to_string(),
            description: "Activation of Dilithium5 post-quantum signatures for all transactions".to_string(),
            event_type: "network_milestone".to_string(),
            start_time: 1772366400, // ~Mar 1, 2026
            color: "#8b5cf6".to_string(),
            icon: "shield".to_string(),
        },
    ];

    Ok(Json(ApiResponse::success(events)))
}

/// GET /api/v1/calendar/community-events — Get community events
async fn get_community_events_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<EventResponse>>>, StatusCode> {
    match state.storage_engine.get_community_events(100).await {
        Ok(events) => {
            let results: Vec<EventResponse> = events.into_iter().map(EventResponse::from).collect();
            Ok(Json(ApiResponse::success(results)))
        }
        Err(e) => Ok(Json(ApiResponse::error(format!("Storage error: {}", e)))),
    }
}

// ============================================================================
// P2P Functions
// ============================================================================

/// Publish a calendar event via P2P gossipsub
async fn publish_calendar_event_p2p(state: &Arc<AppState>, event: &CalendarEvent) {
    let network_id = std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "mainnet-genesis".to_string());
    let topic = format!("/qnk/{}/calendar", network_id);

    match serde_json::to_vec(event) {
        Ok(bytes) => {
            if let Some(ref tx) = state.libp2p_command_tx {
                if let Err(e) = tx.send(q_network::NetworkCommand::PublishTokenAnnouncement {
                    topic,
                    announcement_bytes: bytes,
                }) {
                    warn!("Failed to publish calendar event via P2P: {}", e);
                } else {
                    debug!("📅 Published community event {} via P2P", event.id);
                }
            }
        }
        Err(e) => warn!("Failed to serialize calendar event: {}", e),
    }
}

/// Handle incoming P2P calendar event (called from main.rs gossipsub handler)
pub async fn handle_p2p_calendar_event(state: &Arc<AppState>, data: &[u8]) {
    match serde_json::from_slice::<CalendarEvent>(data) {
        Ok(mut event) => {
            // Only accept community events from P2P
            if event.event_type != CalendarEventType::CommunityEvent {
                debug!("📅 Ignoring non-community P2P calendar event {}", event.id);
                return;
            }

            // Check if we already have this event
            if let Ok(Some(_)) = state.storage_engine.get_calendar_event(&event.id).await {
                debug!("📅 Already have calendar event {}, skipping", event.id);
                return;
            }

            // Set source peer
            event.source_peer = Some("p2p".to_string());

            // Save as community event
            if let Err(e) = state.storage_engine.save_community_event(&event).await {
                error!("Failed to save P2P calendar event: {}", e);
                return;
            }

            // Emit SSE
            let _ = state.event_broadcaster.broadcast(StreamEvent::CalendarEventCreated {
                event_id: event.id.clone(),
                title: event.title.clone(),
                event_type: "community_event".to_string(),
                start_time: event.start_time,
                has_scheduled_tx: false,
                timestamp: chrono::Utc::now(),
            });

            info!("📅 Received community event '{}' via P2P", event.title);
        }
        Err(e) => {
            debug!("Failed to deserialize P2P calendar event: {}", e);
        }
    }
}

/// Background task: Execute pending scheduled transactions
/// Called every 60s from main.rs
pub async fn execute_pending_scheduled_txs(state: &Arc<AppState>) {
    let pending = match state.storage_engine.get_pending_scheduled_transactions().await {
        Ok(p) => p,
        Err(e) => {
            debug!("Failed to get pending scheduled txs: {}", e);
            return;
        }
    };

    for event in pending {
        let scheduled_tx = match &event.scheduled_tx {
            Some(tx) => tx.clone(),
            None => continue,
        };

        info!("📅 Executing scheduled TX {} → {} {} {}", event.id, scheduled_tx.to_wallet, scheduled_tx.amount, scheduled_tx.token);

        // Parse recipient wallet
        let to_wallet_bytes = match hex::decode(&scheduled_tx.to_wallet) {
            Ok(bytes) if bytes.len() == 32 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                arr
            }
            _ => {
                let err = "Invalid recipient wallet address";
                let _ = state.storage_engine.mark_scheduled_tx_failed(&event.id, err).await;
                let _ = state.event_broadcaster.broadcast(StreamEvent::ScheduledTransactionExecuted {
                    event_id: event.id.clone(),
                    title: event.title.clone(),
                    to_wallet: scheduled_tx.to_wallet.clone(),
                    token: scheduled_tx.token.clone(),
                    amount: scheduled_tx.amount.clone(),
                    tx_hash: None,
                    success: false,
                    error: Some(err.to_string()),
                    timestamp: chrono::Utc::now(),
                });
                continue;
            }
        };

        // Parse amount
        let amount: f64 = match scheduled_tx.amount.parse() {
            Ok(a) => a,
            Err(_) => {
                let err = "Invalid amount";
                let _ = state.storage_engine.mark_scheduled_tx_failed(&event.id, err).await;
                continue;
            }
        };

        // Execute the transfer based on token type
        let is_qug = scheduled_tx.token.to_uppercase() == "QUG";

        // Deterministic tx hash: sha3-256 of "scheduled_tx:" || event_id || ":" || start_time
        let tx_hash = {
            use sha3::{Digest, Sha3_256};
            let mut h = Sha3_256::new();
            h.update(b"scheduled_tx:");
            h.update(event.id.as_bytes());
            h.update(b":");
            h.update(event.start_time.to_le_bytes());
            hex::encode(h.finalize())
        };

        let tx_result: Result<String, String> = if is_qug {
            // QUG transfer: update in-memory map then persist both sides to RocksDB.
            let amount_base = (amount * 1e24) as u128;
            let check = {
                let mut balances = state.wallet_balances.write().await;
                let sender_balance = balances.get(&event.wallet).copied().unwrap_or(0u128);
                if sender_balance < amount_base {
                    Err("Insufficient QUG balance".to_string())
                } else {
                    let new_sender = sender_balance - amount_base;
                    let new_recipient = balances.get(&to_wallet_bytes).copied().unwrap_or(0u128) + amount_base;
                    *balances.entry(event.wallet).or_insert(0u128) = new_sender;
                    *balances.entry(to_wallet_bytes).or_insert(0u128) = new_recipient;
                    Ok((new_sender, new_recipient))
                }
            };
            match check {
                Err(e) => Err(e),
                Ok((new_sender_bal, new_recipient_bal)) => {
                    // Persist so the balance survives node restarts.
                    if let Err(e) = state.storage_engine.save_wallet_balance(&event.wallet, new_sender_bal).await {
                        warn!("📅 Failed to persist sender balance after scheduled TX {}: {}", event.id, e);
                    }
                    if let Err(e) = state.storage_engine.save_wallet_balance(&to_wallet_bytes, new_recipient_bal).await {
                        warn!("📅 Failed to persist recipient balance after scheduled TX {}: {}", event.id, e);
                    }
                    Ok(tx_hash)
                }
            }
        } else {
            // Token transfer via token_balances: key = ([u8;32], [u8;32]), value = u128
            let token_addr: [u8; 32] = if scheduled_tx.token.to_uppercase() == "QUGUSD" {
                q_types::QUGUSD_TOKEN_ADDRESS
            } else {
                match hex::decode(&scheduled_tx.token) {
                    Ok(b) if b.len() == 32 => {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&b);
                        arr
                    }
                    _ => {
                        let _ = state.storage_engine.mark_scheduled_tx_failed(&event.id, "Invalid token address").await;
                        continue;
                    }
                }
            };

            let amount_base = (amount * 1e24) as u128;
            let mut token_balances = state.token_balances.write().await;
            let sender_key = (event.wallet, token_addr);
            let sender_balance = token_balances.get(&sender_key).copied().unwrap_or(0u128);

            if sender_balance < amount_base {
                Err(format!("Insufficient {} balance", scheduled_tx.token))
            } else {
                *token_balances.entry(sender_key).or_insert(0u128) -= amount_base;
                let recipient_key = (to_wallet_bytes, token_addr);
                *token_balances.entry(recipient_key).or_insert(0u128) += amount_base;
                Ok(tx_hash)
            }
        };

        match tx_result {
            Ok(tx_hash) => {
                let _ = state.storage_engine.mark_scheduled_tx_executed(&event.id, &tx_hash).await;
                let _ = state.event_broadcaster.broadcast(StreamEvent::ScheduledTransactionExecuted {
                    event_id: event.id.clone(),
                    title: event.title.clone(),
                    to_wallet: scheduled_tx.to_wallet,
                    token: scheduled_tx.token,
                    amount: scheduled_tx.amount,
                    tx_hash: Some(tx_hash),
                    success: true,
                    error: None,
                    timestamp: chrono::Utc::now(),
                });
                info!("📅 ✅ Scheduled TX {} executed successfully", event.id);
            }
            Err(err) => {
                let _ = state.storage_engine.mark_scheduled_tx_failed(&event.id, &err).await;
                let _ = state.event_broadcaster.broadcast(StreamEvent::ScheduledTransactionExecuted {
                    event_id: event.id.clone(),
                    title: event.title.clone(),
                    to_wallet: scheduled_tx.to_wallet,
                    token: scheduled_tx.token,
                    amount: scheduled_tx.amount,
                    tx_hash: None,
                    success: false,
                    error: Some(err.clone()),
                    timestamp: chrono::Utc::now(),
                });
                warn!("📅 ❌ Scheduled TX {} failed: {}", event.id, err);
            }
        }
    }
}

/// Background task: Check for upcoming event reminders
/// Called every 60s from main.rs
pub async fn check_calendar_reminders(state: &Arc<AppState>) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Look ahead 60 minutes for reminders
    let now_dt = chrono::Utc::now();
    let start_date = now_dt.format("%Y%m%d").to_string();
    let end_date = (now_dt + chrono::Duration::days(1)).format("%Y%m%d").to_string();

    // Check all wallets' events
    let balances = state.wallet_balances.read().await;
    let wallets: Vec<[u8; 32]> = balances.keys().copied().collect();
    drop(balances);

    for wallet in wallets.iter().take(50) {
        let events = match state.storage_engine.get_calendar_events_by_date_range(wallet, &start_date, &end_date).await {
            Ok(e) => e,
            Err(_) => continue,
        };

        for event in events {
            if event.cancelled { continue; }

            let reminder_minutes = match &event.reminder_minutes {
                Some(mins) if !mins.is_empty() => mins.clone(),
                _ => continue,
            };

            for &minutes in &reminder_minutes {
                let reminder_time = event.start_time.saturating_sub((minutes as u64) * 60);
                // Fire if within 30s of the reminder time
                if now >= reminder_time && now < reminder_time + 60 {
                    let minutes_until = (event.start_time as i64 - now as i64) / 60;
                    let _ = state.event_broadcaster.broadcast(StreamEvent::CalendarReminder {
                        event_id: event.id.clone(),
                        title: event.title.clone(),
                        event_type: event.event_type.to_string(),
                        minutes_until,
                        start_time: event.start_time,
                        timestamp: chrono::Utc::now(),
                    });
                    debug!("📅 🔔 Reminder for '{}' in {} minutes", event.title, minutes_until);
                }
            }
        }
    }
}
