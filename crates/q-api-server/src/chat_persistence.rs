/// P2P Peer-to-Peer Chat Persistence — Phase 3 nova-chat integration
///
/// Stores P2P chat messages in the EXISTING CF_AI_CHATS column family using
/// a `peer-chat:` key namespace prefix — zero conflict with AI chat keys
/// which use `chat:{id}:msg:{n}`.
///
/// Key scheme:
///   peer-chat:msg:{local_wallet}:{peer_wallet}:{msg_id}
///   peer-chat:contact:{local_wallet}:{peer_wallet}
///   peer-chat:conv:{local_wallet}:{peer_wallet}
///
/// No new column families are added — safe for live mainnet RocksDB.
use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::error;
use uuid::Uuid;

use q_storage::{StorageEngine, CF_AI_CHATS};

use std::sync::Arc;

use crate::AppState;

// ─── Types ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerChatMessage {
    pub id: String,
    /// Sender wallet address
    pub from_peer: String,
    /// Recipient wallet address
    pub to_peer: String,
    pub content: String,
    /// Unix milliseconds
    pub timestamp: u64,
    pub delivery_status: PeerMessageStatus,
    pub reply_to: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PeerMessageStatus {
    Sent,
    Delivered,
    Read,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerContact {
    /// Wallet address of the contact
    pub peer_id: String,
    pub display_name: String,
    pub added_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConversation {
    pub peer_id: String,
    pub last_message: Option<PeerChatMessage>,
    pub unread_count: u32,
    pub updated_at: u64,
}

// ─── Storage key helpers ──────────────────────────────────────────────────────

fn msg_key(local_wallet: &str, peer_wallet: &str, msg_id: &str) -> String {
    format!("peer-chat:msg:{}:{}:{}", local_wallet, peer_wallet, msg_id)
}

fn msg_prefix(local_wallet: &str, peer_wallet: &str) -> String {
    format!("peer-chat:msg:{}:{}:", local_wallet, peer_wallet)
}

fn contact_key(local_wallet: &str, peer_wallet: &str) -> String {
    format!("peer-chat:contact:{}:{}", local_wallet, peer_wallet)
}

fn contact_prefix(local_wallet: &str) -> String {
    format!("peer-chat:contact:{}:", local_wallet)
}

fn conv_key(local_wallet: &str, peer_wallet: &str) -> String {
    format!("peer-chat:conv:{}:{}", local_wallet, peer_wallet)
}

fn conv_prefix(local_wallet: &str) -> String {
    format!("peer-chat:conv:{}:", local_wallet)
}

fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ─── Storage functions ────────────────────────────────────────────────────────

/// Store a peer chat message in RocksDB (CF_AI_CHATS, peer-chat:msg namespace).
pub async fn store_message(
    storage: &StorageEngine,
    local_wallet: &str,
    msg: &PeerChatMessage,
) -> anyhow::Result<()> {
    let kv = storage.get_kv();
    let key = msg_key(local_wallet, &msg.to_peer, &msg.id);
    let value = serde_json::to_vec(msg)?;
    kv.put(CF_AI_CHATS, key.as_bytes(), &value).await?;

    // Update conversation index
    let conv = PeerConversation {
        peer_id: msg.to_peer.clone(),
        last_message: Some(msg.clone()),
        unread_count: 0,
        updated_at: msg.timestamp,
    };
    let conv_key_str = conv_key(local_wallet, &msg.to_peer);
    let conv_value = serde_json::to_vec(&conv)?;
    kv.put(CF_AI_CHATS, conv_key_str.as_bytes(), &conv_value).await?;

    Ok(())
}

/// Retrieve messages between local_wallet and peer_wallet, up to `limit`.
/// Returns messages sorted oldest-first.
pub async fn get_messages(
    storage: &StorageEngine,
    local_wallet: &str,
    peer_wallet: &str,
    limit: usize,
) -> anyhow::Result<Vec<PeerChatMessage>> {
    let kv = storage.get_kv();
    let prefix = msg_prefix(local_wallet, peer_wallet);
    let entries = kv.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;

    let mut messages: Vec<PeerChatMessage> = entries
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice::<PeerChatMessage>(&v).ok())
        .collect();

    // Sort by timestamp ascending (oldest first)
    messages.sort_by_key(|m| m.timestamp);

    // Apply limit (take most recent `limit` messages)
    if limit > 0 && messages.len() > limit {
        let skip = messages.len() - limit;
        messages.drain(..skip);
    }

    Ok(messages)
}

/// Update the delivery status of a specific message.
pub async fn update_message_status(
    storage: &StorageEngine,
    local_wallet: &str,
    peer_wallet: &str,
    msg_id: &str,
    status: PeerMessageStatus,
) -> anyhow::Result<()> {
    let kv = storage.get_kv();
    let key = msg_key(local_wallet, peer_wallet, msg_id);
    if let Some(data) = kv.get(CF_AI_CHATS, key.as_bytes()).await? {
        let mut msg: PeerChatMessage = serde_json::from_slice(&data)?;
        msg.delivery_status = status;
        let updated = serde_json::to_vec(&msg)?;
        kv.put(CF_AI_CHATS, key.as_bytes(), &updated).await?;
    }
    Ok(())
}

/// Persist a contact entry for local_wallet.
pub async fn store_contact(
    storage: &StorageEngine,
    local_wallet: &str,
    contact: &PeerContact,
) -> anyhow::Result<()> {
    let kv = storage.get_kv();
    let key = contact_key(local_wallet, &contact.peer_id);
    let value = serde_json::to_vec(contact)?;
    kv.put(CF_AI_CHATS, key.as_bytes(), &value).await?;
    Ok(())
}

/// Return all contacts for local_wallet.
pub async fn get_contacts(
    storage: &StorageEngine,
    local_wallet: &str,
) -> anyhow::Result<Vec<PeerContact>> {
    let kv = storage.get_kv();
    let prefix = contact_prefix(local_wallet);
    let entries = kv.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;

    let contacts: Vec<PeerContact> = entries
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice::<PeerContact>(&v).ok())
        .collect();

    Ok(contacts)
}

/// Return conversation summaries for local_wallet, sorted by updated_at descending.
pub async fn get_conversations(
    storage: &StorageEngine,
    local_wallet: &str,
) -> anyhow::Result<Vec<PeerConversation>> {
    let kv = storage.get_kv();
    let prefix = conv_prefix(local_wallet);
    let entries = kv.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;

    let mut convs: Vec<PeerConversation> = entries
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice::<PeerConversation>(&v).ok())
        .collect();

    // Most recent conversations first
    convs.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(convs)
}

/// Full-text search across all messages for local_wallet (case-insensitive substring).
pub async fn search_messages(
    storage: &StorageEngine,
    local_wallet: &str,
    query: &str,
    limit: usize,
) -> anyhow::Result<Vec<PeerChatMessage>> {
    let kv = storage.get_kv();
    // Scan all message keys for this wallet
    let prefix = format!("peer-chat:msg:{}:", local_wallet);
    let entries = kv.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;

    let query_lower = query.to_lowercase();
    let mut results: Vec<PeerChatMessage> = entries
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice::<PeerChatMessage>(&v).ok())
        .filter(|m| m.content.to_lowercase().contains(&query_lower))
        .collect();

    // Most recent first for search results
    results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    if limit > 0 {
        results.truncate(limit);
    }
    Ok(results)
}

// ─── REST handler parameter types ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatHistoryParams {
    /// The local user's wallet address
    pub wallet: String,
    /// The peer's wallet address
    pub peer: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct WalletParam {
    pub wallet: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub wallet: String,
    pub q: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AddContactRequest {
    /// The local user's wallet address
    pub wallet: String,
    pub peer_id: String,
    pub display_name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StoreMessageRequest {
    /// The local user's wallet address
    pub wallet: String,
    pub from_peer: String,
    pub to_peer: String,
    pub content: String,
    pub reply_to: Option<String>,
}

// ─── REST handlers ────────────────────────────────────────────────────────────

/// GET /api/v1/peer-chat/messages?wallet=...&peer=...&limit=...
pub async fn get_chat_history(
    Query(params): Query<ChatHistoryParams>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(50);
    match get_messages(&state.storage_engine, &params.wallet, &params.peer, limit).await {
        Ok(messages) => (
            StatusCode::OK,
            Json(json!({
                "wallet": params.wallet,
                "peer": params.peer,
                "messages": messages,
                "count": messages.len()
            })),
        )
            .into_response(),
        Err(e) => {
            error!("peer-chat get_messages error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

/// GET /api/v1/peer-chat/conversations?wallet=...
pub async fn list_conversations(
    Query(params): Query<WalletParam>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    match get_conversations(&state.storage_engine, &params.wallet).await {
        Ok(convs) => (
            StatusCode::OK,
            Json(json!({
                "wallet": params.wallet,
                "conversations": convs,
                "count": convs.len()
            })),
        )
            .into_response(),
        Err(e) => {
            error!("peer-chat list_conversations error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

/// GET /api/v1/peer-chat/search?wallet=...&q=...&limit=...
pub async fn search_chat(
    Query(params): Query<SearchParams>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(20);
    match search_messages(&state.storage_engine, &params.wallet, &params.q, limit).await {
        Ok(results) => (
            StatusCode::OK,
            Json(json!({
                "wallet": params.wallet,
                "query": params.q,
                "results": results,
                "count": results.len()
            })),
        )
            .into_response(),
        Err(e) => {
            error!("peer-chat search_messages error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

/// GET /api/v1/peer-chat/contacts?wallet=...
pub async fn get_contacts_handler(
    Query(params): Query<WalletParam>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    match get_contacts(&state.storage_engine, &params.wallet).await {
        Ok(contacts) => (
            StatusCode::OK,
            Json(json!({
                "wallet": params.wallet,
                "contacts": contacts,
                "count": contacts.len()
            })),
        )
            .into_response(),
        Err(e) => {
            error!("peer-chat get_contacts error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

/// POST /api/v1/peer-chat/contacts
pub async fn add_contact_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddContactRequest>,
) -> impl IntoResponse {
    let contact = PeerContact {
        peer_id: req.peer_id.clone(),
        display_name: req.display_name.clone(),
        added_at: current_timestamp_ms(),
    };
    match store_contact(&state.storage_engine, &req.wallet, &contact).await {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({
                "success": true,
                "contact": contact
            })),
        )
            .into_response(),
        Err(e) => {
            error!("peer-chat add_contact error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

/// POST /api/v1/peer-chat/messages — store an outgoing or incoming message
pub async fn store_message_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StoreMessageRequest>,
) -> impl IntoResponse {
    let msg = PeerChatMessage {
        id: Uuid::new_v4().to_string(),
        from_peer: req.from_peer.clone(),
        to_peer: req.to_peer.clone(),
        content: req.content,
        timestamp: current_timestamp_ms(),
        delivery_status: PeerMessageStatus::Sent,
        reply_to: req.reply_to,
    };
    match store_message(&state.storage_engine, &req.wallet, &msg).await {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({
                "success": true,
                "message": msg
            })),
        )
            .into_response(),
        Err(e) => {
            error!("peer-chat store_message error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}
