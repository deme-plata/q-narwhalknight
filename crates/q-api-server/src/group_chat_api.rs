/// Group Chat API — Discord-like server-backed group chat for Q-NarwhalKnight
///
/// Key namespace (all keys prefixed with "group:" to avoid collisions):
///   group:meta:{group_id}                       → GroupMeta JSON
///   group:member:{group_id}:{wallet}            → GroupMember JSON
///   group:msg:{group_id}:{ts_us:020}:{msg_id}   → GroupMessage JSON
///   group:membership:{wallet}:{group_id}        → b"1" (reverse index)
///   group:invite:{token}                        → GroupInvite JSON
///
/// Storage: RocksDB CF_AI_CHATS column family via KVStore trait.
/// Real-time: tokio broadcast channels per group, exposed via SSE.
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
    routing::{delete, get, post},
    Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use tracing::error;
use uuid::Uuid;

use q_storage::{StorageEngine, CF_AI_CHATS};

use crate::wallet_auth::AuthenticatedWallet;
use crate::AppState;

// ─── SSE broadcast map ────────────────────────────────────────────────────────

/// Per-group broadcast senders. Keyed by group_id.
static GROUP_SSE_SENDERS: OnceLock<DashMap<String, broadcast::Sender<String>>> = OnceLock::new();

fn sse_senders() -> &'static DashMap<String, broadcast::Sender<String>> {
    GROUP_SSE_SENDERS.get_or_init(DashMap::new)
}

/// Get or create a broadcast sender for the given group.
fn sender_for_group(group_id: &str) -> broadcast::Sender<String> {
    let map = sse_senders();
    if let Some(tx) = map.get(group_id) {
        return tx.clone();
    }
    let (tx, _) = broadcast::channel(64);
    map.insert(group_id.to_string(), tx.clone());
    tx
}

/// Broadcast a JSON event to all SSE subscribers of a group.
fn broadcast_event(group_id: &str, payload: serde_json::Value) {
    let map = sse_senders();
    if let Some(tx) = map.get(group_id) {
        let _ = tx.send(payload.to_string());
    }
}

// ─── Domain types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMeta {
    pub id: String,
    pub name: String,
    pub description: String,
    pub owner: String,
    pub created_at: u64,
    pub member_count: u32,
    pub icon: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GroupRole {
    Owner,
    Admin,
    Member,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMember {
    pub wallet: String,
    pub display_name: String,
    pub joined_at: u64,
    pub role: GroupRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMessage {
    pub id: String,
    pub group_id: String,
    pub from: String,
    pub display_name: String,
    pub content: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupInvite {
    pub token: String,
    pub group_id: String,
    pub created_by: String,
    pub expires_at: u64,
    pub max_uses: u32,
    pub use_count: u32,
}

// ─── Request / response types ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct CreateGroupRequest {
    name: String,
    description: Option<String>,
    icon: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SendMessageRequest {
    content: String,
    display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CreateInviteRequest {
    expires_in_hours: Option<u64>,
    max_uses: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct JoinGroupRequest {
    token: String,
    display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessagesQuery {
    /// Unix ms cursor — return messages older than this
    before: Option<u64>,
    limit: Option<usize>,
}

// ─── Key helpers ──────────────────────────────────────────────────────────────

fn key_meta(group_id: &str) -> String {
    format!("group:meta:{}", group_id)
}

fn key_member(group_id: &str, wallet: &str) -> String {
    format!("group:member:{}:{}", group_id, wallet)
}

fn prefix_members(group_id: &str) -> String {
    format!("group:member:{}:", group_id)
}

fn key_msg(group_id: &str, ts_us: u64, msg_id: &str) -> String {
    format!("group:msg:{}:{:020}:{}", group_id, ts_us, msg_id)
}

fn prefix_msgs(group_id: &str) -> String {
    format!("group:msg:{}:", group_id)
}

fn key_membership(wallet: &str, group_id: &str) -> String {
    format!("group:membership:{}:{}", wallet, group_id)
}

fn prefix_memberships(wallet: &str) -> String {
    format!("group:membership:{}:", wallet)
}

fn key_invite(token: &str) -> String {
    format!("group:invite:{}", token)
}

// ─── Time helper ─────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

// ─── Display helper ───────────────────────────────────────────────────────────

fn short_id(addr: &str) -> String {
    if addr.len() > 12 {
        format!("{}…{}", &addr[..6], &addr[addr.len() - 4..])
    } else {
        addr.to_string()
    }
}

// ─── Low-level storage helpers ────────────────────────────────────────────────

async fn load_meta(storage: &StorageEngine, group_id: &str) -> anyhow::Result<Option<GroupMeta>> {
    let kv = storage.get_kv();
    let key = key_meta(group_id);
    match kv.get(CF_AI_CHATS, key.as_bytes()).await? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

async fn save_meta(storage: &StorageEngine, meta: &GroupMeta) -> anyhow::Result<()> {
    let kv = storage.get_kv();
    let key = key_meta(&meta.id);
    let value = serde_json::to_vec(meta)?;
    kv.put(CF_AI_CHATS, key.as_bytes(), &value).await
}

async fn load_member(
    storage: &StorageEngine,
    group_id: &str,
    wallet: &str,
) -> anyhow::Result<Option<GroupMember>> {
    let kv = storage.get_kv();
    let key = key_member(group_id, wallet);
    match kv.get(CF_AI_CHATS, key.as_bytes()).await? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

async fn save_member(
    storage: &StorageEngine,
    group_id: &str,
    member: &GroupMember,
) -> anyhow::Result<()> {
    let kv = storage.get_kv();
    let key = key_member(group_id, &member.wallet);
    let value = serde_json::to_vec(member)?;
    kv.put(CF_AI_CHATS, key.as_bytes(), &value).await
}

async fn delete_member(
    storage: &StorageEngine,
    group_id: &str,
    wallet: &str,
) -> anyhow::Result<()> {
    let kv = storage.get_kv();
    let key = key_member(group_id, wallet);
    kv.delete(CF_AI_CHATS, key.as_bytes()).await
}

async fn list_members(
    storage: &StorageEngine,
    group_id: &str,
) -> anyhow::Result<Vec<GroupMember>> {
    let kv = storage.get_kv();
    let prefix = prefix_members(group_id);
    let entries = kv.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;
    let members = entries
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice::<GroupMember>(&v).ok())
        .collect();
    Ok(members)
}

async fn set_membership_index(
    storage: &StorageEngine,
    wallet: &str,
    group_id: &str,
) -> anyhow::Result<()> {
    let kv = storage.get_kv();
    let key = key_membership(wallet, group_id);
    kv.put(CF_AI_CHATS, key.as_bytes(), b"1").await
}

async fn delete_membership_index(
    storage: &StorageEngine,
    wallet: &str,
    group_id: &str,
) -> anyhow::Result<()> {
    let kv = storage.get_kv();
    let key = key_membership(wallet, group_id);
    kv.delete(CF_AI_CHATS, key.as_bytes()).await
}

async fn list_group_ids_for_wallet(
    storage: &StorageEngine,
    wallet: &str,
) -> anyhow::Result<Vec<String>> {
    let kv = storage.get_kv();
    let prefix = prefix_memberships(wallet);
    let prefix_len = prefix.len();
    let entries = kv.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;
    let ids = entries
        .into_iter()
        .filter_map(|(k, _)| {
            // key bytes: "group:membership:{wallet}:{group_id}"
            // strip prefix bytes to get group_id
            std::str::from_utf8(&k)
                .ok()
                .and_then(|s| s.get(prefix_len..))
                .map(|s| s.to_string())
        })
        .collect();
    Ok(ids)
}

async fn save_invite(storage: &StorageEngine, invite: &GroupInvite) -> anyhow::Result<()> {
    let kv = storage.get_kv();
    let key = key_invite(&invite.token);
    let value = serde_json::to_vec(invite)?;
    kv.put(CF_AI_CHATS, key.as_bytes(), &value).await
}

async fn load_invite(
    storage: &StorageEngine,
    token: &str,
) -> anyhow::Result<Option<GroupInvite>> {
    let kv = storage.get_kv();
    let key = key_invite(token);
    match kv.get(CF_AI_CHATS, key.as_bytes()).await? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

async fn save_message(storage: &StorageEngine, msg: &GroupMessage) -> anyhow::Result<()> {
    let kv = storage.get_kv();
    let ts_us = now_us();
    let key = key_msg(&msg.group_id, ts_us, &msg.id);
    let value = serde_json::to_vec(msg)?;
    kv.put(CF_AI_CHATS, key.as_bytes(), &value).await
}

async fn load_messages(
    storage: &StorageEngine,
    group_id: &str,
    before_ms: Option<u64>,
    limit: usize,
) -> anyhow::Result<Vec<GroupMessage>> {
    let kv = storage.get_kv();
    let prefix = prefix_msgs(group_id);
    let entries = kv.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;

    let mut messages: Vec<GroupMessage> = entries
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice::<GroupMessage>(&v).ok())
        .collect();

    // Sort by timestamp ascending (lex ordering on keys already guarantees this,
    // but scan order is not guaranteed across all KV implementations).
    messages.sort_by_key(|m| m.timestamp);

    // Apply before-cursor filter
    if let Some(before) = before_ms {
        messages.retain(|m| m.timestamp < before);
    }

    // Return the most recent `limit` messages (tail)
    let effective_limit = limit.min(200).max(1);
    if messages.len() > effective_limit {
        let skip = messages.len() - effective_limit;
        messages.drain(..skip);
    }

    Ok(messages)
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// POST / — create a new group (201)
pub async fn create_group(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateGroupRequest>,
) -> impl IntoResponse {
    let wallet = hex::encode(auth.address);

    let name = req.name.trim().to_string();
    if name.is_empty() || name.len() > 64 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Group name must be 1–64 characters"})),
        )
            .into_response();
    }

    let group_id = Uuid::new_v4().to_string().replace('-', "");
    let meta = GroupMeta {
        id: group_id.clone(),
        name,
        description: req.description.unwrap_or_default(),
        owner: wallet.clone(),
        created_at: now_secs(),
        member_count: 1,
        icon: req.icon,
    };

    let owner_member = GroupMember {
        wallet: wallet.clone(),
        display_name: short_id(&wallet),
        joined_at: now_ms(),
        role: GroupRole::Owner,
    };

    let storage = &state.storage_engine;
    if let Err(e) = save_meta(storage, &meta).await {
        error!("group_chat create_group save_meta: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Storage error"})),
        )
            .into_response();
    }
    if let Err(e) = save_member(storage, &group_id, &owner_member).await {
        error!("group_chat create_group save_member: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Storage error"})),
        )
            .into_response();
    }
    if let Err(e) = set_membership_index(storage, &wallet, &group_id).await {
        error!("group_chat create_group membership_index: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Storage error"})),
        )
            .into_response();
    }

    (StatusCode::CREATED, Json(json!({"group": meta}))).into_response()
}

/// GET / — list all groups the authenticated wallet belongs to (200)
pub async fn list_my_groups(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let wallet = hex::encode(auth.address);
    let storage = &state.storage_engine;

    let group_ids = match list_group_ids_for_wallet(storage, &wallet).await {
        Ok(ids) => ids,
        Err(e) => {
            error!("group_chat list_my_groups: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    let mut groups = Vec::with_capacity(group_ids.len());
    for id in &group_ids {
        match load_meta(storage, id).await {
            Ok(Some(meta)) => groups.push(meta),
            Ok(None) => {} // Group was deleted, skip stale index entry
            Err(e) => {
                error!("group_chat list_my_groups load_meta {}: {}", id, e);
            }
        }
    }

    (StatusCode::OK, Json(json!({"groups": groups}))).into_response()
}

/// GET /:id — get group details (member-only)
pub async fn get_group(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Path(group_id): Path<String>,
) -> impl IntoResponse {
    let wallet = hex::encode(auth.address);
    let storage = &state.storage_engine;

    let meta = match load_meta(storage, &group_id).await {
        Ok(Some(m)) => m,
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            error!("group_chat get_group load_meta: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    match load_member(storage, &group_id, &wallet).await {
        Ok(None) => return StatusCode::FORBIDDEN.into_response(),
        Err(e) => {
            error!("group_chat get_group load_member: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
    }

    // Also load full member list so the frontend can populate the members panel
    let members = match list_members(storage, &group_id).await {
        Ok(m) => m,
        Err(e) => {
            error!("group_chat get_group list_members: {}", e);
            vec![]
        }
    };

    (StatusCode::OK, Json(json!({"group": meta, "members": members}))).into_response()
}

/// DELETE /:id — delete group (owner-only)
pub async fn delete_group(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Path(group_id): Path<String>,
) -> impl IntoResponse {
    let wallet = hex::encode(auth.address);
    let storage = &state.storage_engine;

    let meta = match load_meta(storage, &group_id).await {
        Ok(Some(m)) => m,
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            error!("group_chat delete_group load_meta: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    if meta.owner != wallet {
        return StatusCode::FORBIDDEN.into_response();
    }

    // Remove all member records and their reverse-index entries
    let members = match list_members(storage, &group_id).await {
        Ok(m) => m,
        Err(e) => {
            error!("group_chat delete_group list_members: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    for member in &members {
        let _ = delete_member(storage, &group_id, &member.wallet).await;
        let _ = delete_membership_index(storage, &member.wallet, &group_id).await;
    }

    // Delete meta
    let kv = storage.get_kv();
    let _ = kv
        .delete(CF_AI_CHATS, key_meta(&group_id).as_bytes())
        .await;

    broadcast_event(&group_id, json!({"type": "group_deleted", "group_id": group_id}));

    (StatusCode::OK, Json(json!({"success": true}))).into_response()
}

/// POST /:id/invites — create an invite token (member)
pub async fn create_invite(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Path(group_id): Path<String>,
    Json(req): Json<CreateInviteRequest>,
) -> impl IntoResponse {
    let wallet = hex::encode(auth.address);
    let storage = &state.storage_engine;

    match load_meta(storage, &group_id).await {
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            error!("group_chat create_invite load_meta: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
    }

    match load_member(storage, &group_id, &wallet).await {
        Ok(None) => return StatusCode::FORBIDDEN.into_response(),
        Err(e) => {
            error!("group_chat create_invite load_member: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
    }

    let token = Uuid::new_v4().to_string().replace('-', "");
    let expires_at = req
        .expires_in_hours
        .filter(|&h| h > 0)
        .map(|h| now_secs() + h * 3600)
        .unwrap_or(0);

    let invite = GroupInvite {
        token: token.clone(),
        group_id: group_id.clone(),
        created_by: wallet,
        expires_at,
        max_uses: req.max_uses.unwrap_or(0),
        use_count: 0,
    };

    if let Err(e) = save_invite(storage, &invite).await {
        error!("group_chat create_invite save_invite: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Storage error"})),
        )
            .into_response();
    }

    // Return the token at the top level so the frontend can read `data.token` directly
    (StatusCode::CREATED, Json(json!({"token": token, "invite": invite}))).into_response()
}

/// POST /join — join a group via invite token
pub async fn join_group(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(req): Json<JoinGroupRequest>,
) -> impl IntoResponse {
    let wallet = hex::encode(auth.address);
    let storage = &state.storage_engine;

    let mut invite = match load_invite(storage, &req.token).await {
        Ok(Some(i)) => i,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Invite token not found"})),
            )
                .into_response()
        }
        Err(e) => {
            error!("group_chat join_group load_invite: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    // Check expiry
    if invite.expires_at > 0 && now_secs() > invite.expires_at {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Invite token has expired"})),
        )
            .into_response();
    }

    // Check max uses
    if invite.max_uses > 0 && invite.use_count >= invite.max_uses {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Invite token has reached its use limit"})),
        )
            .into_response();
    }

    let group_id = invite.group_id.clone();

    let mut meta = match load_meta(storage, &group_id).await {
        Ok(Some(m)) => m,
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            error!("group_chat join_group load_meta: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    // Check already a member
    match load_member(storage, &group_id, &wallet).await {
        Ok(Some(_)) => {
            return (
                StatusCode::CONFLICT,
                Json(json!({"error": "Already a member of this group"})),
            )
                .into_response()
        }
        Err(e) => {
            error!("group_chat join_group load_member: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(None) => {}
    }

    let display_name = req
        .display_name
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| short_id(&wallet));

    let member = GroupMember {
        wallet: wallet.clone(),
        display_name: display_name.clone(),
        joined_at: now_ms(),
        role: GroupRole::Member,
    };

    if let Err(e) = save_member(storage, &group_id, &member).await {
        error!("group_chat join_group save_member: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Storage error"})),
        )
            .into_response();
    }

    if let Err(e) = set_membership_index(storage, &wallet, &group_id).await {
        error!("group_chat join_group membership_index: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Storage error"})),
        )
            .into_response();
    }

    // Increment use_count on invite
    invite.use_count += 1;
    let _ = save_invite(storage, &invite).await;

    // Update member_count in meta
    meta.member_count = meta.member_count.saturating_add(1);
    let _ = save_meta(storage, &meta).await;

    broadcast_event(
        &group_id,
        json!({
            "type": "member_joined",
            "group_id": group_id,
            "wallet": wallet,
            "display_name": display_name,
        }),
    );

    (StatusCode::OK, Json(json!({"group": meta, "member": member}))).into_response()
}

/// POST /:id/leave — leave group (member, not owner)
pub async fn leave_group(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Path(group_id): Path<String>,
) -> impl IntoResponse {
    let wallet = hex::encode(auth.address);
    let storage = &state.storage_engine;

    let mut meta = match load_meta(storage, &group_id).await {
        Ok(Some(m)) => m,
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            error!("group_chat leave_group load_meta: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    match load_member(storage, &group_id, &wallet).await {
        Ok(None) => return StatusCode::FORBIDDEN.into_response(),
        Err(e) => {
            error!("group_chat leave_group load_member: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
    }

    if meta.owner == wallet {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Owner cannot leave the group. Transfer ownership or delete the group."})),
        )
            .into_response();
    }

    if let Err(e) = delete_member(storage, &group_id, &wallet).await {
        error!("group_chat leave_group delete_member: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Storage error"})),
        )
            .into_response();
    }

    let _ = delete_membership_index(storage, &wallet, &group_id).await;

    meta.member_count = meta.member_count.saturating_sub(1);
    let _ = save_meta(storage, &meta).await;

    broadcast_event(
        &group_id,
        json!({"type": "member_left", "group_id": group_id, "wallet": wallet}),
    );

    (StatusCode::OK, Json(json!({"success": true}))).into_response()
}

/// DELETE /:id/members/:wallet — kick a member (owner or admin)
pub async fn kick_member(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Path((group_id, target_wallet)): Path<(String, String)>,
) -> impl IntoResponse {
    let caller = hex::encode(auth.address);
    let storage = &state.storage_engine;

    let mut meta = match load_meta(storage, &group_id).await {
        Ok(Some(m)) => m,
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            error!("group_chat kick_member load_meta: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    // Caller must be a member with owner or admin role
    let caller_member = match load_member(storage, &group_id, &caller).await {
        Ok(Some(m)) => m,
        Ok(None) => return StatusCode::FORBIDDEN.into_response(),
        Err(e) => {
            error!("group_chat kick_member load_caller: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    if caller_member.role != GroupRole::Owner && caller_member.role != GroupRole::Admin {
        return StatusCode::FORBIDDEN.into_response();
    }

    // Cannot kick the owner
    if meta.owner == target_wallet {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Cannot kick the group owner"})),
        )
            .into_response();
    }

    match load_member(storage, &group_id, &target_wallet).await {
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Member not found"})),
            )
                .into_response()
        }
        Err(e) => {
            error!("group_chat kick_member load_target: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
    }

    if let Err(e) = delete_member(storage, &group_id, &target_wallet).await {
        error!("group_chat kick_member delete_member: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Storage error"})),
        )
            .into_response();
    }

    let _ = delete_membership_index(storage, &target_wallet, &group_id).await;

    meta.member_count = meta.member_count.saturating_sub(1);
    let _ = save_meta(storage, &meta).await;

    broadcast_event(
        &group_id,
        json!({"type": "member_kicked", "group_id": group_id, "wallet": target_wallet}),
    );

    (StatusCode::OK, Json(json!({"success": true}))).into_response()
}

/// GET /:id/messages — paginated message history (member)
pub async fn get_messages(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Path(group_id): Path<String>,
    Query(query): Query<MessagesQuery>,
) -> impl IntoResponse {
    let wallet = hex::encode(auth.address);
    let storage = &state.storage_engine;

    match load_meta(storage, &group_id).await {
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            error!("group_chat get_messages load_meta: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
    }

    match load_member(storage, &group_id, &wallet).await {
        Ok(None) => return StatusCode::FORBIDDEN.into_response(),
        Err(e) => {
            error!("group_chat get_messages load_member: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
    }

    let limit = query.limit.unwrap_or(50).min(200).max(1);
    let messages = match load_messages(storage, &group_id, query.before, limit).await {
        Ok(m) => m,
        Err(e) => {
            error!("group_chat get_messages load_messages: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    (StatusCode::OK, Json(json!({"messages": messages, "count": messages.len()}))).into_response()
}

/// POST /:id/messages — send a message (member, 201)
pub async fn send_message(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Path(group_id): Path<String>,
    Json(req): Json<SendMessageRequest>,
) -> impl IntoResponse {
    let wallet = hex::encode(auth.address);
    let storage = &state.storage_engine;

    match load_meta(storage, &group_id).await {
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            error!("group_chat send_message load_meta: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
    }

    let member = match load_member(storage, &group_id, &wallet).await {
        Ok(Some(m)) => m,
        Ok(None) => return StatusCode::FORBIDDEN.into_response(),
        Err(e) => {
            error!("group_chat send_message load_member: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
    };

    let content = req.content.trim().to_string();
    if content.is_empty() || content.len() > 4000 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Message content must be 1–4000 characters"})),
        )
            .into_response();
    }

    let display_name = req
        .display_name
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| member.display_name.clone());

    let msg = GroupMessage {
        id: Uuid::new_v4().to_string().replace('-', ""),
        group_id: group_id.clone(),
        from: wallet,
        display_name,
        content,
        timestamp: now_ms(),
    };

    if let Err(e) = save_message(storage, &msg).await {
        error!("group_chat send_message save_message: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Storage error"})),
        )
            .into_response();
    }

    broadcast_event(
        &group_id,
        json!({"type": "new_message", "message": msg}),
    );

    (StatusCode::CREATED, Json(json!({"message": msg}))).into_response()
}

/// GET /:id/stream — SSE stream for real-time group events (member)
pub async fn group_sse_stream(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Path(group_id): Path<String>,
) -> impl IntoResponse {
    let wallet = hex::encode(auth.address);
    let storage = &state.storage_engine;

    // Verify group exists and caller is a member
    match load_meta(storage, &group_id).await {
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            error!("group_chat group_sse_stream load_meta: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
    }

    match load_member(storage, &group_id, &wallet).await {
        Ok(None) => return StatusCode::FORBIDDEN.into_response(),
        Err(e) => {
            error!("group_chat group_sse_stream load_member: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Storage error"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
    }

    let tx = sender_for_group(&group_id);
    let rx = tx.subscribe();

    let stream = BroadcastStream::new(rx).filter_map(|result| match result {
        Ok(json_str) => Some(Ok::<Event, axum::Error>(
            Event::default().event("group-event").data(json_str),
        )),
        Err(_) => None, // Lagged — skip
    });

    Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(std::time::Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}

// ─── Router ───────────────────────────────────────────────────────────────────

pub fn group_chat_router() -> Router<Arc<AppState>> {
    Router::new()
        // Group CRUD
        .route("/", post(create_group).get(list_my_groups))
        .route("/:id", get(get_group).delete(delete_group))
        // Membership
        .route("/join", post(join_group))
        .route("/:id/leave", post(leave_group))
        .route("/:id/members/:wallet", delete(kick_member))
        // Invites
        .route("/:id/invites", post(create_invite))
        // Messages
        .route("/:id/messages", get(get_messages).post(send_message))
        // SSE
        .route("/:id/stream", get(group_sse_stream))
}
