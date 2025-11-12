# Q-NarwhalKnight Web Chat Storage Integration

## 🎯 Objective

Integrate **q-storage** (RocksDB-based distributed storage) with the web chat interface for:
- ✅ **Persistent chat storage** using RocksDB (not JSON files)
- ✅ **Decentralized chat history** across network nodes
- ✅ **Proper API endpoints** for chat operations
- ✅ **Distributed AI inference** with q-ai-inference

---

## 📊 Current q-storage Architecture

### Existing Infrastructure

**RocksDB Implementation**:
- Hot DB: Frequent access data (blocks, vertices, certificates)
- Cold DB: Large payloads (Narwhal payloads)
- Column Families: `blocks`, `dag_vertices`, `bullshark_cert`, `manifest`, `narwhal_payloads`, `transactions`

**Features**:
- ✅ Atomic batch writes
- ✅ Prefix scanning
- ✅ Sync writes (survives hard kills)
- ✅ Snapshot management
- ✅ Crash recovery
- ✅ Metrics and health monitoring
- ✅ Wallet balance storage
- ✅ Transaction storage
- ✅ Smart contract storage

### New Column Family for Chats

We'll add a new column family: `CF_AI_CHATS`

```rust
pub const CF_AI_CHATS: &str = "ai_chats";
```

---

## 🗄️ Chat Storage Schema

### Key Format

```
chat:{chat_id}           → ChatMetadata (chat settings)
chat:{chat_id}:msg:{idx} → ChatMessage (individual messages)
chat:user:{user_id}      → Vec<chat_id> (user's chats)
chat:latest:{user_id}    → chat_id (last used chat)
```

### Data Structures

```rust
use serde::{Deserialize, Serialize};

/// Chat metadata stored in q-storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMetadata {
    pub chat_id: String,
    pub user_id: String,          // Wallet address or node ID
    pub title: String,
    pub model: String,            // "Mistral-7B-Instruct-v0.3"
    pub created_at: u64,          // Unix timestamp
    pub updated_at: u64,
    pub message_count: u64,

    // Privacy settings
    pub encryption_enabled: bool,
    pub zk_proofs_enabled: bool,
    pub distributed_enabled: bool,

    // Performance settings
    pub enable_kv_cache: bool,
    pub enable_pipeline_parallel: bool,
    pub enable_load_balancing: bool,
}

/// Individual chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub index: u64,               // Message sequence number
    pub role: String,             // "user" | "assistant" | "system"
    pub content: String,
    pub timestamp: u64,

    // Optional attachments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,  // Base64 encoded images

    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<String>,        // Base64 encoded audio

    // Performance metrics (for assistant messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_stats: Option<GenerationStats>,
}

/// Generation statistics from distributed inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
    pub privacy_overhead_ms: f64,
    pub distribution_overhead_ms: f64,
    pub total_time_ms: f64,
    pub tokens_per_second: f64,
    pub active_nodes: usize,
    pub cache_hit_rate: f64,
}
```

---

## 🔧 Implementation in q-storage

### Add New Methods to QStorage

**File**: `crates/q-storage/src/lib.rs`

```rust
// ========================================
// AI CHAT STORAGE METHODS
// ========================================

/// Create new chat
pub async fn create_chat(
    &self,
    user_id: &str,
    title: &str,
    model: &str,
) -> Result<String> {
    let chat_id = format!("chat_{}", uuid::Uuid::new_v4());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

    let metadata = ChatMetadata {
        chat_id: chat_id.clone(),
        user_id: user_id.to_string(),
        title: title.to_string(),
        model: model.to_string(),
        created_at: now,
        updated_at: now,
        message_count: 0,
        encryption_enabled: true,
        zk_proofs_enabled: true,
        distributed_enabled: true,
        enable_kv_cache: true,
        enable_pipeline_parallel: true,
        enable_load_balancing: true,
    };

    let key = format!("chat:{}", chat_id);
    let value = bincode::serialize(&metadata)?;

    self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;

    // Add to user's chat list
    self.add_chat_to_user_list(user_id, &chat_id).await?;

    // Set as latest chat
    self.set_latest_chat(user_id, &chat_id).await?;

    info!("💬 Created new chat: {} for user {}", chat_id, user_id);
    Ok(chat_id)
}

/// Save chat message
pub async fn save_chat_message(
    &self,
    chat_id: &str,
    message: &ChatMessage,
) -> Result<()> {
    let key = format!("chat:{}:msg:{}", chat_id, message.index);
    let value = bincode::serialize(message)?;

    self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;

    // Update chat metadata
    self.update_chat_metadata(chat_id).await?;

    debug!("💬 Saved message {} to chat {}", message.index, chat_id);
    Ok(())
}

/// Load all messages for a chat
pub async fn load_chat_messages(&self, chat_id: &str) -> Result<Vec<ChatMessage>> {
    let prefix = format!("chat:{}:msg:", chat_id);
    let messages_data = self.hot_db.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;

    let mut messages = Vec::new();
    for (_, value) in messages_data {
        let message: ChatMessage = bincode::deserialize(&value)?;
        messages.push(message);
    }

    // Sort by index
    messages.sort_by_key(|m| m.index);

    debug!("💬 Loaded {} messages from chat {}", messages.len(), chat_id);
    Ok(messages)
}

/// Load chat metadata
pub async fn load_chat_metadata(&self, chat_id: &str) -> Result<Option<ChatMetadata>> {
    let key = format!("chat:{}", chat_id);
    match self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
        Some(value) => {
            let metadata: ChatMetadata = bincode::deserialize(&value)?;
            Ok(Some(metadata))
        }
        None => Ok(None),
    }
}

/// List all chats for a user
pub async fn list_user_chats(&self, user_id: &str) -> Result<Vec<ChatMetadata>> {
    let key = format!("chat:user:{}", user_id);
    match self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
        Some(value) => {
            let chat_ids: Vec<String> = bincode::deserialize(&value)?;
            let mut chats = Vec::new();

            for chat_id in chat_ids {
                if let Some(metadata) = self.load_chat_metadata(&chat_id).await? {
                    chats.push(metadata);
                }
            }

            // Sort by updated_at (most recent first)
            chats.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

            Ok(chats)
        }
        None => Ok(Vec::new()),
    }
}

/// Delete chat
pub async fn delete_chat(&self, chat_id: &str, user_id: &str) -> Result<()> {
    // Load metadata to get message count
    let metadata = self.load_chat_metadata(chat_id).await?;
    if let Some(meta) = metadata {
        // Delete all messages
        for i in 0..meta.message_count {
            let key = format!("chat:{}:msg:{}", chat_id, i);
            self.hot_db.delete(CF_AI_CHATS, key.as_bytes()).await?;
        }

        // Delete metadata
        let key = format!("chat:{}", chat_id);
        self.hot_db.delete(CF_AI_CHATS, key.as_bytes()).await?;

        // Remove from user's chat list
        self.remove_chat_from_user_list(user_id, chat_id).await?;

        info!("💬 Deleted chat: {}", chat_id);
    }

    Ok(())
}

/// Rename chat
pub async fn rename_chat(&self, chat_id: &str, new_title: &str) -> Result<()> {
    if let Some(mut metadata) = self.load_chat_metadata(chat_id).await? {
        metadata.title = new_title.to_string();
        metadata.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let key = format!("chat:{}", chat_id);
        let value = bincode::serialize(&metadata)?;

        self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;

        info!("💬 Renamed chat {} to '{}'", chat_id, new_title);
    }

    Ok(())
}

/// Update chat settings (privacy, performance)
pub async fn update_chat_settings(
    &self,
    chat_id: &str,
    encryption: bool,
    zk_proofs: bool,
    distributed: bool,
) -> Result<()> {
    if let Some(mut metadata) = self.load_chat_metadata(chat_id).await? {
        metadata.encryption_enabled = encryption;
        metadata.zk_proofs_enabled = zk_proofs;
        metadata.distributed_enabled = distributed;
        metadata.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let key = format!("chat:{}", chat_id);
        let value = bincode::serialize(&metadata)?;

        self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;

        debug!("💬 Updated settings for chat {}", chat_id);
    }

    Ok(())
}

// ========================================
// HELPER METHODS
// ========================================

async fn update_chat_metadata(&self, chat_id: &str) -> Result<()> {
    if let Some(mut metadata) = self.load_chat_metadata(chat_id).await? {
        metadata.message_count += 1;
        metadata.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let key = format!("chat:{}", chat_id);
        let value = bincode::serialize(&metadata)?;

        self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;
    }

    Ok(())
}

async fn add_chat_to_user_list(&self, user_id: &str, chat_id: &str) -> Result<()> {
    let key = format!("chat:user:{}", user_id);
    let mut chat_ids: Vec<String> = match self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
        Some(value) => bincode::deserialize(&value)?,
        None => Vec::new(),
    };

    chat_ids.push(chat_id.to_string());

    let value = bincode::serialize(&chat_ids)?;
    self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;

    Ok(())
}

async fn remove_chat_from_user_list(&self, user_id: &str, chat_id: &str) -> Result<()> {
    let key = format!("chat:user:{}", user_id);
    if let Some(value) = self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
        let mut chat_ids: Vec<String> = bincode::deserialize(&value)?;
        chat_ids.retain(|id| id != chat_id);

        let value = bincode::serialize(&chat_ids)?;
        self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;
    }

    Ok(())
}

async fn set_latest_chat(&self, user_id: &str, chat_id: &str) -> Result<()> {
    let key = format!("chat:latest:{}", user_id);
    self.hot_db.put(CF_AI_CHATS, key.as_bytes(), chat_id.as_bytes()).await?;
    Ok(())
}

/// Get latest chat ID for user
pub async fn get_latest_chat(&self, user_id: &str) -> Result<Option<String>> {
    let key = format!("chat:latest:{}", user_id);
    match self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
        Some(value) => Ok(Some(String::from_utf8(value)?)),
        None => Ok(None),
    }
}
```

---

## 🌐 API Endpoints Integration

### Update q-api-server Routes

**File**: `crates/q-api-server/src/handlers.rs`

```rust
use q_storage::{ChatMessage, ChatMetadata, GenerationStats};
use q_ai_inference::MistralIntegration;

// ========================================
// AI CHAT API ENDPOINTS
// ========================================

/// POST /api/chat/create
/// Create a new chat
pub async fn create_chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateChatRequest>,
) -> Result<Json<CreateChatResponse>, AppError> {
    let storage = &state.storage;
    let user_id = req.user_id.unwrap_or_else(|| "default_user".to_string());

    let chat_id = storage.create_chat(
        &user_id,
        &req.title.unwrap_or_else(|| "New Chat".to_string()),
        &req.model.unwrap_or_else(|| "Mistral-7B-Instruct-v0.3".to_string()),
    ).await?;

    Ok(Json(CreateChatResponse { chat_id }))
}

/// GET /api/chat/list
/// List all chats for a user
pub async fn list_chats_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListChatsQuery>,
) -> Result<Json<Vec<ChatMetadata>>, AppError> {
    let storage = &state.storage;
    let user_id = params.user_id.unwrap_or_else(|| "default_user".to_string());

    let chats = storage.list_user_chats(&user_id).await?;

    Ok(Json(chats))
}

/// GET /api/chat/{chat_id}/messages
/// Load all messages for a chat
pub async fn load_chat_messages_handler(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
) -> Result<Json<Vec<ChatMessage>>, AppError> {
    let storage = &state.storage;

    let messages = storage.load_chat_messages(&chat_id).await?;

    Ok(Json(messages))
}

/// POST /api/chat/{chat_id}/message
/// Send a message and get AI response
pub async fn send_message_handler(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Json(req): Json<SendMessageRequest>,
) -> Result<Json<ChatMessage>, AppError> {
    let storage = &state.storage;

    // Load chat metadata to get settings
    let metadata = storage.load_chat_metadata(&chat_id).await?
        .ok_or_else(|| AppError::NotFound("Chat not found".to_string()))?;

    // Get current message index
    let user_index = metadata.message_count;

    // Save user message
    let user_message = ChatMessage {
        index: user_index,
        role: "user".to_string(),
        content: req.content.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        images: req.images,
        audio: req.audio,
        generation_stats: None,
    };

    storage.save_chat_message(&chat_id, &user_message).await?;

    // Load all previous messages for context
    let messages = storage.load_chat_messages(&chat_id).await?;
    let prompt = build_prompt_from_messages(&messages);

    // Generate AI response using q-ai-inference
    let integration = &state.ai_integration;
    let (response_text, stats) = integration.generate_with_privacy(
        &prompt,
        metadata.encryption_enabled,
        metadata.zk_proofs_enabled,
    ).await?;

    // Save assistant message
    let assistant_index = user_index + 1;
    let assistant_message = ChatMessage {
        index: assistant_index,
        role: "assistant".to_string(),
        content: response_text,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        images: None,
        audio: None,
        generation_stats: Some(stats),
    };

    storage.save_chat_message(&chat_id, &assistant_message).await?;

    Ok(Json(assistant_message))
}

/// DELETE /api/chat/{chat_id}
/// Delete a chat
pub async fn delete_chat_handler(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Query(params): Query<DeleteChatQuery>,
) -> Result<Json<SuccessResponse>, AppError> {
    let storage = &state.storage;
    let user_id = params.user_id.unwrap_or_else(|| "default_user".to_string());

    storage.delete_chat(&chat_id, &user_id).await?;

    Ok(Json(SuccessResponse { success: true }))
}

/// PUT /api/chat/{chat_id}/rename
/// Rename a chat
pub async fn rename_chat_handler(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Json(req): Json<RenameChatRequest>,
) -> Result<Json<SuccessResponse>, AppError> {
    let storage = &state.storage;

    storage.rename_chat(&chat_id, &req.title).await?;

    Ok(Json(SuccessResponse { success: true }))
}

/// PUT /api/chat/{chat_id}/settings
/// Update chat privacy/performance settings
pub async fn update_chat_settings_handler(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Json(req): Json<UpdateSettingsRequest>,
) -> Result<Json<SuccessResponse>, AppError> {
    let storage = &state.storage;

    storage.update_chat_settings(
        &chat_id,
        req.encryption_enabled,
        req.zk_proofs_enabled,
        req.distributed_enabled,
    ).await?;

    Ok(Json(SuccessResponse { success: true }))
}

// ========================================
// REQUEST/RESPONSE TYPES
// ========================================

#[derive(Debug, Deserialize)]
pub struct CreateChatRequest {
    pub user_id: Option<String>,
    pub title: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateChatResponse {
    pub chat_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ListChatsQuery {
    pub user_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SendMessageRequest {
    pub content: String,
    pub images: Option<Vec<String>>,
    pub audio: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteChatQuery {
    pub user_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RenameChatRequest {
    pub title: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateSettingsRequest {
    pub encryption_enabled: bool,
    pub zk_proofs_enabled: bool,
    pub distributed_enabled: bool,
}

#[derive(Debug, Serialize)]
pub struct SuccessResponse {
    pub success: bool,
}

// ========================================
// HELPER FUNCTIONS
// ========================================

fn build_prompt_from_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "user" => prompt.push_str(&format!("User: {}\n", msg.content)),
            "assistant" => prompt.push_str(&format!("Assistant: {}\n", msg.content)),
            _ => {}
        }
    }

    prompt
}
```

### Register Routes in main.rs

```rust
// AI Chat routes
.route("/api/chat/create", post(create_chat_handler))
.route("/api/chat/list", get(list_chats_handler))
.route("/api/chat/:chat_id/messages", get(load_chat_messages_handler))
.route("/api/chat/:chat_id/message", post(send_message_handler))
.route("/api/chat/:chat_id", delete(delete_chat_handler))
.route("/api/chat/:chat_id/rename", put(rename_chat_handler))
.route("/api/chat/:chat_id/settings", put(update_chat_settings_handler))
```

---

## 🌐 Decentralization: Distributed Chat Sync

### Multi-Node Chat Synchronization

**Goal**: Chat history replicated across network nodes for high availability

**Implementation**: Use existing `sync_protocol` from q-storage

```rust
/// Sync chat data across nodes
pub async fn sync_chats_across_nodes(&self, target_node: &str) -> Result<()> {
    // Get all chats from local storage
    let local_chats = self.hot_db.scan_prefix(CF_AI_CHATS, b"chat:").await?;

    // Send to peer nodes via libp2p
    for (key, value) in local_chats {
        self.sync_protocol.replicate_to_peer(target_node, &key, &value).await?;
    }

    info!("💬 Synced chats to node {}", target_node);
    Ok(())
}

/// Subscribe to chat updates from network
pub async fn subscribe_to_chat_updates(&self) -> Result<()> {
    self.sync_protocol.subscribe_to_cf_updates(CF_AI_CHATS).await?;
    Ok(())
}
```

---

## ✅ Benefits of Using q-storage

### vs JSON Files:
- ✅ **Performance**: RocksDB is 100x faster than file I/O
- ✅ **Atomicity**: Batch writes ensure consistency
- ✅ **Scalability**: Handles millions of messages without degradation
- ✅ **Prefix scanning**: Fast retrieval of chat messages
- ✅ **Crash recovery**: Auto-recovery after node restarts

### vs SQLite:
- ✅ **Distributed**: Native replication across nodes
- ✅ **LSM-tree**: Optimized for write-heavy workloads
- ✅ **No SQL complexity**: Simple key-value interface
- ✅ **Battle-tested**: Used in production blockchain systems

### vs PostgreSQL:
- ✅ **Embedded**: No separate database server required
- ✅ **Lower latency**: Direct memory-mapped access
- ✅ **Simpler deployment**: Single binary
- ✅ **P2P native**: Integrated with libp2p networking

---

## 🚀 Deployment Strategy

### Phase 1: Local Storage (Single Node)
```bash
# User's data stored locally in RocksDB
~/.local/share/q-narwhalknight/
├── hot/
│   ├── ai_chats/          # Chat metadata and messages
│   ├── blocks/            # Blockchain data
│   └── manifest/          # System state
└── cold/
    └── narwhal_payloads/  # Large data
```

### Phase 2: Distributed Storage (Multi-Node)
```
Node A (User's Computer):
  - Chats stored locally in RocksDB
  - Replicated to Node B and Node C

Node B (Cloud Server):
  - Receives replicated chats
  - Serves as backup

Node C (Friend's Node):
  - Participates in distributed inference
  - Syncs chat history for collaborative features
```

### Phase 3: Encrypted Cloud Backup (Optional)
```rust
/// Export encrypted backup to S3/IPFS
pub async fn export_encrypted_backup(&self, user_id: &str) -> Result<Vec<u8>> {
    let chats = self.list_user_chats(user_id).await?;
    let encrypted_data = self.privacy_layer.encrypt_data(&bincode::serialize(&chats)?)?;
    Ok(encrypted_data)
}
```

---

## 📊 Performance Expectations

### Single Node:
- **Create chat**: <1ms
- **Save message**: <2ms
- **Load chat**: <5ms (100 messages)
- **List chats**: <10ms (1000 chats)

### Multi-Node (3 nodes):
- **Replicate chat**: ~50ms (network latency)
- **Sync message**: ~100ms (3-way replication)
- **Distributed query**: ~150ms (consensus read)

---

## 🎯 Next Steps

1. ✅ Add `CF_AI_CHATS` column family to q-storage
2. ✅ Implement chat storage methods in `QStorage`
3. ✅ Create API endpoints in q-api-server
4. ✅ Fork mistral.rs-web-chat and integrate with API
5. ✅ Test local storage with RocksDB
6. ✅ Enable multi-node synchronization
7. ✅ Add encrypted backup/restore

---

**Status**: Ready to implement
**Priority**: High - Uses production-grade q-storage
**Complexity**: Medium - Build on existing RocksDB infrastructure
