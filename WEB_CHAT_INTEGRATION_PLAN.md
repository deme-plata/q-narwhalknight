# Q-NarwhalKnight Web Chat Integration Plan

## 🎯 Objective

Integrate **mistral.rs web-chat** component with our **distributed AI inference system** (q-ai-inference + mistral.rs), adding **persistent storage** for chat history and **privacy features**.

---

## 📊 Current State Analysis

### What mistral.rs-web-chat Provides ✅

**Architecture**: Full-stack Rust web chat application

**Backend** (`mistralrs-web-chat/src/`):
- ✅ **Axum web server** - HTTP + WebSocket support
- ✅ **Real-time streaming** - WebSocket for chat responses
- ✅ **Multi-model support** - Text, Vision, Speech models
- ✅ **File uploads** - Images, audio, text files
- ✅ **Persistent storage** - JSON files in cache directory
- ✅ **Chat management** - Create, delete, rename, load chats
- ✅ **Web search** - Optional web search integration

**Frontend** (`mistralrs-web-chat/static/`):
- ✅ **Clean UI** - Sidebar + chat window
- ✅ **Markdown rendering** - Using marked.js
- ✅ **File attachments** - Image, audio, text support
- ✅ **Chat history** - Persistent chat list
- ✅ **Model selection** - Switch between models

**Storage Format**:
```json
{
  "title": "Chat Title",
  "model": "Mistral-7B-Instruct-v0.3",
  "kind": "text",
  "created_at": "2025-10-28T04:00:00Z",
  "messages": [
    {
      "role": "user",
      "content": "hello",
      "images": null
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    }
  ]
}
```

**Storage Location**: `~/.cache/mistral.rs/chats/chat_<id>.json`

---

## 🚀 Integration Plan

### Phase 1: Fork and Customize mistral.rs-web-chat

**Create**: `crates/q-web-chat/` (based on `mistralrs-web-chat`)

**Changes Required**:

1. **Rename Package**
   ```toml
   # Cargo.toml
   [package]
   name = "q-web-chat"
   version = "0.1.0"
   ```

2. **Add q-ai-inference Dependency**
   ```toml
   [dependencies]
   q-ai-inference = { path = "../q-ai-inference" }
   mistralrs.workspace = true
   ```

3. **Update Branding**
   - Change "Mistral.rs Chat" → "Q-NarwhalKnight AI Chat"
   - Update favicon and logo
   - Add quantum theme styling

### Phase 2: Integrate Distributed AI Backend

**File**: `crates/q-web-chat/src/inference_backend.rs` (NEW)

```rust
use anyhow::Result;
use q_ai_inference::MistralIntegration;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct DistributedInferenceBackend {
    integration: Arc<RwLock<MistralIntegration>>,
    enable_privacy: bool,
    enable_distributed: bool,
}

impl DistributedInferenceBackend {
    pub async fn new(
        enable_privacy: bool,
        enable_distributed: bool,
    ) -> Result<Self> {
        let integration = MistralIntegration::new().await?;
        Ok(Self {
            integration: Arc::new(RwLock::new(integration)),
            enable_privacy,
            enable_distributed,
        })
    }

    pub async fn generate(
        &self,
        prompt: &str,
    ) -> Result<(String, GenerationStats)> {
        let mut integration = self.integration.write().await;
        integration.generate_with_privacy(
            prompt,
            self.enable_privacy,
            self.enable_privacy,
        ).await
    }

    pub async fn stream_generate(
        &self,
        prompt: &str,
    ) -> Result<impl futures::Stream<Item = Result<String>>> {
        // Implement streaming with distributed inference
        // Connect to q-ai-inference streaming API
        todo!("Implement streaming distributed inference")
    }
}
```

### Phase 3: Enhanced Storage with Privacy

**File**: `crates/q-web-chat/src/storage.rs` (MODIFIED)

**Features**:

1. **Encrypted Chat Storage** (Optional)
   ```rust
   use q_ai_inference::PrivacyLayer;

   pub struct EncryptedChatStorage {
       base_dir: PathBuf,
       privacy_layer: Option<Arc<PrivacyLayer>>,
   }

   impl EncryptedChatStorage {
       pub async fn save_chat(
           &self,
           chat: &ChatFile,
           encrypt: bool,
       ) -> Result<()> {
           let json = serde_json::to_vec_pretty(chat)?;

           let data = if encrypt && self.privacy_layer.is_some() {
               // Encrypt chat history with AEGIS-QL
               let privacy = self.privacy_layer.as_ref().unwrap();
               privacy.encrypt_data(&json)?
           } else {
               json
           };

           let path = self.base_dir.join(format!("chat_{}.json", chat.id));
           tokio::fs::write(path, data).await?;
           Ok(())
       }
   }
   ```

2. **Database Backend** (Optional - Future Enhancement)
   ```rust
   // Use SQLite or PostgreSQL for better querying
   // Table: chats
   // Columns: id, title, model, created_at, updated_at

   // Table: messages
   // Columns: id, chat_id, role, content, images, created_at

   // Benefits:
   // - Full-text search across chats
   // - Better performance for large histories
   // - Easy to add analytics
   ```

3. **Cloud Sync** (Optional - Future Enhancement)
   ```rust
   // Sync chat history to S3/IPFS/etc
   // Enable multi-device access
   // Encrypted backups
   ```

### Phase 4: UI Enhancements

**File**: `crates/q-web-chat/static/index.html` (MODIFIED)

**New Features**:

1. **Privacy Toggle**
   ```html
   <div class="sidebar-card">
     <h3>Privacy Settings</h3>
     <label>
       <input type="checkbox" id="enableEncryption" checked />
       Enable End-to-End Encryption (AEGIS-QL)
     </label>
     <label>
       <input type="checkbox" id="enableZKProofs" checked />
       Enable Zero-Knowledge Proofs (ZK-STARK)
     </label>
     <label>
       <input type="checkbox" id="enableDistributed" checked />
       Enable Distributed Compute
     </label>
   </div>
   ```

2. **Performance Metrics Display**
   ```html
   <div class="sidebar-card">
     <h3>Performance Stats</h3>
     <div id="perfStats">
       <p>Tokens/sec: <span id="tokensPerSec">--</span></p>
       <p>Privacy overhead: <span id="privacyOverhead">--</span></p>
       <p>Active nodes: <span id="activeNodes">--</span></p>
       <p>KV-Cache hit rate: <span id="cacheHitRate">--</span></p>
     </div>
   </div>
   ```

3. **Quantum Visualization**
   ```html
   <div class="sidebar-card">
     <h3>Network Visualization</h3>
     <canvas id="networkCanvas" width="300" height="200"></canvas>
     <!-- Show distributed nodes processing layers -->
   </div>
   ```

4. **Export/Import Chats**
   ```html
   <button id="exportBtn">📤 Export Chat</button>
   <button id="importBtn">📥 Import Chat</button>
   ```

### Phase 5: API Routes

**File**: `crates/q-web-chat/src/handlers/api.rs` (MODIFIED)

**New Endpoints**:

```rust
// Privacy controls
router.route("/api/set_privacy_mode", post(set_privacy_mode))
router.route("/api/get_privacy_status", get(get_privacy_status))

// Performance metrics
router.route("/api/get_stats", get(get_inference_stats))
router.route("/api/get_network_status", get(get_network_status))

// Chat export/import
router.route("/api/export_chat", post(export_chat))
router.route("/api/import_chat", post(import_chat))

// Node management (for distributed inference)
router.route("/api/list_nodes", get(list_active_nodes))
router.route("/api/node_status", get(node_status))
```

### Phase 6: WebSocket Streaming with Metrics

**File**: `crates/q-web-chat/src/handlers/websocket.rs` (MODIFIED)

**Enhanced WebSocket Messages**:

```rust
#[derive(Serialize)]
#[serde(tag = "type")]
enum WsMessage {
    // Existing
    Chunk { content: String },
    Done,
    Error { message: String },

    // NEW: Performance metrics
    Stats {
        tokens_per_second: f64,
        privacy_overhead_ms: f64,
        distribution_overhead_ms: f64,
        cache_hit_rate: f64,
        active_nodes: usize,
    },

    // NEW: Privacy status
    PrivacyStatus {
        encryption_enabled: bool,
        zk_proofs_enabled: bool,
        distributed_enabled: bool,
    },

    // NEW: Node activity
    NodeActivity {
        node_id: String,
        layer_range: (usize, usize),
        status: String, // "processing" | "complete" | "failed"
    },
}
```

---

## 🎨 UI/UX Design

### Color Scheme (Quantum Theme)

```css
:root {
  --primary: #6366f1;      /* Indigo - quantum entanglement */
  --secondary: #8b5cf6;    /* Purple - post-quantum crypto */
  --success: #10b981;      /* Green - successful inference */
  --warning: #f59e0b;      /* Orange - privacy warnings */
  --danger: #ef4444;       /* Red - errors */
  --bg-dark: #0f172a;      /* Dark blue - night sky */
  --bg-light: #1e293b;     /* Lighter blue - panels */
  --text: #f1f5f9;         /* Light text */
  --border: #334155;       /* Borders */
}
```

### Layout

```
┌─────────────────────────────────────────────────────┐
│                 Q-NarwhalKnight AI Chat             │
├──────────┬──────────────────────────────────────────┤
│ Sidebar  │           Chat Window                    │
│          │                                           │
│ Models   │  User: hello                             │
│ ┌──────┐ │  Assistant: Hello! How can I help you?   │
│ │Model1│ │                                           │
│ │Model2│ │  [Input box]                             │
│ └──────┘ │  [Send] [Image] [Audio] [File]           │
│          │                                           │
│ Privacy  │  Performance:                             │
│ ✓ Encrypt│  Tokens/sec: 42.5                        │
│ ✓ ZK     │  Privacy: 89ms (36%)                     │
│ ✓ Distrib│  Nodes: 3                                │
│          │                                           │
│ Chats    │  [Network Visualization Canvas]          │
│ Chat 1   │                                           │
│ Chat 2   │                                           │
│ Chat 3   │                                           │
└──────────┴──────────────────────────────────────────┘
```

---

## 🗄️ Storage Architecture

### Option 1: JSON Files (Current - Simple)

**Pros**: Simple, portable, no dependencies
**Cons**: Limited querying, slower with many chats

**Location**: `~/.cache/q-narwhalknight/chats/`

**Structure**:
```
~/.cache/q-narwhalknight/
├── chats/
│   ├── chat_1.json
│   ├── chat_2.json
│   └── chat_3.json
├── models/
│   └── Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
└── config.json
```

### Option 2: SQLite (Recommended)

**Pros**: Fast queries, full-text search, better for many chats
**Cons**: Slightly more complex

**Schema**:
```sql
CREATE TABLE chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    title TEXT,
    model TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    encrypted BOOLEAN DEFAULT 0
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL, -- 'user' | 'assistant' | 'system'
    content TEXT NOT NULL,
    images TEXT, -- JSON array of base64 images
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
);

CREATE TABLE inference_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    tokens_generated INTEGER,
    generation_time_ms REAL,
    privacy_overhead_ms REAL,
    distribution_overhead_ms REAL,
    tokens_per_second REAL,
    active_nodes INTEGER,
    cache_hit_rate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE INDEX idx_chats_updated ON chats(updated_at DESC);
CREATE INDEX idx_messages_chat ON messages(chat_id, created_at);
CREATE VIRTUAL TABLE messages_fts USING fts5(content);
```

**Dependencies**:
```toml
[dependencies]
sqlx = { version = "0.7", features = ["runtime-tokio", "sqlite"] }
```

### Option 3: PostgreSQL (Enterprise)

**Pros**: Best performance, advanced features, multi-user support
**Cons**: Requires PostgreSQL server

**Use Case**: Production deployments with multiple users

---

## 🔒 Privacy Features

### 1. End-to-End Encryption

**Chat Storage Encryption**:
- User's chat history encrypted with AEGIS-QL before saving
- Decrypted only when loaded into UI
- Encryption key derived from user password/keyfile

**Benefits**:
- Chat history protected at rest
- Even server admin cannot read chats
- Post-quantum secure

### 2. Zero-Knowledge Proofs

**Inference Verification**:
- Every AI response includes ZK-STARK proof
- UI can verify response was computed correctly
- No need to trust the server

**UI Display**:
```
Assistant: Hello! How can I help you today?
✓ Verified (proof: 0x4a7b...) - 8ms verification
```

### 3. Distributed Privacy

**Multi-Node Inference**:
- Input split across multiple nodes
- No single node sees full conversation
- Nodes cannot reconstruct original prompt

**UI Visualization**:
```
Processing:
Node A (30.1.2.5): Layers 0-10  ████████░░ 80%
Node B (30.1.2.8): Layers 11-21 ████████░░ 80%
Node C (30.1.2.9): Layers 22-31 ████░░░░░░ 40%
```

---

## 📋 Implementation Checklist

### Phase 1: Basic Integration ✅
- [x] Fork mistralrs-web-chat → q-web-chat
- [ ] Add q-ai-inference dependency
- [ ] Update branding and styling
- [ ] Test basic chat functionality

### Phase 2: Distributed Backend 🔄
- [ ] Create `inference_backend.rs`
- [ ] Integrate `MistralIntegration`
- [ ] Add privacy toggles to API
- [ ] Implement streaming with distributed inference

### Phase 3: Enhanced Storage 📝
- [ ] Keep JSON files as default (simplest)
- [ ] Add optional SQLite support (future)
- [ ] Implement encrypted storage
- [ ] Add export/import functionality

### Phase 4: UI Enhancements 🎨
- [ ] Add privacy controls to sidebar
- [ ] Display performance metrics
- [ ] Add network visualization canvas
- [ ] Improve mobile responsiveness

### Phase 5: Advanced Features 🚀
- [ ] Full-text search across chats
- [ ] Chat folders/organization
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Syntax highlighting for code blocks

---

## 🚀 Deployment

### Development Mode

```bash
# Build q-web-chat
cd crates/q-web-chat
cargo build --release

# Run with privacy enabled
./target/release/q-web-chat \
  --text-model /opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf \
  --port 8080 \
  --enable-privacy \
  --enable-distributed
```

### Production Mode

```bash
# Compile with optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release --package q-web-chat

# Run as systemd service
sudo systemctl enable q-web-chat
sudo systemctl start q-web-chat

# Access at http://localhost:8080
```

### Docker Deployment

```dockerfile
FROM rust:1.70 AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --package q-web-chat

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates
COPY --from=builder /app/target/release/q-web-chat /usr/local/bin/
COPY --from=builder /app/models /models
EXPOSE 8080
CMD ["q-web-chat", "--text-model", "/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", "--port", "8080"]
```

---

## 📊 Success Metrics

**User Experience**:
- ✅ Chat loads in <100ms
- ✅ First token in <500ms
- ✅ Smooth streaming (no jitter)
- ✅ Mobile-friendly UI

**Privacy**:
- ✅ All chats encrypted at rest
- ✅ ZK proofs verify every response
- ✅ Multi-node distributed inference

**Performance**:
- ✅ 40+ tokens/sec generation
- ✅ <200ms privacy overhead
- ✅ 3-5x speedup with KV-cache

**Storage**:
- ✅ Persistent chat history
- ✅ Export/import functionality
- ✅ Full-text search (SQLite mode)

---

## 🎯 Next Steps

1. **Complete mistral.rs build** - Wait for current build to finish
2. **Fork web-chat** - Create `crates/q-web-chat/`
3. **Basic integration** - Connect to q-ai-inference
4. **Test with "hello"** - Verify end-to-end flow
5. **Deploy locally** - Run on localhost:8080
6. **Add enhancements** - Privacy controls, metrics, visualization

---

**Status**: Ready to implement once mistral.rs build completes
**Priority**: High - User-facing chat interface
**Complexity**: Medium - Build on existing mistral.rs-web-chat foundation
