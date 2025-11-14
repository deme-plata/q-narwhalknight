# Kimi K2 Thinking Integration Plan

**Date**: 2025-11-12
**Version**: v1.0.5-beta
**Status**: 📋 **DESIGN PHASE**

---

## 🎯 **OBJECTIVE**

Integrate **Kimi K2 Thinking** (1T parameter reasoning model) into Q-NarwhalKnight's AI inference system, providing users with advanced reasoning capabilities and transparent thought process visualization.

---

## 📊 **KIMI K2 SPECIFICATIONS**

### **Model Details**:
- **Parameters**: 1 trillion (1T)
- **Context Length**: 98,304 tokens (96K)
- **Quantization**: UD-TQ1_0 (1.8-bit dynamic)
- **File Size**: ~245 GB
- **Architecture**: Mixture-of-Experts (MoE) with sparse activation
- **Special Feature**: `<think>` reasoning tags show model's thought process

### **Optimal Inference Settings**:
```toml
temperature = 1.0
min_p = 0.01
context_size = 98304
top_p = null  # Disabled (use min_p instead)
top_k = null  # Disabled (use min_p instead)
```

### **Model Download**:
- **HuggingFace Repo**: `unsloth/Kimi-K2-Thinking-GGUF`
- **File**: `Kimi-K2-unsloth.UD-TQ1_0.gguf`
- **URL**: `https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF/resolve/main/Kimi-K2-unsloth.UD-TQ1_0.gguf`

---

## 🏗️ **IMPLEMENTATION PHASES**

### **Phase 1: Model Registration** ✅

**Goal**: Add Kimi K2 to model manager's supported models list.

**File**: `crates/q-ai-inference/src/model_manager.rs`

**Changes**:
```rust
// Line ~86-126: Add Kimi K2 case to ModelMetadata::from_name()
} else if name.contains("Kimi-K2") || name.contains("kimi-k2") {
    (
        "Kimi-K2-unsloth.UD-TQ1_0.gguf".to_string(),
        120,  // Estimated layer count for 1T MoE model
        1000.0,  // 1T parameters
        245000,  // 245 GB with UD-TQ1_0 quantization
    )
} else if name.contains("Mistral-7B") {
    // ... existing code
```

**Line ~125**: Update error message:
```rust
return Err(anyhow!("Unknown model: {}. Supported models: Kimi-K2, Mistral-7B, Mistral-Small-3.2-24B, Llama-7B, Llama-13B, Llama-70B", name));
```

---

### **Phase 2: MistralRsEngine Configuration** ✅

**Goal**: Configure mistral.rs inference engine for Kimi K2's specific requirements.

**File**: `crates/q-ai-inference/src/mistralrs_engine.rs`

**Changes**:

#### **A. Model Detection**:
```rust
// Add to MistralRsEngine::new() around line ~100-150
let is_kimi_k2 = model_name.contains("Kimi-K2") || model_name.contains("kimi-k2");

if is_kimi_k2 {
    info!("🧠 Kimi K2 Thinking model detected - enabling reasoning mode");
    info!("   🤔 Thought process will be visible via <think> tags");
}
```

#### **B. Inference Parameters**:
```rust
// In generate_stream() method, add Kimi K2 parameter overrides:
let (temp, min_p, max_tokens_override) = if model_name.contains("Kimi-K2") {
    (1.0, 0.01, Some(98304))  // Kimi K2 optimal settings
} else {
    (temperature, 0.05, None)  // Standard settings
};

let request = Request::Normal(NormalRequest {
    messages,
    sampling_params: SamplingParams {
        temperature: Some(temp),
        min_p: Some(min_p),
        top_p: None,  // Disabled for Kimi K2
        top_k: None,  // Disabled for Kimi K2
        max_seq_len: max_tokens_override,
        // ... rest of params
    },
});
```

#### **C. MoE Layer Offloading**:
```rust
// In model loader configuration (around line ~200-250):
let loader = if is_kimi_k2 {
    // For 1T MoE model, offload most expert layers to CPU/disk
    NormalLoaderBuilder::new(
        NormalSpecificConfig {
            use_flash_attn: false,  // Not needed for MoE
            repeat_last_n: 64,
        },
        None,  // chat_template (use default)
        Some(TokenSource::CacheToken),
        model_id.clone(),
    )
    .with_no_kv_cache(false)
    .with_prefix_cache_n(Some(16))  // Cache 16 layers in RAM
    .build()
} else {
    // Standard loader for other models
    // ... existing code
};
```

---

### **Phase 3: Chat Template & Reasoning Extraction** ✅

**Goal**: Parse Kimi K2's `<think>` reasoning tags and separate them from final answer.

**File**: `crates/q-ai-inference/src/chat_templates.rs`

**New Function**:
```rust
/// Parse Kimi K2 reasoning output
///
/// Kimi K2 Thinking model outputs its reasoning process in <think> tags:
/// ```
/// <think>
/// The user is asking about quantum computing...
/// I should explain superposition first...
/// </think>
/// Quantum computing uses quantum bits...
/// ```
///
/// This function extracts:
/// - `reasoning`: Content inside <think> tags
/// - `answer`: Content outside <think> tags
pub fn parse_kimi_k2_reasoning(output: &str) -> (Option<String>, String) {
    // Match <think>...</think> tags
    let think_regex = regex::Regex::new(r"(?s)<think>(.*?)</think>").unwrap();

    let mut reasoning_parts = Vec::new();
    let mut answer = output.to_string();

    // Extract all <think> blocks
    for cap in think_regex.captures_iter(output) {
        if let Some(thinking) = cap.get(1) {
            reasoning_parts.push(thinking.as_str().trim().to_string());
        }
    }

    // Remove <think> tags from answer
    answer = think_regex.replace_all(&answer, "").to_string().trim().to_string();

    let reasoning = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join("\n\n---\n\n"))
    };

    (reasoning, answer)
}

/// Get chat template for Kimi K2
pub fn get_kimi_k2_template() -> String {
    // Kimi K2 uses a simple instruction format
    // Based on unsloth's documentation
    r#"{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {{- '<|system|>\n' + message['content'] + '\n' }}
    {%- elif message['role'] == 'user' %}
        {{- '<|user|>\n' + message['content'] + '\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|assistant|>\n' + message['content'] + eos_token + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}"#.to_string()
}
```

**Integration into `generate_stream()`**:
```rust
// After generating response, parse Kimi K2 output:
if model_name.contains("Kimi-K2") {
    let (reasoning, answer) = parse_kimi_k2_reasoning(&full_response);

    // Send reasoning as separate event (if present)
    if let Some(thinking) = reasoning {
        callback(InferenceEvent::Reasoning {
            content: thinking
        }).await?;
    }

    // Send answer as main response
    callback(InferenceEvent::Token {
        content: answer,
        is_final: true
    }).await?;
} else {
    // Standard token streaming for other models
    // ... existing code
}
```

**Add to `InferenceEvent` enum**:
```rust
pub enum InferenceEvent {
    Token { content: String, is_final: bool },
    Reasoning { content: String },  // NEW: For Kimi K2 thinking process
    Done,
    Error { message: String },
}
```

---

### **Phase 4: API Integration** ✅

**Goal**: Expose Kimi K2 reasoning in chat API responses.

**File**: `crates/q-api-server/src/chat_api.rs`

**Changes**:

#### **A. Add Reasoning Field to ChatMessage**:
```rust
// In q-storage/src/lib.rs, update ChatMessage struct:
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub role: String,  // "user" | "assistant" | "system"
    pub content: String,
    pub reasoning: Option<String>,  // NEW: For Kimi K2 thinking process
    pub timestamp: u64,
    pub generation_stats: Option<GenerationStats>,
}
```

#### **B. Store Reasoning in Database**:
```rust
// In stream_message() handler (line ~600-700):
let mut reasoning_accumulator = String::new();
let mut answer_accumulator = String::new();

while let Some(event) = engine.generate_stream(...).await {
    match event {
        InferenceEvent::Reasoning { content } => {
            reasoning_accumulator.push_str(&content);

            // Send reasoning event to frontend
            yield Ok(Event::default()
                .event("reasoning")
                .data(serde_json::json!({
                    "reasoning": content
                }).to_string()));
        },
        InferenceEvent::Token { content, is_final } => {
            answer_accumulator.push_str(&content);

            yield Ok(Event::default()
                .event("token")
                .data(content));
        },
        // ... other events
    }
}

// Save message with reasoning
let assistant_msg = ChatMessage {
    id: Uuid::new_v4().to_string(),
    role: "assistant".to_string(),
    content: answer_accumulator,
    reasoning: if reasoning_accumulator.is_empty() {
        None
    } else {
        Some(reasoning_accumulator)
    },
    timestamp: current_timestamp(),
    generation_stats: Some(stats),
};
```

---

### **Phase 5: Frontend UI** ✅

**Goal**: Display Kimi K2 reasoning process in chat interface.

**File**: `gui/quantum-wallet/src/components/AIChatScreen.tsx`

**Changes**:

#### **A. Add Kimi K2 to Model List**:
```typescript
// Line ~40-60: Update available models
const availableModels = [
  {
    id: 'Mistral-7B-Instruct-v0.3',
    name: 'Mistral 7B',
    description: 'Fast and efficient 7B parameter model',
    parameters: '7B',
    ram: '4.1 GB',
  },
  {
    id: 'Mistral-Small-3.2-24B-Instruct',
    name: 'Mistral Small 24B',
    description: 'Powerful 24B model with advanced reasoning',
    parameters: '24B',
    ram: '14 GB',
  },
  {
    id: 'Kimi-K2-Thinking',
    name: 'Kimi K2 Thinking',
    description: '🧠 Advanced reasoning with thought process visualization',
    parameters: '1T',
    ram: '245 GB',
    icon: '🤔',
    special: 'reasoning',
  },
];
```

#### **B. Handle Reasoning Events**:
```typescript
// In SSE event handler (line ~500-600):
eventSource.addEventListener('reasoning', (event) => {
  const data = JSON.parse(event.data);

  // Accumulate reasoning
  setCurrentReasoning((prev) => prev + data.reasoning);
});

eventSource.addEventListener('token', (event) => {
  const token = event.data;

  // Accumulate answer
  setCurrentAnswer((prev) => prev + token);
});

eventSource.addEventListener('done', () => {
  // Save message with reasoning
  const newMessage: Message = {
    id: generateId(),
    role: 'assistant',
    content: currentAnswer,
    reasoning: currentReasoning || undefined,
    timestamp: Date.now(),
  };

  setMessages((prev) => [...prev, newMessage]);

  // Reset accumulators
  setCurrentReasoning('');
  setCurrentAnswer('');
});
```

#### **C. Display Reasoning in Message**:
```typescript
// In message rendering (line ~800-1000):
{message.reasoning && (
  <div className="mt-2 border-l-2 border-purple-400 pl-3">
    <details className="group">
      <summary className="cursor-pointer text-sm text-purple-400 hover:text-purple-300 transition-colors flex items-center gap-2">
        <Brain className="w-4 h-4" />
        <span>View Reasoning Process</span>
        <ChevronDown className="w-4 h-4 group-open:rotate-180 transition-transform" />
      </summary>
      <div className="mt-2 text-sm text-gray-400 whitespace-pre-wrap font-mono bg-purple-500/5 p-3 rounded">
        {message.reasoning}
      </div>
    </details>
  </div>
)}
```

---

### **Phase 6: Model Download** ⏳

**Goal**: Download Kimi K2 GGUF file to models directory.

**Method 1: Manual Download** (Recommended):
```bash
# On Server Beta (185.182.185.227):
cd /opt/orobit/shared/q-narwhalknight/models

# Download with wget (resume-able)
wget -c https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF/resolve/main/Kimi-K2-unsloth.UD-TQ1_0.gguf

# Verify download
ls -lh Kimi-K2-unsloth.UD-TQ1_0.gguf
# Expected: ~245 GB
```

**Method 2: HuggingFace CLI**:
```bash
# Install HF CLI
pip install huggingface_hub

# Download
huggingface-cli download \
  unsloth/Kimi-K2-Thinking-GGUF \
  Kimi-K2-unsloth.UD-TQ1_0.gguf \
  --local-dir /opt/orobit/shared/q-narwhalknight/models \
  --resume-download
```

**Storage Requirements**:
```bash
# Check available space
df -h /opt/orobit/shared/q-narwhalknight/models

# Expected: Need ~250 GB free
# Server Beta has: 1.8 TB available ✅
```

---

### **Phase 7: Testing** ⏳

**Test Cases**:

#### **1. Basic Reasoning**:
```
Prompt: "Explain why the sky is blue using step-by-step reasoning."

Expected:
- Reasoning section shows thought process:
  * "Need to explain Rayleigh scattering..."
  * "Should mention shorter wavelengths..."
  * "Must explain why not violet..."

- Answer section provides clean explanation
```

#### **2. Complex Problem**:
```
Prompt: "Solve: If Alice has twice as many apples as Bob, and Bob has 3 more apples than Carol, and Carol has 5 apples, how many apples does Alice have?"

Expected:
- Reasoning shows:
  * "Carol has 5 apples"
  * "Bob = Carol + 3 = 5 + 3 = 8 apples"
  * "Alice = 2 × Bob = 2 × 8 = 16 apples"

- Answer: "Alice has 16 apples."
```

#### **3. Model Switching**:
```
Test: Switch from Mistral-7B → Kimi K2 → Mistral-Small-24B

Expected:
- Old model unloaded before new model loaded
- RAM usage: max(Mistral-7B, Kimi K2, Mistral-Small) = 245 GB
- Not: 4.1 + 245 + 14 = 263 GB ✅
```

#### **4. UI Reasoning Display**:
```
Test: Send message with Kimi K2 selected

Expected:
- Reasoning appears in collapsed <details> section
- Brain icon (🧠) visible next to "View Reasoning Process"
- Reasoning styled with monospace font
- Click expands/collapses reasoning
```

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Backend**:
- [ ] Add Kimi K2 to `ModelMetadata::from_name()` (model_manager.rs)
- [ ] Configure MistralRsEngine for Kimi K2 parameters (mistralrs_engine.rs)
- [ ] Implement MoE layer offloading configuration (mistralrs_engine.rs)
- [ ] Add `parse_kimi_k2_reasoning()` function (chat_templates.rs)
- [ ] Add `get_kimi_k2_template()` function (chat_templates.rs)
- [ ] Add `Reasoning` variant to `InferenceEvent` enum (lib.rs)
- [ ] Add `reasoning` field to `ChatMessage` struct (q-storage/src/lib.rs)
- [ ] Update SSE streaming to emit `reasoning` events (chat_api.rs)
- [ ] Store reasoning in database with messages (chat_api.rs)

### **Frontend**:
- [ ] Add Kimi K2 to model selection dropdown (AIChatScreen.tsx)
- [ ] Add reasoning event listener to SSE handler (AIChatScreen.tsx)
- [ ] Create reasoning display component (AIChatScreen.tsx)
- [ ] Add Brain icon import from lucide-react (AIChatScreen.tsx)
- [ ] Style reasoning with purple theme (AIChatScreen.tsx)

### **Infrastructure**:
- [ ] Download Kimi K2 GGUF file (245 GB)
- [ ] Verify model file integrity
- [ ] Test model loading without OOM
- [ ] Measure inference latency

### **Testing**:
- [ ] Test basic reasoning prompt
- [ ] Test complex multi-step problem
- [ ] Test model switching (RAM management)
- [ ] Test reasoning UI display
- [ ] Test reasoning persistence in database
- [ ] Test SSE streaming with reasoning events

---

## 🚀 **DEPLOYMENT STEPS**

### **1. Download Model**:
```bash
cd /opt/orobit/shared/q-narwhalknight/models
wget -c https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF/resolve/main/Kimi-K2-unsloth.UD-TQ1_0.gguf
```

### **2. Implement Backend Changes**:
```bash
# Edit files:
vim crates/q-ai-inference/src/model_manager.rs
vim crates/q-ai-inference/src/mistralrs_engine.rs
vim crates/q-ai-inference/src/chat_templates.rs
vim crates/q-ai-inference/src/lib.rs
vim crates/q-storage/src/lib.rs
vim crates/q-api-server/src/chat_api.rs

# Build
timeout 36000 cargo build --release --package q-api-server
```

### **3. Implement Frontend Changes**:
```bash
cd gui/quantum-wallet
vim src/components/AIChatScreen.tsx

# Build
npm run build

# Deploy
cp -r dist/* dist-final/
```

### **4. Restart Services**:
```bash
systemctl restart q-api-server
systemctl restart nginx
```

### **5. Test**:
```bash
# Open wallet UI
open http://quillon.xyz

# Select Kimi K2 Thinking model
# Send test prompt: "Explain quantum entanglement step by step"
# Verify reasoning appears in collapsible section
```

---

## 📊 **PERFORMANCE EXPECTATIONS**

### **Inference Speed**:
- **First Token Latency**: ~15-30 seconds (1T model)
- **Tokens/Second**: ~1-3 tokens/sec (on CPU)
- **With GPU**: ~10-20 tokens/sec (with 4x A100 80GB)

### **Memory Usage**:
- **RAM**: ~245 GB (UD-TQ1_0 quantization)
- **VRAM**: 0 GB (CPU-only inference)
- **Disk**: 245 GB for model file

### **Context Window**:
- **Max Tokens**: 98,304 (96K context)
- **Estimated Cost**: 10-15 seconds per 1K tokens

---

## ⚠️ **KNOWN LIMITATIONS**

### **1. Large Model Size**:
- **245 GB** is substantial
- Requires high-bandwidth connection for download
- May take **hours to days** to download depending on connection

### **2. CPU Inference**:
- Kimi K2 will run on CPU (no GPU required)
- Slower than GPU but still functional
- MoE architecture helps (only ~10-15% of model active per token)

### **3. First Load Time**:
- Initial model load: ~2-5 minutes
- Subsequent loads (if cached): ~30-60 seconds

### **4. RAM Requirements**:
- Server Beta has **128 GB RAM** ✅
- Kimi K2 needs **245 GB** ❌
- **SOLUTION**: Enable disk swapping or use MoE layer offloading
  - Offload 80% of expert layers to disk
  - Keep only active layers in RAM (~50 GB)
  - Inference will be slower but functional

---

## 🎯 **SUCCESS CRITERIA**

### **Functional Requirements**:
✅ Kimi K2 model loads successfully
✅ Reasoning process appears in `<think>` tags
✅ Reasoning separated from final answer
✅ UI displays reasoning in collapsible section
✅ Reasoning persisted in database
✅ Model switching works without OOM

### **Performance Requirements**:
✅ First token latency < 60 seconds
✅ Inference completes within 5 minutes for 500-token response
✅ RAM usage stays under available capacity
✅ UI remains responsive during inference

### **User Experience**:
✅ Clear indication of reasoning model selected
✅ Reasoning styled distinctly from answer
✅ Brain icon (🧠) visible for reasoning messages
✅ Smooth expand/collapse animation
✅ Reasoning readable and well-formatted

---

## 📝 **NEXT STEPS**

1. **Immediate**: Begin model download (245 GB, ~6-24 hours)
2. **Day 1**: Implement backend changes (Phase 1-3)
3. **Day 1**: Implement frontend changes (Phase 5)
4. **Day 2**: Test inference and reasoning extraction
5. **Day 2**: Deploy to production and announce feature

---

**Generated**: 2025-11-12 10:35
**Status**: 📋 **DESIGN COMPLETE - READY FOR IMPLEMENTATION**
**Next**: Download Kimi K2 model file (245 GB)
