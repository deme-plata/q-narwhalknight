# Phase 6: Server-Sent Events (SSE) Streaming for AI Chat

**Date**: 2025-10-28
**Prerequisites**: ✅ Phase 5 Complete (KV-cache AI inference integrated)
**Status**: Planning Phase

---

## 🎯 Goal: Real-Time Token Streaming

Enable real-time streaming of AI-generated tokens as they're produced, providing instant user feedback instead of waiting for entire response completion.

### Current Behavior (Phase 5):
- User sends message via POST `/api/chat/{chat_id}/message`
- Server generates ENTIRE response (e.g., 50 tokens × 6.5s = 325 seconds)
- User waits ~5+ minutes
- Server returns complete response

### Target Behavior (Phase 6):
- User connects to SSE endpoint `/api/chat/{chat_id}/stream`
- Server starts generating response
- **Each token streams immediately** to client (every ~6.5s)
- User sees response building in real-time
- Much better UX - first token appears in ~94s instead of 5+ minutes

---

## 📋 Implementation Plan

### Task 1: Add SSE Endpoint to chat_api.rs

**Location**: `crates/q-api-server/src/chat_api.rs`

**New Endpoint**:
```rust
use axum::response::sse::{Event, Sse};
use tokio_stream::wrappers::ReceiverStream;
use futures::stream::{self, Stream};
use std::convert::Infallible;

/// SSE streaming endpoint for real-time token generation
/// GET /api/chat/:chat_id/stream?content=Hello
pub async fn stream_message(
    State(state): State<Arc<AppState>>,
    Path(chat_id): Path<String>,
    Query(req): Query<SendMessageRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    // Spawn background task for generation
    tokio::spawn(async move {
        // Send initial event
        let _ = tx.send(Ok(Event::default()
            .event("start")
            .data("Generation started")
        )).await;

        // Check if inference engine is available
        if let Some(ref engine) = state.inference_engine {
            let mut engine_guard = engine.lock().await;

            // Format prompt with Mistral instruction template
            let formatted_prompt = format!("[INST] {} [/INST]", req.content);

            // Generate tokens one-by-one
            let max_tokens = 150;
            let mut generated_text = String::new();

            for token_idx in 0..max_tokens {
                match engine_guard.generate_next_token(&formatted_prompt, token_idx).await {
                    Ok((token_id, token_text)) => {
                        generated_text.push_str(&token_text);

                        // Send token event
                        let event = Event::default()
                            .event("token")
                            .data(serde_json::json!({
                                "token": token_text,
                                "token_id": token_id,
                                "position": token_idx,
                                "cumulative": generated_text.clone()
                            }).to_string());

                        if tx.send(Ok(event)).await.is_err() {
                            break; // Client disconnected
                        }

                        // Check for EOS token (model-specific, typically token_id == 2)
                        if token_id == 2 {
                            break;
                        }
                    }
                    Err(e) => {
                        let error_event = Event::default()
                            .event("error")
                            .data(format!("Generation error: {}", e));
                        let _ = tx.send(Ok(error_event)).await;
                        break;
                    }
                }
            }

            // Get final statistics
            let stats = engine_guard.get_stats().await;

            // Send completion event with statistics
            let completion_event = Event::default()
                .event("complete")
                .data(serde_json::json!({
                    "total_tokens": stats.total_tokens_generated,
                    "total_time_ms": stats.total_generation_time_ms,
                    "tokens_per_second": 1000.0 / stats.average_time_per_token_ms,
                    "speedup_factor": stats.speedup_factor,
                    "final_text": generated_text
                }).to_string());

            let _ = tx.send(Ok(completion_event)).await;
        } else {
            // Fallback if no inference engine
            let fallback_event = Event::default()
                .event("error")
                .data("AI inference engine not initialized");
            let _ = tx.send(Ok(fallback_event)).await;
        }
    });

    Sse::new(ReceiverStream::new(rx))
}
```

**Add to Router** (`crates/q-api-server/src/main.rs`):
```rust
.route("/api/chat/:chat_id/stream", get(q_api_server::chat_api::stream_message))
```

### Task 2: Add generate_next_token() to DistributedInferenceWithCache

**Location**: `crates/q-ai-inference/src/distributed_cache.rs`

**New Method**:
```rust
impl DistributedInferenceWithCache {
    /// Generate a single next token (for SSE streaming)
    ///
    /// # Arguments
    /// * `prompt` - Input text (with Mistral instruction formatting)
    /// * `position` - Current token position (0 = first token after prompt)
    ///
    /// # Returns
    /// * `Ok((token_id, token_text))` - Next generated token
    /// * `Err` - Generation error
    pub async fn generate_next_token(
        &mut self,
        prompt: &str,
        position: usize,
    ) -> Result<(u32, String)> {
        // On first call (position == 0), encode prompt and process through model
        if position == 0 {
            // Reset cache for new sequence
            self.kv_cache.clear();

            // Encode prompt
            let tokens = self.tokenizer
                .encode(prompt, true)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

            let input_ids = Tensor::new(tokens.get_ids(), &self.device)?
                .unsqueeze(0)?; // Add batch dimension

            // Initial forward pass through all prompt tokens
            let logits = self.model.forward(&input_ids, &mut self.kv_cache)?;

            // Get last logit for next token prediction
            let next_logits = logits.i((0, tokens.get_ids().len() - 1, ..))?;

            // Sample next token
            let next_token_id = self.sampler.sample(&next_logits)?;

            // Decode token to text
            let token_text = self.tokenizer
                .decode(&[next_token_id], false)
                .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;

            // Update statistics
            self.update_stats(position, std::time::Instant::now());

            Ok((next_token_id, token_text))
        } else {
            // Subsequent tokens: use cached KV states (14.27x speedup!)
            let prev_token_id = self.last_token_id
                .ok_or_else(|| anyhow::anyhow!("No previous token"))?;

            let input_ids = Tensor::new(&[prev_token_id], &self.device)?
                .unsqueeze(0)?;

            // Forward pass with KV-cache (FAST!)
            let logits = self.model.forward(&input_ids, &mut self.kv_cache)?;
            let next_logits = logits.i((0, 0, ..))?; // Single token output

            // Sample next token
            let next_token_id = self.sampler.sample(&next_logits)?;

            // Decode token to text
            let token_text = self.tokenizer
                .decode(&[next_token_id], false)
                .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;

            // Update statistics
            self.update_stats(position, std::time::Instant::now());

            // Store for next iteration
            self.last_token_id = Some(next_token_id);

            Ok((next_token_id, token_text))
        }
    }

    // Helper method to update statistics
    fn update_stats(&mut self, position: usize, start_time: std::time::Instant) {
        let elapsed_ms = start_time.elapsed().as_millis() as f64;
        self.stats.total_tokens_generated = position as u64 + 1;
        self.stats.total_generation_time_ms += elapsed_ms;
        self.stats.average_time_per_token_ms =
            self.stats.total_generation_time_ms / (position as f64 + 1.0);
    }
}
```

**Add State Field**:
```rust
pub struct DistributedInferenceWithCache {
    // ... existing fields
    last_token_id: Option<u32>, // Track last generated token for streaming
}
```

### Task 3: Frontend Integration (JavaScript/TypeScript)

**Location**: `gui/quantum-wallet/src/services/api.ts`

**New Function**:
```typescript
export async function streamChatMessage(
    chatId: string,
    content: string,
    onToken: (token: string, cumulative: string) => void,
    onComplete: (stats: any) => void,
    onError: (error: string) => void
): Promise<void> {
    const url = `${API_BASE}/api/chat/${chatId}/stream?content=${encodeURIComponent(content)}`;

    const eventSource = new EventSource(url);

    eventSource.addEventListener('start', (event) => {
        console.log('[SSE] Generation started');
    });

    eventSource.addEventListener('token', (event) => {
        const data = JSON.parse(event.data);
        onToken(data.token, data.cumulative);
    });

    eventSource.addEventListener('complete', (event) => {
        const data = JSON.parse(event.data);
        onComplete(data);
        eventSource.close();
    });

    eventSource.addEventListener('error', (event) => {
        const error = event.data || 'Unknown error';
        onError(error);
        eventSource.close();
    });

    eventSource.onerror = (err) => {
        console.error('[SSE] Connection error:', err);
        eventSource.close();
        onError('Connection failed');
    };
}
```

**React Component Example**:
```tsx
import React, { useState } from 'react';
import { streamChatMessage } from '../services/api';

export const ChatInterface: React.FC = () => {
    const [messages, setMessages] = useState<string[]>([]);
    const [currentResponse, setCurrentResponse] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);

    const handleSendMessage = async (userMessage: string) => {
        setMessages(prev => [...prev, `User: ${userMessage}`]);
        setIsStreaming(true);
        setCurrentResponse('');

        await streamChatMessage(
            'chat-session-1',
            userMessage,
            (token, cumulative) => {
                // Update UI with each new token
                setCurrentResponse(cumulative);
            },
            (stats) => {
                // Generation complete
                setMessages(prev => [...prev, `AI: ${currentResponse}`]);
                setIsStreaming(false);
                console.log('Stats:', stats);
            },
            (error) => {
                // Error occurred
                console.error('Stream error:', error);
                setIsStreaming(false);
            }
        );
    };

    return (
        <div>
            {messages.map((msg, idx) => (
                <div key={idx}>{msg}</div>
            ))}
            {isStreaming && (
                <div>AI: {currentResponse}<span className="cursor">▊</span></div>
            )}
        </div>
    );
};
```

---

## 📊 Expected Performance

### Without Streaming (Current):
```
User sends message
  ↓
Wait 325 seconds (50 tokens × 6.5s)
  ↓
Entire response appears at once
```

### With Streaming (Phase 6):
```
User sends message
  ↓
First token appears after 94s
  ↓
Token 2 appears after +6.5s (100.5s total)
  ↓
Token 3 appears after +6.5s (107s total)
  ↓
... tokens stream continuously ...
  ↓
Final token (50) appears after 419s total
```

**User Experience Improvement**:
- ✅ Immediate feedback (first token in 94s vs 325s wait)
- ✅ Visual progress indicator (tokens appearing)
- ✅ Can cancel early if response is sufficient
- ✅ Feels more interactive and responsive

---

## 🔧 Implementation Checklist

### Backend (Rust):
- [ ] Add `generate_next_token()` method to `DistributedInferenceWithCache`
- [ ] Add `last_token_id` state field
- [ ] Implement SSE `stream_message()` endpoint in `chat_api.rs`
- [ ] Add SSE route to router in `main.rs`
- [ ] Add EOS token detection (token_id == 2 for Mistral)
- [ ] Add proper error handling for client disconnection
- [ ] Test SSE endpoint with curl:
  ```bash
  curl -N http://localhost:8080/api/chat/test/stream?content=Hello
  ```

### Frontend (TypeScript/React):
- [ ] Implement `streamChatMessage()` in `api.ts`
- [ ] Create `ChatInterface.tsx` component with SSE support
- [ ] Add loading indicator with streaming animation
- [ ] Add cancel/stop button for long generations
- [ ] Display token statistics (tokens/sec, total time)
- [ ] Add error handling and retry logic

### Testing:
- [ ] Test SSE connection stability over long generations
- [ ] Test client disconnection handling
- [ ] Test concurrent streams (multiple users)
- [ ] Verify KV-cache still provides 14.27x speedup
- [ ] Load test with 10+ simultaneous streams

---

## 🚀 Deployment Strategy

### Phase 6.1: Backend SSE Implementation
1. Add `generate_next_token()` to `q-ai-inference`
2. Add SSE endpoint to `q-api-server`
3. Test with curl and manual SSE clients

### Phase 6.2: Frontend Integration
4. Implement TypeScript SSE client
5. Create React streaming chat UI
6. End-to-end testing

### Phase 6.3: Production Deployment
7. Deploy to bootstrap node
8. Monitor SSE connection stability
9. Optimize for mobile/slow connections

---

## 🔍 Technical Considerations

### SSE vs WebSocket

**Why SSE (Server-Sent Events)?**
- ✅ **Simpler**: Unidirectional (server → client only)
- ✅ **HTTP-based**: Works through proxies/firewalls
- ✅ **Auto-reconnection**: Built-in browser support
- ✅ **EventSource API**: Native browser support

**When to use WebSocket?**
- Need bidirectional communication
- Need to send data from client during streaming
- Lower latency required (<10ms)

**Decision**: SSE is perfect for AI token streaming (server → client only)

### Connection Management

**Challenge**: Long-running SSE connections (5+ minutes)

**Solution**:
1. **Heartbeat events**: Send every 30s to prevent timeout
2. **Client reconnection**: Auto-reconnect on disconnect
3. **Server timeout**: Kill idle streams after 10 minutes
4. **Graceful shutdown**: Send "complete" event before closing

### Concurrency Limits

**Challenge**: Multiple users streaming simultaneously

**Solution**:
1. **Per-user rate limiting**: Max 1 concurrent stream per user
2. **Global concurrency limit**: Max 10 simultaneous streams
3. **Queue system**: Queue excess requests
4. **Priority**: Paid users > free users

---

## 📚 References

- [SSE Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [Axum SSE Documentation](https://docs.rs/axum/latest/axum/response/sse/index.html)
- [EventSource MDN](https://developer.mozilla.org/en-US/docs/Web/API/EventSource)

---

**Status**: Planning complete, ready for implementation after v0.1.2-beta deployment
**Next Phase**: Phase 7 - P2P Distributed Inference
