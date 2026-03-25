/**
 * SSE (Server-Sent Events) manager for real-time blockchain updates.
 *
 * Handles reconnection with exponential backoff and event parsing.
 * React Native doesn't have native EventSource, so we use fetch streaming.
 */

type SSEListener = (event: SSEEvent) => void;

export interface SSEEvent {
  type: string;
  data: unknown;
  timestamp: number;
}

interface SSEManagerState {
  connected: boolean;
  reconnecting: boolean;
  lastEventTime: number;
  reconnectAttempt: number;
}

const SSE_BASE_URL = 'https://quillon.xyz/api/v1/events';
const INITIAL_RECONNECT_DELAY_MS = 1_000;
const MAX_RECONNECT_DELAY_MS = 30_000;
const MAX_RECONNECT_ATTEMPTS = 20;
const HEARTBEAT_TIMEOUT_MS = 45_000;
/** Max SSE buffer size before forced flush to prevent memory accumulation */
const MAX_BUFFER_SIZE = 64 * 1024; // 64 KB

class SSEManager {
  private abortController: AbortController | null = null;
  private listeners: Map<string, Set<SSEListener>> = new Map();
  private wildcardListeners: Set<SSEListener> = new Set();
  private state: SSEManagerState = {
    connected: false,
    reconnecting: false,
    lastEventTime: 0,
    reconnectAttempt: 0,
  };
  private heartbeatTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private address: string | null = null;

  /**
   * Connect to the SSE stream for a given wallet address.
   */
  connect(address: string): void {
    if (this.abortController) {
      this.disconnect();
    }

    this.address = address;
    this.state.reconnectAttempt = 0;
    this.startConnection();
  }

  /**
   * Disconnect from the SSE stream.
   */
  disconnect(): void {
    this.abortController?.abort();
    this.abortController = null;
    this.state.connected = false;
    this.state.reconnecting = false;
    this.clearTimers();
    this.emit({ type: 'connection', data: { status: 'disconnected' }, timestamp: Date.now() });
  }

  /**
   * Subscribe to a specific event type (e.g. "balance", "block", "tx").
   * Use "*" or omit type to receive all events.
   */
  on(type: string, listener: SSEListener): () => void {
    if (type === '*') {
      this.wildcardListeners.add(listener);
      return () => this.wildcardListeners.delete(listener);
    }

    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(listener);
    return () => this.listeners.get(type)?.delete(listener);
  }

  /**
   * Get the current connection state.
   */
  getState(): Readonly<SSEManagerState> {
    return { ...this.state };
  }

  // ---------- Internal ----------

  private async startConnection(): Promise<void> {
    if (!this.address) return;

    this.abortController = new AbortController();
    const url = `${SSE_BASE_URL}?address=${this.address}`;

    try {
      this.emit({
        type: 'connection',
        data: { status: 'connecting' },
        timestamp: Date.now(),
      });

      const response = await fetch(url, {
        headers: { Accept: 'text/event-stream', 'Cache-Control': 'no-cache' },
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`SSE connect failed: ${response.status}`);
      }

      this.state.connected = true;
      this.state.reconnecting = false;
      this.state.reconnectAttempt = 0;
      this.resetHeartbeat();

      this.emit({
        type: 'connection',
        data: { status: 'connected' },
        timestamp: Date.now(),
      });

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No readable stream');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Prevent unbounded buffer growth (peer review: memory accumulation risk)
        if (buffer.length > MAX_BUFFER_SIZE) {
          console.warn('[SSE] Buffer exceeded max size, flushing');
          buffer = '';
          continue;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        let currentEventType = 'message';
        let currentData = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData += line.slice(6);
          } else if (line === '') {
            // End of event
            if (currentData) {
              this.resetHeartbeat();
              try {
                const parsed = JSON.parse(currentData);
                this.emit({
                  type: currentEventType,
                  data: parsed,
                  timestamp: Date.now(),
                });
              } catch {
                this.emit({
                  type: currentEventType,
                  data: currentData,
                  timestamp: Date.now(),
                });
              }
            }
            currentEventType = 'message';
            currentData = '';
          }
        }
      }
    } catch (error: unknown) {
      if (error instanceof Error && error.name === 'AbortError') {
        return; // Intentional disconnect
      }
      console.warn('[SSE] Connection error:', error);
    }

    this.state.connected = false;

    // Auto-reconnect unless explicitly disconnected
    if (this.abortController && !this.abortController.signal.aborted) {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.state.reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      console.error('[SSE] Max reconnect attempts reached, falling back to polling');
      this.emit({
        type: 'connection',
        data: { status: 'failed', reason: 'max_retries', fallback: 'polling' },
        timestamp: Date.now(),
      });
      // Emit a polling fallback event so useBalance hook switches to REST polling
      this.emit({
        type: 'polling_fallback',
        data: { active: true },
        timestamp: Date.now(),
      });
      return;
    }

    this.state.reconnecting = true;
    const delay = Math.min(
      INITIAL_RECONNECT_DELAY_MS * Math.pow(2, this.state.reconnectAttempt),
      MAX_RECONNECT_DELAY_MS
    );

    this.state.reconnectAttempt++;

    this.emit({
      type: 'connection',
      data: { status: 'reconnecting', attempt: this.state.reconnectAttempt, delay },
      timestamp: Date.now(),
    });

    this.reconnectTimer = setTimeout(() => {
      this.startConnection();
    }, delay);
  }

  private resetHeartbeat(): void {
    if (this.heartbeatTimer) clearTimeout(this.heartbeatTimer);
    this.state.lastEventTime = Date.now();
    this.heartbeatTimer = setTimeout(() => {
      console.warn('[SSE] Heartbeat timeout, reconnecting...');
      this.abortController?.abort();
      this.scheduleReconnect();
    }, HEARTBEAT_TIMEOUT_MS);
  }

  private clearTimers(): void {
    if (this.heartbeatTimer) clearTimeout(this.heartbeatTimer);
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.heartbeatTimer = null;
    this.reconnectTimer = null;
  }

  private emit(event: SSEEvent): void {
    // Notify type-specific listeners
    const typeListeners = this.listeners.get(event.type);
    if (typeListeners) {
      for (const listener of typeListeners) {
        try {
          listener(event);
        } catch (e) {
          console.error('[SSE] Listener error:', e);
        }
      }
    }

    // Notify wildcard listeners
    for (const listener of this.wildcardListeners) {
      try {
        listener(event);
      } catch (e) {
        console.error('[SSE] Wildcard listener error:', e);
      }
    }
  }
}

/** Singleton SSE manager instance */
export const sseManager = new SSEManager();
