/**
 * v8.4.4: Shared SSE Manager — single connection, multiple subscribers
 *
 * PROBLEMS FIXED:
 * 1. Every component (TopBar, Dashboard, MiningScreen, DexScreen, Explorer...)
 *    created its own EventSource — 5-8 concurrent SSE connections per tab
 * 2. App.tsx used fetch() instead of EventSource — no built-in auto-reconnect
 * 3. Exponential backoff never reset properly on stream end
 * 4. "Disconnected" in DeployControlPanel because events arrived on different
 *    EventSource instances than the one App.tsx dispatched events from
 *
 * SOLUTION: Single shared SSE connection with pub/sub fan-out.
 * Components subscribe via sseManager.on('event-type', callback).
 * Reconnection uses jittered backoff with fast initial retry.
 */

type SSECallback = (data: any) => void;

interface SSEManagerState {
  status: 'disconnected' | 'connecting' | 'connected' | 'reconnecting';
  eventSource: EventSource | null;
  listeners: Map<string, Set<SSECallback>>;
  wildcardListeners: Set<SSECallback>;
  reconnectTimer: ReturnType<typeof setTimeout> | null;
  reconnectAttempts: number;
  lastEventTime: number;
  walletAddress: string;
  healthCheckInterval: ReturnType<typeof setInterval> | null;
}

const state: SSEManagerState = {
  status: 'disconnected',
  eventSource: null,
  listeners: new Map(),
  wildcardListeners: new Set(),
  reconnectTimer: null,
  reconnectAttempts: 0,
  lastEventTime: 0,
  walletAddress: '',
  healthCheckInterval: null,
};

// All SSE event types the backend can send
const SSE_EVENT_TYPES = [
  'mining_reward',
  'balance-updated',
  'node-status',
  'miner-stats',
  'mining_stats',
  'new-block',
  'transaction-submitted',
  'transaction-confirmed',
  'transaction-status',
  'swap_executed',
  'token_price_update',
  'pool-stats-updated',
  'dex-pool-update',
  'contract-deployed',
  'contract-event',
  // v8.5.5: Email events — were MISSING, causing stale unread badges
  'email-received',
  'email-sent',
  'email-unread-count',
  'calendar-reminder',
  'calendar-event-created',
  'scheduled-tx-executed',
  'xlist-campaign-update',
  'verification-update',
  // v8.6.2: Missing event types — these were silently never dispatched
  'token-balance-updated',
  'loan-approved',
  'mining_stats',
  'server-version',
];

function getReconnectDelay(): number {
  // Fast first retry (500ms), then 1s, 2s, 4s, 8s, cap at 15s
  // Add jitter to prevent thundering herd
  const base = state.reconnectAttempts === 0 ? 500 : 1000;
  const delay = Math.min(base * Math.pow(2, Math.max(0, state.reconnectAttempts - 1)), 15000);
  const jitter = Math.random() * Math.min(delay * 0.3, 2000);
  return delay + jitter;
}

function dispatchStatusEvent(status: string) {
  window.dispatchEvent(new Event(`sse-${status}`));
  // Also update localStorage for DeployControlPanel's polling check
  if (status === 'connected') {
    localStorage.setItem('lastBlockTime', Date.now().toString());
  }
}

function handleEvent(type: string, rawData: string) {
  state.lastEventTime = Date.now();
  // Keep lastBlockTime fresh for DeployControlPanel
  localStorage.setItem('lastBlockTime', Date.now().toString());

  try {
    const parsed = JSON.parse(rawData);

    // Fan out to type-specific listeners
    const typeListeners = state.listeners.get(type);
    if (typeListeners) {
      for (const cb of typeListeners) {
        try { cb(parsed); } catch (e) { console.error(`SSE listener error [${type}]:`, e); }
      }
    }

    // Fan out to wildcard listeners
    for (const cb of state.wildcardListeners) {
      try { cb({ type, data: parsed }); } catch (e) { console.error('SSE wildcard listener error:', e); }
    }
  } catch (e) {
    console.error(`SSE parse error [${type}]:`, e);
  }
}

function connect() {
  // Don't reconnect if already connected or connecting
  if (state.eventSource && state.eventSource.readyState !== EventSource.CLOSED) {
    return;
  }

  const walletAddress = state.walletAddress || localStorage.getItem('walletAddress') || '';
  if (!walletAddress) {
    console.log('[SSE Manager] No wallet address, deferring connection');
    return;
  }

  state.status = state.reconnectAttempts > 0 ? 'reconnecting' : 'connecting';
  dispatchStatusEvent(state.status);

  const baseUrl = localStorage.getItem('nodeUrl') || '';
  const url = `${baseUrl}/api/v1/events?wallet_address=${encodeURIComponent(walletAddress)}`;

  console.log(`[SSE Manager] Connecting (attempt ${state.reconnectAttempts})...`);

  const es = new EventSource(url);
  state.eventSource = es;

  es.onopen = () => {
    console.log('[SSE Manager] Connected');
    state.status = 'connected';
    state.reconnectAttempts = 0;
    dispatchStatusEvent('connected');
  };

  // Register listeners for all known event types
  for (const eventType of SSE_EVENT_TYPES) {
    es.addEventListener(eventType, (e: MessageEvent) => {
      handleEvent(eventType, e.data);
    });
  }

  // Also handle unnamed "message" events
  es.onmessage = (e: MessageEvent) => {
    handleEvent('message', e.data);
  };

  es.onerror = () => {
    // EventSource fires onerror on disconnect AND on connection failure
    // readyState: 0=CONNECTING, 1=OPEN, 2=CLOSED
    if (es.readyState === EventSource.CLOSED) {
      console.warn('[SSE Manager] Connection closed, scheduling reconnect...');
      cleanup(false); // Don't clear listeners
      scheduleReconnect();
    } else {
      // CONNECTING state — EventSource is auto-retrying
      console.warn('[SSE Manager] Connection error (auto-retrying)...');
      state.status = 'reconnecting';
      dispatchStatusEvent('reconnecting');
    }
  };
}

function cleanup(clearListeners: boolean) {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  if (state.reconnectTimer) {
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
  }
  state.status = 'disconnected';
  dispatchStatusEvent('disconnected');

  if (clearListeners) {
    state.listeners.clear();
    state.wildcardListeners.clear();
    if (state.healthCheckInterval) {
      clearInterval(state.healthCheckInterval);
      state.healthCheckInterval = null;
    }
  }
}

function scheduleReconnect() {
  if (state.reconnectTimer) return;
  const delay = getReconnectDelay();
  state.reconnectAttempts++;
  console.log(`[SSE Manager] Reconnecting in ${(delay/1000).toFixed(1)}s (attempt ${state.reconnectAttempts})`);
  state.reconnectTimer = setTimeout(() => {
    state.reconnectTimer = null;
    connect();
  }, delay);
}

// Health check: if no events for 3 minutes while "connected", force reconnect
// 60s was too aggressive — server sends SSE comments as heartbeats (not counted as events),
// so in low-activity periods this would reconnect unnecessarily and disrupt state.
function startHealthCheck() {
  if (state.healthCheckInterval) return;
  state.healthCheckInterval = setInterval(() => {
    if (state.status === 'connected' && state.lastEventTime > 0) {
      const staleMs = Date.now() - state.lastEventTime;
      if (staleMs > 180000) {
        console.warn(`[SSE Manager] No events for ${(staleMs/1000).toFixed(0)}s, forcing reconnect`);
        cleanup(false);
        state.reconnectAttempts = 0; // Reset — this is a health issue, not a connection failure
        connect();
      }
    }
  }, 15000);
}

// Visibility handler: reconnect immediately when tab becomes visible
function handleVisibilityChange() {
  if (document.visibilityState === 'visible') {
    if (state.status !== 'connected' || !state.eventSource || state.eventSource.readyState !== EventSource.OPEN) {
      console.log('[SSE Manager] Tab visible, reconnecting...');
      cleanup(false);
      state.reconnectAttempts = 0;
      connect();
    }
  }
}

// ============= PUBLIC API =============

export const sseManager = {
  /** Start SSE connection for a wallet address */
  start(walletAddress?: string) {
    if (walletAddress) state.walletAddress = walletAddress;
    document.addEventListener('visibilitychange', handleVisibilityChange);
    startHealthCheck();
    connect();
  },

  /** Stop SSE and clean up everything */
  stop() {
    document.removeEventListener('visibilitychange', handleVisibilityChange);
    cleanup(true);
  },

  /** Subscribe to a specific SSE event type. Returns unsubscribe function. */
  on(eventType: string, callback: SSECallback): () => void {
    if (!state.listeners.has(eventType)) {
      state.listeners.set(eventType, new Set());
    }
    state.listeners.get(eventType)!.add(callback);
    return () => {
      state.listeners.get(eventType)?.delete(callback);
    };
  },

  /** Subscribe to ALL SSE events. Callback receives { type, data }. */
  onAny(callback: SSECallback): () => void {
    state.wildcardListeners.add(callback);
    return () => { state.wildcardListeners.delete(callback); };
  },

  /** Get current connection status */
  getStatus(): string {
    return state.status;
  },

  /** Force immediate reconnect (e.g. after login or server switch) */
  reconnect() {
    cleanup(false);
    state.reconnectAttempts = 0;
    connect();
  },

  /** Update wallet address and reconnect */
  setWallet(walletAddress: string) {
    if (state.walletAddress === walletAddress && state.status === 'connected') return;
    state.walletAddress = walletAddress;
    cleanup(false);
    state.reconnectAttempts = 0;
    connect();
  },
};

export default sseManager;
