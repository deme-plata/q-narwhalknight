// WebSocket client for the q-api-server signaling endpoint
// Connects to /ws/chat/signal?peer_id=<wallet_address>&auth_header=<signed_json>
// Protocol defined in crates/q-api-server/src/signaling_server.rs

import { getConnectionInfo } from './api';
import { generateAuthHeader } from './walletAuth';

export type CallType = 'audio' | 'video';

export interface PeerInfo {
  peer_id: string;
  display_name: string;
}

export type SignalingPayload =
  | { type: 'call_offer'; sdp: string; call_type: CallType }
  | { type: 'call_answer'; sdp: string }
  | { type: 'ice_candidate'; candidate: string; sdp_mid: string | null; sdp_m_line_index: number | null }
  | { type: 'call_end'; reason?: string }
  | { type: 'chat_message'; content: string; timestamp: number }
  | { type: 'meeting_join'; room_id: string; display_name: string }
  | { type: 'meeting_leave'; room_id: string }
  | { type: 'meeting_peers'; room_id: string; peers: PeerInfo[] }
  | { type: 'meeting_peer_joined'; room_id: string; peer: PeerInfo }
  | { type: 'meeting_peer_left'; room_id: string; peer_id: string }
  | { type: 'ping' }
  | { type: 'pong' };

export interface SignalingEnvelope {
  from: string;
  to: string | null;
  session_id: string | null;
  payload: SignalingPayload;
}

export type SignalingHandler = (envelope: SignalingEnvelope) => void;

// ── Diagnostics ──────────────────────────────────────────────────────────────
// Exposed to UI so users can see exactly why a call isn't going through:
// "is my WebSocket open?", "did auth fail?", "what did the server say?".

export type WsState = 'idle' | 'connecting' | 'open' | 'closed' | 'error';
export type AuthStatus =
  | 'idle'
  | 'no-callback'      // SignalingService constructed without a getPrivateKey
  | 'no-private-key'  // callback returned null (wallet not unlocked / no session)
  | 'sign-failed'     // generateAuthHeader threw
  | 'ok';

export interface SignalingDiag {
  peerId: string;
  apiBaseUrl: string;
  wsUrl: string;
  wsState: WsState;
  authStatus: AuthStatus;
  authError: string | null;
  lastCloseCode: number | null;
  lastCloseReason: string | null;
  reconnectAttempts: number;
  lastOutgoing: { type: string; to: string | null; sid: string | null; ts: number } | null;
  lastIncoming: { type: string; from: string; reason?: string; ts: number } | null;
  serverPeerCount: number | null;     // populated by pingDiag() if user calls it
}

export type DiagListener = (diag: SignalingDiag) => void;

function httpToWs(url: string): string {
  return url.replace(/^https:/, 'wss:').replace(/^http:/, 'ws:');
}

/** Human-readable hint for browser WS close codes when the server didn't send a reason. */
function closeCodeMeaning(code: number): string {
  switch (code) {
    case 1000: return 'normal closure';
    case 1001: return 'going away';
    case 1006: return 'abnormal closure (no close frame — network / TLS / proxy reset)';
    case 1008: return 'policy violation';
    case 1011: return 'server error';
    case 1015: return 'TLS handshake failure';
    case 4001: return 'auth rejected (custom)';
    default: return `unspecified (code ${code})`;
  }
}

export class SignalingService {
  private ws: WebSocket | null = null;
  private peerId: string;
  private getPrivateKey: (() => Uint8Array | null) | null;
  private handlers: SignalingHandler[] = [];
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private closed = false;
  private reconnectAttempts = 0;

  // ── Diagnostic state ──
  private diag: SignalingDiag;
  private diagListeners: DiagListener[] = [];

  /**
   * @param peerId - wallet address (qnk...)
   * @param getPrivateKey - callback returning the Ed25519 private key for CRIT-1 auth.
   *   If null, connection proceeds without auth (for dev/localhost only).
   */
  constructor(peerId: string, getPrivateKey: (() => Uint8Array | null) | null = null) {
    this.peerId = peerId;
    this.getPrivateKey = getPrivateKey;
    this.diag = {
      peerId,
      apiBaseUrl: '',
      wsUrl: '',
      wsState: 'idle',
      authStatus: getPrivateKey ? 'idle' : 'no-callback',
      authError: null,
      lastCloseCode: null,
      lastCloseReason: null,
      reconnectAttempts: 0,
      lastOutgoing: null,
      lastIncoming: null,
      serverPeerCount: null,
    };
  }

  /** Snapshot the current diagnostic state. */
  getDiag(): SignalingDiag {
    return { ...this.diag };
  }

  /** Subscribe to diagnostic updates. Returns an unsubscribe fn. */
  onDiag(listener: DiagListener): () => void {
    this.diagListeners.push(listener);
    // Push current state immediately so UI initializes
    try { listener({ ...this.diag }); } catch { /* listener threw — ignore */ }
    return () => {
      this.diagListeners = this.diagListeners.filter((l) => l !== listener);
    };
  }

  private updateDiag(patch: Partial<SignalingDiag>): void {
    this.diag = { ...this.diag, ...patch };
    for (const l of this.diagListeners) {
      try { l({ ...this.diag }); } catch { /* listener threw — ignore */ }
    }
  }

  /**
   * Poll the server's diagnostic endpoint. Updates `serverPeerCount` in diag
   * and resolves with the parsed response. Lets the UI verify whether the
   * intended call target is actually connected to this backend.
   */
  async pollServerDiag(): Promise<{ peer_count: number; peers: string[]; room_count: number; active_call_count: number } | null> {
    try {
      const { apiBaseUrl } = getConnectionInfo();
      const url = `${apiBaseUrl}/v1/signaling/diag`;
      const res = await fetch(url, { credentials: 'include' });
      if (!res.ok) {
        console.warn('[SignalingService] diag poll failed:', res.status);
        return null;
      }
      const data = await res.json();
      this.updateDiag({ serverPeerCount: data.peer_count ?? null });
      return data;
    } catch (e) {
      console.warn('[SignalingService] diag poll error:', e);
      return null;
    }
  }

  connect(): void {
    this.closed = false;
    this.openSocket();
  }

  private async openSocket(): Promise<void> {
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
    }

    const apiBase = getConnectionInfo().apiBaseUrl.replace(/\/api$/, '');
    const wsBase = httpToWs(apiBase);
    this.updateDiag({ apiBaseUrl: apiBase, wsState: 'connecting' });

    // Generate Ed25519 auth header signed over /ws/chat/signal (CRIT-1 fix)
    let authParam = '';
    let authStatus: AuthStatus = 'no-callback';
    let authError: string | null = null;
    if (this.getPrivateKey) {
      const privateKey = this.getPrivateKey();
      if (privateKey) {
        try {
          const authJson = await generateAuthHeader(privateKey, this.peerId, '/ws/chat/signal');
          authParam = `&auth_header=${encodeURIComponent(authJson)}`;
          authStatus = 'ok';
        } catch (e: any) {
          authStatus = 'sign-failed';
          authError = e?.message ?? String(e);
          console.warn('[SignalingService] auth header generation FAILED:', e);
        }
      } else {
        authStatus = 'no-private-key';
        authError = 'getPrivateKey() returned null — wallet not unlocked or session missing';
        console.warn('[SignalingService] no private key available; server will reject connection');
      }
    } else {
      authError = 'SignalingService constructed without getPrivateKey callback';
      console.warn('[SignalingService] no auth callback configured');
    }
    this.updateDiag({ authStatus, authError });

    const url = `${wsBase}/ws/chat/signal?peer_id=${encodeURIComponent(this.peerId)}${authParam}`;
    this.updateDiag({ wsUrl: url.split('?')[0] }); // log path only, not the signed auth blob

    try {
      this.ws = new WebSocket(url);
    } catch (e: any) {
      console.warn('[SignalingService] WebSocket construct threw:', e);
      this.updateDiag({ wsState: 'error', lastCloseReason: e?.message ?? String(e) });
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      console.log('[SignalingService] WebSocket OPEN for peer_id:', this.peerId);
      this.reconnectAttempts = 0;
      this.updateDiag({ wsState: 'open', reconnectAttempts: 0, lastCloseCode: null, lastCloseReason: null });
      this.startPing();
    };

    this.ws.onmessage = (e) => {
      try {
        const envelope: SignalingEnvelope = JSON.parse(e.data);
        // Track last inbound for diagnostics
        const reason = (envelope.payload as any)?.reason as string | undefined;
        this.updateDiag({
          lastIncoming: { type: envelope.payload.type, from: envelope.from, reason, ts: Date.now() },
        });
        this.handlers.forEach((h) => h(envelope));
      } catch {
        // malformed message — ignore
      }
    };

    this.ws.onerror = (e) => {
      console.warn('[SignalingService] WebSocket error event:', e);
      this.updateDiag({ wsState: 'error' });
    };

    this.ws.onclose = (ev) => {
      console.warn('[SignalingService] WebSocket CLOSED — code:', ev.code, 'reason:', ev.reason || '(empty)', 'peer_id:', this.peerId);
      this.updateDiag({
        wsState: 'closed',
        lastCloseCode: ev.code,
        lastCloseReason: ev.reason || closeCodeMeaning(ev.code),
      });
      this.stopPing();
      if (!this.closed) this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    // Exponential backoff with jitter — prevents thundering herd on server restart (MED-3 fix)
    const base = Math.min(30000, 1000 * Math.pow(2, this.reconnectAttempts));
    const delay = base + Math.random() * 1000;
    this.reconnectAttempts++;
    this.updateDiag({ reconnectAttempts: this.reconnectAttempts });
    console.log('[SignalingService] reconnect scheduled in', Math.round(delay), 'ms (attempt', this.reconnectAttempts, ')');
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.closed) this.openSocket();
    }, delay);
  }

  private startPing(): void {
    this.stopPing();
    this.pingTimer = setInterval(() => {
      this.send(null, null, { type: 'ping' });
    }, 20000);
  }

  private stopPing(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  send(to: string | null, sessionId: string | null, payload: SignalingPayload): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn(
        '[SignalingService] send DROPPED — ws not open (readyState=%s) type=%s to=%s',
        this.ws?.readyState, payload.type, to,
      );
      return;
    }
    const envelope: SignalingEnvelope = {
      from: this.peerId,
      to,
      session_id: sessionId,
      payload,
    };
    this.ws.send(JSON.stringify(envelope));
    // Track outbound — useful when "I clicked Call and nothing happened" turns out
    // to be "the offer was sent but the server reported peer_not_found".
    this.updateDiag({
      lastOutgoing: { type: payload.type, to, sid: sessionId, ts: Date.now() },
    });
  }

  onMessage(handler: SignalingHandler): () => void {
    this.handlers.push(handler);
    return () => {
      this.handlers = this.handlers.filter((h) => h !== handler);
    };
  }

  disconnect(): void {
    this.closed = true;
    this.stopPing();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}
