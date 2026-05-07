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

function httpToWs(url: string): string {
  return url.replace(/^https:/, 'wss:').replace(/^http:/, 'ws:');
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

  /**
   * @param peerId - wallet address (qnk...)
   * @param getPrivateKey - callback returning the Ed25519 private key for CRIT-1 auth.
   *   If null, connection proceeds without auth (for dev/localhost only).
   */
  constructor(peerId: string, getPrivateKey: (() => Uint8Array | null) | null = null) {
    this.peerId = peerId;
    this.getPrivateKey = getPrivateKey;
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

    // Generate Ed25519 auth header signed over /ws/chat/signal (CRIT-1 fix)
    let authParam = '';
    if (this.getPrivateKey) {
      const privateKey = this.getPrivateKey();
      if (privateKey) {
        try {
          const authJson = await generateAuthHeader(privateKey, this.peerId, '/ws/chat/signal');
          authParam = `&auth_header=${encodeURIComponent(authJson)}`;
        } catch {
          // Auth header generation failed — server will reject the connection
        }
      }
    }

    const url = `${wsBase}/ws/chat/signal?peer_id=${encodeURIComponent(this.peerId)}${authParam}`;

    try {
      this.ws = new WebSocket(url);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.startPing();
    };

    this.ws.onmessage = (e) => {
      try {
        const envelope: SignalingEnvelope = JSON.parse(e.data);
        this.handlers.forEach((h) => h(envelope));
      } catch {
        // malformed message — ignore
      }
    };

    this.ws.onerror = () => {
      // onclose fires after onerror, which handles reconnect
    };

    this.ws.onclose = () => {
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
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const envelope: SignalingEnvelope = {
      from: this.peerId,
      to,
      session_id: sessionId,
      payload,
    };
    this.ws.send(JSON.stringify(envelope));
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
