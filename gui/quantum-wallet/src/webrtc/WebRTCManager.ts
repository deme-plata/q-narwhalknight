// WebRTC peer connection manager — browser-native RTCPeerConnection
// Protocol mirrors nova-chat/src/media/webrtc.rs
// WebRTC is used for direct peer audio/video only — NOT as libp2p transport,
// which stays Tor-only to prevent IP leaks.
//
// PRIVACY: iceTransportPolicy is set to 'relay', forcing all media through the
// q-turn STUN/TURN server. No peer real IP addresses appear in ICE candidates —
// only the TURN relay address (quillon.xyz) is exchanged. Zero IP leak.

import { getConnectionInfo } from '../services/api';

export type CallType = 'audio' | 'video';

export type ConnectionState = 'new' | 'connecting' | 'connected' | 'disconnected' | 'failed';

export interface WebRTCCallbacks {
  onStateChange: (peerId: string, state: ConnectionState) => void;
  onRemoteStream: (peerId: string, stream: MediaStream) => void;
  onIceCandidate: (peerId: string, candidate: RTCIceCandidate) => void;
  onLocalOffer: (peerId: string, sdp: string, callType: CallType) => void;
  onLocalAnswer: (peerId: string, sdp: string) => void;
}

interface TurnCredentials {
  username: string;
  password: string;
  ttl:      number;
  uris:     string[];
}

interface PeerCall {
  pc: RTCPeerConnection;
  state: ConnectionState;
  callType: CallType;
  localStream: MediaStream | null;
}

// Cached TURN credentials — refreshed before expiry
let turnCredsCache: { creds: TurnCredentials; fetchedAt: number } | null = null;

async function fetchTurnCredentials(): Promise<RTCIceServer[]> {
  const now = Date.now() / 1000;
  if (turnCredsCache && now - turnCredsCache.fetchedAt < turnCredsCache.creds.ttl - 60) {
    const c = turnCredsCache.creds;
    return [{ urls: c.uris, username: c.username, credential: c.password }];
  }

  try {
    const { apiBaseUrl } = getConnectionInfo();
    const res = await fetch(`${apiBaseUrl}/turn/credentials`, {
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
    });
    if (!res.ok) throw new Error(`TURN credential fetch: ${res.status}`);
    const creds: TurnCredentials = await res.json();
    turnCredsCache = { creds, fetchedAt: now };
    return [{ urls: creds.uris, username: creds.username, credential: creds.password }];
  } catch {
    // If credential fetch fails, fall back to STUN-only (still better than Google).
    // NOTE: with relay-only policy this means call setup will fail — expected.
    return [];
  }
}

export class WebRTCManager {
  private calls = new Map<string, PeerCall>();
  private callbacks: WebRTCCallbacks;

  constructor(callbacks: WebRTCCallbacks) {
    this.callbacks = callbacks;
  }

  private async buildPeerConnection(peerId: string): Promise<RTCPeerConnection> {
    const iceServers = await fetchTurnCredentials();

    const pc = new RTCPeerConnection({
      iceServers,
      // Force relay-only: no host or server-reflexive candidates.
      // Both peers see only the TURN relay address — zero real IP exposure.
      iceTransportPolicy: 'relay',
    });

    pc.onicecandidate = (e) => {
      if (e.candidate) {
        this.callbacks.onIceCandidate(peerId, e.candidate);
      }
    };

    pc.ontrack = (e) => {
      const [stream] = e.streams;
      if (stream) this.callbacks.onRemoteStream(peerId, stream);
    };

    pc.onconnectionstatechange = () => {
      const stateMap: Record<string, ConnectionState> = {
        new: 'new',
        connecting: 'connecting',
        connected: 'connected',
        disconnected: 'disconnected',
        failed: 'failed',
        closed: 'disconnected',
      };
      const mapped = stateMap[pc.connectionState] ?? 'new';
      const call = this.calls.get(peerId);
      if (call) call.state = mapped;
      this.callbacks.onStateChange(peerId, mapped);
    };

    return pc;
  }

  private async getLocalStream(callType: CallType): Promise<MediaStream> {
    const constraints: MediaStreamConstraints = {
      audio: true,
      video: callType === 'video' ? { width: 1280, height: 720, facingMode: 'user' } : false,
    };
    return navigator.mediaDevices.getUserMedia(constraints);
  }

  async initiateCall(peerId: string, callType: CallType): Promise<void> {
    this.hangup(peerId);

    // getUserMedia before createPeerConnection — avoids dangling PC on permission denial (MED-4)
    const localStream = await this.getLocalStream(callType);
    const pc = await this.buildPeerConnection(peerId);
    localStream.getTracks().forEach((t) => pc.addTrack(t, localStream));

    this.calls.set(peerId, { pc, state: 'new', callType, localStream });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    this.callbacks.onLocalOffer(peerId, offer.sdp!, callType);
  }

  async handleOffer(peerId: string, sdp: string, callType: CallType): Promise<void> {
    this.hangup(peerId);

    // getUserMedia before createPeerConnection (MED-4)
    const localStream = await this.getLocalStream(callType);
    const pc = await this.buildPeerConnection(peerId);
    localStream.getTracks().forEach((t) => pc.addTrack(t, localStream));

    this.calls.set(peerId, { pc, state: 'connecting', callType, localStream });

    await pc.setRemoteDescription({ type: 'offer', sdp });
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    this.callbacks.onLocalAnswer(peerId, answer.sdp!);
  }

  async handleAnswer(peerId: string, sdp: string): Promise<void> {
    const call = this.calls.get(peerId);
    if (!call) return;
    await call.pc.setRemoteDescription({ type: 'answer', sdp });
  }

  async handleIceCandidate(
    peerId: string,
    candidate: string,
    sdpMid: string | null,
    sdpMLineIndex: number | null,
  ): Promise<void> {
    const call = this.calls.get(peerId);
    if (!call) return;
    await call.pc.addIceCandidate({
      candidate,
      sdpMid: sdpMid ?? undefined,
      sdpMLineIndex: sdpMLineIndex ?? undefined,
    });
  }

  hangup(peerId: string): void {
    const call = this.calls.get(peerId);
    if (!call) return;
    call.localStream?.getTracks().forEach((t) => t.stop());
    call.pc.close();
    this.calls.delete(peerId);
    this.callbacks.onStateChange(peerId, 'disconnected');
  }

  hangupAll(): void {
    for (const peerId of this.calls.keys()) this.hangup(peerId);
  }

  getState(peerId: string): ConnectionState {
    return this.calls.get(peerId)?.state ?? 'new';
  }

  getLocalStream(peerId: string): MediaStream | null {
    return this.calls.get(peerId)?.localStream ?? null;
  }
}
