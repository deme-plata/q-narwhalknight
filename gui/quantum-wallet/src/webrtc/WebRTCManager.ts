// WebRTC peer connection manager — browser-native RTCPeerConnection
// Protocol mirrors nova-chat/src/media/webrtc.rs
// WebRTC is used for direct peer audio/video only — NOT as libp2p transport,
// which stays Tor-only to prevent IP leaks.
//
// PRIVACY: iceTransportPolicy='relay' forces all media through q-turn.
// No real IP appears in ICE candidates — only quillon.xyz:3478.
//
// QUALITY: Targets 4K60 video with VP9/AV1, Opus stereo at max bitrate.
// Falls back gracefully if camera/browser can't reach those targets.

import { getConnectionInfo } from '../services/api';

export type CallType = 'audio' | 'video';
export type ConnectionState = 'new' | 'connecting' | 'connected' | 'disconnected' | 'failed';

export interface WebRTCCallbacks {
  onStateChange:   (peerId: string, state: ConnectionState) => void;
  onRemoteStream:  (peerId: string, stream: MediaStream) => void;
  onIceCandidate:  (peerId: string, candidate: RTCIceCandidate) => void;
  onLocalOffer:    (peerId: string, sdp: string, callType: CallType) => void;
  onLocalAnswer:   (peerId: string, sdp: string) => void;
}

interface TurnCredentials {
  username: string;
  password: string;
  ttl:      number;
  uris:     string[];
}

interface PeerCall {
  pc:          RTCPeerConnection;
  state:       ConnectionState;
  callType:    CallType;
  localStream: MediaStream | null;
}

// ─── TURN credential cache ────────────────────────────────────────────────────

let turnCredsCache: { creds: TurnCredentials; fetchedAt: number } | null = null;

// Build a single RTCIceServer that covers both UDP and TCP TURN transport.
// Tor Browser is TCP-only, so plain turn: URIs (UDP) will silently fail.
// Adding ?transport=tcp variants ensures TURN works through Tor when WebRTC
// is enabled (about:config → media.peerconnection.enabled = true).
function buildTurnIceServer(uris: string[], username: string, password: string): RTCIceServer {
  const expanded: string[] = [];
  for (const uri of uris) {
    expanded.push(uri);
    // If this is a plain turn: URI with no transport query, add TCP variant
    if (uri.startsWith('turn:') && !uri.includes('transport=')) {
      expanded.push(uri + '?transport=tcp');
    }
    // turns: (TLS) URIs default to TCP — no extra variant needed
  }
  return { urls: expanded, username, credential: password };
}

// ICE connection timeout: if the peer connection is still in 'connecting' state
// after this many ms, force it to 'failed'. Without this, Tor Browser users with
// WebRTC enabled but no working TURN path hang at "Connecting…" indefinitely.
const ICE_TIMEOUT_MS = 30_000;

// ─── Quality targets ──────────────────────────────────────────────────────────

// Video: ideal 4K60, no hard minimums — accepts any camera resolution.
const VIDEO_CONSTRAINTS: MediaTrackConstraints = {
  facingMode:  'user',
  width:       { ideal: 1920 },
  height:      { ideal: 1080 },
  frameRate:   { ideal: 30   },
};

// Audio: 48 kHz stereo with full WebRTC processing pipeline.
const AUDIO_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl:  true,
  sampleRate:       48000,
  channelCount:     2,       // stereo
};

// Target bitrates (bits/s)
const VIDEO_MAX_BITRATE = 8_000_000;   // 8 Mbps — covers 4K VP9 comfortably
const AUDIO_MAX_BITRATE =   510_000;   // 510 kbps — Opus maximum

// ─── Codec preference order ───────────────────────────────────────────────────
// AV1 > VP9 > H.264 for video (better quality per bit each step)
// Opus always wins audio in Chrome/Firefox so this just ensures order.
const VIDEO_CODEC_PRIORITY = ['av01', 'vp09', 'h264', 'vp8'];
const AUDIO_CODEC_PRIORITY = ['opus'];

function preferCodecs(codecs: RTCRtpCodec[], priority: string[]): RTCRtpCodec[] {
  const ranked: RTCRtpCodec[] = [];
  for (const mime of priority) {
    ranked.push(...codecs.filter(c => c.mimeType.toLowerCase().includes(mime)));
  }
  return [...ranked, ...codecs.filter(c => !ranked.includes(c))];
}

// ─── SDP bandwidth injection ──────────────────────────────────────────────────
// Injects b=AS (kbps) and b=TIAS (bps) lines into each m= section.
// This tells the remote end our receiving bandwidth limit, so it throttles
// its encoder rather than letting WebRTC auto-negotiate a low value.

function injectBandwidth(sdp: string, audioBps: number, videoBps: number): string {
  const lines = sdp.split('\r\n');
  const out: string[] = [];
  let section: 'audio' | 'video' | 'other' = 'other';

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('m=audio')) section = 'audio';
    else if (line.startsWith('m=video')) section = 'video';
    else if (line.startsWith('m=')) section = 'other';

    out.push(line);

    // Insert b= lines right after the m= line (before c= and rtcp attributes)
    if ((section === 'audio' || section === 'video') && line.startsWith('m=')) {
      const bps = section === 'audio' ? audioBps : videoBps;
      const kbps = Math.floor(bps / 1000);
      out.push(`b=AS:${kbps}`);
      out.push(`b=TIAS:${bps}`);
    }
  }

  return out.join('\r\n');
}

// Patch Opus fmtp line to enable stereo and max bitrate.
function patchOpusFmtp(sdp: string): string {
  // Find Opus payload type from rtpmap line
  const rtpmap = sdp.match(/a=rtpmap:(\d+) opus\/48000/i);
  if (!rtpmap) return sdp;
  const pt = rtpmap[1];

  // Replace or inject fmtp for this payload type
  const fmtpRe = new RegExp(`(a=fmtp:${pt} )(.*)`, 'i');
  const opusParams = 'minptime=10;useinbandfec=1;stereo=1;sprop-stereo=1;maxaveragebitrate=510000';

  if (fmtpRe.test(sdp)) {
    return sdp.replace(fmtpRe, `$1${opusParams}`);
  }
  // No existing fmtp — insert after the rtpmap line
  return sdp.replace(
    new RegExp(`(a=rtpmap:${pt} opus\\/48000[^\\r\\n]*)`, 'i'),
    `$1\r\na=fmtp:${pt} ${opusParams}`,
  );
}

function enhanceSdp(sdp: string, callType: CallType): string {
  let s = sdp;
  s = injectBandwidth(s, AUDIO_MAX_BITRATE, callType === 'video' ? VIDEO_MAX_BITRATE : AUDIO_MAX_BITRATE);
  s = patchOpusFmtp(s);
  return s;
}

// ─── Encoding parameter push (post-setLocalDescription) ──────────────────────
// RTCRtpSender.setParameters() is the definitive way to control bitrate at
// runtime; SDP b= lines are just the initial negotiation hint.

async function applyEncodingParams(pc: RTCPeerConnection, callType: CallType): Promise<void> {
  for (const sender of pc.getSenders()) {
    const kind = sender.track?.kind;
    if (!kind) continue;
    try {
      const params = sender.getParameters();
      if (!params.encodings || params.encodings.length === 0) {
        params.encodings = [{}];
      }
      for (const enc of params.encodings) {
        if (kind === 'video' && callType === 'video') {
          enc.maxBitrate    = VIDEO_MAX_BITRATE;
          enc.maxFramerate  = 60;
          (enc as any).networkPriority = 'high';
          (enc as any).priority        = 'high';
        } else if (kind === 'audio') {
          enc.maxBitrate    = AUDIO_MAX_BITRATE;
          (enc as any).priority = 'high';
        }
      }
      await sender.setParameters(params);
    } catch {
      // Some browsers reject setParameters before ICE completes — retry on connect
    }
  }
}

// ─── WebRTCManager ────────────────────────────────────────────────────────────

export class WebRTCManager {
  private calls     = new Map<string, PeerCall>();
  private callbacks: WebRTCCallbacks;
  private getAuthHeader?: () => Promise<string | null>;
  // ICE candidates that arrived before setRemoteDescription — flushed once ready.
  private iceCandidateBuffer = new Map<string, RTCIceCandidateInit[]>();

  constructor(callbacks: WebRTCCallbacks, getAuthHeader?: () => Promise<string | null>) {
    this.callbacks = callbacks;
    this.getAuthHeader = getAuthHeader;
  }

  // ─── TURN credential fetch (instance method — needs auth header) ────────────

  private async fetchTurnCredentials(): Promise<RTCIceServer[]> {
    const now = Date.now() / 1000;
    if (turnCredsCache && now - turnCredsCache.fetchedAt < turnCredsCache.creds.ttl - 60) {
      const c = turnCredsCache.creds;
      return [buildTurnIceServer(c.uris, c.username, c.password)];
    }
    try {
      const { apiBaseUrl } = getConnectionInfo();
      const headers: Record<string, string> = {};
      if (this.getAuthHeader) {
        const auth = await this.getAuthHeader();
        if (auth) headers['X-Wallet-Auth'] = auth;
      }
      const res = await fetch(`${apiBaseUrl}/v1/turn/credentials`, {
        credentials: 'include',
        headers,
      });
      if (!res.ok) throw new Error(`TURN credentials: ${res.status}`);
      const creds: TurnCredentials = await res.json();
      turnCredsCache = { creds, fetchedAt: now };
      console.log('[WebRTC] TURN credentials fetched, URIs:', creds.uris);
      return [buildTurnIceServer(creds.uris, creds.username, creds.password)];
    } catch (e) {
      console.warn('[WebRTC] TURN credentials unavailable, falling back to STUN:', e);
      return [];
    }
  }

  // ─── Media acquisition ──────────────────────────────────────────────────────

  private async acquireLocalStream(callType: CallType): Promise<MediaStream> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        'Camera/microphone access is unavailable. Two steps required in Tor Browser: ' +
        '(1) Set Security Level to Standard — click the Shield icon → Change Security Settings → Standard. ' +
        '(2) Open about:config and set media.peerconnection.enabled = true. ' +
        'Then reload the page. ' +
        'Note: calls cannot work at Safer or Safest security levels regardless of about:config settings.'
      );
    }
    try {
      return await navigator.mediaDevices.getUserMedia({
        audio: AUDIO_CONSTRAINTS,
        video: callType === 'video' ? VIDEO_CONSTRAINTS : false,
      });
    } catch (e: any) {
      if (e?.name === 'NotAllowedError' || e?.name === 'PermissionDeniedError') {
        throw new Error(
          'Camera/microphone permission was denied. ' +
          'Click the camera/lock icon in the address bar and allow access, then try again. ' +
          'In Tor Browser you may need to grant permissions each session.'
        );
      }
      if (e?.name === 'NotFoundError' || e?.name === 'DevicesNotFoundError') {
        throw new Error('No camera or microphone found. Please connect a device and try again.');
      }
      throw e;
    }
  }

  // ─── PeerConnection factory ──────────────────────────────────────────────────
  // Only creates the PC + wires event handlers. Tracks are added via addLocalTracks()
  // which uses addTrack(track, localStream) so the remote ontrack gets e.streams[0].
  // On the callee side, addLocalTracks is called BEFORE setRemoteDescription so the
  // transceiver is in a clean state and setRemoteDescription matches by kind.

  private async buildPeerConnection(peerId: string, callType: CallType): Promise<RTCPeerConnection> {
    // Tor Browser disables WebRTC by default (media.peerconnection.enabled = false).
    // Throw immediately so the caller can show a meaningful error instead of
    // the call UI hanging at "Connecting…" forever.
    if (!window.RTCPeerConnection) {
      throw new Error(
        'WebRTC is not available. ' +
        'If you are using Tor Browser, go to about:config and set ' +
        'media.peerconnection.enabled = true to enable voice/video calls.'
      );
    }

    const turnServers = await this.fetchTurnCredentials();

    // Use all ICE candidate types (host, srflx, relay) for maximum compatibility.
    // Include TURN relay for NAT traversal plus STUN for srflx discovery.
    // Note: Tor Browser (when WebRTC is enabled) filters out host/srflx candidates
    // automatically — only relay candidates via TURN will be used in that case.
    const iceServers: RTCIceServer[] = [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun1.l.google.com:19302' },
      ...turnServers,
    ];
    const iceTransportPolicy: RTCIceTransportPolicy = 'all';

    console.log('[WebRTC] ICE servers:', iceServers.length, '(TURN available:', turnServers.length > 0, ')');

    const pc = new RTCPeerConnection({
      iceServers,
      iceTransportPolicy,
      bundlePolicy:  'max-bundle',
      rtcpMuxPolicy: 'require',
    });

    // Re-apply encoding params once ICE connects (setParameters sometimes
    // needs the DTLS handshake to be complete first).
    pc.onconnectionstatechange = () => {
      const stateMap: Record<string, ConnectionState> = {
        new: 'new', connecting: 'connecting', connected: 'connected',
        disconnected: 'disconnected', failed: 'failed', closed: 'disconnected',
      };
      const mapped = stateMap[pc.connectionState] ?? 'new';
      const call = this.calls.get(peerId);
      if (call) call.state = mapped;
      this.callbacks.onStateChange(peerId, mapped);
      console.log('[WebRTC] connection state:', pc.connectionState, 'peer:', peerId.slice(-8));

      if (pc.connectionState === 'connected') {
        applyEncodingParams(pc, callType);
      }
    };

    // Tor Browser (WebRTC enabled but TCP-only): ICE may never reach 'connected'
    // or 'failed' if the TURN server only responds on UDP. After ICE_TIMEOUT_MS,
    // force the state to 'failed' so the UI shows an error rather than hanging.
    const iceTimer = setTimeout(() => {
      if (pc.connectionState !== 'connected' && pc.connectionState !== 'closed') {
        console.warn('[WebRTC] ICE timeout after', ICE_TIMEOUT_MS, 'ms — forcing failed for', peerId.slice(-8));
        const call = this.calls.get(peerId);
        if (call) call.state = 'failed';
        this.callbacks.onStateChange(peerId, 'failed');
      }
    }, ICE_TIMEOUT_MS);
    pc.addEventListener('connectionstatechange', () => {
      if (pc.connectionState === 'connected' || pc.connectionState === 'closed' || pc.connectionState === 'failed') {
        clearTimeout(iceTimer);
      }
    });

    pc.onicecandidate = (e) => {
      if (e.candidate) this.callbacks.onIceCandidate(peerId, e.candidate);
    };

    pc.onicegatheringstatechange = () => {
      console.log('[WebRTC] ICE gathering state:', pc.iceGatheringState, 'peer:', peerId.slice(-8));
    };

    pc.oniceconnectionstatechange = () => {
      console.log('[WebRTC] ICE connection state:', pc.iceConnectionState, 'peer:', peerId.slice(-8));
    };

    pc.ontrack = (e) => {
      console.log('[WebRTC] ontrack fired, streams:', e.streams.length, 'kind:', e.track.kind);
      // e.streams[0] is populated only when the sender used addTrack(track, stream).
      // If empty, build a synthetic stream from the track so audio still plays.
      const stream = e.streams[0] ?? new MediaStream([e.track]);
      this.callbacks.onRemoteStream(peerId, stream);
    };

    return pc;
  }

  // ─── Attach local tracks with stream association ──────────────────────────────
  // Uses addTrack(track, localStream) so the remote ontrack event gets e.streams[0].
  // Sets codec preferences on the transceivers that addTrack creates.

  private addLocalTracks(pc: RTCPeerConnection, localStream: MediaStream, callType: CallType): void {
    localStream.getTracks().forEach(t => pc.addTrack(t, localStream));

    if (typeof RTCRtpSender.getCapabilities === 'function') {
      for (const tcvr of pc.getTransceivers()) {
        const kind = tcvr.sender.track?.kind as 'audio' | 'video' | undefined;
        if (!kind) continue;
        const priority = kind === 'audio' ? AUDIO_CODEC_PRIORITY : VIDEO_CODEC_PRIORITY;
        if (kind === 'video' && callType !== 'video') continue;
        const caps = RTCRtpSender.getCapabilities(kind);
        if (caps) {
          try { tcvr.setCodecPreferences(preferCodecs(caps.codecs, priority)); } catch {}
        }
      }
    }
  }

  // ─── ICE candidate buffering ──────────────────────────────────────────────────
  // Candidates arriving before setRemoteDescription() are stored here and flushed
  // once the remote description is in place.

  private async flushIceCandidates(peerId: string, pc: RTCPeerConnection): Promise<void> {
    const buffered = this.iceCandidateBuffer.get(peerId) ?? [];
    this.iceCandidateBuffer.delete(peerId);
    if (buffered.length > 0) {
      console.log('[WebRTC] flushing', buffered.length, 'buffered ICE candidates for', peerId.slice(-8));
      for (const cand of buffered) {
        try { await pc.addIceCandidate(cand); } catch {}
      }
    }
  }

  // ─── Call initiation (caller side) ──────────────────────────────────────────

  async initiateCall(peerId: string, callType: CallType): Promise<void> {
    this.hangup(peerId);

    // Acquire media before creating PC — prevents dangling PC if permission denied.
    const localStream = await this.acquireLocalStream(callType);
    const pc = await this.buildPeerConnection(peerId, callType);

    // addTrack(track, localStream) is critical: it associates the stream so the
    // remote ontrack event gets e.streams[0] populated.
    this.addLocalTracks(pc, localStream, callType);
    this.calls.set(peerId, { pc, state: 'new', callType, localStream });

    const offer = await pc.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: callType === 'video' });
    const enhancedSdp = enhanceSdp(offer.sdp!, callType);
    await pc.setLocalDescription({ type: 'offer', sdp: enhancedSdp });
    this.callbacks.onLocalOffer(peerId, enhancedSdp, callType);
  }

  // ─── Offer handling (callee side) ────────────────────────────────────────────

  async handleOffer(peerId: string, sdp: string, callType: CallType): Promise<void> {
    this.hangup(peerId);

    const localStream = await this.acquireLocalStream(callType);
    const pc = await this.buildPeerConnection(peerId, callType);

    // Add tracks BEFORE setRemoteDescription: addTrack creates transceivers in a
    // known state and properly associates localStream. setRemoteDescription then
    // matches the offer's m-sections to those transceivers by kind (RFC 8829).
    this.addLocalTracks(pc, localStream, callType);

    // Register the call entry before setRemoteDescription so any ICE candidates
    // that arrive during the async gap are buffered (not dropped).
    this.calls.set(peerId, { pc, state: 'connecting', callType, localStream });

    await pc.setRemoteDescription({ type: 'offer', sdp });

    // Flush any ICE candidates that arrived before setRemoteDescription was ready.
    await this.flushIceCandidates(peerId, pc);

    const answer = await pc.createAnswer();
    const enhancedSdp = enhanceSdp(answer.sdp!, callType);
    await pc.setLocalDescription({ type: 'answer', sdp: enhancedSdp });
    this.callbacks.onLocalAnswer(peerId, enhancedSdp);
  }

  // ─── Answer / ICE handling ───────────────────────────────────────────────────

  async handleAnswer(peerId: string, sdp: string): Promise<void> {
    const call = this.calls.get(peerId);
    if (!call) return;
    await call.pc.setRemoteDescription({ type: 'answer', sdp });
    // Flush any ICE candidates that raced ahead of the answer.
    await this.flushIceCandidates(peerId, call.pc);
  }

  async handleIceCandidate(
    peerId: string,
    candidate: string,
    sdpMid: string | null,
    sdpMLineIndex: number | null,
  ): Promise<void> {
    const init: RTCIceCandidateInit = {
      candidate,
      sdpMid:        sdpMid        ?? undefined,
      sdpMLineIndex: sdpMLineIndex ?? undefined,
    };
    const call = this.calls.get(peerId);
    // Buffer if the peer connection isn't ready or remote description isn't set yet.
    if (!call || !call.pc.remoteDescription) {
      const buf = this.iceCandidateBuffer.get(peerId) ?? [];
      buf.push(init);
      this.iceCandidateBuffer.set(peerId, buf);
      console.log('[WebRTC] ICE candidate buffered for', peerId.slice(-8), '— total:', buf.length);
      return;
    }
    try {
      await call.pc.addIceCandidate(init);
    } catch (e) {
      console.warn('[WebRTC] addIceCandidate error:', e);
    }
  }

  // ─── Teardown ────────────────────────────────────────────────────────────────

  hangup(peerId: string): void {
    const call = this.calls.get(peerId);
    if (!call) return;
    call.localStream?.getTracks().forEach(t => t.stop());
    call.pc.close();
    this.calls.delete(peerId);
    this.iceCandidateBuffer.delete(peerId);
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

  async setVideoQuality(peerId: string, quality: '4k' | 'hd'): Promise<void> {
    const call = this.calls.get(peerId);
    if (!call || call.callType !== 'video') return;

    const constraints: MediaTrackConstraints = quality === '4k'
      ? { facingMode: 'user', width: { ideal: 3840 }, height: { ideal: 2160 }, frameRate: { ideal: 30 } }
      : { facingMode: 'user', width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } };

    try {
      const newStream = await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
      const newTrack = newStream.getVideoTracks()[0];
      if (!newTrack) return;

      const sender = call.pc.getSenders().find(s => s.track?.kind === 'video');
      if (sender) await sender.replaceTrack(newTrack);

      if (call.localStream) {
        call.localStream.getVideoTracks().forEach(t => { t.stop(); call.localStream!.removeTrack(t); });
        call.localStream.addTrack(newTrack);
        this.callbacks.onRemoteStream(peerId, call.localStream);
      }
    } catch (e) {
      console.warn('[WebRTC] setVideoQuality failed:', e);
    }
  }
}
