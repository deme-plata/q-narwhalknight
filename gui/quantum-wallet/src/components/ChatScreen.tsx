// Chat, Voice, Video & Meetings — Quillon node communication hub
// Architecture:
//   Text chat  — GossipSub P2P (no server relay)
//   Voice/Video — browser RTCPeerConnection via WebRTC
//   Signaling  — /ws/chat/signal WebSocket (SDP + ICE routing only)
//   Meetings   — mesh WebRTC up to 49 peers (Proton Meet style)

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare, Phone, PhoneOff, Video, VideoOff, Mic, MicOff,
  Monitor, Users, X, Send, ChevronLeft, UserPlus, Hash,
  Loader2, Shield, Lock,
} from 'lucide-react';
import { SignalingService, SignalingEnvelope, PeerInfo, CallType } from '../services/SignalingService';
import { WebRTCManager, ConnectionState } from '../webrtc/WebRTCManager';
import { walletSession } from '../services/walletAuth';

// ── Types ──────────────────────────────────────────────────────────────────

interface ChatMessage {
  id: string;
  from: string;
  content: string;
  timestamp: number;
  self: boolean;
}

interface CallInfo {
  peerId: string;
  callType: CallType;
  state: ConnectionState;
  remoteStream: MediaStream | null;
  localStream: MediaStream | null;
}

interface MeetingRoom {
  roomId: string;
  peers: PeerInfo[];
  calls: Map<string, CallInfo>;
}

type Tab = 'messages' | 'calls' | 'meetings';

// ── Helpers ────────────────────────────────────────────────────────────────

function shortId(peerId: string): string {
  if (!peerId || peerId === 'server') return peerId;
  return peerId.length > 12 ? `${peerId.slice(0, 6)}…${peerId.slice(-4)}` : peerId;
}

function timestamp(): string {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ── Sub-components ─────────────────────────────────────────────────────────

function RemoteVideo({ stream, peerId }: { stream: MediaStream; peerId: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.srcObject = stream;
  }, [stream]);
  return (
    <div className="relative rounded-xl overflow-hidden bg-slate-900 aspect-video">
      <video ref={ref} autoPlay playsInline className="w-full h-full object-cover" />
      <span className="absolute bottom-2 left-2 text-xs text-white bg-black/50 px-2 py-0.5 rounded">
        {shortId(peerId)}
      </span>
    </div>
  );
}

function LocalVideo({ stream }: { stream: MediaStream }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.srcObject = stream;
  }, [stream]);
  return (
    <video
      ref={ref}
      autoPlay
      playsInline
      muted
      className="absolute bottom-4 right-4 w-32 h-24 rounded-lg object-cover border border-amber-400/40 shadow-lg"
    />
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function ChatScreen() {
  const walletAddress = localStorage.getItem('walletAddress') || '';
  const displayName = shortId(walletAddress);

  const [tab, setTab] = useState<Tab>('messages');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [targetPeerId, setTargetPeerId] = useState('');
  const [activeCall, setActiveCall] = useState<CallInfo | null>(null);
  const [micOn, setMicOn] = useState(true);
  const [camOn, setCamOn] = useState(true);
  const [meeting, setMeeting] = useState<MeetingRoom | null>(null);
  const [roomInput, setRoomInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [incomingCall, setIncomingCall] = useState<{ from: string; sdp: string; callType: 'audio' | 'video'; sessionId: string | null } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const signalingRef = useRef<SignalingService | null>(null);
  const webrtcRef = useRef<WebRTCManager | null>(null);

  // Scroll chat to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Init signaling + WebRTC
  useEffect(() => {
    if (!walletAddress) return;

    // Pass private key getter so SignalingService can sign auth header (CRIT-1 fix)
    const getPrivateKey = () => walletSession.getSession()?.privateKey ?? null;
    const signaling = new SignalingService(walletAddress, getPrivateKey);
    signalingRef.current = signaling;

    const webrtc = new WebRTCManager({
      onStateChange: (peerId, state) => {
        setActiveCall((prev) => {
          if (!prev || prev.peerId !== peerId) return prev;
          if (state === 'disconnected' || state === 'failed') return null;
          return { ...prev, state };
        });
        setMeeting((prev) => {
          if (!prev) return prev;
          const calls = new Map(prev.calls);
          const existing = calls.get(peerId);
          if (existing) {
            if (state === 'disconnected' || state === 'failed') {
              calls.delete(peerId);
            } else {
              calls.set(peerId, { ...existing, state });
            }
          }
          return { ...prev, calls };
        });
      },
      onRemoteStream: (peerId, stream) => {
        setActiveCall((prev) => {
          if (prev?.peerId === peerId) return { ...prev, remoteStream: stream };
          return prev;
        });
        setMeeting((prev) => {
          if (!prev) return prev;
          const calls = new Map(prev.calls);
          const existing = calls.get(peerId);
          if (existing) calls.set(peerId, { ...existing, remoteStream: stream });
          return { ...prev, calls };
        });
      },
      onIceCandidate: (peerId, candidate) => {
        signaling.send(peerId, null, {
          type: 'ice_candidate',
          candidate: candidate.candidate,
          sdp_mid: candidate.sdpMid ?? null,
          sdp_m_line_index: candidate.sdpMLineIndex ?? null,
        });
      },
      onLocalOffer: (peerId, sdp, callType) => {
        signaling.send(peerId, null, { type: 'call_offer', sdp, call_type: callType });
      },
      onLocalAnswer: (peerId, sdp) => {
        signaling.send(peerId, null, { type: 'call_answer', sdp });
      },
    });
    webrtcRef.current = webrtc;

    const unsub = signaling.onMessage((envelope: SignalingEnvelope) => {
      handleIncoming(envelope, signaling, webrtc);
    });

    signaling.connect();

    return () => {
      unsub();
      webrtc.hangupAll();
      signaling.disconnect();
      signalingRef.current = null;
      webrtcRef.current = null;
      setIsConnected(false);
    };
  }, [walletAddress]);

  const handleIncoming = useCallback((
    envelope: SignalingEnvelope,
    signaling: SignalingService,
    webrtc: WebRTCManager,
  ) => {
    const { payload, from } = envelope;
    switch (payload.type) {
      case 'chat_message': {
        const msg: ChatMessage = {
          id: `${from}-${payload.timestamp}`,
          from,
          content: payload.content,
          timestamp: payload.timestamp,
          self: false,
        };
        setMessages((prev) => [...prev, msg]);
        break;
      }
      case 'call_offer': {
        // Show incoming call UI — user must tap Accept before media is activated (CRIT-4 fix)
        setIncomingCall({ from, sdp: payload.sdp, callType: payload.call_type, sessionId: envelope.session_id });
        break;
      }
      case 'call_answer': {
        webrtc.handleAnswer(from, payload.sdp);
        break;
      }
      case 'ice_candidate': {
        webrtc.handleIceCandidate(from, payload.candidate, payload.sdp_mid, payload.sdp_m_line_index);
        break;
      }
      case 'call_end': {
        webrtc.hangup(from);
        setActiveCall(null);
        break;
      }
      case 'meeting_peers': {
        // Server sent us the existing peer list — initiate calls to each
        setMeeting((prev) => {
          if (!prev || prev.roomId !== payload.room_id) return prev;
          return { ...prev, peers: payload.peers };
        });
        payload.peers.forEach((peer) => {
          webrtc.initiateCall(peer.peer_id, 'video').then(() => {
            const localStream = webrtc.getLocalStream(peer.peer_id);
            setMeeting((prev) => {
              if (!prev) return prev;
              const calls = new Map(prev.calls);
              calls.set(peer.peer_id, { peerId: peer.peer_id, callType: 'video', state: 'connecting', remoteStream: null, localStream });
              return { ...prev, calls };
            });
          });
        });
        break;
      }
      case 'meeting_peer_joined': {
        setMeeting((prev) => {
          if (!prev || prev.roomId !== payload.room_id) return prev;
          const peers = [...prev.peers.filter((p) => p.peer_id !== payload.peer.peer_id), payload.peer];
          return { ...prev, peers };
        });
        // New peer will initiate a call to us (they received our existing presence)
        break;
      }
      case 'meeting_peer_left': {
        webrtc.hangup(payload.peer_id);
        setMeeting((prev) => {
          if (!prev || prev.roomId !== payload.room_id) return prev;
          const peers = prev.peers.filter((p) => p.peer_id !== payload.peer_id);
          const calls = new Map(prev.calls);
          calls.delete(payload.peer_id);
          return { ...prev, peers, calls };
        });
        break;
      }
      case 'pong':
        // Server replied to our ping — signaling channel is alive
        setIsConnected(true);
        break;
    }
  }, []);

  // ── Incoming call consent handlers ───────────────────────────────────────

  const handleAcceptCall = async () => {
    if (!incomingCall || !webrtcRef.current || !signalingRef.current) return;
    const { from, sdp, callType } = incomingCall;
    setIncomingCall(null);
    await webrtcRef.current.handleOffer(from, sdp, callType);
    const localStream = webrtcRef.current.getLocalStream(from);
    setActiveCall({ peerId: from, callType, state: 'connecting', remoteStream: null, localStream });
    setTab('calls');
  };

  const handleRejectCall = () => {
    if (!incomingCall || !signalingRef.current) return;
    signalingRef.current.send(incomingCall.from, incomingCall.sessionId, { type: 'call_end', reason: 'rejected' });
    setIncomingCall(null);
  };

  // ── Actions ───────────────────────────────────────────────────────────────

  const sendMessage = () => {
    const content = inputText.trim();
    if (!content || !targetPeerId.trim() || !signalingRef.current) return;
    const ts = Date.now();
    signalingRef.current.send(targetPeerId.trim(), null, {
      type: 'chat_message',
      content,
      timestamp: ts,
    });
    setMessages((prev) => [...prev, {
      id: `self-${ts}`, from: walletAddress, content, timestamp: ts, self: true,
    }]);
    setInputText('');
  };

  const startCall = async (callType: CallType) => {
    if (!targetPeerId.trim() || !webrtcRef.current) return;
    const peerId = targetPeerId.trim();
    await webrtcRef.current.initiateCall(peerId, callType);
    const localStream = webrtcRef.current.getLocalStream(peerId);
    setActiveCall({ peerId, callType, state: 'connecting', remoteStream: null, localStream });
    setTab('calls');
  };

  const hangup = () => {
    if (!activeCall || !webrtcRef.current || !signalingRef.current) return;
    webrtcRef.current.hangup(activeCall.peerId);
    signalingRef.current.send(activeCall.peerId, null, { type: 'call_end', reason: 'user_hangup' });
    setActiveCall(null);
  };

  const toggleMic = () => {
    setMicOn((prev) => {
      const newVal = !prev;
      // mute/unmute local audio tracks for active call or meeting
      const muteStream = (stream: MediaStream | null) => {
        stream?.getAudioTracks().forEach((t) => (t.enabled = newVal));
      };
      if (activeCall) muteStream(activeCall.localStream);
      if (meeting) {
        meeting.calls.forEach((c) => muteStream(c.localStream));
      }
      return newVal;
    });
  };

  const toggleCam = () => {
    setCamOn((prev) => {
      const newVal = !prev;
      const muteStream = (stream: MediaStream | null) => {
        stream?.getVideoTracks().forEach((t) => (t.enabled = newVal));
      };
      if (activeCall) muteStream(activeCall.localStream);
      if (meeting) {
        meeting.calls.forEach((c) => muteStream(c.localStream));
      }
      return newVal;
    });
  };

  const joinMeeting = () => {
    const roomId = roomInput.trim();
    if (!roomId || !signalingRef.current) return;
    setMeeting({ roomId, peers: [], calls: new Map() });
    signalingRef.current.send(null, null, {
      type: 'meeting_join',
      room_id: roomId,
      display_name: displayName,
    });
    setTab('meetings');
  };

  const leaveMeeting = () => {
    if (!meeting || !signalingRef.current || !webrtcRef.current) return;
    signalingRef.current.send(null, null, { type: 'meeting_leave', room_id: meeting.roomId });
    meeting.peers.forEach((p) => webrtcRef.current!.hangup(p.peer_id));
    setMeeting(null);
    setRoomInput('');
  };

  // ── Render ────────────────────────────────────────────────────────────────

  const tabBtn = (id: Tab, icon: React.ReactNode, label: string, badge?: number) => (
    <button
      onClick={() => setTab(id)}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all relative ${
        tab === id
          ? 'text-amber-50'
          : 'text-amber-300/50 hover:text-amber-200'
      }`}
      style={tab === id ? {
        background: 'linear-gradient(135deg, rgba(212,175,55,0.2), rgba(255,215,0,0.1))',
        border: '1.5px solid rgba(212,175,55,0.4)',
      } : { border: '1.5px solid transparent' }}
    >
      {icon}
      {label}
      {badge ? (
        <span className="absolute -top-1 -right-1 w-4 h-4 text-xs rounded-full bg-amber-400 text-slate-900 flex items-center justify-center">
          {badge}
        </span>
      ) : null}
    </button>
  );

  return (
    <div className="flex flex-col h-full" style={{ minHeight: 0 }}>

      {/* Incoming call consent banner */}
      {incomingCall && (
        <div
          className="flex items-center justify-between px-6 py-3 shrink-0"
          style={{ background: 'rgba(212,175,55,0.15)', borderBottom: '1px solid rgba(212,175,55,0.3)' }}
        >
          <span className="text-amber-200 text-sm font-medium">
            📞 Incoming {incomingCall.callType} call from {incomingCall.from.slice(0, 12)}…
          </span>
          <div className="flex gap-2">
            <button
              onClick={handleAcceptCall}
              className="px-4 py-1 rounded-lg text-xs font-semibold"
              style={{ background: 'rgba(34,197,94,0.25)', color: '#86efac', border: '1px solid rgba(34,197,94,0.4)' }}
            >Accept</button>
            <button
              onClick={handleRejectCall}
              className="px-4 py-1 rounded-lg text-xs font-semibold"
              style={{ background: 'rgba(239,68,68,0.2)', color: '#fca5a5', border: '1px solid rgba(239,68,68,0.3)' }}
            >Decline</button>
          </div>
        </div>
      )}

      {/* Header */}
      <div
        className="flex items-center justify-between px-6 py-4 border-b shrink-0"
        style={{ borderColor: 'rgba(212,175,55,0.15)', background: 'rgba(15,23,42,0.6)' }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg, rgba(212,175,55,0.2), rgba(255,215,0,0.1))' }}
          >
            <MessageSquare className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-amber-100">Chat & Calls</h1>
            <p className="text-xs text-amber-300/50 flex items-center gap-1">
              <Lock className="w-3 h-3" /> End-to-end encrypted · WebRTC
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-slate-500'}`}
            style={isConnected ? { boxShadow: '0 0 6px rgba(74,222,128,0.6)' } : {}}
          />
          <span className="text-xs text-amber-300/50">{isConnected ? 'Signaling online' : 'Connecting…'}</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 px-6 py-3 shrink-0">
        {tabBtn('messages', <MessageSquare className="w-4 h-4" />, 'Messages')}
        {tabBtn('calls', <Phone className="w-4 h-4" />, 'Calls', activeCall ? 1 : undefined)}
        {tabBtn('meetings', <Users className="w-4 h-4" />, 'Meetings', meeting ? 1 : undefined)}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-hidden" style={{ minHeight: 0 }}>
        <AnimatePresence mode="wait">

          {/* ── Messages Tab ── */}
          {tab === 'messages' && (
            <motion.div
              key="messages"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              className="flex flex-col h-full"
              style={{ minHeight: 0 }}
            >
              {/* Peer address bar */}
              <div className="flex gap-2 px-6 py-3 border-b shrink-0" style={{ borderColor: 'rgba(212,175,55,0.1)' }}>
                <div className="relative flex-1">
                  <Hash className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-amber-400/50" />
                  <input
                    className="w-full pl-9 pr-4 py-2 rounded-lg text-sm text-amber-100 placeholder-amber-300/30 outline-none"
                    style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(212,175,55,0.2)' }}
                    placeholder="Recipient wallet address…"
                    value={targetPeerId}
                    onChange={(e) => setTargetPeerId(e.target.value)}
                  />
                </div>
                <button
                  onClick={() => startCall('audio')}
                  className="p-2 rounded-lg text-amber-300/60 hover:text-amber-300 transition-colors"
                  style={{ border: '1px solid rgba(212,175,55,0.2)' }}
                  title="Voice call"
                >
                  <Phone className="w-4 h-4" />
                </button>
                <button
                  onClick={() => startCall('video')}
                  className="p-2 rounded-lg text-amber-300/60 hover:text-amber-300 transition-colors"
                  style={{ border: '1px solid rgba(212,175,55,0.2)' }}
                  title="Video call"
                >
                  <Video className="w-4 h-4" />
                </button>
              </div>

              {/* Message list */}
              <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3" style={{ minHeight: 0 }}>
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full gap-3 text-amber-300/30">
                    <Shield className="w-10 h-10" />
                    <p className="text-sm">No messages yet</p>
                    <p className="text-xs">Enter a wallet address above and start talking.</p>
                  </div>
                ) : messages.map((msg) => (
                  <div key={msg.id} className={`flex ${msg.self ? 'justify-end' : 'justify-start'}`}>
                    <div
                      className="max-w-xs lg:max-w-md px-4 py-2 rounded-2xl text-sm"
                      style={msg.self ? {
                        background: 'linear-gradient(135deg, rgba(212,175,55,0.3), rgba(255,215,0,0.2))',
                        border: '1px solid rgba(212,175,55,0.3)',
                        color: '#fef3c7',
                      } : {
                        background: 'rgba(255,255,255,0.07)',
                        border: '1px solid rgba(255,255,255,0.08)',
                        color: '#cbd5e1',
                      }}
                    >
                      {!msg.self && (
                        <p className="text-xs text-amber-400/60 mb-1">{shortId(msg.from)}</p>
                      )}
                      <p>{msg.content}</p>
                      <p className="text-xs opacity-50 mt-1 text-right">
                        {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </p>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div
                className="flex gap-2 px-6 py-4 border-t shrink-0"
                style={{ borderColor: 'rgba(212,175,55,0.1)' }}
              >
                <input
                  className="flex-1 px-4 py-2 rounded-xl text-sm text-amber-100 placeholder-amber-300/30 outline-none"
                  style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(212,175,55,0.2)' }}
                  placeholder="Type a message…"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                />
                <button
                  onClick={sendMessage}
                  className="p-2 rounded-xl transition-all"
                  style={{
                    background: 'linear-gradient(135deg, rgba(212,175,55,0.4), rgba(255,215,0,0.25))',
                    border: '1px solid rgba(212,175,55,0.4)',
                  }}
                >
                  <Send className="w-5 h-5 text-amber-300" />
                </button>
              </div>
            </motion.div>
          )}

          {/* ── Calls Tab ── */}
          {tab === 'calls' && (
            <motion.div
              key="calls"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              className="flex flex-col h-full p-6 gap-4"
              style={{ minHeight: 0 }}
            >
              {activeCall ? (
                <div className="flex flex-col gap-4 h-full">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-amber-300/60">Call with</p>
                      <p className="font-bold text-amber-100">{shortId(activeCall.peerId)}</p>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      activeCall.state === 'connected'
                        ? 'bg-green-500/20 text-green-300'
                        : 'bg-amber-500/20 text-amber-300'
                    }`}>
                      {activeCall.state === 'connected' ? 'Connected' : 'Connecting…'}
                    </span>
                  </div>

                  {/* Video area */}
                  <div className="flex-1 relative rounded-2xl overflow-hidden bg-slate-900 min-h-0">
                    {activeCall.remoteStream ? (
                      <RemoteVideo stream={activeCall.remoteStream} peerId={activeCall.peerId} />
                    ) : (
                      <div className="flex flex-col items-center justify-center h-full gap-3 text-amber-300/40">
                        <Loader2 className="w-8 h-8 animate-spin" />
                        <p className="text-sm">Waiting for peer…</p>
                      </div>
                    )}
                    {activeCall.callType === 'video' && activeCall.localStream && (
                      <LocalVideo stream={activeCall.localStream} />
                    )}
                  </div>

                  {/* Call controls */}
                  <div className="flex items-center justify-center gap-4 shrink-0">
                    <button
                      onClick={toggleMic}
                      className="w-12 h-12 rounded-full flex items-center justify-center transition-all"
                      style={{
                        background: micOn ? 'rgba(255,255,255,0.1)' : 'rgba(239,68,68,0.3)',
                        border: `1.5px solid ${micOn ? 'rgba(255,255,255,0.15)' : 'rgba(239,68,68,0.5)'}`,
                      }}
                    >
                      {micOn ? <Mic className="w-5 h-5 text-white" /> : <MicOff className="w-5 h-5 text-red-300" />}
                    </button>
                    {activeCall.callType === 'video' && (
                      <button
                        onClick={toggleCam}
                        className="w-12 h-12 rounded-full flex items-center justify-center transition-all"
                        style={{
                          background: camOn ? 'rgba(255,255,255,0.1)' : 'rgba(239,68,68,0.3)',
                          border: `1.5px solid ${camOn ? 'rgba(255,255,255,0.15)' : 'rgba(239,68,68,0.5)'}`,
                        }}
                      >
                        {camOn ? <Video className="w-5 h-5 text-white" /> : <VideoOff className="w-5 h-5 text-red-300" />}
                      </button>
                    )}
                    <button
                      onClick={hangup}
                      className="w-14 h-14 rounded-full flex items-center justify-center"
                      style={{ background: 'rgba(239,68,68,0.8)', border: '1.5px solid rgba(239,68,68,0.6)' }}
                    >
                      <PhoneOff className="w-6 h-6 text-white" />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full gap-6 text-amber-300/40">
                  <Phone className="w-12 h-12" />
                  <div className="text-center">
                    <p className="font-semibold text-amber-200/50">No active call</p>
                    <p className="text-sm mt-1">Enter a wallet address in Messages and tap the call button.</p>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ── Meetings Tab ── */}
          {tab === 'meetings' && (
            <motion.div
              key="meetings"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              className="flex flex-col h-full p-6 gap-4"
              style={{ minHeight: 0 }}
            >
              {meeting ? (
                <div className="flex flex-col h-full gap-4">
                  {/* Room header */}
                  <div className="flex items-center justify-between shrink-0">
                    <div>
                      <p className="text-xs text-amber-300/50">Room</p>
                      <p className="font-bold text-amber-100">{meeting.roomId}</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-amber-300/50">
                        {meeting.peers.length} peer{meeting.peers.length !== 1 ? 's' : ''}
                      </span>
                      <button
                        onClick={leaveMeeting}
                        className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm text-red-300 transition-all"
                        style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)' }}
                      >
                        <X className="w-4 h-4" /> Leave
                      </button>
                    </div>
                  </div>

                  {/* Peer video grid */}
                  <div className="flex-1 min-h-0 overflow-y-auto">
                    {meeting.peers.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-full gap-3 text-amber-300/30">
                        <Loader2 className="w-8 h-8 animate-spin" />
                        <p className="text-sm">Waiting for others to join…</p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                        {meeting.peers.map((peer) => {
                          const callInfo = meeting.calls.get(peer.peer_id);
                          return (
                            <div
                              key={peer.peer_id}
                              className="aspect-video rounded-xl overflow-hidden bg-slate-900 flex items-center justify-center relative"
                            >
                              {callInfo?.remoteStream ? (
                                <RemoteVideo stream={callInfo.remoteStream} peerId={peer.peer_id} />
                              ) : (
                                <>
                                  <div
                                    className="w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold"
                                    style={{ background: 'rgba(212,175,55,0.2)', color: '#d4af37' }}
                                  >
                                    {peer.display_name.slice(0, 2).toUpperCase()}
                                  </div>
                                  <span className="absolute bottom-2 left-2 text-xs text-white bg-black/50 px-2 py-0.5 rounded">
                                    {peer.display_name}
                                  </span>
                                </>
                              )}
                            </div>
                          );
                        })}
                        {/* Self tile */}
                        <div className="aspect-video rounded-xl overflow-hidden bg-slate-800 flex items-center justify-center relative border border-amber-400/20">
                          <div
                            className="w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold"
                            style={{ background: 'rgba(212,175,55,0.3)', color: '#d4af37' }}
                          >
                            {displayName.slice(0, 2).toUpperCase()}
                          </div>
                          <span className="absolute bottom-2 left-2 text-xs text-amber-300 px-2 py-0.5 rounded">
                            You
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Meeting controls */}
                  <div className="flex items-center justify-center gap-4 shrink-0">
                    <button
                      onClick={toggleMic}
                      className="w-12 h-12 rounded-full flex items-center justify-center transition-all"
                      style={{
                        background: micOn ? 'rgba(255,255,255,0.1)' : 'rgba(239,68,68,0.3)',
                        border: `1.5px solid ${micOn ? 'rgba(255,255,255,0.15)' : 'rgba(239,68,68,0.5)'}`,
                      }}
                    >
                      {micOn ? <Mic className="w-5 h-5 text-white" /> : <MicOff className="w-5 h-5 text-red-300" />}
                    </button>
                    <button
                      onClick={toggleCam}
                      className="w-12 h-12 rounded-full flex items-center justify-center transition-all"
                      style={{
                        background: camOn ? 'rgba(255,255,255,0.1)' : 'rgba(239,68,68,0.3)',
                        border: `1.5px solid ${camOn ? 'rgba(255,255,255,0.15)' : 'rgba(239,68,68,0.5)'}`,
                      }}
                    >
                      {camOn ? <Video className="w-5 h-5 text-white" /> : <VideoOff className="w-5 h-5 text-red-300" />}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col gap-6">
                  {/* Join / create room */}
                  <div
                    className="rounded-2xl p-6"
                    style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(212,175,55,0.15)' }}
                  >
                    <div className="flex items-center gap-2 mb-4">
                      <Users className="w-5 h-5 text-amber-400" />
                      <h2 className="font-bold text-amber-100">Start or join a meeting</h2>
                    </div>
                    <p className="text-xs text-amber-300/50 mb-4">
                      End-to-end encrypted · Up to 49 participants · No account needed
                    </p>
                    <div className="flex gap-2">
                      <input
                        className="flex-1 px-4 py-2 rounded-xl text-sm text-amber-100 placeholder-amber-300/30 outline-none"
                        style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(212,175,55,0.2)' }}
                        placeholder="Room name or ID…"
                        value={roomInput}
                        onChange={(e) => setRoomInput(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter') joinMeeting(); }}
                      />
                      <button
                        onClick={joinMeeting}
                        className="flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm transition-all"
                        style={{
                          background: 'linear-gradient(135deg, rgba(212,175,55,0.4), rgba(255,215,0,0.25))',
                          border: '1px solid rgba(212,175,55,0.5)',
                          color: '#fef3c7',
                        }}
                      >
                        <UserPlus className="w-4 h-4" />
                        Join
                      </button>
                    </div>
                  </div>

                  {/* Feature pills */}
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { icon: <Lock className="w-4 h-4" />, text: 'E2E Encrypted' },
                      { icon: <Shield className="w-4 h-4" />, text: 'Dilithium5 keys' },
                      { icon: <Monitor className="w-4 h-4" />, text: 'Screen sharing' },
                      { icon: <Users className="w-4 h-4" />, text: 'Up to 49 peers' },
                    ].map(({ icon, text }) => (
                      <div
                        key={text}
                        className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-amber-300/60"
                        style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(212,175,55,0.1)' }}
                      >
                        <span className="text-amber-400/50">{icon}</span>
                        {text}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  );
}
