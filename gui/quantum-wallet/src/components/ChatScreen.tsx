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
  Monitor, Users, X, Send, UserPlus, Hash,
  Loader2, Shield, Lock, Search, Star, ChevronRight, Plus, Bot,
} from 'lucide-react';
import { SignalingService } from '../services/SignalingService';
import type { SignalingEnvelope, PeerInfo, CallType, SignalingDiag } from '../services/SignalingService';
import { WebRTCManager } from '../webrtc/WebRTCManager';
import type { ConnectionState } from '../webrtc/WebRTCManager';
import { walletSession, generateAuthHeader } from '../services/walletAuth';
import { getConnectionInfo, qnkAPI } from '../services/api';
import GroupsTab from './GroupsTab';
import AddressBook from './AddressBook';

// ── Types ──────────────────────────────────────────────────────────────────

interface ChatMessage {
  id: string;
  from: string;
  to?: string;
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

interface SavedContact {
  id: string;
  address: string;
  label: string;
  favorite?: boolean;
}

interface CallHistoryEntry {
  peerId: string;
  callType: CallType;
  timestamp: number;
  duration?: number;
}

interface MeetingHistoryEntry {
  roomId: string;
  timestamp: number;
  peerCount: number;
}

type Tab = 'messages' | 'calls' | 'meetings' | 'groups';

// ── Helpers ────────────────────────────────────────────────────────────────

function shortId(peerId: string): string {
  if (!peerId || peerId === 'server') return peerId;
  return peerId.length > 12 ? `${peerId.slice(0, 6)}…${peerId.slice(-4)}` : peerId;
}

function truncAddr(addr: string): string {
  if (!addr || addr.length < 14) return addr;
  return `${addr.slice(0, 8)}…${addr.slice(-6)}`;
}

function randomRoomId(): string {
  const words = ['nova','echo','quant','delta','orbit','flux','cipher','nexus','lunar','spark'];
  const a = words[Math.floor(Math.random() * words.length)];
  const b = words[Math.floor(Math.random() * words.length)];
  const n = Math.floor(Math.random() * 900) + 100;
  return `${a}-${b}-${n}`;
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
  const [callError, setCallError] = useState<string | null>(null);
  const [micOn, setMicOn] = useState(true);
  const [camOn, setCamOn] = useState(true);
  const [meeting, setMeeting] = useState<MeetingRoom | null>(null);
  const [roomInput, setRoomInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [incomingCall, setIncomingCall] = useState<{ from: string; sdp: string; callType: 'audio' | 'video'; sessionId: string | null } | null>(null);
  // Connection diagnostics — exposed to UI so the user can see exactly why a call isn't going through
  const [diag, setDiag] = useState<SignalingDiag | null>(null);
  const [showDiag, setShowDiag] = useState(false);
  const [serverPeers, setServerPeers] = useState<{ peer_count: number; peers: string[]; room_count: number; active_call_count: number } | null>(null);

  // Contact sidebar state
  const [contacts, setContacts] = useState<SavedContact[]>([]);
  const [contactSearch, setContactSearch] = useState('');
  const [addressInput, setAddressInput] = useState('');
  const [showAddressBook, setShowAddressBook] = useState(false);
  // Unread message tracking
  const [unreadFrom, setUnreadFrom] = useState<Set<string>>(new Set());
  const targetPeerIdRef = useRef(targetPeerId);

  const [callHistory, setCallHistory] = useState<CallHistoryEntry[]>([]);
  const [meetingHistory, setMeetingHistory] = useState<MeetingHistoryEntry[]>([]);
  const callStartTimeRef = useRef<number | null>(null);

  // AI call assistant state
  const [aiEnabled, setAiEnabled] = useState(false);
  const [videoQuality, setVideoQuality] = useState<'hd' | '4k'>('hd');
  const [aiTranscript, setAiTranscript] = useState('');
  const [aiInput, setAiInput] = useState('');
  const [aiSuggestion, setAiSuggestion] = useState('');
  const [aiStreaming, setAiStreaming] = useState(false);
  const [aiSpeechAvailable, setAiSpeechAvailable] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const signalingRef = useRef<SignalingService | null>(null);
  const webrtcRef = useRef<WebRTCManager | null>(null);
  const recognitionRef = useRef<any>(null);
  const aiAbortRef = useRef<AbortController | null>(null);
  // Maps peerId → session_id so call_answer can echo the session_id back to the server
  const sessionIdsRef = useRef<Map<string, string | null>>(new Map());

  // Keep contacts accessible inside stable callbacks
  const contactsRef = useRef(contacts);
  useEffect(() => { contactsRef.current = contacts; }, [contacts]);

  // Listen for external "open this conversation" commands (e.g. from message toast)
  useEffect(() => {
    const handler = (e: Event) => {
      const addr = (e as CustomEvent).detail?.address as string;
      if (addr) {
        setTargetPeerId(addr);
        setTab('messages');
      }
    };
    window.addEventListener('qnk-open-conversation', handler);
    return () => window.removeEventListener('qnk-open-conversation', handler);
  }, []);

  // Keep targetPeerIdRef in sync for use inside stable callbacks
  useEffect(() => {
    targetPeerIdRef.current = targetPeerId;
    // Clear unread when opening a conversation
    if (targetPeerId) {
      setUnreadFrom((prev) => {
        if (!prev.has(targetPeerId)) return prev;
        const next = new Set(prev);
        next.delete(targetPeerId);
        return next;
      });
    }
  }, [targetPeerId]);

  // Scroll chat to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load address book contacts whenever the messages tab is active
  useEffect(() => {
    if (tab !== 'messages') return;
    qnkAPI.getAddressBook().then((res: any) => {
      // Handle multiple possible response shapes
      const addrs =
        res?.data?.addresses ??
        res?.data ??
        res?.addresses ??
        (Array.isArray(res) ? res : []);
      setContacts(Array.isArray(addrs) ? addrs : []);
    }).catch((err: any) => {
      console.warn('[ChatScreen] failed to load address book:', err);
    });
  }, [tab]);

  // Init signaling + WebRTC
  useEffect(() => {
    if (!walletAddress) return;

    const getPrivateKey = () => walletSession.getSession()?.privateKey ?? null;
    const signaling = new SignalingService(walletAddress, getPrivateKey);
    signalingRef.current = signaling;

    const getTurnAuthHeader = async (): Promise<string | null> => {
      const session = walletSession.getSession();
      if (!session) return null;
      return generateAuthHeader(session.privateKey, session.address, '/api/v1/turn/credentials');
    };

    const webrtc = new WebRTCManager({
      onStateChange: (peerId, state) => {
        setActiveCall((prev) => {
          if (!prev || prev.peerId !== peerId) return prev;
          if (state === 'failed') return null;
          return { ...prev, state };
        });
        setMeeting((prev) => {
          if (!prev) return prev;
          const calls = new Map(prev.calls);
          const existing = calls.get(peerId);
          if (existing) {
            if (state === 'failed') {
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
        const sessionId = crypto.randomUUID();
        sessionIdsRef.current.set(peerId, sessionId);
        signaling.send(peerId, sessionId, { type: 'call_offer', sdp, call_type: callType });
      },
      onLocalAnswer: (peerId, sdp) => {
        const sessionId = sessionIdsRef.current.get(peerId) ?? null;
        sessionIdsRef.current.delete(peerId);
        signaling.send(peerId, sessionId, { type: 'call_answer', sdp });
      },
    }, getTurnAuthHeader);
    webrtcRef.current = webrtc;

    const unsub = signaling.onMessage((envelope: SignalingEnvelope) => {
      handleIncoming(envelope, signaling, webrtc);
    });

    const unsubDiag = signaling.onDiag((d) => {
      setDiag(d);
      // Treat 'open' as connected immediately, don't wait for pong (which takes up to 20s).
      if (d.wsState === 'open') setIsConnected(true);
      else if (d.wsState === 'closed' || d.wsState === 'error') setIsConnected(false);
    });

    signaling.connect();

    return () => {
      unsub();
      unsubDiag();
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
        // Mark as unread if this conversation isn't currently open
        setUnreadFrom((prev) => {
          if (from === targetPeerIdRef.current) return prev;
          const next = new Set(prev);
          next.add(from);
          return next;
        });
        // Notify App.tsx so it can show a toast when user is on a different screen
        const contactName = contactsRef.current.find((c) => c.address === from)?.label;
        window.dispatchEvent(new CustomEvent('qnk-new-chat-message', {
          detail: { from, content: payload.content, contactName },
        }));
        break;
      }
      case 'call_offer': {
        sessionIdsRef.current.set(from, envelope.session_id);
        const callDetail = { from, sdp: payload.sdp, callType: payload.call_type, sessionId: envelope.session_id };
        setIncomingCall(callDetail);
        window.dispatchEvent(new CustomEvent('qnk-incoming-call', { detail: { from, callType: payload.call_type } }));
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
        // Server-originated CallEnd carries a `reason` field — surface it so the
        // caller sees "peer not connected" instead of a generic spinner that never resolves.
        if (from === 'server' && payload.reason) {
          const reason = payload.reason;
          const friendly = (
            reason === 'peer_not_found'    ? 'The other wallet is not connected to this signaling server right now. Both wallets must be online at the same time.' :
            reason === 'peer_busy'         ? 'The other wallet is already in a call.' :
            reason === 'server_capacity'   ? 'Server is at maximum concurrent calls. Try again in a moment.' :
            reason === 'timeout'           ? 'Call timed out — the other side didn\'t answer.' :
            reason === 'peer_disconnected' ? 'The other wallet just disconnected.' :
            reason === 'rejected'          ? 'Call was declined.' :
            `Call ended by server (${reason}).`
          );
          console.warn('[ChatScreen] server call_end reason=%s', reason);
          setCallError(friendly);
        }
        setActiveCall((prev) => {
          if (prev?.peerId === from || (from === 'server' && prev)) {
            const peer = prev!.peerId;
            recordCallEnd(peer, prev!.callType);
            return null;
          }
          return prev;
        });
        setIncomingCall((prev) => {
          if (prev?.from === from) {
            window.dispatchEvent(new CustomEvent('qnk-incoming-call-cleared'));
            return null;
          }
          return prev;
        });
        break;
      }
      case 'meeting_peers': {
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
        setIsConnected(true);
        break;
    }
  }, []);

  // ── Incoming call handlers ────────────────────────────────────────────────

  const handleAcceptCall = async () => {
    if (!incomingCall || !webrtcRef.current || !signalingRef.current) return;
    const { from, sdp, callType } = incomingCall;
    setIncomingCall(null);
    window.dispatchEvent(new CustomEvent('qnk-incoming-call-cleared'));
    callStartTimeRef.current = Date.now();
    setActiveCall({ peerId: from, callType, state: 'connecting', remoteStream: null, localStream: null });
    setCallError(null);
    setTab('calls');
    try {
      await webrtcRef.current.handleOffer(from, sdp, callType);
      const localStream = webrtcRef.current.getLocalStream(from);
      setActiveCall((prev) => prev?.peerId === from ? { ...prev, localStream } : prev);
    } catch (e: any) {
      setActiveCall(null);
      callStartTimeRef.current = null;
      setCallError(e?.message ?? 'Failed to start call');
    }
  };

  const handleRejectCall = () => {
    if (!incomingCall || !signalingRef.current) return;
    signalingRef.current.send(incomingCall.from, incomingCall.sessionId, { type: 'call_end', reason: 'rejected' });
    setIncomingCall(null);
    window.dispatchEvent(new CustomEvent('qnk-incoming-call-cleared'));
  };

  useEffect(() => {
    const onAccept = () => { handleAcceptCall(); };
    const onReject = () => { handleRejectCall(); };
    window.addEventListener('qnk-accept-call', onAccept);
    window.addEventListener('qnk-reject-call', onReject);
    return () => {
      window.removeEventListener('qnk-accept-call', onAccept);
      window.removeEventListener('qnk-reject-call', onReject);
    };
  }, [incomingCall]);

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
      id: `self-${ts}`, from: walletAddress, to: targetPeerId.trim(), content, timestamp: ts, self: true,
    }]);
    setInputText('');
  };

  const recordCallEnd = useCallback((peerId: string, callType: CallType) => {
    const started = callStartTimeRef.current;
    const duration = started ? Math.round((Date.now() - started) / 1000) : undefined;
    callStartTimeRef.current = null;
    setCallHistory((prev) => [
      { peerId, callType, timestamp: Date.now(), duration },
      ...prev.slice(0, 49),
    ]);
  }, []);

  const startCall = async (callType: CallType) => {
    if (!targetPeerId.trim() || !webrtcRef.current) return;
    const peerId = targetPeerId.trim();
    callStartTimeRef.current = Date.now();
    setActiveCall({ peerId, callType, state: 'connecting', remoteStream: null, localStream: null });
    setCallError(null);
    setTab('calls');
    try {
      await webrtcRef.current.initiateCall(peerId, callType);
      const localStream = webrtcRef.current.getLocalStream(peerId);
      setActiveCall((prev) => prev?.peerId === peerId ? { ...prev, localStream } : prev);
    } catch (e: any) {
      setActiveCall(null);
      callStartTimeRef.current = null;
      setCallError(e?.message ?? 'Failed to start call');
    }
  };

  const hangup = () => {
    if (!activeCall || !webrtcRef.current || !signalingRef.current) return;
    const { peerId, callType } = activeCall;
    webrtcRef.current.hangup(peerId);
    signalingRef.current.send(peerId, null, { type: 'call_end', reason: 'user_hangup' });
    recordCallEnd(peerId, callType);
    setActiveCall(null);
    // Stop AI when call ends
    recognitionRef.current?.stop();
    aiAbortRef.current?.abort();
    setAiEnabled(false);
    setAiTranscript('');
    setAiSuggestion('');
  };

  const askAI = useCallback(async (transcript: string) => {
    aiAbortRef.current?.abort();
    const controller = new AbortController();
    aiAbortRef.current = controller;
    setAiStreaming(true);
    setAiSuggestion('');
    setAiInput('');
    try {
      const { apiBaseUrl } = getConnectionInfo();
      const res = await fetch(`${apiBaseUrl}/ai/call-assist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        setAiSuggestion(`⚠ AI error ${res.status} — make sure Ollama is running on Epsilon.`);
        setAiStreaming(false);
        return;
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (line.startsWith('data:')) {
            try {
              const json = JSON.parse(line.slice(5).trim());
              if (json.content) setAiSuggestion((p) => p + json.content);
            } catch {}
          }
        }
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') setAiSuggestion('⚠ AI unavailable — check Ollama on Epsilon.');
    } finally {
      setAiStreaming(false);
    }
  }, []);

  const toggleAI = useCallback(() => {
    setAiEnabled((prev) => {
      if (prev) {
        recognitionRef.current?.stop();
        recognitionRef.current = null;
        aiAbortRef.current?.abort();
        setAiTranscript('');
        setAiSuggestion('');
        setAiInput('');
        return false;
      }
      // Try Web Speech API as optional enhancement (auto-populates input)
      const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SR) {
        const rec = new SR();
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = 'en-US';
        let fatalError = false;
        rec.onresult = (event: any) => {
          let interim = '';
          let final = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) final += event.results[i][0].transcript;
            else interim += event.results[i][0].transcript;
          }
          const text = final || interim;
          setAiTranscript(text);
          setAiInput(text);
          if (final.trim().length > 4) askAI(final.trim());
        };
        rec.onerror = (e: any) => {
          const errType = e.error ?? 'unknown';
          if (errType === 'no-speech' || errType === 'aborted') return;
          fatalError = true;
          // Speech not working — fall back to text input silently
          recognitionRef.current = null;
          setAiSpeechAvailable(false);
        };
        rec.onend = () => {
          if (fatalError) return;
          setAiEnabled((still) => {
            if (still && recognitionRef.current === rec) {
              try { rec.start(); } catch {}
            }
            return still;
          });
        };
        rec.start();
        recognitionRef.current = rec;
        setAiSpeechAvailable(true);
      }
      return true;
    });
  }, [askAI]);

  const toggleMic = () => {
    setMicOn((prev) => {
      const newVal = !prev;
      const muteStream = (stream: MediaStream | null) => {
        stream?.getAudioTracks().forEach((t) => (t.enabled = newVal));
      };
      if (activeCall) muteStream(activeCall.localStream);
      if (meeting) meeting.calls.forEach((c) => muteStream(c.localStream));
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
      if (meeting) meeting.calls.forEach((c) => muteStream(c.localStream));
      return newVal;
    });
  };

  const joinMeeting = () => {
    const roomId = roomInput.trim();
    if (!roomId || !signalingRef.current) return;
    setMeeting({ roomId, peers: [], calls: new Map() });
    setMeetingHistory((prev) => {
      const filtered = prev.filter((h) => h.roomId !== roomId);
      return [{ roomId, timestamp: Date.now(), peerCount: 0 }, ...filtered].slice(0, 20);
    });
    signalingRef.current.send(null, null, {
      type: 'meeting_join',
      room_id: roomId,
      display_name: displayName,
    });
    setTab('meetings');
  };

  const leaveMeeting = () => {
    if (!meeting || !signalingRef.current || !webrtcRef.current) return;
    const { roomId, peers } = meeting;
    signalingRef.current.send(null, null, { type: 'meeting_leave', room_id: roomId });
    peers.forEach((p) => webrtcRef.current!.hangup(p.peer_id));
    setMeetingHistory((prev) =>
      prev.map((h) => h.roomId === roomId ? { ...h, peerCount: peers.length } : h)
    );
    setMeeting(null);
    setRoomInput('');
  };

  // ── Derived ───────────────────────────────────────────────────────────────

  const selectedContact = contacts.find((c) => c.address === targetPeerId);

  // Build conversation list: group messages by peer, find last message + timestamp
  const conversationMap = new Map<string, { lastMsg: ChatMessage; unread: boolean }>();
  for (const msg of messages) {
    const peer = msg.self ? (msg.to ?? '') : msg.from;
    if (!peer || peer === walletAddress) continue;
    const existing = conversationMap.get(peer);
    if (!existing || msg.timestamp > existing.lastMsg.timestamp) {
      conversationMap.set(peer, {
        lastMsg: msg,
        unread: unreadFrom.has(peer),
      });
    }
  }
  // Ensure contacts with unread show up even if no messages yet in session
  for (const addr of unreadFrom) {
    if (!conversationMap.has(addr)) {
      conversationMap.set(addr, {
        lastMsg: { id: '', from: addr, content: '', timestamp: 0, self: false },
        unread: true,
      });
    }
  }

  // Sorted conversations: unread first, then by last message time desc
  const conversations = [...conversationMap.entries()]
    .sort(([, a], [, b]) => {
      if (a.unread !== b.unread) return a.unread ? -1 : 1;
      return b.lastMsg.timestamp - a.lastMsg.timestamp;
    });

  const contactByAddress = new Map(contacts.map((c) => [c.address, c]));

  const q = contactSearch.toLowerCase();
  const filteredConversations = q
    ? conversations.filter(([addr]) => {
        const c = contactByAddress.get(addr);
        return addr.toLowerCase().includes(q) || (c && c.label.toLowerCase().includes(q));
      })
    : conversations;

  // Legacy: contacts not yet in conversation list (no messages exchanged yet)
  const conversationAddresses = new Set(conversations.map(([addr]) => addr));
  const filteredContactsOnly = contacts
    .filter((c) => {
      if (conversationAddresses.has(c.address)) return false;
      if (!q) return true;
      return c.label.toLowerCase().includes(q) || c.address.toLowerCase().includes(q);
    })
    .sort((a, b) => (b.favorite ? 1 : 0) - (a.favorite ? 1 : 0));

  // Legacy compat for the "or pick a contact" panel below
  const filteredContacts = contacts
    .filter((c) => {
      return !q || c.label.toLowerCase().includes(q) || c.address.toLowerCase().includes(q);
    })
    .sort((a, b) => (b.favorite ? 1 : 0) - (a.favorite ? 1 : 0));

  const sidebarBg = 'rgba(10,8,30,0.95)';
  const sidebarBorder = 'rgba(212,175,55,0.12)';

  // ── Tab definitions ───────────────────────────────────────────────────────

  const tabs: { id: Tab; icon: React.ReactNode; label: string; badge?: number }[] = [
    { id: 'messages', icon: <MessageSquare className="w-4 h-4" />, label: 'Messages' },
    { id: 'calls', icon: <Phone className="w-4 h-4" />, label: 'Calls', badge: activeCall ? 1 : undefined },
    { id: 'meetings', icon: <Users className="w-4 h-4" />, label: 'Meetings', badge: meeting ? 1 : undefined },
    { id: 'groups', icon: <Hash className="w-4 h-4" />, label: 'Groups' },
  ];

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full relative" style={{ minHeight: 0 }}>

      {/* Incoming call consent banner */}
      {incomingCall && (
        <div
          className="flex items-center justify-between px-6 py-3 shrink-0"
          style={{ background: 'rgba(212,175,55,0.15)', borderBottom: '1px solid rgba(212,175,55,0.3)' }}
        >
          <span className="text-amber-200 text-sm font-medium">
            Incoming {incomingCall.callType} call from {incomingCall.from.slice(0, 12)}…
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

      {/* Two-column body */}
      <div className="flex flex-1 min-h-0 overflow-hidden">

        {/* ── Left Sidebar ── */}
        <div
          className="w-60 flex-shrink-0 flex flex-col"
          style={{ background: sidebarBg, borderRight: `1px solid ${sidebarBorder}` }}
        >
          {/* Sidebar header */}
          <div className="px-4 py-3 shrink-0" style={{ borderBottom: `1px solid ${sidebarBorder}` }}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div
                  className="w-7 h-7 rounded-lg flex items-center justify-center"
                  style={{ background: 'linear-gradient(135deg, rgba(212,175,55,0.25), rgba(255,215,0,0.12))' }}
                >
                  <MessageSquare className="w-4 h-4 text-amber-400" />
                </div>
                <span className="text-sm font-bold text-amber-100">Chat & Calls</span>
              </div>
              <div className="flex items-center gap-2">
                <div
                  className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-slate-500'}`}
                  style={isConnected ? { boxShadow: '0 0 5px rgba(74,222,128,0.6)' } : {}}
                />
                <button
                  onClick={() => setShowAddressBook(true)}
                  className="p-1 rounded-lg transition-colors text-amber-400/50 hover:text-amber-300"
                  style={{ border: '1px solid rgba(212,175,55,0.15)' }}
                  title="Address Book"
                >
                  <Users className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>

          {/* Tab navigation */}
          <nav className="px-2 pt-2 pb-1 shrink-0">
            {tabs.map(({ id, icon, label, badge }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className="w-full flex items-center gap-3 px-3 py-2 rounded-lg mb-0.5 text-sm transition-all relative"
                style={tab === id ? {
                  background: 'linear-gradient(135deg, rgba(212,175,55,0.18), rgba(255,215,0,0.08))',
                  border: '1px solid rgba(212,175,55,0.3)',
                  color: '#fef3c7',
                } : {
                  border: '1px solid transparent',
                  color: 'rgba(252,211,77,0.45)',
                }}
              >
                <span style={tab === id ? { color: '#fbbf24' } : {}}>{icon}</span>
                <span className="font-medium">{label}</span>
                {badge !== undefined && (
                  <span
                    className="ml-auto w-4 h-4 text-xs rounded-full flex items-center justify-center font-bold"
                    style={{ background: 'rgba(212,175,55,0.9)', color: '#1a1200' }}
                  >
                    {badge}
                  </span>
                )}
              </button>
            ))}
          </nav>

          {/* Meetings sidebar panel */}
          {tab === 'meetings' && (
            <div className="flex-1 flex flex-col px-3 py-3 gap-3 overflow-y-auto" style={{ borderTop: `1px solid ${sidebarBorder}`, minHeight: 0 }}>
              {meeting ? (
                <div
                  className="rounded-xl p-3 text-center shrink-0"
                  style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}
                >
                  <p className="text-xs font-bold text-green-400 mb-0.5">In meeting</p>
                  <p className="text-[10px] font-mono text-green-400/60 truncate">{meeting.roomId}</p>
                  <p className="text-[10px] text-green-400/50 mt-1">{meeting.peers.length} peer{meeting.peers.length !== 1 ? 's' : ''} connected</p>
                  <button
                    onClick={leaveMeeting}
                    className="mt-2 w-full flex items-center justify-center gap-1 py-1.5 rounded-lg text-xs font-semibold text-red-300 transition-colors"
                    style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.25)' }}
                  >
                    <X className="w-3 h-3" /> Leave
                  </button>
                </div>
              ) : (
                <>
                  <button
                    onClick={() => {
                      const id = randomRoomId();
                      setRoomInput(id);
                      setTimeout(() => {
                        const newMeeting = { roomId: id, peers: [], calls: new Map() };
                        setMeeting(newMeeting);
                        setMeetingHistory((prev) => {
                          const filtered = prev.filter((h) => h.roomId !== id);
                          return [{ roomId: id, timestamp: Date.now(), peerCount: 0 }, ...filtered].slice(0, 20);
                        });
                        signalingRef.current?.send(null, null, { type: 'meeting_join', room_id: id, display_name: displayName });
                      }, 0);
                    }}
                    className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-semibold transition-all shrink-0"
                    style={{
                      background: 'linear-gradient(135deg, rgba(212,175,55,0.35), rgba(255,215,0,0.2))',
                      border: '1.5px solid rgba(212,175,55,0.4)',
                      color: '#fef3c7',
                    }}
                  >
                    <Plus className="w-4 h-4" /> Create Room
                  </button>
                  <div className="flex items-center gap-2 shrink-0">
                    <div className="flex-1 h-px" style={{ background: sidebarBorder }} />
                    <span className="text-[10px] text-amber-400/30 uppercase tracking-widest">or join</span>
                    <div className="flex-1 h-px" style={{ background: sidebarBorder }} />
                  </div>
                  <div className="flex gap-1.5 shrink-0">
                    <input
                      className="flex-1 px-2.5 py-1.5 rounded-lg text-xs text-amber-100 placeholder-amber-300/25 outline-none"
                      style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(212,175,55,0.15)' }}
                      placeholder="Room ID…"
                      value={roomInput}
                      onChange={(e) => setRoomInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') joinMeeting(); }}
                    />
                    <button
                      onClick={joinMeeting}
                      className="p-1.5 rounded-lg flex-shrink-0 transition-colors"
                      style={{
                        background: 'rgba(212,175,55,0.15)',
                        border: '1px solid rgba(212,175,55,0.25)',
                        color: '#d4af37',
                      }}
                    >
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </>
              )}

              {meetingHistory.length > 0 && !meeting && (
                <>
                  <p className="text-[9px] uppercase tracking-widest text-amber-400/30 px-1 shrink-0">Recent Rooms</p>
                  {meetingHistory.map((entry, i) => {
                    const timeStr = new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    return (
                      <button
                        key={i}
                        onClick={() => {
                          if (!signalingRef.current) return;
                          setRoomInput(entry.roomId);
                          setMeeting({ roomId: entry.roomId, peers: [], calls: new Map() });
                          setMeetingHistory((prev) => {
                            const filtered = prev.filter((h) => h.roomId !== entry.roomId);
                            return [{ roomId: entry.roomId, timestamp: Date.now(), peerCount: 0 }, ...filtered].slice(0, 20);
                          });
                          signalingRef.current.send(null, null, { type: 'meeting_join', room_id: entry.roomId, display_name: displayName });
                          setTab('meetings');
                        }}
                        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-left transition-all shrink-0"
                        style={{ border: '1px solid rgba(212,175,55,0.1)', background: 'rgba(255,255,255,0.02)' }}
                        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(212,175,55,0.06)'; }}
                        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)'; }}
                      >
                        <Hash className="w-3 h-3 text-amber-400/40 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-[10px] font-mono truncate" style={{ color: 'rgba(254,243,199,0.6)' }}>{entry.roomId}</p>
                          <p className="text-[9px]" style={{ color: 'rgba(212,175,55,0.35)' }}>
                            {timeStr}{entry.peerCount > 0 ? ` · ${entry.peerCount}p` : ''}
                          </p>
                        </div>
                        <ChevronRight className="w-3 h-3 text-amber-400/25 flex-shrink-0" />
                      </button>
                    );
                  })}
                </>
              )}
            </div>
          )}

          {/* Calls sidebar panel */}
          {tab === 'calls' && (
            <div className="flex-1 flex flex-col px-3 py-3 gap-2 overflow-y-auto" style={{ borderTop: `1px solid ${sidebarBorder}`, minHeight: 0 }}>
              {callError && (
                <div
                  className="w-full rounded-xl p-3 shrink-0"
                  style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)' }}
                >
                  <p className="text-xs font-bold text-red-400 mb-1">Call failed</p>
                  <p className="text-[10px] text-red-300/80 leading-relaxed">{callError}</p>
                  <button
                    className="mt-2 text-[9px] text-red-400/60 hover:text-red-400 underline"
                    onClick={() => setCallError(null)}
                  >
                    Dismiss
                  </button>
                </div>
              )}

              {/* ── Connection Diagnostics (debug aid) ────────────────────── */}
              <div
                className="w-full rounded-xl shrink-0 text-[10px]"
                style={{ background: 'rgba(15,23,42,0.4)', border: '1px solid rgba(212,175,55,0.15)' }}
              >
                <button
                  onClick={() => {
                    const next = !showDiag;
                    setShowDiag(next);
                    if (next) signalingRef.current?.pollServerDiag().then(setServerPeers);
                  }}
                  className="w-full px-3 py-2 flex items-center justify-between text-amber-300/70 hover:text-amber-300"
                >
                  <span className="font-bold">Connection diagnostics</span>
                  <span className="opacity-60">{showDiag ? '▾' : '▸'}</span>
                </button>
                {showDiag && !diag && (
                  <div className="px-3 pb-3 text-amber-300/70 font-mono">
                    (signaling not initialised — is your wallet unlocked?)
                  </div>
                )}
                {showDiag && diag && (
                  <div className="px-3 pb-3 space-y-1 font-mono text-amber-200/70 leading-snug">
                    <div>my peer_id: <span className="text-amber-300">{truncAddr(diag.peerId) || '(none — wallet not loaded)'}</span></div>
                    <div>target:     <span className="text-amber-300">{targetPeerId ? truncAddr(targetPeerId) : '(none selected)'}</span></div>
                    <div>ws state:   <span className={
                      diag.wsState === 'open' ? 'text-green-400' :
                      diag.wsState === 'connecting' ? 'text-amber-300' :
                      'text-red-400'
                    }>{diag.wsState}</span>{diag.reconnectAttempts > 0 && ` (retry #${diag.reconnectAttempts})`}</div>
                    <div>ws url:     <span className="text-amber-300/60 break-all">{diag.wsUrl || '(not built yet)'}</span></div>
                    <div>auth:       <span className={diag.authStatus === 'ok' ? 'text-green-400' : 'text-red-400'}>
                      {diag.authStatus}
                    </span>{diag.authError && <span className="text-red-300/70"> — {diag.authError}</span>}</div>
                    {diag.lastCloseCode !== null && (
                      <div>last close: code={diag.lastCloseCode} reason="{diag.lastCloseReason}"</div>
                    )}
                    {diag.lastOutgoing && (
                      <div>last sent:  {diag.lastOutgoing.type} → {diag.lastOutgoing.to ? truncAddr(diag.lastOutgoing.to) : '(broadcast)'} {diag.lastOutgoing.sid && `sid=${diag.lastOutgoing.sid.slice(0, 8)}`}</div>
                    )}
                    {diag.lastIncoming && (
                      <div>last recv:  {diag.lastIncoming.type} ← {diag.lastIncoming.from === 'server' ? 'server' : truncAddr(diag.lastIncoming.from)}{diag.lastIncoming.reason && ` reason=${diag.lastIncoming.reason}`}</div>
                    )}
                    <div className="pt-2 border-t border-amber-400/10">
                      <button
                        className="text-amber-400/70 hover:text-amber-300 underline"
                        onClick={() => signalingRef.current?.pollServerDiag().then(setServerPeers)}
                      >
                        ↻ refresh server peer list
                      </button>
                    </div>
                    {serverPeers && (
                      <>
                        <div>server peer_count: <span className="text-amber-300">{serverPeers.peer_count}</span></div>
                        <div>active calls:     <span className="text-amber-300">{serverPeers.active_call_count}</span></div>
                        <div>peers (truncated):</div>
                        <ul className="pl-3 space-y-0.5 max-h-32 overflow-y-auto">
                          {serverPeers.peers.length === 0 ? (
                            <li className="text-red-400">(none — no other wallets are connected to this backend)</li>
                          ) : (
                            serverPeers.peers.map((p, i) => <li key={i} className="text-amber-200/60">{p}</li>)
                          )}
                        </ul>
                        {targetPeerId && serverPeers.peer_count > 0 && (() => {
                          // Server's short_peer() format is "<first 8>…<last 6>".
                          // Match without case sensitivity since the server stores the original case.
                          const tLow = targetPeerId.toLowerCase();
                          const tPrefix = tLow.slice(0, 8);
                          const tSuffix = tLow.slice(-6);
                          const hit = serverPeers.peers.some((p) => {
                            const pLow = p.toLowerCase();
                            return pLow.startsWith(tPrefix) && pLow.endsWith(tSuffix);
                          });
                          return (
                            <div className="pt-1">
                              target {truncAddr(targetPeerId)}{' '}
                              {hit ? (
                                <span className="text-green-400">✓ appears connected to this backend</span>
                              ) : (
                                <span className="text-red-400">✗ NOT seen on this backend (may be on a different load-balanced server, or not connected at all)</span>
                              )}
                            </div>
                          );
                        })()}
                      </>
                    )}
                  </div>
                )}
              </div>


              {activeCall ? (
                <div
                  className="w-full rounded-xl p-3 text-center shrink-0"
                  style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}
                >
                  <p className="text-xs font-bold text-green-400 mb-0.5">Active call</p>
                  <p className="text-[10px] font-mono text-green-400/60 truncate">{shortId(activeCall.peerId)}</p>
                  <span className={`text-[10px] px-2 py-0.5 rounded-full mt-1 inline-block ${
                    activeCall.state === 'connected' ? 'bg-green-500/20 text-green-300' :
                    activeCall.state === 'disconnected' ? 'bg-red-500/20 text-red-300' :
                    'bg-amber-500/20 text-amber-300'
                  }`}>
                    {activeCall.state === 'connected' ? 'Connected' :
                     activeCall.state === 'disconnected' ? 'Reconnecting…' : 'Connecting…'}
                  </span>
                </div>
              ) : !callError && callHistory.length === 0 ? (
                <div className="flex flex-col items-center justify-center flex-1 text-center">
                  <Phone className="w-8 h-8 text-amber-400/20 mx-auto mb-2" />
                  <p className="text-xs text-amber-300/30">No recent calls</p>
                </div>
              ) : null}

              {callHistory.length > 0 && (
                <>
                  <p className="text-[9px] uppercase tracking-widest text-amber-400/30 px-1 shrink-0">Recent Calls</p>
                  {callHistory.map((entry, i) => {
                    const contact = contactByAddress.get(entry.peerId);
                    const label = contact?.label ?? truncAddr(entry.peerId);
                    const timeStr = new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    const durStr = entry.duration != null
                      ? entry.duration >= 60
                        ? `${Math.floor(entry.duration / 60)}m ${entry.duration % 60}s`
                        : `${entry.duration}s`
                      : null;
                    return (
                      <button
                        key={i}
                        onClick={() => { setTargetPeerId(entry.peerId); setTab('messages'); }}
                        className="w-full flex items-center gap-2 px-2 py-2 rounded-lg text-left transition-all shrink-0"
                        style={{ border: '1px solid transparent' }}
                        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(212,175,55,0.06)'; }}
                        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = ''; }}
                      >
                        <div
                          className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
                          style={{ background: 'rgba(212,175,55,0.12)' }}
                        >
                          {entry.callType === 'video'
                            ? <Video className="w-3 h-3 text-amber-400/60" />
                            : <Phone className="w-3 h-3 text-amber-400/60" />}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold truncate" style={{ color: 'rgba(254,243,199,0.75)' }}>{label}</p>
                          <p className="text-[10px]" style={{ color: 'rgba(212,175,55,0.4)' }}>
                            {timeStr}{durStr ? ` · ${durStr}` : ''}
                          </p>
                        </div>
                      </button>
                    );
                  })}
                </>
              )}
            </div>
          )}

          {/* Conversation list — only in messages tab */}
          {tab === 'messages' && (
            <>
              <div className="px-3 py-2 shrink-0" style={{ borderTop: `1px solid ${sidebarBorder}` }}>
                <div className="relative">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-amber-400/40" />
                  <input
                    className="w-full pl-8 pr-3 py-1.5 rounded-lg text-xs text-amber-100 placeholder-amber-300/30 outline-none"
                    style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(212,175,55,0.15)' }}
                    placeholder="Search…"
                    value={contactSearch}
                    onChange={(e) => setContactSearch(e.target.value)}
                  />
                </div>
              </div>

              <div className="flex-1 overflow-y-auto px-2 pb-2" style={{ minHeight: 0 }}>

                {filteredConversations.length === 0 && filteredContactsOnly.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-8 text-center gap-2">
                    <Users className="w-6 h-6 text-amber-400/20" />
                    <p className="text-xs text-amber-300/30">
                      {contactSearch ? 'No matches' : 'No conversations yet'}
                    </p>
                    <p className="text-[10px] text-amber-300/20">Start chatting or add contacts</p>
                  </div>
                ) : (
                  <>
                    {filteredConversations.length > 0 && (
                      <>
                        <p className="text-[9px] uppercase tracking-widest text-amber-400/30 px-2 py-1">Conversations</p>
                        {filteredConversations.map(([addr, { lastMsg, unread }]) => {
                          const isSelected = addr === targetPeerId;
                          const contact = contactByAddress.get(addr);
                          const displayLabel = contact?.label ?? truncAddr(addr);
                          const initials = (contact?.label ?? addr).slice(0, 2).toUpperCase();
                          const timeStr = lastMsg.timestamp > 0
                            ? new Date(lastMsg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                            : '';
                          return (
                            <button
                              key={addr}
                              onClick={() => setTargetPeerId(addr)}
                              className="w-full flex items-center gap-2 px-2 py-2 rounded-lg mb-0.5 text-left transition-all"
                              style={isSelected ? {
                                background: 'linear-gradient(135deg, rgba(212,175,55,0.15), rgba(255,215,0,0.07))',
                                border: '1px solid rgba(212,175,55,0.3)',
                              } : unread ? {
                                background: 'rgba(212,175,55,0.06)',
                                border: '1px solid rgba(212,175,55,0.18)',
                              } : {
                                border: '1px solid transparent',
                              }}
                            >
                              <div className="relative flex-shrink-0">
                                <div
                                  className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
                                  style={{
                                    background: isSelected
                                      ? 'linear-gradient(135deg, rgba(212,175,55,0.5), rgba(255,165,0,0.3))'
                                      : 'rgba(212,175,55,0.15)',
                                    color: '#d4af37',
                                  }}
                                >
                                  {initials}
                                </div>
                                {unread && (
                                  <span
                                    className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-amber-400"
                                  />
                                )}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between gap-1">
                                  <span
                                    className="text-xs font-semibold truncate"
                                    style={{ color: isSelected || unread ? '#fef3c7' : 'rgba(254,243,199,0.7)' }}
                                  >
                                    {contact?.favorite && (
                                      <Star className="inline w-2.5 h-2.5 text-amber-400 mr-0.5 -mt-0.5" style={{ fill: 'currentColor' }} />
                                    )}
                                    {displayLabel}
                                  </span>
                                  {timeStr && (
                                    <span className="text-[9px] text-amber-400/35 flex-shrink-0">{timeStr}</span>
                                  )}
                                </div>
                                {lastMsg.content ? (
                                  <p
                                    className="text-[10px] truncate"
                                    style={{ color: unread ? 'rgba(254,243,199,0.55)' : 'rgba(212,175,55,0.35)' }}
                                  >
                                    {lastMsg.self ? 'You: ' : ''}{lastMsg.content}
                                  </p>
                                ) : (
                                  <p className="text-[10px] font-mono truncate" style={{ color: 'rgba(212,175,55,0.3)' }}>
                                    {truncAddr(addr)}
                                  </p>
                                )}
                              </div>
                            </button>
                          );
                        })}
                      </>
                    )}

                    {filteredContactsOnly.length > 0 && (
                      <>
                        <p className="text-[9px] uppercase tracking-widest text-amber-400/30 px-2 py-1 mt-1">Contacts</p>
                        {filteredContactsOnly.map((contact) => {
                          const isSelected = contact.address === targetPeerId;
                          const initials = contact.label.slice(0, 2).toUpperCase();
                          return (
                            <button
                              key={contact.id}
                              onClick={() => setTargetPeerId(contact.address)}
                              className="w-full flex items-center gap-2 px-2 py-2 rounded-lg mb-0.5 text-left transition-all"
                              style={isSelected ? {
                                background: 'linear-gradient(135deg, rgba(212,175,55,0.15), rgba(255,215,0,0.07))',
                                border: '1px solid rgba(212,175,55,0.3)',
                              } : {
                                border: '1px solid transparent',
                              }}
                            >
                              <div
                                className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                                style={{
                                  background: isSelected
                                    ? 'linear-gradient(135deg, rgba(212,175,55,0.5), rgba(255,165,0,0.3))'
                                    : 'rgba(212,175,55,0.15)',
                                  color: '#d4af37',
                                }}
                              >
                                {initials}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-1">
                                  {contact.favorite && (
                                    <Star className="w-2.5 h-2.5 text-amber-400 flex-shrink-0" style={{ fill: 'currentColor' }} />
                                  )}
                                  <span className="text-xs font-semibold truncate" style={{ color: isSelected ? '#fef3c7' : 'rgba(254,243,199,0.7)' }}>
                                    {contact.label}
                                  </span>
                                </div>
                                <p className="text-[10px] font-mono truncate" style={{ color: 'rgba(212,175,55,0.35)' }}>
                                  {truncAddr(contact.address)}
                                </p>
                              </div>
                            </button>
                          );
                        })}
                      </>
                    )}
                  </>
                )}
              </div>

            </>
          )}

          {/* Sidebar footer */}
          <div
            className="px-3 py-2 shrink-0 flex items-center justify-between"
            style={{ borderTop: `1px solid ${sidebarBorder}` }}
          >
            <div className="flex items-center gap-1.5">
              <Lock className="w-3 h-3 text-amber-400/30" />
              <span className="text-[10px] text-amber-300/30">E2E encrypted</span>
            </div>
            <button
              onClick={() => setShowAddressBook(true)}
              className="flex items-center gap-1 text-[10px] font-semibold text-amber-400/50 hover:text-amber-300 transition-colors"
            >
              <Users className="w-3 h-3" />
              Address Book
            </button>
          </div>
        </div>

        {/* ── Right Panel ── */}
        <div className="flex-1 flex flex-col min-w-0" style={{ minHeight: 0 }}>
          <AnimatePresence mode="wait">

            {/* ── Messages Tab ── */}
            {tab === 'messages' && (
              <motion.div
                key="messages"
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                className="flex flex-col h-full"
                style={{ minHeight: 0 }}
              >
                {/* Conversation header */}
                <div
                  className="flex items-center justify-between px-5 py-3 shrink-0"
                  style={{
                    borderBottom: '1px solid rgba(212,175,55,0.1)',
                    background: 'rgba(15,23,42,0.5)',
                  }}
                >
                  {targetPeerId ? (
                    <div className="flex items-center gap-3 min-w-0">
                      <div
                        className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                        style={{ background: 'linear-gradient(135deg, rgba(212,175,55,0.35), rgba(255,165,0,0.2))', color: '#d4af37' }}
                      >
                        {(selectedContact?.label || targetPeerId).slice(0, 2).toUpperCase()}
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-amber-100 truncate">
                          {selectedContact?.label || truncAddr(targetPeerId)}
                        </p>
                        {selectedContact && (
                          <p className="text-[10px] font-mono text-amber-400/40 truncate">
                            {truncAddr(targetPeerId)}
                          </p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-amber-300/40">
                      <MessageSquare className="w-4 h-4" />
                      <span className="text-sm">Select a contact or enter an address</span>
                    </div>
                  )}

                  {/* Call buttons */}
                  {targetPeerId && (
                    <div className="flex items-center gap-1.5 shrink-0">
                      <button
                        onClick={() => startCall('audio')}
                        className="p-2 rounded-lg text-amber-300/50 hover:text-amber-300 transition-colors"
                        style={{ border: '1px solid rgba(212,175,55,0.15)' }}
                        title="Voice call"
                      >
                        <Phone className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => startCall('video')}
                        className="p-2 rounded-lg text-amber-300/50 hover:text-amber-300 transition-colors"
                        style={{ border: '1px solid rgba(212,175,55,0.15)' }}
                        title="Video call"
                      >
                        <Video className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </div>

                {/* No conversation selected — big picker UI */}
                {!targetPeerId && (
                  <div className="flex-1 flex flex-col items-center justify-center px-8 py-10" style={{ minHeight: 0 }}>
                    <div
                      className="w-full max-w-md rounded-3xl p-8"
                      style={{
                        background: 'linear-gradient(135deg, rgba(20,14,50,0.98) 0%, rgba(35,20,65,0.98) 100%)',
                        border: '1.5px solid rgba(212,175,55,0.25)',
                        boxShadow: '0 0 48px rgba(212,175,55,0.08), inset 0 0 24px rgba(212,175,55,0.03)',
                      }}
                    >
                      <div className="flex items-center gap-3 mb-6">
                        <div
                          className="w-10 h-10 rounded-xl flex items-center justify-center"
                          style={{ background: 'linear-gradient(135deg, rgba(212,175,55,0.25), rgba(255,215,0,0.12))' }}
                        >
                          <MessageSquare className="w-5 h-5 text-amber-400" />
                        </div>
                        <div>
                          <h2 className="text-base font-bold text-amber-100">New Conversation</h2>
                          <p className="text-xs text-amber-400/50">Enter a wallet address to start chatting</p>
                        </div>
                      </div>

                      {/* Address input */}
                      <div className="mb-4">
                        <label className="text-xs font-semibold text-amber-400/60 uppercase tracking-widest mb-2 block">
                          Wallet Address
                        </label>
                        <div className="flex gap-2">
                          <input
                            className="flex-1 px-4 py-3 rounded-xl text-sm text-amber-100 placeholder-amber-300/25 outline-none font-mono"
                            style={{ background: 'rgba(255,255,255,0.05)', border: '1.5px solid rgba(212,175,55,0.2)' }}
                            placeholder="0x…"
                            value={addressInput}
                            onChange={(e) => setAddressInput(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' && addressInput.trim()) {
                                setTargetPeerId(addressInput.trim());
                                setAddressInput('');
                              }
                            }}
                            autoFocus
                          />
                          <button
                            onClick={() => {
                              if (addressInput.trim()) {
                                setTargetPeerId(addressInput.trim());
                                setAddressInput('');
                              }
                            }}
                            className="px-4 py-3 rounded-xl font-semibold text-sm transition-all"
                            style={{
                              background: addressInput.trim()
                                ? 'linear-gradient(135deg, rgba(212,175,55,0.5), rgba(255,215,0,0.3))'
                                : 'rgba(212,175,55,0.1)',
                              border: '1.5px solid rgba(212,175,55,0.35)',
                              color: addressInput.trim() ? '#fef3c7' : 'rgba(212,175,55,0.4)',
                            }}
                            title="Start chat"
                          >
                            <Send className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => {
                              if (addressInput.trim()) {
                                setTargetPeerId(addressInput.trim());
                                setAddressInput('');
                                startCall('audio');
                              }
                            }}
                            className="px-3 py-3 rounded-xl transition-all"
                            style={{
                              background: addressInput.trim() ? 'rgba(212,175,55,0.15)' : 'rgba(212,175,55,0.06)',
                              border: '1.5px solid rgba(212,175,55,0.25)',
                              color: addressInput.trim() ? '#d4af37' : 'rgba(212,175,55,0.3)',
                            }}
                            title="Voice call"
                          >
                            <Phone className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => {
                              if (addressInput.trim()) {
                                setTargetPeerId(addressInput.trim());
                                setAddressInput('');
                                startCall('video');
                              }
                            }}
                            className="px-3 py-3 rounded-xl transition-all"
                            style={{
                              background: addressInput.trim() ? 'rgba(212,175,55,0.15)' : 'rgba(212,175,55,0.06)',
                              border: '1.5px solid rgba(212,175,55,0.25)',
                              color: addressInput.trim() ? '#d4af37' : 'rgba(212,175,55,0.3)',
                            }}
                            title="Video call"
                          >
                            <Video className="w-4 h-4" />
                          </button>
                        </div>
                      </div>

                      {/* Or pick from contacts */}
                      {contacts.length > 0 && (
                        <>
                          <div className="flex items-center gap-3 my-5">
                            <div className="flex-1 h-px" style={{ background: 'rgba(212,175,55,0.12)' }} />
                            <span className="text-[10px] text-amber-400/30 uppercase tracking-widest">or pick a contact</span>
                            <div className="flex-1 h-px" style={{ background: 'rgba(212,175,55,0.12)' }} />
                          </div>
                          <div className="space-y-1.5 max-h-52 overflow-y-auto pr-1">
                            {contacts
                              .sort((a, b) => (b.favorite ? 1 : 0) - (a.favorite ? 1 : 0))
                              .map((c) => (
                                <button
                                  key={c.id}
                                  onClick={() => setTargetPeerId(c.address)}
                                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-all group"
                                  style={{ border: '1px solid rgba(212,175,55,0.08)', background: 'rgba(255,255,255,0.02)' }}
                                  onMouseEnter={(e) => {
                                    (e.currentTarget as HTMLElement).style.background = 'rgba(212,175,55,0.08)';
                                    (e.currentTarget as HTMLElement).style.borderColor = 'rgba(212,175,55,0.22)';
                                  }}
                                  onMouseLeave={(e) => {
                                    (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)';
                                    (e.currentTarget as HTMLElement).style.borderColor = 'rgba(212,175,55,0.08)';
                                  }}
                                >
                                  <div
                                    className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
                                    style={{ background: 'rgba(212,175,55,0.15)', color: '#d4af37' }}
                                  >
                                    {c.label.slice(0, 2).toUpperCase()}
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-1.5">
                                      {c.favorite && <Star className="w-2.5 h-2.5 text-amber-400 flex-shrink-0" style={{ fill: 'currentColor' }} />}
                                      <span className="text-sm font-semibold text-amber-100 truncate">{c.label}</span>
                                    </div>
                                    <p className="text-[10px] font-mono text-amber-400/35 truncate">{truncAddr(c.address)}</p>
                                  </div>
                                  <ChevronRight className="w-4 h-4 text-amber-400/25 group-hover:text-amber-400/60 transition-colors flex-shrink-0" />
                                </button>
                              ))}
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                )}

                {/* Message list */}
                {targetPeerId && (
                <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3" style={{ minHeight: 0 }}>
                  {messages.filter((m) => m.from === targetPeerId || (m.self && m.to === targetPeerId)).length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full gap-3 text-amber-300/30">
                      <Shield className="w-10 h-10" />
                      <div className="text-center">
                        <p className="text-sm">No messages yet</p>
                        <p className="text-xs mt-1">Say hello to {selectedContact?.label || truncAddr(targetPeerId)}</p>
                      </div>
                    </div>
                  ) : (
                    messages.filter((m) => m.from === targetPeerId || (m.self && m.to === targetPeerId)).map((msg) => (
                      <div key={msg.id} className={`flex ${msg.self ? 'justify-end' : 'justify-start'}`}>
                        <div
                          className="max-w-xs lg:max-w-md px-4 py-2 rounded-2xl text-sm"
                          style={msg.self ? {
                            background: 'linear-gradient(135deg, rgba(212,175,55,0.3), rgba(255,215,0,0.18))',
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
                    ))
                  )}
                  <div ref={messagesEndRef} />
                </div>
                )}

                {/* Input bar */}
                {targetPeerId && (
                  <div
                    className="flex gap-2 px-5 py-3 border-t shrink-0"
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
                )}
              </motion.div>
            )}

            {/* ── Calls Tab ── */}
            {tab === 'calls' && (
              <motion.div
                key="calls"
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                className="flex flex-col h-full p-6 gap-4"
                style={{ minHeight: 0 }}
              >
                {activeCall ? (
                  <div className="flex flex-col gap-4 h-full">
                    <div className="flex items-center justify-between shrink-0">
                      <div>
                        <p className="text-xs text-amber-300/60">Call with</p>
                        <p className="font-bold text-amber-100">{shortId(activeCall.peerId)}</p>
                      </div>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        activeCall.state === 'connected'
                          ? 'bg-green-500/20 text-green-300'
                          : activeCall.state === 'disconnected'
                          ? 'bg-red-500/20 text-red-300'
                          : 'bg-amber-500/20 text-amber-300'
                      }`}>
                        {activeCall.state === 'connected' ? 'Connected' :
                         activeCall.state === 'disconnected' ? 'Reconnecting…' : 'Connecting…'}
                      </span>
                    </div>

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

                      {/* AI assistant overlay panel */}
                      <AnimatePresence>
                        {aiEnabled && (
                          <motion.div
                            initial={{ opacity: 0, y: 16 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 16 }}
                            transition={{ type: 'spring', damping: 22, stiffness: 260 }}
                            className="absolute bottom-0 left-0 right-0 p-3"
                            style={{
                              background: 'linear-gradient(to top, rgba(8,5,28,0.97) 0%, rgba(8,5,28,0.85) 80%, transparent 100%)',
                            }}
                          >
                            <div
                              className="rounded-2xl p-3.5"
                              style={{
                                background: 'linear-gradient(135deg, rgba(20,14,50,0.98), rgba(35,20,65,0.96))',
                                border: '1.5px solid rgba(139,92,246,0.35)',
                                boxShadow: '0 0 24px rgba(139,92,246,0.15)',
                              }}
                            >
                              {/* Header */}
                              <div className="flex items-center gap-2 mb-2.5">
                                <div
                                  className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0"
                                  style={{ background: 'linear-gradient(135deg, rgba(139,92,246,0.4), rgba(109,40,217,0.25))' }}
                                >
                                  <Bot className="w-3.5 h-3.5 text-violet-300" />
                                </div>
                                <span className="text-xs font-bold text-violet-300">Gemma 4 Call Assistant</span>
                                {aiStreaming && (
                                  <div className="flex gap-0.5 ml-auto">
                                    {[0,1,2].map((i) => (
                                      <motion.div
                                        key={i}
                                        className="w-1.5 h-1.5 rounded-full bg-violet-400"
                                        animate={{ opacity: [0.3, 1, 0.3] }}
                                        transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                                      />
                                    ))}
                                  </div>
                                )}
                              </div>

                              {/* Text input — type what was said to get suggestions */}
                              <div className="mb-2 flex gap-2">
                                <input
                                  className="flex-1 px-3 py-2 rounded-xl text-xs text-slate-200 placeholder-slate-500 outline-none"
                                  style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(139,92,246,0.25)' }}
                                  placeholder={aiSpeechAvailable ? 'Listening… or type here' : 'Type what was said…'}
                                  value={aiInput}
                                  onChange={(e) => setAiInput(e.target.value)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !aiStreaming && aiInput.trim()) {
                                      e.preventDefault();
                                      setAiTranscript(aiInput.trim());
                                      askAI(aiInput.trim());
                                    }
                                  }}
                                />
                                <button
                                  onClick={() => {
                                    if (aiInput.trim()) {
                                      setAiTranscript(aiInput.trim());
                                      askAI(aiInput.trim());
                                    }
                                  }}
                                  disabled={aiStreaming || !aiInput.trim()}
                                  className="px-3 py-2 rounded-xl text-xs font-semibold transition-all flex-shrink-0"
                                  style={{
                                    background: aiInput.trim() ? 'rgba(139,92,246,0.4)' : 'rgba(139,92,246,0.1)',
                                    border: '1px solid rgba(139,92,246,0.35)',
                                    color: aiInput.trim() ? '#ddd6fe' : 'rgba(167,139,250,0.35)',
                                  }}
                                >
                                  {aiStreaming ? '…' : 'Ask'}
                                </button>
                              </div>

                              {/* AI suggestion */}
                              {(aiSuggestion || aiStreaming) && (
                                <div className="px-3 py-2 rounded-lg" style={{ background: 'rgba(139,92,246,0.08)', border: '1px solid rgba(139,92,246,0.2)' }}>
                                  <p className="text-[10px] text-violet-400/70 mb-0.5 uppercase tracking-wider">
                                    {aiTranscript && <span className="normal-case text-slate-500 mr-1">re: "{aiTranscript.slice(0, 30)}{aiTranscript.length > 30 ? '…' : ''}" —</span>}
                                    Suggestion
                                  </p>
                                  <p className="text-xs text-violet-100 leading-relaxed">
                                    {aiSuggestion || <span className="text-violet-400/40">Thinking…</span>}
                                  </p>
                                </div>
                              )}

                              {!aiSuggestion && !aiStreaming && (
                                <p className="text-[10px] text-violet-300/35 text-center">
                                  {aiSpeechAvailable ? 'Speak or type • Enter to ask Gemma 4' : 'Type what was said • Enter to ask Gemma 4'}
                                </p>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

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
                      {/* 4K quality toggle — video calls only */}
                      {activeCall.callType === 'video' && (
                        <button
                          onClick={() => {
                            const next: '4k' | 'hd' = videoQuality === 'hd' ? '4k' : 'hd';
                            setVideoQuality(next);
                            webrtcRef.current?.setVideoQuality(activeCall.peerId, next);
                          }}
                          title={videoQuality === 'hd' ? 'Switch to 4K' : 'Switch to HD'}
                          className="w-12 h-12 rounded-full flex items-center justify-center text-[11px] font-bold transition-all"
                          style={{
                            background: videoQuality === '4k'
                              ? 'linear-gradient(135deg, rgba(59,130,246,0.6), rgba(37,99,235,0.4))'
                              : 'rgba(255,255,255,0.08)',
                            border: `1.5px solid ${videoQuality === '4k' ? 'rgba(59,130,246,0.7)' : 'rgba(255,255,255,0.12)'}`,
                            color: videoQuality === '4k' ? '#93c5fd' : 'rgba(255,255,255,0.5)',
                          }}
                        >
                          {videoQuality === '4k' ? '4K' : 'HD'}
                        </button>
                      )}
                      {/* AI toggle button */}
                      <motion.button
                        onClick={toggleAI}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="w-12 h-12 rounded-full flex items-center justify-center transition-all relative"
                        style={{
                          background: aiEnabled
                            ? 'linear-gradient(135deg, rgba(139,92,246,0.6), rgba(109,40,217,0.4))'
                            : 'rgba(255,255,255,0.08)',
                          border: `1.5px solid ${aiEnabled ? 'rgba(139,92,246,0.7)' : 'rgba(255,255,255,0.12)'}`,
                          boxShadow: aiEnabled ? '0 0 16px rgba(139,92,246,0.5)' : 'none',
                        }}
                        title={aiEnabled ? 'Turn off AI assistant' : 'Turn on AI assistant (Gemma 4)'}
                      >
                        <Bot className={`w-5 h-5 ${aiEnabled ? 'text-violet-200' : 'text-white/50'}`} />
                        {aiEnabled && (
                          <motion.div
                            className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-violet-400"
                            animate={{ scale: [1, 1.3, 1] }}
                            transition={{ duration: 2, repeat: Infinity }}
                          />
                        )}
                      </motion.button>
                      <button
                        onClick={hangup}
                        className="w-14 h-14 rounded-full flex items-center justify-center"
                        style={{ background: 'rgba(239,68,68,0.8)', border: '1.5px solid rgba(239,68,68,0.6)' }}
                      >
                        <PhoneOff className="w-6 h-6 text-white" />
                      </button>
                    </div>
                  </div>
                ) : callError ? (
                  <div className="flex flex-col items-center justify-center h-full gap-4 px-4">
                    <div
                      className="w-full rounded-2xl p-5 text-center"
                      style={{ background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.2)' }}
                    >
                      <PhoneOff className="w-8 h-8 text-red-400/60 mx-auto mb-3" />
                      <p className="text-sm font-semibold text-red-300 mb-2">Call failed</p>
                      <p className="text-xs text-red-300/70 leading-relaxed">{callError}</p>
                      <button
                        className="mt-4 text-xs text-red-400/60 hover:text-red-400 underline"
                        onClick={() => setCallError(null)}
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                ) : callHistory.length > 0 ? (
                  <div className="flex flex-col h-full overflow-y-auto px-2 py-4" style={{ minHeight: 0 }}>
                    <p className="text-xs font-semibold text-amber-300/50 px-3 mb-3">Recent Calls</p>
                    <div className="space-y-1">
                      {callHistory.map((entry, i) => {
                        const contact = contactByAddress.get(entry.peerId);
                        const label = contact?.label ?? entry.peerId;
                        const dateStr = new Date(entry.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });
                        const timeStr = new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                        const durStr = entry.duration != null
                          ? entry.duration >= 3600
                            ? `${Math.floor(entry.duration / 3600)}h ${Math.floor((entry.duration % 3600) / 60)}m`
                            : entry.duration >= 60
                            ? `${Math.floor(entry.duration / 60)}m ${entry.duration % 60}s`
                            : `${entry.duration}s`
                          : null;
                        return (
                          <div
                            key={i}
                            className="flex items-center gap-3 px-3 py-3 rounded-xl"
                            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(212,175,55,0.08)' }}
                          >
                            <div
                              className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
                              style={{ background: 'rgba(212,175,55,0.12)' }}
                            >
                              {entry.callType === 'video'
                                ? <Video className="w-4 h-4 text-amber-400/60" />
                                : <Phone className="w-4 h-4 text-amber-400/60" />}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-semibold text-amber-100 truncate">{label}</p>
                              <p className="text-xs font-mono text-amber-400/40 truncate">{truncAddr(entry.peerId)}</p>
                              <p className="text-xs text-amber-300/40 mt-0.5">
                                {dateStr} {timeStr}
                                {durStr && <span className="ml-2 text-amber-400/50">· {durStr}</span>}
                                <span className="ml-2 text-amber-400/30">{entry.callType}</span>
                              </p>
                            </div>
                            <button
                              onClick={() => { setTargetPeerId(entry.peerId); startCall(entry.callType); }}
                              className="p-2 rounded-lg flex-shrink-0 transition-colors"
                              style={{ border: '1px solid rgba(212,175,55,0.15)', color: 'rgba(212,175,55,0.5)' }}
                              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = '#d4af37'; }}
                              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = 'rgba(212,175,55,0.5)'; }}
                              title={`Call back (${entry.callType})`}
                            >
                              {entry.callType === 'video' ? <Video className="w-4 h-4" /> : <Phone className="w-4 h-4" />}
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full gap-6 text-amber-300/40">
                    <Phone className="w-12 h-12" />
                    <div className="text-center">
                      <p className="font-semibold text-amber-200/50">No active call</p>
                      <p className="text-sm mt-1">Select a contact in Messages and tap the call button.</p>
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {/* ── Meetings Tab ── */}
            {tab === 'meetings' && (
              <motion.div
                key="meetings"
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                className="flex flex-col h-full p-6 gap-4"
                style={{ minHeight: 0 }}
              >
                {meeting ? (
                  <div className="flex flex-col h-full gap-4">
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
                          <div className="aspect-video rounded-xl overflow-hidden bg-slate-800 flex items-center justify-center relative border border-amber-400/20">
                            <div
                              className="w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold"
                              style={{ background: 'rgba(212,175,55,0.3)', color: '#d4af37' }}
                            >
                              {displayName.slice(0, 2).toUpperCase()}
                            </div>
                            <span className="absolute bottom-2 left-2 text-xs text-amber-300 px-2 py-0.5 rounded">You</span>
                          </div>
                        </div>
                      )}
                    </div>

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
                  <div className="flex flex-col items-center justify-center h-full gap-6 text-amber-300/30">
                    <Users className="w-14 h-14 text-amber-400/15" />
                    <div className="text-center">
                      <p className="font-semibold text-amber-200/40 mb-1">No active meeting</p>
                      <p className="text-sm">Use the left panel to create or join a room</p>
                    </div>
                    <div className="grid grid-cols-2 gap-2 max-w-xs w-full">
                      {[
                        { icon: <Lock className="w-3.5 h-3.5" />, text: 'E2E Encrypted' },
                        { icon: <Shield className="w-3.5 h-3.5" />, text: 'Dilithium5 keys' },
                        { icon: <Monitor className="w-3.5 h-3.5" />, text: 'Screen sharing' },
                        { icon: <Users className="w-3.5 h-3.5" />, text: 'Up to 49 peers' },
                      ].map(({ icon, text }) => (
                        <div
                          key={text}
                          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs text-amber-300/40"
                          style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(212,175,55,0.08)' }}
                        >
                          <span className="text-amber-400/30">{icon}</span>
                          {text}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {/* ── Groups Tab ── */}
            {tab === 'groups' && (
              <motion.div
                key="groups"
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                className="h-full"
                style={{ minHeight: 0 }}
              >
                <GroupsTab
                  walletAddress={walletAddress}
                  getAuthHeader={async () => {
                    const session = walletSession.getSession();
                    if (!session) return null;
                    return generateAuthHeader(session.privateKey, session.address, '/api/v1/groups');
                  }}
                  apiBaseUrl={getConnectionInfo().apiBaseUrl}
                />
              </motion.div>
            )}

          </AnimatePresence>
        </div>

      </div>

      {/* ── Address Book Drawer ── */}
      <AnimatePresence>
        {showAddressBook && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex"
            style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
            onClick={() => setShowAddressBook(false)}
          >
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 28, stiffness: 260 }}
              className="ml-auto w-full max-w-md h-full overflow-y-auto"
              style={{
                background: 'linear-gradient(135deg, rgba(14,10,40,0.99) 0%, rgba(30,18,60,0.99) 100%)',
                borderLeft: '1.5px solid rgba(212,175,55,0.2)',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: '1px solid rgba(212,175,55,0.12)' }}>
                <div className="flex items-center gap-2">
                  <Users className="w-4 h-4 text-amber-400" />
                  <span className="text-sm font-bold text-amber-100">Address Book</span>
                </div>
                <button
                  onClick={() => setShowAddressBook(false)}
                  className="p-1.5 rounded-lg hover:bg-white/5 transition-colors"
                >
                  <X className="w-4 h-4 text-amber-400/60" />
                </button>
              </div>
              <AddressBook
                onSelectAddress={(addr) => {
                  setTargetPeerId(addr);
                  setShowAddressBook(false);
                  setTab('messages');
                }}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
