import { useState, useEffect, useCallback, useRef } from 'react';

// Module-level cache so emails survive tab switches and show instantly on return.
const _emailCache: Map<string, { emails: any[]; ts: number }> = new Map();
const EMAIL_CACHE_TTL_MS = 60_000;

// Derive a colored avatar from a sender email/address string
function senderAvatar(sender: string): { initials: string; bg: string; glow: string } {
  const name = sender.split('@')[0].replace(/[._\-+]/g, ' ');
  const words = name.split(' ').filter(Boolean);
  const initials = words.length >= 2
    ? (words[0][0] + words[words.length - 1][0]).toUpperCase()
    : name.slice(0, 2).toUpperCase() || '??';
  let hash = 0;
  for (let i = 0; i < sender.length; i++) hash = sender.charCodeAt(i) + ((hash << 5) - hash);
  const palette = [
    { bg: 'linear-gradient(135deg,#FFD700,#FF9800)', glow: 'rgba(255,215,0,0.35)' },
    { bg: 'linear-gradient(135deg,#00E5FF,#0288D1)', glow: 'rgba(0,229,255,0.35)' },
    { bg: 'linear-gradient(135deg,#7C4DFF,#9C27B0)', glow: 'rgba(124,77,255,0.35)' },
    { bg: 'linear-gradient(135deg,#00E676,#00897B)', glow: 'rgba(0,230,118,0.35)' },
    { bg: 'linear-gradient(135deg,#FF6B35,#E91E63)', glow: 'rgba(255,107,53,0.35)' },
    { bg: 'linear-gradient(135deg,#26C6DA,#00ACC1)', glow: 'rgba(38,198,218,0.35)' },
    { bg: 'linear-gradient(135deg,#FF4081,#C2185B)', glow: 'rgba(255,64,129,0.35)' },
  ];
  const { bg, glow } = palette[Math.abs(hash) % palette.length];
  return { initials, bg, glow };
}

// Group emails into time buckets for display
function groupEmailsByTime(emails: any[]): { label: string; emails: any[] }[] {
  const now = Date.now();
  const todayStart = new Date(); todayStart.setHours(0, 0, 0, 0);
  const yesterdayStart = new Date(todayStart); yesterdayStart.setDate(yesterdayStart.getDate() - 1);
  const weekStart = new Date(todayStart); weekStart.setDate(weekStart.getDate() - 7);
  const toMs = (ts: number) => ts > 1e10 ? ts : ts * 1000; // handle s or ms
  return [
    { label: 'Today',      emails: emails.filter(e => toMs(e.timestamp) >= todayStart.getTime()) },
    { label: 'Yesterday',  emails: emails.filter(e => toMs(e.timestamp) >= yesterdayStart.getTime() && toMs(e.timestamp) < todayStart.getTime()) },
    { label: 'This Week',  emails: emails.filter(e => toMs(e.timestamp) >= weekStart.getTime() && toMs(e.timestamp) < yesterdayStart.getTime()) },
    { label: 'Older',      emails: emails.filter(e => toMs(e.timestamp) < weekStart.getTime()) },
  ].filter(g => g.emails.length > 0);
}
import { motion, AnimatePresence } from 'framer-motion';
import {
  Mail, Send, Inbox, Archive, Trash2, Search, Plus, ArrowLeft,
  Paperclip, Star, MoreHorizontal, RefreshCw, ChevronDown,
  Eye, Clock, Check, X, Coins, AlertCircle, User, Reply, Forward,
  Sparkles, Bot, Wand2, ChevronUp, Copy, Loader2, Zap, Settings,
  Shield
} from 'lucide-react';
import { qnkAPI } from '../services/api';

// ============================================================================
// Injected Keyframe Animations — Gold/Cyan/Navy palette
// ============================================================================

const STYLE_ID = 'email-screen-goldcyan-styles';

function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    @keyframes emailPulseGlow {
      0%, 100% { box-shadow: 0 0 8px rgba(255, 215, 0, 0.15), 0 0 20px rgba(0, 229, 255, 0.08); }
      50% { box-shadow: 0 0 18px rgba(255, 215, 0, 0.28), 0 0 40px rgba(0, 229, 255, 0.14); }
    }
    @keyframes emailUnreadPulse {
      0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 5px rgba(0, 229, 255, 0.7); }
      50% { transform: scale(1.35); opacity: 0.65; box-shadow: 0 0 12px rgba(0, 229, 255, 0.9); }
    }
    @keyframes emailGoldPulse {
      0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 5px rgba(255, 215, 0, 0.6); }
      50% { transform: scale(1.35); opacity: 0.65; box-shadow: 0 0 12px rgba(255, 215, 0, 0.85); }
    }
    @keyframes emailGradientShift {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
    @keyframes emailFadeSlideIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes emailSendBurst {
      0% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.12); opacity: 0.85; }
      100% { transform: scale(1); opacity: 1; }
    }
    @keyframes emailConicSpin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    @keyframes emailTypingDots {
      0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
      40% { transform: scale(1); opacity: 1; }
    }
    @keyframes emailShimmer {
      from { transform: translateX(-100%) skewX(-15deg); }
      to { transform: translateX(300%) skewX(-15deg); }
    }
    @keyframes emailFloatParticle {
      0% { transform: translateY(0) translateX(0); opacity: 0; }
      20% { opacity: 1; }
      100% { transform: translateY(-60px) translateX(30px); opacity: 0; }
    }
    @keyframes emailPulseRing {
      0%, 100% { transform: scale(0.95); opacity: 0.5; }
      50% { transform: scale(1.08); opacity: 0.25; }
    }
    .email-item-hover:hover {
      background: linear-gradient(135deg, rgba(255, 215, 0, 0.06), rgba(0, 229, 255, 0.04)) !important;
      border-left-color: rgba(255, 215, 0, 0.55) !important;
    }
    .email-neon-text {
      background: linear-gradient(135deg, #FFD700, #FF6B35, #00E5FF);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .email-scrollbar::-webkit-scrollbar {
      width: 4px;
    }
    .email-scrollbar::-webkit-scrollbar-track {
      background: transparent;
    }
    .email-scrollbar::-webkit-scrollbar-thumb {
      background: rgba(0, 229, 255, 0.2);
      border-radius: 2px;
    }
    .email-scrollbar::-webkit-scrollbar-thumb:hover {
      background: rgba(0, 229, 255, 0.4);
    }
    .email-ai-typing-dot {
      animation: emailTypingDots 1.4s infinite ease-in-out both;
    }
    .email-ai-typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .email-ai-typing-dot:nth-child(2) { animation-delay: -0.16s; }
    .email-ai-typing-dot:nth-child(3) { animation-delay: 0; }
    .email-shimmer-bar {
      position: absolute;
      top: 0;
      left: 0;
      width: 50%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,215,0,0.06), transparent);
      animation: emailShimmer 3.5s ease-in-out infinite;
    }
  `;
  document.head.appendChild(style);
}

// ============================================================================
// Types
// ============================================================================

interface EmailMessage {
  id: string;
  from_wallet: number[];
  from_email?: string;
  to_wallet?: number[];
  to_email?: string;
  subject: string;
  body: string;
  body_html?: string;
  encrypted: boolean;
  timestamp: number;
  read: boolean;
  folder: string;
  thread_id?: string;
  in_reply_to?: string;
  crypto_transfer?: {
    token_type: string;
    amount: number;
    tx_hash: number[];
    confirmed: boolean;
  };
  delivery_method: string;
}

interface SendEmailRequest {
  to: string;
  subject: string;
  body: string;
  body_html?: string;
  crypto_amount?: string;
  crypto_token?: string;
  reply_to?: string;
}

type Folder = 'inbox' | 'sent' | 'drafts' | 'trash' | 'quillon-bank';

// ============================================================================
// EmailScreen Component
// ============================================================================

export default function EmailScreen() {
  useEffect(() => { injectStyles(); }, []);

  // State
  const [activeFolder, setActiveFolder] = useState<Folder>('inbox');
  const [emails, setEmails] = useState<EmailMessage[]>([]);
  const [selectedEmail, setSelectedEmail] = useState<EmailMessage | null>(null);
  const [composing, setComposing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [unreadCount, setUnreadCount] = useState(0);

  // Compose state
  const [composeTo, setComposeTo] = useState('');
  const [composeSubject, setComposeSubject] = useState('');
  const [composeBody, setComposeBody] = useState('');
  const [cryptoEnabled, setCryptoEnabled] = useState(false);
  const [cryptoAmount, setCryptoAmount] = useState('');
  const [cryptoToken, setCryptoToken] = useState('QUG');
  const [sending, setSending] = useState(false);
  const [replyTo, setReplyTo] = useState<string | undefined>();
  const [sendSuccess, setSendSuccess] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);

  // Settings modal state
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [emailAlias, setEmailAlias] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [emailSignature, setEmailSignature] = useState('');
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsLoaded, setSettingsLoaded] = useState(false);

  const composeRef = useRef<HTMLTextAreaElement>(null);

  // ============================================================================
  // Data Fetching
  // ============================================================================

  const fetchEmails = useCallback(async (opts?: { forceRefresh?: boolean }) => {
    const cacheKey = searchQuery.trim() ? `search:${searchQuery}` : activeFolder;
    const cached = _emailCache.get(cacheKey);
    const now = Date.now();
    const isFresh = cached && (now - cached.ts) < EMAIL_CACHE_TTL_MS;

    // Show cached data immediately — no spinner if we have something to show
    if (cached && !opts?.forceRefresh) {
      setEmails(cached.emails);
      if (activeFolder === 'inbox') {
        const localUnread = cached.emails.filter((e: any) => !e.read).length;
        setUnreadCount(localUnread);
        window.dispatchEvent(new CustomEvent('email-unread-count', { detail: { count: localUnread } }));
      }
      if (isFresh) return; // cache still fresh — skip the network call entirely
    }

    // Only show spinner when there's nothing to display yet
    if (!cached) setLoading(true);

    try {
      let response;
      if (searchQuery.trim()) {
        response = await qnkAPI.searchEmails(searchQuery);
      } else if (activeFolder === 'inbox') {
        response = await qnkAPI.getEmailInbox();
      } else if (activeFolder === 'sent') {
        response = await qnkAPI.getSentEmails();
      } else {
        response = await qnkAPI.getEmailFolder(activeFolder);
      }
      if (response?.data) {
        _emailCache.set(cacheKey, { emails: response.data, ts: Date.now() });
        setEmails(response.data);
        if (activeFolder === 'inbox') {
          const localUnread = response.data.filter((e: any) => !e.read).length;
          setUnreadCount(localUnread);
          window.dispatchEvent(new CustomEvent('email-unread-count', { detail: { count: localUnread } }));
        }
      }
    } catch (e) {
      console.error('Failed to fetch emails:', e);
    } finally {
      setLoading(false);
    }
  }, [activeFolder, searchQuery]);

  // Kept for external callers (SSE events, send confirmation) but skipped when
  // the inbox fetch already computed the count locally above.
  const fetchUnreadCount = useCallback(async () => {
    // If inbox is cached and fresh, compute locally — skip the extra round-trip
    const cached = _emailCache.get('inbox');
    if (cached && (Date.now() - cached.ts) < EMAIL_CACHE_TTL_MS) {
      const localUnread = cached.emails.filter((e: any) => !e.read).length;
      setUnreadCount(localUnread);
      window.dispatchEvent(new CustomEvent('email-unread-count', { detail: { count: localUnread } }));
      return;
    }
    try {
      const response = await qnkAPI.getEmailUnreadCount();
      if (response?.data?.count !== undefined) {
        setUnreadCount(response.data.count);
        window.dispatchEvent(new CustomEvent('email-unread-count', { detail: { count: response.data.count } }));
      }
    } catch (e) {
      // Silently fail
    }
  }, []);

  useEffect(() => {
    fetchEmails();
    fetchUnreadCount();
  }, [fetchEmails, fetchUnreadCount]);

  // Load email settings + send welcome email on first visit.
  // Settings load runs independently — only triggers a re-fetch if the welcome
  // email was just sent (new account), which invalidates the inbox cache.
  useEffect(() => {
    if (settingsLoaded) return;
    const loadSettings = async () => {
      try {
        const res = await qnkAPI.getEmailSettings();
        if (res?.data) {
          setEmailAlias(res.data.alias || '');
          setDisplayName(res.data.display_name || '');
          setEmailSignature(res.data.signature || '');
          if (!res.data.welcome_sent) {
            await qnkAPI.sendWelcomeEmail();
            // Invalidate cache so the welcome email appears
            _emailCache.delete('inbox');
            fetchEmails({ forceRefresh: true });
          }
        }
        setSettingsLoaded(true);
      } catch (e) {
        console.error('Failed to load email settings:', e);
        setSettingsLoaded(true);
      }
    };
    loadSettings();
  }, [settingsLoaded, fetchEmails]);

  // Save settings handler
  const handleSaveSettings = async () => {
    setSettingsSaving(true);
    try {
      const res = await qnkAPI.updateEmailSettings({
        alias: emailAlias || undefined,
        display_name: displayName || undefined,
        signature: emailSignature || undefined,
      });
      if (res?.data) {
        setEmailAlias(res.data.alias || '');
        setDisplayName(res.data.display_name || '');
        setEmailSignature(res.data.signature || '');
      }
      setSettingsOpen(false);
    } catch (e) {
      console.error('Failed to save settings:', e);
    } finally {
      setSettingsSaving(false);
    }
  };

  // SSE listeners — invalidate cache so new emails appear immediately
  useEffect(() => {
    const handleEmailReceived = () => {
      _emailCache.delete('inbox');
      _emailCache.delete('sent');
      fetchEmails({ forceRefresh: true });
      fetchUnreadCount();
    };
    window.addEventListener('email-received', handleEmailReceived);
    window.addEventListener('email-sent', handleEmailReceived);
    return () => {
      window.removeEventListener('email-received', handleEmailReceived);
      window.removeEventListener('email-sent', handleEmailReceived);
    };
  }, [fetchEmails, fetchUnreadCount]);

  // ============================================================================
  // Actions
  // ============================================================================

  const handleSend = async () => {
    if (!composeTo.trim() || !composeSubject.trim()) return;
    setSending(true);
    setSendError(null);
    try {
      const request: SendEmailRequest = {
        to: composeTo.trim(),
        subject: composeSubject.trim(),
        body: composeBody,
      };
      if (cryptoEnabled && cryptoAmount && parseFloat(cryptoAmount) > 0) {
        request.crypto_amount = cryptoAmount;
        request.crypto_token = cryptoToken;
      }
      if (replyTo) {
        request.reply_to = replyTo;
      }
      const response = await qnkAPI.sendEmail(request);
      if (response?.error) {
        setSendError(response.error);
        return;
      }
      if (response?.data?.email_id) {
        setSendSuccess(true);
        // Invalidate both inbox and sent cache so fresh data loads after send
        _emailCache.delete('inbox');
        _emailCache.delete('sent');
        setTimeout(() => {
          setSendSuccess(false);
          setComposing(false);
          setComposeTo('');
          setComposeSubject('');
          setComposeBody('');
          setCryptoEnabled(false);
          setCryptoAmount('');
          setReplyTo(undefined);
          fetchEmails({ forceRefresh: true });
        }, 1200);
      } else {
        setSendError('Failed to send — no email ID returned');
      }
    } catch (e: any) {
      console.error('Failed to send email:', e);
      setSendError(e?.message || 'Network error — check your connection');
    } finally {
      setSending(false);
    }
  };

  const handleMarkRead = async (emailId: string) => {
    try {
      // v8.5.5: Optimistically update local state + badge FIRST, then persist to server.
      // This ensures the badge clears immediately without waiting for API round-trip.
      setEmails(prev => {
        const updated = prev.map(e =>
          e.id === emailId ? { ...e, read: true } : e
        );
        // Compute new unread count from local state (most reliable source of truth)
        const localUnread = updated.filter(e => !e.read && e.folder === 'inbox').length;
        setUnreadCount(localUnread);
        window.dispatchEvent(new CustomEvent('email-unread-count', { detail: { count: localUnread } }));
        return updated;
      });
      // Persist to server (fire-and-forget — badge already updated)
      await qnkAPI.markEmailRead(emailId);
    } catch (e) {
      console.error('Failed to mark read:', e);
    }
  };

  const handleDelete = async (emailId: string) => {
    try {
      await qnkAPI.deleteEmail(emailId);
      setEmails(prev => prev.filter(e => e.id !== emailId));
      if (selectedEmail?.id === emailId) setSelectedEmail(null);
    } catch (e) {
      console.error('Failed to delete:', e);
    }
  };

  const handleReply = (email: EmailMessage) => {
    setComposing(true);
    setComposeTo(email.from_email || walletToAddress(email.from_wallet));
    setComposeSubject(email.subject.startsWith('Re: ') ? email.subject : `Re: ${email.subject}`);
    setComposeBody(`\n\n--- Original Message ---\n${email.body}`);
    setReplyTo(email.id);
    setTimeout(() => composeRef.current?.focus(), 100);
  };

  const openEmail = async (email: EmailMessage) => {
    setSelectedEmail(email);
    if (!email.read) {
      handleMarkRead(email.id);
    }
  };

  // ============================================================================
  // Helpers
  // ============================================================================

  const walletToAddress = (wallet: number[]) => {
    if (!wallet || wallet.length === 0) return 'Unknown';
    const hex = wallet.map(b => b.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}...${hex.slice(-8)}`;
  };

  const formatTime = (ts: number) => {
    const date = new Date(ts * 1000);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return date.toLocaleDateString(undefined, { weekday: 'short' });
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  const formatCryptoAmount = (amount: number, token: string) => {
    const display = amount / 1e24;
    if (display >= 1000) return `${(display / 1000).toFixed(1)}K ${token}`;
    if (display >= 1) return `${display.toFixed(2)} ${token}`;
    return `${display.toFixed(6)} ${token}`;
  };

  // ============================================================================
  // Render
  // ============================================================================

  const folders = [
    { id: 'inbox' as Folder, label: 'Inbox', icon: Inbox, count: unreadCount },
    { id: 'quillon-bank' as Folder, label: 'Quillon Bank', icon: Shield, count: 0 },
    { id: 'sent' as Folder, label: 'Sent', icon: Send, count: 0 },
    { id: 'drafts' as Folder, label: 'Drafts', icon: Archive, count: 0 },
    { id: 'trash' as Folder, label: 'Trash', icon: Trash2, count: 0 },
  ];

  return (
    <div
      className="h-[calc(100vh-120px)] flex rounded-2xl overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, rgba(10,14,26,0.98), rgba(14,20,35,0.98), rgba(10,14,26,0.98))',
        border: '1px solid rgba(34, 211, 238, 0.1)',
        boxShadow: '0 0 60px rgba(255,215,0,0.04), 0 0 120px rgba(0,229,255,0.03), inset 0 1px 0 rgba(255,255,255,0.03)',
        animation: 'emailPulseGlow 7s ease-in-out infinite',
      }}
    >
      {/* ================================================================== */}
      {/* Left Panel: Folders                                                 */}
      {/* ================================================================== */}
      <div
        className="w-56 flex-shrink-0 flex flex-col"
        style={{
          background: 'rgba(10, 14, 26, 0.95)',
          backdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(34, 211, 238, 0.08)',
        }}
      >
        {/* Compose Button */}
        <div className="p-4">
          <motion.button
            whileHover={{ scale: 1.02, boxShadow: '0 6px 28px rgba(255,215,0,0.35)' }}
            whileTap={{ scale: 0.97 }}
            onClick={() => {
              setComposing(true);
              setSelectedEmail(null);
              setComposeTo('');
              setComposeSubject('');
              setComposeBody('');
              setCryptoEnabled(false);
              setReplyTo(undefined);
            }}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-semibold text-sm transition-all"
            style={{
              background: 'linear-gradient(135deg, #FFD700, #FF6B35)',
              color: '#0a0e1a',
              boxShadow: '0 4px 18px rgba(255,215,0,0.25), inset 0 1px 0 rgba(255,255,255,0.2)',
              fontWeight: 700,
            }}
          >
            <Plus className="w-4 h-4" />
            Compose
          </motion.button>
        </div>

        {/* Folder List */}
        <nav className="flex-1 px-2">
          {folders.map((folder) => {
            const Icon = folder.icon;
            const isActive = activeFolder === folder.id && !composing;
            return (
              <motion.button
                key={folder.id}
                whileHover={{ x: 2 }}
                onClick={() => {
                  setActiveFolder(folder.id);
                  setSelectedEmail(null);
                  setComposing(false);
                }}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 text-sm transition-all relative overflow-hidden"
                style={{
                  background: isActive
                    ? 'linear-gradient(135deg, rgba(255,215,0,0.08), rgba(0,229,255,0.05))'
                    : 'transparent',
                  color: isActive ? '#00E5FF' : 'rgba(156, 163, 175, 0.7)',
                  borderLeft: isActive ? '2px solid #FFD700' : '2px solid transparent',
                }}
              >
                {isActive && (
                  <div
                    className="absolute inset-0 opacity-100"
                    style={{
                      background: 'linear-gradient(90deg, rgba(255,215,0,0.06), transparent)',
                    }}
                  />
                )}
                <Icon
                  className="w-4 h-4 relative z-10 flex-shrink-0"
                  style={{ color: isActive ? '#FFD700' : 'rgba(156,163,175,0.5)' }}
                />
                <span className="flex-1 text-left relative z-10">{folder.label}</span>
                {folder.count > 0 && (
                  <span
                    className="px-2 py-0.5 rounded-full text-xs font-bold relative z-10"
                    style={{
                      background: 'rgba(0,229,255,0.12)',
                      color: '#00E5FF',
                      border: '1px solid rgba(0,229,255,0.25)',
                      animation: 'emailUnreadPulse 2.2s ease-in-out infinite',
                    }}
                  >
                    {folder.count}
                  </span>
                )}
              </motion.button>
            );
          })}
        </nav>

        {/* Bottom: Settings + E2E indicator */}
        <div className="p-3" style={{ borderTop: '1px solid rgba(34,211,238,0.06)' }}>
          <motion.button
            onClick={() => setSettingsOpen(true)}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-all"
            style={{ color: 'rgba(0,229,255,0.6)', background: 'transparent' }}
            whileHover={{ scale: 1.02, backgroundColor: 'rgba(0,229,255,0.06)' }}
          >
            <Settings className="w-3.5 h-3.5" />
            <span>Email Settings</span>
          </motion.button>
          {emailAlias && (
            <div className="px-3 mt-1.5 text-xs truncate" style={{ color: 'rgba(255,215,0,0.4)' }}>
              {emailAlias}@quillon.xyz
            </div>
          )}
          <div className="flex items-center gap-2 text-xs mt-2 px-3" style={{ color: 'rgba(0,230,118,0.7)' }}>
            <div
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{
                background: '#00E676',
                boxShadow: '0 0 6px rgba(0,230,118,0.7)',
              }}
            />
            E2E Encrypted
          </div>
        </div>
      </div>

      {/* ================================================================== */}
      {/* Center Panel: Email List                                            */}
      {/* ================================================================== */}
      <div
        className="w-80 flex-shrink-0 flex flex-col"
        style={{
          background: 'rgba(10, 14, 26, 0.7)',
          backdropFilter: 'blur(16px)',
          borderRight: '1px solid rgba(34,211,238,0.06)',
        }}
      >
        {/* Search Bar */}
        <div className="p-3" style={{ borderBottom: '1px solid rgba(34,211,238,0.06)' }}>
          <div className="relative">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4"
              style={{ color: 'rgba(0,229,255,0.35)' }}
            />
            <input
              type="text"
              placeholder="Search emails..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && fetchEmails({ forceRefresh: true })}
              className="w-full pl-10 pr-4 py-3 rounded-xl text-sm text-white placeholder-gray-600 focus:outline-none transition-all"
              style={{
                background: 'rgba(0,229,255,0.04)',
                border: '1px solid rgba(34,211,238,0.12)',
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = 'rgba(0,229,255,0.4)';
                e.currentTarget.style.boxShadow = '0 0 16px rgba(0,229,255,0.08)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = 'rgba(34,211,238,0.12)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            />
          </div>
        </div>

        {/* Toolbar */}
        <div
          className="px-3 py-2 flex items-center justify-between"
          style={{ borderBottom: '1px solid rgba(34,211,238,0.05)' }}
        >
          <span className="text-xs uppercase tracking-widest" style={{ color: 'rgba(107,114,128,0.8)', fontSize: '10px' }}>
            {emails.length} message{emails.length !== 1 ? 's' : ''}
          </span>
          <div className="flex items-center gap-1">
            {unreadCount > 0 && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={async () => {
                  // v8.5.5: Optimistically clear badge FIRST, then persist
                  setUnreadCount(0);
                  setEmails(prev => prev.map(e => ({ ...e, read: true })));
                  window.dispatchEvent(new CustomEvent('email-unread-count', { detail: { count: 0 } }));
                  try {
                    await qnkAPI.markAllEmailsRead();
                  } catch {}
                }}
                className="px-2 py-1 rounded-lg text-[10px] transition-colors"
                style={{ color: 'rgba(0,229,255,0.5)', background: 'rgba(0,229,255,0.06)', border: '1px solid rgba(0,229,255,0.1)' }}
                title="Mark all as read"
              >
                <Check className="w-3 h-3 inline mr-0.5" />Read all
              </motion.button>
            )}
            <motion.button
              whileHover={{ rotate: 180 }}
              transition={{ duration: 0.4 }}
              onClick={() => { _emailCache.delete(activeFolder); fetchEmails({ forceRefresh: true }); }}
              className="p-1.5 rounded-lg transition-colors"
              style={{ color: 'rgba(0,229,255,0.4)' }}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            </motion.button>
          </div>
        </div>

        {/* Email List */}
        <div className="flex-1 overflow-y-auto email-scrollbar">
          {loading && emails.length === 0 ? (
            <div className="flex items-center justify-center h-40">
              <div className="flex items-center gap-2" style={{ color: 'rgba(0,229,255,0.4)' }}>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm">Loading...</span>
              </div>
            </div>
          ) : emails.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-40">
              <Mail className="w-8 h-8 mb-2" style={{ color: 'rgba(255,215,0,0.15)' }} />
              <span className="text-sm" style={{ color: 'rgba(107,114,128,0.6)' }}>No emails yet</span>
            </div>
          ) : (
            groupEmailsByTime(emails).map(({ label, emails: group }) => (
              <div key={label}>
                {/* Time group header */}
                <div
                  className="px-4 py-1.5 flex items-center gap-2"
                  style={{ position: 'sticky', top: 0, zIndex: 1, background: 'rgba(10,14,26,0.92)', backdropFilter: 'blur(8px)' }}
                >
                  <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'rgba(0,229,255,0.35)' }}>
                    {label}
                  </span>
                  <div className="flex-1 h-px" style={{ background: 'rgba(34,211,238,0.06)' }} />
                </div>

                {group.map((email, idx) => {
                  const sender = email.from_email || walletToAddress(email.from_wallet);
                  const av = senderAvatar(sender);
                  const isSelected = selectedEmail?.id === email.id;
                  return (
                    <motion.button
                      key={email.id}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.025, duration: 0.18 }}
                      whileHover={{ y: -1, boxShadow: '0 4px 20px rgba(0,0,0,0.25)' }}
                      onClick={() => openEmail(email)}
                      className="w-full text-left px-3 py-3 mx-1 transition-all relative group"
                      style={{
                        width: 'calc(100% - 8px)',
                        borderRadius: 12,
                        marginBottom: 2,
                        borderLeft: isSelected ? '2px solid #FFD700' : '2px solid transparent',
                        background: isSelected
                          ? 'linear-gradient(135deg, rgba(255,215,0,0.08), rgba(0,229,255,0.04))'
                          : !email.read
                            ? 'rgba(255,215,0,0.025)'
                            : 'transparent',
                      }}
                    >
                      <div className="flex items-start gap-3">
                        {/* Avatar */}
                        <div
                          className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
                          style={{
                            background: av.bg,
                            boxShadow: `0 0 10px ${av.glow}`,
                            color: '#0a0e1a',
                            fontSize: 11,
                            letterSpacing: '0.02em',
                          }}
                        >
                          {av.initials}
                        </div>

                        {/* Content */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5 mb-0.5">
                            {!email.read && (
                              <div
                                className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                                style={{ background: '#00E5FF', boxShadow: '0 0 5px rgba(0,229,255,0.8)' }}
                              />
                            )}
                            <span
                              className="text-xs truncate flex-1"
                              style={{ color: !email.read ? '#E5E7EB' : 'rgba(156,163,175,0.55)', fontWeight: !email.read ? 600 : 400 }}
                            >
                              {sender}
                            </span>
                            <span className="text-[10px] flex-shrink-0" style={{ color: 'rgba(255,215,0,0.35)' }}>
                              {formatTime(email.timestamp)}
                            </span>
                          </div>
                          <div
                            className="text-sm truncate mb-0.5"
                            style={{ color: !email.read ? '#E5E7EB' : 'rgba(156,163,175,0.5)', fontWeight: !email.read ? 500 : 400 }}
                          >
                            {email.subject || '(No Subject)'}
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs truncate flex-1" style={{ color: 'rgba(107,114,128,0.55)' }}>
                              {email.body.slice(0, 70)}
                            </span>
                            {email.crypto_transfer && (
                              <span
                                className="flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] flex-shrink-0 font-semibold"
                                style={{ background: 'rgba(0,230,118,0.1)', color: '#00E676', border: '1px solid rgba(0,230,118,0.2)' }}
                              >
                                <Coins className="w-2.5 h-2.5" />
                                {formatCryptoAmount(email.crypto_transfer.amount, email.crypto_transfer.token_type)}
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Hover actions — slide in from right */}
                        <div
                          className="flex items-center gap-1 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                          style={{ marginTop: 2 }}
                          onClick={e => e.stopPropagation()}
                        >
                          <button
                            onClick={() => { openEmail(email); handleReply(email); }}
                            className="p-1 rounded-lg transition-colors"
                            style={{ color: 'rgba(0,229,255,0.5)' }}
                            title="Reply"
                          >
                            <Reply className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => handleDelete(email.id)}
                            className="p-1 rounded-lg transition-colors"
                            style={{ color: 'rgba(239,68,68,0.4)' }}
                            title="Delete"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </motion.button>
                  );
                })}
              </div>
            ))
          )}
        </div>
      </div>

      {/* ================================================================== */}
      {/* Right Panel: Email Detail or Compose                                */}
      {/* ================================================================== */}
      <div
        className="flex-1 flex flex-col min-w-0"
        style={{
          background: 'rgba(10, 14, 26, 0.5)',
          backdropFilter: 'blur(10px)',
        }}
      >
        <AnimatePresence mode="wait">
          {composing ? (
            <ComposePanel
              key="compose"
              composeTo={composeTo}
              setComposeTo={setComposeTo}
              composeSubject={composeSubject}
              setComposeSubject={setComposeSubject}
              composeBody={composeBody}
              setComposeBody={setComposeBody}
              cryptoEnabled={cryptoEnabled}
              setCryptoEnabled={setCryptoEnabled}
              cryptoAmount={cryptoAmount}
              setCryptoAmount={setCryptoAmount}
              cryptoToken={cryptoToken}
              setCryptoToken={setCryptoToken}
              sending={sending}
              sendSuccess={sendSuccess}
              sendError={sendError}
              onSend={handleSend}
              onClose={() => { setComposing(false); setSendError(null); }}
              composeRef={composeRef}
            />
          ) : selectedEmail ? (
            <EmailDetailPanel
              key={selectedEmail.id}
              email={selectedEmail}
              onReply={handleReply}
              onDelete={handleDelete}
              onBack={() => setSelectedEmail(null)}
              formatTime={formatTime}
              walletToAddress={walletToAddress}
              formatCryptoAmount={formatCryptoAmount}
            />
          ) : (
            /* ── Empty state with pulse rings like WelcomeMainnetModal ── */
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex-1 flex flex-col items-center justify-center"
            >
              {/* Animated pulse rings */}
              <div className="relative flex items-center justify-center mb-8" style={{ width: 160, height: 160 }}>
                {/* Outer rings */}
                <motion.div
                  animate={{ scale: [0.85, 1.15, 0.85], opacity: [0.12, 0.3, 0.12] }}
                  transition={{ duration: 4.5, repeat: Infinity, ease: 'easeInOut' }}
                  style={{
                    position: 'absolute',
                    width: 160, height: 160,
                    borderRadius: '50%',
                    border: '1px solid #FFD700',
                    boxShadow: '0 0 20px rgba(255,215,0,0.1)',
                  }}
                />
                <motion.div
                  animate={{ scale: [0.85, 1.15, 0.85], opacity: [0.15, 0.35, 0.15] }}
                  transition={{ duration: 4.5, delay: 0.75, repeat: Infinity, ease: 'easeInOut' }}
                  style={{
                    position: 'absolute',
                    width: 120, height: 120,
                    borderRadius: '50%',
                    border: '1px solid #00E5FF',
                    boxShadow: '0 0 16px rgba(0,229,255,0.1)',
                  }}
                />
                <motion.div
                  animate={{ scale: [0.85, 1.15, 0.85], opacity: [0.18, 0.4, 0.18] }}
                  transition={{ duration: 4.5, delay: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                  style={{
                    position: 'absolute',
                    width: 80, height: 80,
                    borderRadius: '50%',
                    border: '1px solid rgba(124,77,255,0.6)',
                    boxShadow: '0 0 12px rgba(124,77,255,0.1)',
                  }}
                />
                {/* Center icon */}
                <motion.div
                  animate={{ scale: [1, 1.08, 1] }}
                  transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                  style={{
                    width: 56, height: 56,
                    borderRadius: '50%',
                    background: 'radial-gradient(circle, rgba(255,215,0,0.15) 0%, rgba(0,229,255,0.08) 60%, transparent 100%)',
                    border: '1px solid rgba(255,215,0,0.2)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    boxShadow: '0 0 24px rgba(255,215,0,0.08)',
                  }}
                >
                  <Mail className="w-7 h-7" style={{ color: 'rgba(255,215,0,0.5)' }} />
                </motion.div>
              </div>

              <p className="text-lg font-semibold mb-1.5 email-neon-text">Quillon Mail</p>
              <p className="text-sm mb-6" style={{ color: 'rgba(107,114,128,0.8)' }}>
                Select an email to read or compose a new one
              </p>

              <div className="flex items-center gap-5">
                <div className="flex items-center gap-1.5 text-xs" style={{ color: 'rgba(0,229,255,0.45)' }}>
                  <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#00E5FF', boxShadow: '0 0 4px #00E5FF' }} />
                  P2P Gossipsub
                </div>
                <div className="flex items-center gap-1.5 text-xs" style={{ color: 'rgba(255,215,0,0.4)' }}>
                  <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#FFD700', boxShadow: '0 0 4px #FFD700' }} />
                  SMTP Bridge
                </div>
                <div className="flex items-center gap-1.5 text-xs" style={{ color: 'rgba(0,230,118,0.45)' }}>
                  <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#00E676', boxShadow: '0 0 4px #00E676' }} />
                  Encrypted
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ================================================================== */}
      {/* Email Settings Modal — animated conic gradient border               */}
      {/* ================================================================== */}
      <AnimatePresence>
        {settingsOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center"
            style={{
              background: 'radial-gradient(ellipse at 30% 20%, rgba(124,77,255,0.06) 0%, transparent 50%), rgba(0,0,0,0.8)',
              backdropFilter: 'blur(10px)',
            }}
            onClick={() => setSettingsOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.88, opacity: 0, y: 24 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.88, opacity: 0, y: 24 }}
              transition={{ type: 'spring', damping: 22, stiffness: 350 }}
              className="w-full max-w-lg mx-4 relative"
              style={{ borderRadius: 24 }}
              onClick={(e: React.MouseEvent) => e.stopPropagation()}
            >
              {/* Animated conic gradient border */}
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 10, repeat: Infinity, ease: 'linear' }}
                style={{
                  position: 'absolute',
                  inset: -2,
                  borderRadius: 26,
                  background: 'conic-gradient(from 0deg, #FFD700, #00E5FF, #7C4DFF, #FF6B35, #00E676, #E040FB, #FFD700)',
                  opacity: 0.4,
                  zIndex: 0,
                }}
              />
              {/* Inner dark panel */}
              <div
                style={{
                  position: 'absolute',
                  inset: 2,
                  borderRadius: 22,
                  background: 'linear-gradient(160deg, rgba(10,14,26,0.98) 0%, rgba(17,24,39,0.97) 50%, rgba(10,14,26,0.98) 100%)',
                  zIndex: 1,
                }}
              />
              {/* Content */}
              <div style={{ position: 'relative', zIndex: 2, borderRadius: 22, overflow: 'hidden' }}>
                {/* Header */}
                <div
                  className="px-6 py-5 flex items-center justify-between"
                  style={{ borderBottom: '1px solid rgba(255,215,0,0.08)' }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-10 h-10 rounded-xl flex items-center justify-center"
                      style={{
                        background: 'linear-gradient(135deg, rgba(255,215,0,0.12), rgba(0,229,255,0.08))',
                        border: '1px solid rgba(255,215,0,0.2)',
                      }}
                    >
                      <Settings className="w-5 h-5" style={{ color: '#FFD700' }} />
                    </div>
                    <div>
                      <h3 className="text-white font-semibold text-lg">Email Settings</h3>
                      <p className="text-xs" style={{ color: 'rgba(107,114,128,0.8)' }}>
                        Configure your Quillon Mail identity
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setSettingsOpen(false)}
                    className="p-2 rounded-lg transition-all"
                    style={{ color: 'rgba(156,163,175,0.5)' }}
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                {/* Settings Form */}
                <div className="px-6 py-5 space-y-5">
                  {/* Email Alias */}
                  <div>
                    <label
                      className="block text-sm font-medium mb-2"
                      style={{ color: 'rgba(229,231,235,0.8)' }}
                    >
                      Email Alias
                    </label>
                    <div className="flex items-center gap-0">
                      <input
                        type="text"
                        value={emailAlias}
                        onChange={(e) =>
                          setEmailAlias(e.target.value.toLowerCase().replace(/[^a-z0-9_.]/g, ''))
                        }
                        placeholder="yourname"
                        maxLength={20}
                        className="flex-1 px-4 py-3 rounded-l-xl text-white text-sm placeholder-gray-700 focus:outline-none transition-all"
                        style={{
                          background: 'rgba(255,215,0,0.04)',
                          border: '1px solid rgba(255,215,0,0.12)',
                          borderRight: 'none',
                        }}
                        onFocus={(e) => {
                          e.currentTarget.style.borderColor = 'rgba(255,215,0,0.3)';
                          e.currentTarget.style.boxShadow = '0 0 10px rgba(255,215,0,0.08)';
                        }}
                        onBlur={(e) => {
                          e.currentTarget.style.borderColor = 'rgba(255,215,0,0.12)';
                          e.currentTarget.style.boxShadow = 'none';
                        }}
                      />
                      <div
                        className="px-4 py-3 rounded-r-xl text-sm font-medium"
                        style={{
                          background: 'rgba(255,215,0,0.08)',
                          border: '1px solid rgba(255,215,0,0.12)',
                          color: 'rgba(255,215,0,0.6)',
                        }}
                      >
                        @quillon.xyz
                      </div>
                    </div>
                    <p className="text-xs mt-1.5" style={{ color: 'rgba(107,114,128,0.6)' }}>
                      3-20 characters. Letters, numbers, dots, underscores.
                    </p>
                  </div>

                  {/* Display Name */}
                  <div>
                    <label
                      className="block text-sm font-medium mb-2"
                      style={{ color: 'rgba(229,231,235,0.8)' }}
                    >
                      Display Name
                    </label>
                    <input
                      type="text"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                      placeholder="Your name (shown to recipients)"
                      maxLength={50}
                      className="w-full px-4 py-3 rounded-xl text-white text-sm placeholder-gray-700 focus:outline-none transition-all"
                      style={{
                        background: 'rgba(0,229,255,0.03)',
                        border: '1px solid rgba(34,211,238,0.1)',
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(0,229,255,0.3)';
                        e.currentTarget.style.boxShadow = '0 0 10px rgba(0,229,255,0.07)';
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(34,211,238,0.1)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>

                  {/* Email Signature */}
                  <div>
                    <label
                      className="block text-sm font-medium mb-2"
                      style={{ color: 'rgba(229,231,235,0.8)' }}
                    >
                      Email Signature
                    </label>
                    <textarea
                      value={emailSignature}
                      onChange={(e) => setEmailSignature(e.target.value)}
                      placeholder="Your email signature (appended to all outgoing emails)"
                      rows={3}
                      maxLength={500}
                      className="w-full px-4 py-3 rounded-xl text-white text-sm placeholder-gray-700 focus:outline-none resize-none transition-all"
                      style={{
                        background: 'rgba(0,229,255,0.03)',
                        border: '1px solid rgba(34,211,238,0.1)',
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(0,229,255,0.3)';
                        e.currentTarget.style.boxShadow = '0 0 10px rgba(0,229,255,0.07)';
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(34,211,238,0.1)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>

                  {/* Preview card with cyan border glow */}
                  {emailAlias && (
                    <motion.div
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="p-4 rounded-xl relative overflow-hidden"
                      style={{
                        background: 'linear-gradient(135deg, rgba(0,229,255,0.04), rgba(255,215,0,0.02))',
                        border: '1px solid rgba(0,229,255,0.18)',
                        boxShadow: '0 0 20px rgba(0,229,255,0.06)',
                      }}
                    >
                      <div className="email-shimmer-bar" />
                      <p className="text-xs mb-1" style={{ color: 'rgba(107,114,128,0.8)' }}>
                        Your email address:
                      </p>
                      <p
                        className="text-sm font-semibold"
                        style={{
                          background: 'linear-gradient(135deg, #FFD700, #00E5FF)',
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                        }}
                      >
                        {emailAlias}@quillon.xyz
                      </p>
                      {displayName && (
                        <p className="text-xs mt-1" style={{ color: 'rgba(156,163,175,0.6)' }}>
                          Displayed as: {displayName} &lt;{emailAlias}@quillon.xyz&gt;
                        </p>
                      )}
                    </motion.div>
                  )}
                </div>

                {/* Footer */}
                <div
                  className="px-6 py-4 flex items-center justify-end gap-3"
                  style={{ borderTop: '1px solid rgba(255,215,0,0.06)' }}
                >
                  <button
                    onClick={() => setSettingsOpen(false)}
                    className="px-5 py-2.5 rounded-xl text-sm font-medium transition-all"
                    style={{
                      color: 'rgba(156,163,175,0.7)',
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    Cancel
                  </button>
                  <motion.button
                    onClick={handleSaveSettings}
                    disabled={settingsSaving}
                    className="px-6 py-2.5 rounded-xl text-sm font-bold transition-all disabled:opacity-50"
                    style={{
                      background: 'linear-gradient(135deg, #FFD700, #FF6B35)',
                      color: '#0a0e1a',
                      boxShadow: '0 4px 18px rgba(255,215,0,0.25)',
                    }}
                    whileHover={{ scale: 1.03, boxShadow: '0 6px 26px rgba(255,215,0,0.35)' }}
                    whileTap={{ scale: 0.97 }}
                  >
                    {settingsSaving ? (
                      <span className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin" /> Saving...
                      </span>
                    ) : (
                      <span className="flex items-center gap-2">
                        <Check className="w-4 h-4" /> Save Settings
                      </span>
                    )}
                  </motion.button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ============================================================================
// AI Email Assistant Hook — v7.3.4: Uses BitNet b1.58-2B-4T (OpenAI-compatible)
// ============================================================================

function useAIAssistant() {
  const [aiMessages, setAiMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [aiInput, setAiInput] = useState('');
  const [aiStreaming, setAiStreaming] = useState(false);
  const [aiStreamText, setAiStreamText] = useState('');
  const abortControllerRef = useRef<AbortController | null>(null);

  const streamAI = async (prompt: string, onToken: (text: string) => void): Promise<string> => {
    setAiStreaming(true);
    setAiStreamText('');

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let fullText = '';

    try {
      // Email AI assistant — /api/v1/ai/email-assist proxies to gemma4 on Epsilon Ollama
      const response = await fetch('/api/v1/ai/email-assist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            ...aiMessages.slice(-4),
            { role: 'user', content: prompt },
          ],
        }),
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error('ReadableStream not supported');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const raw = line.substring(5).trim();
          if (!raw) continue;
          try {
            const parsed = JSON.parse(raw);
            // SSE "token" event: { content: "..." }
            if (parsed.content) {
              fullText += parsed.content;
              setAiStreamText(fullText);
              onToken(fullText);
            }
          } catch {
            console.warn('Failed to parse email AI SSE chunk:', raw?.substring(0, 80));
          }
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        console.error('Email AI error:', e);
      }
    } finally {
      setAiStreaming(false);
      abortControllerRef.current = null;
    }

    return fullText;
  };

  const stopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setAiStreaming(false);
  };

  const sendPrompt = async (prompt: string): Promise<string> => {
    setAiMessages(prev => [...prev, { role: 'user', content: prompt }]);
    const result = await streamAI(prompt, () => {});
    setAiMessages(prev => [...prev, { role: 'assistant', content: result }]);
    setAiStreamText('');
    return result;
  };

  return {
    aiMessages,
    aiInput,
    setAiInput,
    aiStreaming,
    aiStreamText,
    sendPrompt,
    streamAI,
    stopStreaming,
  };
}

// ============================================================================
// Compose Panel with AI Assistant
// ============================================================================

function ComposePanel({
  composeTo, setComposeTo,
  composeSubject, setComposeSubject,
  composeBody, setComposeBody,
  cryptoEnabled, setCryptoEnabled,
  cryptoAmount, setCryptoAmount,
  cryptoToken, setCryptoToken,
  sending, sendSuccess, sendError, onSend, onClose, composeRef,
}: {
  composeTo: string; setComposeTo: (v: string) => void;
  composeSubject: string; setComposeSubject: (v: string) => void;
  composeBody: string; setComposeBody: (v: string) => void;
  cryptoEnabled: boolean; setCryptoEnabled: (v: boolean) => void;
  cryptoAmount: string; setCryptoAmount: (v: string) => void;
  cryptoToken: string; setCryptoToken: (v: string) => void;
  sending: boolean; sendSuccess: boolean; sendError: string | null; onSend: () => void; onClose: () => void;
  composeRef: React.RefObject<HTMLTextAreaElement | null>;
}) {
  const [aiPanelOpen, setAiPanelOpen] = useState(false);
  const aiAssistant = useAIAssistant();
  const [aiLocalInput, setAiLocalInput] = useState('');
  const [aiResult, setAiResult] = useState('');
  const aiResultRef = useRef('');
  const aiScrollRef = useRef<HTMLDivElement>(null);

  const handleAIGenerate = async () => {
    if (!aiLocalInput.trim() || aiAssistant.aiStreaming) return;
    const prompt = aiLocalInput.trim();
    setAiLocalInput('');
    setAiResult('');
    aiResultRef.current = '';

    await aiAssistant.streamAI(
      `You are an email writing assistant. ${prompt}. Write only the email body text, no subject line, no greeting signature unless asked.`,
      (text) => {
        aiResultRef.current = text;
        setAiResult(text);
      }
    );
  };

  const handleAIImprove = async () => {
    if (!composeBody.trim() || aiAssistant.aiStreaming) return;
    setAiResult('');
    aiResultRef.current = '';

    await aiAssistant.streamAI(
      `Improve the following email to be more professional, clear, and well-structured. Keep the same meaning and tone. Return only the improved email body:\n\n${composeBody}`,
      (text) => {
        aiResultRef.current = text;
        setAiResult(text);
      }
    );
  };

  const handleInsertAI = () => {
    if (aiResultRef.current) {
      setComposeBody(aiResultRef.current);
      setAiResult('');
      aiResultRef.current = '';
    }
  };

  useEffect(() => {
    if (aiScrollRef.current) {
      aiScrollRef.current.scrollTop = aiScrollRef.current.scrollHeight;
    }
  }, [aiResult, aiAssistant.aiStreamText]);

  // Send particles state
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; color: string }>>([]);

  const triggerSendParticles = () => {
    const colors = ['#FFD700', '#00E5FF', '#FF6B35', '#00E676'];
    const newParticles = Array.from({ length: 8 }, (_, i) => ({
      id: Date.now() + i,
      x: Math.random() * 40 - 20,
      y: Math.random() * -40 - 10,
      color: colors[i % colors.length],
    }));
    setParticles(newParticles);
    setTimeout(() => setParticles([]), 1000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex-1 flex flex-col"
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-6 py-4"
        style={{
          borderBottom: '1px solid rgba(255,215,0,0.08)',
          background: 'linear-gradient(90deg, rgba(255,215,0,0.04), transparent)',
        }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{
              background: 'linear-gradient(135deg, rgba(255,215,0,0.15), rgba(0,229,255,0.1))',
              border: '1px solid rgba(255,215,0,0.2)',
            }}
          >
            <Mail className="w-4 h-4" style={{ color: '#FFD700' }} />
          </div>
          <h3 className="font-semibold email-neon-text text-base">New Email</h3>
        </div>
        <motion.button
          whileHover={{ scale: 1.1, rotate: 90 }}
          whileTap={{ scale: 0.9 }}
          onClick={onClose}
          className="p-1.5 rounded-lg transition-colors"
          style={{ color: 'rgba(156,163,175,0.4)' }}
        >
          <X className="w-5 h-5" />
        </motion.button>
      </div>

      {/* Fields */}
      <div
        className="px-6 pt-5 pb-2 space-y-3"
        style={{ borderBottom: '1px solid rgba(34,211,238,0.07)' }}
      >
        <div className="flex items-center gap-3">
          <label className="text-xs w-16 font-semibold uppercase tracking-widest flex-shrink-0" style={{ color: 'rgba(0,229,255,0.5)' }}>
            To
          </label>
          <input
            type="text"
            placeholder="Wallet address or email"
            value={composeTo}
            onChange={(e) => setComposeTo(e.target.value)}
            className="flex-1 px-4 py-3 rounded-xl text-white text-base placeholder-gray-600 focus:outline-none transition-all"
            style={{
              background: 'rgba(0,229,255,0.04)',
              border: '1px solid rgba(34,211,238,0.12)',
            }}
            onFocus={e => { e.currentTarget.style.borderColor = 'rgba(0,229,255,0.4)'; e.currentTarget.style.boxShadow = '0 0 16px rgba(0,229,255,0.08)'; }}
            onBlur={e => { e.currentTarget.style.borderColor = 'rgba(34,211,238,0.12)'; e.currentTarget.style.boxShadow = 'none'; }}
          />
        </div>
        <div className="flex items-center gap-3 pb-2">
          <label className="text-xs w-16 font-semibold uppercase tracking-widest flex-shrink-0" style={{ color: 'rgba(0,229,255,0.5)' }}>
            Subject
          </label>
          <input
            type="text"
            placeholder="Email subject"
            value={composeSubject}
            onChange={(e) => setComposeSubject(e.target.value)}
            className="flex-1 px-4 py-3 rounded-xl text-white text-base placeholder-gray-600 focus:outline-none transition-all"
            style={{
              background: 'rgba(0,229,255,0.04)',
              border: '1px solid rgba(34,211,238,0.12)',
            }}
            onFocus={e => { e.currentTarget.style.borderColor = 'rgba(0,229,255,0.4)'; e.currentTarget.style.boxShadow = '0 0 16px rgba(0,229,255,0.08)'; }}
            onBlur={e => { e.currentTarget.style.borderColor = 'rgba(34,211,238,0.12)'; e.currentTarget.style.boxShadow = 'none'; }}
          />
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 px-6 pt-5 pb-4 relative" style={{ minHeight: 0 }}>
        <textarea
          ref={composeRef}
          placeholder="Write your message..."
          value={composeBody}
          onChange={(e) => setComposeBody(e.target.value)}
          className="w-full h-full px-4 py-4 rounded-xl text-base placeholder-gray-600 resize-none focus:outline-none email-scrollbar transition-all"
          style={{
            minHeight: '160px',
            color: '#E5E7EB',
            background: 'rgba(0,229,255,0.03)',
            border: '1px solid rgba(34,211,238,0.08)',
            lineHeight: '1.7',
          }}
          onFocus={e => { e.currentTarget.style.borderColor = 'rgba(0,229,255,0.25)'; e.currentTarget.style.background = 'rgba(0,229,255,0.05)'; }}
          onBlur={e => { e.currentTarget.style.borderColor = 'rgba(34,211,238,0.08)'; e.currentTarget.style.background = 'rgba(0,229,255,0.03)'; }}
        />
      </div>

      {/* AI Assistant Panel */}
      <div style={{ borderTop: '1px solid rgba(0,229,255,0.07)' }}>
        {/* AI Toggle */}
        <motion.button
          whileHover={{ backgroundColor: 'rgba(0,229,255,0.04)' }}
          onClick={() => setAiPanelOpen(!aiPanelOpen)}
          className="w-full flex items-center justify-between px-6 py-2.5 transition-all"
          style={{
            background: aiPanelOpen
              ? 'linear-gradient(90deg, rgba(0,229,255,0.05), rgba(255,215,0,0.03))'
              : 'transparent',
          }}
        >
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4" style={{ color: '#00E5FF' }} />
            <span className="text-xs font-semibold" style={{ color: '#00E5FF' }}>AI Assistant</span>
            {aiAssistant.aiStreaming && (
              <div className="flex items-center gap-0.5 ml-1">
                <div className="w-1 h-1 rounded-full email-ai-typing-dot" style={{ background: '#00E5FF' }} />
                <div className="w-1 h-1 rounded-full email-ai-typing-dot" style={{ background: '#00E5FF' }} />
                <div className="w-1 h-1 rounded-full email-ai-typing-dot" style={{ background: '#00E5FF' }} />
              </div>
            )}
          </div>
          <motion.div animate={{ rotate: aiPanelOpen ? 180 : 0 }}>
            <ChevronUp className="w-4 h-4" style={{ color: 'rgba(0,229,255,0.35)' }} />
          </motion.div>
        </motion.button>

        {/* AI Content */}
        <AnimatePresence>
          {aiPanelOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="overflow-hidden"
            >
              <div
                className="px-6 pb-4 space-y-3"
                style={{
                  background: 'linear-gradient(180deg, rgba(0,229,255,0.03), rgba(255,215,0,0.02))',
                }}
              >
                {/* Quick Actions */}
                <div className="flex items-center gap-2 flex-wrap pt-2">
                  {/* Improve — Gold */}
                  <motion.button
                    whileHover={{ scale: 1.04 }}
                    whileTap={{ scale: 0.96 }}
                    onClick={handleAIImprove}
                    disabled={!composeBody.trim() || aiAssistant.aiStreaming}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all disabled:opacity-30"
                    style={{
                      background: 'linear-gradient(135deg, rgba(255,215,0,0.12), rgba(255,107,53,0.08))',
                      color: '#FFD700',
                      border: '1px solid rgba(255,215,0,0.2)',
                    }}
                  >
                    <Wand2 className="w-3 h-3" />
                    Improve
                  </motion.button>
                  {/* Professional — Cyan */}
                  <motion.button
                    whileHover={{ scale: 1.04 }}
                    whileTap={{ scale: 0.96 }}
                    onClick={() => setAiLocalInput('Write a professional and concise email')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
                    style={{
                      background: 'rgba(0,229,255,0.06)',
                      color: 'rgba(0,229,255,0.8)',
                      border: '1px solid rgba(0,229,255,0.15)',
                    }}
                  >
                    <Zap className="w-3 h-3" />
                    Professional
                  </motion.button>
                  {/* Casual — Orange */}
                  <motion.button
                    whileHover={{ scale: 1.04 }}
                    whileTap={{ scale: 0.96 }}
                    onClick={() => setAiLocalInput('Write a friendly casual email')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
                    style={{
                      background: 'rgba(255,107,53,0.06)',
                      color: 'rgba(255,107,53,0.8)',
                      border: '1px solid rgba(255,107,53,0.15)',
                    }}
                  >
                    <Star className="w-3 h-3" />
                    Casual
                  </motion.button>
                </div>

                {/* AI Input */}
                <div className="flex items-center gap-2">
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={aiLocalInput}
                      onChange={(e) => setAiLocalInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleAIGenerate()}
                      placeholder="Ask AI to write or improve your email..."
                      className="w-full px-3 py-2 rounded-lg text-xs text-white placeholder-gray-600 focus:outline-none transition-all"
                      style={{
                        background: 'rgba(0,229,255,0.04)',
                        border: '1px solid rgba(0,229,255,0.12)',
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(0,229,255,0.3)';
                        e.currentTarget.style.boxShadow = '0 0 10px rgba(0,229,255,0.08)';
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = 'rgba(0,229,255,0.12)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    />
                  </div>
                  {aiAssistant.aiStreaming ? (
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={aiAssistant.stopStreaming}
                      className="p-2 rounded-lg"
                      style={{
                        background: 'rgba(239,68,68,0.12)',
                        border: '1px solid rgba(239,68,68,0.2)',
                        color: '#f87171',
                      }}
                    >
                      <X className="w-3.5 h-3.5" />
                    </motion.button>
                  ) : (
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={handleAIGenerate}
                      disabled={!aiLocalInput.trim()}
                      className="p-2 rounded-lg transition-all disabled:opacity-30"
                      style={{
                        background: 'linear-gradient(135deg, rgba(255,215,0,0.15), rgba(0,229,255,0.1))',
                        border: '1px solid rgba(255,215,0,0.2)',
                        color: '#FFD700',
                      }}
                    >
                      <Send className="w-3.5 h-3.5" />
                    </motion.button>
                  )}
                </div>

                {/* AI Output */}
                <AnimatePresence>
                  {(aiResult || aiAssistant.aiStreaming) && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                    >
                      <div
                        ref={aiScrollRef}
                        className="rounded-lg p-3 max-h-40 overflow-y-auto email-scrollbar relative"
                        style={{
                          background: 'rgba(0,229,255,0.03)',
                          border: '1px solid rgba(0,229,255,0.1)',
                        }}
                      >
                        <div className="email-shimmer-bar" />
                        <div className="flex items-center gap-2 mb-2 relative z-10">
                          <Bot className="w-3.5 h-3.5" style={{ color: '#00E5FF' }} />
                          <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'rgba(0,229,255,0.55)' }}>
                            AI Response
                          </span>
                          {aiAssistant.aiStreaming && (
                            <div className="flex items-center gap-0.5 ml-1">
                              <div className="w-1 h-1 rounded-full email-ai-typing-dot" style={{ background: '#00E5FF' }} />
                              <div className="w-1 h-1 rounded-full email-ai-typing-dot" style={{ background: '#00E5FF' }} />
                              <div className="w-1 h-1 rounded-full email-ai-typing-dot" style={{ background: '#00E5FF' }} />
                            </div>
                          )}
                        </div>
                        <div className="text-xs whitespace-pre-wrap leading-relaxed relative z-10" style={{ color: '#E5E7EB' }}>
                          {aiResult || aiAssistant.aiStreamText}
                          {aiAssistant.aiStreaming && (
                            <span
                              className="inline-block w-1.5 h-3.5 ml-0.5 align-text-bottom"
                              style={{
                                background: '#00E5FF',
                                animation: 'emailUnreadPulse 1s ease-in-out infinite',
                              }}
                            />
                          )}
                        </div>
                      </div>
                      {/* Insert / Dismiss */}
                      {aiResult && !aiAssistant.aiStreaming && (
                        <div className="flex items-center gap-2 mt-2">
                          <motion.button
                            whileHover={{ scale: 1.04 }}
                            whileTap={{ scale: 0.96 }}
                            onClick={handleInsertAI}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all"
                            style={{
                              background: 'linear-gradient(135deg, #FFD700, #FF6B35)',
                              color: '#0a0e1a',
                              boxShadow: '0 2px 12px rgba(255,215,0,0.25)',
                            }}
                          >
                            <Copy className="w-3 h-3" />
                            Insert into Email
                          </motion.button>
                          <motion.button
                            whileHover={{ scale: 1.04 }}
                            whileTap={{ scale: 0.96 }}
                            onClick={() => {
                              setAiResult('');
                              aiResultRef.current = '';
                            }}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
                            style={{
                              background: 'rgba(255,255,255,0.04)',
                              color: 'rgba(156,163,175,0.6)',
                              border: '1px solid rgba(255,255,255,0.08)',
                            }}
                          >
                            <X className="w-3 h-3" />
                            Dismiss
                          </motion.button>
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Crypto Attachment */}
      <div className="px-6 py-3" style={{ borderTop: '1px solid rgba(34,211,238,0.05)' }}>
        <div className="flex items-center gap-3 mb-2">
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            onClick={() => setCryptoEnabled(!cryptoEnabled)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all"
            style={{
              background: cryptoEnabled
                ? 'linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,230,118,0.07))'
                : 'rgba(255,215,0,0.04)',
              color: cryptoEnabled ? '#00E676' : 'rgba(255,215,0,0.5)',
              border: `1px solid ${cryptoEnabled ? 'rgba(0,230,118,0.22)' : 'rgba(255,215,0,0.1)'}`,
              boxShadow: cryptoEnabled ? '0 0 8px rgba(0,230,118,0.12)' : 'none',
            }}
          >
            <Coins className="w-3.5 h-3.5" />
            Attach Crypto
          </motion.button>
        </div>

        <AnimatePresence>
          {cryptoEnabled && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="flex items-center gap-3"
            >
              <select
                value={cryptoToken}
                onChange={(e) => setCryptoToken(e.target.value)}
                className="rounded-lg px-3 py-2 text-sm text-white focus:outline-none"
                style={{
                  background: 'rgba(255,215,0,0.06)',
                  border: '1px solid rgba(255,215,0,0.15)',
                }}
              >
                <option value="QUG">QUG</option>
                <option value="QUGUSD">QUGUSD</option>
              </select>
              <input
                type="number"
                placeholder="Amount"
                value={cryptoAmount}
                onChange={(e) => setCryptoAmount(e.target.value)}
                className="flex-1 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none"
                style={{
                  background: 'rgba(0,229,255,0.04)',
                  border: '1px solid rgba(34,211,238,0.1)',
                }}
                min="0"
                step="0.01"
              />
              <span className="text-xs" style={{ color: 'rgba(255,215,0,0.4)' }}>{cryptoToken}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Send Button */}
      <div
        className="px-6 py-4 flex justify-between items-center"
        style={{ borderTop: '1px solid rgba(255,215,0,0.07)' }}
      >
        <div className="text-xs" style={{ color: 'rgba(107,114,128,0.5)' }}>
          {composeBody.length > 0 && `${composeBody.length} characters`}
        </div>
        <div className="relative">
          {/* Send particles */}
          {particles.map(p => (
            <div
              key={p.id}
              className="absolute pointer-events-none"
              style={{
                left: '50%',
                bottom: '50%',
                width: '4px',
                height: '4px',
                borderRadius: '50%',
                background: p.color,
                boxShadow: `0 0 6px ${p.color}`,
                animation: 'emailFloatParticle 0.8s ease-out forwards',
                transform: `translate(${p.x}px, ${p.y}px)`,
              }}
            />
          ))}

          <motion.button
            whileHover={{ scale: 1.04, boxShadow: '0 6px 28px rgba(255,215,0,0.38)' }}
            whileTap={{ scale: 0.96 }}
            onClick={() => {
              triggerSendParticles();
              onSend();
            }}
            disabled={sending || !composeTo.trim() || !composeSubject.trim()}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-bold text-sm transition-all disabled:opacity-40"
            style={{
              background: sendSuccess
                ? 'linear-gradient(135deg, #00E676, #00a854)'
                : 'linear-gradient(135deg, #FFD700, #FF6B35)',
              backgroundSize: '200% 200%',
              animation: sending ? 'emailSendBurst 0.6s ease-in-out infinite' : undefined,
              color: '#0a0e1a',
              boxShadow: sendSuccess
                ? '0 4px 20px rgba(0,230,118,0.3)'
                : '0 4px 18px rgba(255,215,0,0.28), inset 0 1px 0 rgba(255,255,255,0.2)',
            }}
          >
            {sendSuccess ? (
              <>
                <Check className="w-4 h-4" />
                Sent!
              </>
            ) : sending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Sending...
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                Send Email
              </>
            )}
          </motion.button>
        </div>

        {/* Error display */}
        {sendError && (
          <div className="mt-2 px-4 py-2 rounded-lg bg-red-500/15 border border-red-500/30 text-red-400 text-xs flex items-center gap-2">
            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
            <span>{sendError}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ============================================================================
// Email Detail Panel
// ============================================================================

function EmailDetailPanel({
  email, onReply, onDelete, onBack, formatTime, walletToAddress, formatCryptoAmount,
}: {
  email: EmailMessage;
  onReply: (e: EmailMessage) => void;
  onDelete: (id: string) => void;
  onBack: () => void;
  formatTime: (ts: number) => string;
  walletToAddress: (w: number[]) => string;
  formatCryptoAmount: (a: number, t: string) => string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.2 }}
      className="flex-1 flex flex-col"
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-6 py-4 relative overflow-hidden"
        style={{
          borderBottom: '1px solid rgba(255,215,0,0.07)',
          background: 'linear-gradient(90deg, rgba(255,215,0,0.03), transparent)',
        }}
      >
        {/* Shimmer */}
        <div className="email-shimmer-bar" />
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={onBack}
          className="p-1.5 rounded-lg lg:hidden relative z-10"
          style={{ color: 'rgba(0,229,255,0.5)' }}
        >
          <ArrowLeft className="w-5 h-5" />
        </motion.button>
        <h2 className="text-white font-semibold flex-1 truncate relative z-10">
          {email.subject || '(No Subject)'}
        </h2>
        <div className="flex items-center gap-1 relative z-10">
          <motion.button
            whileHover={{ scale: 1.1, backgroundColor: 'rgba(0,229,255,0.08)' }}
            whileTap={{ scale: 0.9 }}
            onClick={() => onReply(email)}
            className="p-2 rounded-lg transition"
            style={{ color: 'rgba(0,229,255,0.5)' }}
            title="Reply"
          >
            <Reply className="w-4 h-4" />
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.1, backgroundColor: 'rgba(239,68,68,0.08)' }}
            whileTap={{ scale: 0.9 }}
            onClick={() => onDelete(email.id)}
            className="p-2 rounded-lg transition"
            style={{ color: 'rgba(239,68,68,0.4)' }}
            title="Delete"
          >
            <Trash2 className="w-4 h-4" />
          </motion.button>
        </div>
      </div>

      {/* Meta */}
      <div className="px-6 py-4" style={{ borderBottom: '1px solid rgba(34,211,238,0.05)' }}>
        <div className="flex items-center gap-3 mb-3">
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
            style={{
              background: 'linear-gradient(135deg, rgba(255,215,0,0.1), rgba(0,229,255,0.07))',
              border: '1px solid rgba(255,215,0,0.18)',
              boxShadow: '0 0 12px rgba(255,215,0,0.07)',
            }}
          >
            <User className="w-5 h-5" style={{ color: '#FFD700' }} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium truncate" style={{ color: '#E5E7EB' }}>
              {email.from_email || walletToAddress(email.from_wallet)}
            </div>
            <div className="text-xs truncate" style={{ color: 'rgba(107,114,128,0.7)' }}>
              To: {email.to_email || (email.to_wallet ? walletToAddress(email.to_wallet) : 'Unknown')}
            </div>
          </div>
          <div className="text-xs flex items-center gap-1 flex-shrink-0" style={{ color: 'rgba(255,215,0,0.4)' }}>
            <Clock className="w-3 h-3" />
            {formatTime(email.timestamp)}
          </div>
        </div>

        {/* Badges */}
        <div className="flex items-center gap-2 flex-wrap">
          {email.encrypted && (
            <span
              className="px-2 py-0.5 rounded text-xs font-semibold"
              style={{
                background: 'rgba(0,229,255,0.07)',
                color: 'rgba(0,229,255,0.75)',
                border: '1px solid rgba(0,229,255,0.15)',
              }}
            >
              E2E Encrypted
            </span>
          )}
          <span
            className="px-2 py-0.5 rounded text-xs font-semibold"
            style={
              email.delivery_method === 'P2PGossipsub'
                ? {
                    background: 'rgba(0,229,255,0.07)',
                    color: 'rgba(0,229,255,0.7)',
                    border: '1px solid rgba(0,229,255,0.15)',
                  }
                : {
                    background: 'rgba(255,215,0,0.07)',
                    color: 'rgba(255,215,0,0.65)',
                    border: '1px solid rgba(255,215,0,0.15)',
                  }
            }
          >
            {email.delivery_method === 'P2PGossipsub' ? 'P2P Gossipsub' : 'SMTP'}
          </span>
        </div>
      </div>

      {/* Crypto Transfer */}
      {email.crypto_transfer && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mx-6 mt-4 p-4 rounded-xl relative overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, rgba(0,230,118,0.06), rgba(0,230,118,0.02))',
            border: '1px solid rgba(0,230,118,0.15)',
            boxShadow: '0 0 20px rgba(0,230,118,0.05)',
          }}
        >
          <div className="email-shimmer-bar" />
          <div className="flex items-center gap-3 relative z-10">
            <div
              className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
              style={{
                background: 'rgba(0,230,118,0.1)',
                border: '1px solid rgba(0,230,118,0.2)',
                boxShadow: '0 0 10px rgba(0,230,118,0.1)',
              }}
            >
              <Coins className="w-5 h-5 text-green-400" />
            </div>
            <div className="flex-1">
              <div className="text-sm font-semibold" style={{ color: '#00E676' }}>Crypto Transfer</div>
              <div className="text-lg font-bold text-white">
                {formatCryptoAmount(email.crypto_transfer.amount, email.crypto_transfer.token_type)}
              </div>
            </div>
            <div className="flex items-center gap-1 text-xs">
              {email.crypto_transfer.confirmed ? (
                <span
                  className="flex items-center gap-1 px-2 py-1 rounded"
                  style={{
                    background: 'rgba(0,230,118,0.1)',
                    color: '#00E676',
                    border: '1px solid rgba(0,230,118,0.2)',
                  }}
                >
                  <Check className="w-3 h-3" /> Confirmed
                </span>
              ) : (
                <span
                  className="flex items-center gap-1 px-2 py-1 rounded"
                  style={{
                    background: 'rgba(234,179,8,0.1)',
                    color: '#fbbf24',
                    border: '1px solid rgba(234,179,8,0.2)',
                  }}
                >
                  <Clock className="w-3 h-3" /> Pending
                </span>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-6 py-5 email-scrollbar">
        <div className="text-sm whitespace-pre-wrap leading-relaxed" style={{ color: '#D1D5DB' }}>
          {email.body}
        </div>
      </div>

      {/* Quick Reply */}
      <div className="px-6 py-3" style={{ borderTop: '1px solid rgba(255,215,0,0.07)' }}>
        <motion.button
          whileHover={{ scale: 1.01, boxShadow: '0 0 16px rgba(0,229,255,0.1)' }}
          whileTap={{ scale: 0.99 }}
          onClick={() => onReply(email)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm transition-all"
          style={{
            background: 'rgba(0,229,255,0.04)',
            color: 'rgba(0,229,255,0.65)',
            border: '1px solid rgba(0,229,255,0.1)',
          }}
        >
          <Reply className="w-4 h-4" />
          Reply to this email
        </motion.button>
      </div>
    </motion.div>
  );
}
