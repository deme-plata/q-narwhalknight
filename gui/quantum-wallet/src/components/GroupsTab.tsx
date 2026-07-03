// GroupsTab.tsx — Discord-style group chat for Q-NarwhalKnight wallet
// Amber/gold palette on dark slate, SSE via fetch ReadableStream

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Hash, Users, Plus, Send, Copy, LogOut, Shield, Crown, X,
  ChevronRight, MessageSquare, Link, UserPlus, Loader2, Search,
  Check, AlertCircle,
} from 'lucide-react';

// ── Types ─────────────────────────────────────────────────────────────────────

interface GroupMeta {
  id: string;
  name: string;
  description: string;
  owner: string;
  created_at: number;
  member_count: number;
  icon: string | null;
}

interface GroupMember {
  wallet: string;
  display_name: string;
  joined_at: number;
  role: 'owner' | 'admin' | 'member';
}

interface GroupMessage {
  id: string;
  group_id: string;
  from: string;
  display_name: string;
  content: string;
  timestamp: number;
}

interface GroupsTabProps {
  walletAddress: string;
  getAuthHeader: () => Promise<string | null>;
  apiBaseUrl: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function shortAddr(addr: string): string {
  if (!addr || addr.length <= 12) return addr;
  return `${addr.slice(0, 6)}…${addr.slice(-4)}`;
}

function relativeTime(ms: number): string {
  const diff = Date.now() - ms;
  const s = Math.floor(diff / 1000);
  if (s < 10) return 'just now';
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d === 1) return 'Yesterday';
  if (d < 7) return `${d}d ago`;
  return new Date(ms).toLocaleDateString();
}

function sameGrouping(a: GroupMessage, b: GroupMessage): boolean {
  return a.from === b.from && b.timestamp - a.timestamp < 5 * 60 * 1000;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function RoleBadge({ role }: { role: GroupMember['role'] }) {
  if (role === 'owner') {
    return (
      <span className="inline-flex items-center gap-0.5 text-amber-400" title="Owner">
        <Crown size={11} />
      </span>
    );
  }
  if (role === 'admin') {
    return (
      <span className="inline-flex items-center gap-0.5 text-sky-400" title="Admin">
        <Shield size={11} />
      </span>
    );
  }
  return null;
}

interface ModalProps {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}

function Modal({ title, onClose, children }: ModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.7)' }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="w-full max-w-md rounded-xl border p-6 shadow-2xl"
        style={{
          background: 'rgba(15,23,42,0.97)',
          borderColor: 'rgba(212,175,55,0.25)',
        }}
      >
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-amber-300">{title}</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 transition-colors"
          >
            <X size={18} />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

function PrimaryButton({
  onClick,
  disabled,
  loading,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  loading?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-slate-900 transition-opacity disabled:opacity-50"
      style={{
        background: disabled || loading
          ? 'rgba(212,175,55,0.5)'
          : 'linear-gradient(135deg,#f59e0b,#d97706)',
      }}
    >
      {loading && <Loader2 size={14} className="animate-spin" />}
      {children}
    </button>
  );
}

function SecondaryButton({
  onClick,
  disabled,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="px-4 py-2 rounded-lg font-medium text-amber-300 border border-amber-400/30 hover:border-amber-400/60 transition-colors disabled:opacity-50"
      style={{ background: 'transparent' }}
    >
      {children}
    </button>
  );
}

function InputField({
  label,
  value,
  onChange,
  placeholder,
  type = 'text',
  required,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  required?: boolean;
}) {
  return (
    <div className="mb-4">
      <label className="block text-xs font-medium text-slate-400 mb-1.5">
        {label}
        {required && <span className="text-amber-500 ml-1">*</span>}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-slate-100 placeholder-slate-500 text-sm outline-none focus:border-amber-400/60 transition-colors"
      />
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function GroupsTab({ walletAddress, getAuthHeader, apiBaseUrl }: GroupsTabProps) {
  const [groups, setGroups] = useState<GroupMeta[]>([]);
  const [groupsLoading, setGroupsLoading] = useState(true);
  const [groupsError, setGroupsError] = useState<string | null>(null);

  const [selectedGroup, setSelectedGroup] = useState<GroupMeta | null>(null);
  const [members, setMembers] = useState<GroupMember[]>([]);
  const [messages, setMessages] = useState<GroupMessage[]>([]);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [membersOpen, setMembersOpen] = useState(true);

  const [searchQuery, setSearchQuery] = useState('');
  const [composeText, setComposeText] = useState('');
  const [sending, setSending] = useState(false);

  // Modals
  const [showCreate, setShowCreate] = useState(false);
  const [showJoin, setShowJoin] = useState(false);
  const [showInvite, setShowInvite] = useState(false);
  const [showLeaveConfirm, setShowLeaveConfirm] = useState(false);
  const [showKickConfirm, setShowKickConfirm] = useState<GroupMember | null>(null);

  // Create form state
  const [createName, setCreateName] = useState('');
  const [createDesc, setCreateDesc] = useState('');
  const [createIcon, setCreateIcon] = useState('');
  const [createLoading, setCreateLoading] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  // Join form state
  const [joinToken, setJoinToken] = useState('');
  const [joinDisplay, setJoinDisplay] = useState('');
  const [joinLoading, setJoinLoading] = useState(false);
  const [joinError, setJoinError] = useState<string | null>(null);

  // Invite result
  const [inviteToken, setInviteToken] = useState<string | null>(null);
  const [inviteLoading, setInviteLoading] = useState(false);
  const [inviteError, setInviteError] = useState<string | null>(null);
  const [inviteCopied, setInviteCopied] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sseAbortRef = useRef<AbortController | null>(null);

  // ── Auth helper ────────────────────────────────────────────────────────────

  const authFetch = useCallback(
    async (url: string, options: RequestInit = {}) => {
      const token = await getAuthHeader();
      return fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'X-Wallet-Auth': token } : {}),
          ...(options.headers as Record<string, string> | undefined),
        },
      });
    },
    [getAuthHeader],
  );

  // ── Load groups ────────────────────────────────────────────────────────────

  const loadGroups = useCallback(async () => {
    setGroupsError(null);
    try {
      const res = await authFetch(`${apiBaseUrl}/api/v1/groups`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setGroups(data.groups ?? []);
    } catch (e: unknown) {
      setGroupsError(e instanceof Error ? e.message : 'Failed to load groups');
    }
  }, [authFetch, apiBaseUrl]);

  useEffect(() => {
    setGroupsLoading(true);
    loadGroups().finally(() => setGroupsLoading(false));
  }, [loadGroups]);

  // Auto-select first group after initial load
  const hasAutoSelected = useRef(false);
  useEffect(() => {
    if (!hasAutoSelected.current && !groupsLoading && groups.length > 0) {
      hasAutoSelected.current = true;
      selectGroup(groups[0]);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groupsLoading, groups]);

  // ── Load messages + members ────────────────────────────────────────────────

  const loadGroupDetail = useCallback(
    async (group: GroupMeta) => {
      setMessagesLoading(true);
      setMessages([]);
      setMembers([]);
      try {
        const [msgRes, detailRes] = await Promise.all([
          authFetch(`${apiBaseUrl}/api/v1/groups/${group.id}/messages?limit=50`),
          authFetch(`${apiBaseUrl}/api/v1/groups/${group.id}`),
        ]);
        if (msgRes.ok) {
          const msgData = await msgRes.json();
          setMessages(msgData.messages ?? []);
        }
        if (detailRes.ok) {
          const detailData = await detailRes.json();
          setMembers(detailData.members ?? []);
        }
      } catch {
        // non-fatal — show what we have
      } finally {
        setMessagesLoading(false);
      }
    },
    [authFetch, apiBaseUrl],
  );

  const refreshMembers = useCallback(
    async (groupId: string) => {
      try {
        const res = await authFetch(`${apiBaseUrl}/api/v1/groups/${groupId}`);
        if (res.ok) {
          const data = await res.json();
          setMembers(data.members ?? []);
        }
      } catch {
        // ignore
      }
    },
    [authFetch, apiBaseUrl],
  );

  // ── SSE ───────────────────────────────────────────────────────────────────

  const startSSE = useCallback(
    async (group: GroupMeta) => {
      // Cancel any existing SSE
      if (sseAbortRef.current) {
        sseAbortRef.current.abort();
      }
      const controller = new AbortController();
      sseAbortRef.current = controller;

      const token = await getAuthHeader();
      if (!token) return;

      try {
        const response = await fetch(
          `${apiBaseUrl}/api/v1/groups/${group.id}/stream`,
          {
            signal: controller.signal,
            headers: { 'X-Wallet-Auth': token },
          },
        );
        if (!response.ok || !response.body) return;

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        const processEvent = (eventText: string) => {
          const lines = eventText.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const jsonStr = line.slice(6).trim();
              if (!jsonStr) continue;
              try {
                const payload = JSON.parse(jsonStr);
                if (payload.type === 'new_message' && payload.message) {
                  setMessages((prev) => {
                    // Deduplicate by id
                    if (prev.some((m) => m.id === payload.message.id)) return prev;
                    return [...prev, payload.message];
                  });
                } else if (
                  payload.type === 'member_joined' ||
                  payload.type === 'member_left' ||
                  payload.type === 'member_kicked'
                ) {
                  refreshMembers(group.id);
                } else if (payload.type === 'group_deleted') {
                  setSelectedGroup(null);
                  setMessages([]);
                  setMembers([]);
                  loadGroups();
                }
              } catch {
                // malformed SSE data — ignore
              }
            }
          }
        };

        while (!controller.signal.aborted) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop() ?? '';
          for (const part of parts) {
            if (part.trim()) processEvent(part);
          }
        }
      } catch (e) {
        // AbortError is expected on cleanup
        if (e instanceof Error && e.name !== 'AbortError') {
          console.warn('SSE error:', e.message);
        }
      }
    },
    [getAuthHeader, apiBaseUrl, refreshMembers, loadGroups],
  );

  // ── Select group ──────────────────────────────────────────────────────────

  const selectGroup = useCallback(
    (group: GroupMeta) => {
      setSelectedGroup(group);
      setComposeText('');
      loadGroupDetail(group);
      startSSE(group);
    },
    [loadGroupDetail, startSSE],
  );

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      sseAbortRef.current?.abort();
    };
  }, []);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── Send message ──────────────────────────────────────────────────────────

  const sendMessage = useCallback(async () => {
    if (!selectedGroup || !composeText.trim() || sending) return;
    const content = composeText.trim();
    setComposeText('');
    setSending(true);

    // Optimistic append
    const optimistic: GroupMessage = {
      id: `optimistic-${Date.now()}`,
      group_id: selectedGroup.id,
      from: walletAddress,
      display_name: 'You',
      content,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, optimistic]);

    try {
      const res = await authFetch(
        `${apiBaseUrl}/api/v1/groups/${selectedGroup.id}/messages`,
        { method: 'POST', body: JSON.stringify({ content }) },
      );
      if (res.ok) {
        const data = await res.json();
        // Replace optimistic with real message
        setMessages((prev) =>
          prev.map((m) => (m.id === optimistic.id ? data.message : m)),
        );
      }
    } catch {
      // On failure, remove optimistic message
      setMessages((prev) => prev.filter((m) => m.id !== optimistic.id));
    } finally {
      setSending(false);
    }
  }, [selectedGroup, composeText, sending, walletAddress, authFetch, apiBaseUrl]);

  // ── Create group ──────────────────────────────────────────────────────────

  const handleCreate = useCallback(async () => {
    if (!createName.trim()) return;
    setCreateLoading(true);
    setCreateError(null);
    try {
      const res = await authFetch(`${apiBaseUrl}/api/v1/groups`, {
        method: 'POST',
        body: JSON.stringify({
          name: createName.trim(),
          description: createDesc.trim() || undefined,
          icon: createIcon.trim() || undefined,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      const newGroup: GroupMeta = data.group;
      setGroups((prev) => [newGroup, ...prev]);
      setShowCreate(false);
      setCreateName('');
      setCreateDesc('');
      setCreateIcon('');
      selectGroup(newGroup);
    } catch (e: unknown) {
      setCreateError(e instanceof Error ? e.message : 'Failed to create group');
    } finally {
      setCreateLoading(false);
    }
  }, [createName, createDesc, createIcon, authFetch, apiBaseUrl, selectGroup]);

  // ── Join group ────────────────────────────────────────────────────────────

  const handleJoin = useCallback(async () => {
    if (!joinToken.trim()) return;
    setJoinLoading(true);
    setJoinError(null);
    try {
      const res = await authFetch(`${apiBaseUrl}/api/v1/groups/join`, {
        method: 'POST',
        body: JSON.stringify({
          token: joinToken.trim(),
          display_name: joinDisplay.trim() || undefined,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      const joined: GroupMeta = data.group;
      setGroups((prev) => {
        if (prev.some((g) => g.id === joined.id)) return prev;
        return [joined, ...prev];
      });
      setShowJoin(false);
      setJoinToken('');
      setJoinDisplay('');
      selectGroup(joined);
    } catch (e: unknown) {
      setJoinError(e instanceof Error ? e.message : 'Failed to join group');
    } finally {
      setJoinLoading(false);
    }
  }, [joinToken, joinDisplay, authFetch, apiBaseUrl, selectGroup]);

  // ── Generate invite ───────────────────────────────────────────────────────

  const handleInvite = useCallback(async () => {
    if (!selectedGroup) return;
    setInviteLoading(true);
    setInviteError(null);
    setInviteToken(null);
    try {
      const res = await authFetch(
        `${apiBaseUrl}/api/v1/groups/${selectedGroup.id}/invites`,
        { method: 'POST', body: JSON.stringify({ expires_in_hours: 24, max_uses: 50 }) },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      setInviteToken(data.token);
    } catch (e: unknown) {
      setInviteError(e instanceof Error ? e.message : 'Failed to create invite');
    } finally {
      setInviteLoading(false);
    }
  }, [selectedGroup, authFetch, apiBaseUrl]);

  const copyInviteToken = useCallback(() => {
    if (!inviteToken) return;
    navigator.clipboard.writeText(inviteToken).catch(() => {});
    setInviteCopied(true);
    setTimeout(() => setInviteCopied(false), 2000);
  }, [inviteToken]);

  // ── Leave group ───────────────────────────────────────────────────────────

  const handleLeave = useCallback(async () => {
    if (!selectedGroup) return;
    try {
      await authFetch(`${apiBaseUrl}/api/v1/groups/${selectedGroup.id}/leave`, {
        method: 'POST',
      });
    } catch {
      // ignore — remove locally regardless
    }
    setGroups((prev) => prev.filter((g) => g.id !== selectedGroup.id));
    setSelectedGroup(null);
    setMessages([]);
    setMembers([]);
    setShowLeaveConfirm(false);
  }, [selectedGroup, authFetch, apiBaseUrl]);

  // ── Kick member ───────────────────────────────────────────────────────────

  const handleKick = useCallback(
    async (member: GroupMember) => {
      if (!selectedGroup) return;
      try {
        await authFetch(
          `${apiBaseUrl}/api/v1/groups/${selectedGroup.id}/members/${member.wallet}`,
          { method: 'DELETE' },
        );
        setMembers((prev) => prev.filter((m) => m.wallet !== member.wallet));
      } catch {
        // ignore
      }
      setShowKickConfirm(null);
    },
    [selectedGroup, authFetch, apiBaseUrl],
  );

  // ── Filtered groups ───────────────────────────────────────────────────────

  const filteredGroups = groups.filter((g) =>
    g.name.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  const isOwner = selectedGroup
    ? selectedGroup.owner.toLowerCase() === walletAddress.toLowerCase()
    : false;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div
      className="flex flex-col h-full select-none"
      style={{ background: 'rgba(15,23,42,0.95)' }}
    >
      {/* Top bar */}
      <div
        className="flex items-center gap-2 px-4 py-2.5 border-b flex-shrink-0"
        style={{ borderColor: 'rgba(212,175,55,0.15)' }}
      >
        {/* Action buttons */}
        <button
          onClick={() => {
            setShowCreate(true);
            setCreateError(null);
          }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-slate-900 transition-opacity hover:opacity-90"
          style={{ background: 'linear-gradient(135deg,#f59e0b,#d97706)' }}
        >
          <Plus size={14} />
          Create
        </button>
        <button
          onClick={() => {
            setShowJoin(true);
            setJoinError(null);
          }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-amber-300 border border-amber-400/30 hover:border-amber-400/60 transition-colors"
        >
          <UserPlus size={14} />
          Join
        </button>

        {/* Group count */}
        <span className="text-slate-400 text-sm ml-1">
          Groups
          {!groupsLoading && (
            <span className="ml-1 text-amber-400 font-medium">{groups.length}</span>
          )}
        </span>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Search */}
        <div className="relative">
          <Search
            size={13}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500"
          />
          <input
            type="text"
            placeholder="Search groups…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg pl-8 pr-3 py-1.5 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-amber-400/50 transition-colors w-44"
          />
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 min-h-0">
        {/* Left panel — groups list */}
        <div
          className="flex-shrink-0 flex flex-col border-r overflow-y-auto"
          style={{
            width: 220,
            borderColor: 'rgba(212,175,55,0.12)',
            background: 'rgba(15,23,42,0.6)',
          }}
        >
          {groupsLoading ? (
            <div className="flex items-center justify-center flex-1 text-slate-500">
              <Loader2 size={20} className="animate-spin" />
            </div>
          ) : groupsError ? (
            <div className="p-4 text-xs text-red-400 flex gap-2 items-start">
              <AlertCircle size={13} className="mt-0.5 flex-shrink-0" />
              {groupsError}
            </div>
          ) : filteredGroups.length === 0 ? (
            <div className="p-4 text-xs text-slate-500 text-center mt-4">
              {searchQuery ? 'No groups match your search.' : 'No groups yet. Create or join one!'}
            </div>
          ) : (
            <div className="py-2">
              {filteredGroups.map((group) => {
                const active = selectedGroup?.id === group.id;
                return (
                  <button
                    key={group.id}
                    onClick={() => selectGroup(group)}
                    className="w-full flex items-center gap-2 px-3 py-2 text-left transition-colors relative"
                    style={{
                      background: active ? 'rgba(212,175,55,0.08)' : 'transparent',
                      borderLeft: active ? '2px solid #f59e0b' : '2px solid transparent',
                    }}
                  >
                    {group.icon ? (
                      <span className="text-base leading-none flex-shrink-0">{group.icon}</span>
                    ) : (
                      <Hash
                        size={14}
                        className={active ? 'text-amber-400' : 'text-slate-500'}
                      />
                    )}
                    <span
                      className={`text-sm truncate ${active ? 'text-amber-200 font-medium' : 'text-slate-400'}`}
                    >
                      {group.name}
                    </span>
                    <span className="ml-auto text-xs text-slate-600 flex-shrink-0">
                      {group.member_count}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Right panel — chat */}
        {!selectedGroup ? (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-500 gap-3">
            <MessageSquare size={36} className="opacity-30" />
            <span className="text-sm">Select a group to start chatting</span>
          </div>
        ) : (
          <div className="flex-1 flex min-w-0">
            {/* Chat column */}
            <div className="flex-1 flex flex-col min-w-0">
              {/* Chat header */}
              <div
                className="flex items-center gap-2 px-4 py-2.5 border-b flex-shrink-0"
                style={{ borderColor: 'rgba(212,175,55,0.12)' }}
              >
                {selectedGroup.icon ? (
                  <span className="text-base">{selectedGroup.icon}</span>
                ) : (
                  <Hash size={15} className="text-amber-400" />
                )}
                <span className="font-semibold text-slate-100 text-sm">
                  {selectedGroup.name}
                </span>
                {selectedGroup.description && (
                  <span className="text-slate-500 text-xs truncate hidden sm:block">
                    — {selectedGroup.description}
                  </span>
                )}

                <div className="flex-1" />

                {/* Invite button */}
                <button
                  onClick={() => {
                    setShowInvite(true);
                    setInviteToken(null);
                    setInviteError(null);
                    setInviteCopied(false);
                    handleInvite();
                  }}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs text-amber-300 border border-amber-400/25 hover:border-amber-400/50 transition-colors"
                  title="Create invite"
                >
                  <Link size={12} />
                  Invite
                </button>

                {/* Members toggle */}
                <button
                  onClick={() => setMembersOpen((v) => !v)}
                  className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-slate-400 hover:text-slate-200 border border-slate-700 hover:border-slate-500 transition-colors"
                >
                  <Users size={12} />
                  {members.length}
                  <ChevronRight
                    size={12}
                    className={`transition-transform ${membersOpen ? 'rotate-180' : ''}`}
                  />
                </button>

                {/* Leave button */}
                <button
                  onClick={() => setShowLeaveConfirm(true)}
                  className="p-1.5 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-400/10 transition-colors"
                  title="Leave group"
                >
                  <LogOut size={13} />
                </button>
              </div>

              {/* Messages area */}
              <div className="flex-1 overflow-y-auto px-4 py-3 space-y-0.5">
                {messagesLoading ? (
                  <div className="flex items-center justify-center h-full text-slate-500">
                    <Loader2 size={20} className="animate-spin" />
                  </div>
                ) : messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-slate-600 gap-2">
                    <MessageSquare size={28} className="opacity-40" />
                    <span className="text-xs">No messages yet. Say hello!</span>
                  </div>
                ) : (
                  <>
                    {messages.map((msg, idx) => {
                      const prev = idx > 0 ? messages[idx - 1] : null;
                      const grouped = prev ? sameGrouping(prev, msg) : false;
                      const self = msg.from.toLowerCase() === walletAddress.toLowerCase();

                      return (
                        <div
                          key={msg.id}
                          className={`group ${grouped ? 'mt-0.5' : 'mt-3'}`}
                        >
                          {!grouped && (
                            <div className="flex items-baseline gap-2 mb-0.5">
                              <span
                                className={`text-sm font-semibold ${
                                  self ? 'text-amber-400' : 'text-amber-300'
                                }`}
                              >
                                {self ? 'You' : msg.display_name || shortAddr(msg.from)}
                              </span>
                              <span className="text-slate-600 text-xs">
                                {relativeTime(msg.timestamp)}
                              </span>
                            </div>
                          )}
                          <div className="flex items-start gap-2">
                            {grouped && (
                              <span className="w-0 invisible text-xs text-slate-700 group-hover:visible ml-0 min-w-fit pr-1">
                                {relativeTime(msg.timestamp)}
                              </span>
                            )}
                            <p className="text-slate-200 text-sm leading-relaxed break-words min-w-0">
                              {msg.content}
                            </p>
                          </div>
                        </div>
                      );
                    })}
                    <div ref={messagesEndRef} />
                  </>
                )}
              </div>

              {/* Divider */}
              <div
                className="h-px flex-shrink-0"
                style={{ background: 'rgba(212,175,55,0.1)' }}
              />

              {/* Compose bar */}
              <div className="flex items-center gap-2 px-4 py-3 flex-shrink-0">
                <input
                  type="text"
                  placeholder={`Message #${selectedGroup.name}…`}
                  value={composeText}
                  onChange={(e) => setComposeText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-amber-400/50 transition-colors"
                />
                <button
                  onClick={sendMessage}
                  disabled={!composeText.trim() || sending}
                  className="flex-shrink-0 flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-slate-900 transition-opacity disabled:opacity-40 hover:opacity-90"
                  style={{ background: 'linear-gradient(135deg,#f59e0b,#d97706)' }}
                >
                  {sending ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                </button>
              </div>
            </div>

            {/* Members sidebar */}
            {membersOpen && (
              <div
                className="flex-shrink-0 border-l flex flex-col overflow-hidden"
                style={{
                  width: 180,
                  borderColor: 'rgba(212,175,55,0.12)',
                  background: 'rgba(15,23,42,0.5)',
                }}
              >
                <div
                  className="px-3 py-2.5 border-b text-xs font-semibold text-slate-400 uppercase tracking-wider flex-shrink-0"
                  style={{ borderColor: 'rgba(212,175,55,0.1)' }}
                >
                  Members — {members.length}
                </div>
                <div className="flex-1 overflow-y-auto py-2">
                  {members.map((m) => {
                    const self = m.wallet.toLowerCase() === walletAddress.toLowerCase();
                    const canKick = isOwner && !self;
                    return (
                      <div
                        key={m.wallet}
                        className="group flex items-center gap-2 px-3 py-1.5 hover:bg-slate-800/50 transition-colors"
                      >
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1 min-w-0">
                            <RoleBadge role={m.role} />
                            <span
                              className={`text-xs truncate ${self ? 'text-amber-300' : 'text-slate-300'}`}
                            >
                              {m.display_name || shortAddr(m.wallet)}
                            </span>
                          </div>
                        </div>
                        {canKick && (
                          <button
                            onClick={() => setShowKickConfirm(m)}
                            className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-600 hover:text-red-400 flex-shrink-0"
                            title="Kick member"
                          >
                            <X size={11} />
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Create group modal ──────────────────────────────────────────────── */}
      {showCreate && (
        <Modal title="Create Group" onClose={() => setShowCreate(false)}>
          <InputField
            label="Group Name"
            value={createName}
            onChange={setCreateName}
            placeholder="My Awesome Group"
            required
          />
          <InputField
            label="Description"
            value={createDesc}
            onChange={setCreateDesc}
            placeholder="What's this group about?"
          />
          <InputField
            label="Icon (single character or emoji)"
            value={createIcon}
            onChange={(v) => setCreateIcon(v.slice(0, 2))}
            placeholder="Optional icon"
          />
          {createError && (
            <div className="mb-4 flex items-center gap-2 text-red-400 text-xs">
              <AlertCircle size={12} />
              {createError}
            </div>
          )}
          <div className="flex justify-end gap-2 mt-2">
            <SecondaryButton onClick={() => setShowCreate(false)}>Cancel</SecondaryButton>
            <PrimaryButton
              onClick={handleCreate}
              disabled={!createName.trim()}
              loading={createLoading}
            >
              Create Group
            </PrimaryButton>
          </div>
        </Modal>
      )}

      {/* ── Join group modal ────────────────────────────────────────────────── */}
      {showJoin && (
        <Modal title="Join Group" onClose={() => setShowJoin(false)}>
          <InputField
            label="Invite Token"
            value={joinToken}
            onChange={setJoinToken}
            placeholder="Paste your invite token"
            required
          />
          <InputField
            label="Display Name (optional)"
            value={joinDisplay}
            onChange={setJoinDisplay}
            placeholder="How others will see you"
          />
          {joinError && (
            <div className="mb-4 flex items-center gap-2 text-red-400 text-xs">
              <AlertCircle size={12} />
              {joinError}
            </div>
          )}
          <div className="flex justify-end gap-2 mt-2">
            <SecondaryButton onClick={() => setShowJoin(false)}>Cancel</SecondaryButton>
            <PrimaryButton
              onClick={handleJoin}
              disabled={!joinToken.trim()}
              loading={joinLoading}
            >
              Join Group
            </PrimaryButton>
          </div>
        </Modal>
      )}

      {/* ── Invite modal ────────────────────────────────────────────────────── */}
      {showInvite && (
        <Modal
          title="Invite to Group"
          onClose={() => {
            setShowInvite(false);
            setInviteToken(null);
          }}
        >
          {inviteLoading ? (
            <div className="flex items-center justify-center py-6 text-slate-400 gap-2">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-sm">Generating invite…</span>
            </div>
          ) : inviteError ? (
            <div className="flex items-center gap-2 text-red-400 text-sm py-4">
              <AlertCircle size={14} />
              {inviteError}
            </div>
          ) : inviteToken ? (
            <div>
              <p className="text-slate-400 text-xs mb-3">
                Share this token with people you want to invite. It expires in 24 hours.
              </p>
              <div
                className="flex items-center gap-2 rounded-lg px-3 py-2.5 border"
                style={{
                  background: 'rgba(212,175,55,0.06)',
                  borderColor: 'rgba(212,175,55,0.2)',
                }}
              >
                <code className="flex-1 text-xs text-amber-200 break-all select-all font-mono">
                  {inviteToken}
                </code>
                <button
                  onClick={copyInviteToken}
                  className="flex-shrink-0 text-slate-400 hover:text-amber-300 transition-colors"
                  title="Copy token"
                >
                  {inviteCopied ? (
                    <Check size={14} className="text-green-400" />
                  ) : (
                    <Copy size={14} />
                  )}
                </button>
              </div>
              <div className="flex justify-end mt-4">
                <PrimaryButton
                  onClick={() => {
                    setShowInvite(false);
                    setInviteToken(null);
                  }}
                >
                  Done
                </PrimaryButton>
              </div>
            </div>
          ) : null}
        </Modal>
      )}

      {/* ── Leave confirm modal ─────────────────────────────────────────────── */}
      {showLeaveConfirm && selectedGroup && (
        <Modal title="Leave Group" onClose={() => setShowLeaveConfirm(false)}>
          <p className="text-slate-300 text-sm mb-6">
            Are you sure you want to leave{' '}
            <span className="text-amber-300 font-medium">{selectedGroup.name}</span>?
            {isOwner && (
              <span className="block mt-2 text-xs text-amber-500">
                You are the owner. Leaving will transfer ownership or delete the group.
              </span>
            )}
          </p>
          <div className="flex justify-end gap-2">
            <SecondaryButton onClick={() => setShowLeaveConfirm(false)}>Cancel</SecondaryButton>
            <button
              onClick={handleLeave}
              className="px-4 py-2 rounded-lg font-medium text-white bg-red-600 hover:bg-red-500 transition-colors text-sm flex items-center gap-2"
            >
              <LogOut size={13} />
              Leave Group
            </button>
          </div>
        </Modal>
      )}

      {/* ── Kick confirm modal ──────────────────────────────────────────────── */}
      {showKickConfirm && (
        <Modal title="Kick Member" onClose={() => setShowKickConfirm(null)}>
          <p className="text-slate-300 text-sm mb-6">
            Remove{' '}
            <span className="text-amber-300 font-medium">
              {showKickConfirm.display_name || shortAddr(showKickConfirm.wallet)}
            </span>{' '}
            from the group?
          </p>
          <div className="flex justify-end gap-2">
            <SecondaryButton onClick={() => setShowKickConfirm(null)}>Cancel</SecondaryButton>
            <button
              onClick={() => handleKick(showKickConfirm)}
              className="px-4 py-2 rounded-lg font-medium text-white bg-red-600 hover:bg-red-500 transition-colors text-sm flex items-center gap-2"
            >
              <X size={13} />
              Kick Member
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
