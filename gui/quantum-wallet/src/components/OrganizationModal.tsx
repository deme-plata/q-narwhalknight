// OrganizationModal — CEO-owned organization with per-member spending policy.
//
// WHAT THIS IS
// ------------
// The operator's ask: "a CEO wallet with ACL can create an organization with
// members with different adjustment to money usage of wallet."
//
// This modal is the front half of that. It models EXACTLY the schema the node
// will enforce, so the UI does not have to be rewritten once the backend
// lands:
//
//   Organization { name, ceo, treasury, members[], created_at }
//   Member       { address, name, role, per_tx_limit, daily_limit,
//                  approval_threshold, approvals_required }
//
// HONEST STATUS
// -------------
// Limits shown here are a DRAFT until the node enforces them. The chain-side
// pieces (q-multisig wiring, an Org column family in RocksDB, and a policy
// check in the transaction validation path) are not built yet. Until they are,
// a member holding their own seed can bypass any limit by signing a plain
// transfer — the UI cannot stop that, and this modal says so rather than
// implying a safety that does not exist.
//
// Drafts persist to localStorage under `quillon:org:draft` so the operator can
// design the org now and deploy it in one action when the backend ships.

import { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Users,
  Crown,
  Plus,
  Trash2,
  ShieldCheck,
  AlertTriangle,
  Building2,
  Wallet,
  Signature,
  Info,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Schema — mirrors the planned on-chain Organization record
// ---------------------------------------------------------------------------

export type OrgRole = 'ceo' | 'cfo' | 'manager' | 'employee' | 'viewer';

export interface OrgMember {
  address: string;
  name: string;
  role: OrgRole;
  /** Max QUG in a single transaction. 0 = cannot spend at all. */
  perTxLimit: number;
  /** Max QUG total in a rolling 24h window. 0 = cannot spend at all. */
  dailyLimit: number;
  /** Transactions at or above this amount need co-signatures. */
  approvalThreshold: number;
  /** How many OTHER members must co-sign above the threshold. */
  approvalsRequired: number;
}

export interface OrgDraft {
  version: 1;
  name: string;
  ceo: string;
  treasury: string;
  members: OrgMember[];
  createdAt: string;
}

const DRAFT_KEY = 'quillon:org:draft';

interface RolePreset {
  id: OrgRole;
  label: string;
  emoji: string;
  accent: string;
  blurb: string;
  defaults: Omit<OrgMember, 'address' | 'name' | 'role'>;
}

/**
 * Role presets. These are STARTING POINTS — every number stays editable per
 * member, which is the "different adjustment to money usage" part of the ask.
 * A role is a convenient bundle of limits, not a hard class.
 */
const ROLE_PRESETS: RolePreset[] = [
  {
    id: 'ceo',
    label: 'CEO',
    emoji: '👑',
    accent: 'amber',
    blurb: 'Owns the organization. Can add/remove members and change any limit. Unlimited spend.',
    defaults: { perTxLimit: 0, dailyLimit: 0, approvalThreshold: 0, approvalsRequired: 0 },
  },
  {
    id: 'cfo',
    label: 'CFO',
    emoji: '🏦',
    accent: 'emerald',
    blurb: 'Treasury operator. High limits, co-signs large movements, cannot change membership.',
    defaults: { perTxLimit: 50_000, dailyLimit: 200_000, approvalThreshold: 25_000, approvalsRequired: 1 },
  },
  {
    id: 'manager',
    label: 'Manager',
    emoji: '📊',
    accent: 'violet',
    blurb: 'Departmental budget. Moderate limits, needs a co-signature on anything sizeable.',
    defaults: { perTxLimit: 5_000, dailyLimit: 20_000, approvalThreshold: 2_500, approvalsRequired: 1 },
  },
  {
    id: 'employee',
    label: 'Employee',
    emoji: '👤',
    accent: 'cyan',
    blurb: 'Day-to-day expenses. Small limits, everything above petty cash needs approval.',
    defaults: { perTxLimit: 500, dailyLimit: 2_000, approvalThreshold: 250, approvalsRequired: 1 },
  },
  {
    id: 'viewer',
    label: 'Viewer',
    emoji: '👁',
    accent: 'slate',
    blurb: 'Read-only. Sees the treasury and history, cannot move any funds.',
    defaults: { perTxLimit: 0, dailyLimit: 0, approvalThreshold: 0, approvalsRequired: 0 },
  },
];

function presetFor(role: OrgRole): RolePreset {
  return ROLE_PRESETS.find(r => r.id === role) ?? ROLE_PRESETS[3];
}

function accentStyle(accent: string): { bg: string; border: string; text: string } {
  const m: Record<string, { bg: string; border: string; text: string }> = {
    amber:   { bg: 'rgba(245,158,11,0.10)', border: 'rgba(245,158,11,0.30)', text: 'text-amber-200' },
    emerald: { bg: 'rgba(16,185,129,0.10)', border: 'rgba(16,185,129,0.30)', text: 'text-emerald-200' },
    violet:  { bg: 'rgba(168,85,247,0.10)', border: 'rgba(168,85,247,0.30)', text: 'text-violet-200' },
    cyan:    { bg: 'rgba(34,211,238,0.10)', border: 'rgba(34,211,238,0.30)', text: 'text-cyan-200' },
    slate:   { bg: 'rgba(100,116,139,0.10)', border: 'rgba(100,116,139,0.30)', text: 'text-slate-300' },
  };
  return m[accent] ?? m.violet;
}

function truncate(addr: string): string {
  if (!addr) return '—';
  if (addr.length < 16) return addr;
  return `${addr.slice(0, 10)}…${addr.slice(-6)}`;
}

function fmt(n: number): string {
  if (n === 0) return 'unlimited';
  return n.toLocaleString('en-US');
}

function loadDraft(): OrgDraft | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed?.version === 1 ? parsed : null;
  } catch {
    return null;
  }
}

function saveDraft(draft: OrgDraft): void {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  } catch (e) {
    console.error('[OrganizationModal] Failed to persist draft:', e);
  }
}

// ---------------------------------------------------------------------------

interface OrganizationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function OrganizationModal({ isOpen, onClose }: OrganizationModalProps) {
  const [draft, setDraft] = useState<OrgDraft | null>(null);
  const [orgName, setOrgName] = useState('');
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // New-member form
  const [newAddress, setNewAddress] = useState('');
  const [newName, setNewName] = useState('');
  const [newRole, setNewRole] = useState<OrgRole>('employee');

  const walletAddress = typeof window !== 'undefined'
    ? (localStorage.getItem('walletAddress') || '')
    : '';

  useEffect(() => {
    if (!isOpen) return;
    const existing = loadDraft();
    setDraft(existing);
    setOrgName(existing?.name ?? '');
    setAdding(false);
    setError(null);
    setNewAddress('');
    setNewName('');
    setNewRole('employee');
  }, [isOpen]);

  const createOrg = () => {
    if (!orgName.trim()) { setError('Give the organization a name.'); return; }
    if (!walletAddress) { setError('No active wallet — log in first.'); return; }

    const fresh: OrgDraft = {
      version: 1,
      name: orgName.trim(),
      ceo: walletAddress,
      // Treasury defaults to the CEO wallet. Once the backend lands this
      // becomes a derived multisig address from the member set + threshold,
      // exactly like MultisigWallet::new() computes it in q-multisig.
      treasury: walletAddress,
      members: [{
        address: walletAddress,
        name: 'You (CEO)',
        role: 'ceo',
        ...presetFor('ceo').defaults,
      }],
      createdAt: new Date().toISOString(),
    };
    setDraft(fresh);
    saveDraft(fresh);
    setError(null);
  };

  const addMember = () => {
    if (!draft) return;
    const addr = newAddress.trim();
    if (!addr.startsWith('qnk') || addr.length < 20) {
      setError('Member address must be a full qnk… address.');
      return;
    }
    if (draft.members.some(m => m.address === addr)) {
      setError('That address is already a member.');
      return;
    }
    const next: OrgDraft = {
      ...draft,
      members: [...draft.members, {
        address: addr,
        name: newName.trim() || truncate(addr),
        role: newRole,
        ...presetFor(newRole).defaults,
      }],
    };
    setDraft(next);
    saveDraft(next);
    setAdding(false);
    setNewAddress('');
    setNewName('');
    setNewRole('employee');
    setError(null);
  };

  const removeMember = (address: string) => {
    if (!draft) return;
    if (address === draft.ceo) { setError('The CEO cannot be removed from their own organization.'); return; }
    const next = { ...draft, members: draft.members.filter(m => m.address !== address) };
    setDraft(next);
    saveDraft(next);
  };

  const updateMember = (address: string, patch: Partial<OrgMember>) => {
    if (!draft) return;
    const next = {
      ...draft,
      members: draft.members.map(m => (m.address === address ? { ...m, ...patch } : m)),
    };
    setDraft(next);
    saveDraft(next);
  };

  /** Changing role re-applies that role's preset limits. */
  const changeRole = (address: string, role: OrgRole) => {
    updateMember(address, { role, ...presetFor(role).defaults });
  };

  const totalDailyExposure = useMemo(() => {
    if (!draft) return 0;
    // CEO/unlimited members contribute nothing measurable — report only the
    // bounded members, so the number means "what capped members can move".
    return draft.members.reduce((sum, m) => sum + (m.dailyLimit > 0 ? m.dailyLimit : 0), 0);
  }, [draft]);

  const unlimitedCount = useMemo(
    () => (draft?.members.filter(m => m.role !== 'viewer' && m.dailyLimit === 0).length ?? 0),
    [draft]
  );

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[260] flex items-center justify-center p-4"
          style={{ background: 'rgba(2,4,15,0.82)', backdropFilter: 'blur(8px)' }}
          onClick={onClose}
        >
          <motion.div
            initial={{ y: 24, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 24, opacity: 0 }}
            transition={{ type: 'spring', duration: 0.35, bounce: 0.16 }}
            className="rounded-3xl w-full max-w-3xl max-h-[90vh] overflow-hidden border-2 flex flex-col"
            style={{
              background: 'linear-gradient(160deg, rgba(10,11,30,0.98) 0%, rgba(24,16,8,0.98) 100%)',
              borderColor: 'rgba(212,175,55,0.35)',
              boxShadow: '0 0 60px rgba(212,175,55,0.20)',
            }}
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="px-5 py-4 border-b border-amber-500/15 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Building2 className="w-5 h-5 text-amber-400" />
                <h2 className="text-base font-bold text-white">Organization &amp; Multi-Sig</h2>
                {draft && (
                  <span className="text-[10px] uppercase tracking-widest font-bold px-2 py-0.5 rounded-md bg-amber-500/15 text-amber-300 border border-amber-500/30">
                    {draft.members.length} member{draft.members.length === 1 ? '' : 's'}
                  </span>
                )}
              </div>
              <motion.button whileHover={{ rotate: 90, scale: 1.1 }} onClick={onClose} className="text-slate-400 hover:text-white">
                <X className="w-5 h-5" />
              </motion.button>
            </div>

            {/* Honest status banner — do not remove until the node enforces this */}
            <div className="px-5 py-3 bg-amber-500/10 border-b border-amber-500/25">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-300 flex-shrink-0 mt-0.5" />
                <p className="text-[11px] text-amber-100 leading-relaxed">
                  <strong>Draft — not yet enforced by the network.</strong> The limits below are
                  saved in this browser and describe the policy the node will apply once
                  chain-side enforcement ships. Until then a member who holds their own seed can
                  still sign an ordinary transfer and bypass these rules. Treat this as design,
                  not as a control.
                </p>
              </div>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto p-5">
              {!draft ? (
                /* ── Create org ─────────────────────────────────────────── */
                <div className="max-w-md mx-auto py-8 text-center">
                  <Crown className="w-12 h-12 mx-auto mb-4 text-amber-400 opacity-70" />
                  <h3 className="text-lg font-bold text-white mb-2">Create an organization</h3>
                  <p className="text-xs text-slate-400 leading-relaxed mb-6">
                    Your current wallet becomes the CEO — the only role that can add or remove
                    members and change spending limits. Everyone else gets a role with its own
                    caps, which you can tune per person.
                  </p>

                  <div className="text-left mb-4 p-3 rounded-xl bg-slate-950/60 border border-amber-500/20">
                    <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold mb-1">CEO wallet</p>
                    <p className="text-xs font-mono text-amber-200">{truncate(walletAddress)}</p>
                  </div>

                  <input
                    value={orgName}
                    onChange={e => setOrgName(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') createOrg(); }}
                    placeholder="Organization name"
                    className="w-full px-3 py-2.5 rounded-xl bg-slate-950/70 border border-amber-500/30 text-white text-sm placeholder:text-slate-600 focus:outline-none focus:border-amber-400/70 mb-3"
                  />

                  {error && (
                    <p className="text-xs text-rose-300 mb-3">{error}</p>
                  )}

                  <button
                    onClick={createOrg}
                    className="w-full py-2.5 rounded-xl text-slate-900 text-sm font-bold"
                    style={{ background: 'linear-gradient(135deg, #fbbf24 0%, #d97706 100%)' }}
                  >
                    Create organization
                  </button>
                </div>
              ) : (
                /* ── Org detail ─────────────────────────────────────────── */
                <div className="space-y-4">
                  {/* Summary */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="p-3 rounded-xl bg-slate-950/60 border border-amber-500/20">
                      <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold mb-1">Organization</p>
                      <p className="text-sm font-bold text-white truncate">{draft.name}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-950/60 border border-amber-500/20">
                      <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold mb-1 flex items-center gap-1">
                        <Wallet className="w-3 h-3" /> Treasury
                      </p>
                      <p className="text-xs font-mono text-amber-200">{truncate(draft.treasury)}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-950/60 border border-amber-500/20">
                      <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold mb-1">Capped daily outflow</p>
                      <p className="text-sm font-bold text-white">
                        {totalDailyExposure.toLocaleString('en-US')} <span className="text-[10px] text-slate-500">QUG</span>
                      </p>
                      {unlimitedCount > 0 && (
                        <p className="text-[10px] text-amber-400 mt-0.5">
                          + {unlimitedCount} uncapped member{unlimitedCount === 1 ? '' : 's'}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Members */}
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Users className="w-4 h-4 text-amber-400" /> Members
                    </h3>
                    <button
                      onClick={() => { setAdding(true); setError(null); }}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/35 text-amber-200 text-[11px] font-bold transition-colors"
                    >
                      <Plus className="w-3.5 h-3.5" /> Add member
                    </button>
                  </div>

                  {/* Add-member form */}
                  <AnimatePresence>
                    {adding && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="overflow-hidden"
                      >
                        <div className="p-3 rounded-xl bg-slate-950/70 border border-amber-500/25 space-y-2">
                          <input
                            value={newAddress}
                            onChange={e => setNewAddress(e.target.value)}
                            placeholder="qnk… member wallet address"
                            className="w-full px-3 py-2 rounded-lg bg-slate-900/80 border border-slate-700 text-white text-xs font-mono placeholder:text-slate-600 focus:outline-none focus:border-amber-400/60"
                          />
                          <input
                            value={newName}
                            onChange={e => setNewName(e.target.value)}
                            placeholder="Display name (optional)"
                            className="w-full px-3 py-2 rounded-lg bg-slate-900/80 border border-slate-700 text-white text-xs placeholder:text-slate-600 focus:outline-none focus:border-amber-400/60"
                          />
                          <div className="flex flex-wrap gap-1.5">
                            {ROLE_PRESETS.filter(r => r.id !== 'ceo').map(r => {
                              const s = accentStyle(r.accent);
                              const on = newRole === r.id;
                              return (
                                <button
                                  key={r.id}
                                  onClick={() => setNewRole(r.id)}
                                  title={r.blurb}
                                  className={`px-2.5 py-1.5 rounded-lg text-[11px] font-bold border transition-all ${on ? 'ring-2 ring-amber-400/50' : ''} ${s.text}`}
                                  style={{ background: s.bg, borderColor: s.border }}
                                >
                                  {r.emoji} {r.label}
                                </button>
                              );
                            })}
                          </div>
                          <p className="text-[10px] text-slate-500 leading-relaxed">
                            {presetFor(newRole).blurb}
                          </p>
                          <div className="flex gap-2 pt-1">
                            <button
                              onClick={() => { setAdding(false); setError(null); }}
                              className="flex-1 py-2 rounded-lg text-slate-300 text-xs font-bold bg-slate-800/60 hover:bg-slate-700/60"
                            >
                              Cancel
                            </button>
                            <button
                              onClick={addMember}
                              className="flex-1 py-2 rounded-lg text-slate-900 text-xs font-bold"
                              style={{ background: 'linear-gradient(135deg, #fbbf24 0%, #d97706 100%)' }}
                            >
                              Add
                            </button>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {error && (
                    <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-200 text-xs flex items-start gap-2">
                      <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                      {error}
                    </div>
                  )}

                  {/* Member rows */}
                  <div className="space-y-2">
                    {draft.members.map(m => {
                      const preset = presetFor(m.role);
                      const s = accentStyle(preset.accent);
                      const isCeo = m.role === 'ceo';
                      return (
                        <div
                          key={m.address}
                          className="rounded-xl p-3 border"
                          style={{ background: s.bg, borderColor: s.border }}
                        >
                          <div className="flex items-start justify-between gap-3 mb-2">
                            <div className="min-w-0">
                              <div className="flex items-center gap-1.5 flex-wrap">
                                <span className="text-lg">{preset.emoji}</span>
                                <p className={`text-sm font-bold ${s.text}`}>{m.name}</p>
                                <span className="text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-slate-900/60 text-slate-300 border border-slate-700">
                                  {preset.label}
                                </span>
                              </div>
                              <p className="text-[10px] text-slate-500 font-mono mt-0.5">{truncate(m.address)}</p>
                            </div>
                            {!isCeo && (
                              <button
                                onClick={() => removeMember(m.address)}
                                title="Remove member"
                                className="text-slate-500 hover:text-rose-400 transition-colors flex-shrink-0"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            )}
                          </div>

                          {/* Role switcher */}
                          {!isCeo && (
                            <div className="flex flex-wrap gap-1 mb-2">
                              {ROLE_PRESETS.filter(r => r.id !== 'ceo').map(r => (
                                <button
                                  key={r.id}
                                  onClick={() => changeRole(m.address, r.id)}
                                  className={`px-2 py-0.5 rounded text-[10px] font-bold transition-all ${
                                    m.role === r.id
                                      ? 'bg-amber-500/30 text-amber-100 ring-1 ring-amber-400/50'
                                      : 'bg-slate-900/50 text-slate-500 hover:text-slate-300'
                                  }`}
                                >
                                  {r.label}
                                </button>
                              ))}
                            </div>
                          )}

                          {/* Limits — the "different adjustment to money usage" controls */}
                          {isCeo ? (
                            <p className="text-[10px] text-amber-300/80 flex items-center gap-1.5">
                              <Crown className="w-3 h-3" />
                              Unlimited spend · can add/remove members and change every limit
                            </p>
                          ) : (
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                              <LimitField
                                label="Per tx"
                                value={m.perTxLimit}
                                onChange={v => updateMember(m.address, { perTxLimit: v })}
                              />
                              <LimitField
                                label="Per day"
                                value={m.dailyLimit}
                                onChange={v => updateMember(m.address, { dailyLimit: v })}
                              />
                              <LimitField
                                label="Approval above"
                                value={m.approvalThreshold}
                                onChange={v => updateMember(m.address, { approvalThreshold: v })}
                              />
                              <div>
                                <label className="text-[9px] uppercase tracking-wider text-slate-500 font-bold block mb-1">
                                  Co-signers
                                </label>
                                <select
                                  value={m.approvalsRequired}
                                  onChange={e => updateMember(m.address, { approvalsRequired: Number(e.target.value) })}
                                  className="w-full px-2 py-1.5 rounded-lg bg-slate-900/80 border border-slate-700 text-white text-xs focus:outline-none focus:border-amber-400/60"
                                >
                                  {[0, 1, 2, 3].map(n => (
                                    <option key={n} value={n}>{n}</option>
                                  ))}
                                </select>
                              </div>
                            </div>
                          )}

                          {/* Plain-language restatement of the policy row */}
                          {!isCeo && (
                            <p className="text-[10px] text-slate-500 mt-2 leading-relaxed flex items-start gap-1.5">
                              <Info className="w-3 h-3 flex-shrink-0 mt-0.5" />
                              {m.perTxLimit === 0 && m.dailyLimit === 0
                                ? 'Cannot move funds at all — read-only access to the treasury.'
                                : <>
                                    Can send up to <strong className="text-slate-300">{fmt(m.perTxLimit)}</strong> QUG
                                    per transaction and <strong className="text-slate-300">{fmt(m.dailyLimit)}</strong> QUG
                                    per day.{' '}
                                    {m.approvalsRequired > 0
                                      ? <>Anything at or above <strong className="text-slate-300">{fmt(m.approvalThreshold)}</strong> QUG
                                          needs <strong className="text-slate-300">{m.approvalsRequired}</strong> co-signature
                                          {m.approvalsRequired === 1 ? '' : 's'}.</>
                                      : 'No co-signature required.'}
                                  </>}
                            </p>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Footer */}
            {draft && (
              <div className="px-5 py-3 border-t border-amber-500/15 bg-slate-950/60 flex items-center justify-between gap-3">
                <p className="text-[10px] text-slate-500 leading-relaxed flex items-center gap-1.5">
                  <Signature className="w-3 h-3 flex-shrink-0" />
                  Draft saved locally. Deploying writes the org + policy on-chain.
                </p>
                <button
                  disabled
                  title="Chain-side enforcement is not built yet — see the banner above"
                  className="px-4 py-2 rounded-xl text-xs font-bold bg-slate-800/70 text-slate-500 cursor-not-allowed inline-flex items-center gap-2 flex-shrink-0"
                >
                  <ShieldCheck className="w-3.5 h-3.5" />
                  Deploy on-chain
                </button>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/** Numeric limit input. 0 is a meaningful value ("unlimited" / "cannot spend"). */
function LimitField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="text-[9px] uppercase tracking-wider text-slate-500 font-bold block mb-1">
        {label}
      </label>
      <input
        type="number"
        min={0}
        value={value}
        onChange={e => onChange(Math.max(0, Number(e.target.value) || 0))}
        className="w-full px-2 py-1.5 rounded-lg bg-slate-900/80 border border-slate-700 text-white text-xs focus:outline-none focus:border-amber-400/60"
      />
    </div>
  );
}
