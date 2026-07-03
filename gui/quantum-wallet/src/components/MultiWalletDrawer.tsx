// MultiWalletDrawer — switch between multiple Quillon wallets from the topbar.
//
// localStorage layout:
//   walletAddress       — currently-active wallet (string, existing)
//   quillon:wallets     — JSON array of { address, name, template, createdAt }
//                         (new this PR — old single-wallet sessions get auto-
//                         migrated to a one-entry array on first open)
//
// The "+" button surfaces 5 templates so the user picks the *purpose* of the
// new wallet (Savings / Trading / Mining / Agent / Faucet) rather than just
// generating a featureless extra key. The template gets stored alongside the
// address so other surfaces can adapt their default UI (e.g. show DCA controls
// prominently for Trading, hide them for Savings).
//
// Seed generation: crypto.getRandomValues(32 bytes) → sha3_256 → ed25519
// priv → ed25519 pubkey → 'qnk' + hex(pubkey). This matches the X-Wallet-Auth
// derivation used everywhere else in the app + the MCP.

import { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Plus,
  Check,
  Wallet,
  Sparkles,
  TrendingUp,
  Pickaxe,
  Bot,
  Gift,
  PiggyBank,
  Copy,
  ArrowRightLeft,
  AlertCircle,
} from 'lucide-react';
import * as ed25519 from '@noble/ed25519';
import { sha3_256 } from '@noble/hashes/sha3.js';
import { bytesToHex } from '@noble/hashes/utils.js';

export type WalletTemplateId = 'savings' | 'trading' | 'mining' | 'agent' | 'faucet';

interface WalletEntry {
  address: string;
  name: string;
  template: WalletTemplateId | 'main';
  createdAt: string;
}

interface Template {
  id: WalletTemplateId;
  label: string;
  emoji: string;
  icon: React.ComponentType<{ className?: string }>;
  accent: string;          // tailwind palette key
  oneLine: string;
  details: string;
}

const TEMPLATES: Template[] = [
  {
    id: 'savings',
    label: 'Savings',
    emoji: '🏦',
    icon: PiggyBank,
    accent: 'emerald',
    oneLine: 'Hold QUG long-term.',
    details: 'Cold-storage feel. Defaults to no auto-trading, daily balance ping, and a notification if anything moves out unexpectedly.',
  },
  {
    id: 'trading',
    label: 'Trading',
    emoji: '📈',
    icon: TrendingUp,
    accent: 'fuchsia',
    oneLine: 'DEX + DCA + bot strategies.',
    details: 'Surfaces swap quotes, dex_swap one-liners, water-robot tools. Defaults to higher per-tx slippage tolerance because you know what you\'re doing.',
  },
  {
    id: 'mining',
    label: 'Mining',
    emoji: '⛏',
    icon: Pickaxe,
    accent: 'amber',
    oneLine: 'Pool payouts + miner control.',
    details: 'Receive miner rewards here. Hashrate widgets in topbar default to this wallet. Auto-tags incoming coinbase txs as "mining reward" in the inbox.',
  },
  {
    id: 'agent',
    label: 'Agent',
    emoji: '🤖',
    icon: Bot,
    accent: 'violet',
    oneLine: 'X-Wallet-Auth signing for AI agents.',
    details: 'Marks the wallet as agent-operated (per the Skin in the Cathedral framing). Opts into the Connected Agents panel on the public topbar. Restricts max-amount-per-tx by default.',
  },
  {
    id: 'faucet',
    label: 'Faucet',
    emoji: '🎁',
    icon: Gift,
    accent: 'cyan',
    oneLine: 'Small tips + outbound only.',
    details: 'Designed to hold a small float for tipping, gifts, and quick demos. UI hides the long-history view, foregrounds Send. Warn if balance drops below 1 QUG.',
  },
];

function paletteFor(accent: string): { bg: string; border: string; text: string; ring: string } {
  const m: Record<string, { bg: string; border: string; text: string; ring: string }> = {
    emerald: { bg: 'rgba(16,185,129,0.10)', border: 'rgba(16,185,129,0.30)', text: 'text-emerald-200', ring: 'ring-emerald-400/40' },
    fuchsia: { bg: 'rgba(217,70,239,0.10)', border: 'rgba(217,70,239,0.30)', text: 'text-fuchsia-200', ring: 'ring-fuchsia-400/40' },
    amber:   { bg: 'rgba(245,158,11,0.10)', border: 'rgba(245,158,11,0.30)', text: 'text-amber-200',   ring: 'ring-amber-400/40' },
    violet:  { bg: 'rgba(168,85,247,0.10)', border: 'rgba(168,85,247,0.30)', text: 'text-violet-200',  ring: 'ring-violet-400/40' },
    cyan:    { bg: 'rgba(34,211,238,0.10)', border: 'rgba(34,211,238,0.30)', text: 'text-cyan-200',    ring: 'ring-cyan-400/40' },
  };
  return m[accent] ?? m.violet;
}

/**
 * Load the wallet list from localStorage, migrating single-wallet sessions
 * to a one-entry array on first open.
 */
function loadWallets(): WalletEntry[] {
  try {
    const raw = localStorage.getItem('quillon:wallets');
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch { /* corrupted — fall through */ }

  const current = localStorage.getItem('walletAddress');
  if (current) {
    const seed: WalletEntry[] = [{ address: current, name: 'Primary', template: 'main', createdAt: new Date().toISOString() }];
    try { localStorage.setItem('quillon:wallets', JSON.stringify(seed)); } catch {}
    return seed;
  }
  return [];
}

function saveWallets(ws: WalletEntry[]): void {
  try { localStorage.setItem('quillon:wallets', JSON.stringify(ws)); } catch {}
}

/**
 * Derive a fresh Quillon wallet from 32 random bytes. Matches the
 * X-Wallet-Auth derivation used everywhere else: sha3_256(seed-bytes) →
 * ed25519 private key → ed25519 public key → "qnk" + hex(pubkey).
 *
 * Returns BOTH the address and the seed bytes hex — the caller decides
 * how to persist the seed. For Quillon's "client-managed wallet" model
 * the seed lives in localStorage under a separate key keyed by address.
 */
async function generateWallet(): Promise<{ address: string; seedHex: string }> {
  const seedBytes = new Uint8Array(32);
  crypto.getRandomValues(seedBytes);
  const priv = sha3_256(seedBytes);
  const pub = await ed25519.getPublicKey(priv);
  const address = 'qnk' + bytesToHex(pub);
  const seedHex = bytesToHex(seedBytes);
  return { address, seedHex };
}

function truncate(addr: string): string {
  if (addr.length < 16) return addr;
  return `${addr.slice(0, 12)}…${addr.slice(-6)}`;
}

interface MultiWalletDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function MultiWalletDrawer({ isOpen, onClose }: MultiWalletDrawerProps) {
  const [wallets, setWallets] = useState<WalletEntry[]>([]);
  const [activeAddress, setActiveAddress] = useState<string>('');
  const [showTemplates, setShowTemplates] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setWallets(loadWallets());
      setActiveAddress(localStorage.getItem('walletAddress') ?? '');
      setShowTemplates(false);
      setCreateError(null);
    }
  }, [isOpen]);

  const onSwitch = (addr: string) => {
    if (addr === activeAddress) return;
    try {
      // v10.11.16 UI fix: BEFORE switching, snapshot the CURRENT wallet's
      // seed to its per-address slot if missing. The "new wallets show
      // original data" bug: legacy original wallet had `walletSeed`
      // (un-suffixed key) but NO `quillon:seed:<originalAddr>`. When the
      // user created wallet B then switched back to A, the lookup of
      // `quillon:seed:<A>` returned null; the canonical `walletSeed`
      // stayed as B's seed; all subsequent X-Wallet-Auth signing for "A"
      // signed with B's key → server-visible operations all went to B
      // → operator saw their original wallet's data on the new wallet
      // (and vice-versa, depending on direction). This migration creates
      // the missing per-address entry on every switch so future lookups
      // never miss again.
      const currentAddr = activeAddress || localStorage.getItem('walletAddress') || '';
      const currentSeed = localStorage.getItem('walletSeed');
      if (currentAddr && currentSeed) {
        const existingPerAddr = localStorage.getItem(`quillon:seed:${currentAddr}`);
        if (!existingPerAddr) {
          localStorage.setItem(`quillon:seed:${currentAddr}`, currentSeed);
        }
      }

      localStorage.setItem('walletAddress', addr);
      // CRITICAL: copy the per-wallet seed (quillon:seed:<addr>) to the
      // canonical `walletSeed` key the single-wallet auth flow reads.
      // Without this, X-Wallet-Auth signing silently uses the OLD seed
      // after switching → all signed calls (send, dex_swap) hit the wrong
      // wallet on the server. This was the original "+ Wallet just opens
      // standard wallet" bug — switch happened but the seed didn't follow.
      const newSeed = localStorage.getItem(`quillon:seed:${addr}`);
      if (newSeed) {
        localStorage.setItem('walletSeed', newSeed);
      } else {
        // v10.11.16 UI fix: refuse to switch when we don't hold the
        // destination wallet's seed. Pre-fix, walletSeed silently stayed
        // as the previous wallet's seed → operator sees inherited data
        // for the "switched-to" wallet. Better to refuse + tell the
        // operator they need to import.
        console.error(`[MultiWalletDrawer] No seed found for ${addr.slice(0,16)}... Cannot switch safely.`);
        alert(
          `Cannot switch to wallet ${addr.slice(0, 12)}…\n\n` +
          `Its seed isn't stored locally — most likely because this wallet existed BEFORE you created a second wallet (a v10.11.16 → v10.11.17 bug overwrote the canonical seed on wallet creation without snapshotting the old one).\n\n` +
          `To recover:\n` +
          `1. Log out of the wallet UI.\n` +
          `2. Log back in using the BIP39 mnemonic for ${addr.slice(0, 12)}….\n` +
          `3. The login will re-derive the seed and write quillon:seed:${addr.slice(0, 8)}… correctly.\n` +
          `Then switching will work.`
        );
        return; // abort — don't reload to a half-broken state
      }
    } catch {}
    // Hard reload so all components re-read the new wallet from localStorage.
    // Soft state-switch would require threading the address through dozens of
    // existing hooks; reload is honest about the scope of the change.
    window.location.reload();
  };

  const onPickTemplate = async (template: WalletTemplateId) => {
    setCreating(true);
    setCreateError(null);
    try {
      const { address, seedHex } = await generateWallet();
      const t = TEMPLATES.find(x => x.id === template)!;
      // Default name = template label + short suffix so multiple "Trading"
      // wallets distinguish at a glance.
      const suffix = address.slice(-4);
      const entry: WalletEntry = {
        address,
        name: `${t.label} (${suffix})`,
        template,
        createdAt: new Date().toISOString(),
      };
      // Store the new wallet's seed under a per-address key so X-Wallet-Auth
      // signing for THIS wallet works once it's selected.
      try { localStorage.setItem(`quillon:seed:${address}`, seedHex); } catch {}
      const next = [...wallets, entry];
      setWallets(next);
      saveWallets(next);
      setShowTemplates(false);

      // v10.11.17 UI FIX: before overwriting canonical `walletSeed` with
      // the newly created wallet's seed, snapshot the OLD wallet's seed to
      // its per-address slot. Without this, the original main wallet's seed
      // is destroyed the moment a new wallet is created — the user can never
      // switch back (onSwitch's seed guard refuses on missing per-address
      // entry; pre-guard, they'd silently sign with the wrong key).
      try {
        const oldAddr = (localStorage.getItem('walletAddress') || '').trim();
        const oldSeed = localStorage.getItem('walletSeed');
        if (oldAddr && oldSeed && oldAddr !== address) {
          const existingPerAddr = localStorage.getItem(`quillon:seed:${oldAddr}`);
          if (!existingPerAddr) {
            localStorage.setItem(`quillon:seed:${oldAddr}`, oldSeed);
          }
        }
      } catch {}

      // FIX (2026-05-21): auto-switch to the newly created wallet. Without
      // this, the user picked a template, saw no UI change, and reported
      // "+ Wallet just opens the standard original wallet" — because the
      // drawer created an entry but never made it active. Switching here
      // also copies the seed to the canonical `walletSeed` key (see
      // onSwitch) so signing works immediately.
      try {
        localStorage.setItem('walletAddress', address);
        localStorage.setItem('walletSeed', seedHex);
      } catch {}
      // Hard reload — same model as onSwitch — so every hook re-reads the
      // new wallet from localStorage. Without this, the surrounding TopBar
      // still shows the old wallet's balance/avatar.
      window.location.reload();
    } catch (e: any) {
      setCreateError(e?.message ?? 'Wallet generation failed.');
    } finally {
      setCreating(false);
    }
  };

  const onCopy = async (addr: string) => {
    try { await navigator.clipboard.writeText(addr); setCopied(addr); setTimeout(() => setCopied(null), 1200); } catch {}
  };

  const groupedByTemplate = useMemo(() => {
    const map: Record<string, WalletEntry[]> = {};
    for (const w of wallets) {
      const k = w.template === 'main' ? 'savings' : w.template;
      (map[k] ??= []).push(w);
    }
    return map;
  }, [wallets]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[250] flex items-start justify-end p-4"
          style={{ background: 'rgba(2,4,15,0.78)', backdropFilter: 'blur(8px)' }}
          onClick={onClose}
        >
          <motion.div
            initial={{ x: 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 40, opacity: 0 }}
            transition={{ type: 'spring', duration: 0.35, bounce: 0.18 }}
            className="rounded-3xl w-full max-w-md max-h-[88vh] overflow-hidden border-2 flex flex-col"
            style={{
              background: 'linear-gradient(160deg, rgba(10,11,30,0.98) 0%, rgba(18,12,40,0.98) 100%)',
              borderColor: 'rgba(168,85,247,0.35)',
              boxShadow: '0 0 60px rgba(168,85,247,0.25)',
            }}
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="px-5 py-4 border-b border-violet-500/15 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Wallet className="w-5 h-5 text-violet-400" />
                <h2 className="text-base font-bold text-white">My Wallets</h2>
                <span className="text-[10px] uppercase tracking-widest font-bold px-2 py-0.5 rounded-md bg-violet-500/15 text-violet-300 border border-violet-500/30">
                  {wallets.length}
                </span>
                {/* v10.11.17 UI: prominent green + button right in the title row.
                    User feedback: the original "+ Wallet" pill at the bottom of
                    the drawer wasn't intuitive enough — this is the obvious
                    target the moment the drawer opens. */}
                <motion.button
                  whileHover={{ scale: 1.15, boxShadow: '0 0 20px rgba(34,197,94,0.65)' }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setShowTemplates(true)}
                  title="Create new wallet"
                  className="ml-2 inline-flex items-center justify-center w-9 h-9 rounded-full text-white font-bold text-lg shadow-lg"
                  style={{
                    background: 'linear-gradient(135deg, #22c55e 0%, #15803d 100%)',
                    boxShadow: '0 0 14px rgba(34,197,94,0.5), inset 0 -2px 4px rgba(0,0,0,0.2)',
                    border: '1px solid rgba(74,222,128,0.6)',
                  }}
                >
                  +
                </motion.button>
              </div>
              <motion.button whileHover={{ rotate: 90, scale: 1.1 }} onClick={onClose} className="text-slate-400 hover:text-white">
                <X className="w-5 h-5" />
              </motion.button>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {wallets.length === 0 ? (
                <div className="text-center py-10 text-slate-400">
                  <Wallet className="w-10 h-10 mx-auto mb-3 opacity-40" />
                  <p className="text-sm">No wallets yet. Create one below.</p>
                </div>
              ) : (
                wallets.map(wlt => {
                  const isActive = wlt.address === activeAddress;
                  const t = TEMPLATES.find(x => x.id === wlt.template) ?? { ...TEMPLATES[0], label: 'Primary', emoji: '🪙' };
                  const palette = paletteFor(t.accent);
                  return (
                    <motion.div
                      key={wlt.address}
                      whileHover={{ y: -1 }}
                      className={`rounded-2xl p-3 border transition-all ${isActive ? `ring-2 ${palette.ring}` : ''}`}
                      style={{ background: palette.bg, borderColor: palette.border }}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="text-2xl">{t.emoji}</div>
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5">
                              <p className={`text-sm font-bold ${palette.text}`}>{wlt.name}</p>
                              {isActive && (
                                <span className="text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                                  active
                                </span>
                              )}
                            </div>
                            <button onClick={() => onCopy(wlt.address)} className="group inline-flex items-center gap-1 text-[10px] text-slate-400 font-mono mt-0.5 hover:text-slate-200">
                              {truncate(wlt.address)}
                              {copied === wlt.address ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3 opacity-60 group-hover:opacity-100" />}
                            </button>
                          </div>
                        </div>
                        {!isActive && (
                          <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => onSwitch(wlt.address)}
                            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-violet-500/25 hover:bg-violet-500/40 text-violet-100 text-[11px] font-bold transition-colors"
                          >
                            <ArrowRightLeft className="w-3 h-3" />
                            Switch
                          </motion.button>
                        )}
                      </div>
                    </motion.div>
                  );
                })
              )}

              {/* + Add wallet button */}
              {!showTemplates && (
                <motion.button
                  whileHover={{ y: -2, scale: 1.01 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setShowTemplates(true)}
                  className="w-full mt-3 rounded-2xl border-2 border-dashed border-violet-500/30 hover:border-violet-400/60 p-4 text-violet-300 hover:text-violet-100 transition-colors flex items-center justify-center gap-2 font-bold"
                >
                  <Plus className="w-5 h-5" />
                  Add wallet
                </motion.button>
              )}

              {/* Template picker */}
              <AnimatePresence>
                {showTemplates && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.22 }}
                    className="mt-3 overflow-hidden"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-[11px] uppercase tracking-widest font-bold text-violet-300">
                        Pick a purpose
                      </p>
                      <button onClick={() => setShowTemplates(false)} className="text-[11px] text-slate-400 hover:text-slate-200">
                        cancel
                      </button>
                    </div>
                    <div className="grid grid-cols-1 gap-2">
                      {TEMPLATES.map((t, i) => {
                        const Icon = t.icon;
                        const palette = paletteFor(t.accent);
                        return (
                          <motion.button
                            key={t.id}
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.04 }}
                            whileHover={{ y: -2 }}
                            whileTap={{ scale: 0.98 }}
                            disabled={creating}
                            onClick={() => onPickTemplate(t.id)}
                            className="text-left rounded-xl p-3 border transition-all disabled:opacity-50"
                            style={{ background: palette.bg, borderColor: palette.border }}
                          >
                            <div className="flex items-start gap-3">
                              <div className="text-2xl">{t.emoji}</div>
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center gap-1.5">
                                  <Icon className={`w-3.5 h-3.5 ${palette.text}`} />
                                  <p className={`text-sm font-bold ${palette.text}`}>{t.label}</p>
                                </div>
                                <p className="text-[11px] text-slate-300 mt-0.5">{t.oneLine}</p>
                                <p className="text-[10px] text-slate-500 mt-1 leading-snug">{t.details}</p>
                              </div>
                              <Sparkles className={`w-3.5 h-3.5 ${palette.text} flex-shrink-0`} />
                            </div>
                          </motion.button>
                        );
                      })}
                    </div>
                    {createError && (
                      <div className="mt-3 p-2 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-200 text-xs flex items-center gap-2">
                        <AlertCircle className="w-3.5 h-3.5" />
                        {createError}
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Footer */}
            <div className="px-5 py-3 border-t border-violet-500/15 bg-slate-950/60">
              <p className="text-[10px] text-slate-500 leading-relaxed">
                Wallets stored locally per-browser. The seed never leaves your device.
                Switching reloads the page so the new wallet activates everywhere.
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
