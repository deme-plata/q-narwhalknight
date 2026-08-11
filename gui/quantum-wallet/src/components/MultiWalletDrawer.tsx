// MultiWalletDrawer — switch between multiple Quillon wallets from the topbar.
//
// ─────────────────────────────────────────────────────────────────────────────
// v10.11.83 REWRITE — moved onto the REAL credential path.
//
// The previous implementation switched wallets by writing two legacy
// localStorage keys (`walletAddress`, `walletSeed`) and reloading. Those are
// not where credentials live any more, which caused the drawer to:
//
//   • keep signing as the PREVIOUS wallet (walletSession was never re-seated),
//   • create wallets that could never sign and were unrecoverable (raw entropy,
//     no BIP39 mnemonic, no password encryption),
//   • and — worst — leave `walletAddress` pointing at wallet B while the
//     encrypted blobs still held wallet A, so the next login with A's mnemonic
//     hit LoginScreen's "address mismatch" branch and DELETED A's encrypted
//     credentials. That is the "it logs you out of the main wallet" report.
//
// Three prior patches (v10.11.16, v10.11.17, the 2026-05-21 auto-switch) each
// treated a symptom at the legacy-key layer. This one moves the drawer onto
// the same path LoginScreen uses:
//
//   switch  = snapshot(current) → restore(target) → loadWallet(password)
//             → walletSession.setSession(...) → set walletAddress → reload
//   create  = BIP39 mnemonic → storeWallet(mnemonic, password) → snapshot
//             → SHOW THE MNEMONIC → user confirms backup → reload
//
// Nothing is ever deleted, so no wallet can be orphaned by a switch. See
// services/multiWallet.ts for the vault itself.
// ─────────────────────────────────────────────────────────────────────────────

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
  Lock,
  ShieldAlert,
  Loader2,
} from 'lucide-react';
import {
  snapshotCredentials,
  restoreCredentials,
  hasCredentials,
  clearWalletCaches,
  loadWalletList,
  saveWalletList,
  migrateLegacyWallet,
  type WalletEntry,
  type WalletTemplateId,
} from '../services/multiWallet';

interface Template {
  id: WalletTemplateId;
  label: string;
  emoji: string;
  icon: React.ComponentType<{ className?: string }>;
  accent: string;
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

function truncate(addr: string): string {
  if (addr.length < 16) return addr;
  return `${addr.slice(0, 12)}…${addr.slice(-6)}`;
}

/** 16 random bytes → 12-word BIP39 phrase, via walletAuth's existing helper. */
async function freshMnemonic(): Promise<string> {
  const { entropyToMnemonic } = await import('../services/walletAuth');
  const entropy = new Uint8Array(16);
  crypto.getRandomValues(entropy);
  const hex = Array.from(entropy).map(b => b.toString(16).padStart(2, '0')).join('');
  return entropyToMnemonic(hex);
}

/** Which sub-flow the drawer is showing. */
type Mode =
  | { kind: 'list' }
  | { kind: 'templates' }
  | { kind: 'create-password'; template: WalletTemplateId }
  | { kind: 'backup'; address: string; mnemonic: string; name: string }
  | { kind: 'switch-password'; address: string; name: string };

interface MultiWalletDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function MultiWalletDrawer({ isOpen, onClose }: MultiWalletDrawerProps) {
  const [wallets, setWallets] = useState<WalletEntry[]>([]);
  const [activeAddress, setActiveAddress] = useState<string>('');
  const [mode, setMode] = useState<Mode>({ kind: 'list' });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  // Password form state (shared by create + switch flows).
  const [password, setPassword] = useState('');
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [backupAcknowledged, setBackupAcknowledged] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    // Vault any pre-v10.11.83 wallet before the user can touch anything.
    // Without this, a session that already used the old drawer has global
    // credentials belonging to a wallet with no vault slot.
    migrateLegacyWallet();
    setWallets(loadWalletList());
    setActiveAddress(localStorage.getItem('walletAddress') ?? '');
    setMode({ kind: 'list' });
    setError(null);
    setPassword('');
    setPasswordConfirm('');
    setBackupAcknowledged(false);
  }, [isOpen]);

  const resetToList = () => {
    setMode({ kind: 'list' });
    setError(null);
    setPassword('');
    setPasswordConfirm('');
  };

  // ---------------------------------------------------------------------------
  // Switch
  // ---------------------------------------------------------------------------

  const beginSwitch = (entry: WalletEntry) => {
    if (entry.address === activeAddress) return;

    if (!hasCredentials(entry.address)) {
      setError(
        `No local credentials for "${entry.name}". This wallet was created by an older ` +
        `build that stored only raw entropy. Import it from its recovery phrase on the ` +
        `login screen to re-create its credentials, then switching will work.`
      );
      return;
    }

    setError(null);
    setPassword('');
    setMode({ kind: 'switch-password', address: entry.address, name: entry.name });
  };

  const confirmSwitch = async (targetAddress: string) => {
    setBusy(true);
    setError(null);

    const previousAddress = activeAddress || localStorage.getItem('walletAddress') || '';

    try {
      const { loadWallet, recoverMnemonic, walletSession } = await import('../services/walletAuth');

      // 1. Protect the outgoing wallet before touching the globals.
      if (previousAddress) snapshotCredentials(previousAddress);

      // 2. Paint the target wallet into the global credential window.
      if (!restoreCredentials(targetAddress)) {
        throw new Error('Could not load this wallet\'s stored credentials.');
      }

      // 3. Prove the password actually opens it. If this throws we must put
      //    the previous wallet back — leaving the target's credentials in the
      //    globals while walletAddress still names the previous wallet is
      //    exactly the corrupt state this rewrite exists to prevent.
      let keyPair;
      try {
        keyPair = await loadWallet(password);
      } catch {
        if (previousAddress) restoreCredentials(previousAddress);
        throw new Error('Incorrect password for this wallet.');
      }

      // 4. Paranoia check: the decrypted key must derive the address we asked
      //    for. Guards against a vault slot written under the wrong address.
      if (keyPair.address !== targetAddress) {
        if (previousAddress) restoreCredentials(previousAddress);
        throw new Error(
          `Credential mismatch — the stored keys decrypt to ${keyPair.address.slice(0, 12)}…, ` +
          `not ${targetAddress.slice(0, 12)}…. Refusing to switch.`
        );
      }

      // 5. Mnemonic is best-effort: only used to keep "never expire" sessions
      //    convenient. A wallet without an encrypted mnemonic still switches.
      let mnemonic: string | undefined;
      try {
        mnemonic = await recoverMnemonic(password);
      } catch {
        mnemonic = undefined;
      }

      // 6. Commit. Order matters: session first (so nothing can observe a new
      //    walletAddress with a stale session), then the pointer, then caches.
      walletSession.setSession(
        keyPair.privateKey,
        keyPair.address,
        mnemonic,
        keyPair.dilithium5SecretKey,
        keyPair.dilithium5PublicKey
      );
      localStorage.setItem('walletAddress', targetAddress);
      clearWalletCaches();

      // Hard reload so every hook re-reads the new wallet. A soft switch would
      // mean threading the address through dozens of existing hooks.
      window.location.reload();
    } catch (e: any) {
      setError(e?.message ?? 'Switch failed.');
      setBusy(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Create
  // ---------------------------------------------------------------------------

  const confirmCreate = async (template: WalletTemplateId) => {
    setBusy(true);
    setError(null);

    const previousAddress = activeAddress || localStorage.getItem('walletAddress') || '';

    try {
      if (password.length < 8) throw new Error('Password must be at least 8 characters.');
      if (password !== passwordConfirm) throw new Error('Passwords do not match.');

      const { storeWallet, walletSession } = await import('../services/walletAuth');

      // 1. Protect the outgoing wallet FIRST — storeWallet() overwrites every
      //    global credential key.
      if (previousAddress) snapshotCredentials(previousAddress);

      // 2. Real BIP39 mnemonic, so this wallet is recoverable from a phrase
      //    like every other wallet in the app.
      const mnemonic = await freshMnemonic();

      // 3. storeWallet writes the encrypted key, mnemonic, password hash and
      //    the PQ keypairs into the globals, and sets walletAddress.
      const keyPair = await storeWallet(mnemonic, password, true, true, true);

      // 4. Vault the new wallet immediately, so it survives the next switch.
      snapshotCredentials(keyPair.address);

      const t = TEMPLATES.find(x => x.id === template)!;
      const entry: WalletEntry = {
        address: keyPair.address,
        name: `${t.label} (${keyPair.address.slice(-4)})`,
        template,
        createdAt: new Date().toISOString(),
      };
      const next = [...wallets.filter(w => w.address !== keyPair.address), entry];
      setWallets(next);
      saveWalletList(next);

      // 5. Make it active in-session.
      walletSession.setSession(
        keyPair.privateKey,
        keyPair.address,
        mnemonic,
        keyPair.dilithium5SecretKey,
        keyPair.dilithium5PublicKey
      );
      clearWalletCaches();

      // 6. DO NOT reload yet — the mnemonic exists only in memory right now.
      //    Reloading here would leave the user with a funded-able wallet they
      //    cannot recover. Show the phrase and make them acknowledge it.
      setPassword('');
      setPasswordConfirm('');
      setBackupAcknowledged(false);
      setMode({ kind: 'backup', address: keyPair.address, mnemonic, name: entry.name });
      setBusy(false);
    } catch (e: any) {
      // Creation failed partway: put the previous wallet back in the window.
      if (previousAddress) restoreCredentials(previousAddress);
      setError(e?.message ?? 'Wallet creation failed.');
      setBusy(false);
    }
  };

  const onCopy = async (text: string, tag: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(tag);
      setTimeout(() => setCopied(null), 1400);
    } catch { /* clipboard blocked — user can select manually */ }
  };

  const activeEntry = useMemo(
    () => wallets.find(w => w.address === activeAddress),
    [wallets, activeAddress]
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const errorBox = error && (
    <div className="mt-3 p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-200 text-xs flex items-start gap-2">
      <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
      <span className="leading-relaxed">{error}</span>
    </div>
  );

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[250] flex items-start justify-end p-4"
          style={{ background: 'rgba(2,4,15,0.78)', backdropFilter: 'blur(8px)' }}
          onClick={busy ? undefined : onClose}
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
                <h2 className="text-base font-bold text-white">
                  {mode.kind === 'backup' ? 'Save your recovery phrase' : 'My Wallets'}
                </h2>
                {mode.kind === 'list' && (
                  <>
                    <span className="text-[10px] uppercase tracking-widest font-bold px-2 py-0.5 rounded-md bg-violet-500/15 text-violet-300 border border-violet-500/30">
                      {wallets.length}
                    </span>
                    <motion.button
                      whileHover={{ scale: 1.15, boxShadow: '0 0 20px rgba(34,197,94,0.65)' }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => { setError(null); setMode({ kind: 'templates' }); }}
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
                  </>
                )}
              </div>
              {mode.kind !== 'backup' && (
                <motion.button
                  whileHover={{ rotate: 90, scale: 1.1 }}
                  onClick={onClose}
                  disabled={busy}
                  className="text-slate-400 hover:text-white disabled:opacity-40"
                >
                  <X className="w-5 h-5" />
                </motion.button>
              )}
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto p-4">

              {/* ── Wallet list ───────────────────────────────────────────── */}
              {mode.kind === 'list' && (
                <div className="space-y-2">
                  {wallets.length === 0 ? (
                    <div className="text-center py-10 text-slate-400">
                      <Wallet className="w-10 h-10 mx-auto mb-3 opacity-40" />
                      <p className="text-sm">No wallets yet. Create one below.</p>
                    </div>
                  ) : (
                    wallets.map(wlt => {
                      const isActive = wlt.address === activeAddress;
                      const t = TEMPLATES.find(x => x.id === wlt.template)
                        ?? { ...TEMPLATES[0], label: 'Primary', emoji: '🪙' };
                      const palette = paletteFor(t.accent);
                      const recoverable = hasCredentials(wlt.address);
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
                                <div className="flex items-center gap-1.5 flex-wrap">
                                  <p className={`text-sm font-bold ${palette.text}`}>{wlt.name}</p>
                                  {isActive && (
                                    <span className="text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                                      active
                                    </span>
                                  )}
                                  {!recoverable && (
                                    <span
                                      title="No local credentials — import from its recovery phrase to use this wallet"
                                      className="text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 inline-flex items-center gap-1"
                                    >
                                      <ShieldAlert className="w-2.5 h-2.5" />
                                      needs import
                                    </span>
                                  )}
                                </div>
                                <button
                                  onClick={() => onCopy(wlt.address, wlt.address)}
                                  className="group inline-flex items-center gap-1 text-[10px] text-slate-400 font-mono mt-0.5 hover:text-slate-200"
                                >
                                  {truncate(wlt.address)}
                                  {copied === wlt.address
                                    ? <Check className="w-3 h-3 text-emerald-400" />
                                    : <Copy className="w-3 h-3 opacity-60 group-hover:opacity-100" />}
                                </button>
                              </div>
                            </div>
                            {!isActive && (
                              <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => beginSwitch(wlt)}
                                className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-violet-500/25 hover:bg-violet-500/40 text-violet-100 text-[11px] font-bold transition-colors flex-shrink-0"
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

                  <motion.button
                    whileHover={{ y: -2, scale: 1.01 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => { setError(null); setMode({ kind: 'templates' }); }}
                    className="w-full mt-3 rounded-2xl border-2 border-dashed border-violet-500/30 hover:border-violet-400/60 p-4 text-violet-300 hover:text-violet-100 transition-colors flex items-center justify-center gap-2 font-bold"
                  >
                    <Plus className="w-5 h-5" />
                    Add wallet
                  </motion.button>

                  {errorBox}
                </div>
              )}

              {/* ── Template picker ───────────────────────────────────────── */}
              {mode.kind === 'templates' && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-[11px] uppercase tracking-widest font-bold text-violet-300">
                      Pick a purpose
                    </p>
                    <button onClick={resetToList} className="text-[11px] text-slate-400 hover:text-slate-200">
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
                          onClick={() => {
                            setError(null);
                            setPassword('');
                            setPasswordConfirm('');
                            setMode({ kind: 'create-password', template: t.id });
                          }}
                          className="text-left rounded-xl p-3 border transition-all"
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
                  {errorBox}
                </div>
              )}

              {/* ── Password: create ──────────────────────────────────────── */}
              {mode.kind === 'create-password' && (
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Lock className="w-4 h-4 text-violet-300" />
                    <p className="text-sm font-bold text-white">Set a password</p>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed mb-4">
                    This password encrypts the new wallet's keys in this browser (AES-256-GCM,
                    PBKDF2 100k). You'll be asked for it whenever you switch to this wallet.
                    Using the same password as your other wallets is fine.
                  </p>

                  <input
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    placeholder="Password (min 8 characters)"
                    autoFocus
                    className="w-full mb-2 px-3 py-2.5 rounded-xl bg-slate-950/70 border border-violet-500/30 text-white text-sm placeholder:text-slate-600 focus:outline-none focus:border-violet-400/70"
                  />
                  <input
                    type="password"
                    value={passwordConfirm}
                    onChange={e => setPasswordConfirm(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && !busy) confirmCreate(mode.template); }}
                    placeholder="Confirm password"
                    className="w-full px-3 py-2.5 rounded-xl bg-slate-950/70 border border-violet-500/30 text-white text-sm placeholder:text-slate-600 focus:outline-none focus:border-violet-400/70"
                  />

                  {errorBox}

                  <div className="flex gap-2 mt-4">
                    <button
                      onClick={resetToList}
                      disabled={busy}
                      className="flex-1 py-2.5 rounded-xl text-slate-300 text-sm font-bold bg-slate-800/60 hover:bg-slate-700/60 disabled:opacity-40"
                    >
                      Back
                    </button>
                    <button
                      onClick={() => confirmCreate(mode.template)}
                      disabled={busy || !password || !passwordConfirm}
                      className="flex-1 py-2.5 rounded-xl text-white text-sm font-bold disabled:opacity-40 inline-flex items-center justify-center gap-2"
                      style={{ background: 'linear-gradient(135deg, #22c55e 0%, #15803d 100%)' }}
                    >
                      {busy && <Loader2 className="w-4 h-4 animate-spin" />}
                      {busy ? 'Creating…' : 'Create wallet'}
                    </button>
                  </div>
                </div>
              )}

              {/* ── Backup the mnemonic ───────────────────────────────────── */}
              {mode.kind === 'backup' && (
                <div>
                  <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/40 mb-4">
                    <div className="flex items-start gap-2">
                      <ShieldAlert className="w-4 h-4 text-amber-300 flex-shrink-0 mt-0.5" />
                      <p className="text-[11px] text-amber-100 leading-relaxed">
                        <strong>{mode.name}</strong> was created. These 12 words are the ONLY way
                        to recover it if you clear this browser or lose the password. Write them
                        down now — they will not be shown again.
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-1.5 mb-3">
                    {mode.mnemonic.split(' ').map((word, i) => (
                      <div
                        key={i}
                        className="px-2 py-1.5 rounded-lg bg-slate-950/70 border border-violet-500/25 text-center"
                      >
                        <span className="text-[9px] text-slate-600 mr-1">{i + 1}</span>
                        <span className="text-[11px] font-mono text-violet-100">{word}</span>
                      </div>
                    ))}
                  </div>

                  <button
                    onClick={() => onCopy(mode.mnemonic, 'mnemonic')}
                    className="w-full mb-3 py-2 rounded-xl bg-violet-500/20 hover:bg-violet-500/35 text-violet-100 text-xs font-bold inline-flex items-center justify-center gap-2"
                  >
                    {copied === 'mnemonic'
                      ? <><Check className="w-3.5 h-3.5 text-emerald-400" /> Copied</>
                      : <><Copy className="w-3.5 h-3.5" /> Copy recovery phrase</>}
                  </button>

                  <label className="flex items-start gap-2 mb-4 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={backupAcknowledged}
                      onChange={e => setBackupAcknowledged(e.target.checked)}
                      className="mt-0.5 accent-emerald-500"
                    />
                    <span className="text-[11px] text-slate-300 leading-relaxed">
                      I have written down these 12 words somewhere safe and offline.
                    </span>
                  </label>

                  <button
                    onClick={() => window.location.reload()}
                    disabled={!backupAcknowledged}
                    className="w-full py-2.5 rounded-xl text-white text-sm font-bold disabled:opacity-40"
                    style={{ background: 'linear-gradient(135deg, #22c55e 0%, #15803d 100%)' }}
                  >
                    Continue to wallet
                  </button>
                </div>
              )}

              {/* ── Password: switch ──────────────────────────────────────── */}
              {mode.kind === 'switch-password' && (
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Lock className="w-4 h-4 text-violet-300" />
                    <p className="text-sm font-bold text-white">Unlock "{mode.name}"</p>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed mb-1">
                    Enter the password for this wallet to decrypt its keys and make it active.
                  </p>
                  <p className="text-[10px] text-slate-600 font-mono mb-4">{truncate(mode.address)}</p>

                  <input
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && !busy && password) confirmSwitch(mode.address); }}
                    placeholder="Wallet password"
                    autoFocus
                    className="w-full px-3 py-2.5 rounded-xl bg-slate-950/70 border border-violet-500/30 text-white text-sm placeholder:text-slate-600 focus:outline-none focus:border-violet-400/70"
                  />

                  {errorBox}

                  <div className="flex gap-2 mt-4">
                    <button
                      onClick={resetToList}
                      disabled={busy}
                      className="flex-1 py-2.5 rounded-xl text-slate-300 text-sm font-bold bg-slate-800/60 hover:bg-slate-700/60 disabled:opacity-40"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => confirmSwitch(mode.address)}
                      disabled={busy || !password}
                      className="flex-1 py-2.5 rounded-xl text-white text-sm font-bold disabled:opacity-40 inline-flex items-center justify-center gap-2"
                      style={{ background: 'linear-gradient(135deg, #a855f7 0%, #6d28d9 100%)' }}
                    >
                      {busy && <Loader2 className="w-4 h-4 animate-spin" />}
                      {busy ? 'Unlocking…' : 'Switch'}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 border-t border-violet-500/15 bg-slate-950/60">
              <p className="text-[10px] text-slate-500 leading-relaxed">
                {activeEntry ? `Active: ${activeEntry.name}. ` : ''}
                Each wallet's keys are encrypted separately in this browser and never leave your
                device. Switching asks for that wallet's password and reloads the page.
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
