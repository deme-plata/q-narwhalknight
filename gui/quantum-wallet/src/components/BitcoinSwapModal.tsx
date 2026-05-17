import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, ArrowRightLeft, Clock, CheckCircle, AlertCircle, Copy,
  Loader2, Bitcoin, Send, Download, RefreshCw, ChevronRight,
  Shield, Zap, TrendingUp, Sparkles, Droplet, XCircle
} from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';
import { qnkAPI } from '../services/api';

interface BitcoinSwapModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletAddress: string;
}

type Tab = 'wallet' | 'receive' | 'send' | 'swap' | 'lp' | 'history';
type SwapDirection = 'buy_btc' | 'sell_btc';

interface LpIntent {
  intent_id: string;
  pool_id: string;
  btc_amount_sats: number;
  btc_address: string;
  qug_amount_escrowed: string;
  status: any; // { kind: 'awaiting_btc' | 'btc_detected' | 'ready_to_finalize' | 'completed' | 'cancelled' | 'expired' | 'failed', ... }
  created_at: number;
  updated_at: number;
  expires_at: number;
}

interface DepositAddress {
  address: string;
  deposit_id: string;
  expires_at?: string;
  qr_data?: string;
}

interface BridgeBalance {
  balance_sats: number;
  balance_btc: number;
  watched_addresses: string[];
}

interface SwapItem {
  swap_id: string;
  btc_amount: number;
  qnk_amount: string;
  status: string;
  created_at: string;
  hash_lock: string;
}

interface DepositItem {
  deposit_id: string;
  address: string;
  amount_sats?: number;
  status: string;
  created_at: string;
  txid?: string;
  confirmations?: number;
}

const statusBadge = (status: string, small = false) => {
  const colors: Record<string, string> = {
    proposed:    'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    pending:     'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    btc_locked:  'bg-blue-500/20 text-blue-300 border-blue-500/30',
    confirmed:   'bg-blue-500/20 text-blue-300 border-blue-500/30',
    qnk_locked:  'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
    qnk_claimed: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    btc_claimed: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
    completed:   'bg-green-500/20 text-green-300 border-green-500/30',
    credited:    'bg-green-500/20 text-green-300 border-green-500/30',
    refunded:    'bg-red-500/20 text-red-300 border-red-500/30',
    failed:      'bg-red-500/20 text-red-300 border-red-500/30',
    expired:     'bg-gray-500/20 text-gray-400 border-gray-500/30',
  };
  return (
    <span className={`px-2 py-0.5 rounded-full border font-medium ${small ? 'text-[10px]' : 'text-xs'} ${colors[status] || 'bg-gray-500/20 text-gray-400 border-gray-500/30'}`}>
      {status.replace(/_/g, ' ')}
    </span>
  );
};

const fmtBtc = (sats: number) => (sats / 1e8).toFixed(8);
const fmtSats = (sats: number) => sats.toLocaleString() + ' sats';

const BitcoinSwapModal = ({ isOpen, onClose, walletAddress }: BitcoinSwapModalProps) => {
  const [tab, setTab] = useState<Tab>('wallet');

  // Wallet state
  const [balance, setBalance] = useState<BridgeBalance | null>(null);
  const [bridgeOnline, setBridgeOnline] = useState(false);
  const [loadingBalance, setLoadingBalance] = useState(false);

  // Receive state
  const [depositAddr, setDepositAddr] = useState<DepositAddress | null>(null);
  const [creatingAddr, setCreatingAddr] = useState(false);
  const [addrError, setAddrError] = useState<string | null>(null);
  const [deposits, setDeposits] = useState<DepositItem[]>([]);

  // Send state
  const [sendTo, setSendTo] = useState('');
  const [sendAmount, setSendAmount] = useState('');
  const [sendFee, setSendFee] = useState<'economy' | 'normal' | 'fast'>('normal');
  const [sending, setSending] = useState(false);
  const [sendResult, setSendResult] = useState<{ ok: boolean; msg: string } | null>(null);

  // Swap state
  const [direction, setDirection] = useState<SwapDirection>('sell_btc');
  const [btcAmount, setBtcAmount] = useState('');
  const [qnkAmount, setQnkAmount] = useState('');
  const [btcDest, setBtcDest] = useState('');
  const [userBtcPubkey, setUserBtcPubkey] = useState('');
  const [swapping, setSwapping] = useState(false);
  const [swapResult, setSwapResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [swaps, setSwaps] = useState<SwapItem[]>([]);

  // 33-byte compressed secp256k1 pubkey = 66 hex chars starting with 02 or 03.
  const isValidCompressedPubkey = (s: string) =>
    /^0[23][0-9a-fA-F]{64}$/.test(s.trim());

  // Rates
  const [btcUsd, setBtcUsd] = useState(97000);
  const [qugUsd, setQugUsd] = useState(3000);

  // Bridge LP state
  const [lpBtcAmount, setLpBtcAmount] = useState('0.01');
  const [lpIntents, setLpIntents] = useState<LpIntent[]>([]);
  const [lpCreating, setLpCreating] = useState(false);
  const [lpResult, setLpResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [poolReserve, setPoolReserve] = useState<{ qug: number; wbtc: number; lp_supply: number } | null>(null);

  // Copy state
  const [copied, setCopied] = useState<string | null>(null);

  const copy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopied(key);
    setTimeout(() => setCopied(null), 2000);
  };

  const fetchAll = useCallback(async () => {
    try {
      setLoadingBalance(true);
      const [statusRes, balRes, swapRes, depRes, lpRes, poolRes] = await Promise.allSettled([
        qnkAPI.getBitcoinBridgeStatus(),
        qnkAPI.getBitcoinBalance(),
        qnkAPI.listSwaps(),
        qnkAPI.listDeposits?.() ?? Promise.resolve({ success: false }),
        qnkAPI.listLpIntents?.() ?? Promise.resolve({ success: false }),
        fetch('/api/v1/defi/dex/pools')
          .then(r => r.ok ? r.json() : null)
          .catch(() => null),
      ]);
      if (statusRes.status === 'fulfilled' && statusRes.value.success)
        setBridgeOnline(statusRes.value.data?.bridge_enabled ?? false);
      if (balRes.status === 'fulfilled' && balRes.value.success && balRes.value.data)
        setBalance(balRes.value.data);
      if (swapRes.status === 'fulfilled' && swapRes.value.success)
        setSwaps(swapRes.value.data?.swaps ?? []);
      if (depRes.status === 'fulfilled' && (depRes.value as any).success)
        setDeposits((depRes.value as any).data?.deposits ?? []);
      if (lpRes.status === 'fulfilled' && (lpRes.value as any).success)
        setLpIntents((lpRes.value as any).data?.intents ?? []);
      if (poolRes.status === 'fulfilled' && poolRes.value) {
        const data = (poolRes.value as any)?.data ?? poolRes.value;
        const pools: any[] = Array.isArray(data) ? data : data?.pools ?? [];
        const wbtcPool = pools.find((p: any) =>
          (p?.pool_id ?? '').toString().includes('qug-wbtc-bridge') ||
          ((p?.token0 ?? '').toString().toUpperCase() === 'QUG' &&
           (p?.token1 ?? '').toString().toUpperCase() === 'WBTC')
        );
        if (wbtcPool) {
          // Reserves are stored in 24-decimal form.
          const r0 = Number(wbtcPool.reserve0 ?? 0) / 1e24;
          const r1 = Number(wbtcPool.reserve1 ?? 0) / 1e24;
          const lp = Number(wbtcPool.lp_token_supply ?? 0) / 1e24;
          setPoolReserve({ qug: r0, wbtc: r1, lp_supply: lp });
        } else {
          setPoolReserve({ qug: 0, wbtc: 0, lp_supply: 0 });
        }
      }
    } catch {/* silent */} finally {
      setLoadingBalance(false);
    }
  }, []);

  const fetchRates = useCallback(async () => {
    try {
      const r = await fetch('/api/v1/defi/oracle/price/QUG/USD');
      const d = await r.json();
      if (d?.price > 0) setQugUsd(d.price);
    } catch {/* silent */}
    try {
      const r = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd');
      const d = await r.json();
      if (d?.bitcoin?.usd) setBtcUsd(d.bitcoin.usd);
    } catch {/* silent */}
  }, []);

  useEffect(() => {
    if (isOpen) { fetchAll(); fetchRates(); }
  }, [isOpen, fetchAll, fetchRates]);

  const BTC_QNK = qugUsd > 0 ? btcUsd / qugUsd : 32;

  // ── Receive ──────────────────────────────────────────────────
  const handleCreateAddress = async () => {
    setCreatingAddr(true);
    setAddrError(null);
    try {
      const res = await qnkAPI.createDepositAddress();
      if (res.success && res.data) {
        setDepositAddr({
          address: res.data.btc_address,
          deposit_id: res.data.deposit_id,
          expires_at: res.data.expires_in_secs ? `${res.data.expires_in_secs}s` : undefined,
        });
      } else {
        setAddrError(res.error || 'Bridge unavailable — deposit address could not be generated.');
      }
    } catch (e: any) {
      setAddrError(e.message || 'Network error — could not reach the bridge.');
    } finally {
      setCreatingAddr(false);
    }
  };

  // ── Send ─────────────────────────────────────────────────────
  const handleSend = async () => {
    if (!sendTo || !sendAmount) return;
    setSending(true);
    setSendResult(null);
    try {
      const sats = Math.round(parseFloat(sendAmount) * 1e8);
      const res = await qnkAPI.sendBitcoin?.({ to: sendTo, amount_sats: sats, fee_priority: sendFee });
      if (res?.success) {
        setSendResult({ ok: true, msg: `Sent! TXID: ${res.data?.txid?.slice(0, 16)}…` });
        setSendTo(''); setSendAmount('');
        fetchAll();
      } else {
        setSendResult({ ok: false, msg: res?.error || 'Failed to broadcast transaction.' });
      }
    } catch (e: any) {
      setSendResult({ ok: false, msg: e.message || 'Network error.' });
    } finally {
      setSending(false);
    }
  };

  // ── LP intent (one-click bridge LP) ──────────────────────────
  const lpBtcSats = Math.max(0, Math.round(parseFloat(lpBtcAmount || '0') * 1e8));
  // Suggested QUG = BTC_value_usd / qug_price_usd. The pool will use the user-submitted
  // QUG amount (not this suggestion) so they can over- or under-pair if they want.
  const suggestedQugForLp = qugUsd > 0 ? (lpBtcSats / 1e8) * (btcUsd / qugUsd) : 0;
  const handleCreateLpIntent = async () => {
    setLpCreating(true);
    setLpResult(null);
    try {
      if (lpBtcSats < 10000) {
        setLpResult({ ok: false, msg: 'Minimum LP deposit is 0.0001 BTC (10,000 sats).' });
        return;
      }
      if (suggestedQugForLp <= 0) {
        setLpResult({ ok: false, msg: 'Oracle prices not available yet — retry in a moment.' });
        return;
      }
      // qug_amount is in 24-decimal base units.
      const qugBase = BigInt(Math.round(suggestedQugForLp * 1e8)) * BigInt(1e16);
      const res = await qnkAPI.createLpIntent({
        btc_amount_sats: lpBtcSats,
        qug_amount: qugBase.toString(),
      });
      if (res.success && res.data) {
        setLpResult({
          ok: true,
          msg: `LP intent created. Send ${(lpBtcSats / 1e8).toFixed(8)} BTC to ${res.data.btc_address.slice(0, 14)}…`,
        });
        fetchAll();
      } else {
        setLpResult({ ok: false, msg: res.error || 'Failed to create LP intent.' });
      }
    } catch (e: any) {
      setLpResult({ ok: false, msg: e.message || 'Network error.' });
    } finally {
      setLpCreating(false);
    }
  };
  const handleFinalizeLp = async (id: string) => {
    const res = await qnkAPI.finalizeLpIntent(id);
    setLpResult(res.success
      ? { ok: true, msg: 'LP finalized — check your wallet for LP tokens.' }
      : { ok: false, msg: res.error || 'Finalize failed.' });
    fetchAll();
  };
  const handleCancelLp = async (id: string) => {
    const res = await qnkAPI.cancelLpIntent(id);
    setLpResult(res.success
      ? { ok: true, msg: 'LP intent cancelled, QUG refunded.' }
      : { ok: false, msg: res.error || 'Cancel failed.' });
    fetchAll();
  };
  const lpStatusLabel = (s: any): string => {
    if (!s) return 'unknown';
    if (typeof s === 'string') return s;
    const k = s.kind || s;
    if (k === 'btc_detected' && typeof s.confirmations === 'number')
      return `seen · ${s.confirmations}/6 confs`;
    if (k === 'ready_to_finalize') return 'ready to finalize';
    if (k === 'completed') return 'completed';
    if (k === 'cancelled') return 'cancelled';
    if (k === 'expired') return 'expired';
    if (k === 'failed') return `failed: ${s.reason ?? '?'}`;
    return (k || 'awaiting_btc').replace(/_/g, ' ');
  };

  // ── Swap ─────────────────────────────────────────────────────
  const handleSwap = async () => {
    setSwapping(true);
    setSwapResult(null);
    try {
      const btcSats = Math.round(parseFloat(btcAmount) * 1e8);
      const qnkBase = BigInt(Math.round(parseFloat(qnkAmount) * 1e8)) * BigInt(1e16);
      if (btcSats <= 0) { setSwapResult({ ok: false, msg: 'Enter a valid BTC amount.' }); return; }
      if (direction === 'buy_btc' && !btcDest) {
        setSwapResult({ ok: false, msg: 'Enter a Bitcoin destination address.' });
        return;
      }
      if (!isValidCompressedPubkey(userBtcPubkey)) {
        setSwapResult({
          ok: false,
          msg: 'Paste a valid 33-byte compressed BTC public key (66 hex chars, prefix 02/03). This key must be one you control — the HTLC refund path requires its private key.',
        });
        return;
      }
      const res = await qnkAPI.createAtomicSwap({
        direction,
        btc_amount: btcSats,
        qnk_amount: qnkBase.toString(),
        user_btc_pubkey: userBtcPubkey.trim(),
        btc_destination: btcDest || undefined,
      });
      if (res.success && res.data) {
        setSwapResult({ ok: true, msg: `Swap created — ID: ${res.data.swap_id.slice(0, 12)}…` });
        setBtcAmount(''); setQnkAmount(''); setBtcDest('');
        fetchAll();
      } else {
        setSwapResult({ ok: false, msg: res.error || 'Swap failed.' });
      }
    } catch (e: any) {
      setSwapResult({ ok: false, msg: e.message || 'Network error.' });
    } finally {
      setSwapping(false);
    }
  };

  if (!isOpen) return null;

  const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'wallet',  label: 'Wallet',  icon: <Bitcoin size={13} /> },
    { id: 'receive', label: 'Receive', icon: <Download size={13} /> },
    { id: 'send',    label: 'Send',    icon: <Send size={13} /> },
    { id: 'swap',    label: 'Swap',    icon: <ArrowRightLeft size={13} /> },
    { id: 'lp',      label: 'Bridge LP', icon: <Droplet size={13} /> },
    { id: 'history', label: `History (${swaps.length + deposits.length})`, icon: <Clock size={13} /> },
  ];

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 overflow-y-auto"
        onClick={onClose}
      >
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm pointer-events-none" />

        <div className="flex min-h-full items-center justify-center p-4">
        <motion.div
          initial={{ scale: 0.92, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.92, opacity: 0 }}
          transition={{ type: 'spring', damping: 26, stiffness: 300 }}
          onClick={(e) => e.stopPropagation()}
          className="relative w-full max-w-lg rounded-2xl overflow-hidden flex flex-col"
          style={{
            background: 'linear-gradient(145deg, rgba(12,10,20,0.99), rgba(22,14,8,0.98))',
            border: '1px solid rgba(251,146,60,0.25)',
            boxShadow: '0 0 80px rgba(251,146,60,0.12), 0 30px 60px rgba(0,0,0,0.7)',
            maxHeight: '90vh',
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-orange-500/15 flex-shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl flex items-center justify-center text-lg font-bold text-white"
                style={{ background: 'linear-gradient(135deg, #f97316, #dc2626)' }}>₿</div>
              <div>
                <div className="text-white font-bold">Bitcoin Wallet</div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <div className={`w-1.5 h-1.5 rounded-full ${bridgeOnline ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                  <span className="text-[10px] text-gray-500">{bridgeOnline ? 'Bridge online · Knots v28.1' : 'Bridge offline'}</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={fetchAll} className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-gray-300 transition-colors">
                <RefreshCw size={14} className={loadingBalance ? 'animate-spin' : ''} />
              </button>
              <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-0.5 px-4 pt-3 pb-0 flex-shrink-0 overflow-x-auto">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                  tab === t.id
                    ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30'
                    : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                }`}
              >
                {t.icon}{t.label}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="overflow-y-auto flex-1 p-4">
            <AnimatePresence mode="wait">

              {/* ── WALLET TAB ── */}
              {tab === 'wallet' && (
                <motion.div key="wallet" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-4">
                  {/* Balance card */}
                  <div className="rounded-xl p-5 text-center"
                    style={{ background: 'linear-gradient(135deg, rgba(249,115,22,0.12), rgba(234,88,12,0.08))', border: '1px solid rgba(249,115,22,0.2)' }}>
                    <div className="text-xs text-orange-300/60 mb-1 uppercase tracking-widest">Bridge Balance</div>
                    {balance ? (
                      <>
                        <div className="text-3xl font-bold text-white font-mono">{fmtBtc(balance.balance_sats)}</div>
                        <div className="text-sm text-orange-300/70 mt-0.5">{fmtSats(balance.balance_sats)}</div>
                        <div className="text-xs text-gray-500 mt-1">≈ ${(balance.balance_btc * btcUsd).toLocaleString(undefined, { maximumFractionDigits: 2 })} USD</div>
                      </>
                    ) : (
                      <div className="text-2xl font-bold text-gray-600 animate-pulse">— BTC</div>
                    )}
                  </div>

                  {/* Quick actions */}
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { label: 'Receive', icon: <Download size={18} />, tab: 'receive' as Tab, color: 'text-green-400', bg: 'rgba(34,197,94,0.1)', border: 'rgba(34,197,94,0.2)' },
                      { label: 'Send',    icon: <Send size={18} />,     tab: 'send'    as Tab, color: 'text-blue-400',  bg: 'rgba(59,130,246,0.1)',  border: 'rgba(59,130,246,0.2)' },
                      { label: 'Swap',   icon: <ArrowRightLeft size={18} />, tab: 'swap' as Tab, color: 'text-orange-400', bg: 'rgba(249,115,22,0.1)', border: 'rgba(249,115,22,0.2)' },
                    ].map(a => (
                      <button key={a.label} onClick={() => setTab(a.tab)}
                        className="flex flex-col items-center gap-2 rounded-xl py-4 transition-all hover:scale-105"
                        style={{ background: a.bg, border: `1px solid ${a.border}` }}>
                        <span className={a.color}>{a.icon}</span>
                        <span className="text-xs text-gray-300 font-medium">{a.label}</span>
                      </button>
                    ))}
                  </div>

                  {/* Network stats */}
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: 'BTC Price', value: `$${btcUsd.toLocaleString()}`, icon: <TrendingUp size={12} className="text-orange-400" /> },
                      { label: 'QUG Price', value: `$${qugUsd.toLocaleString()}`, icon: <Zap size={12} className="text-amber-400" /> },
                      { label: 'Protocol', value: 'HTLC Atomic Swap', icon: <Shield size={12} className="text-green-400" /> },
                      { label: 'Network', value: 'Bitcoin Mainnet', icon: <Bitcoin size={12} className="text-orange-400" /> },
                    ].map(s => (
                      <div key={s.label} className="rounded-lg px-3 py-2.5 flex items-center gap-2"
                        style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                        {s.icon}
                        <div>
                          <div className="text-[9px] text-gray-600 uppercase tracking-wider">{s.label}</div>
                          <div className="text-xs text-gray-300 font-mono font-medium">{s.value}</div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Watched addresses */}
                  {balance && balance.watched_addresses.length > 0 && (
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase tracking-widest mb-2">Watched Addresses</div>
                      <div className="space-y-1">
                        {balance.watched_addresses.slice(0, 3).map(addr => (
                          <div key={addr} className="flex items-center justify-between rounded-lg px-3 py-2"
                            style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.06)' }}>
                            <span className="text-xs font-mono text-gray-400 truncate flex-1">{addr}</span>
                            <button onClick={() => copy(addr, addr)} className="ml-2 text-gray-600 hover:text-gray-300 flex-shrink-0">
                              {copied === addr ? <CheckCircle size={12} className="text-green-400" /> : <Copy size={12} />}
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              )}

              {/* ── RECEIVE TAB ── */}
              {tab === 'receive' && (
                <motion.div key="receive" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-4">
                  <div className="text-xs text-gray-400 leading-relaxed p-3 rounded-lg"
                    style={{ background: 'rgba(34,197,94,0.06)', border: '1px solid rgba(34,197,94,0.15)' }}>
                    <CheckCircle size={13} className="inline text-green-400 mr-1.5 -mt-0.5" />
                    Generate a Bitcoin deposit address. Funds sent here are automatically detected and credited to your QNK wallet after 6+ confirmations (~60 min).
                  </div>

                  {!depositAddr ? (
                    <div className="space-y-2">
                      <button
                        onClick={handleCreateAddress}
                        disabled={creatingAddr || !bridgeOnline}
                        className="w-full py-3 rounded-xl font-semibold text-white flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        style={{ background: 'linear-gradient(135deg, rgba(34,197,94,0.8), rgba(22,163,74,0.8))' }}
                      >
                        {creatingAddr ? <><Loader2 size={15} className="animate-spin" />Generating…</> : <><Download size={15} />Generate Deposit Address</>}
                      </button>
                      {!bridgeOnline && !addrError && (
                        <p className="text-center text-xs text-red-400/80">Bridge is offline — deposit address generation is currently unavailable.</p>
                      )}
                      {addrError && (
                        <p className="text-center text-xs text-red-400/80">{addrError}</p>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {/* Address display + QR */}
                      <div className="rounded-xl p-4 text-center"
                        style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(34,197,94,0.2)' }}>
                        <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-3">Your Bitcoin Deposit Address</div>
                        {/* QR Code */}
                        <div className="flex justify-center mb-3">
                          <div className="rounded-xl p-2.5" style={{ background: 'white' }}>
                            <QRCodeSVG
                              value={`bitcoin:${depositAddr.address}`}
                              size={140}
                              level="M"
                              includeMargin={false}
                              fgColor="#000000"
                              bgColor="#ffffff"
                            />
                          </div>
                        </div>
                        <div className="font-mono text-xs text-green-300 break-all mb-3 px-2 select-all leading-relaxed">{depositAddr.address}</div>
                        <button
                          onClick={() => copy(depositAddr.address, 'btcaddr')}
                          className="flex items-center gap-2 mx-auto px-4 py-2 rounded-lg text-xs font-medium transition-colors"
                          style={{ background: 'rgba(34,197,94,0.15)', border: '1px solid rgba(34,197,94,0.3)', color: '#86efac' }}
                        >
                          {copied === 'btcaddr' ? <><CheckCircle size={13} />Copied!</> : <><Copy size={13} />Copy Address</>}
                        </button>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="rounded-lg p-2.5" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}>
                          <div className="text-gray-600 mb-0.5">Deposit ID</div>
                          <div className="text-gray-400 font-mono text-[10px]">{depositAddr.deposit_id.slice(0, 16)}…</div>
                        </div>
                        <div className="rounded-lg p-2.5" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}>
                          <div className="text-gray-600 mb-0.5">Min Confirmations</div>
                          <div className="text-gray-400">6 blocks (~60 min)</div>
                        </div>
                      </div>

                      <button
                        onClick={handleCreateAddress}
                        className="w-full py-2 rounded-xl text-xs text-gray-500 hover:text-gray-300 transition-colors flex items-center justify-center gap-1.5"
                        style={{ border: '1px solid rgba(255,255,255,0.08)' }}
                      >
                        <RefreshCw size={12} />New address
                      </button>
                    </div>
                  )}

                  {/* Recent deposits */}
                  {deposits.length > 0 && (
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase tracking-widest mb-2">Recent Deposits</div>
                      {deposits.slice(0, 4).map(d => (
                        <div key={d.deposit_id} className="flex items-center justify-between rounded-lg p-3 mb-1.5"
                          style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.06)' }}>
                          <div>
                            <div className="text-xs text-gray-300 font-mono">{d.address.slice(0, 10)}…{d.address.slice(-6)}</div>
                            {d.amount_sats && <div className="text-[10px] text-gray-500">{fmtBtc(d.amount_sats)} BTC</div>}
                          </div>
                          <div className="text-right">
                            {statusBadge(d.status, true)}
                            {d.confirmations != null && (
                              <div className="text-[9px] text-gray-600 mt-0.5">{d.confirmations} confs</div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </motion.div>
              )}

              {/* ── SEND TAB ── */}
              {tab === 'send' && (
                <motion.div key="send" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-4">
                  <div className="text-xs text-orange-300/70 p-3 rounded-lg"
                    style={{ background: 'rgba(251,146,60,0.07)', border: '1px solid rgba(251,146,60,0.2)' }}>
                    <AlertCircle size={13} className="inline text-orange-400 mr-1.5 -mt-0.5" />
                    Sends BTC from the bridge balance. Only funds received via deposit addresses are available to send.
                  </div>

                  {/* Balance display */}
                  <div className="rounded-xl p-3 flex items-center justify-between"
                    style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <span className="text-xs text-gray-500">Available</span>
                    <span className="text-sm font-mono text-white">{balance ? fmtBtc(balance.balance_sats) : '—'} BTC</span>
                  </div>

                  {/* Recipient */}
                  <div className="rounded-xl p-3 space-y-1" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">Recipient Address</div>
                    <input
                      type="text" value={sendTo} onChange={e => setSendTo(e.target.value)}
                      placeholder="bc1q… or 3… or 1…"
                      className="w-full bg-transparent text-sm font-mono text-white outline-none placeholder-gray-700"
                    />
                  </div>

                  {/* Amount */}
                  <div className="rounded-xl p-3 space-y-1" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div className="flex justify-between text-[10px] text-gray-500 uppercase tracking-wider">
                      <span>Amount (BTC)</span>
                      {sendAmount && <span className="text-gray-400">≈ ${(parseFloat(sendAmount) * btcUsd).toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>}
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="number" value={sendAmount} onChange={e => setSendAmount(e.target.value)}
                        placeholder="0.00000000" step="0.00000001"
                        className="flex-1 bg-transparent text-xl font-mono text-white outline-none placeholder-gray-700"
                      />
                      {balance && (
                        <button onClick={() => setSendAmount(fmtBtc(balance.balance_sats))}
                          className="text-[10px] px-2 py-1 rounded text-orange-400 border border-orange-500/30 hover:bg-orange-500/10">
                          MAX
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Fee priority */}
                  <div>
                    <div className="text-[10px] text-gray-600 uppercase tracking-widest mb-2">Fee Priority</div>
                    <div className="grid grid-cols-3 gap-2">
                      {([
                        { id: 'economy', label: 'Economy', est: '~60 min', sats: '~5 sat/vB' },
                        { id: 'normal',  label: 'Normal',  est: '~30 min', sats: '~15 sat/vB' },
                        { id: 'fast',    label: 'Fast',    est: '~10 min', sats: '~40 sat/vB' },
                      ] as const).map(f => (
                        <button key={f.id} onClick={() => setSendFee(f.id)}
                          className={`rounded-lg p-2.5 text-center text-xs transition-all ${sendFee === f.id ? 'bg-orange-500/20 border border-orange-500/40 text-orange-300' : 'text-gray-500 hover:text-gray-300'}`}
                          style={sendFee !== f.id ? { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' } : {}}>
                          <div className="font-medium">{f.label}</div>
                          <div className="text-[9px] opacity-70 mt-0.5">{f.sats}</div>
                          <div className="text-[9px] opacity-50">{f.est}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {sendResult && (
                    <div className={`flex items-center gap-2 p-3 rounded-lg text-sm ${sendResult.ok ? 'bg-green-500/10 border border-green-500/20 text-green-300' : 'bg-red-500/10 border border-red-500/20 text-red-300'}`}>
                      {sendResult.ok ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
                      {sendResult.msg}
                    </div>
                  )}

                  <button
                    onClick={handleSend}
                    disabled={sending || !sendTo || !sendAmount || parseFloat(sendAmount) <= 0}
                    className="w-full py-3 rounded-xl font-semibold text-white transition-all disabled:opacity-40 flex items-center justify-center gap-2"
                    style={{ background: sending ? 'rgba(59,130,246,0.3)' : 'linear-gradient(135deg, rgba(59,130,246,0.9), rgba(37,99,235,0.9))' }}
                  >
                    {sending ? <><Loader2 size={15} className="animate-spin" />Broadcasting…</> : <><Send size={15} />Send Bitcoin</>}
                  </button>
                </motion.div>
              )}

              {/* ── SWAP TAB ── */}
              {tab === 'swap' && (
                <motion.div key="swap" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-4">
                  {/* Direction */}
                  <div className="flex p-1 rounded-xl" style={{ background: 'rgba(255,255,255,0.04)' }}>
                    {([['sell_btc', 'BTC → QNK'], ['buy_btc', 'QNK → BTC']] as [SwapDirection, string][]).map(([d, label]) => (
                      <button key={d} onClick={() => setDirection(d)}
                        className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${direction === d ? 'bg-orange-500/30 text-orange-200' : 'text-gray-400 hover:text-gray-200'}`}>
                        {label}
                      </button>
                    ))}
                  </div>

                  {/* From */}
                  <div className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                      <span>You send</span><span>{direction === 'sell_btc' ? 'BTC' : 'QNK'}</span>
                    </div>
                    <input type="number" value={direction === 'sell_btc' ? btcAmount : qnkAmount}
                      onChange={e => {
                        const v = e.target.value;
                        if (direction === 'sell_btc') { setBtcAmount(v); setQnkAmount(v ? (parseFloat(v) * BTC_QNK).toFixed(4) : ''); }
                        else { setQnkAmount(v); setBtcAmount(v ? (parseFloat(v) / BTC_QNK).toFixed(8) : ''); }
                      }}
                      placeholder="0.00" className="w-full bg-transparent text-xl font-mono text-white outline-none" />
                    {direction === 'sell_btc' && btcAmount && (
                      <div className="text-[10px] text-gray-600 mt-1">≈ ${(parseFloat(btcAmount) * btcUsd).toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
                    )}
                  </div>

                  <div className="flex justify-center">
                    <div className="p-2 rounded-full" style={{ background: 'rgba(249,115,22,0.15)', border: '1px solid rgba(249,115,22,0.3)' }}>
                      <ArrowRightLeft size={14} className="text-orange-400" />
                    </div>
                  </div>

                  {/* To */}
                  <div className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                      <span>You receive</span><span>{direction === 'sell_btc' ? 'QNK' : 'BTC'}</span>
                    </div>
                    <input type="number" value={direction === 'sell_btc' ? qnkAmount : btcAmount}
                      onChange={e => {
                        const v = e.target.value;
                        if (direction === 'sell_btc') { setQnkAmount(v); setBtcAmount(v ? (parseFloat(v) / BTC_QNK).toFixed(8) : ''); }
                        else { setBtcAmount(v); setQnkAmount(v ? (parseFloat(v) * BTC_QNK).toFixed(4) : ''); }
                      }}
                      placeholder="0.00" className="w-full bg-transparent text-xl font-mono text-white outline-none" />
                  </div>

                  {direction === 'buy_btc' && (
                    <div className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                      <div className="text-[10px] text-gray-500 mb-1">BTC Destination</div>
                      <input type="text" value={btcDest} onChange={e => setBtcDest(e.target.value)}
                        placeholder="bc1q… or 3…" className="w-full bg-transparent text-sm font-mono text-white outline-none" />
                    </div>
                  )}

                  {/* User-controlled BTC pubkey — required for HTLC refund/claim safety.
                      The user MUST own the corresponding secp256k1 private key, otherwise
                      they can never recover funds locked in the HTLC if the swap times out. */}
                  <div className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                      <span>Your Bitcoin Public Key (compressed, 66 hex)</span>
                      {userBtcPubkey && !isValidCompressedPubkey(userBtcPubkey) && (
                        <span className="text-red-400">invalid format</span>
                      )}
                    </div>
                    <input
                      type="text"
                      value={userBtcPubkey}
                      onChange={e => setUserBtcPubkey(e.target.value)}
                      placeholder="02xxxxxxxx… (33-byte compressed pubkey)"
                      className="w-full bg-transparent text-xs font-mono text-white outline-none placeholder-gray-700"
                    />
                    <div className="text-[10px] text-amber-400/70 mt-1 leading-snug">
                      ⚠ You must control the matching private key. The HTLC refund path
                      requires it; the bridge cannot recover funds for you on timeout.
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[10px] text-gray-600 px-1">
                    <span>Rate: 1 BTC ≈ {BTC_QNK.toFixed(2)} QNK</span>
                    <span className="flex items-center gap-1"><Clock size={11} />~60 min · 6 BTC confirmations</span>
                  </div>

                  {swapResult && (
                    <div className={`flex items-center gap-2 p-3 rounded-lg text-sm ${swapResult.ok ? 'bg-green-500/10 border border-green-500/20 text-green-300' : 'bg-red-500/10 border border-red-500/20 text-red-300'}`}>
                      {swapResult.ok ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
                      {swapResult.msg}
                    </div>
                  )}

                  <button onClick={handleSwap} disabled={swapping || !btcAmount || parseFloat(btcAmount) <= 0 || !isValidCompressedPubkey(userBtcPubkey) || (direction === 'buy_btc' && !btcDest)}
                    className="w-full py-3 rounded-xl font-semibold text-white transition-all disabled:opacity-40 flex items-center justify-center gap-2"
                    style={{ background: swapping ? 'rgba(251,146,60,0.3)' : 'linear-gradient(135deg, #f97316, #ea580c)' }}>
                    {swapping ? <><Loader2 size={15} className="animate-spin" />Creating swap…</> : <>Initiate {direction === 'sell_btc' ? 'BTC → QNK' : 'QNK → BTC'} Swap</>}
                  </button>

                  <div className="text-[10px] text-gray-600 text-center">HTLC · Trustless · Non-custodial · Auto-refund on timeout</div>
                </motion.div>
              )}

              {/* ── BRIDGE LP TAB ── */}
              {tab === 'lp' && (
                <motion.div key="lp" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-4">
                  {/* Pool status header */}
                  <div className="rounded-xl p-4"
                    style={{ background: 'linear-gradient(135deg, rgba(34,197,94,0.10), rgba(20,184,166,0.06))', border: '1px solid rgba(34,197,94,0.25)' }}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-1.5">
                        <Droplet size={14} className="text-emerald-300" />
                        <span className="text-xs font-semibold text-emerald-200 uppercase tracking-wider">QUG / wBTC Pool</span>
                      </div>
                      {poolReserve && poolReserve.lp_supply === 0 && (
                        <span className="text-[10px] text-amber-300 bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 rounded-full">
                          Empty — awaiting first LP
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-3 mt-2">
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase">wBTC reserve</div>
                        <div className="text-base font-mono text-white">
                          {poolReserve ? poolReserve.wbtc.toFixed(8) : '…'} <span className="text-xs text-gray-500">wBTC</span>
                        </div>
                        <div className="text-[10px] text-gray-600">
                          ≈ ${poolReserve ? (poolReserve.wbtc * btcUsd).toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase">QUG reserve</div>
                        <div className="text-base font-mono text-white">
                          {poolReserve ? poolReserve.qug.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '…'} <span className="text-xs text-gray-500">QUG</span>
                        </div>
                        <div className="text-[10px] text-gray-600">
                          ≈ ${poolReserve ? (poolReserve.qug * qugUsd).toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
                        </div>
                      </div>
                    </div>
                    <div className="text-[10px] text-emerald-300/70 mt-3 leading-snug">
                      <Shield size={11} className="inline mr-1 -mt-0.5" />
                      Liquidity here is 100% backed: every wBTC token in this pool corresponds to real BTC held by the bridge wallet on Delta.
                    </div>
                  </div>

                  {/* One-click LP wizard */}
                  <div className="rounded-xl p-4"
                    style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}>
                    <div className="flex items-center gap-1.5 mb-2">
                      <Sparkles size={13} className="text-amber-300" />
                      <span className="text-xs font-semibold text-amber-200 uppercase tracking-wider">Become an LP — one click</span>
                    </div>
                    <p className="text-[11px] text-gray-400 leading-snug mb-3">
                      Pick how much BTC you'd like to deposit. We escrow the matching QUG immediately,
                      give you a Bitcoin deposit address, and once 6 confirmations land we auto-pair
                      both sides into the pool and mint LP tokens to your wallet. You earn 0.3% of every
                      trade against this pool.
                    </p>

                    {/* BTC amount input */}
                    <div className="rounded-lg p-3 mb-2" style={{ background: 'rgba(0,0,0,0.25)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <div className="flex justify-between text-[10px] text-gray-500 mb-1 uppercase tracking-wider">
                        <span>You deposit (BTC)</span>
                        <span>≈ ${(lpBtcSats / 1e8 * btcUsd).toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                      </div>
                      <input
                        type="number"
                        step="0.0001"
                        min="0.0001"
                        max="1"
                        value={lpBtcAmount}
                        onChange={e => setLpBtcAmount(e.target.value)}
                        placeholder="0.01"
                        className="w-full bg-transparent text-xl font-mono text-white outline-none placeholder-gray-700"
                      />
                    </div>

                    {/* Auto-matched QUG */}
                    <div className="rounded-lg p-3 mb-3" style={{ background: 'rgba(34,197,94,0.05)', border: '1px solid rgba(34,197,94,0.15)' }}>
                      <div className="flex justify-between text-[10px] text-emerald-300/70 mb-0.5 uppercase tracking-wider">
                        <span>We pair with</span>
                        <span>1 BTC = {qugUsd > 0 ? (btcUsd / qugUsd).toFixed(2) : '—'} QUG</span>
                      </div>
                      <div className="text-base font-mono text-emerald-200">
                        {suggestedQugForLp.toLocaleString(undefined, { maximumFractionDigits: 4 })} <span className="text-xs text-emerald-400/60">QUG</span>
                      </div>
                      <div className="text-[10px] text-emerald-400/50">
                        ≈ ${(suggestedQugForLp * qugUsd).toLocaleString(undefined, { maximumFractionDigits: 2 })} · escrowed when you commit
                      </div>
                    </div>

                    {lpResult && (
                      <div className={`flex items-start gap-2 p-3 rounded-lg text-xs mb-3 ${lpResult.ok ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-200' : 'bg-red-500/10 border border-red-500/20 text-red-300'}`}>
                        {lpResult.ok ? <CheckCircle size={13} className="flex-shrink-0 mt-0.5" /> : <AlertCircle size={13} className="flex-shrink-0 mt-0.5" />}
                        <span className="leading-snug">{lpResult.msg}</span>
                      </div>
                    )}

                    <button
                      onClick={handleCreateLpIntent}
                      disabled={lpCreating || lpBtcSats < 10000 || !bridgeOnline}
                      className="w-full py-3 rounded-xl font-semibold text-white transition-all disabled:opacity-40 flex items-center justify-center gap-2"
                      style={{ background: lpCreating ? 'rgba(34,197,94,0.3)' : 'linear-gradient(135deg, #10b981, #059669)' }}
                    >
                      {lpCreating
                        ? <><Loader2 size={15} className="animate-spin" />Locking QUG + generating address…</>
                        : <><Sparkles size={15} />Lock QUG & Get BTC Address</>}
                    </button>
                    {!bridgeOnline && (
                      <p className="text-center text-[10px] text-red-400/80 mt-2">
                        Bridge offline — LP intents are temporarily unavailable.
                      </p>
                    )}
                  </div>

                  {/* Active intents */}
                  {lpIntents.length > 0 && (
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase tracking-widest mb-2">Your LP intents</div>
                      {lpIntents.map(it => {
                        const kind = (it.status?.kind ?? it.status ?? 'awaiting_btc') as string;
                        const isAwaiting = kind === 'awaiting_btc' || kind === 'btc_detected';
                        const isReady = kind === 'ready_to_finalize';
                        const isDone = kind === 'completed';
                        const isClosed = kind === 'cancelled' || kind === 'expired' || kind === 'failed';
                        return (
                          <div key={it.intent_id} className="rounded-xl p-3 mb-1.5"
                            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}>
                            <div className="flex items-center justify-between mb-1.5">
                              <span className="text-xs font-mono text-gray-300">{it.intent_id.slice(0, 12)}…</span>
                              <span className={`text-[10px] px-2 py-0.5 rounded-full border ${
                                isDone ? 'bg-green-500/15 text-green-300 border-green-500/30' :
                                isReady ? 'bg-blue-500/15 text-blue-300 border-blue-500/30' :
                                isAwaiting ? 'bg-yellow-500/15 text-yellow-300 border-yellow-500/30' :
                                'bg-gray-500/15 text-gray-400 border-gray-500/30'
                              }`}>{lpStatusLabel(it.status)}</span>
                            </div>
                            <div className="flex items-center justify-between text-[11px] text-gray-400 font-mono">
                              <span>{(it.btc_amount_sats / 1e8).toFixed(8)} BTC</span>
                              <ChevronRight size={11} className="text-gray-700" />
                              <span>{(Number(it.qug_amount_escrowed || '0') / 1e24).toFixed(2)} QUG</span>
                            </div>
                            {isAwaiting && (
                              <div className="flex items-center justify-between mt-2 text-[10px]">
                                <button onClick={() => copy(it.btc_address, 'lp-' + it.intent_id)} className="text-gray-500 hover:text-gray-200 flex items-center gap-1">
                                  <Copy size={11} />{it.btc_address.slice(0, 12)}…{it.btc_address.slice(-6)}
                                </button>
                                <button onClick={() => handleCancelLp(it.intent_id)} className="text-red-400 hover:text-red-200 flex items-center gap-1">
                                  <XCircle size={11} />Cancel & refund
                                </button>
                              </div>
                            )}
                            {isReady && (
                              <button onClick={() => handleFinalizeLp(it.intent_id)}
                                className="w-full mt-2 py-2 rounded-lg text-xs font-medium text-white"
                                style={{ background: 'linear-gradient(135deg, #3b82f6, #2563eb)' }}>
                                Claim LP tokens
                              </button>
                            )}
                            {isClosed && (
                              <div className="text-[10px] text-gray-600 mt-1">closed at {new Date(it.updated_at * 1000).toLocaleString()}</div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}

                  <div className="text-[10px] text-gray-600 text-center leading-snug">
                    Honest liquidity: every wBTC in this pool is backed 1:1 by BTC in the bridge wallet.<br />
                    Withdraw wBTC → BTC any time from the Send tab.
                  </div>
                </motion.div>
              )}

              {/* ── HISTORY TAB ── */}
              {tab === 'history' && (
                <motion.div key="history" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-3">
                  {/* Deposits section */}
                  {deposits.length > 0 && (
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase tracking-widest mb-2">Deposits</div>
                      {deposits.map(d => (
                        <div key={d.deposit_id} className="flex items-center justify-between rounded-xl p-3 mb-1.5"
                          style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}>
                          <div>
                            <div className="text-xs font-mono text-gray-300">{d.address.slice(0, 12)}…</div>
                            {d.amount_sats && <div className="text-[10px] text-gray-500 mt-0.5">{fmtBtc(d.amount_sats)} BTC</div>}
                            {d.txid && (
                              <div className="text-[9px] text-gray-600 mt-0.5 flex items-center gap-1">
                                txid: {d.txid.slice(0, 10)}…
                                <button onClick={() => copy(d.txid!, 'txid-' + d.deposit_id)} className="text-gray-600 hover:text-gray-400">
                                  {copied === 'txid-' + d.deposit_id ? <CheckCircle size={9} className="text-green-400" /> : <Copy size={9} />}
                                </button>
                              </div>
                            )}
                          </div>
                          <div className="text-right">
                            {statusBadge(d.status, true)}
                            {d.confirmations != null && <div className="text-[9px] text-gray-600 mt-1">{d.confirmations} confs</div>}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Swaps section */}
                  <div>
                    <div className="text-[10px] text-gray-600 uppercase tracking-widest mb-2">Atomic Swaps</div>
                    {swaps.length === 0 ? (
                      <div className="text-center py-8 text-gray-600">
                        <ArrowRightLeft size={28} className="mx-auto mb-2 opacity-20" />
                        <p className="text-sm">No swaps yet</p>
                      </div>
                    ) : swaps.map(s => (
                      <div key={s.swap_id} className="rounded-xl p-3 mb-1.5 hover:border-orange-500/20 transition-colors"
                        style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}>
                        <div className="flex items-center justify-between mb-1.5">
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs font-mono text-gray-400">{s.swap_id.slice(0, 10)}…</span>
                            <button onClick={() => copy(s.swap_id, s.swap_id)} className="text-gray-600 hover:text-gray-400">
                              {copied === s.swap_id ? <CheckCircle size={11} className="text-green-400" /> : <Copy size={11} />}
                            </button>
                          </div>
                          {statusBadge(s.status, true)}
                        </div>
                        <div className="flex items-center justify-between text-[10px] text-gray-500">
                          <span>{fmtBtc(s.btc_amount)} BTC</span>
                          <ChevronRight size={10} className="text-gray-700" />
                          <span>{(parseFloat(s.qnk_amount) / 1e24).toFixed(4)} QNK</span>
                          <span className="text-gray-700">{new Date(s.created_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}

            </AnimatePresence>
          </div>
        </motion.div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default BitcoinSwapModal;
