import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, ArrowRightLeft, Clock, CheckCircle, AlertCircle, Copy,
  Loader2, Bitcoin, Send, Download, RefreshCw, ChevronRight,
  Shield, Zap, TrendingUp
} from 'lucide-react';
import { qnkAPI } from '../services/api';

interface BitcoinSwapModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletAddress: string;
}

type Tab = 'wallet' | 'receive' | 'send' | 'swap' | 'history';
type SwapDirection = 'buy_btc' | 'sell_btc';

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
  const [swapping, setSwapping] = useState(false);
  const [swapResult, setSwapResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [swaps, setSwaps] = useState<SwapItem[]>([]);

  // Rates
  const [btcUsd, setBtcUsd] = useState(97000);
  const [qugUsd, setQugUsd] = useState(3000);

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
      const [statusRes, balRes, swapRes, depRes] = await Promise.allSettled([
        qnkAPI.getBitcoinBridgeStatus(),
        qnkAPI.getBitcoinBalance(),
        qnkAPI.listSwaps(),
        qnkAPI.listDeposits?.() ?? Promise.resolve({ success: false }),
      ]);
      if (statusRes.status === 'fulfilled' && statusRes.value.success)
        setBridgeOnline(statusRes.value.data?.bridge_enabled ?? false);
      if (balRes.status === 'fulfilled' && balRes.value.success && balRes.value.data)
        setBalance(balRes.value.data);
      if (swapRes.status === 'fulfilled' && swapRes.value.success)
        setSwaps(swapRes.value.data?.swaps ?? []);
      if (depRes.status === 'fulfilled' && (depRes.value as any).success)
        setDeposits((depRes.value as any).data?.deposits ?? []);
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

  // ── Swap ─────────────────────────────────────────────────────
  const handleSwap = async () => {
    setSwapping(true);
    setSwapResult(null);
    try {
      const btcSats = Math.round(parseFloat(btcAmount) * 1e8);
      const qnkBase = BigInt(Math.round(parseFloat(qnkAmount) * 1e8)) * BigInt(1e16);
      if (btcSats <= 0) { setSwapResult({ ok: false, msg: 'Enter a valid BTC amount.' }); return; }
      if (direction === 'buy_btc' && !btcDest) { setSwapResult({ ok: false, msg: 'Enter a Bitcoin destination address.' }); return; }
      const userBtcPubkey = '02' + walletAddress.replace('qnk', '').slice(0, 64);
      const res = await qnkAPI.createAtomicSwap({ direction, btc_amount: btcSats, qnk_amount: qnkBase.toString(), user_btc_pubkey: userBtcPubkey, btc_destination: btcDest || undefined });
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
                    Generate a Bitcoin deposit address. Funds sent here are automatically detected and credited to your QNK wallet after 3+ confirmations.
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
                      {/* Address display */}
                      <div className="rounded-xl p-4 text-center"
                        style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(34,197,94,0.2)' }}>
                        <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">Your Bitcoin Deposit Address</div>
                        <div className="font-mono text-sm text-green-300 break-all mb-3">{depositAddr.address}</div>
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
                          <div className="text-gray-400">3 blocks (~30 min)</div>
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

                  <div className="flex items-center justify-between text-[10px] text-gray-600 px-1">
                    <span>Rate: 1 BTC ≈ {BTC_QNK.toFixed(2)} QNK</span>
                    <span className="flex items-center gap-1"><Clock size={11} />~10 min · 1 BTC confirmation</span>
                  </div>

                  {swapResult && (
                    <div className={`flex items-center gap-2 p-3 rounded-lg text-sm ${swapResult.ok ? 'bg-green-500/10 border border-green-500/20 text-green-300' : 'bg-red-500/10 border border-red-500/20 text-red-300'}`}>
                      {swapResult.ok ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
                      {swapResult.msg}
                    </div>
                  )}

                  <button onClick={handleSwap} disabled={swapping || !btcAmount || parseFloat(btcAmount) <= 0}
                    className="w-full py-3 rounded-xl font-semibold text-white transition-all disabled:opacity-40 flex items-center justify-center gap-2"
                    style={{ background: swapping ? 'rgba(251,146,60,0.3)' : 'linear-gradient(135deg, #f97316, #ea580c)' }}>
                    {swapping ? <><Loader2 size={15} className="animate-spin" />Creating swap…</> : <>Initiate {direction === 'sell_btc' ? 'BTC → QNK' : 'QNK → BTC'} Swap</>}
                  </button>

                  <div className="text-[10px] text-gray-600 text-center">HTLC · Trustless · Non-custodial · Auto-refund on timeout</div>
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
