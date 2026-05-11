import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Copy, CheckCircle, RefreshCw, Send, ArrowLeftRight, Clock, Shield, Zap } from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';
import { qnkAPI } from '../services/api';

interface ZcashWalletModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletAddress: string;
}

interface SwapEntry {
  swap_id: string;
  direction: string;
  zec_amount: number;
  qnk_amount: string;
  status: string;
  z_address?: string;
  created_at: string;
}

type Tab = 'balance' | 'send' | 'swap' | 'history';

const ZcashWalletModal: React.FC<ZcashWalletModalProps> = ({ isOpen, onClose, walletAddress }) => {
  const [activeTab, setActiveTab] = useState<Tab>('balance');
  const [copied, setCopied] = useState(false);
  const [zAddress, setZAddress] = useState('');
  const [balanceZec, setBalanceZec] = useState(0);
  const [balanceZat, setBalanceZat] = useState(0);
  const [bridgeStatus, setBridgeStatus] = useState<any>(null);
  const [swapHistory, setSwapHistory] = useState<SwapEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const [sendAddress, setSendAddress] = useState('');
  const [sendAmount, setSendAmount] = useState('');
  const [sendMemo, setSendMemo] = useState('');

  const [swapDirection, setSwapDirection] = useState<'buy_zec' | 'sell_zec'>('buy_zec');
  const [swapZecAmount, setSwapZecAmount] = useState('');
  const [swapQnkAmount, setSwapQnkAmount] = useState('');
  const [receiveZAddress, setReceiveZAddress] = useState('');

  const [zecUsdRate, setZecUsdRate] = useState(25);
  const [qugUsdRate, setQugUsdRate] = useState(3000);

  useEffect(() => {
    const fetchRates = async () => {
      try {
        const res = await fetch('/api/v1/defi/oracle/price/QUG/USD');
        const data = await res.json();
        if (data?.price && data.price > 0) setQugUsdRate(data.price);
      } catch { /* silent */ }
      try {
        const res = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=zcash&vs_currencies=usd');
        const data = await res.json();
        if (data?.zcash?.usd) setZecUsdRate(data.zcash.usd);
      } catch { /* silent */ }
    };
    if (isOpen) fetchRates();
  }, [isOpen]);

  const ZEC_QNK_RATE = qugUsdRate > 0 ? zecUsdRate / qugUsdRate : 0.008;

  const handleSwapAmountChange = (value: string, field: 'zec' | 'qnk') => {
    const numVal = parseFloat(value) || 0;
    if (field === 'zec') {
      setSwapZecAmount(value);
      setSwapQnkAmount(numVal > 0 ? (numVal * ZEC_QNK_RATE).toFixed(4) : '');
    } else {
      setSwapQnkAmount(value);
      setSwapZecAmount(numVal > 0 && ZEC_QNK_RATE > 0 ? (numVal / ZEC_QNK_RATE).toFixed(8) : '');
    }
  };

  const fetchData = useCallback(async () => {
    try {
      const [balRes, addrRes, bridgeRes, swapsRes] = await Promise.allSettled([
        qnkAPI.getZcashBalance(),
        qnkAPI.getZcashAddress(),
        qnkAPI.getZcashBridgeStatus(),
        qnkAPI.listZcashSwaps(),
      ]);
      if (balRes.status === 'fulfilled' && balRes.value?.data) {
        setBalanceZec(balRes.value.data.balance_zec || 0);
        setBalanceZat(balRes.value.data.balance_zat || 0);
      }
      if (addrRes.status === 'fulfilled' && addrRes.value?.data) {
        setZAddress(addrRes.value.data.z_address || '');
      }
      if (bridgeRes.status === 'fulfilled' && bridgeRes.value?.data) {
        setBridgeStatus(bridgeRes.value.data);
      }
      if (swapsRes.status === 'fulfilled' && swapsRes.value?.data) {
        setSwapHistory(swapsRes.value.data.swaps || []);
      }
    } catch (e) {
      console.error('Failed to fetch Zcash data:', e);
    }
  }, []);

  useEffect(() => {
    if (isOpen) fetchData();
  }, [isOpen, fetchData]);

  const handleSend = async () => {
    const isShielded = sendAddress.startsWith('zs') || sendAddress.startsWith('u1');
    if (!isShielded) { setError('Only shielded addresses accepted (zs1... or u1...).'); return; }
    const amountZat = Math.round(parseFloat(sendAmount) * 100_000_000);
    if (isNaN(amountZat) || amountZat <= 0) { setError('Invalid amount.'); return; }
    setLoading(true); setError(''); setSuccess('');
    try {
      const res = await qnkAPI.sendShieldedZec({ to_z_address: sendAddress, amount_zat: amountZat, memo: sendMemo || undefined });
      if (res.success && res.data) {
        setSuccess(`Transaction submitted: ${res.data.tx_id}`);
        setSendAddress(''); setSendAmount(''); setSendMemo('');
        fetchData();
      } else { setError(res.error || 'Send failed'); }
    } catch (e: any) { setError(e.message || 'Send failed'); }
    finally { setLoading(false); }
  };

  const handleSwap = async () => {
    const zecAmountZat = Math.round(parseFloat(swapZecAmount) * 100_000_000);
    if (isNaN(zecAmountZat) || zecAmountZat <= 0) { setError('Invalid ZEC amount.'); return; }
    setLoading(true); setError(''); setSuccess('');
    try {
      const res = await qnkAPI.createZcashSwap({ direction: swapDirection, zec_amount: zecAmountZat, qnk_amount: swapQnkAmount || '0', z_address: receiveZAddress || zAddress || undefined });
      if (res.success && res.data) {
        setSuccess(`Swap created: ${res.data.swap_id}`);
        setSwapZecAmount(''); setSwapQnkAmount('');
        fetchData();
      } else { setError(res.error || 'Swap failed'); }
    } catch (e: any) { setError(e.message || 'Swap failed'); }
    finally { setLoading(false); }
  };

  const copyAddress = (addr: string) => {
    navigator.clipboard.writeText(addr);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleMaxSend = () => {
    const max = Math.max(0, balanceZec - 0.0001);
    setSendAmount(max > 0 ? max.toFixed(8) : '');
  };

  const formatZec = (zat: number) => (zat / 100_000_000).toFixed(8);
  const truncateAddr = (addr: string) => addr.length > 24 ? `${addr.slice(0, 12)}...${addr.slice(-12)}` : addr;

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'balance', label: 'Wallet', icon: <Shield size={13} /> },
    { id: 'send', label: 'Send', icon: <Send size={13} /> },
    { id: 'swap', label: 'Swap', icon: <ArrowLeftRight size={13} /> },
    { id: 'history', label: 'History', icon: <Clock size={13} /> },
  ];

  const inputClass = "w-full rounded-xl px-4 py-3 text-white placeholder-gray-500 text-sm focus:outline-none transition-all";
  const inputStyle = { background: 'rgba(147,51,234,0.08)', border: '1px solid rgba(147,51,234,0.25)' };
  const inputFocusStyle = { outline: 'none', border: '1px solid rgba(167,139,250,0.6)' };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
          style={{ background: 'rgba(0,0,0,0.88)', backdropFilter: 'blur(14px)' }}
          onClick={e => e.target === e.currentTarget && onClose()}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.93, y: 24 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.93, y: 24 }}
            transition={{ type: 'spring', stiffness: 300, damping: 28 }}
            className="relative w-full max-w-lg rounded-3xl overflow-hidden"
            style={{
              background: 'linear-gradient(160deg, #0d050f 0%, #130620 50%, #0a0414 100%)',
              border: '1.5px solid rgba(147,51,234,0.45)',
              boxShadow: '0 0 80px rgba(147,51,234,0.18), 0 30px 80px rgba(0,0,0,0.8)',
              maxHeight: '90vh',
            }}
          >
            {/* Header */}
            <div className="relative px-5 pt-5 pb-4" style={{ borderBottom: '1px solid rgba(147,51,234,0.2)' }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {/* ZEC Logo */}
                  <div className="w-11 h-11 rounded-2xl flex items-center justify-center font-black text-lg"
                    style={{ background: 'linear-gradient(135deg, #7c3aed, #4f46e5)', boxShadow: '0 0 20px rgba(124,58,237,0.5)' }}>
                    <span style={{ color: '#f4b728', textShadow: '0 0 10px rgba(244,183,40,0.6)' }}>Ⓩ</span>
                  </div>
                  <div>
                    <h2 className="text-base font-bold text-white">Zcash Wallet</h2>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <Shield size={10} className="text-violet-400" />
                      <span className="text-[10px] text-violet-400">Shielded · Private</span>
                      {bridgeStatus && (
                        <span className="ml-1 px-1.5 py-0.5 rounded-full text-[9px] font-semibold"
                          style={{ background: bridgeStatus.zebra_syncing ? 'rgba(251,191,36,0.15)' : 'rgba(52,211,153,0.15)', color: bridgeStatus.zebra_syncing ? '#fbbf24' : '#34d399' }}>
                          {bridgeStatus.zebra_syncing ? `Syncing` : `Synced · ${(bridgeStatus.zebra_height / 1_000_000).toFixed(2)}M`}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={fetchData} className="p-2 rounded-xl text-gray-400 hover:text-violet-300 transition-colors"
                    style={{ background: 'rgba(147,51,234,0.1)' }}>
                    <RefreshCw size={14} />
                  </button>
                  <button onClick={onClose} className="p-2 rounded-xl text-gray-400 hover:text-white transition-colors"
                    style={{ background: 'rgba(255,255,255,0.06)' }}>
                    <X size={15} />
                  </button>
                </div>
              </div>

              {/* Tabs */}
              <div className="flex gap-1 mt-4 rounded-xl p-1" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' }}>
                {tabs.map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => { setActiveTab(tab.id); setError(''); setSuccess(''); }}
                    className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium transition-all"
                    style={{
                      background: activeTab === tab.id ? 'linear-gradient(135deg, rgba(124,58,237,0.5), rgba(79,70,229,0.5))' : 'transparent',
                      color: activeTab === tab.id ? '#c4b5fd' : '#6b7280',
                      border: activeTab === tab.id ? '1px solid rgba(167,139,250,0.3)' : '1px solid transparent',
                    }}
                  >
                    {tab.icon}{tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Alerts */}
            <AnimatePresence>
              {(error || success) && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="px-5 pt-3">
                  {error && <div className="p-3 rounded-xl text-xs text-red-300" style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)' }}>{error}</div>}
                  {success && <div className="p-3 rounded-xl text-xs text-emerald-300" style={{ background: 'rgba(52,211,153,0.1)', border: '1px solid rgba(52,211,153,0.25)' }}>{success}</div>}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Content */}
            <div className="p-5 overflow-y-auto" style={{ maxHeight: 'calc(90vh - 180px)' }}>

              {/* ── Balance Tab ── */}
              {activeTab === 'balance' && (
                <motion.div key="balance" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                  {/* Balance Card */}
                  <div className="rounded-2xl p-5 text-center"
                    style={{ background: 'linear-gradient(135deg, rgba(124,58,237,0.25) 0%, rgba(79,70,229,0.2) 100%)', border: '1px solid rgba(167,139,250,0.3)', boxShadow: '0 0 40px rgba(124,58,237,0.12)' }}>
                    <p className="text-xs text-violet-400/70 mb-2 uppercase tracking-widest">Shielded Balance</p>
                    <p className="text-4xl font-black text-white mb-1">
                      {balanceZec.toFixed(8)}
                    </p>
                    <p className="text-lg font-bold" style={{ color: '#f4b728', textShadow: '0 0 15px rgba(244,183,40,0.4)' }}>ZEC</p>
                    <p className="text-xs text-gray-500 mt-1">{balanceZat.toLocaleString()} zatoshis · ≈ ${(balanceZec * zecUsdRate).toFixed(2)} USD</p>
                  </div>

                  {/* Z-Address + QR */}
                  <div className="rounded-2xl p-4 space-y-3" style={{ background: 'rgba(147,51,234,0.07)', border: '1px solid rgba(147,51,234,0.2)' }}>
                    <p className="text-xs text-violet-400/80 font-medium uppercase tracking-wider">Your Shielded Address</p>
                    {zAddress ? (
                      <>
                        <div className="flex justify-center">
                          <div className="rounded-xl p-2.5" style={{ background: 'white' }}>
                            <QRCodeSVG value={`zcash:${zAddress}`} size={130} level="M" includeMargin={false} fgColor="#000000" bgColor="#ffffff" />
                          </div>
                        </div>
                        <div className="flex items-center gap-2 rounded-xl p-3" style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(147,51,234,0.15)' }}>
                          <code className="text-violet-300 text-[10px] flex-1 break-all font-mono leading-relaxed select-all">{zAddress}</code>
                          <button onClick={() => copyAddress(zAddress)}
                            className="flex-shrink-0 p-2 rounded-lg transition-all"
                            style={{ background: copied ? 'rgba(124,58,237,0.3)' : 'rgba(147,51,234,0.15)', color: copied ? '#c4b5fd' : '#9ca3af' }}>
                            {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
                          </button>
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-4 text-gray-500 text-sm">Loading address…</div>
                    )}
                  </div>

                  {/* Info grid */}
                  <div className="grid grid-cols-2 gap-2.5">
                    {[
                      { label: 'Network', value: 'Zcash Mainnet', color: '#c4b5fd' },
                      { label: 'Address Type', value: 'Sapling', color: '#f4b728' },
                      { label: 'Zebra Height', value: bridgeStatus?.zebra_height?.toLocaleString() ?? '…', color: '#34d399' },
                      { label: 'Privacy', value: 'Maximum', color: '#34d399' },
                    ].map(item => (
                      <div key={item.label} className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}>
                        <p className="text-gray-500 text-[10px] mb-0.5">{item.label}</p>
                        <p className="text-sm font-semibold" style={{ color: item.color }}>{item.value}</p>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* ── Send Tab ── */}
              {activeTab === 'send' && (
                <motion.div key="send" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                  <div>
                    <label className="text-xs text-violet-300/80 font-medium mb-1.5 block">Destination z-address</label>
                    <input
                      type="text"
                      value={sendAddress}
                      onChange={e => setSendAddress(e.target.value)}
                      placeholder="zs1... or u1..."
                      className={inputClass}
                      style={inputStyle}
                      onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
                      onBlur={e => Object.assign(e.target.style, inputStyle)}
                    />
                    <p className="text-[10px] text-gray-600 mt-1">Sapling (zs1...) or Unified (u1...) addresses only</p>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-xs text-violet-300/80 font-medium">Amount (ZEC)</label>
                      <button onClick={handleMaxSend} className="text-[10px] font-bold px-2 py-0.5 rounded-md transition-all"
                        style={{ background: 'rgba(124,58,237,0.2)', color: '#a78bfa' }}>
                        MAX
                      </button>
                    </div>
                    <input
                      type="number" step="0.00000001"
                      value={sendAmount}
                      onChange={e => setSendAmount(e.target.value)}
                      placeholder="0.00000000"
                      className={inputClass}
                      style={inputStyle}
                      onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
                      onBlur={e => Object.assign(e.target.style, inputStyle)}
                    />
                    <div className="flex justify-between mt-1 text-[10px] text-gray-600">
                      <span>Available: {balanceZec.toFixed(8)} ZEC</span>
                      <span>Fee: 0.0001 ZEC</span>
                    </div>
                  </div>

                  <div>
                    <label className="text-xs text-violet-300/80 font-medium mb-1.5 block">Encrypted Memo <span className="text-gray-600">(optional)</span></label>
                    <textarea
                      value={sendMemo}
                      onChange={e => setSendMemo(e.target.value)}
                      placeholder="Private message encrypted on-chain…"
                      maxLength={512}
                      rows={2}
                      className={`${inputClass} resize-none`}
                      style={inputStyle}
                    />
                  </div>

                  <button
                    onClick={handleSend}
                    disabled={loading || !sendAddress || !sendAmount}
                    className="w-full py-3.5 rounded-xl font-semibold text-sm transition-all"
                    style={{
                      background: loading || !sendAddress || !sendAmount ? 'rgba(124,58,237,0.2)' : 'linear-gradient(135deg, #7c3aed, #4f46e5)',
                      color: loading || !sendAddress || !sendAmount ? '#6b7280' : 'white',
                      boxShadow: loading || !sendAddress || !sendAmount ? 'none' : '0 0 20px rgba(124,58,237,0.4)',
                    }}
                  >
                    {loading ? 'Sending…' : 'Send Shielded ZEC'}
                  </button>

                  <div className="flex items-start gap-2 p-3 rounded-xl" style={{ background: 'rgba(124,58,237,0.06)', border: '1px solid rgba(124,58,237,0.15)' }}>
                    <Shield size={12} className="text-violet-500 mt-0.5 flex-shrink-0" />
                    <p className="text-[10px] text-violet-400/60 leading-relaxed">All transactions use Sapling shielded pools. Amounts, sender, and recipient are fully encrypted on-chain.</p>
                  </div>
                </motion.div>
              )}

              {/* ── Swap Tab ── */}
              {activeTab === 'swap' && (
                <motion.div key="swap" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                  {/* Direction toggle */}
                  <div className="flex gap-1.5 p-1 rounded-xl" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' }}>
                    {(['buy_zec', 'sell_zec'] as const).map(dir => (
                      <button key={dir} onClick={() => setSwapDirection(dir)}
                        className="flex-1 py-2.5 rounded-lg text-xs font-semibold transition-all"
                        style={{
                          background: swapDirection === dir ? 'linear-gradient(135deg, rgba(124,58,237,0.5), rgba(79,70,229,0.5))' : 'transparent',
                          color: swapDirection === dir ? '#c4b5fd' : '#6b7280',
                          border: swapDirection === dir ? '1px solid rgba(167,139,250,0.3)' : '1px solid transparent',
                        }}>
                        {dir === 'buy_zec' ? 'QNK → ZEC' : 'ZEC → QNK'}
                      </button>
                    ))}
                  </div>

                  {/* Rate */}
                  <div className="flex items-center justify-between px-3 py-2.5 rounded-xl" style={{ background: 'rgba(244,183,40,0.07)', border: '1px solid rgba(244,183,40,0.2)' }}>
                    <span className="text-[10px] text-gray-500">Rate</span>
                    <span className="text-xs font-semibold" style={{ color: '#f4b728' }}>
                      1 ZEC ≈ {ZEC_QNK_RATE.toFixed(6)} QNK
                      <span className="text-gray-600 font-normal ml-2">${zecUsdRate.toFixed(2)}</span>
                    </span>
                  </div>

                  <div>
                    <label className="text-xs text-violet-300/80 font-medium mb-1.5 block">ZEC Amount</label>
                    <input type="number" step="0.00000001" value={swapZecAmount}
                      onChange={e => handleSwapAmountChange(e.target.value, 'zec')}
                      placeholder="0.00000000"
                      className={inputClass} style={inputStyle}
                      onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
                      onBlur={e => Object.assign(e.target.style, inputStyle)} />
                    {swapZecAmount && parseFloat(swapZecAmount) > 0 && (
                      <p className="text-[10px] text-gray-600 mt-1">≈ ${(parseFloat(swapZecAmount) * zecUsdRate).toFixed(2)} USD</p>
                    )}
                  </div>

                  <div className="flex items-center justify-center">
                    <div className="w-8 h-8 rounded-full flex items-center justify-center"
                      style={{ background: 'rgba(124,58,237,0.2)', border: '1px solid rgba(167,139,250,0.3)' }}>
                      <ArrowLeftRight size={14} className="text-violet-400" />
                    </div>
                  </div>

                  <div>
                    <label className="text-xs text-violet-300/80 font-medium mb-1.5 block">QNK Amount</label>
                    <input type="number" step="0.0001" value={swapQnkAmount}
                      onChange={e => handleSwapAmountChange(e.target.value, 'qnk')}
                      placeholder="0.0000"
                      className={inputClass} style={inputStyle}
                      onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
                      onBlur={e => Object.assign(e.target.style, inputStyle)} />
                    {swapQnkAmount && parseFloat(swapQnkAmount) > 0 && (
                      <p className="text-[10px] text-gray-600 mt-1">≈ ${(parseFloat(swapQnkAmount) * qugUsdRate).toFixed(2)} USD</p>
                    )}
                  </div>

                  {swapDirection === 'buy_zec' && (
                    <div>
                      <label className="text-xs text-violet-300/80 font-medium mb-1.5 block">Receive ZEC at z-address</label>
                      <input type="text" value={receiveZAddress || zAddress}
                        onChange={e => setReceiveZAddress(e.target.value)}
                        placeholder="zs1..."
                        className={`${inputClass} font-mono`} style={inputStyle}
                        onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
                        onBlur={e => Object.assign(e.target.style, inputStyle)} />
                    </div>
                  )}

                  <button onClick={handleSwap} disabled={loading || !swapZecAmount}
                    className="w-full py-3.5 rounded-xl font-semibold text-sm transition-all"
                    style={{
                      background: loading || !swapZecAmount ? 'rgba(124,58,237,0.2)' : 'linear-gradient(135deg, #7c3aed, #4f46e5)',
                      color: loading || !swapZecAmount ? '#6b7280' : 'white',
                      boxShadow: loading || !swapZecAmount ? 'none' : '0 0 20px rgba(124,58,237,0.4)',
                    }}>
                    {loading ? 'Creating Swap…' : `Create ${swapDirection === 'buy_zec' ? 'QNK → ZEC' : 'ZEC → QNK'} Swap`}
                  </button>

                  <div className="p-3 rounded-xl space-y-1.5" style={{ background: 'rgba(124,58,237,0.06)', border: '1px solid rgba(124,58,237,0.15)' }}>
                    <div className="flex items-center gap-1.5 mb-1">
                      <Zap size={11} className="text-violet-400" />
                      <p className="text-[10px] text-violet-300/80 font-semibold">Shielded Atomic Swap</p>
                    </div>
                    <ol className="text-[10px] text-violet-400/50 space-y-0.5 list-decimal list-inside leading-relaxed">
                      <li>Hash-lock generated cryptographically</li>
                      <li>QNK locked in on-chain HTLC escrow</li>
                      <li>ZEC sent to your z-address (shielded)</li>
                      <li>Secret reveal completes both sides</li>
                    </ol>
                  </div>
                </motion.div>
              )}

              {/* ── History Tab ── */}
              {activeTab === 'history' && (
                <motion.div key="history" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-2.5">
                  {swapHistory.length === 0 ? (
                    <div className="text-center py-12">
                      <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-3"
                        style={{ background: 'rgba(124,58,237,0.1)', border: '1px solid rgba(124,58,237,0.2)' }}>
                        <Clock size={24} className="text-violet-500/50" />
                      </div>
                      <p className="text-sm font-medium text-gray-400">No swaps yet</p>
                      <p className="text-xs text-gray-600 mt-1">Your shielded atomic swaps will appear here</p>
                    </div>
                  ) : swapHistory.map(swap => (
                    <div key={swap.swap_id} className="rounded-xl p-4" style={{ background: 'rgba(147,51,234,0.07)', border: '1px solid rgba(147,51,234,0.18)' }}>
                      <div className="flex items-center justify-between mb-2.5">
                        <span className="text-sm font-semibold" style={{ color: swap.direction === 'buy_zec' ? '#a78bfa' : '#34d399' }}>
                          {swap.direction === 'buy_zec' ? 'QNK → ZEC' : 'ZEC → QNK'}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold"
                            style={{
                              background: swap.status === 'completed' ? 'rgba(52,211,153,0.15)' : swap.status === 'proposed' ? 'rgba(96,165,250,0.15)' : swap.status === 'refunded' ? 'rgba(251,191,36,0.15)' : 'rgba(239,68,68,0.15)',
                              color: swap.status === 'completed' ? '#34d399' : swap.status === 'proposed' ? '#60a5fa' : swap.status === 'refunded' ? '#fbbf24' : '#f87171',
                            }}>
                            {swap.status}
                          </span>
                          <span className="text-[10px] text-gray-600">{new Date(swap.created_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div><span className="text-gray-600">ZEC: </span><span className="text-white font-mono">{formatZec(swap.zec_amount)}</span></div>
                        <div><span className="text-gray-600">QNK: </span><span className="text-white">{parseFloat(swap.qnk_amount || '0').toLocaleString()}</span></div>
                      </div>
                      <p className="text-[10px] text-gray-700 font-mono mt-1.5">{truncateAddr(swap.swap_id)}</p>
                    </div>
                  ))}
                </motion.div>
              )}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 flex items-center justify-center gap-2" style={{ borderTop: '1px solid rgba(147,51,234,0.15)' }}>
              <div className="h-px flex-1" style={{ background: 'rgba(147,51,234,0.15)' }} />
              <span className="text-[10px] text-gray-700">Powered by Zebra · Delta node · {bridgeStatus?.zebra_height?.toLocaleString() ?? '…'} blocks</span>
              <div className="h-px flex-1" style={{ background: 'rgba(147,51,234,0.15)' }} />
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default ZcashWalletModal;
