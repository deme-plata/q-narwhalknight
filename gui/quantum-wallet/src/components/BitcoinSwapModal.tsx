import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ArrowRightLeft, Clock, CheckCircle, AlertCircle, Copy, ExternalLink, Loader2 } from 'lucide-react';
import { qnkAPI } from '../services/api';

interface BitcoinSwapModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletAddress: string;
}

type SwapTab = 'swap' | 'history';
type SwapDirection = 'buy_btc' | 'sell_btc';

interface SwapHistoryItem {
  swap_id: string;
  btc_amount: number;
  qnk_amount: string;
  status: string;
  created_at: string;
  hash_lock: string;
}

const BitcoinSwapModal = ({ isOpen, onClose, walletAddress }: BitcoinSwapModalProps) => {
  const [activeTab, setActiveTab] = useState<SwapTab>('swap');
  const [direction, setDirection] = useState<SwapDirection>('sell_btc');
  const [btcAmount, setBtcAmount] = useState('');
  const [qnkAmount, setQnkAmount] = useState('');
  const [btcDestination, setBtcDestination] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [swapHistory, setSwapHistory] = useState<SwapHistoryItem[]>([]);
  const [bridgeStatus, setBridgeStatus] = useState<any>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Fetch bridge status on mount
  useEffect(() => {
    if (isOpen) {
      fetchBridgeStatus();
      fetchSwapHistory();
    }
  }, [isOpen]);

  const fetchBridgeStatus = async () => {
    try {
      const res = await qnkAPI.getBitcoinBridgeStatus();
      if (res.success && res.data) {
        setBridgeStatus(res.data);
      }
    } catch (e) {
      console.warn('Failed to fetch bridge status:', e);
    }
  };

  const fetchSwapHistory = async () => {
    try {
      const res = await qnkAPI.listSwaps();
      if (res.success && res.data) {
        setSwapHistory(res.data.swaps || []);
      }
    } catch (e) {
      console.warn('Failed to fetch swap history:', e);
    }
  };

  // Exchange rates — fetch dynamically
  const [btcUsdRate, setBtcUsdRate] = useState(97000);
  const [qugUsdRate, setQugUsdRate] = useState(3000);

  useEffect(() => {
    // Fetch QUG price from oracle
    const fetchRates = async () => {
      try {
        const res = await fetch('/api/v1/defi/oracle/price/QUG/USD');
        const data = await res.json();
        if (data?.price && data.price > 0) {
          setQugUsdRate(data.price);
        }
      } catch (e) {
        console.warn('Failed to fetch QUG price, using default');
      }
      // Fetch BTC price from public API
      try {
        const res = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd');
        const data = await res.json();
        if (data?.bitcoin?.usd) {
          setBtcUsdRate(data.bitcoin.usd);
        }
      } catch (e) {
        console.warn('Failed to fetch BTC price, using default');
      }
    };
    if (isOpen) fetchRates();
  }, [isOpen]);

  const BTC_QNK_RATE = qugUsdRate > 0 ? btcUsdRate / qugUsdRate : 32; // 1 BTC = ~32 QNK at $97k/$3000
  const BTC_USD_RATE = btcUsdRate;

  const handleAmountChange = (value: string, field: 'btc' | 'qnk') => {
    const numVal = parseFloat(value) || 0;
    if (field === 'btc') {
      setBtcAmount(value);
      setQnkAmount((numVal * BTC_QNK_RATE).toFixed(4));
    } else {
      setQnkAmount(value);
      setBtcAmount((numVal / BTC_QNK_RATE).toFixed(8));
    }
  };

  const handleCreateSwap = async () => {
    setError(null);
    setSuccess(null);
    setIsSubmitting(true);

    try {
      const btcSats = Math.round(parseFloat(btcAmount) * 1e8);
      const qnkBase = BigInt(Math.round(parseFloat(qnkAmount) * 1e8)) * BigInt(1e16); // 24 decimals

      if (btcSats <= 0 || isNaN(btcSats)) {
        setError('Enter a valid BTC amount.');
        setIsSubmitting(false);
        return;
      }

      if (direction === 'buy_btc' && !btcDestination) {
        setError('Enter a Bitcoin destination address.');
        setIsSubmitting(false);
        return;
      }

      // Placeholder pubkey (in production, derive from wallet)
      const userBtcPubkey = '02' + walletAddress.replace('qnk', '').slice(0, 64);

      const res = await qnkAPI.createAtomicSwap({
        direction,
        btc_amount: btcSats,
        qnk_amount: qnkBase.toString(),
        user_btc_pubkey: userBtcPubkey,
        btc_destination: btcDestination || undefined,
      });

      if (res.success && res.data) {
        setSuccess(`Swap created! ID: ${res.data.swap_id}`);
        setBtcAmount('');
        setQnkAmount('');
        setBtcDestination('');
        fetchSwapHistory();
      } else {
        setError(res.error || 'Failed to create swap.');
      }
    } catch (e: any) {
      setError(e.message || 'Network error.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const statusBadge = (status: string) => {
    const colors: Record<string, string> = {
      proposed: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
      btc_locked: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
      qnk_locked: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
      qnk_claimed: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
      btc_claimed: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
      completed: 'bg-green-500/20 text-green-300 border-green-500/30',
      refunded: 'bg-red-500/20 text-red-300 border-red-500/30',
      failed: 'bg-red-500/20 text-red-300 border-red-500/30',
    };
    return (
      <span className={`px-2 py-0.5 rounded-full text-xs border ${colors[status] || 'bg-gray-500/20 text-gray-300'}`}>
        {status.replace(/_/g, ' ')}
      </span>
    );
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center"
        onClick={onClose}
      >
        {/* Backdrop */}
        <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

        {/* Modal */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ type: 'spring', damping: 25 }}
          onClick={(e) => e.stopPropagation()}
          className="relative w-full max-w-lg mx-4 rounded-2xl overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.98), rgba(25, 15, 10, 0.95))',
            border: '1px solid rgba(251, 146, 60, 0.3)',
            boxShadow: '0 0 60px rgba(251, 146, 60, 0.15)',
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-5 border-b border-orange-500/20">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: 'linear-gradient(135deg, #f97316, #f59e0b)' }}>
                <span className="text-lg font-bold text-white">₿</span>
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">Bitcoin Bridge</h2>
                <p className="text-xs text-orange-300/60">QNK ↔ BTC Atomic Swap</p>
              </div>
            </div>
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white">
              <X size={18} />
            </button>
          </div>

          {/* v9.4.0: Bridge Safety Warning Banner */}
          <div className="mx-5 mt-3 p-3 rounded-lg flex items-start gap-2"
            style={{ background: 'rgba(251, 146, 60, 0.1)', border: '1px solid rgba(251, 146, 60, 0.3)' }}>
            <AlertCircle size={16} className="text-orange-400 mt-0.5 flex-shrink-0" />
            <div className="text-xs text-orange-300/80">
              <span className="font-semibold text-orange-300">Bridge requires deposit proof.</span>{' '}
              You must provide your BTC deposit transaction ID when claiming.
              Deposits require 3+ confirmations. Max swap: 0.1 BTC.
            </div>
          </div>

          {/* Bridge Status */}
          <div className="px-5 py-2 flex items-center gap-2 text-xs">
            <div className={`w-2 h-2 rounded-full ${bridgeStatus?.bridge_enabled ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
            <span className="text-gray-400">
              {bridgeStatus?.bridge_enabled ? 'Bridge Connected' : 'Bridge Offline'}
            </span>
            <span className="text-gray-600 ml-auto">HTLC Protocol</span>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 px-5 pt-2">
            {(['swap', 'history'] as SwapTab[]).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab
                    ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                }`}
              >
                {tab === 'swap' ? 'Swap' : `History (${swapHistory.length})`}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="p-5">
            {activeTab === 'swap' && (
              <div className="space-y-4">
                {/* Direction Toggle */}
                <div className="flex items-center gap-2 p-1 rounded-xl bg-white/5">
                  <button
                    onClick={() => setDirection('sell_btc')}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                      direction === 'sell_btc'
                        ? 'bg-orange-500/30 text-orange-200'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    BTC → QNK
                  </button>
                  <button
                    onClick={() => setDirection('buy_btc')}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                      direction === 'buy_btc'
                        ? 'bg-orange-500/30 text-orange-200'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    QNK → BTC
                  </button>
                </div>

                {/* From Amount */}
                <div className="rounded-xl p-3 bg-white/5 border border-white/10">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>You send</span>
                    <span>{direction === 'sell_btc' ? 'BTC' : 'QNK'}</span>
                  </div>
                  <input
                    type="number"
                    value={direction === 'sell_btc' ? btcAmount : qnkAmount}
                    onChange={(e) => handleAmountChange(e.target.value, direction === 'sell_btc' ? 'btc' : 'qnk')}
                    placeholder="0.00"
                    className="w-full bg-transparent text-xl font-mono text-white outline-none"
                  />
                  {direction === 'sell_btc' && btcAmount && (
                    <div className="text-xs text-gray-500 mt-1">
                      ≈ ${(parseFloat(btcAmount) * BTC_USD_RATE).toLocaleString(undefined, { maximumFractionDigits: 2 })} USD
                    </div>
                  )}
                </div>

                {/* Swap Arrow */}
                <div className="flex justify-center">
                  <div className="p-2 rounded-full bg-orange-500/20 border border-orange-500/30">
                    <ArrowRightLeft size={16} className="text-orange-400" />
                  </div>
                </div>

                {/* To Amount */}
                <div className="rounded-xl p-3 bg-white/5 border border-white/10">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>You receive</span>
                    <span>{direction === 'sell_btc' ? 'QNK' : 'BTC'}</span>
                  </div>
                  <input
                    type="number"
                    value={direction === 'sell_btc' ? qnkAmount : btcAmount}
                    onChange={(e) => handleAmountChange(e.target.value, direction === 'sell_btc' ? 'qnk' : 'btc')}
                    placeholder="0.00"
                    className="w-full bg-transparent text-xl font-mono text-white outline-none"
                  />
                </div>

                {/* BTC Destination (for buy_btc) */}
                {direction === 'buy_btc' && (
                  <div className="rounded-xl p-3 bg-white/5 border border-white/10">
                    <div className="text-xs text-gray-400 mb-1">BTC Destination Address</div>
                    <input
                      type="text"
                      value={btcDestination}
                      onChange={(e) => setBtcDestination(e.target.value)}
                      placeholder="bc1q... or 3..."
                      className="w-full bg-transparent text-sm font-mono text-white outline-none"
                    />
                  </div>
                )}

                {/* Rate Info */}
                <div className="flex items-center justify-between text-xs text-gray-500 px-1">
                  <span>Rate: 1 BTC = {BTC_QNK_RATE} QNK</span>
                  <span className="flex items-center gap-1">
                    <Clock size={12} />
                    ~10 min (1 BTC confirmation)
                  </span>
                </div>

                {/* Error/Success */}
                {error && (
                  <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                    <AlertCircle size={14} className="text-red-400" />
                    <span className="text-sm text-red-300">{error}</span>
                  </div>
                )}
                {success && (
                  <div className="flex items-center gap-2 p-3 rounded-lg bg-green-500/10 border border-green-500/20">
                    <CheckCircle size={14} className="text-green-400" />
                    <span className="text-sm text-green-300">{success}</span>
                  </div>
                )}

                {/* Submit Button */}
                <button
                  onClick={handleCreateSwap}
                  disabled={isSubmitting || !btcAmount || parseFloat(btcAmount) <= 0}
                  className="w-full py-3 rounded-xl font-semibold text-white transition-all disabled:opacity-40"
                  style={{
                    background: isSubmitting
                      ? 'rgba(251, 146, 60, 0.3)'
                      : 'linear-gradient(135deg, #f97316, #ea580c)',
                  }}
                >
                  {isSubmitting ? (
                    <span className="flex items-center justify-center gap-2">
                      <Loader2 size={16} className="animate-spin" />
                      Creating Swap...
                    </span>
                  ) : (
                    `Initiate ${direction === 'sell_btc' ? 'BTC → QNK' : 'QNK → BTC'} Swap`
                  )}
                </button>

                {/* Info */}
                <div className="text-xs text-gray-500 text-center leading-relaxed">
                  Atomic swaps use Hash Time-Locked Contracts (HTLC).
                  <br />
                  Trustless, non-custodial, with automatic refund on timeout.
                </div>
              </div>
            )}

            {activeTab === 'history' && (
              <div className="space-y-3">
                {swapHistory.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <ArrowRightLeft size={32} className="mx-auto mb-2 opacity-30" />
                    <p>No swaps yet</p>
                    <p className="text-xs mt-1">Create your first atomic swap above</p>
                  </div>
                ) : (
                  swapHistory.map(swap => (
                    <div
                      key={swap.swap_id}
                      className="rounded-xl p-3 bg-white/5 border border-white/10 hover:border-orange-500/20 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-mono text-gray-300">
                            {swap.swap_id.slice(0, 8)}...
                          </span>
                          <button
                            onClick={() => copyToClipboard(swap.swap_id, swap.swap_id)}
                            className="text-gray-500 hover:text-gray-300"
                          >
                            {copiedId === swap.swap_id ? <CheckCircle size={12} className="text-green-400" /> : <Copy size={12} />}
                          </button>
                        </div>
                        {statusBadge(swap.status)}
                      </div>
                      <div className="flex items-center justify-between text-xs text-gray-400">
                        <span>{(swap.btc_amount / 1e8).toFixed(8)} BTC</span>
                        <ArrowRightLeft size={12} className="text-gray-600" />
                        <span>{parseFloat(swap.qnk_amount) > 1e18
                          ? (parseFloat(swap.qnk_amount) / 1e24).toFixed(4)
                          : swap.qnk_amount} QNK</span>
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        {new Date(swap.created_at).toLocaleString()}
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default BitcoinSwapModal;
