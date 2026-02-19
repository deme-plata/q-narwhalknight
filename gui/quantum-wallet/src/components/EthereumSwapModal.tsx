import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ArrowRightLeft, Clock, CheckCircle, AlertCircle, Copy, Loader2 } from 'lucide-react';
import { qnkAPI } from '../services/api';

interface EthereumSwapModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletAddress: string;
}

type SwapTab = 'swap' | 'history';
type SwapDirection = 'buy_eth' | 'sell_eth';

interface SwapHistoryItem {
  swap_id: string;
  eth_amount: number;
  qnk_amount: string;
  status: string;
  created_at: string;
  hash_lock: string;
}

const EthereumSwapModal = ({ isOpen, onClose, walletAddress }: EthereumSwapModalProps) => {
  const [activeTab, setActiveTab] = useState<SwapTab>('swap');
  const [direction, setDirection] = useState<SwapDirection>('sell_eth');
  const [ethAmount, setEthAmount] = useState('');
  const [qnkAmount, setQnkAmount] = useState('');
  const [ethDestination, setEthDestination] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [swapHistory, setSwapHistory] = useState<SwapHistoryItem[]>([]);
  const [bridgeStatus, setBridgeStatus] = useState<any>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [wethBalance, setWethBalance] = useState<{ balance_wei: string; balance_eth: number } | null>(null);
  const [ethAddress, setEthAddress] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchBridgeStatus();
      fetchSwapHistory();
      fetchWethBalance();
      fetchEthAddress();
    }
  }, [isOpen]);

  const fetchBridgeStatus = async () => {
    try {
      const res = await qnkAPI.getEthBridgeStatus();
      if (res.success && res.data) {
        setBridgeStatus(res.data);
      }
    } catch (e) {
      console.warn('Failed to fetch ETH bridge status:', e);
    }
  };

  const fetchSwapHistory = async () => {
    try {
      const res = await qnkAPI.listEthSwaps();
      if (res.success && res.data) {
        setSwapHistory(res.data.swaps || []);
      }
    } catch (e) {
      console.warn('Failed to fetch ETH swap history:', e);
    }
  };

  const fetchWethBalance = async () => {
    try {
      const res = await qnkAPI.getEthBalance();
      if (res.success && res.data) {
        setWethBalance(res.data);
      }
    } catch (e) {
      console.warn('Failed to fetch wETH balance:', e);
    }
  };

  const fetchEthAddress = async () => {
    try {
      const res = await qnkAPI.getEthAddress();
      if (res.success && res.data) {
        setEthAddress(res.data.eth_address);
      }
    } catch (e) {
      console.warn('Failed to fetch ETH address:', e);
    }
  };

  // Exchange rate (placeholder - in production, fetch from Reth node oracle)
  const ETH_QNK_RATE = 65.0; // 1 ETH = ~65 QNK equivalent
  const ETH_USD_RATE = 2750;

  const handleAmountChange = (value: string, field: 'eth' | 'qnk') => {
    const numVal = parseFloat(value) || 0;
    if (field === 'eth') {
      setEthAmount(value);
      setQnkAmount((numVal * ETH_QNK_RATE).toFixed(4));
    } else {
      setQnkAmount(value);
      setEthAmount((numVal / ETH_QNK_RATE).toFixed(8));
    }
  };

  const handleCreateSwap = async () => {
    setError(null);
    setSuccess(null);
    setIsSubmitting(true);

    try {
      const ethWei = BigInt(Math.round(parseFloat(ethAmount) * 1e18));
      const qnkBase = BigInt(Math.round(parseFloat(qnkAmount) * 1e8)) * BigInt(1e16); // 24 decimals

      if (ethWei <= 0n) {
        setError('Enter a valid ETH amount.');
        setIsSubmitting(false);
        return;
      }

      if (direction === 'buy_eth' && !ethDestination) {
        setError('Enter an Ethereum destination address.');
        setIsSubmitting(false);
        return;
      }

      if (direction === 'buy_eth' && !ethDestination.match(/^0x[0-9a-fA-F]{40}$/)) {
        setError('Invalid Ethereum address. Must start with 0x followed by 40 hex characters.');
        setIsSubmitting(false);
        return;
      }

      const res = await qnkAPI.createEthSwap({
        direction,
        eth_amount: ethWei.toString(),
        qnk_amount: qnkBase.toString(),
        eth_destination: ethDestination || undefined,
      });

      if (res.success && res.data) {
        setSuccess(`Swap created! ID: ${res.data.swap_id}`);
        setEthAmount('');
        setQnkAmount('');
        setEthDestination('');
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
      eth_locked: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
      qnk_locked: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
      qnk_claimed: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
      eth_claimed: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
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
            background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.98), rgba(10, 15, 30, 0.95))',
            border: '1px solid rgba(99, 102, 241, 0.3)',
            boxShadow: '0 0 60px rgba(99, 102, 241, 0.15)',
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-5 border-b border-indigo-500/20">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: 'linear-gradient(135deg, #6366f1, #4f46e5)' }}>
                <span className="text-lg font-bold text-white">Ξ</span>
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">Ethereum Bridge</h2>
                <p className="text-xs text-indigo-300/60">QNK ↔ ETH Atomic Swap</p>
              </div>
            </div>
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white">
              <X size={18} />
            </button>
          </div>

          {/* Bridge Status */}
          <div className="px-5 py-2 flex items-center gap-2 text-xs">
            <div className={`w-2 h-2 rounded-full ${bridgeStatus?.bridge_enabled ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
            <span className="text-gray-400">
              {bridgeStatus?.bridge_enabled ? 'Bridge Connected' : 'Bridge Offline'}
            </span>
            {bridgeStatus?.reth_synced === false && (
              <span className="text-yellow-400 ml-1">(Reth syncing...)</span>
            )}
            <span className="text-gray-600 ml-auto">HTLC Protocol</span>
          </div>

          {/* wETH Balance & ETH Address */}
          <div className="mx-5 mb-2 rounded-xl p-3 bg-indigo-500/5 border border-indigo-500/10">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-gray-500">Your wETH Balance</div>
                <div className="text-lg font-mono text-white">
                  {wethBalance ? `${wethBalance.balance_eth.toFixed(6)} wETH` : '—'}
                </div>
              </div>
              {ethAddress && (
                <div className="text-right">
                  <div className="text-xs text-gray-500">Derived ETH Address</div>
                  <div className="flex items-center gap-1">
                    <span className="text-xs font-mono text-indigo-300/70">
                      {ethAddress.slice(0, 8)}...{ethAddress.slice(-6)}
                    </span>
                    <button
                      onClick={() => copyToClipboard(ethAddress, 'eth-addr')}
                      className="text-gray-500 hover:text-gray-300"
                    >
                      {copiedId === 'eth-addr' ? <CheckCircle size={10} className="text-green-400" /> : <Copy size={10} />}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 px-5 pt-2">
            {(['swap', 'history'] as SwapTab[]).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab
                    ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
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
                    onClick={() => setDirection('sell_eth')}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                      direction === 'sell_eth'
                        ? 'bg-indigo-500/30 text-indigo-200'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    ETH → QNK
                  </button>
                  <button
                    onClick={() => setDirection('buy_eth')}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                      direction === 'buy_eth'
                        ? 'bg-indigo-500/30 text-indigo-200'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    QNK → ETH
                  </button>
                </div>

                {/* From Amount */}
                <div className="rounded-xl p-3 bg-white/5 border border-white/10">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>You send</span>
                    <span>{direction === 'sell_eth' ? 'ETH' : 'QNK'}</span>
                  </div>
                  <input
                    type="number"
                    value={direction === 'sell_eth' ? ethAmount : qnkAmount}
                    onChange={(e) => handleAmountChange(e.target.value, direction === 'sell_eth' ? 'eth' : 'qnk')}
                    placeholder="0.00"
                    className="w-full bg-transparent text-xl font-mono text-white outline-none"
                  />
                  {direction === 'sell_eth' && ethAmount && (
                    <div className="text-xs text-gray-500 mt-1">
                      ≈ ${(parseFloat(ethAmount) * ETH_USD_RATE).toLocaleString(undefined, { maximumFractionDigits: 2 })} USD
                    </div>
                  )}
                </div>

                {/* Swap Arrow */}
                <div className="flex justify-center">
                  <div className="p-2 rounded-full bg-indigo-500/20 border border-indigo-500/30">
                    <ArrowRightLeft size={16} className="text-indigo-400" />
                  </div>
                </div>

                {/* To Amount */}
                <div className="rounded-xl p-3 bg-white/5 border border-white/10">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>You receive</span>
                    <span>{direction === 'sell_eth' ? 'QNK' : 'ETH'}</span>
                  </div>
                  <input
                    type="number"
                    value={direction === 'sell_eth' ? qnkAmount : ethAmount}
                    onChange={(e) => handleAmountChange(e.target.value, direction === 'sell_eth' ? 'qnk' : 'eth')}
                    placeholder="0.00"
                    className="w-full bg-transparent text-xl font-mono text-white outline-none"
                  />
                </div>

                {/* ETH Destination (for buy_eth) */}
                {direction === 'buy_eth' && (
                  <div className="rounded-xl p-3 bg-white/5 border border-white/10">
                    <div className="text-xs text-gray-400 mb-1">ETH Destination Address</div>
                    <input
                      type="text"
                      value={ethDestination}
                      onChange={(e) => setEthDestination(e.target.value)}
                      placeholder="0x..."
                      className="w-full bg-transparent text-sm font-mono text-white outline-none"
                    />
                  </div>
                )}

                {/* Rate Info */}
                <div className="flex items-center justify-between text-xs text-gray-500 px-1">
                  <span>Rate: 1 ETH = {ETH_QNK_RATE} QNK</span>
                  <span className="flex items-center gap-1">
                    <Clock size={12} />
                    ~2 min (12 ETH confirmations)
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
                  disabled={isSubmitting || !ethAmount || parseFloat(ethAmount) <= 0}
                  className="w-full py-3 rounded-xl font-semibold text-white transition-all disabled:opacity-40"
                  style={{
                    background: isSubmitting
                      ? 'rgba(99, 102, 241, 0.3)'
                      : 'linear-gradient(135deg, #6366f1, #4f46e5)',
                  }}
                >
                  {isSubmitting ? (
                    <span className="flex items-center justify-center gap-2">
                      <Loader2 size={16} className="animate-spin" />
                      Creating Swap...
                    </span>
                  ) : (
                    `Initiate ${direction === 'sell_eth' ? 'ETH → QNK' : 'QNK → ETH'} Swap`
                  )}
                </button>

                {/* Info */}
                <div className="text-xs text-gray-500 text-center leading-relaxed">
                  Atomic swaps use Hash Time-Locked Contracts (HTLC).
                  <br />
                  Trustless, non-custodial, with automatic refund on timeout.
                  <br />
                  <span className="text-indigo-400/50">Powered by Reth full node on Server Delta</span>
                </div>
              </div>
            )}

            {activeTab === 'history' && (
              <div className="space-y-3">
                {swapHistory.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <ArrowRightLeft size={32} className="mx-auto mb-2 opacity-30" />
                    <p>No swaps yet</p>
                    <p className="text-xs mt-1">Create your first ETH atomic swap above</p>
                  </div>
                ) : (
                  swapHistory.map(swap => (
                    <div
                      key={swap.swap_id}
                      className="rounded-xl p-3 bg-white/5 border border-white/10 hover:border-indigo-500/20 transition-colors"
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
                        <span>{(swap.eth_amount / 1e18).toFixed(6)} ETH</span>
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

export default EthereumSwapModal;
