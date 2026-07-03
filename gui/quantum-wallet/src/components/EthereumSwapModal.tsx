import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ArrowRightLeft, Clock, CheckCircle, AlertCircle, Copy, Loader2, Wallet, Shield } from 'lucide-react';
import { qnkAPI } from '../services/api';

interface EthereumSwapModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletAddress: string;
}

type SwapTab = 'swap' | 'metamask' | 'history';
type SwapDirection = 'buy_eth' | 'sell_eth';
type MetaMaskStep = 'input' | 'signing' | 'confirming' | 'attesting' | 'complete';

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

  // MetaMask WETH Bridge state
  const [mmStep, setMmStep] = useState<MetaMaskStep>('input');
  const [mmConnected, setMmConnected] = useState(false);
  const [mmAccount, setMmAccount] = useState<string | null>(null);
  const [mmWethAmount, setMmWethAmount] = useState('');
  const [mmQugEstimate, setMmQugEstimate] = useState('');
  const [mmDepositId, setMmDepositId] = useState<string | null>(null);
  const [mmTxHash, setMmTxHash] = useState<string | null>(null);
  const [mmConfirmations, setMmConfirmations] = useState(0);
  const [mmAttestations, setMmAttestations] = useState(0);
  const [mmError, setMmError] = useState<string | null>(null);
  const [mmBridgeInfo, setMmBridgeInfo] = useState<{
    bridge_deposit_address: string;
    weth_contract_address: string;
    chain_id: number;
    min_deposit_wei: string;
    max_deposit_wei: string;
    required_confirmations: number;
    required_attestations: number;
  } | null>(null);
  const [mmRate, setMmRate] = useState<number>(65.0);

  useEffect(() => {
    if (isOpen) {
      fetchBridgeStatus();
      fetchSwapHistory();
      fetchWethBalance();
      fetchEthAddress();
      fetchBridgeInfo();
      fetchBridgeRate();
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

  // MetaMask bridge support functions
  const fetchBridgeInfo = async () => {
    try {
      const res = await qnkAPI.getBridgeDepositAddress();
      if (res.success && res.data) {
        setMmBridgeInfo(res.data);
      }
    } catch (e) {
      console.warn('Failed to fetch bridge deposit info:', e);
    }
  };

  const fetchBridgeRate = async () => {
    try {
      const res = await qnkAPI.getBridgeRate();
      if (res.success && res.data) {
        setMmRate(res.data.weth_to_qug_rate);
      }
    } catch (e) {
      console.warn('Failed to fetch bridge rate:', e);
    }
  };

  const connectMetaMask = async () => {
    setMmError(null);
    const ethereum = (window as any).ethereum;
    if (!ethereum) {
      setMmError('MetaMask not detected. Please install MetaMask.');
      return;
    }
    try {
      const accounts = await ethereum.request({ method: 'eth_requestAccounts' });
      if (accounts.length > 0) {
        // Verify chain is Ethereum mainnet (chain_id 0x1)
        let chainId = await ethereum.request({ method: 'eth_chainId' });
        if (chainId !== '0x1') {
          try {
            await ethereum.request({
              method: 'wallet_switchEthereumChain',
              params: [{ chainId: '0x1' }],
            });
            chainId = '0x1';
          } catch (switchErr: any) {
            setMmError('Failed to switch to Ethereum Mainnet. Please switch manually in MetaMask.');
            return;
          }
        }
        setMmAccount(accounts[0]);
        setMmConnected(true);
      }
    } catch (e: any) {
      setMmError(e.message || 'Failed to connect MetaMask.');
    }
  };

  const getWethBalance = async (address: string): Promise<string> => {
    const ethereum = (window as any).ethereum;
    if (!ethereum || !mmBridgeInfo) return '0';
    // ERC-20 balanceOf(address) calldata
    const paddedAddr = address.slice(2).padStart(64, '0');
    const data = '0x70a08231' + paddedAddr; // balanceOf selector
    try {
      const result = await ethereum.request({
        method: 'eth_call',
        params: [{
          to: mmBridgeInfo.weth_contract_address,
          data,
        }, 'latest'],
      });
      return BigInt(result).toString();
    } catch {
      return '0';
    }
  };

  const sendWethToBridge = async () => {
    setMmError(null);
    if (!mmAccount || !mmBridgeInfo) return;

    const amount = parseFloat(mmWethAmount);
    if (isNaN(amount) || amount <= 0) {
      setMmError('Enter a valid WETH amount.');
      return;
    }

    const amountWei = BigInt(Math.round(amount * 1e18));
    const minWei = BigInt(mmBridgeInfo.min_deposit_wei);
    const maxWei = BigInt(mmBridgeInfo.max_deposit_wei);

    if (amountWei < minWei) {
      setMmError(`Minimum deposit is ${Number(minWei) / 1e18} WETH`);
      return;
    }
    if (amountWei > maxWei) {
      setMmError(`Maximum deposit is ${Number(maxWei) / 1e18} WETH`);
      return;
    }

    // Check WETH balance
    const balanceWei = await getWethBalance(mmAccount);
    if (BigInt(balanceWei) < amountWei) {
      setMmError(`Insufficient WETH balance. Have ${(Number(balanceWei) / 1e18).toFixed(6)} WETH`);
      return;
    }

    setMmStep('signing');

    try {
      const ethereum = (window as any).ethereum;
      // Build ERC-20 transfer(address,uint256) calldata
      const toAddr = mmBridgeInfo.bridge_deposit_address.slice(2).padStart(64, '0');
      const amountHex = amountWei.toString(16).padStart(64, '0');
      const data = '0xa9059cbb' + toAddr + amountHex; // transfer selector

      const txHash = await ethereum.request({
        method: 'eth_sendTransaction',
        params: [{
          from: mmAccount,
          to: mmBridgeInfo.weth_contract_address,
          data,
          // Gas will be estimated by MetaMask
        }],
      });

      setMmTxHash(txHash);
      setMmStep('confirming');

      // Register deposit with backend
      const res = await qnkAPI.registerWethDeposit({
        tx_hash: txHash,
        sender_address: mmAccount,
        amount_wei: amountWei.toString(),
      });

      if (res.success && res.data) {
        setMmDepositId(res.data.deposit_id);
        setMmQugEstimate(res.data.qug_estimate);
        // Start polling for status
        pollDepositStatus(res.data.deposit_id);
      } else {
        setMmError(res.error || 'Failed to register deposit.');
        setMmStep('input');
      }
    } catch (e: any) {
      if (e.code === 4001) {
        // User rejected in MetaMask
        setMmError('Transaction rejected by user.');
      } else {
        setMmError(e.message || 'MetaMask transaction failed.');
      }
      setMmStep('input');
    }
  };

  const pollDepositStatus = useCallback(async (depositId: string) => {
    const poll = async () => {
      try {
        const res = await qnkAPI.getDepositStatus(depositId);
        if (res.success && res.data) {
          setMmConfirmations(res.data.confirmations);
          setMmAttestations(res.data.attestations);

          if (res.data.status === 'confirming') {
            setMmStep('confirming');
          } else if (res.data.status === 'attesting') {
            setMmStep('attesting');
          } else if (res.data.status === 'completed') {
            setMmStep('complete');
            return; // Stop polling
          } else if (res.data.status === 'failed') {
            setMmError('Deposit verification failed.');
            setMmStep('input');
            return;
          }
        }
      } catch (e) {
        console.warn('Deposit status poll error:', e);
      }
      // Poll every 5 seconds
      setTimeout(poll, 5000);
    };
    poll();
  }, []);

  const handleMmAmountChange = (value: string) => {
    setMmWethAmount(value);
    const numVal = parseFloat(value) || 0;
    setMmQugEstimate((numVal * mmRate).toFixed(4));
  };

  const resetMetaMaskFlow = () => {
    setMmStep('input');
    setMmWethAmount('');
    setMmQugEstimate('');
    setMmDepositId(null);
    setMmTxHash(null);
    setMmConfirmations(0);
    setMmAttestations(0);
    setMmError(null);
  };

  // Exchange rate (placeholder - in production, fetch from Reth node oracle)
  const ETH_QNK_RATE = mmRate;
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
        className="fixed inset-0 z-50 overflow-y-auto"
        onClick={onClose}
      >
        {/* Backdrop */}
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm pointer-events-none" />

        <div className="flex min-h-full items-center justify-center p-4">
        {/* Modal */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ type: 'spring', damping: 25 }}
          onClick={(e) => e.stopPropagation()}
          className="relative w-full max-w-lg rounded-2xl overflow-y-auto max-h-[90vh]"
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
          <div className="px-5 py-2 flex flex-col gap-1 text-xs">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${bridgeStatus?.bridge_enabled ? 'bg-green-400 animate-pulse' : bridgeStatus?.sync_progress_pct ? 'bg-yellow-400 animate-pulse' : 'bg-red-400'}`} />
              <span className="text-gray-400">
                {bridgeStatus?.bridge_enabled ? 'Bridge Connected' : bridgeStatus?.sync_progress_pct ? 'Reth Syncing' : 'Bridge Offline'}
              </span>
              {bridgeStatus?.sync_progress_pct != null && !bridgeStatus?.reth_synced && (
                <span className="text-yellow-400 font-mono">
                  {bridgeStatus.sync_progress_pct.toFixed(2)}%
                </span>
              )}
              <span className="text-gray-600 ml-auto">HTLC Protocol</span>
            </div>
            {bridgeStatus?.sync_progress_pct != null && !bridgeStatus?.reth_synced && (
              <div className="flex flex-col gap-1">
                <div className="w-full h-1.5 bg-gray-700/50 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-yellow-500 to-amber-400 rounded-full transition-all duration-1000"
                    style={{ width: `${Math.min(bridgeStatus.sync_progress_pct, 100)}%` }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-gray-500">
                  <span>Block {bridgeStatus.sync_current_block?.toLocaleString() ?? '?'}</span>
                  <span>Target {bridgeStatus.sync_target_block?.toLocaleString() ?? '?'}</span>
                </div>
              </div>
            )}
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
            {(['swap', 'metamask', 'history'] as SwapTab[]).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab
                    ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                }`}
              >
                {tab === 'swap' ? 'Swap' : tab === 'metamask' ? 'MetaMask Bridge' : `History (${swapHistory.length})`}
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

            {activeTab === 'metamask' && (
              <div className="space-y-4">
                {/* Step Progress Bar */}
                <div className="flex items-center justify-between mb-2">
                  {(['input', 'signing', 'confirming', 'attesting', 'complete'] as MetaMaskStep[]).map((step, i) => {
                    const steps: MetaMaskStep[] = ['input', 'signing', 'confirming', 'attesting', 'complete'];
                    const currentIdx = steps.indexOf(mmStep);
                    const stepIdx = i;
                    const isActive = stepIdx === currentIdx;
                    const isDone = stepIdx < currentIdx;
                    return (
                      <div key={step} className="flex items-center flex-1">
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold border-2 transition-colors ${
                          isDone ? 'bg-green-500 border-green-500 text-white' :
                          isActive ? 'bg-indigo-500 border-indigo-400 text-white' :
                          'border-gray-600 text-gray-500'
                        }`}>
                          {isDone ? <CheckCircle size={14} /> : i + 1}
                        </div>
                        {i < 4 && (
                          <div className={`flex-1 h-0.5 mx-1 ${isDone ? 'bg-green-500' : 'bg-gray-700'}`} />
                        )}
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between text-[10px] text-gray-500 -mt-1 mb-3">
                  <span>Input</span><span>Sign</span><span>Confirm</span><span>Attest</span><span>Done</span>
                </div>

                {mmError && (
                  <div className="rounded-lg p-3 bg-red-500/10 border border-red-500/20 text-red-300 text-sm flex items-center gap-2">
                    <AlertCircle size={16} />
                    {mmError}
                  </div>
                )}

                {/* Step: Input */}
                {mmStep === 'input' && (
                  <div className="space-y-4">
                    {!mmConnected ? (
                      <button
                        onClick={connectMetaMask}
                        className="w-full py-3 rounded-xl bg-orange-500/20 border border-orange-500/30 text-orange-300 font-medium hover:bg-orange-500/30 transition-colors flex items-center justify-center gap-2"
                      >
                        <Wallet size={18} />
                        Connect MetaMask
                      </button>
                    ) : (
                      <>
                        <div className="rounded-lg p-3 bg-green-500/10 border border-green-500/20 text-green-300 text-sm flex items-center gap-2">
                          <CheckCircle size={16} />
                          Connected: {mmAccount?.slice(0, 8)}...{mmAccount?.slice(-6)}
                        </div>

                        <div className="rounded-xl p-4 bg-white/5 border border-white/10">
                          <label className="text-xs text-gray-400 mb-1 block">WETH Amount</label>
                          <div className="flex items-center gap-2">
                            <input
                              type="number"
                              value={mmWethAmount}
                              onChange={e => handleMmAmountChange(e.target.value)}
                              placeholder="0.01"
                              step="0.001"
                              min="0.001"
                              max="1.0"
                              className="flex-1 bg-transparent text-xl font-mono text-white outline-none"
                            />
                            <span className="text-gray-400 text-sm">WETH</span>
                          </div>
                          <div className="text-xs text-gray-500 mt-1">Min 0.001 WETH, Max 1.0 WETH</div>
                        </div>

                        <div className="rounded-xl p-4 bg-white/5 border border-white/10">
                          <label className="text-xs text-gray-400 mb-1 block">You Receive (estimate)</label>
                          <div className="flex items-center gap-2">
                            <span className="flex-1 text-xl font-mono text-indigo-300">
                              {mmQugEstimate || '0.0000'}
                            </span>
                            <span className="text-gray-400 text-sm">QUG</span>
                          </div>
                          <div className="text-xs text-gray-500 mt-1">Rate: 1 WETH = {mmRate} QUG</div>
                        </div>

                        {mmBridgeInfo && (
                          <div className="text-xs text-gray-500 space-y-1">
                            <div className="flex justify-between">
                              <span>Bridge Address:</span>
                              <span className="font-mono">{mmBridgeInfo.bridge_deposit_address.slice(0, 10)}...</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Confirmations Required:</span>
                              <span>{mmBridgeInfo.required_confirmations}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Committee Attestations:</span>
                              <span>{mmBridgeInfo.required_attestations}/11</span>
                            </div>
                          </div>
                        )}

                        <button
                          onClick={sendWethToBridge}
                          disabled={!mmWethAmount || parseFloat(mmWethAmount) < 0.001}
                          className="w-full py-3 rounded-xl bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 font-medium hover:bg-indigo-500/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                          <Shield size={18} />
                          Send WETH to Bridge
                        </button>
                      </>
                    )}
                  </div>
                )}

                {/* Step: Signing */}
                {mmStep === 'signing' && (
                  <div className="text-center py-8 space-y-4">
                    <Loader2 size={40} className="mx-auto animate-spin text-orange-400" />
                    <p className="text-lg font-medium text-orange-300">Waiting for MetaMask...</p>
                    <p className="text-sm text-gray-400">Please confirm the WETH transfer in your MetaMask wallet.</p>
                    <p className="text-xs text-gray-500">Sending {mmWethAmount} WETH to bridge deposit address</p>
                  </div>
                )}

                {/* Step: Confirming */}
                {mmStep === 'confirming' && (
                  <div className="text-center py-6 space-y-4">
                    <div className="relative mx-auto w-20 h-20">
                      <svg className="w-20 h-20 -rotate-90">
                        <circle cx="40" cy="40" r="36" fill="none" stroke="#1e1b4b" strokeWidth="4" />
                        <circle cx="40" cy="40" r="36" fill="none" stroke="#818cf8" strokeWidth="4"
                          strokeDasharray={`${(mmConfirmations / 12) * 226} 226`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <span className="absolute inset-0 flex items-center justify-center text-lg font-bold text-indigo-300">
                        {mmConfirmations}/12
                      </span>
                    </div>
                    <p className="text-lg font-medium text-indigo-300">Confirming on Ethereum</p>
                    <p className="text-sm text-gray-400">Waiting for block confirmations...</p>
                    {mmTxHash && (
                      <div className="flex items-center justify-center gap-2 text-xs text-gray-500">
                        <span className="font-mono">{mmTxHash.slice(0, 16)}...{mmTxHash.slice(-8)}</span>
                        <button onClick={() => copyToClipboard(mmTxHash!, 'mm-tx')} className="hover:text-gray-300">
                          {copiedId === 'mm-tx' ? <CheckCircle size={10} className="text-green-400" /> : <Copy size={10} />}
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* Step: Attesting */}
                {mmStep === 'attesting' && (
                  <div className="text-center py-6 space-y-4">
                    <div className="relative mx-auto w-20 h-20">
                      <svg className="w-20 h-20 -rotate-90">
                        <circle cx="40" cy="40" r="36" fill="none" stroke="#1e1b4b" strokeWidth="4" />
                        <circle cx="40" cy="40" r="36" fill="none" stroke="#22c55e" strokeWidth="4"
                          strokeDasharray={`${(mmAttestations / 7) * 226} 226`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <span className="absolute inset-0 flex items-center justify-center text-lg font-bold text-green-300">
                        {mmAttestations}/7
                      </span>
                    </div>
                    <p className="text-lg font-medium text-green-300">Committee Attestation</p>
                    <p className="text-sm text-gray-400">Bridge committee verifying deposit...</p>
                    <p className="text-xs text-gray-500">7 of 11 attestations required</p>
                  </div>
                )}

                {/* Step: Complete */}
                {mmStep === 'complete' && (
                  <div className="text-center py-6 space-y-4">
                    <CheckCircle size={48} className="mx-auto text-green-400" />
                    <p className="text-lg font-medium text-green-300">Bridge Complete!</p>
                    <div className="rounded-xl p-4 bg-green-500/10 border border-green-500/20 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Sent</span>
                        <span className="text-white font-mono">{mmWethAmount} WETH</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Received</span>
                        <span className="text-green-300 font-mono">{mmQugEstimate} QUG</span>
                      </div>
                      {mmTxHash && (
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-500">Tx Hash</span>
                          <span className="text-gray-400 font-mono">{mmTxHash.slice(0, 12)}...{mmTxHash.slice(-6)}</span>
                        </div>
                      )}
                    </div>
                    <button
                      onClick={resetMetaMaskFlow}
                      className="w-full py-2 rounded-xl bg-white/5 border border-white/10 text-gray-300 hover:bg-white/10 transition-colors text-sm"
                    >
                      Bridge More WETH
                    </button>
                  </div>
                )}

                <div className="text-xs text-gray-500 text-center leading-relaxed">
                  MetaMask WETH Bridge uses ERC-20 deposit verification.
                  <br />
                  Deposits are verified by the bridge committee (7-of-11 attestation).
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
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default EthereumSwapModal;
