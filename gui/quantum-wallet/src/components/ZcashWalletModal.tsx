import React, { useState, useEffect, useCallback } from 'react';
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

  // Send form
  const [sendAddress, setSendAddress] = useState('');
  const [sendAmount, setSendAmount] = useState('');
  const [sendMemo, setSendMemo] = useState('');

  // Swap form
  const [swapDirection, setSwapDirection] = useState<'buy_zec' | 'sell_zec'>('buy_zec');
  const [swapZecAmount, setSwapZecAmount] = useState('');
  const [swapQnkAmount, setSwapQnkAmount] = useState('');
  const [receiveZAddress, setReceiveZAddress] = useState('');

  // Exchange rates — fetch dynamically
  const [zecUsdRate, setZecUsdRate] = useState(25);
  const [qugUsdRate, setQugUsdRate] = useState(3000);

  useEffect(() => {
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
      try {
        const res = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=zcash&vs_currencies=usd');
        const data = await res.json();
        if (data?.zcash?.usd) {
          setZecUsdRate(data.zcash.usd);
        }
      } catch (e) {
        console.warn('Failed to fetch ZEC price, using default');
      }
    };
    if (isOpen) fetchRates();
  }, [isOpen]);

  const ZEC_QNK_RATE = qugUsdRate > 0 ? zecUsdRate / qugUsdRate : 0.008; // 1 ZEC ≈ 0.008 QNK at $25/$3000

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
    if (isOpen) {
      fetchData();
    }
  }, [isOpen, fetchData]);

  const handleSend = async () => {
    const isShielded = sendAddress.startsWith('zs') || sendAddress.startsWith('u1');
    if (!isShielded) {
      setError('Only shielded addresses are supported (zs1... for Sapling, u1... for Unified/Orchard).');
      return;
    }
    const amountZat = Math.round(parseFloat(sendAmount) * 100_000_000);
    if (isNaN(amountZat) || amountZat <= 0) {
      setError('Invalid amount.');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const res = await qnkAPI.sendShieldedZec({
        to_z_address: sendAddress,
        amount_zat: amountZat,
        memo: sendMemo || undefined,
      });

      if (res.success && res.data) {
        setSuccess(`Shielded transaction submitted: ${res.data.tx_id}`);
        setSendAddress('');
        setSendAmount('');
        setSendMemo('');
        fetchData();
      } else {
        setError(res.error || 'Send failed');
      }
    } catch (e: any) {
      setError(e.message || 'Send failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSwap = async () => {
    const zecAmountZat = Math.round(parseFloat(swapZecAmount) * 100_000_000);
    if (isNaN(zecAmountZat) || zecAmountZat <= 0) {
      setError('Invalid ZEC amount.');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const res = await qnkAPI.createZcashSwap({
        direction: swapDirection,
        zec_amount: zecAmountZat,
        qnk_amount: swapQnkAmount || '0',
        z_address: receiveZAddress || zAddress || undefined,
      });

      if (res.success && res.data) {
        setSuccess(`Swap created: ${res.data.swap_id}`);
        setSwapZecAmount('');
        setSwapQnkAmount('');
        fetchData();
      } else {
        setError(res.error || 'Swap creation failed');
      }
    } catch (e: any) {
      setError(e.message || 'Swap failed');
    } finally {
      setLoading(false);
    }
  };

  const copyAddress = (addr: string) => {
    navigator.clipboard.writeText(addr);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleMaxSend = () => {
    const fee = 0.0001;
    const max = Math.max(0, balanceZec - fee);
    setSendAmount(max > 0 ? max.toFixed(8) : '');
  };

  if (!isOpen) return null;

  const formatZec = (zat: number) => (zat / 100_000_000).toFixed(8);
  const truncateAddr = (addr: string) => addr.length > 24 ? `${addr.slice(0, 12)}...${addr.slice(-12)}` : addr;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-purple-500/30 rounded-2xl w-full max-w-2xl max-h-[85vh] overflow-hidden shadow-2xl shadow-purple-500/10">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600/20 to-indigo-600/20 border-b border-purple-500/20 p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-indigo-500 flex items-center justify-center text-xl font-bold text-white">Z</div>
              <div>
                <h2 className="text-xl font-bold text-white">Zcash Wallet</h2>
                <p className="text-xs text-purple-300/70">Shielded transactions only</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {bridgeStatus && (
                <span className={`px-2 py-1 rounded text-xs font-medium ${bridgeStatus.zebra_syncing ? 'bg-yellow-500/20 text-yellow-300' : 'bg-green-500/20 text-green-300'}`}>
                  {bridgeStatus.zebra_syncing ? `Syncing (${bridgeStatus.zebra_height?.toLocaleString()})` : `Synced (${bridgeStatus.zebra_height?.toLocaleString()})`}
                </span>
              )}
              <button onClick={fetchData} title="Refresh" className="text-gray-400 hover:text-purple-300 p-1 transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
              </button>
              <button onClick={onClose} className="text-gray-400 hover:text-white p-1">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-700/50">
          {(['balance', 'send', 'swap', 'history'] as Tab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => { setActiveTab(tab); setError(''); setSuccess(''); }}
              className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                activeTab === tab
                  ? 'text-purple-300 border-b-2 border-purple-500 bg-purple-500/5'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {tab === 'balance' && 'Balance'}
              {tab === 'send' && 'Send ZEC'}
              {tab === 'swap' && 'Swap'}
              {tab === 'history' && 'History'}
            </button>
          ))}
        </div>

        {/* Status messages */}
        {error && (
          <div className="mx-5 mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-300 text-sm">{error}</div>
        )}
        {success && (
          <div className="mx-5 mt-3 p-3 bg-green-500/10 border border-green-500/20 rounded-lg text-green-300 text-sm">{success}</div>
        )}

        {/* Content */}
        <div className="p-5 overflow-y-auto" style={{ maxHeight: 'calc(85vh - 200px)' }}>
          {/* Balance Tab */}
          {activeTab === 'balance' && (
            <div className="space-y-5">
              <div className="bg-gradient-to-br from-purple-900/30 to-indigo-900/30 rounded-xl p-6 border border-purple-500/20">
                <p className="text-gray-400 text-sm mb-1">Shielded Balance</p>
                <p className="text-3xl font-bold text-white">{balanceZec.toFixed(8)} <span className="text-purple-400 text-lg">ZEC</span></p>
                <p className="text-gray-500 text-xs mt-1">{balanceZat.toLocaleString()} zatoshis</p>
              </div>

              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/30">
                <p className="text-gray-400 text-xs mb-2">Your Shielded Address (z-address)</p>
                {zAddress ? (
                  <div className="flex items-center gap-2">
                    <code className="text-purple-300 text-xs flex-1 break-all font-mono">{zAddress}</code>
                    <button onClick={() => copyAddress(zAddress)} className="px-3 py-1 bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 rounded text-xs whitespace-nowrap transition-colors">
                      {copied ? '✓ Copied' : 'Copy'}
                    </button>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">Loading...</p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/20">
                  <p className="text-gray-500 text-xs">Network</p>
                  <p className="text-white text-sm font-medium">Zcash Mainnet</p>
                </div>
                <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/20">
                  <p className="text-gray-500 text-xs">Address Type</p>
                  <p className="text-purple-300 text-sm font-medium">Sapling (Shielded)</p>
                </div>
                <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/20">
                  <p className="text-gray-500 text-xs">Zebra Node</p>
                  <p className="text-white text-sm font-medium">{bridgeStatus?.zebra_height?.toLocaleString() || '...'} blocks</p>
                </div>
                <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/20">
                  <p className="text-gray-500 text-xs">Privacy Level</p>
                  <p className="text-green-300 text-sm font-medium">Maximum (Shielded Only)</p>
                </div>
              </div>
            </div>
          )}

          {/* Send Tab */}
          {activeTab === 'send' && (
            <div className="space-y-4">
              <div>
                <label className="text-gray-400 text-sm mb-1 block">Destination z-address</label>
                <input
                  type="text"
                  value={sendAddress}
                  onChange={(e) => setSendAddress(e.target.value)}
                  placeholder="zs1... or u1..."
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm font-mono focus:border-purple-500/50 focus:outline-none"
                />
                <p className="text-gray-500 text-xs mt-1">Sapling (zs1...) or Unified/Orchard (u1...) addresses accepted</p>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-gray-400 text-sm">Amount (ZEC)</label>
                  <button onClick={handleMaxSend} className="text-purple-400 hover:text-purple-300 text-xs font-medium">
                    MAX
                  </button>
                </div>
                <input
                  type="number"
                  step="0.00000001"
                  value={sendAmount}
                  onChange={(e) => setSendAmount(e.target.value)}
                  placeholder="0.00000000"
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-purple-500/50 focus:outline-none"
                />
                <div className="flex justify-between mt-1">
                  <p className="text-gray-500 text-xs">Available: {balanceZec.toFixed(8)} ZEC</p>
                  <p className="text-gray-500 text-xs">Fee: 0.0001 ZEC</p>
                </div>
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-1 block">Encrypted Memo (optional, 512 bytes max)</label>
                <textarea
                  value={sendMemo}
                  onChange={(e) => setSendMemo(e.target.value)}
                  placeholder="Private message..."
                  maxLength={512}
                  rows={2}
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-purple-500/50 focus:outline-none resize-none"
                />
              </div>

              <button
                onClick={handleSend}
                disabled={loading || !sendAddress || !sendAmount}
                className="w-full py-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-all"
              >
                {loading ? 'Sending...' : 'Send Shielded ZEC'}
              </button>

              <div className="bg-purple-500/5 border border-purple-500/10 rounded-lg p-3">
                <p className="text-purple-300/60 text-xs">
                  All transactions use shielded (Sapling) pools. Amounts, sender, and recipient are encrypted on-chain. Only the encrypted memo is visible to the recipient.
                </p>
              </div>
            </div>
          )}

          {/* Swap Tab */}
          {activeTab === 'swap' && (
            <div className="space-y-4">
              <div className="flex gap-2 bg-gray-800/30 rounded-lg p-1">
                <button
                  onClick={() => setSwapDirection('buy_zec')}
                  className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                    swapDirection === 'buy_zec' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  QNK to ZEC
                </button>
                <button
                  onClick={() => setSwapDirection('sell_zec')}
                  className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                    swapDirection === 'sell_zec' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  ZEC to QNK
                </button>
              </div>

              {/* Exchange rate display */}
              <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/20 flex items-center justify-between">
                <span className="text-gray-400 text-xs">Exchange Rate</span>
                <span className="text-purple-300 text-sm font-medium">
                  1 ZEC ≈ {ZEC_QNK_RATE.toFixed(6)} QNK
                  <span className="text-gray-500 ml-2">(${zecUsdRate.toFixed(2)})</span>
                </span>
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-1 block">ZEC Amount</label>
                <input
                  type="number"
                  step="0.00000001"
                  value={swapZecAmount}
                  onChange={(e) => handleSwapAmountChange(e.target.value, 'zec')}
                  placeholder="0.00000000"
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-purple-500/50 focus:outline-none"
                />
                {swapZecAmount && parseFloat(swapZecAmount) > 0 && (
                  <p className="text-gray-500 text-xs mt-1">≈ ${(parseFloat(swapZecAmount) * zecUsdRate).toFixed(2)} USD</p>
                )}
              </div>

              <div className="flex items-center justify-center">
                <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center">
                  <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" /></svg>
                </div>
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-1 block">QNK Amount (QUG)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={swapQnkAmount}
                  onChange={(e) => handleSwapAmountChange(e.target.value, 'qnk')}
                  placeholder="0.0000"
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-purple-500/50 focus:outline-none"
                />
                {swapQnkAmount && parseFloat(swapQnkAmount) > 0 && (
                  <p className="text-gray-500 text-xs mt-1">≈ ${(parseFloat(swapQnkAmount) * qugUsdRate).toFixed(2)} USD</p>
                )}
              </div>

              {swapDirection === 'buy_zec' && (
                <div>
                  <label className="text-gray-400 text-sm mb-1 block">Your z-address (for receiving ZEC)</label>
                  <input
                    type="text"
                    value={receiveZAddress || zAddress}
                    onChange={(e) => setReceiveZAddress(e.target.value)}
                    placeholder="zs1..."
                    className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm font-mono focus:border-purple-500/50 focus:outline-none"
                  />
                </div>
              )}

              <button
                onClick={handleSwap}
                disabled={loading || !swapZecAmount}
                className="w-full py-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-all"
              >
                {loading ? 'Creating Swap...' : `Create ${swapDirection === 'buy_zec' ? 'QNK → ZEC' : 'ZEC → QNK'} Swap`}
              </button>

              <div className="bg-purple-500/5 border border-purple-500/10 rounded-lg p-3 space-y-2">
                <p className="text-purple-300/80 text-xs font-medium">How Shielded Atomic Swaps Work:</p>
                <ol className="text-purple-300/60 text-xs space-y-1 list-decimal list-inside">
                  <li>A cryptographic hash-lock is generated</li>
                  <li>QNK is locked in an on-chain HTLC escrow</li>
                  <li>ZEC is sent to your z-address via shielded memo</li>
                  <li>Revealing the secret completes both sides</li>
                  <li>If timeout expires, both sides are refunded</li>
                </ol>
              </div>
            </div>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <div className="space-y-3">
              {swapHistory.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <p className="text-lg mb-1">No swap history</p>
                  <p className="text-sm">Your shielded atomic swaps will appear here.</p>
                </div>
              ) : (
                swapHistory.map((swap) => (
                  <div key={swap.swap_id} className="bg-gray-800/30 rounded-lg p-4 border border-gray-700/20">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className={`text-sm font-medium ${swap.direction === 'buy_zec' ? 'text-purple-300' : 'text-green-300'}`}>
                          {swap.direction === 'buy_zec' ? 'QNK → ZEC' : 'ZEC → QNK'}
                        </span>
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          swap.status === 'completed' ? 'bg-green-500/20 text-green-300' :
                          swap.status === 'proposed' ? 'bg-blue-500/20 text-blue-300' :
                          swap.status === 'refunded' ? 'bg-yellow-500/20 text-yellow-300' :
                          swap.status === 'failed' ? 'bg-red-500/20 text-red-300' :
                          'bg-gray-500/20 text-gray-300'
                        }`}>
                          {swap.status}
                        </span>
                      </div>
                      <span className="text-gray-500 text-xs">{new Date(swap.created_at).toLocaleString()}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-gray-500">ZEC:</span>{' '}
                        <span className="text-white">{formatZec(swap.zec_amount)}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">QNK:</span>{' '}
                        <span className="text-white">{parseFloat(swap.qnk_amount || '0').toLocaleString()}</span>
                      </div>
                    </div>
                    <p className="text-gray-600 text-xs mt-1 font-mono">{truncateAddr(swap.swap_id)}</p>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ZcashWalletModal;
