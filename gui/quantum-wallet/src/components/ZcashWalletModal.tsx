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

type Tab = 'balance' | 'send' | 'receive' | 'history';

const ZcashWalletModal: React.FC<ZcashWalletModalProps> = ({ isOpen, onClose, walletAddress }) => {
  const [activeTab, setActiveTab] = useState<Tab>('balance');
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
    if (!sendAddress.startsWith('zs1') && !sendAddress.startsWith('zs')) {
      setError('Only shielded z-addresses (zs1...) are supported.');
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
    setSuccess('Address copied to clipboard');
    setTimeout(() => setSuccess(''), 2000);
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
              <button onClick={onClose} className="text-gray-400 hover:text-white p-1">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-700/50">
          {(['balance', 'send', 'receive', 'history'] as Tab[]).map((tab) => (
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
              {tab === 'receive' && 'Swap QNK/ZEC'}
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
                    <button onClick={() => copyAddress(zAddress)} className="px-3 py-1 bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 rounded text-xs whitespace-nowrap">
                      Copy
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
                  placeholder="zs1..."
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm font-mono focus:border-purple-500/50 focus:outline-none"
                />
                <p className="text-gray-500 text-xs mt-1">Only shielded z-addresses accepted (privacy enforced)</p>
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-1 block">Amount (ZEC)</label>
                <input
                  type="number"
                  step="0.00000001"
                  value={sendAmount}
                  onChange={(e) => setSendAmount(e.target.value)}
                  placeholder="0.00000000"
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-purple-500/50 focus:outline-none"
                />
                <p className="text-gray-500 text-xs mt-1">Available: {balanceZec.toFixed(8)} ZEC</p>
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

          {/* Receive / Swap Tab */}
          {activeTab === 'receive' && (
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

              <div>
                <label className="text-gray-400 text-sm mb-1 block">ZEC Amount</label>
                <input
                  type="number"
                  step="0.00000001"
                  value={swapZecAmount}
                  onChange={(e) => setSwapZecAmount(e.target.value)}
                  placeholder="0.00000000"
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-purple-500/50 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-1 block">QNK Amount (QUG)</label>
                <input
                  type="text"
                  value={swapQnkAmount}
                  onChange={(e) => setSwapQnkAmount(e.target.value)}
                  placeholder="0.00"
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-purple-500/50 focus:outline-none"
                />
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
                      <span className="text-gray-500 text-xs">{new Date(swap.created_at).toLocaleDateString()}</span>
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
