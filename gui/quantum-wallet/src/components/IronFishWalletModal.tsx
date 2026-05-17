import React, { useState, useEffect, useCallback } from 'react';
import { qnkAPI } from '../services/api';

interface IronFishWalletModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletAddress: string;
}

interface SwapEntry {
  swap_id: string;
  direction: string;
  iron_amount: number;
  qnk_amount: string;
  status: string;
  iron_address?: string;
  created_at: string;
}

type Tab = 'balance' | 'send' | 'swap' | 'history';

const IronFishWalletModal: React.FC<IronFishWalletModalProps> = ({ isOpen, onClose, walletAddress }) => {
  const [activeTab, setActiveTab] = useState<Tab>('balance');
  const [ironAddress, setIronAddress] = useState('');
  const [balanceIron, setBalanceIron] = useState(0);
  const [balanceOre, setBalanceOre] = useState(0);
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
  const [swapDirection, setSwapDirection] = useState<'buy_iron' | 'sell_iron'>('buy_iron');
  const [swapIronAmount, setSwapIronAmount] = useState('');
  const [swapQnkAmount, setSwapQnkAmount] = useState('');

  const fetchData = useCallback(async () => {
    try {
      const [balRes, addrRes, bridgeRes, swapsRes] = await Promise.allSettled([
        qnkAPI.getIronFishBalance(),
        qnkAPI.getIronFishAddress(),
        qnkAPI.getIronFishBridgeStatus(),
        qnkAPI.listIronFishSwaps(),
      ]);

      if (balRes.status === 'fulfilled' && balRes.value?.data) {
        setBalanceIron(balRes.value.data.balance_iron || 0);
        setBalanceOre(balRes.value.data.balance_ore || 0);
      }
      if (addrRes.status === 'fulfilled' && addrRes.value?.data) {
        setIronAddress(addrRes.value.data.iron_address || '');
      }
      if (bridgeRes.status === 'fulfilled' && bridgeRes.value?.data) {
        setBridgeStatus(bridgeRes.value.data);
      }
      if (swapsRes.status === 'fulfilled' && swapsRes.value?.data) {
        setSwapHistory(swapsRes.value.data.swaps || []);
      }
    } catch (e) {
      console.error('Failed to fetch Iron Fish data:', e);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchData();
    }
  }, [isOpen, fetchData]);

  const handleSend = async () => {
    if (!sendAddress || sendAddress.length < 32) {
      setError('Invalid Iron Fish address.');
      return;
    }
    const amountOre = Math.round(parseFloat(sendAmount) * 100_000_000);
    if (isNaN(amountOre) || amountOre <= 0) {
      setError('Invalid amount.');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const res = await qnkAPI.sendIronFish({
        to_address: sendAddress,
        amount_ore: amountOre,
        memo: sendMemo || undefined,
      });

      if (res.success && res.data) {
        setSuccess(`Transaction submitted: ${res.data.tx_id}`);
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
    const ironAmountOre = Math.round(parseFloat(swapIronAmount) * 100_000_000);
    if (isNaN(ironAmountOre) || ironAmountOre <= 0) {
      setError('Invalid IRON amount.');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const res = await qnkAPI.createIronFishSwap({
        direction: swapDirection,
        iron_amount: ironAmountOre,
        qnk_amount: swapQnkAmount || '0',
        iron_address: ironAddress || undefined,
      });

      if (res.success && res.data) {
        setSuccess(`Swap created: ${res.data.swap_id}`);
        setSwapIronAmount('');
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

  const formatIron = (ore: number) => (ore / 100_000_000).toFixed(8);
  const truncateAddr = (addr: string) => addr.length > 24 ? `${addr.slice(0, 12)}...${addr.slice(-12)}` : addr;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gradient-to-br from-gray-900 to-gray-800 border border-cyan-500/30 rounded-2xl w-full max-w-2xl max-h-[85vh] overflow-hidden shadow-2xl shadow-cyan-500/10">
        {/* Header */}
        <div className="bg-gradient-to-r from-cyan-600/20 to-slate-600/20 border-b border-cyan-500/20 p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-400 to-slate-500 flex items-center justify-center">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/>
                  <path d="M8 12l2.5 2.5L16 9" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Iron Fish Wallet</h2>
                <p className="text-xs text-cyan-300/70">Privacy-first zk-SNARK transactions</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {bridgeStatus && (
                <span className={`px-2 py-1 rounded text-xs font-medium ${bridgeStatus.node_syncing ? 'bg-yellow-500/20 text-yellow-300' : 'bg-green-500/20 text-green-300'}`}>
                  {bridgeStatus.node_syncing
                    ? `Syncing (${bridgeStatus.node_height?.toLocaleString()})`
                    : `Synced (${bridgeStatus.node_height?.toLocaleString()})`
                  }
                </span>
              )}
              {bridgeStatus?.peers > 0 && (
                <span className="px-2 py-1 rounded text-xs font-medium bg-cyan-500/20 text-cyan-300">
                  {bridgeStatus.peers} peers
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
          {(['balance', 'send', 'swap', 'history'] as Tab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => { setActiveTab(tab); setError(''); setSuccess(''); }}
              className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                activeTab === tab
                  ? 'text-cyan-300 border-b-2 border-cyan-500 bg-cyan-500/5'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {tab === 'balance' && 'Balance'}
              {tab === 'send' && 'Send IRON'}
              {tab === 'swap' && 'Swap QNK/IRON'}
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
              <div className="bg-gradient-to-br from-cyan-900/30 to-slate-900/30 rounded-xl p-6 border border-cyan-500/20">
                <p className="text-gray-400 text-sm mb-1">Shielded Balance</p>
                <p className="text-3xl font-bold text-white">{balanceIron.toFixed(8)} <span className="text-cyan-400 text-lg">IRON</span></p>
                <p className="text-gray-500 text-xs mt-1">{balanceOre.toLocaleString()} ore</p>
              </div>

              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/30">
                <p className="text-gray-400 text-xs mb-2">Your Iron Fish Address</p>
                {ironAddress ? (
                  <div className="flex items-center gap-2">
                    <code className="text-cyan-300 text-xs flex-1 break-all font-mono">{ironAddress}</code>
                    <button onClick={() => copyAddress(ironAddress)} className="px-3 py-1 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 rounded text-xs whitespace-nowrap">
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
                  <p className="text-white text-sm font-medium">Iron Fish Mainnet</p>
                </div>
                <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/20">
                  <p className="text-gray-500 text-xs">Consensus</p>
                  <p className="text-cyan-300 text-sm font-medium">PoW (FishHash)</p>
                </div>
                <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/20">
                  <p className="text-gray-500 text-xs">Node Height</p>
                  <p className="text-white text-sm font-medium">{bridgeStatus?.node_height?.toLocaleString() || '...'} blocks</p>
                </div>
                <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/20">
                  <p className="text-gray-500 text-xs">Privacy</p>
                  <p className="text-green-300 text-sm font-medium">zk-SNARK (Default Shielded)</p>
                </div>
                {bridgeStatus?.node_version && bridgeStatus.node_version !== 'offline' && (
                  <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/20 col-span-2">
                    <p className="text-gray-500 text-xs">Node Version</p>
                    <p className="text-white text-sm font-medium">Iron Fish v{bridgeStatus.node_version}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Send Tab */}
          {activeTab === 'send' && (
            <div className="space-y-4">
              <div>
                <label className="text-gray-400 text-sm mb-1 block">Destination Address</label>
                <input
                  type="text"
                  value={sendAddress}
                  onChange={(e) => setSendAddress(e.target.value)}
                  placeholder="Iron Fish public address..."
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm font-mono focus:border-cyan-500/50 focus:outline-none"
                />
                <p className="text-gray-500 text-xs mt-1">All Iron Fish transactions are shielded by default</p>
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-1 block">Amount (IRON)</label>
                <input
                  type="number"
                  step="0.00000001"
                  value={sendAmount}
                  onChange={(e) => setSendAmount(e.target.value)}
                  placeholder="0.00000000"
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-cyan-500/50 focus:outline-none"
                />
                <p className="text-gray-500 text-xs mt-1">Available: {balanceIron.toFixed(8)} IRON</p>
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-1 block">Memo (optional)</label>
                <textarea
                  value={sendMemo}
                  onChange={(e) => setSendMemo(e.target.value)}
                  placeholder="Private message..."
                  maxLength={512}
                  rows={2}
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-cyan-500/50 focus:outline-none resize-none"
                />
              </div>

              <button
                onClick={handleSend}
                disabled={loading || !sendAddress || !sendAmount}
                className="w-full py-3 bg-gradient-to-r from-cyan-600 to-slate-600 hover:from-cyan-500 hover:to-slate-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-all"
              >
                {loading ? 'Sending...' : 'Send IRON'}
              </button>

              <div className="bg-cyan-500/5 border border-cyan-500/10 rounded-lg p-3">
                <p className="text-cyan-300/60 text-xs">
                  Iron Fish uses zk-SNARKs to shield all transaction details. Amounts, sender, and recipient are fully encrypted on the blockchain.
                </p>
              </div>
            </div>
          )}

          {/* Swap Tab */}
          {activeTab === 'swap' && (
            <div className="space-y-4">
              <div className="flex gap-2 bg-gray-800/30 rounded-lg p-1">
                <button
                  onClick={() => setSwapDirection('buy_iron')}
                  className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                    swapDirection === 'buy_iron' ? 'bg-cyan-600 text-white' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  QUG to IRON
                </button>
                <button
                  onClick={() => setSwapDirection('sell_iron')}
                  className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                    swapDirection === 'sell_iron' ? 'bg-cyan-600 text-white' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  IRON to QUG
                </button>
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-1 block">IRON Amount</label>
                <input
                  type="number"
                  step="0.00000001"
                  value={swapIronAmount}
                  onChange={(e) => setSwapIronAmount(e.target.value)}
                  placeholder="0.00000000"
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-cyan-500/50 focus:outline-none"
                />
              </div>

              <div>
                <label className="text-gray-400 text-sm mb-1 block">QUG Amount</label>
                <input
                  type="text"
                  value={swapQnkAmount}
                  onChange={(e) => setSwapQnkAmount(e.target.value)}
                  placeholder="0.00"
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 text-sm focus:border-cyan-500/50 focus:outline-none"
                />
              </div>

              <button
                onClick={handleSwap}
                disabled={loading || !swapIronAmount}
                className="w-full py-3 bg-gradient-to-r from-cyan-600 to-slate-600 hover:from-cyan-500 hover:to-slate-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-all"
              >
                {loading ? 'Creating Swap...' : `Create ${swapDirection === 'buy_iron' ? 'QUG -> IRON' : 'IRON -> QUG'} Swap`}
              </button>

              <div className="bg-cyan-500/5 border border-cyan-500/10 rounded-lg p-3 space-y-2">
                <p className="text-cyan-300/80 text-xs font-medium">How Privacy Atomic Swaps Work:</p>
                <ol className="text-cyan-300/60 text-xs space-y-1 list-decimal list-inside">
                  <li>A cryptographic hash-lock is generated</li>
                  <li>QUG is locked in an on-chain HTLC escrow</li>
                  <li>IRON is sent to your address via shielded transfer</li>
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
                  <p className="text-sm">Your Iron Fish atomic swaps will appear here.</p>
                </div>
              ) : (
                swapHistory.map((swap) => (
                  <div key={swap.swap_id} className="bg-gray-800/30 rounded-lg p-4 border border-gray-700/20">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className={`text-sm font-medium ${swap.direction === 'buy_iron' ? 'text-cyan-300' : 'text-green-300'}`}>
                          {swap.direction === 'buy_iron' ? 'QUG -> IRON' : 'IRON -> QUG'}
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
                        <span className="text-gray-500">IRON:</span>{' '}
                        <span className="text-white">{formatIron(swap.iron_amount)}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">QUG:</span>{' '}
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

export default IronFishWalletModal;
