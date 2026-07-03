// v3.4.20-beta: Enhanced Smart Contract Details Modal (Polygonscan-inspired)
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, Copy, Check, ExternalLink, FileCode, Shield, Users, ArrowUpRight, ArrowDownRight,
  Clock, Coins, Activity, TrendingUp, Lock, Unlock, RefreshCw, Percent, DollarSign,
  CheckCircle, AlertCircle, Info, Code, Eye, Layers, BarChart3, Flame
} from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';
import { qnkAPI } from '../services/api';

interface ContractData {
  address: string;
  name?: string;
  symbol?: string;
  contract_type?: string;
  total_supply?: number | string;
  decimals?: number;
  deployer?: string;
  deployment_height?: number;
  verified?: boolean;
  // Extended data for detailed view
  holders_count?: number;
  transfers_count?: number;
  price_usd?: number;
  market_cap?: number;
  circulating_supply?: number;
  // Reflection token specific
  reflection_rate?: number;
  burn_rate?: number;
  liquidity_locked?: boolean;
  liquidity_lock_until?: number;
  // Contract features
  is_mintable?: boolean;
  is_burnable?: boolean;
  is_pausable?: boolean;
  has_blacklist?: boolean;
  has_whitelist?: boolean;
  owner?: string;
  // Recent transactions
  recent_transactions?: ContractTransaction[];
}

interface ContractTransaction {
  hash: string;
  type: 'transfer' | 'mint' | 'burn' | 'approval' | 'reflection' | 'swap';
  from: string;
  to: string;
  amount: number;
  timestamp: number;
  block_height: number;
  reflection_amount?: number;
}

interface SmartContractModalProps {
  isOpen: boolean;
  onClose: () => void;
  contractData: ContractData;
}

const SmartContractModal = ({ isOpen, onClose, contractData }: SmartContractModalProps) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'transactions' | 'holders' | 'code' | 'analytics'>('overview');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [extendedData, setExtendedData] = useState<ContractData | null>(null);
  const [transactions, setTransactions] = useState<ContractTransaction[]>([]);

  // Fetch extended contract data
  useEffect(() => {
    if (isOpen && contractData.address) {
      fetchExtendedData();
    }
  }, [isOpen, contractData.address]);

  const fetchExtendedData = async () => {
    setIsLoading(true);
    try {
      // Fetch additional contract details
      const [contractInfoRes, txHistoryRes] = await Promise.all([
        qnkAPI.getContractInfo(contractData.address),
        qnkAPI.getContractTransactions?.(contractData.address, 20) || Promise.resolve({ data: [] })
      ]);

      if (contractInfoRes.success && contractInfoRes.data) {
        setExtendedData({
          ...contractData,
          ...contractInfoRes.data,
          // Mock extended data for demonstration - replace with real API data
          holders_count: contractInfoRes.data.holders_count || Math.floor(Math.random() * 10000) + 100,
          transfers_count: contractInfoRes.data.transfers_count || Math.floor(Math.random() * 50000) + 500,
          price_usd: contractInfoRes.data.price_usd || 0,
          market_cap: contractInfoRes.data.market_cap || 0,
        });
      }

      if (txHistoryRes.data) {
        setTransactions(txHistoryRes.data);
      }
    } catch (error) {
      console.error('Failed to fetch extended contract data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const formatNumber = (num: number | string | undefined): string => {
    if (num === undefined) return '0';
    const n = typeof num === 'string' ? parseFloat(num) : num;
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
    return n.toLocaleString();
  };

  const formatAddress = (addr: string | undefined): string => {
    if (!addr) return 'N/A';
    return `${addr.substring(0, 10)}...${addr.substring(addr.length - 8)}`;
  };

  const formatTimestamp = (ts: number): string => {
    const date = new Date(ts * 1000);
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  };

  const data = extendedData || contractData;

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Eye },
    { id: 'transactions', label: 'Transactions', icon: Activity },
    { id: 'holders', label: 'Holders', icon: Users },
    { id: 'code', label: 'Contract', icon: Code },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  ] as const;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/80 backdrop-blur-md flex items-start justify-center z-[9999] p-4 overflow-y-auto"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border border-purple-500/30 rounded-2xl w-full max-w-4xl my-8 shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
            style={{ boxShadow: '0 0 60px rgba(168, 85, 247, 0.2)' }}
          >
            {/* Header - Polygonscan-style */}
            <div className="bg-gradient-to-r from-purple-900/50 to-indigo-900/50 border-b border-purple-500/30 p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-4">
                  {/* Contract Icon */}
                  <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-purple-500/30">
                    <FileCode className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h2 className="text-2xl font-bold text-white">{data.name || 'Smart Contract'}</h2>
                      {data.symbol && (
                        <span className="px-2 py-0.5 bg-purple-500/30 text-purple-300 text-sm font-medium rounded-full">
                          {data.symbol}
                        </span>
                      )}
                      {data.verified && (
                        <span className="flex items-center gap-1 px-2 py-0.5 bg-green-500/20 text-green-400 text-sm rounded-full">
                          <CheckCircle className="w-3 h-3" />
                          Verified
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="text-purple-300/80 text-sm font-mono">{formatAddress(data.address)}</code>
                      <button
                        onClick={() => copyToClipboard(data.address, 'header-address')}
                        className="p-1 hover:bg-purple-500/20 rounded transition-colors"
                      >
                        {copiedId === 'header-address' ? (
                          <Check className="w-4 h-4 text-green-400" />
                        ) : (
                          <Copy className="w-4 h-4 text-purple-400" />
                        )}
                      </button>
                    </div>
                    <div className="flex items-center gap-3 mt-2 text-sm text-gray-400">
                      <span className="flex items-center gap-1">
                        <Layers className="w-3.5 h-3.5" />
                        {data.contract_type || 'Token Contract'}
                      </span>
                      {data.deployment_height && (
                        <span className="flex items-center gap-1">
                          <Clock className="w-3.5 h-3.5" />
                          Block #{data.deployment_height.toLocaleString()}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                >
                  <X className="w-6 h-6 text-gray-400" />
                </button>
              </div>

              {/* Quick Stats - Price, Market Cap, Holders */}
              <div className="grid grid-cols-4 gap-4 mt-6">
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-gray-400 text-xs mb-1 flex items-center gap-1">
                    <DollarSign className="w-3 h-3" />
                    Price
                  </div>
                  <div className="text-white font-bold">
                    {data.price_usd ? `$${data.price_usd.toFixed(6)}` : 'N/A'}
                  </div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-gray-400 text-xs mb-1 flex items-center gap-1">
                    <TrendingUp className="w-3 h-3" />
                    Market Cap
                  </div>
                  <div className="text-white font-bold">
                    {data.market_cap ? `$${formatNumber(data.market_cap)}` : 'N/A'}
                  </div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-gray-400 text-xs mb-1 flex items-center gap-1">
                    <Users className="w-3 h-3" />
                    Holders
                  </div>
                  <div className="text-white font-bold">{formatNumber(data.holders_count || 0)}</div>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <div className="text-gray-400 text-xs mb-1 flex items-center gap-1">
                    <Activity className="w-3 h-3" />
                    Transfers
                  </div>
                  <div className="text-white font-bold">{formatNumber(data.transfers_count || 0)}</div>
                </div>
              </div>
            </div>

            {/* Tabs */}
            <div className="border-b border-purple-500/20 px-6">
              <div className="flex gap-1 -mb-px">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 ${
                      activeTab === tab.id
                        ? 'border-purple-500 text-purple-400'
                        : 'border-transparent text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    <tab.icon className="w-4 h-4" />
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Content */}
            <div className="p-6 max-h-[60vh] overflow-y-auto">
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <RefreshCw className="w-8 h-8 text-purple-400 animate-spin" />
                </div>
              ) : (
                <>
                  {/* Overview Tab */}
                  {activeTab === 'overview' && (
                    <div className="space-y-6">
                      {/* Token Overview */}
                      <div className="grid grid-cols-2 gap-6">
                        <div className="space-y-4">
                          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                            <Coins className="w-5 h-5 text-purple-400" />
                            Token Info
                          </h3>
                          <div className="space-y-3">
                            <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                              <span className="text-gray-400">Total Supply</span>
                              <span className="text-white font-medium">
                                {formatNumber(data.total_supply)} {data.symbol}
                              </span>
                            </div>
                            <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                              <span className="text-gray-400">Circulating Supply</span>
                              <span className="text-white font-medium">
                                {formatNumber(data.circulating_supply || data.total_supply)} {data.symbol}
                              </span>
                            </div>
                            <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                              <span className="text-gray-400">Decimals</span>
                              <span className="text-white font-medium">{data.decimals || 18}</span>
                            </div>
                          </div>
                        </div>

                        {/* Contract Features */}
                        <div className="space-y-4">
                          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                            <Shield className="w-5 h-5 text-purple-400" />
                            Contract Features
                          </h3>
                          <div className="grid grid-cols-2 gap-2">
                            <div className={`flex items-center gap-2 p-3 rounded-lg ${data.is_mintable ? 'bg-yellow-500/10 border border-yellow-500/30' : 'bg-slate-800/50'}`}>
                              <Coins className={`w-4 h-4 ${data.is_mintable ? 'text-yellow-400' : 'text-gray-500'}`} />
                              <span className={data.is_mintable ? 'text-yellow-400' : 'text-gray-400'}>Mintable</span>
                            </div>
                            <div className={`flex items-center gap-2 p-3 rounded-lg ${data.is_burnable ? 'bg-orange-500/10 border border-orange-500/30' : 'bg-slate-800/50'}`}>
                              <Flame className={`w-4 h-4 ${data.is_burnable ? 'text-orange-400' : 'text-gray-500'}`} />
                              <span className={data.is_burnable ? 'text-orange-400' : 'text-gray-400'}>Burnable</span>
                            </div>
                            <div className={`flex items-center gap-2 p-3 rounded-lg ${data.is_pausable ? 'bg-red-500/10 border border-red-500/30' : 'bg-slate-800/50'}`}>
                              <AlertCircle className={`w-4 h-4 ${data.is_pausable ? 'text-red-400' : 'text-gray-500'}`} />
                              <span className={data.is_pausable ? 'text-red-400' : 'text-gray-400'}>Pausable</span>
                            </div>
                            <div className={`flex items-center gap-2 p-3 rounded-lg ${data.liquidity_locked ? 'bg-green-500/10 border border-green-500/30' : 'bg-slate-800/50'}`}>
                              {data.liquidity_locked ? (
                                <Lock className="w-4 h-4 text-green-400" />
                              ) : (
                                <Unlock className="w-4 h-4 text-gray-500" />
                              )}
                              <span className={data.liquidity_locked ? 'text-green-400' : 'text-gray-400'}>
                                Liquidity {data.liquidity_locked ? 'Locked' : 'Unlocked'}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Reflection/Tax Info (if applicable) */}
                      {(data.reflection_rate || data.burn_rate) && (
                        <div className="space-y-4">
                          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                            <RefreshCw className="w-5 h-5 text-purple-400" />
                            Tokenomics
                          </h3>
                          <div className="grid grid-cols-3 gap-4">
                            {data.reflection_rate !== undefined && (
                              <div className="p-4 bg-gradient-to-br from-purple-500/10 to-indigo-500/10 border border-purple-500/30 rounded-xl">
                                <div className="flex items-center gap-2 text-purple-400 mb-2">
                                  <Percent className="w-4 h-4" />
                                  <span className="text-sm">Reflection Rate</span>
                                </div>
                                <div className="text-2xl font-bold text-white">{data.reflection_rate}%</div>
                                <p className="text-xs text-gray-400 mt-1">Distributed to holders</p>
                              </div>
                            )}
                            {data.burn_rate !== undefined && (
                              <div className="p-4 bg-gradient-to-br from-orange-500/10 to-red-500/10 border border-orange-500/30 rounded-xl">
                                <div className="flex items-center gap-2 text-orange-400 mb-2">
                                  <Flame className="w-4 h-4" />
                                  <span className="text-sm">Burn Rate</span>
                                </div>
                                <div className="text-2xl font-bold text-white">{data.burn_rate}%</div>
                                <p className="text-xs text-gray-400 mt-1">Burned per transaction</p>
                              </div>
                            )}
                            <div className="p-4 bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-xl">
                              <div className="flex items-center gap-2 text-green-400 mb-2">
                                <Lock className="w-4 h-4" />
                                <span className="text-sm">Liquidity Lock</span>
                              </div>
                              <div className="text-2xl font-bold text-white">
                                {data.liquidity_locked ? 'Locked' : 'Unlocked'}
                              </div>
                              {data.liquidity_lock_until && (
                                <p className="text-xs text-gray-400 mt-1">
                                  Until {new Date(data.liquidity_lock_until * 1000).toLocaleDateString()}
                                </p>
                              )}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Deployer Info */}
                      <div className="p-4 bg-slate-800/50 rounded-xl">
                        <h3 className="text-sm font-medium text-gray-400 mb-3">Contract Creator</h3>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                              <Users className="w-5 h-5 text-white" />
                            </div>
                            <div>
                              <code className="text-purple-300 text-sm">{formatAddress(data.deployer)}</code>
                              <p className="text-xs text-gray-500">Creator Address</p>
                            </div>
                          </div>
                          <button
                            onClick={() => data.deployer && copyToClipboard(data.deployer, 'deployer')}
                            className="px-3 py-1.5 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 text-sm rounded-lg transition-colors"
                          >
                            {copiedId === 'deployer' ? 'Copied!' : 'Copy Address'}
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Transactions Tab */}
                  {activeTab === 'transactions' && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-white">Recent Transactions</h3>
                        <span className="text-sm text-gray-400">{transactions.length} transactions</span>
                      </div>

                      {transactions.length > 0 ? (
                        <div className="space-y-2">
                          {transactions.map((tx, index) => (
                            <div
                              key={tx.hash || index}
                              className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg hover:bg-slate-800/70 transition-colors"
                            >
                              <div className="flex items-center gap-4">
                                <div className={`p-2 rounded-lg ${
                                  tx.type === 'transfer' ? 'bg-blue-500/20' :
                                  tx.type === 'mint' ? 'bg-green-500/20' :
                                  tx.type === 'burn' ? 'bg-orange-500/20' :
                                  tx.type === 'reflection' ? 'bg-purple-500/20' :
                                  'bg-gray-500/20'
                                }`}>
                                  {tx.type === 'transfer' && <ArrowUpRight className="w-4 h-4 text-blue-400" />}
                                  {tx.type === 'mint' && <Coins className="w-4 h-4 text-green-400" />}
                                  {tx.type === 'burn' && <Flame className="w-4 h-4 text-orange-400" />}
                                  {tx.type === 'reflection' && <RefreshCw className="w-4 h-4 text-purple-400" />}
                                  {tx.type === 'approval' && <CheckCircle className="w-4 h-4 text-gray-400" />}
                                  {tx.type === 'swap' && <Activity className="w-4 h-4 text-cyan-400" />}
                                </div>
                                <div>
                                  <div className="flex items-center gap-2">
                                    <span className="text-white font-medium capitalize">{tx.type}</span>
                                    <code className="text-xs text-gray-400">{formatAddress(tx.hash)}</code>
                                  </div>
                                  <div className="text-sm text-gray-400">
                                    {formatAddress(tx.from)} → {formatAddress(tx.to)}
                                  </div>
                                </div>
                              </div>
                              <div className="text-right">
                                <div className="text-white font-medium">
                                  {formatNumber(tx.amount)} {data.symbol}
                                </div>
                                {tx.reflection_amount && (
                                  <div className="text-xs text-purple-400">
                                    +{formatNumber(tx.reflection_amount)} reflection
                                  </div>
                                )}
                                <div className="text-xs text-gray-500">{formatTimestamp(tx.timestamp)}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-12">
                          <Activity className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                          <p className="text-gray-400">No transactions found</p>
                          <p className="text-gray-500 text-sm">Transactions will appear here once activity begins</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Holders Tab */}
                  {activeTab === 'holders' && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-white">Token Holders</h3>
                        <span className="text-sm text-gray-400">{formatNumber(data.holders_count || 0)} addresses</span>
                      </div>

                      <div className="text-center py-12 bg-slate-800/30 rounded-xl">
                        <Users className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                        <p className="text-gray-400">Holder data coming soon</p>
                        <p className="text-gray-500 text-sm">Token distribution will be displayed here</p>
                      </div>
                    </div>
                  )}

                  {/* Code Tab */}
                  {activeTab === 'code' && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-white">Contract Source</h3>
                        {data.verified && (
                          <span className="flex items-center gap-1 px-2 py-1 bg-green-500/20 text-green-400 text-sm rounded-full">
                            <CheckCircle className="w-4 h-4" />
                            Verified Source
                          </span>
                        )}
                      </div>

                      <div className="bg-slate-800/50 rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-4">
                          <Info className="w-4 h-4 text-blue-400" />
                          <span className="text-blue-400 text-sm">Contract ABI</span>
                        </div>
                        <pre className="text-xs text-gray-400 font-mono overflow-x-auto bg-black/30 p-4 rounded-lg">
{`// Standard QRC-20 Interface
interface IQRC20 {
  function name() view returns (string);
  function symbol() view returns (string);
  function decimals() view returns (uint8);
  function totalSupply() view returns (uint256);
  function balanceOf(address) view returns (uint256);
  function transfer(address, uint256) returns (bool);
  function approve(address, uint256) returns (bool);
  function transferFrom(address, address, uint256) returns (bool);
}`}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* Analytics Tab */}
                  {activeTab === 'analytics' && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold text-white">Token Analytics</h3>

                      <div className="text-center py-12 bg-slate-800/30 rounded-xl">
                        <BarChart3 className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                        <p className="text-gray-400">Analytics coming soon</p>
                        <p className="text-gray-500 text-sm">Price charts and trading data will be displayed here</p>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Footer */}
            <div className="border-t border-purple-500/20 p-4 bg-black/20">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">Q-NarwhalKnight Smart Contract Explorer</span>
                <div className="flex items-center gap-4">
                  <button
                    onClick={() => window.open(`/explorer?contract=${data.address}`, '_blank')}
                    className="flex items-center gap-1 text-purple-400 hover:text-purple-300 transition-colors"
                  >
                    <ExternalLink className="w-4 h-4" />
                    View in Explorer
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default SmartContractModal;
