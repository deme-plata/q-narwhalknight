import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Loader, Zap, Gift, Hash, Blocks, User, X, Copy, Check, CheckCircle, Shield, Play } from 'lucide-react';
import { qnkAPI, type MiningRewardEvent } from '../services/api';
import { TICKER_SYMBOL } from '../constants/ticker';
import NetworkSelector from './NetworkSelector';

interface GlobalTopBarProps {
  authenticated?: boolean;
}

interface SearchResult {
  type: 'transaction' | 'block' | 'address' | 'hint';
  id: string;
  title: string;
  subtitle?: string;
  hash?: string;
  data?: any;
}

export default function GlobalTopBar({ authenticated = false }: GlobalTopBarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [selectedDetail, setSelectedDetail] = useState<SearchResult | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [miningHashRate, setMiningHashRate] = useState(0);
  const [isMining, setIsMining] = useState(false);
  const [showVideoModal, setShowVideoModal] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const walletAddress = localStorage.getItem('walletAddress') || '';

  // SSE for mining hash rate updates
  useEffect(() => {
    if (!walletAddress || !authenticated) return;

    const eventSource = qnkAPI.subscribeToMiningRewards(
      walletAddress,
      (reward: MiningRewardEvent) => {
        if (reward.hash_rate > 0) {
          setMiningHashRate(reward.hash_rate);
          setIsMining(true);
        }
      },
      () => {}
    );

    eventSourceRef.current = eventSource;

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [walletAddress, authenticated]);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const performSearch = async (query: string) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    const results: SearchResult[] = [];

    try {
      // Auto-detect search type
      let searchType = '';
      if (query.match(/^tx_[a-f0-9]+/i) || query.match(/^[a-f0-9]{64}$/i)) searchType = 'transaction';
      else if (query.match(/^qnk[a-z0-9]{39}$/i)) searchType = 'address';
      else if (query.match(/^\d+$/)) searchType = 'block';

      if (searchType === 'block') {
        const blockNum = parseInt(query);
        const blockResponse = await qnkAPI.getBlock(blockNum);
        if (blockResponse.success && blockResponse.data) {
          results.push({
            type: 'block',
            id: blockNum.toString(),
            title: `Block #${blockNum}`,
            subtitle: `${Array.isArray(blockResponse.data) ? blockResponse.data.length : 0} transactions`,
            data: {
              height: blockNum,
              tx_count: Array.isArray(blockResponse.data) ? blockResponse.data.length : 0,
              hash: blockResponse.data[0]?.hash || 'N/A',
              transactions: blockResponse.data
            }
          });
        } else {
          results.push({
            type: 'block',
            id: 'not-found',
            title: 'Block Not Found',
            subtitle: `Block #${blockNum} does not exist yet`
          });
        }
      } else if (searchType === 'transaction') {
        const txResponse = await qnkAPI.getTransactionByHash(query);
        if (txResponse.success && txResponse.data) {
          const txData = txResponse.data;
          const amount = txData.amount ? (Number(txData.amount) / 1e24) : 0;
          results.push({
            type: 'transaction',
            id: query,
            title: 'Transaction Found',
            subtitle: txData.status || 'confirmed',
            hash: txData.hash || query,
            data: {
              hash: txData.hash || query,
              amount,
              status: txData.status || 'confirmed',
              timestamp: txData.timestamp ? new Date(txData.timestamp * 1000).toLocaleString() : 'N/A',
              from: txData.from || 'N/A',
              to: txData.to || 'N/A',
              block_height: txData.block_height,
              confirmations: txData.confirmations,
              fee: txData.fee ? (Number(txData.fee) / 1e24) : 0,
              token_type: txData.token_type
            }
          });
        } else {
          results.push({
            type: 'transaction',
            id: 'not-found',
            title: 'Transaction Not Found',
            subtitle: `${query.substring(0, 16)}...${query.length > 48 ? query.substring(48) : ''}`
          });
        }
      } else if (searchType === 'address') {
        const balanceResponse = await qnkAPI.getWalletBalance(query);
        if (balanceResponse.success && balanceResponse.data) {
          results.push({
            type: 'address',
            id: query,
            title: 'Wallet Address',
            subtitle: `Balance: ${(balanceResponse.data.balance_qnk || 0).toFixed(4)} ${TICKER_SYMBOL}`,
            hash: query,
            data: {
              address: query,
              balance: balanceResponse.data.balance_qnk || 0,
              nonce: balanceResponse.data.nonce || 0
            }
          });
        } else {
          results.push({
            type: 'address',
            id: query,
            title: 'New Wallet Address',
            subtitle: `Balance: 0 ${TICKER_SYMBOL}`,
            hash: query,
            data: { address: query, balance: 0, nonce: 0 }
          });
        }
      } else if (query.length >= 2) {
        // Show helpful search hints
        results.push({
          type: 'hint',
          id: 'hint-block',
          title: 'Search by block number',
          subtitle: 'Enter a number (e.g. 12345)'
        });
        results.push({
          type: 'hint',
          id: 'hint-tx',
          title: 'Search by transaction hash',
          subtitle: 'Enter a 64-character hex hash'
        });
        results.push({
          type: 'hint',
          id: 'hint-address',
          title: 'Search by wallet address',
          subtitle: 'Enter address starting with "qnk"'
        });
      }
    } catch (error) {
      console.error('Search failed:', error);
      results.push({
        type: 'hint',
        id: 'error',
        title: 'Search Failed',
        subtitle: 'Unable to connect to the network'
      });
    }

    setSearchResults(results);
    setIsSearching(false);
  };

  // Debounced search
  const handleInputChange = (value: string) => {
    setSearchQuery(value);
    if (value.trim().length > 0) {
      setShowResults(true);
    } else {
      setShowResults(false);
      setSearchResults([]);
    }

    if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);
    searchTimeoutRef.current = setTimeout(() => {
      performSearch(value.trim());
    }, 250);
  };

  const getResultIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'transaction': return <Hash className="w-4 h-4" />;
      case 'block': return <Blocks className="w-4 h-4" />;
      case 'address': return <User className="w-4 h-4" />;
      default: return <Search className="w-4 h-4" />;
    }
  };

  const getResultColor = (type: SearchResult['type']) => {
    switch (type) {
      case 'transaction': return 'text-pink-400';
      case 'block': return 'text-cyan-400';
      case 'address': return 'text-green-400';
      default: return 'text-amber-400';
    }
  };

  const formatHashRate = (hashRate: number) => {
    if (hashRate >= 1e9) return `${(hashRate / 1e9).toFixed(2)} GH/s`;
    if (hashRate >= 1e6) return `${(hashRate / 1e6).toFixed(2)} MH/s`;
    if (hashRate >= 1e3) return `${(hashRate / 1e3).toFixed(2)} KH/s`;
    return `${hashRate.toFixed(2)} H/s`;
  };

  return (
    <>
      <div className="bg-quantum-dark/80 backdrop-blur-xl border-b border-quantum-purple/20 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center">
              <motion.div
                className="flex items-center gap-3"
                whileHover={{ scale: 1.05 }}
              >
                <div className="relative w-10 h-10">
                  <div className="absolute inset-0 bg-gradient-to-b from-amber-500/20 via-orange-500/20 to-yellow-500/20 rounded-full blur-lg animate-pulse" />
                  <div className="absolute inset-0 rounded-full" style={{
                    background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                    padding: '2px'
                  }}>
                    <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center p-1">
                      <img
                        src="/quillon-logo.png"
                        alt="Quillon Graph Logo"
                        className="w-full h-full object-contain"
                        style={{ filter: 'invert(1)' }}
                      />
                    </div>
                  </div>
                </div>
                <span className="text-lg font-bold bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent">Quillon Graph</span>
              </motion.div>
            </div>

            {/* Search Bar */}
            <div className="flex-1 max-w-2xl mx-8 relative">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-amber-400/60" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => handleInputChange(e.target.value)}
                  onFocus={() => {
                    if (searchQuery.length >= 2) setShowResults(true);
                  }}
                  onBlur={() => setTimeout(() => setShowResults(false), 200)}
                  placeholder="Search tx hash, block height, or wallet address..."
                  className="w-full pl-10 pr-4 py-2 bg-slate-900/50 border border-amber-500/20 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-amber-400/60 focus:shadow-[0_0_12px_rgba(251,191,36,0.15)] transition-all"
                />
                {isSearching && (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2"
                  >
                    <Search className="w-4 h-4 text-amber-400" />
                  </motion.div>
                )}
              </div>

              {/* Search Results Dropdown */}
              <AnimatePresence>
                {showResults && (searchResults.length > 0 || isSearching || searchQuery.length >= 2) && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="absolute top-full left-0 right-0 mt-2 backdrop-blur-xl rounded-xl shadow-2xl max-h-96 overflow-y-auto z-50"
                    style={{
                      background: 'linear-gradient(135deg, rgba(15, 10, 40, 0.98) 0%, rgba(30, 20, 60, 0.98) 100%)',
                      border: '1px solid rgba(212, 175, 55, 0.25)',
                      boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5), 0 0 20px rgba(212, 175, 55, 0.1)'
                    }}
                  >
                    {searchResults.length > 0 ? (
                      searchResults.map((result, index) => (
                        <motion.div
                          key={`${result.type}-${result.id}-${index}`}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.04 }}
                          className="flex items-center gap-3 p-3 border-b last:border-b-0 cursor-pointer transition-colors hover:bg-white/5"
                          style={{ borderColor: 'rgba(212, 175, 55, 0.08)' }}
                          onClick={() => {
                            if (result.data) {
                              setSelectedDetail(result);
                            }
                            setShowResults(false);
                            setSearchQuery('');
                          }}
                        >
                          <div
                            className="p-1.5 rounded-lg flex-shrink-0"
                            style={{
                              background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.15), rgba(255, 215, 0, 0.1))',
                              border: '1px solid rgba(212, 175, 55, 0.2)'
                            }}
                          >
                            <div className={getResultColor(result.type)}>{getResultIcon(result.type)}</div>
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="text-amber-100/90 font-medium text-sm truncate">{result.title}</div>
                            {result.subtitle && (
                              <div className="text-amber-300/40 text-xs truncate">{result.subtitle}</div>
                            )}
                          </div>
                        </motion.div>
                      ))
                    ) : isSearching ? (
                      <div className="p-4 text-center text-gray-400">
                        <Loader className="w-5 h-5 animate-spin mx-auto mb-2 text-amber-400/60" />
                        <span className="text-sm">Searching quantum ledger...</span>
                      </div>
                    ) : searchQuery.length >= 2 ? (
                      <div className="p-4 text-center text-gray-500 text-sm">
                        No results found for "{searchQuery}"
                      </div>
                    ) : null}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Mining Hash Rate Indicator & Status */}
            <div className="flex items-center gap-4">
              <NetworkSelector />

              <motion.button
                onClick={() => window.dispatchEvent(new CustomEvent('open-bounty-modal'))}
                whileHover={{ scale: 1.05 }}
                className="flex items-center gap-2 bg-gradient-to-r from-amber-500/20 to-yellow-500/20 border border-amber-500/40 rounded-lg px-3 py-1.5 cursor-pointer transition-all hover:from-amber-500/30 hover:to-yellow-500/30"
              >
                <Gift className="w-4 h-4 text-amber-400" />
                <span className="text-amber-300 text-sm font-semibold">Bounty Campaign</span>
              </motion.button>

              <motion.button
                onClick={() => setShowVideoModal(true)}
                whileHover={{ scale: 1.05 }}
                className="flex items-center gap-2 bg-gradient-to-r from-red-500/20 to-pink-500/20 border border-red-500/40 rounded-lg px-3 py-1.5 cursor-pointer transition-all hover:from-red-500/30 hover:to-pink-500/30"
              >
                <Play className="w-4 h-4 text-red-400" />
                <span className="text-red-300 text-sm font-semibold">Video</span>
              </motion.button>

              {authenticated && isMining && miningHashRate > 0 && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-center gap-2 bg-quantum-yellow/10 border border-quantum-yellow/30 rounded-lg px-3 py-1.5"
                >
                  <Zap className="w-4 h-4 text-quantum-yellow animate-pulse" />
                  <span className="text-quantum-yellow text-sm font-bold">
                    {formatHashRate(miningHashRate)}
                  </span>
                  <span className="text-gray-400 text-xs">Mining</span>
                </motion.div>
              )}

              {authenticated ? (
                <div className="flex items-center gap-2 text-quantum-green text-sm">
                  <div className="w-2 h-2 bg-quantum-green rounded-full animate-pulse" />
                  Connected
                </div>
              ) : (
                <div className="flex items-center gap-2 text-gray-400 text-sm">
                  <div className="w-2 h-2 bg-gray-400 rounded-full" />
                  Explorer Mode
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Detail Modal (same as ExplorerSearchBar) */}
      <AnimatePresence>
        {selectedDetail && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-start justify-center z-[9999] overflow-y-auto py-8"
            onClick={() => setSelectedDetail(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-amber-500/30 rounded-2xl p-6 max-w-lg w-full mx-4 shadow-2xl my-auto"
              onClick={(e) => e.stopPropagation()}
              style={{ boxShadow: '0 0 40px rgba(212, 175, 55, 0.2)' }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-amber-500/20 rounded-lg">
                    {selectedDetail.type === 'transaction' && <Hash className="w-6 h-6 text-amber-400" />}
                    {selectedDetail.type === 'block' && <Blocks className="w-6 h-6 text-amber-400" />}
                    {selectedDetail.type === 'address' && <User className="w-6 h-6 text-amber-400" />}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-amber-100">{selectedDetail.title}</h3>
                    <p className="text-amber-300/60 text-sm">{selectedDetail.type.toUpperCase()}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedDetail(null)}
                  className="p-2 hover:bg-amber-500/20 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-amber-400" />
                </button>
              </div>

              {/* Transaction Details */}
              {selectedDetail.type === 'transaction' && selectedDetail.data && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2 p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span className="text-green-400 font-medium capitalize">{selectedDetail.data.status}</span>
                    {selectedDetail.data.confirmations && (
                      <span className="text-green-300/60 text-sm">({selectedDetail.data.confirmations} confirmations)</span>
                    )}
                  </div>

                  <div className="grid gap-3">
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Transaction Hash</div>
                      <div className="flex items-center gap-2">
                        <code className="text-amber-100 text-xs font-mono break-all flex-1">{selectedDetail.data.hash}</code>
                        <button
                          onClick={() => copyToClipboard(selectedDetail.data.hash, 'modal-hash')}
                          className="p-1.5 hover:bg-amber-500/20 rounded transition-colors flex-shrink-0"
                        >
                          {copiedId === 'modal-hash' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 bg-slate-800/50 rounded-lg">
                        <div className="text-amber-300/60 text-xs mb-1">Block</div>
                        <div className="text-amber-100 font-medium">#{selectedDetail.data.block_height || 'Pending'}</div>
                      </div>
                      <div className="p-3 bg-slate-800/50 rounded-lg">
                        <div className="text-amber-300/60 text-xs mb-1">Time</div>
                        <div className="text-amber-100 font-medium text-xs">{selectedDetail.data.timestamp}</div>
                      </div>
                    </div>

                    <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                      <div className="flex items-center gap-2 text-purple-300 text-sm">
                        <Shield className="w-4 h-4" />
                        <span>ZK-STARK Privacy: Transaction details are encrypted</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Block Details */}
              {selectedDetail.type === 'block' && selectedDetail.data && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Block Height</div>
                      <div className="text-xl font-bold text-amber-100">#{selectedDetail.data.height}</div>
                    </div>
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Transactions</div>
                      <div className="text-xl font-bold text-amber-100">{selectedDetail.data.tx_count}</div>
                    </div>
                  </div>

                  {selectedDetail.data.hash && selectedDetail.data.hash !== 'N/A' && (
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Block Hash</div>
                      <div className="flex items-center gap-2">
                        <code className="text-amber-100 text-xs font-mono break-all flex-1">{selectedDetail.data.hash}</code>
                        <button
                          onClick={() => copyToClipboard(selectedDetail.data.hash, 'modal-block-hash')}
                          className="p-1.5 hover:bg-amber-500/20 rounded transition-colors flex-shrink-0"
                        >
                          {copiedId === 'modal-block-hash' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Address Details */}
              {selectedDetail.type === 'address' && selectedDetail.data && (
                <div className="space-y-4">
                  <div className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-amber-300/60 text-xs mb-1">Address</div>
                    <div className="flex items-center gap-2">
                      <code className="text-amber-100 text-sm font-mono break-all flex-1">{selectedDetail.data.address}</code>
                      <button
                        onClick={() => copyToClipboard(selectedDetail.data.address, 'modal-address')}
                        className="p-1.5 hover:bg-amber-500/20 rounded transition-colors flex-shrink-0"
                      >
                        {copiedId === 'modal-address' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Balance</div>
                      <div className="text-xl font-bold text-amber-100">{selectedDetail.data.balance?.toFixed(4)} {TICKER_SYMBOL}</div>
                    </div>
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Nonce</div>
                      <div className="text-xl font-bold text-amber-100">{selectedDetail.data.nonce}</div>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Video Modal */}
      <AnimatePresence>
        {showVideoModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[200] flex items-center justify-center bg-black/80 backdrop-blur-sm"
            onClick={() => setShowVideoModal(false)}
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="relative w-[90vw] max-w-4xl aspect-video bg-black rounded-2xl overflow-hidden border border-red-500/30 shadow-2xl shadow-red-500/10"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={() => setShowVideoModal(false)}
                className="absolute top-3 right-3 z-10 w-8 h-8 flex items-center justify-center rounded-full bg-black/60 hover:bg-black/80 text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
              <iframe
                src="https://www.youtube.com/embed/EfDvMe0apTg?autoplay=1&rel=0"
                title="Quillon Video"
                className="w-full h-full"
                allow="autoplay; encrypted-media; picture-in-picture"
                allowFullScreen
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
