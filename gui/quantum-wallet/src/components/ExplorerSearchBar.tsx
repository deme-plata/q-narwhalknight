import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Hash, Blocks, User, X, Copy, Check, CheckCircle, Shield } from 'lucide-react';
import { qnkAPI } from '../services/api';
import { TICKER_SYMBOL } from '../constants/ticker';

interface SearchResult {
  type: 'transaction' | 'block' | 'address' | 'error';
  id: string;
  title: string;
  subtitle?: string;
  hash?: string;
  data?: any;
}

// Standalone explorer search bar for login page (no auth required)
export default function ExplorerSearchBar() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedDetail, setSelectedDetail] = useState<SearchResult | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

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
      // Determine search type based on query format
      // Handle mining-{height}-{nonce} IDs from the activity feed — redirect to block
      const miningHeightMatch = query.match(/^mining-(\d+)/i);
      if (miningHeightMatch) {
        const blockHeight = parseInt(miningHeightMatch[1]);
        const blockResponse = await qnkAPI.getBlock(blockHeight);
        if (blockResponse.success && blockResponse.data) {
          results.push({
            type: 'block',
            id: blockHeight.toString(),
            title: `Block #${blockHeight} (Mining Reward)`,
            subtitle: `${Array.isArray(blockResponse.data) ? blockResponse.data.length : 0} transactions`,
            data: {
              height: blockHeight,
              tx_count: Array.isArray(blockResponse.data) ? blockResponse.data.length : 0,
              hash: blockResponse.data[0]?.hash || 'N/A',
              transactions: blockResponse.data
            }
          });
        } else {
          results.push({ type: 'error', id: 'not-found', title: 'Block Not Found', subtitle: `Block #${blockHeight} could not be loaded` });
        }
        setSearchResults(results);
        setIsSearching(false);
        return;
      }

      let searchType = '';
      if (query.match(/^tx_[a-f0-9]+/i) || query.match(/^[a-f0-9]{32,128}$/i)) searchType = 'transaction';
      else if (query.match(/^qnk[a-z0-9]{39}$/i)) searchType = 'address';
      else if (query.match(/^\d+$/)) searchType = 'block';

      console.log(`🔍 Login search for: ${query} (type: ${searchType})`);

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
            type: 'error',
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
            subtitle: txData.status || 'confirmed',  // Amount is private
            hash: txData.hash || query,
            data: {
              hash: txData.hash || query,
              amount: amount,
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
            type: 'error',
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
            subtitle: 'Balance: 0 ' + TICKER_SYMBOL,
            hash: query,
            data: { address: query, balance: 0, nonce: 0 }
          });
        }
      } else if (query.length >= 3) {
        // Show helpful hints
        results.push({
          type: 'block',
          id: 'hint-block',
          title: 'Search by block number',
          subtitle: 'Enter a number to find a block'
        });
        results.push({
          type: 'transaction',
          id: 'hint-tx',
          title: 'Search by transaction hash',
          subtitle: 'Enter a 64-character hex hash'
        });
        results.push({
          type: 'address',
          id: 'hint-address',
          title: 'Search by wallet address',
          subtitle: 'Enter address starting with "qnk"'
        });
      }
    } catch (error) {
      console.error('Search failed:', error);
      results.push({
        type: 'error',
        id: 'error',
        title: 'Search Failed',
        subtitle: 'Unable to connect to the network'
      });
    }

    setSearchResults(results);
    setIsSearching(false);
  };

  const getResultIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'transaction': return <Hash className="w-4 h-4" />;
      case 'block': return <Blocks className="w-4 h-4" />;
      case 'address': return <User className="w-4 h-4" />;
      default: return <Search className="w-4 h-4" />;
    }
  };

  return (
    <>
      <div className="relative w-full max-w-md mx-auto">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-amber-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setShowResults(true);
              performSearch(e.target.value);
            }}
            onBlur={() => setTimeout(() => setShowResults(false), 200)}
            onFocus={() => setShowResults(true)}
            placeholder="Search tx hash, block, address..."
            className="w-full pl-10 pr-4 py-3 bg-slate-900/70 border-2 border-amber-500/30 rounded-xl text-amber-50 placeholder-amber-300/40 focus:outline-none focus:border-amber-400 focus:shadow-[0_0_15px_rgba(251,191,36,0.3)] transition-all"
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
          {showResults && searchResults.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-full mt-2 w-full backdrop-blur-xl rounded-xl shadow-2xl max-h-80 overflow-y-auto z-50"
              style={{
                background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.98) 0%, rgba(50, 30, 80, 0.98) 100%)',
                border: '2px solid rgba(212, 175, 55, 0.3)',
                boxShadow: '0 10px 40px rgba(212, 175, 55, 0.2)'
              }}
            >
              {searchResults.map((result, index) => (
                <motion.div
                  key={`${result.type}-${result.id}-${index}`}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-center justify-between p-4 border-b last:border-b-0 cursor-pointer"
                  style={{ borderColor: 'rgba(212, 175, 55, 0.1)' }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                  }}
                  onClick={() => {
                    if (result.data) {
                      setSelectedDetail(result);
                    }
                    setShowResults(false);
                    setSearchQuery('');
                  }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="p-2 rounded-lg"
                      style={{
                        background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                        border: '1px solid rgba(212, 175, 55, 0.3)'
                      }}
                    >
                      <div className="text-amber-400">{getResultIcon(result.type)}</div>
                    </div>
                    <div>
                      <div className="text-amber-100 font-semibold">{result.title}</div>
                      {result.subtitle && (
                        <div className="text-amber-300/60 text-sm">{result.subtitle}</div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Detail Modal */}
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
                    {/* ZK-STARK Privacy: Amount, From, To, Fee, Time are all private */}

                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Transaction Hash</div>
                      <div className="flex items-center gap-2">
                        <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.hash}</code>
                        <button
                          onClick={() => copyToClipboard(selectedDetail.data.hash, 'modal-hash')}
                          className="p-1 hover:bg-amber-500/20 rounded transition-colors"
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

                    {/* Privacy notice */}
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
                        <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.hash}</code>
                        <button
                          onClick={() => copyToClipboard(selectedDetail.data.hash, 'modal-block-hash')}
                          className="p-1 hover:bg-amber-500/20 rounded transition-colors"
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
                      <code className="text-amber-100 text-sm font-mono break-all">{selectedDetail.data.address}</code>
                      <button
                        onClick={() => copyToClipboard(selectedDetail.data.address, 'modal-address')}
                        className="p-1 hover:bg-amber-500/20 rounded transition-colors"
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
    </>
  );
}
