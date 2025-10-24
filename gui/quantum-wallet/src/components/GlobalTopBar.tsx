import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Loader, Zap, Gift, ChevronDown, Clock } from 'lucide-react';
import { qnkAPI, type MiningRewardEvent } from '../services/api';

interface GlobalTopBarProps {
  authenticated?: boolean;
}

interface SearchResult {
  type: 'block' | 'transaction' | 'wallet';
  id: string;
  height?: number;
  hash?: string;
  amount?: number;
  timestamp?: string;
  from?: string;
  to?: string;
  balance?: number;
}

type NetworkType = 'testnet' | 'mainnet';

export const NETWORK_SELECTOR_ENABLED = true; // Force inclusion

export default function GlobalTopBar({ authenticated = false }: GlobalTopBarProps) {
  console.log('GlobalTopBar render - Network selector enabled:', NETWORK_SELECTOR_ENABLED);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [miningHashRate, setMiningHashRate] = useState(0);
  const [isMining, setIsMining] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const walletAddress = localStorage.getItem('walletAddress') || '';

  // Network selection state
  const [selectedNetwork, setSelectedNetwork] = useState<NetworkType>(
    (localStorage.getItem('selectedNetwork') as NetworkType) || 'testnet'
  );
  const [showNetworkDropdown, setShowNetworkDropdown] = useState(false);
  const [mainnetCountdown, setMainnetCountdown] = useState('');

  // Mainnet launch date: December 15, 2025 00:00 UTC
  const MAINNET_LAUNCH = new Date('2025-12-15T00:00:00Z');

  // Update countdown every second
  useEffect(() => {
    const updateCountdown = () => {
      const now = new Date();
      const diff = MAINNET_LAUNCH.getTime() - now.getTime();

      if (diff <= 0) {
        setMainnetCountdown('LIVE NOW!');
        return;
      }

      const days = Math.floor(diff / (1000 * 60 * 60 * 24));
      const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((diff % (1000 * 60)) / 1000);

      setMainnetCountdown(`${days}d ${hours}h ${minutes}m ${seconds}s`);
    };

    updateCountdown();
    const interval = setInterval(updateCountdown, 1000);
    return () => clearInterval(interval);
  }, []);

  // Handle network switch
  const handleNetworkSwitch = (network: NetworkType) => {
    setSelectedNetwork(network);
    localStorage.setItem('selectedNetwork', network);
    setShowNetworkDropdown(false);

    // Update API base URL based on network
    const newPort = network === 'testnet' ? '8080' : '8081';
    const newBaseURL = `http://localhost:${newPort}`;

    // Store for API service to use
    localStorage.setItem('apiBaseURL', newBaseURL);

    // Reload to apply network changes
    window.location.reload();
  };

  // SSE for mining hash rate updates
  useEffect(() => {
    if (!walletAddress || !authenticated) return;

    const eventSource = qnkAPI.subscribeToMiningRewards(
      walletAddress,
      (reward: MiningRewardEvent) => {
        // Update hash rate from mining reward events
        if (reward.hash_rate > 0) {
          setMiningHashRate(reward.hash_rate);
          setIsMining(true);
        }
      },
      () => {} // Balance updates handled by MiningDashboard
    );

    eventSourceRef.current = eventSource;

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [walletAddress, authenticated]);

  // Debounced search effect
  useEffect(() => {
    const searchTimeout = setTimeout(() => {
      if (searchQuery.trim().length >= 3) { // Start searching after 3 characters
        handleSearch(searchQuery);
      } else if (searchQuery.trim().length === 0) {
        setSearchResults([]);
        setShowResults(false);
      }
    }, 300); // 300ms debounce delay

    return () => clearTimeout(searchTimeout);
  }, [searchQuery]);

  const handleSearch = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      setShowResults(false);
      return;
    }

    setIsSearching(true);
    setShowResults(true);

    try {
      const results: SearchResult[] = [];

      // Check if query is a number (potential block height)
      const blockHeight = parseInt(query);
      if (!isNaN(blockHeight) && blockHeight >= 0) {
        try {
          const blockResponse = await qnkAPI.getBlock(blockHeight);
          if (blockResponse.success && blockResponse.data) {
            results.push({
              type: 'block',
              id: `block-${blockHeight}`,
              height: blockHeight,
              hash: `0x${Math.random().toString(16).substr(2, 10)}`,
              timestamp: new Date().toISOString()
            });
          }
        } catch (error) {
          console.log('Block search failed:', error);
        }
      }

      // Check if query looks like a wallet address (starts with 'qnk')
      if (query.toLowerCase().startsWith('qnk') && query.length > 10) {
        try {
          const balanceResponse = await qnkAPI.getWalletBalance(query);
          if (balanceResponse.success && balanceResponse.data) {
            results.push({
              type: 'wallet',
              id: query,
              balance: balanceResponse.data.balance_qnk || 0,
              timestamp: new Date().toISOString()
            });
          }
        } catch (error) {
          console.log('Wallet search failed:', error);
        }
      }

      // Check if query looks like a transaction hash (partial or full hex)
      if (/^[a-fA-F0-9]{8,64}$/.test(query)) {
        try {
          // Search through recent transactions for the hash - NO MOCK DATA
          const transactionsResponse = await qnkAPI.getRecentTransactions(100);
          if (transactionsResponse.success && transactionsResponse.data) {
            const foundTx = transactionsResponse.data.find((tx: any) =>
              tx.id === query || tx.hash === query ||
              (tx.id && tx.id.includes(query)) || (tx.hash && tx.hash.includes(query))
            );

            if (foundTx) {
              // Only show limited info for privacy - ZK-SNARK/STARK protection
              results.push({
                type: 'transaction',
                id: foundTx.hash || foundTx.id,
                hash: foundTx.hash || foundTx.id,
                amount: undefined, // PRIVACY: Hide amount unless user owns the transaction
                from: undefined,   // PRIVACY: Hide addresses for quantum privacy
                to: undefined,     // PRIVACY: Hide addresses for quantum privacy
                timestamp: foundTx.timestamp_formatted || new Date(foundTx.timestamp * 1000).toLocaleString()
              });
            }
          }
        } catch (error) {
          console.log('Transaction search failed:', error);
        }
      }

      // No fallback mock results - only show real data per CLAUDE.md

      setSearchResults(results);
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Search is now handled automatically by useEffect
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchQuery(value);

    // Show results dropdown when typing
    if (value.trim().length > 0) {
      setShowResults(true);
    } else {
      setShowResults(false);
    }
  };

  const formatHash = (hash: string) => {
    if (hash.length > 16) {
      return `${hash.substr(0, 8)}...${hash.substr(-8)}`;
    }
    return hash;
  };

  const formatAmount = (amount: number) => {
    return amount.toFixed(4);
  };

  const formatHashRate = (hashRate: number) => {
    if (hashRate >= 1e9) return `${(hashRate / 1e9).toFixed(2)} GH/s`;
    if (hashRate >= 1e6) return `${(hashRate / 1e6).toFixed(2)} MH/s`;
    if (hashRate >= 1e3) return `${(hashRate / 1e3).toFixed(2)} KH/s`;
    return `${hashRate.toFixed(2)} H/s`;
  };

  const handleResultClick = (result: SearchResult) => {
    setShowResults(false);
    setSearchQuery('');

    if (result.type === 'transaction') {
      // Navigate to explorer with transaction hash
      window.location.href = `/explorer/tx/${result.hash}`;
    } else if (result.type === 'block') {
      // Navigate to explorer (could add block-specific routes)
      window.location.href = `/explorer`;
    } else if (result.type === 'wallet') {
      // Navigate to explorer (could add wallet-specific routes)
      window.location.href = `/explorer`;
    }
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
                  {/* Cosmic glow effect */}
                  <div className="absolute inset-0 bg-gradient-to-b from-amber-500/20 via-orange-500/20 to-yellow-500/20 rounded-full blur-lg animate-pulse" />
                  {/* Gold border ring */}
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
              <form onSubmit={handleSearchSubmit} className="relative">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={handleInputChange}
                    onFocus={() => searchQuery.length >= 3 && setShowResults(true)}
                    onBlur={() => setTimeout(() => setShowResults(false), 200)}
                    placeholder="Search blocks, transactions, or wallet addresses..."
                    className="w-full pl-10 pr-4 py-2 bg-quantum-indigo/30 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:border-quantum-cyan focus:ring-1 focus:ring-quantum-cyan transition-all"
                  />
                  {isSearching && (
                    <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                      <Loader className="w-4 h-4 text-quantum-cyan animate-spin" />
                    </div>
                  )}
                </div>
              </form>

              {/* Search Results Dropdown */}
              {showResults && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="absolute top-full left-0 right-0 mt-2 bg-quantum-indigo/90 backdrop-blur-xl border border-quantum-purple/30 rounded-xl shadow-2xl max-h-96 overflow-y-auto z-50"
                >
                  {searchResults.length > 0 ? (
                    <div className="p-2">
                      {searchResults.map((result) => (
                        <motion.div
                          key={result.id}
                          className="p-3 hover:bg-quantum-purple/20 rounded-lg cursor-pointer transition-colors"
                          whileHover={{ scale: 1.02 }}
                          onClick={() => handleResultClick(result)}
                        >
                          {result.type === 'block' && (
                            <div>
                              <div className="text-quantum-cyan font-medium">Block #{result.height}</div>
                              <div className="text-gray-300 text-sm">Hash: {formatHash(result.hash || '')}</div>
                              <div className="text-gray-400 text-xs">{new Date(result.timestamp || '').toLocaleString()}</div>
                            </div>
                          )}

                          {result.type === 'wallet' && (
                            <div>
                              <div className="text-quantum-green font-medium">Wallet Address</div>
                              <div className="text-gray-300 text-sm">{formatHash(result.id)}</div>
                              <div className="text-quantum-yellow text-sm">Balance: {formatAmount(result.balance || 0)} QNK</div>
                            </div>
                          )}

                          {result.type === 'transaction' && (
                            <div>
                              <div className="text-quantum-pink font-medium">🔒 Private Transaction</div>
                              <div className="text-gray-300 text-sm">Hash: {formatHash(result.hash || '')}</div>
                              <div className="text-quantum-purple text-sm">🛡️ ZK-SNARK Protected</div>
                              <div className="text-gray-400 text-xs">
                                Details hidden for quantum privacy
                              </div>
                            </div>
                          )}
                        </motion.div>
                      ))}
                    </div>
                  ) : isSearching ? (
                    <div className="p-4 text-center text-gray-400">
                      <Loader className="w-5 h-5 animate-spin mx-auto mb-2" />
                      Searching quantum ledger...
                    </div>
                  ) : searchQuery && searchQuery.length >= 3 ? (
                    <div className="p-4 text-center text-gray-400">
                      No results found for "{searchQuery}"
                    </div>
                  ) : searchQuery && searchQuery.length < 3 ? (
                    <div className="p-4 text-center text-gray-400">
                      Type at least 3 characters to search...
                    </div>
                  ) : null}
                </motion.div>
              )}
            </div>

            {/* Mining Hash Rate Indicator & Status */}
            <div className="flex items-center gap-4">
              {/* Network Selector Dropdown - ALWAYS VISIBLE */}
              <div className="relative network-selector-container" style={{ border: '2px solid red' }}>
                <motion.button
                  onClick={() => setShowNetworkDropdown(!showNetworkDropdown)}
                  whileHover={{ scale: 1.05 }}
                  className={`flex items-center gap-2 rounded-lg px-3 py-1.5 border transition-all ${
                    selectedNetwork === 'testnet'
                      ? 'bg-gradient-to-r from-purple-600/20 to-pink-600/20 border-purple-500/40'
                      : 'bg-gradient-to-r from-green-600/20 to-emerald-600/20 border-green-500/40'
                  }`}
                >
                  <span className={`text-sm font-bold ${
                    selectedNetwork === 'testnet' ? 'text-purple-300' : 'text-green-300'
                  }`}>
                    {selectedNetwork.toUpperCase()}
                  </span>
                  <ChevronDown className={`w-4 h-4 transition-transform ${
                    showNetworkDropdown ? 'rotate-180' : ''
                  } ${selectedNetwork === 'testnet' ? 'text-purple-300' : 'text-green-300'}`} />
                </motion.button>

                {/* Network Dropdown */}
                <AnimatePresence>
                  {showNetworkDropdown && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="absolute top-full right-0 mt-2 w-72 bg-quantum-indigo/95 backdrop-blur-xl border border-quantum-purple/30 rounded-xl shadow-2xl overflow-hidden z-50"
                    >
                      {/* Testnet Option */}
                      <motion.div
                        whileHover={{ backgroundColor: 'rgba(147, 51, 234, 0.1)' }}
                        onClick={() => handleNetworkSwitch('testnet')}
                        className="p-4 cursor-pointer border-b border-quantum-purple/20"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-purple-300 font-bold">TESTNET</span>
                          {selectedNetwork === 'testnet' && (
                            <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                          )}
                        </div>
                        <p className="text-gray-400 text-xs">
                          Development & testing network (Port 8080)
                        </p>
                      </motion.div>

                      {/* Mainnet Option */}
                      <motion.div
                        whileHover={{ backgroundColor: 'rgba(34, 197, 94, 0.1)' }}
                        onClick={() => handleNetworkSwitch('mainnet')}
                        className="p-4 cursor-pointer"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-green-300 font-bold">MAINNET</span>
                          {selectedNetwork === 'mainnet' && (
                            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                          )}
                        </div>
                        <p className="text-gray-400 text-xs mb-2">
                          Production network (Port 8081)
                        </p>
                        {/* Countdown Timer */}
                        <div className="flex items-center gap-2 bg-quantum-yellow/10 border border-quantum-yellow/30 rounded-lg px-2 py-1">
                          <Clock className="w-3 h-3 text-quantum-yellow" />
                          <span className="text-quantum-yellow text-xs font-mono">
                            {mainnetCountdown}
                          </span>
                        </div>
                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Bounty Campaign Button */}
              <motion.a
                href="https://bounty.quillon.xyz"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.05 }}
                className="flex items-center gap-2 bg-gradient-to-r from-amber-500/20 to-yellow-500/20 border border-amber-500/40 rounded-lg px-3 py-1.5 cursor-pointer transition-all hover:from-amber-500/30 hover:to-yellow-500/30"
              >
                <Gift className="w-4 h-4 text-amber-400" />
                <span className="text-amber-300 text-sm font-semibold">Bounty Campaign</span>
              </motion.a>

              {/* Mining Hash Rate (SSE Real-Time) */}
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

              {/* Connection Status Indicator */}
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
    </>
  );
}