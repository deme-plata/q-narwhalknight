import { useState, useEffect, useRef, memo, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Copy, Check, ExternalLink, Hash, User, Blocks, Shield } from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';
// import { qnkAPI } from '../services/api'; // For future real API integration

interface SearchResult {
  type: 'transaction' | 'block' | 'address' | 'node';
  id: string;
  title: string;
  subtitle?: string;
  hash?: string;
}

interface TopBarProps {
  currentBalance: number;
  nodeId: string;
  blockHeight: number;
  peers: number;
  isOnline: boolean;
  qci: number; // Quantum Coherence Index
}

// v2.4.0: Memoized for performance
const TopBar = memo(function TopBar({ currentBalance, nodeId, blockHeight, peers, isOnline, qci }: TopBarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // v2.9.0-beta: STABLE balance display - prevent bouncing between multiple sources
  const [stableBalance, setStableBalance] = useState<number>(() => {
    // Initialize from localStorage cache to prevent flash
    const cached = localStorage.getItem('cachedBalance');
    const cachedValue = cached ? parseFloat(cached) : 0;
    return !isNaN(cachedValue) && isFinite(cachedValue) ? cachedValue : currentBalance;
  });
  const lastBalanceUpdateRef = useRef<number>(Date.now());
  const balanceStabilityWindowMs = 2000; // Don't change balance more than once per 2 seconds

  // v2.9.0-beta: Stabilized balance getter - prevents rapid flickering
  const getDisplayBalance = (): number => {
    // Check if we have a locked balance from DEX (SOURCE OF TRUTH during cooldown)
    const lockedBalance = localStorage.getItem('dexLockedBalance');
    const cooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');

    if (lockedBalance && Date.now() < cooldownUntil) {
      const locked = parseFloat(lockedBalance);
      if (!isNaN(locked) && isFinite(locked)) {
        return locked;
      }
    }

    // Return the stable balance (updated only when stability window allows)
    return stableBalance;
  };

  // v2.9.0-beta: Update stable balance with debouncing to prevent flickering
  useEffect(() => {
    const cached = localStorage.getItem('cachedBalance');
    const cachedValue = cached ? parseFloat(cached) : currentBalance;
    const newBalance = !isNaN(cachedValue) && isFinite(cachedValue) ? cachedValue : currentBalance;

    // Only update if enough time has passed (prevents rapid flickering)
    const timeSinceLastUpdate = Date.now() - lastBalanceUpdateRef.current;
    const balanceDifference = Math.abs(newBalance - stableBalance);

    // Update if: significant change (>1 QUG) OR stability window passed
    if (balanceDifference > 1 || timeSinceLastUpdate > balanceStabilityWindowMs) {
      // Use the HIGHER of the two values to prevent showing decreased balance from race conditions
      const bestBalance = Math.max(newBalance, stableBalance, currentBalance);

      if (Math.abs(bestBalance - stableBalance) > 0.0001) {
        console.log('💰 TopBar: Stable balance update:', stableBalance.toFixed(4), '→', bestBalance.toFixed(4));
        setStableBalance(bestBalance);
        lastBalanceUpdateRef.current = Date.now();
      }
    }
  }, [currentBalance, stableBalance]);

  // v2.9.0-beta: Listen for balance change events and update stable balance
  useEffect(() => {
    const handleBalanceChanged = (event: Event) => {
      const customEvent = event as CustomEvent;
      const newBalance = customEvent.detail?.balance;
      if (typeof newBalance === 'number' && !isNaN(newBalance) && isFinite(newBalance)) {
        console.log('🔥 TopBar: qug-balance-changed received:', newBalance);
        // v2.9.3-beta: DEX swaps are AUTHORITATIVE - allow both increases AND decreases
        // The qug-balance-changed event is ONLY dispatched by DexScreen after successful swaps
        // so we MUST trust the value even if it's lower (e.g., QUG -> QUGUSD swap)
        setStableBalance(newBalance);
        lastBalanceUpdateRef.current = Date.now();
      }
    };

    const handleDexCooldownExpired = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { qugBalance } = customEvent.detail || {};
      console.log('🔄 TopBar: DEX cooldown expired, balance:', qugBalance);
      if (typeof qugBalance === 'number' && !isNaN(qugBalance) && isFinite(qugBalance)) {
        setStableBalance(qugBalance);
        lastBalanceUpdateRef.current = Date.now();
      }
    };

    const handleWalletBalanceUpdated = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { symbol, balance, reason } = customEvent.detail || {};
      if (symbol !== 'QUG') return;
      if (typeof balance === 'number' && !isNaN(balance) && isFinite(balance)) {
        console.log('💰 TopBar: wallet-balance-updated:', balance, 'reason:', reason);

        // v2.9.1-beta: Mining updates are authoritative - use directly
        // DEX updates use Math.max() to prevent race condition drops
        const isMiningUpdate = reason && (
          reason === 'mining_reward' ||
          reason === 'p2p_mining_reward' ||
          reason === 'mining_stats_update' ||
          reason.startsWith('mining_reward')
        );

        // v2.9.3-beta: Check if this is a DEX swap deduction
        const isDexSwapDeduct = reason && (
          reason === 'dex-swap-deduct' ||
          reason === 'DexScreen.swap.deduct'
        );

        if (isMiningUpdate || isDexSwapDeduct) {
          // Mining updates and DEX swap deductions: use value directly
          // DEX swaps are authoritative - the user just spent QUG, balance MUST decrease
          console.log(isDexSwapDeduct ? '💸 TopBar: DEX deduct - setting balance to:' : '⛏️ TopBar: Mining update - setting balance to:', balance);
          setStableBalance(balance);
        } else {
          // Other updates: use max to prevent race condition drops (but allow higher)
          setStableBalance(prev => Math.max(prev, balance));
        }
        lastBalanceUpdateRef.current = Date.now();
      }
    };

    window.addEventListener('qug-balance-changed', handleBalanceChanged);
    window.addEventListener('dex-cooldown-expired', handleDexCooldownExpired);
    window.addEventListener('wallet-balance-updated', handleWalletBalanceUpdated);

    return () => {
      window.removeEventListener('qug-balance-changed', handleBalanceChanged);
      window.removeEventListener('dex-cooldown-expired', handleDexCooldownExpired);
      window.removeEventListener('wallet-balance-updated', handleWalletBalanceUpdated);
    };
  }, []);

  // Get the display balance (always from localStorage)
  const displayBalance = getDisplayBalance();

  // Mock search function - replace with real API calls
  const performSearch = async (query: string) => {
    if (query.length < 3) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    
    // Simulate API search delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const results: SearchResult[] = [];
    
    // Check if it looks like a hash (64 hex characters)
    if (query.match(/^[a-fA-F0-9]{8,64}$/)) {
      if (query.length === 64) {
        results.push({
          type: 'transaction',
          id: query,
          title: 'Transaction',
          subtitle: `${query.substring(0, 16)}...${query.substring(48)}`,
          hash: query
        });
        results.push({
          type: 'block',
          id: query,
          title: 'Block Hash',
          subtitle: `${query.substring(0, 16)}...${query.substring(48)}`,
          hash: query
        });
      } else {
        results.push({
          type: 'address',
          id: query,
          title: 'Address',
          subtitle: `${query.substring(0, 8)}...${query.substring(-8)}`,
          hash: query
        });
      }
    }
    
    // Check if it looks like a block number
    if (query.match(/^\\d+$/)) {
      const blockNum = parseInt(query);
      results.push({
        type: 'block',
        id: blockNum.toString(),
        title: `Block #${blockNum}`,
        subtitle: blockNum <= blockHeight ? 'Confirmed' : 'Future block',
      });
    }
    
    // Check if it matches node ID pattern
    if (query.toLowerCase().includes(nodeId.substring(0, 8).toLowerCase())) {
      results.push({
        type: 'node',
        id: nodeId,
        title: 'Current Node',
        subtitle: `${nodeId.substring(0, 16)}...`,
        hash: nodeId
      });
    }

    // Add some mock recent transactions/blocks
    if (query.toLowerCase().includes('recent') || query.toLowerCase().includes('latest')) {
      for (let i = 0; i < 3; i++) {
        results.push({
          type: 'block',
          id: (blockHeight - i).toString(),
          title: `Block #${blockHeight - i}`,
          subtitle: `${i === 0 ? 'Latest' : `${i} blocks ago`}`,
        });
      }
    }

    setSearchResults(results);
    setIsSearching(false);
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchQuery.trim()) {
        performSearch(searchQuery);
      } else {
        setSearchResults([]);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getQCIStatus = (qci: number) => {
    if (qci >= 0.9) return 'Sublime';
    if (qci >= 0.8) return 'Coherent';
    return 'Stabilizing';
  };

  const getResultIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'transaction': return <Hash className="w-4 h-4" />;
      case 'block': return <Blocks className="w-4 h-4" />;
      case 'address': return <User className="w-4 h-4" />;
      case 'node': return <User className="w-4 h-4" />;
    }
  };

  return (
    <div
      className="backdrop-blur-xl border-b px-6 py-4 relative z-50"
      style={{
        background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.95) 0%, rgba(50, 30, 80, 0.95) 100%)',
        borderColor: 'rgba(212, 175, 55, 0.2)',
        boxShadow: '0 4px 20px rgba(212, 175, 55, 0.15)'
      }}
    >
      <div className="flex items-center justify-between">
        {/* Left: Search */}
        <div className="flex-1 max-w-md relative">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-amber-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setShowResults(true);
              }}
              onBlur={() => setTimeout(() => setShowResults(false), 200)}
              onFocus={() => setShowResults(true)}
              placeholder="Search transactions, blocks, addresses..."
              className="w-full pl-10 pr-4 py-2 bg-slate-900/70 border-2 border-amber-500/30 rounded-lg text-amber-50 placeholder-amber-300/40 focus:outline-none focus:border-amber-400 focus:shadow-[0_0_15px_rgba(251,191,36,0.3)] transition-all"
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
                className="absolute top-full mt-2 w-full backdrop-blur-xl rounded-lg shadow-2xl max-h-96 overflow-y-auto z-50"
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
                    className="flex items-center justify-between p-4 border-b last:border-b-0 cursor-pointer group"
                    style={{
                      borderColor: 'rgba(212, 175, 55, 0.1)'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                    }}
                    onClick={() => {
                      console.log('Navigate to:', result);
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
                    <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      {result.hash && (
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            copyToClipboard(result.hash!, `${result.type}-${result.id}`);
                          }}
                          className="p-1 text-amber-400/60 hover:text-amber-400 transition-colors"
                        >
                          {copiedId === `${result.type}-${result.id}` ? (
                            <Check className="w-4 h-4 text-green-400" />
                          ) : (
                            <Copy className="w-4 h-4" />
                          )}
                        </motion.button>
                      )}
                      <ExternalLink className="w-4 h-4 text-amber-400/60" />
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Center: Network Status */}
        <div className="flex items-center gap-6">
          <div className="text-center relative group">
            <motion.div
              className="font-bold text-lg bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent cursor-help"
              key="stable-balance-display"
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              title={`Full precision: ${displayBalance.toFixed(24)} ${TICKER_SYMBOL}\n(24 decimal places)`}
            >
              {displayBalance.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 12
              })} {TICKER_SYMBOL}
            </motion.div>
            {/* v3.2.18-beta: Hover tooltip showing full 24-decimal precision */}
            <div className="absolute left-1/2 -translate-x-1/2 top-full mt-2 bg-quantum-dark/95 border border-amber-500/30 rounded-lg px-4 py-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50 whitespace-nowrap">
              <div className="text-xs text-gray-400 mb-1">Full 24-decimal precision:</div>
              <div className="font-mono text-amber-400 text-sm">{displayBalance.toFixed(24)} {TICKER_SYMBOL}</div>
            </div>
            <div className="text-amber-300/60 text-sm font-medium flex items-center gap-2 justify-center">
              <span>Total Balance</span>
              <motion.div
                className="w-1.5 h-1.5 rounded-full bg-green-400"
                animate={{ scale: [1, 1.3, 1], opacity: [0.7, 1, 0.7] }}
                transition={{ duration: 2, repeat: Infinity }}
                title="Live updates enabled"
              />
            </div>
          </div>

          <div className="h-8 w-px bg-gradient-to-b from-transparent via-amber-500/30 to-transparent" />

          <div className="flex items-center gap-3">
            <motion.div
              className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-400' : 'bg-red-500'}`}
              animate={isOnline ? { scale: [1, 1.2, 1], opacity: [0.7, 1, 0.7] } : {}}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <div className="text-amber-100 text-sm font-medium">
              Block #{blockHeight} • {peers} peers
            </div>
          </div>
        </div>

        {/* Right: Coherence Index */}
        <div className="flex items-center gap-3">
          <motion.div
            animate={{
              scale: qci >= 0.9 ? [1, 1.1, 1] : [1, 1.05, 1],
              rotate: [0, 360],
            }}
            transition={{
              scale: { duration: 2, repeat: Infinity },
              rotate: { duration: qci >= 0.9 ? 20 : 40, repeat: Infinity, ease: "linear" }
            }}
          >
            <Shield className="w-5 h-5 text-amber-400" />
          </motion.div>
          <div className="text-right">
            <div className="flex items-center gap-2">
              <motion.span
                className="text-amber-100 text-sm font-bold"
                key={qci}
                initial={{ scale: 1.2, opacity: 0.5 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                {(qci * 100).toFixed(0)}%
              </motion.span>
              <motion.span
                className="text-xs font-semibold bg-gradient-to-r from-amber-400 to-yellow-500 bg-clip-text text-transparent"
                key={getQCIStatus(qci)}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4 }}
              >
                {getQCIStatus(qci)}
              </motion.span>
            </div>
            <div className="text-amber-300/60 text-xs font-medium">Quantum Coherence</div>
          </div>
          <div
            className="relative w-12 h-2 rounded-full overflow-hidden"
            style={{
              background: 'rgba(15, 23, 42, 0.7)',
              border: '1px solid rgba(212, 175, 55, 0.3)',
              boxShadow: qci >= 0.9
                ? '0 0 15px rgba(212, 175, 55, 0.6), 0 0 30px rgba(255, 215, 0, 0.4)'
                : qci >= 0.8
                ? '0 0 10px rgba(212, 175, 55, 0.4)'
                : '0 0 5px rgba(212, 175, 55, 0.2)'
            }}
          >
            <motion.div
              className="h-full relative"
              style={{
                background: qci >= 0.9
                  ? 'linear-gradient(90deg, #FFD700, #FFA500, #FF8C00, #FFD700)'
                  : qci >= 0.8
                  ? 'linear-gradient(90deg, #D4AF37, #FFD700, #FFA500)'
                  : 'linear-gradient(90deg, #8B7355, #D4AF37, #FFD700)',
                backgroundSize: '200% 100%'
              }}
              initial={{ width: 0 }}
              animate={{
                width: `${qci * 100}%`,
                backgroundPosition: qci >= 0.8 ? ['0% 0%', '200% 0%'] : '0% 0%'
              }}
              transition={{
                width: { duration: 1, ease: "easeOut" },
                backgroundPosition: {
                  duration: 2,
                  repeat: Infinity,
                  ease: "linear"
                }
              }}
            >
              {/* Shimmer effect for high coherence */}
              {qci >= 0.8 && (
                <motion.div
                  className="absolute inset-0"
                  style={{
                    background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent)',
                  }}
                  animate={{
                    x: ['-100%', '200%']
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                />
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
});

export default TopBar;