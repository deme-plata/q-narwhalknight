import { useState, useEffect, useRef, memo, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Zap, ChevronLeft, ChevronRight, Flame } from 'lucide-react';
import TokenIcon from './TokenIcon';
import { qnkAPI } from '../services/api';
import NitroSuccessModal from './NitroSuccessModal';

// v3.6.1-beta: SANITY CHECK - Max possible balance is 21 million QUG (total supply)
const MAX_SANE_BALANCE = 21_000_000;

/**
 * v3.6.1-beta: Safe localStorage set for cachedBalance - validates before storing
 */
function safeCacheBalance(balance: number): void {
  if (typeof balance === 'number' && !isNaN(balance) && isFinite(balance) &&
      balance >= 0 && balance <= MAX_SANE_BALANCE) {
    localStorage.setItem('cachedBalance', balance.toString());
  } else {
    console.warn(`🚨 [TokenBar] safeCacheBalance: Refusing to cache invalid balance: ${balance}`);
  }
}

interface Token {
  id: string;
  symbol: string;
  name: string;
  balance: number;
  price: number;
  change24h: number;
  volume24h: number;
  icon: string;
  isNitroBoost?: boolean;
  marketCap?: number; // v2.8.1-beta: Market cap for filtering/sorting
}

interface TokenBarProps {
  onTokenClick?: (token: Token) => void;
}

// v2.4.0: Memoized for performance
const TokenBar = memo(function TokenBar({ onTokenClick }: TokenBarProps) {
  const [tokens, setTokens] = useState<Token[]>([]);
  const [loading, setLoading] = useState(true);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [nitroPoints, setNitroPoints] = useState(0);
  const [showNitroPurchase, setShowNitroPurchase] = useState(false);
  const [purchaseAmount, setPurchaseAmount] = useState(500);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [successModalData, setSuccessModalData] = useState<any>(null);

  // Fetch Nitro Points from localStorage (per wallet address)
  useEffect(() => {
    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (walletAddress) {
      const storedPoints = localStorage.getItem(`nitroPoints_${walletAddress}`);
      if (storedPoints) {
        setNitroPoints(parseInt(storedPoints, 10));
      } else {
        // New wallet - reset to 0
        setNitroPoints(0);
      }
    }
  }, []);

  // Also listen for storage changes (in case DexScreen updates the points)
  // v6.0.3: Use ref instead of state in deps to prevent effect re-registration on every change
  const nitroPointsRef = useRef(nitroPoints);
  nitroPointsRef.current = nitroPoints;

  useEffect(() => {
    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (!walletAddress) return;

    const handleStorageChange = () => {
      const storedPoints = localStorage.getItem(`nitroPoints_${walletAddress}`);
      if (storedPoints) {
        setNitroPoints(parseInt(storedPoints, 10));
      }
    };

    window.addEventListener('storage', handleStorageChange);

    // Also check periodically in case same-tab changes
    const interval = setInterval(() => {
      const storedPoints = localStorage.getItem(`nitroPoints_${walletAddress}`);
      if (storedPoints) {
        const points = parseInt(storedPoints, 10);
        if (points !== nitroPointsRef.current) {
          setNitroPoints(points);
        }
      }
    }, 1000);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(interval);
    };
  }, []);

  // Fetch tokens from API
  useEffect(() => {
    const fetchTokens = async () => {
      try {
        const walletAddress = localStorage.getItem('walletAddress') || '';

        // Fetch native QUG balance
        let nativeQugBalance = 0;
        if (walletAddress) {
          try {
            const balanceResponse = await qnkAPI.getWalletBalance(walletAddress);
            if (balanceResponse.success && balanceResponse.data) {
              nativeQugBalance = balanceResponse.data.balance_qnk || 0;
              // v2.3.27-beta: Check global DEX cooldown before writing to localStorage
              const cooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
              if (Date.now() < cooldownUntil) {
                console.log('🚫 TokenBar: SKIPPING localStorage write - DEX cooldown active');
              } else {
                safeCacheBalance(nativeQugBalance);
                console.log('💰 TokenBar: Cached balance:', nativeQugBalance);
              }
            } else {
              // Authentication failed - use cached balance from localStorage
              console.warn('⚠️ TokenBar: Balance fetch failed, using cached balance');
              const cachedBalance = localStorage.getItem('cachedBalance');
              if (cachedBalance) {
                nativeQugBalance = parseFloat(cachedBalance);
                console.log('💰 TokenBar: Using cached balance:', nativeQugBalance);
              }
            }
          } catch (error) {
            console.error('Failed to fetch native QUG balance:', error);
            // Fallback: use cached balance from localStorage
            const cachedBalance = localStorage.getItem('cachedBalance');
            if (cachedBalance) {
              nativeQugBalance = parseFloat(cachedBalance);
              console.log('💰 TokenBar: Using cached balance (error fallback):', nativeQugBalance);
            }
          }
        }

        // Fetch real QUG price from oracle
        let qugPrice = 3000.00;
        let qugChange = 12.8;
        try {
          const oracleResponse = await qnkAPI.getOraclePrice('QUG/USD');
          if (oracleResponse.success && oracleResponse.data) {
            qugPrice = oracleResponse.data.price;
            qugChange = oracleResponse.data.change_24h;
          }
        } catch (error) {
          console.error('Failed to fetch QUG price:', error);
        }

        // Fetch QUGUSD price
        let qugusdPrice = 1.00;
        let qugusdChange = 0.02;
        try {
          const oracleResponse = await qnkAPI.getOraclePrice('QUGUSD/USD');
          if (oracleResponse.success && oracleResponse.data) {
            qugusdPrice = oracleResponse.data.price;
            qugusdChange = oracleResponse.data.change_24h;
          }
        } catch (error) {
          console.error('Failed to fetch QUGUSD price:', error);
        }

        // Native tokens with Nitro Boost status
        const nativeTokens: Token[] = [
          {
            id: 'native-qug',
            symbol: 'QUG',
            name: 'Quillon',
            balance: nativeQugBalance,
            price: qugPrice,
            change24h: qugChange,
            volume24h: 1850000,
            icon: 'qug-logo',
            isNitroBoost: true, // QUG always has Nitro Boost
          },
          {
            id: 'qugusd-stable',
            symbol: 'QUGUSD',
            name: 'Quillon USD',
            balance: 0,
            price: qugusdPrice,
            change24h: qugusdChange,
            volume24h: 950000,
            icon: 'qugusd-logo',
            isNitroBoost: true, // Stablecoin has Nitro Boost
          },
        ];

        // Fetch custom tokens - v2.8.1-beta: Only show established tokens with minimum requirements
        const response = await qnkAPI.getSupportedTokens();
        let enrichedTokens = nativeTokens;

        // v4.0.12: Fetch actual Nitro boost data from backend
        let nitroBoostMap: Record<string, number> = {};
        try {
          const nitroResponse = await qnkAPI.getNitroBoosts();
          if (nitroResponse.success && nitroResponse.data) {
            nitroBoostMap = nitroResponse.data;
            console.log('🚀 TokenBar: Loaded Nitro boosts:', Object.keys(nitroBoostMap).length, 'tokens boosted');
          }
        } catch (error) {
          console.log('ℹ️ TokenBar: No Nitro boost data available');
        }

        // v2.8.1-beta: Minimum requirements for TokenBar display
        // This prevents brand new tokens from appearing immediately in the top bar
        const MIN_MARKET_CAP = 10000;      // $10,000 minimum market cap
        const MIN_VOLUME_24H = 1000;        // $1,000 minimum 24h volume
        const MIN_LIQUIDITY = 5000;         // $5,000 minimum liquidity (if available)

        if (response.success && response.data) {
          const apiTokensPromises = response.data
            .filter(apiToken => apiToken.symbol !== 'QUG' && apiToken.symbol !== 'QUGUSD')
            .map(async (apiToken) => {
              // Fetch price from oracle FIRST to determine if token meets criteria
              let customPrice = 0;
              let customChange = 0.0;
              let customVolume = 0.0;
              // v4.0.12: Check actual Nitro boost data from backend instead of hardcoded thresholds
              let hasNitroBoost = !!(nitroBoostMap[apiToken.address] && nitroBoostMap[apiToken.address] > 0);

              try {
                const oracleResponse = await qnkAPI.getOraclePrice(apiToken.address);
                if (oracleResponse.success && oracleResponse.data) {
                  customPrice = oracleResponse.data.price || 0;
                  customChange = oracleResponse.data.change_24h || 0;
                  customVolume = oracleResponse.data.volume_24h || 0;
                  // v4.0.12: hasNitroBoost is now set from actual backend data above
                }
              } catch (error) {
                console.log(`ℹ️ No oracle price for ${apiToken.symbol}`);
              }

              // v2.8.1-beta: Calculate market cap from total supply and price
              const totalSupply = parseInt(apiToken.total_supply || '0');
              const decimals = apiToken.decimals || 8;
              const adjustedSupply = totalSupply / Math.pow(10, decimals);
              const marketCap = adjustedSupply * customPrice;

              // v2.8.1-beta: Filter out tokens that don't meet minimum requirements
              // New tokens won't have volume or meaningful market cap
              const meetsRequirements = (
                marketCap >= MIN_MARKET_CAP ||       // Has minimum market cap
                customVolume >= MIN_VOLUME_24H ||    // OR has trading volume
                hasNitroBoost                         // OR has Nitro Boost (manually promoted)
              );

              if (!meetsRequirements) {
                console.log(`🚫 TokenBar: Filtering out ${apiToken.symbol} - marketCap: $${marketCap.toFixed(2)}, volume: $${customVolume.toFixed(2)}`);
                return null; // Will be filtered out
              }

              // Fetch balance only for tokens that meet requirements
              let tokenBalance = 0;
              if (walletAddress) {
                try {
                  const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, apiToken.address);
                  if (balanceResponse.success && balanceResponse.data) {
                    tokenBalance = balanceResponse.data.balance || 0;
                  }
                } catch (error) {
                  console.error(`Failed to fetch balance for ${apiToken.symbol}:`, error);
                }
              }

              return {
                id: apiToken.address,
                symbol: apiToken.symbol,
                name: apiToken.name,
                balance: tokenBalance,
                price: customPrice,
                change24h: customChange,
                volume24h: customVolume,
                icon: '🪙',
                isNitroBoost: hasNitroBoost,
                marketCap: marketCap, // Include for sorting
              };
            });

          const apiTokensRaw = await Promise.all(apiTokensPromises);
          // v2.8.1-beta: Filter out null entries (tokens that didn't meet requirements)
          // and sort by volume (highest first) to show best performing tokens
          const apiTokens = apiTokensRaw
            .filter((t): t is NonNullable<typeof t> => t !== null)
            .sort((a, b) => b.volume24h - a.volume24h);

          enrichedTokens = [...nativeTokens, ...apiTokens];
          console.log(`✅ TokenBar: Showing ${apiTokens.length} custom tokens (filtered by market cap/volume)`);
        }

        setTokens(enrichedTokens);
      } catch (error) {
        console.error('Failed to fetch tokens for TokenBar:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTokens();

    // Refresh every 30 seconds
    const interval = setInterval(fetchTokens, 30000);
    return () => clearInterval(interval);
  }, []);

  // Check scroll position
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const checkScroll = () => {
      setCanScrollLeft(container.scrollLeft > 0);
      setCanScrollRight(
        container.scrollLeft < container.scrollWidth - container.clientWidth - 10
      );
    };

    checkScroll();
    container.addEventListener('scroll', checkScroll);
    window.addEventListener('resize', checkScroll);

    return () => {
      container.removeEventListener('scroll', checkScroll);
      window.removeEventListener('resize', checkScroll);
    };
  }, [tokens]);

  const scroll = (direction: 'left' | 'right') => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const scrollAmount = 300;
    const newPosition = direction === 'left'
      ? container.scrollLeft - scrollAmount
      : container.scrollLeft + scrollAmount;

    container.scrollTo({
      left: newPosition,
      behavior: 'smooth'
    });
  };

  const handlePurchaseNitro = async () => {
    try {
      const walletAddress = localStorage.getItem('walletAddress');
      if (!walletAddress) {
        alert('Please connect your wallet first');
        return;
      }

      // Calculate QUG cost (1 point = 0.01 QUG)
      const qugCost = purchaseAmount * 0.01;

      // Get current QUG balance (with fallback to cached balance)
      const balanceResponse = await qnkAPI.getWalletBalance(walletAddress);
      let currentBalance = 0;

      if (balanceResponse.success && balanceResponse.data) {
        currentBalance = balanceResponse.data.balance_qnk || 0;
      } else {
        // Authentication failed or balance not available - try cached balance
        console.warn('⚠️ Balance query failed, using cached balance:', balanceResponse.error);
        const cachedBalance = localStorage.getItem('cachedBalance');
        if (cachedBalance) {
          currentBalance = parseFloat(cachedBalance);
          console.log('💰 Using cached balance for Nitro purchase:', currentBalance);
        } else {
          alert('❌ Unable to fetch wallet balance. Please log in with your wallet password first.');
          return;
        }
      }
      if (currentBalance < qugCost) {
        alert(`Insufficient QUG balance!\n\nRequired: ${qugCost.toFixed(2)} QUG\nAvailable: ${currentBalance.toFixed(2)} QUG`);
        return;
      }

      // Send transaction to burn QUG for Nitro Points
      // We send to a burn address (all zeros) to permanently remove the QUG
      const burnAddress = 'qnk0000000000000000000000000000000000000000000000000000000000000000';

      const txResponse = await qnkAPI.sendTransaction(
        walletAddress,
        burnAddress,
        qugCost,
        `Nitro Points Purchase: ${purchaseAmount} points`
      );

      if (!txResponse.success) {
        alert(`❌ Transaction failed: ${txResponse.error || 'Unknown error'}\n\nYour QUG was not deducted.`);
        return;
      }

      // Transaction successful - add Nitro Points (per wallet address)
      const newBalance = Math.min(nitroPoints + purchaseAmount, 1500);
      setNitroPoints(newBalance);
      localStorage.setItem(`nitroPoints_${walletAddress}`, newBalance.toString());

      // Dispatch custom event to notify other components (e.g., DexScreen)
      window.dispatchEvent(new Event('nitroPointsUpdated'));

      setShowNitroPurchase(false);
      setPurchaseAmount(500);

      // Show success modal instead of alert
      setSuccessModalData({
        points: purchaseAmount,
        qugCost: qugCost,
        newBalance: newBalance,
        txId: txResponse.data?.transaction_id
      });
      setShowSuccessModal(true);
    } catch (error) {
      console.error('Failed to purchase Nitro points:', error);
      alert(`❌ Failed to purchase Nitro points: ${error instanceof Error ? error.message : 'Unknown error'}\n\nPlease try again.`);
    }
  };

  // v3.9.5-beta: Subscript zero notation for tiny prices (like DEXScreener)
  const SUBSCRIPT_DIGITS = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉'];
  const toSubscript = (n: number): string => {
    return String(n).split('').map(d => SUBSCRIPT_DIGITS[parseInt(d)] || d).join('');
  };
  const formatPrice = (price: number) => {
    if (price >= 1000000) return `$${(price / 1000000).toFixed(2)}M`;
    if (price >= 1000) return `$${(price / 1000).toFixed(2)}K`;
    if (price >= 1) return `$${price.toFixed(2)}`;
    if (price >= 0.01) return `$${price.toFixed(4)}`;
    if (price >= 0.0001) return `$${price.toFixed(6)}`;
    if (price <= 0) return '$0.00';
    // Tiny prices: subscript zero notation ($0.0₇38)
    const str = price.toFixed(20);
    const afterDot = str.split('.')[1] || '';
    let zeroCount = 0;
    for (const ch of afterDot) {
      if (ch === '0') zeroCount++;
      else break;
    }
    const sigDigits = afterDot.slice(zeroCount, zeroCount + 4).replace(/0+$/, '') || '0';
    if (zeroCount >= 2) return `$0.0${toSubscript(zeroCount)}${sigDigits}`;
    return `$${price.toFixed(8).replace(/0+$/, '')}`;
  };

  if (loading) {
    return (
      <div
        className="backdrop-blur-xl border-b px-6 py-3"
        style={{
          background: 'linear-gradient(135deg, rgba(12, 12, 20, 0.95) 0%, rgba(20, 20, 32, 0.95) 100%)',
          borderColor: 'rgba(212, 175, 55, 0.15)',
        }}
      >
        <div className="flex items-center justify-center">
          <div className="text-amber-300/50 text-sm">Loading tokens...</div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="backdrop-blur-xl border-b px-6 py-3 relative z-40"
      style={{
        background: 'linear-gradient(135deg, rgba(12, 12, 20, 0.95) 0%, rgba(20, 20, 32, 0.95) 100%)',
        borderColor: 'rgba(212, 175, 55, 0.15)',
        boxShadow: '0 4px 15px rgba(212, 175, 55, 0.1)'
      }}
    >
      <div className="flex items-center gap-4">
        {/* Nitro Points Display */}
        <motion.div
          className="flex-shrink-0 relative group cursor-pointer"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setShowNitroPurchase(true)}
        >
          <motion.div
            className="absolute -inset-0.5 bg-gradient-to-r from-orange-500 via-yellow-500 to-red-500 rounded-xl blur-md"
            animate={{
              opacity: [0.3, 0.6, 0.3],
              scale: [1, 1.05, 1]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />

          <div
            className="relative flex items-center gap-2 px-4 py-2 rounded-xl border-2"
            style={{
              background: 'linear-gradient(135deg, rgba(255, 140, 0, 0.2), rgba(255, 215, 0, 0.1))',
              borderColor: 'rgba(255, 165, 0, 0.4)',
              boxShadow: '0 0 15px rgba(255, 140, 0, 0.3)'
            }}
          >
            <motion.div
              animate={{
                rotate: [0, 10, -10, 0],
                scale: [1, 1.2, 1]
              }}
              transition={{
                duration: 0.5,
                repeat: Infinity,
                repeatDelay: 2
              }}
            >
              <Zap className="w-5 h-5 text-orange-400" />
            </motion.div>

            <div className="flex flex-col">
              <span className="text-xs text-orange-300/70 font-medium">Nitro Points</span>
              <motion.span
                key={nitroPoints}
                initial={{ scale: 1.05, color: '#FFA500' }}
                animate={{ scale: 1, color: '#FFA500' }}
                transition={{ duration: 0.2, ease: "easeOut" }}
                className="text-lg font-bold bg-gradient-to-r from-orange-400 to-yellow-500 bg-clip-text text-transparent"
              >
                {nitroPoints.toLocaleString()} / 1,500
              </motion.span>
            </div>

            <Flame className="w-4 h-4 text-red-500" />
          </div>
        </motion.div>

        {/* Scroll Left Button */}
        <AnimatePresence>
          {canScrollLeft && (
            <motion.button
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              onClick={() => scroll('left')}
              className="flex-shrink-0 p-2 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 transition-all"
            >
              <ChevronLeft className="w-4 h-4 text-amber-400" />
            </motion.button>
          )}
        </AnimatePresence>

        {/* Token List - Horizontal Scroll */}
        <div
          ref={scrollContainerRef}
          className="flex-1 overflow-x-auto scrollbar-hide"
          style={{
            scrollbarWidth: 'none',
            msOverflowStyle: 'none'
          }}
        >
          <div className="flex items-center gap-4">
            {tokens.map((token, index) => (
              <motion.button
                key={token.id}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                onClick={() => onTokenClick?.(token)}
                className="flex-shrink-0 group relative"
              >
                {/* Nitro Boost Glow Effect */}
                {token.isNitroBoost && (
                  <motion.div
                    className="absolute -inset-1 rounded-xl blur-md opacity-50"
                    style={{
                      background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.5), rgba(255, 215, 0, 0.3))'
                    }}
                    animate={{
                      opacity: [0.3, 0.6, 0.3],
                      scale: [1, 1.05, 1]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  />
                )}

                <div
                  className="relative flex items-center gap-3 px-4 py-2 rounded-xl transition-all"
                  style={{
                    background: token.isNitroBoost
                      ? 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.1))'
                      : 'rgba(30, 20, 50, 0.5)',
                    border: token.isNitroBoost
                      ? '2px solid rgba(212, 175, 55, 0.4)'
                      : '1px solid rgba(100, 100, 120, 0.2)',
                    boxShadow: token.isNitroBoost
                      ? '0 0 15px rgba(212, 175, 55, 0.2)'
                      : 'none'
                  }}
                  onMouseEnter={(e) => {
                    if (token.isNitroBoost) {
                      e.currentTarget.style.background = 'linear-gradient(135deg, rgba(212, 175, 55, 0.3), rgba(255, 215, 0, 0.2))';
                      e.currentTarget.style.borderColor = 'rgba(212, 175, 55, 0.6)';
                      e.currentTarget.style.boxShadow = '0 0 25px rgba(212, 175, 55, 0.4)';
                    } else {
                      e.currentTarget.style.background = 'rgba(50, 40, 70, 0.7)';
                      e.currentTarget.style.borderColor = 'rgba(150, 150, 170, 0.3)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (token.isNitroBoost) {
                      e.currentTarget.style.background = 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.1))';
                      e.currentTarget.style.borderColor = 'rgba(212, 175, 55, 0.4)';
                      e.currentTarget.style.boxShadow = '0 0 15px rgba(212, 175, 55, 0.2)';
                    } else {
                      e.currentTarget.style.background = 'rgba(30, 20, 50, 0.5)';
                      e.currentTarget.style.borderColor = 'rgba(100, 100, 120, 0.2)';
                    }
                  }}
                >
                  {/* Token Icon */}
                  <TokenIcon symbol={token.symbol} icon={token.icon} size={28} />

                  {/* Token Info */}
                  <div className="flex flex-col items-start min-w-[120px]">
                    <div className="flex items-center gap-2">
                      <span className={`font-bold text-sm ${token.isNitroBoost ? 'bg-gradient-to-r from-amber-400 to-yellow-500 bg-clip-text text-transparent' : 'text-white'}`}>
                        {token.symbol}
                      </span>
                      {token.isNitroBoost && (
                        <motion.div
                          animate={{
                            rotate: [0, 10, -10, 0],
                            scale: [1, 1.1, 1]
                          }}
                          transition={{
                            duration: 0.5,
                            repeat: Infinity,
                            repeatDelay: 2
                          }}
                        >
                          <Flame className="w-3 h-3 text-amber-400" />
                        </motion.div>
                      )}
                    </div>
                    <div className="text-xs text-amber-300/70 font-medium">
                      {formatPrice(token.price)}
                    </div>
                  </div>

                  {/* 24h Change */}
                  <div className={`flex items-center gap-1 px-2 py-1 rounded-md ${
                    token.change24h > 0
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-red-500/20 text-red-400'
                  }`}>
                    {token.change24h > 0 ? (
                      <TrendingUp className="w-3 h-3" />
                    ) : (
                      <TrendingDown className="w-3 h-3" />
                    )}
                    <span className="text-xs font-bold">
                      {token.change24h > 0 ? '+' : ''}{token.change24h.toFixed(2)}%
                    </span>
                  </div>

                  {/* Nitro Boost Badge */}
                  {token.isNitroBoost && (
                    <motion.div
                      className="absolute -top-1 -right-1 px-2 py-0.5 rounded-full text-xs font-bold"
                      style={{
                        background: 'linear-gradient(135deg, #D4AF37, #FFD700)',
                        color: '#1a1a2e',
                        boxShadow: '0 0 10px rgba(212, 175, 55, 0.5)'
                      }}
                      animate={{
                        boxShadow: [
                          '0 0 10px rgba(212, 175, 55, 0.5)',
                          '0 0 20px rgba(212, 175, 55, 0.8)',
                          '0 0 10px rgba(212, 175, 55, 0.5)'
                        ]
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    >
                      <Zap className="w-3 h-3" />
                    </motion.div>
                  )}
                </div>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Scroll Right Button */}
        <AnimatePresence>
          {canScrollRight && (
            <motion.button
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              onClick={() => scroll('right')}
              className="flex-shrink-0 p-2 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 transition-all"
            >
              <ChevronRight className="w-4 h-4 text-amber-400" />
            </motion.button>
          )}
        </AnimatePresence>
      </div>

      {/* CSS for hiding scrollbar */}
      <style>{`
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
      `}</style>

      {/* Nitro Purchase Modal - Rendered via Portal */}
      {showNitroPurchase && createPortal(
        <AnimatePresence>
          <div className="fixed inset-0 z-[9999] overflow-y-auto">
            <div className="flex min-h-full items-center justify-center p-4">
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setShowNitroPurchase(false)}
                className="fixed inset-0 bg-black/80 backdrop-blur-sm"
              />

              <motion.div
                initial={{ opacity: 0, scale: 0.95, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 20 }}
                className="relative w-full max-w-md my-8"
              >
              <div className="relative group">
                <motion.div
                  className="absolute -inset-0.5 bg-gradient-to-r from-orange-500 via-yellow-500 to-red-500 rounded-2xl blur-xl"
                  animate={{
                    opacity: [0.5, 0.8, 0.5],
                    scale: [1, 1.05, 1],
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />

                <div className="relative bg-black border-2 border-orange-500/40 rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-yellow-500 bg-clip-text text-transparent flex items-center gap-2">
                      <Zap className="w-6 h-6 text-orange-400" />
                      Purchase Nitro Points
                    </h2>
                    <Flame className="w-6 h-6 text-red-500" />
                  </div>

                  {/* Current Balance */}
                  <div className="mb-6 p-4 rounded-xl border border-orange-500/30 bg-orange-500/10">
                    <div className="text-sm text-orange-300/70 mb-1">Current Balance</div>
                    <div className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-yellow-500 bg-clip-text text-transparent">
                      {nitroPoints.toLocaleString()} / 1,500 Points
                    </div>
                  </div>

                  {/* Purchase Amount Selector */}
                  <div className="mb-6">
                    <label className="block text-sm text-orange-300/70 mb-2">
                      Purchase Amount
                    </label>
                    <input
                      type="range"
                      min="100"
                      max="1500"
                      step="100"
                      value={purchaseAmount}
                      onChange={(e) => setPurchaseAmount(parseInt(e.target.value))}
                      className="w-full"
                      style={{
                        accentColor: '#FF8C00'
                      }}
                    />
                    <div className="flex justify-between mt-2">
                      <span className="text-xs text-orange-300/50">100</span>
                      <motion.span
                        key={purchaseAmount}
                        initial={{ scale: 1.05 }}
                        animate={{ scale: 1 }}
                        transition={{ duration: 0.2, ease: "easeOut" }}
                        className="text-lg font-bold bg-gradient-to-r from-orange-400 to-yellow-500 bg-clip-text text-transparent"
                      >
                        {purchaseAmount} Points
                      </motion.span>
                      <span className="text-xs text-orange-300/50">1,500</span>
                    </div>
                  </div>

                  {/* Cost Preview */}
                  <div className="mb-6 p-4 rounded-xl border border-yellow-500/30 bg-yellow-500/10">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-yellow-300/70">Cost</span>
                      <span className="text-xl font-bold text-yellow-400">
                        {(purchaseAmount * 0.01).toFixed(2)} QUG
                      </span>
                    </div>
                    <div className="text-xs text-yellow-300/50 mt-1">
                      1 Point = 0.01 QUG
                    </div>
                  </div>

                  {/* Benefits */}
                  <div className="mb-6 space-y-2">
                    <div className="text-sm text-orange-300/70 mb-2">Nitro Benefits:</div>
                    <div className="flex items-start gap-2 text-sm text-white/80">
                      <Zap className="w-4 h-4 text-orange-400 mt-0.5 flex-shrink-0" />
                      <span>Boost tokens to top of DEX listing</span>
                    </div>
                    <div className="flex items-start gap-2 text-sm text-white/80">
                      <TrendingUp className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                      <span>Increase token visibility and trading volume</span>
                    </div>
                    <div className="flex items-start gap-2 text-sm text-white/80">
                      <Flame className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                      <span>Premium visual effects and animations</span>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-3">
                    <motion.button
                      onClick={() => setShowNitroPurchase(false)}
                      className="flex-1 px-4 py-3 rounded-xl bg-gray-700 hover:bg-gray-600 text-white font-medium transition-all"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      Cancel
                    </motion.button>
                    <motion.button
                      onClick={handlePurchaseNitro}
                      className="flex-1 px-4 py-3 rounded-xl bg-gradient-to-r from-orange-500 to-red-500 text-white font-bold transition-all flex items-center justify-center gap-2"
                      style={{
                        boxShadow: '0 0 20px rgba(255, 140, 0, 0.5)'
                      }}
                      whileHover={{
                        scale: 1.02,
                        boxShadow: '0 0 30px rgba(255, 140, 0, 0.7)'
                      }}
                      whileTap={{ scale: 0.98 }}
                      disabled={nitroPoints + purchaseAmount > 1500}
                    >
                      <Zap className="w-5 h-5" />
                      Purchase
                    </motion.button>
                  </div>

                  {/* Max Balance Warning */}
                  {nitroPoints + purchaseAmount > 1500 && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-4 p-3 rounded-lg bg-red-500/20 border border-red-500/30 text-sm text-red-300"
                    >
                      ⚠️ This purchase would exceed max balance of 1,500 points
                    </motion.div>
                  )}
                </div>
              </div>
            </motion.div>
            </div>
          </div>
        </AnimatePresence>,
        document.body
      )}

      {/* Success Modal */}
      <NitroSuccessModal
        isOpen={showSuccessModal}
        onClose={() => setShowSuccessModal(false)}
        type="purchase"
        data={successModalData}
      />
    </div>
  );
});

export default TokenBar;
