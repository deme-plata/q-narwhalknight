import { useState, useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Coins, Send, ChevronDown, ChevronUp, Loader2, AlertCircle, TrendingUp, DollarSign, PieChart, Lock, Unlock, Gift, Timer, Award, Search, ArrowUpDown, ArrowUp, ArrowDown, SlidersHorizontal } from 'lucide-react';
import { qnkAPI } from '../services/api';
import VaultModal from './VaultModal';
import ForgeModal from './ForgeModal';

// v3.7.1: Sort options for the token list
type SortField = 'symbol' | 'balance' | 'valueUsd' | 'change24h' | 'volume24h' | 'liquidity';
type SortDirection = 'asc' | 'desc';

interface CustomToken {
  symbol: string;
  name: string;
  balance: number;
  contractAddress: string;
  decimals?: number;
  priceUsd?: number;
  valueUsd?: number;
  change24h?: number;
  volume24h?: number;   // v3.6.12: 24h trading volume
  liquidity?: number;   // v3.6.12: Pool liquidity
}

interface StakePosition {
  amount: number;
  tier: string;
  apy_bps: number;
  unlock_time: number;
  pending_rewards: number;
}

interface CustomTokensCardProps {
  onSendToken: (tokenSymbol: string, contractAddress: string) => void;
}

const STAKING_TIERS = [
  { name: 'Bronze', days: 7, apy: 5, color: 'amber' },
  { name: 'Silver', days: 30, apy: 10, color: 'gray' },
  { name: 'Gold', days: 90, apy: 15, color: 'yellow' },
  { name: 'Diamond', days: 180, apy: 25, color: 'cyan' },
];

export default function CustomTokensCard({ onSendToken }: CustomTokensCardProps) {
  // v2.9.12-beta: Initialize from localStorage if in cooldown period
  const [customTokens, setCustomTokens] = useState<CustomToken[]>(() => {
    try {
      const now = Date.now();
      const cooldownUntil = parseInt(localStorage.getItem('customTokensCooldownUntil') || '0');
      if (now < cooldownUntil) {
        // We're in cooldown - restore cached state
        const cached = localStorage.getItem('customTokensCache');
        if (cached) {
          console.log('🔄 [CustomTokens v2.9.12] Restored from cache during cooldown');
          return JSON.parse(cached);
        }
      }
    } catch (e) {
      console.warn('[CustomTokens] Failed to restore from cache');
    }
    return [];
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(true);

  // v3.7.1: Sorting and filtering state
  const [sortField, setSortField] = useState<SortField>('valueUsd');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [filterText, setFilterText] = useState('');
  const [showFilters, setShowFilters] = useState(false);

  // v4.2.0: VAULT RWA modal state
  const [showVaultModal, setShowVaultModal] = useState(false);
  // v5.1.0: FORGE RWA modal state
  const [showForgeModal, setShowForgeModal] = useState(false);

  // Staking modal state
  const [stakingToken, setStakingToken] = useState<CustomToken | null>(null);
  const [stakeAmount, setStakeAmount] = useState('');
  const [selectedTier, setSelectedTier] = useState(0);
  const [stakeLoading, setStakeLoading] = useState(false);
  const [stakeError, setStakeError] = useState<string | null>(null);
  const [stakeSuccess, setStakeSuccess] = useState<string | null>(null);
  const [stakePositions, setStakePositions] = useState<Record<string, StakePosition>>({});
  // v3.6.18: Track token's reflection rate from contract (set by owner)
  const [tokenReflectionRate, setTokenReflectionRate] = useState<number | null>(null);
  // v3.6.18: Track stake history showing profits
  const [stakeHistory, setStakeHistory] = useState<Array<{
    timestamp: number;
    amount: number;
    reward: number;
    type: 'stake' | 'unstake' | 'reward';
  }>>([]);

  // v2.9.8-beta: Track DEX swap cooldowns per token to prevent stale API data overwriting correct balance
  // Maps token symbol (uppercase) to the timestamp until which we should preserve local balance
  const swapCooldownRef = useRef<Record<string, number>>({});

  // v2.9.13-beta: Track which balances we've set locally (to detect stale backend data)
  const localBalancesRef = useRef<Record<string, number>>({});

  // v2.9.14-beta: Track if we've already processed a DEX swap for each token in this cooldown period
  // Once we process ONE DEX swap event, reject ALL others until cooldown expires
  const processedSwapsRef = useRef<Record<string, boolean>>({});

  // v2.4.1: Only show tokens with balance > 0
  // v2.9.15-beta: During cooldown, OVERRIDE state with localStorage protected values
  // v3.7.1: Added filtering and sorting
  const tokensWithBalance = useMemo(() => {
    const now = Date.now();
    const globalCooldownUntil = parseInt(localStorage.getItem('customTokensCooldownUntil') || '0');
    const isInCooldown = now < globalCooldownUntil;

    let tokens: CustomToken[];

    if (isInCooldown) {
      // During cooldown, use protected balances from localStorage as source of truth
      try {
        const protectedBalances = JSON.parse(localStorage.getItem('protectedTokenBalances') || '{}');
        console.log('🔒 [CustomTokens v2.9.15] In cooldown - using protected balances:', protectedBalances);

        tokens = customTokens
          .map(token => {
            const tokenUpper = token.symbol?.toUpperCase() || '';
            const protectedData = protectedBalances[tokenUpper];
            if (protectedData && protectedData.until > now) {
              // Use protected balance instead of state balance
              console.log(`🔒 [CustomTokens v2.9.15] Overriding ${token.symbol}: state=${token.balance}, protected=${protectedData.balance}`);
              return { ...token, balance: protectedData.balance };
            }
            return token;
          })
          .filter(token => token.balance > 0);
      } catch (e) {
        console.warn('[CustomTokens] Failed to read protected balances');
        tokens = customTokens.filter(token => token.balance > 0);
      }
    } else {
      tokens = customTokens.filter(token => token.balance > 0);
    }

    // v3.7.1: Apply text filter
    if (filterText.trim()) {
      const searchLower = filterText.toLowerCase();
      tokens = tokens.filter(token =>
        token.symbol.toLowerCase().includes(searchLower) ||
        token.name.toLowerCase().includes(searchLower) ||
        token.contractAddress.toLowerCase().includes(searchLower)
      );
    }

    // v3.7.1: Apply sorting
    tokens.sort((a, b) => {
      let aVal: number | string;
      let bVal: number | string;

      switch (sortField) {
        case 'symbol':
          aVal = a.symbol.toLowerCase();
          bVal = b.symbol.toLowerCase();
          break;
        case 'balance':
          aVal = a.balance;
          bVal = b.balance;
          break;
        case 'valueUsd':
          aVal = a.valueUsd || 0;
          bVal = b.valueUsd || 0;
          break;
        case 'change24h':
          aVal = a.change24h || 0;
          bVal = b.change24h || 0;
          break;
        case 'volume24h':
          aVal = a.volume24h || 0;
          bVal = b.volume24h || 0;
          break;
        case 'liquidity':
          aVal = a.liquidity || 0;
          bVal = b.liquidity || 0;
          break;
        default:
          aVal = a.valueUsd || 0;
          bVal = b.valueUsd || 0;
      }

      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortDirection === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
    });

    return tokens;
  }, [customTokens, filterText, sortField, sortDirection]);

  // v2.4.1: Calculate total portfolio value
  const totalPortfolioValue = useMemo(() => {
    return tokensWithBalance.reduce((total, token) => total + (token.valueUsd || 0), 0);
  }, [tokensWithBalance]);

  // v2.4.1: Count tokens with positive value
  const tokensWithValue = useMemo(() => {
    return tokensWithBalance.filter(token => (token.valueUsd || 0) > 0).length;
  }, [tokensWithBalance]);

  useEffect(() => {
    fetchCustomTokens();

    // v3.6.18: Refresh every 15 seconds for faster token updates
    const interval = setInterval(fetchCustomTokens, 15000);

    // v1.4.10-beta: Listen for token balance updates via SSE for instant refresh
    // v2.4.2: Enhanced to trigger full refresh for new tokens
    // v2.9.8-beta: Added cooldown to prevent stale API data overwriting
    // v2.9.14-beta: AGGRESSIVE FIX - only accept FIRST DEX event per token during cooldown
    const handleTokenBalanceUpdate = (event: CustomEvent) => {
      const { tokenSymbol, newBalance, reason, source } = event.detail;
      console.log('🪙 [CustomTokens v2.9.14] Balance update received:', { tokenSymbol, newBalance, reason, source });

      // Guard against undefined tokenSymbol
      if (!tokenSymbol) {
        console.warn('⚠️ [CustomTokens] tokenSymbol is undefined in event, ignoring');
        return;
      }

      const tokenUpper = tokenSymbol.toUpperCase();
      const now = Date.now();

      // v2.9.13-beta: Check if we're in cooldown FIRST, BEFORE processing any events
      const globalCooldownUntil = parseInt(localStorage.getItem('customTokensCooldownUntil') || '0');
      const isInCooldown = now < globalCooldownUntil;

      // v2.9.11-beta: If this is a DEX swap from the FRONTEND (not backend SSE)
      const isDexSwap = reason === 'dex-swap-add' || reason === 'dex-swap-deduct';

      // v2.9.14-beta: AGGRESSIVE PROTECTION - If we're in cooldown and already processed
      // a swap for this token, reject ALL events regardless of source or reason
      if (isInCooldown && processedSwapsRef.current[tokenUpper]) {
        console.log(`🛡️ [CustomTokens v2.9.14] BLOCKED - already processed ${tokenUpper} during cooldown, rejecting ALL events (reason=${reason}, value=${newBalance})`);
        return;
      }

      // v2.9.14-beta: For non-DEX events during cooldown, always block
      if (isInCooldown && !isDexSwap) {
        console.log(`🛡️ [CustomTokens v2.9.14] BLOCKED non-DEX event for ${tokenUpper} during cooldown (reason=${reason})`);
        return;
      }

      // v2.9.14-beta: For DEX events, set up protection and mark as processed
      if (isDexSwap) {
        const cooldownUntil = now + 180000; // v2.9.17: 3 minute cooldown (increased from 60s)
        swapCooldownRef.current[tokenUpper] = cooldownUntil;

        // v2.9.13-beta: Track local balance to detect stale backend data
        localBalancesRef.current[tokenUpper] = newBalance;

        // v2.9.14-beta: Mark this token as already processed - reject all future events
        processedSwapsRef.current[tokenUpper] = true;

        // Store protected balance in localStorage
        const protectedBalances = JSON.parse(localStorage.getItem('protectedTokenBalances') || '{}');
        protectedBalances[tokenUpper] = { balance: newBalance, until: cooldownUntil };
        localStorage.setItem('protectedTokenBalances', JSON.stringify(protectedBalances));

        // v2.9.11-beta: Set GLOBAL cooldown to block ALL fetches
        localStorage.setItem('customTokensCooldownUntil', cooldownUntil.toString());

        console.log(`🔒 [CustomTokens v2.9.17] Protected ${tokenUpper} balance ${newBalance} for 3 MINUTES + marked as processed (NO MORE EVENTS ACCEPTED)`);

        // v2.9.14-beta: Schedule cleanup of processedSwaps flag after cooldown
        setTimeout(() => {
          delete processedSwapsRef.current[tokenUpper];
          console.log(`🔓 [CustomTokens v2.9.17] Cleared processed flag for ${tokenUpper} - new events accepted`);
        }, 180000);
      }

      // Check if token exists in current list
      setCustomTokens(prev => {
        console.log('🔍 [CustomTokens] Checking tokens:', prev.map(t => t.symbol));
        const tokenExists = prev.some(token => token.symbol?.toUpperCase() === tokenUpper);
        console.log(`🔍 [CustomTokens] Token ${tokenSymbol} exists in list: ${tokenExists}`);

        if (!tokenExists && newBalance > 0) {
          // Token doesn't exist and has a balance - this is a first-time buy!
          // Trigger full refresh to fetch the new token's metadata
          console.log(`🆕 [CustomTokens] New token ${tokenSymbol} detected, triggering full refresh...`);
          // v3.6.18: Increased delay to 2s to ensure backend has processed the swap
          // Also trigger immediate fetch + delayed fetch for faster response
          fetchCustomTokens(); // Try immediately first
          setTimeout(() => fetchCustomTokens(), 2000); // Retry after 2s for backend sync
          return prev;
        }

        // Update the balance for the matching token immediately
        const updated = prev.map(token => {
          if (token.symbol?.toUpperCase() === tokenUpper) {
            const newValueUsd = (token.priceUsd || 0) * newBalance;
            console.log(`✅ [CustomTokens] Updated ${token.symbol} balance: ${token.balance} → ${newBalance}`);
            return { ...token, balance: newBalance, valueUsd: newValueUsd };
          }
          return token;
        });
        console.log('🪙 [CustomTokens] After update:', updated.map(t => ({ symbol: t.symbol, balance: t.balance })));

        // v2.9.12-beta: Cache the updated state to localStorage for persistence across remounts
        if (isDexSwap) {
          try {
            localStorage.setItem('customTokensCache', JSON.stringify(updated));
            console.log('💾 [CustomTokens v2.9.12] Cached updated state to localStorage');
          } catch (e) {
            console.warn('[CustomTokens] Failed to cache state');
          }
        }

        return updated;
      });
    };

    window.addEventListener('token-balance-updated', handleTokenBalanceUpdate as EventListener);

    // v4.0.1: Listen for token price updates via SSE for INSTANT value display on dashboard
    // Previously prices only updated every 15s poll, causing tokens to show $0 after a swap.
    const handleTokenPriceUpdate = (event: CustomEvent) => {
      const { token_symbol, token_address, price, change_24h, volume_24h } = event.detail;
      if (!token_symbol && !token_address) return;
      const priceNum = parseFloat(price) || 0;
      if (priceNum <= 0) return;

      setCustomTokens(prev => {
        let updated = false;
        const result = prev.map(token => {
          const matchSymbol = token_symbol && token.symbol?.toUpperCase() === token_symbol.toUpperCase();
          const matchAddr = token_address && token.contractAddress &&
            (token.contractAddress.toLowerCase() === token_address.toLowerCase() ||
             token.contractAddress.toLowerCase().includes(token_address.toLowerCase().replace('qnk', '')));
          if (matchSymbol || matchAddr) {
            const newValueUsd = token.balance * priceNum;
            if (token.priceUsd !== priceNum) {
              updated = true;
              return { ...token, priceUsd: priceNum, valueUsd: newValueUsd, change24h: change_24h || token.change24h, volume24h: volume_24h || token.volume24h };
            }
          }
          return token;
        });
        if (updated) {
          console.log(`💰 [CustomTokens v4.0.1] Instant price update: ${token_symbol} = $${priceNum.toFixed(6)}`);
        }
        return updated ? result : prev;
      });
    };

    window.addEventListener('token-price-updated', handleTokenPriceUpdate as EventListener);

    return () => {
      clearInterval(interval);
      window.removeEventListener('token-balance-updated', handleTokenBalanceUpdate as EventListener);
      window.removeEventListener('token-price-updated', handleTokenPriceUpdate as EventListener);
    };
  }, []);

  const fetchCustomTokens = async () => {
    // v2.9.11-beta: Check global cooldown FIRST - if ANY token is protected, skip entire fetch
    const now = Date.now();
    const globalCooldownUntil = parseInt(localStorage.getItem('customTokensCooldownUntil') || '0');
    if (now < globalCooldownUntil) {
      const remainingSeconds = Math.round((globalCooldownUntil - now) / 1000);
      console.log(`⏸️ [CustomTokens v2.9.11] SKIPPING FETCH - global cooldown active for ${remainingSeconds}s more`);
      // v3.6.14: Clear stuck cooldowns (shouldn't be more than 5 minutes)
      if (remainingSeconds > 300) {
        console.warn(`⚠️ [CustomTokens v3.6.14] Cooldown is stuck at ${remainingSeconds}s - clearing!`);
        localStorage.removeItem('customTokensCooldownUntil');
        localStorage.removeItem('protectedTokenBalances');
      } else {
        setLoading(false);
        return;
      }
    }

    try {
      setLoading(true);
      setError(null);

      console.log('🔄 [CustomTokens v3.6.14] Fetching multi-token balance...');
      const response = await qnkAPI.getMultiTokenBalance();
      console.log('📦 [CustomTokens v3.6.14] API Response:', response);

      if (response.success && response.data && response.data.tokens) {
        const tokensObj = response.data.tokens;
        console.log('🪙 [CustomTokens v3.6.14] Tokens from API:', Object.keys(tokensObj));

        // v2.9.9-beta: Get protected balances from localStorage
        // const now = Date.now(); // already declared above
        let protectedBalances: Record<string, { balance: number; until: number }> = {};
        try {
          protectedBalances = JSON.parse(localStorage.getItem('protectedTokenBalances') || '{}');
          // Clean up expired entries
          let hasExpired = false;
          for (const [symbol, data] of Object.entries(protectedBalances)) {
            if (data.until < now) {
              delete protectedBalances[symbol];
              hasExpired = true;
            }
          }
          if (hasExpired) {
            localStorage.setItem('protectedTokenBalances', JSON.stringify(protectedBalances));
          }
        } catch (e) {
          console.warn('[CustomTokens] Failed to parse protectedTokenBalances');
        }

        console.log(`🔍 [CustomTokens v2.9.9] Protected balances:`, protectedBalances);

        // Filter out native tokens (QUG, QUGUSD) and extract custom tokens
        const customTokensList: CustomToken[] = [];

        for (const [symbol, tokenData] of Object.entries(tokensObj)) {
          const upperSymbol = symbol.toUpperCase();

          // Skip native tokens
          if (upperSymbol === 'QUG' || upperSymbol === 'QUGUSD') {
            continue;
          }

          // Add custom token
          const token = tokenData as any;
          const tokenName = token.name || upperSymbol;

          // v3.0.7-beta: Validate token data to filter out invalid entries (like 404 HTML responses)
          // Skip tokens with invalid symbols or names (contains HTML, too long, or empty)
          if (!upperSymbol ||
              upperSymbol.length > 20 ||
              upperSymbol.includes('<') ||
              upperSymbol.includes('>') ||
              tokenName.includes('<html') ||
              tokenName.includes('<!DOCTYPE') ||
              tokenName.includes('404') ||
              tokenName.length > 100) {
            console.warn('⚠️ [CustomTokens v3.0.7] Skipping invalid token:', { symbol: upperSymbol, name: tokenName });
            continue;
          }

          // v4.0.10: ALWAYS use the pre-formatted `balance` field from the API.
          // The backend divides by 1e24 and returns a human-readable string like "3190649616.00579900".
          // balance_base_units is a u128 serialized as STRING which loses precision in JS Number.
          let apiBalance = parseFloat(token.balance || '0');
          if (isNaN(apiBalance)) apiBalance = 0;
          console.log(`💰 [CustomTokens v4.0.10] ${upperSymbol}: balance=${token.balance}, parsed=${apiBalance}`);

          // v2.9.9-beta: Check if this token has a protected balance (from recent DEX swap)
          const protectedData = protectedBalances[upperSymbol];
          const isProtected = protectedData && protectedData.until > now;

          // Use protected balance if available, otherwise use API balance
          const finalBalance = isProtected ? protectedData.balance : apiBalance;

          if (isProtected && protectedData.balance !== apiBalance) {
            console.log(`🔒 [CustomTokens v2.9.9] Using PROTECTED balance for ${upperSymbol}: ${protectedData.balance} (API returned ${apiBalance})`);
          }

          customTokensList.push({
            symbol: upperSymbol,
            name: tokenName,
            balance: finalBalance,
            contractAddress: token.contract_address || '',
            decimals: token.decimals || 8,
            priceUsd: 0,
            valueUsd: 0,
            change24h: 0,
          });
        }

        // v2.4.1: Fetch prices for tokens with balance
        const tokensWithBalance = customTokensList.filter(t => t.balance > 0);

        // Fetch prices in parallel
        const pricePromises = tokensWithBalance.map(async (token) => {
          try {
            const priceResponse = await fetch(`/api/v1/defi/oracle/price/${token.contractAddress}`);
            if (priceResponse.ok) {
              const priceData = await priceResponse.json();
              if (priceData.success && priceData.data) {
                return {
                  contractAddress: token.contractAddress,
                  price: priceData.data.price || 0,
                  change24h: priceData.data.change_24h || 0,
                };
              }
            }
          } catch (e) {
            console.warn(`Failed to fetch price for ${token.symbol}`);
          }
          return { contractAddress: token.contractAddress, price: 0, change24h: 0 };
        });

        const prices = await Promise.all(pricePromises);
        const priceMap = new Map(prices.map(p => [p.contractAddress, p]));

        // v3.6.12: Fetch liquidity pools to get volume and liquidity data
        // v3.6.14: FIX - Backend returns total_liquidity as raw lp_token_supply (u128 base units)
        // Need to divide by 1e8 (token decimals) to get human-readable amount
        let poolsData: Map<string, { volume24h: number; liquidity: number }> = new Map();
        try {
          const poolsResponse = await qnkAPI.getLiquidityPools();
          if (poolsResponse.success && poolsResponse.data) {
            // Aggregate volume and liquidity by token address
            for (const pool of poolsResponse.data) {
              const tokenA = pool.token_a_address || pool.tokenA?.address || pool.token0;
              const tokenB = pool.token_b_address || pool.tokenB?.address || pool.token1;
              // v3.6.14: total_liquidity is raw u128, divide by 1e8 to get human-readable
              // Then multiply by reserve values to estimate USD liquidity
              const rawLiquidity = parseFloat(pool.total_liquidity || pool.total_liquidity_usd || pool.liquidity_usd || '0');
              // If the value is absurdly large (>1e15), it's likely in base units
              const poolLiquidity = rawLiquidity > 1e15 ? rawLiquidity / 1e8 : rawLiquidity;
              const rawVolume = parseFloat(pool.volume_24h || pool.volume24h || '0');
              const poolVolume = rawVolume > 1e15 ? rawVolume / 1e8 : rawVolume;

              // Add to token A
              if (tokenA) {
                const existing = poolsData.get(tokenA) || { volume24h: 0, liquidity: 0 };
                poolsData.set(tokenA, {
                  volume24h: existing.volume24h + poolVolume,
                  liquidity: existing.liquidity + poolLiquidity,
                });
              }
              // Add to token B
              if (tokenB) {
                const existing = poolsData.get(tokenB) || { volume24h: 0, liquidity: 0 };
                poolsData.set(tokenB, {
                  volume24h: existing.volume24h + poolVolume,
                  liquidity: existing.liquidity + poolLiquidity,
                });
              }
            }
            console.log('💧 [CustomTokens] Loaded pool data for', poolsData.size, 'tokens');
          }
        } catch (e) {
          console.warn('Failed to fetch liquidity pools for volume/liquidity data');
        }

        // Update tokens with prices, volume, and liquidity
        // v2.9.9-beta: Simplified - protected balances already applied above from localStorage
        const tokensWithPrices = customTokensList.map(token => {
          const priceInfo = priceMap.get(token.contractAddress);
          const poolInfo = poolsData.get(token.contractAddress);
          const priceUsd = priceInfo?.price || 0;
          return {
            ...token,
            priceUsd,
            valueUsd: token.balance * priceUsd,
            change24h: priceInfo?.change24h || 0,
            volume24h: poolInfo?.volume24h || 0,
            liquidity: poolInfo?.liquidity || 0,
          };
        });

        // v2.9.9-beta: Simply set the tokens - protected balances already applied from localStorage
        setCustomTokens(tokensWithPrices);

        console.log('🎨 [CustomTokens] Loaded', tokensWithPrices.length, 'custom tokens,',
          tokensWithPrices.filter(t => t.balance > 0).length, 'with balance');
      } else {
        console.warn('⚠️ [CustomTokens] Failed to fetch custom tokens:', response.error);
        setError(response.error || 'Failed to load custom tokens');
      }
    } catch (err) {
      console.error('❌ [CustomTokens] Error fetching custom tokens:', err);
      setError('Failed to load custom tokens');
    } finally {
      setLoading(false);
    }
  };

  const fetchStakeInfo = async (contractAddress: string, walletAddress: string) => {
    try {
      const response = await fetch(`/api/v1/contracts/${contractAddress}/stake-info/${walletAddress}`);
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.data) {
          setStakePositions(prev => ({
            ...prev,
            [contractAddress]: data.data
          }));
        }
      }
    } catch (e) {
      console.warn('Failed to fetch stake info:', e);
    }
  };

  const handleStake = async () => {
    if (!stakingToken || !stakeAmount) return;

    setStakeLoading(true);
    setStakeError(null);
    setStakeSuccess(null);

    try {
      const amount = parseFloat(stakeAmount);
      if (isNaN(amount) || amount <= 0) {
        setStakeError('Please enter a valid amount');
        return;
      }

      if (amount > stakingToken.balance) {
        setStakeError('Insufficient balance');
        return;
      }

      // v3.6.18: Use contract's reflection rate instead of hardcoded tiers
      const response = await fetch(`/api/v1/contracts/${stakingToken.contractAddress}/stake`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: localStorage.getItem('walletAddress') || '',
          amount: Math.floor(amount * 1e8), // Convert to base units (8 decimals for custom tokens)
          reflection_rate: tokenReflectionRate, // Use contract's reflection rate
        }),
      });

      const data = await response.json();
      if (data.success) {
        setStakeSuccess(`Successfully staked ${amount} ${stakingToken.symbol} at ${tokenReflectionRate || 0}% reflection!`);
        setStakeAmount('');
        // Refresh balances
        fetchCustomTokens();
        // Refresh stake info
        const walletAddr = localStorage.getItem('walletAddress') || '';
        if (walletAddr) {
          fetchStakeInfo(stakingToken.contractAddress, walletAddr);
        }
      } else {
        setStakeError(data.error || 'Failed to stake tokens');
      }
    } catch (err) {
      console.error('Stake error:', err);
      setStakeError('Failed to stake tokens');
    } finally {
      setStakeLoading(false);
    }
  };

  const handleUnstake = async () => {
    if (!stakingToken) return;

    setStakeLoading(true);
    setStakeError(null);

    try {
      const response = await fetch(`/api/v1/contracts/${stakingToken.contractAddress}/unstake`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: localStorage.getItem('walletAddress') || '',
        }),
      });

      const data = await response.json();
      if (data.success) {
        setStakeSuccess('Successfully unstaked tokens!');
        fetchCustomTokens();
        const walletAddr = localStorage.getItem('walletAddress') || '';
        if (walletAddr) {
          fetchStakeInfo(stakingToken.contractAddress, walletAddr);
        }
      } else {
        setStakeError(data.error || 'Failed to unstake tokens');
      }
    } catch (err) {
      setStakeError('Failed to unstake tokens');
    } finally {
      setStakeLoading(false);
    }
  };

  const handleClaimRewards = async () => {
    if (!stakingToken) return;

    setStakeLoading(true);
    setStakeError(null);

    try {
      const response = await fetch(`/api/v1/contracts/${stakingToken.contractAddress}/claim-rewards`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: localStorage.getItem('walletAddress') || '',
        }),
      });

      const data = await response.json();
      if (data.success) {
        setStakeSuccess(`Claimed ${(data.data?.rewards_claimed || 0) / 1e24} ${stakingToken.symbol} in rewards!`);
        fetchCustomTokens();
        const walletAddr = localStorage.getItem('walletAddress') || '';
        if (walletAddr) {
          fetchStakeInfo(stakingToken.contractAddress, walletAddr);
        }
      } else {
        setStakeError(data.error || 'Failed to claim rewards');
      }
    } catch (err) {
      setStakeError('Failed to claim rewards');
    } finally {
      setStakeLoading(false);
    }
  };

  const openStakeModal = async (token: CustomToken) => {
    setStakingToken(token);
    setStakeAmount('');
    setSelectedTier(0);
    setStakeError(null);
    setStakeSuccess(null);
    setTokenReflectionRate(null);
    setStakeHistory([]);

    // v3.7.0: Fetch reflection rate from fee-config endpoint (basis points -> percentage)
    try {
      const feeConfigResponse = await fetch(`/api/v1/contracts/${token.contractAddress}/fee-config`);
      if (feeConfigResponse.ok) {
        const feeConfigData = await feeConfigResponse.json();
        if (feeConfigData.success && feeConfigData.data) {
          // reflection_fee_bps is in basis points (200 = 2%), convert to percentage
          const reflectionBps = feeConfigData.data.reflection_fee_bps || 0;
          setTokenReflectionRate(reflectionBps / 100);
        } else {
          // No fee config set - default to 2%
          setTokenReflectionRate(2.0);
        }
      } else {
        // Endpoint error - default to 2%
        setTokenReflectionRate(2.0);
      }
    } catch (e) {
      console.warn('Failed to fetch contract fee config:', e);
      // Network error - default to 2%
      setTokenReflectionRate(2.0);
    }

    // Fetch existing stake info
    const walletAddr = localStorage.getItem('walletAddress') || '';
    if (walletAddr) {
      fetchStakeInfo(token.contractAddress, walletAddr);
      // v3.6.18: Also fetch stake history for this wallet
      try {
        const historyResponse = await fetch(`/api/v1/contracts/${token.contractAddress}/stake-history/${walletAddr}`);
        if (historyResponse.ok) {
          const historyData = await historyResponse.json();
          if (historyData.success && historyData.data) {
            setStakeHistory(historyData.data);
          }
        }
      } catch (e) {
        console.warn('Failed to fetch stake history:', e);
      }
    }
  };

  // v3.2.15-beta: Format large token balances - FULL NUMBER (no abbreviations)
  const formatLargeBalance = (value: number): string => {
    if (!isFinite(value) || value <= 0) return '0';
    if (value >= 1e30) return `${(value / 1e30).toFixed(2)} Nonillion`;
    if (value >= 1e27) return `${(value / 1e27).toFixed(2)} Octillion`;
    if (value >= 1e24) return `${(value / 1e24).toFixed(2)} Septillion`;
    if (value >= 1e21) return `${(value / 1e21).toFixed(2)} Sextillion`;
    if (value >= 1e18) return `${(value / 1e18).toFixed(2)} Quintillion`;
    if (value >= 1e15) return `${(value / 1e15).toFixed(2)} Quadrillion`;
    if (value >= 1e12) return `${(value / 1e12).toFixed(2)} Trillion`;
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)} Billion`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)} Million`;
    return value.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 4 });
  };

  // v3.9.5-beta: Format USD values with subscript zero notation for tiny prices
  const SUBSCRIPT_DIGITS = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉'];
  const toSubscript = (n: number): string => {
    return String(n).split('').map(d => SUBSCRIPT_DIGITS[parseInt(d)] || d).join('');
  };
  const formatUsd = (value: number) => {
    if (!isFinite(value) || value <= 0) return '$0.00';
    try {
      if (value <= Number.MAX_SAFE_INTEGER) {
        if (value >= 1) {
          return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        } else if (value >= 0.0001) {
          return `$${value.toFixed(6)}`;
        } else if (value > 0) {
          // Tiny prices: subscript zero notation ($0.0₇38)
          const str = value.toFixed(20);
          const afterDot = str.split('.')[1] || '';
          let zeroCount = 0;
          for (const ch of afterDot) {
            if (ch === '0') zeroCount++;
            else break;
          }
          const sigDigits = afterDot.slice(zeroCount, zeroCount + 4).replace(/0+$/, '') || '0';
          if (zeroCount >= 2) return `$0.0${toSubscript(zeroCount)}${sigDigits}`;
          return `$${value.toFixed(8).replace(/0+$/, '')}`;
        }
      }
      // For very large values (> MAX_SAFE_INTEGER), use mantissa/exponent approach
      const exponent = Math.floor(Math.log10(value));
      const mantissa = value / Math.pow(10, exponent);
      // Build the full number string from mantissa and exponent
      const mantissaStr = mantissa.toFixed(15).replace('.', '');
      const digits = mantissaStr.slice(0, Math.min(exponent + 1, 31)); // Cap at 31 digits
      const paddedDigits = digits.padEnd(exponent + 1, '0');
      // Add comma separators
      const formatted = paddedDigits.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
      return `$${formatted}.00`;
    } catch {
      return `$${String(value)}`;
    }
  };

  const formatTimeRemaining = (unlockTime: number) => {
    const now = Date.now() / 1000;
    const remaining = unlockTime - now;
    if (remaining <= 0) return 'Unlocked';
    const days = Math.floor(remaining / 86400);
    const hours = Math.floor((remaining % 86400) / 3600);
    if (days > 0) return `${days}d ${hours}h`;
    return `${hours}h`;
  };

  const currentStakePosition = stakingToken ? stakePositions[stakingToken.contractAddress] : null;

  return (
    <>
      <motion.div
        className="rounded-2xl p-6 border-2"
        style={{
          background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(168, 85, 247, 0.05))',
          borderColor: 'rgba(139, 92, 246, 0.3)',
        }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(168, 85, 247, 0.15))',
                border: '2px solid rgba(139, 92, 246, 0.3)',
              }}
            >
              <Coins className="w-5 h-5 text-purple-300" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">My Tokens</h3>
              <p className="text-sm text-gray-400">
                {loading ? 'Loading...' : `${tokensWithBalance.length} token${tokensWithBalance.length !== 1 ? 's' : ''} owned`}
              </p>
            </div>
          </div>

          {/* Expand/Collapse Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-2 rounded-lg transition-colors"
            style={{
              background: 'rgba(139, 92, 246, 0.1)',
              border: '1px solid rgba(139, 92, 246, 0.2)',
            }}
          >
            {isExpanded ? (
              <ChevronUp className="w-5 h-5 text-purple-300" />
            ) : (
              <ChevronDown className="w-5 h-5 text-purple-300" />
            )}
          </motion.button>
        </div>

        {/* v2.4.1: Portfolio Summary */}
        {!loading && tokensWithBalance.length > 0 && (
          <div className="mb-4 p-4 rounded-xl" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-green-500/20">
                  <DollarSign className="w-4 h-4 text-green-400" />
                </div>
                <div>
                  <p className="text-xs text-gray-400">Total Value</p>
                  <p className="text-lg font-bold text-green-400 transition-all duration-700">{formatUsd(totalPortfolioValue)}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-purple-500/20">
                  <PieChart className="w-4 h-4 text-purple-400" />
                </div>
                <div>
                  <p className="text-xs text-gray-400">Tokens Held</p>
                  <p className="text-lg font-bold text-purple-300">{tokensWithBalance.length}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* v3.7.1: Sort & Filter Controls */}
        {!loading && customTokens.filter(t => t.balance > 0).length > 0 && (
          <div className="mb-4 space-y-3">
            {/* Search Input */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search tokens..."
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg bg-black/30 border border-purple-500/20 text-white text-sm placeholder-gray-500 focus:border-purple-500/50 focus:outline-none"
              />
              {filterText && (
                <button
                  onClick={() => setFilterText('')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-300"
                >
                  ×
                </button>
              )}
            </div>

            {/* Sort Controls */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-gray-400 flex items-center gap-1">
                <ArrowUpDown className="w-3 h-3" />
                Sort by:
              </span>
              {(['valueUsd', 'symbol', 'balance', 'change24h', 'volume24h', 'liquidity'] as SortField[]).map((field) => (
                <button
                  key={field}
                  onClick={() => {
                    if (sortField === field) {
                      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
                    } else {
                      setSortField(field);
                      setSortDirection('desc');
                    }
                  }}
                  className={`px-2 py-1 rounded text-xs font-medium transition-colors flex items-center gap-1 ${
                    sortField === field
                      ? 'bg-purple-500/30 text-purple-300 border border-purple-500/50'
                      : 'bg-black/20 text-gray-400 border border-transparent hover:border-purple-500/30 hover:text-purple-300'
                  }`}
                >
                  {{
                    valueUsd: 'Value',
                    symbol: 'Name',
                    balance: 'Balance',
                    change24h: '24h %',
                    volume24h: 'Volume',
                    liquidity: 'Liquidity',
                  }[field]}
                  {sortField === field && (
                    sortDirection === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />
                  )}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Content */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              {loading && customTokens.length === 0 ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 text-purple-400 animate-spin" />
                  <span className="ml-3 text-purple-300">Loading your tokens...</span>
                </div>
              ) : error ? (
                <div className="flex items-center gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/30">
                  <AlertCircle className="w-5 h-5 text-red-400" />
                  <div>
                    <p className="text-sm font-medium text-red-300">Failed to load tokens</p>
                    <p className="text-xs text-red-400/80 mt-1">{error}</p>
                  </div>
                </div>
              ) : tokensWithBalance.length === 0 ? (
                <div className="text-center py-8">
                  <Coins className="w-12 h-12 text-purple-400/50 mx-auto mb-3" />
                  <p className="text-purple-300/70 text-sm">No tokens yet</p>
                  <p className="text-purple-400/50 text-xs mt-1">
                    Buy tokens on the DEX to see them here
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  <AnimatePresence mode="popLayout" initial={false}>
                  {tokensWithBalance.map((token) => (
                    <motion.div
                      key={token.contractAddress}
                      layout
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ duration: 0.2, layout: { duration: 0.3, type: 'spring', stiffness: 300, damping: 30 } }}
                      className={`p-4 rounded-xl border ${token.symbol?.toUpperCase() === 'VAULT' ? 'cursor-pointer hover:border-purple-400/50' : ''} ${token.symbol?.toUpperCase() === 'FORGE' ? 'cursor-pointer hover:border-orange-400/50' : ''}`}
                      style={{
                        background: token.symbol?.toUpperCase() === 'VAULT'
                          ? 'linear-gradient(135deg, rgba(168, 130, 255, 0.1), rgba(108, 92, 231, 0.08))'
                          : token.symbol?.toUpperCase() === 'FORGE'
                          ? 'linear-gradient(135deg, rgba(184, 115, 51, 0.12), rgba(212, 175, 55, 0.08))'
                          : 'rgba(139, 92, 246, 0.05)',
                        borderColor: token.symbol?.toUpperCase() === 'VAULT'
                          ? 'rgba(168, 130, 255, 0.35)'
                          : token.symbol?.toUpperCase() === 'FORGE'
                          ? 'rgba(184, 115, 51, 0.35)'
                          : 'rgba(139, 92, 246, 0.2)',
                      }}
                      onClick={() => {
                        if (token.symbol?.toUpperCase() === 'VAULT') {
                          setShowVaultModal(true);
                        } else if (token.symbol?.toUpperCase() === 'FORGE') {
                          setShowForgeModal(true);
                        }
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h4 className="font-semibold text-white">{token.symbol}</h4>
                            {token.symbol?.toUpperCase() === 'VAULT' && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded-full font-bold" style={{ background: 'linear-gradient(135deg, #6c5ce7, #a882ff)', color: 'white' }}>RWA</span>
                            )}
                            {token.symbol?.toUpperCase() === 'FORGE' && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded-full font-bold" style={{ background: 'linear-gradient(135deg, #B87333, #D4AF37)', color: 'white' }}>RWA</span>
                            )}
                            <span className="text-xs text-gray-500">•</span>
                            <span className="text-xs text-gray-400">{token.name}</span>
                            {(token.change24h ?? 0) !== 0 && (
                              <span className={`text-xs flex items-center gap-1 transition-colors duration-500 ${(token.change24h ?? 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                <TrendingUp className={`w-3 h-3 ${(token.change24h ?? 0) < 0 ? 'rotate-180' : ''}`} />
                                {Math.abs(token.change24h ?? 0).toFixed(1)}%
                              </span>
                            )}
                          </div>
                          <div className="flex items-baseline gap-3">
                            <p className="text-2xl font-bold text-purple-300 transition-all duration-500">
                              {formatLargeBalance(token.balance)}
                            </p>
                            {(token.valueUsd ?? 0) > 0 && (
                              <p className="text-sm text-green-400 font-medium transition-all duration-500">
                                ≈ {formatUsd(token.valueUsd ?? 0)}
                              </p>
                            )}
                          </div>
                          <div className="flex items-center gap-3 mt-1 flex-wrap">
                            {(token.priceUsd ?? 0) > 0 && (
                              <p className="text-xs text-gray-400 transition-all duration-500">
                                @ {formatUsd(token.priceUsd ?? 0)} each
                              </p>
                            )}
                            {/* v3.6.12: Volume and Liquidity from DEX */}
                            {(token.volume24h ?? 0) > 0 && (
                              <p className="text-xs text-blue-400 transition-all duration-500">
                                Vol: {formatUsd(token.volume24h ?? 0)}
                              </p>
                            )}
                            {(token.liquidity ?? 0) > 0 && (
                              <p className="text-xs text-cyan-400 transition-all duration-500">
                                Liq: {formatUsd(token.liquidity ?? 0)}
                              </p>
                            )}
                          </div>
                          <div className="flex items-center mt-1">
                            <p className="text-xs text-gray-500 font-mono truncate">
                              {token.contractAddress.substring(0, 8)}...{token.contractAddress.substring(token.contractAddress.length - 6)}
                            </p>
                          </div>
                        </div>

                        {/* Action Buttons */}
                        <div className="flex items-center gap-2">
                          {/* Stake Button */}
                          <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => openStakeModal(token)}
                            className="px-3 py-2 rounded-lg text-sm font-medium flex items-center gap-2"
                            style={{
                              background: 'linear-gradient(135deg, rgba(234, 179, 8, 0.2), rgba(202, 138, 4, 0.15))',
                              border: '2px solid rgba(234, 179, 8, 0.3)',
                              color: 'rgb(253, 224, 71)',
                            }}
                          >
                            <Lock className="w-4 h-4" />
                            Stake
                          </motion.button>

                          {/* Send Button */}
                          <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => onSendToken(token.symbol, token.contractAddress)}
                            className="px-3 py-2 rounded-lg text-sm font-medium flex items-center gap-2"
                            style={{
                              background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.15))',
                              border: '2px solid rgba(59, 130, 246, 0.3)',
                              color: 'rgb(147, 197, 253)',
                            }}
                            disabled={token.balance === 0}
                          >
                            <Send className="w-4 h-4" />
                            Send
                          </motion.button>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                  </AnimatePresence>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Staking Modal */}
      <AnimatePresence>
        {stakingToken && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
            onClick={() => setStakingToken(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="w-full max-w-md rounded-2xl p-6 border-2"
              style={{
                background: 'linear-gradient(135deg, rgba(30, 27, 75, 0.98), rgba(17, 24, 39, 0.98))',
                borderColor: 'rgba(234, 179, 8, 0.3)',
              }}
              onClick={e => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-xl flex items-center justify-center bg-yellow-500/20 border-2 border-yellow-500/30">
                  <Lock className="w-6 h-6 text-yellow-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">Stake {stakingToken.symbol}</h3>
                  <p className="text-sm text-gray-400">Earn rewards by locking your tokens</p>
                </div>
              </div>

              {/* Current Stake Position */}
              {currentStakePosition && currentStakePosition.amount > 0 && (
                <div className="mb-6 p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/20">
                  <h4 className="text-sm font-semibold text-yellow-300 mb-3 flex items-center gap-2">
                    <Award className="w-4 h-4" />
                    Your Active Stake
                  </h4>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <p className="text-gray-400">Staked</p>
                      <p className="text-white font-bold">{(currentStakePosition.amount / 1e24).toLocaleString()} {stakingToken.symbol}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Tier</p>
                      <p className="text-yellow-300 font-bold">{currentStakePosition.tier}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">APY</p>
                      <p className="text-green-400 font-bold">{(currentStakePosition.apy_bps / 100).toFixed(0)}%</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Unlock In</p>
                      <p className="text-cyan-300 font-bold flex items-center gap-1">
                        <Timer className="w-3 h-3" />
                        {formatTimeRemaining(currentStakePosition.unlock_time)}
                      </p>
                    </div>
                  </div>
                  {currentStakePosition.pending_rewards > 0 && (
                    <div className="mt-3 pt-3 border-t border-yellow-500/20 flex items-center justify-between">
                      <div>
                        <p className="text-gray-400 text-xs">Pending Rewards</p>
                        <p className="text-green-400 font-bold">+{(currentStakePosition.pending_rewards / 1e24).toLocaleString()} {stakingToken.symbol}</p>
                      </div>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleClaimRewards}
                        disabled={stakeLoading}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1 bg-green-500/20 border border-green-500/30 text-green-300"
                      >
                        <Gift className="w-3 h-3" />
                        Claim
                      </motion.button>
                    </div>
                  )}
                  {/* Unstake Button */}
                  {currentStakePosition.unlock_time <= Date.now() / 1000 && (
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleUnstake}
                      disabled={stakeLoading}
                      className="w-full mt-3 px-4 py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-2 bg-cyan-500/20 border border-cyan-500/30 text-cyan-300"
                    >
                      <Unlock className="w-4 h-4" />
                      Unstake All
                    </motion.button>
                  )}
                </div>
              )}

              {/* Stake New Tokens */}
              <div className="space-y-4">
                {/* Amount Input */}
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Amount to Stake</label>
                  <div className="relative">
                    <input
                      type="number"
                      value={stakeAmount}
                      onChange={e => setStakeAmount(e.target.value)}
                      placeholder="0.00"
                      className="w-full px-4 py-3 rounded-xl bg-black/30 border-2 border-yellow-500/20 text-white text-lg focus:border-yellow-500/50 focus:outline-none"
                    />
                    <button
                      onClick={() => setStakeAmount(stakingToken.balance.toString())}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-yellow-400 hover:text-yellow-300"
                    >
                      MAX
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Available: {stakingToken.balance.toLocaleString()} {stakingToken.symbol}
                  </p>
                </div>

                {/* v3.6.18: Reflection Rate from Contract (set by token owner) */}
                <div className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-cyan-500/10 border border-purple-500/30">
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm text-gray-400">Reflection Rate</label>
                    <span className="text-xs text-gray-500">Set by token owner</span>
                  </div>
                  <div className="text-center">
                    <p className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400">
                      {tokenReflectionRate !== null ? `${tokenReflectionRate}%` : 'Loading...'}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {tokenReflectionRate !== null
                        ? `Earn ${tokenReflectionRate}% of all transaction fees redistributed to stakers`
                        : 'Fetching reflection settings...'}
                    </p>
                  </div>
                </div>

                {/* v3.6.18: Stake History */}
                {stakeHistory.length > 0 && (
                  <div className="mt-4">
                    <label className="text-sm text-gray-400 mb-2 block flex items-center gap-2">
                      <Award className="w-4 h-4 text-yellow-400" />
                      Stake History & Profits
                    </label>
                    <div className="max-h-40 overflow-y-auto space-y-2 pr-1">
                      {stakeHistory.map((entry, index) => (
                        <div
                          key={index}
                          className={`p-2 rounded-lg text-sm flex items-center justify-between ${
                            entry.type === 'reward'
                              ? 'bg-green-500/10 border border-green-500/20'
                              : entry.type === 'stake'
                              ? 'bg-yellow-500/10 border border-yellow-500/20'
                              : 'bg-cyan-500/10 border border-cyan-500/20'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            {entry.type === 'reward' ? (
                              <Gift className="w-4 h-4 text-green-400" />
                            ) : entry.type === 'stake' ? (
                              <Lock className="w-4 h-4 text-yellow-400" />
                            ) : (
                              <Unlock className="w-4 h-4 text-cyan-400" />
                            )}
                            <span className={
                              entry.type === 'reward' ? 'text-green-300'
                              : entry.type === 'stake' ? 'text-yellow-300'
                              : 'text-cyan-300'
                            }>
                              {entry.type === 'reward' ? 'Reward' : entry.type === 'stake' ? 'Staked' : 'Unstaked'}
                            </span>
                          </div>
                          <div className="text-right">
                            <p className={`font-bold ${
                              entry.type === 'reward' ? 'text-green-400' : 'text-white'
                            }`}>
                              {entry.type === 'reward' ? '+' : ''}{entry.amount.toLocaleString()} {stakingToken.symbol}
                            </p>
                            <p className="text-xs text-gray-500">
                              {new Date(entry.timestamp * 1000).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                    {/* Total Rewards Summary */}
                    <div className="mt-2 p-2 rounded-lg bg-green-500/10 border border-green-500/30">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-400">Total Rewards Earned</span>
                        <span className="font-bold text-green-400">
                          +{stakeHistory
                            .filter(e => e.type === 'reward')
                            .reduce((sum, e) => sum + e.amount, 0)
                            .toLocaleString()} {stakingToken.symbol}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Error/Success Messages */}
                {stakeError && (
                  <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
                    {stakeError}
                  </div>
                )}
                {stakeSuccess && (
                  <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/30 text-green-300 text-sm">
                    {stakeSuccess}
                  </div>
                )}

                {/* Stake Button */}
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleStake}
                  disabled={stakeLoading || !stakeAmount}
                  className="w-full py-3 rounded-xl font-bold text-lg flex items-center justify-center gap-2"
                  style={{
                    background: 'linear-gradient(135deg, rgba(234, 179, 8, 0.3), rgba(202, 138, 4, 0.2))',
                    border: '2px solid rgba(234, 179, 8, 0.5)',
                    color: stakeAmount ? 'rgb(253, 224, 71)' : 'rgb(156, 163, 175)',
                  }}
                >
                  {stakeLoading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <Lock className="w-5 h-5" />
                      Stake {stakingToken.symbol}
                    </>
                  )}
                </motion.button>

                {/* Cancel Button */}
                <button
                  onClick={() => setStakingToken(null)}
                  className="w-full py-2 text-gray-400 hover:text-gray-300 text-sm"
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* v4.2.0: VAULT RWA Token Modal */}
      <VaultModal
        isOpen={showVaultModal}
        onClose={() => setShowVaultModal(false)}
        isAdmin={(() => {
          const walletAddr = localStorage.getItem('walletAddress') || '';
          const cleanAddr = walletAddr.replace(/^qnk/, '').toLowerCase();
          const bankMaster = '424e4b0000000000000000000000000000000000000000000000000000000000';
          const operatorWallet = '4fff16bc7d825a3d2e3ae0b15c6e70e91dc18dce1c55ec22543a8e4ae9e6c7b2';
          return cleanAddr === bankMaster || cleanAddr === operatorWallet;
        })()}
        vaultBalance={customTokens.find(t => t.symbol?.toUpperCase() === 'VAULT')?.balance ?? 0}
      />

      {/* v5.1.0: FORGE RWA Token Modal — Mining Machine Redemption */}
      <ForgeModal
        isOpen={showForgeModal}
        onClose={() => setShowForgeModal(false)}
        isAdmin={(() => {
          const walletAddr = localStorage.getItem('walletAddress') || '';
          const cleanAddr = walletAddr.replace(/^qnk/, '').toLowerCase();
          const bankMaster = '424e4b0000000000000000000000000000000000000000000000000000000000';
          const operatorWallet = '4fff16bc7d825a3d2e3ae0b15c6e70e91dc18dce1c55ec22543a8e4ae9e6c7b2';
          return cleanAddr === bankMaster || cleanAddr === operatorWallet;
        })()}
        forgeBalance={customTokens.find(t => t.symbol?.toUpperCase() === 'FORGE')?.balance ?? 0}
      />
    </>
  );
}
