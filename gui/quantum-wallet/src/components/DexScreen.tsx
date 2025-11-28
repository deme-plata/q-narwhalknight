import { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowDownUp, Search, TrendingUp, TrendingDown, Settings, Info, Droplet, Zap } from 'lucide-react';
import TokenDetailsModal from './TokenDetailsModal';
import LiquidityModal from './LiquidityModal';
import TokenSelectorModal from './TokenSelectorModal';
import NitroSuccessModal from './NitroSuccessModal';
import MintQUGUSDModal from './MintQUGUSDModal';
import SwapSuccessModal from './SwapSuccessModal';
import { qnkAPI } from '../services/api';

interface Token {
  id: string;
  symbol: string;
  name: string;
  balance: number;
  price: number;
  change1h: number;   // 1-hour price change percentage
  change24h: number;
  change7d: number;   // 7-day price change percentage
  volume24h: number;
  liquidity: number;
  icon: string;
  marketCap: number;
  totalSupply: number;
  circulatingSupply: number;
  holders: number;
  features: {
    reflection: boolean;
    autoLiquidity: boolean;
    buybackAndBurn: boolean;
    antiWhale: boolean;
    quantumSecured: boolean;
  };
  fees: {
    buy: number;
    sell: number;
    transfer: number;
  };
  description: string;
  website?: string;
  whitepaper?: string;
}

export default function DexScreen() {
  const [swapFrom, setSwapFrom] = useState('QUG');
  const [swapTo, setSwapTo] = useState('QUGUSD');
  const [swapAmount, setSwapAmount] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'symbol' | 'price' | 'change24h' | 'volume24h' | 'liquidity' | 'marketCap'>('volume24h');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [filterBy, setFilterBy] = useState<'all' | 'gainers' | 'losers'>('all');
  const [selectedToken, setSelectedToken] = useState<Token | null>(null);
  const [liquidityToken, setLiquidityToken] = useState<Token | null>(null);
  const [tokens, setTokens] = useState<Token[]>([]);
  const [loading, setLoading] = useState(true);
  const [customTokenAddress, setCustomTokenAddress] = useState('');
  const [liquidityPools, setLiquidityPools] = useState<any[]>([]);
  const [nitroBoostTokens, setNitroBoostTokens] = useState<Set<string>>(new Set());
  const [removingPool, setRemovingPool] = useState<any | null>(null);
  const [removePercentage, setRemovePercentage] = useState(50);
  const [nitroBoostToken, setNitroBoostToken] = useState<Token | null>(null);
  const [nitroPoints, setNitroPoints] = useState(0);
  const [boostedTokens, setBoostedTokens] = useState<Map<string, number>>(new Map()); // token_id -> points used
  const [boostCost, setBoostCost] = useState(100); // Points to spend on boost
  const [isFromTokenSelectorOpen, setIsFromTokenSelectorOpen] = useState(false);
  const [isToTokenSelectorOpen, setIsToTokenSelectorOpen] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [successModalData, setSuccessModalData] = useState<any>(null);
  const [isMintQUGUSDModalOpen, setIsMintQUGUSDModalOpen] = useState(false);
  const [showSwapSuccess, setShowSwapSuccess] = useState(false);
  const [swapSuccessData, setSwapSuccessData] = useState<{
    fromToken: string;
    toToken: string;
    fromAmount: number;
    toAmount: number;
    transactionHash?: string;
  } | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0); // Trigger for refetching tokens

  // Load Nitro points from localStorage and boosted tokens from backend with SSE real-time updates
  useEffect(() => {
    let mounted = true;
    let eventSource: EventSource | null = null;

    // Load Nitro points from localStorage (per wallet address)
    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (walletAddress) {
      const storedPoints = localStorage.getItem(`nitroPoints_${walletAddress}`);
      if (storedPoints) {
        setNitroPoints(parseInt(storedPoints, 10));
      }
    }

    // Listen for nitro points updates from other components (e.g., TokenBar)
    const handleNitroPointsUpdate = () => {
      const updatedPoints = localStorage.getItem(`nitroPoints_${walletAddress}`);
      if (updatedPoints && mounted) {
        setNitroPoints(parseInt(updatedPoints, 10));
        console.log('✅ Nitro points updated in DexScreen:', updatedPoints);
      }
    };

    // Listen for custom event from TokenBar when nitro points are purchased
    window.addEventListener('nitroPointsUpdated', handleNitroPointsUpdate);

    // Also listen for storage events (works across tabs/windows)
    window.addEventListener('storage', (e) => {
      if (e.key === `nitroPoints_${walletAddress}` && e.newValue && mounted) {
        setNitroPoints(parseInt(e.newValue, 10));
        console.log('✅ Nitro points synced via storage event:', e.newValue);
      }
    });

    // Load initial boosted tokens from backend API
    const fetchNitroBoosts = async () => {
      if (!mounted) return;
      try {
        const response = await qnkAPI.getNitroBoosts();
        if (response.success && response.data && mounted) {
          // Convert Record<string, number> to Map
          const boostMap = new Map(Object.entries(response.data));
          setBoostedTokens(boostMap);
          console.log('✅ Loaded Nitro boosts from backend:', response.data);
        }
      } catch (error) {
        console.error('Failed to fetch Nitro boosts:', error);
      }
    };

    // Initial fetch
    fetchNitroBoosts();

    // Set up SSE for real-time Nitro boost updates
    const sseUrl = import.meta.env.VITE_API_URL ?
      `${import.meta.env.VITE_API_URL}/v1/events` :
      '/api/v1/events';

    console.log('📡 Setting up SSE for Nitro boosts:', sseUrl);

    try {
      eventSource = new EventSource(sseUrl);

      eventSource.onopen = () => {
        console.log('✅ SSE connection established for Nitro boosts');
      };

      // Listen for nitro_boost events
      eventSource.addEventListener('nitro_boost', (event) => {
        if (!mounted) return;
        try {
          const parsed = JSON.parse(event.data);
          console.log('🚀 Received Nitro boost event:', parsed);

          // Extract data from wrapper - backend sends {type: "NitroBoost", data: {...}}
          const data = parsed.data || parsed;

          // Update boosted tokens map
          setBoostedTokens(prev => {
            const newMap = new Map(prev);
            const tokenId = data.token_id;
            const totalPoints = data.total_points;
            // Use total_points from backend (which is already aggregated) instead of adding
            newMap.set(tokenId, totalPoints);
            console.log(`✅ Updated ${tokenId} Nitro boost: ${totalPoints} total points`);
            return newMap;
          });

          // Show visual boost animation
          setNitroBoostTokens(prev => {
            const newSet = new Set(prev);
            newSet.add(data.token_id);
            return newSet;
          });

          setTimeout(() => {
            if (mounted) {
              setNitroBoostTokens(prev => {
                const newSet = new Set(prev);
                newSet.delete(data.token_id);
                return newSet;
              });
            }
          }, 2000);
        } catch (err) {
          console.error('Failed to parse Nitro boost SSE event:', err);
        }
      });

      eventSource.addEventListener('nitro_boosts_update', (event) => {
        if (!mounted) return;
        try {
          const parsed = JSON.parse(event.data);
          console.log('📊 Received full Nitro boosts update:', parsed);

          // Extract data from wrapper
          const data = parsed.data || parsed;

          // Full update of all boosts
          if (data.boosts) {
            const boostMap = new Map(Object.entries(data.boosts) as [string, number][]);
            setBoostedTokens(boostMap);
            console.log('✅ Updated all Nitro boosts:', Object.keys(data.boosts).length, 'tokens');
          }
        } catch (err) {
          console.error('Failed to parse Nitro boosts update event:', err);
        }
      });

      // Listen for token price updates
      eventSource.addEventListener('token_price_update', (event) => {
        if (!mounted) return;
        try {
          const parsed = JSON.parse(event.data);
          console.log('📈 Received token price update:', parsed);

          // Extract data from wrapper
          const data = parsed.data || parsed;

          // Update token in list
          setTokens(prev => prev.map(token =>
            token.id === data.token_id
              ? { ...token, price: data.price, change24h: data.change_24h || token.change24h, volume24h: data.volume_24h || token.volume24h }
              : token
          ));
        } catch (err) {
          console.error('Failed to parse token price update:', err);
        }
      });

      // Listen for token transactions
      eventSource.addEventListener('token_transaction', (event) => {
        if (!mounted) return;
        try {
          const parsed = JSON.parse(event.data);
          console.log('📜 Received token transaction:', parsed);

          // Extract data from wrapper
          const data = parsed.data || parsed;

          // Update token volume in real-time if we have the data
          if (data.token_id && data.value) {
            setTokens(prev => prev.map(token =>
              token.id === data.token_id
                ? { ...token, volume24h: token.volume24h + (data.value || 0) }
                : token
            ));
          }

          // Transaction data will be consumed by TokenDetailsModal
        } catch (err) {
          console.error('Failed to parse token transaction:', err);
        }
      });

      eventSource.onerror = (error) => {
        console.error('❌ SSE connection error for Nitro boosts:', error);
        // SSE will automatically reconnect
      };

    } catch (error) {
      console.error('Failed to establish SSE connection for Nitro boosts:', error);
    }

    return () => {
      mounted = false;
      window.removeEventListener('nitroPointsUpdated', handleNitroPointsUpdate);
      if (eventSource) {
        console.log('🔌 Closing SSE connection for Nitro boosts');
        eventSource.close();
      }
    };
  }, []);

  // Fetch real tokens from API with SSE real-time updates
  useEffect(() => {
    let mounted = true;
    let sseEventSource: EventSource | null = null;

    const fetchTokens = async () => {
      try {
        // Get wallet address for balance fetching
        const walletAddress = localStorage.getItem('walletAddress') || '';

        // Fetch native QUG, QUGUSD, and USD balances using multi-token API
        let nativeQugBalance = 0;
        let qugusdBalance = 0;
        let usdBalance = 0;
        if (walletAddress) {
          console.log('🔍 [DEX] Fetching multi-token balance for wallet:', walletAddress);
          try {
            const multiTokenResponse = await qnkAPI.getMultiTokenBalance();
            console.log('📊 [DEX] Multi-token balance API response:', multiTokenResponse);
            if (multiTokenResponse.success && multiTokenResponse.data) {
              // Extract QUG balance (already in human-readable form)
              if (multiTokenResponse.data.tokens && multiTokenResponse.data.tokens.QUG) {
                nativeQugBalance = parseFloat(multiTokenResponse.data.tokens.QUG.balance) || 0;
                console.log('✅ [DEX] Native QUG balance fetched:', nativeQugBalance, 'QUG');
              } else {
                console.warn('⚠️ [DEX] QUG balance not in multi-token response');
              }
              // Extract QUGUSD balance (already in human-readable form)
              if (multiTokenResponse.data.tokens && multiTokenResponse.data.tokens.QUGUSD) {
                qugusdBalance = parseFloat(multiTokenResponse.data.tokens.QUGUSD.balance) || 0;
                console.log('✅ [DEX] QUGUSD balance fetched:', qugusdBalance, 'QUGUSD');
              }
            } else {
              console.warn('⚠️ [DEX] Multi-token balance fetch unsuccessful:', multiTokenResponse);
              console.warn('⚠️ [DEX] API error:', multiTokenResponse.error);

              // FALLBACK: Try single wallet balance API (same as Dashboard uses)
              console.log('🔄 [DEX] Trying fallback: getWalletBalance');
              try {
                const fallbackResponse = await qnkAPI.getWalletBalance(walletAddress);
                console.log('📊 [DEX] Fallback balance response:', fallbackResponse);
                if (fallbackResponse.success && fallbackResponse.data) {
                  nativeQugBalance = fallbackResponse.data.balance_qnk || 0;
                  console.log('✅ [DEX] Fallback QUG balance fetched:', nativeQugBalance, 'QUG');
                } else {
                  // FINAL FALLBACK: Use cached balance from localStorage (what Dashboard uses)
                  const cachedBalance = localStorage.getItem('cachedBalance');
                  if (cachedBalance) {
                    nativeQugBalance = parseFloat(cachedBalance);
                    console.log('💰 [DEX] Using cached balance from localStorage:', nativeQugBalance);
                  }
                }
              } catch (fallbackError) {
                console.error('❌ [DEX] Fallback balance fetch failed:', fallbackError);
                // FINAL FALLBACK: Use cached balance from localStorage
                const cachedBalance = localStorage.getItem('cachedBalance');
                if (cachedBalance) {
                  nativeQugBalance = parseFloat(cachedBalance);
                  console.log('💰 [DEX] Using cached balance from localStorage (error fallback):', nativeQugBalance);
                }
              }
            }
          } catch (error) {
            console.error('❌ [DEX] Failed to fetch multi-token balance:', error);
            // FALLBACK: Use cached balance from localStorage
            const cachedBalance = localStorage.getItem('cachedBalance');
            if (cachedBalance) {
              nativeQugBalance = parseFloat(cachedBalance);
              console.log('💰 [DEX] Using cached balance from localStorage (catch fallback):', nativeQugBalance);
            }
          }
        } else {
          console.warn('⚠️ [DEX] No wallet address found in localStorage');
        }

        // Fetch all liquidity pools to calculate real liquidity per token
        // v1.0.49-beta: ENHANCED - Better address/symbol mapping from pools
        let poolsByToken: Map<string, number> = new Map();
        let addressToSymbol: Map<string, string> = new Map(); // Reverse map for display
        try {
          const poolsResponse = await qnkAPI.getLiquidityPools();
          if (poolsResponse.success && poolsResponse.data) {
            // Build a symbol-to-address map for resolving custom tokens
            let symbolToAddress: Map<string, string> = new Map();

            // Fetch user contracts to map symbols to addresses
            if (walletAddress) {
              try {
                const userContractsResponse = await qnkAPI.getUserContracts(walletAddress);
                if (userContractsResponse.success && userContractsResponse.data) {
                  userContractsResponse.data.forEach((contract: any) => {
                    symbolToAddress.set(contract.symbol.toUpperCase(), contract.address);
                    addressToSymbol.set(contract.address, contract.symbol);
                    console.log(`📍 Mapped symbol ${contract.symbol} => ${contract.address}`);
                  });
                }
              } catch (error) {
                console.log('ℹ️ Could not fetch user contracts for symbol mapping:', error);
              }
            }

            // Also fetch supported tokens and add to map
            try {
              const supportedTokensResponse = await qnkAPI.getSupportedTokens();
              if (supportedTokensResponse.success && supportedTokensResponse.data) {
                supportedTokensResponse.data.forEach((token: any) => {
                  symbolToAddress.set(token.symbol.toUpperCase(), token.address);
                  addressToSymbol.set(token.address, token.symbol);
                });
              }
            } catch (error) {
              console.log('ℹ️ Could not fetch supported tokens for symbol mapping:', error);
            }

            // v1.0.49-beta: Aggregate liquidity by token ADDRESS (not symbol)
            // Backend now stores pools with canonical addresses (qnk...) which fixes duplicate pool bug
            poolsResponse.data.forEach((pool: any) => {
              // Resolve token0 to address key
              let token0Key: string;
              if (pool.token0 === 'QUG' || pool.token0.toUpperCase() === 'QUG') {
                token0Key = 'native-qug';
              } else if (pool.token0 === 'QUGUSD' || pool.token0.toUpperCase() === 'QUGUSD') {
                token0Key = 'qugusd-stable';
              } else if (pool.token0.startsWith('qnk') || pool.token0.startsWith('0x')) {
                // Already an address - use directly (v1.0.49-beta: pools now use canonical addresses)
                token0Key = pool.token0;
              } else {
                // It's a symbol, resolve to address
                token0Key = symbolToAddress.get(pool.token0.toUpperCase()) || pool.token0;
                if (token0Key !== pool.token0) {
                  console.log(`🔍 Resolved pool.token0 "${pool.token0}" => "${token0Key}"`);
                }
              }
              poolsByToken.set(token0Key, (poolsByToken.get(token0Key) || 0) + (pool.reserve0 || 0));

              // Resolve token1 to address key
              let token1Key: string;
              if (pool.token1 === 'QUG' || pool.token1.toUpperCase() === 'QUG') {
                token1Key = 'native-qug';
              } else if (pool.token1 === 'QUGUSD' || pool.token1.toUpperCase() === 'QUGUSD') {
                token1Key = 'qugusd-stable';
              } else if (pool.token1.startsWith('qnk') || pool.token1.startsWith('0x')) {
                // Already an address - use directly (v1.0.49-beta: pools now use canonical addresses)
                token1Key = pool.token1;
              } else {
                // It's a symbol, resolve to address
                token1Key = symbolToAddress.get(pool.token1.toUpperCase()) || pool.token1;
                if (token1Key !== pool.token1) {
                  console.log(`🔍 Resolved pool.token1 "${pool.token1}" => "${token1Key}"`);
                }
              }
              poolsByToken.set(token1Key, (poolsByToken.get(token1Key) || 0) + (pool.reserve1 || 0));
            });

            console.log('✅ Calculated liquidity from pools (by address):', Object.fromEntries(poolsByToken));
          }
        } catch (error) {
          console.error('Failed to fetch liquidity pools:', error);
        }

        // Fetch real price from oracle API with all time periods
        let qugPrice = 42.50;
        let qugChange1h = 0;
        let qugChange24h = 0;
        let qugChange7d = 0;
        let qugVolume = 0;
        try {
          const oracleResponse = await qnkAPI.getOraclePrice('QUG/USD');
          if (oracleResponse.success && oracleResponse.data) {
            qugPrice = oracleResponse.data.price;
            qugChange1h = oracleResponse.data.change_1h || 0;
            qugChange24h = oracleResponse.data.change_24h || 0;
            qugChange7d = oracleResponse.data.change_7d || 0;
            qugVolume = oracleResponse.data.volume_24h || 0;
            console.log('✅ Fetched QUG metrics from oracle:', { price: qugPrice, change1h: qugChange1h, change24h: qugChange24h, change7d: qugChange7d, volume: qugVolume });
          }
        } catch (error) {
          console.error('Failed to fetch QUG price from oracle:', error);
        }

        // Fetch QUGUSD price from oracle with all time periods
        let qugusdPrice = 1.00;
        let qugusdChange1h = 0;
        let qugusdChange24h = 0;
        let qugusdChange7d = 0;
        let qugusdVolume = 0;
        try {
          const oracleResponse = await qnkAPI.getOraclePrice('QUGUSD/USD');
          if (oracleResponse.success && oracleResponse.data) {
            qugusdPrice = oracleResponse.data.price;
            qugusdChange1h = oracleResponse.data.change_1h || 0;
            qugusdChange24h = oracleResponse.data.change_24h || 0;
            qugusdChange7d = oracleResponse.data.change_7d || 0;
            qugusdVolume = oracleResponse.data.volume_24h || 0;
            console.log('✅ Fetched QUGUSD metrics from oracle:', { price: qugusdPrice, change1h: qugusdChange1h, change24h: qugusdChange24h, change7d: qugusdChange7d, volume: qugusdVolume });
          }
        } catch (error) {
          console.error('Failed to fetch QUGUSD price from oracle:', error);
        }

        // Fetch USD balance from payment API
        if (walletAddress) {
          try {
            const usdResponse = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/balance`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ wallet_address: walletAddress }),
            });

            if (usdResponse.ok) {
              const usdData = await usdResponse.json();
              if (usdData.success && usdData.data) {
                usdBalance = parseFloat(usdData.data.balance_usd || '0');
                console.log('✅ [DEX] USD balance fetched:', usdBalance, 'USD');
              }
            }
          } catch (error) {
            console.error('❌ [DEX] Failed to fetch USD balance:', error);
          }
        }

        // Add native QUG, QUGUSD stablecoin, and USD
        console.log('🔧 Creating QUG token with balance:', nativeQugBalance);
        const nativeTokens: Token[] = [
          {
            id: 'native-qug',
            symbol: 'QUG',
            name: 'Quillon',
            balance: nativeQugBalance,
            price: qugPrice,
            change1h: qugChange1h,
            change24h: qugChange24h,
            change7d: qugChange7d,
            volume24h: qugVolume,
            liquidity: poolsByToken.get('native-qug') || 0,
            marketCap: 625000000,
            totalSupply: 21000000,
            circulatingSupply: 14700000,
            holders: 18432,
            icon: 'qug-logo',
            features: {
              reflection: true,
              autoLiquidity: true,
              buybackAndBurn: true,
              antiWhale: true,
              quantumSecured: true,
            },
            fees: {
              buy: 2,
              sell: 4,
              transfer: 1,
            },
            description: 'QUG (Quillon) is the native quantum-enhanced token powering the Quillon blockchain. Built on DAG-BFT consensus with post-quantum cryptographic security. Enables staking, governance, and ultra-fast transactions.',
            website: 'https://quillon.xyz',
            whitepaper: 'https://quillon.xyz/whitepaper',
          },
          {
            id: 'qugusd-stable',
            symbol: 'QUGUSD',
            name: 'Quillon USD',
            balance: qugusdBalance,
            price: qugusdPrice,
            change1h: qugusdChange1h,
            change24h: qugusdChange24h,
            change7d: qugusdChange7d,
            volume24h: qugusdVolume,
            liquidity: poolsByToken.get('qugusd-stable') || 0,
            marketCap: 125000000,
            totalSupply: 125000000,
            circulatingSupply: 125000000,
            holders: 5600,
            icon: 'qugusd-logo',
            features: {
              reflection: false,
              autoLiquidity: true,
              buybackAndBurn: false,
              antiWhale: false,
              quantumSecured: true,
            },
            fees: {
              buy: 0,
              sell: 0,
              transfer: 0,
            },
            description: 'QUGUSD is a quantum-secured stablecoin pegged 1:1 to USD, backed by collateralized assets and maintained through algorithmic stability mechanisms. Features zero-knowledge privacy, instant transactions, and quantum-resistant cryptography.',
            website: 'https://quillon.xyz',
            whitepaper: 'https://quillon.xyz/qugusd',
          },
          {
            id: 'fiat-usd',
            symbol: 'USD',
            name: 'US Dollar',
            balance: usdBalance,
            price: 1.00,
            change1h: 0,
            change24h: 0,
            change7d: 0,
            volume24h: 0,
            liquidity: poolsByToken.get('fiat-usd') || 0,
            marketCap: 0,
            totalSupply: 0,
            circulatingSupply: 0,
            holders: 0,
            icon: 'usd-logo',
            features: {
              reflection: false,
              autoLiquidity: false,
              buybackAndBurn: false,
              antiWhale: false,
              quantumSecured: false,
            },
            fees: {
              buy: 0,
              sell: 0,
              transfer: 0,
            },
            description: 'USD (United States Dollar) - Traditional fiat currency integrated with quantum blockchain via Stripe payment processing. Enables instant conversion between crypto and fiat with real-time settlement.',
            website: 'https://quillon.xyz',
            whitepaper: 'https://quillon.xyz/usd-integration',
          },
        ];

        const response = await qnkAPI.getSupportedTokens();
        let enrichedTokens = nativeTokens;

        // Also fetch user-deployed contracts to include in Available Tokens
        let userDeployedTokens: any[] = [];
        if (walletAddress) {
          try {
            const userContractsResponse = await qnkAPI.getUserContracts(walletAddress);
            if (userContractsResponse.success && userContractsResponse.data) {
              userDeployedTokens = userContractsResponse.data;
              console.log('✅ Fetched user deployed contracts:', userDeployedTokens.length, 'contracts');
            }
          } catch (error) {
            console.log('ℹ️ No user deployed contracts found or error fetching:', error);
          }
        }

        if (response.success && response.data) {
          // Get wallet address for balance checks (already defined above, no need to redefine)

          // Convert API token data, excluding duplicates
          const apiTokensPromises = response.data
            .filter(apiToken => apiToken.symbol !== 'QUG' && apiToken.symbol !== 'QUGUSD' && apiToken.symbol !== 'ORBUSD')
            .map(async (apiToken) => {
              // Calculate actual supply using decimals from API
              const decimals = apiToken.decimals || 18;
              const rawSupply = apiToken.total_supply || 0;
              const actualSupply = Number(rawSupply) / Math.pow(10, decimals);

              // Fetch balance for this token if wallet is available
              let tokenBalance = 0;
              if (walletAddress) {
                try {
                  console.log(`🔍 [API Token] Fetching balance for ${apiToken.symbol} (address: ${apiToken.address}, wallet: ${walletAddress})`);
                  const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, apiToken.address);
                  console.log(`📊 [API Token] Balance response for ${apiToken.symbol}:`, balanceResponse);

                  if (balanceResponse.success && balanceResponse.data) {
                    // Backend returns balance in base units, convert to human-readable
                    const rawBalance = balanceResponse.data.balance || 0;
                    tokenBalance = rawBalance / Math.pow(10, decimals);
                    console.log(`✅ [API Token] Converted ${apiToken.symbol} balance from ${rawBalance} to ${tokenBalance} (decimals: ${decimals})`);
                  } else {
                    console.warn(`⚠️ [API Token] Balance fetch unsuccessful for ${apiToken.symbol}:`, balanceResponse.error || balanceResponse);
                  }
                } catch (error) {
                  console.error(`❌ [API Token] Failed to fetch balance for ${apiToken.symbol}:`, error);
                }
              }

              // Fetch custom token metrics from oracle (if available)
              let customPrice = 1.0;
              let customChange1h = 0;
              let customChange24h = 0;
              let customChange7d = 0;
              let customVolume = 0;
              let customMarketCap = 0;
              let customHolders = 0;
              try {
                const oracleResponse = await qnkAPI.getOraclePrice(apiToken.address);
                if (oracleResponse.success && oracleResponse.data) {
                  customPrice = oracleResponse.data.price || 1.0;
                  customChange1h = oracleResponse.data.change_1h || 0;
                  customChange24h = oracleResponse.data.change_24h || 0;
                  customChange7d = oracleResponse.data.change_7d || 0;
                  customVolume = oracleResponse.data.volume_24h || 0;
                  customMarketCap = oracleResponse.data.market_cap || 0;
                  customHolders = oracleResponse.data.holders || 0;
                  console.log(`✅ Fetched ${apiToken.symbol} metrics from oracle:`, {
                    price: customPrice,
                    change1h: customChange1h,
                    change24h: customChange24h,
                    change7d: customChange7d,
                    volume: customVolume,
                    marketCap: customMarketCap,
                    holders: customHolders
                  });
                }
              } catch (error) {
                console.log(`ℹ️ No oracle data for ${apiToken.symbol}, using defaults`);
              }

              // Get real liquidity from pools for this token
              const tokenLiquidity = poolsByToken.get(apiToken.address) || 0;

              // Calculate market cap if not provided by oracle
              const calculatedMarketCap = customMarketCap || (actualSupply * customPrice);

              return {
                id: apiToken.address,
                symbol: apiToken.symbol,
                name: apiToken.name,
                balance: tokenBalance,
                price: customPrice,
                change1h: customChange1h,
                change24h: customChange24h,
                change7d: customChange7d,
                volume24h: customVolume,
                liquidity: tokenLiquidity,
                marketCap: calculatedMarketCap,
                totalSupply: actualSupply || 10000000,
                circulatingSupply: actualSupply || 10000000,
                holders: customHolders,
                icon: '🪙',
                features: {
                  reflection: false,
                  autoLiquidity: false,
                  buybackAndBurn: false,
                  antiWhale: false,
                  quantumSecured: true,
                },
                fees: {
                  buy: 0,
                  sell: 0,
                  transfer: 0,
                },
                description: `${apiToken.name} is a custom token deployed on the Quillon blockchain with quantum-resistant security.`,
                website: 'https://quillon.xyz',
                whitepaper: apiToken.audit_report,
              };
            });

          // Wait for all balance fetches to complete
          const apiTokens = await Promise.all(apiTokensPromises);

          // ✅ FILTER: Only show tokens that have liquidity pools (liquidity > 0)
          // This ensures token pairs are tradeable before appearing in the DEX
          const tokensWithLiquidity = apiTokens.filter(token => {
            const hasLiquidity = token.liquidity > 0;
            if (!hasLiquidity) {
              console.log(`🚫 Filtering out ${token.symbol} - no liquidity pool exists`);
            }
            return hasLiquidity;
          });

          console.log(`✅ Filtered tokens: ${tokensWithLiquidity.length} with liquidity, ${apiTokens.length - tokensWithLiquidity.length} without liquidity`);
          enrichedTokens = [...nativeTokens, ...tokensWithLiquidity];
        }

        // Convert user-deployed contracts to Token objects
        if (userDeployedTokens.length > 0) {
          console.log('🔧 Converting user deployed contracts to Token objects:', userDeployedTokens.length);
          const userTokensPromises = userDeployedTokens.map(async (contract) => {
            console.log(`🔍 Processing contract ${contract.symbol}:`, {
              address: contract.address,
              decimals: contract.decimals,
              total_supply: contract.total_supply
            });

            // Calculate actual supply using decimals
            const decimals = contract.decimals || 18;
            const rawSupply = contract.total_supply || 0;
            const actualSupply = Number(rawSupply) / Math.pow(10, decimals);

            // Fetch balance for this token if wallet is available
            let tokenBalance = 0;
            if (walletAddress) {
              try {
                const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, contract.address);
                console.log(`📊 Balance response for ${contract.symbol}:`, balanceResponse);
                if (balanceResponse.success && balanceResponse.data) {
                  // Balance from backend is in smallest units, convert to human-readable
                  const rawBalance = balanceResponse.data.balance || 0;
                  tokenBalance = rawBalance / Math.pow(10, decimals);
                  console.log(`✅ Converted ${contract.symbol} balance from ${rawBalance} to ${tokenBalance} (decimals: ${decimals})`);
                } else {
                  console.warn(`⚠️ Balance fetch unsuccessful for ${contract.symbol}:`, balanceResponse);
                }
              } catch (error) {
                console.error(`Failed to fetch balance for ${contract.symbol}:`, error);
              }
            }

            // Fetch custom token metrics from oracle (if available)
            let customPrice = 1.0;
            let customChange1h = 0;
            let customChange24h = 0;
            let customChange7d = 0;
            let customVolume = 0;
            let customMarketCap = 0;
            let customHolders = 0;
            try {
              const oracleResponse = await qnkAPI.getOraclePrice(contract.address);
              if (oracleResponse.success && oracleResponse.data) {
                customPrice = oracleResponse.data.price || 1.0;
                customChange1h = oracleResponse.data.change_1h || 0;
                customChange24h = oracleResponse.data.change_24h || 0;
                customChange7d = oracleResponse.data.change_7d || 0;
                customVolume = oracleResponse.data.volume_24h || 0;
                customMarketCap = oracleResponse.data.market_cap || 0;
                customHolders = oracleResponse.data.holders || 0;
                console.log(`✅ Fetched ${contract.symbol} metrics from oracle:`, {
                  price: customPrice,
                  change1h: customChange1h,
                  change24h: customChange24h,
                  change7d: customChange7d,
                  volume: customVolume,
                  marketCap: customMarketCap,
                  holders: customHolders
                });
              }
            } catch (error) {
              console.log(`ℹ️ No oracle data for ${contract.symbol}, using defaults`);
            }

            // Get real liquidity from pools for this token
            const tokenLiquidity = poolsByToken.get(contract.address) || 0;

            // Calculate market cap if not provided by oracle
            const calculatedMarketCap = customMarketCap || (actualSupply * customPrice);

            return {
              id: contract.address,
              symbol: contract.symbol,
              name: contract.name || contract.symbol,
              balance: tokenBalance,
              price: customPrice,
              change1h: customChange1h,
              change24h: customChange24h,
              change7d: customChange7d,
              volume24h: customVolume,
              liquidity: tokenLiquidity,
              marketCap: calculatedMarketCap,
              totalSupply: actualSupply || 0,
              circulatingSupply: actualSupply || 0,
              holders: customHolders,
              icon: '🪙',
              features: {
                reflection: false,
                autoLiquidity: false,
                buybackAndBurn: false,
                antiWhale: false,
                quantumSecured: true,
              },
              fees: {
                buy: 0,
                sell: 0,
                transfer: 0,
              },
              description: `${contract.name || contract.symbol} is a custom token deployed on the Quillon blockchain with quantum-resistant security.`,
              website: 'https://quillon.xyz',
              whitepaper: undefined,
            };
          });

          const userTokens = await Promise.all(userTokensPromises);

          // ✅ FILTER 1: Only show tokens with liquidity pools
          const userTokensWithLiquidity = userTokens.filter(token => {
            const hasLiquidity = token.liquidity > 0;
            if (!hasLiquidity) {
              console.log(`🚫 Filtering out user token ${token.symbol} - no liquidity pool exists`);
            }
            return hasLiquidity;
          });

          // ✅ FILTER 2: Filter out any user tokens that are already in enrichedTokens (avoid duplicates by symbol)
          const existingSymbols = new Set(enrichedTokens.map(t => t.symbol));
          const newUserTokens = userTokensWithLiquidity.filter(t => !existingSymbols.has(t.symbol));

          console.log(`✅ Adding ${newUserTokens.length} user tokens to Available Tokens (${userTokens.length - newUserTokens.length} filtered: ${userTokens.length - userTokensWithLiquidity.length} no liquidity, ${userTokensWithLiquidity.length - newUserTokens.length} duplicates)`);
          enrichedTokens = [...enrichedTokens, ...newUserTokens];
        }
        if (mounted) {
          setTokens(enrichedTokens);
        }
      } catch (error) {
        console.error('Failed to fetch tokens:', error);
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    // Initial fetch
    fetchTokens();

    // Set up SSE for real-time balance updates (same pattern as Dashboard)
    const currentWalletForSSE = localStorage.getItem('walletAddress') || '';
    const sseUrl = import.meta.env.VITE_API_URL ?
      `${import.meta.env.VITE_API_URL}/v1/events?wallet_address=${encodeURIComponent(currentWalletForSSE)}` :
      `/api/v1/events?wallet_address=${encodeURIComponent(currentWalletForSSE)}`;

    console.log('📡 [DEX] Connecting to SSE for balance updates:', sseUrl);

    try {
      sseEventSource = new EventSource(sseUrl);

      sseEventSource.onopen = () => {
        console.log('✅ [DEX] SSE connection established for real-time balance updates');
      };

      // Listen for balance-updated events from backend (sent after swaps, transfers, etc.)
      sseEventSource.addEventListener('balance-updated', (event: MessageEvent) => {
        if (!mounted) return;

        try {
          const data = JSON.parse(event.data);
          console.log('💰 [DEX] Balance update SSE event received:', data);

          // Validate this event is for our wallet
          const currentWalletAddress = localStorage.getItem('walletAddress');
          const currentHex = (currentWalletAddress?.startsWith('qnk')
            ? currentWalletAddress.substring(3)
            : currentWalletAddress)?.toLowerCase();
          const eventHex = data.data?.wallet_address?.toLowerCase() || data.wallet_address?.toLowerCase();

          if (currentHex && eventHex === currentHex) {
            console.log('✅ [DEX] Balance update confirmed for current wallet - refreshing tokens');
            fetchTokens();
          } else {
            console.log('⚠️ [DEX] Balance update ignored (different wallet)');
          }
        } catch (error) {
          console.error('❌ [DEX] Failed to parse balance-updated event:', error);
        }
      });

      // Listen for liquidity pool updates (when new pools are created or liquidity changes)
      sseEventSource.addEventListener('liquidity_pool_update', (event: MessageEvent) => {
        if (!mounted) return;

        try {
          const parsed = JSON.parse(event.data);
          console.log('💧 [DEX] Liquidity pool update SSE event received:', parsed);

          // Extract data from wrapper
          const poolData = parsed.data || parsed;

          // Update liquidity pools state with new/updated pool
          setLiquidityPools(prevPools => {
            const existingIndex = prevPools.findIndex(p => p.pool_id === poolData.pool_id);

            if (existingIndex >= 0) {
              // Update existing pool
              const updatedPools = [...prevPools];
              updatedPools[existingIndex] = {
                ...updatedPools[existingIndex],
                reserve0: poolData.reserve0,
                reserve1: poolData.reserve1,
                total_liquidity: poolData.total_liquidity,
              };
              console.log('✅ [DEX] Updated existing pool:', poolData.pool_id);
              return updatedPools;
            } else {
              // Add new pool
              console.log('✅ [DEX] Added new liquidity pool:', poolData.pool_id);
              return [...prevPools, {
                pool_id: poolData.pool_id,
                token0: poolData.token0,
                token1: poolData.token1,
                reserve0: poolData.reserve0,
                reserve1: poolData.reserve1,
                total_liquidity: poolData.total_liquidity,
              }];
            }
          });

          // Refresh tokens to update liquidity values
          fetchTokens();
        } catch (error) {
          console.error('❌ [DEX] Failed to parse liquidity_pool_update event:', error);
        }
      });

      sseEventSource.onerror = (error) => {
        console.error('❌ [DEX] SSE connection error:', error);
      };
    } catch (error) {
      console.error('❌ [DEX] Failed to establish SSE connection:', error);
    }

    // Listen for CDP mint events to refresh QUGUSD balance
    const handleCDPMint = () => {
      console.log('💵 CDP mint detected in DexScreen - refreshing tokens');
      fetchTokens();
    };

    // Listen for manual refresh events (from swaps)
    const handleManualRefresh = () => {
      console.log('🔄 Manual token refresh triggered - refetching balances');
      fetchTokens();
    };

    window.addEventListener('cdp-mint', handleCDPMint);
    window.addEventListener('manual-token-refresh', handleManualRefresh);

    return () => {
      mounted = false;
      window.removeEventListener('cdp-mint', handleCDPMint);
      window.removeEventListener('manual-token-refresh', handleManualRefresh);
      if (sseEventSource) {
        console.log('🔌 [DEX] Closing SSE connection');
        sseEventSource.close();
      }
    };
  }, []); // Only run once on mount - fetchTokens is called via SSE events

  // Separate effect to handle manual refresh triggers from swaps
  useEffect(() => {
    if (refreshTrigger > 0) {
      console.log('🔄 [DEX] Manual refresh triggered, refetching tokens...');
      // Dispatch a custom event that the SSE listener will pick up
      window.dispatchEvent(new CustomEvent('manual-token-refresh'));
    }
  }, [refreshTrigger]);

  // Fetch liquidity pools once on mount (SSE will handle real-time updates)
  useEffect(() => {
    const fetchPools = async () => {
      try {
        console.log('💧 [DEX] Fetching initial liquidity pools...');
        const response = await qnkAPI.getLiquidityPools();
        if (response.success && response.data) {
          setLiquidityPools(response.data);
          console.log('✅ [DEX] Loaded', response.data.length, 'liquidity pools');
        }
      } catch (error) {
        console.error('Failed to fetch liquidity pools:', error);
      }
    };

    fetchPools();
    // No polling needed - SSE will push updates in real-time
  }, []);

  // Filter tokens based on search and filter
  const filteredTokens = tokens
    .filter(token => {
      const matchesSearch = token.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           token.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           token.id.toLowerCase().includes(searchQuery.toLowerCase());

      if (filterBy === 'gainers') return matchesSearch && token.change24h > 0;
      if (filterBy === 'losers') return matchesSearch && token.change24h < 0;
      return matchesSearch;
    })
    .sort((a, b) => {
      // First, prioritize Nitro boosted tokens (sort by boost points descending)
      const aBoost = boostedTokens.get(a.id) || 0;
      const bBoost = boostedTokens.get(b.id) || 0;

      if (aBoost !== bBoost) {
        return bBoost - aBoost; // Higher boost points come first
      }

      // Then apply the regular sorting
      const multiplier = sortDirection === 'asc' ? 1 : -1;
      const aVal = a[sortBy] as number;
      const bVal = b[sortBy] as number;
      return (aVal - bVal) * multiplier;
    });

  const handleSort = (field: typeof sortBy) => {
    if (sortBy === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortDirection('desc');
    }
  };

  const formatNumber = (num: number, decimals: number = 2) => {
    if (num >= 1000000000) return `$${(num / 1000000000).toFixed(decimals)}B`;
    if (num >= 1000000) return `$${(num / 1000000).toFixed(decimals)}M`;
    if (num >= 1000) return `$${(num / 1000).toFixed(decimals)}K`;
    return `$${num.toFixed(decimals)}`;
  };

  const swapTokens = () => {
    const temp = swapFrom;
    setSwapFrom(swapTo);
    setSwapTo(temp);
  };

  const handleCloseModal = useCallback(() => {
    setSelectedToken(null);
  }, []);

  const handleCloseLiquidityModal = useCallback(() => {
    setLiquidityToken(null);
  }, []);

  const handleAddLiquidity = useCallback(async (tokenA: string, tokenB: string, amountA: number, amountB: number) => {
    console.log(`🔍 Adding liquidity - Raw inputs:`, { tokenA, tokenB, amountA, amountB, typeA: typeof amountA, typeB: typeof amountB });

    try {
      // Get wallet address from localStorage
      const walletAddress = localStorage.getItem('walletAddress');
      if (!walletAddress) {
        alert('Please connect your wallet first');
        return;
      }

      // Find token addresses (use "QUG" for native, otherwise use token ID)
      const token0 = tokenA === 'QUG' ? 'QUG' : tokens.find(t => t.symbol === tokenA)?.id || tokenA;
      const token1 = tokenB === 'QUG' ? 'QUG' : tokens.find(t => t.symbol === tokenB)?.id || tokenB;

      // Convert human-readable amounts to smallest units (base units)
      // QUG and tokens use 8 decimals (like Bitcoin): 1 QUG = 100,000,000 units
      const DECIMALS = 100_000_000; // 10^8
      const amount0 = Math.floor(amountA * DECIMALS);
      const amount1 = Math.floor(amountB * DECIMALS);

      console.log(`💰 Liquidity amounts after conversion:`, { amount0, amount1, token0, token1 });

      // Call API to add liquidity
      const response = await qnkAPI.addLiquidity({
        token0,
        token1,
        amount0,
        amount1,
        provider: walletAddress,
      });

      if (response.success && response.data) {
        alert(`✅ Liquidity added successfully!\n\nPool ID: ${response.data.pool_id}\nTransaction: ${response.data.transaction_id}\n\nYou will receive LP tokens representing your pool share.`);

        // Refresh tokens to update balances
        window.location.reload();
      } else {
        alert(`❌ Failed to add liquidity: ${response.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to add liquidity:', error);
      alert('❌ Failed to add liquidity. Please try again.');
    }
  }, [tokens]);

  const handleAddCustomTokenLiquidity = async () => {
    if (!customTokenAddress || customTokenAddress.length < 10) {
      alert('Please enter a valid token contract address');
      return;
    }

    try {
      // Fetch real token information from the blockchain
      const contractInfo = await qnkAPI.getContractInfo(customTokenAddress);

      if (!contractInfo.success || !contractInfo.data) {
        alert(`Failed to fetch token information: ${contractInfo.error || 'Token contract not found'}`);
        return;
      }

      const contract = contractInfo.data;

      // Get wallet address for balance check
      const walletAddress = localStorage.getItem('walletAddress') || '';
      let tokenBalance = 0;

      if (walletAddress) {
        // Try to fetch token balance
        const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, customTokenAddress);
        if (balanceResponse.success && balanceResponse.data) {
          tokenBalance = balanceResponse.data.balance || 0;
        }
      }

      // Create token object with real data from blockchain
      // Calculate actual supply using decimals
      const decimals = contract.decimals || 18;
      const rawSupply = contract.total_supply || 0;
      const actualSupply = Number(rawSupply) / Math.pow(10, decimals);

      const customToken: Token = {
        id: customTokenAddress,
        symbol: contract.token_symbol || contract.symbol || 'CUSTOM',
        name: contract.token_name || contract.name || 'Custom Token',
        balance: tokenBalance, // Balance is already in human-readable form from backend
        price: 1.0, // Default price, can be calculated from liquidity pools later
        change1h: 0,
        change24h: 0,
        change7d: 0,
        volume24h: 0,
        liquidity: actualSupply, // Use calculated supply
        marketCap: 0,
        totalSupply: actualSupply, // Use calculated supply
        circulatingSupply: actualSupply, // Use calculated supply
        holders: 0,
        icon: '🪙',
        features: {
          reflection: contract.features?.reflection || false,
          autoLiquidity: contract.features?.autoLiquidity || false,
          buybackAndBurn: contract.features?.buybackAndBurn || false,
          antiWhale: contract.features?.antiWhale || false,
          quantumSecured: true,
        },
        fees: {
          buy: contract.fees?.buy || 0,
          sell: contract.fees?.sell || 0,
          transfer: contract.fees?.transfer || 0,
        },
        description: contract.description || `${contract.token_name || 'Custom token'} deployed on Quillon blockchain`,
      };

      console.log('✅ Loaded custom token:', customToken);
      setLiquidityToken(customToken);
      setCustomTokenAddress('');
    } catch (error) {
      console.error('Failed to fetch custom token info:', error);
      alert('Failed to fetch token information. Please check the contract address and try again.');
    }
  };

  const handleNitroBoost = (token: Token) => {
    setNitroBoostToken(token);
    // Reset boost cost to default or clamp to available points
    const maxBoost = Math.min(500, nitroPoints);
    if (boostCost > maxBoost) {
      setBoostCost(Math.max(50, maxBoost));
    } else if (boostCost < 50) {
      setBoostCost(50);
    }
  };

  const confirmNitroBoost = async () => {
    if (!nitroBoostToken) return;

    // Check if user has enough points
    if (nitroPoints < boostCost) {
      alert(`❌ Insufficient Nitro Points!\n\nYou need ${boostCost} points but only have ${nitroPoints} points.\n\nClick on the Nitro Points display in the topbar to purchase more points.`);
      return;
    }

    // Get wallet address
    const walletAddress = localStorage.getItem('walletAddress');
    if (!walletAddress) {
      alert('❌ Please connect your wallet first');
      return;
    }

    try {
      // Post boost to backend
      const response = await qnkAPI.addNitroBoost(
        nitroBoostToken.id,
        boostCost,
        walletAddress
      );

      if (!response.success) {
        alert(`❌ Failed to add Nitro boost: ${response.error}`);
        return;
      }

      // Deduct points from local state (per wallet address)
      const newPoints = nitroPoints - boostCost;
      setNitroPoints(newPoints);
      localStorage.setItem(`nitroPoints_${walletAddress}`, newPoints.toString());

      // Update local boosted tokens map
      const newBoosted = new Map(boostedTokens);
      const existingPoints = newBoosted.get(nitroBoostToken.id) || 0;
      newBoosted.set(nitroBoostToken.id, existingPoints + boostCost);
      setBoostedTokens(newBoosted);

      // Show visual effect
      setNitroBoostTokens(prev => {
        const newSet = new Set(prev);
        newSet.add(nitroBoostToken.id);
        return newSet;
      });

      setTimeout(() => {
        setNitroBoostTokens(prev => {
          const newSet = new Set(prev);
          newSet.delete(nitroBoostToken.id);
          return newSet;
        });
      }, 2000);

      setNitroBoostToken(null);
      setBoostCost(100);

      // Show success modal instead of alert
      setSuccessModalData({
        tokenSymbol: nitroBoostToken.symbol,
        points: boostCost,
        remainingPoints: newPoints,
        totalBoost: existingPoints + boostCost
      });
      setShowSuccessModal(true);

    } catch (error) {
      console.error('Failed to add Nitro boost:', error);
      alert('❌ Failed to add Nitro boost. Please try again.');
    }
  };

  const handleRemoveLiquidity = async () => {
    if (!removingPool) return;

    try {
      const walletAddress = localStorage.getItem('walletAddress');
      if (!walletAddress) {
        alert('Please connect your wallet first');
        return;
      }

      const response = await qnkAPI.removeLiquidity({
        pool_id: removingPool.pool_id,
        percentage: removePercentage,
        provider: walletAddress,
      });

      if (response.success && response.data) {
        alert(`✅ Liquidity removed successfully!\n\n${response.data.amount0_returned.toLocaleString()} ${removingPool.token0} returned\n${response.data.amount1_returned.toLocaleString()} ${removingPool.token1} returned\n\nTransaction: ${response.data.transaction_id}`);

        // Refresh page to update balances and pools
        window.location.reload();
      } else {
        alert(`❌ Failed to remove liquidity: ${response.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to remove liquidity:', error);
      alert('❌ Failed to remove liquidity. Please try again.');
    }

    setRemovingPool(null);
    setRemovePercentage(50);
  };

  const handleSelectFromToken = (token: Token) => {
    setSwapFrom(token.symbol);
    setIsFromTokenSelectorOpen(false);
  };

  const handleSelectToToken = (token: Token) => {
    setSwapTo(token.symbol);
    setIsToTokenSelectorOpen(false);
  };

  // Helper function to find token by symbol or ID (case-insensitive)
  const findToken = (symbolOrId: string) => {
    return tokens.find(t =>
      t.symbol.toUpperCase() === symbolOrId.toUpperCase() ||
      t.id === symbolOrId
    );
  };

  return (
    <>
      {/* Token Details Modal */}
      {selectedToken && (
        <TokenDetailsModal
          token={selectedToken}
          onClose={handleCloseModal}
        />
      )}

      {/* Liquidity Modal */}
      {liquidityToken && (
        <LiquidityModal
          token={liquidityToken}
          availableTokens={tokens}
          onClose={handleCloseLiquidityModal}
          onAddLiquidity={handleAddLiquidity}
        />
      )}

      {/* Token Selector Modal - From Token */}
      <TokenSelectorModal
        isOpen={isFromTokenSelectorOpen}
        onClose={() => setIsFromTokenSelectorOpen(false)}
        onSelectToken={handleSelectFromToken}
        tokens={tokens}
        boostedTokens={boostedTokens}
        currentToken={findToken(swapFrom)}
      />

      {/* Token Selector Modal - To Token */}
      <TokenSelectorModal
        isOpen={isToTokenSelectorOpen}
        onClose={() => setIsToTokenSelectorOpen(false)}
        onSelectToken={handleSelectToToken}
        tokens={tokens}
        boostedTokens={boostedTokens}
        currentToken={findToken(swapTo)}
      />

      {/* Remove Liquidity Modal */}
      {removingPool && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setRemovingPool(null)}
            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="relative w-full max-w-md"
          >
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-red-500 to-orange-500 rounded-2xl blur-xl opacity-50" />

              <div className="relative bg-black border border-red-500/30 rounded-2xl p-6">
                <h2 className="text-2xl font-bold bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent mb-4">
                  Remove Liquidity
                </h2>

                <div className="space-y-4">
                  <div className="p-4 bg-white/5 rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-400">Pool:</span>
                      <span className="text-white font-bold">{removingPool.token0} / {removingPool.token1}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Your Reserves:</span>
                      <div className="text-right">
                        <div className="text-white text-sm">{(removingPool.reserve0 / 100_000_000).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 8 })} {removingPool.token0}</div>
                        <div className="text-white text-sm">{(removingPool.reserve1 / 100_000_000).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 8 })} {removingPool.token1}</div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm text-gray-400">Percentage to Remove: {removePercentage}%</label>
                    <input
                      type="range"
                      min="1"
                      max="100"
                      value={removePercentage}
                      onChange={(e) => setRemovePercentage(Number(e.target.value))}
                      className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, rgb(239, 68, 68) 0%, rgb(239, 68, 68) ${removePercentage}%, rgba(255,255,255,0.1) ${removePercentage}%, rgba(255,255,255,0.1) 100%)`
                      }}
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>1%</span>
                      <span>25%</span>
                      <span>50%</span>
                      <span>75%</span>
                      <span>100%</span>
                    </div>
                  </div>

                  <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
                    <div className="text-sm text-gray-300 mb-2">You will receive:</div>
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span className="text-gray-400">{removingPool.token0}:</span>
                        <span className="text-white font-bold">{((removingPool.reserve0 / 100_000_000) * removePercentage / 100).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 8 })}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">{removingPool.token1}:</span>
                        <span className="text-white font-bold">{((removingPool.reserve1 / 100_000_000) * removePercentage / 100).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 8 })}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <button
                      onClick={() => setRemovingPool(null)}
                      className="flex-1 py-3 bg-white/10 hover:bg-white/20 rounded-xl text-white font-medium transition-all"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleRemoveLiquidity}
                      className="flex-1 py-3 bg-gradient-to-r from-red-500 to-orange-500 rounded-xl text-white font-bold hover:shadow-lg hover:shadow-red-500/50 transition-all"
                    >
                      Remove {removePercentage}%
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Nitro Boost Modal */}
      {nitroBoostToken && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setNitroBoostToken(null)}
            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="relative w-full max-w-md"
          >
            <div className="relative group">
              {/* Animated glow effect */}
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

              <div className="relative bg-black border-2 border-orange-500/50 rounded-2xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Zap className="w-8 h-8 text-yellow-400" />
                  <h2 className="text-2xl font-bold bg-gradient-to-r from-orange-400 via-yellow-400 to-red-400 bg-clip-text text-transparent">
                    Nitro Boost
                  </h2>
                </div>

                <div className="space-y-4">
                  {/* Token Info */}
                  <div className="p-4 bg-gradient-to-br from-orange-500/10 to-red-500/10 border border-orange-500/30 rounded-xl">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="text-3xl">
                        {(nitroBoostToken.icon === 'qug-logo' || nitroBoostToken.icon === 'qugusd-logo' || nitroBoostToken.icon === 'usd-logo') ? (
                          <div className="relative w-10 h-10">
                            <div className="absolute inset-0 rounded-full" style={{
                              background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                              padding: '2px'
                            }}>
                              <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center p-1">
                                <img
                                  src="/quillon-logo.png"
                                  alt="Quillon"
                                  className="w-full h-full object-contain"
                                  style={{ filter: 'invert(1)' }}
                                />
                              </div>
                            </div>
                          </div>
                        ) : (
                          nitroBoostToken.icon
                        )}
                      </div>
                      <div>
                        <div className="font-bold text-white text-lg">{nitroBoostToken.symbol}</div>
                        <div className="text-sm text-gray-400">{nitroBoostToken.name}</div>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-gray-400">Price:</span>
                        <span className="text-white font-bold ml-2">${nitroBoostToken.price.toLocaleString()}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">24h:</span>
                        <span className={`font-bold ml-2 ${nitroBoostToken.change24h > 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {nitroBoostToken.change24h > 0 ? '+' : ''}{nitroBoostToken.change24h.toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Current Nitro Points */}
                  <div className="p-4 bg-orange-500/10 border border-orange-500/30 rounded-xl">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-orange-300/70">Your Nitro Points</span>
                      <span className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-yellow-500 bg-clip-text text-transparent">
                        {nitroPoints.toLocaleString()} pts
                      </span>
                    </div>
                    <div className="text-xs text-orange-300/50">
                      Use points to boost tokens to the top of the DEX
                    </div>
                  </div>

                  {/* Current Boost Level */}
                  {boostedTokens.has(nitroBoostToken.id) && (
                    <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-xl">
                      <div className="text-sm text-yellow-300/70 mb-1">Current Boost Level</div>
                      <div className="text-xl font-bold text-yellow-400">
                        {boostedTokens.get(nitroBoostToken.id)} points invested
                      </div>
                    </div>
                  )}

                  {/* Boost Amount Selector */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="text-sm text-orange-300/70">Points to Spend</label>
                      <motion.span
                        key={boostCost}
                        initial={{ scale: 1.2 }}
                        animate={{ scale: 1 }}
                        className="text-xl font-bold bg-gradient-to-r from-orange-400 to-yellow-500 bg-clip-text text-transparent"
                      >
                        {boostCost} pts
                      </motion.span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max={Math.min(500, nitroPoints)}
                      step="50"
                      value={boostCost}
                      onChange={(e) => setBoostCost(parseInt(e.target.value))}
                      className="w-full h-2 appearance-none rounded-full cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, #FF8C00 0%, #FFA500 ${((boostCost - 50) / (Math.min(500, nitroPoints) - 50)) * 100}%, rgba(255,140,0,0.2) ${((boostCost - 50) / (Math.min(500, nitroPoints) - 50)) * 100}%, rgba(255,140,0,0.2) 100%)`
                      }}
                      disabled={nitroPoints < 50}
                    />
                    <div className="flex justify-between text-xs text-orange-300/50">
                      <span>Min: 50 pts</span>
                      <span>Max: {Math.min(500, nitroPoints)} pts</span>
                    </div>
                  </div>

                  {/* Benefits */}
                  <div className="p-4 bg-gradient-to-br from-orange-500/10 to-yellow-500/10 border border-orange-500/20 rounded-xl">
                    <div className="text-sm text-orange-300/70 mb-2">Boost Benefits:</div>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-sm text-white/80">
                        <Zap className="w-4 h-4 text-orange-400" />
                        <span>Promote token to top of DEX listing</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-white/80">
                        <TrendingUp className="w-4 h-4 text-green-400" />
                        <span>Increase visibility & trading volume</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-white/80">
                        <Zap className="w-4 h-4 text-yellow-400" />
                        <span>Premium visual effects on activation</span>
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-3 mt-6">
                    <button
                      onClick={() => setNitroBoostToken(null)}
                      className="flex-1 py-3 bg-white/10 hover:bg-white/20 rounded-xl text-white font-medium transition-all"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={confirmNitroBoost}
                      className="flex-1 py-3 bg-gradient-to-r from-orange-500 via-yellow-500 to-red-500 rounded-xl text-white font-bold hover:shadow-lg hover:shadow-orange-500/50 transition-all flex items-center justify-center gap-2"
                    >
                      <Zap className="w-5 h-5" />
                      Activate Nitro
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}

      <div className="space-y-6">

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-black bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
            Quantum DEX
          </h1>
          <p className="text-gray-400 mt-1">Decentralized exchange with quantum security</p>
        </div>
      </motion.div>

      {loading ? (
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="inline-block w-12 h-12 border-4 border-quantum-cyan/30 border-t-quantum-cyan rounded-full animate-spin mb-4" />
            <p className="text-gray-400">Loading tokens...</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Swap Interface */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="relative group">
              {/* Glow effect */}
              <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity" />

              <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-quantum-cyan/20 p-6 space-y-4">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-white">Swap Tokens</h2>
                  <button className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors">
                    <Settings className="w-5 h-5 text-gray-400" />
                  </button>
                </div>

              {/* From Token */}
              <div className="space-y-2">
                <label className="text-sm text-gray-400">From</label>
                <div className="relative">
                  <input
                    type="number"
                    value={swapAmount}
                    onChange={(e) => setSwapAmount(e.target.value)}
                    placeholder="0.0"
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-4 pr-36 text-white text-xl focus:outline-none focus:border-quantum-cyan/50 transition-colors"
                  />
                  <button
                    onClick={() => setIsFromTokenSelectorOpen(true)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/30 rounded-lg px-3 py-2 text-white font-bold cursor-pointer focus:outline-none transition-all flex items-center gap-2"
                  >
                    {/* Proper Logo for QUG */}
                    {swapFrom === 'QUG' ? (
                      <div className="relative w-6 h-6">
                        <div className="absolute inset-0 rounded-full" style={{
                          background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                          padding: '1px'
                        }}>
                          <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center">
                            <span className="text-yellow-400 font-bold text-xs">Q</span>
                          </div>
                        </div>
                      </div>
                    ) : swapFrom === 'QUGUSD' ? (
                      /* Proper Logo for QUGUSD */
                      <div className="relative w-6 h-6">
                        <div className="absolute inset-0 rounded-full" style={{
                          background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                          padding: '1px'
                        }}>
                          <div className="w-full h-full bg-gradient-to-b from-slate-900 via-emerald-950 to-slate-900 rounded-full flex items-center justify-center">
                            <span className="text-green-400 font-bold text-xs">$</span>
                          </div>
                        </div>
                      </div>
                    ) : swapFrom === 'USD' ? (
                      /* Proper Logo for USD */
                      <div className="relative w-6 h-6">
                        <div className="absolute inset-0 rounded-full" style={{
                          background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                          padding: '1px'
                        }}>
                          <div className="w-full h-full bg-gradient-to-b from-slate-900 via-green-950 to-slate-900 rounded-full flex items-center justify-center">
                            <span className="text-green-400 font-bold text-xs">$</span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <span className="text-xl">{findToken(swapFrom)?.icon || '💎'}</span>
                    )}
                    <span>{swapFrom}</span>
                    <span className="text-xs opacity-70">▼</span>
                  </button>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-500">
                    Balance: {findToken(swapFrom)?.balance.toFixed(4) || '0.0000'}
                  </span>
                  <button
                    onClick={() => {
                      const fromToken = findToken(swapFrom);
                      if (fromToken) {
                        setSwapAmount(fromToken.balance.toString());
                      }
                    }}
                    className="text-quantum-cyan hover:text-quantum-purple transition-colors font-medium"
                  >
                    MAX
                  </button>
                </div>
              </div>

              {/* KILLER AWESOME SLIDER */}
              <div className="space-y-3 py-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm text-gray-400">Quick Select Amount</label>
                  <span className="text-xs font-bold bg-gradient-to-r from-quantum-cyan to-quantum-purple bg-clip-text text-transparent">
                    {(() => {
                      const fromToken = findToken(swapFrom);
                      if (!fromToken || !swapAmount) return '0%';
                      const percentage = (parseFloat(swapAmount) / fromToken.balance) * 100;
                      return percentage.toFixed(0) + '%';
                    })()}
                  </span>
                </div>
                <div className="relative">
                  {/* Slider Track with Gradient */}
                  <div className="h-3 bg-white/5 rounded-full overflow-hidden relative">
                    <motion.div
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{
                        background: 'linear-gradient(90deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%)',
                        width: `${(() => {
                          const fromToken = findToken(swapFrom);
                          if (!fromToken || !swapAmount) return 0;
                          return Math.min((parseFloat(swapAmount) / fromToken.balance) * 100, 100);
                        })()}%`
                      }}
                      animate={{
                        boxShadow: [
                          '0 0 10px rgba(6, 182, 212, 0.5)',
                          '0 0 20px rgba(139, 92, 246, 0.8)',
                          '0 0 10px rgba(236, 72, 153, 0.5)',
                          '0 0 20px rgba(139, 92, 246, 0.8)',
                          '0 0 10px rgba(6, 182, 212, 0.5)',
                        ]
                      }}
                      transition={{
                        duration: 3,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    />
                  </div>

                  {/* Slider Input */}
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="1"
                    value={(() => {
                      const fromToken = findToken(swapFrom);
                      if (!fromToken || !swapAmount) return 0;
                      return Math.min((parseFloat(swapAmount) / fromToken.balance) * 100, 100);
                    })()}
                    onChange={(e) => {
                      const fromToken = findToken(swapFrom);
                      if (fromToken) {
                        const percentage = parseFloat(e.target.value) / 100;
                        const amount = fromToken.balance * percentage;
                        setSwapAmount(amount.toFixed(8));
                      }
                    }}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                </div>

                {/* Quick Select Buttons */}
                <div className="flex gap-2">
                  {[25, 50, 75, 100].map((percentage) => (
                    <motion.button
                      key={percentage}
                      onClick={() => {
                        const fromToken = findToken(swapFrom);
                        if (fromToken) {
                          const amount = fromToken.balance * (percentage / 100);
                          setSwapAmount(amount.toFixed(8));
                        }
                      }}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="flex-1 py-2 bg-white/5 hover:bg-gradient-to-r hover:from-quantum-cyan/20 hover:to-quantum-purple/20 border border-white/10 hover:border-quantum-cyan/50 rounded-lg text-xs font-medium text-gray-400 hover:text-white transition-all"
                    >
                      {percentage}%
                    </motion.button>
                  ))}
                </div>
              </div>

              {/* Swap Button */}
              <div className="flex justify-center -my-2">
                <button
                  onClick={swapTokens}
                  className="p-3 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-full hover:scale-110 transition-transform"
                >
                  <ArrowDownUp className="w-5 h-5 text-white" />
                </button>
              </div>

              {/* To Token */}
              <div className="space-y-2">
                <label className="text-sm text-gray-400">To (Estimated)</label>
                <div className="relative">
                  <input
                    type="text"
                    value={(() => {
                      if (!swapAmount) return '';
                      const fromToken = findToken(swapFrom);
                      const toToken = findToken(swapTo);
                      if (!fromToken || !toToken) return '';
                      // Calculate actual exchange rate using oracle prices
                      // Account for 0.3% DEX fee
                      const exchangeRate = (fromToken.price / toToken.price) * 0.997;
                      return (parseFloat(swapAmount) * exchangeRate).toFixed(4);
                    })()}
                    placeholder="0.0"
                    readOnly
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-4 pr-36 text-white text-xl focus:outline-none"
                  />
                  <button
                    onClick={() => setIsToTokenSelectorOpen(true)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/30 rounded-lg px-3 py-2 text-white font-bold cursor-pointer focus:outline-none transition-all flex items-center gap-2"
                  >
                    {/* Proper Logo for QUG */}
                    {swapTo === 'QUG' ? (
                      <div className="relative w-6 h-6">
                        <div className="absolute inset-0 rounded-full" style={{
                          background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                          padding: '1px'
                        }}>
                          <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center">
                            <span className="text-yellow-400 font-bold text-xs">Q</span>
                          </div>
                        </div>
                      </div>
                    ) : swapTo === 'QUGUSD' ? (
                      /* Proper Logo for QUGUSD */
                      <div className="relative w-6 h-6">
                        <div className="absolute inset-0 rounded-full" style={{
                          background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                          padding: '1px'
                        }}>
                          <div className="w-full h-full bg-gradient-to-b from-slate-900 via-emerald-950 to-slate-900 rounded-full flex items-center justify-center">
                            <span className="text-green-400 font-bold text-xs">$</span>
                          </div>
                        </div>
                      </div>
                    ) : swapTo === 'USD' ? (
                      /* Proper Logo for USD */
                      <div className="relative w-6 h-6">
                        <div className="absolute inset-0 rounded-full" style={{
                          background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                          padding: '1px'
                        }}>
                          <div className="w-full h-full bg-gradient-to-b from-slate-900 via-green-950 to-slate-900 rounded-full flex items-center justify-center">
                            <span className="text-green-400 font-bold text-xs">$</span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <span className="text-xl">{findToken(swapTo)?.icon || '💵'}</span>
                    )}
                    <span>{swapTo}</span>
                    <span className="text-xs opacity-70">▼</span>
                  </button>
                </div>
                <div className="text-xs text-gray-500 flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" />
                  Price includes 0.3% DEX fee
                </div>
              </div>

              {/* Swap Info */}
              <div className="space-y-2 text-sm p-4 bg-white/5 rounded-xl">
                <div className="flex justify-between text-gray-400">
                  <span>Rate</span>
                  <span className="text-white">
                    1 {swapFrom} ≈ {(() => {
                      const fromToken = findToken(swapFrom);
                      const toToken = findToken(swapTo);
                      if (!fromToken || !toToken) return '0.00';
                      // Calculate actual exchange rate using oracle prices (before fees)
                      const exchangeRate = fromToken.price / toToken.price;
                      return exchangeRate.toFixed(2);
                    })()} {swapTo}
                  </span>
                </div>
                <div className="flex justify-between text-gray-400">
                  <span>Slippage</span>
                  <span className="text-white">0.5%</span>
                </div>
                <div className="flex justify-between text-gray-400">
                  <span>Fee</span>
                  <span className="text-white">0.3%</span>
                </div>
              </div>

              {/* Swap Button */}
              <button
                onClick={async () => {
                  if (!swapAmount || parseFloat(swapAmount) <= 0) {
                    alert('Please enter a valid swap amount');
                    return;
                  }

                  const walletAddress = localStorage.getItem('walletAddress');
                  if (!walletAddress) {
                    alert('Please connect your wallet first');
                    return;
                  }

                  // ✅ Robust token lookup: match by symbol (case-insensitive) or ID
                  const fromToken = findToken(swapFrom);
                  const toToken = findToken(swapTo);

                  if (!fromToken || !toToken) {
                    console.error('❌ Token lookup failed:', {
                      swapFrom,
                      swapTo,
                      fromToken: fromToken?.symbol,
                      toToken: toToken?.symbol,
                      availableTokens: tokens.map(t => ({ symbol: t.symbol, id: t.id, balance: t.balance }))
                    });
                    alert(`Invalid token selection. Could not find: ${!fromToken ? swapFrom : swapTo}`);
                    return;
                  }

                  console.log('✅ Token lookup successful:', {
                    fromToken: { symbol: fromToken.symbol, id: fromToken.id, balance: fromToken.balance },
                    toToken: { symbol: toToken.symbol, id: toToken.id, balance: toToken.balance }
                  });

                  // Handle USD (Stripe balance) swaps - convert to QUGUSD first
                  if (fromToken.id === 'fiat-usd') {
                    // USD → anything: convert USD to QUGUSD, then swap QUGUSD → target
                    try {
                      const usdAmount = parseFloat(swapAmount);

                      // Step 1: Convert USD to QUGUSD (1:1 conversion, 0.1% fee)
                      const qugusdAmount = usdAmount * 0.999; // 0.1% conversion fee

                      // Deduct USD balance and mint QUGUSD
                      const convertResponse = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/convert-to-qugusd`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          wallet_address: walletAddress,
                          usd_amount: swapAmount,
                        }),
                      });

                      if (!convertResponse.ok) {
                        const errorData = await convertResponse.json();
                        alert(`❌ USD conversion failed: ${errorData.error || 'Unknown error'}`);
                        return;
                      }

                      const convertData = await convertResponse.json();
                      if (!convertData.success) {
                        alert(`❌ USD conversion failed: ${convertData.error || 'Unknown error'}`);
                        return;
                      }

                      // Step 2: If target is QUGUSD, we're done
                      if (toToken.id === 'qugusd-stable') {
                        setSwapSuccessData({
                          fromToken: 'USD',
                          toToken: 'QUGUSD',
                          fromAmount: parseFloat(swapAmount),
                          toAmount: qugusdAmount,
                          transactionHash: `${Date.now().toString(16)}-usd-conversion`
                        });
                        setShowSwapSuccess(true);
                        setSwapAmount('');
                        // Trigger balance refresh
                        setRefreshTrigger(prev => prev + 1);
                        return;
                      }

                      // Step 3: If target is something else, swap QUGUSD → target
                      // ✅ Use constant product formula for pool-based swaps
                      const toTokenFormatted = toToken.id === 'native-qug' ? 'QUG' : toToken.id;
                      const matchingPool = liquidityPools.find(pool => {
                        const pool0Upper = pool.token0.toUpperCase();
                        const pool1Upper = pool.token1.toUpperCase();
                        return (pool0Upper === 'QUGUSD' && pool1Upper === toTokenFormatted.toUpperCase()) ||
                               (pool0Upper === toTokenFormatted.toUpperCase() && pool1Upper === 'QUGUSD');
                      });

                      let expectedOutput: number;
                      let minOutput: number;

                      if (matchingPool) {
                        // Use constant product formula
                        const fee = 0.003;
                        const amountInWithFee = qugusdAmount * (1 - fee);
                        const isForward = matchingPool.token0.toUpperCase() === 'QUGUSD';
                        const reserveIn = isForward ? matchingPool.reserve0 / 100_000_000 : matchingPool.reserve1 / 100_000_000;
                        const reserveOut = isForward ? matchingPool.reserve1 / 100_000_000 : matchingPool.reserve0 / 100_000_000;

                        expectedOutput = (amountInWithFee * reserveOut) / (reserveIn + amountInWithFee);
                        minOutput = expectedOutput * 0.995;

                        console.log('💱 USD->Token swap using pool reserves:', {
                          pool: matchingPool.pool_id,
                          reserveIn,
                          reserveOut,
                          expectedOutput,
                          minOutput
                        });
                      } else {
                        // No pool - use oracle pricing (backend will handle)
                        expectedOutput = qugusdAmount * (1.0 / toToken.price);
                        minOutput = expectedOutput * 0.95; // More lenient for oracle
                        console.log('💱 USD->Token swap using oracle pricing');
                      }

                      const swapResponse = await qnkAPI.executeSwap({
                        from_token: 'QUGUSD',
                        to_token: toTokenFormatted,
                        amount_in: Math.floor(qugusdAmount * 100_000_000),
                        min_amount_out: Math.floor(minOutput * 100_000_000),
                        wallet_address: walletAddress
                      });

                      if (swapResponse.success && swapResponse.data) {
                        setSwapSuccessData({
                          fromToken: 'USD',
                          toToken: swapTo,
                          fromAmount: parseFloat(swapAmount),
                          toAmount: swapResponse.data.amount_out / 100_000_000,
                          transactionHash: swapResponse.data.transaction_id
                        });
                        setShowSwapSuccess(true);
                        setSwapAmount('');
                        // Trigger balance refresh
                        setRefreshTrigger(prev => prev + 1);
                      } else {
                        alert(`❌ Swap failed after USD conversion: ${swapResponse.error || 'Unknown error'}\n\nYour USD was converted to QUGUSD but the swap failed.`);
                      }
                    } catch (error) {
                      console.error('USD swap failed:', error);
                      alert('❌ USD swap failed. Please try again.');
                    }
                    return;
                  }

                  // Prevent swapping TO USD (can only swap FROM USD)
                  if (toToken.id === 'fiat-usd') {
                    alert('❌ Cannot swap to USD directly.\n\nUSD is your Stripe wallet balance (off-chain). You can:\n1. Swap tokens → QUGUSD\n2. Withdraw QUGUSD to USD via bank transfer (coming soon)');
                    return;
                  }

                  if (fromToken.balance < parseFloat(swapAmount)) {
                    alert(`Insufficient ${swapFrom} balance. You have ${fromToken.balance.toFixed(4)}`);
                    return;
                  }

                  // Helper function to format token ID for backend
                  const formatTokenForBackend = (tokenId: string): string => {
                    // Handle special cases for native tokens
                    if (tokenId === 'native-qug') return 'QUG';
                    if (tokenId === 'qugusd-stable') return 'QUGUSD';

                    // Custom tokens: Backend expects addresses WITH "qnk" prefix
                    // DO NOT strip the prefix - backend parse_wallet_address() requires it
                    // Return as-is for all other cases (including custom token addresses)
                    return tokenId;
                  };

                  try{

                    // ✅ PROPER FIX: Calculate expected output using constant product formula (x * y = k)
                    // Find matching liquidity pool
                    const fromTokenFormatted = formatTokenForBackend(fromToken.id);
                    const toTokenFormatted = formatTokenForBackend(toToken.id);

                    const matchingPool = liquidityPools.find(pool => {
                      const pool0Upper = pool.token0.toUpperCase();
                      const pool1Upper = pool.token1.toUpperCase();
                      const fromUpper = fromTokenFormatted.toUpperCase();
                      const toUpper = toTokenFormatted.toUpperCase();

                      return (pool0Upper === fromUpper && pool1Upper === toUpper) ||
                             (pool0Upper === toUpper && pool1Upper === fromUpper);
                    });

                    let expectedOutput: number;
                    let minOutput: number;

                    if (matchingPool) {
                      // Use constant product formula: amount_out = (amount_in * reserve_out) / (reserve_in + amount_in)
                      // Apply 0.3% trading fee
                      const amountIn = parseFloat(swapAmount);
                      const fee = 0.003; // 0.3%
                      const amountInWithFee = amountIn * (1 - fee);

                      // Determine if we're swapping forward or reverse in the pool
                      const isForward = matchingPool.token0.toUpperCase() === fromTokenFormatted.toUpperCase();
                      const reserveIn = isForward ? matchingPool.reserve0 / 100_000_000 : matchingPool.reserve1 / 100_000_000;
                      const reserveOut = isForward ? matchingPool.reserve1 / 100_000_000 : matchingPool.reserve0 / 100_000_000;

                      // Constant product formula
                      expectedOutput = (amountInWithFee * reserveOut) / (reserveIn + amountInWithFee);
                      minOutput = expectedOutput * 0.995; // 0.5% slippage tolerance

                      console.log('💱 Swap calculation using pool reserves:', {
                        pool: matchingPool.pool_id,
                        reserveIn,
                        reserveOut,
                        amountIn,
                        amountInWithFee,
                        expectedOutput,
                        minOutput
                      });
                    } else if ((fromTokenFormatted.toUpperCase() === 'QUG' && toTokenFormatted.toUpperCase() === 'QUGUSD') ||
                               (fromTokenFormatted.toUpperCase() === 'QUGUSD' && toTokenFormatted.toUpperCase() === 'QUG')) {
                      // No pool exists - use oracle pricing for QUG<->QUGUSD
                      // The backend will handle this with oracle pricing
                      expectedOutput = parseFloat(swapAmount) * (fromToken.price / toToken.price);
                      minOutput = expectedOutput * 0.95; // More lenient slippage for oracle-based swaps

                      console.log('💱 No pool found - using oracle pricing (backend will handle):', {
                        expectedOutput,
                        minOutput
                      });
                    } else {
                      // No pool and not QUG<->QUGUSD - this will fail but let backend handle the error
                      expectedOutput = parseFloat(swapAmount) * (fromToken.price / toToken.price);
                      minOutput = expectedOutput * 0.995;

                      console.warn('⚠️ No pool found for this token pair:', fromTokenFormatted, '<->', toTokenFormatted);
                    }

                    const response = await qnkAPI.executeSwap({
                      from_token: fromTokenFormatted,
                      to_token: toTokenFormatted,
                      amount_in: Math.floor(parseFloat(swapAmount) * 100_000_000), // 8 decimals (1e8)
                      min_amount_out: Math.floor(minOutput * 100_000_000), // 8 decimals (1e8)
                      wallet_address: walletAddress
                    });

                    if (response.success && response.data) {
                      setSwapSuccessData({
                        fromToken: swapFrom,
                        toToken: swapTo,
                        fromAmount: parseFloat(swapAmount),
                        toAmount: response.data.amount_out / 100_000_000,
                        transactionHash: response.data.transaction_id
                      });
                      setShowSwapSuccess(true);
                      // Reset swap amount
                      setSwapAmount('');
                      // Trigger balance refresh
                      setRefreshTrigger(prev => prev + 1);
                      console.log('🔄 Swap completed - refreshing balances');
                    } else {
                      console.error('❌ Swap API error:', response.error);
                      console.error('❌ Full response:', response);
                      alert(`❌ Swap failed: ${response.error || 'Unknown error'}`);
                    }
                  } catch (error) {
                    console.error('❌ Swap exception:', error);
                    console.error('❌ Swap request details:', {
                      from_token: formatTokenForBackend(fromToken.id),
                      to_token: formatTokenForBackend(toToken.id),
                      amount_in: Math.floor(parseFloat(swapAmount) * 100_000_000),
                      wallet_address: walletAddress
                    });
                    alert(`❌ Swap failed: ${error instanceof Error ? error.message : 'Please try again'}`);
                  }
                }}
                className="w-full py-4 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
              >
                Swap Tokens
              </button>
            </div>
          </div>

          {/* Custom Token Liquidity Section */}
          <div className="relative group mt-6">
            {/* Glow effect */}
            <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity" />

            <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-quantum-purple/20 p-6 space-y-4">
              <div className="flex items-center gap-3 mb-4">
                <Droplet className="w-6 h-6 text-quantum-purple" />
                <h2 className="text-xl font-bold text-white">Add Custom Token Liquidity</h2>
              </div>

              <p className="text-sm text-gray-400 mb-4">
                Created a token via VM? Enter your token address to add liquidity and pair it with QUG or QUGUSD.
              </p>

              {/* Token Address Input */}
              <div className="space-y-2">
                <label className="text-sm text-gray-400">Token Contract Address</label>
                <input
                  type="text"
                  value={customTokenAddress}
                  onChange={(e) => setCustomTokenAddress(e.target.value)}
                  placeholder="Enter token address..."
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-quantum-purple/50 transition-colors"
                />
              </div>

              {/* Add Liquidity Button */}
              <button
                onClick={handleAddCustomTokenLiquidity}
                disabled={!customTokenAddress || customTokenAddress.length < 10}
                className="w-full py-4 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-purple/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                <Droplet className="w-5 h-5" />
                Add Liquidity for Custom Token
              </button>

              {/* Info */}
              <div className="p-3 bg-quantum-purple/10 border border-quantum-purple/20 rounded-xl">
                <p className="text-xs text-gray-400">
                  <strong className="text-quantum-purple">Note:</strong> You can pair your custom token with native QUG or QUGUSD stablecoin to create a liquidity pool.
                </p>
              </div>
            </div>
          </div>

          {/* My Liquidity Pools Section */}
          {liquidityPools.length > 0 && (
            <div className="relative group mt-6">
              {/* Glow effect */}
              <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-cyan to-quantum-green rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity" />

              <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-quantum-cyan/20 p-6 space-y-4">
                <div className="flex items-center gap-3 mb-4">
                  <Droplet className="w-6 h-6 text-quantum-cyan" />
                  <h2 className="text-xl font-bold text-white">My Liquidity Pools</h2>
                </div>

                {/* Pools List */}
                <div className="space-y-3">
                  {liquidityPools.map((pool, index) => (
                    <motion.div
                      key={pool.pool_id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-white/5 border border-white/10 rounded-xl p-4 hover:border-quantum-cyan/50 transition-all"
                    >
                      {/* Pool Header */}
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <div className="text-2xl">💧</div>
                          <div>
                            <div className="font-bold text-white">
                              {pool.token0} / {pool.token1}
                            </div>
                            <div className="text-xs text-gray-400">Pool ID: {pool.pool_id.slice(0, 20)}...</div>
                          </div>
                        </div>
                      </div>

                      {/* Pool Stats */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-quantum-cyan/10 border border-quantum-cyan/20 rounded-lg p-3">
                          <div className="text-xs text-gray-400 mb-1">Reserve {pool.token0}</div>
                          <div className="text-sm font-bold text-white">
                            {(pool.reserve0 / 100_000_000).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 8 })}
                          </div>
                        </div>
                        <div className="bg-quantum-purple/10 border border-quantum-purple/20 rounded-lg p-3">
                          <div className="text-xs text-gray-400 mb-1">Reserve {pool.token1}</div>
                          <div className="text-sm font-bold text-white">
                            {(pool.reserve1 / 100_000_000).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 8 })}
                          </div>
                        </div>
                        <div className="bg-quantum-green/10 border border-quantum-green/20 rounded-lg p-3">
                          <div className="text-xs text-gray-400 mb-1">Your Share</div>
                          <div className="text-sm font-bold text-quantum-green">100%</div>
                        </div>
                        <div className="bg-quantum-pink/10 border border-quantum-pink/20 rounded-lg p-3">
                          <div className="text-xs text-gray-400 mb-1">Created</div>
                          <div className="text-sm font-bold text-white">
                            {new Date(pool.created_at * 1000).toLocaleDateString()}
                          </div>
                        </div>
                      </div>

                      {/* Pool Actions */}
                      <div className="flex gap-2 mt-3">
                        <button
                          onClick={async () => {
                            console.log('🔍 Add More clicked for pool:', pool);
                            console.log('🔍 Looking for token:', pool.token0, 'length:', pool.token0.length);
                            console.log('🔍 Available tokens:', tokens.map(t => ({ id: t.id, symbol: t.symbol })));

                            // Find the token object for this pool's token0
                            let token0Obj = tokens.find(t => t.symbol === pool.token0 || t.id === pool.token0);
                            console.log('🔍 Found token in list?', token0Obj ? 'YES' : 'NO');

                            // If not found, try to fetch it or search by different methods
                            if (!token0Obj) {
                              // Case 1: token0 looks like a contract address (long string)
                              if (pool.token0.length > 20) {
                                console.log('🔍 Token not in list, fetching from contract:', pool.token0);
                                try {
                                  const contractInfo = await qnkAPI.getContractInfo(pool.token0);
                                  console.log('📊 Contract info response:', contractInfo);
                                  if (contractInfo.success && contractInfo.data) {
                                    const contract = contractInfo.data;
                                    const walletAddress = localStorage.getItem('walletAddress') || '';
                                    let tokenBalance = 0;

                                    if (walletAddress) {
                                      const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, pool.token0);
                                      if (balanceResponse.success && balanceResponse.data) {
                                        tokenBalance = balanceResponse.data.balance || 0;
                                      }
                                    }

                                    const decimals = contract.decimals || 18;
                                    const rawSupply = contract.total_supply || 0;
                                    const actualSupply = Number(rawSupply) / Math.pow(10, decimals);

                                    token0Obj = {
                                      id: pool.token0,
                                      symbol: contract.token_symbol || contract.symbol || 'CUSTOM',
                                      name: contract.token_name || contract.name || 'Custom Token',
                                      balance: tokenBalance,
                                      price: 1.0,
                                      change1h: 0,
                                      change24h: 0,
                                      change7d: 0,
                                      volume24h: 0,
                                      liquidity: actualSupply,
                                      marketCap: 0,
                                      totalSupply: actualSupply,
                                      circulatingSupply: actualSupply,
                                      holders: 0,
                                      icon: '🪙',
                                      features: {
                                        reflection: false,
                                        autoLiquidity: false,
                                        buybackAndBurn: false,
                                        antiWhale: false,
                                        quantumSecured: true,
                                      },
                                      fees: {
                                        buy: 0,
                                        sell: 0,
                                        transfer: 0,
                                      },
                                      description: `${contract.token_name || 'Custom token'} deployed on Quillon blockchain`,
                                    };
                                    console.log('✅ Fetched token from contract:', token0Obj);
                                  }
                                } catch (error) {
                                  console.error('Failed to fetch token contract info:', error);
                                }
                              } else {
                                // Case 2: token0 is a symbol (short string like "TEST5")
                                // Try to find the contract address from user's deployed contracts
                                console.log('🔍 token0 appears to be a symbol:', pool.token0);
                                console.log('🔍 Searching for contract address via user contracts...');

                                try {
                                  const walletAddress = localStorage.getItem('walletAddress') || '';
                                  if (!walletAddress) {
                                    console.error('❌ No wallet address found');
                                    throw new Error('No wallet address');
                                  }

                                  const response = await qnkAPI.getUserContracts(walletAddress);
                                  console.log('📊 User contracts response:', response);

                                  if (response.success && response.data) {
                                    const foundToken = response.data.find(t => t.symbol === pool.token0);
                                    if (foundToken) {
                                      console.log('✅ Found contract address:', foundToken.address);
                                      // Now fetch with the contract address
                                      const contractInfo = await qnkAPI.getContractInfo(foundToken.address);
                                      if (contractInfo.success && contractInfo.data) {
                                        const contract = contractInfo.data;
                                        const walletAddress = localStorage.getItem('walletAddress') || '';
                                        let tokenBalance = 0;

                                        if (walletAddress) {
                                          const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, foundToken.address);
                                          if (balanceResponse.success && balanceResponse.data) {
                                            tokenBalance = balanceResponse.data.balance || 0;
                                          }
                                        }

                                        const decimals = contract.decimals || 18;
                                        const rawSupply = contract.total_supply || 0;
                                        const actualSupply = Number(rawSupply) / Math.pow(10, decimals);

                                        token0Obj = {
                                          id: foundToken.address,
                                          symbol: contract.token_symbol || contract.symbol || pool.token0,
                                          name: contract.token_name || contract.name || pool.token0,
                                          balance: tokenBalance,
                                          price: 1.0,
                                          change1h: 0,
                                          change24h: 0,
                                          change7d: 0,
                                          volume24h: 0,
                                          liquidity: actualSupply,
                                          marketCap: 0,
                                          totalSupply: actualSupply,
                                          circulatingSupply: actualSupply,
                                          holders: 0,
                                          icon: '🪙',
                                          features: {
                                            reflection: false,
                                            autoLiquidity: false,
                                            buybackAndBurn: false,
                                            antiWhale: false,
                                            quantumSecured: true,
                                          },
                                          fees: {
                                            buy: 0,
                                            sell: 0,
                                            transfer: 0,
                                          },
                                          description: `${contract.token_name || 'Custom token'} deployed on Quillon blockchain`,
                                        };
                                        console.log('✅ Fetched token via symbol lookup:', token0Obj);
                                      }
                                    } else {
                                      console.warn('⚠️ Symbol not found in supported tokens');
                                    }
                                  }
                                } catch (error) {
                                  console.error('Failed to lookup token by symbol:', error);
                                }
                              }
                            }

                            if (token0Obj) {
                              console.log('✅ Opening liquidity modal with token:', token0Obj);
                              setLiquidityToken(token0Obj);
                            } else {
                              console.error('❌ Token not found after all lookup attempts');
                              alert(`❌ Token ${pool.token0} not found.\n\nPlease check:\n1. Token contract is deployed\n2. Token is registered in the system\n3. Check browser console for details`);
                            }
                          }}
                          className="flex-1 py-2 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-lg text-white text-sm font-medium hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
                        >
                          Add More
                        </button>
                        <button
                          onClick={() => {
                            setRemovingPool(pool);
                            setRemovePercentage(50);
                          }}
                          className="flex-1 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white text-sm font-medium transition-all"
                        >
                          Remove
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>

                {/* Info Box */}
                <div className="p-3 bg-quantum-cyan/10 border border-quantum-cyan/20 rounded-xl">
                  <p className="text-xs text-gray-400">
                    <strong className="text-quantum-cyan">Total Pools:</strong> {liquidityPools.length} active liquidity pools
                  </p>
                </div>
              </div>
            </div>
          )}
        </motion.div>

        {/* Token Table */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-3"
        >
          <div className="relative group">
            {/* Glow effect */}
            <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity" />

            <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-quantum-purple/20 p-6">
              <h2 className="text-xl font-bold text-white mb-6">Available Tokens</h2>

              {/* Search and Filters */}
              <div className="flex flex-col sm:flex-row gap-4 mb-6">
                {/* Search */}
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search by name, symbol, or address (qnk...)..."
                    className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white focus:outline-none focus:border-quantum-cyan/50 transition-colors"
                  />
                </div>

                {/* Filter Buttons */}
                <div className="flex gap-2">
                  {(['all', 'gainers', 'losers'] as const).map((filter) => (
                    <button
                      key={filter}
                      onClick={() => setFilterBy(filter)}
                      className={`px-4 py-3 rounded-xl font-medium transition-all ${
                        filterBy === filter
                          ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                          : 'bg-white/5 text-gray-400 hover:bg-white/10'
                      }`}
                    >
                      {filter.charAt(0).toUpperCase() + filter.slice(1)}
                    </button>
                  ))}
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left py-3 px-4 text-gray-400 font-medium text-sm">Token</th>
                      <th
                        onClick={() => handleSort('price')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        Price {sortBy === 'price' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        onClick={() => handleSort('change24h')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        1h % {sortBy === 'change24h' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        onClick={() => handleSort('change24h')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        24h % {sortBy === 'change24h' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        7d %
                      </th>
                      <th
                        onClick={() => handleSort('volume24h')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        Volume {sortBy === 'volume24h' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        onClick={() => handleSort('marketCap')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        Market Cap {sortBy === 'marketCap' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        onClick={() => handleSort('liquidity')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        Liquidity {sortBy === 'liquidity' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm"
                      >
                        Makers
                      </th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium text-sm">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTokens.map((token, index) => (
                      <motion.tr
                        key={token.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className={`border-b border-white/5 hover:bg-white/5 transition-colors cursor-pointer relative ${
                          nitroBoostTokens.has(token.id) ? 'nitro-boost-active' : ''
                        }`}
                        onClick={() => setSelectedToken(token)}
                      >
                        {/* Nitro boost effects */}
                        {nitroBoostTokens.has(token.id) && (
                          <>
                            <div className="hyperspeed-lines" />
                            <motion.div
                              className="absolute inset-0 pointer-events-none"
                              initial={{ opacity: 0 }}
                              animate={{ opacity: [0, 1, 0] }}
                              transition={{ duration: 2, ease: "easeInOut" }}
                            >
                              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent" />
                            </motion.div>
                          </>
                        )}
                        {/* Token Info */}
                        <td className="py-4 px-4">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-gradient-to-br from-quantum-cyan to-quantum-purple rounded-full flex items-center justify-center text-xl">
                              {(token.icon === 'qug-logo' || token.icon === 'qugusd-logo' || token.icon === 'usd-logo') ? (
                                <div className="relative w-7 h-7">
                                  <div className="absolute inset-0 rounded-full" style={{
                                    background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                                    padding: '1px'
                                  }}>
                                    <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center p-0.5">
                                      <img
                                        src="/quillon-logo.png"
                                        alt="Quillon"
                                        className="w-full h-full object-contain"
                                        style={{ filter: 'invert(1)' }}
                                      />
                                    </div>
                                  </div>
                                </div>
                              ) : (
                                token.icon
                              )}
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <div className="font-bold text-white">{token.symbol}</div>
                                {boostedTokens.has(token.id) && (
                                  <motion.div
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-gradient-to-r from-orange-500 to-yellow-500 text-xs font-bold text-white"
                                  >
                                    <Zap className="w-3 h-3" />
                                    NITRO {boostedTokens.get(token.id)}
                                  </motion.div>
                                )}
                              </div>
                              <div className="text-sm text-gray-400">{token.name}</div>
                            </div>
                          </div>
                        </td>

                        {/* Price */}
                        <td className="py-4 px-4 text-right text-white font-medium">
                          ${token.price.toLocaleString()}
                        </td>

                        {/* 1h Change - REAL DATA */}
                        <td className="py-4 px-4 text-right">
                          <div className={`flex items-center justify-end gap-1 ${
                            token.change1h > 0 ? 'text-quantum-green' : 'text-red-500'
                          }`}>
                            <span className="font-medium">
                              {token.change1h > 0 ? '+' : ''}{token.change1h.toFixed(2)}%
                            </span>
                          </div>
                        </td>

                        {/* 24h Change */}
                        <td className="py-4 px-4 text-right">
                          <div className={`flex items-center justify-end gap-1 ${
                            token.change24h > 0 ? 'text-quantum-green' : 'text-red-500'
                          }`}>
                            {token.change24h > 0 ? (
                              <TrendingUp className="w-4 h-4" />
                            ) : (
                              <TrendingDown className="w-4 h-4" />
                            )}
                            <span className="font-medium">
                              {token.change24h > 0 ? '+' : ''}{token.change24h.toFixed(2)}%
                            </span>
                          </div>
                        </td>

                        {/* 7d Change - REAL DATA */}
                        <td className="py-4 px-4 text-right">
                          <div className={`flex items-center justify-end gap-1 ${
                            token.change7d > 0 ? 'text-quantum-green' : 'text-red-500'
                          }`}>
                            <span className="font-medium">
                              {token.change7d > 0 ? '+' : ''}{token.change7d.toFixed(2)}%
                            </span>
                          </div>
                        </td>

                        {/* Volume */}
                        <td className="py-4 px-4 text-right text-white font-medium">
                          {formatNumber(token.volume24h)}
                        </td>

                        {/* Market Cap */}
                        <td className="py-4 px-4 text-right text-white font-medium">
                          ${formatNumber(token.marketCap)}
                        </td>

                        {/* Liquidity */}
                        <td className="py-4 px-4 text-right text-white font-medium">
                          ${formatNumber(token.liquidity)}
                        </td>

                        {/* Makers */}
                        <td className="py-4 px-4 text-right text-gray-400 font-medium">
                          {token.holders?.toLocaleString() || '0'}
                        </td>

                        {/* Actions */}
                        <td className="py-4 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center justify-end gap-2">
                            {/* Special Mint button for QUGUSD - deposits QUG as collateral to mint QUGUSD */}
                            {token.symbol === 'QUGUSD' && (
                              <motion.button
                                onClick={() => setIsMintQUGUSDModalOpen(true)}
                                className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg text-white font-medium hover:shadow-lg hover:shadow-green-500/50 transition-all flex items-center gap-2"
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                              >
                                <span className="text-lg">💵</span>
                                Mint USD
                              </motion.button>
                            )}
                            <motion.button
                              onClick={() => {
                                setSwapFrom(token.symbol);
                                window.scrollTo({ top: 0, behavior: 'smooth' });
                              }}
                              className="px-4 py-2 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-lg text-white font-medium hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                            >
                              Trade
                            </motion.button>
                            <motion.button
                              onClick={() => setLiquidityToken(token)}
                              className="px-4 py-2 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-lg text-white font-medium hover:shadow-lg hover:shadow-quantum-purple/50 transition-all flex items-center gap-2"
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                            >
                              <Droplet className="w-4 h-4" />
                              Liquidity
                            </motion.button>
                            <motion.button
                              onClick={() => handleNitroBoost(token)}
                              className="px-4 py-2 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg text-white font-medium hover:shadow-lg hover:shadow-orange-500/50 transition-all flex items-center gap-2 turbo-button relative overflow-hidden"
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              disabled={nitroBoostTokens.has(token.id)}
                            >
                              <Zap className="w-4 h-4" />
                              Nitro
                              {nitroBoostTokens.has(token.id) && (
                                <motion.div
                                  className="absolute inset-0 bg-gradient-to-r from-yellow-400 to-orange-600"
                                  initial={{ x: '-100%' }}
                                  animate={{ x: '200%' }}
                                  transition={{ duration: 0.6, repeat: 3 }}
                                />
                              )}
                            </motion.button>
                          </div>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>

                {filteredTokens.length === 0 && (
                  <div className="text-center py-12">
                    <Info className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                    <p className="text-gray-400">No tokens found matching your criteria</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      </div>
      )}
      </div>

      {/* Mint QUGUSD Modal */}
      <MintQUGUSDModal
        isOpen={isMintQUGUSDModalOpen}
        onClose={() => setIsMintQUGUSDModalOpen(false)}
        userQUGBalance={tokens.find(t => t.symbol === 'QUG')?.balance || 0}
        onSuccess={() => {
          // Don't reload - let the CDP event system handle updates
          console.log('✅ CDP mint success callback - no reload needed');
        }}
      />

      {/* Nitro Success Modal */}
      <NitroSuccessModal
        isOpen={showSuccessModal}
        onClose={() => setShowSuccessModal(false)}
        type="activation"
        data={successModalData}
      />

      {/* Swap Success Modal */}
      {swapSuccessData && (
        <SwapSuccessModal
          isOpen={showSwapSuccess}
          onClose={() => {
            setShowSwapSuccess(false);
            setSwapSuccessData(null);
          }}
          fromToken={swapSuccessData.fromToken}
          toToken={swapSuccessData.toToken}
          fromAmount={swapSuccessData.fromAmount}
          toAmount={swapSuccessData.toAmount}
          transactionHash={swapSuccessData.transactionHash}
        />
      )}
    </>
  );
}
