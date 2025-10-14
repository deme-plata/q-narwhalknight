import { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowDownUp, Search, TrendingUp, TrendingDown, Settings, Info, Droplet, Zap } from 'lucide-react';
import TokenDetailsModal from './TokenDetailsModal';
import LiquidityModal from './LiquidityModal';
import TokenSelectorModal from './TokenSelectorModal';
import { qnkAPI } from '../services/api';

interface Token {
  id: string;
  symbol: string;
  name: string;
  balance: number;
  price: number;
  change24h: number;
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
  const [sortBy, setSortBy] = useState<'symbol' | 'price' | 'change24h' | 'volume24h' | 'liquidity'>('volume24h');
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

  // Load Nitro points from localStorage and boosted tokens from backend with SSE real-time updates
  useEffect(() => {
    let mounted = true;
    let eventSource: EventSource | null = null;

    // Load Nitro points from localStorage (user's balance)
    const storedPoints = localStorage.getItem('nitroPoints');
    if (storedPoints) {
      setNitroPoints(parseInt(storedPoints, 10));
    }

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
          const data = JSON.parse(event.data);
          console.log('🚀 Received Nitro boost event:', data);

          // Update boosted tokens map
          setBoostedTokens(prev => {
            const newMap = new Map(prev);
            const tokenId = data.token_id;
            const pointsAdded = data.points;
            const existingPoints = newMap.get(tokenId) || 0;
            newMap.set(tokenId, existingPoints + pointsAdded);
            console.log(`Updated ${tokenId}: ${existingPoints} -> ${existingPoints + pointsAdded} points`);
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
          const data = JSON.parse(event.data);
          console.log('📊 Received full Nitro boosts update:', data);

          // Full update of all boosts
          if (data.boosts) {
            const boostMap = new Map(Object.entries(data.boosts) as [string, number][]);
            setBoostedTokens(boostMap);
          }
        } catch (err) {
          console.error('Failed to parse Nitro boosts update event:', err);
        }
      });

      // Listen for token price updates
      eventSource.addEventListener('token_price_update', (event) => {
        if (!mounted) return;
        try {
          const data = JSON.parse(event.data);
          console.log('📈 Received token price update:', data);

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
          const data = JSON.parse(event.data);
          console.log('📜 Received token transaction:', data);

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
      if (eventSource) {
        console.log('🔌 Closing SSE connection for Nitro boosts');
        eventSource.close();
      }
    };
  }, []);

  // Fetch real tokens from API
  useEffect(() => {
    const fetchTokens = async () => {
      try {
        // Get wallet address for balance fetching
        const walletAddress = localStorage.getItem('walletAddress') || '';

        // Fetch native QUG balance
        let nativeQugBalance = 0;
        if (walletAddress) {
          try {
            const balanceResponse = await qnkAPI.getWalletBalance(walletAddress);
            if (balanceResponse.success && balanceResponse.data) {
              nativeQugBalance = balanceResponse.data.balance_qnk || 0;
            }
          } catch (error) {
            console.error('Failed to fetch native QUG balance:', error);
          }
        }

        // Fetch real price from oracle API
        let qugPrice = 42.50;
        let qugChange = 12.8;
        let qugVolume = 1850000;
        try {
          const oracleResponse = await qnkAPI.getOraclePrice('QUG/USD');
          if (oracleResponse.success && oracleResponse.data) {
            qugPrice = oracleResponse.data.price;
            qugChange = oracleResponse.data.change_24h;
            qugVolume = oracleResponse.data.volume_24h;
            console.log('✅ Fetched QUG price from oracle:', qugPrice);
          }
        } catch (error) {
          console.error('Failed to fetch QUG price from oracle, using default:', error);
        }

        // Fetch QUGUSD price from oracle
        let qugusdPrice = 1.00;
        let qugusdChange = 0.02;
        let qugusdVolume = 950000;
        try {
          const oracleResponse = await qnkAPI.getOraclePrice('QUGUSD/USD');
          if (oracleResponse.success && oracleResponse.data) {
            qugusdPrice = oracleResponse.data.price;
            qugusdChange = oracleResponse.data.change_24h;
            qugusdVolume = oracleResponse.data.volume_24h;
            console.log('✅ Fetched QUGUSD price from oracle:', qugusdPrice);
          }
        } catch (error) {
          console.error('Failed to fetch QUGUSD price from oracle, using default:', error);
        }

        // Add native QUG and QUGUSD stablecoin
        const nativeTokens: Token[] = [
          {
            id: 'native-qug',
            symbol: 'QUG',
            name: 'Quillon',
            balance: nativeQugBalance,
            price: qugPrice,
            change24h: qugChange,
            volume24h: qugVolume,
            liquidity: 8500000,
            marketCap: 625000000,
            totalSupply: 21000000,
            circulatingSupply: 14700000,
            holders: 18432,
            icon: '💎',
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
            balance: 0,
            price: qugusdPrice,
            change24h: qugusdChange,
            volume24h: qugusdVolume,
            liquidity: 12000000,
            marketCap: 125000000,
            totalSupply: 125000000,
            circulatingSupply: 125000000,
            holders: 5600,
            icon: '💵',
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
        ];

        const response = await qnkAPI.getSupportedTokens();
        let enrichedTokens = nativeTokens;

        if (response.success && response.data) {
          // Get wallet address for balance checks
          const walletAddress = localStorage.getItem('walletAddress') || '';

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
                  const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, apiToken.address);
                  if (balanceResponse.success && balanceResponse.data) {
                    tokenBalance = balanceResponse.data.balance || 0;
                  }
                } catch (error) {
                  console.error(`Failed to fetch balance for ${apiToken.symbol}:`, error);
                }
              }

              // Fetch custom token price from oracle (if available)
              let customPrice = 1.0; // Default price
              let customChange = 0.0;
              let customVolume = 0.0;
              try {
                const oracleResponse = await qnkAPI.getOraclePrice(apiToken.address);
                if (oracleResponse.success && oracleResponse.data) {
                  customPrice = oracleResponse.data.price;
                  customChange = oracleResponse.data.change_24h || 0;
                  customVolume = oracleResponse.data.volume_24h || 0;
                  console.log(`✅ Fetched ${apiToken.symbol} price from oracle:`, customPrice);
                }
              } catch (error) {
                console.log(`ℹ️ No oracle price for ${apiToken.symbol}, using default`);
              }

              return {
                id: apiToken.address,
                symbol: apiToken.symbol,
                name: apiToken.name,
                balance: tokenBalance, // Balance is already in human-readable form from backend
                price: customPrice,
                change24h: customChange,
                volume24h: customVolume,
                liquidity: actualSupply || 3000000,
                marketCap: 225000000,
                totalSupply: actualSupply || 10000000,
                circulatingSupply: actualSupply || 10000000,
                holders: 8250,
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
          enrichedTokens = [...nativeTokens, ...apiTokens];
        }
        setTokens(enrichedTokens);
      } catch (error) {
        console.error('Failed to fetch tokens:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTokens();
  }, []);

  // Fetch liquidity pools
  useEffect(() => {
    const fetchPools = async () => {
      try {
        const response = await qnkAPI.getLiquidityPools();
        if (response.success && response.data) {
          setLiquidityPools(response.data);
        }
      } catch (error) {
        console.error('Failed to fetch liquidity pools:', error);
      }
    };

    fetchPools();
    // Refresh every 10 seconds
    const interval = setInterval(fetchPools, 10000);
    return () => clearInterval(interval);
  }, []);

  // Filter tokens based on search and filter
  const filteredTokens = tokens
    .filter(token => {
      const matchesSearch = token.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           token.name.toLowerCase().includes(searchQuery.toLowerCase());

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

      // Token balances are already in human-readable form (not smallest units)
      // So we just use the amounts directly without conversion
      const amount0 = Math.floor(amountA);
      const amount1 = Math.floor(amountB);

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
        change24h: 0,
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

      // Deduct points from local state
      const newPoints = nitroPoints - boostCost;
      setNitroPoints(newPoints);
      localStorage.setItem('nitroPoints', newPoints.toString());

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
      alert(`🚀 Nitro Boost activated!\n\nSpent ${boostCost} points on ${nitroBoostToken.symbol}\n\nRemaining Points: ${newPoints}\nTotal Boost on ${nitroBoostToken.symbol}: ${existingPoints + boostCost} points`);

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
        currentToken={tokens.find(t => t.symbol === swapFrom)}
      />

      {/* Token Selector Modal - To Token */}
      <TokenSelectorModal
        isOpen={isToTokenSelectorOpen}
        onClose={() => setIsToTokenSelectorOpen(false)}
        onSelectToken={handleSelectToToken}
        tokens={tokens}
        boostedTokens={boostedTokens}
        currentToken={tokens.find(t => t.symbol === swapTo)}
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
                        <div className="text-white text-sm">{removingPool.reserve0.toLocaleString()} {removingPool.token0}</div>
                        <div className="text-white text-sm">{removingPool.reserve1.toLocaleString()} {removingPool.token1}</div>
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
                        <span className="text-white font-bold">{Math.floor((removingPool.reserve0 * removePercentage) / 100).toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">{removingPool.token1}:</span>
                        <span className="text-white font-bold">{Math.floor((removingPool.reserve1 * removePercentage) / 100).toLocaleString()}</span>
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
                      <div className="text-3xl">{nitroBoostToken.icon}</div>
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-4 pr-32 text-white text-xl focus:outline-none focus:border-quantum-cyan/50 transition-colors"
                  />
                  <button
                    onClick={() => setIsFromTokenSelectorOpen(true)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/30 rounded-lg px-3 py-2 text-white font-bold cursor-pointer focus:outline-none transition-all flex items-center gap-2"
                  >
                    <span className="text-xl">{tokens.find(t => t.symbol === swapFrom)?.icon || '💎'}</span>
                    <span>{swapFrom}</span>
                    <span className="text-xs opacity-70">▼</span>
                  </button>
                </div>
                <div className="text-xs text-gray-500">
                  Balance: {tokens.find(t => t.symbol === swapFrom)?.balance.toFixed(4) || '0.0000'}
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
                <label className="text-sm text-gray-400">To</label>
                <div className="relative">
                  <input
                    type="text"
                    value={swapAmount ? (parseFloat(swapAmount) * 0.95).toFixed(4) : ''}
                    placeholder="0.0"
                    readOnly
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-4 pr-32 text-white text-xl focus:outline-none"
                  />
                  <button
                    onClick={() => setIsToTokenSelectorOpen(true)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/30 rounded-lg px-3 py-2 text-white font-bold cursor-pointer focus:outline-none transition-all flex items-center gap-2"
                  >
                    <span className="text-xl">{tokens.find(t => t.symbol === swapTo)?.icon || '💵'}</span>
                    <span>{swapTo}</span>
                    <span className="text-xs opacity-70">▼</span>
                  </button>
                </div>
              </div>

              {/* Swap Info */}
              <div className="space-y-2 text-sm p-4 bg-white/5 rounded-xl">
                <div className="flex justify-between text-gray-400">
                  <span>Rate</span>
                  <span className="text-white">1 {swapFrom} ≈ 0.95 {swapTo}</span>
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

                  const fromToken = tokens.find(t => t.symbol === swapFrom);
                  const toToken = tokens.find(t => t.symbol === swapTo);

                  if (!fromToken || !toToken) {
                    alert('Invalid token selection');
                    return;
                  }

                  if (fromToken.balance < parseFloat(swapAmount)) {
                    alert(`Insufficient ${swapFrom} balance. You have ${fromToken.balance.toFixed(4)}`);
                    return;
                  }

                  try {
                    const expectedOutput = parseFloat(swapAmount) * (toToken.price / fromToken.price);
                    const minOutput = expectedOutput * 0.995;

                    const response = await qnkAPI.executeSwap({
                      from_token: fromToken.id === 'native-qug' ? 'QUG' : fromToken.id,
                      to_token: toToken.id === 'qugusd-stable' ? 'QUGUSD' : toToken.id,
                      amount_in: Math.floor(parseFloat(swapAmount) * 1_000_000_000),
                      min_amount_out: Math.floor(minOutput * 1_000_000_000),
                      wallet_address: walletAddress
                    });

                    if (response.success && response.data) {
                      alert(`✅ Swap successful!\n\nSwapped: ${swapAmount} ${swapFrom}\nReceived: ${(response.data.amount_out / 1_000_000_000).toFixed(4)} ${swapTo}\n\nTransaction: ${response.data.transaction_id}`);
                      window.location.reload();
                    } else {
                      alert(`❌ Swap failed: ${response.error || 'Unknown error'}`);
                    }
                  } catch (error) {
                    console.error('Swap failed:', error);
                    alert('❌ Swap failed. Please try again.');
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
                            {pool.reserve0.toLocaleString()}
                          </div>
                        </div>
                        <div className="bg-quantum-purple/10 border border-quantum-purple/20 rounded-lg p-3">
                          <div className="text-xs text-gray-400 mb-1">Reserve {pool.token1}</div>
                          <div className="text-sm font-bold text-white">
                            {pool.reserve1.toLocaleString()}
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
                          onClick={() => {
                            // Find the token object for this pool's token0
                            const token0Obj = tokens.find(t => t.symbol === pool.token0 || t.id === pool.token0);
                            if (token0Obj) {
                              setLiquidityToken(token0Obj);
                            } else {
                              alert(`Token ${pool.token0} not found in available tokens`);
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
          className="lg:col-span-2"
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
                    placeholder="Search tokens..."
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
                        24h Change {sortBy === 'change24h' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        onClick={() => handleSort('volume24h')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        Volume {sortBy === 'volume24h' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th
                        onClick={() => handleSort('liquidity')}
                        className="text-right py-3 px-4 text-gray-400 font-medium text-sm cursor-pointer hover:text-white transition-colors"
                      >
                        Liquidity {sortBy === 'liquidity' && (sortDirection === 'asc' ? '↑' : '↓')}
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
                              {token.icon}
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

                        {/* Volume */}
                        <td className="py-4 px-4 text-right text-white font-medium">
                          {formatNumber(token.volume24h)}
                        </td>

                        {/* Liquidity */}
                        <td className="py-4 px-4 text-right text-white font-medium">
                          {formatNumber(token.liquidity)}
                        </td>

                        {/* Actions */}
                        <td className="py-4 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center justify-end gap-2">
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
    </>
  );
}
