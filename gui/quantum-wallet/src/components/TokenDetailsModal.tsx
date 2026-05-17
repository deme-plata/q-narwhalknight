import { motion } from 'framer-motion';
import { X, TrendingUp, TrendingDown, ExternalLink, Info, Droplet, Zap, Shield, Coins, Users, Activity, ArrowUpDown, ArrowUp, ArrowDown, Filter, Twitter, MessageCircle, Globe, Github, FileText, Lock, Unlock, Clock, Gift, ChevronDown, AlertCircle, CheckCircle } from 'lucide-react';
import { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import TokenIcon from './TokenIcon';
import { qnkAPI, getQCreditStatus, getQCreditPosition, getQCreditTiers, lockQCredit, unlockQCredit, claimQCreditYield } from '../services/api';
import type { QCreditStatus, QCreditPosition, QCreditPositionResponse, QCreditTier } from '../services/api';

// v3.1.1: Helper to safely parse u128 values that may come as strings from the API
// v3.6.14: Also handles base unit conversion - if value is absurdly large, divide by 1e8
const parseU128 = (value: string | number | undefined): number => {
  if (value === undefined || value === null) return 0;
  let parsed: number;
  if (typeof value === 'number') {
    parsed = value;
  } else if (typeof value === 'string') {
    parsed = parseFloat(value);
    if (isNaN(parsed)) return 0;
  } else {
    return 0;
  }
  // v3.6.14: If value is absurdly large (>1e15), it's likely in base units (8 decimals)
  // Convert to human-readable format
  if (parsed > 1e15) {
    return parsed / 1e8;
  }
  return parsed;
};

// v2.7.7-beta: Social links interface for token metadata
interface SocialLinks {
  twitter?: string;       // Twitter/X handle or URL
  discord?: string;       // Discord server invite link
  telegram?: string;      // Telegram group/channel link
  website?: string;       // Official website URL
  github?: string;        // GitHub repository URL
  medium?: string;        // Medium blog URL
  reddit?: string;        // Reddit community URL
  coinmarketcap?: string; // CoinMarketCap listing URL
  coingecko?: string;     // CoinGecko listing URL
}

interface TokenDetails {
  id: string;
  symbol: string;
  name: string;
  icon: string;
  price: number;
  change24h: number;
  marketCap: number;
  fullyDilutedMarketCap?: number;  // FDV = totalSupply * price
  totalSupply: number;
  circulatingSupply: number;
  volume24h: number;
  liquidity: number;
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
  // v2.7.7-beta: Social media links
  socialLinks?: SocialLinks;
}

interface TokenDetailsModalProps {
  token: TokenDetails | null;
  onClose: () => void;
}

interface PriceDataPoint {
  timestamp: number;
  price: number;
  volume: number;
}

interface TokenTransaction {
  id: string;
  timestamp: number;
  type: 'buy' | 'sell' | 'transfer';
  amount: number;
  price: number;
  value: number;
  from: string;
  to: string;
  txHash: string;
}

type SortField = 'timestamp' | 'amount' | 'value' | 'type';
type SortOrder = 'asc' | 'desc';
type TransactionFilter = 'all' | 'buy' | 'sell' | 'transfer';

export default function TokenDetailsModal({ token, onClose }: TokenDetailsModalProps) {
  const [timeframe, setTimeframe] = useState<'1H' | '24H' | '7D' | '30D' | '1Y'>('24H');
  const [priceData, setPriceData] = useState<PriceDataPoint[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<PriceDataPoint | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  // Transaction table state
  const [transactions, setTransactions] = useState<TokenTransaction[]>([]);
  const [sortField, setSortField] = useState<SortField>('timestamp');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [filterType, setFilterType] = useState<TransactionFilter>('all');

  // v8.5.5: QCredit Yield Vault state
  const isQCredit = token?.symbol?.toUpperCase() === 'QCREDIT';
  const [qcreditStatus, setQcreditStatus] = useState<QCreditStatus | null>(null);
  const [qcreditPositions, setQcreditPositions] = useState<QCreditPosition[]>([]);
  const [qcreditTotalLocked, setQcreditTotalLocked] = useState('0');
  const [qcreditTotalPending, setQcreditTotalPending] = useState('0');
  const [qcreditLoading, setQcreditLoading] = useState(false);
  const [lockAmount, setLockAmount] = useState('');
  const [selectedTier, setSelectedTier] = useState('bronze');
  const [tierDropdownOpen, setTierDropdownOpen] = useState(false);
  const [lockLoading, setLockLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState<number | null>(null);
  const [qcreditError, setQcreditError] = useState<string | null>(null);
  const [qcreditSuccess, setQcreditSuccess] = useState<string | null>(null);

  const getAuthHeaders = useCallback(() => {
    const wallet = localStorage.getItem('walletAddress') || '';
    return { 'Authorization': `Bearer ${wallet}`, 'X-Wallet-Auth': wallet };
  }, []);

  const walletAddress = typeof window !== 'undefined' ? localStorage.getItem('walletAddress') || '' : '';

  // Fetch QCredit vault data
  useEffect(() => {
    if (!isQCredit || !token) return;
    let mounted = true;

    const fetchQCreditData = async () => {
      setQcreditLoading(true);
      try {
        const [status, positionData] = await Promise.allSettled([
          getQCreditStatus(),
          walletAddress ? getQCreditPosition(getAuthHeaders()) : Promise.resolve(null),
        ]);

        if (!mounted) return;

        if (status.status === 'fulfilled' && status.value) {
          setQcreditStatus(status.value);
        }

        if (positionData.status === 'fulfilled' && positionData.value) {
          const pd = positionData.value as QCreditPositionResponse;
          setQcreditPositions(pd.positions);
          setQcreditTotalLocked(pd.total_locked);
          setQcreditTotalPending(pd.total_pending_yield);
        }
      } catch (err) {
        console.error('Failed to fetch QCredit data:', err);
      } finally {
        if (mounted) setQcreditLoading(false);
      }
    };

    fetchQCreditData();
    const interval = setInterval(fetchQCreditData, 30000);
    return () => { mounted = false; clearInterval(interval); };
  }, [isQCredit, token, walletAddress]);

  // QCredit actions
  const handleLockQug = async () => {
    if (!lockAmount || parseFloat(lockAmount) <= 0) {
      setQcreditError('Enter a valid amount');
      return;
    }
    setLockLoading(true);
    setQcreditError(null);
    setQcreditSuccess(null);
    try {
      await lockQCredit(walletAddress, lockAmount, selectedTier, getAuthHeaders());
      setQcreditSuccess(`Locked ${lockAmount} QUG in ${selectedTier} tier`);
      setLockAmount('');
      // Refresh data
      const [status, posData] = await Promise.all([
        getQCreditStatus(),
        getQCreditPosition(getAuthHeaders()),
      ]);
      setQcreditStatus(status);
      setQcreditPositions(posData.positions);
      setQcreditTotalLocked(posData.total_locked);
      setQcreditTotalPending(posData.total_pending_yield);
    } catch (err: any) {
      setQcreditError(err.message || 'Lock failed');
    } finally {
      setLockLoading(false);
    }
  };

  const handleUnlock = async (posIndex: number) => {
    setActionLoading(posIndex);
    setQcreditError(null);
    setQcreditSuccess(null);
    try {
      const result = await unlockQCredit(walletAddress, posIndex, getAuthHeaders());
      setQcreditSuccess(`Unlocked! Returned ${result.qug_returned} QUG + ${result.yield_claimed} yield`);
      const [status, posData] = await Promise.all([
        getQCreditStatus(),
        getQCreditPosition(getAuthHeaders()),
      ]);
      setQcreditStatus(status);
      setQcreditPositions(posData.positions);
      setQcreditTotalLocked(posData.total_locked);
      setQcreditTotalPending(posData.total_pending_yield);
    } catch (err: any) {
      setQcreditError(err.message || 'Unlock failed');
    } finally {
      setActionLoading(null);
    }
  };

  const handleClaimYield = async (posIndex: number) => {
    setActionLoading(posIndex + 10000); // offset to differentiate from unlock
    setQcreditError(null);
    setQcreditSuccess(null);
    try {
      const result = await claimQCreditYield(walletAddress, posIndex, getAuthHeaders());
      setQcreditSuccess(`Claimed ${result.yield_claimed} QUG yield`);
      const [status, posData] = await Promise.all([
        getQCreditStatus(),
        getQCreditPosition(getAuthHeaders()),
      ]);
      setQcreditStatus(status);
      setQcreditPositions(posData.positions);
      setQcreditTotalLocked(posData.total_locked);
      setQcreditTotalPending(posData.total_pending_yield);
    } catch (err: any) {
      setQcreditError(err.message || 'Claim failed');
    } finally {
      setActionLoading(null);
    }
  };

  const tierInfo: Record<string, { label: string; days: number; apy: number; color: string; gradient: string }> = {
    bronze:   { label: 'Bronze',   days: 7,   apy: 5,  color: 'text-amber-600',    gradient: 'from-amber-700 to-amber-500' },
    silver:   { label: 'Silver',   days: 30,  apy: 10, color: 'text-slate-300',     gradient: 'from-slate-400 to-slate-200' },
    gold:     { label: 'Gold',     days: 90,  apy: 15, color: 'text-yellow-400',    gradient: 'from-yellow-500 to-yellow-300' },
    platinum: { label: 'Platinum', days: 180, apy: 25, color: 'text-cyan-300',      gradient: 'from-cyan-400 to-purple-400' },
  };

  // v2.3.6-beta: Real market stats from AMM oracle
  const [realMarketStats, setRealMarketStats] = useState<{
    marketCap: number;
    liquidity: number;
    totalSupply: number;
    holders: number;
    price: number;
  } | null>(null);

  // v2.3.6-beta: Fetch real market stats from AMM oracle
  useEffect(() => {
    if (!token) return;
    let mounted = true;

    const fetchRealMarketStats = async () => {
      try {
        // Try both symbol and full token ID for lookup
        // Token ID might be the full contract address like "qnk1964527556e0d0970dc31d94a64ae44407c7a1522663993c3b01acdb1576a5ed"
        const tokenIdToFetch = token.id?.startsWith('qnk') ? token.id : token.symbol;

        // v2.3.8-beta: Fetch QUG price from oracle for proper liquidity calculation
        let qugPriceUsd = token.price; // Fallback to token price
        try {
          const qugResponse = await fetch('/api/v1/oracle/price/QUG');
          const qugData = await qugResponse.json();
          if (qugData.success && qugData.data?.price_usd) {
            qugPriceUsd = qugData.data.price_usd;
          }
        } catch {
          console.warn('Failed to fetch QUG price, using token price as fallback');
        }

        // Fetch from AMM price oracle
        const response = await fetch(`/api/v1/oracle/price/${encodeURIComponent(tokenIdToFetch)}`);
        const data = await response.json();

        if (data.success && data.data && mounted) {
          const priceData = data.data;
          const poolReserves = priceData.pool_reserves;

          // Calculate real liquidity from pool reserves
          let realLiquidity = 0;
          if (poolReserves) {
            // v8.4.5: Fix liquidity calculation — use correct price per token
            // Old code assumed reserve1 is always QUG and multiplied by qugPriceUsd,
            // but for QUG/QUGUSD pool, reserve1 is QUGUSD ($1 stablecoin).
            // Multiplying 27.8M QUGUSD × $2730 QUG price = $75.9B (wrong!)
            const reserve0Parsed = parseU128(poolReserves.reserve0);
            const reserve1Parsed = parseU128(poolReserves.reserve1);
            const token0Upper = (poolReserves.token0 || '').toUpperCase();
            const token1Upper = (poolReserves.token1 || '').toUpperCase();
            // Price each reserve correctly based on what token it is
            const reserve0Price = token0Upper === 'QUGUSD' ? 1.0
              : token0Upper === 'QUG' ? qugPriceUsd
              : priceData.price_usd;
            const reserve1Price = token1Upper === 'QUGUSD' ? 1.0
              : token1Upper === 'QUG' ? qugPriceUsd
              : priceData.price_usd;
            const reserve0Value = reserve0Parsed * reserve0Price;
            const reserve1Value = reserve1Parsed * reserve1Price;
            realLiquidity = reserve0Value + reserve1Value;

            console.log('📊 Pool reserves:', {
              token0: poolReserves.token0,
              reserve0: reserve0Parsed,
              token1: poolReserves.token1,
              reserve1: reserve1Parsed,
              qugPrice: qugPriceUsd,
              calculatedLiquidity: realLiquidity,
            });
          }

          // Calculate market cap from CIRCULATING supply (not total/max supply)
          // v8.4.5: Fix — was using totalSupply (21M) instead of circulatingSupply (~21K)
          // Market cap = circulating supply × price (same as every other blockchain)
          // FDV = total supply × price (shown separately)
          const circulatingSupply = token.circulatingSupply > 0 ? token.circulatingSupply
            : (token.totalSupply > 0 ? token.totalSupply : 0);
          const actualSupply = token.totalSupply > 0 ? token.totalSupply : 0;
          const realMarketCap = circulatingSupply * priceData.price_usd;

          setRealMarketStats({
            price: priceData.price_usd,
            marketCap: realMarketCap,
            liquidity: realLiquidity,
            totalSupply: actualSupply,
            holders: token.holders, // v2.3.8-beta: Use real holder count from token data
          });

          console.log('📊 Real market stats loaded:', {
            price: priceData.price_usd,
            marketCap: realMarketCap,
            liquidity: realLiquidity,
            source: priceData.source,
          });
        } else if (!data.success) {
          // v3.6.14: No oracle data available - USE TOKEN PROP VALUES (from DexScreen)
          // These already have correct data from the API, don't reset to zero!
          console.log('📊 No oracle data for token, using token prop values:', {
            price: token.price,
            marketCap: token.marketCap,
            liquidity: token.liquidity,
          });
          setRealMarketStats({
            price: token.price || 0,
            marketCap: token.marketCap || 0,
            liquidity: token.liquidity || 0,
            totalSupply: token.totalSupply || 0,
            holders: token.holders || 0,
          });
        }
      } catch (error) {
        console.warn('Failed to fetch real market stats, using token data:', error);
        // v2.3.8-beta: Use token data (which now comes from real APIs) instead of zeros
        setRealMarketStats({
          price: token.price || 0,
          marketCap: token.marketCap || 0,
          liquidity: token.liquidity || 0,
          totalSupply: token.totalSupply || 0,
          holders: token.holders || 0,
        });
      }
    };

    fetchRealMarketStats();

    return () => {
      mounted = false;
    };
  }, [token]);

  // Fetch real price history from backend with SSE real-time updates
  useEffect(() => {
    if (!token) return;
    let mounted = true;
    let eventSource: EventSource | null = null;

    // Fetch initial price history from backend
    const fetchPriceHistory = async () => {
      try {
        const response = await qnkAPI.getTokenPriceHistory(token.id, timeframe);
        if (response.success && response.data && mounted) {
          setPriceData(response.data);
          console.log('✅ Loaded price history from backend:', response.data.length, 'data points');
        }
      } catch (error) {
        console.error('Failed to fetch price history:', error);
        // Fallback to current price if no history available
        if (mounted) {
          setPriceData([{
            timestamp: Date.now(),
            price: token.price,
            volume: token.volume24h
          }]);
        }
      }
    };

    fetchPriceHistory();

    // Set up SSE for real-time price updates
    const sseUrl = import.meta.env.VITE_API_URL ?
      `${import.meta.env.VITE_API_URL}/v1/events` :
      '/api/v1/events';

    try {
      eventSource = new EventSource(sseUrl);

      eventSource.addEventListener('token_price_update', (event) => {
        if (!mounted) return;
        try {
          const data = JSON.parse(event.data);
          if (data.token_id === token.id) {
            console.log('📈 Received price update for token:', data);
            // Append new data point to chart
            setPriceData(prev => [...prev, {
              timestamp: data.timestamp * 1000, // Convert to milliseconds
              price: data.price,
              volume: data.volume_24h || 0
            }].slice(-100000)); // Keep last 100k points for performance
          }
        } catch (err) {
          console.error('Failed to parse price update:', err);
        }
      });

    } catch (error) {
      console.error('Failed to establish SSE connection:', error);
    }

    return () => {
      mounted = false;
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [token, timeframe]);

  // Fetch real transaction history from backend with SSE real-time updates
  useEffect(() => {
    if (!token) return;
    let mounted = true;
    let eventSource: EventSource | null = null;

    // Fetch initial transaction history from backend
    const fetchTransactions = async () => {
      try {
        const response = await qnkAPI.getTokenTransactions(token.id);
        if (response.success && response.data && mounted) {
          setTransactions(response.data);
          console.log('✅ Loaded transactions from backend:', response.data.length, 'transactions');
        }
      } catch (error) {
        console.error('Failed to fetch transactions:', error);
        if (mounted) {
          setTransactions([]);
        }
      }
    };

    fetchTransactions();

    // Set up SSE for real-time transaction updates
    const sseUrl = import.meta.env.VITE_API_URL ?
      `${import.meta.env.VITE_API_URL}/v1/events` :
      '/api/v1/events';

    try {
      eventSource = new EventSource(sseUrl);

      eventSource.addEventListener('token_transaction', (event) => {
        if (!mounted) return;
        try {
          const data = JSON.parse(event.data);
          if (data.token_id === token.id) {
            console.log('📜 Received transaction for token:', data);
            // Prepend new transaction to list (keep last 100)
            setTransactions(prev => [{
              id: data.tx_hash,
              timestamp: data.timestamp,
              type: data.type,
              amount: data.amount,
              price: data.price,
              value: data.value,
              from: data.from,
              to: data.to,
              txHash: data.tx_hash,
            }, ...prev].slice(0, 100));
          }
        } catch (err) {
          console.error('Failed to parse transaction:', err);
        }
      });

    } catch (error) {
      console.error('Failed to establish SSE connection:', error);
    }

    return () => {
      mounted = false;
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [token]);

  // Draw the price chart on canvas
  useEffect(() => {
    if (!canvasRef.current || priceData.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    try {
      const width = canvas.width;
      const height = canvas.height;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // v4.0.5: Sort data oldest-first for correct left-to-right chronological rendering
      // Backend returns newest-first (RocksDB inverted timestamps), but chart X-axis = time
      const sortedData = [...priceData].sort((a, b) => a.timestamp - b.timestamp);

      // Calculate price range
      const prices = sortedData.map(d => d.price);
      const minPrice = Math.min(...prices);
      const maxPrice = Math.max(...prices);
      const priceRange = maxPrice - minPrice;
      const padding = 40;

      // Draw grid
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.1)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = padding + (height - 2 * padding) * (i / 5);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
      }

      // Draw gradient area under the line
      const gradient = ctx.createLinearGradient(0, padding, 0, height - padding);
      gradient.addColorStop(0, 'rgba(34, 211, 238, 0.4)');
      gradient.addColorStop(0.5, 'rgba(168, 85, 247, 0.2)');
      gradient.addColorStop(1, 'rgba(168, 85, 247, 0.0)');

      ctx.beginPath();
      sortedData.forEach((point, i) => {
        const x = padding + (width - 2 * padding) * (i / (sortedData.length - 1));
        const y = height - padding - ((point.price - minPrice) / priceRange) * (height - 2 * padding);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.lineTo(width - padding, height - padding);
      ctx.lineTo(padding, height - padding);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();

      // Draw main price line with glow effect
      ctx.shadowBlur = 15;
      ctx.shadowColor = 'rgba(34, 211, 238, 0.8)';
      ctx.strokeStyle = 'rgba(34, 211, 238, 1)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      sortedData.forEach((point, i) => {
        const x = padding + (width - 2 * padding) * (i / (sortedData.length - 1));
        const y = height - padding - ((point.price - minPrice) / priceRange) * (height - 2 * padding);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw Y-axis price labels
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.font = '12px monospace';
      ctx.textAlign = 'right';
      for (let i = 0; i <= 5; i++) {
        const price = maxPrice - (priceRange * (i / 5));
        const y = padding + (height - 2 * padding) * (i / 5);
        ctx.fillText(`$${price.toFixed(4)}`, padding - 10, y + 4);
      }

      // v4.0.5: Draw X-axis time labels
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      const xLabelCount = Math.min(5, sortedData.length);
      for (let i = 0; i < xLabelCount; i++) {
        const dataIdx = Math.floor(i * (sortedData.length - 1) / (xLabelCount - 1));
        const x = padding + (width - 2 * padding) * (dataIdx / (sortedData.length - 1));
        const ts = sortedData[dataIdx].timestamp;
        const d = new Date(ts);
        const label = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        ctx.fillText(label, x, height - padding + 16);
      }

      // Draw hover crosshair and tooltip
      if (hoveredPoint && mousePosition.x > 0) {
        const x = mousePosition.x;
        const y = mousePosition.y;

        // Vertical line
        ctx.strokeStyle = 'rgba(34, 211, 238, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, height - padding);
        ctx.stroke();

        // Horizontal line
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw point
        ctx.fillStyle = 'rgba(34, 211, 238, 1)';
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();

        // Draw glow around point
        ctx.shadowBlur = 20;
        ctx.shadowColor = 'rgba(34, 211, 238, 1)';
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    } catch (error) {
      console.error('Error drawing canvas:', error);
    }
  }, [priceData, hoveredPoint, mousePosition]);

  // Handle mouse move on canvas
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || priceData.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const padding = 40;
    const width = canvas.width;

    // v4.0.5: Use sorted data (oldest-first) to match chart rendering order
    const sortedData = [...priceData].sort((a, b) => a.timestamp - b.timestamp);
    // v8.5.3: Snap to nearest data point with precise Y position
    const dataIndex = Math.round(((x - padding) / (width - 2 * padding)) * (sortedData.length - 1));
    if (dataIndex >= 0 && dataIndex < sortedData.length) {
      const point = sortedData[dataIndex];
      // Calculate the precise Y position from the data point's price
      const height = canvas.height;
      const prices = sortedData.map(p => p.price);
      const minPrice = Math.min(...prices);
      const maxPrice = Math.max(...prices);
      const priceRange = maxPrice - minPrice || 1;
      const snappedY = height - padding - ((point.price - minPrice) / priceRange) * (height - 2 * padding);
      // Calculate snapped X position too
      const snappedX = padding + (dataIndex / Math.max(sortedData.length - 1, 1)) * (width - 2 * padding);
      setHoveredPoint(point);
      setMousePosition({ x: snappedX, y: snappedY });
    }
  };

  const handleCanvasMouseLeave = () => {
    setHoveredPoint(null);
    setMousePosition({ x: 0, y: 0 });
  };

  // Sort and filter transactions
  const filteredAndSortedTransactions = () => {
    // Filter by type
    let filtered = transactions;
    if (filterType !== 'all') {
      filtered = transactions.filter(tx => tx.type === filterType);
    }

    // Sort by selected field
    const sorted = [...filtered].sort((a, b) => {
      let comparison = 0;

      switch (sortField) {
        case 'timestamp':
          comparison = a.timestamp - b.timestamp;
          break;
        case 'amount':
          comparison = a.amount - b.amount;
          break;
        case 'value':
          comparison = a.value - b.value;
          break;
        case 'type':
          comparison = a.type.localeCompare(b.type);
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return sorted;
  };

  // Toggle sort field and order
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      // Toggle order if same field
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      // Set new field with desc as default
      setSortField(field);
      setSortOrder('desc');
    }
  };

  // Get sort icon for column
  const getSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return <ArrowUpDown className="w-4 h-4 opacity-40" />;
    }
    return sortOrder === 'asc'
      ? <ArrowUp className="w-4 h-4 text-quantum-cyan" />
      : <ArrowDown className="w-4 h-4 text-quantum-cyan" />;
  };

  const formatLargeNumber = (num: number, addDollarSign: boolean = true) => {
    const prefix = addDollarSign ? '$' : '';
    if (num >= 1e12) return `${prefix}${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `${prefix}${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${prefix}${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `${prefix}${(num / 1e3).toFixed(2)}K`;
    return `${prefix}${num.toFixed(2)}`;
  };

  // Early return if no token - this prevents the modal from rendering at all
  if (!token) return null;

  const modalContent = (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
        <div
          onClick={(e) => e.stopPropagation()}
          className="relative w-full max-w-6xl max-h-[90vh] overflow-y-auto bg-gradient-to-br from-quantum-dark via-quantum-indigo/20 to-quantum-purple/10 rounded-3xl border border-quantum-cyan/30 shadow-2xl"
        >
          {/* Animated background effects */}
          <div className="absolute inset-0 overflow-hidden rounded-3xl pointer-events-none">
            <motion.div
              className="absolute w-96 h-96 bg-gradient-to-r from-quantum-cyan/20 to-quantum-purple/20 rounded-full blur-3xl"
              animate={{
                x: [0, 100, 0],
                y: [0, 50, 0],
              }}
              transition={{ duration: 15, repeat: Infinity }}
            />
          </div>

          {/* Header */}
          <div className="relative z-10 p-6 border-b border-white/10">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 bg-gradient-to-br from-quantum-cyan to-quantum-purple rounded-2xl flex items-center justify-center shadow-lg">
                  <TokenIcon symbol={token.symbol} icon={token.icon} logoUrl={(token as any).logoUrl} size={48} />
                </div>
                <div>
                  <h2 className="text-3xl font-black text-white">{token.name}</h2>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-lg text-gray-400">{token.symbol}</span>
                    {token.features.quantumSecured && (
                      <span className="px-2 py-1 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-lg text-xs font-bold">
                        ⚛️ Quantum Secured
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
              >
                <X className="w-6 h-6 text-white" />
              </button>
            </div>

            {/* Price and Change */}
            <div className="mt-6 flex items-end gap-4">
              <div className="text-5xl font-black text-white">
                ${hoveredPoint ? (hoveredPoint.price < 1 ? hoveredPoint.price.toPrecision(6) : hoveredPoint.price < 100 ? hoveredPoint.price.toFixed(4) : hoveredPoint.price.toFixed(2)) : token.price.toLocaleString()}
              </div>
              <div className={`flex items-center gap-2 text-2xl font-bold mb-2 ${
                token.change24h > 0 ? 'text-quantum-green' : 'text-red-500'
              }`}>
                {token.change24h > 0 ? <TrendingUp className="w-6 h-6" /> : <TrendingDown className="w-6 h-6" />}
                {token.change24h > 0 ? '+' : ''}{token.change24h.toFixed(2)}%
              </div>
            </div>

            {/* Hover tooltip */}
            {hoveredPoint && (
              <div className="mt-2 text-sm text-gray-400">
                {new Date(hoveredPoint.timestamp).toLocaleString()} - Volume: {formatLargeNumber(hoveredPoint.volume)}
              </div>
            )}
          </div>

          {/* Two Column Layout: Graph + Info */}
          <div className="relative z-10 p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* LEFT COLUMN: Price Chart */}
            <div className="flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-white">Price Chart (100ms Resolution)</h3>
                <div className="flex gap-1">
                  {(['1H', '24H', '7D', '30D', '1Y'] as const).map((tf) => (
                    <button
                      key={tf}
                      onClick={() => setTimeframe(tf)}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                        timeframe === tf
                          ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                          : 'bg-white/5 text-gray-400 hover:bg-white/10'
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              </div>

              <div className="relative bg-black/40 rounded-2xl p-4 border border-quantum-cyan/20 flex-1">
                <canvas
                  ref={canvasRef}
                  width={1200}
                  height={600}
                  className="w-full h-full cursor-crosshair"
                  onMouseMove={handleCanvasMouseMove}
                  onMouseLeave={handleCanvasMouseLeave}
                />
              </div>
            </div>

            {/* RIGHT COLUMN: All Info */}
            <div className="flex flex-col gap-6 overflow-y-auto max-h-[700px]">

              {/* v8.5.5: QCREDIT Yield Vault Panel */}
              {isQCredit ? (
                <>
                  {/* Vault Status Overview */}
                  <div>
                    <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                      <Zap className="w-5 h-5 text-amber-400" />
                      Yield Vault
                      <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded">LIVE</span>
                    </h3>
                    {qcreditLoading && !qcreditStatus ? (
                      <div className="flex items-center justify-center py-8">
                        <div className="w-6 h-6 border-2 border-amber-400/30 border-t-amber-400 rounded-full animate-spin" />
                      </div>
                    ) : qcreditStatus ? (
                      <div className="grid grid-cols-2 gap-3">
                        <StatCard icon={<Lock className="w-5 h-5" />} label="Total Value Locked" value={`${parseFloat(qcreditStatus.total_locked).toLocaleString(undefined, { maximumFractionDigits: 2 })} QUG`} color="from-amber-500 to-orange-500" />
                        <StatCard icon={<Coins className="w-5 h-5" />} label="QCREDIT Supply" value={parseFloat(qcreditStatus.total_qcredit_supply).toLocaleString(undefined, { maximumFractionDigits: 2 })} color="from-cyan-500 to-blue-500" />
                        <StatCard icon={<Shield className="w-5 h-5" />} label="Protocol Reserve" value={`${parseFloat(qcreditStatus.protocol_reserve).toLocaleString(undefined, { maximumFractionDigits: 2 })} QUG`} color="from-emerald-500 to-green-500" />
                        <StatCard icon={<Users className="w-5 h-5" />} label="Active Positions" value={qcreditStatus.position_count.toString()} color="from-purple-500 to-pink-500" />
                      </div>
                    ) : (
                      <div className="text-center py-6 text-gray-500">Unable to load vault status</div>
                    )}
                  </div>

                  {/* Yield Tiers */}
                  <div>
                    <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                      <Gift className="w-5 h-5 text-yellow-400" />
                      Yield Tiers
                    </h3>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(tierInfo).map(([key, info]) => (
                        <div key={key} className={`relative p-3 rounded-xl border transition-all cursor-pointer ${
                          selectedTier === key
                            ? 'border-amber-500/50 bg-gradient-to-br from-amber-500/10 to-orange-500/10 ring-1 ring-amber-500/30'
                            : 'border-white/10 bg-black/40 hover:border-white/20'
                        }`} onClick={() => setSelectedTier(key)}>
                          <div className="flex items-center justify-between mb-1">
                            <span className={`text-sm font-bold ${info.color}`}>{info.label}</span>
                            <span className="text-lg font-black bg-gradient-to-r from-emerald-400 to-green-300 bg-clip-text text-transparent">{info.apy}%</span>
                          </div>
                          <div className="text-xs text-gray-500">Lock: {info.days} days</div>
                          <div className="text-[10px] text-gray-600 mt-0.5">APY paid in QUG</div>
                          {selectedTier === key && (
                            <div className="absolute top-1.5 right-1.5 w-2 h-2 bg-amber-400 rounded-full shadow-lg shadow-amber-400/50" />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Lock QUG Form */}
                  <div>
                    <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                      <Lock className="w-5 h-5 text-amber-400" />
                      Lock QUG
                    </h3>
                    <div className="bg-black/40 rounded-xl border border-white/10 p-4 space-y-3">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 relative">
                          <input
                            type="number"
                            value={lockAmount}
                            onChange={(e) => setLockAmount(e.target.value)}
                            placeholder="Amount of QUG to lock"
                            className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/30 text-sm"
                            min="0"
                            step="0.01"
                          />
                          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-500">QUG</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-gray-500">
                        <span>Selected tier:</span>
                        <span className={`font-bold ${tierInfo[selectedTier]?.color || 'text-white'}`}>
                          {tierInfo[selectedTier]?.label || selectedTier} ({tierInfo[selectedTier]?.apy || 0}% APY, {tierInfo[selectedTier]?.days || 0}d lock)
                        </span>
                      </div>
                      <button
                        onClick={handleLockQug}
                        disabled={lockLoading || !lockAmount || parseFloat(lockAmount) <= 0}
                        className="w-full py-2.5 rounded-lg font-bold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-amber-500 to-orange-500 text-white hover:shadow-lg hover:shadow-amber-500/30"
                      >
                        {lockLoading ? (
                          <span className="flex items-center justify-center gap-2">
                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Locking...
                          </span>
                        ) : (
                          `Lock ${lockAmount || '0'} QUG in ${tierInfo[selectedTier]?.label || selectedTier}`
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Status Messages */}
                  {qcreditError && (
                    <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
                      <AlertCircle className="w-4 h-4 flex-shrink-0" />
                      {qcreditError}
                    </div>
                  )}
                  {qcreditSuccess && (
                    <div className="flex items-center gap-2 p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-emerald-400 text-sm">
                      <CheckCircle className="w-4 h-4 flex-shrink-0" />
                      {qcreditSuccess}
                    </div>
                  )}

                  {/* User Positions */}
                  {walletAddress && (
                    <div>
                      <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-cyan-400" />
                        Your Positions
                        {qcreditPositions.length > 0 && (
                          <span className="text-xs bg-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded">{qcreditPositions.length}</span>
                        )}
                      </h3>

                      {/* Summary */}
                      {qcreditPositions.length > 0 && (
                        <div className="grid grid-cols-2 gap-2 mb-3">
                          <div className="p-2.5 bg-black/40 border border-white/10 rounded-lg">
                            <div className="text-[10px] text-gray-500 uppercase tracking-wider">Your Locked</div>
                            <div className="text-sm font-bold text-white font-mono">{parseFloat(qcreditTotalLocked).toLocaleString(undefined, { maximumFractionDigits: 4 })} QUG</div>
                          </div>
                          <div className="p-2.5 bg-black/40 border border-white/10 rounded-lg">
                            <div className="text-[10px] text-gray-500 uppercase tracking-wider">Pending Yield</div>
                            <div className="text-sm font-bold text-emerald-400 font-mono">{parseFloat(qcreditTotalPending).toLocaleString(undefined, { maximumFractionDigits: 4 })} QUG</div>
                          </div>
                        </div>
                      )}

                      {/* Position Cards */}
                      {qcreditPositions.length === 0 ? (
                        <div className="text-center py-6 bg-black/30 rounded-xl border border-white/5">
                          <Lock className="w-8 h-8 text-gray-600 mx-auto mb-2" />
                          <p className="text-sm text-gray-500">No active positions</p>
                          <p className="text-xs text-gray-600 mt-1">Lock QUG above to start earning yield</p>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {qcreditPositions.map((pos) => {
                            const ti = tierInfo[pos.tier.toLowerCase()] || tierInfo.bronze;
                            const lockProgress = pos.lock_days_remaining > 0
                              ? Math.max(0, Math.min(100, (1 - pos.lock_days_remaining / (ti.days || 1)) * 100))
                              : 100;
                            return (
                              <div key={pos.index} className="bg-black/40 border border-white/10 rounded-xl p-3 space-y-2">
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-2">
                                    <span className={`text-xs font-bold px-2 py-0.5 rounded bg-gradient-to-r ${ti.gradient} text-white`}>
                                      {pos.tier}
                                    </span>
                                    <span className="text-sm font-bold text-white font-mono">{parseFloat(pos.amount_locked).toLocaleString(undefined, { maximumFractionDigits: 4 })} QUG</span>
                                  </div>
                                  <span className="text-xs font-bold text-emerald-400">{pos.apy_percent}% APY</span>
                                </div>

                                {/* Lock progress bar */}
                                <div>
                                  <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                                    <span>Lock progress</span>
                                    <span>{pos.is_unlockable ? 'Unlockable' : `${pos.lock_days_remaining}d remaining`}</span>
                                  </div>
                                  <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                                    <div
                                      className={`h-full rounded-full transition-all ${pos.is_unlockable ? 'bg-emerald-500' : 'bg-gradient-to-r from-amber-500 to-orange-500'}`}
                                      style={{ width: `${lockProgress}%` }}
                                    />
                                  </div>
                                </div>

                                {/* Yield info */}
                                <div className="flex items-center justify-between text-xs">
                                  <div className="text-gray-500">
                                    Pending: <span className="text-emerald-400 font-mono">{parseFloat(pos.pending_yield).toLocaleString(undefined, { maximumFractionDigits: 6 })} QUG</span>
                                  </div>
                                  <div className="text-gray-600">
                                    Claimed: <span className="font-mono">{parseFloat(pos.claimed_yield).toLocaleString(undefined, { maximumFractionDigits: 4 })}</span>
                                  </div>
                                </div>

                                {/* Action Buttons */}
                                <div className="flex gap-2">
                                  <button
                                    onClick={() => handleClaimYield(pos.index)}
                                    disabled={actionLoading !== null || parseFloat(pos.pending_yield) <= 0}
                                    className="flex-1 py-1.5 rounded-lg text-xs font-bold transition-all disabled:opacity-30 disabled:cursor-not-allowed bg-emerald-500/20 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/30"
                                  >
                                    {actionLoading === pos.index + 10000 ? (
                                      <span className="flex items-center justify-center gap-1">
                                        <div className="w-3 h-3 border border-emerald-400/30 border-t-emerald-400 rounded-full animate-spin" />
                                      </span>
                                    ) : (
                                      <span className="flex items-center justify-center gap-1"><Gift className="w-3 h-3" /> Claim Yield</span>
                                    )}
                                  </button>
                                  <button
                                    onClick={() => handleUnlock(pos.index)}
                                    disabled={actionLoading !== null || !pos.is_unlockable}
                                    className="flex-1 py-1.5 rounded-lg text-xs font-bold transition-all disabled:opacity-30 disabled:cursor-not-allowed bg-amber-500/20 text-amber-400 border border-amber-500/20 hover:bg-amber-500/30"
                                  >
                                    {actionLoading === pos.index ? (
                                      <span className="flex items-center justify-center gap-1">
                                        <div className="w-3 h-3 border border-amber-400/30 border-t-amber-400 rounded-full animate-spin" />
                                      </span>
                                    ) : (
                                      <span className="flex items-center justify-center gap-1"><Unlock className="w-3 h-3" /> Unlock</span>
                                    )}
                                  </button>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}

                  {/* How It Works */}
                  <div className="bg-black/30 border border-white/5 rounded-xl p-4">
                    <h4 className="text-sm font-bold text-gray-300 mb-2">How QCREDIT Works</h4>
                    <div className="space-y-1.5 text-xs text-gray-500">
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[10px] font-bold">1</span>
                        Lock QUG in a tier (7-180 days)
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[10px] font-bold">2</span>
                        Receive QCREDIT 1:1 (tradeable on DEX)
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[10px] font-bold">3</span>
                        Earn yield (claim anytime, paid in QUG)
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[10px] font-bold">4</span>
                        Unlock after lock period (burn QCREDIT, get QUG back)
                      </div>
                    </div>
                  </div>
                </>
              ) : (
              <>
              {/* Stats Grid - v2.3.6-beta: Uses real market stats from AMM when available */}
              <div>
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  Market Stats
                  {realMarketStats && (
                    <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded">LIVE</span>
                  )}
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  <StatCard
                    icon={<Activity className="w-5 h-5" />}
                    label="Market Cap"
                    value={formatLargeNumber(realMarketStats?.marketCap ?? token.marketCap)}
                    color="from-cyan-500 to-blue-500"
                  />
                  <StatCard
                    icon={<TrendingUp className="w-5 h-5" />}
                    label="FDV (Fully Diluted)"
                    value={formatLargeNumber(token.totalSupply * (realMarketStats?.price ?? token.price))}
                    color="from-indigo-500 to-purple-500"
                  />
                  <StatCard
                    icon={<Coins className="w-5 h-5" />}
                    label="Total Supply"
                    value={formatLargeNumber(realMarketStats?.totalSupply ?? token.totalSupply, false)}
                    color="from-purple-500 to-pink-500"
                  />
                  <StatCard
                    icon={<Droplet className="w-5 h-5" />}
                    label="Liquidity"
                    value={formatLargeNumber(realMarketStats?.liquidity ?? token.liquidity)}
                    color="from-green-500 to-teal-500"
                  />
                  <StatCard
                    icon={<Users className="w-5 h-5" />}
                    label="Holders"
                    value={(realMarketStats?.holders ?? token.holders).toLocaleString()}
                    color="from-orange-500 to-red-500"
                  />
                  <StatCard
                    icon={<ArrowUpDown className="w-5 h-5" />}
                    label="24h Volume"
                    value={formatLargeNumber(token.volume24h)}
                    color="from-yellow-500 to-orange-500"
                  />
                </div>
              </div>

              {/* Transaction Fees — only show if any fee > 0 */}
              {(token.fees.buy > 0 || token.fees.sell > 0 || token.fees.transfer > 0) && (
              <div>
                <h3 className="text-xl font-bold text-white mb-4">Transaction Fees</h3>
                <div className="grid grid-cols-3 gap-3">
                  <FeeCard label="Buy" percentage={token.fees.buy} />
                  <FeeCard label="Sell" percentage={token.fees.sell} />
                  <FeeCard label="Transfer" percentage={token.fees.transfer} />
                </div>
              </div>
              )}

              {/* Token Features — only show features that are active */}
              {(token.features.reflection || token.features.autoLiquidity || token.features.buybackAndBurn || token.features.antiWhale || token.features.quantumSecured) && (
              <div>
                <h3 className="text-xl font-bold text-white mb-4">Token Features</h3>
                <div className="grid grid-cols-1 gap-3">
                  {token.features.quantumSecured && (
                  <FeatureCard
                    icon={<Shield className="w-5 h-5" />}
                    title="Quantum Security"
                    description="Post-quantum cryptographic protection (Dilithium5 + Kyber1024)"
                    active={true}
                  />
                  )}
                  {token.features.reflection && (
                  <FeatureCard
                    icon={<Droplet className="w-5 h-5" />}
                    title="Reflection"
                    description="Earn passive rewards from every transaction"
                    active={true}
                  />
                  )}
                  {token.features.autoLiquidity && (
                  <FeatureCard
                    icon={<Zap className="w-5 h-5" />}
                    title="Auto-Liquidity"
                    description="Automatic liquidity pool growth"
                    active={true}
                  />
                  )}
                  {token.features.buybackAndBurn && (
                  <FeatureCard
                    icon={<Activity className="w-5 h-5" />}
                    title="Buyback & Burn"
                    description="Deflationary token mechanics"
                    active={true}
                  />
                  )}
                  {token.features.antiWhale && (
                  <FeatureCard
                    icon={<Shield className="w-5 h-5" />}
                    title="Anti-Whale"
                    description="Protection against large holders"
                    active={true}
                  />
                  )}
                </div>
              </div>
              )}

              {/* Description */}
              <div>
                <h3 className="text-xl font-bold text-white mb-4">About {token.name}</h3>
                <p className="text-gray-300 leading-relaxed text-sm">{token.description}</p>
              </div>

              {/* v2.7.7-beta: Social Media & Links */}
              {token.socialLinks && Object.values(token.socialLinks).some(v => v) && (
                <div>
                  <h3 className="text-xl font-bold text-white mb-4">Social Media & Links</h3>
                  <div className="grid grid-cols-2 gap-3">
                    {token.socialLinks.twitter && (
                      <SocialLinkCard
                        icon={<Twitter className="w-5 h-5" />}
                        label="Twitter / X"
                        url={token.socialLinks.twitter.startsWith('http') ? token.socialLinks.twitter : `https://x.com/${token.socialLinks.twitter.replace('@', '')}`}
                        color="from-blue-400 to-blue-600"
                      />
                    )}
                    {token.socialLinks.discord && (
                      <SocialLinkCard
                        icon={<MessageCircle className="w-5 h-5" />}
                        label="Discord"
                        url={token.socialLinks.discord.startsWith('http') ? token.socialLinks.discord : `https://discord.gg/${token.socialLinks.discord}`}
                        color="from-indigo-400 to-purple-600"
                      />
                    )}
                    {token.socialLinks.telegram && (
                      <SocialLinkCard
                        icon={<MessageCircle className="w-5 h-5" />}
                        label="Telegram"
                        url={token.socialLinks.telegram.startsWith('http') ? token.socialLinks.telegram : `https://t.me/${token.socialLinks.telegram}`}
                        color="from-cyan-400 to-blue-500"
                      />
                    )}
                    {token.socialLinks.website && (
                      <SocialLinkCard
                        icon={<Globe className="w-5 h-5" />}
                        label="Website"
                        url={token.socialLinks.website.startsWith('http') ? token.socialLinks.website : `https://${token.socialLinks.website}`}
                        color="from-green-400 to-emerald-600"
                      />
                    )}
                    {token.socialLinks.github && (
                      <SocialLinkCard
                        icon={<Github className="w-5 h-5" />}
                        label="GitHub"
                        url={token.socialLinks.github.startsWith('http') ? token.socialLinks.github : `https://github.com/${token.socialLinks.github}`}
                        color="from-gray-400 to-gray-600"
                      />
                    )}
                    {token.socialLinks.medium && (
                      <SocialLinkCard
                        icon={<FileText className="w-5 h-5" />}
                        label="Medium"
                        url={token.socialLinks.medium.startsWith('http') ? token.socialLinks.medium : `https://medium.com/${token.socialLinks.medium}`}
                        color="from-green-500 to-teal-600"
                      />
                    )}
                    {token.socialLinks.reddit && (
                      <SocialLinkCard
                        icon={<MessageCircle className="w-5 h-5" />}
                        label="Reddit"
                        url={token.socialLinks.reddit.startsWith('http') ? token.socialLinks.reddit : `https://reddit.com/r/${token.socialLinks.reddit}`}
                        color="from-orange-400 to-red-500"
                      />
                    )}
                    {token.socialLinks.coinmarketcap && (
                      <SocialLinkCard
                        icon={<Coins className="w-5 h-5" />}
                        label="CoinMarketCap"
                        url={token.socialLinks.coinmarketcap}
                        color="from-blue-500 to-cyan-500"
                      />
                    )}
                    {token.socialLinks.coingecko && (
                      <SocialLinkCard
                        icon={<Coins className="w-5 h-5" />}
                        label="CoinGecko"
                        url={token.socialLinks.coingecko}
                        color="from-green-400 to-lime-500"
                      />
                    )}
                  </div>
                </div>
              )}
              </>
              )}
            </div>
          </div>

          {/* Transaction History */}
          <div className="relative z-10 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white">Transaction History</h3>

              {/* Filter Controls */}
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-400">Filter:</span>
                </div>
                <div className="flex gap-2">
                  {(['all', 'buy', 'sell', 'transfer'] as TransactionFilter[]).map((filter) => (
                    <button
                      key={filter}
                      onClick={() => setFilterType(filter)}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                        filterType === filter
                          ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                          : 'bg-white/5 text-gray-400 hover:bg-white/10'
                      }`}
                    >
                      {filter.charAt(0).toUpperCase() + filter.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Transaction Table */}
            <div className="bg-black/40 backdrop-blur-xl rounded-2xl border border-white/10 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gradient-to-r from-quantum-cyan/10 to-quantum-purple/10 border-b border-white/10">
                    <tr>
                      <th className="px-4 py-3 text-left">
                        <button
                          onClick={() => handleSort('type')}
                          className="flex items-center gap-2 text-sm font-bold text-white hover:text-quantum-cyan transition-colors"
                        >
                          Type
                          {getSortIcon('type')}
                        </button>
                      </th>
                      <th className="px-4 py-3 text-left">
                        <button
                          onClick={() => handleSort('timestamp')}
                          className="flex items-center gap-2 text-sm font-bold text-white hover:text-quantum-cyan transition-colors"
                        >
                          Time
                          {getSortIcon('timestamp')}
                        </button>
                      </th>
                      <th className="px-4 py-3 text-right">
                        <button
                          onClick={() => handleSort('amount')}
                          className="flex items-center gap-2 text-sm font-bold text-white hover:text-quantum-cyan transition-colors ml-auto"
                        >
                          Amount
                          {getSortIcon('amount')}
                        </button>
                      </th>
                      <th className="px-4 py-3 text-right">
                        <span className="text-sm font-bold text-white">Price</span>
                      </th>
                      <th className="px-4 py-3 text-right">
                        <button
                          onClick={() => handleSort('value')}
                          className="flex items-center gap-2 text-sm font-bold text-white hover:text-quantum-cyan transition-colors ml-auto"
                        >
                          Value
                          {getSortIcon('value')}
                        </button>
                      </th>
                      <th className="px-4 py-3 text-left">
                        <span className="text-sm font-bold text-white">From</span>
                      </th>
                      <th className="px-4 py-3 text-left">
                        <span className="text-sm font-bold text-white">To</span>
                      </th>
                      <th className="px-4 py-3 text-center">
                        <span className="text-sm font-bold text-white">Tx Hash</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {filteredAndSortedTransactions().length === 0 ? (
                      <tr>
                        <td colSpan={8} className="px-4 py-8 text-center text-gray-400">
                          No transactions found
                        </td>
                      </tr>
                    ) : (
                      filteredAndSortedTransactions().map((tx) => (
                        <motion.tr
                          key={tx.id}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="hover:bg-white/5 transition-colors"
                        >
                          {/* Type */}
                          <td className="px-4 py-3">
                            <span
                              className={`px-2 py-1 rounded-lg text-xs font-bold ${
                                tx.type === 'buy'
                                  ? 'bg-quantum-green/20 text-quantum-green'
                                  : tx.type === 'sell'
                                  ? 'bg-red-500/20 text-red-400'
                                  : 'bg-quantum-purple/20 text-quantum-purple'
                              }`}
                            >
                              {tx.type.toUpperCase()}
                            </span>
                          </td>

                          {/* Time */}
                          <td className="px-4 py-3">
                            <div className="text-sm text-gray-300">
                              {new Date(tx.timestamp).toLocaleDateString()}
                            </div>
                            <div className="text-xs text-gray-500">
                              {new Date(tx.timestamp).toLocaleTimeString()}
                            </div>
                          </td>

                          {/* Amount */}
                          <td className="px-4 py-3 text-right">
                            <div className="text-sm font-medium text-white">
                              {tx.amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                            </div>
                            <div className="text-xs text-gray-500">{token.symbol}</div>
                          </td>

                          {/* Price */}
                          <td className="px-4 py-3 text-right">
                            <span className="text-sm text-gray-300">
                              ${tx.price.toFixed(4)}
                            </span>
                          </td>

                          {/* Value */}
                          <td className="px-4 py-3 text-right">
                            <span className="text-sm font-medium text-white">
                              ${tx.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                            </span>
                          </td>

                          {/* From */}
                          <td className="px-4 py-3">
                            <span className="text-xs font-mono text-gray-400">
                              {tx.from ? `${tx.from.slice(0, 6)}...${tx.from.slice(-4)}` : 'N/A'}
                            </span>
                          </td>

                          {/* To */}
                          <td className="px-4 py-3">
                            <span className="text-xs font-mono text-gray-400">
                              {tx.to ? `${tx.to.slice(0, 6)}...${tx.to.slice(-4)}` : 'N/A'}
                            </span>
                          </td>

                          {/* Tx Hash */}
                          <td className="px-4 py-3 text-center">
                            {tx.txHash ? (
                              <a
                                href={`https://explorer.quillon.xyz/tx/${tx.txHash}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-xs font-mono text-quantum-cyan hover:text-quantum-purple transition-colors inline-flex items-center gap-1"
                              >
                                {tx.txHash.slice(0, 6)}...
                                <ExternalLink className="w-3 h-3" />
                              </a>
                            ) : (
                              <span className="text-xs text-gray-500">N/A</span>
                            )}
                          </td>
                        </motion.tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              {/* Transaction count */}
              <div className="px-4 py-3 bg-white/5 border-t border-white/10">
                <p className="text-sm text-gray-400 text-center">
                  Showing {filteredAndSortedTransactions().length} of {transactions.length} transactions
                </p>
              </div>
            </div>
          </div>

          {/* Links */}
          {(token.website || token.whitepaper) && (
            <div className="relative z-10 p-6 border-t border-white/10">
              <div className="flex gap-4">
                {token.website && (
                  <a
                    href={token.website}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl text-white font-bold hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all"
                  >
                    <ExternalLink className="w-5 h-5" />
                    Visit Website
                  </a>
                )}
                {token.whitepaper && (
                  <a
                    href={token.whitepaper}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-6 py-3 bg-white/10 rounded-xl text-white font-bold hover:bg-white/20 transition-all"
                  >
                    <Info className="w-5 h-5" />
                    Read Whitepaper
                  </a>
                )}
              </div>
            </div>
          )}
        </div>
    </div>
  );

  // Render modal in a portal to prevent parent re-renders from unmounting it
  return createPortal(modalContent, document.body);
}

// Helper Components
function StatCard({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string; color: string }) {
  return (
    <div className="relative group">
      <div className={`absolute -inset-0.5 bg-gradient-to-r ${color} rounded-xl blur opacity-30 group-hover:opacity-50 transition-opacity`} />
      <div className="relative bg-black/60 backdrop-blur-xl rounded-xl p-4 border border-white/10">
        <div className="flex items-center gap-2 text-gray-400 mb-2">
          {icon}
          <span className="text-sm font-medium">{label}</span>
        </div>
        <div className="text-2xl font-bold text-white">{value}</div>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description, active }: { icon: React.ReactNode; title: string; description: string; active: boolean }) {
  return (
    <div className={`relative p-4 rounded-xl border transition-all ${
      active
        ? 'bg-gradient-to-br from-quantum-cyan/10 to-quantum-purple/10 border-quantum-cyan/30'
        : 'bg-black/40 border-white/10 opacity-50'
    }`}>
      <div className={`flex items-center gap-3 mb-2 ${active ? 'text-quantum-cyan' : 'text-gray-500'}`}>
        {icon}
        <span className="font-bold">{title}</span>
      </div>
      <p className="text-sm text-gray-400">{description}</p>
      {active && (
        <div className="absolute top-2 right-2 w-2 h-2 bg-quantum-green rounded-full shadow-lg shadow-quantum-green/50" />
      )}
    </div>
  );
}

function FeeCard({ label, percentage }: { label: string; percentage: number }) {
  return (
    <div className="bg-black/40 backdrop-blur-xl rounded-xl p-4 border border-white/10">
      <div className="text-sm text-gray-400 mb-2">{label}</div>
      <div className="text-3xl font-bold bg-gradient-to-r from-quantum-cyan to-quantum-purple bg-clip-text text-transparent">
        {percentage}%
      </div>
    </div>
  );
}

// v2.7.7-beta: Social link card component
function SocialLinkCard({ icon, label, url, color }: { icon: React.ReactNode; label: string; url: string; color: string }) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="group relative block"
    >
      <div className={`absolute -inset-0.5 bg-gradient-to-r ${color} rounded-xl blur opacity-20 group-hover:opacity-50 transition-opacity`} />
      <div className="relative bg-black/60 backdrop-blur-xl rounded-xl p-3 border border-white/10 hover:border-white/30 transition-all">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg bg-gradient-to-r ${color} text-white`}>
            {icon}
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium text-white group-hover:text-quantum-cyan transition-colors">
              {label}
            </div>
            <div className="text-xs text-gray-500 truncate max-w-[150px]">
              {url.replace('https://', '').replace('http://', '')}
            </div>
          </div>
          <ExternalLink className="w-4 h-4 text-gray-500 group-hover:text-quantum-cyan transition-colors" />
        </div>
      </div>
    </a>
  );
}
