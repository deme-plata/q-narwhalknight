import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowDownUp, Search, TrendingUp, TrendingDown, Settings, Info, Droplet, Zap, X, Clock, Shield, AlertTriangle, Brain, Loader2, BarChart3, Activity, Layers, Users, DollarSign, Repeat, PieChart, Wallet, ChevronDown, ChevronUp, ChevronLeft, ChevronRight, Coins } from 'lucide-react';
import TokenDetailsModal from './TokenDetailsModal';
import IndexFundModal from './IndexFundModal';
import LiquidityModal from './LiquidityModal';
import TokenSelectorModal from './TokenSelectorModal';
import TokenIcon from './TokenIcon';
import NitroSuccessModal from './NitroSuccessModal';
import MintQUGUSDModal from './MintQUGUSDModal';
import SwapSuccessModal from './SwapSuccessModal';
import MarketAnalyzerPanel from './MarketAnalyzerPanel';
import XListCrowdfundModal from './XListCrowdfundModal';
import { qnkAPI } from '../services/api';

// v3.1.1: Helper to safely parse u128 values that may come as strings from the API
const parseU128 = (value: string | number | undefined): number => {
  if (value === undefined || value === null) return 0;
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    return isNaN(parsed) ? 0 : parsed;
  }
  return 0;
};

// v2.9.27-beta: Helper to format prices, especially very small ones
// v3.9.5-beta: Subscript zero notation for tiny prices (like DEXScreener)
// $0.000000038 → $0.0₇38 (subscript 7 = count of consecutive zeros after "0.0")
const SUBSCRIPT_DIGITS = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉'];
const toSubscript = (n: number): string => {
  return String(n).split('').map(d => SUBSCRIPT_DIGITS[parseInt(d)] || d).join('');
};

const formatPrice = (price: number): string => {
  if (price === 0) return '0';
  if (price >= 1000) {
    return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  } else if (price >= 1) {
    return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  } else if (price >= 0.01) {
    return price.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 });
  } else if (price >= 0.0001) {
    return price.toFixed(6);
  } else {
    // Tiny prices: use subscript zero notation
    // Count zeros after "0." and show as subscript
    const str = price.toFixed(20);
    const afterDot = str.split('.')[1] || '';
    let zeroCount = 0;
    for (const ch of afterDot) {
      if (ch === '0') zeroCount++;
      else break;
    }
    // Get 2-4 significant digits after the zeros
    const sigDigits = afterDot.slice(zeroCount, zeroCount + 4).replace(/0+$/, '') || '0';
    if (zeroCount >= 2) {
      return `0.0${toSubscript(zeroCount)}${sigDigits}`;
    }
    return price.toFixed(8).replace(/0+$/, '').replace(/\.$/, '');
  }
};

// v3.6.1-beta: SANITY CHECK - Max possible balance is 21 million QUG (total supply)
// Any balance exceeding this is corrupted data and must be rejected
const MAX_SANE_BALANCE = 21_000_000; // 21 million QUG

/**
 * v3.6.1-beta: Validate balance value to prevent corrupted data from being cached
 */
function isValidBalance(balance: number): boolean {
  if (typeof balance !== 'number') return false;
  if (isNaN(balance) || !isFinite(balance)) return false;
  if (balance < 0) return false;
  if (balance > MAX_SANE_BALANCE) {
    console.warn(`🚨 [DexScreen] Rejected corrupted balance: ${balance.toExponential()} > max supply ${MAX_SANE_BALANCE}`);
    return false;
  }
  return true;
}

/**
 * v3.6.1-beta: Safe localStorage set for cachedBalance - validates before storing
 */
function safeCacheBalance(balance: number): void {
  if (isValidBalance(balance)) {
    localStorage.setItem('cachedBalance', balance.toString());
  } else {
    console.warn(`🚨 [DexScreen] safeCacheBalance: Refusing to cache invalid balance: ${balance}`);
  }
}

// v3.2.22-beta: BigInt helpers for handling large token amounts without precision loss
// JavaScript Number.MAX_SAFE_INTEGER is ~9e15, but tokens can have 1e28+ amounts with 24 decimals

// Expand scientific notation strings to full numeric strings for BigInt parsing
const expandScientificNotation = (numStr: string): string => {
  if (!numStr.includes('e') && !numStr.includes('E')) {
    return numStr;
  }
  const num = parseFloat(numStr);
  if (!isFinite(num)) return '0';

  // For very large numbers, manually construct the string
  const match = numStr.match(/^([+-]?)(\d+\.?\d*)[eE]([+-]?\d+)$/);
  if (!match) return String(Math.floor(num)); // Fallback

  const [, sign, mantissa, expStr] = match;
  const exp = parseInt(expStr, 10);
  const mantissaClean = mantissa.replace('.', '');
  const mantissaDecimalPos = mantissa.indexOf('.');
  const decimalShift = mantissaDecimalPos >= 0 ? mantissa.length - mantissaDecimalPos - 1 : 0;
  const totalExp = exp - decimalShift;

  if (totalExp >= 0) {
    // Positive exponent: add zeros to the right
    return (sign === '-' ? '-' : '') + mantissaClean + '0'.repeat(totalExp);
  } else {
    // Negative exponent: would be a decimal, just return 0 for whole part
    return '0';
  }
};

// Parse a string amount to BigInt with specified decimals (for API calls)
const parseAmountToBigInt = (amount: string, decimals: number): bigint => {
  // Handle scientific notation like "1e+30"
  const str = expandScientificNotation(amount);

  // Remove commas that might be in the input (e.g., "99,999,999,999.9999")
  const cleanStr = str.replace(/,/g, '');

  const parts = cleanStr.split('.');
  const wholePart = parts[0] || '0';
  const fracPart = (parts[1] || '').slice(0, decimals).padEnd(decimals, '0');

  return BigInt(wholePart) * (10n ** BigInt(decimals)) + BigInt(fracPart);
};

// Maximum u128 value for validation
const U128_MAX = BigInt('340282366920938463463374607431768211455');

// v3.2.23-beta: Get decimals for a token symbol
// Native tokens (QUG, QUGUSD) use 24 decimals, custom tokens typically use 7-8
const getTokenDecimals = (symbol: string, tokenList?: Array<{symbol: string, decimals?: number}>): number => {
  const upperSymbol = symbol.toUpperCase();
  if (upperSymbol === 'QUG' || upperSymbol === 'QUGUSD' || upperSymbol === 'QCREDIT') {
    return 24;
  }
  // Try to find decimals from token list
  if (tokenList) {
    const token = tokenList.find(t => t.symbol.toUpperCase() === upperSymbol);
    if (token?.decimals) {
      return token.decimals;
    }
  }
  // Default for custom tokens
  return 8;
};

// v3.2.23-beta: Convert raw reserve to display amount using token decimals
const reserveToDisplay = (rawReserve: number | string, tokenSymbol: string, tokenList?: Array<{symbol: string, decimals?: number}>): number => {
  const raw = typeof rawReserve === 'string' ? parseFloat(rawReserve) : rawReserve;
  const decimals = getTokenDecimals(tokenSymbol, tokenList);
  return raw / Math.pow(10, decimals);
};

// v3.6.5-beta: Get display name for a token (symbol if known, shortened address otherwise)
const getTokenDisplayName = (tokenIdOrSymbol: string, tokenList?: Token[]): string => {
  // If it's already a short symbol (like "QUG", "ETH"), return as-is
  if (!tokenIdOrSymbol || tokenIdOrSymbol.length <= 10) {
    return tokenIdOrSymbol || 'Unknown';
  }

  // Try to find the token in the list by ID or symbol
  if (tokenList) {
    const token = tokenList.find(t =>
      t.id === tokenIdOrSymbol ||
      t.id.toLowerCase() === tokenIdOrSymbol.toLowerCase() ||
      t.symbol.toLowerCase() === tokenIdOrSymbol.toLowerCase()
    );
    if (token) {
      return token.symbol;
    }
  }

  // For long addresses, extract a readable short form
  // Format: first 6 chars + "..." (e.g., "qnk241..." from "qnk2411fd4ac0061a852f5c3aa122f4a3a...")
  if (tokenIdOrSymbol.startsWith('qnk')) {
    return tokenIdOrSymbol.slice(0, 9) + '...';
  }
  return tokenIdOrSymbol.slice(0, 6) + '...';
};

// DEX Settings Interface
interface DexSettings {
  slippageTolerance: number; // in percentage (0.1, 0.5, 1.0, custom)
  transactionDeadline: number; // in minutes
  expertMode: boolean;
  multihops: boolean;
  gasPreference: 'low' | 'medium' | 'high';
}

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
  logoUrl?: string;  // v2.4.8: Custom logo URL (data URL or IPFS URL)
  marketCap: number;              // Circulating supply * price
  fullyDilutedMarketCap?: number;  // Total supply * price (FDV) - optional for compatibility
  totalSupply: number;
  circulatingSupply: number;
  holders: number;
  decimals?: number;  // v3.2.15-beta: Token decimals (default 8 for custom tokens, 24 for QUG)
  createdAt?: number;  // v4.1.0: Deployment timestamp (epoch ms)
  txCount?: number;    // v4.1.0: Total transaction/swap count
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
  // Index fund specific properties
  isIndexToken?: boolean;
  indexData?: {
    methodology: 'market_cap_weighted' | 'equal_weighted' | 'custom';
    rebalanceFrequency: string;
    managementFee: number;
    performanceFee: number;
    navPerShare: number;
    totalAUM: number;
    components: Array<{
      symbol: string;
      name: string;
      weight: number;
      price: number;
      change24h: number;
    }>;
    lastRebalance: string;
    nextRebalance: string;
    inceptionDate: string;
    ytdReturn: number;
    allTimeReturn: number;
  };
  // v2.9.27-beta: Perpetual contract properties
  isPerp?: boolean;
  perpData?: {
    market: string;           // e.g., "QUG-PERP"
    markPrice: number;        // Mark price in USD
    indexPrice: number;       // Index/spot price in USD
    fundingRate: number;      // Current funding rate
    openInterest: number;     // Total open interest
    maxLeverage: number;      // Maximum allowed leverage
    maintenanceMargin: number;
    takerFee: number;
    makerFee: number;
  };
  // v8.2.8: XLIST crowdfunding campaign tokens
  isCrowdfund?: boolean;
  campaignData?: {
    campaign_id: string;
    exchange_name: string;
    exchange_logo: string;
    target_usd: number;
    raised_usd: number;
    contributor_count: number;
    early_bird_slots: number;
    early_bird_claimed: number;
    status: 'funding' | 'funded' | 'listed' | 'cancelled';
    tier: string;
    description: string;
    perks: {
      reduced_trading_fees: boolean;
      governance_voting: boolean;
      airdrop_multiplier: number;
      early_access: boolean;
      vip_support: boolean;
      nft_badge: boolean;
    };
  };
}

export default function DexScreen({ isActive }: { isActive?: boolean }) {
  const [swapFrom, setSwapFrom] = useState('QUG');
  const [swapTo, setSwapTo] = useState('QUGUSD');
  const [swapAmount, setSwapAmount] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'symbol' | 'price' | 'change1h' | 'change24h' | 'change7d' | 'volume24h' | 'liquidity' | 'marketCap' | 'holders' | 'txCount' | 'createdAt'>('volume24h');
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
    score?: {
      total?: number;
      components?: Array<{ name: string; value: number; explanation?: string; weight?: number }>;
      weights?: Record<string, number>;
    };
  } | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0); // Trigger for refetching tokens
  // v8.2.8: Swap processing state for loading animation
  const [isSwapping, setIsSwapping] = useState(false);
  const [swapPhase, setSwapPhase] = useState<'verifying' | 'executing' | 'confirming' | 'idle'>('idle');

  // v2.3.7-beta: Ref to track when a swap just completed - prevents SSE from overwriting correct local state
  const swapJustCompletedRef = useRef(false);

  // v2.9.6-beta: Ref to always have the latest tokens for swap balance calculation
  // This avoids stale closure issues where the callback captures old token values
  const tokensRef = useRef<Token[]>([]);

  // v8.2.8: XLIST Crowdfunding campaigns
  const [xlistCampaigns, setXlistCampaigns] = useState<any[]>([]);
  const [selectedCampaign, setSelectedCampaign] = useState<any | null>(null);

  // 💰 v2.4.8-beta: DCA (Dollar Cost Averaging) State
  const [showDcaModal, setShowDcaModal] = useState(false);
  const [dcaInterval, setDcaInterval] = useState<'hourly' | 'daily' | 'weekly' | 'monthly'>('daily');
  const [dcaAmount, setDcaAmount] = useState('');
  const [dcaMaxExecutions, setDcaMaxExecutions] = useState<string>(''); // Empty = unlimited
  const [dcaOrders, setDcaOrders] = useState<any[]>([]);
  const [loadingDca, setLoadingDca] = useState(false);
  const [showDcaOrdersPanel, setShowDcaOrdersPanel] = useState(false);

  // 💰 v2.4.9-beta: DCA Orders Management Functions
  const fetchDcaOrders = useCallback(async () => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/dca/orders/${walletAddr}`);
      if (response.ok) {
        const data = await response.json();
        setDcaOrders(data.orders || []);
        console.log('💰 [DCA] Loaded orders:', data.orders?.length || 0);
      }
    } catch (error) {
      console.error('Failed to fetch DCA orders:', error);
    }
  }, []);

  const pauseDcaOrder = async (orderId: string) => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/dca/orders/${walletAddr}/${orderId}/pause`, {
        method: 'PUT',
      });
      if (response.ok) {
        fetchDcaOrders();
        alert('DCA order paused');
      }
    } catch (error) {
      console.error('Failed to pause DCA order:', error);
      alert('Failed to pause DCA order');
    }
  };

  const resumeDcaOrder = async (orderId: string) => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/dca/orders/${walletAddr}/${orderId}/resume`, {
        method: 'PUT',
      });
      if (response.ok) {
        fetchDcaOrders();
        alert('DCA order resumed');
      }
    } catch (error) {
      console.error('Failed to resume DCA order:', error);
      alert('Failed to resume DCA order');
    }
  };

  const cancelDcaOrder = async (orderId: string) => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    if (!confirm('Are you sure you want to cancel this DCA order?')) return;

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/dca/orders/${walletAddr}/${orderId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        fetchDcaOrders();
        alert('DCA order cancelled');
      }
    } catch (error) {
      console.error('Failed to cancel DCA order:', error);
      alert('Failed to cancel DCA order');
    }
  };

  // 📊 v2.5.0-beta: Perpetual Futures Trading State
  const [dexMode, setDexMode] = useState<'spot' | 'perpetual'>('spot');
  // v4.1.0: Collapsible UI sections
  const [isSwapCollapsed, setIsSwapCollapsed] = useState(true);
  const [isMenuCollapsed, setIsMenuCollapsed] = useState(false);
  const [isAnalyticsCollapsed, setIsAnalyticsCollapsed] = useState(true);
  const [perpSide, setPerpSide] = useState<'long' | 'short'>('long');
  const [perpLeverage, setPerpLeverage] = useState(2);
  const [perpSize, setPerpSize] = useState('');
  const [perpCollateral, setPerpCollateral] = useState('');
  const [perpPositions, setPerpPositions] = useState<any[]>([]);
  const [perpMarket, setPerpMarket] = useState<any>(null);
  const [loadingPerp, setLoadingPerp] = useState(false);
  const [perpError, setPerpError] = useState<string | null>(null);

  // 📈 v2.6.0-beta: Order Book & Limit Orders State
  const [perpOrderType, setPerpOrderType] = useState<'market' | 'limit'>('market');
  const [limitPrice, setLimitPrice] = useState('');
  const [orderBook, setOrderBook] = useState<{
    bids: Array<{ price: number; size: number; order_count: number }>;
    asks: Array<{ price: number; size: number; order_count: number }>;
    best_bid: number | null;
    best_ask: number | null;
    spread: number | null;
  } | null>(null);
  const [limitOrders, setLimitOrders] = useState<any[]>([]);
  const [timeInForce, setTimeInForce] = useState<'gtc' | 'ioc' | 'fok' | 'post_only'>('gtc');

  // 💹 v10.8.5: Spot DEX limit orders (price-triggered AMM swaps)
  const [spotMode, setSpotMode] = useState<'market' | 'limit'>('market');
  const [limitAmount, setLimitAmount] = useState('');
  const [limitTriggerPrice, setLimitTriggerPrice] = useState('');
  const [limitDirection, setLimitDirection] = useState<'below' | 'above'>('below');
  const [limitExpiry, setLimitExpiry] = useState<'gtc' | '1h' | '24h' | '7d'>('gtc');
  const [limitSlippage, setLimitSlippage] = useState('3');
  const [dexLimitOrders, setDexLimitOrders] = useState<any[]>([]);
  const [isPlacingLimitOrder, setIsPlacingLimitOrder] = useState(false);
  const [limitOrderMsg, setLimitOrderMsg] = useState<{type: 'error'|'success', text: string} | null>(null);
  const [showLimitOrdersPanel, setShowLimitOrdersPanel] = useState(false);

  // 📝 v2.7.8-beta: Position Editing State
  const [editingPosition, setEditingPosition] = useState<any | null>(null);
  const [editMode, setEditMode] = useState<'addMargin' | 'removeMargin' | 'adjustLeverage' | null>(null);
  const [editAmount, setEditAmount] = useState('');
  const [newLeverage, setNewLeverage] = useState(2);

  // Fetch perpetual market data
  const fetchPerpMarket = useCallback(async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/markets/QUG-PERP`);
      if (response.ok) {
        const data = await response.json();
        // v2.6.1-beta: Extract market from response (API returns { success, market })
        if (data.success && data.market) {
          setPerpMarket(data.market);
        } else {
          setPerpMarket(data);
        }
      }
    } catch (error) {
      console.error('Failed to fetch perp market:', error);
    }
  }, []);

  // Fetch user's perpetual positions
  const fetchPerpPositions = useCallback(async () => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/positions/${walletAddr}`);
      if (response.ok) {
        const data = await response.json();
        setPerpPositions(data.positions || []);
      }
    } catch (error) {
      console.error('Failed to fetch perp positions:', error);
    }
  }, []);

  // Open perpetual position
  const openPerpPosition = async () => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) {
      setPerpError('Please connect wallet first');
      return;
    }

    if (!perpSize || !perpCollateral) {
      setPerpError('Please enter size and collateral');
      return;
    }

    setLoadingPerp(true);
    setPerpError(null);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/positions/open`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddr,
          market: 'QUG-PERP',
          side: perpSide,
          size: Math.floor(parseFloat(perpSize) * 1e24),
          collateral: Math.floor(parseFloat(perpCollateral) * 1e24),
          leverage: perpLeverage,
        }),
      });

      const data = await response.json();
      if (data.success) {
        setPerpSize('');
        setPerpCollateral('');
        fetchPerpPositions();
        alert(`Position opened at ${(data.entry_price / 1e24).toFixed(4)} QUGUSD`);
      } else {
        setPerpError(data.message || 'Failed to open position');
      }
    } catch (error) {
      console.error('Failed to open position:', error);
      setPerpError('Failed to open position');
    } finally {
      setLoadingPerp(false);
    }
  };

  // Close perpetual position
  const closePerpPosition = async (positionId: string) => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    setLoadingPerp(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/positions/${positionId}/close`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddr,
        }),
      });

      const data = await response.json();
      if (data.success) {
        fetchPerpPositions();
        alert(`Position closed. PnL: ${(data.realized_pnl / 1e24).toFixed(4)} QUGUSD`);
      } else {
        alert(data.message || 'Failed to close position');
      }
    } catch (error) {
      console.error('Failed to close position:', error);
      alert('Failed to close position');
    } finally {
      setLoadingPerp(false);
    }
  };

  // 📝 v2.7.8-beta: Add margin to position
  const addMarginToPosition = async (positionId: string, amount: number) => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    setLoadingPerp(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/positions/${positionId}/margin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddr,
          amount: Math.floor(amount * 1e24),
        }),
      });

      const data = await response.json();
      if (data.success) {
        fetchPerpPositions();
        setEditingPosition(null);
        setEditMode(null);
        setEditAmount('');
        alert(`Added ${amount.toFixed(4)} QUGUSD margin. New liquidation price: $${(data.new_liquidation_price / 1e24).toFixed(4)}`);
      } else {
        alert(data.message || 'Failed to add margin');
      }
    } catch (error) {
      console.error('Failed to add margin:', error);
      alert('Failed to add margin');
    } finally {
      setLoadingPerp(false);
    }
  };

  // 📝 v2.7.8-beta: Remove margin from position
  const removeMarginFromPosition = async (positionId: string, amount: number) => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    setLoadingPerp(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/positions/${positionId}/margin/remove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddr,
          amount: Math.floor(amount * 1e24),
        }),
      });

      const data = await response.json();
      if (data.success) {
        fetchPerpPositions();
        setEditingPosition(null);
        setEditMode(null);
        setEditAmount('');
        alert(`Removed ${(data.returned_amount / 1e24).toFixed(4)} QUGUSD margin. New liquidation price: $${(data.new_liquidation_price / 1e24).toFixed(4)}`);
      } else {
        alert(data.message || 'Failed to remove margin');
      }
    } catch (error) {
      console.error('Failed to remove margin:', error);
      alert('Failed to remove margin');
    } finally {
      setLoadingPerp(false);
    }
  };

  // 📝 v2.7.8-beta: Adjust position leverage
  const adjustPositionLeverage = async (positionId: string, newLev: number) => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    setLoadingPerp(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/positions/${positionId}/leverage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddr,
          new_leverage: newLev,
        }),
      });

      const data = await response.json();
      if (data.success) {
        fetchPerpPositions();
        setEditingPosition(null);
        setEditMode(null);
        const collateralMsg = data.collateral_change > 0
          ? `Required additional ${(data.collateral_change / 1e24).toFixed(4)} QUGUSD`
          : data.collateral_change < 0
            ? `Returned ${(Math.abs(data.collateral_change) / 1e24).toFixed(4)} QUGUSD`
            : '';
        alert(`Leverage changed from ${data.old_leverage}x to ${data.new_leverage}x. ${collateralMsg}`);
      } else {
        alert(data.message || 'Failed to adjust leverage');
      }
    } catch (error) {
      console.error('Failed to adjust leverage:', error);
      alert('Failed to adjust leverage');
    } finally {
      setLoadingPerp(false);
    }
  };

  // 📈 Fetch order book depth
  const fetchOrderBook = useCallback(async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/orderbook/QUG-PERP`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setOrderBook(data);
        }
      }
    } catch (error) {
      console.error('Failed to fetch order book:', error);
    }
  }, []);

  // 📈 Fetch user's limit orders
  const fetchLimitOrders = useCallback(async () => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/limit-orders/${walletAddr}`);
      if (response.ok) {
        const data = await response.json();
        setLimitOrders(data.orders || []);
      }
    } catch (error) {
      console.error('Failed to fetch limit orders:', error);
    }
  }, []);

  // 📈 Place limit order
  const placeLimitOrder = async () => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) {
      setPerpError('Please connect wallet first');
      return;
    }

    if (!perpSize || !limitPrice) {
      setPerpError('Please enter size and price');
      return;
    }

    setLoadingPerp(true);
    setPerpError(null);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/limit-orders`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddr,
          market: 'QUG-PERP',
          side: perpSide === 'long' ? 'buy' : 'sell',
          price: Math.floor(parseFloat(limitPrice) * 1e24),
          size: Math.floor(parseFloat(perpSize) * 1e24),
          leverage: perpLeverage,
          time_in_force: timeInForce,
          reduce_only: false,
          post_only: timeInForce === 'post_only',
        }),
      });

      const data = await response.json();
      if (data.success) {
        setPerpSize('');
        setLimitPrice('');
        fetchLimitOrders();
        fetchOrderBook();
        fetchPerpPositions();
        alert(`Order ${data.status}: ${data.message}`);
      } else {
        setPerpError(data.message || 'Failed to place order');
      }
    } catch (error) {
      console.error('Failed to place limit order:', error);
      setPerpError('Failed to place limit order');
    } finally {
      setLoadingPerp(false);
    }
  };

  // 📈 Cancel limit order
  const cancelLimitOrder = async (orderId: string) => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/limit-orders/${walletAddr}/${orderId}`, {
        method: 'DELETE',
      });

      const data = await response.json();
      if (data.success) {
        fetchLimitOrders();
        fetchOrderBook();
        alert('Order cancelled');
      } else {
        alert(data.message || 'Failed to cancel order');
      }
    } catch (error) {
      console.error('Failed to cancel order:', error);
      alert('Failed to cancel order');
    }
  };

  // 💹 Fetch user's DEX (spot) limit orders
  const fetchDexLimitOrders = useCallback(async () => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;
    try {
      const resp = await fetch(`/api/v1/dex/limit-orders/${walletAddr}`);
      if (resp.ok) {
        const data = await resp.json();
        if (data.success) setDexLimitOrders(data.orders || []);
      }
    } catch { /* silent */ }
  }, []);

  // 💹 Place a spot DEX limit order
  const placeDexLimitOrder = async () => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr || !limitAmount || !limitTriggerPrice) return;
    setIsPlacingLimitOrder(true);
    setLimitOrderMsg(null);
    try {
      const intPart = BigInt(Math.floor(Math.abs(parseFloat(limitAmount))));
      const fracPart = BigInt(Math.round((Math.abs(parseFloat(limitAmount)) % 1) * 1e6));
      const amountU128 = (intPart * BigInt('1000000000000000000000000') + fracPart * BigInt('1000000000000000000')).toString();
      const expiryMs =
        limitExpiry === '1h'  ? Date.now() + 3_600_000 :
        limitExpiry === '24h' ? Date.now() + 86_400_000 :
        limitExpiry === '7d'  ? Date.now() + 604_800_000 : null;
      const resp = await fetch('/api/v1/dex/limit-orders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddr,
          from_token: swapFrom,
          to_token: swapTo,
          amount: amountU128,
          trigger_price: parseFloat(limitTriggerPrice),
          price_token: swapFrom,
          direction: limitDirection,
          max_slippage: parseFloat(limitSlippage) / 100,
          expiry: expiryMs,
        }),
      });
      const data = await resp.json();
      if (data.success) {
        setLimitOrderMsg({ type: 'success', text: `Order placed — ID …${(data.order_id || '').slice(-8)}` });
        setLimitAmount('');
        setLimitTriggerPrice('');
        setShowLimitOrdersPanel(true);
        fetchDexLimitOrders();
      } else {
        setLimitOrderMsg({ type: 'error', text: data.message || 'Failed to place order' });
      }
    } catch {
      setLimitOrderMsg({ type: 'error', text: 'Network error — please try again' });
    } finally {
      setIsPlacingLimitOrder(false);
    }
  };

  // 💹 Cancel a spot DEX limit order (best-effort; server validates ownership)
  const cancelDexLimitOrder = async (orderId: string) => {
    const walletAddr = localStorage.getItem('walletAddress');
    if (!walletAddr) return;
    const tsMs = Date.now();
    try {
      const resp = await fetch(`/api/v1/dex/limit-orders/${walletAddr}/${orderId}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timestamp_ms: tsMs, signature: '' }),
      });
      const data = await resp.json();
      if (data.success) {
        setDexLimitOrders(prev => prev.filter(o => o.id !== orderId));
      }
    } catch { /* silent */ }
  };

  // Load perp data when switching to perpetual mode
  useEffect(() => {
    if (dexMode === 'perpetual') {
      fetchPerpMarket();
      fetchPerpPositions();
      fetchOrderBook();
      fetchLimitOrders();
      // Refresh every 2 seconds for order book, 5 seconds for rest
      const bookInterval = setInterval(fetchOrderBook, 2000);
      const dataInterval = setInterval(() => {
        fetchPerpMarket();
        fetchPerpPositions();
        fetchLimitOrders();
      }, 5000);
      return () => {
        clearInterval(bookInterval);
        clearInterval(dataInterval);
      };
    }
  }, [dexMode, fetchPerpMarket, fetchPerpPositions, fetchOrderBook, fetchLimitOrders]);

  // Load spot limit orders on mount and every 30s
  useEffect(() => {
    fetchDexLimitOrders();
    const iv = setInterval(fetchDexLimitOrders, 30_000);
    return () => clearInterval(iv);
  }, [fetchDexLimitOrders]);

  // v8.2.8: Fetch XLIST crowdfunding campaigns and inject as special DEX tokens
  useEffect(() => {
    const fetchCampaigns = async () => {
      try {
        const resp = await fetch('/api/v1/listing/campaigns');
        if (resp.ok) {
          const data = await resp.json();
          const campaigns = data.campaigns || data || [];
          setXlistCampaigns(campaigns);
          // Inject campaigns as special XLIST tokens into the token list
          if (campaigns.length > 0) {
            const xlistTokens: Token[] = campaigns.map((c: any) => ({
              id: `xlist-${c.campaign_id}`,
              symbol: `XLIST`,
              name: `${c.exchange_name} Listing Fund`,
              balance: 0,
              price: c.raised_usd / Math.max(c.contributor_count, 1),
              change1h: 0,
              change24h: c.contributor_count > 0 ? ((c.raised_usd / c.target_usd) * 100) : 0,
              change7d: 0,
              volume24h: c.raised_usd,
              liquidity: c.target_usd,
              icon: c.exchange_logo || '\u{1F3E6}',
              marketCap: c.raised_usd,
              totalSupply: c.target_usd,
              circulatingSupply: c.raised_usd,
              holders: c.contributor_count,
              features: { reflection: false, autoLiquidity: false, buybackAndBurn: false, antiWhale: false, quantumSecured: false },
              fees: { buy: 0, sell: 0, transfer: 0 },
              description: c.description || `Community crowdfunding to list QUG on ${c.exchange_name}`,
              isCrowdfund: true,
              campaignData: c,
            }));
            setTokens(prev => {
              // Remove old XLIST tokens, then prepend new ones
              const nonXlist = prev.filter(t => !t.id.startsWith('xlist-'));
              return [...xlistTokens, ...nonXlist];
            });
          }
        }
      } catch (e) {
        console.log('[DexScreen] No XLIST campaigns available:', e);
      }
    };
    fetchCampaigns();
    const interval = setInterval(fetchCampaigns, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  // DEX Settings Modal State
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const [dexSettings, setDexSettings] = useState<DexSettings>(() => {
    // Load from localStorage on init
    const saved = localStorage.getItem('dexSettings');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch {
        // Fall back to defaults
      }
    }
    return {
      slippageTolerance: 0.5,
      transactionDeadline: 20,
      expertMode: false,
      multihops: true,
      gasPreference: 'medium' as const,
    };
  });
  const [customSlippage, setCustomSlippage] = useState('');

  // AI Token Analyzer State
  const [aiAnalyzingToken, setAiAnalyzingToken] = useState<Token | null>(null);
  const [aiAnalysisLoading, setAiAnalysisLoading] = useState(false);
  const [aiAnalysisResult, setAiAnalysisResult] = useState<{
    score: number;
    verdict: 'GOOD' | 'CAUTION' | 'RISKY';
    summary: string;
    metrics: {
      liquidity: { value: string; status: 'good' | 'warn' | 'bad' };
      activity: { value: string; status: 'good' | 'warn' | 'bad' };
      holders: { value: string; status: 'good' | 'warn' | 'bad' };
      priceStability: { value: string; status: 'good' | 'warn' | 'bad' };
      volume: { value: string; status: 'good' | 'warn' | 'bad' };
    };
  } | null>(null);

  // Save DEX settings to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('dexSettings', JSON.stringify(dexSettings));
  }, [dexSettings]);

  // v2.9.6-beta: Keep tokensRef in sync with tokens state
  // This ensures swap handlers always have access to the latest token balances
  useEffect(() => {
    tokensRef.current = tokens;
  }, [tokens]);

  // 💰 v2.4.9-beta: Load DCA orders on mount
  useEffect(() => {
    fetchDcaOrders();
  }, [fetchDcaOrders]);

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

    // v1.4.10-beta: Listen for custom token balance updates via SSE
    const handleTokenBalanceUpdate = (event: CustomEvent) => {
      if (!mounted) return;
      const { tokenAddress, tokenSymbol, newBalance, reason } = event.detail;
      console.log('🪙 [DEX] Token balance updated via SSE:', { tokenSymbol, tokenAddress, newBalance, reason });

      // Update the token balance in the tokens state
      setTokens(prev => prev.map(token => {
        // Match by token address (with or without qnk prefix)
        const tokenId = token.id?.startsWith('qnk') ? token.id.substring(3) : (token.id || '');
        const eventAddr = tokenAddress?.startsWith('qnk') ? tokenAddress.substring(3) : (tokenAddress || '');

        if (tokenId === eventAddr) {
          const numericBalance = typeof newBalance === 'string' ? parseFloat(newBalance) || 0 : (Number(newBalance) || 0);
          console.log(`✅ [DEX] Updated ${token.symbol} balance: ${token.balance} → ${numericBalance}`);
          return { ...token, balance: numericBalance };
        }
        return token;
      }));
    };

    window.addEventListener('token-balance-updated', handleTokenBalanceUpdate as EventListener);

    // v5.0.1: Listen for wallet-balance-updated events (dispatched by Dashboard/TopBar on SSE)
    // This ensures QUG/QUGUSD balance updates from SSE are reflected in DEX instantly,
    // even if the REST API returns stale data
    const handleWalletBalanceUpdated = (event: CustomEvent) => {
      if (!mounted) return;
      const { symbol, balance, reason } = event.detail || {};
      if (!symbol || balance === undefined) return;

      const numericBalance = typeof balance === 'number' ? balance :
        typeof balance === 'string' ? parseFloat(balance) || 0 : 0;

      // Only process valid positive balances
      if (numericBalance <= 0) return;

      const displayBalance = numericBalance > 1e15 ? numericBalance / 1e24 : numericBalance;
      console.log(`💰 [DEX] wallet-balance-updated: ${symbol} = ${displayBalance} (reason: ${reason})`);

      setTokens(prev => prev.map(token => {
        if (token.symbol === symbol) {
          return { ...token, balance: displayBalance };
        }
        return token;
      }));

      // Cache QUG balance
      if (symbol === 'QUG') {
        safeCacheBalance(displayBalance);
      }
    };
    window.addEventListener('wallet-balance-updated', handleWalletBalanceUpdated as EventListener);

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
      // v2.4.7: Handle both token_symbol (from backend) and token_id for compatibility
      // v2.9.22-beta: Updated handler to apply change_1h and change_7d from SSE
      // v2.9.25-beta: Added token_address matching for when symbols aren't resolved
      eventSource.addEventListener('token_price_update', (event) => {
        if (!mounted) return;
        try {
          const parsed = JSON.parse(event.data);
          console.log('📈 Received token price update:', parsed);

          // Extract data from wrapper
          const data = parsed.data || parsed;
          const tokenSymbol = data.token_symbol || data.token_id || '';
          const tokenAddress = data.token_address || '';  // v2.9.25-beta: Address for fallback matching

          // Update token in list - match by symbol (case-insensitive), id, or address
          setTokens(prev => prev.map(token => {
            // v2.9.25-beta: Extended matching to include address comparison
            const symbolMatch = token.symbol.toUpperCase() === tokenSymbol.toUpperCase();
            const idMatch = token.id === tokenSymbol || token.id === data.token_id;
            const addressMatch = tokenAddress && (
              token.id === tokenAddress ||
              token.id.toLowerCase() === tokenAddress.toLowerCase()
            );
            const matches = symbolMatch || idMatch || addressMatch;

            if (matches) {
              console.log(`✅ Updating ${token.symbol}: price=${data.price}, 1h=${data.change_1h}, 24h=${data.change_24h}, 7d=${data.change_7d}, vol24h=${data.volume_24h} (matched by: ${symbolMatch ? 'symbol' : idMatch ? 'id' : 'address'})`);
              return {
                ...token,
                price: data.price !== undefined ? data.price : token.price,
                change1h: data.change_1h !== undefined ? data.change_1h : token.change1h,
                change24h: data.change_24h !== undefined ? data.change_24h : token.change24h,
                change7d: data.change_7d !== undefined ? data.change_7d : token.change7d,
                volume24h: data.volume_24h !== undefined ? data.volume_24h : token.volume24h
              };
            }
            return token;
          }));
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
      console.error('Failed to set up SSE for Nitro boosts:', error);
    }

    // v2.9.25-beta: ALSO listen for token-price-updated CustomEvent from App.tsx
    // This is a backup path since App.tsx's authenticated SSE connection is more reliable
    const handleTokenPriceUpdated = (event: CustomEvent) => {
      if (!mounted) return;
      const data = event.detail;
      console.log('📈 [DexScreen] Received token-price-updated from App.tsx:', data);

      const tokenSymbol = data.token_symbol || '';
      const tokenAddress = data.token_address || '';

      setTokens(prev => prev.map(token => {
        const symbolMatch = token.symbol.toUpperCase() === tokenSymbol.toUpperCase();
        const idMatch = token.id === tokenSymbol;
        const addressMatch = tokenAddress && (
          token.id === tokenAddress ||
          token.id.toLowerCase() === tokenAddress.toLowerCase()
        );
        const matches = symbolMatch || idMatch || addressMatch;

        if (matches) {
          console.log(`✅ [DexScreen v2.9.25] Updating ${token.symbol} from App.tsx forward: price=${data.price}, 1h=${data.change_1h}%, 24h=${data.change_24h}%, 7d=${data.change_7d}%`);
          return {
            ...token,
            price: data.price !== undefined ? data.price : token.price,
            change1h: data.change_1h !== undefined ? data.change_1h : token.change1h,
            change24h: data.change_24h !== undefined ? data.change_24h : token.change24h,
            change7d: data.change_7d !== undefined ? data.change_7d : token.change7d,
            volume24h: data.volume_24h !== undefined ? data.volume_24h : token.volume24h
          };
        }
        return token;
      }));
    };

    window.addEventListener('token-price-updated', handleTokenPriceUpdated as EventListener);

    return () => {
      mounted = false;
      if (eventSource) {
        eventSource.close();
      }
      window.removeEventListener('token-price-updated', handleTokenPriceUpdated as EventListener);
      window.removeEventListener('token-balance-updated', handleTokenBalanceUpdate as EventListener);
      window.removeEventListener('wallet-balance-updated', handleWalletBalanceUpdated as EventListener);
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
        let allTokenBalances: Record<string, any> = {}; // v4.3.0: Store all token balances for index funds
        if (walletAddress) {
          console.log('🔍 [DEX] Fetching multi-token balance for wallet:', walletAddress);
          try {
            const multiTokenResponse = await qnkAPI.getMultiTokenBalance(true);
            console.log('📊 [DEX] Multi-token balance API response:', multiTokenResponse);
            if (multiTokenResponse.success && multiTokenResponse.data) {
              // v4.3.0: Save all token balances for index fund lookup
              if (multiTokenResponse.data.tokens) {
                allTokenBalances = multiTokenResponse.data.tokens;
              }
              // Extract QUG balance (already in human-readable form)
              if (multiTokenResponse.data.tokens && multiTokenResponse.data.tokens.QUG) {
                nativeQugBalance = parseFloat(multiTokenResponse.data.tokens.QUG.balance) || 0;
                console.log('✅ [DEX] Native QUG balance fetched:', nativeQugBalance, 'QUG');
              }
              // Extract QUGUSD balance (already in human-readable form)
              if (multiTokenResponse.data.tokens && multiTokenResponse.data.tokens.QUGUSD) {
                qugusdBalance = parseFloat(multiTokenResponse.data.tokens.QUGUSD.balance) || 0;
                console.log('✅ [DEX] QUGUSD balance fetched:', qugusdBalance, 'QUGUSD');
              }
              // v4.0.9: Also check top-level qug_balance field (some API versions put it here)
              if (nativeQugBalance === 0 && multiTokenResponse.data.qug_balance !== undefined) {
                nativeQugBalance = parseFloat(multiTokenResponse.data.qug_balance) || 0;
                console.log('✅ [DEX] QUG balance from top-level field:', nativeQugBalance);
              }
            }

            // v4.0.9: If QUG balance is still 0 after multi-token API, try dedicated wallet balance API
            // QUG is a native coin (wallet_balances), not a token (token_balances), so multi-token
            // may succeed but NOT include QUG. Always fall back to getWalletBalance for QUG.
            if (nativeQugBalance === 0) {
              console.log('🔄 [DEX] QUG balance still 0, trying getWalletBalance fallback');
              try {
                const fallbackResponse = await qnkAPI.getWalletBalance(walletAddress, true);
                if (fallbackResponse.success && fallbackResponse.data) {
                  nativeQugBalance = fallbackResponse.data.balance_qnk || fallbackResponse.data.balance || 0;
                  // Handle raw base units (> 1e15 means base units with 24 decimals)
                  if (nativeQugBalance > 1e15) {
                    nativeQugBalance = nativeQugBalance / 1e24;
                  }
                  console.log('✅ [DEX] Fallback QUG balance:', nativeQugBalance, 'QUG');
                }
              } catch (fallbackError) {
                console.warn('⚠️ [DEX] Fallback getWalletBalance failed:', fallbackError);
              }
            }

            // v4.0.9: Final fallback - use cached balance from localStorage
            if (nativeQugBalance === 0) {
              const cachedBalance = localStorage.getItem('cachedBalance');
              if (cachedBalance) {
                const cached = parseFloat(cachedBalance);
                if (cached > 0) {
                  nativeQugBalance = cached > 1e15 ? cached / 1e24 : cached;
                  console.log('💰 [DEX] Using cached QUG balance:', nativeQugBalance);
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
              const t0 = pool.token0 || '';
              if (t0 === 'QUG' || t0.toUpperCase() === 'QUG') {
                token0Key = 'native-qug';
              } else if (t0 === 'QUGUSD' || t0.toUpperCase() === 'QUGUSD') {
                token0Key = 'qugusd-stable';
              } else if (t0.startsWith('qnk') || t0.startsWith('0x')) {
                // Already an address - use directly (v1.0.49-beta: pools now use canonical addresses)
                token0Key = t0;
              } else {
                // It's a symbol, resolve to address
                token0Key = symbolToAddress.get(t0.toUpperCase()) || t0;
                if (token0Key !== t0) {
                  console.log(`🔍 Resolved pool.token0 "${t0}" => "${token0Key}"`);
                }
              }
              // v4.1.2: Pool API returns raw 24-decimal reserves - divide by 1e24 for display
              const reserve0Display = (parseFloat(pool.reserve0) || 0) / 1e24;
              poolsByToken.set(token0Key, (poolsByToken.get(token0Key) || 0) + reserve0Display);

              // Resolve token1 to address key
              let token1Key: string;
              const t1 = pool.token1 || '';
              if (t1 === 'QUG' || t1.toUpperCase() === 'QUG') {
                token1Key = 'native-qug';
              } else if (t1 === 'QUGUSD' || t1.toUpperCase() === 'QUGUSD') {
                token1Key = 'qugusd-stable';
              } else if (t1.startsWith('qnk') || t1.startsWith('0x')) {
                // Already an address - use directly (v1.0.49-beta: pools now use canonical addresses)
                token1Key = t1;
              } else {
                // It's a symbol, resolve to address
                token1Key = symbolToAddress.get(t1.toUpperCase()) || t1;
                if (token1Key !== t1) {
                  console.log(`🔍 Resolved pool.token1 "${t1}" => "${token1Key}"`);
                }
              }
              // v4.1.2: Pool API returns raw 24-decimal reserves - divide by 1e24 for display
              const reserve1Display = (parseFloat(pool.reserve1) || 0) / 1e24;
              poolsByToken.set(token1Key, (poolsByToken.get(token1Key) || 0) + reserve1Display);
            });

            console.log('✅ Calculated liquidity from pools (display units):', Object.fromEntries(poolsByToken));
          }
        } catch (error) {
          console.error('Failed to fetch liquidity pools:', error);
        }

        // Fetch real price from oracle API with all time periods
        // v8.0.1: Default to 0 — only show a price if the oracle returns one
        let qugPrice = 0;
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

        // v2.3.8-beta: Fetch REAL network supply stats (circulating supply + holders count)
        let qugCirculatingSupply = 0; // v8.0.1: Default to 0 (no blocks = no circulating supply)
        let qugTotalSupply = 21000000; // Max supply
        let qugHolders = 0; // v8.0.1: Default to 0 until API returns real data
        try {
          const supplyResponse = await qnkAPI.getNetworkSupply();
          if (supplyResponse.success && supplyResponse.data) {
            qugCirculatingSupply = supplyResponse.data.total_mined || qugCirculatingSupply;
            qugTotalSupply = supplyResponse.data.max_supply || qugTotalSupply;
            qugHolders = supplyResponse.data.holders || qugHolders;
            console.log('✅ Fetched REAL network supply:', { circulating: qugCirculatingSupply, total: qugTotalSupply, holders: qugHolders });
          }
        } catch (error) {
          console.error('Failed to fetch network supply:', error);
        }

        // v2.3.8-beta: Fetch REAL QUGUSD vault stats (circulating supply + holders count)
        let qugusdCirculatingSupply = 0; // Fallback: no QUGUSD minted yet
        let qugusdTotalSupply = 0; // Total minted = circulating for stablecoin
        let qugusdHolders = 0; // Fallback
        let qugusdCollateralRatio = 150; // Default collateralization %
        try {
          const vaultResponse = await qnkAPI.getVaultStats();
          if (vaultResponse.success && vaultResponse.data) {
            // Convert from base units (1e8) to human-readable
            qugusdCirculatingSupply = (vaultResponse.data.total_qugusd_minted || 0) / 1e8;
            qugusdTotalSupply = qugusdCirculatingSupply; // Stablecoin: minted = supply
            qugusdHolders = vaultResponse.data.num_positions || 0;
            qugusdCollateralRatio = (vaultResponse.data.global_collateral_ratio || 1.5) * 100;
            console.log('✅ Fetched REAL vault stats:', {
              circulating: qugusdCirculatingSupply,
              holders: qugusdHolders,
              collateralRatio: qugusdCollateralRatio.toFixed(2) + '%'
            });
          }
        } catch (error) {
          console.error('Failed to fetch vault stats:', error);
        }

        // Fetch USD balance from payment API
        if (walletAddress) {
          try {
            const usdAbortController = new AbortController();
            const usdFetchTimeout = setTimeout(() => usdAbortController.abort(), 8000);
            const usdResponse = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/balance`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ wallet_address: walletAddress }),
              signal: usdAbortController.signal,
            });
            clearTimeout(usdFetchTimeout);

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
            // v2.3.7-beta: Convert liquidity to USD (token amount * price)
            liquidity: (poolsByToken.get('native-qug') || 0) * qugPrice,
            // v2.3.8-beta: Use REAL supply data from /api/v1/network/supply
            marketCap: qugCirculatingSupply * qugPrice,
            fullyDilutedMarketCap: qugTotalSupply * qugPrice, // FDV = total supply * price
            totalSupply: qugTotalSupply,
            circulatingSupply: qugCirculatingSupply,
            holders: qugHolders,
            icon: 'qug-logo',
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
            description: 'QUG (Quillon) is the native quantum-enhanced token powering the Quillon blockchain. Built on DAG-BFT consensus with post-quantum cryptographic security (Dilithium5 + Kyber1024). 21M max supply, 4-year halving, ~1 second blocks. No fees. No pre-mine.',
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
            // v2.3.7-beta: QUGUSD liquidity in USD (amount * $1 price)
            liquidity: (poolsByToken.get('qugusd-stable') || 0) * qugusdPrice,
            // v2.3.8-beta: Use REAL vault stats from /api/v1/stablecoin/vault/stats
            marketCap: qugusdCirculatingSupply * qugusdPrice,
            fullyDilutedMarketCap: qugusdCirculatingSupply * qugusdPrice, // For stablecoin, FDV = current supply
            totalSupply: qugusdTotalSupply,
            circulatingSupply: qugusdCirculatingSupply,
            holders: qugusdHolders,
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
            fullyDilutedMarketCap: 0, // N/A for fiat
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
              // v4.0.14: Backend now returns supply in token's own decimal base units (8-decimal)
              // computed from actual token_balances (ground truth), not ambiguous deployment_params
              const decimals = apiToken.decimals || 8;
              const rawSupply = apiToken.total_supply || '0';
              const supplyStr = typeof rawSupply === 'string' ? rawSupply : String(rawSupply);
              let actualSupply = 0;
              try {
                // Handle scientific notation strings like "1e+30" that BigInt can't parse
                const cleanStr = supplyStr.toLowerCase().includes('e')
                  ? BigInt(Math.round(parseFloat(supplyStr))).toString()
                  : supplyStr;
                const supplyBigInt = BigInt(cleanStr);
                const divisorBigInt = BigInt(10) ** BigInt(decimals);
                actualSupply = Number(supplyBigInt / divisorBigInt) + Number(supplyBigInt % divisorBigInt) / Number(divisorBigInt);
              } catch {
                actualSupply = parseFloat(supplyStr) || 0;
                if (actualSupply > 1e15) actualSupply = actualSupply / Math.pow(10, decimals);
              }
              // v4.0.15: Allow arbitrary supply sizes - formatNumber handles display up to 1e30+

              // Fetch balance for this token if wallet is available
              let tokenBalance = 0;
              if (walletAddress) {
                try {
                  console.log(`🔍 [API Token] Fetching balance for ${apiToken.symbol} (address: ${apiToken.address}, wallet: ${walletAddress})`);
                  const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, apiToken.address);
                  console.log(`📊 [API Token] Balance response for ${apiToken.symbol}:`, balanceResponse);

                  if (balanceResponse.success && balanceResponse.data) {
                    // v10.3.6: Token balance decimal conversion
                    // Raw balance from API is in base units (display × 10^decimals)
                    // Divide by 10^decimals to get display value
                    // Use string-based division to avoid BigInt→Number precision loss
                    const rawBalance = balanceResponse.data.balance || '0';
                    const balanceStr = typeof rawBalance === 'string' ? rawBalance : String(rawBalance);
                    try {
                      // Try 1x decimals first (standard: raw = display × 10^decimals)
                      const padded = balanceStr.padStart(decimals + 1, '0');
                      const intPart = padded.slice(0, padded.length - decimals) || '0';
                      const fracPart = padded.slice(padded.length - decimals);
                      tokenBalance = parseFloat(`${intPart}.${fracPart}`);
                      if (isNaN(tokenBalance)) tokenBalance = 0;
                    } catch {
                      tokenBalance = parseFloat(balanceStr) || 0;
                    }
                    console.log(`✅ [API Token] Converted ${apiToken.symbol} balance from ${rawBalance} to ${tokenBalance} (2×${decimals}dec)`);
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

              // v2.3.7-beta: Get real liquidity from pools and convert to USD
              const tokenAmountInPool = poolsByToken.get(apiToken.address) || 0;
              const tokenLiquidity = tokenAmountInPool * customPrice;

              const calculatedMarketCap = customMarketCap || (actualSupply * customPrice);
              const fullyDilutedMC = actualSupply * customPrice;

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
                fullyDilutedMarketCap: fullyDilutedMC,
                totalSupply: actualSupply,
                circulatingSupply: actualSupply,
                holders: customHolders,
                createdAt: apiToken.deployed_at ? apiToken.deployed_at * 1000 : undefined,
                txCount: apiToken.tx_count || 0,
                icon: apiToken.symbol, // TokenIcon component handles symbol-based rendering
                logoUrl: apiToken.logo_url || localStorage.getItem(`token_logo_${apiToken.address}`) || undefined,
                isBridgeToken: apiToken.contract_type === 'Wrapped',
                // v3.6.7-beta: CRITICAL - Include decimals for proper amount calculation in swaps!
                // Without this, custom tokens default to 24 decimals which causes U128 overflow
                decimals: decimals,
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

          // v8.2.8: Show ALL tokens (bridge, custom, listed) so users can add liquidity
          // Previously filtered out tokens without liquidity pools, but that prevented
          // newly listed tokens from appearing in the DEX token selector
          const tokensWithLiquidity = apiTokens;

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

            // Calculate actual supply using token's own decimals
            const decimals = contract.decimals || 8;
            // v4.0.12: deployment_params stores supply in token's OWN decimal base units, NOT 24
            const rawSupply = contract.total_supply || '0';
            const supplyStr = typeof rawSupply === 'string' ? rawSupply : String(rawSupply);
            const supplyBigInt = BigInt(supplyStr);
            const divisorBigInt = BigInt(10) ** BigInt(decimals);
            const actualSupply = Number(supplyBigInt / divisorBigInt) + Number(supplyBigInt % divisorBigInt) / Number(divisorBigInt);

            // Fetch balance for this token if wallet is available
            let tokenBalance = 0;
            if (walletAddress) {
              try {
                const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, contract.address);
                console.log(`📊 Balance response for ${contract.symbol}:`, balanceResponse);
                if (balanceResponse.success && balanceResponse.data) {
                  // v4.1.0: Balance stored as display × 10^(2*decimals) due to double-conversion
                  // Divide by 10^(2*decimals) to get display value matching supply
                  const rawBalance = balanceResponse.data.balance || '0';
                  const balanceStr = typeof rawBalance === 'string' ? rawBalance : String(rawBalance);
                  const balanceBigInt = BigInt(balanceStr);
                  const balDivisor = BigInt(10) ** BigInt(2 * decimals);
                  tokenBalance = Number(balanceBigInt / balDivisor) + Number(balanceBigInt % balDivisor) / Number(balDivisor);
                  console.log(`✅ Converted ${contract.symbol} balance from ${rawBalance} to ${tokenBalance} (2×${decimals}dec)`);
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

            // v2.3.7-beta: Get real liquidity from pools and convert to USD
            const tokenAmountInPool = poolsByToken.get(contract.address) || 0;
            const tokenLiquidity = tokenAmountInPool * customPrice;

            // Calculate market cap if not provided by oracle
            const calculatedMarketCap = customMarketCap || (actualSupply * customPrice);
            // Calculate FDV (Fully Diluted Valuation) = total supply * price
            const fullyDilutedMC = actualSupply * customPrice;

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
              fullyDilutedMarketCap: fullyDilutedMC,
              totalSupply: actualSupply,
              circulatingSupply: actualSupply,
              holders: customHolders,
              createdAt: contract.deployed_at ? contract.deployed_at * 1000 : undefined,
              txCount: contract.tx_count || 0,
              icon: '🪙',
              // v3.6.7-beta: CRITICAL - Include decimals for proper amount calculation in swaps!
              // Without this, custom tokens default to 24 decimals which causes U128 overflow
              decimals: decimals,
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
          // v10.2.2: Case-insensitive + trimmed comparison to catch "DERP" vs "Derp" duplicates
          const existingSymbols = new Set(enrichedTokens.map(t => t.symbol.toUpperCase().trim()));
          const newUserTokens = userTokensWithLiquidity.filter(t => !existingSymbols.has(t.symbol.toUpperCase().trim()));

          console.log(`✅ Adding ${newUserTokens.length} user tokens to Available Tokens (${userTokens.length - newUserTokens.length} filtered: ${userTokens.length - userTokensWithLiquidity.length} no liquidity, ${userTokensWithLiquidity.length - newUserTokens.length} duplicates)`);
          enrichedTokens = [...enrichedTokens, ...newUserTokens];
        }

        // v2.3.8-beta: Index Fund tokens (QNK10, DEFI5) - Coming Soon
        // These are synthetic index products that track QUG ecosystem tokens
        // Currently showing projected NAV based on live QUG price
        // TODO: Integrate with q-index-fund crate when deployed on-chain
        const qnk10NavPerShare = qugPrice * 3; // Projected NAV: ~3x QUG price
        const defi5NavPerShare = qugPrice * 2; // Projected NAV: ~2x QUG price

        // v4.3.0: Read index fund balances from multi-token API response
        const qnk10Balance = parseFloat(allTokenBalances?.QNK10?.balance || '0');
        const defi5Balance = parseFloat(allTokenBalances?.DEFI5?.balance || '0');
        console.log(`🏦 [DEX v4.3.0] Index fund balances: QNK10=${qnk10Balance}, DEFI5=${defi5Balance}`);

        const indexFundTokens: Token[] = [
          {
            id: 'index-fund-qnk10',
            symbol: 'QNK10',
            name: 'QNK Top 10 Index',
            balance: qnk10Balance,
            price: qnk10NavPerShare,
            change1h: qugChange1h,
            change24h: qugChange24h,
            change7d: qugChange7d,
            volume24h: 0, // v2.3.8-beta: Not yet trading
            // v2.3.7-beta: Convert QUG liquidity to USD
            liquidity: 0, // v2.3.8-beta: Not yet deployed
            icon: 'index-fund',
            marketCap: 0, // v2.3.8-beta: Not yet deployed
            fullyDilutedMarketCap: 0, // v2.3.8-beta: Not yet deployed
            totalSupply: 0, // v2.3.8-beta: Not yet minted
            circulatingSupply: 0, // v2.3.8-beta: Not yet minted
            holders: 0, // v2.3.8-beta: Not yet deployed
            features: {
              reflection: false,
              autoLiquidity: false,
              buybackAndBurn: false,
              antiWhale: true,
              quantumSecured: true,
            },
            fees: {
              buy: 0.1,
              sell: 0.1,
              transfer: 0,
            },
            description: 'Market-cap weighted index tracking QUG ecosystem tokens. Automatically rebalances monthly to capture market growth.',
            website: 'https://quillon.xyz/index',
            whitepaper: 'https://quillon.xyz/docs/qnk10-whitepaper',
            isIndexToken: true,
            indexData: {
              methodology: 'market_cap_weighted',
              rebalanceFrequency: 'Monthly',
              managementFee: 0.5,
              performanceFee: 10,
              navPerShare: qnk10NavPerShare, // Projected NAV
              totalAUM: 0, // v2.3.8-beta: Not yet deployed
              components: [
                // Primary component uses live oracle price
                { symbol: 'QUG', name: 'Quillon', weight: 60, price: qugPrice, change24h: qugChange24h },
                { symbol: 'QUGUSD', name: 'Quillon USD', weight: 40, price: qugusdPrice, change24h: qugusdChange24h },
              ],
              lastRebalance: 'Coming soon', // v2.3.8-beta: Not yet deployed
              nextRebalance: 'Coming soon', // v2.3.8-beta: Not yet deployed
              inceptionDate: 'Coming soon', // v2.3.8-beta: Coming soon
              ytdReturn: 0, // v2.3.8-beta: Not yet deployed
              allTimeReturn: 0, // v2.3.8-beta: Not yet deployed
            },
          },
          {
            id: 'index-fund-defi5',
            symbol: 'DEFI5',
            name: 'Stable Yield Index',
            balance: defi5Balance,
            price: defi5NavPerShare,
            change1h: qugusdChange1h,
            change24h: qugusdChange24h,
            change7d: qugusdChange7d,
            volume24h: 0, // v2.3.8-beta: Not yet trading
            // v2.3.7-beta: Convert QUGUSD liquidity to USD
            liquidity: 0, // v2.3.8-beta: Not yet deployed
            icon: 'defi-index',
            marketCap: 0, // v2.3.8-beta: Not yet deployed
            fullyDilutedMarketCap: 0, // v2.3.8-beta: Not yet deployed
            totalSupply: 0, // v2.3.8-beta: Not yet minted
            circulatingSupply: 0, // v2.3.8-beta: Not yet minted
            holders: 0, // v2.3.8-beta: Not yet deployed
            features: {
              reflection: false,
              autoLiquidity: false,
              buybackAndBurn: false,
              antiWhale: true,
              quantumSecured: true,
            },
            fees: {
              buy: 0.1,
              sell: 0.1,
              transfer: 0,
            },
            description: 'Stable yield index focused on QUGUSD stablecoin and liquidity pool exposure. Designed for lower volatility.',
            website: 'https://quillon.xyz/index',
            whitepaper: 'https://quillon.xyz/docs/defi5-whitepaper',
            isIndexToken: true,
            indexData: {
              methodology: 'equal_weighted',
              rebalanceFrequency: 'Bi-weekly',
              managementFee: 0.75,
              performanceFee: 15,
              navPerShare: defi5NavPerShare, // Projected NAV
              totalAUM: 0, // v2.3.8-beta: Not yet deployed
              components: [
                // Uses live oracle prices
                { symbol: 'QUGUSD', name: 'Quillon USD', weight: 50, price: qugusdPrice, change24h: qugusdChange24h },
                { symbol: 'QUG', name: 'Quillon', weight: 50, price: qugPrice, change24h: qugChange24h },
              ],
              lastRebalance: 'Coming soon', // v2.3.8-beta: Not yet deployed
              nextRebalance: 'Coming soon', // v2.3.8-beta: Not yet deployed
              inceptionDate: 'Coming soon', // v2.3.8-beta: Coming soon
              ytdReturn: 0, // v2.3.8-beta: Not yet deployed
              allTimeReturn: 0, // v2.3.8-beta: Not yet deployed
            },
          },
        ];

        console.log('📊 Adding index fund tokens to Available Tokens:', indexFundTokens.map(t => t.symbol).join(', '));
        enrichedTokens = [...enrichedTokens, ...indexFundTokens];

        // v2.9.27-beta: Add perpetual markets as tokens
        // Fetch perpetual market data
        let perpTokens: Token[] = [];
        try {
          const perpResponse = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/perp/markets/QUG-PERP`);
          if (perpResponse.ok) {
            const perpData = await perpResponse.json();
            const market = perpData.success && perpData.market ? perpData.market : perpData;

            if (market) {
              const markPrice = (market.mark_price || 0) / 1e24;
              const indexPrice = (market.index_price || 0) / 1e24;

              perpTokens = [{
                id: 'qug-perp',
                symbol: 'QUG-PERP',
                name: 'QUG Perpetual',
                balance: 0, // User positions shown in modal
                price: markPrice || qugPrice,
                change1h: 0, // TODO: Track perp price changes
                change24h: 0,
                change7d: 0,
                volume24h: (market.volume_24h || 0) / 1e24,
                liquidity: (market.open_interest || 0) / 1e24,
                icon: '📈',
                marketCap: (market.open_interest || 0) / 1e24 * markPrice,
                totalSupply: 0,
                circulatingSupply: 0,
                holders: 0,
                features: {
                  reflection: false,
                  autoLiquidity: false,
                  buybackAndBurn: false,
                  antiWhale: false,
                  quantumSecured: true,
                },
                fees: {
                  buy: (market.taker_fee || 0.001) * 100,
                  sell: (market.taker_fee || 0.001) * 100,
                  transfer: 0,
                },
                description: 'Trade QUG with up to 100x leverage. Perpetual contracts allow long and short positions with no expiry date.',
                website: 'https://quillon.xyz',
                isPerp: true,
                perpData: {
                  market: 'QUG-PERP',
                  markPrice: markPrice,
                  indexPrice: indexPrice,
                  fundingRate: (market.funding_rate || 0) * 100,
                  openInterest: (market.open_interest || 0) / 1e24,
                  maxLeverage: market.max_leverage || 100,
                  maintenanceMargin: market.maintenance_margin || 0.01,
                  takerFee: market.taker_fee || 0.001,
                  makerFee: market.maker_fee || 0.0005,
                },
              }];
              console.log('📈 Adding perpetual market to Available Tokens: QUG-PERP');
            }
          }
        } catch (error) {
          console.log('ℹ️ Could not fetch perpetual market data:', error);
        }

        enrichedTokens = [...enrichedTokens, ...perpTokens];

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

    // Initial fetch — safety timeout clears spinner if fetchTokens ever hangs
    const loadingTimeoutId = setTimeout(() => {
      if (mounted) {
        console.warn('[DEX] fetchTokens timed out after 30s — clearing spinner');
        setLoading(false);
      }
    }, 30000);
    fetchTokens().finally(() => clearTimeout(loadingTimeoutId));

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

        // v2.3.28-beta: Check BOTH ref AND localStorage cooldown for race condition protection
        // The ref is set BEFORE API call, but check localStorage too as backup
        const cooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
        const cooldownActive = swapJustCompletedRef.current || Date.now() < cooldownUntil;

        if (cooldownActive) {
          console.log('⏸️ [DEX] Skipping SSE balance refresh - cooldown active (ref:', swapJustCompletedRef.current, ', localStorage:', Date.now() < cooldownUntil, ')');
          return;
        }

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
            console.log('✅ [DEX] Balance update confirmed for current wallet');
            // v5.0.1: Directly update QUG balance from SSE event data instead of
            // re-fetching from API (which can return stale 0 if wallet_balances is out of sync)
            const newBalance = data.data?.new_balance ?? data.new_balance;
            if (typeof newBalance === 'number' && newBalance > 0) {
              const displayBalance = newBalance > 1e15 ? newBalance / 1e24 : newBalance;
              console.log(`✅ [DEX] Updating QUG balance directly from SSE: ${displayBalance}`);
              setTokens(prev => prev.map(token => {
                if (token.symbol === 'QUG') {
                  return { ...token, balance: displayBalance };
                }
                return token;
              }));
              safeCacheBalance(displayBalance);
            } else {
              // Fallback: re-fetch from API
              fetchTokens();
            }
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

          // v2.3.28-beta: Check cooldown before refreshing tokens
          const lpCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
          if (swapJustCompletedRef.current || Date.now() < lpCooldownUntil) {
            console.log('⏸️ [DEX] Skipping liquidity pool refresh - cooldown active');
          } else {
            // Refresh tokens to update liquidity values
            fetchTokens();
          }
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
      // v2.3.28-beta: Check cooldown before refreshing
      const cdpCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      if (swapJustCompletedRef.current || Date.now() < cdpCooldownUntil) {
        console.log('⏸️ [DEX] Skipping CDP mint refresh - cooldown active');
        return;
      }
      console.log('💵 CDP mint detected in DexScreen - refreshing tokens');
      fetchTokens();
    };

    // Listen for manual refresh events (from swaps)
    const handleManualRefresh = () => {
      // v2.3.28-beta: Check cooldown before refreshing
      const manualCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      if (swapJustCompletedRef.current || Date.now() < manualCooldownUntil) {
        console.log('⏸️ [DEX] Skipping manual refresh - cooldown active');
        return;
      }
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

  // Re-fetch when user navigates to DEX screen (isActive becomes true)
  const prevIsActive = useRef<boolean | undefined>(undefined);
  useEffect(() => {
    if (isActive && prevIsActive.current === false) {
      window.dispatchEvent(new CustomEvent('manual-token-refresh'));
    }
    prevIsActive.current = isActive;
  }, [isActive]);

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
    if (!isFinite(num) || isNaN(num)) return '$0.00';
    if (num >= 1e30) return `$${(num / 1e30).toFixed(decimals)} Nonillion`;
    if (num >= 1e27) return `$${(num / 1e27).toFixed(decimals)} Octillion`;
    if (num >= 1e24) return `$${(num / 1e24).toFixed(decimals)} Septillion`;
    if (num >= 1e21) return `$${(num / 1e21).toFixed(decimals)} Sextillion`;
    if (num >= 1e18) return `$${(num / 1e18).toFixed(decimals)} Quintillion`;
    if (num >= 1e15) return `$${(num / 1e15).toFixed(decimals)} Quadrillion`;
    if (num >= 1e12) return `$${(num / 1e12).toFixed(decimals)} Trillion`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(decimals)} Billion`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(decimals)} Million`;
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

  // v3.2.20-beta: Accept string amounts to preserve precision for large numbers
  // JavaScript Number loses precision above ~9×10^15 (Number.MAX_SAFE_INTEGER)
  const handleAddLiquidity = useCallback(async (tokenA: string, tokenB: string, amountA: string, amountB: string) => {
    console.log(`🔍 Adding liquidity - Raw inputs:`, { tokenA, tokenB, amountA, amountB, typeA: typeof amountA, typeB: typeof amountB });

    try {
      // Get wallet address from localStorage
      const walletAddress = localStorage.getItem('walletAddress');
      if (!walletAddress) {
        alert('Please connect your wallet first');
        return;
      }

      // Find token addresses and decimals (use "QUG" for native, otherwise use token ID)
      const tokenAData = tokens.find(t => t.symbol === tokenA);
      const tokenBData = tokens.find(t => t.symbol === tokenB);
      const token0 = tokenA === 'QUG' ? 'QUG' : tokenAData?.id || tokenA;
      const token1 = tokenB === 'QUG' ? 'QUG' : tokenBData?.id || tokenB;

      // v4.1.2: ALL pool reserves use 24 decimals internally.
      // Previously custom tokens used their native decimals (e.g., 8), which caused
      // reserves to be 10^16x smaller than expected when displayed (divided by 1e24).
      // The AMM, swap handler, and pool display ALL assume 24 decimals.
      const decimals0 = 24;
      const decimals1 = 24;

      console.log(`📊 Pool reserve decimals: ${tokenA}=${decimals0}, ${tokenB}=${decimals1}`);

      // v3.2.22-beta: Use shared BigInt helpers (moved to top-level)
      const amount0 = parseAmountToBigInt(amountA, decimals0);
      const amount1 = parseAmountToBigInt(amountB, decimals1);

      console.log(`📐 Liquidity: amountA="${amountA}", amountB="${amountB}", decimals0=${decimals0}, decimals1=${decimals1}`);

      // v3.2.25-beta: Validate amounts don't exceed u128 max (~3.4e38)
      // Max tokens = U128_MAX / 10^decimals
      if (amount0 > U128_MAX) {
        const maxTokens0 = Number(U128_MAX / (10n ** BigInt(decimals0)));
        const maxDisplay0 = maxTokens0.toExponential(2);
        alert(`❌ Amount of ${tokenA} exceeds maximum supported!\n\nWith ${decimals0} decimals, max amount: ~${maxDisplay0} tokens\n\nPlease reduce the amount or use a token with fewer decimals.`);
        return;
      }
      if (amount1 > U128_MAX) {
        const maxTokens1 = Number(U128_MAX / (10n ** BigInt(decimals1)));
        const maxDisplay1 = maxTokens1.toExponential(2);
        alert(`❌ Amount of ${tokenB} exceeds maximum supported!\n\nWith ${decimals1} decimals, max amount: ~${maxDisplay1} tokens\n\nPlease reduce the amount or use a token with fewer decimals.`);
        return;
      }

      console.log(`💰 Liquidity amounts after conversion:`, { amount0: amount0.toString(), amount1: amount1.toString(), token0, token1 });

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
    } catch (error: any) {
      console.error('Failed to add liquidity:', error);
      // v3.2.15-beta: Show actual error message to help debug authentication issues
      const errorMessage = error?.message || error?.error || String(error);
      alert(`❌ Failed to add liquidity: ${errorMessage}`);
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

      // Create token object with real data from blockchain
      // v4.0.12: Supply uses token's own decimals, balances use 24 decimals
      const decimals = contract.decimals || 8;
      const rawSupply = contract.total_supply || '0';
      const supplyStr = typeof rawSupply === 'string' ? rawSupply : String(rawSupply);
      const supplyBigInt = BigInt(supplyStr);
      const supplyDivisor = BigInt(10) ** BigInt(decimals);
      const actualSupply = Number(supplyBigInt / supplyDivisor) + Number(supplyBigInt % supplyDivisor) / Number(supplyDivisor);

      if (walletAddress) {
        // Try to fetch token balance
        const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, customTokenAddress);
        if (balanceResponse.success && balanceResponse.data) {
          // v4.1.2: Token balances are in 10^(2*decimals) format due to double-conversion
          // in contracts_api.rs (frontend sends base units, backend multiplies again).
          // For 8-decimal tokens: raw = display * 10^16, NOT 10^24.
          const rawBalance = balanceResponse.data.balance || '0';
          const balanceStr = typeof rawBalance === 'string' ? rawBalance : String(rawBalance);
          const balanceBigInt = BigInt(balanceStr);
          const balDivisor = BigInt(10) ** BigInt(2 * decimals);
          tokenBalance = Number(balanceBigInt / balDivisor) + Number(balanceBigInt % balDivisor) / Number(balDivisor);
          console.log(`✅ [Custom Token] Converted balance from ${rawBalance} to ${tokenBalance} (decimals: ${decimals}, divisor: 10^${2*decimals})`);
        }
      }

      const customToken: Token = {
        id: customTokenAddress,
        symbol: contract.token_symbol || contract.symbol || 'CUSTOM',
        name: contract.token_name || contract.name || 'Custom Token',
        balance: tokenBalance, // v1.4.9: Now properly converted from raw balance
        price: 1.0, // Default price, can be calculated from liquidity pools later
        change1h: 0,
        change24h: 0,
        change7d: 0,
        volume24h: 0,
        liquidity: actualSupply, // Use calculated supply
        marketCap: 0,
        fullyDilutedMarketCap: actualSupply * 1.0, // FDV = total supply * price
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

  // AI Token Analyzer - sends token data to AI for analysis
  const handleAIAnalyze = async (token: Token) => {
    setAiAnalyzingToken(token);
    setAiAnalysisLoading(true);
    setAiAnalysisResult(null);

    try {
      // Prepare token metrics for AI analysis
      const tokenData = {
        symbol: token.symbol,
        name: token.name,
        price: token.price,
        marketCap: token.marketCap,
        liquidity: token.liquidity,
        volume24h: token.volume24h,
        change1h: token.change1h,
        change24h: token.change24h,
        change7d: token.change7d,
        holders: token.holders,
        totalSupply: token.totalSupply,
        circulatingSupply: token.circulatingSupply,
        features: token.features,
        fees: token.fees,
      };

      // Call the AI chat API for token analysis
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `Analyze this token for investment quality. Provide a score 0-100, verdict (GOOD/CAUTION/RISKY), and key metrics analysis. Token data: ${JSON.stringify(tokenData)}`,
          context: 'token_analysis',
        }),
      });

      if (!response.ok) {
        throw new Error('AI analysis request failed');
      }

      const data = await response.json();

      // Parse AI response or generate analysis from token metrics
      // If AI is unavailable, generate analysis from raw metrics
      const liquidityScore = token.liquidity > 100000 ? 'good' : token.liquidity > 10000 ? 'warn' : 'bad';
      const volumeScore = token.volume24h > 50000 ? 'good' : token.volume24h > 5000 ? 'warn' : 'bad';
      const holdersScore = token.holders > 100 ? 'good' : token.holders > 10 ? 'warn' : 'bad';
      const priceStabilityScore = Math.abs(token.change24h) < 10 ? 'good' : Math.abs(token.change24h) < 30 ? 'warn' : 'bad';
      const activityScore = token.volume24h > 0 && token.liquidity > 0 ? 'good' : 'bad';

      // Calculate overall score
      const scoreMap = { good: 25, warn: 15, bad: 5 };
      const overallScore = Math.min(100,
        scoreMap[liquidityScore] +
        scoreMap[volumeScore] +
        scoreMap[holdersScore] +
        scoreMap[priceStabilityScore] +
        scoreMap[activityScore]
      );

      const verdict = overallScore >= 75 ? 'GOOD' : overallScore >= 50 ? 'CAUTION' : 'RISKY';

      // Try to parse AI response if available
      let summary = data.response || data.message || '';
      if (!summary) {
        summary = `${token.symbol} has ${token.liquidity > 100000 ? 'strong' : token.liquidity > 10000 ? 'moderate' : 'low'} liquidity ($${token.liquidity.toLocaleString()}), ` +
          `${token.volume24h > 50000 ? 'high' : token.volume24h > 5000 ? 'moderate' : 'low'} trading volume ($${token.volume24h.toLocaleString()}/24h), ` +
          `and ${token.holders > 100 ? 'good' : token.holders > 10 ? 'moderate' : 'limited'} holder distribution (${token.holders} holders). ` +
          `Price ${token.change24h >= 0 ? 'up' : 'down'} ${Math.abs(token.change24h).toFixed(2)}% in 24h.`;
      }

      setAiAnalysisResult({
        score: overallScore,
        verdict,
        summary,
        metrics: {
          liquidity: { value: `$${token.liquidity.toLocaleString()}`, status: liquidityScore as 'good' | 'warn' | 'bad' },
          activity: { value: token.volume24h > 0 ? 'Active' : 'Inactive', status: activityScore as 'good' | 'warn' | 'bad' },
          holders: { value: token.holders.toLocaleString(), status: holdersScore as 'good' | 'warn' | 'bad' },
          priceStability: { value: `${token.change24h >= 0 ? '+' : ''}${token.change24h.toFixed(2)}%`, status: priceStabilityScore as 'good' | 'warn' | 'bad' },
          volume: { value: `$${token.volume24h.toLocaleString()}`, status: volumeScore as 'good' | 'warn' | 'bad' },
        },
      });
    } catch (error) {
      console.error('AI analysis error:', error);
      // Generate fallback analysis from raw metrics
      const liquidityScore = token.liquidity > 100000 ? 'good' : token.liquidity > 10000 ? 'warn' : 'bad';
      const volumeScore = token.volume24h > 50000 ? 'good' : token.volume24h > 5000 ? 'warn' : 'bad';
      const holdersScore = token.holders > 100 ? 'good' : token.holders > 10 ? 'warn' : 'bad';
      const priceStabilityScore = Math.abs(token.change24h) < 10 ? 'good' : Math.abs(token.change24h) < 30 ? 'warn' : 'bad';
      const activityScore = token.volume24h > 0 && token.liquidity > 0 ? 'good' : 'bad';

      const scoreMap = { good: 25, warn: 15, bad: 5 };
      const overallScore = Math.min(100,
        scoreMap[liquidityScore] +
        scoreMap[volumeScore] +
        scoreMap[holdersScore] +
        scoreMap[priceStabilityScore] +
        scoreMap[activityScore]
      );

      const verdict = overallScore >= 75 ? 'GOOD' : overallScore >= 50 ? 'CAUTION' : 'RISKY';

      setAiAnalysisResult({
        score: overallScore,
        verdict,
        summary: `${token.symbol} metrics analysis: Liquidity $${token.liquidity.toLocaleString()}, Volume $${token.volume24h.toLocaleString()}/24h, ${token.holders} holders, ${token.change24h >= 0 ? '+' : ''}${token.change24h.toFixed(2)}% 24h change.`,
        metrics: {
          liquidity: { value: `$${token.liquidity.toLocaleString()}`, status: liquidityScore as 'good' | 'warn' | 'bad' },
          activity: { value: token.volume24h > 0 ? 'Active' : 'Inactive', status: activityScore as 'good' | 'warn' | 'bad' },
          holders: { value: token.holders.toLocaleString(), status: holdersScore as 'good' | 'warn' | 'bad' },
          priceStability: { value: `${token.change24h >= 0 ? '+' : ''}${token.change24h.toFixed(2)}%`, status: priceStabilityScore as 'good' | 'warn' | 'bad' },
          volume: { value: `$${token.volume24h.toLocaleString()}`, status: volumeScore as 'good' | 'warn' | 'bad' },
        },
      });
    } finally {
      setAiAnalysisLoading(false);
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
        selectedToken.isIndexToken ? (
          <IndexFundModal
            token={selectedToken}
            onClose={handleCloseModal}
          />
        ) : selectedToken.isPerp ? (
          // v2.9.27-beta: Perpetual Details Modal - Custom UI for perpetual contracts
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={handleCloseModal}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gradient-to-br from-gray-900 via-purple-900/30 to-gray-900 rounded-2xl border border-purple-500/30 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e: React.MouseEvent) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="p-6 border-b border-purple-500/20">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-16 h-16 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center text-2xl">
                      📈
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-white">{selectedToken.symbol}</h2>
                      <p className="text-gray-400">{selectedToken.name}</p>
                    </div>
                  </div>
                  <button onClick={handleCloseModal} className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                    <X className="w-6 h-6 text-gray-400" />
                  </button>
                </div>
              </div>

              {/* Price Info */}
              <div className="p-6 border-b border-purple-500/20">
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Mark Price</div>
                    <div className="text-3xl font-bold text-white">
                      ${formatPrice(selectedToken.perpData?.markPrice || selectedToken.price)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Index Price</div>
                    <div className="text-3xl font-bold text-cyan-400">
                      ${formatPrice(selectedToken.perpData?.indexPrice || selectedToken.price)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Market Stats */}
              <div className="p-6 border-b border-purple-500/20">
                <h3 className="text-lg font-semibold text-white mb-4">Market Statistics</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-black/30 rounded-lg p-3">
                    <div className="text-sm text-gray-400">Funding Rate</div>
                    <div className={`text-lg font-bold ${(selectedToken.perpData?.fundingRate || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {(selectedToken.perpData?.fundingRate || 0) >= 0 ? '+' : ''}{(selectedToken.perpData?.fundingRate || 0).toFixed(4)}%
                    </div>
                  </div>
                  <div className="bg-black/30 rounded-lg p-3">
                    <div className="text-sm text-gray-400">Open Interest</div>
                    <div className="text-lg font-bold text-white">
                      ${(selectedToken.perpData?.openInterest || 0).toLocaleString()}
                    </div>
                  </div>
                  <div className="bg-black/30 rounded-lg p-3">
                    <div className="text-sm text-gray-400">24h Volume</div>
                    <div className="text-lg font-bold text-white">
                      ${selectedToken.volume24h.toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>

              {/* Trading Info */}
              <div className="p-6 border-b border-purple-500/20">
                <h3 className="text-lg font-semibold text-white mb-4">Trading Information</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                    <span className="text-gray-400">Max Leverage</span>
                    <span className="text-white font-bold">{selectedToken.perpData?.maxLeverage || 100}x</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                    <span className="text-gray-400">Maintenance Margin</span>
                    <span className="text-white font-bold">{((selectedToken.perpData?.maintenanceMargin || 0.01) * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                    <span className="text-gray-400">Taker Fee</span>
                    <span className="text-white font-bold">{((selectedToken.perpData?.takerFee || 0.001) * 100).toFixed(3)}%</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-black/30 rounded-lg">
                    <span className="text-gray-400">Maker Fee</span>
                    <span className="text-white font-bold">{((selectedToken.perpData?.makerFee || 0.0005) * 100).toFixed(3)}%</span>
                  </div>
                </div>
              </div>

              {/* Description */}
              <div className="p-6 border-b border-purple-500/20">
                <h3 className="text-lg font-semibold text-white mb-2">About</h3>
                <p className="text-gray-400">{selectedToken.description}</p>
              </div>

              {/* Trade Button */}
              <div className="p-6">
                <button
                  onClick={() => {
                    handleCloseModal();
                    setDexMode('perpetual');
                  }}
                  className="w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 rounded-xl font-bold text-white text-lg transition-all shadow-lg shadow-purple-500/25"
                >
                  Trade {selectedToken.symbol}
                </button>
              </div>
            </motion.div>
          </motion.div>
        ) : (
          <TokenDetailsModal
            token={selectedToken}
            onClose={handleCloseModal}
          />
        )
      )}

      {/* v8.2.8: XLIST Crowdfund Modal */}
      {selectedCampaign && (
        <XListCrowdfundModal
          campaign={selectedCampaign}
          onClose={() => setSelectedCampaign(null)}
          walletAddress={localStorage.getItem('walletAddress') || ''}
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

      {/* Token Selector Modal - From Token (exclude index tokens - they can only be minted/redeemed) */}
      <TokenSelectorModal
        isOpen={isFromTokenSelectorOpen}
        onClose={() => setIsFromTokenSelectorOpen(false)}
        onSelectToken={handleSelectFromToken}
        tokens={tokens.filter(t => !t.isIndexToken && !t.isPerp)}
        boostedTokens={boostedTokens}
        currentToken={findToken(swapFrom)}
      />

      {/* Token Selector Modal - To Token (exclude index tokens and perps - they can only be traded via their own interfaces) */}
      <TokenSelectorModal
        isOpen={isToTokenSelectorOpen}
        onClose={() => setIsToTokenSelectorOpen(false)}
        onSelectToken={handleSelectToToken}
        tokens={tokens.filter(t => !t.isIndexToken && !t.isPerp)}
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
                      <span className="text-white font-bold">{getTokenDisplayName(removingPool.token0, tokens)} / {getTokenDisplayName(removingPool.token1, tokens)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Your Reserves:</span>
                      <div className="text-right">
                        {/* v3.6.7-beta: Pools always use 24 decimals */}
                        <div className="text-white text-sm">
                          {(() => {
                            const raw = parseU128(removingPool.reserve0);
                            const display = raw / 1e24;
                            if (display >= 1e12) return (display / 1e12).toFixed(2) + 'T';
                            if (display >= 1e9) return (display / 1e9).toFixed(2) + 'B';
                            if (display >= 1e6) return (display / 1e6).toFixed(2) + 'M';
                            if (display >= 1e3) return (display / 1e3).toFixed(2) + 'K';
                            return display.toFixed(display < 1 ? 6 : 2);
                          })()} {getTokenDisplayName(removingPool.token0, tokens)}
                        </div>
                        <div className="text-white text-sm">
                          {(() => {
                            const raw = parseU128(removingPool.reserve1);
                            const display = raw / 1e24;
                            return display.toFixed(display < 1 ? 8 : 2);
                          })()} {getTokenDisplayName(removingPool.token1, tokens)}
                        </div>
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
                        <span className="text-gray-400">{getTokenDisplayName(removingPool.token0, tokens)}:</span>
                        <span className="text-white font-bold">
                          {(() => {
                            const raw = parseU128(removingPool.reserve0);
                            const display = (raw / 1e24) * removePercentage / 100;
                            if (display >= 1e12) return (display / 1e12).toFixed(2) + 'T';
                            if (display >= 1e9) return (display / 1e9).toFixed(2) + 'B';
                            if (display >= 1e6) return (display / 1e6).toFixed(2) + 'M';
                            if (display >= 1e3) return (display / 1e3).toFixed(2) + 'K';
                            return display.toFixed(display < 1 ? 6 : 2);
                          })()}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">{getTokenDisplayName(removingPool.token1, tokens)}:</span>
                        <span className="text-white font-bold">
                          {(() => {
                            const raw = parseU128(removingPool.reserve1);
                            const display = (raw / 1e24) * removePercentage / 100;
                            return display.toFixed(display < 1 ? 8 : 2);
                          })()}
                        </span>
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
                      <TokenIcon symbol={nitroBoostToken.symbol} icon={nitroBoostToken.icon} logoUrl={nitroBoostToken.logoUrl} size={40} />
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
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-3xl font-black bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
                Quantum DEX
              </h1>
              {!isMenuCollapsed && <p className="text-gray-400 mt-1">Decentralized exchange with quantum security</p>}
            </div>
            <button
              onClick={() => setIsMenuCollapsed(!isMenuCollapsed)}
              className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              title={isMenuCollapsed ? 'Show menu' : 'Hide menu'}
            >
              {isMenuCollapsed ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronUp className="w-4 h-4 text-gray-400" />}
            </button>
          </div>

          {/* Mode Toggle: Spot / Perpetual */}
          <AnimatePresence>
            {!isMenuCollapsed && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
                className="flex items-center gap-2 bg-white/5 rounded-xl p-1"
              >
                <button
                  onClick={() => setDexMode('spot')}
                  className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                    dexMode === 'spot'
                      ? 'bg-quantum-cyan text-black'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Spot
                </button>
                <button
                  onClick={() => setDexMode('perpetual')}
                  className={`px-4 py-2 rounded-lg font-semibold transition-all flex items-center gap-2 ${
                    dexMode === 'perpetual'
                      ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Perpetual
                  <span className="text-xs bg-red-500/30 px-2 py-0.5 rounded-full">10x</span>
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        {isMenuCollapsed && (
          <span className="text-xs text-gray-500 ml-1">{dexMode === 'spot' ? 'Spot' : 'Perpetual'} mode</span>
        )}
      </motion.div>

      {loading ? (
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="inline-block w-12 h-12 border-4 border-quantum-cyan/30 border-t-quantum-cyan rounded-full animate-spin mb-4" />
            <p className="text-gray-400">Loading tokens...</p>
          </div>
        </div>
      ) : dexMode === 'perpetual' ? (
        /* ========== PERPETUAL FUTURES INTERFACE ========== */
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Trading Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-orange-500 to-red-500 rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity" />
              <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-orange-500/20 p-6 space-y-4">
                <h2 className="text-xl font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
                  QUG-PERP
                </h2>

                {/* Market Info */}
                <div className="grid grid-cols-2 gap-4 p-4 bg-white/5 rounded-xl text-sm">
                  <div>
                    <div className="text-gray-400">Mark Price</div>
                    <div className="text-white font-bold">
                      ${perpMarket?.mark_price ? (perpMarket.mark_price / 1e24).toFixed(4) : '0.0000'}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Index Price</div>
                    <div className="text-white font-bold">
                      ${perpMarket?.index_price ? (perpMarket.index_price / 1e24).toFixed(4) : '0.0000'}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Funding Rate</div>
                    <div className={`font-bold ${(perpMarket?.funding_rate || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {((perpMarket?.funding_rate || 0) * 100).toFixed(4)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Open Interest</div>
                    <div className="text-white font-bold">
                      {((perpMarket?.open_interest_long || 0) + (perpMarket?.open_interest_short || 0)) / 1e24} QUG
                    </div>
                  </div>
                </div>

                {/* Order Type Toggle (Market/Limit) */}
                <div className="flex gap-2 p-1 bg-white/5 rounded-xl">
                  <button
                    onClick={() => setPerpOrderType('market')}
                    className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-all ${
                      perpOrderType === 'market'
                        ? 'bg-orange-500 text-white'
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    Market
                  </button>
                  <button
                    onClick={() => setPerpOrderType('limit')}
                    className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-all ${
                      perpOrderType === 'limit'
                        ? 'bg-blue-500 text-white'
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    Limit
                  </button>
                </div>

                {/* Long/Short Toggle */}
                <div className="flex gap-2">
                  <button
                    onClick={() => setPerpSide('long')}
                    className={`flex-1 py-3 rounded-xl font-bold transition-all ${
                      perpSide === 'long'
                        ? 'bg-green-500 text-white'
                        : 'bg-white/5 text-gray-400 hover:bg-white/10'
                    }`}
                  >
                    Long
                  </button>
                  <button
                    onClick={() => setPerpSide('short')}
                    className={`flex-1 py-3 rounded-xl font-bold transition-all ${
                      perpSide === 'short'
                        ? 'bg-red-500 text-white'
                        : 'bg-white/5 text-gray-400 hover:bg-white/10'
                    }`}
                  >
                    Short
                  </button>
                </div>

                {/* Leverage Slider */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Leverage</span>
                    <span className="text-white font-bold">{perpLeverage}x</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={perpLeverage}
                    onChange={(e) => setPerpLeverage(parseInt(e.target.value))}
                    className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-orange-500"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>1x</span>
                    <span>5x</span>
                    <span>10x</span>
                  </div>
                </div>

                {/* Size Input */}
                <div className="space-y-2">
                  <label className="text-sm text-gray-400">Size (QUG)</label>
                  <input
                    type="number"
                    value={perpSize}
                    onChange={(e) => setPerpSize(e.target.value)}
                    placeholder="0.0"
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-orange-500/50 transition-colors"
                  />
                </div>

                {/* Limit Price Input (for limit orders) */}
                {perpOrderType === 'limit' && (
                  <div className="space-y-2">
                    <label className="text-sm text-gray-400">Limit Price (QUGUSD)</label>
                    <div className="relative">
                      <input
                        type="number"
                        value={limitPrice}
                        onChange={(e) => setLimitPrice(e.target.value)}
                        placeholder={perpMarket?.mark_price ? (perpMarket.mark_price / 1e24).toFixed(4) : '0.0'}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500/50 transition-colors"
                      />
                      {orderBook && (
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex gap-2 text-xs">
                          <button
                            onClick={() => orderBook.best_bid && setLimitPrice((orderBook.best_bid / 1e24).toFixed(4))}
                            className="px-2 py-1 bg-green-500/20 text-green-400 rounded hover:bg-green-500/30"
                          >
                            Bid
                          </button>
                          <button
                            onClick={() => orderBook.best_ask && setLimitPrice((orderBook.best_ask / 1e24).toFixed(4))}
                            className="px-2 py-1 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30"
                          >
                            Ask
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Time-in-Force (for limit orders) */}
                {perpOrderType === 'limit' && (
                  <div className="space-y-2">
                    <label className="text-sm text-gray-400">Time in Force</label>
                    <div className="grid grid-cols-4 gap-2">
                      {(['gtc', 'ioc', 'fok', 'post_only'] as const).map((tif) => (
                        <button
                          key={tif}
                          onClick={() => setTimeInForce(tif)}
                          className={`py-2 rounded-lg text-xs font-semibold transition-all ${
                            timeInForce === tif
                              ? 'bg-blue-500 text-white'
                              : 'bg-white/5 text-gray-400 hover:bg-white/10'
                          }`}
                        >
                          {tif === 'gtc' ? 'GTC' : tif === 'ioc' ? 'IOC' : tif === 'fok' ? 'FOK' : 'POST'}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Collateral Input (for market orders) */}
                {perpOrderType === 'market' && (
                  <div className="space-y-2">
                    <label className="text-sm text-gray-400">Collateral (QUGUSD)</label>
                    <input
                      type="number"
                      value={perpCollateral}
                      onChange={(e) => setPerpCollateral(e.target.value)}
                      placeholder="0.0"
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-orange-500/50 transition-colors"
                    />
                  </div>
                )}

                {/* Position Info - Market Orders */}
                {perpOrderType === 'market' && perpSize && perpCollateral && (
                  <div className="p-4 bg-white/5 rounded-xl text-sm space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Position Value</span>
                      <span className="text-white">
                        ${(parseFloat(perpSize || '0') * (perpMarket?.mark_price || 0) / 1e24).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Liquidation Price</span>
                      <span className="text-red-400">
                        ${((perpMarket?.mark_price || 0) / 1e24 * (perpSide === 'long' ? (1 - 0.9 / perpLeverage) : (1 + 0.9 / perpLeverage))).toFixed(4)}
                      </span>
                    </div>
                  </div>
                )}

                {/* Order Info - Limit Orders */}
                {perpOrderType === 'limit' && perpSize && limitPrice && (
                  <div className="p-4 bg-blue-500/10 rounded-xl text-sm space-y-2 border border-blue-500/20">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Order Value</span>
                      <span className="text-white">
                        ${(parseFloat(perpSize || '0') * parseFloat(limitPrice || '0')).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Required Margin ({perpLeverage}x)</span>
                      <span className="text-blue-400">
                        ${(parseFloat(perpSize || '0') * parseFloat(limitPrice || '0') / perpLeverage).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Est. Liquidation</span>
                      <span className="text-red-400">
                        ${(parseFloat(limitPrice || '0') * (perpSide === 'long' ? (1 - 0.9 / perpLeverage) : (1 + 0.9 / perpLeverage))).toFixed(4)}
                      </span>
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {perpError && (
                  <div className="p-3 bg-red-500/20 border border-red-500/30 rounded-xl text-red-400 text-sm">
                    {perpError}
                  </div>
                )}

                {/* Submit Button */}
                {perpOrderType === 'market' ? (
                  <button
                    onClick={openPerpPosition}
                    disabled={loadingPerp || !perpSize || !perpCollateral}
                    className={`w-full py-4 rounded-xl font-bold text-lg transition-all ${
                      perpSide === 'long'
                        ? 'bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-400 hover:to-emerald-400 text-white'
                        : 'bg-gradient-to-r from-red-500 to-rose-500 hover:from-red-400 hover:to-rose-400 text-white'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {loadingPerp ? (
                      <Loader2 className="w-6 h-6 animate-spin mx-auto" />
                    ) : (
                      `${perpSide === 'long' ? 'Long' : 'Short'} QUG-PERP (Market)`
                    )}
                  </button>
                ) : (
                  <button
                    onClick={placeLimitOrder}
                    disabled={loadingPerp || !perpSize || !limitPrice}
                    className={`w-full py-4 rounded-xl font-bold text-lg transition-all ${
                      perpSide === 'long'
                        ? 'bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-400 hover:to-cyan-400 text-white'
                        : 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-400 hover:to-pink-400 text-white'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {loadingPerp ? (
                      <Loader2 className="w-6 h-6 animate-spin mx-auto" />
                    ) : (
                      `${perpSide === 'long' ? 'Buy' : 'Sell'} Limit @ ${limitPrice}`
                    )}
                  </button>
                )}
              </div>
            </div>
          </motion.div>

          {/* Right: Order Book + Positions Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Order Book Display */}
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl blur-lg opacity-20" />
              <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-blue-500/20 p-6">
                <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  Order Book
                  {orderBook && (
                    <span className="text-xs text-gray-400 font-normal">
                      Spread: {orderBook.spread ? `$${(orderBook.spread / 1e24).toFixed(4)}` : 'N/A'}
                    </span>
                  )}
                </h2>

                <div className="grid grid-cols-2 gap-4">
                  {/* Bids (Buy Orders) */}
                  <div>
                    <div className="text-sm text-green-400 font-semibold mb-2">Bids (Buy)</div>
                    <div className="space-y-1">
                      {orderBook?.bids && orderBook.bids.length > 0 ? (
                        orderBook.bids.slice(0, 8).map((level, idx) => (
                          <div key={idx} className="flex justify-between text-sm py-1 px-2 rounded bg-green-500/10">
                            <span className="text-green-400">${(level.price / 1e24).toFixed(4)}</span>
                            <span className="text-gray-300">{(level.size / 1e24).toFixed(4)}</span>
                          </div>
                        ))
                      ) : (
                        <div className="text-gray-500 text-sm text-center py-4">No buy orders</div>
                      )}
                    </div>
                  </div>

                  {/* Asks (Sell Orders) */}
                  <div>
                    <div className="text-sm text-red-400 font-semibold mb-2">Asks (Sell)</div>
                    <div className="space-y-1">
                      {orderBook?.asks && orderBook.asks.length > 0 ? (
                        orderBook.asks.slice(0, 8).map((level, idx) => (
                          <div key={idx} className="flex justify-between text-sm py-1 px-2 rounded bg-red-500/10">
                            <span className="text-red-400">${(level.price / 1e24).toFixed(4)}</span>
                            <span className="text-gray-300">{(level.size / 1e24).toFixed(4)}</span>
                          </div>
                        ))
                      ) : (
                        <div className="text-gray-500 text-sm text-center py-4">No sell orders</div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Best Bid/Ask Summary */}
                {orderBook && (orderBook.best_bid || orderBook.best_ask) && (
                  <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-xs text-gray-400">Best Bid</div>
                      <div className="text-green-400 font-bold">
                        {orderBook.best_bid ? `$${(orderBook.best_bid / 1e24).toFixed(4)}` : '-'}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Mid Price</div>
                      <div className="text-white font-bold">
                        {orderBook.best_bid && orderBook.best_ask
                          ? `$${((orderBook.best_bid + orderBook.best_ask) / 2 / 1e24).toFixed(4)}`
                          : '-'}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Best Ask</div>
                      <div className="text-red-400 font-bold">
                        {orderBook.best_ask ? `$${(orderBook.best_ask / 1e24).toFixed(4)}` : '-'}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Open Limit Orders */}
            {limitOrders.length > 0 && (
              <div className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur-lg opacity-20" />
                <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-purple-500/20 p-6">
                  <h2 className="text-xl font-bold text-white mb-4">Open Orders ({limitOrders.length})</h2>
                  <div className="space-y-3">
                    {limitOrders.map((order: any) => (
                      <div key={order.id} className="p-3 bg-white/5 rounded-xl border border-white/10 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className={`px-2 py-1 rounded text-xs font-bold ${
                            order.side === 'buy' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                          }`}>
                            {order.side.toUpperCase()}
                          </span>
                          <div>
                            <div className="text-white font-semibold">
                              {(order.remaining_size / 1e24).toFixed(4)} @ ${(order.price / 1e24).toFixed(4)}
                            </div>
                            <div className="text-xs text-gray-400">
                              {order.time_in_force.toUpperCase()} · {order.leverage}x
                            </div>
                          </div>
                        </div>
                        <button
                          onClick={() => cancelLimitOrder(order.id)}
                          className="px-3 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg text-sm transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Positions Panel */}
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-orange-500 to-red-500 rounded-2xl blur-lg opacity-20" />
              <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-orange-500/20 p-6">
                <h2 className="text-xl font-bold text-white mb-4">Open Positions</h2>

                {perpPositions.length === 0 ? (
                  <div className="text-center py-12 text-gray-400">
                    <div className="text-4xl mb-4">📊</div>
                    <p>No open positions</p>
                    <p className="text-sm mt-2">Open a long or short position to start trading</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {perpPositions.map((position: any) => (
                      <div
                        key={position.id}
                        className="p-4 bg-white/5 rounded-xl border border-white/10"
                      >
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <span className={`px-2 py-1 rounded text-xs font-bold ${
                              position.side === 'long' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                            }`}>
                              {position.side.toUpperCase()} {position.leverage}x
                            </span>
                            <span className="text-white font-bold">{position.market}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => {
                                setEditingPosition(position);
                                setEditMode('addMargin');
                                setEditAmount('');
                              }}
                              className="px-3 py-1.5 bg-green-500/20 hover:bg-green-500/30 text-green-400 rounded-lg text-xs font-semibold transition-colors"
                            >
                              + Margin
                            </button>
                            <button
                              onClick={() => {
                                setEditingPosition(position);
                                setEditMode('removeMargin');
                                setEditAmount('');
                              }}
                              className="px-3 py-1.5 bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400 rounded-lg text-xs font-semibold transition-colors"
                            >
                              - Margin
                            </button>
                            <button
                              onClick={() => {
                                setEditingPosition(position);
                                setEditMode('adjustLeverage');
                                setNewLeverage(position.leverage);
                              }}
                              className="px-3 py-1.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg text-xs font-semibold transition-colors"
                            >
                              Leverage
                            </button>
                            <button
                              onClick={() => closePerpPosition(position.id)}
                              className="px-3 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg text-xs font-semibold transition-colors"
                            >
                              Close
                            </button>
                          </div>
                        </div>

                        <div className="grid grid-cols-5 gap-4 text-sm">
                          <div>
                            <div className="text-gray-400">Size</div>
                            <div className="text-white font-semibold">
                              {(position.size / 1e24).toFixed(4)} QUG
                            </div>
                          </div>
                          <div>
                            <div className="text-gray-400">Collateral</div>
                            <div className="text-cyan-400 font-semibold">
                              {(position.collateral / 1e24).toFixed(2)} QUGUSD
                            </div>
                          </div>
                          <div>
                            <div className="text-gray-400">Entry Price</div>
                            <div className="text-white font-semibold">
                              ${(position.entry_price / 1e24).toFixed(4)}
                            </div>
                          </div>
                          <div>
                            <div className="text-gray-400">Liq. Price</div>
                            <div className="text-red-400 font-semibold">
                              ${(position.liquidation_price / 1e24).toFixed(4)}
                            </div>
                          </div>
                          <div>
                            <div className="text-gray-400">Unrealized PnL</div>
                            <div className={`font-bold ${position.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {position.unrealized_pnl >= 0 ? '+' : ''}{(position.unrealized_pnl / 1e24).toFixed(4)} QUGUSD
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Risk Warning */}
                <div className="mt-6 p-4 bg-orange-500/10 border border-orange-500/20 rounded-xl">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
                    <div className="text-sm text-orange-200/80">
                      <p className="font-semibold text-orange-400 mb-1">Leverage Trading Risk</p>
                      Perpetual contracts carry significant risk. Your position may be liquidated if the market moves against you beyond your maintenance margin. Only trade with funds you can afford to lose.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* 📝 Position Edit Modal */}
          {editingPosition && editMode && (
            <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-gray-900 rounded-2xl border border-white/10 p-6 max-w-md w-full"
              >
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold text-white">
                    {editMode === 'addMargin' && '➕ Add Margin'}
                    {editMode === 'removeMargin' && '➖ Remove Margin'}
                    {editMode === 'adjustLeverage' && '⚡ Adjust Leverage'}
                  </h3>
                  <button
                    onClick={() => {
                      setEditingPosition(null);
                      setEditMode(null);
                    }}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    ✕
                  </button>
                </div>

                {/* Position Info */}
                <div className="bg-white/5 rounded-xl p-4 mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
                      editingPosition.side === 'long' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    }`}>
                      {editingPosition.side.toUpperCase()} {editingPosition.leverage}x
                    </span>
                    <span className="text-white font-bold">{editingPosition.market}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm mt-3">
                    <div>
                      <div className="text-gray-400">Current Collateral</div>
                      <div className="text-cyan-400 font-semibold">{(editingPosition.collateral / 1e24).toFixed(4)} QUGUSD</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Liquidation Price</div>
                      <div className="text-red-400 font-semibold">${(editingPosition.liquidation_price / 1e24).toFixed(4)}</div>
                    </div>
                  </div>
                </div>

                {/* Add/Remove Margin Input */}
                {(editMode === 'addMargin' || editMode === 'removeMargin') && (
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm text-gray-400 mb-2 block">
                        {editMode === 'addMargin' ? 'Amount to Add' : 'Amount to Remove'} (QUGUSD)
                      </label>
                      <input
                        type="number"
                        value={editAmount}
                        onChange={(e) => setEditAmount(e.target.value)}
                        placeholder="0.00"
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-cyan-500/50"
                      />
                    </div>
                    <button
                      onClick={() => {
                        const amount = parseFloat(editAmount);
                        if (amount > 0) {
                          if (editMode === 'addMargin') {
                            addMarginToPosition(editingPosition.id, amount);
                          } else {
                            removeMarginFromPosition(editingPosition.id, amount);
                          }
                        }
                      }}
                      disabled={loadingPerp || !editAmount || parseFloat(editAmount) <= 0}
                      className={`w-full py-3 rounded-xl font-bold transition-all ${
                        editMode === 'addMargin'
                          ? 'bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white'
                          : 'bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 text-black'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {loadingPerp ? 'Processing...' : editMode === 'addMargin' ? 'Add Margin' : 'Remove Margin'}
                    </button>
                  </div>
                )}

                {/* Adjust Leverage Input */}
                {editMode === 'adjustLeverage' && (
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm text-gray-400 mb-2 block">
                        New Leverage: <span className="text-white font-bold">{newLeverage}x</span>
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="10"
                        value={newLeverage}
                        onChange={(e) => setNewLeverage(parseInt(e.target.value))}
                        className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>1x</span>
                        <span>5x</span>
                        <span>10x</span>
                      </div>
                    </div>
                    <div className="bg-white/5 rounded-lg p-3 text-sm">
                      <div className="text-gray-400">
                        {newLeverage > editingPosition.leverage ? (
                          <span className="text-yellow-400">⚠️ Increasing leverage requires less collateral but increases liquidation risk</span>
                        ) : newLeverage < editingPosition.leverage ? (
                          <span className="text-green-400">✓ Decreasing leverage requires more collateral but reduces liquidation risk</span>
                        ) : (
                          <span className="text-gray-400">No change in leverage</span>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => adjustPositionLeverage(editingPosition.id, newLeverage)}
                      disabled={loadingPerp || newLeverage === editingPosition.leverage}
                      className="w-full py-3 rounded-xl font-bold bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loadingPerp ? 'Processing...' : `Change to ${newLeverage}x Leverage`}
                    </button>
                  </div>
                )}
              </motion.div>
            </div>
          )}
        </div>
      ) : (
        /* ========== SPOT TRADING INTERFACE ========== */
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* DEX Network Metrics Dashboard - Comprehensive Statistics (above swap + tokens) */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="lg:col-span-4 mb-2"
          >
            {(() => {
              // === Aggregate Metrics ===
              const totalMarketCap = tokens.reduce((sum, t) => sum + (t.marketCap || 0), 0);
              const totalVolume24h = tokens.reduce((sum, t) => sum + (t.volume24h || 0), 0);
              const totalLiquidity = tokens.reduce((sum, t) => sum + (t.liquidity || 0), 0);
              const totalHolders = tokens.reduce((sum, t) => sum + (t.holders || 0), 0);
              const totalTokens = tokens.length;
              const totalPools = liquidityPools.length;
              const totalFDV = tokens.reduce((sum, t) => sum + ((t.totalSupply || 0) * (t.price || 0)), 0);

              // === Price Movement Stats ===
              const gainers = tokens.filter(t => t.change24h > 0).length;
              const losers = tokens.filter(t => t.change24h < 0).length;
              const unchanged = tokens.filter(t => t.change24h === 0).length;
              const avgChange24h = tokens.length > 0 ? tokens.reduce((sum, t) => sum + (t.change24h || 0), 0) / tokens.length : 0;
              const avgChange1h = tokens.length > 0 ? tokens.reduce((sum, t) => sum + (t.change1h || 0), 0) / tokens.length : 0;
              const avgChange7d = tokens.length > 0 ? tokens.reduce((sum, t) => sum + (t.change7d || 0), 0) / tokens.length : 0;
              const sorted24h = [...tokens].sort((a, b) => (b.change24h || 0) - (a.change24h || 0));
              const topGainer = sorted24h[0];
              const topLoser = sorted24h[sorted24h.length - 1];
              const medianChange24h = sorted24h.length > 0 ? (sorted24h[Math.floor(sorted24h.length / 2)]?.change24h || 0) : 0;

              // === Volume Stats ===
              const tokensWithVolume = tokens.filter(t => (t.volume24h || 0) > 0);
              const avgVolume = tokensWithVolume.length > 0 ? totalVolume24h / tokensWithVolume.length : 0;
              const sortedByVolume = [...tokens].sort((a, b) => (b.volume24h || 0) - (a.volume24h || 0));
              const topVolumeToken = sortedByVolume[0];
              const volumeToMcapRatio = totalMarketCap > 0 ? (totalVolume24h / totalMarketCap * 100) : 0;

              // === Liquidity Stats ===
              const avgLiquidityPerPool = totalPools > 0 ? totalLiquidity / totalPools : 0;
              const sortedByLiquidity = [...tokens].sort((a, b) => (b.liquidity || 0) - (a.liquidity || 0));
              const deepestPool = sortedByLiquidity[0];
              const tokensWithLiquidity = tokens.filter(t => (t.liquidity || 0) > 0).length;
              const liquidityToMcapRatio = totalMarketCap > 0 ? (totalLiquidity / totalMarketCap * 100) : 0;

              // === Token Distribution ===
              const totalCirculating = tokens.reduce((sum, t) => sum + (t.circulatingSupply || 0), 0);
              const totalTotalSupply = tokens.reduce((sum, t) => sum + (t.totalSupply || 0), 0);
              // v7.2.5: Bridge token stats
              const bridgeTokens = tokens.filter(t => ['wBTC', 'wZEC', 'wIRON', 'wETH'].includes(t.symbol));
              const bridgeTVL = bridgeTokens.reduce((sum, t) => sum + (t.liquidity || 0), 0);
              const avgCirculatingPct = totalTotalSupply > 0 ? (totalCirculating / totalTotalSupply * 100) : 0;
              const tokensWithFeatures = {
                reflection: tokens.filter(t => t.features?.reflection).length,
                autoLiquidity: tokens.filter(t => t.features?.autoLiquidity).length,
                buybackAndBurn: tokens.filter(t => t.features?.buybackAndBurn).length,
                antiWhale: tokens.filter(t => t.features?.antiWhale).length,
                quantumSecured: tokens.filter(t => t.features?.quantumSecured).length,
              };
              const indexTokens = tokens.filter(t => t.isIndexToken).length;
              const perpTokens = tokens.filter(t => t.isPerp).length;
              const customTokens = tokens.filter(t => !['QUG', 'QUGUSD'].includes(t.symbol) && !t.isIndexToken && !t.isPerp).length;

              // === Fee Stats ===
              const avgBuyFee = tokens.length > 0 ? tokens.reduce((sum, t) => sum + (t.fees?.buy || 0), 0) / tokens.length : 0;
              const avgSellFee = tokens.length > 0 ? tokens.reduce((sum, t) => sum + (t.fees?.sell || 0), 0) / tokens.length : 0;

              // Compact number formatter without dollar sign
              const fmtNum = (n: number): string => {
                if (!isFinite(n) || isNaN(n)) return '0';
                if (n >= 1e30) return `${(n / 1e30).toFixed(2)} Nonillion`;
                if (n >= 1e27) return `${(n / 1e27).toFixed(2)} Octillion`;
                if (n >= 1e24) return `${(n / 1e24).toFixed(2)} Septillion`;
                if (n >= 1e21) return `${(n / 1e21).toFixed(2)} Sextillion`;
                if (n >= 1e18) return `${(n / 1e18).toFixed(2)} Quintillion`;
                if (n >= 1e15) return `${(n / 1e15).toFixed(2)} Quadrillion`;
                if (n >= 1e12) return `${(n / 1e12).toFixed(2)} Trillion`;
                if (n >= 1e9) return `${(n / 1e9).toFixed(2)} Billion`;
                if (n >= 1e6) return `${(n / 1e6).toFixed(2)} Million`;
                if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
                return n.toFixed(2);
              };

              // Primary headline metrics (big cards)
              const headlineMetrics = [
                { label: 'Total Network Value', value: formatNumber(totalMarketCap), sub: `FDV: ${formatNumber(totalFDV)}`, icon: PieChart, gradient: 'from-quantum-cyan via-blue-500 to-quantum-purple' },
                { label: 'Total Value Locked', value: formatNumber(totalLiquidity), sub: `${totalPools} pools | ${tokensWithLiquidity} tokens`, icon: Wallet, gradient: 'from-green-400 via-emerald-500 to-teal-500' },
                { label: '24h Trading Volume', value: formatNumber(totalVolume24h), sub: `Vol/MCap: ${volumeToMcapRatio.toFixed(2)}%`, icon: BarChart3, gradient: 'from-quantum-purple via-violet-500 to-fuchsia-500' },
                { label: 'Active Markets', value: `${totalTokens} Tokens`, sub: `${totalPools} pairs | ${fmtNum(totalHolders)} holders`, icon: Activity, gradient: 'from-amber-400 via-orange-500 to-red-500' },
              ];

              // Secondary detailed metrics grid
              const detailMetrics = [
                // Market Sentiment
                { label: 'Market Sentiment', value: gainers > losers ? 'Bullish' : gainers < losers ? 'Bearish' : 'Neutral', color: gainers > losers ? 'text-green-400' : gainers < losers ? 'text-red-400' : 'text-yellow-400' },
                { label: 'Gainers / Losers', value: `${gainers} / ${losers}`, color: gainers > losers ? 'text-green-400' : 'text-red-400' },
                { label: 'Bullish Ratio', value: `${((gainers / Math.max(totalTokens, 1)) * 100).toFixed(0)}%`, color: 'text-gray-300' },
                { label: 'Avg 1h Change', value: `${avgChange1h >= 0 ? '+' : ''}${avgChange1h.toFixed(2)}%`, color: avgChange1h >= 0 ? 'text-green-400' : 'text-red-400' },
                { label: 'Avg 24h Change', value: `${avgChange24h >= 0 ? '+' : ''}${avgChange24h.toFixed(2)}%`, color: avgChange24h >= 0 ? 'text-green-400' : 'text-red-400' },
                { label: 'Avg 7d Change', value: `${avgChange7d >= 0 ? '+' : ''}${avgChange7d.toFixed(2)}%`, color: avgChange7d >= 0 ? 'text-green-400' : 'text-red-400' },
                { label: 'Median 24h', value: `${medianChange24h >= 0 ? '+' : ''}${medianChange24h.toFixed(2)}%`, color: medianChange24h >= 0 ? 'text-green-400' : 'text-red-400' },
                // Volume
                { label: 'Top Volume', value: topVolumeToken ? `${topVolumeToken.symbol} ${formatNumber(topVolumeToken.volume24h || 0)}` : '-', color: 'text-quantum-cyan' },
                { label: 'Avg Volume/Token', value: formatNumber(avgVolume), color: 'text-gray-300' },
                { label: 'Tokens Trading', value: `${tokensWithVolume.length} / ${totalTokens}`, color: 'text-gray-300' },
                // Liquidity
                { label: 'Avg Liquidity/Pool', value: formatNumber(avgLiquidityPerPool), color: 'text-gray-300' },
                { label: 'Deepest Pool', value: deepestPool ? `${deepestPool.symbol} ${formatNumber(deepestPool.liquidity || 0)}` : '-', color: 'text-quantum-cyan' },
                { label: 'Liq/MCap Ratio', value: `${liquidityToMcapRatio.toFixed(2)}%`, color: 'text-gray-300' },
                // Top Movers
                { label: 'Top Gainer', value: topGainer ? `${topGainer.symbol} +${(topGainer.change24h || 0).toFixed(1)}%` : '-', color: 'text-green-400' },
                { label: 'Top Loser', value: topLoser ? `${topLoser.symbol} ${(topLoser.change24h || 0).toFixed(1)}%` : '-', color: 'text-red-400' },
                // Token Composition
                { label: 'Custom Tokens', value: `${customTokens}`, color: 'text-gray-300' },
                { label: 'Index Funds', value: `${indexTokens}`, color: 'text-purple-400' },
                { label: 'Perpetuals', value: `${perpTokens}`, color: 'text-orange-400' },
                { label: 'Bridge Pairs', value: `${bridgeTokens.length} (wBTC/wZEC/wIRON/wETH)`, color: 'text-amber-400' },
                { label: 'Bridge TVL', value: formatNumber(bridgeTVL), color: 'text-amber-400' },
                // Fees
                { label: 'Avg Buy Fee', value: `${avgBuyFee.toFixed(2)}%`, color: 'text-gray-300' },
                { label: 'Avg Sell Fee', value: `${avgSellFee.toFixed(2)}%`, color: 'text-gray-300' },
                // Supply
                { label: 'Avg Circ. Supply', value: `${avgCirculatingPct.toFixed(0)}%`, color: 'text-gray-300' },
                // Features
                { label: 'Quantum Secured', value: `${tokensWithFeatures.quantumSecured}`, color: 'text-quantum-cyan' },
                { label: 'Auto-Liquidity', value: `${tokensWithFeatures.autoLiquidity}`, color: 'text-gray-300' },
                { label: 'Buyback & Burn', value: `${tokensWithFeatures.buybackAndBurn}`, color: 'text-amber-400' },
              ];

              return (
                <div className="relative group">
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink rounded-2xl blur-lg opacity-20 group-hover:opacity-40 transition-opacity" />
                  <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-quantum-cyan/20 overflow-hidden">
                    {/* Header */}
                    <div className="flex items-center justify-between px-6 pt-5 pb-4 cursor-pointer" onClick={() => setIsAnalyticsCollapsed(!isAnalyticsCollapsed)}>
                      <h2 className="text-lg font-bold text-white flex items-center gap-2">
                        <Layers className="w-5 h-5 text-quantum-cyan" />
                        Quillon DEX Analytics
                      </h2>
                      <div className="flex items-center gap-3">
                        <span className="text-[10px] text-gray-500 font-mono">{new Date().toLocaleTimeString()}</span>
                        <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-green-500/10 border border-green-500/20">
                          <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                          <span className="text-[10px] text-green-400 font-medium">LIVE</span>
                        </div>
                        <button
                          onClick={(e) => { e.stopPropagation(); setIsAnalyticsCollapsed(!isAnalyticsCollapsed); }}
                          className="p-1 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                          title={isAnalyticsCollapsed ? 'Expand analytics' : 'Collapse analytics'}
                        >
                          {isAnalyticsCollapsed ? <ChevronRight className="w-4 h-4 text-gray-400" /> : <ChevronLeft className="w-4 h-4 text-gray-400" />}
                        </button>
                      </div>
                    </div>

                    {/* Headline Metrics - Large Cards */}
                    <AnimatePresence>
                    {!isAnalyticsCollapsed && (
                    <motion.div
                      initial={{ opacity: 0, width: 0 }}
                      animate={{ opacity: 1, width: '100%' }}
                      exit={{ opacity: 0, height: 0, overflow: 'hidden' }}
                      transition={{ duration: 0.3 }}
                    >
                    <div className="px-6 pb-4">
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                        {headlineMetrics.map((m, idx) => {
                          const Icon = m.icon;
                          return (
                            <motion.div
                              key={m.label}
                              initial={{ opacity: 0, scale: 0.95 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ delay: 0.04 * idx }}
                              className="relative overflow-hidden rounded-xl border border-white/[0.06] bg-white/[0.02] p-4 hover:bg-white/[0.04] transition-all"
                            >
                              <div className={`absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r ${m.gradient}`} />
                              <div className="flex items-center gap-2 mb-3">
                                <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${m.gradient} flex items-center justify-center shadow-lg`}>
                                  <Icon className="w-4 h-4 text-white" />
                                </div>
                                <span className="text-[11px] text-gray-400 font-medium">{m.label}</span>
                              </div>
                              <p className="text-xl font-bold text-white mb-1">{m.value}</p>
                              <p className="text-[11px] text-gray-500">{m.sub}</p>
                            </motion.div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Detailed Statistics Grid */}
                    <div className="border-t border-white/[0.04] px-6 py-4">
                      <div className="flex items-center gap-2 mb-3">
                        <BarChart3 className="w-4 h-4 text-gray-500" />
                        <span className="text-xs text-gray-400 font-medium uppercase tracking-wider">Detailed Statistics</span>
                      </div>
                      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-x-4 gap-y-2">
                        {detailMetrics.map((m, idx) => (
                          <div key={m.label} className="py-1.5">
                            <p className="text-[10px] text-gray-500 truncate">{m.label}</p>
                            <p className={`text-xs font-semibold truncate ${m.color}`}>{m.value}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Market Sentiment Bar */}
                    <div className="border-t border-white/[0.04] px-6 py-3">
                      <div className="flex items-center gap-3">
                        <span className="text-[10px] text-gray-500 font-medium whitespace-nowrap">24h Sentiment</span>
                        <div className="flex-1 h-2 rounded-full overflow-hidden bg-white/[0.03] flex">
                          {gainers > 0 && (
                            <div
                              className="h-full bg-gradient-to-r from-green-500 to-emerald-400 transition-all duration-700"
                              style={{ width: `${(gainers / Math.max(totalTokens, 1)) * 100}%` }}
                            />
                          )}
                          {unchanged > 0 && (
                            <div
                              className="h-full bg-gray-600 transition-all duration-700"
                              style={{ width: `${(unchanged / Math.max(totalTokens, 1)) * 100}%` }}
                            />
                          )}
                          {losers > 0 && (
                            <div
                              className="h-full bg-gradient-to-r from-red-400 to-rose-500 transition-all duration-700"
                              style={{ width: `${(losers / Math.max(totalTokens, 1)) * 100}%` }}
                            />
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-[10px]">
                          <span className="text-green-400 flex items-center gap-1">
                            <TrendingUp className="w-3 h-3" /> {gainers}
                          </span>
                          <span className="text-red-400 flex items-center gap-1">
                            <TrendingDown className="w-3 h-3" /> {losers}
                          </span>
                        </div>
                      </div>
                    </div>
                    </motion.div>
                    )}
                    </AnimatePresence>
                  </div>
                </div>
              );
            })()}
          </motion.div>

          {/* Swap + Token Table Row */}
          <div className="lg:col-span-4 flex gap-4">
          {/* Swap Interface - Collapses horizontally */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className={isSwapCollapsed ? "flex-shrink-0" : "w-full lg:w-1/2 flex-shrink-0"}
          >
              {isSwapCollapsed ? (
                /* Collapsed: narrow vertical sidebar */
                <div
                  className="bg-black/60 backdrop-blur-xl rounded-xl border border-quantum-cyan/20 px-0.5 py-2 flex flex-col items-center gap-2 cursor-pointer hover:border-quantum-cyan/40 transition-colors min-h-[140px] w-7"
                  onClick={() => setIsSwapCollapsed(false)}
                >
                  <ChevronRight className="w-3 h-3 text-quantum-cyan flex-shrink-0" />
                  <div className="flex-1 flex items-center">
                    <span className="text-white/80 font-bold text-[10px] [writing-mode:vertical-lr] rotate-180 tracking-[0.2em] uppercase">Swap</span>
                  </div>
                </div>
              ) : (
            <div className="relative group">
              {/* Glow effect */}
              <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity" />
              <div className="relative bg-black/60 backdrop-blur-xl rounded-2xl border border-quantum-cyan/20 p-6 space-y-4">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-xl font-bold text-white">Swap Tokens</h2>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setIsSettingsModalOpen(true)}
                      className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                    >
                      <Settings className="w-5 h-5 text-gray-400" />
                    </button>
                    <button
                      onClick={() => setIsSwapCollapsed(true)}
                      className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                      title="Collapse swap panel"
                    >
                      <ChevronLeft className="w-5 h-5 text-gray-400" />
                    </button>
                  </div>
                </div>

              {/* ── Market / Limit toggle ── */}
              <div className="flex gap-1 p-1 bg-white/5 rounded-xl">
                <button
                  onClick={() => setSpotMode('market')}
                  className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-all ${
                    spotMode === 'market'
                      ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-black'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Market
                </button>
                <button
                  onClick={() => setSpotMode('limit')}
                  className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-all ${
                    spotMode === 'limit'
                      ? 'bg-gradient-to-r from-amber-400 to-orange-500 text-black'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Limit
                </button>
              </div>

              {/* ══════ LIMIT ORDER FORM ══════ */}
              {spotMode === 'limit' && (
                <div className="space-y-4">
                  {/* Token pair — reuses swapFrom / swapTo so the selector works */}
                  <div className="flex items-center gap-2">
                    <div className="flex-1">
                      <div className="text-xs text-gray-500 mb-1">Sell</div>
                      <button
                        onClick={() => setIsFromTokenSelectorOpen(true)}
                        className="w-full py-3 px-4 rounded-xl bg-white/5 border border-white/10 text-white font-bold flex items-center justify-center gap-2 hover:border-quantum-cyan/30 transition-colors"
                      >
                        <TokenIcon symbol={swapFrom} icon={findToken(swapFrom)?.icon} logoUrl={findToken(swapFrom)?.logoUrl} size={20} />
                        <span>{swapFrom}</span>
                        <span className="text-xs opacity-50">▼</span>
                      </button>
                      <div className="text-xs text-gray-600 mt-1 text-right">
                        Balance: {(findToken(swapFrom)?.balance ?? 0).toFixed(4)}
                      </div>
                    </div>
                    <div className="text-gray-600 mt-4 text-lg">→</div>
                    <div className="flex-1">
                      <div className="text-xs text-gray-500 mb-1">Buy</div>
                      <button
                        onClick={() => setIsToTokenSelectorOpen(true)}
                        className="w-full py-3 px-4 rounded-xl bg-white/5 border border-white/10 text-white font-bold flex items-center justify-center gap-2 hover:border-quantum-cyan/30 transition-colors"
                      >
                        <TokenIcon symbol={swapTo} icon={findToken(swapTo)?.icon} logoUrl={findToken(swapTo)?.logoUrl} size={20} />
                        <span>{swapTo}</span>
                        <span className="text-xs opacity-50">▼</span>
                      </button>
                    </div>
                  </div>

                  {/* Amount to spend */}
                  <div>
                    <label className="text-sm text-gray-400">Amount ({swapFrom})</label>
                    <div className="relative mt-1">
                      <input
                        type="number"
                        value={limitAmount}
                        onChange={e => setLimitAmount(e.target.value)}
                        placeholder="0.0"
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 pr-16 text-white text-lg focus:outline-none focus:border-amber-400/50 transition-colors"
                      />
                      <button
                        onClick={() => {
                          const t = findToken(swapFrom);
                          if (t) setLimitAmount(t.balance.toString());
                        }}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-amber-400 hover:text-amber-300 font-bold"
                      >
                        MAX
                      </button>
                    </div>
                  </div>

                  {/* Order direction */}
                  <div>
                    <label className="text-sm text-gray-400 mb-1 block">Trigger Condition</label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => setLimitDirection('below')}
                        className={`py-2.5 rounded-xl text-sm font-semibold border transition-all ${
                          limitDirection === 'below'
                            ? 'bg-quantum-green/20 border-quantum-green/50 text-quantum-green'
                            : 'bg-white/5 border-white/10 text-gray-400 hover:text-white'
                        }`}
                      >
                        Buy when price ↓ below
                      </button>
                      <button
                        onClick={() => setLimitDirection('above')}
                        className={`py-2.5 rounded-xl text-sm font-semibold border transition-all ${
                          limitDirection === 'above'
                            ? 'bg-red-500/20 border-red-500/50 text-red-400'
                            : 'bg-white/5 border-white/10 text-gray-400 hover:text-white'
                        }`}
                      >
                        Sell when price ↑ above
                      </button>
                    </div>
                  </div>

                  {/* Trigger price */}
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-sm text-gray-400">Trigger Price (USD)</label>
                      {(() => {
                        const t = findToken(swapFrom);
                        const p = t?.price;
                        return p && p > 0 ? (
                          <button
                            onClick={() => setLimitTriggerPrice(p.toFixed(6))}
                            className="text-xs text-amber-400/70 hover:text-amber-400 transition-colors"
                          >
                            Now: ${p < 0.001 ? p.toExponential(3) : p.toFixed(6)} (use)
                          </button>
                        ) : null;
                      })()}
                    </div>
                    <input
                      type="number"
                      value={limitTriggerPrice}
                      onChange={e => setLimitTriggerPrice(e.target.value)}
                      placeholder="0.000000"
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white text-lg focus:outline-none focus:border-amber-400/50 transition-colors"
                    />
                  </div>

                  {/* Slippage + Expiry */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-gray-500 mb-1 block">Max Slippage</label>
                      <div className="flex gap-1">
                        {['1', '3', '5'].map(v => (
                          <button
                            key={v}
                            onClick={() => setLimitSlippage(v)}
                            className={`flex-1 py-1.5 rounded-lg text-xs font-bold border transition-all ${
                              limitSlippage === v
                                ? 'bg-amber-400/20 border-amber-400/40 text-amber-300'
                                : 'bg-white/5 border-white/10 text-gray-400 hover:text-white'
                            }`}
                          >
                            {v}%
                          </button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="text-xs text-gray-500 mb-1 block">Expiry</label>
                      <select
                        value={limitExpiry}
                        onChange={e => setLimitExpiry(e.target.value as any)}
                        className="w-full bg-gray-900 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-amber-400/50"
                      >
                        <option value="gtc">GTC (no expiry)</option>
                        <option value="1h">1 Hour</option>
                        <option value="24h">24 Hours</option>
                        <option value="7d">7 Days</option>
                      </select>
                    </div>
                  </div>

                  {/* Order summary */}
                  {limitAmount && limitTriggerPrice && (
                    <div className="p-3 rounded-xl bg-amber-400/5 border border-amber-400/20 text-xs text-gray-300 space-y-1">
                      <div className="flex justify-between">
                        <span className="text-gray-500">You spend</span>
                        <span>{parseFloat(limitAmount).toFixed(4)} {swapFrom}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Trigger</span>
                        <span className={limitDirection === 'below' ? 'text-quantum-green' : 'text-red-400'}>
                          {swapFrom} {limitDirection === 'below' ? '≤' : '≥'} ${parseFloat(limitTriggerPrice).toFixed(6)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Slippage</span>
                        <span>{limitSlippage}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Expiry</span>
                        <span>{limitExpiry === 'gtc' ? 'Good Till Cancelled' : limitExpiry}</span>
                      </div>
                    </div>
                  )}

                  {/* Feedback */}
                  {limitOrderMsg && (
                    <div className={`p-3 rounded-xl text-sm font-medium ${
                      limitOrderMsg.type === 'success'
                        ? 'bg-quantum-green/10 border border-quantum-green/30 text-quantum-green'
                        : 'bg-red-500/10 border border-red-500/30 text-red-400'
                    }`}>
                      {limitOrderMsg.text}
                    </div>
                  )}

                  {/* Place Order button */}
                  <button
                    onClick={placeDexLimitOrder}
                    disabled={isPlacingLimitOrder || !limitAmount || !limitTriggerPrice || parseFloat(limitAmount) <= 0 || parseFloat(limitTriggerPrice) <= 0}
                    className={`w-full py-4 rounded-xl font-bold text-black transition-all ${
                      isPlacingLimitOrder || !limitAmount || !limitTriggerPrice
                        ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                        : 'bg-gradient-to-r from-amber-400 to-orange-500 hover:shadow-lg hover:shadow-amber-400/30'
                    }`}
                  >
                    {isPlacingLimitOrder ? (
                      <span className="flex items-center justify-center gap-2">
                        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        Placing Order…
                      </span>
                    ) : (
                      `Place ${limitDirection === 'below' ? 'Buy' : 'Sell'} Limit Order`
                    )}
                  </button>

                  {/* My Limit Orders toggle */}
                  <button
                    onClick={() => { setShowLimitOrdersPanel(v => !v); fetchDexLimitOrders(); }}
                    className="w-full py-2 bg-gray-800/50 border border-amber-400/20 rounded-xl text-sm text-amber-400/70 hover:text-amber-400 hover:bg-gray-700/50 transition-all flex items-center justify-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    My Limit Orders ({dexLimitOrders.filter(o => o.status === 'open').length} open)
                    <svg className={`w-4 h-4 transition-transform ${showLimitOrdersPanel ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {/* Limit Orders Panel */}
                  {showLimitOrdersPanel && (
                    <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
                      {dexLimitOrders.length === 0 ? (
                        <div className="text-center text-gray-600 text-sm py-4">No limit orders yet</div>
                      ) : (
                        dexLimitOrders
                          .slice()
                          .sort((a, b) => b.created_at - a.created_at)
                          .map((order: any) => (
                            <div
                              key={order.id}
                              className={`p-3 rounded-xl border text-xs ${
                                order.status === 'open'
                                  ? 'bg-amber-400/5 border-amber-400/20'
                                  : order.status === 'filled'
                                  ? 'bg-quantum-green/5 border-quantum-green/20'
                                  : 'bg-gray-800/40 border-gray-700/30'
                              }`}
                            >
                              <div className="flex items-center justify-between mb-1.5">
                                <div className="flex items-center gap-2">
                                  <span className={`px-1.5 py-0.5 rounded font-bold ${
                                    order.status === 'open' ? 'bg-amber-400/20 text-amber-300' :
                                    order.status === 'filled' ? 'bg-quantum-green/20 text-quantum-green' :
                                    order.status === 'processing' ? 'bg-blue-400/20 text-blue-300' :
                                    'bg-gray-700 text-gray-500'
                                  }`}>
                                    {order.status?.toUpperCase()}
                                  </span>
                                  <span className="text-gray-300 font-mono">
                                    {order.from_token} → {order.to_token}
                                  </span>
                                </div>
                                {order.status === 'open' && (
                                  <button
                                    onClick={() => cancelDexLimitOrder(order.id)}
                                    className="text-gray-600 hover:text-red-400 transition-colors text-xs px-2 py-0.5 rounded border border-gray-700/50 hover:border-red-400/30"
                                  >
                                    Cancel
                                  </button>
                                )}
                              </div>
                              <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-gray-500">
                                <span>Amount</span>
                                <span className="text-gray-300 font-mono text-right">
                                  {(Number(order.amount) / 1e24).toFixed(4)} {order.from_token}
                                </span>
                                <span>Trigger</span>
                                <span className={`font-mono text-right ${order.direction === 'below' ? 'text-quantum-green' : 'text-red-400'}`}>
                                  {order.direction === 'below' ? '≤' : '≥'} ${Number(order.trigger_price).toFixed(6)}
                                </span>
                                {order.status === 'filled' && order.fill_price && (
                                  <>
                                    <span>Filled at</span>
                                    <span className="text-quantum-green font-mono text-right">${Number(order.fill_price).toFixed(6)}</span>
                                  </>
                                )}
                                {order.status === 'filled' && order.amount_out > 0 && (
                                  <>
                                    <span>Received</span>
                                    <span className="text-quantum-green font-mono text-right">
                                      {(Number(order.amount_out) / 1e24).toFixed(4)} {order.to_token}
                                    </span>
                                  </>
                                )}
                                <span>Created</span>
                                <span className="text-right">{new Date(order.created_at).toLocaleString()}</span>
                              </div>
                            </div>
                          ))
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* ══════ MARKET SWAP FORM ══════ */}
              <div className={`space-y-4${spotMode === 'limit' ? ' hidden' : ''}`}>
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
                    className="absolute right-2 top-1/2 -translate-y-1/2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/30 rounded-lg px-3 py-2 text-white font-bold cursor-pointer focus:outline-none transition-colors flex items-center gap-2"
                  >
                    {/* Proper Logo for QUG */}
                    <TokenIcon symbol={swapFrom} icon={findToken(swapFrom)?.icon} logoUrl={findToken(swapFrom)?.logoUrl} size={24} />
                    <span>{swapFrom}</span>
                    <span className="text-xs opacity-70">▼</span>
                  </button>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-500">
                    Balance: {(Number(findToken(swapFrom)?.balance) || 0).toFixed(4)}
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
                        console.log(`🔢 [DEX v3.6.9] ${percentage}% clicked:`, {
                          swapFrom,
                          fromToken: fromToken ? { symbol: fromToken.symbol, balance: fromToken.balance } : null,
                          calculatedAmount: fromToken ? fromToken.balance * (percentage / 100) : 'N/A'
                        });
                        if (fromToken) {
                          const amount = fromToken.balance * (percentage / 100);
                          setSwapAmount(amount.toFixed(8));
                          console.log(`✅ [DEX v3.6.9] Set swapAmount to:`, amount.toFixed(8));
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

                      // v2.3.33-beta: FIX - Use pool reserves for accurate estimated output
                      const formatTokenForBackend = (tokenId: string): string => {
                        if (tokenId === 'native-qug') return 'QUG';
                        if (tokenId === 'qugusd-stable') return 'QUGUSD';
                        // Hex token address (64 chars = 32 bytes): decode ASCII symbol
                        // e.g. "7742544300...00" → "wBTC"
                        if (/^[0-9a-fA-F]{64}$/.test(tokenId)) {
                          const trimmed = tokenId.replace(/00+$/, '');
                          let sym = '';
                          for (let i = 0; i + 1 < trimmed.length; i += 2) {
                            const byte = parseInt(trimmed.substring(i, i + 2), 16);
                            if (byte >= 32 && byte < 127) sym += String.fromCharCode(byte);
                          }
                          if (sym.length > 0) return sym;
                        }
                        return tokenId;
                      };

                      const fromTokenFormatted = formatTokenForBackend(fromToken.id);
                      const toTokenFormatted = formatTokenForBackend(toToken.id);

                      // Find matching liquidity pool
                      const matchingPool = liquidityPools.find(pool => {
                        const pool0Upper = pool.token0.toUpperCase();
                        const pool1Upper = pool.token1.toUpperCase();
                        const fromUpper = fromTokenFormatted.toUpperCase();
                        const toUpper = toTokenFormatted.toUpperCase();
                        return (pool0Upper === fromUpper && pool1Upper === toUpper) ||
                               (pool0Upper === toUpper && pool1Upper === fromUpper);
                      });

                      if (matchingPool) {
                        // Use AMM formula: amount_out = (amount_in * reserve_out) / (reserve_in + amount_in)
                        const isForward = matchingPool.token0.toUpperCase() === fromTokenFormatted.toUpperCase();
                        const reserveIn = (isForward ? parseU128(matchingPool.reserve0) : parseU128(matchingPool.reserve1)) / 1e24;
                        const reserveOut = (isForward ? parseU128(matchingPool.reserve1) : parseU128(matchingPool.reserve0)) / 1e24;
                        // v2.4.0: Add NaN protection for zero reserves
                        if (reserveIn <= 0 || reserveOut <= 0) {
                          return '0.0000';
                        }
                        const amountIn = parseFloat(swapAmount) * 0.997; // Apply 0.3% fee
                        const amountOut = (amountIn * reserveOut) / (reserveIn + amountIn);
                        return isFinite(amountOut) && !isNaN(amountOut) ? amountOut.toFixed(4) : '0.0000';
                      }

                      // Fallback to oracle prices for native token pairs (QUG/QUGUSD)
                      // v2.4.0: Add NaN protection
                      const fromPrice = fromToken.price || 1;
                      const toPrice = toToken.price || 1;
                      const exchangeRate = (fromPrice / toPrice) * 0.997;
                      const result = parseFloat(swapAmount) * exchangeRate;
                      return isFinite(result) && !isNaN(result) ? result.toFixed(4) : '0.0000';
                    })()}
                    placeholder="0.0"
                    readOnly
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-4 pr-36 text-white text-xl focus:outline-none"
                  />
                  <button
                    onClick={() => setIsToTokenSelectorOpen(true)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/30 rounded-lg px-3 py-2 text-white font-bold cursor-pointer focus:outline-none transition-colors flex items-center gap-2"
                  >
                    <TokenIcon symbol={swapTo} icon={findToken(swapTo)?.icon} logoUrl={findToken(swapTo)?.logoUrl} size={24} />
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

                      // v2.3.33-beta: FIX - Use pool reserves for accurate exchange rate
                      // Oracle prices are in different units (USD vs QUG) which causes wrong rates
                      const formatTokenForBackend = (tokenId: string): string => {
                        if (tokenId === 'native-qug') return 'QUG';
                        if (tokenId === 'qugusd-stable') return 'QUGUSD';
                        // Hex token address (64 chars = 32 bytes): decode ASCII symbol
                        // e.g. "7742544300...00" → "wBTC"
                        if (/^[0-9a-fA-F]{64}$/.test(tokenId)) {
                          const trimmed = tokenId.replace(/00+$/, '');
                          let sym = '';
                          for (let i = 0; i + 1 < trimmed.length; i += 2) {
                            const byte = parseInt(trimmed.substring(i, i + 2), 16);
                            if (byte >= 32 && byte < 127) sym += String.fromCharCode(byte);
                          }
                          if (sym.length > 0) return sym;
                        }
                        return tokenId;
                      };

                      const fromTokenFormatted = formatTokenForBackend(fromToken.id);
                      const toTokenFormatted = formatTokenForBackend(toToken.id);

                      // Find matching liquidity pool
                      const matchingPool = liquidityPools.find(pool => {
                        const pool0Upper = pool.token0.toUpperCase();
                        const pool1Upper = pool.token1.toUpperCase();
                        const fromUpper = fromTokenFormatted.toUpperCase();
                        const toUpper = toTokenFormatted.toUpperCase();
                        return (pool0Upper === fromUpper && pool1Upper === toUpper) ||
                               (pool0Upper === toUpper && pool1Upper === fromUpper);
                      });

                      if (matchingPool) {
                        // Calculate exchange rate from pool reserves (accurate!)
                        const isForward = matchingPool.token0.toUpperCase() === fromTokenFormatted.toUpperCase();
                        const reserveIn = isForward ? parseU128(matchingPool.reserve0) : parseU128(matchingPool.reserve1);
                        const reserveOut = isForward ? parseU128(matchingPool.reserve1) : parseU128(matchingPool.reserve0);
                        // v2.4.0: Add NaN protection for zero reserves
                        if (reserveIn <= 0 || reserveOut <= 0) {
                          return '0.00';
                        }
                        // Apply 0.3% fee to get realistic rate
                        const exchangeRate = (reserveOut * 0.997) / reserveIn;
                        return isFinite(exchangeRate) && !isNaN(exchangeRate) ? exchangeRate.toFixed(2) : '0.00';
                      }

                      // Fallback to oracle prices for native token pairs (QUG/QUGUSD)
                      // v2.4.0: Add NaN protection
                      const fromPrice = fromToken.price || 1;
                      const toPrice = toToken.price || 1;
                      const exchangeRate = fromPrice / toPrice;
                      return isFinite(exchangeRate) && !isNaN(exchangeRate) ? exchangeRate.toFixed(2) : '1.00';
                    })()} {swapTo}
                  </span>
                </div>
                <div className="flex justify-between text-gray-400">
                  <span>Slippage Tolerance</span>
                  <span className="text-white">{dexSettings.slippageTolerance}%</span>
                </div>
                <div className="flex justify-between text-gray-400">
                  <span>Fee</span>
                  <span className="text-white">0.3%</span>
                </div>
                {/* v5.1.0: Deal Value in USD */}
                {(() => {
                  const fromToken = findToken(swapFrom);
                  const toToken = findToken(swapTo);
                  if (!fromToken || !toToken || !swapAmount || parseFloat(swapAmount) <= 0) return null;
                  const payUsd = parseFloat(swapAmount) * (fromToken.price || 0);
                  if (payUsd <= 0) return null;
                  // Calculate expected output using AMM formula
                  const fmtId = (id: string) => id === 'native-qug' ? 'QUG' : id === 'qugusd-stable' ? 'QUGUSD' : id;
                  const fromFmt = fmtId(fromToken.id);
                  const toFmt = fmtId(toToken.id);
                  const pool = liquidityPools.find(p => {
                    const p0 = p.token0.toUpperCase(), p1 = p.token1.toUpperCase();
                    const f = fromFmt.toUpperCase(), t = toFmt.toUpperCase();
                    const fS = fromToken.symbol.toUpperCase(), tS = toToken.symbol.toUpperCase();
                    return (p0 === f && p1 === t) || (p0 === t && p1 === f) ||
                           (p0 === fS && p1 === tS) || (p0 === tS && p1 === fS);
                  });
                  let receiveUsd = payUsd * 0.997; // default: same minus fee
                  if (pool) {
                    const isFwd = pool.token0.toUpperCase() === fromFmt.toUpperCase() || pool.token0.toUpperCase() === fromToken.symbol.toUpperCase();
                    const rIn = (isFwd ? parseU128(pool.reserve0) : parseU128(pool.reserve1)) / 1e24;
                    const rOut = (isFwd ? parseU128(pool.reserve1) : parseU128(pool.reserve0)) / 1e24;
                    if (rIn > 0 && rOut > 0) {
                      const aIn = parseFloat(swapAmount) * 0.997;
                      const aOut = (aIn * rOut) / (rIn + aIn);
                      receiveUsd = aOut * (toToken.price || 0);
                    }
                  }
                  const fmtUsd = (v: number) => v >= 1_000_000 ? `$${(v/1_000_000).toFixed(2)}M` : v >= 1_000 ? `$${(v/1_000).toFixed(2)}K` : `$${v.toFixed(2)}`;
                  return (
                    <>
                      <div className="flex justify-between text-gray-400 pt-1 border-t border-white/5">
                        <span className="flex items-center gap-1"><DollarSign className="w-3 h-3" />You Pay</span>
                        <span className="text-white font-medium">{fmtUsd(payUsd)}</span>
                      </div>
                      <div className="flex justify-between text-gray-400">
                        <span className="flex items-center gap-1"><DollarSign className="w-3 h-3" />You Receive</span>
                        <span className="text-green-400 font-medium">≈ {fmtUsd(receiveUsd)}</span>
                      </div>
                      {/* v10.3.6: Sanity check — warn if output value is unrealistic */}
                      {receiveUsd > 1_000_000 && (
                        <div className="p-3 rounded-lg border bg-red-500/10 border-red-500/30 mt-2">
                          <div className="text-xs font-bold text-red-400">
                            ⚠️ UNREALISTIC SWAP VALUE
                          </div>
                          <div className="text-xs text-gray-400 mt-1">
                            The estimated output exceeds $1M. This swap will likely fail or give you far less than quoted.
                            The pool does not have sufficient liquidity for this trade. Try a much smaller amount.
                          </div>
                        </div>
                      )}
                      {payUsd > 0 && receiveUsd > 0 && Math.abs(receiveUsd - payUsd) / payUsd > 0.5 && (
                        <div className="p-3 rounded-lg border bg-yellow-500/10 border-yellow-500/30 mt-1">
                          <div className="text-xs font-bold text-yellow-400">
                            ⚠️ Large price discrepancy
                          </div>
                          <div className="text-xs text-gray-400 mt-1">
                            You pay {fmtUsd(payUsd)} but receive {fmtUsd(receiveUsd)} — a {((Math.abs(receiveUsd - payUsd) / payUsd) * 100).toFixed(0)}% difference.
                          </div>
                        </div>
                      )}
                    </>
                  );
                })()}
                {/* v4.0.1: Show price impact to user */}
                {(() => {
                  if (!swapAmount || parseFloat(swapAmount) <= 0) return null;
                  const fromToken = findToken(swapFrom);
                  const toToken = findToken(swapTo);
                  if (!fromToken || !toToken) return null;
                  const formatTokenForBackend = (tokenId: string): string => {
                    if (tokenId === 'native-qug') return 'QUG';
                    if (tokenId === 'qugusd-stable') return 'QUGUSD';
                    if (/^[0-9a-fA-F]{64}$/.test(tokenId)) {
                      const trimmed = tokenId.replace(/00+$/, '');
                      let sym = '';
                      for (let i = 0; i + 1 < trimmed.length; i += 2) {
                        const byte = parseInt(trimmed.substring(i, i + 2), 16);
                        if (byte >= 32 && byte < 127) sym += String.fromCharCode(byte);
                      }
                      if (sym.length > 0) return sym;
                    }
                    return tokenId;
                  };
                  const fromFormatted = formatTokenForBackend(fromToken.id);
                  const toFormatted = formatTokenForBackend(toToken.id);
                  // v4.0.2: Also match by symbol for custom tokens (pool may use address or symbol)
                  const matchingPool = liquidityPools.find(pool => {
                    const p0 = pool.token0.toUpperCase();
                    const p1 = pool.token1.toUpperCase();
                    const f = fromFormatted.toUpperCase();
                    const t = toFormatted.toUpperCase();
                    const fSym = fromToken.symbol.toUpperCase();
                    const tSym = toToken.symbol.toUpperCase();
                    return (p0 === f && p1 === t) || (p0 === t && p1 === f) ||
                           (p0 === fSym && p1 === tSym) || (p0 === tSym && p1 === fSym);
                  });
                  if (!matchingPool) return null;
                  const isForward = matchingPool.token0.toUpperCase() === fromFormatted.toUpperCase() ||
                                    matchingPool.token0.toUpperCase() === fromToken.symbol.toUpperCase();
                  // v4.0.10: ALL pool reserves use 24 decimals internally
                  const reserveIn = (isForward ? parseU128(matchingPool.reserve0) : parseU128(matchingPool.reserve1)) / 1e24;
                  if (reserveIn <= 0) return null;
                  const amountIn = parseFloat(swapAmount);
                  const impact = (amountIn / (reserveIn + amountIn)) * 100;
                  const reserveOut = (isForward ? parseU128(matchingPool.reserve1) : parseU128(matchingPool.reserve0)) / 1e24;
                  const amountOutEstimate = (amountIn * 0.997 * reserveOut) / (reserveIn + amountIn * 0.997);
                  const wouldDrainPool = amountOutEstimate >= reserveOut * 0.95;
                  const impactColor = impact > 50 ? 'text-red-500 font-bold' : impact > 10 ? 'text-red-400' : impact > 3 ? 'text-yellow-400' : 'text-green-400';
                  return (
                    <>
                      <div className="flex justify-between text-gray-400">
                        <span>Price Impact</span>
                        <span className={impactColor}>{impact < 0.01 ? '<0.01' : impact.toFixed(2)}%</span>
                      </div>
                      {impact > 10 && (
                        <div className={`p-3 rounded-lg border ${impact > 50 ? 'bg-red-500/10 border-red-500/30' : 'bg-yellow-500/10 border-yellow-500/30'}`}>
                          <div className={`text-xs font-bold ${impact > 50 ? 'text-red-400' : 'text-yellow-400'}`}>
                            {impact > 50 ? '⚠️ EXTREME PRICE IMPACT' : '⚠️ High Price Impact'}
                          </div>
                          <div className="text-xs text-gray-400 mt-1">
                            {wouldDrainPool
                              ? `This swap would drain ${(amountOutEstimate / reserveOut * 100).toFixed(0)}% of the pool. You will receive far less than the quoted price.`
                              : `You are swapping a large amount relative to pool liquidity. Consider reducing the amount.`
                            }
                          </div>
                          {reserveOut > 0 && (
                            <div className="text-xs text-gray-500 mt-1">
                              Pool liquidity: {reserveOut < 1000 ? reserveOut.toFixed(2) : reserveOut.toLocaleString(undefined, {maximumFractionDigits: 0})} {swapTo}
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  );
                })()}
                {/* v2.4.0: Price discrepancy warning when AMM rate differs from oracle */}
                {(() => {
                  const fromToken = findToken(swapFrom);
                  const toToken = findToken(swapTo);
                  if (!fromToken || !toToken || fromToken.price <= 0 || toToken.price <= 0) return null;

                  // Calculate oracle-implied rate
                  const oracleRate = fromToken.price / toToken.price;

                  // Calculate AMM rate from pool
                  const formatTokenForBackend = (tokenId: string): string => {
                    if (tokenId === 'native-qug') return 'QUG';
                    if (tokenId === 'qugusd-stable') return 'QUGUSD';
                    if (/^[0-9a-fA-F]{64}$/.test(tokenId)) {
                      const trimmed = tokenId.replace(/00+$/, '');
                      let sym = '';
                      for (let i = 0; i + 1 < trimmed.length; i += 2) {
                        const byte = parseInt(trimmed.substring(i, i + 2), 16);
                        if (byte >= 32 && byte < 127) sym += String.fromCharCode(byte);
                      }
                      if (sym.length > 0) return sym;
                    }
                    return tokenId;
                  };
                  const fromFormatted = formatTokenForBackend(fromToken.id);
                  const toFormatted = formatTokenForBackend(toToken.id);

                  const pool = liquidityPools.find(p => {
                    const p0 = p.token0.toUpperCase();
                    const p1 = p.token1.toUpperCase();
                    const f = fromFormatted.toUpperCase();
                    const t = toFormatted.toUpperCase();
                    return (p0 === f && p1 === t) || (p0 === t && p1 === f);
                  });

                  if (!pool) return null;

                  const isForward = pool.token0.toUpperCase() === fromFormatted.toUpperCase();
                  const reserveIn = isForward ? parseU128(pool.reserve0) : parseU128(pool.reserve1);
                  const reserveOut = isForward ? parseU128(pool.reserve1) : parseU128(pool.reserve0);
                  if (reserveIn <= 0) return null;

                  const ammRate = reserveOut / reserveIn;

                  // Check if rates differ by more than 20%
                  const ratioDiff = Math.abs(ammRate - oracleRate) / oracleRate;
                  if (ratioDiff < 0.2) return null;

                  const isBetterDeal = ammRate > oracleRate;

                  return (
                    <div className={`mt-2 p-3 rounded-lg text-xs ${isBetterDeal ? 'bg-green-500/10 border border-green-500/30' : 'bg-amber-900/30 border border-amber-500/40'}`}>
                      <div className="flex items-center gap-1.5">
                        <span>{isBetterDeal ? '🎉' : '⚠️'}</span>
                        <span className={isBetterDeal ? 'text-green-300 font-medium' : 'text-white font-medium'}>
                          {isBetterDeal
                            ? `Pool rate is ${((ammRate/oracleRate - 1) * 100).toFixed(0)}% better than market!`
                            : `Pool rate is ${((1 - ammRate/oracleRate) * 100).toFixed(0)}% worse than market price`
                          }
                        </span>
                      </div>
                      <div className="text-gray-300 mt-1">
                        Market: 1 {swapFrom} = {oracleRate.toFixed(2)} {swapTo} | Pool: {ammRate.toFixed(2)} {swapTo}
                      </div>
                    </div>
                  );
                })()}
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

                        // v2.3.9-beta: Update cachedQugusdBalance for USD→QUGUSD conversion
                        const currentQugusd = tokens.find(t => t.symbol === 'QUGUSD');
                        if (currentQugusd) {
                          const newQugusdBalance = currentQugusd.balance + qugusdAmount;
                          localStorage.setItem('cachedQugusdBalance', newQugusdBalance.toString());
                          console.log(`💾 [DEX] Updated cachedQugusdBalance (USD→QUGUSD): ${currentQugusd.balance} -> ${newQugusdBalance}`);
                        }

                        // v2.3.6-beta: Update local state for USD→QUGUSD
                        setTokens(prevTokens => prevTokens.map(token => {
                          if (token.symbol === 'USD') {
                            return { ...token, balance: Math.max(0, token.balance - parseFloat(swapAmount)) };
                          }
                          if (token.symbol === 'QUGUSD') {
                            return { ...token, balance: token.balance + qugusdAmount };
                          }
                          return token;
                        }));

                        // v2.3.28-beta: Set cooldown flag
                        swapJustCompletedRef.current = true;
                        localStorage.setItem('dexCooldownUntil', (Date.now() + 15000).toString());
                        console.log('🔒 [DEX] USD→QUGUSD completed - cooldown for 15s');
                        setTimeout(() => {
                          swapJustCompletedRef.current = false;
                          localStorage.removeItem('dexCooldownUntil');
                        }, 15000);
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
                        const reserveIn = isForward ? parseU128(matchingPool.reserve0) / 1e24 : parseU128(matchingPool.reserve1) / 1e24;
                        const reserveOut = isForward ? parseU128(matchingPool.reserve1) / 1e24 : parseU128(matchingPool.reserve0) / 1e24;

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

                      // v2.3.30-beta: Set cooldown AND preliminary locked balance BEFORE API call
                      swapJustCompletedRef.current = true;
                      localStorage.setItem('dexCooldownUntil', (Date.now() + 15000).toString());

                      // Set preliminary locked balance for QUG if that's the target
                      if (swapTo === 'QUG') {
                        const currentQug = tokens.find(t => t.symbol === 'QUG');
                        const preliminaryQugBalance = (currentQug?.balance || 0) + expectedOutput;
                        localStorage.setItem('dexLockedBalance', preliminaryQugBalance.toString());
                        console.log('🔒 [DEX] PRE-SWAP (USD→QUG): Set PRELIMINARY locked balance:', preliminaryQugBalance);
                      }

                      // v4.0.13: Send amounts as strings to preserve u128 precision
                      const qugusdAmountBigInt = parseAmountToBigInt(qugusdAmount.toString(), 24);
                      const minOutputBigInt = parseAmountToBigInt(minOutput.toString(), 24);
                      const swapResponse = await qnkAPI.executeSwap({
                        from_token: 'QUGUSD',
                        to_token: toTokenFormatted,
                        amount_in: qugusdAmountBigInt.toString(),
                        min_amount_out: minOutputBigInt.toString(),
                        wallet_address: walletAddress
                      });

                      if (swapResponse.success && swapResponse.data) {
                        const outputAmount = swapResponse.data.amount_out / 1e24;
                        setSwapSuccessData({
                          fromToken: 'USD',
                          toToken: swapTo,
                          fromAmount: parseFloat(swapAmount),
                          toAmount: outputAmount,
                          transactionHash: swapResponse.data.transaction_id
                        });
                        setShowSwapSuccess(true);
                        setSwapAmount('');

                        // v2.3.30-beta: Update local state and LOCKED balance for USD→Token swap
                        if (swapTo === 'QUG') {
                          const currentQug = tokens.find(t => t.symbol === 'QUG');
                          if (currentQug) {
                            const newQugBalance = currentQug.balance + outputAmount;
                            safeCacheBalance(newQugBalance);
                            localStorage.setItem('dexLockedBalance', newQugBalance.toString()); // Update with actual amount
                            console.log(`🔒 [DEX] Updated LOCKED balance (USD→QUG): ${currentQug.balance} -> ${newQugBalance}`);
                          }
                        }

                        setTokens(prevTokens => prevTokens.map(token => {
                          if (token.symbol === 'USD') {
                            return { ...token, balance: Math.max(0, token.balance - parseFloat(swapAmount)) };
                          }
                          if (token.symbol === swapTo) {
                            return { ...token, balance: token.balance + outputAmount };
                          }
                          return token;
                        }));

                        // v2.3.30-beta: Refresh cooldown
                        localStorage.setItem('dexCooldownUntil', (Date.now() + 15000).toString());
                        console.log('🔒 [DEX] USD→Token completed - refreshed cooldown for 15s');
                        setTimeout(() => {
                          swapJustCompletedRef.current = false;
                          localStorage.removeItem('dexCooldownUntil');
                          localStorage.removeItem('dexLockedBalance');
                        }, 15000);
                      } else {
                        // v2.3.30-beta: Clear cooldown on failure
                        swapJustCompletedRef.current = false;
                        localStorage.removeItem('dexCooldownUntil');
                        localStorage.removeItem('dexLockedBalance');
                        alert(`❌ Swap failed after USD conversion: ${swapResponse.error || 'Unknown error'}\n\nYour USD was converted to QUGUSD but the swap failed.`);
                      }
                    } catch (error) {
                      // v2.3.30-beta: Clear cooldown on exception
                      swapJustCompletedRef.current = false;
                      localStorage.removeItem('dexCooldownUntil');
                      localStorage.removeItem('dexLockedBalance');
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
                    alert(`Insufficient ${swapFrom} balance. You have ${(Number(fromToken.balance) || 0).toFixed(4)}`);
                    return;
                  }

                  // Helper function to format token ID for backend
                  const formatTokenForBackend = (tokenId: string): string => {
                    if (tokenId === 'native-qug') return 'QUG';
                    if (tokenId === 'qugusd-stable') return 'QUGUSD';
                    // Bridge wrapped tokens store their symbol as ASCII hex (32-byte zero-padded)
                    // e.g. "7742544300...00" → "wBTC". Decode before sending to backend.
                    if (/^[0-9a-fA-F]{64}$/.test(tokenId)) {
                      const trimmed = tokenId.replace(/00+$/, '');
                      let sym = '';
                      for (let i = 0; i + 1 < trimmed.length; i += 2) {
                        const byte = parseInt(trimmed.substring(i, i + 2), 16);
                        if (byte >= 32 && byte < 127) sym += String.fromCharCode(byte);
                      }
                      if (sym.length > 0) return sym;
                    }
                    // Custom contract tokens: backend expects address as-is (with qnk prefix)
                    return tokenId;
                  };

                  // v8.2.8: Start swap loading animation
                  setIsSwapping(true);
                  setSwapPhase('verifying');

                  try{

                    // ✅ PROPER FIX: Calculate expected output using constant product formula (x * y = k)
                    // Find matching liquidity pool
                    const fromTokenFormatted = formatTokenForBackend(fromToken.id);
                    const toTokenFormatted = formatTokenForBackend(toToken.id);

                    // v4.5.0: Re-fetch pools right before swap to get fresh reserves
                    // Stale reserves cause slippage errors on low-liquidity pools
                    let freshPools = liquidityPools;
                    try {
                      const freshPoolsResp = await qnkAPI.getLiquidityPools();
                      if (freshPoolsResp.success && freshPoolsResp.data) {
                        freshPools = freshPoolsResp.data;
                        console.log('🔄 [DEX v4.5.0] Re-fetched fresh pool reserves before swap');
                      }
                    } catch (e) {
                      console.warn('⚠️ Could not refresh pools, using cached reserves');
                    }

                    const matchingPool = freshPools.find((pool: any) => {
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

                      // v4.0.10: CRITICAL FIX - ALL pool reserves are stored in 24-decimal format internally.
                      // The AMM stores reserves * 1e24 regardless of the token's official decimals.
                      // Using token-specific decimals here caused 10^16x errors for 8-decimal custom tokens.
                      const POOL_DECIMALS = 24; // ALL reserves use 24 decimals
                      const poolDivisor = Math.pow(10, POOL_DECIMALS);

                      const reserveIn = isForward
                        ? parseU128(matchingPool.reserve0) / poolDivisor
                        : parseU128(matchingPool.reserve1) / poolDivisor;
                      const reserveOut = isForward
                        ? parseU128(matchingPool.reserve1) / poolDivisor
                        : parseU128(matchingPool.reserve0) / poolDivisor;

                      console.log(`🔢 [DEX v4.0.10] Pool reserves (24 decimals): reserveIn=${reserveIn}, reserveOut=${reserveOut}`);

                      // v2.4.0: Add NaN protection for zero reserves
                      if (reserveIn <= 0 || reserveOut <= 0) {
                        setIsSwapping(false); setSwapPhase('idle');
                        alert('Pool has insufficient liquidity');
                        return;
                      }

                      // Constant product formula
                      expectedOutput = (amountInWithFee * reserveOut) / (reserveIn + amountInWithFee);
                      if (!isFinite(expectedOutput) || isNaN(expectedOutput)) {
                        setIsSwapping(false); setSwapPhase('idle');
                        alert('Invalid swap calculation - please try a different amount');
                        return;
                      }
                      minOutput = expectedOutput * (1 - dexSettings.slippageTolerance / 100); // Use user's slippage setting

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
                      // v3.2.22-beta: SAFE pricing fallback - QUGUSD is $1, so use QUG price only
                      const fromPrice = fromToken.price;
                      const toPrice = toToken.price;

                      // v3.2.22-beta: CRITICAL - Never use 1:1 fallback which causes massive financial loss
                      if (!fromPrice || fromPrice <= 0 || !toPrice || toPrice <= 0) {
                        setIsSwapping(false); setSwapPhase('idle');
                        alert('❌ Cannot determine safe exchange rate. Please try again later.');
                        return;
                      }

                      expectedOutput = parseFloat(swapAmount) * (fromPrice / toPrice);
                      if (!isFinite(expectedOutput) || isNaN(expectedOutput) || expectedOutput <= 0) {
                        setIsSwapping(false); setSwapPhase('idle');
                        alert('❌ Invalid price calculation. Please refresh prices and try again.');
                        return;
                      }
                      minOutput = expectedOutput * (1 - Math.max(dexSettings.slippageTolerance, 5) / 100); // Oracle needs at least 5% tolerance

                      console.log('💱 No pool found - using oracle pricing (backend will handle):', {
                        expectedOutput,
                        minOutput,
                        fromPrice,
                        toPrice
                      });
                    } else {
                      // No pool and not QUG<->QUGUSD - this will fail but let backend handle the error
                      // v3.2.22-beta: CRITICAL - Require valid prices, never use 1:1 fallback
                      const fromPrice = fromToken.price;
                      const toPrice = toToken.price;

                      if (!fromPrice || fromPrice <= 0 || !toPrice || toPrice <= 0) {
                        setIsSwapping(false); setSwapPhase('idle');
                        alert('❌ Cannot determine safe exchange rate for this pair. Please add liquidity first.');
                        return;
                      }

                      expectedOutput = parseFloat(swapAmount) * (fromPrice / toPrice);
                      if (!isFinite(expectedOutput) || isNaN(expectedOutput) || expectedOutput <= 0) {
                        setIsSwapping(false); setSwapPhase('idle');
                        alert('❌ Invalid price calculation. This token pair may not be tradeable.');
                        return;
                      }
                      minOutput = expectedOutput * 0.995;

                      console.warn('⚠️ No pool found for this token pair:', fromTokenFormatted, '<->', toTokenFormatted);
                    }

                    // v2.3.30-beta: Set cooldown AND preliminary locked balance BEFORE the API call
                    // This prevents the race condition where components render during await
                    swapJustCompletedRef.current = true;
                    localStorage.setItem('dexCooldownUntil', (Date.now() + 15000).toString());

                    // Calculate and set PRELIMINARY locked balance before API call
                    const currentFromToken = tokens.find(t => t.symbol === swapFrom);
                    const currentToToken = tokens.find(t => t.symbol === swapTo);
                    const preliminaryFromBalance = Math.max(0, (currentFromToken?.balance || 0) - parseFloat(swapAmount));
                    const preliminaryToBalance = (currentToToken?.balance || 0) + expectedOutput;

                    if (swapFrom === 'QUG') {
                      localStorage.setItem('dexLockedBalance', preliminaryFromBalance.toString());
                      console.log('🔒 [DEX] PRE-SWAP: Set PRELIMINARY locked balance (QUG will be deducted):', preliminaryFromBalance);
                    } else if (swapTo === 'QUG') {
                      localStorage.setItem('dexLockedBalance', preliminaryToBalance.toString());
                      console.log('🔒 [DEX] PRE-SWAP: Set PRELIMINARY locked balance (QUG will be added):', preliminaryToBalance);
                    }

                    // v4.0.11: ALL amounts sent to backend use 24 decimals
                    // Pool reserves are stored in 24-decimal format internally,
                    // so amount_in and min_amount_out must also be in 24-decimal.
                    const INTERNAL_DECIMALS = 24;
                    const amountInBigInt = parseAmountToBigInt(swapAmount, INTERNAL_DECIMALS);
                    const minOutputBigInt = parseAmountToBigInt(minOutput.toString(), INTERNAL_DECIMALS);

                    // Validate amount doesn't exceed u128 max
                    if (amountInBigInt > U128_MAX) {
                      setIsSwapping(false); setSwapPhase('idle');
                      alert('❌ Amount exceeds maximum supported. Please reduce the amount.');
                      return;
                    }

                    // v4.0.13: CRITICAL FIX - Send amounts as STRINGS to preserve precision.
                    // Number(BigInt) loses precision for 24-decimal amounts (> 2^53).
                    // Backend deserialize_u128_from_any accepts both numbers and strings.
                    const amountInStr = amountInBigInt.toString();
                    const minOutputStr = minOutputBigInt.toString();

                    console.log(`🔢 [DEX v4.0.13] Swap amounts (all 24-dec, string format):`, {
                      swapAmount,
                      internalDecimals: INTERNAL_DECIMALS,
                      amountIn: amountInStr,
                      expectedOutput,
                      minOutput,
                      minAmountOut: minOutputStr
                    });

                    setSwapPhase('executing');
                    const response = await qnkAPI.executeSwap({
                      from_token: fromTokenFormatted,
                      to_token: toTokenFormatted,
                      amount_in: amountInStr, // String to preserve u128 precision
                      min_amount_out: minOutputStr, // String to preserve u128 precision
                      wallet_address: walletAddress,
                    });

                    setSwapPhase('confirming');

                    if (response.success && response.data) {
                      // v2.4.0: NaN protection for amount_out
                      // v4.0.10: Read estimated_amount_out (the actual field name from backend)
                      // ALL amounts from the backend AMM are in 24-decimal format
                      const rawAmountOut = parseFloat(response.data.estimated_amount_out || response.data.amount_out || '0');
                      const amountOut = (rawAmountOut && isFinite(rawAmountOut)) ? rawAmountOut / 1e24 : expectedOutput;
                      const amountIn = parseFloat(swapAmount) || 0;

                      console.log('📊 [DEX] Swap response:', { rawAmountOut, amountOut, amountIn, expectedOutput });

                      setSwapSuccessData({
                        fromToken: swapFrom,
                        toToken: swapTo,
                        fromAmount: amountIn,
                        toAmount: isFinite(amountOut) ? amountOut : expectedOutput,
                        transactionHash: response.data.transaction_id
                      });
                      setIsSwapping(false); setSwapPhase('idle');
                      setShowSwapSuccess(true);
                      // Reset swap amount
                      setSwapAmount('');

                      // v2.9.7-beta: FIX - Use tokensRef for truly current balance (case-insensitive)
                      // The tokens state in the closure is from when the callback was created
                      // tokensRef.current is ALWAYS the latest value thanks to useEffect sync

                      // Get TRULY current balances from the ref (case-insensitive lookup)
                      const swapFromUpper = swapFrom.toUpperCase();
                      const swapToUpper = swapTo.toUpperCase();
                      const refFromToken = tokensRef.current.find(t => t.symbol.toUpperCase() === swapFromUpper);
                      const refToToken = tokensRef.current.find(t => t.symbol.toUpperCase() === swapToUpper);

                      console.log(`🔍 [DEX v2.9.7] Token lookup - swapFrom: "${swapFrom}", swapTo: "${swapTo}"`);
                      console.log(`🔍 [DEX v2.9.7] refFromToken found: ${!!refFromToken}, refToToken found: ${!!refToToken}`);
                      console.log(`🔍 [DEX v2.9.7] tokensRef has ${tokensRef.current.length} tokens`);

                      const baseFromBalance = refFromToken?.balance ?? currentFromToken?.balance ?? 0;
                      const baseToBalance = refToToken?.balance ?? currentToToken?.balance ?? 0;

                      console.log(`🔄 [DEX v2.9.7] CURRENT balances - FROM ${swapFrom}: ${baseFromBalance} (ref: ${refFromToken?.balance}, captured: ${currentFromToken?.balance}), TO ${swapTo}: ${baseToBalance} (ref: ${refToToken?.balance}, captured: ${currentToToken?.balance})`);

                      const newFromBalance = Math.max(0, baseFromBalance - amountIn);
                      const newToBalance = baseToBalance + amountOut;

                      console.log(`🔄 [DEX v2.9.6] NEW balances - FROM: ${baseFromBalance} - ${amountIn} = ${newFromBalance}, TO: ${baseToBalance} + ${amountOut} = ${newToBalance}`);

                      // v2.3.5-beta: Immediately update local token balances for instant UI feedback
                      // v2.9.7-beta: Use case-insensitive comparison
                      setTokens(prevTokens => prevTokens.map(token => {
                        const tokenSymbolUpper = token.symbol.toUpperCase();
                        if (tokenSymbolUpper === swapFromUpper) {
                          console.log(`💸 [DEX] Deducting ${amountIn} from ${token.symbol}: ${token.balance} -> ${newFromBalance}`);
                          return { ...token, balance: newFromBalance };
                        }
                        if (tokenSymbolUpper === swapToUpper) {
                          console.log(`💰 [DEX] Adding ${amountOut} to ${token.symbol}: ${token.balance} -> ${newToBalance}`);
                          return { ...token, balance: newToBalance };
                        }
                        return token;
                      }));

                      console.log(`🔥 [DEX] SWAP SUCCESS - Updating balances:`, {
                        swapFrom, swapTo, amountIn, amountOut,
                        oldFromBalance: currentFromToken?.balance,
                        oldToBalance: currentToToken?.balance,
                        newFromBalance, newToBalance
                      });

                      // v2.3.29-beta: Set LOCKED balance that cannot be overwritten by other code
                      // This is the ONLY source of truth during the cooldown period
                      localStorage.setItem('dexCooldownUntil', (Date.now() + 15000).toString());

                      // v2.3.29-beta: Update localStorage and set LOCKED balance for TopBar
                      if (swapFrom === 'QUG') {
                        const newBal = newFromBalance;
                        // Set both regular and LOCKED balance
                        safeCacheBalance(newBal);
                        localStorage.setItem('dexLockedBalance', newBal.toString()); // LOCKED - TopBar uses this
                        localStorage.setItem('balanceTimestamp', Date.now().toString());
                        console.log('🔒 [DEX] SET LOCKED BALANCE (QUG deducted):', newBal);
                        // Dispatch refresh event
                        window.dispatchEvent(new CustomEvent('qug-balance-changed', { detail: { balance: newBal } }));
                      } else if (swapTo === 'QUG') {
                        const newBal = newToBalance;
                        // Set both regular and LOCKED balance
                        safeCacheBalance(newBal);
                        localStorage.setItem('dexLockedBalance', newBal.toString()); // LOCKED - TopBar uses this
                        localStorage.setItem('balanceTimestamp', Date.now().toString());
                        console.log('🔒 [DEX] SET LOCKED BALANCE (QUG received):', newBal);
                        // Dispatch refresh event
                        window.dispatchEvent(new CustomEvent('qug-balance-changed', { detail: { balance: newBal } }));
                      }

                      // Always dispatch for FROM token (what was spent)
                      window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
                        detail: {
                          symbol: swapFrom,
                          balance: newFromBalance,
                          reason: 'dex-swap-deduct'
                        }
                      }));
                      console.log(`📡 [DEX] Dispatched wallet-balance-updated for ${swapFrom}: ${newFromBalance} (deducted)`);

                      // Always dispatch for TO token (what was received)
                      window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
                        detail: {
                          symbol: swapTo,
                          balance: newToBalance,
                          reason: 'dex-swap-add'
                        }
                      }));
                      console.log(`📡 [DEX] Dispatched wallet-balance-updated for ${swapTo}: ${newToBalance} (added)`);

                      // v2.4.2: Dispatch token-balance-updated for CustomTokensCard (custom tokens only)
                      // CustomTokensCard listens for 'token-balance-updated' with tokenSymbol property
                      const nativeTokens = ['QUG', 'QUGUSD'];
                      console.log(`🔍 [DEX] Checking token dispatch - swapFrom: "${swapFrom}", swapTo: "${swapTo}", nativeTokens:`, nativeTokens);
                      console.log(`🔍 [DEX] swapFrom.toUpperCase(): "${swapFrom.toUpperCase()}", includes: ${nativeTokens.includes(swapFrom.toUpperCase())}`);
                      console.log(`🔍 [DEX] swapTo.toUpperCase(): "${swapTo.toUpperCase()}", includes: ${nativeTokens.includes(swapTo.toUpperCase())}`);

                      if (!nativeTokens.includes(swapFrom.toUpperCase())) {
                        const eventDetail = {
                          tokenSymbol: swapFrom,
                          newBalance: newFromBalance,
                          reason: 'dex-swap-deduct'
                        };
                        console.log(`🪙 [DEX] DISPATCHING token-balance-updated for FROM token:`, eventDetail);
                        window.dispatchEvent(new CustomEvent('token-balance-updated', { detail: eventDetail }));
                      }
                      if (!nativeTokens.includes(swapTo.toUpperCase())) {
                        const eventDetail = {
                          tokenSymbol: swapTo,
                          newBalance: newToBalance,
                          reason: 'dex-swap-add'
                        };
                        console.log(`🪙 [DEX] DISPATCHING token-balance-updated for TO token:`, eventDetail);
                        window.dispatchEvent(new CustomEvent('token-balance-updated', { detail: eventDetail }));
                      }

                      // Also dispatch balance-update for App.tsx TopBar (only for QUG)
                      if (swapFrom === 'QUG') {
                        window.dispatchEvent(new CustomEvent('balance-update', {
                          detail: { balance: newFromBalance, source: 'DexScreen.swap.deduct' }
                        }));
                        console.log(`📡 [DEX] Dispatched balance-update for TopBar: ${newFromBalance}`);
                      } else if (swapTo === 'QUG') {
                        window.dispatchEvent(new CustomEvent('balance-update', {
                          detail: { balance: newToBalance, source: 'DexScreen.swap.add' }
                        }));
                        console.log(`📡 [DEX] Dispatched balance-update for TopBar: ${newToBalance}`);
                      }

                      // Update localStorage for QUG
                      if (swapFrom === 'QUG') {
                        safeCacheBalance(newFromBalance);
                      } else if (swapTo === 'QUG') {
                        safeCacheBalance(newToBalance);
                      }

                      // Update localStorage for QUGUSD
                      if (swapFrom === 'QUGUSD') {
                        localStorage.setItem('cachedQugusdBalance', newFromBalance.toString());
                      } else if (swapTo === 'QUGUSD') {
                        localStorage.setItem('cachedQugusdBalance', newToBalance.toString());
                      }

                      // v2.3.28-beta: swapJustCompletedRef already set BEFORE API call
                      // Just log the success and keep cooldown active
                      console.log('🔒 [DEX] Swap completed - cooldown already active from pre-swap');
                      console.log('✅ [DEX] Local state updated with correct balance - no API refetch needed');

                      // v2.3.33-beta: When cooldown expires, dispatch event to update Dashboard state
                      setTimeout(() => {
                        console.log('🔓 [DEX] Clearing swap lock - dispatching state sync event');
                        swapJustCompletedRef.current = false;

                        // CRITICAL: Before removing locked balance, dispatch event to sync Dashboard state
                        // This tells Dashboard to update its walletBalances state from cachedBalance
                        const cachedQug = localStorage.getItem('cachedBalance');
                        const cachedQugusd = localStorage.getItem('cachedQugusdBalance');

                        console.log('📡 [DEX] Dispatching dex-cooldown-expired with cached balances:', {
                          qug: cachedQug,
                          qugusd: cachedQugusd
                        });

                        window.dispatchEvent(new CustomEvent('dex-cooldown-expired', {
                          detail: {
                            qugBalance: cachedQug ? parseFloat(cachedQug) : null,
                            qugusdBalance: cachedQugusd ? parseFloat(cachedQugusd) : null,
                            source: 'DexScreen.cooldown.expired'
                          }
                        }));

                        // Now safe to remove the locked balance
                        localStorage.removeItem('dexCooldownUntil');
                        localStorage.removeItem('dexLockedBalance');
                        console.log('🔓 [DEX] Cooldown fully cleared');

                        // v2.4.3: Trigger token refresh to get updated prices after swap
                        // The AMM prices change when reserves change, so we need fresh prices
                        console.log('🔄 [DEX] Triggering token price refresh after swap');
                        setRefreshTrigger(prev => prev + 1);
                      }, 15000);
                    } else {
                      // v8.2.8: Clear swap loading state
                      setIsSwapping(false); setSwapPhase('idle');
                      // v2.3.29-beta: Clear cooldown on API error
                      console.error('❌ Swap API error:', response.error);
                      console.error('❌ Full response:', response);
                      swapJustCompletedRef.current = false;
                      localStorage.removeItem('dexCooldownUntil');
                      localStorage.removeItem('dexLockedBalance');
                      console.log('🔓 [DEX] Cleared cooldown due to API error');
                      alert(`❌ Swap failed: ${response.error || 'Unknown error'}`);
                    }
                  } catch (error) {
                    // v8.2.8: Clear swap loading state
                    setIsSwapping(false); setSwapPhase('idle');
                    // v2.3.29-beta: Clear cooldown on exception
                    console.error('❌ Swap exception:', error);
                    console.error('❌ Swap request details:', {
                      from_token: formatTokenForBackend(fromToken.id),
                      to_token: formatTokenForBackend(toToken.id),
                      amount_in: Math.floor(parseFloat(swapAmount) * 1e24),
                      wallet_address: walletAddress
                    });
                    swapJustCompletedRef.current = false;
                    localStorage.removeItem('dexCooldownUntil');
                    localStorage.removeItem('dexLockedBalance');
                    console.log('🔓 [DEX] Cleared cooldown due to exception');
                    alert(`❌ Swap failed: ${error instanceof Error ? error.message : 'Please try again'}`);
                  }
                }}
                disabled={isSwapping}
                className={`w-full py-4 rounded-xl font-bold text-white transition-all ${
                  isSwapping
                    ? 'bg-gray-700 cursor-not-allowed opacity-70'
                    : 'bg-gradient-to-r from-quantum-cyan to-quantum-purple hover:shadow-lg hover:shadow-quantum-cyan/50'
                }`}
              >
                {isSwapping ? (
                  <span className="flex items-center justify-center gap-3">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    {swapPhase === 'verifying' ? 'Verifying...' : swapPhase === 'executing' ? 'Executing Swap...' : 'Confirming...'}
                  </span>
                ) : 'Swap Tokens'}
              </button>

              {/* 💰 v2.4.8-beta: DCA Button */}
              <button
                onClick={() => setShowDcaModal(true)}
                className="w-full py-3 mt-3 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl font-bold text-white hover:shadow-lg hover:shadow-green-500/50 transition-all flex items-center justify-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Setup DCA (Auto-Buy)
              </button>

              {/* 💰 v2.4.9-beta: My DCA Orders Button */}
              <button
                onClick={() => {
                  setShowDcaOrdersPanel(!showDcaOrdersPanel);
                  if (!showDcaOrdersPanel) fetchDcaOrders();
                }}
                className="w-full py-2 mt-2 bg-gray-800/50 border border-green-500/30 rounded-xl text-sm text-green-400 hover:bg-gray-700/50 transition-all flex items-center justify-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                My DCA Orders ({dcaOrders.filter(o => o.status !== 'cancelled' && o.status !== 'completed').length})
                <svg className={`w-4 h-4 transition-transform ${showDcaOrdersPanel ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {/* 🏦 Mint QUGUSD Stablecoin Button */}
              <button
                onClick={() => setIsMintQUGUSDModalOpen(true)}
                className="w-full py-3 mt-3 bg-gradient-to-r from-amber-500 to-yellow-600 rounded-xl font-bold text-white hover:shadow-lg hover:shadow-amber-500/50 transition-all flex items-center justify-center gap-2"
              >
                <Coins className="w-5 h-5" />
                Mint QUGUSD Stablecoin
              </button>

              {/* 💰 v2.4.9-beta: DCA Orders Panel */}
              {showDcaOrdersPanel && (
                <div className="mt-3 space-y-2 max-h-64 overflow-y-auto">
                  {dcaOrders.filter(o => o.status !== 'cancelled' && o.status !== 'completed').length === 0 ? (
                    <p className="text-gray-500 text-sm text-center py-4">No active DCA orders</p>
                  ) : (
                    dcaOrders.filter(o => o.status !== 'cancelled' && o.status !== 'completed').map((order) => (
                      <div key={order.id} className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-3">
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <span className="text-white font-medium">{order.from_token} → {order.to_token}</span>
                            <p className="text-xs text-gray-400">
                              {(order.amount_per_execution / 1e24).toFixed(2)} {order.from_token} / {order.interval}
                            </p>
                          </div>
                          <span className={`text-xs px-2 py-1 rounded ${
                            order.status === 'active' ? 'bg-green-500/20 text-green-400' :
                            order.status === 'paused' ? 'bg-yellow-500/20 text-yellow-400' :
                            'bg-gray-500/20 text-gray-400'
                          }`}>
                            {order.status}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 mb-2">
                          Next: {new Date(order.next_execution_at).toLocaleString()}
                          {order.executions_count > 0 && ` | Executed: ${order.executions_count}x`}
                        </div>
                        <div className="flex gap-2">
                          {order.status === 'active' ? (
                            <button
                              onClick={() => pauseDcaOrder(order.id)}
                              className="flex-1 py-1 text-xs bg-yellow-500/20 text-yellow-400 rounded hover:bg-yellow-500/30"
                            >
                              Pause
                            </button>
                          ) : order.status === 'paused' ? (
                            <button
                              onClick={() => resumeDcaOrder(order.id)}
                              className="flex-1 py-1 text-xs bg-green-500/20 text-green-400 rounded hover:bg-green-500/30"
                            >
                              Resume
                            </button>
                          ) : null}
                          <button
                            onClick={() => cancelDcaOrder(order.id)}
                            className="flex-1 py-1 text-xs bg-red-500/20 text-red-400 rounded hover:bg-red-500/30"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
              </div>
            </div>
            </div>
              )}
        </motion.div>

          {/* Token Table - Right side, next to swap */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex-1 min-w-0"
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
                    placeholder="Search by name, symbol, or address..."
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
              <div className="overflow-x-auto max-h-[700px] overflow-y-auto">
                <table className="w-full">
                  <thead className="sticky top-0 bg-black/90 backdrop-blur-sm z-10">
                    <tr className="border-b border-white/10">
                      <th className="text-left py-3 px-2 text-gray-400 font-medium text-xs">Token</th>
                      <th onClick={() => handleSort('price')} className="text-right py-3 px-2 text-gray-400 font-medium text-xs cursor-pointer hover:text-white transition-colors whitespace-nowrap">
                        Price {sortBy === 'price' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th onClick={() => handleSort('change1h')} className="text-right py-3 px-2 text-gray-400 font-medium text-xs cursor-pointer hover:text-white transition-colors whitespace-nowrap">
                        1h {sortBy === 'change1h' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th onClick={() => handleSort('change24h')} className="text-right py-3 px-2 text-gray-400 font-medium text-xs cursor-pointer hover:text-white transition-colors whitespace-nowrap">
                        24h {sortBy === 'change24h' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th onClick={() => handleSort('change7d')} className="text-right py-3 px-2 text-gray-400 font-medium text-xs cursor-pointer hover:text-white transition-colors whitespace-nowrap">
                        7d {sortBy === 'change7d' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th onClick={() => handleSort('volume24h')} className="text-right py-3 px-2 text-gray-400 font-medium text-xs cursor-pointer hover:text-white transition-colors whitespace-nowrap">
                        Vol {sortBy === 'volume24h' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th onClick={() => handleSort('marketCap')} className="text-right py-3 px-2 text-gray-400 font-medium text-xs cursor-pointer hover:text-white transition-colors whitespace-nowrap">
                        MCap {sortBy === 'marketCap' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th className="text-right py-3 px-2 text-gray-400 font-medium text-xs whitespace-nowrap" title="Fully Diluted Valuation">
                        FDV
                      </th>
                      <th onClick={() => handleSort('liquidity')} className="text-right py-3 px-2 text-gray-400 font-medium text-xs cursor-pointer hover:text-white transition-colors whitespace-nowrap">
                        Liq {sortBy === 'liquidity' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th onClick={() => handleSort('holders')} className="text-right py-3 px-2 text-gray-400 font-medium text-xs cursor-pointer hover:text-white transition-colors whitespace-nowrap">
                        Holders {sortBy === 'holders' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th onClick={() => handleSort('createdAt')} className="text-right py-3 px-2 text-gray-400 font-medium text-xs cursor-pointer hover:text-white transition-colors whitespace-nowrap">
                        Age {sortBy === 'createdAt' && (sortDirection === 'asc' ? '↑' : '↓')}
                      </th>
                      <th className="text-right py-3 px-2 text-gray-400 font-medium text-xs">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTokens.map((token, index) => (
                      <motion.tr
                        key={token.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: index * 0.05 }}
                        className={`border-b border-white/5 hover:bg-white/5 transition-colors cursor-pointer ${
                          nitroBoostTokens.has(token.id) ? 'nitro-boost-active' : ''
                        }`}
                        onClick={() => {
                          if (token.isCrowdfund && token.campaignData) {
                            setSelectedCampaign(token.campaignData);
                          } else {
                            setSelectedToken(token);
                          }
                        }}
                      >
                        <td className="py-3 px-3">
                          <div className="flex items-center gap-2">
                            {token.isCrowdfund ? (
                              <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm overflow-hidden flex-shrink-0 bg-gradient-to-br from-cyan-500 to-blue-600 ring-1 ring-cyan-400/50">
                                <span className="text-base">{token.campaignData?.exchange_logo || '\u{1F3E6}'}</span>
                              </div>
                            ) : (
                              <TokenIcon
                                symbol={token.symbol}
                                icon={token.icon}
                                logoUrl={token.logoUrl}
                                size={32}
                              />
                            )}
                            <div className="min-w-0">
                              <div className="flex items-center gap-1">
                                <span className="font-bold text-white text-sm">{token.symbol}</span>
                                {token.isIndexToken && <span className="px-1 rounded bg-purple-600/50 text-[10px] font-bold text-purple-200">IDX</span>}
                                {token.isPerp && <span className="px-1 rounded bg-pink-600/50 text-[10px] font-bold text-pink-200">PERP</span>}
                                {['wBTC', 'wZEC', 'wIRON', 'wETH'].includes(token.symbol) && <span className="px-1 rounded bg-amber-500/50 text-[10px] font-bold text-amber-200">BRIDGE</span>}
                                {boostedTokens.has(token.id) && <span className="px-1 rounded bg-orange-500/50 text-[10px] font-bold text-orange-200">NITRO</span>}
                                {token.isCrowdfund && <span className="px-1 rounded bg-cyan-500/50 text-[10px] font-bold text-cyan-200 animate-pulse">XLIST</span>}
                              </div>
                              <div className="text-xs text-gray-500 truncate">
                                {token.isCrowdfund ? (
                                  <span className="text-cyan-400/70">{Math.round((token.campaignData?.raised_usd || 0) / (token.campaignData?.target_usd || 1) * 100)}% funded &middot; {token.campaignData?.contributor_count || 0} backers</span>
                                ) : token.name}
                              </div>
                            </div>
                          </div>
                        </td>
                        {token.isCrowdfund ? (
                          <>
                            {/* XLIST: Show funding progress bar spanning multiple columns */}
                            <td colSpan={4} className="py-3 px-2">
                              <div className="flex items-center gap-2">
                                <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                                  <div
                                    className="h-full rounded-full transition-all duration-1000"
                                    style={{
                                      width: `${Math.min(100, (token.campaignData?.raised_usd || 0) / (token.campaignData?.target_usd || 1) * 100)}%`,
                                      background: 'linear-gradient(90deg, #00d4ff, #7b61ff, #00d4ff)',
                                      backgroundSize: '200% 100%',
                                      animation: 'shimmer 2s ease-in-out infinite',
                                    }}
                                  />
                                </div>
                                <span className="text-cyan-400 text-xs font-bold whitespace-nowrap">
                                  ${(token.campaignData?.raised_usd || 0).toLocaleString()} / ${(token.campaignData?.target_usd || 0).toLocaleString()}
                                </span>
                              </div>
                            </td>
                            <td className="py-3 px-2 text-right text-cyan-300 text-xs font-bold whitespace-nowrap">
                              {token.campaignData?.status === 'funded' ? 'FUNDED' : token.campaignData?.status === 'listed' ? 'LISTED' : 'LIVE'}
                            </td>
                            <td className="py-3 px-2 text-right text-white text-xs whitespace-nowrap">
                              ${(token.campaignData?.target_usd || 0).toLocaleString()}
                            </td>
                            <td className="py-3 px-2 text-right text-gray-400 text-xs whitespace-nowrap">-</td>
                            <td className="py-3 px-2 text-right text-gray-400 text-xs whitespace-nowrap">${(token.campaignData?.raised_usd || 0).toLocaleString()}</td>
                            <td className="py-3 px-2 text-right text-cyan-400 text-xs whitespace-nowrap font-medium">{token.campaignData?.contributor_count || 0}</td>
                            <td className="py-3 px-2 text-right text-gray-500 text-xs whitespace-nowrap">
                              {token.campaignData?.early_bird_slots ? `${token.campaignData.early_bird_slots - (token.campaignData.early_bird_claimed || 0)} slots` : '-'}
                            </td>
                          </>
                        ) : (
                          <>
                        <td className="py-3 px-2 text-right text-white font-medium whitespace-nowrap text-xs">${formatPrice(token.price)}</td>
                        <td className="py-3 px-2 text-right">
                          <span className={`font-medium text-xs ${token.change1h > 0 ? 'text-quantum-green' : token.change1h < 0 ? 'text-red-500' : 'text-gray-500'}`}>
                            {token.change1h > 0 ? '+' : ''}{token.change1h.toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-3 px-2 text-right">
                          <span className={`font-medium text-xs ${token.change24h > 0 ? 'text-quantum-green' : token.change24h < 0 ? 'text-red-500' : 'text-gray-500'}`}>
                            {token.change24h > 0 ? '+' : ''}{token.change24h.toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-3 px-2 text-right">
                          <span className={`font-medium text-xs ${token.change7d > 0 ? 'text-quantum-green' : token.change7d < 0 ? 'text-red-500' : 'text-gray-500'}`}>
                            {token.change7d > 0 ? '+' : ''}{token.change7d.toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-3 px-2 text-right text-white text-xs whitespace-nowrap">{formatNumber(token.volume24h)}</td>
                        <td className="py-3 px-2 text-right text-white text-xs whitespace-nowrap">${formatNumber(token.marketCap)}</td>
                        <td className="py-3 px-2 text-right text-gray-400 text-xs whitespace-nowrap" title="Fully Diluted Valuation">${formatNumber(token.totalSupply * token.price)}</td>
                        <td className="py-3 px-2 text-right text-gray-400 text-xs whitespace-nowrap">${formatNumber(token.liquidity)}</td>
                        <td className="py-3 px-2 text-right text-gray-500 text-xs whitespace-nowrap">{token.holders?.toLocaleString() || '0'}</td>
                        <td className="py-3 px-2 text-right text-gray-500 text-xs whitespace-nowrap">
                          {token.createdAt ? (() => {
                            const diff = Date.now() - token.createdAt;
                            const mins = Math.floor(diff / 60000);
                            const hrs = Math.floor(diff / 3600000);
                            const days = Math.floor(diff / 86400000);
                            if (days > 0) return `${days}d`;
                            if (hrs > 0) return `${hrs}h`;
                            return `${mins}m`;
                          })() : '-'}
                        </td>
                          </>
                        )}
                        <td className="py-3 px-3 text-right" onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center justify-end gap-1">
                            {token.isCrowdfund ? (
                              <motion.button
                                onClick={() => token.campaignData && setSelectedCampaign(token.campaignData)}
                                className="px-3 py-1.5 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg text-white text-xs font-bold hover:shadow-lg hover:shadow-cyan-500/30 transition-all"
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                              >
                                Fund
                              </motion.button>
                            ) : (
                            <>
                            <motion.button
                              onClick={() => { setSwapFrom(token.symbol); setIsSwapCollapsed(false); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
                              className="px-2 py-1 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-lg text-white text-xs font-medium hover:shadow-lg transition-all"
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                            >
                              Trade
                            </motion.button>
                            <motion.button
                              onClick={() => setLiquidityToken(token)}
                              className="px-2 py-1 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-lg text-white text-xs font-medium hover:shadow-lg transition-all"
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                            >
                              LP
                            </motion.button>
                            <motion.button
                              onClick={() => handleNitroBoost(token)}
                              className={`px-2 py-1 rounded-lg text-xs font-medium hover:shadow-lg transition-all ${
                                boostedTokens.has(token.id)
                                  ? 'bg-orange-500/80 text-white'
                                  : 'bg-orange-500/20 text-orange-400 hover:bg-orange-500/40'
                              }`}
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              title={boostedTokens.has(token.id) ? `Boosted: ${boostedTokens.get(token.id)} pts` : 'Nitro Boost'}
                            >
                              🚀
                            </motion.button>
                            <motion.button
                              onClick={() => handleAIAnalyze(token)}
                              className="px-2 py-1 bg-blue-500/20 text-blue-400 hover:bg-blue-500/40 rounded-lg text-xs font-medium hover:shadow-lg transition-all"
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              title="AI Analysis"
                            >
                              🤖
                            </motion.button>
                            </>
                            )}
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
          </div>{/* end Swap + Token Table flex row */}

          {/* Liquidity Sections - Full width below */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="lg:col-span-4"
          >

          {/* Custom Token Liquidity Section */}
          <div className="relative group">
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
          {/* v3.4.17-beta: PRIVACY FIX - Only show pools owned by current user */}
          {(() => {
            const walletAddr = localStorage.getItem('walletAddress') || '';
            const myPools = liquidityPools.filter(pool => {
              // Match provider address (case-insensitive)
              const poolProvider = (pool.provider || '').toLowerCase();
              const myAddress = walletAddr.toLowerCase();
              return poolProvider === myAddress ||
                     poolProvider === `qnk${myAddress}` ||
                     poolProvider === myAddress.replace(/^qnk/, '') ||
                     `qnk${poolProvider}` === myAddress;
            });
            return myPools.length > 0 && (
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
                  {myPools.map((pool, index) => (
                    <motion.div
                      key={pool.pool_id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-white/5 border border-white/10 rounded-xl p-4 hover:border-quantum-cyan/50 transition-all"
                    >
                      {/* Pool Header - v3.6.5-beta: Show token symbols instead of addresses */}
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <div className="text-2xl">💧</div>
                          <div>
                            <div className="font-bold text-white text-lg">
                              {getTokenDisplayName(pool.token0, tokens)} / {getTokenDisplayName(pool.token1, tokens)}
                            </div>
                            <div className="text-xs text-gray-400">
                              Pool ID: {pool.pool_id.slice(0, 16)}...
                              {pool.token0.length > 10 && (
                                <span className="ml-2 text-gray-500">
                                  ({pool.token0.slice(0, 12)}...)
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Pool Stats - v3.6.7-beta: Pools always use 24 decimals internally */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-quantum-cyan/10 border border-quantum-cyan/20 rounded-lg p-3">
                          <div className="text-xs text-gray-400 mb-1 truncate">
                            {getTokenDisplayName(pool.token0, tokens)} Reserve
                          </div>
                          <div className="text-sm font-bold text-white truncate">
                            {(() => {
                              const raw = parseU128(pool.reserve0);
                              // Pools always store in 24-decimal format
                              const display = raw / 1e24;
                              if (display >= 1e12) return (display / 1e12).toFixed(2) + 'T';
                              if (display >= 1e9) return (display / 1e9).toFixed(2) + 'B';
                              if (display >= 1e6) return (display / 1e6).toFixed(2) + 'M';
                              if (display >= 1e3) return (display / 1e3).toFixed(2) + 'K';
                              return display.toFixed(display < 1 ? 6 : 2);
                            })()}
                          </div>
                        </div>
                        <div className="bg-quantum-purple/10 border border-quantum-purple/20 rounded-lg p-3">
                          <div className="text-xs text-gray-400 mb-1 truncate">
                            {getTokenDisplayName(pool.token1, tokens)} Reserve
                          </div>
                          <div className="text-sm font-bold text-white truncate">
                            {(() => {
                              const raw = parseU128(pool.reserve1);
                              // Pools always store in 24-decimal format
                              const display = raw / 1e24;
                              if (display >= 1e12) return (display / 1e12).toFixed(2) + 'T';
                              if (display >= 1e9) return (display / 1e9).toFixed(2) + 'B';
                              if (display >= 1e6) return (display / 1e6).toFixed(2) + 'M';
                              if (display >= 1e3) return (display / 1e3).toFixed(2) + 'K';
                              return display.toFixed(display < 1 ? 8 : 2);
                            })()}
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
                                    // v4.0.12: Supply uses token's own decimals, balances use 24 decimals internally
                                    const decimals = contract.decimals || 8;
                                    const rawSupply = contract.total_supply || '0';
                                    const supplyStr = typeof rawSupply === 'string' ? rawSupply : String(rawSupply);
                                    const supplyBigInt = BigInt(supplyStr);
                                    const supplyDivisor = BigInt(10) ** BigInt(decimals);
                                    const actualSupply = Number(supplyBigInt / supplyDivisor) + Number(supplyBigInt % supplyDivisor) / Number(supplyDivisor);
                                    let tokenBalance = 0;

                                    if (walletAddress) {
                                      const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, pool.token0);
                                      if (balanceResponse.success && balanceResponse.data) {
                                        // Balances use 24 decimals internally
                                        const rawBalance = balanceResponse.data.balance || '0';
                                        const balanceStr = typeof rawBalance === 'string' ? rawBalance : String(rawBalance);
                                        const balanceBigInt = BigInt(balanceStr);
                                        const balDivisor = BigInt(10) ** BigInt(24);
                                        tokenBalance = Number(balanceBigInt / balDivisor) + Number(balanceBigInt % balDivisor) / Number(balDivisor);
                                      }
                                    }

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
                                      fullyDilutedMarketCap: actualSupply * 1.0,
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
                                        // v4.0.12: Supply uses token's own decimals, balances use 24 decimals
                                        const decimals = contract.decimals || 8;
                                        const rawSupply = contract.total_supply || '0';
                                        const supplyStr = typeof rawSupply === 'string' ? rawSupply : String(rawSupply);
                                        const supplyBigInt = BigInt(supplyStr);
                                        const supplyDivisor = BigInt(10) ** BigInt(decimals);
                                        const actualSupply = Number(supplyBigInt / supplyDivisor) + Number(supplyBigInt % supplyDivisor) / Number(supplyDivisor);
                                        let tokenBalance = 0;

                                        if (walletAddress) {
                                          const balanceResponse = await qnkAPI.getTokenBalance(walletAddress, foundToken.address);
                                          if (balanceResponse.success && balanceResponse.data) {
                                            // Balances use 24 decimals internally
                                            const rawBalance = balanceResponse.data.balance || '0';
                                            const balanceStr = typeof rawBalance === 'string' ? rawBalance : String(rawBalance);
                                            const balanceBigInt = BigInt(balanceStr);
                                            const balDivisor = BigInt(10) ** BigInt(24);
                                            tokenBalance = Number(balanceBigInt / balDivisor) + Number(balanceBigInt % balDivisor) / Number(balDivisor);
                                          }
                                        }

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
                                          fullyDilutedMarketCap: actualSupply * 1.0,
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
                          onClick={async () => {
                            // v3.6.6-beta: Fetch fresh pool data before opening modal
                            // to ensure reserves are up-to-date after adding liquidity
                            try {
                              const response = await qnkAPI.getLiquidityPools();
                              if (response.success && response.data) {
                                const freshPool = response.data.find((p: any) => p.address === pool.pool_id || p.pool_id === pool.pool_id);
                                if (freshPool) {
                                  console.log('🔄 [REMOVE] Fetched fresh pool data:', freshPool);
                                  setRemovingPool({
                                    ...pool,
                                    reserve0: freshPool.reserve0,
                                    reserve1: freshPool.reserve1,
                                    total_liquidity: freshPool.total_liquidity
                                  });
                                } else {
                                  setRemovingPool(pool);
                                }
                              } else {
                                setRemovingPool(pool);
                              }
                            } catch (e) {
                              console.warn('Failed to refresh pool, using cached data:', e);
                              setRemovingPool(pool);
                            }
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
                    <strong className="text-quantum-cyan">Your Pools:</strong> {myPools.length} active liquidity pool{myPools.length !== 1 ? 's' : ''}
                  </p>
                </div>
              </div>
            </div>
          );})()}
        </motion.div>

        {/* AI Market Analyzer Panel - Full Width Below Token Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="lg:col-span-4"
        >
          <MarketAnalyzerPanel
            onOpportunityClick={(tokenSymbol) => {
              // When user clicks a token, set it as the swap target
              if (tokenSymbol) {
                setSwapTo(tokenSymbol);
              }
            }}
          />
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

      {/* v8.2.8: Swap Processing Overlay */}
      <AnimatePresence>
        {isSwapping && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 backdrop-blur-md"
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              className="relative bg-gray-900/95 border border-quantum-cyan/30 rounded-2xl p-8 max-w-sm w-full mx-4 shadow-2xl shadow-quantum-cyan/20"
            >
              {/* Animated ring */}
              <div className="flex justify-center mb-6">
                <div className="relative w-24 h-24">
                  {/* Outer spinning ring */}
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    className="absolute inset-0 rounded-full border-4 border-transparent border-t-quantum-cyan border-r-quantum-purple"
                  />
                  {/* Inner counter-spinning ring */}
                  <motion.div
                    animate={{ rotate: -360 }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                    className="absolute inset-2 rounded-full border-4 border-transparent border-b-green-400 border-l-blue-400"
                  />
                  {/* Center icon */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    >
                      {swapPhase === 'verifying' && (
                        <svg className="w-8 h-8 text-quantum-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                        </svg>
                      )}
                      {swapPhase === 'executing' && (
                        <svg className="w-8 h-8 text-quantum-purple" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                      )}
                      {swapPhase === 'confirming' && (
                        <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      )}
                    </motion.div>
                  </div>
                </div>
              </div>

              {/* Phase text */}
              <motion.h3
                key={swapPhase}
                initial={{ y: 10, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className="text-xl font-bold text-center text-white mb-2"
              >
                {swapPhase === 'verifying' && 'Verifying Trade'}
                {swapPhase === 'executing' && 'Executing Swap'}
                {swapPhase === 'confirming' && 'Confirming Transaction'}
              </motion.h3>
              <p className="text-sm text-gray-400 text-center mb-4">
                {swapPhase === 'verifying' && 'Checking pool liquidity & calculating optimal route...'}
                {swapPhase === 'executing' && 'Broadcasting swap to the AMM engine...'}
                {swapPhase === 'confirming' && 'Waiting for on-chain confirmation...'}
              </p>

              {/* Progress steps */}
              <div className="flex items-center justify-center gap-2 mt-4">
                {['verifying', 'executing', 'confirming'].map((phase, i) => (
                  <div key={phase} className="flex items-center gap-2">
                    <motion.div
                      className={`w-3 h-3 rounded-full ${
                        phase === swapPhase
                          ? 'bg-quantum-cyan shadow-lg shadow-quantum-cyan/50'
                          : ['verifying', 'executing', 'confirming'].indexOf(swapPhase) > i
                            ? 'bg-green-400'
                            : 'bg-gray-600'
                      }`}
                      animate={phase === swapPhase ? { scale: [1, 1.3, 1] } : {}}
                      transition={{ duration: 0.8, repeat: Infinity }}
                    />
                    {i < 2 && (
                      <div className={`w-8 h-0.5 ${
                        ['verifying', 'executing', 'confirming'].indexOf(swapPhase) > i
                          ? 'bg-green-400'
                          : 'bg-gray-600'
                      }`} />
                    )}
                  </div>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Swap Success Modal */}
      {swapSuccessData && (
        <SwapSuccessModal
          isOpen={showSwapSuccess}
          onClose={() => {
            setShowSwapSuccess(false);
            setSwapSuccessData(null);
            // v2.4.3: Refresh token prices when modal closes
            // The AMM prices changed after the swap, show updated prices immediately
            console.log('🔄 [DEX] SwapSuccessModal closed - triggering price refresh');
            setRefreshTrigger(prev => prev + 1);
          }}
          fromToken={swapSuccessData.fromToken}
          toToken={swapSuccessData.toToken}
          fromAmount={swapSuccessData.fromAmount}
          toAmount={swapSuccessData.toAmount}
          transactionHash={swapSuccessData.transactionHash}
          score={swapSuccessData.score}
        />
      )}

      {/* AI Token Analysis Modal */}
      <AnimatePresence>
        {aiAnalyzingToken && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
            onClick={() => {
              setAiAnalyzingToken(null);
              setAiAnalysisResult(null);
            }}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="relative w-full max-w-lg bg-gradient-to-b from-slate-900 to-slate-950 rounded-2xl border border-violet-500/30 shadow-2xl overflow-hidden"
            >
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-white/10">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 flex items-center justify-center">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-white">AI Token Analysis</h2>
                    <p className="text-sm text-gray-400">{aiAnalyzingToken.symbol} - {aiAnalyzingToken.name}</p>
                  </div>
                </div>
                <button
                  onClick={() => {
                    setAiAnalyzingToken(null);
                    setAiAnalysisResult(null);
                  }}
                  className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-gray-400" />
                </button>
              </div>

              {/* Content */}
              <div className="p-6 space-y-6">
                {aiAnalysisLoading ? (
                  <div className="flex flex-col items-center justify-center py-12">
                    <Loader2 className="w-12 h-12 text-violet-500 animate-spin mb-4" />
                    <p className="text-gray-400">Analyzing {aiAnalyzingToken.symbol}...</p>
                    <p className="text-sm text-gray-500 mt-2">Checking liquidity, activity, and safety metrics</p>
                  </div>
                ) : aiAnalysisResult ? (
                  <>
                    {/* Score & Verdict */}
                    <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-xl border border-white/5">
                      <div className="flex items-center gap-4">
                        <div className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold ${
                          aiAnalysisResult.verdict === 'GOOD' ? 'bg-green-500/20 text-green-400 border-2 border-green-500/50' :
                          aiAnalysisResult.verdict === 'CAUTION' ? 'bg-amber-900/30 text-amber-300 border-2 border-amber-500/50' :
                          'bg-red-500/20 text-red-400 border-2 border-red-500/50'
                        }`}>
                          {aiAnalysisResult.score}
                        </div>
                        <div>
                          <p className="text-sm text-gray-400">Investment Score</p>
                          <p className={`text-lg font-bold ${
                            aiAnalysisResult.verdict === 'GOOD' ? 'text-green-400' :
                            aiAnalysisResult.verdict === 'CAUTION' ? 'text-yellow-400' :
                            'text-red-400'
                          }`}>
                            {aiAnalysisResult.verdict === 'GOOD' ? 'Looks Good' :
                             aiAnalysisResult.verdict === 'CAUTION' ? 'Use Caution' :
                             'High Risk'}
                          </p>
                        </div>
                      </div>
                      <div className={`px-4 py-2 rounded-lg font-semibold ${
                        aiAnalysisResult.verdict === 'GOOD' ? 'bg-green-500/20 text-green-400' :
                        aiAnalysisResult.verdict === 'CAUTION' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {aiAnalysisResult.verdict}
                      </div>
                    </div>

                    {/* Summary */}
                    <div className="p-4 bg-slate-800/30 rounded-xl border border-white/5">
                      <p className="text-sm text-gray-300 leading-relaxed">{aiAnalysisResult.summary}</p>
                    </div>

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-2 gap-3">
                      {Object.entries(aiAnalysisResult.metrics).map(([key, metric]) => (
                        <div
                          key={key}
                          className="p-3 bg-slate-800/30 rounded-lg border border-white/5"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs text-gray-500 capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                            <div className={`w-2 h-2 rounded-full ${
                              metric.status === 'good' ? 'bg-green-500' :
                              metric.status === 'warn' ? 'bg-yellow-500' :
                              'bg-red-500'
                            }`} />
                          </div>
                          <p className={`text-sm font-medium ${
                            metric.status === 'good' ? 'text-green-400' :
                            metric.status === 'warn' ? 'text-yellow-400' :
                            'text-red-400'
                          }`}>
                            {metric.value}
                          </p>
                        </div>
                      ))}
                    </div>

                    {/* Disclaimer */}
                    <div className="flex items-start gap-2 p-3 bg-amber-900/20 rounded-lg border border-amber-500/30">
                      <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                      <p className="text-xs text-gray-300">
                        This analysis is for informational purposes only and should not be considered financial advice. Always DYOR (Do Your Own Research) before investing.
                      </p>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-8 text-gray-400">
                    <p>Analysis failed. Please try again.</p>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="flex items-center justify-end gap-3 p-6 border-t border-white/10">
                <motion.button
                  onClick={() => {
                    setAiAnalyzingToken(null);
                    setAiAnalysisResult(null);
                  }}
                  className="px-6 py-2.5 bg-slate-700 hover:bg-slate-600 rounded-xl text-white font-medium transition-colors"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Close
                </motion.button>
                {aiAnalysisResult && (
                  <motion.button
                    onClick={() => {
                      setSwapFrom(aiAnalyzingToken.symbol);
                      setAiAnalyzingToken(null);
                      setAiAnalysisResult(null);
                      window.scrollTo({ top: 0, behavior: 'smooth' });
                    }}
                    className="px-6 py-2.5 bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-xl text-white font-medium transition-all hover:shadow-lg hover:shadow-violet-500/30"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    Trade {aiAnalyzingToken.symbol}
                  </motion.button>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* DEX Settings Modal */}
      <AnimatePresence>
        {isSettingsModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
            onClick={() => setIsSettingsModalOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="relative w-full max-w-md bg-gradient-to-b from-slate-900 to-slate-950 rounded-2xl border border-quantum-cyan/30 shadow-2xl overflow-hidden"
            >
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-white/10">
                <h3 className="text-xl font-bold text-white flex items-center gap-2">
                  <Settings className="w-5 h-5 text-quantum-cyan" />
                  Transaction Settings
                </h3>
                <button
                  onClick={() => setIsSettingsModalOpen(false)}
                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <X className="w-5 h-5 text-gray-400" />
                </button>
              </div>

              {/* Content */}
              <div className="p-6 space-y-6">
                {/* Slippage Tolerance */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
                      <Shield className="w-4 h-4 text-quantum-cyan" />
                      Slippage Tolerance
                    </label>
                    <span className="text-quantum-cyan font-medium">
                      {dexSettings.slippageTolerance}%
                    </span>
                  </div>
                  <div className="flex gap-2">
                    {[0.1, 0.5, 1.0].map((value) => (
                      <button
                        key={value}
                        onClick={() => {
                          setDexSettings(prev => ({ ...prev, slippageTolerance: value }));
                          setCustomSlippage('');
                        }}
                        className={`flex-1 py-2 px-3 rounded-lg font-medium transition-all ${
                          dexSettings.slippageTolerance === value && !customSlippage
                            ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                            : 'bg-white/5 text-gray-400 hover:bg-white/10'
                        }`}
                      >
                        {value}%
                      </button>
                    ))}
                    <div className="relative flex-1">
                      <input
                        type="number"
                        value={customSlippage}
                        onChange={(e) => {
                          const val = e.target.value;
                          setCustomSlippage(val);
                          if (val && parseFloat(val) > 0) {
                            setDexSettings(prev => ({ ...prev, slippageTolerance: parseFloat(val) }));
                          }
                        }}
                        placeholder="Custom"
                        className="w-full py-2 px-3 bg-white/5 border border-white/10 rounded-lg text-white text-center focus:outline-none focus:border-quantum-cyan/50"
                      />
                      {customSlippage && (
                        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400">%</span>
                      )}
                    </div>
                  </div>
                  {dexSettings.slippageTolerance > 5 && (
                    <div className="flex items-center gap-2 text-yellow-500 text-xs">
                      <AlertTriangle className="w-4 h-4" />
                      High slippage may result in unfavorable rates
                    </div>
                  )}
                </div>

                {/* Transaction Deadline */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
                      <Clock className="w-4 h-4 text-quantum-cyan" />
                      Transaction Deadline
                    </label>
                    <span className="text-quantum-cyan font-medium">
                      {dexSettings.transactionDeadline} min
                    </span>
                  </div>
                  <div className="flex gap-2">
                    {[10, 20, 30].map((value) => (
                      <button
                        key={value}
                        onClick={() => setDexSettings(prev => ({ ...prev, transactionDeadline: value }))}
                        className={`flex-1 py-2 px-3 rounded-lg font-medium transition-all ${
                          dexSettings.transactionDeadline === value
                            ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                            : 'bg-white/5 text-gray-400 hover:bg-white/10'
                        }`}
                      >
                        {value}m
                      </button>
                    ))}
                    <div className="relative flex-1">
                      <input
                        type="number"
                        value={dexSettings.transactionDeadline === 10 || dexSettings.transactionDeadline === 20 || dexSettings.transactionDeadline === 30 ? '' : dexSettings.transactionDeadline}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          if (val > 0) {
                            setDexSettings(prev => ({ ...prev, transactionDeadline: val }));
                          }
                        }}
                        placeholder="Custom"
                        className="w-full py-2 px-3 bg-white/5 border border-white/10 rounded-lg text-white text-center focus:outline-none focus:border-quantum-cyan/50"
                      />
                    </div>
                  </div>
                </div>

                {/* Expert Mode Toggle */}
                <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl">
                  <div className="space-y-1">
                    <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-orange-500" />
                      Expert Mode
                    </label>
                    <p className="text-xs text-gray-500">Allow high price impact trades without confirmation</p>
                  </div>
                  <button
                    onClick={() => setDexSettings(prev => ({ ...prev, expertMode: !prev.expertMode }))}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      dexSettings.expertMode ? 'bg-orange-500' : 'bg-gray-600'
                    }`}
                  >
                    <motion.div
                      className="absolute top-1 w-4 h-4 bg-white rounded-full shadow"
                      animate={{ left: dexSettings.expertMode ? '26px' : '4px' }}
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    />
                  </button>
                </div>

                {/* Multihops Toggle */}
                <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl">
                  <div className="space-y-1">
                    <label className="text-sm font-medium text-gray-300">Multihops</label>
                    <p className="text-xs text-gray-500">Allow routing through multiple pools for best price</p>
                  </div>
                  <button
                    onClick={() => setDexSettings(prev => ({ ...prev, multihops: !prev.multihops }))}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      dexSettings.multihops ? 'bg-quantum-cyan' : 'bg-gray-600'
                    }`}
                  >
                    <motion.div
                      className="absolute top-1 w-4 h-4 bg-white rounded-full shadow"
                      animate={{ left: dexSettings.multihops ? '26px' : '4px' }}
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    />
                  </button>
                </div>

                {/* Gas Preference */}
                <div className="space-y-3">
                  <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
                    <Zap className="w-4 h-4 text-quantum-cyan" />
                    Gas Preference
                  </label>
                  <div className="flex gap-2">
                    {(['low', 'medium', 'high'] as const).map((preference) => (
                      <button
                        key={preference}
                        onClick={() => setDexSettings(prev => ({ ...prev, gasPreference: preference }))}
                        className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                          dexSettings.gasPreference === preference
                            ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                            : 'bg-white/5 text-gray-400 hover:bg-white/10'
                        }`}
                      >
                        <div className="flex flex-col items-center gap-1">
                          <span className="capitalize">{preference}</span>
                          <span className="text-xs opacity-70">
                            {preference === 'low' ? 'Slower' : preference === 'medium' ? 'Normal' : 'Fastest'}
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="p-6 border-t border-white/10">
                <button
                  onClick={() => setIsSettingsModalOpen(false)}
                  className="w-full py-3 px-6 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl text-white font-bold hover:shadow-lg hover:shadow-quantum-cyan/30 transition-all"
                >
                  Save Settings
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 💰 v2.4.8-beta: DCA (Dollar Cost Averaging) Modal */}
      <AnimatePresence>
        {showDcaModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setShowDcaModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="bg-gradient-to-br from-gray-900 to-black border border-green-500/30 rounded-2xl w-full max-w-md overflow-hidden"
              onClick={e => e.stopPropagation()}
            >
              {/* Header */}
              <div className="p-6 border-b border-white/10 bg-gradient-to-r from-green-500/10 to-emerald-600/10">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-green-500/20 rounded-xl">
                      <svg className="w-6 h-6 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white">Setup DCA</h2>
                      <p className="text-sm text-gray-400">Dollar Cost Averaging</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowDcaModal(false)}
                    className="p-2 hover:bg-white/10 rounded-xl transition-colors"
                  >
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              {/* Content */}
              <div className="p-6 space-y-5">
                {/* Token Pair Display */}
                <div className="p-4 bg-white/5 rounded-xl border border-white/10">
                  <div className="text-sm text-gray-400 mb-2">Auto-buy {swapTo} with {swapFrom}</div>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 px-3 py-2 bg-white/5 rounded-lg">
                      <TokenIcon symbol={swapFrom} icon={findToken(swapFrom)?.icon} logoUrl={findToken(swapFrom)?.logoUrl} size={24} />
                      <span className="text-white font-medium">{swapFrom}</span>
                    </div>
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                    <div className="flex items-center gap-2 px-3 py-2 bg-white/5 rounded-lg">
                      <TokenIcon symbol={swapTo} icon={findToken(swapTo)?.icon} logoUrl={findToken(swapTo)?.logoUrl} size={24} />
                      <span className="text-white font-medium">{swapTo}</span>
                    </div>
                  </div>
                </div>

                {/* Amount per Execution */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">Amount per Purchase</label>
                  <div className="relative">
                    <input
                      type="number"
                      value={dcaAmount}
                      onChange={(e) => setDcaAmount(e.target.value)}
                      placeholder="0.00"
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white text-lg focus:outline-none focus:border-green-500/50 transition-colors"
                    />
                    <div className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400">
                      {swapFrom}
                    </div>
                  </div>
                </div>

                {/* Frequency Selection */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">Purchase Frequency</label>
                  <div className="grid grid-cols-4 gap-2">
                    {(['hourly', 'daily', 'weekly', 'monthly'] as const).map((interval) => (
                      <button
                        key={interval}
                        onClick={() => setDcaInterval(interval)}
                        className={`py-2 px-3 rounded-lg font-medium text-sm transition-all ${
                          dcaInterval === interval
                            ? 'bg-gradient-to-r from-green-500 to-emerald-600 text-white'
                            : 'bg-white/5 text-gray-400 hover:bg-white/10'
                        }`}
                      >
                        {interval.charAt(0).toUpperCase() + interval.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Max Executions (Optional) */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">
                    Number of Purchases <span className="text-gray-500">(optional)</span>
                  </label>
                  <input
                    type="number"
                    value={dcaMaxExecutions}
                    onChange={(e) => setDcaMaxExecutions(e.target.value)}
                    placeholder="Unlimited"
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-green-500/50 transition-colors"
                  />
                </div>

                {/* Summary */}
                <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-xl">
                  <div className="text-sm text-green-300">
                    {dcaAmount && parseFloat(dcaAmount) > 0 ? (
                      <>
                        Will spend <strong>{dcaAmount} {swapFrom}</strong> {dcaInterval} to buy <strong>{swapTo}</strong>
                        {dcaMaxExecutions && parseInt(dcaMaxExecutions) > 0 ? (
                          <> for <strong>{dcaMaxExecutions}</strong> times</>
                        ) : (
                          <> indefinitely</>
                        )}
                      </>
                    ) : (
                      'Configure your DCA strategy above'
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowDcaModal(false)}
                    className="flex-1 py-3 px-6 bg-white/5 rounded-xl text-white font-medium hover:bg-white/10 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={async () => {
                      if (!dcaAmount || parseFloat(dcaAmount) <= 0) {
                        alert('Please enter a valid amount');
                        return;
                      }

                      const walletAddress = localStorage.getItem('walletAddress');
                      if (!walletAddress) {
                        alert('Please connect your wallet first');
                        return;
                      }

                      setLoadingDca(true);
                      try {
                        // v4.0.11: ALL amounts use 24 decimals (internal pool format)
                        const fromTokenData = findToken(swapFrom);
                        const dcaAmountBigInt = parseAmountToBigInt(dcaAmount, 24);
                        const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/dca/orders`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({
                            wallet_address: walletAddress,
                            from_token: swapFrom === 'QUG' ? 'QUG' : fromTokenData?.id || swapFrom,
                            to_token: swapTo === 'QUG' ? 'QUG' : findToken(swapTo)?.id || swapTo,
                            amount_per_execution: dcaAmountBigInt.toString(),
                            interval: dcaInterval,
                            max_slippage: 0.03,
                            max_executions: dcaMaxExecutions ? parseInt(dcaMaxExecutions) : null,
                          }),
                        });

                        const data = await response.json();
                        if (data.success) {
                          alert(`✅ DCA order created! First purchase scheduled for ${new Date(data.next_execution_at).toLocaleString()}`);
                          setShowDcaModal(false);
                          setDcaAmount('');
                          setDcaMaxExecutions('');
                          fetchDcaOrders(); // Refresh the orders list
                        } else {
                          alert(`❌ Failed to create DCA order: ${data.message}`);
                        }
                      } catch (error) {
                        console.error('DCA creation error:', error);
                        alert('❌ Failed to create DCA order. Please try again.');
                      } finally {
                        setLoadingDca(false);
                      }
                    }}
                    disabled={loadingDca || !dcaAmount || parseFloat(dcaAmount) <= 0}
                    className="flex-1 py-3 px-6 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl text-white font-bold hover:shadow-lg hover:shadow-green-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loadingDca ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Creating...
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        Start DCA
                      </>
                    )}
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
