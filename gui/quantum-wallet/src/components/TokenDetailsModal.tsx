import { motion } from 'framer-motion';
import { X, TrendingUp, TrendingDown, ExternalLink, Info, Droplet, Zap, Shield, Coins, Users, Activity, ArrowUpDown, ArrowUp, ArrowDown, Filter } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { qnkAPI } from '../services/api';

interface TokenDetails {
  id: string;
  symbol: string;
  name: string;
  icon: string;
  price: number;
  change24h: number;
  marketCap: number;
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

      // Calculate price range
      const prices = priceData.map(d => d.price);
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
      priceData.forEach((point, i) => {
        const x = padding + (width - 2 * padding) * (i / (priceData.length - 1));
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
      priceData.forEach((point, i) => {
        const x = padding + (width - 2 * padding) * (i / (priceData.length - 1));
        const y = height - padding - ((point.price - minPrice) / priceRange) * (height - 2 * padding);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw price labels
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.font = '12px monospace';
      ctx.textAlign = 'right';
      for (let i = 0; i <= 5; i++) {
        const price = maxPrice - (priceRange * (i / 5));
        const y = padding + (height - 2 * padding) * (i / 5);
        ctx.fillText(`$${price.toFixed(4)}`, padding - 10, y + 4);
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

    // Calculate which data point we're hovering over
    const dataIndex = Math.floor(((x - padding) / (width - 2 * padding)) * priceData.length);
    if (dataIndex >= 0 && dataIndex < priceData.length) {
      setHoveredPoint(priceData[dataIndex]);
      setMousePosition({ x, y });
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

  const formatLargeNumber = (num: number) => {
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `$${(num / 1e3).toFixed(2)}K`;
    return `$${num.toFixed(2)}`;
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
                <div className="w-16 h-16 bg-gradient-to-br from-quantum-cyan to-quantum-purple rounded-2xl flex items-center justify-center text-3xl shadow-lg">
                  {(token.icon === 'qug-logo' || token.icon === 'qugusd-logo' || token.icon === 'usd-logo') ? (
                    <div className="relative w-12 h-12">
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
                    token.icon
                  )}
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
                ${hoveredPoint ? hoveredPoint.price.toFixed(4) : token.price.toLocaleString()}
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
              {/* Stats Grid */}
              <div>
                <h3 className="text-xl font-bold text-white mb-4">Market Stats</h3>
                <div className="grid grid-cols-2 gap-3">
                  <StatCard
                    icon={<Activity className="w-5 h-5" />}
                    label="Market Cap"
                    value={formatLargeNumber(token.marketCap)}
                    color="from-cyan-500 to-blue-500"
                  />
                  <StatCard
                    icon={<Coins className="w-5 h-5" />}
                    label="Total Supply"
                    value={formatLargeNumber(token.totalSupply)}
                    color="from-purple-500 to-pink-500"
                  />
                  <StatCard
                    icon={<Droplet className="w-5 h-5" />}
                    label="Liquidity"
                    value={formatLargeNumber(token.liquidity)}
                    color="from-green-500 to-teal-500"
                  />
                  <StatCard
                    icon={<Users className="w-5 h-5" />}
                    label="Holders"
                    value={token.holders.toLocaleString()}
                    color="from-orange-500 to-red-500"
                  />
                </div>
              </div>

              {/* Transaction Fees */}
              <div>
                <h3 className="text-xl font-bold text-white mb-4">Transaction Fees</h3>
                <div className="grid grid-cols-3 gap-3">
                  <FeeCard label="Buy" percentage={token.fees.buy} />
                  <FeeCard label="Sell" percentage={token.fees.sell} />
                  <FeeCard label="Transfer" percentage={token.fees.transfer} />
                </div>
              </div>

              {/* Token Features */}
              <div>
                <h3 className="text-xl font-bold text-white mb-4">Token Features</h3>
                <div className="grid grid-cols-1 gap-3">
                  <FeatureCard
                    icon={<Droplet className="w-5 h-5" />}
                    title="Reflection"
                    description="Earn passive rewards from every transaction"
                    active={token.features.reflection}
                  />
                  <FeatureCard
                    icon={<Zap className="w-5 h-5" />}
                    title="Auto-Liquidity"
                    description="Automatic liquidity pool growth"
                    active={token.features.autoLiquidity}
                  />
                  <FeatureCard
                    icon={<Activity className="w-5 h-5" />}
                    title="Buyback & Burn"
                    description="Deflationary token mechanics"
                    active={token.features.buybackAndBurn}
                  />
                  <FeatureCard
                    icon={<Shield className="w-5 h-5" />}
                    title="Anti-Whale"
                    description="Protection against large holders"
                    active={token.features.antiWhale}
                  />
                  <FeatureCard
                    icon={<Shield className="w-5 h-5" />}
                    title="Quantum Security"
                    description="Post-quantum cryptographic protection"
                    active={token.features.quantumSecured}
                  />
                </div>
              </div>

              {/* Description */}
              <div>
                <h3 className="text-xl font-bold text-white mb-4">About {token.name}</h3>
                <p className="text-gray-300 leading-relaxed text-sm">{token.description}</p>
              </div>
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
                              {tx.from.slice(0, 6)}...{tx.from.slice(-4)}
                            </span>
                          </td>

                          {/* To */}
                          <td className="px-4 py-3">
                            <span className="text-xs font-mono text-gray-400">
                              {tx.to.slice(0, 6)}...{tx.to.slice(-4)}
                            </span>
                          </td>

                          {/* Tx Hash */}
                          <td className="px-4 py-3 text-center">
                            <a
                              href={`https://explorer.quillon.xyz/tx/${tx.txHash}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs font-mono text-quantum-cyan hover:text-quantum-purple transition-colors inline-flex items-center gap-1"
                            >
                              {tx.txHash.slice(0, 6)}...
                              <ExternalLink className="w-3 h-3" />
                            </a>
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
