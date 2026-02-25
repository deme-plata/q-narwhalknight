import { motion } from 'framer-motion';
import { Wallet, TrendingUp, TrendingDown } from 'lucide-react';
import { memo, useMemo } from 'react';

interface BalanceHistoryPoint {
  timestamp: number;
  balance: number;
}

interface WalletCardProps {
  wallet: {
    symbol: string;
    name: string;
    balance: number;
    usdValue?: number;
    icon: 'qug' | 'usd' | 'btc' | 'eth' | 'sol' | 'zec' | 'iron' | 'custom';
    color: string;
    comingSoon?: boolean;
    shieldedOnly?: boolean;
    history?: BalanceHistoryPoint[];
  };
  index: number;
  isAnimating: boolean;
  onCardClick?: () => void;
  children?: React.ReactNode;
}

// Lightweight SVG sparkline component with quantum glow
const MiniGraph = memo(function MiniGraph({
  data,
  color,
  width = 200,
  height = 60
}: {
  data: BalanceHistoryPoint[];
  color: string;
  width?: number;
  height?: number;
}) {
  const { path, gradient, trend, percentChange, plotData } = useMemo(() => {
    if (data.length < 2) {
      return { path: '', gradient: '', trend: 0, percentChange: 0, plotData: [] as BalanceHistoryPoint[] };
    }

    // Sort by timestamp and deduplicate near-identical balance points
    // This prevents zigzag from tiny mining increments
    const sorted = [...data].sort((a, b) => a.timestamp - b.timestamp);
    const deduped: BalanceHistoryPoint[] = [];
    for (const point of sorted) {
      if (deduped.length === 0) {
        deduped.push(point);
        continue;
      }
      const last = deduped[deduped.length - 1];
      const pctDiff = last.balance > 0
        ? Math.abs(point.balance - last.balance) / last.balance
        : (point.balance !== last.balance ? 1 : 0);
      // Only keep point if balance changed by >0.5% or >5 seconds apart with any change
      if (pctDiff > 0.005 || (pctDiff > 0 && point.timestamp - last.timestamp > 5000)) {
        deduped.push(point);
      }
    }
    // Always include the latest point so graph shows current balance
    const lastSorted = sorted[sorted.length - 1];
    if (deduped.length > 0 && deduped[deduped.length - 1].timestamp !== lastSorted.timestamp) {
      deduped.push(lastSorted);
    }
    // Need at least 2 unique points
    const plotData = deduped.length >= 2 ? deduped : sorted;
    if (plotData.length < 2) {
      return { path: '', gradient: '', trend: 0, percentChange: 0 };
    }

    const values = plotData.map(d => d.balance);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    // Calculate trend and percentage change
    const firstValue = values[0] || 0;
    const lastValue = values[values.length - 1] || 0;
    const change = lastValue - firstValue;
    const pctChange = firstValue !== 0 ? (change / firstValue) * 100 : 0;
    const trendDirection = change > 0 ? 1 : change < 0 ? -1 : 0;

    // Create SVG path points
    const points = plotData.map((d, i) => {
      const x = (i / (plotData.length - 1)) * width;
      const y = height - ((d.balance - min) / range) * (height - 10) - 5;
      return { x, y };
    });

    // Build smooth bezier curve path
    let pathData = `M ${points[0].x},${points[0].y}`;

    for (let i = 0; i < points.length - 1; i++) {
      const curr = points[i];
      const next = points[i + 1];
      const midX = (curr.x + next.x) / 2;

      // Smooth bezier curve
      pathData += ` Q ${curr.x},${curr.y} ${midX},${(curr.y + next.y) / 2}`;
      pathData += ` Q ${next.x},${next.y} ${next.x},${next.y}`;
    }

    // Create gradient fill area
    const lastPoint = points[points.length - 1];
    const gradientPath = `${pathData} L ${lastPoint.x},${height} L ${points[0].x},${height} Z`;

    return {
      path: pathData,
      gradient: gradientPath,
      trend: trendDirection,
      percentChange: pctChange,
      plotData
    };
  }, [data, width, height]);

  if (data.length < 2) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-xs">
        Accumulating data...
      </div>
    );
  }

  // Extract color values for gradient
  const isPositive = trend >= 0;
  const gradientColor = isPositive
    ? 'rgba(34, 197, 94, 0.3)' // green
    : 'rgba(239, 68, 68, 0.3)'; // red
  const strokeColor = isPositive ? '#22c55e' : '#ef4444';
  const glowColor = isPositive ? '#86efac' : '#fca5a5';

  return (
    <div className="relative">
      {/* Percentage change badge */}
      <div className={`absolute -top-1 right-0 px-2 py-0.5 rounded-full text-xs font-bold flex items-center gap-1 ${
        isPositive ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
      }`}>
        {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
        {Math.abs(percentChange).toFixed(1)}%
      </div>

      <svg
        width={width}
        height={height}
        className="overflow-visible"
        style={{ filter: 'drop-shadow(0 0 4px rgba(255, 255, 255, 0.1))' }}
      >
        <defs>
          {/* Gradient fill */}
          <linearGradient id={`graphGradient-${color}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={gradientColor} stopOpacity="0.8" />
            <stop offset="100%" stopColor={gradientColor} stopOpacity="0.05" />
          </linearGradient>

          {/* Glow filter for quantum effect */}
          <filter id={`glow-${color}`}>
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* Gradient fill area */}
        <motion.path
          d={gradient}
          fill={`url(#graphGradient-${color})`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        />

        {/* Main line with quantum glow */}
        <motion.path
          d={path}
          fill="none"
          stroke={strokeColor}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          filter={`url(#glow-${color})`}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, ease: "easeOut" }}
        />

        {/* Glowing dots at deduped data points (matches the path curve) */}
        {(plotData || []).map((d, i) => {
          const pts = plotData || [];
          const values = pts.map(p => p.balance);
          const min = Math.min(...values);
          const max = Math.max(...values);
          const range = max - min || 1;
          const x = (i / (pts.length - 1)) * width;
          const y = height - ((d.balance - min) / range) * (height - 10) - 5;

          return (
            <motion.circle
              key={i}
              cx={x}
              cy={y}
              r="2.5"
              fill={glowColor}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 0.8 }}
              transition={{ delay: i * 0.05, duration: 0.3 }}
              style={{
                filter: `drop-shadow(0 0 3px ${glowColor})`,
              }}
            />
          );
        })}
      </svg>

      {/* Animated grid lines */}
      <div className="absolute inset-0 pointer-events-none opacity-10">
        {[...Array(4)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute left-0 right-0 border-t border-gray-400"
            style={{ top: `${(i + 1) * 20}%` }}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ delay: 0.2 + i * 0.1, duration: 0.5 }}
          />
        ))}
      </div>
    </div>
  );
});

const WalletCardWithGraph = memo(function WalletCardWithGraph({
  wallet,
  index,
  isAnimating,
  onCardClick,
  children
}: WalletCardProps) {
  // v2.3.32-beta: For QUG, ALWAYS check localStorage for locked balance
  // This is the final defense against stale balance display
  let displayBalance = wallet.balance;
  if (wallet.symbol === 'QUG') {
    const lockedBalance = localStorage.getItem('dexLockedBalance');
    const cooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
    if (lockedBalance && Date.now() < cooldownUntil) {
      const locked = parseFloat(lockedBalance);
      if (!isNaN(locked) && isFinite(locked)) {
        displayBalance = locked;
        console.log('🔒 WalletCardWithGraph: Using LOCKED QUG balance:', locked, '(prop was:', wallet.balance, ')');
      }
    }
  }

  // Debug logging
  if (wallet.symbol === 'QUG' || wallet.symbol === 'QUGUSD' || wallet.symbol === 'USD') {
    console.log(`📊 WalletCardWithGraph rendering ${wallet.symbol}:`, JSON.stringify({
      propBalance: wallet.balance,
      displayBalance: displayBalance,
      historyLength: wallet.history?.length || 0,
    }, null, 2));
  }

  // Format balance with appropriate decimal places per currency
  const formatBalance = (amount: number) => {
    if (wallet.symbol === 'QUGUSD' || wallet.symbol === 'USD') {
      return amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    // QUG and others: show up to 6 significant decimals
    if (amount >= 1000) return amount.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (amount >= 1) return amount.toLocaleString('en-US', { maximumFractionDigits: 4 });
    return amount.toLocaleString('en-US', { maximumFractionDigits: 6 });
  };

  return (
    <motion.div
      key={wallet.symbol}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 * index }}
      className="p-4 rounded-xl cursor-pointer group relative overflow-hidden"
      style={{
        background: `linear-gradient(135deg, rgba(30, 20, 60, 0.8), rgba(50, 30, 80, 0.8))`,
        border: `2px solid rgba(212, 175, 55, ${wallet.comingSoon ? '0.1' : '0.3'})`,
      }}
      whileHover={{ scale: wallet.comingSoon ? 1 : 1.02 }}
      onClick={onCardClick}
    >
      {/* Coming Soon Badge */}
      {wallet.comingSoon && (
        <div className="absolute top-2 right-2 px-2 py-1 rounded-lg text-xs font-bold bg-gradient-to-r from-purple-500/30 to-pink-500/30 border border-purple-400/30 text-purple-300">
          Coming Soon
        </div>
      )}

      {/* Shielded Badge */}
      {wallet.shieldedOnly && (
        <div className="absolute bottom-2 right-2 px-2 py-1 rounded-lg text-xs font-bold bg-gradient-to-r from-green-500/30 to-emerald-500/30 border border-green-400/30 text-green-300 flex items-center gap-1">
          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 16 16">
            <path d="M8 1l-6 2v5c0 3.5 2.5 6.5 6 7.5 3.5-1 6-4 6-7.5V3l-6-2z"/>
          </svg>
          Shielded
        </div>
      )}

      {/* Header: Icon + Name */}
      <div className="flex items-start justify-between mb-3">
        <div className={`p-2 rounded-lg bg-gradient-to-br ${wallet.color}`}>
          {(wallet.icon === 'qug' || wallet.icon === 'usd') && (
            <div className="relative w-5 h-5">
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
          )}
          {wallet.icon === 'custom' && <Wallet className="w-5 h-5 text-white" />}
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-400 mb-1">{wallet.name}</div>
          <div className="text-lg font-bold text-white">{wallet.symbol}</div>
        </div>
      </div>

      {/* Balance Display */}
      {/* 🚨 v2.3.7-beta: Show loading state only when history is undefined (not yet fetched) */}
      <div className={`mb-3 ${wallet.comingSoon ? 'text-gray-500' : ''}`}>
        {wallet.comingSoon ? (
          <div className="text-2xl font-bold text-gray-500">0.00</div>
        ) : displayBalance === 0 && wallet.history === undefined ? (
          <div className="text-2xl font-bold text-amber-300/60 animate-pulse">Loading...</div>
        ) : (
          <motion.div
            className="text-2xl font-bold text-white"
            animate={isAnimating ? {
              textShadow: [
                '0 0 10px rgba(255, 215, 0, 0.8)',
                '0 0 20px rgba(255, 107, 0, 0.8)',
                '0 0 20px rgba(16, 185, 129, 0.8)',
                '0 0 10px rgba(255, 215, 0, 0.8)',
              ],
            } : {}}
            transition={{ duration: 1.5, repeat: isAnimating ? 1 : 0 }}
          >
            {formatBalance(displayBalance)}
          </motion.div>
        )}
        {wallet.usdValue !== undefined && !wallet.comingSoon && (
          <div className="text-xs text-gray-400 mt-1">
            ≈ ${wallet.usdValue.toFixed(2)} USD
          </div>
        )}
      </div>

      {/* Balance History Graph */}
      {!wallet.comingSoon && wallet.history && wallet.history.length >= 2 ? (
        <div className="mt-3 mb-2 h-16 relative">
          <MiniGraph
            data={wallet.history}
            color={wallet.symbol}
            width={200}
            height={60}
          />
        </div>
      ) : !wallet.comingSoon && (
        <div className="mt-3 mb-2 h-16 relative flex items-center justify-center text-xs text-gray-500">
          {wallet.history && wallet.history.length === 0
            ? 'Click to open bridge'
            : wallet.history
              ? `Waiting for data (${wallet.history.length}/2 points)`
              : 'No history data'}
        </div>
      )}

      {/* Action Buttons (children slot) */}
      {children && <div className="mt-3">{children}</div>}

      {/* Quantum shimmer overlay on hover */}
      <motion.div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none"
        style={{
          background: 'linear-gradient(135deg, transparent 30%, rgba(212, 175, 55, 0.1) 50%, transparent 70%)',
        }}
        animate={{
          x: ['-100%', '200%'],
        }}
        transition={{
          duration: 1.5,
          repeat: Infinity,
          repeatDelay: 2,
        }}
      />
    </motion.div>
  );
});

export default WalletCardWithGraph;
