import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  Shield,
  Clock,
  Users,
  Cpu,
  Database,
  DollarSign,
  Layers,
  ArrowUpDown,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Info,
  Pickaxe,
  Globe,
} from 'lucide-react';
import { qnkAPI, type EmissionStats, type EmissionDailyRecord } from '../services/api';
import { TICKER_SYMBOL } from '../constants/ticker';

// ═══════════════════════════════════════════════════════════════
// AnalyticsScreen — Full blockchain analytics dashboard
// Shows emission tracking, network health, supply economics,
// and historical charts for the Quillon Graph network.
// ═══════════════════════════════════════════════════════════════

interface NetworkOverview {
  blockHeight: number;
  peers: number;
  networkHealth: string;
  tps: number;
  mempoolSize: number;
  hashRate: number;
  hashRateFormatted: string;
  blockReward: number;
  blockRewardFormatted: string;
  totalMined: number;
  totalMinedFormatted: string;
  maxSupply: number;
  maxSupplyFormatted: string;
  circulatingPct: number;
  connectedMiners: number;
}

type TimeRange = '7d' | '30d' | '90d' | 'all';

function formatNumber(n: number, decimals = 2): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(decimals)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(decimals)}K`;
  return n.toFixed(decimals);
}

function formatHashRate(hps: number): string {
  if (hps >= 1e12) return `${(hps / 1e12).toFixed(2)} TH/s`;
  if (hps >= 1e9) return `${(hps / 1e9).toFixed(2)} GH/s`;
  if (hps >= 1e6) return `${(hps / 1e6).toFixed(2)} MH/s`;
  if (hps >= 1e3) return `${(hps / 1e3).toFixed(2)} kH/s`;
  return `${hps.toFixed(0)} H/s`;
}

// Mini sparkline chart using SVG
function Sparkline({ data, color = '#22d3ee', height = 40, width = 120 }: {
  data: number[];
  color?: string;
  height?: number;
  width?: number;
}) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id={`spark-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Area fill */}
      <polygon
        points={`0,${height} ${points} ${width},${height}`}
        fill={`url(#spark-${color.replace('#', '')})`}
      />
    </svg>
  );
}

// Emission daily bar chart
function EmissionBarChart({ data, timeRange }: { data: EmissionDailyRecord[]; timeRange: TimeRange }) {
  const filtered = (() => {
    if (timeRange === 'all') return data;
    const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
    return data.slice(-days);
  })();

  if (filtered.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
        No emission data available
      </div>
    );
  }

  const maxEmitted = Math.max(...filtered.map(d => d.emitted_qug), 1);
  const targetDaily = filtered[0]?.target_daily_qug || 7186;

  return (
    <div className="relative h-56 flex items-end gap-[2px] px-2">
      {/* Target line */}
      <div
        className="absolute left-0 right-0 border-t border-dashed border-amber-400/40"
        style={{ bottom: `${(targetDaily / maxEmitted) * 100}%` }}
      >
        <span className="absolute -top-4 right-0 text-[9px] text-amber-400/70">
          Target: {formatNumber(targetDaily)} {TICKER_SYMBOL}/day
        </span>
      </div>

      {filtered.map((day, i) => {
        const pct = (day.emitted_qug / maxEmitted) * 100;
        const isOverTarget = day.emitted_qug > targetDaily * 1.05;
        const isUnderTarget = day.emitted_qug < targetDaily * 0.95;
        const barColor = isOverTarget
          ? 'from-red-500/80 to-red-600/60'
          : isUnderTarget
          ? 'from-yellow-500/80 to-yellow-600/60'
          : 'from-cyan-400/80 to-cyan-600/60';

        return (
          <div
            key={day.date}
            className="flex-1 min-w-[3px] relative group"
            style={{ height: '100%' }}
          >
            <motion.div
              initial={{ height: 0 }}
              animate={{ height: `${pct}%` }}
              transition={{ delay: i * 0.01, duration: 0.4 }}
              className={`absolute bottom-0 left-0 right-0 rounded-t bg-gradient-to-t ${barColor}`}
            />
            {/* Tooltip on hover */}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-50">
              <div className="bg-slate-900/95 border border-cyan-500/30 rounded-lg p-2 text-[10px] whitespace-nowrap shadow-xl">
                <div className="text-cyan-400 font-bold">{day.date}</div>
                <div className="text-white">{day.emitted_qug.toFixed(2)} {TICKER_SYMBOL}</div>
                <div className="text-gray-400">{day.blocks} blocks</div>
                <div className={day.deviation_pct > 5 ? 'text-red-400' : day.deviation_pct < -5 ? 'text-yellow-400' : 'text-green-400'}>
                  {day.deviation_pct > 0 ? '+' : ''}{day.deviation_pct.toFixed(1)}% from target
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Supply donut chart
function SupplyDonut({ pctMined }: { pctMined: number }) {
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const filled = (pctMined / 100) * circumference;

  return (
    <div className="relative w-36 h-36">
      <svg viewBox="0 0 128 128" className="w-full h-full -rotate-90">
        <circle cx="64" cy="64" r={radius} fill="none" stroke="rgba(148,163,184,0.1)" strokeWidth="10" />
        <motion.circle
          cx="64"
          cy="64"
          r={radius}
          fill="none"
          stroke="url(#donut-gradient)"
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - filled }}
          transition={{ duration: 1.5, ease: 'easeOut' }}
        />
        <defs>
          <linearGradient id="donut-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22d3ee" />
            <stop offset="50%" stopColor="#a78bfa" />
            <stop offset="100%" stopColor="#f59e0b" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-white">{pctMined.toFixed(2)}%</span>
        <span className="text-[10px] text-gray-400 uppercase tracking-wider">Mined</span>
      </div>
    </div>
  );
}

export default function AnalyticsScreen() {
  const [networkOverview, setNetworkOverview] = useState<NetworkOverview | null>(null);
  const [emissionStats, setEmissionStats] = useState<EmissionStats | null>(null);
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [statusRes, supplyRes, emissionRes] = await Promise.all([
        qnkAPI.getNodeStatus(),
        qnkAPI.getNetworkSupply(),
        qnkAPI.getEmissionStats(90),
      ]);

      if (statusRes.success && statusRes.data) {
        const s = statusRes.data;
        const sup = supplyRes.success && supplyRes.data ? supplyRes.data : null;

        setNetworkOverview({
          blockHeight: s.current_height || 0,
          peers: s.connected_peers || 0,
          networkHealth: s.network_health || 'unknown',
          tps: s.tps_current || 0,
          mempoolSize: s.tx_pool_size || 0,
          hashRate: sup?.network_hashrate || 0,
          hashRateFormatted: sup?.network_hashrate_formatted || '0 H/s',
          blockReward: sup?.block_reward || 0,
          blockRewardFormatted: sup?.block_reward_formatted || '0',
          totalMined: sup?.total_mined || 0,
          totalMinedFormatted: sup?.total_mined_formatted || '0',
          maxSupply: sup?.max_supply || 21_000_000,
          maxSupplyFormatted: sup?.max_supply_formatted || '21,000,000',
          circulatingPct: sup?.circulating_percentage || 0,
          connectedMiners: sup?.connected_miners || 0,
        });
      }

      if (emissionRes.success && emissionRes.data) {
        setEmissionStats(emissionRes.data);
      }

      setLastRefresh(new Date());
    } catch (err) {
      console.error('[AnalyticsScreen] Failed to fetch analytics data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30_000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [fetchData]);

  const toggleSection = (id: string) => {
    setExpandedSection(prev => (prev === id ? null : id));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
        >
          <Activity className="w-8 h-8 text-cyan-400" />
        </motion.div>
        <span className="ml-3 text-gray-400 text-sm">Loading analytics...</span>
      </div>
    );
  }

  const summary = emissionStats?.summary;
  const schedule = emissionStats?.schedule;
  const dailyHistory = emissionStats?.daily_history || [];
  const rateDiag = emissionStats?.rate_diagnostics;

  // Compute sparkline data
  const emissionSparkline = dailyHistory.slice(-14).map(d => d.emitted_qug);
  const blockSparkline = dailyHistory.slice(-14).map(d => d.blocks);
  const deviationSparkline = dailyHistory.slice(-14).map(d => d.deviation_pct);

  // Today vs yesterday comparison
  const todayEmission = summary?.today_emitted_qug || 0;
  const yesterdayEmission = dailyHistory.length >= 2 ? dailyHistory[dailyHistory.length - 2]?.emitted_qug || 0 : 0;
  const emissionChange = yesterdayEmission > 0 ? ((todayEmission - yesterdayEmission) / yesterdayEmission) * 100 : 0;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <BarChart3 className="w-7 h-7 text-cyan-400" />
            Network Analytics
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Real-time Quillon Graph blockchain metrics and emission tracking
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-gray-600">
            Updated {lastRefresh.toLocaleTimeString()}
          </span>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => { setLoading(true); fetchData(); }}
            className="p-2 rounded-lg bg-white/5 border border-white/10 hover:border-cyan-500/30 transition-colors"
          >
            <RefreshCw className="w-4 h-4 text-gray-400" />
          </motion.button>
        </div>
      </div>

      {/* ═══ Network Overview Cards ═══ */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          {
            label: 'Block Height',
            value: networkOverview ? networkOverview.blockHeight.toLocaleString() : '--',
            icon: Layers,
            color: 'text-cyan-400',
            borderColor: 'border-cyan-500/20',
            bgGrad: 'from-cyan-500/10 to-cyan-600/5',
          },
          {
            label: 'Network Hash Rate',
            value: networkOverview ? formatHashRate(networkOverview.hashRate) : '--',
            icon: Cpu,
            color: 'text-purple-400',
            borderColor: 'border-purple-500/20',
            bgGrad: 'from-purple-500/10 to-purple-600/5',
          },
          {
            label: 'Connected Peers',
            value: networkOverview ? networkOverview.peers.toString() : '--',
            icon: Users,
            color: 'text-emerald-400',
            borderColor: 'border-emerald-500/20',
            bgGrad: 'from-emerald-500/10 to-emerald-600/5',
          },
          {
            label: 'Active Miners',
            value: networkOverview ? networkOverview.connectedMiners.toString() : '--',
            icon: Pickaxe,
            color: 'text-amber-400',
            borderColor: 'border-amber-500/20',
            bgGrad: 'from-amber-500/10 to-amber-600/5',
          },
        ].map((card, i) => (
          <motion.div
            key={card.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className={`bg-gradient-to-br ${card.bgGrad} backdrop-blur-xl border ${card.borderColor} rounded-xl p-4`}
          >
            <div className="flex items-center justify-between mb-2">
              <card.icon className={`w-5 h-5 ${card.color}`} />
              <span className="text-[10px] text-gray-500 uppercase tracking-wider">{card.label}</span>
            </div>
            <div className="text-xl font-bold text-white">{card.value}</div>
          </motion.div>
        ))}
      </div>

      {/* ═══ Supply Economics ═══ */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="bg-gradient-to-br from-[#030818]/80 to-[#020210]/90 backdrop-blur-xl border border-cyan-500/20 rounded-xl p-6"
      >
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-cyan-400" />
            Supply Economics
          </h2>
          {schedule && (
            <span className="text-xs text-gray-500">
              Era {summary?.current_era || 0} / {schedule.total_eras} | Halving every {schedule.halving_interval_years} years
            </span>
          )}
        </div>

        <div className="flex flex-col md:flex-row items-center gap-8">
          {/* Donut chart */}
          <SupplyDonut pctMined={summary?.pct_mined || networkOverview?.circulatingPct || 0} />

          {/* Stats grid */}
          <div className="flex-1 grid grid-cols-2 gap-4">
            <div>
              <span className="text-[10px] text-gray-500 uppercase tracking-wider">Total Mined</span>
              <p className="text-lg font-bold text-white">
                {summary ? formatNumber(summary.total_supply_qug) : networkOverview?.totalMinedFormatted || '0'} {TICKER_SYMBOL}
              </p>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 uppercase tracking-wider">Max Supply</span>
              <p className="text-lg font-bold text-white">
                {summary ? formatNumber(summary.max_supply_qug) : '21M'} {TICKER_SYMBOL}
              </p>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 uppercase tracking-wider">Remaining</span>
              <p className="text-lg font-bold text-cyan-400">
                {summary?.remaining_supply_qug ? formatNumber(summary.remaining_supply_qug) : '--'} {TICKER_SYMBOL}
              </p>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 uppercase tracking-wider">Block Reward</span>
              <p className="text-lg font-bold text-amber-400">
                {summary?.reward_per_block_qug
                  ? `${summary.reward_per_block_qug.toFixed(6)}`
                  : networkOverview?.blockRewardFormatted || '0'} {TICKER_SYMBOL}
              </p>
            </div>
            {summary?.stock_to_flow != null && (
              <div>
                <span className="text-[10px] text-gray-500 uppercase tracking-wider">Stock-to-Flow</span>
                <p className="text-lg font-bold text-purple-400">{summary.stock_to_flow.toFixed(2)}</p>
              </div>
            )}
            {summary?.inflation_rate_pct != null && (
              <div>
                <span className="text-[10px] text-gray-500 uppercase tracking-wider">Inflation Rate</span>
                <p className="text-lg font-bold text-yellow-400">{summary.inflation_rate_pct.toFixed(2)}%</p>
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* ═══ Emission Tracking ═══ */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-gradient-to-br from-[#030818]/80 to-[#020210]/90 backdrop-blur-xl border border-cyan-500/20 rounded-xl p-6"
      >
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-emerald-400" />
            Emission Tracking
          </h2>
          <div className="flex items-center gap-1">
            {(['7d', '30d', '90d', 'all'] as TimeRange[]).map(range => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                  timeRange === range
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
                    : 'text-gray-500 hover:text-gray-300 border border-transparent'
                }`}
              >
                {range.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Today's snapshot */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase">Today Emitted</span>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-lg font-bold text-white">{todayEmission.toFixed(2)}</span>
              <span className="text-xs text-gray-500">{TICKER_SYMBOL}</span>
            </div>
            {emissionChange !== 0 && (
              <div className={`flex items-center gap-1 text-[10px] mt-1 ${emissionChange > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {emissionChange > 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                {Math.abs(emissionChange).toFixed(1)}% vs yesterday
              </div>
            )}
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase">Today Blocks</span>
            <div className="text-lg font-bold text-white mt-1">{summary?.today_blocks?.toLocaleString() || '--'}</div>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase">Block Rate</span>
            <div className="text-lg font-bold text-cyan-400 mt-1">{summary?.block_rate_bps?.toFixed(3) || '--'} bps</div>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase">Deviation</span>
            <div className={`text-lg font-bold mt-1 ${
              (summary?.today_deviation_pct || 0) > 5 ? 'text-red-400'
              : (summary?.today_deviation_pct || 0) < -5 ? 'text-yellow-400'
              : 'text-emerald-400'
            }`}>
              {summary?.today_deviation_pct != null ? `${summary.today_deviation_pct > 0 ? '+' : ''}${summary.today_deviation_pct.toFixed(1)}%` : '--'}
            </div>
          </div>
        </div>

        {/* Sparklines row */}
        <div className="grid grid-cols-3 gap-4 mb-5">
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase mb-2 block">Emission (14d)</span>
            <Sparkline data={emissionSparkline} color="#22d3ee" width={180} height={36} />
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase mb-2 block">Blocks/day (14d)</span>
            <Sparkline data={blockSparkline} color="#a78bfa" width={180} height={36} />
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase mb-2 block">Deviation % (14d)</span>
            <Sparkline data={deviationSparkline} color="#f59e0b" width={180} height={36} />
          </div>
        </div>

        {/* Daily emission bar chart */}
        <div className="bg-white/5 rounded-lg p-4 border border-white/5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs text-gray-400 font-semibold">Daily Emission ({timeRange.toUpperCase()})</span>
            <span className="text-[10px] text-gray-600">
              Target: {summary?.daily_target_qug?.toFixed(0) || '7,186'} {TICKER_SYMBOL}/day
            </span>
          </div>
          <EmissionBarChart data={dailyHistory} timeRange={timeRange} />
        </div>
      </motion.div>

      {/* ═══ Emission Schedule ═══ */}
      {schedule && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="bg-gradient-to-br from-[#030818]/80 to-[#020210]/90 backdrop-blur-xl border border-purple-500/20 rounded-xl p-6"
        >
          <button
            onClick={() => toggleSection('schedule')}
            className="w-full flex items-center justify-between"
          >
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Clock className="w-5 h-5 text-purple-400" />
              Emission Schedule
            </h2>
            {expandedSection === 'schedule' ? (
              <ChevronUp className="w-5 h-5 text-gray-500" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-500" />
            )}
          </button>

          <AnimatePresence>
            {expandedSection === 'schedule' && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="overflow-hidden"
              >
                <div className="mt-5 space-y-4">
                  {/* Era progress bar */}
                  {summary?.era_progress_pct != null && (
                    <div>
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-gray-400">Era {summary.current_era} Progress</span>
                        <span className="text-purple-400">{summary.era_progress_pct.toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${summary.era_progress_pct}%` }}
                          transition={{ duration: 1 }}
                          className="h-full rounded-full bg-gradient-to-r from-purple-500 to-cyan-500"
                        />
                      </div>
                      {summary.secs_to_halving != null && (
                        <p className="text-[10px] text-gray-600 mt-1">
                          Next halving in ~{Math.round(summary.secs_to_halving / 86400)} days
                        </p>
                      )}
                    </div>
                  )}

                  {/* Schedule table */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                      <span className="text-[10px] text-gray-500 uppercase">Era 0 Annual</span>
                      <p className="text-sm font-bold text-white mt-1">{formatNumber(schedule.era_0_annual)} {TICKER_SYMBOL}</p>
                    </div>
                    <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                      <span className="text-[10px] text-gray-500 uppercase">Era 0 Daily</span>
                      <p className="text-sm font-bold text-white mt-1">{formatNumber(schedule.era_0_daily)} {TICKER_SYMBOL}</p>
                    </div>
                    <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                      <span className="text-[10px] text-gray-500 uppercase">Era 1 Annual</span>
                      <p className="text-sm font-bold text-white mt-1">{formatNumber(schedule.era_1_annual)} {TICKER_SYMBOL}</p>
                    </div>
                    <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                      <span className="text-[10px] text-gray-500 uppercase">Total Eras</span>
                      <p className="text-sm font-bold text-white mt-1">{schedule.total_eras}</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}

      {/* ═══ Rate Diagnostics (Advanced) ═══ */}
      {rateDiag && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-br from-[#030818]/80 to-[#020210]/90 backdrop-blur-xl border border-amber-500/20 rounded-xl p-6"
        >
          <button
            onClick={() => toggleSection('diagnostics')}
            className="w-full flex items-center justify-between"
          >
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-400" />
              Rate Diagnostics
              <span className="text-[10px] bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full border border-amber-500/30 ml-2">
                ADVANCED
              </span>
            </h2>
            {expandedSection === 'diagnostics' ? (
              <ChevronUp className="w-5 h-5 text-gray-500" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-500" />
            )}
          </button>

          <AnimatePresence>
            {expandedSection === 'diagnostics' && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="overflow-hidden"
              >
                <div className="mt-5 grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <span className="text-[10px] text-gray-500 uppercase">Method</span>
                    <p className="text-sm font-bold text-cyan-400 mt-1">{rateDiag.active_method}</p>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <span className="text-[10px] text-gray-500 uppercase">Confidence</span>
                    <p className={`text-sm font-bold mt-1 ${rateDiag.confidence_pct > 80 ? 'text-emerald-400' : rateDiag.confidence_pct > 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {rateDiag.confidence_pct.toFixed(0)}%
                    </p>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <span className="text-[10px] text-gray-500 uppercase">Smoothed Rate</span>
                    <p className="text-sm font-bold text-white mt-1">{rateDiag.smoothed_rate_bps.toFixed(4)} bps</p>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <span className="text-[10px] text-gray-500 uppercase">Correction Factor</span>
                    <p className="text-sm font-bold text-purple-400 mt-1">{rateDiag.correction_factor.toFixed(4)}</p>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <span className="text-[10px] text-gray-500 uppercase">Actual Emission/hr</span>
                    <p className="text-sm font-bold text-white mt-1">{rateDiag.actual_emission_rate_qug_per_hour.toFixed(2)} {TICKER_SYMBOL}</p>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <span className="text-[10px] text-gray-500 uppercase">Target Emission/hr</span>
                    <p className="text-sm font-bold text-white mt-1">{rateDiag.target_emission_rate_qug_per_hour.toFixed(2)} {TICKER_SYMBOL}</p>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <span className="text-[10px] text-gray-500 uppercase">Budget Error</span>
                    <p className={`text-sm font-bold mt-1 ${Math.abs(rateDiag.error_fraction_pct) < 5 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {rateDiag.error_fraction_pct > 0 ? '+' : ''}{rateDiag.error_fraction_pct.toFixed(2)}%
                    </p>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <span className="text-[10px] text-gray-500 uppercase">Convergence ETA</span>
                    <p className="text-sm font-bold text-white mt-1">
                      {rateDiag.convergence_eta_secs != null
                        ? rateDiag.convergence_eta_secs > 86400
                          ? `${(rateDiag.convergence_eta_secs / 86400).toFixed(1)}d`
                          : `${(rateDiag.convergence_eta_secs / 3600).toFixed(1)}h`
                        : 'N/A'}
                    </p>
                  </div>
                </div>

                {/* Window diagnostics */}
                <div className="mt-4 bg-white/5 rounded-lg p-4 border border-white/5">
                  <h3 className="text-xs text-gray-400 font-semibold mb-3">Measurement Windows</h3>
                  <div className="grid grid-cols-3 gap-4 text-xs">
                    <div>
                      <span className="text-gray-500 block mb-1">Sliding Window</span>
                      <span className="text-white font-mono">{rateDiag.window_rate_bps.toFixed(4)} bps</span>
                      <span className="text-gray-600 block">{rateDiag.window_blocks} blocks / {(rateDiag.window_elapsed_secs / 3600).toFixed(1)}h</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block mb-1">Cumulative</span>
                      <span className="text-white font-mono">{rateDiag.cumulative_rate_bps.toFixed(4)} bps</span>
                      <span className="text-gray-600 block">{rateDiag.cumulative_blocks} blocks / {(rateDiag.cumulative_elapsed_secs / 3600).toFixed(1)}h</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block mb-1">Block Timestamp</span>
                      <span className="text-white font-mono">{rateDiag.block_timestamp_rate_bps.toFixed(4)} bps</span>
                      <span className="text-gray-600 block">{rateDiag.block_timestamp_windows} windows</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}

      {/* ═══ Network Health ═══ */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
        className="bg-gradient-to-br from-[#030818]/80 to-[#020210]/90 backdrop-blur-xl border border-emerald-500/20 rounded-xl p-6"
      >
        <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-5">
          <Shield className="w-5 h-5 text-emerald-400" />
          Network Health
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase">Status</span>
            <div className="flex items-center gap-2 mt-1">
              <div className={`w-2.5 h-2.5 rounded-full ${networkOverview?.networkHealth === 'healthy' ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
              <span className={`text-sm font-bold capitalize ${networkOverview?.networkHealth === 'healthy' ? 'text-emerald-400' : 'text-red-400'}`}>
                {networkOverview?.networkHealth || 'Unknown'}
              </span>
            </div>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase">TPS</span>
            <p className="text-sm font-bold text-white mt-1">{networkOverview?.tps?.toFixed(2) || '0'}</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase">Mempool Size</span>
            <p className="text-sm font-bold text-white mt-1">{networkOverview?.mempoolSize?.toLocaleString() || '0'}</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase">Connected Peers</span>
            <p className="text-sm font-bold text-white mt-1">{networkOverview?.peers || 0}</p>
          </div>
        </div>
      </motion.div>

      {/* Footer note */}
      <div className="text-center text-[10px] text-gray-700 pb-4">
        Data refreshes every 30 seconds from /api/v1/ endpoints. Emission data from /api/v1/emission/stats.
      </div>
    </div>
  );
}
