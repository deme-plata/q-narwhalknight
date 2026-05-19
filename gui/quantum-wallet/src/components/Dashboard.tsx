import { useState, useEffect, useCallback, memo, useRef, lazy, Suspense } from 'react';
import { motion, AnimatePresence, Reorder } from 'framer-motion';
import { Activity, Zap, AlertCircle, Copy, Check, Wallet, ChevronLeft, ChevronRight, Calendar, DollarSign, TrendingUp, TrendingDown, QrCode, Info, Plus, Send, BarChart3, Radio, Mail, MessageCircle, Settings2, GripVertical, ArrowUp, ArrowDown, Globe, Newspaper, ExternalLink, Cpu, Shield, Layers } from 'lucide-react';
import { qnkAPI, type NodeStatus, type EmissionDailyRecord } from '../services/api'; // debounce not needed - SSE in App.tsx
import { sseManager } from '../services/sseManager';
import TransactionDetailsModal from './TransactionDetailsModal';
// 🌐 v3.4.3-browser: P2P real-time block streaming
import { useRealtimeBlocks } from '../hooks/useRealtimeBlocks';
import QRCodeModal from './QRCodeModal';
const StripeCheckout = lazy(() => import('./StripeCheckout'));
import DAGKnightVisualization from './DAGKnightVisualization';
import QNOOracleVisualization from './QNOOracleVisualization';
import LoanApplicationModal from './LoanApplicationModal';
import LoanApprovalModal from './LoanApprovalModal';
import LoanPaybackModal from './LoanPaybackModal';
import ActiveLoansCard from './ActiveLoansCard';
import WalletCardWithGraph from './WalletCardWithGraph';
import PhaseTransitionModal from './PhaseTransitionModal';
import MobileSetupModal, { MOBILE_SETUP_STORAGE_KEY } from './MobileSetupModal';
import StakingModal from './StakingModal';
import CustomTokensCard from './CustomTokensCard';
import FinanceModal from './FinanceModal';
import BitcoinSwapModal from './BitcoinSwapModal';
import HiBTDonationModal from './HiBTDonationModal';
import ZcashWalletModal from './ZcashWalletModal';
import IronFishWalletModal from './IronFishWalletModal';
import EthereumSwapModal from './EthereumSwapModal';
import EmailScreen from './EmailScreen';
import CalendarScreen from './CalendarScreen';
import WebSearchScreen from './WebSearchScreen';
import ChatScreen from './ChatScreen';
import { TICKER_SYMBOL } from '../constants/ticker';
import QuantumLoader from './QuantumLoader';
import RecentActivityPanel from './RecentActivityPanel';

// v3.6.1-beta: SANITY CHECK - Max possible balance is 21 million QUG (total supply)
// Any balance exceeding this is corrupted data and must be rejected
const MAX_QUG_SUPPLY = 21_000_000; // 21 million QUG max supply
const MAX_STABLECOIN_BALANCE = 1_000_000_000; // 1 billion (stablecoins are uncapped in practice)

/**
 * v3.6.1-beta: Validate balance value to prevent corrupted data from being cached
 * v10.2.9: Accept symbol parameter — QUGUSD/USD are stablecoins with higher caps
 */
function isValidBalance(balance: number, symbol?: string): boolean {
  if (typeof balance !== 'number') return false;
  if (isNaN(balance) || !isFinite(balance)) return false;
  if (balance < 0) return false;
  const upper = (symbol || '').toUpperCase();
  const isStablecoin = upper === 'QUGUSD' || upper === 'USD' || upper === 'QUSD';
  const cap = isStablecoin ? MAX_STABLECOIN_BALANCE : MAX_QUG_SUPPLY;
  if (balance > cap) {
    console.warn(`🚨 [Dashboard] Rejected corrupted ${symbol || 'QUG'} balance: ${balance.toExponential()} > cap ${cap}`);
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
    console.warn(`🚨 [Dashboard] safeCacheBalance: Refusing to cache invalid balance: ${balance}`);
  }
}

interface Transaction {
  id: string;
  type: 'receive' | 'send' | 'mining' | 'swap';
  amount: number;
  from?: string;
  to?: string;
  timestamp: string;
  txHash: string;
  // v3.5.8-beta: Additional fields for swaps and token transfers
  tokenSymbol?: string;
  tokenAddress?: string;
  amountOut?: string;
  tokenIn?: string;
  tokenOut?: string;
  memo?: string;
  score?: {
    total?: number;
    components?: Array<{ label: string; value: number; explanation?: string; weight?: number }>;
    weights?: Record<string, number>;
  };
}

interface BalanceHistoryPoint {
  timestamp: number;
  balance: number;
}

interface WalletBalance {
  symbol: string;
  name: string;
  balance: number;
  usdValue?: number;
  icon: 'qug' | 'usd' | 'btc' | 'eth' | 'sol' | 'zec' | 'iron' | 'custom';
  color: string;
  comingSoon?: boolean;
  shieldedOnly?: boolean; // For Privacy coins like Zcash
  history?: BalanceHistoryPoint[]; // Balance history for mini-graph
}

interface DashboardProps {
  onNavigateToSend?: (coinSymbol: string) => void;
  liveBalance?: number; // v8.6.5: Live QUG balance from App.tsx SSE (same source as TopBar)
  onNavigateToChat?: () => void; // Navigate to App-level chat screen (avoids duplicate SignalingService)
}

// ── HiBT Listing Donation Banner (v10.5.4) ────────────────────────────────
function HiBTDonationBanner() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="mb-6 cursor-pointer group"
        onClick={() => setOpen(true)}
      >
        <div
          className="relative w-full rounded-2xl overflow-hidden"
          style={{ border: '1.5px solid rgba(132,204,22,0.35)', boxShadow: '0 0 24px rgba(132,204,22,0.1)' }}
        >
          <img
            src="/hibt-banner.png"
            alt="$QUG listing on HiBT — donate BTC"
            className="w-full h-auto block transition-transform duration-300 group-hover:scale-[1.01]"
          />
          {/* Hover overlay */}
          <div
            className="absolute inset-0 flex items-center justify-end pr-6 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
            style={{ background: 'linear-gradient(90deg, transparent 40%, rgba(0,0,0,0.6) 100%)' }}
          >
            <span
              className="text-xs font-bold px-3 py-1.5 rounded-lg"
              style={{ background: 'rgba(132,204,22,0.9)', color: '#000' }}
            >
              Donate BTC →
            </span>
          </div>
        </div>
      </motion.div>
      <HiBTDonationModal isOpen={open} onClose={() => setOpen(false)} />
    </>
  );
}

// ── Emission Trajectory Visualization (v10.7.8) ──────────────────────────────
function EmissionCurveViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrame = useRef<number>(0);
  const [emData, setEmData] = useState<{
    daily: EmissionDailyRecord[];
    totalSupply: number;
    maxSupply: number;
    pctMined: number;
    era: number;
  } | null>(null);

  useEffect(() => {
    qnkAPI.getEmissionStats(90).then(resp => {
      if (resp.success && resp.data?.summary && resp.data?.daily_history?.length) {
        setEmData({
          daily: resp.data.daily_history,
          totalSupply: resp.data.summary.total_supply_qug,
          maxSupply: resp.data.summary.max_supply_qug,
          pctMined: resp.data.summary.pct_mined,
          era: resp.data.summary.current_era,
        });
      }
    });
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !emData || emData.daily.length < 2) return;

    const draw = () => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.offsetWidth;
      const H = canvas.offsetHeight;
      if (W === 0 || H === 0) { animFrame.current = requestAnimationFrame(draw); return; }
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
        ctx.scale(dpr, dpr);
      }

      const P = { t: 28, r: 24, b: 36, l: 52 };
      const pw = W - P.l - P.r;
      const ph = H - P.t - P.b;
      const maxY = emData.maxSupply;
      const days = emData.daily;
      const n = days.length;

      ctx.clearRect(0, 0, W, H);

      const toX = (i: number) => P.l + (i / (n - 1)) * pw;
      const toY = (v: number) => P.t + ph - (v / maxY) * ph;

      // Build target cumulative supply
      let cumTarget = days[0].cumulative_supply_qug - days[0].target_daily_qug;
      const targetCumuls = days.map(d => { cumTarget += d.target_daily_qug; return cumTarget; });

      // Grid lines
      for (let g = 0; g <= 4; g++) {
        const yy = P.t + (g / 4) * ph;
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(P.l, yy); ctx.lineTo(P.l + pw, yy); ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.22)';
        ctx.font = '9px "JetBrains Mono",monospace';
        ctx.textAlign = 'right';
        const label = ((maxY * (1 - g / 4)) / 1_000_000).toFixed(1) + 'M';
        ctx.fillText(label, P.l - 5, yy + 3.5);
      }

      // X-axis date ticks
      ctx.textAlign = 'center';
      ctx.font = '9px "JetBrains Mono",monospace';
      [0, Math.floor(n / 3), Math.floor((2 * n) / 3), n - 1].forEach(idx => {
        const x = toX(idx);
        ctx.strokeStyle = 'rgba(255,255,255,0.03)';
        ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.t + ph); ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.18)';
        ctx.fillText((days[idx]?.date || '').slice(5), x, P.t + ph + 18);
      });

      // Deviation fill between actual and target
      ctx.save();
      ctx.beginPath();
      days.forEach((d, i) => { i === 0 ? ctx.moveTo(toX(i), toY(d.cumulative_supply_qug)) : ctx.lineTo(toX(i), toY(d.cumulative_supply_qug)); });
      for (let i = n - 1; i >= 0; i--) ctx.lineTo(toX(i), toY(targetCumuls[i]));
      ctx.closePath();
      ctx.fillStyle = 'rgba(139,92,246,0.07)';
      ctx.fill();
      ctx.restore();

      // Target curve (dashed indigo)
      ctx.save();
      ctx.setLineDash([3, 6]);
      ctx.strokeStyle = 'rgba(139,92,246,0.55)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      targetCumuls.forEach((v, i) => { i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v)); });
      ctx.stroke();
      ctx.restore();

      // Actual curve — amber gradient fill
      ctx.save();
      const fillGrad = ctx.createLinearGradient(0, P.t, 0, P.t + ph);
      fillGrad.addColorStop(0, 'rgba(245,158,11,0.28)');
      fillGrad.addColorStop(1, 'rgba(245,158,11,0.02)');
      ctx.beginPath();
      days.forEach((d, i) => { i === 0 ? ctx.moveTo(toX(i), toY(d.cumulative_supply_qug)) : ctx.lineTo(toX(i), toY(d.cumulative_supply_qug)); });
      ctx.lineTo(toX(n - 1), P.t + ph);
      ctx.lineTo(toX(0), P.t + ph);
      ctx.closePath();
      ctx.fillStyle = fillGrad;
      ctx.fill();
      ctx.restore();

      // Actual curve line
      ctx.save();
      ctx.strokeStyle = '#F59E0B';
      ctx.lineWidth = 2.2;
      ctx.shadowColor = 'rgba(245,158,11,0.55)';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      days.forEach((d, i) => { i === 0 ? ctx.moveTo(toX(i), toY(d.cumulative_supply_qug)) : ctx.lineTo(toX(i), toY(d.cumulative_supply_qug)); });
      ctx.stroke();
      ctx.restore();

      // Pulsing live dot at last data point
      const lx = toX(n - 1);
      const ly = toY(days[n - 1].cumulative_supply_qug);
      const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 420);
      ctx.beginPath();
      ctx.arc(lx, ly, 7 + pulse * 4, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(245,158,11,${0.10 + 0.07 * pulse})`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#FBBF24';
      ctx.shadowColor = '#F59E0B';
      ctx.shadowBlur = 12;
      ctx.fill();
      ctx.shadowBlur = 0;

      animFrame.current = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(animFrame.current);
  }, [emData]);

  const pct = emData?.pctMined ?? 0;
  const era = emData?.era ?? 0;

  return (
    <div className="backdrop-blur-xl rounded-3xl overflow-hidden" style={{
      background: 'linear-gradient(135deg, rgba(15,15,25,0.92) 0%, rgba(28,20,8,0.92) 100%)',
      border: '2px solid rgba(245,158,11,0.22)',
      boxShadow: '0 0 32px rgba(245,158,11,0.06)',
    }}>
      <div className="p-4" style={{ borderBottom: '1px solid rgba(245,158,11,0.12)' }}>
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-base font-semibold text-amber-100 flex items-center gap-2">
              <span style={{ fontSize: 18 }}>📈</span>
              Emission Trajectory
            </h3>
            <p className="text-xs mt-0.5" style={{ color: 'rgba(245,158,11,0.45)' }}>
              Actual vs. scheduled emission · Era {era} · 90-day window
            </p>
          </div>
          <div className="text-right">
            <div className="text-xl font-bold text-amber-400 tabular-nums">{pct.toFixed(3)}%</div>
            <div className="text-[10px] font-mono" style={{ color: 'rgba(245,158,11,0.45)' }}>of 21M mined</div>
          </div>
        </div>
        <div className="mt-3 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(245,158,11,0.08)' }}>
          <motion.div className="h-full rounded-full"
            style={{ background: 'linear-gradient(90deg, #B45309, #F59E0B, #FCD34D)' }}
            initial={{ width: '0%' }} animate={{ width: `${Math.min(pct, 100)}%` }}
            transition={{ duration: 1.6, ease: 'easeOut' }} />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-[9px] font-mono" style={{ color: 'rgba(245,158,11,0.3)' }}>Genesis</span>
          <span className="text-[9px] font-mono" style={{ color: 'rgba(245,158,11,0.3)' }}>21,000,000 QUG</span>
        </div>
      </div>
      <div className="px-2 pt-2 pb-0">
        <canvas ref={canvasRef} style={{ width: '100%', height: '200px', display: 'block' }} />
      </div>
      <div className="px-4 py-3 flex items-center gap-5">
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 rounded" style={{ background: '#F59E0B', boxShadow: '0 0 4px #F59E0B' }} />
          <span className="text-[10px] font-mono" style={{ color: 'rgba(245,158,11,0.55)' }}>Actual</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0 rounded border-t border-dashed" style={{ borderColor: 'rgba(139,92,246,0.6)' }} />
          <span className="text-[10px] font-mono" style={{ color: 'rgba(139,92,246,0.55)' }}>Target</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full" style={{ background: '#FBBF24', boxShadow: '0 0 6px #F59E0B' }} />
          <span className="text-[10px] font-mono" style={{ color: 'rgba(245,158,11,0.45)' }}>Live</span>
        </div>
        <div className="ml-auto flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm opacity-30" style={{ background: 'rgba(139,92,246,0.3)' }} />
          <span className="text-[10px] font-mono" style={{ color: 'rgba(139,92,246,0.4)' }}>Deviation</span>
        </div>
      </div>
    </div>
  );
}

// ── Mining Decentralization — Lorenz Curve (v10.7.8) ─────────────────────────
function MiningDecentralizationViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrame = useRef<number>(0);
  const [vizData, setVizData] = useState<{
    gini: number;
    nakaCoeff: number;
    lorenz: Array<[number, number]>;
    totalMiners: number;
    totalHashTH: number;
  } | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      const resp = await qnkAPI.getNetworkMiners();
      if (!resp.success || resp.miners.length === 0) return;
      const sorted = resp.miners
        .map(m => m.hash_rate)
        .filter(h => h > 0)
        .sort((a, b) => a - b);
      const n = sorted.length;
      const totalHash = sorted.reduce((s, h) => s + h, 0);

      // Lorenz curve
      const lorenz: Array<[number, number]> = [[0, 0]];
      let cumH = 0;
      sorted.forEach((h, i) => { cumH += h; lorenz.push([(i + 1) / n, cumH / totalHash]); });

      // Gini coefficient
      let gSum = 0;
      sorted.forEach((h, i) => { gSum += (2 * (i + 1) - n - 1) * h; });
      const gini = n > 1 ? gSum / (n * totalHash) : 0;

      // Nakamoto coefficient (fewest miners controlling 51%)
      let cumulative = 0;
      let naka = 0;
      for (const h of [...sorted].sort((a, b) => b - a)) {
        cumulative += h; naka++;
        if (cumulative / totalHash >= 0.51) break;
      }

      setVizData({ gini, nakaCoeff: naka, lorenz, totalMiners: n, totalHashTH: totalHash / 1e12 });
    };
    fetchData();
    const iv = setInterval(fetchData, 30_000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !vizData) return;

    const draw = () => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.offsetWidth;
      const H = canvas.offsetHeight;
      if (W === 0 || H === 0) { animFrame.current = requestAnimationFrame(draw); return; }
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
        ctx.scale(dpr, dpr);
      }

      const P = { t: 20, r: 24, b: 38, l: 50 };
      const pw = W - P.l - P.r;
      const ph = H - P.t - P.b;
      ctx.clearRect(0, 0, W, H);

      const toX = (f: number) => P.l + f * pw;
      const toY = (f: number) => P.t + ph - f * ph;

      // Grid
      for (let g = 0; g <= 4; g++) {
        const f = g / 4;
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(P.l, toY(f)); ctx.lineTo(P.l + pw, toY(f)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(toX(f), P.t); ctx.lineTo(toX(f), P.t + ph); ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.font = '9px "JetBrains Mono",monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`${(f * 100).toFixed(0)}%`, P.l - 5, toY(f) + 3.5);
        if (g > 0 && g < 4) {
          ctx.textAlign = 'center';
          ctx.fillStyle = 'rgba(255,255,255,0.14)';
          ctx.fillText(`${(f * 100).toFixed(0)}%`, toX(f), P.t + ph + 18);
        }
      }
      // X-axis endpoints
      ctx.textAlign = 'center';
      ctx.font = '9px "JetBrains Mono",monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.2)';
      ctx.fillText('0%', toX(0), P.t + ph + 18);
      ctx.fillText('100%', toX(1), P.t + ph + 18);

      // Perfect equality diagonal
      ctx.save();
      ctx.setLineDash([4, 5]);
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(toX(0), toY(0)); ctx.lineTo(toX(1), toY(1)); ctx.stroke();
      ctx.restore();

      // Inequality zone (above Lorenz, below equality)
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(toX(0), toY(0));
      ctx.lineTo(toX(1), toY(1));
      for (let i = vizData.lorenz.length - 1; i >= 0; i--) {
        ctx.lineTo(toX(vizData.lorenz[i][0]), toY(vizData.lorenz[i][1]));
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(239,68,68,0.07)';
      ctx.fill();
      ctx.restore();

      // Lorenz curve fill
      ctx.save();
      const fillGrad = ctx.createLinearGradient(0, P.t, 0, P.t + ph);
      fillGrad.addColorStop(0, 'rgba(16,185,129,0.18)');
      fillGrad.addColorStop(1, 'rgba(16,185,129,0.02)');
      ctx.beginPath();
      vizData.lorenz.forEach(([x, y], i) => { i === 0 ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y)); });
      ctx.lineTo(toX(1), P.t + ph);
      ctx.lineTo(toX(0), P.t + ph);
      ctx.closePath();
      ctx.fillStyle = fillGrad;
      ctx.fill();
      ctx.restore();

      // Lorenz curve line
      ctx.save();
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 2.2;
      ctx.shadowColor = 'rgba(16,185,129,0.5)';
      ctx.shadowBlur = 7;
      ctx.beginPath();
      vizData.lorenz.forEach(([x, y], i) => { i === 0 ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y)); });
      ctx.stroke();
      ctx.restore();

      // Scatter dots for each miner
      if (vizData.lorenz.length <= 40) {
        ctx.save();
        ctx.shadowColor = 'rgba(16,185,129,0.4)';
        ctx.shadowBlur = 4;
        vizData.lorenz.slice(1).forEach(([x, y]) => {
          ctx.beginPath();
          ctx.arc(toX(x), toY(y), 2.8, 0, Math.PI * 2);
          ctx.fillStyle = '#10B981';
          ctx.fill();
        });
        ctx.restore();
      }

      // Nakamoto threshold line (pulsing red vertical)
      if (vizData.totalMiners > 0) {
        const nakaFrac = vizData.nakaCoeff / vizData.totalMiners;
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 700);
        const nx = toX(1 - nakaFrac); // largest miners are at right side (sorted asc, so right = largest)
        ctx.save();
        ctx.setLineDash([2, 4]);
        ctx.strokeStyle = `rgba(239,68,68,${0.35 + 0.2 * pulse})`;
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(nx, P.t); ctx.lineTo(nx, P.t + ph); ctx.stroke();
        ctx.restore();
        ctx.fillStyle = `rgba(239,68,68,${0.5 + 0.2 * pulse})`;
        ctx.font = '8px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('51%', nx, P.t + 10);
      }

      animFrame.current = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(animFrame.current);
  }, [vizData]);

  const giniPct = vizData ? (vizData.gini * 100).toFixed(1) : '—';
  const decentralized = vizData ? vizData.gini < 0.5 : null;

  return (
    <div className="backdrop-blur-xl rounded-3xl overflow-hidden" style={{
      background: 'linear-gradient(135deg, rgba(10,20,15,0.92) 0%, rgba(15,25,20,0.92) 100%)',
      border: '2px solid rgba(16,185,129,0.2)',
      boxShadow: '0 0 32px rgba(16,185,129,0.05)',
    }}>
      <div className="p-4" style={{ borderBottom: '1px solid rgba(16,185,129,0.1)' }}>
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-base font-semibold flex items-center gap-2" style={{ color: 'rgba(167,243,208,0.95)' }}>
              <span style={{ fontSize: 18 }}>⚖️</span>
              Mining Decentralization
            </h3>
            <p className="text-xs mt-0.5" style={{ color: 'rgba(16,185,129,0.4)' }}>
              Lorenz curve · hashrate distribution across {vizData?.totalMiners ?? '…'} active miners
            </p>
          </div>
          <div className="text-right">
            <div className="flex items-baseline gap-1 justify-end">
              <span className="text-xl font-bold tabular-nums" style={{ color: decentralized === null ? '#9CA3AF' : decentralized ? '#10B981' : '#F59E0B' }}>
                {giniPct}%
              </span>
            </div>
            <div className="text-[10px] font-mono" style={{ color: 'rgba(16,185,129,0.4)' }}>Gini coefficient</div>
          </div>
        </div>
        {vizData && (
          <div className="mt-3 grid grid-cols-3 gap-2">
            {[
              { label: 'Gini', value: (vizData.gini * 100).toFixed(1) + '%', sub: 'inequality index', color: vizData.gini < 0.5 ? '#10B981' : '#F59E0B' },
              { label: 'Nakamoto', value: vizData.nakaCoeff.toString(), sub: 'miners for 51%', color: vizData.nakaCoeff >= 3 ? '#10B981' : '#EF4444' },
              { label: 'Network', value: vizData.totalHashTH < 1 ? (vizData.totalHashTH * 1000).toFixed(1) + ' GH/s' : vizData.totalHashTH.toFixed(2) + ' TH/s', sub: 'total hashrate', color: '#6EE7B7' },
            ].map(stat => (
              <div key={stat.label} className="rounded-xl p-2 text-center" style={{ background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.1)' }}>
                <div className="text-sm font-bold tabular-nums" style={{ color: stat.color }}>{stat.value}</div>
                <div className="text-[9px] font-mono mt-0.5" style={{ color: 'rgba(16,185,129,0.4)' }}>{stat.label}</div>
                <div className="text-[8px]" style={{ color: 'rgba(255,255,255,0.2)' }}>{stat.sub}</div>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="px-2 pt-2 pb-0">
        <canvas ref={canvasRef} style={{ width: '100%', height: '200px', display: 'block' }} />
      </div>
      <div className="px-4 py-3 flex items-center gap-5">
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 rounded" style={{ background: '#10B981', boxShadow: '0 0 4px #10B981' }} />
          <span className="text-[10px] font-mono" style={{ color: 'rgba(16,185,129,0.5)' }}>Lorenz curve</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0 border-t border-dashed" style={{ borderColor: 'rgba(255,255,255,0.25)' }} />
          <span className="text-[10px] font-mono" style={{ color: 'rgba(255,255,255,0.3)' }}>Perfect equality</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0 border-t border-dashed" style={{ borderColor: 'rgba(239,68,68,0.5)' }} />
          <span className="text-[10px] font-mono" style={{ color: 'rgba(239,68,68,0.45)' }}>51% threshold</span>
        </div>
      </div>
    </div>
  );
}

const Dashboard = memo(function Dashboard({ onNavigateToSend, liveBalance, onNavigateToChat }: DashboardProps) {
  // 🌐 v3.4.3-browser: P2P real-time block streaming via gossipsub
  const { latestBlock: p2pLatestBlock, blockHistory: p2pBlockHistory, isSubscribed: p2pSubscribed } = useRealtimeBlocks();

  const [nodeStatus, setNodeStatus] = useState<NodeStatus | null>(null);
  const [recentTransactions, setRecentTransactions] = useState<Transaction[]>([]);
  // Skip loading screen if we have cached balance — dashboard renders immediately with cached data
  const [loading, setLoading] = useState(() => {
    const cached = localStorage.getItem('cachedBalance');
    const v = cached ? parseFloat(cached) : 0;
    return isNaN(v) || v <= 0; // show spinner only when no cache exists
  });
  const [error, setError] = useState<string | null>(null);
  const [walletAddress, setWalletAddress] = useState('');
  const [copiedAddress, setCopiedAddress] = useState(false);
  // v7.0.0: Faucet removed — all QUG earned through mining
  const [sseConnected, setSseConnected] = useState(false);
  const [showLoanModal, setShowLoanModal] = useState(false);
  const [showLoanApprovalModal, setShowLoanApprovalModal] = useState(false);
  const [approvedLoanDetails, setApprovedLoanDetails] = useState<any>(null);
  const [showLoanPaybackModal, setShowLoanPaybackModal] = useState(false);
  const [selectedLoanId, setSelectedLoanId] = useState<string | null>(null);
  const [transactionError, setTransactionError] = useState<string | null>(null);
  const [showFinanceModal, setShowFinanceModal] = useState(false);
  const [showBitcoinSwapModal, setShowBitcoinSwapModal] = useState(false);
  const [showZcashWalletModal, setShowZcashWalletModal] = useState(false);
  const [showIronFishWalletModal, setShowIronFishWalletModal] = useState(false);
  const [showEthereumSwapModal, setShowEthereumSwapModal] = useState(false);
  const [activeDashboardTab, setActiveDashboardTab] = useState<'wallet' | 'mail' | 'calendar' | 'chat' | 'search'>('wallet');
  const [unreadEmailCount, setUnreadEmailCount] = useState(0);
  const [chatUnreadCount, setChatUnreadCount] = useState(0);
  const [tabOrder, setTabOrder] = useState<Array<'wallet' | 'mail' | 'calendar' | 'chat' | 'search'>>(() => {
    try {
      const saved = localStorage.getItem('dashboardTabOrder');
      if (saved) {
        const parsed = JSON.parse(saved);
        // Ensure 'search' tab exists in saved order (migration)
        if (!parsed.includes('search')) parsed.push('search');
        return parsed;
      }
    } catch {}
    return ['wallet', 'search', 'mail', 'calendar', 'chat'];
  });
  const [showTabSettings, setShowTabSettings] = useState(false);
  const [btcBalance, setBtcBalance] = useState(0);
  const [zecBalance, setZecBalance] = useState(0);
  const [ethBalance, setEthBalance] = useState(0);

  // Multi-wallet state - 🚨 v2.3.7-beta: Initialize from cache to prevent zero balance on refresh
  const [walletBalances, setWalletBalances] = useState<WalletBalance[]>(() => {
    // v6.5.0: Phase-aware cache clearing — purge stale balances on network phase change
    try {
      const lastPhase = localStorage.getItem('lastNetworkPhase');
      // v1.0.2: Derive phase from server version to auto-clear on upgrades
      const serverVersion = localStorage.getItem('serverVersion') || '';
      const currentPhase = `mainnet-v${serverVersion || '8.6.4'}`;
      if (lastPhase && lastPhase !== currentPhase) {
        console.log(`🔄 Phase transition detected: ${lastPhase} → ${currentPhase}. Clearing balance caches.`);
        localStorage.removeItem('cachedBalance');
        localStorage.removeItem('cachedQugusdBalance');
        localStorage.removeItem('walletBalanceHistory');
        localStorage.removeItem('highestKnownBalances');
        localStorage.setItem('lastNetworkPhase', currentPhase);
      } else if (!lastPhase) {
        localStorage.setItem('lastNetworkPhase', currentPhase);
      }
    } catch (e) {
      console.warn('Failed to check phase cache:', e);
    }

    // Try to load cached balances immediately to avoid showing 0 on refresh
    const cachedQugBalance = localStorage.getItem('cachedBalance');
    const cachedQugValue = cachedQugBalance ? parseFloat(cachedQugBalance) : 0;
    const validQugBalance = !isNaN(cachedQugValue) && isFinite(cachedQugValue) ? cachedQugValue : 0;

    // Also load QUGUSD cached balance
    const cachedQugusdBalance = localStorage.getItem('cachedQugusdBalance');
    const cachedQugusdValue = cachedQugusdBalance ? parseFloat(cachedQugusdBalance) : 0;
    const validQugusdBalance = !isNaN(cachedQugusdValue) && isFinite(cachedQugusdValue) ? cachedQugusdValue : 0;

    // Also load balance history for the graph
    let qugHistory: { timestamp: number; balance: number }[] = [];
    let qugusdHistory: { timestamp: number; balance: number }[] = [];
    try {
      const storedHistory = localStorage.getItem('qnk_balance_long_v1') || localStorage.getItem('walletBalanceHistory');
      if (storedHistory) {
        const parsed = JSON.parse(storedHistory);
        qugHistory = parsed['QUG'] || [];
        qugusdHistory = parsed['QUGUSD'] || [];
      }
    } catch (e) {
      console.warn('Failed to load balance history from cache:', e);
    }

    const initialBalances: WalletBalance[] = [];

    // Add QUG if we have a cached balance
    if (validQugBalance > 0) {
      console.log('🚀 [INIT] Initializing with cached QUG balance:', validQugBalance);
      initialBalances.push({
        symbol: 'QUG',
        name: 'Quillon Gold',
        balance: validQugBalance,
        icon: 'qug' as const,
        color: 'from-amber-400 to-yellow-600',
        history: qugHistory.length >= 2 ? qugHistory : undefined,
      });
    }

    // Add QUGUSD if we have a cached balance
    if (validQugusdBalance > 0) {
      console.log('🚀 [INIT] Initializing with cached QUGUSD balance:', validQugusdBalance);
      initialBalances.push({
        symbol: 'QUGUSD',
        name: 'Quillon USD',
        balance: validQugusdBalance,
        usdValue: validQugusdBalance, // 1:1 peg to USD
        icon: 'usd' as const,
        color: 'from-blue-400 to-cyan-500',
        history: qugusdHistory.length >= 2 ? qugusdHistory : undefined,
      });
    }

    return initialBalances;
  });
  const [usdBalance, setUsdBalance] = useState<number>(0);

  // Animation state for balance updates
  const [balanceAnimations, setBalanceAnimations] = useState<Record<string, boolean>>({});

  // CRITICAL FIX: Track highest known balance per token to prevent showing stale/lower values
  // This prevents the bug where balance jumps from 66 to 0.71 on refresh
  // 🚨 v2.3.7-beta: Initialize from cache immediately (IIFE pattern since useRef doesn't accept functions)
  const highestKnownBalancesRef = useRef<Record<string, number>>((() => {
    const result: Record<string, number> = {};

    // Load QUG from cache
    const cachedQugBalance = localStorage.getItem('cachedBalance');
    const cachedQugValue = cachedQugBalance ? parseFloat(cachedQugBalance) : 0;
    if (!isNaN(cachedQugValue) && isFinite(cachedQugValue) && cachedQugValue > 0) {
      result['QUG'] = cachedQugValue;
    }

    // Load QUGUSD from cache
    const cachedQugusdBalance = localStorage.getItem('cachedQugusdBalance');
    const cachedQugusdValue = cachedQugusdBalance ? parseFloat(cachedQugusdBalance) : 0;
    if (!isNaN(cachedQugusdValue) && isFinite(cachedQugusdValue) && cachedQugusdValue > 0) {
      result['QUGUSD'] = cachedQugusdValue;
    }

    return result;
  })());

  // Initialize highestKnownBalancesRef from localStorage on mount
  // 🚨 v2.3.7-beta: Dispatch cached balance to App.tsx IMMEDIATELY on mount
  // This ensures TopBar shows correct balance before API call completes
  useEffect(() => {
    const cachedBalance = localStorage.getItem('cachedBalance');
    if (cachedBalance) {
      const value = parseFloat(cachedBalance);
      if (!isNaN(value) && isFinite(value) && value > 0) {
        // Update local tracking
        if (value > (highestKnownBalancesRef.current['QUG'] || 0)) {
          highestKnownBalancesRef.current['QUG'] = value;
        }
        // v8.1.6: Removed — App.tsx reads cachedBalance from localStorage directly on mount.
        // Dispatching balance-update here caused zigzag by competing with App.tsx SSE updates.
        console.log('ℹ️ [Dashboard] Cached balance on mount:', value, '(App.tsx reads it directly)');
      }
    }
  }, []); // Run once on mount

  // Balance history tracking — v10.3.15: keep up to 10080 points (7 days) in new key
  // Old key 'walletBalanceHistory' kept for migration; new key stores full long-term history.
  const [_balanceHistory, setBalanceHistory] = useState<Record<string, BalanceHistoryPoint[]>>(() => {
    try {
      // Try new long-term key first, fall back to legacy 20-point key
      const longTerm = localStorage.getItem('qnk_balance_long_v1');
      if (longTerm) return JSON.parse(longTerm);
      const saved = localStorage.getItem('qnk_balance_long_v1') || localStorage.getItem('walletBalanceHistory');
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });

  // Transaction details modal state
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // QR code modal state
  const [isQRModalOpen, setIsQRModalOpen] = useState(false);

  // Node info modal state
  const [isNodeInfoModalOpen, setIsNodeInfoModalOpen] = useState(false);

  // AI Report modal state
  const [isAIReportModalOpen, setIsAIReportModalOpen] = useState(false);
  const [aiReportLoading, setAiReportLoading] = useState(false);
  const [aiReport, setAiReport] = useState<string>('');

  // USD wallet modal states
  const [isAddUSDModalOpen, setIsAddUSDModalOpen] = useState(false);
  const [isSendUSDModalOpen, setIsSendUSDModalOpen] = useState(false);
  const [showStripeCheckout, setShowStripeCheckout] = useState(false);
  const [usdAmount, setUsdAmount] = useState('');
  const [usdRecipient, setUsdRecipient] = useState('');
  const [stripeLoading, setStripeLoading] = useState(false);
  const [stripeError, setStripeError] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Enhanced filtering and pagination state
  const [filterType, setFilterType] = useState<'all' | 'receive' | 'send' | 'mining' | 'swap'>('all');
  const [sortBy, setSortBy] = useState<'date' | 'amount'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  // Phase transition modal state
  const [showPhaseModal, setShowPhaseModal] = useState(false); // Disabled - phase transition modal no longer needed
  const [showStakingModal, setShowStakingModal] = useState(false);
  const [showMobileSetup, setShowMobileSetup] = useState(false);
  const [newsCollapsed, setNewsCollapsed] = useState(true);
  const [selectedArticle, setSelectedArticle] = useState<null | {
    tag: string; tagColor: string; tagBg: string; tagBorder: string;
    icon: React.ReactNode; title: string; excerpt: string; date: string;
    accent: string; border: string; fullContent: string;
  }>(null);

  // v10.3.0: Show mobile setup QR modal once (2s after load)
  useEffect(() => {
    if (localStorage.getItem(MOBILE_SETUP_STORAGE_KEY)) return;
    const timer = setTimeout(() => setShowMobileSetup(true), 2000);
    return () => clearTimeout(timer);
  }, []);

  // v8.5.5: Fetch unread email count on mount + listen for events
  useEffect(() => {
    const fetchUnread = async () => {
      try {
        const res = await qnkAPI.getEmailUnreadCount();
        if (res?.data?.count !== undefined) setUnreadEmailCount(res.data.count);
      } catch {}
    };
    fetchUnread();
    const interval = setInterval(fetchUnread, 30000); // refresh every 30s

    const handleEmailReceived = () => { fetchUnread(); };
    const handleUnreadCount = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.count !== undefined) setUnreadEmailCount(detail.count);
    };
    const handleEmailRead = () => { fetchUnread(); };

    const handleChatMessage = () => { setChatUnreadCount((n) => n + 1); };
    window.addEventListener('email-received', handleEmailReceived);
    window.addEventListener('email-unread-count', handleUnreadCount);
    window.addEventListener('email-read', handleEmailRead);
    window.addEventListener('qnk-new-chat-message', handleChatMessage);
    return () => {
      clearInterval(interval);
      window.removeEventListener('email-received', handleEmailReceived);
      window.removeEventListener('email-unread-count', handleUnreadCount);
      window.removeEventListener('email-read', handleEmailRead);
      window.removeEventListener('qnk-new-chat-message', handleChatMessage);
    };
  }, []);

  // v8.5.5: Re-fetch unread count whenever user switches to/from mail tab
  useEffect(() => {
    const fetchUnread = async () => {
      try {
        const res = await qnkAPI.getEmailUnreadCount();
        if (res?.data?.count !== undefined) setUnreadEmailCount(res.data.count);
      } catch {}
    };
    // Small delay to let EmailScreen's mark-read calls settle
    const timer = setTimeout(fetchUnread, 500);
    return () => clearTimeout(timer);
  }, [activeDashboardTab]);

  // v7.3.4: Persist tab order changes
  const handleTabOrderChange = useCallback((newOrder: Array<'wallet' | 'mail' | 'calendar' | 'chat'>) => {
    setTabOrder(newOrder);
    localStorage.setItem('dashboardTabOrder', JSON.stringify(newOrder));
  }, []);

  const moveTab = useCallback((tabId: string, direction: 'up' | 'down') => {
    setTabOrder(prev => {
      const idx = prev.indexOf(tabId as any);
      if (idx < 0) return prev;
      const newIdx = direction === 'up' ? Math.max(0, idx - 1) : Math.min(prev.length - 1, idx + 1);
      if (newIdx === idx) return prev;
      const newOrder = [...prev];
      [newOrder[idx], newOrder[newIdx]] = [newOrder[newIdx], newOrder[idx]];
      localStorage.setItem('dashboardTabOrder', JSON.stringify(newOrder));
      return newOrder;
    });
  }, []);

  // Generate AI Report — uses Ollama/gemma4 via /api/v1/ai/chat
  const generateAIReport = async () => {
    setAiReportLoading(true);
    setIsAIReportModalOpen(true);
    setAiReport('');

    const networkStats = nodeStatus ? {
      tpsCurrent: nodeStatus.tps_current || 0,
      tpsAverage: nodeStatus.tps_average || 0,
      connectedPeers: nodeStatus.connected_peers || 0,
      isValidator: nodeStatus.is_validator,
      currentHeight: nodeStatus.current_height || 0,
    } : null;

    const walletBalance = nodeStatus?.balance || 0;

    const context = `Wallet: ${walletAddress}
Balance: ${walletBalance.toFixed(4)} QUG
${networkStats
  ? `TPS: ${networkStats.tpsCurrent} current / ${networkStats.tpsAverage} avg
Peers: ${networkStats.connectedPeers}
Height: ${networkStats.currentHeight}
Validator: ${networkStats.isValidator ? 'Active' : 'No'}`
  : 'Network: offline'}
Transactions (recent): ${recentTransactions.slice(0, 10).length}`;

    const query = 'Analyze this wallet and give a short report: balance health, network status, and top 2 recommendations to earn more rewards. Be concise.';

    try {
      const res = await fetch('/api/v1/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, context, stream: true }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        setAiReport(`Error: ${err.error || `Server returned ${res.status}`}`);
        setAiReportLoading(false);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        setAiReport('Error: no response stream.');
        setAiReportLoading(false);
        return;
      }

      const decoder = new TextDecoder();
      let buf = '';
      let full = '';
      let currentEvent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';

        for (const line of lines) {
          const t = line.trim();
          if (!t || t.startsWith(':')) continue;
          if (t.startsWith('event: ')) { currentEvent = t.slice(7); continue; }
          if (t.startsWith('data: ')) {
            const evType = currentEvent || 'token';
            currentEvent = '';
            try {
              const parsed = JSON.parse(t.slice(6));
              if (evType === 'token' && parsed.content) {
                full += parsed.content;
                setAiReport(full);
              } else if (evType === 'done') {
                setAiReportLoading(false);
              } else if (evType === 'error') {
                setAiReport(`Error: ${parsed.message || 'AI error'}`);
                setAiReportLoading(false);
              }
            } catch { /* skip unparseable */ }
          }
        }
      }
      setAiReportLoading(false);
    } catch (error) {
      console.error('AI report error:', error);
      setAiReport('Failed to generate AI report. Please try again.');
      setAiReportLoading(false);
    }
  };

  // Detect balance changes and trigger animations
  // Use ref to track previous balances to avoid re-render loops
  const previousBalancesRef = useRef<Record<string, number>>({});
  const animationTimeoutsRef = useRef<Record<string, NodeJS.Timeout>>({});

  useEffect(() => {
    const newAnimations: Record<string, boolean> = {};

    walletBalances.forEach(wallet => {
      const key = wallet.symbol;
      const prevBalance = previousBalancesRef.current[key];
      const currentBalance = wallet.balance;

      // Update ref with current balance
      previousBalancesRef.current[key] = currentBalance;

      // CRITICAL FIX: Only trigger animation if balance changed by meaningful amount (> 0.0001)
      // This prevents flickering from floating-point rounding or micro-variations
      const balanceDiff = Math.abs((prevBalance ?? currentBalance) - currentBalance);
      const isSignificantChange = prevBalance !== undefined && balanceDiff > 0.0001;

      if (isSignificantChange) {
        newAnimations[key] = true;
        console.log(`🎨 Balance animation triggered for ${key}: ${prevBalance?.toFixed(4)} → ${currentBalance.toFixed(4)} (diff: ${balanceDiff.toFixed(6)})`);

        // Clear any existing timeout for this wallet
        if (animationTimeoutsRef.current[key]) {
          clearTimeout(animationTimeoutsRef.current[key]);
        }

        // Auto-disable animation after 3 seconds
        animationTimeoutsRef.current[key] = setTimeout(() => {
          setBalanceAnimations(prev => ({ ...prev, [key]: false }));
        }, 3000);
      } else {
        // Only set to false if no animation was just triggered
        newAnimations[key] = balanceAnimations[key] || false;
      }
    });

    // Only update state if animations actually changed
    setBalanceAnimations(prev => {
      const hasChanges = Object.keys(newAnimations).some(key => prev[key] !== newAnimations[key]);
      return hasChanges ? newAnimations : prev;
    });

    // Cleanup function
    return () => {
      Object.values(animationTimeoutsRef.current).forEach(timeout => clearTimeout(timeout));
    };
  }, [walletBalances]);

  // Fetch real data from Q-NarwhalKnight API
  useEffect(() => {
    let mounted = true;
    // let eventSource: EventSource | null = null; // Disabled - App.tsx handles SSE

    // Create debounced versions of fetch functions to prevent request storms
    // These will delay execution by 1.5s after the last call
    const fetchNodeStatusCore = async () => {
      console.log('Fetching node status...');
      if (!mounted) return;

      // v2.3.31-beta: Check BOTH local ref AND global localStorage cooldown
      const nodeGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      const nodeGlobalCooldownActive = Date.now() < nodeGlobalCooldownUntil;
      if (dexSwapCooldownRef.current || nodeGlobalCooldownActive) {
        console.log('🚫 [fetchNodeStatusCore] SKIPPING during DEX cooldown (global:', nodeGlobalCooldownActive, ')');
        return;
      }

      try {
        const response = await qnkAPI.getNodeStatus();
        console.log('Node status response:', response);
        if (!mounted) return;

        if (response.success && response.data) {
          const currentWalletAddress = localStorage.getItem('walletAddress');
          let walletBalance = 0;

          if (currentWalletAddress) {
            const previousHighest = highestKnownBalancesRef.current['QUG'] || 0;
            const cachedBalance = localStorage.getItem('cachedBalance');
            const cachedValue = cachedBalance ? parseFloat(cachedBalance) : 0;

            try {
              const balanceResponse = await qnkAPI.getWalletBalance(currentWalletAddress);
              if (!mounted) return;

              if (balanceResponse.success && balanceResponse.data) {
                const fetchedBalance = balanceResponse.data.balance_qnk || 0;
                console.log('✅ Balance fetched:', fetchedBalance, '(highest known:', previousHighest, ', cached:', cachedValue, ')');

                // v1.0.2: Accept API balance as authoritative — only reject near-zero from large values
                const referenceBalance = Math.max(previousHighest, cachedValue);

                if (fetchedBalance > 0 || referenceBalance === 0 || fetchedBalance >= referenceBalance * 0.01) {
                  walletBalance = fetchedBalance;
                  // Update tracking with latest value (not just highest)
                  highestKnownBalancesRef.current['QUG'] = fetchedBalance;
                  // v2.3.31-beta: Check global cooldown for localStorage write
                  const lsGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
                  const lsGlobalCooldownActive = Date.now() < lsGlobalCooldownUntil;
                  if (!dexSwapCooldownRef.current && !lsGlobalCooldownActive) {
                    safeCacheBalance(fetchedBalance);
                  } else {
                    console.log('🚫 [fetchNodeStatusCore] SKIPPING localStorage write during DEX cooldown (global:', lsGlobalCooldownActive, ')');
                  }
                } else {
                  // Near-zero from a large balance — likely stale/corrupt data
                  console.warn(`⚠️ Rejecting near-zero balance: ${fetchedBalance} (reference: ${referenceBalance})`);
                  walletBalance = referenceBalance;
                }
              } else {
                // Authentication failed - use highest known or cached balance
                console.warn('⚠️ Balance query failed:', balanceResponse.error);
                walletBalance = Math.max(previousHighest, cachedValue);
                console.log('💰 Using best known balance:', walletBalance);
              }
            } catch (balanceErr) {
              console.warn('❌ Failed to fetch wallet balance:', balanceErr);
              // Fallback: use highest known or cached balance
              walletBalance = Math.max(previousHighest, cachedValue);
              console.log('💰 Using best known balance (error fallback):', walletBalance);
            }
          }

          if (mounted) {
            setNodeStatus({
              ...response.data,
              balance: walletBalance
            });
            setError(null);
          }
        } else {
          throw new Error(response.error || 'Failed to fetch node status');
        }
      } catch (err) {
        console.error('Error fetching node status:', err);
        if (mounted) {
          // Even if node status fails, try to load best known balance
          const cachedBalance = localStorage.getItem('cachedBalance');
          const cachedValue = cachedBalance ? parseFloat(cachedBalance) : 0;
          const previousHighest = highestKnownBalancesRef.current['QUG'] || 0;
          const bestBalance = Math.max(previousHighest, cachedValue);

          if (bestBalance > 0) {
            console.log('💰 Using best known balance after node status error:', bestBalance);
            setNodeStatus(prev => ({
              ...(prev || {} as NodeStatus),
              balance: bestBalance,
              network_health: 'unknown',
              consensus_status: 'unknown',
              // Keep previous height/stats if available — don't reset to 0
              current_height: prev?.current_height || 0,
              tps_current: prev?.tps_current || 0,
              tps_average: prev?.tps_average || 0,
              uptime_formatted: prev?.uptime_formatted || '0h 0m 0s',
            } as NodeStatus));
          } else {
            setError('Failed to connect to Q-NarwhalKnight node');
          }
        }
      }
    };

    const fetchWalletBalances = async () => {
      console.log('💰 Fetching wallet balances...');
      const currentWalletAddress = localStorage.getItem('walletAddress');

      if (!currentWalletAddress) {
        console.warn('⚠️ No wallet address found');
        return;
      }

      // Fetch fresh QUG balance from API (includes mining rewards)
      // 🚨 v2.3.7-beta CRITICAL FIX: Always have a fallback to localStorage cache
      // This prevents showing 0 balance on refresh when API fails
      const cachedBalanceStr = localStorage.getItem('cachedBalance');
      const cachedBalanceValue = cachedBalanceStr ? parseFloat(cachedBalanceStr) : 0;
      const validCachedBalance = !isNaN(cachedBalanceValue) && isFinite(cachedBalanceValue) ? cachedBalanceValue : 0;

      let qugBalance = validCachedBalance; // Start with cached value, not 0
      const previousHighest = highestKnownBalancesRef.current['QUG'] || validCachedBalance;

      console.log('🔍 [fetchWalletBalances] Starting with:', {
        cachedBalance: validCachedBalance,
        previousHighest: previousHighest,
        refValue: highestKnownBalancesRef.current['QUG']
      });

      // Fetch QUG balance + USD price in parallel for faster display
      const [balanceResponse, priceResp] = await Promise.all([
        qnkAPI.getWalletBalance(currentWalletAddress),
        qnkAPI.getAMMPrice('QUG').catch(() => null),
      ]);

      let qugPriceUsd: number | undefined;
      if (priceResp && priceResp.success && priceResp.data && priceResp.data.price_usd > 0) {
        qugPriceUsd = priceResp.data.price_usd;
        qugPriceUsdRef.current = qugPriceUsd;
      }

      if (balanceResponse.success && balanceResponse.data) {
          const fetchedBalance = balanceResponse.data.balance_qnk || 0;
          console.log('💰 Fresh QUG balance fetched:', fetchedBalance, '(previous highest:', previousHighest, ', cached:', validCachedBalance, ')');

          // v1.0.2: Accept API balance as authoritative — only reject near-zero from large values
          const referenceBalance = Math.max(previousHighest, validCachedBalance);
          if (fetchedBalance > 0 || referenceBalance === 0 || fetchedBalance >= referenceBalance * 0.01) {
            qugBalance = fetchedBalance;
            highestKnownBalancesRef.current['QUG'] = fetchedBalance;
            const wbGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
            const wbGlobalCooldownActive = Date.now() < wbGlobalCooldownUntil;
            if (!dexSwapCooldownRef.current && !wbGlobalCooldownActive) {
              safeCacheBalance(fetchedBalance);
            }
          } else {
            console.warn(`⚠️ Rejecting near-zero balance: ${fetchedBalance} (reference: ${referenceBalance})`);
            qugBalance = referenceBalance;
          }
        } else {
          qugBalance = Math.max(previousHighest, validCachedBalance, nodeStatus?.balance || 0);
          console.warn('⚠️ Balance query failed, using best known:', qugBalance);
        }

      // 🚨 NEVER allow 0 balance if we have cached value
      if (qugBalance === 0 && validCachedBalance > 0) {
        qugBalance = validCachedBalance;
      }

      const now = Date.now();

      // Load saved history from localStorage
      let savedHistory: Record<string, BalanceHistoryPoint[]> = {};
      try {
        const saved = localStorage.getItem('qnk_balance_long_v1') || localStorage.getItem('walletBalanceHistory');
        savedHistory = saved ? JSON.parse(saved) : {};
      } catch {
        savedHistory = {};
      }

      // Merge saved history with new data point (deduplicate tiny changes)
      const qugSavedHistory = savedHistory['QUG'] || [];
      const qugLastPoint = qugSavedHistory[qugSavedHistory.length - 1];
      let qugHistory: BalanceHistoryPoint[];
      const qugShouldAdd = !qugLastPoint
        || (qugLastPoint.balance > 0 && Math.abs(qugBalance - qugLastPoint.balance) / qugLastPoint.balance > 0.005)
        || (qugLastPoint.balance === 0 && qugBalance > 0)
        || (now - qugLastPoint.timestamp > 10000 && qugBalance !== qugLastPoint.balance);
      if (qugShouldAdd) {
        qugHistory = [...qugSavedHistory, { timestamp: now, balance: qugBalance }].slice(-10080);
      } else {
        qugHistory = qugSavedHistory.length >= 2 ? qugSavedHistory : [...qugSavedHistory, { timestamp: now, balance: qugBalance }].slice(-10080);
      }

      if (qugHistory.length < 2) {
        qugHistory = [
          { timestamp: now - 60000, balance: qugBalance },
          { timestamp: now, balance: qugBalance }
        ];
      }

      const balances: WalletBalance[] = [
        {
          symbol: 'QUG',
          name: 'Quillon Graph',
          balance: qugBalance,
          usdValue: qugPriceUsd !== undefined ? qugBalance * qugPriceUsd : undefined,
          icon: 'qug',
          color: 'from-amber-400 to-yellow-500',
          history: qugHistory
        }
      ];

      console.log('📊 [fetchWalletBalances] Initialized QUG with history:', qugHistory.length, 'points (', qugSavedHistory.length, 'from localStorage)');

      // 🚨 v2.3.7-beta: Fetch QUGUSD balance with cache fallback (same pattern as QUG)
      const cachedQugusdStr = localStorage.getItem('cachedQugusdBalance');
      const cachedQugusdValue = cachedQugusdStr ? parseFloat(cachedQugusdStr) : 0;
      const validCachedQugusd = !isNaN(cachedQugusdValue) && isFinite(cachedQugusdValue) ? cachedQugusdValue : 0;

      // v6.5.1: Trust backend for QUGUSD - no anti-zero or anti-drop overrides
      let qugUsdBalance = 0;

      try {
        const response = await qnkAPI.getMultiTokenBalance();
        console.log('🔍 [Dashboard] Multi-token balance response:', JSON.stringify(response, null, 2));
        if (response.success && response.data && response.data.tokens) {
          const tokensObj = response.data.tokens;

          if (tokensObj.qugusd && tokensObj.qugusd.balance !== undefined) {
            qugUsdBalance = parseFloat(tokensObj.qugusd.balance) || 0;
          } else if (tokensObj.QUGUSD && tokensObj.QUGUSD.balance !== undefined) {
            qugUsdBalance = parseFloat(tokensObj.QUGUSD.balance) || 0;
          }
          console.log('💵 [Dashboard] QUGUSD balance fetched:', qugUsdBalance);

          if (qugUsdBalance > 0) {
            localStorage.setItem('cachedQugusdBalance', qugUsdBalance.toString());
            // v10.2.9: Also write a backup key that is NEVER cleared
            // TransactionScreen reads this when cachedQugusdBalance is removed
            localStorage.setItem('lastKnownQugusdBalance', qugUsdBalance.toString());
            if (qugUsdBalance > (highestKnownBalancesRef.current['QUGUSD'] || 0)) {
              highestKnownBalancesRef.current['QUGUSD'] = qugUsdBalance;
            }
          } else {
            // v10.2.9: Don't clear cache when API returns 0 — backend token_balances
            // may not be loaded yet after restart. Keep lastKnownQugusdBalance as backup.
            // Only clear cachedQugusdBalance (TransactionScreen will fall back to lastKnown)
            localStorage.removeItem('cachedQugusdBalance');
            highestKnownBalancesRef.current['QUGUSD'] = 0;
          }
        }
      } catch (error) {
        // On fetch failure only, fall back to cache
        const cachedQugusd = localStorage.getItem('cachedQugusdBalance');
        qugUsdBalance = cachedQugusd ? parseFloat(cachedQugusd) || 0 : 0;
        console.warn('⚠️ Failed to fetch QUGUSD balance, using cached:', qugUsdBalance, error);
      }

      // v6.5.1: Allow QUGUSD to be 0 if backend genuinely returns 0
      // The anti-zero logic was preventing balance resets on network/phase transitions
      if (qugUsdBalance === 0 && validCachedQugusd > 0) {
        console.log('ℹ️ [fetchWalletBalances] QUGUSD is 0 (backend confirmed). Clearing stale cache.');
        localStorage.removeItem('cachedQugusdBalance');
      }

      // Add QUGUSD to balances
      const qugusdSavedHistory = savedHistory['QUGUSD'] || [];
      const qugusdHistory: BalanceHistoryPoint[] = [
        ...qugusdSavedHistory,
        { timestamp: now, balance: qugUsdBalance }
      ].slice(-10080);

      balances.push({
        symbol: 'QUGUSD',
        name: 'Quillon USD',
        balance: qugUsdBalance,
        usdValue: qugUsdBalance, // 1:1 peg to USD
        icon: 'usd',
        color: 'from-blue-400 to-cyan-500',
        history: qugusdHistory
      });

      console.log('📊 [fetchWalletBalances] Initialized QUGUSD with history:', qugusdHistory.length, 'points (', qugusdSavedHistory.length, 'from localStorage), balance:', qugUsdBalance);

      // v8.5.9: Fetch QUSD balance from multi-token API response
      let qusdBalance = 0;
      try {
        const response2 = await qnkAPI.getMultiTokenBalance();
        if (response2.success && response2.data && response2.data.tokens) {
          const tokensObj2 = response2.data.tokens;
          if (tokensObj2.QUSD && tokensObj2.QUSD.balance !== undefined) {
            qusdBalance = parseFloat(tokensObj2.QUSD.balance) || 0;
          } else if (tokensObj2.qusd && tokensObj2.qusd.balance !== undefined) {
            qusdBalance = parseFloat(tokensObj2.qusd.balance) || 0;
          }
        }
      } catch (error) {
        console.warn('⚠️ Failed to fetch QUSD balance:', error);
      }

      if (qusdBalance > 0) {
        const qusdSavedHistory = savedHistory['QUSD'] || [];
        const qusdHistory: BalanceHistoryPoint[] = [
          ...qusdSavedHistory,
          { timestamp: now, balance: qusdBalance }
        ].slice(-10080);

        balances.push({
          symbol: 'QUSD',
          name: 'Quillon USD',
          balance: qusdBalance,
          usdValue: qusdBalance, // 1:1 peg to USD
          icon: 'usd',
          color: 'from-green-400 to-emerald-500',
          history: qusdHistory
        });
        console.log('💵 [Dashboard] QUSD balance:', qusdBalance);
      }

      // Fetch USD balance from payment API - ALWAYS show USD wallet
      let usdValue = 0;
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/balance`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ wallet_address: currentWalletAddress }),
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data) {
            usdValue = parseFloat(data.data.balance_usd || '0');
            setUsdBalance(usdValue);
            console.log('💵 USD balance fetched:', usdValue);
          }
        } else {
          console.warn('⚠️ USD balance fetch failed, showing 0 balance');
        }
      } catch (error) {
        // Silently fail if payment API is not available
        console.warn('⚠️ Payment API not available - USD wallet features disabled');
      }

      // Always add USD wallet (even with 0 balance) so user can see Add/Send controls
      const usdSavedHistory = savedHistory['USD'] || [];
      const usdHistory: BalanceHistoryPoint[] = [
        ...usdSavedHistory,
        { timestamp: now, balance: usdValue }
      ].slice(-10080);

      balances.push({
        symbol: 'USD',
        name: 'US Dollar',
        balance: usdValue,
        icon: 'usd',
        color: 'from-green-400 to-emerald-500',
        history: usdHistory
      });

      console.log('📊 [fetchWalletBalances] Initialized USD with history:', usdHistory.length, 'points (', usdSavedHistory.length, 'from localStorage)');

      // Note: Custom tokens would be fetched here if the API supported them
      // Currently, only QUG and QUGUSD are supported in the multi-token balance endpoint

      // Bridge wallets (with empty history to prevent "Loading..." display)
      balances.push(
        {
          symbol: 'ZEC',
          name: 'Zcash (Shielded)',
          balance: zecBalance,
          icon: 'zec',
          color: 'from-purple-400 to-indigo-600',
          shieldedOnly: true,
          history: [],
        },
        {
          symbol: 'IRON',
          name: 'Iron Fish',
          balance: 0,
          icon: 'iron',
          color: 'from-cyan-400 to-slate-500',
          shieldedOnly: true,
          history: [],
        },
        {
          symbol: 'BTC',
          name: 'Bitcoin',
          balance: btcBalance,
          icon: 'btc',
          color: 'from-orange-400 to-amber-500',
          history: [],
        },
        {
          symbol: 'ETH',
          name: 'Ethereum',
          balance: ethBalance,
          icon: 'eth',
          color: 'from-blue-400 to-indigo-500',
          history: [],
        },
      );

      // v2.3.31-beta: Check BOTH local ref AND global localStorage cooldown
      const globalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      const globalCooldownActive = Date.now() < globalCooldownUntil;
      if (dexSwapCooldownRef.current || globalCooldownActive) {
        console.log('🚫 [fetchWalletBalances] SKIPPING setWalletBalances during DEX cooldown (global:', globalCooldownActive, ')');
        return; // Don't overwrite the correct DEX-updated balance with stale API data
      }

      // v8.6.2: Merge instead of replace — preserve higher QUG balance from SSE
      // to prevent flicker when API response races with SSE updates.
      setWalletBalances(prevWallets => {
        return balances.map(newWallet => {
          if (newWallet.symbol === 'QUG') {
            const prevQug = prevWallets.find(w => w.symbol === 'QUG');
            if (prevQug && prevQug.balance > newWallet.balance) {
              // SSE already gave us a higher balance — keep it, just update history
              return { ...newWallet, balance: prevQug.balance, history: newWallet.history || prevQug.history };
            }
          }
          return newWallet;
        });
      });
    };

    const fetchRecentTransactionsCore = async () => {
      console.log('📋 [fetchRecentTransactions] START - Fetching decentralized wallet history...');
      console.log('📋 [fetchRecentTransactions] Mounted status:', mounted);
      const currentWalletAddress = localStorage.getItem('walletAddress') || '';
      console.log('📋 [fetchRecentTransactions] Current wallet:', currentWalletAddress);

      if (!mounted) {
        console.log('📋 [fetchRecentTransactions] ABORT - Component not mounted');
        return;
      }

      if (!currentWalletAddress) {
        console.log('📋 [fetchRecentTransactions] ABORT - No wallet address');
        return;
      }

      try {
        // v3.5.8-beta: Use new decentralized wallet history API
        // This fetches transactions verified by all nodes (transfers + swaps + token transfers)
        const response = await qnkAPI.getWalletHistory(currentWalletAddress, 100);
        console.log('📋 Wallet history API response:', response);
        if (!mounted) return;

        // Merge with existing client-side mining transactions
        setRecentTransactions(prev => {
          // Preserve mining transactions (client-side added)
          const preservedTxs = prev.filter(tx =>
            tx.id.startsWith('mining-')
          );

          // If API call failed or returned no data, keep preserved transactions
          if (!response.success || !response.data) {
            console.log('📋 API failed or no data, keeping preserved transactions');
            console.log('📋 API error details:', response.error);

            // Set visible error for user
            setTransactionError(response.error || 'Failed to load transactions');

            return preservedTxs;
          }

          // Clear error on successful load
          setTransactionError(null);

          // v3.5.8-beta: Transform UnifiedTransactionEntry[] to frontend Transaction interface
          const burnAddress = '0000000000000000000000000000000000000000000000000000000000000000';
          const transformedTransactions: Transaction[] = response.data
            .filter((tx: any) => {
              // Filter out invalid transactions (swaps may have different structure)
              if (tx.tx_type !== 'swap' && (!tx.from || !tx.to)) return false;
              // Allow burn transactions (to burn address) but not from burn address
              if (tx.from === burnAddress) return false;
              return true;
            })
            .map((tx: any) => {
              // v3.5.8-beta: Map tx_type and direction to Transaction type
              let type: 'receive' | 'send' | 'mining' | 'swap';

              if (tx.tx_type === 'swap') {
                type = 'swap';
              } else if (tx.tx_type === 'mining_reward') {
                type = 'mining';
              } else if (tx.direction === 'received') {
                type = 'receive';
              } else {
                type = 'send';
              }

              console.log('📋 Transaction type detection:', {
                tx_type: tx.tx_type,
                direction: tx.direction,
                detectedType: type
              });

              // Convert Unix timestamp (seconds) to ISO string
              const timestamp = typeof tx.timestamp === 'number'
                ? new Date(tx.timestamp * 1000).toISOString()
                : tx.timestamp;

              // Parse amount - it comes as string from unified API
              // Convert from smallest units to display units (QUG has 24 decimals)
              const rawAmount = typeof tx.amount === 'string'
                ? parseFloat(tx.amount)
                : (tx.amount || 0);
              const amount = rawAmount / 1e24;

              // Label burn address as "Nitro Points Purchase"
              const toHex = tx.to?.startsWith('qnk') ? tx.to.substring(3) : (tx.to || '');
              const fromHex = tx.from?.startsWith('qnk') ? tx.from.substring(3) : (tx.from || '');
              // Ensure qnk prefix on addresses (API may return raw hex)
              const ensureQnk = (a: string) => a && /^[0-9a-fA-F]{64}$/.test(a) ? `qnk${a}` : a;
              const displayTo = toHex === burnAddress ? 'Nitro Points Purchase ⚡' : ensureQnk(tx.to);
              const displayFrom = fromHex === burnAddress ? 'Burn Address' : ensureQnk(tx.from);

              return {
                id: tx.id,
                type,
                amount,
                from: displayFrom,
                to: displayTo,
                timestamp,
                txHash: tx.id,
                // v3.5.8-beta: Additional fields for swaps and token transfers
                tokenSymbol: tx.token_symbol,
                tokenAddress: tx.token_address,
                amountOut: tx.amount_out,
                tokenIn: tx.token_in,
                tokenOut: tx.token_out,
                memo: tx.memo || undefined,
              };
            });

          console.log('📋 Transformed API transactions:', transformedTransactions.length);

          // Merge and deduplicate by id
          const allTxs = [...preservedTxs, ...transformedTransactions];
          const uniqueTxs = allTxs.filter((tx, index, self) =>
            index === self.findIndex(t => t.id === tx.id)
          );

          console.log('📋 Final merged transactions:', uniqueTxs.length);

          // Sort by timestamp descending
          return uniqueTxs.sort((a, b) =>
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
          );
        });
      } catch (err) {
        console.error('❌ Error fetching wallet history:', err);
        // On error, keep existing list intact — don't clear visible activity
      }
    };

    const generateWalletAddress = async () => {
      console.log('Generating wallet address...');
      const storedAddress = localStorage.getItem('walletAddress');
      const storedMnemonic = localStorage.getItem('walletSeed');

      // If address exists, always use it (regardless of mnemonic)
      if (storedAddress) {
        console.log('Using stored wallet address:', storedAddress);
        setWalletAddress(storedAddress);
        return;
      }

      if (storedMnemonic && !storedAddress) {
        try {
          const encoder = new TextEncoder();
          const data = encoder.encode(storedMnemonic);
          const hashBuffer = await crypto.subtle.digest('SHA-256', data);
          const hashArray = Array.from(new Uint8Array(hashBuffer));
          const address = 'qnk' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('').substring(0, 40);

          localStorage.setItem('walletAddress', address);
          setWalletAddress(address);
          console.log('Generated wallet from mnemonic');
          return;
        } catch (error) {
          console.error('Failed to derive address from mnemonic:', error);
        }
      }

      try {
        const response = await qnkAPI.generateMnemonic();
        if (response.success && response.data?.mnemonic) {
          const encoder = new TextEncoder();
          const data = encoder.encode(response.data.mnemonic);
          const hashBuffer = await crypto.subtle.digest('SHA-256', data);
          const hashArray = Array.from(new Uint8Array(hashBuffer));
          const address = 'qnk' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('').substring(0, 40);

          localStorage.setItem('walletAddress', address);
          // SECURITY: Do NOT store plaintext mnemonic
          // User must import wallet via LoginScreen with password encryption

          setWalletAddress(address);
          console.log('Generated new wallet address (mnemonic NOT stored - user must import with password)');
        } else {
          const prefix = 'qnk';
          const randomBytes = new Uint8Array(20);
          crypto.getRandomValues(randomBytes);
          const address = prefix + Array.from(randomBytes).map(b => b.toString(16).padStart(2, '0')).join('');

          // CRITICAL FIX: Save fallback wallet to localStorage
          localStorage.setItem('walletAddress', address);
          setWalletAddress(address);
          console.log('Generated fallback wallet and saved to localStorage');
        }
      } catch (error) {
        console.error('Failed to generate wallet address:', error);
        const prefix = 'qnk';
        const randomBytes = new Uint8Array(20);
        crypto.getRandomValues(randomBytes);
        const address = prefix + Array.from(randomBytes).map(b => b.toString(16).padStart(2, '0')).join('');

        // CRITICAL FIX: Save error fallback wallet to localStorage
        localStorage.setItem('walletAddress', address);
        setWalletAddress(address);
        console.log('Generated error fallback wallet and saved to localStorage');
      }
    };

    // ============================================
    // DEBOUNCED WRAPPERS - DISABLED (SSE in App.tsx)
    // ============================================
    // SSE connection moved to App.tsx, so Dashboard no longer needs these
    // const fetchNodeStatus = debounce(fetchNodeStatusCore, 200);
    // const fetchRecentTransactions = debounce(fetchRecentTransactionsCore, 200);
    console.log('ℹ️  [SSE DISABLED] Dashboard no longer uses local SSE - App.tsx handles it');
    // ============================================

    const loadData = async (retryCount = 0) => {
      // Only show spinner if we have no cached data yet
      const hasCachedBalance = (() => {
        const v = parseFloat(localStorage.getItem('cachedBalance') || '0');
        return !isNaN(v) && v > 0;
      })();
      if (!hasCachedBalance) setLoading(true);
      try {
        await generateWalletAddress();
        // v8.9.9: Add 15s timeout to prevent infinite hang when API doesn't respond
        await Promise.race([
          Promise.all([fetchNodeStatusCore(), fetchRecentTransactionsCore(), fetchWalletBalances()]),
          new Promise((_, reject) => setTimeout(() => reject(new Error('Dashboard load timeout')), 15000)),
        ]);
        setLoading(false);
      } catch (error) {
        console.error('[loadData] Error:', error);
        if (retryCount < 2 && mounted) {
          // Retry with backoff — don't touch loading state (retry will handle it)
          setTimeout(() => { if (mounted) loadData(retryCount + 1); }, (retryCount + 1) * 500);
        } else {
          // Final retry failed or unmounted — always clear loading
          setLoading(false);
        }
      }
    };

    // Fast-path: load native QUG balance immediately before full data load
    const addr = localStorage.getItem('walletAddress');
    if (addr) {
      qnkAPI.getWalletBalance(addr).then(r => {
        if (!mounted || !r.success || !r.data) return;
        const bal = r.data.balance_qnk || 0;
        if (bal > 0) {
          highestKnownBalancesRef.current['QUG'] = bal;
          safeCacheBalance(bal);
          setWalletBalances(prev => {
            const existing = prev.find(w => w.symbol === 'QUG');
            if (existing) {
              return prev.map(w => w.symbol === 'QUG' ? { ...w, balance: bal } : w);
            }
            return [{ symbol: 'QUG', name: 'Quillon Graph', balance: bal, icon: 'qug', color: 'from-amber-400 to-yellow-500', history: [] }, ...prev];
          });
          setLoading(false); // show dashboard immediately once we have the balance
        }
      }).catch(() => {/* silent — full loadData handles fallback */});
    }

    // Full data load (runs concurrently with fast-path above)
    loadData();

    // Set up SSE for real-time balance updates
    // CRITICAL: Pass wallet_address parameter for privacy-filtered SSE
    // const currentWalletForSSE = localStorage.getItem('walletAddress') || ''; // Disabled - App.tsx handles SSE
    // SSE URL construction disabled - App.tsx handles SSE
    // const sseUrl = import.meta.env.VITE_API_URL ?
    //   `${import.meta.env.VITE_API_URL}/v1/events?wallet_address=${encodeURIComponent(currentWalletForSSE)}` :
    //   `/api/v1/events?wallet_address=${encodeURIComponent(currentWalletForSSE)}`;

    // CRITICAL FIX: Dashboard's SSE connection is DISABLED
    // App.tsx already has an SSE connection that handles balance updates
    // Having two SSE connections causes duplicate events and flickering
    // Dashboard will receive balance updates via App.tsx's SSE connection
    console.log('ℹ️  Dashboard SSE disabled - using App.tsx SSE connection instead');

    // Mark as "connected" immediately since App.tsx handles SSE
    if (mounted) {
      setSseConnected(true);
    }

    // Subscribe to sseManager for instant Recent Activity updates.
    // sseManager is App.tsx's shared singleton — subscribing here does NOT
    // create a second SSE connection; it just fans out to another listener.
    const currentWalletAddr = localStorage.getItem('walletAddress') || '';
    let txRefreshDebounce: ReturnType<typeof setTimeout> | null = null;
    const triggerTxRefresh = () => {
      if (!mounted) return;
      if (txRefreshDebounce) clearTimeout(txRefreshDebounce);
      txRefreshDebounce = setTimeout(() => {
        if (mounted) fetchRecentTransactionsCore();
      }, 200); // 200ms debounce — collapse rapid-fire events into one fetch
    };

    // balance-updated: fires when a QUG transfer is sent/received or mining reward credited
    const unsubBalance = sseManager.on('balance-updated', (data: any) => {
      const payload = data?.data ?? data;
      const addr = payload?.wallet_address ?? '';
      if (!addr || addr.includes(currentWalletAddr.replace(/^qnk/, '')) ||
          currentWalletAddr.includes(addr.replace(/^qnk/, ''))) {
        triggerTxRefresh();
      }
    });

    // pending_mining_reward: fires immediately when a block reward is pending
    const unsubMining = sseManager.on('pending_mining_reward', () => triggerTxRefresh());

    // transaction-submitted: fires when our node processes a new transaction
    const unsubTxSubmitted = sseManager.on('transaction-submitted', () => triggerTxRefresh());

    // v3.4.15: Cleanup for the initial delay timeout + SSE subscriptions
    return () => {
      mounted = false;
      // (initialDelay removed — loadData fires immediately)
      if (txRefreshDebounce) clearTimeout(txRefreshDebounce);
      unsubBalance();
      unsubMining();
      unsubTxSubmitted();
    };

    // Early return - skip all SSE setup since App.tsx handles it
    // The cleanup function below will still run on unmount
    /*
    console.log('📡 Attempting SSE connection to:', sseUrl);
    console.log('📡 SSE wallet filter:', currentWalletForSSE);

    try {
      eventSource = new EventSource(sseUrl);

      eventSource.onopen = () => {
        console.log('✅ SSE connection established');
        console.log('🔍 Current wallet address:', localStorage.getItem('walletAddress'));
        if (mounted) {
          setSseConnected(true);
        }
      };

      // Listen for all possible SSE event types that backend might send
      const handleSpecificEvent = (eventType: string) => (event: MessageEvent) => {
        console.log(`🎯 SSE SPECIFIC EVENT [${eventType}]:`, event);
        console.log(`📨 Event data [${eventType}]:`, event.data);
        if (!mounted) return;

        try {
          const data = JSON.parse(event.data);
          console.log(`📦 Parsed data [${eventType}]:`, data);

          // Handle transaction events - refresh recent activity immediately
          if (eventType === 'transaction-submitted' || eventType === 'transaction-confirmed' || eventType === 'transaction-status') {
            console.log(`🔄 Transaction event received [${eventType}] - refreshing recent activity`);
            fetchRecentTransactions();
            fetchNodeStatus(); // Also refresh balance
            return; // Exit early
          }

          // Handle mining reward events - add to recent activity
          if (eventType === 'mining_reward') {
            console.log(`⛏️ Mining reward event received - adding to recent activity`);
            console.log(`⛏️ Mining reward RAW event:`, event);
            console.log(`⛏️ Mining reward RAW data string:`, event.data);
            console.log(`⛏️ Mining reward PARSED data:`, data);
            console.log(`⛏️ Mining reward data keys:`, Object.keys(data));
            console.log(`⛏️ Mining reward data structure:`, JSON.stringify(data, null, 2));

            // The data is nested: data.data contains the actual mining reward fields
            const miningData = data.data || data;
            console.log(`⛏️ Mining reward miningData:`, miningData);
            console.log(`⛏️ Mining reward miningData.reward_qnk:`, miningData.reward_qnk);
            console.log(`⛏️ Mining reward miningData.nonce:`, miningData.nonce);
            console.log(`⛏️ Mining reward miningData.block_height:`, miningData.block_height);
            console.log(`⛏️ Mining reward miningData.miner_address:`, miningData.miner_address);
            console.log(`⛏️ Mining reward miningData.timestamp:`, miningData.timestamp);
            try {
              const miningTransaction: Transaction = {
                id: `mining-${Date.now()}-${miningData.nonce}`,
                type: 'mining',
                amount: miningData.reward_qnk, // Already in QUG units from backend
                from: 'Mining Reward',
                to: miningData.miner_address,
                timestamp: miningData.timestamp,
                txHash: `mining-${miningData.block_height}-${miningData.nonce}`,
              };
              setRecentTransactions(prev => [miningTransaction, ...prev.slice(0, 49)]); // Keep last 50
              console.log(`⛏️ Mining reward added to recent activity: ${miningData.reward_qnk} QUG`);
            } catch (error) {
              console.error('Failed to process mining reward:', error);
            }
            fetchNodeStatus(); // Refresh balance
            return;
          }

          // Handle balance-updated event
          if (eventType === 'balance-updated') {
            const currentWalletAddress = localStorage.getItem('walletAddress');
            console.log('🔍 WALLET COMPARISON DEBUG v2:', {
              raw_current: currentWalletAddress,
              has_prefix: currentWalletAddress?.startsWith('qnk'),
              raw_event: data.wallet_address
            });

            const currentHex = (currentWalletAddress?.startsWith('qnk')
              ? currentWalletAddress.substring(3)
              : currentWalletAddress)?.toLowerCase();
            const eventHex = data.data?.wallet_address?.toLowerCase() || data.wallet_address?.toLowerCase();

            console.log('🔍 AFTER PROCESSING v2:', {
              currentHex,
              eventHex,
              areEqual: eventHex === currentHex,
              currentLength: currentHex?.length,
              eventLength: eventHex?.length
            });

            console.log('💰 BALANCE UPDATE EVENT v2:', {
              eventType,
              eventData: data,
              eventWallet: eventHex,
              currentWallet: currentHex,
              match: eventHex === currentHex,
              newBalance: data.new_balance || data.data?.new_balance,
              oldBalance: data.old_balance || data.data?.old_balance,
              reason: data.change_reason || data.data?.change_reason
            });

            // CRITICAL FIX: Only apply balance update if wallet addresses EXACTLY match
            // Do NOT accept if currentHex is empty - that would apply ALL balance updates
            if (currentHex && eventHex === currentHex) {
              const newBalance = data.data?.new_balance || data.new_balance;

              console.log('✅ APPLYING BALANCE UPDATE v2:', newBalance);
              setNodeStatus(prev => {
                console.log('🔄 Updating nodeStatus from', prev?.balance, 'to', newBalance);
                return prev ? { ...prev, balance: newBalance } : prev;
              });

              // NOTE: No need to dispatch to App.tsx - it has its own SSE connection

              // Refresh recent transactions to show new activity
              console.log('🔄 Refreshing recent transactions after balance update');
              fetchRecentTransactions();
            } else {
              console.log('❌ Balance update IGNORED (different wallet)', {
                reason: !currentHex ? 'no current wallet hex' : 'wallet address mismatch',
                currentHex,
                eventHex
              });
            }
          } else if (eventType === 'mining_reward') {
            console.log('💎 MINING REWARD EVENT:', data);
            const currentWalletAddress = localStorage.getItem('walletAddress');

            // Normalize wallet addresses for comparison (strip "qnk" prefix and compare hex)
            const currentHex = (currentWalletAddress?.startsWith('qnk')
              ? currentWalletAddress.substring(3)
              : currentWalletAddress)?.toLowerCase();
            const minerHex = (data.miner_address?.startsWith('qnk')
              ? data.miner_address.substring(3)
              : data.miner_address)?.toLowerCase();

            console.log('💎 Mining reward wallet comparison:', {
              currentWallet: currentWalletAddress,
              currentHex,
              minerAddress: data.miner_address,
              minerHex,
              match: currentHex && minerHex === currentHex
            });

            // Check if this mining reward is for the current wallet
            if (currentHex && minerHex === currentHex) {
              console.log('✅ Mining reward for current wallet:', {
                reward: data.reward_qnk,
                nonce: data.nonce,
                blockHeight: data.block_height
              });

              // Refresh balance immediately
              fetchNodeStatus();

              // NOTE: No need to dispatch to App.tsx - it has its own SSE connection

              // Add mining transaction to recent activity
              const miningTx: Transaction = {
                id: `mining-${data.nonce}-${data.block_height}`,
                type: 'receive',
                amount: data.reward_qnk,
                from: `Mining Reward (Block ${data.block_height})`,
                to: data.miner_address,
                timestamp: data.timestamp,
                txHash: `mining-${data.nonce}`,
              };

              setRecentTransactions(prev => [miningTx, ...prev]);

              console.log('💎 Mining reward transaction added to recent activity');
            } else {
              console.log('ℹ️ Mining reward for different wallet, ignoring', {
                reason: !currentHex ? 'no current wallet hex' : 'wallet address mismatch'
              });
            }
          } else if (eventType === 'mining_stats') {
            console.log('📊 MINING STATS EVENT:', data);
            const currentWalletAddress = localStorage.getItem('walletAddress');

            // Normalize wallet addresses for comparison (strip "qnk" prefix and compare hex)
            const currentHex = (currentWalletAddress?.startsWith('qnk')
              ? currentWalletAddress.substring(3)
              : currentWalletAddress)?.toLowerCase();
            const minerHex = (data.miner_address?.startsWith('qnk')
              ? data.miner_address.substring(3)
              : data.miner_address)?.toLowerCase();

            console.log('📊 Mining stats wallet comparison:', {
              currentWallet: currentWalletAddress,
              currentHex,
              minerAddress: data.miner_address,
              minerHex,
              match: currentHex && minerHex === currentHex
            });

            // Check if these stats are for the current wallet
            if (currentHex && minerHex === currentHex) {
              console.log('✅ Mining stats for current wallet:', {
                totalRewards: data.total_rewards,
                totalBlocks: data.total_blocks_found,
                currentBalance: data.current_balance
              });

              // Update balance if provided
              if (data.current_balance !== undefined) {
                setNodeStatus(prev => prev ? { ...prev, balance: data.current_balance } : prev);

                // NOTE: No need to dispatch to App.tsx - it has its own SSE connection
              }
            } else {
              console.log('ℹ️ Mining stats for different wallet, ignoring', {
                reason: !currentHex ? 'no current wallet hex' : 'wallet address mismatch'
              });
            }
          }
        } catch (error) {
          console.error(`❌ Error parsing ${eventType} event:`, error);
        }
      };

      // Add listeners for specific event types
      eventSource.addEventListener('balance-updated', handleSpecificEvent('balance-updated'));
      eventSource.addEventListener('transaction-confirmed', handleSpecificEvent('transaction-confirmed'));
      eventSource.addEventListener('transaction-submitted', handleSpecificEvent('transaction-submitted'));
      eventSource.addEventListener('transaction-status', handleSpecificEvent('transaction-status'));
      eventSource.addEventListener('mining_reward', handleSpecificEvent('mining_reward'));
      eventSource.addEventListener('mining_stats', handleSpecificEvent('mining_stats'));

      console.log('✅ SSE event listeners registered for: balance-updated, transaction-confirmed, transaction-submitted, transaction-status, mining_reward, mining_stats');

      eventSource.onmessage = (event) => {
        console.log('📨 SSE DEFAULT MESSAGE (onmessage):', event);
        console.log('📨 Event type:', event.type);
        console.log('📨 Event data:', event.data);
        console.log('📨 Event lastEventId:', event.lastEventId);
        if (!mounted) return;

        try {
          const data = JSON.parse(event.data);
          console.log('📦 SSE data parsed (onmessage):', data);
          console.log('📦 Data type field:', data.type);
          console.log('📦 Data keys:', Object.keys(data));

          // Handle different SSE event types from the data.type field
          if (data.type === 'balance-updated' && data.data?.new_balance !== undefined) {
            const currentWalletAddress = localStorage.getItem('walletAddress');
            const currentHex = (currentWalletAddress?.startsWith('qnk')
              ? currentWalletAddress.substring(3)
              : currentWalletAddress)?.toLowerCase();
            const eventHex = data.data.wallet_address?.toLowerCase();

            console.log('💰 Dashboard: Balance update SSE event (onmessage) v2:', {
              eventWallet: eventHex,
              currentWallet: currentHex,
              match: eventHex === currentHex,
              newBalance: data.data.new_balance,
              reason: data.data.change_reason
            });

            // CRITICAL FIX: Only apply balance update if wallet addresses EXACTLY match
            // Do NOT accept if currentHex is empty - that would apply ALL balance updates
            if (currentHex && eventHex === currentHex) {
              const changeReason = data.data.change_reason || '';
              const isP2PMiningReward = changeReason === 'p2p_mining_reward' || changeReason === 'pending_mining_reward';

              if (isP2PMiningReward) {
                // P2P mining rewards: ACCUMULATE instead of replace
                // The new_balance from bootstrap is STALE (it doesn't have accumulated balance)
                // Calculate reward amount and ADD to current balance
                const rewardAmount = (data.data.new_balance || 0) - (data.data.old_balance || 0);
                console.log('✅ Dashboard: P2P mining reward - ACCUMULATING:', {
                  rewardAmount,
                  oldBalance: data.data.old_balance,
                  newBalance: data.data.new_balance,
                  reason: changeReason
                });
                setNodeStatus(prev => {
                  if (!prev) return prev;
                  const newBalance = (prev.balance || 0) + rewardAmount;
                  console.log('💰 Dashboard: Balance accumulated:', prev.balance, '+', rewardAmount, '=', newBalance);
                  return { ...prev, balance: newBalance };
                });
              } else {
                // Local mining rewards: use new_balance directly (local RocksDB has correct value)
                console.log('✅ Dashboard: Balance update applied (onmessage):', data.data.new_balance);
                setNodeStatus(prev => prev ? { ...prev, balance: data.data.new_balance } : prev);
              }

              // NOTE: No need to dispatch to App.tsx - it has its own SSE connection

              // Refresh recent transactions to show new activity
              console.log('🔄 Refreshing recent transactions after balance update (onmessage)');
              fetchRecentTransactions();
            } else {
              console.log('❌ Dashboard: Balance update ignored (not for current wallet)', {
                reason: !currentHex ? 'no current wallet hex' : 'wallet address mismatch',
                currentHex,
                eventHex
              });
            }
          } else if (data.type === 'transaction-confirmed' || data.type === 'transaction-submitted') {
            console.log('🔄 Transaction event - refreshing data');
            fetchNodeStatus();
            fetchRecentTransactions();
          } else if (data.type === 'Custom' && data.data?.event_type === 'mining_reward') {
            // Mining reward event - data is nested in data.data
            console.log('💎 Mining reward event received:', data);
            const rewardData = data.data.data; // Extract nested reward data
            const currentWalletAddress = localStorage.getItem('walletAddress');

            // Normalize wallet addresses for comparison (strip "qnk" prefix and compare hex)
            const currentHex = (currentWalletAddress?.startsWith('qnk')
              ? currentWalletAddress.substring(3)
              : currentWalletAddress)?.toLowerCase();
            const minerHex = (rewardData.miner_address?.startsWith('qnk')
              ? rewardData.miner_address.substring(3)
              : rewardData.miner_address)?.toLowerCase();

            console.log('💎 Mining reward (Custom) wallet comparison:', {
              currentWallet: currentWalletAddress,
              currentHex,
              minerAddress: rewardData.miner_address,
              minerHex,
              match: currentHex && minerHex === currentHex
            });

            // Check if this mining reward is for the current wallet
            if (currentHex && minerHex === currentHex) {
              console.log('✅ Mining reward for current wallet:', {
                reward: rewardData.reward_qnk,
                newBalance: rewardData.new_balance_qnk,
                nonce: rewardData.nonce
              });

              // Update balance immediately
              setNodeStatus(prev => prev ? { ...prev, balance: rewardData.new_balance_qnk } : prev);

              // NOTE: No need to dispatch to App.tsx - it has its own SSE connection

              // Add mining transaction to recent activity
              const miningTx: Transaction = {
                id: `mining-${rewardData.tx_hash}`,
                type: 'receive',
                amount: rewardData.reward_qnk,
                from: 'Mining Reward (VDF)',
                to: rewardData.miner_address,
                timestamp: rewardData.timestamp,
                txHash: rewardData.tx_hash,
              };

              setRecentTransactions(prev => [miningTx, ...prev]);

              console.log('💎 Mining reward transaction added to recent activity');
            } else {
              console.log('ℹ️ Mining reward (Custom) for different wallet, ignoring', {
                reason: !currentHex ? 'no current wallet hex' : 'wallet address mismatch'
              });
            }
          } else if (data.type === 'new-transaction' && data.transaction) {
            console.log('📬 New transaction via SSE:', data.transaction);
            setRecentTransactions(prev => [data.transaction, ...prev].slice(0, 5));
          } else if (data.type === 'block-confirmed' || data.type === 'consensus-round') {
            console.log('⛓️ Block confirmed - updating status');
            fetchNodeStatus();
          } else {
            console.log('❓ Unknown SSE event type:', data.type);
          }
        } catch (error) {
          console.error('❌ Error processing SSE event:', error);
          console.error('❌ Raw event data:', event.data);
        }
      };

      eventSource.onerror = (error) => {
        console.error('❌ SSE connection error:', error);
        console.log('SSE readyState:', eventSource?.readyState);
        console.log('SSE url:', eventSource?.url);
        if (mounted) {
          setSseConnected(false);
        }
        eventSource?.close();
      };
    } catch (error) {
      console.error('❌ Failed to create SSE connection:', error);
    }

    // Listen for CDP mint events from MintQUGUSDModal
    const handleCDPMint = (event: Event) => {
      if (!mounted) return;

      const customEvent = event as CustomEvent;
      const data = customEvent.detail;

      console.log('💵 CDP Mint Event:', data);

      // Create transaction for Recent Activity
      const cdpTransaction: Transaction = {
        id: `cdp-mint-${data.transaction_id}`,
        type: 'send', // Sending QUG to CDP vault
        amount: data.collateral_amount,
        from: data.wallet_address,
        to: 'CDP Vault (QUGUSD Minting)',
        timestamp: data.timestamp,
        txHash: data.transaction_id,
      };

      // Add to recent transactions
      setRecentTransactions(prev => [cdpTransaction, ...prev]);

      console.log('💵 CDP transaction added to recent activity');
    };

    window.addEventListener('cdp-mint', handleCDPMint);

    return () => {
      mounted = false;
      // eventSource cleanup not needed - SSE disabled, App.tsx handles it
      window.removeEventListener('cdp-mint', handleCDPMint);
    };
    */
  }, []);

  // v2.3.26-beta: Track DEX swap cooldown AND lock the balance value
  const dexSwapCooldownRef = useRef(false);
  const lockedQugBalanceRef = useRef<number | null>(null);
  const qugPriceUsdRef = useRef<number | undefined>(undefined);

  // v2.3.26-beta: Listen for qug-balance-changed event (from DEX swap) - highest priority
  useEffect(() => {
    const handleQugBalanceChanged = (event: Event) => {
      const customEvent = event as CustomEvent;
      const newBalance = customEvent.detail?.balance;
      if (typeof newBalance === 'number') {
        console.log('🔥 Dashboard: qug-balance-changed - LOCKING QUG balance to:', newBalance);

        // LOCK this balance - it cannot be overwritten for 10 seconds
        lockedQugBalanceRef.current = newBalance;
        dexSwapCooldownRef.current = true;
        setTimeout(() => {
          dexSwapCooldownRef.current = false;
          lockedQugBalanceRef.current = null;
          console.log('🔓 Dashboard: DEX swap cooldown ended, balance unlocked');
        }, 10000);

        // Update tracking and localStorage
        highestKnownBalancesRef.current['QUG'] = newBalance;
        safeCacheBalance(newBalance);

        // Update wallet balances state
        setWalletBalances(wallets => {
          return wallets.map(wallet => {
            if (wallet.symbol === 'QUG') {
              console.log(`🔥 Dashboard: Updating QUG balance: ${wallet.balance} -> ${newBalance}`);
              const prev = wallet.history || [];
              const last = prev[prev.length - 1];
              const pctDiff = last && last.balance > 0
                ? Math.abs(newBalance - last.balance) / last.balance : 1;
              // Always add for DEX swaps (big changes), skip tiny noise
              const newHistory = (pctDiff < 0.005 && last && Date.now() - last.timestamp < 3000)
                ? prev
                : [...prev, { timestamp: Date.now(), balance: newBalance }].slice(-10080);
              return {
                ...wallet,
                balance: newBalance,
                usdValue: qugPriceUsdRef.current !== undefined ? newBalance * qugPriceUsdRef.current : wallet.usdValue,
                history: newHistory
              };
            }
            return wallet;
          });
        });
      }
    };

    window.addEventListener('qug-balance-changed', handleQugBalanceChanged);
    return () => window.removeEventListener('qug-balance-changed', handleQugBalanceChanged);
  }, []);

  // v2.3.33-beta: Listen for dex-cooldown-expired to sync walletBalances state from cached values
  // This is CRITICAL: After cooldown expires, walletBalances state needs to be updated with correct values
  useEffect(() => {
    const handleDexCooldownExpired = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { qugBalance, qugusdBalance, source } = customEvent.detail;

      console.log('🔄 Dashboard: Received dex-cooldown-expired event from', source, {
        qugBalance,
        qugusdBalance
      });

      // Update walletBalances state with the cached correct values (no history point for cooldown sync)
      setWalletBalances(wallets => {
        return wallets.map(wallet => {
          if (wallet.symbol === 'QUG' && qugBalance !== null && !isNaN(qugBalance)) {
            console.log(`🔄 Dashboard: Syncing QUG state after cooldown: ${wallet.balance} -> ${qugBalance}`);
            return { ...wallet, balance: qugBalance };
          }
          if (wallet.symbol === 'QUGUSD' && qugusdBalance !== null && !isNaN(qugusdBalance)) {
            console.log(`🔄 Dashboard: Syncing QUGUSD state after cooldown: ${wallet.balance} -> ${qugusdBalance}`);
            return { ...wallet, balance: qugusdBalance };
          }
          return wallet;
        });
      });

      // Also clear local cooldown ref
      dexSwapCooldownRef.current = false;
      lockedQugBalanceRef.current = null;
      console.log('🔓 Dashboard: Cooldown refs cleared after dex-cooldown-expired event');
    };

    window.addEventListener('dex-cooldown-expired', handleDexCooldownExpired);
    return () => window.removeEventListener('dex-cooldown-expired', handleDexCooldownExpired);
  }, []);

  // v2.3.26-beta: Force locked balance during cooldown (check only when cooldown state changes)
  const prevCooldownRef = useRef(false);
  useEffect(() => {
    if (dexSwapCooldownRef.current && lockedQugBalanceRef.current !== null && !prevCooldownRef.current) {
      prevCooldownRef.current = true;
      const lockedBalance = lockedQugBalanceRef.current;
      setWalletBalances(wallets => {
        const qugWallet = wallets.find(w => w.symbol === 'QUG');
        if (qugWallet && qugWallet.balance !== lockedBalance) {
          return wallets.map(wallet =>
            wallet.symbol === 'QUG' ? { ...wallet, balance: lockedBalance } : wallet
          );
        }
        return wallets;
      });
    } else if (!dexSwapCooldownRef.current) {
      prevCooldownRef.current = false;
    }
  }); // Intentionally no deps - but now guarded by ref to prevent infinite loop

  // Listen for real-time balance updates from SSE (via App.tsx custom event)
  // v8.0.3: Track last accepted balance per symbol to prevent zigzag from competing sources
  const lastAcceptedBalanceRef = useRef<Record<string, { balance: number; timestamp: number }>>({});

  useEffect(() => {
    const handleWalletBalanceUpdate = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { symbol, balance: incomingBalance, reason, infoOnly } = customEvent.detail;

      // v8.0.3: Skip info-only events (pending-mining-reward uses balance-updated SSE for actual balance)
      if (infoOnly) {
        return;
      }

      // v2.3.13-beta: DEX swaps are ALWAYS trusted - simplified logic
      const isDexSwap = reason === 'dex-swap-deduct' || reason === 'dex-swap-add';

      // v2.3.31-beta: Check BOTH our ref AND localStorage cooldown (set by DexScreen BEFORE API call)
      const globalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
      const globalCooldownActive = Date.now() < globalCooldownUntil;
      const cooldownActive = dexSwapCooldownRef.current || globalCooldownActive;

      // Block non-DEX updates during cooldown, but ALWAYS allow DEX updates
      if (cooldownActive && !isDexSwap) {
        console.log('🚫 Dashboard: Ignoring non-DEX wallet-balance-updated during cooldown:', symbol, incomingBalance, '(global:', globalCooldownActive, ')');
        return;
      }

      // v1.0.2: Stabilizer — reject spurious balance drops from stale SSE events
      // Allow legitimate decreases from sends/swaps (check change_reason)
      const isLegitimateDecrease = reason === 'transaction_sent' || reason === 'send' || reason === 'transfer' || reason === 'dex_swap' || isDexSwap;
      if (!isDexSwap && !isLegitimateDecrease && symbol === 'QUG') {
        const lastAccepted = lastAcceptedBalanceRef.current[symbol];
        if (lastAccepted && Date.now() - lastAccepted.timestamp < 10000) {
          // Within 10s window, only reject > 10% drops (likely stale data, not a real send)
          if (incomingBalance < lastAccepted.balance * 0.9) {
            console.log(`🚫 Dashboard: Rejecting large balance drop within 10s window: ${incomingBalance} < ${lastAccepted.balance * 0.9}`);
            return;
          }
        }
      }
      // Always update the tracking ref with the latest accepted balance
      lastAcceptedBalanceRef.current[symbol] = { balance: incomingBalance, timestamp: Date.now() };

      console.log(`💰 Dashboard: Received wallet-balance-updated event for ${symbol}:`, incomingBalance, 'Reason:', reason, 'isDexSwap:', isDexSwap);

      // v2.3.13-beta: For DEX swaps, IMMEDIATELY update everything without validation
      // This is the ONLY way to ensure the correct balance is displayed
      if (isDexSwap) {
        console.log(`🔥 Dashboard: DEX SWAP - Force updating ${symbol} to ${incomingBalance}`);

        // Immediately update all tracking refs and storage
        highestKnownBalancesRef.current[symbol] = incomingBalance;
        if (symbol === 'QUG') {
          safeCacheBalance(incomingBalance);
        } else if (symbol === 'QUGUSD') {
          localStorage.setItem('cachedQugusdBalance', incomingBalance.toString());
        }

        // Immediately update wallet balances state
        setWalletBalances(wallets => {
          return wallets.map(wallet => {
            if (wallet.symbol === symbol) {
              console.log(`🔥 Dashboard: DEX SWAP updating ${symbol} balance: ${wallet.balance} -> ${incomingBalance}`);
              return {
                ...wallet,
                balance: incomingBalance,
                history: (() => {
                  const prev = wallet.history || [];
                  const last = prev[prev.length - 1];
                  const pctDiff = last && last.balance > 0
                    ? Math.abs(incomingBalance - last.balance) / last.balance : 1;
                  // Skip if <0.5% change and within 3s (DEX swaps are always big changes)
                  if (last && pctDiff < 0.005 && Date.now() - last.timestamp < 3000) {
                    return prev;
                  }
                  return [...prev, { timestamp: Date.now(), balance: incomingBalance }].slice(-10080);
                })()
              };
            }
            return wallet;
          });
        });

        // Also update balance history state
        setBalanceHistory(prev => {
          const history = prev[symbol] || [];
          const last = history[history.length - 1];
          const pctDiff = last && last.balance > 0
            ? Math.abs(incomingBalance - last.balance) / last.balance : 1;
          if (last && pctDiff < 0.005 && Date.now() - last.timestamp < 3000) {
            return prev;
          }
          const newPoint: BalanceHistoryPoint = { timestamp: Date.now(), balance: incomingBalance };
          const updatedHistory = [...history, newPoint].slice(-10080);
          try {
            localStorage.setItem('qnk_balance_long_v1', JSON.stringify({ ...prev, [symbol]: updatedHistory }));
          } catch {}
          return { ...prev, [symbol]: updatedHistory };
        });

        return; // Skip all other logic for DEX swaps
      }

      // For non-DEX updates, apply anti-fraud validation
      const previousHighest = highestKnownBalancesRef.current[symbol] || 0;
      const cachedBalance = symbol === 'QUG'
        ? parseFloat(localStorage.getItem('cachedBalance') || '0')
        : symbol === 'QUGUSD'
          ? parseFloat(localStorage.getItem('cachedQugusdBalance') || '0')
          : 0;
      const referenceBalance = Math.max(previousHighest, cachedBalance);
      const minAcceptable = Math.max(0, referenceBalance * 0.9 - 1);
      let validatedBalance = incomingBalance;

      // v1.0.2: Only reject if balance drops to near-zero from a large value (likely stale/corrupt)
      // Allow legitimate decreases (sends, swaps) — the server is authoritative
      if (incomingBalance < minAcceptable && referenceBalance > 0 && incomingBalance < referenceBalance * 0.01) {
        console.warn(`⚠️ Dashboard: Rejecting suspicious near-zero balance for ${symbol}: ${incomingBalance} (expected ~${referenceBalance})`);
        validatedBalance = referenceBalance;
      } else if (incomingBalance !== previousHighest) {
        highestKnownBalancesRef.current[symbol] = incomingBalance;
        // v2.3.27-beta: Don't write to localStorage during DEX cooldown
        if (symbol === 'QUG' && !dexSwapCooldownRef.current) {
          safeCacheBalance(incomingBalance);
        }
      }

      console.log(`💰 Dashboard: Non-DEX update for ${symbol}:`, incomingBalance, '-> validated:', validatedBalance);

      // Update balance history and wallet balance atomically
      setBalanceHistory(prev => {
        const history = prev[symbol] || [];
        const last = history[history.length - 1];
        // Deduplicate: skip if <0.5% change and within 5s (mining rewards are tiny increments)
        if (last) {
          const pctDiff = last.balance > 0
            ? Math.abs(validatedBalance - last.balance) / last.balance : (validatedBalance !== last.balance ? 1 : 0);
          if (pctDiff < 0.005 && Date.now() - last.timestamp < 5000) {
            // Still update the displayed balance, just don't add a history point
            setWalletBalances(wallets => wallets.map(wallet =>
              wallet.symbol === symbol ? {
                ...wallet,
                balance: validatedBalance,
                usdValue: symbol === 'QUG' && qugPriceUsdRef.current !== undefined ? validatedBalance * qugPriceUsdRef.current : wallet.usdValue
              } : wallet
            ));
            return prev;
          }
        }
        const newPoint: BalanceHistoryPoint = {
          timestamp: Date.now(),
          balance: validatedBalance
        };
        const updatedHistory = [...history, newPoint].slice(-10080);
        const newHistoryState = { ...prev, [symbol]: updatedHistory };

        // Save to localStorage
        try {
          localStorage.setItem('qnk_balance_long_v1', JSON.stringify(newHistoryState));
        } catch (error) {
          console.warn('Failed to save balance history to localStorage:', error);
        }

        console.log(`📊 Dashboard: Updated history for ${symbol}:`, updatedHistory.length, 'points');

        // Update walletBalances with the new history
        setWalletBalances(wallets => {
          return wallets.map(wallet => {
            if (wallet.symbol === symbol) {
              console.log(`✅ Dashboard: Updating ${symbol} balance from ${wallet.balance} to ${validatedBalance} with ${updatedHistory.length} history points`);
              return {
                ...wallet,
                balance: validatedBalance,
                usdValue: symbol === 'QUG' && qugPriceUsdRef.current !== undefined ? validatedBalance * qugPriceUsdRef.current : wallet.usdValue,
                history: updatedHistory
              };
            }
            return wallet;
          });
        });

        return newHistoryState;
      });
    };

    window.addEventListener('wallet-balance-updated', handleWalletBalanceUpdate);
    console.log('👂 Dashboard: Listening for wallet-balance-updated events');

    return () => {
      window.removeEventListener('wallet-balance-updated', handleWalletBalanceUpdate);
      console.log('🔇 Dashboard: Stopped listening for wallet-balance-updated events');
    };
  }, []);

  // v8.6.5: Sync liveBalance prop (from App.tsx SSE — same source as TopBar) into walletBalances
  // This ensures Dashboard wallet tab matches TopBar balance in real-time
  useEffect(() => {
    if (liveBalance === undefined || liveBalance === null) return;
    if (!isValidBalance(liveBalance)) return;
    // Skip during DEX cooldown
    const globalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
    if (dexSwapCooldownRef.current || Date.now() < globalCooldownUntil) return;

    setWalletBalances(wallets => {
      const qugWallet = wallets.find(w => w.symbol === 'QUG');
      if (!qugWallet) return wallets;
      // Only update if balance actually changed
      if (Math.abs(qugWallet.balance - liveBalance) < 1e-12) return wallets;
      return wallets.map(wallet =>
        wallet.symbol === 'QUG' ? {
          ...wallet,
          balance: liveBalance,
          usdValue: qugPriceUsdRef.current !== undefined ? liveBalance * qugPriceUsdRef.current : wallet.usdValue
        } : wallet
      );
    });
  }, [liveBalance]);

  // v7.1.0: Instant SSE-driven transaction updates — mining rewards & transfers appear immediately
  useEffect(() => {
    let refetchTimer: ReturnType<typeof setTimeout> | null = null;

    const handleInstantTransaction = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { symbol, balance, oldBalance, reason, rewardAmount, blockHeight, blockHash, walletAddress: eventWallet, timestamp } = customEvent.detail;

      // Only handle QUG events with transaction-like reasons
      if (symbol !== 'QUG') return;
      const isMining = reason === 'p2p_mining_reward' || reason === 'pending_mining_reward' || reason === 'coinbase_reward';
      const isTransfer = reason === 'transaction_received' || reason === 'transaction_sent';
      if (!isMining && !isTransfer) return;

      // Calculate amount from reward or balance diff
      const amount = rewardAmount || (balance && oldBalance ? Math.abs(balance - oldBalance) : 0);
      if (amount <= 0) return;

      // Create an instant transaction entry
      const txId = `sse-${reason}-${blockHeight || Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const newTx: Transaction = {
        id: txId,
        type: isMining ? 'mining' : (reason === 'transaction_sent' ? 'send' : 'receive'),
        amount,
        from: isMining ? 'Mining Reward' : (reason === 'transaction_sent' ? (eventWallet || '') : ''),
        to: isMining ? (eventWallet || '') : (reason === 'transaction_received' ? (eventWallet || '') : ''),
        timestamp: timestamp || new Date().toISOString(),
        txHash: blockHash || txId,
      };

      console.log('⚡ Dashboard: Instant SSE transaction:', newTx.type, amount.toFixed(6), 'QUG');

      setRecentTransactions(prev => {
        // Deduplicate: skip if same blockHeight+reason already exists
        if (blockHeight && prev.some(tx => tx.txHash === blockHash)) return prev;
        // Add at top and keep max 100
        const updated = [newTx, ...prev].slice(0, 100);
        return updated;
      });

      // Debounced background refetch to reconcile with full API history (replaces instant entries)
      if (refetchTimer) clearTimeout(refetchTimer);
      refetchTimer = setTimeout(async () => {
        const currentWalletAddress = localStorage.getItem('walletAddress') || '';
        if (!currentWalletAddress) return;
        try {
          const response = await qnkAPI.getWalletHistory(currentWalletAddress, 100);
          if (response.success && response.data) {
            const burnAddress = '0000000000000000000000000000000000000000000000000000000000000000';
            const transformedTransactions: Transaction[] = response.data
              .filter((tx: any) => {
                if (tx.tx_type !== 'swap' && (!tx.from || !tx.to)) return false;
                if (tx.from === burnAddress) return false;
                return true;
              })
              .map((tx: any) => {
                let type: 'receive' | 'send' | 'mining' | 'swap';
                if (tx.tx_type === 'swap') type = 'swap';
                else if (tx.tx_type === 'mining_reward') type = 'mining';
                else if (tx.direction === 'received') type = 'receive';
                else type = 'send';
                const ts = typeof tx.timestamp === 'number' ? new Date(tx.timestamp * 1000).toISOString() : tx.timestamp;
                const rawAmt = typeof tx.amount === 'string' ? parseFloat(tx.amount) : (tx.amount || 0);
                const toHex = tx.to?.startsWith('qnk') ? tx.to.substring(3) : (tx.to || '');
                const fromHex = tx.from?.startsWith('qnk') ? tx.from.substring(3) : (tx.from || '');
                const ensureQnk = (a: string) => a && /^[0-9a-fA-F]{64}$/.test(a) ? `qnk${a}` : a;
                return {
                  id: tx.id,
                  type,
                  amount: rawAmt / 1e24,
                  from: fromHex === burnAddress ? 'Burn Address' : ensureQnk(tx.from),
                  to: toHex === burnAddress ? 'Nitro Points Purchase ⚡' : ensureQnk(tx.to),
                  timestamp: ts,
                  txHash: tx.id,
                  tokenSymbol: tx.token_symbol,
                  tokenAddress: tx.token_address,
                  amountOut: tx.amount_out,
                  tokenIn: tx.token_in,
                  tokenOut: tx.token_out,
                };
              });

            setRecentTransactions(prev => {
              const preserved = prev.filter(tx => tx.id.startsWith('mining-'));
              const all = [...preserved, ...transformedTransactions];
              const unique = all.filter((tx, i, s) => i === s.findIndex(t => t.id === tx.id));
              return unique.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
            });
            setTransactionError(null);
            console.log('📋 Dashboard: Background refetch complete -', transformedTransactions.length, 'transactions');
          }
        } catch (err) {
          console.warn('Background transaction refetch failed:', err);
        }
      }, 3000); // 3s debounce to batch rapid mining rewards
    };

    window.addEventListener('wallet-balance-updated', handleInstantTransaction);
    console.log('⚡ Dashboard: Listening for instant SSE transaction updates');

    return () => {
      window.removeEventListener('wallet-balance-updated', handleInstantTransaction);
      if (refetchTimer) clearTimeout(refetchTimer);
      console.log('🔇 Dashboard: Stopped listening for instant SSE transactions');
    };
  }, []);

  // Listen for loan approval events from SSE (via App.tsx custom event)
  useEffect(() => {
    const handleLoanApproval = (event: Event) => {
      const customEvent = event as CustomEvent;
      const loanData = customEvent.detail;

      console.log('🏦 Dashboard: Received loan-approved event:', loanData);

      // Store loan details and show approval modal
      setApprovedLoanDetails(loanData);
      setShowLoanApprovalModal(true);
    };

    window.addEventListener('loan-approved', handleLoanApproval);
    console.log('👂 Dashboard: Listening for loan-approved events');

    return () => {
      window.removeEventListener('loan-approved', handleLoanApproval);
      console.log('🔇 Dashboard: Stopped listening for loan-approved events');
    };
  }, []);

  // Refresh wallet balances when refreshTrigger changes
  useEffect(() => {
    if (refreshTrigger > 0) {
      console.log('🔄 Refreshing wallet balances...');
      const refresh = async () => {
        const currentWalletAddress = localStorage.getItem('walletAddress');
        if (!currentWalletAddress) return;

        // CRITICAL FIX: Use best known balance, not just nodeStatus
        const cachedBalance = localStorage.getItem('cachedBalance');
        const cachedValue = cachedBalance ? parseFloat(cachedBalance) : 0;
        const previousHighest = highestKnownBalancesRef.current['QUG'] || 0;
        const nodeBalance = nodeStatus?.balance || 0;
        // Use the maximum of all known sources
        const qugBalance = Math.max(previousHighest, cachedValue, nodeBalance);
        console.log('🔄 Refresh using best balance:', qugBalance, '(highest:', previousHighest, ', cached:', cachedValue, ', node:', nodeBalance, ')');

        const now = Date.now();
        // Preserve existing history instead of wiping it
        let savedHistory: Record<string, BalanceHistoryPoint[]> = {};
        try {
          const saved = localStorage.getItem('qnk_balance_long_v1') || localStorage.getItem('walletBalanceHistory');
          savedHistory = saved ? JSON.parse(saved) : {};
        } catch { savedHistory = {}; }

        const qugSaved = savedHistory['QUG'] || [];
        const qugLast = qugSaved[qugSaved.length - 1];
        // Only add point if meaningfully different
        const qugNeedNew = !qugLast || Math.abs(qugBalance - qugLast.balance) / Math.max(qugLast.balance, 0.001) > 0.005;
        const qugHistory = qugNeedNew
          ? [...qugSaved, { timestamp: now, balance: qugBalance }].slice(-10080)
          : (qugSaved.length >= 2 ? qugSaved : [{ timestamp: now - 60000, balance: qugBalance }, { timestamp: now, balance: qugBalance }]);

        const balances: WalletBalance[] = [
          {
            symbol: 'QUG',
            name: 'Quillon Graph',
            balance: qugBalance,
            icon: 'qug',
            color: 'from-amber-400 to-yellow-500',
            history: qugHistory
          }
        ];

        console.log('📊 Refresh QUG with history:', qugHistory.length, 'points (preserved:', qugSaved.length, ')');

        // Fetch USD balance
        try {
          const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/balance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ wallet_address: currentWalletAddress }),
          });

          if (response.ok) {
            const data = await response.json();
            if (data.success && data.data) {
              const usdValue = parseFloat(data.data.balance_usd || '0');
              setUsdBalance(usdValue);

              const usdSaved = savedHistory['USD'] || [];
              const usdLast = usdSaved[usdSaved.length - 1];
              const usdNeedNew = !usdLast || Math.abs(usdValue - usdLast.balance) > 0.01;
              const usdHistory = usdNeedNew
                ? [...usdSaved, { timestamp: now, balance: usdValue }].slice(-10080)
                : (usdSaved.length >= 2 ? usdSaved : [{ timestamp: now - 60000, balance: usdValue }, { timestamp: now, balance: usdValue }]);

              balances.push({
                symbol: 'USD',
                name: 'US Dollar',
                balance: usdValue,
                icon: 'usd',
                color: 'from-green-400 to-emerald-500',
                history: usdHistory
              });

              console.log('📊 Refresh USD with history:', usdHistory.length, 'points (preserved:', usdSaved.length, ')');
            }
          }
        } catch (error) {
          // Silently fail if payment API is not available
          console.warn('⚠️ Payment API not available - USD wallet features disabled');
        }

        // v6.5.1: Fetch QUGUSD balance - trust backend, no anti-zero override
        let qugusdBalance = 0;

        try {
          const response = await qnkAPI.getMultiTokenBalance();
          if (response.success && response.data && response.data.tokens) {
            const tokensObj = response.data.tokens;

            if (tokensObj.QUGUSD && tokensObj.QUGUSD.balance_base_units > 0) {
              qugusdBalance = tokensObj.QUGUSD.balance_base_units / 1e24;
            } else if (tokensObj.qugusd && tokensObj.qugusd.balance !== undefined) {
              qugusdBalance = parseFloat(tokensObj.qugusd.balance) || 0;
            }

            if (qugusdBalance > 0) {
              localStorage.setItem('cachedQugusdBalance', qugusdBalance.toString());
              localStorage.setItem('lastKnownQugusdBalance', qugusdBalance.toString());
              if (qugusdBalance > (highestKnownBalancesRef.current['QUGUSD'] || 0)) {
                highestKnownBalancesRef.current['QUGUSD'] = qugusdBalance;
              }
            } else {
              localStorage.removeItem('cachedQugusdBalance');
              highestKnownBalancesRef.current['QUGUSD'] = 0;
            }
          }
        } catch (error) {
          // On fetch failure, fall back to cache
          const cachedQugusd = localStorage.getItem('cachedQugusdBalance');
          qugusdBalance = cachedQugusd ? parseFloat(cachedQugusd) || 0 : 0;
          console.warn('⚠️ Failed to fetch QUGUSD in refresh, using cached:', qugusdBalance);
        }

        // Always add QUGUSD if we have a balance (cached or fetched)
        if (qugusdBalance > 0) {
          const qugusdSaved = savedHistory['QUGUSD'] || [];
          const qugusdLast = qugusdSaved[qugusdSaved.length - 1];
          const qugusdNeedNew = !qugusdLast || Math.abs(qugusdBalance - qugusdLast.balance) > 0.01;
          const qugusdHistory = qugusdNeedNew
            ? [...qugusdSaved, { timestamp: now, balance: qugusdBalance }].slice(-10080)
            : (qugusdSaved.length >= 2 ? qugusdSaved : [{ timestamp: now - 60000, balance: qugusdBalance }, { timestamp: now, balance: qugusdBalance }]);

          balances.push({
            symbol: 'QUGUSD',
            name: 'Quillon USD',
            balance: qugusdBalance,
            usdValue: qugusdBalance,
            icon: 'usd' as const,
            color: 'from-blue-400 to-cyan-500',
            history: qugusdHistory
          });
          console.log('📊 Refresh QUGUSD with history:', qugusdBalance, '(preserved:', qugusdSaved.length, 'points)');
        }

        // v8.5.9: Fetch QUSD balance from same multi-token response
        let qusdRefreshBalance = 0;
        try {
          const qusdResp = await qnkAPI.getMultiTokenBalance();
          if (qusdResp.success && qusdResp.data && qusdResp.data.tokens) {
            const t = qusdResp.data.tokens;
            if (t.QUSD && t.QUSD.balance !== undefined) {
              qusdRefreshBalance = parseFloat(t.QUSD.balance) || 0;
            } else if (t.qusd && t.qusd.balance !== undefined) {
              qusdRefreshBalance = parseFloat(t.qusd.balance) || 0;
            }
          }
        } catch (_e) { /* ignore */ }

        if (qusdRefreshBalance > 0) {
          const qusdSaved = savedHistory['QUSD'] || [];
          const qusdLast = qusdSaved[qusdSaved.length - 1];
          const qusdNeedNew = !qusdLast || Math.abs(qusdRefreshBalance - qusdLast.balance) > 0.01;
          const qusdHistory = qusdNeedNew
            ? [...qusdSaved, { timestamp: now, balance: qusdRefreshBalance }].slice(-10080)
            : (qusdSaved.length >= 2 ? qusdSaved : [{ timestamp: now - 60000, balance: qusdRefreshBalance }, { timestamp: now, balance: qusdRefreshBalance }]);

          balances.push({
            symbol: 'QUSD',
            name: 'Quillon USD',
            balance: qusdRefreshBalance,
            usdValue: qusdRefreshBalance,
            icon: 'usd' as const,
            color: 'from-green-400 to-emerald-500',
            history: qusdHistory
          });
          console.log('📊 Refresh QUSD with history:', qusdRefreshBalance, '(preserved:', qusdSaved.length, 'points)');
        }

        // Bridge wallets (with empty history to prevent "Loading..." display)
        balances.push(
          {
            symbol: 'ZEC',
            name: 'Zcash (Shielded)',
            balance: zecBalance,
            icon: 'zec',
            color: 'from-purple-400 to-indigo-600',
            shieldedOnly: true,
            history: [],
          },
          {
            symbol: 'IRON',
            name: 'Iron Fish',
            balance: 0,
            icon: 'iron',
            color: 'from-cyan-400 to-slate-500',
            shieldedOnly: true,
            history: [],
          },
          {
            symbol: 'BTC',
            name: 'Bitcoin',
            balance: btcBalance,
            icon: 'btc',
            color: 'from-orange-400 to-amber-500',
            history: [],
          },
          {
            symbol: 'ETH',
            name: 'Ethereum',
            balance: ethBalance,
            icon: 'eth',
            color: 'from-blue-400 to-indigo-500',
            history: [],
          },
        );

        // v2.3.31-beta: Check BOTH local ref AND global localStorage cooldown
        const refreshGlobalCooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');
        const refreshGlobalCooldownActive = Date.now() < refreshGlobalCooldownUntil;
        if (dexSwapCooldownRef.current || refreshGlobalCooldownActive) {
          console.log('🚫 [refreshTrigger] SKIPPING setWalletBalances during DEX cooldown (global:', refreshGlobalCooldownActive, ')');
          return; // Don't overwrite the correct DEX-updated balance
        }

        setWalletBalances(balances);

        // v8.1.6: Removed balance-update dispatch to App.tsx (causes zigzag).
        // App.tsx SSE handles balance updates directly.
      };

      refresh();
    }
  }, [refreshTrigger]); // Removed nodeStatus?.balance - SSE handles balance updates

  // v10.2.0: Consistent decimal formatting — fixed decimals per tier prevents flickering
  const formatBalance = (amount: number, hidden = false) => {
    if (hidden) return '••••••••';
    const abs = Math.abs(amount);
    if (abs >= 1000) return new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(amount);
    if (abs >= 1) return new Intl.NumberFormat('en-US', { minimumFractionDigits: 4, maximumFractionDigits: 4 }).format(amount);
    if (abs >= 0.0001) return new Intl.NumberFormat('en-US', { minimumFractionDigits: 6, maximumFractionDigits: 6 }).format(amount);
    return new Intl.NumberFormat('en-US', { minimumFractionDigits: 8, maximumFractionDigits: 8 }).format(amount);
  };

  // Smart filtering and sorting logic
  const filteredAndSortedTransactions = (() => {
    let filtered = [...recentTransactions];

    // Apply type filter
    if (filterType !== 'all') {
      filtered = filtered.filter(tx => tx.type === filterType);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;

      if (sortBy === 'date') {
        comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      } else if (sortBy === 'amount') {
        comparison = a.amount - b.amount;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  })();

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedTransactions.length / itemsPerPage);
  const paginatedTransactions = filteredAndSortedTransactions.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Reset to first page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [filterType, sortBy, sortOrder]);

  const copyWalletAddress = () => {
    navigator.clipboard.writeText(walletAddress);
    setCopiedAddress(true);
    setTimeout(() => setCopiedAddress(false), 2000);
  };

  // v7.0.0: Faucet removed — all QUG earned through mining

  // Handle Loan Payback
  const handleLoanPayback = (loanId: string) => {
    console.log('💰 Opening loan payback modal for loan:', loanId);
    setSelectedLoanId(loanId);
    setShowLoanPaybackModal(true);
  };

  // Handle Add USD - show Stripe checkout
  const handleAddUSD = async () => {
    if (!usdAmount || parseFloat(usdAmount) <= 0) {
      setStripeError('Please enter a valid amount');
      return;
    }

    const currentWalletAddress = localStorage.getItem('walletAddress');
    if (!currentWalletAddress) {
      setStripeError('No wallet address found');
      return;
    }

    // Show the Stripe checkout component
    setShowStripeCheckout(true);
  };

  // Handle Send USD
  const handleSendUSD = async () => {
    if (!usdAmount || parseFloat(usdAmount) <= 0) {
      setStripeError('Please enter a valid amount');
      return;
    }

    if (!usdRecipient) {
      setStripeError('Please enter a recipient wallet address');
      return;
    }

    const currentWalletAddress = localStorage.getItem('walletAddress');
    if (!currentWalletAddress) {
      setStripeError('No wallet address found');
      return;
    }

    setStripeLoading(true);
    setStripeError(null);

    try {
      // This would call a USD transfer API endpoint (to be implemented)
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/transfer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          from_wallet: currentWalletAddress,
          to_wallet: usdRecipient,
          amount_usd: usdAmount,
        }),
      });

      const data = await response.json();

      if (data.success) {
        console.log('✅ USD sent successfully');
        alert(`Successfully sent $${usdAmount} USD to ${usdRecipient}`);

        // Close modal and refresh balance
        setIsSendUSDModalOpen(false);
        setUsdAmount('');
        setUsdRecipient('');
        setRefreshTrigger(prev => prev + 1);
      } else {
        setStripeError(data.error || 'Failed to send USD');
      }
    } catch (error) {
      console.error('❌ Failed to send USD:', error);
      setStripeError('Network error: Could not send USD');
    } finally {
      setStripeLoading(false);
    }
  };

  console.log('Dashboard rendering, loading:', loading, 'error:', error, 'nodeStatus:', nodeStatus);

  if (loading) {
    return (
      <div className="relative w-full rounded-3xl overflow-hidden" style={{ height: 'calc(100vh - 200px)', minHeight: 480 }}>
        <QuantumLoader
          message="Initializing Quantum Dashboard"
          subMessage="Fetching node telemetry and wallet state..."
          inline
        />
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-8">
        <div className="bg-quantum-pink/20 border border-quantum-pink/50 rounded-3xl p-8 text-center">
          <AlertCircle className="w-12 h-12 text-quantum-pink mx-auto mb-4" />
          <h2 className="text-xl font-bold text-quantum-pink mb-2">Node Connection Error</h2>
          <p className="text-gray-400 mb-4">{error}</p>
          <p className="text-sm text-gray-500">
            Please ensure the Q-NarwhalKnight node is running and accessible at the configured API endpoint.
          </p>
        </div>
      </div>
    );
  }

  if (!nodeStatus) {
    return (
      <div className="relative w-full rounded-3xl overflow-hidden" style={{ height: 'calc(100vh - 200px)', minHeight: 480 }}>
        <QuantumLoader
          message="Initializing Quantum Dashboard"
          subMessage="Fetching node telemetry and wallet state..."
          inline
        />
      </div>
    );
  }

  return (
    <div className="space-y-8 relative">
      {/* Persistent quantum particle background at 10% opacity */}
      <div className="fixed inset-0 z-0 pointer-events-none" style={{ opacity: 0.1 }}>
        <QuantumLoader backgroundOnly />
      </div>

      {/* Mobile Setup QR Modal */}
      {showMobileSetup && (
        <MobileSetupModal onClose={() => setShowMobileSetup(false)} />
      )}
      {/* Phase Transition Modal (legacy) */}
      {showPhaseModal && (
        <PhaseTransitionModal
          onClose={() => {
            setShowPhaseModal(false);
            localStorage.setItem('v0978betaModalSeen', 'true');
          }}
        />
      )}

      {/* QNO Staking Modal */}
      <StakingModal
        isOpen={showStakingModal}
        onClose={() => setShowStakingModal(false)}
        availableBalance={walletBalances.find(w => w.symbol === 'QUG')?.balance || 0}
        walletAddress={walletAddress}
        onStakeSuccess={() => {
          setRefreshTrigger(prev => prev + 1);
        }}
      />


      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* CYBERPUNK TAB NAVIGATION                                       */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <div className="mt-6 mb-4">
        <div
          className="relative rounded-2xl p-1 backdrop-blur-xl overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, rgba(15, 10, 35, 0.8), rgba(20, 15, 40, 0.8))',
            border: '1px solid rgba(34, 211, 238, 0.2)',
            boxShadow: '0 0 20px rgba(34, 211, 238, 0.08), inset 0 0 15px rgba(34, 211, 238, 0.03)'
          }}
        >
          {/* Tab bar with reorder gear */}
          <div className="relative flex gap-1 z-10">
            {(() => {
              const tabDefs: Record<string, { label: string; Icon: any; comingSoon?: boolean }> = {
                wallet: { label: 'WALLET', Icon: Wallet },
                search: { label: 'SEARCH', Icon: Globe },
                mail: { label: 'MAIL', Icon: Mail },
                calendar: { label: 'CALENDAR', Icon: Calendar },
                chat: { label: 'CHAT', Icon: MessageCircle },
              };
              return tabOrder.map((tabId) => {
                const tab = tabDefs[tabId];
                if (!tab) return null;
                const isActive = activeDashboardTab === tabId;
                const isMail = tabId === 'mail';
                const isChat = tabId === 'chat';
                return (
                  <motion.button
                    key={tabId}
                    onClick={() => {
                      if (tab.comingSoon) return;
                      if (tabId === 'chat' && onNavigateToChat) {
                        setChatUnreadCount(0);
                        onNavigateToChat();
                      } else {
                        setActiveDashboardTab(tabId);
                      }
                    }}
                    disabled={tab.comingSoon}
                    className={`
                      flex-1 py-3.5 px-4 rounded-xl font-semibold uppercase tracking-widest
                      transition-all duration-300 relative overflow-hidden
                      text-xs lg:text-sm flex items-center justify-center gap-2
                      ${isActive
                        ? 'text-white'
                        : tab.comingSoon
                        ? 'text-gray-600 cursor-not-allowed'
                        : 'text-cyan-400/70 hover:text-cyan-200 cursor-pointer'
                      }
                    `}
                    whileHover={!tab.comingSoon ? { scale: 1.02 } : {}}
                    whileTap={!tab.comingSoon ? { scale: 0.97 } : {}}
                    style={{
                      background: isActive
                        ? 'linear-gradient(135deg, rgba(34, 211, 238, 0.2), rgba(147, 51, 234, 0.12))'
                        : 'transparent',
                      borderBottom: isActive
                        ? '2px solid rgba(34, 211, 238, 0.7)'
                        : '2px solid transparent',
                    }}
                  >
                    {isActive && (
                      <motion.div
                        className="absolute inset-0 -z-10"
                        style={{ background: 'radial-gradient(circle, rgba(34, 211, 238, 0.15), transparent 70%)' }}
                        animate={{ opacity: [0.3, 0.5, 0.3] }}
                        transition={{ duration: 3, repeat: Infinity }}
                      />
                    )}
                    <div className="relative">
                      <tab.Icon className="w-4 h-4" />
                      {/* Unread email notification badge */}
                      {isChat && chatUnreadCount > 0 && (
                        <div className="absolute -top-2.5 -right-3 pointer-events-none">
                          <motion.div
                            className="absolute inset-0 rounded-full"
                            style={{
                              width: 20, height: 20,
                              background: 'radial-gradient(circle, rgba(212,175,55,0.5), transparent 70%)',
                              filter: 'blur(3px)',
                              transform: 'translate(-3px, -3px)',
                            }}
                            animate={{ scale: [1, 1.8, 1], opacity: [0.7, 0, 0.7] }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                          />
                          <motion.div
                            className="relative flex items-center justify-center rounded-full"
                            style={{
                              minWidth: 16, height: 16,
                              padding: '0 4px',
                              background: 'linear-gradient(135deg, #D4AF37, #FFD700, #D4AF37)',
                              boxShadow: '0 0 8px rgba(212,175,55,0.8), 0 0 16px rgba(212,175,55,0.4)',
                              border: '1.5px solid rgba(255,235,150,0.5)',
                            }}
                            animate={{ scale: [1, 1.12, 1] }}
                            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                          >
                            <span className="text-[9px] font-black text-slate-900 leading-none">
                              {chatUnreadCount > 99 ? '99+' : chatUnreadCount}
                            </span>
                          </motion.div>
                        </div>
                      )}
                      {isMail && unreadEmailCount > 0 && (
                        <div className="absolute -top-2.5 -right-3 pointer-events-none">
                          {/* Outer pulsing ring */}
                          <motion.div
                            className="absolute inset-0 rounded-full"
                            style={{
                              width: 20, height: 20,
                              background: 'radial-gradient(circle, rgba(255, 60, 120, 0.5), transparent 70%)',
                              filter: 'blur(3px)',
                              transform: 'translate(-3px, -3px)',
                            }}
                            animate={{ scale: [1, 1.8, 1], opacity: [0.7, 0, 0.7] }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                          />
                          {/* Second pulse ring offset */}
                          <motion.div
                            className="absolute inset-0 rounded-full"
                            style={{
                              width: 18, height: 18,
                              border: '1px solid rgba(255, 100, 150, 0.6)',
                              transform: 'translate(-2px, -2px)',
                            }}
                            animate={{ scale: [1, 2.2, 1], opacity: [0.5, 0, 0.5] }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut', delay: 0.5 }}
                          />
                          {/* Badge core */}
                          <motion.div
                            className="relative flex items-center justify-center rounded-full"
                            style={{
                              minWidth: 16, height: 16,
                              padding: '0 4px',
                              background: 'linear-gradient(135deg, #FF3C78, #FF6B9D, #E91E8C)',
                              boxShadow: '0 0 8px rgba(255, 60, 120, 0.8), 0 0 16px rgba(255, 60, 120, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.3)',
                              border: '1.5px solid rgba(255, 150, 200, 0.5)',
                            }}
                            animate={{ scale: [1, 1.12, 1] }}
                            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                          >
                            <span className="text-[9px] font-black text-white leading-none" style={{ textShadow: '0 1px 2px rgba(0,0,0,0.4)' }}>
                              {unreadEmailCount > 99 ? '99+' : unreadEmailCount}
                            </span>
                          </motion.div>
                        </div>
                      )}
                    </div>
                    <span>{tab.label}</span>
                    {tab.comingSoon && (
                      <span className="text-[9px] bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded-full border border-amber-500/30 ml-1">
                        SOON
                      </span>
                    )}
                  </motion.button>
                );
              });
            })()}

            {/* Tab order settings gear */}
            <motion.button
              onClick={() => setShowTabSettings(!showTabSettings)}
              className="flex items-center justify-center px-2 rounded-xl transition-all"
              whileHover={{ scale: 1.1, rotate: 30 }}
              whileTap={{ scale: 0.9 }}
              style={{ color: showTabSettings ? '#22D3EE' : 'rgba(34, 211, 238, 0.35)' }}
              title="Reorder tabs"
            >
              <Settings2 className="w-3.5 h-3.5" />
            </motion.button>
          </div>

          {/* Tab reorder dropdown */}
          <AnimatePresence>
            {showTabSettings && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div
                  className="mx-2 mb-2 mt-1 rounded-xl p-3"
                  style={{
                    background: 'linear-gradient(135deg, rgba(10, 5, 30, 0.9), rgba(15, 10, 35, 0.9))',
                    border: '1px solid rgba(34, 211, 238, 0.15)',
                  }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <GripVertical className="w-3 h-3 text-cyan-400/50" />
                    <span className="text-[10px] font-bold uppercase tracking-widest text-cyan-400/60">Tab Order</span>
                  </div>
                  <div className="space-y-1">
                    {tabOrder.map((tabId, idx) => {
                      const labels: Record<string, string> = { wallet: 'Wallet', search: 'Search', mail: 'Mail', calendar: 'Calendar', chat: 'Chat' };
                      const Icons: Record<string, any> = { wallet: Wallet, search: Globe, mail: Mail, calendar: Calendar, chat: MessageCircle };
                      const TabIcon = Icons[tabId];
                      return (
                        <div
                          key={tabId}
                          className="flex items-center gap-2 px-2 py-1.5 rounded-lg"
                          style={{
                            background: activeDashboardTab === tabId
                              ? 'rgba(34, 211, 238, 0.08)'
                              : 'transparent',
                          }}
                        >
                          <span className="text-[10px] font-mono text-cyan-400/40 w-3">{idx + 1}</span>
                          <TabIcon className="w-3.5 h-3.5 text-cyan-300/60" />
                          <span className="text-xs text-gray-300 flex-1">{labels[tabId]}</span>
                          <motion.button
                            whileHover={{ scale: 1.2 }}
                            whileTap={{ scale: 0.8 }}
                            onClick={() => moveTab(tabId, 'up')}
                            disabled={idx === 0}
                            className="p-0.5 rounded disabled:opacity-20"
                            style={{ color: 'rgba(34, 211, 238, 0.6)' }}
                          >
                            <ArrowUp className="w-3 h-3" />
                          </motion.button>
                          <motion.button
                            whileHover={{ scale: 1.2 }}
                            whileTap={{ scale: 0.8 }}
                            onClick={() => moveTab(tabId, 'down')}
                            disabled={idx === tabOrder.length - 1}
                            className="p-0.5 rounded disabled:opacity-20"
                            style={{ color: 'rgba(34, 211, 238, 0.6)' }}
                          >
                            <ArrowDown className="w-3 h-3" />
                          </motion.button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        {/* Decorative scan line */}
        <div className="h-px mt-2 opacity-20" style={{ background: 'linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.5), transparent)' }} />
      </div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* TAB CONTENT                                                    */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <AnimatePresence mode="wait">
        {activeDashboardTab === 'mail' && (
          <motion.div key="mail-tab" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.25 }}>
            <EmailScreen />
          </motion.div>
        )}
        {activeDashboardTab === 'calendar' && (
          <motion.div key="calendar-tab" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.25 }}
            style={{ position: 'relative', minHeight: 600 }}>
            <CalendarScreen />
          </motion.div>
        )}
        {activeDashboardTab === 'search' && (
          <motion.div key="search-tab" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.25 }}>
            <WebSearchScreen />
          </motion.div>
        )}
        {/* Chat tab navigates to App-level ChatScreen via onNavigateToChat (avoids duplicate SignalingService peer_id conflict) */}
      </AnimatePresence>

      {activeDashboardTab === 'wallet' && <>

      {/* ── HiBT Listing Donation Banner */}
      <HiBTDonationBanner />

      {/* ── News & Blog Row ─────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-6"
      >
        <div
          className="flex items-center justify-between mb-4 cursor-pointer select-none"
          onClick={() => setNewsCollapsed(v => !v)}
        >
          <div className="flex items-center gap-2">
            <Newspaper className="w-4 h-4 text-yellow-400/70" />
            <span className="text-sm font-semibold text-gray-300 uppercase tracking-widest">News & Updates</span>
            <motion.span
              animate={{ rotate: newsCollapsed ? -90 : 0 }}
              transition={{ type: 'spring', stiffness: 380, damping: 28 }}
              className="text-gray-600 ml-0.5"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </motion.span>
          </div>
          <a
            href="https://quillon.xyz"
            target="_blank"
            rel="noopener noreferrer"
            onClick={e => e.stopPropagation()}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-yellow-400 transition-colors"
          >
            All posts <ExternalLink className="w-3 h-3" />
          </a>
        </div>
        <AnimatePresence initial={false}>
        {!newsCollapsed && (
        <motion.div
          key="news-cards"
          initial={{ opacity: 0, height: 0, marginTop: 0 }}
          animate={{ opacity: 1, height: 'auto', marginTop: 0 }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.32, ease: [0.4, 0, 0.2, 1] }}
          style={{ overflow: 'hidden' }}
        >
          {(() => {
            const posts = [
              {
                tag: 'Announcement', tagColor: 'text-orange-400', tagBg: 'rgba(251,146,60,0.1)', tagBorder: 'rgba(251,146,60,0.25)',
                icon: <Globe className="w-3.5 h-3.5 text-orange-400" />,
                title: 'Community Campaign: Help Get QUG Listed on HiBT Exchange',
                excerpt: 'The Quillon community is rallying to raise $15,000 toward a QUG/USDT listing on HiBT — a top-40 global exchange with 3.5 million registered users and $7.5 billion in daily spot volume.',
                date: 'May 2026', accent: 'rgba(251,146,60,0.08)', border: 'rgba(251,146,60,0.2)',
                fullContent: `## The Opportunity

HiBT is a globally ranked cryptocurrency exchange (CoinMarketCap #38 Spot, #36 Futures) with over 3.5 million registered users and $7.5 billion in daily spot volume. A QUG/USDT listing on HiBT would give Quillon its first major centralized exchange presence, connecting QUG to traders across North America, Europe, Asia-Pacific, the Middle East, and Africa.

## What the Listing Includes

The Standard listing package covers a full marketing launch:

- **QUG/USDT trading pair** on a regulated platform with real order books
- **Listing announcement** pushed to HiBT's 9 million+ registered user base
- **Banner placements** across the HiBT website and mobile app
- **Multilingual Telegram promotions** across 7 languages, 50,000+ community members
- **Social media blast** across Twitter, LinkedIn, Discord, and Instagram (200,000+ reach)
- **PR articles** written and distributed by HiBT's editorial team
- **Weekly Top Gainer** and quarterly newsletter promotions

## Why QUG Belongs on a CEX

Quillon Graph is not a token on someone else's chain — it is a native Layer 1 blockchain with a live mainnet, real miners, and a built-in DEX. A centralized exchange listing gives miners a straightforward way to realize value from their work and opens the door to institutional-scale liquidity. HiBT's Asia-Pacific and Middle East penetration is particularly valuable for QUG given the strong mining communities in those regions.

Key fundamentals that support a CEX listing right now:

- Quantum-resistant cryptography from genesis (Dilithium5 + Kyber1024) — no retrofit required
- DAG-Knight consensus with sub-3-second finality
- CPU + GPU dual-lane mining — fair to home hardware and professional farms
- Built-in AMM decentralized exchange and WASM smart contracts
- Live mainnet with multiple 10Gbit bootstrap nodes across multiple continents

## The $15,000 Goal — Community Funded

The listing fee is $15,000 USDT. No team treasury funds are being used — this is a community-funded campaign. Every satoshi donated goes directly toward securing the listing. The QUG team has committed to full transparency: all on-chain donations are tracked live and the progress is visible directly in the dashboard.

## How to Contribute

Click the banner at the top of the dashboard to open the donation modal. Send any amount of BTC (on-chain, Bitcoin mainnet) to the listed address. The progress bar updates in real time as contributions arrive. Lightning Network support is being added shortly for smaller, instant contributions.

There is no minimum. Even 0.001 BTC moves the needle. If the goal is not reached, the approach and timeline will be reassessed openly with the community — no funds will be spent without a completed raise.

## What Happens After the Listing

Once live, QUG becomes available to millions of users who have never heard of Quillon. The marketing package activates automatically: announcements, banners, social posts. The built-in DEX remains the primary venue for on-chain swaps, but a CEX listing brings liquidity depth that the DEX alone cannot replicate at this stage.

This is the first step toward broader exchange coverage. Help make it happen.`,
              },
              {
                tag: 'Research', tagColor: 'text-purple-400', tagBg: 'rgba(139,92,246,0.12)', tagBorder: 'rgba(139,92,246,0.25)',
                icon: <Shield className="w-3.5 h-3.5 text-purple-400" />,
                title: 'David and Goliath: QNK Built in 30 Days What Took Monero a Decade',
                excerpt: 'A full technical comparison of cryptographic stacks, mining fairness, and privacy primitives — and why starting from first principles in 2026 changes every assumption.',
                date: 'March 2026', accent: 'rgba(139,92,246,0.1)', border: 'rgba(139,92,246,0.2)',
                fullContent: `## Executive Summary

Monero has spent over a decade hardening its privacy stack. Quillon Network shipped an equivalent — and in several respects superior — cryptographic foundation in 30 days of intensive development. This post examines why that is possible in 2026 and what it means for the ecosystem.

## Cryptographic Stack Comparison

### Ring Signatures
Monero uses CLSAG (Concise Linkable Spontaneous Anonymous Group) signatures with a fixed ring size of 16. Quillon ships CLSAG with configurable ring sizes from 8 to 128, and lays the groundwork for Groth16-based membership proofs that collapse the ring into a single O(1) ZK proof.

### Stealth Addresses
Both protocols implement dual-key stealth addresses (DKSAP). Quillon adds a view-tag byte for O(1) scanning — a technique Monero adopted only in 2022 after years of community pressure. Our implementation ships this on day one.

### Bulletproofs
Monero v0.15 introduced Bulletproofs for range proofs; v0.18 upgraded to Bulletproofs+. Quillon implements Bulletproofs+ natively, skipping three years of migration cost.

### Post-Quantum Layer
Monero has no post-quantum roadmap. Quillon has a four-phase migration plan:
- Phase 0 (live): Ed25519 + BLAKE3
- Phase 1 (in progress): Dilithium5 hybrid signatures
- Phase 2: Kyber1024 key encapsulation
- Phase 3: Fully post-quantum consensus

## Mining Fairness

Monero's RandomX is excellent CPU-friendly PoW, but GPU mining still commands a marginal edge on some configurations. Quillon's Genus 2 upgrade enforces a hard 50/50 split: half the block reward goes to BLAKE3 GPU miners, half to VDF CPU miners. These are separate consensus lanes — GPU farms literally cannot crowd out home hardware.

## DAG Consensus vs Linear Chain

Monero processes transactions in strict linear order, limiting throughput to ~1,700 TPS under ideal conditions. Quillon uses DAG-Knight, a directed acyclic graph BFT protocol with Narwhal mempool. Concurrent vertices are validated in parallel, targeting 48,000+ TPS with sub-3-second finality.

## Conclusion

Starting from first principles in 2026 means inheriting a decade of lessons without inheriting a decade of technical debt. Quillon is not a Monero fork — it is what Monero would build if it started today.`,
              },
              {
                tag: 'Announcement', tagColor: 'text-green-400', tagBg: 'rgba(34,197,94,0.1)', tagBorder: 'rgba(34,197,94,0.25)',
                icon: <Cpu className="w-3.5 h-3.5 text-green-400" />,
                title: 'CPU Mining Returns: VDF Dual-Lane Now Live',
                excerpt: 'The Genus 2 upgrade splits every block reward 50/50 between BLAKE3 GPU miners and VDF CPU miners. Home hardware has a guaranteed lane that GPU farms cannot crowd out.',
                date: 'April 2026', accent: 'rgba(34,197,94,0.08)', border: 'rgba(34,197,94,0.18)',
                fullContent: `## The Problem with Single-Algorithm Mining

Every major proof-of-work coin faces the same trajectory: hobbyist CPU miners → GPU farms → ASIC monopolies. Even RandomX, designed to be CPU-friendly, attracts GPU optimisations that incrementally price out home miners.

## The Genus 2 Solution: Two Lanes, One Block

Starting with block 500,000, every QNK block reward is split into two equal halves:

**Lane A — BLAKE3 GPU Mining (50%)**
Standard high-throughput GPU mining. Fast, competitive, and profitable for professional operations. This keeps QNK attractive to large miners who provide network security.

**Lane B — VDF CPU Mining (50%)**
Verifiable Delay Functions cannot be parallelised. A VDF requires sequential computation — more GPUs do not help. A single modern CPU core competes on equal footing with a rack of GPUs in this lane.

## How VDF Mining Works

1. Each block contains a VDF challenge derived from the previous block hash.
2. CPU miners compute the VDF output sequentially — this takes approximately 10 seconds on a modern core.
3. The first miner to submit a valid VDF proof claims the CPU lane reward.
4. The GPU lane operates in parallel and settles independently.

Both lanes must be satisfied for a block to be considered fully valid. This creates a cooperative dynamic between GPU and CPU miners rather than a competitive one.

## Hardware Requirements

**GPU Lane:** Any GPU with ≥4GB VRAM. RTX 3060 achieves ~420 MH/s.

**CPU Lane:** Any x86-64 CPU released after 2018. A Raspberry Pi 4 is too slow; a laptop i5 is competitive. The key insight: you don't need more cores, you need a fast single-core clock speed.

## Download the CPU Miner

\`\`\`bash
wget https://quillon.xyz/downloads/q-miner-v10.5.3
chmod +x q-miner-v10.5.3
./q-miner-v10.5.3 --mode vdf --wallet YOUR_ADDRESS
\`\`\`

The miner auto-detects your hardware and selects the optimal lane. Run both simultaneously on the same machine for maximum rewards.`,
              },
              {
                tag: 'Update', tagColor: 'text-blue-400', tagBg: 'rgba(59,130,246,0.1)', tagBorder: 'rgba(59,130,246,0.25)',
                icon: <Layers className="w-3.5 h-3.5 text-blue-400" />,
                title: 'Community Roadmap: Sync Speed, Emission Resilience & P2P Hardening',
                excerpt: 'Full post-mortem and three-phase hardening plan covering emission state recovery, turbo-sync reliability, and bootstrap redundancy.',
                date: 'Apr 10, 2026', accent: 'rgba(59,130,246,0.08)', border: 'rgba(59,130,246,0.18)',
                fullContent: `## Post-Mortem: April 4–9 Incident

On April 4 a cascade of issues exposed three independent weaknesses in the network stack. No funds were lost and chain continuity was maintained, but sync reliability and emission accuracy degraded for approximately 48 hours on secondary nodes.

This post documents exactly what happened and our three-phase response.

## What Happened

**Issue 1: Emission Controller Drift**
The emission controller on Epsilon (primary bootstrap) was computing rewards using a stale genesis timestamp after an unclean restart. Blocks produced during this window had slightly incorrect coinbase values — still valid under consensus rules, but inconsistent with the long-term emission schedule.

**Issue 2: Turbo-Sync Gap Detection**
A kill -9 on Epsilon left 180 corrupt block entries in RocksDB. The gap-detection algorithm in turbo-sync counted these corrupt entries as "present", so the sync engine never requested the missing blocks from peers. Nodes syncing from Epsilon accumulated silent gaps.

**Issue 3: Bootstrap Redundancy**
With Epsilon degraded, new nodes had no reliable bootstrap path. The fallback to Beta and Gamma worked, but at 100× lower throughput — new node sync time went from ~6 hours to ~3 days.

## Three-Phase Hardening Plan

### Phase 1 — Emission Resilience (Complete)
- Emission genesis timestamp now persisted to a separate sled column family and cross-checked on every restart.
- Coinbase validation added to block acceptance: any block with a coinbase outside ±0.1% of the expected emission rate is rejected.
- Automated emission audit runs every 1,000 blocks and emits a structured log entry.

### Phase 2 — Sync Hardening (In Progress)
- Corrupt block detection during gap scan: entries that deserialise to an error are treated as absent, not present.
- Post-sync verification pass: after turbo-sync completes, a random 1% sample of blocks is re-verified against stored hashes.
- Sync-down protection at the database layer: any attempt to replace a block at height H with data from a shorter chain is rejected with an explicit error.

### Phase 3 — Bootstrap Redundancy (Planned Q2 2026)
- Four independent bootstrap nodes with automatic health scoring.
- New nodes receive a ranked peer list sorted by sync speed, not just peer availability.
- Introducing Delta (5.79.79.158) as a permanently funded 1Gbit bootstrap node.

## Timeline

| Date | Milestone |
|------|-----------|
| Apr 10 | Phase 1 emission fixes deployed to all nodes |
| Apr 15 | Phase 2 sync hardening in testing |
| Apr 22 | Phase 2 deployed to mainnet |
| May 15 | Phase 3 bootstrap redundancy complete |

We thank the community members who reported degraded sync speeds and helped us reproduce the corrupt block scenario.`,
              },
            ];
            const [featured, ...rest] = posts;
            return (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Featured card — spans 2 cols */}
                <motion.div
                  key="featured"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  whileHover={{ y: -3 }}
                  className="md:col-span-2 rounded-2xl overflow-hidden cursor-pointer flex flex-col"
                  style={{ background: 'linear-gradient(135deg, rgba(12,8,28,0.97) 0%, rgba(20,10,40,0.97) 100%)', border: `1.5px solid ${featured.border}`, boxShadow: `0 4px 32px ${featured.accent}` }}
                  onClick={() => setSelectedArticle(featured)}
                >
                  {/* Gradient header strip */}
                  <div className="h-1.5 w-full" style={{ background: `linear-gradient(90deg, rgba(139,92,246,0.7) 0%, rgba(167,139,250,0.3) 100%)` }} />
                  <div className="p-5 flex flex-col gap-3 flex-1">
                    <div className="flex items-center justify-between">
                      <span className={`flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full ${featured.tagColor}`} style={{ background: featured.tagBg, border: `1px solid ${featured.tagBorder}` }}>
                        {featured.icon}{featured.tag}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-[9px] font-semibold uppercase tracking-widest px-1.5 py-0.5 rounded-full text-amber-400 bg-amber-400/10 border border-amber-400/25">Featured</span>
                        <span className="text-[10px] text-gray-600">{featured.date}</span>
                      </div>
                    </div>
                    <h3 className="text-sm font-bold text-gray-100 leading-snug">{featured.title}</h3>
                    <p className="text-[11px] text-gray-400 leading-relaxed flex-1">{featured.excerpt}</p>
                    <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                      <span className="text-[10px] text-gray-600 uppercase tracking-widest">Quillon Network</span>
                      <div className="flex-1" />
                      <motion.span whileHover={{ x: 2 }} className={`text-[10px] font-semibold ${featured.tagColor} flex items-center gap-1`}>
                        Read full article <ChevronRight className="w-3 h-3" />
                      </motion.span>
                    </div>
                  </div>
                </motion.div>

                {/* Side cards — stacked */}
                <div className="flex flex-col gap-4">
                  {rest.map((post, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.15 + i * 0.07 }}
                      whileHover={{ y: -2 }}
                      className="rounded-2xl overflow-hidden cursor-pointer flex flex-col flex-1"
                      style={{ background: 'linear-gradient(135deg, rgba(12,10,22,0.97) 0%, rgba(18,14,32,0.97) 100%)', border: `1.5px solid ${post.border}`, boxShadow: `0 2px 16px ${post.accent}` }}
                      onClick={() => setSelectedArticle(post)}
                    >
                      <div className="h-1" style={{ background: `linear-gradient(90deg, ${post.tagBorder} 0%, transparent 100%)` }} />
                      <div className="p-4 flex flex-col gap-2 flex-1">
                        <div className="flex items-center justify-between">
                          <span className={`flex items-center gap-1 text-[9px] font-semibold uppercase tracking-widest px-1.5 py-0.5 rounded-full ${post.tagColor}`} style={{ background: post.tagBg, border: `1px solid ${post.tagBorder}` }}>
                            {post.icon}{post.tag}
                          </span>
                          {i === 0 && (
                            <span className="text-[9px] font-semibold uppercase tracking-widest px-1.5 py-0.5 rounded-full text-emerald-400 bg-emerald-400/10 border border-emerald-400/25">New</span>
                          )}
                        </div>
                        <h3 className="text-[11px] font-bold text-gray-200 leading-snug">{post.title}</h3>
                        <p className="text-[10px] text-gray-500 leading-relaxed flex-1 line-clamp-2">{post.excerpt}</p>
                        <div className="flex items-center justify-between pt-1.5" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                          <span className="text-[9px] text-gray-700 uppercase tracking-widest">{post.date}</span>
                          <motion.span whileHover={{ x: 2 }} className={`text-[9px] font-medium ${post.tagColor} flex items-center gap-0.5`}>
                            Read <ChevronRight className="w-2.5 h-2.5" />
                          </motion.span>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            );
          })()}
        </motion.div>
        )}
        </AnimatePresence>
      </motion.div>
      {/* ── /News & Blog Row ─────────────────────────────────────────── */}

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-6 items-start">
      <div className="flex flex-col gap-6">
      {/* Multi-Wallet Card */}
      <motion.div
        className="backdrop-blur-xl rounded-3xl p-6 relative overflow-hidden"
        style={{
          background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.9) 0%, rgba(25, 25, 40, 0.9) 100%)',
          border: '2px solid rgba(212, 175, 55, 0.3)',
          boxShadow: '0 0 30px rgba(212, 175, 55, 0.2), inset 0 0 20px rgba(212, 175, 55, 0.1)'
        }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="relative space-y-6">
          {/* Wallet Address Section */}
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div
                className="p-3 rounded-xl"
                style={{
                  background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                  border: '2px solid rgba(212, 175, 55, 0.3)'
                }}
              >
                <Wallet className="w-6 h-6 text-amber-400" />
              </div>
              <div>
                <h3 className="text-lg font-bold bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent mb-2">
                  Wallet Address
                </h3>
                <div className="font-mono text-xs text-amber-100 break-all max-w-md">
                  {walletAddress || 'Generating...'}
                </div>
              </div>
            </div>

            <div className="flex gap-2 flex-shrink-0">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsQRModalOpen(true)}
                disabled={!walletAddress}
                className="p-3 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                  border: '2px solid rgba(212, 175, 55, 0.3)'
                }}
                title="Show QR Code"
              >
                <QrCode className="w-5 h-5 text-amber-400" />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsNodeInfoModalOpen(true)}
                className="p-3 rounded-xl transition-colors"
                style={{
                  background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.15))',
                  border: '2px solid rgba(59, 130, 246, 0.3)'
                }}
                title="Node Information"
              >
                <Info className="w-5 h-5 text-blue-400" />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={generateAIReport}
                className="p-3 rounded-xl transition-colors group relative"
                style={{
                  background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(147, 51, 234, 0.15))',
                  border: '2px solid rgba(168, 85, 247, 0.3)'
                }}
                title="AI Wallet Analysis"
              >
                <img
                  src="/quantum-ai-logo.svg"
                  alt="AI Report"
                  className="w-5 h-5 object-contain"
                  style={{
                    filter: 'drop-shadow(0 0 8px rgba(168, 85, 247, 0.5))'
                  }}
                />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowFinanceModal(true)}
                className="p-3 rounded-xl transition-colors group relative"
                style={{
                  background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(20, 184, 166, 0.15))',
                  border: '2px solid rgba(6, 182, 212, 0.3)'
                }}
                title="K-Law Financial Intelligence"
              >
                <BarChart3 className="w-5 h-5 text-cyan-400" style={{ filter: 'drop-shadow(0 0 8px rgba(6, 182, 212, 0.5))' }} />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={copyWalletAddress}
                disabled={!walletAddress}
                className="p-3 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                  border: '2px solid rgba(212, 175, 55, 0.3)'
                }}
                title="Copy Address"
              >
                {copiedAddress ? (
                  <Check className="w-5 h-5 text-green-400" />
                ) : (
                  <Copy className="w-5 h-5 text-amber-400" />
                )}
              </motion.button>
            </div>
          </div>

          {/* Multi-Wallet Balances */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold text-white">My Wallets</h3>
              <div className="flex items-center gap-2">
                {/* Apply for Loan Button */}
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => {
                    console.log('🏦 Apply for Loan button clicked - opening modal');
                    setShowLoanModal(true);
                  }}
                  className="px-4 py-2 rounded-xl transition-colors text-sm font-medium flex items-center gap-2"
                  style={{
                    background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                    border: '2px solid rgba(212, 175, 55, 0.3)',
                    color: 'rgb(251, 191, 36)'
                  }}
                >
                  <DollarSign className="w-4 h-4" />
                  Apply for Loan
                </motion.button>

              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {walletBalances.map((wallet, index) => {
                // v10.2.0: DEX lock is handled by event handlers (qug-balance-changed, dex-cooldown-expired)
                // which update walletBalances state directly. WalletCardWithGraph has its own
                // debounced stable balance to prevent decimal flickering. No render-time override needed.
                return (
                  <WalletCardWithGraph
                    key={wallet.symbol}
                    wallet={wallet}
                    index={index}
                    isAnimating={balanceAnimations[wallet.symbol] || false}
                    onCardClick={
                      wallet.symbol === 'BTC' ? () => setShowBitcoinSwapModal(true) :
                      wallet.symbol === 'ZEC' ? () => setShowZcashWalletModal(true) :
                      wallet.symbol === 'IRON' ? () => setShowIronFishWalletModal(true) :
                      wallet.symbol === 'ETH' ? () => setShowEthereumSwapModal(true) :
                      !wallet.comingSoon && wallet.symbol !== 'USD' && onNavigateToSend ? () => onNavigateToSend(wallet.symbol) : undefined
                    }
                  >
                    {/* USD Action Buttons */}
                    {!wallet.comingSoon && wallet.symbol === 'USD' && (
                      <div className="flex gap-2">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setIsAddUSDModalOpen(true);
                            setStripeError(null);
                          }}
                          className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-green-500/20 border border-green-500/30 text-green-300 flex items-center justify-center gap-1"
                          title="Add USD"
                        >
                          <Plus className="w-3 h-3" />
                          Add
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onNavigateToSend) {
                              onNavigateToSend('USD');
                            }
                          }}
                          className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-blue-500/20 border border-blue-500/30 text-blue-300 flex items-center justify-center gap-1"
                          title="Send USD"
                        >
                          <Send className="w-3 h-3" />
                          Send
                        </motion.button>
                      </div>
                    )}

                    {/* Send and Stake Buttons for QUG wallet */}
                    {!wallet.comingSoon && wallet.symbol === 'QUG' && (
                      <div className="flex gap-2">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onNavigateToSend) {
                              onNavigateToSend(wallet.symbol);
                            }
                          }}
                          className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-blue-500/20 border border-blue-500/30 text-blue-300 flex items-center justify-center gap-1"
                          title="Send"
                        >
                          <Send className="w-3 h-3" />
                          Send
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowStakingModal(true);
                          }}
                          className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-purple-500/20 border border-purple-500/30 text-purple-300 flex items-center justify-center gap-1"
                          title="Stake for QNO Predictions"
                        >
                          <Zap className="w-3 h-3" />
                          Stake
                        </motion.button>
                      </div>
                    )}

                    {/* Send Button for QUGUSD wallet */}
                    {!wallet.comingSoon && wallet.symbol === 'QUGUSD' && (
                      <div className="flex gap-2">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onNavigateToSend) {
                              onNavigateToSend(wallet.symbol);
                            }
                          }}
                          className="flex-1 py-2 px-3 rounded-lg text-xs font-medium bg-blue-500/20 border border-blue-500/30 text-blue-300 flex items-center justify-center gap-1"
                          title="Send"
                        >
                          <Send className="w-3 h-3" />
                          Send
                        </motion.button>
                      </div>
                    )}
                  </WalletCardWithGraph>
                );
              })}
            </div>
          </div>

          {transactionError && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3 rounded-xl text-sm bg-red-500/20 text-red-400 border border-red-500/30"
            >
              <div className="font-semibold mb-1">⚠️ Failed to load transaction history</div>
              <div className="text-xs">{transactionError}</div>
              <div className="text-xs mt-2 opacity-80">
                💡 Tip: This usually means you need to log in with your wallet password. Go to Settings → Login/Import Wallet.
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>

      {/* Emission Trajectory — actual vs target supply curve */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.32 }}
      >
        <EmissionCurveViz />
      </motion.div>

      {/* Mining Decentralization — Lorenz curve & Gini coefficient */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.34 }}
      >
        <MiningDecentralizationViz />
      </motion.div>

      {/* Active Loans Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        <ActiveLoansCard onPayback={handleLoanPayback} />
      </motion.div>

      {/* Custom Tokens Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.37 }}
      >
        <CustomTokensCard onSendToken={(symbol, contractAddress) => {
          // When user clicks send on a custom token, navigate to send screen
          // You can enhance this to pass the contract address as well
          if (onNavigateToSend) {
            // Store the contract address in localStorage for the send screen to use
            localStorage.setItem('selectedTokenContract', contractAddress);
            onNavigateToSend(symbol);
          }
        }} />
      </motion.div>

      {/* DAG-Knight Consensus Visualization */}
      <motion.div
        className="mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <DAGKnightVisualization currentHeight={nodeStatus?.current_height || 0} />
      </motion.div>

      {/* QNO Oracle Resolution Visualization */}
      <motion.div
        className="mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="backdrop-blur-xl rounded-3xl overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.9) 0%, rgba(25, 25, 40, 0.9) 100%)',
            border: '2px solid rgba(139, 92, 246, 0.3)',
            boxShadow: '0 0 30px rgba(139, 92, 246, 0.1)'
          }}
        >
          <div className="p-4 border-b border-purple-500/20">
            <h3 className="text-lg font-semibold text-purple-100 flex items-center gap-2">
              <span className="text-xl">🔮</span>
              QNO Oracle Resolution Monitor
            </h3>
            <p className="text-sm text-purple-300/60 mt-1">
              Real-time prediction staking outcomes, oracle feeds, and resolution events
            </p>
          </div>
          <QNOOracleVisualization />
        </div>
      </motion.div>

      </div>{/* /left-column */}
      {/* ── RIGHT COLUMN: Sticky live transaction history ── */}
      <div className="flex flex-col">
        <RecentActivityPanel
          transactions={recentTransactions as any}
          sseConnected={sseConnected}
          onSelectTransaction={(tx) => { setSelectedTransaction(tx as any); setIsModalOpen(true); }}
          currentHeight={nodeStatus?.current_height}
        />
      </div>{/* /right-column */}
      </div>{/* /two-column-grid */}

      {/* Transaction Details Modal */}
      <TransactionDetailsModal
        transaction={selectedTransaction}
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          setSelectedTransaction(null);
        }}
      />

      {/* QR Code Modal */}
      <QRCodeModal
        isOpen={isQRModalOpen}
        onClose={() => setIsQRModalOpen(false)}
        walletAddress={walletAddress}
        balance={nodeStatus?.balance || 0}
      />

      {/* AI Report Modal */}
      <AnimatePresence>
        {isAIReportModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 overflow-y-auto"
            onClick={() => setIsAIReportModalOpen(false)}
          >
            <div className="flex min-h-full items-center justify-center p-4">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="rounded-2xl p-6 max-w-3xl w-full max-h-[90vh] overflow-y-auto shadow-2xl"
              style={{
                background: 'linear-gradient(135deg, rgba(20, 20, 30, 0.98), rgba(30, 30, 45, 0.98))',
                border: '2px solid rgba(168, 85, 247, 0.3)',
                boxShadow: '0 0 40px rgba(168, 85, 247, 0.3)'
              }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl">
                    <img
                      src="/quantum-ai-logo.svg"
                      alt="Quantum AI"
                      className="w-8 h-8 object-contain"
                      style={{
                        filter: 'drop-shadow(0 0 10px rgba(168, 85, 247, 0.6))'
                      }}
                    />
                  </div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">
                    AI Wallet Analysis
                  </h2>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setIsAIReportModalOpen(false)}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </motion.button>
              </div>

              {/* AI Report Content */}
              <div className="space-y-4">
                {aiReportLoading && !aiReport && (
                  <div className="flex items-center justify-center py-12">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="w-16 h-16"
                    >
                      <img
                        src="/quantum-ai-logo.svg"
                        alt="Loading"
                        className="w-full h-full object-contain"
                        style={{
                          filter: 'drop-shadow(0 0 20px rgba(168, 85, 247, 0.8))'
                        }}
                      />
                    </motion.div>
                  </div>
                )}

                {aiReport && (
                  <div className="p-6 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-2 border-purple-500/20">
                    <div className="prose prose-invert max-w-none">
                      <div className="text-gray-200 whitespace-pre-wrap leading-relaxed">
                        {aiReport}
                      </div>
                    </div>

                    {aiReportLoading && (
                      <motion.div
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        className="mt-4 text-purple-400 text-sm flex items-center gap-2"
                      >
                        <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                        Generating analysis...
                      </motion.div>
                    )}
                  </div>
                )}

                {!aiReportLoading && !aiReport && (
                  <div className="text-center py-8 text-gray-400">
                    Click "Generate Report" to analyze your wallet and mining performance.
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="mt-6 flex gap-3 justify-end">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setIsAIReportModalOpen(false)}
                  className="px-6 py-3 rounded-xl transition-colors"
                  style={{
                    background: 'linear-gradient(135deg, rgba(100, 100, 120, 0.2), rgba(80, 80, 100, 0.15))',
                    border: '2px solid rgba(100, 100, 120, 0.3)'
                  }}
                >
                  Close
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={generateAIReport}
                  disabled={aiReportLoading}
                  className="px-6 py-3 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(147, 51, 234, 0.2))',
                    border: '2px solid rgba(168, 85, 247, 0.4)'
                  }}
                >
                  {aiReportLoading ? 'Generating...' : 'Regenerate Report'}
                </motion.button>
              </div>
            </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Node Info Modal */}
      <AnimatePresence>
        {isNodeInfoModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 overflow-y-auto"
            onClick={() => setIsNodeInfoModalOpen(false)}
          >
            <div className="flex min-h-full items-center justify-center p-4">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="rounded-2xl p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl"
              style={{
                background: 'linear-gradient(135deg, rgba(20, 20, 30, 0.98), rgba(30, 30, 45, 0.98))',
                border: '2px solid rgba(59, 130, 246, 0.3)',
                boxShadow: '0 0 40px rgba(59, 130, 246, 0.2)'
              }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border-2 border-blue-500/30">
                    <Info className="w-6 h-6 text-blue-400" />
                  </div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-400">
                    Node Information
                  </h2>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setIsNodeInfoModalOpen(false)}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </motion.button>
              </div>

              {/* Performance Metrics */}
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-2 border-purple-500/20">
                  <h3 className="text-lg font-semibold text-purple-300 mb-3 flex items-center gap-2">
                    <Zap className="w-5 h-5" />
                    Performance Metrics
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Current TPS</div>
                      <div className="text-xl font-bold text-purple-300">
                        {nodeStatus?.tps_current?.toLocaleString() || '0'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Average TPS</div>
                      <div className="text-xl font-bold text-purple-300">
                        {nodeStatus?.tps_average?.toLocaleString() || '0'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg col-span-2">
                      <div className="text-xs text-gray-400 mb-1">Max Theoretical TPS</div>
                      <div className="text-xl font-bold text-purple-300">
                        {nodeStatus?.performance?.max_theoretical_tps?.toLocaleString() || '0'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg col-span-2">
                      <div className="text-xs text-gray-400 mb-1">Optimization Level</div>
                      <div className="text-sm font-semibold text-purple-300">
                        {nodeStatus?.performance?.optimization_level || 'Unknown'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Consensus Information */}
                <div className="p-4 rounded-xl bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-2 border-green-500/20">
                  <h3 className="text-lg font-semibold text-green-300 mb-3 flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    Consensus Status
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Status</div>
                      <div className="text-sm font-bold text-green-300 capitalize">
                        {nodeStatus?.consensus_status || 'unknown'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Validator</div>
                      <div className="text-sm font-bold text-green-300">
                        {nodeStatus?.is_validator ? 'Yes' : 'No'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Round</div>
                      <div className="text-sm font-bold text-green-300">
                        {nodeStatus?.current_round || 0}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Height</div>
                      <div className="text-sm font-bold text-green-300">
                        {nodeStatus?.current_height || 0}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Network Health */}
                <div className="p-4 rounded-xl bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-2 border-blue-500/20">
                  <h3 className="text-lg font-semibold text-blue-300 mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Network Health
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Uptime</div>
                      <div className="text-sm font-bold text-blue-300">
                        {nodeStatus?.uptime_formatted || '0h 0m 0s'}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">TX Pool Size</div>
                      <div className="text-sm font-bold text-blue-300">
                        {nodeStatus?.tx_pool_size || 0}
                      </div>
                    </div>
                    <div className="bg-black/20 p-3 rounded-lg col-span-2">
                      <div className="text-xs text-gray-400 mb-1">Features Enabled</div>
                      <div className="text-xs font-semibold text-blue-300 flex gap-2 flex-wrap mt-1">
                        {nodeStatus?.performance?.simd_crypto_enabled && (
                          <span className="px-2 py-1 bg-green-500/20 border border-green-500/30 rounded">
                            SIMD Crypto
                          </span>
                        )}
                        {nodeStatus?.performance?.kernel_io_enabled && (
                          <span className="px-2 py-1 bg-green-500/20 border border-green-500/30 rounded">
                            Kernel I/O
                          </span>
                        )}
                        {!nodeStatus?.performance?.simd_crypto_enabled && !nodeStatus?.performance?.kernel_io_enabled && (
                          <span className="px-2 py-1 bg-gray-500/20 border border-gray-500/30 rounded">
                            None
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Close Button */}
              <div className="mt-6 flex justify-end">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setIsNodeInfoModalOpen(false)}
                  className="px-6 py-3 rounded-xl font-semibold transition-all"
                  style={{
                    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.15))',
                    border: '2px solid rgba(59, 130, 246, 0.3)',
                    color: 'rgb(96, 165, 250)'
                  }}
                >
                  Close
                </motion.button>
              </div>
            </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Add USD Modal */}
      <AnimatePresence>
        {isAddUSDModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 overflow-y-auto"
            onClick={() => {
              setIsAddUSDModalOpen(false);
              setShowStripeCheckout(false);
              setStripeError(null);
              setUsdAmount('');
            }}
          >
            <div className="flex min-h-full items-center justify-center p-4">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="rounded-2xl p-6 max-w-md w-full max-h-[90vh] overflow-y-auto shadow-2xl"
              style={{
                background: 'linear-gradient(135deg, rgba(20, 30, 20, 0.98), rgba(30, 45, 30, 0.98))',
                border: '2px solid rgba(34, 197, 94, 0.3)',
                boxShadow: '0 0 40px rgba(34, 197, 94, 0.2)'
              }}
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl bg-gradient-to-br from-green-500/20 to-emerald-500/20 border-2 border-green-500/30">
                    <Plus className="w-6 h-6 text-green-400" />
                  </div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-emerald-400">
                    Add USD
                  </h2>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => {
                    setIsAddUSDModalOpen(false);
                    setShowStripeCheckout(false);
                    setStripeError(null);
                    setUsdAmount('');
                  }}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </motion.button>
              </div>

              {!showStripeCheckout ? (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Amount (USD)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0.50"
                      placeholder="0.00"
                      value={usdAmount}
                      onChange={(e) => setUsdAmount(e.target.value)}
                      className="w-full px-4 py-3 rounded-lg text-white placeholder-gray-500"
                      style={{
                        background: 'rgba(0, 0, 0, 0.3)',
                        border: '2px solid rgba(34, 197, 94, 0.2)'
                      }}
                    />
                    <p className="text-xs text-gray-400 mt-1">Minimum: $0.50</p>
                  </div>

                  {stripeError && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="p-3 rounded-lg text-sm bg-red-500/20 text-red-400 border border-red-500/30"
                    >
                      {stripeError}
                    </motion.div>
                  )}

                  <div className="flex gap-3 pt-4">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        setIsAddUSDModalOpen(false);
                        setShowStripeCheckout(false);
                        setStripeError(null);
                        setUsdAmount('');
                      }}
                      className="flex-1 px-6 py-3 rounded-xl font-semibold transition-all"
                      style={{
                        background: 'rgba(107, 114, 128, 0.2)',
                        border: '2px solid rgba(107, 114, 128, 0.3)',
                        color: 'rgb(156, 163, 175)'
                      }}
                    >
                      Cancel
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={handleAddUSD}
                      disabled={stripeLoading || !usdAmount || parseFloat(usdAmount) <= 0}
                      className="flex-1 px-6 py-3 rounded-xl font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{
                        background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(22, 163, 74, 0.2))',
                        border: '2px solid rgba(34, 197, 94, 0.5)',
                        color: 'rgb(74, 222, 128)'
                      }}
                    >
                      {stripeLoading ? 'Processing...' : 'Continue to Payment'}
                    </motion.button>
                  </div>

                  <p className="text-xs text-gray-500 text-center mt-4">
                    You will be redirected to Stripe to complete the payment securely.
                  </p>
                </div>
              ) : (
                <Suspense fallback={<div style={{ display: 'flex', justifyContent: 'center', padding: 32 }}><div style={{ width: 32, height: 32, borderRadius: '50%', border: '3px solid rgba(212,175,55,0.2)', borderTopColor: '#d4af37', animation: 'spin 0.8s linear infinite' }} /></div>}>
                  <StripeCheckout
                    amount={usdAmount}
                    walletAddress={localStorage.getItem('walletAddress') || ''}
                    onSuccess={() => {
                      setShowStripeCheckout(false);
                      setIsAddUSDModalOpen(false);
                      setUsdAmount('');
                      setStripeError(null);
                      setRefreshTrigger(prev => prev + 1);
                    }}
                    onCancel={() => {
                      setShowStripeCheckout(false);
                    }}
                  />
                </Suspense>
              )}
            </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Send USD Modal */}
      <AnimatePresence>
        {isSendUSDModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 overflow-y-auto"
            onClick={() => {
              setIsSendUSDModalOpen(false);
              setStripeError(null);
              setUsdAmount('');
              setUsdRecipient('');
            }}
          >
            <div className="flex min-h-full items-center justify-center p-4">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="rounded-2xl p-6 max-w-md w-full shadow-2xl"
              style={{
                background: 'linear-gradient(135deg, rgba(20, 20, 40, 0.98), rgba(30, 30, 60, 0.98))',
                border: '2px solid rgba(59, 130, 246, 0.3)',
                boxShadow: '0 0 40px rgba(59, 130, 246, 0.2)'
              }}
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border-2 border-blue-500/30">
                    <Send className="w-6 h-6 text-blue-400" />
                  </div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-400">
                    Send USD
                  </h2>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => {
                    setIsSendUSDModalOpen(false);
                    setStripeError(null);
                    setUsdAmount('');
                    setUsdRecipient('');
                  }}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </motion.button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Recipient Wallet Address</label>
                  <input
                    type="text"
                    placeholder="qnk..."
                    value={usdRecipient}
                    onChange={(e) => setUsdRecipient(e.target.value)}
                    className="w-full px-4 py-3 rounded-lg text-white placeholder-gray-500 font-mono text-sm"
                    style={{
                      background: 'rgba(0, 0, 0, 0.3)',
                      border: '2px solid rgba(59, 130, 246, 0.2)'
                    }}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Amount (USD)</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0.01"
                    placeholder="0.00"
                    value={usdAmount}
                    onChange={(e) => setUsdAmount(e.target.value)}
                    className="w-full px-4 py-3 rounded-lg text-white placeholder-gray-500"
                    style={{
                      background: 'rgba(0, 0, 0, 0.3)',
                      border: '2px solid rgba(59, 130, 246, 0.2)'
                    }}
                  />
                  <p className="text-xs text-gray-400 mt-1">Available: ${formatBalance(usdBalance)} USD</p>
                </div>

                {stripeError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-3 rounded-lg text-sm bg-red-500/20 text-red-400 border border-red-500/30"
                  >
                    {stripeError}
                  </motion.div>
                )}

                <div className="flex gap-3 pt-4">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => {
                      setIsSendUSDModalOpen(false);
                      setStripeError(null);
                      setUsdAmount('');
                      setUsdRecipient('');
                    }}
                    className="flex-1 px-6 py-3 rounded-xl font-semibold transition-all"
                    style={{
                      background: 'rgba(107, 114, 128, 0.2)',
                      border: '2px solid rgba(107, 114, 128, 0.3)',
                      color: 'rgb(156, 163, 175)'
                    }}
                  >
                    Cancel
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleSendUSD}
                    disabled={stripeLoading || !usdAmount || parseFloat(usdAmount) <= 0 || !usdRecipient}
                    className="flex-1 px-6 py-3 rounded-xl font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(37, 99, 235, 0.2))',
                      border: '2px solid rgba(59, 130, 246, 0.5)',
                      color: 'rgb(96, 165, 250)'
                    }}
                  >
                    {stripeLoading ? 'Sending...' : 'Send USD'}
                  </motion.button>
                </div>
              </div>
            </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      </>}

      {/* Loan Application Modal */}
      {showLoanModal && (
        <LoanApplicationModal
          onClose={() => setShowLoanModal(false)}
          walletBalances={walletBalances}
          walletAddress={walletAddress}
        />
      )}

      {/* Loan Approval Modal - Triggered by SSE loan-approved event */}
      {showLoanApprovalModal && approvedLoanDetails && (
        <LoanApprovalModal
          onClose={() => {
            setShowLoanApprovalModal(false);
            setApprovedLoanDetails(null);
          }}
          loanDetails={{
            amount: approvedLoanDetails.amount,
            interestRate: approvedLoanDetails.interestRate,
            termMonths: approvedLoanDetails.termMonths,
            monthlyPayment: approvedLoanDetails.monthlyPayment,
            collateralAmount: approvedLoanDetails.collateralAmount,
            collateralType: approvedLoanDetails.collateralType,
          }}
        />
      )}

      {/* Loan Payback Modal */}
      {showLoanPaybackModal && selectedLoanId && (
        <LoanPaybackModal
          loanId={selectedLoanId}
          onClose={() => {
            setShowLoanPaybackModal(false);
            setSelectedLoanId(null);
          }}
        />
      )}

      {/* K-Law Financial Intelligence Modal */}
      <FinanceModal
        isOpen={showFinanceModal}
        onClose={() => setShowFinanceModal(false)}
      />

      {/* Bitcoin Atomic Swap Modal */}
      {showBitcoinSwapModal && (
        <BitcoinSwapModal
          isOpen={showBitcoinSwapModal}
          onClose={() => setShowBitcoinSwapModal(false)}
          walletAddress={walletAddress}
        />
      )}

      {/* Zcash Shielded Wallet Modal */}
      {showZcashWalletModal && (
        <ZcashWalletModal
          isOpen={showZcashWalletModal}
          onClose={() => setShowZcashWalletModal(false)}
          walletAddress={walletAddress}
        />
      )}

      {/* Iron Fish Privacy Wallet Modal */}
      {showIronFishWalletModal && (
        <IronFishWalletModal
          isOpen={showIronFishWalletModal}
          onClose={() => setShowIronFishWalletModal(false)}
          walletAddress={walletAddress}
        />
      )}

      {/* Ethereum Atomic Swap Modal */}
      {showEthereumSwapModal && (
        <EthereumSwapModal
          isOpen={showEthereumSwapModal}
          onClose={() => setShowEthereumSwapModal(false)}
          walletAddress={walletAddress}
        />
      )}

      {/* ── Article Reader Modal ──────────────────────────────────────── */}
      <AnimatePresence>
        {selectedArticle && (
          <>
            {/* Backdrop */}
            <motion.div
              key="article-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="fixed inset-0 z-[90]"
              style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)' }}
              onClick={() => setSelectedArticle(null)}
            />

            {/* Modal panel — slides up from bottom */}
            <motion.div
              key="article-modal"
              initial={{ opacity: 0, y: '100%', scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: '80%', scale: 0.96 }}
              transition={{ type: 'spring', stiffness: 340, damping: 32, mass: 0.9 }}
              className="fixed inset-x-4 bottom-4 top-16 z-[91] flex flex-col rounded-3xl overflow-hidden md:inset-x-auto md:left-1/2 md:-translate-x-1/2 md:w-[720px] md:max-w-[calc(100vw-2rem)]"
              style={{
                background: 'linear-gradient(160deg, rgba(10,10,20,0.98) 0%, rgba(16,16,32,0.98) 100%)',
                border: `1.5px solid ${selectedArticle.border}`,
                boxShadow: `0 0 80px ${selectedArticle.accent}, 0 32px 80px rgba(0,0,0,0.7)`,
              }}
            >
              {/* Animated glow bar at top */}
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ delay: 0.18, duration: 0.6, ease: 'easeOut' }}
                className="h-[2px] w-full origin-left"
                style={{
                  filter: 'blur(1px)',
                  background: selectedArticle.tagColor.includes('purple')
                    ? 'linear-gradient(90deg, transparent, #a78bfa, transparent)'
                    : selectedArticle.tagColor.includes('green')
                    ? 'linear-gradient(90deg, transparent, #4ade80, transparent)'
                    : 'linear-gradient(90deg, transparent, #60a5fa, transparent)',
                }}
              />

              {/* Header */}
              <div className="flex items-start gap-3 px-6 pt-5 pb-4" style={{ borderBottom: `1px solid ${selectedArticle.border}` }}>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className={`flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest px-2.5 py-1 rounded-full ${selectedArticle.tagColor}`}
                      style={{ background: selectedArticle.tagBg, border: `1px solid ${selectedArticle.tagBorder}` }}
                    >
                      {selectedArticle.icon} {selectedArticle.tag}
                    </span>
                    <span className="text-[10px] text-gray-500">{selectedArticle.date}</span>
                  </div>
                  <h2 className="text-base font-bold text-gray-100 leading-snug">{selectedArticle.title}</h2>
                  <p className="text-xs text-gray-500 mt-1">Quillon Network</p>
                </div>
                <motion.button
                  whileHover={{ scale: 1.12, rotate: 90 }}
                  whileTap={{ scale: 0.9 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                  onClick={() => setSelectedArticle(null)}
                  className="shrink-0 w-8 h-8 flex items-center justify-center rounded-full text-gray-400 hover:text-gray-100 mt-0.5"
                  style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)' }}
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                    <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
                  </svg>
                </motion.button>
              </div>

              {/* Scrollable content */}
              <div className="flex-1 overflow-y-auto px-6 py-5" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.1) transparent' }}>
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.15, duration: 0.35 }}
                  className="prose prose-invert prose-sm max-w-none"
                  style={{ color: '#c1c1d0' }}
                >
                  {selectedArticle.fullContent.split('\n').map((line, li) => {
                    if (line.startsWith('## ')) return (
                      <h2 key={li} style={{ color: '#f0f0ff', fontSize: '1rem', fontWeight: 700, marginTop: '1.5rem', marginBottom: '0.6rem', borderBottom: `1px solid ${selectedArticle.border}`, paddingBottom: '0.4rem' }}>
                        {line.slice(3)}
                      </h2>
                    );
                    if (line.startsWith('### ')) return (
                      <h3 key={li} style={{ color: '#d0d0ef', fontSize: '0.8rem', fontWeight: 700, marginTop: '1.1rem', marginBottom: '0.3rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                        {line.slice(4)}
                      </h3>
                    );
                    if (line.startsWith('**') && line.endsWith('**')) return (
                      <p key={li} style={{ color: '#e8e8f8', fontWeight: 600, fontSize: '0.75rem', marginTop: '0.8rem', marginBottom: '0.2rem' }}>
                        {line.slice(2, -2)}
                      </p>
                    );
                    if (line.startsWith('- ')) return (
                      <div key={li} style={{ display: 'flex', gap: '0.6rem', marginBottom: '0.35rem' }}>
                        <span style={{ color: selectedArticle.tagColor.includes('purple') ? '#a78bfa' : selectedArticle.tagColor.includes('green') ? '#4ade80' : '#60a5fa', flexShrink: 0, marginTop: '0.15rem' }}>▸</span>
                        <span style={{ fontSize: '0.75rem', lineHeight: '1.6' }}>{line.slice(2)}</span>
                      </div>
                    );
                    if (line.startsWith('```')) return (
                      <div key={li} style={{ background: 'rgba(0,0,0,0.4)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: '0.1rem 0', margin: '0.6rem 0' }} />
                    );
                    if (line.startsWith('| ') && line.includes('|')) {
                      const cells = line.split('|').filter(c => c.trim());
                      const isHeader = li > 0 && selectedArticle.fullContent.split('\n')[li + 1]?.startsWith('|---');
                      if (line.startsWith('|---')) return null;
                      return (
                        <div key={li} style={{ display: 'grid', gridTemplateColumns: `repeat(${cells.length}, 1fr)`, gap: 1, marginBottom: 1 }}>
                          {cells.map((cell, ci) => (
                            <div key={ci} style={{ fontSize: '0.7rem', padding: '0.3rem 0.5rem', background: isHeader ? 'rgba(255,255,255,0.07)' : 'rgba(255,255,255,0.03)', borderRadius: 4, fontWeight: isHeader ? 600 : 400, color: isHeader ? '#e0e0f0' : '#a0a0b8' }}>
                              {cell.trim()}
                            </div>
                          ))}
                        </div>
                      );
                    }
                    if (line.startsWith('`') && line.endsWith('`') && !line.startsWith('```')) return (
                      <code key={li} style={{ display: 'block', fontFamily: 'monospace', fontSize: '0.7rem', background: 'rgba(0,0,0,0.35)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 6, padding: '0.5rem 0.75rem', margin: '0.2rem 0', color: '#a0e0ff' }}>
                        {line.slice(1, -1)}
                      </code>
                    );
                    if (!line.trim()) return <div key={li} style={{ height: '0.5rem' }} />;
                    return (
                      <p key={li} style={{ fontSize: '0.75rem', lineHeight: '1.75', marginBottom: '0.5rem' }}>
                        {line}
                      </p>
                    );
                  })}
                </motion.div>
              </div>

              {/* Footer */}
              <div className="px-6 py-3 flex items-center justify-between" style={{ borderTop: `1px solid ${selectedArticle.border}`, background: 'rgba(0,0,0,0.3)' }}>
                <span className="text-[10px] text-gray-600">quillon.xyz · {selectedArticle.date}</span>
                <motion.button
                  whileHover={{ scale: 1.04 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => setSelectedArticle(null)}
                  className={`text-[11px] font-semibold px-4 py-1.5 rounded-full ${selectedArticle.tagColor}`}
                  style={{ background: selectedArticle.tagBg, border: `1px solid ${selectedArticle.tagBorder}` }}
                >
                  Close
                </motion.button>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

    </div>
  );
});

export default Dashboard;
