// WalletCardWithGraph.tsx — v10.3.15
// Enhanced wallet card with:
//   • Long-term balance history (7 days, persisted in qnk_balance_long_v1)
//   • Time horizon tabs: 1H | 6H | 24H | 7D
//   • Canvas-based sparkline with gradient fill + hover crosshair
//   • "Wealth Velocity" radial gauge (Times-magazine quality second visual)

import { motion, AnimatePresence } from 'framer-motion';
import { Wallet, TrendingUp, TrendingDown, ChevronUp, BarChart2, Maximize2, X, Activity } from 'lucide-react';
import {
  memo, useMemo, useRef, useEffect, useState, useCallback
} from 'react';
import { createPortal } from 'react-dom';

// ──────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────

export interface BalanceHistoryPoint {
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

// ──────────────────────────────────────────────────────────────
// Time horizon config
// ──────────────────────────────────────────────────────────────

type Horizon = '1H' | '6H' | '24H' | '7D';
const HORIZONS: Horizon[] = ['1H', '6H', '24H', '7D'];
const HORIZON_MS: Record<Horizon, number> = {
  '1H':  3_600_000,
  '6H':  21_600_000,
  '24H': 86_400_000,
  '7D':  604_800_000,
};

// ──────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────

function fmtBalance(amount: number, symbol: string): string {
  if (symbol === 'QUGUSD' || symbol === 'USD') {
    return amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  if (amount >= 1000) return amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (amount >= 1) return amount.toLocaleString('en-US', { minimumFractionDigits: 4, maximumFractionDigits: 4 });
  if (amount >= 0.0001) return amount.toLocaleString('en-US', { minimumFractionDigits: 6, maximumFractionDigits: 6 });
  return amount.toLocaleString('en-US', { minimumFractionDigits: 8, maximumFractionDigits: 8 });
}

function filterByHorizon(data: BalanceHistoryPoint[], horizon: Horizon): BalanceHistoryPoint[] {
  const cutoff = Date.now() - HORIZON_MS[horizon];
  const filtered = data.filter((p) => p.timestamp >= cutoff);
  // Always include the oldest point just before the cutoff for a complete left edge
  if (filtered.length < data.length) {
    const before = data.filter((p) => p.timestamp < cutoff);
    if (before.length > 0) filtered.unshift(before[before.length - 1]);
  }
  return filtered;
}

// ──────────────────────────────────────────────────────────────
// Canvas Sparkline
// ──────────────────────────────────────────────────────────────

function drawSparkline(
  canvas: HTMLCanvasElement,
  data: BalanceHistoryPoint[],
  mouseX: number | null,
  positive: boolean
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 260;
  const H = canvas.clientHeight || 72;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  ctx.clearRect(0, 0, W, H);

  if (data.length < 2) {
    ctx.font = '10px sans-serif';
    ctx.fillStyle = 'rgba(148,163,184,0.3)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Accumulating data…', W / 2, H / 2);
    return;
  }

  const pad = { l: 2, r: 2, t: 8, b: 4 };
  const cW = W - pad.l - pad.r;
  const cH = H - pad.t - pad.b;

  const t0 = data[0].timestamp;
  const tN = data[data.length - 1].timestamp;
  const tRange = tN - t0 || 1;

  const vals = data.map((d) => d.balance);
  let minV = Math.min(...vals);
  let maxV = Math.max(...vals);
  const spread = maxV - minV;
  // Add 10% padding vertically; if flat line add tiny spread for visibility
  minV -= spread * 0.1 + (spread === 0 ? maxV * 0.01 : 0);
  maxV += spread * 0.1 + (spread === 0 ? maxV * 0.01 : 0);
  const vRange = maxV - minV || 1;

  const px = (d: BalanceHistoryPoint) => pad.l + ((d.timestamp - t0) / tRange) * cW;
  const py = (d: BalanceHistoryPoint) => pad.t + cH - ((d.balance - minV) / vRange) * cH;

  const lineColor = positive ? '#22c55e' : '#ef4444';
  const gradTop   = positive ? 'rgba(34,197,94,0.25)' : 'rgba(239,68,68,0.25)';
  const glowColor = positive ? '#86efac' : '#fca5a5';

  // Gradient fill
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + cH);
  grad.addColorStop(0, gradTop);
  grad.addColorStop(1, 'rgba(0,0,0,0)');

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    i === 0 ? ctx.moveTo(px(data[i]), py(data[i])) : ctx.lineTo(px(data[i]), py(data[i]));
  }
  const lp = data[data.length - 1];
  ctx.lineTo(px(lp), pad.t + cH);
  ctx.lineTo(pad.l, pad.t + cH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Main line
  ctx.shadowColor = glowColor;
  ctx.shadowBlur = 5;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    i === 0 ? ctx.moveTo(px(data[i]), py(data[i])) : ctx.lineTo(px(data[i]), py(data[i]));
  }
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Hover crosshair
  if (mouseX !== null && mouseX >= pad.l && mouseX <= pad.l + cW) {
    const tAtM = t0 + ((mouseX - pad.l) / cW) * tRange;
    let ni = 0, nd = Infinity;
    for (let i = 0; i < data.length; i++) {
      const d = Math.abs(data[i].timestamp - tAtM);
      if (d < nd) { nd = d; ni = i; }
    }
    const dp = data[ni];
    const dpX = px(dp), dpY = py(dp);

    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(dpX, pad.t); ctx.lineTo(dpX, pad.t + cH); ctx.stroke();
    ctx.setLineDash([]);

    ctx.shadowColor = glowColor; ctx.shadowBlur = 6;
    ctx.beginPath(); ctx.arc(dpX, dpY, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = lineColor; ctx.fill();
    ctx.shadowBlur = 0;

    // Tooltip
    const balStr = fmtBalance(dp.balance, '');
    const timeStr = new Date(dp.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const bW = 96, bH = 34;
    let bx = dpX + 6;
    if (bx + bW > W - pad.r) bx = dpX - bW - 6;
    const by = Math.max(pad.t, dpY - bH - 4);
    ctx.fillStyle = 'rgba(8,12,24,0.92)';
    ctx.strokeStyle = positive ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(bx, by, bW, bH, 4); ctx.fill(); ctx.stroke();
    ctx.font = '9px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(148,163,184,0.6)'; ctx.fillText(timeStr, bx + 6, by + 4);
    ctx.fillStyle = lineColor; ctx.fillText(balStr, bx + 6, by + 16);
  }
}

// ──────────────────────────────────────────────────────────────
// Wealth Velocity Gauge (radial, Times-magazine quality)
// ──────────────────────────────────────────────────────────────

function drawVelocityGauge(
  canvas: HTMLCanvasElement,
  data: BalanceHistoryPoint[],
  currentBalance: number
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 80;
  const H = canvas.clientHeight || 80;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H / 2;
  const R = Math.min(W, H) * 0.42;

  // Compute velocity: balance growth rate over the available window
  // Expressed as % change per day, clamped to [-100%, +200%]
  let velocityPct = 0;
  if (data.length >= 2) {
    const oldest = data[0];
    const newest = data[data.length - 1];
    const dtDays = (newest.timestamp - oldest.timestamp) / 86_400_000;
    if (dtDays > 0 && oldest.balance > 0) {
      velocityPct = ((newest.balance - oldest.balance) / oldest.balance) * (1 / dtDays) * 100;
    }
  }

  // All-time-high from the full stored history
  const ath = data.length > 0 ? Math.max(...data.map((d) => d.balance), currentBalance) : currentBalance;
  const athPct = ath > 0 ? Math.min(currentBalance / ath, 1) : 0;

  // ── Outer ATH arc (full circle, muted track + colored fill)
  const startAngle = -Math.PI * 0.75; // start bottom-left
  const sweepTotal = Math.PI * 1.5;   // 270° sweep

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, R, startAngle, startAngle + sweepTotal);
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = R * 0.18;
  ctx.lineCap = 'round';
  ctx.stroke();

  // ATH fill — green to gold gradient
  const athSweep = sweepTotal * athPct;
  if (athSweep > 0.05) {
    const athGrad = ctx.createLinearGradient(cx - R, cy, cx + R, cy);
    athGrad.addColorStop(0, 'rgba(34,197,94,0.7)');
    athGrad.addColorStop(1, 'rgba(212,175,55,0.9)');
    ctx.beginPath();
    ctx.arc(cx, cy, R, startAngle, startAngle + athSweep);
    ctx.shadowColor = athPct > 0.9 ? '#d4af37' : '#22c55e';
    ctx.shadowBlur = 6;
    ctx.strokeStyle = athGrad;
    ctx.lineWidth = R * 0.18;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // ── Inner circle background
  const innerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.7);
  innerGrad.addColorStop(0, 'rgba(10,16,36,0.97)');
  innerGrad.addColorStop(1, 'rgba(4,8,20,0.95)');
  ctx.beginPath(); ctx.arc(cx, cy, R * 0.72, 0, Math.PI * 2);
  ctx.fillStyle = innerGrad; ctx.fill();

  // ── Velocity needle (small, inside)
  const velClamped = Math.max(-1, Math.min(2, velocityPct / 100)); // -100% to +200%
  const needleAngle = startAngle + sweepTotal * ((velClamped + 1) / 3); // map [-1,2] → [0,1]
  const needleLen = R * 0.48;
  const needleX = cx + Math.cos(needleAngle) * needleLen;
  const needleY = cy + Math.sin(needleAngle) * needleLen;
  const needleColor = velocityPct >= 0 ? '#4ade80' : '#f87171';

  ctx.shadowColor = needleColor; ctx.shadowBlur = 4;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(needleX, needleY);
  ctx.strokeStyle = needleColor; ctx.lineWidth = 1.5; ctx.lineCap = 'round'; ctx.stroke();
  ctx.shadowBlur = 0;

  // Center dot
  ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
  ctx.fillStyle = needleColor; ctx.fill();

  // ── Center text: ATH%
  ctx.font = `bold ${Math.round(R * 0.38)}px sans-serif`;
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(`${Math.round(athPct * 100)}%`, cx, cy - R * 0.05);

  // Sub-label
  ctx.font = `${Math.round(R * 0.2)}px sans-serif`;
  ctx.fillStyle = 'rgba(148,163,184,0.45)';
  ctx.fillText('of ATH', cx, cy + R * 0.33);
}

// ──────────────────────────────────────────────────────────────
// Momentum Oscilloscope — Times Square worthy second graph
// Shows rate-of-change as glowing vertical bars from center axis
// ──────────────────────────────────────────────────────────────

function drawMomentumOscilloscope(
  canvas: HTMLCanvasElement,
  data: BalanceHistoryPoint[],
  scanOffset: number  // 0..1 animated scan line
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 600;
  const H = canvas.clientHeight || 200;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // Background — deep space dark
  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, 'rgba(2,6,18,1)');
  bg.addColorStop(1, 'rgba(4,10,28,1)');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (H / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  for (let i = 1; i < 8; i++) {
    const x = (W / 8) * i;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }

  const cy = H / 2;

  // Center axis
  ctx.strokeStyle = 'rgba(148,163,184,0.2)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
  ctx.setLineDash([]);

  if (data.length < 2) {
    // Flat line when no data
    ctx.strokeStyle = 'rgba(148,163,184,0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
    ctx.font = 'bold 13px monospace';
    ctx.fillStyle = 'rgba(148,163,184,0.4)';
    ctx.textAlign = 'center';
    ctx.fillText('Collecting balance data...', W / 2, cy - 14);
    return;
  }

  // Downsample to at most MAX_BARS points so individual bars remain visible
  const MAX_BARS = Math.max(60, Math.floor(W / 3));
  const step = Math.max(1, Math.floor((data.length - 1) / MAX_BARS));
  const sampled: typeof data = [];
  for (let i = 0; i < data.length; i += step) sampled.push(data[i]);
  if (sampled[sampled.length - 1] !== data[data.length - 1]) sampled.push(data[data.length - 1]);

  // Compute deltas (% change between consecutive sampled points)
  const deltas: { x: number; pct: number; ts: number }[] = [];
  for (let i = 1; i < sampled.length; i++) {
    const prev = sampled[i - 1];
    const cur = sampled[i];
    const pct = prev.balance > 0 ? ((cur.balance - prev.balance) / prev.balance) * 100 : 0;
    const xFrac = (i - 1) / (sampled.length - 2 || 1);
    deltas.push({ x: xFrac * W, pct, ts: cur.timestamp });
  }

  // Max absolute delta for scaling (use reduce to avoid spread stack overflow on large arrays)
  const maxAbs = deltas.reduce((m, d) => Math.max(m, Math.abs(d.pct)), 0.001);
  const barW = Math.max(2, (W / deltas.length) * 0.7);
  const halfH = cy * 0.82;

  deltas.forEach(({ x, pct }) => {
    const norm = Math.max(-1, Math.min(1, pct / maxAbs));
    const barH = Math.abs(norm) * halfH;
    const barY = norm >= 0 ? cy - barH : cy;
    const alpha = 0.4 + Math.abs(norm) * 0.6;

    // Glow
    if (Math.abs(norm) > 0.05) {
      const glowColor = norm >= 0 ? `rgba(74,222,128,${alpha * 0.35})` : `rgba(248,113,113,${alpha * 0.35})`;
      ctx.shadowColor = norm >= 0 ? '#4ade80' : '#f87171';
      ctx.shadowBlur = 8 + Math.abs(norm) * 12;
      ctx.fillStyle = glowColor;
      ctx.fillRect(x - barW / 2 - 3, barY - 2, barW + 6, barH + 4);
      ctx.shadowBlur = 0;
    }

    // Bar
    const grad = ctx.createLinearGradient(0, barY, 0, barY + (norm >= 0 ? -barH : barH));
    if (norm >= 0) {
      grad.addColorStop(0, `rgba(74,222,128,${alpha})`);
      grad.addColorStop(1, `rgba(16,185,129,${alpha * 0.6})`);
    } else {
      grad.addColorStop(0, `rgba(248,113,113,${alpha * 0.6})`);
      grad.addColorStop(1, `rgba(239,68,68,${alpha})`);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(x - barW / 2, norm >= 0 ? cy - barH : cy, barW, barH);
  });

  // Animated scan line
  const scanX = scanOffset * W;
  const scanGrad = ctx.createLinearGradient(scanX - 60, 0, scanX + 20, 0);
  scanGrad.addColorStop(0, 'rgba(212,175,55,0)');
  scanGrad.addColorStop(0.7, 'rgba(212,175,55,0.15)');
  scanGrad.addColorStop(1, 'rgba(212,175,55,0.6)');
  ctx.fillStyle = scanGrad;
  ctx.fillRect(scanX - 60, 0, 80, H);

  ctx.strokeStyle = 'rgba(212,175,55,0.8)';
  ctx.lineWidth = 1.5;
  ctx.shadowColor = '#d4af37';
  ctx.shadowBlur = 6;
  ctx.beginPath(); ctx.moveTo(scanX, 0); ctx.lineTo(scanX, H); ctx.stroke();
  ctx.shadowBlur = 0;

  // Labels
  ctx.font = `bold 10px monospace`;
  ctx.fillStyle = 'rgba(74,222,128,0.6)';
  ctx.textAlign = 'left';
  ctx.fillText(`+${maxAbs.toFixed(3)}%`, 6, 14);
  ctx.fillStyle = 'rgba(248,113,113,0.6)';
  ctx.fillText(`-${maxAbs.toFixed(3)}%`, 6, H - 5);
  ctx.fillStyle = 'rgba(148,163,184,0.3)';
  ctx.fillText('MOMENTUM OSCILLOSCOPE', W / 2 - 80, H - 5);
}

// ──────────────────────────────────────────────────────────────
// WalletCardWithGraph
// ──────────────────────────────────────────────────────────────

const WalletCardWithGraph = memo(function WalletCardWithGraph({
  wallet,
  index,
  isAnimating,
  onCardClick,
  children,
}: WalletCardProps) {
  const [stableBalance, setStableBalance] = useState(wallet.balance);
  const lastUpdateRef = useRef(Date.now());
  const stableRef = useRef(wallet.balance);
  const [horizon, setHorizon] = useState<Horizon>('24H');
  const [showGauge, setShowGauge] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalHorizon, setModalHorizon] = useState<Horizon>('24H');
  const [modalShowGauge, setModalShowGauge] = useState(false);
  const scanOffsetRef = useRef(0);

  const sparkRef = useRef<HTMLCanvasElement>(null);
  const gaugeRef = useRef<HTMLCanvasElement>(null);
  const modalSparkRef = useRef<HTMLCanvasElement>(null);
  const modalGaugeRef = useRef<HTMLCanvasElement>(null);
  const oscilloRef = useRef<HTMLCanvasElement>(null);
  const sparkAnimRef = useRef<number>(0);
  const gaugeAnimRef = useRef<number>(0);
  const modalAnimRef = useRef<number>(0);
  const oscilloAnimRef = useRef<number>(0);
  const mouseXRef = useRef<number | null>(null);
  const modalMouseXRef = useRef<number | null>(null);

  // Stable balance debounce
  useEffect(() => {
    const incoming = wallet.balance;
    const current = stableRef.current;
    const timeSince = Date.now() - lastUpdateRef.current;
    const delta = Math.abs(incoming - current);
    if (current === 0 && incoming > 0 || delta > 0.01 || (timeSince > 2000 && delta > 0.000001)) {
      stableRef.current = incoming;
      lastUpdateRef.current = Date.now();
      setStableBalance(incoming);
    }
  }, [wallet.balance]);

  // Filter data by horizon
  const filteredData = useMemo(() => {
    const raw = wallet.history ?? [];
    return filterByHorizon(raw, horizon);
  }, [wallet.history, horizon]);

  // Trend
  const { trend, pctChange } = useMemo(() => {
    if (filteredData.length < 2) return { trend: 0, pctChange: 0 };
    const first = filteredData[0].balance;
    const last = filteredData[filteredData.length - 1].balance;
    const pct = first > 0 ? ((last - first) / first) * 100 : 0;
    return { trend: last >= first ? 1 : -1, pctChange: pct };
  }, [filteredData]);

  const positive = trend >= 0;

  // Sparkline animation loop
  const drawSpark = useCallback(() => {
    if (sparkRef.current) {
      drawSparkline(sparkRef.current, filteredData, mouseXRef.current, positive);
    }
    sparkAnimRef.current = requestAnimationFrame(drawSpark);
  }, [filteredData, positive]);

  useEffect(() => {
    sparkAnimRef.current = requestAnimationFrame(drawSpark);
    return () => { if (sparkAnimRef.current) cancelAnimationFrame(sparkAnimRef.current); };
  }, [drawSpark]);

  // Gauge (static, redraws on data/balance change)
  useEffect(() => {
    if (showGauge && gaugeRef.current) {
      drawVelocityGauge(gaugeRef.current, wallet.history ?? [], stableBalance);
    }
  }, [showGauge, wallet.history, stableBalance]);

  // Modal filtered data
  const modalFilteredData = useMemo(() => {
    const raw = wallet.history ?? [];
    return filterByHorizon(raw, modalHorizon);
  }, [wallet.history, modalHorizon]);

  const { trend: modalTrend, pctChange: modalPctChange } = useMemo(() => {
    if (modalFilteredData.length < 2) return { trend: 0, pctChange: 0 };
    const first = modalFilteredData[0].balance;
    const last = modalFilteredData[modalFilteredData.length - 1].balance;
    const pct = first > 0 ? ((last - first) / first) * 100 : 0;
    return { trend: last >= first ? 1 : -1, pctChange: pct };
  }, [modalFilteredData]);
  const modalPositive = modalTrend >= 0;

  // Modal sparkline loop
  const drawModalSpark = useCallback(() => {
    if (modalSparkRef.current) {
      drawSparkline(modalSparkRef.current, modalFilteredData, modalMouseXRef.current, modalPositive);
    }
    modalAnimRef.current = requestAnimationFrame(drawModalSpark);
  }, [modalFilteredData, modalPositive]);

  // Oscilloscope loop
  const drawOscillo = useCallback(() => {
    scanOffsetRef.current = (scanOffsetRef.current + 0.002) % 1;
    if (oscilloRef.current) {
      drawMomentumOscilloscope(oscilloRef.current, wallet.history ?? [], scanOffsetRef.current);
    }
    oscilloAnimRef.current = requestAnimationFrame(drawOscillo);
  }, [wallet.history]);

  useEffect(() => {
    if (showModal) {
      modalAnimRef.current = requestAnimationFrame(drawModalSpark);
      oscilloAnimRef.current = requestAnimationFrame(drawOscillo);
      if (modalShowGauge && modalGaugeRef.current) {
        drawVelocityGauge(modalGaugeRef.current, wallet.history ?? [], stableBalance);
      }
    }
    return () => {
      if (modalAnimRef.current) cancelAnimationFrame(modalAnimRef.current);
      if (oscilloAnimRef.current) cancelAnimationFrame(oscilloAnimRef.current);
    };
  }, [showModal, drawModalSpark, drawOscillo, modalShowGauge, wallet.history, stableBalance]);

  useEffect(() => {
    if (!showModal) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setShowModal(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showModal]);

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    mouseXRef.current = e.clientX - e.currentTarget.getBoundingClientRect().left;
  }, []);
  const handleCanvasMouseLeave = useCallback(() => { mouseXRef.current = null; }, []);
  const handleModalMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    modalMouseXRef.current = e.clientX - e.currentTarget.getBoundingClientRect().left;
  }, []);
  const handleModalMouseLeave = useCallback(() => { modalMouseXRef.current = null; }, []);

  return (
    <motion.div
      key={wallet.symbol}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 * index }}
      className="p-4 rounded-xl cursor-pointer group relative overflow-hidden select-none"
      style={{
        background: `linear-gradient(135deg, rgba(30,20,60,0.8), rgba(50,30,80,0.8))`,
        border: `2px solid rgba(212,175,55,${wallet.comingSoon ? '0.1' : '0.3'})`,
      }}
      whileHover={{ scale: wallet.comingSoon ? 1 : 1.01 }}
      onClick={onCardClick}
    >
      {wallet.comingSoon && (
        <div className="absolute top-2 right-2 px-2 py-1 rounded-lg text-xs font-bold bg-gradient-to-r from-purple-500/30 to-pink-500/30 border border-purple-400/30 text-purple-300">
          Coming Soon
        </div>
      )}
      {wallet.shieldedOnly && (
        <div className="absolute bottom-2 right-2 px-2 py-1 rounded-lg text-xs font-bold bg-gradient-to-r from-green-500/30 to-emerald-500/30 border border-green-400/30 text-green-300 flex items-center gap-1">
          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 16 16">
            <path d="M8 1l-6 2v5c0 3.5 2.5 6.5 6 7.5 3.5-1 6-4 6-7.5V3l-6-2z"/>
          </svg>
          Shielded
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className={`p-2 rounded-lg bg-gradient-to-br ${wallet.color}`}>
          {(wallet.icon === 'qug' || wallet.icon === 'usd') && (
            <div className="relative w-5 h-5">
              <div className="absolute inset-0 rounded-full" style={{
                background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                padding: '1px'
              }}>
                <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center p-0.5">
                  <img src="/quillon-logo.png" alt="Quillon" className="w-full h-full object-contain" style={{ filter: 'invert(1)' }} />
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

      {/* Balance */}
      <div className={`mb-2 ${wallet.comingSoon ? 'text-gray-500' : ''}`}>
        {wallet.comingSoon ? (
          <div className="text-2xl font-bold text-gray-500">0.00</div>
        ) : stableBalance === 0 && wallet.history === undefined ? (
          <div className="text-2xl font-bold text-amber-300/60 animate-pulse">Loading...</div>
        ) : (
          <motion.div
            className="text-2xl font-bold text-white"
            animate={isAnimating ? {
              textShadow: [
                '0 0 10px rgba(255,215,0,0.8)', '0 0 20px rgba(255,107,0,0.8)',
                '0 0 20px rgba(16,185,129,0.8)', '0 0 10px rgba(255,215,0,0.8)',
              ],
            } : {}}
            transition={{ duration: 1.5, repeat: isAnimating ? 1 : 0 }}
          >
            {fmtBalance(stableBalance, wallet.symbol)}
          </motion.div>
        )}
        {wallet.usdValue !== undefined && !wallet.comingSoon && (
          <div className="text-xs text-gray-400 mt-0.5">≈ ${wallet.usdValue.toFixed(2)} USD</div>
        )}
      </div>

      {/* Graph section */}
      {!wallet.comingSoon && (
        <div onClick={(e) => e.stopPropagation()}>
          {/* Controls row: time tabs + gauge toggle */}
          <div className="flex items-center justify-between mb-1.5">
            {/* Time horizon tabs */}
            <div className="flex gap-0.5">
              {HORIZONS.map((h) => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={`px-1.5 py-0.5 rounded text-[9px] font-mono transition-colors ${
                    horizon === h
                      ? (positive ? 'bg-green-500/25 text-green-400' : 'bg-red-500/25 text-red-400')
                      : 'text-gray-600 hover:text-gray-400 hover:bg-white/5'
                  }`}
                >
                  {h}
                </button>
              ))}
            </div>

            {/* Trend badge + gauge toggle */}
            <div className="flex items-center gap-1.5">
              {filteredData.length >= 2 && (
                <span className={`flex items-center gap-0.5 text-[9px] font-bold px-1.5 py-0.5 rounded-full ${
                  positive ? 'bg-green-500/15 text-green-400' : 'bg-red-500/15 text-red-400'
                }`}>
                  {positive ? <TrendingUp className="w-2.5 h-2.5" /> : <TrendingDown className="w-2.5 h-2.5" />}
                  {Math.abs(pctChange).toFixed(1)}%
                </span>
              )}
              <button
                onClick={() => setShowGauge((v) => !v)}
                title="Wealth Velocity Gauge"
                className={`p-0.5 rounded transition-colors ${showGauge ? 'text-yellow-400' : 'text-gray-600 hover:text-gray-400'}`}
              >
                <BarChart2 className="w-3 h-3" />
              </button>
              <button
                onClick={() => setShowModal(true)}
                title="Expand chart"
                className="p-1 rounded transition-all text-amber-500/70 hover:text-amber-400 hover:bg-amber-400/10 border border-amber-500/20 hover:border-amber-400/40"
              >
                <Maximize2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Canvas sparkline */}
          <AnimatePresence mode="wait">
            {!showGauge ? (
              <motion.div
                key="spark"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="relative bg-white/[0.02] rounded-lg overflow-hidden"
                style={{ height: 72 }}
              >
                <canvas
                  ref={sparkRef}
                  className="w-full h-full"
                  onMouseMove={handleCanvasMouseMove}
                  onMouseLeave={handleCanvasMouseLeave}
                />
                {/* Min / max labels */}
                {filteredData.length >= 2 && (() => {
                  const vals = filteredData.map((d) => d.balance);
                  const mn = Math.min(...vals);
                  const mx = Math.max(...vals);
                  return (
                    <>
                      <span className="absolute top-1 left-1.5 text-[8px] text-gray-600 font-mono pointer-events-none">
                        {fmtBalance(mx, wallet.symbol)}
                      </span>
                      <span className="absolute bottom-1 left-1.5 text-[8px] text-gray-600 font-mono pointer-events-none">
                        {fmtBalance(mn, wallet.symbol)}
                      </span>
                    </>
                  );
                })()}
              </motion.div>
            ) : (
              /* ── Wealth Velocity Gauge ── */
              <motion.div
                key="gauge"
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.85 }}
                transition={{ duration: 0.25 }}
                className="flex items-center gap-3 bg-white/[0.02] rounded-lg p-2"
                style={{ height: 72 }}
              >
                <canvas ref={gaugeRef} className="flex-shrink-0" style={{ width: 68, height: 68 }} />
                <div className="flex flex-col gap-1 text-[9px] font-mono">
                  <div>
                    <div className="text-gray-600 uppercase tracking-wider">ATH</div>
                    <div className="text-yellow-400">
                      {fmtBalance(
                        wallet.history && wallet.history.length > 0
                          ? Math.max(...wallet.history.map((d) => d.balance), stableBalance)
                          : stableBalance,
                        wallet.symbol
                      )}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600 uppercase tracking-wider">Velocity</div>
                    <div className={positive ? 'text-green-400' : 'text-red-400'}>
                      {pctChange >= 0 ? '+' : ''}{pctChange.toFixed(2)}% / {horizon}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600 uppercase tracking-wider">Points</div>
                    <div className="text-gray-400">{filteredData.length} stored</div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Expand hint */}
          {showGauge && (
            <div className="flex items-center justify-center gap-1 mt-1 text-[8px] text-gray-700">
              <ChevronUp className="w-2.5 h-2.5" />
              ATH gauge — {horizon} window
            </div>
          )}
        </div>
      )}

      {/* Action Buttons */}
      {children && <div className="mt-3">{children}</div>}

      {/* Quantum shimmer overlay on hover */}
      <motion.div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none"
        style={{
          background: 'linear-gradient(135deg, transparent 30%, rgba(212,175,55,0.08) 50%, transparent 70%)',
        }}
        animate={{ x: ['-100%', '200%'] }}
        transition={{ duration: 1.5, repeat: Infinity, repeatDelay: 2 }}
      />

      {/* ── Expanded Chart Modal ── */}
      {showModal && createPortal(
        <AnimatePresence>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ position: 'fixed', inset: 0, zIndex: 99999, overflowY: 'auto', background: 'rgba(2,4,16,0.92)' }}
            onClick={() => setShowModal(false)}
          >
            <div className="flex min-h-full items-center justify-center p-3 sm:p-6">
              <motion.div
                initial={{ scale: 0.93, opacity: 0, y: 24 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.93, opacity: 0 }}
                transition={{ type: 'spring', stiffness: 280, damping: 26 }}
                className="w-full max-w-4xl rounded-2xl overflow-hidden"
                style={{
                  background: 'linear-gradient(145deg, rgba(10,12,30,0.98), rgba(18,20,50,0.98))',
                  border: '1.5px solid rgba(212,175,55,0.25)',
                  boxShadow: '0 0 80px rgba(212,175,55,0.12), 0 40px 80px rgba(0,0,0,0.7)',
                }}
                onClick={(e) => e.stopPropagation()}
              >
                {/* Modal header */}
                <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg bg-gradient-to-br ${wallet.color}`}>
                      <Wallet className="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <div className="text-white font-bold text-lg leading-none">{wallet.symbol}</div>
                      <div className="text-gray-500 text-xs mt-0.5">{wallet.name}</div>
                    </div>
                    <div className="ml-3 pl-3 border-l border-white/10">
                      <div className="text-2xl font-bold text-white">{fmtBalance(stableBalance, wallet.symbol)}</div>
                      {wallet.usdValue !== undefined && (
                        <div className="text-xs text-gray-400">≈ ${wallet.usdValue.toFixed(2)} USD</div>
                      )}
                    </div>
                    {modalFilteredData.length >= 2 && (
                      <span className={`ml-2 flex items-center gap-1 text-sm font-bold px-2.5 py-1 rounded-full ${
                        modalPositive ? 'bg-green-500/15 text-green-400' : 'bg-red-500/15 text-red-400'
                      }`}>
                        {modalPositive ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                        {modalPctChange >= 0 ? '+' : ''}{modalPctChange.toFixed(2)}%
                      </span>
                    )}
                  </div>
                  <button onClick={() => setShowModal(false)} className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors">
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div className="p-5 space-y-5">
                  {/* Controls row */}
                  <div className="flex items-center justify-between">
                    <div className="flex gap-1.5">
                      {HORIZONS.map((h) => (
                        <button
                          key={h}
                          onClick={() => setModalHorizon(h)}
                          className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                            modalHorizon === h
                              ? (modalPositive
                                  ? 'bg-green-500/20 text-green-400 border border-green-500/40'
                                  : 'bg-red-500/20 text-red-400 border border-red-500/40')
                              : 'text-gray-500 hover:text-gray-300 hover:bg-white/5 border border-transparent'
                          }`}
                        >
                          {h}
                        </button>
                      ))}
                    </div>
                    <button
                      onClick={() => setModalShowGauge((v) => !v)}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all border ${
                        modalShowGauge
                          ? 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30'
                          : 'text-gray-500 hover:text-gray-300 border-white/10 hover:border-white/20'
                      }`}
                    >
                      <BarChart2 className="w-3.5 h-3.5" />
                      Wealth Gauge
                    </button>
                  </div>

                  {/* Big sparkline or gauge */}
                  <AnimatePresence mode="wait">
                    {!modalShowGauge ? (
                      <motion.div
                        key="modal-spark"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="relative rounded-xl overflow-hidden"
                        style={{ height: 240, background: 'rgba(255,255,255,0.015)' }}
                      >
                        <canvas
                          ref={modalSparkRef}
                          className="w-full h-full"
                          onMouseMove={handleModalMouseMove}
                          onMouseLeave={handleModalMouseLeave}
                        />
                        {modalFilteredData.length >= 2 && (() => {
                          const vals = modalFilteredData.map((d) => d.balance);
                          const mn = Math.min(...vals);
                          const mx = Math.max(...vals);
                          const oldest = modalFilteredData[0];
                          const newest = modalFilteredData[modalFilteredData.length - 1];
                          return (
                            <>
                              <span className="absolute top-2 left-3 text-[10px] text-gray-500 font-mono">{fmtBalance(mx, wallet.symbol)}</span>
                              <span className="absolute bottom-2 left-3 text-[10px] text-gray-500 font-mono">{fmtBalance(mn, wallet.symbol)}</span>
                              <span className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-gray-600 font-mono">
                                {new Date(oldest.timestamp).toLocaleTimeString()} → {new Date(newest.timestamp).toLocaleTimeString()}
                              </span>
                              <span className="absolute top-2 right-3 text-[10px] text-gray-600 font-mono">{modalFilteredData.length} points</span>
                            </>
                          );
                        })()}
                      </motion.div>
                    ) : (
                      <motion.div
                        key="modal-gauge"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        className="flex items-center justify-center gap-8 rounded-xl py-6"
                        style={{ background: 'rgba(255,255,255,0.015)', height: 240 }}
                      >
                        <canvas ref={modalGaugeRef} style={{ width: 180, height: 180 }} />
                        <div className="grid grid-cols-2 gap-4 text-sm font-mono">
                          {[
                            { label: 'ATH', value: fmtBalance(wallet.history && wallet.history.length > 0 ? wallet.history.reduce((m, d) => Math.max(m, d.balance), stableBalance) : stableBalance, wallet.symbol), color: 'text-yellow-400' },
                            { label: 'Current', value: fmtBalance(stableBalance, wallet.symbol), color: 'text-white' },
                            { label: `Change ${modalHorizon}`, value: `${modalPctChange >= 0 ? '+' : ''}${modalPctChange.toFixed(3)}%`, color: modalPositive ? 'text-green-400' : 'text-red-400' },
                            { label: 'Data Points', value: `${modalFilteredData.length}`, color: 'text-gray-300' },
                            { label: 'Window', value: modalHorizon, color: 'text-amber-400' },
                            { label: 'Symbol', value: wallet.symbol, color: 'text-gray-300' },
                          ].map(({ label, value, color }) => (
                            <div key={label}>
                              <div className="text-gray-600 text-[10px] uppercase tracking-widest mb-0.5">{label}</div>
                              <div className={`${color} font-bold`}>{value}</div>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Stats row */}
                  <div className="grid grid-cols-4 gap-3">
                    {[
                      { label: 'ALL-TIME HIGH', value: fmtBalance(wallet.history && wallet.history.length > 0 ? wallet.history.reduce((m, d) => Math.max(m, d.balance), stableBalance) : stableBalance, wallet.symbol), color: 'text-yellow-400', sub: 'from history' },
                      { label: `CHANGE ${modalHorizon}`, value: `${modalPctChange >= 0 ? '+' : ''}${modalPctChange.toFixed(3)}%`, color: modalPositive ? 'text-green-400' : 'text-red-400', sub: modalPositive ? 'growing' : 'declining' },
                      { label: 'DATA POINTS', value: `${(wallet.history ?? []).length}`, color: 'text-cyan-400', sub: 'stored history' },
                      { label: 'USD VALUE', value: wallet.usdValue !== undefined ? `$${wallet.usdValue.toFixed(2)}` : '—', color: 'text-emerald-400', sub: 'estimated' },
                    ].map(({ label, value, color, sub }) => (
                      <div key={label} className="rounded-xl p-3" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                        <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-1">{label}</div>
                        <div className={`text-base font-bold font-mono ${color}`}>{value}</div>
                        <div className="text-[9px] text-gray-700 mt-0.5">{sub}</div>
                      </div>
                    ))}
                  </div>

                  {/* Times Square Oscilloscope */}
                  <div className="rounded-xl overflow-hidden" style={{ border: '1px solid rgba(212,175,55,0.12)' }}>
                    <div className="flex items-center gap-2 px-4 py-2.5 border-b" style={{ borderColor: 'rgba(255,255,255,0.05)', background: 'rgba(212,175,55,0.04)' }}>
                      <Activity className="w-3.5 h-3.5 text-amber-400" />
                      <span className="text-[10px] font-bold text-amber-400/80 uppercase tracking-widest">Momentum Oscilloscope</span>
                      <span className="text-[9px] text-gray-700 ml-2">Rate of change between consecutive balance snapshots</span>
                    </div>
                    <canvas ref={oscilloRef} className="w-full" style={{ height: 160 }} />
                  </div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </AnimatePresence>,
        document.getElementById('modal-root') ?? document.body
      )}
    </motion.div>
  );
});

export default WalletCardWithGraph;
