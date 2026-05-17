// NetworkPowerModal.tsx — v10.3.15
// Full-screen slide-in modal showing miner list + hashrate history + radial power ring.
// Opened by clicking the "Network Hash Rate" card in MiningDashboard.
// History cached in localStorage to survive node restarts.

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Search, TrendingUp, Users, Activity, Zap, Award, Clock } from 'lucide-react';
import { qnkAPI } from '../services/api';

// ──────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────

interface NetworkPowerModalProps {
  isOpen: boolean;
  onClose: () => void;
  networkHashRate: number;
  connectedMiners: number;
}

interface ServerMiner {
  address: string;
  worker_id: string;
  worker_name: string | null;
  hash_rate: number;
  blocks_found: number;
  total_solutions: number;
  rewards_earned: string;
  last_seen_secs_ago: number;
  source: string;
  peer_miner_count?: number;
}

interface HashrateHistoryPoint {
  hashrate: number;
  miners: number;
  timestamp: number;
}

type SortKey = 'hash_rate' | 'blocks_found' | 'last_seen_secs_ago';
type SortDir = 'asc' | 'desc';

// ──────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────

const HISTORY_CACHE_KEY = 'qnk_hashrate_history_v2';
const HISTORY_CACHE_TTL = 86400; // 24h in seconds

// ──────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────

function fmtHash(hps: number): string {
  if (hps >= 1e12) return `${(hps / 1e12).toFixed(2)} TH/s`;
  if (hps >= 1e9) return `${(hps / 1e9).toFixed(2)} GH/s`;
  if (hps >= 1e6) return `${(hps / 1e6).toFixed(2)} MH/s`;
  if (hps >= 1e3) return `${(hps / 1e3).toFixed(2)} kH/s`;
  return `${hps.toFixed(0)} H/s`;
}

function truncAddr(addr: string): string {
  if (addr.length <= 14) return addr;
  return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
}

function secsAgo(secs: number): string {
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

function loadCachedHistory(): HashrateHistoryPoint[] {
  try {
    const raw = localStorage.getItem(HISTORY_CACHE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HashrateHistoryPoint[];
    const cutoff = Math.floor(Date.now() / 1000) - HISTORY_CACHE_TTL;
    return parsed.filter((p) => p.timestamp >= cutoff);
  } catch {
    return [];
  }
}

function saveCachedHistory(data: HashrateHistoryPoint[]) {
  try {
    localStorage.setItem(HISTORY_CACHE_KEY, JSON.stringify(data));
  } catch {
    // Quota exceeded — ignore
  }
}

function mergeHistory(
  cached: HashrateHistoryPoint[],
  fresh: HashrateHistoryPoint[]
): HashrateHistoryPoint[] {
  const seen = new Set<number>();
  const merged: HashrateHistoryPoint[] = [];
  for (const p of [...cached, ...fresh]) {
    // Round timestamp to nearest 60s to deduplicate
    const bucket = Math.round(p.timestamp / 60) * 60;
    if (!seen.has(bucket)) {
      seen.add(bucket);
      merged.push(p);
    }
  }
  merged.sort((a, b) => a.timestamp - b.timestamp);
  const cutoff = Math.floor(Date.now() / 1000) - HISTORY_CACHE_TTL;
  return merged.filter((p) => p.timestamp >= cutoff);
}

const RANK_BADGE = ['text-yellow-400', 'text-gray-300', 'text-amber-600'];

// ──────────────────────────────────────────────────────────────
// Main Chart (hashrate + miners line chart)
// ──────────────────────────────────────────────────────────────

function drawMainChart(
  canvas: HTMLCanvasElement,
  data: HashrateHistoryPoint[],
  mouseX: number | null
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 600;
  const H = canvas.clientHeight || 220;

  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  ctx.clearRect(0, 0, W, H);

  const pl = 58, pr = 48, pt = 16, pb = 28;
  const cW = W - pl - pr;
  const cH = H - pt - pb;

  if (data.length < 2) {
    ctx.font = '11px sans-serif';
    ctx.fillStyle = 'rgba(148,163,184,0.4)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Collecting hashrate history…', W / 2, H / 2);
    ctx.font = '9px sans-serif';
    ctx.fillText('Samples arrive every 60s', W / 2, H / 2 + 16);
    return;
  }

  const t0 = data[0].timestamp;
  const tN = data[data.length - 1].timestamp;
  const tRange = tN - t0 || 1;

  let maxH = 0, maxM = 0;
  for (const d of data) {
    if (d.hashrate > maxH) maxH = d.hashrate;
    if (d.miners > maxM) maxM = d.miners;
  }
  maxH = maxH * 1.15 || 1;
  maxM = Math.ceil(maxM * 1.15) || 1;

  // Grid lines
  ctx.setLineDash([2, 4]);
  ctx.strokeStyle = 'rgba(148,163,184,0.07)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pt + (i / 4) * cH;
    ctx.beginPath(); ctx.moveTo(pl, y); ctx.lineTo(pl + cW, y); ctx.stroke();
  }
  ctx.setLineDash([]);

  // Y-left labels (hashrate)
  ctx.font = '9px sans-serif';
  ctx.fillStyle = 'rgba(96,165,250,0.55)';
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    ctx.fillText(fmtHash(maxH * (1 - i / 4)), pl - 5, pt + (i / 4) * cH);
  }

  // Y-right labels (miners)
  ctx.fillStyle = 'rgba(74,222,128,0.55)';
  ctx.textAlign = 'left';
  for (let i = 0; i <= 4; i++) {
    ctx.fillText(`${Math.round(maxM * (1 - i / 4))}`, pl + cW + 5, pt + (i / 4) * cH);
  }

  // X-axis labels
  ctx.fillStyle = 'rgba(148,163,184,0.35)';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let i = 0; i < 6; i++) {
    const t = t0 + (tRange * i) / 5;
    const x = pl + (i / 5) * cW;
    ctx.fillText(new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), x, pt + cH + 5);
  }

  // Hashrate fill + stroke
  const hashGrad = ctx.createLinearGradient(0, pt, 0, pt + cH);
  hashGrad.addColorStop(0, 'rgba(59,130,246,0.28)');
  hashGrad.addColorStop(1, 'rgba(59,130,246,0)');

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = pl + ((data[i].timestamp - t0) / tRange) * cW;
    const y = pt + cH - (data[i].hashrate / maxH) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = 'rgba(59,130,246,0.9)';
  ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();

  const lastD = data[data.length - 1];
  const lastX = pl + ((lastD.timestamp - t0) / tRange) * cW;
  ctx.lineTo(lastX, pt + cH); ctx.lineTo(pl, pt + cH); ctx.closePath();
  ctx.fillStyle = hashGrad; ctx.fill();

  // Miner count line
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = pl + ((data[i].timestamp - t0) / tRange) * cW;
    const y = pt + cH - (data[i].miners / maxM) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = 'rgba(74,222,128,0.75)';
  ctx.lineWidth = 1.5; ctx.lineJoin = 'round'; ctx.stroke();

  // Hover crosshair
  if (mouseX !== null && mouseX >= pl && mouseX <= pl + cW) {
    const tAtMouse = t0 + ((mouseX - pl) / cW) * tRange;
    let nearest = 0, nearestDist = Infinity;
    for (let i = 0; i < data.length; i++) {
      const d = Math.abs(data[i].timestamp - tAtMouse);
      if (d < nearestDist) { nearestDist = d; nearest = i; }
    }
    const dp = data[nearest];
    const dpX = pl + ((dp.timestamp - t0) / tRange) * cW;
    const dpYH = pt + cH - (dp.hashrate / maxH) * cH;
    const dpYM = pt + cH - (dp.miners / maxM) * cH;

    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = 'rgba(148,163,184,0.2)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(dpX, pt); ctx.lineTo(dpX, pt + cH); ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath(); ctx.arc(dpX, dpYH, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#3b82f6'; ctx.fill();
    ctx.beginPath(); ctx.arc(dpX, dpYM, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#4ade80'; ctx.fill();

    const timeStr = new Date(dp.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const boxW = 122, boxH = 54;
    let bx = dpX + 10;
    if (bx + boxW > pl + cW) bx = dpX - boxW - 10;
    ctx.fillStyle = 'rgba(10,15,30,0.93)'; ctx.strokeStyle = 'rgba(59,130,246,0.3)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(bx, pt + 8, boxW, boxH, 6); ctx.fill(); ctx.stroke();
    ctx.font = '10px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(148,163,184,0.65)'; ctx.fillText(timeStr, bx + 8, pt + 14);
    ctx.fillStyle = 'rgba(96,165,250,0.9)'; ctx.fillText(fmtHash(dp.hashrate), bx + 8, pt + 28);
    ctx.fillStyle = 'rgba(74,222,128,0.9)'; ctx.fillText(`${dp.miners} miners`, bx + 8, pt + 42);
  }

  // Axis labels
  ctx.save(); ctx.translate(11, pt + cH / 2); ctx.rotate(-Math.PI / 2);
  ctx.font = '9px sans-serif'; ctx.fillStyle = 'rgba(96,165,250,0.45)'; ctx.textAlign = 'center';
  ctx.fillText('Hash Rate', 0, 0); ctx.restore();

  ctx.save(); ctx.translate(W - 7, pt + cH / 2); ctx.rotate(Math.PI / 2);
  ctx.font = '9px sans-serif'; ctx.fillStyle = 'rgba(74,222,128,0.45)'; ctx.textAlign = 'center';
  ctx.fillText('Miners', 0, 0); ctx.restore();
}

// ──────────────────────────────────────────────────────────────
// Radial Power Ring chart (24h by hour — Times-magazine quality)
// ──────────────────────────────────────────────────────────────

function drawRadialChart(canvas: HTMLCanvasElement, data: HashrateHistoryPoint[]) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 300;
  const H = canvas.clientHeight || 300;

  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  ctx.clearRect(0, 0, W, H);

  const cx = W / 2;
  const cy = H / 2;
  const outerR = Math.min(W, H) * 0.44;
  const innerR = outerR * 0.38;

  if (data.length < 2) {
    ctx.font = '10px sans-serif';
    ctx.fillStyle = 'rgba(148,163,184,0.35)';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('Building power ring…', cx, cy);
    return;
  }

  // Bucket data by hour-of-day
  const hourBuckets: number[][] = Array.from({ length: 24 }, () => []);
  for (const p of data) {
    const h = new Date(p.timestamp * 1000).getHours();
    if (p.hashrate > 0) hourBuckets[h].push(p.hashrate);
  }
  const hourAvg = hourBuckets.map((b) =>
    b.length > 0 ? b.reduce((a, x) => a + x, 0) / b.length : 0
  );
  const maxAvg = Math.max(...hourAvg, 1);

  const segAngle = (Math.PI * 2) / 24;
  const gapAngle = 0.012;
  const startOffset = -Math.PI / 2; // 12 o'clock = hour 0

  // Background ring (faint track)
  ctx.beginPath();
  ctx.arc(cx, cy, outerR, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(148,163,184,0.06)';
  ctx.lineWidth = outerR - innerR;
  ctx.stroke();

  // Draw each hour segment
  for (let h = 0; h < 24; h++) {
    const pct = hourAvg[h] / maxAvg;
    const segInner = innerR;
    const segOuter = innerR + (outerR - innerR) * Math.max(pct, 0.04);

    const a0 = startOffset + h * segAngle + gapAngle / 2;
    const a1 = startOffset + (h + 1) * segAngle - gapAngle / 2;

    // Color: quantum gradient from cyan (low) to purple (high)
    const hue = 200 + pct * 70; // 200 = cyan, 270 = purple
    const sat = 70 + pct * 25;
    const lum = 45 + pct * 20;
    const alpha = 0.25 + pct * 0.65;

    // Glow for high-power segments
    if (pct > 0.6) {
      ctx.shadowColor = `hsla(${hue},${sat}%,${lum}%,0.5)`;
      ctx.shadowBlur = 8 + pct * 12;
    } else {
      ctx.shadowBlur = 0;
    }

    ctx.beginPath();
    ctx.arc(cx, cy, segOuter, a0, a1);
    ctx.arc(cx, cy, segInner, a1, a0, true);
    ctx.closePath();
    ctx.fillStyle = `hsla(${hue},${sat}%,${lum}%,${alpha.toFixed(2)})`;
    ctx.fill();

    // Thin bright edge on outer arc
    ctx.beginPath();
    ctx.arc(cx, cy, segOuter, a0, a1);
    ctx.strokeStyle = `hsla(${hue},${sat + 10}%,${lum + 25}%,${(alpha * 0.7).toFixed(2)})`;
    ctx.lineWidth = 1;
    ctx.shadowBlur = 0;
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // Hour tick labels (0, 6, 12, 18)
  ctx.font = `bold ${Math.round(outerR * 0.13)}px sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const labelR = outerR * 1.18;
  for (const h of [0, 6, 12, 18]) {
    const angle = startOffset + h * segAngle + segAngle / 2;
    const lx = cx + Math.cos(angle) * labelR;
    const ly = cy + Math.sin(angle) * labelR;
    ctx.fillStyle = 'rgba(148,163,184,0.5)';
    ctx.fillText(`${h}h`, lx, ly);
  }

  // Inner circle: dark background + current stats
  const innerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, innerR * 0.9);
  innerGrad.addColorStop(0, 'rgba(6,10,22,0.97)');
  innerGrad.addColorStop(1, 'rgba(3,8,24,0.93)');
  ctx.beginPath(); ctx.arc(cx, cy, innerR * 0.95, 0, Math.PI * 2);
  ctx.fillStyle = innerGrad; ctx.fill();

  // Ring border glow
  ctx.beginPath(); ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(0,255,255,0.12)'; ctx.lineWidth = 1; ctx.stroke();

  // Current hashrate text
  const currentHr = data[data.length - 1]?.hashrate ?? 0;
  const fontSize = Math.round(innerR * 0.25);
  ctx.font = `bold ${fontSize}px 'SF Pro Display', system-ui, sans-serif`;
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';

  const hrStr = fmtHash(currentHr);
  // Split number and unit for two-tone rendering
  const parts = hrStr.match(/^([\d.]+)\s*(.+)$/) ?? [hrStr, hrStr, ''];
  ctx.fillText(parts[1], cx, cy - innerR * 0.12);

  ctx.font = `${Math.round(fontSize * 0.48)}px sans-serif`;
  ctx.fillStyle = 'rgba(0,240,255,0.7)';
  ctx.fillText(parts[2], cx, cy + innerR * 0.2);

  // "24h pattern" label
  ctx.font = `${Math.round(innerR * 0.12)}px sans-serif`;
  ctx.fillStyle = 'rgba(148,163,184,0.4)';
  ctx.fillText('24h pattern', cx, cy + innerR * 0.52);
}

// ──────────────────────────────────────────────────────────────
// Component
// ──────────────────────────────────────────────────────────────

export default function NetworkPowerModal({
  isOpen,
  onClose,
  networkHashRate,
  connectedMiners,
}: NetworkPowerModalProps) {
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('hash_rate');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [historyData, setHistoryData] = useState<HashrateHistoryPoint[]>(() => loadCachedHistory());
  const [serverMiners, setServerMiners] = useState<ServerMiner[]>([]);
  const [totalMinerCount, setTotalMinerCount] = useState(0);
  const [activeTab, setActiveTab] = useState<'timeline' | 'ring'>('timeline');

  const mainCanvasRef = useRef<HTMLCanvasElement>(null);
  const ringCanvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const historyRef = useRef<HashrateHistoryPoint[]>(historyData);
  const mouseXRef = useRef<number | null>(null);

  useEffect(() => { historyRef.current = historyData; }, [historyData]);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await qnkAPI.getHashrateHistory();
      if (res.success && res.history) {
        setHistoryData((prev) => {
          const merged = mergeHistory(prev, res.history);
          saveCachedHistory(merged);
          return merged;
        });
      }
    } catch (err) {
      console.error('[NetworkPowerModal] Failed to fetch hashrate history:', err);
    }
  }, []);

  const fetchMiners = useCallback(async () => {
    try {
      const res = await qnkAPI.getNetworkMiners();
      if (res.success && res.miners) {
        setServerMiners(res.miners);
        setTotalMinerCount(res.total_miners);
      }
    } catch (err) {
      console.error('[NetworkPowerModal] Failed to fetch miners:', err);
    }
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    fetchHistory();
    fetchMiners();
    const hi = setInterval(fetchHistory, 60_000);
    const mi = setInterval(fetchMiners, 30_000);
    return () => { clearInterval(hi); clearInterval(mi); };
  }, [isOpen, fetchHistory, fetchMiners]);

  // Derived stats
  const stats = useMemo(() => {
    if (historyData.length === 0) return null;
    const hrs = historyData.map((d) => d.hashrate);
    const avg = hrs.reduce((a, b) => a + b, 0) / hrs.length;
    const peak = Math.max(...hrs);
    const peakMiners = Math.max(...historyData.map((d) => d.miners));
    const first = historyData[0]?.hashrate ?? 0;
    const last = historyData[historyData.length - 1]?.hashrate ?? 0;
    const growth = first > 0 ? ((last - first) / first) * 100 : 0;
    const coverage = historyData.length; // number of samples
    return { avg, peak, peakMiners, growth, coverage };
  }, [historyData]);

  // Sorted miner list
  const sortedMiners = useMemo(() => {
    let arr = [...serverMiners];
    if (search) {
      const q = search.toLowerCase();
      arr = arr.filter(
        (m) =>
          m.address.toLowerCase().includes(q) ||
          (m.worker_name && m.worker_name.toLowerCase().includes(q)) ||
          m.worker_id.toLowerCase().includes(q)
      );
    }
    arr.sort((a, b) => {
      let cmp = 0;
      if (sortKey === 'hash_rate') cmp = a.hash_rate - b.hash_rate;
      else if (sortKey === 'blocks_found') cmp = a.blocks_found - b.blocks_found;
      else cmp = b.last_seen_secs_ago - a.last_seen_secs_ago;
      return sortDir === 'desc' ? -cmp : cmp;
    });
    return arr;
  }, [serverMiners, search, sortKey, sortDir]);

  const handleSort = useCallback(
    (key: SortKey) => {
      if (sortKey === key) setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
      else { setSortKey(key); setSortDir('desc'); }
    },
    [sortKey]
  );
  const sortArrow = (key: SortKey) => sortKey === key ? (sortDir === 'desc' ? ' ▼' : ' ▲') : '';

  // Animation loop for main chart
  const drawLoop = useCallback(() => {
    if (activeTab === 'timeline' && mainCanvasRef.current) {
      drawMainChart(mainCanvasRef.current, historyRef.current, mouseXRef.current);
    }
    animRef.current = requestAnimationFrame(drawLoop);
  }, [activeTab]);

  // Draw ring chart (static — no animation needed)
  useEffect(() => {
    if (activeTab === 'ring' && ringCanvasRef.current) {
      drawRadialChart(ringCanvasRef.current, historyRef.current);
    }
  }, [activeTab, historyData]);

  useEffect(() => {
    if (!isOpen) return;
    animRef.current = requestAnimationFrame(drawLoop);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [isOpen, drawLoop]);

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    mouseXRef.current = e.clientX - e.currentTarget.getBoundingClientRect().left;
  }, []);
  const handleCanvasMouseLeave = useCallback(() => { mouseXRef.current = null; }, []);

  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  const sourceBadge = (source: string) => {
    switch (source) {
      case 'local': return 'bg-quantum-green/20 text-quantum-green';
      case 'p2p':   return 'bg-quantum-cyan/20 text-quantum-cyan';
      case 'peer':  return 'bg-quantum-purple/20 text-quantum-purple';
      default:      return 'bg-white/10 text-gray-400';
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 bg-black/65 backdrop-blur-sm"
            onClick={onClose}
          />

          <motion.div
            initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="fixed inset-y-0 right-0 z-50 w-full max-w-[1280px] bg-gradient-to-bl from-[#080d1c] to-[#020710] border-l border-quantum-cyan/20 shadow-2xl overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* ── Header ── */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 flex-shrink-0">
              <div className="flex items-center gap-3">
                <Activity className="w-5 h-5 text-quantum-cyan" />
                <h2 className="text-lg font-bold text-white tracking-tight">Network Power</h2>
                <span className="text-xs bg-quantum-cyan/15 text-quantum-cyan px-2.5 py-0.5 rounded-full font-mono">
                  {fmtHash(networkHashRate)}
                </span>
                {connectedMiners > 0 && (
                  <span className="text-xs bg-quantum-green/15 text-quantum-green px-2.5 py-0.5 rounded-full">
                    {connectedMiners} miners
                  </span>
                )}
                {historyData.length > 0 && (
                  <span className="text-[10px] text-gray-600 ml-1">
                    {historyData.length} samples cached
                  </span>
                )}
              </div>
              <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/10 transition-colors">
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* ── Stats bar ── */}
            {stats && (
              <div className="flex items-center gap-px border-b border-white/5 flex-shrink-0 bg-white/[0.015]">
                {[
                  { icon: <Zap className="w-3.5 h-3.5" />, label: 'Peak 24h', value: fmtHash(stats.peak), color: 'text-blue-400' },
                  { icon: <Activity className="w-3.5 h-3.5" />, label: 'Avg 24h', value: fmtHash(stats.avg), color: 'text-cyan-400' },
                  { icon: <Users className="w-3.5 h-3.5" />, label: 'Peak Miners', value: `${stats.peakMiners}`, color: 'text-green-400' },
                  {
                    icon: <TrendingUp className="w-3.5 h-3.5" />,
                    label: '24h Change',
                    value: `${stats.growth >= 0 ? '+' : ''}${stats.growth.toFixed(1)}%`,
                    color: stats.growth >= 0 ? 'text-green-400' : 'text-red-400',
                  },
                  { icon: <Clock className="w-3.5 h-3.5" />, label: 'History', value: `${Math.round(stats.coverage)} min`, color: 'text-purple-400' },
                  { icon: <Award className="w-3.5 h-3.5" />, label: 'Current', value: fmtHash(networkHashRate), color: 'text-yellow-400' },
                ].map((s, i) => (
                  <div key={i} className="flex-1 flex items-center gap-2 px-4 py-2.5">
                    <span className={`${s.color} opacity-60`}>{s.icon}</span>
                    <div>
                      <div className="text-[9px] text-gray-600 uppercase tracking-wider">{s.label}</div>
                      <div className={`text-xs font-bold ${s.color} font-mono`}>{s.value}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* ── Body ── */}
            <div className="flex-1 flex flex-col lg:flex-row overflow-hidden min-h-0">

              {/* Left: Miner List */}
              <div className="w-full lg:w-[420px] flex-shrink-0 flex flex-col border-r border-white/5 min-h-0">
                <div className="px-4 pt-3 pb-2 flex items-center justify-between flex-shrink-0">
                  <div className="flex items-center gap-2">
                    <Users className="w-4 h-4 text-quantum-cyan" />
                    <span className="text-sm font-semibold text-white">
                      {totalMinerCount > 0 ? totalMinerCount : sortedMiners.length} active
                    </span>
                  </div>
                </div>

                <div className="px-4 pb-2 flex-shrink-0">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-600" />
                    <input
                      type="text"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Filter by address or worker…"
                      className="w-full pl-8 pr-3 py-1.5 bg-white/5 border border-white/8 rounded-lg text-xs text-white placeholder-gray-600 focus:outline-none focus:border-quantum-cyan/40"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-[36px_1fr_90px_58px_60px_42px] gap-0.5 px-4 pb-1 text-[9px] text-gray-600 uppercase tracking-wider select-none flex-shrink-0">
                  <span>#</span>
                  <span>Miner</span>
                  <button onClick={() => handleSort('hash_rate')} className="text-left hover:text-gray-400">Hash{sortArrow('hash_rate')}</button>
                  <button onClick={() => handleSort('blocks_found')} className="text-left hover:text-gray-400">Blk{sortArrow('blocks_found')}</button>
                  <button onClick={() => handleSort('last_seen_secs_ago')} className="text-left hover:text-gray-400">Seen{sortArrow('last_seen_secs_ago')}</button>
                  <span>Src</span>
                </div>

                <div className="flex-1 overflow-y-auto min-h-0 px-2">
                  {sortedMiners.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-32 text-gray-600 text-xs gap-1">
                      {search ? 'No miners match filter' : 'Loading miners…'}
                    </div>
                  ) : (
                    sortedMiners.map((miner, idx) => (
                      <div
                        key={`${miner.address}-${miner.worker_id}`}
                        className="grid grid-cols-[36px_1fr_90px_58px_60px_42px] gap-0.5 items-center px-2 py-1.5 rounded-md hover:bg-white/[0.04] transition-colors text-xs"
                      >
                        <span className={`font-mono font-bold text-[10px] ${idx < 3 ? RANK_BADGE[idx] : 'text-gray-600'}`}>
                          {idx < 3 ? ['1st','2nd','3rd'][idx] : `#${idx+1}`}
                        </span>
                        <div className="flex flex-col truncate">
                          <span className="text-gray-300 font-mono truncate text-[10px]" title={miner.address}>
                            {truncAddr(miner.address)}
                          </span>
                          {miner.worker_name && (
                            <span className="text-[9px] text-gray-600 truncate">{miner.worker_name}</span>
                          )}
                        </div>
                        <span className="text-blue-400 font-mono text-[10px]">{fmtHash(miner.hash_rate)}</span>
                        <span className="text-gray-500 font-mono text-[10px]">{miner.blocks_found}</span>
                        <span className="text-gray-600 font-mono text-[9px]">{secsAgo(miner.last_seen_secs_ago)}</span>
                        <span className={`text-[8px] px-1 py-0.5 rounded text-center ${sourceBadge(miner.source)}`}>
                          {miner.source}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Right: Charts */}
              <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
                {/* Tab bar */}
                <div className="flex items-center gap-1 px-4 pt-3 pb-2 border-b border-white/5 flex-shrink-0">
                  <button
                    onClick={() => setActiveTab('timeline')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                      activeTab === 'timeline'
                        ? 'bg-blue-500/20 text-blue-400'
                        : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                    }`}
                  >
                    <TrendingUp className="w-3.5 h-3.5" />
                    Hashrate Timeline
                  </button>
                  <button
                    onClick={() => setActiveTab('ring')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                      activeTab === 'ring'
                        ? 'bg-cyan-500/20 text-cyan-400'
                        : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                    }`}
                  >
                    <Activity className="w-3.5 h-3.5" />
                    24h Power Ring
                  </button>
                  <div className="ml-auto flex items-center gap-3 text-[10px]">
                    {activeTab === 'timeline' && (
                      <>
                        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500" /><span className="text-gray-500">Hash Rate</span></span>
                        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-400" /><span className="text-gray-500">Miners</span></span>
                      </>
                    )}
                    {activeTab === 'ring' && (
                      <span className="text-gray-600">Hourly hashrate distribution — brighter = more power</span>
                    )}
                  </div>
                </div>

                {/* Chart area */}
                <div className="flex-1 min-h-0 p-4 flex flex-col">
                  {/* Timeline tab */}
                  {activeTab === 'timeline' && (
                    <>
                      <div className="flex-1 relative bg-white/[0.015] rounded-xl border border-white/[0.06] overflow-hidden min-h-[180px]">
                        <canvas
                          ref={mainCanvasRef}
                          className="w-full h-full"
                          onMouseMove={handleCanvasMouseMove}
                          onMouseLeave={handleCanvasMouseLeave}
                        />
                      </div>

                      {/* Stats cards */}
                      <div className="grid grid-cols-3 gap-3 mt-3 flex-shrink-0">
                        <div className="bg-white/[0.025] border border-white/[0.06] rounded-xl p-3">
                          <div className="text-[9px] text-gray-600 uppercase tracking-wider mb-1.5">Peak Hashrate</div>
                          <div className="text-sm font-bold text-blue-400 font-mono">
                            {stats ? fmtHash(stats.peak) : '--'}
                          </div>
                          <div className="text-[9px] text-gray-600 mt-0.5">last 24h</div>
                        </div>
                        <div className="bg-white/[0.025] border border-white/[0.06] rounded-xl p-3">
                          <div className="text-[9px] text-gray-600 uppercase tracking-wider mb-1.5">Avg Hashrate</div>
                          <div className="text-sm font-bold text-cyan-400 font-mono">
                            {stats ? fmtHash(stats.avg) : '--'}
                          </div>
                          <div className="text-[9px] text-gray-600 mt-0.5">24h average</div>
                        </div>
                        <div className="bg-white/[0.025] border border-white/[0.06] rounded-xl p-3">
                          <div className="text-[9px] text-gray-600 uppercase tracking-wider mb-1.5">24h Trend</div>
                          <div className={`text-sm font-bold font-mono ${stats && stats.growth >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {stats ? `${stats.growth >= 0 ? '+' : ''}${stats.growth.toFixed(1)}%` : '--'}
                          </div>
                          <div className="text-[9px] text-gray-600 mt-0.5">start vs now</div>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Power Ring tab */}
                  {activeTab === 'ring' && (
                    <div className="flex-1 flex flex-col lg:flex-row gap-4 min-h-0">
                      {/* Ring canvas */}
                      <div className="flex items-center justify-center flex-1 min-h-0">
                        <div className="relative bg-white/[0.015] rounded-2xl border border-white/[0.06] overflow-hidden"
                             style={{ width: '100%', maxWidth: 380, aspectRatio: '1' }}>
                          <canvas
                            ref={ringCanvasRef}
                            className="w-full h-full"
                          />
                        </div>
                      </div>

                      {/* Ring legend / breakdown */}
                      <div className="lg:w-52 flex flex-col gap-3 flex-shrink-0 justify-center">
                        <div>
                          <div className="text-[9px] text-gray-600 uppercase tracking-wider mb-2">Hourly Breakdown</div>
                          <div className="space-y-1 max-h-64 overflow-y-auto pr-1">
                            {Array.from({ length: 24 }, (_, h) => {
                              const pts = historyData.filter(
                                (d) => new Date(d.timestamp * 1000).getHours() === h && d.hashrate > 0
                              );
                              const avg = pts.length > 0 ? pts.reduce((a, p) => a + p.hashrate, 0) / pts.length : 0;
                              const peakAll = stats?.peak ?? 1;
                              const pct = avg / peakAll;
                              return (
                                <div key={h} className="flex items-center gap-2">
                                  <span className="text-[9px] text-gray-600 w-5 text-right font-mono">{h}h</span>
                                  <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                                    <div
                                      className="h-full rounded-full transition-all"
                                      style={{
                                        width: `${pct * 100}%`,
                                        background: `hsl(${200 + pct * 70},${70 + pct * 25}%,${45 + pct * 20}%)`,
                                        opacity: 0.7 + pct * 0.3,
                                      }}
                                    />
                                  </div>
                                  <span className="text-[9px] text-gray-600 font-mono w-14 text-right">
                                    {avg > 0 ? fmtHash(avg) : '—'}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                        </div>

                        <div className="border-t border-white/5 pt-3 space-y-2">
                          <div className="text-[9px] text-gray-600 uppercase tracking-wider">Ring Legend</div>
                          <div className="flex items-center gap-2">
                            <div className="w-6 h-2 rounded" style={{ background: 'hsl(200,70%,45%)' }} />
                            <span className="text-[9px] text-gray-500">Low activity</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-6 h-2 rounded" style={{ background: 'hsl(235,82%,55%)' }} />
                            <span className="text-[9px] text-gray-500">Medium activity</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-6 h-2 rounded" style={{ background: 'hsl(270,95%,65%)' }} />
                            <span className="text-[9px] text-gray-500">Peak activity</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
