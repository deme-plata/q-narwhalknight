// NetworkPowerModal.tsx — v10.3.0
// Full-screen slide-in modal showing miner list + hashrate history chart.
// Opened by clicking the "Network Hash Rate" card in MiningDashboard.
// Fetches live miner list from /api/v1/mining/miners and hashrate history
// from /api/v1/mining/hashrate/history.

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Search, TrendingUp, Users, Activity } from 'lucide-react';
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

/** Miner entry from /api/v1/mining/miners */
interface ServerMiner {
  address: string;
  worker_id: string;
  worker_name: string | null;
  hash_rate: number;
  blocks_found: number;
  total_solutions: number;
  rewards_earned: string;
  last_seen_secs_ago: number;
  source: string; // "local" | "p2p" | "peer"
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

const RANK_BADGE = ['text-yellow-400', 'text-gray-300', 'text-amber-600'];

// ──────────────────────────────────────────────────────────────
// Component
// ──────────────────────────────────────────────────────────────

export default function NetworkPowerModal({
  isOpen,
  onClose,
  networkHashRate,
  connectedMiners,
}: NetworkPowerModalProps) {
  // ── State ──
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('hash_rate');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [historyData, setHistoryData] = useState<HashrateHistoryPoint[]>([]);
  const [serverMiners, setServerMiners] = useState<ServerMiner[]>([]);
  const [totalMinerCount, setTotalMinerCount] = useState(0);

  // ── Refs ──
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const historyRef = useRef<HashrateHistoryPoint[]>([]);
  const mouseXRef = useRef<number | null>(null);

  // Keep refs in sync
  useEffect(() => { historyRef.current = historyData; }, [historyData]);

  // ── Fetch hashrate history ──
  const fetchHistory = useCallback(async () => {
    try {
      const res = await qnkAPI.getHashrateHistory();
      if (res.success && res.history) {
        setHistoryData(res.history);
      }
    } catch (err) {
      console.error('[NetworkPowerModal] Failed to fetch hashrate history:', err);
    }
  }, []);

  // ── Fetch full miner list from server ──
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
    const historyInterval = setInterval(fetchHistory, 60_000);
    const minerInterval = setInterval(fetchMiners, 30_000); // refresh miner list every 30s
    return () => {
      clearInterval(historyInterval);
      clearInterval(minerInterval);
    };
  }, [isOpen, fetchHistory, fetchMiners]);

  // ── Miner list (from server data) ──
  const sortedMiners = useMemo(() => {
    let arr = [...serverMiners];

    // Filter
    if (search) {
      const q = search.toLowerCase();
      arr = arr.filter(
        (m) =>
          m.address.toLowerCase().includes(q) ||
          (m.worker_name && m.worker_name.toLowerCase().includes(q)) ||
          m.worker_id.toLowerCase().includes(q)
      );
    }

    // Sort
    arr.sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case 'hash_rate':
          cmp = a.hash_rate - b.hash_rate;
          break;
        case 'blocks_found':
          cmp = a.blocks_found - b.blocks_found;
          break;
        case 'last_seen_secs_ago':
          // Lower secs_ago = more recent = should rank higher in desc
          cmp = b.last_seen_secs_ago - a.last_seen_secs_ago;
          break;
      }
      return sortDir === 'desc' ? -cmp : cmp;
    });

    return arr;
  }, [serverMiners, search, sortKey, sortDir]);

  const handleSort = useCallback(
    (key: SortKey) => {
      if (sortKey === key) {
        setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
      } else {
        setSortKey(key);
        setSortDir('desc');
      }
    },
    [sortKey]
  );

  const sortArrow = useCallback(
    (key: SortKey) => (sortKey === key ? (sortDir === 'desc' ? ' \u25BC' : ' \u25B2') : ''),
    [sortKey, sortDir]
  );

  // ── Canvas chart ──
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || 600;
    const H = canvas.clientHeight || 300;

    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    ctx.clearRect(0, 0, W, H);

    const pl = 60, pr = 50, pt = 20, pb = 30;
    const cW = W - pl - pr;
    const cH = H - pt - pb;

    const data = historyRef.current;

    if (data.length < 2) {
      ctx.font = '12px sans-serif';
      ctx.fillStyle = 'rgba(148,163,184,0.4)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Collecting hashrate history data...', W / 2, H / 2);
      ctx.font = '10px sans-serif';
      ctx.fillText('Data appears after the server samples (every 60s)', W / 2, H / 2 + 18);
      animRef.current = requestAnimationFrame(drawChart);
      return;
    }

    // Compute ranges
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

    // Grid lines (5 horizontal)
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = 'rgba(148,163,184,0.08)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pt + (i / 4) * cH;
      ctx.beginPath();
      ctx.moveTo(pl, y);
      ctx.lineTo(pl + cW, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Y-axis left labels (hashrate)
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(96,165,250,0.6)';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const val = maxH * (1 - i / 4);
      const y = pt + (i / 4) * cH;
      ctx.fillText(fmtHash(val), pl - 6, y);
    }

    // Y-axis right labels (miner count)
    ctx.fillStyle = 'rgba(74,222,128,0.6)';
    ctx.textAlign = 'left';
    for (let i = 0; i <= 4; i++) {
      const val = Math.round(maxM * (1 - i / 4));
      const y = pt + (i / 4) * cH;
      ctx.fillText(`${val}`, pl + cW + 6, y);
    }

    // X-axis time labels (every 4 hours, 6 labels)
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(148,163,184,0.4)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i < 6; i++) {
      const t = t0 + (tRange * i) / 5;
      const x = pl + (i / 5) * cW;
      const label = new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      ctx.fillText(label, x, pt + cH + 6);
    }

    // Hashrate line + gradient fill
    const hashGrad = ctx.createLinearGradient(0, pt, 0, pt + cH);
    hashGrad.addColorStop(0, 'rgba(59,130,246,0.3)');
    hashGrad.addColorStop(1, 'rgba(59,130,246,0)');

    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = pl + ((data[i].timestamp - t0) / tRange) * cW;
      const y = pt + cH - (data[i].hashrate / maxH) * cH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    // Stroke the hashrate line
    ctx.strokeStyle = 'rgba(59,130,246,0.9)';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Fill area under hashrate line
    const lastD = data[data.length - 1];
    const lastX = pl + ((lastD.timestamp - t0) / tRange) * cW;
    ctx.lineTo(lastX, pt + cH);
    ctx.lineTo(pl, pt + cH);
    ctx.closePath();
    ctx.fillStyle = hashGrad;
    ctx.fill();

    // Miner count line (green, thinner, no fill)
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = pl + ((data[i].timestamp - t0) / tRange) * cW;
      const y = pt + cH - (data[i].miners / maxM) * cH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(74,222,128,0.8)';
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Hover crosshair + tooltip
    const mX = mouseXRef.current;
    if (mX !== null && mX >= pl && mX <= pl + cW) {
      // Find nearest data point
      const tAtMouse = t0 + ((mX - pl) / cW) * tRange;
      let nearest = 0;
      let nearestDist = Infinity;
      for (let i = 0; i < data.length; i++) {
        const dist = Math.abs(data[i].timestamp - tAtMouse);
        if (dist < nearestDist) {
          nearestDist = dist;
          nearest = i;
        }
      }

      const dp = data[nearest];
      const dpX = pl + ((dp.timestamp - t0) / tRange) * cW;
      const dpYH = pt + cH - (dp.hashrate / maxH) * cH;
      const dpYM = pt + cH - (dp.miners / maxM) * cH;

      // Vertical crosshair
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(148,163,184,0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(dpX, pt);
      ctx.lineTo(dpX, pt + cH);
      ctx.stroke();
      ctx.setLineDash([]);

      // Dots at intersection
      ctx.beginPath();
      ctx.arc(dpX, dpYH, 4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(59,130,246,1)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(dpX, dpYM, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(74,222,128,1)';
      ctx.fill();

      // Tooltip box
      const timeStr = new Date(dp.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const hashStr = fmtHash(dp.hashrate);
      const minerStr = `${dp.miners} miners`;
      const boxW = 120;
      const boxH = 52;
      let bx = dpX + 10;
      if (bx + boxW > pl + cW) bx = dpX - boxW - 10;
      const by = pt + 10;

      ctx.fillStyle = 'rgba(15,23,42,0.92)';
      ctx.strokeStyle = 'rgba(59,130,246,0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(bx, by, boxW, boxH, 6);
      ctx.fill();
      ctx.stroke();

      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillStyle = 'rgba(148,163,184,0.7)';
      ctx.fillText(timeStr, bx + 8, by + 6);
      ctx.fillStyle = 'rgba(96,165,250,0.9)';
      ctx.fillText(hashStr, bx + 8, by + 20);
      ctx.fillStyle = 'rgba(74,222,128,0.9)';
      ctx.fillText(minerStr, bx + 8, by + 34);
    }

    // Axis labels
    ctx.save();
    ctx.translate(12, pt + cH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(96,165,250,0.5)';
    ctx.textAlign = 'center';
    ctx.fillText('Hash Rate', 0, 0);
    ctx.restore();

    ctx.save();
    ctx.translate(W - 8, pt + cH / 2);
    ctx.rotate(Math.PI / 2);
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(74,222,128,0.5)';
    ctx.textAlign = 'center';
    ctx.fillText('Miners', 0, 0);
    ctx.restore();

    animRef.current = requestAnimationFrame(drawChart);
  }, []);

  // Start/stop animation loop
  useEffect(() => {
    if (!isOpen) return;
    animRef.current = requestAnimationFrame(drawChart);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isOpen, drawChart]);

  // Mouse tracking for canvas hover
  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    mouseXRef.current = e.clientX - rect.left;
  }, []);

  const handleCanvasMouseLeave = useCallback(() => {
    mouseXRef.current = null;
  }, []);

  // ── Escape key to close ──
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  // Source badge color
  const sourceBadge = (source: string) => {
    switch (source) {
      case 'local': return 'bg-quantum-green/20 text-quantum-green';
      case 'p2p':   return 'bg-quantum-cyan/20 text-quantum-cyan';
      case 'peer':  return 'bg-quantum-purple/20 text-quantum-purple';
      default:      return 'bg-white/10 text-gray-400';
    }
  };

  // ── Render ──
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
            onClick={onClose}
          />

          {/* Modal panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="fixed inset-y-0 right-0 z-50 w-full max-w-[1200px] bg-gradient-to-bl from-[#0a0f1e] to-[#030818] border-l border-quantum-cyan/20 shadow-2xl overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
              <div className="flex items-center gap-3">
                <Activity className="w-5 h-5 text-quantum-cyan" />
                <h2 className="text-lg font-bold text-white">Network Power</h2>
                <span className="text-xs bg-quantum-cyan/15 text-quantum-cyan px-2 py-0.5 rounded-full">
                  {fmtHash(networkHashRate)}
                </span>
                {connectedMiners > 0 && (
                  <span className="text-xs bg-quantum-green/15 text-quantum-green px-2 py-0.5 rounded-full">
                    {connectedMiners} miners
                  </span>
                )}
              </div>
              <button
                onClick={onClose}
                className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* Body: two panels */}
            <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
              {/* Left Panel: Miner List */}
              <div className="flex-1 flex flex-col border-r border-white/5 min-h-0">
                {/* Miner list header */}
                <div className="px-4 pt-4 pb-2 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Users className="w-4 h-4 text-quantum-cyan" />
                    <span className="text-sm font-semibold text-white">
                      {totalMinerCount > 0 ? totalMinerCount : sortedMiners.length} miners active
                    </span>
                  </div>
                </div>

                {/* Search */}
                <div className="px-4 pb-3">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <input
                      type="text"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Filter by address or worker name..."
                      className="w-full pl-9 pr-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan/50"
                    />
                  </div>
                </div>

                {/* Column headers */}
                <div className="grid grid-cols-[40px_1fr_100px_70px_70px_50px] gap-1 px-4 pb-1 text-[10px] text-gray-500 uppercase tracking-wider select-none">
                  <span>#</span>
                  <span>Miner</span>
                  <button
                    onClick={() => handleSort('hash_rate')}
                    className="text-left hover:text-gray-300 transition-colors"
                  >
                    Hash Rate{sortArrow('hash_rate')}
                  </button>
                  <button
                    onClick={() => handleSort('blocks_found')}
                    className="text-left hover:text-gray-300 transition-colors"
                  >
                    Blocks{sortArrow('blocks_found')}
                  </button>
                  <button
                    onClick={() => handleSort('last_seen_secs_ago')}
                    className="text-left hover:text-gray-300 transition-colors"
                  >
                    Seen{sortArrow('last_seen_secs_ago')}
                  </button>
                  <span>Src</span>
                </div>

                {/* Scrollable miner rows */}
                <div className="flex-1 overflow-y-auto min-h-0 px-2">
                  {sortedMiners.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-32 text-gray-500 text-sm gap-1">
                      {search ? 'No miners match filter' : 'Loading miners...'}
                      {!search && serverMiners.length === 0 && (
                        <span className="text-[10px] text-gray-600">Fetching from /api/v1/mining/miners</span>
                      )}
                    </div>
                  ) : (
                    sortedMiners.map((miner, idx) => (
                      <div
                        key={`${miner.address}-${miner.worker_id}`}
                        className="grid grid-cols-[40px_1fr_100px_70px_70px_50px] gap-1 items-center px-2 py-1.5 rounded-md hover:bg-white/5 transition-colors text-xs"
                      >
                        {/* Rank */}
                        <span
                          className={`font-mono font-bold ${
                            idx < 3 ? RANK_BADGE[idx] : 'text-gray-500'
                          }`}
                        >
                          {idx < 3 ? ['1st', '2nd', '3rd'][idx] : `#${idx + 1}`}
                        </span>

                        {/* Address + worker name */}
                        <div className="flex flex-col truncate">
                          <span className="text-gray-300 font-mono truncate" title={miner.address}>
                            {truncAddr(miner.address)}
                          </span>
                          {miner.worker_name && (
                            <span className="text-[10px] text-gray-600 truncate">{miner.worker_name}</span>
                          )}
                        </div>

                        {/* Hash Rate */}
                        <span className="text-blue-400 font-mono">
                          {fmtHash(miner.hash_rate)}
                        </span>

                        {/* Blocks Found */}
                        <span className="text-gray-400 font-mono">{miner.blocks_found}</span>

                        {/* Last Seen */}
                        <span className="text-gray-500 font-mono text-[10px]">
                          {secsAgo(miner.last_seen_secs_ago)}
                        </span>

                        {/* Source badge */}
                        <span className={`text-[9px] px-1.5 py-0.5 rounded text-center ${sourceBadge(miner.source)}`}>
                          {miner.source}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Right Panel: Hashrate History Chart */}
              <div className="flex-1 flex flex-col min-h-0 p-4">
                <div className="flex items-center gap-2 mb-3">
                  <TrendingUp className="w-4 h-4 text-blue-400" />
                  <span className="text-sm font-semibold text-white">Hashrate History (24h)</span>
                  <div className="flex items-center gap-3 ml-auto text-[10px]">
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-blue-500" />
                      <span className="text-gray-400">Hash Rate</span>
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-green-400" />
                      <span className="text-gray-400">Miners</span>
                    </span>
                  </div>
                </div>

                <div className="flex-1 relative bg-white/[0.02] rounded-lg border border-white/5 overflow-hidden min-h-[200px]">
                  <canvas
                    ref={canvasRef}
                    className="w-full h-full"
                    onMouseMove={handleCanvasMouseMove}
                    onMouseLeave={handleCanvasMouseLeave}
                  />
                </div>

                {/* Summary stats below chart */}
                <div className="grid grid-cols-3 gap-3 mt-3">
                  <div className="bg-white/[0.03] border border-white/5 rounded-lg p-3">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Peak Hashrate</div>
                    <div className="text-sm font-bold text-blue-400">
                      {historyData.length > 0
                        ? fmtHash(Math.max(...historyData.map((d) => d.hashrate)))
                        : '--'}
                    </div>
                  </div>
                  <div className="bg-white/[0.03] border border-white/5 rounded-lg p-3">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Peak Miners</div>
                    <div className="text-sm font-bold text-green-400">
                      {historyData.length > 0
                        ? Math.max(...historyData.map((d) => d.miners))
                        : '--'}
                    </div>
                  </div>
                  <div className="bg-white/[0.03] border border-white/5 rounded-lg p-3">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Current</div>
                    <div className="text-sm font-bold text-quantum-cyan">{fmtHash(networkHashRate)}</div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
