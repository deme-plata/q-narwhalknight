import { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Shield, Users, Zap, ChevronDown, ChevronUp } from 'lucide-react';
import { qnkAPI } from '../../services/api';

// ═══════════════════════════════════════════════════════════════
// SecurityFrontierChart v10.2.0 — Rewritten from scratch
//
// Fetches hashpower security data from /api/v1/security/hashpower
// and shows a live chart of security bits over time with tier bands.
// ═══════════════════════════════════════════════════════════════

interface SecuritySnapshot {
  timestamp: number;
  securityBits: number;
  miners: number;
  hashRate: number;
}

const TIERS = [
  { bits: 32,  label: 'VULNERABLE', color: '#ef4444', minMiners: 1 },
  { bits: 64,  label: 'WEAK',       color: '#f97316', minMiners: 3 },
  { bits: 128, label: 'STRONG',     color: '#eab308', minMiners: 10 },
  { bits: 192, label: 'FORTIFIED',  color: '#10b981', minMiners: 50 },
  { bits: 256, label: 'FORTRESS',   color: '#22d3ee', minMiners: 100 },
];

function tierFor(bits: number) {
  for (let i = TIERS.length - 1; i >= 0; i--) {
    if (bits >= TIERS[i].bits) return TIERS[i];
  }
  return TIERS[0];
}

function fmtHash(hps: number): string {
  if (hps >= 1e12) return `${(hps / 1e12).toFixed(2)} TH/s`;
  if (hps >= 1e9) return `${(hps / 1e9).toFixed(2)} GH/s`;
  if (hps >= 1e6) return `${(hps / 1e6).toFixed(2)} MH/s`;
  if (hps >= 1e3) return `${(hps / 1e3).toFixed(2)} kH/s`;
  return `${hps.toFixed(0)} H/s`;
}

export default function SecurityFrontierChart() {
  const [secData, setSecData] = useState<{
    securityBits: number;
    tier: string;
    hashRate: number;
    hashRateFormatted: string;
    cumulativeWork: string;
    blocksProcessed: number;
    collisionResistance: string;
    preimageResistance: string;
    doubleSpendCost: string;
    attackCost: string;
    miners: number;
  } | null>(null);

  const [expanded, setExpanded] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const historyRef = useRef<SecuritySnapshot[]>([]);
  const frameRef = useRef(0);

  // Fetch security data
  const fetchData = useCallback(async () => {
    try {
      const [secRes, supplyRes] = await Promise.all([
        qnkAPI.getHashpowerSecurity(),
        qnkAPI.getNetworkSupply(),
      ]);

      if (secRes.success && secRes.data) {
        const sec = secRes.data;
        const supply = supplyRes.success && supplyRes.data ? supplyRes.data : null;

        const data = {
          securityBits: sec.metrics.security_bits || 0,
          tier: sec.metrics.security_tier || 'unknown',
          hashRate: sec.metrics.network_hashrate || 0,
          hashRateFormatted: (sec.metrics as any).network_hashrate_formatted || fmtHash(sec.metrics.network_hashrate || 0),
          cumulativeWork: sec.metrics.cumulative_work || '0',
          blocksProcessed: sec.metrics.blocks_processed || 0,
          collisionResistance: sec.security_guarantees?.collision_resistance || 'N/A',
          preimageResistance: sec.security_guarantees?.preimage_resistance || 'N/A',
          doubleSpendCost: sec.security_guarantees?.double_spend_cost_usd || 'N/A',
          attackCost: sec.security_guarantees?.['51_percent_attack_cost'] || 'N/A',
          miners: supply?.connected_miners || (sec.metrics as any).connected_peers || 0,
        };
        setSecData(data);

        historyRef.current = [
          ...historyRef.current,
          { timestamp: Date.now(), securityBits: data.securityBits, miners: data.miners, hashRate: data.hashRate },
        ].slice(-60);
      }
    } catch (err) {
      console.error('[SecurityFrontierChart] Fetch error:', err);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 15_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Draw chart
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
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

    frameRef.current++;
    ctx.clearRect(0, 0, W, H);

    const pl = 38, pr = 16, pt = 18, pb = 26;
    const cW = W - pl - pr;
    const cH = H - pt - pb;
    const maxBits = 300;

    // Tier zone bands
    for (let i = 0; i < TIERS.length; i++) {
      const nextBits = i < TIERS.length - 1 ? TIERS[i + 1].bits : maxBits;
      const yTop = pt + cH - (nextBits / maxBits) * cH;
      const yBot = pt + cH - (TIERS[i].bits / maxBits) * cH;

      ctx.fillStyle = `${TIERS[i].color}06`;
      ctx.fillRect(pl, yTop, cW, yBot - yTop);

      // Dashed tier line
      ctx.beginPath();
      ctx.setLineDash([3, 3]);
      ctx.moveTo(pl, yBot);
      ctx.lineTo(pl + cW, yBot);
      ctx.strokeStyle = `${TIERS[i].color}18`;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);

      // Bit label
      ctx.font = '9px sans-serif';
      ctx.fillStyle = `${TIERS[i].color}50`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${TIERS[i].bits}b`, pl - 4, yBot);
    }

    // Y-axis label
    ctx.save();
    ctx.translate(9, pt + cH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(148,163,184,0.35)';
    ctx.textAlign = 'center';
    ctx.fillText('Security Bits', 0, 0);
    ctx.restore();

    // X-axis label
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(148,163,184,0.35)';
    ctx.textAlign = 'center';
    ctx.fillText('Time', pl + cW / 2, H - 4);

    const data = historyRef.current;

    if (data.length < 2) {
      // Single point or no data
      if (data.length === 1) {
        const pt0 = data[0];
        const tier = tierFor(pt0.securityBits);
        const x = pl + cW / 2;
        const y = pt + cH - (pt0.securityBits / maxBits) * cH;

        // Pulsing dot
        const pulse = 0.5 + 0.5 * Math.sin(frameRef.current * 0.05);
        ctx.beginPath();
        ctx.arc(x, y, 6 + pulse * 3, 0, Math.PI * 2);
        ctx.fillStyle = `${tier.color}30`;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = tier.color;
        ctx.fill();

        ctx.font = 'bold 11px sans-serif';
        ctx.fillStyle = '#e2e8f0';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(`${pt0.securityBits.toFixed(1)}-bit`, x, y - 12);
      }

      ctx.font = '11px sans-serif';
      ctx.fillStyle = 'rgba(148,163,184,0.25)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Collecting security frontier data...', pl + cW / 2, pt + cH / 2 + 30);

      animRef.current = requestAnimationFrame(drawChart);
      return;
    }

    // Plot data
    const t0 = data[0].timestamp;
    const tN = data[data.length - 1].timestamp;
    const tRange = tN - t0 || 1;
    const curTier = tierFor(data[data.length - 1].securityBits);

    // Area gradient
    const areaGrad = ctx.createLinearGradient(0, pt, 0, pt + cH);
    areaGrad.addColorStop(0, `${curTier.color}18`);
    areaGrad.addColorStop(1, `${curTier.color}02`);

    // Line + area path
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = pl + ((data[i].timestamp - t0) / tRange) * cW;
      const y = pt + cH - (data[i].securityBits / maxBits) * cH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }

    // Stroke line
    ctx.strokeStyle = curTier.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Fill area
    const lastPt = data[data.length - 1];
    const lastX = pl + ((lastPt.timestamp - t0) / tRange) * cW;
    ctx.lineTo(lastX, pt + cH);
    ctx.lineTo(pl, pt + cH);
    ctx.closePath();
    ctx.fillStyle = areaGrad;
    ctx.fill();

    // Data points
    for (let i = 0; i < data.length; i++) {
      const x = pl + ((data[i].timestamp - t0) / tRange) * cW;
      const y = pt + cH - (data[i].securityBits / maxBits) * cH;
      const tier = tierFor(data[i].securityBits);
      const isLast = i === data.length - 1;

      ctx.beginPath();
      ctx.arc(x, y, isLast ? 4 : 2, 0, Math.PI * 2);
      ctx.fillStyle = isLast ? tier.color : `${tier.color}50`;
      ctx.fill();
    }

    // Current value label
    const lastY = pt + cH - (lastPt.securityBits / maxBits) * cH;
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = curTier.color;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`${lastPt.securityBits.toFixed(1)}-bit`, lastX - 6, lastY - 6);

    animRef.current = requestAnimationFrame(drawChart);
  }, []);

  useEffect(() => {
    animRef.current = requestAnimationFrame(drawChart);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [drawChart]);

  const curTier = secData ? tierFor(secData.securityBits) : TIERS[0];

  return (
    <div className="bg-gradient-to-br from-[#030818]/80 to-[#020210]/90 backdrop-blur-xl border border-cyan-500/25 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-5 pt-4 pb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-bold text-white">Security Frontier</span>
          {secData && (
            <span
              className="text-[10px] font-bold px-2 py-0.5 rounded-full"
              style={{
                background: `${curTier.color}15`,
                color: curTier.color,
                border: `1px solid ${curTier.color}30`,
              }}
            >
              {secData.securityBits.toFixed(1)}-bit {curTier.label}
            </span>
          )}
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="p-1.5 rounded-lg hover:bg-white/5 transition-colors"
        >
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          )}
        </button>
      </div>

      {/* Chart canvas */}
      <div className="px-3 pb-3">
        <canvas
          ref={canvasRef}
          className="w-full rounded-lg"
          style={{ height: 180, display: 'block' }}
        />
      </div>

      {/* Expanded details */}
      {expanded && secData && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="px-5 pb-5 border-t border-white/5"
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
            <div className="bg-white/5 rounded-lg p-3 border border-white/5">
              <span className="text-[10px] text-gray-500 uppercase block">Collision Resistance</span>
              <span className="text-sm font-bold text-cyan-400 mt-1 block">{secData.collisionResistance}</span>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5">
              <span className="text-[10px] text-gray-500 uppercase block">Preimage Resistance</span>
              <span className="text-sm font-bold text-purple-400 mt-1 block">{secData.preimageResistance}</span>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5">
              <span className="text-[10px] text-gray-500 uppercase block">Double-Spend Cost</span>
              <span className="text-sm font-bold text-emerald-400 mt-1 block truncate">{secData.doubleSpendCost}</span>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5">
              <span className="text-[10px] text-gray-500 uppercase block">51% Attack Cost</span>
              <span className="text-sm font-bold text-red-400 mt-1 block truncate">{secData.attackCost}</span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 mt-3">
            <div className="bg-white/5 rounded-lg p-3 border border-white/5 flex items-center gap-2">
              <Shield className="w-4 h-4 text-cyan-400/60" />
              <div>
                <span className="text-[10px] text-gray-500 block">Blocks Processed</span>
                <span className="text-sm font-bold text-white">{secData.blocksProcessed.toLocaleString()}</span>
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5 flex items-center gap-2">
              <Users className="w-4 h-4 text-amber-400/60" />
              <div>
                <span className="text-[10px] text-gray-500 block">Active Miners</span>
                <span className="text-sm font-bold text-white">{secData.miners}</span>
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5 flex items-center gap-2">
              <Zap className="w-4 h-4 text-purple-400/60" />
              <div>
                <span className="text-[10px] text-gray-500 block">Network Hash Rate</span>
                <span className="text-sm font-bold text-white">{secData.hashRateFormatted}</span>
              </div>
            </div>
          </div>

          <div className="mt-3 bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase block mb-1">Cumulative Proof-of-Work</span>
            <span className="text-xs font-mono text-gray-400 break-all">{secData.cumulativeWork}</span>
          </div>
        </motion.div>
      )}
    </div>
  );
}
