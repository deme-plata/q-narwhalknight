import { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Shield, TrendingUp, Users, Zap, ChevronDown, ChevronUp } from 'lucide-react';
import { qnkAPI } from '../../services/api';

// ═══════════════════════════════════════════════════════════════
// SecurityFrontierChart — Shows network security progression
// Fetches hashpower security data from /api/v1/security/hashpower
// and network supply from /api/v1/network/supply, then renders
// a security frontier chart showing the relationship between
// miner count, hash rate, and cryptographic security bits.
// ═══════════════════════════════════════════════════════════════

interface SecuritySnapshot {
  timestamp: number;
  securityBits: number;
  miners: number;
  hashRate: number;
  tier: string;
}

// Security tiers with thresholds
const SECURITY_TIERS = [
  { bits: 32, label: 'VULNERABLE', color: '#ef4444', minMiners: 1 },
  { bits: 64, label: 'WEAK', color: '#f97316', minMiners: 3 },
  { bits: 128, label: 'STRONG', color: '#eab308', minMiners: 10 },
  { bits: 192, label: 'FORTIFIED', color: '#10b981', minMiners: 50 },
  { bits: 256, label: 'FORTRESS', color: '#22d3ee', minMiners: 100 },
];

function getTierForBits(bits: number) {
  for (let i = SECURITY_TIERS.length - 1; i >= 0; i--) {
    if (bits >= SECURITY_TIERS[i].bits) return SECURITY_TIERS[i];
  }
  return SECURITY_TIERS[0];
}

export default function SecurityFrontierChart() {
  const [securityData, setSecurityData] = useState<{
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
  const [history, setHistory] = useState<SecuritySnapshot[]>([]);
  const [expanded, setExpanded] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const historyRef = useRef<SecuritySnapshot[]>([]);

  // Fetch security data
  const fetchSecurityData = useCallback(async () => {
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
          hashRateFormatted: (sec.metrics as any).network_hashrate_formatted || formatHashRate(sec.metrics.network_hashrate || 0),
          cumulativeWork: sec.metrics.cumulative_work || '0',
          blocksProcessed: sec.metrics.blocks_processed || 0,
          collisionResistance: sec.security_guarantees?.collision_resistance || 'N/A',
          preimageResistance: sec.security_guarantees?.preimage_resistance || 'N/A',
          doubleSpendCost: sec.security_guarantees?.double_spend_cost_usd || 'N/A',
          attackCost: sec.security_guarantees?.['51_percent_attack_cost'] || 'N/A',
          miners: supply?.connected_miners || (sec.metrics as any).connected_peers || 0,
        };

        setSecurityData(data);

        // Append to history for chart
        const snapshot: SecuritySnapshot = {
          timestamp: Date.now(),
          securityBits: data.securityBits,
          miners: data.miners,
          hashRate: data.hashRate,
          tier: data.tier,
        };
        setHistory(prev => {
          const updated = [...prev, snapshot].slice(-60); // Keep last 60 data points
          historyRef.current = updated;
          return updated;
        });
      }
    } catch (err) {
      console.error('[SecurityFrontierChart] Fetch error:', err);
    }
  }, []);

  useEffect(() => {
    fetchSecurityData();
    const interval = setInterval(fetchSecurityData, 15_000); // Every 15s
    return () => clearInterval(interval);
  }, [fetchSecurityData]);

  // Draw the frontier chart on canvas
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width || 600;
    const H = rect.height || 200;

    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.scale(dpr, dpr);
    }

    ctx.clearRect(0, 0, W, H);

    const paddingLeft = 40;
    const paddingRight = 20;
    const paddingTop = 20;
    const paddingBottom = 30;
    const chartW = W - paddingLeft - paddingRight;
    const chartH = H - paddingTop - paddingBottom;

    // Draw tier zones (horizontal bands)
    SECURITY_TIERS.forEach((tier, i) => {
      const nextBits = i < SECURITY_TIERS.length - 1 ? SECURITY_TIERS[i + 1].bits : 300;
      const yTop = paddingTop + chartH - (nextBits / 300) * chartH;
      const yBottom = paddingTop + chartH - (tier.bits / 300) * chartH;

      ctx.fillStyle = `${tier.color}08`;
      ctx.fillRect(paddingLeft, yTop, chartW, yBottom - yTop);

      // Tier line
      ctx.beginPath();
      ctx.moveTo(paddingLeft, yBottom);
      ctx.lineTo(paddingLeft + chartW, yBottom);
      ctx.strokeStyle = `${tier.color}20`;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Tier label
      ctx.font = '9px sans-serif';
      ctx.fillStyle = `${tier.color}60`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${tier.bits}b`, paddingLeft - 4, yBottom);
    });

    // Y-axis label
    ctx.save();
    ctx.translate(10, paddingTop + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
    ctx.textAlign = 'center';
    ctx.fillText('Security Bits', 0, 0);
    ctx.restore();

    // X-axis label
    ctx.font = '9px sans-serif';
    ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
    ctx.textAlign = 'center';
    ctx.fillText('Time', paddingLeft + chartW / 2, H - 4);

    const data = historyRef.current;
    if (data.length < 2) {
      // Not enough data - show current point
      if (data.length === 1) {
        const pt = data[0];
        const tier = getTierForBits(pt.securityBits);
        const x = paddingLeft + chartW / 2;
        const y = paddingTop + chartH - (pt.securityBits / 300) * chartH;

        // Pulse circle
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = `${tier.color}80`;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = tier.color;
        ctx.fill();

        ctx.font = 'bold 11px sans-serif';
        ctx.fillStyle = '#e2e8f0';
        ctx.textAlign = 'center';
        ctx.fillText(`${pt.securityBits}-bit`, x, y - 14);
      }

      // "Collecting data" message
      ctx.font = '11px sans-serif';
      ctx.fillStyle = 'rgba(148, 163, 184, 0.3)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Collecting security frontier data...', paddingLeft + chartW / 2, paddingTop + chartH / 2 + 30);

      animRef.current = requestAnimationFrame(drawChart);
      return;
    }

    // Draw the security bits line
    const minTime = data[0].timestamp;
    const maxTime = data[data.length - 1].timestamp;
    const timeRange = maxTime - minTime || 1;

    // Area fill gradient
    const areaGrad = ctx.createLinearGradient(0, paddingTop, 0, paddingTop + chartH);
    const currentTier = getTierForBits(data[data.length - 1].securityBits);
    areaGrad.addColorStop(0, `${currentTier.color}20`);
    areaGrad.addColorStop(1, `${currentTier.color}02`);

    // Build line path
    ctx.beginPath();
    data.forEach((pt, i) => {
      const x = paddingLeft + ((pt.timestamp - minTime) / timeRange) * chartW;
      const y = paddingTop + chartH - (pt.securityBits / 300) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    // Stroke the line
    ctx.strokeStyle = currentTier.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Fill area under line
    const lastPt = data[data.length - 1];
    const lastX = paddingLeft + ((lastPt.timestamp - minTime) / timeRange) * chartW;
    const firstX = paddingLeft;
    ctx.lineTo(lastX, paddingTop + chartH);
    ctx.lineTo(firstX, paddingTop + chartH);
    ctx.closePath();
    ctx.fillStyle = areaGrad;
    ctx.fill();

    // Draw dots at each data point
    data.forEach((pt, i) => {
      const x = paddingLeft + ((pt.timestamp - minTime) / timeRange) * chartW;
      const y = paddingTop + chartH - (pt.securityBits / 300) * chartH;
      const tier = getTierForBits(pt.securityBits);

      ctx.beginPath();
      ctx.arc(x, y, i === data.length - 1 ? 4 : 2, 0, Math.PI * 2);
      ctx.fillStyle = i === data.length - 1 ? tier.color : `${tier.color}60`;
      ctx.fill();
    });

    // Current value label
    {
      const y = paddingTop + chartH - (lastPt.securityBits / 300) * chartH;
      ctx.font = 'bold 11px sans-serif';
      ctx.fillStyle = currentTier.color;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${lastPt.securityBits}-bit`, lastX - 8, y - 6);
    }

    animRef.current = requestAnimationFrame(drawChart);
  }, []);

  useEffect(() => {
    animRef.current = requestAnimationFrame(drawChart);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [drawChart]);

  const currentTier = securityData ? getTierForBits(securityData.securityBits) : SECURITY_TIERS[0];

  return (
    <div
      className="bg-gradient-to-br from-[#030818]/80 to-[#020210]/90 backdrop-blur-xl border border-cyan-500/25 rounded-xl overflow-hidden"
    >
      {/* Header */}
      <div className="px-5 pt-4 pb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-bold text-white">Security Frontier</span>
          {securityData && (
            <span
              className="text-[10px] font-bold px-2 py-0.5 rounded-full"
              style={{
                background: `${currentTier.color}15`,
                color: currentTier.color,
                border: `1px solid ${currentTier.color}30`,
              }}
            >
              {securityData.securityBits}-bit {currentTier.label}
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

      {/* Expanded security details */}
      {expanded && securityData && (
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
              <span className="text-sm font-bold text-cyan-400 mt-1 block">{securityData.collisionResistance}</span>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5">
              <span className="text-[10px] text-gray-500 uppercase block">Preimage Resistance</span>
              <span className="text-sm font-bold text-purple-400 mt-1 block">{securityData.preimageResistance}</span>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5">
              <span className="text-[10px] text-gray-500 uppercase block">Double-Spend Cost</span>
              <span className="text-sm font-bold text-emerald-400 mt-1 block truncate">{securityData.doubleSpendCost}</span>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5">
              <span className="text-[10px] text-gray-500 uppercase block">51% Attack Cost</span>
              <span className="text-sm font-bold text-red-400 mt-1 block truncate">{securityData.attackCost}</span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 mt-3">
            <div className="bg-white/5 rounded-lg p-3 border border-white/5 flex items-center gap-2">
              <Shield className="w-4 h-4 text-cyan-400/60" />
              <div>
                <span className="text-[10px] text-gray-500 block">Blocks Processed</span>
                <span className="text-sm font-bold text-white">{securityData.blocksProcessed.toLocaleString()}</span>
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5 flex items-center gap-2">
              <Users className="w-4 h-4 text-amber-400/60" />
              <div>
                <span className="text-[10px] text-gray-500 block">Active Miners</span>
                <span className="text-sm font-bold text-white">{securityData.miners}</span>
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5 flex items-center gap-2">
              <Zap className="w-4 h-4 text-purple-400/60" />
              <div>
                <span className="text-[10px] text-gray-500 block">Network Hash Rate</span>
                <span className="text-sm font-bold text-white">{securityData.hashRateFormatted}</span>
              </div>
            </div>
          </div>

          {/* Cumulative work */}
          <div className="mt-3 bg-white/5 rounded-lg p-3 border border-white/5">
            <span className="text-[10px] text-gray-500 uppercase block mb-1">Cumulative Proof-of-Work</span>
            <span className="text-xs font-mono text-gray-400 break-all">{securityData.cumulativeWork}</span>
          </div>
        </motion.div>
      )}
    </div>
  );
}

function formatHashRate(hps: number): string {
  if (hps >= 1e12) return `${(hps / 1e12).toFixed(2)} TH/s`;
  if (hps >= 1e9) return `${(hps / 1e9).toFixed(2)} GH/s`;
  if (hps >= 1e6) return `${(hps / 1e6).toFixed(2)} MH/s`;
  if (hps >= 1e3) return `${(hps / 1e3).toFixed(2)} kH/s`;
  return `${hps.toFixed(0)} H/s`;
}
