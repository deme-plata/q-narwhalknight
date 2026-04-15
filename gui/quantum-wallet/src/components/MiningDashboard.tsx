import { useEffect, useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Trophy, TrendingUp, Zap, Clock, Award, Sparkles, DollarSign, HelpCircle, Shield } from 'lucide-react';
import { qnkAPI, type MiningRewardEvent, type BalanceUpdateEvent, type MiningStatsEvent, type WalletMiningStats } from '../services/api';
import SecurityBitsVisualization from './SecurityBitsVisualization';
import SecurityFrontierChart from './charts/SecurityFrontierChart';
import NetworkPowerModal from './NetworkPowerModal';

interface MiningStats {
  totalRewards: number;
  blocksFound: number;
  currentBalance: number;
  avgHashRate: number;
  networkHashRate: number; // v1.1.9-beta: Network-wide hashrate
}

// v3.3.4-beta: Individual miner tracking for hash rate breakdown
interface MinerInfo {
  minerId: string;
  workerName: string | null;
  hashRate: number;
  lastSeen: Date;
  blocksFound: number;
  totalRewards: number;
}

interface RewardWithAnimation extends MiningRewardEvent {
  id: string;
  isNew: boolean;
}

export default function MiningDashboard() {
  const [rewards, setRewards] = useState<RewardWithAnimation[]>([]);
  const [stats, setStats] = useState<MiningStats>({
    totalRewards: 0,
    blocksFound: 0,
    currentBalance: 0,
    avgHashRate: 0,
    networkHashRate: 0,
  });
  const [showRewardPopup, setShowRewardPopup] = useState(false);
  const [latestReward, setLatestReward] = useState<MiningRewardEvent | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // v3.3.4-beta: Track individual miners for hash rate breakdown tooltip
  const [miners, setMiners] = useState<Map<string, MinerInfo>>(new Map());
  const [showMinerTooltip, setShowMinerTooltip] = useState(false);
  const [showVdfTooltip, setShowVdfTooltip] = useState(false);
  const [showSecurityTooltip, setShowSecurityTooltip] = useState(false);
  const [showNetworkModal, setShowNetworkModal] = useState(false);
  const [connectedMiners, setConnectedMiners] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number>(0);
  const displayHashRateRef = useRef<number>(0);
  // v8.5.5: Refs for animation-consumed values — prevents useCallback/useEffect churn
  const connectedMinersRef = useRef<number>(0);
  const networkHashRateRef = useRef<number>(0);

  // Use the same wallet as Dashboard - from localStorage
  const [walletAddress, setWalletAddress] = useState('');

  // v3.4.21-beta: Track session rewards separately from total balance
  // This prevents jumps caused by mixing local accumulation with backend absolute values
  const [sessionRewardsTotal, setSessionRewardsTotal] = useState(0);
  const initialBalanceRef = useRef<number | null>(null);

  // Daily earnings calculation
  const [qugPriceUsd, setQugPriceUsd] = useState(0);
  const [blockReward, setBlockReward] = useState(0);
  // v8.2.8: Fetch actual daily emission from API instead of hardcoded constant
  // v9.0.3: Use ACTUAL emission rate (prorated from today's data) instead of theoretical target
  const [dailyEmissionQug, setDailyEmissionQug] = useState(7186.07); // 2,625,000 / 365.25 default
  const [dailyTargetQug, setDailyTargetQug] = useState(7186.07); // theoretical max (for display)
  const [networkCapacityPct, setNetworkCapacityPct] = useState(100);

  // v8.5.5: Keep refs in sync — animation reads refs (no re-render dependency)
  connectedMinersRef.current = connectedMiners;
  networkHashRateRef.current = stats.networkHashRate;

  // ═══════════════════════════════════════════════════════════════
  // v10.2.0: "The VDF Forge" — Mining Engine Visualization (rewrite)
  // 10 concentric VDF rings (101 BLAKE3 stages grouped into 10 rings).
  // Nonces spawn from miners on the outer rim and spiral inward,
  // color-shifting blue→cyan→white→gold. Golden hexagon difficulty
  // target at center. Failed nonces = red flash, solutions = golden
  // supernova with expanding shockwave rings.
  // ═══════════════════════════════════════════════════════════════
  const startMiningEngineCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | null;
    if (!ctx) return;
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);

    // ── Canvas sizing ──
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width || canvas.parentElement?.clientWidth || 800;
    const H = rect.height || canvas.parentElement?.clientHeight || 300;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    const cx = W / 2;
    const cy = H / 2;
    const maxR = Math.min(W, H) * 0.43;
    const RINGS = 10;
    const TAU = Math.PI * 2;

    // ── Ring radii (outer ring 0 → inner ring 9) ──
    const ringRadii: number[] = [];
    for (let i = 0; i < RINGS; i++) {
      ringRadii.push(maxR * (1 - i / RINGS) * 0.86 + maxR * 0.12);
    }
    const coreRadius = maxR * 0.08;

    // ── Miner dots on outer rim ──
    interface MinerDot { angle: number; phase: number; hue: number }
    const buildMiners = (count: number): MinerDot[] => {
      const n = Math.min(Math.max(count, 1), 400);
      return Array.from({ length: n }, (_, i) => ({
        angle: (i / n) * TAU,
        phase: Math.random() * TAU,
        hue: 185 + Math.random() * 50,
      }));
    };
    let minerDots = buildMiners(connectedMinersRef.current);
    let lastMinerCount = connectedMinersRef.current;

    // ── Stage node markers per ring ──
    interface StageNode { angle: number; ring: number }
    const stageNodes: StageNode[] = [];
    for (let ring = 0; ring < RINGS; ring++) {
      const n = 8 + ring * 3;
      for (let j = 0; j < n; j++) {
        stageNodes.push({ angle: (j / n) * TAU + ring * 0.3, ring });
      }
    }

    // ── Nonce pulses spiraling inward ──
    interface Nonce {
      angle: number; progress: number; speed: number;
      spin: number; brightness: number; minerIdx: number;
      trail: { x: number; y: number; alpha: number }[];
    }
    const nonces: Nonce[] = [];
    let spawnTimer = 0;

    // ── Solution burst particles ──
    interface Particle { x: number; y: number; vx: number; vy: number; hue: number; size: number }
    interface Burst { age: number; life: number; particles: Particle[] }
    const bursts: Burst[] = [];

    // ── Background stars (fixed positions) ──
    const bgStars = Array.from({ length: 60 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      size: 0.3 + Math.random() * 1.2,
      phase: Math.random() * TAU,
      speed: 0.005 + Math.random() * 0.015,
    }));

    // ── Per-ring glow intensity ──
    const ringGlow = new Float32Array(RINGS);
    let failGlow = 0;
    let frame = 0;
    let noncesProcessed = 0;
    let lastSolutionFrame = -999;

    // ── Helper: nonce color from progress (0→1) ──
    const nonceColor = (t: number): [number, number, number] => {
      if (t < 0.3) return [230, 88, 58];           // blue
      if (t < 0.55) return [195, 92, 65];           // cyan
      if (t < 0.8) return [180, 45, 85];            // white-ish
      return [42, 100, 72];                         // gold
    };

    // ── Helper: draw hexagon ──
    const drawHexagon = (x: number, y: number, r: number, rot: number) => {
      ctx.beginPath();
      for (let v = 0; v < 6; v++) {
        const a = rot + (v / 6) * TAU;
        const px = x + Math.cos(a) * r;
        const py = y + Math.sin(a) * r;
        v === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.closePath();
    };

    // ── Main animation loop ──
    const tick = () => {
      frame++;

      // Rebuild miners if count changed
      const curMiners = Math.max(connectedMinersRef.current, 1);
      if (curMiners !== lastMinerCount) {
        minerDots = buildMiners(curMiners);
        lastMinerCount = curMiners;
      }

      ctx.clearRect(0, 0, W, H);

      // ─── 1. BACKGROUND — Deep space with nebula clouds ───
      const bg = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR * 1.6);
      bg.addColorStop(0, '#1a0a08');
      bg.addColorStop(0.3, '#0c0318');
      bg.addColorStop(0.6, '#050210');
      bg.addColorStop(1, '#010108');
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      // Background stars
      for (const star of bgStars) {
        const twinkle = Math.sin(frame * star.speed + star.phase) * 0.4 + 0.6;
        ctx.fillStyle = `rgba(200, 210, 255, ${twinkle * 0.7})`;
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.size * twinkle, 0, TAU);
        ctx.fill();
      }

      // Nebula gas clouds (blue-purple left, gold-amber right)
      ctx.save();
      ctx.globalAlpha = 0.12;
      const nebula1 = ctx.createRadialGradient(cx * 0.3, cy * 0.5, 0, cx * 0.3, cy * 0.5, maxR * 0.9);
      nebula1.addColorStop(0, 'rgba(60, 80, 255, 0.5)');
      nebula1.addColorStop(0.5, 'rgba(120, 40, 200, 0.2)');
      nebula1.addColorStop(1, 'transparent');
      ctx.fillStyle = nebula1;
      ctx.fillRect(0, 0, W, H);
      const nebula2 = ctx.createRadialGradient(cx * 1.7, cy * 0.4, 0, cx * 1.7, cy * 0.4, maxR * 0.7);
      nebula2.addColorStop(0, 'rgba(40, 100, 255, 0.4)');
      nebula2.addColorStop(0.6, 'rgba(80, 40, 180, 0.15)');
      nebula2.addColorStop(1, 'transparent');
      ctx.fillStyle = nebula2;
      ctx.fillRect(0, 0, W, H);
      const nebula3 = ctx.createRadialGradient(cx, cy * 1.8, 0, cx, cy * 1.8, maxR * 0.6);
      nebula3.addColorStop(0, 'rgba(80, 50, 180, 0.3)');
      nebula3.addColorStop(1, 'transparent');
      ctx.fillStyle = nebula3;
      ctx.fillRect(0, 0, W, H);
      ctx.restore();

      // Central warm radiance
      ctx.save();
      ctx.globalAlpha = 0.25;
      const warmCore = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR * 0.8);
      warmCore.addColorStop(0, 'rgba(255, 160, 20, 0.6)');
      warmCore.addColorStop(0.3, 'rgba(255, 120, 0, 0.15)');
      warmCore.addColorStop(0.6, 'rgba(200, 60, 0, 0.04)');
      warmCore.addColorStop(1, 'transparent');
      ctx.fillStyle = warmCore;
      ctx.fillRect(0, 0, W, H);
      ctx.restore();

      // Star-burst rays from center
      ctx.save();
      const rayCount = 8;
      for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * TAU + frame * 0.001;
        const rayLen = maxR * (1.0 + Math.sin(frame * 0.015 + i * 1.3) * 0.3);
        const rayGrad = ctx.createLinearGradient(cx, cy,
          cx + Math.cos(angle) * rayLen, cy + Math.sin(angle) * rayLen);
        rayGrad.addColorStop(0, 'rgba(255, 180, 40, 0.18)');
        rayGrad.addColorStop(0.3, 'rgba(255, 140, 0, 0.06)');
        rayGrad.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        const spread = 0.04;
        ctx.lineTo(cx + Math.cos(angle - spread) * rayLen, cy + Math.sin(angle - spread) * rayLen);
        ctx.lineTo(cx + Math.cos(angle + spread) * rayLen, cy + Math.sin(angle + spread) * rayLen);
        ctx.closePath();
        ctx.fillStyle = rayGrad;
        ctx.fill();
      }
      ctx.restore();

      // ─── 2. VDF RINGS — Warm gold/amber orbital rings ───
      for (let i = 0; i < RINGS; i++) {
        const r = ringRadii[i];
        const glow = ringGlow[i];
        // Inner rings warmer (gold), outer rings cooler (cyan-blue)
        const hue = i < 4 ? 35 + i * 8 : 180 + (i - 4) * 15;
        const sat = i < 4 ? 100 : 75;
        const lit = i < 4 ? 55 : 50;
        const baseAlpha = 0.12 + (i / RINGS) * 0.1;

        // Main ring stroke
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, TAU);
        ctx.strokeStyle = `hsla(${hue}, ${sat}%, ${lit}%, ${Math.min(baseAlpha + glow * 0.8, 1)})`;
        ctx.lineWidth = 0.8 + glow * 5;
        ctx.stroke();

        // Warm glow aura
        if (glow > 0.04) {
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, TAU);
          ctx.strokeStyle = `hsla(${hue}, 100%, 70%, ${glow * 0.4})`;
          ctx.lineWidth = 8;
          ctx.stroke();
        }

        // Subtle ambient ring glow (always visible)
        ctx.save();
        ctx.globalAlpha = 0.03 + glow * 0.1;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, TAU);
        ctx.strokeStyle = `hsla(${hue}, 80%, 65%, 0.5)`;
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.restore();

        ringGlow[i] *= 0.9;
      }

      // ─── 3. VDF STAGE MARKERS ───
      for (const node of stageNodes) {
        const dir = node.ring % 2 === 0 ? 1 : -1;
        const a = node.angle + frame * 0.0004 * dir;
        const x = cx + Math.cos(a) * ringRadii[node.ring];
        const y = cy + Math.sin(a) * ringRadii[node.ring];
        const glow = ringGlow[node.ring];
        const hue = node.ring < 4 ? 35 + node.ring * 8 : 180 + (node.ring - 4) * 15;
        ctx.fillStyle = `hsla(${hue}, 80%, 72%, ${0.15 + glow * 0.7})`;
        ctx.beginPath();
        ctx.arc(x, y, 1.4 + glow * 2.5, 0, TAU);
        ctx.fill();
      }

      // ─── 4. OUTER MINER RING — Vivid star-like dots ───
      const minerRimR = maxR + 12;
      const minerHues = [200, 220, 280, 320, 40, 180, 160, 30]; // varied colors
      for (let mi = 0; mi < minerDots.length; mi++) {
        const m = minerDots[mi];
        const pulse = Math.sin(frame * 0.025 + m.phase) * 0.35 + 0.65;
        const mx = cx + Math.cos(m.angle) * minerRimR;
        const my = cy + Math.sin(m.angle) * minerRimR;
        const mhue = minerHues[mi % minerHues.length];

        // Outer halo (large, soft)
        const halo = ctx.createRadialGradient(mx, my, 0, mx, my, 7 * pulse);
        halo.addColorStop(0, `hsla(${mhue}, 100%, 80%, ${0.35 * pulse})`);
        halo.addColorStop(0.4, `hsla(${mhue}, 90%, 60%, ${0.1 * pulse})`);
        halo.addColorStop(1, 'transparent');
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(mx, my, 7 * pulse, 0, TAU);
        ctx.fill();

        // Bright core
        ctx.fillStyle = `hsla(${mhue}, 90%, 85%, ${0.8 * pulse})`;
        ctx.beginPath();
        ctx.arc(mx, my, 1.8 * pulse, 0, TAU);
        ctx.fill();

        // White hot center
        ctx.fillStyle = `rgba(255, 255, 255, ${0.6 * pulse})`;
        ctx.beginPath();
        ctx.arc(mx, my, 0.8 * pulse, 0, TAU);
        ctx.fill();
      }

      // ─── 5. SPAWN NONCES ───
      spawnTimer++;
      const spawnRate = Math.max(2, 6 - Math.floor(curMiners / 80));
      if (spawnTimer % spawnRate === 0 && nonces.length < 50) {
        const mIdx = Math.floor(Math.random() * minerDots.length);
        const m = minerDots[mIdx];
        nonces.push({
          angle: m.angle,
          progress: 0,
          speed: 0.003 + Math.random() * 0.005,
          spin: (Math.random() - 0.5) * 0.04,
          brightness: 0.65 + Math.random() * 0.35,
          minerIdx: mIdx,
          trail: [],
        });
      }

      // ─── 6. ANIMATE NONCES (spiral through VDF rings) ───
      for (let i = nonces.length - 1; i >= 0; i--) {
        const n = nonces[i];
        n.progress += n.speed;

        // Spiral position
        const curAngle = n.angle + n.progress * n.spin * 140;
        const curR = maxR * (1 - n.progress) * 0.86 + maxR * 0.12;
        const px = cx + Math.cos(curAngle) * curR;
        const py = cy + Math.sin(curAngle) * curR;

        // Store trail point
        n.trail.push({ x: px, y: py, alpha: 0.6 });
        if (n.trail.length > 12) n.trail.shift();

        // Flash VDF rings on crossing
        for (let ring = 0; ring < RINGS; ring++) {
          if (Math.abs(curR - ringRadii[ring]) < 5) {
            ringGlow[ring] = Math.max(ringGlow[ring], 0.4);
          }
        }

        // Color from progress
        const [h, s, l] = nonceColor(n.progress);

        // Draw trail
        for (let t = 0; t < n.trail.length - 1; t++) {
          const tp = n.trail[t];
          const frac = t / n.trail.length;
          tp.alpha *= 0.88;
          ctx.fillStyle = `hsla(${h}, ${s}%, ${l}%, ${tp.alpha * frac * n.brightness * 0.4})`;
          ctx.beginPath();
          ctx.arc(tp.x, tp.y, 1 + frac * 1.5, 0, TAU);
          ctx.fill();
        }

        // Pulse glow aura
        const glowSize = 6 + n.progress * 8;
        const grad = ctx.createRadialGradient(px, py, 0, px, py, glowSize);
        grad.addColorStop(0, `hsla(${h}, ${s}%, ${l}%, ${0.65 * n.brightness})`);
        grad.addColorStop(0.45, `hsla(${h}, ${s}%, ${l}%, ${0.12 * n.brightness})`);
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(px, py, glowSize, 0, TAU);
        ctx.fill();

        // Bright core
        ctx.fillStyle = `hsla(${h}, ${s}%, ${Math.min(l + 25, 100)}%, ${0.9 * n.brightness})`;
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, TAU);
        ctx.fill();

        // Faint beam to source miner (early progress only)
        if (n.progress < 0.15) {
          const m = minerDots[n.minerIdx];
          const mx = cx + Math.cos(m.angle) * minerRimR;
          const my = cy + Math.sin(m.angle) * minerRimR;
          ctx.beginPath();
          ctx.moveTo(mx, my);
          ctx.lineTo(px, py);
          ctx.strokeStyle = `hsla(${m.hue}, 80%, 65%, ${0.06 * (1 - n.progress * 7)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }

        // ─── Reached center → difficulty check ───
        if (n.progress >= 1.0) {
          nonces.splice(i, 1);
          noncesProcessed++;

          const isSolution = noncesProcessed % 35 === 0 ||
            (frame - lastSolutionFrame > 350 && Math.random() < 0.04);

          if (isSolution) {
            // ★ SOLUTION — golden supernova burst ★
            lastSolutionFrame = frame;
            const particles: Particle[] = [];
            for (let j = 0; j < 45; j++) {
              const a = Math.random() * TAU;
              const spd = 1.5 + Math.random() * 4;
              particles.push({
                x: cx, y: cy,
                vx: Math.cos(a) * spd, vy: Math.sin(a) * spd,
                hue: 25 + Math.random() * 35,
                size: 1 + Math.random() * 3,
              });
            }
            bursts.push({ age: 0, life: 80, particles });
            // Flash all rings gold
            for (let r = 0; r < RINGS; r++) ringGlow[r] = 0.8;
          } else {
            // Failed nonce → red flash at core
            failGlow = 0.55;
          }
        }
      }

      // ─── 7. GOLDEN HEXAGON DIFFICULTY TARGET — Intense radiant core ───
      const corePulse = Math.sin(frame * 0.035) * 0.2 + 0.8;
      const cSize = coreRadius * corePulse;

      // Wide golden radiance (large soft glow)
      const wideGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, cSize * 8);
      wideGlow.addColorStop(0, `rgba(255, 200, 40, ${0.4 * corePulse})`);
      wideGlow.addColorStop(0.15, `rgba(255, 160, 0, ${0.2 * corePulse})`);
      wideGlow.addColorStop(0.4, `rgba(255, 100, 0, ${0.06 * corePulse})`);
      wideGlow.addColorStop(1, 'transparent');
      ctx.fillStyle = wideGlow;
      ctx.beginPath();
      ctx.arc(cx, cy, cSize * 8, 0, TAU);
      ctx.fill();

      // Core intense glow
      const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, cSize * 4);
      coreGrad.addColorStop(0, `rgba(255, 240, 180, ${(0.7 + failGlow * 0.3) * corePulse})`);
      coreGrad.addColorStop(0.2, `rgba(255, 200, 40, ${0.5 * corePulse})`);
      coreGrad.addColorStop(0.5, `rgba(255, 140, 0, ${0.15 * corePulse})`);
      coreGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, cSize * 4, 0, TAU);
      ctx.fill();

      // Failed-nonce red flash
      if (failGlow > 0.02) {
        ctx.fillStyle = `rgba(255, 55, 35, ${failGlow * 0.5})`;
        ctx.beginPath();
        ctx.arc(cx, cy, cSize * 3, 0, TAU);
        ctx.fill();
        failGlow *= 0.87;
      }

      // Outer rotating golden hexagon (bright, thick)
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(frame * 0.005);
      drawHexagon(0, 0, cSize * 3, 0);
      ctx.strokeStyle = `rgba(255, 200, 40, ${0.7 * corePulse})`;
      ctx.lineWidth = 2;
      ctx.stroke();
      // Hexagon glow
      drawHexagon(0, 0, cSize * 3, 0);
      ctx.strokeStyle = `rgba(255, 160, 0, ${0.2 * corePulse})`;
      ctx.lineWidth = 6;
      ctx.stroke();
      ctx.restore();

      // Middle hexagon (counter-rotate)
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-frame * 0.008);
      drawHexagon(0, 0, cSize * 1.8, Math.PI / 6);
      ctx.strokeStyle = `rgba(255, 220, 80, ${0.5 * corePulse})`;
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.restore();

      // Inner filled hexagon (solid golden core)
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(frame * 0.003);
      drawHexagon(0, 0, cSize * 1.0, 0);
      const hexFill = ctx.createRadialGradient(0, 0, 0, 0, 0, cSize * 1.0);
      hexFill.addColorStop(0, `rgba(255, 240, 200, ${0.9 * corePulse})`);
      hexFill.addColorStop(0.5, `rgba(255, 180, 20, ${0.6 * corePulse})`);
      hexFill.addColorStop(1, `rgba(255, 140, 0, ${0.3 * corePulse})`);
      ctx.fillStyle = hexFill;
      ctx.fill();
      ctx.restore();

      // ─── 8. SOLUTION BURSTS ───
      for (let i = bursts.length - 1; i >= 0; i--) {
        const b = bursts[i];
        b.age++;
        if (b.age > b.life) { bursts.splice(i, 1); continue; }
        const prog = b.age / b.life;

        // Primary shockwave ring
        const waveR = prog * maxR * 1.4;
        const waveAlpha = (1 - prog) * 0.5;
        ctx.beginPath();
        ctx.arc(cx, cy, waveR, 0, TAU);
        ctx.strokeStyle = `rgba(255, 184, 0, ${waveAlpha})`;
        ctx.lineWidth = 3 * (1 - prog);
        ctx.stroke();

        // Secondary shockwave (delayed)
        if (prog > 0.12) {
          const wave2R = (prog - 0.12) * maxR * 1.3;
          ctx.beginPath();
          ctx.arc(cx, cy, wave2R, 0, TAU);
          ctx.strokeStyle = `rgba(255, 220, 80, ${(1 - prog) * 0.22})`;
          ctx.lineWidth = 1.8 * (1 - prog);
          ctx.stroke();
        }

        // Particles
        for (const pt of b.particles) {
          pt.x += pt.vx;
          pt.y += pt.vy;
          pt.vx *= 0.97;
          pt.vy *= 0.97;
          const alpha = (1 - prog) * 0.85;
          const sz = pt.size * (1 - prog * 0.35);

          // Particle glow
          ctx.fillStyle = `hsla(${pt.hue}, 100%, 72%, ${alpha})`;
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, sz, 0, TAU);
          ctx.fill();

          // Motion trail
          ctx.fillStyle = `hsla(${pt.hue}, 100%, 55%, ${alpha * 0.25})`;
          ctx.beginPath();
          ctx.arc(pt.x - pt.vx * 2.5, pt.y - pt.vy * 2.5, sz * 0.5, 0, TAU);
          ctx.fill();
        }
      }

      // ─── 9. STATS OVERLAY ───
      ctx.textBaseline = 'middle';

      // Center: miner count inside hexagon
      ctx.textAlign = 'center';
      ctx.font = 'bold 24px "SF Mono", "Fira Code", "Cascadia Code", monospace';
      ctx.fillStyle = '#ffffff';
      ctx.shadowColor = 'rgba(255, 184, 0, 0.7)';
      ctx.shadowBlur = 14;
      ctx.fillText(`${connectedMinersRef.current || curMiners}`, cx, cy - 5);
      ctx.shadowBlur = 0;
      ctx.font = '9px system-ui, -apple-system, sans-serif';
      ctx.fillStyle = 'rgba(255, 184, 0, 0.9)';
      ctx.fillText('MINERS', cx, cy + 13);

      // Top-left: Algorithm
      ctx.textAlign = 'left';
      ctx.font = 'bold 12px "SF Mono", "Fira Code", monospace';
      ctx.fillStyle = 'rgba(120, 160, 255, 0.92)';
      ctx.fillText('BLAKE3 \u00D7 101 VDF', 14, 20);
      ctx.font = '10px system-ui, sans-serif';
      ctx.fillStyle = 'rgba(120, 160, 255, 0.55)';
      ctx.fillText('Sequential Proof-of-Work', 14, 34);

      // Top-right: Network hashrate
      ctx.textAlign = 'right';
      const netHR = networkHashRateRef.current;
      const khs = netHR / 1000;
      let hrLabel: string;
      if (khs >= 1e6) hrLabel = `${(khs / 1e6).toFixed(2)} TH/s`;
      else if (khs >= 1000) hrLabel = `${(khs / 1000).toFixed(1)} GH/s`;
      else if (khs >= 1) hrLabel = `${khs.toFixed(0)} MH/s`;
      else hrLabel = `${netHR.toFixed(0)} KH/s`;
      ctx.font = 'bold 13px "SF Mono", "Fira Code", monospace';
      ctx.fillStyle = 'rgba(0, 230, 200, 0.92)';
      ctx.fillText(hrLabel, W - 14, 20);
      ctx.font = '10px system-ui, sans-serif';
      ctx.fillStyle = 'rgba(0, 230, 200, 0.55)';
      ctx.fillText('Network Hashrate', W - 14, 34);

      // Bottom-left: Your share
      ctx.textAlign = 'left';
      const yourPct = netHR > 0
        ? ((displayHashRateRef.current / netHR) * 100).toFixed(2) : '0.00';
      ctx.font = 'bold 11px system-ui, sans-serif';
      ctx.fillStyle = 'rgba(180, 140, 255, 0.82)';
      ctx.fillText(`Your share: ${yourPct}%`, 14, H - 20);

      // Bottom-right: VDF pipeline label
      ctx.textAlign = 'right';
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillStyle = 'rgba(255, 184, 0, 0.6)';
      ctx.fillText('nonce \u2192 101\u00D7 BLAKE3 \u2192 target', W - 14, H - 20);

      animFrameRef.current = requestAnimationFrame(tick);
    };

    tick();
  }, []);

  // v3.4.21-beta: Fetch authoritative balance from API
  const fetchBalance = async () => {
    if (!walletAddress) return;
    try {
      const balanceResponse = await qnkAPI.getWalletBalance(walletAddress);
      if (balanceResponse.success && balanceResponse.data) {
        const balance = balanceResponse.data.balance_qnk || 0;
        console.log('💰 [MiningDashboard] Fetched authoritative balance from API:', balance);

        // Store initial balance on first fetch
        if (initialBalanceRef.current === null) {
          initialBalanceRef.current = balance;
          console.log('💰 [MiningDashboard] Set initial balance reference:', balance);
        }

        setStats(prev => ({
          ...prev,
          currentBalance: balance,
          totalRewards: balance, // Total rewards = current balance (authoritative from API)
        }));
      }
    } catch (error) {
      console.error('Failed to fetch balance:', error);
    }
  };

  // v3.5.0-beta: Fetch mining stats from backend (blocks found, hash rate)
  // This allows stats to survive page refresh instead of resetting to 0
  const fetchMiningStats = async () => {
    if (!walletAddress) return;
    try {
      const miningStatsResponse = await qnkAPI.getMiningStats(walletAddress);
      if (miningStatsResponse.success && miningStatsResponse.data) {
        const serverStats = miningStatsResponse.data;
        console.log('⛏️ [MiningDashboard] Fetched mining stats from server:', serverStats);

        setStats(prev => ({
          ...prev,
          blocksFound: serverStats.blocks_found,
          avgHashRate: serverStats.hash_rate, // KH/s from server
        }));

        // v7.4.2: Populate miners map from REST response (preserves names across refresh)
        // v7.4.5: Use server's hash_rate directly (0 for inactive workers) instead of falling back to stale local value
        if (serverStats.workers && serverStats.workers.length > 0) {
          setMiners(prev => {
            const newMiners = new Map(prev);
            for (const worker of serverStats.workers!) {
              const minerId = worker.worker_id || 'unknown';
              const existing = newMiners.get(minerId);
              newMiners.set(minerId, {
                minerId,
                workerName: worker.worker_name || existing?.workerName || null,
                hashRate: worker.hash_rate, // Trust server value — 0 means offline
                lastSeen: existing?.lastSeen || new Date(),
                blocksFound: worker.blocks_found || existing?.blocksFound || 0,
                totalRewards: existing?.totalRewards || 0,
              });
            }
            return newMiners;
          });
        }

        // Only log if there are actual mining stats
        if (serverStats.blocks_found > 0 || serverStats.hash_rate > 0) {
          console.log(`⛏️ [MiningDashboard] Restored: ${serverStats.blocks_found} blocks, ${serverStats.hash_rate.toFixed(2)} KH/s, ${serverStats.workers?.length || 0} workers`);
        }
      }
    } catch (error) {
      console.error('Failed to fetch mining stats:', error);
    }
  };

  // Get wallet address from localStorage (same as Dashboard)
  useEffect(() => {
    const storedWallet = localStorage.getItem('walletAddress');
    if (storedWallet) {
      setWalletAddress(storedWallet);
      console.log('✅ Mining Dashboard tracking wallet:', storedWallet);
    } else {
      console.warn('⚠️ No wallet found in localStorage - user needs to login/create wallet first');
    }
  }, []);

  useEffect(() => {
    if (!walletAddress) {
      console.warn('⚠️ No wallet address found for mining dashboard');
      return;
    }

    console.log('🔌 Connecting to SSE for wallet:', walletAddress);

    // v3.4.21-beta: Fetch initial balance from API (single source of truth)
    fetchBalance();

    // v3.5.0-beta: Fetch mining stats (blocks found, hash rate) from backend
    // This restores stats on page refresh instead of starting from 0
    fetchMiningStats();

    // v3.4.21-beta: Refresh balance periodically to stay in sync with backend
    const balanceRefreshInterval = setInterval(fetchBalance, 10000); // Every 10s

    // v3.5.0-beta: Refresh mining stats periodically (every 30s)
    const miningStatsInterval = setInterval(fetchMiningStats, 30000);

    // v1.1.9-beta: Fetch network hashrate on load and periodically
    fetchNetworkHashrate();
    const networkHashrateInterval = setInterval(fetchNetworkHashrate, 30000); // Every 30s

    // Fetch QUG price, block reward, and actual daily emission for earnings calculation
    const fetchEarningsData = async () => {
      try {
        const [priceRes, challengeRes, emissionRes] = await Promise.all([
          fetch('/api/v1/oracle/price/QUG').catch(() => null),
          fetch('/api/v1/mining/challenge').catch(() => null),
          fetch('/api/v1/emission/stats?days=1').catch(() => null),
        ]);
        if (priceRes?.ok) {
          const json = await priceRes.json();
          const price = json.data?.price_usd || json.data?.price || 0;
          if (price > 0 && price < 1_000_000) setQugPriceUsd(price);
        }
        if (challengeRes?.ok) {
          const json = await challengeRes.json();
          const reward = json.data?.block_reward || 0;
          if (reward > 0) setBlockReward(reward);
        }
        // v9.0.3: Use ACTUAL emission rate from today's data, not theoretical target.
        // The theoretical target assumes 100% uptime — actual is lower during sync/restarts.
        // This prevents the "Expected 300/day but got 20" confusion.
        if (emissionRes?.ok) {
          const json = await emissionRes.json();
          const dailyTarget = json.data?.summary?.daily_target_qug || 0;
          const todayEmitted = json.data?.summary?.today_emitted_qug || 0;
          if (dailyTarget > 0) {
            setDailyTargetQug(dailyTarget);
          }
          // Calculate actual daily rate from today's emission prorated to 24h
          const now = new Date();
          const hoursElapsedToday = now.getUTCHours() + now.getUTCMinutes() / 60;
          if (hoursElapsedToday > 1 && todayEmitted > 0 && dailyTarget > 0) {
            const projectedDaily = (todayEmitted / hoursElapsedToday) * 24;
            const capacity = Math.min(100, Math.round((projectedDaily / dailyTarget) * 100));
            setDailyEmissionQug(projectedDaily);
            setNetworkCapacityPct(capacity);
            console.log(`📊 Actual emission rate: ${projectedDaily.toFixed(0)} QUG/day (${capacity}% of ${dailyTarget.toFixed(0)} target)`);
          } else if (dailyTarget > 0) {
            // Early in the day or no data — use theoretical target
            setDailyEmissionQug(dailyTarget);
            setNetworkCapacityPct(100);
          }
        }
      } catch { /* endpoints may not be available */ }
    };
    fetchEarningsData();
    const priceInterval = setInterval(fetchEarningsData, 60000);

    // Subscribe to mining rewards via SSE
    const eventSource = qnkAPI.subscribeToMiningRewards(
      walletAddress,
      handleMiningReward,
      handleBalanceUpdate,
      handleMiningStats
    );

    console.log('✅ SSE EventSource created:', eventSource.url);
    eventSourceRef.current = eventSource;

    // Request notification permissions
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    // Cleanup on unmount
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      clearInterval(networkHashrateInterval);
      clearInterval(balanceRefreshInterval);
      clearInterval(miningStatsInterval);
      clearInterval(priceInterval);
    };
  }, [walletAddress]);

  // v7.4.4: Auto-start VDF Forge canvas animation
  // v8.9.3: Use requestAnimationFrame delay to ensure canvas is laid out before reading dimensions
  useEffect(() => {
    // Wait one frame so the motion.div has been laid out by the browser
    const raf = requestAnimationFrame(() => {
      startMiningEngineCanvas();
    });
    return () => {
      cancelAnimationFrame(raf);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [startMiningEngineCanvas]);

  const handleMiningReward = (reward: MiningRewardEvent) => {
    console.log('🎉 Mining reward received:', reward);
    console.log('🎉 Reward details:', {
      miner: reward.miner_address,
      amount: reward.reward_qnk,
      nonce: reward.nonce,
      hash_rate: reward.hash_rate
    });

    // Add to rewards list with animation flag
    const rewardWithId: RewardWithAnimation = {
      ...reward,
      id: `${reward.block_height}-${reward.nonce}`,
      isNew: true,
    };

    setRewards(prev => {
      const updated = [rewardWithId, ...prev].slice(0, 10); // Keep last 10
      return updated;
    });

    // v3.4.21-beta: Track session rewards (small incremental values only)
    // This is separate from total balance which comes from API
    if (reward.reward_qnk > 0 && reward.reward_qnk < 10) { // Sanity check: individual rewards should be < 10 QUG
      setSessionRewardsTotal(prev => prev + reward.reward_qnk);
    }

    // v3.4.21-beta: Update hash rate and blocks found, but NOT balance
    // Balance is fetched from API periodically to stay authoritative
    setStats(prev => ({
      ...prev,
      blocksFound: prev.blocksFound + 1,
      avgHashRate: reward.hash_rate > 0 ? reward.hash_rate : prev.avgHashRate,
    }));

    // v3.4.21-beta: Trigger a balance refresh from API after receiving a reward
    // This ensures we show the authoritative balance, not a locally accumulated one
    setTimeout(() => fetchBalance(), 500);

    // v3.3.4-beta: Track individual miners for hash rate breakdown
    const minerId = reward.miner_id || reward.miner_address.substring(0, 16);
    setMiners(prev => {
      const newMiners = new Map(prev);
      const existing = newMiners.get(minerId);

      newMiners.set(minerId, {
        minerId,
        workerName: reward.worker_name || null,
        hashRate: reward.hash_rate > 0 ? reward.hash_rate : (existing?.hashRate || 0),
        lastSeen: new Date(),
        blocksFound: (existing?.blocksFound || 0) + 1,
        totalRewards: (existing?.totalRewards || 0) + reward.reward_qnk,
      });

      console.log('⛏️ [MiningDashboard] Updated miner tracking:', {
        minerId,
        workerName: reward.worker_name,
        hashRate: reward.hash_rate,
        totalMiners: newMiners.size
      });

      return newMiners;
    });

    // Show reward popup
    setLatestReward(reward);
    setShowRewardPopup(true);
    setTimeout(() => setShowRewardPopup(false), 5000);

    // Show browser notification
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Mining Reward Received!', {
        body: `You earned ${reward.reward_qnk.toFixed(8)} QUG from block #${reward.block_height}`,
        icon: '/quillon-logo.png',
        tag: `mining-reward-${reward.block_height}`,
      });
    }

    // Remove animation flag after animation completes
    setTimeout(() => {
      setRewards(prev =>
        prev.map(r => (r.id === rewardWithId.id ? { ...r, isNew: false } : r))
      );
    }, 1000);
  };

  const handleBalanceUpdate = (update: BalanceUpdateEvent) => {
    // v3.4.21-beta: SIMPLIFIED - Just log and trigger API refresh
    // We no longer try to accumulate locally - API is the single source of truth
    console.log('🔔 [MiningDashboard] handleBalanceUpdate - triggering API refresh:', {
      wallet: update.wallet_address,
      old: update.old_balance,
      new: update.new_balance,
      reason: update.change_reason
    });

    // Check if this is a P2P mining reward (needs to be added to rewards list since no MiningReward event)
    const isP2PMiningReward = update.change_reason === 'p2p_mining_reward' ||
                              update.change_reason === 'pending_mining_reward';

    if (isP2PMiningReward) {
      // P2P mining rewards - add to rewards list for display only
      const rewardAmount = update.new_balance - update.old_balance;

      // Sanity check: individual rewards should be small (< 10 QUG)
      if (rewardAmount > 0 && rewardAmount < 10) {
        console.log('⛏️ [MiningDashboard] P2P Mining reward - adding to list:', rewardAmount);

        const rewardWithId: RewardWithAnimation = {
          id: `p2p-${update.timestamp}-${Math.random()}`,
          miner_address: update.wallet_address,
          reward_qnk: rewardAmount,
          nonce: 0,
          block_height: (update as { block_height?: number }).block_height || 0,
          difficulty: '0',
          hash_rate: 0,
          timestamp: update.timestamp,
          isNew: true,
        };

        setRewards(prev => [rewardWithId, ...prev].slice(0, 10));
        setSessionRewardsTotal(prev => prev + rewardAmount);
        setStats(prev => ({ ...prev, blocksFound: prev.blocksFound + 1 }));

        setTimeout(() => {
          setRewards(prev => prev.map(r => r.id === rewardWithId.id ? { ...r, isNew: false } : r));
        }, 1000);
      }
    }

    // v3.4.21-beta: Trigger API refresh to get authoritative balance
    // This is the ONLY place we update the displayed balance
    fetchBalance();
  };

  const handleMiningStats = (statsUpdate: MiningStatsEvent) => {
    // v3.4.21-beta: SIMPLIFIED - Only update hashrate, let API handle balance
    console.log('📊 [MiningDashboard] Mining stats received:', {
      miner: statsUpdate.miner_address,
      hash_rate: statsUpdate.avg_hash_rate,
      miner_id: statsUpdate.miner_id,
      worker_id: statsUpdate.worker_id
    });

    // Only update hashrate - balance comes from API
    setStats(prev => ({
      ...prev,
      avgHashRate: statsUpdate.avg_hash_rate,
    }));

    // Track individual miners for hash rate breakdown
    if (statsUpdate.worker_id || statsUpdate.miner_id) {
      const minerId = statsUpdate.miner_id || statsUpdate.worker_id || 'unknown';
      setMiners(prev => {
        const newMiners = new Map(prev);
        const existing = newMiners.get(minerId);

        newMiners.set(minerId, {
          minerId,
          workerName: statsUpdate.worker_name || existing?.workerName || null,
          hashRate: statsUpdate.avg_hash_rate,
          lastSeen: new Date(),
          blocksFound: existing?.blocksFound || 0,
          totalRewards: existing?.totalRewards || 0,
        });

        console.log('⛏️ [MiningDashboard] Updated miner hashrate:', {
          minerId,
          hashRate: statsUpdate.avg_hash_rate,
          totalMiners: newMiners.size
        });

        return newMiners;
      });
    }
  };

  // v1.1.9-beta: Fetch network-wide hashrate from /api/v1/network/supply
  const fetchNetworkHashrate = async () => {
    try {
      const response = await fetch('/api/v1/network/supply');
      if (response.ok) {
        const json = await response.json();
        // API returns { success: true, data: { network_hashrate: 186539, ... } }
        const networkHashrate = json.data?.network_hashrate || json.network_hashrate || 0;
        const minersCount = json.data?.connected_miners || 0;
        console.log('🌐 Network hashrate fetched:', networkHashrate, 'H/s, miners:', minersCount);
        setStats(prev => ({
          ...prev,
          networkHashRate: networkHashrate,
        }));
        if (minersCount > 0) setConnectedMiners(minersCount);
      }
    } catch (error) {
      console.error('Failed to fetch network hashrate:', error);
    }
  };

  const formatHashRate = (hashRate: number) => {
    if (hashRate >= 1e9) return `${(hashRate / 1e9).toFixed(2)} GH/s`;
    if (hashRate >= 1e6) return `${(hashRate / 1e6).toFixed(2)} MH/s`;
    if (hashRate >= 1e3) return `${(hashRate / 1e3).toFixed(2)} KH/s`;
    return `${hashRate.toFixed(2)} H/s`;
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  // v3.3.4-beta: Calculate total hash rate from all tracked miners
  const totalMinerHashRate = Array.from(miners.values()).reduce(
    (sum, miner) => sum + miner.hashRate,
    0
  );
  // v3.5.4-beta: Use maximum of API-reported hashrate and SSE-tracked miners
  // SSE events may have hash_rate=0, but API endpoint calculates from solution timestamps
  const displayHashRate = Math.max(totalMinerHashRate, stats.avgHashRate);
  displayHashRateRef.current = displayHashRate;

  if (!walletAddress) {
    return (
      <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-yellow/30 rounded-xl p-6">
        <p className="text-quantum-yellow">
          No mining wallet configured. Please set VITE_DEFAULT_MINING_WALLET in .env or connect your wallet.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Reward Popup */}
      <AnimatePresence>
        {showRewardPopup && latestReward && (
          <motion.div
            initial={{ opacity: 0, y: -50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -50, scale: 0.9 }}
            className="fixed top-20 left-1/2 transform -translate-x-1/2 z-50 max-w-md w-full"
          >
            <div className="bg-gradient-to-r from-quantum-green/20 to-quantum-cyan/20 backdrop-blur-xl border-2 border-quantum-green rounded-2xl p-6 shadow-2xl">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <Sparkles className="w-12 h-12 text-quantum-green animate-pulse" />
                  <div className="absolute inset-0 bg-quantum-green/20 rounded-full animate-ping" />
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold text-quantum-green mb-1">
                    Mining Reward!
                  </h3>
                  <p className="text-white text-lg">
                    +{latestReward.reward_qnk.toFixed(8)} QUG
                  </p>
                  <p className="text-gray-300 text-sm">
                    Block #{latestReward.block_height}
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-cyan/30 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <TrendingUp className="w-6 h-6 text-quantum-cyan" />
            <span className="text-sm text-gray-400">Balance</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.currentBalance.toFixed(4)}
          </div>
          <div className="text-sm text-quantum-cyan">QUG</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-green/30 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <Trophy className="w-6 h-6 text-quantum-green" />
            <span className="text-sm text-gray-400">Total Rewards</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.totalRewards.toFixed(4)}
          </div>
          <div className="text-sm text-quantum-green">QUG</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <Award className="w-6 h-6 text-quantum-purple" />
            <span className="text-sm text-gray-400">Blocks Found</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.blocksFound}
          </div>
          <div className="text-sm text-quantum-purple">Blocks</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-yellow/30 rounded-xl p-6 relative cursor-pointer overflow-visible"
          style={{ zIndex: showMinerTooltip ? 100 : 1 }}
          onMouseEnter={() => setShowMinerTooltip(true)}
          onMouseLeave={() => setShowMinerTooltip(false)}
        >
          <div className="flex items-center justify-between mb-3">
            <Zap className="w-6 h-6 text-quantum-yellow" />
            <span className="text-sm text-gray-400">Your Hash Rate</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {formatHashRate(displayHashRate)}
          </div>
          <div className="text-sm text-quantum-yellow flex items-center gap-2">
            Personal
            {miners.size > 0 && (
              <span className="text-xs bg-quantum-yellow/20 px-2 py-0.5 rounded-full">
                {miners.size} miner{miners.size !== 1 ? 's' : ''}
              </span>
            )}
          </div>

          {/* v3.3.4-beta: Miner List Tooltip */}
          <AnimatePresence>
            {showMinerTooltip && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                className="absolute left-0 right-0 top-full mt-2 bg-quantum-dark/95 backdrop-blur-xl border border-quantum-yellow/40 rounded-xl p-4 shadow-2xl"
                style={{ zIndex: 9999 }}
              >
                <div className="text-sm font-semibold text-quantum-yellow mb-3 flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Active Miners ({miners.size})
                </div>
                {miners.size > 0 ? (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {Array.from(miners.values())
                      .sort((a, b) => b.hashRate - a.hashRate)
                      .map((miner) => (
                        <div
                          key={miner.minerId}
                          className="flex items-center justify-between bg-quantum-indigo/20 rounded-lg p-3 border border-quantum-purple/20"
                        >
                          <div className="flex flex-col">
                            <span className="text-white font-medium">
                              {miner.workerName || `Miner ${miner.minerId.substring(0, 8)}...`}
                            </span>
                            <span className="text-xs text-gray-400">
                              ID: {miner.minerId.substring(0, 12)}...
                            </span>
                          </div>
                          <div className="flex flex-col items-end">
                            <span className="text-quantum-cyan font-bold">
                              {formatHashRate(miner.hashRate)}
                            </span>
                            <span className="text-xs text-gray-400">
                              {miner.blocksFound} block{miner.blocksFound !== 1 ? 's' : ''} | {miner.totalRewards.toFixed(4)} QUG
                            </span>
                          </div>
                        </div>
                      ))}
                  </div>
                ) : (
                  <div className="text-gray-400 text-sm text-center py-4">
                    <div className="mb-2">No miners detected yet</div>
                    <div className="text-xs">Mining rewards will appear here as they are received via SSE</div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* ═══ VDF Mining Engine — Full-Width Visualization ═══ */}
      <div
        className="bg-gradient-to-br from-[#0c0318]/80 to-[#050210]/90 backdrop-blur-xl border border-quantum-cyan/25 rounded-xl overflow-hidden"
        style={{ position: 'relative', height: 300, width: '100%' }}
      >
        <canvas
          ref={canvasRef}
          style={{ width: '100%', height: '100%', display: 'block' }}
        />

        {/* "?" Help icon — reveals educational VDF tooltip */}
        <div
          className="absolute top-3 right-[140px] z-10"
          onMouseEnter={() => setShowVdfTooltip(true)}
          onMouseLeave={() => setShowVdfTooltip(false)}
        >
          <div className="w-6 h-6 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-sm flex items-center justify-center cursor-help transition-colors border border-white/10">
            <HelpCircle className="w-3.5 h-3.5 text-gray-300" />
          </div>
        </div>

        {/* Educational tooltip — VDF mining explained */}
        <AnimatePresence>
          {showVdfTooltip && (
            <motion.div
              initial={{ opacity: 0, y: -8, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.96 }}
              transition={{ duration: 0.2 }}
              className="absolute top-12 right-4 z-50 w-[420px] max-h-[520px] overflow-y-auto bg-[#0d0b1a]/98 backdrop-blur-2xl border border-quantum-cyan/30 rounded-xl p-5 shadow-2xl shadow-black/60"
              onMouseEnter={() => setShowVdfTooltip(true)}
              onMouseLeave={() => setShowVdfTooltip(false)}
            >
              <h3 className="text-sm font-bold text-quantum-cyan mb-3 flex items-center gap-2">
                <Zap className="w-4 h-4 text-quantum-yellow" />
                How Q-NarwhalKnight Mining Works
              </h3>

              <div className="space-y-3 text-xs text-gray-300 leading-relaxed">
                <p>
                  What you're watching is a <span className="text-white font-semibold">real-time visualization of the DAG-Knight VDF mining algorithm</span>.
                  Every dot on the outer ring represents one of the{' '}
                  <span className="text-quantum-cyan font-semibold">{connectedMiners || 0} miners</span>{' '}
                  currently contributing computational power to the Q-NarwhalKnight network.
                </p>

                <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                  <p className="font-semibold text-quantum-yellow mb-1.5">The Algorithm: BLAKE3 &times; 101 VDF</p>
                  <p>
                    Each miner picks a random number called a <span className="text-white font-medium">nonce</span>,
                    combines it with the current block's challenge hash, and feeds it into{' '}
                    <span className="text-white font-medium">BLAKE3</span> — one of the fastest cryptographic hash
                    functions in the world (over 2,100 MB/s on a single core). But here's the twist: the miner
                    doesn't just hash once. It feeds the output back into BLAKE3{' '}
                    <span className="text-quantum-cyan font-semibold">100 more times</span> in a row. That's the{' '}
                    <span className="text-white font-medium">Verifiable Delay Function (VDF)</span> — a chain of
                    101 sequential hash operations that <em>cannot be parallelized or shortcut</em>.
                  </p>
                </div>

                <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                  <p className="font-semibold text-purple-300 mb-1.5">What You're Seeing</p>
                  <p>
                    The <span className="text-purple-300">concentric rings</span> represent the 101 stages of the VDF chain
                    (grouped into 10 visual rings of ~10 iterations each). The glowing pulses spiraling inward are{' '}
                    <span className="text-white font-medium">nonces being tested</span> — each one traveling through all
                    101 BLAKE3 hash stages. Watch how they change color as they progress:{' '}
                    <span className="text-blue-400">blue</span> (initial hash) &rarr;{' '}
                    <span className="text-cyan-400">cyan</span> (mid-chain) &rarr;{' '}
                    <span className="text-white">white</span> (late stages) &rarr;{' '}
                    <span className="text-yellow-400">gold</span> (approaching the target).
                  </p>
                </div>

                <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                  <p className="font-semibold text-yellow-400 mb-1.5">The Difficulty Target</p>
                  <p>
                    The <span className="text-yellow-400">golden hexagon</span> at the center is the{' '}
                    <span className="text-white font-medium">difficulty target</span>. After all 101 hashes, the final
                    result must be <em>numerically smaller</em> than this target — like rolling 101 dice and needing
                    the final result under a certain number. Most nonces fail (you'll see brief{' '}
                    <span className="text-red-400">red flashes</span> at the center). But every few seconds, one nonce
                    beats the target — that's the{' '}
                    <span className="text-yellow-300 font-semibold">golden explosion</span> with shockwaves rippling
                    outward. That miner just found a valid block and earned a QUG reward.
                  </p>
                </div>

                <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                  <p className="font-semibold text-emerald-400 mb-1.5">Why VDF? Closing the ASIC Gap</p>
                  <p>
                    With Bitcoin's SHA-256, a company can build a custom ASIC chip that packs{' '}
                    <span className="text-white font-medium">thousands of hash cores running in parallel</span>,
                    achieving a 10,000&times; advantage over a regular computer. That's because every SHA-256 nonce
                    is independent — you just stamp out more cores and try more nonces simultaneously.
                  </p>
                  <p className="mt-2">
                    Q-NarwhalKnight's VDF chain changes the math. Each of the 100 VDF iterations{' '}
                    <span className="text-white font-medium">depends on the output of the previous one</span> — you
                    can't skip ahead, and you can't run them in parallel. A faster CPU{' '}
                    <span className="text-white font-medium">does</span> complete each 101-step chain faster,
                    which means it tries more nonces per second and earns proportionally more. But the advantage
                    is <span className="text-emerald-400 font-semibold">linear, not exponential</span>: a CPU that
                    hashes 2&times; faster gets 2&times; the hashrate — not 10,000&times;. You can't just bolt on
                    thousands of parallel VDF pipelines the way Bitcoin ASICs bolt on SHA-256 cores, because each
                    pipeline still hits the same sequential bottleneck.
                  </p>
                  <p className="mt-2">
                    The whitepaper quantifies this: BLAKE3's ASIC advantage factor is{' '}
                    <span className="text-emerald-400 font-semibold">&lt;10&times;</span>, compared to{' '}
                    <span className="text-red-400">10,000&times;</span> for SHA-256. And as the chain grows, the
                    VDF depth <em>increases</em> (100 at genesis, 200 at block 10K, 1,100 at block 100K), making
                    the sequential portion asymptotically approach 99.9% of the work. Building an ASIC becomes
                    economically pointless when your million-dollar chip is only 5&times; faster than a $300
                    desktop CPU. That's what keeps mining{' '}
                    <span className="text-emerald-400 font-semibold">accessible to individuals</span>.
                  </p>
                </div>

                <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                  <p className="font-semibold text-cyan-400 mb-1.5">Performance Engineering</p>
                  <p>
                    The miner binary uses <span className="text-white font-medium">zero-allocation hot loops</span> (no
                    memory allocation inside the mining loop — everything fits in 112 bytes of CPU cache),{' '}
                    <span className="text-white font-medium">SIMD vectorization</span> via AVX2/AVX-512 for BLAKE3,{' '}
                    <span className="text-white font-medium">lock-free atomic counters</span> flushed every 1,024 hashes
                    to avoid cache-line bouncing between cores, and{' '}
                    <span className="text-white font-medium">Link-Time Optimization (LTO)</span> that inlines BLAKE3
                    directly into the mining loop across crate boundaries. Combined, these optimizations deliver a{' '}
                    <span className="text-cyan-400 font-semibold">35–85% speedup</span> over naive implementations.
                  </p>
                </div>

                <p className="text-gray-500 text-[10px] mt-2 text-center italic">
                  Read the full technical whitepaper at quillon.xyz/downloads/Q-NarwhalKnight-Miner-Whitepaper.pdf
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ═══ Security Bits Hardening — Full-Width Visualization ═══ */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.45 }}
        className="bg-gradient-to-br from-[#030818]/80 to-[#020210]/90 backdrop-blur-xl border border-cyan-500/25 rounded-xl overflow-hidden relative"
        style={{ height: 340 }}
      >
          {/* Label badge */}
          <div className="absolute top-2 left-3 z-10 flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-black/40 border border-cyan-400/20">
            <Shield className="w-3 h-3 text-cyan-400" />
            <span className="text-[10px] font-bold text-cyan-400/80 tracking-wider">SECURITY HARDENING</span>
          </div>

          {/* "?" Help icon for security tooltip */}
          <div
            className="absolute top-2 right-3 z-10"
            onMouseEnter={() => setShowSecurityTooltip(true)}
            onMouseLeave={() => setShowSecurityTooltip(false)}
          >
            <div className="w-6 h-6 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-sm flex items-center justify-center cursor-help transition-colors border border-white/10">
              <HelpCircle className="w-3.5 h-3.5 text-gray-300" />
            </div>
          </div>

          <SecurityBitsVisualization
            connectedMiners={connectedMiners}
            networkHashRate={stats.networkHashRate / 1000} // Convert H/s to kH/s
            blockHeight={latestReward?.block_height || 0}
            height={340}
          />

          {/* Educational tooltip — Security Bits explained */}
          <AnimatePresence>
            {showSecurityTooltip && (
              <motion.div
                initial={{ opacity: 0, y: -8, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -8, scale: 0.96 }}
                transition={{ duration: 0.2 }}
                className="absolute top-12 right-4 z-50 w-[420px] max-h-[520px] overflow-y-auto bg-[#0d0b1a]/98 backdrop-blur-2xl border border-cyan-400/30 rounded-xl p-5 shadow-2xl shadow-black/60"
                onMouseEnter={() => setShowSecurityTooltip(true)}
                onMouseLeave={() => setShowSecurityTooltip(false)}
              >
                <h3 className="text-sm font-bold text-cyan-400 mb-3 flex items-center gap-2">
                  <Shield className="w-4 h-4 text-cyan-400" />
                  How Network Security Hardens
                </h3>

                <div className="space-y-3 text-xs text-gray-300 leading-relaxed">
                  <p>
                    This visualization shows Q-NarwhalKnight's <span className="text-white font-semibold">real-time security hardening</span> —
                    every miner that joins the network adds <span className="text-cyan-400 font-semibold">bits of computational security</span> to the protocol's
                    cryptographic shield.
                  </p>

                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <p className="font-semibold text-cyan-400 mb-1.5">Security Bits = Hashpower Density</p>
                    <p>
                      In cryptography, <span className="text-white font-medium">"bits of security"</span> measures how much work
                      an attacker would need to break the system. A{' '}
                      <span className="text-yellow-400">128-bit</span> system requires 2<sup>128</sup> operations to defeat —
                      more than all atoms in the universe.
                    </p>
                  </div>

                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <p className="font-semibold text-emerald-400 mb-1.5">The 8 Security Rings</p>
                    <p>
                      Each concentric ring represents a <span className="text-white font-medium">32-bit security layer</span> (8 rings &times; 32 = 256 bits maximum).
                      As miners join, binary digits (<span className="text-cyan-300">0</span>s and <span className="text-cyan-300">1</span>s) stream inward and
                      <span className="text-white font-medium"> lock into the lattice</span>, filling each ring.
                    </p>
                    <div className="mt-2 grid grid-cols-2 gap-1">
                      <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-red-500"></div><span className="text-[10px]">Ring 0: Outer (32-bit)</span></div>
                      <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-orange-500"></div><span className="text-[10px]">Ring 1-2: Mid-outer</span></div>
                      <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-yellow-400"></div><span className="text-[10px]">Ring 3-4: Mid</span></div>
                      <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-cyan-400"></div><span className="text-[10px]">Ring 5-7: Inner (256-bit)</span></div>
                    </div>
                  </div>

                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <p className="font-semibold text-red-400 mb-1.5">Attack Deflection</p>
                    <p>
                      Red arrows represent theoretical <span className="text-red-400 font-medium">51% attacks</span>.
                      Watch them <span className="text-emerald-400 font-medium">deflect off the shield</span> — the stronger
                      your network's security (more locked bits), the harder the deflection.
                      A fully-hardened 256-bit shield makes attacks computationally impossible.
                    </p>
                  </div>

                  <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                    <p className="font-semibold text-purple-400 mb-1.5">Security Tiers</p>
                    <div className="space-y-1 mt-1">
                      <div className="flex justify-between"><span className="text-red-400">VULNERABLE</span><span className="text-gray-500">32-bit (1 miner)</span></div>
                      <div className="flex justify-between"><span className="text-orange-400">WEAK</span><span className="text-gray-500">64-bit (3+ miners)</span></div>
                      <div className="flex justify-between"><span className="text-yellow-400">STRONG</span><span className="text-gray-500">128-bit (10+ miners)</span></div>
                      <div className="flex justify-between"><span className="text-emerald-400">FORTIFIED</span><span className="text-gray-500">192-bit (50+ miners)</span></div>
                      <div className="flex justify-between"><span className="text-cyan-400">FORTRESS</span><span className="text-gray-500">256-bit (100+ miners)</span></div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-lg p-3 border border-cyan-500/10">
                    <p className="font-semibold text-white mb-1">Why This Matters</p>
                    <p>
                      Every miner running the Q-NarwhalKnight node doesn't just earn QUG rewards —
                      they're <span className="text-cyan-400 font-semibold">actively hardening</span> the protocol's security for everyone.
                      The hexagonal shield tessellation at the center represents the{' '}
                      <span className="text-white font-medium">DAG-Knight consensus lattice</span> — each facet strengthens
                      as the network grows. More hashpower = more bits = stronger shield = safer for all users.
                    </p>
                  </div>

                  <p className="text-gray-500 text-[10px] mt-2 text-center italic">
                    Q-NarwhalKnight uses BLAKE3 &times; 101 VDF mining — ASIC-resistant, CPU-friendly, quantum-ready
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
      </motion.div>

      {/* Security Frontier — Progression Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <SecurityFrontierChart />
      </motion.div>

      {/* Network Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Daily Earnings Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="bg-gradient-to-br from-quantum-indigo/40 to-emerald-500/20 backdrop-blur-xl border border-emerald-500/40 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <DollarSign className="w-6 h-6 text-emerald-400" />
            <span className="text-sm text-gray-400">Est. Daily Earnings</span>
          </div>
          {(() => {
            // v9.0.3: Use ACTUAL daily emission rate (prorated from today's data) instead of theoretical.
            // Shows realistic earnings based on current network performance, not best-case scenario.
            const DEV_FEE = 0.01; // 1% dev fee
            const yourShare = stats.networkHashRate > 0 ? displayHashRate / stats.networkHashRate : 0;
            const dailyQug = yourShare * dailyEmissionQug * (1 - DEV_FEE);
            const dailyUsd = dailyQug * qugPriceUsd;
            const maxDailyQug = yourShare * dailyTargetQug * (1 - DEV_FEE);
            const showCapacityWarning = networkCapacityPct < 90;
            return (
              <>
                <div className="text-3xl font-bold text-white mb-1">
                  {dailyUsd > 0 ? `$${dailyUsd.toFixed(2)}` : '$0.00'}
                </div>
                <div className="text-sm text-emerald-400">{dailyQug.toFixed(4)} QUG/day</div>
                {showCapacityWarning && maxDailyQug > 0 && (
                  <div className="text-xs text-yellow-400 mt-1">
                    Max potential: {maxDailyQug.toFixed(2)} QUG/day
                  </div>
                )}
                {showCapacityWarning && (
                  <div className="text-xs text-yellow-500/80 mt-0.5">
                    Network at {networkCapacityPct}% capacity
                  </div>
                )}
                {qugPriceUsd > 0 && (
                  <div className="text-xs text-gray-500 mt-1">@ ${qugPriceUsd.toFixed(2)}/QUG</div>
                )}
                {yourShare > 0 && (
                  <div className="text-xs text-gray-500">Network share: {(yourShare * 100).toFixed(2)}%</div>
                )}
              </>
            );
          })()}
        </motion.div>

        {/* Network Hash Rate Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="bg-gradient-to-br from-quantum-indigo/40 to-quantum-purple/20 backdrop-blur-xl border border-quantum-cyan/40 rounded-xl p-6 cursor-pointer hover:border-quantum-cyan/80 hover:shadow-lg hover:shadow-quantum-cyan/20 transition-all duration-200"
          onClick={() => setShowNetworkModal(true)}
        >
          <div className="flex items-center justify-between mb-3">
            <TrendingUp className="w-6 h-6 text-quantum-cyan" />
            <span className="text-sm text-gray-400">Network Hash Rate</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {formatHashRate(stats.networkHashRate)}
          </div>
          <div className="text-sm text-quantum-cyan flex items-center gap-2">
            Total Network Power
            {connectedMiners > 0 && (
              <span className="text-xs bg-quantum-cyan/20 px-2 py-0.5 rounded-full">
                {connectedMiners} miners
              </span>
            )}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-gradient-to-br from-quantum-indigo/40 to-quantum-green/20 backdrop-blur-xl border border-quantum-green/40 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <Sparkles className="w-6 h-6 text-quantum-green" />
            <span className="text-sm text-gray-400">Your Share</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.networkHashRate > 0
              ? ((stats.avgHashRate / stats.networkHashRate) * 100).toFixed(2)
              : '0.00'}%
          </div>
          <div className="text-sm text-quantum-green">of Network Power</div>
        </motion.div>
      </div>

      {/* Recent Rewards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6"
      >
        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Clock className="w-5 h-5 text-quantum-cyan" />
          Recent Mining Rewards
        </h3>

        {rewards.length === 0 ? (
          <div className="text-center py-8">
            <Zap className="w-12 h-12 text-gray-500 mx-auto mb-3 opacity-50" />
            <p className="text-gray-400">
              Waiting for mining rewards...
            </p>
            <p className="text-gray-500 text-sm mt-1">
              Rewards will appear here in real-time via SSE
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <AnimatePresence>
              {rewards.map((reward) => (
                <motion.div
                  key={reward.id}
                  initial={reward.isNew ? { opacity: 0, x: -20, scale: 0.95 } : false}
                  animate={{ opacity: 1, x: 0, scale: 1 }}
                  exit={{ opacity: 0, x: 20, scale: 0.95 }}
                  className={`bg-quantum-dark/50 rounded-lg p-4 border transition-all ${
                    reward.isNew
                      ? 'border-quantum-green shadow-lg shadow-quantum-green/20'
                      : 'border-quantum-purple/20'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Trophy className="w-4 h-4 text-quantum-green" />
                        <span className="text-quantum-green font-bold">
                          +{reward.reward_qnk.toFixed(8)} QUG
                        </span>
                        {reward.isNew && (
                          <span className="text-xs bg-quantum-green/20 text-quantum-green px-2 py-0.5 rounded-full animate-pulse">
                            NEW
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-300">
                        Block #{reward.block_height} • Nonce: {reward.nonce}
                      </div>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {reward.origin_node_name && (
                          <span className="text-xs bg-quantum-purple/20 text-quantum-purple px-2 py-0.5 rounded">
                            {reward.origin_node_name}
                          </span>
                        )}
                        {reward.worker_name && (
                          <span className="text-xs bg-quantum-cyan/20 text-quantum-cyan px-2 py-0.5 rounded">
                            Miner: {reward.worker_name}
                          </span>
                        )}
                        {reward.miner_id && !reward.worker_name && (
                          <span className="text-xs bg-quantum-yellow/20 text-quantum-yellow px-2 py-0.5 rounded">
                            ID: {reward.miner_id.substring(0, 8)}...
                          </span>
                        )}
                        {reward.miner_id && reward.worker_name && (
                          <span className="text-xs bg-quantum-yellow/20 text-quantum-yellow px-2 py-0.5 rounded opacity-70">
                            [{reward.miner_id.substring(0, 8)}]
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Difficulty: {reward.difficulty}
                        {reward.origin_node_id && (
                          <span className="ml-2 text-gray-600">
                            • Node: {reward.origin_node_id.substring(0, 12)}...
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-400">
                        {formatTime(reward.timestamp)}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}

        {/* Smart Accumulation Tip */}
        {rewards.length >= 2 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 bg-quantum-cyan/10 border border-quantum-cyan/30 rounded-lg p-4"
          >
            <div className="flex items-center gap-2 text-quantum-cyan text-sm">
              <TrendingUp className="w-4 h-4" />
              <span className="font-semibold">Accumulation Rate</span>
            </div>
            <div className="text-gray-300 text-sm mt-2">
              {(() => {
                // Calculate rewards per hour based on recent activity
                const recentRewards = rewards.slice(0, Math.min(5, rewards.length));
                const totalAmount = recentRewards.reduce((sum, r) => sum + r.reward_qnk, 0);
                const firstTime = new Date(recentRewards[recentRewards.length - 1]?.timestamp || Date.now()).getTime();
                const lastTime = new Date(recentRewards[0]?.timestamp || Date.now()).getTime();
                const timeDiffHours = Math.max((lastTime - firstTime) / (1000 * 60 * 60), 0.01);
                const ratePerHour = totalAmount / timeDiffHours;
                const ratePerDay = ratePerHour * 24;

                if (ratePerHour > 0.001) {
                  return (
                    <>
                      At your current hashrate, you're earning approximately{' '}
                      <span className="text-quantum-green font-bold">{ratePerHour.toFixed(4)} QUG/hour</span>
                      {' '}({ratePerDay.toFixed(2)} QUG/day)
                    </>
                  );
                }
                return 'Keep mining to calculate your accumulation rate!';
              })()}
            </div>
            <div className="text-gray-500 text-xs mt-2">
              All mining rewards are credited instantly across the network via P2P propagation
            </div>
          </motion.div>
        )}
      </motion.div>

      {/* Download Miner */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-quantum-green/10 border border-quantum-green/30 rounded-xl p-6"
      >
        <h4 className="text-lg font-bold text-quantum-green mb-3 flex items-center gap-2">
          <Zap className="w-5 h-5" />
          Download Dual-Lane Miner v10.3.5
        </h4>
        <p className="text-gray-300 text-sm mb-4">
          v10.3.5: GPU BLAKE3 + CPU VDF dual-lane mining | 50/50 reward split | Genus-2 Jacobian VDF
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <a
            href="https://quillon.xyz/downloads/q-miner-linux-x64"
            download
            className="flex items-center justify-center gap-2 bg-quantum-green/20 hover:bg-quantum-green/30 border border-quantum-green/50 text-quantum-green font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            Linux x64 (Latest)
          </a>
          <a
            href="https://quillon.xyz/downloads/q-miner-windows-x64.exe"
            download
            className="flex items-center justify-center gap-2 bg-quantum-cyan/20 hover:bg-quantum-cyan/30 border border-quantum-cyan/50 text-quantum-cyan font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            Windows x64
          </a>
          <a
            href="https://quillon.xyz/downloads/q-miner-macos-arm64"
            download
            className="flex items-center justify-center gap-2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/50 text-quantum-purple font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            macOS ARM64
          </a>
          <a
            href="https://quillon.xyz/downloads/q-miner-macos-x64"
            download
            className="flex items-center justify-center gap-2 bg-quantum-yellow/20 hover:bg-quantum-yellow/30 border border-quantum-yellow/50 text-quantum-yellow font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            macOS Intel x64
          </a>
        </div>
        <div className="mt-4 p-4 bg-black/30 rounded-lg space-y-3">
          <div>
            <p className="text-xs text-gray-400 mb-1">Connect to Network (with miner name):</p>
            <code className="text-xs text-quantum-cyan block">
              ./q-miner --wallet {walletAddress.slice(0, 20)}... --server https://quillon.xyz --miner-name "My Rig"
            </code>
          </div>
          <div>
            <p className="text-xs text-gray-400 mb-1">Solo Mining (Local Node):</p>
            <code className="text-xs text-quantum-yellow block">
              ./q-miner --wallet {walletAddress.slice(0, 20)}... --server http://localhost:8080 --miner-name "Local"
            </code>
            <p className="text-xs text-gray-500 mt-1">Start node with: Q_ALLOW_SOLO_MINING=true ./q-api-server</p>
          </div>
        </div>
      </motion.div>

      {/* Mining Tips */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-quantum-purple/10 border border-quantum-purple/30 rounded-xl p-6"
      >
        <h4 className="text-lg font-bold text-quantum-purple mb-3">Mining Tips</h4>
        <ul className="space-y-2 text-gray-300 text-sm">
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Real-time rewards appear instantly via Server-Sent Events (SSE)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Dashboard updates automatically when your miner finds a block</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Browser notifications alert you to new rewards (enable in settings)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Hash rate is calculated from your actual mining performance</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>New optimized miner: 2.5x faster with AVX2 SIMD + CPU core pinning</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-yellow mt-0.5">⚡</span>
            <span>Solo mining: Balance updates in &lt;100ms with AEGIS-256 authenticated rewards</span>
          </li>
        </ul>
      </motion.div>

      {/* v10.3.0: Network Power Modal */}
      <NetworkPowerModal
        isOpen={showNetworkModal}
        onClose={() => setShowNetworkModal(false)}
        networkHashRate={stats.networkHashRate}
        connectedMiners={connectedMiners}
      />
    </div>
  );
}
