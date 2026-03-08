import { useRef, useEffect, useCallback } from 'react';

// ═══════════════════════════════════════════════════════════════
// SecurityBitsVisualization — Canvas-based network security display
//
// Renders 8 concentric security rings representing 256-bit security.
// Each ring = 32 bits. Binary digits stream inward and "lock" into the
// lattice as miners join. Attack arrows deflect off the hardened shield.
// A hexagonal shield tessellation forms at the center (DAG-Knight lattice).
//
// Props:
//   connectedMiners: number of active miners on the network
//   networkHashRate: network hash rate in kH/s
//   blockHeight: current blockchain height
//   height: pixel height of the canvas container
// ═══════════════════════════════════════════════════════════════

interface Props {
  connectedMiners: number;
  networkHashRate: number; // kH/s
  blockHeight: number;
  height: number;
}

// Security tier based on miner count
function getSecurityTier(miners: number): {
  tier: string;
  bits: number;
  color: string;
  filledRings: number;
} {
  if (miners >= 100) return { tier: 'FORTRESS', bits: 256, color: '#22d3ee', filledRings: 8 };
  if (miners >= 50) return { tier: 'FORTIFIED', bits: 192, color: '#10b981', filledRings: 6 };
  if (miners >= 10) return { tier: 'STRONG', bits: 128, color: '#eab308', filledRings: 4 };
  if (miners >= 3) return { tier: 'WEAK', bits: 64, color: '#f97316', filledRings: 2 };
  return { tier: 'VULNERABLE', bits: 32, color: '#ef4444', filledRings: 1 };
}

// Ring colors from outer to inner
const RING_COLORS = [
  '#ef4444', // Ring 0 — outer (red)
  '#f97316', // Ring 1
  '#f59e0b', // Ring 2
  '#eab308', // Ring 3
  '#84cc16', // Ring 4
  '#22d3ee', // Ring 5
  '#06b6d4', // Ring 6
  '#0891b2', // Ring 7 — inner (deep cyan)
];

interface StreamingBit {
  angle: number;
  radius: number;
  speed: number;
  value: '0' | '1';
  opacity: number;
  targetRing: number;
  locked: boolean;
  lockAngle: number;
}

interface AttackArrow {
  angle: number;
  radius: number;
  speed: number;
  deflected: boolean;
  deflectAngle: number;
  opacity: number;
  life: number;
}

export default function SecurityBitsVisualization({ connectedMiners, networkHashRate, blockHeight, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const bitsRef = useRef<StreamingBit[]>([]);
  const attacksRef = useRef<AttackArrow[]>([]);
  const frameCountRef = useRef(0);
  const propsRef = useRef({ connectedMiners, networkHashRate, blockHeight });

  propsRef.current = { connectedMiners, networkHashRate, blockHeight };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width || canvas.parentElement?.clientWidth || 800;
    const H = rect.height || canvas.parentElement?.clientHeight || 340;

    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.scale(dpr, dpr);
    }

    const { connectedMiners: miners, networkHashRate: hashRate, blockHeight: bh } = propsRef.current;
    const security = getSecurityTier(miners);
    const frame = frameCountRef.current++;
    const cx = W / 2;
    const cy = H / 2;
    const maxRadius = Math.min(W, H) * 0.42;

    // Clear
    ctx.clearRect(0, 0, W, H);

    // Background radial glow
    const bgGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxRadius * 1.3);
    bgGrad.addColorStop(0, `${security.color}08`);
    bgGrad.addColorStop(0.5, `${security.color}03`);
    bgGrad.addColorStop(1, 'transparent');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);

    // Draw 8 concentric rings
    for (let ring = 0; ring < 8; ring++) {
      const outerR = maxRadius - (ring * maxRadius) / 8;
      const innerR = maxRadius - ((ring + 1) * maxRadius) / 8;
      const isFilled = ring < security.filledRings;
      const ringColor = RING_COLORS[ring];

      // Ring arc
      ctx.beginPath();
      ctx.arc(cx, cy, (outerR + innerR) / 2, 0, Math.PI * 2);
      ctx.strokeStyle = isFilled ? `${ringColor}60` : `${ringColor}15`;
      ctx.lineWidth = (outerR - innerR) * 0.6;
      ctx.stroke();

      // Filled ring glow
      if (isFilled) {
        const ringGrad = ctx.createRadialGradient(cx, cy, innerR, cx, cy, outerR);
        ringGrad.addColorStop(0, `${ringColor}10`);
        ringGrad.addColorStop(0.5, `${ringColor}08`);
        ringGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = ringGrad;
        ctx.beginPath();
        ctx.arc(cx, cy, outerR, 0, Math.PI * 2);
        ctx.arc(cx, cy, innerR, 0, Math.PI * 2, true);
        ctx.fill();

        // Binary digits locked in this ring
        const bitsInRing = 32;
        for (let b = 0; b < bitsInRing; b++) {
          const angle = (b / bitsInRing) * Math.PI * 2 + (ring * 0.3) + (frame * 0.0003 * (ring + 1));
          const r = (outerR + innerR) / 2;
          const x = cx + Math.cos(angle) * r;
          const y = cy + Math.sin(angle) * r;
          const pulse = 0.4 + 0.6 * Math.abs(Math.sin(frame * 0.02 + b * 0.2));

          ctx.font = `${Math.max(7, (outerR - innerR) * 0.35)}px monospace`;
          ctx.fillStyle = `${ringColor}${Math.floor(pulse * 180).toString(16).padStart(2, '0')}`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(b % 2 === 0 ? '1' : '0', x, y);
        }
      }

      // Ring label (bits)
      const labelAngle = -Math.PI / 2 + 0.15;
      const labelR = (outerR + innerR) / 2;
      const labelX = cx + Math.cos(labelAngle) * labelR;
      const labelY = cy + Math.sin(labelAngle) * labelR;
      ctx.font = `bold ${Math.max(8, (outerR - innerR) * 0.3)}px monospace`;
      ctx.fillStyle = isFilled ? `${ringColor}90` : `${ringColor}25`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${(ring + 1) * 32}`, labelX, labelY);
    }

    // Streaming bits (flying inward)
    const bits = bitsRef.current;

    // Spawn new bits
    if (frame % 3 === 0 && bits.length < 60) {
      const angle = Math.random() * Math.PI * 2;
      bits.push({
        angle,
        radius: maxRadius * 1.2 + Math.random() * 30,
        speed: 0.4 + Math.random() * 0.8,
        value: Math.random() > 0.5 ? '1' : '0',
        opacity: 0.8,
        targetRing: Math.floor(Math.random() * security.filledRings),
        locked: false,
        lockAngle: angle,
      });
    }

    // Update and draw bits
    for (let i = bits.length - 1; i >= 0; i--) {
      const bit = bits[i];
      const targetR = maxRadius - ((bit.targetRing + 0.5) * maxRadius) / 8;

      if (!bit.locked) {
        bit.radius -= bit.speed;
        bit.angle += 0.005;

        if (bit.radius <= targetR + 3) {
          bit.locked = true;
          bit.lockAngle = bit.angle;
          bit.radius = targetR;
        }
      } else {
        bit.opacity -= 0.02;
        if (bit.opacity <= 0) {
          bits.splice(i, 1);
          continue;
        }
      }

      const x = cx + Math.cos(bit.angle) * bit.radius;
      const y = cy + Math.sin(bit.angle) * bit.radius;
      const ringColor = RING_COLORS[bit.targetRing] || '#22d3ee';

      ctx.font = `bold 10px monospace`;
      ctx.fillStyle = bit.locked
        ? `${ringColor}${Math.floor(bit.opacity * 255).toString(16).padStart(2, '0')}`
        : `rgba(255, 255, 255, ${bit.opacity * 0.7})`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(bit.value, x, y);

      // Trail for streaming bits
      if (!bit.locked) {
        ctx.beginPath();
        const trailX = cx + Math.cos(bit.angle - 0.02) * (bit.radius + 8);
        const trailY = cy + Math.sin(bit.angle - 0.02) * (bit.radius + 8);
        ctx.moveTo(trailX, trailY);
        ctx.lineTo(x, y);
        ctx.strokeStyle = `rgba(255, 255, 255, ${bit.opacity * 0.15})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // Attack arrows (red, deflecting off shield)
    const attacks = attacksRef.current;

    // Spawn attack arrows occasionally
    if (frame % 120 === 0 && attacks.length < 3) {
      const angle = Math.random() * Math.PI * 2;
      attacks.push({
        angle,
        radius: maxRadius * 1.5,
        speed: 1.5 + Math.random() * 1,
        deflected: false,
        deflectAngle: 0,
        opacity: 1,
        life: 0,
      });
    }

    // Update and draw attacks
    for (let i = attacks.length - 1; i >= 0; i--) {
      const atk = attacks[i];
      atk.life++;

      const shieldRadius = maxRadius - (security.filledRings * maxRadius) / 8;

      if (!atk.deflected) {
        atk.radius -= atk.speed;
        if (atk.radius <= shieldRadius + maxRadius / 8) {
          atk.deflected = true;
          atk.deflectAngle = atk.angle + (Math.random() - 0.5) * 1.5 + Math.PI;
          atk.speed *= 0.7;
        }
      } else {
        atk.radius += atk.speed * 0.5;
        atk.angle += (atk.deflectAngle - atk.angle) * 0.1;
        atk.opacity -= 0.02;
      }

      if (atk.opacity <= 0 || atk.life > 200) {
        attacks.splice(i, 1);
        continue;
      }

      const ax = cx + Math.cos(atk.angle) * atk.radius;
      const ay = cy + Math.sin(atk.angle) * atk.radius;

      // Arrow body
      const tailLen = 18;
      const tailX = cx + Math.cos(atk.angle) * (atk.radius + tailLen);
      const tailY = cy + Math.sin(atk.angle) * (atk.radius + tailLen);

      ctx.beginPath();
      ctx.moveTo(tailX, tailY);
      ctx.lineTo(ax, ay);
      ctx.strokeStyle = atk.deflected
        ? `rgba(239, 68, 68, ${atk.opacity * 0.4})`
        : `rgba(239, 68, 68, ${atk.opacity * 0.8})`;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Arrowhead
      const headAngle = Math.atan2(ay - tailY, ax - tailX);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - 6 * Math.cos(headAngle - 0.4), ay - 6 * Math.sin(headAngle - 0.4));
      ctx.lineTo(ax - 6 * Math.cos(headAngle + 0.4), ay - 6 * Math.sin(headAngle + 0.4));
      ctx.closePath();
      ctx.fillStyle = atk.deflected
        ? `rgba(239, 68, 68, ${atk.opacity * 0.4})`
        : `rgba(239, 68, 68, ${atk.opacity * 0.8})`;
      ctx.fill();

      // Deflection spark
      if (atk.deflected && atk.life < 10) {
        const sparkR = 8 + atk.life * 2;
        const sparkGrad = ctx.createRadialGradient(ax, ay, 0, ax, ay, sparkR);
        sparkGrad.addColorStop(0, `rgba(255, 200, 50, ${0.8 - atk.life * 0.08})`);
        sparkGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = sparkGrad;
        ctx.beginPath();
        ctx.arc(ax, ay, sparkR, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Center hexagonal shield tessellation
    const hexR = maxRadius * 0.12;
    const hexPositions = [
      [0, 0],
      [hexR * 1.5, hexR * 0.87],
      [hexR * 1.5, -hexR * 0.87],
      [-hexR * 1.5, hexR * 0.87],
      [-hexR * 1.5, -hexR * 0.87],
      [0, hexR * 1.74],
      [0, -hexR * 1.74],
    ];

    hexPositions.forEach(([hx, hy], hi) => {
      const pulse = 0.3 + 0.7 * Math.abs(Math.sin(frame * 0.015 + hi * 0.8));
      drawHexagon(ctx, cx + hx, cy + hy, hexR * 0.48, security.color, pulse * 0.5);
    });

    // Center label
    ctx.font = 'bold 20px monospace';
    ctx.fillStyle = `${security.color}e0`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${security.bits}`, cx, cy - 8);

    ctx.font = 'bold 9px monospace';
    ctx.fillStyle = `${security.color}90`;
    ctx.fillText('BITS', cx, cy + 8);

    // Security tier label (top-right area)
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = `${security.color}c0`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(security.tier, W - 16, 32);

    // Stats row (bottom)
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    const statsText = `${miners} miners | ${hashRate >= 1000 ? `${(hashRate / 1000).toFixed(1)} MH/s` : `${hashRate.toFixed(1)} kH/s`} | Block #${bh.toLocaleString()}`;
    ctx.fillText(statsText, cx, H - 8);

    animRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full"
      style={{ height, display: 'block' }}
    />
  );
}

// Draw a hexagon
function drawHexagon(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  color: string,
  opacity: number
) {
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.strokeStyle = `${color}${Math.floor(opacity * 255).toString(16).padStart(2, '0')}`;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.fillStyle = `${color}${Math.floor(opacity * 0.15 * 255).toString(16).padStart(2, '0')}`;
  ctx.fill();
}
