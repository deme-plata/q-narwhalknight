import { useRef, useEffect } from 'react';

// ═══════════════════════════════════════════════════════════════
// SecurityBitsVisualization v10.2.2 — ResizeObserver + wrapper div
//
// 8 concentric security rings (each = 32 bits, total 256-bit max).
// Binary digits stream inward and lock into the lattice as miners join.
// Red attack arrows deflect off the hardened shield.
// Hexagonal DAG-Knight lattice tessellation at center.
//
// Uses ResizeObserver on a wrapper div to guarantee we have real
// pixel dimensions before initializing the canvas. No more guessing
// with timeouts or rAF delays.
// ═══════════════════════════════════════════════════════════════

interface Props {
  connectedMiners: number;
  networkHashRate: number; // kH/s
  blockHeight: number;
  height: number;
}

const RING_COLORS: [number, number, number][] = [
  [239, 68, 68],   // Ring 0 (outermost) — red
  [249, 115, 22],  // Ring 1 — orange
  [245, 158, 11],  // Ring 2 — amber
  [234, 179, 8],   // Ring 3 — yellow
  [132, 204, 22],  // Ring 4 — lime
  [34, 211, 238],  // Ring 5 — cyan
  [6, 182, 212],   // Ring 6 — dark cyan
  [8, 145, 178],   // Ring 7 (innermost) — deep cyan
];

function getSecurityTier(miners: number) {
  if (miners >= 100) return { tier: 'FORTRESS', bits: 256, filledRings: 8, color: [34, 211, 238] as number[] };
  if (miners >= 50)  return { tier: 'FORTIFIED', bits: 192, filledRings: 6, color: [16, 185, 129] as number[] };
  if (miners >= 10)  return { tier: 'STRONG', bits: 128, filledRings: 4, color: [234, 179, 8] as number[] };
  if (miners >= 3)   return { tier: 'WEAK', bits: 64, filledRings: 2, color: [249, 115, 22] as number[] };
  return { tier: 'VULNERABLE', bits: 32, filledRings: 1, color: [239, 68, 68] as number[] };
}

export default function SecurityBitsVisualization({ connectedMiners, networkHashRate, blockHeight, height }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);
  const frameRef = useRef(0);
  const initedRef = useRef(false);
  const propsRef = useRef({ connectedMiners, networkHashRate, blockHeight });
  propsRef.current = { connectedMiners, networkHashRate, blockHeight };

  const bitsRef = useRef<{ angle: number; radius: number; speed: number; val: number; ring: number; locked: boolean; age: number }[]>([]);
  const atksRef = useRef<{ angle: number; radius: number; speed: number; deflected: boolean; dir: number; age: number }[]>([]);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    function initCanvas() {
      if (initedRef.current) return;
      const rect = wrap!.getBoundingClientRect();
      const W = Math.round(rect.width);
      const H = Math.round(rect.height);
      if (W < 20 || H < 20) return; // not laid out yet — wait

      initedRef.current = true;
      const maybeCtx = canvas!.getContext('2d');
      if (!maybeCtx) return;
      const ctx: CanvasRenderingContext2D = maybeCtx;

      const dpr = window.devicePixelRatio || 1;
      canvas!.width = W * dpr;
      canvas!.height = H * dpr;
      ctx.scale(dpr, dpr);

      const cx = W / 2;
      const cy = H / 2;
      const maxR = Math.min(W, H) * 0.40;
      const TAU = Math.PI * 2;
      const rw = maxR / 8; // ring width
      const bits = bitsRef.current;
      const atks = atksRef.current;

      function rgba(c: number[], a: number) {
        return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
      }

      function tick() {
        const { connectedMiners: miners, networkHashRate: hr, blockHeight: bh } = propsRef.current;
        const sec = getSecurityTier(miners);
        const f = frameRef.current++;

        ctx.clearRect(0, 0, W, H);

        // ── Background glow ──
        const bg = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR * 1.4);
        bg.addColorStop(0, rgba(sec.color, 0.06));
        bg.addColorStop(0.6, rgba(sec.color, 0.02));
        bg.addColorStop(1, 'transparent');
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, W, H);

        // ── 8 Concentric Security Rings ──
        for (let r = 0; r < 8; r++) {
          const outerR = maxR - r * rw;
          const innerR = outerR - rw;
          const midR = (outerR + innerR) / 2;
          const filled = r < sec.filledRings;
          const rc = RING_COLORS[r];

          // Ring arc
          ctx.beginPath();
          ctx.arc(cx, cy, midR, 0, TAU);
          ctx.strokeStyle = rgba(rc, filled ? 0.45 : 0.10);
          ctx.lineWidth = rw * 0.55;
          ctx.stroke();

          if (filled) {
            // Glow fill
            const g = ctx.createRadialGradient(cx, cy, innerR, cx, cy, outerR);
            g.addColorStop(0, rgba(rc, 0.10));
            g.addColorStop(0.5, rgba(rc, 0.04));
            g.addColorStop(1, 'transparent');
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(cx, cy, outerR, 0, TAU);
            ctx.arc(cx, cy, innerR, 0, TAU, true);
            ctx.fill();

            // Binary digits orbiting ring
            const n = 32;
            const fs = Math.max(8, rw * 0.32);
            ctx.font = `bold ${fs}px monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            for (let b = 0; b < n; b++) {
              const a = (b / n) * TAU + r * 0.35 + f * 0.0004 * (r % 2 === 0 ? 1 : -1);
              const bx = cx + Math.cos(a) * midR;
              const by = cy + Math.sin(a) * midR;
              const pulse = 0.4 + 0.6 * Math.abs(Math.sin(f * 0.018 + b * 0.25 + r));
              ctx.fillStyle = rgba(rc, pulse * 0.75);
              ctx.fillText(b % 2 === 0 ? '1' : '0', bx, by);
            }
          }

          // Bit-count label
          const la = -Math.PI / 2 + 0.15;
          ctx.font = `bold ${Math.max(9, rw * 0.30)}px monospace`;
          ctx.fillStyle = rgba(rc, filled ? 0.60 : 0.18);
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(`${(r + 1) * 32}`, cx + Math.cos(la) * midR, cy + Math.sin(la) * midR);
        }

        // ── Streaming bit particles ──
        if (f % 4 === 0 && bits.length < 50) {
          bits.push({
            angle: Math.random() * TAU,
            radius: maxR * 1.15 + Math.random() * 25,
            speed: 0.5 + Math.random() * 0.7,
            val: Math.random() > 0.5 ? 1 : 0,
            ring: Math.floor(Math.random() * Math.max(sec.filledRings, 1)),
            locked: false,
            age: 0,
          });
        }

        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for (let i = bits.length - 1; i >= 0; i--) {
          const b = bits[i];
          const tgtR = maxR - (b.ring + 0.5) * rw;

          if (!b.locked) {
            b.radius -= b.speed;
            b.angle += 0.006;
            if (b.radius <= tgtR + 2) {
              b.locked = true;
              b.radius = tgtR;
              b.age = 0;
            }
          } else {
            b.age++;
            if (b.age > 45) { bits.splice(i, 1); continue; }
          }

          const bx = cx + Math.cos(b.angle) * b.radius;
          const by = cy + Math.sin(b.angle) * b.radius;
          const rc = RING_COLORS[b.ring];

          if (!b.locked) {
            const tx = cx + Math.cos(b.angle - 0.015) * (b.radius + 10);
            const ty = cy + Math.sin(b.angle - 0.015) * (b.radius + 10);
            ctx.beginPath();
            ctx.moveTo(tx, ty);
            ctx.lineTo(bx, by);
            ctx.strokeStyle = 'rgba(255,255,255,0.15)';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = 'rgba(255,255,255,0.8)';
            ctx.fillText(b.val.toString(), bx, by);
          } else {
            const alpha = 1 - b.age / 45;
            if (b.age < 6) {
              const fr = 4 + b.age * 2;
              const fg = ctx.createRadialGradient(bx, by, 0, bx, by, fr);
              fg.addColorStop(0, rgba(rc, 0.5 * alpha));
              fg.addColorStop(1, 'transparent');
              ctx.fillStyle = fg;
              ctx.beginPath();
              ctx.arc(bx, by, fr, 0, TAU);
              ctx.fill();
            }
            ctx.fillStyle = rgba(rc, alpha * 0.85);
            ctx.fillText(b.val.toString(), bx, by);
          }
        }

        // ── Attack arrows ──
        const shieldR = maxR - sec.filledRings * rw;

        if (f % 120 === 0 && atks.length < 4) {
          atks.push({
            angle: Math.random() * TAU,
            radius: maxR * 1.5,
            speed: 1.2 + Math.random() * 1.3,
            deflected: false,
            dir: 0,
            age: 0,
          });
        }

        for (let i = atks.length - 1; i >= 0; i--) {
          const a = atks[i];
          a.age++;

          if (!a.deflected) {
            a.radius -= a.speed;
            if (a.radius <= shieldR + rw) {
              a.deflected = true;
              a.dir = (Math.random() - 0.5) * 2;
              a.speed *= 0.6;
            }
          } else {
            a.radius += a.speed * 0.4;
            a.angle += a.dir * 0.03;
          }

          const fa = a.deflected ? Math.max(0, 1 - (a.age - 30) / 60) : 0.85;
          if (fa <= 0 || a.age > 200) { atks.splice(i, 1); continue; }

          const ax = cx + Math.cos(a.angle) * a.radius;
          const ay = cy + Math.sin(a.angle) * a.radius;
          const tailR = a.radius + 20;
          const tx = cx + Math.cos(a.angle) * tailR;
          const ty = cy + Math.sin(a.angle) * tailR;

          ctx.beginPath();
          ctx.moveTo(tx, ty);
          ctx.lineTo(ax, ay);
          ctx.strokeStyle = `rgba(239,68,68,${fa * (a.deflected ? 0.5 : 0.9)})`;
          ctx.lineWidth = 2.5;
          ctx.stroke();

          const ha = Math.atan2(ay - ty, ax - tx);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(ax - 8 * Math.cos(ha - 0.35), ay - 8 * Math.sin(ha - 0.35));
          ctx.lineTo(ax - 8 * Math.cos(ha + 0.35), ay - 8 * Math.sin(ha + 0.35));
          ctx.closePath();
          ctx.fillStyle = `rgba(239,68,68,${fa * (a.deflected ? 0.6 : 1.0)})`;
          ctx.fill();

          // Deflection spark
          if (a.deflected && a.age < 45) {
            const sp = (a.age - 20) / 25;
            if (sp > -1 && sp < 1) {
              const sr = 8 + Math.abs(sp) * 14;
              const sa = Math.max(0, 0.8 - Math.abs(sp) * 0.8);
              const sg = ctx.createRadialGradient(ax, ay, 0, ax, ay, sr);
              sg.addColorStop(0, `rgba(255,200,50,${sa})`);
              sg.addColorStop(1, 'transparent');
              ctx.fillStyle = sg;
              ctx.beginPath();
              ctx.arc(ax, ay, sr, 0, TAU);
              ctx.fill();
            }
          }
        }

        // ── Center: Hexagonal DAG-Knight lattice ──
        const hexR = maxR * 0.11;
        const hexOff: [number, number][] = [
          [0, 0],
          [hexR * 1.55, hexR * 0.9], [hexR * 1.55, -hexR * 0.9],
          [-hexR * 1.55, hexR * 0.9], [-hexR * 1.55, -hexR * 0.9],
          [0, hexR * 1.8], [0, -hexR * 1.8],
        ];

        for (let h = 0; h < hexOff.length; h++) {
          const [hx, hy] = hexOff[h];
          const pulse = 0.35 + 0.65 * Math.abs(Math.sin(f * 0.013 + h * 0.9));
          const hcx = cx + hx;
          const hcy = cy + hy;
          const hr = hexR * 0.45;

          ctx.beginPath();
          for (let v = 0; v < 6; v++) {
            const va = (TAU / 6) * v - Math.PI / 6;
            const px = hcx + hr * Math.cos(va);
            const py = hcy + hr * Math.sin(va);
            v === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
          }
          ctx.closePath();
          ctx.strokeStyle = rgba(sec.color, pulse * 0.6);
          ctx.lineWidth = 1.4;
          ctx.stroke();
          ctx.fillStyle = rgba(sec.color, pulse * 0.10);
          ctx.fill();
        }

        // Center text
        ctx.save();
        ctx.shadowColor = rgba(sec.color, 0.5);
        ctx.shadowBlur = 12;
        ctx.font = 'bold 24px monospace';
        ctx.fillStyle = rgba(sec.color, 0.95);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${sec.bits}`, cx, cy - 8);
        ctx.restore();

        ctx.font = 'bold 10px monospace';
        ctx.fillStyle = rgba(sec.color, 0.65);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('BITS', cx, cy + 10);

        // Tier badge top-right
        ctx.font = 'bold 12px sans-serif';
        ctx.fillStyle = rgba(sec.color, 0.80);
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText(sec.tier, W - 14, 30);

        // Stats bottom
        ctx.font = '10px monospace';
        ctx.fillStyle = 'rgba(148,163,184,0.50)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        let hrLabel: string;
        if (hr >= 1e6) hrLabel = `${(hr / 1e6).toFixed(1)} TH/s`;
        else if (hr >= 1e3) hrLabel = `${(hr / 1e3).toFixed(1)} GH/s`;
        else if (hr >= 1) hrLabel = `${hr.toFixed(1)} MH/s`;
        else hrLabel = `${(hr * 1000).toFixed(0)} kH/s`;
        ctx.fillText(`${miners} miners  |  ${hrLabel}  |  Block #${bh.toLocaleString()}`, cx, H - 8);

        animRef.current = requestAnimationFrame(tick);
      }

      tick();
    }

    // Strategy: try immediately, then rAF, then ResizeObserver as failsafe
    initCanvas();
    const raf = requestAnimationFrame(() => initCanvas());
    const ro = new ResizeObserver(() => initCanvas());
    ro.observe(wrap);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      if (animRef.current) cancelAnimationFrame(animRef.current);
      initedRef.current = false;
    };
  }, []);

  return (
    <div ref={wrapRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block', position: 'absolute', top: 0, left: 0 }}
      />
    </div>
  );
}
