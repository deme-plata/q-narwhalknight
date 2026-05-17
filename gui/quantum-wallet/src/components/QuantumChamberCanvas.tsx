import { useRef, useEffect, useCallback } from 'react';

interface QuantumChamberCanvasProps {
  fractalOverlay: boolean;
  photonWaterfall: boolean;
  entanglementMoire: boolean;
  rainbowBoxes: boolean;
}

// ═══════════════════════════════════════════════════════════════
// Quantum Visualization Chamber — Canvas Renderer
// Replaces flat CSS/SVG with a single <canvas> animation loop.
// Pattern follows MiningDashboard VDF Forge canvas.
// ═══════════════════════════════════════════════════════════════

export default function QuantumChamberCanvas({
  fractalOverlay,
  photonWaterfall,
  entanglementMoire,
  rainbowBoxes,
}: QuantumChamberCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animRef = useRef<number>(0);

  // Refs so the rAF loop reads latest toggle state without restarting
  const toggles = useRef({ fractalOverlay, photonWaterfall, entanglementMoire, rainbowBoxes });
  toggles.current = { fractalOverlay, photonWaterfall, entanglementMoire, rainbowBoxes };

  const startAnimation = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (animRef.current) cancelAnimationFrame(animRef.current);
    animRef.current = 0;

    // ── Canvas sizing (DPR-aware) ──
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const parentRect = canvas.parentElement?.getBoundingClientRect();
    const W = rect.width || parentRect?.width || canvas.offsetWidth || 600;
    const H = rect.height || parentRect?.height || canvas.offsetHeight || 320;
    // Skip if not laid out yet
    if (W < 10 || H < 10) return;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // ── Color palette ──
    const CYAN = '#00D4FF';
    const PURPLE = '#6B46C1';
    const PINK = '#EC4899';
    const GREEN = '#10B981';
    const AMBER = '#F59E0B';
    const RED = '#FF6B6B';
    const TEAL = '#4ECDC4';
    const PALETTE = [CYAN, PURPLE, PINK, GREEN, AMBER, RED, TEAL];
    const GEMSTONE_HUES = [0, 45, 90, 150, 200, 260, 310, 35];

    // ── Photon streaks (25) ──
    const streaks = Array.from({ length: 25 }, (_, i) => ({
      x: (i / 25) * W + Math.random() * (W / 25),
      speed: 1.2 + Math.random() * 2.5,
      len: 30 + Math.random() * 40,
      color: PALETTE[i % PALETTE.length],
      y: Math.random() * H,
      width: 2 + Math.random() * 2,
    }));

    // ── Sparkles (20, recycled) ──
    const sparkles = Array.from({ length: 20 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      phase: Math.random() * Math.PI * 2,
      speed: 0.02 + Math.random() * 0.03,
    }));

    // ── Entangled pairs (4) ──
    const pairs = Array.from({ length: 4 }, (_, i) => ({
      baseY: H * (0.25 + i * 0.15),
      hue: 180 + i * 40,
      phaseOffset: i * 0.5,
    }));

    // ── Interference ripples ──
    const rippleDelays = [0, 0.6, 1.2, 1.8, 2.4];

    let frame = 0;

    function hexRgba(hex: string, a: number): string {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${a})`;
    }

    function hsl(h: number, s: number, l: number, a = 1): string {
      return `hsla(${h},${s}%,${l}%,${a})`;
    }

    const tick = () => {
      frame++;
      const t = frame / 60;
      const fx = toggles.current;

      // ── 1. Background ──
      const bgGrad = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.7);
      bgGrad.addColorStop(0, '#1a0a2e');
      bgGrad.addColorStop(0.5, '#0f0a1a');
      bgGrad.addColorStop(1, '#060310');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, W, H);

      // ── 2. Interference Web (fractalOverlay) ──
      if (fx.fractalOverlay) {
        ctx.save();
        ctx.globalAlpha = 0.2;
        for (let i = 0; i < 14; i++) {
          const xStart = (i / 14) * W * 1.2;
          const cpx = xStart + 15 + Math.sin(t * 0.5 + i) * 25;
          const cpy = H * 0.35 + Math.sin(t * 0.3 + i * 0.7) * (H * 0.2);
          ctx.beginPath();
          ctx.moveTo(xStart, 0);
          ctx.quadraticCurveTo(cpx, cpy, xStart, H);
          ctx.strokeStyle = hsl((i * 22 + 180) % 360, 80, 55, 0.6);
          ctx.lineWidth = 1.5 + Math.sin(t + i) * 0.5;
          ctx.stroke();
        }
        // Rotating dashed circle
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.lineDashOffset = -frame * 0.3;
        const cx = W / 2, cy = H / 2;
        ctx.arc(cx, cy, Math.min(W, H) * 0.2, 0, Math.PI * 2);
        const circGrad = ctx.createLinearGradient(cx - 60, cy - 60, cx + 60, cy + 60);
        circGrad.addColorStop(0, CYAN);
        circGrad.addColorStop(0.5, PURPLE);
        circGrad.addColorStop(1, PINK);
        ctx.strokeStyle = circGrad;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      }

      // ── 3. Photon Rain (photonWaterfall) ──
      if (fx.photonWaterfall) {
        ctx.save();
        for (const s of streaks) {
          s.y += s.speed;
          if (s.y > H + s.len) {
            s.y = -s.len;
            s.x = Math.random() * W;
          }
          const trailSteps = 6;
          for (let j = trailSteps; j >= 0; j--) {
            const alpha = Math.pow(0.88, j);
            const yy = s.y - j * (s.len / trailSteps);
            if (yy < -10 || yy > H + 10) continue;
            ctx.globalAlpha = alpha * 0.7;
            ctx.fillStyle = s.color;
            ctx.fillRect(s.x - s.width / 2, yy, s.width, s.len / trailSteps + 1);
          }
          // Bloom at leading edge
          ctx.globalAlpha = 0.9;
          const bloom = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.width * 3);
          bloom.addColorStop(0, hexRgba(s.color, 0.9));
          bloom.addColorStop(1, hexRgba(s.color, 0));
          ctx.fillStyle = bloom;
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.width * 3, 0, Math.PI * 2);
          ctx.fill();
        }
        // Sparkles
        for (const sp of sparkles) {
          sp.phase += sp.speed;
          const a = Math.max(0, Math.sin(sp.phase));
          if (a < 0.05) {
            sp.x = Math.random() * W;
            sp.y = Math.random() * H;
          }
          ctx.globalAlpha = a;
          ctx.fillStyle = '#ffffff';
          ctx.beginPath();
          ctx.arc(sp.x, sp.y, 1.5 * a, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      // ── 4. Energy Connections (entanglementMoire) ──
      if (fx.entanglementMoire) {
        ctx.save();
        const leftX = W * 0.25;
        const rightX = W * 0.75;
        for (const p of pairs) {
          const phase = t * 1.2 + p.phaseOffset;
          const ly = p.baseY + Math.sin(phase) * 20;
          const lx = leftX + Math.cos(phase * 0.7) * 15;
          const ry = p.baseY - Math.sin(phase) * 20;
          const rx = rightX - Math.cos(phase * 0.7) * 15;
          const r = 8 + Math.sin(phase * 2) * 4;

          // Draw both orbs
          for (const [px, py] of [[lx, ly], [rx, ry]]) {
            // Outer aura (4x)
            ctx.globalAlpha = 0.4;
            const aura = ctx.createRadialGradient(px, py, 0, px, py, r * 4);
            aura.addColorStop(0, hsl(p.hue, 80, 60, 0.5));
            aura.addColorStop(0.4, hsl(p.hue, 60, 40, 0.2));
            aura.addColorStop(1, 'transparent');
            ctx.fillStyle = aura;
            ctx.beginPath();
            ctx.arc(px, py, r * 4, 0, Math.PI * 2);
            ctx.fill();

            // Bright core (1.3x)
            ctx.globalAlpha = 0.9;
            const core = ctx.createRadialGradient(px, py, 0, px, py, r * 1.3);
            core.addColorStop(0, '#ffffff');
            core.addColorStop(0.4, hsl(p.hue, 80, 60, 0.8));
            core.addColorStop(1, hsl(p.hue, 80, 40, 0));
            ctx.fillStyle = core;
            ctx.beginPath();
            ctx.arc(px, py, r * 1.3, 0, Math.PI * 2);
            ctx.fill();
          }

          // Animated bezier connection
          ctx.globalAlpha = 0.6;
          ctx.beginPath();
          const cpY = p.baseY + Math.sin(phase * 2) * 30;
          ctx.moveTo(lx, ly);
          ctx.quadraticCurveTo(W / 2, cpY, rx, ry);
          ctx.strokeStyle = hsl(p.hue, 80, 60);
          ctx.lineWidth = 2;
          ctx.setLineDash([8, 4]);
          ctx.lineDashOffset = -frame * 0.8;
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Interference ripples
        for (const delay of rippleDelays) {
          const age = (t - delay) % 3;
          if (age < 0) continue;
          const radius = 20 + age * (80 / 3) * (W / 400);
          const alpha = Math.max(0, 0.6 - age * 0.2);
          ctx.globalAlpha = alpha;
          ctx.beginPath();
          ctx.arc(W / 2, H / 2, radius, 0, Math.PI * 2);
          ctx.strokeStyle = CYAN;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        ctx.restore();
      }

      // ── 5. Rainbow Gemstones (rainbowBoxes) ──
      if (fx.rainbowBoxes) {
        ctx.save();
        for (let i = 0; i < 8; i++) {
          const angle = (i / 8) * Math.PI + Math.sin(t * 0.3) * 0.1;
          const arcRadius = Math.min(W, H) * 0.3;
          const px = W / 2 + Math.cos(angle - Math.PI / 2) * arcRadius * (W / H);
          const py = H * 0.5 + Math.sin(angle - Math.PI / 2) * arcRadius * 0.6;
          const baseR = 10 + i * 2.5;
          const pulse = 1 + Math.sin(t * 1.5 + i * 0.8) * 0.25;
          const r = baseR * pulse;
          const gemHue = GEMSTONE_HUES[i];

          // Morph: 0=circle, 1=hexagon
          const morph = (Math.sin(t * 0.4 + i * 0.5) + 1) / 2;

          // Outer bloom (6x)
          ctx.globalAlpha = 0.35;
          const outerBloom = ctx.createRadialGradient(px, py, 0, px, py, r * 6);
          outerBloom.addColorStop(0, hsl(gemHue, 90, 60, 0.4));
          outerBloom.addColorStop(0.5, hsl(gemHue, 80, 50, 0.1));
          outerBloom.addColorStop(1, 'transparent');
          ctx.fillStyle = outerBloom;
          ctx.beginPath();
          ctx.arc(px, py, r * 6, 0, Math.PI * 2);
          ctx.fill();

          // Core gradient with 3D offset
          ctx.globalAlpha = 0.9;
          const coreGrad = ctx.createRadialGradient(
            px - r * 0.2, py - r * 0.2, 0,
            px, py, r * 1.2
          );
          coreGrad.addColorStop(0, hsl(gemHue, 90, 75));
          coreGrad.addColorStop(0.5, hsl((gemHue + 60) % 360, 90, 55));
          coreGrad.addColorStop(1, hsl((gemHue + 120) % 360, 90, 40, 0.6));
          ctx.fillStyle = coreGrad;

          // Morphing hexagon↔circle
          ctx.beginPath();
          const sides = 6;
          const facetAngle = (Math.PI * 2) / sides;
          for (let s = 0; s <= sides * 4; s++) {
            const a = (s / (sides * 4)) * Math.PI * 2 - Math.PI / 2 + t * 0.5;
            const withinFacet = ((a % facetAngle) + facetAngle) % facetAngle - facetAngle / 2;
            const facetR = r / Math.cos(withinFacet);
            const finalR = r + (facetR - r) * morph;
            const sx = px + Math.cos(a) * finalR;
            const sy = py + Math.sin(a) * finalR;
            if (s === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          }
          ctx.closePath();
          ctx.fill();

          // White highlight spot
          ctx.globalAlpha = 0.5;
          const hlGrad = ctx.createRadialGradient(
            px - r * 0.3, py - r * 0.3, 0,
            px - r * 0.3, py - r * 0.3, r * 0.5
          );
          hlGrad.addColorStop(0, 'rgba(255,255,255,0.8)');
          hlGrad.addColorStop(1, 'rgba(255,255,255,0)');
          ctx.fillStyle = hlGrad;
          ctx.beginPath();
          ctx.arc(px - r * 0.3, py - r * 0.3, r * 0.5, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      // ── 6. Central Orb (always) ──
      {
        ctx.save();
        const cx = W / 2, cy = H / 2;
        const orbR = 16 + Math.sin(t * 1.2) * 4;
        ctx.globalAlpha = 0.6 + Math.sin(t * 1.2) * 0.2;
        const orbGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, orbR);
        orbGrad.addColorStop(0, 'rgba(255,255,255,0.9)');
        orbGrad.addColorStop(0.4, hexRgba(PURPLE, 0.6));
        orbGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = orbGrad;
        ctx.beginPath();
        ctx.arc(cx, cy, orbR, 0, Math.PI * 2);
        ctx.fill();
        // Outer glow
        ctx.globalAlpha = 0.15;
        const glowGrad = ctx.createRadialGradient(cx, cy, orbR, cx, cy, orbR * 3);
        glowGrad.addColorStop(0, hexRgba(PURPLE, 0.3));
        glowGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = glowGrad;
        ctx.beginPath();
        ctx.arc(cx, cy, orbR * 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      animRef.current = requestAnimationFrame(tick);
    };

    tick();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Use ResizeObserver on parent (which has the h-80 sizing) to detect layout
    const target = canvas.parentElement || canvas;
    const ro = new ResizeObserver(() => {
      startAnimation();
    });
    ro.observe(target);

    // Also try on next frame in case ResizeObserver fires before layout
    const raf = requestAnimationFrame(() => startAnimation());

    return () => {
      ro.disconnect();
      cancelAnimationFrame(raf);
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [startAnimation]);

  return (
    <canvas
      ref={canvasRef}
      className="quantum-chamber-canvas"
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'block' }}
    />
  );
}
