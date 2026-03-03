import React, { useEffect, useRef, useState } from 'react';

/**
 * QuantumLoader — Dramatic quantum particle visualization
 *
 * Full-screen particle physics show:
 * - Orbiting electrons, photon streaks, qubit superposition flickers
 * - Pulsing nucleus with hexagonal core
 * - Wave function interference patterns
 * - Neural energy connections between particles
 * - Expanding pulse rings
 * - Grid matrix that materializes
 * - Data stream waterfalls
 * - Energy bursts and lightning arcs
 *
 * The particles are the STAR — not a boring skeleton UI.
 */

interface QuantumLoaderProps {
  message?: string;
  subMessage?: string;
  progress?: number;
  onComplete?: () => void;
  inline?: boolean;
  backgroundOnly?: boolean; // Canvas-only mode: no overlays, no text — just particles
}

interface Particle {
  x: number; y: number;
  vx: number; vy: number;
  orbit: number;
  orbitSpeed: number;
  orbitPhase: number;
  radius: number;
  hue: number;
  life: number;
  maxLife: number;
  type: 'electron' | 'photon' | 'qubit' | 'data' | 'spark';
  trail: { x: number; y: number; alpha: number }[];
}

interface PulseRing {
  x: number; y: number;
  radius: number;
  maxRadius: number;
  hue: number;
  life: number;
}

interface EnergyArc {
  x1: number; y1: number;
  x2: number; y2: number;
  hue: number;
  life: number;
  maxLife: number;
  segments: { ox: number; oy: number }[];
}

const QuantumLoader: React.FC<QuantumLoaderProps> = ({
  message = 'Initializing Quantum Dashboard',
  subMessage,
  progress,
  onComplete,
  inline = false,
  backgroundOnly = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const [statusText, setStatusText] = useState('');

  useEffect(() => {
    const msgs = [
      'Bootstrapping quantum state vectors...',
      'Entangling wallet keypairs...',
      'Establishing P2P gossipsub mesh...',
      'Synchronizing DAG-Knight consensus...',
      'Calibrating post-quantum lattices...',
      'Resolving superposition states...',
      'Initializing Dilithium-5 signatures...',
      'Mapping Kademlia DHT routes...',
    ];
    let i = 0;
    setStatusText(msgs[0]);
    const iv = setInterval(() => { i = (i + 1) % msgs.length; setStatusText(msgs[i]); }, 2200);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    let W = 0, H = 0, cx = 0, cy = 0;
    let t = 0;

    const particles: Particle[] = [];
    const pulseRings: PulseRing[] = [];
    const energyArcs: EnergyArc[] = [];
    let nextPulse = 0;
    let nextArc = 0;
    let nextSpark = 0;

    const resize = () => {
      if (inline && canvas.parentElement) {
        W = canvas.parentElement.clientWidth;
        H = canvas.parentElement.clientHeight;
      } else {
        W = window.innerWidth;
        H = window.innerHeight;
      }
      canvas.width = W;
      canvas.height = H;
      cx = W / 2;
      cy = H / 2;
    };

    resize();
    window.addEventListener('resize', resize);

    // ── Create particle ──
    const createParticle = (type: Particle['type'], fromX?: number, fromY?: number): Particle => {
      const baseOrbit = 40 + Math.random() * Math.min(W, H) * 0.4;
      const p: Particle = {
        x: fromX ?? cx, y: fromY ?? cy,
        vx: 0, vy: 0,
        orbit: baseOrbit,
        orbitSpeed: (0.002 + Math.random() * 0.008) * (Math.random() > 0.5 ? 1 : -1),
        orbitPhase: Math.random() * Math.PI * 2,
        radius: type === 'photon' ? 1 : type === 'qubit' ? 2.5 : type === 'spark' ? 0.8 : type === 'data' ? 1.2 : 2,
        hue: type === 'electron' ? 195 : type === 'photon' ? 45 : type === 'qubit' ? 280 : type === 'data' ? 160 : 30,
        life: 0, maxLife: type === 'spark' ? 30 + Math.random() * 40 : 200 + Math.random() * 500,
        type,
        trail: [],
      };
      if (type === 'data') {
        // Data particles fall like matrix rain
        p.x = Math.random() * W;
        p.y = -10;
        p.vx = 0;
        p.vy = 1.5 + Math.random() * 3;
        p.orbit = 0;
      }
      if (type === 'spark') {
        const angle = Math.random() * Math.PI * 2;
        const speed = 2 + Math.random() * 5;
        p.vx = Math.cos(angle) * speed;
        p.vy = Math.sin(angle) * speed;
        p.orbit = 0;
      }
      return p;
    };

    // Initialize particle population
    for (let i = 0; i < 80; i++) {
      const types: Particle['type'][] = ['electron', 'electron', 'photon', 'qubit', 'electron'];
      particles.push(createParticle(types[i % types.length]));
    }

    // ── Draw functions ──

    // Background grid matrix
    const drawGrid = () => {
      const gridSize = 60;
      const gridAlpha = 0.015 + 0.008 * Math.sin(t * 0.3);
      ctx.strokeStyle = `rgba(34, 211, 238, ${gridAlpha})`;
      ctx.lineWidth = 0.3;
      // Horizontal lines
      for (let y = gridSize; y < H; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }
      // Vertical lines
      for (let x = gridSize; x < W; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
      }
    };

    // Multiple wave functions with interference
    const drawWaves = () => {
      for (let w = 0; w < 3; w++) {
        ctx.beginPath();
        const hue = 195 + w * 40;
        ctx.strokeStyle = `hsla(${hue}, 100%, 60%, ${0.06 - w * 0.015})`;
        ctx.lineWidth = 1 + w * 0.3;
        const freq = 0.004 + w * 0.003;
        const amp = 30 + w * 15;
        const speed = 0.4 + w * 0.15;
        const yOff = cy + (w - 1) * 40;
        for (let x = 0; x < W; x += 3) {
          const y = yOff
            + Math.sin(x * freq + t * speed) * amp
            + Math.sin(x * freq * 2.3 - t * speed * 0.7) * amp * 0.4
            + Math.cos(x * freq * 0.5 + t * speed * 0.3) * amp * 0.6;
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    };

    // Pulsing nucleus with hexagonal core and energy rings
    const drawNucleus = () => {
      const pulse = 0.5 + 0.5 * Math.sin(t * 1.5);
      const pulse2 = 0.5 + 0.5 * Math.sin(t * 2.3 + 1);

      // Outer glow
      const g1 = ctx.createRadialGradient(cx, cy, 0, cx, cy, 120 * pulse);
      g1.addColorStop(0, `hsla(45, 100%, 80%, ${0.15 * pulse})`);
      g1.addColorStop(0.3, `hsla(195, 100%, 60%, ${0.06 * pulse})`);
      g1.addColorStop(0.6, `hsla(280, 100%, 60%, ${0.03 * pulse})`);
      g1.addColorStop(1, 'transparent');
      ctx.fillStyle = g1;
      ctx.fillRect(cx - 150, cy - 150, 300, 300);

      // Inner core glow
      const g2 = ctx.createRadialGradient(cx, cy, 0, cx, cy, 25);
      g2.addColorStop(0, `hsla(45, 100%, 90%, ${0.3 * pulse2})`);
      g2.addColorStop(0.5, `hsla(45, 100%, 70%, ${0.1 * pulse2})`);
      g2.addColorStop(1, 'transparent');
      ctx.fillStyle = g2;
      ctx.fillRect(cx - 30, cy - 30, 60, 60);

      // Rotating hexagonal core
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(t * 0.12);
      const r = 12 + 3 * Math.sin(t * 2);
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 3) * i - Math.PI / 6;
        const px = Math.cos(a) * r;
        const py = Math.sin(a) * r;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.strokeStyle = `hsla(45, 100%, 70%, ${0.5 + 0.3 * Math.sin(t * 3)})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Inner triangle
      ctx.beginPath();
      for (let i = 0; i < 3; i++) {
        const a = (Math.PI * 2 / 3) * i - Math.PI / 2;
        const px = Math.cos(a) * r * 0.5;
        const py = Math.sin(a) * r * 0.5;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.strokeStyle = `hsla(195, 100%, 70%, ${0.3 + 0.2 * Math.sin(t * 4)})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();

      // "Q" letter in center
      ctx.fillStyle = `hsla(45, 100%, 80%, ${0.4 + 0.2 * pulse})`;
      ctx.font = `bold ${8 + pulse * 3}px monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Q', 0, 0);
      ctx.restore();
    };

    // Orbit rings with dashes
    const drawOrbits = () => {
      const rings = [
        { r: 60, h: 195, speed: 0.05 },
        { r: 100, h: 210, speed: -0.03 },
        { r: 150, h: 230, speed: 0.02 },
        { r: 210, h: 250, speed: -0.04 },
        { r: 280, h: 270, speed: 0.015 },
      ];
      rings.forEach((ring, i) => {
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(t * ring.speed);
        ctx.beginPath();
        ctx.ellipse(0, 0, ring.r, ring.r * 0.5, 0, 0, Math.PI * 2);
        ctx.strokeStyle = `hsla(${ring.h}, 50%, 45%, ${0.04 + 0.02 * Math.sin(t + i)})`;
        ctx.lineWidth = 0.5;
        ctx.setLineDash([4, 8 + i * 2]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      });
    };

    // Particles with trails
    const drawParticles = () => {
      particles.forEach((p, idx) => {
        p.life++;

        // Update position
        if (p.type === 'data') {
          p.y += p.vy;
          if (p.y > H + 10) {
            p.y = -10;
            p.x = Math.random() * W;
            p.life = 0;
          }
        } else if (p.type === 'spark') {
          p.x += p.vx;
          p.y += p.vy;
          p.vx *= 0.96;
          p.vy *= 0.96;
        } else {
          p.orbitPhase += p.orbitSpeed;
          const wobble = Math.sin(t * 0.5 + p.orbitPhase * 3) * 15;
          p.x = cx + Math.cos(p.orbitPhase) * (p.orbit + wobble);
          p.y = cy + Math.sin(p.orbitPhase) * ((p.orbit + wobble) * 0.5);
        }

        // Trail
        if (p.type !== 'data') {
          p.trail.push({ x: p.x, y: p.y, alpha: 1 });
          if (p.trail.length > (p.type === 'photon' ? 15 : p.type === 'spark' ? 8 : 6)) {
            p.trail.shift();
          }
        }

        // Lifecycle
        const lifePct = p.life / p.maxLife;
        const alpha = lifePct < 0.08 ? lifePct / 0.08 : lifePct > 0.85 ? (1 - lifePct) / 0.15 : 1;
        if (p.life > p.maxLife) {
          if (p.type === 'spark') {
            particles.splice(idx, 1);
            return;
          }
          particles[idx] = createParticle(p.type);
          return;
        }

        // Draw trail
        if (p.trail.length > 1 && p.type !== 'data') {
          ctx.beginPath();
          ctx.moveTo(p.trail[0].x, p.trail[0].y);
          for (let i = 1; i < p.trail.length; i++) {
            ctx.lineTo(p.trail[i].x, p.trail[i].y);
          }
          const trailAlpha = alpha * (p.type === 'photon' ? 0.4 : p.type === 'spark' ? 0.6 : 0.15);
          ctx.strokeStyle = `hsla(${p.hue}, 100%, 70%, ${trailAlpha})`;
          ctx.lineWidth = p.type === 'photon' ? 1.2 : p.type === 'spark' ? 0.6 : 0.8;
          ctx.stroke();
        }

        // Draw particle
        if (p.type === 'data') {
          // Matrix rain characters
          const char = String.fromCharCode(0x30A0 + Math.floor(Math.random() * 96));
          ctx.fillStyle = `hsla(${p.hue}, 80%, 60%, ${alpha * 0.25})`;
          ctx.font = '10px monospace';
          ctx.fillText(char, p.x, p.y);
        } else if (p.type === 'qubit') {
          // Superposition flicker
          const state = Math.sin(t * 8 + p.orbitPhase * 5) > 0;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius * (state ? 1.4 : 0.5), 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${state ? 280 : 195}, 100%, 75%, ${alpha * 0.6})`;
          ctx.fill();
          // State label
          ctx.fillStyle = `hsla(${state ? 280 : 195}, 100%, 85%, ${alpha * 0.15})`;
          ctx.font = '7px monospace';
          ctx.fillText(state ? '|1⟩' : '|0⟩', p.x + 5, p.y - 3);
        } else if (p.type === 'photon') {
          // Streaking light
          const trLen = 12;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(
            p.x - Math.cos(p.orbitPhase) * p.orbitSpeed * trLen * p.orbit,
            p.y - Math.sin(p.orbitPhase) * p.orbitSpeed * trLen * p.orbit * 0.5
          );
          ctx.strokeStyle = `hsla(${p.hue}, 100%, 85%, ${alpha * 0.5})`;
          ctx.lineWidth = 1;
          ctx.stroke();
          // Bright point
          ctx.beginPath();
          ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${p.hue}, 100%, 95%, ${alpha * 0.8})`;
          ctx.fill();
        } else if (p.type === 'spark') {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${p.hue}, 100%, 85%, ${alpha * 0.8})`;
          ctx.fill();
        } else {
          // Electron with radial glow
          const gl = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius * 4);
          gl.addColorStop(0, `hsla(${p.hue}, 100%, 85%, ${alpha * 0.5})`);
          gl.addColorStop(0.4, `hsla(${p.hue}, 100%, 65%, ${alpha * 0.12})`);
          gl.addColorStop(1, 'transparent');
          ctx.fillStyle = gl;
          ctx.fillRect(p.x - p.radius * 4, p.y - p.radius * 4, p.radius * 8, p.radius * 8);
          // Bright core
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius * 0.6, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${p.hue}, 100%, 90%, ${alpha * 0.6})`;
          ctx.fill();
        }
      });
    };

    // Neural energy connections between nearby particles
    const drawConnections = () => {
      const orbiting = particles.filter(p => p.type !== 'data' && p.type !== 'spark');
      const maxDist = 120;
      for (let i = 0; i < orbiting.length; i++) {
        for (let j = i + 1; j < orbiting.length; j++) {
          const dx = orbiting[i].x - orbiting[j].x;
          const dy = orbiting[i].y - orbiting[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < maxDist) {
            const alpha = (1 - dist / maxDist) * 0.08;
            const hue = (orbiting[i].hue + orbiting[j].hue) / 2;
            ctx.beginPath();
            ctx.moveTo(orbiting[i].x, orbiting[i].y);
            ctx.lineTo(orbiting[j].x, orbiting[j].y);
            ctx.strokeStyle = `hsla(${hue}, 70%, 60%, ${alpha})`;
            ctx.lineWidth = 0.4;
            ctx.stroke();
          }
        }
      }
    };

    // Expanding pulse rings
    const drawPulseRings = () => {
      for (let i = pulseRings.length - 1; i >= 0; i--) {
        const ring = pulseRings[i];
        ring.radius += 2.5;
        ring.life++;
        const alpha = (1 - ring.radius / ring.maxRadius) * 0.15;
        if (alpha <= 0) {
          pulseRings.splice(i, 1);
          continue;
        }
        ctx.beginPath();
        ctx.arc(ring.x, ring.y, ring.radius, 0, Math.PI * 2);
        ctx.strokeStyle = `hsla(${ring.hue}, 80%, 60%, ${alpha})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    };

    // Lightning energy arcs
    const drawEnergyArcs = () => {
      for (let i = energyArcs.length - 1; i >= 0; i--) {
        const arc = energyArcs[i];
        arc.life++;
        const alpha = (1 - arc.life / arc.maxLife) * 0.5;
        if (alpha <= 0) {
          energyArcs.splice(i, 1);
          continue;
        }
        ctx.beginPath();
        ctx.moveTo(arc.x1, arc.y1);
        arc.segments.forEach(seg => {
          ctx.lineTo(seg.ox, seg.oy);
        });
        ctx.lineTo(arc.x2, arc.y2);
        ctx.strokeStyle = `hsla(${arc.hue}, 100%, 80%, ${alpha})`;
        ctx.lineWidth = 1;
        ctx.stroke();
        // Glow pass
        ctx.strokeStyle = `hsla(${arc.hue}, 100%, 60%, ${alpha * 0.3})`;
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    };

    // Data hex values floating up from bottom
    const drawHexStream = () => {
      const hexAlpha = 0.04 + 0.02 * Math.sin(t * 0.5);
      ctx.fillStyle = `rgba(212, 175, 55, ${hexAlpha})`;
      ctx.font = '9px monospace';
      for (let col = 0; col < 8; col++) {
        const x = W * 0.1 + col * (W * 0.1);
        for (let row = 0; row < 3; row++) {
          const yBase = H - 60 + row * 16;
          const y = yBase - ((t * 20 + col * 7) % 60);
          const hex = Math.floor(Math.random() * 0xFFFF).toString(16).padStart(4, '0');
          const rowAlpha = (1 - row / 3) * hexAlpha * 6;
          ctx.fillStyle = `rgba(212, 175, 55, ${rowAlpha})`;
          ctx.fillText(hex, x, y);
        }
      }
    };

    // Corner decorations — sci-fi HUD brackets
    const drawHUD = () => {
      const bLen = 30;
      const pad = 20;
      const alpha = 0.08 + 0.04 * Math.sin(t * 0.7);
      ctx.strokeStyle = `rgba(34, 211, 238, ${alpha})`;
      ctx.lineWidth = 1;
      // Top-left
      ctx.beginPath(); ctx.moveTo(pad, pad + bLen); ctx.lineTo(pad, pad); ctx.lineTo(pad + bLen, pad); ctx.stroke();
      // Top-right
      ctx.beginPath(); ctx.moveTo(W - pad - bLen, pad); ctx.lineTo(W - pad, pad); ctx.lineTo(W - pad, pad + bLen); ctx.stroke();
      // Bottom-left
      ctx.beginPath(); ctx.moveTo(pad, H - pad - bLen); ctx.lineTo(pad, H - pad); ctx.lineTo(pad + bLen, H - pad); ctx.stroke();
      // Bottom-right
      ctx.beginPath(); ctx.moveTo(W - pad - bLen, H - pad); ctx.lineTo(W - pad, H - pad); ctx.lineTo(W - pad, H - pad - bLen); ctx.stroke();

      // Scanning line
      const scanY = (t * 50) % (H + 60) - 30;
      const scanG = ctx.createLinearGradient(0, scanY - 15, 0, scanY + 15);
      scanG.addColorStop(0, 'transparent');
      scanG.addColorStop(0.5, 'hsla(195, 100%, 60%, 0.025)');
      scanG.addColorStop(1, 'transparent');
      ctx.fillStyle = scanG;
      ctx.fillRect(0, scanY - 15, W, 30);
    };

    // ═══ MAIN ANIMATION LOOP ═══
    const animate = () => {
      t += 1 / 60;

      // Clear with trail effect
      ctx.fillStyle = 'rgba(6, 8, 16, 0.15)';
      ctx.fillRect(0, 0, W, H);
      // Periodic full clear to prevent ghost buildup
      if (Math.floor(t * 60) % 180 === 0) {
        ctx.fillStyle = '#060810';
        ctx.fillRect(0, 0, W, H);
      }

      // Spawn pulse rings periodically
      if (t > nextPulse) {
        pulseRings.push({
          x: cx + (Math.random() - 0.5) * 60,
          y: cy + (Math.random() - 0.5) * 30,
          radius: 5,
          maxRadius: 200 + Math.random() * 200,
          hue: [45, 195, 280][Math.floor(Math.random() * 3)],
          life: 0,
        });
        nextPulse = t + 1.5 + Math.random() * 2;
      }

      // Spawn energy arcs between random particles
      if (t > nextArc && particles.length > 5) {
        const orb = particles.filter(p => p.type === 'electron' || p.type === 'qubit');
        if (orb.length >= 2) {
          const a = orb[Math.floor(Math.random() * orb.length)];
          const b = orb[Math.floor(Math.random() * orb.length)];
          if (a !== b) {
            const segs: { ox: number; oy: number }[] = [];
            const steps = 4 + Math.floor(Math.random() * 4);
            for (let s = 1; s < steps; s++) {
              const frac = s / steps;
              segs.push({
                ox: a.x + (b.x - a.x) * frac + (Math.random() - 0.5) * 40,
                oy: a.y + (b.y - a.y) * frac + (Math.random() - 0.5) * 25,
              });
            }
            energyArcs.push({
              x1: a.x, y1: a.y, x2: b.x, y2: b.y,
              hue: [45, 195, 280][Math.floor(Math.random() * 3)],
              life: 0, maxLife: 15 + Math.random() * 20,
              segments: segs,
            });
          }
        }
        nextArc = t + 0.4 + Math.random() * 0.8;
      }

      // Spawn data rain particles
      if (Math.random() < 0.15) {
        particles.push(createParticle('data'));
      }

      // Spawn sparks from nucleus periodically
      if (t > nextSpark) {
        for (let s = 0; s < 3 + Math.floor(Math.random() * 5); s++) {
          particles.push(createParticle('spark', cx + (Math.random() - 0.5) * 10, cy + (Math.random() - 0.5) * 10));
        }
        nextSpark = t + 2 + Math.random() * 3;
      }

      // Cap particle count
      while (particles.length > 200) {
        const dataIdx = particles.findIndex(p => p.type === 'data' || p.type === 'spark');
        if (dataIdx >= 0) particles.splice(dataIdx, 1);
        else break;
      }

      // Draw layers (back to front)
      drawGrid();
      drawWaves();
      drawOrbits();
      drawConnections();
      drawPulseRings();
      drawEnergyArcs();
      drawParticles();
      drawNucleus();
      drawHexStream();
      drawHUD();

      animRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener('resize', resize);
    };
  }, [inline]);

  return (
    <div className={`${inline ? 'absolute inset-0' : 'fixed inset-0 z-[9999]'} ${backgroundOnly ? '' : 'bg-[#060810]'} flex flex-col items-center justify-center overflow-hidden`}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

      {/* Center overlay — Q logo + status (hidden in backgroundOnly mode) */}
      {!backgroundOnly && (
        <div className="relative z-10 flex flex-col items-center gap-3 pointer-events-none">
          <div
            className="w-14 h-14 rounded-xl flex items-center justify-center"
            style={{
              background: 'linear-gradient(135deg, rgba(212,175,55,0.15), rgba(255,215,0,0.1))',
              border: '1px solid rgba(212,175,55,0.25)',
              boxShadow: '0 0 40px rgba(212,175,55,0.15), 0 0 80px rgba(212,175,55,0.05)',
              backdropFilter: 'blur(8px)',
            }}
          >
            <span className="text-amber-400 font-bold text-2xl" style={{ textShadow: '0 0 15px rgba(212,175,55,0.5)' }}>Q</span>
          </div>

          <p className="text-sm text-gray-300/60 font-mono tracking-wide">{message}</p>
          <p className="text-[10px] text-cyan-500/30 font-mono tracking-wider h-3 transition-opacity duration-300">{statusText}</p>

          {progress !== undefined ? (
            <div className="w-48 mt-1">
              <div className="h-[2px] w-full bg-gray-800/40 rounded-full overflow-hidden">
                <div className="h-full rounded-full" style={{
                  width: `${progress}%`,
                  background: 'linear-gradient(90deg, #d4af37, #06b6d4, #8b5cf6)',
                  boxShadow: '0 0 8px rgba(212,175,55,0.5)',
                  transition: 'width 0.3s ease-out',
                }} />
              </div>
            </div>
          ) : (
            <div className="flex gap-1.5 mt-1">
              {[0, 1, 2, 3, 4, 5, 6].map(i => (
                <div key={i} className="w-[3px] h-[3px] rounded-full" style={{
                  backgroundColor: `hsla(${40 + i * 20}, 70%, 55%, 0.5)`,
                  animation: `qDot 1.2s ease-in-out ${i * 0.1}s infinite`,
                }} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Bottom telemetry (hidden in backgroundOnly mode) */}
      {!backgroundOnly && (
        <div className="absolute bottom-4 left-0 right-0 flex justify-center z-10">
          <div className="flex gap-6 text-[8px] font-mono tracking-[0.2em] uppercase text-gray-600/40">
            <span>CONSENSUS: <span className="text-cyan-500/40">ACTIVE</span></span>
            <span className="text-gray-800/20">|</span>
            <span>P2P: <span className="text-green-500/40">CONNECTED</span></span>
            <span className="text-gray-800/20">|</span>
            <span>PQ-CRYPTO: <span className="text-purple-400/40">READY</span></span>
          </div>
        </div>
      )}

      {!backgroundOnly && <style>{`
        @keyframes qDot {
          0%, 100% { transform: scaleY(0.4); opacity: 0.2; }
          50% { transform: scaleY(3); opacity: 0.9; }
        }
      `}</style>}
    </div>
  );
};

export default QuantumLoader;
