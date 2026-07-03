import { useEffect, useRef } from 'react';

// --- Quantum Field Particle Canvas ---
// Reusable canvas-based particle animation from the login screen.
// Renders twinkling stars, nebula clouds, physics-based string particles
// with resonance coupling, and network pulse rings.

interface StringParticle {
  x: number; y: number;
  amplitude: number;
  frequency: number;
  phase: number;
  mode: number;
  vx: number; vy: number;
  radius: number;
  life: number;
  type: 'honest' | 'resonance' | 'entangled' | 'finalized';
}

interface NetworkPulse {
  x: number; y: number;
  radius: number;
  maxRadius: number;
  alpha: number;
  color: string;
}

interface QuantumParticleCanvasProps {
  className?: string;
  style?: React.CSSProperties;
  /** Number of background stars (default 200) */
  starCount?: number;
  /** Max concurrent particles (default 80) */
  maxParticles?: number;
  /** Initial seed particles (default 40) */
  seedParticles?: number;
  /** Global opacity multiplier (default 0.7) */
  opacity?: number;
}

export default function QuantumParticleCanvas({
  className = '',
  style,
  starCount = 200,
  maxParticles = 80,
  seedParticles = 40,
  opacity = 0.7,
}: QuantumParticleCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let w = canvas.width = parent.clientWidth;
    let h = canvas.height = parent.clientHeight;

    const ro = new ResizeObserver(() => {
      w = canvas.width = parent.clientWidth;
      h = canvas.height = parent.clientHeight;
    });
    ro.observe(parent);

    // Particle system
    const particles: StringParticle[] = [];
    const pulses: NetworkPulse[] = [];
    const bgStars: Array<{ x: number; y: number; r: number; a: number; s: number }> = [];

    for (let i = 0; i < starCount; i++) {
      bgStars.push({
        x: Math.random() * w, y: Math.random() * h,
        r: Math.random() * 1.2 + 0.2,
        a: Math.random() * 0.6 + 0.2,
        s: Math.random() * 0.015 + 0.003,
      });
    }

    const spawnParticle = (type: StringParticle['type'] = 'honest') => {
      const stakeWeight = 0.3 + Math.random() * 0.7;
      const priority = Math.random();
      const colors: Record<string, number[]> = {
        honest: [0.55, 0.65],
        resonance: [0.08, 0.15],
        entangled: [0.48, 0.52],
        finalized: [0.75, 0.85],
      };
      const hueRange = colors[type];
      const phaseHue = hueRange[0] + Math.random() * (hueRange[1] - hueRange[0]);

      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        amplitude: Math.sqrt(stakeWeight) * (2.5 + Math.random() * 2.5),
        frequency: 2 * Math.PI * (priority * 0.8 + 0.2),
        phase: phaseHue * Math.PI * 2,
        mode: Math.floor(Math.random() * 4) + 1,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.3 - 0.1,
        radius: Math.sqrt(stakeWeight) * (1.5 + Math.random() * 2),
        life: 1.0,
        type,
      });

      if (Math.random() < 0.15) {
        pulses.push({
          x: Math.random() * w, y: Math.random() * h,
          radius: 0, maxRadius: 60 + Math.random() * 80,
          alpha: 0.25, color: type === 'finalized' ? '#9455F7' : '#D4AF37',
        });
      }
    };

    // Seed initial particles
    const types: StringParticle['type'][] = ['honest', 'resonance', 'entangled', 'finalized'];
    for (let i = 0; i < seedParticles; i++) {
      spawnParticle(types[Math.floor(Math.random() * types.length)]);
      particles[particles.length - 1].life = Math.random();
    }

    let frame = 0;
    let lastSpawn = 0;

    const draw = () => {
      frame++;
      const t = frame * 0.016;
      ctx.clearRect(0, 0, w, h);

      // Nebula clouds
      const nebulaPositions = [
        { x: w * 0.2, y: h * 0.3, rx: 250, ry: 160, c: 'rgba(59,130,246,0.05)' },
        { x: w * 0.7, y: h * 0.6, rx: 280, ry: 200, c: 'rgba(139,92,246,0.04)' },
        { x: w * 0.5, y: h * 0.15, rx: 220, ry: 140, c: 'rgba(212,175,55,0.035)' },
        { x: w * 0.85, y: h * 0.2, rx: 160, ry: 120, c: 'rgba(6,182,212,0.04)' },
      ];
      for (const neb of nebulaPositions) {
        const breathe = 1 + Math.sin(t * 0.15 + neb.x * 0.01) * 0.08;
        const ng = ctx.createRadialGradient(neb.x, neb.y, 0, neb.x, neb.y, neb.rx * breathe);
        ng.addColorStop(0, neb.c);
        ng.addColorStop(0.6, neb.c.replace(/[\d.]+\)$/, '0.005)'));
        ng.addColorStop(1, 'transparent');
        ctx.fillStyle = ng;
        ctx.save();
        ctx.scale(1, neb.ry / neb.rx);
        ctx.beginPath();
        ctx.arc(neb.x, neb.y * (neb.rx / neb.ry), neb.rx * breathe, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // Background stars with twinkling
      for (const s of bgStars) {
        const twinkle = Math.sin(frame * s.s + s.x * 0.1) * 0.3 + 0.7;
        ctx.globalAlpha = Math.min(1, s.a * twinkle * 1.5);
        ctx.fillStyle = '#fff8e8';
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
        if (s.r > 0.8) {
          const sg = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r * 4);
          sg.addColorStop(0, `rgba(255,250,230,${0.25 * twinkle})`);
          sg.addColorStop(0.5, `rgba(255,248,220,${0.08 * twinkle})`);
          sg.addColorStop(1, 'transparent');
          ctx.fillStyle = sg;
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.r * 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;

      // Spawn particles
      if (frame - lastSpawn > 10) {
        lastSpawn = frame;
        const weights = [0.4, 0.25, 0.2, 0.15];
        let r = Math.random();
        let chosen: StringParticle['type'] = 'honest';
        for (let i = 0; i < types.length; i++) {
          r -= weights[i];
          if (r <= 0) { chosen = types[i]; break; }
        }
        if (particles.length < maxParticles) spawnParticle(chosen);
      }

      // Resonance coupling lines
      const coupledPairs: Array<[number, number, number]> = [];
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const pi = particles[i], pj = particles[j];
          const dx = pi.x - pj.x, dy = pi.y - pj.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > 120) continue;
          const phaseDiff = pi.phase - pj.phase;
          const freqDiff = Math.abs(pi.frequency - pj.frequency);
          const resonance = pi.amplitude * pj.amplitude *
            Math.pow(Math.cos(phaseDiff / 2), 2) *
            Math.exp(-freqDiff * freqDiff / 2);
          if (resonance > 1.5) coupledPairs.push([i, j, resonance]);
        }
      }

      for (const [i, j, R] of coupledPairs) {
        const pi = particles[i], pj = particles[j];
        const lineAlpha = Math.min(0.4, R * 0.07) * Math.min(pi.life, pj.life);
        const hue = ((pi.phase + pj.phase) / 2 / (Math.PI * 2)) * 360;
        ctx.strokeStyle = `hsla(${hue}, 75%, 70%, ${lineAlpha})`;
        ctx.lineWidth = 0.7 + R * 0.15;
        ctx.beginPath();
        ctx.moveTo(pi.x, pi.y);
        const mx = (pi.x + pj.x) / 2 + Math.sin(t + i) * 8;
        const my = (pi.y + pj.y) / 2 + Math.cos(t + j) * 8;
        ctx.quadraticCurveTo(mx, my, pj.x, pj.y);
        ctx.stroke();
      }

      // Update and draw particles
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        const psi = p.amplitude * Math.sin(p.frequency * t + p.phase) *
          Math.sin((p.mode * Math.PI * p.x) / w);

        p.x += p.vx + psi * 0.15;
        p.y += p.vy + Math.cos(p.frequency * t * 0.7 + p.phase) * 0.12;

        if (p.x < -20) p.x = w + 20;
        if (p.x > w + 20) p.x = -20;
        if (p.y < -20) p.y = h + 20;
        if (p.y > h + 20) p.y = -20;

        p.life -= 0.0008;
        if (p.life <= 0) { particles.splice(i, 1); continue; }

        const alpha = Math.min(1, p.life * 3) * Math.min(1, (1 - p.life) * 5);
        const hue = (p.phase / (Math.PI * 2)) * 360;
        const pulseR = p.radius * (1 + Math.sin(p.frequency * t + p.phase) * 0.25);

        // Outer glow
        const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, pulseR * 6);
        glow.addColorStop(0, `hsla(${hue}, 85%, 75%, ${alpha * 0.4})`);
        glow.addColorStop(0.3, `hsla(${hue}, 75%, 60%, ${alpha * 0.12})`);
        glow.addColorStop(0.6, `hsla(${hue}, 70%, 55%, ${alpha * 0.03})`);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(p.x, p.y, pulseR * 6, 0, Math.PI * 2);
        ctx.fill();

        // Core particle
        ctx.globalAlpha = alpha;
        const coreGrad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, pulseR * 1.3);
        coreGrad.addColorStop(0, `hsla(${hue}, 95%, 92%, 1)`);
        coreGrad.addColorStop(0.35, `hsla(${hue}, 85%, 72%, 0.9)`);
        coreGrad.addColorStop(0.7, `hsla(${hue}, 75%, 55%, 0.4)`);
        coreGrad.addColorStop(1, `hsla(${hue}, 70%, 50%, 0)`);
        ctx.fillStyle = coreGrad;
        ctx.beginPath();
        ctx.arc(p.x, p.y, pulseR, 0, Math.PI * 2);
        ctx.fill();

        // Cross-spike for finalized particles — enhanced with 8-point starburst
        if (p.type === 'finalized' && alpha > 0.4) {
          const sLen = pulseR * 3;
          ctx.strokeStyle = `hsla(${hue}, 90%, 80%, ${alpha * 0.5})`;
          ctx.lineWidth = 0.8;
          for (let ray = 0; ray < 8; ray++) {
            const angle = (ray * Math.PI) / 4 + t * 0.3;
            const len = ray % 2 === 0 ? sLen : sLen * 0.6;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p.x + Math.cos(angle) * len, p.y + Math.sin(angle) * len);
            ctx.stroke();
          }
        }

        // Entangled particles — quantum orbital rings
        if (p.type === 'entangled' && alpha > 0.3) {
          ctx.strokeStyle = `hsla(${hue}, 85%, 70%, ${alpha * 0.25})`;
          ctx.lineWidth = 0.5;
          const orbitR = pulseR * 4;
          ctx.save();
          ctx.translate(p.x, p.y);
          ctx.rotate(t * 0.5 + p.phase);
          ctx.scale(1, 0.35);
          ctx.beginPath();
          ctx.arc(0, 0, orbitR, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
          ctx.save();
          ctx.translate(p.x, p.y);
          ctx.rotate(-t * 0.4 + p.phase + 1.2);
          ctx.scale(0.35, 1);
          ctx.beginPath();
          ctx.arc(0, 0, orbitR * 0.8, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }
        ctx.globalAlpha = 1;
      }

      // Firework bursts — triggered randomly every ~4 seconds
      if (frame % 240 === 0 && Math.random() < 0.6) {
        const fx = Math.random() * w * 0.6 + w * 0.2;
        const fy = Math.random() * h * 0.4 + h * 0.1;
        const burstHue = Math.random() * 360;
        const burstCount = 20 + Math.floor(Math.random() * 15);
        for (let b = 0; b < burstCount; b++) {
          const angle = (b / burstCount) * Math.PI * 2 + Math.random() * 0.3;
          const speed = 1.5 + Math.random() * 2.5;
          const trail = Math.random() < 0.3;
          particles.push({
            x: fx, y: fy,
            amplitude: 0.5 + Math.random() * 0.5,
            frequency: 0.5,
            phase: (burstHue / 360) * Math.PI * 2 + (trail ? Math.random() * 0.5 : 0),
            mode: 1,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed - 0.5,
            radius: trail ? 0.8 : 1.5 + Math.random() * 1.5,
            life: 0.4 + Math.random() * 0.3,
            type: 'finalized',
          });
        }
        // Burst flash
        pulses.push({
          x: fx, y: fy,
          radius: 0, maxRadius: 100 + Math.random() * 60,
          alpha: 0.5, color: `hsl(${burstHue}, 90%, 75%)`,
        });
      }

      // Quantum metadata streams — flowing data threads between particles
      if (particles.length > 5) {
        ctx.globalAlpha = 0.15;
        ctx.lineWidth = 0.4;
        for (let s = 0; s < Math.min(3, particles.length - 1); s++) {
          const src = particles[s * 3 % particles.length];
          const dst = particles[(s * 3 + 2) % particles.length];
          if (!src || !dst) continue;
          const srcHue = (src.phase / (Math.PI * 2)) * 360;
          ctx.strokeStyle = `hsla(${srcHue}, 70%, 65%, 0.3)`;
          ctx.beginPath();
          ctx.moveTo(src.x, src.y);
          // Flowing sine wave path between particles
          const segments = 12;
          for (let seg = 1; seg <= segments; seg++) {
            const frac = seg / segments;
            const mx = src.x + (dst.x - src.x) * frac;
            const my = src.y + (dst.y - src.y) * frac +
              Math.sin(frac * Math.PI * 3 + t * 2) * 8 * (1 - Math.abs(frac - 0.5) * 2);
            ctx.lineTo(mx, my);
          }
          ctx.stroke();
          // Data packet dots moving along the stream
          const packetPos = (t * 0.3 + s * 0.33) % 1;
          const px = src.x + (dst.x - src.x) * packetPos;
          const py = src.y + (dst.y - src.y) * packetPos +
            Math.sin(packetPos * Math.PI * 3 + t * 2) * 8 * (1 - Math.abs(packetPos - 0.5) * 2);
          ctx.fillStyle = `hsla(${srcHue}, 90%, 80%, 0.6)`;
          ctx.beginPath();
          ctx.arc(px, py, 1.5, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      // Network pulses — enhanced with double-ring effect
      for (let i = pulses.length - 1; i >= 0; i--) {
        const pulse = pulses[i];
        pulse.radius += 1.8;
        pulse.alpha -= 0.005;
        if (pulse.alpha <= 0 || pulse.radius > pulse.maxRadius) {
          pulses.splice(i, 1); continue;
        }
        if (pulse.color.startsWith('#') || pulse.color.startsWith('hsl')) {
          if (pulse.color.startsWith('#')) {
            const r = parseInt(pulse.color.slice(1, 3), 16);
            const g = parseInt(pulse.color.slice(3, 5), 16);
            const b = parseInt(pulse.color.slice(5, 7), 16);
            ctx.strokeStyle = `rgba(${r},${g},${b},${pulse.alpha})`;
          } else {
            ctx.strokeStyle = pulse.color.replace(')', `, ${pulse.alpha})`).replace('hsl(', 'hsla(');
          }
        }
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(pulse.x, pulse.y, pulse.radius, 0, Math.PI * 2);
        ctx.stroke();
        // Inner echo ring
        if (pulse.radius > 15) {
          ctx.globalAlpha = pulse.alpha * 0.3;
          ctx.beginPath();
          ctx.arc(pulse.x, pulse.y, pulse.radius * 0.6, 0, Math.PI * 2);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      cancelAnimationFrame(animationRef.current);
      ro.disconnect();
    };
  }, [starCount, maxParticles, seedParticles]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 w-full h-full pointer-events-none ${className}`}
      style={{ opacity, mixBlendMode: 'screen', ...style }}
    />
  );
}
