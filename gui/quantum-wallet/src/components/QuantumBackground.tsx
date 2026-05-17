import { useEffect, useRef, memo } from 'react';

// Performance-optimized quantum background
// v2.4.0: Reduced from 50 to 15 particles, 30fps cap, spatial optimization
const QuantumBackground = memo(function QuantumBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const lastFrameTime = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    // Use device pixel ratio for crisp rendering but cap at 1 for performance
    const dpr = Math.min(window.devicePixelRatio || 1, 1);

    const resize = () => {
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.scale(dpr, dpr);
    };
    resize();

    // Reduced particle count for performance (50 → 15)
    const PARTICLE_COUNT = 15;
    const CONNECTION_DISTANCE = 120;
    const TARGET_FPS = 30;
    const FRAME_INTERVAL = 1000 / TARGET_FPS;

    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      color: string;
    }> = [];

    const colors = ['#D4AF37', '#FFD700', '#FFA500', '#8B5CF6'];

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particles.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        size: Math.random() * 2 + 1,
        color: colors[Math.floor(Math.random() * colors.length)]
      });
    }

    const animate = (currentTime: number) => {
      // Frame rate limiting - skip frames to maintain 30fps
      const elapsed = currentTime - lastFrameTime.current;
      if (elapsed < FRAME_INTERVAL) {
        animationRef.current = requestAnimationFrame(animate);
        return;
      }
      lastFrameTime.current = currentTime - (elapsed % FRAME_INTERVAL);

      const width = window.innerWidth;
      const height = window.innerHeight;

      ctx.clearRect(0, 0, width, height);

      // Update and draw particles
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;

        // Bounce off walls
        if (p.x < 0 || p.x > width) p.vx *= -1;
        if (p.y < 0 || p.y > height) p.vy *= -1;

        // Draw particle
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();

        // Only check connections with particles ahead in array (avoid duplicates)
        // This reduces checks from n² to n(n-1)/2
        for (let j = i + 1; j < particles.length; j++) {
          const other = particles[j];
          const dx = p.x - other.x;
          const dy = p.y - other.y;

          // Quick distance check using squared distance (avoid sqrt)
          const distSq = dx * dx + dy * dy;
          const maxDistSq = CONNECTION_DISTANCE * CONNECTION_DISTANCE;

          if (distSq < maxDistSq) {
            const alpha = 0.15 * (1 - distSq / maxDistSq);
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(other.x, other.y);
            ctx.strokeStyle = `rgba(212, 175, 55, ${alpha})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    window.addEventListener('resize', resize, { passive: true });

    return () => {
      cancelAnimationFrame(animationRef.current);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="quantum-bg-canvas"
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: 0,
        opacity: 0.25,
        pointerEvents: 'none',
        willChange: 'auto', // Let browser decide, avoid constant compositing
      }}
    />
  );
});

export default QuantumBackground;
