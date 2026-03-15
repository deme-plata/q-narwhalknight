/**
 * v9.6.0: Wicked quantum transaction processing animation
 *
 * Multi-stage visualization showing the actual cryptographic pipeline:
 * 1. Signing with Dilithium5 (post-quantum lattice)
 * 2. STARK proof generation (zero-knowledge)
 * 3. P2P gossipsub broadcast
 * 4. Validator consensus (2f+1 BFT)
 *
 * Uses canvas for particle effects + framer-motion for stage transitions
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Radio, Check, Zap, Lock, Cpu, Globe } from 'lucide-react';

interface ProcessingStage {
  id: string;
  label: string;
  sublabel: string;
  icon: typeof Shield;
  color: string;
  glowColor: string;
  durationMs: number;
}

const STAGES: ProcessingStage[] = [
  {
    id: 'signing',
    label: 'Signing Transaction',
    sublabel: 'Dilithium5 post-quantum lattice signature',
    icon: Lock,
    color: '#D4AF37',
    glowColor: 'rgba(212, 175, 55, 0.6)',
    durationMs: 800,
  },
  {
    id: 'proof',
    label: 'Generating STARK Proof',
    sublabel: 'Zero-knowledge cryptographic commitment',
    icon: Shield,
    color: '#8B5CF6',
    glowColor: 'rgba(139, 92, 246, 0.6)',
    durationMs: 1200,
  },
  {
    id: 'broadcast',
    label: 'Broadcasting to Network',
    sublabel: 'P2P gossipsub mesh propagation',
    icon: Radio,
    color: '#06B6D4',
    glowColor: 'rgba(6, 182, 212, 0.6)',
    durationMs: 600,
  },
  {
    id: 'consensus',
    label: 'Awaiting Consensus',
    sublabel: 'DAG-Knight BFT validator confirmation',
    icon: Globe,
    color: '#22C55E',
    glowColor: 'rgba(34, 197, 94, 0.6)',
    durationMs: 1500,
  },
];

// Particle types for the canvas animation
interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  alpha: number;
  color: string;
  life: number;
  maxLife: number;
  type: 'spark' | 'ring' | 'trail' | 'hex';
}

interface TransactionProcessingAnimationProps {
  isActive: boolean;
  /** Called when animation reaches "complete" visual state (does NOT control actual tx) */
  onVisualComplete?: () => void;
  /** If true, show success burst animation */
  showSuccess?: boolean;
}

export default function TransactionProcessingAnimation({
  isActive,
  onVisualComplete,
  showSuccess,
}: TransactionProcessingAnimationProps) {
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [stageProgress, setStageProgress] = useState(0);
  const [overallProgress, setOverallProgress] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animFrameRef = useRef<number>(0);
  const stageStartRef = useRef<number>(Date.now());

  // Advance through stages
  useEffect(() => {
    if (!isActive) {
      setCurrentStageIndex(0);
      setStageProgress(0);
      setOverallProgress(0);
      return;
    }

    stageStartRef.current = Date.now();

    const totalDuration = STAGES.reduce((sum, s) => sum + s.durationMs, 0);
    let elapsed = 0;
    for (let i = 0; i < currentStageIndex; i++) {
      elapsed += STAGES[i].durationMs;
    }

    const interval = setInterval(() => {
      const stageElapsed = Date.now() - stageStartRef.current;
      const stageDuration = STAGES[currentStageIndex]?.durationMs || 1000;
      const progress = Math.min(1, stageElapsed / stageDuration);
      setStageProgress(progress);

      const overall = Math.min(1, (elapsed + stageElapsed) / totalDuration);
      setOverallProgress(overall);

      if (progress >= 1 && currentStageIndex < STAGES.length - 1) {
        setCurrentStageIndex(prev => prev + 1);
        stageStartRef.current = Date.now();
        elapsed += stageDuration;
      } else if (progress >= 1 && currentStageIndex === STAGES.length - 1) {
        onVisualComplete?.();
        clearInterval(interval);
      }
    }, 30);

    return () => clearInterval(interval);
  }, [isActive, currentStageIndex, onVisualComplete]);

  // Reset when becoming active
  useEffect(() => {
    if (isActive) {
      setCurrentStageIndex(0);
      setStageProgress(0);
      setOverallProgress(0);
      stageStartRef.current = Date.now();
    }
  }, [isActive]);

  // Canvas particle animation
  const spawnParticles = useCallback((color: string, cx: number, cy: number, count: number) => {
    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count + Math.random() * 0.5;
      const speed = 1 + Math.random() * 3;
      const types: Particle['type'][] = ['spark', 'ring', 'trail', 'hex'];
      particlesRef.current.push({
        x: cx,
        y: cy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        size: 1 + Math.random() * 3,
        alpha: 0.8 + Math.random() * 0.2,
        color,
        life: 0,
        maxLife: 40 + Math.random() * 60,
        type: types[Math.floor(Math.random() * types.length)],
      });
    }
  }, []);

  useEffect(() => {
    if (!isActive || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * 2;
      canvas.height = rect.height * 2;
      ctx.scale(2, 2);
    };
    resize();

    let frame = 0;

    const draw = () => {
      const w = canvas.width / 2;
      const h = canvas.height / 2;
      ctx.clearRect(0, 0, w, h);

      const stage = STAGES[currentStageIndex];
      if (!stage) return;

      // Spawn particles from center
      if (frame % 3 === 0) {
        spawnParticles(stage.color, w / 2, h / 2, 2);
      }

      // Success burst
      if (showSuccess && frame === 0) {
        spawnParticles('#22C55E', w / 2, h / 2, 40);
        spawnParticles('#D4AF37', w / 2, h / 2, 20);
      }

      // Draw orbital ring
      const ringRadius = 30 + Math.sin(frame * 0.02) * 5;
      ctx.strokeStyle = stage.color + '30';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, ringRadius, 0, Math.PI * 2);
      ctx.stroke();

      // Draw rotating marker on ring
      const markerAngle = frame * 0.05;
      const mx = w / 2 + Math.cos(markerAngle) * ringRadius;
      const my = h / 2 + Math.sin(markerAngle) * ringRadius;
      ctx.fillStyle = stage.color;
      ctx.beginPath();
      ctx.arc(mx, my, 2, 0, Math.PI * 2);
      ctx.fill();

      // Second ring (counter-rotating)
      const ring2Radius = 45 + Math.cos(frame * 0.015) * 8;
      ctx.strokeStyle = stage.color + '18';
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, ring2Radius, 0, Math.PI * 2);
      ctx.stroke();

      const m2Angle = -frame * 0.03;
      const m2x = w / 2 + Math.cos(m2Angle) * ring2Radius;
      const m2y = h / 2 + Math.sin(m2Angle) * ring2Radius;
      ctx.fillStyle = stage.color + '80';
      ctx.beginPath();
      ctx.arc(m2x, m2y, 1.5, 0, Math.PI * 2);
      ctx.fill();

      // Draw connection lines between orbital markers
      ctx.strokeStyle = stage.color + '15';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(mx, my);
      ctx.lineTo(m2x, m2y);
      ctx.stroke();

      // Update and draw particles
      particlesRef.current = particlesRef.current.filter(p => {
        p.life++;
        if (p.life > p.maxLife) return false;

        p.x += p.vx;
        p.y += p.vy;
        p.vx *= 0.97;
        p.vy *= 0.97;
        p.alpha = Math.max(0, 1 - p.life / p.maxLife);

        ctx.globalAlpha = p.alpha * 0.7;

        if (p.type === 'spark') {
          ctx.fillStyle = p.color;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * p.alpha, 0, Math.PI * 2);
          ctx.fill();
        } else if (p.type === 'ring') {
          ctx.strokeStyle = p.color;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * 2 * (1 - p.alpha) + 1, 0, Math.PI * 2);
          ctx.stroke();
        } else if (p.type === 'trail') {
          ctx.fillStyle = p.color;
          const len = 3;
          for (let t = 0; t < len; t++) {
            const trailAlpha = p.alpha * (1 - t / len);
            ctx.globalAlpha = trailAlpha * 0.5;
            ctx.beginPath();
            ctx.arc(p.x - p.vx * t * 2, p.y - p.vy * t * 2, p.size * 0.6, 0, Math.PI * 2);
            ctx.fill();
          }
        } else if (p.type === 'hex') {
          ctx.strokeStyle = p.color;
          ctx.lineWidth = 0.5;
          const s = p.size * 1.5;
          ctx.beginPath();
          for (let i = 0; i < 6; i++) {
            const angle = (Math.PI / 3) * i + frame * 0.02;
            const hx = p.x + Math.cos(angle) * s;
            const hy = p.y + Math.sin(angle) * s;
            i === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
          }
          ctx.closePath();
          ctx.stroke();
        }

        ctx.globalAlpha = 1;
        return true;
      });

      // Center glow pulse
      const pulseSize = 8 + Math.sin(frame * 0.08) * 3;
      const gradient = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, pulseSize);
      gradient.addColorStop(0, stage.color + '40');
      gradient.addColorStop(1, stage.color + '00');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, pulseSize, 0, Math.PI * 2);
      ctx.fill();

      frame++;
      animFrameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      particlesRef.current = [];
    };
  }, [isActive, currentStageIndex, showSuccess, spawnParticles]);

  if (!isActive && !showSuccess) return null;

  const currentStage = STAGES[currentStageIndex];
  const StageIcon = currentStage?.icon || Shield;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: -10 }}
        transition={{ duration: 0.3 }}
        className="relative overflow-hidden rounded-2xl p-6"
        style={{
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 20, 50, 0.9) 100%)',
          border: `1px solid ${currentStage?.color || '#D4AF37'}30`,
          boxShadow: `0 0 40px ${currentStage?.glowColor || 'rgba(212,175,55,0.2)'}`,
        }}
      >
        {/* Canvas particle layer */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ opacity: 0.6 }}
        />

        {/* Content overlay */}
        <div className="relative z-10">
          {/* Stage indicator dots */}
          <div className="flex items-center justify-center gap-2 mb-5">
            {STAGES.map((stage, i) => (
              <div key={stage.id} className="flex items-center gap-2">
                <motion.div
                  className="relative"
                  animate={{
                    scale: i === currentStageIndex ? [1, 1.2, 1] : 1,
                  }}
                  transition={{
                    duration: 1,
                    repeat: i === currentStageIndex ? Infinity : 0,
                  }}
                >
                  <div
                    className="w-3 h-3 rounded-full transition-all duration-300"
                    style={{
                      background:
                        i < currentStageIndex
                          ? stage.color
                          : i === currentStageIndex
                          ? `linear-gradient(135deg, ${stage.color}, ${stage.color}80)`
                          : 'rgba(255,255,255,0.1)',
                      boxShadow:
                        i <= currentStageIndex
                          ? `0 0 8px ${stage.color}80`
                          : 'none',
                    }}
                  />
                  {i < currentStageIndex && (
                    <Check
                      className="absolute -top-0.5 -left-0.5 w-4 h-4"
                      style={{ color: stage.color }}
                    />
                  )}
                </motion.div>
                {i < STAGES.length - 1 && (
                  <div
                    className="w-8 h-0.5 rounded-full transition-all duration-500"
                    style={{
                      background:
                        i < currentStageIndex
                          ? `linear-gradient(90deg, ${STAGES[i].color}, ${STAGES[i + 1].color})`
                          : 'rgba(255,255,255,0.08)',
                    }}
                  />
                )}
              </div>
            ))}
          </div>

          {/* Main stage display */}
          <div className="text-center mb-5">
            <motion.div
              key={currentStage?.id}
              initial={{ opacity: 0, y: 10, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.3 }}
              className="flex flex-col items-center"
            >
              <motion.div
                animate={{ rotate: currentStageIndex === 1 ? 360 : 0 }}
                transition={{
                  duration: 2,
                  repeat: currentStageIndex === 1 ? Infinity : 0,
                  ease: 'linear',
                }}
                className="w-14 h-14 rounded-2xl flex items-center justify-center mb-3"
                style={{
                  background: `linear-gradient(135deg, ${currentStage?.color}25, ${currentStage?.color}10)`,
                  border: `1px solid ${currentStage?.color}40`,
                  boxShadow: `0 0 20px ${currentStage?.glowColor}`,
                }}
              >
                <StageIcon
                  className="w-7 h-7"
                  style={{ color: currentStage?.color }}
                />
              </motion.div>

              <h3
                className="text-lg font-bold mb-1"
                style={{ color: currentStage?.color }}
              >
                {showSuccess ? 'Transaction Confirmed!' : currentStage?.label}
              </h3>
              <p className="text-sm text-gray-400">
                {showSuccess
                  ? 'Quantum-secured and verified by consensus'
                  : currentStage?.sublabel}
              </p>
            </motion.div>
          </div>

          {/* Progress bar */}
          <div className="relative h-1.5 rounded-full overflow-hidden mb-3"
            style={{ background: 'rgba(255,255,255,0.06)' }}
          >
            <motion.div
              className="absolute inset-y-0 left-0 rounded-full"
              style={{
                background: `linear-gradient(90deg, ${STAGES[0].color}, ${currentStage?.color || STAGES[0].color})`,
                boxShadow: `0 0 12px ${currentStage?.glowColor}`,
              }}
              animate={{ width: `${overallProgress * 100}%` }}
              transition={{ duration: 0.1, ease: 'linear' }}
            />
            {/* Shimmer effect */}
            <motion.div
              className="absolute inset-y-0 w-20 rounded-full"
              style={{
                background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
              }}
              animate={{ x: [-80, 400] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
            />
          </div>

          {/* Stage details row */}
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>
              Stage {currentStageIndex + 1}/{STAGES.length}
            </span>
            <span className="font-mono">
              {Math.round(overallProgress * 100)}%
            </span>
          </div>

          {/* Crypto details (scrolling ticker) */}
          <motion.div
            className="mt-3 overflow-hidden h-5 relative"
            style={{
              maskImage: 'linear-gradient(90deg, transparent, black 10%, black 90%, transparent)',
            }}
          >
            <motion.div
              className="flex items-center gap-6 text-xs font-mono whitespace-nowrap absolute"
              animate={{ x: [0, -600] }}
              transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
              style={{ color: currentStage?.color + '60' }}
            >
              <span>DILITHIUM5-SHA3-256</span>
              <span>LATTICE-NIST-PQ-L5</span>
              <span>STARK-FRI-PROOF</span>
              <span>GOSSIPSUB-MESH-v1.1</span>
              <span>DAG-KNIGHT-BFT-2f+1</span>
              <span>KYBER-1024-KEM</span>
              <span>BLAKE3-MERKLE-ROOT</span>
              <span>NOISE-XX-ENCRYPTED</span>
              <span>DILITHIUM5-SHA3-256</span>
              <span>LATTICE-NIST-PQ-L5</span>
              <span>STARK-FRI-PROOF</span>
              <span>GOSSIPSUB-MESH-v1.1</span>
            </motion.div>
          </motion.div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
