import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Trophy, Bug, Globe, Users, Pickaxe, Zap, X, ChevronRight, ChevronLeft,
  ExternalLink, Shield, Target, Star, Flame, Award, TrendingUp, Gift,
  MessageCircle, Github, Twitter, ArrowRight
} from 'lucide-react';

const STORAGE_KEY = 'bounty_modal_seen_v1';
const BOUNTY_URL = 'https://bounty.quillon.xyz';

interface BountyModalProps {
  onClose: () => void;
  /** Ref to the TopBar bounty button — genie animation flies into this element */
  genieTargetRef?: React.RefObject<HTMLElement | null>;
}

// ─── Hexagonal Particle Grid ──────────────────────────────────────────────
const HexParticle: React.FC<{ index: number; total: number }> = ({ index, total }) => {
  const seed = useMemo(() => {
    const angle = (index / total) * Math.PI * 2 + (Math.random() - 0.5) * 1.2;
    return {
      angle,
      radius: 15 + Math.random() * 100,
      orbitSpeed: 6 + Math.random() * 18,
      size: 1.2 + Math.random() * 4.5,
      opacity: 0.25 + Math.random() * 0.75,
      delay: Math.random() * 4,
      // Emerald/Teal/Rose/Orange/Violet palette — distinctly NOT the gold/cyan of WelcomeMainnet
      color: [
        '#10B981', '#14B8A6', '#F43F5E', '#F97316', '#8B5CF6',
        '#06B6D4', '#EC4899', '#22D3EE', '#A78BFA', '#34D399',
      ][index % 10],
      drift: (Math.random() - 0.5) * 40,
      wobble: Math.random() * 20,
    };
  }, [index, total]);

  return (
    <motion.div
      initial={{ x: 0, y: 0, opacity: 0, scale: 0 }}
      animate={{
        x: [
          Math.cos(seed.angle) * seed.radius * 0.1,
          Math.cos(seed.angle + 1.0) * seed.radius + seed.drift,
          Math.cos(seed.angle + 2.0) * seed.radius * 0.5 + seed.wobble,
          Math.cos(seed.angle + 3.0) * seed.radius * 0.8,
          Math.cos(seed.angle + 4.0) * seed.radius * 0.2,
        ],
        y: [
          Math.sin(seed.angle) * seed.radius * 0.1,
          Math.sin(seed.angle + 1.0) * seed.radius + seed.drift,
          Math.sin(seed.angle + 2.0) * seed.radius * 0.5 - seed.wobble,
          Math.sin(seed.angle + 3.0) * seed.radius * 0.8,
          Math.sin(seed.angle + 4.0) * seed.radius * 0.2,
        ],
        opacity: [0, seed.opacity, seed.opacity * 0.6, seed.opacity * 0.9, 0],
        scale: [0, 1.4, 0.7, 1.1, 0],
      }}
      transition={{ duration: seed.orbitSpeed, delay: seed.delay, repeat: Infinity, ease: 'easeInOut' }}
      style={{
        position: 'absolute',
        width: seed.size,
        height: seed.size,
        borderRadius: '50%',
        background: `radial-gradient(circle, ${seed.color} 0%, ${seed.color}60 40%, transparent 70%)`,
        boxShadow: `0 0 ${seed.size * 4}px ${seed.color}50, 0 0 ${seed.size * 8}px ${seed.color}20`,
        filter: 'blur(0.2px)',
        pointerEvents: 'none',
      }}
    />
  );
};

// ─── Canvas Particle Field (Physics-based, from LoginScreen) ──────────────
const BountyParticleCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let w = canvas.offsetWidth;
    let h = canvas.offsetHeight;
    canvas.width = w * 2;
    canvas.height = h * 2;
    ctx.scale(2, 2);

    // Stars
    const stars = Array.from({ length: 150 }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: 0.3 + Math.random() * 1.2,
      s: 0.5 + Math.random() * 2,
    }));

    // Particles with bounty-themed colors
    interface BP { x: number; y: number; vx: number; vy: number; r: number; life: number; hue: number; sat: number }
    const particles: BP[] = [];
    const hues = [160, 170, 340, 25, 270, 185, 330, 145]; // emerald, teal, rose, orange, violet, cyan, pink, green

    let frame = 0;
    const loop = () => {
      frame++;
      ctx.clearRect(0, 0, w, h);

      // Twinkling stars
      for (const s of stars) {
        const twinkle = 0.3 + 0.7 * Math.abs(Math.sin(frame * 0.015 * s.s + s.x * 0.05));
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${twinkle * 0.6})`;
        ctx.fill();
        if (s.r > 0.8) {
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.r * 2.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(200,230,255,${twinkle * 0.08})`;
          ctx.fill();
        }
      }

      // Spawn
      if (frame % 6 === 0 && particles.length < 60) {
        const hue = hues[Math.floor(Math.random() * hues.length)];
        particles.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.6,
          vy: (Math.random() - 0.5) * 0.5,
          r: 1.5 + Math.random() * 3,
          life: 1,
          hue,
          sat: 70 + Math.random() * 25,
        });
      }

      // Update & draw particles
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.life -= 0.003;
        if (p.life <= 0) { particles.splice(i, 1); continue; }

        const alpha = Math.min(p.life, 0.8);
        // Outer glow
        const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 5);
        grd.addColorStop(0, `hsla(${p.hue},${p.sat}%,75%,${alpha * 0.4})`);
        grd.addColorStop(0.4, `hsla(${p.hue},${p.sat}%,55%,${alpha * 0.15})`);
        grd.addColorStop(1, `hsla(${p.hue},${p.sat}%,40%,0)`);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * 5, 0, Math.PI * 2);
        ctx.fillStyle = grd;
        ctx.fill();
        // Core
        const core = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 1.2);
        core.addColorStop(0, `hsla(${p.hue},95%,92%,${alpha})`);
        core.addColorStop(0.5, `hsla(${p.hue},90%,70%,${alpha * 0.7})`);
        core.addColorStop(1, `hsla(${p.hue},80%,50%,0)`);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * 1.2, 0, Math.PI * 2);
        ctx.fillStyle = core;
        ctx.fill();
      }

      // Coupling lines between nearby particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const a = particles[i], b = particles[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            const strength = (1 - dist / 120) * Math.min(a.life, b.life);
            if (strength > 0.15) {
              ctx.beginPath();
              ctx.moveTo(a.x, a.y);
              const mx = (a.x + b.x) / 2 + (Math.random() - 0.5) * 15;
              const my = (a.y + b.y) / 2 + (Math.random() - 0.5) * 15;
              ctx.quadraticCurveTo(mx, my, b.x, b.y);
              ctx.strokeStyle = `hsla(${(a.hue + b.hue) / 2},70%,60%,${strength * 0.25})`;
              ctx.lineWidth = 0.5;
              ctx.stroke();
            }
          }
        }
      }

      animRef.current = requestAnimationFrame(loop);
    };

    loop();
    const resizeObserver = new ResizeObserver(() => {
      w = canvas.offsetWidth;
      h = canvas.offsetHeight;
      canvas.width = w * 2;
      canvas.height = h * 2;
      ctx.setTransform(2, 0, 0, 2, 0, 0);
    });
    resizeObserver.observe(canvas);

    return () => {
      cancelAnimationFrame(animRef.current);
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', opacity: 0.65 }}
    />
  );
};

// ─── Animated Counter ──────────────────────────────────────────────────────
const AnimCounter: React.FC<{ target: number; duration?: number; prefix?: string; suffix?: string; decimals?: number }> = ({
  target, duration = 2000, prefix = '', suffix = '', decimals = 0,
}) => {
  const [value, setValue] = useState(0);
  const startRef = useRef(0);
  const animRef = useRef(0);

  useEffect(() => {
    startRef.current = performance.now();
    const tick = (now: number) => {
      const elapsed = now - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(eased * target);
      if (progress < 1) animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [target, duration]);

  return <span>{prefix}{decimals > 0 ? value.toFixed(decimals) : Math.round(value).toLocaleString()}{suffix}</span>;
};

// ─── Pulse Rings ───────────────────────────────────────────────────────────
const PulseRing: React.FC<{ delay: number; color: string; size: number }> = ({ delay, color, size }) => (
  <motion.div
    initial={{ scale: 0.3, opacity: 0 }}
    animate={{ scale: [0.3, 1.4, 0.3], opacity: [0, 0.35, 0] }}
    transition={{ duration: 4.5, delay, repeat: Infinity, ease: 'easeInOut' }}
    style={{
      position: 'absolute',
      width: size,
      height: size,
      borderRadius: '50%',
      border: `1.5px solid ${color}`,
      boxShadow: `0 0 15px ${color}15, inset 0 0 15px ${color}08`,
      pointerEvents: 'none',
    }}
  />
);

// ─── Shooting Stars ────────────────────────────────────────────────────────
const ShootingStar: React.FC<{ delay: number; color: string }> = ({ delay, color }) => (
  <motion.div
    initial={{ x: -50, y: -30, opacity: 0 }}
    animate={{ x: [0, 350], y: [0, 180], opacity: [0, 1, 0] }}
    transition={{ duration: 0.9, delay, repeat: Infinity, repeatDelay: 5 + Math.random() * 10, ease: 'easeOut' }}
    style={{
      position: 'absolute',
      width: 60,
      height: 1.5,
      borderRadius: 4,
      background: `linear-gradient(90deg, ${color}, transparent)`,
      transform: 'rotate(25deg)',
      filter: `drop-shadow(0 0 4px ${color})`,
      pointerEvents: 'none',
      top: `${15 + Math.random() * 40}%`,
      left: `${5 + Math.random() * 30}%`,
    }}
  />
);

// ═══════════════════════════════════════════════════════════════════════════
// MAIN BOUNTY MODAL
// ═══════════════════════════════════════════════════════════════════════════
export default function BountyModal({ onClose, genieTargetRef }: BountyModalProps) {
  const [step, setStep] = useState(0);
  const totalSteps = 4;
  const [isClosing, setIsClosing] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const [genieAnim, setGenieAnim] = useState<{
    x: number; y: number; scale: number; opacity: number;
  } | null>(null);

  const handleClose = useCallback(() => {
    localStorage.setItem(STORAGE_KEY, 'true');

    // If we have a genie target, animate into it
    if (genieTargetRef?.current && cardRef.current) {
      const targetRect = genieTargetRef.current.getBoundingClientRect();
      const cardRect = cardRef.current.getBoundingClientRect();

      // Calculate delta to move card center → target center
      const targetCenterX = targetRect.left + targetRect.width / 2;
      const targetCenterY = targetRect.top + targetRect.height / 2;
      const cardCenterX = cardRect.left + cardRect.width / 2;
      const cardCenterY = cardRect.top + cardRect.height / 2;

      setGenieAnim({
        x: targetCenterX - cardCenterX,
        y: targetCenterY - cardCenterY,
        scale: targetRect.width / cardRect.width,
        opacity: 0,
      });
      setIsClosing(true);

      // Wait for animation to complete, then unmount
      setTimeout(() => {
        onClose();
      }, 500);
    } else {
      onClose();
    }
  }, [onClose, genieTargetRef]);

  const handleNext = () => setStep(s => Math.min(s + 1, totalSteps - 1));
  const handlePrev = () => setStep(s => Math.max(s - 1, 0));

  const stepLabels = ['Rewards', 'Categories', 'Multipliers', 'Join Now'];

  // ═══ STEP CONTENT ═══════════════════════════════════════════════════════
  const renderStep = () => {
    switch (step) {
      // ─── Step 0: Hook — Big Numbers ──────────────────────────────────
      case 0:
        return (
          <motion.div
            key="s0"
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -40 }}
            transition={{ duration: 0.35 }}
            className="text-center"
          >
            {/* Trophy icon */}
            <motion.div
              className="relative mx-auto mb-5"
              style={{ width: 80, height: 80 }}
              animate={{ rotate: 360 }}
              transition={{ duration: 40, repeat: Infinity, ease: 'linear' }}
            >
              {/* Glow rings */}
              <PulseRing delay={0} color="#10B981" size={80} />
              <PulseRing delay={0.8} color="#F43F5E" size={80} />
              <PulseRing delay={1.6} color="#8B5CF6" size={80} />
              <div className="absolute inset-0 flex items-center justify-center">
                <motion.div
                  animate={{ scale: [1, 1.12, 1] }}
                  transition={{ duration: 2.5, repeat: Infinity }}
                >
                  <Trophy className="w-10 h-10" style={{
                    color: '#10B981',
                    filter: 'drop-shadow(0 0 20px rgba(16,185,129,0.7)) drop-shadow(0 0 40px rgba(16,185,129,0.3))',
                  }} />
                </motion.div>
              </div>
            </motion.div>

            <h2 className="text-2xl font-bold mb-2" style={{
              background: 'linear-gradient(135deg, #10B981, #22D3EE, #A78BFA)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              Bounty Campaign is Live
            </h2>
            <p className="text-sm text-gray-400 mb-6">
              Earn rewards for running nodes, finding bugs, and growing the community
            </p>

            {/* Big reward stats */}
            <div className="grid grid-cols-3 gap-3 mb-6">
              {[
                { label: 'Daily Pool', value: 306, suffix: ' USD', color: '#10B981', icon: <Gift className="w-4 h-4" /> },
                { label: 'Categories', value: 5, suffix: '', color: '#F43F5E', icon: <Target className="w-4 h-4" /> },
                { label: 'Max Bonus', value: 2.0, suffix: 'x', color: '#8B5CF6', icon: <Flame className="w-4 h-4" />, decimals: 1 },
              ].map((stat, i) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + i * 0.15 }}
                  className="p-3 rounded-xl relative overflow-hidden"
                  style={{
                    background: `linear-gradient(160deg, ${stat.color}12 0%, ${stat.color}06 100%)`,
                    border: `1px solid ${stat.color}30`,
                  }}
                >
                  <div className="flex items-center justify-center gap-1.5 mb-1" style={{ color: stat.color }}>
                    {stat.icon}
                    <span className="text-xs font-medium">{stat.label}</span>
                  </div>
                  <p className="text-xl font-black" style={{
                    background: `linear-gradient(135deg, ${stat.color}, ${stat.color}CC)`,
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}>
                    <AnimCounter target={stat.value} prefix={stat.label === 'Total Pool' ? '' : ''} suffix={stat.suffix} decimals={(stat as any).decimals || 0} />
                  </p>
                </motion.div>
              ))}
            </div>

            {/* OAuth2 badge */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
              className="flex items-center justify-center gap-2 px-4 py-2 rounded-full mx-auto"
              style={{
                background: 'linear-gradient(135deg, rgba(16,185,129,0.1), rgba(139,92,246,0.1))',
                border: '1px solid rgba(16,185,129,0.25)',
                width: 'fit-content',
              }}
            >
              <Shield className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-xs text-emerald-300/80">OAuth2 Wallet Connect Available</span>
            </motion.div>
          </motion.div>
        );

      // ─── Step 1: Categories ──────────────────────────────────────────
      case 1:
        return (
          <motion.div
            key="s1"
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -40 }}
            transition={{ duration: 0.35 }}
          >
            <h2 className="text-lg font-bold text-center mb-1" style={{
              background: 'linear-gradient(135deg, #14B8A6, #F43F5E)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              5 Ways to Earn
            </h2>
            <p className="text-xs text-gray-500 text-center mb-4">Stack points across every category</p>

            <div className="space-y-2.5">
              {[
                { icon: <Pickaxe className="w-5 h-5" />, label: 'Node Operations', desc: 'Run a node, maintain uptime, sync blocks', pts: 'Up to 500 pts/day', color: '#10B981', bg: '#10B98112' },
                { icon: <Zap className="w-5 h-5" />, label: 'Transactions', desc: 'Send, receive, swap tokens on the DEX', pts: 'Up to 200 pts/day', color: '#06B6D4', bg: '#06B6D412' },
                { icon: <Bug className="w-5 h-5" />, label: 'Bug Reports', desc: 'Find bugs — Critical: 100 pts, High: 50 pts', pts: '10-100 pts each', color: '#F43F5E', bg: '#F43F5E12' },
                { icon: <Users className="w-5 h-5" />, label: 'Community', desc: 'Help others on Discord, answer questions', pts: 'Up to 150 pts/day', color: '#8B5CF6', bg: '#8B5CF612' },
                { icon: <Globe className="w-5 h-5" />, label: 'Social & Content', desc: 'Tweets, threads, articles, videos, PRs', pts: '10-200 pts each', color: '#F97316', bg: '#F9731612' },
              ].map((cat, i) => (
                <motion.div
                  key={cat.label}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 + i * 0.08 }}
                  className="flex items-center gap-3 p-3 rounded-xl"
                  style={{ background: cat.bg, border: `1px solid ${cat.color}20` }}
                >
                  <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ background: `${cat.color}18`, color: cat.color }}>
                    {cat.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-white">{cat.label}</p>
                    <p className="text-xs text-gray-400 truncate">{cat.desc}</p>
                  </div>
                  <div className="text-xs font-medium px-2 py-1 rounded-full flex-shrink-0"
                    style={{ background: `${cat.color}15`, color: cat.color, border: `1px solid ${cat.color}25` }}>
                    {cat.pts}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        );

      // ─── Step 2: Multipliers & Tiers ─────────────────────────────────
      case 2:
        return (
          <motion.div
            key="s2"
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -40 }}
            transition={{ duration: 0.35 }}
          >
            <h2 className="text-lg font-bold text-center mb-1" style={{
              background: 'linear-gradient(135deg, #F97316, #EC4899)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              Bonus Multipliers
            </h2>
            <p className="text-xs text-gray-500 text-center mb-4">Stack multipliers to maximize your rewards</p>

            <div className="grid grid-cols-2 gap-3 mb-5">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
                className="p-4 rounded-xl text-center relative overflow-hidden"
                style={{
                  background: 'linear-gradient(160deg, rgba(249,115,22,0.1), rgba(249,115,22,0.03))',
                  border: '1px solid rgba(249,115,22,0.25)',
                }}
              >
                <Flame className="w-6 h-6 mx-auto mb-2 text-orange-400" />
                <p className="text-2xl font-black text-orange-400">2.0x</p>
                <p className="text-xs text-orange-300/70 font-medium">Early Bird</p>
                <p className="text-[10px] text-gray-500 mt-1">Join in the first week</p>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 }}
                className="p-4 rounded-xl text-center relative overflow-hidden"
                style={{
                  background: 'linear-gradient(160deg, rgba(236,72,153,0.1), rgba(236,72,153,0.03))',
                  border: '1px solid rgba(236,72,153,0.25)',
                }}
              >
                <TrendingUp className="w-6 h-6 mx-auto mb-2 text-pink-400" />
                <p className="text-2xl font-black text-pink-400">1.2x</p>
                <p className="text-xs text-pink-300/70 font-medium">Consistency</p>
                <p className="text-[10px] text-gray-500 mt-1">Daily activity streak</p>
              </motion.div>
            </div>

            {/* Tier progression */}
            <p className="text-xs text-gray-500 text-center mb-3 font-medium">Tier Progression</p>
            <div className="flex items-center justify-between gap-1 px-2">
              {[
                { name: 'Bronze', color: '#CD7F32', pts: '0+' },
                { name: 'Silver', color: '#C0C0C0', pts: '500+' },
                { name: 'Gold', color: '#FFD700', pts: '2,000+' },
                { name: 'Diamond', color: '#B9F2FF', pts: '5,000+' },
              ].map((tier, i) => (
                <motion.div
                  key={tier.name}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 + i * 0.1 }}
                  className="flex-1 text-center"
                >
                  <motion.div
                    animate={{ rotateY: [0, 180, 360] }}
                    transition={{ duration: 3, delay: i * 0.5, repeat: Infinity, repeatDelay: 8 }}
                    className="w-10 h-10 mx-auto rounded-lg flex items-center justify-center mb-1"
                    style={{
                      background: `linear-gradient(135deg, ${tier.color}25, ${tier.color}10)`,
                      border: `1.5px solid ${tier.color}40`,
                      boxShadow: `0 0 12px ${tier.color}15`,
                    }}
                  >
                    <Award className="w-5 h-5" style={{ color: tier.color }} />
                  </motion.div>
                  <p className="text-[11px] font-bold" style={{ color: tier.color }}>{tier.name}</p>
                  <p className="text-[9px] text-gray-500">{tier.pts}</p>
                  {i < 3 && (
                    <div className="absolute" style={{ display: 'none' }} /> // spacer
                  )}
                </motion.div>
              ))}
            </div>

            {/* Combined multiplier */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
              className="mt-4 p-3 rounded-xl text-center"
              style={{
                background: 'linear-gradient(135deg, rgba(16,185,129,0.08), rgba(139,92,246,0.08))',
                border: '1px dashed rgba(16,185,129,0.3)',
              }}
            >
              <p className="text-xs text-gray-400">Combined max multiplier</p>
              <p className="text-3xl font-black mt-1" style={{
                background: 'linear-gradient(135deg, #10B981, #8B5CF6, #F43F5E)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>
                <AnimCounter target={2.4} suffix="x" decimals={1} duration={1500} /> {/* 2.0x early + 1.2x consistency */}
              </p>
            </motion.div>
          </motion.div>
        );

      // ─── Step 3: CTA — Join Now ──────────────────────────────────────
      case 3:
        return (
          <motion.div
            key="s3"
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -40 }}
            transition={{ duration: 0.35 }}
            className="text-center"
          >
            <motion.div
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, #10B981, #06B6D4, #8B5CF6)',
                boxShadow: '0 0 30px rgba(16,185,129,0.3), 0 0 60px rgba(16,185,129,0.1)',
              }}
            >
              <Star className="w-8 h-8 text-white" />
            </motion.div>

            <h2 className="text-xl font-bold mb-2" style={{
              background: 'linear-gradient(135deg, #10B981, #22D3EE)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              Ready to Start Earning?
            </h2>
            <p className="text-sm text-gray-400 mb-6">
              Register with your wallet address or connect via OAuth2
            </p>

            {/* Platform badges */}
            <div className="flex items-center justify-center gap-3 mb-6">
              {[
                { icon: <Twitter className="w-4 h-4" />, label: 'Twitter', color: '#1DA1F2' },
                { icon: <Github className="w-4 h-4" />, label: 'GitHub', color: '#8B5CF6' },
                { icon: <MessageCircle className="w-4 h-4" />, label: 'Discord', color: '#5865F2' },
              ].map((p) => (
                <div
                  key={p.label}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs"
                  style={{ background: `${p.color}15`, border: `1px solid ${p.color}30`, color: p.color }}
                >
                  {p.icon}
                  <span>{p.label}</span>
                </div>
              ))}
            </div>

            {/* CTA Button */}
            <motion.a
              href={BOUNTY_URL}
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.04, y: -2 }}
              whileTap={{ scale: 0.97 }}
              className="inline-flex items-center gap-2 px-8 py-3.5 rounded-2xl font-bold text-base transition-all"
              style={{
                background: 'linear-gradient(135deg, #10B981, #06B6D4, #8B5CF6)',
                backgroundSize: '200% 200%',
                color: '#fff',
                boxShadow: '0 4px 24px rgba(16,185,129,0.35), 0 0 60px rgba(16,185,129,0.15)',
                animation: 'bountyGradientShift 4s ease infinite',
              }}
            >
              <Trophy className="w-5 h-5" />
              Join Bounty Campaign
              <ExternalLink className="w-4 h-4" />
            </motion.a>

            <p className="text-[11px] text-gray-500 mt-4 flex items-center justify-center gap-1">
              <Shield className="w-3 h-3 text-emerald-500" />
              Secure OAuth2 wallet connection — no private keys shared
            </p>
          </motion.div>
        );

      default:
        return null;
    }
  };

  // ═══ RENDER ═════════════════════════════════════════════════════════════
  const modal = (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: isClosing ? 0 : 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: isClosing ? 0.4 : 0.2 }}
        onClick={handleClose}
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 99999,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'radial-gradient(ellipse at 25% 15%, rgba(16,185,129,0.06) 0%, transparent 60%), radial-gradient(ellipse at 75% 85%, rgba(139,92,246,0.05) 0%, transparent 60%), radial-gradient(ellipse at center, rgba(0,0,0,0.92) 0%, rgba(0,0,0,0.97) 100%)',
          backdropFilter: 'blur(24px)',
        }}
      >
        {/* Canvas particles behind everything */}
        <BountyParticleCanvas />

        {/* Shooting stars */}
        <ShootingStar delay={1} color="#10B981" />
        <ShootingStar delay={5} color="#F43F5E" />
        <ShootingStar delay={9} color="#8B5CF6" />
        <ShootingStar delay={13} color="#06B6D4" />

        {/* Modal card — genie animation on close */}
        <motion.div
          ref={cardRef}
          initial={{ scale: 0.85, opacity: 0, y: 30 }}
          animate={genieAnim ? {
            x: genieAnim.x,
            y: genieAnim.y,
            scale: genieAnim.scale,
            opacity: 0,
            borderRadius: 100,
          } : { scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.85, opacity: 0, y: 30 }}
          transition={genieAnim ? {
            duration: 0.45,
            ease: [0.4, 0, 0.2, 1],
          } : { type: 'spring', damping: 22, stiffness: 280 }}
          onClick={e => e.stopPropagation()}
          style={{
            position: 'relative',
            width: '100%',
            maxWidth: 520,
            margin: '0 16px',
            borderRadius: 28,
            overflow: 'hidden',
            zIndex: 2,
          }}
        >
          {/* Animated border — rotating conic gradient */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
            style={{
              position: 'absolute',
              inset: -2,
              borderRadius: 30,
              background: 'conic-gradient(from 0deg, #10B981, #06B6D4, #8B5CF6, #F43F5E, #F97316, #EC4899, #10B981)',
              opacity: 0.5,
              zIndex: 0,
            }}
          />

          {/* Inner container */}
          <div
            style={{
              position: 'relative',
              zIndex: 1,
              borderRadius: 28,
              background: 'linear-gradient(160deg, rgba(8,12,20,0.98) 0%, rgba(14,20,32,0.97) 50%, rgba(8,12,20,0.98) 100%)',
              padding: '28px 24px 20px',
            }}
          >
            {/* Nebula particles inside the card */}
            <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', borderRadius: 28, pointerEvents: 'none' }}>
              <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
                {Array.from({ length: 50 }, (_, i) => (
                  <HexParticle key={i} index={i} total={50} />
                ))}
              </div>
            </div>

            {/* Close button — z-50 to stay above step content (z-10) */}
            <button
              onClick={handleClose}
              className="absolute top-4 right-4 p-1.5 rounded-full transition-all hover:scale-110 z-50"
              style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)' }}
            >
              <X className="w-4 h-4 text-gray-400" />
            </button>

            {/* Step content */}
            <div className="relative z-10" style={{ minHeight: 320 }}>
              <AnimatePresence mode="wait">
                {renderStep()}
              </AnimatePresence>
            </div>

            {/* Step indicators + navigation */}
            <div className="relative z-10 flex items-center justify-between mt-4 pt-4" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
              {/* Back button */}
              <button
                onClick={handlePrev}
                disabled={step === 0}
                className="flex items-center gap-1 px-3 py-2 rounded-xl text-sm transition-all disabled:opacity-20 hover:bg-white/5"
                style={{ border: '1px solid rgba(255,255,255,0.08)' }}
              >
                <ChevronLeft className="w-4 h-4 text-gray-400" />
                <span className="text-gray-400">Back</span>
              </button>

              {/* Dots */}
              <div className="flex items-center gap-2">
                {stepLabels.map((label, i) => (
                  <button
                    key={i}
                    onClick={() => setStep(i)}
                    className="transition-all"
                    title={label}
                  >
                    <motion.div
                      animate={{
                        width: i === step ? 28 : 8,
                        background: i === step
                          ? 'linear-gradient(90deg, #10B981, #8B5CF6)'
                          : i < step
                          ? '#10B981'
                          : 'rgba(255,255,255,0.15)',
                      }}
                      style={{
                        height: 8,
                        borderRadius: 4,
                        boxShadow: i === step ? '0 0 10px rgba(16,185,129,0.5)' : 'none',
                      }}
                    />
                  </button>
                ))}
              </div>

              {/* Next / Join button */}
              {step < totalSteps - 1 ? (
                <button
                  onClick={handleNext}
                  className="flex items-center gap-1 px-4 py-2 rounded-xl text-sm font-medium transition-all hover:scale-105"
                  style={{
                    background: 'linear-gradient(135deg, #10B981, #06B6D4)',
                    color: '#fff',
                    boxShadow: '0 2px 12px rgba(16,185,129,0.3)',
                  }}
                >
                  <span>Next</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              ) : (
                <motion.a
                  href={BOUNTY_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="flex items-center gap-1 px-4 py-2 rounded-xl text-sm font-bold transition-all"
                  style={{
                    background: 'linear-gradient(135deg, #10B981, #8B5CF6)',
                    color: '#fff',
                    boxShadow: '0 2px 16px rgba(16,185,129,0.4)',
                  }}
                  onClick={() => { localStorage.setItem(STORAGE_KEY, 'true'); }}
                >
                  <Trophy className="w-4 h-4" />
                  <span>Join</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </motion.a>
              )}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );

  return createPortal(modal, document.body);
}
