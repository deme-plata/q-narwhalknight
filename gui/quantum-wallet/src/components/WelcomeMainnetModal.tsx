import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Shield, Rocket, X, ChevronRight, ChevronLeft, Globe, Zap, Lock, Cpu, Coins, BarChart3, Users, Pickaxe } from 'lucide-react';
import { qnkAPI } from '../services/api';

const STORAGE_KEY = 'mainnet-genesis_welcomed_v2';

interface WelcomeMainnetModalProps {
  onClose: () => void;
}

// ─── Nebula Particle Field ──────────────────────────────────────────────────
// Simulates a quantum nebula with orbiting, pulsing particles
const NebulaParticle: React.FC<{ index: number; total: number }> = ({ index, total }) => {
  const seed = useMemo(() => ({
    angle: (index / total) * Math.PI * 2 + (Math.random() - 0.5) * 0.8,
    radius: 20 + Math.random() * 80,
    orbitSpeed: 8 + Math.random() * 16,
    pulseSpeed: 2 + Math.random() * 4,
    size: 1.5 + Math.random() * 4,
    opacity: 0.3 + Math.random() * 0.7,
    delay: Math.random() * 3,
    color: ['#FFD700', '#00E5FF', '#7C4DFF', '#FF6B35', '#00E676', '#FF4081', '#E040FB', '#448AFF'][index % 8],
    drift: (Math.random() - 0.5) * 30,
  }), [index, total]);

  return (
    <motion.div
      initial={{
        x: Math.cos(seed.angle) * seed.radius * 0.2,
        y: Math.sin(seed.angle) * seed.radius * 0.2,
        opacity: 0,
        scale: 0,
      }}
      animate={{
        x: [
          Math.cos(seed.angle) * seed.radius * 0.2,
          Math.cos(seed.angle + 0.8) * seed.radius + seed.drift,
          Math.cos(seed.angle + 1.6) * seed.radius * 0.6,
          Math.cos(seed.angle + 2.4) * seed.radius * 0.9,
          Math.cos(seed.angle + 3.2) * seed.radius * 0.3,
        ],
        y: [
          Math.sin(seed.angle) * seed.radius * 0.2,
          Math.sin(seed.angle + 0.8) * seed.radius + seed.drift,
          Math.sin(seed.angle + 1.6) * seed.radius * 0.6,
          Math.sin(seed.angle + 2.4) * seed.radius * 0.9,
          Math.sin(seed.angle + 3.2) * seed.radius * 0.3,
        ],
        opacity: [0, seed.opacity, seed.opacity * 0.7, seed.opacity, 0],
        scale: [0, 1.2, 0.8, 1, 0],
      }}
      transition={{
        duration: seed.orbitSpeed,
        delay: seed.delay,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
      style={{
        position: 'absolute',
        width: seed.size,
        height: seed.size,
        borderRadius: '50%',
        background: `radial-gradient(circle, ${seed.color}, ${seed.color}00)`,
        boxShadow: `0 0 ${seed.size * 4}px ${seed.color}60, 0 0 ${seed.size * 8}px ${seed.color}20`,
        pointerEvents: 'none',
        filter: 'blur(0.3px)',
      }}
    />
  );
};

// ─── Pulse Ring ─────────────────────────────────────────────────────────────
const PulseRing: React.FC<{ size: number; delay: number; color: string; thickness?: number }> = ({ size, delay, color, thickness = 1 }) => (
  <motion.div
    initial={{ scale: 0.3, opacity: 0 }}
    animate={{
      scale: [0.3, 1.3, 0.3],
      opacity: [0, 0.4, 0],
    }}
    transition={{
      duration: 5,
      delay,
      repeat: Infinity,
      ease: 'easeInOut',
    }}
    style={{
      position: 'absolute',
      width: size,
      height: size,
      borderRadius: '50%',
      border: `${thickness}px solid ${color}`,
      boxShadow: `0 0 30px ${color}20, inset 0 0 20px ${color}08`,
      pointerEvents: 'none',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
    }}
  />
);

// ─── Shooting Star ──────────────────────────────────────────────────────────
const ShootingStar: React.FC<{ delay: number }> = ({ delay }) => {
  const startX = useMemo(() => -50 + Math.random() * 200, []);
  const startY = useMemo(() => -20 + Math.random() * 40, []);
  const color = useMemo(() => ['#FFD700', '#00E5FF', '#FF6B35'][Math.floor(Math.random() * 3)], []);

  return (
    <motion.div
      initial={{ x: startX, y: startY, opacity: 0 }}
      animate={{
        x: startX + 300,
        y: startY + 150,
        opacity: [0, 1, 0],
      }}
      transition={{
        duration: 0.8,
        delay,
        repeat: Infinity,
        repeatDelay: 6 + Math.random() * 10,
        ease: 'easeOut',
      }}
      style={{
        position: 'absolute',
        width: 40,
        height: 1.5,
        background: `linear-gradient(90deg, ${color}, transparent)`,
        borderRadius: 1,
        boxShadow: `0 0 8px ${color}80`,
        pointerEvents: 'none',
        transformOrigin: 'left center',
        transform: 'rotate(30deg)',
      }}
    />
  );
};

// ─── Typewriter Text ────────────────────────────────────────────────────────
const TypewriterText: React.FC<{ text: string; delay: number; speed?: number }> = ({ text, delay, speed = 25 }) => {
  const [displayText, setDisplayText] = useState('');

  useEffect(() => {
    const timer = setTimeout(() => {
      let i = 0;
      const interval = setInterval(() => {
        if (i <= text.length) {
          setDisplayText(text.slice(0, i));
          i++;
        } else {
          clearInterval(interval);
        }
      }, speed);
      return () => clearInterval(interval);
    }, delay);
    return () => clearTimeout(timer);
  }, [text, delay, speed]);

  return (
    <span>
      {displayText}
      <motion.span
        animate={{ opacity: [1, 0] }}
        transition={{ duration: 0.6, repeat: Infinity }}
        style={{ opacity: displayText.length < text.length ? 1 : 0 }}
      >
        |
      </motion.span>
    </span>
  );
};

// ─── Counter Animation ──────────────────────────────────────────────────────
const AnimatedCounter: React.FC<{ target: number; duration?: number; prefix?: string; suffix?: string; decimals?: number }> = ({
  target, duration = 2000, prefix = '', suffix = '', decimals = 0,
}) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const step = () => {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(target * eased);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration]);

  const formatted = decimals > 0
    ? count.toFixed(decimals)
    : Math.round(count).toLocaleString();

  return <span>{prefix}{formatted}{suffix}</span>;
};

// ─── Steps ──────────────────────────────────────────────────────────────────
const steps = [
  { id: 'welcome', title: 'Welcome' },
  { id: 'network', title: 'Network' },
  { id: 'features', title: 'Features' },
  { id: 'start', title: 'Get Started' },
];

// ─── Main Component ─────────────────────────────────────────────────────────
const WelcomeMainnetModal: React.FC<WelcomeMainnetModalProps> = ({ onClose }) => {
  const [step, setStep] = useState(0);
  const [networkStats, setNetworkStats] = useState<{
    height: number;
    peers: number;
    version: string;
    tps: number;
  }>({ height: 0, peers: 0, version: '', tps: 0 });

  // Mark as seen immediately on mount AND on close (belt-and-suspenders)
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, 'true');
    localStorage.setItem('mainnetWelcomeSeen', 'true');
  }, []);

  // Fetch live network stats for the Network step
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const resp = await qnkAPI.getNodeStatus();
        const status = resp?.data;
        if (status) {
          setNetworkStats({
            height: status.current_height || 0,
            peers: status.connected_peers || 0,
            version: 'v7.3.0',
            tps: status.tps_current || 0,
          });
        }
      } catch {
        // Silently fail — stats are nice-to-have
      }
    };
    fetchStats();
  }, []);

  const handleClose = useCallback(() => {
    localStorage.setItem(STORAGE_KEY, 'true');
    localStorage.setItem('mainnetWelcomeSeen', 'true');
    onClose();
  }, [onClose]);

  const nextStep = () => setStep((s) => Math.min(s + 1, steps.length - 1));
  const prevStep = () => setStep((s) => Math.max(s - 1, 0));

  // Background confetti — persistent across steps
  const confetti = useMemo(() => Array.from({ length: 35 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    delay: Math.random() * 4,
    duration: 4 + Math.random() * 5,
    size: 2 + Math.random() * 5,
    color: ['#FFD700', '#00E5FF', '#FF6B35', '#7C4DFF', '#00E676', '#FF4081', '#E040FB'][i % 7],
    shape: Math.random() > 0.6 ? '50%' : Math.random() > 0.3 ? '2px' : '30%',
    wobble: (Math.random() - 0.5) * 60,
  })), []);

  const renderStep = () => {
    switch (step) {
      case 0: // ─── Welcome ────────────────────────────────────────────────
        return (
          <motion.div
            key="welcome"
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -60 }}
            transition={{ duration: 0.4 }}
            style={{ textAlign: 'center' }}
          >
            {/* Quantum nebula hero */}
            <div style={{
              position: 'relative',
              width: 200,
              height: 200,
              margin: '0 auto 16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              {/* Glow rings */}
              <PulseRing size={200} delay={0} color="#FFD700" thickness={1} />
              <PulseRing size={150} delay={0.7} color="#00E5FF" />
              <PulseRing size={100} delay={1.4} color="#7C4DFF" />
              <PulseRing size={60} delay={2.1} color="#FF6B35" />

              {/* Nebula particles */}
              {Array.from({ length: 60 }, (_, i) => (
                <NebulaParticle key={i} index={i} total={60} />
              ))}

              {/* Shooting stars */}
              <ShootingStar delay={1} />
              <ShootingStar delay={5} />
              <ShootingStar delay={9} />

              {/* Central icon */}
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 30, repeat: Infinity, ease: 'linear' }}
                style={{
                  position: 'absolute',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  zIndex: 5,
                }}
              >
                <motion.div
                  animate={{ scale: [1, 1.15, 1] }}
                  transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                >
                  <Sparkles size={52} color="#FFD700" style={{ filter: 'drop-shadow(0 0 30px rgba(255,215,0,0.8)) drop-shadow(0 0 60px rgba(255,215,0,0.3))' }} />
                </motion.div>
              </motion.div>

              {/* Core glow */}
              <div style={{
                position: 'absolute',
                width: 40,
                height: 40,
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(255,215,0,0.4) 0%, transparent 70%)',
                filter: 'blur(10px)',
                zIndex: 3,
              }} />
            </div>

            <motion.h1
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3, type: 'spring', stiffness: 200 }}
              style={{
                fontSize: 34,
                fontWeight: 800,
                background: 'linear-gradient(135deg, #FFD700 0%, #FF6B35 40%, #00E5FF 80%, #7C4DFF 100%)',
                backgroundSize: '200% 200%',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: 10,
                letterSpacing: '-0.5px',
                lineHeight: 1.2,
              }}
            >
              Mainnet 2026.1 is Live
            </motion.h1>

            <motion.p
              initial={{ y: 15, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.5 }}
              style={{ color: '#9CA3AF', fontSize: 14, lineHeight: 1.7, maxWidth: 400, margin: '0 auto' }}
            >
              The world's first post-quantum blockchain is running.
              Mine QUG, trade on the DEX, and deploy smart contracts
              — secured by lattice-based cryptography.
            </motion.p>
          </motion.div>
        );

      case 1: // ─── Network Stats (Live from API) ──────────────────────────
        return (
          <motion.div
            key="network"
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -60 }}
            transition={{ duration: 0.4 }}
          >
            <div style={{ textAlign: 'center', marginBottom: 18 }}>
              <motion.div
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
              >
                <Globe size={38} color="#00E5FF" style={{ margin: '0 auto 10px', filter: 'drop-shadow(0 0 20px rgba(0,229,255,0.6))' }} />
              </motion.div>
              <h2 style={{ fontSize: 23, fontWeight: 700, color: '#E5E7EB', marginBottom: 4 }}>Live Network</h2>
              <p style={{ color: '#6B7280', fontSize: 12 }}>Real-time stats from mainnet</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 16 }}>
              {[
                {
                  label: 'Block Height',
                  value: networkStats.height > 0 ? <AnimatedCounter target={networkStats.height} duration={1500} /> : <TypewriterText text="syncing..." delay={200} />,
                  icon: <BarChart3 size={18} color="#FFD700" />,
                  color: '#FFD700',
                },
                {
                  label: 'Connected Peers',
                  value: networkStats.peers > 0 ? <AnimatedCounter target={networkStats.peers} duration={1000} /> : <TypewriterText text="discovering..." delay={400} />,
                  icon: <Users size={18} color="#00E5FF" />,
                  color: '#00E5FF',
                },
                {
                  label: 'Max Supply',
                  value: <TypewriterText text="21,000,000 QUG" delay={600} speed={20} />,
                  icon: <Coins size={18} color="#00E676" />,
                  color: '#00E676',
                },
                {
                  label: 'Halving Eras',
                  value: <TypewriterText text="64 eras / 256 yrs" delay={800} speed={20} />,
                  icon: <BarChart3 size={18} color="#7C4DFF" />,
                  color: '#7C4DFF',
                },
              ].map((stat, i) => (
                <motion.div
                  key={stat.label}
                  initial={{ scale: 0.7, opacity: 0, y: 15 }}
                  animate={{ scale: 1, opacity: 1, y: 0 }}
                  transition={{ delay: 0.15 + i * 0.1, type: 'spring', stiffness: 300, damping: 20 }}
                  style={{
                    padding: '14px 12px',
                    borderRadius: 14,
                    background: `linear-gradient(145deg, ${stat.color}10, ${stat.color}04)`,
                    border: `1px solid ${stat.color}25`,
                    textAlign: 'center',
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  {/* Subtle shimmer */}
                  <motion.div
                    animate={{ x: [-100, 200] }}
                    transition={{ duration: 3, delay: i * 0.5, repeat: Infinity, repeatDelay: 5 }}
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: 60,
                      height: '100%',
                      background: `linear-gradient(90deg, transparent, ${stat.color}08, transparent)`,
                      transform: 'skewX(-15deg)',
                    }}
                  />
                  <div style={{ marginBottom: 6, position: 'relative' }}>{stat.icon}</div>
                  <div style={{ color: '#E5E7EB', fontSize: 14, fontWeight: 700, position: 'relative', minHeight: 20 }}>
                    {stat.value}
                  </div>
                  <div style={{ color: '#6B7280', fontSize: 11, marginTop: 3, position: 'relative' }}>{stat.label}</div>
                </motion.div>
              ))}
            </div>

            {/* Live status row */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                padding: '8px 16px',
                borderRadius: 20,
                background: 'rgba(0, 230, 118, 0.08)',
                border: '1px solid rgba(0, 230, 118, 0.2)',
              }}
            >
              <motion.div
                animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
                style={{ width: 6, height: 6, borderRadius: '50%', background: '#00E676' }}
              />
              <span style={{ color: '#00E676', fontSize: 12, fontWeight: 600 }}>Mainnet Active</span>
              <span style={{ color: '#4B5563', fontSize: 11 }}>|</span>
              <span style={{ color: '#6B7280', fontSize: 11 }}>Dilithium5 + Kyber1024</span>
            </motion.div>
          </motion.div>
        );

      case 2: // ─── Features ────────────────────────────────────────────────
        return (
          <motion.div
            key="features"
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -60 }}
            transition={{ duration: 0.4 }}
          >
            <div style={{ textAlign: 'center', marginBottom: 14 }}>
              <Cpu size={32} color="#7C4DFF" style={{ margin: '0 auto 8px', filter: 'drop-shadow(0 0 16px rgba(124,77,255,0.6))' }} />
              <h2 style={{ fontSize: 22, fontWeight: 700, color: '#E5E7EB', marginBottom: 2 }}>Built-in Features</h2>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {[
                { icon: <Shield size={17} color="#00E676" />, title: 'Post-Quantum Security', desc: 'Dilithium5 signatures & Kyber1024 key exchange.', color: '#00E676' },
                { icon: <Zap size={17} color="#FFD700" />, title: 'DAG-Knight Consensus', desc: 'Zero-message BFT with sub-second finality.', color: '#FFD700' },
                { icon: <Globe size={17} color="#00E5FF" />, title: 'Full DeFi Stack', desc: 'DEX, tokens, staking, lending & privacy mixing.', color: '#00E5FF' },
                { icon: <Lock size={17} color="#7C4DFF" />, title: 'Privacy Transactions', desc: 'Ring signatures & bulletproofs for confidentiality.', color: '#7C4DFF' },
                { icon: <Coins size={17} color="#FF6B35" />, title: 'Bitcoin-Style Emission', desc: '21M supply cap, 64 halving eras, 256 years.', color: '#FF6B35' },
                { icon: <Pickaxe size={17} color="#E040FB" />, title: 'CPU-Friendly Mining', desc: 'Ring-LWE VRF — mine on any computer.', color: '#E040FB' },
              ].map((feat, i) => (
                <motion.div
                  key={feat.title}
                  initial={{ x: -25, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 0.1 + i * 0.07, type: 'spring', stiffness: 250 }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 11,
                    padding: '9px 13px',
                    borderRadius: 10,
                    background: `linear-gradient(135deg, ${feat.color}0D, ${feat.color}04)`,
                    border: `1px solid ${feat.color}20`,
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  <div style={{
                    width: 30, height: 30, borderRadius: 8,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: `${feat.color}15`, flexShrink: 0,
                  }}>
                    {feat.icon}
                  </div>
                  <div>
                    <div style={{ color: '#E5E7EB', fontSize: 12.5, fontWeight: 700 }}>{feat.title}</div>
                    <div style={{ color: '#6B7280', fontSize: 10.5, lineHeight: 1.4 }}>{feat.desc}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        );

      case 3: // ─── Get Started ─────────────────────────────────────────────
        return (
          <motion.div
            key="start"
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -60 }}
            transition={{ duration: 0.4 }}
            style={{ textAlign: 'center' }}
          >
            {/* Animated rocket with particle trail */}
            <div style={{ position: 'relative', width: 120, height: 120, margin: '0 auto 16px' }}>
              <PulseRing size={120} delay={0} color="#FF6B35" />
              <PulseRing size={80} delay={0.5} color="#FFD700" />
              {Array.from({ length: 20 }, (_, i) => (
                <NebulaParticle key={i} index={i} total={20} />
              ))}
              <motion.div
                animate={{
                  y: [-5, 5, -5],
                  rotate: [0, 3, -3, 0],
                }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  zIndex: 5,
                }}
              >
                <Rocket size={44} color="#FF6B35" style={{ filter: 'drop-shadow(0 0 24px rgba(255,107,53,0.7))' }} />
              </motion.div>
            </div>

            <h2 style={{ fontSize: 25, fontWeight: 700, color: '#E5E7EB', marginBottom: 6 }}>Ready to Begin</h2>
            <p style={{ color: '#9CA3AF', fontSize: 13, lineHeight: 1.6, maxWidth: 360, margin: '0 auto 18px' }}>
              Your node is live on Q-NarwhalKnight mainnet. Start earning, trading, and building.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 6 }}>
              {[
                { label: 'Start mining from the Mining tab', color: '#FFD700', icon: <Pickaxe size={14} color="#FFD700" /> },
                { label: 'Trade tokens on the built-in DEX', color: '#00E5FF', icon: <BarChart3 size={14} color="#00E5FF" /> },
                { label: 'Deploy custom tokens via VittuaVM', color: '#7C4DFF', icon: <Cpu size={14} color="#7C4DFF" /> },
              ].map((tip, i) => (
                <motion.div
                  key={tip.label}
                  initial={{ x: -20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 0.15 + i * 0.1, type: 'spring', stiffness: 250 }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '10px 14px',
                    borderRadius: 10,
                    background: `${tip.color}08`,
                    border: `1px solid ${tip.color}15`,
                  }}
                >
                  {tip.icon}
                  <span style={{ color: '#D1D5DB', fontSize: 13 }}>{tip.label}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        );

      default:
        return null;
    }
  };

  return createPortal(
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 999999,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'radial-gradient(ellipse at 30% 20%, rgba(124,77,255,0.08) 0%, transparent 50%), radial-gradient(ellipse at 70% 80%, rgba(0,229,255,0.06) 0%, transparent 50%), radial-gradient(ellipse at center, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.97) 100%)',
          backdropFilter: 'blur(20px)',
        }}
        onClick={(e) => e.target === e.currentTarget && handleClose()}
      >
        {/* Falling confetti with wobble */}
        {confetti.map((p) => (
          <motion.div
            key={p.id}
            initial={{ y: -20, x: `${p.x}vw`, opacity: 0.9, rotate: 0 }}
            animate={{
              y: '110vh',
              x: [`${p.x}vw`, `${p.x + p.wobble / 10}vw`, `${p.x}vw`],
              rotate: 720 * (p.id % 2 === 0 ? 1 : -1),
              opacity: [0.9, 0.8, 0],
            }}
            transition={{
              duration: p.duration,
              delay: p.delay,
              repeat: Infinity,
              ease: 'linear',
            }}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              width: p.size,
              height: p.size * (Math.random() > 0.5 ? 1 : 1.5),
              borderRadius: p.shape,
              background: p.color,
              boxShadow: `0 0 ${p.size}px ${p.color}40`,
              pointerEvents: 'none',
              zIndex: 999998,
            }}
          />
        ))}

        {/* Modal */}
        <motion.div
          initial={{ scale: 0.8, y: 40, opacity: 0 }}
          animate={{ scale: 1, y: 0, opacity: 1 }}
          exit={{ scale: 0.8, y: 40, opacity: 0 }}
          transition={{ type: 'spring', damping: 20, stiffness: 250 }}
          style={{
            position: 'relative',
            width: '93%',
            maxWidth: 500,
            borderRadius: 28,
            overflow: 'hidden',
            boxShadow: '0 0 80px rgba(0, 229, 255, 0.15), 0 0 160px rgba(124, 77, 255, 0.08), 0 25px 60px rgba(0,0,0,0.5)',
          }}
        >
          {/* Animated conic gradient border */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 10, repeat: Infinity, ease: 'linear' }}
            style={{
              position: 'absolute',
              inset: -2,
              borderRadius: 30,
              background: 'conic-gradient(from 0deg, #FFD700, #00E5FF, #7C4DFF, #FF6B35, #00E676, #E040FB, #FFD700)',
              opacity: 0.45,
            }}
          />
          {/* Glass inner panel */}
          <div style={{
            position: 'absolute',
            inset: 2,
            borderRadius: 26,
            background: 'linear-gradient(160deg, rgba(10,14,26,0.98) 0%, rgba(17,24,39,0.97) 50%, rgba(10,14,26,0.98) 100%)',
            zIndex: 1,
          }} />

          {/* Content */}
          <div style={{
            position: 'relative',
            zIndex: 2,
            padding: '26px 22px 18px',
          }}>
            {/* Close */}
            <button
              onClick={handleClose}
              style={{
                position: 'absolute', top: 10, right: 10,
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '50%',
                width: 28, height: 28,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: 'pointer', color: '#6B7280',
                transition: 'all 0.2s', zIndex: 10,
              }}
              onMouseOver={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.12)'; e.currentTarget.style.color = '#fff'; }}
              onMouseOut={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.06)'; e.currentTarget.style.color = '#6B7280'; }}
            >
              <X size={13} />
            </button>

            {/* Step indicators */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: 5, marginBottom: 18 }}>
              {steps.map((s, i) => (
                <motion.div
                  key={s.id}
                  animate={{
                    width: i === step ? 28 : 8,
                    background: i === step
                      ? 'linear-gradient(90deg, #FFD700, #FF6B35)'
                      : i < step ? '#00E5FF' : 'rgba(255,255,255,0.12)',
                    boxShadow: i === step ? '0 0 8px rgba(255,215,0,0.4)' : 'none',
                  }}
                  transition={{ type: 'spring', stiffness: 400, damping: 25 }}
                  style={{ height: 4, borderRadius: 2, cursor: 'pointer' }}
                  onClick={() => setStep(i)}
                />
              ))}
            </div>

            {/* Step content */}
            <div style={{ minHeight: 340 }}>
              <AnimatePresence mode="wait">
                {renderStep()}
              </AnimatePresence>
            </div>

            {/* Navigation */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              marginTop: 14, gap: 12,
            }}>
              {step > 0 ? (
                <button
                  onClick={prevStep}
                  style={{
                    padding: '9px 16px', borderRadius: 10,
                    border: '1px solid rgba(255,255,255,0.1)',
                    background: 'rgba(255,255,255,0.04)',
                    color: '#9CA3AF', fontSize: 13, fontWeight: 600,
                    cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5,
                    transition: 'all 0.2s',
                  }}
                  onMouseOver={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; }}
                  onMouseOut={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; }}
                >
                  <ChevronLeft size={14} /> Back
                </button>
              ) : <div />}

              {step < steps.length - 1 ? (
                <motion.button
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={nextStep}
                  style={{
                    padding: '10px 22px', borderRadius: 10,
                    border: 'none',
                    background: 'linear-gradient(135deg, #FFD700 0%, #FF6B35 100%)',
                    color: '#000', fontSize: 13, fontWeight: 700,
                    cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5,
                    boxShadow: '0 4px 18px rgba(255,215,0,0.25)',
                  }}
                >
                  Next <ChevronRight size={14} />
                </motion.button>
              ) : (
                <motion.button
                  whileHover={{ scale: 1.03, boxShadow: '0 8px 32px rgba(255,215,0,0.4)' }}
                  whileTap={{ scale: 0.97 }}
                  onClick={handleClose}
                  style={{
                    padding: '11px 26px', borderRadius: 12,
                    border: 'none',
                    background: 'linear-gradient(135deg, #FFD700 0%, #FF6B35 50%, #E040FB 100%)',
                    backgroundSize: '200% 100%',
                    color: '#000', fontSize: 14, fontWeight: 700,
                    cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 7,
                    boxShadow: '0 4px 24px rgba(255,215,0,0.3)',
                    transition: 'background-position 0.5s',
                  }}
                  onMouseOver={(e) => { e.currentTarget.style.backgroundPosition = '100% 0'; }}
                  onMouseOut={(e) => { e.currentTarget.style.backgroundPosition = '0% 0'; }}
                >
                  <Rocket size={16} />
                  Enter Mainnet
                  <ChevronRight size={16} />
                </motion.button>
              )}
            </div>

            {/* Version */}
            <div style={{ textAlign: 'center', marginTop: 12, color: '#374151', fontSize: 10 }}>
              Q-NarwhalKnight v7.3.0-mainnet2026.2 | Genesis: February 22, 2026
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body
  );
};

export default WelcomeMainnetModal;
