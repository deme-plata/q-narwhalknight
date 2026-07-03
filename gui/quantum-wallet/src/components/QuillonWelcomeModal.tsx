import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, ChevronRight, ChevronLeft, Sparkles, Shield, Zap, Lock, Cpu, Coins,
  BarChart3, Pickaxe, Rocket, Bot, Terminal, Copy, Check, Bitcoin, Network, Users,
} from 'lucide-react';

// One-shot startup welcome. Bumping this key re-shows the modal once for everyone.
// v4: everyone who saw the v3 broken-headline render gets the fixed modal once.
export const QUILLON_WELCOME_STORAGE_KEY = 'quillon_welcome_seen_v4';

interface QuillonWelcomeModalProps {
  onClose: () => void;
}

const SETUP_CMD = 'curl -fsSL https://quillon.xyz/setup-ai.sh | bash';
const SETUP_CMD_WIN = 'irm https://quillon.xyz/setup-ai.ps1 | iex';

// ⚠️ COLOR TRAP: the frameless theme carries [style*="255, 215, 0"] / [style*="212, 175, 55"]
// !important overrides that indigo-wash ANY element whose inline style serializes those gold
// rgb values (plus purples 139/92/246, 147/51/234, 168/85/247 and slates 15/23/42, 30/41/59,
// 20/15/40, 40/25/60). Use QW_GOLD (#FFC93C → "255, 201, 60") — never #FFD700 / #D4AF37 —
// and keep every inline color off that blacklist.
const QW_GOLD = '#FFC93C';

// Shared keyframes for the animated gradient sweeps (headline + CTA).
const QW_KEYFRAMES = `
@keyframes qwGradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
`;

// ─── Aurora sweep — soft 3-stop radial halo drifting behind everything ──────
const AuroraLayer: React.FC<{ hue: string; delay: number; x: string; y: string }> = ({ hue, delay, x, y }) => (
  <motion.div
    className="absolute rounded-full pointer-events-none"
    style={{
      width: '70%',
      height: '70%',
      left: x,
      top: y,
      background: `radial-gradient(circle at 50% 50%, ${hue} 0%, rgba(124,77,255,0.05) 40%, transparent 70%)`,
      filter: 'blur(48px)',
    }}
    animate={{
      x: ['-8%', '10%', '-5%', '-8%'],
      y: ['-6%', '8%', '12%', '-6%'],
      scale: [1, 1.25, 0.9, 1],
      opacity: [0.4, 0.65, 0.45, 0.4],
    }}
    transition={{ duration: 16, delay, repeat: Infinity, ease: 'easeInOut' }}
  />
);

// ─── Static starfield (cheap: one twinkle loop per star, no layout work) ─────
const Starfield: React.FC = () => {
  const stars = useMemo(
    () =>
      Array.from({ length: 46 }, (_, i) => ({
        id: i,
        left: Math.random() * 100,
        top: Math.random() * 100,
        size: 0.8 + Math.random() * 1.6,
        opacity: 0.25 + Math.random() * 0.55,
        twinkle: 2.5 + Math.random() * 4,
        delay: Math.random() * 4,
      })),
    []
  );
  return (
    <>
      {stars.map((s) => (
        <motion.div
          key={s.id}
          className="absolute rounded-full bg-white pointer-events-none"
          style={{ left: `${s.left}%`, top: `${s.top}%`, width: s.size, height: s.size }}
          animate={{ opacity: [s.opacity, s.opacity * 0.3, s.opacity] }}
          transition={{ duration: s.twinkle, delay: s.delay, repeat: Infinity, ease: 'easeInOut' }}
        />
      ))}
    </>
  );
};

// ─── Drifting ember particles ────────────────────────────────────────────────
const Ember: React.FC<{ index: number }> = ({ index }) => {
  const seed = useMemo(
    () => ({
      left: Math.random() * 100,
      size: 1.5 + Math.random() * 3,
      duration: 7 + Math.random() * 10,
      delay: Math.random() * 8,
      drift: (Math.random() - 0.5) * 80,
      color: ['#00E5FF', QW_GOLD, '#7C4DFF', '#00E676', '#FF6B35'][index % 5],
    }),
    [index]
  );
  return (
    <motion.div
      className="absolute rounded-full pointer-events-none"
      style={{
        left: `${seed.left}%`,
        bottom: '-2%',
        width: seed.size,
        height: seed.size,
        background: seed.color,
        boxShadow: `0 0 ${seed.size * 3}px ${seed.color}`,
      }}
      initial={{ y: 0, opacity: 0 }}
      animate={{ y: '-105vh', x: seed.drift, opacity: [0, 0.9, 0.7, 0] }}
      transition={{ duration: seed.duration, delay: seed.delay, repeat: Infinity, ease: 'linear' }}
    />
  );
};

// ─── Living DAG constellation — blocks weaving into the graph ────────────────
const DAG_NODES: Array<{ x: number; y: number; r: number; c: string; tip?: boolean }> = [
  { x: 26, y: 88, r: 4, c: '#7C4DFF' },
  { x: 62, y: 112, r: 3.4, c: '#00E5FF' },
  { x: 66, y: 62, r: 3.4, c: '#00E5FF' },
  { x: 108, y: 92, r: 4.4, c: '#00E676' },
  { x: 112, y: 34, r: 3.2, c: '#7C4DFF' },
  { x: 122, y: 138, r: 3.2, c: '#FF6B35' },
  { x: 158, y: 66, r: 4, c: '#00E5FF' },
  { x: 164, y: 116, r: 3.6, c: '#E040FB' },
  { x: 204, y: 90, r: 4.6, c: '#00E676' },
  { x: 212, y: 40, r: 3.2, c: '#00E5FF' },
  { x: 222, y: 140, r: 3.2, c: '#7C4DFF' },
  { x: 254, y: 64, r: 3.8, c: '#FF6B35' },
  { x: 262, y: 114, r: 3.8, c: '#00E5FF' },
  { x: 300, y: 88, r: 6, c: QW_GOLD, tip: true },
];
const DAG_EDGES: Array<[number, number]> = [
  [0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [1, 5], [4, 6], [3, 6], [3, 7], [5, 7],
  [6, 8], [7, 8], [6, 9], [7, 10], [9, 11], [8, 11], [8, 12], [10, 12], [11, 13], [12, 13],
];

const DagConstellation: React.FC = () => (
  <div
    className="relative w-full flex justify-center rounded-2xl"
    style={{
      // ambient depth: twin radial glows guiding the eye to the graph
      background:
        'radial-gradient(ellipse at 30% 20%, rgba(0,229,255,0.12) 0%, transparent 60%), radial-gradient(ellipse at 70% 80%, rgba(124,77,255,0.08) 0%, transparent 50%)',
    }}
  >
    <motion.svg
      viewBox="0 0 326 176"
      className="w-full max-w-[380px] h-auto"
      animate={{ y: [-3, 3, -3] }}
      transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
    >
      <defs>
        <filter id="qwGlow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="2.4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <radialGradient id="qwTip">
          <stop offset="0%" stopColor="#FFF3C4" />
          <stop offset="45%" stopColor={QW_GOLD} />
          <stop offset="100%" stopColor="#FF6B35" />
        </radialGradient>
      </defs>

      {DAG_EDGES.map(([a, b], i) => (
        <motion.line
          key={`e${i}`}
          x1={DAG_NODES[a].x}
          y1={DAG_NODES[a].y}
          x2={DAG_NODES[b].x}
          y2={DAG_NODES[b].y}
          strokeWidth="1.1"
          strokeLinecap="round"
          style={{ stroke: 'rgba(0,229,255,0.35)' }}
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ delay: 0.25 + i * 0.07, duration: 0.5, ease: 'easeOut' }}
        />
      ))}

      {DAG_NODES.map((n, i) => (
        <g key={`n${i}`}>
          {n.tip && (
            <motion.circle
              cx={n.x}
              cy={n.y}
              r={n.r}
              fill="none"
              stroke={QW_GOLD}
              strokeWidth="1"
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 0.55, 0], scale: [1, 2.6, 1] }}
              transition={{ delay: 1.9, duration: 2.4, repeat: Infinity, ease: 'easeOut' }}
              style={{ transformOrigin: `${n.x}px ${n.y}px` }}
            />
          )}
          <motion.circle
            cx={n.x}
            cy={n.y}
            r={n.r}
            fill={n.tip ? 'url(#qwTip)' : n.c}
            filter="url(#qwGlow)"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.35 + i * 0.09, type: 'spring', stiffness: 320, damping: 16 }}
            style={{ transformOrigin: `${n.x}px ${n.y}px` }}
          />
        </g>
      ))}

      {/* a spark riding the spine of the DAG toward the tip — opacity cycle matches the
          2.8s SMIL path period exactly so the fade always tracks the ride */}
      <motion.circle r="2.2" fill="#fff" filter="url(#qwGlow)" initial={{ opacity: 0 }} animate={{ opacity: [0, 1, 1, 0] }} transition={{ delay: 1.6, duration: 2.8, times: [0, 0.12, 0.88, 1], repeat: Infinity }}>
        <animateMotion
          dur="2.8s"
          begin="1.6s"
          repeatCount="indefinite"
          path="M26,88 L108,92 L158,66 L204,90 L262,114 L300,88"
        />
      </motion.circle>
    </motion.svg>
  </div>
);

// ─── Animated counter ────────────────────────────────────────────────────────
const AnimatedCounter: React.FC<{ target: number; duration?: number }> = ({ target, duration = 1600 }) => {
  const [count, setCount] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const step = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(target * eased);
      if (progress < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return <span>{Math.round(count).toLocaleString()}</span>;
};

// ─── Stat chip ───────────────────────────────────────────────────────────────
const StatChip: React.FC<{ icon: React.ReactNode; label: string; value: React.ReactNode; delay: number; color: string }> = ({ icon, label, value, delay, color }) => (
  <motion.div
    initial={{ opacity: 0, y: 14, scale: 0.92 }}
    animate={{ opacity: 1, y: 0, scale: 1 }}
    transition={{ delay, type: 'spring', stiffness: 240, damping: 18 }}
    className="flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl border backdrop-blur-sm"
    style={{ background: `linear-gradient(140deg, ${color}14, ${color}05)`, borderColor: `${color}30` }}
  >
    <span style={{ color }}>{icon}</span>
    <div className="leading-tight">
      <div className="text-[10px] uppercase tracking-wider text-gray-400">{label}</div>
      <div className="text-sm font-bold text-white font-mono">{value}</div>
    </div>
  </motion.div>
);

// ─── Terminal typewriter for the MCP step ────────────────────────────────────
const TERMINAL_SCRIPT: Array<{ text: string; cls: string; prompt?: boolean }> = [
  { text: SETUP_CMD, cls: 'text-cyan-200', prompt: true },
  { text: '✓ Quillon Wallet MCP v2.19 installed', cls: 'text-emerald-300' },
  { text: '✓ 80+ on-chain tools registered with your agent', cls: 'text-emerald-300' },
  { text: '✓ Claude Code · Cursor · Codex · Qwen · Grok detected automatically', cls: 'text-emerald-300' },
  { text: '» now just say: "create a wallet and start mining"', cls: 'text-amber-200' },
];

const TerminalDemo: React.FC = () => {
  const [lineIdx, setLineIdx] = useState(0);
  const [charIdx, setCharIdx] = useState(0);

  useEffect(() => {
    if (lineIdx >= TERMINAL_SCRIPT.length) return;
    const line = TERMINAL_SCRIPT[lineIdx];
    // the command types slowly; output lines land fast
    const speed = lineIdx === 0 ? 22 : 6;
    if (charIdx < line.text.length) {
      const t = setTimeout(() => setCharIdx((c) => c + 1), speed);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => {
      setLineIdx((l) => l + 1);
      setCharIdx(0);
    }, lineIdx === 0 ? 550 : 260);
    return () => clearTimeout(t);
  }, [lineIdx, charIdx]);

  return (
    <div
      className="rounded-2xl overflow-hidden font-mono text-[12px] sm:text-[13px]"
      style={{
        // floating glass terminal, not a dead black box
        border: '1px solid rgba(0,229,255,0.15)',
        background: 'linear-gradient(145deg, rgba(9,14,34,0.85), rgba(20,30,60,0.7))',
        backdropFilter: 'blur(12px)',
        boxShadow: '0 12px 40px rgba(0,0,0,0.45), 0 0 40px rgba(0,229,255,0.06)',
      }}
    >
      <div
        className="flex items-center gap-1.5 px-3.5 py-2.5"
        style={{
          borderBottom: '1px solid transparent',
          borderImage: 'linear-gradient(90deg, rgba(0,229,255,0.3), rgba(224,64,251,0.2), transparent) 1',
          background: 'rgba(255,255,255,0.03)',
        }}
      >
        <span className="w-2.5 h-2.5 rounded-full bg-[#FF5F57]" />
        <span className="w-2.5 h-2.5 rounded-full bg-[#FEBC2E]" />
        <span className="w-2.5 h-2.5 rounded-full bg-[#28C840]" />
        <span className="ml-2 text-[10px] uppercase tracking-widest text-gray-500">your-ai — quillon graph</span>
      </div>
      <div className="px-4 py-3.5 space-y-1.5 min-h-[124px]">
        {TERMINAL_SCRIPT.slice(0, lineIdx + 1).map((line, i) => {
          const done = i < lineIdx;
          const shown = done ? line.text : line.text.slice(0, charIdx);
          if (i >= TERMINAL_SCRIPT.length) return null;
          return (
            <div key={i} className={`${line.cls} leading-relaxed break-all`}>
              {line.prompt && <span className="text-fuchsia-400 select-none">$ </span>}
              {shown}
              {!done && i === lineIdx && (
                <motion.span
                  animate={{ opacity: [1, 0] }}
                  transition={{ duration: 0.6, repeat: Infinity }}
                  className="inline-block w-1.5 h-3.5 ml-0.5 align-middle bg-cyan-300/90"
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ─── Gradient section eyebrow ────────────────────────────────────────────────
const Eyebrow: React.FC<{ icon: React.ReactNode; text: string; color: string }> = ({ icon, text, color }) => (
  <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
    <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.28em] font-semibold" style={{ color }}>
      {icon}
      {text}
    </div>
    <div
      className="mt-1.5 h-px w-24"
      style={{ background: `linear-gradient(90deg, ${color}99, transparent)` }}
    />
  </motion.div>
);

// ─── Steps ───────────────────────────────────────────────────────────────────
const STEPS = ['Genesis', 'Powers', 'AI Agents', 'Launch'];

// ─── Main modal ──────────────────────────────────────────────────────────────
const QuillonWelcomeModal: React.FC<QuillonWelcomeModalProps> = ({ onClose }) => {
  const [step, setStep] = useState(0);
  const [copied, setCopied] = useState(false);
  const [stats, setStats] = useState<{ height?: number; peers?: number; version?: string }>({});

  // Belt and suspenders: mark seen on mount so a mid-view reload never loops it.
  useEffect(() => {
    localStorage.setItem(QUILLON_WELCOME_STORAGE_KEY, 'true');
  }, []);

  const dismiss = useCallback(() => {
    localStorage.setItem(QUILLON_WELCOME_STORAGE_KEY, 'true');
    onClose();
  }, [onClose]);

  useEffect(() => {
    const esc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') dismiss();
    };
    window.addEventListener('keydown', esc);
    return () => window.removeEventListener('keydown', esc);
  }, [dismiss]);

  useEffect(() => {
    // /api/v1/status only carries version — height + peers live on /api/v1/node/status
    // (both public, no auth). Fetch both and merge.
    fetch('/api/v1/node/status')
      .then((r) => r.json())
      .then((j) => {
        const d = j?.data ?? j;
        setStats((s) => ({
          ...s,
          height: d?.current_height ?? d?.contiguous_height,
          peers: d?.connected_peers ?? d?.peer_count,
        }));
      })
      .catch(() => {});
    fetch('/api/v1/status')
      .then((r) => r.json())
      .then((j) => {
        const d = j?.data ?? j;
        setStats((s) => ({ ...s, version: d?.version }));
      })
      .catch(() => {});
  }, []);

  const copyCmd = useCallback(() => {
    try {
      navigator.clipboard?.writeText(SETUP_CMD).then(
        () => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2200);
        },
        () => {}
      );
    } catch {
      /* clipboard unavailable — non-fatal */
    }
  }, []);

  const embers = useMemo(() => Array.from({ length: 10 }, (_, i) => i), []);
  const next = () => setStep((s) => Math.min(s + 1, STEPS.length - 1));
  const prev = () => setStep((s) => Math.max(s - 1, 0));

  const renderStep = () => {
    switch (step) {
      // ═══ 0 · GENESIS ═══════════════════════════════════════════════════
      case 0:
        return (
          <motion.div
            key="genesis"
            initial={{ opacity: 0, x: 48 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -48 }}
            transition={{ duration: 0.35 }}
          >
            <Eyebrow icon={<Network size={13} />} text="mainnet-genesis · live" color="#00E5FF" />

            {/* SVG gradient text: CSS background-clip text kept getting stripped by
                theme/override rules (and framer-motion resets the clip on motion
                elements), leaving invisible glyphs. SVG fill via inline style is
                untouchable by any of the app's theme CSS. */}
            <motion.h2
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.18 }}
              className="mt-3"
              aria-label="Welcome to Quillon Graph"
            >
              <svg
                viewBox="0 0 420 100"
                className="w-full max-w-[440px] h-auto block"
                style={{ filter: 'drop-shadow(0 2px 14px rgba(0,229,255,0.25))', overflow: 'visible' }}
                aria-hidden="true"
              >
                <defs>
                  <linearGradient id="qwHeadGrad" x1="0" y1="0" x2="1" y2="0.12">
                    <stop offset="0" stopColor="#FFFFFF" />
                    <stop offset="0.3" stopColor="#00E5FF" />
                    <stop offset="0.6" stopColor={QW_GOLD} />
                    <stop offset="0.85" stopColor="#FF6B35" />
                    <stop offset="1" stopColor="#E040FB" />
                    {/* slow gradient sweep, SMIL — no CSS animation dependency */}
                    <animate attributeName="x1" values="0;-0.8;0" dur="7s" repeatCount="indefinite" />
                    <animate attributeName="x2" values="1;1.8;1" dur="7s" repeatCount="indefinite" />
                  </linearGradient>
                </defs>
                <text x="0" y="38" style={{ fill: 'url(#qwHeadGrad)', fontWeight: 900, fontSize: 40, letterSpacing: '-0.5px' }}>
                  Welcome to
                </text>
                <text x="0" y="86" style={{ fill: 'url(#qwHeadGrad)', fontWeight: 900, fontSize: 40, letterSpacing: '-0.5px' }}>
                  Quillon Graph.
                </text>
              </svg>
            </motion.h2>

            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mt-3 text-[14.5px] leading-relaxed text-gray-300 max-w-lg"
            >
              A post-quantum blockDAG where blocks weave in parallel and finality lands in
              under a second — built for humans <span className="text-white font-semibold">and</span>{' '}
              for AI agents that hold wallets, mine, and trade on their own.
            </motion.p>

            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }} className="mt-4">
              <DagConstellation />
            </motion.div>

            <div className="mt-4 flex flex-wrap gap-2.5 justify-center">
              <StatChip
                icon={<BarChart3 size={15} />}
                label="Block height"
                value={stats.height ? <AnimatedCounter target={stats.height} /> : '—'}
                delay={0.55}
                color={QW_GOLD}
              />
              <StatChip icon={<Users size={15} />} label="Peers" value={stats.peers ?? '—'} delay={0.65} color="#00E5FF" />
              <StatChip icon={<Coins size={15} />} label="Supply cap" value="21M QUG" delay={0.75} color="#00E676" />
              <StatChip icon={<Zap size={15} />} label="Node" value={stats.version ? `v${stats.version}` : 'live'} delay={0.85} color="#7C4DFF" />
            </div>
          </motion.div>
        );

      // ═══ 1 · POWERS ════════════════════════════════════════════════════
      case 1:
        return (
          <motion.div
            key="powers"
            initial={{ opacity: 0, x: 48 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -48 }}
            transition={{ duration: 0.35 }}
          >
            <div className="text-center mb-4">
              <motion.div
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 24, repeat: Infinity, ease: 'linear' }}
                className="inline-block"
              >
                <Cpu size={30} color="#7C4DFF" style={{ filter: 'drop-shadow(0 0 16px rgba(124,77,255,0.6))' }} />
              </motion.div>
              <h2 className="mt-1.5 text-2xl font-extrabold text-gray-100">Everything on one chain</h2>
              <div
                className="mx-auto mt-2 h-px w-32"
                style={{ background: 'linear-gradient(90deg, transparent, rgba(124,77,255,0.8), transparent)' }}
              />
              <p className="text-xs text-gray-500 mt-2">No extensions, no wrapped layers — it all ships in the node.</p>
            </div>

            <div className="grid sm:grid-cols-2 gap-2.5">
              {[
                { icon: <Shield size={16} />, title: 'Post-Quantum Security', body: 'Dilithium5 + Kyber1024 hybrid cryptography, hardened for the quantum era.', c: '#00E676' },
                { icon: <Zap size={16} />, title: 'DAG-Knight Consensus', body: 'Parallel blockDAG ordering with sub-second finality.', c: QW_GOLD },
                { icon: <BarChart3 size={16} />, title: 'Built-in DEX', body: 'AMM pools, swaps and LP fees — trade QUG and custom tokens natively.', c: '#00E5FF' },
                { icon: <Lock size={16} />, title: 'Privacy Layer', body: 'Ring signatures, bulletproofs and mixing for confidential transfers.', c: '#7C4DFF' },
                { icon: <Bitcoin size={16} />, title: 'Bitcoin Bridge + Lightning', body: 'BTC deposits and withdrawals, plus instant Lightning payments.', c: '#FF6B35' },
                { icon: <Pickaxe size={16} />, title: 'Fair CPU Mining', body: 'Mine QUG on any computer. 21M cap, 64 halving eras, 256 years.', c: '#E040FB' },
                { icon: <Cpu size={16} />, title: 'Smart Contracts', body: 'Deploy tokens and dApps on the built-in WASM VM.', c: '#5EF2C2' },
                { icon: <Bot size={16} />, title: 'Agent Economy', body: 'AI agents hold wallets, earn QUG for real work, and trade with each other.', c: '#FF4081' },
              ].map((f, i) => (
                // gradient-border card: 1px gradient shell around a glass core
                <motion.div
                  key={f.title}
                  initial={{ opacity: 0, y: 16, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ delay: 0.08 + i * 0.06, type: 'spring', stiffness: 260, damping: 20 }}
                  whileHover={{ y: -3, transition: { duration: 0.15 } }}
                  className="rounded-2xl p-[1px]"
                  style={{ background: `linear-gradient(135deg, ${f.c}59 0%, ${f.c}14 45%, transparent 70%, ${f.c}26 100%)` }}
                >
                  <div
                    className="p-3 rounded-2xl h-full"
                    style={{ background: `linear-gradient(140deg, ${f.c}10 0%, rgba(9,14,34,0.92) 55%, rgba(5,9,24,0.95) 100%)` }}
                  >
                    <div className="flex items-center gap-2 text-sm font-bold" style={{ color: f.c }}>
                      <span
                        className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                        style={{ background: `linear-gradient(135deg, ${f.c}30, ${f.c}0D)` }}
                      >
                        {f.icon}
                      </span>
                      <span className="text-gray-100">{f.title}</span>
                    </div>
                    <div className="mt-1.5 text-[11.5px] leading-relaxed text-gray-400">{f.body}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        );

      // ═══ 2 · AI AGENTS (MCP) ═══════════════════════════════════════════
      case 2:
        return (
          <motion.div
            key="agents"
            initial={{ opacity: 0, x: 48 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -48 }}
            transition={{ duration: 0.35 }}
          >
            <Eyebrow icon={<Bot size={13} />} text="quillon wallet mcp" color="#E040FB" />

            <h2 className="mt-2.5" aria-label="Plug your AI into the chain.">
              <svg
                viewBox="0 0 470 40"
                className="w-full max-w-[480px] h-auto block"
                style={{ filter: 'drop-shadow(0 2px 12px rgba(224,64,251,0.2))', overflow: 'visible' }}
                aria-hidden="true"
              >
                <defs>
                  <linearGradient id="qwAiGrad" x1="0" y1="0" x2="1" y2="0.1">
                    <stop offset="0.1" stopColor="#FFFFFF" />
                    <stop offset="0.55" stopColor="#E040FB" />
                    <stop offset="0.95" stopColor="#00E5FF" />
                  </linearGradient>
                </defs>
                <text x="0" y="30" style={{ fill: 'url(#qwAiGrad)', fontWeight: 900, fontSize: 31, letterSpacing: '-0.4px' }}>
                  Plug your AI into the chain.
                </text>
              </svg>
            </h2>

            <p className="mt-2.5 text-[13.5px] leading-relaxed text-gray-300">
              One command turns any AI coding agent into a first-class citizen of Quillon Graph —
              with its own wallet, mining rig and DEX seat. This isn&apos;t a demo: autonomous agents
              already live here, earning QUG for real engineering work.
            </p>

            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }} className="mt-4">
              <TerminalDemo />
            </motion.div>

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <motion.button
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
                whileTap={{ scale: 0.96 }}
                onClick={copyCmd}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold border transition-colors"
                style={{
                  color: copied ? '#00E676' : '#9CA3AF',
                  borderColor: copied ? 'rgba(0,230,118,0.4)' : 'rgba(255,255,255,0.12)',
                  background: copied ? 'rgba(0,230,118,0.08)' : 'rgba(255,255,255,0.04)',
                }}
              >
                {copied ? <Check size={12} /> : <Copy size={12} />}
                {copied ? 'Copied!' : 'Copy command'}
              </motion.button>
              <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="text-[10.5px] text-gray-500 font-mono break-all min-w-0">
                Windows: {SETUP_CMD_WIN}
              </motion.span>
            </div>

            <div className="mt-4 grid sm:grid-cols-3 gap-2.5">
              {[
                { icon: <Coins size={15} />, title: 'Wallets & sends', body: '"Send 5 QUG to Adrian" — signed and settled on-chain.', c: QW_GOLD },
                { icon: <BarChart3 size={15} />, title: 'Trading & tokens', body: '"Swap QUG for CULTURE" or "deploy my own token".', c: '#00E5FF' },
                { icon: <Pickaxe size={15} />, title: 'Mining & LP', body: '"Start mining" and "add liquidity" — income while you sleep.', c: '#00E676' },
              ].map((f, i) => (
                <motion.div
                  key={f.title}
                  initial={{ opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.55 + i * 0.1 }}
                  className="rounded-xl p-[1px]"
                  style={{ background: `linear-gradient(135deg, ${f.c}4D, transparent 60%, ${f.c}1F)` }}
                >
                  <div className="p-3 rounded-xl h-full" style={{ background: `linear-gradient(140deg, ${f.c}0D, rgba(9,14,34,0.92))` }}>
                    <div className="flex items-center gap-1.5 text-[12.5px] font-bold" style={{ color: f.c }}>
                      {f.icon}
                      <span className="text-gray-200">{f.title}</span>
                    </div>
                    <div className="mt-1 text-[11px] leading-relaxed text-gray-400">{f.body}</div>
                  </div>
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.9 }}
              className="mt-3.5 flex items-center justify-center gap-2 text-[10.5px] text-gray-500"
            >
              <Terminal size={11} />
              Works with Claude Code · Cursor · Codex · Qwen Coder · Grok — auto-detected
            </motion.div>
          </motion.div>
        );

      // ═══ 3 · LAUNCH ════════════════════════════════════════════════════
      case 3:
        return (
          <motion.div
            key="launch"
            initial={{ opacity: 0, x: 48 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -48 }}
            transition={{ duration: 0.35 }}
            className="text-center"
          >
            <div className="relative w-28 h-28 mx-auto mb-3">
              {[0, 0.6, 1.2].map((d, i) => (
                <motion.div
                  key={i}
                  className="absolute inset-0 m-auto rounded-full border pointer-events-none"
                  style={{ width: 110 - i * 28, height: 110 - i * 28, borderColor: ['#FF6B35', QW_GOLD, '#00E5FF'][i] }}
                  initial={{ opacity: 0, scale: 0.4 }}
                  animate={{ opacity: [0, 0.5, 0], scale: [0.4, 1.35, 0.4] }}
                  transition={{ duration: 4.5, delay: d, repeat: Infinity, ease: 'easeInOut' }}
                />
              ))}
              <motion.div
                animate={{ y: [-4, 4, -4], rotate: [0, 3, -3, 0] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                className="absolute inset-0 flex items-center justify-center"
              >
                <Rocket size={42} color="#FF6B35" style={{ filter: 'drop-shadow(0 0 22px rgba(255,107,53,0.7))' }} />
              </motion.div>
            </div>

            <h2 className="text-2xl sm:text-[1.8rem] font-extrabold text-gray-100">You&apos;re on the Graph.</h2>
            <p className="mt-1.5 text-[13px] text-gray-400 max-w-md mx-auto leading-relaxed">
              Your gateway node is synced and listening. Here&apos;s where the fun starts:
            </p>

            <div className="mt-4 flex flex-col gap-2 max-w-md mx-auto text-left">
              {[
                { icon: <Pickaxe size={14} />, label: 'Open the Mining tab and earn your first QUG', c: QW_GOLD },
                { icon: <BarChart3 size={14} />, label: 'Swap and provide liquidity on the DEX', c: '#00E5FF' },
                { icon: <Cpu size={14} />, label: 'Deploy your own token in a few clicks', c: '#7C4DFF' },
                { icon: <Bot size={14} />, label: 'Onboard your AI agent with the MCP one-liner', c: '#FF4081' },
              ].map((t, i) => (
                <motion.div
                  key={t.label}
                  initial={{ opacity: 0, x: -18 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.15 + i * 0.1, type: 'spring', stiffness: 250 }}
                  className="relative overflow-hidden flex items-center gap-2.5 pl-4 pr-3.5 py-2.5 rounded-xl border"
                  style={{ background: `linear-gradient(90deg, ${t.c}12, ${t.c}04 60%, transparent)`, borderColor: `${t.c}1F` }}
                >
                  {/* gradient accent bar */}
                  <span
                    className="absolute left-0 top-0 bottom-0 w-[3px]"
                    style={{ background: `linear-gradient(180deg, ${t.c}, ${t.c}00)` }}
                  />
                  <span style={{ color: t.c }}>{t.icon}</span>
                  <span className="text-[13px] text-gray-300">{t.label}</span>
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
        key="quillon-welcome-backdrop"
        className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        style={{ background: 'rgba(2, 6, 18, 0.9)', backdropFilter: 'blur(12px)' }}
        onClick={dismiss}
      >
        <style>{QW_KEYFRAMES}</style>
        {/* deep-space field */}
        <Starfield />
        <AuroraLayer hue="rgba(0,229,255,0.16)" delay={0} x="-10%" y="-15%" />
        <AuroraLayer hue="rgba(124,77,255,0.14)" delay={4} x="50%" y="40%" />
        <AuroraLayer hue="rgba(255,64,129,0.09)" delay={8} x="20%" y="60%" />
        {embers.map((i) => (
          <Ember key={i} index={i} />
        ))}

        <motion.div
          key="quillon-welcome-card"
          onClick={(e) => e.stopPropagation()}
          initial={{ opacity: 0, scale: 0.86, y: 36, rotateX: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0, rotateX: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          transition={{ type: 'spring', stiffness: 200, damping: 22 }}
          className="relative w-full max-w-2xl rounded-3xl"
          style={{ boxShadow: '0 0 90px rgba(0,229,255,0.1), 0 0 50px rgba(124,77,255,0.08), 0 24px 80px rgba(0,0,0,0.6)' }}
        >
          {/* static premium gradient ring (calm, no rotation) */}
          <div
            className="absolute inset-0 rounded-3xl pointer-events-none"
            style={{
              background:
                `linear-gradient(135deg, rgba(0,229,255,0.55) 0%, rgba(124,77,255,0.35) 28%, rgba(224,64,251,0.3) 52%, rgba(255,201,60,0.45) 78%, rgba(255,107,53,0.4) 100%)`,
            }}
          />

          {/* glass panel — translucent gradient, reads as real glass */}
          <div
            className="relative rounded-3xl overflow-hidden"
            style={{
              margin: 1.5,
              background: 'linear-gradient(160deg, rgba(9,14,34,0.9) 0%, rgba(16,24,48,0.86) 45%, rgba(5,9,24,0.93) 100%)',
              backdropFilter: 'blur(14px)',
            }}
          >
            {/* top shimmer line */}
            <motion.div
              className="absolute inset-x-0 top-0 h-px pointer-events-none"
              style={{
                background: `linear-gradient(90deg, transparent, #00E5FF, ${QW_GOLD}, #E040FB, transparent)`,
                backgroundSize: '200% 100%',
              }}
              animate={{ backgroundPosition: ['200% 0', '-200% 0'] }}
              transition={{ duration: 5, repeat: Infinity, ease: 'linear' }}
            />

            <button
              onClick={dismiss}
              className="absolute top-4 right-4 z-10 p-2 rounded-full bg-white/5 hover:bg-white/15 text-gray-400 hover:text-white transition-colors"
              aria-label="Close"
            >
              <X size={16} />
            </button>

            <div className="p-6 sm:p-8 max-h-[88vh] overflow-y-auto">
              {/* step dots */}
              <div className="flex justify-center gap-1.5 mb-5">
                {STEPS.map((label, i) => (
                  <motion.button
                    key={label}
                    onClick={() => setStep(i)}
                    animate={{
                      width: i === step ? 30 : 9,
                      // all states share the same gradient format so framer-motion can tween them
                      background:
                        i === step
                          ? `linear-gradient(90deg, #00E5FF, ${QW_GOLD})`
                          : i < step
                            ? 'linear-gradient(90deg, rgba(0,229,255,0.7), rgba(0,229,255,0.7))'
                            : 'linear-gradient(90deg, rgba(255,255,255,0.14), rgba(255,255,255,0.14))',
                      boxShadow: i === step ? '0 0 10px rgba(0,229,255,0.45)' : '0 0 0px rgba(0,229,255,0)',
                    }}
                    transition={{ type: 'spring', stiffness: 400, damping: 26 }}
                    className="h-[5px] rounded-full cursor-pointer border-0 p-0"
                    aria-label={label}
                  />
                ))}
              </div>

              {/* step content */}
              <div className="min-h-[380px]">
                <AnimatePresence mode="wait">{renderStep()}</AnimatePresence>
              </div>

              {/* nav */}
              <div className="mt-5 flex items-center justify-between gap-3">
                {step > 0 ? (
                  <button
                    onClick={prev}
                    className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl text-[13px] font-semibold text-gray-400 bg-white/[0.04] border border-white/10 hover:bg-white/[0.08] transition-colors"
                  >
                    <ChevronLeft size={14} /> Back
                  </button>
                ) : (
                  <div />
                )}

                {step < STEPS.length - 1 ? (
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={next}
                    className="flex items-center gap-1.5 px-6 py-2.5 rounded-xl text-[13px] font-bold text-gray-900"
                    style={{
                      background: `linear-gradient(95deg, #00E5FF 0%, #5EF2C2 55%, ${QW_GOLD} 100%)`,
                      boxShadow: '0 4px 20px rgba(0,229,255,0.25)',
                    }}
                  >
                    {step === 1 ? 'Meet the agents' : 'Next'} <ChevronRight size={14} />
                  </motion.button>
                ) : (
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={dismiss}
                    className="relative overflow-hidden flex items-center gap-2 px-7 py-3 rounded-xl text-[14px] font-bold text-white"
                    style={{
                      // living metallic gradient (DeepSeek pick): gold → orange → magenta → violet
                      background: `linear-gradient(135deg, ${QW_GOLD}, #FF6B35, #E040FB, #7C4DFF)`,
                      backgroundSize: '300% 300%',
                      animation: 'qwGradientShift 4s ease infinite',
                      boxShadow: '0 6px 28px rgba(224,64,251,0.3)',
                      textShadow: '0 1px 6px rgba(0,0,0,0.35)',
                    }}
                  >
                    <Sparkles size={16} />
                    Enter the Graph
                    <ChevronRight size={16} />
                  </motion.button>
                )}
              </div>

              {/* footer */}
              <div className="mt-4 text-center text-[10px] text-gray-600">
                Quillon Graph · mainnet-genesis {stats.version ? `· node v${stats.version}` : ''} · quillon.xyz
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body
  );
};

export default QuillonWelcomeModal;
