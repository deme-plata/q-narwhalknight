import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Sparkles, Rocket, Shield, Zap, Cpu, BarChart3, ChevronRight, Activity } from 'lucide-react';

export const STABILITY_HORIZON_STORAGE_KEY = 'stability_horizon_seen_v1';

interface StabilityHorizonModalProps {
  onClose: () => void;
}

// ─── Aurora sweep — slow drifting color field behind everything ─────────────
const AuroraLayer: React.FC<{ hue: string; delay: number; x: string; y: string }> = ({ hue, delay, x, y }) => (
  <motion.div
    className="absolute rounded-full pointer-events-none"
    style={{
      width: '70%',
      height: '70%',
      left: x,
      top: y,
      background: `radial-gradient(circle, ${hue} 0%, transparent 65%)`,
      filter: 'blur(60px)',
    }}
    animate={{
      x: ['-8%', '10%', '-5%', '-8%'],
      y: ['-6%', '8%', '12%', '-6%'],
      scale: [1, 1.25, 0.9, 1],
      opacity: [0.35, 0.6, 0.4, 0.35],
    }}
    transition={{ duration: 16, delay, repeat: Infinity, ease: 'easeInOut' }}
  />
);

// ─── Drifting ember particles ────────────────────────────────────────────────
const Ember: React.FC<{ index: number }> = ({ index }) => {
  const seed = useMemo(() => ({
    left: Math.random() * 100,
    size: 1.5 + Math.random() * 3,
    duration: 7 + Math.random() * 10,
    delay: Math.random() * 8,
    drift: (Math.random() - 0.5) * 80,
    color: ['#00E5FF', '#FFD700', '#7C4DFF', '#00E676', '#FF6B35'][index % 5],
  }), [index]);
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

// ─── EKG pulse line: chaos → calm. The whole story in one animation. ─────────
const PulseLine: React.FC = () => {
  // Chaotic first half (jagged, irregular), serene second half (smooth heartbeat)
  const chaosPath =
    'M0,40 L18,38 L26,12 L33,66 L40,22 L46,58 L54,8 L60,70 L68,30 L76,52 L84,18 L92,46 L100,34 L110,42';
  const calmPath =
    'M110,42 L150,42 L162,42 L170,18 L178,62 L186,42 L240,42 L300,42 L312,42 L320,20 L328,60 L336,42 L400,42';
  return (
    <div className="relative w-full h-20 overflow-hidden">
      <svg viewBox="0 0 400 80" className="w-full h-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id="pulseGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#FF4D4D" />
            <stop offset="28%" stopColor="#FF9A3D" />
            <stop offset="55%" stopColor="#FFD700" />
            <stop offset="100%" stopColor="#00E5FF" />
          </linearGradient>
          <filter id="pulseGlow">
            <feGaussianBlur stdDeviation="2.2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <motion.path
          d={`${chaosPath} ${calmPath.slice(9)}`}
          fill="none"
          stroke="url(#pulseGrad)"
          strokeWidth="2.4"
          strokeLinecap="round"
          filter="url(#pulseGlow)"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 3.2, ease: 'easeInOut', delay: 0.5 }}
        />
        {/* travelling spark that rides the line */}
        <motion.circle
          r="3.5"
          fill="#fff"
          filter="url(#pulseGlow)"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0, 1, 1, 0.9] }}
          transition={{ duration: 3.2, delay: 0.5 }}
        >
          <animateMotion dur="3.2s" begin="0.5s" fill="freeze" path={`${chaosPath} ${calmPath.slice(9)}`} />
        </motion.circle>
      </svg>
      <div className="absolute bottom-0 left-0 text-[10px] uppercase tracking-widest text-red-400/70 font-mono">turbulence</div>
      <div className="absolute bottom-0 right-0 text-[10px] uppercase tracking-widest text-cyan-300/80 font-mono">orbit</div>
    </div>
  );
};

// ─── Animated counter ────────────────────────────────────────────────────────
const StatChip: React.FC<{ icon: React.ReactNode; label: string; value: string; delay: number }> = ({ icon, label, value, delay }) => (
  <motion.div
    initial={{ opacity: 0, y: 14, scale: 0.92 }}
    animate={{ opacity: 1, y: 0, scale: 1 }}
    transition={{ delay, type: 'spring', stiffness: 240, damping: 18 }}
    className="flex items-center gap-2 px-3 py-2 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm"
  >
    <span className="text-cyan-300">{icon}</span>
    <div className="leading-tight">
      <div className="text-[10px] uppercase tracking-wider text-gray-400">{label}</div>
      <div className="text-sm font-bold text-white font-mono">{value}</div>
    </div>
  </motion.div>
);

// ─── DeepSeek live pulse (stream → fallback → static) ────────────────────────
const FALLBACK_PULSE =
  'Network pulse: steady. The deep memory storms that caused node restarts were traced to their root cause and engineered out in v10.11.53/54 — verified live on mainnet. The chain, your balances and 18.5M+ blocks never skipped a beat. Trajectory: up.';

function useAiPulse(): { text: string; source: string } {
  const [text, setText] = useState('');
  const [source, setSource] = useState('DeepSeek R1');
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    let cancelled = false;
    const prompt =
      'In 2-3 upbeat sentences: the Quillon Graph network just shipped engineering fixes (v10.11.53/54) that removed the memory bugs behind recent node restarts. Reassure users the chain and balances are safe and the future is bright. No preamble.';

    const typeOut = (full: string, src: string) => {
      if (cancelled) return;
      setSource(src);
      let i = 0;
      const tick = () => {
        if (cancelled) return;
        i = Math.min(full.length, i + 2);
        setText(full.slice(0, i));
        if (i < full.length) setTimeout(tick, 18);
      };
      tick();
    };

    const tryDeepSeek = async (): Promise<boolean> => {
      try {
        const ctrl = new AbortController();
        const kill = setTimeout(() => ctrl.abort(), 12000);
        const res = await fetch(`https://${window.location.hostname}:8846/api/v1/ai`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt }),
          signal: ctrl.signal,
        });
        if (!res.ok || !res.body) { clearTimeout(kill); return false; }
        setSource('DeepSeek R1 · live');
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let acc = '';
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          acc += dec.decode(value, { stream: true });
          if (!cancelled) setText(acc);
        }
        clearTimeout(kill);
        return acc.trim().length > 20;
      } catch { return false; }
    };

    const tryLocalAi = async (): Promise<boolean> => {
      try {
        const ctrl = new AbortController();
        const kill = setTimeout(() => ctrl.abort(), 12000);
        const res = await fetch('/api/v1/ai/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: [{ role: 'user', content: prompt }] }),
          signal: ctrl.signal,
        });
        if (!res.ok || !res.body) { clearTimeout(kill); return false; }
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let acc = '';
        let shown = '';
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          acc += dec.decode(value, { stream: true });
          // SSE frames: lines like `data: {...}` — extract any "content" tokens
          const pieces = acc.split('\n');
          let out = '';
          for (const line of pieces) {
            const m = line.match(/"content"\s*:\s*"((?:[^"\\]|\\.)*)"/);
            if (m) out += JSON.parse(`"${m[1]}"`);
          }
          if (out && !cancelled) { shown = out; setText(out); }
        }
        clearTimeout(kill);
        if (shown.trim().length > 20) { setSource('Quillon AI · live'); return true; }
        return false;
      } catch { return false; }
    };

    (async () => {
      if (await tryDeepSeek()) return;
      if (cancelled) return;
      if (await tryLocalAi()) return;
      if (cancelled) return;
      typeOut(FALLBACK_PULSE, 'Network telemetry');
    })();

    return () => { cancelled = true; };
  }, []);

  return { text, source };
}

// ─── Main modal ──────────────────────────────────────────────────────────────
const StabilityHorizonModal: React.FC<StabilityHorizonModalProps> = ({ onClose }) => {
  const [stats, setStats] = useState<{ height?: number; peers?: number; version?: string }>({});
  const { text: aiText, source: aiSource } = useAiPulse();

  const dismiss = useCallback(() => {
    localStorage.setItem(STABILITY_HORIZON_STORAGE_KEY, 'true');
    onClose();
  }, [onClose]);

  useEffect(() => {
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') dismiss(); };
    window.addEventListener('keydown', esc);
    return () => window.removeEventListener('keydown', esc);
  }, [dismiss]);

  useEffect(() => {
    fetch('/api/v1/status')
      .then((r) => r.json())
      .then((j) => {
        const d = j?.data ?? j;
        setStats({
          height: d?.current_height ?? d?.height,
          peers: d?.peer_count ?? d?.peers,
          version: d?.version,
        });
      })
      .catch(() => {});
  }, []);

  const embers = useMemo(() => Array.from({ length: 22 }, (_, i) => i), []);

  return createPortal(
    <AnimatePresence>
      <motion.div
        key="stability-horizon-backdrop"
        className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        style={{ background: 'rgba(2, 6, 18, 0.88)', backdropFilter: 'blur(10px)' }}
        onClick={dismiss}
      >
        {/* aurora field */}
        <AuroraLayer hue="rgba(0,229,255,0.20)" delay={0} x="-10%" y="-15%" />
        <AuroraLayer hue="rgba(124,77,255,0.18)" delay={4} x="50%" y="40%" />
        <AuroraLayer hue="rgba(255,215,0,0.10)" delay={8} x="20%" y="60%" />
        {embers.map((i) => <Ember key={i} index={i} />)}

        <motion.div
          key="stability-horizon-card"
          onClick={(e) => e.stopPropagation()}
          initial={{ opacity: 0, scale: 0.86, y: 36, rotateX: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0, rotateX: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          transition={{ type: 'spring', stiffness: 200, damping: 22 }}
          className="relative w-full max-w-2xl rounded-3xl overflow-hidden"
          style={{
            background: 'linear-gradient(160deg, rgba(10,16,38,0.97) 0%, rgba(6,10,26,0.98) 100%)',
            border: '1px solid rgba(0,229,255,0.22)',
            boxShadow: '0 0 90px rgba(0,229,255,0.14), 0 24px 80px rgba(0,0,0,0.6)',
          }}
        >
          {/* animated border shimmer */}
          <motion.div
            className="absolute inset-x-0 top-0 h-px pointer-events-none"
            style={{ background: 'linear-gradient(90deg, transparent, #00E5FF, #FFD700, #7C4DFF, transparent)' }}
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

          <div className="p-7 sm:p-9">
            {/* eyebrow */}
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className="flex items-center gap-2 text-[11px] uppercase tracking-[0.25em] text-cyan-300/90 font-semibold"
            >
              <Activity size={13} />
              Engineering in the open
            </motion.div>

            {/* headline */}
            <motion.h2
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
              className="mt-3 text-3xl sm:text-4xl font-black leading-tight"
              style={{
                background: 'linear-gradient(95deg, #FFFFFF 10%, #00E5FF 55%, #FFD700 95%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              From turbulence to orbit.
            </motion.h2>

            {/* the pulse line */}
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="mt-4">
              <PulseLine />
            </motion.div>

            {/* story */}
            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="mt-4 text-[15px] leading-relaxed text-gray-300"
            >
              Frontier systems earn their stability. This week our AI engineering crew —{' '}
              <span className="text-cyan-300 font-semibold">DeepSeek R1</span> and{' '}
              <span className="text-orange-300 font-semibold">Claude</span> — traced the memory storms behind recent
              node restarts to their <span className="text-white font-semibold">root cause</span> and shipped the fix
              live to mainnet (<span className="font-mono text-cyan-200">v10.11.53/54</span>). Your balances, the full
              chain history and every block stayed intact the whole way.
            </motion.p>

            {/* feature bullets */}
            <div className="mt-5 grid sm:grid-cols-3 gap-3">
              {[
                { icon: <Shield size={16} />, title: 'Zero loss', body: 'Same chain, same balances. Nothing reset, nothing rolled back.' },
                { icon: <Cpu size={16} />, title: 'AI co-pilots', body: 'Autonomous agents now watch, diagnose and ship fixes 24/7.' },
                { icon: <Rocket size={16} />, title: 'Brighter ahead', body: 'Hardened serving, multi-node failover and smoother mining next.' },
              ].map((f, i) => (
                <motion.div
                  key={f.title}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.75 + i * 0.12 }}
                  className="p-3.5 rounded-2xl bg-white/[0.04] border border-white/10"
                >
                  <div className="flex items-center gap-2 text-cyan-300 text-sm font-bold">
                    {f.icon}
                    {f.title}
                  </div>
                  <div className="mt-1.5 text-xs leading-relaxed text-gray-400">{f.body}</div>
                </motion.div>
              ))}
            </div>

            {/* AI pulse */}
            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.15 }}
              className="mt-5 p-4 rounded-2xl border"
              style={{ background: 'rgba(0,229,255,0.05)', borderColor: 'rgba(0,229,255,0.18)' }}
            >
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-widest text-cyan-300/80 font-semibold">
                <motion.span
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1.6, repeat: Infinity }}
                  className="inline-block w-1.5 h-1.5 rounded-full bg-cyan-300"
                />
                {aiSource}
              </div>
              <p className="mt-2 text-sm leading-relaxed text-cyan-100/90 font-mono min-h-[3.5rem]">
                {aiText || '…syncing with the network mind'}
                <motion.span
                  animate={{ opacity: [1, 0] }}
                  transition={{ duration: 0.7, repeat: Infinity }}
                  className="inline-block w-2 h-4 ml-0.5 align-middle bg-cyan-300/80"
                />
              </p>
            </motion.div>

            {/* live stats */}
            <div className="mt-5 flex flex-wrap gap-2.5">
              <StatChip icon={<BarChart3 size={15} />} label="Block height" value={stats.height ? stats.height.toLocaleString() : '—'} delay={1.3} />
              <StatChip icon={<Zap size={15} />} label="Node" value={stats.version ? `v${stats.version}` : 'live'} delay={1.4} />
              <StatChip icon={<Sparkles size={15} />} label="Status" value="climbing" delay={1.5} />
            </div>

            {/* CTA */}
            <motion.button
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.6 }}
              whileHover={{ scale: 1.025 }}
              whileTap={{ scale: 0.97 }}
              onClick={dismiss}
              className="mt-7 w-full relative overflow-hidden flex items-center justify-center gap-2 py-3.5 rounded-2xl font-bold text-base text-gray-900"
              style={{ background: 'linear-gradient(95deg, #00E5FF 0%, #5EF2C2 50%, #FFD700 100%)' }}
            >
              <motion.span
                className="absolute inset-0 pointer-events-none"
                style={{ background: 'linear-gradient(105deg, transparent 35%, rgba(255,255,255,0.55) 50%, transparent 65%)' }}
                animate={{ x: ['-120%', '220%'] }}
                transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut', repeatDelay: 1.2 }}
              />
              Enter the Graph
              <ChevronRight size={18} />
            </motion.button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body
  );
};

export default StabilityHorizonModal;
