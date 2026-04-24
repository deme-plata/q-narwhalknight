import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Trophy, Bug, Globe, Users, Pickaxe, Zap, X, Shield, Target, Flame,
  Gift, CheckCircle, Loader, ArrowRight, Star, Twitter, MessageCircle, Github,
  ExternalLink, Maximize2, Minimize2,
} from 'lucide-react';

// ─── Bounty Site Iframe Modal ────────────────────────────────────────────────
export const BountySiteModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [loaded, setLoaded] = useState(false);
  const [expanded, setExpanded] = useState(false);

  return createPortal(
    <AnimatePresence>
      <motion.div
        key="bounty-site-overlay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-[10002] flex items-center justify-center"
        style={{ background: 'rgba(2,4,12,0.88)', backdropFilter: 'blur(18px)' }}
        onClick={e => e.target === e.currentTarget && onClose()}
      >
        {/* Animated border container */}
        <motion.div
          initial={{ scale: 0.88, opacity: 0, y: 40 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.92, opacity: 0, y: 20 }}
          transition={{ type: 'spring', damping: 22, stiffness: 260 }}
          className="relative"
          style={{
            width: expanded ? '98vw' : 'min(920px, 92vw)',
            height: expanded ? '96vh' : 'min(680px, 88vh)',
            borderRadius: 22,
            padding: 2,
            background: 'linear-gradient(135deg, #10B981, #06B6D4, #8B5CF6, #F43F5E, #10B981)',
            backgroundSize: '300% 300%',
            animation: 'bountyBorderSpin 4s linear infinite',
            boxShadow: '0 0 60px rgba(16,185,129,0.35), 0 0 120px rgba(139,92,246,0.2), 0 30px 80px rgba(0,0,0,0.7)',
          }}
        >
          <style>{`
            @keyframes bountyBorderSpin {
              0% { background-position: 0% 50%; }
              50% { background-position: 100% 50%; }
              100% { background-position: 0% 50%; }
            }
          `}</style>

          {/* Inner shell */}
          <div
            className="relative w-full h-full flex flex-col overflow-hidden"
            style={{ borderRadius: 20, background: 'linear-gradient(180deg, #060c18 0%, #0a1428 100%)' }}
          >
            {/* Chrome bar */}
            <div
              className="flex items-center gap-3 px-4 py-3 flex-shrink-0"
              style={{
                background: 'linear-gradient(90deg, rgba(16,185,129,0.1), rgba(139,92,246,0.1))',
                borderBottom: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              {/* Traffic lights */}
              <div className="flex gap-1.5">
                <button onClick={onClose} className="w-3 h-3 rounded-full bg-red-500 hover:bg-red-400 transition-colors" title="Close" />
                <div className="w-3 h-3 rounded-full bg-yellow-500 opacity-60" />
                <button onClick={() => setExpanded(e => !e)} className="w-3 h-3 rounded-full bg-green-500 hover:bg-green-400 transition-colors" title="Expand" />
              </div>

              {/* URL bar */}
              <div
                className="flex-1 flex items-center gap-2 px-3 py-1 rounded-lg text-xs"
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)' }}
              >
                <Shield className="w-3 h-3 text-emerald-400 flex-shrink-0" />
                <span className="text-gray-300 font-mono truncate">https://bounty.quillon.xyz</span>
              </div>

              {/* Controls */}
              <div className="flex items-center gap-1">
                <motion.button
                  whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.92 }}
                  onClick={() => setExpanded(e => !e)}
                  className="p-1.5 rounded-lg text-gray-400 hover:text-white transition-colors"
                  style={{ background: 'rgba(255,255,255,0.05)' }}
                >
                  {expanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
                </motion.button>
                <motion.a
                  href="https://bounty.quillon.xyz"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.92 }}
                  className="p-1.5 rounded-lg text-gray-400 hover:text-white transition-colors"
                  style={{ background: 'rgba(255,255,255,0.05)' }}
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                </motion.a>
                <motion.button
                  whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.92 }}
                  onClick={onClose}
                  className="p-1.5 rounded-lg text-gray-400 hover:text-red-400 transition-colors ml-1"
                  style={{ background: 'rgba(255,255,255,0.05)' }}
                >
                  <X className="w-3.5 h-3.5" />
                </motion.button>
              </div>
            </div>

            {/* Iframe + loader */}
            <div className="relative flex-1 overflow-hidden">
              {!loaded && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-4" style={{ background: '#060c18' }}>
                  {/* Animated logo */}
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                    className="w-14 h-14 rounded-full"
                    style={{
                      background: 'conic-gradient(from 0deg, #10B981, #06B6D4, #8B5CF6, #10B981)',
                      padding: 3,
                    }}
                  >
                    <div className="w-full h-full rounded-full flex items-center justify-center" style={{ background: '#060c18' }}>
                      <Trophy className="w-6 h-6 text-emerald-400" />
                    </div>
                  </motion.div>
                  <div className="text-center">
                    <p className="text-white font-bold text-sm">Loading Bounty Portal</p>
                    <p className="text-gray-500 text-xs mt-1">bounty.quillon.xyz</p>
                  </div>
                  {/* Shimmer bar */}
                  <div className="w-48 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: 'linear-gradient(90deg, #10B981, #06B6D4)' }}
                      animate={{ x: ['-100%', '200%'] }}
                      transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
                    />
                  </div>
                </div>
              )}
              <iframe
                src="https://bounty.quillon.xyz"
                className="w-full h-full border-0"
                style={{ display: loaded ? 'block' : 'none', colorScheme: 'dark' }}
                onLoad={() => setLoaded(true)}
                title="Quillon Bounty Portal"
                sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox"
              />
            </div>

            {/* Bottom glow bar */}
            <div
              className="h-0.5 flex-shrink-0"
              style={{ background: 'linear-gradient(90deg, #10B981, #06B6D4, #8B5CF6, #F43F5E, #10B981)', backgroundSize: '300% 100%', animation: 'bountyBorderSpin 3s linear infinite' }}
            />
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body
  );
};

const BOUNTY_REGISTER_URL = '/bounty-api/v1/testnet/register';

interface BountyModalProps {
  onClose: () => void;
  onStartEarning?: () => void;
  genieTargetRef?: React.RefObject<HTMLElement | null>;
}

// ─── Animated gradient background canvas ────────────────────────────────────
const BountyBg: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let w = canvas.offsetWidth, h = canvas.offsetHeight;
    canvas.width = w * 2; canvas.height = h * 2; ctx.scale(2, 2);
    const stars = Array.from({ length: 120 }, () => ({
      x: Math.random() * w, y: Math.random() * h, r: 0.3 + Math.random() * 1.1, s: 0.5 + Math.random() * 2,
    }));
    interface P { x: number; y: number; vx: number; vy: number; r: number; life: number; hue: number }
    const pts: P[] = [];
    const hues = [160, 170, 270, 25, 185, 330];
    let frame = 0;
    const loop = () => {
      frame++;
      ctx.clearRect(0, 0, w, h);
      for (const s of stars) {
        const t = 0.3 + 0.7 * Math.abs(Math.sin(frame * 0.014 * s.s + s.x * 0.05));
        ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${t * 0.55})`; ctx.fill();
      }
      if (frame % 7 === 0 && pts.length < 50) {
        pts.push({ x: Math.random() * w, y: Math.random() * h, vx: (Math.random() - 0.5) * 0.5, vy: (Math.random() - 0.5) * 0.4, r: 1.5 + Math.random() * 2.5, life: 1, hue: hues[Math.floor(Math.random() * hues.length)] });
      }
      for (let i = pts.length - 1; i >= 0; i--) {
        const p = pts[i]; p.x += p.vx; p.y += p.vy; p.life -= 0.003;
        if (p.life <= 0) { pts.splice(i, 1); continue; }
        const a = Math.min(p.life, 0.7);
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4);
        g.addColorStop(0, `hsla(${p.hue},80%,70%,${a * 0.5})`);
        g.addColorStop(1, `hsla(${p.hue},70%,40%,0)`);
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r * 4, 0, Math.PI * 2); ctx.fillStyle = g; ctx.fill();
      }
      animRef.current = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(animRef.current);
  }, []);
  return <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', opacity: 0.55 }} />;
};

// ─── Reward pill ─────────────────────────────────────────────────────────────
const Pill: React.FC<{ icon: React.ReactNode; label: string; pts: string; color: string }> = ({ icon, label, pts, color }) => (
  <div className="flex items-center gap-2 p-2 rounded-lg" style={{ background: `${color}0e`, border: `1px solid ${color}22` }}>
    <div className="flex-shrink-0" style={{ color }}>{icon}</div>
    <div className="flex-1 min-w-0">
      <p className="text-xs font-semibold text-white leading-tight">{label}</p>
      <p className="text-[10px] leading-tight" style={{ color: `${color}bb` }}>{pts}</p>
    </div>
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════
export default function BountyModal({ onClose, onStartEarning }: BountyModalProps) {
  const walletAddress = localStorage.getItem('walletAddress') || '';
  const [address, setAddress] = useState(walletAddress);
  const [twitter, setTwitter] = useState('');
  const [discord, setDiscord] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');
  const [userId, setUserId] = useState('');

  const handleJoin = async () => {
    if (!address.trim()) { setErrorMsg('Please enter your wallet address.'); return; }
    setStatus('loading'); setErrorMsg('');
    try {
      const res = await fetch(BOUNTY_REGISTER_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ testnet_address: address.trim() }),
      });
      if (res.ok) {
        const data = await res.json();
        setUserId(data.user_id || address.slice(0, 12) + '…');
        setStatus('success');
        localStorage.setItem('bounty_registered', 'true');
        localStorage.setItem('bounty_user_id', data.user_id || '');
      } else {
        // Server returned an error — likely already registered, treat as success
        const data = await res.json().catch(() => ({}));
        if (res.status === 409 || data.message?.toLowerCase().includes('already')) {
          setUserId(data.user_id || address.slice(0, 12) + '…');
          setStatus('success');
        } else {
          setErrorMsg(data.error || data.message || `Server error ${res.status}`);
          setStatus('error');
        }
      }
    } catch {
      // Network error — save locally and show success anyway
      localStorage.setItem('bounty_registered', 'true');
      localStorage.setItem('bounty_pending_address', address.trim());
      setUserId(address.slice(0, 12) + '…');
      setStatus('success');
    }
  };

  const modal = (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 99999,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'radial-gradient(ellipse at 30% 20%, rgba(16,185,129,0.07) 0%, transparent 55%), radial-gradient(ellipse at 70% 80%, rgba(139,92,246,0.06) 0%, transparent 55%), rgba(0,0,0,0.88)',
        backdropFilter: 'blur(20px)',
      }}
    >
      <BountyBg />
      <motion.div
        initial={{ scale: 0.88, opacity: 0, y: 24 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.88, opacity: 0, y: 24 }}
        transition={{ type: 'spring', damping: 22, stiffness: 280 }}
        onClick={e => e.stopPropagation()}
        style={{
          position: 'relative', width: '100%', maxWidth: 460, margin: '0 16px',
          borderRadius: 24, overflow: 'hidden', zIndex: 2,
        }}
      >
        {/* Rotating border */}
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 9, repeat: Infinity, ease: 'linear' }}
          style={{
            position: 'absolute', inset: -2, borderRadius: 26,
            background: 'conic-gradient(from 0deg, #10B981, #06B6D4, #8B5CF6, #F43F5E, #F97316, #10B981)',
            opacity: 0.45, zIndex: 0,
          }}
        />

        <div style={{
          position: 'relative', zIndex: 1, borderRadius: 24,
          background: 'linear-gradient(160deg, rgba(7,11,18,0.98) 0%, rgba(12,18,30,0.97) 100%)',
          padding: '24px 22px 22px',
        }}>
          {/* Close */}
          <button onClick={onClose} className="absolute top-4 right-4 p-1.5 rounded-full hover:scale-110 transition-all" style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', zIndex: 10 }}>
            <X className="w-4 h-4 text-gray-400" />
          </button>

          <AnimatePresence mode="wait">
            {status === 'success' ? (
              // ── Success ───────────────────────────────────────────────────
              <motion.div key="success" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="text-center py-4">
                <motion.div animate={{ scale: [1, 1.15, 1] }} transition={{ duration: 1.5, repeat: Infinity }} className="w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #10B981, #06B6D4)', boxShadow: '0 0 30px rgba(16,185,129,0.35)' }}>
                  <CheckCircle className="w-8 h-8 text-white" />
                </motion.div>
                <h2 className="text-xl font-bold mb-1" style={{ background: 'linear-gradient(135deg, #10B981, #22D3EE)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                  You're In!
                </h2>
                <p className="text-sm text-gray-400 mb-4">Start earning points immediately.</p>
                <div className="p-3 rounded-xl mb-5 text-left" style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)' }}>
                  <p className="text-xs text-gray-500 mb-0.5">Registered as</p>
                  <p className="text-sm font-mono text-emerald-400 break-all">{userId}</p>
                </div>
                <div className="grid grid-cols-3 gap-2 mb-5">
                  {[
                    { icon: <Pickaxe className="w-4 h-4" />, label: 'Run a node', pts: '500 pts/day', color: '#10B981' },
                    { icon: <Bug className="w-4 h-4" />, label: 'Report bugs', pts: '10–100 pts', color: '#F43F5E' },
                    { icon: <Globe className="w-4 h-4" />, label: 'Create content', pts: '10–200 pts', color: '#F97316' },
                  ].map(c => (
                    <div key={c.label} className="p-2 rounded-lg text-center" style={{ background: `${c.color}0d`, border: `1px solid ${c.color}20` }}>
                      <div className="flex justify-center mb-1" style={{ color: c.color }}>{c.icon}</div>
                      <p className="text-[10px] text-white font-medium">{c.label}</p>
                      <p className="text-[10px]" style={{ color: `${c.color}aa` }}>{c.pts}</p>
                    </div>
                  ))}
                </div>
                <motion.button
                  onClick={() => { onClose(); onStartEarning?.(); }}
                  whileHover={{ scale: 1.04 }}
                  whileTap={{ scale: 0.97 }}
                  className="w-full py-2.5 rounded-xl font-bold text-sm text-white flex items-center justify-center gap-2"
                  style={{ background: 'linear-gradient(135deg, #10B981, #06B6D4)', boxShadow: '0 4px 18px rgba(16,185,129,0.3)' }}
                >
                  <Trophy className="w-4 h-4" />
                  Start Earning →
                </motion.button>
              </motion.div>
            ) : (
              // ── Join Form ─────────────────────────────────────────────────
              <motion.div key="form" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                {/* Header */}
                <div className="text-center mb-5">
                  <motion.div animate={{ rotate: [0, -8, 8, -4, 0] }} transition={{ duration: 2.5, repeat: Infinity, repeatDelay: 3 }} className="inline-flex mb-3">
                    <Trophy className="w-9 h-9 text-emerald-400" style={{ filter: 'drop-shadow(0 0 12px rgba(16,185,129,0.6))' }} />
                  </motion.div>
                  <h2 className="text-xl font-bold leading-tight" style={{ background: 'linear-gradient(135deg, #10B981, #22D3EE, #A78BFA)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                    Bounty Campaign
                  </h2>
                  <p className="text-xs text-gray-400 mt-1">Earn QUG for running nodes, finding bugs &amp; growing the community</p>
                </div>

                {/* Reward pills */}
                <div className="grid grid-cols-2 gap-2 mb-5">
                  <Pill icon={<Pickaxe className="w-3.5 h-3.5" />} label="Node Operations" pts="Up to 500 pts/day" color="#10B981" />
                  <Pill icon={<Zap className="w-3.5 h-3.5" />} label="Transactions" pts="Up to 200 pts/day" color="#06B6D4" />
                  <Pill icon={<Bug className="w-3.5 h-3.5" />} label="Bug Reports" pts="10–100 pts each" color="#F43F5E" />
                  <Pill icon={<Globe className="w-3.5 h-3.5" />} label="Social & Content" pts="10–200 pts each" color="#F97316" />
                </div>

                {/* Multiplier badge */}
                <div className="flex items-center justify-center gap-3 mb-5">
                  <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs" style={{ background: 'rgba(249,115,22,0.1)', border: '1px solid rgba(249,115,22,0.25)', color: '#F97316' }}>
                    <Flame className="w-3.5 h-3.5" />
                    <span>2× Early Bird bonus active</span>
                  </div>
                  <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs" style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.25)', color: '#10B981' }}>
                    <Gift className="w-3.5 h-3.5" />
                    <span>$306 daily pool</span>
                  </div>
                </div>

                {/* Registration form */}
                <div className="space-y-3 mb-4">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1.5 font-medium">Your Quillon Wallet Address <span className="text-emerald-400">*</span></label>
                    <input
                      value={address}
                      onChange={e => setAddress(e.target.value)}
                      placeholder="qnk..."
                      spellCheck={false}
                      className="w-full px-3 py-2.5 rounded-xl text-sm font-mono text-white placeholder-gray-600 outline-none transition-all"
                      style={{ background: 'rgba(255,255,255,0.05)', border: `1px solid ${address ? 'rgba(16,185,129,0.4)' : 'rgba(255,255,255,0.1)'}`, boxShadow: address ? '0 0 0 3px rgba(16,185,129,0.08)' : 'none' }}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-xs text-gray-500 mb-1 flex items-center gap-1"><Twitter className="w-3 h-3 text-sky-400" /> Twitter (optional)</label>
                      <input value={twitter} onChange={e => setTwitter(e.target.value)} placeholder="@handle" className="w-full px-3 py-2 rounded-xl text-sm text-white placeholder-gray-600 outline-none" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }} />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 mb-1 flex items-center gap-1"><MessageCircle className="w-3 h-3 text-indigo-400" /> Discord (optional)</label>
                      <input value={discord} onChange={e => setDiscord(e.target.value)} placeholder="user#0000" className="w-full px-3 py-2 rounded-xl text-sm text-white placeholder-gray-600 outline-none" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }} />
                    </div>
                  </div>
                </div>

                {errorMsg && (
                  <p className="text-xs text-red-400 mb-3 px-1">{errorMsg}</p>
                )}

                <motion.button
                  onClick={handleJoin}
                  disabled={status === 'loading'}
                  whileHover={{ scale: 1.03, y: -1 }}
                  whileTap={{ scale: 0.97 }}
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-2xl font-bold text-sm text-white disabled:opacity-60"
                  style={{ background: 'linear-gradient(135deg, #10B981, #06B6D4, #8B5CF6)', boxShadow: '0 4px 20px rgba(16,185,129,0.3)' }}
                >
                  {status === 'loading' ? (
                    <><Loader className="w-4 h-4 animate-spin" /> Registering…</>
                  ) : (
                    <><Trophy className="w-4 h-4" /> Join Bounty Campaign <ArrowRight className="w-4 h-4" /></>
                  )}
                </motion.button>

                <p className="text-[10px] text-gray-600 text-center mt-3 flex items-center justify-center gap-1">
                  <Shield className="w-3 h-3 text-emerald-600" />
                  No private keys — only your public wallet address is stored
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </motion.div>
  );

  return createPortal(modal, document.body);
}
