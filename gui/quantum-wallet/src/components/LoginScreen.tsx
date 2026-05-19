import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Key, AlertCircle, Search, HelpCircle, X, Shield, Zap, Lock, Globe, Pickaxe, Download, Monitor, Laptop, Terminal as TerminalIcon, Blocks, Activity, Cpu, Users, Clock, ChevronDown, Hash, TrendingUp, Wallet, BookOpen } from 'lucide-react';
import { qnkAPI } from '../services/api';
import { storeWallet, walletSession, verifyPasswordHash, hasPasswordHash } from '../services/walletAuth';
import ExplorerSearchBar from './ExplorerSearchBar';
import PapersLibraryModal from './PapersLibraryModal';

interface LoginScreenProps {
  onAuthenticate: () => void;
}

// --- Quantum Field Background ---
// Inspired by the theoretical physics whitepaper: String-Theoretic Resonance Consensus
// Particles implement ψ(x,t) = A · e^(i(kx - ωt + φ)) · sin(nπx/L)
// Powered by real-time production API data from /api/v1/health

interface StringParticle {
  x: number; y: number;
  amplitude: number;      // A = √(stake_weight) → particle size
  frequency: number;      // ω = 2π·priority → oscillation speed
  phase: number;          // φ ∈ [0, 2π) → color hue
  mode: number;           // n = harmonic mode → wave complexity
  vx: number; vy: number; // velocity in consensus space
  radius: number;
  life: number;           // 0..1 lifecycle
  type: 'honest' | 'resonance' | 'entangled' | 'finalized';
}

interface NetworkPulse {
  x: number; y: number;
  radius: number;
  maxRadius: number;
  alpha: number;
  color: string;
}

function QuantumFieldBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const networkDataRef = useRef({ height: 0, peers: 0, uptime: 0, status: 'starting' });
  const prevHeightRef = useRef(0);

  // Fetch real-time network data
  useEffect(() => {
    const fetchNetworkData = async () => {
      try {
        const res = await fetch('/api/v1/health');
        if (res.ok) {
          const json = await res.json();
          const d = json?.data;
          if (d) {
            const prevHeight = networkDataRef.current.height;
            networkDataRef.current = {
              height: d.height || 0,
              peers: d.peers || 0,
              uptime: d.uptime_secs || 0,
              status: d.status || 'starting',
            };
            prevHeightRef.current = prevHeight;
          }
        }
      } catch { /* silent */ }
    };
    fetchNetworkData();
    const interval = setInterval(fetchNetworkData, 4000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let w = canvas.width = window.innerWidth;
    let h = canvas.height = window.innerHeight;

    const handleResize = () => {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    // Particle system
    const particles: StringParticle[] = [];
    const pulses: NetworkPulse[] = [];
    const bgStars: Array<{ x: number; y: number; r: number; a: number; s: number }> = [];

    // Initialize background stars
    for (let i = 0; i < 300; i++) {
      bgStars.push({
        x: Math.random() * w, y: Math.random() * h,
        r: Math.random() * 1.2 + 0.2,
        a: Math.random() * 0.6 + 0.2,
        s: Math.random() * 0.015 + 0.003,
      });
    }

    // Spawn a new string particle with physics properties
    const spawnParticle = (type: StringParticle['type'] = 'honest') => {
      const nd = networkDataRef.current;
      const stakeWeight = 0.3 + Math.random() * 0.7;
      const priority = Math.random();
      const colors: Record<string, number[]> = {
        honest: [0.55, 0.65],      // blue range
        resonance: [0.08, 0.15],   // gold range
        entangled: [0.48, 0.52],   // cyan range
        finalized: [0.75, 0.85],   // purple range
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

      // Emit pulse on block height change
      if (nd.height > prevHeightRef.current && Math.random() < 0.3) {
        pulses.push({
          x: Math.random() * w, y: Math.random() * h,
          radius: 0, maxRadius: 80 + Math.random() * 120,
          alpha: 0.3, color: type === 'finalized' ? '#9455F7' : '#D4AF37',
        });
      }
    };

    // Seed initial particles
    for (let i = 0; i < 60; i++) {
      const types: StringParticle['type'][] = ['honest', 'resonance', 'entangled', 'finalized'];
      spawnParticle(types[Math.floor(Math.random() * types.length)]);
      particles[particles.length - 1].life = Math.random(); // stagger lifecycle
    }

    let frame = 0;
    let lastSpawn = 0;

    const draw = () => {
      frame++;
      const t = frame * 0.016; // ~60fps time in seconds
      const nd = networkDataRef.current;

      // Clear canvas - transparent so background image shows through
      ctx.clearRect(0, 0, w, h);

      // Nebula clouds (subtle, atmospheric)
      const nebulaPositions = [
        { x: w * 0.2, y: h * 0.3, rx: 300, ry: 200, c: 'rgba(59,130,246,0.06)' },
        { x: w * 0.7, y: h * 0.6, rx: 350, ry: 250, c: 'rgba(139,92,246,0.05)' },
        { x: w * 0.5, y: h * 0.15, rx: 280, ry: 180, c: 'rgba(212,175,55,0.04)' },
        { x: w * 0.85, y: h * 0.2, rx: 200, ry: 160, c: 'rgba(6,182,212,0.05)' },
        { x: w * 0.15, y: h * 0.75, rx: 260, ry: 200, c: 'rgba(148,85,247,0.045)' },
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

      // Background stars with twinkling - brighter for visibility over image
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

      // Spawn particles based on network activity
      if (frame - lastSpawn > 8) {
        lastSpawn = frame;
        const types: StringParticle['type'][] = ['honest', 'resonance', 'entangled', 'finalized'];
        const weights = [0.4, 0.25, 0.2, 0.15];
        let r = Math.random();
        let chosen: StringParticle['type'] = 'honest';
        for (let i = 0; i < types.length; i++) {
          r -= weights[i];
          if (r <= 0) { chosen = types[i]; break; }
        }
        if (particles.length < 120) spawnParticle(chosen);
      }

      // Physics: compute resonance coupling and draw connections
      const coupledPairs: Array<[number, number, number]> = [];
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const pi = particles[i], pj = particles[j];
          const dx = pi.x - pj.x, dy = pi.y - pj.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > 150) continue;
          // R_ij = A_i · A_j · cos²(Δφ/2) · e^(-Δω²/2)
          const phaseDiff = pi.phase - pj.phase;
          const freqDiff = Math.abs(pi.frequency - pj.frequency);
          const resonance = pi.amplitude * pj.amplitude *
            Math.pow(Math.cos(phaseDiff / 2), 2) *
            Math.exp(-freqDiff * freqDiff / 2);
          if (resonance > 1.5) {
            coupledPairs.push([i, j, resonance]);
          }
        }
      }

      // Draw resonance coupling lines
      for (const [i, j, R] of coupledPairs) {
        const pi = particles[i], pj = particles[j];
        const lineAlpha = Math.min(0.4, R * 0.07) * Math.min(pi.life, pj.life);
        const hue = ((pi.phase + pj.phase) / 2 / (Math.PI * 2)) * 360;
        ctx.strokeStyle = `hsla(${hue}, 75%, 70%, ${lineAlpha})`;
        ctx.lineWidth = 0.7 + R * 0.15;
        ctx.beginPath();
        ctx.moveTo(pi.x, pi.y);
        // Slight curve for elegance
        const mx = (pi.x + pj.x) / 2 + Math.sin(t + i) * 8;
        const my = (pi.y + pj.y) / 2 + Math.cos(t + j) * 8;
        ctx.quadraticCurveTo(mx, my, pj.x, pj.y);
        ctx.stroke();
      }

      // Update and draw particles
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];

        // Wavefunction modulation: ψ(x,t) = A · sin(ωt + φ) · sin(nπx/L)
        const psi = p.amplitude * Math.sin(p.frequency * t + p.phase) *
          Math.sin((p.mode * Math.PI * p.x) / w);

        // Update position with velocity + wavefunction perturbation
        p.x += p.vx + psi * 0.15;
        p.y += p.vy + Math.cos(p.frequency * t * 0.7 + p.phase) * 0.12;

        // Wrap around edges
        if (p.x < -20) p.x = w + 20;
        if (p.x > w + 20) p.x = -20;
        if (p.y < -20) p.y = h + 20;
        if (p.y > h + 20) p.y = -20;

        // Lifecycle decay
        p.life -= 0.0008;
        if (p.life <= 0) { particles.splice(i, 1); continue; }

        const alpha = Math.min(1, p.life * 3) * Math.min(1, (1 - p.life) * 5);
        const hue = (p.phase / (Math.PI * 2)) * 360;
        const pulseR = p.radius * (1 + Math.sin(p.frequency * t + p.phase) * 0.25);

        // Outer glow - boosted for visibility over background image
        const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, pulseR * 6);
        glow.addColorStop(0, `hsla(${hue}, 85%, 75%, ${alpha * 0.45})`);
        glow.addColorStop(0.3, `hsla(${hue}, 75%, 60%, ${alpha * 0.15})`);
        glow.addColorStop(0.6, `hsla(${hue}, 70%, 55%, ${alpha * 0.04})`);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(p.x, p.y, pulseR * 6, 0, Math.PI * 2);
        ctx.fill();

        // Core particle - brighter and more vivid
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

        // Cross-spike for finalized particles (high amplitude)
        if (p.type === 'finalized' && alpha > 0.4) {
          const sLen = pulseR * 2.5;
          ctx.strokeStyle = `hsla(${hue}, 80%, 75%, ${alpha * 0.4})`;
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.moveTo(p.x - sLen, p.y);
          ctx.lineTo(p.x + sLen, p.y);
          ctx.moveTo(p.x, p.y - sLen);
          ctx.lineTo(p.x, p.y + sLen);
          ctx.stroke();
        }

        ctx.globalAlpha = 1;
      }

      // Draw network pulses (block confirmations)
      for (let i = pulses.length - 1; i >= 0; i--) {
        const pulse = pulses[i];
        pulse.radius += 1.5;
        pulse.alpha -= 0.004;
        if (pulse.alpha <= 0 || pulse.radius > pulse.maxRadius) {
          pulses.splice(i, 1); continue;
        }
        ctx.strokeStyle = pulse.color.replace(')', `, ${pulse.alpha})`).replace('rgb', 'rgba');
        if (pulse.color.startsWith('#')) {
          const r = parseInt(pulse.color.slice(1, 3), 16);
          const g = parseInt(pulse.color.slice(3, 5), 16);
          const b = parseInt(pulse.color.slice(5, 7), 16);
          ctx.strokeStyle = `rgba(${r},${g},${b},${pulse.alpha})`;
        }
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(pulse.x, pulse.y, pulse.radius, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Network data overlay - subtle standing wave at bottom
      if (nd.height > 0) {
        ctx.globalAlpha = 0.04;
        ctx.strokeStyle = '#D4AF37';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x < w; x += 2) {
          const wave = Math.sin((x / w) * Math.PI * (nd.peers + 2)) *
            Math.cos(t * 0.5) * 20 + h - 40;
          if (x === 0) ctx.moveTo(x, wave);
          else ctx.lineTo(x, wave);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      cancelAnimationFrame(animationRef.current);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <>
      {/* Background image layer - coastal sunset with rugged cliffs */}
      <div
        className="fixed inset-0 w-full h-full"
        style={{
          zIndex: 0,
          backgroundImage: 'url(/login-bg.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      />
      {/* Dark overlay to let particles pop over the image */}
      <div
        className="fixed inset-0 w-full h-full"
        style={{
          zIndex: 0,
          background: 'rgba(0, 0, 0, 0.35)',
        }}
      />
      {/* Particle canvas overlay - ON TOP of the background image, screen-blended for glow */}
      <canvas
        ref={canvasRef}
        className="fixed inset-0 w-full h-full pointer-events-none"
        style={{ zIndex: 1, opacity: 0.9, mixBlendMode: 'screen' }}
      />
    </>
  );
}

// --- Floating particles around the logo ---
function FloatingParticles() {
  const particles = Array.from({ length: 20 }, (_, i) => ({
    id: i,
    delay: Math.random() * 5,
    duration: Math.random() * 4 + 3,
    x: Math.random() * 160 - 80,
    y: Math.random() * 160 - 80,
    size: Math.random() * 3 + 1,
  }));

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map((p) => (
        <motion.div
          key={p.id}
          className="absolute rounded-full"
          style={{
            width: p.size,
            height: p.size,
            left: '50%',
            top: '50%',
            background: `radial-gradient(circle, rgba(255, 215, 0, 0.8), rgba(212, 175, 55, 0))`,
          }}
          animate={{
            x: [0, p.x, -p.x * 0.5, p.x * 0.3, 0],
            y: [0, p.y, -p.y * 0.7, p.y * 0.5, 0],
            opacity: [0, 0.8, 0.4, 0.7, 0],
            scale: [0, 1.5, 0.8, 1.2, 0],
          }}
          transition={{
            duration: p.duration,
            delay: p.delay,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
}


export default function LoginScreen({ onAuthenticate }: LoginScreenProps) {
  const [seedPhrase, setSeedPhrase] = useState('');
  const [password, setPassword] = useState('');
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [showQuantumGenerator, setShowQuantumGenerator] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showInfoModal, setShowInfoModal] = useState(false);
  const [showMinerModal, setShowMinerModal] = useState(false);
  const [showNodeModal, setShowNodeModal] = useState(false);
  const [showSlintModal, setShowSlintModal] = useState(false);
  const [showAIModal, setShowAIModal] = useState(false);
  const [aiCopied, setAiCopied] = useState(false);
  const [showPapersLibrary, setShowPapersLibrary] = useState(false);
  const [isMetaMaskConnecting, setIsMetaMaskConnecting] = useState(false);
  const [hasMetaMask, setHasMetaMask] = useState(false);
  // Persisted Tor onion address - fetched from backend, fallback to hardcoded
  const [torOnionUrl, setTorOnionUrl] = useState("http://ca3jpub2haxboxjw4ws6run36ekdh3pv7pneqg2tbac5rxzvxhd2i5id.onion");

  // Explorer dropdown state - live data from API + SSE
  const [showExplorerDropdown, setShowExplorerDropdown] = useState(false);
  const [menuHovered, setMenuHovered] = useState(false);
  const [explorerData, setExplorerData] = useState<{
    blocks: any[];
    transactions: any[];
    health: any;
    networkStats: any;
  }>({ blocks: [], transactions: [], health: null, networkStats: null });
  const [explorerLoading, setExplorerLoading] = useState(false);
  const [liveBlockHeight, setLiveBlockHeight] = useState(0);
  const [livePeers, setLivePeers] = useState(0);
  const [blockPulse, setBlockPulse] = useState(false);
  const [qugPrice, setQugPrice] = useState(0);
  const [qugMarketCap, setQugMarketCap] = useState(0);
  const [minerCount, setMinerCount] = useState(0);
  const [networkHashrate, setNetworkHashrate] = useState('');
  const explorerTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const explorerContainerRef = useRef<HTMLDivElement>(null);

  // SSE connection for real-time block/node-status updates (works without auth)
  useEffect(() => {
    if (!showExplorerDropdown) return;
    let es: EventSource | null = null;
    try {
      es = new EventSource('/api/v1/events');
      es.addEventListener('node-status', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          if (data.current_height && data.current_height > liveBlockHeight) {
            setLiveBlockHeight(data.current_height);
            setBlockPulse(true);
            setTimeout(() => setBlockPulse(false), 600);
          }
          if (data.connected_peers !== undefined) setLivePeers(data.connected_peers);
        } catch { /* ignore parse errors */ }
      });
      es.addEventListener('block-mined', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          if (data.height) {
            // Prepend new block to the list
            setExplorerData(prev => ({
              ...prev,
              blocks: [{ height: data.height, tx_count: data.tx_count || 0, timestamp: Math.floor(Date.now() / 1000), dag_round: data.dag_round || 0 }, ...prev.blocks].slice(0, 5),
            }));
            setBlockPulse(true);
            setTimeout(() => setBlockPulse(false), 600);
          }
        } catch { /* ignore */ }
      });
      es.onerror = () => { /* SSE reconnects automatically */ };
    } catch { /* SSE not supported - fallback to polling */ }
    return () => { es?.close(); };
  }, [showExplorerDropdown]);

  // Initial data fetch + periodic refresh via HTTP (fallback)
  useEffect(() => {
    if (!showExplorerDropdown) return;
    let cancelled = false;
    const fetchExplorerData = async () => {
      setExplorerLoading(true);
      try {
        const [blocksRes, txRes, healthRes, statsRes, priceRes, emissionRes, supplyRes] = await Promise.allSettled([
          fetch('/api/v1/blocks/recent?limit=5'),
          fetch('/api/v1/transactions/explorer?limit=6'),
          fetch('/api/v1/health'),
          fetch('/api/v1/statistics/network'),
          fetch('/api/v1/oracle/price/QUG'),
          fetch('/api/v1/emission/stats'),
          fetch('/api/v1/network/supply'),
        ]);
        if (cancelled) return;
        const blocks = blocksRes.status === 'fulfilled' && blocksRes.value.ok
          ? (await blocksRes.value.json())?.data || [] : [];
        const transactions = txRes.status === 'fulfilled' && txRes.value.ok
          ? (await txRes.value.json())?.data || [] : [];
        const health = healthRes.status === 'fulfilled' && healthRes.value.ok
          ? (await healthRes.value.json())?.data || null : null;
        const networkStats = statsRes.status === 'fulfilled' && statsRes.value.ok
          ? (await statsRes.value.json())?.data || null : null;
        // QUG price from oracle
        if (priceRes.status === 'fulfilled' && priceRes.value.ok) {
          const priceData = (await priceRes.value.json())?.data;
          if (priceData?.price_usd) setQugPrice(priceData.price_usd);
        }
        // Circulating supply from emission stats → market cap
        if (emissionRes.status === 'fulfilled' && emissionRes.value.ok) {
          const emData = (await emissionRes.value.json())?.data;
          const history = emData?.daily_history;
          if (Array.isArray(history) && history.length > 0) {
            const latest = history[history.length - 1];
            const supply = latest?.cumulative_supply_qug || 0;
            if (supply > 0) {
              const price = qugPrice || 3000;
              setQugMarketCap(supply * price);
            }
          }
        }
        // Miner count and hashrate from supply endpoint
        if (supplyRes.status === 'fulfilled' && supplyRes.value.ok) {
          const supplyData = (await supplyRes.value.json())?.data;
          if (supplyData?.connected_miners) setMinerCount(supplyData.connected_miners);
          if (supplyData?.network_hashrate_formatted) setNetworkHashrate(supplyData.network_hashrate_formatted);
          // Also use supply data for market cap if available
          if (supplyData?.total_mined && supplyData.total_mined > 0) {
            const price = qugPrice || 3000;
            setQugMarketCap(supplyData.total_mined * price);
          }
        }
        setExplorerData({ blocks, transactions, health, networkStats });
        if (health?.height) setLiveBlockHeight(health.height);
        if (health?.peers) setLivePeers(health.peers);
      } catch { /* silent */ }
      setExplorerLoading(false);
    };
    fetchExplorerData();
    const interval = setInterval(fetchExplorerData, 12000); // Less frequent since SSE handles live updates
    return () => { cancelled = true; clearInterval(interval); };
  }, [showExplorerDropdown]);

  // Click-outside closes the explorer dropdown
  useEffect(() => {
    if (!showExplorerDropdown) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (explorerContainerRef.current && !explorerContainerRef.current.contains(e.target as Node)) {
        setShowExplorerDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showExplorerDropdown]);

  const formatTimeAgo = (timestamp: number) => {
    const seconds = Math.floor(Date.now() / 1000 - timestamp);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  const formatUptime = (secs: number) => {
    const d = Math.floor(secs / 86400);
    const h = Math.floor((secs % 86400) / 3600);
    const m = Math.floor((secs % 3600) / 60);
    if (d > 0) return `${d}d ${h}h`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
  };

  // Track Tor service status
  const [torActive, setTorActive] = useState(false);

  // Check Tor status - onion address is configured in Nginx, always available
  useEffect(() => {
    const fetchTorStatus = async () => {
      try {
        const res = await fetch('/api/v1/tor/status');
        if (res.ok) {
          const data = await res.json();
          const torRunning = data?.data?.active || data?.data?.tor_enabled || false;
          // Onion service is configured in Nginx - always available when Tor infra is active
          setTorActive(torRunning);
          // Use backend onion_address if provided, otherwise keep hardcoded default
          const addr = data?.data?.onion_address;
          if (addr && typeof addr === 'string' && addr.length > 10) {
            const cleanAddr = addr.endsWith('.onion') ? addr : `${addr}.onion`;
            setTorOnionUrl(`http://${cleanAddr}`);
          }
        }
      } catch {
        // Even if API fails, onion service may still be accessible
        setTorActive(true);
      }
    };
    fetchTorStatus();
  }, []);

  // Open Tor version of the site
  const openTorSite = () => {
    window.open(torOnionUrl, '_blank', 'noopener,noreferrer');
  };

  const handleDownloadMiner = (platform: 'linux' | 'linux-arm64' | 'windows' | 'macos-intel' | 'macos-arm') => {
    if (platform === 'windows') {
      window.open('https://quillon.xyz/downloads/q-miner-windows-x64.exe', '_blank');
    } else if (platform === 'linux-arm64') {
      window.open('https://quillon.xyz/downloads/q-miner-linux-arm64', '_blank');
    } else if (platform === 'macos-intel') {
      window.open('https://quillon.xyz/downloads/q-miner-macos-x64', '_blank');
    } else if (platform === 'macos-arm') {
      window.open('https://quillon.xyz/downloads/q-miner-macos-arm64', '_blank');
    } else {
      window.open('https://quillon.xyz/downloads/q-miner-linux-x64', '_blank');
    }
  };

  // Validate seed phrase is a proper BIP39 mnemonic (12 or 24 words)
  const validateSeedPhrase = (phrase: string): { valid: boolean; error?: string } => {
    const words = phrase.trim().split(/\s+/).filter(w => w.length > 0);

    if (words.length === 0) {
      return { valid: false, error: 'Seed phrase is required' };
    }

    if (words.length !== 12 && words.length !== 24) {
      return { valid: false, error: `Seed phrase must be exactly 12 or 24 words (got ${words.length} words)` };
    }

    // Check each word is at least 3 characters (BIP39 words are 3-8 chars)
    const invalidWords = words.filter(w => w.length < 3 || w.length > 8);
    if (invalidWords.length > 0) {
      return { valid: false, error: `Invalid word length detected. BIP39 words are 3-8 characters.` };
    }

    return { valid: true };
  };

  // Find MetaMask provider (handles multiple wallet extensions)
  const getMetaMaskProvider = (): any => {
    const win = window as any;
    // EIP-6963: Modern provider discovery
    if (win.ethereum?.providers?.length) {
      return win.ethereum.providers.find((p: any) => p.isMetaMask) || null;
    }
    // Legacy: single provider
    if (win.ethereum?.isMetaMask) return win.ethereum;
    return null;
  };

  // MetaMask detection
  useEffect(() => {
    const checkMetaMask = () => {
      setHasMetaMask(!!getMetaMaskProvider());
    };
    checkMetaMask();
    // Check again after a brief delay (MetaMask may inject late)
    const timer = setTimeout(checkMetaMask, 1000);
    return () => clearTimeout(timer);
  }, []);

  // MetaMask sign-in handler: sign a deterministic message, derive wallet from signature
  const handleMetaMaskLogin = async () => {
    setIsMetaMaskConnecting(true);
    setGenerationError(null);

    try {
      const ethereum = getMetaMaskProvider();
      if (!ethereum) {
        throw new Error('MetaMask not detected. Please install the MetaMask browser extension.');
      }

      // 1. Request accounts (triggers MetaMask popup)
      const accounts: string[] = await ethereum.request({ method: 'eth_requestAccounts' });
      if (!accounts || accounts.length === 0) throw new Error('No accounts returned from MetaMask');
      const ethAddress = accounts[0];

      // 2. Sign a deterministic message to derive wallet entropy
      // This message is always the same for a given ETH address, so the same
      // MetaMask account always derives the same QNK wallet (deterministic)
      const message = `Q-NarwhalKnight Wallet Derivation\nAddress: ${ethAddress.toLowerCase()}\nChain: QNK Mainnet 2026.1`;
      const signature: string = await ethereum.request({
        method: 'personal_sign',
        params: [message, ethAddress],
      });

      // 3. Derive BIP39 mnemonic from the 65-byte signature
      // Use the signature bytes as entropy for a 12-word seed phrase
      const sigBytes = signature.startsWith('0x') ? signature.slice(2) : signature;
      // Take first 32 hex chars (16 bytes = 128 bits) for 12-word BIP39
      const entropyHex = sigBytes.slice(0, 32);

      // Convert hex entropy to a mnemonic using the wallet's BIP39 implementation
      const { entropyToMnemonic } = await import('../services/walletAuth');
      let mnemonic: string;
      try {
        mnemonic = entropyToMnemonic(entropyHex);
      } catch {
        // Fallback: use SHA-256 of signature for cleaner entropy
        const encoder = new TextEncoder();
        const hashBuffer = await crypto.subtle.digest('SHA-256', encoder.encode(signature));
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
        // 16 bytes = 128-bit entropy = 12 words
        const fallbackEntropy = hashHex.slice(0, 32);
        mnemonic = entropyToMnemonic(fallbackEntropy);
      }

      // 4. Set a password derived from the signature (so user doesn't need to type one)
      const pwBytes = sigBytes.slice(32, 64);
      const autoPassword = `mm_${pwBytes.slice(0, 16)}`;

      // v8.3.0: Persist auto-password in sessionStorage so sendTransaction()
      // can silently decrypt the mnemonic without showing a password prompt.
      // sessionStorage is cleared on browser close — same lifetime as the session.
      sessionStorage.setItem('metamaskAutoPassword', autoPassword);

      // 5. Auto-fill and authenticate
      setSeedPhrase(mnemonic);
      setPassword(autoPassword);

      // 6. Create wallet via API
      const response = await qnkAPI.createWallet(mnemonic, autoPassword);

      if (response.success && response.data) {
        localStorage.setItem('walletAddress', response.data.address_formatted || '');
        localStorage.setItem('walletId', response.data.id);
        localStorage.setItem('metamaskLinked', ethAddress.toLowerCase());
        localStorage.removeItem('cachedBalance');
        localStorage.removeItem('cachedQugusdBalance');
        localStorage.removeItem('walletBalanceHistory');

        const wallet = await storeWallet(mnemonic, autoPassword, true, true, true);
        walletSession.setSession(
          wallet.privateKey,
          wallet.address,
          mnemonic,
          wallet.dilithium5SecretKey,
          wallet.dilithium5PublicKey
        );

        await new Promise(resolve => setTimeout(resolve, 500));
        onAuthenticate();
      } else {
        throw new Error(response.error || 'Failed to create wallet');
      }
    } catch (error: any) {
      console.error('MetaMask login failed:', error);
      if (error?.code === 4001) {
        setGenerationError('MetaMask sign request was rejected. Please approve to continue.');
      } else {
        setGenerationError(error?.message || 'MetaMask login failed');
      }
    } finally {
      setIsMetaMaskConnecting(false);
    }
  };

  const handleAuthenticate = async () => {
    setIsAuthenticating(true);
    setGenerationError(null);

    try {
      // Validate seed phrase FIRST
      const seedValidation = validateSeedPhrase(seedPhrase);
      if (!seedValidation.valid) {
        throw new Error(seedValidation.error);
      }

      // Password is REQUIRED
      if (!password) {
        throw new Error('Password is required for wallet encryption');
      }

      // CRITICAL SECURITY FIX v1.0.68-beta: MANDATORY password verification for existing wallets
      const encryptedMnemonic = localStorage.getItem('walletEncryptedMnemonic');
      const encryptedKey = localStorage.getItem('walletEncryptedKey');
      const storedAddress = localStorage.getItem('walletAddress');

      const { keypairFromMnemonic, recoverMnemonic, loadWallet } = await import('../services/walletAuth');
      const providedKeyPair = await keypairFromMnemonic(seedPhrase);
      const providedWalletAddress = providedKeyPair.address;

      const hasExistingEncryptedWallet = storedAddress && (encryptedMnemonic || encryptedKey);

      if (hasExistingEncryptedWallet && providedWalletAddress === storedAddress) {
        console.log('🔐 Existing wallet found - MANDATORY password verification...');
        console.log('🔐 Same wallet detected (addresses match) - password verification is MANDATORY');

        try {
          if (encryptedMnemonic) {
            const storedMnemonic = await recoverMnemonic(password);
            if (storedMnemonic.trim() !== seedPhrase.trim()) {
              console.error('❌ CRITICAL: Address matched but mnemonic different!');
              throw new Error('Wallet data corruption detected. Please contact support.');
            }
            console.log('✅ Password verified via mnemonic decryption!');
          } else if (encryptedKey) {
            await loadWallet(password);
            console.log('✅ Password verified via key decryption (legacy wallet)!');
          }
        } catch (decryptError) {
          // v7.2.12: User has correct mnemonic (address matches) but stored encrypted
          // data uses a different password. This happens when:
          // 1. User changed their password
          // 2. localStorage has stale data from a previous session
          // 3. Encrypted data got corrupted
          // Since mnemonic proves ownership, clear stale encrypted data and re-encrypt
          // with the new password via storeWallet() below.
          console.warn('⚠️ Stored encrypted data uses different password - re-encrypting with new password');
          console.warn('   (User proved ownership via correct mnemonic → address match)');
          localStorage.removeItem('walletEncryptedMnemonic');
          localStorage.removeItem('walletEncryptedKey');
          localStorage.removeItem('walletEncryptedAegisKey');
          localStorage.removeItem('walletAegisPublicKey');
          localStorage.removeItem('walletEncryptedSQIsignKey');
          localStorage.removeItem('walletSQIsignPublicKey');
          localStorage.removeItem('walletEncryptedDilithium5Key');
          localStorage.removeItem('walletDilithium5PublicKey');
          localStorage.removeItem('walletPasswordHash');
        }
      } else if (storedAddress && providedWalletAddress !== storedAddress) {
        console.warn('⚠️ Different wallet detected (address mismatch)');
        localStorage.removeItem('walletEncryptedMnemonic');
        localStorage.removeItem('walletEncryptedKey');
        localStorage.removeItem('walletAddress');
        localStorage.removeItem('walletPublicKey');
        localStorage.removeItem('walletEncryptedAegisKey');
        localStorage.removeItem('walletAegisPublicKey');
        localStorage.removeItem('walletPasswordHash');
        localStorage.removeItem('cachedBalance');
        localStorage.removeItem('cachedQugusdBalance');
        localStorage.removeItem('walletBalanceHistory');
      } else if (!hasExistingEncryptedWallet && hasPasswordHash()) {
        // Stale password hash from a previous wallet (e.g., after logout or switching wallets).
        // No encrypted wallet data exists, so the hash is orphaned. Clear it and proceed
        // with the new wallet — the user will get a fresh password hash from storeWallet().
        console.warn('⚠️ Stale password hash found with no encrypted wallet data - clearing');
        localStorage.removeItem('walletPasswordHash');
      }

      const response = await qnkAPI.createWallet(seedPhrase, password);

      if (response.success && response.data) {
        localStorage.setItem('walletAddress', response.data.address_formatted || '');
        localStorage.setItem('walletId', response.data.id);
        localStorage.removeItem('cachedBalance');
        localStorage.removeItem('cachedQugusdBalance');
        localStorage.removeItem('walletBalanceHistory');
        console.log('🗑️ Cleared cached balance from previous wallet');

        try {
          const wallet = await storeWallet(seedPhrase, password, true, true, true);
          walletSession.setSession(
            wallet.privateKey,
            wallet.address,
            seedPhrase,
            wallet.dilithium5SecretKey,
            wallet.dilithium5PublicKey
          );
          console.log('✅ Wallet encrypted with password-protected AES-256-GCM');
          console.log('✅ AEGIS-QL post-quantum keys generated and encrypted');
          if (wallet.dilithium5SecretKey) {
            console.log('✅ Dilithium5 post-quantum keys stored in session (NIST Level 5)');
          }
        } catch (error) {
          console.error('Failed to encrypt wallet:', error);
          throw new Error(`Wallet encryption failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }

        await new Promise(resolve => setTimeout(resolve, 1000));
        onAuthenticate();
      } else {
        throw new Error(response.error || 'Failed to import wallet');
      }
    } catch (error) {
      console.error('Wallet authentication failed:', error);
      setGenerationError(error instanceof Error ? error.message : 'Authentication failed');
      setIsAuthenticating(false);
    }
  };

  const generateQuantumSeed = async () => {
    setIsGenerating(true);
    setGenerationError(null);
    setShowQuantumGenerator(true);

    try {
      await new Promise(resolve => setTimeout(resolve, 800));

      const response = await qnkAPI.generateMnemonic();

      if (response.success && response.data) {
        setSeedPhrase(response.data.mnemonic);
        await new Promise(resolve => setTimeout(resolve, 700));
      } else {
        throw new Error(response.error || 'Failed to generate mnemonic');
      }
    } catch (error) {
      console.error('Quantum seed generation failed:', error);
      setGenerationError(error instanceof Error ? error.message : 'Unknown error');
      setSeedPhrase('abandon ability able about above absent absorb abstract absurd abuse access accident');
    } finally {
      setShowQuantumGenerator(false);
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen px-4 relative overflow-hidden">
      {/* Quantum Field Background - Physics-inspired particles powered by live API data */}
      <QuantumFieldBackground />

      {/* Subtle radial vignette overlay - above particles */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          zIndex: 2,
          background: 'radial-gradient(ellipse at 50% 40%, transparent 0%, rgba(0,0,0,0.3) 70%, rgba(0,0,0,0.55) 100%)',
        }}
      />

      {/* All content sits above the background + particles */}
      <div className="relative" style={{ zIndex: 3 }}>

        {/* Quick Actions Bar - Top Right */}
        <div
          className="absolute top-3 right-3 flex items-center gap-1.5 z-50"
          onMouseEnter={() => setMenuHovered(true)}
          onMouseLeave={() => setMenuHovered(false)}
        >
          {/* Slint Native Wallet */}
          <motion.button
            className="relative flex flex-col items-center gap-0.5 px-2.5 py-1.5 bg-emerald-600/20 hover:bg-emerald-600/40 border border-emerald-400/30 hover:border-emerald-400/60 rounded-xl transition-all cursor-pointer group backdrop-blur-md overflow-hidden"
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15, type: "spring", stiffness: 260, damping: 20 }}
            onClick={() => setShowSlintModal(true)}
          >
            <motion.div
              className="absolute inset-0 rounded-xl bg-gradient-to-b from-emerald-400/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"
            />
            <Wallet className="w-5 h-5 text-emerald-400 relative z-10" />
            <span className="text-[9px] font-bold text-emerald-300/90 tracking-wider uppercase relative z-10">Wallet</span>
          </motion.button>

          {/* Mining Download */}
          <motion.button
            className="relative flex flex-col items-center gap-0.5 px-2.5 py-1.5 bg-amber-600/20 hover:bg-amber-600/40 border border-amber-400/30 hover:border-amber-400/60 rounded-xl transition-all cursor-pointer group backdrop-blur-md"
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 260, damping: 20 }}
            onClick={() => setShowMinerModal(true)}
          >
            <motion.div
              className="absolute inset-0 rounded-xl bg-gradient-to-b from-amber-400/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"
            />
            <Pickaxe className="w-5 h-5 text-amber-400 relative z-10" />
            <span className="text-[9px] font-bold text-amber-300/90 tracking-wider uppercase relative z-10">Mine</span>
          </motion.button>

          {/* Node Download */}
          <motion.button
            className="relative flex flex-col items-center gap-0.5 px-2.5 py-1.5 bg-cyan-600/20 hover:bg-cyan-600/40 border border-cyan-400/30 hover:border-cyan-400/60 rounded-xl transition-all cursor-pointer group backdrop-blur-md"
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25, type: "spring", stiffness: 260, damping: 20 }}
            onClick={() => setShowNodeModal(true)}
          >
            <motion.div className="absolute inset-0 rounded-xl bg-gradient-to-b from-cyan-400/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <svg viewBox="0 0 100 100" className="w-5 h-5 relative z-10" fill="none">
              <rect x="18" y="15" width="64" height="20" rx="4" fill="#06B6D4" opacity="0.9"/>
              <rect x="18" y="40" width="64" height="20" rx="4" fill="#0891B2" opacity="0.85"/>
              <rect x="18" y="65" width="64" height="20" rx="4" fill="#0E7490" opacity="0.8"/>
              <circle cx="30" cy="25" r="3" fill="#34D399"/>
              <circle cx="40" cy="25" r="3" fill="#34D399"/>
              <circle cx="30" cy="50" r="3" fill="#34D399"/>
              <circle cx="40" cy="50" r="3" fill="#FBBF24"/>
              <circle cx="30" cy="75" r="3" fill="#34D399"/>
              <circle cx="40" cy="75" r="3" fill="#34D399"/>
              {/* Drive bays */}
              <rect x="55" y="21" width="20" height="8" rx="1.5" fill="#164E63" opacity="0.6"/>
              <rect x="55" y="46" width="20" height="8" rx="1.5" fill="#164E63" opacity="0.6"/>
              <rect x="55" y="71" width="20" height="8" rx="1.5" fill="#164E63" opacity="0.6"/>
            </svg>
            <span className="text-[9px] font-bold text-cyan-300/90 tracking-wider uppercase relative z-10">Node</span>
          </motion.button>

          {/* Tor Onion */}
          <motion.button
            className={`relative flex flex-col items-center gap-0.5 px-2.5 py-1.5 ${torActive ? 'bg-purple-600/20 hover:bg-purple-600/40 border-purple-400/30 hover:border-purple-400/60' : 'bg-gray-600/15 hover:bg-gray-600/30 border-gray-500/20 hover:border-gray-500/40'} border rounded-xl transition-all cursor-pointer group backdrop-blur-md`}
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, type: "spring", stiffness: 260, damping: 20 }}
            onClick={openTorSite}
          >
            <motion.div className={`absolute inset-0 rounded-xl bg-gradient-to-b ${torActive ? 'from-purple-400/10' : 'from-gray-400/5'} to-transparent opacity-0 group-hover:opacity-100 transition-opacity`} />
            <svg viewBox="0 0 100 100" className="w-5 h-5 relative z-10" fill="none">
              <ellipse cx="50" cy="55" rx="35" ry="40" fill={torActive ? "#7B4397" : "#555"} opacity="0.9"/>
              <ellipse cx="50" cy="53" rx="28" ry="32" fill={torActive ? "#9B59B6" : "#666"}/>
              <ellipse cx="50" cy="51" rx="21" ry="24" fill={torActive ? "#A569BD" : "#777"}/>
              <ellipse cx="50" cy="49" rx="14" ry="16" fill={torActive ? "#BB8FCE" : "#888"}/>
              <path d="M50 15 Q52 20 50 25 Q48 30 50 35" stroke="#5D4E37" strokeWidth="4" fill="none" strokeLinecap="round"/>
              <path d="M50 20 Q60 15 58 25 Q55 30 50 25" fill="#27AE60"/>
            </svg>
            <span className={`text-[9px] font-bold ${torActive ? 'text-purple-300/90' : 'text-gray-400/70'} tracking-wider uppercase relative z-10`}>Tor</span>
          </motion.button>

          {/* Research Library */}
          <motion.button
            className="relative flex flex-col items-center gap-0.5 px-2.5 py-1.5 bg-orange-600/20 hover:bg-orange-600/40 border border-orange-400/30 hover:border-orange-400/60 rounded-xl transition-all cursor-pointer group backdrop-blur-md"
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35, type: "spring", stiffness: 260, damping: 20 }}
            onClick={() => setShowPapersLibrary(true)}
          >
            <motion.div className="absolute inset-0 rounded-xl bg-gradient-to-b from-orange-400/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <BookOpen className="w-5 h-5 text-orange-400 relative z-10" />
            <span className="text-[9px] font-bold text-orange-300/90 tracking-wider uppercase relative z-10">Papers</span>
          </motion.button>

          {/* AI Setup */}
          <motion.button
            className="relative flex flex-col items-center gap-0.5 px-2.5 py-1.5 bg-violet-600/20 hover:bg-violet-600/40 border border-violet-400/30 hover:border-violet-400/60 rounded-xl transition-all cursor-pointer group backdrop-blur-md overflow-hidden"
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, type: "spring", stiffness: 260, damping: 20 }}
            onClick={() => setShowAIModal(true)}
          >
            <motion.div className="absolute inset-0 rounded-xl bg-gradient-to-b from-violet-400/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            {/* Subtle AI pulse */}
            <motion.div
              className="absolute inset-0 rounded-xl border border-violet-400/20"
              animate={{ opacity: [0.3, 0.8, 0.3] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
            />
            <Sparkles className="w-5 h-5 text-violet-400 relative z-10" />
            <span className="text-[9px] font-bold text-violet-300/90 tracking-wider uppercase relative z-10">AI</span>
          </motion.button>

          {/* Help */}
          <motion.button
            className="relative flex flex-col items-center gap-0.5 px-2.5 py-1.5 bg-gray-600/15 hover:bg-gray-600/30 border border-gray-500/20 hover:border-gray-500/40 rounded-xl transition-all cursor-pointer group backdrop-blur-md"
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.45, type: "spring", stiffness: 260, damping: 20 }}
            onClick={() => setShowInfoModal(true)}
          >
            <motion.div className="absolute inset-0 rounded-xl bg-gradient-to-b from-gray-400/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <HelpCircle className="w-5 h-5 text-gray-400 relative z-10" />
            <span className="text-[9px] font-bold text-gray-400/80 tracking-wider uppercase relative z-10">Help</span>
          </motion.button>
        </div>

        {/* Fade out main UI when hovering top-right menu */}
        <motion.div
          animate={{ opacity: menuHovered ? 0.15 : 1 }}
          transition={{ duration: 0.3 }}
          style={{ pointerEvents: menuHovered ? 'none' as const : 'auto' as const }}
        >

        {/* Explorer Search Bar + Live Data Dropdown */}
        <motion.div
          ref={explorerContainerRef}
          className="pt-6 pb-4"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: menuHovered ? 0 : 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div
            className="flex items-center justify-center gap-3 mb-2 cursor-pointer group"
            onClick={() => setShowExplorerDropdown(prev => !prev)}
          >
            <Search className="w-4 h-4 text-amber-400" />
            <span className="text-amber-200/70 text-sm group-hover:text-amber-200 transition-colors">
              Network Explorer
            </span>
            <motion.div
              animate={{ rotate: showExplorerDropdown ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <ChevronDown className="w-4 h-4 text-amber-400/60" />
            </motion.div>
          </div>
          <ExplorerSearchBar />

          {/* Live Explorer Data Dropdown */}
          <AnimatePresence>
            {showExplorerDropdown && (
              <motion.div
                initial={{ opacity: 0, y: -15, scaleY: 0.8 }}
                animate={{ opacity: 1, y: 0, scaleY: 1 }}
                exit={{ opacity: 0, y: -10, scaleY: 0.8 }}
                transition={{ duration: 0.25, ease: 'easeOut' }}
                className="mt-3 max-w-2xl mx-auto rounded-2xl overflow-hidden backdrop-blur-xl"
                style={{
                  background: 'linear-gradient(145deg, rgba(8,12,30,0.95) 0%, rgba(15,10,35,0.95) 50%, rgba(8,15,30,0.95) 100%)',
                  border: '1px solid rgba(212,175,55,0.25)',
                  boxShadow: '0 20px 60px rgba(0,0,0,0.5), 0 0 40px rgba(212,175,55,0.1), inset 0 1px 0 rgba(255,255,255,0.05)',
                  transformOrigin: 'top center',
                }}
              >
                {/* Network Stats Bar - SSE-driven live data */}
                {(explorerData.health || liveBlockHeight > 0) && (
                  <div className="px-4 py-3 border-b border-amber-500/15 flex items-center justify-between flex-wrap gap-2">
                    <div className="flex items-center gap-4 text-xs">
                      <div className="flex items-center gap-1.5">
                        <div className={`w-2 h-2 rounded-full ${(explorerData.health?.status === 'ready') ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]' : 'bg-amber-400 animate-pulse'}`} />
                        <span className="text-amber-100/80 font-medium uppercase tracking-wide">{explorerData.health?.status || 'connecting'}</span>
                      </div>
                      <motion.div
                        className="flex items-center gap-1"
                        animate={blockPulse ? { scale: [1, 1.2, 1], color: ['#67e8f9', '#fbbf24', '#67e8f9'] } : {}}
                        transition={{ duration: 0.5 }}
                      >
                        <Blocks className="w-3 h-3 text-cyan-400" />
                        <span className="text-cyan-300/90 font-mono font-bold">{(liveBlockHeight || explorerData.health?.height || 0).toLocaleString()}</span>
                        {blockPulse && <span className="text-emerald-400 text-[9px] font-bold ml-0.5 animate-pulse">NEW</span>}
                      </motion.div>
                      <div className="flex items-center gap-1">
                        <Users className="w-3 h-3 text-purple-400" />
                        <span className="text-purple-300/90">{livePeers || explorerData.health?.peers || 0} peers</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3 text-amber-400" />
                        <span className="text-amber-300/80">{formatUptime(explorerData.health?.uptime_secs || 0)}</span>
                      </div>
                      {qugMarketCap > 0 && (
                        <div className="flex items-center gap-1">
                          <TrendingUp className="w-3 h-3 text-emerald-400" />
                          <span className="text-emerald-300/90 font-mono font-bold">${qugMarketCap >= 1_000_000 ? (qugMarketCap / 1_000_000).toFixed(2) + 'M' : qugMarketCap >= 1_000 ? (qugMarketCap / 1_000).toFixed(1) + 'K' : qugMarketCap.toFixed(0)}</span>
                          <span className="text-emerald-400/50 text-[9px]">MCap</span>
                        </div>
                      )}
                      {qugPrice > 0 && (
                        <div className="flex items-center gap-1">
                          <span className="text-amber-200/70 font-mono text-[10px]">QUG</span>
                          <span className="text-amber-300/90 font-mono font-bold">${qugPrice.toFixed(2)}</span>
                        </div>
                      )}
                      {minerCount > 0 && (
                        <div className="flex items-center gap-1">
                          <Pickaxe className="w-3 h-3 text-orange-400" />
                          <span className="text-orange-300/90 font-mono font-bold">{minerCount}</span>
                          <span className="text-orange-400/50 text-[9px]">miners</span>
                        </div>
                      )}
                      {networkHashrate && (
                        <div className="flex items-center gap-1">
                          <Cpu className="w-3 h-3 text-sky-400" />
                          <span className="text-sky-300/90 font-mono font-bold">{networkHashrate}</span>
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5 text-[10px] text-amber-400/50 font-mono">
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-400/60 animate-pulse" />
                      <span>LIVE</span>
                      <span className="text-amber-400/30">|</span>
                      <span>v{explorerData.health?.version || '...'}</span>
                    </div>
                  </div>
                )}

                {explorerLoading && !explorerData.health && (
                  <div className="px-4 py-6 flex items-center justify-center gap-2">
                    <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
                      <Activity className="w-4 h-4 text-amber-400" />
                    </motion.div>
                    <span className="text-amber-300/60 text-sm">Loading live blockchain data...</span>
                  </div>
                )}

                {/* Two-Column Layout: Recent Blocks + Recent Activity */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-0 divide-y md:divide-y-0 md:divide-x divide-amber-500/10">
                  {/* Recent Blocks */}
                  <div className="p-3">
                    <div className="flex items-center gap-2 mb-2.5 px-1">
                      <Blocks className="w-3.5 h-3.5 text-cyan-400" />
                      <span className="text-xs font-bold text-cyan-300/90 uppercase tracking-wider">Recent Blocks</span>
                    </div>
                    <div className="space-y-1.5">
                      {explorerData.blocks.slice(0, 5).map((block: any, idx: number) => (
                        <motion.div
                          key={block.height || idx}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.05 }}
                          className="p-2.5 rounded-xl hover:bg-cyan-500/8 transition-all group cursor-pointer"
                          style={{ background: 'rgba(6,182,212,0.03)' }}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <div className="w-7 h-7 rounded-lg bg-cyan-500/15 border border-cyan-500/20 flex items-center justify-center group-hover:bg-cyan-500/25 transition-colors">
                                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                              </div>
                              <div>
                                <div className="text-xs font-bold text-cyan-100 font-mono">#{(block.height || 0).toLocaleString()}</div>
                                <div className="text-[10px] text-cyan-300/50">{block.tx_count || 0} txs &middot; round {block.dag_round || '?'}</div>
                              </div>
                            </div>
                            <div className="text-[10px] text-cyan-400/50">
                              {block.timestamp ? formatTimeAgo(block.timestamp) : ''}
                            </div>
                          </div>
                        </motion.div>
                      ))}
                      {explorerData.blocks.length === 0 && !explorerLoading && (
                        <div className="text-center text-amber-300/40 text-xs py-4">No blocks yet</div>
                      )}
                    </div>
                  </div>

                  {/* Recent Activity */}
                  <div className="p-3">
                    <div className="flex items-center gap-2 mb-2.5 px-1">
                      <Activity className="w-3.5 h-3.5 text-amber-400" />
                      <span className="text-xs font-bold text-amber-300/90 uppercase tracking-wider">Recent Activity</span>
                    </div>
                    <div className="space-y-1.5">
                      {explorerData.transactions.slice(0, 5).map((tx: any, idx: number) => {
                        const isMining = tx.type === 'mining_rewards';
                        const icon = isMining
                          ? <Pickaxe className="w-3.5 h-3.5 text-amber-400" />
                          : <Hash className="w-3.5 h-3.5 text-purple-400" />;
                        const bgColor = isMining ? 'rgba(212,175,55,0.03)' : 'rgba(148,85,247,0.03)';
                        const borderColor = isMining ? 'border-amber-500/20' : 'border-purple-500/20';
                        const iconBg = isMining ? 'bg-amber-500/15' : 'bg-purple-500/15';

                        return (
                          <motion.div
                            key={tx.id || idx}
                            initial={{ opacity: 0, x: 10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.05 }}
                            className={`p-2.5 rounded-xl hover:brightness-125 transition-all cursor-pointer`}
                            style={{ background: bgColor }}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <div className={`w-7 h-7 rounded-lg ${iconBg} border ${borderColor} flex items-center justify-center`}>
                                  {icon}
                                </div>
                                <div>
                                  <div className="text-xs font-medium text-amber-100 truncate max-w-[140px]">
                                    {tx.amount || tx.type || 'Activity'}
                                  </div>
                                  <div className="text-[10px] text-amber-300/50 truncate max-w-[140px]">
                                    {tx.from || 'Private'} &rarr; {tx.to || 'Private'}
                                  </div>
                                </div>
                              </div>
                              <div className="flex flex-col items-end gap-0.5">
                                <div className="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300/80 border border-emerald-500/20">
                                  {tx.status || 'confirmed'}
                                </div>
                                <div className="text-[10px] text-amber-400/40">
                                  {tx.timestamp ? formatTimeAgo(tx.timestamp) : ''}
                                </div>
                              </div>
                            </div>
                          </motion.div>
                        );
                      })}
                      {explorerData.transactions.length === 0 && !explorerLoading && (
                        <div className="text-center text-amber-300/40 text-xs py-4">No recent activity</div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Network Analytics Footer */}
                {explorerData.networkStats && (
                  <div className="px-4 py-2.5 border-t border-amber-500/10 flex items-center justify-between text-[10px] text-amber-300/50">
                    <div className="flex items-center gap-3">
                      <span>Health: <span className={`font-bold ${(explorerData.networkStats.network_health_score || 0) > 0.7 ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {((explorerData.networkStats.network_health_score || 0) * 100).toFixed(0)}%
                      </span></span>
                      {explorerData.networkStats.tor_active && (
                        <span className="flex items-center gap-1 text-purple-400">
                          <Shield className="w-2.5 h-2.5" /> Tor
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1 text-amber-400/40">
                      <TrendingUp className="w-2.5 h-2.5" />
                      <span>Live from Quillon Graph</span>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        <div className="flex-1 flex items-center justify-center">
          <motion.div
            className="w-full max-w-md"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            {/* Logo and Title */}
            <div className="text-center mb-12">
              <motion.div
                className="inline-block mb-6"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <div className="w-40 h-40 mx-auto relative">
                  {/* Floating particles around logo */}
                  <FloatingParticles />

                  {/* Cosmic glow effect - enhanced with pulsing rings */}
                  <motion.div
                    className="absolute inset-[-20px] rounded-full"
                    style={{
                      background: 'radial-gradient(circle, rgba(212,175,55,0.15) 0%, rgba(255,165,0,0.08) 40%, transparent 70%)',
                    }}
                    animate={{
                      scale: [1, 1.15, 1],
                      opacity: [0.6, 1, 0.6],
                    }}
                    transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                  />

                  {/* Outer orbit ring */}
                  <motion.div
                    className="absolute inset-[-10px] rounded-full border border-amber-500/10"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                  >
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-amber-400/60" />
                  </motion.div>

                  {/* Gold border ring */}
                  <div className="absolute inset-0 rounded-full" style={{
                    background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                    padding: '3px'
                  }}>
                    <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center p-4">
                      <img
                        src="/quillon-logo.png"
                        alt="Quillon Graph Logo"
                        className="w-full h-full object-contain"
                        style={{ filter: 'invert(1)' }}
                      />
                    </div>
                  </div>
                </div>
              </motion.div>

              <motion.h1
                className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent"
                style={{ textShadow: '0 0 40px rgba(251,191,36,0.3)' }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.6 }}
              >
                Quillon Graph
              </motion.h1>
              <motion.p
                className="text-amber-200/80 mt-2 font-semibold text-base tracking-wide"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5, duration: 0.6 }}
              >
                Private Settlement Infrastructure
              </motion.p>
              <motion.p
                className="text-amber-200/40 mt-1.5 text-xs tracking-widest uppercase"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.65, duration: 0.6 }}
              >
                Post-quantum settlement rails for operators and sovereign networks
              </motion.p>
            </div>

            {/* Login Form - frosted glass card */}
            <motion.div
              className="relative rounded-3xl p-8 backdrop-blur-xl"
              style={{
                background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.85) 100%)',
                border: '2px solid',
                borderImage: 'linear-gradient(135deg, #D4AF37, #FFD700, #FFA500, #FFD700, #D4AF37) 1',
                boxShadow: '0 0 40px rgba(212, 175, 55, 0.15), inset 0 0 30px rgba(212, 175, 55, 0.05), 0 25px 50px rgba(0,0,0,0.5)'
              }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.5 }}
            >
              <div className="space-y-6">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-sm font-medium text-amber-200">
                      BIP39 Seed Phrase
                    </label>
                    <button
                      type="button"
                      onClick={generateQuantumSeed}
                      className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-emerald-300 bg-emerald-900/30 hover:bg-emerald-800/50 border border-emerald-500/30 hover:border-emerald-400/60 rounded-lg transition-all"
                    >
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
                      Generate new
                    </button>
                  </div>
                  <textarea
                    value={seedPhrase}
                    onChange={(e) => setSeedPhrase(e.target.value)}
                    className="w-full h-24 px-4 py-3 bg-slate-900/70 border-2 border-amber-500/30 rounded-xl text-amber-50 placeholder-slate-400 focus:outline-none focus:border-amber-400 focus:shadow-[0_0_15px_rgba(251,191,36,0.3)] transition-all resize-none backdrop-blur-sm"
                    placeholder="Enter your 12-word seed phrase, or click 'Generate new' to create one..."
                  />
                  {/* Word count hint */}
                  {seedPhrase && (
                    <div className={`text-xs mt-1 ${
                      seedPhrase.trim().split(/\s+/).filter(w => w.length > 0).length === 12 ||
                      seedPhrase.trim().split(/\s+/).filter(w => w.length > 0).length === 24
                        ? 'text-green-400'
                        : 'text-amber-400/60'
                    }`}>
                      {seedPhrase.trim().split(/\s+/).filter(w => w.length > 0).length} / 12 words
                      {seedPhrase.trim().split(/\s+/).filter(w => w.length > 0).length !== 12 &&
                       seedPhrase.trim().split(/\s+/).filter(w => w.length > 0).length !== 24 &&
                        ' (need 12 or 24)'}
                    </div>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-amber-200 mb-2">
                    Password (Required for wallet encryption)
                  </label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && validateSeedPhrase(seedPhrase).valid && password && !isAuthenticating) {
                        handleAuthenticate();
                      }
                    }}
                    className="w-full px-4 py-3 bg-slate-900/70 border-2 border-amber-500/30 rounded-xl text-amber-50 placeholder-slate-400 focus:outline-none focus:border-amber-400 focus:shadow-[0_0_15px_rgba(251,191,36,0.3)] transition-all backdrop-blur-sm"
                    placeholder="Enter password for wallet encryption..."
                    required
                  />
                </div>

                {/* Quantum Generator Button */}
                <motion.button
                  onClick={generateQuantumSeed}
                  disabled={isGenerating}
                  className="w-full py-4 px-6 bg-gradient-to-r from-amber-900/40 to-yellow-900/40 border-2 border-amber-500/40 rounded-xl text-amber-100 font-medium flex items-center justify-center gap-3 hover:border-amber-400 hover:shadow-[0_0_20px_rgba(251,191,36,0.3)] transition-all disabled:opacity-50 disabled:cursor-not-allowed backdrop-blur-sm"
                  whileHover={{ scale: isGenerating ? 1 : 1.02 }}
                  whileTap={{ scale: isGenerating ? 1 : 0.98 }}
                >
                  {isGenerating ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Sparkles className="w-5 h-5 text-amber-400" />
                      </motion.div>
                      <span>Generating BIP39 Mnemonic...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5 text-amber-400" />
                      <span>Generate New Seed Phrase</span>
                    </>
                  )}
                </motion.button>

                {/* Error Display */}
                {generationError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300 text-sm flex items-center gap-2 backdrop-blur-sm"
                  >
                    <AlertCircle className="w-4 h-4" />
                    <span>Generation failed: {generationError}</span>
                  </motion.div>
                )}

                {/* Quantum Generator Animation */}
                {showQuantumGenerator && (
                  <motion.div
                    className="h-32 rounded-xl overflow-hidden relative border-2 border-amber-500/30"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-amber-500/20 via-yellow-500/20 to-orange-500/20 animate-pulse" />
                    <div className="absolute inset-0 bg-slate-900/80 flex items-center justify-center backdrop-blur-sm">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Sparkles className="w-12 h-12 text-amber-400" />
                      </motion.div>
                    </div>
                  </motion.div>
                )}

                {/* Authenticate Button */}
                <motion.button
                  onClick={handleAuthenticate}
                  disabled={!validateSeedPhrase(seedPhrase).valid || !password || isAuthenticating}
                  className="w-full py-5 px-6 bg-gradient-to-r from-amber-600 to-yellow-600 rounded-xl text-slate-900 font-bold text-lg flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-[0_0_30px_rgba(251,191,36,0.5)] hover:from-amber-500 hover:to-yellow-500 transition-all"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {isAuthenticating ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Key className="w-6 h-6" />
                      </motion.div>
                      <span>Authenticating...</span>
                    </>
                  ) : (
                    <>
                      <Key className="w-6 h-6" />
                      <span>Authenticate</span>
                    </>
                  )}
                </motion.button>

                {/* MetaMask Divider */}
                <div className="flex items-center gap-3 my-1">
                  <div className="flex-1 h-px bg-gradient-to-r from-transparent via-amber-500/30 to-transparent" />
                  <span className="text-amber-400/60 text-xs font-medium uppercase tracking-wider">or</span>
                  <div className="flex-1 h-px bg-gradient-to-r from-transparent via-amber-500/30 to-transparent" />
                </div>

                {/* MetaMask Sign-In Button */}
                <motion.button
                  onClick={handleMetaMaskLogin}
                  disabled={isMetaMaskConnecting || isAuthenticating}
                  className="w-full py-4 px-6 bg-gradient-to-r from-[#E2761B]/20 to-[#CD6116]/20 border-2 border-[#E2761B]/40 rounded-xl text-white font-bold flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed hover:border-[#E2761B]/70 hover:shadow-[0_0_25px_rgba(226,118,27,0.3)] hover:bg-[#E2761B]/30 transition-all backdrop-blur-sm"
                  whileHover={{ scale: isMetaMaskConnecting ? 1 : 1.02 }}
                  whileTap={{ scale: isMetaMaskConnecting ? 1 : 0.98 }}
                >
                  {isMetaMaskConnecting ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                          <path d="M21.17 5.17L12.85 1.15a2 2 0 00-1.7 0L2.83 5.17A2 2 0 001.83 7v10a2 2 0 001 1.73l8.32 4.24a2 2 0 001.7 0l8.32-4.24a2 2 0 001-1.73V7a2 2 0 00-1-1.83z" stroke="#E2761B" strokeWidth="2" fill="none"/>
                        </svg>
                      </motion.div>
                      <span>Connecting MetaMask...</span>
                    </>
                  ) : (
                    <>
                      <svg width="28" height="28" viewBox="0 0 318.6 318.6" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M274.1 35.5l-99.5 73.9L193 65.8z" fill="#E2761B" stroke="#E2761B" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M44.4 35.5l98.7 74.6-17.5-44.3zm187.9 174.5l-26.5 40.6 56.7 15.6 16.3-55.3zm-204.4.9L44.1 266.2l56.7-15.6-26.5-40.6z" fill="#E4761B" stroke="#E4761B" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M97.9 209.3l-15.8 23.9 56.3 2.5-2-60.5zm118.8 0l-39.1-34.8-1.3 61.2 56.2-2.5zM100.8 250.6l34-16.6-29.3-22.9zm83 -16.6l34 16.6-4.7-39.5z" fill="#E4761B" stroke="#E4761B" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M217.8 250.6l-34-16.6 2.7 22.1-.3 9.3zm-117 0l31.5 14.8-.2-9.3 2.5-22.1z" fill="#D7C1B3" stroke="#D7C1B3" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M132.3 198.6l-28.2-8.3 19.9-9.1zm54 0l8.3-17.4 20 9.1z" fill="#233447" stroke="#233447" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M100.8 250.6l4.8-40.6-31.3.9zM213 210l4.8 40.6 26.5-39.7zM230.8 171.6l-56.2 2.5 5.2 28.9 8.3-17.4 20 9.1zM104.1 195.1l20-9.1 8.2 17.4 5.3-28.9-56.3-2.5z" fill="#CD6116" stroke="#CD6116" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M81.8 172l58.3 2.8-5.3 28.9zm154.6 0l-53.1 31.4 5.2-28.9zm-152.6 2.5l56.3 2.5-4.5 19.4-1-.5-19.9-9.1zm110.8 0l-30.8 12.3-20 9.1-.9.5-4.5-19.4z" fill="#E4751F" stroke="#E4751F" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M183.8 234l-2.5 22.1.3 2.5 21.7-16.9zm-83-16.6l-2.5 22.1 2.7 22.1.2-2.5 21.7-16.9z" fill="#F6851B" stroke="#F6851B" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      <span>Sign in with MetaMask</span>
                      {!hasMetaMask && <span className="text-xs text-amber-400/60 ml-1">(not detected)</span>}
                    </>
                  )}
                </motion.button>
              </div>

              {/* Photon Waterfall Effect */}
              {(isAuthenticating || isMetaMaskConnecting) && (
                <motion.div
                  className="mt-6 h-2 rounded-full overflow-hidden border border-amber-500/30"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <motion.div
                    className="h-full bg-gradient-to-r from-amber-600 via-yellow-500 to-amber-600"
                    initial={{ x: '-100%' }}
                    animate={{ x: '100%' }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                </motion.div>
              )}
            </motion.div>

            {/* Security Note */}
            <motion.p
              className="text-center text-amber-300/60 text-sm mt-6 font-medium"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
            >
              Private Settlement Rails &bull; Deployable Sovereign Stack &bull; Post-Quantum Networking
            </motion.p>
          </motion.div>
        </div>

        </motion.div>{/* end menu-hover fade wrapper */}
      </div>

      {/* Info Modal - What is Quillon Graph? */}
      <AnimatePresence>
        {/* AI Setup Modal */}
        {showAIModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[100] overflow-y-auto py-8"
            onClick={() => setShowAIModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="bg-gradient-to-br from-slate-900 via-violet-950/30 to-slate-900 border-2 border-violet-500/30 rounded-2xl p-6 max-w-lg w-full mx-4 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
              style={{ boxShadow: '0 0 80px rgba(139, 92, 246, 0.25)' }}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-violet-500/30 to-purple-500/30 border border-violet-500/50 flex items-center justify-center relative">
                    <Sparkles className="w-7 h-7 text-violet-400" />
                    <motion.div
                      className="absolute inset-0 rounded-full border-2 border-violet-400/30"
                      animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold bg-gradient-to-r from-violet-400 to-purple-400 bg-clip-text text-transparent">
                      AI Setup
                    </h2>
                    <p className="text-violet-300/60 text-sm">Manage wallet & mining with natural language</p>
                  </div>
                </div>
                <button onClick={() => setShowAIModal(false)} className="p-2 hover:bg-violet-500/20 rounded-lg transition-colors">
                  <X className="w-6 h-6 text-violet-400" />
                </button>
              </div>

              {/* What you can do */}
              <div className="mb-5 p-4 bg-violet-500/5 border border-violet-500/15 rounded-xl">
                <p className="text-violet-200/80 text-sm mb-3 font-medium">After setup, just say in natural language:</p>
                <div className="space-y-2">
                  {[
                    { cmd: '"Create a wallet"', desc: 'Generate address + recovery phrase', icon: '🔐' },
                    { cmd: '"Start mining"', desc: 'Download miner & begin earning QUG', icon: '⛏️' },
                    { cmd: '"Create a node"', desc: 'Download & run a full network node', icon: '🖥️' },
                    { cmd: '"What\'s my balance?"', desc: 'Check any wallet instantly', icon: '💰' },
                    { cmd: '"Send 10 QUG to qnk..."', desc: 'Transfer funds with one sentence', icon: '🚀' },
                    { cmd: '"Network status"', desc: 'Height, peers, hashrate', icon: '📡' },
                  ].map((item, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.1 * i }}
                      className="flex items-center gap-3 p-2 bg-slate-800/40 rounded-lg"
                    >
                      <span className="text-lg">{item.icon}</span>
                      <div className="flex-1">
                        <span className="text-violet-200 text-sm font-mono">{item.cmd}</span>
                        <span className="text-violet-400/50 text-xs ml-2">— {item.desc}</span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Setup Command */}
              <div className="mb-5">
                <p className="text-violet-300/70 text-xs mb-2 uppercase tracking-wider font-bold">One command to set up:</p>
                <div className="relative group">
                  <pre className="bg-black/60 border border-violet-500/20 rounded-xl p-4 text-sm font-mono text-violet-300 overflow-x-auto">
                    curl -fsSL https://quillon.xyz/setup-ai.sh | bash
                  </pre>
                  <motion.button
                    className={`absolute top-2 right-2 px-3 py-1.5 ${aiCopied ? 'bg-emerald-500/30 border-emerald-400/50' : 'bg-violet-500/20 border-violet-400/30 hover:bg-violet-500/40'} border rounded-lg text-xs font-bold transition-all`}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => {
                      navigator.clipboard.writeText('curl -fsSL https://quillon.xyz/setup-ai.sh | bash');
                      setAiCopied(true);
                      setTimeout(() => setAiCopied(false), 2000);
                    }}
                  >
                    <span className={aiCopied ? 'text-emerald-300' : 'text-violet-300'}>
                      {aiCopied ? '✓ Copied!' : 'Copy'}
                    </span>
                  </motion.button>
                </div>
              </div>

              {/* How it works */}
              <div className="flex items-center gap-3 mb-5">
                {[
                  { step: '1', label: 'Paste in terminal', color: 'violet' },
                  { step: '2', label: 'Open AI assistant', color: 'purple' },
                  { step: '3', label: 'Say "create a wallet" or "create a node"', color: 'fuchsia' },
                ].map((s, i) => (
                  <div key={i} className="flex-1 text-center">
                    <div className={`w-8 h-8 rounded-full bg-${s.color}-500/20 border border-${s.color}-400/30 flex items-center justify-center mx-auto mb-1`}>
                      <span className={`text-sm font-black text-${s.color}-400`}>{s.step}</span>
                    </div>
                    <span className="text-[10px] text-violet-300/60">{s.label}</span>
                  </div>
                ))}
              </div>

              {/* Supported AIs */}
              <div className="p-3 bg-slate-800/40 rounded-xl border border-violet-500/10">
                <p className="text-violet-300/50 text-[10px] uppercase tracking-wider font-bold mb-2">Supported AI Assistants</p>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 bg-violet-500/20 border border-violet-400/30 rounded-full text-[10px] text-violet-300 font-bold">Claude Code</span>
                  <span className="px-2 py-0.5 bg-gray-500/15 border border-gray-500/20 rounded-full text-[10px] text-gray-400">ChatGPT (coming soon)</span>
                  <span className="px-2 py-0.5 bg-gray-500/15 border border-gray-500/20 rounded-full text-[10px] text-gray-400">More TBA</span>
                </div>
              </div>

              {/* Footer */}
              <p className="text-center text-violet-400/30 text-[10px] mt-4">
                No GPG signatures. No air-gapped computers. Just works.
              </p>
            </motion.div>
          </motion.div>
        )}

        {showInfoModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-start justify-center z-[100] overflow-y-auto py-8"
            onClick={() => setShowInfoModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-amber-500/30 rounded-2xl p-6 max-w-2xl w-full mx-4 shadow-2xl my-auto"
              onClick={(e) => e.stopPropagation()}
              style={{ boxShadow: '0 0 60px rgba(212, 175, 55, 0.3)' }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-amber-500/30 to-yellow-500/30 border border-amber-500/50 flex items-center justify-center">
                    <img src="/quillon-logo.png" alt="Quillon" className="w-8 h-8" style={{ filter: 'invert(1)' }} />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold bg-gradient-to-r from-amber-400 to-yellow-500 bg-clip-text text-transparent">
                      What is Quillon Graph?
                    </h2>
                    <p className="text-amber-300/60 text-sm">Next-Generation Quantum-Resistant Blockchain</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowInfoModal(false)}
                  className="p-2 hover:bg-amber-500/20 rounded-lg transition-colors"
                >
                  <X className="w-6 h-6 text-amber-400" />
                </button>
              </div>

              {/* Content */}
              <div className="space-y-6 text-amber-100/90">
                <p className="text-lg leading-relaxed">
                  <span className="font-bold text-amber-400">Quillon Graph</span> is a revolutionary Layer 1 blockchain
                  built from the ground up to be <span className="text-amber-300">quantum-resistant</span>,
                  <span className="text-amber-300"> privacy-preserving</span>, and
                  <span className="text-amber-300"> blazingly fast</span>.
                </p>

                {/* Features Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 bg-slate-800/50 rounded-xl border border-amber-500/20">
                    <div className="flex items-center gap-3 mb-2">
                      <Shield className="w-5 h-5 text-amber-400" />
                      <h3 className="font-bold text-amber-200">Post-Quantum Security</h3>
                    </div>
                    <p className="text-sm text-amber-100/70">
                      Protected by Dilithium5 & Kyber1024 cryptography, designed to withstand attacks from future quantum computers.
                    </p>
                  </div>

                  <div className="p-4 bg-slate-800/50 rounded-xl border border-amber-500/20">
                    <div className="flex items-center gap-3 mb-2">
                      <Lock className="w-5 h-5 text-amber-400" />
                      <h3 className="font-bold text-amber-200">ZK-STARK Privacy</h3>
                    </div>
                    <p className="text-sm text-amber-100/70">
                      Zero-knowledge proofs ensure transaction details remain private while still being verifiable on-chain.
                    </p>
                  </div>

                  <div className="p-4 bg-slate-800/50 rounded-xl border border-amber-500/20">
                    <div className="flex items-center gap-3 mb-2">
                      <Zap className="w-5 h-5 text-amber-400" />
                      <h3 className="font-bold text-amber-200">DAG-BFT Consensus</h3>
                    </div>
                    <p className="text-sm text-amber-100/70">
                      Narwhal-Bullshark DAG consensus achieves 100,000+ TPS with sub-second finality.
                    </p>
                  </div>

                  <div className="p-4 bg-slate-800/50 rounded-xl border border-amber-500/20">
                    <div className="flex items-center gap-3 mb-2">
                      <Globe className="w-5 h-5 text-amber-400" />
                      <h3 className="font-bold text-amber-200">Tor Integration</h3>
                    </div>
                    <p className="text-sm text-amber-100/70">
                      Built-in onion routing provides network-level anonymity and censorship resistance.
                    </p>
                  </div>
                </div>

                {/* Token Info */}
                <div className="p-4 bg-gradient-to-r from-amber-500/10 to-yellow-500/10 rounded-xl border border-amber-500/30">
                  <h3 className="font-bold text-amber-300 mb-2">Native Token: QUG</h3>
                  <p className="text-sm text-amber-100/70">
                    QUG powers the Quillon Graph network - used for transaction fees, staking, governance,
                    and accessing the decentralized exchange (DEX) with privacy-preserving swaps.
                  </p>
                </div>

                {/* Getting Started */}
                <div className="pt-4 border-t border-amber-500/20">
                  <h3 className="font-bold text-amber-200 mb-2">Getting Started</h3>
                  <ol className="text-sm text-amber-100/70 space-y-2 list-decimal list-inside">
                    <li>Click <span className="text-amber-300">"Generate Quantum Entropy"</span> to create a new wallet</li>
                    <li>Securely save your 12-word seed phrase - it's the only way to recover your wallet</li>
                    <li>Set a strong password to encrypt your wallet locally</li>
                    <li>Start mining, trading, or exploring the quantum-resistant blockchain!</li>
                  </ol>
                </div>
              </div>

              {/* Close Button */}
              <div className="mt-6 flex justify-center">
                <motion.button
                  onClick={() => setShowInfoModal(false)}
                  className="px-8 py-3 bg-gradient-to-r from-amber-600 to-yellow-600 rounded-xl text-slate-900 font-bold hover:shadow-[0_0_30px_rgba(251,191,36,0.5)] transition-all"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Got it!
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Node Download Modal */}
      <AnimatePresence>
        {showNodeModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[100] overflow-y-auto py-8"
            onClick={() => setShowNodeModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-cyan-500/30 rounded-2xl p-6 max-w-lg w-full mx-4 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
              style={{ boxShadow: '0 0 60px rgba(6, 182, 212, 0.25)' }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-cyan-500/30 to-blue-500/30 border border-cyan-500/50 flex items-center justify-center">
                    <svg viewBox="0 0 100 100" className="w-7 h-7" fill="none">
                      <rect x="18" y="15" width="64" height="20" rx="4" fill="#06B6D4" opacity="0.9"/>
                      <rect x="18" y="40" width="64" height="20" rx="4" fill="#0891B2" opacity="0.85"/>
                      <rect x="18" y="65" width="64" height="20" rx="4" fill="#0E7490" opacity="0.8"/>
                      <circle cx="30" cy="25" r="3" fill="#34D399"/>
                      <circle cx="40" cy="25" r="3" fill="#34D399"/>
                      <circle cx="30" cy="50" r="3" fill="#34D399"/>
                      <circle cx="30" cy="75" r="3" fill="#34D399"/>
                    </svg>
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                      Run a Full Node
                    </h2>
                    <p className="text-cyan-300/60 text-sm">Quantum-Resistant Validator Node</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowNodeModal(false)}
                  className="p-2 hover:bg-cyan-500/20 rounded-lg transition-colors"
                >
                  <X className="w-6 h-6 text-cyan-400" />
                </button>
              </div>

              {/* Feature highlights */}
              <div className="grid grid-cols-3 gap-2 mb-5">
                <div className="p-2 bg-cyan-500/10 border border-cyan-500/20 rounded-lg text-center">
                  <div className="text-lg font-bold text-cyan-300">WarpSync</div>
                  <div className="text-[10px] text-cyan-400/60">900K blocks in 5min</div>
                </div>
                <div className="p-2 bg-purple-500/10 border border-purple-500/20 rounded-lg text-center">
                  <div className="text-lg font-bold text-purple-300">PQ-Crypto</div>
                  <div className="text-[10px] text-purple-400/60">Dilithium5 + Kyber</div>
                </div>
                <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-center">
                  <div className="text-lg font-bold text-emerald-300">DAG-BFT</div>
                  <div className="text-[10px] text-emerald-400/60">Sub-second finality</div>
                </div>
              </div>

              {/* Download Links */}
              <div className="space-y-3">
                {/* Linux x64 */}
                <a
                  href="https://quillon.xyz/downloads/q-api-server-v10.3.8"
                  download="q-api-server"
                  className="w-full p-4 bg-slate-800/60 hover:bg-slate-700/60 border border-cyan-500/20 hover:border-cyan-500/40 rounded-xl transition-all flex items-center gap-4 group block"
                >
                  <div className="w-10 h-10 rounded-lg bg-cyan-500/20 flex items-center justify-center shrink-0">
                    <TerminalIcon className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div className="text-left flex-1">
                    <div className="font-bold text-cyan-100">Linux x86_64</div>
                    <div className="text-xs text-cyan-300/50">Ubuntu 20.04+ / Debian 11+ / RHEL 8+</div>
                  </div>
                  <Download className="w-5 h-5 text-cyan-400/60 group-hover:text-cyan-400 transition-colors" />
                </a>

                {/* Windows x64 */}
                <a
                  href="https://quillon.xyz/downloads/q-api-server-v10.3.8-windows-x64.exe"
                  download="q-api-server-v10.3.8-windows-x64.exe"
                  className="w-full p-4 bg-slate-800/60 hover:bg-slate-700/60 border border-cyan-500/20 hover:border-cyan-500/40 rounded-xl transition-all flex items-center gap-4 group block"
                >
                  <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center shrink-0">
                    <Monitor className="w-5 h-5 text-blue-400" />
                  </div>
                  <div className="text-left flex-1">
                    <div className="font-bold text-cyan-100">Windows x64</div>
                    <div className="text-xs text-cyan-300/50">Windows 10/11 — Single EXE with TUI (v10.3.8)</div>
                  </div>
                  <Download className="w-5 h-5 text-cyan-400/60 group-hover:text-cyan-400 transition-colors" />
                </a>

                {/* macOS Build from Source */}
                <div className="w-full p-4 bg-slate-800/60 border border-cyan-500/10 rounded-xl flex items-center gap-4">
                  <div className="w-10 h-10 rounded-lg bg-gray-500/20 flex items-center justify-center shrink-0">
                    <Laptop className="w-5 h-5 text-gray-300" />
                  </div>
                  <div className="text-left flex-1">
                    <div className="font-bold text-cyan-100">macOS (Intel & Apple Silicon)</div>
                    <div className="text-xs text-cyan-300/50">Build from source - see Quick Start below</div>
                  </div>
                </div>
              </div>

              {/* System Requirements */}
              <div className="mt-4 p-3 bg-slate-800/40 rounded-xl border border-cyan-500/10">
                <h3 className="text-sm font-bold text-cyan-300 mb-2">System Requirements</h3>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <div className="text-cyan-400/70 font-semibold mb-1">Minimum</div>
                    <div className="text-cyan-100/60">4 CPU cores, 8 GB RAM</div>
                    <div className="text-cyan-100/60">50 GB SSD, 10 Mbps</div>
                  </div>
                  <div>
                    <div className="text-emerald-400/70 font-semibold mb-1">Recommended</div>
                    <div className="text-cyan-100/60">8+ cores, 32 GB RAM</div>
                    <div className="text-cyan-100/60">500 GB NVMe, 100 Mbps</div>
                  </div>
                </div>
              </div>

              {/* Quick Start */}
              <div className="mt-3 p-3 bg-slate-800/40 rounded-xl border border-cyan-500/10">
                <h3 className="text-sm font-bold text-cyan-300 mb-2">Quick Start (Linux)</h3>
                <code className="text-[11px] text-cyan-100/70 block whitespace-pre-wrap break-all font-mono leading-relaxed">
{`wget https://quillon.xyz/downloads/q-api-server-v10.3.8
chmod +x q-api-server-v10.3.8
./q-api-server-v10.3.8 --port 8080`}
                </code>
                <div className="text-[10px] text-emerald-400/70 mt-2">WarpSync auto-discovers peers & syncs 900K+ blocks in minutes</div>
              </div>

              {/* macOS Build */}
              <div className="mt-3 p-3 bg-slate-800/40 rounded-xl border border-cyan-500/10">
                <h3 className="text-sm font-bold text-cyan-300 mb-2">macOS Build from Source</h3>
                <code className="text-[11px] text-cyan-100/70 block whitespace-pre-wrap break-all font-mono leading-relaxed">
{`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://code.quillon.xyz/repo.git && cd q-narwhalknight
cargo build --release --package q-api-server
./target/release/q-api-server --port 8080`}
                </code>
              </div>

              {/* Close Button */}
              <div className="mt-5 flex justify-center">
                <motion.button
                  onClick={() => setShowNodeModal(false)}
                  className="px-8 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-xl text-white font-bold hover:shadow-[0_0_30px_rgba(6,182,212,0.5)] transition-all"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Close
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Miner Download Modal */}
      <AnimatePresence>
        {showMinerModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[100] overflow-y-auto py-8"
            onClick={() => setShowMinerModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-amber-500/30 rounded-2xl p-6 max-w-lg w-full mx-4 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
              style={{ boxShadow: '0 0 60px rgba(212, 175, 55, 0.3)' }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-amber-500/30 to-yellow-500/30 border border-amber-500/50 flex items-center justify-center">
                    <Pickaxe className="w-7 h-7 text-amber-400" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold bg-gradient-to-r from-amber-400 to-yellow-500 bg-clip-text text-transparent">
                      Download Miner
                    </h2>
                    <p className="text-amber-300/60 text-sm">Quantum-Resistant Solo & Pool Mining</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowMinerModal(false)}
                  className="p-2 hover:bg-amber-500/20 rounded-lg transition-colors"
                >
                  <X className="w-6 h-6 text-amber-400" />
                </button>
              </div>

              {/* v10.10.0: Benchmark-mode browser mining quick-launch (no install required) */}
              <motion.a
                href="/webgpu-miner/index.html?benchmark=true"
                target="_blank"
                rel="noopener noreferrer"
                className="w-full p-4 mb-3 bg-gradient-to-br from-purple-500/20 to-cyan-500/20 hover:from-purple-500/30 hover:to-cyan-500/30 border border-purple-500/40 hover:border-purple-500/60 rounded-xl transition-all flex items-center gap-4 group cursor-pointer"
                whileHover={{ scale: 1.02, x: 4 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="w-10 h-10 rounded-lg bg-purple-500/30 flex items-center justify-center shrink-0">
                  <Zap className="w-5 h-5 text-purple-300" />
                </div>
                <div className="text-left flex-1">
                  <div className="font-bold text-purple-100 flex items-center gap-2">
                    Test your hashrate (browser, no install)
                    <span className="text-[10px] px-1.5 py-0.5 bg-purple-500/30 text-purple-200 rounded font-normal">WebGPU</span>
                  </div>
                  <div className="text-xs text-purple-300/70">Runs SHA3 benchmark in your browser tab — see ~MH/s before downloading the real miner</div>
                </div>
                <span className="text-purple-300 text-sm font-mono opacity-60 group-hover:opacity-100 transition-opacity">→</span>
              </motion.a>

              {/* Platform Downloads */}
              <div className="space-y-3">
                {/* Linux x64 */}
                <motion.button
                  onClick={() => handleDownloadMiner('linux')}
                  className="w-full p-4 bg-slate-800/60 hover:bg-slate-700/60 border border-amber-500/20 hover:border-amber-500/40 rounded-xl transition-all flex items-center gap-4 group"
                  whileHover={{ scale: 1.02, x: 4 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center shrink-0">
                    <TerminalIcon className="w-5 h-5 text-amber-400" />
                  </div>
                  <div className="text-left flex-1">
                    <div className="font-bold text-amber-100">Linux x86_64</div>
                    <div className="text-xs text-amber-300/50">Ubuntu, Debian, Fedora, Arch</div>
                  </div>
                  <Download className="w-5 h-5 text-amber-400/60 group-hover:text-amber-400 transition-colors" />
                </motion.button>

                {/* Linux ARM64 */}
                <motion.button
                  onClick={() => handleDownloadMiner('linux-arm64')}
                  className="w-full p-4 bg-slate-800/60 hover:bg-slate-700/60 border border-amber-500/20 hover:border-amber-500/40 rounded-xl transition-all flex items-center gap-4 group"
                  whileHover={{ scale: 1.02, x: 4 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center shrink-0">
                    <TerminalIcon className="w-5 h-5 text-amber-400" />
                  </div>
                  <div className="text-left flex-1">
                    <div className="font-bold text-amber-100">Linux ARM64</div>
                    <div className="text-xs text-amber-300/50">Raspberry Pi, AWS Graviton</div>
                  </div>
                  <Download className="w-5 h-5 text-amber-400/60 group-hover:text-amber-400 transition-colors" />
                </motion.button>

                {/* Windows */}
                <motion.button
                  onClick={() => handleDownloadMiner('windows')}
                  className="w-full p-4 bg-slate-800/60 hover:bg-slate-700/60 border border-amber-500/20 hover:border-amber-500/40 rounded-xl transition-all flex items-center gap-4 group"
                  whileHover={{ scale: 1.02, x: 4 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center shrink-0">
                    <Monitor className="w-5 h-5 text-blue-400" />
                  </div>
                  <div className="text-left flex-1">
                    <div className="font-bold text-amber-100">Windows x64</div>
                    <div className="text-xs text-amber-300/50">Windows 10/11</div>
                  </div>
                  <Download className="w-5 h-5 text-amber-400/60 group-hover:text-amber-400 transition-colors" />
                </motion.button>

                {/* macOS Intel */}
                <motion.button
                  onClick={() => handleDownloadMiner('macos-intel')}
                  className="w-full p-4 bg-slate-800/60 hover:bg-slate-700/60 border border-amber-500/20 hover:border-amber-500/40 rounded-xl transition-all flex items-center gap-4 group"
                  whileHover={{ scale: 1.02, x: 4 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="w-10 h-10 rounded-lg bg-gray-500/20 flex items-center justify-center shrink-0">
                    <Laptop className="w-5 h-5 text-gray-300" />
                  </div>
                  <div className="text-left flex-1">
                    <div className="font-bold text-amber-100">macOS Intel</div>
                    <div className="text-xs text-amber-300/50">Intel-based Macs</div>
                  </div>
                  <Download className="w-5 h-5 text-amber-400/60 group-hover:text-amber-400 transition-colors" />
                </motion.button>

                {/* macOS ARM (Apple Silicon) */}
                <motion.button
                  onClick={() => handleDownloadMiner('macos-arm')}
                  className="w-full p-4 bg-slate-800/60 hover:bg-slate-700/60 border border-amber-500/20 hover:border-amber-500/40 rounded-xl transition-all flex items-center gap-4 group"
                  whileHover={{ scale: 1.02, x: 4 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="w-10 h-10 rounded-lg bg-gray-500/20 flex items-center justify-center shrink-0">
                    <Laptop className="w-5 h-5 text-gray-300" />
                  </div>
                  <div className="text-left flex-1">
                    <div className="font-bold text-amber-100">macOS Apple Silicon</div>
                    <div className="text-xs text-amber-300/50">M1, M2, M3, M4 chips</div>
                  </div>
                  <Download className="w-5 h-5 text-amber-400/60 group-hover:text-amber-400 transition-colors" />
                </motion.button>
              </div>

              {/* Quick Start */}
              <div className="mt-5 p-4 bg-slate-800/40 rounded-xl border border-amber-500/10">
                <h3 className="text-sm font-bold text-amber-300 mb-2">Quick Start (Linux)</h3>
                <code className="text-xs text-amber-100/70 block whitespace-pre-wrap break-all font-mono">
{`wget https://quillon.xyz/downloads/q-miner-v10.5.3
chmod +x q-miner-v10.5.3
./q-miner-v10.5.3 --mode solo --wallet YOUR_WALLET --threads 4 --server https://quillon.xyz`}
                </code>
              </div>

              {/* Close Button */}
              <div className="mt-5 flex justify-center">
                <motion.button
                  onClick={() => setShowMinerModal(false)}
                  className="px-8 py-3 bg-gradient-to-r from-amber-600 to-yellow-600 rounded-xl text-slate-900 font-bold hover:shadow-[0_0_30px_rgba(251,191,36,0.5)] transition-all"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Close
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Slint Native Wallet Modal */}
      <AnimatePresence>
        {showSlintModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[100] overflow-y-auto py-8"
            onClick={() => setShowSlintModal(false)}
          >
            <motion.div
              initial={{ scale: 0.85, opacity: 0, y: 30, rotateX: 8 }}
              animate={{ scale: 1, opacity: 1, y: 0, rotateX: 0 }}
              exit={{ scale: 0.85, opacity: 0, y: 30 }}
              transition={{ type: "spring", stiffness: 300, damping: 25 }}
              className="relative bg-gradient-to-br from-slate-900 via-emerald-950/50 to-slate-900 border-2 border-emerald-500/40 rounded-2xl p-6 max-w-lg w-full mx-4 shadow-2xl overflow-hidden"
              onClick={(e) => e.stopPropagation()}
              style={{ boxShadow: '0 0 80px rgba(16, 185, 129, 0.25), 0 0 160px rgba(16, 185, 129, 0.08)' }}
            >
              {/* Animated background orbs */}
              <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/[0.08] rounded-full blur-3xl -translate-y-1/3 translate-x-1/3 pointer-events-none" />
              <div className="absolute bottom-0 left-0 w-48 h-48 bg-teal-500/[0.08] rounded-full blur-3xl translate-y-1/3 -translate-x-1/3 pointer-events-none" />
              <motion.div
                className="absolute top-1/2 left-1/2 w-32 h-32 rounded-full pointer-events-none"
                style={{ background: 'radial-gradient(circle, rgba(52,211,153,0.12) 0%, transparent 70%)' }}
                animate={{ x: ['-50%', '-30%', '-60%', '-50%'], y: ['-50%', '-70%', '-30%', '-50%'], scale: [1, 1.3, 0.8, 1] }}
                transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
              />

              {/* Modal Header */}
              <div className="relative flex items-center justify-between mb-5">
                <div className="flex items-center gap-4">
                  <motion.div
                    className="w-14 h-14 rounded-xl bg-gradient-to-br from-emerald-500/40 to-teal-500/40 border border-emerald-400/60 flex items-center justify-center relative"
                    whileHover={{ rotate: [0, -5, 5, 0], scale: 1.05 }}
                    transition={{ duration: 0.5 }}
                  >
                    <Wallet className="w-8 h-8 text-emerald-300" />
                    <motion.div
                      className="absolute inset-0 rounded-xl border-2 border-emerald-400/30"
                      animate={{ opacity: [0.3, 0.8, 0.3] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                  </motion.div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h2 className="text-2xl font-bold bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent">
                        Slint Wallet
                      </h2>
                      <motion.span
                        className="px-2 py-0.5 bg-emerald-500/30 border border-emerald-400/40 text-emerald-300 text-[10px] font-bold rounded-full uppercase"
                        animate={{ opacity: [0.7, 1, 0.7] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      >
                        NEW
                      </motion.span>
                    </div>
                    <p className="text-emerald-300/60 text-sm">Native desktop wallet — pure Rust, no browser</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowSlintModal(false)}
                  className="p-2 hover:bg-emerald-500/20 rounded-lg transition-colors"
                >
                  <X className="w-6 h-6 text-emerald-400" />
                </button>
              </div>

              {/* Feature Cards */}
              <div className="relative grid grid-cols-3 gap-2 mb-5">
                <motion.div
                  className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-center"
                  whileHover={{ scale: 1.05, borderColor: 'rgba(52,211,153,0.5)' }}
                >
                  <Zap className="w-5 h-5 text-emerald-400 mx-auto mb-1" />
                  <div className="text-sm font-bold text-emerald-300">Instant</div>
                  <div className="text-[10px] text-emerald-400/60">Sub-second startup</div>
                </motion.div>
                <motion.div
                  className="p-3 bg-teal-500/10 border border-teal-500/20 rounded-xl text-center"
                  whileHover={{ scale: 1.05, borderColor: 'rgba(20,184,166,0.5)' }}
                >
                  <Shield className="w-5 h-5 text-teal-400 mx-auto mb-1" />
                  <div className="text-sm font-bold text-teal-300">PQ-Safe</div>
                  <div className="text-[10px] text-teal-400/60">Dilithium5 + Kyber</div>
                </motion.div>
                <motion.div
                  className="p-3 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-center"
                  whileHover={{ scale: 1.05, borderColor: 'rgba(6,182,212,0.5)' }}
                >
                  <Pickaxe className="w-5 h-5 text-cyan-400 mx-auto mb-1" />
                  <div className="text-sm font-bold text-cyan-300">Mine</div>
                  <div className="text-[10px] text-cyan-400/60">Built-in miner</div>
                </motion.div>
              </div>

              {/* Feature List */}
              <div className="relative space-y-2 mb-5">
                <motion.div
                  className="flex items-center gap-3 p-2 rounded-lg bg-slate-800/40 border border-slate-700/40"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <div className="w-9 h-9 rounded-lg bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center shrink-0">
                    <span className="text-emerald-400 text-[10px] font-bold">15M</span>
                  </div>
                  <span className="text-sm text-slate-200">Tiny binary — runs on anything</span>
                </motion.div>
                <motion.div
                  className="flex items-center gap-3 p-2 rounded-lg bg-slate-800/40 border border-slate-700/40"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <div className="w-9 h-9 rounded-lg bg-teal-500/20 border border-teal-500/30 flex items-center justify-center shrink-0">
                    <span className="text-teal-400 text-[10px] font-bold">QR</span>
                  </div>
                  <span className="text-sm text-slate-200">Point of Sale, QR payment requests</span>
                </motion.div>
                <motion.div
                  className="flex items-center gap-3 p-2 rounded-lg bg-slate-800/40 border border-slate-700/40"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  <div className="w-9 h-9 rounded-lg bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center shrink-0">
                    <span className="text-cyan-400 text-[10px] font-bold">P2P</span>
                  </div>
                  <span className="text-sm text-slate-200">Connects to quillon.xyz or your own node</span>
                </motion.div>
              </div>

              {/* Download Buttons */}
              <div className="relative space-y-3">
                <motion.a
                  href="https://quillon.xyz/downloads/slint-wallet-linux-x86_64"
                  download="slint-wallet-linux-x86_64"
                  className="w-full p-4 bg-gradient-to-r from-emerald-600/80 to-teal-600/80 hover:from-emerald-500 hover:to-teal-500 border border-emerald-400/30 rounded-xl transition-all flex items-center gap-4 group block"
                  whileHover={{ scale: 1.02, x: 4 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="w-10 h-10 rounded-lg bg-emerald-400/20 flex items-center justify-center shrink-0">
                    <TerminalIcon className="w-5 h-5 text-emerald-300" />
                  </div>
                  <div className="text-left flex-1">
                    <div className="font-bold text-white">Linux x86_64</div>
                    <div className="text-xs text-emerald-200/60">Ubuntu 20.04+ / Debian 11+ / RHEL 8+</div>
                  </div>
                  <Download className="w-5 h-5 text-emerald-300/60 group-hover:text-white transition-colors" />
                </motion.a>

                <motion.a
                  href="https://quillon.xyz/downloads/slint-wallet-windows-x64.exe"
                  download="slint-wallet-windows-x64.exe"
                  className="w-full p-4 bg-gradient-to-r from-teal-600/80 to-cyan-600/80 hover:from-teal-500 hover:to-cyan-500 border border-teal-400/30 rounded-xl transition-all flex items-center gap-4 group block"
                  whileHover={{ scale: 1.02, x: 4 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center shrink-0">
                    <Monitor className="w-5 h-5 text-blue-400" />
                  </div>
                  <div className="text-left flex-1">
                    <div className="font-bold text-white">Windows x64</div>
                    <div className="text-xs text-teal-200/60">Windows 10/11 — no install, just run</div>
                  </div>
                  <Download className="w-5 h-5 text-teal-300/60 group-hover:text-white transition-colors" />
                </motion.a>
              </div>

              {/* Quick Start */}
              <div className="relative mt-4 p-3 bg-slate-800/50 rounded-xl border border-emerald-500/15">
                <h3 className="text-sm font-bold text-emerald-300 mb-2">Quick Start (Linux)</h3>
                <code className="text-[11px] text-emerald-100/70 block whitespace-pre-wrap break-all font-mono leading-relaxed">
{`wget https://quillon.xyz/downloads/slint-wallet-linux-x86_64
chmod +x slint-wallet-linux-x86_64
./slint-wallet-linux-x86_64`}
                </code>
              </div>

              {/* Close Button */}
              <div className="relative mt-5 flex justify-center">
                <motion.button
                  onClick={() => setShowSlintModal(false)}
                  className="px-8 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 rounded-xl text-white font-bold hover:shadow-[0_0_30px_rgba(16,185,129,0.5)] transition-all"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Close
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Research Library Modal */}
      <PapersLibraryModal
        isOpen={showPapersLibrary}
        onClose={() => setShowPapersLibrary(false)}
      />

    </div>
  );
}
